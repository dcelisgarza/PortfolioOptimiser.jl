function _noc_risks(rm::AbstractVector, port, returns, sigma, w1, w2, w3)
    rm = reduce(vcat, rm)
    risk1 = 0.0
    risk2 = 0.0
    risk3 = 0.0
    for r ∈ rm
        scale = r.settings.scale
        solver_flag, sigma_flag, skew_flag, sskew_flag = set_rm_properties!(r, port.solvers,
                                                                            sigma, port.V,
                                                                            port.SV)
        risk1 += calc_risk(r, w1; X = returns) * scale
        risk2 += calc_risk(r, w2; X = returns) * scale
        risk3 += calc_risk(r, w3; X = returns) * scale
        unset_set_rm_properties!(r, solver_flag, sigma_flag, skew_flag, sskew_flag)
    end
    return risk1, risk2, risk3
end
function _noc_risks(rm, port, returns, sigma, w1, w2, w3)
    rm = reduce(vcat, rm)
    scale = rm.settings.scale
    solver_flag, sigma_flag, skew_flag, sskew_flag = set_rm_properties!(rm, port.solvers,
                                                                        sigma, port.V,
                                                                        port.SV)
    risk1 = calc_risk(rm, w1; X = returns) * scale
    risk2 = calc_risk(rm, w2; X = returns) * scale
    risk3 = calc_risk(rm, w3; X = returns) * scale
    unset_set_rm_properties!(rm, solver_flag, sigma_flag, skew_flag, sskew_flag)
    return risk1, risk2, risk3
end
function noc_risk_ret(type, port, rm, obj, kelly, class, w_ini, c_const_obj_pen)
    mu, sigma, returns = mu_sigma_returns_class(port, class)

    w1 = if isempty(type.w_min)
        _w_min = optimise!(port; rm = rm, type = type.type, obj = MinRisk(), kelly = kelly,
                           class = class, w_ini = type.w_min_ini,
                           c_const_obj_pen = c_const_obj_pen)
        !isempty(_w_min) ? _w_min.weights : Vector{eltype(returns)}(undef, 0)
    else
        type.w_min
    end

    w2 = if isempty(type.w_max)
        _w_max = optimise!(port; rm = rm, type = type.type, obj = MaxRet(), kelly = kelly,
                           class = class, w_ini = type.w_max_ini,
                           c_const_obj_pen = c_const_obj_pen)
        !isempty(_w_max) ? _w_max.weights : Vector{eltype(returns)}(undef, 0)

    else
        type.w_max
    end

    w3 = if isempty(type.w_opt)
        _w_opt = optimise!(port; rm = rm, type = type.type, obj = obj, kelly = kelly,
                           class = class, w_ini = w_ini, c_const_obj_pen = c_const_obj_pen)
        !isempty(_w_opt) ? _w_opt.weights : Vector{eltype(returns)}(undef, 0)
    else
        type.w_opt
    end

    if isa(kelly, NoKelly)
        ret1 = dot(mu, w1)
        ret2 = dot(mu, w2)
        ret3 = dot(mu, w3)
    else
        ret1 = sum(log.(one(eltype(mu)) .+ returns * w1)) / size(returns, 1)
        ret2 = sum(log.(one(eltype(mu)) .+ returns * w2)) / size(returns, 1)
        ret3 = sum(log.(one(eltype(mu)) .+ returns * w3)) / size(returns, 1)
    end

    risk1, risk2, risk3 = _noc_risks(rm, port, returns, sigma, w1, w2, w3)

    d_ret = (ret2 - ret1) / type.bins
    d_risk = (risk2 - risk1) / type.bins

    ret3 -= d_ret
    risk3 += d_risk

    return risk3, ret3
end
function noc_constraints(model, risk0, ret0)
    w = model[:w]
    risk = model[:risk]
    ret = model[:ret]
    N = length(w)
    @variable(model, log_ret)
    @variable(model, log_risk)
    @variable(model, log_w[1:N])
    @variable(model, log_1mw[1:N])
    @constraint(model, [log_risk, 1, risk0 - risk] in MOI.ExponentialCone())
    @constraint(model, [log_ret, 1, ret - ret0] in MOI.ExponentialCone())
    @constraint(model, [i = 1:N], [log_w[i], 1, w[i]] ∈ MOI.ExponentialCone())
    @constraint(model, [i = 1:N], [log_1mw[i], 1, 1 - w[i]] ∈ MOI.ExponentialCone())
    return nothing
end
function _optimise!(type::NOC, port::Portfolio, rm::Union{AbstractVector, <:RiskMeasure},
                    obj::ObjectiveFunction, kelly::RetType, class::PortClass,
                    w_ini::AbstractVector,
                    c_const_obj_pen::Union{<:CustomConstraintObjectivePenalty, Nothing},
                    str_names::Bool)
    risk0, ret0 = noc_risk_ret(type, port, rm, obj, kelly, class, w_ini, c_const_obj_pen)
    mu, sigma, returns = mu_sigma_returns_class(port, class)
    port.model = JuMP.Model()
    set_string_names_on_creation(port.model, str_names)
    initial_w(port, w_ini)
    kelly_approx_idx = Int[]
    risk_constraints(port, nothing, Trad(), rm, mu, sigma, returns, kelly_approx_idx)
    return_constraints(port, nothing, kelly, mu, sigma, returns, kelly_approx_idx)
    weight_constraints(port, nothing)
    noc_constraints(port.model, risk0, ret0)
    set_objective_function(port, nothing, type, nothing)
    return convex_optimisation(port, nothing, type, class)
end
