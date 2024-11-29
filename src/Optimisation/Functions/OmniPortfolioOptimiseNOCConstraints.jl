function noc_constraints(port::OmniPortfolio, risk0, ret0)
    model = port.model
    w = model[:w]
    risk = model[:risk]
    ret = model[:ret]
    N = length(w)
    @variables(model, begin
                   log_ret
                   log_risk
                   log_w[1:N]
                   log_1mw[1:N]
               end)
    @constraints(model, begin
                     [log_risk, 1, risk0 - risk] in MOI.ExponentialCone()
                     [log_ret, 1, ret - ret0] in MOI.ExponentialCone()
                     [i = 1:N], [log_w[i], 1, w[i]] ∈ MOI.ExponentialCone()
                     [i = 1:N], [log_1mw[i], 1, 1 - w[i]] ∈ MOI.ExponentialCone()
                 end)
    return nothing
end
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
function noc_risk_ret(port::OmniPortfolio, type)
    (; bins, w_min, w_max, w_min_ini, w_max_ini, w_opt, rm, obj, kelly, class, w_ini) = type

    mu, sigma, returns = mu_sigma_returns_class(port, class)
    w1 = if isempty(w_min)
        _w_min = optimise!(port,
                           Trad(; rm = rm, obj = MinRisk(), kelly = kelly, class = class,
                                w_ini = w_min_ini))
        !isempty(_w_min) ? _w_min.weights : Vector{eltype(returns)}(undef, 0)
    else
        w_min
    end

    w2 = if isempty(w_max)
        _w_max = optimise!(port,
                           Trad(; rm = rm, obj = MaxRet(), kelly = kelly, class = class,
                                w_ini = w_max_ini))
        !isempty(_w_max) ? _w_max.weights : Vector{eltype(returns)}(undef, 0)
    else
        w_max
    end

    w3 = if isempty(w_opt)
        _w_opt = optimise!(port,
                           Trad(; rm = rm, obj = obj, kelly = kelly, class = class,
                                w_ini = w_ini))
        !isempty(_w_opt) ? _w_opt.weights : Vector{eltype(returns)}(undef, 0)
    else
        w_opt
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

    d_ret = (ret2 - ret1) / bins
    d_risk = (risk2 - risk1) / bins

    ret3 -= d_ret
    risk3 += d_risk

    return risk3, ret3
end
