# sharpe ratio k
function _sr_k(::SR, model)
    @variable(model, k >= 0)
    return nothing
end
function _sr_k(::Any, ::Any)
    return nothing
end
# Risk upper bounds
function _set_rm_risk_upper_bound(::SR, ::Trad2, model, rm_risk, ub)
    if isfinite(ub)
        @constraint(model, rm_risk <= ub * model[:k])
    end
    return nothing
end
function _set_rm_risk_upper_bound(::ObjectiveFunction, ::Trad2, model, rm_risk, ub)
    if isfinite(ub)
        @constraint(model, rm_risk <= ub)
    end
    return nothing
end
# SD risk upper bound (special case)
function _set_rm_risk_upper_bound(::SDP2, ::SR, ::Trad2, model, rm_risk, ub)
    if isfinite(ub)
        @constraint(model, rm_risk <= ub^2 * model[:k])
    end
    return nothing
end
function _set_rm_risk_upper_bound(::SDP2, ::ObjectiveFunction, ::Trad2, model, rm_risk, ub)
    if isfinite(ub)
        @constraint(model, rm_risk <= ub^2)
    end
    return nothing
end
function _set_rm_risk_upper_bound(::Any, ::SR, ::Trad2, model, rm_risk, ub)
    if isfinite(ub)
        @constraint(model, rm_risk <= ub * model[:k])
    end
    return nothing
end
function _set_rm_risk_upper_bound(::Any, ::ObjectiveFunction, ::Trad2, model, rm_risk, ub)
    if isfinite(ub)
        @constraint(model, rm_risk <= ub)
    end
    return nothing
end
# Risk expression
function _set_risk_expression(model, rm_risk, scale, flag::Bool)
    if flag
        if !haskey(model, :risk)
            @expression(model, risk, scale * rm_risk)
        else
            @expression(model, tmp, model[:risk] + scale * rm_risk)
            unregister(model, :risk)
            @expression(model, risk, tmp)
            unregister(model, :tmp)
        end
    end
    return nothing
end

function _sdp_m2(::Union{Trad2, WC2}, ::SR, model)
    @expression(model, M2, vcat(model[:w], model[:k]))
    return nothing
end
function _sdp_m2(::Any, ::Any, model)
    @expression(model, M2, vcat(model[:w], 1))
    return nothing
end
function _sdp(port, type::Union{Trad2, RP2, WC2}, obj)
    model = port.model
    if !haskey(model, :W)
        N = size(port.returns, 2)
        @variable(model, W[1:N, 1:N], Symmetric)
        @expression(model, M1, vcat(W, transpose(model[:w])))
        _sdp_m2(type, obj, model)
        @expression(model, M3, hcat(M1, model[:M2]))
        @constraint(model, M3 ∈ PSDCone())
    end
    return nothing
end

function _num_assets_weight_constraints(::SR, port)
    if port.num_assets_u > 0
        N = size(port.returns, 2)
        model = port.model

        @variable(model, tnau_bin[1:N], binary = true)
        @constraint(model, sum(tnau_bin) <= port.num_assets_u)
        # Sharpe ratio
        @variable(model, tnau_bin_sharpe[1:N] .>= 0)
        @constraint(model, tnau_bin_sharpe .<= model[:k])
        @constraint(model, tnau_bin_sharpe .<= port.num_assets_u_scale * model[:tnau_bin])
        @constraint(model,
                    tnau_bin_sharpe .>=
                    model[:k] .- port.num_assets_u_scale * (1 .- model[:tnau_bin]))
        # Long and short
        @constraint(model, model[:w] .<= port.long_u * tnau_bin_sharpe)
        if port.short
            @constraint(model, model[:w] .>= -port.short_u * tnau_bin_sharpe)
        end
    end
    return nothing
end
function _num_assets_weight_constraints(::Any, port)
    if port.num_assets_u > 0
        N = size(port.returns, 2)
        model = port.model

        @variable(model, tnau_bin[1:N], binary = true)
        @constraint(model, sum(tnau_bin) <= port.num_assets_u)
        # Long and short
        @constraint(model, model[:w] .<= port.long_u * tnau_bin)
        if port.short
            @constraint(model, model[:w] .>= -port.short_u * tnau_bin)
        end
    end
    return nothing
end
function _weight_constraints(::SR, port)
    N = size(port.returns, 2)
    model = port.model
    @constraint(model, sum(model[:w]) == port.sum_short_long * model[:k])
    if !port.short
        @constraint(model, model[:w] .<= port.long_u * model[:k])
        @constraint(model, model[:w] .>= 0)
    else
        @variable(model, tw_ulong[1:N] .>= 0)
        @variable(model, tw_ushort[1:N] .>= 0)

        @constraint(model, sum(tw_ulong) <= long_u * model[:k])
        @constraint(model, sum(tw_ushort) <= short_u * model[:k])

        @constraint(model, model[:w] .<= tw_ulong)
        @constraint(model, model[:w] .>= -tw_ushort)
    end
    return nothing
end
function _weight_constraints(::Any, port)
    N = size(port.returns, 2)
    model = port.model
    @constraint(model, sum(model[:w]) == port.sum_short_long)
    if !port.short
        @constraint(model, model[:w] .<= port.long_u)
        @constraint(model, model[:w] .>= 0)
    else
        @variable(model, tw_ulong[1:N] .>= 0)
        @variable(model, tw_ushort[1:N] .>= 0)

        @constraint(model, sum(tw_ulong) <= long_u)
        @constraint(model, sum(tw_ushort) <= short_u)

        @constraint(model, model[:w] .<= tw_ulong)
        @constraint(model, model[:w] .>= -tw_ushort)
    end
    return nothing
end
function weight_constraints(port, type::Union{Trad2, WC2}, obj)
    _num_assets_weight_constraints(obj, port)
    _weight_constraints(obj, port)
    return nothing
end
function _ntwk_constraints(::NoNtwk, args...)
    return nothing
end
end
function _ntwk_constraints(::IP2, port, type::Union{Trad2, WC2}, ::SR)
    N = size(port.returns, 2)
    model = port.model

    @variable(model, tip_bin2[1:N], binary = true)
    @constraint(model, unique(port.network_ip + I; dims = 1) * tip_bin2 .<= 1)
    # Sharpe ratio
    @variable(model, tip_bin_sharpe2[1:N] .>= 0)
    @constraint(model, tip_bin_sharpe2 .<= model[:k])
    @constraint(model, tip_bin_sharpe2 .<= port.network_ip_scale * tip_bin2)
    @constraint(model,
                tip_bin_sharpe2 .>= model[:k] .- port.network_ip_scale * (1 .- tip_bin2))
    # Long and short
    @constraint(model, model[:w] .<= port.long_u * tip_bin_sharpe2)
    if port.short
        @constraint(model, model[:w] .>= -port.short_u * tip_bin_sharpe2)
    end
    return nothing
end
function _ntwk_constraints(::IP2, port, type::Union{Trad2, WC2}, ::Any)
    N = size(port.returns, 2)
    model = port.model

    @variable(model, tip_bin2[1:N], binary = true)
    @constraint(model, unique(port.network_ip + I; dims = 1) * tip_bin2 .<= 1)
    # Long and short
    @constraint(model, model[:w] .<= port.long_u * model[:tip_bin2])
    if port.short
        @constraint(model, model[:w] .>= -port.short_u * model[:tip_bin2])
    end
    return nothing
end
function _ntwk_penalty(port, ::Trad2)
    if !haskey(port.model, :sd_risk)
        @expression(port.model, ntwk_penalty, port.network_penalty * tr(port.model[:W]))
    end
    return nothing
end
function _ntwk_penalty(::Any, ::Any)
    return nothing
end
function _ntwk_constraints(::SDP2, port, type::Union{Trad2, WC2}, obj)
    _sdp(port, type, obj)
    @constraint(port.model, port.network_sdp .* port.model[:W] .== 0)
    _ntwk_penalty(port, type)
    return nothing
end
function ntwk_constraints(port, type::Union{Trad2, WC2}, obj)
    _ntwk_constraints(port.network_method, port, type, obj)
    return nothing
end
function _sd_risk(::SDP2, ::Union{Trad2, WC2}, model, sigma, count::Integer, idx::Integer)
    if isone(count)
        @expression(model, sd_risk, tr(sigma * model[:W]))
    else
        if isone(idx)
            @variable(model, sd_risk[1:count])
        end
        @constraint(model, model[:sd_risk][idx] == tr(sigma * model[:W]))
    end
    return nothing
end
function _sd_risk(::Any, ::Any, model, sigma, count::Integer, idx::Integer)
    G = sqrt(sigma)
    if isone(count)
        @variable(model, sd_risk)
        @constraint(model, [sd_risk; G * model[:w]] ∈ SecondOrderCone())
    else
        if isone(idx)
            @variable(model, sd_risk[1:count])
        end
        @constraint(model, [model[:sd_risk][idx]; G * model[:w]] ∈ SecondOrderCone())
    end
    return nothing
end
function set_rm(port::Portfolio2, rm::SD2, type::Union{Trad2, RP2}, obj::ObjectiveFunction,
                count::Integer, idx::Integer)
    model = port.model
    if (isnothing(rm.sigma) || isempty(rm.sigma))
        rm.sigma = port.cov
    end

    _sd_risk(port.network_method, type, model, rm.sigma, count, idx)
    _set_rm_risk_upper_bound(port.network_method, obj, type, model, model[:sd_risk],
                             rm.settings.ub)
    if isone(count)
        _set_risk_expression(model, model[:sd_risk], rm.settings.scale, rm.settings.flag)
    else
        _set_risk_expression(model, model[:sd_risk][idx], rm.settings.scale,
                             rm.settings.flag)
    end

    return nothing
end
function set_rm(port::Portfolio2, rm::CVaR2, type::Union{Trad2, RP2},
                obj::ObjectiveFunction, count::Integer, idx::Integer)
    model = port.model
    if !haskey(model, :X)
        @expression(model, X, port.returns * model[:w])
    end
    T = size(port.returns, 1)
    iat = inv(rm.alpha * T)

    if isone(count)
        @variable(model, var)
        @variable(model, z_var[1:T] .>= 0)
        @expression(model, cvar_risk, var + sum(z_var) * iat)

        @constraint(model, z_var .>= -model[:X] .- var)

        _set_rm_risk_upper_bound(obj, type, model, cvar_risk, rm.settings.ub)
        _set_risk_expression(model, cvar_risk, rm.settings.scale, rm.settings.flag)
    else
        if isone(idx)
            @variable(model, var[1:count])
            @variable(model, z_var[1:T, 1:count] .>= 0)
            @variable(model, cvar_risk[1:count])
        end

        @constraint(model,
                    model[:cvar_risk][idx] ==
                    model[:var][idx] + sum(model[:z_var][1:T, idx]) * iat)
        @constraint(model, model[:z_var][1:T, idx] .>= -model[:X] .- model[:var][idx])

        _set_rm_risk_upper_bound(obj, type, model, model[:cvar_risk][idx], rm.settings.ub)
        _set_risk_expression(model, model[:cvar_risk][idx], rm.settings.scale,
                             rm.settings.flag)
    end

    return nothing
end

function set_rm(port::Portfolio2, rm::CDaR2, type::Union{Trad2, RP2},
                obj::ObjectiveFunction, count::Integer, idx::Integer)
    model = port.model
    if !haskey(model, :X)
        @expression(model, X, port.returns * model[:w])
    end

    T = size(port.returns, 1)
    iat = inv(rm.alpha * T)

    if isone(count)
        @variable(model, cdar[1:(T + 1)])
        @variable(model, dar)
        @variable(model, z_cdar[1:T] .>= 0)
        @expression(model, cdar_risk, dar + sum(z_cdar) * iat)

        @constraint(model, cdar[2:end] .>= cdar[1:(end - 1)] .- model[:X])
        @constraint(model, cdar[2:end] .>= 0)
        @constraint(model, cdar[1] == 0)
        @constraint(model, z_cdar .>= cdar[2:end] .- dar)

        _set_rm_risk_upper_bound(obj, type, model, cdar_risk, rm.settings.ub)
        _set_risk_expression(model, cdar_risk, rm.settings.scale, rm.settings.flag)
    else
        if isone(idx)
            @variable(model, cdar[1:(T + 1), 1:count])
            @variable(model, dar[1:count])
            @variable(model, z_cdar[1:T, 1:count] .>= 0)
            @variable(model, cdar_risk[1:count])
        end

        @constraint(model,
                    model[:cdar_risk][idx] ==
                    model[:dar][idx] + sum(model[:z_cdar][:, idx]) * iat)
        @constraint(model,
                    model[:cdar][2:end, idx] .>=
                    model[:cdar][1:(end - 1), idx] .- model[:X])
        @constraint(model, model[:cdar][2:end, idx] .>= 0)
        @constraint(model, model[:cdar][1, idx] == 0)
        @constraint(model,
                    model[:z_cdar][:, idx] .>= model[:cdar][2:end, idx] .- model[:dar][idx])

        _set_rm_risk_upper_bound(obj, type, model, model[:cdar_risk][idx], rm.settings.ub)
        _set_risk_expression(model, model[:cdar_risk][idx], rm.settings.scale,
                             rm.settings.flag)
    end

    return nothing
end

export set_rm, MinRisk, Util, SR, MaxRet, Trad2

function optimise2!(port::Portfolio2; rm::Union{Vector{TradRiskMeasure}, TradRiskMeasure},
                    str_names::Bool = false, save_params::Bool = false) end

"""
```julia
optimise!(portfolio::Portfolio2; class::Symbol = :Classic, hist::Integer = 1,
          kelly::Symbol = :None, l::Real = 2.0, obj::Symbol = :Sharpe, rf::Real = 0.0,
          rm::Symbol = :SD, rrp_penalty::Real = 1.0, rrp_ver::Symbol = :None,
          save_opt_params::Bool = true, string_names::Bool = false, type::Symbol = :Trad,
          u_cov::Symbol = :Box, u_mu::Symbol = :Box)
```
"""
function optimise!(portfolio::Portfolio2, opt::OptimiseOpt = OptimiseOpt(;);
                   string_names::Bool = false, save_opt_params::Bool = false)
    type = opt.type
    rm = opt.rm
    obj = opt.obj
    kelly = opt.kelly
    class = opt.class
    rrp_ver = opt.rrp_ver
    u_cov = opt.u_cov
    u_mu = opt.u_mu
    sd_cone = opt.sd_cone
    owa_approx = opt.owa_approx
    near_opt = opt.near_opt
    hist = opt.hist
    rf = opt.rf
    l = opt.l
    rrp_penalty = opt.rrp_penalty
    w_ini = opt.w_ini
    w_min = opt.w_min
    w_max = opt.w_max

    @smart_assert(obj ∈ ObjFuncs)

    if near_opt
        w_min = opt.w_min
        if !isempty(w_min)
            @smart_assert(length(w_min) == size(portfolio.returns, 2))
        end
        w_max = opt.w_max
        if !isempty(w_max)
            @smart_assert(length(w_max) == size(portfolio.returns, 2))
        end
    end
    _p_save_opt_params(portfolio, opt, string_names, save_opt_params)

    mu, sigma, returns = _setup_model_class(portfolio, class, hist)
    T, N = size(returns)
    kurtosis = portfolio.kurt
    skurtosis = portfolio.skurt
    network_method = portfolio.network_method

    portfolio.model = JuMP.Model()
    model = portfolio.model
    set_string_names_on_creation(model, string_names)
    @variable(model, w[1:N])

    if !isempty(w_ini)
        @smart_assert(length(w_ini) == size(portfolio.returns, 2))
        set_start_value.(w, w_ini)
    end

    if type == :Trad
        _setup_sharpe_k(model, obj)
        _risk_setup(portfolio, :Trad, rm, kelly, obj, rf, T, N, mu, returns, sigma,
                    kurtosis, skurtosis, network_method, sd_cone, owa_approx)
        _setup_trad_return(portfolio, class, kelly, obj, T, rf, returns, mu)
        _setup_trad_wc_constraints(portfolio, obj, T, N, :Trad, class, kelly, l, returns)
    elseif type == :RP
        _rp_setup(portfolio, N, class)
        _risk_setup(portfolio, :RP, rm, kelly, obj, rf, T, N, mu, returns, sigma, kurtosis,
                    skurtosis, network_method, sd_cone, owa_approx)
        _setup_rp_rrp_return_and_obj(portfolio, kelly, T, returns, mu)
    elseif type == :RRP
        _setup_risk_budget(portfolio)
        _mv_setup(portfolio, sigma, rm, kelly, obj, :RRP, network_method, sd_cone)
        _rrp_setup(portfolio, sigma, N, rrp_ver, rrp_penalty)
        _setup_rp_rrp_return_and_obj(portfolio, kelly, T, returns, mu)
    else
        _setup_sharpe_k(model, obj)
        _wc_setup(portfolio, kelly, obj, T, N, rf, mu, sigma, u_mu, u_cov, network_method,
                  sd_cone)
        _setup_trad_wc_constraints(portfolio, obj, T, N, :WC, class, kelly, l, returns)
    end

    _setup_linear_constraints(portfolio, obj, type)

    term_status, solvers_tried = _optimise_portfolio(portfolio, class, type, obj)
    retval = _handle_errors_and_finalise(portfolio, class, term_status, returns, N,
                                         solvers_tried, type, rm, obj)

    if near_opt && type ∈ (:Trad, :WC)
        retval = _near_optimal_centering(portfolio, class, mu, returns, sigma, retval, T, N,
                                         opt)
    end

    return retval
end

"""
```julia
frontier_limits!(portfolio::Portfolio2; class::Symbol = :Classic, hist::Integer = 1,
                 kelly::Symbol = :None, rf::Real = 0.0, rm::Symbol = :SD,
                 save_model::Bool = false)
```
"""
function frontier_limits!(portfolio::Portfolio2, opt::OptimiseOpt = OptimiseOpt(;);
                          save_model::Bool = false)
    obj1 = opt.obj
    near_opt1 = opt.near_opt
    opt.near_opt = false
    optimal1 = deepcopy(portfolio.optimal)
    fail1 = deepcopy(portfolio.fail)
    if save_model
        model1 = copy(portfolio.model)
    end

    opt.obj = :Min_Risk
    w_min = optimise!(portfolio, opt)

    opt.obj = :Max_Ret
    w_max = optimise!(portfolio, opt)

    limits = hcat(w_min, DataFrame(; x1 = w_max[!, 2]))
    DataFrames.rename!(limits, :weights => :w_min, :x1 => :w_max)
    portfolio.limits[opt.rm] = limits

    opt.obj = obj1
    opt.near_opt = near_opt1
    portfolio.optimal = optimal1
    portfolio.fail = fail1
    if save_model
        portfolio.model = model1
    end

    return portfolio.limits[opt.rm]
end

"""
```julia
efficient_frontier!(portfolio::Portfolio2; class::Symbol = :Classic, hist::Integer = 1,
                    kelly::Symbol = :None, rf::Real = 0.0, rm::Symbol = :SD,
                    points::Integer = 20)
```
"""
function efficient_frontier!(portfolio::Portfolio2, opt::OptimiseOpt = OptimiseOpt(;);
                             points::Integer = 20)
    @smart_assert(opt.type == :Trad)
    obj1 = opt.obj
    w_ini1 = opt.w_ini
    optimal1 = deepcopy(portfolio.optimal)
    fail1 = deepcopy(portfolio.fail)

    class = opt.class
    hist = opt.hist
    mu, sigma, returns = _setup_model_class(portfolio, class, hist)

    fl = frontier_limits!(portfolio, opt)

    w1 = fl.w_min
    w2 = fl.w_max

    if opt.kelly == :None
        ret1 = dot(mu, w1)
        ret2 = dot(mu, w2)
    else
        ret1 = sum(log.(one(eltype(mu)) .+ returns * w1)) / size(returns, 1)
        ret2 = sum(log.(one(eltype(mu)) .+ returns * w2)) / size(returns, 1)
    end

    alpha_i = portfolio.alpha_i
    alpha = portfolio.alpha
    a_sim = portfolio.a_sim
    beta_i = portfolio.beta_i
    beta = portfolio.beta
    b_sim = portfolio.b_sim
    kappa = portfolio.kappa
    owa_w = portfolio.owa_w
    V = portfolio.V
    SV = portfolio.SV
    solvers = portfolio.solvers

    rm = opt.rm
    rf = opt.rf

    risk1, risk2 = _ul_risk(rm, returns, w1, w2, sigma, rf, solvers, alpha, kappa, alpha_i,
                            beta, a_sim, beta_i, b_sim, owa_w, V, SV, 0)

    mus = range(ret1; stop = ret2, length = points)
    risks = range(risk1; stop = risk2, length = points)

    rmf = Symbol(lowercase(string(rm)) * "_u")

    frontier = Vector{typeof(risk1)}(undef, 0)
    srisk = Vector{typeof(risk1)}(undef, 0)
    w_ini = Vector{typeof(risk1)}(undef, 0)

    i = 0
    for (j, (r, m)) ∈ enumerate(zip(risks, mus))
        if i == 0
            opt.obj = :Min_Risk
            w = optimise!(portfolio, opt)
        else
            if !isempty(w)
                opt.w_ini = w.weights
            end
            if j != length(risks)
                setproperty!(portfolio, rmf, r)
            else
                setproperty!(portfolio, rmf, Inf)
            end
            opt.obj = :Max_Ret
            w = optimise!(portfolio, opt)
            # Fallback in case :Max_Ret with maximum risk bounds fails.
            if isempty(w)
                opt.obj = :Min_Risk
                setproperty!(portfolio, rmf, Inf)
                portfolio.mu_l = m
                w = optimise!(portfolio, opt)
                portfolio.mu_l = Inf
            end
        end
        if isempty(w)
            continue
        end
        rk = calc_risk(w.weights, returns; rm = rm, rf = rf, sigma = sigma,
                       alpha_i = alpha_i, alpha = alpha, a_sim = a_sim, beta_i = beta_i,
                       beta = beta, b_sim = b_sim, kappa = kappa, owa_w = owa_w, V = V,
                       SV = SV, solvers = solvers)

        append!(frontier, w.weights)
        push!(srisk, rk)
        i += 1
    end
    setproperty!(portfolio, rmf, Inf)

    opt.obj = :Sharpe
    w = optimise!(portfolio, opt)
    sharpe = false
    if !isempty(w)
        rk = calc_risk(w.weights, returns; rm = rm, rf = rf, sigma = sigma,
                       alpha_i = alpha_i, alpha = alpha, a_sim = a_sim, beta_i = beta_i,
                       beta = beta, b_sim = b_sim, kappa = kappa, owa_w = owa_w, V = V,
                       SV = SV, solvers = solvers)
        append!(frontier, w.weights)
        push!(srisk, rk)
        i += 1
        sharpe = true
    end

    key = if opt.near_opt
        Symbol("Near_" * string(rm))
    else
        rm
    end

    portfolio.frontier[key] = Dict(:weights => hcat(DataFrame(; tickers = portfolio.assets),
                                                    DataFrame(reshape(frontier, length(w1),
                                                                      :),
                                                              string.(range(1, i)))),
                                   :opt => opt, :points => points, :risk => srisk,
                                   :sharpe => sharpe)

    opt.obj = obj1
    opt.w_ini = w_ini1
    portfolio.optimal = optimal1
    portfolio.fail = fail1

    return portfolio.frontier[key]
end
