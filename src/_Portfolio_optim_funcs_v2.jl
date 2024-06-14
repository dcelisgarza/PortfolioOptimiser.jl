abstract type ObjectiveFunction end
struct MinRisk <: ObjectiveFunction end
struct Util <: ObjectiveFunction end
struct SR <: ObjectiveFunction end
struct MaxRet <: ObjectiveFunction end

function set_upper_bound(::SR, model, rm_risk, ub, type)
    if isfinite(ub) && type == :Trad
        @constraint(model, rm_risk <= ub * model[:k])
    end
end
function set_upper_bound(::ObjectiveFunction, model, rm_risk, ub, type)
    if isfinite(ub) && type == :Trad
        @constraint(model, rm_risk <= ub)
    end
end
function setup_risk(model, rm_risk, scale, flag::Bool)
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
end

function setup_rm(port::Portfolio2, rm::CVaR2, flag::Bool, ub::Real, scale::Real,
                  type::Symbol, obj::ObjectiveFunction, count::Integer, idx::Integer)
    model = port.model
    if !haskey(model, :X)
        @expression(model, X, port.returns * model[:w])
    end
    T = size(port.returns, 1)
    iat = inv(rm.alpha * T)

    if count == 1
        @variable(model, var)
        @variable(model, z_var[1:T] .>= 0)
        @constraint(model, z_var .>= -model[:X] .- var)
        @expression(model, cvar_risk, var + sum(z_var) * iat)

        set_upper_bound(obj, model, cvar_risk, ub, type)
        setup_risk(model, cvar_risk, scale, flag)
    else
        if idx == 1
            @variable(model, var[1:count])
            @variable(model, z_var[1:T, 1:count] .>= 0)
            @variable(model, cvar_risk[1:count])
        end

        @constraint(model, model[:z_var][1:T, idx] .>= -model[:X] .- model[:var][idx])
        @constraint(model,
                    model[:cvar_risk][idx] ==
                    model[:var][idx] + sum(model[:z_var][1:T, idx]) * iat)

        set_upper_bound(obj, model, model[:cvar_risk][idx], ub, type)
        setup_risk(model, model[:cvar_risk][idx], scale, flag)
    end

    return nothing
end

export setup_rm, MinRisk, Util, SR, MaxRet

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
