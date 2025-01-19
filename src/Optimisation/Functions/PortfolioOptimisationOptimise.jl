"""
    optimise!(port::Portfolio, type::OptimType)
"""
function optimise!(port::Portfolio, type::Trad)
    (; rm, obj, kelly, class, w_ini, custom_constr, custom_obj, scalarisation, str_names) = type
    empty!(port.fail)
    port.model = JuMP.Model()
    set_string_names_on_creation(port.model, str_names)
    mu, sigma, returns = mu_sigma_returns_class(port, class)
    set_scale_obj_constrs(port)
    optimal_homogenisation_factor(port, mu, obj)
    initial_w(port, w_ini)
    set_k(port, obj)
    # Weight constraints
    weight_constraints(port)
    MIP_constraints(port)
    SDP_network_cluster_constraints(port, type)
    # Tracking
    tracking_error_constraints(port, returns)
    turnover_constraints(port)
    # Fees
    management_fee(port)
    rebalance_fee(port)
    # Risk
    kelly_approx_idx = Int[]
    risk_constraints(port, type, rm, mu, sigma, returns, kelly_approx_idx)
    scalarise_risk_expression(port, scalarisation)
    # Returns
    expected_return_constraints(port, obj, kelly, mu, sigma, returns, kelly_approx_idx)
    # Objective function penalties
    L1_regularisation(port)
    L2_regularisation(port)
    SDP_network_cluster_penalty(port)
    # Custom constraints
    custom_constraint(port, custom_constr)
    # Objective function and custom penalties
    set_objective_function(port, obj, type, kelly, custom_obj)
    return convex_optimisation(port, obj, type, class)
end
function optimise!(port::Portfolio, type::RB)
    (; rm, kelly, class, w_ini, custom_constr, custom_obj, scalarisation, str_names) = type
    empty!(port.fail)
    port.model = JuMP.Model()
    set_string_names_on_creation(port.model, str_names)
    mu, sigma, returns = mu_sigma_returns_class(port, class)
    set_scale_obj_constrs(port)
    rp_constraints(port, class, w_ini)
    # Weight constraints
    weight_constraints(port, false)
    MIP_constraints(port, false)
    SDP_network_cluster_constraints(port, nothing)
    # Tracking
    tracking_error_constraints(port, returns)
    turnover_constraints(port)
    # Fees
    management_fee(port)
    rebalance_fee(port)
    # Risk
    kelly_approx_idx = Int[]
    risk_constraints(port, type, rm, mu, sigma, returns, kelly_approx_idx)
    scalarise_risk_expression(port, scalarisation)
    # Returns
    expected_return_constraints(port, nothing, kelly, mu, sigma, returns, kelly_approx_idx)
    # Objective function penalties
    L1_regularisation(port)
    L2_regularisation(port)
    SDP_network_cluster_penalty(port)
    # Custom constraints
    custom_constraint(port, custom_constr)
    # Objective function and custom penalties
    set_objective_function(port, type, custom_obj)
    return convex_optimisation(port, nothing, type, class)
end
function optimise!(port::Portfolio, type::RRB)
    (; version, kelly, class, w_ini, custom_constr, custom_obj, str_names) = type
    empty!(port.fail)
    port.model = JuMP.Model()
    set_string_names_on_creation(port.model, str_names)
    mu, sigma, returns = mu_sigma_returns_class(port, class)
    set_scale_obj_constrs(port)
    # Weight constraints
    initial_w(port, w_ini)
    set_k(port, nothing)
    weight_constraints(port, false)
    MIP_constraints(port, false)
    SDP_network_cluster_constraints(port, nothing)
    # Tracking
    tracking_error_constraints(port, returns)
    turnover_constraints(port)
    # Fees
    management_fee(port)
    rebalance_fee(port)
    # Risk
    rrp_constraints(port, version, sigma)
    # Returns
    expected_return_constraints(port, nothing, kelly, mu, sigma, returns, nothing)
    # Objective function penalties
    L1_regularisation(port)
    L2_regularisation(port)
    SDP_network_cluster_penalty(port)
    # Custom constraints
    custom_constraint(port, custom_constr)
    # Objective function and custom penalties
    set_objective_function(port, type, custom_obj)
    return convex_optimisation(port, nothing, type, class)
end
function optimise!(port::Portfolio, type::NOC)
    (; flag, rm, obj, kelly, class, custom_constr, custom_obj, scalarisation, str_names) = type
    empty!(port.fail)
    w0, risk0, ret0 = noc_risk_ret(port, type)
    port.model = JuMP.Model()
    set_string_names_on_creation(port.model, str_names)
    set_scale_obj_constrs(port)
    mu, sigma, returns = mu_sigma_returns_class(port, class)
    optimal_homogenisation_factor(port, mu, obj)
    initial_w(port, w0)
    set_k(port, nothing)
    # Weight constraints
    weight_constraints(port, false)
    if flag
        MIP_constraints(port, false)
        SDP_network_cluster_constraints(port, nothing)
        # Tracking
        tracking_error_constraints(port, returns)
        turnover_constraints(port)
    else
        custom_constr = NoCustomConstraint()
        custom_obj = NoCustomObjective()
    end
    # Fees
    management_fee(port)
    rebalance_fee(port)
    # Risk
    kelly_approx_idx = Int[]
    risk_constraints(port, type, rm, mu, sigma, returns, kelly_approx_idx)
    scalarise_risk_expression(port, scalarisation)
    # Returns
    expected_return_constraints(port, nothing, kelly, mu, sigma, returns, kelly_approx_idx)
    if flag
        # Objective function penalties
        L1_regularisation(port)
        L2_regularisation(port)
        SDP_network_cluster_penalty(port)
    end
    # NOC constraints
    noc_constraints(port, risk0, ret0)
    # Custom constraints
    custom_constraint(port, custom_constr)
    # Objective function and custom penalties
    set_objective_function(port, type, custom_obj)
    return convex_optimisation(port, obj, type, class)
end
function frontier_limits!(port::Portfolio, type::Union{Trad, NOC, NCO} = Trad();
                          w_min_ini::AbstractVector = Vector{Float64}(undef, 0),
                          w_max_ini::AbstractVector = Vector{Float64}(undef, 0))
    old_obj = type.obj
    old_w_ini = type.w_ini

    type.obj = MinRisk()
    type.w_ini = w_min_ini
    w_min = optimise!(port, type)

    type.obj = MaxRet()
    type.w_ini = w_max_ini
    w_max = optimise!(port, type)

    type.obj = old_obj
    type.w_ini = old_w_ini

    limits = hcat(w_min, DataFrame(; x1 = w_max[!, 2]))
    DataFrames.rename!(limits, :weights => :w_min, :x1 => :w_max)

    rmsym = get_rm_symbol(type.rm)
    port.limits[rmsym] = limits

    return port.limits[rmsym]
end
"""
```
efficient_frontier!(port::Portfolio, type::Union{Trad, NOC, NCO} = Trad();
                             w_min_ini::AbstractVector = Vector{Float64}(undef, 0),
                             w_max_ini::AbstractVector = Vector{Float64}(undef, 0),
                             points::Integer = 20, rf::Real = 0.0)
```
"""
function efficient_frontier!(port::Portfolio, type::Union{Trad, NOC, NCO} = Trad();
                             w_min_ini::AbstractVector = Vector{Float64}(undef, 0),
                             w_max_ini::AbstractVector = Vector{Float64}(undef, 0),
                             points::Integer = 20, rf::Real = 0.0)
    optimal1 = deepcopy(port.optimal)
    fail1 = deepcopy(port.fail)

    (; class, kelly, rm) = type

    mu, sigma, returns = mu_sigma_returns_class(port, class)

    fl = frontier_limits!(port, type; w_min_ini = w_min_ini, w_max_ini = w_max_ini)
    w1 = fl.w_min
    w2 = fl.w_max

    if isa(kelly, NoKelly)
        ret1 = dot(mu, w1)
        ret2 = dot(mu, w2)
    else
        ret1 = sum(log.(one(eltype(mu)) .+ returns * w1)) / size(returns, 1)
        ret2 = sum(log.(one(eltype(mu)) .+ returns * w2)) / size(returns, 1)
    end

    rm_i = get_first_rm(rm)
    old_ub = rm_i.settings.ub
    rm_i.settings.ub = Inf

    solver_flag, sigma_flag, skew_flag, sskew_flag = set_rm_properties!(rm_i, port.solvers,
                                                                        sigma, port.V,
                                                                        port.SV)
    risk1, risk2 = risk_bounds(rm_i, w1, w2; X = returns, delta = 0)

    mus = range(ret1; stop = ret2, length = points)
    risks = range(risk1; stop = risk2, length = points)

    frontier = Vector{typeof(risk1)}(undef, 0)
    optim_risk = Vector{typeof(risk1)}(undef, 0)
    w_ini = Vector{typeof(risk1)}(undef, 0)

    old_obj = type.obj
    old_w_ini = type.w_ini

    i = 0
    for (j, (r, m)) ∈ enumerate(zip(risks, mus)) #! Do not change this enumerate to pairs.
        if i == 0
            type.obj = MinRisk()
            w = optimise!(port, type)
        else
            if !isempty(w)
                w_ini = w.weights
            end
            if j != length(risks)
                rm_i.settings.ub = r
            else
                rm_i.settings.ub = Inf
            end
            type.obj = MaxRet()
            w = optimise!(port, type)
            # Fallback in case :Max_Ret with maximum risk bounds fails.
            if isempty(w)
                rm_i.settings.ub = Inf
                port.mu_l = m
                type.obj = MinRisk()
                w = optimise!(port, type)
                port.mu_l = Inf
            end
        end
        if isempty(w)
            continue
        end
        rk = calc_risk(rm_i, w.weights; X = returns)
        append!(frontier, w.weights)
        push!(optim_risk, rk)
        i += 1
    end
    rm_i.settings.ub = Inf
    type.obj = Sharpe(; rf = rf)
    w = optimise!(port, type)
    sharpe = false
    if !isempty(w)
        rk = calc_risk(rm_i, w.weights; X = returns)
        append!(frontier, w.weights)
        push!(optim_risk, rk)
        i += 1
        sharpe = true
    end
    rmsym = get_rm_symbol(rm)
    port.frontier[rmsym] = Dict(:weights => hcat(DataFrame(; tickers = port.assets),
                                                 DataFrame(reshape(frontier, length(w1), :),
                                                           string.(range(1, i)))),
                                :risks => optim_risk, :sharpe => sharpe)
    port.optimal = optimal1
    port.fail = fail1
    type.obj = old_obj
    type.w_ini = old_w_ini
    rm_i.settings.ub = old_ub
    unset_set_rm_properties!(rm_i, solver_flag, sigma_flag, skew_flag, sskew_flag)
    return port.frontier[rmsym]
end
