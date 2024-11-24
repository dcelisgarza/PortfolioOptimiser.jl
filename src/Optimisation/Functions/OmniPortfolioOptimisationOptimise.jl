function optimise!(port::OmniPortfolio, ::Trad;
                   rm::Union{AbstractVector, <:RiskMeasure} = Variance(),
                   obj::ObjectiveFunction = MinRisk(), kelly::RetType = NoKelly(),
                   class::PortClass = Classic(),
                   w_ini::AbstractVector = Vector{Float64}(undef, 0),
                   custom_constr::CustomConstraint = NoCustomConstraint(),
                   custom_obj::CustomObjective = NoCustomObjective(), ohf::Real = 1.0,
                   str_names::Bool = false, kwargs...)
    empty!(port.fail)
    port.model = JuMP.Model()
    set_string_names_on_creation(port.model, str_names)
    mu, sigma, returns = mu_sigma_returns_class(port, class)
    optimal_homogenisation_factor(port, mu, obj, ohf)
    initial_w(port, w_ini)
    set_k(port, obj)
    # Weight constraints
    weight_constraints(port)
    MIP_constraints(port)
    SDP_network_cluster_constraints(port, Trad())
    # Tracking
    tracking_error_constraints(port, returns)
    turnover_constraints(port)
    # Fees
    management_fee(port)
    rebalance_fee(port)
    # Risk
    kelly_approx_idx = Int[]
    risk_constraints(port, Trad(), rm, mu, sigma, returns, kelly_approx_idx)
    # Returns
    expected_return_constraints(port, obj, kelly, mu, sigma, returns, kelly_approx_idx)
    # Objective function penalties
    L1_regularisation(port)
    L2_regularisation(port)
    SDP_network_cluster_penalty(port)
    # Custom constraints
    custom_constraint(port, custom_constr)
    # Objective function and custom penalties
    set_objective_function(port, obj, Trad(), kelly, custom_obj)
    return convex_optimisation(port, obj, Trad(), class)
end
function optimise!(port::OmniPortfolio, type::RP;
                   rm::Union{AbstractVector, <:RiskMeasure} = Variance(),
                   kelly::RetType = NoKelly(), class::PortClass = Classic(),
                   w_ini::AbstractVector = Vector{Float64}(undef, 0),
                   custom_constr::CustomConstraint = NoCustomConstraint(),
                   custom_obj::CustomObjective = NoCustomObjective(),
                   str_names::Bool = false, kwargs...)
    empty!(port.fail)
    old_short = nothing
    if port.short
        old_short = port.short
        port.short = false
    end
    port.model = JuMP.Model()
    set_string_names_on_creation(port.model, str_names)
    mu, sigma, returns = mu_sigma_returns_class(port, class)
    rp_constraints(port, class, w_ini)
    # Weight constraints
    weight_constraints(port)
    MIP_constraints(port)
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
    retval = convex_optimisation(port, nothing, type, class)
    if !isnothing(old_short)
        port.short = old_short
    end
    return retval
end
function optimise!(port::OmniPortfolio, type::RRP; kelly::RetType = NoKelly(),
                   class::PortClass = Classic(),
                   w_ini::AbstractVector = Vector{Float64}(undef, 0),
                   custom_constr::CustomConstraint = NoCustomConstraint(),
                   custom_obj::CustomObjective = NoCustomObjective(),
                   str_names::Bool = false, kwargs...)
    empty!(port.fail)
    old_short = nothing
    if port.short
        old_short = port.short
        port.short = false
    end
    port.model = JuMP.Model()
    set_string_names_on_creation(port.model, str_names)
    mu, sigma, returns = mu_sigma_returns_class(port, class)
    # Weight constraints
    initial_w(port, w_ini)
    set_k(port, nothing)
    weight_constraints(port)
    MIP_constraints(port)
    SDP_network_cluster_constraints(port, nothing)
    # Tracking
    tracking_error_constraints(port, returns)
    turnover_constraints(port)
    # Fees
    management_fee(port)
    rebalance_fee(port)
    # Risk
    rrp_constraints(port, type, sigma)
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
    retval = convex_optimisation(port, nothing, type, class)
    if !isnothing(old_short)
        port.short = old_short
    end
    return retval
end
function optimise!(port::OmniPortfolio, type::NOC;
                   rm::Union{AbstractVector, <:RiskMeasure} = SD(),
                   obj::ObjectiveFunction = MinRisk(), kelly::RetType = NoKelly(),
                   class::PortClass = Classic(),
                   w_ini::AbstractVector = Vector{Float64(undef, 0)},
                   custom_constr::CustomConstraint = NoCustomConstraint(),
                   custom_obj::CustomObjective = NoCustomObjective(), ohf::Real,
                   str_names::Bool = false, kwargs...)
    empty!(port.fail)
    old_short = nothing
    if port.short
        old_short = port.short
        port.short = false
    end
    risk0, ret0 = noc_risk_ret(port, type, rm, obj, kelly, class, w_ini, custom_constr,
                               custom_obj, ohf)
    port.model = JuMP.Model()
    set_string_names_on_creation(port.model, str_names)
    mu, sigma, returns = mu_sigma_returns_class(port, class)
    optimal_homogenisation_factor(port, mu, obj, ohf)
    initial_w(port, w_ini)
    set_k(port, nothing)
    # Weight constraints
    weight_constraints(port)
    flag = type.flag
    if flag
        MIP_constraints(port)
        SDP_network_cluster_constraints(port, nothing)
        # Tracking
        tracking_error_constraints(port, returns)
        turnover_constraints(port)
    else
        old_ntwk_adj = port.network_adj
        old_clst_adj = port.cluster_adj
        port.network_adj = NoAdj()
        port.cluster_adj = NoAdj()
        custom_constr = nothing
        custom_obj = nothing
    end
    # Fees
    management_fee(port)
    rebalance_fee(port)
    # Risk
    kelly_approx_idx = Int[]
    risk_constraints(port, type.type, rm, mu, sigma, returns, kelly_approx_idx)
    # Returns
    expected_return_constraints(port, nothing, kelly, mu, sigma, returns, kelly_approx_idx)
    if flag
        # Objective function penalties
        L1_regularisation(port)
        L2_regularisation(port)
        SDP_network_cluster_penalty(port)
    else
        port.network_adj = old_ntwk_adj
        port.cluster_adj = old_clst_adj
    end
    # NOC constraints
    noc_constraints(port, risk0, ret0)
    # Custom constraints
    custom_constraint(port, custom_constr)
    # Objective function and custom penalties
    set_objective_function(port, type, custom_obj)
    retval = convex_optimisation(port, obj, type, class)
    if !isnothing(old_short)
        port.short = old_short
    end
    return retval
end
function frontier_limits!(port::OmniPortfolio, type::Union{Trad, NOC} = Trad();
                          rm::Union{AbstractVector, <:RiskMeasure} = Variance(),
                          kelly::RetType = NoKelly(), class::PortClass = Classic(),
                          w_min_ini::AbstractVector = Vector{Float64}(undef, 0),
                          w_max_ini::AbstractVector = Vector{Float64}(undef, 0),
                          custom_constr::CustomConstraint = NoCustomConstraint(),
                          custom_obj::CustomObjective = NoCustomObjective(),
                          ohf::Real = 1.0)
    w_min = optimise!(port, type; rm = rm, obj = MinRisk(), kelly = kelly, class = class,
                      w_ini = w_min_ini, custom_constr = custom_constr,
                      custom_obj = custom_obj, ohf = ohf)
    w_max = optimise!(port, type; rm = rm, obj = MaxRet(), kelly = kelly, class = class,
                      w_ini = w_max_ini, custom_constr = custom_constr,
                      custom_obj = custom_obj, ohf = ohf)

    limits = hcat(w_min, DataFrame(; x1 = w_max[!, 2]))
    DataFrames.rename!(limits, :weights => :w_min, :x1 => :w_max)

    rmsym = get_rm_symbol(rm)
    port.limits[rmsym] = limits

    return port.limits[rmsym]
end
function efficient_frontier!(port::OmniPortfolio, type::Union{Trad, NOC} = Trad();
                             rm::Union{AbstractVector, <:RiskMeasure} = Variance(),
                             kelly::RetType = NoKelly(), class::PortClass = Classic(),
                             w_min_ini::AbstractVector = Vector{Float64}(undef, 0),
                             w_max_ini::AbstractVector = Vector{Float64}(undef, 0),
                             custom_constr::CustomConstraint = NoCustomConstraint(),
                             custom_obj::CustomObjective = NoCustomObjective(),
                             ohf::Real = 1.0, points::Integer = 20, rf::Real = 0.0)
    optimal1 = deepcopy(port.optimal)
    fail1 = deepcopy(port.fail)
    mu, sigma, returns = mu_sigma_returns_class(port, class)

    fl = frontier_limits!(port, type; rm = rm, kelly = kelly, class = class,
                          w_min_ini = w_min_ini, w_max_ini = w_max_ini,
                          custom_constr = custom_constr, custom_obj = custom_obj, ohf = ohf)
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

    i = 0
    for (j, (r, m)) ∈ enumerate(zip(risks, mus)) #! Do not change this enumerate to pairs.
        if i == 0
            w = optimise!(port, type; rm = rm, obj = MinRisk(), kelly = kelly,
                          class = class, w_ini = w_min_ini, custom_constr = custom_constr,
                          custom_obj = custom_obj, ohf = ohf)
        else
            if !isempty(w)
                w_ini = w.weights
            end
            if j != length(risks)
                rm_i.settings.ub = r
            else
                rm_i.settings.ub = Inf
            end
            w = optimise!(port, type; rm = rm, obj = MaxRet(), kelly = kelly, class = class,
                          w_ini = w_ini, custom_constr = custom_constr,
                          custom_obj = custom_obj, ohf = ohf)
            # Fallback in case :Max_Ret with maximum risk bounds fails.
            if isempty(w)
                rm_i.settings.ub = Inf
                port.mu_l = m
                w = optimise!(port, type; rm = rm, obj = MinRisk(), kelly = kelly,
                              class = class, w_ini = w_ini, custom_constr = custom_constr,
                              custom_obj = custom_obj, ohf = ohf)
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
    w = optimise!(port, type; rm = rm, obj = Sharpe(; rf = rf), kelly = kelly,
                  class = class, w_ini = w_min_ini, custom_constr = custom_constr,
                  custom_obj = custom_obj, ohf = ohf)
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
    unset_set_rm_properties!(rm_i, solver_flag, sigma_flag, skew_flag, sskew_flag)
    return port.frontier[rmsym]
end
