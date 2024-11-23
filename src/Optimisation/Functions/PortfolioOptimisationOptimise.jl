"""
```
optimise!(port::Portfolio; rm::Union{AbstractVector, <:RiskMeasure} = SD(),
          type::OptimType = Trad(), obj::ObjectiveFunction = MinRisk(),
          kelly::RetType = NoKelly(), class::PortClass = Classic(),
          w_ini::AbstractVector = Vector{Float64}(undef, 0),
          str_names::Bool = false)
```
"""
function optimise!(port::Portfolio; rm::Union{AbstractVector, <:RiskMeasure} = SD(),
                   type::OptimType = Trad(), obj::ObjectiveFunction = MinRisk(),
                   kelly::RetType = NoKelly(), class::PortClass = Classic(),
                   w_ini::AbstractVector = Vector{Float64}(undef, 0),
                   c_const_obj_pen::Union{<:CustomConstraintObjectivePenalty, Nothing} = nothing,
                   str_names::Bool = false)
    empty!(port.fail)
    return _optimise!(type, port, rm, obj, kelly, class, w_ini, c_const_obj_pen, str_names)
end
function optimise!(port::OmniPortfolio;
                   rm::Union{AbstractVector, <:RiskMeasure} = Variance(),
                   type::OptimType = Trad(), obj::ObjectiveFunction = MinRisk(),
                   kelly::RetType = NoKelly(), class::PortClass = Classic(),
                   w_ini::AbstractVector = Vector{Float64}(undef, 0),
                   custom_constraint = nothing, custom_objective = nothing, ohf::Real = 1.0,
                   str_names::Bool = false)
    empty!(port.fail)
    return _optimise!(type, port, rm, obj, kelly, class, w_ini, custom_constraint,
                      custom_objective, ohf, str_names)
end

function frontier_limits!(port::Portfolio; rm::Union{AbstractVector, <:RiskMeasure} = SD(),
                          kelly::RetType = NoKelly(), class::PortClass = Classic(),
                          w_min_ini::AbstractVector = Vector{Float64}(undef, 0),
                          w_max_ini::AbstractVector = Vector{Float64}(undef, 0),
                          c_const_obj_pen::Union{<:CustomConstraintObjectivePenalty,
                                                 Nothing} = nothing)
    w_min = optimise!(port; rm = rm, obj = MinRisk(), kelly = kelly, class = class,
                      w_ini = w_min_ini, c_const_obj_pen = c_const_obj_pen)
    w_max = optimise!(port; rm = rm, obj = MaxRet(), kelly = kelly, class = class,
                      w_ini = w_max_ini, c_const_obj_pen = c_const_obj_pen)

    limits = hcat(w_min, DataFrame(; x1 = w_max[!, 2]))
    DataFrames.rename!(limits, :weights => :w_min, :x1 => :w_max)

    rmsym = get_rm_symbol(rm)
    port.limits[rmsym] = limits

    return port.limits[rmsym]
end
function frontier_limits!(port::OmniPortfolio;
                          rm::Union{AbstractVector, <:RiskMeasure} = Variance(),
                          kelly::RetType = NoKelly(), class::PortClass = Classic(),
                          w_min_ini::AbstractVector = Vector{Float64}(undef, 0),
                          w_max_ini::AbstractVector = Vector{Float64}(undef, 0),
                          custom_constraint = nothing, custom_objective = nothing,
                          ohf::Real = 1.0)
    w_min = optimise!(port; rm = rm, obj = MinRisk(), kelly = kelly, class = class,
                      w_ini = w_min_ini, custom_constraint = custom_constraint,
                      custom_objective = custom_objective, ohf = ohf)
    w_max = optimise!(port; rm = rm, obj = MaxRet(), kelly = kelly, class = class,
                      w_ini = w_max_ini, custom_constraint = custom_constraint,
                      custom_objective = custom_objective, ohf = ohf)

    limits = hcat(w_min, DataFrame(; x1 = w_max[!, 2]))
    DataFrames.rename!(limits, :weights => :w_min, :x1 => :w_max)

    rmsym = get_rm_symbol(rm)
    port.limits[rmsym] = limits

    return port.limits[rmsym]
end

"""
```
efficient_frontier!(port::Portfolio; type::Union{Trad, NOC} = Trad(),
                             rm::Union{AbstractVector, <:RiskMeasure} = SD(),
                             kelly::RetType = NoKelly(), class::PortClass = Classic(),
                             w_min_ini::AbstractVector = Vector{Float64}(undef, 0),
                             w_max_ini::AbstractVector = Vector{Float64}(undef, 0),
                             points::Integer = 20, rf::Real = 0.0)
```
"""
function efficient_frontier!(port::Portfolio; type::Union{Trad, NOC} = Trad(),
                             rm::Union{AbstractVector, <:RiskMeasure} = SD(),
                             kelly::RetType = NoKelly(), class::PortClass = Classic(),
                             w_min_ini::AbstractVector = Vector{Float64}(undef, 0),
                             w_max_ini::AbstractVector = Vector{Float64}(undef, 0),
                             c_const_obj_pen::Union{<:CustomConstraintObjectivePenalty,
                                                    Nothing} = nothing,
                             points::Integer = 20, rf::Real = 0.0)
    optimal1 = deepcopy(port.optimal)
    fail1 = deepcopy(port.fail)
    mu, sigma, returns = mu_sigma_returns_class(port, class)

    fl = frontier_limits!(port; rm = rm, kelly = kelly, class = class,
                          w_min_ini = w_min_ini, w_max_ini = w_max_ini,
                          c_const_obj_pen = c_const_obj_pen)
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
            w = optimise!(port; rm = rm, type = type, obj = MinRisk(), kelly = kelly,
                          class = class, w_ini = w_min_ini,
                          c_const_obj_pen = c_const_obj_pen)
        else
            if !isempty(w)
                w_ini = w.weights
            end
            if j != length(risks)
                rm_i.settings.ub = r
            else
                rm_i.settings.ub = Inf
            end
            w = optimise!(port; rm = rm, type = type, obj = MaxRet(), kelly = kelly,
                          class = class, w_ini = w_ini, c_const_obj_pen = c_const_obj_pen)
            # Fallback in case :Max_Ret with maximum risk bounds fails.
            if isempty(w)
                rm_i.settings.ub = Inf
                port.mu_l = m
                w = optimise!(port; rm = rm, type = type, obj = MinRisk(), kelly = kelly,
                              class = class, w_ini = w_ini,
                              c_const_obj_pen = c_const_obj_pen)
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
    w = optimise!(port; rm = rm, type = type, obj = Sharpe(; rf = rf), kelly = kelly,
                  class = class, w_ini = w_min_ini, c_const_obj_pen = c_const_obj_pen)
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
function efficient_frontier!(port::OmniPortfolio; type::Union{Trad, NOC} = Trad(),
                             rm::Union{AbstractVector, <:RiskMeasure} = Variance(),
                             kelly::RetType = NoKelly(), class::PortClass = Classic(),
                             w_min_ini::AbstractVector = Vector{Float64}(undef, 0),
                             w_max_ini::AbstractVector = Vector{Float64}(undef, 0),
                             custom_constraint = nothing, custom_objective = nothing,
                             ohf::Real = 1.0, points::Integer = 20, rf::Real = 0.0)
    optimal1 = deepcopy(port.optimal)
    fail1 = deepcopy(port.fail)
    mu, sigma, returns = mu_sigma_returns_class(port, class)

    fl = frontier_limits!(port; rm = rm, kelly = kelly, class = class,
                          w_min_ini = w_min_ini, w_max_ini = w_max_ini,
                          custom_constraint = custom_constraint,
                          custom_objective = custom_objective, ohf = ohf)
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
            w = optimise!(port; rm = rm, type = type, obj = MinRisk(), kelly = kelly,
                          class = class, w_ini = w_min_ini,
                          custom_constraint = custom_constraint,
                          custom_objective = custom_objective, ohf = ohf)
        else
            if !isempty(w)
                w_ini = w.weights
            end
            if j != length(risks)
                rm_i.settings.ub = r
            else
                rm_i.settings.ub = Inf
            end
            w = optimise!(port; rm = rm, type = type, obj = MaxRet(), kelly = kelly,
                          class = class, w_ini = w_ini,
                          custom_constraint = custom_constraint,
                          custom_objective = custom_objective, ohf = ohf)
            # Fallback in case :Max_Ret with maximum risk bounds fails.
            if isempty(w)
                rm_i.settings.ub = Inf
                port.mu_l = m
                w = optimise!(port; rm = rm, type = type, obj = MinRisk(), kelly = kelly,
                              class = class, w_ini = w_ini,
                              custom_constraint = custom_constraint,
                              custom_objective = custom_objective, ohf = ohf)
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
    w = optimise!(port; rm = rm, type = type, obj = Sharpe(; rf = rf), kelly = kelly,
                  class = class, w_ini = w_min_ini, custom_constraint = custom_constraint,
                  custom_objective = custom_objective, ohf = ohf)
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
