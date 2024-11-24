function _rebuild_B(B::DataFrame, ::Any, ::Any)
    return Matrix(B[!, setdiff(names(B), ("tickers", "const"))])
end
function _rebuild_B(B::DataFrame, factors::AbstractMatrix, regression::PCAReg)
    B = Matrix(B[!, setdiff(names(B), ("tickers", "const"))])
    X = transpose(factors)
    X_std = StatsBase.standardize(StatsBase.ZScoreTransform, X; dims = 2)
    model = fit(regression.target, X_std)
    Vp = projection(model)
    sdev = if isnothing(regression.std_w)
        vec(std(regression.ve, X; dims = 2))
    else
        vec(std(regression.ve, X, regression.std_w; dims = 2))
    end
    return transpose(pinv(Vp) * transpose(B .* transpose(sdev)))
end
function factors_b1_b2_b3(B::DataFrame, factors::AbstractMatrix, regression::RegressionType)
    B = _rebuild_B(B, factors, regression)
    b1 = pinv(transpose(B))
    b2 = pinv(transpose(nullspace(transpose(B))))
    b3 = pinv(transpose(b2))
    return b1, b2, b3, B
end
function _rp_class_constraints(port, ::Any, w_ini)
    model = port.model
    if isempty(port.risk_budget)
        port.risk_budget = ()
    elseif !isapprox(sum(port.risk_budget), one(eltype(port.returns)))
        port.risk_budget ./= sum(port.risk_budget)
    end
    N = length(port.risk_budget)
    initial_w(port, w_ini)
    w = port.model[:w]
    @variable(model, log_w[1:N])
    @constraint(model, dot(port.risk_budget, log_w) >= 1)
    @constraint(model, [i = 1:N], [log_w[i], 1, w[i]] ∈ MOI.ExponentialCone())
    @constraint(model, w .>= 0)
    return nothing
end
function _rp_class_constraints(port, class::FC, w_ini)
    model = port.model
    N = size(port.returns, 2)
    if class.flag
        b1, b2 = factors_b1_b2_b3(port.loadings, port.f_returns, port.regression_type)[1:2]
        N_f = size(b1, 2)
        @variable(model, w1[1:N_f])
        @variable(model, w2[1:(N - N_f)])
        @expression(model, w, b1 * w1 + b2 * w2)
    else
        b1 = factors_b1_b2_b3(port.loadings, port.f_returns, port.regression_type)[1]
        N_f = size(b1, 2)
        @variable(model, w1[1:N_f])
        @expression(model, w, b1 * w1)
    end

    set_w_ini(w1, w_ini)

    if isempty(port.f_risk_budget) || length(port.f_risk_budget) != N_f
        port.f_risk_budget = fill(inv(N_f), N_f)
    elseif !isapprox(sum(port.f_risk_budget), one(eltype(port.returns)))
        port.f_risk_budget ./= sum(port.f_risk_budget)
    end

    @variable(model, log_w[1:N_f])
    @constraint(model, dot(port.f_risk_budget, log_w) >= 1)
    @constraint(model, [i = 1:N_f], [log_w[i], 1, w1[i]] ∈ MOI.ExponentialCone())
    return nothing
end
function rp_constraints(port, class, w_ini)
    _rp_class_constraints(port, class, w_ini)
    model = port.model
    w = model[:w]
    @variable(model, k)
    @constraint(model, sum(w) == k)
    return nothing
end
function _optimise!(type::RP, port::Portfolio, rm::Union{AbstractVector, <:RiskMeasure},
                    ::Any, ::Any, class::PortClass, w_ini::AbstractVector, ::Any,
                    str_names::Bool)
    mu, sigma, returns = mu_sigma_returns_class(port, class)
    port.model = JuMP.Model()
    model = port.model
    set_string_names_on_creation(model, str_names)
    rp_constraints(port, class, w_ini)
    risk_constraints(port, nothing, type, rm, mu, sigma, returns)
    set_returns(nothing, NoKelly(), port; mu = mu)
    linear_constraints(port, type)
    risk = model[:risk]
    @objective(model, Min, risk)
    return convex_optimisation(port, nothing, type, class)
end
function rp_constraints(port::OmniPortfolio, ::Any, w_ini)
    N = size(port.returns, 2)
    risk_budget = port.risk_budget
    if isempty(risk_budget)
        risk_budget = port.risk_budget = fill(inv(N), N)
    end
    initial_w(port, w_ini)
    model = port.model
    w = model[:w]
    @variables(model, begin
                   k
                   log_w[1:N]
               end)
    @constraints(model, begin
                     dot(risk_budget, log_w) >= 1
                     [i = 1:N], [log_w[i], 1, w[i]] ∈ MOI.ExponentialCone()
                     w .>= 0
                     sum(w) == k
                 end)
    return nothing
end
function rp_constraints(port::OmniPortfolio, class::FC, w_ini)
    model = port.model
    f_returns = port.f_returns
    loadings = port.loadings
    regression_type = port.regression_type
    if class.flag
        b1, b2 = factors_b1_b2_b3(loadings, f_returns, regression_type)[1:2]
        N = size(port.returns, 2)
        N_f = size(b1, 2)
        @variables(model, begin
                       w1[1:N_f]
                       w2[1:(N - N_f)]
                   end)
        @expression(model, w, b1 * w1 + b2 * w2)
    else
        b1 = factors_b1_b2_b3(loadings, f_returns, regression_type)[1]
        N_f = size(b1, 2)
        @variable(model, w1[1:N_f])
        @expression(model, w, b1 * w1)
    end

    set_w_ini(w1, w_ini)

    if isempty(port.f_risk_budget) || length(port.f_risk_budget) != N_f
        port.f_risk_budget = fill(inv(N_f), N_f)
    end
    f_risk_budget = port.f_risk_budget
    @variables(model, begin
                   k
                   log_w[1:N_f]
               end)
    @constraints(model, begin
                     dot(f_risk_budget, log_w) >= 1
                     [i = 1:N_f], [log_w[i], 1, w1[i]] ∈ MOI.ExponentialCone()
                 end)
    return nothing
end
function _optimise!(type::RP, port::OmniPortfolio, rm::Union{AbstractVector, <:RiskMeasure},
                    ::Any, kelly::RetType, class::PortClass, w_ini::AbstractVector,
                    custom_constr, custom_obj, ::Any, str_names::Bool = false)
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
