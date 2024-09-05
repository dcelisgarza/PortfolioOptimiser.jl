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
function _factors_b1_b2_b3(B::DataFrame, factors::AbstractMatrix,
                           regression::RegressionType)
    B = _rebuild_B(B, factors, regression)
    b1 = pinv(transpose(B))
    b2 = pinv(transpose(nullspace(transpose(B))))
    b3 = pinv(transpose(b2))
    return b1, b2, b3, B
end
function _rp_class_constraints(::Any, port)
    model = port.model
    if isempty(port.risk_budget)
        port.risk_budget = ()
    elseif !isapprox(sum(port.risk_budget), one(eltype(port.returns)))
        port.risk_budget ./= sum(port.risk_budget)
    end
    N = length(port.risk_budget)
    @variable(model, w[1:N])
    @variable(model, log_w[1:N])
    @constraint(model, dot(port.risk_budget, log_w) >= 1)
    @constraint(model, [i = 1:N], [log_w[i], 1, w[i]] ∈ MOI.ExponentialCone())
    @constraint(model, w .>= 0)
    return nothing
end
function _rp_class_constraints(class::FC, port)
    model = port.model
    N = size(port.returns, 2)
    if class.flag
        b1, b2 = _factors_b1_b2_b3(port.loadings, port.f_returns, port.loadings_opt)[1:2]
        N_f = size(b1, 2)
        @variable(model, w1[1:N_f])
        @variable(model, w2[1:(N - N_f)])
        @expression(model, w, b1 * w1 + b2 * w2)
    else
        b1 = _factors_b1_b2_b3(port.loadings, port.f_returns, port.loadings_opt)[1]
        N_f = size(b1, 2)
        @variable(model, w1[1:N_f])
        @expression(model, w, b1 * w1)
    end

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
function rp_constraints(port, class)
    model = port.model
    _rp_class_constraints(class, port)
    @variable(model, k)
    w = model[:w]
    k = model[:k]
    @constraint(model, sum(w) == k)
    return nothing
end
function _optimise!(type::RP, port::Portfolio, rm::Union{AbstractVector, <:TradRiskMeasure},
                    ::Any, ::Any, class::Union{Classic, FM, FC}, w_ini::AbstractVector,
                    str_names::Bool)
    mu, sigma, returns = mu_sigma_returns_class(port, class)
    port.model = JuMP.Model()
    model = port.model
    set_string_names_on_creation(model, str_names)
    rp_constraints(port, class)
    initial_w(port, w_ini)
    risk_constraints(port, nothing, RP(), rm, mu, sigma, returns)
    set_returns(nothing, NoKelly(), port.model, port.mu_l; mu = mu)
    linear_constraints(port, type)
    risk = model[:risk]
    @objective(model, Min, risk)
    return convex_optimisation(port, nothing, RP(), class)
end