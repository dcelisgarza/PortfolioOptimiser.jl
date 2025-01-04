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
function rp_constraints(port::Portfolio, ::Any, w_ini)
    N = size(port.returns, 2)
    risk_budget = port.risk_budget
    if isempty(risk_budget)
        risk_budget = port.risk_budget = fill(inv(N), N)
    end
    initial_w(port, w_ini)
    model = port.model
    w = model[:w]
    scale_constr = model[:scale_constr]
    @variables(model, begin
                   k
                   log_w[1:N]
                   c >= 0
               end)
    @constraints(model,
                 begin
                     constr_log_w[i = 1:N],
                     [scale_constr * log_w[i], scale_constr * 1, scale_constr * w[i]] ∈
                     MOI.ExponentialCone()
                     constr_risk_budget,
                     scale_constr * dot(risk_budget, log_w) >= scale_constr * c
                 end)
    return nothing
end
function rp_constraints(port::Portfolio, class::FC, w_ini)
    model = port.model
    scale_constr = model[:scale_constr]
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
                   c >= 0
               end)
    @constraints(model,
                 begin
                     constr_factor_log_w[i = 1:N_f],
                     [scale_constr * log_w[i], scale_constr * 1, scale_constr * w1[i]] ∈
                     MOI.ExponentialCone()
                     constr_factor_risk_budget,
                     scale_constr * dot(f_risk_budget, log_w) >= scale_constr * c
                 end)
    return nothing
end
