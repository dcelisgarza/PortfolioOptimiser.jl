abstract type AbstractEffEVaR <: AbstractEfficient end

struct EffEVaR{T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14} <:
       AbstractEffEVaR
    tickers::T1
    mean_ret::T2
    weights::T3
    returns::T4
    beta::T5
    rf::T6
    market_neutral::T7
    risk_aversion::T8
    target_risk::T9
    target_ret::T10
    extra_vars::T11
    extra_constraints::T12
    extra_obj_terms::T13
    model::T14
end

function EffEVaR(tickers, mean_ret, returns; weight_bounds = (0.0, 1.0), beta = 0.95,
                 rf = 1.02^(1 / 252) - 1, market_neutral = false, risk_aversion = 1.0,
                 target_risk = mean(maximum(returns; dims = 2)),
                 target_ret = !isnothing(mean_ret) ? mean(mean_ret) : 0, extra_vars = [],
                 extra_constraints = [], extra_obj_terms = [],)
    num_tickers = length(tickers)
    @assert num_tickers == size(returns, 2)
    if !isnothing(mean_ret)
        @assert(num_tickers == length(mean_ret))
    end

    weights = zeros(num_tickers)

    beta = _val_compare_benchmark(beta, >=, 1, 0.95, "beta")
    beta = _val_compare_benchmark(beta, <, 0, 0.95, "beta")

    if beta <= 0.2
        @warn("beta: $beta is the confidence level, not percentile. It is typically 0.8, 0.9, 0.95")
    end

    model = Model()
    samples = size(returns, 1)
    @variable(model, w[1:num_tickers])

    # EVaR variables
    @variable(model, t)
    @variable(model, s >= 0)
    @variable(model, u[1:samples] >= 0)
    @expression(model, X, returns * w)

    # EVaR constraints.
    @constraint(model, sum_u_leq_s, sum(u) <= s)
    @constraint(model, evar_con[i = 1:samples],
                [-X[i] - t, s, u[i]] in MOI.ExponentialCone())

    lower_bounds, upper_bounds = _create_weight_bounds(num_tickers, weight_bounds)

    @constraint(model, lower_bounds, w .>= lower_bounds)
    @constraint(model, upper_bounds, w .<= upper_bounds)

    _make_weight_sum_constraint!(model, market_neutral)

    # Add extra variables for extra variables and objective functions.
    if !isempty(extra_vars)
        _add_var_to_model!.(model, extra_vars)
    end

    # We need to add the extra constraints.
    if !isempty(extra_constraints)
        constraint_keys = [Symbol("extra_constraint$(i)")
                           for i ∈ 1:length(extra_constraints)]
        _add_constraint_to_model!.(model, constraint_keys, extra_constraints)
    end

    if !isnothing(mean_ret)
        @expression(model, ret, port_return(w, mean_ret))
    end
    @expression(model, risk, t + s * log(1 / ((1 - beta) * samples)))

    return EffEVaR(tickers, mean_ret, weights, returns, beta, rf, market_neutral,
                   risk_aversion, target_risk, target_ret, extra_vars, extra_constraints,
                   extra_obj_terms, model)
end
