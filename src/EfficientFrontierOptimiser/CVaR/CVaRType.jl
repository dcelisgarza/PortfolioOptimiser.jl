abstract type AbstractEffCVaR <: AbstractEfficient end

struct EffCVaR{T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14} <:
       AbstractEffCVaR
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

function EffCVaR(
    tickers,
    mean_ret,
    returns;
    weight_bounds = (0.0, 1.0),
    beta = 0.95,
    rf = 1.02^(1 / 252) - 1,
    market_neutral = false,
    risk_aversion = 1.0,
    target_risk = mean(maximum(returns, dims = 2)),
    target_ret = !isnothing(mean_ret) ? mean(mean_ret) : 0,
    extra_vars = [],
    extra_constraints = [],
    extra_obj_terms = [],
)
    num_tickers = length(tickers)
    @assert num_tickers == size(returns, 2)
    !isnothing(mean_ret) && @assert(num_tickers == length(mean_ret))

    weights = zeros(num_tickers)

    beta = _val_compare_benchmark(beta, >=, 1, 0.95, "beta")
    beta = _val_compare_benchmark(beta, <, 0, 0.95, "beta")

    if beta <= 0.2
        @warn(
            "beta: $beta is the confidence level, not percentile. It is typically 0.8, 0.9, 0.95"
        )
    end

    model = Model()
    samples = size(returns, 1)
    @variable(model, w[1:num_tickers])

    # CVaR variables
    @variable(model, alpha)
    @variable(model, u[1:samples] >= 0)

    # CVaR constraints.
    @constraint(model, vw_a_u_geq_0, returns * w + u .+ alpha .>= 0)

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
        constraint_keys =
            [Symbol("extra_constraint$(i)") for i in 1:length(extra_constraints)]
        _add_constraint_to_model!.(model, constraint_keys, extra_constraints)
    end

    !isnothing(mean_ret) && @expression(model, ret, port_return(w, mean_ret))
    @expression(model, risk, cvar(alpha, u, samples, beta))

    return EffCVaR(
        tickers,
        mean_ret,
        weights,
        returns,
        beta,
        rf,
        market_neutral,
        risk_aversion,
        target_risk,
        target_ret,
        extra_vars,
        extra_constraints,
        extra_obj_terms,
        model,
    )
end
