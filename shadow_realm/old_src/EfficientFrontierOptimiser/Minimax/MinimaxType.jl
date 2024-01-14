abstract type AbstractEffMinimax <: AbstractEfficient end

struct EffMinimax{T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13} <:
       AbstractEffMinimax
    tickers::T1
    mean_ret::T2
    weights::T3
    returns::T4
    rf::T5
    market_neutral::T6
    risk_aversion::T7
    target_risk::T8
    target_ret::T9
    extra_vars::T10
    extra_constraints::T11
    extra_obj_terms::T12
    model::T13
end

function EffMinimax(tickers, mean_ret, returns; weight_bounds = (0.0, 1.0),
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

    model = Model()
    @variable(model, w[1:num_tickers])

    # Minimax variables
    @variable(model, m)

    # Minimax constraints.
    @constraint(model, minimax, returns * w .+ m .>= 0)

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
    @expression(model, risk, m)

    return EffMinimax(tickers, mean_ret, weights, returns, rf, market_neutral,
                      risk_aversion, target_risk, target_ret, extra_vars, extra_constraints,
                      extra_obj_terms, model)
end
