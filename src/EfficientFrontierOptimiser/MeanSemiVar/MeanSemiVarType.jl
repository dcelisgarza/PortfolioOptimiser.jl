abstract type AbstractEfficientSemiVar <: AbstractEfficient end

"""
```
struct EfficientSemiVar{
    T1,
    T2,
    T3,
    T4,
    T5,
    T6,
    T7,
    T8,
    T9,
    T10,
    T11,
    T12,
    T13,
    T14,
    T15,
} <: AbstractEfficientSemiVar
    tickers::T1
    mean_ret::T2
    weights::T3
    returns::T4
    freq::T5
    benchmark::T6
    rf::T7
    market_neutral::T8
    risk_aversion::T9
    target_semidev::T10
    target_ret::T11
    extra_vars::T12
    extra_constraints::T13
    extra_obj_terms::T14
    model::T15
end
```
"""
struct EfficientSemiVar{T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15} <:
       AbstractEfficientSemiVar
    tickers::T1
    mean_ret::T2
    weights::T3
    returns::T4
    freq::T5
    benchmark::T6
    rf::T7
    market_neutral::T8
    risk_aversion::T9
    target_semidev::T10
    target_ret::T11
    extra_vars::T12
    extra_constraints::T13
    extra_obj_terms::T14
    model::T15
end
function EfficientSemiVar(
    tickers,
    mean_ret,
    returns;
    weight_bounds = (0.0, 1.0),
    freq = 252,
    benchmark = 0,#1.02^(1 / 252) - 1,
    rf = 0.02,
    market_neutral = false,
    risk_aversion = 1.0,
    target_semidev = std(returns),
    target_ret = mean(mean_ret),
    extra_vars = [],
    extra_constraints = [],
    extra_obj_terms = [],
)
    num_tickers = length(tickers)
    @assert num_tickers == length(mean_ret) == size(returns, 2)
    weights = zeros(num_tickers)

    model = Model()
    @variable(model, w[1:num_tickers])

    samples = size(returns, 1)
    @variable(model, p[1:samples] >= 0)
    @variable(model, n[1:samples] >= 0)

    B = semi_ret(returns, benchmark)
    @constraint(model, semi_var, B * w .- p .+ n .== 0)

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

    return EfficientSemiVar(
        tickers,
        mean_ret,
        weights,
        returns,
        freq,
        benchmark,
        rf,
        market_neutral,
        risk_aversion,
        target_semidev,
        target_ret,
        extra_vars,
        extra_constraints,
        extra_obj_terms,
        model,
    )
end