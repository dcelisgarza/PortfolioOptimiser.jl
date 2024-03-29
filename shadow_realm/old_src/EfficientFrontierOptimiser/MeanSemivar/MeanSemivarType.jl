"""
```
abstract type AbstractEffMeanSemivar <: AbstractEfficient end
```

Abstract type for subtyping efficient mean semivariance optimisers.
"""
abstract type AbstractEffMeanSemivar <: AbstractEfficient end

"""
```
struct EffMeanSemivar{T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15} <:
       AbstractEffMeanSemivar
    tickers::T1
    mean_ret::T2
    weights::T3
    returns::T4
    target::T6
    rf::T7
    market_neutral::T8
    risk_aversion::T9
    target_risk::T10
    target_ret::T11
    extra_vars::T12
    extra_constraints::T13
    extra_obj_terms::T14
    model::T15
end
```

Structure for a mean-semivariance portfolio.

  - `tickers`: list of tickers.
  - `mean_ret`: mean returns, don't need it to optimise for minimum variance.
  - `weights`: weight of each ticker in the portfolio.
  - `returns`: asset historical returns.
  - `target`: returns target, to differentiate between "downside" (less than `target`) and "upside" (greater than `target`) returns.
  - `rf`: risk free rate.
  - `market_neutral`: whether a portfolio is market neutral or not. Used in [`max_utility!`](@ref), [`efficient_risk!`](@ref), [`efficient_return!`](@ref).
  - `risk_aversion`: risk aversion parameter. Used in [`max_utility!`](@ref).
  - `target_risk`: target volatility parameter. Used in [`efficient_risk!`](@ref).
  - `target_ret`: target return parameter. Used in [`efficient_return!`](@ref).
  - `extra_vars`: extra variables for the model.
  - `extra_constraints`: extra constraints for the model.
  - `extra_obj_terms`: extra objective terms for the model.
  - `model`: model for optimising portfolio.
"""
struct EffMeanSemivar{T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14} <:
       AbstractEffMeanSemivar
    tickers::T1
    mean_ret::T2
    weights::T3
    returns::T4
    target::T5
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

"""
```
EffMeanSemivar(
    tickers,
    mean_ret,
    returns;
    weight_bounds = (0.0, 1.0),
    target = 1.02^(1 / 252) - 1,
    rf = 1.02^(1 / 252) - 1,
    market_neutral = false,
    risk_aversion = 1.0,
    target_risk = std(returns),
    target_ret = mean(mean_ret),
    extra_vars = [],
    extra_constraints = [],
    extra_obj_terms = [],
)
```

Create an [`EffMeanSemivar`](@ref) structure to be optimised via JuMP.

  - `tickers`: list of tickers.
  - `mean_ret`: mean returns, don't need it to optimise for minimum variance.
  - `returns`: asset historical returns.
  - `weight_bounds`: weight bounds for tickers. If it's a Tuple of length 2, the first entry will be the lower bound for all weights, the second entry will be the upper bound for all weights. If it's a vector, its length must be equal to that of `tickers`, each element must be a tuple of length 2. In that case, each tuple corresponds to the lower and upper bounds for the corresponding ticker. See [`_create_weight_bounds`](@ref) for further details.
  - `target`: returns target, to differentiate between "downside" (less than `target`) and "upside" (greater than `target`) returns.
  - `rf`: risk free rate. Must be consistent with the frequency of returns. The default value assumes daily returns.
  - `market_neutral`: whether a portfolio is market neutral or not. If it is market neutral, the sum of the weights will be equal to 0, else the sum will be equal to 1. Used in [`max_utility!`](@ref), [`efficient_risk!`](@ref), [`efficient_return!`](@ref).
  - `risk_aversion`: risk aversion parameter, the larger it is, the lower the risk. Used in [`max_utility!`](@ref).
  - `target_risk`: target volatility parameter. Used in [`efficient_risk!`](@ref).
  - `target_ret`: target return parameter. Used in [`efficient_return!`](@ref).
  - `extra_vars`: extra variables for the model. See [`_add_var_to_model!`](@ref) for details on how to use this.
  - `extra_constraints`: extra constraints for the model. See [`_add_constraint_to_model!`](@ref) for details on how to use this.
  - `extra_obj_terms`: extra objective terms for the model. See [`_add_to_objective!`](@ref) for details on how to use this.
"""
function EffMeanSemivar(tickers, mean_ret, returns; weight_bounds = (0.0, 1.0),
                        target = 1.02^(1 / 252) - 1, rf = 1.02^(1 / 252) - 1,
                        market_neutral = false, risk_aversion = 1.0,
                        target_risk = std(returns),
                        target_ret = !isnothing(mean_ret) ? mean(mean_ret) : 0,
                        extra_vars = [], extra_constraints = [], extra_obj_terms = [],)
    num_tickers = length(tickers)
    @assert num_tickers == size(returns, 2)
    if !isnothing(mean_ret)
        @assert(num_tickers == length(mean_ret))
    end

    weights = zeros(num_tickers)

    model = Model()
    @variable(model, w[1:num_tickers])

    samples = size(returns, 1)

    @variable(model, n[1:samples] >= 0)

    B = returns .- target
    @constraint(model, semi_var, B * w + n .>= 0)

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

    # Return and risk.
    if !isnothing(mean_ret)
        @expression(model, ret, port_return(w, mean_ret))
    end
    @expression(model, risk, dot(n, n) / (samples - 1))

    # # Second order cone constraints
    # @variable(model, g >= 0)
    # @constraint(model, g_cone, [g; n] in SecondOrderCone())
    # @expression(model, risk, g^2)

    return EffMeanSemivar(tickers, mean_ret, weights, returns, target, rf, market_neutral,
                          risk_aversion, target_risk, target_ret, extra_vars,
                          extra_constraints, extra_obj_terms, model)
end
