"""
```
abstract type AbstractEfficientMeanVar <: AbstractEfficient end
```

Abstract type for subtyping efficient mean variance optimisers.
"""
abstract type AbstractEfficientMeanVar <: AbstractEfficient end

"""
```
struct EfficientMeanVar{T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13} <:
       AbstractEfficientMeanVar
    tickers::T1
    mean_ret::T2
    weights::T3
    cov_mtx::T4
    rf::T5
    market_neutral::T6
    risk_aversion::T7
    target_volatility::T8
    target_ret::T9
    extra_vars::T10
    extra_constraints::T11
    extra_obj_terms::T12
    model::T13
end
```

Structure for a mean-variance portfolio.

- `tickers`: list of tickers.
- `mean_ret`: mean returns, don't need it to optimise for minimum variance.
- `weights`: weight of each ticker in the portfolio.
- `cov_mtx`: covariance matrix.
- `rf`: risk free return.
- `market_neutral`: whether a portfolio is market neutral or not. Used in [`max_quadratic_utility!`](@ref), [`efficient_risk!`](@ref), [`efficient_return!`](@ref).
- `risk_aversion`: risk aversion parameter. Used in [`max_quadratic_utility!`](@ref).
- `target_volatility`: target volatility parameter. Used in [`efficient_risk!`](@ref).
- `target_ret`: target return parameter. Used in [`efficient_return!`](@ref).
- `extra_vars`: extra variables for the model.
- `extra_constraints`: extra constraints for the model.
- `extra_obj_terms`: extra objective terms for the model.
- `model`: model for optimising portfolio.
"""
struct EfficientMeanVar{T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13} <:
       AbstractEfficientMeanVar
    tickers::T1
    mean_ret::T2
    weights::T3
    cov_mtx::T4
    rf::T5
    market_neutral::T6
    risk_aversion::T7
    target_volatility::T8
    target_ret::T9
    extra_vars::T10
    extra_constraints::T11
    extra_obj_terms::T12
    model::T13
end

"""
```
EfficientMeanVar(
    tickers,
    mean_ret,
    cov_mtx;
    weight_bounds = (0, 1),
    rf = 0.02,
    market_neutral = false,
    risk_aversion = 1.0,
    target_volatility = rank(cov_mtx) < size(cov_mtx, 1) ? 1 / sum(diag(cov_mtx)) :
                        sqrt(1 / sum(inv(cov_mtx))),
    target_ret = mean(mean_ret),
    extra_vars = [],
    extra_constraints = [],
    extra_obj_terms = [],
)
```

Function for creating a JuMP model for creating an efficient mean-var portfolio to be optimised.

- `tickers`: list of tickers.
- `mean_ret`: mean returns, don't need it to optimise for minimum variance.
- `weights`: weight of each ticker in the portfolio.
- `cov_mtx`: covariance matrix.
- `rf`: risk free return.
- `market_neutral`: whether a portfolio is market neutral or not. Used in [`max_quadratic_utility!`](@ref), [`efficient_risk!`](@ref), [`efficient_return!`](@ref).
- `risk_aversion`: risk aversion parameter. Used in [`max_quadratic_utility!`](@ref).
- `target_volatility`: target volatility parameter. Used in [`efficient_risk!`](@ref).
- `target_ret`: target return parameter. Used in [`efficient_return!`](@ref).
- `extra_vars`: extra variables for the model. See [`_add_var_to_model!`](@ref) for details on how to use this.
- `extra_constraints`: extra constraints for the model. See [`_add_constraint_to_model!`](@ref) for details on how to use this.
- `extra_obj_terms`: extra objective terms for the model. See [`_add_to_objective!`](@ref) for details on how to use this.
"""
function EfficientMeanVar(
    tickers,
    mean_ret,
    cov_mtx;
    weight_bounds = (0, 1),
    rf = 0.02,
    market_neutral = false,
    risk_aversion = 1.0,
    target_volatility = rank(cov_mtx) < size(cov_mtx, 1) ? 1 / sum(diag(cov_mtx)) :
                        sqrt(1 / sum(inv(cov_mtx))),
    target_ret = mean(mean_ret),
    extra_vars = [],
    extra_constraints = [],
    extra_obj_terms = [],
)
    num_tickers = length(tickers)
    @assert num_tickers == length(mean_ret) == size(cov_mtx, 1) == size(cov_mtx, 2)
    weights = zeros(num_tickers)

    model = Model()
    @variable(model, w[1:num_tickers])

    lower_bounds, upper_bounds = _create_weight_bounds(num_tickers, weight_bounds)

    @constraint(model, lower_bounds, w .>= lower_bounds)
    @constraint(model, upper_bounds, w .<= upper_bounds)

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

    return EfficientMeanVar(
        tickers,
        mean_ret,
        weights,
        cov_mtx,
        rf,
        market_neutral,
        risk_aversion,
        target_volatility,
        target_ret,
        extra_vars,
        extra_constraints,
        extra_obj_terms,
        model,
    )
end