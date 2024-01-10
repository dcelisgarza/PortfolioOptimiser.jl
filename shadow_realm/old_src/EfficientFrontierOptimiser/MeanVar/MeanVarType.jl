"""
```
abstract type AbstractEffMeanVar <: AbstractEfficient end
```

Abstract type for subtyping efficient mean variance optimisers.
"""
abstract type AbstractEffMeanVar <: AbstractEfficient end

"""
```
struct EffMeanVar{T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13} <:
       AbstractEffMeanVar
    tickers::T1
    mean_ret::T2
    weights::T3
    cov_mtx::T4
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
```

Structure for a mean-variance portfolio.

- `tickers`: list of tickers.
- `mean_ret`: mean returns, don't need it to optimise for minimum variance.
- `weights`: weight of each ticker in the portfolio.
- `cov_mtx`: covariance matrix.
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
struct EffMeanVar{T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13} <:
       AbstractEffMeanVar
    tickers::T1
    mean_ret::T2
    weights::T3
    cov_mtx::T4
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

"""
```
EffMeanVar(
    tickers,
    mean_ret,
    cov_mtx;
    weight_bounds = (0, 1),
    rf = 1.02^(1/252)-1,
    market_neutral = false,
    risk_aversion = 1.0,
    target_risk = rank(cov_mtx) < size(cov_mtx, 1) ? 1 / sum(diag(cov_mtx)) :
                        sqrt(1 / sum(inv(cov_mtx))),
    target_ret = mean(mean_ret),
    extra_vars = [],
    extra_constraints = [],
    extra_obj_terms = [],
)
```

Create an [`EffMeanVar`](@ref) structure to be optimised via JuMP.

- `tickers`: list of tickers.
- `mean_ret`: mean returns, don't need it to optimise for minimum variance.
- `cov_mtx`: covariance matrix.
- `weight_bounds`: weight bounds for tickers. If it's a Tuple of length 2, the first entry will be the lower bound for all weights, the second entry will be the upper bound for all weights. If it's a vector, its length must be equal to that of `tickers`, each element must be a tuple of length 2. In that case, each tuple corresponds to the lower and upper bounds for the corresponding ticker. See [`_create_weight_bounds`](@ref) for further details.
- `rf`: risk free rate. Must be consistent with the frequency at which `mean_ret` and `cov_mtx` were calculated. The default value assumes daily returns.
- `market_neutral`: whether a portfolio is market neutral or not. If it is market neutral, the sum of the weights will be equal to 0, else the sum will be equal to 1. Used in [`max_utility!`](@ref), [`efficient_risk!`](@ref), [`efficient_return!`](@ref).
- `risk_aversion`: risk aversion parameter, the larger it is, the lower the risk. Used in [`max_utility!`](@ref).
- `target_risk`: target volatility parameter. Used in [`efficient_risk!`](@ref).
- `target_ret`: target return parameter. Used in [`efficient_return!`](@ref).
- `extra_vars`: extra variables for the model. See [`_add_var_to_model!`](@ref) for details on how to use this.
- `extra_constraints`: extra constraints for the model. See [`_add_constraint_to_model!`](@ref) for details on how to use this.
- `extra_obj_terms`: extra objective terms for the model. See [`_add_to_objective!`](@ref) for details on how to use this.
"""
function EffMeanVar(tickers,
                    mean_ret,
                    cov_mtx;
                    weight_bounds = (0, 1),
                    rf = 1.02^(1 / 252) - 1,
                    market_neutral = false,
                    risk_aversion = 1.0,
                    target_risk = rank(cov_mtx) < size(cov_mtx, 1) ?
                                  1 / sum(diag(cov_mtx)) :
                                  sqrt(1 / sum(inv(cov_mtx))),
                    target_ret = !isnothing(mean_ret) ? mean(mean_ret) : 0,
                    extra_vars = [],
                    extra_constraints = [],
                    extra_obj_terms = [],)
    num_tickers = length(tickers)
    @assert num_tickers == size(cov_mtx, 1) == size(cov_mtx, 2)
    !isnothing(mean_ret) && @assert(num_tickers == length(mean_ret))

    weights = zeros(num_tickers)

    model = Model()
    @variable(model, w[1:num_tickers])

    lower_bounds, upper_bounds = _create_weight_bounds(num_tickers,
                                                       weight_bounds)

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
                           for i in 1:length(extra_constraints)]
        _add_constraint_to_model!.(model, constraint_keys, extra_constraints)
    end

    # Return and risk.
    !isnothing(mean_ret) && @expression(model, ret, port_return(w, mean_ret))
    @expression(model, risk, port_variance(w, cov_mtx))

    # Second order conic constraints.
    # @variable(model, g >= 0)
    # G = sqrt(cov_mtx)
    # @constraint(model, g_cone, [g; G * w] in SecondOrderCone())
    # @expression(model, risk, g^2)

    return EffMeanVar(tickers,
                      mean_ret,
                      weights,
                      cov_mtx,
                      rf,
                      market_neutral,
                      risk_aversion,
                      target_risk,
                      target_ret,
                      extra_vars,
                      extra_constraints,
                      extra_obj_terms,
                      model)
end
