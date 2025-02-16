# Copywrite (c) 2025
# Author: Daniel Celis Garza <daniel.celis.garza@gmail.com>
# SPDX-License-Identifier: MIT

"""
```
expected_risk(port::AbstractPortfolio; X::AbstractMatrix = port.returns,
          type::Symbol = isa(port, Portfolio) || isa(port, Portfolio) ? :Trad : :HRP,
          rm::AbstractRiskMeasure = SD())
```

Compute the risk for an [`AbstractRiskMeasure`](@ref) for a portfolio.

# Inputs

  - `port`: portfolio.
  - `X`: `T×N` returns matrix.
  - `type`: optimisation type used to retrieve the weights vector from `port.optimal[type]`.
  - `rm`: rism measure.

# Outputs

  - `r`: risk.
"""
function expected_risk(port::AbstractPortfolio, key = :Trad;
                       X::AbstractMatrix = port.returns,
                       rm::AbstractRiskMeasure = Variance())
    solver_flag, sigma_flag, skew_flag, sskew_flag = set_rm_properties!(rm, port.solvers,
                                                                        port.cov, port.V,
                                                                        port.SV)
    risk = expected_risk(rm, port.optimal[key].weights; X = X, fees = port.fees,
                         rebalance = port.rebalance)
    unset_set_rm_properties!(rm, solver_flag, sigma_flag, skew_flag, sskew_flag)
    return risk
end

function calc_fees(port::AbstractPortfolio, key = :Trad;
                   latest_prices::AbstractVector = port.latest_prices)
    return calc_fees(port.optimal[key].weights, latest_prices, port.fees, port.rebalance)
end

function expected_ret(port::AbstractPortfolio, key = :Trad; mu::AbstractVector = port.mu,
                      X::AbstractMatrix = port.returns, kelly::Union{Bool, RetType} = false)
    return expected_ret(port.optimal[key].weights; mu = mu, X = X, kelly = kelly,
                        fees = port.fees, rebalance = port.rebalance)
end

"""
```
risk_contribution(port::AbstractPortfolio; X::AbstractMatrix = port.returns,
                  type::Symbol = isa(port, Portfolio) || isa(port, Portfolio) ? :Trad : :HRP,
                  rm::AbstractRiskMeasure = SD(), delta::Real = 1e-6,
                  marginal::Bool = false)
```

Compute the asset risk contribution for an [`AbstractRiskMeasure`](@ref) for a portfolio.

# Inputs

  - `port`: portfolio.

  - `X`: `T×N` returns matrix.
  - `type`: optimisation type used to retrieve the weights vector from `port.optimal[type]`.
  - `rm`: risk measure [`AbstractRiskMeasure`](@ref).
  - `delta`: small displacement used for computing the marginal risk and equal risk measure [`Equal`](@ref).
  - `marginal`:

      + if `true`: compute the marginal risk contribution.
      + else: compute the risk by contribution by multiplying the marginal risk by the asset weight.

# Outputs

  - `rc`: `Na×1` vector of risk contribution per asset.
"""
function risk_contribution(port::AbstractPortfolio, key = :Trad;
                           X::AbstractMatrix = port.returns,
                           rm::AbstractRiskMeasure = Variance(), delta::Real = 1e-6,
                           marginal::Bool = false)
    solver_flag, sigma_flag, skew_flag, sskew_flag = set_rm_properties!(rm, port.solvers,
                                                                        port.cov, port.V,
                                                                        port.SV)

    risk = risk_contribution(rm, port.optimal[key].weights; X = X, delta = delta,
                             marginal = marginal, fees = port.fees,
                             rebalance = port.rebalance)
    unset_set_rm_properties!(rm, solver_flag, sigma_flag, skew_flag, sskew_flag)
    return risk
end

"""
```
factor_risk_contribution(port::AbstractPortfolio; X::AbstractMatrix = port.returns,
                         F::AbstractMatrix = port.f_returns,
                         type::Symbol = isa(port, Portfolio) || isa(port, Portfolio) ? :Trad : :HRP,
                         rm::AbstractRiskMeasure = SD(), delta::Real = 1e-6)
```

Compute the factor risk contribution for an [`AbstractRiskMeasure`](@ref) for a portfolio.

# Inputs

  - `port`: portfolio.
  - `w`: `Na×1` vector of asset weights.
  - `X`: `T×Na` matrix of asset returns.
  - `F`: `T×Nf` matrix of factor returns.
  - `type`: optimisation type used to retrieve the weights vector from `port.optimal[type]`.
  - `rm`: risk measure [`AbstractRiskMeasure`](@ref).
  - `delta`: small displacement used for computing the marginal risk and equal risk measure [`Equal`](@ref).

# Outputs

  - `rc_f`: `Nf×1` vector of risk contribution per factor.
"""
function factor_risk_contribution(port::AbstractPortfolio, key = :Trad;
                                  X::AbstractMatrix = port.returns,
                                  F::AbstractMatrix = port.f_returns,
                                  rm::AbstractRiskMeasure = Variance(), delta::Real = 1e-6)
    solver_flag, sigma_flag, skew_flag, sskew_flag = set_rm_properties!(rm, port.solvers,
                                                                        port.cov, port.V,
                                                                        port.SV)
    risk = factor_risk_contribution(rm, port.optimal[key].weights; X = X,
                                    assets = port.assets, F = F, f_assets = port.f_assets,
                                    B = port.loadings,
                                    regression_type = port.regression_type, delta = delta,
                                    fees = port.fees, rebalance = port.rebalance,
                                    scale = true)
    unset_set_rm_properties!(rm, solver_flag, sigma_flag, skew_flag, sskew_flag)
    return risk
end

"""
```
sharpe_ratio(port::AbstractPortfolio; X::AbstractMatrix = port.returns,
             mu::AbstractVector = port.mu,
             type::Symbol = isa(port, Portfolio) || isa(port, Portfolio) ? :Trad : :HRP,
             rm::AbstractRiskMeasure = SD(), delta::Real = 1e-6, rf::Real = 0.0,
             kelly::Bool = false)
```

Compute the risk-adjusted return ratio for an [`AbstractRiskMeasure`](@ref) for a portfolio.

# Inputs

  - `port`: portfolio.

  - `X`: `T×N` matrix of asset returns.
  - `mu`: `N×1` vector of expected returns.
  - `type`: optimisation type used to retrieve the weights vector from `port.optimal[type]`.
  - `rm`: risk measure [`AbstractRiskMeasure`](@ref).
  - `delta`: small displacement used for computing the [`Equal`](@ref) risk measure.
  - `rf`: risk free rate.
  - `kelly`:

      + if `true`: use the kelly return.
      + else: use the arithmetic return.

# Outputs

  - `sr`: risk adjusted return ratio.
"""
function sharpe_ratio(port::AbstractPortfolio, key = :Trad;
                      X::AbstractMatrix = port.returns, mu::AbstractVector = port.mu,
                      rm::AbstractRiskMeasure = Variance(), delta::Real = 1e-6,
                      rf::Real = 0.0, kelly::Bool = false)
    solver_flag, sigma_flag, skew_flag, sskew_flag = set_rm_properties!(rm, port.solvers,
                                                                        port.cov, port.V,
                                                                        port.SV)
    sr = sharpe_ratio(rm, port.optimal[key].weights; mu = mu, X = X, delta = delta, rf = rf,
                      kelly = kelly, fees = port.fees, rebalance = port.rebalance)
    unset_set_rm_properties!(rm, solver_flag, sigma_flag, skew_flag, sskew_flag)
    return sr
end
function expected_ret_risk_sharpe(port::AbstractPortfolio, key = :Trad;
                                  X::AbstractMatrix = port.returns,
                                  mu::AbstractVector = port.mu,
                                  rm::AbstractRiskMeasure = Variance(), delta::Real = 1e-6,
                                  rf::Real = 0.0, kelly::Bool = false)
    solver_flag, sigma_flag, skew_flag, sskew_flag = set_rm_properties!(rm, port.solvers,
                                                                        port.cov, port.V,
                                                                        port.SV)
    ret, risk, sharpe = expected_ret_risk_sharpe(rm, port.optimal[key].weights; mu = mu,
                                                 X = X, delta = delta, rf = rf,
                                                 kelly = kelly, fees = port.fees,
                                                 rebalance = port.rebalance)
    unset_set_rm_properties!(rm, solver_flag, sigma_flag, skew_flag, sskew_flag)
    return ret, risk, sharpe
end
function sharpe_ratio_info_criterion(port::AbstractPortfolio, key = :Trad;
                                     X::AbstractMatrix = port.returns,
                                     mu::AbstractVector = port.mu,
                                     rm::AbstractRiskMeasure = Variance(),
                                     delta::Real = 1e-6, rf::Real = 0.0,
                                     kelly::Bool = false)
    solver_flag, sigma_flag, skew_flag, sskew_flag = set_rm_properties!(rm, port.solvers,
                                                                        port.cov, port.V,
                                                                        port.SV)
    sric = sharpe_ratio_info_criterion(rm, port.optimal[key].weights; mu = mu, X = X,
                                       delta = delta, rf = rf, kelly = kelly,
                                       fees = port.fees, rebalance = port.rebalance)
    unset_set_rm_properties!(rm, solver_flag, sigma_flag, skew_flag, sskew_flag)
    return sric
end
"""
```
number_effective_assets(port; type::Symbol = isa(port, Portfolio) || isa(port, Portfolio) ? :Trad : :HRP)
```

Compute the number of effective assets.
"""
function number_effective_assets(port::AbstractPortfolio, key = :Trad)
    return number_effective_assets(port.optimal[key].weights)
end
