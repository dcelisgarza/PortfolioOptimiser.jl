"""
```
calc_risk(port::AbstractPortfolio; X::AbstractMatrix = port.returns,
          type::Symbol = isa(port, Portfolio) ? :Trad : :HRP,
          rm::AbstractRiskMeasure = SD())
```

Compute the risk for an [`AbstractRiskMeasure`](@ref) for a portfolio.

# Inputs

  - `port`: portfolio.
  - `X`: `T×N` returns matrix.
  - `type`: optimisation type used to retrieve the weights vector from `port.optimal[type]`.
  - `rm`: rism measure.
"""
function calc_risk(port::AbstractPortfolio; X::AbstractMatrix = port.returns,
                   type::Symbol = isa(port, Portfolio) ? :Trad : :HRP,
                   rm::AbstractRiskMeasure = SD())
    solver_flag, sigma_flag = set_rm_properties(rm, port.solvers, port.cov)
    risk = calc_risk(rm, port.optimal[type].weights; X = X, V = port.V, SV = port.SV)
    unset_set_rm_properties(rm, solver_flag, sigma_flag)
    return risk
end

"""
```
risk_contribution(port::AbstractPortfolio; X::AbstractMatrix = port.returns,
                  type::Symbol = isa(port, Portfolio) ? :Trad : :HRP,
                  rm::AbstractRiskMeasure = SD(), delta::Real = 1e-6,
                  marginal::Bool = false)
```

Compute the asset risk contribution for an [`AbstractRiskMeasure`](@ref) for a portfolio.

# Inputs

  - `port`: portfolio.

  - `X`: `T×N` returns matrix.
  - `type`: optimisation type used to retrieve the weights vector from `port.optimal[type]`.
  - `rm`: risk measure.
  - `delta`: small displacement used for computing the marginal risk and equal risk measure [`Equal`](@ref).
  - `marginal`:

      + if `true`: compute the marginal risk contribution.
      + else: compute the risk by contribution by multiplying the marginal risk by the asset weight.

# Outputs

  - `rc`: `Na×1` vector of risk contribution per asset.
"""
function risk_contribution(port::AbstractPortfolio; X::AbstractMatrix = port.returns,
                           type::Symbol = isa(port, Portfolio) ? :Trad : :HRP,
                           rm::AbstractRiskMeasure = SD(), delta::Real = 1e-6,
                           marginal::Bool = false)
    solver_flag, sigma_flag = set_rm_properties(rm, port.solvers, port.cov)
    risk = risk_contribution(rm, port.optimal[type].weights; X = X, V = port.V,
                             SV = port.SV, delta = delta, marginal = marginal)
    unset_set_rm_properties(rm, solver_flag, sigma_flag)
    return risk
end

"""
```
factor_risk_contribution(port::AbstractPortfolio; X::AbstractMatrix = port.returns,
                         F::AbstractMatrix = port.f_returns,
                         type::Symbol = isa(port, Portfolio) ? :Trad : :HRP,
                         rm::AbstractRiskMeasure = SD(), delta::Real = 1e-6)
```

Compute the factor risk contribution for an [`AbstractRiskMeasure`](@ref) for a portfolio.

# Inputs

  - `port`: portfolio.
  - `w`: `Na×1` vector of asset weights.
  - `X`: `T×Na` matrix of asset returns.
  - `F`: `T×Nf` matrix of factor returns.
  - `type`: optimisation type used to retrieve the weights vector from `port.optimal[type]`.
  - `rm`: risk measure.
  - `delta`: small displacement used for computing the marginal risk and equal risk measure [`Equal`](@ref).

# Outputs

  - `rc_f`: `Nf×1` vector of risk contribution per factor.
"""
function factor_risk_contribution(port::AbstractPortfolio; X::AbstractMatrix = port.returns,
                                  F::AbstractMatrix = port.f_returns,
                                  type::Symbol = isa(port, Portfolio) ? :Trad : :HRP,
                                  rm::AbstractRiskMeasure = SD(), delta::Real = 1e-6)
    solver_flag, sigma_flag = set_rm_properties(rm, port.solvers, port.cov)
    risk = factor_risk_contribution(rm, port.optimal[type].weights; X = X,
                                    assets = port.assets, F = F, f_assets = port.f_assets,
                                    B = port.loadings,
                                    regression_type = port.regression_type, V = port.V,
                                    SV = port.SV, delta = delta)
    unset_set_rm_properties(rm, solver_flag, sigma_flag)
    return risk
end

"""
```
sharpe_ratio(port::AbstractPortfolio; X::AbstractMatrix = port.returns,
             mu::AbstractVector = port.mu,
             type::Symbol = isa(port, Portfolio) ? :Trad : :HRP,
             rm::AbstractRiskMeasure = SD(), delta::Real = 1e-6, rf::Real = 0.0,
             kelly::Bool = false)
```

Compute the risk-adjusted return ratio for an [`AbstractRiskMeasure`](@ref) for a portfolio.

# Inputs

  - `port`: portfolio.

  - `X`: `T×N` matrix of asset returns.
  - `mu`: `N×1` vector of expected returns.
  - `type`: optimisation type used to retrieve the weights vector from `port.optimal[type]`.
  - `rm`: risk measure.
  - `delta`: small displacement used for computing the [`Equal`](@ref) risk measure.
  - `rf`: risk free rate.
  - `kelly`:

        + if `true`: use the kelly return.
        + else: use the arithmetic return.
"""
function sharpe_ratio(port::AbstractPortfolio; X::AbstractMatrix = port.returns,
                      mu::AbstractVector = port.mu,
                      type::Symbol = isa(port, Portfolio) ? :Trad : :HRP,
                      rm::AbstractRiskMeasure = SD(), delta::Real = 1e-6, rf::Real = 0.0,
                      kelly::Bool = false)
    solver_flag, sigma_flag = set_rm_properties(rm, port.solvers, port.cov)
    risk = sharpe_ratio(rm, port.optimal[type].weights; mu = mu, X = X, V = port.V,
                        SV = port.SV, delta = delta, rf = rf, kelly = kelly)
    unset_set_rm_properties(rm, solver_flag, sigma_flag)
    return risk
end
