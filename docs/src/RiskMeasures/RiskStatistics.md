# Risk statistics

It is possible to compute risk-derived statistics given a [`RiskMeasure`](@ref)/[`HCRiskMeasure`](@ref), a vector of weights, together with other relevant data.

There are similarly named higher level functions that operate at the level of [`PortfolioOptimiser.AbstractPortfolio`](@ref).

```@docs
risk_bounds(::PortfolioOptimiser.AbstractRiskMeasure, ::AbstractVector, ::AbstractVector)
risk_contribution(::PortfolioOptimiser.AbstractRiskMeasure, ::AbstractVector)
factor_risk_contribution(::PortfolioOptimiser.AbstractRiskMeasure, ::AbstractVector)
sharpe_ratio(::PortfolioOptimiser.AbstractRiskMeasure, ::AbstractVector)
```
