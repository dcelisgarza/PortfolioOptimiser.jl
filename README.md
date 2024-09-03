# PortfolioOptimiser

| Category | Badge                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
|:--------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| Docs     | [![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://dcelisgarza.github.io/PortfolioOptimiser.jl/stable) [![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://dcelisgarza.github.io/PortfolioOptimiser.jl/dev)                                                                                                                                                                                                                                                                                                                                                    |
| Examples | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/dcelisgarza/PortfolioOptimiser.jl/HEAD?labpath=%2Fexamples)                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| CI       | [![Tests](https://github.com/dcelisgarza/PortfolioOptimiser.jl/actions/workflows/Tests.yml/badge.svg)](https://github.com/dcelisgarza/PortfolioOptimiser.jl/actions/workflows/Tests.yml) [![Documentation](https://github.com/dcelisgarza/PortfolioOptimiser.jl/actions/workflows/Documentation.yml/badge.svg)](https://github.com/dcelisgarza/PortfolioOptimiser.jl/actions/workflows/Documentation.yml) [![Aqua](https://github.com/dcelisgarza/PortfolioOptimiser.jl/actions/workflows/Aqua.yml/badge.svg)](https://github.com/dcelisgarza/PortfolioOptimiser.jl/actions/workflows/Aqua.yml) |
| Coverage | [![Codecov](https://codecov.io/gh/dcelisgarza/PortfolioOptimiser.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/dcelisgarza/PortfolioOptimiser.jl) [![Coveralls](https://coveralls.io/repos/github/dcelisgarza/PortfolioOptimiser.jl/badge.svg?branch=main)](https://coveralls.io/github/dcelisgarza/PortfolioOptimiser.jl?branch=main)                                                                                                                                                                                                                                                 |

## Description

PortfolioOptimiser is a library for portfolio optimisation. It was written with composability and extensibility in mind. It offers a broad range of functionality out of the box.

There are two main types which define what kinds of portfolio can be optimised.

### Portfolio optimisation

These types of optimisations act on instances of [`Portfolio`](@ref).

#### Traditional, [`Trad`](@ref)

This type of optimisation is the traditional efficient frontier optimisation with optional constraints.

##### Classes, [`PortClass`](@ref)

  - Classic, [`Classic`](@ref).
  - Factor model, [`FM`](@ref).
  - Black Litterman, [`BL`](@ref).
  - Black Litterman Factor model, [`BLFM`](@ref).

##### Expected returns

  - Arithmetic returns, [`NoKelly`](@ref).
  - Approximate logarithmic mean returns, [`AKelly`](@ref).
  - Exact logarithmic mean returns, [`EKelly`](@ref).

##### Objective functions

  - Minimum risk, [`MinRisk`](@ref).
  - Maximum utility, [`Utility`](@ref).
  - Maximum risk adjusted return ratio, [`Sharpe`](@ref).
  - Maximum return, [`MaxRet`](@ref).

##### Constraints

  - Maximum expected risk constraints.
  - Minimum expected return constraint.
  - Linear weight constraints.
  - Connected asset centrality constraints.
  - Asset network constraints.
  - Leverage constraints.
  - Maximum number of assets constraint.
  - Minimum number of effective assets constraint.
  - Tracking error (weights or returns) constraint.
  - Turnover constraint.
  - Rebalancing penalty.

##### Risk measures

###### Dispersion

  - Full dispersion.
    
      + Standard deviation, [`SD`](@ref).
      + Mean absolute deviation (MAD), [`MAD`](@ref).
      + Square root kurtosis, [`Kurt`](@ref).
      + Range, [`RG`](@ref).
      + Conditional value at risk range (CVaR range), [`RCVaR`](@ref).
      + Tail Gini range, [`RTG`](@ref).
      + Gini mean difference (GMD), [`GMD`](@ref).
      + Quadratic negative skewness, [`Skew`](@ref).
      + Brownian distance variance (dVar), [`dVar`](@ref).
  - Downside dispersion.
    
      + Semi standard deviation, [`SSD`](@ref).
      + First lower partial moment (Omega ratio), [`FLPM`](@ref).
      + Second lower partial moment (Sortino ratio), [`SLPM`](@ref).
      + Square root semi kurtosis, [`SKurt`](@ref).
      + Quadratic negative semi skewness, [`SSkew`](@ref).

###### Downside

  - Worst case realisation (Minimax), [`WR`](@ref).
  - Conditional value at risk (CVaR), [`CVaR`](@ref).
  - Entropic value at risk (EVaR), [`EVaR`](@ref).
  - Relativistic value at risk (RLVaR), [`RLVaR`](@ref).
  - Tail Gini, [`TG`](@ref).

###### Drawdown

  - Maximum drawdown (Calmar ratio) for uncompounded cumulative returns, [`MDD`](@ref).
  - Average drawdown for uncompounded cumulative returns, [`ADD`](@ref).
  - Ulcer index for uncompounded cumulative returns, [`UCI`](@ref).
  - Conditional drawdown at risk for uncompounded cumulative returns (CDaR), [`CDaR`](@ref).
  - Entropic drawdown at risk for uncompounded cumulative returns (EDaR), [`EDaR`](@ref).
  - Relativistic drawdown at risk for uncompounded cumulative returns (RLDaR), [`RLDaR`](@ref).

#### Worst case mean variance, [`WC`](@ref)

This type of optimisation requires worst case sets for the covariance and expected returns. The optimisation uses these sets to perform a mean variance optimisation.

##### Constraints

  - Maximum expected worst case standard deviation constraint.
  - Minimum expected worst case return constraint.
  - Linear weight constraints.
  - Connected asset centrality constraints.
  - Asset network constraints.
  - Leverage constraints.
  - Maximum number of assets constraint.
  - Minimum number of effective assets constraint.
  - Tracking error (weights or returns) constraint.
  - Turnover constraint.
  - Rebalancing penalty.

#### Risk parity, [`RP`](@ref)

This type of optimisation requires a risk budget per asset or factor. The optimisation attempts to minimise the difference between the risk budget and risk contribution of the asset or factor in the optimised portfolio.

##### Classes, [`PortClass`](@ref)

  - Classic, [`Classic`](@ref).
  - Factor model, [`FM`](@ref).
  - Factor risk contribution, [`FC`](@ref).

##### Constraints

  - Minimum expected return constraint.
  - Linear weight constraints.

##### Risk measures

###### Dispersion

  - Full dispersion.
    
      + Standard deviation, [`SD`](@ref).
      + Mean absolute deviation (MAD), [`MAD`](@ref).
      + Square root kurtosis, [`Kurt`](@ref).
      + Range, [`RG`](@ref).
      + Conditional value at risk range (CVaR range), [`RCVaR`](@ref).
      + Tail Gini range, [`RTG`](@ref).
      + Gini mean difference (GMD), [`GMD`](@ref).
      + Quadratic negative skewness, [`Skew`](@ref).
      + Brownian distance variance (dVar), [`dVar`](@ref).
  - Downside dispersion.
    
      + Semi standard deviation, [`SSD`](@ref).
      + First lower partial moment (Omega ratio), [`FLPM`](@ref).
      + Second lower partial moment (Sortino ratio), [`SLPM`](@ref).
      + Square root semi kurtosis, [`SKurt`](@ref).
      + Quadratic negative semi skewness, [`SSkew`](@ref).

###### Downside

  - Worst case realisation (Minimax), [`WR`](@ref).
  - Conditional value at risk (CVaR), [`CVaR`](@ref).
  - Entropic value at risk (EVaR), [`EVaR`](@ref).
  - Relativistic value at risk (RLVaR), [`RLVaR`](@ref).
  - Tail Gini, [`TG`](@ref).

###### Drawdown

  - Maximum drawdown (Calmar ratio) for uncompounded cumulative returns, [`MDD`](@ref).
  - Average drawdown for uncompounded cumulative returns, [`ADD`](@ref).
  - Ulcer index for uncompounded cumulative returns, [`UCI`](@ref).
  - Conditional drawdown at risk for uncompounded cumulative returns (CDaR), [`CDaR`](@ref).
  - Entropic drawdown at risk for uncompounded cumulative returns (EDaR), [`EDaR`](@ref).
  - Relativistic drawdown at risk for uncompounded cumulative returns (RLDaR), [`RLDaR`](@ref).

#### Relaxed risk parity mean variance, [`RRP`](@ref)

This type of optimisation requires a risk budget per asset. The optimisation attempts to minimise the difference between the risk budget and relaxed formulation of the standard deviation risk measure.

##### Classes, [`PortClass`](@ref)

  - Classic, [`Classic`](@ref).
  - Factor model, [`FM`](@ref).

##### Constraints

  - Minimum expected return constraint.
  - Linear weight constraints.

#### Near Optimal Centering, [`NOC`](@ref)

##### Classes, [`PortClass`](@ref)

  - Classic, [`Classic`](@ref).
  - Factor model, [`FM`](@ref).
  - Black Litterman, [`BL`](@ref).
  - Black Litterman Factor model, [`BLFM`](@ref).

##### Expected returns

  - Arithmetic returns, [`NoKelly`](@ref).
  - Approximate logarithmic mean returns, [`AKelly`](@ref).
  - Exact logarithmic mean returns, [`EKelly`](@ref).

##### Constraints

  - Maximum expected risk constraints.
  - Minimum expected return constraint.

##### Risk measures

###### Dispersion

  - Full dispersion.
    
      + Standard deviation, [`SD`](@ref).
      + Mean absolute deviation (MAD), [`MAD`](@ref).
      + Square root kurtosis, [`Kurt`](@ref).
      + Range, [`RG`](@ref).
      + Conditional value at risk range (CVaR range), [`RCVaR`](@ref).
      + Tail Gini range, [`RTG`](@ref).
      + Gini mean difference (GMD), [`GMD`](@ref).
      + Quadratic negative skewness, [`Skew`](@ref).
      + Brownian distance variance (dVar), [`dVar`](@ref).
  - Downside dispersion.
    
      + Semi standard deviation, [`SSD`](@ref).
      + First lower partial moment (Omega ratio), [`FLPM`](@ref).
      + Second lower partial moment (Sortino ratio), [`SLPM`](@ref).
      + Square root semi kurtosis, [`SKurt`](@ref).
      + Quadratic negative semi skewness, [`SSkew`](@ref).

###### Downside

  - Worst case realisation (Minimax), [`WR`](@ref).
  - Conditional value at risk (CVaR), [`CVaR`](@ref).
  - Entropic value at risk (EVaR), [`EVaR`](@ref).
  - Relativistic value at risk (RLVaR), [`RLVaR`](@ref).
  - Tail Gini, [`TG`](@ref).

###### Drawdown

  - Maximum drawdown (Calmar ratio) for uncompounded cumulative returns, [`MDD`](@ref).
  - Average drawdown for uncompounded cumulative returns, [`ADD`](@ref).
  - Ulcer index for uncompounded cumulative returns, [`UCI`](@ref).
  - Conditional drawdown at risk for uncompounded cumulative returns (CDaR), [`CDaR`](@ref).
  - Entropic drawdown at risk for uncompounded cumulative returns (EDaR), [`EDaR`](@ref).
  - Relativistic drawdown at risk for uncompounded cumulative returns (RLDaR), [`RLDaR`](@ref).

### Hierarchical portfolio optimisation

These types of optimisations act on instances of [`HCPortfolio`](@ref).

#### Hierarchical Risk Parity, [`HRP`](@ref)

#### Hierarchical Equal Risk Contribution, [`HERC`](@ref)

#### Nested Clustered Optimisation, [`NOC`](@ref)
