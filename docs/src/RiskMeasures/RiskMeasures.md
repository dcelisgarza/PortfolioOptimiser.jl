# Measures

Risk measures are the backbone of portfolio optimisation. They allow for the quantification of risk in different ways. This section describes the various risk measures included in [`PortfolioOptimiser`](https://github.com/dcelisgarza/PortfolioOptimiser.jl).

We go into detail on their use, characteristics, compatibility, formulations, functionality, and implementation.

This is meant to be not only an API reference but a tutorial and how-to guide on creating custom functionality.

## Settings

### Upper bounds and multiple risk measures simultaneously

Portfolio Optimisation has typically limited itself to using a single risk measure. However, some risk measures can/must use internal parameters in order to compute the risk. There is nothing preventing us from using multiple measures simultaneously, or even multiple instances of the same risk measure with different hyperparameters. This can be achieved via multiple objective optimisation (not currently supported, but may in the future), or a scalarisation procedure [`PortfolioOptimiser.AbstractScalarisation`](@ref). Each instance's contribution to the overall risk expression is tunable via a weight parameter (called `scale` for disambiguation purposes).

For risk measures that can be used in optimisations using [`JuMP`](https://github.com/jump-dev/JuMP.jl) models, we must also provide a way to define their upper bound, and whether or not they should be included in the risk expression or used only for setting a risk upper bound.

For these purposes, we have two special structures that need to be included in risk measures that are to be used in optimisations. The choice of which one to use depends on what optimisation types the risk measure is meant to be compatible with.

```@docs
PortfolioOptimiser.AbstractRMSettings
RMSettings
HCRMSettings
```

### Ordered Weight Array settings

Certain risk measures make use of Ordered Weight Array formulations in optimisations which use [`JuMP`](https://github.com/jump-dev/JuMP.jl) models. [`PortfolioOptimiser`](https://github.com/dcelisgarza/PortfolioOptimiser.jl/) implements two formulations.

  - An exact but expensive formulation.
  - An approximate, tunable one based on [3D Power Cones](https://jump.dev/JuMP.jl/stable/tutorials/conic/tips_and_tricks/#PowerCone).

We therefore provide a structure for dispatching the exact method, and a structure for tuning and dispatching the approximate one.

```@docs
PortfolioOptimiser.OWAFormulation
OWAExact
OWAApprox
```

## Abstract types

[`PortfolioOptimiser`](https://github.com/dcelisgarza/PortfolioOptimiser.jl/)'s risk measures are implemented using `Julia`'s type hierarchy. This makes it easy to add new ones by implementing the relevant types and methods.

By properly subtyping and following the implementation instructions of each abstract type, it's possible to define risk measures that will be entirely compatible with current functionality, without the need for extending method definitions and/or subtyping any of the myriad of abstract types that allow users to define custom behaviour within the library.

```@docs
PortfolioOptimiser.AbstractRiskMeasure
RiskMeasure
HCRiskMeasure
NoOptRiskMeasure
RiskMeasureSigma
RiskMeasureMu
HCRiskMeasureMu
NoOptRiskMeasureMu
RiskMeasureTarget       
HCRiskMeasureTarget
RiskMeasureSolvers
HCRiskMeasureSolvers
RiskMeasureOWA
RiskMeasureSkew
```

It is useful to group risk measures by their properties as well as their types. By defining constants which group various risk measure types with certain characteristics, we can facilitate their use in the many different optimisation types offered by [`PortfolioOptimiser`](https://github.com/dcelisgarza/PortfolioOptimiser.jl/). This makes adding new features easier and simplifies the definition of customised behaviour.

```@docs
PortfolioOptimiser.RMSolvers
PortfolioOptimiser.RMSigma
PortfolioOptimiser.RMSkew
PortfolioOptimiser.RMOWA
PortfolioOptimiser.RMMu
PortfolioOptimiser.RMTarget
```

## Utility functions

Some risk measures require the computation of certain statistics, these are performed by the following functions.

```@docs
PortfolioOptimiser.calc_ret_mu
PortfolioOptimiser.calc_target_ret_mu
```

## Dispersion risk measures

These measure the spread of the returns distribution.

### Full dispersion risk measures

These measure how far the returns deviate from the mean in both the positive and negative directions.

```@docs
PortfolioOptimiser.VarianceFormulation
Quad
SOC
Variance
SD
MAD
Kurt
RG
CVaRRG
TGRG
GMD
Skew
BDVariance
WCVariance
TCM
FTCM
```

## Tracking and turnover risk measures

```@docs
PortfolioOptimiser.AbstractTR
PortfolioOptimiser.TrackingErr
TrackRet
TrackWeight
NoTracking
TR
NoTR
```

## Downside dispersion risk measures

These measure how far the returns deviate from the mean in the negative direction.

```@docs
SSD
FLPM
SLPM
TLPM
FTLPM
SKurt
SSkew
Kurtosis
SKurtosis
```

## Worst case variance

```@docs
PortfolioOptimiser.WorstCaseSet
Box
Ellipse
NoWC
```

## Downside risk measures

These measure different aspects of the tail (negative side) of the returns distribution.

```@docs
WR
CVaR
EVaR
RLVaR
TG
```

## Drawdown risk measures

These measure the drops in portfolio value from local maxima to subsequent local minima.

```@docs
MDD
ADD
UCI
CDaR
EDaR
RLDaR
```

## Linear moments (L-moments)

This is used to measure linear moments of the returns distribution.

```@docs
OWA
```

## Hierarchical risk measures

These risk measures are compatible with . Different risk measures account for different aspects of the returns.

## Dispersion hierarchical risk measures

These measure the characteristics of the returns distribution.

## Downside dispersion hierarchical risk measures

These measure how far the returns deviate from the mean in the negative direction.

```@docs
SVariance
```

## Downside hierarchical risk measures

These measure different aspects of the tail (negative side) of the returns distribution.

```@docs
VaR
```

## Drawdown hierarchical risk measures

These measure the drops in portfolio value from local maxima to subsequent local minima.

```@docs
DaR
DaR_r
MDD_r
ADD_r
UCI_r
CDaR_r
EDaR_r
RLDaR_r
```

## Equal risk contribution

This assumes the risk is equally distributed among the variables.

```@docs
Equal
```
