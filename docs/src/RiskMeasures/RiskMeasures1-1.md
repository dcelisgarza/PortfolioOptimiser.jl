# Generalities

This section details general types, structures, and constants that play a supporting role in defining risk measures.

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

## Constants

It is useful to group risk measures by their properties as well as their types. By defining constants which group various risk measure types with certain characteristics, we can facilitate their use in the many different optimisation types offered by [`PortfolioOptimiser`](https://github.com/dcelisgarza/PortfolioOptimiser.jl/). This makes adding new features easier and simplifies the definition of customised behaviour.

```@docs
PortfolioOptimiser.RMSolvers
PortfolioOptimiser.RMSigma
PortfolioOptimiser.RMSkew
PortfolioOptimiser.RMOWA
PortfolioOptimiser.RMMu
PortfolioOptimiser.RMTarget
```

## Settings

Portfolio Optimisation has typically limited itself to using a single risk measure. However, some risk measures can/must use internal parameters in order to compute the risk. There is nothing preventing us from using multiple measures simultaneously, or even multiple instances of the same risk measure with different hyperparameters. This can be achieved via multiple objective optimisation (not currently supported, but may in the future), or a scalarisation procedure [`PortfolioOptimiser.AbstractScalarisation`](@ref). Each instance's contribution to the overall risk expression is tunable via a weight parameter (called `scale` for disambiguation purposes).

For risk measures that can be used in optimisations using [`JuMP`](https://github.com/jump-dev/JuMP.jl) models, we must also provide a way to define their upper bound, and whether or not they should be included in the risk expression or used only for setting a risk upper bound.

For these purposes, we have two special structures that need to be included in risk measures that are to be used in optimisations. The choice of which one to use depends on what optimisation types the risk measure is meant to be compatible with.

```@docs
PortfolioOptimiser.AbstractRMSettings
RMSettings
HCRMSettings
```
