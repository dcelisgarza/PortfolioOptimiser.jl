# Specifics

This section details types, structures, and constants that play a supporting role in defining specific risk measures.

## [`RMOWA`](@ref)

Certain risk measures make use of Ordered Weight Array formulations in optimisations which use [`JuMP`](https://github.com/jump-dev/JuMP.jl) models. [`PortfolioOptimiser`](https://github.com/dcelisgarza/PortfolioOptimiser.jl/) implements two formulations.

  - An exact but expensive formulation.
  - An approximate, tunable one based on [PowerCone](https://jump.dev/JuMP.jl/stable/tutorials/conic/tips_and_tricks/#PowerCone) constraints.

We therefore provide a structure for dispatching the exact method, and a structure for tuning and dispatching the approximate one.

```@docs
PortfolioOptimiser.OWAFormulation
OWAExact
OWAApprox
```

## [`Variance`](@ref) & [`SVariance`](@ref)

```@docs
PortfolioOptimiser.VarianceFormulation
Quad
SOC
RSOC
```

## [`TrackingRM`](@ref)

```@docs
PortfolioOptimiser.TrackingErr
NoTracking
TrackRet
TrackWeight
```

## [`TurnoverRM`](@ref)

```@docs
PortfolioOptimiser.AbstractTR
NoTR
TR
```

## [`RMMu`](@ref)

```@docs
PortfolioOptimiser.calc_ret_mu
```

## [`RMTarget`](@ref)

```@docs
PortfolioOptimiser.calc_target_ret_mu
```

## [`RMSolvers`](@ref)

```@docs
PortfolioOptimiser.ERM
PortfolioOptimiser.RRM
```
