# [`RiskMeasure`](@ref)

This contains the documentation for all available [`RiskMeasure`](@ref) subtypes.

## Dispersion [`RiskMeasure`](@ref)

These measure the spread of the returns distribution.

```@docs
PortfolioOptimiser.VarianceFormulation
Quad
SOC
RSOC
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
```

## Downside Dispersion [`RiskMeasure`](@ref)

These measure how far the returns deviate from the mean in the negative direction.

```@docs
SSD
FLPM
SLPM
SKurt
SSkew
```

## Downside [`RiskMeasure`](@ref)

These measure different aspects of the tail (negative side) of the returns distribution.

```@docs
WR
CVaR
EVaR
RLVaR
TG
```

## Drawdown [`RiskMeasure`](@ref)

These measure the drops in portfolio value from local maxima to subsequent local minima.

```@docs
MDD
ADD
UCI
CDaR
EDaR
RLDaR
```

## Linear moments (L-moments) [`RiskMeasure`](@ref)

These measure different combinations of linear moments of the returns distribution.

```@docs
OWA
```

## Tracking and Turnover [`RiskMeasure`](@ref)

These measure how far a portfolio deviates from a benchmark.

```@docs
TrackingRM
TurnoverRM
```
