# [`HCRiskMeasure`](@ref)

This contains the documentation for all available [`HCRiskMeasure`](@ref) subtypes.

## Dispersion [`HCRiskMeasure`](@ref)

These measure the spread of the returns distribution.

```@docs
TCM
FTCM
```

## Downside Dispersion [`HCRiskMeasure`](@ref)

These measure how far the returns deviate from the mean in the negative direction.

```@docs
SVariance
TLPM
FTLPM
Kurtosis
SKurtosis
```

## Downside [`HCRiskMeasure`](@ref)

These measure different aspects of the tail (negative side) of the returns distribution.

```@docs
VaR
```

## Drawdown [`HCRiskMeasure`](@ref)

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

## Equal Risk Contribution [`HCRiskMeasure`](@ref)

This assumes the risk is equally distributed among the variables.

```@docs
Equal
```
