# [`HCRiskMeasure`](@ref)

This contains the documentation for all available [`HCRiskMeasure`](@ref) subtypes.

## Dispersion

These measure the spread of the returns distribution.

```@docs
FTCM
```

## Downside dispersion

These measure how far the returns deviate from the mean in the negative direction.

```@docs
SVariance
TLPM
FTLPM
Kurtosis
SKurtosis
```

## Downside

These measure different aspects of the tail (negative side) of the returns distribution.

```@docs
VaR
```

## Drawdown

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

## Equal Risk Contribution

This assumes the risk is equally distributed among the variables.

```@docs
Equal
```
