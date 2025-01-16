# Risk value

Despite the fact that [`HCRiskMeasure`](@ref) are only compatible with , it is possible to compute the value of every risk measure for a given returns matrix and vector of weights.

There are similarly named higher level functions that operate at the level of [`PortfolioOptimiser.AbstractPortfolio`](@ref).

## Dispersion

These measure the spread of the returns distribution.

### Full dispersion

These measure how far the returns deviate from the mean in both the positive and negative directions.

```@docs
calc_risk(::Variance, ::AbstractVector)
PortfolioOptimiser._Variance
calc_risk(::SD, ::AbstractVector)
PortfolioOptimiser._SD
calc_risk(::MAD, ::AbstractVector)
PortfolioOptimiser._MAD
calc_risk(::Kurt, ::AbstractVector)
PortfolioOptimiser._Kurt
calc_risk(::RG, ::AbstractVector)
PortfolioOptimiser._RG
calc_risk(::CVaRRG, ::AbstractVector)
PortfolioOptimiser._CVaRRG
calc_risk(::TGRG, ::AbstractVector)
PortfolioOptimiser._TGRG
calc_risk(::GMD, ::AbstractVector)
PortfolioOptimiser._GMD
calc_risk(::PortfolioOptimiser.RMSkew, w::AbstractVector)
PortfolioOptimiser._Skew
calc_risk(::BDVariance, ::AbstractVector)
PortfolioOptimiser._BDVariance
```

### Downside dispersion

These measure how far the returns deviate from the mean in the negative direction.

```@docs
calc_risk(::SVariance, ::AbstractVector)
PortfolioOptimiser._SVariance
calc_risk(::SSD, ::AbstractVector)
PortfolioOptimiser._SSD
calc_risk(::FLPM, ::AbstractVector)
PortfolioOptimiser._FLPM
calc_risk(::SLPM, ::AbstractVector)
PortfolioOptimiser._SLPM
calc_risk(::SKurt, ::AbstractVector)
PortfolioOptimiser._SKurt
```

## Downside

These measure different aspects of the tail (negative side) of the returns distribution.

```@docs
calc_risk(::WR, ::AbstractVector)
PortfolioOptimiser._WR
calc_risk(::VaR, ::AbstractVector)
PortfolioOptimiser._VaR
calc_risk(::Union{CVaR, DRCVaR}, ::AbstractVector)
PortfolioOptimiser._CVaR
PortfolioOptimiser.ERM
calc_risk(::EVaR, ::AbstractVector)
PortfolioOptimiser._EVaR
PortfolioOptimiser.RRM
calc_risk(::RLVaR, ::AbstractVector)
PortfolioOptimiser._RLVaR
calc_risk(::TG, ::AbstractVector)
PortfolioOptimiser._TG
```

```@docs
calc_risk(::DaR, ::AbstractVector)
PortfolioOptimiser._DaR
calc_risk(::MDD, ::AbstractVector)
PortfolioOptimiser._MDD
calc_risk(::ADD, ::AbstractVector)
PortfolioOptimiser._ADD
calc_risk(::CDaR, ::AbstractVector)
PortfolioOptimiser._CDaR
calc_risk(::UCI, ::AbstractVector)
PortfolioOptimiser._UCI
calc_risk(::EDaR, ::AbstractVector)
PortfolioOptimiser._EDaR
calc_risk(::RLDaR, ::AbstractVector)
PortfolioOptimiser._RLDaR
calc_risk(::DaR_r, ::AbstractVector)
PortfolioOptimiser._DaR_r
calc_risk(::MDD_r, ::AbstractVector)
PortfolioOptimiser._MDD_r
calc_risk(::ADD_r, ::AbstractVector)
PortfolioOptimiser._ADD_r
calc_risk(::CDaR_r, ::AbstractVector)
PortfolioOptimiser._CDaR_r
calc_risk(::UCI_r, ::AbstractVector)
PortfolioOptimiser._UCI_r
calc_risk(::EDaR_r, ::AbstractVector)
PortfolioOptimiser._EDaR_r
calc_risk(::RLDaR_r, ::AbstractVector)
PortfolioOptimiser._RLDaR_r
```

## Linear moments (L-moments)

This is used to measure linear moments of the returns distribution.

```@docs
calc_risk(::OWA, ::AbstractVector)
PortfolioOptimiser._OWA
```

## Equal risk contribution

This assumes the risk is equally distributed among the variables.

```@docs
calc_risk(::Equal, ::AbstractVector)
```
