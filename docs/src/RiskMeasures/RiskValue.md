# Risk value

Despite the fact that [`HCRiskMeasure`](@ref) are only compatible with [`HCPortfolio`](@ref), it is possible to compute the value of every risk measure for a given returns matrix and vector of weights.

There are similarly named higher level functions that operate at the level of [`PortfolioOptimiser.AbstractPortfolio`](@ref).

```@docs
calc_risk(::Variance, ::AbstractVector)
PortfolioOptimiser._Variance
calc_risk(::SVariance, ::AbstractVector)
PortfolioOptimiser._SVariance
calc_risk(::SD, ::AbstractVector)
PortfolioOptimiser._SD
calc_risk(::MAD, ::AbstractVector)
PortfolioOptimiser._MAD
calc_risk(::SSD, ::AbstractVector)
PortfolioOptimiser._SSD
calc_risk(::FLPM, ::AbstractVector)
PortfolioOptimiser._FLPM
calc_risk(::SLPM, ::AbstractVector)
PortfolioOptimiser._SLPM
calc_risk(::WR, ::AbstractVector)
PortfolioOptimiser._WR
calc_risk(::VaR, ::AbstractVector)
PortfolioOptimiser._VaR
calc_risk(::CVaR, ::AbstractVector)
PortfolioOptimiser._CVaR
PortfolioOptimiser.ERM
calc_risk(::EVaR, ::AbstractVector)
PortfolioOptimiser._EVaR
PortfolioOptimiser.RRM
calc_risk(::RLVaR, ::AbstractVector)
PortfolioOptimiser._RLVaR
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
calc_risk(::Kurt, ::AbstractVector)
PortfolioOptimiser._Kurt
calc_risk(::SKurt, ::AbstractVector)
PortfolioOptimiser._SKurt
calc_risk(::GMD, ::AbstractVector)
PortfolioOptimiser._GMD
calc_risk(::RG, ::AbstractVector)
PortfolioOptimiser._RG
calc_risk(::CVaRRG, ::AbstractVector)
PortfolioOptimiser._CVaRRG
calc_risk(::TG, ::AbstractVector)
PortfolioOptimiser._TG
calc_risk(::TGRG, ::AbstractVector)
PortfolioOptimiser._TGRG
calc_risk(::OWA, ::AbstractVector)
PortfolioOptimiser._OWA
calc_risk(::BDVariance, ::AbstractVector)
PortfolioOptimiser._BDVariance
calc_risk(::Skew, ::AbstractVector)
calc_risk(::SSkew, ::AbstractVector)
PortfolioOptimiser._Skew
calc_risk(::Equal, ::AbstractVector)
```
