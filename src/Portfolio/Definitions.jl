"""
Abstract type for portfolios. Concrete portfolios subtype this see [`Portfolio`](@ref) and [`HCPortfolio`](@ref).
```julia
abstract type AbstractPortfolio end
```
"""
abstract type AbstractPortfolio end

"""
Available risk measures for `type = :trad` optimisations of [`Portfolio`](@ref).
    - `:SD` = standard deviation
    - `:MAD` = max absolute deviation
    - `:SSD` = semi standard deviation
    - `:FLPM` = first lower partial moment (omega ratio)
    - `:SLPM` = second lower partial moment (sortino ratio)
    - `:WR` = worst realisation
    - `:CVaR` = conditional value at risk
    - `:EVaR` = entropic value at risk
    - `:RVaR` = relativistic value at risk
    - `:MDD` = maximum drawdown
    - `:ADD` = average drawdown
    - `:CDaR` = conditional drawdown at risk
    - `:UCI` = ulcer index
    - `:EDaR` = entropic drawdown at risk
    - `:RDaR` = relativistic drawdown at risk
    - `:Kurt` = kurtosis
    - `:SKurt` = semi-kurtosis
    - `:GMD` = gini mean difference
    - `:RG` = range of returns
    - `:RCVaR` = range of conditional value at risk
    - `:TG` = tail gini
    - `:RTG` = range of tail gini
    - `:OWA` = ordered weight array (generic OWA weights)
```
const RiskMeasures = (
    :SD,    # _mv
    :MAD,   # _mad
    :SSD,   # _mad
    :FLPM,  # _lpm
    :SLPM,  # _lpm
    :WR,    # _wr
    :CVaR,  # _var
    :EVaR,  # _var
    :RVaR,  # _var
    :MDD,   # _dar
    :ADD,   # _dar
    :CDaR,  # _dar
    :UCI,   # _dar
    :EDaR,  # _dar
    :RDaR,  # _dar
    :Kurt,   # _krt
    :SKurt,  # _krt
    :GMD,   # _owa
    :RG,    # _owa
    :RCVaR, # _owa
    :TG,    # _owa
    :RTG,   # _owa
    :OWA,   # _owa
)
```
"""
const RiskMeasures = (
    :SD,    # _mv
    :MAD,   # _mad
    :SSD,   # _mad
    :FLPM,  # _lpm
    :SLPM,  # _lpm
    :WR,    # _wr
    :CVaR,  # _var
    :EVaR,  # _var
    :RVaR,  # _var
    :MDD,   # _dar
    :ADD,   # _dar
    :CDaR,  # _dar
    :UCI,   # _dar
    :EDaR,  # _dar
    :RDaR,  # _dar
    :Kurt,  # _krt
    :SKurt, # _krt
    :GMD,   # _owa
    :RG,    # _owa
    :RCVaR, # _owa
    :TG,    # _owa
    :RTG,   # _owa
    :OWA,   # _owa
)

"""
Available types of Kelly returns for [`Portfolio`](@ref).
```
const KellyRet = (:none, :approx, :exact)
```
"""
const KellyRet = (:none, :approx, :exact)

"""
Available kinds of tracking errors for [`Portfolio`](@ref).
```
const TrackingErrKinds = (:weights, :returns)
```
"""
const TrackingErrKinds = (:weights, :returns)

"""
Available objective functions for [`Portfolio`](@ref).
```
const ObjFuncs = (:min_risk, :utility, :sharpe, :max_ret)
```
"""
const ObjFuncs = (:min_risk, :utility, :sharpe, :max_ret)

"""
Valid JuMP terminations after optimising an instance of [`Portfolio`](@ref).
```
const ValidTermination =
    (MOI.OPTIMAL, MOI.ALMOST_OPTIMAL, MOI.LOCALLY_SOLVED, MOI.ALMOST_LOCALLY_SOLVED)
```
"""
const ValidTermination =
    (MOI.OPTIMAL, MOI.ALMOST_OPTIMAL, MOI.LOCALLY_SOLVED, MOI.ALMOST_LOCALLY_SOLVED)

"""
Available classes for [`Portfolio`](@ref).
```
const PortClasses = (:classic,)
```
"""
const PortClasses = (:classic,)

"""
Available optimisation types for [`Portfolio`](@ref).
```
const PortTypes = (:trad, :rp, :rrp, :OWA, :wc)
```
"""
const PortTypes = (:trad, :rp, :rrp, :OWA, :wc)

"""
Available versions of relaxed risk parity optimisations of [`Portfolio`](@ref).
```
const RRPVersions = (:none, :reg, :reg_pen)
```
"""
const RRPVersions = (:none, :reg, :reg_pen)

"""
Types of risk parity constraints for [`rp_constraints`](@ref).
```
const RPConstraintTypes = (:assets, :classes)
```
"""
const RPConstraintTypes = (:assets, :classes)

"""
Types of uncertainty sets for worst case optimisations of [`Portfolio`](@ref).
```
const UncertaintyTypes = (:none, :box, :ellipse)
```
"""
const UncertaintyTypes = (:none, :box, :ellipse)

"""
Bootstrap for worst case optimisations.
```
const KindBootstrap = (:stationary, :circular, :moving)
```
"""
const KindBootstrap = (:stationary, :circular, :moving)
"""
Ellipse and box types for worst case optimisations.
```
const EllipseTypes = (:stationary, :circular, :moving, :normal)
const BoxTypes = (EllipseTypes..., :delta)
```
"""
const EllipseTypes = (:stationary, :circular, :moving, :normal)
const BoxTypes = (EllipseTypes..., :delta)

# Hierarchical portfolios.

# DBHT root methods.
const DBHTRootMethods = (:unique, :Equal)

# OWA Methods.
const OWAMethods = (:crra, :me, :mss, :SD)

# Mutual and variation info bins and types.
const BinTypes = (:kn, :fd, :sc, :hgr)
const InfoTypes = (:mutual, :variation)

# Portfolio risk measures.

# HRPortfolio risk measures.
const HRRiskMeasures = (
    RiskMeasures...,
    :Variance,
    :Equal,
    :VaR,
    :DaR,
    :MDD_r,
    :ADD_r,
    :DaR_r,
    :CDaR_r,
    :EDaR_r,
    :RDaR_r,
)

const HRTypes = (:hrp, :herc, :herc2, :nco)
const CodepTypes = (
    :pearson,
    :spearman,
    :kendall,
    :gerber1,
    :gerber2,
    :abs_pearson,
    :abs_spearman,
    :abs_kendall,
    :distance,
    :mutual_info,
    :tail,
    :custom_cov,
    :custom_cor,
)
const LinkageTypes = (:single, :complete, :average, :ward_presquared, :ward, :dbht)
const BranchOrderTypes = (:optimal, :barjoseph, :r, :default)
const HRObjFuncs = (:min_risk, :utility, :sharpe, :erc)

export AbstractPortfolio,
    RiskMeasures,
    KellyRet,
    TrackingErrKinds,
    ObjFuncs,
    ValidTermination,
    PortClasses,
    PortTypes,
    RRPVersions,
    RPConstraintTypes,
    UncertaintyTypes,
    KindBootstrap,
    EllipseTypes,
    BoxTypes
