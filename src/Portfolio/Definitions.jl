"""
Abstract type for portfolios. Concrete portfolios subtype this see [`Portfolio`](@ref) and [`HCPortfolio`](@ref).
```
abstract type AbstractPortfolio end
```
"""
abstract type AbstractPortfolio end

"""
Available risk measures for `type = :trad` optimisations of [`Portfolio`](@ref).
    - `:mv` = standard deviation
    - `:mad` = max absolute deviation
    - `:msv` = semi standard deviation
    - `:flpm` = first lower partial moment (omega ratio)
    - `:slpm` = second lower partial moment (sortino ratio)
    - `:wr` = worst realisation
    - `:cvar` = conditional value at risk
    - `:evar` = entropic value at risk
    - `:rvar` = relativistic value at risk
    - `:mdd` = maximum drawdown
    - `:add` = average drawdown
    - `:cdar` = conditional drawdown at risk
    - `:uci` = ulcer index
    - `:edar` = entropic drawdown at risk
    - `:rdar` = relativistic drawdown at risk
    - `:krt` = kurtosis
    - `:skrt` = semi-kurtosis
    - `:gmd` = gini mean difference
    - `:rg` = range of returns
    - `:rcvar` = range of conditional value at risk
    - `:tg` = tail gini
    - `:rtg` = range of tail gini
    - `:owa` = ordered weight array (generic OWA weights)
```
const RiskMeasures = (
    :mv,    # _mv
    :mad,   # _mad
    :msv,   # _mad
    :flpm,  # _lpm
    :slpm,  # _lpm
    :wr,    # _wr
    :cvar,  # _var
    :evar,  # _var
    :rvar,  # _var
    :mdd,   # _dar
    :add,   # _dar
    :cdar,  # _dar
    :uci,   # _dar
    :edar,  # _dar
    :rdar,  # _dar
    :krt,   # _krt
    :skrt,  # _krt
    :gmd,   # _owa
    :rg,    # _owa
    :rcvar, # _owa
    :tg,    # _owa
    :rtg,   # _owa
    :owa,   # _owa
)
```
"""
const RiskMeasures = (
    :mv,    # _mv
    :mad,   # _mad
    :msv,   # _mad
    :flpm,  # _lpm
    :slpm,  # _lpm
    :wr,    # _wr
    :cvar,  # _var
    :evar,  # _var
    :rvar,  # _var
    :mdd,   # _dar
    :add,   # _dar
    :cdar,  # _dar
    :uci,   # _dar
    :edar,  # _dar
    :rdar,  # _dar
    :krt,   # _krt
    :skrt,  # _krt
    :gmd,   # _owa
    :rg,    # _owa
    :rcvar, # _owa
    :tg,    # _owa
    :rtg,   # _owa
    :owa,   # _owa
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
const PortTypes = (:trad, :rp, :rrp, :owa, :wc)
```
"""
const PortTypes = (:trad, :rp, :rrp, :owa, :wc)

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
const DBHTRootMethods = (:unique, :equal)

# OWA Methods.
const OWAMethods = (:crra, :me, :mss, :msd)

# Mutual and variation info bins and types.
const BinTypes = (:kn, :fd, :sc, :hgr)
const InfoTypes = (:mutual, :variation)

# Portfolio risk measures.

# HRPortfolio risk measures.
const HRRiskMeasures = (
    :msd,
    RiskMeasures...,
    :equal,
    :var,
    :dar,
    :mdd_r,
    :add_r,
    :dar_r,
    :cdar_r,
    :edar_r,
    :rdar_r,
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
