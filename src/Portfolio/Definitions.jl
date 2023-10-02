"""
```julia
AbstractPortfolio
```
Abstract type for portfolios. Concrete portfolios subtype this see [`Portfolio`](@ref) and [`HCPortfolio`](@ref).
"""
abstract type AbstractPortfolio end

"""
```julia
RiskMeasures = (
    :SD,
    :MAD,
    :SSD,
    :FLPM,
    :SLPM,
    :WR,
    :CVaR,
    :EVaR,
    :RVaR,
    :MDD,
    :ADD,
    :CDaR,
    :UCI,
    :EDaR,
    :RDaR,
    :Kurt,
    :SKurt,
    :GMD,
    :RG,
    :RCVaR,
    :TG,
    :RTG,
    :OWA,
)
```
Available risk measures for `:trad` and `:rp` type (see [`PortTypes`](@ref)) of [`Portfolio`](@ref).
- `:SD` = standard deviation ([`SD`](@ref));
- `:MAD` = max absolute deviation ([`MAD`](@ref));
- `:SSD` = semi standard deviation ([`SSD`](@ref));
- `:FLPM` = first lower partial moment (omega ratio) ([`FLPM`](@ref));
- `:SLPM` = second lower partial moment (sortino ratio) ([`SLPM`](@ref));
- `:WR` = worst realisation ([`WR`](@ref));
- `:CVaR` = conditional value at risk ([`CVaR`](@ref));
- `:EVaR` = entropic value at risk ([`EVaR`](@ref));
- `:RVaR` = relativistic value at risk ([`RVaR`](@ref));
- `:MDD` = maximum drawdown of uncompounded cumulative returns ([`MDD_abs`](@ref));
- `:ADD` = average drawdown of uncompounded cumulative returns ([`ADD_abs`](@ref));
- `:CDaR` = conditional drawdown at risk of uncompounded cumulative returns ([`CDaR_abs`](@ref));
- `:UCI` = ulcer index of uncompounded cumulative returns ([`UCI_abs`](@ref));
- `:EDaR` = entropic drawdown at risk of uncompounded cumulative returns ([`EDaR_abs`](@ref));
- `:RDaR` = relativistic drawdown at risk of uncompounded cumulative returns ([`RDaR_abs`](@ref));
- `:Kurt` = square root kurtosis ([`Kurt`](@ref));
- `:SKurt` = square root semi-kurtosis ([`SKurt`](@ref));
- `:GMD` = gini mean difference ([`GMD`](@ref));
- `:RG` = range of returns ([`RG`](@ref));
- `:RCVaR` = range of conditional value at risk ([`RCVaR`](@ref));
- `:TG` = tail gini ([`TG`](@ref));
- `:RTG` = range of tail gini ([`RTG`](@ref));
- `:OWA` = ordered weight array (generic OWA weights) ([`OWA`](@ref)).
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
```julia
KellyRet = (:none, :approx, :exact)
```
Available types of Kelly returns for [`Portfolio`](@ref).
- `:none`: arithmetic mean return, ``R(\\bm{w}) = \\bm{\\mu} \\cdot \\bm{w}``;
- `:approx`: first moment approximation of the logarithmic returns, ``R(\\bm{w}) = \\bm{\\mu} \\cdot \\bm{w} - \\dfrac{1}{2} \\bm{w}^{\\intercal} \\mathbf{\\Sigma} \\bm{w}``;
- `:exact`: exact logarithmic returns, ``R(\\bm{w}) = \\dfrac{1}{T}\\sum\\limits_{t=1}^{T}\\ln\\left(1 + \\bm{x}_t \\cdot \\bm{w}\\right)``.
Where:
- ``\\mathbf{\\Sigma}`` is the covariance matrix of the asset returns; 
- ``\\bm{x}_t`` is the vector of asset returns at timestep ``t``; 
- ``\\bm{\\mu}`` is the vector of expected returns for each asset; 
- and ``\\bm{w}`` is the asset weights vector.
"""
const KellyRet = (:none, :approx, :exact)

"""
```julia
TrackingErrKinds = (:weights, :returns)
```
Available kinds of tracking errors for [`Portfolio`](@ref).
- `:weights`: provide a vector of asset weights which is used to compute the vector of benchmark returns;
- `:returns`: directly provide the vector of benchmark returns.
The benchmark is then used as a reference to optimise a portfolio that tracks it up to a given error.
"""
const TrackingErrKinds = (:weights, :returns)

"""
```julia
ObjFuncs = (:min_risk, :utility, :sharpe, :max_ret)
```
Objective functions available for use in `type = :trad` and `type = :wc` optimisations (see [`PortTypes`](@ref)).
"""
const ObjFuncs = (:min_risk, :utility, :sharpe, :max_ret)

"""
```julia
ValidTermination =
(MOI.OPTIMAL, MOI.ALMOST_OPTIMAL, MOI.LOCALLY_SOLVED, MOI.ALMOST_LOCALLY_SOLVED)
```
Valid `JuMP` termination codes after optimising an instance of [`Portfolio`](@ref). If the termination code is different to these, then the failures are logged in the `.fail` field of [`HCPortfolio`](@ref) and [`Portfolio`](@ref).
"""
const ValidTermination =
    (MOI.OPTIMAL, MOI.ALMOST_OPTIMAL, MOI.LOCALLY_SOLVED, MOI.ALMOST_LOCALLY_SOLVED)

"""
```julia
PortClasses = (:classic,)
```
Available classes for [`Portfolio`](@ref).
"""
const PortClasses = (:classic,)

"""
```julia
PortTypes = (:trad, :rp, :rrp, :wc)
```
Available optimisation types for [`Portfolio`](@ref).
### `:trad` -- Traditional Optimisations
Available objective functions for `type = :trad` optimisations. We can chose any of the objective functions in [`ObjFuncs`](@ref) and risk measures in [`RiskMeasures`](@ref).
- `:min_risk`: minimum risk portfolio,
```math
\\begin{align*}
\\underset{\\bm{w}}{\\min} &\\qquad \\phi_{j}(\\bm{w}) \\\\
\\mathrm{s.t.} &\\qquad \\mathbf{A} \\bm{w} \\geq \\mathbf{B} \\\\
&\\qquad \\phi_{i}(\\bm{w}) \\leq c_{i} \\, \\forall \\, \\phi_{i} \\in \\left\\{\\Phi\\right\\} \\\\
&\\qquad R(\\bm{w}) \\geq \\overline{\\mu}
\\end{align*}\\,.
```
- `:utility`: maximum utility portfolio,
```math
\\begin{align*}
\\underset{\\bm{w}}{\\max} &\\qquad R(\\bm{w}) - \\lambda \\phi_{j}(\\bm{w}) \\\\
\\mathrm{s.t.} &\\qquad \\mathbf{A} \\bm{w} \\geq \\mathbf{B} \\\\
&\\qquad \\phi_{i}(\\bm{w}) \\leq c_{i} \\, \\forall \\, \\phi_{i} \\in \\left\\{\\Phi\\right\\} \\\\
&\\qquad R(\\bm{w}) \\geq \\overline{\\mu}
\\end{align*}\\,.
```
- `:sharpe`: maximum risk-adjusted return ratio portfolio,
```math
\\begin{align*}
\\underset{\\bm{w}}{\\max} &\\qquad \\dfrac{R(\\bm{w}) - r}{\\phi_{j}(\\bm{w})} \\\\
\\mathrm{s.t.} &\\qquad \\mathbf{A} \\bm{w} \\geq \\mathbf{B} \\\\
&\\qquad \\phi_{i}(\\bm{w}) \\leq c_{i} \\, \\forall \\, \\phi_{i} \\in \\left\\{\\Phi\\right\\} \\\\
&\\qquad R(\\bm{w}) \\geq \\overline{\\mu}
\\end{align*}\\,.
```
- `:max_ret`: maximum return portfolio,
```math
\\begin{align*}
\\underset{\\bm{w}}{\\max} &\\qquad R(\\bm{w}) \\\\
\\mathrm{s.t.} &\\qquad \\mathbf{A} \\bm{w} \\geq \\mathbf{B} \\\\
&\\qquad \\phi_{i}(\\bm{w}) \\leq c_{i} \\, \\forall \\, \\phi_{i} \\in \\left\\{\\Phi\\right\\} \\\\
&\\qquad R(\\bm{w}) \\geq \\overline{\\mu}
\\end{align*}\\,.
```
Where:
- ``\\bm{w}`` are the asset weights;
- ``\\phi_{i}`` is risk measure ``i`` from the set of available risk measures ``\\left\\{\\Phi\\right\\}`` (see [`RiskMeasures`](@ref));
- ``\\mathbf{A} \\bm{w} \\geq \\mathbf{B}`` are the asset linear constraints;
- ``c_{i}`` is the maximum acceptable value for risk measure ``\\phi_{i}`` of the optimised portfolio;
- ``R(\\bm{w})`` is the return function from [`KellyRet`](@ref);
- ``\\overline{\\mu}`` is the minimum acceptable return of the optimised portfolio;
- ``\\lambda`` is the risk aversion coefficient;
- and ``r`` is the risk-free rate.

### `:rp` -- Risk Parity Optimisations
Optimises portfolios based on a vector of risk contributions per asset. We can chose any of the risk measures in [`RiskMeasures`](@ref).
```math
\\begin{align*}
\\underset{\\bm{w}}{\\min} &\\qquad \\phi(\\bm{w}) \\\\
\\mathrm{s.t.} &\\qquad \\mathbf{A} \\bm{w} \\geq \\mathbf{B} \\\\
&\\qquad \\bm{b} \\cdot \\ln(\\bm{w}) \\geq c \\\\
&\\qquad R(\\bm{w}) \\geq \\overline{\\mu}\\\\
&\\qquad \\bm{w} \\geq \\bm{0}
\\end{align*}\\,.
```
Where:
- ``\\bm{w}`` are the asset weights;
- ``\\phi`` a risk measure from the set of available risk measures (see [`RiskMeasures`](@ref));
- ``\\mathbf{A} \\bm{w} \\geq \\mathbf{B}`` are the asset linear constraints;
- ``\\bm{b}`` is the vector of maximum allowable risk contribution per asset to the optimised portfolio;
- ``c`` is an auxiliary variable;
- ``R(\\bm{w})`` is the return function from [`KellyRet`](@ref);
- and ``\\overline{\\mu}`` is the minimum acceptable return of the optimised portfolio.

### `:rrp` -- Relaxed Risk Parity Optimisations
Optimises portfolios based on a vector of risk contributions per asset. Defines its own risk measure using the portfolio returns covariance.
```math
\\begin{align*}
\\underset{\\bm{w}}{\\min} &\\qquad \\psi - \\gamma \\\\
\\mathrm{s.t.} &\\qquad \\mathbf{A} \\bm{w} \\geq \\mathbf{B} \\\\
&\\qquad \\bm{w}^\\intercal \\, \\mathbf{\\Sigma}\\, \\bm{w} \\leq (\\psi^{2} - \\rho^{2})\\\\
&\\qquad \\lambda \\bm{w}^\\intercal \\, \\mathbf{\\Theta}\\, \\bm{w} \\leq \\rho^{2}\\\\
&\\qquad \\bm{\\zeta} = \\mathbf{\\Sigma} \\bm{w}\\\\
&\\qquad w_{i} \\zeta_{i} \\geq \\gamma^{2} b_{i} \\qquad \\forall \\, i = 1,\\, \\ldots{},\\, N\\\\
&\\qquad R(\\bm{w}) \\geq \\overline{\\mu}\\\\
&\\qquad \\sum\\limits_{i=1}^{N} w_{i} = 1\\\\
&\\qquad \\bm{w} \\geq \\bm{0}\\\\
&\\qquad \\psi,\\, \\gamma,\\, \\rho \\geq 0
\\end{align*}\\,.
```
Where:
- ``\\bm{w}`` are the asset weights;
- ``\\psi`` is the average risk of the portfolio;
- ``\\gamma`` is the lower bound of the risk contribution for each asset;
- ``\\mathbf{A} \\bm{w} \\geq \\mathbf{B}`` are the asset linear constraints;
- ``\\mathbf{\\Sigma}`` is the portfolio covariance;
- ``\\rho`` is a regularisation variable;
- ``\\mathbf{\\Theta} = \\mathrm{diag}\\left(\\mathbf{\\Sigma}\\right)`` ;
- ``\\lambda`` is a penalty parameter for ``\\rho``, taken from the available choices in [`RRPVersions`](@ref);
- ``\\bm{\\zeta}`` is the vector of marginal risk for each asset;
- ``b_{i}`` is the maximum allowable risk contribution for asset ``i``;
- ``N`` is the number of assets;
- ``R(\\bm{w})`` is the return function from [`KellyRet`](@ref);
- and ``\\overline{\\mu}`` is the minimum acceptable return of the optimised portfolio.

### `:wc` -- Worst Case Mean Variance Optimisations
Computes the worst case mean variance portfolio according to user-selected uncertainty sets (see [`UncertaintyTypes`](@ref)) for the portfolio return and covariance. We can chose any of the objective functions in [`ObjFuncs`](@ref).
- `:min_risk`: worst case minimum risk mean-variance portfolio,
```math
\\begin{align*}
\\underset{\\bm{w}}{\\max} &\\qquad \\underset{\\mathbf{\\Sigma}\\, \\in\\, U_{\\mathbf{\\Sigma}}}{\\max}\\, \\bm{w}^{\\intercal}\\, \\mathbf{\\Sigma}\\, \\bm{w}\\\\
\\mathrm{s.t.} &\\qquad \\mathbf{A} \\bm{w} \\geq \\mathbf{B}\\,.
\\end{align*}
```
- `:utility`: worst case maximum utility mean-variance portfolio,
```math
\\begin{align*}
\\underset{\\bm{w}}{\\max} &\\qquad \\underset{\\bm{\\mu}\\, \\in\\, U_{\\bm{\\mu}}}{\\min} R(\\bm{w})\\, -\\, \\underset{\\mathbf{\\Sigma}\\, \\in\\, U_{\\mathbf{\\Sigma}}}{\\max}\\, \\lambda \\bm{w}^{\\intercal}\\, \\mathbf{\\Sigma}\\, \\bm{w}\\\\
\\mathrm{s.t.} &\\qquad \\mathbf{A} \\bm{w} \\geq \\mathbf{B}\\,.
\\end{align*}
```
- `:sharpe`: worst case maximum risk-adjusted return ratio mean-variance portfolio,
```math
\\begin{align*}
\\underset{\\bm{w}}{\\max} &\\qquad \\dfrac{\\underset{\\bm{\\mu}\\, \\in\\, U_{\\bm{\\mu}}}{\\min} R(\\bm{w}) - r}{\\underset{\\mathbf{\\Sigma}\\, \\in\\, U_{\\mathbf{\\Sigma}}}{\\max}\\, \\left(\\bm{w}^{\\intercal}\\, \\mathbf{\\Sigma}\\, \\bm{w}\\right)^{1/2}} \\\\
\\mathrm{s.t.} &\\qquad \\mathbf{A} \\bm{w} \\geq \\mathbf{B}\\,.
\\end{align*}
```
- `:max_ret`: worst case maximum return mean-variance portfolio,
```math
\\begin{align*}
\\underset{\\bm{w}}{\\max} &\\qquad \\underset{\\bm{\\mu}\\, \\in\\, U_{\\bm{\\mu}}}{\\min} R(\\bm{w})\\\\
\\mathrm{s.t.} &\\qquad \\mathbf{A} \\bm{w} \\geq \\mathbf{B}\\,.
\\end{align*}
```
Where:
- ``\\bm{w}`` are the asset weights;
- ``\\mathbf{\\Sigma}`` is the covariance matrix of asset returns;
- ``U_{\\Sigma}`` is the uncertainty set for the covariance matrix, they can be:
```math
\\begin{align*}
U_{\\Sigma}^{\\mathrm{box}} &= \\left\\{\\mathbf{\\Sigma}\\, \\vert\\, \\mathbf{\\Sigma}_{l} \\leq \\mathbf{\\Sigma} \\leq \\mathbf{\\Sigma}_{u},\\, \\mathbf{\\Sigma} \\succeq 0\\right\\}\\\\
U_{\\Sigma}^{\\mathrm{ellipse}} &= \\left\\{\\mathbf{\\Sigma}\\, \\vert\\, \\left[\\mathrm{vec}\\left(\\mathbf{\\Sigma}\\right) - \\mathrm{vec}\\left(\\hat{\\mathbf{\\Sigma}}\\right)\\right] \\mathbf{\\Sigma}_{\\mathbf{\\Sigma}}^{-1} \\left[\\mathrm{vec}\\left(\\mathbf{\\Sigma}\\right) - \\mathrm{vec}\\left(\\hat{\\mathbf{\\Sigma}}\\right)\\right]^{\\intercal} \\leq k_{\\mathbf{\\Sigma}}^2 ,\\, \\mathbf{\\Sigma} \\succeq 0\\right\\}\\,.
\\end{align*}
```
- Where the following variables are estimated by assuming that the portfolio's asset return covariance can be generated by *some* matrix distribution. The distribution is sampled, and the estimates are calculated from them. Available choices can be found in [`BoxTypes`](@ref) and [`EllipseTypes`](@ref) for the box and ellipse sets respectively: 
    - the ``\\mathrm{l}`` and ``\\mathrm{u}`` subscripts denote lower and upper bounds for the covariance matrix given the samples;
    - ``\\mathbf{\\Sigma}_{\\mathbf{\\Sigma}}`` is the covariance of the samples;
    - ``\\hat{\\mathbf{\\Sigma}}`` the expected covariance given the samples;
    - and ``k_{\\mathbf{\\Sigma}}`` is a significance parameter of the matrix distribution.
- ``\\mathbf{A} \\bm{w} \\geq \\mathbf{B}`` are the asset linear constraints;
- ``\\bm{\\mu}`` is the vector of expected returns for each asset; 
- ``R(\\bm{w})`` is the return function from [`KellyRet`](@ref);
- ``U_{\\bm{\\mu}}`` is the uncertainty set for the asset returns, they can be:
```math
\\begin{align*}
U_{\\bm{\\mu}}^{\\mathrm{box}} &= \\left\\{\\bm{\\mu}\\, \\vert\\, \\vert \\bm{\\mu} - \\bm{\\hat{\\mu}} \\vert \\leq \\delta \\right\\}\\\\
U_{\\bm{\\mu}}^{\\mathrm{ellipse}} &= \\left\\{\\bm{\\mu}\\, \\vert\\, \\left(\\bm{\\mu} - \\bm{\\hat{\\mu}}\\right) \\mathbf{\\Sigma}_{\\bm{\\mu}}^{-1} \\left(\\bm{\\mu} - \\bm{\\hat{\\mu}}\\right)^{\\intercal} \\leq k_{\\bm{\\mu}}^{2}\\right\\}\\,.
\\end{align*}
```
- Where the following variables are estimated by assuming that the portfolio's asset mean returns can be generated by *some* distribution. The distribution is sampled, and the estimates are calculated from them. Available choices can be found in [`BoxTypes`](@ref) and [`EllipseTypes`](@ref) for the box and ellipse sets respectively:
    - ``\\hat{\\bm{\\mu}}`` the expected portfolio asset mean returns given the samples;
    - ``\\mathbf{\\Sigma}_{\\bm{\\mu}}`` is the covariance of the samples;
    - and ``k_{\\bm{\\mu}}`` is a significance parameter of the distribution.
- ``\\lambda`` is the risk aversion coefficient;
- and ``r`` is the risk-free rate.
"""
const PortTypes = (:trad, :rp, :rrp, :wc)

"""
```julia
RRPVersions = (:none, :reg, :reg_pen)
```
Available versions of relaxed risk parity optimisations of [`Portfolio`](@ref).
"""
const RRPVersions = (:none, :reg, :reg_pen)

"""
```julia
RPConstraintTypes = (:assets, :classes)
```
Types of risk parity constraints for [`rp_constraints`](@ref).
"""
const RPConstraintTypes = (:assets, :classes)

"""
```julia
UncertaintyTypes = (:none, :box, :ellipse)
```
Types of uncertainty sets for worst case optimisations of [`Portfolio`](@ref).
"""
const UncertaintyTypes = (:none, :box, :ellipse)

"""
Bootstrap for worst case optimisations.
```
KindBootstrap = (:stationary, :circular, :moving)
```
"""
const KindBootstrap = (:stationary, :circular, :moving)
"""
Ellipse and box types for worst case optimisations.
```
EllipseTypes = (:stationary, :circular, :moving, :normal)
```
"""
const EllipseTypes = (:stationary, :circular, :moving, :normal)
"""
```julia
BoxTypes = (:stationary, :circular, :moving, :normal, :delta)
```
"""
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
"""
```julia
HRRiskMeasures = (
    :SD,
    :MAD,
    :SSD,
    :FLPM,
    :SLPM,
    :WR,
    :CVaR,
    :EVaR,
    :RVaR,
    :MDD,
    :ADD,
    :CDaR,
    :UCI,
    :EDaR,
    :RDaR,
    :Kurt,
    :SKurt,
    :GMD,
    :RG,
    :RCVaR,
    :TG,
    :RTG,
    :OWA,
    :Variance,
    :Equal,
    :VaR,
    :DaR,
    :DaR_r,
    :MDD_r,
    :ADD_r,
    :CDaR_r,
    :EDaR_r,
    :RDaR_r,
)
```
Available risk measures for optimisations of [`HCPortfolio`](@ref).
- `:SD` = standard deviation ([`SD`](@ref));
- `:MAD` = max absolute deviation ([`MAD`](@ref));
- `:SSD` = semi standard deviation ([`SSD`](@ref));
- `:FLPM` = first lower partial moment (Omega ratio) ([`FLPM`](@ref));
- `:SLPM` = second lower partial moment (Sortino ratio) ([`SLPM`](@ref));
- `:WR` = worst realisation ([`WR`](@ref));
- `:CVaR` = conditional value at risk ([`CVaR`](@ref));
- `:EVaR` = entropic value at risk ([`EVaR`](@ref));
- `:RVaR` = relativistic value at risk ([`RVaR`](@ref));
- `:MDD` = maximum drawdown of uncompounded cumulative returns (Calmar ratio) ([`MDD_abs`](@ref));
- `:ADD` = average drawdown of uncompounded cumulative returns ([`ADD_abs`](@ref));
- `:CDaR` = conditional drawdown at risk of uncompounded cumulative returns ([`CDaR_abs`](@ref));
- `:UCI` = ulcer index of uncompounded cumulative returns ([`UCI_abs`](@ref));
- `:EDaR` = entropic drawdown at risk of uncompounded cumulative returns ([`EDaR_abs`](@ref));
- `:RDaR` = relativistic drawdown at risk of uncompounded cumulative returns ([`RDaR_abs`](@ref));
- `:Kurt` = square root kurtosis ([`Kurt`](@ref));
- `:SKurt` = square root semi-kurtosis ([`SKurt`](@ref));
- `:GMD` = gini mean difference ([`GMD`](@ref));
- `:RG` = range of returns ([`RG`](@ref));
- `:RCVaR` = range of conditional value at risk ([`RCVaR`](@ref));
- `:TG` = tail gini ([`TG`](@ref));
- `:RTG` = range of tail gini ([`RTG`](@ref));
- `:OWA` = ordered weight array (generic OWA weights) ([`OWA`](@ref));
- `:Variance` = variance ([`Variance`](@ref));
- `:Equal` = equal risk contribution, `1/N` where N is the number of assets;
- `:VaR` = value at risk ([`VaR`](@ref));
- `:DaR` = drawdown at risk of uncompounded cumulative returns ([`DaR_abs`](@ref));
- `:DaR_r` = drawdown at risk of compounded cumulative returns ([`DaR_rel`](@ref));
- `:MDD`_r = maximum drawdown of compounded cumulative returns ([`MDD_rel`](@ref));
- `:ADD`_r = average drawdown of compounded cumulative returns ([`ADD_rel`](@ref));
- `:CDaR_r` = conditional drawdown at risk of compounded cumulative returns ([`CDaR_rel`](@ref));
- `:UCI`_r = ulcer index of compounded cumulative returns ([`UCI_rel`](@ref));
- `:EDaR_r` = entropic drawdown at risk of compounded cumulative returns ([`EDaR_rel`](@ref));
- `:RDaR_r` = relativistic drawdown at risk of compounded cumulative returns ([`RDaR_rel`](@ref)).
"""
const HRRiskMeasures = (
    RiskMeasures...,
    :Variance,
    :Equal,
    :VaR,
    :DaR,
    :DaR_r,
    :MDD_r,
    :ADD_r,
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
    BoxTypes,
    HRRiskMeasures
