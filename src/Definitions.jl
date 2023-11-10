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
Available risk measures for `:Trad` and `:RP` type (see [`PortTypes`](@ref)) of [`Portfolio`](@ref).
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
KellyRet = (:None, :Approx, :Exact)
```
Available types of Kelly returns for [`Portfolio`](@ref).
- `:None`: arithmetic mean return, ``R(\\bm{w}) = \\bm{\\mu} \\cdot \\bm{w}``;
- `:Approx`: first moment approximation of the logarithmic returns, ``R(\\bm{w}) = \\bm{\\mu} \\cdot \\bm{w} - \\dfrac{1}{2} \\bm{w}^{\\intercal} \\mathbf{\\Sigma} \\bm{w}``;
- `:Exact`: exact logarithmic returns, ``R(\\bm{w}) = \\dfrac{1}{T}\\sum\\limits_{t=1}^{T}\\ln\\left(1 + \\bm{x}_t \\cdot \\bm{w}\\right)``.
Where:
- ``\\mathbf{\\Sigma}`` is the covariance matrix of the asset returns; 
- ``\\bm{x}_t`` is the vector of asset returns at timestep ``t``; 
- ``\\bm{\\mu}`` is the vector of expected returns for each asset; 
- and ``\\bm{w}`` is the asset weights vector.
"""
const KellyRet = (:None, :Approx, :Exact)

"""
```julia
TrackingErrKinds = (:Weights, :Returns)
```
Available kinds of tracking errors for [`Portfolio`](@ref).
- `:Weights`: provide a vector of asset weights which is used to compute the vector of benchmark returns;
- `:Returns`: directly provide the vector of benchmark returns.
The benchmark is then used as a reference to optimise a portfolio that tracks it up to a given error.
"""
const TrackingErrKinds = (:Weights, :Returns)

"""
```julia
ObjFuncs = (:Min_Risk, :Utility, :Sharpe, :Max_Ret)
```
Objective functions available for use in `:Trad` and `:WC` optimisations of [`Portfolio`](@ref) (see [`PortTypes`](@ref)).
"""
const ObjFuncs = (:Min_Risk, :Utility, :Sharpe, :Max_Ret)

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
PortClasses = (:Classic,)
```
Available classes for [`Portfolio`](@ref).
"""
const PortClasses = (:Classic,)

"""
```julia
PortTypes = (:Trad, :RP, :RRP, :WC)
```
Available optimisation types for [`Portfolio`](@ref).
### `:Trad` -- Traditional Optimisations
Available objective functions for `:Trad` optimisations. We can chose any of the objective functions in [`ObjFuncs`](@ref) and risk measures in [`RiskMeasures`](@ref).
- `:Min_Risk`: minimum risk portfolio,
```math
\\begin{align*}
\\underset{\\bm{w}}{\\min} &\\qquad \\phi_{j}(\\bm{w}) \\\\
\\mathrm{s.t.} &\\qquad \\mathbf{A} \\bm{w} \\geq \\mathbf{B} \\\\
&\\qquad \\phi_{i}(\\bm{w}) \\leq c_{i} \\, \\forall \\, \\phi_{i} \\in \\left\\{\\Phi\\right\\} \\\\
&\\qquad R(\\bm{w}) \\geq \\overline{\\mu}
\\end{align*}\\,.
```
- `:Utility`: maximum utility portfolio,
```math
\\begin{align*}
\\underset{\\bm{w}}{\\max} &\\qquad R(\\bm{w}) - \\lambda \\phi_{j}(\\bm{w}) \\\\
\\mathrm{s.t.} &\\qquad \\mathbf{A} \\bm{w} \\geq \\mathbf{B} \\\\
&\\qquad \\phi_{i}(\\bm{w}) \\leq c_{i} \\, \\forall \\, \\phi_{i} \\in \\left\\{\\Phi\\right\\} \\\\
&\\qquad R(\\bm{w}) \\geq \\overline{\\mu}
\\end{align*}\\,.
```
- `:Sharpe`: maximum risk-adjusted return ratio portfolio,
```math
\\begin{align*}
\\underset{\\bm{w}}{\\max} &\\qquad \\dfrac{R(\\bm{w}) - r}{\\phi_{j}(\\bm{w})} \\\\
\\mathrm{s.t.} &\\qquad \\mathbf{A} \\bm{w} \\geq \\mathbf{B} \\\\
&\\qquad \\phi_{i}(\\bm{w}) \\leq c_{i} \\, \\forall \\, \\phi_{i} \\in \\left\\{\\Phi\\right\\} \\\\
&\\qquad R(\\bm{w}) \\geq \\overline{\\mu}
\\end{align*}\\,.
```
- `:Max_Ret`: maximum return portfolio,
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
- ``\\mathbf{A} \\bm{w} \\geq \\mathbf{B}`` is a set of linear constraints;
- ``c_{i}`` is the maximum acceptable value for risk measure ``\\phi_{i}`` of the optimised portfolio;
- ``R(\\bm{w})`` is the return function from [`KellyRet`](@ref);
- ``\\overline{\\mu}`` is the minimum acceptable return of the optimised portfolio;
- ``\\lambda`` is the risk aversion coefficient;
- and ``r`` is the risk-free rate.

### `:RP` -- Risk Parity Optimisations
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
- ``\\mathbf{A} \\bm{w} \\geq \\mathbf{B}`` is a set of linear constraints;
- ``\\bm{b}`` is the vector of maximum allowable risk contribution per asset to the optimised portfolio;
- ``c`` is an auxiliary variable;
- ``R(\\bm{w})`` is the return function from [`KellyRet`](@ref);
- and ``\\overline{\\mu}`` is the minimum acceptable return of the optimised portfolio.

### `:RRP` -- Relaxed Risk Parity Optimisations
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
- ``\\mathbf{A} \\bm{w} \\geq \\mathbf{B}`` is a set of linear constraints;
- ``\\mathbf{\\Sigma}`` is the portfolio covariance;
- ``\\rho`` is a regularisation variable;
- ``\\mathbf{\\Theta} = \\mathrm{diag}\\left(\\mathbf{\\Sigma}\\right)`` ;
- ``\\lambda`` is a penalty parameter for ``\\rho``, taken from the available choices in [`RRPVersions`](@ref);
- ``\\bm{\\zeta}`` is the vector of marginal risk for each asset;
- ``b_{i}`` is the maximum allowable risk contribution for asset ``i``;
- ``N`` is the number of assets;
- ``R(\\bm{w})`` is the return function from [`KellyRet`](@ref);
- and ``\\overline{\\mu}`` is the minimum acceptable return of the optimised portfolio.

### `:WC` -- Worst Case Mean Variance Optimisations
Computes the worst case mean variance portfolio according to user-selected uncertainty sets (see [`UncertaintyTypes`](@ref)) for the portfolio return and covariance. We can chose any of the objective functions in [`ObjFuncs`](@ref).
- `:Min_Risk`: worst case minimum risk mean-variance portfolio,
```math
\\begin{align*}
\\underset{\\bm{w}}{\\max} &\\qquad \\underset{\\mathbf{\\Sigma}\\, \\in\\, U_{\\mathbf{\\Sigma}}}{\\max}\\, \\bm{w}^{\\intercal}\\, \\mathbf{\\Sigma}\\, \\bm{w}\\\\
\\mathrm{s.t.} &\\qquad \\mathbf{A} \\bm{w} \\geq \\mathbf{B}\\,.
\\end{align*}
```
- `:Utility`: worst case maximum utility mean-variance portfolio,
```math
\\begin{align*}
\\underset{\\bm{w}}{\\max} &\\qquad \\underset{\\bm{\\mu}\\, \\in\\, U_{\\bm{\\mu}}}{\\min} R(\\bm{w})\\, -\\, \\underset{\\mathbf{\\Sigma}\\, \\in\\, U_{\\mathbf{\\Sigma}}}{\\max}\\, \\lambda \\bm{w}^{\\intercal}\\, \\mathbf{\\Sigma}\\, \\bm{w}\\\\
\\mathrm{s.t.} &\\qquad \\mathbf{A} \\bm{w} \\geq \\mathbf{B}\\,.
\\end{align*}
```
- `:Sharpe`: worst case maximum risk-adjusted return ratio mean-variance portfolio,
```math
\\begin{align*}
\\underset{\\bm{w}}{\\max} &\\qquad \\dfrac{\\underset{\\bm{\\mu}\\, \\in\\, U_{\\bm{\\mu}}}{\\min} R(\\bm{w}) - r}{\\underset{\\mathbf{\\Sigma}\\, \\in\\, U_{\\mathbf{\\Sigma}}}{\\max}\\, \\left(\\bm{w}^{\\intercal}\\, \\mathbf{\\Sigma}\\, \\bm{w}\\right)^{1/2}} \\\\
\\mathrm{s.t.} &\\qquad \\mathbf{A} \\bm{w} \\geq \\mathbf{B}\\,.
\\end{align*}
```
- `:Max_Ret`: worst case maximum return mean-variance portfolio,
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
- ``\\mathbf{A} \\bm{w} \\geq \\mathbf{B}`` is a set of linear constraints;
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
const PortTypes = (:Trad, :RP, :RRP, :WC)

"""
```julia
RRPVersions = (:None, :Reg, :Reg_Pen)
```
Available versions of Relaxed Risk Parity Optimisations (see [`PortTypes`](@ref)).
Where:
- `:None`: no penalty;
- `:Reg`: regularisation constraint, ``\\rho``;
- `:Reg_Pen`: regularisation and penalisation constraints, ``\\lambda`` and ``\\rho``.
"""
const RRPVersions = (:None, :Reg, :Reg_Pen)

"""
```julia
RPConstraintTypes = (:Assets, :Classes)
```
Types of risk parity constraints for building the set of linear constraints via [`rp_constraints`](@ref).
"""
const RPConstraintTypes = (:Assets, :Classes)

"""
```julia
UncertaintyTypes = (:None, :Box, :Ellipse)
```
Types of uncertainty sets for Worst Case Optimisations of [`Portfolio`](@ref) (see [`PortTypes`](@ref)).
"""
const UncertaintyTypes = (:None, :Box, :Ellipse)

"""
```
KindBootstrap = (:Stationary, :Circular, :Moving)
```
Kind of bootstrap for computing the uncertainty sets with [`wc_statistics!`](@ref), which are used by Worst Case Mean Variance Optimisations (see [`PortTypes`](@ref)).
"""
const KindBootstrap = (:Stationary, :Circular, :Moving)

"""
```
EllipseTypes = (:Stationary, :Circular, :Moving, :Normal)
```
Available types of elliptical sets that can be computed with [`wc_statistics!`](@ref), which are used by Worst Case Mean Variance Optimisations (see [`PortTypes`](@ref)).
"""
const EllipseTypes = (:Stationary, :Circular, :Moving, :Normal)

"""
```julia
BoxTypes = (:Stationary, :Circular, :Moving, :Normal, :Delta)
```
Available types of box sets that can be computed with [`wc_statistics!`](@ref), which are used by Worst Case Mean Variance Optimisations (see [`PortTypes`](@ref)).
"""
const BoxTypes = (EllipseTypes..., :Delta)

"""
```julia
DBHTRootMethods = (:Unique, :Equal)
```
Methods for finding the root of a Direct Bubble Hierarchical Clustering Tree in [`DBHTs`](@ref).
"""
const DBHTRootMethods = (:Unique, :Equal)

"""
```julia
OWAMethods = (:CRRA, :ME, :MSS, :SD)
```
Methods for computing the higher order L-moments of Ordered Weight Arrays in [`owa_l_moment_crm`](@ref).
"""
const OWAMethods = (:CRRA, :ME, :MSS, :SD)

"""
```julia
BinTypes = (:KN, :FD, :SC, :HGR)
```
Bin type choices for calculating bin widths for mutual and variational information matrices computed by [`mut_var_info_mtx`](@ref).
"""
const BinTypes = (:KN, :FD, :SC, :HGR)

"""
```julia
InfoTypes = (:Mutual, :Variation)
```
Type of information matrix to compute when choosing `:Mutual_Info` from [`CodepTypes`](@ref) in [`asset_statistics!`](@ref).
"""
const InfoTypes = (:Mutual, :Variation)

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
- `:MDD_r` = maximum drawdown of compounded cumulative returns ([`MDD_rel`](@ref));
- `:ADD_r` = average drawdown of compounded cumulative returns ([`ADD_rel`](@ref));
- `:CDaR_r` = conditional drawdown at risk of compounded cumulative returns ([`CDaR_rel`](@ref));
- `:UCI_r` = ulcer index of compounded cumulative returns ([`UCI_rel`](@ref));
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

"""
```julia
HCPortTypes = (:HRP, :HERC, :HERC2, :NCO)
```
"""
const HCPortTypes = (:HRP, :HERC, :HERC2, :NCO)

const MuTypes = (:Default, :JS, :BS, :BOP, :Custom_Func, :Custom_Val)
const MuTargets = (:GM, :VW, :SE)
const CovTypes = (:Full, :Semi, :Gerber1, :Gerber2, :Custom_Func, :Custom_Val)
const PosdefFixes = (:None, :Custom_Func)
const RegCriteria = (:pval, GLM.aic, GLM.aicc, GLM.bic, GLM.r2, GLM.adjr2)
const LoadingMtxType = (:FReg, :BReg, :PCR)

"""
```julia
CodepTypes = (
    :Pearson,
    :Spearman,
    :Kendall,
    :Gerber1,
    :Gerber2,
    :Abs_Pearson,
    :Abs_Spearman,
    :Abs_Kendall,
    :Distance,
    :Mutual_Info,
    :Tail,
    :Cov_to_Cor,
    :Custom_Func,
    :Custom_Val,
)
```
Available types of codependency measures computed in [`asset_statistics!`](@ref) when the portfolio is a [`HCPortfolio`](@ref).
"""
const CodepTypes = (
    :Pearson,
    :Spearman,
    :Kendall,
    :Gerber1,
    :Gerber2,
    :Abs_Pearson,
    :Abs_Spearman,
    :Abs_Kendall,
    :Distance,
    :Mutual_Info,
    :Tail,
    :Cov_to_Cor,
    :Custom_Func,
    :Custom_Val,
)

"""
```julia
LinkageTypes = (:single, :complete, :average, :ward, :ward_presquared, :DBHT)
```
Linkage types available when optimising a [`HCPortfolio`](@ref). Where:
- `:DBHT`: is Direct Bubble Hierarchical Tree clustering.
- The rest are linkage types supported by [Clustering.hclust](https://juliastats.org/Clustering.jl/stable/hclust.html#Clustering.hclust).
"""
const LinkageTypes = (:single, :complete, :average, :ward_presquared, :ward, :DBHT)

"""
```julia
BranchOrderTypes = (:optimal, :barjoseph, :r, :default)
```
Algorithm to order leaves and branches.
- `:default`: if linkage is `:DBHT`, the leaves and branches remain as the algorithm orders them. If any other linkage is used, they fall back to `:r` as that is their default according to [Clustering.hclust](https://juliastats.org/Clustering.jl/stable/hclust.html#Clustering.hclust).
- All other branch orders are as defined by [Clustering.hclust](https://juliastats.org/Clustering.jl/stable/hclust.html#Clustering.hclust).
"""
const BranchOrderTypes = (:optimal, :barjoseph, :r, :default)

"""
```julia
HRObjFuncs = (:Min_Risk, :Utility, :Sharpe, :Max_Ret, :Equal)
```
Objective funcions for `:NCO` [`HCPortTypes`](@ref) of [`HCPortfolio`](@ref). 
- `:Min_Risk`, `:Utility`, `:Sharpe` and `:Max_Ret` optimise the sub-portfolios as `:Trad` [`PortTypes`](@ref) of [`Portfolio`](@ref) according to their respective definitions in [`ObjFuncs`](@ref). 
- `:Equal` optimises the sub-portfolios as `:RP` [`PortTypes`](@ref) of [`Portfolio`](@ref) with equal risk contribution per asset. We can't offer customiseable risk contributions because the size and composition of the clusters is initially unknown and depends on the chosen linkage method.
"""
const HRObjFuncs = (ObjFuncs..., :Equal)

const AllocTypes = (:LP, :Greedy)

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
    DBHTRootMethods,
    OWAMethods,
    BinTypes,
    InfoTypes,
    HRRiskMeasures,
    HCPortTypes,
    MuTypes,
    CovTypes,
    CodepTypes,
    LinkageTypes,
    BranchOrderTypes,
    HRObjFuncs,
    AllocTypes,
    RegCriteria
