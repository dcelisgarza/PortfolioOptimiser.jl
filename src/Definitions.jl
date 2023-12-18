"""
```julia
KellyRet = (:None, :Approx, :Exact)
```
Available types of Kelly returns for [`Portfolio`](@ref).
- `:None`: arithmetic mean return, ``R(\\bm{w}) = \\bm{\\mu} \\cdot \\bm{w}``.
- `:Approx`: first moment approximation of the logarithmic returns, ``R(\\bm{w}) = \\bm{\\mu} \\cdot \\bm{w} - \\dfrac{1}{2} \\bm{w}^{\\intercal} \\mathbf{\\Sigma} \\bm{w}``.
- `:Exact`: exact logarithmic returns, ``R(\\bm{w}) = \\dfrac{1}{T}\\sum\\limits_{t=1}^{T}\\ln\\left(1 + \\bm{x}_t \\cdot \\bm{w}\\right)``.
Where:
- ``\\mathbf{\\Sigma}`` is the covariance matrix of the asset returns. 
- ``\\bm{x}_t`` is the vector of asset returns at timestep ``t``. 
- ``\\bm{\\mu}`` is the vector of expected returns for each asset. 
- and ``\\bm{w}`` is the asset weights vector.
"""
const KellyRet = (:None, :Approx, :Exact)

"""
```julia
ObjFuncs = (:Min_Risk, :Utility, :Sharpe, :Max_Ret)
```
Objective functions available for use in `:Trad` and `:WC` optimisations of [`Portfolio`](@ref) (see [`PortTypes`](@ref)).
"""
const ObjFuncs = (:Min_Risk, :Utility, :Sharpe, :Max_Ret)

"""
```julia
ValidTermination = (
    MOI.OPTIMAL,
    MOI.ALMOST_OPTIMAL,
    MOI.LOCALLY_SOLVED,
    MOI.ALMOST_LOCALLY_SOLVED,
    MOI.SOLUTION_LIMIT,
    MOI.OBJECTIVE_LIMIT,
)
```
Valid `JuMP` termination codes after optimising an instance of [`Portfolio`](@ref). If the termination code is different to these, then the failures are logged in the `.fail` field of [`HCPortfolio`](@ref) and [`Portfolio`](@ref).
"""
const ValidTermination = (
    MOI.OPTIMAL,
    MOI.ALMOST_OPTIMAL,
    MOI.LOCALLY_SOLVED,
    MOI.ALMOST_LOCALLY_SOLVED,
    MOI.SOLUTION_LIMIT,
    MOI.OBJECTIVE_LIMIT,
)

"""
```julia
PortClasses = (:Classic, :FM, :BL, :BLFM)
```
Available choicees of summary parameters ``\\bm{\\mu}`` and ``\\bm{\\Sigma}`` [`Portfolio`](@ref).
- `:Classic`: when optimising with this option, ``\\bm{\\mu}`` and ``\\bm{\\Sigma}`` take their values from historical estimates computed by [`asset_statistics!`](@ref).
- `:FM`: when optimising with this option, ``\\bm{\\mu}`` and ``\\bm{\\Sigma}`` take their values from the factor model computed by [`factor_statistics!`](@ref).
- `:BL`: when optimising with this option, ``\\bm{\\mu}`` and ``\\bm{\\Sigma}`` take their values from the Black-Litterman model computed by [`black_litterman_statistics!`](@ref).
- `:BLFM`: when optimising with this option, ``\\bm{\\mu}`` and ``\\bm{\\Sigma}`` take their values from the factor Black-Litterman model computed by [`black_litterman_factor_satistics!`](@ref). This model has two versions defined in [`BLFMType`](@ref).
"""
const PortClasses = (:Classic, :FM, :BL, :BLFM)

"""
```julia
BLFMType = (:A, :B)
```
Versions of the factor Black-Litterman Model.
- `:B`: Bayesian Black-Litterman, which uses the factors to generate the Black-Litterman estimates.
- `:A`: Augmented Black-Litterman, which uses the factors to adjust the Black-Litterman views.
"""
const BLFMType = (:A, :B)

"""
```julia
PortTypes = (:Trad, :RP, :RRP, :WC)
```
Available optimisation types for [`Portfolio`](@ref).
# `:Trad` -- Traditional Optimisations
Available objective functions for `:Trad` optimisations. We can chose any of the objective functions in [`ObjFuncs`](@ref) and risk measures in [`RiskMeasures`](@ref).
- `:Min_Risk`: minimum risk portfolio,
```math
\\begin{align*}
\\underset{\\bm{w}}{\\min} &\\qquad \\phi_{j}(\\bm{w}) \\\\
\\mathrm{s.t.} &\\qquad \\mathbf{A} \\bm{w} \\geq \\bm{B} \\\\
&\\qquad \\phi_{i}(\\bm{w}) \\leq c_{i} \\, \\forall \\, \\phi_{i} \\in \\left\\{\\Phi\\right\\} \\\\
&\\qquad R(\\bm{w}) \\geq \\overline{\\mu}
\\end{align*}\\,.
```
- `:Utility`: maximum utility portfolio,
```math
\\begin{align*}
\\underset{\\bm{w}}{\\max} &\\qquad R(\\bm{w}) - \\lambda \\phi_{j}(\\bm{w}) \\\\
\\mathrm{s.t.} &\\qquad \\mathbf{A} \\bm{w} \\geq \\bm{B} \\\\
&\\qquad \\phi_{i}(\\bm{w}) \\leq c_{i} \\, \\forall \\, \\phi_{i} \\in \\left\\{\\Phi\\right\\} \\\\
&\\qquad R(\\bm{w}) \\geq \\overline{\\mu}
\\end{align*}\\,.
```
- `:Sharpe`: maximum risk-adjusted return ratio portfolio,
```math
\\begin{align*}
\\underset{\\bm{w}}{\\max} &\\qquad \\dfrac{R(\\bm{w}) - r}{\\phi_{j}(\\bm{w})} \\\\
\\mathrm{s.t.} &\\qquad \\mathbf{A} \\bm{w} \\geq \\bm{B} \\\\
&\\qquad \\phi_{i}(\\bm{w}) \\leq c_{i} \\, \\forall \\, \\phi_{i} \\in \\left\\{\\Phi\\right\\} \\\\
&\\qquad R(\\bm{w}) \\geq \\overline{\\mu}
\\end{align*}\\,.
```
- `:Max_Ret`: maximum return portfolio,
```math
\\begin{align*}
\\underset{\\bm{w}}{\\max} &\\qquad R(\\bm{w}) \\\\
\\mathrm{s.t.} &\\qquad \\mathbf{A} \\bm{w} \\geq \\bm{B} \\\\
&\\qquad \\phi_{i}(\\bm{w}) \\leq c_{i} \\, \\forall \\, \\phi_{i} \\in \\left\\{\\Phi\\right\\} \\\\
&\\qquad R(\\bm{w}) \\geq \\overline{\\mu}
\\end{align*}\\,.
```
Where:
- ``\\bm{w}`` are the asset weights.
- ``\\phi_{i}`` is risk measure ``i`` from the set of available risk measures ``\\left\\{\\Phi\\right\\}`` (see [`RiskMeasures`](@ref)).
- ``\\mathbf{A} \\bm{w} \\geq \\bm{B}`` is a set of linear constraints.
- ``c_{i}`` is the maximum acceptable value for risk measure ``\\phi_{i}`` of the optimised portfolio.
- ``R(\\bm{w})`` is the return function from [`KellyRet`](@ref).
- ``\\overline{\\mu}`` is the minimum acceptable return of the optimised portfolio.
- ``\\lambda`` is the risk aversion coefficient.
- and ``r`` is the risk-free rate.

# `:RP` -- Risk Parity Optimisations
Optimises portfolios based on a vector of risk contributions per asset. We can chose any of the risk measures in [`RiskMeasures`](@ref).
```math
\\begin{align*}
\\underset{\\bm{w}}{\\min} &\\qquad \\phi(\\bm{w}) \\\\
\\mathrm{s.t.} &\\qquad \\mathbf{A} \\bm{w} \\geq \\bm{B} \\\\
&\\qquad \\bm{b} \\cdot \\ln(\\bm{w}) \\geq c \\\\
&\\qquad R(\\bm{w}) \\geq \\overline{\\mu}\\\\
&\\qquad \\bm{w} \\geq \\bm{0}
\\end{align*}\\,.
```
Where:
- ``\\bm{w}`` are the asset weights.
- ``\\phi`` a risk measure from the set of available risk measures (see [`RiskMeasures`](@ref)).
- ``\\mathbf{A} \\bm{w} \\geq \\bm{B}`` is a set of linear constraints.
- ``\\bm{b}`` is the vector of maximum allowable risk contribution per asset to the optimised portfolio.
- ``c`` is an auxiliary variable.
- ``R(\\bm{w})`` is the return function from [`KellyRet`](@ref).
- and ``\\overline{\\mu}`` is the minimum acceptable return of the optimised portfolio.

# `:RRP` -- Relaxed Risk Parity Optimisations
Optimises portfolios based on a vector of risk contributions per asset. Defines its own risk measure using the portfolio returns covariance.
```math
\\begin{align*}
\\underset{\\bm{w}}{\\min} &\\qquad \\psi - \\gamma \\\\
\\mathrm{s.t.} &\\qquad \\mathbf{A} \\bm{w} \\geq \\bm{B} \\\\
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
- ``\\bm{w}`` are the asset weights.
- ``\\psi`` is the average risk of the portfolio.
- ``\\gamma`` is the lower bound of the risk contribution for each asset.
- ``\\mathbf{A} \\bm{w} \\geq \\bm{B}`` is a set of linear constraints.
- ``\\mathbf{\\Sigma}`` is the portfolio covariance.
- ``\\rho`` is a regularisation variable.
- ``\\mathbf{\\Theta} = \\mathrm{diag}\\left(\\mathbf{\\Sigma}\\right)`` .
- ``\\lambda`` is a penalty parameter for ``\\rho``, taken from the available choices in [`RRPVersions`](@ref).
- ``\\bm{\\zeta}`` is the vector of marginal risk for each asset.
- ``b_{i}`` is the maximum allowable risk contribution for asset ``i``.
- ``N`` is the number of assets.
- ``R(\\bm{w})`` is the return function from [`KellyRet`](@ref).
- and ``\\overline{\\mu}`` is the minimum acceptable return of the optimised portfolio.

# `:WC` -- Worst Case Mean Variance Optimisations
Computes the worst case mean variance portfolio according to user-selected uncertainty sets (see [`UncertaintyTypes`](@ref)) for the portfolio return and covariance. We can chose any of the objective functions in [`ObjFuncs`](@ref).
- `:Min_Risk`: worst case minimum risk mean-variance portfolio,
```math
\\begin{align*}
\\underset{\\bm{w}}{\\max} &\\qquad \\underset{\\mathbf{\\Sigma}\\, \\in\\, U_{\\mathbf{\\Sigma}}}{\\max}\\, \\bm{w}^{\\intercal}\\, \\mathbf{\\Sigma}\\, \\bm{w}\\\\
\\mathrm{s.t.} &\\qquad \\mathbf{A} \\bm{w} \\geq \\bm{B}\\,.
\\end{align*}
```
- `:Utility`: worst case maximum utility mean-variance portfolio,
```math
\\begin{align*}
\\underset{\\bm{w}}{\\max} &\\qquad \\underset{\\bm{\\mu}\\, \\in\\, U_{\\bm{\\mu}}}{\\min} R(\\bm{w})\\, -\\, \\underset{\\mathbf{\\Sigma}\\, \\in\\, U_{\\mathbf{\\Sigma}}}{\\max}\\, \\lambda \\bm{w}^{\\intercal}\\, \\mathbf{\\Sigma}\\, \\bm{w}\\\\
\\mathrm{s.t.} &\\qquad \\mathbf{A} \\bm{w} \\geq \\bm{B}\\,.
\\end{align*}
```
- `:Sharpe`: worst case maximum risk-adjusted return ratio mean-variance portfolio,
```math
\\begin{align*}
\\underset{\\bm{w}}{\\max} &\\qquad \\dfrac{\\underset{\\bm{\\mu}\\, \\in\\, U_{\\bm{\\mu}}}{\\min} R(\\bm{w}) - r}{\\underset{\\mathbf{\\Sigma}\\, \\in\\, U_{\\mathbf{\\Sigma}}}{\\max}\\, \\left(\\bm{w}^{\\intercal}\\, \\mathbf{\\Sigma}\\, \\bm{w}\\right)^{1/2}} \\\\
\\mathrm{s.t.} &\\qquad \\mathbf{A} \\bm{w} \\geq \\bm{B}\\,.
\\end{align*}
```
- `:Max_Ret`: worst case maximum return mean-variance portfolio,
```math
\\begin{align*}
\\underset{\\bm{w}}{\\max} &\\qquad \\underset{\\bm{\\mu}\\, \\in\\, U_{\\bm{\\mu}}}{\\min} R(\\bm{w})\\\\
\\mathrm{s.t.} &\\qquad \\mathbf{A} \\bm{w} \\geq \\bm{B}\\,.
\\end{align*}
```
Where:
- ``\\bm{w}`` are the asset weights.
- ``\\mathbf{\\Sigma}`` is the covariance matrix of asset returns.
- ``U_{\\Sigma}`` is the uncertainty set for the covariance matrix, they can be:
```math
\\begin{align*}
U_{\\Sigma}^{\\mathrm{box}} &= \\left\\{\\mathbf{\\Sigma}\\, \\vert\\, \\mathbf{\\Sigma}_{l} \\leq \\mathbf{\\Sigma} \\leq \\mathbf{\\Sigma}_{u},\\, \\mathbf{\\Sigma} \\succeq 0\\right\\}\\\\
U_{\\Sigma}^{\\mathrm{ellipse}} &= \\left\\{\\mathbf{\\Sigma}\\, \\vert\\, \\left[\\mathrm{vec}\\left(\\mathbf{\\Sigma}\\right) - \\mathrm{vec}\\left(\\hat{\\mathbf{\\Sigma}}\\right)\\right] \\mathbf{\\Sigma}_{\\mathbf{\\Sigma}}^{-1} \\left[\\mathrm{vec}\\left(\\mathbf{\\Sigma}\\right) - \\mathrm{vec}\\left(\\hat{\\mathbf{\\Sigma}}\\right)\\right]^{\\intercal} \\leq k_{\\mathbf{\\Sigma}}^2 ,\\, \\mathbf{\\Sigma} \\succeq 0\\right\\}\\,.
\\end{align*}
```
- Where the following variables are estimated by assuming that the portfolio's asset return covariance can be generated by *some* matrix distribution. The distribution is sampled, and the estimates are calculated from them. Available choices can be found in [`BoxTypes`](@ref) and [`EllipseTypes`](@ref) for the box and ellipse sets respectively: 
    - the ``\\mathrm{l}`` and ``\\mathrm{u}`` subscripts denote lower and upper bounds for the covariance matrix given the samples.
    - ``\\mathbf{\\Sigma}_{\\mathbf{\\Sigma}}`` is the covariance of the samples.
    - ``\\hat{\\mathbf{\\Sigma}}`` the expected covariance given the samples.
    - and ``k_{\\mathbf{\\Sigma}}`` is a significance parameter of the matrix distribution.
- ``\\mathbf{A} \\bm{w} \\geq \\bm{B}`` is a set of linear constraints.
- ``\\bm{\\mu}`` is the vector of expected returns for each asset. 
- ``R(\\bm{w})`` is the return function from [`KellyRet`](@ref).
- ``U_{\\bm{\\mu}}`` is the uncertainty set for the asset returns, they can be:
```math
\\begin{align*}
U_{\\bm{\\mu}}^{\\mathrm{box}} &= \\left\\{\\bm{\\mu}\\, \\vert\\, \\vert \\bm{\\mu} - \\bm{\\hat{\\mu}} \\vert \\leq \\delta \\right\\}\\\\
U_{\\bm{\\mu}}^{\\mathrm{ellipse}} &= \\left\\{\\bm{\\mu}\\, \\vert\\, \\left(\\bm{\\mu} - \\bm{\\hat{\\mu}}\\right) \\mathbf{\\Sigma}_{\\bm{\\mu}}^{-1} \\left(\\bm{\\mu} - \\bm{\\hat{\\mu}}\\right)^{\\intercal} \\leq k_{\\bm{\\mu}}^{2}\\right\\}\\,.
\\end{align*}
```
- Where the following variables are estimated by assuming that the portfolio's asset mean returns can be generated by *some* distribution. The distribution is sampled, and the estimates are calculated from them. Available choices can be found in [`BoxTypes`](@ref) and [`EllipseTypes`](@ref) for the box and ellipse sets respectively:
    - ``\\hat{\\bm{\\mu}}`` the expected portfolio asset mean returns given the samples.
    - ``\\mathbf{\\Sigma}_{\\bm{\\mu}}`` is the covariance of the samples.
    - and ``k_{\\bm{\\mu}}`` is a significance parameter of the distribution.
- ``\\lambda`` is the risk aversion coefficient.
- and ``r`` is the risk-free rate.

The worst case uncertainty sets are computed by [`wc_statistics!`](@ref).
"""
const PortTypes = (:Trad, :RP, :RRP, :WC)

"""
```julia
RRPVersions = (:None, :Reg, :Reg_Pen)
```
Available versions of Relaxed Risk Parity Optimisations (see [`PortTypes`](@ref)).
- `:None`: no penalty.
- `:Reg`: regularisation constraint, ``\\rho``.
- `:Reg_Pen`: regularisation and penalisation constraints, ``\\lambda`` and ``\\rho``.
"""
const RRPVersions = (:None, :Reg, :Reg_Pen)

"""
```julia
UncertaintyTypes = (:None, :Box, :Ellipse)
```
Available types of uncertainty sets that can be computed with [`wc_statistics!`](@ref), which are used by Worst Case Mean Variance Optimisations (see [`PortTypes`](@ref), [`EllipseTypes`](@ref), and [`BoxTypes`](@ref)).
- `:Box`: are box uncertainty sets, ie the sets are full matrices.
- `:Ellipse`: are elliptical uncertainty sets, ie the sets are diagonal matrices.
"""
const UncertaintyTypes = (:None, :Box, :Ellipse)

"""
```julia
EllipseTypes = (:Stationary, :Circular, :Moving, :Normal)
```
Available types of elliptical sets that can be computed with [`wc_statistics!`](@ref), which are used by Worst Case Mean Variance Optimisations (see [`PortTypes`](@ref)).
- `:Stationary`: stationary bootstrapping method.
- `:Circular`: circular block bootstrapping method.
- `:Moving`: moving block bootstrapping method.
- `:Normal`: sets generated by assuming returns are normally distributed.
"""
const EllipseTypes = (:Stationary, :Circular, :Moving, :Normal)

"""
```julia
BoxTypes = (:Stationary, :Circular, :Moving, :Normal, :Delta)
```
Available types of box sets that can be computed with [`wc_statistics!`](@ref), which are used by Worst Case Mean Variance Optimisations (see [`PortTypes`](@ref)).
- `:Stationary`: stationary bootstrapping method.
- `:Circular`: circular block bootstrapping method.
- `:Moving`: moving block bootstrapping method.
- `:Normal`: normally distributed covariance and mean samples.
- `:Delta`: box sets are assumed to fall in the extrema of a well-defined interval.
"""
const BoxTypes = (EllipseTypes..., :Delta)

"""
```julia
KindBootstrap = (:Stationary, :Circular, :Moving)
```
Kind of bootstrap for computing the uncertainty sets with [`wc_statistics!`](@ref), which are used by Worst Case Mean Variance Optimisations (see [`PortTypes`](@ref)).
- `:Stationary`: stationary bootstrapping method.
- `:Circular`: circular block bootstrapping method.
- `:Moving`: moving block bootstrapping method.
"""
const KindBootstrap = (:Stationary, :Circular, :Moving)

"""
```julia
HCPortTypes = (:HRP, :HERC, :NCO)
```
Available optimisation types for [`HCPortfolio`](@ref).
- `:HRP`: Hierarchical Risk Parity.
- `:HERC`: Hierarhical Equal Risk Contribution.
- `:NCO`: Nested Clustered Optimisation.
Both `:HERC` and `:NCO` split their optimisations into two parts:
1. inter-cluster optimisation.
2. intra-cluster optimisation.
Threfore they can make use of extra parameters:
- `:HERC`: accepts an extra risk measure `rm_i` and OWA weights `owa_w_i` parameters, which are used for the intra-cluster optimisations. They default to the same value as their external counterparts.
- `:NCO`: accepts an extra objective function `obj_i` kelly return `kelly_i` risk aversion parameter `l_i` risk measure `rm_i` and OWA weights `owa_w_i` parameters. They default to the same value as their external counterparts.
"""
const HCPortTypes = (:HRP, :HERC, :NCO)

"""
```julia
MuTypes = (:Default, :JS, :BS, :BOP, :CAPM, :Custom_Func, :Custom_Val)
```
Methods for estimating the mean returns vector ``\\bm{\\mu}`` in [`mean_vec`](@ref).
- `:Default`: is the standard historical.
- `:JS`: James-Stein.
- `:BS`: Bayes-Stein.
- `:BOP`: Bodnar-Okhrin-Parolya.
- `:CAPM`: Capital Asset Pricing Model.
- `:Custom_Func`: custom function provided.
- `:Custom_Val`: custom value provided.
"""
const MuTypes = (:Default, :JS, :BS, :BOP, :CAPM, :Custom_Func, :Custom_Val)

"""
```julia
MuTargets = (:GM, :VW, :SE)
```
Targets for the `:JS`, `:BS` and `:BOP` estimators in [`mean_vec`](@ref) and [`mu_estimator`](@ref).
- `:GM`: grand mean.
- `:VW`: volatility-weighted grand mean.
- `:SE`: mean square error of sample mean.
"""
const MuTargets = (:GM, :VW, :SE)

"""
```julia
CovTypes = (:Full, :Semi, :Gerber0, :Gerber1, :Gerber2, :Custom_Func, :Custom_Val)
```
Methods for estimating the covariance matrix ``\\mathbf{\\Sigma}``.
- `:Full`: full covariance matrix.
- `:Semi`: semi-covariance matrix.
- `:Gerber0`: Gerber statistic 0.
- `:Gerber1`: Gerber statistic 1.
- `:Gerber2`: Gerber statistic 2.
- `:Custom_Func`: custom function provided.
- `:Custom_Val`: custom value provided.
"""
const CovTypes = (:Full, :Semi, :Gerber0, :Gerber1, :Gerber2, :Custom_Func, :Custom_Val)

"""
```julia
PosdefFixes = (:None, :Nearest, :Custom_Func)
```
Methods for fixing non-positive definite matrices.
- `:None`: no fix is applied.
- `:Nearest`: nearest correlation matrix.
- `Custom_Func`: custom function provided.
"""
const PosdefFixes = (:None, :Nearest, :Custom_Func)

"""
```julia
DenoiseMethods = (:Fixed, :Spectral, :Shrink)
```
Methods for matrix denoising.
- `:Fixed`: fixed.
- `:Spectral`: spectral.
- `:Shrink`: shrink.
"""
const DenoiseMethods = (:Fixed, :Spectral, :Shrink)

"""
```julia
RegCriteria = (:pval, GLM.aic, GLM.aicc, GLM.bic, GLM.r2, GLM.adjr2)
```
Criteria for feature selection in regression functions.
- `:pval`: p-value feature selection.
- The rest are methods applied to a fitted General Linear Model from [GLM.jl](https://juliastats.org/GLM.jl/stable/#Methods-applied-to-fitted-models).
"""
const RegCriteria = (:pval, GLM.aic, GLM.aicc, GLM.bic, GLM.r2, GLM.adjr2)

"""
```julia
FSType = (:FReg, :BReg, :PCR)
```
Methods for feature selection when creating the loadings matrix.
- `:FReg`: forward regression;
- `:Breg`: backward regression;
- `:PCR`: Principal Component Regression using [PCA](https://juliastats.org/MultivariateStats.jl/stable/pca/).
"""
const FSType = (:FReg, :BReg, :PCR)

"""
```julia
BLHist = (1, 2, 3)
```
Choice of what estimate of ``\\mathbf{\\Sigma}`` to use. Factor models, `:FM` and `:BLFM`, can also use a factor model estimate of the returns matrix ``\\mathbf{X}``.
- `1`:
    - `:FM`: uses the factor model estimates of ``\\mathbf{\\Sigma}`` and ``\\mathbf{X}``.
    - `:BL`: uses the Black Litterman estimate of ``\\mathbf{\\Sigma}``.
    - `:BLFM`: uses the Black Litterman factor model estimate of ``\\mathbf{\\Sigma}`` and factor model estimate of ``\\mathbf{X}``.
- `2`:
    - `:FM`: uses the standard estimates of ``\\mathbf{\\Sigma}`` and ``\\mathbf{X}``.
    - `:BL`: uses the standard estimate of ``\\mathbf{\\Sigma}``.
    - `:BLFM`: uses the standard estimates of ``\\mathbf{\\Sigma}`` and ``\\mathbf{X}``.
- `3`
    - `:FM` and `:BL` do not support this option.
    - `:BLFM`: uses the factor model estimates of ``\\mathbf{\\Sigma}`` and ``\\mathbf{X}``.
"""
const BLHist = (1, 2, 3)

"""
```julia
CodepTypes = (
    :Pearson,
    :Spearman,
    :Kendall,
    :Gerber0,
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
Methods for estimating the codependence (correlation) matrix ``\\mathbf{P}``, and distance matrix ``\\mathbf{D}``.
- `:Pearson`: Pearson correlation, ``\\mathbf{D}_{i,\\,j} = \\sqrt{\\dfrac{1}{2} \\left(1- \\mathbf{P}_{i,\\,j} \\right)}``.
- `:Spearman`: Spearman correlation, ``\\mathbf{D}_{i,\\,j} = \\sqrt{\\dfrac{1}{2} \\left(1- \\mathbf{P}_{i,\\,j} \\right)}``.
- `:Kendall`: Kendall correlation, ``\\mathbf{D}_{i,\\,j} = \\sqrt{\\dfrac{1}{2} \\left(1- \\mathbf{P}_{i,\\,j} \\right)}``.
- `:Gerber0`: Gerber statistic 0, ``\\mathbf{D}_{i,\\,j} = \\sqrt{\\dfrac{1}{2} \\left(1- \\mathbf{P}_{i,\\,j} \\right)}``.
- `:Gerber1`: Gerber statistic 1, ``\\mathbf{D}_{i,\\,j} = \\sqrt{\\dfrac{1}{2} \\left(1- \\mathbf{P}_{i,\\,j} \\right)}``.
- `:Gerber2`: Gerber statistic 2, ``\\mathbf{D}_{i,\\,j} = \\sqrt{\\dfrac{1}{2} \\left(1- \\mathbf{P}_{i,\\,j} \\right)}``.
- `:Abs_Pearson`: absolute value of the Pearson correlation, ``\\mathbf{D}_{i,\\,j} = \\sqrt{1 - \\left\\lvert\\mathbf{P}_{i,\\,j}\\right\\rvert}``.
- `:Abs_Spearman`: absolute value of the Spearman correlation, ``\\mathbf{D}_{i,\\,j} = \\sqrt{1 - \\left\\lvert\\mathbf{P}_{i,\\,j}\\right\\rvert}``.
- `:Abs_Kendall`: absolute value of the Kendall correlation, ``\\mathbf{D}_{i,\\,j} = \\sqrt{1 - \\left\\lvert\\mathbf{P}_{i,\\,j}\\right\\rvert}``.
- `:Distance`: distance correlation matrix, , ``\\mathbf{D}_{i,\\,j} = \\sqrt{1 - \\mathbf{P}_{i,\\,j}}``
- `:Mutual_Info`: mutual information matrix, ``\\mathbf{D}_{i,\\,j}`` is the variation information matrix.
- `:Tail`: lower tail dependence index matrix, ``\\mathbf{D}_{i,\\,j} = -\\log\\left(\\mathbf{P}_{i,\\,j}\\right)``
- `:Cov_to_Cor`: the covariance matrix is converted to a correlation matrix, the distance matrix is computed by a distance function which defaults to ``\\mathbf{D}_{i,\\,j} = \\sqrt{\\dfrac{1}{2} \\left(1- \\mathbf{P}_{i,\\,j} \\right)}``.
- `:Custom_Func`: custom function provided, the distance matrix is computed by a distance function which defaults to ``\\mathbf{D}_{i,\\,j} = \\sqrt{\\dfrac{1}{2} \\left(1- \\mathbf{P}_{i,\\,j} \\right)}``.
- `:Custom_Val`: custom value provided, the distance matrix is computed by a distance function which defaults to ``\\mathbf{D}_{i,\\,j} = \\sqrt{\\dfrac{1}{2} \\left(1- \\mathbf{P}_{i,\\,j} \\right)}``.
"""
const CodepTypes = (
    :Pearson,
    :Spearman,
    :Kendall,
    :Gerber0,
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
Linkage types available when optimising a [`HCPortfolio`](@ref).
- `:DBHT`: is Direct Bubble Hierarchical Tree clustering.
- The rest are linkage types supported by [Clustering.hclust](https://juliastats.org/Clustering.jl/stable/hclust.html#Clustering.hclust).
"""
const LinkageTypes = (:single, :complete, :average, :ward_presquared, :ward, :DBHT)

"""
```julia
BranchOrderTypes = (:optimal, :barjoseph, :r, :default)
```
Choice of algorithm for ordering hierarchical clustering dendrogram leaves and branches.
- `:default`: if linkage is `:DBHT`, the leaves and branches remain as the algorithm orders them. If any other linkage is used, they fall back to `:r` as that is their default according to [Clustering.hclust](https://juliastats.org/Clustering.jl/stable/hclust.html#Clustering.hclust).
- All other branch orders are as defined by [Clustering.hclust](https://juliastats.org/Clustering.jl/stable/hclust.html#Clustering.hclust).
"""
const BranchOrderTypes = (:optimal, :barjoseph, :r, :default)

"""
```julia
HRObjFuncs = (:Min_Risk, :Utility, :Sharpe, :Max_Ret, :Equal)
```
Objective funcions for `:NCO` [`HCPortTypes`](@ref) of [`HCPortfolio`](@ref). 
- `:Min_Risk`, `:Utility`, `:Sharpe` and `:Max_Ret`: optimise the sub-portfolios as `:Trad` [`PortTypes`](@ref) of [`Portfolio`](@ref) according to their respective definitions in [`ObjFuncs`](@ref). 
- `:Equal`: optimises the sub-portfolios as `:RP` optimisations with equal risk contribution per asset/cluster. We can't offer customiseable risk contributions because the size and composition of the clusters is initially unknown and depends on the chosen linkage method.
"""
const HRObjFuncs = (ObjFuncs..., :Equal)

"""
```julia
AllocTypes = (:LP, :Greedy)
```
Methods for allocating assets to an [`AbstractPortfolio`](@ref) according to the optimised weights and latest asset prices.
- `:LP`: uses MI(LP) optimisation to allocate the portfolio.
- `:Greedy`: uses a greedy iterative algorithm.
"""
const AllocTypes = (:LP, :Greedy)

const ASH = AverageShiftedHistograms
const NCM = NearestCorrelationMatrix

const RiskMeasureNames = (
    SD = "Standard Deviation",
    MAD = "Mean Absolute Deviation",
    SSD = "Semi Standard Deviation",
    FLPM = "First Lower Partial Moment",
    SLPM = "Second Lower Partial Moment",
    WR = "Worst Realisation",
    CVaR = "Conditional Value at Risk",
    EVaR = "Entropic Value at Risk",
    RVaR = "Relativistic Value at Risk",
    MDD = "Max Drawdown",
    ADD = "Average Drawdown",
    CDaR = "Conditional Drawdown at Risk",
    UCI = "Ulcer Index",
    EDaR = "Entropic Drawdown at Risk",
    RDaR = "Relativistic Drawdown at Risk",
    Kurt = "Square Root Kurtosis",
    SKurt = "Square Root Semi Kurtosis",
    GMD = "Gini Mean Difference",
    RG = "Range",
    RCVaR = "Conditional Value at Risk Range",
    TG = "Tail Gini",
    RTG = "Tail Gini Range",
    OWA = "Ordered Weight Average",
)

function _sigdom(sym::Symbol)
    return if sym == :a
        "alpha"
    elseif sym == :b
        "beta"
    end * " ∈ (0, 1)"
end

function _sigdef(msg::String, sym::Symbol)
    ab = if sym == :a
        "alpha"
    elseif sym == :b
        "beta"
    end

    return "- `$ab`: significance level of $msg, `$(_sigdom(sym))`."
end

function _isigdef(msg::String, sym::Symbol)
    alfbet, ab = if sym == :a
        "alpha", "a"
    elseif sym == :b
        "beta", "b"
    end

    return "- `$(alfbet)_i`: initial significance level of $msg, `0 < $(alfbet)_i < $(alfbet) < 1`.\n- `$(ab)_sim`: number of CVaRs to approximate the $msg, `$(ab)_sim > 0`."
end

function _ndef(sym::Symbol)
    return if sym == :a1
        "`N` is the number of assets"
    elseif sym == :a2
        "`Na` is the number of assets"
    elseif sym == :a3
        "``N`` is the number of assets"
    elseif sym == :f1
        "`N` is the number of factors"
    elseif sym == :f2
        "`Nf` is the number of factors"
    elseif sym == :c1
        "`Nc` is the number of constraints"
    end
end

function _tstr(sym::Symbol)
    if sym == :t1
        "`T` is the number of returns observations"
    elseif sym == :t2
        "``T`` is the number of returns observations"
    end
end

function _mudef(msg::String, sym::Symbol = :a2)
    if sym == :a2
        n = "Na"
    elseif sym == :f2
        n = "Nf"
    end

    "`$n×1` vector, where $(_ndef(sym))). Set the value of the $(msg) mean returns at instance construction. When choosing `:Custom_Val` in `mu_type`, this is the value of `mu` used, can also be set after a call to [`mean_vec`](@ref) to replace the old value with the new."
end

function _covdef(msg::String, sym::Symbol = :a2)
    if sym == :a2
        n = "Na"
    elseif sym == :f2
        n = "Nf"
    elseif sym == :a22
        n = "(Na×Na)"
        sym = :a2
    end

    "`$n×$n` matrix, where $(_ndef(sym)). Set the value of the $(msg) covariance matrix at instance construction. When choosing `:Custom_Val` in `cov_type`, this is the value of `cov` used by [`covar_mtx`](@ref)."
end

function _dircomp(msg::String)
    "Can be directly computed by $msg."
end

const _tdef = "- `T`: number of returns observations."
const _owaw = "- `w`: `T×1` ordered weight vector."
const _edst = "Empty concrete subtype of `AbstractDictionary`"

function _filled_by(msg::String)
    return "Filled by $msg."
end

function _assert_value_message(lo::Real, hi::Real, args...) end
function _assert_category_message(sym::Symbol, collection) end
function _assert_generic_message(cmp, message) end

export KellyRet,
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
    HCPortTypes,
    MuTypes,
    CovTypes,
    CodepTypes,
    LinkageTypes,
    BranchOrderTypes,
    HRObjFuncs,
    AllocTypes,
    RegCriteria,
    BLFMType,
    MuTargets,
    PosdefFixes,
    FSType,
    BLHist
