"""
```julia
TrackingErrKinds = (:Weights, :Returns)
```

Available kinds of tracking errors for [`Portfolio`]().

  - `:Weights`: provide a vector of asset weights which is used to compute the vector of benchmark returns,    - ``\\bm{b} = \\mathbf{X} \\bm{w}``,where ``\\bm{b}`` is the benchmark returns vector, ``\\mathbf{X}`` the ``T \\times{} N`` asset returns matrix, and ``\\bm{w}`` the asset weights vector.
  - `:Returns`: directly provide the vector of benchmark returns.
    The benchmark is then used as a reference to optimise a portfolio that tracks it up to a given error.
"""
const TrackingErrKinds = (:None, :Weights, :Returns)

"""
```julia
NetworkMethods = (:None, :SDP, :IP)
```

Methods for enforcing network constraints for optimising [`Portfolio`]().

  - `:None`: No network constraint is used.
  - `:SDP`: Semi-definite programming constraint.
  - `:IP`: Integer programming constraint.
"""
const NetworkMethods = (:None, :SDP, :IP)

"""
```julia
BLFMMethods = (:A, :B)
```

Versions of the factor Black-Litterman Model.

# `:A` -- Augmented Black-Litterman

# `:B`-- Bayesian Black-Litterman

Estimates the covariance and expected returns vector based on a modified Black-Litterman model which updates its views with Bayesian statistics [^BBL].

```math
\\begin{align*}
\\mathbf{\\Sigma} &= \\mathbf{B} \\mathbf{\\Sigma}_{F} \\mathbf{B}^{\\intercal} + \\mathbf{D} \\\\
\\mathbf{D} &= \\begin{cases}\\mathrm{Diagonal}\\left(\\mathrm{var}\\left(\\mathbf{X} - \\mathbf{F} \\mathbf{B}^{\\intercal},\\, \\mathrm{dims} = 1\\right)\\right) &\\quad \\mathrm{if~ flag = true}\\\\
\\mathbf{0} &\\quad \\mathrm{if~ flag = false}
\\end{cases}\\\\
\\overline{\\mathbf{\\Sigma}}_{F} &= \\left(\\mathbf{\\Sigma}_{F}^{-1} + \\mathbf{P}_{F}^{\\intercal} \\mathbf{\\Omega}_{F}^{-1} \\mathbf{P}_{F}\\right)^{-1}\\\\
\\mathbf{\\Omega}_{F} &= \\dfrac{1}{T}\\mathrm{Diagonal}\\left(\\mathbf{P}_{F} \\mathbf{\\Sigma}_{F} \\mathbf{P}_{F}^{\\intercal}\\right)\\\\
\\overline{\\bm{\\Pi}}_{F} &= \\overline{\\mathbf{\\Sigma}}_{F}^{-1} \\left(\\mathbf{\\Sigma}_{F}^{-1} \\bm{\\Pi}_{F} + \\mathbf{P}_{F}^{\\intercal} \\mathbf{\\Omega}_{F}^{-1} \\mathbf{Q}_{F}\\right)\\\\
\\bm{\\Pi}_{F} &= \\bm{\\mu}_{F} - r\\\\
\\mathbf{\\Sigma}_{\\mathrm{BF}} &= \\mathbf{\\Sigma}^{-1} \\mathbf{B} \\left( \\overline{\\mathbf{\\Sigma}}_{F}^{-1} + \\mathbf{B}^{\\intercal} \\mathbf{\\Sigma}^{-1} \\mathbf{B} \\right)^{-1}\\\\
\\mathbf{\\Sigma}_{\\mathrm{BLB}} &= \\left(\\mathbf{\\Sigma}^{-1} - \\mathbf{\\Sigma}_{\\mathrm{BF}} \\mathbf{B}^{\\intercal} \\mathbf{\\Sigma}^{-1}\\right)^{-1}\\\\
\\bm{\\mu}_{\\mathrm{BLB}} &= \\mathbf{\\Sigma}_{\\mathrm{BLB}} \\mathbf{\\Sigma}_{\\mathrm{BF}} \\overline{\\mathbf{\\Sigma}}_{F}^{-1} \\overline{\\bm{\\Pi}}_{F} + r
\\end{align*}
```

Where:

  - ``\\mathbf{\\Sigma}`` is the estimated asset covariance matrix computed using the factor model.
  - ``\\mathbf{B}`` is the loadings matrix.
  - ``\\mathbf{\\Sigma}_{F}`` is the covariance matrix of the factors.
  - ``\\mathbf{D}`` is a diagonal matrix constructed from the variances of the errors between the asset and factor returns corrected by the loadings matrix i.e. the variance is taken for all ``T`` timestamps of ``N_{a}`` assets. If a flag is false this matrix can be set to ``\\mathbf{0}``.
  - ``\\mathbf{X}`` is the `T×Na` matrix of asset returns, where `T` is the number of returns observations and `Na` the number of assets.
  - ``\\mathbf{F}`` is the `T×Nf` matrix of factor returns, where `T` is the number of returns observations and `Nf` the number of factors.
  - ``\\overline{\\mathbf{\\Sigma}}_{F}`` is the posterior covariance matrix of the factors after adjusting by the factor views.
  - ``\\mathbf{P}_{F}`` is the factor views matrix.
  - ``\\mathbf{\\Omega}_{F}`` is the covariance matrix of the of the factor views.
  - ``\\overline{\\bm{\\Pi}}_{F}`` is the posterior equilibrium excess returns of the factors after adjusting by the factor views.
  - ``\\bm{\\Pi}_{F}`` is the equilibrium excess returns vector of the factors.
  - ``\\bm{\\mu}_{F}`` is the expected returns vector of the factors.
  - ``r`` is the risk-free rate.
  - ``\\mathbf{Q}_{F}`` is the factor views returns matrix.
  - ``\\mathbf{\\Sigma}_{\\mathrm{BF}}`` is an intermediate covariance matrix.
  - ``\\mathbf{\\Sigma}_{\\mathrm{BLB}}`` is the posterior asset covariance matrix, aka the asset covariance matrix obtained via the Bayesian Black-Litterman model.
  - ``\\bm{\\mu}_{\\mathrm{BLB}}`` is the posterior asset expected returns vector, aka the asset returns vector obtained via the Bayesian Black-Litterman model.

[^BBL]: Petter Kolm, Gordon Ritter, "On the Bayesian interpretation of Black–Litterman", European Journal of Operational Research, Volume 258, Issue 2, 2017, Pages 564-572, ISSN 0377-2217, https://doi.org/10.1016/j.ejor.2016.10.027. (https://www.sciencedirect.com/science/article/pii/S037722171630861X)
"""
const BLFMMethods = (:A, :B)

"""
```julia
UncertaintyTypes = (:None, :Box, :Ellipse)
```

Available types of uncertainty sets that can be computed with [`wc_statistics!`](), which are used by Worst Case Mean Variance Optimisations (see [`PortTypes`](), [`EllipseMethods`](), and [`BoxMethods`]()).

  - `:Box`: are box uncertainty sets, ie the sets are full matrices.
  - `:Ellipse`: are elliptical uncertainty sets, ie the sets are diagonal matrices.
"""
const UncertaintyTypes = (:None, :Box, :Ellipse)

"""
```julia
RRPVersions = (:None, :Reg, :Reg_Pen)
```

Available versions of Relaxed Risk Parity Optimisations (see [`PortTypes`]()).

  - `:None`: no penalty.
  - `:Reg`: regularisation constraint, ``\\rho``.
  - `:Reg_Pen`: regularisation and penalisation constraints, ``\\lambda`` and ``\\rho``.
"""
const RRPVersions = (:None, :Reg, :Reg_Pen)

"""
```julia
EllipseMethods = (:Stationary, :Circular, :Moving, :Normal)
```

Available types of elliptical sets that can be computed with [`wc_statistics!`](), which are used by Worst Case Mean Variance Optimisations (see [`PortTypes`]()).

  - `:Stationary`: [stationary](https://bashtage.github.io/arch/bootstrap/generated/arch.bootstrap.StationaryBootstrap.html#arch.bootstrap.StationaryBootstrap).
  - `:Circular`: [circular block](https://bashtage.github.io/arch/bootstrap/generated/arch.bootstrap.CircularBlockBootstrap.html#arch.bootstrap.CircularBlockBootstrap).
  - `:Moving`: [moving block](https://bashtage.github.io/arch/bootstrap/generated/arch.bootstrap.MovingBlockBootstrap.html#arch.bootstrap.MovingBlockBootstrap).
  - `:Normal`: sets generated by assuming the returns are normally distributed.
"""
const EllipseMethods = (:Stationary, :Circular, :Moving, :Normal)

"""
```julia
BoxMethods = (:Stationary, :Circular, :Moving, :Normal, :Delta)
```

Available types of box sets that can be computed with [`wc_statistics!`](), which are used by Worst Case Mean Variance Optimisations (see [`PortTypes`]()).

  - `:Stationary`: [stationary](https://bashtage.github.io/arch/bootstrap/generated/arch.bootstrap.StationaryBootstrap.html#arch.bootstrap.StationaryBootstrap).
  - `:Circular`: [circular block](https://bashtage.github.io/arch/bootstrap/generated/arch.bootstrap.CircularBlockBootstrap.html#arch.bootstrap.CircularBlockBootstrap).
  - `:Moving`: [moving block](https://bashtage.github.io/arch/bootstrap/generated/arch.bootstrap.MovingBlockBootstrap.html#arch.bootstrap.MovingBlockBootstrap).
  - `:Normal`: sets generated by assuming the returns are normally distributed.
  - `:Delta`: the set limits are assumed to fall in the extrema of a well-defined interval parametrised by a percentage of the covariance matrix and mean returns vector.
"""
const BoxMethods = (EllipseMethods..., :Delta)

"""
```julia
BootstrapMethods = (:Stationary, :Circular, :Moving)
```

Bootstrapping method for computing the uncertainty sets with [`wc_statistics!`](), which are used by Worst Case Mean Variance Optimisations (see [`PortTypes`]()).

  - `:Stationary`: [stationary](https://bashtage.github.io/arch/bootstrap/generated/arch.bootstrap.StationaryBootstrap.html#arch.bootstrap.StationaryBootstrap).
  - `:Circular`: [circular block](https://bashtage.github.io/arch/bootstrap/generated/arch.bootstrap.CircularBlockBootstrap.html#arch.bootstrap.CircularBlockBootstrap).
  - `:Moving`: [moving block](https://bashtage.github.io/arch/bootstrap/generated/arch.bootstrap.MovingBlockBootstrap.html#arch.bootstrap.MovingBlockBootstrap).
"""
const BootstrapMethods = (:Stationary, :Circular, :Moving)

"""
```julia
MuMethods = (:Default, :JS, :BS, :BOP, :CAPM, :Custom_Func, :Custom_Val)
```

Methods for estimating the mean returns vector in [`mean_vec`]().

  - `:Default`: is the standard historical.
  - `:JS`: James-Stein.
  - `:BS`: Bayes-Stein.
  - `:BOP`: Bodnar-Okhrin-Parolya.
  - `:CAPM`: Capital Asset Pricing Model.
  - `:Custom_Func`: custom function provided.
  - `:Custom_Val`: custom value provided.
"""
const MuMethods = (:Default, :JS, :BS, :BOP, :CAPM, :Custom_Func, :Custom_Val)

"""
```julia
MuTargets = (:GM, :VW, :SE)
```

Targets for the `:JS`, `:BS` and `:BOP` estimators in [`mean_vec`]() and [`mu_estimator`]().

  - `:GM`: grand mean.
  - `:VW`: volatility-weighted grand mean.
  - `:SE`: mean square error of sample mean.
"""
const MuTargets = (:GM, :VW, :SE)

"""
```julia
CovMethods = (:Full, :Semi, :Gerber0, :Gerber1, :Gerber2, :Custom_Func, :Custom_Val)
```

Methods for estimating the covariance matrix in [`covar_mtx`]().

  - `:Full`: full covariance matrix. Uses the [`StatsBase.CovarianceEstimator`](https://juliastats.org/StatsBase.jl/stable/cov/#StatsBase.CovarianceEstimator) interface and [`StatsBase.cov`](https://juliastats.org/StatsBase.jl/stable/cov/#Statistics.cov-Tuple%7BCovarianceEstimator,%20AbstractMatrix%7D) method. Letting the user take advantage of weights and covariance estimation packages.
  - `:Semi`: lower semi-covariance matrix. Uses the [`StatsBase.CovarianceEstimator`](https://juliastats.org/StatsBase.jl/stable/cov/#StatsBase.CovarianceEstimator) interface and [`StatsBase.cov`](https://juliastats.org/StatsBase.jl/stable/cov/#Statistics.cov-Tuple%7BCovarianceEstimator,%20AbstractMatrix%7D) method. Letting the user take advantage of weights and covariance estimation packages. Also lets the user provide a threshold value below which returns are considered bad enough to be included in the lower semi-covariance.
  - `:Gerber0`: Gerber statistic 0 [^Gerber].
  - `:Gerber1`: Gerber statistic 1 [^Gerber].
  - `:Gerber2`: Gerber statistic 2 [^Gerber].
  - `:Custom_Func`: custom function provided.
  - `:Custom_Val`: custom value provided.

[^Gerber]: [Gerber, Sander and Markowitz, Harry and Ernst, Philip and Miao, Yinsen and Name, No and Sargen, Paul, The Gerber Statistic: A Robust Co-Movement Measure for Portfolio Optimization (July 4, 2021). Available at SSRN: https://ssrn.com/abstract=3880054 or http://dx.doi.org/10.2139/ssrn.3880054](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3880054)
"""
const CovMethods = (:Full, :Semi, :Gerber0, :Gerber1, :Gerber2, :Custom_Func, :Custom_Val)

"""
```julia
PosdefFixMethods = (:None, :Nearest, :Custom_Func)
```

Methods for fixing non-positive definite matrices.

  - `:None`: no fix is applied.
  - `:Nearest`: nearest correlation matrix.
  - `Custom_Func`: custom function provided.
"""
const PosdefFixMethods = (:None, :Nearest, :Custom_Func)

"""
```julia
DenoiseMethods = (:Fixed, :Spectral, :Shrink)
```

Methods for matrix denoising.

  - `:Fixed`: fixed.
  - `:Spectral`: spectral.
  - `:Shrink`: shrink.
"""
const DenoiseMethods = (:None, :Fixed, :Spectral, :Shrink)

"""
```julia
RegCriteria = (:pval, :aic, :aicc, :bic, :r2, :adjr2)
```

Criteria for feature selection in regression functions.

  - `:pval`: p-value feature selection.
  - The rest are methods applied to a fitted General Linear Model from [GLM.jl](https://juliastats.org/GLM.jl/stable/#Methods-applied-to-fitted-models).
"""
const RegCriteria = (:pval, :aic, :aicc, :bic, :r2, :adjr2)

"""
```julia
FSMethods = (:FReg, :BReg, :PCR)
```

Methods for feature selection when creating the loadings matrix.

  - `:FReg`: forward regression.
  - `:Breg`: backward regression.
  - `:PCR`: Principal Component Regression using [PCA](https://juliastats.org/MultivariateStats.jl/stable/pca/).
"""
const FSMethods = (:FReg, :BReg, :PCR)

"""
```julia
CorMethods = (:Pearson, :Spearman, :Kendall, :Gerber0, :Gerber1, :Gerber2, :Abs_Pearson,
              :Abs_Spearman, :Abs_Kendall, :Distance, :Mutual_Info, :Tail, :Cov_to_Cor,
              :Custom_Func, :Custom_Val)
```

Methods for estimating the correlation and distance matrices, ``\\mathbf{C}`` and ``\\mathbf{D}`` respectively, in [`cor_dist_mtx`]().

  - `:Pearson`: Pearson correlation, ``\\mathbf{D}_{i,\\,j} = \\sqrt{\\dfrac{1}{2} \\left(1- \\mathbf{C}_{i,\\,j} \\right)}``.
  - `:Spearman`: Spearman correlation, ``\\mathbf{D}_{i,\\,j} = \\sqrt{\\dfrac{1}{2} \\left(1- \\mathbf{C}_{i,\\,j} \\right)}``.
  - `:Kendall`: Kendall correlation, ``\\mathbf{D}_{i,\\,j} = \\sqrt{\\dfrac{1}{2} \\left(1- \\mathbf{C}_{i,\\,j} \\right)}``.
  - `:Gerber0`: Gerber statistic 0 [^Gerber], ``\\mathbf{D}_{i,\\,j} = \\sqrt{\\dfrac{1}{2} \\left(1- \\mathbf{C}_{i,\\,j} \\right)}``.
  - `:Gerber1`: Gerber statistic 1 [^Gerber], ``\\mathbf{D}_{i,\\,j} = \\sqrt{\\dfrac{1}{2} \\left(1- \\mathbf{C}_{i,\\,j} \\right)}``.
  - `:Gerber2`: Gerber statistic 2 [^Gerber], ``\\mathbf{D}_{i,\\,j} = \\sqrt{\\dfrac{1}{2} \\left(1- \\mathbf{C}_{i,\\,j} \\right)}``.
  - `:Abs_Pearson`: absolute value of the Pearson correlation, ``\\mathbf{D}_{i,\\,j} = \\sqrt{1 - \\left\\lvert\\mathbf{C}_{i,\\,j}\\right\\rvert}``.
  - `:Abs_Spearman`: absolute value of the Spearman correlation, ``\\mathbf{D}_{i,\\,j} = \\sqrt{1 - \\left\\lvert\\mathbf{C}_{i,\\,j}\\right\\rvert}``.
  - `:Abs_Kendall`: absolute value of the Kendall correlation, ``\\mathbf{D}_{i,\\,j} = \\sqrt{1 - \\left\\lvert\\mathbf{C}_{i,\\,j}\\right\\rvert}``.
  - `:Distance`: distance correlation matrix, ``\\mathbf{D}_{i,\\,j} = \\sqrt{1 - \\mathbf{C}_{i,\\,j}}``
  - `:Mutual_Info`: mutual information matrix, ``\\mathbf{D}_{i,\\,j}`` is the variation information matrix.
  - `:Tail`: lower tail dependence index matrix, ``\\mathbf{D}_{i,\\,j} = -\\log\\left(\\mathbf{C}_{i,\\,j}\\right)``
  - `:Cov_to_Cor`: the covariance matrix is converted to a correlation matrix, the distance matrix is computed by a distance function which defaults to ``\\mathbf{D}_{i,\\,j} = \\sqrt{\\dfrac{1}{2} \\left(1- \\mathbf{C}_{i,\\,j} \\right)}``.
  - `:Custom_Func`: custom function provided, the distance matrix is computed by a distance function which defaults to ``\\mathbf{D}_{i,\\,j} = \\sqrt{\\dfrac{1}{2} \\left(1- \\mathbf{C}_{i,\\,j} \\right)}``.
  - `:Custom_Val`: custom value provided, the distance matrix is computed by a distance function which defaults to ``\\mathbf{D}_{i,\\,j} = \\sqrt{\\dfrac{1}{2} \\left(1- \\mathbf{C}_{i,\\,j} \\right)}``.
"""
const CorMethods = (:Pearson, :Semi_Pearson, :Spearman, :Kendall, :Gerber0, :Gerber1,
                    :Gerber2, :Abs_Pearson, :Abs_Semi_Pearson, :Abs_Spearman, :Abs_Kendall,
                    :Distance, :Mutual_Info, :Tail, :Cov_to_Cor, :Custom_Func, :Custom_Val)

"""
```julia
BinMethods = (:KN, :FD, :SC, :HGR)
```

Methods for calculating optimal bin widths for the mutual and variational information matrices computed by [`mut_var_info_mtx`]().

  - `:KN`: [Knuth's](https://docs.astropy.org/en/stable/api/astropy.stats.knuth_bin_width.html).
  - `:FD`: [Freedman-Diaconis'](https://docs.astropy.org/en/stable/api/astropy.stats.freedman_bin_width.html).
  - `:SC`: [Scotts'](https://docs.astropy.org/en/stable/api/astropy.stats.scott_bin_width.html).
  - `:HGR`: Hacine-Gharbi and Ravier's.
"""
const BinMethods = (:KN, :FD, :SC, :HGR)

"""
```julia
KellyRet = (:None, :Approx, :Exact)
```

Available types of Kelly returns for [`Portfolio`]().

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

Objective functions available for use in `:Trad` and `:WC` optimisations of [`Portfolio`]() (see [`PortTypes`]()).
"""
const ObjFuncs = (:Min_Risk, :Utility, :Sharpe, :Max_Ret)

"""
```julia
ValidTermination = (MOI.OPTIMAL, MOI.ALMOST_OPTIMAL, MOI.LOCALLY_SOLVED,
                    MOI.ALMOST_LOCALLY_SOLVED, MOI.SOLUTION_LIMIT, MOI.OBJECTIVE_LIMIT)
```

Valid `JuMP` termination codes after optimising an instance of [`Portfolio`](). If the termination code is different to these, then the failures are logged in the `.fail` field of [`HCPortfolio`]() and [`Portfolio`]().
"""
const ValidTermination = (MOI.OPTIMAL, MOI.ALMOST_OPTIMAL, MOI.LOCALLY_SOLVED,
                          MOI.ALMOST_LOCALLY_SOLVED, MOI.SOLUTION_LIMIT,
                          MOI.OBJECTIVE_LIMIT)

"""
```julia
PortClasses = (:Classic, :FM, :BL, :BLFM)
```

Available choicees of summary parameters ``\\bm{\\mu}`` and ``\\bm{\\Sigma}`` [`Portfolio`]().

  - `:Classic`: when optimising with this option, ``\\bm{\\mu}`` and ``\\bm{\\Sigma}`` take their values from historical estimates computed by [`asset_statistics!`]().
  - `:FM`: when optimising with this option, ``\\bm{\\mu}`` and ``\\bm{\\Sigma}`` take their values from the factor model computed by [`factor_statistics!`]().
  - `:BL`: when optimising with this option, ``\\bm{\\mu}`` and ``\\bm{\\Sigma}`` take their values from the Black-Litterman model computed by [`black_litterman_statistics!`]().
  - `:BLFM`: when optimising with this option, ``\\bm{\\mu}`` and ``\\bm{\\Sigma}`` take their values from the factor Black-Litterman model computed by [`black_litterman_factor_satistics!`](). This model has two versions defined in [`BLFMMethods`]().
"""
const PortClasses = (:Classic, :FM, :BL, :BLFM)

"""
```julia
PortTypes = (:Trad, :RP, :RRP, :WC)
```

Available optimisation types for [`Portfolio`]().

# `:Trad` -- Traditional Optimisations

Available objective functions for `:Trad` optimisations. We can chose any of the objective functions in [`ObjFuncs`]() and risk measures in [`RiskMeasures`]().

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
  - ``\\phi_{i}`` is risk measure ``i`` from the set of available risk measures ``\\left\\{\\Phi\\right\\}`` (see [`RiskMeasures`]()).
  - ``\\mathbf{A} \\bm{w} \\geq \\bm{B}`` is a set of linear constraints.
  - ``c_{i}`` is the maximum acceptable value for risk measure ``\\phi_{i}`` of the optimised portfolio.
  - ``R(\\bm{w})`` is the return function from [`KellyRet`]().
  - ``\\overline{\\mu}`` is the minimum acceptable return of the optimised portfolio.
  - ``\\lambda`` is the risk aversion coefficient.
  - and ``r`` is the risk-free rate.

# `:RP` -- Risk Parity Optimisations

Optimises portfolios based on a vector of risk contributions per asset. We can chose any of the risk measures in [`RiskMeasures`]().

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
  - ``\\phi`` a risk measure from the set of available risk measures (see [`RiskMeasures`]()).
  - ``\\mathbf{A} \\bm{w} \\geq \\bm{B}`` is a set of linear constraints.
  - ``\\bm{b}`` is the vector of maximum allowable risk contribution per asset to the optimised portfolio.
  - ``c`` is an auxiliary variable.
  - ``R(\\bm{w})`` is the return function from [`KellyRet`]().
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
  - ``\\lambda`` is a penalty parameter for ``\\rho``, taken from the available choices in [`RRPVersions`]().
  - ``\\bm{\\zeta}`` is the vector of marginal risk for each asset.
  - ``b_{i}`` is the maximum allowable risk contribution for asset ``i``.
  - ``N`` is the number of assets.
  - ``R(\\bm{w})`` is the return function from [`KellyRet`]().
  - and ``\\overline{\\mu}`` is the minimum acceptable return of the optimised portfolio.

# `:WC` -- Worst Case Mean Variance Optimisations

Computes the worst case mean variance portfolio according to user-selected uncertainty sets (see [`UncertaintyTypes`]()) for the portfolio return and covariance. We can chose any of the objective functions in [`ObjFuncs`]().

  - `:Min_Risk`: worst case minimum risk mean-variance portfolio,

```math
  \\begin{align*}
  \\underset{\\bm{w}}{\\min} &\\qquad \\underset{\\mathbf{\\Sigma}\\, \\in\\, U_{\\mathbf{\\Sigma}}}{\\max}\\, \\bm{w}^{\\intercal}\\, \\mathbf{\\Sigma}\\, \\bm{w}\\\\
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

  - Where the following variables are estimated by assuming that the portfolio's asset return covariance can be generated by *some* matrix distribution. The distribution is sampled, and the estimates are calculated from them. Available choices can be found in [`BoxMethods`]() and [`EllipseMethods`]() for the box and ellipse sets respectively:

      + the ``\\mathrm{l}`` and ``\\mathrm{u}`` subscripts denote lower and upper bounds for the covariance matrix given the samples.
      + ``\\mathbf{\\Sigma}_{\\mathbf{\\Sigma}}`` is the covariance of the samples.
      + ``\\hat{\\mathbf{\\Sigma}}`` the expected covariance given the samples.
      + and ``k_{\\mathbf{\\Sigma}}`` is a significance parameter of the matrix distribution.

  - ``\\mathbf{A} \\bm{w} \\geq \\bm{B}`` is a set of linear constraints.
  - ``\\bm{\\mu}`` is the vector of expected returns for each asset.
  - ``R(\\bm{w})`` is the return function from [`KellyRet`]().
  - ``U_{\\bm{\\mu}}`` is the uncertainty set for the asset returns, they can be:

```math
\\begin{align*}
U_{\\bm{\\mu}}^{\\mathrm{box}} &= \\left\\{\\bm{\\mu}\\, \\vert\\, \\vert \\bm{\\mu} - \\bm{\\hat{\\mu}} \\vert \\leq \\delta \\right\\}\\\\
U_{\\bm{\\mu}}^{\\mathrm{ellipse}} &= \\left\\{\\bm{\\mu}\\, \\vert\\, \\left(\\bm{\\mu} - \\bm{\\hat{\\mu}}\\right) \\mathbf{\\Sigma}_{\\bm{\\mu}}^{-1} \\left(\\bm{\\mu} - \\bm{\\hat{\\mu}}\\right)^{\\intercal} \\leq k_{\\bm{\\mu}}^{2}\\right\\}\\,.
\\end{align*}
```

  - Where the following variables are estimated by assuming that the portfolio's asset mean returns can be generated by *some* distribution. The distribution is sampled, and the estimates are calculated from them. Available choices can be found in [`BoxMethods`]() and [`EllipseMethods`]() for the box and ellipse sets respectively:

      + ``\\hat{\\bm{\\mu}}`` the expected portfolio asset mean returns given the samples.
      + ``\\mathbf{\\Sigma}_{\\bm{\\mu}}`` is the covariance of the samples.
      + and ``k_{\\bm{\\mu}}`` is a significance parameter of the distribution.

  - ``\\lambda`` is the risk aversion coefficient.
  - and ``r`` is the risk-free rate.

The worst case uncertainty sets are computed by [`wc_statistics!`]().
"""
const PortTypes = (:Trad, :RP, :RRP, :WC)

"""
```julia
HCPortTypes = (:HRP, :HERC, :NCO)
```

Available optimisation types for [`HCPortfolio`]().

  - `:HRP`: Hierarchical Risk Parity.
  - `:HERC`: Hierarhical Equal Risk Contribution.
  - `:NCO`: Nested Clustered Optimisation.

Both `:HERC` and `:NCO` types classify assets into `k` clusters and split the optimisation into two sub-optimisations.

 1. Intra-cluster optimisation, where each cluster is optimised as its own independent portfolio.
 2. Inter-cluster optimisation, each cluster is treated as a synthetic asset, for which relevant statistics are computed. These are then optimised like a regular portfolio.

Both optimisations are combined to produce the final answer.
"""
const HCPortTypes = (:HRP, :HERC, :NCO)

"""
```julia
BLHist = (1, 2, 3)
```

Choice of estimate of the covariance matrix ``\\mathbf{\\Sigma}``, means vector ``\\bm{\\mu}``, and returns matrix ``\\mathbf{X}`` for optimising with different [`PortClasses`](@ref). Each different estimate is appended by the suffixes:

  - No suffix: standard, computed by [`asset_statistics!`]().
  - ``\\mathrm{fm}``: factor model, computed by [`factor_statistics!`]().
  - ``\\mathrm{bl}``: Black-Litterman model, computed by [`black_litterman_statistics!`]().
  - ``\\mathrm{bl, fm}``: Black-Litterman factor model, computed by [`black_litterman_factor_satistics!`]().

The choices are:

  - `:Classic`: `BLHist` has no effect, always use ``\\bm{\\mu}``, ``\\mathbf{\\Sigma}``, ``\\mathbf{X}``.

  - `:FM`: ``\\bm{\\mu}_{\\mathrm{fm}}``;

      + `1`: ``\\mathbf{\\Sigma}_\\mathrm{fm}``, ``\\mathbf{X}_\\mathrm{fm}``;
      + `2`: ``\\mathbf{\\Sigma}``, ``\\mathbf{X}``;
      + `3`: throws error.
  - `:BL`: ``\\bm{\\mu}_\\mathrm{bl}``, ``\\mathbf{X}``;

      + `1`: ``\\mathbf{\\Sigma}_\\mathrm{bl}``;
      + `2`: ``\\mathbf{\\Sigma}``;
      + `3`: throws error.
  - `:BLFM`: ``\\bm{\\mu}_\\mathrm{bl, fm}``;

      + `1`: ``\\mathbf{\\Sigma}_\\mathrm{bl, fm}``, ``\\mathbf{X}_\\mathrm{fm}``;
      + `2`: ``\\mathbf{\\Sigma}``, ``\\mathbf{X}``;
      + `3`: ``\\mathbf{\\Sigma}_\\mathrm{fm}``, ``\\mathbf{X}_\\mathrm{fm}``.
"""
const BLHist = (1, 2, 3)

"""
```julia
LinkageTypes = (:single, :complete, :average, :ward, :ward_presquared, :DBHT)
```

Linkage types available when optimising a [`HCPortfolio`]().

  - `:DBHT`: is Direct Bubble Hierarchical Tree clustering.
  - The rest are linkage types supported by [Clustering.hclust](https://juliastats.org/Clustering.jl/stable/hclust.html#Clustering.hclust).
"""
const LinkageTypes = (:single, :complete, :average, :ward_presquared, :ward, :DBHT)

"""
```julia
HCObjFuncs = (:Min_Risk, :Utility, :Sharpe, :Max_Ret, :Equal)
```

Objective funcions for `:NCO` [`HCPortTypes`]() of [`HCPortfolio`]().

  - `:Min_Risk`, `:Utility`, `:Sharpe` and `:Max_Ret`: optimise the sub-portfolios as `:Trad` [`PortTypes`]() of [`Portfolio`]() according to their respective definitions in [`ObjFuncs`]().
  - `:Equal`: optimises the sub-portfolios as `:RP` optimisations with equal risk contribution per asset/cluster. We can't offer customiseable risk contributions because the size and composition of the clusters is initially unknown and depends on the chosen linkage method.
"""
const HCObjFuncs = (ObjFuncs..., :Equal)

"""
```julia
AllocTypes = (:LP, :Greedy)
```

Methods for allocating assets to an [`AbstractPortfolio`]() according to the optimised weights and latest asset prices.

  - `:LP`: uses Mixed-Integer Linear Programming optimisation to allocate the portfolio.
  - `:Greedy`: uses a greedy iterative algorithm.
"""
const AllocTypes = (:LP, :Greedy)

const ASH = AverageShiftedHistograms
const NCM = NearestCorrelationMatrix

"""
```
RiskMeasureNames = (SD = "Standard Deviation", MAD = "Mean Absolute Deviation",
                    SSD = "Semi Standard Deviation",
                    FLPM = "First Lower Partial Moment",
                    SLPM = "Second Lower Partial Moment", WR = "Worst Realisation",
                    CVaR = "Conditional Value at Risk",
                    EVaR = "Entropic Value at Risk",
                    RVaR = "Relativistic Value at Risk",
                    MDD = "Uncompounded Max Drawdown",
                    ADD = "Uncompounded Average Drawdown",
                    CDaR = "Conditional Uncompounded Drawdown at Risk",
                    UCI = "Uncompounded Ulcer Index",
                    EDaR = "Entropic Uncompounded Drawdown at Risk",
                    RDaR = "Relativistic Uncompounded Drawdown at Risk",
                    Kurt = "Square Root Kurtosis",
                    SKurt = "Square Root Semi Kurtosis", GMD = "Gini Mean Difference",
                    RG = "Range", RCVaR = "Conditional Value at Risk Range",
                    TG = "Tail Gini", RTG = "Tail Gini Range",
                    OWA = "Ordered Weight Average", Variance = "Variance",
                    Equal = "Equal Risk", VaR = "Value at Risk",
                    DaR = "Uncompounded Drawdown at Risk",
                    DaR_r = "Compounded Drawdown at Risk",
                    MDD_r = "Compounded Max Drawdown",
                    ADD_r = "Compounded Average Drawdown",
                    CDaR_r = "Conditional Uncompounded Drawdown at Risk",
                    UCI_r = "Compounded Ulcer Index",
                    EDaR_r = "Entropic Compounded Drawdown at Risk",
                    RDaR_r = "Relativistic Compounded Drawdown at Risk")
```

Names of risk measures.
"""
const RiskMeasureNames = (SD = "Standard Deviation", MAD = "Mean Absolute Deviation",
                          SSD = "Semi Standard Deviation",
                          FLPM = "First Lower Partial Moment",
                          SLPM = "Second Lower Partial Moment", WR = "Worst Realisation",
                          CVaR = "Conditional Value at Risk",
                          EVaR = "Entropic Value at Risk",
                          RVaR = "Relativistic Value at Risk",
                          MDD = "Uncompounded Max Drawdown",
                          ADD = "Uncompounded Average Drawdown",
                          CDaR = "Conditional Uncompounded Drawdown at Risk",
                          UCI = "Uncompounded Ulcer Index",
                          EDaR = "Entropic Uncompounded Drawdown at Risk",
                          RDaR = "Relativistic Uncompounded Drawdown at Risk",
                          Kurt = "Square Root Kurtosis",
                          SKurt = "Square Root Semi Kurtosis", GMD = "Gini Mean Difference",
                          RG = "Range", RCVaR = "Conditional Value at Risk Range",
                          TG = "Tail Gini", RTG = "Tail Gini Range",
                          OWA = "Ordered Weight Average", Variance = "Variance",
                          Equal = "Equal Risk", VaR = "Value at Risk",
                          DaR = "Uncompounded Drawdown at Risk",
                          DaR_r = "Compounded Drawdown at Risk",
                          MDD_r = "Compounded Max Drawdown",
                          ADD_r = "Compounded Average Drawdown",
                          CDaR_r = "Conditional Uncompounded Drawdown at Risk",
                          UCI_r = "Compounded Ulcer Index",
                          EDaR_r = "Entropic Compounded Drawdown at Risk",
                          RDaR_r = "Relativistic Compounded Drawdown at Risk")

function _sigdom(sym::Symbol)
    return if sym == :a
        "alpha"
    elseif sym == :b
        "beta"
    end * " in (0, 1)"
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

    return "`$n×1` vector, where $(_ndef(sym))). Set the value of the $(msg) mean returns at instance construction. When choosing `:Custom_Val` in `mu_method`, this is the value of `mu` used, can also be set after a call to [`mean_vec`]() to replace the old value with the new."
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

    return "`$n×$n` matrix, where $(_ndef(sym)). Set the value of the $(msg) covariance matrix at instance construction. When choosing `:Custom_Val` in `cov_method`, this is the value of `cov` used by [`covar_mtx`]()."
end

function _dircomp(msg::String)
    return "Can be directly computed by $msg."
end

const _tdef = "- `T`: number of returns observations."
const _owaw = "- `w`: `T×1` ordered weight vector."
const _edst = "Empty concrete subtype of `AbstractDictionary`"
function _solver_desc(msg::String = "the `JuMP` model.", pref::String = "",
                      req::String = "")
    return """
           - `$(pref*"solvers")`: Provides the solvers and corresponding parameters for solving $msg `Dict` or `NamedTuple` with key value pairs where the values are other `Dict`s or `NamedTuple`s, e.g. `Dict(solver_key => Dict(...))`, the keys of the sub-dictionary/tuple must be:
               - `:solver`: which contains the JuMP optimiser.$(_solver_reqs(req))
               - `:params`: (optional) for the solver-specific parameters.
           """
end
function _solver_reqs(msg::String)
    return if isempty(msg)
        ""
    else
        " Solver must support $msg, or `JuMP` must be able to transform it/them into a supported form."
    end
end
function _filled_by(msg::String)
    return "This parameter is filled after calling $msg."
end

const _rmstr = """
                - `:SD`: standard deviation ([`SD`]()).$(_solver_reqs("`MOI.SecondOrderCone`"))
                - `:MAD`: max absolute deviation ([`MAD`]()).
                - `:SSD`: semi standard deviation ([`SSD`]()).$(_solver_reqs("`MOI.SecondOrderCone`"))
                - `:FLPM`: first lower partial moment (omega ratio) ([`FLPM`]()).
                - `:SLPM`: second lower partial moment (sortino ratio) ([`SLPM`]()).$(_solver_reqs("`MOI.SecondOrderCone`"))
                - `:WR`: worst realisation ([`WR`]()).
                - `:CVaR`: conditional value at risk ([`CVaR`]()).
                - `:EVaR`: entropic value at risk ([`EVaR`]()).$(_solver_reqs("`MOI.ExponentialCone`"))
                - `:RVaR`: relativistic value at risk ([`RVaR`]()).$(_solver_reqs("`MOI.PowerCone`"))
                - `:MDD`: maximum drawdown of uncompounded cumulative returns ([`MDD_abs`]()).
                - `:ADD`: average drawdown of uncompounded cumulative returns ([`ADD_abs`]()).
                - `:CDaR`: conditional drawdown at risk of uncompounded cumulative returns ([`CDaR_abs`]()).
                - `:UCI`: ulcer index of uncompounded cumulative returns ([`UCI_abs`]()).$(_solver_reqs("`MOI.SecondOrderCone`"))
                - `:EDaR`: entropic drawdown at risk of uncompounded cumulative returns ([`EDaR_abs`]()).$(_solver_reqs("`MOI.ExponentialCone`"))
                - `:RDaR`: relativistic drawdown at risk of uncompounded cumulative returns ([`RDaR_abs`]()).$(_solver_reqs("`MOI.PowerCone`"))
                - `:Kurt`: square root kurtosis ([`Kurt`]()).$(_solver_reqs("`MOI.PSDCone` and `MOI.SecondOrderCone`"))
                - `:SKurt`: square root semi-kurtosis ([`SKurt`]()).$(_solver_reqs("`MOI.PSDCone` and `MOI.SecondOrderCone`"))
                - `:GMD`: gini mean difference ([`GMD`]()).
                - `:RG`: range of returns ([`RG`]()).
                - `:RCVaR`: range of conditional value at risk ([`RCVaR`]()).
                - `:TG`: tail gini ([`TG`]()).
                - `:RTG`: range of tail gini ([`RTG`]()).
                - `:OWA`: ordered weight array (generic OWA weights) ([`OWA`]()).
               """

"""
```julia
RiskMeasures = (:SD, :MAD, :SSD, :FLPM, :SLPM, :WR, :CVaR, :EVaR, :RVaR, :MDD, :ADD,
                :CDaR, :UCI, :EDaR, :RDaR, :Kurt, :SKurt, :GMD, :RG, :RCVaR, :TG,
                :RTG, :OWA)
```
Available risk measures for `:Trad` and `:RP` type (see [`PortTypes`]()) of [`Portfolio`]().
$_rmstr
"""
const RiskMeasures = (:SD, :MAD, :SSD, :FLPM, :SLPM, :WR, :CVaR, :EVaR, :RVaR, :MDD, :ADD,
                      :CDaR, :UCI, :EDaR, :RDaR, :Kurt, :SKurt, :GMD, :RG, :RCVaR, :TG,
                      :RTG, :OWA)

"""
```julia
HCRiskMeasures = (:SD, :MAD, :SSD, :FLPM, :SLPM, :WR, :CVaR, :EVaR, :RVaR, :MDD, :ADD,
                  :CDaR, :UCI, :EDaR, :RDaR, :Kurt, :SKurt, :GMD, :RG, :RCVaR, :TG,
                  :RTG, :OWA, :Variance, :Equal, :VaR, :DaR, :DaR_r, :MDD_r, :ADD_r,
                  :CDaR_r, :UCI_r, :EDaR_r, :RDaR_r)
```
Available risk measures for optimisations of [`HCPortfolio`](). 

These risk measures are available for all optimisation types:
$_rmstr

These risk measures are not available with `:NCO` optimisations.
- `:Variance`: variance ([`Variance`]()).
- `:Equal`: equal risk contribution, `1/N` where N is the number of assets.
- `:VaR`: value at risk ([`VaR`]()).
- `:DaR`: drawdown at risk of uncompounded cumulative returns ([`DaR_abs`]()).
- `:DaR_r`: drawdown at risk of compounded cumulative returns ([`DaR_rel`]()).
- `:MDD_r`: maximum drawdown of compounded cumulative returns ([`MDD_rel`]()).
- `:ADD_r`: average drawdown of compounded cumulative returns ([`ADD_rel`]()).
- `:CDaR_r`: conditional drawdown at risk of compounded cumulative returns ([`CDaR_rel`]()).
- `:UCI_r`: ulcer index of compounded cumulative returns ([`UCI_rel`]()).$(_solver_reqs("`MOI.SecondOrderCone`"))
- `:EDaR_r`: entropic drawdown at risk of compounded cumulative returns ([`EDaR_rel`]()).$(_solver_reqs("`MOI.ExponentialCone`"))
- `:RDaR_r`: relativistic drawdown at risk of compounded cumulative returns ([`RDaR_rel`]()).$(_solver_reqs("`MOI.PowerCone`"))
"""
const HCRiskMeasures = (RiskMeasures..., :Variance, :Equal, :VaR, :DaR, :DaR_r, :MDD_r,
                        :ADD_r, :CDaR_r, :UCI_r, :EDaR_r, :RDaR_r)
