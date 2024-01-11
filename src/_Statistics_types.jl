"""
```julia
BLFMMethods = (:A, :B)
```

Versions of the factor Black-Litterman Model.

  - `:B`: Bayesian Black-Litterman, which uses the factors to generate the Black-Litterman estimates.
  - `:A`: Augmented Black-Litterman, which uses the factors to adjust the Black-Litterman views.
"""
const BLFMMethods = (:A, :B)

"""
```julia
UncertaintyTypes = (:None, :Box, :Ellipse)
```

Available types of uncertainty sets that can be computed with [`wc_statistics!`](@ref), which are used by Worst Case Mean Variance Optimisations (see [`PortTypes`](@ref), [`EllipseMethods`](@ref), and [`BoxMethods`](@ref)).

  - `:Box`: are box uncertainty sets, ie the sets are full matrices.
  - `:Ellipse`: are elliptical uncertainty sets, ie the sets are diagonal matrices.
"""
const UncertaintyTypes = (:None, :Box, :Ellipse)

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
EllipseMethods = (:Stationary, :Circular, :Moving, :Normal)
```

Available types of elliptical sets that can be computed with [`wc_statistics!`](@ref), which are used by Worst Case Mean Variance Optimisations (see [`PortTypes`](@ref)).

  - `:Stationary`: stationary bootstrapping method.
  - `:Circular`: circular block bootstrapping method.
  - `:Moving`: moving block bootstrapping method.
  - `:Normal`: sets generated by assuming returns are normally distributed.
"""
const EllipseMethods = (:Stationary, :Circular, :Moving, :Normal)

"""
```julia
BoxMethods = (:Stationary, :Circular, :Moving, :Normal, :Delta)
```

Available types of box sets that can be computed with [`wc_statistics!`](@ref), which are used by Worst Case Mean Variance Optimisations (see [`PortTypes`](@ref)).

  - `:Stationary`: stationary bootstrapping method.
  - `:Circular`: circular block bootstrapping method.
  - `:Moving`: moving block bootstrapping method.
  - `:Normal`: normally distributed covariance and mean samples.
  - `:Delta`: box sets are assumed to fall in the extrema of a well-defined interval.
"""
const BoxMethods = (EllipseMethods..., :Delta)

"""
```julia
BootstrapMethods = (:Stationary, :Circular, :Moving)
```

Kind of bootstrap for computing the uncertainty sets with [`wc_statistics!`](@ref), which are used by Worst Case Mean Variance Optimisations (see [`PortTypes`](@ref)).

  - `:Stationary`: stationary bootstrapping method.
  - `:Circular`: circular block bootstrapping method.
  - `:Moving`: moving block bootstrapping method.
"""
const BootstrapMethods = (:Stationary, :Circular, :Moving)

"""
```julia
MuMethods = (:Default, :JS, :BS, :BOP, :CAPM, :Custom_Func, :Custom_Val)
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
const MuMethods = (:Default, :JS, :BS, :BOP, :CAPM, :Custom_Func, :Custom_Val)

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
CovMethods = (:Full, :Semi, :Gerber0, :Gerber1, :Gerber2, :Custom_Func, :Custom_Val)
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
RegCriteria = (:pval, GLM.aic, GLM.aicc, GLM.bic, GLM.r2, GLM.adjr2)
```

Criteria for feature selection in regression functions.

  - `:pval`: p-value feature selection.
  - The rest are methods applied to a fitted General Linear Model from [GLM.jl](https://juliastats.org/GLM.jl/stable/#Methods-applied-to-fitted-models).
"""
const RegCriteria = (:pval, GLM.aic, GLM.aicc, GLM.bic, GLM.r2, GLM.adjr2)

"""
```julia
FSMethods = (:FReg, :BReg, :PCR)
```

Methods for feature selection when creating the loadings matrix.

  - `:FReg`: forward regression;- `:Breg`: backward regression;- `:PCR`: Principal Component Regression using [PCA](https://juliastats.org/MultivariateStats.jl/stable/pca/).
"""
const FSMethods = (:FReg, :BReg, :PCR)

"""
```julia
CorMethods = (:Pearson, :Spearman, :Kendall, :Gerber0, :Gerber1, :Gerber2, :Abs_Pearson,
              :Abs_Spearman, :Abs_Kendall, :Distance, :Mutual_Info, :Tail, :Cov_to_Cor,
              :Custom_Func, :Custom_Val)
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
  - `:Distance`: distance correlation matrix, ``\\mathbf{D}_{i,\\,j} = \\sqrt{1 - \\mathbf{P}_{i,\\,j}}``
  - `:Mutual_Info`: mutual information matrix, ``\\mathbf{D}_{i,\\,j}`` is the variation information matrix.
  - `:Tail`: lower tail dependence index matrix, ``\\mathbf{D}_{i,\\,j} = -\\log\\left(\\mathbf{P}_{i,\\,j}\\right)``
  - `:Cov_to_Cor`: the covariance matrix is converted to a correlation matrix, the distance matrix is computed by a distance function which defaults to ``\\mathbf{D}_{i,\\,j} = \\sqrt{\\dfrac{1}{2} \\left(1- \\mathbf{P}_{i,\\,j} \\right)}``.
  - `:Custom_Func`: custom function provided, the distance matrix is computed by a distance function which defaults to ``\\mathbf{D}_{i,\\,j} = \\sqrt{\\dfrac{1}{2} \\left(1- \\mathbf{P}_{i,\\,j} \\right)}``.
  - `:Custom_Val`: custom value provided, the distance matrix is computed by a distance function which defaults to ``\\mathbf{D}_{i,\\,j} = \\sqrt{\\dfrac{1}{2} \\left(1- \\mathbf{P}_{i,\\,j} \\right)}``.
"""
const CorMethods = (:Pearson, :Semi_Pearson, :Spearman, :Kendall, :Gerber0, :Gerber1,
                    :Gerber2, :Abs_Pearson, :Abs_Semi_Pearson, :Abs_Spearman, :Abs_Kendall,
                    :Distance, :Mutual_Info, :Tail, :Cov_to_Cor, :Custom_Func, :Custom_Val)

"""
```julia
BinMethods = (:KN, :FD, :SC, :HGR)
```

Methods for calculating optimal bin widths for the mutual and variational information matrices computed by [`mut_var_info_mtx`](@ref).

  - `:KN`: Knuth's choice.
  - `:FD`: Freedman-Diaconis' choice.
  - `:SC`: Schotts' choice.
  - `:HGR`: Hacine-Gharbi and Ravier's choice.
"""
const BinMethods = (:KN, :FD, :SC, :HGR)

@kwdef mutable struct GenericFunc
    func::Function = x -> x
    args::Tuple = ()
    kwargs::NamedTuple = (;)
end

@kwdef mutable struct CovEstSettings
    estimator::CovarianceEstimator = StatsBase.SimpleCovariance(; corrected = true)
    target_ret::Union{<:AbstractVector{<:Real},<:Real} = 0.0
    genfunc::GenericFunc = GenericFunc(; func = StatsBase.cov)
    custom::Union{<:AbstractMatrix{<:Real},Nothing} = nothing
end

mutable struct PosdefFixSettings
    method::Symbol
    genfunc::GenericFunc
end
function PosdefFixSettings(; method::Symbol = :Nearest,
                           genfunc::GenericFunc = GenericFunc(;),)
    @smart_assert(method in PosdefFixMethods)

    return PosdefFixSettings(method, genfunc)
end
function Base.setproperty!(obj::PosdefFixSettings, sym::Symbol, val)
    if sym == :method
        @smart_assert(val in PosdefFixMethods)
    end
    return setfield!(obj, sym, val)
end

mutable struct GerberSettings{T1 <: Real}
    threshold::T1
    genfunc::GenericFunc
    posdef::PosdefFixSettings
end
function GerberSettings(; threshold::Real = 0.5,
                        genfunc::GenericFunc = GenericFunc(; func = StatsBase.std,
                                                           kwargs = (; dims = 1)),
                        posdef::PosdefFixSettings = PosdefFixSettings(;),)
    @smart_assert(0 < threshold < 1)

    return GerberSettings{typeof(threshold)}(threshold, genfunc, posdef)
end
function Base.setproperty!(obj::GerberSettings, sym::Symbol, val)
    if sym == :threshold
        @smart_assert(0 < val < 1)
    end
    return setfield!(obj, sym, val)
end

mutable struct DenoiseSettings{T1 <: Real,T2 <: Integer,T3,T4 <: Integer,T5 <: Integer}
    method::Symbol
    alpha::T1
    detone::Bool
    mkt_comp::T2
    kernel::T3
    m::T4
    n::T5
    genfunc::GenericFunc
end
function DenoiseSettings(; method::Symbol = :None, alpha::Real = 0.0, detone::Bool = false,
                         mkt_comp::Integer = 1, kernel = ASH.Kernels.gaussian,
                         m::Integer = 10, n::Integer = 1000,
                         genfunc::GenericFunc = GenericFunc(; func = x -> nothing),)
    @smart_assert(method in DenoiseMethods)
    @smart_assert(0 <= alpha <= 1)

    return DenoiseSettings{typeof(alpha),typeof(mkt_comp),typeof(kernel),typeof(m),
                           typeof(n)}(method, alpha, detone, mkt_comp, kernel, m, n,
                                      genfunc)
end
function Base.setproperty!(obj::DenoiseSettings, sym::Symbol, val)
    if sym == :method
        @smart_assert(val in DenoiseMethods)
    elseif sym == :alpha
        @smart_assert(0 <= val <= 1)
    end
    return setfield!(obj, sym, val)
end

"""
```
CovSettings
```

  - `cov_method`: method for estimating the covariance matrices `cov`, `cov_fm`, `cov_bl`, `cov_bl_fm` in [`covar_mtx`](@ref), see [`CovMethods`](@ref) for available choices.
"""
mutable struct CovSettings
    # Cov method
    method::Symbol
    # Estimation
    estimation::CovEstSettings
    # Gerber
    gerber::GerberSettings
    # Denoise
    denoise::DenoiseSettings
    # Posdef fix
    posdef::PosdefFixSettings
    # J-LoGo
    jlogo::Bool
end
function CovSettings(; method::Symbol = :Full,
                     estimation::CovEstSettings = CovEstSettings(;),
                     gerber::GerberSettings = GerberSettings(;),
                     denoise::DenoiseSettings = DenoiseSettings(;),
                     posdef::PosdefFixSettings = PosdefFixSettings(;), jlogo::Bool = false,)
    @smart_assert(method in CovMethods)

    return CovSettings(method, estimation, gerber, denoise, posdef, jlogo)
end
function Base.setproperty!(obj::CovSettings, sym::Symbol, val)
    if sym == :method
        @smart_assert(val in CovMethods)
    end
    return setfield!(obj, sym, val)
end

"""
```
MuSettings
```

  - `mu_method`: method for estimating the mean returns vectors `mu`, `mu_fm`, `mu_bl`, `mu_bl_fm` in [`mean_vec`](@ref), see [`MuMethods`](@ref) for available choices.
"""
mutable struct MuSettings{T1 <: Real}
    method::Symbol
    target::Symbol
    rf::T1
    genfunc::GenericFunc
    custom::Union{<:AbstractVector{<:Real},Nothing}
    mkt_ret::Union{<:AbstractVector{<:Real},Nothing}
    sigma::Union{<:AbstractMatrix{<:Real},Nothing}
end
function MuSettings(; method::Symbol = :Default, target::Symbol = :GM, rf::Real = 0.0,
                    genfunc::GenericFunc = GenericFunc(; func = StatsBase.mean,
                                                       kwargs = (; dims = 1)),
                    custom::Union{<:AbstractVector{<:Real},Nothing} = nothing,
                    mkt_ret::Union{<:AbstractVector{<:Real},Nothing} = nothing,
                    sigma::Union{<:AbstractMatrix{<:Real},Nothing} = nothing,)
    @smart_assert(method in MuMethods)
    @smart_assert(target in MuTargets)

    return MuSettings{typeof(rf)}(method, target, rf, genfunc, custom, mkt_ret, sigma)
end
function Base.setproperty!(obj::MuSettings, sym::Symbol, val)
    if sym == :method
        @smart_assert(val in MuMethods)
    elseif sym == :target
        @smart_assert(val in MuTargets)
    end
    return setfield!(obj, sym, val)
end

@kwdef mutable struct KurtEstSettings
    target_ret::Union{<:AbstractVector{<:Real},<:Real} = 0.0
    custom_kurt::Union{<:AbstractMatrix{<:Real},Nothing} = nothing
    custom_skurt::Union{<:AbstractMatrix{<:Real},Nothing} = nothing
end
mutable struct KurtSettings
    # Estimation
    estimation::KurtEstSettings
    # Denoise
    denoise::DenoiseSettings
    # Posdef fix
    posdef::PosdefFixSettings
    # J-LoGo
    jlogo::Bool
end
function KurtSettings(; estimation::KurtEstSettings = KurtEstSettings(;),
                      denoise::DenoiseSettings = DenoiseSettings(;),
                      posdef::PosdefFixSettings = PosdefFixSettings(;),
                      jlogo::Bool = false,)
    return KurtSettings(estimation, denoise, posdef, jlogo)
end

mutable struct CorEstSettings{T1 <: Real}
    estimator::CovarianceEstimator
    alpha::T1
    bins_info::Union{Symbol,<:Integer}
    cor_genfunc::GenericFunc
    dist_genfunc::GenericFunc
    target_ret::Union{<:AbstractVector{<:Real},<:Real}
    custom_cor::Union{<:AbstractMatrix{<:Real},Nothing}
    custom_dist::Union{<:AbstractMatrix{<:Real},Nothing}
    sigma::Union{<:AbstractMatrix{<:Real},Nothing}
end
function CorEstSettings(;
                        estimator::CovarianceEstimator = StatsBase.SimpleCovariance(;
                                                                                    corrected = true),
                        alpha::Real = 0.05, bins_info::Union{Symbol,<:Integer} = :KN,
                        cor_genfunc::GenericFunc = GenericFunc(; func = StatsBase.cor),
                        dist_genfunc::GenericFunc = GenericFunc(;
                                                                func = x -> sqrt.(clamp!((1 .-
                                                                                          x) /
                                                                                         2,
                                                                                         0,
                                                                                         1)),),
                        target_ret::Union{<:AbstractVector{<:Real},<:Real} = 0.0,
                        custom_cor::Union{<:AbstractMatrix{<:Real},Nothing} = nothing,
                        custom_dist::Union{<:AbstractMatrix{<:Real},Nothing} = nothing,
                        sigma::Union{<:AbstractMatrix{<:Real},Nothing} = nothing,)
    @smart_assert(0 <= alpha <= 1)
    @smart_assert(bins_info in BinMethods ||
                  isa(bins_info, Int) && bins_info > zero(bins_info))
    # @smart_assert(
    #     size(custom_cor) == size(custom_dist) == size(sigma),    #     "size(custom_cor) == $(size(custom_cor)), size(custom_dist) == $(size(custom_dist)) and size(sigma) == $(size(sigma)), must all be equal"
    # )
    # @smart_assert(
    #     size(custom_cor, 1) == size(custom_cor, 2),    #     "custom_cor must be a square matrix, size(custom_cor) = $(size(custom_cor))"
    # )
    # @smart_assert(
    #     size(custom_dist, 1) == size(custom_dist, 2),    #     "custom_dist must be a square matrix, size(custom_dist) = $(size(custom_dist))"
    # )
    # @smart_assert(
    #     size(sigma, 1) == size(sigma, 2),    #     "sigma must be a square matrix, size(sigma) = $(size(sigma))"
    # )

    return CorEstSettings{typeof(alpha)}(estimator, alpha, bins_info, cor_genfunc,
                                         dist_genfunc, target_ret, custom_cor, custom_dist,
                                         sigma)
end
function Base.setproperty!(obj::CorEstSettings, sym::Symbol, val)
    if sym == :alpha
        @smart_assert(0 <= val <= 1)
    elseif sym == :bins_info
        @smart_assert(val in BinMethods || isa(val, Int) && val > zero(val))
        # elseif sym in (:custom_cor, :custom_dist, :sigma)
        # @smart_assert(
        #     size(obj.custom_cor) == size(obj.custom_dist) == size(obj.sigma),        #     "size(custom_cor) == $(size(obj.custom_cor)), size(custom_dist) == $(size(obj.custom_dist)) and size(sigma) == $(size(obj.sigma)), must all be equal"
        # )
        # @smart_assert(
        #     size(val, 1) == size(val, 2),        #     "$sym must be a square matrix, size($sym) = $(size(val))"
        # )
    end
    return setfield!(obj, sym, val)
end

mutable struct CorSettings
    # Cov method
    method::Symbol
    # Estimation
    estimation::CorEstSettings
    # Gerber
    gerber::GerberSettings
    # Denoise
    denoise::DenoiseSettings
    # Posdef fix
    posdef::PosdefFixSettings
    # J-LoGo
    jlogo::Bool
    # uplo
    uplo::Symbol
end
function CorSettings(; method::Symbol = :Pearson,
                     estimation::CorEstSettings = CorEstSettings(;),
                     gerber::GerberSettings = GerberSettings(;),
                     denoise::DenoiseSettings = DenoiseSettings(;),
                     posdef::PosdefFixSettings = PosdefFixSettings(;), jlogo::Bool = false,
                     uplo::Symbol = :L,)
    @smart_assert(method in CorMethods)

    return CorSettings(method, estimation, gerber, denoise, posdef, jlogo, uplo)
end
function Base.setproperty!(obj::CorSettings, sym::Symbol, val)
    if sym == :method
        @smart_assert(val in CorMethods)
    end
    return setfield!(obj, sym, val)
end

mutable struct WCSettings{T1 <: Real,T2 <: Real,T3 <: Real,T4,T5 <: Integer,T6 <: Integer}
    calc_box::Bool
    calc_ellipse::Bool
    box::Symbol
    ellipse::Symbol
    dcov::T1
    dmu::T2
    q::T3
    rng::T4
    seed::Union{<:Integer,Nothing}
    n_sim::T5
    window::T6
    posdef::PosdefFixSettings
end
function WCSettings(; calc_box::Bool = true, calc_ellipse::Bool = true,
                    box::Symbol = :Stationary, ellipse::Symbol = :Stationary,
                    dcov::Real = 0.1, dmu::Real = 0.1, q::Real = 0.05,
                    rng = Random.default_rng(), seed::Union{<:Integer,Nothing} = nothing,
                    n_sim::Integer = 3_000, window::Integer = 3,
                    posdef::PosdefFixSettings = PosdefFixSettings(;),)
    @smart_assert(box in BoxMethods)
    @smart_assert(ellipse in EllipseMethods)
    @smart_assert(0 < q < 1)

    return WCSettings{typeof(dcov),typeof(dmu),typeof(q),typeof(rng),typeof(n_sim),
                      typeof(window)}(calc_box, calc_ellipse, box, ellipse, dcov, dmu, q,
                                      rng, seed, n_sim, window, posdef)
end
function Base.setproperty!(obj::WCSettings, sym::Symbol, val)
    if sym == :box
        @smart_assert(val in BoxMethods)
    elseif sym == :ellipse
        @smart_assert(val in EllipseMethods)
    elseif sym == :q
        @smart_assert(0 < val < 1)
    end
    return setfield!(obj, sym, val)
end

mutable struct PCRSettings
    mean_genfunc::GenericFunc
    std_genfunc::GenericFunc
    pca_s_genfunc::GenericFunc
    pca_genfunc::GenericFunc
end
function PCRSettings(;
                     mean_genfunc::GenericFunc = GenericFunc(; func = StatsBase.mean,
                                                             kwargs = (; dims = 2)),
                     std_genfunc::GenericFunc = GenericFunc(; func = StatsBase.std,
                                                            kwargs = (; dims = 2)),
                     pca_s_genfunc::GenericFunc = GenericFunc(;
                                                              func = StatsBase.standardize,
                                                              args = (StatsBase.ZScoreTransform,),
                                                              kwargs = (; dims = 2),),
                     pca_genfunc::GenericFunc = GenericFunc(; func = MultivariateStats.fit,
                                                            args = (MultivariateStats.PCA,),),)
    return PCRSettings(mean_genfunc, std_genfunc, pca_s_genfunc, pca_genfunc)
end

mutable struct LoadingsSettings{T1 <: Real}
    method::Symbol
    criterion::Union{Symbol,Function}
    threshold::T1
    pcr_settings::PCRSettings
end
function LoadingsSettings(; method::Symbol = :FReg,
                          criterion::Union{Symbol,Function} = :pval, threshold::Real = 0.05,
                          pcr_settings::PCRSettings = PCRSettings(;),)
    @smart_assert(method in FSMethods)
    @smart_assert(criterion in RegCriteria)
    return LoadingsSettings{typeof(threshold)}(method, criterion, threshold, pcr_settings)
end
function Base.setproperty!(obj::LoadingsSettings, sym::Symbol, val)
    if sym == :method
        @smart_assert(val in FSMethods)
    elseif sym == :criterion
        @smart_assert(val in RegCriteria)
    end
    return setfield!(obj, sym, val)
end

mutable struct FactorSettings
    B::Union{DataFrame,Nothing}
    loadings_settings::LoadingsSettings
    error::Bool
    var_genfunc::GenericFunc
end
function FactorSettings(; B::Union{DataFrame,Nothing} = nothing,
                        loadings_settings::LoadingsSettings = LoadingsSettings(;),
                        error::Bool = true,
                        var_genfunc::GenericFunc = GenericFunc(; func = StatsBase.var,
                                                               kwargs = (; dims = 1)),)
    return FactorSettings(B, loadings_settings, error, var_genfunc)
end

mutable struct BLSettings{T1 <: Real}
    method::Symbol
    constant::Bool
    diagonal::Bool
    eq::Bool
    delta::Union{Nothing,<:Real}
    rf::T1
    var_genfunc::GenericFunc
end
function BLSettings(; method::Symbol = :B, constant::Bool = true, eq::Bool = true,
                    diagonal::Bool = true, delta::Real = 1.0, rf::Real = 0.0,
                    var_genfunc::GenericFunc = GenericFunc(; func = StatsBase.var,
                                                           kwargs = (; dims = 1)),)
    @smart_assert(method in BLFMMethods)

    return BLSettings{typeof(rf)}(method, constant, eq, diagonal, delta, rf, var_genfunc)
end
function Base.setproperty!(obj::BLSettings, sym::Symbol, val)
    if sym == :method
        @smart_assert(val in BLFMMethods)
    end
    return setfield!(obj, sym, val)
end

export CovSettings, CovEstSettings, GerberSettings, DenoiseSettings, PosdefFixSettings,
       GenericFunc, MuSettings, CorSettings, CorEstSettings, WCSettings, KurtSettings,
       PCRSettings, LoadingsSettings, FactorSettings
