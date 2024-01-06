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
UncertaintyTypes = (:None, :Box, :Ellipse)
```
Available types of uncertainty sets that can be computed with [`wc_statistics!`](@ref), which are used by Worst Case Mean Variance Optimisations (see [`PortTypes`](@ref), [`EllipseTypes`](@ref), and [`BoxTypes`](@ref)).
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
- `:Distance`: distance correlation matrix, ``\\mathbf{D}_{i,\\,j} = \\sqrt{1 - \\mathbf{P}_{i,\\,j}}``
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
BinTypes = (:KN, :FD, :SC, :HGR)
```
Methods for calculating optimal bin widths for the mutual and variational information matrices computed by [`mut_var_info_mtx`](@ref).
- `:KN`: Knuth's choice.
- `:FD`: Freedman-Diaconis' choice.
- `:SC`: Schotts' choice.
- `:HGR`: Hacine-Gharbi and Ravier's choice.
"""
const BinTypes = (:KN, :FD, :SC, :HGR)

@kwdef mutable struct GenericFunc
    func::Function = x -> x
    args::Tuple = ()
    kwargs::NamedTuple = (;)
end

@kwdef mutable struct CovEstSettings
    estimator::CovarianceEstimator = StatsBase.SimpleCovariance(; corrected = true)
    target_ret::Union{<:AbstractVector{<:Real}, <:Real} = 0.0
    genfunc::GenericFunc = GenericFunc(; func = StatsBase.cov)
    custom::Union{<:AbstractMatrix{<:Real}, Nothing} = nothing
end

mutable struct PosdefFixSettings
    method::Symbol
    genfunc::GenericFunc
end
function PosdefFixSettings(;
    method::Symbol = :Nearest,
    genfunc::GenericFunc = GenericFunc(;),
)
    @assert(method ∈ PosdefFixes, "method = $method, must be one of $PosdefFixes")

    return PosdefFixSettings(method, genfunc)
end
function Base.setproperty!(obj::PosdefFixSettings, sym::Symbol, val)
    if sym == :method
        @assert(val ∈ PosdefFixes, "$sym = $val, must be one of $PosdefFixes")
    end
    setfield!(obj, sym, val)
end

mutable struct GerberSettings{T1 <: Real}
    threshold::T1
    genfunc::GenericFunc
    posdef::PosdefFixSettings
end
function GerberSettings(;
    threshold::Real = 0.5,
    genfunc::GenericFunc = GenericFunc(; func = StatsBase.std, kwargs = (; dims = 1)),
    posdef::PosdefFixSettings = PosdefFixSettings(;),
)
    @assert(
        0 < threshold < 1,
        "threshold = $threshold, must be greater than 0 and less than 1"
    )

    return GerberSettings{typeof(threshold)}(threshold, genfunc, posdef)
end
function Base.setproperty!(obj::GerberSettings, sym::Symbol, val)
    if sym == :threshold
        @assert(0 < val < 1, "$sym = $val, must be greater than 0 and less than 1")
    end
    setfield!(obj, sym, val)
end

mutable struct DenoiseSettings{T1 <: Real, T2 <: Integer, T3, T4 <: Integer, T5 <: Integer}
    method::Symbol
    alpha::T1
    detone::Bool
    mkt_comp::T2
    kernel::T3
    m::T4
    n::T5
    genfunc::GenericFunc
end
function DenoiseSettings(;
    method::Symbol = :None,
    alpha::Real = 0.0,
    detone::Bool = false,
    mkt_comp::Integer = 1,
    kernel = ASH.Kernels.gaussian,
    m::Integer = 10,
    n::Integer = 1000,
    genfunc::GenericFunc = GenericFunc(; func = x -> nothing),
)
    @assert(method ∈ DenoiseMethods, "method = $method, must be one of $DenoiseMethods")
    @assert(0 <= alpha <= 1, "alpha = $alpha, must be 0 <= alpha <= 1")

    return DenoiseSettings{
        typeof(alpha),
        typeof(mkt_comp),
        typeof(kernel),
        typeof(m),
        typeof(n),
    }(
        method,
        alpha,
        detone,
        mkt_comp,
        kernel,
        m,
        n,
        genfunc,
    )
end
function Base.setproperty!(obj::DenoiseSettings, sym::Symbol, val)
    if sym == :method
        @assert(val ∈ DenoiseMethods, "$sym = $val, must be one of $DenoiseMethods")
    elseif sym == :alpha
        @assert(0 <= val <= 1, "$sym = $val, must be 0 <= alpha <= 1")
    end
    setfield!(obj, sym, val)
end

"""
```
CovSettings
```
- `cov_type`: method for estimating the covariance matrices `cov`, `cov_fm`, `cov_bl`, `cov_bl_fm` in [`covar_mtx`](@ref), see [`CovTypes`](@ref) for available choices.
"""
mutable struct CovSettings
    # Cov type
    type::Symbol
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
function CovSettings(;
    type::Symbol = :Full,
    estimation::CovEstSettings = CovEstSettings(;),
    gerber::GerberSettings = GerberSettings(;),
    denoise::DenoiseSettings = DenoiseSettings(;),
    posdef::PosdefFixSettings = PosdefFixSettings(;),
    jlogo::Bool = false,
)
    @assert(type ∈ CovTypes, "type = $type, must be one of $CovTypes")

    return CovSettings(type, estimation, gerber, denoise, posdef, jlogo)
end
function Base.setproperty!(obj::CovSettings, sym::Symbol, val)
    if sym == :type
        @assert(val ∈ CovTypes, "$sym = $val, must be one of $CovTypes")
    end
    setfield!(obj, sym, val)
end

"""
```
MuSettings
```
- `mu_type`: method for estimating the mean returns vectors `mu`, `mu_fm`, `mu_bl`, `mu_bl_fm` in [`mean_vec`](@ref), see [`MuTypes`](@ref) for available choices.
"""
mutable struct MuSettings{T1 <: Real}
    type::Symbol
    target::Symbol
    rf::T1
    genfunc::GenericFunc
    custom::Union{<:AbstractVector{<:Real}, Nothing}
    mkt_ret::Union{<:AbstractVector{<:Real}, Nothing}
    sigma::Union{<:AbstractMatrix{<:Real}, Nothing}
end
function MuSettings(;
    type::Symbol = :Default,
    target::Symbol = :GM,
    rf::Real = 0.0,
    genfunc::GenericFunc = GenericFunc(; func = StatsBase.mean, kwargs = (; dims = 1)),
    custom::Union{<:AbstractVector{<:Real}, Nothing} = nothing,
    mkt_ret::Union{<:AbstractVector{<:Real}, Nothing} = nothing,
    sigma::Union{<:AbstractMatrix{<:Real}, Nothing} = nothing,
)
    @assert(type ∈ MuTypes, "type = $type, must be one of $MuTypes")
    @assert(target ∈ MuTargets, "target = $target, must be one of $MuTargets")

    return MuSettings{typeof(rf)}(type, target, rf, genfunc, custom, mkt_ret, sigma)
end
function Base.setproperty!(obj::MuSettings, sym::Symbol, val)
    if sym == :type
        @assert(val ∈ MuTypes, "$sym = $val, must be one of $MuTypes")
    elseif sym == :target
        @assert(val ∈ MuTargets, "$sym = $val, must be one of $MuTargets")
    end
    setfield!(obj, sym, val)
end

@kwdef mutable struct KurtEstSettings
    target_ret::Union{<:AbstractVector{<:Real}, <:Real} = 0.0
    custom_kurt::Union{<:AbstractMatrix{<:Real}, Nothing} = nothing
    custom_skurt::Union{<:AbstractMatrix{<:Real}, Nothing} = nothing
end
mutable struct KurtSettings
    # Estimation
    estimation::KurtEstSettings
    # Gerber
    gerber::GerberSettings
    # Denoise
    denoise::DenoiseSettings
    # Posdef fix
    posdef::PosdefFixSettings
    # J-LoGo
    jlogo::Bool
end
function KurtSettings(;
    estimation::KurtEstSettings = KurtEstSettings(;),
    gerber::GerberSettings = GerberSettings(;),
    denoise::DenoiseSettings = DenoiseSettings(;),
    posdef::PosdefFixSettings = PosdefFixSettings(;),
    jlogo::Bool = false,
)
    return KurtSettings(estimation, gerber, denoise, posdef, jlogo)
end

mutable struct CorEstSettings{T1 <: Real}
    alpha::T1
    bins_info::Union{Symbol, <:Integer}
    cor_genfunc::GenericFunc
    dist_genfunc::GenericFunc
    custom_cor::Union{<:AbstractMatrix{<:Real}, Nothing}
    custom_dist::Union{<:AbstractMatrix{<:Real}, Nothing}
    sigma::Union{<:AbstractMatrix{<:Real}, Nothing}
end
function CorEstSettings(;
    alpha::Real = 0.05,
    bins_info::Union{Symbol, <:Integer} = :KN,
    cor_genfunc::GenericFunc = GenericFunc(; func = StatsBase.cor),
    dist_genfunc::GenericFunc = GenericFunc(;
        func = x -> sqrt.(clamp!((1 .- x) / 2, 0, 1)),
    ),
    custom_cor::Union{<:AbstractMatrix{<:Real}, Nothing} = nothing,
    custom_dist::Union{<:AbstractMatrix{<:Real}, Nothing} = nothing,
    sigma::Union{<:AbstractMatrix{<:Real}, Nothing} = nothing,
)
    @assert(
        0 <= alpha <= 1,
        "alpha = $alpha, must be greater than or equal to 0 and less than or equal to 1"
    )
    @assert(
        bins_info ∈ BinTypes || isa(bins_info, Int) && bins_info > zero(bins_info),
        "bins_info = $bins_info, has to either be in $BinTypes, or an integer value greater than 0"
    )
    # @assert(
    #     size(custom_cor) == size(custom_dist) == size(sigma),
    #     "size(custom_cor) == $(size(custom_cor)), size(custom_dist) == $(size(custom_dist)) and size(sigma) == $(size(sigma)), must all be equal"
    # )
    # @assert(
    #     size(custom_cor, 1) == size(custom_cor, 2),
    #     "custom_cor must be a square matrix, size(custom_cor) = $(size(custom_cor))"
    # )
    # @assert(
    #     size(custom_dist, 1) == size(custom_dist, 2),
    #     "custom_dist must be a square matrix, size(custom_dist) = $(size(custom_dist))"
    # )
    # @assert(
    #     size(sigma, 1) == size(sigma, 2),
    #     "sigma must be a square matrix, size(sigma) = $(size(sigma))"
    # )

    return CorEstSettings{typeof(alpha)}(
        alpha,
        bins_info,
        cor_genfunc,
        dist_genfunc,
        custom_cor,
        custom_dist,
        sigma,
    )
end
function Base.setproperty!(obj::CorEstSettings, sym::Symbol, val)
    if sym == :alpha
        @assert(
            0 <= val <= 1,
            "$sym = $val, must be greater than or equal to 0 and less than or equal to 1"
        )
    elseif sym == :bins_info
        @assert(
            val ∈ BinTypes || isa(val, Int) && val > zero(val),
            "$sym = $val, has to either be in $BinTypes, or an integer value greater than 0"
        )
        # elseif sym ∈ (:custom_cor, :custom_dist, :sigma)
        # @assert(
        #     size(obj.custom_cor) == size(obj.custom_dist) == size(obj.sigma),
        #     "size(custom_cor) == $(size(obj.custom_cor)), size(custom_dist) == $(size(obj.custom_dist)) and size(sigma) == $(size(obj.sigma)), must all be equal"
        # )
        # @assert(
        #     size(val, 1) == size(val, 2),
        #     "$sym must be a square matrix, size($sym) = $(size(val))"
        # )
    end
    setfield!(obj, sym, val)
end

mutable struct CorSettings
    # Cov type
    type::Symbol
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
function CorSettings(;
    type::Symbol = :Pearson,
    estimation::CorEstSettings = CorEstSettings(;),
    gerber::GerberSettings = GerberSettings(;),
    denoise::DenoiseSettings = DenoiseSettings(;),
    posdef::PosdefFixSettings = PosdefFixSettings(;),
    jlogo::Bool = false,
    uplo::Symbol = :L,
)
    @assert(type ∈ CodepTypes, "type = $type, must be one of $CodepTypes")

    return CorSettings(type, estimation, gerber, denoise, posdef, jlogo, uplo)
end
function Base.setproperty!(obj::CorSettings, sym::Symbol, val)
    if sym == :type
        @assert(val ∈ CodepTypes, "$sym = $val, must be one of $CodepTypes")
    end
    setfield!(obj, sym, val)
end

mutable struct WCSettings{
    T1 <: Real,
    T2 <: Real,
    T3 <: Real,
    T4,
    T5 <: Integer,
    T6 <: Integer,
}
    calc_box::Bool
    calc_ellipse::Bool
    box::Symbol
    ellipse::Symbol
    dcov::T1
    dmu::T2
    q::T3
    rng::T4
    seed::Union{<:Integer, Nothing}
    n_sim::T5
    window::T6
    posdef::PosdefFixSettings
end
function WCSettings(;
    calc_box::Bool = true,
    calc_ellipse::Bool = true,
    box::Symbol = :Stationary,
    ellipse::Symbol = :Stationary,
    dcov::Real = 0.1,
    dmu::Real = 0.1,
    q::Real = 0.05,
    rng = Random.default_rng(),
    seed::Union{<:Integer, Nothing} = nothing,
    n_sim::Integer = 3_000,
    window::Integer = 3,
    posdef::PosdefFixSettings = PosdefFixSettings(;),
)
    @assert(box ∈ BoxTypes, "box = $box, must be one of $BoxTypes")
    @assert(ellipse ∈ EllipseTypes, "ellipse = $ellipse, must be one of $EllipseTypes")
    @smart_assert(0 < q < 1)

    return WCSettings{
        typeof(dcov),
        typeof(dmu),
        typeof(q),
        typeof(rng),
        typeof(n_sim),
        typeof(window),
    }(
        calc_box,
        calc_ellipse,
        box,
        ellipse,
        dcov,
        dmu,
        q,
        rng,
        seed,
        n_sim,
        window,
        posdef,
    )
end
function Base.setproperty!(obj::WCSettings, sym::Symbol, val)
    if sym == :box
        @smart_assert(val ∈ BoxTypes, "$sym = $val, must be one of $BoxTypes")
    elseif sym == :ellipse
        @smart_assert(val ∈ EllipseTypes, "$sym = $val, must be one of $EllipseTypes")
    elseif sym == :q
        @smart_assert(0 < val < 1)
    end
    setfield!(obj, sym, val)
end

export CovSettings,
    CovEstSettings,
    GerberSettings,
    DenoiseSettings,
    PosdefFixSettings,
    GenericFunc,
    MuSettings,
    CorSettings,
    CorEstSettings,
    WCSettings
