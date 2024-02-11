"""
```julia
@kwdef mutable struct GenericFunction
    func::Union{Nothing, Function} = nothing
    args::Tuple = ()
    kwargs::NamedTuple = (;)
end
```

Structure and keyword constructor for storing functions, their arguments and keyword arguments.

# Inputs

  - `func`: function.
  - `args`: function args.
  - `kwargs`: function kwargs.
"""
mutable struct GenericFunction
    func::Union{Nothing, Function}
    args::Tuple
    kwargs::NamedTuple
end
function GenericFunction(; func::Union{Nothing, Function} = nothing, args::Tuple = (),
                         kwargs::NamedTuple = (;))
    return GenericFunction(func, args, kwargs)
end

"""
```julia
@kwdef mutable struct CovEstOpt
    estimator::CovarianceEstimator = StatsBase.SimpleCovariance(; corrected = true)
    target_ret::Union{<:AbstractVector{<:Real}, <:Real} = 0.0
    genfunc::GenericFunction = GenericFunction(; func = StatsBase.cov)
    custom::Union{<:AbstractMatrix{<:Real}, Nothing} = nothing
end
```

Structure and keyword constructor for storing the options for estimating covariance matrices.

# Inputs

  - `estimator`: covariance estimator as defined by [`StatsBase`](https://juliastats.org/StatsBase.jl/stable/cov/#StatsBase.CovarianceEstimator).

  - `target_ret`: target return for semicovariance estimation.
  - `genfunc`: [`GenericFunction`](@ref) for computing the covariance matrix.
  - `custom`: custom covariance function.
"""
mutable struct CovEstOpt
    estimator::CovarianceEstimator
    target_ret::Union{<:AbstractVector{<:Real}, <:Real}
    genfunc::GenericFunction
    custom::Union{<:AbstractMatrix{<:Real}, Nothing}
end
function CovEstOpt(;
                   estimator::CovarianceEstimator = StatsBase.SimpleCovariance(;
                                                                               corrected = true),
                   target_ret::Union{<:AbstractVector{<:Real}, <:Real} = 0.0,
                   genfunc::GenericFunction = GenericFunction(; func = StatsBase.cov),
                   custom::Union{<:AbstractMatrix{<:Real}, Nothing} = nothing)
    return CovEstOpt(estimator, target_ret, genfunc, custom)
end

"""
```julia
@kwdef mutable struct PosdefFixOpt
    method::Symbol = :Nearest
    genfunc::GenericFunction = GenericFunction(; func = x -> x)
end
```

Structure and keyword constructor for storing the options for fixing non-positive definite matrices.

# Inputs

  - `method`: method must be one of [`PosdefFixMethods`](@ref).
  - `genfunc`: [`GenericFunction`](@ref) when `method == :Custom`, for fixing non-positive definite matrices.
"""
mutable struct PosdefFixOpt
    method::Symbol
    genfunc::GenericFunction
end
function PosdefFixOpt(; method::Symbol = :Nearest,
                      genfunc::GenericFunction = GenericFunction(; func = x -> x))
    @smart_assert(method ∈ PosdefFixMethods)

    return PosdefFixOpt(method, genfunc)
end
function Base.setproperty!(obj::PosdefFixOpt, sym::Symbol, val)
    if sym == :method
        @smart_assert(val ∈ PosdefFixMethods)
    end
    return setfield!(obj, sym, val)
end

"""
```julia
@kwdef mutable struct GerberOpt{T1 <: Real}
    threshold::T1 = 0.5
    genfunc::GenericFunction = GenericFunction(; func = StatsBase.std,
                                               kwargs = (; dims = 1))
    posdef::PosdefFixOpt = PosdefFixOpt(;)
end
```

Structure and keyword constructor for storing the options for fixing non-positive definite matrices.

# Inputs

  - `threshold`: significance threshold for Gerber covariance matrix methods, must be ∈ (0, 1).
  - `genfunc`: [`GenericFunction`](@ref) for computing the standard deviation in Gerber covariance matrix methods.
  - `posdef`: [`PosdefFixOpt`](@ref) options for fixing non-positive definite matrices.
"""
mutable struct GerberOpt{T1 <: Real}
    threshold::T1
    genfunc::GenericFunction
    posdef::PosdefFixOpt
end
function GerberOpt(; threshold::Real = 0.5,
                   genfunc::GenericFunction = GenericFunction(; func = StatsBase.std,
                                                              kwargs = (; dims = 1)),
                   posdef::PosdefFixOpt = PosdefFixOpt(;))
    @smart_assert(zero(threshold) < threshold < one(threshold))

    return GerberOpt{typeof(threshold)}(threshold, genfunc, posdef)
end
function Base.setproperty!(obj::GerberOpt, sym::Symbol, val)
    if sym == :threshold
        @smart_assert(zero(val) < val < one(val))
    end
    return setfield!(obj, sym, val)
end

"""
```julia
@kwdef mutable struct DenoiseOpt{T1 <: Real, T2 <: Integer, T3, T4 <: Integer,
                                 T5 <: Integer}
    method::Symbol = :None
    alpha::T1 = 0.0
    detone::Bool = false
    mkt_comp::T2 = 1
    kernel::T3 = AverageShiftedHistograms.Kernels.gaussian
    m::T4 = 10
    n::T5 = 1000
    genfunc::GenericFunction = GenericFunction(;)
end
```

Structure and keyword constructor for storing the options for denoising matrices.

# Inputs

  - `method`: method for denoising matrices, must be in [`DenoiseMethods`](@ref).
  - `alpha`: shrink method significance level, must be ∈ (0, 1).
  - `detone`: if `true`, take only the largest `mkt_comp` eigenvalues from the correlation matrix.
  - `mkt_comp`: the number of largest eigenvalues to keep from the correlation matrix.
  - `kernel`: kernel for fitting the average shifted histograms according to the covariance matrix's kernel density.
  - `m`: number of adjacent histograms to smooth over.
  - `n`: number of points used when creating the range of values to which the average shifted histogram is to be fitted.
  - `genfunc`: only `genfunc.args` and `genfunc.kwargs` are used. These are the `args` and `kwargs` passed to `Optim.optimize`. This is used for finding the eigenvalue that minimises the residual error between the fitted average shifted histogram and an idealised state. Eigenvalues larger than this are considered significant.

!!! warning

    Keeping only the largest eigenvalues from a denoised covariance matrix may yield a singular matrix. In other words, if `detone = true` and `mkt_comp < N` where the covariance matrix is an `N×N` matrix, may yield a matrix that cannot be inverted.
"""
mutable struct DenoiseOpt{T1 <: Real, T2 <: Integer, T3, T4 <: Integer, T5 <: Integer}
    method::Symbol
    alpha::T1
    detone::Bool
    mkt_comp::T2
    kernel::T3
    m::T4
    n::T5
    genfunc::GenericFunction
end
function DenoiseOpt(; method::Symbol = :None, alpha::Real = 0.0, detone::Bool = false,
                    mkt_comp::Integer = 1, kernel = ASH.Kernels.gaussian, m::Integer = 10,
                    n::Integer = 1000, genfunc::GenericFunction = GenericFunction(;))
    @smart_assert(method ∈ DenoiseMethods)
    @smart_assert(zero(alpha) <= alpha <= one(alpha))

    return DenoiseOpt{typeof(alpha), typeof(mkt_comp), typeof(kernel), typeof(m),
                      typeof(n)}(method, alpha, detone, mkt_comp, kernel, m, n, genfunc)
end
function Base.setproperty!(obj::DenoiseOpt, sym::Symbol, val)
    if sym == :method
        @smart_assert(val ∈ DenoiseMethods)
    elseif sym == :alpha
        @smart_assert(zero(val) <= val <= one(val))
    end
    return setfield!(obj, sym, val)
end

"""
```
CovOpt
```

  - `cov_method`: method for estimating the covariance matrices `cov`, `cov_fm`, `cov_bl`, `cov_bl_fm` in [`covar_mtx`](), see [`CovMethods`]() for available choices.
"""
mutable struct CovOpt
    # Cov method
    method::Symbol
    # Estimation
    estimation::CovEstOpt
    # Gerber
    gerber::GerberOpt
    # Denoise
    denoise::DenoiseOpt
    # Posdef fix
    posdef::PosdefFixOpt
    # J-LoGo
    jlogo::Bool
end
function CovOpt(; method::Symbol = :Full, estimation::CovEstOpt = CovEstOpt(;),
                gerber::GerberOpt = GerberOpt(;), denoise::DenoiseOpt = DenoiseOpt(;),
                posdef::PosdefFixOpt = PosdefFixOpt(;), jlogo::Bool = false)
    @smart_assert(method ∈ CovMethods)

    return CovOpt(method, estimation, gerber, denoise, posdef, jlogo)
end
function Base.setproperty!(obj::CovOpt, sym::Symbol, val)
    if sym == :method
        @smart_assert(val ∈ CovMethods)
    end
    return setfield!(obj, sym, val)
end

"""
```
MuOpt
```

  - `mu_method`: method for estimating the mean returns vectors `mu`, `mu_fm`, `mu_bl`, `mu_bl_fm` in [`mean_vec`](), see [`MuMethods`]() for available choices.
"""
mutable struct MuOpt{T1 <: Real}
    method::Symbol
    target::Symbol
    rf::T1
    genfunc::GenericFunction
    custom::Union{<:AbstractVector{<:Real}, Nothing}
    mkt_ret::Union{<:AbstractVector{<:Real}, Nothing}
    sigma::Union{<:AbstractMatrix{<:Real}, Nothing}
end
function MuOpt(; method::Symbol = :Default, target::Symbol = :GM, rf::Real = 0.0,
               genfunc::GenericFunction = GenericFunction(; func = StatsBase.mean,
                                                          kwargs = (; dims = 1)),
               custom::Union{<:AbstractVector{<:Real}, Nothing} = nothing,
               mkt_ret::Union{<:AbstractVector{<:Real}, Nothing} = nothing,
               sigma::Union{<:AbstractMatrix{<:Real}, Nothing} = nothing)
    @smart_assert(method ∈ MuMethods)
    @smart_assert(target ∈ MuTargets)

    return MuOpt{typeof(rf)}(method, target, rf, genfunc, custom, mkt_ret, sigma)
end
function Base.setproperty!(obj::MuOpt, sym::Symbol, val)
    if sym == :method
        @smart_assert(val ∈ MuMethods)
    elseif sym == :target
        @smart_assert(val ∈ MuTargets)
    end
    return setfield!(obj, sym, val)
end

"""
```
@kwdef mutable struct KurtEstOpt
    target_ret::Union{<:AbstractVector{<:Real},<:Real} = 0.0
    custom_kurt::Union{<:AbstractMatrix{<:Real},Nothing} = nothing
    custom_skurt::Union{<:AbstractMatrix{<:Real},Nothing} = nothing
end
```
"""
mutable struct KurtEstOpt
    target_ret::Union{<:AbstractVector{<:Real}, <:Real}
    custom_kurt::Union{<:AbstractMatrix{<:Real}, Nothing}
    custom_skurt::Union{<:AbstractMatrix{<:Real}, Nothing}
end
function KurtEstOpt(; target_ret::Union{<:AbstractVector{<:Real}, <:Real} = 0.0,
                    custom_kurt::Union{<:AbstractMatrix{<:Real}, Nothing} = nothing,
                    custom_skurt::Union{<:AbstractMatrix{<:Real}, Nothing} = nothing)
    return KurtEstOpt(target_ret, custom_kurt, custom_skurt)
end

mutable struct KurtOpt
    # Estimation
    estimation::KurtEstOpt
    # Denoise
    denoise::DenoiseOpt
    # Posdef fix
    posdef::PosdefFixOpt
    # J-LoGo
    jlogo::Bool
end
function KurtOpt(; estimation::KurtEstOpt = KurtEstOpt(;),
                 denoise::DenoiseOpt = DenoiseOpt(;),
                 posdef::PosdefFixOpt = PosdefFixOpt(;), jlogo::Bool = false)
    return KurtOpt(estimation, denoise, posdef, jlogo)
end

mutable struct CorEstOpt{T1 <: Real, T2 <: AbstractMatrix{<:Real},
                         T3 <: AbstractMatrix{<:Real}, T4 <: AbstractMatrix{<:Real}}
    estimator::CovarianceEstimator
    alpha::T1
    bins_info::Union{Symbol, <:Integer}
    cor_genfunc::GenericFunction
    dist_genfunc::GenericFunction
    target_ret::Union{<:AbstractVector{<:Real}, <:Real}
    custom_cor::T2
    custom_dist::T3
    sigma::T4
end
function CorEstOpt(;
                   estimator::CovarianceEstimator = StatsBase.SimpleCovariance(;
                                                                               corrected = true),
                   alpha::Real = 0.05, bins_info::Union{Symbol, <:Integer} = :KN,
                   cor_genfunc::GenericFunction = GenericFunction(; func = StatsBase.cor),
                   dist_genfunc::GenericFunction = GenericFunction(;
                                                                   func = x -> sqrt.(clamp!((1 .-
                                                                                             x) /
                                                                                            2,
                                                                                            0,
                                                                                            1))),
                   target_ret::Union{<:AbstractVector{<:Real}, <:Real} = 0.0,
                   custom_cor::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
                   custom_dist::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
                   sigma::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0))
    @smart_assert(0 <= alpha <= 1)
    @smart_assert(bins_info ∈ BinMethods ||
                  isa(bins_info, Int) && bins_info > zero(bins_info))
    if !isempty(custom_cor)
        @smart_assert(size(custom_cor, 1) == size(custom_cor, 2))
    end
    if !isempty(custom_dist)
        @smart_assert(size(custom_dist, 1) == size(custom_dist, 2))
    end
    if !isempty(sigma)
        @smart_assert(size(custom_dist, 1) == size(custom_dist, 2))
    end

    return CorEstOpt{typeof(alpha), typeof(custom_cor), typeof(custom_dist), typeof(sigma)}(estimator,
                                                                                            alpha,
                                                                                            bins_info,
                                                                                            cor_genfunc,
                                                                                            dist_genfunc,
                                                                                            target_ret,
                                                                                            custom_cor,
                                                                                            custom_dist,
                                                                                            sigma)
end
function Base.setproperty!(obj::CorEstOpt, sym::Symbol, val)
    if sym == :alpha
        @smart_assert(0 <= val <= 1)
    elseif sym == :bins_info
        @smart_assert(val ∈ BinMethods || isa(val, Int) && val > zero(val))
    elseif sym == :custom_cor
        if !isempty(val)
            @smart_assert(size(val, 1) == size(val, 2))
        end
        val = convert(typeof(getfield(obj, sym)), val)
    elseif sym == :custom_dist
        if !isempty(val)
            @smart_assert(size(val, 1) == size(val, 2))
        end
        val = convert(typeof(getfield(obj, sym)), val)
    elseif sym == :sigma
        if !isempty(val)
            @smart_assert(size(val, 1) == size(val, 2))
        end
        val = convert(typeof(getfield(obj, sym)), val)
    end
    return setfield!(obj, sym, val)
end

mutable struct CorOpt
    # Cov method
    method::Symbol
    # Estimation
    estimation::CorEstOpt
    # Gerber
    gerber::GerberOpt
    # Denoise
    denoise::DenoiseOpt
    # Posdef fix
    posdef::PosdefFixOpt
    # J-LoGo
    jlogo::Bool
    # uplo
    uplo::Symbol
end
function CorOpt(; method::Symbol = :Pearson, estimation::CorEstOpt = CorEstOpt(;),
                gerber::GerberOpt = GerberOpt(;), denoise::DenoiseOpt = DenoiseOpt(;),
                posdef::PosdefFixOpt = PosdefFixOpt(;), jlogo::Bool = false,
                uplo::Symbol = :L)
    @smart_assert(method ∈ CorMethods)

    return CorOpt(method, estimation, gerber, denoise, posdef, jlogo, uplo)
end
function Base.setproperty!(obj::CorOpt, sym::Symbol, val)
    if sym == :method
        @smart_assert(val ∈ CorMethods)
    end
    return setfield!(obj, sym, val)
end

mutable struct WCOpt{T1 <: Real, T2 <: Real, T3 <: Real, T4, T5 <: Integer, T6 <: Integer}
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
    posdef::PosdefFixOpt
end
function WCOpt(; calc_box::Bool = true, calc_ellipse::Bool = true,
               box::Symbol = :Stationary, ellipse::Symbol = :Stationary, dcov::Real = 0.1,
               dmu::Real = 0.1, q::Real = 0.05, rng = Random.default_rng(),
               seed::Union{<:Integer, Nothing} = nothing, n_sim::Integer = 3_000,
               window::Integer = 3, posdef::PosdefFixOpt = PosdefFixOpt(;))
    @smart_assert(box ∈ BoxMethods)
    @smart_assert(ellipse ∈ EllipseMethods)
    @smart_assert(0 < q < 1)

    return WCOpt{typeof(dcov), typeof(dmu), typeof(q), typeof(rng), typeof(n_sim),
                 typeof(window)}(calc_box, calc_ellipse, box, ellipse, dcov, dmu, q, rng,
                                 seed, n_sim, window, posdef)
end
function Base.setproperty!(obj::WCOpt, sym::Symbol, val)
    if sym == :box
        @smart_assert(val ∈ BoxMethods)
    elseif sym == :ellipse
        @smart_assert(val ∈ EllipseMethods)
    elseif sym == :q
        @smart_assert(0 < val < 1)
    end
    return setfield!(obj, sym, val)
end

mutable struct PCROpt
    mean_genfunc::GenericFunction
    std_genfunc::GenericFunction
    pca_s_genfunc::GenericFunction
    pca_genfunc::GenericFunction
end
function PCROpt(;
                mean_genfunc::GenericFunction = GenericFunction(; func = StatsBase.mean,
                                                                kwargs = (; dims = 2)),
                std_genfunc::GenericFunction = GenericFunction(; func = StatsBase.std,
                                                               kwargs = (; dims = 2)),
                pca_s_genfunc::GenericFunction = GenericFunction(;
                                                                 func = StatsBase.standardize,
                                                                 args = (StatsBase.ZScoreTransform,),
                                                                 kwargs = (; dims = 2)),
                pca_genfunc::GenericFunction = GenericFunction(;
                                                               func = MultivariateStats.fit,
                                                               args = (MultivariateStats.PCA,)))
    return PCROpt(mean_genfunc, std_genfunc, pca_s_genfunc, pca_genfunc)
end

mutable struct LoadingsOpt{T1 <: Real}
    method::Symbol
    criterion::Symbol
    threshold::T1
    pcr_opt::PCROpt
end
function LoadingsOpt(; method::Symbol = :FReg, criterion::Symbol = :pval,
                     threshold::Real = 0.05, pcr_opt::PCROpt = PCROpt(;))
    @smart_assert(method ∈ FSMethods)
    @smart_assert(criterion ∈ RegCriteria)
    return LoadingsOpt{typeof(threshold)}(method, criterion, threshold, pcr_opt)
end
function Base.setproperty!(obj::LoadingsOpt, sym::Symbol, val)
    if sym == :method
        @smart_assert(val ∈ FSMethods)
    elseif sym == :criterion
        @smart_assert(val ∈ RegCriteria)
    end
    return setfield!(obj, sym, val)
end

mutable struct FactorOpt
    B::Union{DataFrame, Nothing}
    loadings_opt::LoadingsOpt
    error::Bool
    var_genfunc::GenericFunction
end
function FactorOpt(; B::Union{DataFrame, Nothing} = nothing,
                   loadings_opt::LoadingsOpt = LoadingsOpt(;), error::Bool = true,
                   var_genfunc::GenericFunction = GenericFunction(; func = StatsBase.var,
                                                                  kwargs = (; dims = 1)))
    return FactorOpt(B, loadings_opt, error, var_genfunc)
end

mutable struct BLOpt{T1 <: Real}
    method::Symbol
    constant::Bool
    diagonal::Bool
    eq::Bool
    delta::Union{Nothing, <:Real}
    rf::T1
    var_genfunc::GenericFunction
    denoise::DenoiseOpt
    posdef::PosdefFixOpt
    jlogo::Bool
end
function BLOpt(; method::Symbol = :B, constant::Bool = true, eq::Bool = true,
               diagonal::Bool = true, delta::Real = 1.0, rf::Real = 0.0,
               var_genfunc::GenericFunction = GenericFunction(; func = StatsBase.var,
                                                              kwargs = (; dims = 1)),
               denoise::DenoiseOpt = DenoiseOpt(;), posdef::PosdefFixOpt = PosdefFixOpt(;),
               jlogo::Bool = false)
    @smart_assert(method ∈ BLFMMethods)

    return BLOpt{typeof(rf)}(method, constant, eq, diagonal, delta, rf, var_genfunc,
                             denoise, posdef, jlogo)
end
function Base.setproperty!(obj::BLOpt, sym::Symbol, val)
    if sym == :method
        @smart_assert(val ∈ BLFMMethods)
    end
    return setfield!(obj, sym, val)
end

mutable struct ClusterOpt{T1 <: Integer, T2 <: Integer}
    linkage::Symbol
    branchorder::Symbol
    dbht_method::Symbol
    max_k::T1
    k::T2
end
function ClusterOpt(; linkage::Symbol = :single, branchorder::Symbol = :optimal,
                    dbht_method::Symbol = :Unique, max_k::Integer = 10, k::Integer = 0)
    @smart_assert(linkage ∈ LinkageTypes)
    @smart_assert(branchorder ∈ BranchOrderTypes)
    @smart_assert(dbht_method ∈ DBHTRootMethods)

    return ClusterOpt{typeof(max_k), typeof(k)}(linkage, branchorder, dbht_method, max_k, k)
end
function Base.setproperty!(obj::ClusterOpt, sym::Symbol, val)
    if sym == :linkage
        @smart_assert(val ∈ LinkageTypes)
    elseif sym == :branchorder
        @smart_assert(val ∈ BranchOrderTypes)
    elseif sym == :dbht_method
        @smart_assert(val ∈ DBHTRootMethods)
    end
    return setfield!(obj, sym, val)
end
mutable struct OptimiseOpt{T1 <: Integer, T2 <: Real, T3 <: Real, T4 <: Real, T5 <: Real}
    type::Symbol
    rm::Symbol
    obj::Symbol
    kelly::Symbol
    class::Symbol
    rrp_ver::Symbol
    u_cov::Symbol
    u_mu::Symbol
    sd_cone::Bool
    near_opt::Bool
    hist::T1
    rf::T2
    l::T3
    rrp_penalty::T4
    n::T5
    w_ini::AbstractVector
    w_min::AbstractVector
    w_max::AbstractVector
end
function OptimiseOpt(; type::Symbol = :Trad, rm::Symbol = :SD, obj::Symbol = :Sharpe,
                     kelly::Symbol = :None, class::Symbol = :Classic,
                     rrp_ver::Symbol = :None, u_cov::Symbol = :Box, u_mu::Symbol = :Box,
                     sd_cone::Bool = true, near_opt::Bool = false, hist::Integer = 1,
                     rf::Real = 0.0, l::Real = 2.0, rrp_penalty::Real = 1.0, n::Real = 20.0,
                     w_ini::AbstractVector = Vector{typeof(rf)}(undef, 0),
                     w_min::AbstractVector = Vector{typeof(rf)}(undef, 0),
                     w_max::AbstractVector = Vector{typeof(rf)}(undef, 0))
    @smart_assert(type ∈ PortTypes)
    @smart_assert(class ∈ PortClasses)
    @smart_assert(rm ∈ RiskMeasures)
    @smart_assert(obj ∈ HCObjFuncs)
    @smart_assert(kelly ∈ KellyRet)
    @smart_assert(rrp_ver ∈ RRPVersions)
    @smart_assert(u_mu ∈ UncertaintyTypes)
    @smart_assert(u_cov ∈ UncertaintyTypes)
    if near_opt
        @smart_assert(n > zero(n))
    end

    return OptimiseOpt{typeof(hist), typeof(rf), typeof(l), typeof(rrp_penalty), typeof(n)}(type,
                                                                                            rm,
                                                                                            obj,
                                                                                            kelly,
                                                                                            class,
                                                                                            rrp_ver,
                                                                                            u_cov,
                                                                                            u_mu,
                                                                                            sd_cone,
                                                                                            near_opt,
                                                                                            hist,
                                                                                            rf,
                                                                                            l,
                                                                                            rrp_penalty,
                                                                                            n,
                                                                                            w_ini,
                                                                                            w_min,
                                                                                            w_max)
end
function Base.setproperty!(obj::OptimiseOpt, sym::Symbol, val)
    if sym == :type
        @smart_assert(val ∈ PortTypes)
    elseif sym == :class
        @smart_assert(val ∈ PortClasses)
    elseif sym == :rm
        @smart_assert(val ∈ RiskMeasures)
    elseif sym == :obj
        @smart_assert(val ∈ HCObjFuncs)
    elseif sym == :kelly
        @smart_assert(val ∈ KellyRet)
    elseif sym == :rrp_ver
        @smart_assert(val ∈ RRPVersions)
    elseif sym == :u_mu
        @smart_assert(val ∈ UncertaintyTypes)
    elseif sym == :u_cov
        @smart_assert(val ∈ UncertaintyTypes)
    elseif sym == :n
        if obj.near_opt
            @smart_assert(val > zero(val))
        end
    else
        val = convert(typeof(getproperty(obj, sym)), val)
    end

    return setfield!(obj, sym, val)
end

export CovOpt, CovEstOpt, GerberOpt, DenoiseOpt, PosdefFixOpt, GenericFunction, MuOpt,
       CorOpt, CorEstOpt, WCOpt, KurtOpt, PCROpt, LoadingsOpt, FactorOpt, BLOpt, ClusterOpt,
       OptimiseOpt
