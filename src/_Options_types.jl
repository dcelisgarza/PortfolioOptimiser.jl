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

Structure and keyword constructor for estimating covariance matrices. This is part of [`CovOpt`](@ref), and as such some of these are only relevant when `method` has a specific value.

# Inputs

  - `estimator`:

      + `method ∈ (:Full, :Semi)`: abstract covariance estimator as defined by [`StatsBase`](https://juliastats.org/StatsBase.jl/stable/cov/#StatsBase.CovarianceEstimator). Enables users to use packages which subtype this interface such as [CovarianceEstimation.jl](https://github.com/mateuszbaran/CovarianceEstimation.jl).

  - `target_ret`:

      + `method == :Semi`: returns less than or equal to this value are considered downside returns.
  - `genfunc`:

      + `method ∈ (:Full, :Semi, :Custom_Func)`: generic function [`GenericFunction`](@ref) for computing the covariance matrix.
  - `custom`:

      + `method == :Custom_Val`: custom covariance matrix.
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
    solvers::Union{<:AbstractDict, NamedTuple} = Dict()
    genfunc::GenericFunction = GenericFunction(; func = x -> x)
end
```

Structure and keyword constructor for fixing non-positive definite matrices.

# Inputs

  - `method`: method for fixing non-positive definite matrices from [`PosdefFixMethods`](@ref).

  - `solvers`:

      + `method == :PSD`: provides the solvers and corresponding parameters for solving the `JuMP` model. There can be two `key => value` pairs.

          * `:solver => value`: `value` is a `JuMP` optimiser. The optimiser can be declared alongside its attributes by using `JuMP.solver_with_attributes`. Solver must support `MOI.PSDCone`, or `JuMP` must be able to transform it/them into a supported form.
          * `:params => value`: (optional) `value` must be a `Dict` or `NamedTuple` whose `key => value` pairs are the solver-specific settings.
  - `genfunc`:

      + `method == :Custom`: generic function [`GenericFunction`](@ref) for fixing non-positive definite matrices.
"""
mutable struct PosdefFixOpt
    method::Symbol
    solvers::Union{<:AbstractDict, NamedTuple}
    genfunc::GenericFunction
end
function PosdefFixOpt(; method::Symbol = :Nearest,
                      solvers::Union{<:AbstractDict, NamedTuple} = Dict(),
                      genfunc::GenericFunction = GenericFunction(; func = x -> x))
    @smart_assert(method ∈ PosdefFixMethods)

    return PosdefFixOpt(method, solvers, genfunc)
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
    normalise::Bool = false
    mean_func::GenericFunction = GenericFunction(; func = StatsBase.mean,
                                                 kwargs = (; dims = 1))
    std_func::GenericFunction = GenericFunction(; func = StatsBase.std,
                                                kwargs = (; dims = 1))
    posdef::PosdefFixOpt = PosdefFixOpt(;)
end
```

Structure and keyword constructor for Gerber and Gerber-derived covariance and correlation matrices. This is part of [`CovOpt`](@ref) and [`CorOpt`](@ref). It is only relevant when `method ∈ (:Gerber0, :Gerber1, :Gerber2, :SB0, :SB1, :Gerber_SB0, :Gerber_SB1)`.

# Inputs

  - `threshold`: significance threshold, `threshold ∈ (0, 1)`.
  - `normalise`: whether to normalise the returns to have a mean equal to `0` and a standard deviation equal to `1`.
  - `mean_func`: [`GenericFunction`](@ref) for computing the expected returns vector.
  - `std_func`: [`GenericFunction`](@ref) for computing the standard deviation of the returns.
  - `posdef`: [`PosdefFixOpt`](@ref) options for fixing non-positive definite matrices.
"""
mutable struct GerberOpt{T1 <: Real}
    threshold::T1
    normalise::Bool
    mean_func::GenericFunction
    std_func::GenericFunction
    posdef::PosdefFixOpt
end
function GerberOpt(; threshold::Real = 0.5, normalise = false,
                   mean_func::GenericFunction = GenericFunction(; func = StatsBase.mean,
                                                                kwargs = (; dims = 1)),
                   std_func::GenericFunction = GenericFunction(; func = StatsBase.std,
                                                               kwargs = (; dims = 1)),
                   posdef::PosdefFixOpt = PosdefFixOpt(;))
    @smart_assert(zero(threshold) < threshold < one(threshold))

    return GerberOpt{typeof(threshold)}(threshold, normalise, mean_func, std_func, posdef)
end
function Base.setproperty!(obj::GerberOpt, sym::Symbol, val)
    if sym == :threshold
        @smart_assert(zero(val) < val < one(val))
    end
    return setfield!(obj, sym, val)
end

"""
```julia
@kwdef mutable struct SBOpt{T1 <: Real, T2 <: Real, T3 <: Real, T4 <: Real}
    c1::T1 = 0.5
    c2::T2 = 0.5
    c3::T3 = 4
    n::T4 = 2
end
```

Structure for storing options for computing Smyth-Broby modifications of the Gerber statistic when `method ∈ (:SB0, :SB1, :Gerber_SB0, :Gerber_SB1)` from [`CovMethods`](@ref) or [`CorMethods`](@ref) [SB](@cite).

# Inputs

  - `c1`: confusion zone threshold (``c_1`` in the paper), `c1 ∈ (0, 1]`.
  - `c2`: indecision zone threshold (``c_2`` in the paper), `c2 ∈ (0, 1]`.
  - `c3`: large co-movement threshold (4 in the paper).
  - `n`: exponent of the regularisation term (``n = 2`` in the paper).
"""
mutable struct SBOpt{T1 <: Real, T2 <: Real, T3 <: Real, T4 <: Real}
    c1::T1
    c2::T2
    c3::T3
    n::T4
end
function SBOpt(; c1::Real = 0.5, c2::Real = 0.5, c3::Real = 4, n::Real = 2)
    @smart_assert(zero(c1) < c1 <= one(c1))

    return SBOpt{typeof(c1), typeof(c2), typeof(c3), typeof(n)}(c1, c2, c3, n)
end
function Base.setproperty!(obj::SBOpt, sym::Symbol, val)
    if sym ∈ (:c1, :c2)
        @smart_assert(zero(val) < val <= one(val))
    elseif sym == :c3
        @smart_assert(val >= obj.c2)
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

Structure and keyword constructor for denoising matrices.

# Inputs

  - `method`: method for denoising matrices from [`DenoiseMethods`](@ref).

  - `alpha`:

      + `method == :Shrink`: significance level, `alpha ∈ [0, 1]`.
  - `detone`:

      + `true`: take only the largest `mkt_comp` eigenvalues from the correlation matrix.
  - `mkt_comp`: the number of largest eigenvalues to keep from the correlation matrix.
  - `kernel`: kernel for fitting the average shifted histograms from [AverageShiftedHistograms.jl Kernel Functions](https://joshday.github.io/AverageShiftedHistograms.jl/latest/kernels/).
  - `m`: number of adjacent histograms to smooth over [AverageShiftedHistograms.jl Usage](https://joshday.github.io/AverageShiftedHistograms.jl/latest/#Usage).
  - `n`: number of points used when creating the range of values to which the average shifted histogram is to be fitted [AverageShiftedHistograms.jl Usage](https://joshday.github.io/AverageShiftedHistograms.jl/latest/#Usage).
  - `genfunc`: only `genfunc.args` and `genfunc.kwargs` are used. These are the `args` and `kwargs` passed to [`Optim.optimize`](https://julianlsolvers.github.io/Optim.jl/stable/#user/minimization/). This is used for finding the eigenvalue that minimises the residual error between the fitted average shifted histogram and a covariance matrix arising from normally distributed returns. Eigenvalues larger than this are considered significant [MLAM; Ch. 2](@cite).

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
@kwdef mutable struct CovOpt
    method::Symbol = :Full
    estimation::CovEstOpt = CovEstOpt(;)
    gerber::GerberOpt = GerberOpt(;)
    sb::SBOpt = SBOpt(;)
    denoise::DenoiseOpt = DenoiseOpt(;)
    posdef::PosdefFixOpt = PosdefFixOpt(;)
    jlogo::Bool = false
    uplo::Symbol = :U
end
```

Structure and keyword constructor for computing covariance matrices.

# Inputs

  - `method`: covariance estimation method from [`CovMethods`](@ref).

  - `estimation`: covariance estimation options [`CovEstOpt`](@ref).
  - `gerber`: Gerber covariance options [`GerberOpt`](@ref).
  - `sb`: options for Smyth-Broby modifications of the Gerber statistic [`SBOpt`](@ref).
  - `denoise`: denoising options [`DenoiseOpt`](@ref).
  - `posdef`: options for fixing non-positive definite matrices [`PosdefFixOpt`](@ref).
  - `jlogo`:

      + `true`: uses [`PMFG_T2s`](@ref) and [`J_LoGo`](@ref) to estimate the covariance from its relationship structure.
  - `uplo`: argument for [Symmetric](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#LinearAlgebra.Symmetric) to ensure the covariance matrix is symmetric (to combat floating point arithmetic problems).
"""
mutable struct CovOpt
    # Cov method
    method::Symbol
    # Estimation
    estimation::CovEstOpt
    # Gerber
    gerber::GerberOpt
    # SB
    sb::SBOpt
    # Denoise
    denoise::DenoiseOpt
    # Posdef fix
    posdef::PosdefFixOpt
    # J-LoGo
    jlogo::Bool
    # Symmetric
    uplo::Symbol
end
function CovOpt(; method::Symbol = :Full, estimation::CovEstOpt = CovEstOpt(;),
                gerber::GerberOpt = GerberOpt(;), sb::SBOpt = SBOpt(;),
                denoise::DenoiseOpt = DenoiseOpt(;), posdef::PosdefFixOpt = PosdefFixOpt(;),
                jlogo::Bool = false, uplo::Symbol = :U)
    @smart_assert(method ∈ CovMethods)

    return CovOpt(method, estimation, gerber, sb, denoise, posdef, jlogo, uplo)
end
function Base.setproperty!(obj::CovOpt, sym::Symbol, val)
    if sym == :method
        @smart_assert(val ∈ CovMethods)
    end
    return setfield!(obj, sym, val)
end

"""
```
@kwdef mutable struct MuOpt{T1 <: Real}
    method::Symbol = :Default
    target::Symbol = :GM
    rf::T1 = 0.0
    genfunc::GenericFunction = GenericFunction(; func = StatsBase.mean,
                                                kwargs = (; dims = 1))
    custom::Union{<:AbstractVector{<:Real}, Nothing}
    mkt_ret::Union{<:AbstractVector{<:Real}, Nothing}
    sigma::Union{<:AbstractMatrix{<:Real}, Nothing}
end
```

Structure and keyword constructor for computing expected returns vectors.

# Inputs

  - `method`: method for estimating the expected returns vector from [`MuMethods`](@ref).

  - `target`:

      + `method ∈ (:JS, :BS, :BOP)`: target from [`MuTargets`](@ref).
  - `rf`: risk-free rate.
  - `genfunc`: generic function [`GenericFunction`](@ref) for estimating the unadjusted expected returns vector.

      + `method ∈ (:Default, :Custom_Func)`: return this value.
      + `method ∈ (:JS, :BS, :BOP, :CAPM)`: the value is used as the starting point for computing the adjusted expected returns vector using the given `method` and---if applicable---`target`.
  - `custom`:

      + `method == :Custom_Val`: value of the mean returns vector.
  - `mkt_ret`:

      + `method == :CAPM`: market returns.
  - `sigma`:

      + `method ∈ (:JS, :BS, :BOP, :CAPM)`: value of the covariance matrix used when computing the adjusted expected returns vector. When computing from [`asset_statistics!`](@ref), this value is set automatically.
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
    target_ret::Union{<:AbstractVector{<:Real}, <:Real} = 0.0
    custom_kurt::Union{<:AbstractMatrix{<:Real}, Nothing} = nothing
    custom_skurt::Union{<:AbstractMatrix{<:Real}, Nothing} = nothing
end
```

Structure and keyword constructor for estimating cokurtosis matrices. This is part of [`KurtOpt`](@ref).

# Inputs

  - `target_ret`: returns less than or equal to this value are considered downside returns when computing the semi cokurtosis matrix.
  - `custom_kurt`: custom value for the cokurtosis matrix.
  - `custom_skurt`: custom value for the semi cokurtosis matrix.
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

"""
```julia
@kwdef mutable struct KurtOpt
    estimation::KurtEstOpt = KurtEstOpt(;)
    denoise::DenoiseOpt = DenoiseOpt(;)
    posdef::PosdefFixOpt = PosdefFixOpt(;)
    jlogo::Bool = false
end
```

Structure and keyword constructor for computing cokurtosis matrices.

# Inputs

  - `estimation`: cokurtosis estimation options [`KurtEstOpt`](@ref).

  - `denoise`: denoising options [`DenoiseOpt`](@ref).
  - `posdef`: options for fixing non-positive definite matrices [`PosdefFixOpt`](@ref).
  - `jlogo`:

      + `true`: uses [`PMFG_T2s`](@ref) and [`J_LoGo`](@ref) to estimate the cokurtosis from its relationship structure.
"""
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

"""
```julia
@kwdef mutable struct CorEstOpt{T1 <: Real, T2 <: AbstractMatrix{<:Real},
                                T3 <: AbstractMatrix{<:Real}, T4 <: AbstractMatrix{<:Real}}
    estimator::CovarianceEstimator = StatsBase.SimpleCovariance(; corrected = true)
    alpha::T1 = 0.05
    bins_info::Union{Symbol, <:Integer} = :KN
    cor_genfunc::GenericFunction = GenericFunction(; func = StatsBase.cor)
    dist_genfunc::GenericFunction = GenericFunction(;
                                                    func = x -> sqrt.(clamp!((1 .- x) / 2,
                                                                             0, 1)))
    target_ret::Union{<:AbstractVector{<:Real}, <:Real} = 0.0
    custom_cor::T2 = Matrix{Float64}(undef, 0, 0)
    custom_dist::T3 = Matrix{Float64}(undef, 0, 0)
    sigma::T4 = Matrix{Float64}(undef, 0, 0)
end
```

Structure and keyword constructor for estimating covariance matrices. This is part of [`CorOpt`](@ref), and as such some of these are only relevant when `method` has a specific value.

# Inputs

  - `estimator`:

      + `method ∈ (:Full, :Semi)`: abstract covariance estimator as defined by [`StatsBase`](https://juliastats.org/StatsBase.jl/stable/cov/#StatsBase.CovarianceEstimator). Enables users to use packages which subtype this interface such as [CovarianceEstimation.jl](https://github.com/mateuszbaran/CovarianceEstimation.jl).

  - `alpha`:

      + `method == :Tail`: significance parameter, `alpha ∈ (0, 1)`.
  - `bins_info`:

      + `method == :Mutual_Info`: number of bins. It can take on two types of values:

          * `isa(bins_info, Symbol)`: bin width choice method must be one of [`BinMethods`](@ref).
          * `isa(bins_info, Integer)`: the data is split into as many bins as specified.
  - `cor_genfunc`:

      + `method ∈ (:Full, :Semi, :Custom_Func)`: generic function [`GenericFunction`](@ref) for computing the correlation matrix.
  - `dist_genfunc`:

      + `method ∈ (:Full, :Semi, :Custom_Func)`: generic function [`GenericFunction`](@ref) for computing the distance matrix.
  - `custom_cor`:

      + `method == :Custom_Val`: custom correlation matrix.
  - `custom_dist`:

      + `method == :Custom_Val`: custom distance matrix.
  - `target_ret`:

      + `method == :Semi`: returns less than or equal to this value are considered downside returns.
  - `sigma`:

      + `method == :Cov_to_Cor`: value of covariance matrix from which the correlation is to be computed. When computing from [`asset_statistics!`](@ref), this value is set automatically.
"""
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
    @smart_assert(zero(alpha) < alpha < one(alpha))
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
        @smart_assert(zero(val) <= val <= one(val))
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

"""
```
@kwdef mutable struct CorOpt
    method::Symbol = :Pearson
    estimation::CorEstOpt = CovEstOpt(;)
    gerber::GerberOpt = GerberOpt(;)
    sb::SBOpt = SBOpt(;)
    denoise::DenoiseOpt = DenoiseOpt(;)
    posdef::PosdefFixOpt = PosdefFixOpt(;)
    jlogo::Bool = false
    uplo::Symbol = :U
end
```

Structure and keyword constructor for computing covariance matrices.

# Inputs

  - `method`: covariance estimation method from [`CorMethods`](@ref).

  - `estimation`: covariance estimation options [`CovEstOpt`](@ref).
  - `gerber`: Gerber covariance options [`GerberOpt`](@ref).
  - `sb`: options for Smyth-Broby modifications of the Gerber statistic [`SBOpt`](@ref).
  - `denoise`: denoising options [`DenoiseOpt`](@ref).
  - `posdef`: options for fixing non-positive definite matrices [`PosdefFixOpt`](@ref).
  - `jlogo`:

      + `true`: uses [`PMFG_T2s`](@ref) and [`J_LoGo`](@ref) to estimate the covariance from its relationship structure.
  - `uplo`: argument for [Symmetric](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#LinearAlgebra.Symmetric) to ensure the correlation and distance matrices are symmetric (to combat floating point arithmetic problems).
"""
mutable struct CorOpt
    # Cov method
    method::Symbol
    # Estimation
    estimation::CorEstOpt
    # Gerber
    gerber::GerberOpt
    # SB
    sb::SBOpt
    # Denoise
    denoise::DenoiseOpt
    # Posdef fix
    posdef::PosdefFixOpt
    # J-LoGo
    jlogo::Bool
    # Symmetric
    uplo::Symbol
end
function CorOpt(; method::Symbol = :Pearson, estimation::CorEstOpt = CorEstOpt(;),
                gerber::GerberOpt = GerberOpt(;), sb::SBOpt = SBOpt(;),
                denoise::DenoiseOpt = DenoiseOpt(;), posdef::PosdefFixOpt = PosdefFixOpt(;),
                jlogo::Bool = false, uplo::Symbol = :U)
    @smart_assert(method ∈ CorMethods)

    return CorOpt(method, estimation, gerber, sb, denoise, posdef, jlogo, uplo)
end
function Base.setproperty!(obj::CorOpt, sym::Symbol, val)
    if sym == :method
        @smart_assert(val ∈ CorMethods)
    end
    return setfield!(obj, sym, val)
end

"""
```julia
@kwdef mutable struct WCOpt{T1 <: Real, T2 <: Real, T3 <: Real, T4, T5 <: Integer,
                            T6 <: Integer}
    calc_box::Bool = true
    calc_ellipse::Bool = true
    diagonal::Bool = false
    box::Symbol = :Stationary
    ellipse::Symbol = :Stationary
    k_mu_method::Union{Symbol, <:Real} = :Normal
    k_sigma_method::Union{Symbol, <:Real} = :Normal
    dcov::T1 = 0.1
    dmu::T2 = 0.1
    q::T3 = 0.05
    rng::T4 = Random.default_rng()
    seed::Union{<:Integer, Nothing} = nothing
    n_sim::T5 = 3_000
    block_size::T6 = 3
    posdef::PosdefFixOpt = PosdefFixOpt(;)
end
```

Structure and keyword constructor for computing worst case statistics in [`wc_statistics!`](@ref).

# Inputs

  - `calc_box`: whether to compute box sets.

  - `calc_ellipse`: whether to compute elliptical sets.
  - `diagonal`: whether to consider only the diagonal of the covariance matrices of estimation errors.
  - `box`: method from [`BoxMethods`](@ref) for computing box sets.
  - `ellipse`: method from [`EllipseMethods`](@ref) for computing elliptical sets.
  - `k_mu_method`:

      + `isa(k_sigma_method, Symbol)`: method from [`kMethods`](@ref) for computing the distance parameter of the elliptical set for the asset expected returns vector.
      + `isa(k_sigma_method, Real)`: value of the distance parameter of the elliptical set for the asset expected returns vector.
  - `k_sigma_method`:

      + `isa(k_sigma_method, Symbol)`: method from [`kMethods`](@ref) for computing the distance parameter of the elliptical set for the asset covariance matrix.
      + `isa(k_sigma_method, Real)`: value of the distance parameter of the elliptical set for the asset covariance matrix.
  - `dcov`:

      + `box == :Delta`: the percentage of the covariance matrix that parametrises its box set.
  - `dmu`:

      + `box == :Delta`: the percentage of the expected returns vector that parametrises its box set.
  - `q`: significance level of the selected bootstrapping method.
  - `rng`: generator for the random numbers used in [`gen_bootstrap`](@ref) and [`wc_statistics!`](@ref).
  - `seed`: seed for the random number generator used in [`gen_bootstrap`](@ref) and [`wc_statistics!`](@ref). If `rng` does not support the seed, the function will error.
  - `n_sim`: number of simulations for the bootstrapping method.
  - `block_size`:

      + `box ∈ (:Stationary, :Circular, :Moving)`: average block size when bootstrapping methods.
  - `posdef`: options for fixing non-positive definite matrices [`PosdefFixOpt`](@ref).

      + `ellipse ∈ (:Stationary, :Circular, :Moving)`: average block size when bootstrapping methods.
  - `posdef`: options for fixing non-positive definite matrices [`PosdefFixOpt`](@ref).
"""
mutable struct WCOpt{T1 <: Real, T2 <: Real, T3 <: Real, T4, T5 <: Integer, T6 <: Integer}
    calc_box::Bool
    calc_ellipse::Bool
    diagonal::Bool
    box::Symbol
    ellipse::Symbol
    k_mu_method::Union{Symbol, <:Real}
    k_sigma_method::Union{Symbol, <:Real}
    dcov::T1
    dmu::T2
    q::T3
    rng::T4
    seed::Union{<:Integer, Nothing}
    n_sim::T5
    block_size::T6
    posdef::PosdefFixOpt
end
function WCOpt(; calc_box::Bool = true, calc_ellipse::Bool = true, diagonal::Bool = false,
               box::Symbol = :Stationary, ellipse::Symbol = :Stationary,
               k_mu_method::Union{Symbol, <:Real} = :Normal,
               k_sigma_method::Union{Symbol, <:Real} = :Normal, dcov::Real = 0.1,
               dmu::Real = 0.1, q::Real = 0.05, rng = Random.default_rng(),
               seed::Union{<:Integer, Nothing} = nothing, n_sim::Integer = 3_000,
               block_size::Integer = 3, posdef::PosdefFixOpt = PosdefFixOpt(;))
    @smart_assert(box ∈ BoxMethods)
    @smart_assert(ellipse ∈ EllipseMethods)
    if isa(k_mu_method, Symbol)
        @smart_assert(k_mu_method ∈ kMethods)
    end
    if isa(k_sigma_method, Symbol)
        @smart_assert(k_sigma_method ∈ kMethods)
    end
    @smart_assert(zero(q) < q < one(q))

    return WCOpt{typeof(dcov), typeof(dmu), typeof(q), typeof(rng), typeof(n_sim),
                 typeof(block_size)}(calc_box, calc_ellipse, diagonal, box, ellipse,
                                     k_mu_method, k_sigma_method, dcov, dmu, q, rng, seed,
                                     n_sim, block_size, posdef)
end
function Base.setproperty!(obj::WCOpt, sym::Symbol, val)
    if sym == :box
        @smart_assert(val ∈ BoxMethods)
    elseif sym == :ellipse
        @smart_assert(val ∈ EllipseMethods)
    elseif sym ∈ (:k_mu_method, :k_sigma_method)
        if isa(val, Symbol)
            @smart_assert(val ∈ kMethods)
        end
    elseif sym == :q
        @smart_assert(zero(val) < val < one(val))
    end
    return setfield!(obj, sym, val)
end

"""
```julia
@kwdef mutable struct PCROpt
    mean_genfunc::GenericFunction = GenericFunction(; func = StatsBase.mean,
                                                    kwargs = (; dims = 2))
    std_genfunc::GenericFunction = GenericFunction(; func = StatsBase.std,
                                                   kwargs = (; dims = 2))
    pca_s_genfunc::GenericFunction = GenericFunction(; func = StatsBase.standardize,
                                                     args = (StatsBase.ZScoreTransform,),
                                                     kwargs = (; dims = 2))
    pca_genfunc::GenericFunction = GenericFunction(; func = MultivariateStats.fit,
                                                   args = (MultivariateStats.PCA,))
end
```

Structure and keyword constructor for the `:PCR` method from [`FSMethods`](@ref) of [`loadings_matrix`](@ref).

# Inputs

  - `mean_genfunc`: generic function [`GenericFunction`](@ref) for computing the mean of the observations in the PCR function.
  - `std_genfunc`: generic function [`GenericFunction`](@ref) for computing the standard deviation of the observations in the PCR function.
  - `pca_s_genfunc`: generic function [`GenericFunction`](@ref) for standardising the data to prepare it for PCA.
  - `pca_genfunc`: generic function [`GenericFunction`](@ref) for standardising fitting the data to a PCA model.
"""
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

"""
```julia
@kwdef mutable struct LoadingsOpt{T1 <: Real}
    method::Symbol = :FReg
    criterion::Symbol = :pval
    threshold::T1 = 0.05
    pcr_opt::PCROpt = PCROpt(;)
end
```

Structure and keyword constructor for computing the loadings matrix in [`loadings_matrix`](@ref).

# Inputs

  - `method`: method for computing the loadings matrix from [`FSMethods`](@ref) .

  - `criterion`:

      + `method ∈ (:FReg, :BReg)`: regression criterion from [`RegCriteria`](@ref) for feature selection.
  - `threshold`:

      + `method ∈ (:FReg, :BReg) && criterion == :pval`: values greater than this are considered significant.
  - `pcr_opt`:

      + `method == :PCR`: options for the method [`PCROpt`](@ref).
"""
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

"""
```julia
@kwdef mutable struct FactorOpt
    B::Union{DataFrame, Nothing} = nothing
    loadings_opt::LoadingsOpt = LoadingsOpt(;)
    error::Bool = true
    var_genfunc::GenericFunction = GenericFunction(; func = StatsBase.var,
                                                   kwargs = (; dims = 1))
end
```

Structure and keyword constructor for computing factor statistics.

# Input

  - `B`: loadings matrix in dataframe form. The number of rows must be equal to the number of asset and factor returns observations, `T`. Must have a few different columns.

      + `tickers`: (optional) contains the list of tickers.
      + `const`: (optional) contains the regression constant.
      + The other columns must be the names of the factors.

  - `loadings_opt`: options for computing the loadings matrix [`LoadingsOpt`](@ref).
  - `error`:

      + `true`: account for the error between the asset returns and factor regression.
  - `var_genfunc`:

      + `error == true`: generic function [`GenericFunction`](@ref) for computing the variance of the error between the asset returns and factor regression.
"""
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

"""
```julia
@kwdef mutable struct BLOpt{T1 <: Real}
    method::Symbol = :B
    constant::Bool = true
    diagonal::Bool = true
    eq::Bool = true
    delta::Union{Nothing, <:Real} = 1.0
    rf::T1 = 0.0
    var_genfunc::GenericFunction = GenericFunction(; func = StatsBase.var,
                                                   kwargs = (; dims = 1))
    denoise::DenoiseOpt = DenoiseOpt(;)
    posdef::PosdefFixOpt = PosdefFixOpt(;)
    jlogo::Bool = false
end
```

Structure and keyword constructor for computing Black-Litterman statistics in [`black_litterman_statistics!`](@ref) and [`black_litterman_factor_satistics!`](@ref).

  - [`black_litterman_statistics!`](@ref): calls [`black_litterman`](@ref).

  - [`black_litterman_factor_satistics!`](@ref):

      + `method == :A`: calls [`augmented_black_litterman`](@ref).
      + `method == :B`: calls [`bayesian_black_litterman`](@ref).

# Inputs

  - `method`: one of [`BLFMMethods`](@ref).

      + [`black_litterman_factor_satistics!`](@ref): `method` for choosing what Black-Litterman Factor model to use.

  - `constant`:

      + [`black_litterman_factor_satistics!`](@ref): indicates whether the loadings matrix contains a `"const"` column, automatically set inside the function.
  - `diagonal`:

      + [`black_litterman_factor_satistics!`](@ref):

          * `method == :B`: flag used in the definition of ``\\mathbf{D}``.
  - `eq`:

      + [`black_litterman_statistics!`](@ref): flag used in the definition of ``\\bm{\\Pi}``.

      + [`black_litterman_factor_satistics!`](@ref):

          * `method == :A`: flag used in the definition of ``\\bm{\\Pi}_{a}``.
  - `delta`:

      + [`black_litterman_statistics!`](@ref): value of ``\\delta`` in the definition of ``\\bm{\\Pi}`` when `eq == true`.

      + [`black_litterman_factor_satistics!`](@ref):

          * `method == :A`: value of ``\\delta`` in the definition of ``\\bm{\\Pi}_{a}`` when `eq == true`.
  - `rf`: risk-free rate.
  - `var_genfunc`:

      + [`black_litterman_factor_satistics!`](@ref):

          * `method == :B && diagonal == true`: generic function [`GenericFunction`](@ref) for the variance in the definition of ``\\mathbf{D}`` .
  - `denoise`: denoising options [`DenoiseOpt`](@ref).
  - `posdef`: options for fixing non-positive definite matrices [`PosdefFixOpt`](@ref).
  - `jlogo`:

      + `true`: uses [`PMFG_T2s`](@ref) and [`J_LoGo`](@ref) to estimate the covariance from its relationship structure.
"""
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
function BLOpt(; method::Symbol = :B, constant::Bool = true, diagonal::Bool = true,
               eq::Bool = true, delta::Real = 1.0, rf::Real = 0.0,
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

"""
```julia
@kwdef mutable struct ClusterOpt{T1 <: Integer, T2 <: Integer}
    linkage::Symbol = :single
    branchorder::Symbol = :optimal
    dbht_method::Symbol = :Unique
    max_k::T1 = 0
    k::T2 = 0
    genfunc::GenericFunction = GenericFunction(; func = x -> 2 .- (x .^ 2) / 2)
end
```

Structure and keyword constructor for clustering options in [`optimise!`](@ref), [`cluster_matrix`](@ref).

  - [`optimise!`](@ref): calls [`_hierarchical_clustering`](@ref).
  - [`cluster_matrix`](@ref): calls
  - [`_hierarchical_clustering`](@ref): calls [`_hcluster_choice`](@ref).

# Inputs

  - `linkage`: clustering linkage method from [`LinkageMethods`](@ref).

  - `branchorder`: branch order for ordering leaves and branches from [hclust](https://juliastats.org/Clustering.jl/stable/hclust.html#Clustering.hclust).
  - `dbht_method`: root finding method from [`DBHTRootMethods`](@ref).
  - `max_k`: maximum number of clusters to cut the sample into.

      + `iszero(max_k)`: computes the value in [`_two_diff_gap_stat`](@ref) as ``k_{\\mathrm{max}} = \\left\\lceil \\sqrt{N} \\right\\rceil`` where ``N`` is the number of assets.
  - `k`: number of clusters to cut the sample into.

      + `iszero(k)`: determined by [`_two_diff_gap_stat`](@ref).
  - `genfunc`: function for computing a non-negative distance matrix from the correlation matrix when `method == :DBHT` as per [`DBHTs`](@ref).
"""
mutable struct ClusterOpt{T1 <: Integer, T2 <: Integer}
    linkage::Symbol
    branchorder::Symbol
    dbht_method::Symbol
    max_k::T1
    k::T2
    genfunc::GenericFunction
end
function ClusterOpt(; linkage::Symbol = :single, branchorder::Symbol = :optimal,
                    dbht_method::Symbol = :Unique, max_k::Integer = 0, k::Integer = 0,
                    genfunc::GenericFunction = GenericFunction(;
                                                               func = x -> 2 .- (x .^ 2) / 2))
    @smart_assert(linkage ∈ LinkageMethods)
    @smart_assert(branchorder ∈ BranchOrderTypes)
    @smart_assert(dbht_method ∈ DBHTRootMethods)

    return ClusterOpt{typeof(max_k), typeof(k)}(linkage, branchorder, dbht_method, max_k, k,
                                                genfunc)
end
function Base.setproperty!(obj::ClusterOpt, sym::Symbol, val)
    if sym == :linkage
        @smart_assert(val ∈ LinkageMethods)
    elseif sym == :branchorder
        @smart_assert(val ∈ BranchOrderTypes)
    elseif sym == :dbht_method
        @smart_assert(val ∈ DBHTRootMethods)
    end
    return setfield!(obj, sym, val)
end

"""
```julia
@kwdef mutable struct OptimiseOpt{T1 <: Integer, T2 <: Real, T3 <: Real, T4 <: Real,
                                  T5 <: Real}
    type::Symbol = :Trad
    rm::Symbol = :SD
    obj::Symbol = :Sharpe
    kelly::Symbol = :None
    class::Symbol = :Classic
    rrp_ver::Symbol = :None
    u_cov::Symbol = :Box
    u_mu::Symbol = :Box
    sd_cone::Bool = true
    owa_approx::Bool = true
    near_opt::Bool = false
    hist::T1 = 1
    rf::T2 = 0.0
    l::T3 = 2.0
    rrp_penalty::T4 = 1.0
    n::T5 = 20.0
    w_ini::AbstractVector = Vector{typeof(rf)}(undef, 0)
    w_min::AbstractVector = Vector{typeof(rf)}(undef, 0)
    w_max::AbstractVector = Vector{typeof(rf)}(undef, 0)
end
```

Structure and keyword constructor for storing the options to optimising portfolios in [`optimise!`](@ref) for all [`PortTypes`](@ref) and `:NCO` [`HCPortTypes`](@ref).

# Inputs

  - `type`: portfolio type from [`PortTypes`](@ref).

  - `rm`: risk measure from [`RiskMeasures`](@ref).
  - `obj`: objective function from [`ObjFuncs`](@ref).
  - `kelly`: kelly return from [`KellyRet`](@ref).
  - `class`: portfolio class from [`PortClasses`](@ref).
  - `rrp_ver`: version of relaxed risk parity from [`RRPVersions`](@ref) when `type == :RRP`
  - `u_cov`: type of uncertainty set from [`UncertaintyTypes`](@ref) for covariance matrix.
  - `u_mu`: type of uncertainty set from [`UncertaintyTypes`](@ref) for expected returns vector.
  - `sd_cone`:

      + `true`: use `MOI.SecondOrderCone` to model the standard deviation when `rm == :SD`.
      + `false`: use the quadratic expression for the standard deviation when `rm == :SD`.
  - `owa_approx`:

      + `true`: use the power cone expansion approximation when `rm ∈ (:GMD, :TG, :RTG, :OWA)`.
      + `false`: use the full Ordered Weight formulation when `rm ∈ (:GMD, :TG, :RTG, :OWA)`.
  - `near_opt`:
    + `true`: use the near optimal centering formulation. May not work with all risk measures depending on the solver. $(_solver_reqs("`MOI.ExponentialCone`"))
    + `false`: normal optimisation.
  - `hist`: choice of expected returns vector and covariance matrix from [`ClassHist`](@ref).
  - `rf`: risk-free rate.
  - `l`: risk aversion parameter when `obj == :Utility`.
  - `rrp_penalty`: 
    + `type == :RRP && rrp_ver == :Reg_Pen`: value of the relaxed risk penalty.
  - `n`: 
    + `near_opt == true`: number of sections to split the range between the minimum risk and maximum return portfolios.
  - `w_ini`:
    + `!isempty(w_ini)`: initial guess for the weights of the optimised portfolio. Some solvers do not support initial guesses and will see no benefit.
    + `isempty(w_ini)`: no initial for the weights of the optimised portfolio.
  - `w_min`:
    + `near_opt == true`:
      * `!isempty(w_min)`: assumed to be the value of the weights of the minimum risk portfolio.
      * `isempty(w_min)`: weights of the minimum risk portfolio are computed by [`optimise!`](@ref).
  - `w_max`:
    + `near_opt == true`:
      * `!isempty(w_min)`: assumed to be the value of the weights of the maximum return portfolio.
      * `isempty(w_min)`: weights of the maximum return portfolio are computed by [`optimise!`](@ref).
"""
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
    owa_approx::Bool
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
                     sd_cone::Bool = true, owa_approx::Bool = true, near_opt::Bool = false,
                     hist::Integer = 1, rf::Real = 0.0, l::Real = 2.0,
                     rrp_penalty::Real = 1.0, n::Real = 20.0,
                     w_ini::AbstractVector = Vector{typeof(rf)}(undef, 0),
                     w_min::AbstractVector = Vector{typeof(rf)}(undef, 0),
                     w_max::AbstractVector = Vector{typeof(rf)}(undef, 0))
    @smart_assert(type ∈ PortTypes)
    @smart_assert(rm ∈ RiskMeasures)
    @smart_assert(obj ∈ HCObjFuncs)
    @smart_assert(kelly ∈ KellyRet)
    @smart_assert(class ∈ PortClasses)
    @smart_assert(rrp_ver ∈ RRPVersions)
    @smart_assert(u_cov ∈ UncertaintyTypes)
    @smart_assert(u_mu ∈ UncertaintyTypes)
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
                                                                                            owa_approx,
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

"""
```julia
@kwdef mutable struct AllocOpt
    port_type::Symbol = :Trad
    method::Symbol = :LP
    latest_prices::AbstractVector = Float64[]
    investment::Real = 1e6
    rounding::Integer = 1
    reinvest::Bool = false
end
```

Structure and keyword constructor for storing the options to optimising the discrete allocation of portfolios in [`allocate!`](@ref).

# Inputs

  - `port_type`: key for the `optimal` fieldname of [`Portfolio`](@ref) and [`HCPortfolio`](@ref) which contains the asset weights of the optimised portfolio.

  - `method`: method for discretely allocating the portfolio from [`AllocMethods`](@ref).
  - `latest_prices`: vector of latest stock prices.
  - `investment`: amount of money available for buying assets.
  - `rounding`:

      + `method == :Greedy`: the number of decimal places used for rounding stocks.
  - `reinvest`: only has an effect if the portfolio being allocated shorts stocks.

      + `true`: reinvest the money acquired from shorting.
      + `false`: do not reinvest the money acquired from shorting.

!!! warning

    `latest_prices` and `investment` must be in the same currency.
"""
mutable struct AllocOpt
    port_type::Symbol
    method::Symbol
    latest_prices::AbstractVector
    investment::Real
    rounding::Integer
    reinvest::Bool
end
function AllocOpt(; port_type::Symbol = :Trad, method::Symbol = :LP,
                  latest_prices::AbstractVector = Float64[], investment::Real = 1e6,
                  rounding::Integer = 1, reinvest::Bool = false)
    @smart_assert(method ∈ AllocMethods)
    return AllocOpt(port_type, method, latest_prices, investment, rounding, reinvest)
end
function Base.setproperty!(obj::AllocOpt, sym::Symbol, val)
    if sym == :method
        @smart_assert(val ∈ AllocMethods)
    end
    return setfield!(obj, sym, val)
end

export CovOpt, CovEstOpt, GerberOpt, DenoiseOpt, PosdefFixOpt, GenericFunction, MuOpt,
       CorOpt, CorEstOpt, WCOpt, KurtOpt, KurtEstOpt, PCROpt, LoadingsOpt, FactorOpt, BLOpt,
       ClusterOpt, OptimiseOpt, SBOpt, AllocOpt
