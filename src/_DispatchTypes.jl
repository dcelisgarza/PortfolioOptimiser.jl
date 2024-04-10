import StatsBase: cov, cor, mean, std

abstract type PO_CovarianceEstimator <: StatsBase.CovarianceEstimator end
abstract type PO_StdDevEstimator <: StatsBase.CovarianceEstimator end
abstract type PO_MeanEstimator end
abstract type PO_PosdefFix <: NCM.NCMAlgorithm end
abstract type PO_DistanceEstimator end
abstract type PO_MatrixDenoiser end

struct SimpleMean <: PO_MeanEstimator end
@kwdef mutable struct PO_SimpleStdDevEstimator <: PO_StdDevEstimator
    corrected::Bool = true
end
@kwdef mutable struct PosdefPSD <: PO_PosdefFix
    solvers::Union{<:AbstractDict, NamedTuple} = Dict()
end
function _confirm_posdef!(X, _X, msg)
    if !isposdef(_X)
        @warn(msg *
              "matrix could not be made positive definite, please try a different method or a tighter tolerance.")
    else
        X .= _X
    end
end
function posdef_fix!(pf::NCM.NCMAlgorithm, X::AbstractMatrix; cov_flag::Bool = true,
                     msg::String = "")
    if isposdef(X)
        return nothing
    end
    _X = posdef_nearest(X, pf; cov_flag = cov_flag)
    return _confirm_posdef!(X, _X, msg)
end
function posdef_fix!(pf::PosdefPSD, X::AbstractMatrix; cov_flag::Bool = true,
                     msg::String = "")
    if isposdef(X)
        return nothing
    end
    _X = posdef_psd(X, pf.solvers; cov_flag = cov_flag)
    return _confirm_posdef!(X, _X, msg)
end

@kwdef mutable struct FullCov <: PO_CovarianceEstimator
    ce::CovarianceEstimator = StatsBase.SimpleCovariance(; corrected = true)
end
@kwdef mutable struct SemiCov <: PO_CovarianceEstimator
    ce::CovarianceEstimator = StatsBase.SimpleCovariance(; corrected = true)
    target::Union{<:AbstractVector{<:Real}, <:Real} = 0.0
end
@kwdef mutable struct Kendall <: PO_CovarianceEstimator
    se::PO_StdDevEstimator = PO_SimpleStdDevEstimator(; corrected = true)
end
@kwdef struct Spearman <: PO_CovarianceEstimator
    se::PO_StdDevEstimator = PO_SimpleStdDevEstimator(; corrected = true)
end

mutable struct Gerber0 <: PO_CovarianceEstimator
    se::PO_StdDevEstimator
    me::PO_MeanEstimator
    posdef::NCM.NCMAlgorithm
    normalise::Bool
    threshold::Real
end
function Gerber0(; se::PO_StdDevEstimator = PO_SimpleStdDevEstimator(; corrected = true),
                 me::PO_MeanEstimator = SimpleMean(),
                 posdef::NCM.NCMAlgorithm = NCM.Newton(; tau = 1e-12),
                 normalise::Bool = false, threshold::Real = 0.5)
    @smart_assert(zero(threshold) < threshold < one(threshold))

    return Gerber0(se, me, posdef, normalise, threshold)
end
mutable struct Gerber1 <: PO_CovarianceEstimator
    se::PO_StdDevEstimator
    me::PO_MeanEstimator
    posdef::NCM.NCMAlgorithm
    normalise::Bool
    threshold::Real
end
function Gerber1(; se::PO_StdDevEstimator = PO_SimpleStdDevEstimator(; corrected = true),
                 me::PO_MeanEstimator = SimpleMean(),
                 posdef::NCM.NCMAlgorithm = NCM.Newton(; tau = 1e-12),
                 normalise::Bool = false, threshold::Real = 0.5)
    @smart_assert(zero(threshold) < threshold < one(threshold))

    return Gerber1(se, me, posdef, normalise, threshold)
end
mutable struct Gerber2 <: PO_CovarianceEstimator
    se::PO_StdDevEstimator
    me::PO_MeanEstimator
    posdef::NCM.NCMAlgorithm
    normalise::Bool
    threshold::Real
end
function Gerber2(; se::PO_StdDevEstimator = PO_SimpleStdDevEstimator(; corrected = true),
                 me::PO_MeanEstimator = SimpleMean(),
                 posdef::NCM.NCMAlgorithm = NCM.Newton(; tau = 1e-12),
                 normalise::Bool = false, threshold::Real = 0.5)
    @smart_assert(zero(threshold) < threshold < one(threshold))

    return Gerber2(se, me, posdef, normalise, threshold)
end

function Base.setproperty!(obj::Union{Gerber0, Gerber1, Gerber2}, sym::Symbol, val)
    if sym == :threshold
        @smart_assert(zero(val) < val < one(val))
    end
    return setfield!(obj, sym, val)
end

mutable struct SB0 <: PO_CovarianceEstimator
    se::PO_StdDevEstimator
    me::PO_MeanEstimator
    posdef::NCM.NCMAlgorithm
    normalise::Bool
    threshold::Real
    c1::Real
    c2::Real
    c3::Real
    n::Real
end
function SB0(; se::PO_StdDevEstimator = PO_SimpleStdDevEstimator(; corrected = true),
             me::PO_MeanEstimator = SimpleMean(),
             posdef::NCM.NCMAlgorithm = NCM.Newton(; tau = 1e-12), normalise::Bool = false,
             threshold::Real = 0.5, c1 = 0.5, c2 = 0.5, c3 = 4, n = 2)
    @smart_assert(zero(threshold) < threshold < one(threshold))
    @smart_assert(zero(c1) < c1 <= one(c1))
    @smart_assert(zero(c2) < c2 <= one(c2))
    @smart_assert(c3 > c2)

    return SB0(se, me, posdef, normalise, threshold, c1, c2, c3, n)
end

mutable struct SB1 <: PO_CovarianceEstimator
    se::PO_StdDevEstimator
    me::PO_MeanEstimator
    posdef::NCM.NCMAlgorithm
    normalise::Bool
    threshold::Real
    c1::Real
    c2::Real
    c3::Real
    n::Real
end
function SB1(; se::PO_StdDevEstimator = PO_SimpleStdDevEstimator(; corrected = true),
             me::PO_MeanEstimator = SimpleMean(),
             posdef::NCM.NCMAlgorithm = NCM.Newton(; tau = 1e-12), normalise::Bool = false,
             threshold::Real = 0.5, c1 = 0.5, c2 = 0.5, c3 = 4, n = 2)
    @smart_assert(zero(threshold) < threshold < one(threshold))
    @smart_assert(zero(c1) < c1 <= one(c1))
    @smart_assert(zero(c2) < c2 <= one(c2))
    @smart_assert(c3 > c2)

    return SB1(se, me, posdef, normalise, threshold, c1, c2, c3, n)
end

mutable struct Gerber_SB0 <: PO_CovarianceEstimator
    se::PO_StdDevEstimator
    me::PO_MeanEstimator
    posdef::NCM.NCMAlgorithm
    normalise::Bool
    threshold::Real
    c1::Real
    c2::Real
    c3::Real
    n::Real
end
function Gerber_SB0(; se::PO_StdDevEstimator = PO_SimpleStdDevEstimator(; corrected = true),
                    me::PO_MeanEstimator = SimpleMean(),
                    posdef::NCM.NCMAlgorithm = NCM.Newton(; tau = 1e-12),
                    normalise::Bool = false, threshold::Real = 0.5, c1 = 0.5, c2 = 0.5,
                    c3 = 4, n = 2)
    @smart_assert(zero(threshold) < threshold < one(threshold))
    @smart_assert(zero(c1) < c1 <= one(c1))
    @smart_assert(zero(c2) < c2 <= one(c2))
    @smart_assert(c3 > c2)

    return Gerber_SB0(se, me, posdef, normalise, threshold, c1, c2, c3, n)
end

mutable struct Gerber_SB1 <: PO_CovarianceEstimator
    se::PO_StdDevEstimator
    me::PO_MeanEstimator
    posdef::NCM.NCMAlgorithm
    normalise::Bool
    threshold::Real
    c1::Real
    c2::Real
    c3::Real
    n::Real
end
function Gerber_SB1(; se::PO_StdDevEstimator = PO_SimpleStdDevEstimator(; corrected = true),
                    me::PO_MeanEstimator = SimpleMean(),
                    posdef::NCM.NCMAlgorithm = NCM.Newton(; tau = 1e-12),
                    normalise::Bool = false, threshold::Real = 0.5, c1 = 0.5, c2 = 0.5,
                    c3 = 4, n = 2)
    @smart_assert(zero(threshold) < threshold < one(threshold))
    @smart_assert(zero(c1) < c1 <= one(c1))
    @smart_assert(zero(c2) < c2 <= one(c2))
    @smart_assert(c3 > c2)

    return Gerber_SB1(se, me, posdef, normalise, threshold, c1, c2, c3, n)
end

function Base.setproperty!(obj::Union{SB0, SB1, Gerber_SB0, Gerber_SB1}, sym::Symbol, val)
    if sym == :threshold
        @smart_assert(zero(val) < val < one(val))
    elseif sym ∈ (:c1, :c2)
        @smart_assert(zero(val) < val <= one(val))
    elseif sym == :c3
        @smart_assert(val > obj.c2)
    end
    return setfield!(obj, sym, val)
end

mutable struct TailCov <: PO_CovarianceEstimator
    se::PO_StdDevEstimator
    alpha::Real
end
function TailCov(; se::PO_StdDevEstimator = PO_SimpleStdDevEstimator(; corrected = true),
                 alpha::Real = 0.05)
    @smart_assert(zero(alpha) < alpha < one(alpha))
    return TailCov(se, alpha)
end
function Base.setproperty!(obj::TailCov, sym::Symbol, val)
    if sym == :alpha
        @smart_assert(zero(val) < val < one(val))
    end
    return setfield!(obj, sym, val)
end

@kwdef mutable struct DistanceCov <: PO_CovarianceEstimator
    se::PO_StdDevEstimator = PO_SimpleStdDevEstimator(; corrected = true)
    metric::Distances.UnionMetric = Distances.Euclidean()
    args::Tuple = ()
    kwargs::NamedTuple = (;)
end

mutable struct MutualInfoCov <: PO_CovarianceEstimator
    se::PO_StdDevEstimator
    normalise::Bool
    bins::Union{Symbol, <:Integer}
end
function MutualInfoCov(;
                       se::PO_StdDevEstimator = PO_SimpleStdDevEstimator(;
                                                                         corrected = true),
                       normalise::Bool = true, bins::Union{Symbol, <:Integer} = :KN)
    @smart_assert(bins ∈ BinMethods || isa(bins, Int) && bins > zero(bins))

    return MutualInfoCov(se, normalise, bins)
end
function Base.setproperty!(obj::MutualInfoCov, sym::Symbol, val)
    if sym == :bins_info
        @smart_assert(val ∈ BinMethods || isa(val, Int) && val > zero(val))
    end
    return setfield!(obj, sym, val)
end

mutable struct VariationInfo <: Distances.UnionMetric
    normalise::Bool
    bins::Union{Symbol, <:Integer}
end
function VariationInfo(; normalise::Bool = true, bins::Union{Symbol, <:Integer} = :KN)
    @smart_assert(bins ∈ BinMethods || isa(bins, Int) && bins > zero(bins))

    return VariationInfo(normalise, bins)
end
function Base.setproperty!(obj::VariationInfo, sym::Symbol, val)
    if sym == :bins_info
        @smart_assert(val ∈ BinMethods || isa(val, Int) && val > zero(val))
    end
    return setfield!(obj, sym, val)
end
function Distances.pairwise(metric::VariationInfo, X::AbstractMatrix, args...; kwargs...)
    return variation_info_mtx(X, metric.bins, metric.normalise)
end

struct Cov2Cor <: PO_CovarianceEstimator end

function StatsBase.mean(::SimpleMean, X::AbstractArray,
                        w::Union{AbstractWeights, Nothing} = nothing; dims::Int = 1)
    return if isnothing(w)
        StatsBase.mean(X; dims = dims)
    else
        StatsBase.mean(X, w; dims = dims)
    end
end

function StatsBase.std(ce::PO_SimpleStdDevEstimator, X::AbstractArray,
                       w::Union{AbstractWeights, Nothing} = nothing; mean = nothing,
                       dims::Int = 1)
    return if isnothing(w)
        StatsBase.std(X; corrected = ce.corrected, mean = mean, dims = dims)
    else
        StatsBase.std(X .* w; corrected = ce.corrected, mean = mean, dims = dims)
    end
end

function StatsBase.cov(ce::FullCov, X::AbstractMatrix,
                       w::Union{AbstractWeights, Nothing} = nothing; dims::Int = 1,
                       kwargs...)
    return if isnothing(w)
        StatsBase.cov(ce.ce, X; dims = dims, kwargs...)
    else
        StatsBase.cov(ce.ce, X, w; dims = dims, kwargs...)
    end
end

function StatsBase.cov(ce::SemiCov, X::AbstractMatrix,
                       w::Union{AbstractWeights, Nothing} = nothing; dims::Int = 1,
                       kwargs...)
    X = if isa(ce.target, Real)
        min.(X .- ce.target, zero(eltype(X)))
    else
        min.(X .- transpose(ce.target), zero(eltype(X)))
    end

    mean = zero(eltype(X))

    return if isnothing(w)
        StatsBase.cov(ce.ce, X; mean = mean, dims = dims, kwargs...)
    else
        StatsBase.cov(ce.ce, X, w; mean = mean, dims = dims, kwargs...)
    end
end

function StatsBase.cor(ce::Union{FullCov, SemiCov}, X::AbstractMatrix,
                       w::Union{AbstractWeights, Nothing} = nothing; dims::Int = 1,
                       kwargs...)
    sigma = cov(ce, X, w; dims = dims, kwargs...)
    return Symmetric(cov2cor(isa(sigma, Matrix) ? sigma : Matrix(sigma)))
end

function StatsBase.cov(ce::Gerber0, X::AbstractMatrix,
                       w::Union{AbstractWeights, Nothing} = nothing; dims::Int = 1,
                       kwargs...)
    normalise = ce.normalise
    threshold = ce.threshold

    std_vec = vec(StatsBase.std(ce.se, X, w; dims = dims))

    sigma = if normalise
        mean_vec = vec(StatsBase.mean(ce.me, X, w; dims = dims))
        _gerber0_norm(X, mean_vec, std_vec, threshold)
    else
        _gerber0(X, std_vec, threshold)
    end

    posdef_fix!(ce.posdef, sigma; cov_flag = false, msg = "Gerber0 Covariance ")

    return Symmetric(sigma .* (std_vec * transpose(std_vec)))
end

function StatsBase.cor(ce::Gerber0, X::AbstractMatrix,
                       w::Union{AbstractWeights, Nothing} = nothing; dims::Int = 1,
                       kwargs...)
    normalise = ce.normalise
    threshold = ce.threshold

    std_vec = vec(StatsBase.std(ce.se, X, w; dims = dims))

    sigma = if normalise
        mean_vec = vec(StatsBase.mean(ce.me, X, w; dims = dims))
        _gerber0_norm(X, mean_vec, std_vec, threshold)
    else
        _gerber0(X, std_vec, threshold)
    end

    posdef_fix!(ce.posdef, sigma; cov_flag = false, msg = "Gerber0 Correlation ")

    return Symmetric(sigma)
end

function StatsBase.cov(ce::Gerber1, X::AbstractMatrix,
                       w::Union{AbstractWeights, Nothing} = nothing; dims::Int = 1,
                       kwargs...)
    normalise = ce.normalise
    threshold = ce.threshold

    std_vec = vec(StatsBase.std(ce.se, X, w; dims = dims))

    sigma = if normalise
        mean_vec = vec(StatsBase.mean(ce.me, X, w; dims = dims))
        _gerber1_norm(X, mean_vec, std_vec, threshold)
    else
        _gerber1(X, std_vec, threshold)
    end

    posdef_fix!(ce.posdef, sigma; cov_flag = false, msg = "Gerber1 Correlation ")

    return Symmetric(sigma .* (std_vec * transpose(std_vec)))
end

function StatsBase.cor(ce::Gerber1, X::AbstractMatrix,
                       w::Union{AbstractWeights, Nothing} = nothing; dims::Int = 1,
                       kwargs...)
    normalise = ce.normalise
    threshold = ce.threshold

    std_vec = vec(StatsBase.std(ce.se, X, w; dims = dims))

    sigma = if normalise
        mean_vec = vec(StatsBase.mean(ce.me, X, w; dims = dims))
        _gerber1_norm(X, mean_vec, std_vec, threshold)
    else
        _gerber1(X, std_vec, threshold)
    end

    posdef_fix!(ce.posdef, sigma; cov_flag = false, msg = "Gerber1 Correlation ")

    return Symmetric(sigma)
end

function StatsBase.cov(ce::Gerber2, X::AbstractMatrix,
                       w::Union{AbstractWeights, Nothing} = nothing; dims::Int = 1,
                       kwargs...)
    normalise = ce.normalise
    threshold = ce.threshold

    std_vec = vec(StatsBase.std(ce.se, X, w; dims = dims, kwargs...))

    sigma = if normalise
        mean_vec = vec(StatsBase.mean(ce.me, X, w; dims = dims))
        _gerber2_norm(X, mean_vec, std_vec, threshold)
    else
        _gerber2(X, std_vec, threshold)
    end

    posdef_fix!(ce.posdef, sigma; cov_flag = false, msg = "Gerber2 Correlation ")

    return Symmetric(sigma .* (std_vec * transpose(std_vec)))
end

function StatsBase.cor(ce::Gerber2, X::AbstractMatrix,
                       w::Union{AbstractWeights, Nothing} = nothing; dims::Int = 1,
                       kwargs...)
    normalise = ce.normalise
    threshold = ce.threshold

    std_vec = vec(StatsBase.std(ce.se, X, w; dims = dims, kwargs...))

    sigma = if normalise
        mean_vec = vec(StatsBase.mean(ce.me, X, w; dims = dims))
        _gerber2_norm(X, mean_vec, std_vec, threshold)
    else
        _gerber2(X, std_vec, threshold)
    end

    posdef_fix!(ce.posdef, sigma; cov_flag = false, msg = "Gerber2 Correlation ")

    return Symmetric(sigma)
end

function StatsBase.cov(ce::SB0, X::AbstractMatrix,
                       w::Union{AbstractWeights, Nothing} = nothing; dims::Int = 1,
                       kwargs...)
    normalise = ce.normalise
    threshold = ce.threshold
    c1 = ce.c1
    c2 = ce.c2
    c3 = ce.c3
    n = ce.n

    std_vec = vec(StatsBase.std(ce.se, X, w; dims = dims, kwargs...))
    mean_vec = vec(StatsBase.mean(ce.me, X, w; dims = dims))

    sigma = if normalise
        _sb0_norm(X, mean_vec, std_vec, threshold, c1, c2, c3, n)
    else
        _sb0(X, mean_vec, std_vec, threshold, c1, c2, c3, n)
    end

    posdef_fix!(ce.posdef, sigma; cov_flag = false, msg = "SB0 Correlation ")

    return Symmetric(sigma .* (std_vec * transpose(std_vec)))
end

function StatsBase.cor(ce::SB0, X::AbstractMatrix,
                       w::Union{AbstractWeights, Nothing} = nothing; dims::Int = 1,
                       kwargs...)
    normalise = ce.normalise
    threshold = ce.threshold
    c1 = ce.c1
    c2 = ce.c2
    c3 = ce.c3
    n = ce.n

    std_vec = vec(StatsBase.std(ce.se, X, w; dims = dims, kwargs...))
    mean_vec = vec(StatsBase.mean(ce.me, X, w; dims = dims))

    sigma = if normalise
        _sb0_norm(X, mean_vec, std_vec, threshold, c1, c2, c3, n)
    else
        _sb0(X, mean_vec, std_vec, threshold, c1, c2, c3, n)
    end

    posdef_fix!(ce.posdef, sigma; cov_flag = false, msg = "SB0 Correlation ")

    return Symmetric(sigma)
end

function StatsBase.cov(ce::SB1, X::AbstractMatrix,
                       w::Union{AbstractWeights, Nothing} = nothing; dims::Int = 1,
                       kwargs...)
    normalise = ce.normalise
    threshold = ce.threshold
    c1 = ce.c1
    c2 = ce.c2
    c3 = ce.c3
    n = ce.n

    std_vec = vec(StatsBase.std(ce.se, X, w; dims = dims, kwargs...))
    mean_vec = vec(StatsBase.mean(ce.me, X, w; dims = dims))

    sigma = if normalise
        _sb1_norm(X, mean_vec, std_vec, threshold, c1, c2, c3, n)
    else
        _sb1(X, mean_vec, std_vec, threshold, c1, c2, c3, n)
    end

    posdef_fix!(ce.posdef, sigma; cov_flag = false, msg = "SB1 Correlation ")

    return Symmetric(sigma .* (std_vec * transpose(std_vec)))
end

function StatsBase.cor(ce::SB1, X::AbstractMatrix,
                       w::Union{AbstractWeights, Nothing} = nothing; dims::Int = 1,
                       kwargs...)
    normalise = ce.normalise
    threshold = ce.threshold
    c1 = ce.c1
    c2 = ce.c2
    c3 = ce.c3
    n = ce.n

    std_vec = vec(StatsBase.std(ce.se, X, w; dims = dims, kwargs...))
    mean_vec = vec(StatsBase.mean(ce.me, X, w; dims = dims))

    sigma = if normalise
        _sb1_norm(X, mean_vec, std_vec, threshold, c1, c2, c3, n)
    else
        _sb1(X, mean_vec, std_vec, threshold, c1, c2, c3, n)
    end

    posdef_fix!(ce.posdef, sigma; cov_flag = false, msg = "SB1 Correlation ")

    return Symmetric(sigma)
end

function StatsBase.cov(ce::Gerber_SB0, X::AbstractMatrix,
                       w::Union{AbstractWeights, Nothing} = nothing; dims::Int = 1,
                       kwargs...)
    normalise = ce.normalise
    threshold = ce.threshold
    c1 = ce.c1
    c2 = ce.c2
    c3 = ce.c3
    n = ce.n

    std_vec = vec(StatsBase.std(ce.se, X, w; dims = dims, kwargs...))
    mean_vec = vec(StatsBase.mean(ce.me, X, w; dims = dims))

    sigma = if normalise
        _gerbersb0_norm(X, mean_vec, std_vec, threshold, c1, c2, c3, n)
    else
        _gerbersb0(X, mean_vec, std_vec, threshold, c1, c2, c3, n)
    end

    posdef_fix!(ce.posdef, sigma; cov_flag = false, msg = "Gerber_SB0 Correlation ")

    return Symmetric(sigma .* (std_vec * transpose(std_vec)))
end

function StatsBase.cor(ce::Gerber_SB0, X::AbstractMatrix,
                       w::Union{AbstractWeights, Nothing} = nothing; dims::Int = 1,
                       kwargs...)
    normalise = ce.normalise
    threshold = ce.threshold
    c1 = ce.c1
    c2 = ce.c2
    c3 = ce.c3
    n = ce.n

    std_vec = vec(StatsBase.std(ce.se, X, w; dims = dims, kwargs...))
    mean_vec = vec(StatsBase.mean(ce.me, X, w; dims = dims))

    sigma = if normalise
        _gerbersb0_norm(X, mean_vec, std_vec, threshold, c1, c2, c3, n)
    else
        _gerbersb0(X, mean_vec, std_vec, threshold, c1, c2, c3, n)
    end

    posdef_fix!(ce.posdef, sigma; cov_flag = false, msg = "Gerber_SB0 Correlation ")

    return Symmetric(sigma)
end

function StatsBase.cov(ce::Gerber_SB1, X::AbstractMatrix,
                       w::Union{AbstractWeights, Nothing} = nothing; dims::Int = 1,
                       kwargs...)
    normalise = ce.normalise
    threshold = ce.threshold
    c1 = ce.c1
    c2 = ce.c2
    c3 = ce.c3
    n = ce.n

    std_vec = vec(StatsBase.std(ce.se, X, w; dims = dims, kwargs...))
    mean_vec = vec(StatsBase.mean(ce.me, X, w; dims = dims))

    sigma = if normalise
        _gerbersb1_norm(X, mean_vec, std_vec, threshold, c1, c2, c3, n)
    else
        _gerbersb1(X, mean_vec, std_vec, threshold, c1, c2, c3, n)
    end

    posdef_fix!(ce.posdef, sigma; cov_flag = false, msg = "Gerber_SB1 Correlation ")

    return Symmetric(sigma .* (std_vec * transpose(std_vec)))
end

function StatsBase.cor(ce::Gerber_SB1, X::AbstractMatrix,
                       w::Union{AbstractWeights, Nothing} = nothing; dims::Int = 1,
                       kwargs...)
    normalise = ce.normalise
    threshold = ce.threshold
    c1 = ce.c1
    c2 = ce.c2
    c3 = ce.c3
    n = ce.n

    std_vec = vec(StatsBase.std(ce.se, X, w; dims = dims, kwargs...))
    mean_vec = vec(StatsBase.mean(ce.me, X, w; dims = dims))

    sigma = if normalise
        _gerbersb1_norm(X, mean_vec, std_vec, threshold, c1, c2, c3, n)
    else
        _gerbersb1(X, mean_vec, std_vec, threshold, c1, c2, c3, n)
    end

    posdef_fix!(ce.posdef, sigma; cov_flag = false, msg = "Gerber_SB1 Correlation ")

    return Symmetric(sigma)
end

function StatsBase.cor(ce::Cov2Cor, X::AbstractMatrix, args...; kwargs...)
    return Symmetric(cov2cor(isa(X, Matrix) ? X : Matrix(X)))
end

function StatsBase.cov(ce::Union{Kendall, Spearman}, X::AbstractMatrix,
                       w::Union{AbstractWeights, Nothing} = nothing; kwargs...)
    std_vec = vec(StatsBase.std(ce.se, X, w; kwargs...))

    sigma = cor(ce, X, w; kwargs...)

    return Symmetric(sigma .* (std_vec * transpose(std_vec)))
end

function StatsBase.cor(ce::Kendall, X::AbstractMatrix,
                       w::Union{AbstractWeights, Nothing} = nothing; kwargs...)
    return Symmetric(isnothing(w) ? corkendall(X) : corkendall(X .* w))
end

function StatsBase.cor(ce::Spearman, X::AbstractMatrix,
                       w::Union{AbstractWeights, Nothing} = nothing; kwargs...)
    return Symmetric(isnothing(w) ? corspearman(X) : corspearman(X .* w))
end

function cordistance(ce::DistanceCov, v1::AbstractVector, v2::AbstractVector)
    N = length(v1)
    @smart_assert(N == length(v2) && N > 1)

    N2 = N^2

    a = pairwise(ce.metric, v1, ce.args...; ce.kwargs...)
    b = pairwise(ce.metric, v2, ce.args...; ce.kwargs...)
    A = a .- mean(a; dims = 1) .- mean(a; dims = 2) .+ mean(a)
    B = b .- mean(b; dims = 1) .- mean(b; dims = 2) .+ mean(b)

    dcov2_xx = sum(A .* A) / N2
    dcov2_xy = sum(A .* B) / N2
    dcov2_yy = sum(B .* B) / N2

    val = sqrt(dcov2_xy) / sqrt(sqrt(dcov2_xx) * sqrt(dcov2_yy))

    return val
end

function cordistance(ce::DistanceCov, X::AbstractMatrix)
    N = size(X, 2)
    sigma = Matrix{eltype(X)}(undef, N, N)
    @inbounds for j ∈ 1:N
        xj = X[:, j]
        for i ∈ 1:j
            sigma[i, j] = cordistance(ce, X[:, i], xj)
        end
    end

    return Symmetric(sigma, :U)
end

function StatsBase.cov(ce::DistanceCov, X::AbstractMatrix,
                       w::Union{AbstractWeights, Nothing} = nothing; dims::Int = 1,
                       kwargs...)
    std_vec = vec(StatsBase.std(ce.se, X, w; dims = dims, kwargs...))

    sigma = cor(ce, X, w; kwargs...)

    return Symmetric(sigma .* (std_vec * transpose(std_vec)))
end

function StatsBase.cor(ce::DistanceCov, X::AbstractMatrix,
                       w::Union{AbstractWeights, Nothing} = nothing; kwargs...)
    return isnothing(w) ? cordistance(ce, X; kwargs...) : cordistance(ce, X .* w; kwargs...)
end

function StatsBase.cov(ce::TailCov, X::AbstractMatrix,
                       w::Union{AbstractWeights, Nothing} = nothing; dims::Int = 1,
                       kwargs...)
    std_vec = vec(StatsBase.std(ce.se, X, w; dims = dims, kwargs...))

    sigma = cor(ce, X, w; kwargs...)

    return Symmetric(sigma .* (std_vec * transpose(std_vec)))
end

function StatsBase.cor(ce::TailCov, X::AbstractMatrix,
                       w::Union{AbstractWeights, Nothing} = nothing; kwargs...)
    return isnothing(w) ? ltdi_mtx(X, ce.alpha) : ltdi_mtx(X .* w, ce.alpha)
end

function StatsBase.cov(ce::MutualInfoCov, X::AbstractMatrix,
                       w::Union{AbstractWeights, Nothing} = nothing; dims::Int = 1,
                       kwargs...)
    std_vec = vec(StatsBase.std(ce.se, X, w; dims = dims, kwargs...))

    sigma = cor(ce, X, w; kwargs...)

    return Symmetric(sigma .* (std_vec * transpose(std_vec)))
end

function StatsBase.cor(ce::MutualInfoCov, X::AbstractMatrix,
                       w::Union{AbstractWeights, Nothing} = nothing; kwargs...)
    return if isnothing(w)
        mutual_info_mtx(X, ce.bins, ce.normalise)
    else
        mutual_info_mtx(X .* w, ce.bins, ce.normalise)
    end
end

mutable struct ShrinkDenoiser <: PO_MatrixDenoiser
    alpha::Real
    detone::Bool
    mkt_comp::Integer
    kernel::Function
    m::Integer
    n::Integer
    args::Tuple
    kwargs::NamedTuple
end
function ShrinkDenoiser(; alpha::Real = 0.0, detone::Bool = false, mkt_comp::Integer = 1,
                        kernel = ASH.Kernels.gaussian, m::Integer = 10, n::Integer = 1000,
                        args::Tuple = (), kwargs::NamedTuple = (;))
    @smart_assert(zero(alpha) <= alpha <= one(alpha))
    return ShrinkDenoiser(alpha, detone, mkt_comp, kernel, m, n, args, kwargs)
end
function Base.setproperty!(obj::ShrinkDenoiser, sym::Symbol, val)
    if sym == :alpha
        @smart_assert(zero(val) <= val <= one(val))
    end
    return setfield!(obj, sym, val)
end

@kwdef mutable struct FixedDenoiser <: PO_MatrixDenoiser
    detone::Bool = false
    mkt_comp::Integer = 1
    kernel::Function = ASH.Kernels.gaussian
    m::Integer = 10
    n::Integer = 1000
    args::Tuple = ()
    kwargs::NamedTuple = (;)
end

@kwdef mutable struct SpectralDenoiser <: PO_MatrixDenoiser
    detone::Bool = false
    mkt_comp::Integer = 1
    kernel::Function = ASH.Kernels.gaussian
    m::Integer = 10
    n::Integer = 1000
    args::Tuple = ()
    kwargs::NamedTuple = (;)
end

struct NoDenoiser <: PO_MatrixDenoiser end

abstract type PO_JLoGO end
@kwdef mutable struct JLoGoCov <: PO_JLoGO
    metric::Distances.UnionMetric = Distances.Euclidean()
    func::Function = (corr, dist, args...; kwargs...) -> exp.(-dist)
    args::Tuple = ()
    kwargs::NamedTuple = (;)
end
struct NoJLoGo <: PO_JLoGO end

@kwdef mutable struct CovType <: PO_CovarianceEstimator
    absolute::Bool = false
    ce::PO_CovarianceEstimator = FullCov()
    denoiser::PO_MatrixDenoiser = NoDenoiser()
    posdef::NCM.NCMAlgorithm = NCM.Newton(; tau = 1e-12)
    jlogo::PO_JLoGO = NoJLoGo()
    w::Union{AbstractWeights, Nothing} = nothing
end

@kwdef mutable struct DistType <: PO_DistanceEstimator
    metric::Distances.UnionMetric = Distances.Euclidean()
    args::Tuple = ()
    kwargs::NamedTuple = (;)
end

function denoise_cor(::FixedDenoiser, vals, vecs, num_factors)
    _vals = copy(vals)
    _vals[1:num_factors] .= sum(_vals[1:num_factors]) / num_factors

    return cov2cor(vecs * Diagonal(_vals) * transpose(vecs))
end

function denoise_cor(::SpectralDenoiser, vals, vecs, num_factors)
    _vals = copy(vals)
    _vals[1:num_factors] .= 0

    return cov2cor(vecs * Diagonal(_vals) * transpose(vecs))
end

function denoise_cor(denoiser::ShrinkDenoiser, vals, vecs, num_factors)
    vals_l = vals[1:num_factors]
    vecs_l = vecs[:, 1:num_factors]

    # Large
    vals_r = vals[(num_factors + 1):end]
    vecs_r = vecs[:, (num_factors + 1):end]

    corr0 = vecs_r * Diagonal(vals_r) * transpose(vecs_r)
    corr1 = vecs_l * Diagonal(vals_l) * transpose(vecs_l)

    alpha = denoiser.alpha

    return corr0 + alpha * corr1 + (1 - alpha) * Diagonal(corr1)
end

function denoise(denoiser::PO_MatrixDenoiser, X::AbstractMatrix, q::Real, cov_flag = true)
    if cov_flag
        corr = cov2cor(X)
        s = sqrt.(diag(X))
    else
        corr = X
    end

    vals, vecs = eigen(corr)

    max_val, missing = find_max_eval(vals, q; kernel = denoiser.kernel, m = denoiser.m,
                                     n = denoiser.n, args = denoiser.args,
                                     kwargs = denoiser.kwargs)

    num_factors = findlast(vals .< max_val)

    corr = denoise_cor(denoiser, vals, vecs, num_factors)

    if denoiser.detone
        mkt_comp = denoiser.mkt_comp
        @smart_assert(one(size(X, 1)) <= mkt_comp <= size(X, 1))
        mkt_comp -= mkt_comp - 1
        _vals = Diagonal(vals)[(end - mkt_comp):end, (end - mkt_comp):end]
        _vecs = vecs[:, (end - mkt_comp):end]
        _corr = _vecs * _vals * transpose(_vecs)
        corr .-= _corr
    end

    return Symmetric(cov_flag ? cor2cov(corr, s) : corr)
end

function denoise(::NoDenoiser, X::AbstractMatrix, q::Real, cov_flag = true)
    return X
end

function StatsBase.cov(ce::CovType, X::AbstractMatrix)
    sigma = !ce.absolute ? cov(ce.ce, X, ce.w) : abs.(cov(ce.ce, X, ce.w))

    T, N = size(X)
    sigma = denoise(ce.denoiser, sigma, T / N, true)
    sigma = jlogo(ce.jlogo, sigma, true)

    return Symmetric(sigma)
end

function StatsBase.cor(ce::CovType, X::AbstractMatrix)
    sigma = !ce.absolute ? cor(ce.ce, X, ce.w) : abs.(cor(ce.ce, X, ce.w))

    T, N = size(X)
    sigma = denoise(ce.denoiser, sigma, T / N, false)
    sigma = jlogo(ce.jlogo, sigma, false)

    return Symmetric(sigma)
end

function dist(ce::DistType, X::AbstractMatrix)
    return Symmetric(reshape(pairwise(ce.metric, 1, X, ce.args...; ce.kwargs...), size(X)))
end

function jlogo(ce::JLoGoCov, X::AbstractMatrix, cov_flag = true)
    corr = cov_flag ? cov2cor(X) : X
    dist = pairwise(ce.metric, one(eltype(corr)), corr)
    S = ce.func(corr, dist, ce.args..., ; ce.kwargs...)
    separators, cliques = PMFG_T2s(S, 4)[3:4]
    return Symmetric(J_LoGo(X, separators, cliques) \ I)
end
function jlogo(::NoJLoGo, X, cov_flag = true)
    return Symmetric(X)
end

export FullCov, SemiCov, SimpleMean, Gerber0, Gerber1, Gerber2, SB0, SB1, Gerber_SB0,
       Gerber_SB1, TailCov, Cov2Cor, DistanceCov, Kendall, Spearman, MutualInfoCov,
       VariationInfo, FixedDenoiser, SpectralDenoiser, ShrinkDenoiser, NoDenoiser, CovType,
       JLoGoCov, DistType, dist
