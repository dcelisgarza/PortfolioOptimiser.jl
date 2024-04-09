import StatsBase: cov, cor, mean, std

abstract type PO_CovarianceEstimator <: StatsBase.CovarianceEstimator end
abstract type MeanEstimator end
struct SimpleMean <: MeanEstimator end

abstract type PosdefFix end
mutable struct PosdefNearestCor <: PosdefFix
    method::Union{UnionAll, NCMAlgorithm}
end
function PosdefNearestCor(; method = NCM.Newton)
    return PosdefNearestCor(method)
end
@kwdef mutable struct PosdefPSD <: PosdefFix
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

function posdef_fix!(pf::PosdefNearestCor, X::AbstractMatrix; cov_flag::Bool = true,
                     msg::String = "")
    if isposdef(X)
        return nothing
    end
    _X = posdef_nearest(X, pf.method; cov_flag = cov_flag)
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
mutable struct Gerber0{T1 <: Real} <: PO_CovarianceEstimator
    ce::CovarianceEstimator
    me::MeanEstimator
    posdef::PosdefFix
    threshold::T1
    normalise::Bool
end
function Gerber0(; ce::CovarianceEstimator = StatsBase.SimpleCovariance(; corrected = true),
                 me::MeanEstimator = SimpleMean(), posdef::PosdefFix = PosdefNearestCor(;),
                 threshold::Real = 0.5, normalise::Bool = false)
    @smart_assert(zero(threshold) < threshold < one(threshold))

    return Gerber0(ce, me, posdef, threshold, normalise)
end
mutable struct Gerber1{T1 <: Real} <: PO_CovarianceEstimator
    ce::CovarianceEstimator
    me::MeanEstimator
    posdef::PosdefFix
    threshold::T1
    normalise::Bool
end
function Gerber1(; ce::CovarianceEstimator = StatsBase.SimpleCovariance(; corrected = true),
                 me::MeanEstimator = SimpleMean(), posdef::PosdefFix = PosdefNearestCor(;),
                 threshold::Real = 0.5, normalise::Bool = false)
    @smart_assert(zero(threshold) < threshold < one(threshold))

    return Gerber1(ce, me, posdef, threshold, normalise)
end
mutable struct Gerber2{T1 <: Real} <: PO_CovarianceEstimator
    ce::CovarianceEstimator
    me::MeanEstimator
    posdef::PosdefFix
    threshold::T1
    normalise::Bool
end
function Gerber2(; ce::CovarianceEstimator = StatsBase.SimpleCovariance(; corrected = true),
                 me::MeanEstimator = SimpleMean(), posdef::PosdefFix = PosdefNearestCor(;),
                 threshold::Real = 0.5, normalise::Bool = false)
    @smart_assert(zero(threshold) < threshold < one(threshold))

    return Gerber2(ce, me, posdef, threshold, normalise)
end

function Base.setproperty!(obj::Union{Gerber0, Gerber1, Gerber2}, sym::Symbol, val)
    if sym == :threshold
        @smart_assert(zero(val) < val < one(val))
    end
    return setfield!(obj, sym, val)
end

function StatsBase.cov(ce::FullCov, X::AbstractMatrix,
                       w::Union{AbstractWeights, Nothing} = nothing; mean = nothing,
                       dims::Int = 1)
    return if isnothing(w)
        StatsBase.cov(ce.ce, X; mean = mean, dims = dims)
    else
        StatsBase.cov(ce.ce, X, w; mean = mean, dims = dims)
    end
end
function StatsBase.cov(ce::SemiCov, X::AbstractMatrix,
                       w::Union{AbstractWeights, Nothing} = nothing; mean = nothing,
                       dims::Int = 1)
    X = if isa(ce.target, Real)
        min.(X .- ce.target, zero(eltype(X)))
    else
        min.(X .- transpose(ce.target), zero(eltype(X)))
    end

    mean = zero(eltype(X))

    return if isnothing(w)
        StatsBase.cov(ce.ce, X; mean = mean, dims = dims)
    else
        StatsBase.cov(ce.ce, X, w; mean = mean, dims = dims)
    end
end

function StatsBase.mean(::SimpleMean, X::AbstractArray,
                        w::Union{AbstractWeights, Nothing} = nothing; dims::Int = 1)
    return if isnothing(w)
        StatsBase.mean(X; dims = dims)
    else
        StatsBase.mean(X, w; dims = dims)
    end
end

function StatsBase.std(ce::StatsBase.SimpleCovariance, X::AbstractArray,
                       w::Union{AbstractWeights, Nothing} = nothing; mean = nothing,
                       dims::Int = 1)
    return if isnothing(w)
        StatsBase.std(X; corrected = ce.corrected, mean = mean, dims = dims)
    else
        StatsBase.std(X .* w; corrected = ce.corrected, mean = mean, dims = dims)
    end
end

function StatsBase.cov(ce::Gerber0, X::AbstractMatrix,
                       w::Union{AbstractWeights, Nothing} = nothing; mean = nothing,
                       dims::Int = 1)
    threshold = ce.threshold
    normalise = ce.normalise

    std_vec = vec(StatsBase.std(ce.ce, X, w; mean = mean, dims = dims))

    sigma = if normalise
        me = ce.me
        mean_vec = vec(StatsBase.mean(me, X, w; dims = dims))
        _gerber0_norm(X, mean_vec, std_vec, threshold)
    else
        _gerber0(X, std_vec, threshold)
    end

    posdef_fix!(ce.posdef, sigma; cov_flag = false, msg = "Gerber0 Correlation ")

    return sigma, Matrix(Symmetric(sigma .* (std_vec * transpose(std_vec)), :U))
end

function StatsBase.cov(ce::Gerber1, X::AbstractMatrix,
                       w::Union{AbstractWeights, Nothing} = nothing; mean = nothing,
                       dims::Int = 1)
    threshold = ce.threshold
    normalise = ce.normalise

    std_vec = vec(StatsBase.std(ce.ce, X, w; mean = mean, dims = dims))

    sigma = if normalise
        me = ce.me
        mean_vec = vec(StatsBase.mean(me, X, w; dims = dims))
        _gerber1_norm(X, mean_vec, std_vec, threshold)
    else
        _gerber1(X, std_vec, threshold)
    end

    posdef_fix!(ce.posdef, sigma; cov_flag = false, msg = "Gerber1 Correlation ")

    return sigma, Matrix(Symmetric(sigma .* (std_vec * transpose(std_vec)), :U))
end

function StatsBase.cov(ce::Gerber2, X::AbstractMatrix,
                       w::Union{AbstractWeights, Nothing} = nothing; mean = nothing,
                       dims::Int = 1)
    threshold = ce.threshold
    normalise = ce.normalise

    std_vec = vec(StatsBase.std(ce.ce, X, w; mean = mean, dims = dims))

    sigma = if normalise
        me = ce.me
        mean_vec = vec(StatsBase.mean(me, X, w; dims = dims))
        _gerber2_norm(X, mean_vec, std_vec, threshold)
    else
        _gerber2(X, std_vec, threshold)
    end

    posdef_fix!(ce.posdef, sigma; cov_flag = false, msg = "Gerber2 Correlation ")

    return sigma, Matrix(Symmetric(sigma .* (std_vec * transpose(std_vec)), :U))
end

export FullCov, SemiCov, SimpleMean, Gerber0, Gerber1, Gerber2

# https://en.wikipedia.org/wiki/Sample_mean_and_covariance#Weighted_samples
# (1/(1-sum(w.^2)))*(w*w' .* cov) for weighted covariance, where w is the weight.
