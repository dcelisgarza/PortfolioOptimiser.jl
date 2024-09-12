"""
```
abstract type PortfolioOptimiserCovCor <: StatsBase.CovarianceEstimator end
```

Abstract type for subtyping portfolio optimiser covaraince and correlation estimators.
"""
abstract type PortfolioOptimiserCovCor <: StatsBase.CovarianceEstimator end

"""
```
abstract type CorPearson <: PortfolioOptimiserCovCor end
```

Abstract type for subtyping Pearson type covariance estimators.
"""
abstract type CorPearson <: PortfolioOptimiserCovCor end

"""
```
abstract type CorRank <: PortfolioOptimiserCovCor end
```

Abstract type for subtyping rank based covariance estimators.
"""
abstract type CorRank <: PortfolioOptimiserCovCor end

"""
```
@kwdef mutable struct CovFull <: CorPearson
    absolute::Bool = false
    ce::StatsBase.CovarianceEstimator = StatsBase.SimpleCovariance(; corrected = true)
    w::Union{<:AbstractWeights, Nothing} = nothing
end
```

Full Pearson-type covariance and correlation estimator.

# Parameters

  - `absolute`:

      + if `true`: compute an absolute correlation, `abs.(cor(X))`.

  - `ce`: [covariance estimator](https://juliastats.org/StatsBase.jl/stable/cov/#StatsBase.CovarianceEstimator).
  - `w`: weights for computing the covariance, if `nothing` apply no weights.
"""
mutable struct CovFull <: CorPearson
    absolute::Bool
    ce::StatsBase.CovarianceEstimator
    w::Union{<:AbstractWeights, Nothing}
end
function CovFull(; absolute::Bool = false,
                 ce::StatsBase.CovarianceEstimator = StatsBase.SimpleCovariance(;
                                                                                corrected = true),
                 w::Union{<:AbstractWeights, Nothing} = nothing)
    return CovFull(absolute, ce, w)
end

"""
```
@kwdef mutable struct SimpleVariance <: StatsBase.CovarianceEstimator
    corrected::Bool = true
end
```

Simple variance estimator.

# Parameters

  - `corrected`: if true `correct` the bias by dividing by `N-1`, if `false` the bias is not corrected and the division is by `N`.
"""
mutable struct SimpleVariance <: StatsBase.CovarianceEstimator
    corrected::Bool
end
function SimpleVariance(; corrected::Bool = true)
    return SimpleVariance(corrected)
end

"""
```
@kwdef mutable struct CovSemi <: CorPearson
    absolute::Bool = false
    ce::StatsBase.CovarianceEstimator = StatsBase.SimpleCovariance(; corrected = true)
    target::Union{<:Real, AbstractVector{<:Real}} = 0.0
    w::Union{<:AbstractWeights, Nothing} = nothing
end
```

Semi Pearson-type covariance and correlation estimator.

# Parameters

  - `absolute`:

      + if `true`: compute an absolute correlation, `abs.(cor(X))`.

  - `ce`: [covariance estimator](https://juliastats.org/StatsBase.jl/stable/cov/#StatsBase.CovarianceEstimator).
  - `target`: minimum acceptable return target.

      + `if isa(target, Real)`: apply the same target to all assets.
      + `if isa(target, AbstractVector)`: apply individual target to each asset.
  - `w`: weights for computing the covariance, if `nothing` apply no weights.
"""
mutable struct CovSemi <: CorPearson
    absolute::Bool
    ce::StatsBase.CovarianceEstimator
    target::Union{<:Real, AbstractVector{<:Real}}
    w::Union{<:AbstractWeights, Nothing}
end
function CovSemi(; absolute::Bool = false,
                 ce::StatsBase.CovarianceEstimator = StatsBase.SimpleCovariance(;
                                                                                corrected = true),
                 target::Union{<:Real, AbstractVector{<:Real}} = 0.0,
                 w::Union{<:AbstractWeights, Nothing} = nothing)
    return CovSemi(absolute, ce, target, w)
end

"""
```
@kwdef mutable struct CorSpearman <: CorRank
    absolute::Bool = false
end
```

Spearman type correlation estimator.

# Parameters

  - `absolute`:

      + if `true`: compute an absolute correlation, `abs.(corspearman(X))`.
"""
mutable struct CorSpearman <: CorRank
    absolute::Bool
end
function CorSpearman(; absolute::Bool = false)
    return CorSpearman(absolute)
end

"""
```
@kwdef mutable struct CorKendall <: CorRank
    absolute::Bool = false
end
```

Kendall type correlation estimator.

# Parameters

  - `absolute`:

      + if `flag`: compute an absolute correlation, `abs.(corkendall(X))`.
"""
mutable struct CorKendall <: CorRank
    absolute::Bool
end
function CorKendall(; absolute::Bool = false)
    return CorKendall(absolute)
end

const AbsoluteCovCor = Union{CovFull, CovSemi, CorSpearman, CorKendall}

"""
```
@kwdef mutable struct CorMutualInfo <: PortfolioOptimiserCovCor
    bins::Union{<:Integer, <:AbstractBins} = HGR()
    normalise::Bool = true
    ve::StatsBase.CovarianceEstimator = SimpleVariance()
    w::Union{<:AbstractWeights, Nothing} = nothing
end
```

Define the mutual information correlation matrix.

# Parameters

  - `bins`: defines the bin function, or bin width directly and if so `bins > 0`.
  - `normalise`: whether or not to normalise the mutual information.
  - `ve`: variance estimator.
  - `w`: variance weights, if `nothing`
"""
mutable struct CorMutualInfo <: PortfolioOptimiserCovCor
    bins::Union{<:Integer, <:AbstractBins}
    normalise::Bool
    ve::StatsBase.CovarianceEstimator
    w::Union{<:AbstractWeights, Nothing}
end
function CorMutualInfo(; bins::Union{<:Integer, <:AbstractBins} = HGR(),
                       normalise::Bool = true,
                       ve::StatsBase.CovarianceEstimator = SimpleVariance(),
                       w::Union{<:AbstractWeights, Nothing} = nothing)
    if isa(bins, Integer)
        @smart_assert(bins > zero(bins))
    end
    return CorMutualInfo(bins, normalise, ve, w)
end
function Base.setproperty!(obj::CorMutualInfo, sym::Symbol, val)
    if sym == :bins
        if isa(val, Integer)
            @smart_assert(val > zero(val))
        end
    end
    return setfield!(obj, sym, val)
end

"""
```
mutable struct CorDistance <: PortfolioOptimiserCovCor
    distance::Distances.UnionMetric
    dist_args::Tuple
    dist_kwargs::NamedTuple
    mean_w1::Union{<:AbstractWeights, Nothing}
    mean_w2::Union{<:AbstractWeights, Nothing}
    mean_w3::Union{<:AbstractWeights, Nothing}
end
```
"""
mutable struct CorDistance <: PortfolioOptimiserCovCor
    distance::Distances.UnionMetric
    dist_args::Tuple
    dist_kwargs::NamedTuple
    mean_w1::Union{<:AbstractWeights, Nothing}
    mean_w2::Union{<:AbstractWeights, Nothing}
    mean_w3::Union{<:AbstractWeights, Nothing}
end
function CorDistance(; distance::Distances.UnionMetric = Distances.Euclidean(),
                     dist_args::Tuple = (), dist_kwargs::NamedTuple = (;),
                     mean_w1::Union{<:AbstractWeights, Nothing} = nothing,
                     mean_w2::Union{<:AbstractWeights, Nothing} = nothing,
                     mean_w3::Union{<:AbstractWeights, Nothing} = nothing)
    return CorDistance(distance, dist_args, dist_kwargs, mean_w1, mean_w2, mean_w3)
end

"""
```
mutable struct CorLTD <: PortfolioOptimiserCovCor
    alpha::Real
    ve::StatsBase.CovarianceEstimator
    w::Union{<:AbstractWeights, Nothing}
end
```
"""
mutable struct CorLTD <: PortfolioOptimiserCovCor
    alpha::Real
    ve::StatsBase.CovarianceEstimator
    w::Union{<:AbstractWeights, Nothing}
end
function CorLTD(; alpha::Real = 0.05, ve::StatsBase.CovarianceEstimator = SimpleVariance(),
                w::Union{<:AbstractWeights, Nothing} = nothing)
    @smart_assert(zero(alpha) < alpha < one(alpha))
    return CorLTD(alpha, ve, w)
end
function Base.setproperty!(obj::CorLTD, sym::Symbol, val)
    if sym == :alpha
        @smart_assert(zero(val) <= val <= one(val))
    end
    return setfield!(obj, sym, val)
end

abstract type CorGerber <: PortfolioOptimiserCovCor end
abstract type CorGerberBasic <: CorGerber end
abstract type CorSB <: CorGerber end
abstract type CorGerberSB <: CorGerber end

"""
```
mutable struct CorGerber0{T1, T2, T3, T4, T5} <: CorSB
    normalise::Bool
    threshold::T1
    ve::StatsBase.CovarianceEstimator
    std_w::Union{<:AbstractWeights, Nothing}
    mean_w::Union{<:AbstractWeights, Nothing}
    posdef::PosdefFix
end
```
"""
mutable struct CorGerber0{T1 <: Real} <: CorGerberBasic
    normalise::Bool
    threshold::T1
    ve::StatsBase.CovarianceEstimator
    std_w::Union{<:AbstractWeights, Nothing}
    mean_w::Union{<:AbstractWeights, Nothing}
    posdef::PosdefFix
end
function CorGerber0(; normalise::Bool = false, threshold::Real = 0.5,
                    ve::StatsBase.CovarianceEstimator = SimpleVariance(),
                    std_w::Union{<:AbstractWeights, Nothing} = nothing,
                    mean_w::Union{<:AbstractWeights, Nothing} = nothing,
                    posdef::PosdefFix = PosdefNearest())
    @smart_assert(zero(threshold) < threshold < one(threshold))
    return CorGerber0{typeof(threshold)}(normalise, threshold, ve, std_w, mean_w, posdef)
end

"""
```
mutable struct CorGerber1{T1, T2, T3, T4, T5} <: CorSB
    normalise::Bool
    threshold::T1
    ve::StatsBase.CovarianceEstimator
    std_w::Union{<:AbstractWeights, Nothing}
    mean_w::Union{<:AbstractWeights, Nothing}
    posdef::PosdefFix
end
```
"""
mutable struct CorGerber1{T1 <: Real} <: CorGerberBasic
    normalise::Bool
    threshold::T1
    ve::StatsBase.CovarianceEstimator
    std_w::Union{<:AbstractWeights, Nothing}
    mean_w::Union{<:AbstractWeights, Nothing}
    posdef::PosdefFix
end
function CorGerber1(; normalise::Bool = false, threshold::Real = 0.5,
                    ve::StatsBase.CovarianceEstimator = SimpleVariance(),
                    std_w::Union{<:AbstractWeights, Nothing} = nothing,
                    mean_w::Union{<:AbstractWeights, Nothing} = nothing,
                    posdef::PosdefFix = PosdefNearest())
    @smart_assert(zero(threshold) < threshold < one(threshold))
    return CorGerber1{typeof(threshold)}(normalise, threshold, ve, std_w, mean_w, posdef)
end

"""
```
mutable struct CorGerber2{T1, T2, T3, T4, T5} <: CorSB
    normalise::Bool
    threshold::T1
    ve::StatsBase.CovarianceEstimator
    std_w::Union{<:AbstractWeights, Nothing}
    mean_w::Union{<:AbstractWeights, Nothing}
    posdef::PosdefFix
end
```
"""
mutable struct CorGerber2{T1 <: Real} <: CorGerberBasic
    normalise::Bool
    threshold::T1
    ve::StatsBase.CovarianceEstimator
    std_w::Union{<:AbstractWeights, Nothing}
    mean_w::Union{<:AbstractWeights, Nothing}
    posdef::PosdefFix
end
function CorGerber2(; normalise::Bool = false, threshold::Real = 0.5,
                    ve::StatsBase.CovarianceEstimator = SimpleVariance(),
                    std_w::Union{<:AbstractWeights, Nothing} = nothing,
                    mean_w::Union{<:AbstractWeights, Nothing} = nothing,
                    posdef::PosdefFix = PosdefNearest())
    @smart_assert(zero(threshold) < threshold < one(threshold))
    return CorGerber2{typeof(threshold)}(normalise, threshold, ve, std_w, mean_w, posdef)
end
function Base.setproperty!(obj::CorGerberBasic, sym::Symbol, val)
    if sym == :threshold
        @smart_assert(zero(val) < val < one(val))
    end
    return setfield!(obj, sym, val)
end

"""
```
mutable struct CorSB0{T1, T2, T3, T4, T5} <: CorSB
    normalise::Bool
    threshold::T1
    c1::T2
    c2::T3
    c3::T4
    n::T5
    ve::StatsBase.CovarianceEstimator
    std_w::Union{<:AbstractWeights, Nothing}
    mean_w::Union{<:AbstractWeights, Nothing}
    posdef::PosdefFix
end
```
"""
mutable struct CorSB0{T1, T2, T3, T4, T5} <: CorSB
    normalise::Bool
    threshold::T1
    c1::T2
    c2::T3
    c3::T4
    n::T5
    ve::StatsBase.CovarianceEstimator
    std_w::Union{<:AbstractWeights, Nothing}
    mean_w::Union{<:AbstractWeights, Nothing}
    posdef::PosdefFix
end
function CorSB0(; normalise::Bool = false, threshold::Real = 0.5, c1::Real = 0.5,
                c2::Real = 0.5, c3::Real = 4.0, n::Real = 2.0,
                ve::StatsBase.CovarianceEstimator = SimpleVariance(),
                std_w::Union{<:AbstractWeights, Nothing} = nothing,
                mean_w::Union{<:AbstractWeights, Nothing} = nothing,
                posdef::PosdefFix = PosdefNearest())
    @smart_assert(zero(threshold) < threshold < one(threshold))
    @smart_assert(zero(c1) < c1 <= one(c1))
    @smart_assert(zero(c2) < c2 <= one(c2))
    @smart_assert(c3 > c2)
    return CorSB0{typeof(threshold), typeof(c1), typeof(c2), typeof(c3), typeof(n)}(normalise,
                                                                                    threshold,
                                                                                    c1, c2,
                                                                                    c3, n,
                                                                                    ve,
                                                                                    std_w,
                                                                                    mean_w,
                                                                                    posdef)
end

"""
```
mutable struct CorSB1{T1, T2, T3, T4, T5} <: CorSB
    normalise::Bool
    threshold::T1
    c1::T2
    c2::T3
    c3::T4
    n::T5
    ve::StatsBase.CovarianceEstimator
    std_w::Union{<:AbstractWeights, Nothing}
    mean_w::Union{<:AbstractWeights, Nothing}
    posdef::PosdefFix
end
```
"""
mutable struct CorSB1{T1, T2, T3, T4, T5} <: CorSB
    normalise::Bool
    threshold::T1
    c1::T2
    c2::T3
    c3::T4
    n::T5
    ve::StatsBase.CovarianceEstimator
    std_w::Union{<:AbstractWeights, Nothing}
    mean_w::Union{<:AbstractWeights, Nothing}
    posdef::PosdefFix
end
function CorSB1(; normalise::Bool = false, threshold::Real = 0.5, c1::Real = 0.5,
                c2::Real = 0.5, c3::Real = 4.0, n::Real = 2.0,
                ve::StatsBase.CovarianceEstimator = SimpleVariance(),
                std_w::Union{<:AbstractWeights, Nothing} = nothing,
                mean_w::Union{<:AbstractWeights, Nothing} = nothing,
                posdef::PosdefFix = PosdefNearest())
    @smart_assert(zero(threshold) < threshold < one(threshold))
    @smart_assert(zero(c1) < c1 <= one(c1))
    @smart_assert(zero(c2) < c2 <= one(c2))
    @smart_assert(c3 > c2)
    return CorSB1{typeof(threshold), typeof(c1), typeof(c2), typeof(c3), typeof(n)}(normalise,
                                                                                    threshold,
                                                                                    c1, c2,
                                                                                    c3, n,
                                                                                    ve,
                                                                                    std_w,
                                                                                    mean_w,
                                                                                    posdef)
end

"""
```
mutable struct CorGerberSB0{T1, T2, T3, T4, T5} <: CorSB
    normalise::Bool
    threshold::T1
    c1::T2
    c2::T3
    c3::T4
    n::T5
    ve::StatsBase.CovarianceEstimator
    std_w::Union{<:AbstractWeights, Nothing}
    mean_w::Union{<:AbstractWeights, Nothing}
    posdef::PosdefFix
end
```
"""
mutable struct CorGerberSB0{T1, T2, T3, T4, T5} <: CorSB
    normalise::Bool
    threshold::T1
    c1::T2
    c2::T3
    c3::T4
    n::T5
    ve::StatsBase.CovarianceEstimator
    std_w::Union{<:AbstractWeights, Nothing}
    mean_w::Union{<:AbstractWeights, Nothing}
    posdef::PosdefFix
end
function CorGerberSB0(; normalise::Bool = false, threshold::Real = 0.5, c1::Real = 0.5,
                      c2::Real = 0.5, c3::Real = 4.0, n::Real = 2.0,
                      ve::StatsBase.CovarianceEstimator = SimpleVariance(),
                      std_w::Union{<:AbstractWeights, Nothing} = nothing,
                      mean_w::Union{<:AbstractWeights, Nothing} = nothing,
                      posdef::PosdefFix = PosdefNearest())
    @smart_assert(zero(threshold) < threshold < one(threshold))
    @smart_assert(zero(c1) < c1 <= one(c1))
    @smart_assert(zero(c2) < c2 <= one(c2))
    @smart_assert(c3 > c2)
    return CorGerberSB0{typeof(threshold), typeof(c1), typeof(c2), typeof(c3), typeof(n)}(normalise,
                                                                                          threshold,
                                                                                          c1,
                                                                                          c2,
                                                                                          c3,
                                                                                          n,
                                                                                          ve,
                                                                                          std_w,
                                                                                          mean_w,
                                                                                          posdef)
end

"""
```
mutable struct CorGerberSB1{T1, T2, T3, T4, T5} <: CorSB
    normalise::Bool
    threshold::T1
    c1::T2
    c2::T3
    c3::T4
    n::T5
    ve::StatsBase.CovarianceEstimator
    std_w::Union{<:AbstractWeights, Nothing}
    mean_w::Union{<:AbstractWeights, Nothing}
    posdef::PosdefFix
end
```
"""
mutable struct CorGerberSB1{T1, T2, T3, T4, T5} <: CorSB
    normalise::Bool
    threshold::T1
    c1::T2
    c2::T3
    c3::T4
    n::T5
    ve::StatsBase.CovarianceEstimator
    std_w::Union{<:AbstractWeights, Nothing}
    mean_w::Union{<:AbstractWeights, Nothing}
    posdef::PosdefFix
end
function CorGerberSB1(; normalise::Bool = false, threshold::Real = 0.5, c1::Real = 0.5,
                      c2::Real = 0.5, c3::Real = 4.0, n::Real = 2.0,
                      ve::StatsBase.CovarianceEstimator = SimpleVariance(),
                      std_w::Union{<:AbstractWeights, Nothing} = nothing,
                      mean_w::Union{<:AbstractWeights, Nothing} = nothing,
                      posdef::PosdefFix = PosdefNearest())
    @smart_assert(zero(threshold) < threshold < one(threshold))
    @smart_assert(zero(c1) < c1 <= one(c1))
    @smart_assert(zero(c2) < c2 <= one(c2))
    @smart_assert(c3 > c2)
    return CorGerberSB1{typeof(threshold), typeof(c1), typeof(c2), typeof(c3), typeof(n)}(normalise,
                                                                                          threshold,
                                                                                          c1,
                                                                                          c2,
                                                                                          c3,
                                                                                          n,
                                                                                          ve,
                                                                                          std_w,
                                                                                          mean_w,
                                                                                          posdef)
end
function Base.setproperty!(obj::CorSB, sym::Symbol, val)
    if sym == :threshold
        @smart_assert(zero(val) < val < one(val))
    elseif sym ∈ (:c1, :c2)
        @smart_assert(zero(val) < val <= one(val) && val < obj.c3)
    elseif sym == :c3
        @smart_assert(val > obj.c2)
    end
    return setfield!(obj, sym, val)
end

"""
```
abstract type AbstractLoGo end
```
"""
abstract type AbstractLoGo end

"""
```
struct NoLoGo <: AbstractLoGo end
```
"""
struct NoLoGo <: AbstractLoGo end

"""
```
@kwdef mutable struct LoGo <: AbstractLoGo
    distance::DistanceMethod = DistanceMLP()
    similarity::DBHTSimilarity = DBHTMaxDist()
end
```
"""
mutable struct LoGo <: AbstractLoGo
    distance::DistanceMethod
    similarity::DBHTSimilarity
end
function LoGo(; distance::DistanceMethod = DistanceMLP(),
              similarity::DBHTSimilarity = DBHTMaxDist())
    return LoGo(distance, similarity)
end

abstract type KurtEstimator end

"""
```
@kwdef mutable struct KurtFull <: KurtEstimator
    posdef::PosdefFix = PosdefNearest(;)
    denoise::Denoise = NoDenoise(;)
    logo::AbstractLoGo = NoLoGo(;)
end
```
"""
mutable struct KurtFull <: KurtEstimator
    posdef::PosdefFix
    denoise::Denoise
    logo::AbstractLoGo
end
function KurtFull(; posdef::PosdefFix = PosdefNearest(;), denoise::Denoise = NoDenoise(;),
                  logo::AbstractLoGo = NoLoGo(;))
    return KurtFull(posdef, denoise, logo)
end

"""
```
@kwdef mutable struct KurtSemi <: KurtEstimator
    target::Union{<:Real, AbstractVector{<:Real}} = 0.0
    posdef::PosdefFix = PosdefNearest(;)
    denoise::Denoise = NoDenoise(;)
    logo::AbstractLoGo = NoLoGo(;)
end
```
"""
mutable struct KurtSemi <: KurtEstimator
    target::Union{<:Real, AbstractVector{<:Real}}
    posdef::PosdefFix
    denoise::Denoise
    logo::AbstractLoGo
end
function KurtSemi(; target::Union{<:Real, AbstractVector{<:Real}} = 0.0,
                  posdef::PosdefFix = PosdefNearest(;), denoise::Denoise = NoDenoise(;),
                  logo::AbstractLoGo = NoLoGo(;))
    return KurtSemi(target, posdef, denoise, logo)
end

abstract type SkewEstimator end

"""
```
struct SkewFull <: SkewEstimator end
```
"""
struct SkewFull <: SkewEstimator end

"""
```
@kwdef mutable struct SkewSemi <: SkewEstimator
    target::Union{<:Real, AbstractVector{<:Real}} = 0.0
end
```
"""
mutable struct SkewSemi <: SkewEstimator
    target::Union{<:Real, AbstractVector{<:Real}}
end
function SkewSemi(; target::Union{<:Real, AbstractVector{<:Real}} = 0.0)
    return SkewSemi(target)
end

"""
```
@kwdef mutable struct PortCovCor <: PortfolioOptimiserCovCor
    ce::CovarianceEstimator = CovFull(;)
    posdef::PosdefFix = PosdefNearest(;)
    denoise::Denoise = NoDenoise(;)
    logo::AbstractLoGo = NoLoGo(;)
end
```
"""
mutable struct PortCovCor <: PortfolioOptimiserCovCor
    ce::CovarianceEstimator
    posdef::PosdefFix
    denoise::Denoise
    logo::AbstractLoGo
end
function PortCovCor(; ce::CovarianceEstimator = CovFull(;),
                    posdef::PosdefFix = PosdefNearest(;), denoise::Denoise = NoDenoise(;),
                    logo::AbstractLoGo = NoLoGo(;))
    return PortCovCor(ce, posdef, denoise, logo)
end

const PosdefFixCovCor = Union{<:CorGerber, PortCovCor}

export CovFull, SimpleVariance, CovSemi, CorSpearman, CorKendall, CorMutualInfo,
       CorDistance, CorLTD, CorGerber0, CorGerber1, CorGerber2, CorSB0, CorSB1,
       CorGerberSB0, CorGerberSB1, NoLoGo, LoGo, KurtFull, KurtSemi, SkewFull, SkewSemi,
       PortCovCor
