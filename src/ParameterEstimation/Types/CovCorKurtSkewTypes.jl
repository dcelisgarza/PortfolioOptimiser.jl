"""
```
    abstract type AbstractCustomMtxProcess end
```
"""
abstract type AbstractCustomMtxProcess end
struct NoCustomMtxProcess <: AbstractCustomMtxProcess end

"""
```
abstract type PortfolioOptimiserCovCor <: StatsBase.CovarianceEstimator end
```

Abstract type for subtyping portfolio covariance and correlation estimators.
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
  - `w`: optional `T×1` vector of weights for computing the covariance.
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

  - `corrected`:

      + if `true`: correct the bias dividing by `N-1` instead of `N`.
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
  - `target`: minimum return threshold for classifying downside returns.

      + if `isa(target, Real)`: apply the same target to all assets.
      + if `isa(target, AbstractVector)`: apply individual target to each asset.
  - `w`: optional `T×1` vector of weights for computing the covariance.
"""
mutable struct CovSemi{T1} <: CorPearson
    absolute::Bool
    ce::StatsBase.CovarianceEstimator
    target::T1
    mu::Union{<:AbstractVector, Nothing}
    cov_w::Union{<:AbstractWeights, Nothing}
    mu_w::Union{<:AbstractWeights, Nothing}
end
function CovSemi(; absolute::Bool = false,
                 ce::StatsBase.CovarianceEstimator = StatsBase.SimpleCovariance(;
                                                                                corrected = true),
                 target::Real = 0.0, mu::Union{<:AbstractVector, Nothing} = nothing,
                 cov_w::Union{<:AbstractWeights, Nothing} = nothing,
                 mu_w::Union{<:AbstractWeights, Nothing} = nothing)
    return CovSemi{typeof(target)}(absolute, ce, target, mu, cov_w, mu_w)
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

      + if `true`: compute an absolute correlation, `abs.(corkendall(X))`.
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
@kwdef mutable struct CovMutualInfo <: PortfolioOptimiserCovCor
    bins::Union{<:Integer, <:AbstractBins} = HGR()
    normalise::Bool = true
    ve::StatsBase.CovarianceEstimator = SimpleVariance()
    w::Union{<:AbstractWeights, Nothing} = nothing
end
```

Mutual information correlation matrix estimator.

# Parameters

  - `bins`:

      + if `isa(bins, AbstractBins)`: defines the function for computing bin widths.
      + if `isa(bins, Integer)` and `bins > 0`: directly provide the number of bins.

  - `normalise`:

      + if `true`: normalise the mutual information.
  - Only used when computing covariance matrices:

      + `ve`: variance estimator [`StatsBase.CovarianceEstimator`](https://juliastats.org/StatsBase.jl/stable/cov/#StatsBase.CovarianceEstimator).
      + `w`: optional `T×1` vector of weights for computing the variance.
"""
mutable struct CovMutualInfo <: PortfolioOptimiserCovCor
    bins::Union{<:Integer, <:AbstractBins}
    normalise::Bool
    ve::StatsBase.CovarianceEstimator
    w::Union{<:AbstractWeights, Nothing}
end
function CovMutualInfo(; bins::Union{<:Integer, <:AbstractBins} = HGR(),
                       normalise::Bool = true,
                       ve::StatsBase.CovarianceEstimator = SimpleVariance(),
                       w::Union{<:AbstractWeights, Nothing} = nothing)
    if isa(bins, Integer)
        @smart_assert(bins > zero(bins))
    end
    return CovMutualInfo(bins, normalise, ve, w)
end
function Base.setproperty!(obj::CovMutualInfo, sym::Symbol, val)
    if sym == :bins
        if isa(val, Integer)
            @smart_assert(val > zero(val))
        end
    end
    return setfield!(obj, sym, val)
end

"""
```
mutable struct CovDistance <: PortfolioOptimiserCovCor
    distance::Distances.Metric
    dist_args::Tuple
    dist_kwargs::NamedTuple
    mean_w1::Union{<:AbstractWeights, Nothing}
    mean_w2::Union{<:AbstractWeights, Nothing}
    mean_w3::Union{<:AbstractWeights, Nothing}
end
```

Distance covariance and correlation matrix estimator.

# Parameters

  - `distance`: distance metric from [Distances.jl](https://github.com/JuliaStats/Distances.jl).
  - `dist_args`: args for the `Distances.pairwise` function of [Distances.jl](https://github.com/JuliaStats/Distances.jl).
  - `dist_kwargs`: kwargs for the `Distances.pairwise` function of [Distances.jl](https://github.com/JuliaStats/Distances.jl).
  - `mean_w1`: optional `T×1` vector of weights for computing the mean of the pairwise distance matrices along its rows (`dims = 1`).
  - `mean_w2`: optional `T×1` vector of weights for computing the mean of the pairwise distance matrices along its columns (`dims = 2`).
  - `mean_w3`: optional `T×1` vector of weights for computing the mean of the entirety of the pairwise distance matrices.
"""
mutable struct CovDistance <: PortfolioOptimiserCovCor
    distance::Distances.Metric
    dist_args::Tuple
    dist_kwargs::NamedTuple
    mean_w1::Union{<:AbstractWeights, Nothing}
    mean_w2::Union{<:AbstractWeights, Nothing}
    mean_w3::Union{<:AbstractWeights, Nothing}
end
function CovDistance(; distance::Distances.Metric = Distances.Euclidean(),
                     dist_args::Tuple = (), dist_kwargs::NamedTuple = (;),
                     mean_w1::Union{<:AbstractWeights, Nothing} = nothing,
                     mean_w2::Union{<:AbstractWeights, Nothing} = nothing,
                     mean_w3::Union{<:AbstractWeights, Nothing} = nothing)
    return CovDistance(distance, dist_args, dist_kwargs, mean_w1, mean_w2, mean_w3)
end

"""
```
mutable struct CovLTD <: PortfolioOptimiserCovCor
    alpha::Real
    ve::StatsBase.CovarianceEstimator
    w::Union{<:AbstractWeights, Nothing}
end
```

Lower tail dependence correlation and covariance matrix estimator.

# Parameters

  - `alpha`: significance level of the lower tail dependence, `alpha ∈ (0, 1)`.

  - Only used when computing covariance matrices:

      + `ve`: variance estimator [`StatsBase.CovarianceEstimator`](https://juliastats.org/StatsBase.jl/stable/cov/#StatsBase.CovarianceEstimator).
      + `w`: optional `T×1` vector of weights for computing the variance.
"""
mutable struct CovLTD <: PortfolioOptimiserCovCor
    alpha::Real
    ve::StatsBase.CovarianceEstimator
    w::Union{<:AbstractWeights, Nothing}
end
function CovLTD(; alpha::Real = 0.05, ve::StatsBase.CovarianceEstimator = SimpleVariance(),
                w::Union{<:AbstractWeights, Nothing} = nothing)
    @smart_assert(zero(alpha) < alpha < one(alpha))
    return CovLTD(alpha, ve, w)
end
function Base.setproperty!(obj::CovLTD, sym::Symbol, val)
    if sym == :alpha
        @smart_assert(zero(val) <= val <= one(val))
    end
    return setfield!(obj, sym, val)
end

"""
```
abstract type CovGerber <: PortfolioOptimiserCovCor end
```

Abstract type for subtyping Gerber type covariance and correlation estimators.
"""
abstract type CovGerber <: PortfolioOptimiserCovCor end

"""
```
abstract type CovGerberBasic <: CovGerber end
```

Abstract type for subtyping the original Gerber type covariance and correlation estimators.
"""
abstract type CovGerberBasic <: CovGerber end

"""
```
abstract type CovSB <: CovGerber end
```

Abstract type for subtyping the Smyth-Broby modifications of Gerber type covariance and correlation estimators.
"""
abstract type CovSB <: CovGerber end

"""
```
abstract type CovSB <: CovGerber end
```

Abstract type for subtyping the Smyth-Broby modifications with vote counting of Gerber type covariance and correlation estimators.
"""
abstract type CovGerberSB <: CovGerber end

"""
```
@kwdef mutable struct CovGerber0{T1 <: Real} <: CovGerberBasic
    normalise::Bool = false
    threshold::T1 = 0.5
    ve::StatsBase.CovarianceEstimator = SimpleVariance()
    std_w::Union{<:AbstractWeights, Nothing} = nothing
    mean_w::Union{<:AbstractWeights, Nothing} = nothing
    posdef::AbstractPosdefFix = PosdefNearest()
end
```

Gerber type 0 covariance and correlation matrices.

# Parameters

  - `normalise`:

      + if `true`: Z-normalise the data before applying the Gerber criteria.

  - `threshold`: Gerber significance threshold, `threshold ∈ (0, 1)`.
  - `ve`: variance estimator [`StatsBase.CovarianceEstimator`](https://juliastats.org/StatsBase.jl/stable/cov/#StatsBase.CovarianceEstimator).
  - `std_w`: optional `T×1` vector of weights for computing the variance.
  - Only used when `normalise == true`:

      + `mean_w`: optional `T×1` vector of weights for computing the mean.
  - `posdef`: type for fixing the Gerber type 0 correaltion matrix [`AbstractPosdefFix`](@ref).
"""
mutable struct CovGerber0{T1 <: Real} <: CovGerberBasic
    normalise::Bool
    threshold::T1
    ve::StatsBase.CovarianceEstimator
    std_w::Union{<:AbstractWeights, Nothing}
    mean_w::Union{<:AbstractWeights, Nothing}
    posdef::AbstractPosdefFix
end
function CovGerber0(; normalise::Bool = false, threshold::Real = 0.5,
                    ve::StatsBase.CovarianceEstimator = SimpleVariance(),
                    std_w::Union{<:AbstractWeights, Nothing} = nothing,
                    mean_w::Union{<:AbstractWeights, Nothing} = nothing,
                    posdef::AbstractPosdefFix = PosdefNearest())
    @smart_assert(zero(threshold) < threshold < one(threshold))
    return CovGerber0{typeof(threshold)}(normalise, threshold, ve, std_w, mean_w, posdef)
end

"""
```
@kwdef mutable struct CovGerber1{T1 <: Real} <: CovGerberBasic
    normalise::Bool = false
    threshold::T1 = 0.5
    ve::StatsBase.CovarianceEstimator = SimpleVariance()
    std_w::Union{<:AbstractWeights, Nothing} = nothing
    mean_w::Union{<:AbstractWeights, Nothing} = nothing
    posdef::AbstractPosdefFix = PosdefNearest()
end
```

Gerber type 1 covariance and correlation matrices.

# Parameters

  - `normalise`:

      + if `true`: Z-normalise the data before applying the Gerber criteria.

  - `threshold`: Gerber significance threshold, `threshold ∈ (0, 1)`.
  - `ve`: variance estimator [`StatsBase.CovarianceEstimator`](https://juliastats.org/StatsBase.jl/stable/cov/#StatsBase.CovarianceEstimator).
  - `std_w`: optional `T×1` vector of weights for computing the variance.
  - Only used when `normalise == true`:

      + `mean_w`: optional `T×1` vector of weights for computing the mean.
  - `posdef`: type for fixing the Gerber type 1 correaltion matrix [`AbstractPosdefFix`](@ref).
"""
mutable struct CovGerber1{T1 <: Real} <: CovGerberBasic
    normalise::Bool
    threshold::T1
    ve::StatsBase.CovarianceEstimator
    std_w::Union{<:AbstractWeights, Nothing}
    mean_w::Union{<:AbstractWeights, Nothing}
    posdef::AbstractPosdefFix
end
function CovGerber1(; normalise::Bool = false, threshold::Real = 0.5,
                    ve::StatsBase.CovarianceEstimator = SimpleVariance(),
                    std_w::Union{<:AbstractWeights, Nothing} = nothing,
                    mean_w::Union{<:AbstractWeights, Nothing} = nothing,
                    posdef::AbstractPosdefFix = PosdefNearest())
    @smart_assert(zero(threshold) < threshold < one(threshold))
    return CovGerber1{typeof(threshold)}(normalise, threshold, ve, std_w, mean_w, posdef)
end

"""
```
@kwdef mutable struct CovGerber2{T1 <: Real} <: CovGerberBasic
    normalise::Bool = false
    threshold::T1 = 0.5
    ve::StatsBase.CovarianceEstimator = SimpleVariance()
    std_w::Union{<:AbstractWeights, Nothing} = nothing
    mean_w::Union{<:AbstractWeights, Nothing} = nothing
    posdef::AbstractPosdefFix = PosdefNearest()
end
```

Gerber type 2 covariance and correlation matrices.

# Parameters

  - `normalise`:

      + if `true`: Z-normalise the data before applying the Gerber criteria.

  - `threshold`: Gerber significance threshold, `threshold ∈ (0, 1)`.
  - `ve`: variance estimator [`StatsBase.CovarianceEstimator`](https://juliastats.org/StatsBase.jl/stable/cov/#StatsBase.CovarianceEstimator).
  - `std_w`: optional `T×1` vector of weights for computing the variance.
  - Only used when `normalise == true`:

      + `mean_w`: optional `T×1` vector of weights for computing the mean.
  - `posdef`: type for fixing the Gerber type 2 correaltion matrix [`AbstractPosdefFix`](@ref).
"""
mutable struct CovGerber2{T1 <: Real} <: CovGerberBasic
    normalise::Bool
    threshold::T1
    ve::StatsBase.CovarianceEstimator
    std_w::Union{<:AbstractWeights, Nothing}
    mean_w::Union{<:AbstractWeights, Nothing}
    posdef::AbstractPosdefFix
end
function CovGerber2(; normalise::Bool = false, threshold::Real = 0.5,
                    ve::StatsBase.CovarianceEstimator = SimpleVariance(),
                    std_w::Union{<:AbstractWeights, Nothing} = nothing,
                    mean_w::Union{<:AbstractWeights, Nothing} = nothing,
                    posdef::AbstractPosdefFix = PosdefNearest())
    @smart_assert(zero(threshold) < threshold < one(threshold))
    return CovGerber2{typeof(threshold)}(normalise, threshold, ve, std_w, mean_w, posdef)
end
function Base.setproperty!(obj::CovGerberBasic, sym::Symbol, val)
    if sym == :threshold
        @smart_assert(zero(val) < val < one(val))
    end
    return setfield!(obj, sym, val)
end

"""
```
@kwdef mutable struct CovSB0{T1, T2, T3, T4, T5} <: CovSB
    normalise::Bool = false
    threshold::T1 = 0.5
    c1::T2 = 0.5
    c2::T3 = 0.5
    c3::T4 = 4.0
    n::T5 = 2.0
    ve::StatsBase.CovarianceEstimator = SimpleVariance()
    std_w::Union{<:AbstractWeights, Nothing} = nothing
    mean_w::Union{<:AbstractWeights, Nothing} = nothing
    posdef::AbstractPosdefFix = PosdefNearest()
end
```

Smyth-Broby modification of the Gerber type 0 covariance and correlation matrices.

# Parameters

  - `normalise`:

      + if `true`: Z-normalise the data before applying the Gerber criteria.

  - `threshold`: Gerber significance threshold, `threshold ∈ (0, 1)`.
  - `c1`: confusion zone threshold (``c_1`` in the paper), `c1 ∈ (0, 1]`.
  - `c2`: indecision zone threshold (``c_2`` in the paper), `c2 ∈ (0, 1]`.
  - `c3`: large co-movement threshold (4 in the paper).
  - `n`: exponent of the regularisation term (``n = 2`` in the paper).
  - `ve`: variance estimator [`StatsBase.CovarianceEstimator`](https://juliastats.org/StatsBase.jl/stable/cov/#StatsBase.CovarianceEstimator).
  - `std_w`: optional `T×1` vector of weights for computing the variance.
  - `mean_w`: optional `T×1` vector of weights for computing the mean.
  - `posdef`: type for fixing the Smyth-Broby modification of the Gerber type 0 correaltion matrix [`AbstractPosdefFix`](@ref).
"""
mutable struct CovSB0{T1, T2, T3, T4, T5} <: CovSB
    normalise::Bool
    threshold::T1
    c1::T2
    c2::T3
    c3::T4
    n::T5
    ve::StatsBase.CovarianceEstimator
    std_w::Union{<:AbstractWeights, Nothing}
    mean_w::Union{<:AbstractWeights, Nothing}
    posdef::AbstractPosdefFix
end
function CovSB0(; normalise::Bool = false, threshold::Real = 0.5, c1::Real = 0.5,
                c2::Real = 0.5, c3::Real = 4.0, n::Real = 2.0,
                ve::StatsBase.CovarianceEstimator = SimpleVariance(),
                std_w::Union{<:AbstractWeights, Nothing} = nothing,
                mean_w::Union{<:AbstractWeights, Nothing} = nothing,
                posdef::AbstractPosdefFix = PosdefNearest())
    @smart_assert(zero(threshold) < threshold < one(threshold))
    @smart_assert(zero(c1) < c1 <= one(c1))
    @smart_assert(zero(c2) < c2 <= one(c2) && c3 > c2)
    return CovSB0{typeof(threshold), typeof(c1), typeof(c2), typeof(c3), typeof(n)}(normalise,
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
@kwdef mutable struct CovSB1{T1, T2, T3, T4, T5} <: CovSB
    normalise::Bool = false
    threshold::T1 = 0.5
    c1::T2 = 0.5
    c2::T3 = 0.5
    c3::T4 = 4.0
    n::T5 = 2.0
    ve::StatsBase.CovarianceEstimator = SimpleVariance()
    std_w::Union{<:AbstractWeights, Nothing} = nothing
    mean_w::Union{<:AbstractWeights, Nothing} = nothing
    posdef::AbstractPosdefFix = PosdefNearest()
end
```

Smyth-Broby modification of the Gerber type 1 covariance and correlation matrices.

# Parameters

  - `normalise`:

      + if `true`: Z-normalise the data before applying the Gerber criteria.

  - `threshold`: Gerber significance threshold, `threshold ∈ (0, 1)`.
  - `c1`: confusion zone threshold (``c_1`` in the paper), `c1 ∈ (0, 1]`.
  - `c2`: indecision zone threshold (``c_2`` in the paper), `c2 ∈ (0, 1]`.
  - `c3`: large co-movement threshold (4 in the paper).
  - `n`: exponent of the regularisation term (``n = 2`` in the paper).
  - `ve`: variance estimator [`StatsBase.CovarianceEstimator`](https://juliastats.org/StatsBase.jl/stable/cov/#StatsBase.CovarianceEstimator).
  - `std_w`: optional `T×1` vector of weights for computing the variance.
  - `mean_w`: optional `T×1` vector of weights for computing the mean.
  - `posdef`: type for fixing the Smyth-Broby modification of the Gerber type 1 correaltion matrix [`AbstractPosdefFix`](@ref).
"""
mutable struct CovSB1{T1, T2, T3, T4, T5} <: CovSB
    normalise::Bool
    threshold::T1
    c1::T2
    c2::T3
    c3::T4
    n::T5
    ve::StatsBase.CovarianceEstimator
    std_w::Union{<:AbstractWeights, Nothing}
    mean_w::Union{<:AbstractWeights, Nothing}
    posdef::AbstractPosdefFix
end
function CovSB1(; normalise::Bool = false, threshold::Real = 0.5, c1::Real = 0.5,
                c2::Real = 0.5, c3::Real = 4.0, n::Real = 2.0,
                ve::StatsBase.CovarianceEstimator = SimpleVariance(),
                std_w::Union{<:AbstractWeights, Nothing} = nothing,
                mean_w::Union{<:AbstractWeights, Nothing} = nothing,
                posdef::AbstractPosdefFix = PosdefNearest())
    @smart_assert(zero(threshold) < threshold < one(threshold))
    @smart_assert(zero(c1) < c1 <= one(c1))
    @smart_assert(zero(c2) < c2 <= one(c2) && c3 > c2)
    return CovSB1{typeof(threshold), typeof(c1), typeof(c2), typeof(c3), typeof(n)}(normalise,
                                                                                    threshold,
                                                                                    c1, c2,
                                                                                    c3, n,
                                                                                    ve,
                                                                                    std_w,
                                                                                    mean_w,
                                                                                    posdef)
end

"""
    CovSB2{T1, T2, T3, T4, T5} <: CovSB
"""
mutable struct CovSB2{T1, T2, T3, T4, T5} <: CovSB
    normalise::Bool
    threshold::T1
    c1::T2
    c2::T3
    c3::T4
    n::T5
    ve::StatsBase.CovarianceEstimator
    std_w::Union{<:AbstractWeights, Nothing}
    mean_w::Union{<:AbstractWeights, Nothing}
    posdef::AbstractPosdefFix
end
function CovSB2(; normalise::Bool = false, threshold::Real = 0.5, c1::Real = 0.5,
                c2::Real = 0.5, c3::Real = 4.0, n::Real = 2.0,
                ve::StatsBase.CovarianceEstimator = SimpleVariance(),
                std_w::Union{<:AbstractWeights, Nothing} = nothing,
                mean_w::Union{<:AbstractWeights, Nothing} = nothing,
                posdef::AbstractPosdefFix = PosdefNearest())
    @smart_assert(zero(threshold) < threshold < one(threshold))
    @smart_assert(zero(c1) < c1 <= one(c1))
    @smart_assert(zero(c2) < c2 <= one(c2) && c3 > c2)
    return CovSB2{typeof(threshold), typeof(c1), typeof(c2), typeof(c3), typeof(n)}(normalise,
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
@kwdef mutable struct CovGerberSB0{T1, T2, T3, T4, T5} <: CovSB
    normalise::Bool = false
    threshold::T1 = 0.5
    c1::T2 = 0.5
    c2::T3 = 0.5
    c3::T4 = 4.0
    n::T5 = 2.0
    ve::StatsBase.CovarianceEstimator = SimpleVariance()
    std_w::Union{<:AbstractWeights, Nothing} = nothing
    mean_w::Union{<:AbstractWeights, Nothing} = nothing
    posdef::AbstractPosdefFix = PosdefNearest()
end
```

Smyth-Broby modification with vote counting of the Gerber type 0 covariance and correlation matrices.

# Parameters

  - `normalise`:

      + if `true`: Z-normalise the data before applying the Gerber criteria.

  - `threshold`: Gerber significance threshold, `threshold ∈ (0, 1)`.
  - `c1`: confusion zone threshold (``c_1`` in the paper), `c1 ∈ (0, 1]`.
  - `c2`: indecision zone threshold (``c_2`` in the paper), `c2 ∈ (0, 1]`.
  - `c3`: large co-movement threshold (4 in the paper).
  - `n`: exponent of the regularisation term (``n = 2`` in the paper).
  - `ve`: variance estimator [`StatsBase.CovarianceEstimator`](https://juliastats.org/StatsBase.jl/stable/cov/#StatsBase.CovarianceEstimator).
  - `std_w`: optional `T×1` vector of weights for computing the variance.
  - `mean_w`: optional `T×1` vector of weights for computing the mean.
  - `posdef`: type for fixing the Smyth-Broby modification with vote counting of the Gerber type 0 correaltion matrix [`AbstractPosdefFix`](@ref).
"""
mutable struct CovGerberSB0{T1, T2, T3, T4, T5} <: CovSB
    normalise::Bool
    threshold::T1
    c1::T2
    c2::T3
    c3::T4
    n::T5
    ve::StatsBase.CovarianceEstimator
    std_w::Union{<:AbstractWeights, Nothing}
    mean_w::Union{<:AbstractWeights, Nothing}
    posdef::AbstractPosdefFix
end
function CovGerberSB0(; normalise::Bool = false, threshold::Real = 0.5, c1::Real = 0.5,
                      c2::Real = 0.5, c3::Real = 4.0, n::Real = 2.0,
                      ve::StatsBase.CovarianceEstimator = SimpleVariance(),
                      std_w::Union{<:AbstractWeights, Nothing} = nothing,
                      mean_w::Union{<:AbstractWeights, Nothing} = nothing,
                      posdef::AbstractPosdefFix = PosdefNearest())
    @smart_assert(zero(threshold) < threshold < one(threshold))
    @smart_assert(zero(c1) < c1 <= one(c1))
    @smart_assert(zero(c2) < c2 <= one(c2) && c3 > c2)
    return CovGerberSB0{typeof(threshold), typeof(c1), typeof(c2), typeof(c3), typeof(n)}(normalise,
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
@kwdef mutable struct CovGerberSB1{T1, T2, T3, T4, T5} <: CovSB
    normalise::Bool = false
    threshold::T1 = 0.5
    c1::T2 = 0.5
    c2::T3 = 0.5
    c3::T4 = 4.0
    n::T5 = 2.0
    ve::StatsBase.CovarianceEstimator = SimpleVariance()
    std_w::Union{<:AbstractWeights, Nothing} = nothing
    mean_w::Union{<:AbstractWeights, Nothing} = nothing
    posdef::AbstractPosdefFix = PosdefNearest()
end
```

Smyth-Broby modification with vote counting of the Gerber type 1 covariance and correlation matrices.

# Parameters

  - `normalise`:

      + if `true`: Z-normalise the data before applying the Gerber criteria.

  - `threshold`: Gerber significance threshold, `threshold ∈ (0, 1)`.
  - `c1`: confusion zone threshold (``c_1`` in the paper), `c1 ∈ (0, 1]`.
  - `c2`: indecision zone threshold (``c_2`` in the paper), `c2 ∈ (0, 1]`.
  - `c3`: large co-movement threshold (4 in the paper).
  - `n`: exponent of the regularisation term (``n = 2`` in the paper).
  - `ve`: variance estimator [`StatsBase.CovarianceEstimator`](https://juliastats.org/StatsBase.jl/stable/cov/#StatsBase.CovarianceEstimator).
  - `std_w`: optional `T×1` vector of weights for computing the variance.
  - `mean_w`: optional `T×1` vector of weights for computing the mean.
  - `posdef`: type for fixing the Smyth-Broby modification with vote counting of the Gerber type 1 correaltion matrix [`AbstractPosdefFix`](@ref).
"""
mutable struct CovGerberSB1{T1, T2, T3, T4, T5} <: CovSB
    normalise::Bool
    threshold::T1
    c1::T2
    c2::T3
    c3::T4
    n::T5
    ve::StatsBase.CovarianceEstimator
    std_w::Union{<:AbstractWeights, Nothing}
    mean_w::Union{<:AbstractWeights, Nothing}
    posdef::AbstractPosdefFix
end
function CovGerberSB1(; normalise::Bool = false, threshold::Real = 0.5, c1::Real = 0.5,
                      c2::Real = 0.5, c3::Real = 4.0, n::Real = 2.0,
                      ve::StatsBase.CovarianceEstimator = SimpleVariance(),
                      std_w::Union{<:AbstractWeights, Nothing} = nothing,
                      mean_w::Union{<:AbstractWeights, Nothing} = nothing,
                      posdef::AbstractPosdefFix = PosdefNearest())
    @smart_assert(zero(threshold) < threshold < one(threshold))
    @smart_assert(zero(c1) < c1 <= one(c1))
    @smart_assert(zero(c2) < c2 <= one(c2) && c3 > c2)
    return CovGerberSB1{typeof(threshold), typeof(c1), typeof(c2), typeof(c3), typeof(n)}(normalise,
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
    mutable struct CovGerberSB2{T1, T2, T3, T4, T5} <: CovSB
"""
mutable struct CovGerberSB2{T1, T2, T3, T4, T5} <: CovSB
    normalise::Bool
    threshold::T1
    c1::T2
    c2::T3
    c3::T4
    n::T5
    ve::StatsBase.CovarianceEstimator
    std_w::Union{<:AbstractWeights, Nothing}
    mean_w::Union{<:AbstractWeights, Nothing}
    posdef::AbstractPosdefFix
end
function CovGerberSB2(; normalise::Bool = false, threshold::Real = 0.5, c1::Real = 0.5,
                      c2::Real = 0.5, c3::Real = 4.0, n::Real = 2.0,
                      ve::StatsBase.CovarianceEstimator = SimpleVariance(),
                      std_w::Union{<:AbstractWeights, Nothing} = nothing,
                      mean_w::Union{<:AbstractWeights, Nothing} = nothing,
                      posdef::AbstractPosdefFix = PosdefNearest())
    @smart_assert(zero(threshold) < threshold < one(threshold))
    @smart_assert(zero(c1) < c1 <= one(c1))
    @smart_assert(zero(c2) < c2 <= one(c2) && c3 > c2)
    return CovGerberSB2{typeof(threshold), typeof(c1), typeof(c2), typeof(c3), typeof(n)}(normalise,
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
function Base.setproperty!(obj::CovSB, sym::Symbol, val)
    if sym == :threshold
        @smart_assert(zero(val) < val < one(val))
    elseif sym == :c1
        @smart_assert(zero(val) < val <= one(val))
    elseif sym == :c2
        @smart_assert(zero(val) < val <= one(val) && obj.c3 > val)
    elseif sym == :c3
        @smart_assert(val > obj.c2)
    end
    return setfield!(obj, sym, val)
end

"""
```
abstract type KurtEstimator end
```

Abstract type for subtyping cokurtosis estimators.
"""
abstract type KurtEstimator end

"""
```
@kwdef mutable struct KurtFull <: KurtEstimator
    posdef::AbstractPosdefFix = PosdefNearest(;)
    denoise::AbstractDenoise = NoDenoise(;)
    logo::AbstractLoGo = NoLoGo(;)
end
```

Full cokurtosis estimator.

# Parameters

  - `posdef`: type for fixing non a positive definite cokurtosis matrix [`AbstractPosdefFix`](@ref).
  - `denoise`: type for denoising the cokurtosis matrix [`AbstractDenoise`](@ref).
  - `logo`: type for computing the LoGo cokurtosis matrix [`AbstractLoGo`](@ref).
"""
mutable struct KurtFull <: KurtEstimator
    posdef::AbstractPosdefFix
    denoise::AbstractDenoise
    detone::AbstractDetone
    logo::AbstractLoGo
    custom::AbstractCustomMtxProcess
end
function KurtFull(; posdef::AbstractPosdefFix = PosdefNearest(;),
                  denoise::AbstractDenoise = NoDenoise(;),
                  detone::AbstractDetone = NoDetone(;), logo::AbstractLoGo = NoLoGo(;),
                  custom::AbstractCustomMtxProcess = NoCustomMtxProcess())
    return KurtFull(posdef, denoise, detone, logo, custom)
end

"""
```
@kwdef mutable struct KurtSemi <: KurtEstimator
    target::Union{<:Real, AbstractVector{<:Real}} = 0.0
    posdef::AbstractPosdefFix = PosdefNearest(;)
    denoise::AbstractDenoise = NoDenoise(;)
    logo::AbstractLoGo = NoLoGo(;)
end
```

Semi cokurtosis estimator.

# Parameters

  - `target`: minimum return threshold for classifying downside returns.

      + if `isa(target, Real)`: apply the same target to all assets.
      + if `isa(target, AbstractVector)`: apply individual target to each asset.

  - `posdef`: type for fixing non a positive definite semi cokurtosis matrix [`AbstractPosdefFix`](@ref).
  - `denoise`: type for denoising the semi cokurtosis matrix [`AbstractDenoise`](@ref).
  - `logo`: type for computing the LoGo semi cokurtosis matrix [`AbstractLoGo`](@ref).
"""
mutable struct KurtSemi <: KurtEstimator
    target::Union{<:Real, AbstractVector{<:Real}}
    posdef::AbstractPosdefFix
    denoise::AbstractDenoise
    detone::AbstractDetone
    logo::AbstractLoGo
    custom::AbstractCustomMtxProcess
end
function KurtSemi(; target::Union{<:Real, AbstractVector{<:Real}} = 0.0,
                  posdef::AbstractPosdefFix = PosdefNearest(;),
                  denoise::AbstractDenoise = NoDenoise(;),
                  detone::AbstractDetone = NoDetone(;), logo::AbstractLoGo = NoLoGo(;),
                  custom::AbstractCustomMtxProcess = NoCustomMtxProcess())
    return KurtSemi(target, posdef, denoise, detone, logo, custom)
end

"""
```
abstract type SkewEstimator end
```

Abstract type for subtyping coskew estimators.
"""
abstract type SkewEstimator end

"""
```
struct SkewFull <: SkewEstimator end
```

Full cokurtosis estimator.
"""
struct SkewFull <: SkewEstimator end

"""
```
@kwdef mutable struct SkewSemi <: SkewEstimator
    target::Union{<:Real, AbstractVector{<:Real}} = 0.0
end
```

Semi cokurtosis estimator.

# Parameters

  - `target`: minimum return threshold for classifying downside returns.

      + if `isa(target, Real)`: apply the same target to all assets.
      + if `isa(target, AbstractVector)`: apply individual target to each asset.
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
    posdef::AbstractPosdefFix = PosdefNearest(;)
    denoise::AbstractDenoise = NoDenoise(;)
    logo::AbstractLoGo = NoLoGo(;)
end
```

PortfolioOptimiser covariance and correlation estimator.

# Parameters

  - `ce`: [covariance estimator](https://juliastats.org/StatsBase.jl/stable/cov/#StatsBase.CovarianceEstimator).
  - `posdef`: type for fixing the portfolio covariance or correlation matrix [`AbstractPosdefFix`](@ref).
  - `denoise`: type for denoising the portfolio covariance or correlation matrix [`AbstractDenoise`](@ref).
  - `logo`: type for computing the LoGo portfolio covariance or correlation matrix [`AbstractLoGo`](@ref).
"""
mutable struct PortCovCor <: PortfolioOptimiserCovCor
    ce::CovarianceEstimator
    posdef::AbstractPosdefFix
    denoise::AbstractDenoise
    detone::AbstractDetone
    logo::AbstractLoGo
    custom::AbstractCustomMtxProcess
end
function PortCovCor(; ce::CovarianceEstimator = CovFull(;),
                    posdef::AbstractPosdefFix = PosdefNearest(;),
                    denoise::AbstractDenoise = NoDenoise(;),
                    detone::AbstractDetone = NoDetone(), logo::AbstractLoGo = NoLoGo(;),
                    custom::AbstractCustomMtxProcess = NoCustomMtxProcess())
    return PortCovCor(ce, posdef, denoise, detone, logo, custom)
end

"""
```
const PosdefFixCovCor = Union{<:CovGerber, PortCovCor}
```

Covariance and correlation estimators that support positive definite fixes.
"""
const PosdefFixCovCor = Union{<:CovGerber, PortCovCor}

export CovFull, SimpleVariance, CovSemi, CorSpearman, CorKendall, CovMutualInfo,
       CovDistance, CovLTD, CovGerber0, CovGerber1, CovGerber2, CovSB0, CovSB1, CovSB2,
       CovGerberSB0, CovGerberSB1, CovGerberSB2, KurtFull, KurtSemi, SkewFull, SkewSemi,
       PortCovCor, NoCustomMtxProcess
