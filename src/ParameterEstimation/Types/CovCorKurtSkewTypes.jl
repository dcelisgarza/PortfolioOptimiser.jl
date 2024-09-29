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
@kwdef mutable struct CorMutualInfo <: PortfolioOptimiserCovCor
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
mutable struct CovDistance <: PortfolioOptimiserCovCor
    distance::Distances.UnionMetric
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
    distance::Distances.UnionMetric
    dist_args::Tuple
    dist_kwargs::NamedTuple
    mean_w1::Union{<:AbstractWeights, Nothing}
    mean_w2::Union{<:AbstractWeights, Nothing}
    mean_w3::Union{<:AbstractWeights, Nothing}
end
function CovDistance(; distance::Distances.UnionMetric = Distances.Euclidean(),
                     dist_args::Tuple = (), dist_kwargs::NamedTuple = (;),
                     mean_w1::Union{<:AbstractWeights, Nothing} = nothing,
                     mean_w2::Union{<:AbstractWeights, Nothing} = nothing,
                     mean_w3::Union{<:AbstractWeights, Nothing} = nothing)
    return CovDistance(distance, dist_args, dist_kwargs, mean_w1, mean_w2, mean_w3)
end

"""
```
mutable struct CorLTD <: PortfolioOptimiserCovCor
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

"""
```
abstract type CorGerber <: PortfolioOptimiserCovCor end
```

Abstract type for subtyping Gerber type covariance and correlation estimators.
"""
abstract type CorGerber <: PortfolioOptimiserCovCor end

"""
```
abstract type CorGerberBasic <: CorGerber end
```

Abstract type for subtyping the original Gerber type covariance and correlation estimators.
"""
abstract type CorGerberBasic <: CorGerber end

"""
```
abstract type CorSB <: CorGerber end
```

Abstract type for subtyping the Smyth-Broby modifications of Gerber type covariance and correlation estimators.
"""
abstract type CorSB <: CorGerber end

"""
```
abstract type CorSB <: CorGerber end
```

Abstract type for subtyping the Smyth-Broby modifications with vote counting of Gerber type covariance and correlation estimators.
"""
abstract type CorGerberSB <: CorGerber end

"""
```
@kwdef mutable struct CorGerber0{T1 <: Real} <: CorGerberBasic
    normalise::Bool = false
    threshold::T1 = 0.5
    ve::StatsBase.CovarianceEstimator = SimpleVariance()
    std_w::Union{<:AbstractWeights, Nothing} = nothing
    mean_w::Union{<:AbstractWeights, Nothing} = nothing
    posdef::PosdefFix = PosdefNearest()
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
  - `posdef`: method for fixing the Gerber type 0 correaltion matrix [`PosdefFix`](@ref).
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
@kwdef mutable struct CorGerber1{T1 <: Real} <: CorGerberBasic
    normalise::Bool = false
    threshold::T1 = 0.5
    ve::StatsBase.CovarianceEstimator = SimpleVariance()
    std_w::Union{<:AbstractWeights, Nothing} = nothing
    mean_w::Union{<:AbstractWeights, Nothing} = nothing
    posdef::PosdefFix = PosdefNearest()
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
  - `posdef`: method for fixing the Gerber type 1 correaltion matrix [`PosdefFix`](@ref).
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
@kwdef mutable struct CorGerber2{T1 <: Real} <: CorGerberBasic
    normalise::Bool = false
    threshold::T1 = 0.5
    ve::StatsBase.CovarianceEstimator = SimpleVariance()
    std_w::Union{<:AbstractWeights, Nothing} = nothing
    mean_w::Union{<:AbstractWeights, Nothing} = nothing
    posdef::PosdefFix = PosdefNearest()
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
  - `posdef`: method for fixing the Gerber type 2 correaltion matrix [`PosdefFix`](@ref).
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
@kwdef mutable struct CorSB0{T1, T2, T3, T4, T5} <: CorSB
    normalise::Bool = false
    threshold::T1 = 0.5
    c1::T2 = 0.5
    c2::T3 = 0.5
    c3::T4 = 4.0
    n::T5 = 2.0
    ve::StatsBase.CovarianceEstimator = SimpleVariance()
    std_w::Union{<:AbstractWeights, Nothing} = nothing
    mean_w::Union{<:AbstractWeights, Nothing} = nothing
    posdef::PosdefFix = PosdefNearest()
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
  - `posdef`: method for fixing the Smyth-Broby modification of the Gerber type 0 correaltion matrix [`PosdefFix`](@ref).
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
@kwdef mutable struct CorSB1{T1, T2, T3, T4, T5} <: CorSB
    normalise::Bool = false
    threshold::T1 = 0.5
    c1::T2 = 0.5
    c2::T3 = 0.5
    c3::T4 = 4.0
    n::T5 = 2.0
    ve::StatsBase.CovarianceEstimator = SimpleVariance()
    std_w::Union{<:AbstractWeights, Nothing} = nothing
    mean_w::Union{<:AbstractWeights, Nothing} = nothing
    posdef::PosdefFix = PosdefNearest()
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
  - `posdef`: method for fixing the Smyth-Broby modification of the Gerber type 1 correaltion matrix [`PosdefFix`](@ref).
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
@kwdef mutable struct CorGerberSB0{T1, T2, T3, T4, T5} <: CorSB
    normalise::Bool = false
    threshold::T1 = 0.5
    c1::T2 = 0.5
    c2::T3 = 0.5
    c3::T4 = 4.0
    n::T5 = 2.0
    ve::StatsBase.CovarianceEstimator = SimpleVariance()
    std_w::Union{<:AbstractWeights, Nothing} = nothing
    mean_w::Union{<:AbstractWeights, Nothing} = nothing
    posdef::PosdefFix = PosdefNearest()
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
  - `posdef`: method for fixing the Smyth-Broby modification with vote counting of the Gerber type 0 correaltion matrix [`PosdefFix`](@ref).
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
@kwdef mutable struct CorGerberSB1{T1, T2, T3, T4, T5} <: CorSB
    normalise::Bool = false
    threshold::T1 = 0.5
    c1::T2 = 0.5
    c2::T3 = 0.5
    c3::T4 = 4.0
    n::T5 = 2.0
    ve::StatsBase.CovarianceEstimator = SimpleVariance()
    std_w::Union{<:AbstractWeights, Nothing} = nothing
    mean_w::Union{<:AbstractWeights, Nothing} = nothing
    posdef::PosdefFix = PosdefNearest()
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
  - `posdef`: method for fixing the Smyth-Broby modification with vote counting of the Gerber type 1 correaltion matrix [`PosdefFix`](@ref).
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

Abstract type for subtyping LoGo covariance and correlation matrix estimators.
"""
abstract type AbstractLoGo end

"""
```
struct NoLoGo <: AbstractLoGo end
```

Leave the matrix as is.
"""
struct NoLoGo <: AbstractLoGo end

"""
```
@kwdef mutable struct LoGo <: AbstractLoGo
    distance::DistMethod = DistMLP()
    similarity::DBHTSimilarity = DBHTMaxDist()
end
```

Compute the LoGo covariance and correlation matrix estimator.

# Parameters

  - `distance`: method for computing the distance (disimilarity) matrix from the correlation matrix if the distance matrix is not provided to [`logo!`](@ref).
  - `similarity`: method for computing the similarity matrix from the correlation and distance matrices. The distance matrix is used to compute sparsity pattern of the inverse of the LoGo covariance and correlation matrices.
"""
mutable struct LoGo <: AbstractLoGo
    distance::DistMethod
    similarity::DBHTSimilarity
end
function LoGo(; distance::DistMethod = DistMLP(),
              similarity::DBHTSimilarity = DBHTMaxDist())
    return LoGo(distance, similarity)
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
    posdef::PosdefFix = PosdefNearest(;)
    denoise::Denoise = NoDenoise(;)
    logo::AbstractLoGo = NoLoGo(;)
end
```

Full cokurtosis estimator.

# Parameters

  - `posdef`: method for fixing non a positive definite cokurtosis matrix [`PosdefFix`](@ref).
  - `denoise`: method for denoising the cokurtosis matrix [`Denoise`](@ref).
  - `logo`: method for computing the LoGo cokurtosis matrix [`AbstractLoGo`](@ref).
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

Semi cokurtosis estimator.

# Parameters

  - `target`: minimum return threshold for classifying downside returns.

      + if `isa(target, Real)`: apply the same target to all assets.
      + if `isa(target, AbstractVector)`: apply individual target to each asset.

  - `posdef`: method for fixing non a positive definite semi cokurtosis matrix [`PosdefFix`](@ref).
  - `denoise`: method for denoising the semi cokurtosis matrix [`Denoise`](@ref).
  - `logo`: method for computing the LoGo semi cokurtosis matrix [`AbstractLoGo`](@ref).
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
    posdef::PosdefFix = PosdefNearest(;)
    denoise::Denoise = NoDenoise(;)
    logo::AbstractLoGo = NoLoGo(;)
end
```

PortfolioOptimiser covariance and correlation estimator.

# Parameters

  - `ce`: [covariance estimator](https://juliastats.org/StatsBase.jl/stable/cov/#StatsBase.CovarianceEstimator).
  - `posdef`: method for fixing the portfolio covariance or correlation matrix [`PosdefFix`](@ref).
  - `denoise`: method for denoising the portfolio covariance or correlation matrix [`Denoise`](@ref).
  - `logo`: method for computing the LoGo portfolio covariance or correlation matrix [`AbstractLoGo`](@ref).
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

"""
```
const PosdefFixCovCor = Union{<:CorGerber, PortCovCor}
```

Covariance and correlation estimators that support positive definite fixes.
"""
const PosdefFixCovCor = Union{<:CorGerber, PortCovCor}

export CovFull, SimpleVariance, CovSemi, CorSpearman, CorKendall, CorMutualInfo,
       CovDistance, CorLTD, CorGerber0, CorGerber1, CorGerber2, CorSB0, CorSB1,
       CorGerberSB0, CorGerberSB1, NoLoGo, LoGo, KurtFull, KurtSemi, SkewFull, SkewSemi,
       PortCovCor
