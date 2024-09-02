# # Parameter estimation

# ## Postitive definite matrices

"""
```
abstract type PosdefFix end
```

Abstract type for subtyping methods for fixing non positive definite matrices.
"""
abstract type PosdefFix end

"""
```
struct NoPosdef <: PosdefFix end
```

Non positive definite matrices will not be fixed in in [`posdef_fix!`](@ref).
"""
struct NoPosdef <: PosdefFix end

"""
```
@kwdef mutable struct PosdefNearest <: PosdefFix
    method::NearestCorrelationMatrix.NCMAlgorithm = NearestCorrelationMatrix.Newton(;
                                                                                    tau = 1e-12)
end
```

Defines which method from [`NearestCorrelationMatrix`](https://github.com/adknudson/NearestCorrelationMatrix.jl) to use in [`posdef_fix!`](@ref).
"""
mutable struct PosdefNearest <: PosdefFix
    method::NearestCorrelationMatrix.NCMAlgorithm
end
function PosdefNearest(;
                       method::NearestCorrelationMatrix.NCMAlgorithm = NearestCorrelationMatrix.Newton(;
                                                                                                       tau = 1e-12))
    return PosdefNearest(method)
end

# ## Matrix denoising

"""
```
abstract type Denoise end
```

Abstract type for subtyping denoising methods.
"""
abstract type Denoise end

"""
```
struct NoDenoise <: Denoise end
```

No denoising is performed in [`denoise!`](@ref).
"""
struct NoDenoise <: Denoise end

"""
```
@kwdef mutable struct DenoiseFixed{T1, T2, T3, T4} <: Denoise
    detone::Bool = false
    mkt_comp::Integer = 1
    kernel = AverageShiftedHistograms.Kernels.gaussian
    m::Integer = 10
    n::Integer = 1000
    args::Tuple = ()
    kwargs::NamedTuple = (;)
end
```

Defines the parameters for using the fixed method in [`denoise!`](@ref) [MLAM; Chapter 2](@cite).

# Parameters

  - `detone`:

      + `true`: remove the largest `mkt_comp` eigenvalues from the correlation matrix.

!!! warning

    Removing eigenvalues from the matrix may make it singular.

  - `mkt_comp`: the number of largest eigenvalues to remove from the correlation matrix.
  - `kernel`: kernel for fitting the average shifted histograms from [`AverageShiftedHistograms.jl` Kernel Functions](https://joshday.github.io/AverageShiftedHistograms.jl/latest/kernels/).
  - `m`: number of adjacent histograms to smooth over [`AverageShiftedHistograms.ash`](https://joshday.github.io/AverageShiftedHistograms.jl/latest/#Usage).
  - `n`: number of points used when creating the range of values to which the average shifted histogram is to be fitted [`AverageShiftedHistograms.ash`](https://joshday.github.io/AverageShiftedHistograms.jl/latest/#Usage).
  - `args`: arguments for [`Optim.optimize`](https://julianlsolvers.github.io/Optim.jl/stable/user/config/)
  - `kwargs`: keyword arguments for [`Optim.optimize`](https://julianlsolvers.github.io/Optim.jl/stable/user/config/)
"""
mutable struct DenoiseFixed{T1, T2, T3, T4} <: Denoise
    detone::Bool
    mkt_comp::T1
    kernel::T2
    m::T3
    n::T4
    args::Tuple
    kwargs::NamedTuple
end
function DenoiseFixed(; detone::Bool = false, mkt_comp::Integer = 1,
                      kernel = AverageShiftedHistograms.Kernels.gaussian, m::Integer = 10,
                      n::Integer = 1000, args::Tuple = (), kwargs::NamedTuple = (;))
    return DenoiseFixed{typeof(mkt_comp), typeof(kernel), typeof(m), typeof(n)}(detone,
                                                                                mkt_comp,
                                                                                kernel, m,
                                                                                n, args,
                                                                                kwargs)
end

"""
```
@kwdef mutable struct DenoiseSpectral{T1, T2, T3, T4} <: Denoise
    detone::Bool = false
    mkt_comp::Integer = 1
    kernel = AverageShiftedHistograms.Kernels.gaussian
    m::Integer = 10
    n::Integer = 1000
    args::Tuple = ()
    kwargs::NamedTuple = (;)
end
```

Defines the parameters for using the spectral method in [`denoise!`](@ref) [MLAM; Chapter 2](@cite).

# Parameters

  - `detone`:

      + `true`: take only the largest `mkt_comp` eigenvalues from the correlation matrix.

!!! warning

    Removing eigenvalues from the matrix may make it singular.

  - `mkt_comp`: the number of largest eigenvalues to keep from the correlation matrix.
  - `kernel`: kernel for fitting the average shifted histograms from [`AverageShiftedHistograms.jl` Kernel Functions](https://joshday.github.io/AverageShiftedHistograms.jl/latest/kernels/).
  - `m`: number of adjacent histograms to smooth over [`AverageShiftedHistograms.ash`](https://joshday.github.io/AverageShiftedHistograms.jl/latest/#Usage).
  - `n`: number of points used when creating the range of values to which the average shifted histogram is to be fitted [`AverageShiftedHistograms.ash`](https://joshday.github.io/AverageShiftedHistograms.jl/latest/#Usage).
  - `args`: arguments for [`Optim.optimize`](https://julianlsolvers.github.io/Optim.jl/stable/user/config/)
  - `kwargs`: keyword arguments for [`Optim.optimize`](https://julianlsolvers.github.io/Optim.jl/stable/user/config/)
"""
mutable struct DenoiseSpectral{T1, T2, T3, T4} <: Denoise
    detone::Bool
    mkt_comp::T1
    kernel::T2
    m::T3
    n::T4
    args::Tuple
    kwargs::NamedTuple
end
function DenoiseSpectral(; detone::Bool = false, mkt_comp::Integer = 1,
                         kernel = AverageShiftedHistograms.Kernels.gaussian,
                         m::Integer = 10, n::Integer = 1000, args::Tuple = (),
                         kwargs::NamedTuple = (;))
    return DenoiseSpectral{typeof(mkt_comp), typeof(kernel), typeof(m), typeof(n)}(detone,
                                                                                   mkt_comp,
                                                                                   kernel,
                                                                                   m, n,
                                                                                   args,
                                                                                   kwargs)
end

"""
```
@kwdef mutable struct DenoiseShrink{T1, T2, T3, T4, T5} <: Denoise
    alpha::Real = 0.0
    detone::Bool = false
    mkt_comp::Integer = 1
    kernel = AverageShiftedHistograms.Kernels.gaussian
    m::Integer = 10
    n::Integer = 1000
    args::Tuple = ()
    kwargs::NamedTuple = (;)
end
```

Defines the parameters for using the shrink method in [`denoise!`](@ref) [MLAM; Chapter 2](@cite).

# Parameters

  - `alpha`: tuning parameter for how much the matrix should be shrunk, `alpha ∈ [0, 1]`.

  - `detone`:

      + `true`: take only the largest `mkt_comp` eigenvalues from the correlation matrix.

!!! warning

    Removing eigenvalues from the matrix may make it singular.

  - `mkt_comp`: the number of largest eigenvalues to keep from the correlation matrix.
  - `kernel`: kernel for fitting the average shifted histograms from [`AverageShiftedHistograms.jl` Kernel Functions](https://joshday.github.io/AverageShiftedHistograms.jl/latest/kernels/).
  - `m`: number of adjacent histograms to smooth over [`AverageShiftedHistograms.ash`](https://joshday.github.io/AverageShiftedHistograms.jl/latest/#Usage).
  - `n`: number of points used when creating the range of values to which the average shifted histogram is to be fitted [`AverageShiftedHistograms.ash`](https://joshday.github.io/AverageShiftedHistograms.jl/latest/#Usage).
  - `args`: arguments for [`Optim.optimize`](https://julianlsolvers.github.io/Optim.jl/stable/user/config/)
  - `kwargs`: keyword arguments for [`Optim.optimize`](https://julianlsolvers.github.io/Optim.jl/stable/user/config/)
"""
mutable struct DenoiseShrink{T1, T2, T3, T4, T5} <: Denoise
    alpha::T1
    detone::Bool
    mkt_comp::T2
    kernel::T3
    m::T4
    n::T5
    args::Tuple
    kwargs::NamedTuple
end
function DenoiseShrink(; alpha::Real = 0.0, detone::Bool = false, mkt_comp::Integer = 1,
                       kernel = AverageShiftedHistograms.Kernels.gaussian, m::Integer = 10,
                       n::Integer = 1000, args::Tuple = (), kwargs::NamedTuple = (;))
    @smart_assert(zero(alpha) <= alpha <= one(alpha))
    return DenoiseShrink{typeof(alpha), typeof(mkt_comp), typeof(kernel), typeof(m),
                         typeof(n)}(alpha, detone, mkt_comp, kernel, m, n, args, kwargs)
end

# ## Distances

"""
```
abstract type DistanceMethod end
```

Abstract type for subtyping methods for computing distance matrices from correlation ones.
"""
abstract type DistanceMethod end

"""
```
@kwdef mutable struct DistanceMLP <: DistanceMethod
    absolute::Bool = false
end
```

Defines the distance matrix from a correlation matrix [HRP1](@cite) in [`dist`](@ref).

```math
D_{i,\\,j} = 
    \\begin{cases}
        \\sqrt{\\dfrac{1}{2} \\left(\\mathbf{1} - C_{i,\\,j}\\right)} &\\quad \\mathrm{if~ absolute = false}\\\\
        \\sqrt{1 - \\lvert C_{i,\\,j} \\rvert} &\\quad \\mathrm{if~ absolute = true}\\nonumber\\,.
    \\end{cases}
```

Where:

  - ``D_{i,\\,j}``: is the ``(i,\\,j)``-th entry of the `N×N` distance matrix ``\\mathbf{C}``.
  - ``C_{i,\\,j}``: is the ``(i,\\,j)``-th entry of the `N×N` correlation matrix ``\\mathbf{D}``.
  - absolute: flag for stating whether or not an absolute correlation is being used.

# Parameters

  - `absolute`: flag for stating whether or not an absolute correlation is being used.
"""
mutable struct DistanceMLP <: DistanceMethod
    absolute::Bool
end
function DistanceMLP(; absolute::Bool = false)
    return DistanceMLP(absolute)
end

"""
```
@kwdef mutable struct DistanceSqMLP <: DistanceMethod
    absolute::Bool = false
    distance::Distances.UnionMetric = Distances.Euclidean()
    args::Tuple = ()
    kwargs::NamedTuple = (;)
end
```

Defines the distance of distances matrix from a correlation matrix [HRP1](@cite) in [`dist`](@ref).

```math
\\tilde{D}_{i,\\,j} = f_{m}\\left(\\bm{D}_{i},\\, \\bm{D}_j\\right)\\,.
```

Where:

  - ``\\bm{D}_{i}``: is the ``i``-th column/row of the `N×N` distance matrix defined in [`DistanceMLP`](@ref).
  - ``f_{m}``: is the pairwise distance function for metric ``m``. We use the [`Distances.pairwise`](https://github.com/JuliaStats/Distances.jl?tab=readme-ov-file#computing-pairwise-distances) function which computes the entire matrix at once, the output is a vector so it gets reshaped into an `N×N` matrix.
  - ``\\tilde{D}_{i,\\,j}``: is the ``(i,\\,j)``-th entry of the `N×N` distances of distances matrix.
  - absolute: is a flag whether the correlation is absolute.

# Parameters

  - `absolute`: flag for stating whether or not an absolute correlation is being used.
  - `distance`: distance metric from [`Distances.jl`](https://github.com/JuliaStats/Distances.jl).
  - `args`: args for the [`Distances.pairwise`](https://github.com/JuliaStats/Distances.jl?tab=readme-ov-file#computing-pairwise-distances) function.
  - `kwargs`: key word args for the [`Distances.pairwise`](https://github.com/JuliaStats/Distances.jl?tab=readme-ov-file#computing-pairwise-distances) function.
"""
mutable struct DistanceSqMLP <: DistanceMethod
    absolute::Bool
    distance::Distances.UnionMetric
    args::Tuple
    kwargs::NamedTuple
end
function DistanceSqMLP(; absolute::Bool = false,
                       distance::Distances.UnionMetric = Distances.Euclidean(),
                       args::Tuple = (), kwargs::NamedTuple = (;))
    return DistanceSqMLP(absolute, distance, args, kwargs)
end

"""
```
struct DistanceLog <: DistanceMethod end
```

Defines the log-distance matrix from the correlation matrix.

```math
D_{i,\\,j} = -\\log\\left(C_{i,\\,j}\\right)\\,.
```

Where:

  - ``D_{i,\\,j}``: is the ``(i,\\,j)``-th entry of the `N×N` log-distance matrix.
  - ``C_{i,\\,j}``: is the  ``(i,\\,j)``-th entry of an absolute correlation matrix.
"""
struct DistanceLog <: DistanceMethod end

"""
```
struct DistanceCanonical <: DistanceMethod end
```

Struct for computing the canonical distance for a given correlation subtype of [`PortfolioOptimiserCovCor`](@ref).
"""
struct DistanceCanonical <: DistanceMethod end

"""
```
abstract type AbstractBins end
```

Abstract type for defining bin width estimation functions when computing [`DistanceVarInfo`](@ref) and [`CorMutualInfo`](@ref) distance and correlation matrices respectively.
"""
abstract type AbstractBins end

"""
```
abstract type AstroBins <: AbstractBins end
```

Abstract type for defining which bin width function to use from [`astropy`](https://docs.astropy.org/en/stable/visualization/histogram.html).
"""
abstract type AstroBins <: AbstractBins end

"""
```
struct Knuth <: AstroBins end
```

Knuth's bin width algorithm from [`astropy`](https://docs.astropy.org/en/stable/api/astropy.stats.knuth_bin_width.html#astropy.stats.knuth_bin_width).
"""
struct Knuth <: AstroBins end

"""
```
struct Freedman <: AstroBins end
```

Freedman's bin width algorithm from [`astropy`](https://docs.astropy.org/en/stable/api/astropy.stats.freedman_bin_width.html#astropy.stats.freedman_bin_width).
"""
struct Freedman <: AstroBins end

"""
```
struct Scott <: AstroBins end
```

Scott's bin width algorithm from [`astropy`](https://docs.astropy.org/en/stable/api/astropy.stats.scott_bin_width.html#astropy.stats.scott_bin_width).
"""
struct Scott <: AstroBins end

"""
```
struct HGR <: AbstractBins end
```

Hacine-Gharbi and Ravier's bin width algorithm [HGR](@cite).
"""
struct HGR <: AbstractBins end

"""
```
@kwdef mutable struct DistanceVarInfo <: DistanceMethod
    bins::Union{<:Integer, <:AbstractBins} = HGR()
    normalise::Bool = true
end
```

Defines the variation of information distance matrix.

# Parameters

  - `bins`: defines the bin function, or bin width directly and if so `bins > 0`.
  - `normalise`: whether or not to normalise the variation of information.
"""
mutable struct DistanceVarInfo <: DistanceMethod
    bins::Union{<:Integer, <:AbstractBins}
    normalise::Bool
end
function DistanceVarInfo(; bins::Union{<:Integer, <:AbstractBins} = HGR(),
                         normalise::Bool = true)
    if isa(bins, Integer)
        @smart_assert(bins > zero(bins))
    end
    return DistanceVarInfo(bins, normalise)
end

# ## Clustering

"""
```
abstract type HClustAlg end
```

Abstract type for subtyping hierarchical clustering methods.
"""
abstract type HClustAlg end

"""
```
@kwdef mutable struct HAC <: HClustAlg
    linkage::Symbol = :ward
end
```

Use a hierarchical clustering algorithm from [`Clustering.jl`](https://github.com/JuliaStats/Clustering.jl).

# Parameters

  - `linkage`: linkage type supported by [`hclust`](https://juliastats.org/Clustering.jl/stable/hclust.html#Clustering.hclust).
"""
mutable struct HAC <: HClustAlg
    linkage::Symbol
end
function HAC(; linkage::Symbol = :ward)
    return HAC(linkage)
end

"""
```
abstract type DBHTSimilarity end
```

Abstract type for subtyping methods for defining functions for computing similarity matrices used in DBHT clustering [`PMFG_T2s`](@ref) [DBHTs, PMFG](@cite).
"""
abstract type DBHTSimilarity end

"""
```
struct DBHTExp <: DBHTSimilarity end
```

Defines the similarity matrix for use in [`PMFG_T2s`](@ref) as the element-wise exponential decay of the dissimilarity matrix in [`dbht_similarity`](@ref).
"""
struct DBHTExp <: DBHTSimilarity end

"""
```
struct DBHTMaxDist <: DBHTSimilarity end
```

Defines the similarity matrix for use in [`PMFG_T2s`](@ref) as the element-wise distance from the maximum value of the dissimilarity matrix [`dbht_similarity`](@ref).
"""
struct DBHTMaxDist <: DBHTSimilarity end

"""
```
abstract type DBHTRootMethod end
```

Abstract type for subtyping methods creating roots of cliques in [`CliqueRoot`](@ref) [NHPG](@cite).
"""
abstract type DBHTRootMethod end

"""
```
struct UniqueDBHT <: DBHTRootMethod end
```

Create a unique root for a clique in [`CliqueRoot`](@ref) [NHPG](@cite).
"""
struct UniqueDBHT <: DBHTRootMethod end

"""
```
struct EqualDBHT <: DBHTRootMethod end
```

Create a clique's root from its adjacency tree in [`CliqueRoot`](@ref).
"""
struct EqualDBHT <: DBHTRootMethod end

"""
```
mutable struct DBHT <: HClustAlg
    distance::DistanceMethod
    similarity::DBHTSimilarity
    root_method::DBHTRootMethod
end
```

Defines the parameters for computing [`DBHTs`](@ref) [DBHTs](@cite).
"""
mutable struct DBHT <: HClustAlg
    distance::DistanceMethod
    similarity::DBHTSimilarity
    root_method::DBHTRootMethod
end
function DBHT(; distance::DistanceMethod = DistanceMLP(),
              similarity::DBHTSimilarity = DBHTMaxDist(),
              root_method::DBHTRootMethod = UniqueDBHT())
    return DBHT(distance, similarity, root_method)
end

"""
```
abstract type NumClusterMethod end
```

Abstract type for subtyping methods for determining the number of clusters in a [`Clustering.Hclust`](https://juliastats.org/Clustering.jl/stable/hclust.html#Clustering.Hclust) when calling [`calc_k_clusters`](@ref).
"""
abstract type NumClusterMethod end

"""
```
struct TwoDiff <: NumClusterMethod end
```

Use the two difference gap statistic for computing the number of clusters in [`calc_k_clusters`](@ref).
"""
struct TwoDiff <: NumClusterMethod end

"""
```
@kwdef mutable struct StdSilhouette <: NumClusterMethod
    metric::Union{Distances.SemiMetric, Nothing} = nothing
end
```

Use the standardised silhouette score for computing the number of clusters in [`calc_k_clusters`](@ref).
"""
mutable struct StdSilhouette <: NumClusterMethod
    metric::Union{Distances.SemiMetric, Nothing}
end
function StdSilhouette(; metric::Union{Distances.SemiMetric, Nothing} = nothing)
    return StdSilhouette(metric)
end

"""
```
@kwdef mutable struct HCOpt{T1 <: Integer, T2 <: Integer}
    branchorder::Symbol = :optimal
    k_method::NumClusterMethod = TwoDiff()
    k::T1 = 0
    max_k::T2 = 0
end
```

Defines the options for processing clustering results in an instance of [`Clustering.Hclust`](https://juliastats.org/Clustering.jl/stable/hclust.html#Clustering.Hclust).

# Parameters

  - `branchorder`: parameter for ordering a dendrogram's branches accepted by [`Clustering.jl`](https://github.com/JuliaStats/Clustering.jl).
  - `k_method`: method subtyping [`NumClusterMethod`](@ref) for computing the number of clusters.
  - `k`: directly provide the number of clusters, if `0` use `k_method` for computing the number of clusters.
  - `max_k`: maximum number of clusters, if `0` defaults to `⌈sqrt(N)⌉`, where `N` is the number of assets.
"""
mutable struct HCOpt{T1 <: Integer, T2 <: Integer}
    branchorder::Symbol
    k_method::NumClusterMethod
    k::T1
    max_k::T2
end
function HCOpt(; branchorder::Symbol = :optimal, k_method::NumClusterMethod = TwoDiff(),
               k::Integer = 0, max_k::Integer = 0)
    return HCOpt{typeof(k), typeof(max_k)}(branchorder, k_method, k, max_k)
end

"""
```
struct ClusterNode{tid, tl, tr, td, tcnt}
    id::tid
    left::tl
    right::tr
    height::td
    level::tcnt

    function ClusterNode(id, left::Union{ClusterNode, Nothing} = nothing,
                         right::Union{ClusterNode, Nothing} = nothing, height::Real = 0.0,
                         level::Int = 1)
        ilevel = isnothing(left) ? level : (left.level + right.level)

        return new{typeof(id), typeof(left), typeof(right), typeof(height), typeof(level)}(id,
                                                                                         left,
                                                                                         right,
                                                                                         height,
                                                                                         ilevel)
    end
end
```

Structure for definining a cluster node. This is used for turning a clustering result into a tree, [`is_leaf`](@ref), [`pre_order`](@ref), [`to_tree`](@ref).

# Parameters

  - `id`: node ID.
  - `left`: node to the left.
  - `right`: node to the right.
  - `height`: node height.
  - `level`: node level.
"""
struct ClusterNode{tid, tl, tr, td, tcnt}
    id::tid
    left::tl
    right::tr
    height::td
    level::tcnt

    function ClusterNode(id, left::Union{ClusterNode, Nothing} = nothing,
                         right::Union{ClusterNode, Nothing} = nothing, height::Real = 0.0,
                         level::Int = 1)
        ilevel = isnothing(left) ? level : (left.level + right.level)

        return new{typeof(id), typeof(left), typeof(right), typeof(height), typeof(level)}(id,
                                                                                           left,
                                                                                           right,
                                                                                           height,
                                                                                           ilevel)
    end
end

# ## Covariance, correlation, kurt and skew

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

  - `absolute`: whether or not to compute an absolute correlation.
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

  - `absolute`: whether or not to compute an absolute correlation, `abs.(cor(X))`.

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

  - `absolute`: whether or not to compute an absolute correlation, `abs.(corspearman(X))`.
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

  - `absolute`: whether or not to compute an absolute correlation, `abs.(corkendall(X))`.
"""
mutable struct CorKendall <: CorRank
    absolute::Bool
end
function CorKendall(; absolute::Bool = false)
    return CorKendall(absolute)
end

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

abstract type CorGerber <: PortfolioOptimiserCovCor end
abstract type CorGerberBasic <: CorGerber end
abstract type CorSB <: CorGerber end
abstract type CorGerberSB <: CorGerber end

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

abstract type AbstractLoGo end
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

@kwdef mutable struct KurtFull <: KurtEstimator
    posdef::PosdefFix = PosdefNearest(;)
    denoise::Denoise = NoDenoise(;)
    logo::AbstractLoGo = NoLoGo(;)
end

@kwdef mutable struct KurtSemi <: KurtEstimator
    target::Union{<:Real, AbstractVector{<:Real}} = 0.0
    posdef::PosdefFix = PosdefNearest(;)
    denoise::Denoise = NoDenoise(;)
    logo::AbstractLoGo = NoLoGo(;)
end

abstract type SkewEstimator end

struct SkewFull <: SkewEstimator end

@kwdef mutable struct SkewSemi <: SkewEstimator
    target::Union{<:Real, AbstractVector{<:Real}} = 0.0
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

# ## Mean estimator

"""
```
abstract type MeanEstimator end
```
"""
abstract type MeanEstimator end

"""
```
abstract type MeanTarget end
```
"""
abstract type MeanTarget end

"""
```
struct GM <: MeanTarget end
```
"""
struct GM <: MeanTarget end

"""
```
struct VW <: MeanTarget end
```
"""
struct VW <: MeanTarget end

"""
```
struct SE <: MeanTarget end
```
"""
struct SE <: MeanTarget end

"""
```
@kwdef mutable struct MuSimple <: MeanEstimator
    w::Union{<:AbstractWeights, Nothing} = nothing
end
```
"""
mutable struct MuSimple <: MeanEstimator
    w::Union{<:AbstractWeights, Nothing}
end
function MuSimple(; w::Union{<:AbstractWeights, Nothing} = nothing)
    return MuSimple(w)
end

"""
```
@kwdef mutable struct MuJS{T1} <: MeanEstimator
    target::MeanTarget = GM()
    w::Union{<:AbstractWeights, Nothing} = nothing
    sigma::T1 = Matrix{Float64}(undef, 0, 0)
end
```
"""
mutable struct MuJS{T1} <: MeanEstimator
    target::MeanTarget
    w::Union{<:AbstractWeights, Nothing}
    sigma::T1
end
function MuJS(; target::MeanTarget = GM(), w::Union{<:AbstractWeights, Nothing} = nothing,
              sigma::AbstractMatrix = Matrix{Float64}(undef, 0, 0))
    return MuJS{typeof(sigma)}(target, w, sigma)
end

"""
```
@kwdef mutable struct MuBS{T1} <: MeanEstimator
    target::MeanTarget = GM()
    w::Union{<:AbstractWeights, Nothing} = nothing
    sigma::T1 = Matrix{Float64}(undef, 0, 0)
end
```
"""
mutable struct MuBS{T1} <: MeanEstimator
    target::MeanTarget
    w::Union{<:AbstractWeights, Nothing}
    sigma::T1
end
function MuBS(; target::MeanTarget = GM(), w::Union{<:AbstractWeights, Nothing} = nothing,
              sigma::AbstractMatrix = Matrix{Float64}(undef, 0, 0))
    return MuBS{typeof(sigma)}(target, w, sigma)
end

"""
```
@kwdef mutable struct MuBOP{T1} <: MeanEstimator
    target::MeanTarget = GM()
    w::Union{<:AbstractWeights, Nothing} = nothing
    sigma::T1 = Matrix{Float64}(undef, 0, 0)
end
```
"""
mutable struct MuBOP{T1} <: MeanEstimator
    target::MeanTarget
    w::Union{<:AbstractWeights, Nothing}
    sigma::T1
end
function MuBOP(; target::MeanTarget = GM(), w::Union{<:AbstractWeights, Nothing} = nothing,
               sigma::AbstractMatrix = Matrix{Float64}(undef, 0, 0))
    return MuBOP{typeof(sigma)}(target, w, sigma)
end

# ## Worst case statistics

abstract type WorstCaseSet end
struct Box <: WorstCaseSet end
struct Ellipse <: WorstCaseSet end
@kwdef mutable struct NoWC <: WorstCaseSet
    formulation::SDSquaredFormulation = SOCSD()
end

abstract type WorstCaseMethod end
abstract type WorstCaseArchMethod <: WorstCaseMethod end
struct StationaryBS <: WorstCaseArchMethod end
struct CircularBS <: WorstCaseArchMethod end
struct MovingBS <: WorstCaseArchMethod end

@kwdef mutable struct ArchWC <: WorstCaseMethod
    bootstrap::WorstCaseArchMethod = StationaryBS()
    n_sim::Integer = 3_000
    block_size::Integer = 3
    q::Real = 0.05
    seed::Union{<:Integer, Nothing} = nothing
end

@kwdef mutable struct NormalWC <: WorstCaseMethod
    n_sim::Integer = 3_000
    q::Real = 0.05
    rng::AbstractRNG = Random.default_rng()
    seed::Union{<:Integer, Nothing} = nothing
end

@kwdef mutable struct DeltaWC <: WorstCaseMethod
    dcov::Real = 0.1
    dmu::Real = 0.1
end

abstract type WorstCaseKMethod end
struct KNormalWC <: WorstCaseKMethod end
struct KGeneralWC <: WorstCaseKMethod end

@kwdef mutable struct WCType
    cov_type::PortfolioOptimiserCovCor = PortCovCor(;)
    mu_type::MeanEstimator = MuSimple(;)
    box::WorstCaseMethod = NormalWC(;)
    ellipse::WorstCaseMethod = NormalWC(;)
    k_sigma::Union{<:Real, WorstCaseKMethod} = KNormalWC(;)
    k_mu::Union{<:Real, WorstCaseKMethod} = KNormalWC(;)
    posdef::PosdefFix = PosdefNearest(;)
    diagonal::Bool = false
end

# ## Regression statistics

abstract type RegressionType end
abstract type StepwiseRegression <: RegressionType end
abstract type RegressionCriteria end
abstract type MinValRegressionCriteria <: RegressionCriteria end
abstract type MaxValRegressionCriteria <: RegressionCriteria end
struct AIC <: MinValRegressionCriteria end
struct AICC <: MinValRegressionCriteria end
struct BIC <: MinValRegressionCriteria end
struct RSq <: MaxValRegressionCriteria end
struct AdjRSq <: MaxValRegressionCriteria end

mutable struct PVal{T1 <: Real} <: RegressionCriteria
    threshold::T1
end
function PVal(; threshold::Real = 0.05)
    @smart_assert(zero(threshold) < threshold < one(threshold))
    return PVal{typeof(threshold)}(threshold)
end

abstract type DimensionReductionTarget end
@kwdef mutable struct PCATarget <: DimensionReductionTarget
    kwargs::NamedTuple = (;)
end

@kwdef mutable struct FReg <: StepwiseRegression
    criterion::RegressionCriteria = PVal(;)
end

@kwdef mutable struct BReg <: StepwiseRegression
    criterion::RegressionCriteria = PVal(;)
end

@kwdef mutable struct DRR <: RegressionType
    ve::StatsBase.CovarianceEstimator = SimpleVariance(;)
    std_w::Union{<:AbstractWeights, Nothing} = nothing
    mean_w::Union{<:AbstractWeights, Nothing} = nothing
    pcr::DimensionReductionTarget = PCATarget(;)
end

@kwdef mutable struct FactorType
    error::Bool = true
    B::Union{Nothing, DataFrame} = nothing
    method::RegressionType = FReg(;)
    ve::StatsBase.CovarianceEstimator = SimpleVariance(;)
    var_w::Union{<:AbstractWeights, Nothing} = nothing
end

# ## Black Litterman

abstract type BlackLitterman end
abstract type BlackLittermanFactor <: BlackLitterman end

@kwdef mutable struct BLType{T1 <: Real} <: BlackLitterman
    eq::Bool = true
    delta::Union{<:Real, Nothing} = 1.0
    rf::T1 = 0.0
    posdef::PosdefFix = PosdefNearest()
    denoise::Denoise = NoDenoise()
    logo::AbstractLoGo = NoLoGo()
end

@kwdef mutable struct BBLType{T1 <: Real} <: BlackLittermanFactor
    constant::Bool = true
    error::Bool = true
    delta::Union{<:Real, Nothing} = 1.0
    rf::T1 = 0.0
    ve::StatsBase.CovarianceEstimator = SimpleVariance()
    var_w::Union{<:AbstractWeights, Nothing} = nothing
end

@kwdef mutable struct ABLType{T1 <: Real} <: BlackLittermanFactor
    constant::Bool = true
    eq::Bool = true
    delta::Union{<:Real, Nothing} = 1.0
    rf::T1 = 0.0
    posdef::PosdefFix = PosdefNearest()
    denoise::Denoise = NoDenoise()
    logo::AbstractLoGo = NoLoGo()
end

# ## Ordered Weight Array statistics

"""
```
OWAMethods = (:CRRA, :E, :SS, :SD)
```

Methods for computing the weights used to combine L-moments higher than 2 [OWAL](@cite).

  - `:CRRA:` Normalised Constant Relative Risk Aversion Coefficients.
  - `:E`: Maximum Entropy. Solver must support `MOI.RelativeEntropyCone` and `MOI.NormOneCone`.
  - `:SS`: Minimum Sum of Squares. Solver must support `MOI.SecondOrderCone`.
  - `:SD`: Minimum Square Distance. Solver must support `MOI.SecondOrderCone`.
"""
abstract type OWAMethods end

mutable struct CRRA{T1 <: Real} <: OWAMethods
    g::T1
end
function CRRA(; g::Real = 0.5)
    @smart_assert(zero(g) < g < one(g))
    return CRRA{typeof(g)}(g)
end

mutable struct MaxEntropy{T1 <: Real} <: OWAMethods
    max_phi::T1
end
function MaxEntropy(; max_phi::Real = 0.5)
    @smart_assert(zero(max_phi) < max_phi < one(max_phi))
    return MaxEntropy{typeof(max_phi)}(max_phi)
end

mutable struct MinSumSq{T1 <: Real} <: OWAMethods
    max_phi::T1
end
function MinSumSq(; max_phi::Real = 0.5)
    @smart_assert(zero(max_phi) < max_phi < one(max_phi))
    return MinSumSq{typeof(max_phi)}(max_phi)
end

mutable struct MinSqDist{T1 <: Real} <: OWAMethods
    max_phi::T1
end
function MinSqDist(; max_phi::Real = 0.5)
    @smart_assert(zero(max_phi) < max_phi < one(max_phi))
    return MinSqDist{typeof(max_phi)}(max_phi)
end

# ## Network statistics

abstract type CentralityType end

@kwdef mutable struct DegreeCentrality <: CentralityType
    args::Tuple = ()
    kwargs::NamedTuple = (;)
end

abstract type TreeType end
@kwdef mutable struct KruskalTree <: TreeType
    args::Tuple = ()
    kwargs::NamedTuple = (;)
end

abstract type NetworkType end

@kwdef mutable struct TMFG{T1 <: Integer} <: NetworkType
    similarity::DBHTSimilarity = DBHTMaxDist()
    steps::T1 = 1
    centrality::CentralityType = DegreeCentrality()
end

@kwdef mutable struct MST{T1 <: Integer} <: NetworkType
    tree::TreeType = KruskalTree()
    steps::T1 = 1
    centrality::CentralityType = DegreeCentrality()
end

export NoPosdef, PosdefNearest, NoDenoise, DenoiseFixed, DenoiseSpectral, DenoiseShrink,
       DistanceMLP, DistanceSqMLP, DistanceLog, DistanceCanonical, Knuth, Freedman, Scott,
       HGR, DistanceVarInfo, HAC, DBHTExp, DBHTMaxDist, UniqueDBHT, EqualDBHT, DBHT,
       TwoDiff, StdSilhouette, HCOpt, ClusterNode, CovFull, SimpleVariance, CovSemi,
       CorSpearman, CorKendall, CorMutualInfo, CorDistance, CorLTD, CorGerber0, CorGerber1,
       CorGerber2, CorSB0, CorSB1, CorGerberSB0, CorGerberSB1, NoLoGo, LoGo, KurtFull,
       KurtSemi, SkewFull, SkewSemi, PortCovCor, GM, VW, SE, MuSimple, MuJS, MuBS, MuBOP,
       Box, Ellipse, NoWC, StationaryBS, CircularBS, MovingBS, ArchWC, NormalWC, DeltaWC,
       KNormalWC, KGeneralWC, WCType, AIC, AICC, BIC, RSq, AdjRSq, PVal, PCATarget, FReg,
       BReg, DRR, FactorType, BLType, BBLType, ABLType, CRRA, MaxEntropy, MinSumSq,
       MinSqDist, DegreeCentrality, KruskalTree, TMFG, MST
