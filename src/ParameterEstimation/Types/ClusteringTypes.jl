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

Abstract type for subtyping methods for defining functions for computing similarity matrices from used in DBHT clustering [`PMFG_T2s`](@ref) [DBHTs, PMFG](@cite).
"""
abstract type DBHTSimilarity end

"""
```
struct DBHTExp <: DBHTSimilarity end
```

Defines the similarity matrix for use in [`PMFG_T2s`](@ref) as the element-wise exponential decay of the dissimilarity matrix in [`dbht_similarity`](@ref).

```math
\\begin{align}
S_{i,\\,j} = \\exp(-D_{i,\\,j})\\,.
\\end{align}
```

Where:

  - ``S_{i,\\,j}`` is the ``(i,\\,j)``-th entry in the similarity matrix.
  - ``D_{i,\\,j}`` is the ``(i,\\,j)``-th entry in the distance matrix.
"""
struct DBHTExp <: DBHTSimilarity end

"""
```
struct DBHTMaxDist <: DBHTSimilarity end
```

Defines the similarity matrix for use in [`PMFG_T2s`](@ref) as the element-wise squared distance from the maximum value of the dissimilarity matrix [`dbht_similarity`](@ref).

```math
\\begin{align}
S_{i,\\,j} = \\left\\lceil (\\max \\mathbf{D})^2 \\right\\rceil - D_{i,\\,j} ^ 2\\,.
\\end{align}
```

Where:

  - ``S_{i,\\,j}`` is the ``(i,\\,j)``-th entry in the similarity matrix.
  - ``D_{i,\\,j}`` is the ``(i,\\,j)``-th entry in the distance matrix.
  - ``\\mathbf{D}`` is the distance matrix.

```
```
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

# Parameters

  - `distance`: method for computing the distance matrix from correlation ones [`DistanceMethod`](@ref).
  - `similarity`: method for computing the similarity matrix from the correlation and/or distance ones [`DBHTSimilarity`](@ref), [`dbht_similarity`](@ref).
  - `root_method`: method for choosing clique roots [`DBHTRootMethod`](@ref).
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

# Parameters

  - `metric`: metric for computing the [`silhouettes`](https://juliastats.org/Clustering.jl/stable/validate.html#silhouettes_index).
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
  - `k`:

      + if `iszero(k)`: use `k_method` for computing the number of clusters.
      + else: directly provide the number of clusters.
  - `max_k`: maximum number of clusters, capped to `⌈sqrt(N)⌉`.

      + if `0`: defaults to `⌈sqrt(N)⌉`.
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

export HAC, DBHTExp, DBHTMaxDist, UniqueDBHT, EqualDBHT, DBHT, TwoDiff, StdSilhouette,
       HCOpt, ClusterNode
