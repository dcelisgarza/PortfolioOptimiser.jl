"""
```
abstract type ClustAlg end
```

Abstract type for subtyping hierarchical clustering types.
"""
abstract type ClustAlg end

"""
```
@kwdef mutable struct HAC <: ClustAlg
    linkage::Symbol = :ward
end
```

Use a hierarchical clustering algorithm from [`Clustering.jl`](https://github.com/JuliaStats/Clustering.jl).

# Parameters

  - `linkage`: linkage type supported by [`hclust`](https://juliastats.org/Clustering.jl/stable/hclust.html#Clustering.hclust).
"""
mutable struct HAC <: ClustAlg
    linkage::Symbol
end
function HAC(; linkage::Symbol = :ward)
    return HAC(linkage)
end

"""
```
abstract type DBHTSimilarity end
```

Abstract type for subtyping types for defining functions for computing similarity matrices from used in DBHT clustering [`PMFG_T2s`](@ref) [DBHTs, PMFG](@cite).
"""
abstract type DBHTSimilarity end

"""
```
struct DBHTExp <: DBHTSimilarity end
```

Defines the similarity matrix for use in [`PMFG_T2s`](@ref) as the element-wise exponential decay of the dissimilarity matrix in [`dbht_similarity`](@ref).

```math
\\begin{align}
S_{i,\\,j} &= \\exp(-D_{i,\\,j})\\,.
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
S_{i,\\,j} &= \\left\\lceil (\\max \\mathbf{D})^2 \\right\\rceil - D_{i,\\,j} ^ 2\\,.
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
abstract type DBHTRootType end
```

Abstract type for subtyping types creating roots of cliques in [`CliqueRoot`](@ref) [NHPG](@cite).
"""
abstract type DBHTRootType end

"""
```
struct UniqueDBHT <: DBHTRootType end
```

Create a unique root for a clique in [`CliqueRoot`](@ref) [NHPG](@cite).
"""
struct UniqueDBHT <: DBHTRootType end

"""
```
struct EqualDBHT <: DBHTRootType end
```

Create a clique's root from its adjacency tree in [`CliqueRoot`](@ref).
"""
struct EqualDBHT <: DBHTRootType end

"""
```
mutable struct DBHT <: ClustAlg
    distance::DistType
    similarity::DBHTSimilarity
    root_type::DBHTRootType
end
```

Defines the parameters for computing [`DBHTs`](@ref) [DBHTs](@cite).

# Parameters

  - `distance`: type for computing the distance matrix from correlation ones [`DistType`](@ref).
  - `similarity`: type for computing the similarity matrix from the correlation and/or distance ones [`DBHTSimilarity`](@ref), [`dbht_similarity`](@ref).
  - `root_type`: type for choosing clique roots [`DBHTRootType`](@ref).
"""
mutable struct DBHT <: ClustAlg
    distance::DistType
    similarity::DBHTSimilarity
    root_type::DBHTRootType
end
function DBHT(; distance::DistType = DistMLP(), similarity::DBHTSimilarity = DBHTMaxDist(),
              root_type::DBHTRootType = UniqueDBHT())
    return DBHT(distance, similarity, root_type)
end

"""
```
abstract type NumClusterType end
```

Abstract type for subtyping types for determining the number of clusters in a [`Clustering.Hclust`](https://juliastats.org/Clustering.jl/stable/hclust.html#Clustering.Hclust) when calling [`calc_k_clusters`](@ref).
"""
abstract type NumClusterType end

"""
```
struct TwoDiff <: NumClusterType end
```

Use the two difference gap statistic for computing the number of clusters in [`calc_k_clusters`](@ref).
"""
struct TwoDiff <: NumClusterType end

"""
```
@kwdef mutable struct StdSilhouette <: NumClusterType
    metric::Union{Distances.SemiMetric, Nothing} = nothing
end
```

Use the standardised silhouette score for computing the number of clusters in [`calc_k_clusters`](@ref).

# Parameters

  - `metric`: metric for computing the [`silhouettes`](https://juliastats.org/Clustering.jl/stable/validate.html#silhouettes_index).
"""
mutable struct StdSilhouette <: NumClusterType
    metric::Union{Distances.SemiMetric, Nothing}
end
function StdSilhouette(; metric::Union{Distances.SemiMetric, Nothing} = nothing)
    return StdSilhouette(metric)
end

"""
```
@kwdef mutable struct ClustOpt{T1 <: Integer, T2 <: Integer}
    branchorder::Symbol = :optimal
    k_type::NumClusterType = TwoDiff()
    k::T1 = 0
    max_k::T2 = 0
end
```

Defines the options for processing clustering results in an instance of [`Clustering.Hclust`](https://juliastats.org/Clustering.jl/stable/hclust.html#Clustering.Hclust).

# Parameters

  - `branchorder`: parameter for ordering a dendrogram's branches accepted by [`Clustering.jl`](https://github.com/JuliaStats/Clustering.jl).

  - `k_type`: type subtyping [`NumClusterType`](@ref) for computing the number of clusters.
  - `k`:

      + if `iszero(k)`: use `k_type` for computing the number of clusters.
      + else: directly provide the number of clusters.
  - `max_k`: maximum number of clusters, capped to `⌈sqrt(N)⌉`.

      + if `0`: defaults to `⌈sqrt(N)⌉`.
"""
mutable struct ClustOpt{T1 <: Integer, T2 <: Integer}
    branchorder::Symbol
    k_type::NumClusterType
    k::T1
    max_k::T2
end
function ClustOpt(; branchorder::Symbol = :optimal, k_type::NumClusterType = TwoDiff(),
                  k::Integer = 0, max_k::Integer = 0)
    return ClustOpt{typeof(k), typeof(max_k)}(branchorder, k_type, k, max_k)
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
       ClustOpt, ClusterNode
