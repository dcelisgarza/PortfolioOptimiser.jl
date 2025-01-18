"""
```
_clusterise(ca::HAC, port::Union{HCPortfolio, Portfolio}, clust_opt::ClustOpt = ClustOpt())
```

Use [`Clustering.hclust`](https://juliastats.org/Clustering.jl/stable/hclust.html#Clustering.hclust) to hierarchically cluster the assets in a hierarchical portfolio using the covariance and distance matrices stored in the portfolio. See the arguments types' docs for details.

# Inputs

  - `ca`: linkage for [`Clustering.hclust`](https://juliastats.org/Clustering.jl/stable/hclust.html#Clustering.hclust).
  - `port`: hierarchical clustering portfolio .
  - `clust_opt`: options for determining the number of clusters [`ClustOpt`](@ref).

# Outputs

  - `clustering`: [`Clustering.Hclust`](https://juliastats.org/Clustering.jl/stable/hclust.html#Clustering.Hclust) of the portfolio assets.
  - `k`: optimum number of clusters.
"""
function _clusterise(ca::HAC, port::Portfolio, clust_opt::ClustOpt = ClustOpt())
    clustering = hclust(port.dist; linkage = ca.linkage,
                        branchorder = clust_opt.branchorder)
    k = calc_k_clusters(clust_opt, port.dist, clustering)
    return clustering, k
end

"""
```
_clusterise(ca::DBHT, port::Union{HCPortfolio, Portfolio}, clust_opt::ClustOpt = ClustOpt())
```

Use [`DBHTs`](@ref) to hierarchically cluster the assets in a hierarchical portfolio using the covariance and distance matrices stored in the portfolio. See the arguments types' docs for details.

# Inputs

  - `ca`: [`DBHT`] options for clustering with [`DBHTs`](@ref).
  - `port`: hierarchical clustering portfolio .
  - `clust_opt`: options for determining the number of clusters [`ClustOpt`](@ref).

# Outputs

  - `clustering`: [`Clustering.Hclust`](https://juliastats.org/Clustering.jl/stable/hclust.html#Clustering.Hclust) of the portfolio assets.
  - `k`: optimum number of clusters.
"""
function _clusterise(ca::DBHT, port::Portfolio, clust_opt::ClustOpt = ClustOpt())
    S = port.cor
    D = port.dist
    S = dbht_similarity(ca.similarity, S, D)
    clustering = DBHTs(D, S; branchorder = clust_opt.branchorder, type = ca.root_type)[end]
    k = calc_k_clusters(clust_opt, D, clustering)
    return clustering, k
end

"""
```
cluster_assets(port::Union{HCPortfolio, Portfolio}; clust_alg::ClustAlg = HAC(),
               clust_opt::ClustOpt = ClustOpt())
```

Hierarchically cluster the assets in a hierarchical portfolio using the covariance and distance matrices stored in the portfolio. See the arguments types' docs for details.

# Inputs

  - `port`: hierarchical clustering portfolio .
  - `clust_alg`: hierarchical clustering algorithm [`ClustAlg`](@ref).
  - `clust_opt`: options for determining the number of clusters [`ClustOpt`](@ref).

# Outputs

  - `idx`: clustering assignments after cutting the tree into `k` levels [`Clustering.cutree`](https://juliastats.org/Clustering.jl/stable/hclust.html#Clustering.cutree).
  - `clustering`: [`Clustering.Hclust`](https://juliastats.org/Clustering.jl/stable/hclust.html#Clustering.Hclust) of the portfolio assets.
  - `k`: optimum number of clusters.
"""
function cluster_assets(port::Portfolio; clust_alg::ClustAlg = HAC(),
                        clust_opt::ClustOpt = ClustOpt())
    clustering, k = _clusterise(clust_alg, port, clust_opt)
    idx = cutree(clustering; k = k)
    return idx, clustering, k
end

"""
```
cluster_assets!(port::Union{HCPortfolio, Portfolio}; clust_alg::ClustAlg = HAC(),
                clust_opt::ClustOpt = ClustOpt())
```

Hierarchically cluster the assets in a hierarchical portfolio using the covariance and distance matrices stored in the portfolio. Save the results in the portfolio. See the arguments types' docs for details.

# Inputs

  - `port`: hierarchical clustering portfolio .
  - `clust_alg`: hierarchical clustering algorithm [`ClustAlg`](@ref).
  - `clust_opt`: options for determining the number of clusters [`ClustOpt`](@ref).
"""
function cluster_assets!(port::Portfolio; clust_alg::ClustAlg = HAC(),
                         clust_opt::ClustOpt = ClustOpt())
    clustering, k = _clusterise(clust_alg, port, clust_opt)
    port.clusters = clustering
    port.k = k
    return nothing
end

#=
"""
```
cluster_assets(port::Portfolio; cor_type::PortfolioOptimiserCovCor = PortCovCor(),
               dist_type::DistType = DistCanonical(),
               clust_alg::ClustAlg = HAC(), clust_opt::ClustOpt = ClustOpt())
```

Hierarchically cluster the assets in a hierarchical portfolio using the covariance and distance matrices stored in the portfolio. See the arguments types' docs for details.

# Inputs

  - `cor_type`: correlation matrix estimator [`PortfolioOptimiserCovCor`](@ref).
  - `dist_type`: type for computing the distance matrix [`DistType`](@ref).
  - `port`: hierarchical clustering portfolio .
  - `clust_alg`: hierarchical clustering algorithm [`ClustAlg`](@ref).
  - `clust_opt`: options for determining the number of clusters [`ClustOpt`](@ref).

# Outputs

  - `idx`: clustering assignments after cutting the tree into `k` levels [`Clustering.cutree`](https://juliastats.org/Clustering.jl/stable/hclust.html#Clustering.cutree).
  - `clustering`: [`Clustering.Hclust`](https://juliastats.org/Clustering.jl/stable/hclust.html#Clustering.Hclust) of the portfolio assets.
  - `k`: optimum number of clusters.
  - `S`: `N×N` asset correlation matrix.
  - `D`: `N×N` asset distance matrix.
"""
=#

export cluster_assets!
