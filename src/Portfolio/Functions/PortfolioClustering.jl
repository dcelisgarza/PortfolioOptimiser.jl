"""
```
_hcluster(ca::HAC, port::HCPortfolio, hclust_opt::HCOpt = HCOpt())
```

Use [`Clustering.hclust`](https://juliastats.org/Clustering.jl/stable/hclust.html#Clustering.hclust) to hierarchically cluster the assets in a hierarchical portfolio [`HCPortfolio`](@ref) using the covariance and distance matrices stored in the portfolio. See the arguments types' docs for details.

# Inputs

  - `ca`: linkage for [`Clustering.hclust`](https://juliastats.org/Clustering.jl/stable/hclust.html#Clustering.hclust).
  - `port`: hierarchical clustering portfolio [`HCPortfolio`](@ref).
  - `hclust_opt`: options for determining the number of clusters [`HCOpt`](@ref).

# Outputs

  - `clustering`: [`Clustering.Hclust`](https://juliastats.org/Clustering.jl/stable/hclust.html#Clustering.Hclust) of the portfolio assets.
  - `k`: optimum number of clusters.
"""
function _hcluster(ca::HAC, port::HCPortfolio, hclust_opt::HCOpt = HCOpt())
    clustering = hclust(port.dist; linkage = ca.linkage,
                        branchorder = hclust_opt.branchorder)
    k = calc_k_clusters(hclust_opt, port.dist, clustering)
    return clustering, k
end

"""
```
_hcluster(ca::DBHT, port::HCPortfolio, hclust_opt::HCOpt = HCOpt())
```

Use [`DBHTs`](@ref) to hierarchically cluster the assets in a hierarchical portfolio [`HCPortfolio`](@ref) using the covariance and distance matrices stored in the portfolio. See the arguments types' docs for details.

# Inputs

  - `ca`: [`DBHT`] options for clustering with [`DBHTs`](@ref).
  - `port`: hierarchical clustering portfolio [`HCPortfolio`](@ref).
  - `hclust_opt`: options for determining the number of clusters [`HCOpt`](@ref).

# Outputs

  - `clustering`: [`Clustering.Hclust`](https://juliastats.org/Clustering.jl/stable/hclust.html#Clustering.Hclust) of the portfolio assets.
  - `k`: optimum number of clusters.
"""
function _hcluster(ca::DBHT, port::HCPortfolio, hclust_opt::HCOpt = HCOpt())
    S = port.cor
    D = port.dist
    S = dbht_similarity(ca.similarity, S, D)
    clustering = DBHTs(D, S; branchorder = hclust_opt.branchorder, method = ca.root_method)[end]
    k = calc_k_clusters(hclust_opt, D, clustering)
    return clustering, k
end

"""
```
cluster_assets(port::HCPortfolio; hclust_alg::HClustAlg = HAC(),
               hclust_opt::HCOpt = HCOpt())
```

Hierarchically cluster the assets in a hierarchical portfolio [`HCPortfolio`](@ref) using the covariance and distance matrices stored in the portfolio. See the arguments types' docs for details.

# Inputs

  - `port`: hierarchical clustering portfolio [`HCPortfolio`](@ref).
  - `hclust_alg`: hierarchical clustering algorithm [`HClustAlg`](@ref).
  - `hclust_opt`: options for determining the number of clusters [`HCOpt`](@ref).

# Outputs

  - `idx`: clustering assignments after cutting the tree into `k` levels [`Clustering.cutree`](https://juliastats.org/Clustering.jl/stable/hclust.html#Clustering.cutree).
  - `clustering`: [`Clustering.Hclust`](https://juliastats.org/Clustering.jl/stable/hclust.html#Clustering.Hclust) of the portfolio assets.
  - `k`: optimum number of clusters.
"""
function cluster_assets(port::HCPortfolio; hclust_alg::HClustAlg = HAC(),
                        hclust_opt::HCOpt = HCOpt())
    clustering, k = _hcluster(hclust_alg, port, hclust_opt)
    idx = cutree(clustering; k = k)
    return idx, clustering, k
end

"""
```
cluster_assets!(port::HCPortfolio; hclust_alg::HClustAlg = HAC(),
                hclust_opt::HCOpt = HCOpt())
```

Hierarchically cluster the assets in a hierarchical portfolio [`HCPortfolio`](@ref) using the covariance and distance matrices stored in the portfolio. Save the results in the portfolio. See the arguments types' docs for details.

# Inputs

  - `port`: hierarchical clustering portfolio [`HCPortfolio`](@ref).
  - `hclust_alg`: hierarchical clustering algorithm [`HClustAlg`](@ref).
  - `hclust_opt`: options for determining the number of clusters [`HCOpt`](@ref).
"""
function cluster_assets!(port::HCPortfolio; hclust_alg::HClustAlg = HAC(),
                         hclust_opt::HCOpt = HCOpt())
    clustering, k = _hcluster(hclust_alg, port, hclust_opt)
    port.clusters = clustering
    port.k = k
    return nothing
end

"""
```
cluster_assets(port::Portfolio; cor_type::PortfolioOptimiserCovCor = PortCovCor(),
               dist_type::DistanceMethod = DistCanonical(),
               hclust_alg::HClustAlg = HAC(), hclust_opt::HCOpt = HCOpt())
```

Hierarchically cluster the assets in a hierarchical portfolio [`HCPortfolio`](@ref) using the covariance and distance matrices stored in the portfolio. See the arguments types' docs for details.

# Inputs

  - `cor_type`: correlation matrix estimator [`PortfolioOptimiserCovCor`](@ref).
  - `dist_type`: method for computing the distance matrix [`DistanceMethod`](@ref).
  - `port`: hierarchical clustering portfolio [`HCPortfolio`](@ref).
  - `hclust_alg`: hierarchical clustering algorithm [`HClustAlg`](@ref).
  - `hclust_opt`: options for determining the number of clusters [`HCOpt`](@ref).

# Outputs

  - `idx`: clustering assignments after cutting the tree into `k` levels [`Clustering.cutree`](https://juliastats.org/Clustering.jl/stable/hclust.html#Clustering.cutree).
  - `clustering`: [`Clustering.Hclust`](https://juliastats.org/Clustering.jl/stable/hclust.html#Clustering.Hclust) of the portfolio assets.
  - `k`: optimum number of clusters.
  - `S`: `N×N` asset correlation matrix.
  - `D`: `N×N` asset distance matrix.
"""
function cluster_assets(port::Portfolio; cor_type::PortfolioOptimiserCovCor = PortCovCor(),
                        dist_type::DistanceMethod = DistCanonical(),
                        hclust_alg::HClustAlg = HAC(), hclust_opt::HCOpt = HCOpt())
    return cluster_assets(port.returns; cor_type = cor_type, dist_type = dist_type,
                          hclust_alg = hclust_alg, hclust_opt = hclust_opt)
end

export cluster_assets!
