"""
```
connection_matrix(port::AbstractPortfolio; X::AbstractMatrix = port.returns,
                  cor_type::PortfolioOptimiserCovCor = PortCovCor(),
                  dist_type::DistanceMethod = DistanceCanonical(),
                  network_type::NetworkType = MST())
```

Compute the connection matrix [`connection_matrix`](@ref). See the estimators' docs for details.

# Inputs

  - `port`: portfolio [`AbstractPortfolio`](@ref).
  - `X`: `T×N` returns matrix.
  - `cor_type`: correlation matrix estimator [`PortfolioOptimiserCovCor`](@ref).
  - `dist_type`: method for computing the distance matrix [`DistanceMethod`](@ref).
  - `network_type`: method for computing the asset network [`NetworkType`](@ref).

# Outputs

  - `C`: `N×N` connection matrix.
"""
function connection_matrix(port::AbstractPortfolio; X::AbstractMatrix = port.returns,
                           cor_type::PortfolioOptimiserCovCor = PortCovCor(),
                           dist_type::DistanceMethod = DistanceCanonical(),
                           network_type::NetworkType = MST())
    return connection_matrix(X; cor_type = cor_type, dist_type = dist_type,
                             network_type = network_type)
end

"""
```
centrality_vector(port::AbstractPortfolio; X::AbstractMatrix = port.returns,
                  cor_type::PortfolioOptimiserCovCor = PortCovCor(),
                  dist_type::DistanceMethod = DistanceCanonical(),
                  network_type::NetworkType = MST())
```

Compute the centrality vector [`centrality_vector`](@ref). See the estimators' docs for details.

# Inputs

  - `port`: portfolio [`AbstractPortfolio`](@ref).
  - `X`: `T×N` returns matrix.
  - `cor_type`: correlation matrix estimator [`PortfolioOptimiserCovCor`](@ref).
  - `dist_type`: method for computing the distance matrix [`DistanceMethod`](@ref).
  - `network_type`: method for computing the asset network [`NetworkType`](@ref).

# Outputs

  - `C`: `N×1` centrality vector.
"""
function centrality_vector(port::AbstractPortfolio; X::AbstractMatrix = port.returns,
                           cor_type::PortfolioOptimiserCovCor = PortCovCor(),
                           dist_type::DistanceMethod = DistanceCanonical(),
                           network_type::NetworkType = MST())
    return centrality_vector(X; cor_type = cor_type, dist_type = dist_type,
                             network_type = network_type)
end

function cluster_matrix(port::AbstractPortfolio; X::AbstractMatrix = port.returns,
                        cor_type::PortfolioOptimiserCovCor = PortCovCor(),
                        dist_type::DistanceMethod = DistanceCanonical(),
                        hclust_alg::HClustAlg = HAC(), hclust_opt::HCOpt = HCOpt())
    return cluster_matrix(X; cor_type = cor_type, dist_type = dist_type,
                          hclust_alg = hclust_alg, hclust_opt = hclust_opt)
end

function connected_assets(port::AbstractPortfolio; X::AbstractMatrix = port.returns,
                          type::Symbol = isa(port, Portfolio) ? :Trad : :HRP,
                          cor_type::PortfolioOptimiserCovCor = PortCovCor(),
                          dist_type::DistanceMethod = DistanceCanonical(),
                          network_type::NetworkType = MST())
    return connected_assets(X, port.optimal[type].weights; cor_type = cor_type,
                            dist_type = dist_type, network_type = network_type)
end

function related_assets(port::AbstractPortfolio; X::AbstractMatrix = port.returns,
                        type::Symbol = isa(port, Portfolio) ? :Trad : :HRP,
                        cor_type::PortfolioOptimiserCovCor = PortCovCor(),
                        dist_type::DistanceMethod = DistanceCanonical(),
                        hclust_alg::HClustAlg = HAC(), hclust_opt::HCOpt = HCOpt())
    return related_assets(X, port.optimal[type].weights; cor_type = cor_type,
                          dist_type = dist_type, hclust_alg = hclust_alg,
                          hclust_opt = hclust_opt)
end
