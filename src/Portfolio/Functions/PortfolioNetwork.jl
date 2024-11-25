"""
```
connection_matrix(port::AbstractPortfolio; X::AbstractMatrix = port.returns,
                  cor_type::PortfolioOptimiserCovCor = PortCovCor(),
                  dist_type::DistMethod = DistCanonical(),
                  network_type::NetworkType = MST())
```

Compute the connection matrix [`connection_matrix`](@ref). See the argument types' docs for details.

# Inputs

  - `port`: portfolio [`AbstractPortfolio`](@ref).
  - `X`: `T×N` returns matrix.
  - `cor_type`: correlation matrix estimator [`PortfolioOptimiserCovCor`](@ref).
  - `dist_type`: method for computing the distance matrix [`DistMethod`](@ref).
  - `network_type`: method for computing the asset network [`NetworkType`](@ref).

# Outputs

  - `C`: `N×N` connection-based adjacency matrix.
"""
function connection_matrix(port::AbstractPortfolio; X::AbstractMatrix = port.returns,
                           cor_type::PortfolioOptimiserCovCor = PortCovCor(),
                           dist_type::DistMethod = DistCanonical(),
                           network_type::NetworkType = MST())
    return connection_matrix(X; cor_type = cor_type, dist_type = dist_type,
                             network_type = network_type)
end

"""
```
centrality_vector(port::AbstractPortfolio; X::AbstractMatrix = port.returns,
                  cor_type::PortfolioOptimiserCovCor = PortCovCor(),
                  dist_type::DistMethod = DistCanonical(),
                  network_type::NetworkType = MST())
```

Compute the centrality vector [`centrality_vector`](@ref). See the argument types' docs for details.

# Inputs

  - `port`: portfolio [`AbstractPortfolio`](@ref).
  - `X`: `T×N` returns matrix.
  - `cor_type`: correlation matrix estimator [`PortfolioOptimiserCovCor`](@ref).
  - `dist_type`: method for computing the distance matrix [`DistMethod`](@ref).
  - `network_type`: method for computing the asset network [`NetworkType`](@ref).

# Outputs

  - `C`: `N×1` centrality vector.
"""
function centrality_vector(port::AbstractPortfolio; X::AbstractMatrix = port.returns,
                           cor_type::PortfolioOptimiserCovCor = PortCovCor(),
                           dist_type::DistMethod = DistCanonical(),
                           network_type::NetworkType = MST())
    return centrality_vector(X; cor_type = cor_type, dist_type = dist_type,
                             network_type = network_type)
end

"""
```
cluster_matrix(port::AbstractPortfolio; X::AbstractMatrix = port.returns,
               cor_type::PortfolioOptimiserCovCor = PortCovCor(),
               dist_type::DistMethod = DistCanonical(),
               clust_alg::ClustAlg = HAC(), clust_opt::ClustOpt = ClustOpt())
```

Compute the centrality vector [`cluster_matrix`](@ref). See the argument types' docs for details.

# Inputs

  - `port`: portfolio [`AbstractPortfolio`](@ref).
  - `X`: `T×N` returns matrix.
  - `cor_type`: correlation matrix estimator [`PortfolioOptimiserCovCor`](@ref).
  - `dist_type`: method for computing the distance matrix [`DistMethod`](@ref).
  - `clust_alg`: method for hierarhically clustering assets [`ClustAlg`](@ref).
  - `clust_opt`: options for determining the number of clusters [`ClustOpt`](@ref).

# Outputs

  - `C`: `N×N` cluster-based adjacency matrix.
"""
function cluster_matrix(port::AbstractPortfolio; X::AbstractMatrix = port.returns,
                        cor_type::PortfolioOptimiserCovCor = PortCovCor(),
                        dist_type::DistMethod = DistCanonical(),
                        clust_alg::ClustAlg = HAC(), clust_opt::ClustOpt = ClustOpt())
    return cluster_matrix(X; cor_type = cor_type, dist_type = dist_type,
                          clust_alg = clust_alg, clust_opt = clust_opt)
end

"""
```
connected_assets(port::AbstractPortfolio; X::AbstractMatrix = port.returns,
                 cor_type::PortfolioOptimiserCovCor = PortCovCor(),
                 dist_type::DistMethod = DistCanonical(),
                 network_type::NetworkType = MST())
```

Compute the percentage of the portfolio comprised of connected assets [`connected_assets`](@ref) via a connection-based adjacency matrix. See the argument types' docs for details.

# Inputs

  - `port`: portfolio [`AbstractPortfolio`](@ref).
  - `X`: `T×N` returns matrix.
  - `cor_type`: correlation matrix estimator [`PortfolioOptimiserCovCor`](@ref).
  - `dist_type`: method for computing the distance matrix [`DistMethod`](@ref).
  - `network_type`: method for computing the asset network [`NetworkType`](@ref).

# Outputs

  - `c`: percentage of the portfolio comprised of assets connected via a connection-based adjacency matrix.
"""
function connected_assets(port::AbstractPortfolio; X::AbstractMatrix = port.returns,
                          type::Symbol = if isa(port, Portfolio) ||
                                            isa(port, OmniPortfolio)
                              :Trad
                          else
                              :HRP
                          end, cor_type::PortfolioOptimiserCovCor = PortCovCor(),
                          dist_type::DistMethod = DistCanonical(),
                          network_type::NetworkType = MST())
    return connected_assets(X, port.optimal[type].weights; cor_type = cor_type,
                            dist_type = dist_type, network_type = network_type)
end

"""
```
related_assets(port::AbstractPortfolio; X::AbstractMatrix = port.returns,
               type::Symbol = isa(port, Portfolio) || isa(port, OmniPortfolio) ? :Trad : :HRP,
               cor_type::PortfolioOptimiserCovCor = PortCovCor(),
               dist_type::DistMethod = DistCanonical(),
               clust_alg::ClustAlg = HAC(), clust_opt::ClustOpt = ClustOpt())
```

Compute the percentage of the portfolio comprised of related assets  [`related_assets`](@ref) via a cluster-based adjacency matrix. See the argument types' docs for details.

# Inputs

  - `port`: portfolio [`AbstractPortfolio`](@ref).
  - `X`: `T×N` returns matrix.
  - `cor_type`: correlation matrix estimator [`PortfolioOptimiserCovCor`](@ref).
  - `dist_type`: method for computing the distance matrix [`DistMethod`](@ref).
  - `clust_alg`: method for hierarhically clustering assets [`ClustAlg`](@ref).
  - `clust_opt`: options for determining the number of clusters [`ClustOpt`](@ref).

# Outputs

  - `c`: percentage of the portfolio comprised of related assets via a connection-based adjacency matrix.
"""
function related_assets(port::AbstractPortfolio; X::AbstractMatrix = port.returns,
                        type::Symbol = if isa(port, Portfolio) || isa(port, OmniPortfolio)
                            :Trad
                        else
                            :HRP
                        end, cor_type::PortfolioOptimiserCovCor = PortCovCor(),
                        dist_type::DistMethod = DistCanonical(),
                        clust_alg::ClustAlg = HAC(), clust_opt::ClustOpt = ClustOpt())
    return related_assets(X, port.optimal[type].weights; cor_type = cor_type,
                          dist_type = dist_type, clust_alg = clust_alg,
                          clust_opt = clust_opt)
end

function average_centrality(port::AbstractPortfolio; X::AbstractMatrix = port.returns,
                            type::Symbol = if if !(isa(port, Portfolio))
                                isa(port, OmniPortfolio)
                            end
                                :Trad
                            else
                                :HRP
                            end, cor_type::PortfolioOptimiserCovCor = PortCovCor(),
                            dist_type::DistMethod = DistCanonical(),
                            network_type::NetworkType = MST())
    return average_centrality(X, port.optimal[type].weights; cor_type = cor_type,
                              dist_type = dist_type, network_type = network_type)
end
