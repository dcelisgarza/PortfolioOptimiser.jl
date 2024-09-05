function connection_matrix(port::AbstractPortfolio;
                           cor_type::PortfolioOptimiserCovCor = PortCovCor(),
                           dist_type::DistanceMethod = DistanceCanonical(),
                           network_type::NetworkType = MST())
    return connection_matrix(port.returns; cor_type = cor_type, dist_type = dist_type,
                             network_type = network_type)
end

function centrality_vector(port::AbstractPortfolio;
                           cor_type::PortfolioOptimiserCovCor = PortCovCor(),
                           dist_type::DistanceMethod = DistanceCanonical(),
                           network_type::NetworkType = MST())
    return centrality_vector(port.returns; cor_type = cor_type, dist_type = dist_type,
                             network_type = network_type)
end

function cluster_matrix(port::AbstractPortfolio;
                        cor_type::PortfolioOptimiserCovCor = PortCovCor(),
                        dist_type::DistanceMethod = DistanceCanonical(),
                        hclust_alg::HClustAlg = HAC(), hclust_opt::HCOpt = HCOpt())
    return cluster_matrix(port.returns; cor_type = cor_type, dist_type = dist_type,
                          hclust_alg = hclust_alg, hclust_opt = hclust_opt)
end

function connected_assets(port::AbstractPortfolio;
                          type::Symbol = isa(port, Portfolio) ? :Trad : :HRP,
                          cor_type::PortfolioOptimiserCovCor = PortCovCor(),
                          dist_type::DistanceMethod = DistanceCanonical(),
                          network_type::NetworkType = MST())
    return connected_assets(port.returns, port.optimal[type].weights; cor_type = cor_type,
                            dist_type = dist_type, network_type = network_type)
end

function related_assets(port::AbstractPortfolio;
                        type::Symbol = isa(port, Portfolio) ? :Trad : :HRP,
                        cor_type::PortfolioOptimiserCovCor = PortCovCor(),
                        dist_type::DistanceMethod = DistanceCanonical(),
                        hclust_alg::HClustAlg = HAC(), hclust_opt::HCOpt = HCOpt())
    return related_assets(port.returns, port.optimal[type].weights; cor_type = cor_type,
                          dist_type = dist_type, hclust_alg = hclust_alg,
                          hclust_opt = hclust_opt)
end
