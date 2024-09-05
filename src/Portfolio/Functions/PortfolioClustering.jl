function _hcluster(ca::HAC, port::HCPortfolio, hclust_opt::HCOpt = HCOpt())
    clustering = hclust(port.dist; linkage = ca.linkage,
                        branchorder = hclust_opt.branchorder)
    k = calc_k_clusters(hclust_opt, port.dist, clustering)

    return clustering, k
end
function _hcluster(ca::DBHT, port::HCPortfolio, hclust_opt::HCOpt = HCOpt())
    S = port.cor
    D = port.dist
    S = dbht_similarity(ca.similarity, S, D)

    clustering = DBHTs(D, S; branchorder = hclust_opt.branchorder, method = ca.root_method)[end]
    k = calc_k_clusters(hclust_opt, D, clustering)

    return clustering, k
end
function cluster_assets(port::HCPortfolio; hclust_alg::HClustAlg = HAC(),
                        hclust_opt::HCOpt = HCOpt())
    clustering, k = _hcluster(hclust_alg, port, hclust_opt)

    idx = cutree(clustering; k = k)

    return idx, clustering, k
end
function cluster_assets!(port::HCPortfolio; hclust_alg::HClustAlg = HAC(),
                         hclust_opt::HCOpt = HCOpt())
    clustering, k = _hcluster(hclust_alg, port, hclust_opt)

    port.clusters = clustering
    port.k = k

    return nothing
end
function cluster_assets(port::Portfolio; cor_type::PortfolioOptimiserCovCor = PortCovCor(),
                        dist_type::DistanceMethod = DistanceCanonical(),
                        hclust_alg::HClustAlg = HAC(), hclust_opt::HCOpt = HCOpt())
    return cluster_assets(port.returns; cor_type = cor_type, dist_type = dist_type,
                          hclust_alg = hclust_alg, hclust_opt = hclust_opt)
end

export cluster_assets!
