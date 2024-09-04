"""
```
asset_statistics!(port::AbstractPortfolio;
                           cov_type::PortfolioOptimiserCovCor = PortCovCor(),
                           set_cov::Bool = true, mu_type::MeanEstimator = MuSimple(),
                           set_mu::Bool = true, kurt_type::KurtFull = KurtFull(),
                           set_kurt::Bool = true, skurt_type::KurtSemi = KurtSemi(),
                           set_skurt::Bool = true, skew_type::SkewFull = SkewFull(),
                           set_skew::Bool = true, sskew_type::SkewSemi = SkewSemi(),
                           set_sskew::Bool = true,
                           cor_type::PortfolioOptimiserCovCor = PortCovCor(),
                           set_cor::Bool = true,
                           dist_type::DistanceMethod = DistanceCanonical(),
                           set_dist::Bool = true)
```
"""
function asset_statistics!(port::AbstractPortfolio;
                           cov_type::PortfolioOptimiserCovCor = PortCovCor(),
                           set_cov::Bool = true, mu_type::MeanEstimator = MuSimple(),
                           set_mu::Bool = true, kurt_type::KurtFull = KurtFull(),
                           set_kurt::Bool = true, skurt_type::KurtSemi = KurtSemi(),
                           set_skurt::Bool = true, skew_type::SkewFull = SkewFull(),
                           set_skew::Bool = true, sskew_type::SkewSemi = SkewSemi(),
                           set_sskew::Bool = true,
                           cor_type::PortfolioOptimiserCovCor = PortCovCor(),
                           set_cor::Bool = true,
                           dist_type::DistanceMethod = DistanceCanonical(),
                           set_dist::Bool = true)
    returns = port.returns

    if set_cov || set_mu && hasproperty(mu_type, :sigma) && isempty(mu_type.sigma)
        sigma = cov(cov_type, returns)
        if set_cov
            port.cov = sigma
        end
    end
    if set_mu || set_kurt || set_skurt || set_skew || set_sskew
        sigma_flag = false
        if hasproperty(mu_type, :sigma) && isempty(mu_type.sigma)
            mu_type.sigma = sigma
            sigma_flag = true
        end
        mu = vec(mean(mu_type, returns))
        if set_mu
            port.mu = mu
        end
        if sigma_flag
            mu_type.sigma = Matrix{eltype(sigma)}(undef, 0, 0)
        end
    end
    if set_kurt || set_skurt
        if set_kurt
            port.kurt = cokurt(kurt_type, returns, mu)
        end
        if set_skurt
            port.skurt = cokurt(skurt_type, returns, mu)
        end
        port.L_2, port.S_2 = dup_elim_sum_matrices(size(returns, 2))[2:3]
    end
    if set_skew
        port.skew, port.V = coskew(skew_type, returns, mu)
    end
    if set_sskew
        port.sskew, port.SV = coskew(sskew_type, returns, mu)
    end

    if isa(port, HCPortfolio)
        if set_cor || set_dist
            rho = cor(cor_type, returns)
            if set_cor
                port.cor = rho
                port.cor_type = cor_type
            end
        end

        if set_dist
            dist_type = _get_default_dist(dist_type, cor_type)
            port.dist = dist(dist_type, rho, returns)
        end
    end

    return nothing
end

"""
```
wc_statistics!(port::Portfolio, wc::WCType = WCType(); set_box::Bool = true,
                        set_ellipse::Bool = true)
```
"""
function wc_statistics!(port::Portfolio, wc::WCType = WCType(); set_box::Bool = true,
                        set_ellipse::Bool = true)
    returns = port.returns
    cov_type = wc.cov_type
    mu_type = wc.mu_type
    posdef = wc.posdef

    sigma, mu = _sigma_mu(returns, cov_type, mu_type)

    covs = nothing
    cov_mu = nothing
    if set_box
        cov_l, cov_u, d_mu, covs, cov_mu = calc_sets(Box(), wc.box, wc.cov_type, wc.mu_type,
                                                     returns, sigma, mu)
        posdef_fix!(posdef, cov_l)
        posdef_fix!(posdef, cov_u)

        port.cov_l = cov_l
        port.cov_u = cov_u
        port.d_mu = d_mu
    end

    if set_ellipse
        cov_sigma, cov_mu, A_sigma, A_mu = calc_sets(Ellipse(), wc.ellipse, wc.cov_type,
                                                     wc.mu_type, returns, sigma, mu, covs,
                                                     cov_mu)
        posdef_fix!(posdef, cov_sigma)
        posdef_fix!(posdef, cov_mu)

        if wc.diagonal
            cov_mu .= Diagonal(cov_mu)
            cov_sigma .= Diagonal(cov_sigma)
        end

        k_sigma = calc_k_wc(wc.k_sigma, wc.ellipse.q, A_sigma, cov_sigma)
        k_mu = calc_k_wc(wc.k_mu, wc.ellipse.q, A_mu, cov_mu)

        port.cov_mu = cov_mu
        port.cov_sigma = cov_sigma
        port.k_mu = k_mu
        port.k_sigma = k_sigma
    end

    return nothing
end

"""
```
factor_statistics!(port::Portfolio; factor_type::FactorType = FactorType(),
                            cov_type::PortfolioOptimiserCovCor = PortCovCor(),
                            mu_type::MeanEstimator = MuSimple())
```
"""
function factor_statistics!(port::Portfolio; factor_type::FactorType = FactorType(),
                            cov_type::PortfolioOptimiserCovCor = PortCovCor(),
                            mu_type::MeanEstimator = MuSimple())
    returns = port.returns
    f_returns = port.f_returns

    port.f_cov, port.f_mu = _sigma_mu(f_returns, cov_type, mu_type)

    port.fm_mu, port.fm_cov, port.fm_returns, port.loadings = risk_factors(DataFrame(f_returns,
                                                                                     port.f_assets),
                                                                           DataFrame(returns,
                                                                                     port.assets);
                                                                           factor_type = factor_type,
                                                                           cov_type = cov_type,
                                                                           mu_type = mu_type)

    port.loadings_opt = factor_type.method

    return nothing
end

"""
```
black_litterman_statistics!(port::Portfolio; P::AbstractMatrix, Q::AbstractVector,
                                     w::AbstractVector = port.bl_bench_weights,
                                     cov_type::PortfolioOptimiserCovCor = PortCovCor(),
                                     mu_type::MeanEstimator = MuSimple(),
                                     bl_type::BLType = BLType())
```
"""
function black_litterman_statistics!(port::Portfolio; P::AbstractMatrix, Q::AbstractVector,
                                     w::AbstractVector = port.bl_bench_weights,
                                     cov_type::PortfolioOptimiserCovCor = PortCovCor(),
                                     mu_type::MeanEstimator = MuSimple(),
                                     bl_type::BLType = BLType())
    if isempty(w)
        w = fill(1 / size(port.returns, 2), size(port.returns, 2))
    end
    port.bl_bench_weights = w

    if isnothing(bl_type.delta)
        bl_type.delta = (dot(port.mu, w) - bl_type.rf) / dot(w, port.cov, w)
    end

    port.bl_mu, port.bl_cov, missing = black_litterman(bl_type, port.returns, P, Q, w;
                                                       cov_type = cov_type,
                                                       mu_type = mu_type)

    return nothing
end

"""
```
black_litterman_factor_statistics!(port::Portfolio;
                                            w::AbstractVector = port.bl_bench_weights,
                                            B::Union{DataFrame, Nothing} = port.loadings,
                                            P::Union{AbstractMatrix, Nothing} = nothing,
                                            P_f::Union{AbstractMatrix, Nothing} = nothing,
                                            Q::Union{AbstractVector, Nothing} = nothing,
                                            Q_f::Union{AbstractVector, Nothing} = nothing,
                                            factor_type::FactorType = FactorType(),
                                            cov_type::PortfolioOptimiserCovCor = PortCovCor(),
                                            mu_type::MeanEstimator = MuSimple(),
                                            f_cov_type::PortfolioOptimiserCovCor = PortCovCor(),
                                            f_mu_type::MeanEstimator = MuSimple(),
                                            bl_type::BlackLittermanFactor = BBLType())
```
"""
function black_litterman_factor_statistics!(port::Portfolio;
                                            w::AbstractVector = port.bl_bench_weights,
                                            B::Union{DataFrame, Nothing} = port.loadings,
                                            P::Union{AbstractMatrix, Nothing} = nothing,
                                            P_f::Union{AbstractMatrix, Nothing} = nothing,
                                            Q::Union{AbstractVector, Nothing} = nothing,
                                            Q_f::Union{AbstractVector, Nothing} = nothing,
                                            factor_type::FactorType = FactorType(),
                                            cov_type::PortfolioOptimiserCovCor = PortCovCor(),
                                            mu_type::MeanEstimator = MuSimple(),
                                            f_cov_type::PortfolioOptimiserCovCor = PortCovCor(),
                                            f_mu_type::MeanEstimator = MuSimple(),
                                            bl_type::BlackLittermanFactor = BBLType())
    if isempty(w)
        w = fill(1 / size(port.returns, 2), size(port.returns, 2))
    end
    port.bl_bench_weights = w

    if isnothing(bl_type.delta)
        bl_type.delta = (dot(port.mu, w) - bl_type.rf) / dot(w, port.cov, w)
    end

    if isnothing(B) || isempty(B)
        if isempty(port.loadings)
            port.loadings = regression(factor_type.method,
                                       DataFrame(port.f_returns, port.f_assets),
                                       DataFrame(port.returns, port.assets))
            port.loadings_opt = factor_type.method
        end
        B = port.loadings
    else
        port.loadings = B
    end

    namesB = names(B)
    bl_type.constant = "const" ∈ namesB
    B = Matrix(B[!, setdiff(namesB, ("tickers",))])

    port.blfm_mu, port.blfm_cov, missing = black_litterman(bl_type, port.returns; w = w,
                                                           F = port.f_returns, B = B, P = P,
                                                           P_f = P_f, Q = Q, Q_f = Q_f,
                                                           cov_type = cov_type,
                                                           mu_type = mu_type,
                                                           f_cov_type = f_cov_type,
                                                           f_mu_type = f_mu_type)
    return nothing
end

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

export asset_statistics!, wc_statistics!, factor_statistics!, black_litterman_statistics!,
       black_litterman_factor_statistics!, cluster_assets!
