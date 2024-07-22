#=
"""
```
_std_silhouette_score(dist, clustering, max_k = 0)
```
"""
function _std_silhouette_score(dist, clustering, max_k = 0, metric = nothing)
    N = size(dist, 1)
    cluster_lvls = [cutree(clustering; k = i) for i ∈ 1:N]

    if iszero(max_k)
        max_k = ceil(Int, sqrt(size(dist, 1)))
    end

    c1 = min(N, max_k)
    W_list = Vector{eltype(dist)}(undef, c1)
    W_list[1] = -Inf
    for i ∈ 2:c1
        lvl = cluster_lvls[i]
        sl = silhouettes(lvl, dist; metric = metric)
        msl = mean(sl)
        W_list[i] = msl / std(sl; mean = msl)
    end

    k = all(.!isfinite.(W_list)) ? length(W_list) : k = argmax(W_list)

    return k
end

"""
```
_two_diff_gap_stat(dist, clustering, max_k = 0)
```
"""
function _two_diff_gap_stat(dist, clustering, max_k = 0)
    N = size(dist, 1)
    cluster_lvls = [cutree(clustering; k = i) for i ∈ 1:N]

    if iszero(max_k)
        max_k = ceil(Int, sqrt(size(dist, 1)))
    end

    c1 = min(N, max_k)
    W_list = Vector{eltype(dist)}(undef, c1)

    for i ∈ 1:c1
        lvl = cluster_lvls[i]
        c2 = maximum(unique(lvl))
        D_list = Vector{eltype(dist)}(undef, c2)
        for j ∈ 1:c2
            cluster = findall(lvl .== j)
            cluster_dist = dist[cluster, cluster]
            if isempty(cluster_dist)
                continue
            end
            M = size(cluster_dist, 1)
            C_list = Vector{eltype(dist)}(undef, Int(M * (M - 1) / 2))
            k = 1
            for col ∈ 1:M
                for row ∈ (col + 1):M
                    C_list[k] = cluster_dist[row, col]
                    k += 1
                end
            end
            D_list[j] = k == 1 ? zero(eltype(dist)) : std(C_list; corrected = false)
        end
        W_list[i] = sum(D_list)
    end

    gaps = fill(-Inf, c1)

    if c1 > 2
        gaps[1:(end - 2)] .= W_list[1:(end - 2)] .+ W_list[3:end] .- 2 * W_list[2:(end - 1)]
    end

    k = all(isinf.(gaps)) ? length(gaps) : k = argmax(gaps)

    return k
end
=#
function _calc_k(::TwoDiff, dist::AbstractMatrix, clustering, max_k::Integer)
    return _two_diff_gap_stat(dist, clustering, max_k)
end
function _calc_k(method::StdSilhouette, dist::AbstractMatrix, clustering, max_k::Integer)
    return _std_silhouette_score(dist, clustering, max_k, method.metric)
end
function calc_k(hclust_opt::HClustOpt, dist::AbstractMatrix, clustering)
    if !iszero(hclust_opt.k)
        return hclust_opt.k
    end

    return _calc_k(hclust_opt.k_method, dist, clustering, hclust_opt.max_k)
end
function _hcluster(ca::HAClustering, portfolio::HCPortfolio2,
                   hclust_opt::HClustOpt = HClustOpt())
    clustering = hclust(portfolio.dist; linkage = ca.linkage,
                        branchorder = hclust_opt.branchorder)
    k = calc_k(hclust_opt, portfolio.dist, clustering)

    return clustering, k
end
function _hcluster(ca::DBHT, portfolio::HCPortfolio2, hclust_opt::HClustOpt = HClustOpt())
    S = portfolio.cor
    D = portfolio.dist
    S = dbht_similarity(ca.similarity, S, D)

    clustering = DBHTs(D, S; branchorder = hclust_opt.branchorder, method = ca.root_method)[end]
    k = calc_k(hclust_opt, D, clustering)

    return clustering, k
end
function cluster_assets2(portfolio::HCPortfolio2; hclust_alg::HClustAlg = HAClustering(),
                         hclust_opt::HClustOpt = HClustOpt())
    clustering, k = _hcluster(hclust_alg, portfolio, hclust_opt)

    idx = cutree(clustering; k = k)

    return idx, clustering, k
end
function cluster_assets2!(ca::HClustAlg, portfolio::HCPortfolio2,
                          hclust_opt::HClustOpt = HClustOpt())
    clustering, k = _hcluster(ca, portfolio, hclust_opt)

    portfolio.clusters = clustering
    portfolio.k = k

    return nothing
end
function _hcluster(ca::HAClustering, X::AbstractMatrix,
                   cor_type::PortfolioOptimiserCovCor = PortCovCor(),
                   dist_type::DistanceMethod = DistanceDefault(),
                   hclust_opt::HClustOpt = HClustOpt())
    dist_type = _get_default_dist(dist_type, cor_type)
    if hasproperty(cor_type.ce, :absolute) && hasproperty(dist_type, :absolute)
        dist_type.absolute = cor_type.ce.absolute
    end

    S = cor(cor_type, X)
    D = dist(dist_type, S, X)

    clustering = hclust(D; linkage = ca.linkage, branchorder = hclust_opt.branchorder)
    k = calc_k(hclust_opt, D, clustering)

    return clustering, k, S, D
end
function _hcluster(ca::DBHT, X::AbstractMatrix,
                   cor_type::PortfolioOptimiserCovCor = PortCovCor(),
                   dist_type::DistanceMethod = DistanceDefault(),
                   hclust_opt::HClustOpt = HClustOpt())
    dist_type = _get_default_dist(dist_type, cor_type)
    if hasproperty(cor_type.ce, :absolute) && hasproperty(dist_type, :absolute)
        dist_type.absolute = cor_type.ce.absolute
    end

    S = cor(cor_type, X)
    D = dist(dist_type, S, X)
    S = dbht_similarity(ca.similarity, S, D)

    clustering = DBHTs(D, S; branchorder = hclust_opt.branchorder, method = ca.root_method)[end]
    k = calc_k(hclust_opt, D, clustering)

    return clustering, k, S, D
end

function cluster_assets2(X::AbstractMatrix;
                         cor_type::PortfolioOptimiserCovCor = PortCovCor(),
                         dist_type::DistanceMethod = DistanceDefault(),
                         hclust_alg::HClustAlg = HAClustering(),
                         hclust_opt::HClustOpt = HClustOpt())
    clustering, k, S, D = _hcluster(hclust_alg, X, cor_type, dist_type, hclust_opt)

    idx = cutree(clustering; k = k)

    return idx, clustering, k, S, D
end
function cluster_assets2(portfolio::Portfolio2;
                         cor_type::PortfolioOptimiserCovCor = PortCovCor(),
                         dist_type::DistanceMethod = DistanceDefault(),
                         hclust_alg::HClustAlg = HAClustering(),
                         hclust_opt::HClustOpt = HClustOpt())
    return cluster_assets2(portfolio.returns; cor_type = cor_type, dist_type = dist_type,
                           hclust_alg = hclust_alg, hclust_opt = hclust_opt)
end

export TwoDiff, StdSilhouette, _hcluster, cluster_assets2, HClustOpt, cluster_assets2!
