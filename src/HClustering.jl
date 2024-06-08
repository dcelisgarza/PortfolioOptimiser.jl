abstract type NumClusterMethod end
struct TwoDiff <: NumClusterMethod end
struct StdSilhouette <: NumClusterMethod end
#=
"""
```
_std_silhouette_score(dist, clustering, max_k = 0)
```
"""
function _std_silhouette_score(dist, clustering, max_k = 0)
    N = size(dist, 1)
    cluster_lvls = [cutree(clustering; k = i) for i ∈ 1:N]

    if iszero(max_k)
        max_k = ceil(Int, sqrt(size(dist, 1)))
    end

    c1 = min(N, max_k) - 1
    W_list = Vector{eltype(dist)}(undef, c1)

    for i ∈ 1:c1
        lvl = cluster_lvls[i + 1]
        sl = silhouettes(lvl, dist)
        msl = mean(sl)
        W_list[i] = msl / std(sl; mean = msl)
    end

    limit_k = floor(Int, min(max_k, sqrt(N), c1))
    W_list = W_list[1:limit_k]

    k = all(.!isfinite.(W_list)) ? length(W_list) : k = argmax(W_list) + 1

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
        mean_dist = 0.0
        for j ∈ 1:c2
            cluster = findall(lvl .== j)
            cluster_dist = dist[cluster, cluster]
            if isempty(cluster_dist)
                continue
            end

            val = 0.0
            counter = 0
            M = size(cluster_dist, 1)
            for col ∈ 1:M
                for row ∈ (col + 1):M
                    val += cluster_dist[row, col]
                    counter += 1
                end
            end
            if counter == 0
                continue
            end
            mean_dist += val / counter
        end
        W_list[i] = mean_dist
    end

    limit_k = floor(Int, min(max_k, sqrt(N)))
    gaps = fill(-Inf, length(W_list))

    if length(W_list) > 2
        gaps[3:end] .= W_list[3:end] .+ W_list[1:(end - 2)] .- 2 * W_list[2:(end - 1)]
    end

    gaps = gaps[1:limit_k]

    k = all(isinf.(gaps)) ? length(gaps) : k = argmax(gaps) + 1

    return k
end
=#
@kwdef mutable struct HClustOpt{T1 <: Integer, T2 <: Integer}
    branchorder::Symbol = :optimal
    k_method::NumClusterMethod = TwoDiff()
    k::T1 = 0
    max_k::T2 = 0
end
function _calc_k(::TwoDiff, dist::AbstractMatrix, clustering, max_k::Integer)
    return _two_diff_gap_stat(dist, clustering, max_k)
end
function _calc_k(::StdSilhouette, dist::AbstractMatrix, clustering, max_k::Integer)
    return _std_silhouette_score(dist, clustering, max_k)
end
function calc_k(hclust_opt::HClustOpt, dist::AbstractMatrix, clustering)
    if !iszero(hclust_opt.k)
        return hclust_opt.k
    end

    return _calc_k(hclust_opt.k_method, dist, clustering, hclust_opt.max_k)
end
function _hcluster(ca::HAClustering, portfolio::HCPortfolio,
                   hclust_opt::HClustOpt = HClustOpt())
    clustering = hclust(portfolio.dist; linkage = ca.linkage,
                        branchorder = hclust_opt.branchorder)
    k = calc_k(hclust_opt, portfolio.dist, clustering)

    return clustering, k
end
function _hcluster(ca::DBHT, portfolio::HCPortfolio, hclust_opt::HClustOpt = HClustOpt())
    S = portfolio.cor
    D = portfolio.dist
    S = dbht_similarity(ca.similarity, S, D)

    clustering = DBHTs(D, S; branchorder = hclust_opt.branchorder, method = ca.root_method)[end]
    k = calc_k(hclust_opt, D, clustering)

    return clustering, k
end
function cluster_assets2(portfolio::HCPortfolio; hclust_alg::HClustAlg = HAClustering(),
                         hclust_opt::HClustOpt = HClustOpt())
    clustering, k = _hcluster(hclust_alg, portfolio, hclust_opt)

    idx = cutree(clustering; k = k)

    return idx, clustering, k
end
function cluster_assets2!(ca::HClustAlg, portfolio::HCPortfolio,
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
function cluster_assets2(portfolio::Portfolio;
                         cor_type::PortfolioOptimiserCovCor = PortCovCor(),
                         dist_type::DistanceMethod = DistanceDefault(),
                         hclust_alg::HClustAlg = HAClustering(),
                         hclust_opt::HClustOpt = HClustOpt())
    return cluster_assets2(portfolio.returns; cor_type = cor_type, dist_type = dist_type,
                           hclust_algo = hclust_algo, hclust_opt = hclust_opt)
end

export TwoDiff, StdSilhouette, _hcluster, cluster_assets2, HClustOpt, cluster_assets2!
