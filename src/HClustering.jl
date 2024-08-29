
function is_leaf(a::ClusterNode)
    return isnothing(a.left)
end
function pre_order(a::ClusterNode, func::Function = x -> x.id)
    n = a.count
    curNode = Vector{ClusterNode}(undef, 2 * n)
    lvisited = Set()
    rvisited = Set()
    curNode[1] = a
    k = 1
    preorder = Int[]

    while k >= 1
        nd = curNode[k]
        ndid = nd.id
        if is_leaf(nd)
            push!(preorder, func(nd))
            k = k - 1
        else
            if ndid ∉ lvisited
                curNode[k + 1] = nd.left
                push!(lvisited, ndid)
                k = k + 1
            elseif ndid ∉ rvisited
                curNode[k + 1] = nd.right
                push!(rvisited, ndid)
                k = k + 1
                # If we've visited the left and right of this non-leaf
                # node already, go up in the tree.
            else
                k = k - 1
            end
        end
    end

    return preorder
end
function to_tree(a::Hclust)
    n = length(a.order)
    d = Vector{ClusterNode}(undef, 2 * n - 1)
    for i ∈ 1:n
        d[i] = ClusterNode(i)
    end
    merges = a.merges
    heights = a.heights
    nd = nothing

    for (i, height) ∈ pairs(heights)
        fi = merges[i, 1]
        fj = merges[i, 2]

        fi = fi < 0 ? -fi : fi + n
        fj = fj < 0 ? -fj : fj + n

        nd = ClusterNode(i + n, d[fi], d[fj], height)
        d[n + i] = nd
    end
    return nd, d
end
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
function _calc_k(::TwoDiff, dist::AbstractMatrix, clustering, max_k::Integer)
    return _two_diff_gap_stat(dist, clustering, max_k)
end
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
function _calc_k(method::StdSilhouette, dist::AbstractMatrix, clustering, max_k::Integer)
    return _std_silhouette_score(dist, clustering, max_k, method.metric)
end
function calc_k(hclust_opt::HCType, dist::AbstractMatrix, clustering)
    if !iszero(hclust_opt.k)
        return hclust_opt.k
    end
    return _calc_k(hclust_opt.k_method, dist, clustering, hclust_opt.max_k)
end
function _hcluster(ca::HAC, portfolio::HCPortfolio, hclust_opt::HCType = HCType())
    clustering = hclust(portfolio.dist; linkage = ca.linkage,
                        branchorder = hclust_opt.branchorder)
    k = calc_k(hclust_opt, portfolio.dist, clustering)

    return clustering, k
end
function _hcluster(ca::DBHT, portfolio::HCPortfolio, hclust_opt::HCType = HCType())
    S = portfolio.cor
    D = portfolio.dist
    S = dbht_similarity(ca.similarity, S, D)

    clustering = DBHTs(D, S; branchorder = hclust_opt.branchorder, method = ca.root_method)[end]
    k = calc_k(hclust_opt, D, clustering)

    return clustering, k
end
function cluster_assets(portfolio::HCPortfolio; hclust_alg::HClustAlg = HAC(),
                        hclust_opt::HCType = HCType())
    clustering, k = _hcluster(hclust_alg, portfolio, hclust_opt)

    idx = cutree(clustering; k = k)

    return idx, clustering, k
end
function cluster_assets!(portfolio::HCPortfolio; hclust_alg::HClustAlg = HAC(),
                         hclust_opt::HCType = HCType())
    clustering, k = _hcluster(hclust_alg, portfolio, hclust_opt)

    portfolio.clusters = clustering
    portfolio.k = k

    return nothing
end
function _hcluster(ca::HAC, X::AbstractMatrix,
                   cor_type::PortfolioOptimiserCovCor = PortCovCor(),
                   dist_type::DistanceMethod = DistanceDefault(),
                   hclust_opt::HCType = HCType())
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
                   hclust_opt::HCType = HCType())
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
function cluster_assets(X::AbstractMatrix;
                        cor_type::PortfolioOptimiserCovCor = PortCovCor(),
                        dist_type::DistanceMethod = DistanceDefault(),
                        hclust_alg::HClustAlg = HAC(), hclust_opt::HCType = HCType())
    clustering, k, S, D = _hcluster(hclust_alg, X, cor_type, dist_type, hclust_opt)

    idx = cutree(clustering; k = k)

    return idx, clustering, k, S, D
end
function cluster_assets(portfolio::Portfolio;
                        cor_type::PortfolioOptimiserCovCor = PortCovCor(),
                        dist_type::DistanceMethod = DistanceDefault(),
                        hclust_alg::HClustAlg = HAC(), hclust_opt::HCType = HCType())
    return cluster_assets(portfolio.returns; cor_type = cor_type, dist_type = dist_type,
                          hclust_alg = hclust_alg, hclust_opt = hclust_opt)
end

export is_leaf, pre_order, to_tree, calc_k, cluster_assets, cluster_assets!
