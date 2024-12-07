"""
```
dbht_similarity(::DBHTExp, S, D)
```

Computes the [`DBHTExp`](@ref) similarity matrix.
"""
function dbht_similarity(::DBHTExp, S, D)
    return exp.(-D)
end
"""
```
dbht_similarity(::DBHTMaxDist, S, D)
```

Computes the [`DBHTMaxDist`](@ref) similarity matrix.
"""
function dbht_similarity(::DBHTMaxDist, S, D)
    return ceil(maximum(D)^2) .- D .^ 2
end

"""
```
is_leaf(a::ClusterNode)
```
"""
function is_leaf(a::ClusterNode)
    return isnothing(a.left)
end

"""
```
pre_order(a::ClusterNode, func::Function = x -> x.id)
```
"""
function pre_order(a::ClusterNode, func::Function = x -> x.id)
    n = a.level
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

"""
```
to_tree(a::Hclust)
```
"""
function to_tree(a::Hclust)
    N = length(a.order)
    d = Vector{ClusterNode}(undef, 2 * N - 1)
    for i ∈ eachindex(a.order)
        d[i] = ClusterNode(i)
    end
    merges = a.merges
    heights = a.heights
    nd = nothing

    for (i, height) ∈ pairs(heights)
        fi = merges[i, 1]
        fj = merges[i, 2]

        fi = fi < 0 ? -fi : fi + N
        fj = fj < 0 ? -fj : fj + N

        nd = ClusterNode(i + N, d[fi], d[fj], height)
        d[N + i] = nd
    end
    return nd, d
end
function _validate_k_value(clustering::Hclust, nodes, k)
    idx = cutree(clustering; k = k)
    clusters = Vector{Vector{Int}}(undef, length(minimum(idx):maximum(idx)))
    for i ∈ eachindex(clusters)
        clusters[i] = findall(idx .== i)
    end

    for i ∈ nodes[1:(k - 1)]
        if is_leaf(i)
            continue
        end
        count = 0
        ln = pre_order(i.left)
        rn = pre_order(i.right)

        for cluster ∈ clusters
            if issubset(cluster, ln) || issubset(cluster, rn)
                count += 1
            end
        end

        if count == 0
            return false
        end
    end
    return true
end
function _valid_k_clusters(arr::AbstractVector, clustering::Hclust)
    nodes = to_tree(clustering)[2]
    heights = [i.height for i ∈ nodes]
    nodes = nodes[sortperm(heights; rev = true)]

    while true
        k = all(.!isfinite.(arr)) ? length(arr) : argmax(arr)
        if _validate_k_value(clustering, nodes, k)
            return k
        elseif all(isinf.(arr))
            return 1
        end
        arr[k] = -Inf
    end
end
function _calc_k_clusters(k::Integer, max_k::Integer, clustering::Hclust)
    N = length(clustering.order)
    if iszero(max_k)
        max_k = ceil(Int, sqrt(N))
    end
    max_k = min(ceil(Int, sqrt(N)), max_k)
    if k > max_k
        k = max_k
    end

    nodes = to_tree(clustering)[2]
    heights = [i.height for i ∈ nodes]
    nodes = nodes[sortperm(heights; rev = true)]
    flag = _validate_k_value(clustering, nodes, k)

    if !flag
        # Above k
        flagu = false
        du = 0
        ku = k
        for i ∈ (k + 1):max_k
            flagu = _validate_k_value(clustering, nodes, i)
            if flagu
                ku = i
                break
            end
        end
        if flagu
            du = ku - k
        end

        # Below k
        flagl = false
        dl = 0
        kl = k
        for i ∈ (k - 1):-1:1
            flagl = _validate_k_value(clustering, nodes, i)
            if flagl
                kl = i
                break
            end
        end
        if flagl
            dl = k - kl
        end

        if du != 0 && dl == 0
            k = ku
        elseif du == 0 && dl != 0
            k = kl
        elseif du == dl
            k = max_k - ku > kl - 1 ? ku : kl
        else
            k = min(du, dl) == du ? ku : kl
        end
    end

    return k
end
function _calc_k_clusters(::TwoDiff, max_k::Integer, dist::AbstractMatrix,
                          clustering::Hclust)
    N = size(dist, 1)
    if iszero(max_k)
        max_k = ceil(Int, sqrt(N))
    end
    c1 = min(ceil(Int, sqrt(N)), max_k)
    cluster_lvls = [cutree(clustering; k = i) for i ∈ 1:c1]
    W_list = Vector{eltype(dist)}(undef, c1)
    W_list[1] = -Inf
    for i ∈ 2:c1
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

    return _valid_k_clusters(gaps, clustering)
end
function _calc_k_clusters(method::StdSilhouette, max_k::Integer, dist::AbstractMatrix,
                          clustering::Hclust)
    metric = method.metric
    N = size(dist, 1)
    if iszero(max_k)
        max_k = ceil(Int, sqrt(N))
    end
    c1 = min(ceil(Int, sqrt(N)), max_k)
    cluster_lvls = [cutree(clustering; k = i) for i ∈ 1:c1]
    W_list = Vector{eltype(dist)}(undef, c1)

    W_list[1] = -Inf
    for i ∈ 2:c1
        lvl = cluster_lvls[i]
        sl = silhouettes(lvl, dist; metric = metric)
        msl = mean(sl)
        W_list[i] = msl / std(sl; mean = msl)
    end

    return _valid_k_clusters(W_list, clustering)
end
"""
```
calc_k_clusters(clust_opt::ClustOpt, dist::AbstractMatrix, clustering)
```
"""
function calc_k_clusters(clust_opt::ClustOpt, dist::AbstractMatrix, clustering)
    return if !iszero(clust_opt.k)
        _calc_k_clusters(clust_opt.k, clust_opt.max_k, clustering)
    else
        _calc_k_clusters(clust_opt.k_method, clust_opt.max_k, dist, clustering)
    end
end

function _clusterise(ca::HAC, X::AbstractMatrix,
                     cor_type::PortfolioOptimiserCovCor = PortCovCor(),
                     dist_type::DistMethod = DistCanonical(),
                     clust_opt::ClustOpt = ClustOpt())
    dist_type = get_default_dist(dist_type, cor_type)
    _set_absolute_dist!(cor_type, dist_type)

    S = cor(cor_type, X)
    D = dist(dist_type, S, X)

    clustering = hclust(D; linkage = ca.linkage, branchorder = clust_opt.branchorder)
    k = calc_k_clusters(clust_opt, D, clustering)

    return clustering, k, S, D
end
function _clusterise(ca::DBHT, X::AbstractMatrix,
                     cor_type::PortfolioOptimiserCovCor = PortCovCor(),
                     dist_type::DistMethod = DistCanonical(),
                     clust_opt::ClustOpt = ClustOpt())
    dist_type = get_default_dist(dist_type, cor_type)
    _set_absolute_dist!(cor_type, dist_type)

    S = cor(cor_type, X)
    D = dist(dist_type, S, X)
    S = dbht_similarity(ca.similarity, S, D)

    clustering = DBHTs(D, S; branchorder = clust_opt.branchorder, method = ca.root_method)[end]
    k = calc_k_clusters(clust_opt, D, clustering)

    return clustering, k, S, D
end
function cluster_assets(X::AbstractMatrix;
                        cor_type::PortfolioOptimiserCovCor = PortCovCor(),
                        dist_type::DistMethod = DistCanonical(),
                        clust_alg::ClustAlg = HAC(), clust_opt::ClustOpt = ClustOpt())
    clustering, k, S, D = _clusterise(clust_alg, X, cor_type, dist_type, clust_opt)
    idx = cutree(clustering; k = k)
    return idx, clustering, k, S, D
end

export dbht_similarity, is_leaf, pre_order, to_tree, calc_k_clusters, cluster_assets
