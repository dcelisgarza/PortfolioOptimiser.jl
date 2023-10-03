
function _naive_risk(portfolio, returns, covariance; rm = :SD, rf = 0.0)
    N = size(returns, 2)
    tcov = eltype(covariance)

    if rm == :Equal
        weights = fill(tcov(1 / N), N)
    else
        inv_risk = Vector{tcov}(undef, N)
        w = Vector{tcov}(undef, N)
        for i in eachindex(w)
            w .= zero(tcov)
            w[i] = one(tcov)
            risk = calc_risk(
                w,
                returns;
                rm = rm,
                rf = rf,
                cov = covariance,
                alpha = portfolio.alpha,
                a_sim = portfolio.a_sim,
                beta = portfolio.beta,
                b_sim = portfolio.b_sim,
                kappa = portfolio.kappa,
                solvers = portfolio.solvers,
            )
            inv_risk[i] = 1 / risk
        end
        weights = inv_risk / sum(inv_risk)
    end

    return weights
end

function _opt_w(
    portfolio,
    assets,
    returns,
    mu,
    icov;
    obj = :Min_Risk,
    kelly = :None,
    rm = :SD,
    rf = 0.0,
    l = 2.0,
)
    port = Portfolio(assets = assets, ret = returns, solvers = portfolio.solvers)
    asset_statistics!(port; calc_kurt = (rm == :Kurt || rm == :SKurt) ? true : false)
    port.cov = icov

    weights = if obj != :Equal
        !isnothing(mu) && (port.mu = mu)
        opt_port!(
            port;
            type = :Trad,
            class = :Classic,
            rm = rm,
            obj = obj,
            kelly = kelly,
            rf = rf,
            l = l,
        )
    else
        opt_port!(port; type = :RP, class = :Classic, rm = rm, kelly = kelly, rf = rf)
    end

    w = !isempty(weights) ? weights.weights : zeros(eltype(returns), length(assets))

    return w, port.fail
end

function _two_diff_gap_stat(dist, clustering, max_k = 10)
    N = size(dist, 1)
    cluster_lvls = [cutree(clustering; k = i) for i in 1:N]

    c1 = min(N, max_k)
    W_list = Vector{eltype(dist)}(undef, c1)

    for i in 1:c1
        lvl = cluster_lvls[i]
        c2 = maximum(unique(lvl))
        mean_dist = 0.0
        for j in 1:c2
            cluster = findall(lvl .== j)
            cluster_dist = dist[cluster, cluster]
            isempty(cluster_dist) && continue

            val = 0.0
            counter = 0
            M = size(cluster_dist, 1)
            for col in 1:M
                for row in (col + 1):M
                    val += cluster_dist[row, col]
                    counter += 1
                end
            end
            counter == 0 && continue
            mean_dist += val / counter
        end
        W_list[i] = mean_dist
    end

    limit_k = floor(Int, min(max_k, sqrt(N)))
    gaps = fill(-Inf, length(W_list))

    length(W_list) > 2 &&
        (gaps[3:end] .= W_list[3:end] .+ W_list[1:(end - 2)] .- 2 * W_list[2:(end - 1)])

    gaps = gaps[1:limit_k]

    k = all(isinf.(gaps)) ? length(gaps) : k = argmax(gaps) + 1

    return k
end

function _hierarchical_clustering(
    portfolio::HCPortfolio,
    type = :HRP,
    linkage = :single,
    max_k = 10,
    branchorder = :optimal,
)
    codep_type = portfolio.codep_type
    codep = portfolio.codep
    dist = portfolio.dist

    codeps1 = (:Pearson, :Spearman, :Kendall, :Gerber1, :Gerber2, :custom)

    if linkage == :DBHT
        codep = codep_type ∈ codeps1 ? 1 .- dist .^ 2 : codep
        missing, missing, missing, missing, missing, missing, clustering =
            DBHTs(dist, codep, branchorder = branchorder)
    else
        clustering = hclust(
            dist;
            linkage = linkage,
            branchorder = branchorder == :default ? :r : branchorder,
        )
    end

    k = type ∈ (:HERC, :HERC2, :NCO) ? _two_diff_gap_stat(dist, clustering, max_k) : nothing

    return clustering, k
end

function cluster_assets(
    portfolio::HCPortfolio;
    linkage = :single,
    max_k = 10,
    branchorder = :optimal,
    k = portfolio.k,
)
    clustering, tk = _hierarchical_clustering(portfolio, :HERC, linkage, max_k, branchorder)

    k = isnothing(k) ? tk : k

    clustering_idx = cutree(clustering; k = k)

    return DataFrame(Assets = portfolio.assets, Clusters = clustering_idx)
end
export cluster_assets

function _cluster_risk(portfolio, returns, covariance, cluster; rm = :SD, rf = 0.0)
    cret = returns[:, cluster]
    ccov = covariance[cluster, cluster]
    cw = _naive_risk(portfolio, cret, ccov; rm = rm, rf = rf)
    crisk = calc_risk(
        cw,
        cret;
        rm = rm,
        rf = rf,
        cov = ccov,
        alpha = portfolio.alpha,
        a_sim = portfolio.a_sim,
        beta = portfolio.beta,
        b_sim = portfolio.b_sim,
        kappa = portfolio.kappa,
        solvers = portfolio.solvers,
    )

    return crisk
end

function _hr_weight_bounds(upper_bound, lower_bound, weights, lc, rc, alpha_1)
    !(any(upper_bound .< weights) || any(lower_bound .> weights)) && return alpha_1

    lmaxw = weights[lc[1]]
    a1 = sum(upper_bound[lc]) / lmaxw
    a2 = max(sum(lower_bound[lc]) / lmaxw, alpha_1)
    alpha_1 = min(a1, a2)

    rmaxw = weights[rc[1]]
    a1 = sum(upper_bound[rc]) / rmaxw
    a2 = max(sum(lower_bound[rc]) / rmaxw, 1 - alpha_1)
    alpha_1 = 1 - min(a1, a2)

    return alpha_1
end

function _recursive_bisection(
    portfolio;
    rm = :SD,
    rf = 0.0,
    upper_bound = nothing,
    lower_bound = nothing,
)
    N = length(portfolio.assets)
    weights = ones(N)
    w = ones(N)
    sort_order = portfolio.clusters.order
    items = [sort_order]
    returns = portfolio.returns
    covariance = portfolio.cov

    while length(items) > 0
        items = [
            i[j:k] for i in items for
            (j, k) in ((1, div(length(i), 2)), (1 + div(length(i), 2), length(i))) if
            length(i) > 1
        ]

        for i in 1:2:length(items)
            lc = items[i]
            rc = items[i + 1]

            # Left cluster.
            lrisk = _cluster_risk(portfolio, returns, covariance, lc; rm = rm, rf = rf)

            # Right cluster.
            rrisk = _cluster_risk(portfolio, returns, covariance, rc; rm = rm, rf = rf)

            # Allocate weight to clusters.
            alpha_1 = 1 - lrisk / (lrisk + rrisk)

            # Weight constraints.
            alpha_1 = _hr_weight_bounds(upper_bound, lower_bound, weights, lc, rc, alpha_1)

            weights[lc] *= alpha_1
            weights[rc] *= 1 - alpha_1
        end
    end

    return weights
end

struct ClusterNode{tid, tl, tr, td, tcnt}
    id::tid
    left::tl
    right::tr
    dist::td
    count::tcnt

    function ClusterNode(
        id,
        left::Union{ClusterNode, Nothing} = nothing,
        right::Union{ClusterNode, Nothing} = nothing,
        dist::AbstractFloat = 0.0,
        count::Int = 1,
    )
        icount = isnothing(left) ? count : (left.count + right.count)

        new{typeof(id), typeof(left), typeof(right), typeof(dist), typeof(count)}(
            id,
            left,
            right,
            dist,
            icount,
        )
    end
end
import Base.>, Base.<, Base.==
<(a::ClusterNode, b::ClusterNode) = a.dist < b.dist
>(a::ClusterNode, b::ClusterNode) = a.dist > b.dist
==(a::ClusterNode, b::ClusterNode) = a.dist == b.dist
function is_leaf(a::ClusterNode)
    isnothing(a.left)
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
    for i in 1:n
        d[i] = ClusterNode(i)
    end
    merges = a.merges
    heights = a.heights
    nd = nothing

    for (i, height) in enumerate(heights)
        fi = merges[i, 1]
        fj = merges[i, 2]

        fi = fi < 0 ? -fi : fi + n
        fj = fj < 0 ? -fj : fj + n

        nd = ClusterNode(i + n, d[fi], d[fj], height)
        d[n + i] = nd
    end
    return nd, d
end

function _hierarchical_recursive_bisection(
    portfolio;
    rm = :SD,
    rf = 0.0,
    type = :HERC,
    upper_bound = nothing,
    lower_bound = nothing,
)
    returns = portfolio.returns
    covariance = portfolio.cov
    clustering = portfolio.clusters
    k = portfolio.k
    root, nodes = to_tree(clustering)
    dists = [i.dist for i in nodes]
    idx = sortperm(dists, rev = true)
    nodes = nodes[idx]

    weights = ones(length(portfolio.assets))

    clustering_idx = cutree(clustering; k = k)

    uidx = minimum(clustering_idx):maximum(clustering_idx)

    clusters = Vector{Vector{Int}}(undef, length(uidx))
    for i in eachindex(clusters)
        clusters[i] = findall(clustering_idx .== i)
    end

    # Calculate intra cluster weights. Drill down into clusters closer in similarity.
    for i in nodes[1:(k - 1)]
        is_leaf(i) && continue

        ln = pre_order(i.left)
        rn = pre_order(i.right)

        lrisk = 0.0
        rrisk = 0.0

        lc = Int[]
        rc = Int[]

        if rm == :Equal
            alpha_1 = 0.5
        else
            for j in eachindex(clusters)
                if issubset(clusters[j], ln)
                    _lrisk = _cluster_risk(
                        portfolio,
                        returns,
                        covariance,
                        clusters[j];
                        rm = rm,
                        rf = rf,
                    )
                    lrisk += _lrisk
                    append!(lc, clusters[j])
                elseif issubset(clusters[j], rn)
                    _rrisk = _cluster_risk(
                        portfolio,
                        returns,
                        covariance,
                        clusters[j];
                        rm = rm,
                        rf = rf,
                    )
                    rrisk += _rrisk
                    append!(rc, clusters[j])
                end
            end

            alpha_1 = 1 - lrisk / (lrisk + rrisk)

            alpha_1 = _hr_weight_bounds(upper_bound, lower_bound, weights, lc, rc, alpha_1)
        end

        weights[lc] *= alpha_1
        weights[rc] *= 1 - alpha_1
    end

    # If herc2, then each cluster contributes an equal amount of risk.
    type == :HERC2 && (rm = :Equal)
    # We multiply the intra cluster weights by the weights by the weights of the cluster.
    for i in 1:k
        cidx = clustering_idx .== i
        cret = returns[:, cidx]
        ccov = covariance[cidx, cidx]
        cweights = _naive_risk(portfolio, cret, ccov; rm = rm, rf = rf)
        weights[cidx] .*= cweights
    end

    return weights
end

function _intra_weights(
    portfolio;
    obj = :Min_Risk,
    kelly = :None,
    rm = :SD,
    rf = 0.0,
    l = 2.0,
)
    returns = portfolio.returns
    mu = portfolio.mu
    covariance = portfolio.cov
    clustering = portfolio.clusters
    k = portfolio.k
    clustering_idx = cutree(clustering; k = k)

    intra_weights = zeros(eltype(covariance), length(portfolio.assets), k)
    cfails = Dict{Int, Dict}()
    for i in 1:k
        idx = clustering_idx .== i
        cmu = !isnothing(mu) ? mu[idx] : nothing
        ccov = covariance[idx, idx]
        cret = returns[:, idx]
        cassets = portfolio.assets[idx]
        weights, cfail = _opt_w(
            portfolio,
            cassets,
            cret,
            cmu,
            ccov;
            obj = obj,
            kelly = kelly,
            rm = rm,
            rf = rf,
            l = l,
        )
        intra_weights[idx, i] .= weights
        cfails[i] = cfail
    end

    return intra_weights, cfails
end

function _inter_weights(
    portfolio,
    intra_weights;
    obj = :Min_Risk,
    kelly = :None,
    rm = :SD,
    rf = 0.0,
    l = 2.0,
)
    mu = portfolio.mu
    returns = portfolio.returns
    covariance = portfolio.cov
    tmu = !isnothing(mu) ? transpose(intra_weights) * mu : nothing
    tcov = transpose(intra_weights) * covariance * intra_weights
    tret = returns * intra_weights
    inter_weights, inter_fail = _opt_w(
        portfolio,
        1:length(tmu),
        tret,
        tmu,
        tcov;
        obj = obj,
        kelly = kelly,
        rm = rm,
        rf = rf,
        l = l,
    )

    weights = intra_weights * inter_weights

    return weights, inter_fail
end

function _setup_hr_weights(w_max, w_min, N)
    @assert(
        all(w_max .>= w_min),
        "all upper bounds must be bigger than their corresponding lower bounds"
    )

    upper_bound = if isnothing(w_max)
        ones(N)
    elseif isa(w_max, Real)
        fill(min(1.0, w_max), N)
    else
        min.(1.0, w_max)
    end

    lower_bound = if isnothing(w_min)
        zeros(N)
    elseif isa(w_min, Real)
        fill(max(0.0, w_min), N)
    else
        max.(0.0, w_min)
    end

    return upper_bound, lower_bound
end

function _opt_weight_bounds(upper_bound, lower_bound, weights, max_iter = 100)
    !(any(upper_bound .< weights) || any(lower_bound .> weights)) && return weights

    for _ in 1:max_iter
        !(any(upper_bound .< weights) || any(lower_bound .> weights)) && break

        old_w = copy(weights)
        weights = max.(min.(weights, upper_bound), lower_bound)
        idx = weights .< upper_bound .&& weights .> lower_bound
        w_add = sum(max.(old_w - upper_bound, 0.0))
        w_sub = sum(min.(old_w - lower_bound, 0.0))
        delta = w_add + w_sub

        delta != 0 && (weights[idx] += delta * weights[idx] / sum(weights[idx]))
    end

    return weights
end

function opt_port!(
    portfolio::HCPortfolio;
    type::Symbol = :HRP,
    rm::Symbol = :SD,
    obj::Symbol = :Min_Risk,
    kelly::Symbol = :None,
    rf::Real = 0.0,
    l::Real = 2.0,
    linkage::Symbol = :single,
    k = portfolio.k,
    max_k::Int = 10,
    branchorder = :optimal,
    max_iter = 100,
    cluster = true,
)
    @assert(type ∈ HRTypes, "type must be one of $HRTypes")
    @assert(rm ∈ HRRiskMeasures, "rm must be one of $HRRiskMeasures")
    @assert(obj ∈ HRObjFuncs, "obj must be one of $HRObjFuncs")
    @assert(kelly ∈ KellyRet, "kelly must be one of $KellyRet")
    @assert(linkage ∈ LinkageTypes, "linkage must be one of $LinkageTypes")
    @assert(portfolio.codep_type ∈ CodepTypes, "codep_type must be one of $CodepTypes")
    @assert(
        0 < portfolio.alpha < 1,
        "portfolio.alpha must be greater than 0 and smaller than 1"
    )
    @assert(
        0 < portfolio.kappa < 1,
        "portfolio.kappa must be greater than 0 and smaller than 1"
    )

    N = length(portfolio.assets)

    if cluster
        portfolio.clusters, tk =
            _hierarchical_clustering(portfolio, type, linkage, max_k, branchorder)
        portfolio.k = isnothing(k) ? tk : k
    end

    upper_bound, lower_bound = _setup_hr_weights(portfolio.w_max, portfolio.w_min, N)
    if type == :HRP
        weights = _recursive_bisection(
            portfolio;
            rm = rm,
            rf = rf,
            upper_bound = upper_bound,
            lower_bound = lower_bound,
        )
    elseif type == :HERC || type == :HERC2
        weights = _hierarchical_recursive_bisection(
            portfolio;
            rm = rm,
            rf = rf,
            type = type,
            upper_bound = upper_bound,
            lower_bound = lower_bound,
        )
    else
        intra_weights, intra_fails =
            _intra_weights(portfolio; obj = obj, kelly = kelly, rm = rm, rf = rf, l = l)
        weights, inter_fails = _inter_weights(
            portfolio,
            intra_weights,
            obj = obj,
            kelly = kelly,
            rm = rm,
            rf = rf,
            l = l,
        )
        !isempty(intra_fails) && (portfolio.fail[:intra] = intra_fails)
        !isempty(inter_fails) && (portfolio.fail[:inter] = inter_fails)
    end
    weights = _opt_weight_bounds(upper_bound, lower_bound, weights, max_iter)

    portfolio.p_optimal = DataFrame(tickers = portfolio.assets, weights = weights)
    return portfolio.p_optimal
end

export opt_port!
