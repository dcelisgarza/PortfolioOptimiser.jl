function naive_risk(port, sigma, cluster, rm)
    sigma_old = _set_hc_rm_sigma(rm, sigma, cluster)
    cret = view(port.returns, :, cluster)
    old_V, old_skew = gen_cluster_skew_sskew(rm, port, cluster)
    crisk = _naive_risk(rm, cret)
    _unset_hc_rm_sigma(rm, sigma_old)
    _unset_hc_rm_skew(rm, old_V, old_skew)
    return crisk
end
function _optimise!(::HERC, port::HCPortfolio, class::PortClass,
                    rm_i::Union{AbstractVector, <:AbstractRiskMeasure},
                    rm_o::Union{AbstractVector, <:AbstractRiskMeasure}, w_min, w_max)
    sigma, returns = mu_sigma_returns_class(port, class)[2:3]

    nodes = to_tree(port.clusters)[2]
    heights = [i.height for i ∈ nodes]
    nodes = nodes[sortperm(heights; rev = true)]

    weights = ones(eltype(returns), size(returns, 2))

    idx = cutree(port.clusters; k = port.k)

    clusters = Vector{Vector{Int}}(undef, length(minimum(idx):maximum(idx)))
    for i ∈ eachindex(clusters)
        clusters[i] = findall(idx .== i)
    end

    # Treat each cluster as its own portfolio and optimise each one individually.
    # Calculate the weight of each cluster relative to the other clusters.
    for i ∈ nodes[1:(port.k - 1)]
        if is_leaf(i)
            continue
        end

        # Do this recursively accounting for the dendrogram structure.
        ln = pre_order(i.left)
        rn = pre_order(i.right)

        lrisk = 0.0
        rrisk = 0.0

        lc = Int[]
        rc = Int[]
        for r ∈ rm_o
            solver_flag = _set_rm_solvers!(r, port.solvers)
            scale = r.settings.scale
            for cluster ∈ clusters
                if issubset(cluster, ln)
                    lrisk += cluster_risk(port, sigma, cluster, r) * scale
                    append!(lc, cluster)
                elseif issubset(cluster, rn)
                    rrisk += cluster_risk(port, sigma, cluster, r) * scale
                    append!(rc, cluster)
                end
            end
            _unset_rm_solvers!(r, solver_flag)
        end

        risk = lrisk + rrisk
        # Allocate weight to clusters.
        alpha_1 = one(lrisk) - lrisk / risk
        # Weight constraints.
        weights[ln] *= alpha_1
        weights[rn] *= 1 - alpha_1
    end

    risk = zeros(eltype(returns), size(returns, 2))
    for i ∈ 1:(port.k)
        cidx = idx .== i
        clusters = findall(cidx)
        for r ∈ rm_i
            solver_flag = _set_rm_solvers!(r, port.solvers)
            scale = r.settings.scale
            risk[cidx] .+= naive_risk(port, sigma, clusters, r) * scale
            _unset_rm_solvers!(r, solver_flag)
        end
        weights[cidx] .*= risk[cidx]
    end
    weights ./= sum(weights)

    return weights
end
