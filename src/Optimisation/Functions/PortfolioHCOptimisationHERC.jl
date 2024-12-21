function naive_risk(port, sigma, returns, cluster, rm)
    sigma_old = _set_hc_rm_sigma(rm, sigma, cluster)
    cret = view(returns, :, cluster)
    old_V, old_skew = gen_cluster_skew_sskew(rm, port, cluster)
    crisk = _naive_risk(rm, cret)
    _unset_hc_rm_sigma(rm, sigma_old)
    _unset_hc_rm_skew(rm, old_V, old_skew)
    return crisk
end
function herc_optimise(port::Portfolio, rm_i::Union{AbstractVector, <:AbstractRiskMeasure},
                       rm_o::Union{AbstractVector, <:AbstractRiskMeasure},
                       sigma_i::AbstractMatrix, sigma_o::AbstractMatrix,
                       returns_i::AbstractMatrix, returns_o::AbstractMatrix)
    nodes = to_tree(port.clusters)[2]
    heights = [i.height for i ∈ nodes]
    nodes = nodes[sortperm(heights; rev = true)]

    weights = ones(eltype(returns_o), size(returns_o, 2))
    k = port.k
    idx = cutree(port.clusters; k = k)

    clusters = Vector{Vector{Int}}(undef, k)
    crisk = zeros(eltype(returns_o), k)
    risk = zeros(eltype(returns_i), size(returns_i, 2))
    for i ∈ eachindex(clusters)
        cidx = idx .== i
        clusters[i] = findall(cidx)
        for r ∈ rm_o
            solver_flag = _set_rm_solvers!(r, port.solvers)
            scale = r.settings.scale
            crisk[i] += cluster_risk(port, sigma_o, returns_o, clusters[i], r) * scale
            _unset_rm_solvers!(r, solver_flag)
        end
        for r ∈ rm_i
            solver_flag = _set_rm_solvers!(r, port.solvers)
            scale = r.settings.scale
            risk[cidx] .+= naive_risk(port, sigma_i, returns_i, clusters[i], r) * scale
            _unset_rm_solvers!(r, solver_flag)
        end
        weights[cidx] .*= risk[cidx]
    end

    # Treat each cluster as its own portfolio and optimise each one individually.
    # Calculate the weight of each cluster relative to the other clusters.
    for i ∈ nodes[1:(k - 1)]
        if is_leaf(i)
            continue
        end

        # Do this recursively accounting for the dendrogram structure.
        ln = pre_order(i.left)
        rn = pre_order(i.right)

        lc = Int[]
        rc = Int[]

        for (cidx, cluster) ∈ pairs(clusters)
            if issubset(cluster, ln)
                push!(ln, cidx)
            elseif issubset(cluster, rn)
                push!(rn, cidx)
            end
        end

        lrisk = sum(crisk[ln])
        rrisk = sum(crisk[rn])
        
        risk = lrisk + rrisk
        # Allocate weight to clusters.
        alpha_1 = one(lrisk) - lrisk / risk
        # Weight constraints.
        weights[ln] *= alpha_1
        weights[rn] *= 1 - alpha_1
    end

    weights ./= sum(weights)

    return weights
end
