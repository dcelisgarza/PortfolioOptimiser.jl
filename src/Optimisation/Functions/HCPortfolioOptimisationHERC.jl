function _optimise!(::HERC, port::HCPortfolio, rmi::Union{AbstractVector, <:RiskMeasure},
                    rmo::Union{AbstractVector, <:RiskMeasure}, obji::ObjectiveFunction,
                    objo::ObjectiveFunction, kellyi::RetType, kellyo::RetType, w_min, w_max)
    nodes = to_tree(port.clusters)[2]
    heights = [i.height for i ∈ nodes]
    nodes = nodes[sortperm(heights; rev = true)]

    weights = ones(eltype(port.returns), size(port.returns, 2))

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
        for r ∈ rmo
            solver_flag = false
            if hasproperty(r, :solvers) && (isnothing(r.solvers) || isempty(r.solvers))
                r.solvers = port.solvers
                solver_flag = true
            end
            scale = r.settings.scale
            for cluster ∈ clusters
                _risk = cluster_risk(port, objo, kellyo, cluster, r) * scale
                if issubset(cluster, ln)
                    lrisk += _risk
                    append!(lc, cluster)
                elseif issubset(cluster, rn)
                    rrisk += _risk
                    append!(rc, cluster)
                end
            end
            if solver_flag
                r.solvers = nothing
            end
        end
        # Allocate weight to clusters.
        alpha_1 = one(lrisk) - lrisk / (lrisk + rrisk)
        # Weight constraints.
        alpha_1 = cluster_weight_bounds(w_min, w_max, weights, lc, rc, alpha_1)

        weights[ln] *= alpha_1
        weights[rn] *= 1 - alpha_1
    end

    risk = zeros(eltype(port.returns), size(port.returns, 2))
    for i ∈ 1:(port.k)
        cidx = idx .== i
        clusters = findall(cidx)
        for r ∈ rmi
            solver_flag = false
            if hasproperty(r, :solvers) && (isnothing(r.solvers) || isempty(r.solvers))
                r.solvers = port.solvers
                solver_flag = true
            end
            scale = r.settings.scale
            risk[cidx] .+= naive_risk(port, obji, kellyi, clusters, r) * scale
            if solver_flag
                r.solvers = nothing
            end
        end
        weights[cidx] .*= risk[cidx]
    end

    return weights
end
