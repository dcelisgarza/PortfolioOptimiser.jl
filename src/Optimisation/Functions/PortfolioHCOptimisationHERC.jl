# Copywrite (c) 2025
# Author: Daniel Celis Garza <daniel.celis.garza@gmail.com>
# SPDX-License-Identifier: MIT

function get_cluster_fees(port, cluster)
    long_fees = port.long_fees
    rebalance = port.rebalance

    if isa(long_fees, AbstractVector) && !(isempty(long_fees) || all(iszero.(long_fees)))
        long_fees = long_fees[cluster]
    end

    if isa(rebalance, TR)
        rebal_fees = rebalance.val
        benchmark = rebalance.w
        if isa(rebal_fees, AbstractVector)
            rebal_fees = rebal_fees[cluster]
        end
        rebalance = TR(; val = rebal_fees, w = benchmark[cluster])
    end

    return long_fees, rebalance
end
function naive_risk(port::AbstractPortfolio, sigma, returns, cluster,
                    rm::Union{RiskMeasure, HCRiskMeasure})
    old_custom = set_custom_hc_rm!(rm, port, sigma, cluster)
    old_mu, old_target = set_hc_rm_mu_w!(rm, cluster)
    old_sigma = set_hc_rm_sigma!(rm, sigma, cluster)
    old_tr = set_tracking_rm!(rm, cluster)
    old_to = set_turnover_rm!(rm, cluster)
    cret = view(returns, :, cluster)
    clong_fees, crebalance = get_cluster_fees(port, cluster)
    old_V, old_skew = gen_cluster_skew_sskew(rm, port, cluster)
    crisk = naive_risk(rm, cret, clong_fees, crebalance)
    unset_set_hc_rm_mu_w!(rm, old_mu, old_target)
    unset_hc_rm_sigma!(rm, old_sigma)
    unset_hc_rm_skew!(rm, old_V, old_skew)
    unset_tracking_rm!(rm, old_tr)
    unset_turnover_rm!(rm, old_to)
    unset_custom_hc_rm!(rm, old_custom)
    return crisk
end

function herc_scalarise_risk_o(port, sigma, returns, rm, cluster, ::ScalarSum)
    crisk = zero(eltype(returns))
    for r ∈ rm
        solver_flag = set_rm_solvers!(r, port.solvers)
        scale = r.settings.scale
        crisk += cluster_risk(port, sigma, returns, cluster, r) * scale
        unset_rm_solvers!(r, solver_flag)
    end
    return crisk
end
function herc_scalarise_risk_o(port, sigma, returns, rm, cluster,
                               scalarisation::ScalarLogSumExp)
    gamma = scalarisation.gamma
    crisk = zero(eltype(returns))
    for r ∈ rm
        solver_flag = set_rm_solvers!(r, port.solvers)
        scale = r.settings.scale
        crisk += cluster_risk(port, sigma, returns, cluster, r) * scale
        unset_rm_solvers!(r, solver_flag)
    end
    return log(exp(crisk)) / gamma
end
function herc_scalarise_risk_o(port, sigma, returns, rm, cluster, ::ScalarMax)
    crisk = -Inf
    for r ∈ rm
        solver_flag = set_rm_solvers!(r, port.solvers)
        scale = r.settings.scale
        crisk_n = cluster_risk(port, sigma, returns, cluster, r) * scale
        if crisk_n > crisk
            crisk = crisk_n
        end
        unset_rm_solvers!(r, solver_flag)
    end
    return crisk
end
function herc_scalarise_risk_i(port, sigma, returns, rm, cluster, ::ScalarSum)
    risk = zeros(eltype(returns), length(cluster))
    for r ∈ rm
        solver_flag = set_rm_solvers!(r, port.solvers)
        scale = r.settings.scale
        risk .+= naive_risk(port, sigma, returns, cluster, r) * scale
        unset_rm_solvers!(r, solver_flag)
    end
    return risk
end
function herc_scalarise_risk_i(port, sigma, returns, rm, cluster,
                               scalarisation::ScalarLogSumExp)
    gamma = scalarisation.gamma
    risk = zeros(eltype(returns), length(cluster))
    for r ∈ rm
        solver_flag = set_rm_solvers!(r, port.solvers)
        scale = r.settings.scale * gamma
        risk .+= naive_risk(port, sigma, returns, cluster, r) * scale
        unset_rm_solvers!(r, solver_flag)
    end
    return log.(exp.(risk)) / gamma
end
function herc_scalarise_risk_i(port, sigma, returns, rm, cluster, ::ScalarMax)
    trisk = -Inf
    risk = zeros(eltype(returns), length(cluster))
    for r ∈ rm
        solver_flag = set_rm_solvers!(r, port.solvers)
        scale = r.settings.scale
        risk_n = naive_risk(port, sigma, returns, cluster, r) * scale
        if sum(risk_n) > trisk
            risk .= risk_n
        end
        unset_rm_solvers!(r, solver_flag)
    end
    return risk
end
function herc_optimise(port::Portfolio,
                       rm_i::Union{AbstractVector, <:Union{RiskMeasure, HCRiskMeasure}},
                       rm_o::Union{AbstractVector, <:Union{RiskMeasure, HCRiskMeasure}},
                       sigma_i::AbstractMatrix, sigma_o::AbstractMatrix,
                       returns_i::AbstractMatrix, returns_o::AbstractMatrix,
                       scalarisation_i::AbstractScalarisation,
                       scalarisation_o::AbstractScalarisation)
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
        crisk[i] = herc_scalarise_risk_o(port, sigma_o, returns_o, rm_o, clusters[i],
                                         scalarisation_o)
        risk[cidx] .= herc_scalarise_risk_i(port, sigma_i, returns_i, rm_i, clusters[i],
                                            scalarisation_i)
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
                push!(lc, cidx)
            elseif issubset(cluster, rn)
                push!(rc, cidx)
            end
        end

        lrisk = sum(crisk[lc])
        rrisk = sum(crisk[rc])

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
