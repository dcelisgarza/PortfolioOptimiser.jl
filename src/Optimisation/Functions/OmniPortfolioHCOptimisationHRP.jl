function hrp_optimise(port::OmniPortfolio, rm::Union{AbstractVector, <:AbstractRiskMeasure},
                      sigma::AbstractMatrix, returns::AbstractMatrix)
    N = size(returns, 2)
    weights = ones(eltype(returns), N)
    items = [port.clusters.order]

    while length(items) > 0
        items = [i[j:k] for i ∈ items
                 for (j, k) ∈ ((1, div(length(i), 2)), (1 + div(length(i), 2), length(i)))
                 if length(i) > 1]

        for i ∈ 1:2:length(items)
            lc = items[i]
            rc = items[i + 1]
            lrisk = zero(eltype(weights))
            rrisk = zero(eltype(weights))
            for r ∈ rm
                solver_flag = _set_rm_solvers!(r, port.solvers)
                scale = r.settings.scale
                # Left risk.
                lrisk += cluster_risk(port, sigma, returns, lc, r) * scale
                # Right risk.
                rrisk += cluster_risk(port, sigma, returns, rc, r) * scale
                _unset_rm_solvers!(r, solver_flag)
            end
            # Allocate weight to clusters.
            alpha_1 = one(lrisk) - lrisk / (lrisk + rrisk)
            # Weight constraints.
            weights[lc] *= alpha_1
            weights[rc] *= one(alpha_1) - alpha_1
        end
    end
    weights ./= sum(weights)

    return weights
end
