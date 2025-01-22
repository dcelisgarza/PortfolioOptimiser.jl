# Copywrite (c) 2025
# Author: Daniel Celis Garza <daniel.celis.garza@gmail.com>
# SPDX-License-Identifier: MIT

function hrp_scalarise_risk(port, sigma, returns, rm, lc, rc, ::ScalarSum)
    lrisk = zero(eltype(returns))
    rrisk = zero(eltype(returns))
    for r ∈ rm
        solver_flag = set_rm_solvers!(r, port.solvers)
        scale = r.settings.scale
        # Left risk.
        lrisk += cluster_risk(port, sigma, returns, lc, r) * scale
        # Right risk.
        rrisk += cluster_risk(port, sigma, returns, rc, r) * scale
        unset_rm_solvers!(r, solver_flag)
    end
    return lrisk, rrisk
end
function hrp_scalarise_risk(port, sigma, returns, rm, lc, rc,
                            scalarisation::ScalarLogSumExp)
    gamma = scalarisation.gamma
    igamma = inv(gamma)
    lrisk = zero(eltype(returns))
    rrisk = zero(eltype(returns))
    for r ∈ rm
        solver_flag = set_rm_solvers!(r, port.solvers)
        scale = r.settings.scale * gamma
        # Left risk.
        lrisk += cluster_risk(port, sigma, returns, lc, r) * scale
        # Right risk.
        rrisk += cluster_risk(port, sigma, returns, rc, r) * scale
        unset_rm_solvers!(r, solver_flag)
    end
    return log(exp(lrisk)) * igamma, log(exp(rrisk)) * igamma
end
function hrp_scalarise_risk(port, sigma, returns, rm, lc, rc, ::ScalarMax)
    trisk = -Inf
    lrisk = zero(eltype(returns))
    rrisk = zero(eltype(returns))
    for r ∈ rm
        solver_flag = set_rm_solvers!(r, port.solvers)
        scale = r.settings.scale
        # Left risk.
        lrisk_n = cluster_risk(port, sigma, returns, lc, r) * scale
        # Right risk.
        rrisk_n = cluster_risk(port, sigma, returns, rc, r) * scale
        # Total risk.
        trisk_n = lrisk_n + rrisk_n
        if trisk_n > trisk
            trisk = trisk_n
            lrisk = lrisk_n
            rrisk = rrisk_n
        end
        unset_rm_solvers!(r, solver_flag)
    end
    return lrisk, rrisk
end
function hrp_optimise(port::Portfolio,
                      rm::Union{AbstractVector, <:Union{RiskMeasure, HCRiskMeasure}},
                      sigma::AbstractMatrix, returns::AbstractMatrix,
                      scalarisation::AbstractScalarisation)
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
            lrisk, rrisk = hrp_scalarise_risk(port, sigma, returns, rm, lc, rc,
                                              scalarisation)
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
