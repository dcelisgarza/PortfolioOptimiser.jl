# Copywrite (c) 2025
# Author: Daniel Celis Garza <daniel.celis.garza@gmail.com>
# SPDX-License-Identifier: MIT

function naive_risk(::Equal, returns, args...)
    N = size(returns, 2)
    return fill(eltype(returns)(inv(N)), N)
end
function naive_risk(rm::Union{RiskMeasure, HCRiskMeasure}, returns, long_fees, rebalance)
    N = size(returns, 2)
    inv_risk = Vector{eltype(returns)}(undef, N)
    w = Vector{eltype(returns)}(undef, N)
    for i ∈ eachindex(w)
        w .= zero(eltype(returns))
        w[i] = one(eltype(returns))
        risk = calc_risk(rm, w; X = returns, long_fees = long_fees, rebalance = rebalance)
        inv_risk[i] = inv(risk)
    end
    return inv_risk / sum(inv_risk)
end
function set_hc_rm_sigma!(rm::RMSigma, sigma, cluster)
    sigma_old = rm.sigma
    if isnothing(rm.sigma) || isempty(rm.sigma)
        rm.sigma = view(sigma, cluster, cluster)
    else
        rm.sigma = view(sigma_old, cluster, cluster)
    end
    return sigma_old
end
function set_hc_rm_sigma!(args...)
    return nothing
end
function unset_hc_rm_sigma!(rm::RMSigma, sigma_old)
    rm.sigma = sigma_old
    return nothing
end
function unset_hc_rm_sigma!(args...)
    return nothing
end
function unset_hc_rm_skew!(rm::RMSkew, old_V, old_skew)
    rm.skew = old_skew
    rm.V = old_V
    return nothing
end
function unset_hc_rm_skew!(args...)
    return nothing
end
function cluster_risk(port, sigma, returns, cluster, rm::Union{RiskMeasure, HCRiskMeasure})
    sigma_old = set_hc_rm_sigma!(rm, sigma, cluster)
    cret = view(returns, :, cluster)
    clong_fees, crebalance = get_cluster_fees(port, cluster)
    old_V, old_skew = gen_cluster_skew_sskew(rm, port, cluster)
    cw = naive_risk(rm, cret, clong_fees, crebalance)
    crisk = calc_risk(rm, cw; X = cret, long_fees = clong_fees, rebalance = crebalance)
    unset_hc_rm_sigma!(rm, sigma_old)
    unset_hc_rm_skew!(rm, old_V, old_skew)
    return crisk
end
