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
function set_hc_rm_target!(rm::RMTarget, cluster)
    old_target = rm.target
    if isa(rm.target, AbstractVector) && !isempty(rm.target)
        rm.target = view(old_target, cluster)
    end
    return old_target
end
function set_hc_rm_target!(args...)
    return nothing
end
function unset_set_hc_rm_target!(rm::RMTarget, old_target)
    rm.target = old_target
    return nothing
end
function unset_set_hc_rm_target!(args...)
    return nothing
end
function set_hc_rm_mu_w!(rm::RMMu, cluster)
    old_mu = rm.mu
    if !(isnothing(rm.mu) || isempty(rm.mu))
        rm.mu = view(rm.mu, cluster)
    end
    old_target = set_hc_rm_target!(rm, cluster)
    return old_mu, old_target
end
function set_hc_rm_mu_w!(args...)
    return nothing, nothing
end
function unset_set_hc_rm_mu_w!(rm::RMMu, old_mu, old_target)
    rm.mu = old_mu
    unset_set_hc_rm_target!(rm, old_target)
    return nothing
end
function unset_set_hc_rm_mu_w!(args...)
    return nothing
end
function set_hc_rm_sigma!(rm::RMSigma, sigma, cluster)
    old_sigma = rm.sigma
    if isnothing(rm.sigma) || isempty(rm.sigma)
        rm.sigma = view(sigma, cluster, cluster)
    else
        rm.sigma = view(old_sigma, cluster, cluster)
    end
    return old_sigma
end
function set_hc_rm_sigma!(args...)
    return nothing
end
function unset_hc_rm_sigma!(rm::RMSigma, old_sigma)
    rm.sigma = old_sigma
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
function set_tracking_rm!(args...)
    return nothing
end
function set_tracking_rm!(rm::TrackingRM, cluster)
    old_tr = rm.tr
    rm.tr = get_cluster_tracking(rm.tr, cluster)
    return old_tr
end
function unset_tracking_rm!(rm::TrackingRM, old_tr)
    rm.tr = old_tr
    return nothing
end
function unset_tracking_rm!(args...)
    return nothing
end
function set_turnover_rm!(args...)
    return nothing
end
function set_turnover_rm!(rm::TurnoverRM, cluster)
    old_to = rm.tr
    rm.tr = get_cluster_tr(rm.tr, cluster)
    return old_to
end
function unset_turnover_rm!(rm::TurnoverRM, old_to)
    rm.tr = old_to
    return nothing
end
function unset_turnover_rm!(args...)
    return nothing
end
function cluster_risk(port, sigma, returns, cluster, rm::Union{RiskMeasure, HCRiskMeasure})
    old_mu, old_target = set_hc_rm_mu_w!(rm, cluster)
    old_sigma = set_hc_rm_sigma!(rm, sigma, cluster)
    old_tr = set_tracking_rm!(rm, cluster)
    old_to = set_turnover_rm!(rm, cluster)
    cret = view(returns, :, cluster)
    clong_fees, crebalance = get_cluster_fees(port, cluster)
    old_V, old_skew = gen_cluster_skew_sskew(rm, port, cluster)
    cw = naive_risk(rm, cret, clong_fees, crebalance)
    crisk = calc_risk(rm, cw; X = cret, long_fees = clong_fees, rebalance = crebalance)
    unset_set_hc_rm_mu_w!(rm, old_mu, old_target)
    unset_hc_rm_sigma!(rm, old_sigma)
    unset_hc_rm_skew!(rm, old_V, old_skew)
    unset_tracking_rm!(rm, old_tr)
    unset_turnover_rm!(rm, old_to)
    return crisk
end
