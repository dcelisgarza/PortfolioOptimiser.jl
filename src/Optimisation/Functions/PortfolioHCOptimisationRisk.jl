function naive_risk(::Equal, returns)
    N = size(returns, 2)
    return fill(eltype(returns)(inv(N)), N)
end
function naive_risk(rm::Union{RiskMeasure, HCRiskMeasure}, returns)
    N = size(returns, 2)
    inv_risk = Vector{eltype(returns)}(undef, N)
    w = Vector{eltype(returns)}(undef, N)

    for i ∈ eachindex(w)
        w .= zero(eltype(returns))
        w[i] = one(eltype(returns))
        risk = calc_risk(rm, w; X = returns)
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
function cluster_risk(port, sigma, returns, cluster, rm)
    sigma_old = set_hc_rm_sigma!(rm, sigma, cluster)
    cret = view(returns, :, cluster)
    old_V, old_skew = gen_cluster_skew_sskew(rm, port, cluster)
    cw = naive_risk(rm, cret)
    crisk = calc_risk(rm, cw; X = cret)
    unset_hc_rm_sigma!(rm, sigma_old)
    unset_hc_rm_skew!(rm, old_V, old_skew)
    return crisk
end
