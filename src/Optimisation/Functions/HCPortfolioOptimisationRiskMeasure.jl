function _naive_risk(::Equal, returns, ::Any)
    N = size(returns, 2)
    return fill(eltype(returns)(inv(N)), N)
end
function _naive_risk(rm::AbstractRiskMeasure, returns, cV)
    N = size(returns, 2)
    inv_risk = Vector{eltype(returns)}(undef, N)
    w = Vector{eltype(returns)}(undef, N)

    for i ∈ eachindex(w)
        w .= zero(eltype(returns))
        w[i] = one(eltype(returns))
        risk = calc_risk(rm, w; X = returns, V = cV, SV = cV)
        inv_risk[i] = inv(risk)
    end
    return inv_risk / sum(inv_risk)
end
function _set_hc_rm_sigma(rm::RMSigma, port, cluster)
    sigma_old = rm.sigma
    if isnothing(rm.sigma) || isempty(rm.sigma)
        rm.sigma = view(port.cov, cluster, cluster)
    else
        rm.sigma = view(sigma_old, cluster, cluster)
    end
    return sigma_old
end
function _set_hc_rm_sigma(args...)
    return nothing
end
function _unset_hc_rm_sigma(rm::RMSigma, sigma_old)
    rm.sigma = sigma_old
    return nothing
end
function _unset_hc_rm_sigma(args...)
    return nothing
end
function cluster_risk(port, cluster, rm)
    sigma_old = _set_hc_rm_sigma(rm, port, cluster)
    cret = view(port.returns, :, cluster)
    cV = gen_cluster_skew_sskew(rm, port, cluster)
    cw = _naive_risk(rm, cret, cV)
    crisk = calc_risk(rm, cw; X = cret, V = cV, SV = cV)
    _unset_hc_rm_sigma(rm, sigma_old)
    return crisk
end
