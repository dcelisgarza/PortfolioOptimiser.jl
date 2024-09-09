function _get_skew(::Union{Skew, Val{true}}, port, cluster, idx)
    return view(port.skew, cluster, idx)
end
function _get_skew(::Union{SSkew, Val{false}}, port, cluster, idx)
    return view(port.sskew, cluster, idx)
end
function gen_cluster_skew_sskew(args...)
    return Matrix(undef, 0, 0)
end
function gen_cluster_skew_sskew(rm::Union{Skew, SSkew, Val{true}, Val{false}}, port,
                                cluster, Nc = nothing, idx = nothing)
    if isnothing(idx)
        idx = Int[]
        N = size(port.returns, 2)
        Nc = length(cluster)
        sizehint!(idx, Nc^2)
        for c ∈ cluster
            append!(idx, (((c - 1) * N + 1):(c * N))[cluster])
        end
    end
    skew = _get_skew(rm, port, cluster, idx)
    V = zeros(eltype(skew), Nc, Nc)
    for i ∈ 1:Nc
        j = (i - 1) * Nc + 1
        k = i * Nc
        vals, vecs = eigen(skew[:, j:k])
        vals = clamp.(real.(vals), -Inf, 0) .+ clamp.(imag.(vals), -Inf, 0)im
        V .-= real(vecs * Diagonal(vals) * transpose(vecs))
    end
    if all(iszero.(diag(V)))
        V .= V + eps(eltype(skew)) * I
    end
    return V
end
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
function naive_risk(port, cluster, rm)
    sigma_old = _set_hc_rm_sigma(rm, port, cluster)
    cret = view(port.returns, :, cluster)
    cV = gen_cluster_skew_sskew(rm, port, cluster)
    crisk = _naive_risk(rm, cret, cV)
    _unset_hc_rm_sigma(rm, sigma_old)

    return crisk
end
function find_kurt_skew_rm(rm::Union{AbstractVector, <:RiskMeasure})
    kurt_idx = Vector{Int}(undef, 0)
    skurt_idx = Vector{Int}(undef, 0)
    set_skew = false
    set_sskew = false
    if !isa(rm, AbstractVector)
        if isa(rm, Kurt)
            push!(kurt_idx, 1)
        end
        if isa(rm, SKurt)
            push!(skurt_idx, 1)
        end
        set_skew = isa(rm, Skew)
        set_sskew = isa(rm, SSkew)
    else
        rm_flat = reduce(vcat, rm)
        for (i, r) ∈ enumerate(rm_flat)
            if isa(r, Kurt)
                push!(kurt_idx, i)
            end
            if isa(r, SKurt)
                push!(skurt_idx, i)
            end
            if !set_skew
                set_skew = isa(r, Skew)
            end
            if !set_sskew
                set_sskew = isa(r, SSkew)
            end
        end
    end

    return kurt_idx, skurt_idx, set_skew, set_sskew
end
function _get_port_kt(::Val{true}, port, idx)
    return view(port.kurt, idx, idx)
end
function _get_port_kt(::Val{false}, port, idx)
    return view(port.skurt, idx, idx)
end
function _set_kt_rm(val::Union{Val{true}, Val{false}}, rm, port, kt_idx, idx, old_kts)
    if !isa(rm, AbstractVector)
        if isnothing(rm.kt) || isempty(rm.kt)
            push!(old_kts, rm.kt)
            rm.kt = _get_port_kt(val, port, idx)
        else
            kt_old = rm.kt
            rm.kt = _get_port_kt(val, port, idx)
            push!(old_kts, kt_old)
        end
    else
        rm_flat = reduce(vcat, rm)
        for r ∈ view(rm_flat, kt_idx)
            if isnothing(r.kt) || isempty(r.kt)
                push!(old_kts, r.kt)
                r.kt = _get_port_kt(val, port, idx)
            else
                kt_old = r.kt
                r.kt = _get_port_kt(val, port, idx)
                push!(old_kts, kt_old)
            end
        end
    end
end
function gen_cluster_stats(port, rm, cidx, kurt_idx, skurt_idx, set_skew, set_sskew)
    cassets = port.assets[cidx]
    cret = port.returns[:, cidx]
    cmu = port.mu[cidx]
    ccov = port.cov[cidx, cidx]
    old_kurts = Vector{Union{Matrix{eltype(port.returns)}, Nothing}}(undef, 0)
    old_skurts = Vector{Union{Matrix{eltype(port.returns)}, Nothing}}(undef, 0)
    cV = Matrix{eltype(port.returns)}(undef, 0, 0)
    cSV = Matrix{eltype(port.returns)}(undef, 0, 0)
    if !isempty(kurt_idx) || !isempty(skurt_idx) || set_skew || set_sskew
        idx = Int[]
        N = size(port.returns, 2)
        cluster = findall(cidx)
        Nc = length(cluster)
        sizehint!(idx, Nc^2)
        for c ∈ cluster
            append!(idx, (((c - 1) * N + 1):(c * N))[cluster])
        end
        if !isempty(kurt_idx)
            _set_kt_rm(Val(true), rm, port, kurt_idx, idx, old_kurts)
        end
        if !isempty(skurt_idx)
            _set_kt_rm(Val(false), rm, port, skurt_idx, idx, old_skurts)
        end
        if set_skew
            cV = gen_cluster_skew_sskew(Val(true), port, cluster, Nc, idx)
        end
        if set_sskew
            cSV = gen_cluster_skew_sskew(Val(false), port, cluster, Nc, idx)
        end
    end
    return cassets, cret, cmu, ccov, old_kurts, old_skurts, cV, cSV
end
function _reset_kt_rm(rm, kt_idx, old_kts)
    if !isempty(kt_idx)
        if !isa(rm, AbstractVector)
            rm.kt = old_kts[1]
        else
            rm_flat = reduce(vcat, rm)
            for (r, old_kt) ∈ zip(view(rm_flat, kt_idx), old_kts)
                r.kt = old_kt
            end
        end
    end
end
function reset_kurt_and_skurt_rm(rm, kurt_idx, old_kurts, skurt_idx, old_skurts)
    _reset_kt_rm(rm, kurt_idx, old_kurts)
    _reset_kt_rm(rm, skurt_idx, old_skurts)
    return nothing
end
