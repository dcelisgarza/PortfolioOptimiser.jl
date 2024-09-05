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
                                cluster)
    idx = Int[]
    N = size(port.returns, 2)
    Nc = length(cluster)
    sizehint!(idx, Nc^2)
    for c ∈ cluster
        append!(idx, (((c - 1) * N + 1):(c * N))[cluster])
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

function _calc_hc_ret(::NoKelly, mu, w, args...)
    return dot(mu, w)
end
function _calc_hc_ret(::Any, ::Any, w, X::AbstractVector)
    return sum(log.(one(eltype(X)) .+ X * w)) / size(X, 1)
end
function hc_objective(::MinRisk, ::Any, ::Any, w, rm, returns, V)
    return calc_risk(rm, w; X = returns, V = V, SV = V)
end
function hc_objective(obj::Utility, kelly::RetType, mu, w, rm, returns, V)
    ret = _calc_hc_ret(kelly, mu, w, returns)
    risk = calc_risk(rm, w; X = returns, V = V, SV = V)
    utility = ret - obj.l * risk
    return utility <= zero(utility) ? risk : inv(utility)
end
function hc_objective(obj::Sharpe, kelly::RetType, mu, w, rm, returns, V)
    ret = _calc_hc_ret(kelly, mu, w, returns)
    risk = calc_risk(rm, w; X = returns, V = V, SV = V)
    inv_sharpe = risk / (ret - obj.rf)
    return inv_sharpe <= zero(inv_sharpe) ? risk : inv_sharpe
end
function hc_objective(::MaxRet, kelly::RetType, mu, w, rm, returns, V)
    ret = _calc_hc_ret(kelly, mu, w, returns)
    return ret <= zero(ret) ? calc_risk(rm, w; X = returns, V = V, SV = V) : inv(ret)
end

function _naive_risk(::Any, ::Any, ::Any, ::Equal, returns, ::Any)
    N = size(returns, 2)
    return fill(eltype(returns)(inv(N)), N)
end
function _naive_risk(obj, kelly, mu, rm::RiskMeasure, returns, V)
    N = size(returns, 2)
    inv_risk = Vector{eltype(returns)}(undef, N)
    w = Vector{eltype(returns)}(undef, N)

    for i ∈ eachindex(w)
        w .= zero(eltype(returns))
        w[i] = one(eltype(returns))
        # risk = calc_risk(rm, w; X = returns, V = V, SV = V)
        risk = hc_objective(obj, kelly, mu, w, rm, returns, V)
        inv_risk[i] = inv(risk)
    end
    return inv_risk / sum(inv_risk)
end
function cluster_risk(port, obj, kelly, cluster, rm)
    if hasproperty(rm, :sigma)
        rm.sigma = view(port.cov, cluster, cluster)
    end
    cret = view(port.returns, :, cluster)
    cmu = view(port.mu, cluster)
    cV = gen_cluster_skew_sskew(rm, port, cluster)
    cw = _naive_risk(obj, kelly, cmu, rm, cret, cV)
    # crisk = calc_risk(rm, cw; X = cret, V = cV, SV = cV)
    crisk = hc_objective(obj, kelly, cmu, cw, rm, cret, cV)
    if hasproperty(rm, :sigma)
        rm.sigma = nothing
    end
    return crisk
end
function naive_risk(port, obj, kelly, cluster, rm)
    if hasproperty(rm, :sigma)
        rm.sigma = view(port.cov, cluster, cluster)
    end
    cret = view(port.returns, :, cluster)
    cmu = view(port.mu, cluster)
    cV = gen_cluster_skew_sskew(rm, port, cluster)
    crisk = _naive_risk(obj, kelly, cmu, rm, cret, cV)
    if hasproperty(rm, :sigma)
        rm.sigma = nothing
    end
    return crisk
end
function find_kurt_skew_rm(rm::Union{AbstractVector, <:TradRiskMeasure})
    set_kurt = false
    set_skurt = false
    set_skew = false
    set_sskew = false
    if !isa(rm, AbstractVector)
        set_kurt = isa(rm, Kurt)
        set_skurt = isa(rm, SKurt)
        set_skew = isa(rm, Skew)
        set_sskew = isa(rm, SSkew)
    else
        rm_flat = reduce(vcat, rm)
        for r ∈ rm_flat
            if !set_kurt
                set_kurt = isa(r, Kurt)
            end
            if !set_skurt
                set_skurt = isa(r, SKurt)
            end
            if !set_skew
                set_skew = isa(r, Skew)
            end
            if !set_sskew
                set_sskew = isa(r, SSkew)
            end
        end
    end

    return set_kurt, set_skurt, set_skew, set_sskew
end
function gen_cluster_stats(port, cidx, set_kurt, set_skurt, set_skew, set_sskew)
    cassets = port.assets[cidx]
    cret = port.returns[:, cidx]
    cmu = port.mu[cidx]
    ccov = port.cov[cidx, cidx]
    ckurt = Matrix{eltype(port.returns)}(undef, 0, 0)
    cskurt = Matrix{eltype(port.returns)}(undef, 0, 0)
    cV = Matrix{eltype(port.returns)}(undef, 0, 0)
    cSV = Matrix{eltype(port.returns)}(undef, 0, 0)
    if set_kurt || set_skurt || set_skew || set_sskew
        idx = Int[]
        N = size(port.returns, 2)
        cluster = findall(cidx)
        Nc = length(cluster)
        sizehint!(idx, Nc^2)
        for c ∈ cluster
            append!(idx, (((c - 1) * N + 1):(c * N))[cluster])
        end
        if set_kurt
            ckurt = view(port.kurt, idx, idx)
        end
        if set_skurt
            cskurt = view(port.skurt, idx, idx)
        end
        if set_skew
            cV = gen_cluster_skew_sskew(Val(true), port, cluster)
        end
        if set_sskew
            cSV = gen_cluster_skew_sskew(Val(false), port, cluster)
        end
    end
    return cassets, cret, cmu, ccov, ckurt, cskurt, cV, cSV
end
