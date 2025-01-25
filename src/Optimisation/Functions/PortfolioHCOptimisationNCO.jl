# Copywrite (c) 2025
# Author: Daniel Celis Garza <daniel.celis.garza@gmail.com>
# SPDX-License-Identifier: MIT

struct NCOSpecialRMIdx{T1, T2, T3, T4, T5, T6, T7}
    cov_idx::T1
    kurt_idx::T2
    skurt_idx::T3
    skew_idx::T4
    sskew_idx::T5
    wc_idx::T6
    trto_idx::T7
end
function find_special_rm(::Nothing)
    return NCOSpecialRMIdx(Vector{Int}(undef, 0), Vector{Int}(undef, 0),
                           Vector{Int}(undef, 0), Vector{Int}(undef, 0),
                           Vector{Int}(undef, 0), Vector{Int}(undef, 0),
                           Vector{Int}(undef, 0))
end
function find_special_rm(rm::Union{AbstractVector, <:Union{RiskMeasure, HCRiskMeasure}})
    cov_idx = Vector{Int}(undef, 0)
    kurt_idx = Vector{Int}(undef, 0)
    skurt_idx = Vector{Int}(undef, 0)
    skew_idx = Vector{Int}(undef, 0)
    sskew_idx = Vector{Int}(undef, 0)
    wc_idx = Vector{Int}(undef, 0)
    trto_idx = Vector{Int}(undef, 0)
    if !isa(rm, AbstractVector)
        if isa(rm, SD) || isa(rm, Variance)
            push!(cov_idx, 1)
        elseif isa(rm, Kurt)
            push!(kurt_idx, 1)
        elseif isa(rm, SKurt)
            push!(skurt_idx, 1)
        elseif isa(rm, Skew)
            push!(skew_idx, 1)
        elseif isa(rm, SSkew)
            push!(sskew_idx, 1)
        elseif isa(rm, WCVariance)
            push!(wc_idx, 1)
        elseif isa(rm, TurnoverRM) || isa(rm, TrackingRM) && isa(rm.tr, TrackWeight)
            push!(trto_idx, 1)
        end
    else
        rm_flat = reduce(vcat, rm)
        for (i, r) ∈ enumerate(rm_flat) #! Do not change this enumerate to pairs.
            if isa(r, SD) || isa(r, Variance)
                push!(cov_idx, i)
            elseif isa(r, Kurt)
                push!(kurt_idx, i)
            elseif isa(r, SKurt)
                push!(skurt_idx, i)
            elseif isa(r, Skew)
                push!(skew_idx, i)
            elseif isa(r, SSkew)
                push!(sskew_idx, i)
            elseif isa(r, WCVariance)
                push!(wc_idx, i)
            elseif isa(r, TurnoverRM) || isa(r, TrackingRM) && isa(r.tr, TrackWeight)
                push!(trto_idx, i)
            end
        end
    end

    return NCOSpecialRMIdx(cov_idx, kurt_idx, skurt_idx, skew_idx, sskew_idx, wc_idx,
                           trto_idx)
end
function set_cov_rm!(rm, cov_idx, idx, old_covs)
    if !isa(rm, AbstractVector)
        if !(isnothing(rm.sigma) || isempty(rm.sigma))
            push!(old_covs, rm.sigma)
            rm.sigma = view(rm.sigma, idx, idx)
        end
    else
        rm_flat = reduce(vcat, rm)
        for r ∈ view(rm_flat, cov_idx)
            if !(isnothing(r.sigma) || isempty(r.sigma))
                push!(old_covs, r.sigma)
                r.sigma = view(r.sigma, idx, idx)
            end
        end
    end
    return nothing
end
function get_port_kt(::Val{true}, port, idx)
    return view(port.kurt, idx, idx)
end
function get_port_kt(::Val{false}, port, idx)
    return view(port.skurt, idx, idx)
end
function set_kt_rm!(val::Union{Val{true}, Val{false}}, rm, port, kt_idx, idx, old_kts)
    if !isa(rm, AbstractVector)
        if isnothing(rm.kt) || isempty(rm.kt)
            push!(old_kts, rm.kt)
            rm.kt = get_port_kt(val, port, idx)
        else
            push!(old_kts, rm.kt)
            rm.kt = view(rm.kt, idx, idx)#get_port_kt(val, port, idx)
        end
    else
        rm_flat = reduce(vcat, rm)
        for r ∈ view(rm_flat, kt_idx)
            if isnothing(r.kt) || isempty(r.kt)
                push!(old_kts, r.kt)
                r.kt = get_port_kt(val, port, idx)
            else
                push!(old_kts, r.kt)
                r.kt = view(r.kt, idx, idx)#get_port_kt(val, port, idx)
            end
        end
    end
    return nothing
end
function get_skew(::Skew, port, cluster, idx)
    return view(port.skew, cluster, idx)
end
function get_skew(::SSkew, port, cluster, idx)
    return view(port.sskew, cluster, idx)
end
function gen_cluster_skew_sskew(args...)
    return nothing, nothing
end
function gen_cluster_skew_sskew(rm::RMSkew, port, cluster, Nc = nothing, idx = nothing)
    old_skew = rm.skew
    old_V = rm.V
    if isnothing(idx)
        idx = Int[]
        N = size(port.returns, 2)
        Nc = length(cluster)
        sizehint!(idx, Nc^2)
        for c ∈ cluster
            append!(idx, (((c - 1) * N + 1):(c * N))[cluster])
        end
    end
    skew = if isnothing(rm.skew) || isempty(rm.skew)
        get_skew(rm, port, cluster, idx)
    else
        view(rm.skew, cluster, idx)
    end
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
    rm.V = V
    return old_V, old_skew
end
function set_skew_rm!(rm, port, skew_idx, cluster, Nc, idx, old_Vs, old_skews)
    if !isa(rm, AbstractVector)
        old_V, old_skew = gen_cluster_skew_sskew(rm, port, cluster, Nc, idx)
        push!(old_Vs, old_V)
        push!(old_skews, old_skew)
    else
        rm_flat = reduce(vcat, rm)
        for r ∈ view(rm_flat, skew_idx)
            old_V, old_skew = gen_cluster_skew_sskew(r, port, cluster, Nc, idx)
            push!(old_Vs, old_V)
            push!(old_skews, old_skew)
        end
    end
    return nothing
end
function set_wc_var_rm!(rm, port, wc_idx, cidx, idx, old_wc_rms)
    if !isa(rm, AbstractVector)
        push!(old_wc_rms, deepcopy(rm))
        if isnothing(rm.sigma) || isempty(rm.sigma)
            sigma = !isempty(port.cov) ? view(port.cov, cidx, cidx) : nothing
        else
            sigma = !isempty(rm.sigma) ? view(rm.sigma, cidx, cidx) : nothing
        end
        if isnothing(rm.cov_l) ||
           isempty(rm.cov_l) ||
           isnothing(rm.cov_u) ||
           isempty(rm.cov_u)
            cov_l = !isempty(port.cov_l) ? view(port.cov_l, cidx, cidx) : nothing
            cov_u = !isempty(port.cov_u) ? view(port.cov_u, cidx, cidx) : nothing
        else
            cov_l = !isempty(rm.cov_l) ? view(rm.cov_l, cidx, cidx) : nothing
            cov_u = !isempty(rm.cov_u) ? view(rm.cov_u, cidx, cidx) : nothing
        end
        if isnothing(rm.cov_sigma) || isempty(rm.cov_sigma)
            cov_sigma = !isempty(port.cov_sigma) ? view(port.cov_sigma, idx, idx) : nothing
        else
            cov_sigma = !isempty(rm.cov_sigma) ? view(rm.cov_sigma, idx, idx) : nothing
        end
        rm.sigma = sigma
        rm.cov_l = cov_l
        rm.cov_u = cov_u
        rm.cov_sigma = cov_sigma
    else
        rm_flat = reduce(vcat, rm)
        for r ∈ view(rm_flat, wc_idx)
            push!(old_wc_rms, deepcopy(rm))
            if isnothing(r.sigma) || isempty(r.sigma)
                sigma = !isempty(port.sigma) ? view(port.sigma, cidx, cidx) : nothing
            else
                sigma = !isempty(r.sigma) ? view(r.sigma, cidx, cidx) : nothing
            end
            if isnothing(r.cov_l) ||
               isempty(r.cov_l) ||
               isnothing(r.cov_u) ||
               isempty(r.cov_u)
                cov_l = !isempty(port.cov_l) ? view(port.cov_l, cidx, cidx) : nothing
                cov_u = !isempty(port.cov_u) ? view(port.cov_u, cidx, cidx) : nothing
            else
                cov_l = !isempty(r.cov_l) ? view(r.cov_l, cidx, cidx) : nothing
                cov_u = !isempty(r.cov_u) ? view(r.cov_u, cidx, cidx) : nothing
            end
            if isnothing(r.cov_sigma) || isempty(r.cov_sigma)
                cov_sigma = if !isempty(port.cov_sigma)
                    view(port.cov_sigma, idx, idx)
                else
                    nothing
                end
                k_sigma = port.k_sigma
            else
                cov_sigma = !isempty(r.cov_sigma) ? view(r.cov_sigma, idx, idx) : nothing
                k_sigma = r.k_sigma
            end
            r.sigma = sigma
            r.cov_l = cov_l
            r.cov_u = cov_u
            r.cov_sigma = cov_sigma
            r.k_sigma = k_sigma
        end
    end
    return nothing
end
function set_trto_rm!(rm, trto_idx, idx, old_trtos)
    if !isa(rm, AbstractVector)
        push!(old_trtos, rm.tr.w)
        rm.tr.w = view(rm.tr.w, idx)
    else
        rm_flat = reduce(vcat, rm)
        for r ∈ view(rm_flat, trto_idx)
            push!(old_trtos, r.tr.w)
            r.tr.w = view(r.tr.w, idx)
        end
    end
    return nothing
end
struct NCOClusterStats{T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
                       T16}
    cassets::T1
    cret::T2
    cmu::T3
    ccov::T4
    ccor::T5
    cdist::T6
    ccov_mu::T7
    cd_mu::T8
end
struct NCOOldStats{T1, T2, T3, T4, T5, T6, T7, T8, T9}
    old_covs::T1
    old_kurts::T2
    old_skurts::T3
    old_Vs::T4
    old_skews::T5
    old_SVs::T6
    old_sskews::T7
    old_wc_rms::T8
    old_trtos::T9
end
function reset_kt_rm!(rm, kt_idx, old_kts)
    if !isempty(kt_idx) && !isempty(old_kts)
        if !isa(rm, AbstractVector)
            rm.kt = old_kts[1]
        else
            rm_flat = reduce(vcat, rm)
            for (r, old_kt) ∈ zip(view(rm_flat, kt_idx), old_kts)
                r.kt = old_kt
            end
        end
    end
    return nothing
end
function reset_cov_rm!(rm, cov_idx, old_covs)
    if !isempty(cov_idx) && !isempty(old_covs)
        if !isa(rm, AbstractVector)
            rm.sigma = old_covs[1]
        else
            rm_flat = reduce(vcat, rm)
            for (r, old_cov) ∈ zip(view(rm_flat, cov_idx), old_covs)
                r.sigma = old_cov
            end
        end
    end
    return nothing
end
function reset_skew_rm!(rm, skew_idx, old_Vs, old_skews)
    if !isempty(skew_idx) && !isempty(old_Vs)
        if !isa(rm, AbstractVector)
            rm.skew = old_skews[1]
            rm.V = old_Vs[1]
        else
            rm_flat = reduce(vcat, rm)
            for (r, old_V, old_skew) ∈ zip(view(rm_flat, skew_idx), old_Vs, old_skews)
                r.skew = old_skew
                r.V = old_V
            end
        end
    end
end
function reset_wc_var_rm!(rm, wc_idx, old_wc_rms)
    if !isempty(wc_idx) && !isempty(old_wc_rms)
        if !isa(rm, AbstractVector)
            old_wc = old_wc_rms[1]
            rm.sigma = old_wc.sigma
            rm.cov_l = old_wc.cov_l
            rm.cov_u = old_wc.cov_u
            rm.cov_sigma = old_wc.cov_sigma
        else
            rm_flat = reduce(vcat, rm)
            for (r, old_wc) ∈ zip(view(rm_flat, wc_idx), old_wc_rms)
                r.sigma = old_wc.sigma
                r.cov_l = old_wc.cov_l
                r.cov_u = old_wc.cov_u
                r.cov_sigma = old_wc.cov_sigma
            end
        end
    end
    return nothing
end
function reset_trto_rm!(rm, trto_idx, old_trtos)
    if !isempty(trto_idx) && !isempty(old_trtos)
        if !isa(rm, AbstractVector)
            rm.tr.w = old_trtos[1]
        else
            rm_flat = reduce(vcat, rm)
            for (r, old_trto) ∈ zip(view(rm_flat, trto_idx), old_trtos)
                r.tr.w = old_trto
            end
        end
    end
    return nothing
end
function reset_special_rms!(::Nothing, args...)
    return nothing
end
function reset_special_rms!(rm::Union{AbstractVector, AbstractRiskMeasure},
                            special_rm_idx::NCOSpecialRMIdx, old_stats::NCOOldStats)
    (; cov_idx, kurt_idx, skurt_idx, skew_idx, sskew_idx, wc_idx, trto_idx) = special_rm_idx
    (; old_covs, old_kurts, old_skurts, old_Vs, old_skews, old_SVs, old_sskews, old_wc_rms, old_trtos) = old_stats

    reset_cov_rm!(rm, cov_idx, old_covs)
    reset_kt_rm!(rm, kurt_idx, old_kurts)
    reset_kt_rm!(rm, skurt_idx, old_skurts)
    reset_skew_rm!(rm, skew_idx, old_Vs, old_skews)
    reset_skew_rm!(rm, sskew_idx, old_SVs, old_sskews)
    reset_wc_var_rm!(rm, wc_idx, old_wc_rms)
    reset_trto_rm!(rm, trto_idx, old_trtos)
    return nothing
end
function get_cluster_matrix(x::AbstractMatrix, idx)
    return !isempty(x) ? x[idx, idx] : x
end
function get_cluster_vector(x::AbstractVector, idx)
    return !isempty(x) ? x[idx] : x
end
function get_cluster_real_or_vector(x::AbstractVector, idx)
    return !isempty(x) ? x[idx] : x
end
function get_cluster_real_or_vector(x::Real, ::Any)
    return x
end
function get_cluster_returns(x::AbstractMatrix, idx)
    return !isempty(x) ? x[:, idx] : x
end
function get_cluster_loadings(x::DataFrame, idx)
    return !isempty(x) ? x[idx, :] : x
end
function gen_cluster_skew(skew::AbstractMatrix, cluster, Nc, idx)
    if !isempty(skew)
        skew = skew[cluster, idx]
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
    else
        V = Matrix{eltype(skew)}(undef, 0, 0)
    end
    return skew, V
end
function get_cluster_tracking(x::Union{NoTracking, TrackRet}, ::Any)
    return x
end
function get_cluster_tracking(x::TrackWeight, idx)
    return TrackWeight(; err = x.err, w = get_cluster_vector(x.w, idx))
end
function get_cluster_tr(x::NoTR, ::Any)
    return x
end
function get_cluster_tr(x::TR, idx)
    return TR(; val = get_cluster_real_or_vector(x.val, idx),
              w = get_cluster_vector(x.w, idx))
end
function get_external_returns(x::AbstractMatrix, w)
    return !isempty(x) ? x * w : x
end
function get_external_vector(x::AbstractVector, w)
    return !isempty(x) ? transpose(w) * x : x
end
function get_external_matrix(x::AbstractMatrix, w)
    return !isempty(x) ? transpose(w) * x * w : x
end
function get_external_loadings(x::DataFrame, w)
    return DataFrame(!isempty(x) ? transpose(w) * Matrix(x) : x, names(x))
end
function get_external_real_or_vector(x::Real, w)
    return x
end
function get_external_real_or_vector(x::AbstractVector, w)
    return !isempty(x) ? transpose(w) * x : x
end
function get_external_cluster_tracking(x::Union{NoTracking, TrackRet}, ::Any)
    return x
end
function get_external_cluster_tracking(x::TrackWeight, w)
    return TrackWeight(; err = x.err, w = get_external_vector(x.w, w))
end
function get_external_tr(x::NoTR, ::Any)
    return x
end
function get_external_tr(x::TR, w)
    return TR(; val = get_external_real_or_vector(x.val, w),
              w = get_external_vector(x.w, w))
end
function pre_modify_intra_port!(::NoNCOModify, intra_port, internal_args, i, cluster, cidx,
                                idx_sq, Nc, special_rm_idx)
    return nothing
end
function post_modify_intra_port!(::NoNCOModify, intra_port, internal_args, i, cluster, cidx,
                                 idx_sq, Nc, special_rm_idx)
    return nothing
end
function get_cluster_portfolio(port, internal_args, i, cluster, cidx, idx_sq, Nc,
                               special_rm_idx)
    (; type, pre_modify, post_modify, port_kwargs, stats_kwargs, wc_kwargs, factor_kwargs, cluster_kwargs) = internal_args[i]

    kelly = hasproperty(type, :kelly) ? type.kelly : NoKelly()
    class = hasproperty(type, :class) ? type.class : Classic()

    (; assets, returns, f_assets, f_returns, loadings, regression_type, mu_l, mu, cov, cor, dist, k, max_num_assets_kurt, max_num_assets_kurt_scale, kurt, skurt, L_2, S_2, skew, V, sskew, SV, f_mu, f_cov, fm_returns, fm_mu, fm_cov, bl_bench_weights, bl_mu, bl_cov, blfm_mu, blfm_cov, cov_l, cov_u, cov_mu, cov_sigma, d_mu, k_mu, k_sigma, w_min, w_max, risk_budget, f_risk_budget, short, long_l, long_ub, short_l, short_lb, budget_lb, budget, budget_ub, short_budget_ub, short_budget, short_budget_lb, card_scale, card, nea, tracking, turnover, l1, l2, long_fees, short_fees, rebalance, solvers) = port

    (; cov_idx, kurt_idx, skurt_idx, skew_idx, sskew_idx, wc_idx) = special_rm_idx
    wc_flag = !isempty(wc_idx)
    cov_flag = !isempty(cov_idx) || !isempty(cov) || isa(kelly, AKelly) || wc_flag
    kurt_flag = !isempty(kurt_idx)
    skurt_flag = !isempty(skurt_idx)
    skew_flag = !isempty(skew_idx)
    sskew_flag = !isempty(sskew_idx)
    factor_flag = isa(class, Union{FM, FC})
    bl_flag = isa(class, Union{BL, BLFM})
    blfm_flag = isa(class, BLFM)
    hc_flag = isa(type, HCOptimType)
    rp_flag = isa(type, Union{RB, RRB})
    cvx_flag = isa(type, OptimType)

    assets = get_cluster_vector(assets, cidx)
    returns = get_cluster_returns(returns, cidx)

    mu = get_cluster_vector(mu, cidx)

    cov = cov_flag ? get_cluster_matrix(cov, cidx) : Matrix{eltype(returns)}(undef, 0, 0)

    if isa(type, HCOptimType)
        cor = get_cluster_matrix(cor, cidx)
        dist = get_cluster_matrix(dist, cidx)
    else
        cor = Matrix{eltype(returns)}(undef, 0, 0)
        dist = Matrix{eltype(returns)}(undef, 0, 0)
    end

    if kurt_flag
        kurt = get_cluster_matrix(kurt, idx_sq)
    else
        kurt = Matrix{eltype(returns)}(undef, 0, 0)
    end

    if skurt_flag
        skurt = get_cluster_matrix(skurt, idx_sq)
    else
        skurt = Matrix{eltype(returns)}(undef, 0, 0)
    end

    if kurt_flag || skurt_flag
        L_2, S_2 = dup_elim_sum_matrices(size(returns, 2))[2:3]
    else
        L_2 = SparseMatrixCSC{Float64, Int}(undef, 0, 0)
        S_2 = SparseMatrixCSC{Float64, Int}(undef, 0, 0)
    end

    if skew_flag
        skew, V = gen_cluster_skew(skew, cluster, Nc, idx_sq)
    else
        skew = Matrix{eltype(returns)}(undef, 0, 0)
        V = Matrix{eltype(returns)}(undef, 0, 0)
    end

    if sskew_flag
        sskew, SV = gen_cluster_skew(sskew, cluster, Nc, idx_sq)
    else
        sskew = Matrix{eltype(returns)}(undef, 0, 0)
        SV = Matrix{eltype(returns)}(undef, 0, 0)
    end

    if factor_flag
        fm_returns = get_cluster_returns(fm_returns, cidx)
        fm_mu = get_cluster_vector(fm_mu, cidx)
        if cov_flag
            fm_cov = get_cluster_matrix(fm_cov, cidx)
        else
            fm_cov = Matrix{eltype(returns)}(undef, 0, 0)
        end
        loadings = get_cluster_loadings(loadings, cidx)
    else
        fm_returns = Matrix{eltype(returns)}(undef, 0, 0)
        fm_mu = Vector{eltype(returns)}(undef, 0)
        fm_cov = Matrix{eltype(returns)}(undef, 0, 0)
        loadings = DataFrame()
    end

    if bl_flag
        bl_bench_weights = get_cluster_vector(bl_bench_weights, cidx)
        bl_mu = get_cluster_vector(bl_mu, cidx)
        if cov_flag
            bl_cov = get_cluster_matrix(bl_cov, cidx)
        else
            bl_cov = Matrix{eltype(returns)}(undef, 0, 0)
        end

        if blfm_flag
            blfm_mu = get_cluster_vector(blfm_mu, cidx)
            if cov_flag
                blfm_cov = get_cluster_matrix(blfm_cov, cidx)
            else
                blfm_cov = Matrix{eltype(returns)}(undef, 0, 0)
            end
        else
            blfm_mu = Vector{eltype(returns)}(undef, 0)
            blfm_cov = Matrix{eltype(returns)}(undef, 0, 0)
        end
    else
        bl_bench_weights = Vector{eltype(returns)}(undef, 0)
        bl_mu = Vector{eltype(returns)}(undef, 0)
        bl_cov = Matrix{eltype(returns)}(undef, 0, 0)
        blfm_mu = Vector{eltype(returns)}(undef, 0)
        blfm_cov = Matrix{eltype(returns)}(undef, 0, 0)
    end

    if wc_flag
        cov_l = get_cluster_matrix(cov_l, cidx)
        cov_u = get_cluster_matrix(cov_u, cidx)
        cov_mu = get_cluster_matrix(cov_mu, cidx)
        cov_sigma = get_cluster_matrix(cov_sigma, idx_sq)
        d_mu = get_cluster_vector(d_mu, cidx)
    else
        cov_l = Matrix{eltype(returns)}(undef, 0, 0)
        cov_u = Matrix{eltype(returns)}(undef, 0, 0)
        cov_mu = Matrix{eltype(returns)}(undef, 0, 0)
        cov_sigma = Matrix{eltype(returns)}(undef, 0, 0)
        d_mu = Vector{eltype(returns)}(undef, 0)
    end

    if hc_flag
        w_min = get_cluster_real_or_vector(w_min, cidx)
        w_max = get_cluster_real_or_vector(w_max, cidx)
    else
        w_min = 0.0
        w_max = 1.0
    end

    if rp_flag
        risk_budget = get_cluster_vector(risk_budget, cidx)
    end

    if cvx_flag
        long_l = get_cluster_real_or_vector(long_l, cidx)
        long_ub = get_cluster_real_or_vector(long_ub, cidx)
        short_l = get_cluster_real_or_vector(short_l, cidx)
        short_lb = get_cluster_real_or_vector(short_lb, cidx)
        tracking = get_cluster_tracking(tracking, cidx)
        turnover = get_cluster_tr(turnover, cidx)
        long_fees = get_cluster_real_or_vector(long_fees, cidx)
        short_fees = get_cluster_real_or_vector(short_fees, cidx)
        rebalance = get_cluster_tr(rebalance, cidx)
    else
        long_l = 0.0
        long_ub = 1.0
        short_l = -0.0
        short_lb = -0.2
        tracking = NoTracking()
        turnover = NoTR()
        long_fees = 0.0
        short_fees = 0.0
        rebalance = NoTR()
    end

    intra_port = Portfolio(; assets = assets, ret = returns, f_assets = f_assets,
                           f_ret = f_returns, loadings = loadings,
                           regression_type = regression_type, mu_l = mu_l, mu = mu,
                           cov = cov, cor = cor, dist = dist, k = k,
                           max_num_assets_kurt = max_num_assets_kurt,
                           max_num_assets_kurt_scale = max_num_assets_kurt_scale,
                           kurt = kurt, skurt = skurt, L_2 = L_2, S_2 = S_2, skew = skew,
                           V = V, sskew = sskew, SV = SV, f_mu = f_mu, f_cov = f_cov,
                           fm_returns = fm_returns, fm_mu = fm_mu, fm_cov = fm_cov,
                           bl_bench_weights = bl_bench_weights, bl_mu = bl_mu,
                           bl_cov = bl_cov, blfm_mu = blfm_mu, blfm_cov = blfm_cov,
                           cov_l = cov_l, cov_u = cov_u, cov_mu = cov_mu,
                           cov_sigma = cov_sigma, d_mu = d_mu, k_mu = k_mu,
                           k_sigma = k_sigma, w_min = w_min, w_max = w_max,
                           risk_budget = risk_budget, f_risk_budget = f_risk_budget,
                           short = short, long_l = long_l, long_ub = long_ub,
                           short_l = short_l, short_lb = short_lb, budget_lb = budget_lb,
                           budget = budget, budget_ub = budget_ub,
                           short_budget_ub = short_budget_ub, short_budget = short_budget,
                           short_budget_lb = short_budget_lb, card_scale = card_scale,
                           card = card, nea = nea, tracking = tracking, turnover = turnover,
                           l1 = l1, l2 = l2, long_fees = long_fees, short_fees = short_fees,
                           rebalance = rebalance, solvers = solvers, port_kwargs...)

    pre_modify_intra_port!(pre_modify, intra_port, internal_args, i, cluster, cidx, idx_sq,
                           Nc, special_rm_idx)

    if !isempty(stats_kwargs)
        asset_statistics!(intra_port; stats_kwargs...)
    end
    if !isempty(factor_kwargs) && factor_flag
        factor_statistics!(intra_port; factor_kwargs...)
    end
    if !isempty(wc_kwargs) && wc_flag
        wc_statistics!(intra_port; wc_kwargs...)
    end
    if hc_flag
        cluster_assets!(intra_port; cluster_kwargs...)
    end

    post_modify_intra_port!(post_modify, intra_port, internal_args, i, cluster, cidx,
                            idx_sq, Nc, special_rm_idx)

    w = optimise!(intra_port, type)
    if !isempty(w)
        w = w.weights
    else
        w = zeros(eltype(returns), size(returns, 2))
    end

    return w, intra_port.fail
end
function set_rm_stats!(::Portfolio, ::Nothing, args...)
    return nothing
end
function set_rm_stats!(port::Portfolio, rm, cluster, cidx, idx_sq, Nc, special_rm_idx)
    (; cov_idx, kurt_idx, skurt_idx, skew_idx, sskew_idx, wc_idx, trto_idx) = special_rm_idx

    cov_flag = !isempty(cov_idx)
    kurt_flag = !isempty(kurt_idx)
    skurt_flag = !isempty(skurt_idx)
    skew_flag = !isempty(skew_idx)
    sskew_flag = !isempty(sskew_idx)
    wc_flag = !isempty(wc_idx)
    trto_flag = !isempty(trto_idx)

    old_covs = Vector{Union{Matrix{eltype(port.returns)}, Nothing}}(undef, 0)
    old_kurts = Vector{Union{Matrix{eltype(port.returns)}, Nothing}}(undef, 0)
    old_skurts = Vector{Union{Matrix{eltype(port.returns)}, Nothing}}(undef, 0)
    old_Vs = Vector{Union{Matrix{eltype(port.returns)}, Nothing}}(undef, 0)
    old_SVs = Vector{Union{Matrix{eltype(port.returns)}, Nothing}}(undef, 0)
    old_skews = Vector{Union{Matrix{eltype(port.returns)}, Nothing}}(undef, 0)
    old_sskews = Vector{Union{Matrix{eltype(port.returns)}, Nothing}}(undef, 0)
    old_wc_rms = Vector{Union{WCVariance}}(undef, 0)
    old_trtos = Vector{Union{Vector{eltype(port.returns)}, Nothing}}(undef, 0)

    if cov_flag
        set_cov_rm!(rm, cov_idx, cidx, old_covs)
    end
    if kurt_flag
        set_kt_rm!(Val(true), rm, port, kurt_idx, idx_sq, old_kurts)
    end
    if skurt_flag
        set_kt_rm!(Val(false), rm, port, skurt_idx, idx_sq, old_skurts)
    end
    if skew_flag
        set_skew_rm!(rm, port, skew_idx, cluster, Nc, idx_sq, old_Vs, old_skews)
    end
    if sskew_flag
        set_skew_rm!(rm, port, sskew_idx, cluster, Nc, idx_sq, old_SVs, old_sskews)
    end
    if wc_flag
        set_wc_var_rm!(rm, port, wc_idx, cidx, idx_sq, old_wc_rms)
    end
    if trto_flag
        set_trto_rm!(rm, trto_idx, cidx, old_trtos)
    end
    return NCOOldStats(old_covs, old_kurts, old_skurts, old_Vs, old_skews, old_SVs,
                       old_sskews, old_wc_rms, old_trtos)
end
function calc_intra_weights(port::Portfolio, internal_args)
    k = port.k
    idx = cutree(port.clusters; k = k)
    w = zeros(eltype(port.returns), size(port.returns, 2), k)
    cfails = Dict{Int, Dict}()

    flag = isa(internal_args, AbstractVector)
    if !flag
        rm = internal_args.type.rm
        special_rm_idx = find_special_rm(rm)
        (; kurt_idx, skurt_idx, skew_idx, sskew_idx, wc_idx) = special_rm_idx
        kurt_flag = !isempty(kurt_idx)
        skurt_flag = !isempty(skurt_idx)
        skew_flag = !isempty(skew_idx)
        sskew_flag = !isempty(sskew_idx)
        wc_flag = !isempty(wc_idx)
    end

    N = size(port.returns, 2)

    for i ∈ 1:k
        if flag
            rm = internal_args[i].type.rm
            special_rm_idx = find_special_rm(rm)
            (; kurt_idx, skurt_idx, skew_idx, sskew_idx, wc_idx) = special_rm_idx
            kurt_flag = !isempty(kurt_idx)
            skurt_flag = !isempty(skurt_idx)
            skew_flag = !isempty(skew_idx)
            sskew_flag = !isempty(sskew_idx)
            wc_flag = !isempty(wc_idx)
        end

        cidx = idx .== i

        if any((kurt_flag, skurt_flag, skew_flag, sskew_flag, wc_flag))
            idx_sq = Int[]
            cluster = findall(cidx)
            Nc = length(cluster)
            sizehint!(idx_sq, Nc^2)
            for c ∈ cluster
                append!(idx_sq, (((c - 1) * N + 1):(c * N))[cluster])
            end
        else
            cluster = nothing
            idx_sq = nothing
            Nc = nothing
        end

        old_stats = set_rm_stats!(port, rm, cluster, cidx, idx_sq, Nc, special_rm_idx)
        cw, cfail = get_cluster_portfolio(port, internal_args, i, cluster, cidx, idx_sq, Nc,
                                          special_rm_idx)
        reset_special_rms!(rm, special_rm_idx, old_stats)

        w[cidx, i] .= cw
        if !isempty(cfail)
            cfails[i] = cfail
        end
    end
    if !isempty(cfails)
        port.fail[:intra] = cfails
    end
    return w
end
function compute_cov_rm!(rm, cov_idx, wi, old_covs)
    if !isa(rm, AbstractVector)
        if !(isnothing(rm.sigma) || isempty(rm.sigma))
            push!(old_covs, rm.sigma)
            rm.sigma = transpose(wi) * rm.sigma * wi
        end
    else
        rm_flat = reduce(vcat, rm)
        for r ∈ view(rm_flat, cov_idx)
            if !(isnothing(r.sigma) || isempty(r.sigma))
                push!(old_covs, r.sigma)
                r.sigma = transpose(wi) * r.sigma * wi
            end
        end
    end
    return nothing
end
function set_kt_rm_nothing!(rm, kt_idx, old_kts)
    if !isa(rm, AbstractVector)
        if !(isnothing(rm.kt) || isempty(rm.kt))
            push!(old_kts, rm.kt)
            rm.kt = nothing
        end
    else
        rm_flat = reduce(vcat, rm)
        for r ∈ view(rm_flat, kt_idx)
            if !(isnothing(r.kt) || isempty(r.kt))
                push!(old_kts, r.kt)
                r.kt = nothing
            end
        end
    end
    return nothing
end
function set_skew_rm_nothing!(rm, skew_idx, old_Vs, old_skews)
    if !isa(rm, AbstractVector)
        if !(isnothing(rm.V) || isempty(rm.V))
            push!(old_skews, rm.skew)
            push!(old_Vs, rm.V)
            rm.skew = nothing
            rm.V = nothing
        end
    else
        rm_flat = reduce(vcat, rm)
        for r ∈ view(rm_flat, skew_idx)
            if !(isnothing(r.V) || isempty(r.V))
                push!(old_skews, r.skew)
                push!(old_Vs, r.V)
                r.skew = nothing
                r.V = nothing
            end
        end
    end
end
function set_wc_var_rm_nothing!(rm, wc_idx, old_wc_rms)
    if !isa(rm, AbstractVector)
        if !(isnothing(rm.sigma) ||
             isempty(rm.sigma) ||
             isnothing(rm.cov_l) ||
             isempty(rm.cov_l) ||
             isnothing(rm.cov_u) ||
             isempty(rm.cov_u) ||
             isnothing(rm.cov_mu) ||
             isempty(rm.cov_mu) ||
             isnothing(rm.cov_sigma) ||
             isempty(rm.cov_sigma))
            push!(old_wc_rms, rm)
            rm.sigma = nothing
            rm.cov_l = nothing
            rm.cov_u = nothing
            rm.cov_mu = nothing
            rm.cov_sigma = nothing
        end
    else
        rm_flat = reduce(vcat, rm)
        for r ∈ view(rm_flat, wc_idx)
            if !(isnothing(r.sigma) ||
                 isempty(r.sigma) ||
                 isnothing(r.cov_l) ||
                 isempty(r.cov_l) ||
                 isnothing(r.cov_u) ||
                 isempty(r.cov_u) ||
                 isnothing(r.cov_mu) ||
                 isempty(r.cov_mu) ||
                 isnothing(r.cov_sigma) ||
                 isempty(r.cov_sigma))
                push!(old_wc_rms, r)
                r.sigma = nothing
                r.cov_l = nothing
                r.cov_u = nothing
                r.cov_mu = nothing
                r.cov_sigma = nothing
            end
        end
    end
    return nothing
end
function compute_trto_rm!(rm, trto_idx, wi, old_trtos)
    if !isa(rm, AbstractVector)
        push!(old_trtos, rm.tr.w)
        rm.tr.w = transpose(wi) * rm.tr.w
    else
        rm_flat = reduce(vcat, rm)
        for r ∈ view(rm_flat, trto_idx)
            push!(old_trtos, r.tr.w)
            r.tr.w = transpose(wi) * r.tr.w
        end
    end
    return nothing
end
function set_rm_stats!(port, rm, wi, special_rm_idx)
    (; cov_idx, kurt_idx, skurt_idx, skew_idx, sskew_idx, wc_idx, trto_idx) = special_rm_idx

    cov_flag = !isempty(cov_idx)
    kurt_flag = !isempty(kurt_idx)
    skurt_flag = !isempty(skurt_idx)
    skew_flag = !isempty(skew_idx)
    sskew_flag = !isempty(sskew_idx)
    wc_flag = !isempty(wc_idx)
    trto_flag = !isempty(trto_idx)

    old_covs = Vector{Union{Matrix{eltype(port.returns)}, Nothing}}(undef, 0)
    old_kurts = Vector{Union{Matrix{eltype(port.returns)}, Nothing}}(undef, 0)
    old_skurts = Vector{Union{Matrix{eltype(port.returns)}, Nothing}}(undef, 0)
    old_Vs = Vector{Union{Matrix{eltype(port.returns)}, Nothing}}(undef, 0)
    old_SVs = Vector{Union{Matrix{eltype(port.returns)}, Nothing}}(undef, 0)
    old_skews = Vector{Union{Matrix{eltype(port.returns)}, Nothing}}(undef, 0)
    old_sskews = Vector{Union{Matrix{eltype(port.returns)}, Nothing}}(undef, 0)
    old_wc_rms = Vector{Union{WCVariance}}(undef, 0)
    old_trtos = Vector{Union{Vector{eltype(port.returns)}, Nothing}}(undef, 0)

    if cov_flag
        compute_cov_rm!(rm, cov_idx, wi, old_covs)
    end
    if kurt_flag
        set_kt_rm_nothing!(rm, kurt_idx, old_kurts)
    end
    if skurt_flag
        set_kt_rm_nothing!(rm, skurt_idx, old_skurts)
    end
    if skew_flag
        set_skew_rm_nothing!(rm, skew_idx, old_Vs, old_skews)
    end
    if sskew_flag
        set_skew_rm_nothing!(rm, sskew_idx, old_SVs, old_sskews)
    end
    if wc_flag
        set_wc_var_rm_nothing!(rm, wc_idx, old_wc_rms)
    end
    if trto_flag
        compute_trto_rm!(rm, trto_idx, wi, old_trtos)
    end
    return NCOOldStats(old_covs, old_kurts, old_skurts, old_Vs, old_skews, old_SVs,
                       old_sskews, old_wc_rms, old_trtos)
end
function pre_modify_inter_port!(::NoNCOModify, inter_port, wi, external_args,
                                special_rm_idx)
    return nothing
end
function post_modify_inter_port!(::NoNCOModify, inter_port, wi, external_args,
                                 special_rm_idx)
    return nothing
end
function get_external_portfolio(port, wi, external_args, special_rm_idx)
    (; type, pre_modify, post_modify, port_kwargs, stats_kwargs, wc_kwargs, factor_kwargs, cluster_kwargs) = external_args

    kelly = hasproperty(type, :kelly) ? type.kelly : NoKelly()
    class = hasproperty(type, :class) ? type.class : Classic()

    (; assets, returns, f_assets, f_returns, loadings, regression_type, mu_l, mu, cov, k, max_num_assets_kurt, max_num_assets_kurt_scale, f_mu, f_cov, fm_returns, fm_mu, fm_cov, bl_bench_weights, bl_mu, bl_cov, blfm_mu, blfm_cov, w_min, w_max, risk_budget, f_risk_budget, short, long_l, long_ub, short_l, short_lb, budget_lb, budget, budget_ub, short_budget_ub, short_budget, short_budget_lb, card_scale, card, nea, tracking, turnover, l1, l2, long_fees, short_fees, rebalance, solvers) = port

    (; cov_idx, kurt_idx, skurt_idx, skew_idx, sskew_idx, wc_idx) = special_rm_idx
    wc_flag = !isempty(wc_idx)
    cov_flag = !isempty(cov_idx) || !isempty(cov) || isa(kelly, AKelly) || wc_flag
    kurt_flag = !isempty(kurt_idx)
    skurt_flag = !isempty(skurt_idx)
    skew_flag = !isempty(skew_idx)
    sskew_flag = !isempty(sskew_idx)
    factor_flag = isa(class, Union{FM, FC})
    bl_flag = isa(class, Union{BL, BLFM})
    blfm_flag = isa(class, BLFM)
    hc_flag = isa(type, HCOptimType)
    rp_flag = isa(type, Union{RB, RRB})
    cvx_flag = isa(type, OptimType)

    assets = 1:size(wi, 2)
    returns = get_external_returns(returns, wi)
    mu = get_external_vector(mu, wi)
    cov = cov_flag ? get_external_matrix(cov, wi) : Matrix{eltype(returns)}(undef, 0, 0)

    if factor_flag
        fm_returns = get_external_returns(fm_returns, wi)
        fm_mu = get_external_vector(fm_mu, wi)
        if cov_flag
            fm_cov = get_external_matrix(fm_cov, wi)
        else
            fm_cov = Matrix{eltype(returns)}(undef, 0, 0)
        end
        loadings = get_external_loadings(loadings, wi)
    else
        fm_returns = Matrix{eltype(returns)}(undef, 0, 0)
        fm_mu = Vector{eltype(returns)}(undef, 0)
        fm_cov = Matrix{eltype(returns)}(undef, 0, 0)
        loadings = DataFrame()
    end

    if bl_flag
        bl_bench_weights = get_external_vector(bl_bench_weights, wi)
        bl_mu = get_external_vector(bl_mu, wi)
        if cov_flag
            bl_cov = get_external_matrix(bl_cov, wi)
        else
            bl_cov = Matrix{eltype(returns)}(undef, 0, 0)
        end

        if blfm_flag
            blfm_mu = get_external_vector(blfm_mu, wi)
            if cov_flag
                blfm_cov = get_external_matrix(blfm_cov, wi)
            else
                blfm_cov = Matrix{eltype(returns)}(undef, 0, 0)
            end
        else
            blfm_mu = Vector{eltype(returns)}(undef, 0)
            blfm_cov = Matrix{eltype(returns)}(undef, 0, 0)
        end
    else
        bl_bench_weights = Vector{eltype(returns)}(undef, 0)
        bl_mu = Vector{eltype(returns)}(undef, 0)
        bl_cov = Matrix{eltype(returns)}(undef, 0, 0)
        blfm_mu = Vector{eltype(returns)}(undef, 0)
        blfm_cov = Matrix{eltype(returns)}(undef, 0, 0)
    end

    if hc_flag
        w_min = get_external_real_or_vector(w_min, wi)
        w_max = get_external_real_or_vector(w_max, wi)
    else
        w_min = 0.0
        w_max = 1.0
    end

    if rp_flag
        risk_budget = get_external_vector(risk_budget, wi)
    end

    if cvx_flag
        long_l = get_external_real_or_vector(long_l, wi)
        long_ub = get_external_real_or_vector(long_ub, wi)
        short_l = get_external_real_or_vector(short_l, wi)
        short_lb = get_external_real_or_vector(short_lb, wi)
        tracking = get_external_cluster_tracking(tracking, wi)
        turnover = get_external_tr(turnover, wi)
        long_fees = get_external_real_or_vector(long_fees, wi)
        short_fees = get_external_real_or_vector(short_fees, wi)
        rebalance = get_external_tr(rebalance, wi)
    else
        long_l = 0.0
        long_ub = 1.0
        short_l = -0.0
        short_lb = -0.2
        tracking = NoTracking()
        turnover = NoTR()
        long_fees = 0.0
        short_fees = 0.0
        rebalance = NoTR()
    end

    inter_port = Portfolio(; assets = assets, ret = returns, f_assets = f_assets,
                           f_ret = f_returns, loadings = loadings,
                           regression_type = regression_type, mu_l = mu_l, mu = mu,
                           cov = cov, k = k, max_num_assets_kurt = max_num_assets_kurt,
                           max_num_assets_kurt_scale = max_num_assets_kurt_scale,
                           f_mu = f_mu, f_cov = f_cov, fm_returns = fm_returns,
                           fm_mu = fm_mu, fm_cov = fm_cov,
                           bl_bench_weights = bl_bench_weights, bl_mu = bl_mu,
                           bl_cov = bl_cov, blfm_mu = blfm_mu, blfm_cov = blfm_cov,
                           w_min = w_min, w_max = w_max, risk_budget = risk_budget,
                           f_risk_budget = f_risk_budget, short = short, long_l = long_l,
                           long_ub = long_ub, short_l = short_l, short_lb = short_lb,
                           budget_lb = budget_lb, budget = budget, budget_ub = budget_ub,
                           short_budget_ub = short_budget_ub, short_budget = short_budget,
                           short_budget_lb = short_budget_lb, card_scale = card_scale,
                           card = card, nea = nea, tracking = tracking, turnover = turnover,
                           l1 = l1, l2 = l2, long_fees = long_fees, short_fees = short_fees,
                           rebalance = rebalance, solvers = solvers, port_kwargs...)

    pre_modify_inter_port!(pre_modify, inter_port, wi, external_args, special_rm_idx)

    asset_statistics!(inter_port; set_cov = false, set_mu = false, set_cor = hc_flag,
                      set_dist = hc_flag, set_kurt = kurt_flag, set_skurt = skurt_flag,
                      set_skew = skew_flag, set_sskew = sskew_flag, stats_kwargs...)
    if !isempty(factor_kwargs) && factor_flag
        factor_statistics!(inter_port; factor_kwargs...)
    end
    if wc_flag
        wc_statistics!(inter_port; wc_kwargs...)
    end
    if hc_flag
        cluster_assets!(inter_port; cluster_kwargs...)
    end

    post_modify_inter_port!(post_modify, inter_port, wi, external_args, special_rm_idx)

    w = optimise!(inter_port, type)
    if !isempty(w)
        w = w.weights
    else
        w = zeros(eltype(returns), size(returns, 2))
    end

    return w, inter_port.fail
end
function calc_inter_weights(port::Portfolio, wi, external_args)
    rm = external_args.type.rm
    special_rm_idx = find_special_rm(rm)
    old_stats = set_rm_stats!(port, rm, wi, special_rm_idx)
    cw, cfail = get_external_portfolio(port, wi, external_args, special_rm_idx)
    reset_special_rms!(rm, special_rm_idx, old_stats)

    w = wi * cw
    if !isempty(cfail)
        port.fail[:inter] = cfail
    end

    return w
end
function nco_optimise(port, type)
    wi = calc_intra_weights(port, type.internal)
    w = calc_inter_weights(port, wi, type.external)
    return w
end
