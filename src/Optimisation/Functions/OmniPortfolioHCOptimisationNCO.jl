struct NCOSpecialRMIdx{T1, T2, T3, T4, T5, T6}
    cov_idx::T1
    kurt_idx::T2
    skurt_idx::T3
    skew_idx::T4
    sskew_idx::T5
    wc_idx::T6
end
function find_special_rm(rm::Union{AbstractVector, <:AbstractRiskMeasure})
    cov_idx = Vector{Int}(undef, 0)
    kurt_idx = Vector{Int}(undef, 0)
    skurt_idx = Vector{Int}(undef, 0)
    skew_idx = Vector{Int}(undef, 0)
    sskew_idx = Vector{Int}(undef, 0)
    wc_idx = Vector{Int}(undef, 0)
    if !isa(rm, AbstractVector)
        if isa(rm, SD) || isa(rm, Variance)
            push!(cov_idx, 1)
        end
        if isa(rm, Kurt)
            push!(kurt_idx, 1)
        end
        if isa(rm, SKurt)
            push!(skurt_idx, 1)
        end
        if isa(rm, Skew)
            push!(skew_idx, 1)
        end
        if isa(rm, SSkew)
            push!(sskew_idx, 1)
        end
        if isa(rm, WCVariance)
            push!(wc_idx, 1)
        end
    else
        rm_flat = reduce(vcat, rm)
        for (i, r) ∈ enumerate(rm_flat) #! Do not change this enumerate to pairs.
            if isa(r, SD) || isa(r, Variance)
                push!(cov_idx, i)
            end
            if isa(r, Kurt)
                push!(kurt_idx, i)
            end
            if isa(r, SKurt)
                push!(skurt_idx, i)
            end
            if isa(r, Skew)
                push!(skew_idx, i)
            end
            if isa(r, SSkew)
                push!(sskew_idx, i)
            end
            if isa(r, WCVariance)
                push!(wc_idx, i)
            end
        end
    end

    return NCOSpecialRMIdx(cov_idx, kurt_idx, skurt_idx, skew_idx, sskew_idx, wc_idx)
end
function _set_cov_rm(rm, cov_idx, idx, old_covs)
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
            push!(old_kts, rm.kt)
            rm.kt = view(rm.kt, idx, idx)#_get_port_kt(val, port, idx)
        end
    else
        rm_flat = reduce(vcat, rm)
        for r ∈ view(rm_flat, kt_idx)
            if isnothing(r.kt) || isempty(r.kt)
                push!(old_kts, r.kt)
                r.kt = _get_port_kt(val, port, idx)
            else
                push!(old_kts, r.kt)
                r.kt = view(r.kt, idx, idx)#_get_port_kt(val, port, idx)
            end
        end
    end
    return nothing
end
function _get_skew(::Skew, port, cluster, idx)
    return view(port.skew, cluster, idx)
end
function _get_skew(::SSkew, port, cluster, idx)
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
        _get_skew(rm, port, cluster, idx)
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
function _set_skew_rm(rm, port, skew_idx, cluster, Nc, idx, old_Vs, old_skews)
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
function _set_wc_var_rm(rm, port, wc_idx, cidx, idx, old_wc_rms)
    if !isa(rm, AbstractVector)
        if isnothing(rm.sigma) || isempty(rm.sigma)
            sigma = !isempty(port.sigma) ? view(port.sigma, cidx, cidx) : nothing
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
        if isnothing(rm.cov_mu) ||
           isempty(rm.cov_mu) ||
           isnothing(rm.cov_sigma) ||
           isempty(rm.cov_sigma)
            cov_mu = !isempty(port.cov_mu) ? view(port.cov_mu, cidx, cidx) : nothing
            cov_sigma = !isempty(port.cov_sigma) ? view(port.cov_sigma, idx, idx) : nothing
        else
            cov_mu = !isempty(rm.cov_mu) ? view(rm.cov_mu, cidx, cidx) : nothing
            cov_sigma = !isempty(rm.cov_sigma) ? view(rm.cov_sigma, idx, idx) : nothing
        end
        push!(old_wc_rms, rm)
        rm.sigma = sigma
        rm.cov_l = cov_l
        rm.cov_u = cov_u
        rm.cov_mu = cov_mu
        rm.cov_sigma = cov_sigma
    else
        rm_flat = reduce(vcat, rm)
        for r ∈ view(rm_flat, wc_idx)
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
            if isnothing(r.cov_mu) ||
               isempty(r.cov_mu) ||
               isnothing(r.cov_sigma) ||
               isempty(r.cov_sigma)
                cov_mu = !isempty(port.cov_mu) ? view(port.cov_mu, cidx, cidx) : nothing
                cov_sigma = if !isempty(port.cov_sigma)
                    view(port.cov_sigma, idx, idx)
                else
                    nothing
                end
                k_sigma = port.k_sigma
            else
                cov_mu = !isempty(r.cov_mu) ? view(r.cov_mu, cidx, cidx) : nothing
                cov_sigma = !isempty(r.cov_sigma) ? view(r.cov_sigma, idx, idx) : nothing
                k_sigma = r.k_sigma
            end
            push!(old_wc_rms, rm)
            r.sigma = sigma
            r.cov_l = cov_l
            r.cov_u = cov_u
            r.cov_mu = cov_mu
            r.cov_sigma = cov_sigma
            r.k_sigma = k_sigma
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
struct NCOOldStats{T1, T2, T3, T4, T5, T6, T7, T8}
    old_covs::T1
    old_kurts::T2
    old_skurts::T3
    old_Vs::T4
    old_skews::T5
    old_SVs::T6
    old_sskews::T7
    old_wc_rms::T8
end
function _reset_kt_rm(rm, kt_idx, old_kts)
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
function _reset_cov_rm(rm, cov_idx, old_covs)
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
function _reset_skew_rm(rm, skew_idx, old_Vs, old_skews)
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
function _reset_wc_var_rm(rm, wc_idx, old_wc_rms)
    if !isempty(wc_idx) && !isempty(old_wc_rms)
        if !isa(rm, AbstractVector)
            old_wc = old_wc_rms[1]
            rm.sigma = old_wc.sigma
            rm.cov_l = old_wc.cov_l
            rm.cov_u = old_wc.cov_u
            rm.cov_mu = old_wc.rm.cov_mu
            rm.cov_sigma = old_wc.rm.cov_sigma
        else
            rm_flat = reduce(vcat, rm)
            for (r, old_wc) ∈ zip(view(rm_flat, wc_idx), old_wc_rms)
                r.sigma = old_wc.sigma
                r.cov_l = old_wc.cov_l
                r.cov_u = old_wc.cov_u
                r.cov_mu = old_wc.rm.cov_mu
                r.cov_sigma = old_wc.rm.cov_sigma
            end
        end
    end
    return nothing
end
function reset_special_rms(rm, special_rm_idx::NCOSpecialRMIdx, old_stats::NCOOldStats)
    (; cov_idx, kurt_idx, skurt_idx, skew_idx, sskew_idx, wc_idx) = special_rm_idx
    (; old_covs, old_kurts, old_skurts, old_Vs, old_skews, old_SVs, old_sskews, old_wc_rms) = old_stats

    _reset_cov_rm(rm, cov_idx, old_covs)
    _reset_kt_rm(rm, kurt_idx, old_kurts)
    _reset_kt_rm(rm, skurt_idx, old_skurts)
    _reset_skew_rm(rm, skew_idx, old_Vs, old_skews)
    _reset_skew_rm(rm, sskew_idx, old_SVs, old_sskews)
    _reset_wc_var_rm(rm, wc_idx, old_wc_rms)
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
function get_cluster_portfolio(port, rm, internal_args, cluster, cidx, idx_sq, Nc,
                               special_rm_idx)
    (; opt_kwargs, port_kwargs, stats_kwargs, wc_kwargs, factor_kwargs, cluster_kwargs) = internal_args

    type = !haskey(opt_kwargs, :type) ? Trad() : opt_kwargs.type
    kelly = !haskey(opt_kwargs, :kelly) ? NoKelly() : opt_kwargs.kelly
    class = !haskey(opt_kwargs, :classic) ? Classic() : opt_kwargs.class

    (; assets, returns, f_assets, f_returns, loadings, regression_type, mu_l, mu, cov, cor, dist, k, max_num_assets_kurt, max_num_assets_kurt_scale, kurt, skurt, L_2, S_2, skew, V, sskew, SV, f_mu, f_cov, fm_returns, fm_mu, fm_cov, bl_bench_weights, bl_mu, bl_cov, blfm_mu, blfm_cov, cov_l, cov_u, cov_mu, cov_sigma, d_mu, k_mu, k_sigma, w_min, w_max, risk_budget, f_risk_budget, short, long_l, long_u, short_l, short_u, min_budget, budget, max_budget, min_short_budget, short_budget, max_short_budget, card_scale, card, nea, tracking, turnover, l1, l2, long_fees, short_fees, rebalance, solvers) = port

    (; cov_idx, kurt_idx, skurt_idx, skew_idx, sskew_idx, wc_idx) = special_rm_idx
    wc_flag = !isempty(wc_idx)
    cov_flag = !isempty(cov_idx) || isa(kelly, AKelly) || wc_flag
    kurt_flag = !isempty(kurt_idx)
    skurt_flag = !isempty(skurt_idx)
    skew_flag = !isempty(skew_idx)
    sskew_flag = !isempty(sskew_idx)
    factor_flag = isa(class, Union{FM, FC})
    bl_flag = isa(class, Union{BL, BLFM})
    blfm_flag = isa(class, BLFM)
    hc_flag = isa(type, HCOptimType)
    rp_flag = isa(type, Union{RP, RRP})
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
        skurt = get_cluster_matrix(skurt, cidx)
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
        long_u = get_cluster_real_or_vector(long_u, cidx)
        short_l = get_cluster_real_or_vector(short_l, cidx)
        short_u = get_cluster_real_or_vector(short_u, cidx)
        tracking = get_cluster_tracking(tracking, cidx)
        turnover = get_cluster_tr(turnover, cidx)
        long_fees = get_cluster_real_or_vector(long_fees, cidx)
        short_fees = get_cluster_real_or_vector(short_fees, cidx)
        rebalance = get_cluster_tr(rebalance, cidx)
    else
        long_l = 0.0
        long_u = 1.0
        short_l = -0.0
        short_u = -0.2
        tracking = NoTracking()
        turnover = NoTR()
        long_fees = 0.0
        short_fees = 0.0
        rebalance = NoTR()
    end

    intra_port = OmniPortfolio(; assets = assets, ret = returns, f_assets = f_assets,
                               f_ret = f_returns, loadings = loadings,
                               regression_type = regression_type, mu_l = mu_l, mu = mu,
                               cov = cov, cor = cor, dist = dist, k = k,
                               max_num_assets_kurt = max_num_assets_kurt,
                               max_num_assets_kurt_scale = max_num_assets_kurt_scale,
                               kurt = kurt, skurt = skurt, L_2 = L_2, S_2 = S_2,
                               skew = skew, V = V, sskew = sskew, SV = SV, f_mu = f_mu,
                               f_cov = f_cov, fm_returns = fm_returns, fm_mu = fm_mu,
                               fm_cov = fm_cov, bl_bench_weights = bl_bench_weights,
                               bl_mu = bl_mu, bl_cov = bl_cov, blfm_mu = blfm_mu,
                               blfm_cov = blfm_cov, cov_l = cov_l, cov_u = cov_u,
                               cov_mu = cov_mu, cov_sigma = cov_sigma, d_mu = d_mu,
                               k_mu = k_mu, k_sigma = k_sigma, w_min = w_min, w_max = w_max,
                               risk_budget = risk_budget, f_risk_budget = f_risk_budget,
                               short = short, long_l = long_l, long_u = long_u,
                               short_l = short_l, short_u = short_u,
                               min_budget = min_budget, budget = budget,
                               max_budget = max_budget, min_short_budget = min_short_budget,
                               short_budget = short_budget,
                               max_short_budget = max_short_budget, card_scale = card_scale,
                               card = card, nea = nea, tracking = tracking,
                               turnover = turnover, l1 = l1, l2 = l2, long_fees = long_fees,
                               short_fees = short_fees, rebalance = rebalance,
                               solvers = solvers, port_kwargs...)

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

    w = optimise!(intra_port, type; rm = rm, opt_kwargs...)
    if !isempty(w)
        w = w.weights
    else
        w = zeros(eltype(returns), size(returns, 2))
    end

    return w, intra_port.fail
end
function set_rm_stats(port::OmniPortfolio, rm, cluster, cidx, idx_sq, Nc, special_rm_idx)
    (; cov_idx, kurt_idx, skurt_idx, skew_idx, sskew_idx, wc_idx) = special_rm_idx

    cov_flag = !isempty(cov_idx)
    kurt_flag = !isempty(kurt_idx)
    skurt_flag = !isempty(skurt_idx)
    skew_flag = !isempty(skew_idx)
    sskew_flag = !isempty(sskew_idx)
    wc_flag = !isempty(wc_idx)

    old_covs = Vector{Union{Matrix{eltype(port.returns)}, Nothing}}(undef, 0)
    old_kurts = Vector{Union{Matrix{eltype(port.returns)}, Nothing}}(undef, 0)
    old_skurts = Vector{Union{Matrix{eltype(port.returns)}, Nothing}}(undef, 0)
    old_Vs = Vector{Union{Matrix{eltype(port.returns)}, Nothing}}(undef, 0)
    old_SVs = Vector{Union{Matrix{eltype(port.returns)}, Nothing}}(undef, 0)
    old_skews = Vector{Union{Matrix{eltype(port.returns)}, Nothing}}(undef, 0)
    old_sskews = Vector{Union{Matrix{eltype(port.returns)}, Nothing}}(undef, 0)
    old_wc_rms = Vector{Union{WCVariance}}(undef, 0)

    if cov_flag
        _set_cov_rm(rm, cov_idx, cidx, old_covs)
    end
    if kurt_flag
        _set_kt_rm(Val(true), rm, port, kurt_idx, idx_sq, old_kurts)
    end
    if skurt_flag
        _set_kt_rm(Val(false), rm, port, skurt_idx, idx_sq, old_skurts)
    end
    if skew_flag
        _set_skew_rm(rm, port, skew_idx, cluster, Nc, idx_sq, old_Vs, old_skews)
    end
    if sskew_flag
        _set_skew_rm(rm, port, sskew_idx, cluster, Nc, idx_sq, old_SVs, old_sskews)
    end
    if wc_flag
        _set_wc_var_rm(rm, port, wc_idx, cidx, idx_sq, old_wc_rms)
    end
    return NCOOldStats(old_covs, old_kurts, old_skurts, old_Vs, old_skews, old_SVs,
                       old_sskews, old_wc_rms)
end
function calc_intra_weights(port::OmniPortfolio, internal_args, rm)
    k = port.k
    idx = cutree(port.clusters; k = k)
    w = zeros(eltype(port.returns), size(port.returns, 2), k)
    cfails = Dict{Int, Dict}()

    special_rm_idx = find_special_rm(rm)
    (; kurt_idx, skurt_idx, skew_idx, sskew_idx, wc_idx) = special_rm_idx
    kurt_flag = !isempty(kurt_idx)
    skurt_flag = !isempty(skurt_idx)
    skew_flag = !isempty(skew_idx)
    sskew_flag = !isempty(sskew_idx)
    wc_flag = !isempty(wc_idx)

    N = size(port.returns, 2)

    for i ∈ 1:k
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

        old_stats = set_rm_stats(port, rm, cluster, cidx, idx_sq, Nc, special_rm_idx)
        cw, cfail = get_cluster_portfolio(port, rm, internal_args, cluster, cidx, idx_sq,
                                          Nc, special_rm_idx)
        reset_special_rms(rm, special_rm_idx, old_stats)

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
function nco_optimise(port, type, rm_i, rm_o)
    wi = calc_intra_weights(port, type.internal, rm_i)
    w = calc_inter_weights(port, type.external, rm_o)

    return w
end