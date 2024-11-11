function find_cov_kurt_skew_rm(rm::Union{AbstractVector, <:AbstractRiskMeasure})
    cov_idx = Vector{Int}(undef, 0)
    kurt_idx = Vector{Int}(undef, 0)
    skurt_idx = Vector{Int}(undef, 0)
    skew_idx = Vector{Int}(undef, 0)
    sskew_idx = Vector{Int}(undef, 0)
    if !isa(rm, AbstractVector)
        if isa(rm, SD)
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
    else
        rm_flat = reduce(vcat, rm)
        for (i, r) ∈ enumerate(rm_flat) #! Do not change this enumerate to pairs.
            if isa(r, SD)
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
        end
    end

    return cov_idx, kurt_idx, skurt_idx, skew_idx, sskew_idx
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
function gen_cluster_stats(port, rm, cidx, cov_idx, kurt_idx, skurt_idx, skew_idx,
                           sskew_idx, opt_kwargs)
    cassets = port.assets[cidx]
    cret = view(port.returns, :, cidx)
    cmu = !isempty(port.mu) ? port.mu[cidx] : Vector{eltype(port.returns)}(undef, 0)
    ccov = if !isempty(port.cov)
        view(port.cov, cidx, cidx)
    else
        Matrix{eltype(port.returns)}(undef, 0, 0)
    end
    ccor, cdist = if haskey(opt_kwargs, :type) && isa(opt_kwargs.type, HCOptimType)
        view(port.cor, cidx, cidx), view(port.dist, cidx, cidx)
    else
        Matrix{eltype(port.returns)}(undef, 0, 0), Matrix{eltype(port.returns)}(undef, 0,
                                                                                0)
    end

    old_covs = Vector{Union{Matrix{eltype(port.returns)}, Nothing}}(undef, 0)
    old_kurts = Vector{Union{Matrix{eltype(port.returns)}, Nothing}}(undef, 0)
    old_skurts = Vector{Union{Matrix{eltype(port.returns)}, Nothing}}(undef, 0)
    old_Vs = Vector{Union{Matrix{eltype(port.returns)}, Nothing}}(undef, 0)
    old_SVs = Vector{Union{Matrix{eltype(port.returns)}, Nothing}}(undef, 0)
    old_skews = Vector{Union{Matrix{eltype(port.returns)}, Nothing}}(undef, 0)
    old_sskews = Vector{Union{Matrix{eltype(port.returns)}, Nothing}}(undef, 0)
    if !isempty(cov_idx)
        _set_cov_rm(rm, cov_idx, cidx, old_covs)
    end
    if !isempty(kurt_idx) ||
       !isempty(skurt_idx) ||
       !isempty(skew_idx) ||
       !isempty(sskew_idx)
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
        if !isempty(skew_idx)
            _set_skew_rm(rm, port, skew_idx, cluster, Nc, idx, old_Vs, old_skews)
        end
        if !isempty(sskew_idx)
            _set_skew_rm(rm, port, sskew_idx, cluster, Nc, idx, old_SVs, old_sskews)
        end
    end
    return cassets, cret, cmu, ccov, ccor, cdist, old_covs, old_kurts, old_skurts, old_Vs,
           old_skews, old_SVs, old_sskews
end
function _get_port_kt(::Val{true}, port, idx)
    return view(port.kurt, idx, idx)
end
function _get_port_kt(::Val{false}, port, idx)
    return view(port.skurt, idx, idx)
end
function _set_kt_rm_nothing(rm, kt_idx, old_kts)
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
end
function _set_skew_rm_nothing(rm, skew_idx, old_Vs, old_skews)
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
function set_kurt_skurt_skew_nothing(port, rm, kurt_idx, skurt_idx, skew_idx, sskew_idx)
    old_kurts = Vector{Union{Matrix{eltype(port.returns)}, Nothing}}(undef, 0)
    old_skurts = Vector{Union{Matrix{eltype(port.returns)}, Nothing}}(undef, 0)
    old_Vs = Vector{Union{Matrix{eltype(port.returns)}, Nothing}}(undef, 0)
    old_SVs = Vector{Union{Matrix{eltype(port.returns)}, Nothing}}(undef, 0)
    old_skews = Vector{Union{Matrix{eltype(port.returns)}, Nothing}}(undef, 0)
    old_sskews = Vector{Union{Matrix{eltype(port.returns)}, Nothing}}(undef, 0)
    if !isempty(kurt_idx)
        _set_kt_rm_nothing(rm, kurt_idx, old_kurts)
    end
    if !isempty(skurt_idx)
        _set_kt_rm_nothing(rm, skurt_idx, old_skurts)
    end
    if !isempty(skew_idx)
        _set_skew_rm_nothing(rm, skew_idx, old_Vs, old_skews)
    end
    if !isempty(sskew_idx)
        _set_skew_rm_nothing(rm, sskew_idx, old_SVs, old_sskews)
    end
    return old_kurts, old_skurts, old_Vs, old_skews, old_SVs, old_sskews
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
function reset_cov_kurt_and_skurt_rm(rm, cov_idx, old_covs, kurt_idx, old_kurts, skurt_idx,
                                     old_skurts, skew_idx, old_Vs, old_skews, sskew_idx,
                                     old_SVs, old_sskews)
    _reset_cov_rm(rm, cov_idx, old_covs)
    _reset_kt_rm(rm, kurt_idx, old_kurts)
    _reset_kt_rm(rm, skurt_idx, old_skurts)
    _reset_skew_rm(rm, skew_idx, old_Vs, old_skews)
    _reset_skew_rm(rm, sskew_idx, old_SVs, old_sskews)
    return nothing
end
function intra_nco_opt(port, rm, cassets, cret, cmu, ccov, ccor, cdist, set_kurt,
                       set_sskurt, opt_kwargs, port_kwargs, factor_kwargs, wc_kwargs,
                       cluster_kwargs)
    L_2, S_2 = if set_kurt || set_sskurt
        dup_elim_sum_matrices(size(cret, 2))[2:3]
    else
        SparseMatrixCSC{eltype(cret), Int}(undef, 0, 0),
        SparseMatrixCSC{eltype(cret), Int}(undef, 0, 0)
    end
    if !haskey(opt_kwargs, :type) ||
       haskey(opt_kwargs, :type) && isa(opt_kwargs.type, OptimType)
        intra_port = Portfolio(; assets = cassets, ret = cret, mu = cmu, cov = ccov,
                               L_2 = L_2, S_2 = S_2, solvers = port.solvers, port_kwargs...)
        if haskey(opt_kwargs, :class) && isa(opt_kwargs.class, Union{FM, FC})
            factor_statistics!(intra_port; factor_kwargs...)
        end
        if haskey(opt_kwargs, :type) && isa(opt_kwargs.type, WC)
            wc_statistics!(intra_port; wc_kwargs...)
        end
    else
        intra_port = HCPortfolio(; assets = cassets, ret = cret, mu = cmu, cov = ccov,
                                 cor = ccor, dist = cdist, L_2 = L_2, S_2 = S_2,
                                 solvers = port.solvers, port_kwargs...)
        cluster_assets!(intra_port; cluster_kwargs...)
    end

    w = optimise!(intra_port; rm = rm, opt_kwargs...)
    if !isempty(w)
        w = w.weights
    else
        w = zeros(eltype(cret), size(cret, 2))
    end

    return w, intra_port.fail
end
function calc_intra_weights(port, rm::Union{AbstractVector, <:AbstractRiskMeasure},
                            opt_kwargs, port_kwargs, factor_kwargs, wc_kwargs,
                            cluster_kwargs)
    idx = cutree(port.clusters; k = port.k)
    w = zeros(eltype(port.returns), size(port.returns, 2), port.k)
    cfails = Dict{Int, Dict}()

    cov_idx, kurt_idx, skurt_idx, skew_idx, sskew_idx = find_cov_kurt_skew_rm(rm)
    for i ∈ 1:(port.k)
        cidx = idx .== i
        cassets, cret, cmu, ccov, ccor, cdist, old_covs, old_kurts, old_skurts, old_Vs, old_skews, old_SVs, old_sskews = gen_cluster_stats(port,
                                                                                                                                           rm,
                                                                                                                                           cidx,
                                                                                                                                           cov_idx,
                                                                                                                                           kurt_idx,
                                                                                                                                           skurt_idx,
                                                                                                                                           skew_idx,
                                                                                                                                           sskew_idx,
                                                                                                                                           opt_kwargs)
        cw, cfail = intra_nco_opt(port, rm, cassets, cret, cmu, ccov, ccor, cdist,
                                  !isempty(kurt_idx), !isempty(skurt_idx), opt_kwargs,
                                  port_kwargs, factor_kwargs, wc_kwargs, cluster_kwargs)
        reset_cov_kurt_and_skurt_rm(rm, cov_idx, old_covs, kurt_idx, old_kurts, skurt_idx,
                                    old_skurts, skew_idx, old_Vs, old_skews, sskew_idx,
                                    old_SVs, old_sskews)

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
function inter_nco_opt(port, rm, cassets, wi, set_kurt, set_skurt, set_skew, set_sskew,
                       opt_kwargs, port_kwargs, factor_kwargs, wc_kwargs, cluster_kwargs,
                       stat_kwargs)
    cret = port.returns * wi
    cmu = if !isempty(port.mu)
        transpose(wi) * port.mu
    else
        Vector{eltype(port.returns)}(undef, 0)
    end
    ccov = if !isempty(port.cov)
        transpose(wi) * port.cov * wi
    else
        Matrix{eltype(port.returns)}(undef, 0, 0)
    end
    if !haskey(opt_kwargs, :type) ||
       haskey(opt_kwargs, :type) && isa(opt_kwargs.type, OptimType)
        inter_port = Portfolio(; assets = cassets, ret = cret, mu = cmu, cov = ccov,
                               solvers = port.solvers, port_kwargs...)
        asset_statistics!(inter_port; set_cov = false, set_mu = false, set_cor = false,
                          set_dist = false, set_kurt = set_kurt, set_skurt = set_skurt,
                          set_skew = set_skew, set_sskew = set_sskew, stat_kwargs...)
        if haskey(opt_kwargs, :class) && isa(opt_kwargs.class, Union{FM, FC})
            factor_statistics!(inter_port; factor_kwargs...)
        end
        if haskey(opt_kwargs, :type) && isa(opt_kwargs.type, WC)
            wc_statistics!(inter_port; wc_kwargs...)
        end
    else
        inter_port = HCPortfolio(; assets = cassets, ret = cret, mu = cmu, cov = ccov,
                                 solvers = port.solvers, port_kwargs...)
        asset_statistics!(inter_port; set_cov = false, set_mu = false, set_kurt = set_kurt,
                          set_skurt = set_skurt, set_skew = set_skew, set_sskew = set_sskew,
                          stat_kwargs...)
        cluster_assets!(inter_port; cluster_kwargs...)
    end
    w = optimise!(inter_port; rm = rm, opt_kwargs...)

    if !isempty(w)
        w = w.weights
    else
        w = zeros(eltype(cret), size(cret, 2))
    end

    return w, inter_port.fail
end
function compute_cov_rm(port, rm, cov_idx, wi)
    old_covs = Vector{Union{Matrix{eltype(port.returns)}, Nothing}}(undef, 0)
    if !isempty(cov_idx)
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
    end
    return old_covs
end
function calc_inter_weights(port, wi, rm, opt_kwargs, port_kwargs, factor_kwargs, wc_kwargs,
                            cluster_kwargs, stat_kwargs)
    cov_idx, kurt_idx, skurt_idx, skew_idx, sskew_idx = find_cov_kurt_skew_rm(rm)
    old_covs = compute_cov_rm(port, rm, cov_idx, wi)
    old_kurts, old_skurts, old_Vs, old_skews, old_SVs, old_sskews = set_kurt_skurt_skew_nothing(port,
                                                                                                rm,
                                                                                                kurt_idx,
                                                                                                skurt_idx,
                                                                                                skew_idx,
                                                                                                sskew_idx)
    cw, cfail = inter_nco_opt(port, rm, 1:size(wi, 2), wi, !isempty(kurt_idx),
                              !isempty(skurt_idx), !isempty(skew_idx), !isempty(sskew_idx),
                              opt_kwargs, port_kwargs, factor_kwargs, wc_kwargs,
                              cluster_kwargs, stat_kwargs)
    reset_cov_kurt_and_skurt_rm(rm, cov_idx, old_covs, kurt_idx, old_kurts, skurt_idx,
                                old_skurts, skew_idx, old_Vs, old_skews, sskew_idx, old_SVs,
                                old_sskews)

    w = wi * cw
    if !isempty(cfail)
        port.fail[:inter] = cfail
    end

    return w
end
function _optimise!(type::NCO, port::HCPortfolio,
                    rm_i::Union{AbstractVector, <:AbstractRiskMeasure},
                    rm_o::Union{AbstractVector, <:AbstractRiskMeasure}, ::Any, ::Any)
    port.fail = Dict()
    wi = calc_intra_weights(port, rm_i, type.opt_kwargs, type.port_kwargs,
                            type.factor_kwargs, type.wc_kwargs, type.cluster_kwargs)
    w = calc_inter_weights(port, wi, rm_o, type.opt_kwargs_o, type.port_kwargs_o,
                           type.factor_kwargs_o, type.wc_kwargs_o, type.cluster_kwargs_o,
                           type.stat_kwargs_o)

    return w
end
