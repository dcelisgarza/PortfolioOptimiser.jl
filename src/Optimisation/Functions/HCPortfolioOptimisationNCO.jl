function find_cov_kurt_skew_rm(rm::Union{AbstractVector, <:RiskMeasure})
    cov_idx = Vector{Int}(undef, 0)
    kurt_idx = Vector{Int}(undef, 0)
    skurt_idx = Vector{Int}(undef, 0)
    set_skew = false
    set_sskew = false
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
        set_skew = isa(rm, Skew)
        set_sskew = isa(rm, SSkew)
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
            if !set_skew
                set_skew = isa(r, Skew)
            end
            if !set_sskew
                set_sskew = isa(r, SSkew)
            end
        end
    end

    return cov_idx, kurt_idx, skurt_idx, set_skew, set_sskew
end
function _get_port_kt(::Val{true}, port, idx)
    return view(port.kurt, idx, idx)
end
function _get_port_kt(::Val{false}, port, idx)
    return view(port.skurt, idx, idx)
end
function _get_port_kt(::Val{true}, port)
    return port.kurt
end
function _get_port_kt(::Val{false}, port)
    return port.skurt
end
function _set_kt_rm(val::Union{Val{true}, Val{false}}, rm, port, kt_idx, idx, old_kts)
    if !isa(rm, AbstractVector)
        if isnothing(rm.kt) || isempty(rm.kt)
            push!(old_kts, rm.kt)
            rm.kt = _get_port_kt(val, port, idx)
        else
            push!(old_kts, rm.kt)
            rm.kt = view(rm.kt, idx, idx)
        end
    else
        rm_flat = reduce(vcat, rm)
        for r ∈ view(rm_flat, kt_idx)
            if isnothing(r.kt) || isempty(r.kt)
                push!(old_kts, r.kt)
                r.kt = _get_port_kt(val, port, idx)
            else
                push!(old_kts, r.kt)
                r.kt = view(r.kt, idx, idx)
            end
        end
    end
    return nothing
end
function _set_cov_rm(rm, port, cov_idx, idx, old_covs)
    if !isa(rm, AbstractVector)
        if isnothing(rm.sigma) || isempty(rm.sigma)
            push!(old_covs, rm.sigma)
            rm.sigma = view(port.cov, idx, idx)
        else
            push!(old_covs, rm.sigma)
            rm.sigma = view(rm.sigma, idx, idx)
        end
    else
        rm_flat = reduce(vcat, rm)
        for r ∈ view(rm_flat, cov_idx)
            if isnothing(r.sigma) || isempty(r.sigma)
                push!(old_covs, r.sigma)
                r.sigma = view(port.cov, idx, idx)
            else
                push!(old_covs, r.sigma)
                r.sigma = view(r.sigma, idx, idx)
            end
        end
    end
    return nothing
end
function _compute_cov_rm(rm, port, cov_idx, wi)
    old_covs = Vector{Union{Matrix{eltype(port.returns)}, Nothing}}(undef, 0)
    if !isa(rm, AbstractVector)
        if isnothing(rm.sigma) || isempty(rm.sigma)
            push!(old_covs, rm.sigma)
            rm.sigma = transpose(wi) * port.cov * wi
        else
            push!(old_covs, rm.sigma)
            rm.sigma = transpose(wi) * rm.sigma * wi
        end
    else
        rm_flat = reduce(vcat, rm)
        for r ∈ view(rm_flat, cov_idx)
            if isnothing(r.sigma) || isempty(r.sigma)
                push!(old_covs, r.sigma)
                r.sigma = transpose(wi) * port.cov * wi
            else
                push!(old_covs, r.sigma)
                r.sigma = transpose(wi) * r.sigma * wi
            end
        end
    end

    return old_covs
end
function _compute_kurt_rm(val::Union{Val{true}, Val{false}}, rm, port, kurt_idx, wi)
    old_kts = Vector{Union{Matrix{eltype(port.returns)}, Nothing}}(undef, 0)
    o = transpose(range(; start = one(eltype(wi)), stop = one(eltype(wi)),
                        length = size(port.returns, 2) * size(wi, 2)))
    z = reshape(kron(o, wi) .* kron(wi, o), size(port.returns, 2)^2, :)
    if !isa(rm, AbstractVector)
        if isnothing(rm.kt) || isempty(rm.kt)
            push!(old_kts, rm.kt)
            rm.kt = transpose(z) * _get_port_kt(val, port) * z
        else
            push!(old_kts, rm.kt)
            rm.kt = transpose(z) * rm.kt * z
        end
    else
        rm_flat = reduce(vcat, rm)
        for r ∈ view(rm_flat, kurt_idx)
            if isnothing(r.kt) || isempty(r.kt)
                push!(old_kts, r.kt)
                r.kt = transpose(z) * _get_port_kt(val, port) * z
            else
                push!(old_kts, r.kt)
                r.kt = transpose(z) * r.kt * z
            end
        end
    end

    return old_kts
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
function _reset_cov_rm(rm, cov_idx, old_covs)
    if !isempty(cov_idx)
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
function reset_cov_kurt_and_skurt_rm(rm, cov_idx, old_covs, kurt_idx, old_kurts, skurt_idx,
                                     old_skurts)
    _reset_cov_rm(rm, cov_idx, old_covs)
    _reset_kt_rm(rm, kurt_idx, old_kurts)
    _reset_kt_rm(rm, skurt_idx, old_skurts)
    return nothing
end
function gen_cluster_stats(port, rm, cidx, cov_idx, kurt_idx, skurt_idx, set_skew,
                           set_sskew)
    cassets = port.assets[cidx]
    cret = port.returns[:, cidx]
    cmu = !isempty(port.mu) ? port.mu[cidx] : Vector{eltype(port.returns)}(undef, 0)
    old_covs = Vector{Union{Matrix{eltype(port.returns)}, Nothing}}(undef, 0)
    old_kurts = Vector{Union{Matrix{eltype(port.returns)}, Nothing}}(undef, 0)
    old_skurts = Vector{Union{Matrix{eltype(port.returns)}, Nothing}}(undef, 0)
    cV = Matrix{eltype(port.returns)}(undef, 0, 0)
    cSV = Matrix{eltype(port.returns)}(undef, 0, 0)
    if !isempty(cov_idx)
        _set_cov_rm(rm, port, cov_idx, cidx, old_covs)
    end
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
    return cassets, cret, cmu, old_covs, old_kurts, old_skurts, cV, cSV
end
function intra_nco_opt(port, rm, cassets, cret, cmu, use_kurt_skurt, cV, cSV, opt_kwargs,
                       port_kwargs, factor_kwargs)
    intra_port = Portfolio(; assets = cassets, ret = cret, mu = cmu, V = cV, SV = cSV,
                           solvers = port.solvers, port_kwargs...)
    if use_kurt_skurt
        intra_port.L_2, intra_port.S_2 = dup_elim_sum_matrices(size(cret, 2))[2:3]
    end
    if haskey(opt_kwargs, :class) && isa(opt_kwargs.class, Union{FM, FC})
        factor_statistics!(intra_port, factor_kwargs...)
    end

    w = optimise!(intra_port; rm = rm, opt_kwargs...)
    if !isempty(w)
        w = w.weights
    else
        w = zeros(eltype(cret), size(cret, 2))
    end

    return w, intra_port.fail
end
function calc_intra_weights(port, rm::Union{AbstractVector, <:RiskMeasure}, opt_kwargs,
                            port_kwargs, factor_kwargs)
    idx = cutree(port.clusters; k = port.k)
    w = zeros(eltype(port.returns), size(port.returns, 2), port.k)
    cfails = Dict{Int, Dict}()

    cov_idx, kurt_idx, skurt_idx, set_skew, set_sskew = find_cov_kurt_skew_rm(rm)
    use_kurt_skurt = !isempty(kurt_idx) || !isempty(skurt_idx)

    for i ∈ 1:(port.k)
        cidx = idx .== i
        cassets, cret, cmu, old_covs, old_kurts, old_skurts, cV, cSV = gen_cluster_stats(port,
                                                                                         rm,
                                                                                         cidx,
                                                                                         cov_idx,
                                                                                         kurt_idx,
                                                                                         skurt_idx,
                                                                                         set_skew,
                                                                                         set_sskew)
        cw, cfail = intra_nco_opt(port, rm, cassets, cret, cmu, use_kurt_skurt, cV, cSV,
                                  opt_kwargs, port_kwargs, factor_kwargs)
        reset_cov_kurt_and_skurt_rm(rm, cov_idx, old_covs, kurt_idx, old_kurts, skurt_idx,
                                    old_skurts)

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
function inter_nco_opt(port, rm, cassets, cret, cmu, use_kurt_skurt, set_skew, set_sskew,
                       opt_kwargs, port_kwargs, factor_kwargs, stat_kwargs)
    inter_port = Portfolio(; assets = cassets, ret = cret, mu = cmu, solvers = port.solvers,
                           port_kwargs...)
    asset_statistics!(inter_port; set_cov = false, set_mu = false, set_kurt = false,
                      set_skurt = false, set_skew = set_skew, set_sskew = set_sskew,
                      stat_kwargs...)
    if use_kurt_skurt
        inter_port.L_2, inter_port.S_2 = dup_elim_sum_matrices(size(cret, 2))[2:3]
    end
    if haskey(opt_kwargs, :class) && isa(opt_kwargs.class, Union{FM, FC})
        factor_statistics!(inter_port, factor_kwargs...)
    end

    w = optimise!(inter_port; rm = rm, opt_kwargs...)
    if !isempty(w)
        w = w.weights
    else
        w = zeros(eltype(cret), size(cret, 2))
    end

    return w, inter_port.fail
end
function _compute_inter_statistics(port, rm, cov_idx, kurt_idx, skurt_idx, wi)
    cmu = if !isempty(port.mu)
        transpose(wi) * port.mu
    else
        Vector{eltype(port.returns)}(undef, 0)
    end

    old_covs = if !isempty(cov_idx)
        _compute_cov_rm(rm, port, cov_idx, wi)
    else
        Vector{Int}(undef, 0)
    end

    old_kurts = if !isempty(kurt_idx)
        _compute_kurt_rm(Val(true), rm, port, kurt_idx, wi)
    else
        Vector{Int}(undef, 0)
    end

    old_skurts = if !isempty(skurt_idx)
        _compute_kurt_rm(Val(false), rm, port, skurt_idx, wi)
    else
        Vector{Int}(undef, 0)
    end

    return cmu, old_covs, old_kurts, old_skurts
end
function calc_inter_weights(port, wi, rm, opt_kwargs, port_kwargs, factor_kwargs,
                            stat_kwargs)
    cret = port.returns * wi

    cov_idx, kurt_idx, skurt_idx, set_skew, set_sskew = find_cov_kurt_skew_rm(rm)
    use_kurt_skurt = !isempty(kurt_idx) || !isempty(skurt_idx)

    cmu, old_covs, old_kurts, old_skurts = _compute_inter_statistics(port, rm, cov_idx,
                                                                     kurt_idx, skurt_idx,
                                                                     wi)
    cw, cfail = inter_nco_opt(port, rm, 1:size(cret, 2), cret, cmu, use_kurt_skurt,
                              set_skew, set_sskew, opt_kwargs, port_kwargs, factor_kwargs,
                              stat_kwargs)
    reset_cov_kurt_and_skurt_rm(rm, cov_idx, old_covs, kurt_idx, old_kurts, skurt_idx,
                                old_skurts)
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
                            type.factor_kwargs)
    w = calc_inter_weights(port, wi, rm_o, type.opt_kwargs_o, type.port_kwargs_o,
                           type.factor_kwargs_o, type.stat_kwargs_o)

    return w
end
