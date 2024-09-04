function w_limits(type::NCO, datatype = Float64)
    port_kwargs = type.port_kwargs
    port_kwargs_o = type.port_kwargs_o
    lo, hi = if isa(type, NCO) && (haskey(port_kwargs, :short) && port_kwargs.short ||
                                   haskey(port_kwargs_o, :short) && port_kwargs_o.short)
        la = nothing
        ha = nothing
        lb = nothing
        hb = nothing

        if haskey(port_kwargs, :short) && port_kwargs.short
            if haskey(port_kwargs, :short_u)
                la = port_kwargs.short_u
            end
            if haskey(port_kwargs, :long_u)
                ha = port_kwargs.long_u
            end
        end

        if haskey(port_kwargs_o, :short) && port_kwargs_o.short
            if haskey(port_kwargs_o, :short_u)
                lb = port_kwargs_o.short_u
            end
            if haskey(port_kwargs_o, :long_u)
                hb = port_kwargs_o.long_u
            end
        end

        if isnothing(la) && isnothing(lb)
            la = lb = 0.2 * one(datatype)
        elseif isnothing(la)
            la = lb
        elseif isnothing(lb)
            lb = la
        end

        if isnothing(ha) && isnothing(hb)
            ha = hb = one(datatype)
        elseif isnothing(ha)
            ha = hb
        elseif isnothing(hb)
            hb = ha
        end

        -max(la, lb), max(ha, hb)
    else
        zero(datatype), one(datatype)
    end

    return lo, hi
end
function w_limits(::Any, datatype = Float64)
    return zero(datatype), one(datatype)
end
function set_hc_weights(w_min, w_max, N, lo = 0.0, hi = 1.0)
    lower_bound = if isa(w_min, AbstractVector) && isempty(w_min)
        zeros(N)
    elseif isa(w_min, AbstractVector) && !isempty(w_min)
        max.(lo, w_min)
    else
        fill(max(lo, w_min), N)
    end

    upper_bound = if isa(w_max, AbstractVector) && isempty(w_max)
        ones(N)
    elseif isa(w_max, AbstractVector) && !isempty(w_max)
        min.(hi, w_max)
    else
        fill(min(hi, w_max), N)
    end

    @smart_assert(all(upper_bound .>= lower_bound))

    return lower_bound, upper_bound
end
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
function _naive_risk(::Equal, returns, ::Any)
    N = size(returns, 2)
    return fill(eltype(returns)(inv(N)), N)
end
function _naive_risk(rm::RiskMeasure, returns, cV)
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
function cluster_risk(port, cluster, rm)
    if hasproperty(rm, :sigma)
        rm.sigma = view(port.cov, cluster, cluster)
    end
    cret = view(port.returns, :, cluster)
    cV = gen_cluster_skew_sskew(rm, port, cluster)
    cw = _naive_risk(rm, cret, cV)
    crisk = calc_risk(rm, cw; X = cret, V = cV, SV = cV)
    if hasproperty(rm, :sigma)
        rm.sigma = nothing
    end
    return crisk
end
function naive_risk(port, cluster, rm)
    if hasproperty(rm, :sigma)
        rm.sigma = view(port.cov, cluster, cluster)
    end
    cret = view(port.returns, :, cluster)
    cV = gen_cluster_skew_sskew(rm, port, cluster)
    crisk = _naive_risk(rm, cret, cV)
    if hasproperty(rm, :sigma)
        rm.sigma = nothing
    end
    return crisk
end
function cluster_weight_bounds(w_min, w_max, weights, lc, rc, alpha_1)
    if !(any(w_max .< weights) || any(w_min .> weights))
        return alpha_1
    end
    lmaxw = weights[lc[1]]
    a1 = sum(w_max[lc]) / lmaxw
    a2 = max(sum(w_min[lc]) / lmaxw, alpha_1)
    alpha_1 = min(a1, a2)

    rmaxw = weights[rc[1]]
    a1 = sum(w_max[rc]) / rmaxw
    a2 = max(sum(w_min[rc]) / rmaxw, 1 - alpha_1)
    alpha_1 = one(a1) - min(a1, a2)
    return alpha_1
end
function _optimise!(::HRP, port::HCPortfolio, rm::Union{AbstractVector, <:RiskMeasure},
                    ::Any, w_min, w_max)
    N = size(port.returns, 2)
    weights = ones(eltype(port.returns), N)
    items = [port.clusters.order]

    while length(items) > 0
        items = [i[j:k] for i ∈ items
                 for (j, k) ∈ ((1, div(length(i), 2)), (1 + div(length(i), 2), length(i)))
                 if length(i) > 1]

        for i ∈ 1:2:length(items)
            lc = items[i]
            rc = items[i + 1]
            lrisk = zero(eltype(weights))
            rrisk = zero(eltype(weights))
            for r ∈ rm
                solver_flag = false
                if hasproperty(r, :solvers) && (isnothing(r.solvers) || isempty(r.solvers))
                    r.solvers = port.solvers
                    solver_flag = true
                end
                scale = r.settings.scale
                # Left risk.
                lrisk += cluster_risk(port, lc, r) * scale
                # Right risk.
                rrisk += cluster_risk(port, rc, r) * scale
                if solver_flag
                    r.solvers = nothing
                end
            end
            # Allocate weight to clusters.
            alpha_1 = one(lrisk) - lrisk / (lrisk + rrisk)
            # Weight constraints.
            alpha_1 = cluster_weight_bounds(w_min, w_max, weights, lc, rc, alpha_1)
            weights[lc] *= alpha_1
            weights[rc] *= one(alpha_1) - alpha_1
        end
    end
    return weights
end
function opt_weight_bounds(w_min, w_max, weights, max_iter = 100)
    if !(any(w_max .< weights) || any(w_min .> weights))
        return weights
    end

    for _ ∈ 1:max_iter
        if !(any(w_max .< weights) || any(w_min .> weights))
            break
        end

        old_w = copy(weights)
        weights = max.(min.(weights, w_max), w_min)
        idx = weights .< w_max .&& weights .> w_min
        w_add = sum(max.(old_w - w_max, 0.0))
        w_sub = sum(min.(old_w - w_min, 0.0))
        delta = w_add + w_sub

        if delta != 0
            weights[idx] += delta * weights[idx] / sum(weights[idx])
        end
    end
    return weights
end
function finalise_weights(type::Any, port, weights, w_min, w_max, max_iter)
    stype = Symbol(type)
    weights = opt_weight_bounds(w_min, w_max, weights, max_iter)
    weights ./= sum(weights)
    port.optimal[stype] = DataFrame(; tickers = port.assets, weights = weights)
    return port.optimal[stype]
end
function finalise_weights(type::NCO, port, weights, w_min, w_max, max_iter)
    stype = Symbol(type)
    port_kwargs = type.port_kwargs
    port_kwargs_o = type.port_kwargs_o
    port.optimal[stype] = if !isempty(port.fail) || any(.!isfinite.(weights))
        port.fail[:port] = DataFrame(; tickers = port.assets, weights = weights)
        DataFrame()
    elseif haskey(port_kwargs, :short) && port_kwargs.short ||
           haskey(port_kwargs_o, :short) && port_kwargs_o.short
        weights = opt_weight_bounds(w_min, w_max, weights, max_iter)
        DataFrame(; tickers = port.assets, weights = weights)
    else
        weights = opt_weight_bounds(w_min, w_max, weights, max_iter)
        weights ./= sum(weights)
        DataFrame(; tickers = port.assets, weights = weights)
    end
    return port.optimal[stype]
end
function _optimise!(::HERC, port::HCPortfolio, rmi::Union{AbstractVector, <:RiskMeasure},
                    rmo::Union{AbstractVector, <:RiskMeasure}, w_min, w_max)
    nodes = to_tree(port.clusters)[2]
    heights = [i.height for i ∈ nodes]
    nodes = nodes[sortperm(heights; rev = true)]

    weights = ones(eltype(port.returns), size(port.returns, 2))

    idx = cutree(port.clusters; k = port.k)

    clusters = Vector{Vector{Int}}(undef, length(minimum(idx):maximum(idx)))
    for i ∈ eachindex(clusters)
        clusters[i] = findall(idx .== i)
    end

    # Treat each cluster as its own portfolio and optimise each one individually.
    # Calculate the weight of each cluster relative to the other clusters.
    for i ∈ nodes[1:(port.k - 1)]
        if is_leaf(i)
            continue
        end

        # Do this recursively accounting for the dendrogram structure.
        ln = pre_order(i.left)
        rn = pre_order(i.right)

        lrisk = 0.0
        rrisk = 0.0

        lc = Int[]
        rc = Int[]
        for r ∈ rmo
            solver_flag = false
            if hasproperty(r, :solvers) && (isnothing(r.solvers) || isempty(r.solvers))
                r.solvers = port.solvers
                solver_flag = true
            end
            scale = r.settings.scale
            for cluster ∈ clusters
                _risk = cluster_risk(port, cluster, r) * scale
                if issubset(cluster, ln)
                    lrisk += _risk
                    append!(lc, cluster)
                elseif issubset(cluster, rn)
                    rrisk += _risk
                    append!(rc, cluster)
                end
            end
            if solver_flag
                r.solvers = nothing
            end
        end
        # Allocate weight to clusters.
        alpha_1 = one(lrisk) - lrisk / (lrisk + rrisk)
        # Weight constraints.
        alpha_1 = cluster_weight_bounds(w_min, w_max, weights, lc, rc, alpha_1)

        weights[ln] *= alpha_1
        weights[rn] *= 1 - alpha_1
    end

    risk = zeros(eltype(port.returns), size(port.returns, 2))
    for i ∈ 1:(port.k)
        cidx = idx .== i
        clusters = findall(cidx)
        for r ∈ rmi
            solver_flag = false
            if hasproperty(r, :solvers) && (isnothing(r.solvers) || isempty(r.solvers))
                r.solvers = port.solvers
                solver_flag = true
            end
            scale = r.settings.scale
            risk[cidx] .+= naive_risk(port, clusters, r) * scale
            if solver_flag
                r.solvers = nothing
            end
        end
        weights[cidx] .*= risk[cidx]
    end

    return weights
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
function intra_nco_opt(port, rm, cassets, cret, cmu, ccov, ckurt, cskurt, cV, cSV,
                       opt_kwargs, port_kwargs)
    intra_port = Portfolio(; assets = cassets, ret = cret, mu = cmu, cov = ccov,
                           kurt = ckurt, skurt = cskurt, V = cV, SV = cSV,
                           solvers = port.solvers, port_kwargs...)
    if !isempty(ckurt) || !isempty(cskurt)
        intra_port.L_2, intra_port.S_2 = dup_elim_sum_matrices(size(cret, 2))[2:3]
    end

    w = optimise!(intra_port; rm = rm, opt_kwargs...)
    if !isempty(w)
        w = w.weights
    else
        w = zeros(eltype(cret), size(cret, 2))
    end

    return w, intra_port.fail
end
function calc_intra_weights(port, rm::Union{AbstractVector, <:TradRiskMeasure}, opt_kwargs,
                            port_kwargs)
    idx = cutree(port.clusters; k = port.k)
    w = zeros(eltype(port.returns), size(port.returns, 2), port.k)
    cfails = Dict{Int, Dict}()

    set_kurt, set_skurt, set_skew, set_sskew = find_kurt_skew_rm(rm)

    for i ∈ 1:(port.k)
        cidx = idx .== i
        cassets, cret, cmu, ccov, ckurt, cskurt, cV, cSV = gen_cluster_stats(port, cidx,
                                                                             set_kurt,
                                                                             set_skurt,
                                                                             set_skew,
                                                                             set_sskew)
        cw, cfail = intra_nco_opt(port, rm, cassets, cret, cmu, ccov, ckurt, cskurt, cV,
                                  cSV, opt_kwargs, port_kwargs)
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
function inter_nco_opt(port, rm, cassets, cret, cmu, ccov, set_kurt, set_skurt, set_skew,
                       set_sskew, opt_kwargs, port_kwargs, stat_kwargs)
    inter_port = Portfolio(; assets = cassets, ret = cret, mu = cmu, cov = ccov,
                           solvers = port.solvers, port_kwargs...)
    asset_statistics!(inter_port; set_cov = false, set_mu = false, set_kurt = set_kurt,
                      set_skurt = set_skurt, set_skew = set_skew, set_sskew = set_sskew,
                      stat_kwargs...)

    w = optimise!(inter_port; rm = rm, opt_kwargs...)

    if !isempty(w)
        w = w.weights
    else
        w = zeros(eltype(cret), size(cret, 2))
    end

    return w, inter_port.fail
end
function calc_inter_weights(port, wi, rm, opt_kwargs, port_kwargs, stat_kwargs)
    cret = port.returns * wi
    cmu = transpose(wi) * port.mu
    ccov = transpose(wi) * port.cov * wi

    set_kurt, set_skurt, set_skew, set_sskew = find_kurt_skew_rm(rm)
    cw, cfail = inter_nco_opt(port, rm, 1:size(cret, 2), cret, cmu, ccov, set_kurt,
                              set_skurt, set_skew, set_sskew, opt_kwargs, port_kwargs,
                              stat_kwargs)

    w = wi * cw

    if !isempty(cfail)
        port.fail[:inter] = cfail
    end

    return w
end
function _optimise!(type::NCO, port::HCPortfolio, rmi::Union{AbstractVector, <:RiskMeasure},
                    rmo::Union{AbstractVector, <:RiskMeasure}, ::Any, ::Any)
    port.fail = Dict()
    wi = calc_intra_weights(port, rmi, type.opt_kwargs, type.port_kwargs)
    w = calc_inter_weights(port, wi, rmo, type.opt_kwargs_o, type.port_kwargs_o,
                           type.stat_kwargs_o)

    return w
end
function optimise!(port::HCPortfolio; rm::Union{AbstractVector, <:RiskMeasure} = SD(),
                   rmo::Union{AbstractVector, <:RiskMeasure} = rm,
                   type::HCOptimType = HRP(), cluster::Bool = true,
                   hclust_alg::HClustAlg = HAC(), hclust_opt::HCOpt = HCOpt(),
                   max_iter::Int = 100)
    if cluster
        cluster_assets!(port; hclust_alg = hclust_alg, hclust_opt = hclust_opt)
    end
    lo, hi = w_limits(type, eltype(port.returns))
    w_min, w_max = set_hc_weights(port.w_min, port.w_max, size(port.returns, 2), lo, hi)
    w = _optimise!(type, port, rm, rmo, w_min, w_max)
    return finalise_weights(type, port, w, w_min, w_max, max_iter)
end
