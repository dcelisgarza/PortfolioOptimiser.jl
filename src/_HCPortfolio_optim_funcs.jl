function _naive_risk(portfolio, returns, covariance; rm = :SD, rf = 0.0,
                     portfolio_kwargs::NamedTuple = (; alpha = portfolio.alpha,
                                                     a_sim = portfolio.a_sim,
                                                     beta = portfolio.beta,
                                                     b_sim = portfolio.b_sim,
                                                     kappa = portfolio.kappa,
                                                     owa_w = portfolio.owa_w,
                                                     solvers = portfolio.solvers))
    N = size(returns, 2)
    tcov = eltype(covariance)

    if rm == :Equal
        weights = fill(tcov(1 / N), N)
    else
        inv_risk = Vector{tcov}(undef, N)
        w = Vector{tcov}(undef, N)
        for i ∈ eachindex(w)
            w .= zero(tcov)
            w[i] = one(tcov)
            risk = calc_risk(w, returns; rm = rm, rf = rf, sigma = covariance,
                             portfolio_kwargs...)
            inv_risk[i] = 1 / risk
        end
        weights = inv_risk / sum(inv_risk)
    end

    return weights
end

function _opt_w(portfolio, assets, returns, imu, icov, opt;
                asset_stat_kwargs = (; calc_mu = false, calc_cov = false,
                                     calc_kurt = if opt.rm ∈ (:Kurt, :SKurt)
                                         true
                                     else
                                         false
                                     end),
                portfolio_kwargs::NamedTuple = (; alpha = portfolio.alpha,
                                                a_sim = portfolio.a_sim,
                                                beta = portfolio.beta,
                                                b_sim = portfolio.b_sim,
                                                kappa = portfolio.kappa,
                                                owa_w = portfolio.owa_w,
                                                solvers = portfolio.solvers,
                                                max_num_assets_kurt = portfolio.max_num_assets_kurt))
    port = Portfolio(; assets = assets, ret = returns, portfolio_kwargs...)

    port.mu = imu
    port.cov = icov
    if opt.rm ∈ (:Kurt, :SKurt)
        asset_statistics!(port; asset_stat_kwargs...)
    end

    type1 = opt.type
    class1 = opt.class
    obj1 = opt.obj
    weights = if obj1 != :Equal
        opt.type = :Trad
        opt.class = :Classic
        optimise!(port, opt)
    else
        opt.type = :RP
        opt.class = :Classic
        opt.obj = :Min_Risk
        optimise!(port, opt)
    end
    opt.type = type1
    opt.class = class1
    opt.obj = obj1

    if !isempty(weights)
        w = weights.weights
        success = true
    else
        w = zeros(eltype(returns), length(assets))
        success = false
    end

    return w, port.fail, success
end

function _two_diff_gap_stat(dist, clustering, max_k = ceil(Int, sqrt(size(dist, 1))))
    N = size(dist, 1)
    cluster_lvls = [cutree(clustering; k = i) for i ∈ 1:N]

    c1 = min(N, max_k)
    W_list = Vector{eltype(dist)}(undef, c1)

    for i ∈ 1:c1
        lvl = cluster_lvls[i]
        c2 = maximum(unique(lvl))
        mean_dist = 0.0
        for j ∈ 1:c2
            cluster = findall(lvl .== j)
            cluster_dist = dist[cluster, cluster]
            if isempty(cluster_dist)
                continue
            end

            val = 0.0
            counter = 0
            M = size(cluster_dist, 1)
            for col ∈ 1:M
                for row ∈ (col + 1):M
                    val += cluster_dist[row, col]
                    counter += 1
                end
            end
            if counter == 0
                continue
            end
            mean_dist += val / counter
        end
        W_list[i] = mean_dist
    end

    limit_k = floor(Int, min(max_k, sqrt(N)))
    gaps = fill(-Inf, length(W_list))

    if length(W_list) > 2
        gaps[3:end] .= W_list[3:end] .+ W_list[1:(end - 2)] .- 2 * W_list[2:(end - 1)]
    end

    gaps = gaps[1:limit_k]

    k = all(isinf.(gaps)) ? length(gaps) : k = argmax(gaps) + 1

    return k
end

function _hcluster_choice(dist, corr, cor_method, cluster_opt::ClusterOpt)
    linkage = cluster_opt.linkage
    branchorder = cluster_opt.branchorder
    max_k = cluster_opt.max_k
    if linkage == :DBHT
        dbht_method = cluster_opt.dbht_method
        cors = (:Pearson, :Semi_Pearson, :Spearman, :Kendall, :Gerber0, :Gerber1, :Gerber2)
        corr = cor_method ∈ cors ? 1 .- dist .^ 2 : corr
        missing, missing, missing, missing, missing, missing, clustering = DBHTs(dist, corr;
                                                                                 branchorder = branchorder,
                                                                                 method = dbht_method)
    else
        clustering = hclust(dist; linkage = linkage,
                            branchorder = branchorder == :default ? :r : branchorder)
    end

    k = _two_diff_gap_stat(dist, clustering, max_k)

    return clustering, k
end
function _hierarchical_clustering(portfolio::HCPortfolio,
                                  cluster_opt::ClusterOpt = ClusterOpt(;
                                                                       max_k = ceil(Int,
                                                                                    sqrt(size(portfolio.dist,
                                                                                              1)))))
    cor_method = portfolio.cor_method
    corr = portfolio.cor
    dist = portfolio.dist

    clustering, k = _hcluster_choice(dist, corr, cor_method, cluster_opt)

    return clustering, k
end

function _cluster_risk(portfolio, returns, covariance, cluster; rm = :SD, rf = 0.0,
                       portfolio_kwargs::NamedTuple = (; alpha = portfolio.alpha,
                                                       a_sim = portfolio.a_sim,
                                                       beta = portfolio.beta,
                                                       b_sim = portfolio.b_sim,
                                                       kappa = portfolio.kappa,
                                                       owa_w = portfolio.owa_w,
                                                       solvers = portfolio.solvers))
    cret = returns[:, cluster]
    ccov = covariance[cluster, cluster]
    cw = _naive_risk(portfolio, cret, ccov; rm = rm, rf = rf,
                     portfolio_kwargs = portfolio_kwargs)

    crisk = calc_risk(cw, cret; rm = rm, rf = rf, sigma = ccov, portfolio_kwargs...)

    return crisk
end

function _hr_weight_bounds(upper_bound, lower_bound, weights, lc, rc, alpha_1)
    if !(any(upper_bound .< weights) || any(lower_bound .> weights))
        return alpha_1
    end
    lmaxw = weights[lc[1]]
    a1 = sum(upper_bound[lc]) / lmaxw
    a2 = max(sum(lower_bound[lc]) / lmaxw, alpha_1)
    alpha_1 = min(a1, a2)

    rmaxw = weights[rc[1]]
    a1 = sum(upper_bound[rc]) / rmaxw
    a2 = max(sum(lower_bound[rc]) / rmaxw, 1 - alpha_1)
    alpha_1 = 1 - min(a1, a2)
    return alpha_1
end

function _recursive_bisection(portfolio; rm = :SD, rf = 0.0,
                              portfolio_kwargs::NamedTuple = (; alpha = portfolio.alpha,
                                                              a_sim = portfolio.a_sim,
                                                              beta = portfolio.beta,
                                                              b_sim = portfolio.b_sim,
                                                              kappa = portfolio.kappa,
                                                              owa_w = portfolio.owa_w,
                                                              solvers = portfolio.solvers),
                              upper_bound = nothing, lower_bound = nothing)
    N = size(portfolio.returns, 2)
    weights = ones(N)
    sort_order = portfolio.clusters.order

    items = [sort_order]
    returns = portfolio.returns
    covariance = portfolio.cov

    while length(items) > 0
        items = [i[j:k] for i ∈ items
                 for (j, k) ∈ ((1, div(length(i), 2)), (1 + div(length(i), 2), length(i)))
                 if length(i) > 1]

        for i ∈ 1:2:length(items)
            lc = items[i]
            rc = items[i + 1]

            # Left cluster.
            lrisk = _cluster_risk(portfolio, returns, covariance, lc; rm = rm, rf = rf,
                                  portfolio_kwargs = portfolio_kwargs)

            # Right cluster.
            rrisk = _cluster_risk(portfolio, returns, covariance, rc; rm = rm, rf = rf,
                                  portfolio_kwargs = portfolio_kwargs)

            # Allocate weight to clusters.
            alpha_1 = 1 - lrisk / (lrisk + rrisk)

            # Weight constraints.
            alpha_1 = _hr_weight_bounds(upper_bound, lower_bound, weights, lc, rc, alpha_1)

            weights[lc] *= alpha_1
            weights[rc] *= 1 - alpha_1
        end
    end

    return weights
end

struct ClusterNode{tid, tl, tr, td, tcnt}
    id::tid
    left::tl
    right::tr
    dist::td
    count::tcnt

    function ClusterNode(id, left::Union{ClusterNode, Nothing} = nothing,
                         right::Union{ClusterNode, Nothing} = nothing, dist::Real = 0.0,
                         count::Int = 1)
        icount = isnothing(left) ? count : (left.count + right.count)

        return new{typeof(id), typeof(left), typeof(right), typeof(dist), typeof(count)}(id,
                                                                                         left,
                                                                                         right,
                                                                                         dist,
                                                                                         icount)
    end
end
export ClusterNode
import Base.>, Base.<, Base.==
<(a::ClusterNode, b::ClusterNode) = a.dist < b.dist
>(a::ClusterNode, b::ClusterNode) = a.dist > b.dist
==(a::ClusterNode, b::ClusterNode) = a.dist == b.dist
function is_leaf(a::ClusterNode)
    return isnothing(a.left)
end
function pre_order(a::ClusterNode, func::Function = x -> x.id)
    n = a.count
    curNode = Vector{ClusterNode}(undef, 2 * n)
    lvisited = Set()
    rvisited = Set()
    curNode[1] = a
    k = 1
    preorder = Int[]

    while k >= 1
        nd = curNode[k]
        ndid = nd.id
        if is_leaf(nd)
            push!(preorder, func(nd))
            k = k - 1
        else
            if ndid ∉ lvisited
                curNode[k + 1] = nd.left
                push!(lvisited, ndid)
                k = k + 1
            elseif ndid ∉ rvisited
                curNode[k + 1] = nd.right
                push!(rvisited, ndid)
                k = k + 1
                # If we've visited the left and right of this non-leaf
                # node already, go up in the tree.
            else
                k = k - 1
            end
        end
    end

    return preorder
end
function to_tree(a::Hclust)
    n = length(a.order)
    d = Vector{ClusterNode}(undef, 2 * n - 1)
    for i ∈ 1:n
        d[i] = ClusterNode(i)
    end
    merges = a.merges
    heights = a.heights
    nd = nothing

    for (i, height) ∈ pairs(heights)
        fi = merges[i, 1]
        fj = merges[i, 2]

        fi = fi < 0 ? -fi : fi + n
        fj = fj < 0 ? -fj : fj + n

        nd = ClusterNode(i + n, d[fi], d[fj], height)
        d[n + i] = nd
    end
    return nd, d
end

function _hierarchical_recursive_bisection(portfolio; rm = :SD, rm_o = rm, rf = 0.0,
                                           rf_o = rf, upper_bound = nothing,
                                           lower_bound = nothing,
                                           portfolio_kwargs::NamedTuple = (;
                                                                           alpha = portfolio.alpha,
                                                                           a_sim = portfolio.a_sim,
                                                                           beta = portfolio.beta,
                                                                           b_sim = portfolio.b_sim,
                                                                           kappa = portfolio.kappa,
                                                                           owa_w = portfolio.owa_w,
                                                                           solvers = portfolio.solvers),
                                           portfolio_kwargs_o = portfolio_kwargs)
    returns = portfolio.returns
    covariance = portfolio.cov
    clustering = portfolio.clusters

    k = portfolio.k
    root, nodes = to_tree(clustering)
    dists = [i.dist for i ∈ nodes]
    idx = sortperm(dists; rev = true)
    nodes = nodes[idx]

    weights = ones(size(portfolio.returns, 2))

    clustering_idx = cutree(clustering; k = k)

    uidx = minimum(clustering_idx):maximum(clustering_idx)

    clusters = Vector{Vector{Int}}(undef, length(uidx))
    for i ∈ eachindex(clusters)
        clusters[i] = findall(clustering_idx .== i)
    end

    # Treat each cluster as its own portfolio and optimise each one individually.
    # Calculate the weight of each cluster relative to the other clusters.
    for i ∈ nodes[1:(k - 1)]
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

        if rm == :Equal
            alpha_1 = 0.5
        else
            for j ∈ eachindex(clusters)
                if issubset(clusters[j], ln)
                    _lrisk = _cluster_risk(portfolio, returns, covariance, clusters[j];
                                           rm = rm, rf = rf,
                                           portfolio_kwargs = portfolio_kwargs)
                    lrisk += _lrisk
                    append!(lc, clusters[j])
                elseif issubset(clusters[j], rn)
                    _rrisk = _cluster_risk(portfolio, returns, covariance, clusters[j];
                                           rm = rm, rf = rf,
                                           portfolio_kwargs = portfolio_kwargs)
                    rrisk += _rrisk
                    append!(rc, clusters[j])
                end
            end

            alpha_1 = 1 - lrisk / (lrisk + rrisk)

            alpha_1 = _hr_weight_bounds(upper_bound, lower_bound, weights, lc, rc, alpha_1)
        end

        weights[ln] *= alpha_1
        weights[rn] *= 1 - alpha_1
    end

    # Treat each cluster as an asset in a portfolio and optimise.
    for i ∈ 1:k
        cidx = clustering_idx .== i
        cret = returns[:, cidx]
        ccov = covariance[cidx, cidx]
        cweights = _naive_risk(portfolio, cret, ccov; rm = rm_o, rf = rf_o,
                               portfolio_kwargs = portfolio_kwargs_o)
        weights[cidx] .*= cweights
    end

    return weights
end

function _intra_weights(portfolio, opt;
                        asset_stat_kwargs = (; calc_mu = false, calc_cov = false,
                                             calc_kurt = if opt.rm ∈ (:Kurt, :SKurt)
                                                 true
                                             else
                                                 false
                                             end),
                        portfolio_kwargs::NamedTuple = (; alpha = portfolio.alpha,
                                                        a_sim = portfolio.a_sim,
                                                        beta = portfolio.beta,
                                                        b_sim = portfolio.b_sim,
                                                        kappa = portfolio.kappa,
                                                        owa_w = portfolio.owa_w,
                                                        solvers = portfolio.solvers,
                                                        max_num_assets_kurt = portfolio.max_num_assets_kurt))
    returns = portfolio.returns
    mu = portfolio.mu
    covariance = portfolio.cov
    clustering = portfolio.clusters
    k = portfolio.k
    clustering_idx = cutree(clustering; k = k)

    intra_weights = zeros(eltype(covariance), size(portfolio.returns, 2), k)
    cfails = Dict{Int, Dict}()

    for i ∈ 1:k
        idx = clustering_idx .== i
        cmu = !isempty(mu) ? mu[idx] : Vector{eltype(returns)}(undef, 0)
        ccov = covariance[idx, idx]
        cret = returns[:, idx]
        cassets = portfolio.assets[idx]
        weights, cfail, success = _opt_w(portfolio, cassets, cret, cmu, ccov, opt;
                                         asset_stat_kwargs = asset_stat_kwargs,
                                         portfolio_kwargs = portfolio_kwargs)
        intra_weights[idx, i] .= weights
        if !success
            cfails[i] = cfail
        end
    end

    if !isempty(cfails)
        portfolio.fail[:intra] = cfails
    end

    return intra_weights
end

function _inter_weights(portfolio, intra_weights, opt;
                        asset_stat_kwargs = (; calc_mu = false, calc_cov = false,
                                             calc_kurt = if opt.rm ∈ (:Kurt, :SKurt)
                                                 true
                                             else
                                                 false
                                             end),
                        portfolio_kwargs::NamedTuple = (; alpha = portfolio.alpha,
                                                        a_sim = portfolio.a_sim,
                                                        beta = portfolio.beta,
                                                        b_sim = portfolio.b_sim,
                                                        kappa = portfolio.kappa,
                                                        owa_w = portfolio.owa_w,
                                                        solvers = portfolio.solvers,
                                                        max_num_assets_kurt = portfolio.max_num_assets_kurt))
    mu = portfolio.mu
    returns = portfolio.returns
    covariance = portfolio.cov
    tmu = !isempty(mu) ? transpose(intra_weights) * mu : Vector{eltype(returns)}(undef, 0)
    tcov = transpose(intra_weights) * covariance * intra_weights
    tret = returns * intra_weights
    inter_weights, inter_fail, success = _opt_w(portfolio, 1:size(tret, 2), tret, tmu, tcov,
                                                opt; asset_stat_kwargs = asset_stat_kwargs,
                                                portfolio_kwargs = portfolio_kwargs)
    weights = intra_weights * inter_weights
    if !success
        portfolio.fail[:inter] = inter_fail
    end

    return weights
end

function _nco_weights(portfolio, opt, opt_o;
                      asset_stat_kwargs = (; calc_mu = false, calc_cov = false,
                                           calc_kurt = if opt.rm ∈ (:Kurt, :SKurt)
                                               true
                                           else
                                               false
                                           end, kurt_opt = KurtOpt(;)),
                      asset_stat_kwargs_o = (; calc_mu = false, calc_cov = false,
                                             calc_kurt = if opt_o.rm ∈ (:Kurt, :SKurt)
                                                 true
                                             else
                                                 false
                                             end, kurt_opt = KurtOpt(;)),
                      portfolio_kwargs::NamedTuple = (; alpha = portfolio.alpha,
                                                      a_sim = portfolio.a_sim,
                                                      beta = portfolio.beta,
                                                      b_sim = portfolio.b_sim,
                                                      kappa = portfolio.kappa,
                                                      owa_w = portfolio.owa_w,
                                                      solvers = portfolio.solvers,
                                                      max_num_assets_kurt = portfolio.max_num_assets_kurt),
                      portfolio_kwargs_o::NamedTuple = portfolio_kwargs)

    # Treat each cluster as an independent portfolio and optimise each one individually.
    intra_weights = _intra_weights(portfolio, opt; asset_stat_kwargs = asset_stat_kwargs,
                                   portfolio_kwargs = portfolio_kwargs)

    # Treat each cluster as an asset in a portfolio and optimise the portfolio.
    weights = _inter_weights(portfolio, intra_weights, opt_o;
                             asset_stat_kwargs = asset_stat_kwargs_o,
                             portfolio_kwargs = portfolio_kwargs_o)

    return weights
end

function _setup_hr_weights(w_max, w_min, N)
    upper_bound = if isa(w_max, AbstractVector) && isempty(w_max)
        ones(N)
    elseif isa(w_max, AbstractVector) && !isempty(w_max)
        min.(1.0, w_max)
    else
        fill(min(1.0, w_max), N)
    end

    lower_bound = if isa(w_min, AbstractVector) && isempty(w_min)
        zeros(N)
    elseif isa(w_min, AbstractVector) && !isempty(w_min)
        max.(0.0, w_min)
    else
        fill(max(0.0, w_min), N)
    end

    @smart_assert(all(upper_bound .>= lower_bound))

    return upper_bound, lower_bound
end

function _opt_weight_bounds(upper_bound, lower_bound, weights, max_iter = 100)
    if !(any(upper_bound .< weights) || any(lower_bound .> weights))
        return weights
    end

    for _ ∈ 1:max_iter
        if !(any(upper_bound .< weights) || any(lower_bound .> weights))
            break
        end

        old_w = copy(weights)
        weights = max.(min.(weights, upper_bound), lower_bound)
        idx = weights .< upper_bound .&& weights .> lower_bound
        w_add = sum(max.(old_w - upper_bound, 0.0))
        w_sub = sum(min.(old_w - lower_bound, 0.0))
        delta = w_add + w_sub

        if delta != 0
            weights[idx] += delta * weights[idx] / sum(weights[idx])
        end
    end

    return weights
end

function _hcp_save_opt_params(portfolio, type, rm, rm_o, rf, rf_o, nco_opt, nco_opt_o,
                              cluster, cluster_opt, asset_stat_kwargs, asset_stat_kwargs_o,
                              portfolio_kwargs, portfolio_kwargs_o, max_iter,
                              save_opt_params)
    if !save_opt_params
        return nothing
    end

    Dict(:type => type, :rm => rm, :rm_o => rm_o, :rf => rf, :rf_o => rf_o,
         :nco_opt => nco_opt, :nco_opt_o => nco_opt_o, :cluster => cluster,
         :cluster_opt => cluster_opt, :asset_stat_kwargs => asset_stat_kwargs,
         :asset_stat_kwargs_o => asset_stat_kwargs_o, :portfolio_kwargs => portfolio_kwargs,
         :portfolio_kwargs_o => portfolio_kwargs_o, :max_iter => max_iter,
         :save_opt_params => save_opt_params)

    opt_params_dict = if type != :NCO
        Dict(:type => type, :rm => rm, :rm_o => rm_o, :rf => rf, :rf_o => rf_o,
             :cluster => cluster, :cluster_opt => cluster_opt,
             :asset_stat_kwargs => asset_stat_kwargs,
             :asset_stat_kwargs_o => asset_stat_kwargs_o,
             :portfolio_kwargs => portfolio_kwargs,
             :portfolio_kwargs_o => portfolio_kwargs_o, :max_iter => max_iter,
             :save_opt_params => save_opt_params)
    else
        Dict(:type => type, :rm => rm, :rm_o => rm_o, :rf => rf, :rf_o => rf_o,
             :nco_opt => nco_opt, :nco_opt_o => nco_opt_o, :cluster => cluster,
             :cluster_opt => cluster_opt, :asset_stat_kwargs => asset_stat_kwargs,
             :asset_stat_kwargs_o => asset_stat_kwargs_o,
             :portfolio_kwargs => portfolio_kwargs,
             :portfolio_kwargs_o => portfolio_kwargs_o, :max_iter => max_iter,
             :save_opt_params => save_opt_params)
    end

    portfolio.opt_params[type] = opt_params_dict

    return nothing
end

function _finalise_hcportfolio(portfolio, type, weights, upper_bound, lower_bound, max_iter)
    portfolio.optimal[type] = if !isempty(portfolio.fail) || any(.!isfinite.(weights))
        portfolio.fail[:portfolio] = DataFrame(; tickers = portfolio.assets,
                                               weights = weights)
        DataFrame()
    else
        weights = _opt_weight_bounds(upper_bound, lower_bound, weights, max_iter)
        weights ./= sum(weights)
        DataFrame(; tickers = portfolio.assets, weights = weights)
    end

    return portfolio.optimal[type]
end

function optimise!(portfolio::HCPortfolio; type::Symbol = :HRP, rm::Symbol = :SD,
                   rm_o::Symbol = rm, rf::Real = 0.0, rf_o::Real = rf,
                   nco_opt::OptimiseOpt = OptimiseOpt(;), nco_opt_o::OptimiseOpt = nco_opt,
                   cluster::Bool = true,
                   cluster_opt::ClusterOpt = ClusterOpt(; k = if cluster
                                                            0
                                                        else
                                                            portfolio.k
                                                        end,
                                                        max_k = ceil(Int,
                                                                     sqrt(size(portfolio.returns,
                                                                               2)))),
                   asset_stat_kwargs::NamedTuple = (; calc_mu = false, calc_cov = false,
                                                    calc_kurt = if type != :NCO &&
                                                                   rm ∈ (:Kurt, :SKurt) ||
                                                                   type == :NCO &&
                                                                   nco_opt.rm ∈
                                                                   (:Kurt, :SKurt)
                                                        true
                                                    else
                                                        false
                                                    end),
                   asset_stat_kwargs_o::NamedTuple = (; calc_mu = false, calc_cov = false,
                                                      calc_kurt = if type != :NCO &&
                                                                     rm_o ∈
                                                                     (:Kurt, :SKurt) ||
                                                                     type == :NCO &&
                                                                     nco_opt_o.rm ∈
                                                                     (:Kurt, :SKurt)
                                                          true
                                                      else
                                                          false
                                                      end),
                   portfolio_kwargs::NamedTuple = if type != :NCO
                       (; alpha = portfolio.alpha, a_sim = portfolio.a_sim,
                        beta = portfolio.beta, b_sim = portfolio.b_sim,
                        kappa = portfolio.kappa, owa_w = portfolio.owa_w,
                        solvers = portfolio.solvers)
                   else
                       (; alpha = portfolio.alpha, a_sim = portfolio.a_sim,
                        beta = portfolio.beta, b_sim = portfolio.b_sim,
                        kappa = portfolio.kappa, owa_w = portfolio.owa_w,
                        solvers = portfolio.solvers,
                        max_num_assets_kurt = portfolio.max_num_assets_kurt)
                   end, portfolio_kwargs_o::NamedTuple = portfolio_kwargs,
                   max_iter::Integer = 100, save_opt_params::Bool = false)
    @smart_assert(type ∈ HCPortTypes)
    @smart_assert(rm ∈ HCRiskMeasures)
    @smart_assert(rm_o ∈ HCRiskMeasures)
    portfolio.fail = Dict()

    _hcp_save_opt_params(portfolio, type, rm, rm_o, rf, rf_o, nco_opt, nco_opt_o, cluster,
                         cluster_opt, asset_stat_kwargs, asset_stat_kwargs_o,
                         portfolio_kwargs, portfolio_kwargs_o, max_iter, save_opt_params)

    N = size(portfolio.returns, 2)

    if cluster
        portfolio.clusters, tk = _hierarchical_clustering(portfolio, cluster_opt)
        portfolio.k = iszero(cluster_opt.k) ? tk : cluster_opt.k
    end

    upper_bound, lower_bound = _setup_hr_weights(portfolio.w_max, portfolio.w_min, N)

    weights = if type == :HRP
        _recursive_bisection(portfolio; rm = rm, rf = rf, upper_bound = upper_bound,
                             lower_bound = lower_bound, portfolio_kwargs = portfolio_kwargs)
    elseif type == :HERC
        _hierarchical_recursive_bisection(portfolio; rm = rm, rm_o = rm_o, rf = rf,
                                          rf_o = rf_o, portfolio_kwargs = portfolio_kwargs,
                                          portfolio_kwargs_o = portfolio_kwargs_o,
                                          upper_bound = upper_bound,
                                          lower_bound = lower_bound)
    else
        _nco_weights(portfolio, nco_opt, nco_opt_o; asset_stat_kwargs = asset_stat_kwargs,
                     asset_stat_kwargs_o = asset_stat_kwargs_o,
                     portfolio_kwargs = portfolio_kwargs,
                     portfolio_kwargs_o = portfolio_kwargs_o)
    end
    retval = _finalise_hcportfolio(portfolio, type, weights, upper_bound, lower_bound,
                                   max_iter)

    return retval
end
