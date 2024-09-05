function intra_nco_opt(port, rm, cassets, cret, cmu, ccov, ckurt, cskurt, cV, cSV, obj,
                       kelly, opt_kwargs, port_kwargs)
    intra_port = Portfolio(; assets = cassets, ret = cret, mu = cmu, cov = ccov,
                           kurt = ckurt, skurt = cskurt, V = cV, SV = cSV,
                           solvers = port.solvers, port_kwargs...)
    if !isempty(ckurt) || !isempty(cskurt)
        intra_port.L_2, intra_port.S_2 = dup_elim_sum_matrices(size(cret, 2))[2:3]
    end

    w = optimise!(intra_port; rm = rm, obj = obj, kelly = kelly, opt_kwargs...)
    if !isempty(w)
        w = w.weights
    else
        w = zeros(eltype(cret), size(cret, 2))
    end

    return w, intra_port.fail
end
function calc_intra_weights(port, rm::Union{AbstractVector, <:TradRiskMeasure},
                            obj::ObjectiveFunction, kelly::RetType, opt_kwargs, port_kwargs)
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
                                  cSV, obj, kelly, opt_kwargs, port_kwargs)
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
                       set_sskew, obj, kelly, opt_kwargs, port_kwargs, stat_kwargs)
    inter_port = Portfolio(; assets = cassets, ret = cret, mu = cmu, cov = ccov,
                           solvers = port.solvers, port_kwargs...)
    asset_statistics!(inter_port; set_cov = false, set_mu = false, set_kurt = set_kurt,
                      set_skurt = set_skurt, set_skew = set_skew, set_sskew = set_sskew,
                      stat_kwargs...)

    w = optimise!(inter_port; rm = rm, obj = obj, kelly = kelly, opt_kwargs...)

    if !isempty(w)
        w = w.weights
    else
        w = zeros(eltype(cret), size(cret, 2))
    end

    return w, inter_port.fail
end
function calc_inter_weights(port, wi, rm, obj, kelly, opt_kwargs, port_kwargs, stat_kwargs)
    cret = port.returns * wi
    cmu = transpose(wi) * port.mu
    ccov = transpose(wi) * port.cov * wi

    set_kurt, set_skurt, set_skew, set_sskew = find_kurt_skew_rm(rm)
    cw, cfail = inter_nco_opt(port, rm, 1:size(cret, 2), cret, cmu, ccov, set_kurt,
                              set_skurt, set_skew, set_sskew, obj, kelly, opt_kwargs,
                              port_kwargs, stat_kwargs)

    w = wi * cw

    if !isempty(cfail)
        port.fail[:inter] = cfail
    end

    return w
end
function _optimise!(type::NCO, port::HCPortfolio, rmi::Union{AbstractVector, <:RiskMeasure},
                    rmo::Union{AbstractVector, <:RiskMeasure}, obji::ObjectiveFunction,
                    objo::ObjectiveFunction, kellyi::RetType, kellyo::RetType, ::Any, ::Any)
    port.fail = Dict()
    wi = calc_intra_weights(port, rmi, obji, kellyi, type.opt_kwargs, type.port_kwargs)
    w = calc_inter_weights(port, wi, rmo, objo, kellyo, type.opt_kwargs_o,
                           type.port_kwargs_o, type.stat_kwargs_o)

    return w
end
