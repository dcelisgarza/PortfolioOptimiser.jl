function nco_optimise(port::OmniPortfolio, type::NCO,
                      rm_i::Union{AbstractVector, <:AbstractRiskMeasure},
                      rm_o::Union{AbstractVector, <:AbstractRiskMeasure}, mu, sigma,
                      returns)
    wi = calc_intra_weights(port, mu, sigma, returns, rm_i, type.opt_kwargs,
                            type.port_kwargs, type.factor_kwargs, type.wc_kwargs,
                            type.cluster_kwargs)
    w = calc_inter_weights(port, mu, sigma, returns, wi, rm_o, type.opt_kwargs_o,
                           type.port_kwargs_o, type.factor_kwargs_o, type.wc_kwargs_o,
                           type.cluster_kwargs_o, type.stat_kwargs_o)

    return w
end