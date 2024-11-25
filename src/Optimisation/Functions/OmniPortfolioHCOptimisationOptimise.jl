function optimise!(port::OmniPortfolio, type::HRP;
                   rm::Union{AbstractVector, <:AbstractRiskMeasure} = Variance(),
                   class::PortClass = Classic(), max_iter::Int = 100)
    empty!(port.fail)
    sigma, returns = mu_sigma_returns_class(port, class)[2:3]
    lo, hi = w_limits(type, eltype(returns))
    w_min, w_max = set_hc_weights(port.w_min, port.w_max, size(port.returns, 2), lo, hi)
    w = hrp_optimise(port, rm, sigma, returns)
    return finalise_weights(type, port, w, w_min, w_max, max_iter)
end
function optimise!(port::OmniPortfolio, type::HERC;
                   rm_i::Union{AbstractVector, <:AbstractRiskMeasure} = Variance(),
                   rm_o::Union{AbstractVector, <:AbstractRiskMeasure} = rm_i,
                   class::PortClass = Classic(), max_iter::Int = 100)
    empty!(port.fail)
    sigma, returns = mu_sigma_returns_class(port, class)[2:3]
    lo, hi = w_limits(type, eltype(returns))
    w_min, w_max = set_hc_weights(port.w_min, port.w_max, size(port.returns, 2), lo, hi)
    w = herc_optimise(port, rm_i, rm_o, sigma, returns)
    return finalise_weights(type, port, w, w_min, w_max, max_iter)
end
function optimise!(port::OmniPortfolio, type::NCO;
                   rm_i::Union{AbstractVector, <:AbstractRiskMeasure} = Variance(),
                   rm_o::Union{AbstractVector, <:AbstractRiskMeasure} = rm_i,
                   class::PortClass = Classic(), max_iter::Int = 100)
    empty!(port.fail)
    mu, sigma, returns = mu_sigma_returns_class(port, class)
    lo, hi = w_limits(type, eltype(returns))
    w_min, w_max = set_hc_weights(port.w_min, port.w_max, size(returns, 2), lo, hi)
    w = nco_optimise(port, type, rm_i, rm_o, mu, sigma, returns)
    return finalise_weights(type, port, w, w_min, w_max, max_iter)
end
