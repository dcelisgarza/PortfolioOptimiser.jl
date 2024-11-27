function optimise!(port::OmniPortfolio, type::HRP)
    (; rm, class, max_iter) = type
    empty!(port.fail)
    sigma, returns = mu_sigma_returns_class(port, class)[2:3]
    lo, hi = w_limits(type, eltype(returns))
    w_min, w_max = set_hc_weights(port.w_min, port.w_max, size(port.returns, 2), lo, hi)
    w = hrp_optimise(port, rm, sigma, returns)
    return finalise_weights(type, port, w, w_min, w_max, max_iter)
end
function optimise!(port::OmniPortfolio, type::HERC)
    (; rm, rm_o, class, class_o, max_iter) = type
    empty!(port.fail)
    sigma, returns = mu_sigma_returns_class(port, class)[2:3]
    sigma_o, returns_o = if !isequal(class, class_o)
        mu_sigma_returns_class(port, class_o)[2:3]
    else
        sigma, returns
    end
    lo, hi = w_limits(type, eltype(returns))
    w_min, w_max = set_hc_weights(port.w_min, port.w_max, size(returns, 2), lo, hi)
    w = herc_optimise(port, rm, rm_o, sigma, sigma_o, returns, returns_o)
    return finalise_weights(type, port, w, w_min, w_max, max_iter)
end
function optimise!(port::OmniPortfolio, type::NCO)
    empty!(port.fail)
    lo, hi = w_limits(type, eltype(port.returns))
    w_min, w_max = set_hc_weights(port.w_min, port.w_max, size(port.returns, 2), lo, hi)
    w = nco_optimise(port, type)
    return finalise_weights(type, port, w, w_min, w_max, type.max_iter)
end