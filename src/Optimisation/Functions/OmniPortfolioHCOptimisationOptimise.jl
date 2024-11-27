function optimise!(port::OmniPortfolio, type::HRP;
                   rm::Union{AbstractVector, <:AbstractRiskMeasure} = Variance(),
                   class::PortClass = Classic(), max_iter::Int = 100, kwargs...)
    empty!(port.fail)
    sigma, returns = mu_sigma_returns_class(port, class)[2:3]
    lo, hi = w_limits(type, eltype(returns))
    w_min, w_max = set_hc_weights(port.w_min, port.w_max, size(port.returns, 2), lo, hi)
    w = hrp_optimise(port, rm, sigma, returns)
    return finalise_weights(type, port, w, w_min, w_max, max_iter)
end
function optimise!(port::OmniPortfolio, type::HERC;
                   rm::Union{AbstractVector, <:AbstractRiskMeasure} = Variance(),
                   rm_o::Union{AbstractVector, <:AbstractRiskMeasure} = rm,
                   class::PortClass = Classic(), class_o::PortClass = class,
                   max_iter::Int = 100, kwargs...)
    empty!(port.fail)
    sigma, returns = mu_sigma_returns_class(port, class)[2:3]
    sigma_o, returns_o = if !isequal(class, class_o)
        mu_sigma_returns_class(port, class_o)[2:3]
    else
        sigma, returns
    end
    lo, hi = w_limits(type, eltype(returns))
    w_min, w_max = set_hc_weights(port.w_min, port.w_max, size(port.returns, 2), lo, hi)
    w = herc_optimise(port, rm, rm_o, sigma, sigma_o, returns, returns_o)
    return finalise_weights(type, port, w, w_min, w_max, max_iter)
end
function optimise!(port::OmniPortfolio, type::NCO; max_iter::Int = 100,
                   rm::Union{AbstractVector, <:AbstractRiskMeasure} = Variance(),
                   rm_o::Union{AbstractVector, <:AbstractRiskMeasure} = rm, kwargs...)
    empty!(port.fail)
    lo, hi = w_limits(type, eltype(port.returns))
    w_min, w_max = set_hc_weights(port.w_min, port.w_max, size(port.returns, 2), lo, hi)
    w = nco_optimise(port, type, rm, rm_o)
    return finalise_weights(type, port, w, w_min, w_max, max_iter)
end