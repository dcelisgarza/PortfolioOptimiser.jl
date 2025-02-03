# Copywrite (c) 2025
# Author: Daniel Celis Garza <daniel.celis.garza@gmail.com>
# SPDX-License-Identifier: MIT

function optimise!(port::Portfolio, type::HRP)
    (; rm, class, scalarisation, finaliser) = type
    empty!(port.fail)
    mu, sigma, returns = mu_sigma_returns_class(port, class)
    lo, hi = w_limits(type, eltype(returns))
    w_min, w_max = set_hc_weights(port.w_min, port.w_max, size(returns, 2), lo, hi)
    w = hrp_optimise(port, rm, mu, sigma, returns, scalarisation)
    return finalise_weights(type, port, w, w_min, w_max, finaliser)
end
function optimise!(port::Portfolio, type::HERC)
    (; rm, rm_o, class, class_o, scalarisation, scalarisation_o, finaliser) = type
    empty!(port.fail)
    mu, sigma, returns = mu_sigma_returns_class(port, class)
    mu_o, sigma_o, returns_o = if !isequal(class, class_o)
        mu_sigma_returns_class(port, class_o)
    else
        mu, sigma, returns
    end
    lo, hi = w_limits(type, eltype(returns))
    w_min, w_max = set_hc_weights(port.w_min, port.w_max, size(returns, 2), lo, hi)
    w = herc_optimise(port, rm, rm_o, mu, mu_o, sigma, sigma_o, returns, returns_o,
                      scalarisation, scalarisation_o)
    return finalise_weights(type, port, w, w_min, w_max, finaliser)
end
function optimise!(port::Portfolio, type::NCO)
    if isa(type.external, AbstractVector)
        @smart_assert(length(type.external) == port.k)
    end
    empty!(port.fail)
    lo, hi = w_limits(port, type)
    w_min, w_max = set_hc_weights(port.w_min, port.w_max, size(port.returns, 2), lo, hi)
    w = nco_optimise(port, type)
    return finalise_weights(type, port, w, w_min, w_max, type.finaliser)
end
function optimise!(port::Portfolio, type::SchurHRP)
    (; params, class, finaliser) = type
    empty!(port.fail)
    sigma = mu_sigma_returns_class(port, class)[2]
    lo, hi = w_limits(type, eltype(port.returns))
    w_min, w_max = set_hc_weights(port.w_min, port.w_max, size(sigma, 1), lo, hi)
    w = schur_optimise(port, params, sigma)
    return finalise_weights(type, port, w, w_min, w_max, finaliser)
end
