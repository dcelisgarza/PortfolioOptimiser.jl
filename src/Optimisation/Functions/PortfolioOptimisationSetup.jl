# Copywrite (c) 2025
# Author: Daniel Celis Garza <daniel.celis.garza@gmail.com>
# SPDX-License-Identifier: MIT

function set_w_ini(w, w_ini)
    if !isempty(w_ini)
        @smart_assert(length(w_ini) == length(w))
        set_start_value.(w, w_ini)
    end
    return nothing
end
function initial_w(port, w_ini)
    model = port.model
    N = size(port.returns, 2)
    @variable(model, w[1:N])
    set_w_ini(w, w_ini)
    return nothing
end
function set_scale_obj_constrs(port)
    model = port.model
    scale_constr = port.scale_constr
    scale_obj = port.scale_obj
    @expression(model, scale_constr, scale_constr)
    @expression(model, scale_obj, scale_obj)
    return nothing
end
function mu_sigma_returns_class(port, ::Union{Classic, FC})
    return port.mu, port.cov, port.returns
end
function mu_sigma_returns_class(port, class::FM)
    mu = port.fm_mu
    if class.type == 1
        sigma = port.fm_cov
        returns = port.fm_returns
    else
        sigma = port.cov
        returns = port.returns
    end
    return mu, sigma, returns
end
function mu_sigma_returns_class(port, class::BL)
    mu = port.bl_mu
    returns = port.returns
    if class.type == 1
        sigma = port.bl_cov
    else
        sigma = port.cov
    end
    return mu, sigma, returns
end
function mu_sigma_returns_class(port, class::BLFM)
    mu = port.blfm_mu
    if class.type == 1
        sigma = port.blfm_cov
        returns = port.fm_returns
    elseif class.type == 2
        sigma = port.cov
        returns = port.returns
    else
        sigma = port.fm_cov
        returns = port.fm_returns
    end
    return mu, sigma, returns
end
function optimal_homogenisation_factor(port, mu, obj::Sharpe)
    ohf = obj.ohf
    if iszero(ohf)
        ohf = if !isempty(mu)
            min(1e3, max(1e-3, mean(abs.(mu))))
        else
            one(eltype(port.returns))
        end
    end
    model = port.model
    @expression(model, ohf, ohf)
    return nothing
end
function optimal_homogenisation_factor(args...)
    return nothing
end
function set_k(port, ::Sharpe)
    model = port.model
    @variable(model, k >= 0)
    return nothing
end
function set_k(port, ::Any)
    model = port.model
    @expression(model, k, 1)
    return nothing
end
