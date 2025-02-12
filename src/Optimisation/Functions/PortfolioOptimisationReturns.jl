# Copywrite (c) 2025
# Author: Daniel Celis Garza <daniel.celis.garza@gmail.com>
# SPDX-License-Identifier: MIT

function return_bounds(port)
    mu_l = port.mu_l
    if isinf(mu_l)
        return nothing
    end

    model = port.model
    scale_constr = model[:scale_constr]
    k = model[:k]
    ret = model[:ret]
    @constraint(model, constr_ret_lb, scale_constr * ret >= scale_constr * mu_l * k)

    return nothing
end
function sharpe_returns_constraints(port, obj::Sharpe, mu)
    model = port.model
    scale_constr = model[:scale_constr]
    k = model[:k]
    ohf = model[:ohf]
    ret = model[:ret]
    rf = obj.rf
    if all(mu .<= zero(eltype(mu))) || haskey(model, :abs_w) || haskey(model, :t_gw)
        risk = model[:risk]
        add_to_expression!(ret, -rf, k)
        @constraint(model, constr_sr_risk, scale_constr * risk <= scale_constr * ohf)
    else
        @constraint(model, constr_sr_ret,
                    scale_constr * (ret - rf * k) == scale_constr * ohf)
    end
    return nothing
end
function sharpe_returns_constraints(args...)
    return nothing
end
function wc_return_constraints(port, mu, ::Box)
    model = port.model
    get_fees(model)
    scale_constr = model[:scale_constr]
    w = model[:w]
    fees = model[:fees]
    N = length(mu)
    @variable(model, abs_w[1:N])
    @constraint(model, constr_box_wc_ret_set[i = 1:N],
                [scale_constr * abs_w[i]; scale_constr * w[i]] ∈ MOI.NormOneCone(2))
    @expression(model, ret, dot(mu, w) - fees - dot(port.d_mu, abs_w))
    return nothing
end
function wc_return_constraints(port, mu, ::Ellipse)
    model = port.model
    get_fees(model)
    scale_constr = model[:scale_constr]
    w = model[:w]
    fees = model[:fees]
    G = sqrt(port.cov_mu)
    k_mu = port.k_mu
    @expression(model, x_gw, G * w)
    @variable(model, t_gw)
    @constraint(model, constr_ellip_wc_ret_set,
                [scale_constr * t_gw; scale_constr * x_gw] ∈ SecondOrderCone())
    @expression(model, ret, dot(mu, w) - fees - k_mu * t_gw)
    return nothing
end
function wc_return_constraints(port, mu, ::NoWC)
    model = port.model
    get_fees(model)
    fees = model[:fees]
    w = port.model[:w]
    @expression(model, ret, dot(mu, w) - fees)
    return nothing
end
function return_constraints(port, obj, kelly::NoKelly, mu, args...)
    if isempty(mu)
        return nothing
    end

    wc_return_constraints(port, mu, kelly.wc_set)
    sharpe_returns_constraints(port, obj, mu)
    return_bounds(port)

    return nothing
end
function return_constraints(port, ::Any, kelly::AKelly, mu, sigma, returns,
                            kelly_approx_idx)
    if isempty(mu)
        return nothing
    end

    model = port.model
    get_fees(model)
    w = model[:w]
    fees = model[:fees]
    if isnothing(kelly_approx_idx) ||
       isempty(kelly_approx_idx) ||
       iszero(kelly_approx_idx[1])
        if !haskey(model, :variance_risk)
            calc_variance_risk(get_ntwk_clust_type(port), kelly.formulation, model, mu,
                               sigma, returns)
        end
        variance_risk = model[:variance_risk]
        @expression(model, ret, dot(mu, w) - fees - 0.5 * variance_risk)
    else
        variance_risk = model[:variance_risk]
        @expression(model, ret,
                    dot(mu, w) - fees - 0.5 * variance_risk[kelly_approx_idx[1]])
    end

    return_bounds(port)

    return nothing
end
function return_constraints(port, obj::Sharpe, kelly::AKelly, mu, sigma, returns,
                            kelly_approx_idx)
    return_sharpe_akelly_constraints(port, obj, kelly, get_ntwk_clust_type(port), mu, sigma,
                                     returns, kelly_approx_idx)
    return nothing
end
function return_sharpe_akelly_constraints(port, obj::Sharpe, kelly::AKelly,
                                          adjacency_constraint::Union{NoAdj, IP}, mu, sigma,
                                          returns, kelly_approx_idx)
    if isempty(mu)
        return nothing
    end

    model = port.model
    get_fees(model)
    scale_constr = model[:scale_constr]
    w = model[:w]
    k = model[:k]
    fees = model[:fees]
    ohf = model[:ohf]
    risk = model[:risk]
    rf = obj.rf
    @variable(model, tapprox_kelly)
    @constraint(model, constr_sr_akelly_risk, scale_constr * risk <= scale_constr * ohf)
    @expression(model, ret, dot(mu, w) - fees - 0.5 * tapprox_kelly - k * rf)
    if isnothing(kelly_approx_idx) ||
       isempty(kelly_approx_idx) ||
       iszero(kelly_approx_idx[1])
        if !haskey(model, :variance_risk)
            calc_variance_risk(adjacency_constraint, kelly.formulation, model, mu, sigma,
                               returns)
        end
        dev = model[:dev]
        @constraint(model, constr_sr_akelly_ret,
                    [scale_constr * (k + tapprox_kelly)
                     scale_constr * 2 * dev
                     scale_constr * (k - tapprox_kelly)] ∈ SecondOrderCone())
    else
        dev = model[:dev]
        @constraint(model, constr_sr_akelly_ret,
                    [scale_constr * (k + tapprox_kelly)
                     scale_constr * 2 * dev[kelly_approx_idx[1]]
                     scale_constr * (k - tapprox_kelly)] ∈ SecondOrderCone())
    end
    return_bounds(port)

    return nothing
end
function return_sharpe_akelly_constraints(port, obj::Sharpe, ::AKelly, ::SDP, ::Any, ::Any,
                                          returns, ::Any)
    return_constraints(port, obj, EKelly(), nothing, nothing, returns, nothing)
    return nothing
end
function return_sharpe_ekelly_constraints(ret, model, obj::Sharpe, k)
    scale_constr = model[:scale_constr]
    ohf = model[:ohf]
    risk = model[:risk]
    rf = obj.rf
    add_to_expression!(ret, -k, rf)
    @constraint(model, constr_sr_ekelly_risk, scale_constr * risk <= scale_constr * ohf)
    return nothing
end
function return_sharpe_ekelly_constraints(args...)
    return nothing
end
function return_constraints(port, obj, ::EKelly, ::Any, ::Any, returns, ::Any)
    model = port.model
    get_fees(model)
    scale_constr = model[:scale_constr]
    w = model[:w]
    k = model[:k]
    fees = model[:fees]
    T = size(returns, 1)
    @variable(model, texact_kelly[1:T])
    @expression(model, ret, sum(texact_kelly) / T - fees)
    return_sharpe_ekelly_constraints(ret, model, obj, k)
    @expression(model, kret, k .+ returns * w)
    @constraint(model, constr_ekelly_ret[i = 1:T],
                [scale_constr * texact_kelly[i], scale_constr * k, scale_constr * kret[i]] ∈
                MOI.ExponentialCone())
    return_bounds(port)

    return nothing
end
function expected_return_constraints(port, obj, kelly, mu, sigma, returns, kelly_approx_idx)
    return_constraints(port, obj, kelly, mu, sigma, returns, kelly_approx_idx)
    return nothing
end
