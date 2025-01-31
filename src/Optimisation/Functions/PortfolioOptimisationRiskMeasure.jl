# Copywrite (c) 2025
# Author: Daniel Celis Garza <daniel.celis.garza@gmail.com>
# SPDX-License-Identifier: MIT

# Risk expression
function set_risk_expression(model, rm_risk, scale, flag::Bool)
    if !flag
        return nothing
    end

    if !haskey(model, :risk_vec)
        @expression(model, risk_vec, Union{AffExpr, QuadExpr}[])
    end

    risk_vec = model[:risk_vec]
    push!(risk_vec, scale * rm_risk)

    return nothing
end
function scalarise_risk_expression(port, ::ScalarSum)
    model = port.model
    risk_vec = model[:risk_vec]

    if any(isa.(risk_vec, QuadExpr))
        @expression(model, risk, zero(QuadExpr))
    else
        @expression(model, risk, zero(AffExpr))
    end

    for rm_risk ∈ risk_vec
        add_to_expression!(risk, rm_risk)
    end

    return nothing
end
function scalarise_risk_expression(port, scalarisation::ScalarLogSumExp)
    model = port.model
    risk_vec = model[:risk_vec]
    scale_constr = model[:scale_constr]
    N = length(risk_vec)
    gamma = scalarisation.gamma

    @variable(model, risk)
    @variable(model, ulse_risk[1:N])
    @constraint(model, constr_scalar_risk_log_sum_exp_u,
                scale_constr * sum(ulse_risk) <= scale_constr * 1)
    @constraint(model, constr_scalar_risk_log_sum_exp[i = 1:N],
                [scale_constr * gamma * (risk_vec[i] - risk), scale_constr * 1,
                 scale_constr * ulse_risk[i]] in MOI.ExponentialCone())

    return nothing
end
function scalarise_risk_expression(port, ::ScalarMax)
    model = port.model
    risk_vec = model[:risk_vec]

    @variable(model, risk)
    @constraint(model, constr_scalar_risk_max, risk .>= risk_vec)

    return nothing
end
function get_ntwk_clust_type(port)
    model = port.model
    return if haskey(model, :constr_ntwk_sdp) || haskey(model, :constr_clst_sdp)
        SDP()
    else
        NoAdj()
    end
end
function set_rm_risk_upper_bound(args...)
    return nothing
end
function set_rm_risk_upper_bound(::Union{Trad, NOC}, model, rm_risk, ub, key)
    if isinf(ub)
        return nothing
    end

    k = model[:k]
    scale_constr = model[:scale_constr]
    model[Symbol("$(key)_ub")] = @constraint(model,
                                             scale_constr * rm_risk .<=
                                             scale_constr * ub * k)

    return nothing
end
function calc_variance_risk(::SDP, ::Any, model, sigma)
    W = model[:W]
    @expression(model, variance_risk, tr(sigma * W))
    return nothing
end
function calc_variance_risk(::SDP, model::JuMP.Model, sigma, count::Integer)
    @expression(model, variance_risk[1:count], zero(AffExpr))
    return nothing
end
function calc_variance_risk(::SDP, ::Any, model, sigma, idx::Integer)
    W = model[:W]
    variance_risk = model[:variance_risk]
    add_to_expression!(variance_risk[idx], tr(sigma * W))
    return nothing
end
function calc_variance_risk(::Union{NoAdj, IP}, ::SOC, model::JuMP.Model,
                            sigma::AbstractMatrix)
    scale_constr = model[:scale_constr]
    w = model[:w]
    G = sqrt(sigma)
    @variable(model, dev)
    @expression(model, variance_risk, dev^2)
    @constraint(model, constr_dev_soc,
                [scale_constr * dev; scale_constr * G * w] ∈ SecondOrderCone())
    return nothing
end
function calc_variance_risk(::Union{NoAdj, IP}, model::JuMP.Model, sigma::AbstractMatrix,
                            count::Integer)
    @variable(model, dev[1:count])
    @expression(model, variance_risk[1:count], zero(QuadExpr))
    return nothing
end
function calc_variance_risk(::Union{NoAdj, IP}, ::SOC, model::JuMP.Model,
                            sigma::AbstractMatrix, idx::Integer)
    scale_constr = model[:scale_constr]
    w = model[:w]
    dev = model[:dev][idx]
    variance_risk = model[:variance_risk][idx]
    G = sqrt(sigma)
    add_to_expression!(variance_risk, dev, dev)
    model[Symbol("constr_dev_soc_$(idx)")] = @constraint(model,
                                                         [scale_constr * dev;
                                                          scale_constr * G * w] ∈
                                                         SecondOrderCone())
    return nothing
end
function calc_variance_risk(::Union{NoAdj, IP}, ::Quad, model, sigma)
    scale_constr = model[:scale_constr]
    w = model[:w]
    G = sqrt(sigma)
    @variable(model, dev)
    @expression(model, variance_risk, dot(w, sigma, w))
    @constraint(model, constr_dev_soc,
                [scale_constr * dev; scale_constr * G * w] ∈ SecondOrderCone())
    return nothing
end
function calc_variance_risk(::Union{NoAdj, IP}, ::Quad, model::JuMP.Model,
                            sigma::AbstractMatrix, idx::Integer)
    scale_constr = model[:scale_constr]
    w = model[:w]
    dev = model[:dev][idx]
    variance_risk = model[:variance_risk][idx]
    G = sqrt(sigma)
    add_to_expression!(variance_risk, dot(w, sigma, w))
    model[Symbol("constr_dev_soc_$(idx)")] = @constraint(model,
                                                         [scale_constr * dev;
                                                          scale_constr * G * w] ∈
                                                         SecondOrderCone())
    return nothing
end
function variance_risk_bounds_expr(::SDP, model)
    return model[:variance_risk], "variance_risk"
end
function variance_risk_bounds_expr(::Union{NoAdj, IP}, model)
    return model[:dev], "dev"
end
function variance_risk_bounds_val(::SDP, ub)
    return ub
end
function variance_risk_bounds_val(::Union{NoAdj, IP}, ub)
    return sqrt(ub)
end
"""
```
set_rm(port, rm::RiskMeasure, type::Union{Trad, RB, NOC}; kwargs...)
set_rm(port, rm::AbstractVector{<:RiskMeasure}, type::Union{Trad, RB, NOC}; kwargs...)
```
"""
function set_rm(port, rm::Variance, type::Union{Trad, RB, NOC};
                sigma::AbstractMatrix{<:Real},
                kelly_approx_idx::Union{AbstractVector{<:Integer}, Nothing}, kwargs...)
    model = port.model
    use_portfolio_sigma = isnothing(rm.sigma) || isempty(rm.sigma)
    if !isnothing(kelly_approx_idx) && use_portfolio_sigma
        if isempty(kelly_approx_idx)
            push!(kelly_approx_idx, 0)
        end
    end
    if !use_portfolio_sigma
        sigma = rm.sigma
    end
    adjacency_constraint = get_ntwk_clust_type(port)
    calc_variance_risk(adjacency_constraint, rm.formulation, model, sigma)
    variance_risk = model[:variance_risk]
    var_bound_expr, var_bound_key = variance_risk_bounds_expr(adjacency_constraint, model)
    ub = variance_risk_bounds_val(adjacency_constraint, rm.settings.ub)
    set_rm_risk_upper_bound(type, model, var_bound_expr, ub, var_bound_key)
    set_risk_expression(model, model[:variance_risk], rm.settings.scale, rm.settings.flag)
    return nothing
end
function set_rm(port, rms::AbstractVector{<:Variance}, type::Union{Trad, RB, NOC};
                sigma::AbstractMatrix{<:Real},
                kelly_approx_idx::Union{AbstractVector{<:Integer}, Nothing}, kwargs...)
    model = port.model
    adjacency_constraint = get_ntwk_clust_type(port)
    count = length(rms)
    calc_variance_risk(adjacency_constraint, model, sigma, count)
    variance_risk = model[:variance_risk]
    var_bound_expr, var_bound_key = variance_risk_bounds_expr(adjacency_constraint, model)
    for (i, rm) ∈ pairs(rms)
        use_portfolio_sigma = (isnothing(rm.sigma) || isempty(rm.sigma))
        if !isnothing(kelly_approx_idx) && use_portfolio_sigma
            if isempty(kelly_approx_idx)
                push!(kelly_approx_idx, i)
            end
        end
        if !use_portfolio_sigma
            sigma = rm.sigma
        end
        calc_variance_risk(adjacency_constraint, rm.formulation, model, sigma, i)
        ub = variance_risk_bounds_val(adjacency_constraint, rm.settings.ub)
        set_rm_risk_upper_bound(type, model, var_bound_expr[i], ub, "$(var_bound_key)_$(i)")
        set_risk_expression(model, variance_risk[i], rm.settings.scale, rm.settings.flag)
    end
    return nothing
end
function choose_wc_stats_port_rm(port, rm)
    sigma = if !(isnothing(rm.sigma) || isempty(rm.sigma))
        rm.sigma
    else
        port.cov
    end

    cov_l = if !(isnothing(rm.cov_l) || isempty(rm.cov_l))
        rm.cov_l
    else
        port.cov_l
    end

    cov_u = if !(isnothing(rm.cov_u) || isempty(rm.cov_u))
        rm.cov_u
    else
        port.cov_u
    end

    cov_sigma = if !(isnothing(rm.cov_sigma) || isempty(rm.cov_sigma))
        rm.cov_sigma
    else
        port.cov_sigma
    end

    k_sigma = if isfinite(rm.k_sigma)
        rm.k_sigma
    else
        port.k_sigma
    end

    return sigma, cov_l, cov_u, cov_sigma, k_sigma
end
function wc_variance_risk_variables(::Box, model)
    if haskey(model, :Au)
        return nothing
    end
    scale_constr = model[:scale_constr]
    W = model[:W]
    N = size(W, 1)
    @variables(model, begin
                   Au[1:N, 1:N] .>= 0, Symmetric
                   Al[1:N, 1:N] .>= 0, Symmetric
               end)
    @constraint(model, constr_box_wc_variance_set,
                scale_constr * (Au .- Al) .== scale_constr * W)
    return nothing
end
function wc_variance_risk_variables(::Ellipse, model)
    if haskey(model, :E)
        return nothing
    end
    scale_constr = model[:scale_constr]
    W = model[:W]
    N = size(W, 1)
    @variable(model, E[1:N, 1:N], Symmetric)
    @expression(model, WpE, W .+ E)
    @constraint(model, constr_ellipse_wc_variance_set, scale_constr * E ∈ PSDCone())
    return nothing
end
function calc_wc_variance_risk(::Box, model, sigma, cov_l, cov_u, cov_sigma, k_sigma)
    Au = model[:Au]
    Al = model[:Al]
    @expression(model, wc_variance_risk, tr(Au * cov_u) - tr(Al * cov_l))
    return nothing
end
function calc_wc_variance_risk(::Ellipse, model, sigma, cov_l, cov_u, cov_sigma, k_sigma)
    scale_constr = model[:scale_constr]
    WpE = model[:WpE]
    G_sigma = sqrt(cov_sigma)
    @variable(model, t_ge)
    @expressions(model, begin
                     x_ge, G_sigma * vec(WpE)
                     wc_variance_risk, tr(sigma * WpE) + k_sigma * t_ge
                 end)
    @constraint(model, constr_ge_soc,
                [scale_constr * t_ge; scale_constr * x_ge] ∈ SecondOrderCone())
    return nothing
end
function calc_wc_variance_risk(::Box, model, sigma, cov_l, cov_u, cov_sigma, k_sigma,
                               wc_variance_risk, ::Any)
    Au = model[:Au]
    Al = model[:Al]
    add_to_expression!(wc_variance_risk, tr(Au * cov_u) - tr(Al * cov_l))
    return nothing
end
function calc_wc_variance_risk(::Ellipse, model, sigma, cov_l, cov_u, cov_sigma, k_sigma,
                               wc_variance_risk, i)
    scale_constr = model[:scale_constr]
    WpE = model[:WpE]
    G_sigma = sqrt(cov_sigma)
    model[Symbol("t_ge_$(i)")] = t_ge = @variable(model)
    model[Symbol("x_ge_$(i)")] = x_ge = @expression(model, G_sigma * vec(WpE))
    add_to_expression!(wc_variance_risk, tr(sigma * WpE))
    add_to_expression!(wc_variance_risk, k_sigma, t_ge)
    model[Symbol("constr_wc_variance_risk_$(i)")] = @constraint(model,
                                                                [scale_constr * t_ge;
                                                                 scale_constr * x_ge] ∈
                                                                SecondOrderCone())
    return nothing
end
function set_rm(port, rm::WCVariance, type::Union{Trad, RB, NOC};
                sigma::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
    SDP_constraints(model, type)
    sigma, cov_l, cov_u, cov_sigma, k_sigma = choose_wc_stats_port_rm(port, rm)
    wc_variance_risk_variables(rm.wc_set, model)
    calc_wc_variance_risk(rm.wc_set, model, sigma, cov_l, cov_u, cov_sigma, k_sigma)
    wc_variance_risk = model[:wc_variance_risk]
    set_rm_risk_upper_bound(type, model, wc_variance_risk, rm.settings.ub,
                            "wc_variance_risk")
    set_risk_expression(model, wc_variance_risk, rm.settings.scale, rm.settings.flag)
    return nothing
end
function set_rm(port, rms::AbstractVector{<:WCVariance}, type::Union{Trad, RB, NOC};
                sigma::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
    SDP_constraints(model, type)
    count = length(rms)
    @expression(model, wc_variance_risk[1:count], zero(AffExpr))
    for (i, rm) ∈ pairs(rms)
        sigma, cov_l, cov_u, cov_sigma, k_sigma = choose_wc_stats_port_rm(port, rm)
        wc_variance_risk_variables(rm.wc_set, model)
        calc_wc_variance_risk(rm.wc_set, model, sigma, cov_l, cov_u, cov_sigma, k_sigma,
                              wc_variance_risk[i], i)
        calc_wc_variance_risk(rm.wc_set, model, sigma, cov_l, cov_u, cov_sigma, k_sigma)
        set_rm_risk_upper_bound(type, model, wc_variance_risk[i], rm.settings.ub,
                                "wc_variance_risk_$(i)")
        set_risk_expression(model, wc_variance_risk[i], rm.settings.scale, rm.settings.flag)
    end
    return nothing
end
function set_rm(port, rm::SD, type::Union{Trad, RB, NOC}; sigma::AbstractMatrix{<:Real},
                kwargs...)
    model = port.model
    scale_constr = model[:scale_constr]
    w = model[:w]
    @variable(model, sd_risk)
    use_portfolio_sigma = isnothing(rm.sigma) || isempty(rm.sigma)
    if !use_portfolio_sigma
        sigma = rm.sigma
    end
    G = sqrt(sigma)
    @constraint(model, constr_sd_risk_soc,
                [scale_constr * sd_risk; scale_constr * G * w] ∈ SecondOrderCone())
    set_rm_risk_upper_bound(type, model, sd_risk, rm.settings.ub, "sd_risk")
    set_risk_expression(model, sd_risk, rm.settings.scale, rm.settings.flag)
    return nothing
end
function set_rm(port, rms::AbstractVector{<:SD}, type::Union{Trad, RB, NOC};
                sigma::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
    scale_constr = model[:scale_constr]
    w = model[:w]
    count = length(rms)
    @variable(model, sd_risk[1:count])
    for (i, rm) ∈ pairs(rms)
        use_portfolio_sigma = (isnothing(rm.sigma) || isempty(rm.sigma))
        if !use_portfolio_sigma
            sigma = rm.sigma
        end
        G = sqrt(sigma)
        model[Symbol("constr_sd_risk_soc_$(i)")] = @constraint(model,
                                                               [scale_constr * sd_risk[i];
                                                                scale_constr * G * w] ∈
                                                               SecondOrderCone())
        set_rm_risk_upper_bound(type, model, sd_risk[i], rm.settings.ub, "sd_risk_$(i)")
        set_risk_expression(model, sd_risk[i], rm.settings.scale, rm.settings.flag)
    end
    return nothing
end
function set_rm(port::Portfolio, rm::MAD, type::Union{Trad, RB, NOC};
                mu::AbstractVector{<:Real}, returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
    scale_constr = model[:scale_constr]
    w = model[:w]
    T = size(returns, 1)
    get_net_portfolio_returns(model, returns)
    net_X = model[:net_X]
    if !(isnothing(rm.mu) || isempty(rm.mu))
        mu = rm.mu
    end
    @variable(model, mad[1:T] .>= 0)
    we = rm.we
    # if isnothing(we)
    #     @expression(model, mad_risk, 2 * mean(mad))
    # else
    #     @expression(model, mad_risk, 2 * mean(mad, we))
    # end
    # @constraint(model, constr_mar_mad,
    #             scale_constr * (net_X .- dot(mu, w)) .>= scale_constr * -mad)
    @expression(model, mar_mad, scale_constr * (net_X .- dot(mu, w) + mad))
    if isnothing(we)
        @expression(model, mad_risk, mean(mad + mar_mad))
    else
        @expression(model, mad_risk, mean(mad + mar_mad, we))
    end
    @constraint(model, constr_mar_mad, mar_mad .>= 0)
    set_rm_risk_upper_bound(type, model, mad_risk, rm.settings.ub, "mad_risk")
    set_risk_expression(model, mad_risk, rm.settings.scale, rm.settings.flag)
    return nothing
end
function set_rm(port::Portfolio, rms::AbstractVector{<:MAD}, type::Union{Trad, RB, NOC};
                mu::AbstractVector{<:Real}, returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
    scale_constr = model[:scale_constr]
    w = model[:w]
    T = size(returns, 1)
    get_net_portfolio_returns(model, returns)
    net_X = model[:net_X]
    count = length(rms)
    @variable(model, mad[1:T, 1:count] .>= 0)
    @expression(model, mad_risk[1:count], zero(AffExpr))
    for (i, rm) ∈ pairs(rms)
        if !(isnothing(rm.mu) || isempty(rm.mu))
            mu = rm.mu
        end
        mar_mad = model[Symbol("mar_mad_$(i)")] = @expression(model,
                                                              scale_constr *
                                                              (net_X .- dot(mu, w) +
                                                               view(mad, :, i)))
        model[Symbol("constr_mar_mad_$(i)")] = @constraint(model, mar_mad .>= 0)
        we = rm.we
        if isnothing(we)
            add_to_expression!(mad_risk[i], mean(view(mad, :, i) + mar_mad))
        else
            add_to_expression!(mad_risk[i], mean(view(mad, :, i) + mar_mad, we))
        end
        set_rm_risk_upper_bound(type, model, mad_risk[i], rm.settings.ub, "mad_risk_$(i)")
        set_risk_expression(model, mad_risk[i], rm.settings.scale, rm.settings.flag)
    end
    return nothing
end
function semi_variance_risk(::Quad, model::JuMP.Model, svariance, iTm1)
    @expression(model, svariance_risk, dot(svariance, svariance) * iTm1)
    return nothing
end
function semi_variance_risk(::Quad, ::Any, svariance, svariance_risk, iTm1)
    add_to_expression!(svariance_risk, iTm1, dot(svariance, svariance))
    return nothing
end
function semi_variance_risk(::SOC, model::JuMP.Model, svariance, iTm1)
    scale_constr = model[:scale_constr]
    @variable(model, tsvariance)
    @constraint(model, constr_svariance_soc,
                [scale_constr * tsvariance; 0.5; scale_constr * svariance] in
                RotatedSecondOrderCone())
    @expression(model, svariance_risk, tsvariance * iTm1)
    return nothing
end
function semi_variance_risk(::SOC, model, svariance, svariance_risk, iTm1, i)
    scale_constr = model[:scale_constr]
    model[Symbol("tsvariance_$(i)")] = tsvariance = @variable(model)
    model[Symbol("constr_svariance_soc_$(i)")] = @constraint(model,
                                                             [scale_constr * tsvariance;
                                                              0.5;
                                                              scale_constr * svariance] in
                                                             RotatedSecondOrderCone())
    add_to_expression!(svariance_risk, iTm1, tsvariance)
    return nothing
end
function set_rm(port::Portfolio, rm::SVariance, type::Union{Trad, RB, NOC};
                mu::AbstractVector{<:Real}, returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
    scale_constr = model[:scale_constr]
    w = model[:w]
    k = model[:k]
    T = size(returns, 1)
    get_net_portfolio_returns(model, returns)
    net_X = model[:net_X]
    if !(isnothing(rm.mu) || isempty(rm.mu))
        mu = rm.mu
    end
    @variable(model, svariance[1:T] .>= 0)
    semi_variance_risk(rm.formulation, model, svariance, inv(T - 1))
    @constraint(model, constr_svariance_mar,
                scale_constr * (net_X .- dot(mu, w)) .>= scale_constr * -svariance)
    svariance_risk = model[:svariance_risk]
    set_rm_risk_upper_bound(type, model, svariance_risk, rm.settings.ub, "svariance_risk")
    set_risk_expression(model, svariance_risk, rm.settings.scale, rm.settings.flag)

    return nothing
end
function set_rm(port::Portfolio, rms::AbstractVector{<:SVariance},
                type::Union{Trad, RB, NOC}; mu::AbstractVector{<:Real},
                returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
    scale_constr = model[:scale_constr]
    w = model[:w]
    k = model[:k]
    T = size(returns, 1)
    get_net_portfolio_returns(model, returns)
    net_X = model[:net_X]
    iTm1 = inv(T - 1)
    count = length(rms)
    @variable(model, svariance[1:T, 1:count] .>= 0)
    @expression(model, svariance_risk[1:count], zero(QuadExpr))
    for (i, rm) ∈ pairs(rms)
        if !(isnothing(rm.mu) || isempty(rm.mu))
            mu = rm.mu
        end
        model[Symbol("constr_svariance_mar_$(i)")] = @constraint(model,
                                                                 scale_constr *
                                                                 (net_X .- dot(mu, w)) .>=
                                                                 scale_constr *
                                                                 -view(svariance, :, i))
        semi_variance_risk(rm.formulation, model, view(svariance, :, i), svariance_risk[i],
                           iTm1, i)
        set_rm_risk_upper_bound(type, model, svariance_risk[i], rm.settings.ub,
                                "svariance_risk_$(i)")
        set_risk_expression(model, svariance_risk[i], rm.settings.scale, rm.settings.flag)
    end
    return nothing
end
function set_rm(port::Portfolio, rm::SSD, type::Union{Trad, RB, NOC};
                mu::AbstractVector{<:Real}, returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
    scale_constr = model[:scale_constr]
    w = model[:w]
    k = model[:k]
    T = size(returns, 1)
    get_net_portfolio_returns(model, returns)
    net_X = model[:net_X]
    if !(isnothing(rm.mu) || isempty(rm.mu))
        mu = rm.mu
    end
    @variables(model, begin
                   ssd[1:T] .>= 0
                   sdev
               end)
    @expression(model, sdev_risk, sdev / sqrt(T - 1))
    @constraints(model,
                 begin
                     constr_ssd_mar,
                     scale_constr * (net_X .- dot(mu, w)) .>= scale_constr * -ssd
                     constr_sdev_soc,
                     [scale_constr * sdev; scale_constr * ssd] ∈ SecondOrderCone()
                 end)
    set_rm_risk_upper_bound(type, model, sdev_risk, rm.settings.ub, "sdev_risk")
    set_risk_expression(model, sdev_risk, rm.settings.scale, rm.settings.flag)

    return nothing
end
function set_rm(port::Portfolio, rms::AbstractVector{<:SSD}, type::Union{Trad, RB, NOC};
                mu::AbstractVector{<:Real}, returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
    scale_constr = model[:scale_constr]
    w = model[:w]
    k = model[:k]
    T = size(returns, 1)
    get_net_portfolio_returns(model, returns)
    net_X = model[:net_X]
    iTm1 = inv(sqrt(T - 1))
    count = length(rms)
    @variable(model, ssd[1:T, 1:count] .>= 0)
    @variable(model, sdev[1:count])
    @expression(model, sdev_risk[1:count], zero(AffExpr))
    for (i, rm) ∈ pairs(rms)
        if !(isnothing(rm.mu) || isempty(rm.mu))
            mu = rm.mu
        end
        model[Symbol("constr_ssd_mar_$(i)")], model[Symbol("constr_sdev_soc_$(i)")] = @constraints(model,
                                                                                                   begin
                                                                                                       scale_constr *
                                                                                                       (net_X .-
                                                                                                        dot(mu,
                                                                                                            w)) .>=
                                                                                                       scale_constr *
                                                                                                       -view(ssd,
                                                                                                             :,
                                                                                                             i)
                                                                                                       [scale_constr *
                                                                                                        sdev[i]
                                                                                                        scale_constr *
                                                                                                        view(ssd,
                                                                                                             :,
                                                                                                             i)] ∈
                                                                                                       SecondOrderCone()
                                                                                                   end)
        add_to_expression!(sdev_risk[i], iTm1, sdev[i])
        set_rm_risk_upper_bound(type, model, sdev_risk[i], rm.settings.ub, "sdev_risk_$(i)")
        set_risk_expression(model, sdev_risk[i], rm.settings.scale, rm.settings.flag)
    end
    return nothing
end
function set_rm(port::Portfolio, rm::FLPM, type::Union{Trad, RB, NOC};
                returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
    scale_constr = model[:scale_constr]
    w = model[:w]
    k = model[:k]
    T = size(returns, 1)
    get_net_portfolio_returns(model, returns)
    net_X = model[:net_X]
    target = rm.target
    target = if isnothing(target) || isa(target, AbstractVector) && isempty(target)
        mu = rm.mu
        if isnothing(mu) || isempty(mu)
            dot(port.mu, w)
        else
            dot(mu, w)
        end
    else
        target * k
    end
    @variable(model, flpm[1:T] .>= 0)
    @expression(model, flpm_risk, sum(flpm) / T)
    @constraint(model, constr_flpm,
                scale_constr * flpm .>= scale_constr * (target .- net_X))
    set_rm_risk_upper_bound(type, model, flpm_risk, rm.settings.ub, "flpm_risk")
    set_risk_expression(model, flpm_risk, rm.settings.scale, rm.settings.flag)
    return nothing
end
function set_rm(port::Portfolio, rms::AbstractVector{<:FLPM}, type::Union{Trad, RB, NOC};
                returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
    scale_constr = model[:scale_constr]
    w = model[:w]
    k = model[:k]
    T = size(returns, 1)
    get_net_portfolio_returns(model, returns)
    net_X = model[:net_X]
    iT = inv(T)
    count = length(rms)
    @variable(model, flpm[1:T, 1:count] .>= 0)
    @expression(model, flpm_risk[1:count], zero(AffExpr))
    for (i, rm) ∈ pairs(rms)
        target = rm.target
        target = if isnothing(target) || isa(target, AbstractVector) && isempty(target)
            mu = rm.mu
            if isnothing(mu) || isempty(mu)
                dot(port.mu, w)
            else
                dot(mu, w)
            end
        else
            target * k
        end
        add_to_expression!(flpm_risk[i], iT, sum(view(flpm, :, i)))
        model[Symbol("constr_flpm_$(i)")] = @constraint(model,
                                                        scale_constr * view(flpm, :, i) .>=
                                                        scale_constr * (target .- net_X))
        set_rm_risk_upper_bound(type, model, flpm_risk[i], rm.settings.ub, "flpm_risk_$(i)")
        set_risk_expression(model, flpm_risk[i], rm.settings.scale, rm.settings.flag)
    end
    return nothing
end
function set_rm(port::Portfolio, rm::SLPM, type::Union{Trad, RB, NOC};
                returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
    scale_constr = model[:scale_constr]
    w = model[:w]
    k = model[:k]
    T = size(returns, 1)
    get_net_portfolio_returns(model, returns)
    net_X = model[:net_X]
    target = rm.target
    target = if isnothing(target) || isa(target, AbstractVector) && isempty(target)
        mu = rm.mu
        if isnothing(mu) || isempty(mu)
            dot(port.mu, w)
        else
            dot(mu, w)
        end
    else
        target * k
    end
    @variables(model, begin
                   slpm[1:T] .>= 0
                   tslpm
               end)
    @expression(model, slpm_risk, tslpm / sqrt(T - 1))
    @constraints(model,
                 begin
                     constr_slpm, scale_constr * slpm .>= scale_constr * (target .- net_X)
                     constr_slpm_soc,
                     [scale_constr * tslpm; scale_constr * slpm] ∈ SecondOrderCone()
                 end)
    set_rm_risk_upper_bound(type, model, slpm_risk, rm.settings.ub, "slpm_risk")
    set_risk_expression(model, slpm_risk, rm.settings.scale, rm.settings.flag)
    return nothing
end
function set_rm(port::Portfolio, rms::AbstractVector{<:SLPM}, type::Union{Trad, RB, NOC};
                returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
    scale_constr = model[:scale_constr]
    w = model[:w]
    k = model[:k]
    T = size(returns, 1)
    get_net_portfolio_returns(model, returns)
    net_X = model[:net_X]
    iTm1 = sqrt(inv(T - 1))
    count = length(rms)
    @variables(model, begin
                   slpm[1:T, 1:count] .>= 0
                   tslpm[1:count]
               end)
    @expression(model, slpm_risk[1:count], zero(AffExpr))
    for (i, rm) ∈ pairs(rms)
        target = rm.target
        target = if isnothing(target) || isa(target, AbstractVector) && isempty(target)
            mu = rm.mu
            if isnothing(mu) || isempty(mu)
                dot(port.mu, w)
            else
                dot(mu, w)
            end
        else
            target * k
        end
        add_to_expression!(slpm_risk[i], iTm1, tslpm[i])
        model[Symbol("constr_slpm_$(i)")], model[Symbol("constr_slpm_soc_$(i)")] = @constraints(model,
                                                                                                begin
                                                                                                    scale_constr *
                                                                                                    view(slpm,
                                                                                                         :,
                                                                                                         i) .>=
                                                                                                    scale_constr *
                                                                                                    (target .-
                                                                                                     net_X)
                                                                                                    [scale_constr *
                                                                                                     tslpm[i]
                                                                                                     scale_constr *
                                                                                                     view(slpm,
                                                                                                          :,
                                                                                                          i)] ∈
                                                                                                    SecondOrderCone()
                                                                                                end)
        set_rm_risk_upper_bound(type, model, slpm_risk[i], rm.settings.ub, "slpm_risk_$(i)")
        set_risk_expression(model, slpm_risk[i], rm.settings.scale, rm.settings.flag)
    end
    return nothing
end
function wr_risk_expression(model, returns)
    if haskey(model, :wr)
        return nothing
    end
    scale_constr = model[:scale_constr]
    get_net_portfolio_returns(model, returns)
    net_X = model[:net_X]
    @variable(model, wr)
    @expression(model, wr_risk, wr)
    @constraint(model, constr_wr, scale_constr * -net_X .<= scale_constr * wr)

    return nothing
end
function set_rm(port::Portfolio, rm::WR, type::Union{Trad, RB, NOC};
                returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
    wr_risk_expression(model, returns)
    wr_risk = model[:wr_risk]
    set_rm_risk_upper_bound(type, model, wr_risk, rm.settings.ub, "wr_risk")
    set_risk_expression(model, wr_risk, rm.settings.scale, rm.settings.flag)
    return nothing
end
function set_rm(port::Portfolio, rm::RG, type::Union{Trad, RB, NOC};
                returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
    scale_constr = model[:scale_constr]
    wr_risk_expression(model, returns)
    wr_risk = model[:wr_risk]
    net_X = model[:net_X]
    @variable(model, br)
    @expression(model, rg_risk, wr_risk - br)
    @constraint(model, constr_br, scale_constr * -net_X .>= scale_constr * br)
    set_rm_risk_upper_bound(type, model, rg_risk, rm.settings.ub, "rg_risk")
    set_risk_expression(model, rg_risk, rm.settings.scale, rm.settings.flag)
    return nothing
end
function set_rm(port::Portfolio, rm::CVaR, type::Union{Trad, RB, NOC};
                returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
    scale_constr = model[:scale_constr]
    get_net_portfolio_returns(model, returns)
    net_X = model[:net_X]
    T = size(returns, 1)
    iat = inv(rm.alpha * T)
    @variables(model, begin
                   var
                   z_cvar[1:T] .>= 0
               end)
    @expression(model, cvar_risk, var + sum(z_cvar) * iat)
    @constraint(model, constr_cvar,
                scale_constr * z_cvar .>= scale_constr * (-net_X .- var))
    set_rm_risk_upper_bound(type, model, cvar_risk, rm.settings.ub, "cvar_risk")
    set_risk_expression(model, cvar_risk, rm.settings.scale, rm.settings.flag)
    return nothing
end
function set_rm(port::Portfolio, rms::AbstractVector{<:CVaR}, type::Union{Trad, RB, NOC};
                returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
    scale_constr = model[:scale_constr]
    get_net_portfolio_returns(model, returns)
    net_X = model[:net_X]
    T = size(returns, 1)
    count = length(rms)
    @variables(model, begin
                   var[1:count]
                   z_cvar[1:T, 1:count] .>= 0
               end)
    @expression(model, cvar_risk[1:count], zero(AffExpr))
    for (i, rm) ∈ pairs(rms)
        iat = inv(rm.alpha * T)
        add_to_expression!(cvar_risk[i], var[i])
        add_to_expression!(cvar_risk[i], iat, sum(view(z_cvar, :, i)))
        model[Symbol("constr_cvar_$(i)")] = @constraint(model,
                                                        scale_constr *
                                                        view(z_cvar, :, i) .>=
                                                        scale_constr * (-net_X .- var[i]))
        set_rm_risk_upper_bound(type, model, cvar_risk[i], rm.settings.ub, "cvar_risk_$(i)")
        set_risk_expression(model, cvar_risk[i], rm.settings.scale, rm.settings.flag)
    end
    return nothing
end
function set_rm(port::Portfolio, rm::DRCVaR, type::Union{Trad, RB, NOC};
                returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
    scale_constr = model[:scale_constr]
    get_net_portfolio_returns(model, returns)
    get_one_plus_returns(model, returns)
    w = model[:w]
    net_X = model[:net_X]
    ret_p_1 = model[:ret_p_1]
    T, N = size(returns)

    b1 = rm.l
    alpha = rm.alpha
    radius = rm.r

    a1 = -one(alpha)
    a2 = -one(alpha) - b1 * inv(alpha)
    b2 = b1 * (one(alpha) - inv(alpha))

    @variables(model, begin
                   lb
                   tau
                   s[1:T]
                   u[1:T, 1:N] .>= 0
                   v[1:T, 1:N] .>= 0
                   tu_drcvar[1:T]
                   tv_drcvar[1:T]
               end)
    @constraints(model,
                 begin
                     constr_u_drcvar,
                     scale_constr *
                     (b1 * tau .+ a1 * net_X .+ vec(sum(u .* ret_p_1; dims = 2))) .<=
                     scale_constr * s
                     constr_v_drcvar,
                     scale_constr *
                     (b2 * tau .+ a2 * net_X .+ vec(sum(v .* ret_p_1; dims = 2))) .<=
                     scale_constr * s
                     constr_u_drcvar_infnorm[i = 1:T],
                     [scale_constr * tu_drcvar[i];
                      scale_constr * (-view(u, i, :) .- a1 * w)] in
                     MOI.NormInfinityCone(1 + N)
                     constr_v_drcvar_infnorm[i = 1:T],
                     [scale_constr * tv_drcvar[i];
                      scale_constr * (-view(v, i, :) .- a2 * w)] in
                     MOI.NormInfinityCone(1 + N)
                     constr_u_drcvar_lb, scale_constr * tu_drcvar .<= scale_constr * lb
                     constr_v_drcvar_lb, scale_constr * tv_drcvar .<= scale_constr * lb
                 end)

    @expression(model, drcvar_risk, radius * lb + mean(s))
    set_rm_risk_upper_bound(type, model, drcvar_risk, rm.settings.ub, "drcvar_risk")
    set_risk_expression(model, drcvar_risk, rm.settings.scale, rm.settings.flag)
    return nothing
end
function set_rm(port::Portfolio, rms::AbstractVector{<:DRCVaR}, type::Union{Trad, RB, NOC};
                returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
    scale_constr = model[:scale_constr]
    get_net_portfolio_returns(model, returns)
    get_one_plus_returns(model, returns)
    w = model[:w]
    net_X = model[:net_X]
    ret_p_1 = model[:ret_p_1]
    T, N = size(returns)

    count = length(rms)

    @variables(model, begin
                   lb[1:count]
                   tau[1:count]
                   s[1:T, 1:count]
                   u[1:T, 1:N, 1:count] .>= 0
                   v[1:T, 1:N, 1:count] .>= 0
                   tu_drcvar[1:T, 1:count]
                   tv_drcvar[1:T, 1:count]
               end)
    @expression(model, drcvar_risk[1:count], zero(AffExpr))
    for (j, rm) ∈ pairs(rms)
        b1 = rm.l
        alpha = rm.alpha
        radius = rm.r

        a1 = -one(alpha)
        a2 = a1 - b1 * inv(alpha)
        b2 = b1 * (one(alpha) - inv(alpha))

        model[Symbol("constr_u_drcvar_$(j)")], model[Symbol("constr_v_drcvar_$(j)")], model[Symbol("constr_u_drcvar_infnorm_$(j)")], model[Symbol("constr_v_drcvar_infnorm_$(j)")], model[Symbol("constr_u_drcvar_lb_$(j)")], model[Symbol("constr_v_drcvar_lb_$(j)")] = @constraints(model,
                                                                                                                                                                                                                                                                                      begin
                                                                                                                                                                                                                                                                                          scale_constr *
                                                                                                                                                                                                                                                                                          (b1 *
                                                                                                                                                                                                                                                                                           tau[j] .+
                                                                                                                                                                                                                                                                                           a1 *
                                                                                                                                                                                                                                                                                           net_X .+
                                                                                                                                                                                                                                                                                           vec(sum(view(u,
                                                                                                                                                                                                                                                                                                        :,
                                                                                                                                                                                                                                                                                                        :,
                                                                                                                                                                                                                                                                                                        j) .*
                                                                                                                                                                                                                                                                                                   ret_p_1;
                                                                                                                                                                                                                                                                                                   dims = 2))) .<=
                                                                                                                                                                                                                                                                                          scale_constr *
                                                                                                                                                                                                                                                                                          view(s,
                                                                                                                                                                                                                                                                                               :,
                                                                                                                                                                                                                                                                                               j)
                                                                                                                                                                                                                                                                                          scale_constr *
                                                                                                                                                                                                                                                                                          (b2 *
                                                                                                                                                                                                                                                                                           tau[j] .+
                                                                                                                                                                                                                                                                                           a2 *
                                                                                                                                                                                                                                                                                           net_X .+
                                                                                                                                                                                                                                                                                           vec(sum(view(v,
                                                                                                                                                                                                                                                                                                        :,
                                                                                                                                                                                                                                                                                                        :,
                                                                                                                                                                                                                                                                                                        j) .*
                                                                                                                                                                                                                                                                                                   ret_p_1;
                                                                                                                                                                                                                                                                                                   dims = 2))) .<=
                                                                                                                                                                                                                                                                                          scale_constr *
                                                                                                                                                                                                                                                                                          view(s,
                                                                                                                                                                                                                                                                                               :,
                                                                                                                                                                                                                                                                                               j)
                                                                                                                                                                                                                                                                                          [i = 1:T],
                                                                                                                                                                                                                                                                                          [scale_constr *
                                                                                                                                                                                                                                                                                           tu_drcvar[i,
                                                                                                                                                                                                                                                                                                     j]
                                                                                                                                                                                                                                                                                           scale_constr *
                                                                                                                                                                                                                                                                                           (-view(u,
                                                                                                                                                                                                                                                                                                  i,
                                                                                                                                                                                                                                                                                                  :,
                                                                                                                                                                                                                                                                                                  j) .-
                                                                                                                                                                                                                                                                                            a1 *
                                                                                                                                                                                                                                                                                            w)] in
                                                                                                                                                                                                                                                                                          MOI.NormInfinityCone(1 +
                                                                                                                                                                                                                                                                                                               N)
                                                                                                                                                                                                                                                                                          [i = 1:T],
                                                                                                                                                                                                                                                                                          [scale_constr *
                                                                                                                                                                                                                                                                                           tv_drcvar[i,
                                                                                                                                                                                                                                                                                                     j]
                                                                                                                                                                                                                                                                                           scale_constr *
                                                                                                                                                                                                                                                                                           (-view(v,
                                                                                                                                                                                                                                                                                                  i,
                                                                                                                                                                                                                                                                                                  :,
                                                                                                                                                                                                                                                                                                  j) .-
                                                                                                                                                                                                                                                                                            a2 *
                                                                                                                                                                                                                                                                                            w)] in
                                                                                                                                                                                                                                                                                          MOI.NormInfinityCone(1 +
                                                                                                                                                                                                                                                                                                               N)
                                                                                                                                                                                                                                                                                          scale_constr *
                                                                                                                                                                                                                                                                                          view(tu_drcvar,
                                                                                                                                                                                                                                                                                               :,
                                                                                                                                                                                                                                                                                               j) .<=
                                                                                                                                                                                                                                                                                          scale_constr *
                                                                                                                                                                                                                                                                                          lb[j]
                                                                                                                                                                                                                                                                                          scale_constr *
                                                                                                                                                                                                                                                                                          view(tv_drcvar,
                                                                                                                                                                                                                                                                                               :,
                                                                                                                                                                                                                                                                                               j) .<=
                                                                                                                                                                                                                                                                                          scale_constr *
                                                                                                                                                                                                                                                                                          lb[j]
                                                                                                                                                                                                                                                                                      end)
        add_to_expression!(drcvar_risk[j], radius, lb[j])
        add_to_expression!(drcvar_risk[j], mean(view(s, :, j)))
        set_rm_risk_upper_bound(type, model, drcvar_risk[j], rm.settings.ub,
                                "drcvar_risk_$(j)")
        set_risk_expression(model, drcvar_risk[j], rm.settings.scale, rm.settings.flag)
    end
    return nothing
end
function set_rm(port::Portfolio, rm::CVaRRG, type::Union{Trad, RB, NOC};
                returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
    scale_constr = model[:scale_constr]
    get_net_portfolio_returns(model, returns)
    net_X = model[:net_X]
    T = size(returns, 1)
    iat = inv(rm.alpha * T)
    ibt = inv(rm.beta * T)
    @variables(model, begin
                   var_l
                   z_cvar_l[1:T] .>= 0
                   var_h
                   z_cvar_h[1:T] .<= 0
               end)
    @expressions(model, begin
                     cvar_risk_l, var_l + sum(z_cvar_l) * iat
                     cvar_risk_h, var_h + sum(z_cvar_h) * ibt
                     rcvar_risk, cvar_risk_l - cvar_risk_h
                 end)
    @constraints(model,
                 begin
                     constr_cvar_l,
                     scale_constr * z_cvar_l .>= scale_constr * (-net_X .- var_l)
                     constr_cvar_h,
                     scale_constr * z_cvar_h .<= scale_constr * (-net_X .- var_h)
                 end)

    set_rm_risk_upper_bound(type, model, rcvar_risk, rm.settings.ub, "rcvar_risk")
    set_risk_expression(model, rcvar_risk, rm.settings.scale, rm.settings.flag)
    return nothing
end
function set_rm(port::Portfolio, rms::AbstractVector{<:CVaRRG}, type::Union{Trad, RB, NOC};
                returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
    scale_constr = model[:scale_constr]
    get_net_portfolio_returns(model, returns)
    net_X = model[:net_X]
    T = size(returns, 1)
    count = length(rms)
    @variables(model, begin
                   var_l[1:count]
                   z_cvar_l[1:T, 1:count] .>= 0
                   var_h[1:count]
                   z_cvar_h[1:T, 1:count] .<= 0
               end)
    @expressions(model, begin
                     cvar_risk_l[1:count], zero(AffExpr)
                     cvar_risk_h[1:count], zero(AffExpr)
                     rcvar_risk[1:count], zero(AffExpr)
                 end)
    for (i, rm) ∈ pairs(rms)
        iat = inv(rm.alpha * T)
        ibt = inv(rm.beta * T)

        model[Symbol("constr_cvar_l_$(i)")], model[Symbol("constr_cvar_h_$(i)")] = @constraints(model,
                                                                                                begin
                                                                                                    scale_constr *
                                                                                                    view(z_cvar_l,
                                                                                                         :,
                                                                                                         i) .>=
                                                                                                    scale_constr *
                                                                                                    (-net_X .-
                                                                                                     var_l[i])
                                                                                                    scale_constr *
                                                                                                    view(z_cvar_h,
                                                                                                         :,
                                                                                                         i) .<=
                                                                                                    scale_constr *
                                                                                                    (-net_X .-
                                                                                                     var_h[i])
                                                                                                end)
        add_to_expression!(cvar_risk_l[i], var_l[i])
        add_to_expression!(cvar_risk_l[i], iat, sum(view(z_cvar_l, :, i)))
        add_to_expression!(cvar_risk_h[i], var_h[i])
        add_to_expression!(cvar_risk_h[i], ibt, sum(view(z_cvar_h, :, i)))
        add_to_expression!(rcvar_risk[i], cvar_risk_l[i])
        add_to_expression!(rcvar_risk[i], -1, cvar_risk_h[i])
        set_rm_risk_upper_bound(type, model, rcvar_risk[i], rm.settings.ub,
                                "rcvar_risk_$(i)")
        set_risk_expression(model, rcvar_risk[i], rm.settings.scale, rm.settings.flag)
    end
    return nothing
end
function set_rm(port::Portfolio, rm::EVaR, type::Union{Trad, RB, NOC};
                returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
    scale_constr = model[:scale_constr]
    get_net_portfolio_returns(model, returns)
    net_X = model[:net_X]
    T = size(returns, 1)
    at = rm.alpha * T
    @variables(model, begin
                   t_evar
                   z_evar >= 0
                   u_evar[1:T]
               end)
    @expression(model, evar_risk, t_evar - z_evar * log(at))
    @constraints(model,
                 begin
                     constr_evar, scale_constr * sum(u_evar) <= scale_constr * z_evar
                     constr_evar_exp_cone[i = 1:T],
                     [scale_constr * (-net_X[i] - t_evar), scale_constr * z_evar,
                      scale_constr * u_evar[i]] ∈ MOI.ExponentialCone()
                 end)
    set_rm_risk_upper_bound(type, model, evar_risk, rm.settings.ub, "evar_risk")
    set_risk_expression(model, evar_risk, rm.settings.scale, rm.settings.flag)

    return nothing
end
function set_rm(port::Portfolio, rms::AbstractVector{<:EVaR}, type::Union{Trad, RB, NOC};
                returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
    scale_constr = model[:scale_constr]
    get_net_portfolio_returns(model, returns)
    net_X = model[:net_X]
    T = size(returns, 1)
    count = length(rms)
    @variables(model, begin
                   t_evar[1:count]
                   z_evar[1:count] .>= 0
                   u_evar[1:T, 1:count]
               end)
    @expression(model, evar_risk[1:count], zero(AffExpr))
    for (j, rm) ∈ pairs(rms)
        at = rm.alpha * T
        model[Symbol("constr_evar_$(j)")], model[Symbol("constr_evar_exp_cone_$(j)")] = @constraints(model,
                                                                                                     begin
                                                                                                         scale_constr *
                                                                                                         sum(view(u_evar,
                                                                                                                  :,
                                                                                                                  j)) <=
                                                                                                         scale_constr *
                                                                                                         z_evar[j]
                                                                                                         [i = 1:T],
                                                                                                         [scale_constr *
                                                                                                          (-net_X[i] -
                                                                                                           t_evar[j]),
                                                                                                          scale_constr *
                                                                                                          z_evar[j],
                                                                                                          scale_constr *
                                                                                                          u_evar[i,
                                                                                                                 j]] ∈
                                                                                                         MOI.ExponentialCone()
                                                                                                     end)
        add_to_expression!(evar_risk[j], t_evar[j])
        add_to_expression!(evar_risk[j], -log(at), z_evar[j])
        set_rm_risk_upper_bound(type, model, evar_risk[j], rm.settings.ub, "evar_risk_$(j)")
        set_risk_expression(model, evar_risk[j], rm.settings.scale, rm.settings.flag)
    end
    return nothing
end
function set_rm(port::Portfolio, rm::RLVaR, type::Union{Trad, RB, NOC};
                returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
    scale_constr = model[:scale_constr]
    get_net_portfolio_returns(model, returns)
    net_X = model[:net_X]
    T = size(returns, 1)
    iat = inv(rm.alpha * T)
    lnk = (iat^rm.kappa - iat^(-rm.kappa)) / (2 * rm.kappa)
    opk = one(rm.kappa) + rm.kappa
    omk = one(rm.kappa) - rm.kappa
    ik2 = inv(2 * rm.kappa)
    ik = inv(rm.kappa)
    iopk = inv(opk)
    iomk = inv(omk)
    @variables(model, begin
                   t_rlvar
                   z_rlvar >= 0
                   omega_rlvar[1:T]
                   psi_rlvar[1:T]
                   theta_rlvar[1:T]
                   epsilon_rlvar[1:T]
               end)
    @expression(model, rlvar_risk, t_rlvar + lnk * z_rlvar + sum(psi_rlvar .+ theta_rlvar))
    @constraints(model,
                 begin
                     constr_rlvar_pcone_a[i = 1:T],
                     [scale_constr * z_rlvar * opk * ik2,
                      scale_constr * psi_rlvar[i] * opk * ik,
                      scale_constr * epsilon_rlvar[i]] ∈ MOI.PowerCone(iopk)
                     constr_rlvar_pcone_b[i = 1:T],
                     [scale_constr * omega_rlvar[i] * iomk,
                      scale_constr * theta_rlvar[i] * ik, scale_constr * -z_rlvar * ik2] ∈
                     MOI.PowerCone(omk)
                     constr_rlvar,
                     scale_constr * (-net_X .- t_rlvar .+ epsilon_rlvar .+ omega_rlvar) .<=
                     0
                 end)
    set_rm_risk_upper_bound(type, model, rlvar_risk, rm.settings.ub, "rlvar_risk")
    set_risk_expression(model, rlvar_risk, rm.settings.scale, rm.settings.flag)
    return nothing
end
function set_rm(port::Portfolio, rms::AbstractVector{<:RLVaR}, type::Union{Trad, RB, NOC};
                returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
    scale_constr = model[:scale_constr]
    get_net_portfolio_returns(model, returns)
    net_X = model[:net_X]
    T = size(returns, 1)
    count = length(rms)
    @variables(model, begin
                   t_rlvar[1:count]
                   z_rlvar[1:count] .>= 0
                   omega_rlvar[1:T, 1:count]
                   psi_rlvar[1:T, 1:count]
                   theta_rlvar[1:T, 1:count]
                   epsilon_rlvar[1:T, 1:count]
               end)
    @expression(model, rlvar_risk[1:count], zero(AffExpr))
    for (j, rm) ∈ pairs(rms)
        iat = inv(rm.alpha * T)
        lnk = (iat^rm.kappa - iat^(-rm.kappa)) / (2 * rm.kappa)
        opk = one(rm.kappa) + rm.kappa
        omk = one(rm.kappa) - rm.kappa
        ik2 = inv(2 * rm.kappa)
        ik = inv(rm.kappa)
        iopk = inv(opk)
        iomk = inv(omk)
        model[Symbol("constr_rlvar_pcone_a_$(j)")], model[Symbol("constr_rlvar_pcone_b_$(j)")], model[Symbol("constr_rlvar_$(j)")] = @constraints(model,
                                                                                                                                                  begin
                                                                                                                                                      [i = 1:T],
                                                                                                                                                      [scale_constr *
                                                                                                                                                       z_rlvar[j] *
                                                                                                                                                       opk *
                                                                                                                                                       ik2,
                                                                                                                                                       scale_constr *
                                                                                                                                                       psi_rlvar[i,
                                                                                                                                                                 j] *
                                                                                                                                                       opk *
                                                                                                                                                       ik,
                                                                                                                                                       scale_constr *
                                                                                                                                                       epsilon_rlvar[i,
                                                                                                                                                                     j]] ∈
                                                                                                                                                      MOI.PowerCone(iopk)
                                                                                                                                                      [i = 1:T],
                                                                                                                                                      [scale_constr *
                                                                                                                                                       omega_rlvar[i,
                                                                                                                                                                   j] *
                                                                                                                                                       iomk,
                                                                                                                                                       scale_constr *
                                                                                                                                                       theta_rlvar[i,
                                                                                                                                                                   j] *
                                                                                                                                                       ik,
                                                                                                                                                       scale_constr *
                                                                                                                                                       -z_rlvar[j] *
                                                                                                                                                       ik2] ∈
                                                                                                                                                      MOI.PowerCone(omk)
                                                                                                                                                      scale_constr *
                                                                                                                                                      (-net_X .-
                                                                                                                                                       t_rlvar[j] .+
                                                                                                                                                       view(epsilon_rlvar,
                                                                                                                                                            :,
                                                                                                                                                            j) .+
                                                                                                                                                       view(omega_rlvar,
                                                                                                                                                            :,
                                                                                                                                                            j)) .<=
                                                                                                                                                      0
                                                                                                                                                  end)
        add_to_expression!(rlvar_risk[j], t_rlvar[j])
        add_to_expression!(rlvar_risk[j], lnk, z_rlvar[j])
        add_to_expression!(rlvar_risk[j],
                           sum(view(psi_rlvar, :, j) .+ view(theta_rlvar, :, j)))
        set_rm_risk_upper_bound(type, model, rlvar_risk[j], rm.settings.ub,
                                "rlvar_risk_$(j)")
        set_risk_expression(model, rlvar_risk[j], rm.settings.scale, rm.settings.flag)
    end
    return nothing
end
function DD_constraints(model, returns)
    if haskey(model, :dd)
        return nothing
    end

    scale_constr = model[:scale_constr]
    get_net_portfolio_returns(model, returns)
    net_X = model[:net_X]
    T = size(returns, 1)
    @variable(model, dd[1:(T + 1)])
    @constraints(model,
                 begin
                     constr_dd_start, scale_constr * dd[1] == 0
                     constr_dd_geq_0, scale_constr * view(dd, 2:(T + 1)) .>= 0
                     constr_dd,
                     scale_constr * view(dd, 2:(T + 1)) .>=
                     scale_constr * (view(dd, 1:T) .- net_X)
                 end)

    return nothing
end
function set_rm(port::Portfolio, rm::MDD, type::Union{Trad, RB, NOC};
                returns::AbstractMatrix{<:Real}, kargs...)
    model = port.model
    scale_constr = model[:scale_constr]
    DD_constraints(model, returns)
    dd = model[:dd]
    T = size(returns, 1)
    @variable(model, mdd)
    @expression(model, mdd_risk, mdd)
    @constraint(model, constr_mdd,
                scale_constr * mdd .>= scale_constr * view(dd, 2:(T + 1)))
    set_rm_risk_upper_bound(type, model, mdd_risk, rm.settings.ub, "mdd_risk")
    set_risk_expression(model, mdd_risk, rm.settings.scale, rm.settings.flag)
    return nothing
end
function set_rm(port::Portfolio, rm::ADD, type::Union{Trad, RB, NOC};
                returns::AbstractMatrix{<:Real}, kargs...)
    model = port.model
    DD_constraints(model, returns)
    dd = model[:dd]
    T = size(returns, 1)
    w = rm.w
    if isnothing(w)
        @expression(model, add_risk, mean(view(dd, 2:(T + 1))))
    else
        @expression(model, add_risk, mean(view(dd, 2:(T + 1)), w))
    end
    set_rm_risk_upper_bound(type, model, add_risk, rm.settings.ub, "add_risk")
    set_risk_expression(model, add_risk, rm.settings.scale, rm.settings.flag)
    return nothing
end
function set_rm(port::Portfolio, rms::AbstractVector{<:ADD}, type::Union{Trad, RB, NOC};
                returns::AbstractMatrix{<:Real}, kargs...)
    model = port.model
    DD_constraints(model, returns)
    dd = model[:dd]
    T = size(returns, 1)
    count = length(rms)
    @expression(model, add_risk[1:count], zero(AffExpr))
    for (i, rm) ∈ pairs(rms)
        w = rm.w
        if isnothing(w)
            add_to_expression!(add_risk[i], mean(view(dd, 2:(T + 1))))
        else
            add_to_expression!(add_risk[i], mean(view(dd, 2:(T + 1)), w))
        end
        set_rm_risk_upper_bound(type, model, add_risk[i], rm.settings.ub, "add_risk_$(i)")
        set_risk_expression(model, add_risk[i], rm.settings.scale, rm.settings.flag)
    end
    return nothing
end
function set_rm(port::Portfolio, rm::UCI, type::Union{Trad, RB, NOC};
                returns::AbstractMatrix{<:Real}, kargs...)
    model = port.model
    scale_constr = model[:scale_constr]
    DD_constraints(model, returns)
    dd = model[:dd]
    T = size(returns, 1)
    @variable(model, uci)
    @expression(model, uci_risk, uci / sqrt(T))
    @constraint(model, constr_uci_soc,
                [scale_constr * uci; scale_constr * view(dd, 2:(T + 1))] ∈
                SecondOrderCone())
    set_rm_risk_upper_bound(type, model, uci_risk, rm.settings.ub, "uci_risk")
    set_risk_expression(model, uci_risk, rm.settings.scale, rm.settings.flag)
    return nothing
end
function set_rm(port::Portfolio, rm::CDaR, type::Union{Trad, RB, NOC};
                returns::AbstractMatrix{<:Real}, kargs...)
    model = port.model
    scale_constr = model[:scale_constr]
    DD_constraints(model, returns)
    dd = model[:dd]
    T = size(returns, 1)
    iat = inv(rm.alpha * T)
    @variables(model, begin
                   dar
                   z_cdar[1:T] .>= 0
               end)
    @expression(model, cdar_risk, dar + sum(z_cdar) * iat)
    @constraint(model, constr_cdar,
                scale_constr * z_cdar .>= scale_constr * (view(dd, 2:(T + 1)) .- dar))
    set_rm_risk_upper_bound(type, model, cdar_risk, rm.settings.ub, "cdar_risk")
    set_risk_expression(model, cdar_risk, rm.settings.scale, rm.settings.flag)
    return nothing
end
function set_rm(port::Portfolio, rms::AbstractVector{<:CDaR}, type::Union{Trad, RB, NOC};
                returns::AbstractMatrix{<:Real}, kargs...)
    model = port.model
    scale_constr = model[:scale_constr]
    DD_constraints(model, returns)
    dd = model[:dd]
    T = size(returns, 1)
    count = length(rms)
    @variables(model, begin
                   dar[1:count]
                   z_cdar[1:T, 1:count] .>= 0
               end)
    @expression(model, cdar_risk[1:count], zero(AffExpr))
    for (i, rm) ∈ pairs(rms)
        iat = inv(rm.alpha * T)
        model[Symbol("constr_cdar_$(i)")] = @constraint(model,
                                                        scale_constr *
                                                        view(z_cdar, :, i) .>=
                                                        scale_constr *
                                                        (view(dd, 2:(T + 1)) .- dar[i]))
        add_to_expression!(cdar_risk[i], dar[i])
        add_to_expression!(cdar_risk[i], iat, sum(view(z_cdar, :, i)))
        set_rm_risk_upper_bound(type, model, cdar_risk[i], rm.settings.ub, "cdar_risk_$(i)")
        set_risk_expression(model, cdar_risk[i], rm.settings.scale, rm.settings.flag)
    end
    return nothing
end
function set_rm(port::Portfolio, rm::EDaR, type::Union{Trad, RB, NOC};
                returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
    scale_constr = model[:scale_constr]
    DD_constraints(model, returns)
    dd = model[:dd]
    T = size(returns, 1)
    at = rm.alpha * T
    @variables(model, begin
                   t_edar
                   z_edar >= 0
                   u_edar[1:T]
               end)
    @expression(model, edar_risk, t_edar - z_edar * log(at))
    @constraints(model,
                 begin
                     constr_edar, scale_constr * sum(u_edar) <= scale_constr * z_edar
                     constr_edar_exp_cone[i = 1:T],
                     [scale_constr * (dd[i + 1] - t_edar), scale_constr * z_edar,
                      scale_constr * u_edar[i]] ∈ MOI.ExponentialCone()
                 end)
    set_rm_risk_upper_bound(type, model, edar_risk, rm.settings.ub, "edar_risk")
    set_risk_expression(model, edar_risk, rm.settings.scale, rm.settings.flag)
    return nothing
end
function set_rm(port::Portfolio, rms::AbstractVector{<:EDaR}, type::Union{Trad, RB, NOC};
                returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
    scale_constr = model[:scale_constr]
    DD_constraints(model, returns)
    dd = model[:dd]
    T = size(returns, 1)
    count = length(rms)
    @variables(model, begin
                   t_edar[1:count]
                   z_edar[1:count] .>= 0
                   u_edar[1:T, 1:count]
               end)
    @expression(model, edar_risk[1:count], zero(AffExpr))
    for (j, rm) ∈ pairs(rms)
        at = rm.alpha * T

        model[Symbol("constr_edar_$(j)")], model[Symbol("constr_edar_exp_cone_$(j)")] = @constraints(model,
                                                                                                     begin
                                                                                                         scale_constr *
                                                                                                         sum(view(u_edar,
                                                                                                                  :,
                                                                                                                  j)) <=
                                                                                                         scale_constr *
                                                                                                         z_edar[j]
                                                                                                         [i = 1:T],
                                                                                                         [scale_constr *
                                                                                                          (dd[i + 1] -
                                                                                                           t_edar[j]),
                                                                                                          scale_constr *
                                                                                                          z_edar[j],
                                                                                                          scale_constr *
                                                                                                          u_edar[i,
                                                                                                                 j]] ∈
                                                                                                         MOI.ExponentialCone()
                                                                                                     end)
        add_to_expression!(edar_risk[j], t_edar[j])
        add_to_expression!(edar_risk[j], -log(at), z_edar[j])
        set_rm_risk_upper_bound(type, model, edar_risk[j], rm.settings.ub, "edar_risk_$(j)")
        set_risk_expression(model, edar_risk[j], rm.settings.scale, rm.settings.flag)
    end
    return nothing
end
function set_rm(port::Portfolio, rm::RLDaR, type::Union{Trad, RB, NOC};
                returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
    scale_constr = model[:scale_constr]
    DD_constraints(model, returns)
    dd = model[:dd]
    T = size(returns, 1)
    iat = inv(rm.alpha * T)
    lnk = (iat^rm.kappa - iat^(-rm.kappa)) / (2 * rm.kappa)
    opk = one(rm.kappa) + rm.kappa
    omk = one(rm.kappa) - rm.kappa
    ik2 = inv(2 * rm.kappa)
    ik = inv(rm.kappa)
    iopk = inv(opk)
    iomk = inv(omk)
    @variables(model, begin
                   t_rldar
                   z_rldar >= 0
                   omega_rldar[1:T]
                   psi_rldar[1:T]
                   theta_rldar[1:T]
                   epsilon_rldar[1:T]
               end)
    @expression(model, rldar_risk, t_rldar + lnk * z_rldar + sum(psi_rldar .+ theta_rldar))
    @constraints(model,
                 begin
                     constr_rldar_pcone_a[i = 1:T],
                     [scale_constr * z_rldar * opk * ik2,
                      scale_constr * psi_rldar[i] * opk * ik,
                      scale_constr * epsilon_rldar[i]] ∈ MOI.PowerCone(iopk)
                     constr_rldar_pcone_b[i = 1:T],
                     [scale_constr * omega_rldar[i] * iomk,
                      scale_constr * theta_rldar[i] * ik, scale_constr * -z_rldar * ik2] ∈
                     MOI.PowerCone(omk)
                     constr_rldar,
                     scale_constr *
                     (view(dd, 2:(T + 1)) .- t_rldar .+ epsilon_rldar .+ omega_rldar) .<= 0
                 end)
    set_rm_risk_upper_bound(type, model, rldar_risk, rm.settings.ub, "rldar_risk")
    set_risk_expression(model, rldar_risk, rm.settings.scale, rm.settings.flag)
    return nothing
end
function set_rm(port::Portfolio, rms::AbstractVector{<:RLDaR}, type::Union{Trad, RB, NOC};
                returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
    scale_constr = model[:scale_constr]
    DD_constraints(model, returns)
    dd = model[:dd]
    T = size(returns, 1)
    count = length(rms)
    @variables(model, begin
                   t_rldar[1:count]
                   z_rldar[1:count] .>= 0
                   omega_rldar[1:T, 1:count]
                   psi_rldar[1:T, 1:count]
                   theta_rldar[1:T, 1:count]
                   epsilon_rldar[1:T, 1:count]
               end)
    @expression(model, rldar_risk[1:count], zero(AffExpr))
    for (j, rm) ∈ pairs(rms)
        iat = inv(rm.alpha * T)
        lnk = (iat^rm.kappa - iat^(-rm.kappa)) / (2 * rm.kappa)
        opk = one(rm.kappa) + rm.kappa
        omk = one(rm.kappa) - rm.kappa
        ik2 = inv(2 * rm.kappa)
        ik = inv(rm.kappa)
        iopk = inv(opk)
        iomk = inv(omk)
        model[Symbol("constr_rldar_pcone_a_$(j)")], model[Symbol("constr_rldar_pcone_b_$(j)")], model[Symbol("constr_rldar_$(j)")] = @constraints(model,
                                                                                                                                                  begin
                                                                                                                                                      [i = 1:T],
                                                                                                                                                      [scale_constr *
                                                                                                                                                       z_rldar[j] *
                                                                                                                                                       opk *
                                                                                                                                                       ik2,
                                                                                                                                                       scale_constr *
                                                                                                                                                       psi_rldar[i,
                                                                                                                                                                 j] *
                                                                                                                                                       opk *
                                                                                                                                                       ik,
                                                                                                                                                       scale_constr *
                                                                                                                                                       epsilon_rldar[i,
                                                                                                                                                                     j]] ∈
                                                                                                                                                      MOI.PowerCone(iopk)
                                                                                                                                                      [i = 1:T],
                                                                                                                                                      [scale_constr *
                                                                                                                                                       omega_rldar[i,
                                                                                                                                                                   j] *
                                                                                                                                                       iomk,
                                                                                                                                                       scale_constr *
                                                                                                                                                       theta_rldar[i,
                                                                                                                                                                   j] *
                                                                                                                                                       ik,
                                                                                                                                                       scale_constr *
                                                                                                                                                       -z_rldar[j] *
                                                                                                                                                       ik2] ∈
                                                                                                                                                      MOI.PowerCone(omk)
                                                                                                                                                      scale_constr *
                                                                                                                                                      (view(dd,
                                                                                                                                                            2:(T + 1)) .-
                                                                                                                                                       t_rldar[j] .+
                                                                                                                                                       view(epsilon_rldar,
                                                                                                                                                            :,
                                                                                                                                                            j) .+
                                                                                                                                                       view(omega_rldar,
                                                                                                                                                            :,
                                                                                                                                                            j)) .<=
                                                                                                                                                      0
                                                                                                                                                  end)
        add_to_expression!(rldar_risk[j], t_rldar[j])
        add_to_expression!(rldar_risk[j], lnk, z_rldar[j])
        add_to_expression!(rldar_risk[j],
                           sum(view(psi_rldar, :, j) .+ view(theta_rldar, :, j)))
        set_rm_risk_upper_bound(type, model, rldar_risk[j], rm.settings.ub,
                                "rldar_risk_$(j)")
        set_risk_expression(model, rldar_risk[j], rm.settings.scale, rm.settings.flag)
    end
    return nothing
end
function set_rm(port::Portfolio, rm::Kurt, type::Union{Trad, RB, NOC}; kwargs...)
    model = port.model
    scale_constr = model[:scale_constr]
    SDP_constraints(model, type)
    W = model[:W]
    N = size(port.returns, 2)
    @variable(model, kurt_risk)
    kt = if (isnothing(rm.kt) || isempty(rm.kt))
        port.kurt
    else
        rm.kt
    end
    if !iszero(port.max_num_assets_kurt) && N > port.max_num_assets_kurt
        f = port.max_num_assets_kurt_scale
        Nf = f * N
        @variable(model, x_kurt[1:Nf])
        A = block_vec_pq(kt, N, N)
        vals_A, vecs_A = eigen(A)
        vals_A = clamp.(real(vals_A), 0, Inf) .+ clamp.(imag(vals_A), 0, Inf)im
        Bi = Vector{Matrix{eltype(kt)}}(undef, Nf)
        N_eig = length(vals_A)
        for i ∈ 1:Nf
            j = i - 1
            B = reshape(real(complex(sqrt(vals_A[end - j])) * view(vecs_A, :, N_eig - j)),
                        N, N)
            Bi[i] = B
        end
        @constraints(model,
                     begin
                         constr_approx_kurt_soc,
                         [scale_constr * kurt_risk; scale_constr * x_kurt] ∈
                         SecondOrderCone()
                         constr_approx_kurt[i = 1:Nf],
                         scale_constr * x_kurt[i] == scale_constr * tr(Bi[i] * W)
                     end)
    else
        L_2 = port.L_2
        S_2 = port.S_2
        sqrt_sigma_4 = sqrt(S_2 * kt * transpose(S_2))
        @expression(model, zkurt, L_2 * vec(W))
        @constraint(model, constr_kurt_soc,
                    [scale_constr * kurt_risk; scale_constr * sqrt_sigma_4 * zkurt] ∈
                    SecondOrderCone())
    end
    set_rm_risk_upper_bound(type, model, kurt_risk, rm.settings.ub, "kurt_risk")
    set_risk_expression(model, kurt_risk, rm.settings.scale, rm.settings.flag)
    return nothing
end
function set_rm(port::Portfolio, rms::AbstractVector{<:Kurt}, type::Union{Trad, RB, NOC};
                kwargs...)
    model = port.model
    scale_constr = model[:scale_constr]
    SDP_constraints(model, type)
    W = model[:W]
    N = size(port.returns, 2)
    count = length(rms)
    @variable(model, kurt_risk[1:count])
    W = model[:W]
    if !iszero(port.max_num_assets_kurt) && N > port.max_num_assets_kurt
        f = port.max_num_assets_kurt_scale
        Nf = f * N
        @variable(model, x_kurt[1:Nf, 1:count])
        for (idx, rm) ∈ pairs(rms)
            kt = if (isnothing(rm.kt) || isempty(rm.kt))
                port.kurt
            else
                rm.kt
            end
            A = block_vec_pq(kt, N, N)
            vals_A, vecs_A = eigen(A)
            vals_A = clamp.(real(vals_A), 0, Inf) .+ clamp.(imag(vals_A), 0, Inf)im
            Bi = Vector{Matrix{eltype(kt)}}(undef, Nf)
            N_eig = length(vals_A)
            for i ∈ 1:Nf
                j = i - 1
                B = reshape(real(complex(sqrt(vals_A[end - j])) *
                                 view(vecs_A, :, N_eig - j)), N, N)
                Bi[i] = B
            end
            model[Symbol("constr_approx_kurt_soc_$(idx)")], model[Symbol("constr_approx_kurt_$(idx)")] = @constraints(model,
                                                                                                                      begin
                                                                                                                          [scale_constr *
                                                                                                                           kurt_risk[idx]
                                                                                                                           scale_constr *
                                                                                                                           view(x_kurt,
                                                                                                                                :,
                                                                                                                                idx)] ∈
                                                                                                                          SecondOrderCone()
                                                                                                                          [i = 1:Nf],
                                                                                                                          scale_constr *
                                                                                                                          x_kurt[i,
                                                                                                                                 idx] ==
                                                                                                                          scale_constr *
                                                                                                                          tr(Bi[i] *
                                                                                                                             W)
                                                                                                                      end)
            set_rm_risk_upper_bound(type, model, kurt_risk[idx], rm.settings.ub,
                                    "kurt_risk_$(idx)")
            set_risk_expression(model, kurt_risk[idx], rm.settings.scale, rm.settings.flag)
        end
    else
        L_2 = port.L_2
        S_2 = port.S_2
        @expression(model, zkurt[1:size(port.L_2, 1), 1:count], zero(AffExpr))
        for (idx, rm) ∈ pairs(rms)
            kt = if (isnothing(rm.kt) || isempty(rm.kt))
                port.kurt
            else
                rm.kt
            end
            sqrt_sigma_4 = sqrt(S_2 * kt * transpose(S_2))
            add_to_expression!.(view(zkurt, :, idx), L_2 * vec(W))
            model[Symbol("constr_kurt_soc_$(idx)")] = @constraint(model,
                                                                  [scale_constr *
                                                                   kurt_risk[idx]
                                                                   scale_constr *
                                                                   sqrt_sigma_4 *
                                                                   view(zkurt, :, idx)] ∈
                                                                  SecondOrderCone())
            set_rm_risk_upper_bound(type, model, kurt_risk[idx], rm.settings.ub,
                                    "kurt_risk_$(idx)")
            set_risk_expression(model, kurt_risk[idx], rm.settings.scale, rm.settings.flag)
        end
    end

    return nothing
end
function set_rm(port::Portfolio, rm::SKurt, type::Union{Trad, RB, NOC}; kwargs...)
    model = port.model
    scale_constr = model[:scale_constr]
    SDP_constraints(model, type)
    W = model[:W]
    N = size(port.returns, 2)
    @variable(model, skurt_risk)
    kt = if (isnothing(rm.kt) || isempty(rm.kt))
        port.skurt
    else
        rm.kt
    end
    if !iszero(port.max_num_assets_kurt) && N > port.max_num_assets_kurt
        f = port.max_num_assets_kurt_scale
        Nf = f * N
        @variable(model, x_skurt[1:Nf])
        A = block_vec_pq(kt, N, N)
        vals_A, vecs_A = eigen(A)
        vals_A = clamp.(real(vals_A), 0, Inf) .+ clamp.(imag(vals_A), 0, Inf)im
        Bi = Vector{Matrix{eltype(kt)}}(undef, Nf)
        N_eig = length(vals_A)
        for i ∈ 1:Nf
            j = i - 1
            B = reshape(real(complex(sqrt(vals_A[end - j])) * view(vecs_A, :, N_eig - j)),
                        N, N)
            Bi[i] = B
        end
        @constraints(model,
                     begin
                         constr_approx_skurt_soc,
                         [scale_constr * skurt_risk; scale_constr * x_skurt] ∈
                         SecondOrderCone()
                         constr_approx_skurt[i = 1:Nf],
                         scale_constr * x_skurt[i] == scale_constr * tr(Bi[i] * W)
                     end)
    else
        L_2 = port.L_2
        S_2 = port.S_2
        sqrt_sigma_4 = sqrt(S_2 * kt * transpose(S_2))
        @expression(model, zskurt, L_2 * vec(W))
        @constraint(model, constr_skurt_soc,
                    [scale_constr * skurt_risk; scale_constr * sqrt_sigma_4 * zskurt] ∈
                    SecondOrderCone())
    end
    set_rm_risk_upper_bound(type, model, skurt_risk, rm.settings.ub, "skurt_risk")
    set_risk_expression(model, skurt_risk, rm.settings.scale, rm.settings.flag)
    return nothing
end
function set_rm(port::Portfolio, rms::AbstractVector{<:SKurt}, type::Union{Trad, RB, NOC};
                kwargs...)
    model = port.model
    scale_constr = model[:scale_constr]
    SDP_constraints(model, type)
    W = model[:W]
    N = size(port.returns, 2)
    count = length(rms)
    @variable(model, skurt_risk[1:count])
    W = model[:W]
    if !iszero(port.max_num_assets_kurt) && N > port.max_num_assets_kurt
        f = port.max_num_assets_kurt_scale
        Nf = f * N
        @variable(model, x_skurt[1:Nf, 1:count])
        for (idx, rm) ∈ pairs(rms)
            kt = if (isnothing(rm.kt) || isempty(rm.kt))
                port.skurt
            else
                rm.kt
            end
            A = block_vec_pq(kt, N, N)
            vals_A, vecs_A = eigen(A)
            vals_A = clamp.(real(vals_A), 0, Inf) .+ clamp.(imag(vals_A), 0, Inf)im
            Bi = Vector{Matrix{eltype(kt)}}(undef, Nf)
            N_eig = length(vals_A)
            for i ∈ 1:Nf
                j = i - 1
                B = reshape(real(complex(sqrt(vals_A[end - j])) *
                                 view(vecs_A, :, N_eig - j)), N, N)
                Bi[i] = B
            end
            model[Symbol("constr_approx_skurt_soc_$(idx)")], model[Symbol("constr_approx_skurt_$(idx)")] = @constraints(model,
                                                                                                                        begin
                                                                                                                            [scale_constr *
                                                                                                                             skurt_risk[idx]
                                                                                                                             scale_constr *
                                                                                                                             view(x_skurt,
                                                                                                                                  :,
                                                                                                                                  idx)] ∈
                                                                                                                            SecondOrderCone()
                                                                                                                            [i = 1:Nf],
                                                                                                                            scale_constr *
                                                                                                                            x_skurt[i,
                                                                                                                                    idx] ==
                                                                                                                            scale_constr *
                                                                                                                            tr(Bi[i] *
                                                                                                                               W)
                                                                                                                        end)
            set_rm_risk_upper_bound(type, model, skurt_risk[idx], rm.settings.ub,
                                    "skurt_risk_$(idx)")
            set_risk_expression(model, skurt_risk[idx], rm.settings.scale, rm.settings.flag)
        end
    else
        L_2 = port.L_2
        S_2 = port.S_2
        @expression(model, zskurt[1:size(port.L_2, 1), 1:count], zero(AffExpr))
        for (idx, rm) ∈ pairs(rms)
            kt = if (isnothing(rm.kt) || isempty(rm.kt))
                port.skurt
            else
                rm.kt
            end
            sqrt_sigma_4 = sqrt(S_2 * kt * transpose(S_2))
            add_to_expression!.(view(zskurt, :, idx), L_2 * vec(W))
            model[Symbol("constr_skurt_soc_$(idx)")] = @constraint(model,
                                                                   [scale_constr *
                                                                    skurt_risk[idx]
                                                                    scale_constr *
                                                                    sqrt_sigma_4 *
                                                                    view(zskurt, :, idx)] ∈
                                                                   SecondOrderCone())
            set_rm_risk_upper_bound(type, model, skurt_risk[idx], rm.settings.ub,
                                    "skurt_risk_$(idx)")
            set_risk_expression(model, skurt_risk[idx], rm.settings.scale, rm.settings.flag)
        end
    end

    return nothing
end
function OWA_constraints(model, returns)
    if haskey(model, :owa)
        return nothing
    end
    scale_constr = model[:scale_constr]
    get_net_portfolio_returns(model, returns)
    net_X = model[:net_X]
    T = size(returns, 1)
    @variable(model, owa[1:T])
    @constraint(model, scale_constr * net_X == scale_constr * owa)

    return nothing
end
function set_rm(port::Portfolio, rm::GMD, type::Union{Trad, RB, NOC};
                returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
    scale_constr = model[:scale_constr]
    T = size(returns, 1)

    if !rm.owa.approx
        OWA_constraints(model, returns)
        owa = model[:owa]
        ovec = range(1; stop = 1, length = T)
        @variables(model, begin
                       gmda[1:T]
                       gmdb[1:T]
                   end)
        @expression(model, gmd_risk, sum(gmda .+ gmdb))
        gmd_w = owa_gmd(T)
        @constraint(model, constr_gmd,
                    scale_constr * owa * transpose(gmd_w) .<=
                    scale_constr * (ovec * transpose(gmda) + gmdb * transpose(ovec)))
    else
        get_net_portfolio_returns(model, returns)
        net_X = model[:net_X]
        owa_p = rm.owa.p
        M = length(owa_p)

        @variables(model, begin
                       gmd_t
                       gmd_nu[1:T] .>= 0
                       gmd_eta[1:T] .>= 0
                       gmd_epsilon[1:T, 1:M]
                       gmd_psi[1:T, 1:M]
                       gmd_z[1:M]
                       gmd_y[1:M] .>= 0
                   end)

        gmd_w = -owa_gmd(T)
        gmd_s = sum(gmd_w)
        gmd_l = minimum(gmd_w)
        gmd_h = maximum(gmd_w)
        gmd_d = [norm(gmd_w, p) for p ∈ owa_p]

        @expression(model, gmd_risk,
                    gmd_s * gmd_t - gmd_l * sum(gmd_nu) +
                    gmd_h * sum(gmd_eta) +
                    dot(gmd_d, gmd_y))
        @constraints(model,
                     begin
                         constr_approx_1_gmd,
                         scale_constr * (net_X .+ gmd_t .- gmd_nu .+ gmd_eta .-
                                         vec(sum(gmd_epsilon; dims = 2))) .== 0
                         constr_approx_2_gmd,
                         scale_constr * (gmd_z .+ gmd_y) .==
                         scale_constr * vec(sum(gmd_psi; dims = 1))
                         constr_approx_gmd_pcone[i = 1:M, j = 1:T],
                         [scale_constr * -gmd_z[i] * owa_p[i],
                          scale_constr * gmd_psi[j, i] * owa_p[i] / (owa_p[i] - 1),
                          scale_constr * gmd_epsilon[j, i]] ∈ MOI.PowerCone(inv(owa_p[i]))
                     end)
    end
    set_rm_risk_upper_bound(type, model, gmd_risk, rm.settings.ub, "gmd_risk")
    set_risk_expression(model, gmd_risk, rm.settings.scale, rm.settings.flag)
    return nothing
end
function set_rm(port::Portfolio, rm::TG, type::Union{Trad, RB, NOC};
                returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
    scale_constr = model[:scale_constr]
    T = size(returns, 1)
    alpha = rm.alpha
    a_sim = rm.a_sim
    alpha_i = rm.alpha_i
    if !rm.owa.approx
        OWA_constraints(model, returns)
        owa = model[:owa]
        ovec = range(1; stop = 1, length = T)
        @variables(model, begin
                       tga[1:T]
                       tgb[1:T]
                   end)
        @expression(model, tg_risk, sum(tga .+ tgb))
        tg_w = owa_tg(T; alpha_i = alpha_i, alpha = alpha, a_sim = a_sim)
        @constraint(model, constr_tg,
                    scale_constr * owa * transpose(tg_w) .<=
                    scale_constr * (ovec * transpose(tga) + tgb * transpose(ovec)))
    else
        get_net_portfolio_returns(model, returns)
        net_X = model[:net_X]
        owa_p = rm.owa.p
        M = length(owa_p)

        @variables(model, begin
                       tg_t
                       tg_nu[1:T] .>= 0
                       tg_eta[1:T] .>= 0
                       tg_epsilon[1:T, 1:M]
                       tg_psi[1:T, 1:M]
                       tg_z[1:M]
                       tg_y[1:M] .>= 0
                   end)

        tg_w = -owa_tg(T; alpha_i = alpha_i, alpha = alpha, a_sim = a_sim)
        tg_s = sum(tg_w)
        tg_l = minimum(tg_w)
        tg_h = maximum(tg_w)
        tg_d = [norm(tg_w, p) for p ∈ owa_p]

        @expression(model, tg_risk,
                    tg_s * tg_t - tg_l * sum(tg_nu) + tg_h * sum(tg_eta) + dot(tg_d, tg_y))
        @constraints(model,
                     begin
                         constr_approx_1_tg,
                         scale_constr * (net_X .+ tg_t .- tg_nu .+ tg_eta .-
                                         vec(sum(tg_epsilon; dims = 2))) .== 0
                         constr_approx_2_tg,
                         scale_constr * (tg_z .+ tg_y) .==
                         scale_constr * vec(sum(tg_psi; dims = 1))
                         constr_approx_tg_pcone[i = 1:M, j = 1:T],
                         [scale_constr * -tg_z[i] * owa_p[i],
                          scale_constr * tg_psi[j, i] * owa_p[i] / (owa_p[i] - 1),
                          scale_constr * tg_epsilon[j, i]] ∈ MOI.PowerCone(inv(owa_p[i]))
                     end)
    end
    set_rm_risk_upper_bound(type, model, tg_risk, rm.settings.ub, "tg_risk")
    set_risk_expression(model, tg_risk, rm.settings.scale, rm.settings.flag)

    return nothing
end
function set_rm(port::Portfolio, rms::AbstractVector{<:TG}, type::Union{Trad, RB, NOC};
                returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
    scale_constr = model[:scale_constr]
    T = size(returns, 1)
    count = length(rms)
    @expression(model, tg_risk[1:count], zero(AffExpr))
    for (idx, rm) ∈ pairs(rms)
        alpha = rm.alpha
        a_sim = rm.a_sim
        alpha_i = rm.alpha_i
        if !rm.owa.approx
            ovec = range(1; stop = 1, length = T)
            if !haskey(model, :tga)
                OWA_constraints(model, returns)
                @variables(model, begin
                               tga[1:T, 1:count]
                               tgb[1:T, 1:count]
                           end)
            end
            tga = model[:tga]
            tgb = model[:tgb]
            owa = model[:owa]
            add_to_expression!(tg_risk[idx], sum(view(tga, :, idx) .+ view(tgb, :, idx)))
            tg_w = owa_tg(T; alpha_i = alpha_i, alpha = alpha, a_sim = a_sim)
            model[Symbol("constr_tg_$(idx)")] = @constraint(model,
                                                            scale_constr *
                                                            owa *
                                                            transpose(tg_w) .<=
                                                            scale_constr * (ovec *
                                                                            transpose(view(tga, :, idx)) +
                                                                            view(tgb, :, idx) *
                                                                            transpose(ovec)))
        else
            get_net_portfolio_returns(model, returns)
            net_X = model[:net_X]
            owa_p = rm.owa.p
            M = length(owa_p)

            tg_w = -owa_tg(T; alpha_i = alpha_i, alpha = alpha, a_sim = a_sim)
            tg_s = sum(tg_w)
            tg_l = minimum(tg_w)
            tg_h = maximum(tg_w)
            tg_d = [norm(tg_w, p) for p ∈ owa_p]
            if !haskey(model, :tg_t)
                M = length(rm.owa.p)
                @variables(model, begin
                               tg_t[1:count]
                               tg_nu[1:T, 1:count] .>= 0
                               tg_eta[1:T, 1:count] .>= 0
                               tg_epsilon[1:T, 1:M, 1:count]
                               tg_psi[1:T, 1:M, 1:count]
                               tg_z[1:M, 1:count]
                               tg_y[1:M, 1:count] .>= 0
                           end)
            end
            tg_t = model[:tg_t]
            tg_nu = model[:tg_nu]
            tg_eta = model[:tg_eta]
            tg_epsilon = model[:tg_epsilon]
            tg_psi = model[:tg_psi]
            tg_z = model[:tg_z]
            tg_y = model[:tg_y]
            add_to_expression!(tg_risk[idx], tg_s, tg_t[idx])
            add_to_expression!(tg_risk[idx], -tg_l, sum(view(tg_nu, :, idx)))
            add_to_expression!(tg_risk[idx], tg_h, sum(view(tg_eta, :, idx)))
            add_to_expression!(tg_risk[idx], dot(tg_d, view(tg_y, :, idx)))
            model[Symbol("constr_approx_1_tg_$(idx)")], model[Symbol("constr_approx_2_tg_$(idx)")], model[Symbol("constr_approx_tg_pcone_$(idx)")] = @constraints(model,
                                                                                                                                                                  begin
                                                                                                                                                                      scale_constr *
                                                                                                                                                                      (net_X .+
                                                                                                                                                                       tg_t[idx] .-
                                                                                                                                                                       view(tg_nu,
                                                                                                                                                                            :,
                                                                                                                                                                            idx) .+
                                                                                                                                                                       view(tg_eta,
                                                                                                                                                                            :,
                                                                                                                                                                            idx) .-
                                                                                                                                                                       vec(sum(view(tg_epsilon,
                                                                                                                                                                                    :,
                                                                                                                                                                                    :,
                                                                                                                                                                                    idx);
                                                                                                                                                                               dims = 2))) .==
                                                                                                                                                                      0
                                                                                                                                                                      scale_constr *
                                                                                                                                                                      (tg_z[:,
                                                                                                                                                                            idx] .+
                                                                                                                                                                       tg_y[:,
                                                                                                                                                                            idx]) .==
                                                                                                                                                                      scale_constr *
                                                                                                                                                                      vec(sum(view(tg_psi,
                                                                                                                                                                                   :,
                                                                                                                                                                                   :,
                                                                                                                                                                                   idx);
                                                                                                                                                                              dims = 1))
                                                                                                                                                                      [i = 1:M,
                                                                                                                                                                       j = 1:T],
                                                                                                                                                                      [scale_constr *
                                                                                                                                                                       -tg_z[i,
                                                                                                                                                                             idx] *
                                                                                                                                                                       owa_p[i],
                                                                                                                                                                       scale_constr *
                                                                                                                                                                       tg_psi[j,
                                                                                                                                                                              i,
                                                                                                                                                                              idx] *
                                                                                                                                                                       owa_p[i] /
                                                                                                                                                                       (owa_p[i] -
                                                                                                                                                                        1),
                                                                                                                                                                       scale_constr *
                                                                                                                                                                       tg_epsilon[j,
                                                                                                                                                                                  i,
                                                                                                                                                                                  idx]] ∈
                                                                                                                                                                      MOI.PowerCone(inv(owa_p[i]))
                                                                                                                                                                  end)
        end
        set_rm_risk_upper_bound(type, model, tg_risk[idx], rm.settings.ub, "tg_risk_$(idx)")
        set_risk_expression(model, tg_risk[idx], rm.settings.scale, rm.settings.flag)
    end
    return nothing
end
function set_rm(port::Portfolio, rm::TGRG, type::Union{Trad, RB, NOC};
                returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
    scale_constr = model[:scale_constr]
    T = size(returns, 1)
    alpha = rm.alpha
    a_sim = rm.a_sim
    alpha_i = rm.alpha_i
    beta = rm.beta
    b_sim = rm.b_sim
    beta_i = rm.beta_i
    if !rm.owa.approx
        OWA_constraints(model, returns)
        owa = model[:owa]
        ovec = range(1; stop = 1, length = T)
        @variables(model, begin
                       rtga[1:T]
                       rtgb[1:T]
                   end)
        @expression(model, rtg_risk, sum(rtga .+ rtgb))
        rtg_w = owa_rtg(T; alpha_i = alpha_i, alpha = alpha, a_sim = a_sim, beta_i = beta_i,
                        beta = beta, b_sim = b_sim)
        @constraint(model, constr_rtg,
                    scale_constr * owa * transpose(rtg_w) .<=
                    scale_constr * (ovec * transpose(rtga) + rtgb * transpose(ovec)))
    else
        get_net_portfolio_returns(model, returns)
        net_X = model[:net_X]
        owa_p = rm.owa.p
        M = length(owa_p)

        @variables(model, begin
                       rltg_t
                       rltg_nu[1:T] .>= 0
                       rltg_eta[1:T] .>= 0
                       rltg_epsilon[1:T, 1:M]
                       rltg_psi[1:T, 1:M]
                       rltg_z[1:M]
                       rltg_y[1:M] .>= 0
                   end)

        rltg_w = -owa_tg(T; alpha_i = alpha_i, alpha = alpha, a_sim = a_sim)
        rltg_s = sum(rltg_w)
        rltg_l = minimum(rltg_w)
        rltg_h = maximum(rltg_w)
        rltg_d = [norm(rltg_w, p) for p ∈ owa_p]

        @expression(model, rltg_risk,
                    rltg_s * rltg_t - rltg_l * sum(rltg_nu) +
                    rltg_h * sum(rltg_eta) +
                    dot(rltg_d, rltg_y))
        @constraints(model,
                     begin
                         constr_approx_1_rtg,
                         scale_constr * (net_X .+ rltg_t .- rltg_nu .+ rltg_eta .-
                                         vec(sum(rltg_epsilon; dims = 2))) .== 0
                         constr_approx_2_rtg,
                         scale_constr * (rltg_z .+ rltg_y) .==
                         scale_constr * vec(sum(rltg_psi; dims = 1))
                         constr_approx_1_rtg_pcone[i = 1:M, j = 1:T],
                         [scale_constr * -rltg_z[i] * owa_p[i],
                          scale_constr * rltg_psi[j, i] * owa_p[i] / (owa_p[i] - 1),
                          scale_constr * rltg_epsilon[j, i]] ∈ MOI.PowerCone(inv(owa_p[i]))
                     end)

        @variables(model, begin
                       rhtg_t
                       rhtg_nu[1:T] .>= 0
                       rhtg_eta[1:T] .>= 0
                       rhtg_epsilon[1:T, 1:M]
                       rhtg_psi[1:T, 1:M]
                       rhtg_z[1:M]
                       rhtg_y[1:M] .>= 0
                   end)

        rhtg_w = -owa_tg(T; alpha_i = beta_i, alpha = beta, a_sim = b_sim)
        rhtg_s = sum(rhtg_w)
        rhtg_l = minimum(rhtg_w)
        rhtg_h = maximum(rhtg_w)
        rhtg_d = [norm(rhtg_w, p) for p ∈ owa_p]

        @expressions(model,
                     begin
                         rhtg_risk,
                         rhtg_s * rhtg_t - rhtg_l * sum(rhtg_nu) +
                         rhtg_h * sum(rhtg_eta) +
                         dot(rhtg_d, rhtg_y)
                         rtg_risk, rltg_risk + rhtg_risk
                     end)
        @constraints(model,
                     begin
                         constr_approx_3_rtg,
                         scale_constr * (-net_X .+ rhtg_t .- rhtg_nu .+ rhtg_eta .-
                                         vec(sum(rhtg_epsilon; dims = 2))) .== 0
                         constr_approx_4_rtg,
                         scale_constr * (rhtg_z .+ rhtg_y) .==
                         scale_constr * vec(sum(rhtg_psi; dims = 1))
                         constr_approx_2_rtg_pcone[i = 1:M, j = 1:T],
                         [scale_constr * -rhtg_z[i] * owa_p[i],
                          scale_constr * rhtg_psi[j, i] * owa_p[i] / (owa_p[i] - 1),
                          scale_constr * rhtg_epsilon[j, i]] ∈ MOI.PowerCone(inv(owa_p[i]))
                     end)
    end
    set_rm_risk_upper_bound(type, model, rtg_risk, rm.settings.ub, "rtg_risk")
    set_risk_expression(model, rtg_risk, rm.settings.scale, rm.settings.flag)
    return nothing
end
function set_rm(port::Portfolio, rms::AbstractVector{<:TGRG}, type::Union{Trad, RB, NOC};
                returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
    scale_constr = model[:scale_constr]
    T = size(returns, 1)
    count = length(rms)
    @expression(model, rtg_risk[1:count], zero(AffExpr))
    for (idx, rm) ∈ pairs(rms)
        alpha = rm.alpha
        a_sim = rm.a_sim
        alpha_i = rm.alpha_i
        beta = rm.beta
        b_sim = rm.b_sim
        beta_i = rm.beta_i
        if !rm.owa.approx
            ovec = range(1; stop = 1, length = T)
            if !haskey(model, :rtga)
                OWA_constraints(model, returns)
                @variables(model, begin
                               rtga[1:T, 1:count]
                               rtgb[1:T, 1:count]
                           end)
            end
            rtga = model[:rtga]
            rtgb = model[:rtgb]
            owa = model[:owa]
            add_to_expression!(rtg_risk[idx], sum(view(rtga, :, idx) .+ view(rtgb, :, idx)))
            rtg_w = owa_rtg(T; alpha_i = alpha_i, alpha = alpha, a_sim = a_sim,
                            beta_i = beta_i, beta = beta, b_sim = b_sim)
            model[Symbol("constr_rtg_$(idx)")] = @constraint(model,
                                                             scale_constr *
                                                             owa *
                                                             transpose(rtg_w) .<=
                                                             scale_constr * (ovec *
                                                                             transpose(view(rtga, :, idx)) +
                                                                             view(rtgb, :, idx) *
                                                                             transpose(ovec)))
        else
            get_net_portfolio_returns(model, returns)
            net_X = model[:net_X]
            owa_p = rm.owa.p
            M = length(owa_p)

            rltg_w = -owa_tg(T; alpha_i = alpha_i, alpha = alpha, a_sim = a_sim)
            rltg_s = sum(rltg_w)
            rltg_l = minimum(rltg_w)
            rltg_h = maximum(rltg_w)
            rltg_d = [norm(rltg_w, p) for p ∈ owa_p]
            if !haskey(model, :rltg_t)
                M = length(rm.owa.p)
                @variables(model, begin
                               rltg_t[1:count]
                               rltg_nu[1:T, 1:count] .>= 0
                               rltg_eta[1:T, 1:count] .>= 0
                               rltg_epsilon[1:T, 1:M, 1:count]
                               rltg_psi[1:T, 1:M, 1:count]
                               rltg_z[1:M, 1:count]
                               rltg_y[1:M, 1:count] .>= 0
                               rhtg_t[1:count]
                               rhtg_nu[1:T, 1:count] .>= 0
                               rhtg_eta[1:T, 1:count] .>= 0
                               rhtg_epsilon[1:T, 1:M, 1:count]
                               rhtg_psi[1:T, 1:M, 1:count]
                               rhtg_z[1:M, 1:count]
                               rhtg_y[1:M, 1:count] .>= 0
                           end)
                @expressions(model, begin
                                 rltg_risk[1:count], zero(AffExpr)
                                 rhtg_risk[1:count], zero(AffExpr)
                             end)
            end
            rltg_t = model[:rltg_t]
            rltg_nu = model[:rltg_nu]
            rltg_eta = model[:rltg_eta]
            rltg_epsilon = model[:rltg_epsilon]
            rltg_psi = model[:rltg_psi]
            rltg_z = model[:rltg_z]
            rltg_y = model[:rltg_y]
            rltg_risk = model[:rltg_risk]
            rhtg_t = model[:rhtg_t]
            rhtg_nu = model[:rhtg_nu]
            rhtg_eta = model[:rhtg_eta]
            rhtg_epsilon = model[:rhtg_epsilon]
            rhtg_psi = model[:rhtg_psi]
            rhtg_z = model[:rhtg_z]
            rhtg_y = model[:rhtg_y]
            rhtg_risk = model[:rhtg_risk]
            add_to_expression!(rltg_risk[idx], rltg_s, rltg_t[idx])
            add_to_expression!(rltg_risk[idx], -rltg_l, sum(view(rltg_nu, :, idx)))
            add_to_expression!(rltg_risk[idx], rltg_h, sum(view(rltg_eta, :, idx)))
            add_to_expression!(rltg_risk[idx], dot(rltg_d, view(rltg_y, :, idx)))

            model[Symbol("constr_approx_1_rtg_$(idx)")], model[Symbol("constr_approx_2_rtg_$(idx)")], model[Symbol("constr_approx_1_rtg_pcone_$(idx)")] = @constraints(model,
                                                                                                                                                                       begin
                                                                                                                                                                           scale_constr *
                                                                                                                                                                           (net_X .+
                                                                                                                                                                            rltg_t[idx] .-
                                                                                                                                                                            view(rltg_nu,
                                                                                                                                                                                 :,
                                                                                                                                                                                 idx) .+
                                                                                                                                                                            view(rltg_eta,
                                                                                                                                                                                 :,
                                                                                                                                                                                 idx) .-
                                                                                                                                                                            vec(sum(view(rltg_epsilon,
                                                                                                                                                                                         :,
                                                                                                                                                                                         :,
                                                                                                                                                                                         idx);
                                                                                                                                                                                    dims = 2))) .==
                                                                                                                                                                           0
                                                                                                                                                                           scale_constr *
                                                                                                                                                                           (view(rltg_z,
                                                                                                                                                                                 :,
                                                                                                                                                                                 idx) .+
                                                                                                                                                                            view(rltg_y,
                                                                                                                                                                                 :,
                                                                                                                                                                                 idx)) .==
                                                                                                                                                                           scale_constr *
                                                                                                                                                                           vec(sum(view(rltg_psi,
                                                                                                                                                                                        :,
                                                                                                                                                                                        :,
                                                                                                                                                                                        idx);
                                                                                                                                                                                   dims = 1))
                                                                                                                                                                           [i = 1:M,
                                                                                                                                                                            j = 1:T],
                                                                                                                                                                           [scale_constr *
                                                                                                                                                                            -rltg_z[i,
                                                                                                                                                                                    idx] *
                                                                                                                                                                            owa_p[i],
                                                                                                                                                                            scale_constr *
                                                                                                                                                                            rltg_psi[j,
                                                                                                                                                                                     i,
                                                                                                                                                                                     idx] *
                                                                                                                                                                            owa_p[i] /
                                                                                                                                                                            (owa_p[i] -
                                                                                                                                                                             1),
                                                                                                                                                                            scale_constr *
                                                                                                                                                                            rltg_epsilon[j,
                                                                                                                                                                                         i,
                                                                                                                                                                                         idx]] ∈
                                                                                                                                                                           MOI.PowerCone(inv(owa_p[i]))
                                                                                                                                                                       end)
            rhtg_w = -owa_tg(T; alpha_i = beta_i, alpha = beta, a_sim = b_sim)
            rhtg_s = sum(rhtg_w)
            rhtg_l = minimum(rhtg_w)
            rhtg_h = maximum(rhtg_w)
            rhtg_d = [norm(rhtg_w, p) for p ∈ owa_p]
            add_to_expression!(rhtg_risk[idx], rhtg_s, rhtg_t[idx])
            add_to_expression!(rhtg_risk[idx], -rhtg_l, sum(view(rhtg_nu, :, idx)))
            add_to_expression!(rhtg_risk[idx], rhtg_h, sum(view(rhtg_eta, :, idx)))
            add_to_expression!(rhtg_risk[idx], dot(rhtg_d, view(rhtg_y, :, idx)))
            add_to_expression!(rtg_risk[idx], rltg_risk[idx])
            add_to_expression!(rtg_risk[idx], rhtg_risk[idx])
            model[Symbol("constr_approx_3_rtg_$(idx)")], model[Symbol("constr_approx_4_rtg_$(idx)")], model[Symbol("constr_approx_2_rtg_pcone_$(idx)")] = @constraints(model,
                                                                                                                                                                       begin
                                                                                                                                                                           scale_constr *
                                                                                                                                                                           (-net_X .+
                                                                                                                                                                            rhtg_t[idx] .-
                                                                                                                                                                            view(rhtg_nu,
                                                                                                                                                                                 :,
                                                                                                                                                                                 idx) .+
                                                                                                                                                                            view(rhtg_eta,
                                                                                                                                                                                 :,
                                                                                                                                                                                 idx) .-
                                                                                                                                                                            vec(sum(rhtg_epsilon[:,
                                                                                                                                                                                                 :,
                                                                                                                                                                                                 idx];
                                                                                                                                                                                    dims = 2))) .==
                                                                                                                                                                           0
                                                                                                                                                                           scale_constr *
                                                                                                                                                                           (view(rhtg_z,
                                                                                                                                                                                 :,
                                                                                                                                                                                 idx) .+
                                                                                                                                                                            view(rhtg_y,
                                                                                                                                                                                 :,
                                                                                                                                                                                 idx)) .==
                                                                                                                                                                           scale_constr *
                                                                                                                                                                           vec(sum(view(rhtg_psi,
                                                                                                                                                                                        :,
                                                                                                                                                                                        :,
                                                                                                                                                                                        idx);
                                                                                                                                                                                   dims = 1))
                                                                                                                                                                           [i = 1:M,
                                                                                                                                                                            j = 1:T],
                                                                                                                                                                           [scale_constr *
                                                                                                                                                                            -rhtg_z[i,
                                                                                                                                                                                    idx] *
                                                                                                                                                                            owa_p[i],
                                                                                                                                                                            scale_constr *
                                                                                                                                                                            rhtg_psi[j,
                                                                                                                                                                                     i,
                                                                                                                                                                                     idx] *
                                                                                                                                                                            owa_p[i] /
                                                                                                                                                                            (owa_p[i] -
                                                                                                                                                                             1),
                                                                                                                                                                            scale_constr *
                                                                                                                                                                            rhtg_epsilon[j,
                                                                                                                                                                                         i,
                                                                                                                                                                                         idx]] ∈
                                                                                                                                                                           MOI.PowerCone(inv(owa_p[i]))
                                                                                                                                                                       end)
        end
        set_rm_risk_upper_bound(type, model, rtg_risk[idx], rm.settings.ub,
                                "rtg_risk_$(idx)")
        set_risk_expression(model, rtg_risk[idx], rm.settings.scale, rm.settings.flag)
    end
    return nothing
end
function set_rm(port::Portfolio, rm::OWA, type::Union{Trad, RB, NOC};
                returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
    scale_constr = model[:scale_constr]
    T = size(returns, 1)

    if !rm.owa.approx
        OWA_constraints(model, returns)
        owa = model[:owa]
        ovec = range(1; stop = 1, length = T)
        @variables(model, begin
                       owa_a[1:T]
                       owa_b[1:T]
                   end)
        @expression(model, owa_risk, sum(owa_a .+ owa_b))
        owa_w = (isnothing(rm.w) || isempty(rm.w)) ? owa_gmd(T) : rm.w
        @constraint(model, constr_owa,
                    scale_constr * owa * transpose(owa_w) .<=
                    scale_constr * (ovec * transpose(owa_a) + owa_b * transpose(ovec)))
    else
        get_net_portfolio_returns(model, returns)
        net_X = model[:net_X]
        owa_p = rm.owa.p
        M = length(owa_p)

        @variables(model, begin
                       owa_t
                       owa_nu[1:T] .>= 0
                       owa_eta[1:T] .>= 0
                       owa_epsilon[1:T, 1:M]
                       owa_psi[1:T, 1:M]
                       owa_z[1:M]
                       owa_y[1:M] .>= 0
                   end)

        owa_w = (isnothing(rm.w) || isempty(rm.w)) ? -owa_gmd(T) : -rm.w
        owa_s = sum(owa_w)
        owa_l = minimum(owa_w)
        owa_h = maximum(owa_w)
        owa_d = [norm(owa_w, p) for p ∈ owa_p]

        @expression(model, owa_risk,
                    owa_s * owa_t - owa_l * sum(owa_nu) +
                    owa_h * sum(owa_eta) +
                    dot(owa_d, owa_y))
        @constraints(model,
                     begin
                         constr_approx_1_owa,
                         scale_constr * (net_X .+ owa_t .- owa_nu .+ owa_eta .-
                                         vec(sum(owa_epsilon; dims = 2))) .== 0
                         constr_approx_2_owa,
                         scale_constr * (owa_z .+ owa_y) .==
                         scale_constr * vec(sum(owa_psi; dims = 1))
                         constr_approx_owa_pcone[i = 1:M, j = 1:T],
                         [scale_constr * -owa_z[i] * owa_p[i],
                          scale_constr * owa_psi[j, i] * owa_p[i] / (owa_p[i] - 1),
                          scale_constr * owa_epsilon[j, i]] ∈ MOI.PowerCone(inv(owa_p[i]))
                     end)
    end
    set_rm_risk_upper_bound(type, model, owa_risk, rm.settings.ub, "owa_risk")
    set_risk_expression(model, owa_risk, rm.settings.scale, rm.settings.flag)
    return nothing
end
function set_rm(port::Portfolio, rms::AbstractVector{<:OWA}, type::Union{Trad, RB, NOC};
                returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
    scale_constr = model[:scale_constr]
    T = size(returns, 1)
    count = length(rms)
    @expression(model, owa_risk[1:count], zero(AffExpr))
    for (idx, rm) ∈ pairs(rms)
        if !rm.owa.approx
            ovec = range(1; stop = 1, length = T)
            if !haskey(model, :owa_a)
                OWA_constraints(model, returns)
                @variables(model, begin
                               owa_a[1:T, 1:count]
                               owa_b[1:T, 1:count]
                           end)
            end
            owa_a = model[:owa_a]
            owa_b = model[:owa_b]
            owa = model[:owa]
            add_to_expression!(owa_risk[idx],
                               sum(view(owa_a, :, idx) .+ view(owa_b, :, idx)))
            owa_w = (isnothing(rm.w) || isempty(rm.w)) ? owa_gmd(T) : rm.w
            model[Symbol("constr_owa_$(idx)")] = @constraint(model,
                                                             scale_constr *
                                                             owa *
                                                             transpose(owa_w) .<=
                                                             scale_constr * (ovec *
                                                                             transpose(view(owa_a, :, idx)) +
                                                                             view(owa_b, :, idx) *
                                                                             transpose(ovec)))
        else
            get_net_portfolio_returns(model, returns)
            net_X = model[:net_X]
            owa_p = rm.owa.p
            M = length(owa_p)

            owa_w = (isnothing(rm.w) || isempty(rm.w)) ? -owa_gmd(T) : -rm.w
            owa_s = sum(owa_w)
            owa_l = minimum(owa_w)
            owa_h = maximum(owa_w)
            owa_d = [norm(owa_w, p) for p ∈ owa_p]

            if !haskey(model, :owa_t)
                M = length(rm.owa.p)
                @variables(model, begin
                               owa_t[1:count]
                               owa_nu[1:T, 1:count] .>= 0
                               owa_eta[1:T, 1:count] .>= 0
                               owa_epsilon[1:T, 1:M, 1:count]
                               owa_psi[1:T, 1:M, 1:count]
                               owa_z[1:M, 1:count]
                               owa_y[1:M, 1:count] .>= 0
                           end)
            end
            owa_t = model[:owa_t]
            owa_nu = model[:owa_nu]
            owa_eta = model[:owa_eta]
            owa_epsilon = model[:owa_epsilon]
            owa_psi = model[:owa_psi]
            owa_z = model[:owa_z]
            owa_y = model[:owa_y]
            add_to_expression!(owa_risk[idx], owa_s, owa_t[idx])
            add_to_expression!(owa_risk[idx], -owa_l, sum(view(owa_nu, :, idx)))
            add_to_expression!(owa_risk[idx], owa_h, sum(view(owa_eta, :, idx)))
            add_to_expression!(owa_risk[idx], dot(owa_d, view(owa_y, :, idx)))

            model[Symbol("constr_approx_1_owa_$(idx)")], model[Symbol("constr_approx_2_owa_$(idx)")], model[Symbol("constr_approx_owa_pcone_$(idx)")] = @constraints(model,
                                                                                                                                                                     begin
                                                                                                                                                                         scale_constr *
                                                                                                                                                                         (net_X .+
                                                                                                                                                                          owa_t[idx] .-
                                                                                                                                                                          view(owa_nu,
                                                                                                                                                                               :,
                                                                                                                                                                               idx) .+
                                                                                                                                                                          view(owa_eta,
                                                                                                                                                                               :,
                                                                                                                                                                               idx) .-
                                                                                                                                                                          vec(sum(view(owa_epsilon,
                                                                                                                                                                                       :,
                                                                                                                                                                                       :,
                                                                                                                                                                                       idx);
                                                                                                                                                                                  dims = 2))) .==
                                                                                                                                                                         0
                                                                                                                                                                         scale_constr *
                                                                                                                                                                         (view(owa_z,
                                                                                                                                                                               :,
                                                                                                                                                                               idx) .+
                                                                                                                                                                          view(owa_y,
                                                                                                                                                                               :,
                                                                                                                                                                               idx)) .==
                                                                                                                                                                         scale_constr *
                                                                                                                                                                         vec(sum(view(owa_psi,
                                                                                                                                                                                      :,
                                                                                                                                                                                      :,
                                                                                                                                                                                      idx);
                                                                                                                                                                                 dims = 1))
                                                                                                                                                                         [i = 1:M,
                                                                                                                                                                          j = 1:T],
                                                                                                                                                                         [scale_constr *
                                                                                                                                                                          -owa_z[i,
                                                                                                                                                                                 idx] *
                                                                                                                                                                          owa_p[i],
                                                                                                                                                                          scale_constr *
                                                                                                                                                                          owa_psi[j,
                                                                                                                                                                                  i,
                                                                                                                                                                                  idx] *
                                                                                                                                                                          owa_p[i] /
                                                                                                                                                                          (owa_p[i] -
                                                                                                                                                                           1),
                                                                                                                                                                          scale_constr *
                                                                                                                                                                          owa_epsilon[j,
                                                                                                                                                                                      i,
                                                                                                                                                                                      idx]] ∈
                                                                                                                                                                         MOI.PowerCone(inv(owa_p[i]))
                                                                                                                                                                     end)
        end
        set_rm_risk_upper_bound(type, model, owa_risk[idx], rm.settings.ub,
                                "owa_risk_$(idx)")
        set_risk_expression(model, owa_risk[idx], rm.settings.scale, rm.settings.flag)
    end
    return nothing
end
function BDVariance_constraints(::BDVAbsVal, model, Dt, Dx, T)
    scale_constr = model[:scale_constr]
    @constraint(model, constr_bdvariance_noc[i = 1:T, j = i:T],
                [scale_constr * Dt[i, j]; scale_constr * Dx[i, j]] in MOI.NormOneCone(2))
    return nothing
end
function BDVariance_constraints(::BDVIneq, model, Dt, Dx, T)
    scale_constr = model[:scale_constr]
    @constraints(model,
                 begin
                     constr_p_bdvariance[i = 1:T, j = i:T],
                     scale_constr * Dt[i, j] >= scale_constr * Dx[i, j]
                     constr_n_bdvariance[i = 1:T, j = i:T],
                     scale_constr * Dt[i, j] >= scale_constr * -Dx[i, j]
                 end)
    return nothing
end
function set_rm(port::Portfolio, rm::BDVariance, type::Union{Trad, RB, NOC};
                returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
    get_net_portfolio_returns(model, returns)
    net_X = model[:net_X]
    T = size(returns, 1)
    iT2 = inv(T^2)
    ovec = range(1; stop = 1, length = T)
    @variable(model, Dt[1:T, 1:T], Symmetric)
    @expressions(model, begin
                     Dx, net_X * transpose(ovec) - ovec * transpose(net_X)
                     bd_variance_risk, iT2 * (dot(Dt, Dt) + iT2 * sum(Dt)^2)
                 end)
    BDVariance_constraints(rm.formulation, model, Dt, Dx, T)
    set_rm_risk_upper_bound(type, model, bd_variance_risk, rm.settings.ub,
                            "bd_variance_risk")
    set_risk_expression(model, bd_variance_risk, rm.settings.scale, rm.settings.flag)
    return nothing
end
function set_rm(port::Portfolio, rm::Skew, type::Union{Trad, RB, NOC}; kwargs...)
    model = port.model
    scale_constr = model[:scale_constr]
    w = model[:w]
    @variable(model, t_skew)
    V = if (isnothing(rm.V) || isempty(rm.V))
        port.V
    else
        rm.V
    end
    G = real(sqrt(V))
    @constraint(model, constr_skew_soc,
                [scale_constr * t_skew; scale_constr * G * w] ∈ SecondOrderCone())
    @expression(model, skew_risk, t_skew^2)
    set_rm_risk_upper_bound(type, model, t_skew, rm.settings.ub, "skew_risk")
    set_risk_expression(model, skew_risk, rm.settings.scale, rm.settings.flag)
    return nothing
end
function set_rm(port::Portfolio, rms::AbstractVector{<:Skew}, type::Union{Trad, RB, NOC};
                kwargs...)
    model = port.model
    scale_constr = model[:scale_constr]
    w = model[:w]
    count = length(rms)
    @variable(model, t_skew[1:count])
    @expression(model, skew_risk, t_skew .^ 2)
    for (i, rm) ∈ pairs(rms)
        V = if (isnothing(rm.V) || isempty(rm.V))
            port.V
        else
            rm.V
        end
        G = real(sqrt(V))
        model[Symbol("constr_skew_soc_$(i)")] = @constraint(model,
                                                            [scale_constr * t_skew[i];
                                                             scale_constr * G * w] ∈
                                                            SecondOrderCone())
        set_rm_risk_upper_bound(type, model, t_skew[i], rm.settings.ub, "skew_risk_$(i)")
        set_risk_expression(model, skew_risk[i], rm.settings.scale, rm.settings.flag)
    end
    return nothing
end
function set_rm(port::Portfolio, rm::SSkew, type::Union{Trad, RB, NOC}; kwargs...)
    model = port.model
    scale_constr = model[:scale_constr]
    w = model[:w]
    @variable(model, t_sskew)
    SV = if (isnothing(rm.V) || isempty(rm.V))
        port.SV
    else
        rm.V
    end
    G = real(sqrt(SV))
    @constraint(model, constr_sskew_soc,
                [scale_constr * t_sskew; scale_constr * G * w] ∈ SecondOrderCone())
    @expression(model, sskew_risk, t_sskew^2)
    set_rm_risk_upper_bound(type, model, t_sskew, rm.settings.ub, "sskew_risk")
    set_risk_expression(model, sskew_risk, rm.settings.scale, rm.settings.flag)
    return nothing
end
function set_rm(port::Portfolio, rms::AbstractVector{<:SSkew}, type::Union{Trad, RB, NOC};
                kwargs...)
    model = port.model
    scale_constr = model[:scale_constr]
    w = model[:w]
    count = length(rms)
    @variable(model, t_sskew[1:count])
    @expression(model, sskew_risk, t_sskew .^ 2)
    for (i, rm) ∈ pairs(rms)
        V = if (isnothing(rm.V) || isempty(rm.V))
            port.SV
        else
            rm.V
        end
        G = real(sqrt(V))
        model[Symbol("constr_sskew_soc_$(i)")] = @constraint(model,
                                                             [scale_constr * t_sskew[i];
                                                              scale_constr * G * w] ∈
                                                             SecondOrderCone())
        set_rm_risk_upper_bound(type, model, t_sskew[i], rm.settings.ub, "sskew_risk_$(i)")
        set_risk_expression(model, sskew_risk[i], rm.settings.scale, rm.settings.flag)
    end
    return nothing
end
function set_rm(port::Portfolio, rm::TrackingRM, type::Union{Trad, RB, NOC};
                returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
    scale_constr = model[:scale_constr]
    w = model[:w]
    k = model[:k]
    get_net_portfolio_returns(model, returns)
    net_X = model[:net_X]
    T = size(returns, 1)
    @variable(model, t_tracking_risk)
    @expression(model, tracking_risk, t_tracking_risk / sqrt(T - 1))
    tracking = rm.tr
    benchmark = tracking_error_benchmark(tracking, returns)
    @expression(model, tracking_error_rm, net_X .- benchmark * k)
    @constraint(model, constr_tracking_rm_soc,
                [scale_constr * t_tracking_risk; scale_constr * tracking_error_rm] ∈
                SecondOrderCone())
    set_rm_risk_upper_bound(type, model, tracking_risk, rm.settings.ub, "tracking_risk")
    set_risk_expression(model, tracking_risk, rm.settings.scale, rm.settings.flag)
    return nothing
end
function set_rm(port::Portfolio, rms::AbstractVector{<:TrackingRM},
                type::Union{Trad, RB, NOC}; returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
    scale_constr = model[:scale_constr]
    w = model[:w]
    k = model[:k]
    get_net_portfolio_returns(model, returns)
    net_X = model[:net_X]
    T = size(returns, 1)
    iTm1 = inv(sqrt(T - 1))
    count = length(rms)
    @variable(model, t_tracking_risk[1:count])
    @expression(model, tracking_risk[1:count], zero(AffExpr))
    for (i, rm) ∈ pairs(rms)
        tracking = rm.tr
        benchmark = tracking_error_benchmark(tracking, returns)
        tracking_error_rm = model[Symbol("tracking_error_rm_$(i)")] = @expression(model,
                                                                                  net_X .-
                                                                                  benchmark *
                                                                                  k)
        model[Symbol("constr_tracking_rm_soc_$(i)")] = @constraint(model,
                                                                   [scale_constr *
                                                                    t_tracking_risk[i]
                                                                    scale_constr *
                                                                    tracking_error_rm] ∈
                                                                   SecondOrderCone())
        add_to_expression!(tracking_risk[i], iTm1, t_tracking_risk[i])
        set_rm_risk_upper_bound(type, model, tracking_risk[i], rm.settings.ub,
                                "tracking_risk_$(i)")
        set_risk_expression(model, tracking_risk[i], rm.settings.scale, rm.settings.flag)
    end
    return nothing
end
function set_rm(port::Portfolio, rm::TurnoverRM, type::Union{Trad, RB, NOC};
                returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
    scale_constr = model[:scale_constr]
    w = model[:w]
    k = model[:k]
    N = length(w)
    @variable(model, turnover_risk)
    benchmark = rm.tr.w
    @expression(model, turnover_rm, w .- benchmark * k)
    @constraint(model, constr_turnover_rm_noc,
                [scale_constr * turnover_risk; scale_constr * turnover_rm] ∈
                MOI.NormOneCone(1 + N))
    set_rm_risk_upper_bound(type, model, turnover_risk, rm.settings.ub, "turnover_risk")
    set_risk_expression(model, turnover_risk, rm.settings.scale, rm.settings.flag)
    return nothing
end
function set_rm(port::Portfolio, rms::AbstractVector{<:TurnoverRM},
                type::Union{Trad, RB, NOC}; returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
    scale_constr = model[:scale_constr]
    w = model[:w]
    k = model[:k]
    N = length(w)
    count = length(rms)
    @variable(model, turnover_risk[1:count])
    @expression(model, turnover_rm[1:N, 1:count], zero(AffExpr))
    for (i, rm) ∈ pairs(rms)
        benchmark = rm.tr.w
        add_to_expression!.(view(turnover_rm, :, i), w .- benchmark * k)
        model[Symbol("constr_turnover_rm_noc_$(i)")] = @constraint(model,
                                                                   [scale_constr *
                                                                    turnover_risk[i]
                                                                    scale_constr *
                                                                    view(turnover_rm, :, i)] ∈
                                                                   MOI.NormOneCone(1 + N))
        set_rm_risk_upper_bound(type, model, turnover_risk[i], rm.settings.ub,
                                "turnover_risk_$(i)")
        set_risk_expression(model, turnover_risk[i], rm.settings.scale, rm.settings.flag)
    end
    return nothing
end
function risk_constraints(port, type::Union{Trad, RB, NOC},
                          rms::Union{RiskMeasure, AbstractVector}, mu, sigma, returns,
                          kelly_approx_idx = nothing)
    for rm ∈ rms
        set_rm(port, rm, type; mu = mu, sigma = sigma, returns = returns,
               kelly_approx_idx = kelly_approx_idx)
    end
    return nothing
end
