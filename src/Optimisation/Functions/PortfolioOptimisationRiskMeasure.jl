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
function get_ntwk_clust_type(port, a_rc, b_rc)
    model = port.model
    return if haskey(model, :constr_ntwk_sdp) ||
              haskey(model, :constr_clst_sdp) ||
              haskey(model, :rc_variance) ||
              !isnothing(a_rc) && !isnothing(b_rc) && !isempty(a_rc) && !isempty(b_rc)
        SDP()
    else
        NoAdj()
    end
end
function set_rm_risk_upper_bound(args...)
    return nothing
end
function set_rm_risk_upper_bound(::Union{Trad, FRC}, model, rm_risk, ub, key)
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
function calc_variance_risk(::SDP, ::Any, model, ::Any, sigma, ::Any)
    W = model[:W]
    w = model[:w]
    scale_constr = model[:scale_constr]
    @expressions(model, begin
                     sigma_W, sigma * W
                     variance_risk, tr(sigma_W)
                 end)
    return nothing
end
function setup_variance_risk(::SDP, model::JuMP.Model, count::Integer)
    @expression(model, variance_risk[1:count], zero(AffExpr))
    return nothing
end
function calc_variance_risk(::SDP, ::Any, model, ::Any, sigma, ::Any, idx::Integer)
    W = model[:W]
    variance_risk = model[:variance_risk][idx]
    sigma_W = model[Symbol("sigma_W_$(idx)")] = @expression(model, sigma * W)
    add_to_expression!(variance_risk, tr(sigma_W))
    return nothing
end
function calc_variance_risk(::Union{NoAdj, IP}, ::SOC, model::JuMP.Model, ::Any,
                            sigma::AbstractMatrix, ::Any)
    scale_constr = model[:scale_constr]
    w = model[:w]
    G = sqrt(sigma)
    @variable(model, dev)
    @expression(model, variance_risk, dev^2)
    @constraint(model, constr_dev_soc,
                [scale_constr * dev; scale_constr * G * w] ∈ SecondOrderCone())
    return nothing
end
function setup_variance_risk(::Union{NoAdj, IP}, model::JuMP.Model, count::Integer)
    @variable(model, dev[1:count])
    @expression(model, variance_risk[1:count], zero(QuadExpr))
    return nothing
end
function calc_variance_risk(::Union{NoAdj, IP}, ::SOC, model::JuMP.Model, ::Any,
                            sigma::AbstractMatrix, ::Any, idx::Integer)
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
function calc_variance_risk(::Union{NoAdj, IP}, ::Quad, model, ::Any, sigma, ::Any)
    scale_constr = model[:scale_constr]
    w = model[:w]
    G = sqrt(sigma)
    @variable(model, dev)
    @expression(model, variance_risk, dot(w, sigma, w))
    @constraint(model, constr_dev_soc,
                [scale_constr * dev; scale_constr * G * w] ∈ SecondOrderCone())
    return nothing
end
function calc_variance_risk(::Union{NoAdj, IP}, ::Quad, model::JuMP.Model, ::Any,
                            sigma::AbstractMatrix, ::Any, idx::Integer)
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
function calc_variance_risk(::Union{NoAdj, IP}, ::RSOC, model, mu, sigma, returns)
    T = size(returns, 1)
    scale_constr = model[:scale_constr]
    w = model[:w]
    get_net_portfolio_returns(model, returns)
    net_X = model[:net_X]
    G = sqrt(sigma)
    @variables(model, begin
                   tvariance
                   dev
               end)
    @expressions(model, begin
                     variance, net_X .- dot(mu, w)
                     variance_risk, tvariance / (T - one(T))
                 end)
    @constraints(model,
                 begin
                     constr_dev_soc,
                     [scale_constr * dev; scale_constr * G * w] ∈ SecondOrderCone()
                     constr_variance_rsoc,
                     [scale_constr * tvariance; 0.5; scale_constr * variance] in
                     RotatedSecondOrderCone()
                 end)
    return nothing
end
function calc_variance_risk(::Union{NoAdj, IP}, ::RSOC, model::JuMP.Model, mu,
                            sigma::AbstractMatrix, returns, idx::Integer)
    T = size(returns, 1)
    scale_constr = model[:scale_constr]
    w = model[:w]
    get_net_portfolio_returns(model, returns)
    net_X = model[:net_X]
    variance_risk = model[:variance_risk][idx]
    dev = model[:dev][idx]
    G = sqrt(sigma)
    tvariance = model[Symbol("tvariance_$(idx)")] = @variable(model)
    variance = model[Symbol("variance_$(idx)")] = @expression(model, net_X .- dot(mu, w))
    model[Symbol("constr_dev_soc_$(idx)")], model[Symbol("constr_variance_rsoc_$(idx)")] = @constraints(model,
                                                                                                        begin
                                                                                                            [scale_constr *
                                                                                                             dev
                                                                                                             scale_constr *
                                                                                                             G *
                                                                                                             w] ∈
                                                                                                            SecondOrderCone()
                                                                                                            [scale_constr *
                                                                                                             tvariance
                                                                                                             0.5;
                                                                                                             scale_constr *
                                                                                                             variance] in
                                                                                                            RotatedSecondOrderCone()
                                                                                                        end)
    add_to_expression!(variance_risk, inv(T - one(T)), tvariance)
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
function sdp_rc_variance(model, type::Union{Trad, RB, NOC}, a_rc, b_rc)
    rc_flag = !isnothing(a_rc) && !isnothing(b_rc) && !isempty(a_rc) && !isempty(b_rc)
    if rc_flag
        SDP_constraints(model, type)
    end
    return rc_flag
end
"""
```
set_rm(port, rm::RiskMeasure, type::Union{Trad, RB, NOC}; kwargs...)
set_rm(port, rm::AbstractVector{<:RiskMeasure}, type::Union{Trad, RB, NOC}; kwargs...)
```
"""
function set_rm(port, rm::Variance, type::Union{Trad, RB, NOC}; mu::AbstractVector{<:Real},
                sigma::AbstractMatrix{<:Real}, returns::AbstractMatrix{<:Real},
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
    a_rc = rm.a_rc
    b_rc = rm.b_rc
    rc_flag = sdp_rc_variance(model, type, a_rc, b_rc)
    adjacency_constraint = get_ntwk_clust_type(port, a_rc, b_rc)
    calc_variance_risk(adjacency_constraint, rm.formulation, model, mu, sigma, returns)
    variance_risk = model[:variance_risk]
    if rc_flag
        W = model[:W]
        sigma_W = model[:sigma_W]
        scale_constr = model[:scale_constr]
        @expression(model, rc_variance, true)
        @constraint(model, constr_rc_variance,
                    scale_constr * a_rc * vec(diag(sigma_W)) >=
                    scale_constr * b_rc * variance_risk)
    end
    var_bound_expr, var_bound_key = variance_risk_bounds_expr(adjacency_constraint, model)
    ub = variance_risk_bounds_val(adjacency_constraint, rm.settings.ub)
    set_rm_risk_upper_bound(type, model, var_bound_expr, ub, var_bound_key)
    set_risk_expression(model, variance_risk, rm.settings.scale, rm.settings.flag)
    return nothing
end
function set_rm(port, rms::AbstractVector{<:Variance}, type::Union{Trad, RB, NOC};
                mu::AbstractVector{<:Real}, sigma::AbstractMatrix{<:Real},
                returns::AbstractMatrix{<:Real},
                kelly_approx_idx::Union{AbstractVector{<:Integer}, Nothing}, kwargs...)
    model = port.model
    a_rc_flag = nothing
    b_rc_flag = nothing
    for rm ∈ rms
        a_rc = rm.a_rc
        b_rc = rm.b_rc
        rc_flag = sdp_rc_variance(model, type, a_rc, b_rc)
        if rc_flag
            a_rc_flag = Tuple(1)
            b_rc_flag = Tuple(1)
            break
        end
    end
    adjacency_constraint = get_ntwk_clust_type(port, a_rc_flag, b_rc_flag)
    count = length(rms)
    setup_variance_risk(adjacency_constraint, model, count)
    variance_risk = model[:variance_risk]
    var_bound_expr, var_bound_key = variance_risk_bounds_expr(adjacency_constraint, model)
    for (i, rm) ∈ pairs(rms)
        use_portfolio_sigma = (isnothing(rm.sigma) || isempty(rm.sigma))
        if !isnothing(kelly_approx_idx) && use_portfolio_sigma
            if isempty(kelly_approx_idx)
                push!(kelly_approx_idx, i)
            end
        end
        sigma_i = if !use_portfolio_sigma
            rm.sigma
        else
            sigma
        end
        calc_variance_risk(adjacency_constraint, rm.formulation, model, mu, sigma_i,
                           returns, i)
        if !isnothing(a_rc_flag)
            a_rc = rm.a_rc
            b_rc = rm.b_rc
            adjacency_constraint = get_ntwk_clust_type(port, a_rc, b_rc)
            if !isnothing(a_rc) && !isnothing(b_rc) && !isempty(a_rc) && !isempty(b_rc)
                if !haskey(model, :rc_variance)
                    @expression(model, rc_variance, true)
                end
                W = model[:W]
                sigma_W = model[Symbol("sigma_W_$(i)")]
                scale_constr = model[:scale_constr]
                model[Symbol("constr_rc_variance_$(i)")] = @constraint(model,
                                                                       scale_constr *
                                                                       a_rc *
                                                                       vec(diag(sigma_W)) >=
                                                                       scale_constr *
                                                                       b_rc *
                                                                       variance_risk[i])
            end
        end
        ub = variance_risk_bounds_val(adjacency_constraint, rm.settings.ub)
        set_rm_risk_upper_bound(type, model, var_bound_expr[i], ub, "$(var_bound_key)_$(i)")
        set_risk_expression(model, variance_risk[i], rm.settings.scale, rm.settings.flag)
    end
    return nothing
end
function set_rm(port, rm::Variance, type::FRC; mu::AbstractVector{<:Real},
                sigma::AbstractMatrix{<:Real}, returns::AbstractMatrix{<:Real},
                kelly_approx_idx::Union{AbstractVector{<:Integer}, Nothing},
                b1::AbstractMatrix, kwargs...)
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
    SDP_frc_constraints(model)
    W1 = model[:W1]
    @expressions(model, begin
                     frc_sigma_W, transpose(b1) * sigma * b1 * W1
                     frc_variance_risk, tr(frc_sigma_W)
                 end)
    a_rc = rm.a_rc
    b_rc = rm.b_rc
    if !isnothing(a_rc) && !isnothing(b_rc) && !isempty(a_rc) && !isempty(b_rc)
        W = model[:W1]
        scale_constr = model[:scale_constr]
        @constraint(model, constr_frc_variance,
                    scale_constr * a_rc * vec(diag(frc_sigma_W)) >=
                    scale_constr * b_rc * frc_variance_risk)
    end
    set_rm_risk_upper_bound(type, model, frc_variance_risk, rm.settings.ub,
                            "frc_variance_risk")
    set_risk_expression(model, frc_variance_risk, rm.settings.scale, rm.settings.flag)
    return nothing
end
function set_rm(port, rms::AbstractVector{<:Variance}, type::FRC;
                mu::AbstractVector{<:Real}, sigma::AbstractMatrix{<:Real},
                returns::AbstractMatrix{<:Real},
                kelly_approx_idx::Union{AbstractVector{<:Integer}, Nothing},
                b1::AbstractMatrix, kwargs...)
    model = port.model
    SDP_frc_constraints(model)
    W1 = model[:W1]
    count = length(rms)
    Nf = length(model[:w1])
    @expressions(model, begin
                     frc_sigma_W[1:Nf, 1:Nf, 1:count], zero(AffExpr)
                     frc_variance_risk[1:count], zero(AffExpr)
                 end)

    for (i, rm) ∈ pairs(rms)
        use_portfolio_sigma = (isnothing(rm.sigma) || isempty(rm.sigma))
        if !isnothing(kelly_approx_idx) && use_portfolio_sigma
            if isempty(kelly_approx_idx)
                push!(kelly_approx_idx, i)
            end
        end
        sigma_i = if !use_portfolio_sigma
            rm.sigma
        else
            sigma
        end
        add_to_expression!.(view(frc_sigma_W, :, :, i), transpose(b1) * sigma_i * b1 * W1)
        add_to_expression!(frc_variance_risk[i], tr(view(frc_sigma_W, :, :, i)))
        a_rc = rm.a_rc
        b_rc = rm.b_rc
        if !isnothing(a_rc) && !isnothing(b_rc) && !isempty(a_rc) && !isempty(b_rc)
            scale_constr = model[:scale_constr]
            @constraint(model, constr_frc_variance,
                        scale_constr * a_rc * vec(diag(view(frc_sigma_W, :, :, i))) >=
                        scale_constr * b_rc * frc_variance_risk[i])
        end
        set_rm_risk_upper_bound(type, model, frc_variance_risk[i], rm.settings.ub,
                                "frc_variance_risk_$(i)")
        set_risk_expression(model, frc_variance_risk[i], rm.settings.scale,
                            rm.settings.flag)
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
function wcvariance_risk_variables(::Box, model)
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
function wcvariance_risk_variables(::Ellipse, model)
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
function calc_wcvariance_risk(::Box, model, sigma, cov_l, cov_u, cov_sigma, k_sigma)
    Au = model[:Au]
    Al = model[:Al]
    @expression(model, wcvariance_risk, tr(Au * cov_u) - tr(Al * cov_l))
    return nothing
end
function calc_wcvariance_risk(::Ellipse, model, sigma, cov_l, cov_u, cov_sigma, k_sigma)
    scale_constr = model[:scale_constr]
    WpE = model[:WpE]
    G_sigma = sqrt(cov_sigma)
    @variable(model, t_ge)
    @expressions(model, begin
                     x_ge, G_sigma * vec(WpE)
                     wcvariance_risk, tr(sigma * WpE) + k_sigma * t_ge
                 end)
    @constraint(model, constr_ge_soc,
                [scale_constr * t_ge; scale_constr * x_ge] ∈ SecondOrderCone())
    return nothing
end
function calc_wcvariance_risk(::Box, model, sigma, cov_l, cov_u, cov_sigma, k_sigma,
                              wcvariance_risk, ::Any)
    Au = model[:Au]
    Al = model[:Al]
    add_to_expression!(wcvariance_risk, tr(Au * cov_u) - tr(Al * cov_l))
    return nothing
end
function calc_wcvariance_risk(::Ellipse, model, sigma, cov_l, cov_u, cov_sigma, k_sigma,
                              wcvariance_risk, i)
    scale_constr = model[:scale_constr]
    WpE = model[:WpE]
    G_sigma = sqrt(cov_sigma)
    model[Symbol("t_ge_$(i)")] = t_ge = @variable(model)
    model[Symbol("x_ge_$(i)")] = x_ge = @expression(model, G_sigma * vec(WpE))
    add_to_expression!(wcvariance_risk, tr(sigma * WpE))
    add_to_expression!(wcvariance_risk, k_sigma, t_ge)
    model[Symbol("constr_wcvariance_risk_$(i)")] = @constraint(model,
                                                               [scale_constr * t_ge;
                                                                scale_constr * x_ge] ∈
                                                               SecondOrderCone())
    return nothing
end
function set_rm(port, rm::WCVariance, type::Union{Trad, RB, NOC}; kwargs...)
    model = port.model
    SDP_constraints(model, type)
    sigma, cov_l, cov_u, cov_sigma, k_sigma = choose_wc_stats_port_rm(port, rm)
    wcvariance_risk_variables(rm.wc_set, model)
    calc_wcvariance_risk(rm.wc_set, model, sigma, cov_l, cov_u, cov_sigma, k_sigma)
    wcvariance_risk = model[:wcvariance_risk]
    set_rm_risk_upper_bound(type, model, wcvariance_risk, rm.settings.ub, "wcvariance_risk")
    set_risk_expression(model, wcvariance_risk, rm.settings.scale, rm.settings.flag)
    return nothing
end
function set_rm(port, rms::AbstractVector{<:WCVariance}, type::Union{Trad, RB, NOC};
                kwargs...)
    model = port.model
    SDP_constraints(model, type)
    count = length(rms)
    @expression(model, wcvariance_risk[1:count], zero(AffExpr))
    for (i, rm) ∈ pairs(rms)
        sigma, cov_l, cov_u, cov_sigma, k_sigma = choose_wc_stats_port_rm(port, rm)
        wcvariance_risk_variables(rm.wc_set, model)
        calc_wcvariance_risk(rm.wc_set, model, sigma, cov_l, cov_u, cov_sigma, k_sigma,
                             wcvariance_risk[i], i)
        calc_wcvariance_risk(rm.wc_set, model, sigma, cov_l, cov_u, cov_sigma, k_sigma)
        set_rm_risk_upper_bound(type, model, wcvariance_risk[i], rm.settings.ub,
                                "wcvariance_risk_$(i)")
        set_risk_expression(model, wcvariance_risk[i], rm.settings.scale, rm.settings.flag)
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
        sigma_i = if !use_portfolio_sigma
            rm.sigma
        else
            sigma
        end
        G = sqrt(sigma_i)
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
    k = model[:k]
    T = size(returns, 1)
    get_net_portfolio_returns(model, returns)
    net_X = model[:net_X]
    target = calc_rm_target(rm, mu, w, k)
    @variable(model, mad[1:T] .>= 0)
    we = rm.we
    if isnothing(we)
        @expression(model, mad_risk, mean(2 * mad))
    else
        @expression(model, mad_risk, mean(2 * mad, we))
    end
    @constraint(model, constr_mar_mad, scale_constr * (net_X .- target .+ mad) .>= 0)
    set_rm_risk_upper_bound(type, model, mad_risk, rm.settings.ub, "mad_risk")
    set_risk_expression(model, mad_risk, rm.settings.scale, rm.settings.flag)
    return nothing
end
function set_rm(port::Portfolio, rms::AbstractVector{<:MAD}, type::Union{Trad, RB, NOC};
                mu::AbstractVector{<:Real}, returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
    scale_constr = model[:scale_constr]
    w = model[:w]
    k = model[:k]
    T = size(returns, 1)
    get_net_portfolio_returns(model, returns)
    net_X = model[:net_X]
    count = length(rms)
    @variable(model, mad[1:T, 1:count] .>= 0)
    @expression(model, mar_mad[1:T, 1:count], zero(AffExpr))
    @expression(model, mad_risk[1:count], zero(AffExpr))
    for (i, rm) ∈ pairs(rms)
        target = calc_rm_target(rm, mu, w, k)
        mar_mad[:, i] .= @expression(model, net_X .- target .+ view(mad, :, i))
        model[Symbol("constr_mar_mad_$(i)")] = @constraint(model,
                                                           scale_constr *
                                                           view(mar_mad, :, i) .>= 0)
        we = rm.we
        if isnothing(we)
            add_to_expression!(mad_risk[i], mean(view(mad, :, i) + view(mar_mad, :, i)))
        else
            add_to_expression!(mad_risk[i], mean(view(mad, :, i) + view(mar_mad, :, i), we))
        end
        set_rm_risk_upper_bound(type, model, mad_risk[i], rm.settings.ub, "mad_risk_$(i)")
        set_risk_expression(model, mad_risk[i], rm.settings.scale, rm.settings.flag)
    end
    return nothing
end
function semi_variance_risk(::Quad, model::JuMP.Model, iTm1)
    svariance = model[:svariance]
    @expression(model, svariance_risk, iTm1 * dot(svariance, svariance))
    return nothing
end
function semi_variance_risk(::Quad, model, iTm1, i)
    svariance = view(model[:svariance], :, i)
    svariance_risk = model[:svariance_risk][i]
    add_to_expression!(svariance_risk, iTm1, dot(svariance, svariance))
    return nothing
end
function semi_variance_risk(::RSOC, model::JuMP.Model, iTm1)
    scale_constr = model[:scale_constr]
    svariance = model[:svariance]
    @variable(model, tsvariance)
    @constraint(model, constr_svariance_rsoc,
                [scale_constr * tsvariance; 0.5; scale_constr * svariance] in
                RotatedSecondOrderCone())
    @expression(model, svariance_risk, iTm1 * tsvariance)
    return nothing
end
function semi_variance_risk(::RSOC, model, iTm1, i)
    scale_constr = model[:scale_constr]
    svariance = view(model[:svariance], :, i)
    svariance_risk = model[:svariance_risk][i]
    model[Symbol("tsvariance_$(i)")] = tsvariance = @variable(model)
    model[Symbol("constr_svariance_rsoc_$(i)")] = @constraint(model,
                                                              [scale_constr * tsvariance;
                                                               0.5;
                                                               scale_constr * svariance] in
                                                              RotatedSecondOrderCone())
    add_to_expression!(svariance_risk, iTm1, tsvariance)
    return nothing
end
function semi_variance_risk(::SOC, model::JuMP.Model, iTm1)
    sdev = model[:sdev]
    @expression(model, svariance_risk, iTm1 * sdev^2)
    return nothing
end
function semi_variance_risk(::SOC, model, iTm1, i)
    sdev = model[:sdev][i]
    svariance_risk = model[:svariance_risk][i]
    add_to_expression!(svariance_risk, iTm1, sdev^2)
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
    target = calc_rm_target(rm, mu, w, k)
    @variables(model, begin
                   svariance[1:T] .>= 0
                   sdev
               end)
    Tm1 = T - one(T)
    iTm1 = inv(Tm1)
    srtTm1 = sqrt(Tm1)
    semi_variance_risk(rm.formulation, model, iTm1)
    @constraints(model,
                 begin
                     constr_svariance_mar,
                     scale_constr * (net_X .- target) .>= scale_constr * -svariance
                     constr_svar_soc,
                     [scale_constr * sdev; scale_constr * svariance] ∈ SecondOrderCone()
                 end)
    svariance_risk = model[:svariance_risk]
    set_rm_risk_upper_bound(type, model, sdev, sqrt(rm.settings.ub) * srtTm1, "sdev")
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
    Tm1 = T - one(T)
    iTm1 = inv(Tm1)
    srtTm1 = sqrt(Tm1)
    count = length(rms)
    @variables(model, begin
                   svariance[1:T, 1:count] .>= 0
                   sdev[1:count]
               end)
    @expression(model, svariance_risk[1:count], zero(QuadExpr))
    for (i, rm) ∈ pairs(rms)
        target = calc_rm_target(rm, mu, w, k)
        model[Symbol("constr_svariance_mar_$(i)")] = @constraint(model,
                                                                 scale_constr *
                                                                 (net_X .- target) .>=
                                                                 scale_constr *
                                                                 -view(svariance, :, i))
        model[Symbol("constr_svar_soc_$(i)")] = @constraint(model,
                                                            [scale_constr * sdev[i];
                                                             scale_constr *
                                                             view(svariance, :, i)] ∈
                                                            SecondOrderCone())
        semi_variance_risk(rm.formulation, model, iTm1, i)
        set_rm_risk_upper_bound(type, model, sdev[i], sqrt(rm.settings.ub) * srtTm1,
                                "sdev_$(i)")
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
    target = calc_rm_target(rm, mu, w, k)
    @variables(model, begin
                   tssd[1:T] .>= 0
                   ssd
               end)
    @expression(model, ssd_risk, ssd / sqrt(T - one(T)))
    @constraints(model,
                 begin
                     constr_tssd_mar,
                     scale_constr * (net_X .- target) .>= scale_constr * -tssd
                     constr_ssd_soc,
                     [scale_constr * ssd; scale_constr * tssd] ∈ SecondOrderCone()
                 end)
    set_rm_risk_upper_bound(type, model, ssd_risk, rm.settings.ub, "ssd_risk")
    set_risk_expression(model, ssd_risk, rm.settings.scale, rm.settings.flag)

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
    iTm1 = inv(sqrt(T - one(T)))
    count = length(rms)
    @variable(model, tssd[1:T, 1:count] .>= 0)
    @variable(model, ssd[1:count])
    @expression(model, ssd_risk[1:count], zero(AffExpr))
    for (i, rm) ∈ pairs(rms)
        target = calc_rm_target(rm, mu, w, k)
        model[Symbol("constr_tssd_mar_$(i)")], model[Symbol("constr_ssd_soc_$(i)")] = @constraints(model,
                                                                                                   begin
                                                                                                       scale_constr *
                                                                                                       (net_X .-
                                                                                                        target) .>=
                                                                                                       scale_constr *
                                                                                                       -view(tssd,
                                                                                                             :,
                                                                                                             i)
                                                                                                       [scale_constr *
                                                                                                        ssd[i]
                                                                                                        scale_constr *
                                                                                                        view(tssd,
                                                                                                             :,
                                                                                                             i)] ∈
                                                                                                       SecondOrderCone()
                                                                                                   end)
        add_to_expression!(ssd_risk[i], iTm1, ssd[i])
        set_rm_risk_upper_bound(type, model, ssd_risk[i], rm.settings.ub, "ssd_risk_$(i)")
        set_risk_expression(model, ssd_risk[i], rm.settings.scale, rm.settings.flag)
    end
    return nothing
end
"""
"""
function calc_rm_target(rm, mu, w, k)
    target = rm.target
    return if isnothing(target) || isa(target, AbstractVector) && isempty(target)
        mu_i = rm.mu
        if isnothing(mu_i) || isempty(mu_i)
            dot(mu, w)
        else
            dot(mu_i, w)
        end
    else
        target * k
    end
end
function set_rm(port::Portfolio, rm::FLPM, type::Union{Trad, RB, NOC};
                returns::AbstractMatrix{<:Real}, mu::AbstractVector{<:Real}, kwargs...)
    model = port.model
    scale_constr = model[:scale_constr]
    w = model[:w]
    k = model[:k]
    T = size(returns, 1)
    get_net_portfolio_returns(model, returns)
    net_X = model[:net_X]
    target = calc_rm_target(rm, mu, w, k)
    @variable(model, flpm[1:T] .>= 0)
    @expression(model, flpm_risk, sum(flpm) / T)
    @constraint(model, constr_flpm,
                scale_constr * flpm .>= scale_constr * (target .- net_X))
    set_rm_risk_upper_bound(type, model, flpm_risk, rm.settings.ub, "flpm_risk")
    set_risk_expression(model, flpm_risk, rm.settings.scale, rm.settings.flag)
    return nothing
end
function set_rm(port::Portfolio, rms::AbstractVector{<:FLPM}, type::Union{Trad, RB, NOC};
                returns::AbstractMatrix{<:Real}, mu::AbstractVector{<:Real}, kwargs...)
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
        target = calc_rm_target(rm, mu, w, k)
        add_to_expression!(flpm_risk[i], iT, sum(view(flpm, :, i)))
        model[Symbol("constr_flpm_$(i)")] = @constraint(model,
                                                        scale_constr * view(flpm, :, i) .>=
                                                        scale_constr * (target .- net_X))
        set_rm_risk_upper_bound(type, model, flpm_risk[i], rm.settings.ub, "flpm_risk_$(i)")
        set_risk_expression(model, flpm_risk[i], rm.settings.scale, rm.settings.flag)
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
                     cvarrg_risk, cvar_risk_l - cvar_risk_h
                 end)
    @constraints(model,
                 begin
                     constr_cvar_l,
                     scale_constr * z_cvar_l .>= scale_constr * (-net_X .- var_l)
                     constr_cvar_h,
                     scale_constr * z_cvar_h .<= scale_constr * (-net_X .- var_h)
                 end)

    set_rm_risk_upper_bound(type, model, cvarrg_risk, rm.settings.ub, "cvarrg_risk")
    set_risk_expression(model, cvarrg_risk, rm.settings.scale, rm.settings.flag)
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
                     cvarrg_risk[1:count], zero(AffExpr)
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
        add_to_expression!(cvarrg_risk[i], cvar_risk_l[i])
        add_to_expression!(cvarrg_risk[i], -1, cvar_risk_h[i])
        set_rm_risk_upper_bound(type, model, cvarrg_risk[i], rm.settings.ub,
                                "rcvar_risk_$(i)")
        set_risk_expression(model, cvarrg_risk[i], rm.settings.scale, rm.settings.flag)
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
function set_rm(port::Portfolio, rm::EVaRRG, type::Union{Trad, RB, NOC};
                returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
    scale_constr = model[:scale_constr]
    get_net_portfolio_returns(model, returns)
    net_X = model[:net_X]
    T = size(returns, 1)
    at = rm.alpha * T
    bt = rm.beta * T
    @variables(model, begin
                   t_evar_l
                   z_evar_l >= 0
                   u_evar_l[1:T]
                   t_evar_h
                   z_evar_h <= 0
                   u_evar_h[1:T]
               end)
    @expressions(model, begin
                     evar_risk_l, t_evar_l - z_evar_l * log(at)
                     evar_risk_h, t_evar_h - z_evar_h * log(bt)
                     evarrg_risk, evar_risk_l - evar_risk_h
                 end)

    @constraints(model,
                 begin
                     constr_evar_l, scale_constr * sum(u_evar_l) <= scale_constr * z_evar_l
                     constr_evar_l_exp_cone[i = 1:T],
                     [scale_constr * (-net_X[i] - t_evar_l), scale_constr * z_evar_l,
                      scale_constr * u_evar_l[i]] ∈ MOI.ExponentialCone()
                     constr_evar_h, scale_constr * sum(u_evar_h) >= scale_constr * z_evar_h
                     constr_evar_h_exp_cone[i = 1:T],
                     [scale_constr * (net_X[i] + t_evar_h), scale_constr * -z_evar_h,
                      scale_constr * -u_evar_h[i]] ∈ MOI.ExponentialCone()
                 end)
    set_rm_risk_upper_bound(type, model, evarrg_risk, rm.settings.ub, "evarrg_risk")
    set_risk_expression(model, evarrg_risk, rm.settings.scale, rm.settings.flag)

    return nothing
end
function set_rm(port::Portfolio, rms::AbstractVector{<:EVaRRG}, type::Union{Trad, RB, NOC};
                returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
    scale_constr = model[:scale_constr]
    get_net_portfolio_returns(model, returns)
    net_X = model[:net_X]
    T = size(returns, 1)
    count = length(rms)
    @variables(model, begin
                   t_evar_l[1:count]
                   z_evar_l[1:count] .>= 0
                   u_evar_l[1:T, 1:count]
                   t_evar_h[1:count]
                   z_evar_h[1:count] .<= 0
                   u_evar_h[1:T, 1:count]
               end)
    @expressions(model, begin
                     evar_risk_l[1:count], zero(AffExpr)
                     evar_risk_h[1:count], zero(AffExpr)
                     evarrg_risk[1:count], zero(AffExpr)
                 end)
    for (j, rm) ∈ pairs(rms)
        at = rm.alpha * T
        bt = rm.beta * T
        model[Symbol("constr_evar_l_$(j)")], model[Symbol("constr_evar_l_exp_cone_$(j)")] = @constraints(model,
                                                                                                         begin
                                                                                                             scale_constr *
                                                                                                             sum(view(u_evar_l,
                                                                                                                      :,
                                                                                                                      j)) <=
                                                                                                             scale_constr *
                                                                                                             z_evar_l[j]
                                                                                                             [i = 1:T],
                                                                                                             [scale_constr *
                                                                                                              (-net_X[i] -
                                                                                                               t_evar_l[j]),
                                                                                                              scale_constr *
                                                                                                              z_evar_l[j],
                                                                                                              scale_constr *
                                                                                                              u_evar_l[i,
                                                                                                                       j]] ∈
                                                                                                             MOI.ExponentialCone()
                                                                                                         end)
        add_to_expression!(evar_risk_l[j], t_evar_l[j])
        add_to_expression!(evar_risk_l[j], -log(at), z_evar_l[j])
        model[Symbol("constr_evar_h_$(j)")], model[Symbol("constr_evar_h_exp_cone_$(j)")] = @constraints(model,
                                                                                                         begin
                                                                                                             scale_constr *
                                                                                                             sum(view(u_evar_h,
                                                                                                                      :,
                                                                                                                      j)) >=
                                                                                                             scale_constr *
                                                                                                             z_evar_h[j]
                                                                                                             [i = 1:T],
                                                                                                             [scale_constr *
                                                                                                              (net_X[i] +
                                                                                                               t_evar_h[j]),
                                                                                                              scale_constr *
                                                                                                              -z_evar_h[j],
                                                                                                              scale_constr *
                                                                                                              -u_evar_h[i,
                                                                                                                        j]] ∈
                                                                                                             MOI.ExponentialCone()
                                                                                                         end)
        add_to_expression!(evar_risk_h[j], t_evar_h[j])
        add_to_expression!(evar_risk_h[j], -log(bt), z_evar_h[j])
        add_to_expression!(evarrg_risk[j], evar_risk_l[j])
        add_to_expression!(evarrg_risk[j], -evar_risk_h[j])
        set_rm_risk_upper_bound(type, model, evarrg_risk[j], rm.settings.ub,
                                "evarrg_risk_$(j)")
        set_risk_expression(model, evarrg_risk[j], rm.settings.scale, rm.settings.flag)
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
function set_rm(port::Portfolio, rm::RLVaRRG, type::Union{Trad, RB, NOC};
                returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
    scale_constr = model[:scale_constr]
    get_net_portfolio_returns(model, returns)
    net_X = model[:net_X]
    T = size(returns, 1)

    iat = inv(rm.alpha * T)
    lnk_a = (iat^rm.kappa_a - iat^(-rm.kappa_a)) / (2 * rm.kappa_a)
    opk_a = one(rm.kappa_a) + rm.kappa_a
    omk_a = one(rm.kappa_a) - rm.kappa_a
    ik2_a = inv(2 * rm.kappa_a)
    ik_a = inv(rm.kappa_a)
    iopk_a = inv(opk_a)
    iomk_a = inv(omk_a)
    ibt = inv(rm.beta * T)
    lnk_b = (ibt^rm.kappa_b - ibt^(-rm.kappa_b)) / (2 * rm.kappa_b)
    opk_b = one(rm.kappa_b) + rm.kappa_b
    omk_b = one(rm.kappa_b) - rm.kappa_b
    ik2_b = inv(2 * rm.kappa_b)
    ik_b = inv(rm.kappa_b)
    iopk_b = inv(opk_b)
    iomk_b = inv(omk_b)
    @variables(model, begin
                   t_rlvar_l
                   z_rlvar_l >= 0
                   omega_rlvar_l[1:T]
                   psi_rlvar_l[1:T]
                   theta_rlvar_l[1:T]
                   epsilon_rlvar_l[1:T]
                   t_rlvar_h
                   z_rlvar_h <= 0
                   omega_rlvar_h[1:T]
                   psi_rlvar_h[1:T]
                   theta_rlvar_h[1:T]
                   epsilon_rlvar_h[1:T]
               end)
    @expressions(model,
                 begin
                     rlvar_risk_l,
                     t_rlvar_l + lnk_a * z_rlvar_l + sum(psi_rlvar_l .+ theta_rlvar_l)
                     rlvar_risk_h,
                     t_rlvar_h + lnk_b * z_rlvar_h + sum(psi_rlvar_h .+ theta_rlvar_h)
                     rlvarrg_risk, rlvar_risk_l - rlvar_risk_h
                 end)
    @constraints(model,
                 begin
                     constr_rlvar_l_pcone_a[i = 1:T],
                     [scale_constr * z_rlvar_l * opk_a * ik2_a,
                      scale_constr * psi_rlvar_l[i] * opk_a * ik_a,
                      scale_constr * epsilon_rlvar_l[i]] ∈ MOI.PowerCone(iopk_a)
                     constr_rlvar_l_pcone_b[i = 1:T],
                     [scale_constr * omega_rlvar_l[i] * iomk_a,
                      scale_constr * theta_rlvar_l[i] * ik_a,
                      scale_constr * -z_rlvar_l * ik2_a] ∈ MOI.PowerCone(omk_a)
                     constr_rlvar_l,
                     scale_constr *
                     (-net_X .- t_rlvar_l .+ epsilon_rlvar_l .+ omega_rlvar_l) .<= 0

                     constr_rlvar_h_pcone_a[i = 1:T],
                     [scale_constr * -z_rlvar_h * opk_b * ik2_b,
                      scale_constr * -psi_rlvar_h[i] * opk_b * ik_b,
                      scale_constr * -epsilon_rlvar_h[i]] ∈ MOI.PowerCone(iopk_b)
                     constr_rlvar_h_pcone_b[i = 1:T],
                     [scale_constr * -omega_rlvar_h[i] * iomk_b,
                      scale_constr * -theta_rlvar_h[i] * ik_b,
                      scale_constr * z_rlvar_h * ik2_b] ∈ MOI.PowerCone(omk_b)
                     constr_rlvar_h,
                     scale_constr *
                     (net_X .+ t_rlvar_h .- epsilon_rlvar_h .- omega_rlvar_h) .<= 0
                 end)
    set_rm_risk_upper_bound(type, model, rlvarrg_risk, rm.settings.ub, "rlvarrg_risk")
    set_risk_expression(model, rlvarrg_risk, rm.settings.scale, rm.settings.flag)
    return nothing
end
function set_rm(port::Portfolio, rms::AbstractVector{<:RLVaRRG}, type::Union{Trad, RB, NOC};
                returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
    scale_constr = model[:scale_constr]
    get_net_portfolio_returns(model, returns)
    net_X = model[:net_X]
    T = size(returns, 1)
    count = length(rms)
    @variables(model, begin
                   t_rlvar_l[1:count]
                   z_rlvar_l[1:count] .>= 0
                   omega_rlvar_l[1:T, 1:count]
                   psi_rlvar_l[1:T, 1:count]
                   theta_rlvar_l[1:T, 1:count]
                   epsilon_rlvar_l[1:T, 1:count]
                   t_rlvar_h[1:count]
                   z_rlvar_h[1:count] .<= 0
                   omega_rlvar_h[1:T, 1:count]
                   psi_rlvar_h[1:T, 1:count]
                   theta_rlvar_h[1:T, 1:count]
                   epsilon_rlvar_h[1:T, 1:count]
               end)
    @expressions(model, begin
                     rlvar_risk_l[1:count], zero(AffExpr)
                     rlvar_risk_h[1:count], zero(AffExpr)
                     rlvarrg_risk[1:count], zero(AffExpr)
                 end)
    for (j, rm) ∈ pairs(rms)
        iat = inv(rm.alpha * T)
        lnk_a = (iat^rm.kappa_a - iat^(-rm.kappa_a)) / (2 * rm.kappa_a)
        opk_a = one(rm.kappa_a) + rm.kappa_a
        omk_a = one(rm.kappa_a) - rm.kappa_a
        ik2_a = inv(2 * rm.kappa_a)
        ik_a = inv(rm.kappa_a)
        iopk_a = inv(opk_a)
        iomk_a = inv(omk_a)
        ibt = inv(rm.beta * T)
        lnk_b = (ibt^rm.kappa_b - ibt^(-rm.kappa_b)) / (2 * rm.kappa_b)
        opk_b = one(rm.kappa_b) + rm.kappa_b
        omk_b = one(rm.kappa_b) - rm.kappa_b
        ik2_b = inv(2 * rm.kappa_b)
        ik_b = inv(rm.kappa_b)
        iopk_b = inv(opk_b)
        iomk_b = inv(omk_b)

        model[Symbol("constr_rlvar_l_pcone_a_$(j)")], model[Symbol("constr_rlvar_l_pcone_b_$(j)")], model[Symbol("constr_rlvar_l_$(j)")] = @constraints(model,
                                                                                                                                                        begin
                                                                                                                                                            [i = 1:T],
                                                                                                                                                            [scale_constr *
                                                                                                                                                             z_rlvar_l[j] *
                                                                                                                                                             opk_a *
                                                                                                                                                             ik2_a,
                                                                                                                                                             scale_constr *
                                                                                                                                                             psi_rlvar_l[i,
                                                                                                                                                                         j] *
                                                                                                                                                             opk_a *
                                                                                                                                                             ik_a,
                                                                                                                                                             scale_constr *
                                                                                                                                                             epsilon_rlvar_l[i,
                                                                                                                                                                             j]] ∈
                                                                                                                                                            MOI.PowerCone(iopk_a)
                                                                                                                                                            [i = 1:T],
                                                                                                                                                            [scale_constr *
                                                                                                                                                             omega_rlvar_l[i,
                                                                                                                                                                           j] *
                                                                                                                                                             iomk_a,
                                                                                                                                                             scale_constr *
                                                                                                                                                             theta_rlvar_l[i,
                                                                                                                                                                           j] *
                                                                                                                                                             ik_a,
                                                                                                                                                             scale_constr *
                                                                                                                                                             -z_rlvar_l[j] *
                                                                                                                                                             ik2_a] ∈
                                                                                                                                                            MOI.PowerCone(omk_a)
                                                                                                                                                            scale_constr *
                                                                                                                                                            (-net_X .-
                                                                                                                                                             t_rlvar_l[j] .+
                                                                                                                                                             view(epsilon_rlvar_l,
                                                                                                                                                                  :,
                                                                                                                                                                  j) .+
                                                                                                                                                             view(omega_rlvar_l,
                                                                                                                                                                  :,
                                                                                                                                                                  j)) .<=
                                                                                                                                                            0
                                                                                                                                                        end)
        add_to_expression!(rlvar_risk_l[j], t_rlvar_l[j])
        add_to_expression!(rlvar_risk_l[j], lnk_a, z_rlvar_l[j])
        add_to_expression!(rlvar_risk_l[j],
                           sum(view(psi_rlvar_l, :, j) .+ view(theta_rlvar_l, :, j)))

        model[Symbol("constr_rlvar_h_pcone_a_$(j)")], model[Symbol("constr_rlvar_h_pcone_b_$(j)")], model[Symbol("constr_rlvar_h_$(j)")] = @constraints(model,
                                                                                                                                                        begin
                                                                                                                                                            [i = 1:T],
                                                                                                                                                            [scale_constr *
                                                                                                                                                             -z_rlvar_h[j] *
                                                                                                                                                             opk_b *
                                                                                                                                                             ik2_b,
                                                                                                                                                             scale_constr *
                                                                                                                                                             -psi_rlvar_h[i,
                                                                                                                                                                          j] *
                                                                                                                                                             opk_b *
                                                                                                                                                             ik_b,
                                                                                                                                                             scale_constr *
                                                                                                                                                             -epsilon_rlvar_h[i,
                                                                                                                                                                              j]] ∈
                                                                                                                                                            MOI.PowerCone(iopk_b)
                                                                                                                                                            [i = 1:T],
                                                                                                                                                            [scale_constr *
                                                                                                                                                             -omega_rlvar_h[i,
                                                                                                                                                                            j] *
                                                                                                                                                             iomk_b,
                                                                                                                                                             scale_constr *
                                                                                                                                                             -theta_rlvar_h[i,
                                                                                                                                                                            j] *
                                                                                                                                                             ik_b,
                                                                                                                                                             scale_constr *
                                                                                                                                                             z_rlvar_h[j] *
                                                                                                                                                             ik2_b] ∈
                                                                                                                                                            MOI.PowerCone(omk_b)
                                                                                                                                                            scale_constr *
                                                                                                                                                            (net_X .+
                                                                                                                                                             t_rlvar_h[j] .-
                                                                                                                                                             view(epsilon_rlvar_h,
                                                                                                                                                                  :,
                                                                                                                                                                  j) .-
                                                                                                                                                             view(omega_rlvar_h,
                                                                                                                                                                  :,
                                                                                                                                                                  j)) .<=
                                                                                                                                                            0
                                                                                                                                                        end)
        add_to_expression!(rlvar_risk_h[j], t_rlvar_h[j])
        add_to_expression!(rlvar_risk_h[j], lnk_b, z_rlvar_h[j])
        add_to_expression!(rlvar_risk_h[j],
                           sum(view(psi_rlvar_h, :, j) .+ view(theta_rlvar_h, :, j)))

        add_to_expression!(rlvarrg_risk[j], rlvar_risk_l[j])
        add_to_expression!(rlvarrg_risk[j], -rlvar_risk_h[j])

        set_rm_risk_upper_bound(type, model, rlvarrg_risk[j], rm.settings.ub,
                                "rlvarrg_risk_$(j)")
        set_risk_expression(model, rlvarrg_risk[j], rm.settings.scale, rm.settings.flag)
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
            zkurt[:, idx] .= @expression(model, L_2 * vec(W))
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
            zskurt[:, idx] .= @expression(model, L_2 * vec(W))
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
function gmd_risk_formulation(::OWAExact, model, returns)
    scale_constr = model[:scale_constr]
    T = size(returns, 1)
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
    return gmd_risk
end
function gmd_risk_formulation(formulation::OWAApprox, model, returns)
    scale_constr = model[:scale_constr]
    T = size(returns, 1)
    get_net_portfolio_returns(model, returns)
    net_X = model[:net_X]
    owa_p = formulation.p
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

    return gmd_risk
end
function set_rm(port::Portfolio, rm::GMD, type::Union{Trad, RB, NOC};
                returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
    gmd_risk = gmd_risk_formulation(rm.formulation, model, returns)
    set_rm_risk_upper_bound(type, model, gmd_risk, rm.settings.ub, "gmd_risk")
    set_risk_expression(model, gmd_risk, rm.settings.scale, rm.settings.flag)
    return nothing
end
function tg_risk_formulation(::OWAExact, model, returns, T, alpha, a_sim, alpha_i)
    scale_constr = model[:scale_constr]
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
    return tg_risk
end
function tg_risk_formulation(formulation::OWAApprox, model, returns, T, alpha, a_sim,
                             alpha_i)
    scale_constr = model[:scale_constr]
    get_net_portfolio_returns(model, returns)
    net_X = model[:net_X]
    owa_p = formulation.p
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
                     scale_constr *
                     (net_X .+ tg_t .- tg_nu .+ tg_eta .- vec(sum(tg_epsilon; dims = 2))) .==
                     0
                     constr_approx_2_tg,
                     scale_constr * (tg_z .+ tg_y) .==
                     scale_constr * vec(sum(tg_psi; dims = 1))
                     constr_approx_tg_pcone[i = 1:M, j = 1:T],
                     [scale_constr * -tg_z[i] * owa_p[i],
                      scale_constr * tg_psi[j, i] * owa_p[i] / (owa_p[i] - 1),
                      scale_constr * tg_epsilon[j, i]] ∈ MOI.PowerCone(inv(owa_p[i]))
                 end)
    return tg_risk
end
function set_rm(port::Portfolio, rm::TG, type::Union{Trad, RB, NOC};
                returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
    scale_constr = model[:scale_constr]
    T = size(returns, 1)
    alpha = rm.alpha
    a_sim = rm.a_sim
    alpha_i = rm.alpha_i
    tg_risk = tg_risk_formulation(rm.formulation, model, returns, T, alpha, a_sim, alpha_i)
    set_rm_risk_upper_bound(type, model, tg_risk, rm.settings.ub, "tg_risk")
    set_risk_expression(model, tg_risk, rm.settings.scale, rm.settings.flag)
    return nothing
end
function tg_risk_vec_formulation(::OWAExact, model, returns, T, count, alpha, a_sim,
                                 alpha_i, idx, tg_risk)
    scale_constr = model[:scale_constr]
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
                                                    scale_constr * owa * transpose(tg_w) .<=
                                                    scale_constr *
                                                    (ovec * transpose(view(tga, :, idx)) +
                                                     view(tgb, :, idx) * transpose(ovec)))
    return nothing
end
function tg_risk_vec_formulation(formulation::OWAApprox, model, returns, T, count, alpha,
                                 a_sim, alpha_i, idx, tg_risk)
    scale_constr = model[:scale_constr]
    get_net_portfolio_returns(model, returns)
    net_X = model[:net_X]
    owa_p = formulation.p
    M = length(owa_p)

    tg_w = -owa_tg(T; alpha_i = alpha_i, alpha = alpha, a_sim = a_sim)
    tg_s = sum(tg_w)
    tg_l = minimum(tg_w)
    tg_h = maximum(tg_w)
    tg_d = [norm(tg_w, p) for p ∈ owa_p]
    if !haskey(model, :tg_t)
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
        tg_risk_vec_formulation(rm.formulation, model, returns, T, count, alpha, a_sim,
                                alpha_i, idx, tg_risk)
        set_rm_risk_upper_bound(type, model, tg_risk[idx], rm.settings.ub, "tg_risk_$(idx)")
        set_risk_expression(model, tg_risk[idx], rm.settings.scale, rm.settings.flag)
    end
    return nothing
end
function tgrg_risk_formulation(::OWAExact, model, returns, T, alpha, a_sim, alpha_i, beta,
                               b_sim, beta_i)
    scale_constr = model[:scale_constr]
    OWA_constraints(model, returns)
    owa = model[:owa]
    ovec = range(1; stop = 1, length = T)
    @variables(model, begin
                   tgrga[1:T]
                   tgrgb[1:T]
               end)
    @expression(model, tgrg_risk, sum(tgrga .+ tgrgb))
    tgrg_w = owa_tgrg(T; alpha_i = alpha_i, alpha = alpha, a_sim = a_sim, beta_i = beta_i,
                      beta = beta, b_sim = b_sim)
    @constraint(model, constr_tgrg,
                scale_constr * owa * transpose(tgrg_w) .<=
                scale_constr * (ovec * transpose(tgrga) + tgrgb * transpose(ovec)))
    return tgrg_risk
end
function tgrg_risk_formulation(formulation::OWAApprox, model, returns, T, alpha, a_sim,
                               alpha_i, beta, b_sim, beta_i)
    scale_constr = model[:scale_constr]
    get_net_portfolio_returns(model, returns)
    net_X = model[:net_X]
    owa_p = formulation.p
    M = length(owa_p)

    @variables(model, begin
                   tgrg_l_t
                   tgrg_l_nu[1:T] .>= 0
                   tgrg_l_eta[1:T] .>= 0
                   tgrg_l_epsilon[1:T, 1:M]
                   tgrg_l_psi[1:T, 1:M]
                   tgrg_l_z[1:M]
                   tgrg_l_y[1:M] .>= 0
               end)

    tgrg_l_w = -owa_tg(T; alpha_i = alpha_i, alpha = alpha, a_sim = a_sim)
    tgrg_l_s = sum(tgrg_l_w)
    tgrg_l_l = minimum(tgrg_l_w)
    tgrg_l_h = maximum(tgrg_l_w)
    tgrg_l_d = [norm(tgrg_l_w, p) for p ∈ owa_p]

    @expression(model, tgrg_l_risk,
                tgrg_l_s * tgrg_l_t - tgrg_l_l * sum(tgrg_l_nu) +
                tgrg_l_h * sum(tgrg_l_eta) +
                dot(tgrg_l_d, tgrg_l_y))
    @constraints(model,
                 begin
                     constr_approx_1_tgrg,
                     scale_constr * (net_X .+ tgrg_l_t .- tgrg_l_nu .+ tgrg_l_eta .-
                                     vec(sum(tgrg_l_epsilon; dims = 2))) .== 0
                     constr_approx_2_tgrg,
                     scale_constr * (tgrg_l_z .+ tgrg_l_y) .==
                     scale_constr * vec(sum(tgrg_l_psi; dims = 1))
                     constr_approx_1_tgrg_pcone[i = 1:M, j = 1:T],
                     [scale_constr * -tgrg_l_z[i] * owa_p[i],
                      scale_constr * tgrg_l_psi[j, i] * owa_p[i] / (owa_p[i] - 1),
                      scale_constr * tgrg_l_epsilon[j, i]] ∈ MOI.PowerCone(inv(owa_p[i]))
                 end)

    @variables(model, begin
                   tgrg_h_t
                   tgrg_h_nu[1:T] .>= 0
                   tgrg_h_eta[1:T] .>= 0
                   tgrg_h_epsilon[1:T, 1:M]
                   tgrg_h_psi[1:T, 1:M]
                   tgrg_h_z[1:M]
                   tgrg_h_y[1:M] .>= 0
               end)

    tgrg_h_w = -owa_tg(T; alpha_i = beta_i, alpha = beta, a_sim = b_sim)
    tgrg_h_s = sum(tgrg_h_w)
    tgrg_h_l = minimum(tgrg_h_w)
    tgrg_h_h = maximum(tgrg_h_w)
    tgrg_h_d = [norm(tgrg_h_w, p) for p ∈ owa_p]

    @expressions(model,
                 begin
                     tgrg_h_risk,
                     tgrg_h_s * tgrg_h_t - tgrg_h_l * sum(tgrg_h_nu) +
                     tgrg_h_h * sum(tgrg_h_eta) +
                     dot(tgrg_h_d, tgrg_h_y)
                     tgrg_risk, tgrg_l_risk + tgrg_h_risk
                 end)
    @constraints(model,
                 begin
                     constr_approx_3_tgrg,
                     scale_constr * (-net_X .+ tgrg_h_t .- tgrg_h_nu .+ tgrg_h_eta .-
                                     vec(sum(tgrg_h_epsilon; dims = 2))) .== 0
                     constr_approx_4_tgrg,
                     scale_constr * (tgrg_h_z .+ tgrg_h_y) .==
                     scale_constr * vec(sum(tgrg_h_psi; dims = 1))
                     constr_approx_2_tgrg_pcone[i = 1:M, j = 1:T],
                     [scale_constr * -tgrg_h_z[i] * owa_p[i],
                      scale_constr * tgrg_h_psi[j, i] * owa_p[i] / (owa_p[i] - 1),
                      scale_constr * tgrg_h_epsilon[j, i]] ∈ MOI.PowerCone(inv(owa_p[i]))
                 end)
    return tgrg_risk
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
    tgrg_risk = tgrg_risk_formulation(rm.formulation, model, returns, T, alpha, a_sim,
                                      alpha_i, beta, b_sim, beta_i)
    set_rm_risk_upper_bound(type, model, tgrg_risk, rm.settings.ub, "tgrg_risk")
    set_risk_expression(model, tgrg_risk, rm.settings.scale, rm.settings.flag)
    return nothing
end
function tgrg_risk_vec_formulation(::OWAExact, model, returns, T, count, alpha, a_sim,
                                   alpha_i, idx, tgrg_risk, beta, b_sim, beta_i)
    scale_constr = model[:scale_constr]
    ovec = range(1; stop = 1, length = T)
    if !haskey(model, :tgrga)
        OWA_constraints(model, returns)
        @variables(model, begin
                       tgrga[1:T, 1:count]
                       tgrgb[1:T, 1:count]
                   end)
    end
    tgrga = model[:tgrga]
    tgrgb = model[:tgrgb]
    owa = model[:owa]
    add_to_expression!(tgrg_risk[idx], sum(view(tgrga, :, idx) .+ view(tgrgb, :, idx)))
    tgrg_w = owa_tgrg(T; alpha_i = alpha_i, alpha = alpha, a_sim = a_sim, beta_i = beta_i,
                      beta = beta, b_sim = b_sim)
    model[Symbol("constr_tgrg_$(idx)")] = @constraint(model,
                                                      scale_constr *
                                                      owa *
                                                      transpose(tgrg_w) .<=
                                                      scale_constr * (ovec *
                                                                      transpose(view(tgrga, :, idx)) +
                                                                      view(tgrgb, :, idx) *
                                                                      transpose(ovec)))
    return nothing
end
function tgrg_risk_vec_formulation(formulation::OWAApprox, model, returns, T, count, alpha,
                                   a_sim, alpha_i, idx, tgrg_risk, beta, b_sim, beta_i)
    scale_constr = model[:scale_constr]
    get_net_portfolio_returns(model, returns)
    net_X = model[:net_X]
    owa_p = formulation.p
    M = length(owa_p)

    tgrg_l_w = -owa_tg(T; alpha_i = alpha_i, alpha = alpha, a_sim = a_sim)
    tgrg_l_s = sum(tgrg_l_w)
    tgrg_l_l = minimum(tgrg_l_w)
    tgrg_l_h = maximum(tgrg_l_w)
    tgrg_l_d = [norm(tgrg_l_w, p) for p ∈ owa_p]
    if !haskey(model, :tgrg_l_t)
        @variables(model, begin
                       tgrg_l_t[1:count]
                       tgrg_l_nu[1:T, 1:count] .>= 0
                       tgrg_l_eta[1:T, 1:count] .>= 0
                       tgrg_l_epsilon[1:T, 1:M, 1:count]
                       tgrg_l_psi[1:T, 1:M, 1:count]
                       tgrg_l_z[1:M, 1:count]
                       tgrg_l_y[1:M, 1:count] .>= 0
                       tgrg_h_t[1:count]
                       tgrg_h_nu[1:T, 1:count] .>= 0
                       tgrg_h_eta[1:T, 1:count] .>= 0
                       tgrg_h_epsilon[1:T, 1:M, 1:count]
                       tgrg_h_psi[1:T, 1:M, 1:count]
                       tgrg_h_z[1:M, 1:count]
                       tgrg_h_y[1:M, 1:count] .>= 0
                   end)
        @expressions(model, begin
                         tgrg_l_risk[1:count], zero(AffExpr)
                         tgrg_h_risk[1:count], zero(AffExpr)
                     end)
    end
    tgrg_l_t = model[:tgrg_l_t]
    tgrg_l_nu = model[:tgrg_l_nu]
    tgrg_l_eta = model[:tgrg_l_eta]
    tgrg_l_epsilon = model[:tgrg_l_epsilon]
    tgrg_l_psi = model[:tgrg_l_psi]
    tgrg_l_z = model[:tgrg_l_z]
    tgrg_l_y = model[:tgrg_l_y]
    tgrg_l_risk = model[:tgrg_l_risk]
    tgrg_h_t = model[:tgrg_h_t]
    tgrg_h_nu = model[:tgrg_h_nu]
    tgrg_h_eta = model[:tgrg_h_eta]
    tgrg_h_epsilon = model[:tgrg_h_epsilon]
    tgrg_h_psi = model[:tgrg_h_psi]
    tgrg_h_z = model[:tgrg_h_z]
    tgrg_h_y = model[:tgrg_h_y]
    tgrg_h_risk = model[:tgrg_h_risk]
    add_to_expression!(tgrg_l_risk[idx], tgrg_l_s, tgrg_l_t[idx])
    add_to_expression!(tgrg_l_risk[idx], -tgrg_l_l, sum(view(tgrg_l_nu, :, idx)))
    add_to_expression!(tgrg_l_risk[idx], tgrg_l_h, sum(view(tgrg_l_eta, :, idx)))
    add_to_expression!(tgrg_l_risk[idx], dot(tgrg_l_d, view(tgrg_l_y, :, idx)))

    model[Symbol("constr_approx_1_tgrg_$(idx)")], model[Symbol("constr_approx_2_tgrg_$(idx)")], model[Symbol("constr_approx_1_tgrg_pcone_$(idx)")] = @constraints(model,
                                                                                                                                                                  begin
                                                                                                                                                                      scale_constr *
                                                                                                                                                                      (net_X .+
                                                                                                                                                                       tgrg_l_t[idx] .-
                                                                                                                                                                       view(tgrg_l_nu,
                                                                                                                                                                            :,
                                                                                                                                                                            idx) .+
                                                                                                                                                                       view(tgrg_l_eta,
                                                                                                                                                                            :,
                                                                                                                                                                            idx) .-
                                                                                                                                                                       vec(sum(view(tgrg_l_epsilon,
                                                                                                                                                                                    :,
                                                                                                                                                                                    :,
                                                                                                                                                                                    idx);
                                                                                                                                                                               dims = 2))) .==
                                                                                                                                                                      0
                                                                                                                                                                      scale_constr *
                                                                                                                                                                      (view(tgrg_l_z,
                                                                                                                                                                            :,
                                                                                                                                                                            idx) .+
                                                                                                                                                                       view(tgrg_l_y,
                                                                                                                                                                            :,
                                                                                                                                                                            idx)) .==
                                                                                                                                                                      scale_constr *
                                                                                                                                                                      vec(sum(view(tgrg_l_psi,
                                                                                                                                                                                   :,
                                                                                                                                                                                   :,
                                                                                                                                                                                   idx);
                                                                                                                                                                              dims = 1))
                                                                                                                                                                      [i = 1:M,
                                                                                                                                                                       j = 1:T],
                                                                                                                                                                      [scale_constr *
                                                                                                                                                                       -tgrg_l_z[i,
                                                                                                                                                                                 idx] *
                                                                                                                                                                       owa_p[i],
                                                                                                                                                                       scale_constr *
                                                                                                                                                                       tgrg_l_psi[j,
                                                                                                                                                                                  i,
                                                                                                                                                                                  idx] *
                                                                                                                                                                       owa_p[i] /
                                                                                                                                                                       (owa_p[i] -
                                                                                                                                                                        1),
                                                                                                                                                                       scale_constr *
                                                                                                                                                                       tgrg_l_epsilon[j,
                                                                                                                                                                                      i,
                                                                                                                                                                                      idx]] ∈
                                                                                                                                                                      MOI.PowerCone(inv(owa_p[i]))
                                                                                                                                                                  end)
    tgrg_h_w = -owa_tg(T; alpha_i = beta_i, alpha = beta, a_sim = b_sim)
    tgrg_h_s = sum(tgrg_h_w)
    tgrg_h_l = minimum(tgrg_h_w)
    tgrg_h_h = maximum(tgrg_h_w)
    tgrg_h_d = [norm(tgrg_h_w, p) for p ∈ owa_p]
    add_to_expression!(tgrg_h_risk[idx], tgrg_h_s, tgrg_h_t[idx])
    add_to_expression!(tgrg_h_risk[idx], -tgrg_h_l, sum(view(tgrg_h_nu, :, idx)))
    add_to_expression!(tgrg_h_risk[idx], tgrg_h_h, sum(view(tgrg_h_eta, :, idx)))
    add_to_expression!(tgrg_h_risk[idx], dot(tgrg_h_d, view(tgrg_h_y, :, idx)))
    add_to_expression!(tgrg_risk[idx], tgrg_l_risk[idx])
    add_to_expression!(tgrg_risk[idx], tgrg_h_risk[idx])
    model[Symbol("constr_approx_3_tgrg_$(idx)")], model[Symbol("constr_approx_4_tgrg_$(idx)")], model[Symbol("constr_approx_2_tgrg_pcone_$(idx)")] = @constraints(model,
                                                                                                                                                                  begin
                                                                                                                                                                      scale_constr *
                                                                                                                                                                      (-net_X .+
                                                                                                                                                                       tgrg_h_t[idx] .-
                                                                                                                                                                       view(tgrg_h_nu,
                                                                                                                                                                            :,
                                                                                                                                                                            idx) .+
                                                                                                                                                                       view(tgrg_h_eta,
                                                                                                                                                                            :,
                                                                                                                                                                            idx) .-
                                                                                                                                                                       vec(sum(tgrg_h_epsilon[:,
                                                                                                                                                                                              :,
                                                                                                                                                                                              idx];
                                                                                                                                                                               dims = 2))) .==
                                                                                                                                                                      0
                                                                                                                                                                      scale_constr *
                                                                                                                                                                      (view(tgrg_h_z,
                                                                                                                                                                            :,
                                                                                                                                                                            idx) .+
                                                                                                                                                                       view(tgrg_h_y,
                                                                                                                                                                            :,
                                                                                                                                                                            idx)) .==
                                                                                                                                                                      scale_constr *
                                                                                                                                                                      vec(sum(view(tgrg_h_psi,
                                                                                                                                                                                   :,
                                                                                                                                                                                   :,
                                                                                                                                                                                   idx);
                                                                                                                                                                              dims = 1))
                                                                                                                                                                      [i = 1:M,
                                                                                                                                                                       j = 1:T],
                                                                                                                                                                      [scale_constr *
                                                                                                                                                                       -tgrg_h_z[i,
                                                                                                                                                                                 idx] *
                                                                                                                                                                       owa_p[i],
                                                                                                                                                                       scale_constr *
                                                                                                                                                                       tgrg_h_psi[j,
                                                                                                                                                                                  i,
                                                                                                                                                                                  idx] *
                                                                                                                                                                       owa_p[i] /
                                                                                                                                                                       (owa_p[i] -
                                                                                                                                                                        1),
                                                                                                                                                                       scale_constr *
                                                                                                                                                                       tgrg_h_epsilon[j,
                                                                                                                                                                                      i,
                                                                                                                                                                                      idx]] ∈
                                                                                                                                                                      MOI.PowerCone(inv(owa_p[i]))
                                                                                                                                                                  end)
    return nothing
end
function set_rm(port::Portfolio, rms::AbstractVector{<:TGRG}, type::Union{Trad, RB, NOC};
                returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
    scale_constr = model[:scale_constr]
    T = size(returns, 1)
    count = length(rms)
    @expression(model, tgrg_risk[1:count], zero(AffExpr))
    for (idx, rm) ∈ pairs(rms)
        alpha = rm.alpha
        a_sim = rm.a_sim
        alpha_i = rm.alpha_i
        beta = rm.beta
        b_sim = rm.b_sim
        beta_i = rm.beta_i
        tgrg_risk_vec_formulation(rm.formulation, model, returns, T, count, alpha, a_sim,
                                  alpha_i, idx, tgrg_risk, beta, b_sim, beta_i)
        set_rm_risk_upper_bound(type, model, tgrg_risk[idx], rm.settings.ub,
                                "tgrg_risk_$(idx)")
        set_risk_expression(model, tgrg_risk[idx], rm.settings.scale, rm.settings.flag)
    end
    return nothing
end
function owa_risk_formulation(::OWAExact, model, returns, owa_w)
    scale_constr = model[:scale_constr]
    T = size(returns, 1)
    OWA_constraints(model, returns)
    owa = model[:owa]
    ovec = range(1; stop = 1, length = T)
    @variables(model, begin
                   owa_a[1:T]
                   owa_b[1:T]
               end)
    @expression(model, owa_risk, sum(owa_a .+ owa_b))
    if isnothing(owa_w) || isempty(owa_w)
        owa_w = owa_gmd(T)
    end
    @constraint(model, constr_owa,
                scale_constr * owa * transpose(owa_w) .<=
                scale_constr * (ovec * transpose(owa_a) + owa_b * transpose(ovec)))
    return owa_risk
end
function owa_risk_formulation(formulation::OWAApprox, model, returns, owa_w)
    scale_constr = model[:scale_constr]
    T = size(returns, 1)
    get_net_portfolio_returns(model, returns)
    net_X = model[:net_X]
    owa_p = formulation.p
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

    owa_w = (isnothing(owa_w) || isempty(owa_w)) ? -owa_gmd(T) : -owa_w
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
    return owa_risk
end
function set_rm(port::Portfolio, rm::OWA, type::Union{Trad, RB, NOC};
                returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
    owa_risk = owa_risk_formulation(rm.formulation, model, returns, rm.w)
    set_rm_risk_upper_bound(type, model, owa_risk, rm.settings.ub, "owa_risk")
    set_risk_expression(model, owa_risk, rm.settings.scale, rm.settings.flag)
    return nothing
end
function owa_risk_vec_formulation(::OWAExact, model, returns, T, count, owa_w, idx,
                                  owa_risk)
    scale_constr = model[:scale_constr]
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
    add_to_expression!(owa_risk[idx], sum(view(owa_a, :, idx) .+ view(owa_b, :, idx)))
    if isnothing(owa_w) || isempty(owa_w)
        owa_w = owa_gmd(T)
    end
    model[Symbol("constr_owa_$(idx)")] = @constraint(model,
                                                     scale_constr *
                                                     owa *
                                                     transpose(owa_w) .<=
                                                     scale_constr * (ovec *
                                                                     transpose(view(owa_a, :, idx)) +
                                                                     view(owa_b, :, idx) * transpose(ovec)))
    return nothing
end
function owa_risk_vec_formulation(formulation::OWAApprox, model, returns, T, count, owa_w,
                                  idx, owa_risk)
    scale_constr = model[:scale_constr]
    get_net_portfolio_returns(model, returns)
    net_X = model[:net_X]
    owa_p = formulation.p
    M = length(owa_p)

    owa_w = (isnothing(owa_w) || isempty(owa_w)) ? -owa_gmd(T) : -owa_w
    owa_s = sum(owa_w)
    owa_l = minimum(owa_w)
    owa_h = maximum(owa_w)
    owa_d = [norm(owa_w, p) for p ∈ owa_p]

    if !haskey(model, :owa_t)
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
        owa_risk_vec_formulation(rm.formulation, model, returns, T, count, rm.w, idx,
                                 owa_risk)
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
                     bdvariance_risk, iT2 * (dot(Dt, Dt) + iT2 * sum(Dt)^2)
                 end)
    BDVariance_constraints(rm.formulation, model, Dt, Dx, T)
    set_rm_risk_upper_bound(type, model, bdvariance_risk, rm.settings.ub, "bdvariance_risk")
    set_risk_expression(model, bdvariance_risk, rm.settings.scale, rm.settings.flag)
    return nothing
end
function set_rm(port::Portfolio, rm::NQSkew, type::Union{Trad, RB, NOC}; kwargs...)
    model = port.model
    scale_constr = model[:scale_constr]
    w = model[:w]
    @variable(model, t_nqskew)
    V = if (isnothing(rm.V) || isempty(rm.V))
        port.V
    else
        rm.V
    end
    G = real(sqrt(V))
    @constraint(model, constr_nqskew_soc,
                [scale_constr * t_nqskew; scale_constr * G * w] ∈ SecondOrderCone())
    @expression(model, nqskew_risk, t_nqskew^2)
    set_rm_risk_upper_bound(type, model, t_nqskew, sqrt(rm.settings.ub), "t_nqskew")
    set_risk_expression(model, nqskew_risk, rm.settings.scale, rm.settings.flag)
    return nothing
end
function set_rm(port::Portfolio, rms::AbstractVector{<:NQSkew}, type::Union{Trad, RB, NOC};
                kwargs...)
    model = port.model
    scale_constr = model[:scale_constr]
    w = model[:w]
    count = length(rms)
    @variable(model, t_nqskew[1:count])
    @expression(model, nqskew_risk, t_nqskew .^ 2)
    for (i, rm) ∈ pairs(rms)
        V = if (isnothing(rm.V) || isempty(rm.V))
            port.V
        else
            rm.V
        end
        G = real(sqrt(V))
        model[Symbol("constr_nqskew_soc_$(i)")] = @constraint(model,
                                                              [scale_constr * t_nqskew[i];
                                                               scale_constr * G * w] ∈
                                                              SecondOrderCone())
        set_rm_risk_upper_bound(type, model, t_nqskew[i], sqrt(rm.settings.ub),
                                "t_nqskew_$(i)")
        set_risk_expression(model, nqskew_risk[i], rm.settings.scale, rm.settings.flag)
    end
    return nothing
end
function set_rm(port::Portfolio, rm::NQSSkew, type::Union{Trad, RB, NOC}; kwargs...)
    model = port.model
    scale_constr = model[:scale_constr]
    w = model[:w]
    @variable(model, t_nqsskew)
    SV = if (isnothing(rm.V) || isempty(rm.V))
        port.SV
    else
        rm.V
    end
    G = real(sqrt(SV))
    @constraint(model, constr_nqsskew_soc,
                [scale_constr * t_nqsskew; scale_constr * G * w] ∈ SecondOrderCone())
    @expression(model, nqsskew_risk, t_nqsskew^2)
    set_rm_risk_upper_bound(type, model, t_nqsskew, sqrt(rm.settings.ub), "t_nqsskew")
    set_risk_expression(model, nqsskew_risk, rm.settings.scale, rm.settings.flag)
    return nothing
end
function set_rm(port::Portfolio, rms::AbstractVector{<:NQSSkew}, type::Union{Trad, RB, NOC};
                kwargs...)
    model = port.model
    scale_constr = model[:scale_constr]
    w = model[:w]
    count = length(rms)
    @variable(model, t_nqsskew[1:count])
    @expression(model, nqsskew_risk, t_nqsskew .^ 2)
    for (i, rm) ∈ pairs(rms)
        V = if (isnothing(rm.V) || isempty(rm.V))
            port.SV
        else
            rm.V
        end
        G = real(sqrt(V))
        model[Symbol("constr_nqsskew_soc_$(i)")] = @constraint(model,
                                                               [scale_constr * t_nqsskew[i];
                                                                scale_constr * G * w] ∈
                                                               SecondOrderCone())
        set_rm_risk_upper_bound(type, model, t_nqsskew[i], sqrt(rm.settings.ub),
                                "t_nqsskew_$(i)")
        set_risk_expression(model, nqsskew_risk[i], rm.settings.scale, rm.settings.flag)
    end
    return nothing
end
######
function set_rm(port::Portfolio, rm::NSkew, type::Union{Trad, RB, NOC}; kwargs...)
    model = port.model
    scale_constr = model[:scale_constr]
    w = model[:w]
    @variable(model, nskew_risk)
    V = if (isnothing(rm.V) || isempty(rm.V))
        port.V
    else
        rm.V
    end
    G = real(sqrt(V))
    @constraint(model, constr_nskew_soc,
                [scale_constr * nskew_risk; scale_constr * G * w] ∈ SecondOrderCone())
    set_rm_risk_upper_bound(type, model, nskew_risk, rm.settings.ub, "nskew_risk")
    set_risk_expression(model, nskew_risk, rm.settings.scale, rm.settings.flag)
    return nothing
end
function set_rm(port::Portfolio, rms::AbstractVector{<:NSkew}, type::Union{Trad, RB, NOC};
                kwargs...)
    model = port.model
    scale_constr = model[:scale_constr]
    w = model[:w]
    count = length(rms)
    @variable(model, nskew_risk[1:count])
    for (i, rm) ∈ pairs(rms)
        V = if (isnothing(rm.V) || isempty(rm.V))
            port.V
        else
            rm.V
        end
        G = real(sqrt(V))
        model[Symbol("constr_nskew_soc_$(i)")] = @constraint(model,
                                                             [scale_constr * nskew_risk[i];
                                                              scale_constr * G * w] ∈
                                                             SecondOrderCone())
        set_rm_risk_upper_bound(type, model, nskew_risk[i], rm.settings.ub,
                                "nskew_risk_$(i)")
        set_risk_expression(model, nskew_risk[i], rm.settings.scale, rm.settings.flag)
    end
    return nothing
end
function set_rm(port::Portfolio, rm::NSSkew, type::Union{Trad, RB, NOC}; kwargs...)
    model = port.model
    scale_constr = model[:scale_constr]
    w = model[:w]
    @variable(model, nsskew_risk)
    SV = if (isnothing(rm.V) || isempty(rm.V))
        port.SV
    else
        rm.V
    end
    G = real(sqrt(SV))
    @constraint(model, constr_nsskew_soc,
                [scale_constr * nsskew_risk; scale_constr * G * w] ∈ SecondOrderCone())
    set_rm_risk_upper_bound(type, model, nsskew_risk, rm.settings.ub, "nsskew_risk")
    set_risk_expression(model, nsskew_risk, rm.settings.scale, rm.settings.flag)
    return nothing
end
function set_rm(port::Portfolio, rms::AbstractVector{<:NSSkew}, type::Union{Trad, RB, NOC};
                kwargs...)
    model = port.model
    scale_constr = model[:scale_constr]
    w = model[:w]
    count = length(rms)
    @variable(model, nsskew_risk[1:count])
    for (i, rm) ∈ pairs(rms)
        V = if (isnothing(rm.V) || isempty(rm.V))
            port.SV
        else
            rm.V
        end
        G = real(sqrt(V))
        model[Symbol("constr_nsskew_soc_$(i)")] = @constraint(model,
                                                              [scale_constr *
                                                               nsskew_risk[i]
                                                               scale_constr * G * w] ∈
                                                              SecondOrderCone())
        set_rm_risk_upper_bound(type, model, nsskew_risk[i], rm.settings.ub,
                                "nsskew_risk_$(i)")
        set_risk_expression(model, nsskew_risk[i], rm.settings.scale, rm.settings.flag)
    end
    return nothing
end
######
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
    @expression(model, tracking_risk, t_tracking_risk / sqrt(T - one(T)))
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
    iTm1 = inv(sqrt(T - one(T)))
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
        turnover_rm[:, i] .= @expression(model, w .- benchmark * k)
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
function risk_constraints(port, type::Union{Trad, FRC, RB, NOC},
                          rms::Union{RiskMeasure, AbstractVector}, mu, sigma, returns,
                          kelly_approx_idx = nothing; kwargs...)
    for rm ∈ rms
        set_rm(port, rm, type; mu = mu, sigma = sigma, returns = returns,
               kelly_approx_idx = kelly_approx_idx, kwargs...)
    end
    return nothing
end
