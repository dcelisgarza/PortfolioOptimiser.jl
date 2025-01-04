# Risk expression
function _set_risk_expression(model, rm_risk, scale, flag::Bool)
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
function _get_ntwk_clust_method(port)
    model = port.model
    return if haskey(model, :constr_ntwk_sdp) || haskey(model, :constr_clst_sdp)
        SDP()
    else
        NoAdj()
    end
end
function _set_rm_risk_upper_bound(args...)
    return nothing
end
function _set_rm_risk_upper_bound(::Union{Trad, NOC}, model, rm_risk, ub, key)
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
function _variance_risk(::SDP, ::Any, model, sigma)
    W = model[:W]
    @expression(model, variance_risk, tr(sigma * W))
    return nothing
end
function _variance_risk(::SDP, model::JuMP.Model, sigma, count::Integer)
    @expression(model, variance_risk[1:count], zero(AffExpr))
    return nothing
end
function _variance_risk(::SDP, ::Any, model, sigma, idx::Integer)
    W = model[:W]
    variance_risk = model[:variance_risk]
    add_to_expression!(variance_risk[idx], tr(sigma * W))
    return nothing
end
function _variance_risk(::Union{NoAdj, IP}, ::SOC, model::JuMP.Model, sigma::AbstractMatrix)
    scale_constr = model[:scale_constr]
    w = model[:w]
    G = sqrt(sigma)
    @variable(model, dev)
    @expression(model, variance_risk, dev^2)
    @constraint(model, [scale_constr * dev; scale_constr * G * w] ∈ SecondOrderCone())
    return nothing
end
function _variance_risk(::Union{NoAdj, IP}, model::JuMP.Model, sigma::AbstractMatrix,
                        count::Integer)
    @variable(model, dev[1:count])
    @expression(model, variance_risk[1:count], zero(QuadExpr))
    return nothing
end
function _variance_risk(::Union{NoAdj, IP}, ::SOC, model::JuMP.Model, sigma::AbstractMatrix,
                        idx::Integer)
    scale_constr = model[:scale_constr]
    w = model[:w]
    dev = model[:dev][idx]
    variance_risk = model[:variance_risk][idx]
    G = sqrt(sigma)
    add_to_expression!(variance_risk, dev, dev)
    @constraint(model, [scale_constr * dev; scale_constr * G * w] ∈ SecondOrderCone())
    return nothing
end
function _variance_risk(::Union{NoAdj, IP}, ::Quad, model, sigma)
    scale_constr = model[:scale_constr]
    w = model[:w]
    G = sqrt(sigma)
    @variable(model, dev)
    @expression(model, variance_risk, dot(w, sigma, w))
    @constraint(model, [scale_constr * dev; scale_constr * G * w] ∈ SecondOrderCone())
    return nothing
end
function _variance_risk(::Union{NoAdj, IP}, ::Quad, model::JuMP.Model,
                        sigma::AbstractMatrix, idx::Integer)
    scale_constr = model[:scale_constr]
    w = model[:w]
    dev = model[:dev][idx]
    variance_risk = model[:variance_risk][idx]
    G = sqrt(sigma)
    add_to_expression!(variance_risk, dot(w, sigma, w))
    @constraint(model, [scale_constr * dev; scale_constr * G * w] ∈ SecondOrderCone())
    return nothing
end
function _variance_risk_bounds_expr(::SDP, model)
    return model[:variance_risk], "variance_risk"
end
function _variance_risk_bounds_expr(::Union{NoAdj, IP}, model)
    return model[:dev], "dev"
end
function _variance_risk_bounds_val(::SDP, ub)
    return ub
end
function _variance_risk_bounds_val(::Union{NoAdj, IP}, ub)
    return sqrt(ub)
end
function set_rm(port, rm::Variance, type::Union{Trad, RP, NOC};
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
    adjacency_constraint = _get_ntwk_clust_method(port)
    _variance_risk(adjacency_constraint, rm.formulation, model, sigma)
    variance_risk = model[:variance_risk]
    var_bound_expr, var_bound_key = _variance_risk_bounds_expr(adjacency_constraint, model)
    ub = _variance_risk_bounds_val(adjacency_constraint, rm.settings.ub)
    _set_rm_risk_upper_bound(type, model, var_bound_expr, ub, var_bound_key)
    _set_risk_expression(model, model[:variance_risk], rm.settings.scale, rm.settings.flag)
    return nothing
end
function set_rm(port, rms::AbstractVector{<:Variance}, type::Union{Trad, RP, NOC};
                sigma::AbstractMatrix{<:Real},
                kelly_approx_idx::Union{AbstractVector{<:Integer}, Nothing}, kwargs...)
    model = port.model
    adjacency_constraint = _get_ntwk_clust_method(port)
    count = length(rms)
    _variance_risk(adjacency_constraint, model, sigma, count)
    variance_risk = model[:variance_risk]
    var_bound_expr, var_bound_key = _variance_risk_bounds_expr(adjacency_constraint, model)
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
        _variance_risk(adjacency_constraint, rm.formulation, model, sigma, i)
        ub = _variance_risk_bounds_val(adjacency_constraint, rm.settings.ub)
        _set_rm_risk_upper_bound(type, model, var_bound_expr[i], ub,
                                 "$(var_bound_key)_$(i)")
        _set_risk_expression(model, variance_risk[i], rm.settings.scale, rm.settings.flag)
    end
    return nothing
end
function _choose_wc_stats_port_rm(port, rm)
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
function _wc_variance_risk_variables(::Box, model)
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
function _wc_variance_risk_variables(::Ellipse, model)
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
function _wc_variance_risk(::Box, model, sigma, cov_l, cov_u, cov_sigma, k_sigma)
    Au = model[:Au]
    Al = model[:Al]
    @expression(model, wc_variance_risk, tr(Au * cov_u) - tr(Al * cov_l))
    return nothing
end
function _wc_variance_risk(::Ellipse, model, sigma, cov_l, cov_u, cov_sigma, k_sigma)
    scale_constr = model[:scale_constr]
    WpE = model[:WpE]
    G_sigma = sqrt(cov_sigma)
    @variable(model, t_ge)
    @expressions(model, begin
                     x_ge, G_sigma * vec(WpE)
                     wc_variance_risk, tr(sigma * WpE) + k_sigma * t_ge
                 end)
    @constraint(model, [scale_constr * t_ge; scale_constr * x_ge] ∈ SecondOrderCone())
    return nothing
end
function _wc_variance_risk(::Box, model, sigma, cov_l, cov_u, cov_sigma, k_sigma,
                           wc_variance_risk)
    Au = model[:Au]
    Al = model[:Al]
    add_to_expression!(wc_variance_risk, tr(Au * cov_u) - tr(Al * cov_l))
    return nothing
end
function _wc_variance_risk(::Ellipse, model, sigma, cov_l, cov_u, cov_sigma, k_sigma,
                           wc_variance_risk)
    scale_constr = model[:scale_constr]
    WpE = model[:WpE]
    G_sigma = sqrt(cov_sigma)
    t_ge = @variable(model)
    x_ge = @expression(model, G_sigma * vec(WpE))
    add_to_expression!(wc_variance_risk, tr(sigma * WpE))
    add_to_expression!(wc_variance_risk, k_sigma, t_ge)
    @constraint(model, [scale_constr * t_ge; scale_constr * x_ge] ∈ SecondOrderCone())
    return nothing
end
function set_rm(port, rm::WCVariance, type::Union{Trad, RP, NOC};
                sigma::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
    _SDP_constraints(model, type)
    sigma, cov_l, cov_u, cov_sigma, k_sigma = _choose_wc_stats_port_rm(port, rm)
    _wc_variance_risk_variables(rm.wc_set, model)
    _wc_variance_risk(rm.wc_set, model, sigma, cov_l, cov_u, cov_sigma, k_sigma)
    wc_variance_risk = model[:wc_variance_risk]
    _set_rm_risk_upper_bound(type, model, wc_variance_risk, rm.settings.ub,
                             "wc_variance_risk")
    _set_risk_expression(model, wc_variance_risk, rm.settings.scale, rm.settings.flag)
    return nothing
end
function set_rm(port, rms::AbstractVector{<:WCVariance}, type::Union{Trad, RP, NOC};
                sigma::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
    _SDP_constraints(model, type)
    count = length(rms)
    @expression(model, wc_variance_risk[1:count], zero(AffExpr))
    for (i, rm) ∈ pairs(rms)
        sigma, cov_l, cov_u, cov_sigma, k_sigma = _choose_wc_stats_port_rm(port, rm)
        _wc_variance_risk_variables(rm.wc_set, model)
        _wc_variance_risk(rm.wc_set, model, sigma, cov_l, cov_u, cov_sigma, k_sigma,
                          wc_variance_risk[i])
        _wc_variance_risk(rm.wc_set, model, sigma, cov_l, cov_u, cov_sigma, k_sigma)
        _set_rm_risk_upper_bound(type, model, wc_variance_risk[i], rm.settings.ub,
                                 "wc_variance_risk_$(i)")
        _set_risk_expression(model, wc_variance_risk[i], rm.settings.scale,
                             rm.settings.flag)
    end
    return nothing
end
function set_rm(port, rm::SD, type::Union{Trad, RP, NOC}; sigma::AbstractMatrix{<:Real},
                kwargs...)
    model = port.model
    scale_constr = model[:scale_constr]
    w = model[:w]
    use_portfolio_sigma = isnothing(rm.sigma) || isempty(rm.sigma)
    if !use_portfolio_sigma
        sigma = rm.sigma
    end
    @variable(model, sd_risk)
    G = sqrt(sigma)
    @constraint(model, [scale_constr * sd_risk; scale_constr * G * w] ∈ SecondOrderCone())
    _set_rm_risk_upper_bound(type, model, sd_risk, rm.settings.ub, "sd_risk")
    _set_risk_expression(model, sd_risk, rm.settings.scale, rm.settings.flag)
    return nothing
end
function set_rm(port, rms::AbstractVector{<:SD}, type::Union{Trad, RP, NOC};
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
        @constraint(model,
                    [scale_constr * sd_risk[i]; scale_constr * G * w] ∈ SecondOrderCone())
        _set_rm_risk_upper_bound(type, model, sd_risk[i], rm.settings.ub, "sd_risk_$(i)")
        _set_risk_expression(model, sd_risk[i], rm.settings.scale, rm.settings.flag)
    end
    return nothing
end
function set_rm(port::Portfolio, rm::MAD, type::Union{Trad, RP, NOC};
                mu::AbstractVector{<:Real}, returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
    scale_constr = model[:scale_constr]
    w = model[:w]
    T = size(returns, 1)
    if !(isnothing(rm.mu) || isempty(rm.mu))
        mu = rm.mu
    end
    mar = returns .- transpose(mu)
    @variable(model, mad[1:T] .>= 0)
    @expression(model, mad_risk, 2 * sum(mad) / T)
    @constraints(model, begin
                     scale_constr * mar * w .>= scale_constr * -mad
                 end)
    _set_rm_risk_upper_bound(type, model, mad_risk, rm.settings.ub, "mad_risk")
    _set_risk_expression(model, mad_risk, rm.settings.scale, rm.settings.flag)
    return nothing
end
function set_rm(port::Portfolio, rms::AbstractVector{<:MAD}, type::Union{Trad, RP, NOC};
                mu::AbstractVector{<:Real}, returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
    scale_constr = model[:scale_constr]
    w = model[:w]
    T = size(returns, 1)
    iT2 = 2 * inv(T)
    count = length(rms)
    @variable(model, mad[1:T, 1:count] .>= 0)
    @expression(model, mad_risk[1:count], zero(AffExpr))
    for (i, rm) ∈ pairs(rms)
        if !(isnothing(rm.mu) || isempty(rm.mu))
            mu = rm.mu
        end
        mar = returns .- transpose(mu)
        @constraints(model, begin
                         scale_constr * mar * w .>= scale_constr * -view(mad, :, i)
                     end)
        add_to_expression!(mad_risk[i], iT2, sum(view(mad, :, i)))
        _set_rm_risk_upper_bound(type, model, mad_risk[i], rm.settings.ub, "mad_risk_$(i)")
        _set_risk_expression(model, mad_risk[i], rm.settings.scale, rm.settings.flag)
    end
    return nothing
end
function _semi_variance_risk(::Quad, model::JuMP.Model, svariance, iTm1)
    @expression(model, svariance_risk, dot(svariance, svariance) * iTm1)
    return nothing
end
function _semi_variance_risk(::Quad, ::Any, svariance, svariance_risk, iTm1)
    add_to_expression!(svariance_risk, iTm1, dot(svariance, svariance))
    return nothing
end
function _semi_variance_risk(::SOC, model::JuMP.Model, svariance, iTm1)
    scale_constr = model[:scale_constr]
    @variable(model, tsvariance)
    @constraint(model,
                [scale_constr * tsvariance; 0.5; scale_constr * svariance] in
                RotatedSecondOrderCone())
    @expression(model, svariance_risk, tsvariance * iTm1)
    return nothing
end
function _semi_variance_risk(::SOC, model, svariance, svariance_risk, iTm1)
    scale_constr = model[:scale_constr]
    tsvariance = @variable(model)
    @constraint(model,
                [scale_constr * tsvariance; 0.5; scale_constr * svariance] in
                RotatedSecondOrderCone())
    add_to_expression!(svariance_risk, iTm1, tsvariance)
    return nothing
end
function set_rm(port::Portfolio, rm::SVariance, type::Union{Trad, RP, NOC};
                mu::AbstractVector{<:Real}, returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
    scale_constr = model[:scale_constr]
    w = model[:w]
    k = model[:k]
    T = size(returns, 1)
    if !(isnothing(rm.mu) || isempty(rm.mu))
        mu = rm.mu
    end
    mar = returns .- transpose(mu)
    target = rm.target
    @variable(model, svariance[1:T])
    _semi_variance_risk(rm.formulation, model, svariance, inv(T - 1))
    @constraints(model, begin
                     scale_constr * svariance .>= scale_constr * target * k
                     scale_constr * mar * w .>= scale_constr * -svariance
                 end)
    svariance_risk = model[:svariance_risk]
    _set_rm_risk_upper_bound(type, model, svariance_risk, rm.settings.ub, "svariance_risk")
    _set_risk_expression(model, svariance_risk, rm.settings.scale, rm.settings.flag)

    return nothing
end
function set_rm(port::Portfolio, rms::AbstractVector{<:SVariance},
                type::Union{Trad, RP, NOC}; mu::AbstractVector{<:Real},
                returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
    scale_constr = model[:scale_constr]
    w = model[:w]
    k = model[:k]
    T = size(returns, 1)
    iTm1 = inv(T - 1)
    count = length(rms)
    @variable(model, svariance[1:T, 1:count])
    @expression(model, svariance_risk[1:count], zero(QuadExpr))
    for (i, rm) ∈ pairs(rms)
        if !(isnothing(rm.mu) || isempty(rm.mu))
            mu = rm.mu
        end
        mar = returns .- transpose(mu)
        target = rm.target
        @constraints(model,
                     begin
                         scale_constr * view(svariance, :, i) .>= scale_constr * target * k
                         scale_constr * mar * w .>= scale_constr * -view(svariance, :, i)
                     end)
        _semi_variance_risk(rm.formulation, model, view(svariance, :, i), svariance_risk[i],
                            iTm1)
        _set_rm_risk_upper_bound(type, model, svariance_risk[i], rm.settings.ub,
                                 "svariance_risk_$(i)")
        _set_risk_expression(model, svariance_risk[i], rm.settings.scale, rm.settings.flag)
    end
    return nothing
end
function set_rm(port::Portfolio, rm::SSD, type::Union{Trad, RP, NOC};
                mu::AbstractVector{<:Real}, returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
    scale_constr = model[:scale_constr]
    w = model[:w]
    k = model[:k]
    T = size(returns, 1)
    if !(isnothing(rm.mu) || isempty(rm.mu))
        mu = rm.mu
    end
    mar = returns .- transpose(mu)
    target = rm.target
    @variables(model, begin
                   ssd[1:T]
                   sdev
               end)
    @expression(model, sdev_risk, sdev / sqrt(T - 1))
    @constraints(model, begin
                     scale_constr * ssd .>= scale_constr * target * k
                     scale_constr * mar * w .>= scale_constr * -ssd
                     [scale_constr * sdev; scale_constr * ssd] ∈ SecondOrderCone()
                 end)
    _set_rm_risk_upper_bound(type, model, sdev_risk, rm.settings.ub, "sdev_risk")
    _set_risk_expression(model, sdev_risk, rm.settings.scale, rm.settings.flag)

    return nothing
end
function set_rm(port::Portfolio, rms::AbstractVector{<:SSD}, type::Union{Trad, RP, NOC};
                mu::AbstractVector{<:Real}, returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
    scale_constr = model[:scale_constr]
    w = model[:w]
    k = model[:k]
    T = size(returns, 1)
    iTm1 = inv(sqrt(T - 1))
    count = length(rms)
    @variable(model, ssd[1:T, 1:count])
    @variable(model, sdev[1:count])
    @expression(model, sdev_risk[1:count], zero(AffExpr))
    for (i, rm) ∈ pairs(rms)
        if !(isnothing(rm.mu) || isempty(rm.mu))
            mu = rm.mu
        end
        mar = returns .- transpose(mu)
        target = rm.target
        @constraints(model,
                     begin
                         scale_constr * view(ssd, :, i) .>= scale_constr * target * k
                         scale_constr * mar * w .>= scale_constr * -view(ssd, :, i)
                         [scale_constr * sdev[i]; scale_constr * view(ssd, :, i)] ∈
                         SecondOrderCone()
                     end)
        add_to_expression!(sdev_risk[i], iTm1, sdev[i])
        _set_rm_risk_upper_bound(type, model, sdev_risk[i], rm.settings.ub,
                                 "sdev_risk_$(i)")
        _set_risk_expression(model, sdev_risk[i], rm.settings.scale, rm.settings.flag)
    end
    return nothing
end
function set_rm(port::Portfolio, rm::FLPM, type::Union{Trad, RP, NOC};
                returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
    scale_constr = model[:scale_constr]
    w = model[:w]
    k = model[:k]
    T = size(returns, 1)
    ret_target = rm.ret_target
    if isinf(ret_target)
        ret_target = port.mu
    end
    target = rm.target
    mar = returns .- transpose(ret_target)
    @variable(model, flpm[1:T] .>= 0)
    @expression(model, flpm_risk, sum(flpm) / T)
    @constraints(model, begin
                     scale_constr * flpm .>= scale_constr * (target * k .- mar * w)
                 end)
    _set_rm_risk_upper_bound(type, model, flpm_risk, rm.settings.ub, "flpm_risk")
    _set_risk_expression(model, flpm_risk, rm.settings.scale, rm.settings.flag)
    return nothing
end
function set_rm(port::Portfolio, rms::AbstractVector{<:FLPM}, type::Union{Trad, RP, NOC};
                returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
    scale_constr = model[:scale_constr]
    w = model[:w]
    k = model[:k]
    T = size(returns, 1)
    iT = inv(T)
    count = length(rms)
    @variable(model, flpm[1:T, 1:count] .>= 0)
    @expression(model, flpm_risk[1:count], zero(AffExpr))
    for (i, rm) ∈ pairs(rms)
        ret_target = rm.ret_target
        if isinf(ret_target)
            ret_target = port.mu
        end
        target = rm.target
        mar = returns .- transpose(ret_target)
        add_to_expression!(flpm_risk[i], iT, sum(view(flpm, :, i)))
        @constraints(model,
                     begin
                         scale_constr * view(flpm, :, i) .>=
                         scale_constr * (target * k .- mar * w)
                     end)
        _set_rm_risk_upper_bound(type, model, flpm_risk[i], rm.settings.ub,
                                 "flpm_risk_$(i)")
        _set_risk_expression(model, flpm_risk[i], rm.settings.scale, rm.settings.flag)
    end
    return nothing
end
function set_rm(port::Portfolio, rm::SLPM, type::Union{Trad, RP, NOC};
                returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
    scale_constr = model[:scale_constr]
    w = model[:w]
    k = model[:k]
    T = size(returns, 1)
    ret_target = rm.ret_target
    if isinf(ret_target)
        ret_target = port.mu
    end
    target = rm.target
    mar = returns .- transpose(ret_target)
    @variables(model, begin
                   slpm[1:T] .>= 0
                   tslpm
               end)
    @expression(model, slpm_risk, tslpm / sqrt(T - 1))
    @constraints(model, begin
                     scale_constr * slpm .>= scale_constr * (target * k .- mar * w)
                     [scale_constr * tslpm; scale_constr * slpm] ∈ SecondOrderCone()
                 end)
    _set_rm_risk_upper_bound(type, model, slpm_risk, rm.settings.ub, "slpm_risk")
    _set_risk_expression(model, slpm_risk, rm.settings.scale, rm.settings.flag)
    return nothing
end
function set_rm(port::Portfolio, rms::AbstractVector{<:SLPM}, type::Union{Trad, RP, NOC};
                returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
    scale_constr = model[:scale_constr]
    w = model[:w]
    k = model[:k]
    T = size(returns, 1)
    iTm1 = sqrt(inv(T - 1))
    count = length(rms)
    @variables(model, begin
                   slpm[1:T, 1:count] .>= 0
                   tslpm[1:count]
               end)
    @expression(model, slpm_risk[1:count], zero(AffExpr))
    for (i, rm) ∈ pairs(rms)
        ret_target = rm.ret_target
        if isinf(ret_target)
            ret_target = port.mu
        end
        target = rm.target
        mar = returns .- transpose(ret_target)
        add_to_expression!(slpm_risk[i], iTm1, tslpm[i])
        @constraints(model,
                     begin
                         scale_constr * view(slpm, :, i) .>=
                         scale_constr * (target * k .- mar * w)
                         [scale_constr * tslpm[i]; scale_constr * view(slpm, :, i)] ∈
                         SecondOrderCone()
                     end)
        _set_rm_risk_upper_bound(type, model, slpm_risk[i], rm.settings.ub,
                                 "slpm_risk_$(i)")
        _set_risk_expression(model, slpm_risk[i], rm.settings.scale, rm.settings.flag)
    end
    return nothing
end
function _wr_risk(model, returns)
    if haskey(model, :wr)
        return nothing
    end
    scale_constr = model[:scale_constr]
    get_net_portfolio_returns(model, returns)
    net_X = model[:net_X]
    @variable(model, wr)
    @expression(model, wr_risk, wr)
    @constraint(model, scale_constr * -net_X .<= scale_constr * wr)

    return nothing
end
function set_rm(port::Portfolio, rm::WR, type::Union{Trad, RP, NOC};
                returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
    _wr_risk(model, returns)
    wr_risk = model[:wr_risk]
    _set_rm_risk_upper_bound(type, model, wr_risk, rm.settings.ub, "wr_risk")
    _set_risk_expression(model, wr_risk, rm.settings.scale, rm.settings.flag)
    return nothing
end
function set_rm(port::Portfolio, rm::RG, type::Union{Trad, RP, NOC};
                returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
    scale_constr = model[:scale_constr]
    _wr_risk(model, returns)
    wr_risk = model[:wr_risk]
    net_X = model[:net_X]
    @variable(model, br)
    @expression(model, rg_risk, wr_risk - br)
    @constraint(model, scale_constr * -net_X .>= scale_constr * br)
    _set_rm_risk_upper_bound(type, model, rg_risk, rm.settings.ub, "rg_risk")
    _set_risk_expression(model, rg_risk, rm.settings.scale, rm.settings.flag)
    return nothing
end
function set_rm(port::Portfolio, rm::CVaR, type::Union{Trad, RP, NOC};
                returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
    scale_constr = model[:scale_constr]
    get_net_portfolio_returns(model, returns)
    net_X = model[:net_X]
    T = size(returns, 1)
    iat = inv(rm.alpha * T)
    @variables(model, begin
                   var
                   z_var[1:T] .>= 0
               end)
    @expression(model, cvar_risk, var + sum(z_var) * iat)
    @constraints(model, begin
                     scale_constr * z_var .>= scale_constr * (-net_X .- var)
                 end)
    _set_rm_risk_upper_bound(type, model, cvar_risk, rm.settings.ub, "cvar_risk")
    _set_risk_expression(model, cvar_risk, rm.settings.scale, rm.settings.flag)
    return nothing
end
function set_rm(port::Portfolio, rms::AbstractVector{<:CVaR}, type::Union{Trad, RP, NOC};
                returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
    scale_constr = model[:scale_constr]
    get_net_portfolio_returns(model, returns)
    net_X = model[:net_X]
    T = size(returns, 1)
    count = length(rms)
    @variables(model, begin
                   var[1:count]
                   z_var[1:T, 1:count] .>= 0
               end)
    @expression(model, cvar_risk[1:count], zero(AffExpr))
    for (i, rm) ∈ pairs(rms)
        iat = inv(rm.alpha * T)
        add_to_expression!(cvar_risk[i], var[i])
        add_to_expression!(cvar_risk[i], iat, sum(view(z_var, :, i)))
        @constraints(model,
                     begin
                         scale_constr * view(z_var, :, i) .>=
                         scale_constr * (-net_X .- var[i])
                     end)
        _set_rm_risk_upper_bound(type, model, cvar_risk[i], rm.settings.ub,
                                 "cvar_risk_$(i)")
        _set_risk_expression(model, cvar_risk[i], rm.settings.scale, rm.settings.flag)
    end
    return nothing
end
######
function set_rm(port::Portfolio, rm::DRCVaR, type::Union{Trad, RP, NOC};
                returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
    scale_constr = model[:scale_constr]
    get_portfolio_returns(model, returns)
    get_one_plus_returns(model, returns)
    w = model[:w]
    X = model[:X]
    RP1 = model[:RP1]
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
                     scale_constr * (b1 * tau .+ a1 * X .+ vec(sum(u .* RP1; dims = 2))) .<=
                     scale_constr * s
                     scale_constr * (b2 * tau .+ a2 * X .+ vec(sum(v .* RP1; dims = 2))) .<=
                     scale_constr * s
                     [i = 1:T],
                     [scale_constr * tu_drcvar[i];
                      scale_constr * (-view(u, i, :) .- a1 * w)] in
                     MOI.NormInfinityCone(1 + N)
                     [i = 1:T],
                     [scale_constr * tv_drcvar[i];
                      scale_constr * (-view(v, i, :) .- a2 * w)] in
                     MOI.NormInfinityCone(1 + N)
                     scale_constr * tu_drcvar .<= scale_constr * lb
                     scale_constr * tv_drcvar .<= scale_constr * lb
                 end)

    @expression(model, drcvar_risk, radius * lb + sum(s) * inv(T))
    _set_rm_risk_upper_bound(type, model, drcvar_risk, rm.settings.ub, "drcvar_risk")
    _set_risk_expression(model, drcvar_risk, rm.settings.scale, rm.settings.flag)
    return nothing
end
function set_rm(port::Portfolio, rms::AbstractVector{<:DRCVaR}, type::Union{Trad, RP, NOC};
                returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
    scale_constr = model[:scale_constr]
    get_portfolio_returns(model, returns)
    get_one_plus_returns(model, returns)
    w = model[:w]
    X = model[:X]
    RP1 = model[:RP1]
    T, N = size(returns)

    iT = inv(T)
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

        @constraints(model,
                     begin
                         scale_constr * (b1 * tau[j] .+ a1 * X .+
                                         vec(sum(view(u, :, :, j) .* RP1; dims = 2))) .<=
                         scale_constr * view(s, :, j)
                         scale_constr * (b2 * tau[j] .+ a2 * X .+
                                         vec(sum(view(v, :, :, j) .* RP1; dims = 2))) .<=
                         scale_constr * view(s, :, j)
                         [i = 1:T],
                         [scale_constr * tu_drcvar[i, j];
                          scale_constr * (-view(u, i, :, j) .- a1 * w)] in
                         MOI.NormInfinityCone(1 + N)
                         [i = 1:T],
                         [scale_constr * tv_drcvar[i, j];
                          scale_constr * (-view(v, i, :, j) .- a2 * w)] in
                         MOI.NormInfinityCone(1 + N)
                         scale_constr * view(tu_drcvar, :, j) .<= scale_constr * lb[j]
                         scale_constr * view(tv_drcvar, :, j) .<= scale_constr * lb[j]
                     end)
        add_to_expression!(drcvar_risk[j], radius, lb[j])
        add_to_expression!(drcvar_risk[j], iT, sum(view(s, :, j)))
        _set_rm_risk_upper_bound(type, model, drcvar_risk[j], rm.settings.ub,
                                 "drcvar_risk_$(j)")
        _set_risk_expression(model, drcvar_risk[j], rm.settings.scale, rm.settings.flag)
    end
    return nothing
end
function set_rm(port::Portfolio, rm::CVaRRG, type::Union{Trad, RP, NOC};
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
                   z_var_l[1:T] .>= 0
                   var_h
                   z_var_h[1:T] .<= 0
               end)
    @expressions(model, begin
                     cvar_risk_l, var_l + sum(z_var_l) * iat
                     cvar_risk_h, var_h + sum(z_var_h) * ibt
                     rcvar_risk, cvar_risk_l - cvar_risk_h
                 end)
    @constraints(model, begin
                     scale_constr * z_var_l .>= scale_constr * (-net_X .- var_l)
                     scale_constr * z_var_h .<= scale_constr * (-net_X .- var_h)
                 end)

    _set_rm_risk_upper_bound(type, model, rcvar_risk, rm.settings.ub, "rcvar_risk")
    _set_risk_expression(model, rcvar_risk, rm.settings.scale, rm.settings.flag)
    return nothing
end
function set_rm(port::Portfolio, rms::AbstractVector{<:CVaRRG}, type::Union{Trad, RP, NOC};
                returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
    scale_constr = model[:scale_constr]
    get_net_portfolio_returns(model, returns)
    net_X = model[:net_X]
    T = size(returns, 1)
    count = length(rms)
    @variables(model, begin
                   var_l[1:count]
                   z_var_l[1:T, 1:count] .>= 0
                   var_h[1:count]
                   z_var_h[1:T, 1:count] .<= 0
               end)
    @expressions(model, begin
                     cvar_risk_l[1:count], zero(AffExpr)
                     cvar_risk_h[1:count], zero(AffExpr)
                     rcvar_risk[1:count], zero(AffExpr)
                 end)
    for (i, rm) ∈ pairs(rms)
        iat = inv(rm.alpha * T)
        ibt = inv(rm.beta * T)
        @constraints(model,
                     begin
                         scale_constr * view(z_var_l, :, i) .>=
                         scale_constr * (-net_X .- var_l[i])
                         scale_constr * view(z_var_h, :, i) .<=
                         scale_constr * (-net_X .- var_h[i])
                     end)
        add_to_expression!(cvar_risk_l[i], var_l[i])
        add_to_expression!(cvar_risk_l[i], iat, sum(view(z_var_l, :, i)))
        add_to_expression!(cvar_risk_h[i], var_h[i])
        add_to_expression!(cvar_risk_h[i], ibt, sum(view(z_var_h, :, i)))
        add_to_expression!(rcvar_risk[i], cvar_risk_l[i])
        add_to_expression!(rcvar_risk[i], -1, cvar_risk_h[i])
        _set_rm_risk_upper_bound(type, model, rcvar_risk[i], rm.settings.ub,
                                 "rcvar_risk_$(i)")
        _set_risk_expression(model, rcvar_risk[i], rm.settings.scale, rm.settings.flag)
    end
    return nothing
end
function set_rm(port::Portfolio, rm::EVaR, type::Union{Trad, RP, NOC};
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
                     scale_constr * sum(u_evar) <= scale_constr * z_evar
                     [i = 1:T],
                     [scale_constr * (-net_X[i] - t_evar), scale_constr * z_evar,
                      scale_constr * u_evar[i]] ∈ MOI.ExponentialCone()
                 end)
    _set_rm_risk_upper_bound(type, model, evar_risk, rm.settings.ub, "evar_risk")
    _set_risk_expression(model, evar_risk, rm.settings.scale, rm.settings.flag)

    return nothing
end
function set_rm(port::Portfolio, rms::AbstractVector{<:EVaR}, type::Union{Trad, RP, NOC};
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
        @constraints(model,
                     begin
                         scale_constr * sum(view(u_evar, :, j)) <= scale_constr * z_evar[j]
                         [i = 1:T],
                         [scale_constr * (-net_X[i] - t_evar[j]), scale_constr * z_evar[j],
                          scale_constr * u_evar[i, j]] ∈ MOI.ExponentialCone()
                     end)
        add_to_expression!(evar_risk[j], t_evar[j])
        add_to_expression!(evar_risk[j], -log(at), z_evar[j])
        _set_rm_risk_upper_bound(type, model, evar_risk[j], rm.settings.ub,
                                 "evar_risk_$(j)")
        _set_risk_expression(model, evar_risk[j], rm.settings.scale, rm.settings.flag)
    end
    return nothing
end
function set_rm(port::Portfolio, rm::RLVaR, type::Union{Trad, RP, NOC};
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
                   t_rvar
                   z_rvar >= 0
                   omega_rvar[1:T]
                   psi_rvar[1:T]
                   theta_rvar[1:T]
                   epsilon_rvar[1:T]
               end)
    @expression(model, rvar_risk, t_rvar + lnk * z_rvar + sum(psi_rvar .+ theta_rvar))
    @constraints(model,
                 begin
                     [i = 1:T],
                     [scale_constr * z_rvar * opk * ik2,
                      scale_constr * psi_rvar[i] * opk * ik,
                      scale_constr * epsilon_rvar[i]] ∈ MOI.PowerCone(iopk)
                     [i = 1:T],
                     [scale_constr * omega_rvar[i] * iomk,
                      scale_constr * theta_rvar[i] * ik, scale_constr * -z_rvar * ik2] ∈
                     MOI.PowerCone(omk)
                     scale_constr * (-net_X .- t_rvar .+ epsilon_rvar .+ omega_rvar) .<= 0
                 end)
    _set_rm_risk_upper_bound(type, model, rvar_risk, rm.settings.ub, "rvar_risk")
    _set_risk_expression(model, rvar_risk, rm.settings.scale, rm.settings.flag)
    return nothing
end
function set_rm(port::Portfolio, rms::AbstractVector{<:RLVaR}, type::Union{Trad, RP, NOC};
                returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
    scale_constr = model[:scale_constr]
    get_net_portfolio_returns(model, returns)
    net_X = model[:net_X]
    T = size(returns, 1)
    count = length(rms)
    @variables(model, begin
                   t_rvar[1:count]
                   z_rvar[1:count] .>= 0
                   omega_rvar[1:T, 1:count]
                   psi_rvar[1:T, 1:count]
                   theta_rvar[1:T, 1:count]
                   epsilon_rvar[1:T, 1:count]
               end)
    @expression(model, rvar_risk[1:count], zero(AffExpr))
    for (j, rm) ∈ pairs(rms)
        iat = inv(rm.alpha * T)
        lnk = (iat^rm.kappa - iat^(-rm.kappa)) / (2 * rm.kappa)
        opk = one(rm.kappa) + rm.kappa
        omk = one(rm.kappa) - rm.kappa
        ik2 = inv(2 * rm.kappa)
        ik = inv(rm.kappa)
        iopk = inv(opk)
        iomk = inv(omk)
        @constraints(model,
                     begin
                         [i = 1:T],
                         [scale_constr * z_rvar[j] * opk * ik2,
                          scale_constr * psi_rvar[i, j] * opk * ik,
                          scale_constr * epsilon_rvar[i, j]] ∈ MOI.PowerCone(iopk)
                         [i = 1:T],
                         [scale_constr * omega_rvar[i, j] * iomk,
                          scale_constr * theta_rvar[i, j] * ik,
                          scale_constr * -z_rvar[j] * ik2] ∈ MOI.PowerCone(omk)
                         scale_constr *
                         (-net_X .- t_rvar[j] .+ view(epsilon_rvar, :, j) .+
                          view(omega_rvar, :, j)) .<= 0
                     end)
        add_to_expression!(rvar_risk[j], t_rvar[j])
        add_to_expression!(rvar_risk[j], lnk, z_rvar[j])
        add_to_expression!(rvar_risk[j],
                           sum(view(psi_rvar, :, j) .+ view(theta_rvar, :, j)))
        _set_rm_risk_upper_bound(type, model, rvar_risk[j], rm.settings.ub,
                                 "rvar_risk_$(j)")
        _set_risk_expression(model, rvar_risk[j], rm.settings.scale, rm.settings.flag)
    end
    return nothing
end
function _DD_constraints(model, returns)
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
                     scale_constr * dd[1] == 0
                     scale_constr * view(dd, 2:(T + 1)) .>= 0
                     scale_constr * view(dd, 2:(T + 1)) .>=
                     scale_constr * (view(dd, 1:T) .- net_X)
                 end)

    return nothing
end
function set_rm(port::Portfolio, rm::MDD, type::Union{Trad, RP, NOC};
                returns::AbstractMatrix{<:Real}, kargs...)
    model = port.model
    scale_constr = model[:scale_constr]
    _DD_constraints(model, returns)
    dd = model[:dd]
    T = size(returns, 1)
    @variable(model, mdd)
    @expression(model, mdd_risk, mdd)
    @constraint(model, scale_constr * mdd .>= scale_constr * view(dd, 2:(T + 1)))
    _set_rm_risk_upper_bound(type, model, mdd_risk, rm.settings.ub, "mdd_risk")
    _set_risk_expression(model, mdd_risk, rm.settings.scale, rm.settings.flag)
    return nothing
end
function set_rm(port::Portfolio, rm::ADD, type::Union{Trad, RP, NOC};
                returns::AbstractMatrix{<:Real}, kargs...)
    model = port.model
    _DD_constraints(model, returns)
    dd = model[:dd]
    T = size(returns, 1)
    w = rm.w
    if isnothing(w)
        @expression(model, add_risk, mean(view(dd, 2:(T + 1))))
    else
        @expression(model, add_risk, mean(view(dd, 2:(T + 1)), w))
    end
    _set_rm_risk_upper_bound(type, model, add_risk, rm.settings.ub, "add_risk")
    _set_risk_expression(model, add_risk, rm.settings.scale, rm.settings.flag)
    return nothing
end
function set_rm(port::Portfolio, rms::AbstractVector{<:ADD}, type::Union{Trad, RP, NOC};
                returns::AbstractMatrix{<:Real}, kargs...)
    model = port.model
    _DD_constraints(model, returns)
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
        _set_rm_risk_upper_bound(type, model, add_risk[i], rm.settings.ub, "add_risk_$(i)")
        _set_risk_expression(model, add_risk[i], rm.settings.scale, rm.settings.flag)
    end
    return nothing
end
function set_rm(port::Portfolio, rm::UCI, type::Union{Trad, RP, NOC};
                returns::AbstractMatrix{<:Real}, kargs...)
    model = port.model
    scale_constr = model[:scale_constr]
    _DD_constraints(model, returns)
    dd = model[:dd]
    T = size(returns, 1)
    @variable(model, uci)
    @expression(model, uci_risk, uci / sqrt(T))
    @constraint(model,
                [scale_constr * uci; scale_constr * view(dd, 2:(T + 1))] ∈
                SecondOrderCone())
    _set_rm_risk_upper_bound(type, model, uci_risk, rm.settings.ub, "uci_risk")
    _set_risk_expression(model, uci_risk, rm.settings.scale, rm.settings.flag)
    return nothing
end
function set_rm(port::Portfolio, rm::CDaR, type::Union{Trad, RP, NOC};
                returns::AbstractMatrix{<:Real}, kargs...)
    model = port.model
    scale_constr = model[:scale_constr]
    _DD_constraints(model, returns)
    dd = model[:dd]
    T = size(returns, 1)
    iat = inv(rm.alpha * T)
    @variables(model, begin
                   dar
                   z_cdar[1:T] .>= 0
               end)
    @expression(model, cdar_risk, dar + sum(z_cdar) * iat)
    @constraints(model,
                 begin
                     scale_constr * z_cdar .>= scale_constr * (view(dd, 2:(T + 1)) .- dar)
                 end)
    _set_rm_risk_upper_bound(type, model, cdar_risk, rm.settings.ub, "cdar_risk")
    _set_risk_expression(model, cdar_risk, rm.settings.scale, rm.settings.flag)
    return nothing
end
function set_rm(port::Portfolio, rms::AbstractVector{<:CDaR}, type::Union{Trad, RP, NOC};
                returns::AbstractMatrix{<:Real}, kargs...)
    model = port.model
    scale_constr = model[:scale_constr]
    _DD_constraints(model, returns)
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
        @constraints(model,
                     begin
                         scale_constr * view(z_cdar, :, i) .>=
                         scale_constr * (view(dd, 2:(T + 1)) .- dar[i])
                     end)
        add_to_expression!(cdar_risk[i], dar[i])
        add_to_expression!(cdar_risk[i], iat, sum(view(z_cdar, :, i)))
        _set_rm_risk_upper_bound(type, model, cdar_risk[i], rm.settings.ub,
                                 "cdar_risk_$(i)")
        _set_risk_expression(model, cdar_risk[i], rm.settings.scale, rm.settings.flag)
    end
    return nothing
end
function set_rm(port::Portfolio, rm::EDaR, type::Union{Trad, RP, NOC};
                returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
    scale_constr = model[:scale_constr]
    _DD_constraints(model, returns)
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
                     scale_constr * sum(u_edar) <= scale_constr * z_edar
                     [i = 1:T],
                     [scale_constr * (dd[i + 1] - t_edar), scale_constr * z_edar,
                      scale_constr * u_edar[i]] ∈ MOI.ExponentialCone()
                 end)
    _set_rm_risk_upper_bound(type, model, edar_risk, rm.settings.ub, "edar_risk")
    _set_risk_expression(model, edar_risk, rm.settings.scale, rm.settings.flag)
    return nothing
end
function set_rm(port::Portfolio, rms::AbstractVector{<:EDaR}, type::Union{Trad, RP, NOC};
                returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
    scale_constr = model[:scale_constr]
    _DD_constraints(model, returns)
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
        @constraints(model,
                     begin
                         scale_constr * sum(view(u_edar, :, j)) <= scale_constr * z_edar[j]
                         [i = 1:T],
                         [scale_constr * (dd[i + 1] - t_edar[j]), scale_constr * z_edar[j],
                          scale_constr * u_edar[i, j]] ∈ MOI.ExponentialCone()
                     end)
        add_to_expression!(edar_risk[j], t_edar[j])
        add_to_expression!(edar_risk[j], -log(at), z_edar[j])
        _set_rm_risk_upper_bound(type, model, edar_risk[j], rm.settings.ub,
                                 "edar_risk_$(j)")
        _set_risk_expression(model, edar_risk[j], rm.settings.scale, rm.settings.flag)
    end
    return nothing
end
function set_rm(port::Portfolio, rm::RLDaR, type::Union{Trad, RP, NOC};
                returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
    scale_constr = model[:scale_constr]
    _DD_constraints(model, returns)
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
                   t_rdar
                   z_rdar >= 0
                   omega_rdar[1:T]
                   psi_rdar[1:T]
                   theta_rdar[1:T]
                   epsilon_rdar[1:T]
               end)
    @expression(model, rdar_risk, t_rdar + lnk * z_rdar + sum(psi_rdar .+ theta_rdar))
    @constraints(model,
                 begin
                     [i = 1:T],
                     [scale_constr * z_rdar * opk * ik2,
                      scale_constr * psi_rdar[i] * opk * ik,
                      scale_constr * epsilon_rdar[i]] ∈ MOI.PowerCone(iopk)
                     [i = 1:T],
                     [scale_constr * omega_rdar[i] * iomk,
                      scale_constr * theta_rdar[i] * ik, scale_constr * -z_rdar * ik2] ∈
                     MOI.PowerCone(omk)
                     scale_constr *
                     (view(dd, 2:(T + 1)) .- t_rdar .+ epsilon_rdar .+ omega_rdar) .<= 0
                 end)
    _set_rm_risk_upper_bound(type, model, rdar_risk, rm.settings.ub, "rdar_risk")
    _set_risk_expression(model, rdar_risk, rm.settings.scale, rm.settings.flag)
    return nothing
end
function set_rm(port::Portfolio, rms::AbstractVector{<:RLDaR}, type::Union{Trad, RP, NOC};
                returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
    scale_constr = model[:scale_constr]
    _DD_constraints(model, returns)
    dd = model[:dd]
    T = size(returns, 1)
    count = length(rms)
    @variables(model, begin
                   t_rdar[1:count]
                   z_rdar[1:count] .>= 0
                   omega_rdar[1:T, 1:count]
                   psi_rdar[1:T, 1:count]
                   theta_rdar[1:T, 1:count]
                   epsilon_rdar[1:T, 1:count]
               end)
    @expression(model, rdar_risk[1:count], zero(AffExpr))
    for (j, rm) ∈ pairs(rms)
        iat = inv(rm.alpha * T)
        lnk = (iat^rm.kappa - iat^(-rm.kappa)) / (2 * rm.kappa)
        opk = one(rm.kappa) + rm.kappa
        omk = one(rm.kappa) - rm.kappa
        ik2 = inv(2 * rm.kappa)
        ik = inv(rm.kappa)
        iopk = inv(opk)
        iomk = inv(omk)
        @constraints(model,
                     begin
                         [i = 1:T],
                         [scale_constr * z_rdar[j] * opk * ik2,
                          scale_constr * psi_rdar[i, j] * opk * ik,
                          scale_constr * epsilon_rdar[i, j]] ∈ MOI.PowerCone(iopk)
                         [i = 1:T],
                         [scale_constr * omega_rdar[i, j] * iomk,
                          scale_constr * theta_rdar[i, j] * ik,
                          scale_constr * -z_rdar[j] * ik2] ∈ MOI.PowerCone(omk)
                         scale_constr *
                         (view(dd, 2:(T + 1)) .- t_rdar[j] .+ view(epsilon_rdar, :, j) .+
                          view(omega_rdar, :, j)) .<= 0
                     end)
        add_to_expression!(rdar_risk[j], t_rdar[j])
        add_to_expression!(rdar_risk[j], lnk, z_rdar[j])
        add_to_expression!(rdar_risk[j],
                           sum(view(psi_rdar, :, j) .+ view(theta_rdar, :, j)))
        _set_rm_risk_upper_bound(type, model, rdar_risk[j], rm.settings.ub,
                                 "rdar_risk_$(j)")
        _set_risk_expression(model, rdar_risk[j], rm.settings.scale, rm.settings.flag)
    end
    return nothing
end
function set_rm(port::Portfolio, rm::Kurt, type::Union{Trad, RP, NOC}; kwargs...)
    model = port.model
    scale_constr = model[:scale_constr]
    _SDP_constraints(model, type)
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
                         [scale_constr * kurt_risk; scale_constr * x_kurt] ∈
                         SecondOrderCone()
                         [i = 1:Nf],
                         scale_constr * x_kurt[i] == scale_constr * tr(Bi[i] * W)
                     end)
    else
        L_2 = port.L_2
        S_2 = port.S_2
        sqrt_sigma_4 = sqrt(S_2 * kt * transpose(S_2))
        @expression(model, zkurt, L_2 * vec(W))
        @constraint(model,
                    [scale_constr * kurt_risk; scale_constr * sqrt_sigma_4 * zkurt] ∈
                    SecondOrderCone())
    end
    _set_rm_risk_upper_bound(type, model, kurt_risk, rm.settings.ub, "kurt_risk")
    _set_risk_expression(model, kurt_risk, rm.settings.scale, rm.settings.flag)
    return nothing
end
function set_rm(port::Portfolio, rms::AbstractVector{<:Kurt}, type::Union{Trad, RP, NOC};
                kwargs...)
    model = port.model
    scale_constr = model[:scale_constr]
    _SDP_constraints(model, type)
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
            @constraints(model,
                         begin
                             [scale_constr * kurt_risk[idx];
                              scale_constr * view(x_kurt, :, idx)] ∈ SecondOrderCone()
                             [i = 1:Nf],
                             scale_constr * x_kurt[i, idx] == scale_constr * tr(Bi[i] * W)
                         end)
            _set_rm_risk_upper_bound(type, model, kurt_risk[idx], rm.settings.ub,
                                     "kurt_risk_$(idx)")
            _set_risk_expression(model, kurt_risk[idx], rm.settings.scale, rm.settings.flag)
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
            @constraint(model,
                        [scale_constr * kurt_risk[idx];
                         scale_constr * sqrt_sigma_4 * view(zkurt, :, idx)] ∈
                        SecondOrderCone())
            _set_rm_risk_upper_bound(type, model, kurt_risk[idx], rm.settings.ub,
                                     "kurt_risk_$(idx)")
            _set_risk_expression(model, kurt_risk[idx], rm.settings.scale, rm.settings.flag)
        end
    end

    return nothing
end
function set_rm(port::Portfolio, rm::SKurt, type::Union{Trad, RP, NOC}; kwargs...)
    model = port.model
    scale_constr = model[:scale_constr]
    _SDP_constraints(model, type)
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
                         [scale_constr * skurt_risk; scale_constr * x_skurt] ∈
                         SecondOrderCone()
                         [i = 1:Nf],
                         scale_constr * x_skurt[i] == scale_constr * tr(Bi[i] * W)
                     end)
    else
        L_2 = port.L_2
        S_2 = port.S_2
        sqrt_sigma_4 = sqrt(S_2 * kt * transpose(S_2))
        @expression(model, zskurt, L_2 * vec(W))
        @constraint(model,
                    [scale_constr * skurt_risk; scale_constr * sqrt_sigma_4 * zskurt] ∈
                    SecondOrderCone())
    end
    _set_rm_risk_upper_bound(type, model, skurt_risk, rm.settings.ub, "skurt_risk")
    _set_risk_expression(model, skurt_risk, rm.settings.scale, rm.settings.flag)
    return nothing
end
function set_rm(port::Portfolio, rms::AbstractVector{<:SKurt}, type::Union{Trad, RP, NOC};
                kwargs...)
    model = port.model
    scale_constr = model[:scale_constr]
    _SDP_constraints(model, type)
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
            @constraints(model,
                         begin
                             [scale_constr * skurt_risk[idx];
                              scale_constr * view(x_skurt, :, idx)] ∈ SecondOrderCone()
                             [i = 1:Nf],
                             scale_constr * x_skurt[i, idx] == scale_constr * tr(Bi[i] * W)
                         end)
            _set_rm_risk_upper_bound(type, model, skurt_risk[idx], rm.settings.ub,
                                     "skurt_risk_$(idx)")
            _set_risk_expression(model, skurt_risk[idx], rm.settings.scale,
                                 rm.settings.flag)
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
            @constraint(model,
                        [scale_constr * skurt_risk[idx];
                         scale_constr * sqrt_sigma_4 * view(zskurt, :, idx)] ∈
                        SecondOrderCone())
            _set_rm_risk_upper_bound(type, model, skurt_risk[idx], rm.settings.ub,
                                     "skurt_risk_$(idx)")
            _set_risk_expression(model, skurt_risk[idx], rm.settings.scale,
                                 rm.settings.flag)
        end
    end

    return nothing
end
function _OWA_constraints(model, returns)
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
function set_rm(port::Portfolio, rm::GMD, type::Union{Trad, RP, NOC};
                returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
    scale_constr = model[:scale_constr]
    T = size(returns, 1)

    if !rm.owa.approx
        _OWA_constraints(model, returns)
        owa = model[:owa]
        ovec = range(1; stop = 1, length = T)
        @variables(model, begin
                       gmda[1:T]
                       gmdb[1:T]
                   end)
        @expression(model, gmd_risk, sum(gmda .+ gmdb))
        gmd_w = owa_gmd(T)
        @constraint(model,
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
                         scale_constr * (net_X .+ gmd_t .- gmd_nu .+ gmd_eta .-
                                         vec(sum(gmd_epsilon; dims = 2))) .== 0
                         scale_constr * (gmd_z .+ gmd_y) .==
                         scale_constr * vec(sum(gmd_psi; dims = 1))
                         [i = 1:M, j = 1:T],
                         [scale_constr * -gmd_z[i] * owa_p[i],
                          scale_constr * gmd_psi[j, i] * owa_p[i] / (owa_p[i] - 1),
                          scale_constr * gmd_epsilon[j, i]] ∈ MOI.PowerCone(inv(owa_p[i]))
                     end)
    end
    _set_rm_risk_upper_bound(type, model, gmd_risk, rm.settings.ub, "gmd_risk")
    _set_risk_expression(model, gmd_risk, rm.settings.scale, rm.settings.flag)
    return nothing
end
function set_rm(port::Portfolio, rm::TG, type::Union{Trad, RP, NOC};
                returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
    scale_constr = model[:scale_constr]
    T = size(returns, 1)
    alpha = rm.alpha
    a_sim = rm.a_sim
    alpha_i = rm.alpha_i
    if !rm.owa.approx
        _OWA_constraints(model, returns)
        owa = model[:owa]
        ovec = range(1; stop = 1, length = T)
        @variables(model, begin
                       tga[1:T]
                       tgb[1:T]
                   end)
        @expression(model, tg_risk, sum(tga .+ tgb))
        tg_w = owa_tg(T; alpha_i = alpha_i, alpha = alpha, a_sim = a_sim)
        @constraint(model,
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
                         scale_constr * (net_X .+ tg_t .- tg_nu .+ tg_eta .-
                                         vec(sum(tg_epsilon; dims = 2))) .== 0
                         scale_constr * (tg_z .+ tg_y) .==
                         scale_constr * vec(sum(tg_psi; dims = 1))
                         [i = 1:M, j = 1:T],
                         [scale_constr * -tg_z[i] * owa_p[i],
                          scale_constr * tg_psi[j, i] * owa_p[i] / (owa_p[i] - 1),
                          scale_constr * tg_epsilon[j, i]] ∈ MOI.PowerCone(inv(owa_p[i]))
                     end)
    end
    _set_rm_risk_upper_bound(type, model, tg_risk, rm.settings.ub, "tg_risk")
    _set_risk_expression(model, tg_risk, rm.settings.scale, rm.settings.flag)

    return nothing
end
function set_rm(port::Portfolio, rms::AbstractVector{<:TG}, type::Union{Trad, RP, NOC};
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
                _OWA_constraints(model, returns)
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
            @constraint(model,
                        scale_constr * owa * transpose(tg_w) .<=
                        scale_constr * (ovec * transpose(view(tga, :, idx)) +
                                        view(tgb, :, idx) * transpose(ovec)))
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
            @constraints(model,
                         begin
                             scale_constr *
                             (net_X .+ tg_t[idx] .- view(tg_nu, :, idx) .+
                              view(tg_eta, :, idx) .-
                              vec(sum(view(tg_epsilon, :, :, idx); dims = 2))) .== 0
                             scale_constr * (tg_z[:, idx] .+ tg_y[:, idx]) .==
                             scale_constr * vec(sum(view(tg_psi, :, :, idx); dims = 1))
                             [i = 1:M, j = 1:T],
                             [scale_constr * -tg_z[i, idx] * owa_p[i],
                              scale_constr * tg_psi[j, i, idx] * owa_p[i] / (owa_p[i] - 1),
                              scale_constr * tg_epsilon[j, i, idx]] ∈
                             MOI.PowerCone(inv(owa_p[i]))
                         end)
        end
        _set_rm_risk_upper_bound(type, model, tg_risk[idx], rm.settings.ub,
                                 "tg_risk_$(idx)")
        _set_risk_expression(model, tg_risk[idx], rm.settings.scale, rm.settings.flag)
    end
    return nothing
end
function set_rm(port::Portfolio, rm::TGRG, type::Union{Trad, RP, NOC};
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
        _OWA_constraints(model, returns)
        owa = model[:owa]
        ovec = range(1; stop = 1, length = T)
        @variables(model, begin
                       rtga[1:T]
                       rtgb[1:T]
                   end)
        @expression(model, rtg_risk, sum(rtga .+ rtgb))
        rtg_w = owa_rtg(T; alpha_i = alpha_i, alpha = alpha, a_sim = a_sim, beta_i = beta_i,
                        beta = beta, b_sim = b_sim)
        @constraint(model,
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
                         scale_constr * (net_X .+ rltg_t .- rltg_nu .+ rltg_eta .-
                                         vec(sum(rltg_epsilon; dims = 2))) .== 0
                         scale_constr * (rltg_z .+ rltg_y) .==
                         scale_constr * vec(sum(rltg_psi; dims = 1))
                         [i = 1:M, j = 1:T],
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
                         scale_constr * (-net_X .+ rhtg_t .- rhtg_nu .+ rhtg_eta .-
                                         vec(sum(rhtg_epsilon; dims = 2))) .== 0
                         scale_constr * (rhtg_z .+ rhtg_y) .==
                         scale_constr * vec(sum(rhtg_psi; dims = 1))
                         [i = 1:M, j = 1:T],
                         [scale_constr * -rhtg_z[i] * owa_p[i],
                          scale_constr * rhtg_psi[j, i] * owa_p[i] / (owa_p[i] - 1),
                          scale_constr * rhtg_epsilon[j, i]] ∈ MOI.PowerCone(inv(owa_p[i]))
                     end)
    end
    _set_rm_risk_upper_bound(type, model, rtg_risk, rm.settings.ub, "rtg_risk")
    _set_risk_expression(model, rtg_risk, rm.settings.scale, rm.settings.flag)
    return nothing
end
function set_rm(port::Portfolio, rms::AbstractVector{<:TGRG}, type::Union{Trad, RP, NOC};
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
                _OWA_constraints(model, returns)
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
            @constraint(model,
                        scale_constr * owa * transpose(rtg_w) .<=
                        scale_constr * (ovec * transpose(view(rtga, :, idx)) +
                                        view(rtgb, :, idx) * transpose(ovec)))
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
            @constraints(model,
                         begin
                             scale_constr *
                             (net_X .+ rltg_t[idx] .- view(rltg_nu, :, idx) .+
                              view(rltg_eta, :, idx) .-
                              vec(sum(view(rltg_epsilon, :, :, idx); dims = 2))) .== 0
                             scale_constr *
                             (view(rltg_z, :, idx) .+ view(rltg_y, :, idx)) .==
                             scale_constr * vec(sum(view(rltg_psi, :, :, idx); dims = 1))
                             [i = 1:M, j = 1:T],
                             [scale_constr * -rltg_z[i, idx] * owa_p[i],
                              scale_constr * rltg_psi[j, i, idx] * owa_p[i] /
                              (owa_p[i] - 1), scale_constr * rltg_epsilon[j, i, idx]] ∈
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
            @constraints(model,
                         begin
                             scale_constr *
                             (-net_X .+ rhtg_t[idx] .- view(rhtg_nu, :, idx) .+
                              view(rhtg_eta, :, idx) .-
                              vec(sum(rhtg_epsilon[:, :, idx]; dims = 2))) .== 0
                             scale_constr *
                             (view(rhtg_z, :, idx) .+ view(rhtg_y, :, idx)) .==
                             scale_constr * vec(sum(view(rhtg_psi, :, :, idx); dims = 1))
                             [i = 1:M, j = 1:T],
                             [scale_constr * -rhtg_z[i, idx] * owa_p[i],
                              scale_constr * rhtg_psi[j, i, idx] * owa_p[i] /
                              (owa_p[i] - 1), scale_constr * rhtg_epsilon[j, i, idx]] ∈
                             MOI.PowerCone(inv(owa_p[i]))
                         end)
        end
        _set_rm_risk_upper_bound(type, model, rtg_risk[idx], rm.settings.ub,
                                 "rtg_risk_$(idx)")
        _set_risk_expression(model, rtg_risk[idx], rm.settings.scale, rm.settings.flag)
    end
    return nothing
end
function set_rm(port::Portfolio, rm::OWA, type::Union{Trad, RP, NOC};
                returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
    scale_constr = model[:scale_constr]
    T = size(returns, 1)

    if !rm.owa.approx
        _OWA_constraints(model, returns)
        owa = model[:owa]
        ovec = range(1; stop = 1, length = T)
        @variables(model, begin
                       owa_a[1:T]
                       owa_b[1:T]
                   end)
        @expression(model, owa_risk, sum(owa_a .+ owa_b))
        owa_w = (isnothing(rm.w) || isempty(rm.w)) ? owa_gmd(T) : rm.w
        @constraint(model,
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
                         scale_constr * (net_X .+ owa_t .- owa_nu .+ owa_eta .-
                                         vec(sum(owa_epsilon; dims = 2))) .== 0
                         scale_constr * (owa_z .+ owa_y) .==
                         scale_constr * vec(sum(owa_psi; dims = 1))
                         [i = 1:M, j = 1:T],
                         [scale_constr * -owa_z[i] * owa_p[i],
                          scale_constr * owa_psi[j, i] * owa_p[i] / (owa_p[i] - 1),
                          scale_constr * owa_epsilon[j, i]] ∈ MOI.PowerCone(inv(owa_p[i]))
                     end)
    end
    _set_rm_risk_upper_bound(type, model, owa_risk, rm.settings.ub, "owa_risk")
    _set_risk_expression(model, owa_risk, rm.settings.scale, rm.settings.flag)
    return nothing
end
function set_rm(port::Portfolio, rms::AbstractVector{<:OWA}, type::Union{Trad, RP, NOC};
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
                _OWA_constraints(model, returns)
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
            @constraint(model,
                        scale_constr * owa * transpose(owa_w) .<=
                        scale_constr * (ovec * transpose(view(owa_a, :, idx)) +
                                        view(owa_b, :, idx) * transpose(ovec)))
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
            @constraints(model,
                         begin
                             scale_constr *
                             (net_X .+ owa_t[idx] .- view(owa_nu, :, idx) .+
                              view(owa_eta, :, idx) .-
                              vec(sum(view(owa_epsilon, :, :, idx); dims = 2))) .== 0
                             scale_constr * (view(owa_z, :, idx) .+ view(owa_y, :, idx)) .==
                             scale_constr * vec(sum(view(owa_psi, :, :, idx); dims = 1))
                             [i = 1:M, j = 1:T],
                             [scale_constr * -owa_z[i, idx] * owa_p[i],
                              scale_constr * owa_psi[j, i, idx] * owa_p[i] / (owa_p[i] - 1),
                              scale_constr * owa_epsilon[j, i, idx]] ∈
                             MOI.PowerCone(inv(owa_p[i]))
                         end)
        end
        _set_rm_risk_upper_bound(type, model, owa_risk[idx], rm.settings.ub,
                                 "owa_risk_$(idx)")
        _set_risk_expression(model, owa_risk[idx], rm.settings.scale, rm.settings.flag)
    end
    return nothing
end
function _BDVariance_constraints(::BDVAbsVal, model, Dt, Dx, T)
    scale_constr = model[:scale_constr]
    @constraint(model, [i = 1:T, j = i:T],
                [scale_constr * Dt[i, j]; scale_constr * Dx[i, j]] in MOI.NormOneCone(2))
    return nothing
end
function _BDVariance_constraints(::BDVIneq, model, Dt, Dx, T)
    scale_constr = model[:scale_constr]
    @constraints(model,
                 begin
                     [i = 1:T, j = i:T], scale_constr * Dt[i, j] >= scale_constr * Dx[i, j]
                     [i = 1:T, j = i:T],
                     scale_constr * Dt[i, j] >= scale_constr * -Dx[i, j]
                 end)
    return nothing
end
function set_rm(port::Portfolio, rm::BDVariance, type::Union{Trad, RP, NOC};
                returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
    get_portfolio_returns(model, returns)
    X = model[:X]
    T = size(returns, 1)
    iT2 = inv(T^2)
    ovec = range(1; stop = 1, length = T)
    @variable(model, Dt[1:T, 1:T], Symmetric)
    @expressions(model, begin
                     Dx, X * transpose(ovec) - ovec * transpose(X)
                     bd_variance_risk, iT2 * (dot(Dt, Dt) + iT2 * sum(Dt)^2)
                 end)
    _BDVariance_constraints(rm.formulation, model, Dt, Dx, T)
    _set_rm_risk_upper_bound(type, model, bd_variance_risk, rm.settings.ub,
                             "bd_variance_risk")
    _set_risk_expression(model, bd_variance_risk, rm.settings.scale, rm.settings.flag)
    return nothing
end
function set_rm(port::Portfolio, rm::Skew, type::Union{Trad, RP, NOC}; kwargs...)
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
    @constraint(model, [scale_constr * t_skew; scale_constr * G * w] ∈ SecondOrderCone())
    @expression(model, skew_risk, t_skew^2)
    _set_rm_risk_upper_bound(type, model, t_skew, rm.settings.ub, "skew_risk")
    _set_risk_expression(model, skew_risk, rm.settings.scale, rm.settings.flag)
    return nothing
end
function set_rm(port::Portfolio, rms::AbstractVector{<:Skew}, type::Union{Trad, RP, NOC};
                kwargs...)
    model = port.model
    scale_constr = model[:scale_constr]
    w = model[:w]
    count = length(rms)
    @variable(model, t_skew[1:count])
    @expression(model, skew_risk, t_skew .^ 2)
    for (idx, rm) ∈ pairs(rms)
        V = if (isnothing(rm.V) || isempty(rm.V))
            port.V
        else
            rm.V
        end
        G = real(sqrt(V))
        @constraint(model,
                    [scale_constr * t_skew[idx]; scale_constr * G * w] ∈ SecondOrderCone())
        _set_rm_risk_upper_bound(type, model, t_skew[idx], rm.settings.ub,
                                 "skew_risk_$(idx)")
        _set_risk_expression(model, skew_risk[idx], rm.settings.scale, rm.settings.flag)
    end
    return nothing
end
function set_rm(port::Portfolio, rm::SSkew, type::Union{Trad, RP, NOC}; kwargs...)
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
    @constraint(model, [scale_constr * t_sskew; scale_constr * G * w] ∈ SecondOrderCone())
    @expression(model, sskew_risk, t_sskew^2)
    _set_rm_risk_upper_bound(type, model, t_sskew, rm.settings.ub, "sskew_risk")
    _set_risk_expression(model, sskew_risk, rm.settings.scale, rm.settings.flag)
    return nothing
end
function set_rm(port::Portfolio, rms::AbstractVector{<:SSkew}, type::Union{Trad, RP, NOC};
                kwargs...)
    model = port.model
    scale_constr = model[:scale_constr]
    w = model[:w]
    count = length(rms)
    @variable(model, t_sskew[1:count])
    @expression(model, sskew_risk, t_sskew .^ 2)
    for (idx, rm) ∈ pairs(rms)
        V = if (isnothing(rm.V) || isempty(rm.V))
            port.SV
        else
            rm.V
        end
        G = real(sqrt(V))
        @constraint(model,
                    [scale_constr * t_sskew[idx]; scale_constr * G * w] ∈ SecondOrderCone())
        _set_rm_risk_upper_bound(type, model, t_sskew[idx], rm.settings.ub,
                                 "sskew_risk_$(idx)")
        _set_risk_expression(model, sskew_risk[idx], rm.settings.scale, rm.settings.flag)
    end
    return nothing
end
function risk_constraints(port, type::Union{Trad, RP, NOC},
                          rms::Union{RiskMeasure, AbstractVector}, mu, sigma, returns,
                          kelly_approx_idx = nothing)
    for rm ∈ rms
        set_rm(port, rm, type; mu = mu, sigma = sigma, returns = returns,
               kelly_approx_idx = kelly_approx_idx)
    end
    return nothing
end
