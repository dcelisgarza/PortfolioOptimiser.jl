# Risk expression
function _set_risk_expression(model, rm_risk, scale, flag::Bool)
    if flag
        if !haskey(model, :risk)
            @expression(model, risk, scale * rm_risk)
        else
            try
                risk = model[:risk]
                add_to_expression!(risk, scale, rm_risk)
            catch
                risk = model[:risk]
                @expression(model, tmp, risk + scale * rm_risk)
                unregister(model, :risk)
                @expression(model, risk, tmp)
                unregister(model, :tmp)
            end
        end
    end
    return nothing
end
function _get_ntwk_clust_method(port)
    return if isa(port.network_adj, SDP) || isa(port.cluster_adj, SDP)
        SDP()
    else
        NoAdj()
    end
end
function _set_rm_risk_upper_bound(args...)
    return nothing
end
function _set_rm_risk_upper_bound(::Union{Trad, NOC}, model, rm_risk, ub)
    if isfinite(ub)
        k = model[:k]
        @constraint(model, rm_risk .<= ub * k)
    end
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
    G = sqrt(sigma)
    w = model[:w]
    @variable(model, dev)
    @expression(model, variance_risk, dev^2)
    @constraint(model, [dev; G * w] ∈ SecondOrderCone())
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
    G = sqrt(sigma)
    w = model[:w]
    dev = model[:dev][idx]
    variance_risk = model[:variance_risk][idx]
    add_to_expression!(variance_risk, dev, dev)
    @constraint(model, [dev; G * w] ∈ SecondOrderCone())
    return nothing
end
function _variance_risk(::Union{NoAdj, IP}, ::Quad, model, sigma)
    G = sqrt(sigma)
    w = model[:w]
    @variable(model, dev)
    @expression(model, variance_risk, dot(w, sigma, w))
    @constraint(model, [dev; G * w] ∈ SecondOrderCone())
    return nothing
end
function _variance_risk(::Union{NoAdj, IP}, ::Quad, model::JuMP.Model,
                        sigma::AbstractMatrix, idx::Integer)
    G = sqrt(sigma)
    w = model[:w]
    dev = model[:dev][idx]
    variance_risk = model[:variance_risk][idx]
    add_to_expression!(variance_risk, dot(w, sigma, w))
    @constraint(model, [dev; G * w] ∈ SecondOrderCone())
    return nothing
end
function _variance_risk_bounds_expr(::SDP, model)
    return model[:variance_risk]
end
function _variance_risk_bounds_expr(::Union{NoAdj, IP}, model)
    return model[:dev]
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
    if isa(adjacency_constraint, SDP) && !haskey(model, :W)
        _SDP_constraints(model, type)
    end
    _variance_risk(adjacency_constraint, rm.formulation, model, sigma)
    variance_risk = model[:variance_risk]
    var_bound_expr = _variance_risk_bounds_expr(adjacency_constraint, model)
    ub = _variance_risk_bounds_val(adjacency_constraint, rm.settings.ub)
    _set_rm_risk_upper_bound(type, model, var_bound_expr, ub)
    _set_risk_expression(model, model[:variance_risk], rm.settings.scale, rm.settings.flag)
    return nothing
end
function set_rm(port, rms::AbstractVector{<:Variance}, type::Union{Trad, RP, NOC};
                sigma::AbstractMatrix{<:Real},
                kelly_approx_idx::Union{AbstractVector{<:Integer}, Nothing}, kwargs...)
    model = port.model
    adjacency_constraint = _get_ntwk_clust_method(port)
    if isa(adjacency_constraint, SDP) && !haskey(model, :W)
        _SDP_constraints(model, type)
    end
    count = length(rms)
    _variance_risk(adjacency_constraint, model, sigma, count)
    variance_risk = model[:variance_risk]
    var_bound_expr = _variance_risk_bounds_expr(adjacency_constraint, model)
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
        _set_rm_risk_upper_bound(type, model, var_bound_expr[i], ub)
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

    W = model[:W]
    N = size(W, 1)
    @variables(model, begin
                   Au[1:N, 1:N] .>= 0, Symmetric
                   Al[1:N, 1:N] .>= 0, Symmetric
               end)
    @constraint(model, Au .- Al .== W)
    return nothing
end
function _wc_variance_risk_variables(::Ellipse, model)
    if haskey(model, :E)
        return nothing
    end

    W = model[:W]
    N = size(W, 1)
    @variable(model, E[1:N, 1:N], Symmetric)
    @expression(model, WpE, W .+ E)
    @constraint(model, E ∈ PSDCone())
    return nothing
end
function _wc_variance_risk(::Box, model, sigma, cov_l, cov_u, cov_sigma, k_sigma)
    Au = model[:Au]
    Al = model[:Al]
    @expression(model, wc_variance_risk, tr(Au * cov_u) - tr(Al * cov_l))
    return nothing
end
function _wc_variance_risk(::Ellipse, model, sigma, cov_l, cov_u, cov_sigma, k_sigma)
    WpE = model[:WpE]
    G_sigma = sqrt(cov_sigma)
    @variable(model, t_ge)
    @expressions(model, begin
                     x_ge, G_sigma * vec(WpE)
                     wc_variance_risk, tr(sigma * WpE) + k_sigma * t_ge
                 end)
    @constraint(model, [t_ge; x_ge] ∈ SecondOrderCone())
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
    WpE = model[:WpE]
    G_sigma = sqrt(cov_sigma)
    t_ge = @variable(model)
    x_ge = @expression(model, G_sigma * vec(WpE))
    add_to_expression!(wc_variance_risk, tr(sigma * WpE))
    add_to_expression!(wc_variance_risk, k_sigma, t_ge)
    @constraint(model, [t_ge; x_ge] ∈ SecondOrderCone())
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
    _set_rm_risk_upper_bound(type, model, wc_variance_risk, rm.settings.ub)
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
        _set_rm_risk_upper_bound(type, model, wc_variance_risk[i], rm.settings.ub)
        _set_risk_expression(model, wc_variance_risk[i], rm.settings.scale,
                             rm.settings.flag)
    end
    return nothing
end
function set_rm(port, rm::SD, type::Union{Trad, RP, NOC}; sigma::AbstractMatrix{<:Real},
                kwargs...)
    model = port.model
    w = model[:w]
    use_portfolio_sigma = isnothing(rm.sigma) || isempty(rm.sigma)
    if !use_portfolio_sigma
        sigma = rm.sigma
    end
    @variable(model, sd_risk)
    G = sqrt(sigma)
    @constraint(model, [sd_risk; G * w] ∈ SecondOrderCone())
    _set_rm_risk_upper_bound(type, model, sd_risk, rm.settings.ub)
    _set_risk_expression(model, sd_risk, rm.settings.scale, rm.settings.flag)
    return nothing
end
function set_rm(port, rms::AbstractVector{<:SD}, type::Union{Trad, RP, NOC};
                sigma::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
    w = model[:w]
    count = length(rms)
    @variable(model, sd_risk[1:count])
    for (i, rm) ∈ pairs(rms)
        use_portfolio_sigma = (isnothing(rm.sigma) || isempty(rm.sigma))
        if !use_portfolio_sigma
            sigma = rm.sigma
        end
        G = sqrt(sigma)
        @constraint(model, [sd_risk[i]; G * w] ∈ SecondOrderCone())
        _set_rm_risk_upper_bound(type, model, sd_risk[i], rm.settings.ub)
        _set_risk_expression(model, sd_risk[i], rm.settings.scale, rm.settings.flag)
    end
    return nothing
end
function set_rm(port::OmniPortfolio, rm::MAD, type::Union{Trad, RP, NOC};
                mu::AbstractVector{<:Real}, returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
    w = model[:w]
    T = size(returns, 1)
    if !(isnothing(rm.mu) || isempty(rm.mu))
        mu = rm.mu
    end
    mar = returns .- transpose(mu)
    @variable(model, mad[1:T] .>= 0)
    @expression(model, mad_risk, 2 * sum(mad) / T)
    @constraint(model, mar * w .>= -mad)
    _set_rm_risk_upper_bound(type, model, mad_risk, rm.settings.ub)
    _set_risk_expression(model, mad_risk, rm.settings.scale, rm.settings.flag)
    return nothing
end
function set_rm(port::OmniPortfolio, rms::AbstractVector{<:MAD}, type::Union{Trad, RP, NOC};
                mu::AbstractVector{<:Real}, returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
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
        @constraint(model, mar * w .>= -view(mad, :, i))
        add_to_expression!(mad_risk[i], iT2, sum(view(mad, :, i)))
        _set_rm_risk_upper_bound(type, model, mad_risk[i], rm.settings.ub)
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
    @variable(model, tsvariance)
    @constraint(model, [tsvariance; 0.5; svariance] in RotatedSecondOrderCone())
    @expression(model, svariance_risk, tsvariance * iTm1)
    return nothing
end
function _semi_variance_risk(::SOC, model, svariance, svariance_risk, iTm1)
    tsvariance = @variable(model)
    @constraint(model, [tsvariance; 0.5; svariance] in RotatedSecondOrderCone())
    add_to_expression!(svariance_risk, iTm1, tsvariance)
    return nothing
end
function set_rm(port::OmniPortfolio, rm::SVariance, type::Union{Trad, RP, NOC};
                mu::AbstractVector{<:Real}, returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
    w = model[:w]
    T = size(returns, 1)
    if !(isnothing(rm.mu) || isempty(rm.mu))
        mu = rm.mu
    end
    mar = returns .- transpose(mu)
    target = rm.target
    @variable(model, svariance[1:T] .>= target)
    _semi_variance_risk(rm.formulation, model, svariance, inv(T - 1))
    @constraint(model, mar * w .>= -svariance)
    svariance_risk = model[:svariance_risk]
    _set_rm_risk_upper_bound(type, model, svariance_risk, rm.settings.ub)
    _set_risk_expression(model, svariance_risk, rm.settings.scale, rm.settings.flag)

    return nothing
end
function set_rm(port::OmniPortfolio, rms::AbstractVector{<:SVariance},
                type::Union{Trad, RP, NOC}; mu::AbstractVector{<:Real},
                returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
    w = model[:w]
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
        @constraints(model, begin
                         view(svariance, :, i) .>= target
                         mar * w .>= -view(svariance, :, i)
                     end)
        _semi_variance_risk(rm.formulation, model, view(svariance, :, i), svariance_risk[i],
                            iTm1)
        _set_rm_risk_upper_bound(type, model, svariance_risk[i], rm.settings.ub)
        _set_risk_expression(model, svariance_risk[i], rm.settings.scale, rm.settings.flag)
    end
    return nothing
end
function set_rm(port::OmniPortfolio, rm::SSD, type::Union{Trad, RP, NOC};
                mu::AbstractVector{<:Real}, returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
    w = model[:w]
    T = size(returns, 1)
    if !(isnothing(rm.mu) || isempty(rm.mu))
        mu = rm.mu
    end
    mar = returns .- transpose(mu)
    target = rm.target
    @variables(model, begin
                   ssd[1:T] .>= target
                   sdev
               end)
    @expression(model, sdev_risk, sdev / sqrt(T - 1))
    @constraints(model, begin
                     mar * w .>= -ssd
                     [sdev; ssd] ∈ SecondOrderCone()
                 end)
    _set_rm_risk_upper_bound(type, model, sdev_risk, rm.settings.ub)
    _set_risk_expression(model, sdev_risk, rm.settings.scale, rm.settings.flag)

    return nothing
end
function set_rm(port::OmniPortfolio, rms::AbstractVector{<:SSD}, type::Union{Trad, RP, NOC};
                mu::AbstractVector{<:Real}, returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
    w = model[:w]
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
        @constraints(model, begin
                         view(ssd, :, i) .>= target
                         mar * w .>= -view(ssd, :, i)
                         [sdev[i]; view(ssd, :, i)] ∈ SecondOrderCone()
                     end)
        add_to_expression!(sdev_risk[i], iTm1, sdev[i])
        _set_rm_risk_upper_bound(type, model, sdev_risk[i], rm.settings.ub)
        _set_risk_expression(model, sdev_risk[i], rm.settings.scale, rm.settings.flag)
    end
    return nothing
end
function set_rm(port::OmniPortfolio, rm::FLPM, type::Union{Trad, RP, NOC};
                returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
    w = model[:w]
    k = model[:k]
    T = size(returns, 1)
    ret_target = rm.ret_target
    target = rm.target
    mar = returns .- transpose(ret_target)
    @variable(model, flpm[1:T] .>= 0)
    @expression(model, flpm_risk, sum(flpm) / T)
    @constraint(model, flpm .>= target * k .- mar * w)
    _set_rm_risk_upper_bound(type, model, flpm_risk, rm.settings.ub)
    _set_risk_expression(model, flpm_risk, rm.settings.scale, rm.settings.flag)
    return nothing
end
function set_rm(port::OmniPortfolio, rms::AbstractVector{<:FLPM},
                type::Union{Trad, RP, NOC}; returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
    w = model[:w]
    k = model[:k]
    T = size(returns, 1)
    iT = inv(T)
    count = length(rms)
    @variable(model, flpm[1:T, 1:count] .>= 0)
    @expression(model, flpm_risk[1:count], zero(AffExpr))
    for (i, rm) ∈ pairs(rms)
        ret_target = rm.ret_target
        target = rm.target
        mar = returns .- transpose(ret_target)
        add_to_expression!(flpm_risk[i], iT, sum(view(flpm, :, i)))
        @constraint(model, view(flpm, :, i) .>= target * k .- mar * w)
        _set_rm_risk_upper_bound(type, model, flpm_risk[i], rm.settings.ub)
        _set_risk_expression(model, flpm_risk[i], rm.settings.scale, rm.settings.flag)
    end
    return nothing
end
function set_rm(port::OmniPortfolio, rm::SLPM, type::Union{Trad, RP, NOC};
                returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
    w = model[:w]
    k = model[:k]
    T = size(returns, 1)
    ret_target = rm.ret_target
    target = rm.target
    mar = returns .- transpose(ret_target)
    @variables(model, begin
                   slpm[1:T] .>= 0
                   tslpm
               end)
    @expression(model, slpm_risk, tslpm / sqrt(T - 1))
    @constraints(model, begin
                     slpm .>= target * k .- mar * w
                     [tslpm; slpm] ∈ SecondOrderCone()
                 end)
    _set_rm_risk_upper_bound(type, model, slpm_risk, rm.settings.ub)
    _set_risk_expression(model, slpm_risk, rm.settings.scale, rm.settings.flag)
    return nothing
end
function set_rm(port::OmniPortfolio, rms::AbstractVector{<:SLPM},
                type::Union{Trad, RP, NOC}; returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
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
        target = rm.target
        mar = returns .- transpose(ret_target)
        add_to_expression!(slpm_risk[i], iTm1, tslpm[i])
        @constraints(model, begin
                         view(slpm, :, i) .>= target * k .- mar * w
                         [tslpm[i]; view(slpm, :, i)] ∈ SecondOrderCone()
                     end)
        _set_rm_risk_upper_bound(type, model, slpm_risk[i], rm.settings.ub)
        _set_risk_expression(model, slpm_risk[i], rm.settings.scale, rm.settings.flag)
    end
    return nothing
end
function _wr_risk(model, returns)
    if haskey(model, :wr)
        return nothing
    end
    get_net_portfolio_returns(model, returns)
    net_X = model[:net_X]
    @variable(model, wr)
    @expression(model, wr_risk, wr)
    @constraint(model, -net_X .<= wr)

    return nothing
end
function set_rm(port::OmniPortfolio, rm::WR, type::Union{Trad, RP, NOC};
                returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
    _wr_risk(model, returns)
    wr_risk = model[:wr_risk]
    _set_rm_risk_upper_bound(type, model, wr_risk, rm.settings.ub)
    _set_risk_expression(model, wr_risk, rm.settings.scale, rm.settings.flag)
    return nothing
end
function set_rm(port::OmniPortfolio, rm::RG, type::Union{Trad, RP, NOC};
                returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
    _wr_risk(model, returns)
    wr_risk = model[:wr_risk]
    net_X = model[:net_X]
    @variable(model, br)
    @expression(model, rg_risk, wr_risk - br)
    @constraint(model, -net_X .>= br)
    _set_rm_risk_upper_bound(type, model, rg_risk, rm.settings.ub)
    _set_risk_expression(model, rg_risk, rm.settings.scale, rm.settings.flag)
    return nothing
end
function set_rm(port::OmniPortfolio, rm::CVaR, type::Union{Trad, RP, NOC};
                returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
    get_net_portfolio_returns(model, returns)
    net_X = model[:net_X]
    T = size(returns, 1)
    iat = inv(rm.alpha * T)
    @variables(model, begin
                   var
                   z_var[1:T] .>= 0
               end)
    @expression(model, cvar_risk, var + sum(z_var) * iat)
    @constraint(model, z_var .>= -net_X .- var)
    _set_rm_risk_upper_bound(type, model, cvar_risk, rm.settings.ub)
    _set_risk_expression(model, cvar_risk, rm.settings.scale, rm.settings.flag)
    return nothing
end
function set_rm(port::OmniPortfolio, rms::AbstractVector{<:CVaR},
                type::Union{Trad, RP, NOC}; returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
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
        @constraint(model, view(z_var, :, i) .>= -net_X .- var[i])
        _set_rm_risk_upper_bound(type, model, cvar_risk[i], rm.settings.ub)
        _set_risk_expression(model, cvar_risk[i], rm.settings.scale, rm.settings.flag)
    end
    return nothing
end
######
function set_rm(port::OmniPortfolio, rm::DRCVaR, type::Union{Trad, RP, NOC};
                returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
    get_net_portfolio_returns(model, returns)
    get_one_plus_net_returns(model, returns)
    net_X = model[:net_X]
    net_RP1 = model[:net_RP1]
    T, N = size(returns)

    l1 = rm.l
    alpha = rm.alpha
    radius = rm.r

    a1 = -one(alpha)
    a2 = a1 - l1 * inv(alpha)
    l2 = l1 * (one(alpha) - inv(alpha))
    ovec = range(; start = one(alpha), stop = one(alpha), length = N)

    @variables(model, begin
                   lb
                   tau
                   s[1:T]
                   u[1:T, 1:N] >= 0
                   v[1:T, 1:N] >= 0
                   tu_drcvar[1:T]
                   tv_drcvar[1:T]
               end)
    @constraints(model,
                 begin
                     l1 * tau .+ a1 * net_X .+ (u .* net_RP1) * ovec .<= s
                     l2 * tau .+ a2 * net_X .+ (v .* net_RP1) * ovec .<= s
                     [i = 1:T],
                     [tu_drcvar[i]; -view(u, i, :) .- a1 * w] in
                     MOI.NormInfinityCone(1 + N)
                     [i = 1:T],
                     [tv_drcvar[i]; -view(v, i, :) .- a2 * w] in
                     MOI.NormInfinityCone(1 + N)
                     tu_drcvar .<= lb
                     tv_drcvar .<= lb
                 end)

    @expression(model, drcvar_risk, radius * lb + sum(s) * inv(T))
    _set_rm_risk_upper_bound(type, model, drcvar_risk, rm.settings.ub)
    _set_risk_expression(model, drcvar_risk, rm.settings.scale, rm.settings.flag)
    return nothing
end
function set_rm(port::OmniPortfolio, rms::AbstractVector{<:DRCVaR},
                type::Union{Trad, RP, NOC}; returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
    get_net_portfolio_returns(model, returns)
    get_one_plus_net_returns(model, returns)
    net_X = model[:net_X]
    net_RP1 = model[:net_RP1]
    T, N = size(returns)

    iT = inv(T)
    count = length(rms)

    @variables(model, begin
                   lb[1:count]
                   tau[1:count]
                   s[1:T, 1:count]
                   u[1:T, 1:N, 1:count] >= 0
                   v[1:T, 1:N, 1:count] >= 0
                   tu_drcvar[1:T, 1:count]
                   tv_drcvar[1:T, 1:count]
               end)
    @expression(model, drcvar_risk[1:count], zero(AffExpr))
    for (j, rm) ∈ pairs(rms)
        l1 = rm.l
        alpha = rm.alpha
        radius = rm.r

        a1 = -one(alpha)
        a2 = a1 - l1 * inv(alpha)
        l2 = l1 * (one(alpha) - inv(alpha))
        ovec = range(; start = one(alpha), stop = one(alpha), length = N)

        @constraints(model,
                     begin
                         l1 * tau[j] .+ a1 * net_X .+ (u .* net_RP1) * ovec .<=
                         view(s, :, j)
                         l2 * tau[j] .+ a2 * net_X .+ (v .* net_RP1) * ovec .<=
                         view(s, :, j)
                         [i = 1:T],
                         [tu_drcvar[i, j]; -view(u, i, :, j) .- a1 * w] in
                         MOI.NormInfinityCone(1 + N)
                         [i = 1:T],
                         [tv_drcvar[i, j]; -view(v, i, :, j) .- a2 * w] in
                         MOI.NormInfinityCone(1 + N)
                         view(tu_drcvar, :, j) .<= lb[j]
                         view(tv_drcvar, :, j) .<= lb[j]
                     end)

        @expression(model, drcvar_risk, radius * lb + sum(s) * inv(T))

        add_to_expression!(drcvar_risk, radius, lb[j])
        add_to_expression!(drcvar_risk, iT, sum(view(s, :, j)))
        _set_rm_risk_upper_bound(type, model, drcvar_risk[i], rm.settings.ub)
        _set_risk_expression(model, drcvar_risk[i], rm.settings.scale, rm.settings.flag)
    end
    return nothing
end
######
function set_rm(port::OmniPortfolio, rm::CVaRRG, type::Union{Trad, RP, NOC};
                returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
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
                     z_var_l .>= -net_X .- var_l
                     z_var_h .<= -net_X .- var_h
                 end)

    _set_rm_risk_upper_bound(type, model, rcvar_risk, rm.settings.ub)
    _set_risk_expression(model, rcvar_risk, rm.settings.scale, rm.settings.flag)
    return nothing
end
function set_rm(port::OmniPortfolio, rms::AbstractVector{<:CVaRRG},
                type::Union{Trad, RP, NOC}; returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
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
        @constraints(model, begin
                         view(z_var_l, :, i) .>= -net_X .- var_l[i]
                         view(z_var_h, :, i) .<= -net_X .- var_h[i]
                     end)
        add_to_expression!(cvar_risk_l[i], var_l[i])
        add_to_expression!(cvar_risk_l[i], iat, sum(view(z_var_l, :, i)))
        add_to_expression!(cvar_risk_h[i], var_h[i])
        add_to_expression!(cvar_risk_h[i], ibt, sum(view(z_var_h, :, i)))
        add_to_expression!(rcvar_risk[i], cvar_risk_l[i])
        add_to_expression!(rcvar_risk[i], -1, cvar_risk_h[i])
        _set_rm_risk_upper_bound(type, model, rcvar_risk[i], rm.settings.ub)
        _set_risk_expression(model, rcvar_risk[i], rm.settings.scale, rm.settings.flag)
    end
    return nothing
end
function set_rm(port::OmniPortfolio, rm::EVaR, type::Union{Trad, RP, NOC};
                returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
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
                     sum(u_evar) <= z_evar
                     [i = 1:T],
                     [-net_X[i] - t_evar, z_evar, u_evar[i]] ∈ MOI.ExponentialCone()
                 end)
    _set_rm_risk_upper_bound(type, model, evar_risk, rm.settings.ub)
    _set_risk_expression(model, evar_risk, rm.settings.scale, rm.settings.flag)

    return nothing
end
function set_rm(port::OmniPortfolio, rms::AbstractVector{<:EVaR},
                type::Union{Trad, RP, NOC}; returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
    get_net_portfolio_returns(model, returns)
    net_X = model[:net_X]
    T = size(returns, 1)
    count = length(rms)
    @variables(model, begin
                   t_evar[1:count]
                   z_evar[1:count] >= 0
                   u_evar[1:T, 1:count]
               end)
    @expression(model, evar_risk[1:count], zero(AffExpr))
    for (j, rm) ∈ pairs(rms)
        at = rm.alpha * T
        @constraints(model,
                     begin
                         sum(view(u_evar, :, j)) <= z_evar[j]
                         [i = 1:T],
                         [-net_X[i] - t_evar[j], z_evar[j], u_evar[i, j]] ∈
                         MOI.ExponentialCone()
                     end)
        add_to_expression!(evar_risk[j], t_evar[j])
        add_to_expression!(evar_risk[j], -log(at), z_evar[j])
        _set_rm_risk_upper_bound(type, model, evar_risk[j], rm.settings.ub)
        _set_risk_expression(model, evar_risk[j], rm.settings.scale, rm.settings.flag)
    end
    return nothing
end
function set_rm(port::OmniPortfolio, rm::RLVaR, type::Union{Trad, RP, NOC};
                returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
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
                     [z_rvar * opk * ik2, psi_rvar[i] * opk * ik, epsilon_rvar[i]] ∈
                     MOI.PowerCone(iopk)
                     [i = 1:T],
                     [omega_rvar[i] * iomk, theta_rvar[i] * ik, -z_rvar * ik2] ∈
                     MOI.PowerCone(omk)
                 end)
    @constraint(model, -net_X .- t_rvar .+ epsilon_rvar .+ omega_rvar .<= 0)
    _set_rm_risk_upper_bound(type, model, rvar_risk, rm.settings.ub)
    _set_risk_expression(model, rvar_risk, rm.settings.scale, rm.settings.flag)
    return nothing
end
function set_rm(port::OmniPortfolio, rms::AbstractVector{<:RLVaR},
                type::Union{Trad, RP, NOC}; returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
    get_net_portfolio_returns(model, returns)
    net_X = model[:net_X]
    T = size(returns, 1)
    count = length(rms)
    @variables(model, begin
                   t_rvar[1:count]
                   z_rvar[1:count] >= 0
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
                         [z_rvar[j] * opk * ik2, psi_rvar[i, j] * opk * ik,
                          epsilon_rvar[i, j]] ∈ MOI.PowerCone(iopk)
                         [i = 1:T],
                         [omega_rvar[i, j] * iomk, theta_rvar[i, j] * ik,
                          -z_rvar[j] * ik2] ∈ MOI.PowerCone(omk)
                         -net_X .- t_rvar[j] .+ view(epsilon_rvar, :, j) .+
                         view(omega_rvar, :, j) .<= 0
                     end)
        add_to_expression!(rvar_risk[j], t_rvar[j])
        add_to_expression!(rvar_risk[j], lnk, z_rvar[j])
        add_to_expression!(rvar_risk[j],
                           sum(view(psi_rvar, :, j) .+ view(theta_rvar, :, j)))
        _set_rm_risk_upper_bound(type, model, rvar_risk[j], rm.settings.ub)
        _set_risk_expression(model, rvar_risk[j], rm.settings.scale, rm.settings.flag)
    end
    return nothing
end
function _DD_constraints(model, returns)
    if haskey(model, :dd)
        return nothing
    end

    get_net_portfolio_returns(model, returns)
    net_X = model[:net_X]
    T = size(returns, 1)
    @variable(model, dd[1:(T + 1)])
    @constraints(model, begin
                     view(dd, 2:(T + 1)) .>= view(dd, 1:T) .- net_X
                     view(dd, 2:(T + 1)) .>= 0
                     dd[1] == 0
                 end)

    return nothing
end
function set_rm(port::OmniPortfolio, rm::MDD, type::Union{Trad, RP, NOC};
                returns::AbstractMatrix{<:Real}, kargs...)
    model = port.model
    _DD_constraints(model, returns)
    dd = model[:dd]
    T = size(returns, 1)
    @variable(model, mdd)
    @expression(model, mdd_risk, mdd)
    @constraint(model, mdd .>= view(dd, 2:(T + 1)))
    _set_rm_risk_upper_bound(type, model, view(dd, 2:(T + 1)), rm.settings.ub)
    _set_risk_expression(model, mdd_risk, rm.settings.scale, rm.settings.flag)
    return nothing
end
function set_rm(port::OmniPortfolio, rm::ADD, type::Union{Trad, RP, NOC};
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
    _set_rm_risk_upper_bound(type, model, add_risk, rm.settings.ub)
    _set_risk_expression(model, add_risk, rm.settings.scale, rm.settings.flag)
    return nothing
end
function set_rm(port::OmniPortfolio, rms::AbstractVector{<:ADD}, type::Union{Trad, RP, NOC};
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
        _set_rm_risk_upper_bound(type, model, add_risk[i], rm.settings.ub)
        _set_risk_expression(model, add_risk[i], rm.settings.scale, rm.settings.flag)
    end
    return nothing
end
function set_rm(port::OmniPortfolio, rm::UCI, type::Union{Trad, RP, NOC};
                returns::AbstractMatrix{<:Real}, kargs...)
    model = port.model
    _DD_constraints(model, returns)
    dd = model[:dd]
    T = size(returns, 1)
    @variable(model, uci)
    @expression(model, uci_risk, uci / sqrt(T))
    @constraint(model, [uci; view(dd, 2:(T + 1))] ∈ SecondOrderCone())
    _set_rm_risk_upper_bound(type, model, uci_risk, rm.settings.ub)
    _set_risk_expression(model, uci_risk, rm.settings.scale, rm.settings.flag)
    return nothing
end
function set_rm(port::OmniPortfolio, rm::CDaR, type::Union{Trad, RP, NOC};
                returns::AbstractMatrix{<:Real}, kargs...)
    model = port.model
    _DD_constraints(model, returns)
    dd = model[:dd]
    T = size(returns, 1)
    iat = inv(rm.alpha * T)
    @variables(model, begin
                   dar
                   z_cdar[1:T] .>= 0
               end)
    @expression(model, cdar_risk, dar + sum(z_cdar) * iat)
    @constraint(model, z_cdar .>= view(dd, 2:(T + 1)) .- dar)
    _set_rm_risk_upper_bound(type, model, cdar_risk, rm.settings.ub)
    _set_risk_expression(model, cdar_risk, rm.settings.scale, rm.settings.flag)
    return nothing
end
function set_rm(port::OmniPortfolio, rms::AbstractVector{<:CDaR},
                type::Union{Trad, RP, NOC}; returns::AbstractMatrix{<:Real}, kargs...)
    model = port.model
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
        @constraint(model, view(z_cdar, :, i) .>= view(dd, 2:(T + 1)) .- dar[i])
        add_to_expression!(cdar_risk[i], dar[i])
        add_to_expression!(cdar_risk[i], iat, sum(view(z_cdar, :, i)))
        _set_rm_risk_upper_bound(type, model, cdar_risk[i], rm.settings.ub)
        _set_risk_expression(model, cdar_risk[i], rm.settings.scale, rm.settings.flag)
    end
    return nothing
end
function set_rm(port::OmniPortfolio, rm::EDaR, type::Union{Trad, RP, NOC};
                returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
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
                     sum(u_edar) <= z_edar
                     [i = 1:T],
                     [dd[i + 1] - t_edar, z_edar, u_edar[i]] ∈ MOI.ExponentialCone()
                 end)
    _set_rm_risk_upper_bound(type, model, edar_risk, rm.settings.ub)
    _set_risk_expression(model, edar_risk, rm.settings.scale, rm.settings.flag)
    return nothing
end
function set_rm(port::OmniPortfolio, rms::AbstractVector{<:EDaR},
                type::Union{Trad, RP, NOC}; returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
    _DD_constraints(model, returns)
    dd = model[:dd]
    T = size(returns, 1)
    count = length(rms)
    @variables(model, begin
                   t_edar[1:count]
                   z_edar[1:count] >= 0
                   u_edar[1:T, 1:count]
               end)
    @expression(model, edar_risk[1:count], zero(AffExpr))
    for (j, rm) ∈ pairs(rms)
        at = rm.alpha * T
        @constraints(model,
                     begin
                         sum(view(u_edar, :, j)) <= z_edar[j]
                         [i = 1:T],
                         [dd[i + 1] - t_edar[j], z_edar[j], u_edar[i, j]] ∈
                         MOI.ExponentialCone()
                     end)
        add_to_expression!(edar_risk[j], t_edar[j])
        add_to_expression!(edar_risk[j], -log(at), z_edar[j])
        _set_rm_risk_upper_bound(type, model, edar_risk[j], rm.settings.ub)
        _set_risk_expression(model, edar_risk[j], rm.settings.scale, rm.settings.flag)
    end
    return nothing
end
function set_rm(port::OmniPortfolio, rm::RLDaR, type::Union{Trad, RP, NOC};
                returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
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
                     [z_rdar * opk * ik2, psi_rdar[i] * opk * ik, epsilon_rdar[i]] ∈
                     MOI.PowerCone(iopk)
                     [i = 1:T],
                     [omega_rdar[i] * iomk, theta_rdar[i] * ik, -z_rdar * ik2] ∈
                     MOI.PowerCone(omk)
                     view(dd, 2:(T + 1)) .- t_rdar .+ epsilon_rdar .+ omega_rdar .<= 0
                 end)
    _set_rm_risk_upper_bound(type, model, rdar_risk, rm.settings.ub)
    _set_risk_expression(model, rdar_risk, rm.settings.scale, rm.settings.flag)
    return nothing
end
function set_rm(port::OmniPortfolio, rms::AbstractVector{<:RLDaR},
                type::Union{Trad, RP, NOC}; returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
    _DD_constraints(model, returns)
    dd = model[:dd]
    T = size(returns, 1)
    count = length(rms)
    @variables(model, begin
                   t_rdar[1:count]
                   z_rdar[1:count] >= 0
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
                         [z_rdar[j] * opk * ik2, psi_rdar[i, j] * opk * ik,
                          epsilon_rdar[i, j]] ∈ MOI.PowerCone(iopk)
                         [i = 1:T],
                         [omega_rdar[i, j] * iomk, theta_rdar[i, j] * ik,
                          -z_rdar[j] * ik2] ∈ MOI.PowerCone(omk)
                         view(dd, 2:(T + 1)) .- t_rdar[j] .+ view(epsilon_rdar, :, j) .+
                         view(omega_rdar, :, j) .<= 0
                     end)
        add_to_expression!(rdar_risk[j], t_rdar[j])
        add_to_expression!(rdar_risk[j], lnk, z_rdar[j])
        add_to_expression!(rdar_risk[j],
                           sum(view(psi_rdar, :, j) .+ view(theta_rdar, :, j)))
        _set_rm_risk_upper_bound(type, model, rdar_risk[j], rm.settings.ub)
        _set_risk_expression(model, rdar_risk[j], rm.settings.scale, rm.settings.flag)
    end
    return nothing
end
function set_rm(port::OmniPortfolio, rm::Kurt, type::Union{Trad, RP, NOC}; kwargs...)
    model = port.model
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
        @constraints(model, begin
                         [kurt_risk; x_kurt] ∈ SecondOrderCone()
                         [i = 1:Nf], x_kurt[i] == tr(Bi[i] * W)
                     end)
    else
        L_2 = port.L_2
        S_2 = port.S_2
        sqrt_sigma_4 = sqrt(S_2 * kt * transpose(S_2))
        @expression(model, zkurt, L_2 * vec(W))
        @constraint(model, [kurt_risk; sqrt_sigma_4 * zkurt] ∈ SecondOrderCone())
    end
    _set_rm_risk_upper_bound(type, model, kurt_risk, rm.settings.ub)
    _set_risk_expression(model, kurt_risk, rm.settings.scale, rm.settings.flag)
    return nothing
end
function set_rm(port::OmniPortfolio, rms::AbstractVector{<:Kurt},
                type::Union{Trad, RP, NOC}; kwargs...)
    model = port.model
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
                             [kurt_risk[idx]; view(x_kurt, :, idx)] ∈ SecondOrderCone()
                             [i = 1:Nf], x_kurt[i, idx] == tr(Bi[i] * W)
                         end)
            _set_rm_risk_upper_bound(type, model, kurt_risk[idx], rm.settings.ub)
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
                        [kurt_risk[idx];
                         sqrt_sigma_4 * view(zkurt, :, idx)] ∈ SecondOrderCone())
            _set_rm_risk_upper_bound(type, model, kurt_risk[idx], rm.settings.ub)
            _set_risk_expression(model, kurt_risk[idx], rm.settings.scale, rm.settings.flag)
        end
    end

    return nothing
end
function set_rm(port::OmniPortfolio, rm::SKurt, type::Union{Trad, RP, NOC}; kwargs...)
    model = port.model
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
        @constraints(model, begin
                         [skurt_risk; x_skurt] ∈ SecondOrderCone()
                         [i = 1:Nf], x_skurt[i] == tr(Bi[i] * W)
                     end)
    else
        L_2 = port.L_2
        S_2 = port.S_2
        sqrt_sigma_4 = sqrt(S_2 * kt * transpose(S_2))
        @expression(model, zskurt, L_2 * vec(W))
        @constraint(model, [skurt_risk; sqrt_sigma_4 * zskurt] ∈ SecondOrderCone())
    end
    _set_rm_risk_upper_bound(type, model, skurt_risk, rm.settings.ub)
    _set_risk_expression(model, skurt_risk, rm.settings.scale, rm.settings.flag)
    return nothing
end
function set_rm(port::OmniPortfolio, rms::AbstractVector{<:SKurt},
                type::Union{Trad, RP, NOC}; kwargs...)
    model = port.model
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
                             [skurt_risk[idx]; view(x_skurt, :, idx)] ∈ SecondOrderCone()
                             [i = 1:Nf], x_skurt[i, idx] == tr(Bi[i] * W)
                         end)
            _set_rm_risk_upper_bound(type, model, skurt_risk[idx], rm.settings.ub)
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
                        [skurt_risk[idx];
                         sqrt_sigma_4 * view(zskurt, :, idx)] ∈ SecondOrderCone())
            _set_rm_risk_upper_bound(type, model, skurt_risk[idx], rm.settings.ub)
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

    get_net_portfolio_returns(model, returns)
    net_X = model[:net_X]
    T = size(returns, 1)
    @variable(model, owa[1:T])
    @constraint(model, net_X == owa)

    return nothing
end
function set_rm(port::OmniPortfolio, rm::GMD, type::Union{Trad, RP, NOC};
                returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
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
                    owa * transpose(gmd_w) .<=
                    ovec * transpose(gmda) + gmdb * transpose(ovec))
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
                         net_X .+ gmd_t .- gmd_nu .+ gmd_eta .-
                         vec(sum(gmd_epsilon; dims = 2)) .== 0
                         gmd_z .+ gmd_y .== vec(sum(gmd_psi; dims = 1))
                         [i = 1:M, j = 1:T],
                         [-gmd_z[i] * owa_p[i], gmd_psi[j, i] * owa_p[i] / (owa_p[i] - 1),
                          gmd_epsilon[j, i]] ∈ MOI.PowerCone(inv(owa_p[i]))
                     end)
    end
    _set_rm_risk_upper_bound(type, model, gmd_risk, rm.settings.ub)
    _set_risk_expression(model, gmd_risk, rm.settings.scale, rm.settings.flag)
    return nothing
end
function set_rm(port::OmniPortfolio, rm::TG, type::Union{Trad, RP, NOC};
                returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
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
                    owa * transpose(tg_w) .<= ovec * transpose(tga) + tgb * transpose(ovec))
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
                         net_X .+ tg_t .- tg_nu .+ tg_eta .-
                         vec(sum(tg_epsilon; dims = 2)) .== 0
                         tg_z .+ tg_y .== vec(sum(tg_psi; dims = 1))
                         [i = 1:M, j = 1:T],
                         [-tg_z[i] * owa_p[i], tg_psi[j, i] * owa_p[i] / (owa_p[i] - 1),
                          tg_epsilon[j, i]] ∈ MOI.PowerCone(inv(owa_p[i]))
                     end)
    end
    _set_rm_risk_upper_bound(type, model, tg_risk, rm.settings.ub)
    _set_risk_expression(model, tg_risk, rm.settings.scale, rm.settings.flag)

    return nothing
end
function set_rm(port::OmniPortfolio, rms::AbstractVector{<:TG}, type::Union{Trad, RP, NOC};
                returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
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
                        owa * transpose(tg_w) .<=
                        ovec * transpose(view(tga, :, idx)) +
                        view(tgb, :, idx) * transpose(ovec))
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
                             net_X .+ tg_t[idx] .- view(tg_nu, :, idx) .+
                             view(tg_eta, :, idx) .-
                             vec(sum(view(tg_epsilon, :, :, idx); dims = 2)) .== 0
                             tg_z[:, idx] .+ tg_y[:, idx] .==
                             vec(sum(view(tg_psi, :, :, idx); dims = 1))
                             [i = 1:M, j = 1:T],
                             [-tg_z[i, idx] * owa_p[i],
                              tg_psi[j, i, idx] * owa_p[i] / (owa_p[i] - 1),
                              tg_epsilon[j, i, idx]] ∈ MOI.PowerCone(inv(owa_p[i]))
                         end)
        end
        _set_rm_risk_upper_bound(type, model, tg_risk[idx], rm.settings.ub)
        _set_risk_expression(model, tg_risk[idx], rm.settings.scale, rm.settings.flag)
    end
    return nothing
end
function set_rm(port::OmniPortfolio, rm::TGRG, type::Union{Trad, RP, NOC};
                returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
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
                    owa * transpose(rtg_w) .<=
                    ovec * transpose(rtga) + rtgb * transpose(ovec))
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
                         net_X .+ rltg_t .- rltg_nu .+ rltg_eta .-
                         vec(sum(rltg_epsilon; dims = 2)) .== 0
                         rltg_z .+ rltg_y .== vec(sum(rltg_psi; dims = 1))
                         [i = 1:M, j = 1:T],
                         [-rltg_z[i] * owa_p[i], rltg_psi[j, i] * owa_p[i] / (owa_p[i] - 1),
                          rltg_epsilon[j, i]] ∈ MOI.PowerCone(inv(owa_p[i]))
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
                         -net_X .+ rhtg_t .- rhtg_nu .+ rhtg_eta .-
                         vec(sum(rhtg_epsilon; dims = 2)) .== 0
                         rhtg_z .+ rhtg_y .== vec(sum(rhtg_psi; dims = 1))
                         [i = 1:M, j = 1:T],
                         [-rhtg_z[i] * owa_p[i], rhtg_psi[j, i] * owa_p[i] / (owa_p[i] - 1),
                          rhtg_epsilon[j, i]] ∈ MOI.PowerCone(inv(owa_p[i]))
                     end)
    end
    _set_rm_risk_upper_bound(type, model, rtg_risk, rm.settings.ub)
    _set_risk_expression(model, rtg_risk, rm.settings.scale, rm.settings.flag)
    return nothing
end
function set_rm(port::OmniPortfolio, rms::AbstractVector{<:TGRG},
                type::Union{Trad, RP, NOC}; returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
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
                @variable(model, rtga[1:T, 1:count])
                @variable(model, rtgb[1:T, 1:count])
            end
            rtga = model[:rtga]
            rtgb = model[:rtgb]
            owa = model[:owa]
            add_to_expression!(rtg_risk[idx], sum(view(rtga, :, idx) .+ view(rtgb, :, idx)))
            rtg_w = owa_rtg(T; alpha_i = alpha_i, alpha = alpha, a_sim = a_sim,
                            beta_i = beta_i, beta = beta, b_sim = b_sim)
            @constraint(model,
                        owa * transpose(rtg_w) .<=
                        ovec * transpose(view(rtga, :, idx)) +
                        view(rtgb, :, idx) * transpose(ovec))
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
                @variable(model, rltg_t[1:count])
                @variable(model, rltg_nu[1:T, 1:count] .>= 0)
                @variable(model, rltg_eta[1:T, 1:count] .>= 0)
                @variable(model, rltg_epsilon[1:T, 1:M, 1:count])
                @variable(model, rltg_psi[1:T, 1:M, 1:count])
                @variable(model, rltg_z[1:M, 1:count])
                @variable(model, rltg_y[1:M, 1:count] .>= 0)
                @expression(model, rltg_risk[1:count], zero(AffExpr))
                @variable(model, rhtg_t[1:count])
                @variable(model, rhtg_nu[1:T, 1:count] .>= 0)
                @variable(model, rhtg_eta[1:T, 1:count] .>= 0)
                @variable(model, rhtg_epsilon[1:T, 1:M, 1:count])
                @variable(model, rhtg_psi[1:T, 1:M, 1:count])
                @variable(model, rhtg_z[1:M, 1:count])
                @variable(model, rhtg_y[1:M, 1:count] .>= 0)
                @expression(model, rhtg_risk[1:count], zero(AffExpr))
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
            @constraint(model,
                        net_X .+ rltg_t[idx] .- view(rltg_nu, :, idx) .+
                        view(rltg_eta, :, idx) .-
                        vec(sum(view(rltg_epsilon, :, :, idx); dims = 2)) .== 0)
            @constraint(model,
                        view(rltg_z, :, idx) .+ view(rltg_y, :, idx) .==
                        vec(sum(view(rltg_psi, :, :, idx); dims = 1)))
            @constraint(model, [i = 1:M, j = 1:T],
                        [-rltg_z[i, idx] * owa_p[i],
                         rltg_psi[j, i, idx] * owa_p[i] / (owa_p[i] - 1),
                         rltg_epsilon[j, i, idx]] ∈ MOI.PowerCone(inv(owa_p[i])))
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
            @constraint(model,
                        -net_X .+ rhtg_t[idx] .- view(rhtg_nu, :, idx) .+
                        view(rhtg_eta, :, idx) .-
                        vec(sum(rhtg_epsilon[:, :, idx]; dims = 2)) .== 0)
            @constraint(model,
                        view(rhtg_z, :, idx) .+ view(rhtg_y, :, idx) .==
                        vec(sum(view(rhtg_psi, :, :, idx); dims = 1)))
            @constraint(model, [i = 1:M, j = 1:T],
                        [-rhtg_z[i, idx] * owa_p[i],
                         rhtg_psi[j, i, idx] * owa_p[i] / (owa_p[i] - 1),
                         rhtg_epsilon[j, i, idx]] ∈ MOI.PowerCone(inv(owa_p[i])))
        end
        _set_rm_risk_upper_bound(type, model, rtg_risk[idx], rm.settings.ub)
        _set_risk_expression(model, rtg_risk[idx], rm.settings.scale, rm.settings.flag)
    end
    return nothing
end
function set_rm(port::OmniPortfolio, rm::OWA, type::Union{Trad, RP, NOC};
                returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
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
                    owa * transpose(owa_w) .<=
                    ovec * transpose(owa_a) + owa_b * transpose(ovec))
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
                         net_X .+ owa_t .- owa_nu .+ owa_eta .-
                         vec(sum(owa_epsilon; dims = 2)) .== 0
                         owa_z .+ owa_y .== vec(sum(owa_psi; dims = 1))
                         [i = 1:M, j = 1:T],
                         [-owa_z[i] * owa_p[i], owa_psi[j, i] * owa_p[i] / (owa_p[i] - 1),
                          owa_epsilon[j, i]] ∈ MOI.PowerCone(inv(owa_p[i]))
                     end)
    end
    _set_rm_risk_upper_bound(type, model, owa_risk, rm.settings.ub)
    _set_risk_expression(model, owa_risk, rm.settings.scale, rm.settings.flag)
    return nothing
end
function set_rm(port::OmniPortfolio, rms::AbstractVector{<:OWA}, type::Union{Trad, RP, NOC};
                returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
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
                        owa * transpose(owa_w) .<=
                        ovec * transpose(view(owa_a, :, idx)) +
                        view(owa_b, :, idx) * transpose(ovec))
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
                             net_X .+ owa_t[idx] .- view(owa_nu, :, idx) .+
                             view(owa_eta, :, idx) .-
                             vec(sum(view(owa_epsilon, :, :, idx); dims = 2)) .== 0
                             view(owa_z, :, idx) .+ view(owa_y, :, idx) .==
                             vec(sum(view(owa_psi, :, :, idx); dims = 1))
                             [i = 1:M, j = 1:T],
                             [-owa_z[i, idx] * owa_p[i],
                              owa_psi[j, i, idx] * owa_p[i] / (owa_p[i] - 1),
                              owa_epsilon[j, i, idx]] ∈ MOI.PowerCone(inv(owa_p[i]))
                         end)
        end
        _set_rm_risk_upper_bound(type, model, owa_risk[idx], rm.settings.ub)
        _set_risk_expression(model, owa_risk[idx], rm.settings.scale, rm.settings.flag)
    end
    return nothing
end
function _BDVariance_constraints(::BDVAbsVal, model, Dt, Dx, T)
    @constraint(model, [i = 1:T, j = i:T], [Dt[i, j]; Dx[i, j]] in MOI.NormOneCone(2))
    return nothing
end
function _BDVariance_constraints(::BDVIneq, model, Dt, Dx, T)
    @constraints(model, begin
                     [i = 1:T, j = i:T], Dt[i, j] .>= Dx[i, j]
                     [i = 1:T, j = i:T], Dt[i, j] .>= -Dx[i, j]
                 end)
    return nothing
end
function set_rm(port::OmniPortfolio, rm::BDVariance, type::Union{Trad, RP, NOC};
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
    _set_rm_risk_upper_bound(type, model, bd_variance_risk, rm.settings.ub)
    _set_risk_expression(model, bd_variance_risk, rm.settings.scale, rm.settings.flag)
    return nothing
end
function set_rm(port::OmniPortfolio, rm::Skew, type::Union{Trad, RP, NOC}; kwargs...)
    model = port.model
    w = model[:w]
    @variable(model, t_skew)
    V = if (isnothing(rm.V) || isempty(rm.V))
        port.V
    else
        rm.V
    end
    G = real(sqrt(V))
    @constraint(model, [t_skew; G * w] ∈ SecondOrderCone())
    @expression(model, skew_risk, t_skew^2)
    _set_rm_risk_upper_bound(type, model, t_skew, rm.settings.ub)
    _set_risk_expression(model, skew_risk, rm.settings.scale, rm.settings.flag)
    return nothing
end
function set_rm(port::OmniPortfolio, rms::AbstractVector{<:Skew},
                type::Union{Trad, RP, NOC}; kwargs...)
    model = port.model
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
        @constraint(model, [t_skew[idx]; G * w] ∈ SecondOrderCone())
        _set_rm_risk_upper_bound(type, model, t_skew[idx], rm.settings.ub)
        _set_risk_expression(model, skew_risk[idx], rm.settings.scale, rm.settings.flag)
    end
    return nothing
end
function set_rm(port::OmniPortfolio, rm::SSkew, type::Union{Trad, RP, NOC}; kwargs...)
    model = port.model
    w = model[:w]
    @variable(model, t_sskew)
    SV = if (isnothing(rm.V) || isempty(rm.V))
        port.SV
    else
        rm.V
    end
    G = real(sqrt(SV))
    @constraint(model, [t_sskew; G * w] ∈ SecondOrderCone())
    @expression(model, sskew_risk, t_sskew^2)
    _set_rm_risk_upper_bound(type, model, t_sskew, rm.settings.ub)
    _set_risk_expression(model, sskew_risk, rm.settings.scale, rm.settings.flag)
    return nothing
end
function set_rm(port::OmniPortfolio, rms::AbstractVector{<:SSkew},
                type::Union{Trad, RP, NOC}; kwargs...)
    model = port.model
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
        @constraint(model, [t_sskew[idx]; G * w] ∈ SecondOrderCone())
        _set_rm_risk_upper_bound(type, model, t_sskew[idx], rm.settings.ub)
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