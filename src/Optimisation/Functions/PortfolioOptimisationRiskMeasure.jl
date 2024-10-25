# Risk upper bounds
function _set_rm_risk_upper_bound(args...)
    return nothing
end
function _set_rm_risk_upper_bound(::Sharpe, ::Trad, model, rm_risk, ub)
    if isfinite(ub)
        k = model[:k]
        @constraint(model, rm_risk .<= ub * k)
    end
    return nothing
end
function _set_rm_risk_upper_bound(::Any, ::Trad, model, rm_risk, ub)
    if isfinite(ub)
        @constraint(model, rm_risk .<= ub)
    end
    return nothing
end
# SD risk upper bound (special case)
function _set_sd_risk_upper_bound(args...)
    return nothing
end
function _set_sd_risk_upper_bound(::SDP, ::Sharpe, ::Trad, model, ub)
    if isfinite(ub)
        sd_risk = model[:sd_risk]
        k = model[:k]
        @constraint(model, sd_risk .<= ub^2 * k)
    end
    return nothing
end
function _set_sd_risk_upper_bound(::SDP, ::Sharpe, ::Trad, model, ub, idx)
    if isfinite(ub)
        sd_risk = model[:sd_risk]
        k = model[:k]
        @constraint(model, sd_risk[idx] .<= ub^2 * k)
    end
    return nothing
end
function _set_sd_risk_upper_bound(::SDP, ::Any, ::Trad, model, ub)
    if isfinite(ub)
        sd_risk = model[:sd_risk]
        @constraint(model, sd_risk .<= ub^2)
    end
    return nothing
end
function _set_sd_risk_upper_bound(::SDP, ::Any, ::Trad, model, ub, idx)
    if isfinite(ub)
        sd_risk = model[:sd_risk]
        @constraint(model, sd_risk[idx] .<= ub^2)
    end
    return nothing
end
function _set_sd_risk_upper_bound(::Any, ::Sharpe, ::Trad, model, ub)
    if isfinite(ub)
        dev = model[:dev]
        k = model[:k]
        @constraint(model, dev .<= ub * k)
    end
    return nothing
end
function _set_sd_risk_upper_bound(::Any, ::Sharpe, ::Trad, model, ub, idx)
    if isfinite(ub)
        dev = model[:dev]
        k = model[:k]
        @constraint(model, dev[idx] .<= ub * k)
    end
    return nothing
end
function _set_sd_risk_upper_bound(::Any, ::Any, ::Trad, model, ub)
    if isfinite(ub)
        dev = model[:dev]
        @constraint(model, dev .<= ub)
    end
    return nothing
end
function _set_sd_risk_upper_bound(::Any, ::Any, ::Trad, model, ub, idx)
    if isfinite(ub)
        dev = model[:dev]
        @constraint(model, dev[idx] .<= ub)
    end
    return nothing
end
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
function _sd_risk(::SDP, ::Any, model, sigma)
    W = model[:W]
    @expression(model, sd_risk, tr(sigma * W))
    return nothing
end
function _sd_risk(::SDP, model, sigma, count::Integer)
    @expression(model, sd_risk[1:count], zero(AffExpr))
    return nothing
end
function _sd_risk(::SDP, ::Any, model, sigma, idx::Integer)
    sd_risk = model[:sd_risk]
    W = model[:W]
    add_to_expression!(sd_risk[idx], tr(sigma * W))
    return nothing
end
function _sd_risk(::Union{NoAdj, IP}, ::SOCSD, model::JuMP.Model, sigma::AbstractMatrix)
    G = sqrt(sigma)
    @variable(model, dev)
    @expression(model, sd_risk, dev^2)
    w = model[:w]
    @constraint(model, [dev; G * w] ∈ SecondOrderCone())
    return nothing
end
function _sd_risk(::Union{NoAdj, IP}, model::JuMP.Model, sigma::AbstractMatrix,
                  count::Integer)
    @variable(model, dev[1:count])
    @expression(model, sd_risk[1:count], zero(QuadExpr))
    return nothing
end
function _sd_risk(::Union{NoAdj, IP}, ::SOCSD, model::JuMP.Model, sigma::AbstractMatrix,
                  idx::Integer)
    G = sqrt(sigma)
    sd_risk = model[:sd_risk]
    dev = model[:dev]
    add_to_expression!(sd_risk[idx], dev[idx], dev[idx])
    w = model[:w]
    @constraint(model, [dev[idx]; G * w] ∈ SecondOrderCone())
    return nothing
end
function _sd_risk(::Union{NoAdj, IP}, ::QuadSD, model, sigma)
    G = sqrt(sigma)
    @variable(model, dev)
    w = model[:w]
    @expression(model, sd_risk, dot(w, sigma, w))
    @constraint(model, [dev; G * w] ∈ SecondOrderCone())
    return nothing
end
function _sd_risk(::Union{NoAdj, IP}, ::QuadSD, model::JuMP.Model, sigma::AbstractMatrix,
                  idx::Integer)
    G = sqrt(sigma)
    sd_risk = model[:sd_risk]
    w = model[:w]
    add_to_expression!(sd_risk[idx], dot(w, sigma, w))
    dev = model[:dev]
    @constraint(model, [dev[idx]; G * w] ∈ SecondOrderCone())
    return nothing
end
function _sd_risk(::Union{NoAdj, IP}, ::SimpleSD, model::JuMP.Model, sigma::AbstractMatrix)
    G = sqrt(sigma)
    @variable(model, dev)
    @expression(model, sd_risk, dev)
    w = model[:w]
    @constraint(model, [dev; G * w] ∈ SecondOrderCone())
    return nothing
end
function _sd_risk(::Union{NoAdj, IP}, ::SimpleSD, model, sigma, idx::Integer)
    G = sqrt(sigma)
    sd_risk = model[:sd_risk]
    dev = model[:dev]
    add_to_expression!(sd_risk[idx], dev[idx])
    w = model[:w]
    @constraint(model, [dev[idx]; G * w] ∈ SecondOrderCone())
    return nothing
end
function _get_ntwk_clust_method(::Trad, port)
    return if isa(port.network_adj, SDP) || isa(port.cluster_adj, SDP)
        SDP()
    else
        NoAdj()
    end
end
function _get_ntwk_clust_method(args...)
    return NoAdj()
end
"""
    set_rm(port::Portfolio, rm::SD, type::Union{Trad, RP}, obj;
                sigma::AbstractMatrix{<:Real},
                kelly_approx_idx::Union{AbstractVector{<:Integer}, Nothing}, kwargs...)
"""
function set_rm(port::Portfolio, rm::SD, type::Union{Trad, RP}, obj;
                sigma::AbstractMatrix{<:Real},
                kelly_approx_idx::Union{AbstractVector{<:Integer}, Nothing}, kwargs...)
    use_portfolio_sigma = (isnothing(rm.sigma) || isempty(rm.sigma))
    if !isnothing(kelly_approx_idx) && use_portfolio_sigma
        if isempty(kelly_approx_idx)
            push!(kelly_approx_idx, 0)
        end
    end
    if !use_portfolio_sigma
        sigma = rm.sigma
    end
    model = port.model

    adjacency_constraint = _get_ntwk_clust_method(type, port)
    _sdp(adjacency_constraint, port, obj)
    _sd_risk(adjacency_constraint, rm.formulation, model, sigma)
    _set_sd_risk_upper_bound(adjacency_constraint, obj, type, model, rm.settings.ub)
    sd_risk = model[:sd_risk]
    _set_risk_expression(model, sd_risk, rm.settings.scale, rm.settings.flag)
    return nothing
end
function set_rm(port::Portfolio, rms::AbstractVector{<:SD}, type::Union{Trad, RP}, obj;
                sigma::AbstractMatrix{<:Real},
                kelly_approx_idx::Union{AbstractVector{<:Integer}, Nothing}, kwargs...)
    model = port.model

    adjacency_constraint = _get_ntwk_clust_method(type, port)
    _sdp(adjacency_constraint, port, obj)
    count = length(rms)
    _sd_risk(adjacency_constraint, model, sigma, count)
    sd_risk = model[:sd_risk]
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
        _sd_risk(adjacency_constraint, rm.formulation, model, sigma, i)
        _set_sd_risk_upper_bound(adjacency_constraint, obj, type, model, rm.settings.ub, i)
        _set_risk_expression(model, sd_risk[i], rm.settings.scale, rm.settings.flag)
    end
    return nothing
end
function set_rm(port::Portfolio, rm::MAD, type::Union{Trad, RP}, obj;
                mu::AbstractVector{<:Real}, returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
    T = size(returns, 1)
    if !(isnothing(rm.mu) || isempty(rm.mu))
        mu = rm.mu
    end
    abs_dev = returns .- transpose(mu)
    @variable(model, mad[1:T] >= 0)
    @expression(model, mad_risk, sum(mad) / T)
    w = model[:w]
    @constraint(model, abs_dev * w .>= -mad)
    _set_rm_risk_upper_bound(obj, type, model, mad_risk, 0.5 * rm.settings.ub)
    _set_risk_expression(model, mad_risk, rm.settings.scale, rm.settings.flag)
    return nothing
end
function set_rm(port::Portfolio, rms::AbstractVector{<:MAD}, type::Union{Trad, RP}, obj;
                mu::AbstractVector{<:Real}, returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
    T = size(returns, 1)
    count = length(rms)
    @variable(model, mad[1:T, 1:count] >= 0)
    @expression(model, mad_risk[1:count], zero(AffExpr))
    w = model[:w]
    for (i, rm) ∈ pairs(rms)
        if !(isnothing(rm.mu) || isempty(rm.mu))
            mu = rm.mu
        end
        abs_dev = returns .- transpose(mu)
        add_to_expression!(mad_risk[i], inv(T), sum(view(mad, :, i)))
        @constraint(model, abs_dev * w .>= -view(mad, :, i))
        _set_rm_risk_upper_bound(obj, type, model, mad_risk[i], 0.5 * rm.settings.ub)
        _set_risk_expression(model, mad_risk[i], rm.settings.scale, rm.settings.flag)
    end
    return nothing
end
function set_rm(port::Portfolio, rm::SSD, type::Union{Trad, RP}, obj;
                mu::AbstractVector{<:Real}, returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
    T = size(returns, 1)
    if !(isnothing(rm.mu) || isempty(rm.mu))
        mu = rm.mu
    end
    abs_dev = returns .- transpose(mu)
    target = rm.target
    @variable(model, ssd[1:T] .>= target)
    @variable(model, sdev)
    @expression(model, sdev_risk, sdev / sqrt(T - 1))
    w = model[:w]
    @constraint(model, abs_dev * w .>= -ssd)
    @constraint(model, [sdev; ssd] ∈ SecondOrderCone())
    _set_rm_risk_upper_bound(obj, type, model, sdev_risk, rm.settings.ub)
    _set_risk_expression(model, sdev_risk, rm.settings.scale, rm.settings.flag)

    return nothing
end
function set_rm(port::Portfolio, rms::AbstractVector{<:SSD}, type::Union{Trad, RP}, obj;
                mu::AbstractVector{<:Real}, returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
    T = size(returns, 1)
    count = length(rms)
    @variable(model, ssd[1:T, 1:count])
    @variable(model, sdev[1:count])
    @expression(model, sdev_risk[1:count], zero(AffExpr))
    w = model[:w]
    for (i, rm) ∈ pairs(rms)
        if !(isnothing(rm.mu) || isempty(rm.mu))
            mu = rm.mu
        end
        abs_dev = returns .- transpose(mu)
        target = rm.target
        @constraint(model, view(ssd, :, i) .>= target)
        add_to_expression!(sdev_risk[i], inv(sqrt(T - 1)), sdev[i])
        @constraint(model, abs_dev * w .>= -view(ssd, :, i))
        @constraint(model, [sdev[i]; view(ssd, :, i)] ∈ SecondOrderCone())
        _set_rm_risk_upper_bound(obj, type, model, sdev_risk[i], rm.settings.ub)
        _set_risk_expression(model, sdev_risk[i], rm.settings.scale, rm.settings.flag)
    end
    return nothing
end
function _lpm_risk(::RP, ::Any, model, lpm, target)
    k = model[:k]
    X = model[:X]
    @constraint(model, lpm .>= target * k .- X)
    return nothing
end
function _lpm_risk(::Trad, ::Sharpe, model, lpm, target)
    k = model[:k]
    X = model[:X]
    @constraint(model, lpm .>= target * k .- X)
    return nothing
end
function _lpm_risk(::Any, ::Any, model, lpm, target)
    X = model[:X]
    @constraint(model, lpm .>= target .- X)
    return nothing
end
function set_rm(port::Portfolio, rm::FLPM, type::Union{Trad, RP}, obj;
                returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
    T = size(returns, 1)
    if !haskey(model, :X)
        w = model[:w]
        @expression(model, X, returns * w)
    end
    @variable(model, flpm[1:T] .>= 0)
    @expression(model, flpm_risk, sum(flpm) / T)
    _lpm_risk(type, obj, model, flpm, rm.target)
    _set_rm_risk_upper_bound(obj, type, model, flpm_risk, rm.settings.ub)
    _set_risk_expression(model, flpm_risk, rm.settings.scale, rm.settings.flag)
    return nothing
end
function set_rm(port::Portfolio, rms::AbstractVector{<:FLPM}, type::Union{Trad, RP}, obj;
                returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
    if !haskey(model, :X)
        w = model[:w]
        @expression(model, X, returns * w)
    end
    T = size(returns, 1)
    count = length(rms)
    @variable(model, flpm[1:T, 1:count] .>= 0)
    @expression(model, flpm_risk[1:count], zero(AffExpr))
    for (i, rm) ∈ pairs(rms)
        add_to_expression!(flpm_risk[i], inv(T), sum(view(flpm, :, i)))
        _lpm_risk(type, obj, model, view(flpm, :, i), rm.target)
        _set_rm_risk_upper_bound(obj, type, model, flpm_risk[i], rm.settings.ub)
        _set_risk_expression(model, flpm_risk[i], rm.settings.scale, rm.settings.flag)
    end
    return nothing
end
function set_rm(port::Portfolio, rm::SLPM, type::Union{Trad, RP}, obj;
                returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
    if !haskey(model, :X)
        w = model[:w]
        @expression(model, X, returns * w)
    end
    T = size(returns, 1)
    @variable(model, slpm[1:T] .>= 0)
    @variable(model, tslpm)
    @expression(model, slpm_risk, tslpm / sqrt(T - 1))
    @constraint(model, [tslpm; slpm] ∈ SecondOrderCone())
    _lpm_risk(type, obj, model, slpm, rm.target)
    _set_rm_risk_upper_bound(obj, type, model, slpm_risk, rm.settings.ub)
    _set_risk_expression(model, slpm_risk, rm.settings.scale, rm.settings.flag)
    return nothing
end
function set_rm(port::Portfolio, rms::AbstractVector{<:SLPM}, type::Union{Trad, RP}, obj;
                returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
    if !haskey(model, :X)
        w = model[:w]
        @expression(model, X, returns * w)
    end
    T = size(returns, 1)
    count = length(rms)
    @variable(model, slpm[1:T, 1:count] .>= 0)
    @variable(model, tslpm[1:count])
    @expression(model, slpm_risk[1:count], zero(AffExpr))
    for (i, rm) ∈ pairs(rms)
        add_to_expression!(slpm_risk[i], inv(sqrt(T - 1)), tslpm[i])
        @constraint(model, [tslpm[i]; view(slpm, :, i)] ∈ SecondOrderCone())
        _lpm_risk(type, obj, model, view(slpm, :, i), rm.target)
        _set_rm_risk_upper_bound(obj, type, model, slpm_risk[i], rm.settings.ub)
        _set_risk_expression(model, slpm_risk[i], rm.settings.scale, rm.settings.flag)
    end
    return nothing
end
function _wr_setup(model, returns)
    if !haskey(model, :X)
        w = model[:w]
        @expression(model, X, returns * w)
    end
    if !haskey(model, :wr)
        @variable(model, wr)
        @expression(model, wr_risk, wr)
        X = model[:X]
        @constraint(model, -X .<= wr)
    end
end
function set_rm(port::Portfolio, rm::WR, type::Union{Trad, RP}, obj;
                returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
    _wr_setup(model, returns)
    X = model[:X]
    _set_rm_risk_upper_bound(obj, type, model, -X, rm.settings.ub)
    wr_risk = model[:wr_risk]
    _set_risk_expression(model, wr_risk, rm.settings.scale, rm.settings.flag)
    return nothing
end
function set_rm(port::Portfolio, rm::RG, type::Union{Trad, RP}, obj;
                returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
    _wr_setup(model, returns)
    @variable(model, br)
    wr_risk = model[:wr_risk]
    @expression(model, rg_risk, wr_risk - br)
    X = model[:X]
    @constraint(model, -X .>= br)
    _set_rm_risk_upper_bound(obj, type, model, rg_risk, rm.settings.ub)
    _set_risk_expression(model, rg_risk, rm.settings.scale, rm.settings.flag)
    return nothing
end
function set_rm(port::Portfolio, rm::CVaR, type::Union{Trad, RP}, obj;
                returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
    T = size(returns, 1)
    if !haskey(model, :X)
        w = model[:w]
        @expression(model, X, returns * w)
    end
    iat = inv(rm.alpha * T)
    @variable(model, var)
    @variable(model, z_var[1:T] .>= 0)
    @expression(model, cvar_risk, var + sum(z_var) * iat)
    X = model[:X]
    @constraint(model, z_var .>= -X .- var)
    _set_rm_risk_upper_bound(obj, type, model, cvar_risk, rm.settings.ub)
    _set_risk_expression(model, cvar_risk, rm.settings.scale, rm.settings.flag)
    return nothing
end
function set_rm(port::Portfolio, rms::AbstractVector{<:CVaR}, type::Union{Trad, RP}, obj;
                returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model

    if !haskey(model, :X)
        w = model[:w]
        @expression(model, X, returns * w)
    end
    T = size(returns, 1)
    count = length(rms)
    @variable(model, var[1:count])
    @variable(model, z_var[1:T, 1:count] .>= 0)
    @expression(model, cvar_risk[1:count], zero(AffExpr))
    X = model[:X]
    for (i, rm) ∈ pairs(rms)
        iat = inv(rm.alpha * T)
        add_to_expression!(cvar_risk[i], var[i])
        add_to_expression!(cvar_risk[i], iat, sum(view(z_var, :, i)))
        @constraint(model, view(z_var, :, i) .>= -X .- var[i])
        _set_rm_risk_upper_bound(obj, type, model, cvar_risk[i], rm.settings.ub)
        _set_risk_expression(model, cvar_risk[i], rm.settings.scale, rm.settings.flag)
    end
    return nothing
end
function set_rm(port::Portfolio, rm::CVaRRG, type::Union{Trad, RP}, obj;
                returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
    if !haskey(model, :X)
        w = model[:w]
        @expression(model, X, returns * w)
    end
    T = size(returns, 1)
    iat = inv(rm.alpha * T)
    ibt = inv(rm.beta * T)
    @variable(model, var_l)
    @variable(model, z_var_l[1:T] .>= 0)
    @expression(model, cvar_risk_l, var_l + sum(z_var_l) * iat)
    X = model[:X]
    @constraint(model, z_var_l .>= -X .- var_l)
    @variable(model, var_h)
    @variable(model, z_var_h[1:T] .<= 0)
    @expression(model, cvar_risk_h, var_h + sum(z_var_h) * ibt)
    @constraint(model, z_var_h .<= -X .- var_h)
    @expression(model, rcvar_risk, cvar_risk_l - cvar_risk_h)
    _set_rm_risk_upper_bound(obj, type, model, rcvar_risk, rm.settings.ub)
    _set_risk_expression(model, rcvar_risk, rm.settings.scale, rm.settings.flag)
    return nothing
end
function set_rm(port::Portfolio, rms::AbstractVector{<:CVaRRG}, type::Union{Trad, RP}, obj;
                returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model

    if !haskey(model, :X)
        w = model[:w]
        @expression(model, X, returns * model[:w])
    end
    T = size(returns, 1)
    count = length(rms)
    @variable(model, var_l[1:count])
    @variable(model, z_var_l[1:T, 1:count] .>= 0)
    @expression(model, cvar_risk_l[1:count], zero(AffExpr))
    @variable(model, var_h[1:count])
    @variable(model, z_var_h[1:T, 1:count] .<= 0)
    @expression(model, cvar_risk_h[1:count], zero(AffExpr))
    @expression(model, rcvar_risk[1:count], zero(AffExpr))
    X = model[:X]
    for (i, rm) ∈ pairs(rms)
        iat = inv(rm.alpha * T)
        ibt = inv(rm.beta * T)
        add_to_expression!(cvar_risk_l[i], var_l[i])
        add_to_expression!(cvar_risk_l[i], iat, sum(view(z_var_l, :, i)))
        @constraint(model, view(z_var_l, :, i) .>= -X .- var_l[i])
        add_to_expression!(cvar_risk_h[i], var_h[i])
        add_to_expression!(cvar_risk_h[i], ibt, sum(view(z_var_h, :, i)))
        @constraint(model, view(z_var_h, :, i) .<= -X .- var_h[i])
        add_to_expression!(rcvar_risk[i], cvar_risk_l[i])
        add_to_expression!(rcvar_risk[i], -1, cvar_risk_h[i])
        _set_rm_risk_upper_bound(obj, type, model, rcvar_risk[i], rm.settings.ub)
        _set_risk_expression(model, rcvar_risk[i], rm.settings.scale, rm.settings.flag)
    end
    return nothing
end
function set_rm(port::Portfolio, rm::EVaR, type::Union{Trad, RP}, obj;
                returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
    if !haskey(model, :X)
        w = model[:w]
        @expression(model, X, returns * w)
    end
    T = size(returns, 1)
    at = rm.alpha * T
    @variable(model, t_evar)
    @variable(model, z_evar >= 0)
    @variable(model, u_evar[1:T])
    @expression(model, evar_risk, t_evar - z_evar * log(at))
    @constraint(model, sum(u_evar) <= z_evar)
    X = model[:X]
    @constraint(model, [i = 1:T],
                [-X[i] - t_evar, z_evar, u_evar[i]] ∈ MOI.ExponentialCone())
    _set_rm_risk_upper_bound(obj, type, model, evar_risk, rm.settings.ub)
    _set_risk_expression(model, evar_risk, rm.settings.scale, rm.settings.flag)

    return nothing
end
function set_rm(port::Portfolio, rms::AbstractVector{<:EVaR}, type::Union{Trad, RP}, obj;
                returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
    if !haskey(model, :X)
        w = model[:w]
        @expression(model, X, returns * w)
    end
    T = size(returns, 1)
    count = length(rms)
    @variable(model, t_evar[1:count])
    @variable(model, z_evar[1:count] >= 0)
    @variable(model, u_evar[1:T, 1:count])
    @expression(model, evar_risk[1:count], zero(AffExpr))
    X = model[:X]
    for (j, rm) ∈ pairs(rms)
        at = rm.alpha * T
        add_to_expression!(evar_risk[j], t_evar[j])
        add_to_expression!(evar_risk[j], -log(at), z_evar[j])
        @constraint(model, sum(view(u_evar, :, j)) <= z_evar[j])
        @constraint(model, [i = 1:T],
                    [-X[i] - t_evar[j], z_evar[j], u_evar[i, j]] ∈ MOI.ExponentialCone())
        _set_rm_risk_upper_bound(obj, type, model, evar_risk[j], rm.settings.ub)
        _set_risk_expression(model, evar_risk[j], rm.settings.scale, rm.settings.flag)
    end
    return nothing
end
function set_rm(port::Portfolio, rm::RLVaR, type::Union{Trad, RP}, obj;
                returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
    if !haskey(model, :X)
        w = model[:w]
        @expression(model, X, returns * model[:w])
    end
    T = size(returns, 1)
    iat = inv(rm.alpha * T)
    lnk = (iat^rm.kappa - iat^(-rm.kappa)) / (2 * rm.kappa)
    opk = one(rm.kappa) + rm.kappa
    omk = one(rm.kappa) - rm.kappa
    ik2 = inv(2 * rm.kappa)
    ik = inv(rm.kappa)
    iopk = inv(opk)
    iomk = inv(omk)
    @variable(model, t_rvar)
    @variable(model, z_rvar >= 0)
    @variable(model, omega_rvar[1:T])
    @variable(model, psi_rvar[1:T])
    @variable(model, theta_rvar[1:T])
    @variable(model, epsilon_rvar[1:T])
    @expression(model, rvar_risk, t_rvar + lnk * z_rvar + sum(psi_rvar .+ theta_rvar))
    @constraint(model, [i = 1:T],
                [z_rvar * opk * ik2, psi_rvar[i] * opk * ik, epsilon_rvar[i]] ∈
                MOI.PowerCone(iopk))
    @constraint(model, [i = 1:T],
                [omega_rvar[i] * iomk, theta_rvar[i] * ik, -z_rvar * ik2] ∈
                MOI.PowerCone(omk))
    X = model[:X]
    @constraint(model, -X .- t_rvar .+ epsilon_rvar .+ omega_rvar .<= 0)
    _set_rm_risk_upper_bound(obj, type, model, rvar_risk, rm.settings.ub)
    _set_risk_expression(model, rvar_risk, rm.settings.scale, rm.settings.flag)
    return nothing
end
function set_rm(port::Portfolio, rms::AbstractVector{<:RLVaR}, type::Union{Trad, RP}, obj;
                returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
    if !haskey(model, :X)
        w = model[:w]
        @expression(model, X, returns * w)
    end
    T = size(returns, 1)
    count = length(rms)
    @variable(model, t_rvar[1:count])
    @variable(model, z_rvar[1:count] >= 0)
    @variable(model, omega_rvar[1:T, 1:count])
    @variable(model, psi_rvar[1:T, 1:count])
    @variable(model, theta_rvar[1:T, 1:count])
    @variable(model, epsilon_rvar[1:T, 1:count])
    @expression(model, rvar_risk[1:count], zero(AffExpr))
    X = model[:X]
    for (j, rm) ∈ pairs(rms)
        iat = inv(rm.alpha * T)
        lnk = (iat^rm.kappa - iat^(-rm.kappa)) / (2 * rm.kappa)
        opk = one(rm.kappa) + rm.kappa
        omk = one(rm.kappa) - rm.kappa
        ik2 = inv(2 * rm.kappa)
        ik = inv(rm.kappa)
        iopk = inv(opk)
        iomk = inv(omk)
        add_to_expression!(rvar_risk[j], t_rvar[j])
        add_to_expression!(rvar_risk[j], lnk, z_rvar[j])
        add_to_expression!(rvar_risk[j],
                           sum(view(psi_rvar, :, j) .+ view(theta_rvar, :, j)))
        @constraint(model, [i = 1:T],
                    [z_rvar[j] * opk * ik2, psi_rvar[i, j] * opk * ik, epsilon_rvar[i, j]] ∈
                    MOI.PowerCone(iopk))
        @constraint(model, [i = 1:T],
                    [omega_rvar[i, j] * iomk, theta_rvar[i, j] * ik, -z_rvar[j] * ik2] ∈
                    MOI.PowerCone(omk))
        @constraint(model,
                    -X .- t_rvar[j] .+ view(epsilon_rvar, :, j) .+
                    view(omega_rvar, :, j) .<= 0)
        _set_rm_risk_upper_bound(obj, type, model, rvar_risk[j], rm.settings.ub)
        _set_risk_expression(model, rvar_risk[j], rm.settings.scale, rm.settings.flag)
    end
    return nothing
end
function _DaR_setup(model, returns)
    if !haskey(model, :X)
        w = model[:w]
        @expression(model, X, returns * w)
    end
    if !haskey(model, :dd)
        T = size(returns, 1)
        @variable(model, dd[1:(T + 1)])
        X = model[:X]
        @constraint(model, view(dd, 2:(T + 1)) .>= view(dd, 1:T) .- X)
        @constraint(model, view(dd, 2:(T + 1)) .>= 0)
        @constraint(model, dd[1] == 0)
    end
    return nothing
end
function set_rm(port::Portfolio, rm::MDD, type::Union{Trad, RP}, obj;
                returns::AbstractMatrix{<:Real}, kargs...)
    model = port.model
    T = size(returns, 1)
    _DaR_setup(model, returns)
    @variable(model, mdd)
    @expression(model, mdd_risk, mdd)
    dd = model[:dd]
    @constraint(model, mdd .>= view(dd, 2:(T + 1)))
    _set_rm_risk_upper_bound(obj, type, model, view(dd, 2:(T + 1)), rm.settings.ub)
    _set_risk_expression(model, mdd_risk, rm.settings.scale, rm.settings.flag)
    return nothing
end
function set_rm(port::Portfolio, rm::ADD, type::Union{Trad, RP}, obj;
                returns::AbstractMatrix{<:Real}, kargs...)
    model = port.model
    T = size(returns, 1)
    _DaR_setup(model, returns)
    dd = model[:dd]
    @expression(model, add_risk, sum(view(dd, 2:(T + 1))) / T)
    _set_rm_risk_upper_bound(obj, type, model, add_risk, rm.settings.ub)
    _set_risk_expression(model, add_risk, rm.settings.scale, rm.settings.flag)
    return nothing
end
function set_rm(port::Portfolio, rm::UCI, type::Union{Trad, RP}, obj;
                returns::AbstractMatrix{<:Real}, kargs...)
    model = port.model
    T = size(returns, 1)
    _DaR_setup(model, returns)
    @variable(model, uci)
    @expression(model, uci_risk, uci / sqrt(T))
    dd = model[:dd]
    @constraint(model, [uci; view(dd, 2:(T + 1))] ∈ SecondOrderCone())
    _set_rm_risk_upper_bound(obj, type, model, uci_risk, rm.settings.ub)
    _set_risk_expression(model, uci_risk, rm.settings.scale, rm.settings.flag)
    return nothing
end
function set_rm(port::Portfolio, rm::CDaR, type::Union{Trad, RP}, obj;
                returns::AbstractMatrix{<:Real}, kargs...)
    model = port.model
    _DaR_setup(model, returns)
    T = size(returns, 1)
    iat = inv(rm.alpha * T)
    @variable(model, dar)
    @variable(model, z_cdar[1:T] .>= 0)
    @expression(model, cdar_risk, dar + sum(z_cdar) * iat)
    dd = model[:dd]
    @constraint(model, z_cdar .>= view(dd, 2:(T + 1)) .- dar)
    _set_rm_risk_upper_bound(obj, type, model, cdar_risk, rm.settings.ub)
    _set_risk_expression(model, cdar_risk, rm.settings.scale, rm.settings.flag)
    return nothing
end
function set_rm(port::Portfolio, rms::AbstractVector{<:CDaR}, type::Union{Trad, RP}, obj;
                returns::AbstractMatrix{<:Real}, kargs...)
    model = port.model
    _DaR_setup(model, returns)
    T = size(returns, 1)
    count = length(rms)
    @variable(model, dar[1:count])
    @variable(model, z_cdar[1:T, 1:count] .>= 0)
    @expression(model, cdar_risk[1:count], zero(AffExpr))
    dd = model[:dd]
    for (i, rm) ∈ pairs(rms)
        iat = inv(rm.alpha * T)
        add_to_expression!(cdar_risk[i], dar[i])
        add_to_expression!(cdar_risk[i], iat, sum(view(z_cdar, :, i)))
        @constraint(model, view(z_cdar, :, i) .>= view(dd, 2:(T + 1)) .- dar[i])
        _set_rm_risk_upper_bound(obj, type, model, cdar_risk[i], rm.settings.ub)
        _set_risk_expression(model, cdar_risk[i], rm.settings.scale, rm.settings.flag)
    end
    return nothing
end
function set_rm(port::Portfolio, rm::EDaR, type::Union{Trad, RP}, obj;
                returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
    T = size(returns, 1)
    _DaR_setup(model, returns)
    at = rm.alpha * T
    @variable(model, t_edar)
    @variable(model, z_edar >= 0)
    @variable(model, u_edar[1:T])
    @expression(model, edar_risk, t_edar - z_edar * log(at))
    @constraint(model, sum(u_edar) <= z_edar)
    dd = model[:dd]
    @constraint(model, [i = 1:T],
                [dd[i + 1] - t_edar, z_edar, u_edar[i]] ∈ MOI.ExponentialCone())
    _set_rm_risk_upper_bound(obj, type, model, edar_risk, rm.settings.ub)
    _set_risk_expression(model, edar_risk, rm.settings.scale, rm.settings.flag)
    return nothing
end
function set_rm(port::Portfolio, rms::AbstractVector{<:EDaR}, type::Union{Trad, RP}, obj;
                returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
    _DaR_setup(model, returns)
    T = size(returns, 1)
    count = length(rms)
    @variable(model, t_edar[1:count])
    @variable(model, z_edar[1:count] >= 0)
    @variable(model, u_edar[1:T, 1:count])
    @expression(model, edar_risk[1:count], zero(AffExpr))
    dd = model[:dd]
    for (j, rm) ∈ pairs(rms)
        at = rm.alpha * T
        add_to_expression!(edar_risk[j], t_edar[j])
        add_to_expression!(edar_risk[j], -log(at), z_edar[j])
        @constraint(model, sum(view(u_edar, :, j)) <= z_edar[j])
        @constraint(model, [i = 1:T],
                    [dd[i + 1] - t_edar[j], z_edar[j], u_edar[i, j]] ∈
                    MOI.ExponentialCone())
        _set_rm_risk_upper_bound(obj, type, model, edar_risk[j], rm.settings.ub)
        _set_risk_expression(model, edar_risk[j], rm.settings.scale, rm.settings.flag)
    end
    return nothing
end
function set_rm(port::Portfolio, rm::RLDaR, type::Union{Trad, RP}, obj;
                returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
    _DaR_setup(model, returns)
    T = size(returns, 1)
    iat = inv(rm.alpha * T)
    lnk = (iat^rm.kappa - iat^(-rm.kappa)) / (2 * rm.kappa)
    opk = one(rm.kappa) + rm.kappa
    omk = one(rm.kappa) - rm.kappa
    ik2 = inv(2 * rm.kappa)
    ik = inv(rm.kappa)
    iopk = inv(opk)
    iomk = inv(omk)
    @variable(model, t_rdar)
    @variable(model, z_rdar >= 0)
    @variable(model, omega_rdar[1:T])
    @variable(model, psi_rdar[1:T])
    @variable(model, theta_rdar[1:T])
    @variable(model, epsilon_rdar[1:T])
    @expression(model, rdar_risk, t_rdar + lnk * z_rdar + sum(psi_rdar .+ theta_rdar))
    @constraint(model, [i = 1:T],
                [z_rdar * opk * ik2, psi_rdar[i] * opk * ik, epsilon_rdar[i]] ∈
                MOI.PowerCone(iopk))
    @constraint(model, [i = 1:T],
                [omega_rdar[i] * iomk, theta_rdar[i] * ik, -z_rdar * ik2] ∈
                MOI.PowerCone(omk))
    dd = model[:dd]
    @constraint(model, view(dd, 2:(T + 1)) .- t_rdar .+ epsilon_rdar .+ omega_rdar .<= 0)
    _set_rm_risk_upper_bound(obj, type, model, rdar_risk, rm.settings.ub)
    _set_risk_expression(model, rdar_risk, rm.settings.scale, rm.settings.flag)
    return nothing
end
function set_rm(port::Portfolio, rms::AbstractVector{<:RLDaR}, type::Union{Trad, RP}, obj;
                returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
    _DaR_setup(model, returns)
    T = size(returns, 1)
    count = length(rms)
    @variable(model, t_rdar[1:count])
    @variable(model, z_rdar[1:count] >= 0)
    @variable(model, omega_rdar[1:T, 1:count])
    @variable(model, psi_rdar[1:T, 1:count])
    @variable(model, theta_rdar[1:T, 1:count])
    @variable(model, epsilon_rdar[1:T, 1:count])
    @expression(model, rdar_risk[1:count], zero(AffExpr))
    dd = model[:dd]
    for (j, rm) ∈ pairs(rms)
        iat = inv(rm.alpha * T)
        lnk = (iat^rm.kappa - iat^(-rm.kappa)) / (2 * rm.kappa)
        opk = one(rm.kappa) + rm.kappa
        omk = one(rm.kappa) - rm.kappa
        ik2 = inv(2 * rm.kappa)
        ik = inv(rm.kappa)
        iopk = inv(opk)
        iomk = inv(omk)
        add_to_expression!(rdar_risk[j], t_rdar[j])
        add_to_expression!(rdar_risk[j], lnk, z_rdar[j])
        add_to_expression!(rdar_risk[j],
                           sum(view(psi_rdar, :, j) .+ view(theta_rdar, :, j)))
        @constraint(model, [i = 1:T],
                    [z_rdar[j] * opk * ik2, psi_rdar[i, j] * opk * ik, epsilon_rdar[i, j]] ∈
                    MOI.PowerCone(iopk))
        @constraint(model, [i = 1:T],
                    [omega_rdar[i, j] * iomk, theta_rdar[i, j] * ik, -z_rdar[j] * ik2] ∈
                    MOI.PowerCone(omk))
        @constraint(model,
                    view(dd, 2:(T + 1)) .- t_rdar[j] .+ view(epsilon_rdar, :, j) .+
                    view(omega_rdar, :, j) .<= 0)
        _set_rm_risk_upper_bound(obj, type, model, rdar_risk[j], rm.settings.ub)
        _set_risk_expression(model, rdar_risk[j], rm.settings.scale, rm.settings.flag)
    end
    return nothing
end
function set_rm(port::Portfolio, rm::Kurt, type::Union{Trad, RP}, obj; kwargs...)
    _sdp(port, obj)
    model = port.model
    N = size(port.returns, 2)
    @variable(model, kurt_risk)
    kt = if (isnothing(rm.kt) || isempty(rm.kt))
        port.kurt
    else
        rm.kt
    end
    W = model[:W]
    if !iszero(port.max_num_assets_kurt) && N > port.max_num_assets_kurt
        f = port.max_num_assets_kurt_scale
        Nf = f * N
        @variable(model, x_kurt[1:Nf])
        @constraint(model, [kurt_risk; x_kurt] ∈ SecondOrderCone())
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
        @constraint(model, [i = 1:Nf], x_kurt[i] == tr(Bi[i] * W))
    else
        L_2 = port.L_2
        S_2 = port.S_2
        sqrt_sigma_4 = sqrt(S_2 * kt * transpose(S_2))
        @expression(model, zkurt, L_2 * vec(W))
        @constraint(model, [kurt_risk; sqrt_sigma_4 * zkurt] ∈ SecondOrderCone())
    end
    _set_rm_risk_upper_bound(obj, type, model, kurt_risk, rm.settings.ub)
    _set_risk_expression(model, kurt_risk, rm.settings.scale, rm.settings.flag)
    return nothing
end
function set_rm(port::Portfolio, rms::AbstractVector{<:Kurt}, type::Union{Trad, RP}, obj;
                kwargs...)
    _sdp(port, obj)
    model = port.model
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
            @constraint(model, [kurt_risk[idx]; view(x_kurt, :, idx)] ∈ SecondOrderCone())
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
            @constraint(model, [i = 1:Nf], x_kurt[i, idx] == tr(Bi[i] * W))
            _set_rm_risk_upper_bound(obj, type, model, kurt_risk[idx], rm.settings.ub)
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
            _set_rm_risk_upper_bound(obj, type, model, kurt_risk[idx], rm.settings.ub)
            _set_risk_expression(model, kurt_risk[idx], rm.settings.scale, rm.settings.flag)
        end
    end

    return nothing
end
function set_rm(port::Portfolio, rm::SKurt, type::Union{Trad, RP}, obj; kwargs...)
    _sdp(port, obj)
    model = port.model
    N = size(port.returns, 2)
    @variable(model, skurt_risk)
    kt = if (isnothing(rm.kt) || isempty(rm.kt))
        port.skurt
    else
        rm.kt
    end
    W = model[:W]
    if !iszero(port.max_num_assets_kurt) && N > port.max_num_assets_kurt
        f = port.max_num_assets_kurt_scale
        Nf = f * N
        @variable(model, x_skurt[1:Nf])
        @constraint(model, [skurt_risk; x_skurt] ∈ SecondOrderCone())
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
        @constraint(model, [i = 1:Nf], x_skurt[i] == tr(Bi[i] * W))
    else
        L_2 = port.L_2
        S_2 = port.S_2
        sqrt_sigma_4 = sqrt(S_2 * kt * transpose(S_2))
        @expression(model, zskurt, L_2 * vec(W))
        @constraint(model, [skurt_risk; sqrt_sigma_4 * zskurt] ∈ SecondOrderCone())
    end
    _set_rm_risk_upper_bound(obj, type, model, skurt_risk, rm.settings.ub)
    _set_risk_expression(model, skurt_risk, rm.settings.scale, rm.settings.flag)
    return nothing
end
function set_rm(port::Portfolio, rms::AbstractVector{<:SKurt}, type::Union{Trad, RP}, obj;
                kwargs...)
    _sdp(port, obj)
    model = port.model
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
            @constraint(model, [skurt_risk[idx]; view(x_skurt, :, idx)] ∈ SecondOrderCone())
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
            @constraint(model, [i = 1:Nf], x_skurt[i, idx] == tr(Bi[i] * W))
            _set_rm_risk_upper_bound(obj, type, model, skurt_risk[idx], rm.settings.ub)
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
            _set_rm_risk_upper_bound(obj, type, model, skurt_risk[idx], rm.settings.ub)
            _set_risk_expression(model, skurt_risk[idx], rm.settings.scale,
                                 rm.settings.flag)
        end
    end

    return nothing
end
function _owa_setup(model, T)
    if !haskey(model, :owa)
        @variable(model, owa[1:T])
        X = model[:X]
        @constraint(model, X == owa)
    end
    return nothing
end
function set_rm(port::Portfolio, rm::GMD, type::Union{Trad, RP}, obj;
                returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
    T = size(returns, 1)

    if !haskey(model, :X)
        w = model[:w]
        @expression(model, X, returns * w)
    end

    if !rm.owa.approx
        ovec = range(1; stop = 1, length = T)
        _owa_setup(model, T)
        @variable(model, gmda[1:T])
        @variable(model, gmdb[1:T])
        @expression(model, gmd_risk, sum(gmda .+ gmdb))
        gmd_w = owa_gmd(T)
        owa = model[:owa]
        @constraint(model,
                    owa * transpose(gmd_w) .<=
                    ovec * transpose(gmda) + gmdb * transpose(ovec))
    else
        owa_p = rm.owa.p
        M = length(owa_p)

        @variable(model, gmd_t)
        @variable(model, gmd_nu[1:T] .>= 0)
        @variable(model, gmd_eta[1:T] .>= 0)
        @variable(model, gmd_epsilon[1:T, 1:M])
        @variable(model, gmd_psi[1:T, 1:M])
        @variable(model, gmd_z[1:M])
        @variable(model, gmd_y[1:M] .>= 0)

        gmd_w = -owa_gmd(T)
        gmd_s = sum(gmd_w)
        gmd_l = minimum(gmd_w)
        gmd_h = maximum(gmd_w)
        gmd_d = [norm(gmd_w, p) for p ∈ owa_p]

        @expression(model, gmd_risk,
                    gmd_s * gmd_t - gmd_l * sum(gmd_nu) +
                    gmd_h * sum(gmd_eta) +
                    dot(gmd_d, gmd_y))
        X = model[:X]
        @constraint(model,
                    X .+ gmd_t .- gmd_nu .+ gmd_eta .- vec(sum(gmd_epsilon; dims = 2)) .==
                    0)
        @constraint(model, gmd_z .+ gmd_y .== vec(sum(gmd_psi; dims = 1)))
        @constraint(model, [i = 1:M, j = 1:T],
                    [-gmd_z[i] * owa_p[i], gmd_psi[j, i] * owa_p[i] / (owa_p[i] - 1),
                     gmd_epsilon[j, i]] ∈ MOI.PowerCone(inv(owa_p[i])))
    end
    _set_rm_risk_upper_bound(obj, type, model, gmd_risk, rm.settings.ub)
    _set_risk_expression(model, gmd_risk, rm.settings.scale, rm.settings.flag)
    return nothing
end
function set_rm(port::Portfolio, rm::TG, type::Union{Trad, RP}, obj;
                returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
    T = size(returns, 1)
    if !haskey(model, :X)
        w = model[:w]
        @expression(model, X, returns * w)
    end

    alpha = rm.alpha
    a_sim = rm.a_sim
    alpha_i = rm.alpha_i
    if !rm.owa.approx
        ovec = range(1; stop = 1, length = T)
        _owa_setup(model, T)
        @variable(model, tga[1:T])
        @variable(model, tgb[1:T])
        @expression(model, tg_risk, sum(tga .+ tgb))
        tg_w = owa_tg(T; alpha_i = alpha_i, alpha = alpha, a_sim = a_sim)
        owa = model[:owa]
        @constraint(model,
                    owa * transpose(tg_w) .<= ovec * transpose(tga) + tgb * transpose(ovec))
    else
        owa_p = rm.owa.p
        M = length(owa_p)

        @variable(model, tg_t)
        @variable(model, tg_nu[1:T] .>= 0)
        @variable(model, tg_eta[1:T] .>= 0)
        @variable(model, tg_epsilon[1:T, 1:M])
        @variable(model, tg_psi[1:T, 1:M])
        @variable(model, tg_z[1:M])
        @variable(model, tg_y[1:M] .>= 0)

        tg_w = -owa_tg(T; alpha_i = alpha_i, alpha = alpha, a_sim = a_sim)
        tg_s = sum(tg_w)
        tg_l = minimum(tg_w)
        tg_h = maximum(tg_w)
        tg_d = [norm(tg_w, p) for p ∈ owa_p]

        @expression(model, tg_risk,
                    tg_s * tg_t - tg_l * sum(tg_nu) + tg_h * sum(tg_eta) + dot(tg_d, tg_y))
        X = model[:X]
        @constraint(model,
                    X .+ tg_t .- tg_nu .+ tg_eta .- vec(sum(tg_epsilon; dims = 2)) .== 0)
        @constraint(model, tg_z .+ tg_y .== vec(sum(tg_psi; dims = 1)))
        @constraint(model, [i = 1:M, j = 1:T],
                    [-tg_z[i] * owa_p[i], tg_psi[j, i] * owa_p[i] / (owa_p[i] - 1),
                     tg_epsilon[j, i]] ∈ MOI.PowerCone(inv(owa_p[i])))
    end
    _set_rm_risk_upper_bound(obj, type, model, tg_risk, rm.settings.ub)
    _set_risk_expression(model, tg_risk, rm.settings.scale, rm.settings.flag)

    return nothing
end
function set_rm(port::Portfolio, rms::AbstractVector{<:TG}, type::Union{Trad, RP}, obj;
                returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
    if !haskey(model, :X)
        w = model[:w]
        @expression(model, X, returns * w)
    end
    T = size(returns, 1)
    count = length(rms)
    @expression(model, tg_risk[1:count], zero(AffExpr))
    X = model[:X]
    for (idx, rm) ∈ pairs(rms)
        alpha = rm.alpha
        a_sim = rm.a_sim
        alpha_i = rm.alpha_i
        if !rm.owa.approx
            ovec = range(1; stop = 1, length = T)
            if !haskey(model, :tga)
                _owa_setup(model, T)
                @variable(model, tga[1:T, 1:count])
                @variable(model, tgb[1:T, 1:count])
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
            owa_p = rm.owa.p
            M = length(owa_p)
            tg_w = -owa_tg(T; alpha_i = alpha_i, alpha = alpha, a_sim = a_sim)
            tg_s = sum(tg_w)
            tg_l = minimum(tg_w)
            tg_h = maximum(tg_w)
            tg_d = [norm(tg_w, p) for p ∈ owa_p]
            if !haskey(model, :tg_t)
                M = length(rm.owa.p)
                @variable(model, tg_t[1:count])
                @variable(model, tg_nu[1:T, 1:count] .>= 0)
                @variable(model, tg_eta[1:T, 1:count] .>= 0)
                @variable(model, tg_epsilon[1:T, 1:M, 1:count])
                @variable(model, tg_psi[1:T, 1:M, 1:count])
                @variable(model, tg_z[1:M, 1:count])
                @variable(model, tg_y[1:M, 1:count] .>= 0)
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
            @constraint(model,
                        X .+ tg_t[idx] .- view(tg_nu, :, idx) .+ view(tg_eta, :, idx) .-
                        vec(sum(view(tg_epsilon, :, :, idx); dims = 2)) .== 0)
            @constraint(model,
                        tg_z[:, idx] .+ tg_y[:, idx] .==
                        vec(sum(view(tg_psi, :, :, idx); dims = 1)))
            @constraint(model, [i = 1:M, j = 1:T],
                        [-tg_z[i, idx] * owa_p[i],
                         tg_psi[j, i, idx] * owa_p[i] / (owa_p[i] - 1),
                         tg_epsilon[j, i, idx]] ∈ MOI.PowerCone(inv(owa_p[i])))
        end
        _set_rm_risk_upper_bound(obj, type, model, tg_risk[idx], rm.settings.ub)
        _set_risk_expression(model, tg_risk[idx], rm.settings.scale, rm.settings.flag)
    end
    return nothing
end
function set_rm(port::Portfolio, rm::TGRG, type::Union{Trad, RP}, obj;
                returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
    T = size(returns, 1)
    if !haskey(model, :X)
        w = model[:w]
        @expression(model, X, returns * model[:w])
    end

    alpha = rm.alpha
    a_sim = rm.a_sim
    alpha_i = rm.alpha_i
    beta = rm.beta
    b_sim = rm.b_sim
    beta_i = rm.beta_i
    if !rm.owa.approx
        ovec = range(1; stop = 1, length = T)
        _owa_setup(model, T)
        @variable(model, rtga[1:T])
        @variable(model, rtgb[1:T])
        @expression(model, rtg_risk, sum(rtga .+ rtgb))
        rtg_w = owa_rtg(T; alpha_i = alpha_i, alpha = alpha, a_sim = a_sim, beta_i = beta_i,
                        beta = beta, b_sim = b_sim)
        owa = model[:owa]
        @constraint(model,
                    owa * transpose(rtg_w) .<=
                    ovec * transpose(rtga) + rtgb * transpose(ovec))
    else
        owa_p = rm.owa.p
        M = length(owa_p)

        @variable(model, rltg_t)
        @variable(model, rltg_nu[1:T] .>= 0)
        @variable(model, rltg_eta[1:T] .>= 0)
        @variable(model, rltg_epsilon[1:T, 1:M])
        @variable(model, rltg_psi[1:T, 1:M])
        @variable(model, rltg_z[1:M])
        @variable(model, rltg_y[1:M] .>= 0)

        rltg_w = -owa_tg(T; alpha_i = alpha_i, alpha = alpha, a_sim = a_sim)
        rltg_s = sum(rltg_w)
        rltg_l = minimum(rltg_w)
        rltg_h = maximum(rltg_w)
        rltg_d = [norm(rltg_w, p) for p ∈ owa_p]

        @expression(model, rltg_risk,
                    rltg_s * rltg_t - rltg_l * sum(rltg_nu) +
                    rltg_h * sum(rltg_eta) +
                    dot(rltg_d, rltg_y))
        X = model[:X]
        @constraint(model,
                    X .+ rltg_t .- rltg_nu .+ rltg_eta .-
                    vec(sum(rltg_epsilon; dims = 2)) .== 0)
        @constraint(model, rltg_z .+ rltg_y .== vec(sum(rltg_psi; dims = 1)))
        @constraint(model, [i = 1:M, j = 1:T],
                    [-rltg_z[i] * owa_p[i], rltg_psi[j, i] * owa_p[i] / (owa_p[i] - 1),
                     rltg_epsilon[j, i]] ∈ MOI.PowerCone(inv(owa_p[i])))

        @variable(model, rhtg_t)
        @variable(model, rhtg_nu[1:T] .>= 0)
        @variable(model, rhtg_eta[1:T] .>= 0)
        @variable(model, rhtg_epsilon[1:T, 1:M])
        @variable(model, rhtg_psi[1:T, 1:M])
        @variable(model, rhtg_z[1:M])
        @variable(model, rhtg_y[1:M] .>= 0)

        rhtg_w = -owa_tg(T; alpha_i = beta_i, alpha = beta, a_sim = b_sim)
        rhtg_s = sum(rhtg_w)
        rhtg_l = minimum(rhtg_w)
        rhtg_h = maximum(rhtg_w)
        rhtg_d = [norm(rhtg_w, p) for p ∈ owa_p]

        @expression(model, rhtg_risk,
                    rhtg_s * rhtg_t - rhtg_l * sum(rhtg_nu) +
                    rhtg_h * sum(rhtg_eta) +
                    dot(rhtg_d, rhtg_y))
        @expression(model, rtg_risk, rltg_risk + rhtg_risk)
        @constraint(model,
                    -X .+ rhtg_t .- rhtg_nu .+ rhtg_eta .-
                    vec(sum(rhtg_epsilon; dims = 2)) .== 0)
        @constraint(model, rhtg_z .+ rhtg_y .== vec(sum(rhtg_psi; dims = 1)))
        @constraint(model, [i = 1:M, j = 1:T],
                    [-rhtg_z[i] * owa_p[i], rhtg_psi[j, i] * owa_p[i] / (owa_p[i] - 1),
                     rhtg_epsilon[j, i]] ∈ MOI.PowerCone(inv(owa_p[i])))
    end
    _set_rm_risk_upper_bound(obj, type, model, rtg_risk, rm.settings.ub)
    _set_risk_expression(model, rtg_risk, rm.settings.scale, rm.settings.flag)
    return nothing
end
function set_rm(port::Portfolio, rms::AbstractVector{<:TGRG}, type::Union{Trad, RP}, obj;
                returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
    if !haskey(model, :X)
        w = model[:w]
        @expression(model, X, returns * w)
    end
    T = size(returns, 1)
    count = length(rms)
    @expression(model, rtg_risk[1:count], zero(AffExpr))
    X = model[:X]
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
                _owa_setup(model, T)
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
                        X .+ rltg_t[idx] .- view(rltg_nu, :, idx) .+
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
                        -X .+ rhtg_t[idx] .- view(rhtg_nu, :, idx) .+
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
        _set_rm_risk_upper_bound(obj, type, model, rtg_risk[idx], rm.settings.ub)
        _set_risk_expression(model, rtg_risk[idx], rm.settings.scale, rm.settings.flag)
    end
    return nothing
end
function set_rm(port::Portfolio, rm::OWA, type::Union{Trad, RP}, obj;
                returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
    T = size(returns, 1)
    if !haskey(model, :X)
        w = model[:w]
        @expression(model, X, returns * w)
    end
    if !rm.owa.approx
        ovec = range(1; stop = 1, length = T)
        _owa_setup(model, T)
        @variable(model, owa_a[1:T])
        @variable(model, owa_b[1:T])
        @expression(model, owa_risk, sum(owa_a .+ owa_b))
        owa_w = (isnothing(rm.w) || isempty(rm.w)) ? owa_gmd(T) : rm.w
        owa = model[:owa]
        @constraint(model,
                    owa * transpose(owa_w) .<=
                    ovec * transpose(owa_a) + owa_b * transpose(ovec))
    else
        owa_p = rm.owa.p
        M = length(owa_p)

        @variable(model, owa_t)
        @variable(model, owa_nu[1:T] .>= 0)
        @variable(model, owa_eta[1:T] .>= 0)
        @variable(model, owa_epsilon[1:T, 1:M])
        @variable(model, owa_psi[1:T, 1:M])
        @variable(model, owa_z[1:M])
        @variable(model, owa_y[1:M] .>= 0)

        owa_w = (isnothing(rm.w) || isempty(rm.w)) ? -owa_gmd(T) : -rm.w
        owa_s = sum(owa_w)
        owa_l = minimum(owa_w)
        owa_h = maximum(owa_w)
        owa_d = [norm(owa_w, p) for p ∈ owa_p]

        @expression(model, owa_risk,
                    owa_s * owa_t - owa_l * sum(owa_nu) +
                    owa_h * sum(owa_eta) +
                    dot(owa_d, owa_y))
        X = model[:X]
        @constraint(model,
                    X .+ owa_t .- owa_nu .+ owa_eta .- vec(sum(owa_epsilon; dims = 2)) .==
                    0)
        @constraint(model, owa_z .+ owa_y .== vec(sum(owa_psi; dims = 1)))
        @constraint(model, [i = 1:M, j = 1:T],
                    [-owa_z[i] * owa_p[i], owa_psi[j, i] * owa_p[i] / (owa_p[i] - 1),
                     owa_epsilon[j, i]] ∈ MOI.PowerCone(inv(owa_p[i])))
    end
    _set_rm_risk_upper_bound(obj, type, model, owa_risk, rm.settings.ub)
    _set_risk_expression(model, owa_risk, rm.settings.scale, rm.settings.flag)
    return nothing
end
function set_rm(port::Portfolio, rms::AbstractVector{<:OWA}, type::Union{Trad, RP}, obj;
                returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
    if !haskey(model, :X)
        w = model[:w]
        @expression(model, X, returns * w)
    end
    T = size(returns, 1)
    count = length(rms)
    @expression(model, owa_risk[1:count], zero(AffExpr))
    X = model[:X]
    for (idx, rm) ∈ pairs(rms)
        if !rm.owa.approx
            ovec = range(1; stop = 1, length = T)
            if !haskey(model, :owa_a)
                _owa_setup(model, T)
                @variable(model, owa_a[1:T, 1:count])
                @variable(model, owa_b[1:T, 1:count])
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
            owa_p = rm.owa.p
            M = length(owa_p)

            owa_w = (isnothing(rm.w) || isempty(rm.w)) ? -owa_gmd(T) : -rm.w
            owa_s = sum(owa_w)
            owa_l = minimum(owa_w)
            owa_h = maximum(owa_w)
            owa_d = [norm(owa_w, p) for p ∈ owa_p]

            if !haskey(model, :owa_t)
                M = length(rm.owa.p)
                @variable(model, owa_t[1:count])
                @variable(model, owa_nu[1:T, 1:count] .>= 0)
                @variable(model, owa_eta[1:T, 1:count] .>= 0)
                @variable(model, owa_epsilon[1:T, 1:M, 1:count])
                @variable(model, owa_psi[1:T, 1:M, 1:count])
                @variable(model, owa_z[1:M, 1:count])
                @variable(model, owa_y[1:M, 1:count] .>= 0)
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
            @constraint(model,
                        X .+ owa_t[idx] .- view(owa_nu, :, idx) .+ view(owa_eta, :, idx) .-
                        vec(sum(view(owa_epsilon, :, :, idx); dims = 2)) .== 0)
            @constraint(model,
                        view(owa_z, :, idx) .+ view(owa_y, :, idx) .==
                        vec(sum(view(owa_psi, :, :, idx); dims = 1)))
            @constraint(model, [i = 1:M, j = 1:T],
                        [-owa_z[i, idx] * owa_p[i],
                         owa_psi[j, i, idx] * owa_p[i] / (owa_p[i] - 1),
                         owa_epsilon[j, i, idx]] ∈ MOI.PowerCone(inv(owa_p[i])))
        end
        _set_rm_risk_upper_bound(obj, type, model, owa_risk[idx], rm.settings.ub)
        _set_risk_expression(model, owa_risk[idx], rm.settings.scale, rm.settings.flag)
    end
    return nothing
end
function set_rm(port::Portfolio, rm::dVar, type::Union{Trad, RP}, obj;
                returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
    T = size(returns, 1)

    if !haskey(model, :X)
        w = model[:w]
        @expression(model, X, returns * w)
    end

    ovec = range(1; stop = 1, length = T)
    @variable(model, Dt[1:T, 1:T], Symmetric)
    X = model[:X]
    @expression(model, Dx, X * transpose(ovec) - ovec * transpose(X))
    @constraint(model, [i = 1:T, j = i:T], [Dt[i, j]; Dx[i, j]] in MOI.NormOneCone(2))
    dt = vec(Dt)
    iT2 = inv(T^2)
    @expression(model, sum_Dt, sum(Dt))
    @expression(model, dvar_risk, iT2 * (dot(dt, dt) + iT2 * sum_Dt^2))
    _set_rm_risk_upper_bound(obj, type, model, dvar_risk, rm.settings.ub)
    _set_risk_expression(model, dvar_risk, rm.settings.scale, rm.settings.flag)
    return nothing
end
function set_rm(port::Portfolio, rm::Skew, type::Union{Trad, RP}, obj; kwargs...)
    model = port.model

    G = real(sqrt(port.V))
    @variable(model, t_skew)
    w = model[:w]
    @constraint(model, [t_skew; G * w] ∈ SecondOrderCone())
    @expression(model, skew_risk, t_skew^2)
    _set_rm_risk_upper_bound(obj, type, model, t_skew, rm.settings.ub)
    _set_risk_expression(model, skew_risk, rm.settings.scale, rm.settings.flag)
    return nothing
end
function set_rm(port::Portfolio, rm::SSkew, type::Union{Trad, RP}, obj; kwargs...)
    model = port.model

    G = real(sqrt(port.SV))
    @variable(model, t_sskew)
    w = model[:w]
    @constraint(model, [t_sskew; G * w] ∈ SecondOrderCone())
    @expression(model, sskew_risk, t_sskew^2)
    _set_rm_risk_upper_bound(obj, type, model, t_sskew, rm.settings.ub)
    _set_risk_expression(model, sskew_risk, rm.settings.scale, rm.settings.flag)
    return nothing
end
function risk_constraints(port, obj, type::Union{Trad, RP}, rm::RiskMeasure, mu, sigma,
                          returns, kelly_approx_idx = nothing)
    set_rm(port, rm, type, obj; mu = mu, sigma = sigma, returns = returns,
           kelly_approx_idx = kelly_approx_idx)
    return nothing
end
function risk_constraints(port, obj, type::Union{Trad, RP}, rms::AbstractVector, mu, sigma,
                          returns, kelly_approx_idx = nothing)
    for rm ∈ rms
        set_rm(port, rm, type, obj; mu = mu, sigma = sigma, returns = returns,
               kelly_approx_idx = kelly_approx_idx)
    end
    return nothing
end
