function initial_w(port, w_ini)
    if !haskey(port.model, :w)
        @variable(port.model, w[1:size(port.returns, 2)])
    end
    if !isempty(w_ini)
        @smart_assert(length(w_ini) == length(w))
        set_start_value.(w, w_ini)
    end
    return nothing
end
function mu_sigma_returns_class(port, ::Union{Classic2, FC2})
    return port.mu, port.cov, port.returns
end
function mu_sigma_returns_class(port, class::FM2)
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
function mu_sigma_returns_class(port, class::BL2)
    mu = port.bl_mu
    returns = port.returns
    if class.type == 1
        sigma = port.bl_cov
    else
        sigma = port.cov
    end
    return mu, sigma, returns
end
function mu_sigma_returns_class(port, class::BLFM2)
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
# sharpe ratio k
function set_sr_k(::SR, model)
    @variable(model, k >= 0)
    return nothing
end
function set_sr_k(::Any, ::Any)
    return nothing
end
# Risk upper bounds
function _set_rm_risk_upper_bound(args...)
    return nothing
end
function _set_rm_risk_upper_bound(::SR, ::Trad2, model, rm_risk, ub)
    if isfinite(ub)
        @constraint(model, rm_risk .<= ub * model[:k])
    end
    return nothing
end
function _set_rm_risk_upper_bound(::Any, ::Trad2, model, rm_risk, ub)
    if isfinite(ub)
        @constraint(model, rm_risk .<= ub)
    end
    return nothing
end
# SD risk upper bound (special case)
function _set_sd_risk_upper_bound(args...)
    return nothing
end
function _set_sd_risk_upper_bound(::SDP2, ::SR, ::Trad2, model, ub, count, idx)
    if isfinite(ub)
        if isone(count)
            @constraint(model, model[:sd_risk] .<= ub^2 * model[:k])
        else
            @constraint(model, model[:sd_risk][idx] .<= ub^2 * model[:k])
        end
    end
    return nothing
end
function _set_sd_risk_upper_bound(::SDP2, ::Any, ::Trad2, model, ub, count, idx)
    if isfinite(ub)
        if isone(count)
            @constraint(model, model[:sd_risk] .<= ub^2)
        else
            @constraint(model, model[:sd_risk][idx] .<= ub^2)
        end
    end
    return nothing
end
function _set_sd_risk_upper_bound(::Any, ::SR, ::Trad2, model, ub, count, idx)
    if isfinite(ub)
        if isone(count)
            @constraint(model, model[:dev] .<= ub * model[:k])
        else
            @constraint(model, model[:dev][idx] .<= ub * model[:k])
        end
    end
    return nothing
end
function _set_sd_risk_upper_bound(::Any, ::Any, ::Trad2, model, ub, count, idx)
    if isfinite(ub)
        if isone(count)
            @constraint(model, model[:dev] .<= ub)
        else
            @constraint(model, model[:dev][idx] .<= ub)
        end
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
                add_to_expression!(model[:risk], scale, rm_risk)
            catch
                @expression(model, tmp, model[:risk] + scale * rm_risk)
                unregister(model, :risk)
                @expression(model, risk, tmp)
                unregister(model, :tmp)
            end
        end
    end
    return nothing
end
function _sdp_m2(::SR, model)
    @expression(model, M2, vcat(model[:w], model[:k]))
    return nothing
end
function _sdp_m2(::Any, model)
    @expression(model, M2, vcat(model[:w], 1))
    return nothing
end
function _sdp(port, obj)
    model = port.model
    if !haskey(model, :W)
        N = size(port.returns, 2)
        @variable(model, W[1:N, 1:N], Symmetric)
        @expression(model, M1, vcat(W, transpose(model[:w])))
        _sdp_m2(obj, model)
        @expression(model, M3, hcat(M1, model[:M2]))
        @constraint(model, M3 ∈ PSDCone())
    end
    return nothing
end
function _sdp(::SDP2, port, obj)
    model = port.model
    if !haskey(model, :W)
        N = size(port.returns, 2)
        @variable(model, W[1:N, 1:N], Symmetric)
        @expression(model, M1, vcat(W, transpose(model[:w])))
        _sdp_m2(obj, model)
        @expression(model, M3, hcat(M1, model[:M2]))
        @constraint(model, M3 ∈ PSDCone())
    end
    return nothing
end
function _sdp(::Any, ::Any, ::Any)
    return nothing
end
function num_assets_constraints(port, ::SR)
    if port.num_assets_u > 0
        N = size(port.returns, 2)
        model = port.model

        @variable(model, tnau_bin[1:N], binary = true)
        @constraint(model, sum(tnau_bin) <= port.num_assets_u)
        # Sharpe ratio
        @variable(model, tnau_bin_sharpe[1:N] .>= 0)
        @constraint(model, tnau_bin_sharpe .<= model[:k])
        @constraint(model, tnau_bin_sharpe .<= port.num_assets_u_scale * model[:tnau_bin])
        @constraint(model,
                    tnau_bin_sharpe .>=
                    model[:k] .- port.num_assets_u_scale * (1 .- model[:tnau_bin]))
        # Long and short
        @constraint(model, model[:w] .<= port.long_u * tnau_bin_sharpe)
        if port.short
            @constraint(model, model[:w] .>= -port.short_u * tnau_bin_sharpe)
        end
    end
    if port.num_assets_l > 0
        N = size(port.returns, 2)
        model = port.model
        @variable(model, tnal >= 0)
        @constraint(model, [tnal; model[:w]] ∈ SecondOrderCone())
        @constraint(model, tnal * sqrt(port.num_assets_l) <= model[:k])
    end
    return nothing
end
function num_assets_constraints(port, ::Any)
    if port.num_assets_u > 0
        N = size(port.returns, 2)
        model = port.model

        @variable(model, tnau_bin[1:N], binary = true)
        @constraint(model, sum(tnau_bin) <= port.num_assets_u)
        # Long and short
        @constraint(model, model[:w] .<= port.long_u * tnau_bin)
        if port.short
            @constraint(model, model[:w] .>= -port.short_u * tnau_bin)
        end
    end
    if port.num_assets_l > 0
        N = size(port.returns, 2)
        model = port.model
        @variable(model, tnal >= 0)
        @constraint(model, [tnal; model[:w]] ∈ SecondOrderCone())
        @constraint(model, tnal * sqrt(port.num_assets_l) <= 1)
    end
    return nothing
end
function weight_constraints(port, ::SR)
    N = size(port.returns, 2)
    model = port.model
    @constraint(model, sum(model[:w]) == port.sum_short_long * model[:k])
    if !port.short
        @constraint(model, model[:w] .<= port.long_u * model[:k])
        @constraint(model, model[:w] .>= 0)
    else
        @variable(model, tw_ulong[1:N] .>= 0)
        @variable(model, tw_ushort[1:N] .>= 0)

        @constraint(model, sum(tw_ulong) <= port.long_u * model[:k])
        @constraint(model, sum(tw_ushort) <= port.short_u * model[:k])

        @constraint(model, model[:w] .<= tw_ulong)
        @constraint(model, model[:w] .>= -tw_ushort)
    end
    return nothing
end
function weight_constraints(port, ::Any)
    N = size(port.returns, 2)
    model = port.model
    @constraint(model, sum(model[:w]) == port.sum_short_long)
    if !port.short
        @constraint(model, model[:w] .<= port.long_u)
        @constraint(model, model[:w] .>= 0)
    else
        @variable(model, tw_ulong[1:N] .>= 0)
        @variable(model, tw_ushort[1:N] .>= 0)

        @constraint(model, sum(tw_ulong) <= port.long_u)
        @constraint(model, sum(tw_ushort) <= port.short_u)

        @constraint(model, model[:w] .<= tw_ulong)
        @constraint(model, model[:w] .>= -tw_ushort)
    end
    return nothing
end
function network_constraints(args...)
    return nothing
end
function network_constraints(network::IP2, port, ::SR, ::Any)
    N = size(port.returns, 2)
    model = port.model

    @variable(model, tip_bin2[1:N], binary = true)
    @constraint(model, unique(network.A + I; dims = 1) * tip_bin2 .<= 1)
    # Sharpe ratio
    @variable(model, tip_bin_sharpe2[1:N] .>= 0)
    @constraint(model, tip_bin_sharpe2 .<= model[:k])
    @constraint(model, tip_bin_sharpe2 .<= network.scale * tip_bin2)
    @constraint(model, tip_bin_sharpe2 .>= model[:k] .- network.scale * (1 .- tip_bin2))
    # Long and short
    @constraint(model, model[:w] .<= port.long_u * tip_bin_sharpe2)
    if port.short
        @constraint(model, model[:w] .>= -port.short_u * tip_bin_sharpe2)
    end
    return nothing
end
function network_constraints(network::IP2, port, ::Any, ::Any)
    N = size(port.returns, 2)
    model = port.model

    @variable(model, tip_bin2[1:N], binary = true)
    @constraint(model, unique(network.A + I; dims = 1) * tip_bin2 .<= 1)
    # Long and short
    @constraint(model, model[:w] .<= port.long_u * model[:tip_bin2])
    if port.short
        @constraint(model, model[:w] .>= -port.short_u * model[:tip_bin2])
    end
    return nothing
end
function network_constraints(network::SDP2, port, obj, ::Trad2)
    _sdp(port, obj)
    @constraint(port.model, network.A .* port.model[:W] .== 0)
    if !haskey(port.model, :sd_risk) && hasproperty(port.network_method, :penalty)
        @expression(port.model, network_penalty, network.penalty * tr(port.model[:W]))
    end
    return nothing
end
function network_constraints(network::SDP2, port, obj, ::WC2)
    _sdp(port, obj)
    @constraint(port.model, network.A .* port.model[:W] .== 0)
    return nothing
end
function _centrality_constraints(::SR, model, A, B)
    @constraint(model, dot(A, model[:w]) - B * model[:k] == 0)
    return nothing
end
function _centrality_constraints(::Any, model, A, B)
    @constraint(model, dot(A, model[:w]) - B == 0)
    return nothing
end
function centrality_constraints(port, obj)
    if !(isempty(port.a_vec_cent) || isinf(port.b_cent))
        _centrality_constraints(obj, port.model, port.a_vec_cent, port.b_cent)
    end
    return nothing
end
function _linear_constraints(::Union{SR, RP2}, model, A, B)
    @constraint(model, A * model[:w] .- B * model[:k] .>= 0)
    return nothing
end
function _linear_constraints(::Any, model, A, B)
    @constraint(model, A * model[:w] .- B .>= 0)
    return nothing
end
function linear_constraints(port, obj_type)
    if !(isempty(port.a_mtx_ineq) || isempty(port.b_vec_ineq))
        _linear_constraints(obj_type, port.model, port.a_mtx_ineq, port.b_vec_ineq)
    end
    return nothing
end
function _sd_risk(::SDP2, ::Any, model, sigma, count::Integer, idx::Integer)
    if isone(count)
        @expression(model, sd_risk, tr(sigma * model[:W]))
    else
        if isone(idx)
            @variable(model, sd_risk[1:count])
        end
        @constraint(model, model[:sd_risk][idx] == tr(sigma * model[:W]))
    end
    return nothing
end
function _sd_risk(::Union{NoNtwk, IP2}, ::SOCSD, model, sigma, count::Integer, idx::Integer)
    G = sqrt(sigma)
    if isone(count)
        @variable(model, dev)
        @expression(model, sd_risk, dev^2)
        @constraint(model, [dev; G * model[:w]] ∈ SecondOrderCone())
    else
        if isone(idx)
            @variable(model, dev[1:count])
            @expression(model, sd_risk[1:count], zero(QuadExpr))
        end
        add_to_expression!(model[:sd_risk][idx], model[:dev][idx], model[:dev][idx])
        @constraint(model, [model[:dev][idx]; G * model[:w]] ∈ SecondOrderCone())
    end
    return nothing
end
function _sd_risk(::Union{NoNtwk, IP2}, ::QuadSD, model, sigma, count::Integer,
                  idx::Integer)
    G = sqrt(sigma)
    if isone(count)
        @variable(model, dev)
        @expression(model, sd_risk, dot(model[:w], sigma, model[:w]))
        @constraint(model, [dev; G * model[:w]] ∈ SecondOrderCone())
    else
        if isone(idx)
            @variable(model, dev[1:count])
            @expression(model, sd_risk[1:count], zero(QuadExpr))
        end
        add_to_expression!(model[:sd_risk][idx], dot(model[:w], sigma, model[:w]))
        @constraint(model, [model[:dev][idx]; G * model[:w]] ∈ SecondOrderCone())
    end
    return nothing
end
function _sd_risk(::Union{NoNtwk, IP2}, ::SimpleSD, model, sigma, count::Integer,
                  idx::Integer)
    G = sqrt(sigma)
    if isone(count)
        @variable(model, dev)
        @expression(model, sd_risk, dev)
        @constraint(model, [dev; G * model[:w]] ∈ SecondOrderCone())
    else
        if isone(idx)
            @variable(model, dev[1:count])
            @expression(model, sd_risk[1:count], zero(QuadExpr))
        end
        add_to_expression!(model[:sd_risk][idx], model[:dev][idx])
        @constraint(model, [model[:dev][idx]; G * model[:w]] ∈ SecondOrderCone())
    end
    return nothing
end
function _dev_setup(args...)
    return nothing
end
function _dev_setup(::SDP2, model, idx::Union{Nothing, Integer} = nothing)
    if isnothing(idx)
        @variable(model, dev)
        @constraint(model, [model[:sd_risk], 1, dev] in MOI.PowerCone(0.5))
    else
        @variable(model, dev)
        @constraint(model, [model[:sd_risk][idx], 1, dev] in MOI.PowerCone(0.5))
    end
    return nothing
end
function _get_ntwk_method(::Trad2, port)
    return port.network_method
end
function _get_ntwk_method(args...)
    return NoNtwk()
end
function set_rm(port::Portfolio2, rm::SD2, type::Union{Trad2, RP2}, obj::ObjectiveFunction,
                count::Integer, idx::Integer; sigma::AbstractMatrix{<:Real},
                kelly_approx_idx::Union{AbstractVector{<:Integer}, Nothing}, kwargs...)
    use_portfolio_sigma = (isnothing(rm.sigma) || isempty(rm.sigma))
    if !isnothing(kelly_approx_idx) && use_portfolio_sigma
        if isempty(kelly_approx_idx)
            push!(kelly_approx_idx, isone(count) ? 0 : idx)
        end
    end
    if !use_portfolio_sigma
        sigma = rm.sigma
    end
    model = port.model

    network_method = _get_ntwk_method(Trad2(), port)
    _sdp(network_method, port, obj)
    _sd_risk(network_method, rm.formulation, model, sigma, count, idx)
    _set_sd_risk_upper_bound(network_method, obj, type, model, rm.settings.ub, count, idx)
    if isone(count)
        _set_risk_expression(model, model[:sd_risk], rm.settings.scale, rm.settings.flag)
    else
        _set_risk_expression(model, model[:sd_risk][idx], rm.settings.scale,
                             rm.settings.flag)
    end
    return nothing
end
function set_rm(port::Portfolio2, rm::MAD2, type::Union{Trad2, RP2}, obj::ObjectiveFunction,
                count::Integer, idx::Integer; mu::AbstractVector{<:Real},
                returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
    T = size(returns, 1)
    if !(isnothing(rm.mu) || isempty(rm.mu))
        mu = rm.mu
    end
    abs_dev = returns .- transpose(mu)

    if isone(count)
        @variable(model, mad[1:T] >= 0)
        @expression(model, mad_risk, sum(mad) / T)
        @constraint(model, abs_dev * model[:w] .>= -mad)
        _set_rm_risk_upper_bound(obj, type, model, mad_risk, 0.5 * rm.settings.ub)
        _set_risk_expression(model, mad_risk, rm.settings.scale, rm.settings.flag)
    else
        if isone(idx)
            @variable(model, mad[1:T, 1:count] >= 0)
            @expression(model, mad_risk[1:count], zero(AffExpr))
        end
        add_to_expression!(model[:mad_risk][idx], inv(T), sum(view(model[:mad], :, idx)))
        @constraint(model, abs_dev * model[:w] .>= -view(model[:mad], :, idx))
        _set_rm_risk_upper_bound(obj, type, model, model[:mad_risk][idx],
                                 0.5 * rm.settings.ub)
        _set_risk_expression(model, model[:mad_risk][idx], rm.settings.scale,
                             rm.settings.flag)
    end
    return nothing
end
function set_rm(port::Portfolio2, rm::SSD2, type::Union{Trad2, RP2}, obj::ObjectiveFunction,
                count::Integer, idx::Integer; mu::AbstractVector{<:Real},
                returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
    T = size(returns, 1)
    if !(isnothing(rm.mu) || isempty(rm.mu))
        mu = rm.mu
    end
    abs_dev = returns .- transpose(mu)

    if isone(count)
        @variable(model, ssd[1:T] >= 0)
        @variable(model, sdev)
        @expression(model, sdev_risk, sdev / sqrt(T - 1))
        @constraint(model, abs_dev * model[:w] .>= -ssd)
        @constraint(model, [sdev; ssd] ∈ SecondOrderCone())
        _set_rm_risk_upper_bound(obj, type, model, sdev_risk, rm.settings.ub)
        _set_risk_expression(model, sdev_risk, rm.settings.scale, rm.settings.flag)
    else
        if isone(idx)
            @variable(model, ssd[1:T, 1:count] >= 0)
            @variable(model, sdev[1:count])
            @expression(model, sdev_risk[1:count], zero(AffExpr))
        end
        add_to_expression!(model[:sdev_risk][idx], inv(sqrt(T - 1)), model[:sdev][idx])
        @constraint(model, abs_dev * model[:w] .>= -view(model[:ssd], :, idx))
        @constraint(model,
                    [model[:sdev][idx]; view(model[:ssd], :, idx)] ∈ SecondOrderCone())
        _set_rm_risk_upper_bound(obj, type, model, model[:sdev_risk][idx], rm.settings.ub)
        _set_risk_expression(model, model[:sdev_risk][idx], rm.settings.scale,
                             rm.settings.flag)
    end
    return nothing
end
function _lpm_risk(::RP2, ::Any, model, lpm, target)
    @constraint(model, lpm .>= target * model[:k] .- model[:X])
    return nothing
end
function _lpm_risk(::Trad2, ::SR, model, lpm, target)
    @constraint(model, lpm .>= target * model[:k] .- model[:X])
    return nothing
end
function _lpm_risk(::Any, ::Any, model, lpm, target)
    @constraint(model, lpm .>= target .- model[:X])
    return nothing
end
function set_rm(port::Portfolio2, rm::FLPM2, type::Union{Trad2, RP2},
                obj::ObjectiveFunction, count::Integer, idx::Integer;
                returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
    T = size(returns, 1)

    if !haskey(model, :X)
        @expression(model, X, returns * model[:w])
    end

    if isone(count)
        @variable(model, flpm[1:T] .>= 0)
        @expression(model, flpm_risk, sum(flpm) / T)
        _lpm_risk(type, obj, model, flpm, rm.target)
        _set_rm_risk_upper_bound(obj, type, model, flpm_risk, rm.settings.ub)
        _set_risk_expression(model, flpm_risk, rm.settings.scale, rm.settings.flag)
    else
        if isone(idx)
            @variable(model, flpm[1:T, 1:count] .>= 0)
            @expression(model, flpm_risk[1:count], zero(AffExpr))
        end
        add_to_expression!(model[:flpm_risk][idx], inv(T), sum(view(model[:flpm], :, idx)))
        _lpm_risk(type, obj, model, view(model[:flpm], :, idx), rm.target)
        _set_rm_risk_upper_bound(obj, type, model, model[:flpm_risk][idx], rm.settings.ub)
        _set_risk_expression(model, model[:flpm_risk][idx], rm.settings.scale,
                             rm.settings.flag)
    end
    return nothing
end
function set_rm(port::Portfolio2, rm::SLPM2, type::Union{Trad2, RP2},
                obj::ObjectiveFunction, count::Integer, idx::Integer;
                returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
    T = size(returns, 1)

    if !haskey(model, :X)
        @expression(model, X, returns * model[:w])
    end

    if isone(count)
        @variable(model, slpm[1:T] .>= 0)
        @variable(model, tslpm)
        @expression(model, slpm_risk, tslpm / sqrt(T - 1))
        @constraint(model, [tslpm; slpm] ∈ SecondOrderCone())
        _lpm_risk(type, obj, model, slpm, rm.target)
        _set_rm_risk_upper_bound(obj, type, model, slpm_risk, rm.settings.ub)
        _set_risk_expression(model, slpm_risk, rm.settings.scale, rm.settings.flag)
    else
        if isone(idx)
            @variable(model, slpm[1:T, 1:count] .>= 0)
            @variable(model, tslpm[1:count])
            @expression(model, slpm_risk[1:count], zero(AffExpr))
        end
        add_to_expression!(model[:slpm_risk][idx], inv(sqrt(T - 1)), model[:tslpm][idx])
        @constraint(model,
                    [model[:tslpm][idx]; view(model[:slpm], :, idx)] ∈ SecondOrderCone())
        _lpm_risk(type, obj, model, view(model[:slpm], :, idx), rm.target)
        _set_rm_risk_upper_bound(obj, type, model, model[:slpm_risk][idx], rm.settings.ub)
        _set_risk_expression(model, model[:slpm_risk][idx], rm.settings.scale,
                             rm.settings.flag)
    end
    return nothing
end
function _wr_setup(model, returns)
    if !haskey(model, :X)
        @expression(model, X, returns * model[:w])
    end
    if !haskey(model, :wr)
        @variable(model, wr)
        @expression(model, wr_risk, wr)
        @constraint(model, -model[:X] .<= wr)
    end
end
function set_rm(port::Portfolio2, rm::WR2, type::Union{Trad2, RP2}, obj::ObjectiveFunction,
                ::Any, ::Any; returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
    _wr_setup(model, returns)
    _set_rm_risk_upper_bound(obj, type, model, -model[:X], rm.settings.ub)
    _set_risk_expression(model, model[:wr_risk], rm.settings.scale, rm.settings.flag)
    return nothing
end
function set_rm(port::Portfolio2, rm::RG2, type::Union{Trad2, RP2}, obj::ObjectiveFunction,
                ::Any, ::Any; returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
    _wr_setup(model, returns)
    @variable(model, br)
    @expression(model, rg_risk, model[:wr_risk] - br)
    @constraint(model, -model[:X] .>= br)
    _set_rm_risk_upper_bound(obj, type, model, rg_risk, rm.settings.ub)
    _set_risk_expression(model, rg_risk, rm.settings.scale, rm.settings.flag)
    return nothing
end
function set_rm(port::Portfolio2, rm::CVaR2, type::Union{Trad2, RP2},
                obj::ObjectiveFunction, count::Integer, idx::Integer;
                returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
    T = size(returns, 1)
    iat = inv(rm.alpha * T)

    if !haskey(model, :X)
        @expression(model, X, returns * model[:w])
    end

    if isone(count)
        @variable(model, var)
        @variable(model, z_var[1:T] .>= 0)
        @expression(model, cvar_risk, var + sum(z_var) * iat)
        @constraint(model, z_var .>= -model[:X] .- var)
        _set_rm_risk_upper_bound(obj, type, model, cvar_risk, rm.settings.ub)
        _set_risk_expression(model, cvar_risk, rm.settings.scale, rm.settings.flag)
    else
        if isone(idx)
            @variable(model, var[1:count])
            @variable(model, z_var[1:T, 1:count] .>= 0)
            @expression(model, cvar_risk[1:count], zero(AffExpr))
        end
        add_to_expression!(model[:cvar_risk][idx], model[:var][idx])
        add_to_expression!(model[:cvar_risk][idx], iat, sum(view(model[:z_var], :, idx)))
        @constraint(model, view(model[:z_var], :, idx) .>= -model[:X] .- model[:var][idx])
        _set_rm_risk_upper_bound(obj, type, model, model[:cvar_risk][idx], rm.settings.ub)
        _set_risk_expression(model, model[:cvar_risk][idx], rm.settings.scale,
                             rm.settings.flag)
    end
    return nothing
end
function set_rm(port::Portfolio2, rm::RCVaR2, type::Union{Trad2, RP2},
                obj::ObjectiveFunction, count::Integer, idx::Integer;
                returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
    T = size(returns, 1)
    iat = inv(rm.alpha * T)
    ibt = inv(rm.beta * T)

    if !haskey(model, :X)
        @expression(model, X, returns * model[:w])
    end

    if isone(count)
        @variable(model, var_l)
        @variable(model, z_var_l[1:T] .>= 0)
        @expression(model, cvar_risk_l, var_l + sum(z_var_l) * iat)
        @constraint(model, z_var_l .>= -model[:X] .- var_l)
        @variable(model, var_h)
        @variable(model, z_var_h[1:T] .<= 0)
        @expression(model, cvar_risk_h, var_h + sum(z_var_h) * ibt)
        @constraint(model, z_var_h .<= -model[:X] .- var_h)
        @expression(model, rcvar_risk, cvar_risk_l - cvar_risk_h)
        _set_rm_risk_upper_bound(obj, type, model, rcvar_risk, rm.settings.ub)
        _set_risk_expression(model, rcvar_risk, rm.settings.scale, rm.settings.flag)
    else
        if isone(idx)
            @variable(model, var_l[1:count])
            @variable(model, z_var_l[1:T, 1:count] .>= 0)
            @expression(model, cvar_risk_l[1:count], zero(AffExpr))
            @variable(model, var_h[1:count])
            @variable(model, z_var_h[1:T, 1:count] .<= 0)
            @expression(model, cvar_risk_h[1:count], zero(AffExpr))
            @expression(model, rcvar_risk[1:count], zero(AffExpr))
        end
        add_to_expression!(model[:cvar_risk_l][idx], model[:var_l][idx])
        add_to_expression!(model[:cvar_risk_l][idx], iat,
                           sum(view(model[:z_var_l], :, idx)))
        @constraint(model,
                    view(model[:z_var_l], :, idx) .>= -model[:X] .- model[:var_l][idx])
        add_to_expression!(model[:cvar_risk_h][idx], model[:var_h][idx])
        add_to_expression!(model[:cvar_risk_h][idx], ibt,
                           sum(view(model[:z_var_h], :, idx)))
        @constraint(model,
                    view(model[:z_var_h], :, idx) .<= -model[:X] .- model[:var_h][idx])
        add_to_expression!(model[:rcvar_risk][idx], model[:cvar_risk_l][idx])
        add_to_expression!(model[:rcvar_risk][idx], -1, model[:cvar_risk_h][idx])
        _set_rm_risk_upper_bound(obj, type, model, model[:rcvar_risk][idx], rm.settings.ub)
        _set_risk_expression(model, model[:rcvar_risk][idx], rm.settings.scale,
                             rm.settings.flag)
    end
    return nothing
end
function set_rm(port::Portfolio2, rm::EVaR2, type::Union{Trad2, RP2},
                obj::ObjectiveFunction, count::Integer, idx::Integer;
                returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
    T = size(returns, 1)
    at = rm.alpha * T

    if !haskey(model, :X)
        @expression(model, X, returns * model[:w])
    end

    if isone(count)
        @variable(model, t_evar)
        @variable(model, z_evar >= 0)
        @variable(model, u_evar[1:T])
        @expression(model, evar_risk, t_evar - z_evar * log(at))
        @constraint(model, sum(u_evar) <= z_evar)
        @constraint(model, [i = 1:T],
                    [-model[:X][i] - t_evar, z_evar, u_evar[i]] ∈ MOI.ExponentialCone())
        _set_rm_risk_upper_bound(obj, type, model, evar_risk, rm.settings.ub)
        _set_risk_expression(model, evar_risk, rm.settings.scale, rm.settings.flag)
    else
        if isone(idx)
            @variable(model, t_evar[1:count])
            @variable(model, z_evar[1:count] >= 0)
            @variable(model, u_evar[1:T, 1:count])
            @expression(model, evar_risk[1:count], zero(AffExpr))
        end
        add_to_expression!(model[:evar_risk][idx], model[:t_evar][idx])
        add_to_expression!(model[:evar_risk][idx], -log(at), model[:z_evar][idx])

        @constraint(model, sum(view(model[:u_evar], :, idx)) <= model[:z_evar][idx])
        @constraint(model, [i = 1:T],
                    [-model[:X][i] - model[:t_evar][idx], model[:z_evar][idx],
                     model[:u_evar][i, idx]] ∈ MOI.ExponentialCone())
        _set_rm_risk_upper_bound(obj, type, model, model[:evar_risk][idx], rm.settings.ub)
        _set_risk_expression(model, model[:evar_risk][idx], rm.settings.scale,
                             rm.settings.flag)
    end
    return nothing
end
function set_rm(port::Portfolio2, rm::RVaR2, type::Union{Trad2, RP2},
                obj::ObjectiveFunction, count::Integer, idx::Integer;
                returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
    T = size(returns, 1)
    iat = inv(rm.alpha * T)
    lnk = (iat^rm.kappa - iat^(-rm.kappa)) / (2 * rm.kappa)
    opk = one(typeof(rm.kappa)) + rm.kappa
    omk = one(typeof(rm.kappa)) - rm.kappa
    ik2 = one(typeof(rm.kappa)) / (2 * rm.kappa)
    ik = one(typeof(rm.kappa)) / rm.kappa
    iopk = one(typeof(rm.kappa)) / opk
    iomk = one(typeof(rm.kappa)) / omk

    if !haskey(model, :X)
        @expression(model, X, returns * model[:w])
    end

    if isone(count)
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
        @constraint(model, -model[:X] .- t_rvar .+ epsilon_rvar .+ omega_rvar .<= 0)
        _set_rm_risk_upper_bound(obj, type, model, rvar_risk, rm.settings.ub)
        _set_risk_expression(model, rvar_risk, rm.settings.scale, rm.settings.flag)
    else
        if isone(idx)
            @variable(model, t_rvar[1:count])
            @variable(model, z_rvar[1:count] >= 0)
            @variable(model, omega_rvar[1:T, 1:count])
            @variable(model, psi_rvar[1:T, 1:count])
            @variable(model, theta_rvar[1:T, 1:count])
            @variable(model, epsilon_rvar[1:T, 1:count])
            @expression(model, rvar_risk[1:count], zero(AffExpr))
        end
        add_to_expression!(model[:rvar_risk][idx], model[:t_rvar][idx])
        add_to_expression!(model[:rvar_risk][idx], lnk, model[:z_rvar][idx])
        add_to_expression!(model[:rvar_risk][idx],
                           sum(view(model[:psi_rvar], :, idx) .+
                               view(model[:theta_rvar], :, idx)))

        @constraint(model, [i = 1:T],
                    [model[:z_rvar][idx] * opk * ik2, model[:psi_rvar][i, idx] * opk * ik,
                     model[:epsilon_rvar][i, idx]] ∈ MOI.PowerCone(iopk))
        @constraint(model, [i = 1:T],
                    [model[:omega_rvar][i, idx] * iomk, model[:theta_rvar][i, idx] * ik,
                     -model[:z_rvar][idx] * ik2] ∈ MOI.PowerCone(omk))
        @constraint(model,
                    -model[:X] .- model[:t_rvar][idx] .+
                    view(model[:epsilon_rvar], :, idx) .+
                    view(model[:omega_rvar], :, idx) .<= 0)
        _set_rm_risk_upper_bound(obj, type, model, model[:rvar_risk][idx], rm.settings.ub)
        _set_risk_expression(model, model[:rvar_risk][idx], rm.settings.scale,
                             rm.settings.flag)
    end
    return nothing
end
function _DaR_setup(model, returns)
    if !haskey(model, :X)
        @expression(model, X, returns * model[:w])
    end
    if !haskey(model, :dd)
        T = size(returns, 1)
        @variable(model, dd[1:(T + 1)])
        @constraint(model, view(dd, 2:(T + 1)) .>= view(dd, 1:T) .- model[:X])
        @constraint(model, view(dd, 2:(T + 1)) .>= 0)
        @constraint(model, dd[1] == 0)
    end
    return nothing
end
function set_rm(port::Portfolio2, rm::MDD2, type::Union{Trad2, RP2}, obj::ObjectiveFunction,
                ::Any, ::Any; returns::AbstractMatrix{<:Real}, kargs...)
    model = port.model
    T = size(returns, 1)

    _DaR_setup(model, returns)
    @variable(model, mdd)
    @expression(model, mdd_risk, mdd)
    @constraint(model, mdd .>= view(model[:dd], 2:(T + 1)))
    _set_rm_risk_upper_bound(obj, type, model, view(model[:dd], 2:(T + 1)), rm.settings.ub)
    _set_risk_expression(model, mdd_risk, rm.settings.scale, rm.settings.flag)
    return nothing
end
function set_rm(port::Portfolio2, rm::ADD2, type::Union{Trad2, RP2}, obj::ObjectiveFunction,
                ::Any, ::Any; returns::AbstractMatrix{<:Real}, kargs...)
    model = port.model
    T = size(returns, 1)

    _DaR_setup(model, returns)
    @expression(model, add_risk, sum(view(model[:dd], 2:(T + 1))) / T)
    _set_rm_risk_upper_bound(obj, type, model, add_risk, rm.settings.ub)
    _set_risk_expression(model, add_risk, rm.settings.scale, rm.settings.flag)
    return nothing
end
function set_rm(port::Portfolio2, rm::UCI2, type::Union{Trad2, RP2}, obj::ObjectiveFunction,
                ::Any, ::Any; returns::AbstractMatrix{<:Real}, kargs...)
    model = port.model
    T = size(returns, 1)

    _DaR_setup(model, returns)
    @variable(model, uci)
    @expression(model, uci_risk, uci / sqrt(T))
    @constraint(model, [uci; view(model[:dd], 2:(T + 1))] ∈ SecondOrderCone())
    _set_rm_risk_upper_bound(obj, type, model, uci_risk, rm.settings.ub)
    _set_risk_expression(model, uci_risk, rm.settings.scale, rm.settings.flag)
    return nothing
end
function set_rm(port::Portfolio2, rm::CDaR2, type::Union{Trad2, RP2},
                obj::ObjectiveFunction, count::Integer, idx::Integer;
                returns::AbstractMatrix{<:Real}, kargs...)
    model = port.model
    T = size(returns, 1)
    iat = inv(rm.alpha * T)

    _DaR_setup(model, returns)
    if isone(count)
        @variable(model, dar)
        @variable(model, z_cdar[1:T] .>= 0)
        @expression(model, cdar_risk, dar + sum(z_cdar) * iat)
        @constraint(model, z_cdar .>= view(model[:dd], 2:(T + 1)) .- dar)
        _set_rm_risk_upper_bound(obj, type, model, cdar_risk, rm.settings.ub)
        _set_risk_expression(model, cdar_risk, rm.settings.scale, rm.settings.flag)
    else
        if isone(idx)
            @variable(model, dar[1:count])
            @variable(model, z_cdar[1:T, 1:count] .>= 0)
            @expression(model, cdar_risk[1:count], zero(AffExpr))
        end
        add_to_expression!(model[:cdar_risk][idx], model[:dar][idx])
        add_to_expression!(model[:cdar_risk][idx], iat, sum(view(model[:z_cdar], :, idx)))
        @constraint(model,
                    view(model[:z_cdar], :, idx) .>=
                    view(model[:dd], 2:(T + 1)) .- model[:dar][idx])
        _set_rm_risk_upper_bound(obj, type, model, model[:cdar_risk][idx], rm.settings.ub)
        _set_risk_expression(model, model[:cdar_risk][idx], rm.settings.scale,
                             rm.settings.flag)
    end
    return nothing
end
function set_rm(port::Portfolio2, rm::EDaR2, type::Union{Trad2, RP2},
                obj::ObjectiveFunction, count::Integer, idx::Integer;
                returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
    T = size(returns, 1)
    at = rm.alpha * T

    _DaR_setup(model, returns)
    if isone(count)
        @variable(model, t_edar)
        @variable(model, z_edar >= 0)
        @variable(model, u_edar[1:T])
        @expression(model, edar_risk, t_edar - z_edar * log(at))
        @constraint(model, sum(u_edar) <= z_edar)
        @constraint(model, [i = 1:T],
                    [model[:dd][i + 1] - t_edar, z_edar, u_edar[i]] ∈ MOI.ExponentialCone())
        _set_rm_risk_upper_bound(obj, type, model, edar_risk, rm.settings.ub)
        _set_risk_expression(model, edar_risk, rm.settings.scale, rm.settings.flag)
    else
        if isone(idx)
            @variable(model, t_edar[1:count])
            @variable(model, z_edar[1:count] >= 0)
            @variable(model, u_edar[1:T, 1:count])
            @expression(model, edar_risk[1:count], zero(AffExpr))
        end
        add_to_expression!(model[:edar_risk][idx], model[:t_edar][idx])
        add_to_expression!(model[:edar_risk][idx], -log(at), model[:z_edar][idx])
        @constraint(model, sum(view(model[:u_edar], :, idx)) <= model[:z_edar][idx])
        @constraint(model, [i = 1:T],
                    [model[:dd][i + 1] - model[:t_edar][idx], model[:z_edar][idx],
                     model[:u_edar][i, idx]] ∈ MOI.ExponentialCone())
        _set_rm_risk_upper_bound(obj, type, model, model[:edar_risk][idx], rm.settings.ub)
        _set_risk_expression(model, model[:edar_risk][idx], rm.settings.scale,
                             rm.settings.flag)
    end
    return nothing
end
function set_rm(port::Portfolio2, rm::RDaR2, type::Union{Trad2, RP2},
                obj::ObjectiveFunction, count::Integer, idx::Integer;
                returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
    T = size(returns, 1)
    iat = inv(rm.alpha * T)
    lnk = (iat^rm.kappa - iat^(-rm.kappa)) / (2 * rm.kappa)
    opk = one(typeof(rm.kappa)) + rm.kappa
    omk = one(typeof(rm.kappa)) - rm.kappa
    ik2 = one(typeof(rm.kappa)) / (2 * rm.kappa)
    ik = one(typeof(rm.kappa)) / rm.kappa
    iopk = one(typeof(rm.kappa)) / opk
    iomk = one(typeof(rm.kappa)) / omk

    _DaR_setup(model, returns)
    if isone(count)
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
        @constraint(model,
                    view(model[:dd], 2:(T + 1)) .- t_rdar .+ epsilon_rdar .+ omega_rdar .<=
                    0)
        _set_rm_risk_upper_bound(obj, type, model, rdar_risk, rm.settings.ub)
        _set_risk_expression(model, rdar_risk, rm.settings.scale, rm.settings.flag)
    else
        if isone(idx)
            @variable(model, t_rdar[1:count])
            @variable(model, z_rdar[1:count] >= 0)
            @variable(model, omega_rdar[1:T, 1:count])
            @variable(model, psi_rdar[1:T, 1:count])
            @variable(model, theta_rdar[1:T, 1:count])
            @variable(model, epsilon_rdar[1:T, 1:count])
            @expression(model, rdar_risk[1:count], zero(AffExpr))
        end
        add_to_expression!(model[:rdar_risk][idx], model[:t_rdar][idx])
        add_to_expression!(model[:rdar_risk][idx], lnk, model[:z_rdar][idx])
        add_to_expression!(model[:rdar_risk][idx],
                           sum(view(model[:psi_rdar], :, idx) .+
                               view(model[:theta_rdar], :, idx)))
        @constraint(model, [i = 1:T],
                    [model[:z_rdar][idx] * opk * ik2, model[:psi_rdar][i, idx] * opk * ik,
                     model[:epsilon_rdar][i, idx]] ∈ MOI.PowerCone(iopk))
        @constraint(model, [i = 1:T],
                    [model[:omega_rdar][i, idx] * iomk, model[:theta_rdar][i, idx] * ik,
                     -model[:z_rdar][idx] * ik2] ∈ MOI.PowerCone(omk))
        @constraint(model,
                    view(model[:dd], 2:(T + 1)) .- model[:t_rdar][idx] .+
                    view(model[:epsilon_rdar], :, idx) .+
                    view(model[:omega_rdar], :, idx) .<= 0)
        _set_rm_risk_upper_bound(obj, type, model, model[:rdar_risk][idx], rm.settings.ub)
        _set_risk_expression(model, model[:rdar_risk][idx], rm.settings.scale,
                             rm.settings.flag)
    end
    return nothing
end
function set_rm(port::Portfolio2, rm::Kurt2, type::Union{Trad2, RP2},
                obj::ObjectiveFunction, count::Integer, idx::Integer; kwargs...)
    model = port.model
    N = size(port.returns, 2)
    kt = if (isnothing(rm.kt) || isempty(rm.kt))
        port.kurt
    else
        rm.kt
    end
    nmax = port.max_num_assets_kurt
    if !iszero(nmax) && N > nmax
        f = port.max_num_assets_kurt_scale
        Nf = f * N
    end

    _sdp(port, obj)

    if isone(count)
        @variable(model, kurt_risk)
        if !iszero(nmax) && N > nmax
            @variable(model, x_kurt[1:Nf])
            @constraint(model, [kurt_risk; x_kurt] ∈ SecondOrderCone())
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
            @constraint(model, [i = 1:Nf], x_kurt[i] == tr(Bi[i] * model[:W]))
        else
            L_2 = port.L_2
            S_2 = port.S_2
            sqrt_sigma_4 = sqrt(S_2 * kt * transpose(S_2))
            @expression(model, zkurt, L_2 * vec(model[:W]))
            @constraint(model, [kurt_risk; sqrt_sigma_4 * zkurt] ∈ SecondOrderCone())
        end
        _set_rm_risk_upper_bound(obj, type, model, kurt_risk, rm.settings.ub)
        _set_risk_expression(model, kurt_risk, rm.settings.scale, rm.settings.flag)
    else
        if isone(idx)
            @variable(model, kurt_risk[1:count])
            if !iszero(nmax) && N > nmax
                @variable(model, x_kurt[1:Nf, 1:count])
            else
                @expression(model, zkurt[1:size(port.L_2, 1), 1:count], zero(AffExpr))
            end
        end
        if !iszero(nmax) && N > nmax
            @constraint(model,
                        [model[:kurt_risk][idx]; view(model[:x_kurt], :, idx)] ∈
                        SecondOrderCone())
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
            @constraint(model, [i = 1:Nf], model[:x_kurt][i, idx] == tr(Bi[i] * model[:W]))
        else
            L_2 = port.L_2
            S_2 = port.S_2
            sqrt_sigma_4 = sqrt(S_2 * kt * transpose(S_2))
            add_to_expression!.(view(model[:zkurt], :, idx), L_2 * vec(model[:W]))
            @constraint(model,
                        [model[:kurt_risk][idx];
                         sqrt_sigma_4 * view(model[:zkurt], :, idx)] ∈ SecondOrderCone())
        end
        _set_rm_risk_upper_bound(obj, type, model, model[:kurt_risk][idx], rm.settings.ub)
        _set_risk_expression(model, model[:kurt_risk][idx], rm.settings.scale,
                             rm.settings.flag)
    end
    return nothing
end
function set_rm(port::Portfolio2, rm::SKurt2, type::Union{Trad2, RP2},
                obj::ObjectiveFunction, count::Integer, idx::Integer; kwargs...)
    model = port.model
    N = size(port.returns, 2)
    kt = if (isnothing(rm.kt) || isempty(rm.kt))
        port.skurt
    else
        rm.kt
    end
    nmax = port.max_num_assets_kurt
    if !iszero(nmax) && N > nmax
        f = port.max_num_assets_kurt_scale
        Nf = f * N
    end

    _sdp(port, obj)

    if isone(count)
        @variable(model, skurt_risk)
        if !iszero(nmax) && N > nmax
            @variable(model, x_skurt[1:Nf])
            @constraint(model, [skurt_risk; x_skurt] ∈ SecondOrderCone())
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
            @constraint(model, [i = 1:Nf], x_skurt[i] == tr(Bi[i] * model[:W]))
        else
            L_2 = port.L_2
            S_2 = port.S_2
            sqrt_sigma_4 = sqrt(S_2 * kt * transpose(S_2))
            @expression(model, zskurt, L_2 * vec(model[:W]))
            @constraint(model, [skurt_risk; sqrt_sigma_4 * zskurt] ∈ SecondOrderCone())
        end
        _set_rm_risk_upper_bound(obj, type, model, skurt_risk, rm.settings.ub)
        _set_risk_expression(model, skurt_risk, rm.settings.scale, rm.settings.flag)
    else
        if isone(idx)
            @variable(model, skurt_risk[1:count])
            if !iszero(nmax) && N > nmax
                @variable(model, x_skurt[1:Nf, 1:count])
            else
                @expression(model, zskurt[1:size(port.L_2, 1), 1:count], zero(AffExpr))
            end
        end
        if !iszero(nmax) && N > nmax
            @constraint(model,
                        [model[:skurt_risk][idx]; view(model[:x_skurt], :, idx)] ∈
                        SecondOrderCone())
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
            @constraint(model, [i = 1:Nf], model[:x_skurt][i, idx] == tr(Bi[i] * model[:W]))
        else
            L_2 = port.L_2
            S_2 = port.S_2
            sqrt_sigma_4 = sqrt(S_2 * kt * transpose(S_2))
            add_to_expression!.(view(model[:zskurt], :, idx), L_2 * vec(model[:W]))

            @constraint(model,
                        [model[:skurt_risk][idx];
                         sqrt_sigma_4 * view(model[:zskurt], :, idx)] ∈ SecondOrderCone())
        end
        _set_rm_risk_upper_bound(obj, type, model, model[:skurt_risk][idx], rm.settings.ub)
        _set_risk_expression(model, model[:skurt_risk][idx], rm.settings.scale,
                             rm.settings.flag)
    end
    return nothing
end
function _owa_setup(model, T)
    if !haskey(model, :owa)
        @variable(model, owa[1:T])
        @constraint(model, model[:X] == owa)
    end
    return nothing
end
function set_rm(port::Portfolio2, rm::GMD2, type::Union{Trad2, RP2}, obj::ObjectiveFunction,
                ::Any, ::Any; returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
    T = size(returns, 1)

    if !haskey(model, :X)
        @expression(model, X, returns * model[:w])
    end

    if !rm.owa.approx
        ovec = range(1; stop = 1, length = T)
        _owa_setup(model, T)
        @variable(model, gmda[1:T])
        @variable(model, gmdb[1:T])
        @expression(model, gmd_risk, sum(gmda .+ gmdb))
        gmd_w = owa_gmd(T)
        @constraint(model,
                    model[:owa] * transpose(gmd_w) .<=
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
        @constraint(model,
                    model[:X] .+ gmd_t .- gmd_nu .+ gmd_eta .-
                    vec(sum(gmd_epsilon; dims = 2)) .== 0)
        @constraint(model, gmd_z .+ gmd_y .== vec(sum(gmd_psi; dims = 1)))
        @constraint(model, [i = 1:M, j = 1:T],
                    [-gmd_z[i] * owa_p[i], gmd_psi[j, i] * owa_p[i] / (owa_p[i] - 1),
                     gmd_epsilon[j, i]] ∈ MOI.PowerCone(1 / owa_p[i]))
    end
    _set_rm_risk_upper_bound(obj, type, model, gmd_risk, rm.settings.ub)
    _set_risk_expression(model, gmd_risk, rm.settings.scale, rm.settings.flag)
    return nothing
end
function set_rm(port::Portfolio2, rm::TG2, type::Union{Trad2, RP2}, obj::ObjectiveFunction,
                count::Integer, idx::Integer; returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
    T = size(returns, 1)

    if !haskey(model, :X)
        @expression(model, X, returns * model[:w])
    end

    alpha = rm.alpha
    a_sim = rm.a_sim
    alpha_i = rm.alpha_i
    if isone(count)
        if !rm.owa.approx
            ovec = range(1; stop = 1, length = T)
            _owa_setup(model, T)
            @variable(model, tga[1:T])
            @variable(model, tgb[1:T])
            @expression(model, tg_risk, sum(tga .+ tgb))
            tg_w = owa_tg(T; alpha_i = alpha_i, alpha = alpha, a_sim = a_sim)
            @constraint(model,
                        model[:owa] * transpose(tg_w) .<=
                        ovec * transpose(tga) + tgb * transpose(ovec))
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
                        tg_s * tg_t - tg_l * sum(tg_nu) +
                        tg_h * sum(tg_eta) +
                        dot(tg_d, tg_y))
            @constraint(model,
                        model[:X] .+ tg_t .- tg_nu .+ tg_eta .-
                        vec(sum(tg_epsilon; dims = 2)) .== 0)
            @constraint(model, tg_z .+ tg_y .== vec(sum(tg_psi; dims = 1)))
            @constraint(model, [i = 1:M, j = 1:T],
                        [-tg_z[i] * owa_p[i], tg_psi[j, i] * owa_p[i] / (owa_p[i] - 1),
                         tg_epsilon[j, i]] ∈ MOI.PowerCone(1 / owa_p[i]))
        end
        _set_rm_risk_upper_bound(obj, type, model, tg_risk, rm.settings.ub)
        _set_risk_expression(model, tg_risk, rm.settings.scale, rm.settings.flag)
    else
        if isone(idx)
            @expression(model, tg_risk[1:count], zero(AffExpr))
            if !rm.owa.approx
                _owa_setup(model, T)
                @variable(model, tga[1:T, 1:count])
                @variable(model, tgb[1:T, 1:count])
            else
                M = length(rm.owa.p)
                @variable(model, tg_t[1:count])
                @variable(model, tg_nu[1:T, 1:count] .>= 0)
                @variable(model, tg_eta[1:T, 1:count] .>= 0)
                @variable(model, tg_epsilon[1:T, 1:M, 1:count])
                @variable(model, tg_psi[1:T, 1:M, 1:count])
                @variable(model, tg_z[1:M, 1:count])
                @variable(model, tg_y[1:M, 1:count] .>= 0)
            end
        end
        if !rm.owa.approx
            ovec = range(1; stop = 1, length = T)
            if !haskey(model, :tga)
                _owa_setup(model, T)
                @variable(model, tga[1:T, 1:count])
                @variable(model, tgb[1:T, 1:count])
            end
            add_to_expression!(model[:tg_risk][idx],
                               sum(view(model[:tga], :, idx) .+ view(model[:tgb], :, idx)))
            tg_w = owa_tg(T; alpha_i = alpha_i, alpha = alpha, a_sim = a_sim)
            @constraint(model,
                        model[:owa] * transpose(tg_w) .<=
                        ovec * transpose(view(model[:tga], :, idx)) +
                        view(model[:tgb], :, idx) * transpose(ovec))
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
            add_to_expression!(model[:tg_risk][idx], tg_s, model[:tg_t][idx])
            add_to_expression!(model[:tg_risk][idx], -tg_l,
                               sum(view(model[:tg_nu], :, idx)))
            add_to_expression!(model[:tg_risk][idx], tg_h,
                               sum(view(model[:tg_eta], :, idx)))
            add_to_expression!(model[:tg_risk][idx], dot(tg_d, view(model[:tg_y], :, idx)))
            @constraint(model,
                        model[:X] .+ model[:tg_t][idx] .- view(model[:tg_nu], :, idx) .+
                        view(model[:tg_eta], :, idx) .-
                        vec(sum(view(model[:tg_epsilon], :, :, idx); dims = 2)) .== 0)
            @constraint(model,
                        model[:tg_z][:, idx] .+ model[:tg_y][:, idx] .==
                        vec(sum(view(model[:tg_psi], :, :, idx); dims = 1)))
            @constraint(model, [i = 1:M, j = 1:T],
                        [-model[:tg_z][i, idx] * owa_p[i],
                         model[:tg_psi][j, i, idx] * owa_p[i] / (owa_p[i] - 1),
                         model[:tg_epsilon][j, i, idx]] ∈ MOI.PowerCone(1 / owa_p[i]))
        end
        _set_rm_risk_upper_bound(obj, type, model, model[:tg_risk][idx], rm.settings.ub)
        _set_risk_expression(model, model[:tg_risk][idx], rm.settings.scale,
                             rm.settings.flag)
    end
    return nothing
end
function set_rm(port::Portfolio2, rm::RTG2, type::Union{Trad2, RP2}, obj::ObjectiveFunction,
                count::Integer, idx::Integer; returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
    T = size(returns, 1)

    if !haskey(model, :X)
        @expression(model, X, returns * model[:w])
    end

    alpha = rm.alpha
    a_sim = rm.a_sim
    alpha_i = rm.alpha_i
    beta = rm.beta
    b_sim = rm.b_sim
    beta_i = rm.beta_i
    if isone(count)
        if !rm.owa.approx
            ovec = range(1; stop = 1, length = T)
            _owa_setup(model, T)
            @variable(model, rtga[1:T])
            @variable(model, rtgb[1:T])
            @expression(model, rtg_risk, sum(rtga .+ rtgb))
            rtg_w = owa_rtg(T; alpha_i = alpha_i, alpha = alpha, a_sim = a_sim,
                            beta_i = beta_i, beta = beta, b_sim = b_sim)
            @constraint(model,
                        model[:owa] * transpose(rtg_w) .<=
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
            @constraint(model,
                        model[:X] .+ rltg_t .- rltg_nu .+ rltg_eta .-
                        vec(sum(rltg_epsilon; dims = 2)) .== 0)
            @constraint(model, rltg_z .+ rltg_y .== vec(sum(rltg_psi; dims = 1)))
            @constraint(model, [i = 1:M, j = 1:T],
                        [-rltg_z[i] * owa_p[i], rltg_psi[j, i] * owa_p[i] / (owa_p[i] - 1),
                         rltg_epsilon[j, i]] ∈ MOI.PowerCone(1 / owa_p[i]))

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
                        -model[:X] .+ rhtg_t .- rhtg_nu .+ rhtg_eta .-
                        vec(sum(rhtg_epsilon; dims = 2)) .== 0)
            @constraint(model, rhtg_z .+ rhtg_y .== vec(sum(rhtg_psi; dims = 1)))
            @constraint(model, [i = 1:M, j = 1:T],
                        [-rhtg_z[i] * owa_p[i], rhtg_psi[j, i] * owa_p[i] / (owa_p[i] - 1),
                         rhtg_epsilon[j, i]] ∈ MOI.PowerCone(1 / owa_p[i]))
        end
        _set_rm_risk_upper_bound(obj, type, model, rtg_risk, rm.settings.ub)
        _set_risk_expression(model, rtg_risk, rm.settings.scale, rm.settings.flag)
    else
        if isone(idx)
            @expression(model, rtg_risk[1:count], zero(AffExpr))
            if !rm.owa.approx
                _owa_setup(model, T)
                @variable(model, rtga[1:T, 1:count])
                @variable(model, rtgb[1:T, 1:count])
            else
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
        end
        if !rm.owa.approx
            ovec = range(1; stop = 1, length = T)
            if !haskey(model, :rtga)
                _owa_setup(model, T)
                @variable(model, rtga[1:T, 1:count])
                @variable(model, rtgb[1:T, 1:count])
            end
            add_to_expression!(model[:rtg_risk][idx],
                               sum(view(model[:rtga], :, idx) .+ view(model[:rtgb], :, idx)))

            rtg_w = owa_rtg(T; alpha_i = alpha_i, alpha = alpha, a_sim = a_sim,
                            beta_i = beta_i, beta = beta, b_sim = b_sim)
            @constraint(model,
                        model[:owa] * transpose(rtg_w) .<=
                        ovec * transpose(view(model[:rtga], :, idx)) +
                        view(model[:rtgb], :, idx) * transpose(ovec))
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

            add_to_expression!(model[:rltg_risk][idx], rltg_s, model[:rltg_t][idx])
            add_to_expression!(model[:rltg_risk][idx], -rltg_l,
                               sum(view(model[:rltg_nu], :, idx)))
            add_to_expression!(model[:rltg_risk][idx], rltg_h,
                               sum(view(model[:rltg_eta], :, idx)))
            add_to_expression!(model[:rltg_risk][idx],
                               dot(rltg_d, view(model[:rltg_y], :, idx)))
            @constraint(model,
                        model[:X] .+ model[:rltg_t][idx] .- view(model[:rltg_nu], :, idx) .+
                        view(model[:rltg_eta], :, idx) .-
                        vec(sum(view(model[:rltg_epsilon], :, :, idx); dims = 2)) .== 0)
            @constraint(model,
                        view(model[:rltg_z], :, idx) .+ view(model[:rltg_y], :, idx) .==
                        vec(sum(view(model[:rltg_psi], :, :, idx); dims = 1)))
            @constraint(model, [i = 1:M, j = 1:T],
                        [-model[:rltg_z][i, idx] * owa_p[i],
                         model[:rltg_psi][j, i, idx] * owa_p[i] / (owa_p[i] - 1),
                         model[:rltg_epsilon][j, i, idx]] ∈ MOI.PowerCone(1 / owa_p[i]))
            rhtg_w = -owa_tg(T; alpha_i = beta_i, alpha = beta, a_sim = b_sim)
            rhtg_s = sum(rhtg_w)
            rhtg_l = minimum(rhtg_w)
            rhtg_h = maximum(rhtg_w)
            rhtg_d = [norm(rhtg_w, p) for p ∈ owa_p]
            add_to_expression!(model[:rhtg_risk][idx], rhtg_s, model[:rhtg_t][idx])
            add_to_expression!(model[:rhtg_risk][idx], -rhtg_l,
                               sum(view(model[:rhtg_nu], :, idx)))
            add_to_expression!(model[:rhtg_risk][idx], rhtg_h,
                               sum(view(model[:rhtg_eta], :, idx)))
            add_to_expression!(model[:rhtg_risk][idx],
                               dot(rhtg_d, view(model[:rhtg_y], :, idx)))
            add_to_expression!(model[:rtg_risk][idx], model[:rltg_risk][idx])
            add_to_expression!(model[:rtg_risk][idx], model[:rhtg_risk][idx])
            @constraint(model,
                        -model[:X] .+ model[:rhtg_t][idx] .-
                        view(model[:rhtg_nu], :, idx) .+ view(model[:rhtg_eta], :, idx) .-
                        vec(sum(model[:rhtg_epsilon][:, :, idx]; dims = 2)) .== 0)
            @constraint(model,
                        view(model[:rhtg_z], :, idx) .+ view(model[:rhtg_y], :, idx) .==
                        vec(sum(view(model[:rhtg_psi], :, :, idx); dims = 1)))
            @constraint(model, [i = 1:M, j = 1:T],
                        [-model[:rhtg_z][i, idx] * owa_p[i],
                         model[:rhtg_psi][j, i, idx] * owa_p[i] / (owa_p[i] - 1),
                         model[:rhtg_epsilon][j, i, idx]] ∈ MOI.PowerCone(1 / owa_p[i]))
        end
        _set_rm_risk_upper_bound(obj, type, model, model[:rtg_risk][idx], rm.settings.ub)
        _set_risk_expression(model, model[:rtg_risk][idx], rm.settings.scale,
                             rm.settings.flag)
    end
    return nothing
end
function set_rm(port::Portfolio2, rm::OWA2, type::Union{Trad2, RP2}, obj::ObjectiveFunction,
                count::Integer, idx::Integer; returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
    T = size(returns, 1)

    if !haskey(model, :X)
        @expression(model, X, returns * model[:w])
    end

    if isone(count)
        if !rm.owa.approx
            ovec = range(1; stop = 1, length = T)
            _owa_setup(model, T)
            @variable(model, owa_a[1:T])
            @variable(model, owa_b[1:T])
            @expression(model, owa_risk, sum(owa_a .+ owa_b))
            owa_w = (isnothing(rm.w) || isempty(rm.w)) ? owa_gmd(T) : rm.w
            @constraint(model,
                        model[:owa] * transpose(owa_w) .<=
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
            @constraint(model,
                        model[:X] .+ owa_t .- owa_nu .+ owa_eta .-
                        vec(sum(owa_epsilon; dims = 2)) .== 0)
            @constraint(model, owa_z .+ owa_y .== vec(sum(owa_psi; dims = 1)))
            @constraint(model, [i = 1:M, j = 1:T],
                        [-owa_z[i] * owa_p[i], owa_psi[j, i] * owa_p[i] / (owa_p[i] - 1),
                         owa_epsilon[j, i]] ∈ MOI.PowerCone(1 / owa_p[i]))
        end
        _set_rm_risk_upper_bound(obj, type, model, owa_risk, rm.settings.ub)
        _set_risk_expression(model, owa_risk, rm.settings.scale, rm.settings.flag)
    else
        if isone(idx)
            @expression(model, owa_risk[1:count], zero(AffExpr))
            if !rm.owa.approx
                _owa_setup(model, T)
                @variable(model, owa_a[1:T, 1:count])
                @variable(model, owa_b[1:T, 1:count])
            else
                M = length(rm.owa.p)
                @variable(model, owa_t[1:count])
                @variable(model, owa_nu[1:T, 1:count] .>= 0)
                @variable(model, owa_eta[1:T, 1:count] .>= 0)
                @variable(model, owa_epsilon[1:T, 1:M, 1:count])
                @variable(model, owa_psi[1:T, 1:M, 1:count])
                @variable(model, owa_z[1:M, 1:count])
                @variable(model, owa_y[1:M, 1:count] .>= 0)
            end
        end
        if !rm.owa.approx
            ovec = range(1; stop = 1, length = T)
            if !haskey(model, :owa_a)
                _owa_setup(model, T)
                @variable(model, owa_a[1:T, 1:count])
                @variable(model, owa_b[1:T, 1:count])
            end

            add_to_expression!(model[:owa_risk][idx],
                               sum(view(model[:owa_a], :, idx) .+
                                   view(model[:owa_b], :, idx)))
            owa_w = (isnothing(rm.w) || isempty(rm.w)) ? owa_gmd(T) : rm.w
            @constraint(model,
                        model[:owa] * transpose(owa_w) .<=
                        ovec * transpose(view(model[:owa_a], :, idx)) +
                        view(model[:owa_b], :, idx) * transpose(ovec))
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

            add_to_expression!(model[:owa_risk][idx], owa_s, model[:owa_t][idx])
            add_to_expression!(model[:owa_risk][idx], -owa_l,
                               sum(view(model[:owa_nu], :, idx)))
            add_to_expression!(model[:owa_risk][idx], owa_h,
                               sum(view(model[:owa_eta], :, idx)))
            add_to_expression!(model[:owa_risk][idx],
                               dot(owa_d, view(model[:owa_y], :, idx)))

            @constraint(model,
                        model[:X] .+ model[:owa_t][idx] .- view(model[:owa_nu], :, idx) .+
                        view(model[:owa_eta], :, idx) .-
                        vec(sum(view(model[:owa_epsilon], :, :, idx); dims = 2)) .== 0)
            @constraint(model,
                        view(model[:owa_z], :, idx) .+ view(model[:owa_y], :, idx) .==
                        vec(sum(view(model[:owa_psi], :, :, idx); dims = 1)))
            @constraint(model, [i = 1:M, j = 1:T],
                        [-model[:owa_z][i, idx] * owa_p[i],
                         model[:owa_psi][j, i, idx] * owa_p[i] / (owa_p[i] - 1),
                         model[:owa_epsilon][j, i, idx]] ∈ MOI.PowerCone(1 / owa_p[i]))
        end
        _set_rm_risk_upper_bound(obj, type, model, model[:owa_risk][idx], rm.settings.ub)
        _set_risk_expression(model, model[:owa_risk][idx], rm.settings.scale,
                             rm.settings.flag)
    end
    return nothing
end
function set_rm(port::Portfolio2, rm::DVar2, type::Union{Trad2, RP2},
                obj::ObjectiveFunction, ::Any, ::Any; returns::AbstractMatrix{<:Real},
                kwargs...)
    model = port.model
    T = size(returns, 1)

    if !haskey(model, :X)
        @expression(model, X, returns * model[:w])
    end

    ovec = range(1; stop = 1, length = T)
    @variable(model, Dt[1:T, 1:T], Symmetric)
    @expression(model, Dx, model[:X] * transpose(ovec) - ovec * transpose(model[:X]))
    @constraint(model, [i = 1:T, j = i:T], [Dt[i, j]; Dx[i, j]] in MOI.NormOneCone(2))
    dt = vec(Dt)
    iT2 = inv(T^2)
    @expression(model, dvar_risk, iT2 * (dot(dt, dt) + iT2 * dot(ovec, Dt, ovec)^2))
    _set_rm_risk_upper_bound(obj, type, model, dvar_risk, rm.settings.ub)
    _set_risk_expression(model, dvar_risk, rm.settings.scale, rm.settings.flag)
    return nothing
end
function set_rm(port::Portfolio2, rm::Skew2, type::Union{Trad2, RP2},
                obj::ObjectiveFunction, ::Any, ::Any; kwargs...)
    model = port.model

    G = real(sqrt(port.V))
    @variable(model, t_skew)
    @constraint(model, [t_skew; G * model[:w]] ∈ SecondOrderCone())
    @expression(model, skew_risk, t_skew^2)
    _set_rm_risk_upper_bound(obj, type, model, t_skew, rm.settings.ub)
    _set_risk_expression(model, skew_risk, rm.settings.scale, rm.settings.flag)
    return nothing
end
function set_rm(port::Portfolio2, rm::SSkew2, type::Union{Trad2, RP2},
                obj::ObjectiveFunction, ::Any, ::Any; kwargs...)
    model = port.model

    G = real(sqrt(port.SV))
    @variable(model, t_sskew)
    @constraint(model, [t_sskew; G * model[:w]] ∈ SecondOrderCone())
    @expression(model, sskew_risk, t_sskew^2)
    _set_rm_risk_upper_bound(obj, type, model, t_sskew, rm.settings.ub)
    _set_risk_expression(model, sskew_risk, rm.settings.scale, rm.settings.flag)
    return nothing
end
function risk_constraints(port, obj, type::Union{Trad2, RP2}, rm, mu, sigma, returns,
                          kelly_approx_idx = nothing)
    if !isa(rm, AbstractVector)
        rm = (rm,)
    end
    for rv ∈ rm
        if !isa(rv, AbstractVector)
            rv = (rv,)
        end
        count = length(rv)
        for (i, r) ∈ enumerate(rv)
            set_rm(port, r, type, obj, count, i; mu = mu, sigma = sigma, returns = returns,
                   kelly_approx_idx = kelly_approx_idx)
        end
    end
    return nothing
end
function _return_bounds(::Any, model, mu_l::Real)
    if isfinite(mu_l)
        @constraint(model, model[:ret] >= mu_l)
    end
    return nothing
end
function _return_bounds(::SR, model, mu_l::Real)
    if isfinite(mu_l)
        @constraint(model, model[:ret] >= mu_l * model[:k])
    end
    return nothing
end
function set_returns(obj::Any, ::NoKelly, model, mu_l::Real; mu::AbstractVector, kwargs...)
    if !isempty(mu)
        @expression(model, ret, dot(mu, model[:w]))
        _return_bounds(obj, model, mu_l)
    end
    return nothing
end
function set_returns(obj::SR, ::NoKelly, model, mu_l::Real; mu::AbstractVector, kwargs...)
    if !isempty(mu)
        @expression(model, ret, dot(mu, model[:w]))
        @constraint(model, ret - obj.rf * model[:k] == 1)
        _return_bounds(obj, model, mu_l)
    end
    return nothing
end
function set_returns(obj::Any, kelly::AKelly, model, mu_l::Real; mu::AbstractVector,
                     kelly_approx_idx::Union{AbstractVector{<:Integer}, Nothing} = nothing,
                     network_method::NetworkMethods2, sigma::AbstractMatrix, kwargs...)
    if !isempty(mu)
        if isnothing(kelly_approx_idx) ||
           isempty(kelly_approx_idx) ||
           iszero(kelly_approx_idx[1])
            if !haskey(model, :sd_risk)
                _sd_risk(network_method, kelly.formulation, model, sigma, 1, 1)
            end
            @expression(model, ret, dot(mu, model[:w]) - 0.5 * model[:sd_risk])
        else
            @expression(model, ret,
                        dot(mu, model[:w]) - 0.5 * model[:sd_risk][kelly_approx_idx[1]])
        end
        _return_bounds(obj, model, mu_l)
    end
    return nothing
end
function set_returns(obj::SR, kelly::AKelly, model, mu_l::Real; mu::AbstractVector,
                     kelly_approx_idx::AbstractVector{<:Integer},
                     network_method::NetworkMethods2, sigma::AbstractMatrix, kwargs...)
    if !isempty(mu)
        @variable(model, tapprox_kelly)
        @constraint(model, model[:risk] <= 1)
        @expression(model, ret, dot(mu, model[:w]) - 0.5 * tapprox_kelly)
        if isnothing(kelly_approx_idx) ||
           isempty(kelly_approx_idx) ||
           iszero(kelly_approx_idx[1])
            if !haskey(model, :sd_risk)
                _sd_risk(network_method, kelly.formulation, model, sigma, 1, 1)
            end
            _dev_setup(network_method, model)
            @constraint(model,
                        [model[:k] + tapprox_kelly
                         2 * model[:dev]
                         model[:k] - tapprox_kelly] ∈ SecondOrderCone())
        else
            _dev_setup(network_method, model, kelly_approx_idx[1])
            @constraint(model,
                        [model[:k] + tapprox_kelly
                         2 * model[:dev][kelly_approx_idx[1]]
                         model[:k] - tapprox_kelly] ∈ SecondOrderCone())
        end
        _return_bounds(obj, model, mu_l)
    end
    return nothing
end
function set_returns(obj::Any, ::EKelly, model, mu_l::Real; mu::AbstractVector,
                     returns::AbstractMatrix, kwargs...)
    if !isempty(mu)
        T = size(returns, 1)
        @variable(model, texact_kelly[1:T])
        @expression(model, ret, sum(texact_kelly) / T)
        @expression(model, kret, 1 .+ returns * model[:w])
        @constraint(model, [i = 1:T], [texact_kelly[i], 1, kret[i]] ∈ MOI.ExponentialCone())
        _return_bounds(obj, model, mu_l)
    end
    return nothing
end
function set_returns(obj::SR, ::EKelly, model, mu_l::Real; mu::AbstractVector,
                     returns::AbstractMatrix, kwargs...)
    if !isempty(mu)
        T = size(returns, 1)
        @variable(model, texact_kelly[1:T])
        @expression(model, ret, sum(texact_kelly) / T - obj.rf * model[:k])
        @expression(model, kret, model[:k] .+ returns * model[:w])
        @constraint(model, [i = 1:T],
                    [texact_kelly[i], model[:k], kret[i]] ∈ MOI.ExponentialCone())
        @constraint(model, model[:risk] <= 1)
        _return_bounds(obj, model, mu_l)
    end
    return nothing
end
function return_constraints(port, obj, kelly, class::Classic2, mu, sigma, returns,
                            kelly_approx_idx)
    set_returns(obj, kelly, port.model, port.mu_l; mu = mu, sigma = sigma,
                returns = returns, kelly_approx_idx = kelly_approx_idx,
                network_method = port.network_method)
    return nothing
end
function return_constraints(port, obj, ::Any, ::Any, mu, sigma, returns, ::Any)
    set_returns(obj, NoKelly(), port.model, port.mu_l; mu = mu, sigma = sigma,
                returns = returns, network_method = port.network_method)
    return nothing
end
function setup_tracking_err_constraints(::NoTracking, args...)
    return nothing
end
function _tracking_err_constraints(::Any, model, returns, tracking_err, benchmark)
    T = size(returns, 1)
    @variable(model, t_track_err >= 0)
    @expression(model, track_err, returns * model[:w] .- benchmark)
    @constraint(model, [t_track_err; track_err] ∈ SecondOrderCone())
    @constraint(model, t_track_err <= tracking_err * sqrt(T - 1))
    return nothing
end
function _tracking_err_constraints(::SR, model, returns, tracking_err, benchmark)
    T = size(returns, 1)
    @variable(model, t_track_err >= 0)
    @expression(model, track_err, returns * model[:w] .- benchmark * model[:k])
    @constraint(model, [t_track_err; track_err] ∈ SecondOrderCone())
    @constraint(model, t_track_err <= tracking_err * model[:k] * sqrt(T - 1))
    return nothing
end
function tracking_err_constraints(args...)
    return nothing
end
function tracking_err_constraints(tracking_err::TrackWeight, port, returns, obj)
    if !(isempty(isempty(tracking_err.w)) || isinf(tracking_err.err))
        _tracking_err_constraints(obj, port.model, returns, tracking_err.err,
                                  returns * tracking_err.w)
    end
    return nothing
end
function tracking_err_constraints(tracking_err::TrackRet, port, returns, obj)
    if !(isempty(isempty(tracking_err.w)) || isinf(tracking_err.err))
        _tracking_err_constraints(obj, port.model, returns, tracking_err.err,
                                  tracking_err.w)
    end
    return nothing
end
function _turnover_constraints(::Any, model, turnover)
    N = length(turnover.w)
    @variable(model, t_turnov[1:N] >= 0)
    @expression(model, turnov, model[:w] .- turnover.w)
    @constraint(model, [i = 1:N], [t_turnov[i]; turnov[i]] ∈ MOI.NormOneCone(2))
    @constraint(model, t_turnov .<= turnover.val)
    return nothing
end
function _turnover_constraints(::SR, model, turnover)
    N = length(turnover.w)
    @variable(model, t_turnov[1:N] >= 0)
    @expression(model, turnov, model[:w] .- turnover.w * model[:k])
    @constraint(model, [i = 1:N], [t_turnov[i]; turnov[i]] ∈ MOI.NormOneCone(2))
    @constraint(model, t_turnov .<= turnover.val * model[:k])
    return nothing
end
function turnover_constraints(turnover::NoTR, ::Any, ::Any)
    return nothing
end
function turnover_constraints(turnover::TR, port, obj)
    if !(isa(turnover.val, Real) && isinf(turnover.val) ||
         isa(turnover.val, AbstractVector) && isempty(turnover.val) ||
         isempty(turnover.w))
        _turnover_constraints(obj, port.model, turnover)
    end
    return nothing
end
function _rebalance_constraints(::Any, model, rebalance)
    N = length(rebalance.w)
    @variable(model, t_rebal[1:N] >= 0)
    @expression(model, rebal, model[:w] .- rebalance.w)
    @constraint(model, [i = 1:N], [t_rebal[i]; rebal[i]] ∈ MOI.NormOneCone(2))
    @expression(model, sum_t_rebal, sum(rebalance.val .* t_rebal))
    return nothing
end
function _rebalance_constraints(::SR, model, rebalance)
    N = length(rebalance.w)
    @variable(model, t_rebal[1:N] >= 0)
    @expression(model, rebal, model[:w] .- rebalance.w * model[:k])
    @constraint(model, [i = 1:N], [t_rebal[i]; rebal[i]] ∈ MOI.NormOneCone(2))
    @expression(model, sum_t_rebal, sum(rebalance.val .* t_rebal))
    return nothing
end
function rebalance_constraints(turnover::NoTR, ::Any, ::Any)
    return nothing
end
function rebalance_constraints(rebalance::TR, port, obj)
    if !(isa(rebalance.val, Real) && (isinf(rebalance.val) || iszero(rebalance.val)) ||
         isa(rebalance.val, AbstractVector) && isempty(rebalance.val) ||
         isempty(rebalance.w))
        _rebalance_constraints(obj, port.model, port.rebalance)
    end
    return nothing
end
function _objective(::Trad2, ::SR, ::Classic2, ::Union{AKelly, EKelly}, model, p)
    @objective(model, Max, model[:ret] - p)
    return nothing
end
function _objective(::Trad2, ::Union{SR, MinRisk}, ::Any, ::Any, model, p)
    @objective(model, Min, model[:risk] + p)
    return nothing
end
function _objective(::WC2, obj::SR, ::Any, ::Any, model, p)
    @objective(model, Max, model[:ret] - obj.rf * model[:k] - p)
    return nothing
end
function _objective(::WC2, ::MinRisk, ::Any, ::Any, model, p)
    @objective(model, Min, model[:risk] + p)
    return nothing
end
function _objective(::Any, obj::Util, ::Any, ::Any, model, p)
    @objective(model, Max, model[:ret] - obj.l * model[:risk] - p)
    return nothing
end
function _objective(::Any, obj::MaxRet, ::Any, ::Any, model, p)
    @objective(model, Max, model[:ret] - p)
    return nothing
end
function objective_function(port, obj, ::Trad2, class, kelly)
    npf = zero(eltype(port.returns))
    if haskey(port.model, :network_penalty)
        npf = port.model[:network_penalty]
    end
    rbf = zero(eltype(port.returns))
    if haskey(port.model, :sum_t_rebal)
        npf = port.model[:sum_t_rebal]
    end
    _objective(Trad2(), obj, class, kelly, port.model, npf + rbf)
    return nothing
end
function objective_function(port, obj, type::WC2, ::Any, ::Any)
    rbf = zero(eltype(port.returns))
    if haskey(port.model, :sum_t_rebal)
        npf = port.model[:sum_t_rebal]
    end
    _objective(type, obj, nothing, nothing, port.model, rbf)
    return nothing
end
function _cleanup_weights(port, ::SR, ::Union{Trad2, WC2}, ::Any)
    val_k = value(port.model[:k])
    val_k = val_k > 0 ? val_k : 1
    weights = value.(port.model[:w]) / val_k
    short = port.short
    sum_short_long = port.sum_short_long
    if short == false
        sum_w = sum(abs.(weights))
        sum_w = sum_w > eps() ? sum_w : 1
        weights .= abs.(weights) / sum_w * sum_short_long
    end
    return weights
end
function _cleanup_weights(port, ::Any, ::Union{Trad2, WC2}, ::Any)
    weights = value.(port.model[:w])
    short = port.short
    sum_short_long = port.sum_short_long
    if short == false
        sum_w = sum(abs.(weights))
        sum_w = sum_w > eps() ? sum_w : 1
        weights .= abs.(weights) / sum_w * sum_short_long
    end
    return weights
end
function _cleanup_weights(port, ::Any, ::RP2, ::FC2)
    weights = value.(port.model[:w])
    sum_w = value(port.model[:k])
    sum_w = sum_w > eps() ? sum_w : 1
    weights .= weights / sum_w
    return weights
end
function _cleanup_weights(port, ::Any, ::RP2, ::Any)
    weights = value.(port.model[:w])
    sum_w = sum(abs.(weights))
    sum_w = sum_w > eps() ? sum_w : 1
    weights .= abs.(weights) / sum_w
    return weights
end
function _cleanup_weights(port, ::Any, ::RRP2, ::Any)
    weights = value.(port.model[:w])
    sum_w = sum(abs.(weights))
    sum_w = sum_w > eps() ? sum_w : 1
    weights .= abs.(weights) / sum_w
    return weights
end
function convex_optimisation(port, obj, type, class)
    solvers = port.solvers
    model = port.model

    term_status = termination_status(model)
    solvers_tried = Dict()

    fail = true
    strtype = "_" * String(type)
    for (key, val) ∈ solvers
        key = Symbol(String(key) * strtype)

        if haskey(val, :solver)
            set_optimizer(model, val[:solver])
        end

        if haskey(val, :params)
            for (attribute, value) ∈ val[:params]
                set_attribute(model, attribute, value)
            end
        end

        try
            JuMP.optimize!(model)
        catch jump_error
            push!(solvers_tried, key => Dict(:jump_error => jump_error))
            continue
        end

        term_status = termination_status(model)
        all_finite_weights = all(isfinite.(value.(model[:w])))
        all_non_zero_weights = !all(isapprox.(abs.(value.(model[:w])),
                                              zero(eltype(port.returns))))

        if term_status ∈ ValidTermination && all_finite_weights && all_non_zero_weights
            fail = false
            break
        end

        weights = _cleanup_weights(port, obj, type, class)

        push!(solvers_tried,
              key => Dict(:objective_val => objective_value(model),
                          :term_status => term_status,
                          :params => haskey(val, :params) ? val[:params] : missing,
                          :finite_weights => all_finite_weights,
                          :nonzero_weights => all_non_zero_weights,
                          :portfolio => DataFrame(; tickers = port.assets,
                                                  weights = weights)))
    end

    return if fail
        @warn("Model could not be optimised satisfactorily.\nSolvers: $solvers_tried.")
        port.fail = solvers_tried
        port.optimal[Symbol(type)] = DataFrame()
    else
        isempty(solvers_tried) ? port.fail = Dict() : port.fail = solvers_tried
        weights = _cleanup_weights(port, obj, type, class)
        port.optimal[Symbol(type)] = DataFrame(; tickers = port.assets, weights = weights)
    end
end
function _optimise!(::Trad2, port::Portfolio2, rm::Union{AbstractVector, <:TradRiskMeasure},
                    obj::ObjectiveFunction, kelly::RetType, class::PortClass,
                    w_ini::AbstractVector, str_names::Bool)
    mu, sigma, returns = mu_sigma_returns_class(port, class)
    model = port.model = JuMP.Model()
    set_string_names_on_creation(model, str_names)
    initial_w(port, w_ini)
    set_sr_k(obj, model)
    kelly_approx_idx = Int[]
    risk_constraints(port, obj, Trad2(), rm, mu, sigma, returns, kelly_approx_idx)
    return_constraints(port, obj, kelly, class, mu, sigma, returns, kelly_approx_idx)
    linear_constraints(port, obj)
    centrality_constraints(port, obj)
    weight_constraints(port, obj)
    num_assets_constraints(port, obj)
    network_constraints(port.network_method, port, obj, Trad2())
    tracking_err_constraints(port.tracking_err, port, returns, obj)
    turnover_constraints(port.turnover, port, obj)
    rebalance_constraints(port.rebalance, port, obj)
    objective_function(port, obj, Trad2(), class, kelly)
    return convex_optimisation(port, obj, Trad2(), class)
end
function _wc_return_constraints(::WCBox, port)
    model = port.model
    N = length(port.mu)
    @variable(model, abs_w[1:N])
    @constraint(model, [i = 1:N], [abs_w[i]; model[:w][i]] ∈ MOI.NormOneCone(2))
    @expression(model, ret, dot(port.mu, model[:w]) - dot(port.d_mu, abs_w))
    return nothing
end
function _wc_return_constraints(::WCEllipse, port)
    model = port.model
    G = sqrt(port.cov_mu)
    @expression(model, x_gw, G * model[:w])
    @variable(model, t_gw)
    @constraint(model, [t_gw; x_gw] ∈ SecondOrderCone())
    @expression(model, ret, dot(port.mu, model[:w]) - port.k_mu * t_gw)
    return nothing
end
function _wc_return_constraints(::NoWC, port)
    @expression(port.model, ret, dot(port.mu, port.model[:w]))
    return nothing
end
function _wc_risk_constraints(::WCBox, port, obj)
    _sdp(port, obj)
    model = port.model
    N = size(port.returns, 2)
    @variable(model, Au[1:N, 1:N] .>= 0, Symmetric)
    @variable(model, Al[1:N, 1:N] .>= 0, Symmetric)
    @constraint(model, Au .- Al .== model[:W])
    @expression(model, risk, tr(Au * port.cov_u) - tr(Al * port.cov_l))
    return nothing
end
function _wc_risk_constraints(::WCEllipse, port, obj)
    _sdp(port, obj)
    sigma = port.cov
    G_sigma = sqrt(port.cov_sigma)
    model = port.model
    N = size(port.returns, 2)
    @variable(model, E[1:N, 1:N], Symmetric)
    @constraint(model, E ∈ PSDCone())
    @expression(model, W_p_E, model[:W] .+ E)
    @expression(model, x_ge, G_sigma * vec(W_p_E))
    @variable(model, t_ge)
    @constraint(model, [t_ge; x_ge] ∈ SecondOrderCone())
    @expression(model, risk, tr(sigma * W_p_E) + port.k_sigma * t_ge)
    return nothing
end
function _wc_risk_constraints(type::NoWC, port, ::Any)
    _sd_risk(port.network_method, type.formulation, port.model, port.cov, 1, 1)
    @expression(port.model, risk, port.model[:sd_risk])
    return nothing
end
function _wc_sharpe_constraints(obj::SR, port)
    @constraint(port.model, port.model[:risk] <= 1)
    return nothing
end
function _wc_sharpe_constraints(::Any, ::Any)
    return nothing
end
function wc_constraints(port, obj, type)
    _wc_return_constraints(type.mu, port)
    _wc_risk_constraints(type.cov, port, obj)
    _wc_sharpe_constraints(obj, port)
    return nothing
end
function _optimise!(type::WC2, port::Portfolio2,
                    rm::Union{AbstractVector, <:TradRiskMeasure}, obj::ObjectiveFunction,
                    ::Any, ::Any, w_ini::AbstractVector, str_names::Bool)
    port.model = JuMP.Model()
    model = port.model
    set_string_names_on_creation(model, str_names)
    initial_w(port, w_ini)
    set_sr_k(obj, model)
    network_constraints(port.network_method, port, obj, type)
    wc_constraints(port, obj, type)
    linear_constraints(port, obj)
    centrality_constraints(port, obj)
    weight_constraints(port, obj)
    num_assets_constraints(port, obj)
    tracking_err_constraints(port.tracking_err, port, port.returns, obj)
    turnover_constraints(port.turnover, port, obj)
    rebalance_constraints(port.rebalance, port, obj)
    objective_function(port, obj, type, nothing, nothing)
    return convex_optimisation(port, obj, type, nothing)
end
function _rebuild_B(B::DataFrame, ::Any, ::Any)
    return Matrix(B[!, setdiff(names(B), ("tickers", "const"))])
end
function _rebuild_B(B::DataFrame, factors::AbstractMatrix,
                    regression::DimensionReductionReg)
    X = transpose(factors)
    X_std = StatsBase.standardize(StatsBase.ZScoreTransform, X; dims = 2)
    model = fit(regression.pcr, X_std)
    Vp = projection(model)
    sdev = if isnothing(regression.std_w)
        vec(std(regression.ve, X; dims = 2))
    else
        vec(std(regression.ve, X, regression.std_w; dims = 2))
    end
    return transpose(pinv(Vp) * transpose(B .* transpose(sdev)))
end
function _factors_b1_b2_b3(B::DataFrame, factors::AbstractMatrix,
                           regression::RegressionType)
    B = _rebuild_B(B, regression, factors)
    b1 = pinv(transpose(B))
    b2 = pinv(transpose(nullspace(transpose(B))))
    b3 = pinv(transpose(b2))
    return b1, b2, b3, B
end
function _rp_class_constraints(::Any, port)
    model = port.model
    if isempty(port.risk_budget)
        port.risk_budget = ()
    elseif !isapprox(sum(port.risk_budget), one(eltype(port.returns)))
        port.risk_budget ./= sum(port.risk_budget)
    end
    rb = port.risk_budget
    N = length(rb)
    @variable(model, w[1:N])
    @variable(model, log_w[1:N])
    @constraint(model, dot(rb, log_w) >= 1)
    @constraint(model, [i = 1:N], [log_w[i], 1, w[i]] ∈ MOI.ExponentialCone())
    @constraint(model, w .>= 0)
    return nothing
end
function _rp_class_constraints(class::FC2, port)
    if class.nullflag
        b1, b2 = _factors_b1_b2_b3(port.loadings, port.f_returns, port.loadings_opt)[1:2]
        N_f = size(b1, 2)
        @variable(model, w1[1:N_f])
        @variable(model, w2[1:(N - N_f)])
        @expression(model, w, b1 * w1 + b2 * w2)
    else
        b1 = _factors_b1_b2_b3(port.loadings, port.f_returns, port.loadings_opt)[1]
        N_f = size(b1, 2)
        @variable(model, w1[1:N_f])
        @expression(model, w, b1 * w1)
    end

    rb = port.f_risk_budget
    if isempty(rb) || length(rb) != N_f
        rb = port.f_risk_budget = fill(1 / N_f, N_f)
    elseif !isapprox(sum(port.f_risk_budget), one(eltype(port.returns)))
        port.f_risk_budget ./= sum(port.f_risk_budget)
    end

    @variable(model, log_w[1:N_f])
    @constraint(model, dot(rb, log_w) >= 1)
    @constraint(model, [i = 1:N_f], [log_w[i], 1, model[:w1][i]] ∈ MOI.ExponentialCone())
    return nothing
end
function rp_constraints(port, class)
    model = port.model
    _rp_class_constraints(class, port)
    @variable(model, k)
    @constraint(model, sum(model[:w]) == model[:k])
    return nothing
end
function _optimise!(type::RP2, port::Portfolio2,
                    rm::Union{AbstractVector, <:TradRiskMeasure}, ::Any, ::Any,
                    class::Union{Classic2, FM2, FC2}, w_ini::AbstractVector,
                    str_names::Bool)
    mu, sigma, returns = mu_sigma_returns_class(port, class)
    port.model = JuMP.Model()
    model = port.model
    set_string_names_on_creation(model, str_names)
    rp_constraints(port, class)
    initial_w(port, w_ini)
    risk_constraints(port, MinRisk(), RP2(), rm, mu, sigma, returns)
    set_returns(nothing, NoKelly(), port.model, port.mu_l; mu = mu)
    linear_constraints(port, MinRisk())
    @objective(model, Min, model[:risk])
    return convex_optimisation(port, nothing, RP2(), class)
end
function _rrp_ver_constraints(::NoRRP, model, sigma)
    G = sqrt(sigma)
    @constraint(model, [model[:psi]; G * model[:w]] ∈ SecondOrderCone())
    return nothing
end
function _rrp_ver_constraints(::RegRRP, model, sigma)
    G = sqrt(sigma)
    @variable(model, rho)
    @constraint(model, [2 * model[:psi]; 2 * G * model[:w]; -2 * rho] ∈ SecondOrderCone())
    @constraint(model, [rho; G * model[:w]] ∈ SecondOrderCone())
    return nothing
end
function _rrp_ver_constraints(version::RegPenRRP, model, sigma)
    G = sqrt(sigma)
    @variable(model, rho)
    @constraint(model, [2 * model[:psi]; 2 * G * model[:w]; -2 * rho] ∈ SecondOrderCone())
    theta = Diagonal(sqrt.(diag(sigma)))
    @constraint(model, [rho; sqrt(version.penalty) * theta * model[:w]] ∈ SecondOrderCone())
    return nothing
end
function _rrp_constraints(type::RRP2, port, sigma)
    model = port.model
    N = size(port.returns, 2)
    @variable(model, psi)
    @variable(model, gamma >= 0)
    @variable(model, zeta[1:N] .>= 0)
    @expression(model, risk, psi - gamma)
    # RRP constraints.
    @constraint(model, zeta .== sigma * model[:w])
    @constraint(model, sum(model[:w]) == 1)
    @constraint(model, model[:w] >= 0)
    @constraint(model, [i = 1:N],
                [model[:w][i] + zeta[i]
                 2 * gamma * sqrt(port.risk_budget[i])
                 model[:w][i] - zeta[i]] ∈ SecondOrderCone())
    _rrp_ver_constraints(type.version, model, sigma)
    return nothing
end
function rrp_constraints(type::RRP2, port, sigma)
    model = port.model
    @variable(model, k)
    if isempty(port.risk_budget)
        port.risk_budget = ()
    elseif !isapprox(sum(port.risk_budget), one(eltype(port.returns)))
        port.risk_budget ./= sum(port.risk_budget)
    end

    _sd_risk(NoNtwk(), SOCSD(), model, sigma, 1, 1)
    _set_sd_risk_upper_bound(nothing, MinRisk(), type, model, Inf, 1, 1)
    _rrp_constraints(type, port, sigma)
    return nothing
end
function _optimise!(type::RRP2, port::Portfolio2, ::Any, ::Any, ::Any,
                    class::Union{Classic2, FM2}, w_ini::AbstractVector, str_names::Bool)
    mu, sigma, returns = mu_sigma_returns_class(port, class)

    port.model = JuMP.Model()
    model = port.model
    set_string_names_on_creation(model, str_names)
    initial_w(port, w_ini)
    rrp_constraints(type, port, sigma)
    set_returns(nothing, NoKelly(), port.model, port.mu_l; mu = mu)
    linear_constraints(port, MinRisk())
    @objective(model, Min, model[:risk])
    return convex_optimisation(port, nothing, type, class)
end
function optimise2!(port::Portfolio2; rm::Union{AbstractVector, <:TradRiskMeasure} = SD2(),
                    type::PortType = Trad2(), obj::ObjectiveFunction = MinRisk(),
                    kelly::RetType = NoKelly(), class::PortClass = Classic2(),
                    w_ini::AbstractVector = Vector{Float64}(undef, 0),
                    str_names::Bool = false)
    return _optimise!(type, port, rm, obj, kelly, class, w_ini, str_names)
end
export optimise2!

function get_rm_string(rm::Union{AbstractVector, <:TradRiskMeasure})
    rmstr = ""
    if !isa(rm, AbstractVector)
        rstr = string(rm)
        rstr = rstr[1:(findfirst('{', rstr) - 1)]
        rmstr *= rstr
    else
        for (i, r) ∈ enumerate(rm)
            if !isa(r, AbstractVector)
                rstr = string(r)
                rstr = rstr[1:(findfirst('{', rstr) - 1)]
                rmstr *= rstr
            else
                for (j, ri) ∈ enumerate(r)
                    ristr = string(r)
                    ristr = ristr[1:(findfirst('{', ristr) - 1)]
                    rmstr *= ristr
                    if j != length(r)
                        rmstr *= '_'
                    end
                end
            end
            if i != length(rm)
                rmstr *= '_'
            end
        end
    end
    return Symbol(rmstr)
end

function frontier_limits!(port::Portfolio2;
                          rm::Union{AbstractVector, <:TradRiskMeasure} = SD2(),
                          kelly::RetType = NoKelly(), class::PortClass = Classic2(),
                          w_min_ini::AbstractVector = Vector{Float64}(undef, 0),
                          w_max_ini::AbstractVector = Vector{Float64}(undef, 0),
                          save_model::Bool = false)
    optimal1 = deepcopy(port.optimal)
    fail1 = deepcopy(port.fail)
    if save_model
        model1 = copy(port.model)
    end

    w_min = optimise2!(port; rm = rm, obj = MinRisk(), kelly = kelly, class = class,
                       w_ini = w_min_ini)
    w_max = optimise2!(port; rm = rm, obj = MaxRet(), kelly = kelly, class = class,
                       w_ini = w_max_ini)

    limits = hcat(w_min, DataFrame(; x1 = w_max[!, 2]))
    DataFrames.rename!(limits, :weights => :w_min, :x1 => :w_max)

    rmstr = get_rm_string(rm)

    port.limits[rmstr] = limits

    port.optimal = optimal1
    port.fail = fail1
    if save_model
        port.model = model1
    end

    return port.limits[rmstr]
end

function efficient_frontier!(port::Portfolio2;
                             rm::Union{AbstractVector, <:TradRiskMeasure} = SD2(),
                             kelly::RetType = NoKelly(), class::PortClass = Classic2(),
                             w_min_ini::AbstractVector = Vector{Float64}(undef, 0),
                             w_max_ini::AbstractVector = Vector{Float64}(undef, 0),
                             save_model::Bool = false, points::Integer = 20, rf::Real = 0.0)
    optimal1 = deepcopy(port.optimal)
    fail1 = deepcopy(port.fail)
    mu, sigma, returns = mu_sigma_returns_class(port, class)

    fl = frontier_limits!(port; rm = rm, kelly = kelly, class = class,
                          w_min_ini = w_min_ini, w_max_ini = w_max_ini,
                          save_model = save_model)
    w1 = fl.w_min
    w2 = fl.w_max

    if isa(kelly, NoKelly)
        ret1 = dot(mu, w1)
        ret2 = dot(mu, w2)
    else
        ret1 = sum(log.(one(eltype(mu)) .+ returns * w1)) / size(returns, 1)
        ret2 = sum(log.(one(eltype(mu)) .+ returns * w2)) / size(returns, 1)
    end

    rmi = if !isa(rm, AbstractVector)
        rm
    else
        if !isa(rm[1], AbstractVector)
            rm[1]
        else
            rm[1][1]
        end
    end
    rmi.settings.ub = Inf

    set_rm_properties(rmi, port.solvers, sigma)
    risk1, risk2 = calc_risk_bounds(rmi, w1, w2; X = returns, V = port.V, SV = port.SV,
                                    delta = 0)

    mus = range(ret1; stop = ret2, length = points)
    risks = range(risk1; stop = risk2, length = points)

    frontier = Vector{typeof(risk1)}(undef, 0)
    srisk = Vector{typeof(risk1)}(undef, 0)
    w_ini = Vector{typeof(risk1)}(undef, 0)

    i = 0
    for (j, (r, m)) ∈ enumerate(zip(risks, mus))
        if i == 0
            w = optimise2!(port; rm = rm, obj = MinRisk(), kelly = kelly, class = class,
                           w_ini = w_min_ini)
        else
            if !isempty(w)
                w_ini = w.weights
            end
            if j != length(risks)
                rmi.settings.ub = r
            else
                rmi.settings.ub = Inf
            end
            w = optimise2!(port; rm = rm, obj = MaxRet(), kelly = kelly, class = class,
                           w_ini = w_ini)
            # Fallback in case :Max_Ret with maximum risk bounds fails.
            if isempty(w)
                rmi.settings.ub = Inf
                port.mu_l = m
                w = optimise2!(port; rm = rm, obj = MinRisk(), kelly = kelly, class = class,
                               w_ini = w_ini)
                port.mu_l = Inf
            end
        end
        if isempty(w)
            continue
        end

        rk = calc_risk(port; X = returns, type = :Trad2, rm = rmi)

        append!(frontier, w.weights)
        push!(srisk, rk)
        i += 1
    end
    rmi.settings.ub = Inf

    w = optimise2!(port; rm = rm, obj = SR(; rf = rf), kelly = kelly, class = class,
                   w_ini = w_min_ini)
    sharpe = false
    if !isempty(w)
        rk = calc_risk(port; X = returns, type = :Trad2, rm = rmi)
        append!(frontier, w.weights)
        push!(srisk, rk)
        i += 1
        sharpe = true
    end

    rmstr = get_rm_string(rm)

    port.frontier[rmstr] = Dict(:weights => hcat(DataFrame(; tickers = port.assets),
                                                 DataFrame(reshape(frontier, length(w1), :),
                                                           string.(range(1, i)))),
                                :points => points, :risk => srisk, :sharpe => sharpe)

    port.optimal = optimal1
    port.fail = fail1

    return port.frontier[rmstr]
end

export get_rm_string
