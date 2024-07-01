function initial_w(port, w_ini)
    @variable(port.model, w[1:size(port.returns, 2)])
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
            @expression(model, tmp, model[:risk] + scale * rm_risk)
            unregister(model, :risk)
            @expression(model, risk, tmp)
            unregister(model, :tmp)
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

        @constraint(model, sum(tw_ulong) <= long_u * model[:k])
        @constraint(model, sum(tw_ushort) <= short_u * model[:k])

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

        @constraint(model, sum(tw_ulong) <= long_u)
        @constraint(model, sum(tw_ushort) <= short_u)

        @constraint(model, model[:w] .<= tw_ulong)
        @constraint(model, model[:w] .>= -tw_ushort)
    end
    return nothing
end
function network_constraints(args...)
    return nothing
end
function network_constraints(port, network::IP2, ::SR, ::Any)
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
function network_constraints(port, network::IP2, ::Any, ::Any)
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
function network_constraints(port, network::SDP2, obj, ::Trad2)
    _sdp(port, obj)
    @constraint(port.model, network.A .* port.model[:W] .== 0)
    if !haskey(port.model, :sd_risk) && hasproperty(port.network_method, :penalty)
        @expression(port.model, network_penalty,
                    port.network_method.penalty * tr(port.model[:W]))
    end
    return nothing
end
function network_constraints(port, network::SDP2, obj, ::WC2)
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
function _sd_risk(::SDP2, model, sigma, count::Integer, idx::Integer)
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
function _sd_risk(::Any, model, sigma, count::Integer, idx::Integer)
    G = sqrt(sigma)
    if isone(count)
        @variable(model, dev)
        @expression(model, sd_risk, dev^2)
        @constraint(model, [dev; G * model[:w]] ∈ SecondOrderCone())
    else
        if isone(idx)
            @variable(model, dev[1:count])
            @variable(model, sd_risk[1:count])
        end
        @constraint(model, model[:sd_risk][idx] == model[:dev][idx]^2)
        @constraint(model, [model[:dev][idx]; G * model[:w]] ∈ SecondOrderCone())
    end
    return nothing
end
function set_rm(port::Portfolio2, rm::SD2, type::Union{Trad2, RP2}, obj::ObjectiveFunction,
                count::Integer, idx::Integer; sigma::AbstractMatrix{<:Real},
                kelly_approx_idx::Union{AbstractVector{<:Integer}, Nothing}, kwargs...)
    if !isnothing(kelly_approx_idx) && (isnothing(rm.sigma) || isempty(rm.sigma))
        push!(kelly_approx_idx, isone(count) ? 0 : idx)
    else
        sigma = rm.sigma
    end
    model = port.model

    _sd_risk(port.network_method, model, sigma, count, idx)
    _set_sd_risk_upper_bound(port.network_method, obj, type, model, rm.settings.ub, count,
                             idx)
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
            @variable(model, mad_risk[1:count])
        end
        @constraint(model, model[:mad_risk][idx] == sum(model[:mad][idx]) / T)
        @constraint(model, abs_dev * model[:w] .>= -model[:mad][idx])
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
            @variable(model, sdev_risk[1:count])
        end
        @constraint(model, model[:sdev_risk][idx] == model[:sdev][idx] / sqrt(T - 1))
        @constraint(model, abs_dev * model[:w] .>= -model[:ssd][idx])
        @constraint(model, [model[:sdev][idx]; model[:ssd][idx]] ∈ SecondOrderCone())
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
            @variable(model, flpm_risk[1:count])
        end
        @constraint(model, model[:flpm_risk][idx] == sum(model[:flpm][idx]) / T)
        _lpm_risk(type, obj, model, model[:flpm][idx], rm.target)
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
            @variable(model, slpm_risk[1:count])
        end
        @constraint(model, model[:slpm_risk][idx] == model[:tslpm][idx] / sqrt(T - 1))
        @constraint(model, [model[:tslpm][idx]; model[:slpm][idx]] ∈ SecondOrderCone())
        _lpm_risk(type, obj, model, model[:slpm][idx], rm.target)
        _set_rm_risk_upper_bound(obj, type, model, model[:slpm_risk][idx], rm.settings.ub)
        _set_risk_expression(model, model[:slpm_risk][idx], rm.settings.scale,
                             rm.settings.flag)
    end
    return nothing
end
function set_rm(port::Portfolio2, rm::WR2, type::Union{Trad2, RP2}, obj::ObjectiveFunction,
                ::Any, ::Any; returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
    if !haskey(model, :X)
        @expression(model, X, returns * model[:w])
    end

    @variable(model, wr)
    @expression(model, wr_risk, wr)
    @constraint(model, -model[:X] .<= wr)
    _set_rm_risk_upper_bound(obj, type, model, -model[:X], rm.settings.ub)
    _set_risk_expression(model, wr_risk, rm.settings.scale, rm.settings.flag)
    return nothing
end
function set_rm(port::Portfolio2, rm::RG2, type::Union{Trad2, RP2}, obj::ObjectiveFunction,
                ::Any, ::Any; returns::AbstractMatrix{<:Real}, kwargs...)
    model = port.model
    if !haskey(model, :X)
        @expression(model, X, returns * model[:w])
    end
    if !haskey(model, :wr)
        @variable(model, wr)
        @expression(model, wr_risk, wr)
        @constraint(model, -model[:X] .<= wr)
    end

    @variable(model, br)
    @expression(model, rg_risk, wr_risk - br)
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
            @variable(model, cvar_risk[1:count])
        end
        @constraint(model,
                    model[:cvar_risk][idx] ==
                    model[:var][idx] + sum(model[:z_var][1:T, idx]) * iat)
        @constraint(model, model[:z_var][1:T, idx] .>= -model[:X] .- model[:var][idx])
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
            @variable(model, cvar_risk_l[1:count])
            @variable(model, var_h[1:count])
            @variable(model, z_var_h[1:T, 1:count] .<= 0)
            @variable(model, cvar_risk_h[1:count])
            @variable(model, rcvar_risk[1:count])
        end
        @constraint(model,
                    model[:cvar_risk_l][idx] ==
                    model[:var_l][idx] + sum(model[:z_var_l][1:T, idx]) * iat)
        @constraint(model, model[:z_var_l][1:T, idx] .>= -model[:X] .- model[:var_l][idx])
        @constraint(model,
                    model[:cvar_risk_h][idx] ==
                    model[:var_h][idx] + sum(model[:z_var_h][1:T, idx]) * ibt)
        @constraint(model, model[:z_var_h][1:T, idx] .<= -model[:X] .- model[:var_h][idx])
        @constraint(model,
                    model[:rcvar_risk][idx] ==
                    model[:cvar_risk_l][idx] - model[:cvar_risk_h][idx])
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
            @variable(model, evar_risk[1:count])
        end
        @constraint(model,
                    model[:evar_risk][idx] ==
                    model[:t_evar][idx] - model[:z_evar][idx] * log(at))
        @constraint(model, sum(model[:u_evar][:, idx]) <= model[:z_evar][idx])
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
            @variable(model, rvar_risk[1:count])
        end
        @constraint(model,
                    model[:rvar_risk][idx] ==
                    model[:t_rvar][idx] +
                    lnk * model[:z_rvar][idx] +
                    sum(model[:psi_rvar][:, idx] .+ model[:theta_rvar][:, idx]))
        @constraint(model, [i = 1:T],
                    [model[:z_rvar][idx] * opk * ik2, model[:psi_rvar][i, idx] * opk * ik,
                     model[:epsilon_rvar][i, idx]] ∈ MOI.PowerCone(iopk))
        @constraint(model, [i = 1:T],
                    [model[:omega_rvar][i, idx] * iomk, model[:theta_rvar][i, idx] * ik,
                     -model[:z_rvar][idx] * ik2] ∈ MOI.PowerCone(omk))
        @constraint(model,
                    -model[:X] .- model[:t_rvar][idx] .+ model[:epsilon_rvar][:, idx] .+
                    model[:omega_rvar][:, idx] .<= 0)
        _set_rm_risk_upper_bound(obj, type, model, model[:rvar_risk][idx], rm.settings.ub)
        _set_risk_expression(model, model[:rvar_risk][idx], rm.settings.scale,
                             rm.settings.flag)
    end
    return nothing
end
function set_rm(port::Portfolio2, rm::MDD2, type::Union{Trad2, RP2}, obj::ObjectiveFunction,
                ::Any, ::Any; returns::AbstractMatrix{<:Real}, kargs...)
    model = port.model
    T = size(returns, 1)

    if !haskey(model, :X)
        @expression(model, X, returns * model[:w])
    end
    if !haskey(model, :dd)
        @variable(model, dd[1:(T + 1)])
        @constraint(model, dd[2:end] .>= dd[1:(end - 1)] .- model[:X])
        @constraint(model, dd[2:end] .>= 0)
        @constraint(model, dd[1] == 0)
    end

    @variable(model, mdd)
    @expression(model, mdd_risk, mdd)
    @constraint(model, mdd .>= model[:dd][2:end])
    _set_rm_risk_upper_bound(obj, type, model, model[:dd][2:end], rm.settings.ub)
    _set_risk_expression(model, mdd_risk, rm.settings.scale, rm.settings.flag)
    return nothing
end
function set_rm(port::Portfolio2, rm::ADD2, type::Union{Trad2, RP2}, obj::ObjectiveFunction,
                ::Any, ::Any; returns::AbstractMatrix{<:Real}, kargs...)
    model = port.model
    T = size(returns, 1)

    if !haskey(model, :X)
        @expression(model, X, returns * model[:w])
    end
    if !haskey(model, :dd)
        @variable(model, dd[1:(T + 1)])
        @constraint(model, dd[2:end] .>= dd[1:(end - 1)] .- model[:X])
        @constraint(model, dd[2:end] .>= 0)
        @constraint(model, dd[1] == 0)
    end

    @expression(model, add_risk, sum(model[:dd][2:end]) / T)
    _set_rm_risk_upper_bound(obj, type, model, add_risk, rm.settings.ub)
    _set_risk_expression(model, add_risk, rm.settings.scale, rm.settings.flag)
    return nothing
end
function set_rm(port::Portfolio2, rm::UCI2, type::Union{Trad2, RP2}, obj::ObjectiveFunction,
                ::Any, ::Any; returns::AbstractMatrix{<:Real}, kargs...)
    model = port.model
    T = size(returns, 1)

    if !haskey(model, :X)
        @expression(model, X, returns * model[:w])
    end
    if !haskey(model, :dd)
        @variable(model, dd[1:(T + 1)])
        @constraint(model, dd[2:end] .>= dd[1:(end - 1)] .- model[:X])
        @constraint(model, dd[2:end] .>= 0)
        @constraint(model, dd[1] == 0)
    end

    @variable(model, uci)
    @expression(model, uci_risk, uci / sqrt(T))
    @constraint(model, [uci; dd[2:end]] ∈ SecondOrderCone())
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

    if !haskey(model, :X)
        @expression(model, X, returns * model[:w])
    end
    if !haskey(model, :dd)
        @variable(model, dd[1:(T + 1)])
        @constraint(model, dd[2:end] .>= dd[1:(end - 1)] .- model[:X])
        @constraint(model, dd[2:end] .>= 0)
        @constraint(model, dd[1] == 0)
    end

    if isone(count)
        @variable(model, dar)
        @variable(model, z_cdar[1:T] .>= 0)
        @expression(model, cdar_risk, dar + sum(z_cdar) * iat)
        @constraint(model, z_cdar .>= dd[2:end] .- dar)
        _set_rm_risk_upper_bound(obj, type, model, cdar_risk, rm.settings.ub)
        _set_risk_expression(model, cdar_risk, rm.settings.scale, rm.settings.flag)
    else
        if isone(idx)
            @variable(model, dar[1:count])
            @variable(model, z_cdar[1:T, 1:count] .>= 0)
            @variable(model, cdar_risk[1:count])
        end
        @constraint(model,
                    model[:cdar_risk][idx] ==
                    model[:dar][idx] + sum(model[:z_cdar][:, idx]) * iat)
        @constraint(model, model[:z_cdar][:, idx] .>= model[:dd][2:end] .- model[:dar][idx])
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

    if !haskey(model, :X)
        @expression(model, X, returns * model[:w])
    end
    if !haskey(model, :dd)
        @variable(model, dd[1:(T + 1)])
        @constraint(model, dd[2:end] .>= dd[1:(end - 1)] .- model[:X])
        @constraint(model, dd[2:end] .>= 0)
        @constraint(model, dd[1] == 0)
    end

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
            @variable(model, edar_risk[1:count])
        end
        @constraint(model,
                    model[:edar_risk][idx] ==
                    model[:t_edar][idx] - model[:z_edar][idx] * log(at))
        @constraint(model, sum(model[:u_edar][:, idx]) <= model[:z_edar][idx])
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

    if !haskey(model, :X)
        @expression(model, X, returns * model[:w])
    end
    if !haskey(model, :dd)
        @variable(model, dd[1:(T + 1)])
        @constraint(model, dd[2:end] .>= dd[1:(end - 1)] .- model[:X])
        @constraint(model, dd[2:end] .>= 0)
        @constraint(model, dd[1] == 0)
    end

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
        @constraint(model, model[:dd][2:end] .- t_rdar .+ epsilon_rdar .+ omega_rdar .<= 0)
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
            @variable(model, rdar_risk[1:count])
        end
        @constraint(model,
                    model[:rdar_risk][idx] ==
                    model[:t_rdar][idx] +
                    lnk * model[:z_rdar][idx] +
                    sum(model[:psi_rdar][:, idx] .+ model[:theta_rdar][:, idx]))
        @constraint(model, [i = 1:T],
                    [model[:z_rdar][idx] * opk * ik2, model[:psi_rdar][i, idx] * opk * ik,
                     model[:epsilon_rdar][i, idx]] ∈ MOI.PowerCone(iopk))
        @constraint(model, [i = 1:T],
                    [model[:omega_rdar][i, idx] * iomk, model[:theta_rdar][i, idx] * ik,
                     -model[:z_rdar][idx] * ik2] ∈ MOI.PowerCone(omk))
        @constraint(model,
                    model[:dd][2:end] .- model[:t_rdar][idx] .+
                    model[:epsilon_rdar][:, idx] .+ model[:omega_rdar][:, idx] .<= 0)
        _set_rm_risk_upper_bound(obj, type, model, model[:rdar_risk][idx], rm.settings.ub)
        _set_risk_expression(model, model[:rdar_risk][idx], rm.settings.scale,
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
    @expression(model, Dx,
                Symmetric(model[:X] * transpose(ovec) - ovec * transpose(model[:X])))
    @constraint(model, [i = 1:T, j = i:T], Dt[i, j] >= -Dx[i, j])
    @constraint(model, [i = 1:T, j = i:T], Dt[i, j] >= Dx[i, j])
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
    f = port.max_num_assets_kurt_scale

    if isone(count)
        @variable(model, kurt_risk)
        if !iszero(nmax) && N > nmax
            Nf = f * nmax
            @variable(model, x_kurt[1:Nf])
            @constraint(model, [kurt_risk; x_kurt] ∈ SecondOrderCone())
            A = block_vec_pq(kt, N, N)
            vals_A, vecs_A = eigen(A)
            vals_A = clamp.(real(vals_A), 0, Inf) .+ clamp.(imag(vals_A), 0, Inf)im
            Bi = Vector{Matrix{eltype(kt)}}(undef, Nf)
            for i ∈ 1:Nf
                j = i - 1
                B = reshape(real(complex(sqrt(vals_A[end - j])) * vecs_A[:, end - j]), N, N)
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
                @variable(model, zkurt[1:count])
            end
        end
        if !iszero(nmax) && N > nmax
            @constraint(model,
                        [model[:kurt_risk][idx]; model[:x_kurt][:, idx]] ∈
                        SecondOrderCone())
            A = block_vec_pq(kt, N, N)
            vals_A, vecs_A = eigen(A)
            vals_A = clamp.(real(vals_A), 0, Inf) .+ clamp.(imag(vals_A), 0, Inf)im
            Bi = Vector{Matrix{eltype(kt)}}(undef, Nf)
            for i ∈ 1:Nf
                j = i - 1
                B = reshape(real(complex(sqrt(vals_A[end - j])) * vecs_A[:, end - j]), N, N)
                Bi[i] = B
            end
            @constraint(model, [i = 1:Nf], model[:x_kurt][i, idx] == tr(Bi[i] * model[:W]))
        else
            L_2 = port.L_2
            S_2 = port.S_2
            sqrt_sigma_4 = sqrt(S_2 * kt * transpose(S_2))
            @constraint(model, model[:zkurt][idx] == L_2 * vec(model[:W]))
            @constraint(model,
                        [model[:kurt_risk][idx]; sqrt_sigma_4 * model[:zkurt][idx]] ∈
                        SecondOrderCone())
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
    f = port.max_num_assets_kurt_scale

    if isone(count)
        @variable(model, skurt_risk)
        if !iszero(nmax) && N > nmax
            Nf = f * nmax
            @variable(model, x_skurt[1:Nf])
            @constraint(model, [skurt_risk; x_skurt] ∈ SecondOrderCone())
            A = block_vec_pq(kt, N, N)
            vals_A, vecs_A = eigen(A)
            vals_A = clamp.(real(vals_A), 0, Inf) .+ clamp.(imag(vals_A), 0, Inf)im
            Bi = Vector{Matrix{eltype(kt)}}(undef, Nf)
            for i ∈ 1:Nf
                j = i - 1
                B = reshape(real(complex(sqrt(vals_A[end - j])) * vecs_A[:, end - j]), N, N)
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
                @variable(model, zskurt[1:count])
            end
        end
        if !iszero(nmax) && N > nmax
            @constraint(model,
                        [model[:skurt_risk][idx]; model[:x_skurt][:, idx]] ∈
                        SecondOrderCone())
            A = block_vec_pq(kt, N, N)
            vals_A, vecs_A = eigen(A)
            vals_A = clamp.(real(vals_A), 0, Inf) .+ clamp.(imag(vals_A), 0, Inf)im
            Bi = Vector{Matrix{eltype(kt)}}(undef, Nf)
            for i ∈ 1:Nf
                j = i - 1
                B = reshape(real(complex(sqrt(vals_A[end - j])) * vecs_A[:, end - j]), N, N)
                Bi[i] = B
            end
            @constraint(model, [i = 1:Nf], model[:x_skurt][i, idx] == tr(Bi[i] * model[:W]))
        else
            L_2 = port.L_2
            S_2 = port.S_2
            sqrt_sigma_4 = sqrt(S_2 * kt * transpose(S_2))
            @constraint(model, model[:zskurt][idx] == L_2 * vec(model[:W]))
            @constraint(model,
                        [model[:skurt_risk][idx]; sqrt_sigma_4 * model[:zskurt][idx]] ∈
                        SecondOrderCone())
        end
        _set_rm_risk_upper_bound(obj, type, model, model[:skurt_risk][idx], rm.settings.ub)
        _set_risk_expression(model, model[:skurt_risk][idx], rm.settings.scale,
                             rm.settings.flag)
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
        if !haskey(model, :owa)
            @variable(model, owa[1:T])
            @constraint(model, model[:X] == owa)
        end
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
            if !haskey(model, :owa)
                @variable(model, owa[1:T])
                @constraint(model, model[:X] == owa)
            end
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
            @variable(model, tg_risk[1:count])
            if !rm.owa.approx
                if !haskey(model, :owa)
                    @variable(model, owa[1:T])
                    @constraint(model, model[:X] == owa)
                end
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
            @constraint(model,
                        model[:tg_risk][idx] ==
                        sum(model[:tga][:, idx] .+ model[:tgb][:, idx]))
            tg_w = owa_tg(T; alpha_i = alpha_i, alpha = alpha, a_sim = a_sim)
            @constraint(model,
                        model[:owa] * transpose(tg_w) .<=
                        ovec * transpose(model[:tga][:, idx]) +
                        model[:tgb][:, idx] * transpose(ovec))
        else
            owa_p = rm.owa.p
            M = length(owa_p)
            tg_w = -owa_tg(T; alpha_i = alpha_i, alpha = alpha, a_sim = a_sim)
            tg_s = sum(tg_w)
            tg_l = minimum(tg_w)
            tg_h = maximum(tg_w)
            tg_d = [norm(tg_w, p) for p ∈ owa_p]

            @constraint(model,
                        model[:tg_risk][idx] ==
                        tg_s * model[:tg_t][idx] - tg_l * sum(model[:tg_nu][:, idx]) +
                        tg_h * sum(model[:tg_eta][:, idx]) +
                        dot(tg_d, model[:tg_y][:, idx]))
            @constraint(model,
                        model[:X] .+ model[:tg_t][idx] .- model[:tg_nu][:, idx] .+
                        model[:tg_eta][:, idx] .-
                        vec(sum(model[:tg_epsilon][:, :, idx]; dims = 2)) .== 0)
            @constraint(model,
                        model[:tg_z][:, idx] .+ model[:tg_y][:, idx] .==
                        vec(sum(model[:tg_psi][:, :, idx]; dims = 1)))

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
            if !haskey(model, :owa)
                @variable(model, owa[1:T])
                @constraint(model, model[:X] == owa)
            end
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
            @variable(model, rtg_risk[1:count])
            if !rm.owa.approx
                if !haskey(model, :owa)
                    @variable(model, owa[1:T])
                    @constraint(model, model[:X] == owa)
                end
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
                @variable(model, rltg_risk[1:count])
                @variable(model, rhtg_t[1:count])
                @variable(model, rhtg_nu[1:T, 1:count] .>= 0)
                @variable(model, rhtg_eta[1:T, 1:count] .>= 0)
                @variable(model, rhtg_epsilon[1:T, 1:M, 1:count])
                @variable(model, rhtg_psi[1:T, 1:M, 1:count])
                @variable(model, rhtg_z[1:M, 1:count])
                @variable(model, rhtg_y[1:M, 1:count] .>= 0)
                @variable(model, rhtg_risk[1:count])
            end
        end
        if !rm.owa.approx
            ovec = range(1; stop = 1, length = T)
            @constraint(model,
                        model[:rtg_risk][idx] ==
                        sum(model[:rtga][:, idx] .+ model[:rtgb][:, idx]))
            rtg_w = owa_rtg(T; alpha_i = alpha_i, alpha = alpha, a_sim = a_sim,
                            beta_i = beta_i, beta = beta, b_sim = b_sim)
            @constraint(model,
                        model[:owa] * transpose(rtg_w) .<=
                        ovec * transpose(model[:rtga][:, idx]) +
                        model[:rtgb][:, idx] * transpose(ovec))
        else
            owa_p = rm.owa.p
            M = length(owa_p)
            rltg_w = -owa_tg(T; alpha_i = alpha_i, alpha = alpha, a_sim = a_sim)
            rltg_s = sum(rltg_w)
            rltg_l = minimum(rltg_w)
            rltg_h = maximum(rltg_w)
            rltg_d = [norm(rltg_w, p) for p ∈ owa_p]

            @constraint(model,
                        model[:rltg_risk][idx] ==
                        rltg_s * model[:rltg_t][idx] -
                        rltg_l * sum(model[:rltg_nu][:, idx]) +
                        rltg_h * sum(model[:rltg_eta][:, idx]) +
                        dot(rltg_d, model[:rltg_y][:, idx]))
            @constraint(model,
                        model[:X] .+ model[:rltg_t][idx] .- model[:rltg_nu][:, idx] .+
                        model[:rltg_eta][:, idx] .-
                        vec(sum(model[:rltg_epsilon][:, :, idx]; dims = 2)) .== 0)
            @constraint(model,
                        model[:rltg_z][:, idx] .+ model[:rltg_y][:, idx] .==
                        vec(sum(model[:rltg_psi][:, :, idx]; dims = 1)))
            @constraint(model, [i = 1:M, j = 1:T],
                        [-model[:rltg_z][i, idx] * owa_p[i],
                         model[:rltg_psi][j, i, idx] * owa_p[i] / (owa_p[i] - 1),
                         model[:rltg_epsilon][j, i, idx]] ∈ MOI.PowerCone(1 / owa_p[i]))

            rhtg_w = -owa_tg(T; alpha_i = beta_i, alpha = beta, a_sim = b_sim)
            rhtg_s = sum(rhtg_w)
            rhtg_l = minimum(rhtg_w)
            rhtg_h = maximum(rhtg_w)
            rhtg_d = [norm(rhtg_w, p) for p ∈ owa_p]

            @constraint(model,
                        model[:rhtg_risk][idx] ==
                        rhtg_s * model[:rhtg_t][idx] -
                        rhtg_l * sum(model[:rhtg_nu][:, idx]) +
                        rhtg_h * sum(model[:rhtg_eta][:, idx]) +
                        dot(rhtg_d, model[:rhtg_y][:, idx]))
            @constraint(model,
                        model[:rtg_risk][idx] ==
                        model[:rltg_risk][idx] + model[:rhtg_risk][idx])
            @constraint(model,
                        -model[:X] .+ model[:rhtg_t][idx] .- model[:rhtg_nu][:, idx] .+
                        model[:rhtg_eta][:, idx] .-
                        vec(sum(model[:rhtg_epsilon][:, :, idx]; dims = 2)) .== 0)
            @constraint(model,
                        model[:rhtg_z][:, idx] .+ model[:rhtg_y][:, idx] .==
                        vec(sum(model[:rhtg_psi][:, :, idx]; dims = 1)))
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
            if !haskey(model, :owa)
                @variable(model, owa[1:T])
                @constraint(model, model[:X] == owa)
            end

            @variable(model, owa_a[1:T])
            @variable(model, owa_b[1:T])
            @expression(model, owa_risk, sum(owa_a .+ owa_b))
            owa_w = (isnothing(rm.w) || isempty(rm.w)) ? owa_gmd(T) : rm.w
            @constraint(model,
                        model[:owa] * transpose(owa_w) .<=
                        ovec * transpose(owa_a) + owa_b * transpose(ovec))
        else
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
            @expression(model, owa_risk[1:count])
            if !rm.owa.approx
                if !haskey(model, :owa)
                    @variable(model, owa[1:T])
                    @constraint(model, model[:X] == owa)
                end
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
            @constraint(model,
                        model[:owa_risk][idx] ==
                        sum(model[:owa_a][:, idx] .+ model[:owa_b][:, idx]))
            owa_w = (isnothing(rm.w) || isempty(rm.w)) ? owa_gmd(T) : rm.w
            @constraint(model,
                        model[:owa] * transpose(owa_w) .<=
                        ovec * transpose(model[:owa_a][:, idx]) +
                        model[:owa_b][:, idx] * transpose(ovec))
        else
            owa_p = rm.owa.p
            M = length(owa_p)

            owa_w = (isnothing(rm.w) || isempty(rm.w)) ? -owa_gmd(T) : -rm.w
            owa_s = sum(owa_w)
            owa_l = minimum(owa_w)
            owa_h = maximum(owa_w)
            owa_d = [norm(owa_w, p) for p ∈ owa_p]

            @constraint(model,
                        model[:owa_risk][idx] ==
                        owa_s * model[:owa_t][idx] - owa_l * sum(model[:owa_nu][:, idx]) +
                        owa_h * sum(model[:owa_eta][:, idx]) +
                        dot(owa_d, model[:owa_y][:, idx]))
            @constraint(model,
                        model[:X] .+ model[:owa_t][idx] .- model[:owa_nu][:, idx] .+
                        model[:owa_eta][:, idx] .-
                        vec(sum(model[:owa_epsilon][:, :, idx]; dims = 2)) .== 0)
            @constraint(model,
                        model[:owa_z][:, idx] .+ model[:owa_y][:, idx] .==
                        vec(sum(model[:owa_psi][:, :, idx]; dims = 1)))
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
        @constraint(model, ret >= mu_l)
    end
    return nothing
end
function _return_bounds(::SR, model, mu_l::Real)
    if isfinite(mu_l)
        @constraint(model, ret >= mu_l * model[:k])
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
function set_returns(obj::Any, ::AKelly, model, mu_l::Real; mu::AbstractVector,
                     kelly_approx_idx::Union{AbstractVector{<:Integer}, Nothing},
                     network_method::NetworkMethods2, sigma::AbstractMatrix, kwargs...)
    if !isempty(mu)
        if isnothing(kelly_approx_idx) || isempty(kelly_approx_idx)
            _sd_risk(network_method, model, sigma, 1, 1)
            @expression(model, ret, dot(mu, model[:w]) - 0.5 * model[:sd_risk])
        else
            if iszero(kelly_approx_idx[1])
                @expression(model, ret, dot(mu, model[:w]) - 0.5 * model[:sd_risk])
            else
                @expression(model, ret,
                            dot(mu, model[:w]) - 0.5 * model[:sd_risk][kelly_approx_idx[1]])
            end
        end
        _return_bounds(obj, model, mu_l)
    end
    return nothing
end
function set_returns(obj::SR, ::AKelly, model, mu_l::Real; mu::AbstractVector,
                     kelly_approx_idx::Union{AbstractVector{<:Integer}, Nothing},
                     network_method::NetworkMethods2, sigma::AbstractMatrix, kwargs...)
    if !isempty(mu)
        @variable(model, tapprox_kelly)
        @constraint(model, model[:risk] <= 1)
        @expression(model, ret, dot(mu, model[:w]) - 0.5 * tapprox_kelly)
        if isempty(kelly_approx_idx)
            _sd_risk(network_method, model, sigma, 1, 1)
            @constraint(model,
                        [model[:k] + tapprox_kelly
                         2 * model[:dev]
                         model[:k] - tapprox_kelly] ∈ SecondOrderCone())
        else
            if isnothing(kelly_approx_idx) || iszero(kelly_approx_idx[1])
                @constraint(model,
                            [model[:k] + tapprox_kelly
                             2 * model[:dev]
                             model[:k] - tapprox_kelly] ∈ SecondOrderCone())
            else
                @constraint(model,
                            [model[:k] + tapprox_kelly
                             2 * model[:dev][kelly_approx_idx[1]]
                             model[:k] - tapprox_kelly] ∈ SecondOrderCone())
            end
        end
        _return_bounds(obj, model, mu_l)
    end
    return nothing
end
function set_returns(obj::Any, ::EKelly, model, mu_l::Real; returns::AbstractMatrix,
                     kwargs...)
    if !isempty(mu)
        @variable(model, texact_kelly[1:T])
        @expression(model, ret, sum(texact_kelly) / T)
        @expression(model, kret, 1 .+ returns * model[:w])
        @constraint(model, [i = 1:T], [texact_kelly[i], 1, kret[i]] ∈ MOI.ExponentialCone())
        _return_bounds(obj, model, mu_l)
    end
    return nothing
end
function set_returns(obj::SR, ::EKelly, model, mu_l::Real; returns::AbstractMatrix,
                     kwargs...)
    if !isempty(mu)
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
function return_constraints(port, obj, ::Any, ::Any, mu, sigma, returns, kelly_approx_idx)
    set_returns(obj, NoKelly(), port.model, port.mu_l; mu = mu, sigma = sigma,
                returns = returns, kelly_approx_idx = kelly_approx_idx,
                network_method = port.network_method)
    return nothing
end
function setup_tracking_err_constraints(::NoTracking, args...)
    return nothing
end
function _tracking_err_constraints(::Any, model, returns, tracking_err, benchmark)
    @variable(model, t_track_err >= 0)
    @expression(model, track_err, returns * model[:w] .- benchmark)
    @constraint(model, [t_track_err; track_err] ∈ SecondOrderCone())
    @constraint(model, t_track_err <= tracking_err * sqrt(T - 1))
    return nothing
end
function _tracking_err_constraints(::SR, model, returns, tracking_err, benchmark)
    @variable(model, t_track_err >= 0)
    @expression(model, track_err, returns * model[:w] .- benchmark * model[:k])
    @constraint(model, [t_track_err; track_err] ∈ SecondOrderCone())
    @constraint(model, t_track_err <= tracking_err * model[:k] * sqrt(T - 1))
    return nothing
end
function tracking_err_constraints(args...)
    return nothing
end
function tracking_err_constraints(::TrackWeight, port, returns, obj)
    if !(isempty(isempty(port.tracking_err_weights)) || isinf(port.tracking_err))
        _tracking_err_constraints(obj, port.model, returns, port.tracking_err,
                                  returns * port.tracking_err_weights)
    end
    return nothing
end
function tracking_err_constraints(::TrackRet, port, returns, obj)
    if !(isempty(isempty(port.tracking_err_returns)) || isinf(port.tracking_err))
        _tracking_err_constraints(obj, port.model, returns, port.tracking_err,
                                  port.tracking_err_returns)
    end
    return nothing
end
function _turnover_constraints(::Any, model, turnover, turnover_weights)
    N = length(turnover_weights)
    @variable(model, t_turnov[1:N] >= 0)
    @expression(model, turnov, model[:w] .- turnover_weights)
    @constraint(model, [i = 1:N], [t_turnov[i]; turnov[i]] ∈ MOI.NormOneCone(2))
    @constraint(model, t_turnov .<= turnover)
    return nothing
end
function _turnover_constraints(::SR, model, turnover, turnover_weights)
    N = length(turnover_weights)
    @variable(model, t_turnov[1:N] >= 0)
    @expression(model, turnov, model[:w] .- turnover_weights * model[:k])
    @constraint(model, [i = 1:N], [t_turnov[i]; turnov[i]] ∈ MOI.NormOneCone(2))
    @constraint(model, t_turnov .<= turnover * model[:k])
    return nothing
end
function turnover_constraints(port, obj)
    if !(isa(port.turnover, Real) && isinf(port.turnover) ||
         isa(port.turnover, AbstractVector) && isempty(port.turnover) ||
         isempty(port.turnover_weights))
        _turnover_constraints(obj, port.model, port.turnover, port.turnover_weights)
    end
    return nothing
end
function _rebalance_constraints(::Any, model, rebalance, rebalance_weights)
    N = length(rebalance_weights)
    @variable(model, t_rebal[1:N] >= 0)
    @expression(model, rebal, model[:w] .- rebalance_weights)
    @constraint(model, [i = 1:N], [t_rebal[i]; rebal[i]] ∈ MOI.NormOneCone(2))
    @expression(model, sum_t_rebal, sum(rebalance .* t_rebal))
    return nothing
end
function _rebalance_constraints(::SR, model, rebalance, rebalance_weights)
    N = length(rebalance_weights)
    @variable(model, t_rebal[1:N] >= 0)
    @expression(model, rebal, model[:w] .- rebalance_weights * model[:k])
    @constraint(model, [i = 1:N], [t_rebal[i]; rebal[i]] ∈ MOI.NormOneCone(2))
    @expression(model, sum_t_rebal, sum(rebalance .* t_rebal))
    return nothing
end
function rebalance_constraints(port, obj)
    rebalance = port.rebalance
    rebalance_weights = port.rebalance_weights
    if !(isa(port.rebalance, Real) && (isinf(port.rebalance) || iszero(port.rebalance)) ||
         isa(port.rebalance, AbstractVector) && isempty(port.rebalance) ||
         isempty(port.rebalance_weights))
        _rebalance_constraints(obj, port.model, port.rebalance, port.rebalance_weights)
    end
    return nothing
end
function _objective(::SR, ::Classic2, ::Union{AKelly, EKelly}, model, p)
    @objective(model, Max, model[:ret] - p)
    return nothing
end
function _objective(::Union{SR, MinRisk}, ::Any, ::Any, model, p)
    @objective(model, Min, model[:risk] + p)
    return nothing
end
function _objective(obj::Util, ::Any, ::Any, model, p)
    @objective(model, Max, model[:ret] - obj.l * model[:risk] - p)
    return nothing
end
function _objective(obj::MaxRet, ::Any, ::Any, model, p)
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
    _objective(obj, class, kelly, port.model, npf + rbf)
    return nothing
end
function objective_function(port, obj, ::Any, class, kelly)
    rbf = zero(eltype(port.returns))
    if haskey(port.model, :sum_t_rebal)
        npf = port.model[:sum_t_rebal]
    end
    _objective(obj, class, kelly, port.model, rbf)
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
    network_constraints(port, port.network_method, obj, Trad2())
    tracking_err_constraints(port.tracking_err, port, returns, obj)
    turnover_constraints(port, obj)
    rebalance_constraints(port, obj)
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
    @variable(model, Au[1:N, 1:N] .>= 0, Symmetric)
    @variable(model, Al[1:N, 1:N] .>= 0, Symmetric)
    @constraint(model, Au .- Al .== model[:W])
    @expression(model, risk, tr(Au * port.cov_u) - tr(Al * port.cov_l))
    return nothing
end
function _wc_risk_constraints(::WCEllipse, port, obj)
    _sdp(port, obj)
    G_sigma = sqrt(portfolio.cov_sigma)
    model = port.model
    @variable(model, E[1:N, 1:N], Symmetric)
    @constraint(model, E ∈ PSDCone())
    @expression(model, W_p_E, model[:W] .+ E)
    @expression(model, x_ge, G_sigma * vec(W_p_E))
    @variable(model, t_ge)
    @constraint(model, [t_ge; x_ge] ∈ SecondOrderCone())
    @expression(model, risk, tr(sigma * W_p_E) + port.k_sigma * t_ge)
    return nothing
end
function _wc_risk_constraints(::NoWC, port)
    _sd_risk(port.network_method, port.model, port.cov, 1, 1)
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
    _wc_risk_constraints(type.cov, port)
    _wc_sharpe_constraints(obj, port)
    return nothing
end
function _optimise!(type::WC2, port::Portfolio2,
                    rm::Union{AbstractVector, <:TradRiskMeasure}, obj::ObjectiveFunction,
                    ::Any, ::Any, w_ini::AbstractVector, str_names::Bool, save_params::Bool)
    port.model = JuMP.Model()
    model = port.model
    set_string_names_on_creation(model, str_names)
    initial_w(port, w_ini)
    set_sr_k(obj, model)
    wc_constraints(port, obj, type)
    linear_constraints(port, obj)
    centrality_constraints(port, obj)
    weight_constraints(port, obj)
    num_assets_constraints(port, obj)
    network_constraints(port, port.network_method, obj, type)
    tracking_err_constraints(port.tracking_err, port, port.returns, obj)
    turnover_constraints(port, obj)
    rebalance_constraints(port, obj)
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
    model = fit(method.pcr, X_std)
    Vp = projection(model)
    sdev = if isnothing(method.std_w)
        vec(std(method.ve, X; dims = 2))
    else
        vec(std(method.ve, X, method.std_w; dims = 2))
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
    @variable(model, log_w[1:N])
    @constraint(model, dot(rb, log_w) >= 1)
    @constraint(model, [i = 1:N], [log_w[i], 1, model[:w][i]] ∈ MOI.ExponentialCone())
    @constraint(model, model[:w] .>= 0)
    return nothing
end
function _rp_class_constraints(::FC2, port)
    b1, b2, missing, missing = _factors_b1_b2_b3(port.loadings, port.f_returns,
                                                 port.loadings_opt)
    N_f = size(b1, 2)

    rb = port.f_risk_budget
    if isempty(rb) || length(rb) != N_f
        rb = port.f_risk_budget = fill(1 / N_f, N_f)
    elseif !isapprox(sum(port.f_risk_budget), one(eltype(port.returns)))
        port.f_risk_budget ./= sum(port.f_risk_budget)
    end

    @variable(model, w1[1:N_f])
    @variable(model, w2[1:(N - N_f)])
    delete(model, model[:w])
    unregister(model, :w)
    @expression(model, w, b1 * w1 + b2 * w2)
    @variable(model, log_w[1:N_f])
    @constraint(model, dot(rb, log_w) >= 1)
    @constraint(model, [i = 1:N_f], [log_w[i], 1, model[:w1][i]] ∈ MOI.ExponentialCone())
    return nothing
end
function rp_constraints(port, class)
    model = port.model
    @variable(model, k)
    _rp_class_constraints(class, port)
    @constraint(model, sum(model[:w]) == model[:k])
    return nothing
end
function _optimise!(type::RP2, port::Portfolio2,
                    rm::Union{AbstractVector, <:TradRiskMeasure}, ::Any, ::Any,
                    class::Union{Classic2, FM2, FC2}, w_ini::AbstractVector,
                    str_names::Bool, save_params::Bool)
    mu, sigma, returns = mu_sigma_returns_class(port, class)

    port.model = JuMP.Model()
    model = port.model
    set_string_names_on_creation(model, str_names)
    initial_w(port, w_ini)
    rp_constraints(port, class)
    risk_constraints(port, nothing, RP2(), rm, mu, sigma, returns)
    set_returns(nothing, NoKelly(), nothing, port.model, port.mu_l; mu = mu)
    linear_constraints(port, MinRisk())
    objective_function(port, MinRisk(), RP2(), class, nothing)
    return convex_optimisation(port, nothing, RP2(), class)
end
function _rrp_ver_constraints(::NoRRP, model, G)
    @constraint(model, [model[:psi]; G * model[:w]] ∈ SecondOrderCone())
    return nothing
end
function _rrp_ver_constraints(::RegRRP, model, G)
    @variable(model, rho)
    @constraint(model, [2 * model[:psi]; 2 * G * model[:w]; -2 * rho] ∈ SecondOrderCone())
    @constraint(model, [rho; G * model[:w]] ∈ SecondOrderCone())
    return nothing
end
function _rrp_ver_constraints(version::RegPenRRP, model, G)
    @variable(model, rho)
    @constraint(model, [2 * model[:psi]; 2 * G * model[:w]; -2 * rho] ∈ SecondOrderCone())
    theta = Diagonal(sqrt.(diag(sigma)))
    @constraint(model, [rho; sqrt(version.penalty) * theta * model[:w]] ∈ SecondOrderCone())
    return nothing
end
function _rrp_constraints(type::RRP2, port, sigma)
    model = port.model

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
    _rrp_ver_constraints(type.version, model, sqrt(sigma))

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

    _sd_risk(nothing, type, model, sigma, 1, 1)
    _set_sd_risk_upper_bound(nothing, MinRisk(), type, model, rm.settings.ub, 1, 1)
    _rrp_constraints(type, port, sigma)
    return nothing
end
function _optimise!(type::RRP2, port::Portfolio2, ::Any, ::Any, ::Any,
                    class::Union{Classic2, FM2}, w_ini::AbstractVector, str_names::Bool,
                    save_params::Bool)
    mu, sigma, returns = mu_sigma_returns_class(port, class)

    port.model = JuMP.Model()
    model = port.model
    set_string_names_on_creation(model, str_names)
    initial_w(port, w_ini)

    rrp_constraints(type, port, sigma)
    set_returns(nothing, NoKelly(), nothing, port.model, port.mu_l; mu = mu)
    linear_constraints(port, MinRisk())
    objective_function(port, MinRisk(), type, class, nothing)
    return convex_optimisation(port, nothing, type, class)
end
function optimise2!(port::Portfolio2; rm::Union{AbstractVector, <:TradRiskMeasure} = SD2(),
                    type::PortType = Trad2(), obj::ObjectiveFunction = MinRisk(),
                    kelly::RetType = NoKelly(), class::PortClass = Classic2(),
                    w_ini::AbstractVector = Vector{Float64}(undef, 0),
                    str_names::Bool = false)
    return _optimise!(type, port, rm, obj, kelly, class, w_ini, str_names)
end
export set_rm, MinRisk, Util, SR, MaxRet, Trad2, optimise2!

"""
```julia
optimise!(portfolio::Portfolio2; class::Symbol = :Classic, hist::Integer = 1,
          kelly::Symbol = :None, l::Real = 2.0, obj::Symbol = :Sharpe, rf::Real = 0.0,
          rm::Symbol = :SD, rrp_penalty::Real = 1.0, rrp_ver::Symbol = :None,
          save_opt_params::Bool = true, string_names::Bool = false, type::Symbol = :Trad,
          u_cov::Symbol = :Box, u_mu::Symbol = :Box)
```
"""
function optimise!(portfolio::Portfolio2, opt::OptimiseOpt = OptimiseOpt(;);
                   string_names::Bool = false, save_opt_params::Bool = false)
    type = opt.type
    rm = opt.rm
    obj = opt.obj
    kelly = opt.kelly
    class = opt.class
    rrp_ver = opt.rrp_ver
    u_cov = opt.u_cov
    u_mu = opt.u_mu
    sd_cone = opt.sd_cone
    owa_approx = opt.owa_approx
    near_opt = opt.near_opt
    hist = opt.hist
    rf = opt.rf
    l = opt.l
    rrp_penalty = opt.rrp_penalty
    w_ini = opt.w_ini
    w_min = opt.w_min
    w_max = opt.w_max

    @smart_assert(obj ∈ ObjFuncs)

    if near_opt
        w_min = opt.w_min
        if !isempty(w_min)
            @smart_assert(length(w_min) == size(portfolio.returns, 2))
        end
        w_max = opt.w_max
        if !isempty(w_max)
            @smart_assert(length(w_max) == size(portfolio.returns, 2))
        end
    end
    _p_save_opt_params(portfolio, opt, string_names, save_opt_params)

    mu, sigma, returns = _setup_model_class(portfolio, class, hist)
    T, N = size(returns)
    kurtosis = portfolio.kurt
    skurtosis = portfolio.skurt
    network_method = portfolio.network_method

    portfolio.model = JuMP.Model()
    model = portfolio.model
    set_string_names_on_creation(model, string_names)
    @variable(model, w[1:N])

    if !isempty(w_ini)
        @smart_assert(length(w_ini) == size(portfolio.returns, 2))
        set_start_value.(w, w_ini)
    end

    if type == :Trad
        _setup_sharpe_k(model, obj)
        _risk_setup(portfolio, :Trad, rm, kelly, obj, rf, T, N, mu, returns, sigma,
                    kurtosis, skurtosis, network_method, sd_cone, owa_approx)
        _setup_trad_return(portfolio, class, kelly, obj, T, rf, returns, mu)
        _setup_trad_wc_constraints(portfolio, obj, T, N, :Trad, class, kelly, l, returns)
    elseif type == :RP
        _rp_setup(portfolio, N, class)
        _risk_setup(portfolio, :RP, rm, kelly, obj, rf, T, N, mu, returns, sigma, kurtosis,
                    skurtosis, network_method, sd_cone, owa_approx)
        _setup_rp_rrp_return_and_obj(portfolio, kelly, T, returns, mu)
    elseif type == :RRP
        _setup_risk_budget(portfolio)
        _mv_setup(portfolio, sigma, rm, kelly, obj, :RRP, network_method, sd_cone)
        _rrp_setup(portfolio, sigma, N, rrp_ver, rrp_penalty)
        _setup_rp_rrp_return_and_obj(portfolio, kelly, T, returns, mu)
    else
        _setup_sharpe_k(model, obj)
        _wc_setup(portfolio, kelly, obj, T, N, rf, mu, sigma, u_mu, u_cov, network_method,
                  sd_cone)
        _setup_trad_wc_constraints(portfolio, obj, T, N, :WC, class, kelly, l, returns)
    end

    _setup_linear_constraints(portfolio, obj, type)

    term_status, solvers_tried = _optimise_portfolio(portfolio, class, type, obj)
    retval = _handle_errors_and_finalise(portfolio, class, term_status, returns, N,
                                         solvers_tried, type, rm, obj)

    if near_opt && type ∈ (:Trad, :WC)
        retval = _near_optimal_centering(portfolio, class, mu, returns, sigma, retval, T, N,
                                         opt)
    end

    return retval
end

"""
```julia
frontier_limits!(portfolio::Portfolio2; class::Symbol = :Classic, hist::Integer = 1,
                 kelly::Symbol = :None, rf::Real = 0.0, rm::Symbol = :SD,
                 save_model::Bool = false)
```
"""
function frontier_limits!(portfolio::Portfolio2, opt::OptimiseOpt = OptimiseOpt(;);
                          save_model::Bool = false)
    obj1 = opt.obj
    near_opt1 = opt.near_opt
    opt.near_opt = false
    optimal1 = deepcopy(portfolio.optimal)
    fail1 = deepcopy(portfolio.fail)
    if save_model
        model1 = copy(portfolio.model)
    end

    opt.obj = :Min_Risk
    w_min = optimise!(portfolio, opt)

    opt.obj = :Max_Ret
    w_max = optimise!(portfolio, opt)

    limits = hcat(w_min, DataFrame(; x1 = w_max[!, 2]))
    DataFrames.rename!(limits, :weights => :w_min, :x1 => :w_max)
    portfolio.limits[opt.rm] = limits

    opt.obj = obj1
    opt.near_opt = near_opt1
    portfolio.optimal = optimal1
    portfolio.fail = fail1
    if save_model
        portfolio.model = model1
    end

    return portfolio.limits[opt.rm]
end

"""
```julia
efficient_frontier!(portfolio::Portfolio2; class::Symbol = :Classic, hist::Integer = 1,
                    kelly::Symbol = :None, rf::Real = 0.0, rm::Symbol = :SD,
                    points::Integer = 20)
```
"""
function efficient_frontier!(portfolio::Portfolio2, opt::OptimiseOpt = OptimiseOpt(;);
                             points::Integer = 20)
    @smart_assert(opt.type == :Trad)
    obj1 = opt.obj
    w_ini1 = opt.w_ini
    optimal1 = deepcopy(portfolio.optimal)
    fail1 = deepcopy(portfolio.fail)

    class = opt.class
    hist = opt.hist
    mu, sigma, returns = _setup_model_class(portfolio, class, hist)

    fl = frontier_limits!(portfolio, opt)

    w1 = fl.w_min
    w2 = fl.w_max

    if opt.kelly == :None
        ret1 = dot(mu, w1)
        ret2 = dot(mu, w2)
    else
        ret1 = sum(log.(one(eltype(mu)) .+ returns * w1)) / size(returns, 1)
        ret2 = sum(log.(one(eltype(mu)) .+ returns * w2)) / size(returns, 1)
    end

    alpha_i = portfolio.alpha_i
    alpha = portfolio.alpha
    a_sim = portfolio.a_sim
    beta_i = portfolio.beta_i
    beta = portfolio.beta
    b_sim = portfolio.b_sim
    kappa = portfolio.kappa
    owa_w = portfolio.owa_w
    V = portfolio.V
    SV = portfolio.SV
    solvers = portfolio.solvers

    rm = opt.rm
    rf = opt.rf

    risk1, risk2 = _ul_risk(rm, returns, w1, w2, sigma, rf, solvers, alpha, kappa, alpha_i,
                            beta, a_sim, beta_i, b_sim, owa_w, V, SV, 0)

    mus = range(ret1; stop = ret2, length = points)
    risks = range(risk1; stop = risk2, length = points)

    rmf = Symbol(lowercase(string(rm)) * "_u")

    frontier = Vector{typeof(risk1)}(undef, 0)
    srisk = Vector{typeof(risk1)}(undef, 0)
    w_ini = Vector{typeof(risk1)}(undef, 0)

    i = 0
    for (j, (r, m)) ∈ enumerate(zip(risks, mus))
        if i == 0
            opt.obj = :Min_Risk
            w = optimise!(portfolio, opt)
        else
            if !isempty(w)
                opt.w_ini = w.weights
            end
            if j != length(risks)
                setproperty!(portfolio, rmf, r)
            else
                setproperty!(portfolio, rmf, Inf)
            end
            opt.obj = :Max_Ret
            w = optimise!(portfolio, opt)
            # Fallback in case :Max_Ret with maximum risk bounds fails.
            if isempty(w)
                opt.obj = :Min_Risk
                setproperty!(portfolio, rmf, Inf)
                portfolio.mu_l = m
                w = optimise!(portfolio, opt)
                portfolio.mu_l = Inf
            end
        end
        if isempty(w)
            continue
        end
        rk = calc_risk(w.weights, returns; rm = rm, rf = rf, sigma = sigma,
                       alpha_i = alpha_i, alpha = alpha, a_sim = a_sim, beta_i = beta_i,
                       beta = beta, b_sim = b_sim, kappa = kappa, owa_w = owa_w, V = V,
                       SV = SV, solvers = solvers)

        append!(frontier, w.weights)
        push!(srisk, rk)
        i += 1
    end
    setproperty!(portfolio, rmf, Inf)

    opt.obj = :Sharpe
    w = optimise!(portfolio, opt)
    sharpe = false
    if !isempty(w)
        rk = calc_risk(w.weights, returns; rm = rm, rf = rf, sigma = sigma,
                       alpha_i = alpha_i, alpha = alpha, a_sim = a_sim, beta_i = beta_i,
                       beta = beta, b_sim = b_sim, kappa = kappa, owa_w = owa_w, V = V,
                       SV = SV, solvers = solvers)
        append!(frontier, w.weights)
        push!(srisk, rk)
        i += 1
        sharpe = true
    end

    key = if opt.near_opt
        Symbol("Near_" * string(rm))
    else
        rm
    end

    portfolio.frontier[key] = Dict(:weights => hcat(DataFrame(; tickers = portfolio.assets),
                                                    DataFrame(reshape(frontier, length(w1),
                                                                      :),
                                                              string.(range(1, i)))),
                                   :opt => opt, :points => points, :risk => srisk,
                                   :sharpe => sharpe)

    opt.obj = obj1
    opt.w_ini = w_ini1
    portfolio.optimal = optimal1
    portfolio.fail = fail1

    return portfolio.frontier[key]
end
