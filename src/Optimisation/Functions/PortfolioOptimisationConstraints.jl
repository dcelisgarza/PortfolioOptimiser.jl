function num_assets_constraints(port, ::Sharpe)
    if port.num_assets_u > 0
        N = size(port.returns, 2)
        model = port.model

        @variable(model, tnau_bin[1:N], binary = true)
        @constraint(model, sum(tnau_bin) <= port.num_assets_u)
        # Sharpe ratio
        @variable(model, tnau_bin_sharpe[1:N] .>= 0)
        k = model[:k]
        @constraint(model, tnau_bin_sharpe .<= k)
        @constraint(model, tnau_bin_sharpe .<= port.num_assets_u_scale * tnau_bin)
        @constraint(model,
                    tnau_bin_sharpe .>= k .- port.num_assets_u_scale * (1 .- tnau_bin))
        # Long and short
        w = model[:w]
        if !port.short
            @constraint(model, w .<= port.long_u * tnau_bin_sharpe)
        else
            @constraint(model,
                        w .<=
                        min(port.long_u, port.budget + port.short_budget) * tnau_bin_sharpe)
            @constraint(model,
                        w .>= -min(port.short_u, port.short_budget) * tnau_bin_sharpe)
        end
    end
    if port.num_assets_l > 0
        N = size(port.returns, 2)
        model = port.model
        @variable(model, tnal >= 0)
        w = model[:w]
        @constraint(model, [tnal; w] ∈ SecondOrderCone())
        k = model[:k]
        @constraint(model, tnal * sqrt(port.num_assets_l) <= k)
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
        w = model[:w]
        if !port.short
            @constraint(model, w .<= port.long_u * tnau_bin)
        else
            @constraint(model,
                        w .<= min(port.long_u, port.budget + port.short_budget) * tnau_bin)
            @constraint(model, w .>= -min(port.short_u, port.short_budget) * tnau_bin)
        end
    end
    if port.num_assets_l > 0
        N = size(port.returns, 2)
        model = port.model
        @variable(model, tnal >= 0)
        w = model[:w]
        @constraint(model, [tnal; w] ∈ SecondOrderCone())
        @constraint(model, tnal * sqrt(port.num_assets_l) <= 1)
    end
    return nothing
end
function weight_constraints(port, ::Sharpe)
    N = size(port.returns, 2)
    model = port.model
    w = model[:w]
    k = model[:k]
    @constraint(model, sum(w) == port.budget * k)
    if !port.short
        @constraint(model, w .<= port.long_u * k)
        @constraint(model, w .>= 0)
    else
        @variable(model, tw_ulong[1:N] .>= 0)
        @variable(model, tw_ushort[1:N] .>= 0)

        @constraint(model, sum(tw_ulong) <= (port.budget + port.short_u) * k)
        @constraint(model, sum(tw_ushort) <= port.short_budget * k)

        @constraint(model, w .<= tw_ulong)
        @constraint(model, w .>= -tw_ushort)
    end
    return nothing
end
function weight_constraints(port, ::Any)
    N = size(port.returns, 2)
    model = port.model
    w = model[:w]
    @constraint(model, sum(w) == port.budget)
    if !port.short
        @constraint(model, w .<= port.long_u)
        @constraint(model, w .>= 0)
    else
        @variable(model, tw_ulong[1:N] .>= 0)
        @variable(model, tw_ushort[1:N] .>= 0)

        @constraint(model, sum(tw_ulong) <= port.budget + port.short_u)
        @constraint(model, sum(tw_ushort) <= port.short_budget)

        @constraint(model, w .<= tw_ulong)
        @constraint(model, w .>= -tw_ushort)
    end
    return nothing
end
function network_constraints(args...)
    return nothing
end
function network_constraints(network::IP, port, ::Sharpe, ::Any)
    N = size(port.returns, 2)
    model = port.model

    @variable(model, tip_bin[1:N], binary = true)
    @constraint(model, network.A * tip_bin .<= network.k)
    # Sharpe ratio
    @variable(model, tip_bin_sharpe[1:N] .>= 0)
    k = model[:k]
    @constraint(model, tip_bin_sharpe .<= k)
    @constraint(model, tip_bin_sharpe .<= network.scale * tip_bin)
    @constraint(model, tip_bin_sharpe .>= k .- network.scale * (1 .- tip_bin))
    # Long and short
    w = model[:w]
    if !port.short
        @constraint(model, w .<= port.long_u * tip_bin_sharpe)
    else
        @constraint(model,
                    w .<=
                    min(port.long_u, port.budget + port.short_budget) * tip_bin_sharpe)
        @constraint(model, w .>= -min(port.short_u, port.short_budget) * tip_bin_sharpe)
    end
    return nothing
end
function network_constraints(network::IP, port, ::Any, ::Any)
    N = size(port.returns, 2)
    model = port.model

    @variable(model, tip_bin[1:N], binary = true)
    @constraint(model, network.A * tip_bin .<= network.k)
    # Long and short
    w = model[:w]
    if !port.short
        @constraint(model, w .<= port.long_u * tip_bin)
    else
        @constraint(model,
                    w .<= min(port.long_u, port.budget + port.short_budget) * tip_bin)
        @constraint(model, w .>= -min(port.short_u, port.short_budget) * tip_bin)
    end
    return nothing
end
function network_constraints(network::SDP, port, obj, ::Trad)
    _sdp(port, obj)
    W = port.model[:W]
    @constraint(port.model, network.A .* W .== 0)
    if !haskey(port.model, :sd_risk)
        @expression(port.model, network_penalty, network.penalty * tr(W))
    end
    return nothing
end
function network_constraints(network::SDP, port, obj, ::WC)
    _sdp(port, obj)
    W = port.model[:W]
    @constraint(port.model, network.A .* W .== 0)
    return nothing
end
###########################
###########################
function cluster_constraints(args...)
    return nothing
end
function cluster_constraints(cluster::IP, port, ::Sharpe, ::Any)
    N = size(port.returns, 2)
    model = port.model

    @variable(model, tip_bin2[1:N], binary = true)
    @constraint(model, cluster.A * tip_bin2 .<= cluster.k)
    # Sharpe ratio
    @variable(model, tip_bin_sharpe2[1:N] .>= 0)
    k = model[:k]
    @constraint(model, tip_bin_sharpe2 .<= k)
    @constraint(model, tip_bin_sharpe2 .<= cluster.scale * tip_bin2)
    @constraint(model, tip_bin_sharpe2 .>= k .- cluster.scale * (1 .- tip_bin2))
    # Long and short
    w = model[:w]
    if !port.short
        @constraint(model, w .<= port.long_u * tip_bin_sharpe2)
    else
        @constraint(model,
                    w .<=
                    min(port.long_u, port.budget + port.short_budget) * tip_bin_sharpe2)
        @constraint(model, w .>= -min(port.short_u, port.short_budget) * tip_bin_sharpe2)
    end
    return nothing
end
function cluster_constraints(cluster::IP, port, ::Any, ::Any)
    N = size(port.returns, 2)
    model = port.model

    @variable(model, tip_bin2[1:N], binary = true)
    @constraint(model, cluster.A * tip_bin2 .<= cluster.k)
    # Long and short
    w = model[:w]
    if !port.short
        @constraint(model, w .<= port.long_u * tip_bin2)
    else
        @constraint(model,
                    w .<= min(port.long_u, port.budget + port.short_budget) * tip_bin2)
        @constraint(model, w .>= -min(port.short_u, port.short_budget) * tip_bin2)
    end
    return nothing
end
function cluster_constraints(cluster::SDP, port, obj, ::Trad)
    _sdp(port, obj)
    W = port.model[:W]
    @constraint(port.model, cluster.A .* W .== 0)
    if !haskey(port.model, :sd_risk)
        @expression(port.model, cluster_penalty, cluster.penalty * tr(W))
    end
    return nothing
end
function cluster_constraints(cluster::SDP, port, obj, ::WC)
    _sdp(port, obj)
    W = port.model[:W]
    @constraint(port.model, cluster.A .* W .== 0)
    return nothing
end
###########################
###########################
function _centrality_constraints(::Sharpe, model, A, B)
    w = model[:w]
    k = model[:k]
    @constraint(model, dot(A, w) - B * k == 0)
    return nothing
end
function _centrality_constraints(::Any, model, A, B)
    w = model[:w]
    @constraint(model, dot(A, w) - B == 0)
    return nothing
end
function centrality_constraints(port, obj)
    if !(isempty(port.a_vec_cent) || isinf(port.b_cent))
        _centrality_constraints(obj, port.model, port.a_vec_cent, port.b_cent)
    end
    return nothing
end
function _linear_constraints(::Union{Sharpe, RP}, model, A, B)
    w = model[:w]
    k = model[:k]
    @constraint(model, A * w .- B * k .>= 0)
    return nothing
end
function _linear_constraints(::Any, model, A, B)
    w = model[:w]
    @constraint(model, A * w .- B .>= 0)
    return nothing
end
function linear_constraints(port, obj_type)
    if !(isempty(port.a_mtx_ineq) || isempty(port.b_vec_ineq))
        _linear_constraints(obj_type, port.model, port.a_mtx_ineq, port.b_vec_ineq)
    end
    return nothing
end
function _tracking_err_constraints(::Any, model, returns, tracking_err, benchmark)
    T = size(returns, 1)
    @variable(model, t_track_err >= 0)
    w = model[:w]
    @expression(model, track_err, returns * w .- benchmark)
    @constraint(model, [t_track_err; track_err] ∈ SecondOrderCone())
    @constraint(model, t_track_err <= tracking_err * sqrt(T - 1))
    return nothing
end
function _tracking_err_constraints(::Sharpe, model, returns, tracking_err, benchmark)
    T = size(returns, 1)
    @variable(model, t_track_err >= 0)
    w = model[:w]
    k = model[:k]
    @expression(model, track_err, returns * w .- benchmark * k)
    @constraint(model, [t_track_err; track_err] ∈ SecondOrderCone())
    @constraint(model, t_track_err <= tracking_err * k * sqrt(T - 1))
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
    w = model[:w]
    @expression(model, turnov, w .- turnover.w)
    @constraint(model, [i = 1:N], [t_turnov[i]; turnov[i]] ∈ MOI.NormOneCone(2))
    @constraint(model, t_turnov .<= turnover.val)
    return nothing
end
function _turnover_constraints(::Sharpe, model, turnover)
    N = length(turnover.w)
    @variable(model, t_turnov[1:N] >= 0)
    w = model[:w]
    k = model[:k]
    @expression(model, turnov, w .- turnover.w * k)
    @constraint(model, [i = 1:N], [t_turnov[i]; turnov[i]] ∈ MOI.NormOneCone(2))
    @constraint(model, t_turnov .<= turnover.val * k)
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
    w = model[:w]
    @expression(model, rebal, w .- rebalance.w)
    @constraint(model, [i = 1:N], [t_rebal[i]; rebal[i]] ∈ MOI.NormOneCone(2))
    @expression(model, sum_t_rebal, sum(rebalance.val .* t_rebal))
    return nothing
end
function _rebalance_constraints(::Sharpe, model, rebalance)
    N = length(rebalance.w)
    @variable(model, t_rebal[1:N] >= 0)
    w = model[:w]
    k = model[:k]
    @expression(model, rebal, w .- rebalance.w * k)
    @constraint(model, [i = 1:N], [t_rebal[i]; rebal[i]] ∈ MOI.NormOneCone(2))
    @expression(model, sum_t_rebal, sum(rebalance.val .* t_rebal))
    return nothing
end
function rebalance_constraints(turnover::NoTR, ::Any, ::Any)
    return nothing
end
function rebalance_constraints(rebalance::TR, port, obj)
    if !(isa(rebalance.val, Real) && iszero(rebalance.val) ||
         isa(rebalance.val, AbstractVector) && isempty(rebalance.val) ||
         isempty(rebalance.w))
        _rebalance_constraints(obj, port.model, port.rebalance)
    end
    return nothing
end
