function num_assets_constraints(port, ::Sharpe)
    if size(port.returns, 2) > port.num_assets_u > 0
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
            @constraint(model, w .<= port.long_u * tnau_bin_sharpe)
            @constraint(model, w .>= -port.short_u * tnau_bin_sharpe)
        end
    end
    if !isnothing(port.group_num_assets_u)
        group_num_assets_u = port.group_num_assets_u
        model = port.model
        w = model[:w]
        k = model[:k]
        for gnau ∈ group_num_assets_u
            assets = gnau.assets
            if !isempty(assets)
                N = length(assets)
                number = group_num_assets_u.number
                if number < N
                    if !gnau.flag
                        assets = [findfirst(x -> x == x, port.assets) for x ∈ assets]
                    end
                    scale = group_num_assets_u.scale
                    tgnau_bin = @variable(model, [1:N], binary = true)
                    @constraint(model, sum(tgnau_bin) <= number)
                    # Sharpe ratio
                    tgnau_bin_sharpe = @variable(model, [1:N], lower_bound = 0)
                    @constraint(model, tgnau_bin_sharpe .<= k)
                    @constraint(model, tgnau_bin_sharpe .<= scale * tgnau_bin)
                    @constraint(model, tgnau_bin_sharpe .>= k .- scale * (1 .- tgnau_bin))
                    # Long and short
                    wa = w[assets]
                    if !port.short
                        @constraint(model, wa .<= port.long_u * tgnau_bin_sharpe)
                    else
                        @constraint(model, wa .<= port.long_u * tgnau_bin_sharpe)
                        @constraint(model, wa .>= -port.short_u * tgnau_bin_sharpe)
                    end
                end
            end
        end
    end
    if port.num_assets_l > 0
        model = port.model
        @variable(model, tnal >= 0)
        w = model[:w]
        @constraint(model, [tnal; w] ∈ SecondOrderCone())
        k = model[:k]
        @constraint(model, tnal * sqrt(port.num_assets_l) <= k)
    end
    if !isnothing(port.group_num_assets_l)
        group_num_assets_l = port.group_num_assets_l
        model = port.model
        w = model[:w]
        k = model[:k]
        for gnal ∈ group_num_assets_l
            assets = gnal.assets
            if !isempty(assets)
                if !gnal.flag
                    assets = [findfirst(x -> x == x, port.assets) for x ∈ assets]
                end
                number = gnal.number
                tgnal = @variable(model, lower_bound = 0)
                wa = w[assets]
                @constraint(model, [tgnal; wa] ∈ SecondOrderCone())
                @constraint(model, tgnal * sqrt(number) <= k)
            end
        end
    end
    return nothing
end
function num_assets_constraints(port, ::Any)
    if size(port.returns, 2) > port.num_assets_u > 0
        N = size(port.returns, 2)
        model = port.model
        @variable(model, tnau_bin[1:N], binary = true)
        @constraint(model, sum(tnau_bin) <= port.num_assets_u)
        # Long and short
        w = model[:w]
        if !port.short
            @constraint(model, w .<= port.long_u * tnau_bin)
        else
            @constraint(model, w .<= port.long_u * tnau_bin)
            @constraint(model, w .>= -port.short_u * tnau_bin)
        end
    end
    if !isnothing(port.group_num_assets_u)
        group_num_assets_u = port.group_num_assets_u
        model = port.model
        w = model[:w]
        for gnau ∈ group_num_assets_u
            assets = gnau.assets
            if !isempty(assets)
                N = length(assets)
                number = gnau.number
                if number < N
                    if !gnau.flag
                        assets = [findfirst(x -> x == x, port.assets) for x ∈ assets]
                    end
                    tgnau_bin = @variable(model, [1:N], binary = true)
                    @constraint(model, sum(tgnau_bin) <= number)
                    # Long and short
                    wa = w[assets]
                    if !port.short
                        @constraint(model, wa .<= port.long_u * tgnau_bin)
                    else
                        @constraint(model, wa .<= port.long_u * tgnau_bin)
                        @constraint(model, wa .>= -port.short_u * tgnau_bin)
                    end
                end
            end
        end
    end
    if port.num_assets_l > 0
        model = port.model
        @variable(model, tnal >= 0)
        w = model[:w]
        @constraint(model, [tnal; w] ∈ SecondOrderCone())
        @constraint(model, tnal * sqrt(port.num_assets_l) <= 1)
    end
    if !isnothing(port.group_num_assets_l)
        group_num_assets_l = port.group_num_assets_l
        model = port.model
        w = model[:w]
        for gnal ∈ group_num_assets_l
            assets = gnal.assets
            if !isempty(assets)
                if !gnal.flag
                    assets = [findfirst(x -> x == x, port.assets) for x ∈ assets]
                end
                number = gnal.number
                tgnal = @variable(model, lower_bound = 0)
                wa = w[assets]
                @constraint(model, [tgnal; wa] ∈ SecondOrderCone())
                @constraint(model, tgnal * sqrt(number) <= 1)
            end
        end
    end
    return nothing
end
function management_fees(fees, model, w)
    if !(isa(fees, Real) && iszero(fees) ||
         isa(fees, AbstractVector) && isempty(fees) ||
         isa(fees, AbstractVector) && all(iszero.(fees)))
        @expression(model, total_fee, sum(fees .* w))
        if !haskey(model, :obj_penalty)
            @expression(model, obj_penalty, zero(AffExpr))
        end
        add_to_expression!(model[:obj_penalty], total_fee)
    end
    return nothing
end
function short_long_management_fees(fees, short_fees, model, long_w, short_w)
    if !(isa(fees, Real) && iszero(fees) ||
         isa(fees, AbstractVector) && isempty(fees) ||
         isa(fees, AbstractVector) && all(iszero.(fees)))
        @expression(model, long_fee, sum(fees .* long_w))
        if !haskey(model, :obj_penalty)
            @expression(model, obj_penalty, zero(AffExpr))
        end
        add_to_expression!(model[:obj_penalty], long_fee)
    end

    if !(isa(short_fees, Real) && iszero(short_fees) ||
         isa(short_fees, AbstractVector) && isempty(short_fees) ||
         isa(short_fees, AbstractVector) && all(iszero.(short_fees)))
        @expression(model, short_fee, sum(short_fees .* short_w))
        if !haskey(model, :obj_penalty)
            @expression(model, obj_penalty, zero(AffExpr))
        end
        add_to_expression!(model[:obj_penalty], short_fee)
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

        management_fees(port.fees, model, w)
    else
        @variable(model, tw_ulong[1:N] .>= 0)
        @variable(model, tw_ushort[1:N] .>= 0)

        @constraint(model, sum(tw_ulong) <= (port.budget + port.short_budget) * k)
        @constraint(model, sum(tw_ushort) <= port.short_budget * k)

        @constraint(model, w .<= tw_ulong)
        @constraint(model, w .>= -tw_ushort)

        short_long_management_fees(port.fees, port.short_fees, model, tw_ulong, tw_ushort)
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

        management_fees(port.fees, model, w)
    else
        @variable(model, tw_ulong[1:N] .>= 0)
        @variable(model, tw_ushort[1:N] .>= 0)

        @constraint(model, sum(tw_ulong) <= port.budget + port.short_budget)
        @constraint(model, sum(tw_ushort) <= port.short_budget)

        @constraint(model, w .<= tw_ulong)
        @constraint(model, w .>= -tw_ushort)

        short_long_management_fees(port.fees, port.short_fees, model, tw_ulong, tw_ushort)
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
        @constraint(model, w .<= port.long_u * tip_bin_sharpe)
        @constraint(model, w .>= -port.short_u * tip_bin_sharpe)
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
        @constraint(model, w .<= port.long_u * tip_bin)
        @constraint(model, w .>= -port.short_u * tip_bin)
    end
    return nothing
end
function network_constraints(network::SDP, port, obj, ::Trad)
    _sdp(port, obj)
    model = port.model
    W = model[:W]
    @constraint(model, network.A .* W .== 0)
    if !haskey(model, :sd_risk)
        @expression(model, network_penalty, network.penalty * tr(W))
        if !haskey(model, :obj_penalty)
            @expression(model, obj_penalty, zero(AffExpr))
        end
        add_to_expression!(model[:obj_penalty], network_penalty)
    end
    return nothing
end
function network_constraints(network::SDP, port, obj, ::WC)
    _sdp(port, obj)
    W = port.model[:W]
    @constraint(port.model, network.A .* W .== 0)
    return nothing
end
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
        @constraint(model, w .<= port.long_u * tip_bin_sharpe2)
        @constraint(model, w .>= -port.short_u * tip_bin_sharpe2)
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
        @constraint(model, w .<= port.long_u * tip_bin2)
        @constraint(model, w .>= -port.short_u * tip_bin2)
    end
    return nothing
end
function cluster_constraints(cluster::SDP, port, obj, ::Trad)
    _sdp(port, obj)
    model = port.model
    W = model[:W]
    @constraint(model, cluster.A .* W .== 0)
    if !haskey(model, :sd_risk)
        @expression(model, cluster_penalty, cluster.penalty * tr(W))
        if !haskey(model, :obj_penalty)
            @expression(model, obj_penalty, zero(AffExpr))
        end
        add_to_expression!(model[:obj_penalty], cluster_penalty)
    end
    return nothing
end
function cluster_constraints(cluster::SDP, port, obj, ::WC)
    _sdp(port, obj)
    W = port.model[:W]
    @constraint(port.model, cluster.A .* W .== 0)
    return nothing
end
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
    if !(isempty(port.a_vec_cent) || iszero(port.b_cent))
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
function turnover_constraints(::NoTR, ::Any, ::Any)
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
function _rebalance_penalty(::Any, model, rebalance)
    N = length(rebalance.w)
    @variable(model, t_rebal[1:N] >= 0)
    w = model[:w]
    @expression(model, rebal, w .- rebalance.w)
    @constraint(model, [i = 1:N], [t_rebal[i]; rebal[i]] ∈ MOI.NormOneCone(2))
    @expression(model, rebalance_penalty, sum(rebalance.val .* t_rebal))
    if !haskey(model, :obj_penalty)
        @expression(model, obj_penalty, zero(AffExpr))
    end
    add_to_expression!(model[:obj_penalty], rebalance_penalty)
    return nothing
end
function _rebalance_penalty(::Sharpe, model, rebalance)
    N = length(rebalance.w)
    @variable(model, t_rebal[1:N] >= 0)
    w = model[:w]
    k = model[:k]
    @expression(model, rebal, w .- rebalance.w * k)
    @constraint(model, [i = 1:N], [t_rebal[i]; rebal[i]] ∈ MOI.NormOneCone(2))
    @expression(model, rebalance_penalty, sum(rebalance.val .* t_rebal))
    if !haskey(model, :obj_penalty)
        @expression(model, obj_penalty, zero(AffExpr))
    end
    add_to_expression!(model[:obj_penalty], rebalance_penalty)
    return nothing
end
function rebalance_penalty(::NoTR, ::Any, ::Any)
    return nothing
end
function rebalance_penalty(rebalance::TR, port, obj)
    if !(isa(rebalance.val, Real) && iszero(rebalance.val) ||
         isa(rebalance.val, AbstractVector) && isempty(rebalance.val) ||
         isempty(rebalance.w))
        _rebalance_penalty(obj, port.model, port.rebalance)
    end
    return nothing
end
function L1_reg(port)
    if !iszero(port.l1)
        model = port.model
        w = model[:w]
        @variable(model, t_l1 >= 0)
        @constraint(model, [t_l1; w] in MOI.NormOneCone(1 + length(w)))
        @expression(model, l1_reg, port.l1 * t_l1)
        if !haskey(model, :obj_penalty)
            @expression(model, obj_penalty, zero(AffExpr))
        end
        add_to_expression!(model[:obj_penalty], l1_reg)
    end
    return nothing
end
function L2_reg(port)
    if !iszero(port.l2)
        model = port.model
        w = model[:w]
        @variable(model, t_l2 >= 0)
        @constraint(model, [t_l2; w] in SecondOrderCone())
        @expression(model, l2_reg, port.l2 * t_l2)
        if !haskey(model, :obj_penalty)
            @expression(model, obj_penalty, zero(AffExpr))
        end
        add_to_expression!(model[:obj_penalty], l2_reg)
    end
    return nothing
end
function custom_constraint_objective_penatly(::Nothing, port)
    return nothing
end
