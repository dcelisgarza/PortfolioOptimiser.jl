# Copywrite (c) 2025
# Author: Daniel Celis Garza <daniel.celis.garza@gmail.com>
# SPDX-License-Identifier: MIT

function get_portfolio_returns(model, returns)
    if haskey(model, :X)
        return nothing
    end
    w = model[:w]
    @expression(model, X, returns * w)

    return nothing
end
function long_w_budget(budget_flag, budget_lb_flag, budget_ub_flag, budget_lb, budget,
                       budget_ub, short_budget, model, k, long_w, scale_constr, key)
    if budget_flag
        model[Symbol("constr_budget_$(key)")] = @constraint(model,
                                                            scale_constr * sum(long_w) ==
                                                            scale_constr *
                                                            (budget - short_budget) *
                                                            k)
    else
        if budget_lb_flag
            model[Symbol("constr_budget_lb_$(key)")] = @constraint(model,
                                                                   scale_constr *
                                                                   sum(long_w) >=
                                                                   scale_constr *
                                                                   (budget_lb -
                                                                    short_budget) *
                                                                   k)
        end
        if budget_ub_flag
            model[Symbol("constr_budget_ub_$(key)")] = @constraint(model,
                                                                   scale_constr *
                                                                   sum(long_w) <=
                                                                   scale_constr *
                                                                   (budget_ub -
                                                                    short_budget) *
                                                                   k)
        end
    end

    return nothing
end
function weight_constraints(port, allow_shorting::Bool = true)
    #=
    # Weight constraints
    =#
    model = port.model
    scale_constr = model[:scale_constr]
    w = model[:w]
    k = model[:k]
    N = length(w)

    #=
    ## Portfolio budget constraints
    =#
    budget_lb = port.budget_lb
    budget = port.budget
    budget_ub = port.budget_ub
    budget_lb_flag = isfinite(budget_lb)
    budget_flag = isfinite(budget)
    budget_ub_flag = isfinite(budget_ub)
    if budget_flag
        @constraint(model, constr_budget,
                    scale_constr * sum(w) == scale_constr * budget * k)
    else
        if budget_lb_flag
            @constraint(model, constr_budget_lb,
                        scale_constr * sum(w) >= scale_constr * budget_lb * k)
        end
        if budget_ub_flag
            @constraint(model, constr_budget_ub,
                        scale_constr * sum(w) <= scale_constr * budget_ub * k)
        end
    end

    #=
    ## Min and max weights, short budget weights.
    =#
    short = port.short
    long_ub = port.long_ub
    if !short
        @constraints(model,
                     begin
                         constr_w_ub, scale_constr * w .<= scale_constr * long_ub * k
                         constr_w_lb, w .>= 0
                     end)
        @expression(model, long_w, w)
    elseif short && allow_shorting
        #=
        ## Short min and max weights
        =#
        short_lb = port.short_lb
        short_budget_ub = port.short_budget_ub
        short_budget = port.short_budget
        short_budget_lb = port.short_budget_lb

        @variables(model, begin
                       long_w[1:N] .>= 0
                       short_w[1:N] .<= 0
                   end)

        @constraints(model,
                     begin
                         constr_w_ub, scale_constr * w .<= scale_constr * long_ub * k
                         constr_w_lb, scale_constr * w .>= scale_constr * short_lb * k
                         constr_long_w_ub, scale_constr * w .<= scale_constr * long_w
                         constr_short_w_lb, scale_constr * w .>= scale_constr * short_w
                     end)

        #=
        ## Long-short portfolio budget constraints
        =#
        short_budget_ub_flag = isfinite(short_budget_ub)
        short_budget_flag = isfinite(short_budget)
        short_budget_lb_flag = isfinite(short_budget_lb)
        if short_budget_flag
            long_w_budget(budget_flag, budget_lb_flag, budget_ub_flag, budget_lb, budget,
                          budget_ub, short_budget, model, k, long_w, scale_constr,
                          "short_budget")
            @constraint(model, constr_short_budget,
                        scale_constr * sum(short_w) == scale_constr * short_budget * k)
        else
            if short_budget_ub_flag
                long_w_budget(budget_flag, budget_lb_flag, budget_ub_flag, budget_lb,
                              budget, budget_ub, short_budget_ub, model, k, long_w,
                              scale_constr, "short_budget_ub")
                @constraint(model, constr_short_budget_ub,
                            scale_constr * sum(short_w) <=
                            scale_constr * short_budget_ub * k)
            end
            if short_budget_lb_flag
                long_w_budget(budget_flag, budget_lb_flag, budget_ub_flag, budget_lb,
                              budget, budget_ub, short_budget_lb, model, k, long_w,
                              scale_constr, "short_budget_lb")
                @constraint(model, constr_short_budget_lb,
                            scale_constr * sum(short_w) >=
                            scale_constr * short_budget_lb * k)
            end
        end
    end

    #=
    ## Number of effective assets
    =#
    nea = port.nea
    if nea > zero(nea)
        @variable(model, nea_var)
        @constraints(model,
                     begin
                         constr_nea_soc,
                         [scale_constr * nea_var; scale_constr * w] ∈ SecondOrderCone()
                         constr_nea, scale_constr * nea_var * sqrt(nea) <= scale_constr * k
                     end)
    end

    #=
    ## Linear constraints
    =#
    A = port.a_ineq
    B = port.b_ineq
    if !(isempty(A) || isempty(B))
        @constraint(model, constr_ineq, scale_constr * A * w .>= scale_constr * B * k)
    end
    A = port.a_eq
    B = port.b_eq
    if !(isempty(A) || isempty(B))
        @constraint(model, constr_eq, scale_constr * A * w .== scale_constr * B * k)
    end

    return nothing
end
function MIP_constraints(port, allow_shorting::Bool = true)
    #=
    # MIP constraints

    ## Flags

    Flags for deciding whether the problem is MIP.
    =#
    card = port.card
    a_card_ineq = port.a_card_ineq
    b_card_ineq = port.b_card_ineq
    a_card_eq = port.a_card_eq
    b_card_eq = port.b_card_eq
    network_adj = port.network_adj
    cluster_adj = port.cluster_adj

    card_flag = size(port.returns, 2) > card > 0
    gcard_ineq_flag = !(isempty(a_card_ineq) || isempty(b_card_ineq))
    gcard_eq_flag = !(isempty(a_card_eq) || isempty(b_card_eq))
    ntwk_flag = isa(network_adj, IP)
    clst_flag = isa(cluster_adj, IP)
    if !(card_flag || gcard_ineq_flag || gcard_eq_flag || ntwk_flag || clst_flag)
        return nothing
    end

    model = port.model
    scale_constr = model[:scale_constr]
    w = model[:w]
    k = model[:k]
    N = length(w)

    #=
    ## Universal MIP variables

    Variables used for all MIP constraints.

    ### Short and short threshold

    - If shorting enabled and short threshold is finite, we need separate boolean variables for the short and long sides of the portfolio.
    - If no short threshold is infinite (no threshold) we can use a single set of booleans.

    ### Objective function

    - `k` is a constant if the objective is not [`Sharpe`](@ref).
    - `k` is a variable if the objective is [`Sharpe`](@ref).
        - Extra variables are needed.
    =#
    short = port.short
    long_lb = port.long_lb
    long_ub = port.long_ub
    short_ub = port.short_ub
    short_lb = port.short_lb
    if (isa(short_ub, Real) && !iszero(short_ub) ||
        isa(short_ub, AbstractVector) && (!isempty(short_ub) || any(.!iszero(short_ub)))) &&
       short &&
       allow_shorting
        scale = port.card_scale
        @variables(model, begin
                       is_invested_long_bool[1:N], (binary = true)
                       is_invested_short_bool[1:N], (binary = true)
                   end)
        @expression(model, is_invested_bool, is_invested_long_bool + is_invested_short_bool)
        if isa(k, Real)
            @expressions(model, begin
                             is_invested_long, is_invested_long_bool
                             is_invested_short, is_invested_short_bool
                         end)
        else
            @variables(model, begin
                           is_invested_long_float[1:N] .>= 0
                           is_invested_short_float[1:N] .>= 0
                       end)
            @constraints(model,
                         begin
                             constr_is_invested_long_float_ub, is_invested_long_float .<= k
                             constr_is_invested_short_float_ub,
                             is_invested_short_float .<= k
                             constr_is_invested_long_float_decision_ub,
                             is_invested_long_float .<= scale * is_invested_long_bool
                             constr_is_invested_short_float_decision_ub,
                             is_invested_short_float .<= scale * is_invested_short_bool
                             constr_is_invested_long_float_decision_lb,
                             is_invested_long_float .>=
                             k .- scale * (1 .- is_invested_long_bool)
                             constr_is_invested_short_float_decision_lb,
                             is_invested_short_float .>=
                             k .- scale * (1 .- is_invested_short_bool)
                         end)

            @expressions(model, begin
                             is_invested_long, is_invested_long_float
                             is_invested_short, is_invested_short_float
                         end)
        end
        @constraints(model,
                     begin
                         constr_is_invested_bool_ub, is_invested_bool .<= 1
                         constr_w_mip_ub,
                         scale_constr * w .<= scale_constr * is_invested_long .* long_ub
                         constr_w_mip_lb,
                         scale_constr * w .>= scale_constr * is_invested_short .* short_lb
                         constr_long_w_mip_lb,
                         scale_constr * w .>=
                         scale_constr *
                         (is_invested_long .* long_lb - scale * (1 - is_invested_long_bool))
                         constr_short_w_mip_ub,
                         scale_constr * w .<=
                         scale_constr * (is_invested_short .* short_ub +
                                         scale * (1 - is_invested_short_bool))
                     end)
    else
        @variable(model, is_invested_bool[1:N], binary = true)
        if isa(k, Real)
            @expression(model, is_invested, is_invested_bool)
        else
            @variable(model, is_invested_float[1:N] .>= 0)
            scale = port.card_scale
            @constraints(model,
                         begin
                             constr_is_invested_float_ub, is_invested_float .<= k
                             constr_is_invested_float_decision_ub,
                             is_invested_float .<= scale * is_invested_bool
                             constr_is_invested_float_decision_lb,
                             is_invested_float .>= k .- scale * (1 .- is_invested_bool)
                         end)
            @expression(model, is_invested, is_invested_float)
        end
        @constraint(model, constr_w_mip_ub,
                    scale_constr * w .<= scale_constr * is_invested .* long_ub)
        if (isa(long_lb, Real) && !iszero(long_lb) ||
            isa(long_lb, AbstractVector) && (!isempty(long_lb) || any(.!iszero(long_lb))))
            @constraint(model, constr_long_w_mip_lb,
                        scale_constr * w .>= scale_constr * is_invested .* long_lb)
        end
        if short && allow_shorting
            @constraint(model, constr_w_mip_lb,
                        scale_constr * w .>= scale_constr * is_invested .* short_lb)
        end
    end

    #=
    ## Portfolio cardinality
    =#
    if card_flag
        @constraint(model, constr_card, sum(is_invested_bool) <= card)
    end

    #=
    ## Group cardinality
    =#
    if gcard_ineq_flag
        @constraint(model, constr_card_ineq, a_card_ineq * is_invested_bool .>= b_card_ineq)
    end
    if gcard_eq_flag
        @constraint(model, constr_card_eq, a_card_eq * is_invested_bool .== b_card_eq)
    end

    #=
    ## Network cardinality
    =#
    if ntwk_flag
        ntwk_A = network_adj.A
        ntwk_k = network_adj.k
        @constraint(model, constr_ntwk_card, ntwk_A * is_invested_bool .<= ntwk_k)
    end

    #=
    ## Cluster cardinality
    =#
    if clst_flag
        clst_A = cluster_adj.A
        clst_k = cluster_adj.k
        @constraint(model, constr_clst_card, clst_A * is_invested_bool .<= clst_k)
    end

    return nothing
end
function tracking_error_benchmark(tracking::TrackWeight, returns)
    w = tracking.w
    long_fees = tracking.long_fees
    short_fees = tracking.short_fees
    rebalance = tracking.rebalance
    fees = calc_fees(w, long_fees, short_fees, rebalance)
    return returns * w .- fees
end
function tracking_error_benchmark(tracking::TrackRet, ::Any)
    return tracking.w
end
function tracking_error_constraints(port, returns)
    tracking = port.tracking
    if isa(tracking, NoTracking) || isempty(isempty(tracking.w)) || isinf(tracking.err)
        return nothing
    end

    model = port.model
    scale_constr = model[:scale_constr]
    k = model[:k]
    get_net_portfolio_returns(model, returns)
    net_X = model[:net_X]
    T = size(returns, 1)

    benchmark = tracking_error_benchmark(tracking, returns)
    err = tracking.err

    @variable(model, t_tracking_error)
    @expression(model, tracking_error, net_X .- benchmark * k)
    @constraints(model,
                 begin
                     constr_tracking_soc,
                     [scale_constr * t_tracking_error; scale_constr * tracking_error] ∈
                     SecondOrderCone()
                     constr_tracking,
                     scale_constr * t_tracking_error <=
                     scale_constr * err * k * sqrt(T - 1)
                 end)

    return nothing
end
function turnover_constraints(port)
    turnover = port.turnover
    if isa(turnover, NoTR) ||
       isa(turnover.val, Real) && isinf(turnover.val) ||
       isa(turnover.val, AbstractVector) &&
       (isempty(turnover.val) || all(isinf.(turnover.val))) ||
       isempty(turnover.w)
        return nothing
    end

    model = port.model
    scale_constr = model[:scale_constr]
    w = model[:w]
    k = model[:k]
    N = length(w)

    benchmark = turnover.w
    val = turnover.val

    @variable(model, t_turnover[1:N])
    @expression(model, turnover, w .- benchmark * k)
    @constraints(model,
                 begin
                     constr_turnover_noc[i = 1:N],
                     [scale_constr * t_turnover[i]; scale_constr * turnover[i]] ∈
                     MOI.NormOneCone(2)
                     constr_turnover, scale_constr * t_turnover .<= scale_constr * val * k
                 end)

    return nothing
end
function management_fee(port)
    short = port.short
    long_fees = port.long_fees
    short_fees = port.short_fees
    model = port.model

    if !(isa(long_fees, Real) && iszero(long_fees) ||
         isa(long_fees, AbstractVector) && (isempty(long_fees) || all(iszero.(long_fees))))
        long_w = model[:long_w]
        @expression(model, long_fees, sum(long_fees .* long_w))
    end

    if short && !(isa(short_fees, Real) && iszero(short_fees) ||
                  isa(short_fees, AbstractVector) &&
                  (isempty(short_fees) || all(iszero.(short_fees))))
        short_w = model[:short_w]
        @expression(model, short_fees, sum(short_fees .* short_w))
    end

    return nothing
end
function rebalance_fees(port)
    rebalance = port.rebalance
    if isa(rebalance, NoTR) ||
       isa(rebalance.val, Real) && iszero(rebalance.val) ||
       isa(rebalance.val, AbstractVector) &&
       (isempty(rebalance.val) || all(iszero.(rebalance.val))) ||
       isempty(rebalance.w)
        return nothing
    end

    model = port.model
    scale_constr = model[:scale_constr]
    w = model[:w]
    k = model[:k]
    N = length(w)

    benchmark = rebalance.w
    val = rebalance.val

    @variable(model, t_rebalance[1:N])
    @expressions(model, begin
                     rebalance, w .- benchmark * k
                     rebalance_fees, sum(val .* t_rebalance)
                 end)
    @constraint(model, constr_rebalance[i = 1:N],
                [scale_constr * t_rebalance[i]; scale_constr * rebalance[i]] ∈
                MOI.NormOneCone(2))

    return nothing
end
function get_fees(model)
    if haskey(model, :fees)
        return nothing
    end
    @expression(model, fees, zero(AffExpr))
    if haskey(model, :long_fees)
        add_to_expression!(fees, model[:long_fees])
    end
    if haskey(model, :short_fees)
        add_to_expression!(fees, model[:short_fees])
    end
    if haskey(model, :rebalance_fees)
        add_to_expression!(fees, model[:rebalance_fees])
    end
    return nothing
end
function get_one_plus_returns(model, returns)
    if haskey(model, :ret_p_1)
        return nothing
    end
    @expression(model, ret_p_1, one(eltype(returns)) .+ returns)
    return nothing
end
function get_net_portfolio_returns(model, returns)
    if haskey(model, :net_X)
        return nothing
    end
    get_portfolio_returns(model, returns)
    get_fees(model)
    X = model[:X]
    fees = model[:fees]
    @expression(model, net_X, X .- fees)
    return nothing
end
function SDP_constraints(model, ::Trad)
    if haskey(model, :W)
        return nothing
    end
    scale_constr = model[:scale_constr]
    w = model[:w]
    k = model[:k]
    N = length(w)

    @variable(model, W[1:N, 1:N], Symmetric)
    @expression(model, M, hcat(vcat(W, transpose(w)), vcat(w, k)))
    @constraint(model, constr_M_PSD, scale_constr * M ∈ PSDCone())

    return nothing
end
function SDP_constraints(model, ::Any)
    if haskey(model, :W)
        return nothing
    end
    scale_constr = model[:scale_constr]
    w = model[:w]
    N = length(w)

    @variable(model, W[1:N, 1:N], Symmetric)
    @expression(model, M, hcat(vcat(W, transpose(w)), vcat(w, 1)))
    @constraint(model, constr_M_PSD, scale_constr * M ∈ PSDCone())

    return nothing
end
function SDP_network_cluster_constraints(port, type)
    network_adj = port.network_adj
    cluster_adj = port.cluster_adj
    ntwk_flag = isa(network_adj, SDP)
    clst_flag = isa(cluster_adj, SDP)
    if !(ntwk_flag || clst_flag)
        return nothing
    end

    model = port.model
    scale_constr = model[:scale_constr]
    SDP_constraints(model, type)
    W = model[:W]

    if ntwk_flag
        A = network_adj.A
        @constraint(model, constr_ntwk_sdp, scale_constr * A .* W .== 0)
    end

    if clst_flag
        A = cluster_adj.A
        @constraint(model, constr_clst_sdp, scale_constr * A .* W .== 0)
    end

    return nothing
end
function SDP_network_cluster_penalty(port)
    model = port.model
    ntwk_flag = haskey(model, :constr_ntwk_sdp)
    clst_flag = haskey(model, :constr_clst_sdp)
    if !(ntwk_flag || clst_flag)
        return nothing
    end

    W = model[:W]
    if !(haskey(model, :variance_risk) || haskey(model, :wcvariance_risk))
        if ntwk_flag
            network_adj = port.network_adj
            penalty = network_adj.penalty
            @expression(model, network_penalty, penalty * tr(W))
        end
        if clst_flag
            cluster_adj = port.cluster_adj
            penalty = cluster_adj.penalty
            @expression(model, cluster_penalty, penalty * tr(W))
        end
    end

    return nothing
end
function custom_constraint(port, ::Union{NoCustomConstraint, Nothing})
    return nothing
end
function L1_regularisation(port)
    l1 = port.l1
    if iszero(l1)
        return nothing
    end

    model = port.model
    scale_constr = model[:scale_constr]
    w = model[:w]

    @variable(model, t_l1)
    @constraint(model, constr_l1,
                [scale_constr * t_l1; scale_constr * w] in MOI.NormOneCone(1 + length(w)))
    @expression(model, l1_reg, l1 * t_l1)

    return nothing
end
function L2_regularisation(port)
    l2 = port.l2
    if iszero(l2)
        return nothing
    end

    model = port.model
    scale_constr = model[:scale_constr]
    w = model[:w]

    @variable(model, t_l2)
    @constraint(model, constr_l2,
                [scale_constr * t_l2; scale_constr * w] in SecondOrderCone())
    @expression(model, l2_reg, l2 * t_l2)

    return nothing
end
