function get_portfolio_returns(model, returns)
    if haskey(model, :X)
        return nothing
    end
    w = model[:w]
    @expression(model, X, returns * w)

    return nothing
end
function _long_w_budget(budget_flag, min_budget_flag, max_budget_flag, min_budget, budget,
                        max_budget, short_budget, model, k, long_w, constr_scale)
    if budget_flag
        @constraint(model,
                    constr_scale * sum(long_w) ==
                    constr_scale * (budget - short_budget) * k)
    else
        if min_budget_flag
            @constraint(model,
                        constr_scale * sum(long_w) >=
                        constr_scale * (min_budget - short_budget) * k)
        end
        if max_budget_flag
            @constraint(model,
                        constr_scale * sum(long_w) <=
                        constr_scale * (max_budget - short_budget) * k)
        end
    end

    return nothing
end
function weight_constraints(port, allow_shorting::Bool = true)
    #=
    # Weight constraints
    =#
    model = port.model
    constr_scale = model[:constr_scale]
    w = model[:w]
    k = model[:k]
    N = length(w)

    #=
    ## Portfolio budget constraints
    =#
    min_budget = port.min_budget
    budget = port.budget
    max_budget = port.max_budget
    min_budget_flag = isfinite(min_budget)
    budget_flag = isfinite(budget)
    max_budget_flag = isfinite(max_budget)
    if budget_flag
        @constraint(model, constr_scale * sum(w) == constr_scale * budget * k)
    else
        if min_budget_flag
            @constraint(model, constr_scale * sum(w) >= constr_scale * min_budget * k)
        end
        if max_budget_flag
            @constraint(model, constr_scale * sum(w) <= constr_scale * max_budget * k)
        end
    end

    #=
    ## Min and max weights, short budget weights.
    =#
    short = port.short
    long_u = port.long_u
    if !short
        @constraints(model, begin
                         constr_scale * w .<= constr_scale * long_u * k
                         w .>= 0
                     end)
        @expression(model, long_w, w)
    elseif short && allow_shorting
        #=
        ## Short min and max weights
        =#
        short_u = port.short_u
        min_short_budget = port.min_short_budget
        short_budget = port.short_budget
        max_short_budget = port.max_short_budget

        @variables(model, begin
                       long_w[1:N] .>= 0
                       short_w[1:N] .<= 0
                   end)

        @constraints(model, begin
                         constr_scale * w .<= constr_scale * long_u * k
                         constr_scale * w .>= constr_scale * short_u * k
                         constr_scale * w .<= constr_scale * long_w
                         constr_scale * w .>= constr_scale * short_w
                     end)

        #=
        ## Long-short portfolio budget constraints
        =#
        min_short_budget_flag = isfinite(min_short_budget)
        short_budget_flag = isfinite(short_budget)
        max_short_budget_flag = isfinite(max_short_budget)
        if short_budget_flag
            _long_w_budget(budget_flag, min_budget_flag, max_budget_flag, min_budget,
                           budget, max_budget, short_budget, model, k, long_w, constr_scale)
            @constraint(model,
                        constr_scale * sum(short_w) == constr_scale * short_budget * k)
        else
            if min_short_budget_flag
                _long_w_budget(budget_flag, min_budget_flag, max_budget_flag, min_budget,
                               budget, max_budget, min_short_budget, model, k, long_w,
                               constr_scale)
                @constraint(model,
                            constr_scale * sum(short_w) <=
                            constr_scale * min_short_budget * k)
            end
            if max_short_budget_flag
                _long_w_budget(budget_flag, min_budget_flag, max_budget_flag, min_budget,
                               budget, max_budget, max_short_budget, model, k, long_w,
                               constr_scale)
                @constraint(model,
                            constr_scale * sum(short_w) >=
                            constr_scale * max_short_budget * k)
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
                         [constr_scale * nea_var; constr_scale * w] ∈ SecondOrderCone()
                         constr_scale * nea_var * sqrt(nea) <= constr_scale * k
                     end)
    end

    #=
    ## Linear constraints
    =#
    A = port.a_ineq
    B = port.b_ineq
    if !(isempty(A) || isempty(B))
        @constraint(model, constr_scale * A * w .>= constr_scale * B * k)
    end
    A = port.a_eq
    B = port.b_eq
    if !(isempty(A) || isempty(B))
        @constraint(model, constr_scale * A * w .== constr_scale * B * k)
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
    constr_scale = model[:constr_scale]
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
    long_l = port.long_l
    long_u = port.long_u
    short_l = port.short_l
    short_u = port.short_u
    if (isa(short_l, Real) && !iszero(short_l) ||
        isa(short_l, AbstractVector) && (!isempty(short_l) || any(.!iszero(short_l)))) &&
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
                             is_invested_long_float .<= k
                             is_invested_short_float .<= k
                             is_invested_long_float .<= scale * is_invested_long_bool
                             is_invested_short_float .<= scale * is_invested_short_bool
                             is_invested_long_float .>=
                             k .- scale * (1 .- is_invested_long_bool)
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
                         is_invested .<= 1
                         constr_scale * w .<= constr_scale * is_invested_long .* long_u
                         constr_scale * w .>= constr_scale * is_invested_short .* short_u
                         constr_scale * w .>=
                         constr_scale *
                         (is_invested_long .* long_l - scale * (1 - is_invested_long_bool))
                         constr_scale * w .<=
                         constr_scale * (is_invested_short .* short_l +
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
                             is_invested_float .<= k
                             is_invested_float .<= scale * is_invested_bool
                             is_invested_float .>= k .- scale * (1 .- is_invested_bool)
                         end)
            @expression(model, is_invested, is_invested_float)
        end
        @constraint(model, constr_scale * w .<= constr_scale * is_invested .* long_u)
        if (isa(long_l, Real) && !iszero(long_l) ||
            isa(long_l, AbstractVector) && (!isempty(long_l) || any(.!iszero(long_l))))
            @constraint(model, constr_scale * w .>= constr_scale * is_invested .* long_l)
        end
        if short && allow_shorting
            @constraint(model, constr_scale * w .>= constr_scale * is_invested .* short_u)
        end
    end

    #=
    ## Portfolio cardinality
    =#
    if card_flag
        @constraint(model, sum(is_invested_bool) <= card)
    end

    #=
    ## Group cardinality
    =#
    if gcard_ineq_flag
        @constraint(model, a_card_ineq * is_invested_bool .>= b_card_ineq)
    end
    if gcard_eq_flag
        @constraint(model, a_card_eq * is_invested_bool .== b_card_eq)
    end

    #=
    ## Network cardinality
    =#
    if ntwk_flag
        ntwk_A = network_adj.A
        ntwk_k = network_adj.k
        @constraint(model, ntwk_A * is_invested_bool .<= ntwk_k)
    end

    #=
    ## Cluster cardinality
    =#
    if clst_flag
        clst_A = cluster_adj.A
        clst_k = cluster_adj.k
        @constraint(model, clst_A * is_invested_bool .<= clst_k)
    end

    return nothing
end
function tracking_error_benchmark(tracking::TrackWeight, returns)
    return returns * tracking.w
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
    constr_scale = model[:constr_scale]
    k = model[:k]
    get_portfolio_returns(model, returns)
    X = model[:X]
    T = size(returns, 1)

    benchmark = tracking_error_benchmark(tracking, returns)
    err = tracking.err

    @variable(model, t_tracking_error)
    @expression(model, tracking_error, X .- benchmark * k)
    @constraints(model,
                 begin
                     [constr_scale * t_tracking_error; constr_scale * tracking_error] ∈
                     SecondOrderCone()
                     constr_scale * t_tracking_error <=
                     constr_scale * err * k * sqrt(T - 1)
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
    constr_scale = model[:constr_scale]
    w = model[:w]
    k = model[:k]
    N = length(w)

    benchmark = turnover.w
    val = turnover.val

    @variable(model, t_turnover[1:N])
    @expression(model, turnover, w .- benchmark * k)
    @constraints(model,
                 begin
                     [i = 1:N],
                     [constr_scale * t_turnover[i]; constr_scale * turnover[i]] ∈
                     MOI.NormOneCone(2)
                     constr_scale * t_turnover .<= constr_scale * val * k
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
        @expression(model, long_fee, sum(long_fees .* long_w))
    end

    if short && !(isa(short_fees, Real) && iszero(short_fees) ||
                  isa(short_fees, AbstractVector) &&
                  (isempty(short_fees) || all(iszero.(short_fees))))
        short_w = model[:short_w]
        @expression(model, short_fee, sum(short_fees .* short_w))
    end

    return nothing
end
function rebalance_fee(port)
    rebalance = port.rebalance
    if isa(rebalance, NoTR) ||
       isa(rebalance.val, Real) && iszero(rebalance.val) ||
       isa(rebalance.val, AbstractVector) &&
       (isempty(rebalance.val) || all(iszero.(rebalance.val))) ||
       isempty(rebalance.w)
        return nothing
    end

    model = port.model
    constr_scale = model[:constr_scale]
    w = model[:w]
    k = model[:k]
    N = length(w)

    benchmark = rebalance.w
    val = rebalance.val

    @variable(model, t_rebalance[1:N])
    @expressions(model, begin
                     rebalance, w .- benchmark * k
                     rebalance_fee, sum(val .* t_rebalance)
                 end)
    @constraint(model, [i = 1:N],
                [constr_scale * t_rebalance[i]; constr_scale * rebalance[i]] ∈
                MOI.NormOneCone(2))

    return nothing
end
function get_fees(model)
    if haskey(model, :fees)
        return nothing
    end
    @expression(model, fees, zero(AffExpr))
    if haskey(model, :long_fee)
        add_to_expression!(fees, model[:long_fee])
    end
    if haskey(model, :short_fee)
        add_to_expression!(fees, model[:short_fee])
    end
    if haskey(model, :rebalance_fee)
        add_to_expression!(fees, model[:rebalance_fee])
    end
    return nothing
end
function get_one_plus_returns(model, returns)
    if haskey(model, :RP1)
        return nothing
    end
    @expression(model, RP1, one(eltype(returns)) .+ returns)
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
function _SDP_constraints(model, ::Trad)
    if haskey(model, :W)
        return nothing
    end
    constr_scale = model[:constr_scale]
    w = model[:w]
    k = model[:k]
    N = length(w)

    @variable(model, W[1:N, 1:N], Symmetric)
    @expression(model, M, hcat(vcat(W, transpose(w)), vcat(w, k)))
    @constraint(model, constr_scale * M ∈ PSDCone())

    return nothing
end
function _SDP_constraints(model, ::Any)
    if haskey(model, :W)
        return nothing
    end
    constr_scale = model[:constr_scale]
    w = model[:w]
    N = length(w)

    @variable(model, W[1:N, 1:N], Symmetric)
    @expression(model, M, hcat(vcat(W, transpose(w)), vcat(w, 1)))
    @constraint(model, constr_scale * M ∈ PSDCone())

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
    constr_scale = model[:constr_scale]
    _SDP_constraints(model, type)
    W = model[:W]

    if ntwk_flag
        A = network_adj.A
        @constraint(model, c_ntwk_sdp, constr_scale * A .* W .== 0)
    end

    if clst_flag
        A = cluster_adj.A
        @constraint(model, c_clst_sdp, constr_scale * A .* W .== 0)
    end

    return nothing
end
function SDP_network_cluster_penalty(port)
    model = port.model
    ntwk_flag = haskey(model, :c_ntwk_sdp)
    clst_flag = haskey(model, :c_clst_sdp)
    if !(ntwk_flag || clst_flag)
        return nothing
    end

    W = model[:W]
    if !(haskey(model, :variance_risk) || haskey(model, :wc_variance_risk))
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
    constr_scale = model[:constr_scale]
    w = model[:w]

    @variable(model, t_l1)
    @constraint(model,
                [constr_scale * t_l1; constr_scale * w] in MOI.NormOneCone(1 + length(w)))
    @expression(model, l1_reg, l1 * t_l1)

    return nothing
end
function L2_regularisation(port)
    l2 = port.l2
    if iszero(l2)
        return nothing
    end

    model = port.model
    constr_scale = model[:constr_scale]
    w = model[:w]

    @variable(model, t_l2)
    @constraint(model, [constr_scale * t_l2; constr_scale * w] in SecondOrderCone())
    @expression(model, l2_reg, l2 * t_l2)

    return nothing
end
