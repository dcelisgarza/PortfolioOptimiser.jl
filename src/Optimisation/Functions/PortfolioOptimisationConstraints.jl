function MIP_constraints(port)
    #=
    # MIP constraints

    ## Flags

    Flags for deciding whether the problem is MIP.
    =#
    card_flag = size(port.returns, 2) > port.card > 0
    gcard_ineq_flag = !(isempty(port.a_card_ineq) || isempty(port.b_card_ineq))
    gcard_eq_flag = !(isempty(port.a_card_eq) || isempty(port.b_card_eq))
    ntwk_flag = isa(port.network, IP)
    clst_flag = isa(port.cluster, IP)

    if !(card_flag || gcard_ineq_flag || gcard_eq_flag || ntwk_flag || clst_flag)
        return nothing
    end

    model = port.model
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
    long_t = port.long_t
    long_u = port.long_u
    short_t = port.short_t
    short_u = port.short_u
    if !iszero(short_t) && short
        scale = port.card_scale
        @variables(model, begin
                       is_invested_long_bool[1:N], binary = true
                       is_invested_short_bool[1:N], binary = true
                   end)
        @expression(model, is_invested_bool, is_invested_long_bool + is_invested_short_bool)
        if is_fixed(k)
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
                         w .<= is_invested_long .* long_u
                         w .>= is_invested_short .* short_u
                         w .>=
                         is_invested_long .* long_t - scale * (1 - is_invested_long_bool)
                         w .<=
                         is_invested_short .* short_t -
                         scale * (1 - is_invested_short_bool)
                     end)
    else
        @variable(model, is_invested_bool[1:N], binary = true)
        if is_fixed(k)
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
        @constraint(model, w .<= is_invested .* long_u)
        if !iszero(long_t)
            @constraint(model, w .>= is_invested .* long_t)
        end
        if short
            @constraint(model, w .>= is_invested .* short_u)
        end
    end

    #=
    ## Portfolio cardinality
    =#
    if card_flag
        card = port.card
        @constraint(model, sum(is_invested_bool) <= card)
    end

    #=
    ## Group cardinality
    =#
    if gcard_ineq_flag
        A = port.a_card_ineq
        B = port.b_card_ineq
        @constraint(model, A * is_invested_bool .>= B)
    end
    if gcard_eq_flag
        A = port.a_card_eq
        B = port.b_card_eq
        @constraint(model, A * is_invested_bool .== B)
    end

    #=
    ## Network cardinality
    =#
    if ntwk_flag
        A = port.network.A
        k = port.network.k
        @constraint(model, A * is_invested_bool .<= k)
    end

    #=
    ## Cluster cardinality
    =#
    if clst_flag
        A = port.cluster.A
        k = port.cluster.k
        @constraint(model, A * is_invested_bool .<= k)
    end

    return nothing
end
function _long_w_budget(budget_flag, max_budget_flag, min_budget_flag, min_budget, budget,
                        max_budget, short_budget, model, k, long_w)
    if !budget_flag
        @constraint(model, sum(long_w) == (budget - short_budget) * k)
    end
    if !max_budget_flag && budget_flag
        @constraint(model, sum(long_w) <= (max_budget - short_budget) * k)
    end
    if !min_budget_flag && budget_flag
        @constraint(model, sum(long_w) >= (min_budget - short_budget) * k)
    end
    if !budget_flag && !max_budget_flag && !min_budget_flag
        @constraint(model, sum(long_w) == (1 - short_budget) * k)
    end

    return nothing
end
function weight_constraints(port)
    #=
    # Weight constraints
    =#
    model = port.model
    w = model[:w]
    k = model[:k]

    #=
    ## Portfolio budget constraints
    =#
    min_budget = port.min_budget
    budget = port.budget
    max_budget = port.max_budget
    budget_flag = isfinite(budget)
    max_budget_flag = isfinite(max_budget)
    min_budget_flag = isfinite(min_budget)
    if budget_flag
        @constraint(model, sum(w) == budget * k)
    end
    if max_budget_flag && !budget_flag
        @constraint(model, sum(w) <= max_budget * k)
    end
    if min_budget_flag && !budget_flag
        @constraint(model, sum(w) >= min_budget * k)
    end
    if !budget_flag && !max_budget_flag && !min_budget_flag
        @constraint(model, sum(w) == 1 * k)
    end

    #=
    ## Min and max weights, short budget weights.
    =#
    short = port.short
    if !short
        long_u = port.long_u
        @constraints(model, begin
                         w .<= long_u * k
                         w .>= 0
                     end)
        @expression(model, long_w, w)
    else
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
                         long_w .<= long_u * k
                         short_w .>= short_u * k
                         w .<= long_w
                         w .>= short_w
                     end)

        #=
        ## Long-short portfolio budget constraints
        =#
        short_budget_flag = isfinite(short_budget)
        max_short_budget_flag = isfinite(max_short_budget)
        min_short_budget_flag = isfinite(min_short_budget)
        if short_budget_flag
            @constraint(model, sum(short_w) == short_budget * k)
            _long_w_budget(budget_flag, max_budget_flag, min_budget_flag, min_budget,
                           budget, max_budget, short_budget, model, k, long_w)
        end
        if max_short_budget_flag && !short_budget_flag
            @constraint(model, sum(short_w) >= max_short_budget * k)
            _long_w_budget(budget_flag, max_budget_flag, min_budget_flag, min_budget,
                           budget, max_budget, max_short_budget, model, k, long_w)
        end
        if min_short_budget_flag && !short_budget_flag
            @constraint(model, sum(short_w) <= min_short_budget * k)
            _long_w_budget(budget_flag, max_budget_flag, min_budget_flag, min_budget,
                           budget, max_budget, min_short_budget, model, k, long_w)
        end
        if !short_budget_flag && !max_short_budget_flag && !min_short_budget_flag
            @constraint(model, sum(short_w) == -0.2 * k)
            _long_w_budget(budget_flag, max_budget_flag, min_budget_flag, min_budget,
                           budget, max_budget, -0.2, model, k, long_w)
        end
    end

    #=
    ## Number of effective assets
    =#
    nea = port.nea
    if nea > zero(nea)
        @variable(model, nea_var >= 0)
        @constraints(model, begin
                         [nea_var; w] ∈ SecondOrderCone()
                         nea_var * sqrt(nea) <= k
                     end)
    end

    #=
    ## Linear constraints
    =#
    A = port.a_ineq
    B = port.b_ineq
    if !(isempty(A) || isempty(B))
        @constraint(model, A * w .>= B * k)
    end
    A = port.a_eq
    B = port.b_eq
    if !(isempty(A) || isempty(B))
        @constraint(model, A * w .== B * k)
    end

    #=
    ### Centrality constraints
    =#
    A = port.a_cent_ineq
    B = port.b_cent_ineq
    if !(isempty(A) || isempty(B))
        @constraint(model, dot(A, w) .>= B * k)
    end
    A = port.a_cent_eq
    B = port.b_cent_eq
    if !(isempty(A) || isempty(B))
        @constraint(model, dot(A, w) .== B * k)
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
    w = model[:w]
    k = model[:k]

    T = size(returns, 1)
    benchmark = tracking_error_benchmark(tracking, returns)
    err = tracking.err

    @variable(model, t_tracking_error >= 0)
    @expression(model, tracking_error, returns * w .- benchmark * k)
    @constraints(model, begin
                     [t_tracking_error; tracking_error] ∈ SecondOrderCone()
                     t_tracking_error <= err * k * sqrt(T - 1)
                 end)

    return nothing
end
function turnover_constraints(port)
    turnover = port.turnover
    if isa(turnover, NoTR) ||
       isa(turnover.val, Real) && isinf(turnover.val) ||
       isa(turnover.val, AbstractVector) && isempty(turnover.val) ||
       isempty(turnover.w)
        return nothing
    end

    model = port.model
    w = model[:w]
    k = model[:k]

    N = length(w)
    benchmark = turnover.w
    val = turnover.val

    @variable(model, t_turnover[1:N] >= 0)
    @expression(model, turnover, w .- benchmark * k)
    @constraints(model, begin
                     [i = 1:N], [t_turnover[i]; turnover[i]] ∈ MOI.NormOneCone(2)
                     t_turnover .<= val * k
                 end)

    return nothing
end
function L1_regularisation(port)
    l1 = port.l1
    if iszero(l1)
        return nothing
    end

    model = port.model
    w = model[:w]

    @variable(model, t_l1 >= 0)
    @constraint(model, [t_l1; w] in MOI.NormOneCone(1 + length(w)))
    @expression(model, l1_reg, l1 * t_l1)

    return nothing
end
function L2_regularisation(port)
    l2 = port.l2
    if iszero(l2)
        return nothing
    end

    model = port.model
    w = model[:w]

    @variable(model, t_l2 >= 0)
    @constraint(model, [t_l2; w] in SecondOrderCone())
    @expression(model, l2_reg, l2 * t_l2)

    return nothing
end
function management_fee(port)
    short = port.short
    long_fees = port.long_fees
    short_fees = port.short_fees
    model = port.model

    if !(isa(long_fees, Real) && iszero(long_fees) ||
         isa(long_fees, AbstractVector) && isempty(long_fees) ||
         isa(long_fees, AbstractVector) && all(iszero.(long_fees)))
        long_w = model[:long_w]
        @expression(model, long_fee, sum(long_fees .* long_w))
    end

    if short && !(isa(short_fees, Real) && iszero(short_fees) ||
                  isa(short_fees, AbstractVector) && isempty(short_fees) ||
                  isa(short_fees, AbstractVector) && all(iszero.(short_fees)))
        short_w = model[:short_w]
        @expression(model, short_fee, sum(short_fees .* short_w))
    end

    return nothing
end
function rebalance_cost(port)
    rebalance = port.rebalance
    model = port.model
    if isa(rebalance, NoTR) ||
       isa(rebalance.val, Real) && iszero(rebalance.val) ||
       isa(rebalance.val, AbstractVector) && isempty(rebalance.val) ||
       isempty(rebalance.w)
        return nothing
    end

    benchmark = rebalance.w
    val = rebalance.val

    w = model[:w]
    k = model[:k]
    N = length(w)

    @variable(model, t_rebalance[1:N] >= 0)
    @expression(model, rebalance, w .- benchmark * k)
    @constraint(model, [i = 1:N], [t_rebalance[i]; rebalance[i]] ∈ MOI.NormOneCone(2))
    @expression(model, rebalance_cost, sum(val .* t_rebalance))

    return nothing
end
function _SDP_constraints(model)
    w = model[:w]
    k = model[:k]
    N = length(w)

    @variable(model, W[1:N, 1:N], Symmetric)
    @expression(model, M, hcat(vcat(W, transpose(w)), vcat(w, k)))
    @constraint(model, M ∈ PSDCone())

    return nothing
end
function SDP_network_cluster_constraints(port, ntwk_flag::Bool = true)
    network_cluster = ntwk_flag ? port.network : port.cluster
    model = port.model
    if !isa(network_cluster, SDP)
        return nothing
    end

    if !haskey(model, :W)
        _SDP_constraints(model)
    end

    W = model[:W]
    A = network_cluster.A
    @constraint(model, A .* W .== 0)

    if !haskey(model, :sd_risk)
        penalty = network_cluster.penalty
        if ntwk_flag
            @expression(model, network_penalty, penalty * tr(W))
        else
            @expression(model, cluster_penalty, penalty * tr(W))
        end
    end

    return nothing
end
function custom_constraint(port, ::Nothing)
    return nothing
end
function custom_objective(port, ::Nothing)
    return nothing
end

###########
###########

##########
##########

function custom_constraint_objective_penatly(::Nothing, port)
    return nothing
end
##########
##########
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
    if !(isempty(port.a_cmtx_ineq) || isempty(port.b_cvec_ineq))
        A = port.a_cmtx_ineq
        B = port.b_cvec_ineq
        model = port.model
        w = model[:w]
        if !haskey(model, :tnau_bin)
            N = length(w)
            @variable(model, tnau_bin[1:N], binary = true)
            # Sharpe ratio
            @variable(model, tnau_bin_sharpe[1:N] .>= 0)
            k = model[:k]
            @constraint(model, tnau_bin_sharpe .<= k)
            @constraint(model, tnau_bin_sharpe .<= port.num_assets_u_scale * tnau_bin)
            @constraint(model,
                        tnau_bin_sharpe .>= k .- port.num_assets_u_scale * (1 .- tnau_bin))
            # Long and short
            if !port.short
                @constraint(model, w .<= port.long_u * tnau_bin_sharpe)
            else
                @constraint(model, w .<= port.long_u * tnau_bin_sharpe)
                @constraint(model, w .>= -port.short_u * tnau_bin_sharpe)
            end
        end
        tnau_bin = model[:tnau_bin]
        @constraint(model, A * tnau_bin .- B .>= 0)
    end
    if port.num_assets_l > 0
        model = port.model
        @variable(model, tnal >= 0)
        w = model[:w]
        @constraint(model, [tnal; w] ∈ SecondOrderCone())
        k = model[:k]
        @constraint(model, tnal * sqrt(port.num_assets_l) <= k)
    end
    if !(isempty(port.a_smtx_ineq) || isempty(port.b_svec_ineq))
        A = port.a_smtx_ineq
        B = port.b_svec_ineq
        model = port.model
        w = model[:w]
        k = model[:k]
        N = length(B)
        @variable(model, tgnal[1:N] >= 0)
        @constraint(model, [i = 1:N], [tgnal[i]; w[A[i, :]]] ∈ SecondOrderCone())
        @constraint(model, tgnal .* sqrt.(B) .<= k)
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
    if !(isempty(port.a_cmtx_ineq) || isempty(port.b_cvec_ineq))
        A = port.a_cmtx_ineq
        B = port.b_cvec_ineq
        model = port.model
        w = model[:w]
        if !haskey(model, :tnau_bin)
            N = length(w)
            @variable(model, tnau_bin[1:N], binary = true)
            # Long and short
            if !port.short
                @constraint(model, w .<= port.long_u * tnau_bin)
            else
                @constraint(model, w .<= port.long_u * tnau_bin)
                @constraint(model, w .>= -port.short_u * tnau_bin)
            end
        end
        tnau_bin = model[:tnau_bin]
        @constraint(model, A * tnau_bin .- B .>= 0)
    end
    if port.num_assets_l > 0
        model = port.model
        @variable(model, tnal >= 0)
        w = model[:w]
        @constraint(model, [tnal; w] ∈ SecondOrderCone())
        @constraint(model, tnal * sqrt(port.num_assets_l) <= 1)
    end
    if !(isempty(port.a_smtx_ineq) || isempty(port.b_svec_ineq))
        A = port.a_smtx_ineq
        B = port.b_svec_ineq
        model = port.model
        w = model[:w]
        N = length(B)
        @variable(model, tgnal[1:N] >= 0)
        @constraint(model, [i = 1:N], [tgnal[i]; w[A[i, :]]] ∈ SecondOrderCone())
        @constraint(model, tgnal .* sqrt.(B) .<= 1)
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

        @constraint(model, sum(tw_ulong) <= (port.budget + port.short_budget) * k)
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

        @constraint(model, sum(tw_ulong) <= port.budget + port.short_budget)
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
