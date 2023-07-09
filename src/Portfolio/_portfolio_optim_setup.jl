function _return_setup(portfolio, class, kelly, obj, T, rf, returns, mu)
    model = portfolio.model
    w = model[:w]
    k = model[:k]
    risk = model[:risk]

    if class == :classic && (kelly == :exact || kelly == :approx)
        if kelly == :exact
            @variable(model, texact_kelly[1:T])
            if obj == :sharpe
                @expression(model, ret, sum(texact_kelly) / T - rf * k)
                @expression(model, kret, k .+ returns * w)
                @constraint(
                    model,
                    [i = 1:T],
                    [texact_kelly[i], k, kret[i]] in MOI.ExponentialCone()
                )
                @constraint(model, risk <= 1)
            else
                @expression(model, ret, sum(texact_kelly) / T)
                @expression(model, kret, 1 .+ returns * w)
                @constraint(
                    model,
                    [i = 1:T],
                    [texact_kelly[i], 1, kret[i]] in MOI.ExponentialCone()
                )
            end
        elseif kelly == :approx
            tdev = model[:tdev]
            dev_risk = model[:dev_risk]
            if obj == :sharpe
                @variable(model, tapprox_kelly >= 0)
                @constraint(
                    model,
                    [k + tapprox_kelly; 2 * tdev + k - tapprox_kelly] in SecondOrderCone()
                )
                @expression(model, ret, dot(mu, w) - 0.5 * tapprox_kelly)
                @constraint(model, risk <= 1)
            else
                @expression(model, ret, dot(mu, w) - 0.5 * dev_risk)
            end
        end
    else
        @expression(model, ret, dot(mu, w))
        if obj == :sharpe
            @constraint(model, ret - rf * k == 1)
        end
    end

    # Return constraints.
    mu_l = portfolio.mu_l
    !isfinite(mu_l) && (return nothing)

    if obj == :sharpe
        @constraint(model, ret >= mu_l * k)
    else
        @constraint(model, ret >= mu_l)
    end

    return nothing
end

function _setup_weights(portfolio, obj, N)
    max_number_assets = portfolio.max_number_assets
    short = portfolio.short
    short_u = portfolio.short_u
    long_u = portfolio.long_u
    sum_short_long = portfolio.sum_short_long

    model = portfolio.model
    w = model[:w]
    k = model[:k]

    # Boolean variables max number of assets.
    if max_number_assets > 0
        if obj == :sharpe
            @variable(model, tass_bin[1:N], binary = true)
            @variable(model, tass_bin_sharpe[1:N] >= 0)
        else
            @variable(model, tass_bin[1:N], binary = true)
        end
    end

    # Weight constraints.
    if obj == :sharpe
        @constraint(model, sum(w) == sum_short_long * k)

        # Maximum number of assets constraints.
        if max_number_assets > 0
            @constraint(model, sum(tass_bin) <= max_number_assets)
            @constraint(model, tass_bin_sharpe .<= k)
            @constraint(model, tass_bin_sharpe .<= 100000 * tass_bin)
            @constraint(model, tass_bin_sharpe .>= k - 100000 * (1 .- tass_bin))
            @constraint(model, w .<= long_u * tass_bin_sharpe)
        end

        if short == false
            @constraint(model, w .<= long_u * k)
            @constraint(model, w .>= 0)
        else
            @variable(model, tw_ulong[1:N] .>= 0)
            @variable(model, tw_ushort[1:N] .>= 0)

            @constraint(model, sum(tw_ulong) <= long_u * k)
            @constraint(model, sum(tw_ushort) <= short_u * k)

            @constraint(model, w .- tw_ulong .<= 0)
            @constraint(model, w .+ tw_ushort .>= 0)

            # Maximum number of assets constraints.
            if max_number_assets > 0
                @constraint(model, w .>= -short_u * tass_bin_sharpe)
            end
        end
    else
        @constraint(model, sum(w) == sum_short_long)

        # Maximum number of assets constraints.
        if max_number_assets > 0
            @constraint(model, sum(tass_bin) <= max_number_assets)
            @constraint(model, w .<= long_u * tass_bin)
        end

        if short == false
            @constraint(model, w .<= long_u)
            @constraint(model, w .>= 0)
        else
            @variable(model, tw_ulong[1:N] .>= 0)
            @variable(model, tw_ushort[1:N] .>= 0)

            @constraint(model, sum(tw_ulong) <= long_u)
            @constraint(model, sum(tw_ushort) <= short_u)

            @constraint(model, w .- tw_ulong .<= 0)
            @constraint(model, w .+ tw_ushort .>= 0)

            # Maximum number of assets constraints.
            if max_number_assets > 0
                @constraint(model, w .>= -short_u * tass_bin)
            end
        end
    end

    return nothing
end

function _setup_linear_constraints(portfolio, obj)
    A = portfolio.a_mtx_ineq
    B = portfolio.b_vec_ineq

    (isempty(A) || isempty(B)) && (return nothing)

    model = portfolio.model
    w = model[:w]
    k = model[:k]
    # Linear weight constraints.
    if obj == :sharpe
        @constraint(model, A * w .- B * k .>= 0)
    else
        @constraint(model, A * w .- B .>= 0)
    end

    return nothing
end

function _setup_min_number_effective_assets(portfolio, obj)
    mnea = portfolio.min_number_effective_assets

    (mnea < 1) && (return nothing)

    model = portfolio.model
    w = model[:w]
    k = model[:k]

    @variable(model, tmnea >= 0)
    @constraint(model, [tmnea; w] in SecondOrderCone())
    if obj == :sharpe
        @constraint(model, tmnea * sqrt(mnea) <= k)
    else
        @constraint(model, tmnea * sqrt(mnea) <= 1)
    end

    return nothing
end

function _setup_tracking_err(portfolio, returns, obj, T)
    tracking_err = portfolio.tracking_err

    !isfinite(tracking_err) && (return nothing)

    kind_tracking_err = portfolio.kind_tracking_err
    tracking_err_weights = portfolio.tracking_err_weights
    tracking_err_returns = portfolio.tracking_err_returns

    tracking_err_flag = false

    if kind_tracking_err == :weights && !isempty(tracking_err_weights)
        benchmark = returns * tracking_err_weights[!, :weights]
        tracking_err_flag = true
    elseif kind_tracking_err == :returns && !isempty(tracking_err_returns)
        benchmark = tracking_err_returns[!, :returns]
        tracking_err_flag = true
    end

    !(tracking_err_flag) && (return nothing)

    model = portfolio.model
    w = model[:w]
    k = model[:k]

    @variable(model, t_track_err >= 0)
    if obj == :sharpe
        @expression(model, track_err, returns * w .- benchmark * k)
        @constraint(model, [t_track_err; track_err] in SecondOrderCone())
        @constraint(model, t_track_err <= tracking_err * k * sqrt(T - 1))
    else
        @expression(model, track_err, returns * w .- benchmark)
        @constraint(model, [t_track_err; track_err] in SecondOrderCone())
        @constraint(model, t_track_err <= tracking_err * sqrt(T - 1))
    end

    return nothing
end

function _setup_turnover(portfolio, N, obj)
    turnover = portfolio.turnover
    turnover_weights = portfolio.turnover_weights

    (isinf(turnover) || isempty(turnover_weights)) && (return nothing)

    model = portfolio.model
    w = model[:w]
    k = model[:k]
    @variable(model, t_turnov[1:N] >= 0)
    if obj == :sharpe
        @expression(model, turnov, w .- turnover_weights[!, :weights] * k)
        @constraint(model, [i = 1:N], [t_turnov[i]; turnov[i]] in MOI.NormOneCone(2))
        @constraint(model, t_turnov .<= turnover * k)
    else
        @expression(model, turnov, w .- turnover_weights[!, :weights])
        @constraint(model, [i = 1:N], [t_turnov[i]; turnov[i]] in MOI.NormOneCone(2))
        @constraint(model, t_turnov .<= turnover)
    end

    return nothing
end

function _setup_objective_function(portfolio, obj, kelly, l)
    model = portfolio.model
    ret = model[:ret]
    risk = model[:risk]

    if obj == :sharpe
        if model == :classic && (kelly == :exact || kelly == :approx)
            @objective(model, Max, ret)
        else
            @objective(model, Min, risk)
        end
    elseif obj == :min_risk
        @objective(model, Min, risk)
    elseif obj == :utility
        @objective(model, Max, ret - l * risk)
    elseif obj == :max_ret
        @objective(model, Max, ret)
    end

    return nothing
end

function _optimize_portfolio(portfolio, N)
    solvers = portfolio.solvers
    sol_params = portfolio.sol_params
    model = portfolio.model
    w = model[:w]

    term_status = termination_status(model)
    solvers_tried = Dict()

    for (solver_name, solver) in solvers
        set_optimizer(model, solver)
        if haskey(sol_params, solver_name)
            for (attribute, value) in sol_params[solver_name]
                set_attribute(model, attribute, value)
            end
        end
        try
            optimize!(model)
        catch jump_error
            push!(solvers_tried, solver_name => Dict("error" => jump_error))
            continue
        end
        term_status = termination_status(model)

        if term_status in ValidTermination &&
           all(isfinite.(value.(w))) &&
           all(abs.(0.0 .- value.(w)) .> N^2 * eps())
            break
        end
        push!(
            solvers_tried,
            solver_name => Dict(
                "objective_val" => objective_value(model),
                "term_status" => term_status,
                "sol_params" =>
                    haskey(sol_params, solver_name) ? sol_params[solver_name] : missing,
            ),
        )
    end

    return term_status, solvers_tried
end

function _cleanup_weights(portfolio, returns, N, obj, solvers_tried)
    model = portfolio.model
    w = model[:w]
    k = model[:k]

    weights = Vector{eltype(returns)}(undef, N)
    if obj == :sharpe
        weights .= value.(w) / value(k)
    else
        weights .= value.(w)
    end

    short = portfolio.short
    sum_short_long = portfolio.sum_short_long
    if short == false
        weights .= abs.(weights) / sum(abs.(weights)) * sum_short_long
    end

    portfolio.p_optimal =
        DataFrame(tickers = names(portfolio.returns)[2:end], weights = weights)

    if isempty(solvers_tried)
        portfolio.fail = Dict()
    else
        portfolio.fail = solvers_tried
    end

    return nothing
end