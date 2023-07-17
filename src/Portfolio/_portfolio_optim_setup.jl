function _return_setup(portfolio, class, kelly, obj, T, rf, returns, mu)
    model = portfolio.model

    if class == :classic && (kelly == :exact || kelly == :approx)
        if kelly == :exact
            @variable(model, texact_kelly[1:T])
            if obj == :sharpe
                @expression(model, ret, sum(texact_kelly) / T - rf * model[:k])
                @expression(model, kret, model[:k] .+ returns * model[:w])
                @constraint(
                    model,
                    [i = 1:T],
                    [texact_kelly[i], model[:k], kret[i]] in MOI.ExponentialCone()
                )
                @constraint(model, model[:risk] <= 1)
            else
                @expression(model, ret, sum(texact_kelly) / T)
                @expression(model, kret, 1 .+ returns * model[:w])
                @constraint(
                    model,
                    [i = 1:T],
                    [texact_kelly[i], 1, kret[i]] in MOI.ExponentialCone()
                )
            end
        elseif kelly == :approx
            if obj == :sharpe
                @variable(model, tapprox_kelly >= 0)
                @constraint(
                    model,
                    [
                        model[:k] + tapprox_kelly
                        2 * model[:dev] + model[:k] - tapprox_kelly
                    ] in SecondOrderCone()
                )
                @expression(model, ret, dot(mu, model[:w]) - 0.5 * tapprox_kelly)
                @constraint(model, model[:risk] <= 1)
            else
                @expression(model, ret, dot(mu, model[:w]) - 0.5 * model[:dev_risk])
            end
        end
    else
        @expression(model, ret, dot(mu, model[:w]))
        if obj == :sharpe
            @constraint(model, ret - rf * model[:k] == 1)
        end
    end

    # Return constraints.
    mu_l = portfolio.mu_l
    !isfinite(mu_l) && (return nothing)

    if obj == :sharpe
        @constraint(model, ret >= mu_l * model[:k])
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
        @constraint(model, sum(model[:w]) == sum_short_long * model[:k])

        # Maximum number of assets constraints.
        if max_number_assets > 0
            @constraint(model, sum(tass_bin) <= max_number_assets)
            @constraint(model, tass_bin_sharpe .<= model[:k])
            @constraint(model, tass_bin_sharpe .<= 100000 * tass_bin)
            @constraint(model, tass_bin_sharpe .>= model[:k] - 100000 * (1 .- tass_bin))
            @constraint(model, model[:w] .<= long_u * tass_bin_sharpe)
        end

        if short == false
            @constraint(model, model[:w] .<= long_u * model[:k])
            @constraint(model, model[:w] .>= 0)
        else
            @variable(model, tw_ulong[1:N] .>= 0)
            @variable(model, tw_ushort[1:N] .>= 0)

            @constraint(model, sum(tw_ulong) <= long_u * model[:k])
            @constraint(model, sum(tw_ushort) <= short_u * model[:k])

            @constraint(model, model[:w] .- tw_ulong .<= 0)
            @constraint(model, model[:w] .+ tw_ushort .>= 0)

            # Maximum number of assets constraints.
            if max_number_assets > 0
                @constraint(model, model[:w] .>= -short_u * tass_bin_sharpe)
            end
        end
    else
        @constraint(model, sum(model[:w]) == sum_short_long)

        # Maximum number of assets constraints.
        if max_number_assets > 0
            @constraint(model, sum(tass_bin) <= max_number_assets)
            @constraint(model, model[:w] .<= long_u * tass_bin)
        end

        if short == false
            @constraint(model, model[:w] .<= long_u)
            @constraint(model, model[:w] .>= 0)
        else
            @variable(model, tw_ulong[1:N] .>= 0)
            @variable(model, tw_ushort[1:N] .>= 0)

            @constraint(model, sum(tw_ulong) <= long_u)
            @constraint(model, sum(tw_ushort) <= short_u)

            @constraint(model, model[:w] .- tw_ulong .<= 0)
            @constraint(model, model[:w] .+ tw_ushort .>= 0)

            # Maximum number of assets constraints.
            if max_number_assets > 0
                @constraint(model, model[:w] .>= -short_u * tass_bin)
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

    # Linear weight constraints.
    if obj == :sharpe
        @constraint(model, A * model[:w] .- B * model[:k] .>= 0)
    else
        @constraint(model, A * model[:w] .- B .>= 0)
    end

    return nothing
end

function _setup_min_number_effective_assets(portfolio, obj)
    mnea = portfolio.min_number_effective_assets

    (mnea < 1) && (return nothing)

    model = portfolio.model

    @variable(model, tmnea >= 0)
    @constraint(model, [tmnea; model[:w]] in SecondOrderCone())
    if obj == :sharpe
        @constraint(model, tmnea * sqrt(mnea) <= model[:k])
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

    @variable(model, t_track_err >= 0)
    if obj == :sharpe
        @expression(model, track_err, returns * model[:w] .- benchmark * model[:k])
        @constraint(model, [t_track_err; track_err] in SecondOrderCone())
        @constraint(model, t_track_err <= tracking_err * model[:k] * sqrt(T - 1))
    else
        @expression(model, track_err, returns * model[:w] .- benchmark)
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

    @variable(model, t_turnov[1:N] >= 0)
    if obj == :sharpe
        @expression(model, turnov, model[:w] .- turnover_weights[!, :weights] * model[:k])
        @constraint(model, [i = 1:N], [t_turnov[i]; turnov[i]] in MOI.NormOneCone(2))
        @constraint(model, t_turnov .<= turnover * model[:k])
    else
        @expression(model, turnov, model[:w] .- turnover_weights[!, :weights])
        @constraint(model, [i = 1:N], [t_turnov[i]; turnov[i]] in MOI.NormOneCone(2))
        @constraint(model, t_turnov .<= turnover)
    end

    return nothing
end

function _setup_objective_function(portfolio, obj, kelly, l)
    model = portfolio.model

    if obj == :sharpe
        if model == :classic && (kelly == :exact || kelly == :approx)
            @objective(model, Max, model[:ret])
        else
            @objective(model, Min, model[:risk])
        end
    elseif obj == :min_risk
        @objective(model, Min, model[:risk])
    elseif obj == :utility
        @objective(model, Max, model[:ret] - l * model[:risk])
    elseif obj == :max_ret
        @objective(model, Max, model[:ret])
    end

    return nothing
end

function _optimize_portfolio(portfolio, N)
    solvers = portfolio.solvers
    sol_params = portfolio.sol_params
    model = portfolio.model

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
           all(isfinite.(value.(model[:w]))) &&
           all(abs.(0.0 .- value.(model[:w])) .> N^2 * eps())
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

    weights = Vector{eltype(returns)}(undef, N)
    if obj == :sharpe
        weights .= value.(model[:w]) / value(model[:k])
    else
        weights .= value.(model[:w])
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