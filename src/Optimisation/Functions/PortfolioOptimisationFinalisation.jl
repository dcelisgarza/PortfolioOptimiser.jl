# Copywrite (c) 2025
# Author: Daniel Celis Garza <daniel.celis.garza@gmail.com>
# SPDX-License-Identifier: MIT

function cleanup_weights(port, ::Trad, ::Any)
    val_k = value(port.model[:k])
    val_k = val_k > 0 ? val_k : 1
    weights = value.(port.model[:w]) / val_k
    short = port.short
    budget = port.budget
    if short == false
        sum_w = sum(abs.(weights))
        sum_w = sum_w > eps() ? sum_w : 1
        weights .= abs.(weights) / sum_w * budget
    end
    return weights
end
function cleanup_weights(port, ::RB, ::FC)
    weights = value.(port.model[:w])
    sum_w = value(port.model[:k])
    sum_w = abs(sum_w) > eps() ? sum_w : 1
    weights .= weights / sum_w
    return weights
end
function cleanup_weights(port, ::Union{RB, NOC}, ::Any)
    weights = value.(port.model[:w])
    sum_w = sum(abs.(weights))
    sum_w = sum_w > eps() ? sum_w : 1
    weights .= abs.(weights) / sum_w
    return weights
end
function cleanup_weights(port, ::RRB, ::Any)
    weights = value.(port.model[:w])
    sum_w = sum(abs.(weights))
    sum_w = sum_w > eps() ? sum_w : 1
    weights .= abs.(weights) / sum_w
    return weights
end
function finilise_fees(port, weights)
    model = port.model
    fees = Dict{Symbol, eltype(weights)}()
    fees_long = zero(eltype(weights))
    fees_short = zero(eltype(weights))
    fees_rebal = zero(eltype(weights))
    total_fees = zero(eltype(weights))

    if haskey(model, :fees_long)
        fees_long = calc_fees(weights, port.fees.long, .>=)
        total_fees += fees_long
        fees[:fees_long] = fees_long
    end
    if haskey(model, :fees_fixed_long)
        fees_fixed_long = calc_fixed_fees(weights, port.fees.fixed_long,
                                          port.fees.tol_kwargs, .>=)
        total_fees += fees_fixed_long
        fees[:fees_fixed_long] = fees_fixed_long
    end
    if haskey(model, :fees_short)
        fees_short = calc_fees(weights, fees.short, .<)
        total_fees += fees_short
        fees[:fees_short] = fees_short
    end
    if haskey(model, :fees_fixed_short)
        fees_fixed_short = -calc_fixed_fees(weights, port.fees.fixed_short,
                                            port.fees.tol_kwargs, .<)
        total_fees += fees_fixed_short
        fees[:fees_fixed_short] = fees_fixed_short
    end
    if haskey(model, :fees_rebalance)
        fees_rebal = calc_fees(weights, port.rebalance)
        total_fees += fees_rebal
        fees[:fees_rebal] = fees_rebal
    end

    if !iszero(total_fees)
        fees[:total_fees] = total_fees
    end

    return fees
end
function push_solvers_tried!(port, type, class, solvers_tried, key, err, all_finite_weights,
                             all_non_zero_weights)
    model = port.model
    weights = cleanup_weights(port, type, class)
    fees = finilise_fees(port, weights)
    return push!(solvers_tried,
                 key => Dict(:objective_val => objective_value(model), :err => err,
                             :params => params, :finite_weights => all_finite_weights,
                             :nonzero_weights => all_non_zero_weights,
                             :port => DataFrame(; tickers = port.assets, weights = weights),
                             :fees => fees))
end
function optimise_portfolio_model(port, type, class)
    solvers = port.solvers
    model = port.model

    solvers_tried = Dict()

    success = false
    strtype = "_" * String(type)
    for solver ∈ solvers
        name = solver.name
        solver_i = solver.solver
        params = solver.params
        add_bridges = solver.add_bridges
        check_sol = solver.check_sol

        key = Symbol(String(name) * strtype)
        set_optimizer(model, solver_i; add_bridges = add_bridges)
        if !isnothing(params) && !isempty(params)
            for (k, v) ∈ params
                set_attribute(model, k, v)
            end
        end

        try
            JuMP.optimize!(model)
        catch jump_error
            push!(solvers_tried, key => Dict(:JuMP_error => jump_error))
            continue
        end

        all_finite_weights = all(isfinite.(value.(model[:w])))
        all_non_zero_weights = !all(isapprox.(abs.(value.(model[:w])),
                                              zero(eltype(port.returns))))

        try
            assert_is_solved_and_feasible(model; check_sol...)
            if all_finite_weights && all_non_zero_weights
                success = true
                break
            end
        catch err
            push_solvers_tried!(port, type, class, solvers_tried, key, err,
                                all_finite_weights, all_non_zero_weights)
        end

        err = solution_summary(model)
        push_solvers_tried!(port, type, class, solvers_tried, key, err, all_finite_weights,
                            all_non_zero_weights)
    end

    return if success
        if !isempty(solvers_tried)
            port.fail = solvers_tried
        end
        weights = cleanup_weights(port, type, class)
        fees = finilise_fees(port, weights)
        if !isempty(fees)
            port.optimal[Symbol(String(type) * "_fees")] = fees
        end
        port.optimal[Symbol(type)] = DataFrame(; tickers = port.assets, weights = weights)
    else
        @warn("Model could not be optimised satisfactorily.\nSolvers: $solvers_tried.")
        port.fail = solvers_tried
        port.optimal[Symbol(type)] = DataFrame()
    end
end
