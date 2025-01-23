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
    long_fees = zero(eltype(weights))
    short_fees = zero(eltype(weights))
    rebal_fees = zero(eltype(weights))
    total_fees = zero(eltype(weights))
    if haskey(model, :long_fees)
        idx = weights .>= zero(eltype(weights))
        long_fees = port.long_fees
        long_fees = if isa(long_fees, Real)
            sum(long_fees * weights[idx])
        else
            dot(long_fees[idx], weights[idx])
        end
        total_fees += long_fees
        fees[:long_fees] = long_fees
    end
    if haskey(model, :short_fees)
        idx = weights .< zero(eltype(weights))
        short_fees = port.short_fees
        short_fees = if isa(short_fees, Real)
            sum(short_fees * weights[idx])
        else
            dot(short_fees[idx], weights[idx])
        end
        total_fees += short_fees
        fees[:short_fees] = short_fees
    end
    if haskey(model, :rebalance_fees)
        rebalance = port.rebalance
        rebal_fees = rebalance.val
        benchmark = rebalance.w
        rebal_fees = if isa(rebal_fees, Real)
            sum(rebal_fees * abs.(benchmark .- weights))
        else
            dot(rebal_fees, abs.(benchmark .- weights))
        end
        total_fees += rebal_fees
        fees[:rebal_fees] = rebal_fees
    end

    if !iszero(total_fees)
        fees[:total_fees] = total_fees
    end

    return fees
end
function optimise_portfolio_model(port, obj, type, class)
    solvers = port.solvers
    model = port.model

    term_status = termination_status(model)
    solvers_tried = Dict()

    success = false
    strtype = "_" * String(type)
    for (key, val) ∈ solvers
        key = Symbol(String(key) * strtype)

        if haskey(val, :solver)
            set_optimizer(model, val[:solver];
                          add_bridges = if !haskey(val, :add_bridges)
                              true
                          else
                              val[:add_bridges]
                          end)
        end

        if haskey(val, :params)
            for (attribute, value) ∈ val[:params]
                set_attribute(model, attribute, value)
            end
        end

        if haskey(val, :check_sol)
            check_sol = val[:check_sol]
        else
            check_sol = (;)
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

        if is_solved_and_feasible(model; check_sol...) &&
           all_finite_weights &&
           all_non_zero_weights
            success = true
            break
        else
            term_status = termination_status(model)
        end

        weights = cleanup_weights(port, type, class)
        fees = finilise_fees(port, weights)
        push!(solvers_tried,
              key => Dict(:objective_val => objective_value(model),
                          :term_status => term_status,
                          :params => haskey(val, :params) ? val[:params] : missing,
                          :finite_weights => all_finite_weights,
                          :nonzero_weights => all_non_zero_weights,
                          :port => DataFrame(; tickers = port.assets, weights = weights),
                          :fees => fees))
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
