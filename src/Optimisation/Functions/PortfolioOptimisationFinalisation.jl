function _cleanup_weights(port, ::Sharpe, ::Union{Trad, WC}, ::Any)
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
function _cleanup_weights(port, ::Any, ::Union{Trad, WC}, ::Any)
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
function _cleanup_weights(port, ::Any, ::RP, ::FC)
    weights = value.(port.model[:w])
    sum_w = value(port.model[:k])
    sum_w = abs(sum_w) > eps() ? sum_w : 1
    weights .= weights / sum_w
    return weights
end
function _cleanup_weights(port, ::Any, ::Union{RP, NOC}, ::Any)
    weights = value.(port.model[:w])
    sum_w = sum(abs.(weights))
    sum_w = sum_w > eps() ? sum_w : 1
    weights .= abs.(weights) / sum_w
    return weights
end
function _cleanup_weights(port, ::Any, ::RRP, ::Any)
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

    success = false
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

        weights = _cleanup_weights(port, obj, type, class)

        push!(solvers_tried,
              key => Dict(:objective_val => objective_value(model),
                          :term_status => term_status,
                          :params => haskey(val, :params) ? val[:params] : missing,
                          :finite_weights => all_finite_weights,
                          :nonzero_weights => all_non_zero_weights,
                          :port => DataFrame(; tickers = port.assets, weights = weights)))
    end

    return if success
        isempty(solvers_tried) ? port.fail = Dict() : port.fail = solvers_tried
        weights = _cleanup_weights(port, obj, type, class)
        port.optimal[Symbol(type)] = DataFrame(; tickers = port.assets, weights = weights)
    else
        @warn("Model could not be optimised satisfactorily.\nSolvers: $solvers_tried.")
        port.fail = solvers_tried
        port.optimal[Symbol(type)] = DataFrame()
    end
end
