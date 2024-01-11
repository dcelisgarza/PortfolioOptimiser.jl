function refresh_model!(portfolio::EffMinimax)
    default_keys = (:w, :lower_bounds, :upper_bounds, :sum_w, :m, :minimax, :ret, :risk)
    _refresh_add_var_and_constraints(default_keys, portfolio)

    return nothing
end

function portfolio_performance(portfolio::EffMinimax; rf = portfolio.rf, verbose = false)
    model = portfolio.model
    mean_ret = portfolio.mean_ret

    term_status = termination_status(model)
    if term_status == OPTIMIZE_NOT_CALLED
        @warn("Portfolio has not been optimised yet.")
        return 0.0, 0.0
    else
        w = portfolio.weights
        !isnothing(mean_ret) ? μ = port_return(w, mean_ret) : μ = NaN

        minimax = value.(portfolio.model[:m])
        if haskey(model, :k)
            (minimax /= value(model[:k]))
        end

        if verbose
            println(term_status)
            println("Expected return: $(round(100*μ, digits=2)) %")
            println("Worst realisation: $(round(100*minimax, digits=2)) %")
            println("Sharpe Ratio: $(round((μ-rf)/minimax, digits=3))")
        end

        return μ, minimax
    end
end
