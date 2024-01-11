function refresh_model!(portfolio::EffMaxDaR)
    default_keys = (:w, :lower_bounds, :upper_bounds, :sum_w, :alpha, :u, :uf_geq_uimvw,
                    :u1_eq_0, :u2e_geq_0, :u2e_leq_alpha, :ret, :risk)
    _refresh_add_var_and_constraints(default_keys, portfolio)

    return nothing
end

function portfolio_performance(portfolio::EffMaxDaR; rf = portfolio.rf, verbose = false)
    model = portfolio.model
    mean_ret = portfolio.mean_ret

    term_status = termination_status(model)
    if term_status == OPTIMIZE_NOT_CALLED
        @warn("Portfolio has not been optimised yet.")
        return 0.0, 0.0
    else
        w = portfolio.weights
        !isnothing(mean_ret) ? μ = port_return(w, mean_ret) : μ = NaN

        alpha = value(portfolio.model[:alpha])
        if haskey(model, :k)
            (alpha /= value(model[:k]))
        end

        if verbose
            println(term_status)
            println("Expected return: $(round(100*μ, digits=2)) %")
            println("Maximum Drawdown at Risk: $(round(100*alpha, digits=2)) %")
            println("Sharpe ratio: $(round((μ-rf)/alpha, digits=3))")
        end

        return μ, alpha
    end
end
