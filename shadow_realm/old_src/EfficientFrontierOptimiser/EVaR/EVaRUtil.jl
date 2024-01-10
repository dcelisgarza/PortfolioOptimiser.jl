function refresh_model!(portfolio::EffEVaR)
    default_keys = (:w,
                    :lower_bounds,
                    :upper_bounds,
                    :sum_w,
                    :t,
                    :s,
                    :u,
                    :X,
                    :sum_u_leq_s,
                    :evar_con,
                    :ret,
                    :risk)
    _refresh_add_var_and_constraints(default_keys, portfolio)

    return nothing
end

function portfolio_performance(portfolio::EffEVaR; rf = portfolio.rf,
                               verbose = false)
    model = portfolio.model
    mean_ret = portfolio.mean_ret

    term_status = termination_status(model)
    if term_status == OPTIMIZE_NOT_CALLED
        @warn("Portfolio has not been optimised yet.")
        return 0.0, 0.0
    else
        w = portfolio.weights
        !isnothing(mean_ret) ? μ = port_return(w, mean_ret) : μ = NaN

        evar_val = value(model[:risk])
        haskey(model, :k) && (evar_val /= value(model[:k]))

        if verbose
            println(term_status)
            println("Expected return: $(round(100*μ, digits=2)) %")
            println("Entropic Value at Risk: $(round(100*evar_val, digits=2)) %")
            println("Sharpe Ratio: $(round((μ-rf)/evar_val, digits=3))")
        end

        return μ, evar_val
    end
end
