function refresh_model!(portfolio::EffCVaR)
    default_keys = (:w, :lower_bounds, :upper_bounds, :sum_w, :alpha, :u, :vw_a_u_geq_0,
                    :ret, :risk)
    _refresh_add_var_and_constraints(default_keys, portfolio)

    return nothing
end

function portfolio_performance(portfolio::EffCVaR; rf = portfolio.rf, verbose = false)
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
        u = value.(portfolio.model[:u])
        samples = size(portfolio.returns, 1)
        beta = portfolio.beta

        cvar_val = cvar(alpha, u, samples, beta)
        if haskey(model, :k)
            (cvar_val /= value(model[:k]))
        end

        if verbose
            println(term_status)
            println("Expected return: $(round(100*μ, digits=2)) %")
            println("Conditional Value at Risk: $(round(100*cvar_val, digits=2)) %")
            println("Sharpe Ratio: $(round((μ-rf)/cvar_val, digits=3))")
        end

        return μ, cvar_val
    end
end
