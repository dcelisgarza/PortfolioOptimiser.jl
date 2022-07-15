function refresh_model!(portfolio::EffCDaR)
    default_keys = (
        :w,
        :lower_bounds,
        :upper_bounds,
        :sum_w,
        :alpha,
        :u,
        :z,
        :z_geq_uma,
        :uf_geq_uimvw,
        :u1_eq_0,
        :z_geq_0,
        :u2e_geq_0,
        :ret,
        :risk,
    )
    _refresh_add_var_and_constraints(default_keys, portfolio)

    return nothing
end

function portfolio_performance(portfolio::EffCDaR; rf = portfolio.rf, verbose = false)
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
        z = value.(portfolio.model[:z])
        samples = size(portfolio.returns, 1)
        beta = portfolio.beta

        cdar_val = cdar(alpha, z, samples, beta)

        if verbose
            println(term_status)
            println("Expected annual return: $(round(100*μ, digits=2)) %")
            println("Conditional Drawdown at Risk: $(round(100*cdar_val, digits=2)) %")
            println("Ratio: $(round((μ-rf)/cdar_val, digits=3))")
        end

        return μ, cdar_val
    end
end
