function refresh_model!(portfolio::EffEDaR)
    default_keys = (
        :w,
        :lower_bounds,
        :upper_bounds,
        :sum_w,
        :t,
        :s,
        :u,
        :z,
        :u1_eq_0,
        :u2e_geq_0,
        :uf_geq_uimvw,
        :sum_z_leq_s,
        :edar_con,
        :ret,
        :risk,
    )
    _refresh_add_var_and_constraints(default_keys, portfolio)

    return nothing
end

function portfolio_performance(portfolio::EffEDaR; rf = portfolio.rf, verbose = false)
    model = portfolio.model
    mean_ret = portfolio.mean_ret

    term_status = termination_status(model)
    if term_status == OPTIMIZE_NOT_CALLED
        @warn("Portfolio has not been optimised yet.")
        return 0.0, 0.0
    else
        w = portfolio.weights
        !isnothing(mean_ret) ? μ = port_return(w, mean_ret) : μ = NaN

        edar_val = value(model[:risk])
        haskey(model, :k) && (edar_val /= value(model[:k]))

        if verbose
            println(term_status)
            println("Expected return: $(round(100*μ, digits=2)) %")
            println("Entropic Drawdown at Risk: $(round(100*edar_val, digits=2)) %")
            println("Ratio: $(round((μ-rf)/edar_val, digits=3))")
        end

        return μ, edar_val
    end
end
