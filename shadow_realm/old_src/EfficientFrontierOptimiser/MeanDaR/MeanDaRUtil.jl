function refresh_model!(portfolio::EffMeanDaR)
    default_keys = (:w, :lower_bounds, :upper_bounds, :sum_w, :alpha, :u, :uf_geq_uimvw,
                    :u1_eq_0, :u2e_geq_0, :ret, :risk)
    _refresh_add_var_and_constraints(default_keys, portfolio)

    return nothing
end

function portfolio_performance(portfolio::EffMeanDaR; rf = portfolio.rf, verbose = false)
    model = portfolio.model
    mean_ret = portfolio.mean_ret

    term_status = termination_status(model)
    if term_status == OPTIMIZE_NOT_CALLED
        @warn("Portfolio has not been optimised yet.")
        return 0.0, 0.0
    else
        w = portfolio.weights
        !isnothing(mean_ret) ? μ = port_return(w, mean_ret) : μ = NaN
        samples = size(portfolio.returns, 1)
        u = model[:u]

        mean_dar = sum(u[2:end]) / samples
        if haskey(model, :k)
            (mean_dar /= value(model[:k]))
        end

        if verbose
            println(term_status)
            println("Expected return: $(round(100*μ, digits=2)) %")
            println("Mean Drawdown at Risk: $(round(100*mean_dar, digits=2)) %")
            println("Ratio: $(round((μ-rf)/mean_dar, digits=3))")
        end

        return μ, mean_dar
    end
end
