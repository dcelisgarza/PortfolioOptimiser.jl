function refresh_model!(portfolio::EffUlcer)
    default_keys = (
        :w,
        :lower_bounds,
        :upper_bounds,
        :sum_w,
        :u,
        :norm_u,
        :soc_u,
        :uf_geq_uimvw,
        :u1_eq_0,
        :u2e_geq_0,
        :ret,
        :risk,
    )
    _refresh_add_var_and_constraints(default_keys, portfolio)

    return nothing
end

function portfolio_performance(portfolio::EffUlcer; rf = portfolio.rf, verbose = false)
    model = portfolio.model
    mean_ret = portfolio.mean_ret

    term_status = termination_status(model)
    if term_status == OPTIMIZE_NOT_CALLED
        @warn("Portfolio has not been optimised yet.")
        return 0.0, 0.0
    else
        w = portfolio.weights
        !isnothing(mean_ret) ? μ = port_return(w, mean_ret) : μ = NaN

        norm_u = value(model[:norm_u])
        samples = size(portfolio.returns, 1)

        haskey(model, :k) && (norm_u /= value(model[:k]))
        ulcer_index = norm_u / sqrt(samples)

        if verbose
            println(term_status)
            println("Expected return: $(round(100*μ, digits=2)) %")
            println("Ulcer Index: $(round(100*ulcer_index, digits=2)) %")
            println("Ratio: $(round((μ-rf)/ulcer_index, digits=3))")
        end

        return μ, ulcer_index
    end
end
