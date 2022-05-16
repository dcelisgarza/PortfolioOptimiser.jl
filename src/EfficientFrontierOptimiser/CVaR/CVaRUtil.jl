function portfolio_performance(portfolio::EfficientCVaR, verbose = true)
    model = portfolio.model
    mean_ret = portfolio.mean_ret

    term_status = termination_status(model)
    if term_status == OPTIMIZE_NOT_CALLED
        @warn("Portfolio has not been optimised yet.")
        return 0.0, 0.0
    else
        w = portfolio.weights
        μ = port_return(w, mean_ret)

        alpha = value(portfolio.model[:alpha])
        u = value.(portfolio.model[:u])
        samples = size(portfolio.returns, 1)
        beta = portfolio.beta

        cvar_val = cvar(alpha, u, samples, beta)

        if verbose
            println(term_status)
            println("Expected annual return: $(round(100*μ, digits=2)) %")
            println("Conditional Value at Risk: $(round(100*cvar_val, digits=2)) %")
        end

        return μ, cvar_val
    end
    return nothing
end