function portfolio_performance(portfolio::EfficientFrontier; verbose = true)
    rf = portfolio.rf
    mean_ret = portfolio.mean_ret

    model = portfolio.model
    term_status = termination_status(model)

    if term_status == OPTIMIZE_NOT_CALLED
        @warn("Portfolio has not been optimised yet.")
        return 0.0, 0.0, 0.0
    else
        w = portfolio.weights
        cov_mtx = portfolio.cov_mtx

        μ = port_return(w, mean_ret)
        σ = sqrt(port_variance(w, cov_mtx))
        sr = sharpe_ratio(μ, σ, rf)

        if verbose
            println(term_status)
            println("Expected annual return: $(round(100*μ, digits=2)) %")
            println("Annual volatility: $(round(100*σ, digits=2)) %")
            println("Sharpe Ratio: $(round(sr, digits=3))")
        end

        return μ, σ, sr
    end
    return nothing
end
