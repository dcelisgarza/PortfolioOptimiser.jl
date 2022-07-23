function portfolio_performance(portfolio::HRPOpt; rf = portfolio.rf, verbose = false)
    w = portfolio.weights

    if isnothing(portfolio.returns)
        cov_mtx = portfolio.cov_mtx
        μ = NaN
        sr = NaN

        σ = sqrt(port_variance(w, cov_mtx))

        if verbose
            println("Annual volatility: $(round(100*σ, digits=2)) %")
        end
    else
        cov_mtx = cov(portfolio.returns)
        mean_ret = vec(mean(portfolio.returns, dims = 1))

        μ = dot(w, mean_ret)
        σ = sqrt(port_variance(w, cov_mtx))
        sr = sharpe_ratio(μ, σ, rf)

        if verbose
            println("Expected return: $(round(100*μ, digits=2)) %")
            println("Volatility: $(round(100*σ, digits=2)) %")
            println("Sharpe Ratio: $(round(sr, digits=3))")
        end
    end

    return μ, σ, sr
end