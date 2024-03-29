function portfolio_performance(portfolio::AbstractBlackLitterman, rf = 1.02^(1 / 252) - 1;
                               verbose = false,)
    mean_ret = portfolio.post_ret
    cov_mtx = portfolio.post_cov
    w = portfolio.weights

    μ = port_return(w, mean_ret)
    σ = sqrt(port_variance(w, cov_mtx))
    sr = sharpe_ratio(μ, σ, rf)

    if verbose
        println("Expected return: $(round(100*μ, digits=2)) %")
        println("Volatility: $(round(100*σ, digits=2)) %")
        println("Sharpe Ratio: $(round(sr, digits=3))")
    end

    return μ, σ, sr
end
