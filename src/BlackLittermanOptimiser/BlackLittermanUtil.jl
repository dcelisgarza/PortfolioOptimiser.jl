function portfolio_performance(
    portfolio::AbstractBlackLitterman,
    rf = 0.02;
    verbose = true,
)
    mean_ret = portfolio.post_ret
    cov_mtx = portfolio.post_cov
    w = portfolio.weights

    μ = port_return(w, mean_ret)
    σ = sqrt(port_variance(w, cov_mtx))
    sr = sharpe_ratio(μ, σ, rf)

    if verbose
        println("Expected annual return: $(round(100*μ, digits=2)) %")
        println("Annual volatility: $(round(100*σ, digits=2)) %")
        println("Sharpe Ratio: $(round(sr, digits=3))")
    end

    return μ, σ, sr
end