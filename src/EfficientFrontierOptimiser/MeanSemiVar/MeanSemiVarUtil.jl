function portfolio_performance(portfolio::EfficientSemiVar, rf = 0.02; verbose = true)
    mean_ret = portfolio.mean_ret
    freq = portfolio.freq

    model = portfolio.model
    term_status = termination_status(model)

    if term_status == OPTIMIZE_NOT_CALLED
        @warn("Portfolio has not been optimised yet.")
        return 0.0, 0.0, 0.0
    else
        w = portfolio.weights

        μ = port_return(w, mean_ret)

        benchmark = portfolio.benchmark
        returns = portfolio.returns
        port_ret = returns * w
        port_ret = min.(port_ret .- benchmark, 0)

        semi_σ = sqrt(dot(port_ret, port_ret) / size(returns, 1) * freq)
        sortino_ratio = (μ - rf) / semi_σ

        if verbose
            println(term_status)
            println("Expected annual return: $(round(100*μ, digits=2)) %")
            println("Annual semi-deviation: $(round(100*semi_σ, digits=2)) %")
            println("Sortino Ratio: $(round(sortino_ratio, digits=3))")
        end

        return μ, semi_σ, sortino_ratio
    end
    return nothing
end
