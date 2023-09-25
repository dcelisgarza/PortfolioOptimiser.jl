function portfolio_performance(
    portfolio::NearOptCentering;
    rf = portfolio.opt_port.rf,
    verbose = false,
)
    model = portfolio.model
    mean_ret = portfolio.opt_port.mean_ret

    term_status = termination_status(model)
    if term_status == OPTIMIZE_NOT_CALLED
        @warn("Portfolio has not been optimised yet.")
        return 0.0, 0.0
    else
        if !isnothing(mean_ret)
            ret = value(portfolio.model[:ret])
        else
            ret = NaN
        end

        risk = value(portfolio.model[:risk])

        if haskey(model, :k)
            risk /= value(model[:k])
            ret /= value(model[:k])
        end

        sr = (ret - rf) / risk

        if verbose
            println(term_status)
            println("Near Optimal Centering:")
            println("Expected return: $(round(100*ret, digits=2)) %")
            println("Risk: $(round(100*risk, digits=2)) %")
            println("Ratio: $(round(sr, digits=3))")
            println("\nOptimal portfolio:")
        end

        opt_stats = portfolio_performance(portfolio.opt_port; rf, verbose)

        return ret, risk, sr, opt_stats
    end
end
