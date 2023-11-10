"""
```
refresh_model!(portfolio::EffMeanVar)
```

Refreshes an [`AbstractEffMeanVar`](@ref) model.
"""
function refresh_model!(portfolio::EffMeanVar)
    default_keys = (:w, :lower_bounds, :upper_bounds, :sum_w, :ret, :risk)
    _refresh_add_var_and_constraints(default_keys, portfolio)

    return nothing
end

"""
```
portfolio_performance(portfolio::EffMeanVar; rf = portfolio.rf, verbose = false)
```

Computes the portfolio return ([`port_return`](@ref)), volatility (square root of [`port_variance`](@ref)), and sharpe ratio ([`sharpe_ratio`](@ref)) for a given risk free rate, `rf`.

Returns a tuple of:

`(return, volatility, sharpe ratio)`

If `verbose == true`, it prints out this information.
"""
function portfolio_performance(portfolio::EffMeanVar; rf = portfolio.rf, verbose = false)
    mean_ret = portfolio.mean_ret

    model = portfolio.model
    term_status = termination_status(model)

    if term_status == OPTIMIZE_NOT_CALLED
        @warn("Portfolio has not been optimised yet.")
        return 0.0, 0.0, 0.0
    else
        w = portfolio.weights
        cov_mtx = portfolio.cov_mtx

        !isnothing(mean_ret) ? μ = port_return(w, mean_ret) : μ = NaN
        σ = sqrt(port_variance(w, cov_mtx))
        sr = sharpe_ratio(μ, σ, rf)

        if verbose
            println(term_status)
            println("Expected return: $(round(100*μ, digits=2)) %")
            println("Volatility: $(round(100*σ, digits=2)) %")
            println("Sharpe Ratio: $(round(sr, digits=3))")
        end

        return μ, σ, sr
    end
end
