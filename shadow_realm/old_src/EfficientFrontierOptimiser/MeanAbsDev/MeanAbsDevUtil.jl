"""
```
refresh_model!(portfolio::EffMeanAbsDev)
```

Refreshes an [`AbstractEffMeanSemivar`](@ref) model.
"""
function refresh_model!(portfolio::EffMeanAbsDev)
    default_keys = (:w, :lower_bounds, :upper_bounds, :sum_w, :n, :abs_diff,
                    :ret, :risk)
    _refresh_add_var_and_constraints(default_keys, portfolio)

    return nothing
end

"""
```
portfolio_performance(portfolio::EffMeanSemivar; rf = portfolio.rf, verbose = false)
```

Computes the portfolio return ([`port_return`](@ref)), semideviation (square root of [`port_semivar`](@ref)), and sortino ratio ([`sharpe_ratio`](@ref) adjusted to the semideviation) for a given risk free rate, `rf`.

Returns a tuple of:

`(return, semideviation, sortino ratio)`

If `verbose == true`, it prints out this information.
"""
function portfolio_performance(portfolio::EffMeanAbsDev; rf = portfolio.rf,
                               verbose = false)
    mean_ret = portfolio.mean_ret

    model = portfolio.model
    term_status = termination_status(model)

    if term_status == OPTIMIZE_NOT_CALLED
        @warn("Portfolio has not been optimised yet.")
        return 0.0, 0.0, 0.0
    else
        w = portfolio.weights

        !isnothing(mean_ret) ? μ = port_return(w, mean_ret) : μ = NaN

        target = portfolio.target
        returns = portfolio.returns

        mean_abs_dev = port_mean_abs_dev(w, returns, target)
        sr = sharpe_ratio(μ, mean_abs_dev, rf)

        if verbose
            println(term_status)
            println("Expected return: $(round(100*μ, digits=2)) %")
            println("Mean absolute deviation: $(round(100*mean_abs_dev, digits=2)) %")
            println("Sharpe Ratio: $(round(sr, digits=3))")
        end

        return μ, mean_abs_dev, sr
    end
end
