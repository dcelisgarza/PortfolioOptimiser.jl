"""
```
refresh_model!(portfolio::AbstractEffMeanSemivar)
```

Refreshes an [`AbstractEffMeanSemivar`](@ref) model.
"""
function refresh_model!(portfolio::AbstractEffMeanSemivar)
    default_keys = (:w, :lower_bounds, :upper_bounds, :sum_w, :n, :semi_var, :ret, :risk)
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
function portfolio_performance(
    portfolio::EffMeanSemivar;
    rf = portfolio.rf,
    verbose = false,
)
    mean_ret = portfolio.mean_ret
    freq = portfolio.freq

    model = portfolio.model
    term_status = termination_status(model)

    if term_status == OPTIMIZE_NOT_CALLED
        @warn("Portfolio has not been optimised yet.")
        return 0.0, 0.0, 0.0
    else
        w = portfolio.weights

        !isnothing(mean_ret) ? μ = port_return(w, mean_ret) : μ = NaN

        benchmark = portfolio.benchmark
        returns = portfolio.returns

        semi_σ = sqrt(port_semivar(w, returns, benchmark, freq))
        sortino_ratio = sharpe_ratio(μ, semi_σ, rf)

        if verbose
            println(term_status)
            println("Expected annual return: $(round(100*μ, digits=2)) %")
            println("Annual semi-deviation: $(round(100*semi_σ, digits=2)) %")
            println("Sortino Ratio: $(round(sortino_ratio, digits=3))")
        end

        return μ, semi_σ, sortino_ratio
    end
end
