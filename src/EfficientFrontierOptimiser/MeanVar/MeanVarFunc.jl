"""
```
min_risk!(
    portfolio::EffMeanVar;
    optimiser = Ipopt.Optimizer,
    silent = true,
    optimiser_attributes = (),
)
```

Minimise the volatility ([`port_variance`](@ref)) of a [`EffMeanVar`](@ref) portfolio.

- `portfolio`: [`EffMeanVar`](@ref) structure.
- `optimiser`: `JuMP`-supported optimiser, must support quadratic objectives.
- `silent`: if `true` the optimiser will not print to console, if `false` the optimiser will print to console.
"""
function min_risk!(
    portfolio::EffMeanVar;
    optimiser = Ipopt.Optimizer,
    silent = true,
    optimiser_attributes = (),
)
    termination_status(portfolio.model) != OPTIMIZE_NOT_CALLED && refresh_model!(portfolio)

    model = portfolio.model
    w = model[:w]
    risk = model[:risk]

    @objective(model, Min, risk)
    # Add extra terms to objective function.
    extra_obj_terms = portfolio.extra_obj_terms
    if !isempty(extra_obj_terms)
        _add_to_objective!.(model, extra_obj_terms)
    end

    _setup_and_optimise(model, optimiser, silent, optimiser_attributes)

    portfolio.weights .= value.(w)

    return nothing
end

"""
```
max_return(portfolio::EffMeanVar; optimiser = Ipopt.Optimizer, silent = true, optimiser_attributes = ())
```

Maximise the return ([`port_return`](@ref)) of a [`EffMeanVar`](@ref) portfolio. Internally minimises the negative of the portfolio return.

- `portfolio`: [`EffMeanVar`](@ref) structure.
- `optimiser`: `JuMP`-supported optimiser, must support quadratic objectives.
- `silent`: if `true` the optimiser will not print to console, if `false` the optimiser will print to console.

!!! warning
    This should not be used for optimising portfolios. It's used by [`efficient_return!`](@ref) to validate the target return. This yields portfolios with large volatilities.
"""
function max_return(
    portfolio::EffMeanVar;
    optimiser = Ipopt.Optimizer,
    silent = true,
    optimiser_attributes = (),
)
    termination_status(portfolio.model) != OPTIMIZE_NOT_CALLED && refresh_model!(portfolio)

    model = copy(portfolio.model)
    ret = model[:ret]

    @objective(model, Min, -ret)
    # Add extra terms to objective function.
    extra_obj_terms = portfolio.extra_obj_terms
    if !isempty(extra_obj_terms)
        _add_to_objective!.(model, extra_obj_terms)
    end

    _setup_and_optimise(model, optimiser, silent, optimiser_attributes)

    return model
end

"""
```
max_sharpe!(
    portfolio::EffMeanVar,
    rf = portfolio.rf;
    optimiser = Ipopt.Optimizer,
    silent = true,
    optimiser_attributes = (),
)
```

Maximise the sharpe ratio ([`sharpe_ratio`](@ref)) of a [`EffMeanVar`](@ref) portfolio.

Uses a variable transformation to turn the nonlinear objective that is the sharpe ratio, into a convex quadratic minimisation problem. See [Cornuejols and Tutuncu (2006)](http://web.math.ku.dk/~rolf/CT_FinOpt.pdf) page 158 for more details.

- `portfolio`: [`EffMeanVar`](@ref) structure.
- `rf`: risk free rate.
- `optimiser`: `JuMP`-supported optimiser, must support quadratic objectives.
- `silent`: if `true` the optimiser will not print to console, if `false` the optimiser will print to console.

!!! warning
    The variable transformation also modifies every constraint in `portfolio`, including all extra constraints. Therefore, one should not call other optimsiations on the same instance after optimising the sharpe ratio, create a fresh intance instead.

!!! warning
    The variable transformation means any extra terms in the objective function may not work as intended. If you need to add extra objective terms, use [`custom_nloptimiser!`](@ref) (see the example) and add the extra objective terms in the objective function.
"""
function max_sharpe!(
    portfolio::EffMeanVar,
    rf = portfolio.rf;
    optimiser = Ipopt.Optimizer,
    silent = true,
    optimiser_attributes = (),
)
    termination_status(portfolio.model) != OPTIMIZE_NOT_CALLED && throw(
        ArgumentError(
            "Max sharpe uses a variable transformation that changes the constraints and objective function. Please create a new instance instead.",
        ),
    )

    rf = _val_compare_benchmark(rf, <, 0, 0.02, "rf")
    # _function_vs_portfolio_val_warn(rf, portfolio.rf, "rf")

    model = portfolio.model

    # We need a new variable for max_sharpe_optim.
    @variable(model, k)

    _transform_constraints_sharpe(model, k)

    # Add constraints for the transformed sharpe ratio.
    w = model[:w]
    # We have to ensure k is positive.
    @constraint(model, k_positive, k >= 0)

    ret = model[:ret]
    # Since we increased the unbounded the sum of the weights to potentially be as large as k, leave this be. Equation 8.13 in the pdf linked in docs.
    @constraint(model, max_sharpe_return, ret - rf * k == 1)

    # Objective function.
    risk = model[:risk]
    # We only have to minimise the unbounded weights and cov_mtx.
    @objective(model, Min, risk)
    # Add extra terms to objective function.
    extra_obj_terms = portfolio.extra_obj_terms
    if !isempty(extra_obj_terms)
        @warn(
            "Sharpe ratio optimisation uses a variable transformation which means extra objective terms may not behave as expected. Use custom_nloptimiser if extra objective terms are needed.",
        )
        _add_to_objective!.(model, extra_obj_terms)
    end

    _setup_and_optimise(model, optimiser, silent, optimiser_attributes)

    portfolio.weights .= value.(w) / value(k)

    return nothing
end

"""
```
max_utility!(
    portfolio::EffMeanVar,
    risk_aversion = portfolio.risk_aversion;
    optimiser = Ipopt.Optimizer,
    silent = true,
    optimiser_attributes = (),
)
```

Maximise the [`quadratic_utility`](@ref) of a [`EffMeanVar`](@ref) portfolio. Internally minimises the negative of the quadratic utility.

- `portfolio`: [`EffMeanVar`](@ref) structure.
- `risk_aversion`: the risk aversion parameter, the larger it is the lower the risk.
- `optimiser`: `JuMP`-supported optimiser, must support quadratic objectives.
- `silent`: if `true` the optimiser will not print to console, if `false` the optimiser will print to console.
"""
function max_utility!(
    portfolio::EffMeanVar,
    risk_aversion = portfolio.risk_aversion;
    optimiser = Ipopt.Optimizer,
    silent = true,
    optimiser_attributes = (),
)
    termination_status(portfolio.model) != OPTIMIZE_NOT_CALLED && refresh_model!(portfolio)

    # _function_vs_portfolio_val_warn(risk_aversion, portfolio.risk_aversion, "risk_aversion")
    risk_aversion = _val_compare_benchmark(risk_aversion, <=, 0, 1, "risk_aversion")

    model = portfolio.model

    w = model[:w]
    ret = model[:ret]
    risk = model[:risk]

    @objective(model, Min, -ret + 0.5 * risk_aversion * risk)

    # Add extra terms to objective function.
    extra_obj_terms = portfolio.extra_obj_terms
    if !isempty(extra_obj_terms)
        _add_to_objective!.(model, extra_obj_terms)
    end

    _setup_and_optimise(model, optimiser, silent, optimiser_attributes)

    portfolio.weights .= value.(w)

    return nothing
end

"""
```
efficient_return!(
    portfolio::EffMeanVar,
    target_ret = portfolio.target_ret;
    optimiser = Ipopt.Optimizer,
    silent = true,
    optimiser_attributes = (),
)
```

Minimise the [`port_variance`](@ref) of a [`EffMeanVar`](@ref) portfolio subject to the constraint for the return to be greater than or equal to `target_ret`. The portfolio is guaranteed to have a return at least equal to `target_ret`.

- `portfolio`: [`EffMeanVar`](@ref) structure.
- `target_ret`: the target return of the portfolio.
- `optimiser`: `JuMP`-supported optimiser, must support quadratic objectives.
- `silent`: if `true` the optimiser will not print to console, if `false` the optimiser will print to console.
"""
function efficient_return!(
    portfolio::EffMeanVar,
    target_ret = portfolio.target_ret;
    optimiser = Ipopt.Optimizer,
    silent = true,
    optimiser_attributes = (),
)
    termination_status(portfolio.model) != OPTIMIZE_NOT_CALLED && refresh_model!(portfolio)

    mean_ret = portfolio.mean_ret
    max_ret_model = max_return(portfolio; optimiser, silent, optimiser_attributes)
    w_max_ret = value.(max_ret_model[:w])
    max_ret = port_return(w_max_ret, mean_ret)

    correction = max(max_ret / 2, 0)
    # _function_vs_portfolio_val_warn(target_ret, portfolio.target_ret, "target_ret")
    target_ret = _val_compare_benchmark(target_ret, >, max_ret, correction, "target_ret")
    target_ret = _val_compare_benchmark(target_ret, <, 0, correction, "target_ret")

    model = portfolio.model
    w = model[:w]
    ret = model[:ret]
    risk = model[:risk]
    # Set constraint to set the portfolio return to be greater than or equal to the target return.
    @constraint(model, target_ret, ret >= target_ret)

    @objective(model, Min, risk)
    # Add extra terms to objective function.
    extra_obj_terms = portfolio.extra_obj_terms
    if !isempty(extra_obj_terms)
        _add_to_objective!.(model, extra_obj_terms)
    end

    _setup_and_optimise(model, optimiser, silent, optimiser_attributes)

    portfolio.weights .= value.(w)

    return nothing
end

"""
```
efficient_risk!(
    portfolio::EffMeanVar,
    target_risk = portfolio.target_risk;
    optimiser = Ipopt.Optimizer,
    silent = true,
    optimiser_attributes = (),
)
```

Maximise the [`port_return`](@ref) of a [`EffMeanVar`](@ref) portfolio subject to the constraint for the volatility to be less than or equal to `target_risk`. The portfolio is guaranteed to have a volatility of at most `target_risk`.

- `portfolio`: [`EffMeanVar`](@ref) structure.
- `target_ret`: the target return of the portfolio.
- `optimiser`: `JuMP`-supported optimiser, must support quadratic objectives.
- `silent`: if `true` the optimiser will not print to console, if `false` the optimiser will print to console.
"""
function efficient_risk!(
    portfolio::EffMeanVar,
    target_risk = portfolio.target_risk;
    optimiser = Ipopt.Optimizer,
    silent = true,
    optimiser_attributes = (),
)
    termination_status(portfolio.model) != OPTIMIZE_NOT_CALLED && refresh_model!(portfolio)

    cov_mtx = portfolio.cov_mtx
    min_volatility = sqrt(1 / sum(inv(cov_mtx)))

    # _function_vs_portfolio_val_warn(
    #     target_risk,
    #     portfolio.target_risk,
    #     "target_risk",
    # )
    target_risk = _val_compare_benchmark(
        target_risk,
        <,
        min_volatility,
        min_volatility,
        "min_volatility",
    )

    model = portfolio.model
    w = model[:w]
    ret = model[:ret]
    risk = model[:risk]
    # Make variance constraint.
    target_variance = target_risk^2
    @constraint(model, target_variance, risk <= target_variance)

    @objective(model, Min, -ret)
    # Add extra terms to objective function.
    extra_obj_terms = portfolio.extra_obj_terms
    if !isempty(extra_obj_terms)
        _add_to_objective!.(model, extra_obj_terms)
    end

    _setup_and_optimise(model, optimiser, silent, optimiser_attributes)

    portfolio.weights .= value.(w)

    return nothing
end
