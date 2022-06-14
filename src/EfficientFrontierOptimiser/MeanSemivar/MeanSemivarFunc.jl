function min_semivar!(
    portfolio::MeanSemivar;
    optimiser = Ipopt.Optimizer,
    silent = true,
    optimiser_attributes = (),
)
    termination_status(portfolio.model) != OPTIMIZE_NOT_CALLED && refresh_model!(portfolio)

    model = portfolio.model

    w = model[:w]
    n = model[:n]

    @objective(model, Min, dot(n, n))
    # Add extra terms to objective function.
    extra_obj_terms = portfolio.extra_obj_terms
    if !isempty(extra_obj_terms)
        _add_to_objective!.(model, extra_obj_terms)
    end

    _setup_and_optimise(model, optimiser, silent, optimiser_attributes)

    portfolio.weights .= value.(w)

    return nothing
end

function max_return(
    portfolio::MeanSemivar;
    optimiser = Ipopt.Optimizer,
    silent = true,
    optimiser_attributes = (),
)
    termination_status(portfolio.model) != OPTIMIZE_NOT_CALLED && refresh_model!(portfolio)

    model = copy(portfolio.model)

    w = model[:w]

    mean_ret = portfolio.mean_ret
    @objective(model, Min, -port_return(w, mean_ret))
    # Add extra terms to objective function.
    extra_obj_terms = portfolio.extra_obj_terms
    if !isempty(extra_obj_terms)
        _add_to_objective!.(model, extra_obj_terms)
    end

    _setup_and_optimise(model, optimiser, silent, optimiser_attributes)

    return model
end

function max_sortino!(
    portfolio::MeanSemivar,
    rf = portfolio.rf;
    optimiser = Ipopt.Optimizer,
    silent = true,
    optimiser_attributes = (),
)
    termination_status(portfolio.model) != OPTIMIZE_NOT_CALLED && throw(
        ArgumentError(
            "Max sortino uses a variable transformation that changes the constraints and objective function. Please create a new instance instead.",
        ),
    )

    rf = _val_compare_benchmark(rf, <=, 0, 0.02, "rf")
    # _function_vs_portfolio_val_warn(rf, portfolio.rf, "rf")

    model = portfolio.model

    # We need a new variable for max_sharpe_optim.
    @variable(model, k)

    _transform_constraints_sharpe(model, k, "max_sortino!")

    # Add constraints for the transformed sharpe ratio.
    w = model[:w]
    # We have to ensure k is positive.
    @constraint(model, k_positive, k >= 0)
    # Scale the sum so that it can equal k.
    if haskey(model, :sum_w)
        delete.(model, model[:sum_w])
        unregister(model, :sum_w)
    end
    @constraint(model, sum_w, sum(w) - k == 0)

    mean_ret = portfolio.mean_ret .- rf
    # Since we increased the unbounded sum of the weights to potentially be as large as k, leave this be. Equation 8.13 in the pdf linked in docs.
    @constraint(model, max_sharpe_return, port_return(w, mean_ret) == 1)

    n = model[:n]
    # Objective function.
    @objective(model, Min, dot(n, n))

    # Add extra terms to objective function.
    extra_obj_terms = portfolio.extra_obj_terms
    if !isempty(extra_obj_terms)
        @warn(
            "Sharpe ratio optimisation uses a variable transformation which means extra objective terms may not behave as expected.",
        )
        _add_to_objective!.(model, extra_obj_terms)
    end

    _setup_and_optimise(model, optimiser, silent, optimiser_attributes)

    portfolio.weights .= value.(w) / value(k)

    return nothing
end

function max_quadratic_utility!(
    portfolio::MeanSemivar,
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
    n = model[:n]

    mean_ret = portfolio.mean_ret
    freq = portfolio.freq
    μ = port_return(w, mean_ret) / freq
    @objective(model, Min, -μ + 0.5 * risk_aversion * dot(n, n))

    # Add extra terms to objective function.
    extra_obj_terms = portfolio.extra_obj_terms
    if !isempty(extra_obj_terms)
        _add_to_objective!.(model, extra_obj_terms)
    end

    _setup_and_optimise(model, optimiser, silent, optimiser_attributes)

    portfolio.weights .= value.(w)

    return nothing
end

function efficient_return!(
    portfolio::MeanSemivar,
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
    @constraint(model, target_ret, port_return(w, mean_ret) >= target_ret)

    n = model[:n]
    @objective(model, Min, dot(n, n))

    # Add extra terms to objective function.
    extra_obj_terms = portfolio.extra_obj_terms
    if !isempty(extra_obj_terms)
        _add_to_objective!.(model, extra_obj_terms)
    end

    _setup_and_optimise(model, optimiser, silent, optimiser_attributes)

    portfolio.weights .= value.(w)

    return nothing
end

function efficient_risk!(
    portfolio::MeanSemivar,
    target_semidev = portfolio.target_semidev;
    optimiser = Ipopt.Optimizer,
    silent = true,
    optimiser_attributes = (),
)
    termination_status(portfolio.model) != OPTIMIZE_NOT_CALLED && refresh_model!(portfolio)

    # _function_vs_portfolio_val_warn(
    #     target_semidev,
    #     portfolio.target_semidev,
    #     "target_semidev",
    # )

    model = portfolio.model
    n = model[:n]

    freq = portfolio.freq
    target_semivariance = target_semidev^2
    @constraint(model, target_semivariance, freq * dot(n, n) <= target_semivariance)

    w = model[:w]
    mean_ret = portfolio.mean_ret
    @objective(model, Min, -port_return(w, mean_ret))

    # Add extra terms to objective function.
    extra_obj_terms = portfolio.extra_obj_terms
    if !isempty(extra_obj_terms)
        _add_to_objective!.(model, extra_obj_terms)
    end

    _setup_and_optimise(model, optimiser, silent, optimiser_attributes)

    portfolio.weights .= value.(w)

    return nothing
end
