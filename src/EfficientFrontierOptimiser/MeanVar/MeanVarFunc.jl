"""
```
min_volatility!(
    portfolio::MeanVar;
    optimiser = Ipopt.Optimizer,
    silent = true,
    optimiser_attributes = (),
)
```

Minimise the volatility ([`port_variance`](@ref)) of a [`MeanVar`](@ref) portfolio.

- `portfolio`: [`MeanVar`](@ref) structure.
- `optimiser`: `JuMP`-supported optimiser, must support quadratic objectives.
- `silent`: if `true` the optimiser will not print to console, if `false` the optimiser will print to console.
"""
function min_volatility!(
    portfolio::MeanVar;
    optimiser = Ipopt.Optimizer,
    silent = true,
    optimiser_attributes = (),
)
    termination_status(portfolio.model) != OPTIMIZE_NOT_CALLED && refresh_model!(portfolio)

    model = portfolio.model

    w = model[:w]
    @constraint(model, sum_w, sum(w) == 1)
    cov_mtx = portfolio.cov_mtx
    @objective(model, Min, port_variance(w, cov_mtx))
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
max_return(portfolio::MeanVar; optimiser = Ipopt.Optimizer, silent = true, optimiser_attributes = ())
```

Maximise the return ([`port_return`](@ref)) of a [`MeanVar`](@ref) portfolio. Internally minimises the negative of the portfolio return.

- `portfolio`: [`MeanVar`](@ref) structure.
- `optimiser`: `JuMP`-supported optimiser, must support quadratic objectives.
- `silent`: if `true` the optimiser will not print to console, if `false` the optimiser will print to console.

!!! warning
    This should not be used for optimising portfolios. It's used by [`efficient_return!`](@ref) to validate the target return. This yields portfolios with large volatilities.
"""
function max_return(
    portfolio::MeanVar;
    optimiser = Ipopt.Optimizer,
    silent = true,
    optimiser_attributes = (),
)
    termination_status(portfolio.model) != OPTIMIZE_NOT_CALLED && refresh_model!(portfolio)

    model = copy(portfolio.model)

    w = model[:w]
    @constraint(model, sum_w, sum(w) == 1)

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

"""
```
max_sharpe!(
    portfolio::MeanVar,
    rf = portfolio.rf;
    optimiser = Ipopt.Optimizer,
    silent = true,
    optimiser_attributes = (),
)
```

Maximise the sharpe ratio ([`sharpe_ratio`](@ref)) of a [`MeanVar`](@ref) portfolio.

Uses a variable transformation to turn the nonlinear objective that is the sharpe ratio, into a convex quadratic minimisation problem. See [Cornuejols and Tutuncu (2006)](http://web.math.ku.dk/~rolf/CT_FinOpt.pdf) page 158 for more details.

- `portfolio`: [`MeanVar`](@ref) structure.
- `rf`: risk free rate.
- `optimiser`: `JuMP`-supported optimiser, must support quadratic objectives.
- `silent`: if `true` the optimiser will not print to console, if `false` the optimiser will print to console.

!!! warning
    The variable transformation also modifies every constraint in `portfolio`, including all extra constraints. Therefore, one should not call other optimsiations on the same instance after optimising the sharpe ratio, create a fresh intance instead.

!!! warning
    The variable transformation means any extra terms in the objective function may not work as intended. If you need to add extra objective terms, use [`custom_nloptimiser!`](@ref) (see the example) and add the extra objective terms in the objective function.
"""
function max_sharpe!(
    portfolio::MeanVar,
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

    _function_vs_portfolio_val_warn(rf, portfolio.rf, "rf")
    rf = _val_compare_benchmark(rf, <=, 0, 0.02, "rf")

    model = portfolio.model

    # We need a new variable for max_sharpe_optim.
    @variable(model, k)

    # Modify the old constraints to account for the variable transformation.
    constTypes = list_of_constraint_types(model)
    # Go through all constraint types one by one.
    @inbounds for constType in constTypes
        constraints = all_constraints(model, constType...)
        exprArr = []
        constKey = nothing
        constName = nothing
        # Go through all constraints of one a single type.
        for (i, constraint) in enumerate(constraints)
            const_obj = constraint_object(constraint)

            intfType = typeof(const_obj.set)
            if intfType <: JuMP.MathOptInterface.EqualTo{<:Number}
                intfKey = :value
            elseif intfType <: JuMP.MathOptInterface.GreaterThan{<:Number}
                intfKey = :lower
            elseif intfType <: JuMP.MathOptInterface.LessThan{<:Number}
                intfKey = :upper
            end

            # If the constant is zero, then we continue as there's nothing to multiply k by.
            getfield(const_obj.set, intfKey) == 0 && continue
            #=
            The constraints are originally of the form.
                expr == c
                expr >= c
                expr <= c
            By adding the variable k, they become,
                expr - c*k == 0
                expr - c*k >= 0
                expr - c*k <= 0
            because the variable times the constant is a variable.
            =#
            add_to_expression!(const_obj.func, -getfield(const_obj.set, intfKey), k)
            expr = @expression(model, const_obj.func)
            # Push them to the array.
            push!(exprArr, expr)

            # Find the key of the constraint type we're transforming. We only need to do this for the first of the same type.
            if i == 1
                for (key, value) in model.obj_dict
                    if eltype(value) <: ConstraintRef
                        co = constraint_object(value[1])
                        typeof(co.set) <: intfType && (constKey = key)
                    end
                end
                constName = name(constraint)
            end
            isnothing(constKey) && continue

            # Delete old constraints and make an adjusted one with the same key.
            delete(model, constraint)
            if i == length(constraints)
                # When we're at the end of the constraints, delete the constraint key and add the new transformed constraints.
                unregister(model, constKey)

                if intfType <: JuMP.MathOptInterface.EqualTo{<:Number}
                    model[constKey] =
                        @constraint(model, exprArr .== 0, base_name = constName)
                elseif intfType <: JuMP.MathOptInterface.GreaterThan{<:Number}
                    model[constKey] =
                        @constraint(model, exprArr .>= 0, base_name = constName)
                elseif intfType <: JuMP.MathOptInterface.LessThan{<:Number}
                    model[constKey] =
                        @constraint(model, exprArr .<= 0, base_name = constName)
                end
            end
        end
    end

    # Add constraints for the transformed sharpe ratio.
    w = model[:w]
    # We have to ensure k is positive.
    @constraint(model, k_positive, k >= 0)
    # Scale the sum so that it can equal k.
    @constraint(model, sum_w, sum(w) == k)

    mean_ret = portfolio.mean_ret
    # Since we increased the unbounded the sum of the weights to potentially be as large as k, leave this be. Equation 8.13 in the pdf linked in docs.
    @constraint(model, max_sharpe_return, dot((mean_ret .- rf), w) == 1)

    # Objective function.
    cov_mtx = portfolio.cov_mtx
    # We only have to minimise the unbounded weights and cov_mtx.
    @objective(model, Min, port_variance(w, cov_mtx))
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

"""
```
max_quadratic_utility!(
    portfolio::MeanVar,
    risk_aversion = portfolio.risk_aversion;
    optimiser = Ipopt.Optimizer,
    silent = true,
    optimiser_attributes = (),
)
```

Maximise the [`quadratic_utility`](@ref) of a [`MeanVar`](@ref) portfolio. Internally minimises the negative of the quadratic utility.

- `portfolio`: [`MeanVar`](@ref) structure.
- `risk_aversion`: the risk aversion parameter, the larger it is the lower the risk.
- `optimiser`: `JuMP`-supported optimiser, must support quadratic objectives.
- `silent`: if `true` the optimiser will not print to console, if `false` the optimiser will print to console.
"""
function max_quadratic_utility!(
    portfolio::MeanVar,
    risk_aversion = portfolio.risk_aversion;
    optimiser = Ipopt.Optimizer,
    silent = true,
    optimiser_attributes = (),
)
    termination_status(portfolio.model) != OPTIMIZE_NOT_CALLED && refresh_model!(portfolio)

    _function_vs_portfolio_val_warn(risk_aversion, portfolio.risk_aversion, "risk_aversion")
    risk_aversion = _val_compare_benchmark(risk_aversion, <=, 0, 1, "risk_aversion")

    model = portfolio.model

    _make_weight_sum_constraint!(model, portfolio.market_neutral)

    w = model[:w]
    mean_ret = portfolio.mean_ret
    cov_mtx = portfolio.cov_mtx

    @objective(model, Min, -quadratic_utility(w, mean_ret, cov_mtx, risk_aversion))

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
    portfolio::MeanVar,
    target_ret = portfolio.target_ret;
    optimiser = Ipopt.Optimizer,
    silent = true,
    optimiser_attributes = (),
)
```

Minimise the [`port_variance`](@ref) of a [`MeanVar`](@ref) portfolio subject to the constraint for the return to be greater than or equal to `target_ret`. The portfolio is guaranteed to have a return at least equal to `target_ret`.

- `portfolio`: [`MeanVar`](@ref) structure.
- `target_ret`: the target return of the portfolio.
- `optimiser`: `JuMP`-supported optimiser, must support quadratic objectives.
- `silent`: if `true` the optimiser will not print to console, if `false` the optimiser will print to console.
"""
function efficient_return!(
    portfolio::MeanVar,
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
    _function_vs_portfolio_val_warn(target_ret, portfolio.target_ret, "target_ret")
    target_ret = _val_compare_benchmark(target_ret, >, max_ret, correction, "target_ret")
    target_ret = _val_compare_benchmark(target_ret, <, 0, correction, "target_ret")

    model = portfolio.model
    w = model[:w]
    cov_mtx = portfolio.cov_mtx
    # Set constraint to set the portfolio return to be greater than or equal to the target return.
    @constraint(model, target_ret, port_return(w, mean_ret) >= target_ret)

    # Make weight sum constraint.
    _make_weight_sum_constraint!(model, portfolio.market_neutral)

    @objective(model, Min, port_variance(w, cov_mtx))
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
    portfolio::MeanVar,
    target_volatility = portfolio.target_volatility;
    optimiser = Ipopt.Optimizer,
    silent = true,
    optimiser_attributes = (),
)
```

Maximise the [`port_return`](@ref) of a [`MeanVar`](@ref) portfolio subject to the constraint for the volatility to be less than or equal to `target_volatility`. The portfolio is guaranteed to have a volatility of at most `target_volatility`.

- `portfolio`: [`MeanVar`](@ref) structure.
- `target_ret`: the target return of the portfolio.
- `optimiser`: `JuMP`-supported optimiser, must support quadratic objectives.
- `silent`: if `true` the optimiser will not print to console, if `false` the optimiser will print to console.
"""
function efficient_risk!(
    portfolio::MeanVar,
    target_volatility = portfolio.target_volatility;
    optimiser = Ipopt.Optimizer,
    silent = true,
    optimiser_attributes = (),
)
    termination_status(portfolio.model) != OPTIMIZE_NOT_CALLED && refresh_model!(portfolio)

    cov_mtx = portfolio.cov_mtx
    min_volatility = sqrt(1 / sum(inv(cov_mtx)))

    _function_vs_portfolio_val_warn(
        target_volatility,
        portfolio.target_volatility,
        "target_volatility",
    )
    target_volatility = _val_compare_benchmark(
        target_volatility,
        <,
        min_volatility,
        min_volatility,
        "min_volatility",
    )

    model = portfolio.model
    w = model[:w]
    mean_ret = portfolio.mean_ret

    # Make variance constraint.
    target_variance = target_volatility^2
    variance = port_variance(w, cov_mtx)
    @constraint(model, target_variance, variance <= target_variance)

    # Make weight sum constraint.
    _make_weight_sum_constraint!(model, portfolio.market_neutral)

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