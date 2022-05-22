function min_semivar!(portfolio::MeanSemivar, optimiser = Ipopt.Optimizer, silent = true)
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

    MOI.set(model, MOI.Silent(), silent)
    set_optimizer(model, optimiser)
    optimize!(model)

    portfolio.weights .= value.(w)

    return nothing
end

function max_sortino!(portfolio::MeanSemivar, optimiser = Ipopt.Optimizer, silent = true)
    termination_status(portfolio.model) != OPTIMIZE_NOT_CALLED && throw(
        ArgumentError(
            "Max sortino uses a variable transformation that changes the constraints and objective function. Please create a new instance instead.",
        ),
    )

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
    if haskey(model, :sum_w)
        delete.(model, model[:sum_w])
        unregister(model, :sum_w)
    end
    @constraint(model, sum_w, sum(w) - k == 0)

    rf = portfolio.rf
    mean_ret = portfolio.mean_ret
    # Since we increased the unbounded sum of the weights to potentially be as large as k, leave this be. Equation 8.13 in the pdf linked in docs.
    @constraint(model, max_sharpe_return, dot((mean_ret .- rf), w) == 1)

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

    MOI.set(model, MOI.Silent(), silent)
    set_optimizer(model, optimiser)
    optimize!(model)
    portfolio.weights .= value.(w) / value(k)

    return nothing
end

function max_quadratic_utility!(
    portfolio::MeanSemivar,
    risk_aversion = portfolio.risk_aversion,
    optimiser = Ipopt.Optimizer,
    silent = true,
)
    termination_status(portfolio.model) != OPTIMIZE_NOT_CALLED && refresh_model!(portfolio)

    _function_vs_portfolio_val_warn(risk_aversion, portfolio.risk_aversion, "risk_aversion")
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

    MOI.set(model, MOI.Silent(), silent)
    set_optimizer(model, optimiser)
    optimize!(model)

    portfolio.weights .= value.(w)

    return nothing
end

function efficient_return!(
    portfolio::MeanSemivar,
    target_ret = portfolio.target_ret,
    optimiser = Ipopt.Optimizer,
    silent = true,
)
    termination_status(portfolio.model) != OPTIMIZE_NOT_CALLED && refresh_model!(portfolio)

    model = portfolio.model

    mean_ret = portfolio.mean_ret
    max_ret = maximum(mean_ret)

    correction = max(max_ret / 2, 0)
    _function_vs_portfolio_val_warn(target_ret, portfolio.target_ret, "target_ret")
    target_ret = _val_compare_benchmark(target_ret, >, max_ret, correction, "target_ret")
    target_ret = _val_compare_benchmark(target_ret, <, 0, correction, "target_ret")

    w = model[:w]
    @constraint(model, target_ret, port_return(w, mean_ret) >= target_ret)

    n = model[:n]
    @objective(model, Min, dot(n, n))

    # Add extra terms to objective function.
    extra_obj_terms = portfolio.extra_obj_terms
    if !isempty(extra_obj_terms)
        _add_to_objective!.(model, extra_obj_terms)
    end

    MOI.set(model, MOI.Silent(), silent)
    set_optimizer(model, optimiser)
    optimize!(model)

    portfolio.weights .= value.(w)

    return nothing
end

function efficient_risk!(
    portfolio::MeanSemivar,
    target_semidev = portfolio.target_semidev,
    optimiser = Ipopt.Optimizer,
    silent = true,
)
    termination_status(portfolio.model) != OPTIMIZE_NOT_CALLED && refresh_model!(portfolio)

    _function_vs_portfolio_val_warn(
        target_semidev,
        portfolio.target_semidev,
        "target_semidev",
    )

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

    MOI.set(model, MOI.Silent(), silent)
    set_optimizer(model, optimiser)
    optimize!(model)

    portfolio.weights .= value.(w)

    return nothing
end
