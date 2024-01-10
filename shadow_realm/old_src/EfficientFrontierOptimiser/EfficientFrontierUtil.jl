"""
```
_refresh_add_var_and_constraints(default_keys, portfolio)
```

Helper function for refreshing portfolios and re-adding extra variables and constraints used to build the model via the constructor.
"""
function _refresh_add_var_and_constraints(default_keys, portfolio)
    model = portfolio.model
    extra_vars = portfolio.extra_vars
    extra_constraints = portfolio.extra_constraints

    _refresh_model!(default_keys, model)

    # Add extra variables for extra variables and objective functions back.
    if !isempty(extra_vars)
        _add_var_to_model!.(model, extra_vars)
    end

    # We need to add the extra constraints back.
    if !isempty(extra_constraints)
        constraint_keys = [Symbol("extra_constraint$(i)")
                           for i in 1:length(extra_constraints)]
        _add_constraint_to_model!.(model, constraint_keys, extra_constraints)
    end

    return nothing
end

function _transform_constraints_sharpe(model, k, fname = "max_sharpe!")
    # Go through all registered variables and only look at constraint and constraint arrays.
    for (key, value) in model.obj_dict
        if eltype(value) <: ConstraintRef
            constraints = constraint_object.(value)
            constKey = key
            constName = name(value[1])
        elseif typeof(value) <: ConstraintRef
            # Make this a tuple so we can use the same looping code.
            constraints = (constraint_object(value),)
            constKey = key
            constName = name(value)
        else
            # Continue if the value is not a constraint or constraint array.
            continue
        end

        intfType = typeof(constraints[1].set)
        if intfType <: JuMP.MOI.EqualTo{<:Number}
            intfKey = :value
        elseif intfType <: JuMP.MOI.GreaterThan{<:Number}
            intfKey = :lower
        elseif intfType <: JuMP.MOI.LessThan{<:Number}
            intfKey = :upper
        else
            @warn("$fname optimisation uses a variable transformation which means constraint types other than linear and quadratic may not behave as expected. Use custom_nloptimiser! if constraints of type $intfType are needed.",)
            continue
        end

        exprArr = []
        skip = false

        for constraint in constraints
            # If the constant is zero, then we continue as there's nothing to multiply k by.

            if getfield(constraint.set, intfKey) == 0
                skip = true
                break
            end
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
            add_to_expression!(constraint.func,
                               -getfield(constraint.set, intfKey), k)
            expr = @expression(model, constraint.func)
            # Push them to the array.
            push!(exprArr, expr)
        end

        # If the inner loop broke out, then we don't want to change anything and we continue to the next iteration.
        skip && continue

        # Delete old constraint.
        delete(model, value)

        unregister(model, constKey)
        # When we're at the end of the constraints, delete the constraint key and add the new transformed constraints.

        if intfType <: JuMP.MOI.EqualTo{<:Number}
            model[constKey] = @constraint(model, exprArr .== 0,
                                          base_name = constName)
        elseif intfType <: JuMP.MOI.GreaterThan{<:Number}
            model[constKey] = @constraint(model, exprArr .>= 0,
                                          base_name = constName)
        elseif intfType <: JuMP.MOI.LessThan{<:Number}
            model[constKey] = @constraint(model, exprArr .<= 0,
                                          base_name = constName)
        end
    end
end
