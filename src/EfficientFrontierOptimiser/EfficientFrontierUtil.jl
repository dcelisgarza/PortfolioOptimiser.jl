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
        constraint_keys =
            [Symbol("extra_constraint$(i)") for i in 1:length(extra_constraints)]
        _add_constraint_to_model!.(model, constraint_keys, extra_constraints)
    end

    return nothing
end

function _function_vs_portfolio_val_warn(fval, pval, name)
    if fval != pval
        @warn(
            "The value of $(name): $fval, provided to the function does not match the one in the portfolio: $(pval). Using function value: $fval, instead."
        )
    end
    return nothing
end

function _val_compare_benchmark(val, op, benchmark, correction, name)
    if op(val, benchmark)
        @warn(
            "Value of $name, $val $(String(Symbol(op))) $benchmark. Correcting to $correction."
        )
        val = correction
    end
    return val
end
