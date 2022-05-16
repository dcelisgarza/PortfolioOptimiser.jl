function refresh_model!(portfolio::AbstractEfficient)
    model = portfolio.model

    if typeof(portfolio) <: AbstractEfficientFrontier
        default_keys = (:w, :lower_bounds, :upper_bounds)
    elseif typeof(portfolio) <: AbstractEfficientSemiVar
        default_keys = (:w, :lower_bounds, :upper_bounds, :sum_w, :p, :n, :semi_var)
    elseif typeof(portfolio) <: AbstractEfficientCDaR
        default_keys = (
            :w,
            :lower_bounds,
            :upper_bounds,
            :sum_w,
            :alpha,
            :u,
            :z,
            :z_geq_uma,
            :uf_geq_uimvw,
            :u1_eq_0,
            :z_geq_0,
            :u2e_geq_0,
        )
    elseif typeof(portfolio) <: AbstractEfficientCVaR
        default_keys =
            (:w, :lower_bounds, :upper_bounds, :sum_w, :alpha, :u, :u_geq_0, :vw_a_u_geq_0)
    end
    _refresh_model!(default_keys, model)

    # Add extra variables for extra variables and objective functions back.
    extra_vars = portfolio.extra_vars
    if !isempty(extra_vars)
        _add_var_to_model!.(model, extra_vars)
    end

    # We need to add the extra constraints back.
    extra_constraints = portfolio.extra_constraints
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
