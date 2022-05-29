"""
```
_refresh_model!(default_keys, model)
```

Helper function for cleaning up `JuMP.Model()` objects.

- `default_keys`: collection of keys that are to be kept in the model.
- `model`: `JuMP.Model()` object to be cleaned up.
"""
function _refresh_model!(default_keys, model)
    for key in keys(model.obj_dict)
        key ∈ default_keys && continue
        delete.(model, model[key])
        unregister(model, key)
    end

    return nothing
end

"""
```
_create_weight_bounds(num_tickers, bounds)
```

Create the upper and lower bounds for asset weights.

- `num_tickers`: The number of tickers.
- `bounds`: weight bounds.
    1. If `bounds` is a collection of length two, it assumes the first entry `bounds[1]` are the *lower bounds* for *all* weights, and `bounds[2]` is the *upper bounds* for *all* weights.
    2. If `bounds` is an abstract vector of length equal to `num_tickers`, and each element of the vector is a tuple of length 2. Then each element corresponds to an asset, the first element of the tuple is the lower bound, the second the upper bound for said asset.
    3. Else throw an error.

It returns a tuple of:

- `(lower_bounds, upper_bounds)`: where the length of `lower_bounds` and `upper_bounds` is equal to `num_tickers`. These are then used by [`AbstractEfficient`](@ref) portfolios to set the bounds constraints for their `JuMP.Model()`.
"""
function _create_weight_bounds(num_tickers, bounds)
    # If the bounds is an array of tuples, assume each is a bound.
    if typeof(bounds) <: AbstractVector &&
       length(bounds) == num_tickers &&
       (eltype(bounds) <: Tuple || eltype(bounds) <: AbstractVector) &&
       all(x -> x == 2, length.(bounds))
        bounds = Float64.(reshape(collect(Iterators.flatten(bounds)), 2, :))
        lower_bounds, upper_bounds = bounds
    elseif length(bounds) == 2
        lower, upper = Float64.(bounds)
        lower_bounds = fill(lower, num_tickers)
        upper_bounds = fill(upper, num_tickers)
    else
        throw(
            ArgumentError(
                "Weight bounds must either be a tuple of length 2, or a vector of length $num_tickers of tuples/vectors of length 2.",
            ),
        )
    end

    return lower_bounds, upper_bounds
end

"""
```
_make_weight_sum_constraint!(model, market_neutral)
```

Create the weight sum constraints for a `JuMP.Model()`.

- `model`: a `JuMP.Model()` of a portfolio optimiser.
- `market_neutral`: if `true` the portfolio is market neutral and the weights will be constrained to sum to `0`. If `false`, the portfolio is not market neutral and the weights will be constrained to sum to `1`. 
    - If `market_neutral` is `true` and *all* `lower_bounds` constraints are greater than or equal to zero, then a portfolio cannot be market neutral. If this is the case, the `lower_bounds` constraint for all assets will be changed to be greater than or equal to `-1`.
"""
function _make_weight_sum_constraint!(model, market_neutral)
    w = model[:w]
    # If portfolio is not market neutral.
    if !market_neutral
        @constraint(model, sum_w, sum(w) == 1)
    else
        port_possible = true
        lower_boundsConst = model[:lower_bounds]
        port_possible = any(
            getfield.(getfield.(constraint_object.(lower_boundsConst), :set), :lower) .< 0,
        )

        # If the at least one lower weight bound is not negative, we need to make them.
        if !port_possible
            @warn(
                "Market neutrality requires shorting, upper and lower bounds changed to (-1, 1) for all tickers."
            )

            num_tickers = length(w)
            lower_bounds, upper_bounds = _create_weight_bounds(num_tickers, (-1, 1))

            delete.(model, lower_boundsConst)
            unregister(model, :lower_bounds)
            delete.(model, model[:upper_bounds])
            unregister(model, :upper_bounds)

            @constraint(model, lower_bounds, w .>= lower_bounds)
            @constraint(model, upper_bounds, w .<= upper_bounds)
        end

        @constraint(model, sum_w, sum(w) == 0)
    end

    return nothing
end

"""
```
_add_var_to_model!(model, var, args...)
```

Function to add a `JuMP.@variable` to a `JuMP.Model()`.

- `model`: a `JuMP.Model()`.
- `var`: symbol or expression representing a `JuMP.@variable`. Extra array variables have to be expressions. Extra scalar variables can be symbols or expressions.
- `args...`: expression representing the `args...` to a `JuMP.@variable`.

Variables can also be directly added to a `model` using `JuMP.@variable`. This is used internally by [`AbstractEfficient`](@ref) portfolios, so that when a portfolio optimiser is refreshed, the extra variables can be added back automatically.

## Example

We can use this to add a vector variable `c` of length 3, a scalar variable `k`, and a `3 x 5 x 4` multidimensional array variable `m`.

```julia
extra_vars = [ :( c[1:3] ), :k, :( m[1:3, 1:5, 1:4] ) ]
ef = MeanVar(tickers, mean_ret, cov_mtx)
_add_var_to_model!.(ef.model, extra_vars)
```

Alternatively, we can make sure they get registered in the optimiser. If the portfolio optimiser is refreshed, the extra variables are added back to the model automatically. They'd have to be added back manually otherwise.

```julia
extra_vars = [ :( c[1:3] ), :k, :( m[1:3, 1:5, 1:4] ) ]
ef = MeanVar(tickers, mean_ret, cov_mtx; extra_vars = extra_vars)
```

!!! note
    When adding variables to the constructor, they must be a collection (Tuple or vector), even if it's of length 1.
"""
function _add_var_to_model!(model, var, args...)
    if typeof(var) <: Expr
        if var.head == :call
            larg = var.args[2:end]
            larg = larg[typeof.(larg) .<: Symbol]
            svar = Symbol(larg)
        elseif var.head == :ref
            svar = var.args[1]
        else
            svar = var.head
        end
    else
        svar = var
    end
    if !haskey(model, svar)
        eval(quote
            @variable($model, $var, $(args...))
        end)
    else
        @warn("Variable $var already in model.")
    end

    return nothing
end

"""
```
_add_constraint_to_model!(model, key, constraint)
```

Function to add a `JuMP.@constraint` to a `JuMP.Model()`.

- `model`: a `JuMP.Model()`.
- `key`: the key under which the constraint will be registered in the `JuMP.Model()`.
- `constraint`: expression representing the constraint.

Constraints can also be directly added to a `model` using `JuMP.@constraint`. This is used internally by [`AbstractEfficient`](@ref) portfolios, so that when a portfolio optimiser is refreshed, the extra constraints can be added back automatically.

!!! warning
    Variables used within the `constraint` must be registered in the `JuMP.Model()` object (hence [`_add_var_to_model!`](@ref)), and referred to as `model[<key>]`, where `<key>` is the key of the variable. For example, when adding a constraint such that the first weight is less than or equal to 0.2, `:(model[:w][1] <= 0.2)`.

!!! warning
    Constraints must be expressions `:()`, not quotes `quote ... end`, because quotes can contain be multiple expressions.

!!! note
    This uses `JuMP.@constraint` internally, so any constraint given must be compatible with `JuMP.@constraint`.

## Example

We can use this to add a constraint, `:const1` such that the weight of the second asset is greater than or equal to 2, and `:const2` such that the sum of all weights is less than or equal to 5. These are nonsensical constraints, but they illustrate the point. A constraint will work as long as it is a valid `JuMP.@constraint`. For a more complex example of a constraint see [`transaction_cost`](@ref).

```julia
extra_constraints = [:(model[:w][2] >= 2), :(sum(model[:w]) <= 5)]
ef = MeanVar(tickers, mean_ret, cov_mtx)
_add_constraint_to_model!.(ef.model, [:const1, :const2], extra_constraints)
```

We can also use variable interpolation into the expressions.

```julia
ef = MeanVar(tickers, mean_ret, cov_mtx)
w = ef.model[:w]
extra_constraints = [:(\$(w[2]) >= 2), :(sum(\$w) <= 5)]
_add_constraint_to_model!.(ef.model, [:const1, :const2], extra_constraints)
```

If we want the extra constraints to be added back automatically when refreshing a portfolio optimiser, we can add them to the constructor.

```julia
extra_constraints = [:(model[:w][2] >= 2), :(sum(model[:w]) <= 5)]
ef = MeanVar(tickers, mean_ret, cov_mtx; extra_constraints = extra_constraints)
```

!!! note
    When adding constraints to the constructor, they must be a collection (Tuple or vector), even if it's of length 1.
"""
function _add_constraint_to_model!(model, key, constraint)
    if haskey(model, key)
        @warn("Constraint $key already in model. Deleting and replacing it with new key.")
        delete.(model, model[key])
        unregister(model, key)
    end
    eval(quote
        model = $model
        @constraint($model, $key, $constraint)
    end)
    return nothing
end

"""
```
_add_to_objective!(model, expr)
```

Function to add a term to a `JuMP.@objective` to a `JuMP.Model()`.

- `model`: a `JuMP.Model()`.
- `expr`: the expression to add to the objective function.

!!! warning
    Variables used within the extra objective must be registered in the `JuMP.Model()` object (hence [`_add_var_to_model!`](@ref)), and referred to as `model[<key>]`, where `<key>` is the key of the variable. For example, `quote model[:w][1] * 5 - sum(model[:w]) end` or `:( model[:w][1] * 5 - sum(model[:w]) )`. Extra objective terms can be quotes or expressions. Using quotes is recommended because extra objective terms can be functions, as shown in the example.

!!! note
    Can only be used for `JuMP.@objective`. For problems which require a non-linear objective, use [`custom_nloptimiser!`](@ref) and add the extra terms to the custom function.

## Example

We can add extra objective terms to be minimised in the following way. We can use functions defined by `PortfolioOptimiser`, user defined functions, or anything else that is a valid `JuMP.@objective`. You can also use variable interpolation, but we don't use it in this example.

```julia
extra_obj_terms = [
    quote
        model[:w][1] * 5 - sum(model[:w])
    end,
    quote
       L2_reg(model[:w], 0.05)
    end
]
ef = MeanVar(tickers, mean_ret, cov_mtx)
_add_to_objective!.(ef.model, extra_obj_terms)
```

They can also be added to the constructor so they are added back automatically when the portfolio optimiser is refreshed.

```julia
extra_obj_terms = [
    quote
        model[:w][1] * 5 - sum(model[:w])
    end,
    quote
       L2_reg(model[:w], 0.05)
    end
]
ef = MeanVar(tickers, mean_ret, cov_mtx; extra_obj_terms = extra_obj_terms)
```

!!! note
    When adding elements to an objective in the constructor, they must be a collection (Tuple or vector), even if it's of length 1.
"""
function _add_to_objective!(model, expr)
    eval(quote
        model = $model
        lex = @expressions($model, $expr)[1]
    end)

    objFun = objective_function(model)
    objSense = objective_sense(model)
    add_to_expression!.(objFun, lex)
    @objective(model, objSense, objFun)
end

# """
# ```
# # Now try with a nonconvex objective from  Kolm et al (2014)
# objective = quote
#     function deviation_risk_parity(w::T...) where {T}
#         w = [i for i in w]
#         tmp = w .* (value.(cov_mtx) * w)
#         diff = tmp .- tmp'
#         return sum(diff .* diff)
#     end
# end

# obj_args = [:w]
# obj_params =
#     [(:(cov_mtx[i = 1:length(model[:w]), j = 1:length(model[:w])]), :(portfolio.cov_mtx))]
# ```
# obj_args are the arguments of the objective function
# obj_params array of tuples, first entry in tuple is the parametr, the second is the value of the parameter

# deviation_risk_parity the arguments are obj_args, and the part where it says value.(cov_mtx) is the parameter
# """
# function _add_nlparameter_to_model!(portfolio, param, param_val)
#     model = portfolio.model
#     key = param.args[1]
#     eval(quote
#         model = $model
#         @NLparameter($model, $param == 0)
#         portfolio = $portfolio
#         set_value.($key, $param_val)
#     end)
# end

"""
```
add_sector_constraint!(
    portfolio::AbstractPortfolioOptimiser,
    sector_map,
    sector_lower,
    sector_upper,
)
```

Constrains the sum of the weights of the tickers belonging to a sector to be between that sector's lower and upper bounds. For example, if we have two technology tickers, and we define the lower and upper bounds of the technology sector to be (0.1, 0.3), the sum of the weights of those tickers will be between (0.1, 0.3).

- `sector_map`: a dictionary mapping tickers to sectors where they key is the ticker and the value the sector, `Dict(<ticker> => <sector>)`.
- `sector_lower`: a dictionary mapping a sector to its lower bound, `Dict(<sector> => sector_lower_bound)`.
- `sector_upper`: a dictionary mapping a sector to its upper bound, `Dict(<sector> => sector_upper_bound)`.
"""
function add_sector_constraint!(
    portfolio::AbstractPortfolioOptimiser,
    sector_map,
    sector_lower,
    sector_upper,
)
    short =
        getfield.(
            getfield.(constraint_object.(portfolio.model[:lower_bounds]), :set),
            :lower,
        )
    if any(short .< 0)
        @warn(
            "Negative sector constraints (for shorting) may produce unreasonable results."
        )
    end

    tickers = portfolio.tickers
    model = portfolio.model
    w = model[:w]

    for (key, val) in sector_lower
        is_sector = findall(sector_map[ticker] == key for ticker in tickers)
        if !isempty(is_sector)
            sector_lower_key = Symbol(String(key) * "_lower")
            if haskey(model, sector_lower_key)
                delete(model, model[sector_lower_key])
                unregister(model, sector_lower_key)
            end
            model[sector_lower_key] =
                @constraint(model, sum(w[is_sector[i]] for i in 1:length(is_sector)) >= val)
        end
    end

    for (key, val) in sector_upper
        is_sector = findall(sector_map[ticker] == key for ticker in tickers)
        if !isempty(is_sector)
            sector_upper_key = Symbol(String(key) * "_upper")
            if haskey(model, sector_upper_key)
                delete(model, model[sector_upper_key])
                unregister(model, sector_upper_key)
            end
            model[sector_upper_key] =
                @constraint(model, sum(w[is_sector[i]] for i in 1:length(is_sector)) <= val)
        end
    end

    return nothing
end

"""
```
_function_vs_portfolio_val_warn(fval, pval, name)
```

Helper function for throwing generic warnings about inconsistent values between calls to optimiser functions and the values registered in the portfolio.
"""
function _function_vs_portfolio_val_warn(fval, pval, name)
    if fval != pval
        @warn(
            "The value of $(name): $fval, provided to the function does not match the one in the portfolio: $(pval). Using function value: $fval, instead."
        )
    end

    return nothing
end

"""
```
_val_compare_benchmark(val, op, benchmark, correction, name)
```

Helper function for throwing generic warnings about values out of the domain and ammending those values.
"""
function _val_compare_benchmark(val, op, benchmark, correction, name)
    if op(val, benchmark)
        @warn(
            "Value of $name, $val $(String(Symbol(op))) $benchmark. Correcting to $correction."
        )
        val = correction
    end
    return val
end

function _setup_and_optimise(model, optimiser, silent, optimiser_attributes = ())
    MOI.set(model, MOI.Silent(), silent)
    set_optimizer(model, optimiser)

    if isempty(optimiser_attributes)
        # Do nothing.
    elseif typeof(optimiser_attributes) <: Pair
        set_optimizer_attributes(model, optimiser_attributes)
    else
        set_optimizer_attributes(model, optimiser_attributes...)
    end

    optimize!(model)

    term_status = termination_status(model)
    if term_status ∉ (MOI.OPTIMAL, MOI.LOCALLY_SOLVED)
        @warn("The optimiser returned an infeasable solution with code: $term_status.")
    end
end