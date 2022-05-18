"""
```
custom_optimiser!(
    portfolio::AbstractPortfolioOptimiser,
    obj,
    obj_params = (),
    initial_guess = nothing,
    optimiser = Ipopt.Optimizer,
    silent = true,
)
```

Users provide their own objectives that are supported by `JuMP.@objective`.

- `portfolio`: any concrete subtype of `AbstractPortfolioOptimiser`.
- `obj`: objective function supported by `JuMP.@objective`.
- `obj_params`: vector or tuple of arguments of `obj`.
- `initial_guess`: initial guess for optimiser, if `nothing` lets the optimiser decide.
- `optimiser`: optimiser for solving optimisation problem.
- `silent`: if `false`, the optimiser prints to console.

## Example

This example defines a new function, identical to [`kelly_objective`](@ref), but you can use the predefined functions in `PortfolioOptimiser` as long as they are supported by `JuMP.@objective`.

```
function kelly_objective2(w, mean_ret, cov_mtx, k = 3)
    variance = dot(w, cov_mtx, w)
    objective = variance * 0.5 * k - dot(w, mean_ret)
    return objective
end

ef = EfficientFrontier(tickers, expected_returns, cov_mtx)
obj_params = (ef.mean_ret, ef.cov_mtx, 1000)
custom_optimiser!(ef, kelly_objective2, obj_params)
```

Unlike, [`custom_nloptimiser!`](@ref), functions defined by `PortfolioOptimiser` don't have to be prepended with `PortfolioOptimiser`, they can be used as normal. Here we use [`kelly_objective`](@ref), which is provided by the package and as previously mentioned is identical to `kelly_objective2` of the previous example.

```
ef = EfficientFrontier(tickers, expected_returns, cov_mtx)
obj_params = (ef.mean_ret, ef.cov_mtx, 1000)
custom_optimiser!(ef, kelly_objective, obj_params)
```

!!! note
    `obj_params` can be any variable that can be spatted. It is also optional, so objectives with no parameters are valid too.
"""
function custom_optimiser!(
    portfolio::AbstractPortfolioOptimiser,
    obj,
    obj_params = (),
    initial_guess = nothing,
    optimiser = Ipopt.Optimizer,
    silent = true,
)
    model = portfolio.model

    termination_status(model) != OPTIMIZE_NOT_CALLED && refresh_model!(portfolio)

    !haskey(model, :sum_w) && _make_weight_sum_constraint(model, portfolio.market_neutral)

    w = model[:w]
    if !isnothing(initial_guess)
        @assert length(w) == length(initial_guess)
        set_start_value.(w, initial_guess)
    end

    @objective(model, Min, obj(w, obj_params...))

    MOI.set(model, MOI.Silent(), silent)
    set_optimizer(model, optimiser)
    optimize!(model)

    portfolio.weights .= value.(w)
    return nothing
end

"""
```
custom_nloptimiser!(
    portfolio::AbstractPortfolioOptimiser,
    obj,
    obj_params = (),
    initial_guess = nothing,
    optimiser = Ipopt.Optimizer,
    silent = true,
)
```

Users provide their own objectives that are supported by `JuMP.@NLobjective`.

- `portfolio`: any concrete subtype of `AbstractPortfolioOptimiser`.
- `obj`: objective function supported by `JuMP.@NLobjective`.
- `obj_params`: vector or tuple of arguments of `obj`.
- `initial_guess`: initial guess for optimiser, if `nothing` lets the optimiser decide.
- `optimiser`: optimiser for solving optimisation problem.
- `silent`: if `false`, the optimiser prints to console.

!!! warning
    As of JuMP 1.0, `JuMP.@NLobjective` only supports scalar arguments, as such, it's important to define the nonlinear objective to take in scalar arguments only. The example shows how this can be done.

## Example

In this first example we define a logarithmic barrier function identical to [`logarithmic_barrier`](@ref), then we have to define the scalar-only function that `JuMP.@NLobjective` can use.

```
function logarithmic_barrier2(w, cov_mtx, k = 0.1)
    # Add eps() to avoid log(0) divergence.
    log_sum = sum(log.(w .+ eps()))
    var = dot(w, cov_mtx, w)
    return var - k * log_sum
end

function logarithmic_barrier(w::T...) where {T}
    cov_mtx = obj_params[1]
    k = obj_params[2]
    w = [i for i in w]
    logarithmic_barrier2(w, cov_mtx, k)
end

ef = EfficientFrontier(names(df)[2:end], mu, S)
obj_params = (ef.cov_mtx, 0.001)
custom_nloptimiser!(ef, logarithmic_barrier, obj_params)
```

We can also use objective functions that are already defined by `PortfolioOptimiser`. However, given how `JuMP.@NLobjective` requires user-defined functions to be registered, we need to prepend them with `PortfolioOptimiser` so that it can be recognised and registered by JuMP into the model.

```
function logarithmic_barrier(w::T...) where {T}
    cov_mtx = obj_params[1]
    k = obj_params[2]
    w = [i for i in w]
    PortfolioOptimiser.logarithmic_barrier(w, cov_mtx, k)
end

ef = EfficientFrontier(names(df)[2:end], mu, S)
obj_params = [ef.cov_mtx, 0.001]
custom_nloptimiser!(ef, logarithmic_barrier, obj_params)

# This definition will error in custom_nloptimiser!
function logarithmic_barrier(w::T...) where {T}
    cov_mtx = obj_params[1]
    k = obj_params[2]
    w = [i for i in w]
    logarithmic_barrier(w, cov_mtx, k)
end

ef = EfficientFrontier(names(df)[2:end], mu, S)
obj_params = [ef.cov_mtx, 0.001]
custom_nloptimiser!(ef, logarithmic_barrier, obj_params)
```

!!! warning
    Note how in both cases, the parameters in the scalar-only function are not declared in the function signature. As such, they must be defined in terms of `obj_params`, or as literals, as the variable must be known to `custom_nloptimiser!`. If a variable name other than `obj_params` is used, the model will error becuase the variable will be unkown to the `custom_nloptimiser!`. For example, the following custom nonlinear objective function will not work because `obj_param` and `obj_parameter` do not exist inside `custom_nloptimiser!`.

    ```
    function logarithmic_barrier(w::T...) where {T}
        cov_mtx = obj_param[1]  # Unkown variable, change to obj_params[1]
        k = obj_parameter[2]    # Unkown variable, change to obj_params[2]
        w = [i for i in w]
        PortfolioOptimiser.logarithmic_barrier(w, cov_mtx, k)
    end

    ef = EfficientFrontier(names(df)[2:end], mu, S)
    obj_params = [ef.cov_mtx, 0.001]
    custom_nloptimiser!(ef, logarithmic_barrier, obj_params)
    ```

!!! note
    `obj_params` can be any variable that can be spatted. It is also optional, so nonlinear objectives with no parameters are valid too.
"""
function custom_nloptimiser!(
    portfolio::AbstractPortfolioOptimiser,
    obj,
    obj_params = (),
    initial_guess = nothing,
    optimiser = Ipopt.Optimizer,
    silent = true,
)
    model = portfolio.model

    if termination_status(model) != OPTIMIZE_NOT_CALLED
        throw(
            ArgumentError(
                "Cannot deregister user defined functions from JuMP, model. Please make a new instance of portfolio.",
            ),
        )
    end

    !haskey(model, :sum_w) && _make_weight_sum_constraint(model, portfolio.market_neutral)

    w = model[:w]
    n = length(w)
    if !isnothing(initial_guess)
        @assert length(w) == length(initial_guess)
        set_start_value.(w, initial_guess)
    end

    register(model, :obj, n, obj, autodiff = true)
    @NLobjective(model, Min, obj(w...))

    MOI.set(model, MOI.Silent(), silent)
    set_optimizer(model, optimiser)
    optimize!(model)

    portfolio.weights .= value.(w)
    return nothing
end