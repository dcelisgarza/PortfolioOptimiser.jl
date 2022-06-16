"""
```
custom_optimiser!(
    portfolio::AbstractPortfolioOptimiser,
    obj,
    obj_params = ();
    initial_guess = nothing,
    optimiser = Ipopt.Optimizer,
    silent = true,
    optimiser_attributes = (),
)
```

Minimises user-provided objectives that are supported by `JuMP.@objective`.

- `portfolio`: any concrete subtype of `AbstractPortfolioOptimiser`.
- `obj`: objective function supported by `JuMP.@objective`.
- `obj_params`: vector or tuple of arguments of `obj`.
- `initial_guess`: initial guess for optimiser, if `nothing` defaults to uniform weights.
- `optimiser`: optimiser for solving optimisation problem.
- `silent`: if `false`, the optimiser prints to console.

## Example

This example defines a new function, identical to [`kelly_objective`](@ref), but you can use the predefined functions in `PortfolioOptimiser` as long as they are supported by `JuMP.@objective`.

```julia
function kelly_objective2(w, mean_ret, cov_mtx, k = 3)
    variance = dot(w, cov_mtx, w)
    objective = variance * 0.5 * k - dot(w, mean_ret)
    return objective
end

ef = EffMeanVar(tickers, expected_returns, cov_mtx)
obj_params = (ef.mean_ret, ef.cov_mtx, 1000)
custom_optimiser!(ef, kelly_objective2, obj_params)
```

Unlike, [`custom_nloptimiser!`](@ref), functions defined by `PortfolioOptimiser` don't have to be prepended with `PortfolioOptimiser`, they can be used as normal. Here we use [`kelly_objective`](@ref), which is provided by the package and as previously mentioned is identical to `kelly_objective2` of the previous example.

```julia
ef = EffMeanVar(tickers, expected_returns, cov_mtx)
obj_params = (ef.mean_ret, ef.cov_mtx, 1000)
custom_optimiser!(ef, kelly_objective, obj_params)
```

!!! note
    `obj_params` can be any variable that can be splatted. It is also optional, so objectives with no parameters are valid too.

!!! warning
    This minimises `obj`, so if you want to maximise a function, make it negative---for example, the portfolio return. Furthermore, this does not add extra objective terms, so they must be added to the definition of `obj`. We illustrate how we can use this to maximise the return subject to L2 regularisation (don't do this, 'tis a silly idea).

    ```julia
    function max_ret_l2_reg(w, mean_ret, γ = 1)
        μ = port_return(w, mean_ret)
        l2 = L2_reg(w, γ)

        # Minimise L2 regularisation, maximise the return by minimising its negative.
        return l2 - μ
    end

    ef = EffMeanVar(tickers, mean_ret, cov_mtx)

    γ = 1
    obj_params = (mean_ret, γ)

    custom_optimiser!(ef, max_ret_l2_reg, obj_params)
    ```
"""
function custom_optimiser!(
    portfolio::AbstractPortfolioOptimiser,
    obj,
    obj_params = ();
    initial_guess = nothing,
    optimiser = Ipopt.Optimizer,
    silent = true,
    optimiser_attributes = (),
)
    model = portfolio.model

    termination_status(model) != OPTIMIZE_NOT_CALLED && refresh_model!(portfolio)

    !haskey(model, :sum_w) && _make_weight_sum_constraint!(model, portfolio.market_neutral)

    w = model[:w]
    n = length(w)
    if !isnothing(initial_guess)
        @assert length(w) == length(initial_guess)
        set_start_value.(w, initial_guess)
    else
        set_start_value.(w, 1 / n)
    end

    @objective(model, Min, obj(w, obj_params...))

    _setup_and_optimise(model, optimiser, silent, optimiser_attributes)

    portfolio.weights .= value.(w)
    return nothing
end

"""
```
custom_nloptimiser!(
    portfolio::AbstractPortfolioOptimiser,
    obj,
    obj_params = (),
    extra_vars = ();
    initial_guess = nothing,
    optimiser = Ipopt.Optimizer,
    silent = true,
    optimiser_attributes = (),
)
```

Minimises user-provided nonlinear objectives that are supported by `JuMP.@NLobjective`.

- `portfolio`: any concrete subtype of `AbstractPortfolioOptimiser`.
- `obj`: objective function supported by `JuMP.@NLobjective`.
- `obj_params`: vector or tuple of arguments of `obj`.
- `extra_vars`: collection of tuples of extra variables for the nonlinear optimiser and their starting values, each element takes the form of `(variable, value)`. If `!isnothing(value)`, it sets/overrides the start value of `variable`, else it takes the default. This is important because the optimiser can fail if the start value causes a discontinuity in the `obj` function, or one of its derivatives. Furthermore, `variable` must be a variable registered to the model.
- `initial_guess`: initial guess for optimiser, if `nothing` defaults to unifrom weights.
- `optimiser`: optimiser for solving optimisation problem.
- `silent`: if `false`, the optimiser prints to console.

!!! warning
    As of JuMP 1.0, `JuMP.@NLobjective` only supports scalar arguments, as such, it's important to define the nonlinear objective to take in scalar arguments only. The example shows how this can be done.

## Example

In this first example we define a logarithmic barrier function identical to [`logarithmic_barrier`](@ref), then we have to define the scalar-only function that `JuMP.@NLobjective` can use.

```julia
# Import logarithmic_barrier so we can add the scalar method.
import PortfolioOptimiser: logarithmic_barrier

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

ef = EffMeanVar(tickers, mean_ret, cov_mtx)
obj_params = (ef.cov_mtx, 0.001)
custom_nloptimiser!(ef, logarithmic_barrier, obj_params)
```

We can also use objective functions that are already defined by `PortfolioOptimiser`. However, given how `JuMP.@NLobjective` requires user-defined functions to be registered, we need to prepend them with `PortfolioOptimiser` so that it can be recognised and registered by JuMP into the model.

```julia
# Import logarithmic_barrier so we can add the scalar method.
import PortfolioOptimiser: logarithmic_barrier

function logarithmic_barrier(w::T...) where {T}
    cov_mtx = obj_params[1]
    k = obj_params[2]
    w = [i for i in w]
    PortfolioOptimiser.logarithmic_barrier(w, cov_mtx, k)
end

ef = EffMeanVar(tickers, mean_ret, cov_mtx)
obj_params = [ef.cov_mtx, 0.001]
custom_nloptimiser!(ef, logarithmic_barrier, obj_params)

# This definition will error in custom_nloptimiser!
function logarithmic_barrier(w::T...) where {T}
    cov_mtx = obj_params[1]
    k = obj_params[2]
    w = [i for i in w]
    logarithmic_barrier(w, cov_mtx, k)
end

ef = EffMeanVar(tickers, mean_ret, cov_mtx)
obj_params = [ef.cov_mtx, 0.001]
custom_nloptimiser!(ef, logarithmic_barrier, obj_params)
```

!!! warning
    Note how in both cases, the parameters in the scalar-only function are not declared in the function signature. As such, they must be defined in terms of `obj_params`, or as literals, as the variable must be known to `custom_nloptimiser!`. If a variable name other than `obj_params` is used, the model will error becuase the variable will be unkown to the `custom_nloptimiser!`. For example, the following custom nonlinear objective function will not work because `obj_param` and `obj_parameter` do not exist inside `custom_nloptimiser!`.

    ```julia
    # Import logarithmic_barrier so we can add the scalar method.
    import PortfolioOptimiser: logarithmic_barrier

    function logarithmic_barrier(w::T...) where {T}
        cov_mtx = obj_param[1]  # Unkown variable, change to obj_params[1]
        k = obj_parameter[2]    # Unkown variable, change to obj_params[2]
        w = [i for i in w]
        PortfolioOptimiser.logarithmic_barrier(w, cov_mtx, k)
    end

    ef = EffMeanVar(tickers, mean_ret, cov_mtx)
    obj_params = [ef.cov_mtx, 0.001]
    custom_nloptimiser!(ef, logarithmic_barrier, obj_params)
    ```

!!! note
    `obj_params` can be any variable that can be splatted. It is also optional, so nonlinear objectives with no parameters are valid too.

!!! warning
    This minimises `obj`, so if you want to maximise a function, make it negative---for example, the sharpe ratio. Furthermore, this does not add extra objective terms, so they must be added to the definition of `obj`. We illustrate how we can use this to maximise the return subject to L2 regularisation (don't do this, 'tis a silly idea).

    ```julia
    function sharpe_l2_reg(w::T...) where {T}
        mean_ret = obj_params[1]
        cov_mtx = obj_params[2]
        rf = obj_params[3]
        γ = obj_params[4]

        w = [i for i in w]
        sr = PortfolioOptimiser.sharpe_ratio(w, mean_ret, cov_mtx, rf)
        l2 = PortfolioOptimiser.L2_reg(w, γ)

        # L2 reg has to be minimised. To maximise the sharpe ratio we minimise its negative.
        return l2 - sr
    end

    ef = EffMeanVar(tickers, mean_ret, cov_mtx)

    γ = 1
    obj_params = (mean_ret, cov_mtx, ef.rf, γ)

    custom_nloptimiser!(ef, sharpe_l2_reg, obj_params)
    ```
"""
function custom_nloptimiser!(
    portfolio::AbstractPortfolioOptimiser,
    obj,
    obj_params = (),
    extra_vars = ();
    initial_guess = nothing,
    optimiser = Ipopt.Optimizer,
    silent = true,
    optimiser_attributes = (),
)
    model = portfolio.model

    if termination_status(model) != OPTIMIZE_NOT_CALLED
        throw(
            ArgumentError(
                "Cannot deregister user defined functions from JuMP, model. Please make a new instance of portfolio.",
            ),
        )
    end

    !haskey(model, :sum_w) && _make_weight_sum_constraint!(model, portfolio.market_neutral)

    w = model[:w]
    n = length(w)
    if !isnothing(initial_guess)
        @assert length(w) == length(initial_guess)
        set_start_value.(w, initial_guess)
    else
        set_start_value.(w, 1 / n)
    end

    # Extra, non-weight variables for the model.
    if !isempty(extra_vars)
        for (val, start_val) in extra_vars
            if !isnothing(start_val)
                set_start_value.(val, start_val)
            end
            w = [w; val]
        end
    end
    m = length(w) - n

    register(model, :obj, n + m, obj, autodiff = true)
    @NLobjective(model, Min, obj(w...))

    _setup_and_optimise(model, optimiser, silent, optimiser_attributes)

    portfolio.weights .= value.(w)[1:n]
    return nothing
end