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
_map_bounds_to_constraints(bounds) 
```
Convert the bounds to JuMP constraints.
"""
function _create_weight_bounds(num_tickers, bounds)
    # If the bounds is an array of tuples, assume each is a bound.
    if typeof(bounds) <: AbstractVector &&
       length(bounds) == num_tickers &&
       eltype(bounds) <: Tuple &&
       length(bounds[1]) == 2
        bounds = Float64.(reshape(collect(Iterators.flatten(bounds)), 2, :))
        lower_bounds, upper_bounds = bounds
    elseif length(bounds) == 2
        lower, upper = Float64.(bounds)
        lower_bounds = fill(lower, num_tickers)
        upper_bounds = fill(upper, num_tickers)
    else
        throw(
            ArgumentError(
                "Weight bounds must either be a tuple of length 2, or a vector of length $num_tickers of tuples of length 2.",
            ),
        )
    end

    return lower_bounds, upper_bounds
end

function _make_weight_sum_constraint(model, market_neutral)
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
extra_vars = [:(c[1:3]), :k]
maxSharpe = EfficientFrontier(MaxSharpe(), tickers, mean_ret, cov_mtx, weight_bounds)
_add_var_to_model!.(maxSharpe.model, extra_vars)
```
Extra array variables have to be expressions. Extra scalar variables can be symbols or expressions.
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
extra_constraints = [:(model[:w][2] >= 2), :(sum(model[:w]) <= 5)]
maxSharpe = EfficientFrontier(MaxSharpe(), tickers, mean_ret, cov_mtx, weight_bounds)
_add_constraint_to_model!.(maxSharpe.model, [:const1, :const2], extra_constraints)
```
Extra constraints have to be expressions. Variables have to be registered in the model and they have to be explicitly written as `model[<variable_key>]`. This is because the variables in the constraints must be within scope when the code is automatically generated.
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
extra_obj_terms = [
    quote
        model[:w][1] * 5 - sum(model[:w])
    end, 
    quote
        model[:k]
    end
]
maxSharpe = EfficientFrontier(MaxSharpe(), tickers, mean_ret, cov_mtx, weight_bounds)
_add_to_objective!.(maxSharpe.model, extra_obj_terms)
```
Extra objective function terms have to be quotes because they are automatically spliced into code. Again the variables used have to be registered in the model, and `model[<variable_key>]` has to be explicitly typed to ensure the variables are within scope when the code is automatically generated. The terms have to be compliant with the type of objective function.
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

"""
```
# Now try with a nonconvex objective from  Kolm et al (2014)
objective = quote
    function deviation_risk_parity(w::T...) where {T}
        w = [i for i in w]
        tmp = w .* (value.(cov_mtx) * w)
        diff = tmp .- tmp'
        return sum(diff .* diff)
    end
end

obj_args = [:w]
obj_params =
    [(:(cov_mtx[i = 1:length(model[:w]), j = 1:length(model[:w])]), :(portfolio.cov_mtx))]
```
obj_args are the arguments of the objective function
obj_params array of tuples, first entry in tuple is the parametr, the second is the value of the parameter

deviation_risk_parity the arguments are obj_args, and the part where it says value.(cov_mtx) is the parameter
"""
function _add_nlparameter_to_model!(portfolio, param, param_val)
    model = portfolio.model
    key = param.args[1]
    eval(quote
        model = $model
        @NLparameter($model, $param == 0)
        portfolio = $portfolio
        set_value.($key, $param_val)
    end)
end

"""
```
function add_sector_constraint!(opt::PortTypes, sector_map, sector_lower, sector_upper)
```
Adds constraints on the sum of the weights of different tickers in the same group/sector. The portfolio will only be exposed to <= x % of each sector according to the weights.
```
sector_map = Dict(<ticker> => <sector>)
sector_lower = Dict(<sector> => sector_lower_bound)
sector_upper = Dict(<sector> => sector_upper_bound)
```
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