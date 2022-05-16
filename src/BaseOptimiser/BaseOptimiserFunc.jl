function custom_optimiser!(
    portfolio::AbstractPortfolioOptimiser,
    obj,
    obj_params = [],
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

function custom_nloptimiser!(
    portfolio::AbstractPortfolioOptimiser,
    obj,
    obj_params = [], #! This is used in the generated function that is obj. Do not delete.
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