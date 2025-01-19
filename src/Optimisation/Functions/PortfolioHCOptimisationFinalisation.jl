function finaliser_constraint_obj(::ROJWF, model, weights, scale_constr, scale_obj)
    N = length(weights)
    w = model[:w]
    @variable(model, t)
    weights[iszero.(weights)] .= eps(eltype(weights))
    @constraint(model,
                [scale_constr * t; scale_constr * (w ./ weights .- one(eltype(weights)))] in
                MOI.NormOneCone(N + 1))
    @objective(model, Min, scale_obj * t)

    return nothing
end
function finaliser_constraint_obj(::RSJWF, model, weights, scale_constr, scale_obj)
    w = model[:w]
    @variable(model, t)
    weights[iszero.(weights)] .= eps(eltype(weights))
    @constraint(model,
                [scale_constr * t; scale_constr * (w ./ weights .- one(eltype(weights)))] in
                SecondOrderCone())
    @objective(model, Min, scale_obj * t)

    return nothing
end
function finaliser_constraint_obj(::AOJWF, model, weights, scale_constr, scale_obj)
    N = length(weights)
    w = model[:w]
    @variable(model, t)
    @constraint(model,
                [scale_constr * t; scale_constr * (w .- weights)] in MOI.NormOneCone(N + 1))
    @objective(model, Min, scale_obj * t)

    return nothing
end
function finaliser_constraint_obj(::ASJWF, model, weights, scale_constr, scale_obj)
    w = model[:w]
    @variable(model, t)
    @constraint(model,
                [scale_constr * t; scale_constr * (w .- weights)] in SecondOrderCone())
    @objective(model, Min, scale_obj * t)

    return nothing
end
function opt_weight_bounds(port, w_min, w_max, weights, finaliser::JWF)
    if !(any(w_max .< weights) || any(w_min .> weights))
        return weights
    end
    scale_constr = port.scale_constr
    scale_obj = port.scale_obj
    solvers = port.solvers
    version = finaliser.version

    model = JuMP.Model()

    N = length(weights)
    budget = sum(weights)

    @variable(model, w[1:N])
    @constraint(model, sum(w) == budget)
    if all(weights .>= 0)
        @constraints(model, begin
                         scale_constr * w .>= 0
                         scale_constr * w .>= scale_constr * w_min
                         scale_constr * w .<= scale_constr * w_max
                     end)
    else
        short_budget = sum(weights[weights .< zero(eltype(weights))])

        @variables(model, begin
                       long_w[1:N] .>= 0
                       short_w[1:N] .<= 0
                   end)

        @constraints(model,
                     begin
                         scale_constr * long_w .<= scale_constr * w_max
                         scale_constr * short_w .>= scale_constr * w_min
                         scale_constr * w .<= scale_constr * long_w
                         scale_constr * w .>= scale_constr * short_w
                         scale_constr * sum(short_w) == scale_constr * short_budget
                         scale_constr * sum(long_w) ==
                         scale_constr * (budget - short_budget)
                     end)
    end
    finaliser_constraint_obj(version, model, weights, scale_constr, scale_obj)

    success, solvers_tried = optimise_JuMP_model(model, solvers)

    return if success
        value.(model[:w])
    else
        funcname = "$(fullname(PortfolioOptimiser)[1]).$(nameof(PortfolioOptimiser.opt_weight_bounds))"
        @warn("$funcname: model could not be optimised satisfactorily.\nVersion: $version\nSolvers: $solvers_tried.\nReverting to Heuristic type.")
        opt_weight_bounds(nothing, w_min, w_max, weights, HWF())
    end
end
function opt_weight_bounds(::Any, w_min, w_max, weights, finaliser::HWF)
    if !(any(w_max .< weights) || any(w_min .> weights))
        return weights
    end

    max_iter = finaliser.max_iter

    s1 = sum(weights)
    for _ ∈ 1:max_iter
        if !(any(w_max .< weights) || any(w_min .> weights))
            break
        end

        old_w = copy(weights)
        weights = max.(min.(weights, w_max), w_min)

        idx = weights .< w_max .&& weights .> w_min
        w_add = sum(max.(old_w - w_max, 0.0))
        w_sub = sum(min.(old_w - w_min, 0.0))
        delta = w_add + w_sub

        if delta != 0
            weights[idx] += delta * weights[idx] / sum(weights[idx])
        end
        weights .*= s1 / sum(weights)
    end

    return weights
end
function finalise_weights(type::HCOptimType, port, weights, w_min, w_max, finaliser)
    stype = Symbol(type)
    weights = opt_weight_bounds(port, w_min, w_max, weights, finaliser)
    weights ./= sum(weights)
    port.optimal[stype] = if any(.!isfinite.(weights)) || all(iszero.(weights))
        port.fail[:port] = DataFrame(; tickers = port.assets, weights = weights)
        DataFrame()
    else
        DataFrame(; tickers = port.assets, weights = weights)
    end
    return port.optimal[stype]
end
function finalise_weights(type::NCO, port, weights, w_min, w_max, finaliser)
    stype = Symbol(type)
    weights = opt_weight_bounds(port, w_min, w_max, weights, finaliser)
    port.optimal[stype] = if any(.!isfinite.(weights)) || all(iszero.(weights))
        port.fail[:port] = DataFrame(; tickers = port.assets, weights = weights)
        DataFrame()
    else
        DataFrame(; tickers = port.assets, weights = weights)
    end
    return port.optimal[stype]
end
