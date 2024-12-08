function _finaliser_type_constraint(type, model, weights, constr_scale)
    N = length(weights)
    w = model[:w]
    t = model[:t]
    if type == 1
        weights[iszero.(weights)] .= eps(eltype(weights))
        @constraint(model,
                    [constr_scale * t; constr_scale * (w ./ weights .- 1)] in
                    MOI.NormOneCone(N + 1))
    elseif type == 2
        weights[iszero.(weights)] .= eps(eltype(weights))
        @constraint(model,
                    [constr_scale * t; constr_scale * (w ./ weights .- 1)] in
                    SecondOrderCone())
    elseif type == 3
        @constraint(model,
                    [constr_scale * t; constr_scale * (w .- weights)] in
                    MOI.NormOneCone(N + 1))
    else
        @constraint(model,
                    [constr_scale * t; constr_scale * (w .- weights)] in SecondOrderCone())
    end
    return nothing
end
function opt_weight_bounds(port, w_min, w_max, weights, finaliser::JWF)
    if !(any(w_max .< weights) || any(w_min .> weights))
        return weights
    end
    constr_scale = port.constr_scale
    obj_scale = port.obj_scale
    solvers = port.solvers
    type = finaliser.type

    model = JuMP.Model()

    N = length(weights)
    budget = sum(weights)

    @variable(model, w[1:N])
    @constraint(model, sum(w) == budget)
    if all(weights .>= 0)
        @constraints(model, begin
                         constr_scale * w .>= 0
                         constr_scale * w .>= constr_scale * w_min
                         constr_scale * w .<= constr_scale * w_max
                     end)
    else
        short_budget = sum(weights[weights .< zero(eltype(weights))])

        @variables(model, begin
                       long_w[1:N] .>= 0
                       short_w[1:N] .<= 0
                   end)

        @constraints(model,
                     begin
                         constr_scale * long_w .<= constr_scale * w_max
                         constr_scale * short_w .>= constr_scale * w_min
                         constr_scale * w .<= constr_scale * long_w
                         constr_scale * w .>= constr_scale * short_w
                         constr_scale * sum(short_w) == constr_scale * short_budget
                         constr_scale * sum(long_w) ==
                         constr_scale * (budget - short_budget)
                     end)
    end
    @variable(model, t)
    _finaliser_type_constraint(type, model, weights, constr_scale)
    @objective(model, Min, obj_scale * t)

    success, solvers_tried = _optimise_JuMP_model(model, solvers)

    return if success
        value.(model[:w])
    else
        funcname = "$(fullname(PortfolioOptimiser)[1]).$(nameof(PortfolioOptimiser.opt_weight_bounds))"
        @warn("$funcname: model could not be optimised satisfactorily.\nMethod: $method\nSolvers: $solvers_tried.\nReverting to Heuristic method.")
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
