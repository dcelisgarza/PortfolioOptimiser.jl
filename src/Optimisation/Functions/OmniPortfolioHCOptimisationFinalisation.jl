function opt_weight_bounds(port, w_min, w_max, weights, ::JWF)
    if !(any(w_max .< weights) || any(w_min .> weights))
        return weights
    end

    solvers = port.solvers
    model = JuMP.Model()

    N = length(weights)
    budget = sum(weights)

    @variable(model, w[1:N])
    @constraint(model, sum(w) == budget)
    if all(weights .>= 0)
        @constraints(model, begin
                         w .>= 0
                         w .>= w_min
                         w .<= w_max
                     end)
    else
        short_budget = sum(weights[weights .< zero(eltype(weights))])

        @variables(model, begin
                       long_w[1:N] .>= 0
                       short_w[1:N] .<= 0
                   end)

        @constraints(model, begin
                         long_w .<= w_max
                         short_w .>= w_min
                         w .<= long_w
                         w .>= short_w
                         sum(short_w) == short_budget
                         sum(long_w) == budget - short_budget
                     end)
    end
    @variable(model, t)
    @constraint(model, [t; (w ./ weights .- 1)] in SecondOrderCone())
    @objective(model, Min, t)

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
    (; internal, external) = type
    port_kwargs = internal.port_kwargs
    port_kwargs_o = external.port_kwargs
    class = internal.type.class
    class_o = external.type.class
    port_short_i = haskey(port_kwargs, :short) && port_kwargs.short
    port_short_o = haskey(port_kwargs_o, :short) && port_kwargs_o.short
    port_short = port.short

    weights = opt_weight_bounds(port, w_min, w_max, weights, finaliser)
    port.optimal[stype] = if any(.!isfinite.(weights)) || all(iszero.(weights))
        port.fail[:port] = DataFrame(; tickers = port.assets, weights = weights)
        DataFrame()
    elseif port_short_i ||
           port_short_o ||
           port_short ||
           isa(class, Union{FM, FC}) ||
           isa(class_o, Union{FM, FC})
        DataFrame(; tickers = port.assets, weights = weights)
    else
        DataFrame(; tickers = port.assets, weights = weights)
    end
    return port.optimal[stype]
end
