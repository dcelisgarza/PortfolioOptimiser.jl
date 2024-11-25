function opt_weight_bounds(w_min, w_max, weights, max_iter = 100)
    if !(any(w_max .< weights) || any(w_min .> weights))
        return weights
    end

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
function finalise_weights(type::Any, port, weights, w_min, w_max, max_iter)
    stype = Symbol(type)
    weights = opt_weight_bounds(w_min, w_max, weights, max_iter)
    weights ./= sum(weights)
    port.optimal[stype] = if any(.!isfinite.(weights)) || all(iszero.(weights))
        port.fail[:port] = DataFrame(; tickers = port.assets, weights = weights)
        DataFrame()
    else
        DataFrame(; tickers = port.assets, weights = weights)
    end
    return port.optimal[stype]
end
function finalise_weights(type::NCO, port, weights, w_min, w_max, max_iter)
    stype = Symbol(type)
    opt_kwargs = type.opt_kwargs
    opt_kwargs_o = type.opt_kwargs_o
    port_kwargs = type.port_kwargs
    port_kwargs_o = type.port_kwargs_o
    weights = opt_weight_bounds(w_min, w_max, weights, max_iter)
    port.optimal[stype] = if any(.!isfinite.(weights)) || all(iszero.(weights))
        port.fail[:port] = DataFrame(; tickers = port.assets, weights = weights)
        DataFrame()
    elseif haskey(port_kwargs, :short) && port_kwargs.short ||
           haskey(port_kwargs_o, :short) && port_kwargs_o.short ||
           haskey(opt_kwargs, :class) && isa(opt_kwargs.class, Union{FM, FC}) ||
           haskey(opt_kwargs_o, :class) && isa(opt_kwargs_o.class, Union{FM, FC})
        DataFrame(; tickers = port.assets, weights = weights)
    else
        weights ./= sum(weights)
        DataFrame(; tickers = port.assets, weights = weights)
    end
    return port.optimal[stype]
end
