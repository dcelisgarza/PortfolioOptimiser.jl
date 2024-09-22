function finalise_weights(type::Any, port, weights, w_min, w_max, max_iter)
    stype = Symbol(type)
    weights = opt_weight_bounds(w_min, w_max, weights, max_iter)
    weights ./= sum(weights)
    port.optimal[stype] = DataFrame(; tickers = port.assets, weights = weights)
    return port.optimal[stype]
end
function finalise_weights(type::NCO, port, weights, w_min, w_max, max_iter)
    stype = Symbol(type)
    opt_kwargs = type.opt_kwargs
    opt_kwargs_o = type.opt_kwargs_o
    port_kwargs = type.port_kwargs
    port_kwargs_o = type.port_kwargs_o
    port.optimal[stype] = if all(iszero.(weights)) || any(.!isfinite.(weights))
        port.fail[:port] = DataFrame(; tickers = port.assets, weights = weights)
        DataFrame()
    elseif haskey(port_kwargs, :short) && port_kwargs.short ||
           haskey(port_kwargs_o, :short) && port_kwargs_o.short ||
           haskey(opt_kwargs, :class) && isa(opt_kwargs.class, Union{FM, FC}) ||
           haskey(opt_kwargs_o, :class) && isa(opt_kwargs_o.class, Union{FM, FC})
        weights = opt_weight_bounds(w_min, w_max, weights, max_iter)
        DataFrame(; tickers = port.assets, weights = weights)
    else
        weights = opt_weight_bounds(w_min, w_max, weights, max_iter)
        weights ./= sum(weights)
        DataFrame(; tickers = port.assets, weights = weights)
    end
    return port.optimal[stype]
end
