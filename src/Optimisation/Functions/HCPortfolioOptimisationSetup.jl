function w_limits(type::NCO, datatype = Float64)
    opt_kwargs = type.opt_kwargs
    opt_kwargs_o = type.opt_kwargs_o
    port_kwargs = type.port_kwargs
    port_kwargs_o = type.port_kwargs_o
    lo, hi = if haskey(port_kwargs, :short) && port_kwargs.short ||
                haskey(port_kwargs_o, :short) && port_kwargs_o.short
        la = nothing
        ha = nothing
        lb = nothing
        hb = nothing

        if haskey(port_kwargs, :short) && port_kwargs.short
            if haskey(port_kwargs, :short_u)
                la = port_kwargs.short_u
            end
            if haskey(port_kwargs, :long_u)
                ha = port_kwargs.long_u
            end
        end

        if haskey(port_kwargs_o, :short) && port_kwargs_o.short
            if haskey(port_kwargs_o, :short_u)
                lb = port_kwargs_o.short_u
            end
            if haskey(port_kwargs_o, :long_u)
                hb = port_kwargs_o.long_u
            end
        end

        if isnothing(la) && isnothing(lb)
            la = lb = 0.2 * one(datatype)
        elseif isnothing(la)
            la = lb
        elseif isnothing(lb)
            lb = la
        end

        if isnothing(ha) && isnothing(hb)
            ha = hb = one(datatype)
        elseif isnothing(ha)
            ha = hb
        elseif isnothing(hb)
            hb = ha
        end

        -max(la, lb), max(ha, hb)
    elseif haskey(opt_kwargs, :class) && isa(opt_kwargs.class, Union{FM, FC}) ||
           haskey(opt_kwargs_o, :class) && isa(opt_kwargs_o.class, Union{FM, FC})
        -Inf, Inf
    else
        zero(datatype), one(datatype)
    end

    return lo, hi
end
