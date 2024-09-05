function w_limits(type::NCO, datatype = Float64)
    port_kwargs = type.port_kwargs
    port_kwargs_o = type.port_kwargs_o
    lo, hi = if isa(type, NCO) && (haskey(port_kwargs, :short) && port_kwargs.short ||
                                   haskey(port_kwargs_o, :short) && port_kwargs_o.short)
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
    else
        zero(datatype), one(datatype)
    end

    return lo, hi
end
function w_limits(::Any, datatype = Float64)
    return zero(datatype), one(datatype)
end
function set_hc_weights(w_min, w_max, N, lo = 0.0, hi = 1.0)
    lower_bound = if isa(w_min, AbstractVector) && isempty(w_min)
        zeros(N)
    elseif isa(w_min, AbstractVector) && !isempty(w_min)
        max.(lo, w_min)
    else
        fill(max(lo, w_min), N)
    end

    upper_bound = if isa(w_max, AbstractVector) && isempty(w_max)
        ones(N)
    elseif isa(w_max, AbstractVector) && !isempty(w_max)
        min.(hi, w_max)
    else
        fill(min(hi, w_max), N)
    end

    @smart_assert(all(upper_bound .>= lower_bound))

    return lower_bound, upper_bound
end
