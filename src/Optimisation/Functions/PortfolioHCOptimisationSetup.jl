# Copywrite (c) 2025
# Author: Daniel Celis Garza <daniel.celis.garza@gmail.com>
# SPDX-License-Identifier: MIT

function w_limits(port::Portfolio, type::NCO)
    datatype = eltype(port.returns)
    (; internal, external) = type
    port_kwargs = internal.port_kwargs
    port_kwargs_o = external.port_kwargs
    class = internal.type.class
    class_o = external.type.class
    port_short_i = haskey(port_kwargs, :short) && port_kwargs.short
    port_short_o = haskey(port_kwargs_o, :short) && port_kwargs_o.short
    port_short = port.short
    short_flag = port_short_i || port_short_o || port_short
    fc_flag = isa(class, FC) || isa(class_o, FC)

    lo, hi = if short_flag && !fc_flag
        la = nothing
        ha = nothing
        lb = nothing
        hb = nothing
        lc = nothing
        hc = nothing

        if port_short_i
            if haskey(port_kwargs, :short_lb)
                la = port_kwargs.short_lb
            end
            if haskey(port_kwargs, :long_ub)
                ha = port_kwargs.long_ub
            end
        end

        if port_short_o
            if haskey(port_kwargs_o, :short_lb)
                lb = port_kwargs_o.short_lb
            end
            if haskey(port_kwargs_o, :long_ub)
                hb = port_kwargs_o.long_ub
            end
        end

        if port_short
            lc = port.short_lb
            hc = port.long_ub
        end

        if isnothing(la) && isnothing(lb) && isnothing(lc)
            la = lb = lc = -0.2 * one(datatype)
        elseif isnothing(la) && isnothing(lb)
            la = lb = lc
        elseif isnothing(la) && isnothing(lc)
            la = lc = lb
        elseif isnothing(lb) && isnothing(lc)
            lb = lc = la
        elseif isnothing(la)
            la = min(lb, lc)
        elseif isnothing(lb)
            lb = min(la, lc)
        elseif isnothing(lc)
            lc = min(la, lb)
        end

        if isnothing(ha) && isnothing(hb) && isnothing(hc)
            ha = hb = hc = one(datatype)
        elseif isnothing(ha) && isnothing(hb)
            ha = hb = hc
        elseif isnothing(ha) && isnothing(hc)
            ha = hc = hb
        elseif isnothing(hb) && isnothing(hc)
            hb = hc = ha
        elseif isnothing(ha)
            ha = max(hb, hc)
        elseif isnothing(hb)
            hb = max(ha, hc)
        elseif isnothing(hc)
            hc = max(ha, hb)
        end

        min(la, lb), max(ha, hb)
    elseif fc_flag
        -Inf, Inf
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
