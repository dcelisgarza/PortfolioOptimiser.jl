abstract type AbstractDetone end
struct NoDetone <: AbstractDetone end
mutable struct Detone{T1} <: AbstractDetone
    mkt_comp::T1
end
function Detone(; mkt_comp::Integer = 1)
    @smart_assert(mkt_comp >= zero(mkt_comp))
    return Detone{typeof(mkt_comp)}(mkt_comp)
end
function Base.setproperty!(obj::Detone, sym::Symbol, val)
    if sym == :mkt_comp
        @smart_assert(val >= zero(val))
    end
    return setfield!(obj, sym, val)
end

export NoDetone, Detone
