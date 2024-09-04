abstract type AllocationMethod end

struct LP <: AllocationMethod end

@kwdef mutable struct Greedy{T1 <: Real} <: AllocationMethod
    rounding::T1 = 1.0
end

export LP, Greedy
