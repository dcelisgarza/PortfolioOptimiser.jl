abstract type AllocationType end

"""
```
struct LP <: AllocationType end
```
"""
struct LP <: AllocationType end

@kwdef mutable struct Greedy{T1 <: Real} <: AllocationType
    rounding::T1 = 1.0
end

export LP, Greedy
