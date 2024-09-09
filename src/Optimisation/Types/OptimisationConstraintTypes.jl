# ## Constraints

# ### Network

"""
```
abstract type NetworkMethods end
```
"""
abstract type NetworkMethods end

"""
```
struct NoNtwk <: NetworkMethods end
```
"""
struct NoNtwk <: NetworkMethods end

"""
```
@kwdef mutable struct SDP{T1 <: AbstractMatrix{<:Real}, T2 <: Real} <: NetworkMethods
    A::T1 = Matrix{Float64}(undef, 0, 0)
    penalty::T2 = 0.05
end
```
"""
mutable struct SDP{T1 <: AbstractMatrix{<:Real}, T2 <: Real} <: NetworkMethods
    A::T1
    penalty::T2
end
function SDP(; A::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
             penalty::Real = 0.05)
    return SDP{typeof(A), typeof(penalty)}(A, penalty)
end

"""
```
@kwdef mutable struct IP{T1 <: AbstractMatrix{<:Real}, T2 <: Real} <: NetworkMethods
    A::T1 = Matrix{Float64}(undef, 0, 0)
    scale::T2 = 100_000.0
end
```
"""
mutable struct IP{T1 <: AbstractMatrix{<:Real}, T2 <: Real} <: NetworkMethods
    A::T1
    scale::T2
end
function IP(; A::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
            scale::Real = 100_000.0)
    return IP{typeof(A), typeof(scale)}(A, typeof)
end

# ### Return

"""
```
abstract type RetType end
```
"""
abstract type RetType end

"""
```
struct NoKelly <: RetType end
```
"""
struct NoKelly <: RetType end

"""
```
@kwdef mutable struct AKelly <: RetType
    formulation::SDSquaredFormulation = SOCSD()
end
```
"""
mutable struct AKelly <: RetType
    formulation::SDSquaredFormulation
end
function AKelly(; formulation::SDSquaredFormulation = SOCSD())
    return AKelly(formulation)
end

struct EKelly <: RetType end

# ### Tracking

"""
```
abstract type TrackingErr end
```
"""
abstract type TrackingErr end

"""
```
struct NoTracking <: TrackingErr end
```
"""
struct NoTracking <: TrackingErr end

"""
```
@kwdef mutable struct TrackWeight{T1 <: Real, T2 <: AbstractVector{<:Real}} <: TrackingErr
    err::T1 = 0.0
    w::T2 = Vector{Float64}(undef, 0)
end
```
"""
mutable struct TrackWeight{T1 <: Real, T2 <: AbstractVector{<:Real}} <: TrackingErr
    err::T1
    w::T2
end
function TrackWeight(; err::Real = 0.0,
                     w::AbstractVector{<:Real} = Vector{Float64}(undef, 0))
    return TrackWeight{typeof(err), typeof(w)}(err, w)
end

"""
```
@kwdef mutable struct TrackRet{T1 <: Real, T2 <: AbstractVector{<:Real}} <: TrackingErr
    err::T1 = 0.0
    w::T2 = Vector{Float64}(undef, 0)
end
```
"""
mutable struct TrackRet{T1 <: Real, T2 <: AbstractVector{<:Real}} <: TrackingErr
    err::T1
    w::T2
end
function TrackRet(; err::Real = 0.0, w::AbstractVector{<:Real} = Vector{Float64}(undef, 0))
    return TrackRet{typeof(err), typeof(w)}(err, w)
end

# ### Turnover and rebalance

"""
```
abstract type AbstractTR end
```
"""
abstract type AbstractTR end

"""
```
struct NoTR <: AbstractTR end
```
"""
struct NoTR <: AbstractTR end

"""
```
@kwdef mutable struct TR{T1 <: Union{<:Real, <:AbstractVector{<:Real}},
                         T2 <: AbstractVector{<:Real}} <: AbstractTR
    val::T1 = 0.0
    w::T2 = Vector{Float64}(undef, 0)
end
```
"""
mutable struct TR{T1 <: Union{<:Real, <:AbstractVector{<:Real}},
                  T2 <: AbstractVector{<:Real}} <: AbstractTR
    val::T1
    w::T2
end
function TR(; val::Union{<:Real, <:AbstractVector{<:Real}} = 0.0,
            w::AbstractVector{<:Real} = Vector{Float64}(undef, 0))
    return TR{typeof(val), typeof(w)}(val, w)
end

export NoNtwk, SDP, IP, NoKelly, AKelly, EKelly, NoTracking, TrackWeight, TrackRet, NoTR, TR
