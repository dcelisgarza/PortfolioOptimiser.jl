# ## Constraints

# ### Network

abstract type NetworkMethods end
struct NoNtwk <: NetworkMethods end
@kwdef mutable struct SDP{T1 <: AbstractMatrix{<:Real}, T2 <: Real} <: NetworkMethods
    A::T1 = Matrix{Float64}(undef, 0, 0)
    penalty::T2 = 0.05
end
@kwdef mutable struct IP{T1 <: AbstractMatrix{<:Real}, T2 <: Real, T3 <: Real} <:
                      NetworkMethods
    A::T1 = Matrix{Float64}(undef, 0, 0)
    scale::T3 = 100_000.0
end

# ### Return

abstract type RetType end

struct NoKelly <: RetType end

@kwdef mutable struct AKelly <: RetType
    formulation::SDSquaredFormulation = SOCSD()
end

struct EKelly <: RetType end

# ### Tracking

abstract type TrackingErr end

struct NoTracking <: TrackingErr end

@kwdef mutable struct TrackWeight{T1 <: Real, T2 <: AbstractVector{<:Real}} <: TrackingErr
    err::T1 = 0.0
    w::T2 = Vector{Float64}(undef, 0)
end

@kwdef mutable struct TrackRet{T1 <: Real, T2 <: AbstractVector{<:Real}} <: TrackingErr
    err::T1 = 0.0
    w::T2 = Vector{Float64}(undef, 0)
end

# ### Turnover and rebalance

abstract type AbstractTR end

struct NoTR <: AbstractTR end

@kwdef mutable struct TR{T1 <: Union{<:Real, <:AbstractVector{<:Real}},
                         T2 <: AbstractVector{<:Real}} <: AbstractTR
    val::T1 = 0.0
    w::T2 = Vector{Float64}(undef, 0)
end

export NoNtwk, SDP, IP, NoKelly, AKelly, EKelly, NoTracking, TrackWeight, TrackRet, NoTR, TR
