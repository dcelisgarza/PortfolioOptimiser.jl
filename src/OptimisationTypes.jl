
# ## Objective functions

abstract type ObjectiveFunction end
struct MinRisk <: ObjectiveFunction end
@kwdef mutable struct Utility{T1 <: Real} <: ObjectiveFunction
    l::T1 = 2.0
end
@kwdef mutable struct Sharpe{T1 <: Real} <: ObjectiveFunction
    rf::T1 = 0.0
end
struct MaxRet <: ObjectiveFunction end

# ## Portfolio types

abstract type AbstractPortType end
abstract type PortType <: AbstractPortType end
abstract type HCPortType <: AbstractPortType end

struct Trad <: PortType end

struct RP <: PortType end

abstract type RRPVersion end

struct BasicRRP <: RRPVersion end

struct RegRRP <: RRPVersion end

@kwdef mutable struct RegPenRRP{T1 <: Real} <: RRPVersion
    penalty::T1 = 1.0
end

@kwdef mutable struct RRP <: PortType
    version::RRPVersion = BasicRRP()
end

@kwdef mutable struct WC <: PortType
    mu::WorstCaseSet = Box()
    cov::WorstCaseSet = Box()
end

@kwdef mutable struct NOC{T1 <: Real, T2 <: AbstractVector{<:Real},
                          T3 <: AbstractVector{<:Real}, T4 <: AbstractVector{<:Real},
                          T5 <: AbstractVector{<:Real}, T6 <: AbstractVector{<:Real}} <:
                      PortType
    type::Union{WC, Trad} = Trad()
    bins::T1 = 20.0
    w_opt::T2 = Vector{Float64}(undef, 0)
    w_min::T3 = Vector{Float64}(undef, 0)
    w_max::T4 = Vector{Float64}(undef, 0)
    w_min_ini::T5 = Vector{Float64}(undef, 0)
    w_max_ini::T6 = Vector{Float64}(undef, 0)
end

struct HRP <: HCPortType end

struct HERC <: HCPortType end

@kwdef mutable struct NCO <: HCPortType
    opt_kwargs::NamedTuple = (;)
    opt_kwargs_o::NamedTuple = opt_kwargs
    port_kwargs::NamedTuple = (;)
    port_kwargs_o::NamedTuple = port_kwargs
    stat_kwargs_o::NamedTuple = (;)
end

for (op, name) ∈ zip((Trad, RP, RRP, WC, NOC, HRP, HERC, NCO),
                     ("Trad", "RP", "RRP", "WC", "NOC", "HRP", "HERC", "NCO"))
    eval(quote
             function Base.String(::$op)
                 return $name
             end
             function Base.Symbol(s::$op)
                 return Symbol($name)
             end
         end)
end

# ## Portfolio classes

abstract type PortClass end

struct Classic <: PortClass end

@kwdef mutable struct FC <: PortClass
    flag::Bool = true
end

mutable struct FM{T1 <: Integer} <: PortClass
    type::T1
end
function FM(; type::Integer = 1)
    @smart_assert(type ∈ (1, 2))
    return FM{typeof(type)}(type)
end

mutable struct BL{T1 <: Integer} <: PortClass
    type::T1
end
function BL(; type::Integer = 1)
    @smart_assert(type ∈ (1, 2))
    return BL{typeof(type)}(type)
end

mutable struct BLFM{T1 <: Integer} <: PortClass
    type::T1
end
function BLFM(; type::Integer = 1)
    @smart_assert(type ∈ (1, 2, 3))
    return BLFM{typeof(type)}(type)
end

# ## Asset allocation

abstract type AllocationMethod end

struct LP <: AllocationMethod end

@kwdef mutable struct Greedy{T1 <: Real} <: AllocationMethod
    rounding::T1 = 1.0
end

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
    penalty::T2 = 0.05
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

export MinRisk, Utility, Sharpe, MaxRet, Trad, RP, BasicRRP, RegRRP, RegPenRRP, RRP, WC,
       NOC, HRP, HERC, NCO, Classic, FC, FM, BL, BLFM, LP, Greedy, NoNtwk, SDP, IP, NoKelly,
       AKelly, EKelly, NoTracking, TrackWeight, TrackRet, NoTR, TR
