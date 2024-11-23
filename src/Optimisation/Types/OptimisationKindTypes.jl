"""
```
abstract type AbstractOptimType end
```
"""
abstract type AbstractOptimType end

"""
```
abstract type OptimType <: AbstractOptimType end
```
"""
abstract type OptimType <: AbstractOptimType end

"""
```
abstract type HCOptimType <: AbstractOptimType end
```
"""
abstract type HCOptimType <: AbstractOptimType end

"""
```
struct Trad <: OptimType end
```
"""
struct Trad <: OptimType end

"""
```
struct RP <: OptimType end
```
"""
struct RP <: OptimType end

"""
```
abstract type RRPVersion end
```
"""
abstract type RRPVersion end

"""
```
struct BasicRRP <: RRPVersion end
```
"""
struct BasicRRP <: RRPVersion end

"""
```
struct RegRRP <: RRPVersion end
```
"""
struct RegRRP <: RRPVersion end

"""
```
@kwdef mutable struct RegPenRRP{T1 <: Real} <: RRPVersion
    penalty::T1 = 1.0
end
```
"""
mutable struct RegPenRRP{T1 <: Real} <: RRPVersion
    penalty::T1
end
function RegPenRRP(; penalty::Real = 1.0)
    return RegPenRRP(penalty)
end

"""
```
@kwdef mutable struct RRP <: OptimType
    version::RRPVersion = BasicRRP()
end
```
"""
mutable struct RRP <: OptimType
    version::RRPVersion
    formulation::VarianceFormulation
end
function RRP(; version::RRPVersion = BasicRRP(), formulation = SOC())
    return RRP(version, formulation)
end

"""
```
@kwdef mutable struct WC <: OptimType
    mu::WorstCaseSet = Box()
    cov::WorstCaseSet = Box()
end
```
"""
mutable struct WC <: OptimType
    mu::WorstCaseSet
    cov::WorstCaseSet
end
function WC(; mu::WorstCaseSet = Box(), cov::WorstCaseSet = Box())
    return WC(mu, cov)
end

"""
```
@kwdef mutable struct NOC{T1 <: Real, T2 <: AbstractVector{<:Real},
                          T3 <: AbstractVector{<:Real}, T4 <: AbstractVector{<:Real},
                          T5 <: AbstractVector{<:Real}, T6 <: AbstractVector{<:Real}} <:
                      OptimType
    type::Union{WC, Trad} = Trad()
    bins::T1 = 20.0
    w_opt::T2 = Vector{Float64}(undef, 0)
    w_min::T3 = Vector{Float64}(undef, 0)
    w_max::T4 = Vector{Float64}(undef, 0)
    w_min_ini::T5 = Vector{Float64}(undef, 0)
    w_max_ini::T6 = Vector{Float64}(undef, 0)
end
```
"""
mutable struct NOC{T1 <: Real, T2 <: AbstractVector{<:Real}, T3 <: AbstractVector{<:Real},
                   T4 <: AbstractVector{<:Real}, T5 <: AbstractVector{<:Real},
                   T6 <: AbstractVector{<:Real}} <: OptimType
    type::Union{WC, Trad}
    bins::T1
    w_opt::T2
    w_min::T3
    w_max::T4
    w_min_ini::T5
    w_max_ini::T6
end
function NOC(; type::Union{WC, Trad} = Trad(), bins::Real = 20.0,
             w_opt::AbstractVector{<:Real} = Vector{Float64}(undef, 0),
             w_min::AbstractVector{<:Real} = Vector{Float64}(undef, 0),
             w_max::AbstractVector{<:Real} = Vector{Float64}(undef, 0),
             w_min_ini::AbstractVector{<:Real} = Vector{Float64}(undef, 0),
             w_max_ini::AbstractVector{<:Real} = Vector{Float64}(undef, 0))
    return NOC{typeof(bins), typeof(w_opt), typeof(w_min), typeof(w_max), typeof(w_min_ini),
               typeof(w_max_ini)}(type, bins, w_opt, w_min, w_max, w_min_ini, w_max_ini)
end

"""
```
struct HRP <: HCOptimType end
```
"""
struct HRP <: HCOptimType end

"""
```
struct HERC <: HCOptimType end
```
"""
struct HERC <: HCOptimType end

"""
```
@kwdef mutable struct NCO <: HCOptimType
    opt_kwargs::NamedTuple = (;)
    opt_kwargs_o::NamedTuple = opt_kwargs
    port_kwargs::NamedTuple = (;)
    port_kwargs_o::NamedTuple = port_kwargs
    stat_kwargs_o::NamedTuple = (;)
end
```
"""
mutable struct NCO <: HCOptimType
    opt_kwargs::NamedTuple
    opt_kwargs_o::NamedTuple
    port_kwargs::NamedTuple
    port_kwargs_o::NamedTuple
    factor_kwargs::NamedTuple
    factor_kwargs_o::NamedTuple
    wc_kwargs::NamedTuple
    wc_kwargs_o::NamedTuple
    cluster_kwargs::NamedTuple
    cluster_kwargs_o::NamedTuple
    stat_kwargs_o::NamedTuple
end
function NCO(; opt_kwargs::NamedTuple = (;), opt_kwargs_o::NamedTuple = opt_kwargs,
             port_kwargs::NamedTuple = (;), port_kwargs_o::NamedTuple = port_kwargs,
             factor_kwargs::NamedTuple = (;), factor_kwargs_o::NamedTuple = factor_kwargs,
             wc_kwargs::NamedTuple = (;), wc_kwargs_o::NamedTuple = wc_kwargs,
             cluster_kwargs::NamedTuple = (;),
             cluster_kwargs_o::NamedTuple = cluster_kwargs, stat_kwargs_o::NamedTuple = (;))
    return NCO(opt_kwargs, opt_kwargs_o, port_kwargs, port_kwargs_o, factor_kwargs,
               factor_kwargs_o, wc_kwargs, wc_kwargs_o, cluster_kwargs, cluster_kwargs_o,
               stat_kwargs_o)
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

export Trad, RP, BasicRRP, RegRRP, RegPenRRP, RRP, WC, NOC, HRP, HERC, NCO
