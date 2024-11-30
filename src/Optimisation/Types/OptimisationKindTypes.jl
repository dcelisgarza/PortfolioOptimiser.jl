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
mutable struct Trad{T1, T2} <: OptimType
    rm::Union{AbstractVector, <:RiskMeasure}
    obj::ObjectiveFunction
    kelly::RetType
    class::PortClass
    w_ini::T1
    custom_constr::CustomConstraint
    custom_obj::CustomObjective
    ohf::T2
    str_names::Bool
end
function Trad(; rm::Union{AbstractVector, <:RiskMeasure} = Variance(),
              obj::ObjectiveFunction = MinRisk(), kelly::RetType = NoKelly(),
              class::PortClass = Classic(),
              w_ini::AbstractVector = Vector{Float64}(undef, 0),
              custom_constr::CustomConstraint = NoCustomConstraint(),
              custom_obj::CustomObjective = NoCustomObjective(), ohf::Real = 1.0,
              str_names::Bool = false)
    return Trad{typeof(w_ini), typeof(ohf)}(rm, obj, kelly, class, w_ini, custom_constr,
                                            custom_obj, ohf, str_names)
end

mutable struct DRCVaR{T1, T2, T3, T4} <: OptimType
    l::T1
    alpha::T2
    r::T3
    class::PortClass
    w_ini::T4
    custom_constr::CustomConstraint
    custom_obj::CustomObjective
    str_names::Bool
end
function DRCVaR(; l::Real = 1.0, alpha::Real = 0.05, r::Real = 0.02,
                class::PortClass = Classic(),
                w_ini::AbstractVector = Vector{Float64}(undef, 0),
                custom_constr::CustomConstraint = NoCustomConstraint(),
                custom_obj::CustomObjective = NoCustomObjective(), str_names::Bool = false)
    return DRCVaR{typeof(l), typeof(alpha), typeof(r), typeof(w_ini)}(l, alpha, r, class,
                                                                      w_ini, custom_constr,
                                                                      custom_obj, str_names)
end

"""
```
struct RP <: OptimType end
```
"""
mutable struct RP{T1} <: OptimType
    rm::Union{AbstractVector, <:RiskMeasure}
    kelly::RetType
    class::PortClass
    w_ini::T1
    custom_constr::CustomConstraint
    custom_obj::CustomObjective
    str_names::Bool
end
function RP(; rm::Union{AbstractVector, <:RiskMeasure} = Variance(),
            kelly::RetType = NoKelly(), class::PortClass = Classic(),
            w_ini::AbstractVector = Vector{Float64}(undef, 0),
            custom_constr::CustomConstraint = NoCustomConstraint(),
            custom_obj::CustomObjective = NoCustomObjective(), str_names::Bool = false)
    return RP{typeof(w_ini)}(rm, kelly, class, w_ini, custom_constr, custom_obj, str_names)
end

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
mutable struct RegPenRRP{T1} <: RRPVersion
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
mutable struct RRP{T1} <: OptimType
    version::RRPVersion
    kelly::RetType
    class::PortClass
    w_ini::T1
    custom_constr::CustomConstraint
    custom_obj::CustomObjective
    str_names::Bool
end
function RRP(; version::RRPVersion = BasicRRP(), kelly::RetType = NoKelly(),
             class::PortClass = Classic(),
             w_ini::AbstractVector = Vector{Float64}(undef, 0),
             custom_constr::CustomConstraint = NoCustomConstraint(),
             custom_obj::CustomObjective = NoCustomObjective(), str_names::Bool = false,)
    return RRP{typeof(w_ini)}(version, kelly, class, w_ini, custom_constr, custom_obj,
                              str_names)
end
function Base.getproperty(obj::RRP, sym::Symbol)
    return if sym == :rm
        nothing
    else
        getfield(obj, sym)
    end
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
mutable struct NOC{T1, T2, T3, T4, T5, T6, T7, T8} <: OptimType
    flag::Bool
    bins::T1
    w_opt::T2
    w_min::T3
    w_max::T4
    w_min_ini::T5
    w_max_ini::T6
    rm::Union{AbstractVector, <:RiskMeasure}
    obj::ObjectiveFunction
    kelly::RetType
    class::PortClass
    w_ini::T7
    custom_constr::CustomConstraint
    custom_obj::CustomObjective
    ohf::T8
    str_names::Bool
end
function NOC(; flag::Bool = true, bins::Real = 20.0,
             w_opt::AbstractVector{<:Real} = Vector{Float64}(undef, 0),
             w_min::AbstractVector{<:Real} = Vector{Float64}(undef, 0),
             w_max::AbstractVector{<:Real} = Vector{Float64}(undef, 0),
             w_min_ini::AbstractVector{<:Real} = Vector{Float64}(undef, 0),
             w_max_ini::AbstractVector{<:Real} = Vector{Float64}(undef, 0),
             rm::Union{AbstractVector, <:RiskMeasure} = Variance(),
             obj::ObjectiveFunction = MinRisk(), kelly::RetType = NoKelly(),
             class::PortClass = Classic(),
             w_ini::AbstractVector{<:Real} = Vector{Float64}(undef, 0),
             custom_constr::CustomConstraint = NoCustomConstraint(),
             custom_obj::CustomObjective = NoCustomObjective(), ohf::Real = 1.0,
             str_names::Bool = false)
    return NOC{typeof(bins), typeof(w_opt), typeof(w_min), typeof(w_max), typeof(w_min_ini),
               typeof(w_max_ini), typeof(w_ini), typeof(ohf)}(flag, bins, w_opt, w_min,
                                                              w_max, w_min_ini, w_max_ini,
                                                              rm, obj, kelly, class, w_ini,
                                                              custom_constr, custom_obj,
                                                              ohf, str_names)
end

abstract type HCOptWeightFinaliser end
mutable struct HWF{T1} <: HCOptWeightFinaliser
    max_iter::T1
end
function HWF(; max_iter::Integer = 100)
    return HWF{typeof(max_iter)}(max_iter)
end
struct JWF <: HCOptWeightFinaliser end

"""
```
struct HRP <: HCOptimType end
```
"""
mutable struct HRP <: HCOptimType
    rm::Union{AbstractVector, <:AbstractRiskMeasure}
    class::PortClass
    finaliser::HCOptWeightFinaliser
end
function HRP(; rm::Union{AbstractVector, <:AbstractRiskMeasure} = Variance(),
             class::PortClass = Classic(), finaliser::HCOptWeightFinaliser = HWF())
    return HRP(rm, class, finaliser)
end

mutable struct SchurParams{T1, T2, T3, T4}
    rm::RMSigma
    gamma::T1
    prop_coef::T2
    tol::T3
    max_iter::T4
end
function SchurParams(; rm = Variance(;), gamma::Real = 0.5, prop_coef::Real = 0.5,
                     tol::Real = 1e-2, max_iter::Integer = 10)
    @smart_assert(zero(gamma) <= gamma <= one(gamma))
    return SchurParams{typeof(gamma), typeof(prop_coef), typeof(tol), typeof(max_iter)}(rm,
                                                                                        gamma,
                                                                                        prop_coef,
                                                                                        tol,
                                                                                        max_iter)
end
mutable struct SchurHRP <: HCOptimType
    params::Union{AbstractVector, <:SchurParams}
    class::PortClass
    finaliser::HCOptWeightFinaliser
end
function SchurHRP(; params::Union{AbstractVector, <:SchurParams} = SchurParams(),
                  class::PortClass = Classic(), finaliser::HCOptWeightFinaliser = HWF())
    return SchurHRP(params, class, finaliser)
end

"""
```
struct HERC <: HCOptimType end
```
"""
mutable struct HERC <: HCOptimType
    rm::Union{AbstractVector, <:AbstractRiskMeasure}
    rm_o::Union{AbstractVector, <:AbstractRiskMeasure}
    class::PortClass
    class_o::PortClass
    finaliser::HCOptWeightFinaliser
end
function HERC(; rm::Union{AbstractVector, <:AbstractRiskMeasure} = Variance(),
              rm_o::Union{AbstractVector, <:AbstractRiskMeasure} = rm,
              class::PortClass = Classic(), class_o::PortClass = class,
              finaliser::HCOptWeightFinaliser = HWF())
    return HERC(rm, rm_o, class, class_o, finaliser)
end

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

mutable struct NCOArgs
    type::AbstractOptimType
    port_kwargs::NamedTuple
    stats_kwargs::NamedTuple
    wc_kwargs::NamedTuple
    factor_kwargs::NamedTuple
    cluster_kwargs::NamedTuple
end
function NCOArgs(; type::AbstractOptimType = Trad(), port_kwargs::NamedTuple = (;),
                 stats_kwargs::NamedTuple = (;), wc_kwargs::NamedTuple = (;),
                 factor_kwargs::NamedTuple = (;), cluster_kwargs::NamedTuple = (;))
    return NCOArgs(type, port_kwargs, stats_kwargs, wc_kwargs, factor_kwargs,
                   cluster_kwargs)
end
mutable struct NCO <: HCOptimType
    internal::NCOArgs
    external::NCOArgs
    finaliser::HCOptWeightFinaliser
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
function NCO(; internal::NCOArgs = NCOArgs(;), external::NCOArgs = internal,
             finaliser::HCOptWeightFinaliser = HWF(), opt_kwargs::NamedTuple = (;),
             opt_kwargs_o::NamedTuple = opt_kwargs, port_kwargs::NamedTuple = (;),
             port_kwargs_o::NamedTuple = port_kwargs, factor_kwargs::NamedTuple = (;),
             factor_kwargs_o::NamedTuple = factor_kwargs, wc_kwargs::NamedTuple = (;),
             wc_kwargs_o::NamedTuple = wc_kwargs, cluster_kwargs::NamedTuple = (;),
             cluster_kwargs_o::NamedTuple = cluster_kwargs, stat_kwargs_o::NamedTuple = (;))
    return NCO(internal, external, finaliser, opt_kwargs, opt_kwargs_o, port_kwargs,
               port_kwargs_o, factor_kwargs, factor_kwargs_o, wc_kwargs, wc_kwargs_o,
               cluster_kwargs, cluster_kwargs_o, stat_kwargs_o)
end
function Base.getproperty(nco::NCO, sym::Symbol)
    if sym ∈ (:rm, :obj, :kelly, :class, :w_ini, :custom_constr, :custom_obj, :str_names)
        type = nco.internal.type
        isa(type, NCO) ? getproperty(type, sym) : getfield(type, sym)
    elseif sym ∈
           (:rm_o, :obj_o, :kelly_o, :class_o, :w_ini_o, :custom_constr_o, :custom_obj_o,
            :str_names_o)
        type = nco.external.type
        if isa(type, NCO)
            getproperty(type, sym)
        else
            str_sym = string(sym)
            sym = contains(str_sym, "_o") ? Symbol(str_sym[1:(end - 2)]) : sym
            getfield(type, sym)
        end
    else
        getfield(nco, sym)
    end
end

for (op, name) ∈ zip((Trad, RP, RRP, WC, NOC, HRP, HERC, NCO, SchurHRP, DRCVaR),
                     ("Trad", "RP", "RRP", "WC", "NOC", "HRP", "HERC", "NCO", "SchurHRP", "DRCVaR"))
    eval(quote
             function Base.String(::$op)
                 return $name
             end
             function Base.Symbol(s::$op)
                 return Symbol($name)
             end
         end)
end

export Trad, RP, BasicRRP, RegRRP, RegPenRRP, RRP, WC, NOC, HRP, HERC, NCO, NCOArgs,
       SchurHRP, SchurParams, HWF, JWF, DRCVaR
