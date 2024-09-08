# Risk measures

"""
```
abstract type AbstractRiskMeasure end
```

Abstract type for subtyping risk measures.
"""
abstract type AbstractRiskMeasure end

"""
```
abstract type RiskMeasure <: AbstractRiskMeasure end
```

Abstract type for subtyping risk measures that can be used to optimise [`Portfolio`](@ref) and [`HCPortfolio`](@ref).
"""
abstract type RiskMeasure <: AbstractRiskMeasure end

"""
```
abstract type HCRiskMeasure <: AbstractRiskMeasure end
```

Abstract type for subtyping risk meaasures that can only be used to optimise [`HCPortfolio`](@ref).
"""
abstract type HCRiskMeasure <: AbstractRiskMeasure end

"""
```
mutable struct RMSettings{T1 <: Real, T2 <: Real}
    flag::Bool = true
    scale::T1 = 1.0
    ub::T2 = Inf
end
```

Risk measure settings for concrete subtypes of [`RiskMeasure`](@ref).

# Parameters

## When optimising a [`Portfolio`](@ref).

  - `flag`: if `true` the risk will contribute to the [`JuMP`] model's risk expression.
  - `scale`: factor for scaling the risk when adding it to the [`JuMP`] model's risk expression.
  - `ub`: if is finite, sets the upper bound for the risk.

## When optimising a [`HCPortfolio`](@ref).

  - `flag`: does nothing.
  - `scale`: factor for scaling the risk when adding it to the risk being minimised.
  - `ub`: does nothing.
"""
mutable struct RMSettings{T1 <: Real, T2 <: Real}
    flag::Bool
    scale::T1
    ub::T2
end
function RMSettings(; flag::Bool = true, scale::Real = 1.0, ub::Real = Inf)
    return RMSettings{typeof(scale), typeof{ub}}(flag, scale, ub)
end

"""
```
@kwdef mutable struct HCRMSettings{T1 <: Real}
    scale::T1 = 1.0
end
```

Risk measure settings for concrete subtypes of [`HCRiskMeasure`](@ref).

# Parameters

  - `scale`: factor for scaling the risk when adding it to the risk being minimised.
"""
mutable struct HCRMSettings{T1 <: Real}
    scale::T1
end
function HCRMSettings(; scale::Real = 1.0)
    return HCRMSettings{typeof(scale)}(scale)
end

# Standard deviation

abstract type SDFormulation end
abstract type SDSquaredFormulation <: SDFormulation end
struct QuadSD <: SDSquaredFormulation end
struct SOCSD <: SDSquaredFormulation end
struct SimpleSD <: SDFormulation end
mutable struct SD{T1 <: Union{AbstractMatrix, Nothing}} <: RiskMeasure
    settings::RMSettings
    formulation::SDFormulation
    sigma::T1
end
function SD(; settings::RMSettings = RMSettings(), formulation = SOCSD(),
            sigma::Union{<:AbstractMatrix, Nothing} = nothing)
    if !isnothing(sigma)
        @smart_assert(size(sigma, 1) == size(sigma, 2))
    end
    return SD{Union{<:AbstractMatrix, Nothing}}(settings, formulation, sigma)
end
function Base.setproperty!(obj::SD, sym::Symbol, val)
    if sym == :sigma
        if !isnothing(val)
            @smart_assert(size(val, 1) == size(val, 2))
        end
    end
    return setfield!(obj, sym, val)
end

@kwdef mutable struct MAD <: RiskMeasure
    settings::RMSettings = RMSettings()
    w::Union{<:AbstractWeights, Nothing} = nothing
    mu::Union{<:AbstractVector, Nothing} = nothing
end

@kwdef mutable struct SVariance{T1 <: Real} <: HCRiskMeasure
    settings::RMSettings = RMSettings()
    target::T1 = 0.0
    w::Union{<:AbstractWeights, Nothing} = nothing
    mu::Union{<:AbstractVector, Nothing} = nothing
end

@kwdef mutable struct SSD{T1 <: Real} <: RiskMeasure
    settings::RMSettings = RMSettings()
    target::T1 = 0.0
    w::Union{<:AbstractWeights, Nothing} = nothing
    mu::Union{<:AbstractVector, Nothing} = nothing
end

@kwdef mutable struct FLPM{T1 <: Real} <: RiskMeasure
    settings::RMSettings = RMSettings()
    target::T1 = 0.0
end

@kwdef mutable struct SLPM{T1 <: Real} <: RiskMeasure
    settings::RMSettings = RMSettings()
    target::T1 = 0.0
end

@kwdef struct WR <: RiskMeasure
    settings::RMSettings = RMSettings()
end

mutable struct CVaR{T1 <: Real} <: RiskMeasure
    settings::RMSettings
    alpha::T1
end
function CVaR(; settings::RMSettings = RMSettings(), alpha::Real = 0.05)
    @smart_assert(zero(alpha) < alpha < one(alpha))
    return CVaR{typeof(alpha)}(settings, alpha)
end
function Base.setproperty!(obj::CVaR, sym::Symbol, val)
    if sym == :alpha
        @smart_assert(zero(val) < val < one(val))
    end
    return setfield!(obj, sym, val)
end

mutable struct EVaR{T1 <: Real} <: RiskMeasure
    settings::RMSettings
    alpha::T1
    solvers::Union{<:AbstractDict, Nothing}
end
function EVaR(; settings::RMSettings = RMSettings(), alpha::Real = 0.05,
              solvers::Union{<:AbstractDict, Nothing} = nothing)
    @smart_assert(zero(alpha) < alpha < one(alpha))
    return EVaR{typeof(alpha)}(settings, alpha, solvers)
end
function Base.setproperty!(obj::EVaR, sym::Symbol, val)
    if sym == :alpha
        @smart_assert(zero(val) < val < one(val))
    end
    return setfield!(obj, sym, val)
end

mutable struct RLVaR{T1 <: Real, T2 <: Real} <: RiskMeasure
    settings::RMSettings
    alpha::T1
    kappa::T2
    solvers::Union{<:AbstractDict, Nothing}
end
function RLVaR(; settings::RMSettings = RMSettings(), alpha::Real = 0.05, kappa = 0.3,
               solvers::Union{<:AbstractDict, Nothing} = nothing)
    @smart_assert(zero(alpha) < alpha < one(alpha))
    @smart_assert(zero(kappa) < kappa < one(kappa))
    return RLVaR{typeof(alpha), typeof(kappa)}(settings, alpha, kappa, solvers)
end
function Base.setproperty!(obj::RLVaR, sym::Symbol, val)
    if sym ∈ (:alpha, :kappa)
        @smart_assert(zero(val) < val < one(val))
    end
    return setfield!(obj, sym, val)
end

@kwdef struct MDD <: RiskMeasure
    settings::RMSettings = RMSettings()
end

@kwdef struct ADD <: RiskMeasure
    settings::RMSettings = RMSettings()
end

mutable struct CDaR{T1 <: Real} <: RiskMeasure
    settings::RMSettings
    alpha::T1
end
function CDaR(; settings::RMSettings = RMSettings(), alpha::Real = 0.05)
    @smart_assert(zero(alpha) < alpha < one(alpha))
    return CDaR{typeof(alpha)}(settings, alpha)
end
function Base.setproperty!(obj::CDaR, sym::Symbol, val)
    if sym == :alpha
        @smart_assert(zero(val) < val < one(val))
    end
    return setfield!(obj, sym, val)
end

@kwdef struct UCI <: RiskMeasure
    settings::RMSettings = RMSettings()
end

mutable struct EDaR{T1 <: Real} <: RiskMeasure
    settings::RMSettings
    alpha::T1
    solvers::Union{<:AbstractDict, Nothing}
end
function EDaR(; settings::RMSettings = RMSettings(), alpha::Real = 0.05,
              solvers::Union{<:AbstractDict, Nothing} = nothing)
    @smart_assert(zero(alpha) < alpha < one(alpha))
    return EDaR{typeof(alpha)}(settings, alpha, solvers)
end
function Base.setproperty!(obj::EDaR, sym::Symbol, val)
    if sym == :alpha
        @smart_assert(zero(val) < val < one(val))
    end
    return setfield!(obj, sym, val)
end

mutable struct RLDaR{T1 <: Real, T2 <: Real} <: RiskMeasure
    settings::RMSettings
    alpha::T1
    kappa::T2
    solvers::Union{<:AbstractDict, Nothing}
end
function RLDaR(; settings = RMSettings(), alpha::Real = 0.05, kappa = 0.3,
               solvers::Union{<:AbstractDict, Nothing} = nothing)
    @smart_assert(zero(alpha) < alpha < one(alpha))
    @smart_assert(zero(kappa) < kappa < one(kappa))
    return RLDaR{typeof(alpha), typeof(kappa)}(settings, alpha, kappa, solvers)
end
function Base.setproperty!(obj::RLDaR, sym::Symbol, val)
    if sym ∈ (:alpha, :kappa)
        @smart_assert(zero(val) < val < one(val))
    end
    return setfield!(obj, sym, val)
end

@kwdef mutable struct Kurt <: RiskMeasure
    settings::RMSettings = RMSettings()
    w::Union{<:AbstractWeights, Nothing} = nothing
    kt::Union{<:AbstractMatrix, Nothing} = nothing
end

@kwdef mutable struct SKurt{T1 <: Real} <: RiskMeasure
    settings::RMSettings = RMSettings()
    target::T1 = 0.0
    w::Union{<:AbstractWeights, Nothing} = nothing
    kt::Union{<:AbstractMatrix, Nothing} = nothing
end

@kwdef struct RG <: RiskMeasure
    settings::RMSettings = RMSettings()
end

mutable struct CVaRRG{T1, T2} <: RiskMeasure
    settings::RMSettings
    alpha::T1
    beta::T2
end
function CVaRRG(; settings::RMSettings = RMSettings(), alpha::Real = 0.05,
                beta::Real = 0.05)
    @smart_assert(zero(alpha) < alpha < one(alpha))
    @smart_assert(zero(beta) < beta < one(beta))
    return CVaRRG{typeof(alpha), typeof(beta)}(settings, alpha, beta)
end
function Base.setproperty!(obj::CVaRRG, sym::Symbol, val)
    if sym ∈ (:alpha, :beta)
        @smart_assert(zero(val) < val < one(val))
    end
    return setfield!(obj, sym, val)
end

@kwdef mutable struct OWASettings{T1}
    approx::Bool = true
    p::T1 = Float64[2, 3, 4, 10, 50]
end

@kwdef struct GMD <: RiskMeasure
    settings::RMSettings = RMSettings()
    owa::OWASettings = OWASettings()
end

mutable struct TG{T1, T2, T3} <: RiskMeasure
    settings::RMSettings
    owa::OWASettings
    alpha_i::T1
    alpha::T2
    a_sim::T3
end
function TG(; settings::RMSettings = RMSettings(), owa::OWASettings = OWASettings(),
            alpha_i::Real = 0.0001, alpha::Real = 0.05, a_sim::Integer = 100)
    @smart_assert(zero(alpha) < alpha_i < alpha < one(alpha))
    @smart_assert(a_sim > zero(a_sim))
    return TG{typeof(alpha_i), typeof(alpha), typeof(a_sim)}(settings, owa, alpha_i, alpha,
                                                             a_sim)
end
function Base.setproperty!(obj::TG, sym::Symbol, val)
    if sym == :alpha_i
        @smart_assert(zero(val) < val < obj.alpha < one(val))
    elseif sym == :alpha
        @smart_assert(zero(val) < obj.alpha_i < val < one(val))
    elseif sym == :a_sim
        @smart_assert(val > zero(val))
    end
    return setfield!(obj, sym, val)
end

mutable struct TGRG{T1, T2, T3, T4, T5, T6} <: RiskMeasure
    settings::RMSettings
    owa::OWASettings
    alpha_i::T1
    alpha::T2
    a_sim::T3
    beta_i::T4
    beta::T5
    b_sim::T6
end
function TGRG(; settings::RMSettings = RMSettings(), owa::OWASettings = OWASettings(),
              alpha_i = 0.0001, alpha::Real = 0.05, a_sim::Integer = 100, beta_i = 0.0001,
              beta::Real = 0.05, b_sim::Integer = 100)
    @smart_assert(zero(alpha) < alpha_i < alpha < one(alpha))
    @smart_assert(a_sim > zero(a_sim))
    @smart_assert(zero(beta) < beta_i < beta < one(beta))
    @smart_assert(b_sim > zero(b_sim))
    return TGRG{typeof(alpha_i), typeof(alpha), typeof(a_sim), typeof(beta_i), typeof(beta),
                typeof(b_sim)}(settings, owa, alpha_i, alpha, a_sim, beta_i, beta, b_sim)
end
function Base.setproperty!(obj::TGRG, sym::Symbol, val)
    if sym == :alpha_i
        @smart_assert(zero(val) < val < obj.alpha < one(val))
    elseif sym == :alpha
        @smart_assert(zero(val) < obj.alpha_i < val < one(val))
    elseif sym == :a_sim
        @smart_assert(val > zero(val))
    elseif sym == :beta_i
        @smart_assert(zero(val) < val < obj.beta < one(val))
    elseif sym == :beta
        @smart_assert(zero(val) < obj.beta_i < val < one(val))
    elseif sym == :b_sim
        @smart_assert(val > zero(val))
    end
    return setfield!(obj, sym, val)
end

@kwdef mutable struct OWA <: RiskMeasure
    settings::RMSettings = RMSettings()
    owa::OWASettings = OWASettings()
    w::Union{<:AbstractVector, Nothing} = nothing
end

@kwdef struct dVar <: RiskMeasure
    settings::RMSettings = RMSettings()
end

@kwdef struct Skew <: RiskMeasure
    settings::RMSettings = RMSettings()
end

@kwdef struct SSkew <: RiskMeasure
    settings::RMSettings = RMSettings()
end

# ## HCPortfolio risk measures

mutable struct Variance{T1 <: Union{AbstractMatrix, Nothing}} <: HCRiskMeasure
    sigma::T1
    settings::HCRMSettings
end
function Variance(; sigma::Union{<:AbstractMatrix, Nothing} = nothing,
                  settings::HCRMSettings = HCRMSettings())
    if !isnothing(sigma)
        @smart_assert(size(sigma, 1) == size(sigma, 2))
    end
    return Variance{Union{<:AbstractMatrix, Nothing}}(sigma, settings)
end
function Base.setproperty!(obj::Variance, sym::Symbol, val)
    if sym == :sigma
        if !isnothing(val)
            @smart_assert(size(val, 1) == size(val, 2))
        end
    end
    return setfield!(obj, sym, val)
end

mutable struct VaR{T1 <: Real} <: HCRiskMeasure
    alpha::T1
    settings::HCRMSettings
end
function VaR(; alpha::Real = 0.05, settings::HCRMSettings = HCRMSettings())
    @smart_assert(zero(alpha) < alpha < one(alpha))
    return VaR{typeof(alpha)}(alpha, settings)
end
function Base.setproperty!(obj::VaR, sym::Symbol, val)
    if sym == :alpha
        @smart_assert(zero(val) < val < one(val))
    end
    return setfield!(obj, sym, val)
end

mutable struct DaR{T1 <: Real} <: HCRiskMeasure
    alpha::T1
    settings::HCRMSettings
end
function DaR(; alpha::Real = 0.05, settings::HCRMSettings = HCRMSettings())
    @smart_assert(zero(alpha) < alpha < one(alpha))
    return DaR{typeof(alpha)}(alpha, settings)
end
function Base.setproperty!(obj::DaR, sym::Symbol, val)
    if sym == :alpha
        @smart_assert(zero(val) < val < one(val))
    end
    return setfield!(obj, sym, val)
end

mutable struct DaR_r{T1 <: Real} <: HCRiskMeasure
    alpha::T1
    settings::HCRMSettings
end
function DaR_r(; alpha::Real = 0.05, settings::HCRMSettings = HCRMSettings())
    @smart_assert(zero(alpha) < alpha < one(alpha))
    return DaR_r{typeof(alpha)}(alpha, settings)
end
function Base.setproperty!(obj::DaR_r, sym::Symbol, val)
    if sym == :alpha
        @smart_assert(zero(val) < val < one(val))
    end
    return setfield!(obj, sym, val)
end

@kwdef struct MDD_r <: HCRiskMeasure
    settings::HCRMSettings = HCRMSettings()
end

@kwdef struct ADD_r <: HCRiskMeasure
    settings::HCRMSettings = HCRMSettings()
end

mutable struct CDaR_r{T1 <: Real} <: RiskMeasure
    settings::RMSettings
    alpha::T1
end
function CDaR_r(; settings::RMSettings = RMSettings(), alpha::Real = 0.05)
    @smart_assert(zero(alpha) < alpha < one(alpha))
    return CDaR_r{typeof(alpha)}(settings, alpha)
end
function Base.setproperty!(obj::CDaR_r, sym::Symbol, val)
    if sym == :alpha
        @smart_assert(zero(val) < val < one(val))
    end
    return setfield!(obj, sym, val)
end

@kwdef struct UCI_r <: HCRiskMeasure
    settings::HCRMSettings = HCRMSettings()
end

mutable struct EDaR_r{T1 <: Real} <: RiskMeasure
    settings::RMSettings
    alpha::T1
    solvers::Union{<:AbstractDict, Nothing}
end
function EDaR_r(; settings::RMSettings = RMSettings(), alpha::Real = 0.05,
                solvers::Union{<:AbstractDict, Nothing} = nothing)
    @smart_assert(zero(alpha) < alpha < one(alpha))
    return EDaR_r{typeof(alpha)}(settings, alpha, solvers)
end
function Base.setproperty!(obj::EDaR_r, sym::Symbol, val)
    if sym == :alpha
        @smart_assert(zero(val) < val < one(val))
    end
    return setfield!(obj, sym, val)
end

mutable struct RLDaR_r{T1 <: Real, T2 <: Real} <: RiskMeasure
    settings::RMSettings
    alpha::T1
    kappa::T2
    solvers::Union{<:AbstractDict, Nothing}
end
function RLDaR_r(; settings::RMSettings = RMSettings(), alpha::Real = 0.05, kappa = 0.3,
                 solvers::Union{<:AbstractDict, Nothing} = nothing)
    @smart_assert(zero(alpha) < alpha < one(alpha))
    @smart_assert(zero(kappa) < kappa < one(kappa))
    return RLDaR_r{typeof(alpha), typeof(kappa)}(settings, alpha, kappa, solvers)
end
function Base.setproperty!(obj::RLDaR_r, sym::Symbol, val)
    if sym ∈ (:alpha, :kappa)
        @smart_assert(zero(val) < val < one(val))
    end
    return setfield!(obj, sym, val)
end

@kwdef struct Equal <: HCRiskMeasure
    settings::HCRMSettings = HCRMSettings()
end

for (op, name) ∈
    zip((SD, Variance, MAD, SSD, SVariance, FLPM, SLPM, WR, VaR, CVaR, EVaR, RLVaR, DaR,
         MDD, ADD, CDaR, UCI, EDaR, RLDaR, DaR_r, MDD_r, ADD_r, CDaR_r, UCI_r, EDaR_r,
         RLDaR_r, Kurt, SKurt, GMD, RG, CVaRRG, TG, TGRG, OWA, dVar, Skew, SSkew, Equal),
        ("SD", "Variance", "MAD", "SSD", "SVariance", "FLPM", "SLPM", "WR", "VaR", "CVaR",
         "EVaR", "RLVaR", "DaR", "MDD", "ADD", "CDaR", "UCI", "EDaR", "RLDaR", "DaR_r",
         "MDD_r", "ADD_r", "CDaR_r", "UCI_r", "EDaR_r", "RLDaR_r", "Kurt", "SKurt", "GMD",
         "RG", "CVaRRG", "TG", "TGRG", "OWA", "dVar", "Skew", "SSkew", "Equal"))
    eval(quote
             Base.iterate(S::$op, state = 1) = state > 1 ? nothing : (S, state + 1)
             function Base.String(s::$op)
                 return $name
             end
             function Base.Symbol(::$op)
                 return Symbol($name)
             end
             function Base.length(::$op)
                 return 1
             end
             function Base.getindex(S::$op, I::Integer...)
                 return S
             end
         end)
end

export RMSettings, HCRMSettings, QuadSD, SOCSD, SimpleSD, SD, MAD, SSD, FLPM, SLPM, WR,
       CVaR, EVaR, RLVaR, MDD, ADD, CDaR, UCI, EDaR, RLDaR, Kurt, SKurt, RG, CVaRRG,
       OWASettings, GMD, TG, TGRG, OWA, dVar, Skew, SSkew, Variance, SVariance, VaR, DaR,
       DaR_r, MDD_r, ADD_r, CDaR_r, UCI_r, EDaR_r, RLDaR_r, Equal
