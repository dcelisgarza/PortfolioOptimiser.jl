# # Risk measures

# ## Abstract and support types

abstract type RiskMeasure end
abstract type TradRiskMeasure <: RiskMeasure end
abstract type HCRiskMeasure <: RiskMeasure end
@kwdef mutable struct RiskMeasureSettings{T1 <: Real, T2 <: Real}
    flag::Bool = true
    scale::T1 = 1.0
    ub::T2 = Inf
end
@kwdef mutable struct HCRiskMeasureSettings{T1 <: Real}
    scale::T1 = 1.0
end

# ## Portfolio risk measures

abstract type SDFormulation end
abstract type SDSquaredFormulation <: SDFormulation end
struct QuadSD <: SDSquaredFormulation end
struct SOCSD <: SDSquaredFormulation end
struct SimpleSD <: SDFormulation end
mutable struct SD{T1 <: Union{AbstractMatrix, Nothing}} <: TradRiskMeasure
    settings::RiskMeasureSettings
    formulation::SDFormulation
    sigma::T1
end
function SD(; settings::RiskMeasureSettings = RiskMeasureSettings(), formulation = SOCSD(),
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

@kwdef mutable struct MAD <: TradRiskMeasure
    settings::RiskMeasureSettings = RiskMeasureSettings()
    w::Union{<:AbstractWeights, Nothing} = nothing
    mu::Union{<:AbstractVector, Nothing} = nothing
end

@kwdef mutable struct SSD{T1 <: Real} <: TradRiskMeasure
    settings::RiskMeasureSettings = RiskMeasureSettings()
    target::T1 = 0.0
    w::Union{<:AbstractWeights, Nothing} = nothing
    mu::Union{<:AbstractVector, Nothing} = nothing
end

@kwdef mutable struct FLPM{T1 <: Real} <: TradRiskMeasure
    settings::RiskMeasureSettings = RiskMeasureSettings()
    target::T1 = 0.0
end

@kwdef mutable struct SLPM{T1 <: Real} <: TradRiskMeasure
    settings::RiskMeasureSettings = RiskMeasureSettings()
    target::T1 = 0.0
end

@kwdef struct WR <: TradRiskMeasure
    settings::RiskMeasureSettings = RiskMeasureSettings()
end

mutable struct CVaR{T1 <: Real} <: TradRiskMeasure
    settings::RiskMeasureSettings
    alpha::T1
end
function CVaR(; settings::RiskMeasureSettings = RiskMeasureSettings(), alpha::Real = 0.05)
    @smart_assert(zero(alpha) < alpha < one(alpha))
    return CVaR{typeof(alpha)}(settings, alpha)
end
function Base.setproperty!(obj::CVaR, sym::Symbol, val)
    if sym == :alpha
        @smart_assert(zero(val) < val < one(val))
    end
    return setfield!(obj, sym, val)
end

mutable struct EVaR{T1 <: Real} <: TradRiskMeasure
    settings::RiskMeasureSettings
    alpha::T1
    solvers::Union{<:AbstractDict, NamedTuple, Nothing}
end
function EVaR(; settings::RiskMeasureSettings = RiskMeasureSettings(), alpha::Real = 0.05,
              solvers::Union{<:AbstractDict, NamedTuple, Nothing} = nothing)
    @smart_assert(zero(alpha) < alpha < one(alpha))
    return EVaR{typeof(alpha)}(settings, alpha, solvers)
end
function Base.setproperty!(obj::EVaR, sym::Symbol, val)
    if sym == :alpha
        @smart_assert(zero(val) < val < one(val))
    end
    return setfield!(obj, sym, val)
end

mutable struct RVaR{T1 <: Real, T2 <: Real} <: TradRiskMeasure
    settings::RiskMeasureSettings
    alpha::T1
    kappa::T2
    solvers::Union{<:AbstractDict, NamedTuple, Nothing}
end
function RVaR(; settings::RiskMeasureSettings = RiskMeasureSettings(), alpha::Real = 0.05,
              kappa = 0.3, solvers::Union{<:AbstractDict, NamedTuple, Nothing} = nothing)
    @smart_assert(zero(alpha) < alpha < one(alpha))
    @smart_assert(zero(kappa) < kappa < one(kappa))
    return RVaR{typeof(alpha), typeof(kappa)}(settings, alpha, kappa, solvers)
end
function Base.setproperty!(obj::RVaR, sym::Symbol, val)
    if sym ∈ (:alpha, :kappa)
        @smart_assert(zero(val) < val < one(val))
    end
    return setfield!(obj, sym, val)
end

@kwdef struct MDD <: TradRiskMeasure
    settings::RiskMeasureSettings = RiskMeasureSettings()
end

@kwdef struct ADD <: TradRiskMeasure
    settings::RiskMeasureSettings = RiskMeasureSettings()
end

mutable struct CDaR{T1 <: Real} <: TradRiskMeasure
    settings::RiskMeasureSettings
    alpha::T1
end
function CDaR(; settings::RiskMeasureSettings = RiskMeasureSettings(), alpha::Real = 0.05)
    @smart_assert(zero(alpha) < alpha < one(alpha))
    return CDaR{typeof(alpha)}(settings, alpha)
end
function Base.setproperty!(obj::CDaR, sym::Symbol, val)
    if sym == :alpha
        @smart_assert(zero(val) < val < one(val))
    end
    return setfield!(obj, sym, val)
end

@kwdef struct UCI <: TradRiskMeasure
    settings::RiskMeasureSettings = RiskMeasureSettings()
end

mutable struct EDaR{T1 <: Real} <: TradRiskMeasure
    settings::RiskMeasureSettings
    alpha::T1
    solvers::Union{<:AbstractDict, NamedTuple, Nothing}
end
function EDaR(; settings::RiskMeasureSettings = RiskMeasureSettings(), alpha::Real = 0.05,
              solvers::Union{<:AbstractDict, NamedTuple, Nothing} = nothing)
    @smart_assert(zero(alpha) < alpha < one(alpha))
    return EDaR{typeof(alpha)}(settings, alpha, solvers)
end
function Base.setproperty!(obj::EDaR, sym::Symbol, val)
    if sym == :alpha
        @smart_assert(zero(val) < val < one(val))
    end
    return setfield!(obj, sym, val)
end

mutable struct RDaR{T1 <: Real, T2 <: Real} <: TradRiskMeasure
    settings::RiskMeasureSettings
    alpha::T1
    kappa::T2
    solvers::Union{<:AbstractDict, NamedTuple, Nothing}
end
function RDaR(; settings = RiskMeasureSettings(), alpha::Real = 0.05, kappa = 0.3,
              solvers::Union{<:AbstractDict, NamedTuple, Nothing} = nothing)
    @smart_assert(zero(alpha) < alpha < one(alpha))
    @smart_assert(zero(kappa) < kappa < one(kappa))
    return RDaR{typeof(alpha), typeof(kappa)}(settings, alpha, kappa, solvers)
end
function Base.setproperty!(obj::RDaR, sym::Symbol, val)
    if sym ∈ (:alpha, :kappa)
        @smart_assert(zero(val) < val < one(val))
    end
    return setfield!(obj, sym, val)
end

@kwdef mutable struct Kurt <: TradRiskMeasure
    settings::RiskMeasureSettings = RiskMeasureSettings()
    w::Union{<:AbstractWeights, Nothing} = nothing
    kt::Union{<:AbstractMatrix, Nothing} = nothing
end

@kwdef mutable struct SKurt <: TradRiskMeasure
    settings::RiskMeasureSettings = RiskMeasureSettings()
    w::Union{<:AbstractWeights, Nothing} = nothing
    kt::Union{<:AbstractMatrix, Nothing} = nothing
end

@kwdef struct RG <: TradRiskMeasure
    settings::RiskMeasureSettings = RiskMeasureSettings()
end

mutable struct RCVaR{T1, T2} <: TradRiskMeasure
    settings::RiskMeasureSettings
    alpha::T1
    beta::T2
end
function RCVaR(; settings::RiskMeasureSettings = RiskMeasureSettings(), alpha::Real = 0.05,
               beta::Real = 0.05)
    @smart_assert(zero(alpha) < alpha < one(alpha))
    @smart_assert(zero(beta) < beta < one(beta))
    return RCVaR{typeof(alpha), typeof(beta)}(settings, alpha, beta)
end
function Base.setproperty!(obj::RCVaR, sym::Symbol, val)
    if sym ∈ (:alpha, :beta)
        @smart_assert(zero(val) < val < one(val))
    end
    return setfield!(obj, sym, val)
end

@kwdef mutable struct OWASettings{T1}
    approx::Bool = true
    p::T1 = Float64[2, 3, 4, 10, 50]
end

@kwdef struct GMD <: TradRiskMeasure
    settings::RiskMeasureSettings = RiskMeasureSettings()
    owa::OWASettings = OWASettings()
end

mutable struct TG{T1, T2, T3} <: TradRiskMeasure
    settings::RiskMeasureSettings
    owa::OWASettings
    alpha_i::T1
    alpha::T2
    a_sim::T3
end
function TG(; settings::RiskMeasureSettings = RiskMeasureSettings(),
            owa::OWASettings = OWASettings(), alpha_i::Real = 0.0001, alpha::Real = 0.05,
            a_sim::Integer = 100)
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

mutable struct RTG{T1, T2, T3, T4, T5, T6} <: TradRiskMeasure
    settings::RiskMeasureSettings
    owa::OWASettings
    alpha_i::T1
    alpha::T2
    a_sim::T3
    beta_i::T4
    beta::T5
    b_sim::T6
end
function RTG(; settings::RiskMeasureSettings = RiskMeasureSettings(),
             owa::OWASettings = OWASettings(), alpha_i = 0.0001, alpha::Real = 0.05,
             a_sim::Integer = 100, beta_i = 0.0001, beta::Real = 0.05, b_sim::Integer = 100)
    @smart_assert(zero(alpha) < alpha_i < alpha < one(alpha))
    @smart_assert(a_sim > zero(a_sim))
    @smart_assert(zero(beta) < beta_i < beta < one(beta))
    @smart_assert(b_sim > zero(b_sim))
    return RTG{typeof(alpha_i), typeof(alpha), typeof(a_sim), typeof(beta_i), typeof(beta),
               typeof(b_sim)}(settings, owa, alpha_i, alpha, a_sim, beta_i, beta, b_sim)
end
function Base.setproperty!(obj::RTG, sym::Symbol, val)
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

@kwdef mutable struct OWA <: TradRiskMeasure
    settings::RiskMeasureSettings = RiskMeasureSettings()
    owa::OWASettings = OWASettings()
    w::Union{<:AbstractVector, Nothing} = nothing
end

@kwdef struct DVar <: TradRiskMeasure
    settings::RiskMeasureSettings = RiskMeasureSettings()
end

@kwdef struct Skew <: TradRiskMeasure
    settings::RiskMeasureSettings = RiskMeasureSettings()
end

@kwdef struct SSkew <: TradRiskMeasure
    settings::RiskMeasureSettings = RiskMeasureSettings()
end

# ## HCPortfolio risk measures

mutable struct Variance{T1 <: Union{AbstractMatrix, Nothing}} <: HCRiskMeasure
    sigma::T1
    settings::HCRiskMeasureSettings
end
function Variance(; sigma::Union{<:AbstractMatrix, Nothing} = nothing,
                  settings::HCRiskMeasureSettings = HCRiskMeasureSettings())
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
    settings::HCRiskMeasureSettings
end
function VaR(; alpha::Real = 0.05,
             settings::HCRiskMeasureSettings = HCRiskMeasureSettings())
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
    settings::HCRiskMeasureSettings
end
function DaR(; alpha::Real = 0.05,
             settings::HCRiskMeasureSettings = HCRiskMeasureSettings())
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
    settings::HCRiskMeasureSettings
end
function DaR_r(; alpha::Real = 0.05,
               settings::HCRiskMeasureSettings = HCRiskMeasureSettings())
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
    settings::HCRiskMeasureSettings = HCRiskMeasureSettings()
end

@kwdef struct ADD_r <: HCRiskMeasure
    settings::HCRiskMeasureSettings = HCRiskMeasureSettings()
end

mutable struct CDaR_r{T1 <: Real} <: TradRiskMeasure
    settings::RiskMeasureSettings
    alpha::T1
end
function CDaR_r(; settings::RiskMeasureSettings = RiskMeasureSettings(), alpha::Real = 0.05)
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
    settings::HCRiskMeasureSettings = HCRiskMeasureSettings()
end

mutable struct EDaR_r{T1 <: Real} <: TradRiskMeasure
    settings::RiskMeasureSettings
    alpha::T1
    solvers::Union{<:AbstractDict, NamedTuple, Nothing}
end
function EDaR_r(; settings::RiskMeasureSettings = RiskMeasureSettings(), alpha::Real = 0.05,
                solvers::Union{<:AbstractDict, NamedTuple, Nothing} = nothing)
    @smart_assert(zero(alpha) < alpha < one(alpha))
    return EDaR_r{typeof(alpha)}(settings, alpha, solvers)
end
function Base.setproperty!(obj::EDaR_r, sym::Symbol, val)
    if sym == :alpha
        @smart_assert(zero(val) < val < one(val))
    end
    return setfield!(obj, sym, val)
end

mutable struct RDaR_r{T1 <: Real, T2 <: Real} <: TradRiskMeasure
    settings::RiskMeasureSettings
    alpha::T1
    kappa::T2
    solvers::Union{<:AbstractDict, NamedTuple, Nothing}
end
function RDaR_r(; settings::RiskMeasureSettings = RiskMeasureSettings(), alpha::Real = 0.05,
                kappa = 0.3, solvers::Union{<:AbstractDict, NamedTuple, Nothing} = nothing)
    @smart_assert(zero(alpha) < alpha < one(alpha))
    @smart_assert(zero(kappa) < kappa < one(kappa))
    return RDaR_r{typeof(alpha), typeof(kappa)}(settings, alpha, kappa, solvers)
end
function Base.setproperty!(obj::RDaR_r, sym::Symbol, val)
    if sym ∈ (:alpha, :kappa)
        @smart_assert(zero(val) < val < one(val))
    end
    return setfield!(obj, sym, val)
end

@kwdef struct Equal <: HCRiskMeasure
    settings::HCRiskMeasureSettings = HCRiskMeasureSettings()
end

export RiskMeasureSettings, HCRiskMeasureSettings, QuadSD, SOCSD, SimpleSD, SD, MAD, SSD,
       FLPM, SLPM, WR, CVaR, EVaR, RVaR, MDD, ADD, CDaR, UCI, EDaR, RDaR, Kurt, SKurt, RG,
       RCVaR, OWASettings, GMD, TG, RTG, OWA, DVar, Skew, SSkew, Variance, VaR, DaR, DaR_r,
       MDD_r, ADD_r, CDaR_r, UCI_r, EDaR_r, RDaR_r, Equal
