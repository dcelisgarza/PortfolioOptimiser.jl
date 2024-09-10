abstract type BlackLitterman end

"""
```
abstract type BlackLittermanFactor <: BlackLitterman end
```
"""
abstract type BlackLittermanFactor <: BlackLitterman end

"""
```
@kwdef mutable struct BLType{T1 <: Real} <: BlackLitterman
    eq::Bool = true
    delta::Union{<:Real, Nothing} = 1.0
    rf::T1 = 0.0
    posdef::PosdefFix = PosdefNearest()
    denoise::Denoise = NoDenoise()
    logo::AbstractLoGo = NoLoGo()
end
```
"""
mutable struct BLType{T1 <: Real} <: BlackLitterman
    eq::Bool
    delta::Union{<:Real, Nothing}
    rf::T1
    posdef::PosdefFix
    denoise::Denoise
    logo::AbstractLoGo
end
function BLType(; eq::Bool = true, delta::Union{<:Real, Nothing} = 1.0, rf::Real = 0.0,
                posdef::PosdefFix = PosdefNearest(), denoise::Denoise = NoDenoise(),
                logo::AbstractLoGo = NoLoGo())
    return BLType{typeof(rf)}(eq, delta, rf, posdef, denoise, logo)
end

"""
```
@kwdef mutable struct ABLType{T1 <: Real} <: BlackLittermanFactor
    constant::Bool = true
    eq::Bool = true
    delta::Union{<:Real, Nothing} = 1.0
    rf::T1 = 0.0
    posdef::PosdefFix = PosdefNearest()
    denoise::Denoise = NoDenoise()
    logo::AbstractLoGo = NoLoGo()
end
```
"""
mutable struct ABLType{T1 <: Real} <: BlackLittermanFactor
    constant::Bool
    eq::Bool
    delta::Union{<:Real, Nothing}
    rf::T1
    posdef::PosdefFix
    denoise::Denoise
    logo::AbstractLoGo
end
function ABLType(; constant::Bool = true, eq::Bool = true,
                 delta::Union{<:Real, Nothing} = 1.0, rf::Real = 0.0,
                 posdef::PosdefFix = PosdefNearest(), denoise::Denoise = NoDenoise(),
                 logo::AbstractLoGo = NoLoGo())
    return ABLType{typeof(rf)}(constant, eq, delta, rf, posdef, denoise, logo)
end

"""
```
@kwdef mutable struct BBLType{T1 <: Real} <: BlackLittermanFactor
    constant::Bool = true
    error::Bool = true
    delta::Union{<:Real, Nothing} = 1.0
    rf::T1 = 0.0
    ve::StatsBase.CovarianceEstimator = SimpleVariance()
    var_w::Union{<:AbstractWeights, Nothing} = nothing
end
```
"""
mutable struct BBLType{T1 <: Real} <: BlackLittermanFactor
    constant::Bool
    error::Bool
    delta::Union{<:Real, Nothing}
    rf::T1
    ve::StatsBase.CovarianceEstimator
    var_w::Union{<:AbstractWeights, Nothing}
end
function BBLType(; constant::Bool = true, error::Bool = true,
                 delta::Union{<:Real, Nothing} = 1.0, rf::Real = 0.0,
                 ve::StatsBase.CovarianceEstimator = SimpleVariance(),
                 var_w::Union{<:AbstractWeights, Nothing} = nothing)
    return BBLType{typeof(rf)}(constant, error, delta, rf, ve, var_w)
end

export BLType, BBLType, ABLType
