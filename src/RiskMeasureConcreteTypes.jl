# Standard deviation

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