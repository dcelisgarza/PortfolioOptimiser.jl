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
