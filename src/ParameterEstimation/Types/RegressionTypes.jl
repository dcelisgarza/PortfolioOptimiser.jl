"""
```
abstract type RegressionType end
```
"""
abstract type RegressionType end
abstract type StepwiseRegression <: RegressionType end
abstract type DimensionReductionRegression <: RegressionType end
abstract type RegressionCriteria end
abstract type MinValRegressionCriteria <: RegressionCriteria end
abstract type MaxValRegressionCriteria <: RegressionCriteria end
struct AIC <: MinValRegressionCriteria end
struct AICC <: MinValRegressionCriteria end
struct BIC <: MinValRegressionCriteria end
struct RSq <: MaxValRegressionCriteria end
struct AdjRSq <: MaxValRegressionCriteria end

mutable struct PVal{T1 <: Real} <: RegressionCriteria
    threshold::T1
end
function PVal(; threshold::Real = 0.05)
    @smart_assert(zero(threshold) < threshold < one(threshold))
    return PVal{typeof(threshold)}(threshold)
end
function Base.setproperty!(obj::PVal, sym::Symbol, val)
    if sym == :threshold
        @smart_assert(zero(val) < val < one(val))
    end
    return setfield!(obj, sym, val)
end

abstract type DimensionReductionTarget end
@kwdef mutable struct PCATarget <: DimensionReductionTarget
    kwargs::NamedTuple = (;)
end
@kwdef mutable struct PPCATarget <: DimensionReductionTarget
    kwargs::NamedTuple = (;)
end

@kwdef mutable struct FReg <: StepwiseRegression
    criterion::RegressionCriteria = PVal(;)
end

@kwdef mutable struct BReg <: StepwiseRegression
    criterion::RegressionCriteria = PVal(;)
end

@kwdef mutable struct PCAReg <: DimensionReductionRegression
    ve::StatsBase.CovarianceEstimator = SimpleVariance(;)
    std_w::Union{<:AbstractWeights, Nothing} = nothing
    mean_w::Union{<:AbstractWeights, Nothing} = nothing
    target::DimensionReductionTarget = PCATarget(;)
end

@kwdef mutable struct FactorType
    error::Bool = true
    B::Union{Nothing, DataFrame} = nothing
    method::RegressionType = FReg(;)
    ve::StatsBase.CovarianceEstimator = SimpleVariance(;)
    var_w::Union{<:AbstractWeights, Nothing} = nothing
end

export AIC, AICC, BIC, RSq, AdjRSq, PVal, PCATarget, PPCATarget, FReg, BReg, PCAReg,
       FactorType
