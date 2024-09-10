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

"""
```
struct AIC <: MinValRegressionCriteria end
```
"""
struct AIC <: MinValRegressionCriteria end

"""
```
struct AICC <: MinValRegressionCriteria end
```
"""
struct AICC <: MinValRegressionCriteria end

"""
```
struct BIC <: MinValRegressionCriteria end
```
"""
struct BIC <: MinValRegressionCriteria end

"""
```
struct RSq <: MaxValRegressionCriteria end
```
"""
struct RSq <: MaxValRegressionCriteria end

"""
```
struct AdjRSq <: MaxValRegressionCriteria end
```
"""
struct AdjRSq <: MaxValRegressionCriteria end

"""
```
@kwdef mutable struct PVal{T1 <: Real} <: RegressionCriteria
    threshold::T1 = 0.05
end
```
"""
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

"""
```
abstract type DimensionReductionTarget end
```
"""
abstract type DimensionReductionTarget end

"""
```
@kwdef mutable struct PCATarget <: DimensionReductionTarget
    kwargs::NamedTuple = (;)
end
```
"""
mutable struct PCATarget <: DimensionReductionTarget
    kwargs::NamedTuple
end
function PCATarget(; kwargs::NamedTuple = (;))
    return PCATarget(kwargs)
end

"""
```
@kwdef mutable struct PPCATarget <: DimensionReductionTarget
    kwargs::NamedTuple = (;)
end
```
"""
mutable struct PPCATarget <: DimensionReductionTarget
    kwargs::NamedTuple
end
function PPCATarget(; kwargs::NamedTuple = (;))
    return PPCATarget(kwargs)
end

"""
```
@kwdef mutable struct FReg <: StepwiseRegression
    criterion::RegressionCriteria = PVal(;)
end
```
"""
mutable struct FReg <: StepwiseRegression
    criterion::RegressionCriteria
end
function FReg(; criterion::RegressionCriteria = PVal(;))
    return FReg(criterion)
end

"""
```
@kwdef mutable struct BReg <: StepwiseRegression
    criterion::RegressionCriteria = PVal(;)
end
```
"""
mutable struct BReg <: StepwiseRegression
    criterion::RegressionCriteria
end
function BReg(; criterion::RegressionCriteria = PVal(;))
    return BReg(criterion)
end

"""
```
@kwdef mutable struct PCAReg <: DimensionReductionRegression
    ve::StatsBase.CovarianceEstimator = SimpleVariance(;)
    std_w::Union{<:AbstractWeights, Nothing} = nothing
    mean_w::Union{<:AbstractWeights, Nothing} = nothing
    target::DimensionReductionTarget = PCATarget(;)
end
```
"""
mutable struct PCAReg <: DimensionReductionRegression
    ve::StatsBase.CovarianceEstimator
    std_w::Union{<:AbstractWeights, Nothing}
    mean_w::Union{<:AbstractWeights, Nothing}
    target::DimensionReductionTarget
end
function PCAReg(; ve::StatsBase.CovarianceEstimator = SimpleVariance(;),
                std_w::Union{<:AbstractWeights, Nothing} = nothing,
                mean_w::Union{<:AbstractWeights, Nothing} = nothing,
                target::DimensionReductionTarget = PCATarget(;))
    return PCAReg(ve, std_w, mean_w, target)
end

"""
```
@kwdef mutable struct FactorType
    error::Bool = true
    B::Union{Nothing, DataFrame} = nothing
    method::RegressionType = FReg(;)
    ve::StatsBase.CovarianceEstimator = SimpleVariance(;)
    var_w::Union{<:AbstractWeights, Nothing} = nothing
end
```
"""
mutable struct FactorType
    error::Bool
    B::Union{Nothing, DataFrame}
    method::RegressionType
    ve::StatsBase.CovarianceEstimator
    var_w::Union{<:AbstractWeights, Nothing}
end
function FactorType(; error::Bool = true, B::Union{Nothing, DataFrame} = nothing,
                    method::RegressionType = FReg(;),
                    ve::StatsBase.CovarianceEstimator = SimpleVariance(;),
                    var_w::Union{<:AbstractWeights, Nothing} = nothing)
    return FactorType(error, B, method, ve, var_w)
end

export AIC, AICC, BIC, RSq, AdjRSq, PVal, PCATarget, PPCATarget, FReg, BReg, PCAReg,
       FactorType
