# Copywrite (c) 2025
# Author: Daniel Celis Garza <daniel.celis.garza@gmail.com>
# SPDX-License-Identifier: MIT

"""
```
abstract type RegressionType end
```

Abstract type for subtyping regression types for computing the loadings matrix in [`loadings_matrix`](@ref).
"""
abstract type RegressionType end

"""
```
abstract type StepwiseRegression <: RegressionType end
```

Abstract type for subtyping stepwise regression types for computing the loadings matrix in [`loadings_matrix`](@ref).
"""
abstract type StepwiseRegression <: RegressionType end

"""
```
abstract type DimensionReductionRegression <: RegressionType end
```

Abstract type for subtyping dimensionality reduction regression types for computing the loadings matrix in [`loadings_matrix`](@ref).
"""
abstract type DimensionReductionRegression <: RegressionType end

"""
```
abstract type StepwiseRegressionCriteria end
```

Abstract type for subtyping selection criteria for selecting significant features when using [`StepwiseRegression`](@ref) types.
"""
abstract type StepwiseRegressionCriteria end

"""
```
abstract type MinValStepwiseRegressionCriteria <: StepwiseRegressionCriteria end
```

Abstract type for subtyping selection criteria where smaller values are more significant.
"""
abstract type MinValStepwiseRegressionCriteria <: StepwiseRegressionCriteria end

"""
```
abstract type MinValStepwiseRegressionCriteria <: StepwiseRegressionCriteria end
```

Abstract type for subtyping selection criteria where larger values are more significant.
"""
abstract type MaxValStepwiseRegressionCriteria <: StepwiseRegressionCriteria end

"""
```
struct AIC <: MinValStepwiseRegressionCriteria end
```

[Akaike's Information Criterion](https://juliastats.org/GLM.jl/stable/#Types-applied-to-fitted-models).
"""
struct AIC <: MinValStepwiseRegressionCriteria end

"""
```
struct AICC <: MinValStepwiseRegressionCriteria end
```

[Corrected Akaike's Information Criterion](https://juliastats.org/GLM.jl/stable/#Types-applied-to-fitted-models).
"""
struct AICC <: MinValStepwiseRegressionCriteria end

"""
```
struct BIC <: MinValStepwiseRegressionCriteria end
```

[Bayesian Information Criterion](https://juliastats.org/GLM.jl/stable/#Types-applied-to-fitted-models).
"""
struct BIC <: MinValStepwiseRegressionCriteria end

"""
```
struct RSq <: MaxValStepwiseRegressionCriteria end
```

[R² of a linear model criterion](https://juliastats.org/GLM.jl/stable/#Types-applied-to-fitted-models).
"""
struct RSq <: MaxValStepwiseRegressionCriteria end

"""
```
struct AdjRSq <: MaxValStepwiseRegressionCriteria end
```

[Adjusted R² for a linear model criterion](https://juliastats.org/GLM.jl/stable/#Types-applied-to-fitted-models).
"""
struct AdjRSq <: MaxValStepwiseRegressionCriteria end

"""
```
@kwdef mutable struct PVal{T1 <: Real} <: StepwiseRegressionCriteria
    threshold::T1 = 0.05
end
```

P-value as feature selection criterion.

# Parameters

  - `threshold`: threshold for classifying significant p-values. Only features whose p-values are lower than `threshold` are considered significant.
"""
mutable struct PVal{T1 <: Real} <: StepwiseRegressionCriteria
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
    criterion::StepwiseRegressionCriteria = PVal(;)
end
```

Forward stepwise regression. Starts by assuming no factor is significant and uses `criterion` to add the best performing one each iteration.

# Parameters

  - `criterion`: criterion for feature selection.

      + `isa(criterion, PVal)`: when no asset meets the selecion criterion, the list of significant features can be empty, in such cases the best factor is added to the list.
"""
mutable struct FReg <: StepwiseRegression
    criterion::StepwiseRegressionCriteria
end
function FReg(; criterion::StepwiseRegressionCriteria = PVal(;))
    return FReg(criterion)
end

"""
```
@kwdef mutable struct BReg <: StepwiseRegression
    criterion::StepwiseRegressionCriteria = PVal(;)
end
```

Backward stepwise regression. Starts by assuming all features are significant and uses `criterion` to remove the worst performing one each iteration.

# Parameters

  - `criterion`: criterion for feature selection.

      + `isa(criterion, PVal)`: when no asset meets the selecion criterion, the list of significant features can be empty, in such cases the best factor is added to the list.
"""
mutable struct BReg <: StepwiseRegression
    criterion::StepwiseRegressionCriteria
end
function BReg(; criterion::StepwiseRegressionCriteria = PVal(;))
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
    type::RegressionType = FReg(;)
    ve::StatsBase.CovarianceEstimator = SimpleVariance(;)
    var_w::Union{<:AbstractWeights, Nothing} = nothing
end
```
"""
mutable struct FactorType
    error::Bool
    B::Union{Nothing, DataFrame}
    type::RegressionType
    ve::StatsBase.CovarianceEstimator
    var_w::Union{<:AbstractWeights, Nothing}
end
function FactorType(; error::Bool = true, B::Union{Nothing, DataFrame} = nothing,
                    type::RegressionType = FReg(;),
                    ve::StatsBase.CovarianceEstimator = SimpleVariance(;),
                    var_w::Union{<:AbstractWeights, Nothing} = nothing)
    return FactorType(error, B, type, ve, var_w)
end

export AIC, AICC, BIC, RSq, AdjRSq, PVal, PCATarget, PPCATarget, FReg, BReg, PCAReg,
       FactorType
