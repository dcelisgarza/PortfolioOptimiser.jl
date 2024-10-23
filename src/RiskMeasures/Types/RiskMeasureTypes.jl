"""
    abstract type AbstractRiskMeasure end

Root abstract type for all risk measures in the type hierarchy. Serves as the base type for implementing various risk measurement approaches.
"""
abstract type AbstractRiskMeasure end

"""
    abstract type RiskMeasure <: AbstractRiskMeasure end

Abstract type for risk measures that are compatible with both `Portfolio` and `HCPortfolio` optimization. Concrete subtypes can be used in either context.
"""
abstract type RiskMeasure <: AbstractRiskMeasure end

"""
    abstract type HCRiskMeasure <: AbstractRiskMeasure end

Abstract type for specialized risk measures that can only be used with `HCPortfolio` optimization. These risk measures are not compatible with standard `Portfolio` optimization.
"""
abstract type HCRiskMeasure <: AbstractRiskMeasure end

"""
    mutable struct RMSettings{T1 <: Real, T2 <: Real}

Configuration settings for risk measures that subtype `RiskMeasure`.

# Fields

  - `flag::Bool`: Controls risk contribution to the optimization model
  - `scale::T1`: Scaling factor for the risk measure
  - `ub::T2`: Upper bound constraint for the risk measure

# Behavior

## For Portfolio optimization:

  - `flag`: When true, includes risk in the JuMP model's risk expression
  - `scale`: Scaling factor applied when adding the risk to the optimisation objective
  - `ub`: When finite, sets the upper bound constraint on the risk measure

## For HCPortfolio optimization:

  - `flag`: No effect
  - `scale`: Scaling factor applied when adding the risk to the optimisation objective
  - `ub`: No effect

# Examples

```julia
# Default settings
settings = RMSettings()

# Custom settings
settings = RMSettings(; flag = true, scale = 2.0, ub = 0.5)
```
"""
mutable struct RMSettings{T1 <: Real, T2 <: Real}
    flag::Bool
    scale::T1
    ub::T2
end
function RMSettings(; flag::Bool = true, scale::Real = 1.0, ub::Real = Inf)
    return RMSettings{typeof(scale), typeof(ub)}(flag, scale, ub)
end

"""
    mutable struct HCRMSettings{T1 <: Real}

Settings configuration for hierarchical clustering (HC) risk measures.

# Type Parameters

  - `T1`: Numeric type for the scale parameter, must be a subtype of `Real`

# Fields

  - `scale::T1`: Scaling factor applied when adding the risk to the minimization objective

# Examples

```julia
# Default settings
settings = HCRMSettings()

# Custom scale
settings = HCRMSettings(; scale = 2.5)
```
"""
mutable struct HCRMSettings{T1 <: Real}
    scale::T1
end
function HCRMSettings(; scale::Real = 1.0)
    return HCRMSettings{typeof(scale)}(scale)
end

"""
    abstract type SDFormulation end

Abstract type hierarchy for Mean-Variance optimization formulations. Serves as the root type for different standard deviation calculation approaches in portfolio optimization.
"""
abstract type SDFormulation end

"""
    abstract type SDSquaredFormulation <: SDFormulation end

Abstract type for Mean-Variance formulations that produce quadratic expressions for the JuMP model's standard deviation risk.

# Implementation Notes

  - Produces a `JuMP.QuadExpr` for the model's `sd_risk`
  - [`NOC`](@ref) (Near Optimal Centering) optimizations require strictly convex risk functions and are only compatible with `SimpleSD`
"""
abstract type SDSquaredFormulation <: SDFormulation end

"""
    struct QuadSD <: SDSquaredFormulation end

Explicit quadratic formulation for variance calculation in portfolio optimization.

# Risk Expression

The risk is computed as `dot(w, sigma, w)` where:

  - `w`: N×1 vector of portfolio weights
  - `sigma`: N×N covariance matrix

# Use Cases

Suitable when direct quadratic form optimization is desired/needed or when specific solver requirements necessitate explicit quadratic expressions.
"""
struct QuadSD <: SDSquaredFormulation end

"""
    struct SOCSD <: SDSquaredFormulation end

Second-Order Cone (SOC) formulation for standard deviation calculation in portfolio optimization.

# Implementation Details

  - Uses `MOI.SecondOrderCone` constraints
  - Defines a standard deviation variable `dev`
  - Sets risk expression as `sd_risk = dev^2`

# Advantages

  - Can be more numerically stable than explicit quadratic formulation
  - Often more efficient for larger portfolios
"""
struct SOCSD <: SDSquaredFormulation end

"""
    struct SimpleSD <: SDFormulation end

Linear standard deviation formulation using Second-Order Cone constraints.

# Implementation Details

  - Uses `MOI.SecondOrderCone` constraints
  - Defines standard deviation variable `dev`
  - Sets risk expression as `sd_risk = dev`

# Key Features

  - Compatible with [`NOC`](@ref) optimizations due to them requiring strictly convex risk functions
  - Provides direct standard deviation optimization rather than variance
"""
struct SimpleSD <: SDFormulation end

"""
    mutable struct SD <: RiskMeasure

Standard Deviation risk measure implementation for portfolio optimization.

# Fields

  - `settings::RMSettings`: Risk measure configuration settings

  - `formulation::SDFormulation`: Strategy for standard deviation/variance calculation
  - `sigma::Union{AbstractMatrix, Nothing}`: Optional covariance matrix

      + If `nothing`: Uses the covariance matrix from [`Portfolio`](@ref)/[`HCPortfolio`](@ref)
      + Otherwise: Uses the provided matrix

# Validation

  - When setting `sigma`, the matrix must be square (N×N)
  - Includes runtime dimension checks for covariance matrix

# Examples

```julia
# Basic usage with default settings
sd_risk = SD()

# Custom configuration with specific covariance matrix
my_sigma = [1.0 0.2; 0.2 1.0]
sd_risk = SD(; settings = RMSettings(; scale = 2.0), formulation = SOCSD(),
             sigma = my_sigma)

# Using portfolio's built-in covariance
sd_risk = SD(; formulation = QuadSD(), sigma = nothing)
```
"""
mutable struct SD <: RiskMeasure
    settings::RMSettings
    formulation::SDFormulation
    sigma::Union{AbstractMatrix, Nothing}
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

"""
    mutable struct MAD <: RiskMeasure

Mean Absolute Deviation risk measure implementation.

# Fields

  - `settings::RMSettings`: Configuration settings for the risk measure
  - `w::Union{<:AbstractWeights, Nothing}`: Optional T×1 vector of weights for expected return calculation
  - `mu::Union{<:AbstractVector, Nothing}`: Optional N×1 vector of expected asset returns

# Notes

  - If `mu` is `nothing`, the implementation uses expected returns from the Portfolio instance.

# Examples

```julia
# Basic usage with default settings
mad = MAD()

# Custom configuration
weights = ones(10) ./ 10  # Equal weights
returns = rand(10)        # Sample returns
mad = MAD(; settings = RMSettings(; scale = 2.0), w = weights, mu = returns)
```
"""
mutable struct MAD <: RiskMeasure
    settings::RMSettings
    w::Union{<:AbstractWeights, Nothing}
    mu::Union{<:AbstractVector, Nothing}
end
function MAD(; settings::RMSettings = RMSettings(),
             w::Union{<:AbstractWeights, Nothing} = nothing,
             mu::Union{<:AbstractVector, Nothing} = nothing)
    return MAD(settings, w, mu)
end

"""
    mutable struct SSD{T1 <: Real} <: RiskMeasure

Semi Standard Deviation risk measure implementation.

# Type Parameters

  - `T1`: Numeric type for the target threshold

# Fields

  - `settings::RMSettings`: Configuration settings for the risk measure
  - `target::T1`: Minimum return threshold for downside classification
  - `w::Union{<:AbstractWeights, Nothing}`: Optional T×1 vector of weights for expected return calculation
  - `mu::Union{<:AbstractVector, Nothing}`: Optional N×1 vector of expected asset returns

# Notes

  - Measures deviation only for returns below the target threshold
  - Uses Portfolio's expected returns if `mu` is `nothing`

# Examples

```julia
# Basic usage with default settings (target = 0.0)
ssd = SSD()

# Custom configuration with specific target
ssd = SSD(; settings = RMSettings(; scale = 1.5), target = 0.02,  # 2% minimum return threshold
          w = weights, mu = returns)
```
"""
mutable struct SSD{T1 <: Real} <: RiskMeasure
    settings::RMSettings
    target::T1
    w::Union{<:AbstractWeights, Nothing}
    mu::Union{<:AbstractVector, Nothing}
end
function SSD(; settings::RMSettings = RMSettings(), target::Real = 0.0,
             w::Union{<:AbstractWeights, Nothing} = nothing,
             mu::Union{<:AbstractVector, Nothing} = nothing)
    return SSD{typeof(target)}(settings, target, w, mu)
end

"""
    mutable struct FLPM{T1 <: Real} <: RiskMeasure

First Lower Partial Moment (Omega ratio) risk measure.

# Type Parameters

  - `T1`: Numeric type for the target threshold

# Fields

  - `settings::RMSettings`: Configuration settings for the risk measure
  - `target::T1`: Minimum return threshold for downside classification

# Notes

  - Used in Omega ratio calculations
  - Measures expected shortfall below target return

# Examples

```julia
# Default configuration (target = 0.0)
flpm = FLPM()

# Custom target return
flpm = FLPM(; target = 0.01)  # 1% minimum return threshold
```
"""
mutable struct FLPM{T1 <: Real} <: RiskMeasure
    settings::RMSettings
    target::T1
end
function FLPM(; settings::RMSettings = RMSettings(), target::Real = 0.0)
    return FLPM{typeof(target)}(settings, target)
end

"""
    mutable struct SLPM{T1 <: Real} <: RiskMeasure

Second Lower Partial Moment (Sortino ratio) risk measure.

# Type Parameters

  - `T1`: Numeric type for the target threshold

# Fields

  - `settings::RMSettings`: Configuration settings for the risk measure
  - `target::T1`: Minimum return threshold for downside classification

# Notes

  - Used in Sortino ratio calculations
  - Measures variance of returns below target threshold

# Examples

```julia
# Default configuration (target = 0.0)
slpm = SLPM()

# Custom settings
slpm = SLPM(; settings = RMSettings(; scale = 2.0), target = 0.005)
```
"""
mutable struct SLPM{T1 <: Real} <: RiskMeasure
    settings::RMSettings
    target::T1
end
function SLPM(; settings::RMSettings = RMSettings(), target::Real = 0.0)
    return SLPM{typeof(target)}(settings, target)
end

"""
    struct WR <: RiskMeasure

Worst Realization risk measure.

# Fields

  - `settings::RMSettings`: Configuration settings for the risk measure

# Notes

  - Considers the worst historical return as the risk measure
  - Useful for extremely conservative risk assessment

# Examples

```julia
# Basic usage
wr = WR()

# Custom settings
wr = WR(; settings = RMSettings(; scale = 1.5))
```
"""
struct WR <: RiskMeasure
    settings::RMSettings
end
function WR(; settings::RMSettings = RMSettings())
    return WR(settings)
end

"""
    mutable struct CVaR{T1 <: Real} <: RiskMeasure

Conditional Value at Risk (Expected Shortfall) risk measure.

# Type Parameters

  - `T1`: Numeric type for the significance level

# Fields

  - `settings::RMSettings`: Configuration settings for the risk measure
  - `alpha::T1`: Significance level, must be in (0,1)

# Notes

  - Measures expected loss in the worst α% of cases
  - Input validation ensures 0 < α < 1

# Examples

```julia
# Default configuration (α = 0.05)
cvar = CVaR()

# Custom significance level
cvar = CVaR(; settings = RMSettings(; scale = 1.0), alpha = 0.01)
```
"""
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

"""
    mutable struct EVaR{T1 <: Real} <: RiskMeasure

Entropic Value at Risk risk measure.

# Type Parameters

  - `T1`: Numeric type for the significance level

# Fields

  - `settings::RMSettings`: Configuration settings
  - `alpha::T1`: Significance level, must be in (0,1)
  - `solvers::Union{<:AbstractDict, Nothing}`: Optional JuMP-compatible solvers for exponential cone problems

# Notes

  - Requires solver capability for exponential cone problems
  - Uses [`Portfolio`](@ref)/[`HCPortfolio`](@ref) solvers if `solvers` is `nothing`
  - Input validation ensures 0 < α < 1

# Examples

```julia
# Default configuration
evar = EVaR()

# Custom configuration with specific solver
evar = EVaR(; alpha = 0.025,  # 2.5% significance level
            solvers = Dict("solver" => my_solver))
```
"""
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

"""
    mutable struct RLVaR{T1 <: Real, T2 <: Real} <: RiskMeasure

Relativistic Value at Risk risk measure.

# Type Parameters

  - `T1`: Numeric type for the significance level
  - `T2`: Numeric type for the relativistic deformation parameter

# Fields

  - `settings::RMSettings`: Configuration settings
  - `alpha::T1`: Significance level, must be in (0,1)
  - `kappa::T2`: Relativistic deformation parameter, must be in (0,1)
  - `solvers::Union{<:AbstractDict, Nothing}`: Optional JuMP-compatible solvers for 3D power cone problems

# Notes

  - Requires solver capability for 3D power cone problems
  - Uses [`Portfolio`](@ref)/[`HCPortfolio`](@ref) solvers if `solvers` is `nothing`
  - Input validation ensures both α and κ are in (0,1)

# Examples

```julia
# Default configuration
rlvar = RLVaR()

# Custom configuration
rlvar = RLVaR(; alpha = 0.05,   # 5% significance level
              kappa = 0.3,    # Deformation parameter
              solvers = Dict("solver" => my_solver))
```
"""
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

"""
    struct MDD <: RiskMeasure

Maximum Drawdown (Calmar ratio) risk measure for uncompounded returns.

# Fields

  - `settings::RMSettings`: Configuration settings for the risk measure

# Notes

  - Measures the largest peak-to-trough decline in uncompounded returns

# Examples

```julia
# Basic usage
mdd = MDD()

# Custom settings
mdd = MDD(; settings = RMSettings(; scale = 2.0))
```
"""
struct MDD <: RiskMeasure
    settings::RMSettings
end
function MDD(; settings::RMSettings = RMSettings())
    return MDD(settings)
end

"""
    struct ADD <: RiskMeasure

Average Drawdown risk measure for uncompounded returns.

# Fields

  - `settings::RMSettings`: Configuration settings for the risk measure

# Notes

  - Measures the average of all peak-to-trough declines
  - Provides a more balanced view than Maximum Drawdown

# Examples

```julia
# Basic usage
add = ADD()

# Custom settings
add = ADD(; settings = RMSettings(; scale = 1.5))
```
"""
struct ADD <: RiskMeasure
    settings::RMSettings
end
function ADD(; settings::RMSettings = RMSettings())
    return ADD(settings)
end

"""
    mutable struct CDaR{T1 <: Real} <: RiskMeasure

Conditional Drawdown at Risk risk measure.

# Type Parameters

  - `T1`: Numeric type for the significance level

# Fields

  - `settings::RMSettings`: Configuration settings for the risk measure
  - `alpha::T1`: Significance level, must be in (0,1)

# Notes

  - Measures expected loss in the worst α% of cases
  - Input validation ensures 0 < α < 1

# Examples

```julia
# Default configuration (α = 0.05)
cdar = CDaR()

# Custom significance level
cdar = CDaR(; settings = RMSettings(; scale = 1.0), alpha = 0.01) # 1% significance level
```
"""
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

"""
```
@kwdef mutable struct UCI{T1 <: Real} <: RiskMeasure
    settings::RMSettings = RMSettings()
end
```

Defines the Ulcer Index [`_UCI`](@ref) risk measure.

# Parameters

  - `settings`: risk measure settings [`RMSettings`](@ref).
"""
struct UCI <: RiskMeasure
    settings::RMSettings
end
function UCI(; settings::RMSettings = RMSettings())
    return UCI(settings)
end

"""
    mutable struct EDaR{T1 <: Real} <: RiskMeasure

Entropic Drawdown at Risk risk measure.

# Type Parameters

  - `T1`: Numeric type for the significance level

# Fields

  - `settings::RMSettings`: Configuration settings
  - `alpha::T1`: Significance level, must be in (0,1)
  - `solvers::Union{<:AbstractDict, Nothing}`: Optional JuMP-compatible solvers for exponential cone problems

# Notes

  - Requires solver capability for exponential cone problems
  - Uses [`Portfolio`](@ref)/[`HCPortfolio`](@ref) solvers if `solvers` is `nothing`
  - Input validation ensures 0 < α < 1

# Examples

```julia
# Default configuration
edar = EDaR()

# Custom configuration with specific solver
edar = EDaR(; alpha = 0.025,  # 2.5% significance level
            solvers = Dict("solver" => my_solver))
```
"""
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

"""
    mutable struct RLDaR{T1 <: Real, T2 <: Real} <: RiskMeasure

Relativistic Drawdown at Risk risk measure.

# Type Parameters

  - `T1`: Numeric type for the significance level
  - `T2`: Numeric type for the relativistic deformation parameter

# Fields

  - `settings::RMSettings`: Configuration settings
  - `alpha::T1`: Significance level, must be in (0,1)
  - `kappa::T2`: Relativistic deformation parameter, must be in (0,1)
  - `solvers::Union{<:AbstractDict, Nothing}`: Optional JuMP-compatible solvers for 3D power cone problems

# Notes

  - Requires solver capability for 3D power cone problems
  - Uses [`Portfolio`](@ref)/[`HCPortfolio`](@ref) solvers if `solvers` is `nothing`
  - Input validation ensures both α and κ are in (0,1)

# Examples

```julia
# Default configuration
rldar = RLDaR()

# Custom configuration
rldar = RLDaR(; alpha = 0.05,   # 5% significance level
              kappa = 0.3,    # 30% Deformation parameter
              solvers = Dict("solver" => my_solver))
```
"""
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

"""
    mutable struct Kurt <: RiskMeasure

Square root kurtosis risk measure implementation for portfolio optimization.

# Fields

  - `settings::RMSettings`: Risk measure configuration settings

  - `w::Union{<:AbstractWeights, Nothing}`: Optional T×1 vector of weights for expected return calculation
  - `kt::Union{AbstractMatrix, Nothing}`: Optional cokurtosis matrix

      + If `nothing`: Uses the cokurtosis matrix from [`Portfolio`](@ref)/[`HCPortfolio`](@ref)
      + Otherwise: Uses the provided matrix

# Validation

  - When setting `kt`, the matrix must be square (N^2×N^2)
  - Includes runtime dimension checks for cokurtosis matrix

# Examples

```julia
# Basic usage with default settings
kurt = Kurt()

# Custom configuration with specific cokurtosis matrix
my_kt = [1.0 0.2; 0.2 1.0]
kurt = Kurt(; settings = RMSettings(; scale = 2.0), kt = my_kt)

# Using portfolio's built-in cokurtosis matrix
kurt = Kurt(; kt = nothing)
```
"""
mutable struct Kurt <: RiskMeasure
    settings::RMSettings
    w::Union{<:AbstractWeights, Nothing}
    kt::Union{<:AbstractMatrix, Nothing}
end
function Kurt(; settings::RMSettings = RMSettings(),
              w::Union{<:AbstractWeights, Nothing} = nothing,
              kt::Union{<:AbstractMatrix, Nothing} = nothing)
    if !isnothing(kt)
        @smart_assert(size(kt, 1) == size(kt, 2))
    end
    return Kurt(settings, w, kt)
end

"""
    mutable struct SKurt{T1 <: Real} <: RiskMeasure

Square root semikurtosis risk measure implementation for portfolio optimization.

# Type Parameters

  - `T1`: Numeric type for the target threshold

# Fields

  - `settings::RMSettings`: Risk measure configuration settings

  - `target::T1`: Minimum return threshold for downside classification
  - `w::Union{<:AbstractWeights, Nothing}`: Optional T×1 vector of weights for expected return calculation
  - `kt::Union{AbstractMatrix, Nothing}`: Optional cokurtosis matrix

      + If `nothing`: Uses the cokurtosis matrix from [`Portfolio`](@ref)/[`HCPortfolio`](@ref)
      + Otherwise: Uses the provided matrix

# Notes

  - Measures deviation only for returns below the target threshold

# Validation

  - When setting `kt`, the matrix must be square (N^2×N^2)
  - Includes runtime dimension checks for cokurtosis matrix

# Examples

```julia
# Basic usage with default settings
skurt = SKurt()

# Custom configuration with specific cokurtosis matrix
my_kt = [1.0 0.2; 0.2 1.0]
skurt = SKurt(; settings = RMSettings(; scale = 2.0), kt = my_kt)

# Using portfolio's built-in cokurtosis matrix
skurt = SKurt(; kt = nothing, target = 0.015) # 1.5% minimum return threshold
```
"""
mutable struct SKurt{T1 <: Real} <: RiskMeasure
    settings::RMSettings
    target::T1
    w::Union{<:AbstractWeights, Nothing}
    kt::Union{<:AbstractMatrix, Nothing}
end
function SKurt(; settings::RMSettings = RMSettings(), target::Real = 0.0,
               w::Union{<:AbstractWeights, Nothing} = nothing,
               kt::Union{<:AbstractMatrix, Nothing} = nothing)
    if !isnothing(kt)
        @smart_assert(size(kt, 1) == size(kt, 2))
    end
    return SKurt{typeof(target)}(settings, target, w, kt)
end

"""
```
@kwdef mutable struct RG{T1 <: Real} <: RiskMeasure
    settings::RMSettings = RMSettings()
end
```

Defines the Range [`_RG`](@ref) risk measure.

# Parameters

  - `settings`: risk measure settings [`RMSettings`](@ref).
"""
struct RG <: RiskMeasure
    settings::RMSettings
end
function RG(; settings::RMSettings = RMSettings())
    return RG(settings)
end

"""
```
@kwdef mutable struct CVaRRG{T1 <: Real} <: RiskMeasure
    settings::RMSettings = RMSettings()
end
```

Defines the Conditional Value at Risk Range [`_CVaRRG`](@ref) risk measure.

# Parameters

  - `settings`: risk measure settings [`RMSettings`](@ref).
  - `alpha`: significance level of CVaR losses, `alpha ∈ (0, 1)`.
  - `beta`: significance level of CVaR gains, `beta ∈ (0, 1)`.
"""
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

"""
```
@kwdef mutable struct OWASettings{T1<:AbstractVector{<:Real}}
    approx::Bool = true
    p::T1 = Float64[2, 3, 4, 10, 50]
end
```

Defines the settings for Ordered Weight Array (OWA) risk measures.

# Parameters

  - `approx`:

      + if `true`: use the approximate formulation based on power cone norms.

  - `p`: only used when `approx = true`. Vector of the order of p-norms to use in the approximation.
"""
mutable struct OWASettings{T1 <: AbstractVector{<:Real}}
    approx::Bool
    p::T1
end
function OWASettings(; approx::Bool = true,
                     p::AbstractVector{<:Real} = Float64[2, 3, 4, 10, 50])
    return OWASettings{typeof(p)}(approx, p)
end

"""
```
@kwdef struct GMD <: RiskMeasure
    settings::RMSettings = RMSettings()
    owa::OWASettings = OWASettings()
end
```

Defines the Gini Mean Difference [`_GMD`](@ref) risk measure.

# Parameters

  - `settings`: risk measure settings [`RMSettings`](@ref).
  - `owa`: OWA risk measure settings [`OWASettings`](@ref).
"""
struct GMD <: RiskMeasure
    settings::RMSettings
    owa::OWASettings
end
function GMD(; settings::RMSettings = RMSettings(), owa::OWASettings = OWASettings())
    return GMD(settings, owa)
end

"""
```
@kwdef mutable struct TG{T1 <: Real, T2 <: Real, T3 <: Integer} <: RiskMeasure
    settings::RMSettings
    owa::OWASettings
    alpha_i::T1 = 0.0001
    alpha::T2 = 0.05
    a_sim::T3 = 100
end
```

Defines the Tail Gini Difference [`_TG`](@ref) risk measure.

# Parameters

  - `settings`: risk measure settings [`RMSettings`](@ref).
  - `owa`: OWA risk measure settings [`OWASettings`](@ref).
  - `alpha_i`: start value of the significance level of CVaR losses, `0 < alpha_i < alpha < 1`.
  - `alpha`: end value of the significance level of CVaR losses, `alpha ∈ (0, 1)`.
  - `a_sim`: number of CVaRs to approximate the Tail Gini losses, `a_sim > 0`.
"""
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

"""
```
mutable struct TGRG{T1 <: Real, T2 <: Real, T3 <: Integer, T4 <: Real, T5 <: Real, T6 <: Integer} <: RiskMeasure
    settings::RMSettings
    owa::OWASettings
    alpha_i::T1 = 0.0001
    alpha::T2 = 0.05
    a_sim::T3 = 100
    beta_i::T4 = 0.0001
    beta::T5 = 0.05
    b_sim::T6 = 100
end
```

Defines the Tail Gini Difference Range [`_TGRG`](@ref) risk measure.

# Parameters

  - `settings`: risk measure settings [`RMSettings`](@ref).
  - `owa`: OWA risk measure settings [`OWASettings`](@ref).
  - `alpha_i`: start value of the significance level of CVaR losses, `0 < alpha_i < alpha < 1`.
  - `alpha`: end value of the significance level of CVaR losses, `alpha ∈ (0, 1)`.
  - `a_sim`: number of CVaRs to approximate the Tail Gini losses, `a_sim > 0`.
  - `beta_i`: start value of the significance level of CVaR gains, `0 < beta_i < beta < 1`.
  - `beta`: end value of the significance level of CVaR gains, `beta ∈ (0, 1)`.
  - `b_sim`: number of CVaRs to approximate the Tail Gini gains, `b_sim > 0`.
"""
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

"""
```
@kwdef struct OWA <: RiskMeasure
    settings::RMSettings = RMSettings()
    owa::OWASettings = OWASettings()
    w::Union{<:AbstractVector, Nothing} = nothing
end
```

Defines the generic Ordered Weight Array [`_OWA`](@ref) risk measure.

# Parameters

  - `settings`: risk measure settings [`RMSettings`](@ref).

  - `owa`: OWA risk measure settings [`OWASettings`](@ref).
  - `w`: optional `T×1` vector of ordered weights.

      + if `nothing`: use [`owa_gmd`](@ref) to compute the weights.
      + else: use this value.
"""
mutable struct OWA <: RiskMeasure
    settings::RMSettings
    owa::OWASettings
    w::Union{<:AbstractVector, Nothing}
end
function OWA(; settings::RMSettings = RMSettings(), owa::OWASettings = OWASettings(),
             w::Union{<:AbstractVector, Nothing} = nothing)
    return OWA(settings, owa, w)
end

"""
```
@kwdef struct dVar <: RiskMeasure
    settings::RMSettings = RMSettings()
end
```

Define the Brownian Distance Variance [`_dVar`](@ref) risk measure.

# Parameters

  - `settings`: risk measure settings [`RMSettings`](@ref).
"""
struct dVar <: RiskMeasure
    settings::RMSettings
end
function dVar(; settings::RMSettings = RMSettings())
    return dVar(settings)
end

"""
```
@kwdef struct Skew <: RiskMeasure
    settings::RMSettings = RMSettings()
end
```

Define the Quadratic Skewness [`_Skew`](@ref) risk measure.

# Parameters

  - `settings`: risk measure settings [`RMSettings`](@ref).
"""
struct Skew <: RiskMeasure
    settings::RMSettings
end
function Skew(; settings::RMSettings = RMSettings())
    return Skew(settings)
end

"""
```
@kwdef struct SSkew <: RiskMeasure
    settings::RMSettings = RMSettings()
end
```

Define the Quadratic SSkewness [`_Skew`](@ref) risk measure.

# Parameters

  - `settings`: risk measure settings [`RMSettings`](@ref).
"""
struct SSkew <: RiskMeasure
    settings::RMSettings
end
function SSkew(; settings::RMSettings = RMSettings())
    return SSkew(settings)
end

"""
```
@kwdef mutable struct Variance{T1 <: Union{AbstractMatrix, Nothing}} <: HCRiskMeasure
    settings::HCRMSettings = HCRMSettings()
    sigma::Union{<:AbstractMatrix, Nothing} = nothing
end
```

Defines the Variance [`_Variance`](@ref) risk measure.

# Parameters

  - `settings`: risk measure settings [`RMSettings`](@ref).

  - `sigma`: optional `N×N` covariance matrix.

      + if `nothing`: use the covariance matrix stored in the instance of [`Portfolio`](@ref) or [`HCPortfolio`](@ref).
      + else: use this one.
"""
mutable struct Variance{T1 <: Union{AbstractMatrix, Nothing}} <: HCRiskMeasure
    settings::HCRMSettings
    sigma::T1
end
function Variance(; settings::HCRMSettings = HCRMSettings(),
                  sigma::Union{<:AbstractMatrix, Nothing} = nothing)
    if !isnothing(sigma)
        @smart_assert(size(sigma, 1) == size(sigma, 2))
    end
    return Variance{Union{<:AbstractMatrix, Nothing}}(settings, sigma)
end
function Base.setproperty!(obj::Variance, sym::Symbol, val)
    if sym == :sigma
        if !isnothing(val)
            @smart_assert(size(val, 1) == size(val, 2))
        end
    end
    return setfield!(obj, sym, val)
end

"""
```
@kwdef mutable struct SVariance{T1 <: Real} <: HCRiskMeasure
    settings::HCRMSettings = HCRMSettings()
    target::T1 = 0.0
    w::Union{<:AbstractWeights, Nothing} = nothing
    mu::Union{<:AbstractVector, Nothing} = nothing
end
```

Defines the Semi Variance [`_SVariance`](@ref) risk measure.

# Parameters

  - `settings`: risk measure settings [`HCRMSettings`](@ref).

  - `target`: minimum return threshold for classifying downside returns.
  - `w`: optional `T×1` vector of weights for computing the expected return in [`_SVariance`](@ref).
  - `mu`: optional `N×1` vector of expected asset returns.

      + if `nothing`: use the expected asset returns stored in the instance of [`Portfolio`](@ref).
      + else: use this one.
"""
mutable struct SVariance{T1 <: Real} <: HCRiskMeasure
    settings::HCRMSettings
    target::T1
    w::Union{<:AbstractWeights, Nothing}
    mu::Union{<:AbstractVector, Nothing}
end
function SVariance(; settings::HCRMSettings = HCRMSettings(), target::Real = 0.0,
                   w::Union{<:AbstractWeights, Nothing} = nothing,
                   mu::Union{<:AbstractVector, Nothing} = nothing)
    return SVariance{typeof(target)}(settings, target, w, mu)
end

"""
```
@kwdef mutable struct VaR{T1 <: Real} <: HCRiskMeasure
    settings::HCRMSettings = HCRMSettings()
    alpha::T1 = 0.05
end
```

Defines the Value at Risk [`_VaR`](@ref) risk measure.

# Parameters

  - `settings`: risk measure settings [`HCRMSettings`](@ref).
  - `alpha`: significance level, `alpha ∈ (0, 1)`.
"""
mutable struct VaR{T1 <: Real} <: HCRiskMeasure
    settings::HCRMSettings
    alpha::T1
end
function VaR(; settings::HCRMSettings = HCRMSettings(), alpha::Real = 0.05)
    @smart_assert(zero(alpha) < alpha < one(alpha))
    return VaR{typeof(alpha)}(settings, alpha)
end
function Base.setproperty!(obj::VaR, sym::Symbol, val)
    if sym == :alpha
        @smart_assert(zero(val) < val < one(val))
    end
    return setfield!(obj, sym, val)
end

"""
```
@kwdef mutable struct DaR{T1 <: Real} <: HCRiskMeasure
    settings::HCRMSettings = HCRMSettings()
    alpha::T1 = 0.05
end
```

Defines the Drawdown at Risk of uncompounded cumulative returns [`_DaR`](@ref) risk measure.

# Parameters

  - `settings`: risk measure settings [`HCRMSettings`](@ref).
  - `alpha`: significance level, `alpha ∈ (0, 1)`.
"""
mutable struct DaR{T1 <: Real} <: HCRiskMeasure
    settings::HCRMSettings
    alpha::T1
end
function DaR(; settings::HCRMSettings = HCRMSettings(), alpha::Real = 0.05)
    @smart_assert(zero(alpha) < alpha < one(alpha))
    return DaR{typeof(alpha)}(settings, alpha)
end
function Base.setproperty!(obj::DaR, sym::Symbol, val)
    if sym == :alpha
        @smart_assert(zero(val) < val < one(val))
    end
    return setfield!(obj, sym, val)
end

"""
```
@kwdef mutable struct DaR_r{T1 <: Real} <: HCRiskMeasure
    settings::HCRMSettings = HCRMSettings()
    alpha::T1 = 0.05
end
```

Defines the Drawdown at Risk of compounded cumulative returns [`_DaR_r`](@ref) risk measure.

# Parameters

  - `settings`: risk measure settings [`HCRMSettings`](@ref).
  - `alpha`: significance level, `alpha ∈ (0, 1)`.
"""
mutable struct DaR_r{T1 <: Real} <: HCRiskMeasure
    settings::HCRMSettings
    alpha::T1
end
function DaR_r(; settings::HCRMSettings = HCRMSettings(), alpha::Real = 0.05)
    @smart_assert(zero(alpha) < alpha < one(alpha))
    return DaR_r{typeof(alpha)}(settings, alpha)
end
function Base.setproperty!(obj::DaR_r, sym::Symbol, val)
    if sym == :alpha
        @smart_assert(zero(val) < val < one(val))
    end
    return setfield!(obj, sym, val)
end

"""
```
@kwdef mutable struct MDD_r{T1 <: Real} <: HCRiskMeasure
    settings::HCRMSettings = HCRMSettings()
    alpha::T1 = 0.05
end
```

Defines the Maximum Drawdown (Calmar ratio) of compounded cumulative returns [`_MDD_r`](@ref) risk measure.

# Parameters

  - `settings`: risk measure settings [`HCRMSettings`](@ref).
"""
struct MDD_r <: HCRiskMeasure
    settings::HCRMSettings
end
function MDD_r(; settings::HCRMSettings = HCRMSettings())
    return MDD_r(settings)
end

"""
```
@kwdef mutable struct ADD_r{T1 <: Real} <: HCRiskMeasure
    settings::HCRMSettings = HCRMSettings()
    alpha::T1 = 0.05
end
```

Defines the Average Drawdown of compounded cumulative returns [`_ADD_r`](@ref) risk measure.

# Parameters

  - `settings`: risk measure settings [`HCRMSettings`](@ref).
"""
struct ADD_r <: HCRiskMeasure
    settings::HCRMSettings
end
function ADD_r(; settings::HCRMSettings = HCRMSettings())
    return ADD_r(settings)
end

"""
```
@kwdef mutable struct CDaR_r{T1 <: Real} <: HCRiskMeasure
    settings::HCRMSettings = HCRMSettings()
    alpha::T1 = 0.05
end
```

Defines the Conditional Drawdown at Risk of compounded cumulative returns [`_CDaR_r`](@ref) risk measure.

# Parameters

  - `settings`: risk measure settings [`HCRMSettings`](@ref).
  - `alpha`: significance level, `alpha ∈ (0, 1)`.
"""
mutable struct CDaR_r{T1 <: Real} <: HCRiskMeasure
    settings::HCRMSettings
    alpha::T1
end
function CDaR_r(; settings::HCRMSettings = HCRMSettings(), alpha::Real = 0.05)
    @smart_assert(zero(alpha) < alpha < one(alpha))
    return CDaR_r{typeof(alpha)}(settings, alpha)
end
function Base.setproperty!(obj::CDaR_r, sym::Symbol, val)
    if sym == :alpha
        @smart_assert(zero(val) < val < one(val))
    end
    return setfield!(obj, sym, val)
end

"""
```
@kwdef mutable struct UCI_r{T1 <: Real} <: HCRiskMeasure
    settings::HCRMSettings = HCRMSettings()
    alpha::T1 = 0.05
end
```

Defines the Ulcer Index of compounded cumulative returns [`_UCI_r`](@ref) risk measure.

# Parameters

  - `settings`: risk measure settings [`HCRMSettings`](@ref).
"""
struct UCI_r <: HCRiskMeasure
    settings::HCRMSettings
end
function UCI_r(; settings::HCRMSettings = HCRMSettings())
    return UCI_r(settings)
end

"""
```
@kwdef mutable struct EDaR_r{T1 <: Real} <: HCRiskMeasure
    settings::HCRMSettings = HCRMSettings()
    alpha::T1 = 0.05
    solvers::Union{<:AbstractDict, Nothing} = nothing
end
```

Defines the Entropic Drawdown at Risk of compounded cumulative returns [`_EDaR_r`](@ref) risk measure.

# Parameters

  - `settings`: risk measure settings [`HCRMSettings`](@ref).

  - `alpha`: significance level, `alpha ∈ (0, 1)`.
  - `solvers`: optional abstract dict containing the a JuMP-compatible solver capable of solving 3D power cone problems.

      + if `nothing`: use the solvers stored in the instance of [`Portfolio`](@ref) or [`HCPortfolio`](@ref).
      + else: use these ones.
"""
mutable struct EDaR_r{T1 <: Real} <: HCRiskMeasure
    settings::HCRMSettings
    alpha::T1
    solvers::Union{<:AbstractDict, Nothing}
end
function EDaR_r(; settings::HCRMSettings = HCRMSettings(), alpha::Real = 0.05,
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

"""
```
@kwdef mutable struct RLDaR_r{T1 <: Real} <: RiskMeasure
    settings::RMSettings = RMSettings()
    alpha::T1 = 0.05
    solvers::Union{<:AbstractDict, Nothing} = nothing
end
```

Defines the Relativistic Drawdown at Risk of compounded cumulative returns [`_RLDaR_r`](@ref) risk measure.

# Parameters

  - `settings`: risk measure settings [`RMSettings`](@ref).

  - `alpha`: significance level, `alpha ∈ (0, 1)`.
  - `kappa`: relativistic deformation parameter, `κ ∈ (0, 1)`.
  - `solvers`: optional abstract dict containing the a JuMP-compatible solver capable of solving 3D power cone problems.

      + if `nothing`: use the solvers stored in the instance of [`Portfolio`](@ref) or [`HCPortfolio`](@ref).
      + else: use these ones.
"""
mutable struct RLDaR_r{T1 <: Real, T2 <: Real} <: HCRiskMeasure
    settings::HCRMSettings
    alpha::T1
    kappa::T2
    solvers::Union{<:AbstractDict, Nothing}
end
function RLDaR_r(; settings::HCRMSettings = HCRMSettings(), alpha::Real = 0.05, kappa = 0.3,
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

"""
```
@kwdef struct Equal <: HCRiskMeasure
    settings::HCRMSettings = HCRMSettings()
end
```

Defines the Equal risk measure, where risk is allocated evenly among a group of assets.

# Parameters

  - `settings`: risk measure settings [`HCRMSettings`](@ref).
"""
mutable struct Equal <: HCRiskMeasure
    settings::HCRMSettings
end
function Equal(; settings::HCRMSettings = HCRMSettings())
    return Equal(settings)
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
             function Base.getindex(S::$op, ::Any)
                 return S
             end
             function Base.view(S::$op, ::Any)
                 return S
             end
         end)
end

const RMSolvers = Union{EVaR, EDaR, EDaR_r, RLVaR, RLDaR, RLDaR_r}
const RMSigma = Union{SD, Variance}

export RMSettings, HCRMSettings, QuadSD, SOCSD, SimpleSD, SD, MAD, SSD, FLPM, SLPM, WR,
       CVaR, EVaR, RLVaR, MDD, ADD, CDaR, UCI, EDaR, RLDaR, Kurt, SKurt, RG, CVaRRG,
       OWASettings, GMD, TG, TGRG, OWA, dVar, Skew, SSkew, Variance, SVariance, VaR, DaR,
       DaR_r, MDD_r, ADD_r, CDaR_r, UCI_r, EDaR_r, RLDaR_r, Equal
