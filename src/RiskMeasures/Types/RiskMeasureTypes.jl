"""
    abstract type AbstractRiskMeasure end

# Description

Serves as the foundational type for all risk measurement approaches in the library.

See also: [`RiskMeasure`](@ref), [`HCRiskMeasure`](@ref).

# Type Hierarchy

  - Direct subtypes: [`RiskMeasure`](@ref), [`HCRiskMeasure`](@ref).
"""
abstract type AbstractRiskMeasure end

"""
    abstract type RiskMeasure <: AbstractRiskMeasure end

# Description

Defines the interface for risk measures that can be used in both [`Portfolio`](@ref) and [`HCPortfolio`](@ref) optimisation contexts.

See also: [`AbstractRiskMeasure`](@ref), [`RMSettings`](@ref), [`Portfolio`](@ref), [`HCPortfolio`](@ref), [`optimise!`](@ref), [`calc_risk`](@ref), [`set_rm`](@ref), [`OptimType`](@ref), [`ObjectiveFunction`](@ref).

# Type Hierarchy

# Implementation Requirements

To ensure concrete subtypes will handle both [`Portfolio`](@ref) and [`HCPortfolio`](@ref) contexts appropriately, they must implement:

  - Risk calculation method [`calc_risk`](@ref).
  - Scalar [`JuMP`](https://github.com/jump-dev/JuMP.jl) model implementation, if appropriate a vector equivalent.
  - Include a `settings::RMSettings = RMSettings()` field for configuration when optimising a [`Portfolio`](@ref).
  - If the risk calculation involves solving a [`JuMP`](https://github.com/jump-dev/JuMP.jl) model, it must include a `solvers::Union{Nothing, <:AbstractDict}` field.

# Examples

```@example
# Creating a concrete risk measure that subtypes RiskMeasure
struct CustomRisk <: RiskMeasure
    settings::RMSettings
    # implementation details
end

# Creating risk calculation method
function PortfolioOptimiser.calc_risk(risk::CustomRisk, w::AbstractVector; kwargs...)
    # implementation details
end

# Creating a scalar JuMP model implementation
function PortfolioOptimiser.set_rm(port::Portfolio, rm::CustomRisk, type::OptimType,
                                   obj::ObjectiveFunction; kwargs...)
    # implementation details
end

# Creating a vector JuMP model implementation
function PortfolioOptimiser.set_rm(port::Portfolio, rms::AbstractVector{<:CustomRisk},
                                   type::OptimType, obj::ObjectiveFunction; kwargs...)
    # implementation details
end
```
"""
abstract type RiskMeasure <: AbstractRiskMeasure end

"""
    abstract type HCRiskMeasure <: AbstractRiskMeasure end

# Description

Defines the interface for risk measures specifically designed for use with [`HCPortfolio`](@ref) optimisation.

See also: [`AbstractRiskMeasure`](@ref), [`HCRMSettings`](@ref), [`Portfolio`](@ref), [`HCPortfolio`](@ref), [`optimise!`](@ref), [`calc_risk`](@ref).

# Implementation Requirements

Concrete subtypes must implement:

  - Risk calculation method [`calc_risk`](@ref).
  - If the risk calculation involves solving a [`JuMP`](https://github.com/jump-dev/JuMP.jl) model, it must include a `solvers::Union{Nothing, <:AbstractDict}` field.

# Behaviour

  - Not compatible with standard [`Portfolio`](@ref) optimisation.

# Examples

```@example
# Creating a concrete risk measure that subtypes RiskMeasure
struct CustomHCRisk <: HCRiskMeasure
    settings::HCRMSettings
    # implementation details
end

# Creating risk calculation method
function PortfolioOptimiser.calc_risk(risk::CustomHCRisk, w::AbstractVector; kwargs...)
    # implementation details
end
```
"""
abstract type HCRiskMeasure <: AbstractRiskMeasure end

"""
    mutable struct RMSettings{T1 <: Real, T2 <: Real}

# Description

Configuration settings for [`RiskMeasure`](@ref) and [`HCRiskMeasure`](@ref).

See also: [`AbstractRiskMeasure`](@ref), [`RiskMeasure`](@ref), [`HCRiskMeasure`](@ref), [`Portfolio`](@ref), [`HCPortfolio`](@ref), [`optimise!`](@ref), [`calc_risk`](@ref), [`set_rm`](@ref).

# Fields

  - `flag::Bool = true`: controls risk inclusion to the risk expression in the optimisation objective.
  - `scale::T1 = 1.0`: risk measure scaling factor.
  - `ub::T2 = Inf`: upper bound risk constraint.

# Behaviour

## [`Portfolio`](@ref) Optimisation

With `R(w)` being a risk measure.

  - When `flag == true`: adds `scale * R(w)` to the risk expression in the optimisation objective.
  - `scale`: multiplier for this risk term in the risk expression.
  - When `ub < Inf` (exclusive to [`Portfolio`](@ref) optimisation): adds constraint `R(w) ≤ ub`.

## [`HCPortfolio`](@ref) Optimisation

  - `flag`: no effect.
  - `scale`: multiplier for this risk term in the risk expression. Always adds `scale * R(w)` to the risk expression in the optimisation objective.
  - `ub`: no effect.

## Validation

  - `scale`: must be positive.
  - `ub`: must be positive when finite.

# Notes

  - `scale`: typically used when combining different risk measures in a single optimisation.

# Examples

```@example
# Default settings
settings = RMSettings()

# Risk-averse configuration, whatever risk measure this is applied 
# to will contribute 8 * risk to the risk expression
settings = RMSettings(; scale = 8.0)

# Risk not added to the objective but constrainted
settings = RMSettings(; flag = false, ub = 0.25)
```
"""
mutable struct RMSettings{T1 <: Real, T2 <: Real}
    flag::Bool
    scale::T1
    ub::T2
end
function RMSettings(; flag::Bool = true, scale::Real = 1.0, ub::Real = Inf)
    @smart_assert(scale > zero(scale))
    if isfinite(ub)
        @smart_assert(ub > zero(ub))
    end
    return RMSettings{typeof(scale), typeof(ub)}(flag, scale, ub)
end
function Base.setproperty!(obj::RMSettings, sym::Symbol, val)
    if sym == :scale
        @smart_assert(val > zero(val))
    elseif sym == :ub
        if isfinite(val)
            @smart_assert(val > zero(val))
        end
    end
    return setfield!(obj, sym, val)
end

"""
    mutable struct HCRMSettings{T1 <: Real}

# Description

Configuration settings for [`HCRiskMeasure`](@ref).

See also: [`AbstractRiskMeasure`](@ref), [`HCRiskMeasure`](@ref), [`HCPortfolio`](@ref), [`optimise!`](@ref), [`calc_risk`](@ref).

# Fields

  - `scale::T1 = 1.0`: multiplier for this risk term in the risk expression.

# Behaviour

With `R(w)` being a risk measure.

  - `scale`: multiplier for this risk term in the risk expression. Always adds `scale * R(w)` to the risk expression in the optimisation objective.
  - Does not include flag or bounds as hierarchical optimisations cannot constrain the risk, only the weights of the assets.

## Validation

  - `scale`: must be positive.

# Notes

  - `scale`: typically used when combining different risk measures in a single optimisation.

# Examples

```@example
# Default settings
settings = HCRMSettings()

# Contribute more risk to the risk expression
settings = HCRMSettings(; scale = 2.5)
```
"""
mutable struct HCRMSettings{T1 <: Real}
    scale::T1
end
function HCRMSettings(; scale::Real = 1.0)
    @smart_assert(scale > zero(scale))
    return HCRMSettings{typeof(scale)}(scale)
end
function Base.setproperty!(obj::HCRMSettings, sym::Symbol, val)
    if sym == :scale
        @smart_assert(val > zero(val))
    end
    return setfield!(obj, sym, val)
end

"""
    abstract type SDFormulation end

# Description

Base type for implementing various approaches to Mean-Variance and standard deviation calculation strategies in [`Portfolio`](@ref) optimisation, each offering different computational and numerical properties.

See also: [`SDSquaredFormulation`](@ref), [`QuadSD`](@ref), [`SOCSD`](@ref), [`SimpleSD`](@ref), [`SD`](@ref).

# Type Hierarchy

Direct subtypes:

  - [`SDSquaredFormulation`](@ref): for quadratic expressions of the variance.
  - [`SimpleSD`](@ref): for direct standard deviation optimisation.

# Behaviour

## [`Portfolio`](@ref) Optimisation

  - Concrete subtypes define how standard deviation/variance is represented in the [`JuMP`](https://github.com/jump-dev/JuMP.jl) model.
  - Choice of formulation can significantly impact solver performance and numerical stability.
  - Each formulation may have different solver compatibility requirements.

## [`HCPortfolio`](@ref) Optimisation

  - No effect.
"""
abstract type SDFormulation end

"""
    abstract type SDSquaredFormulation <: SDFormulation end

# Description

Abstract type for Mean-Variance formulations using quadratic variance expressions for [`Portfolio`](@ref) optimisations.

See also: [`SDFormulation`](@ref), [`QuadSD`](@ref), [`SOCSD`](@ref), [`SimpleSD`](@ref), [`SD`](@ref).

# Type Hierarchy

Direct subtypes:

  - [`QuadSD`](@ref): explicit quadratic formulation of the portfolio variance.
  - [`SOCSD`](@ref): second-Order Cone (SOC) formulation of the portfolio variance.

# Mathematical Description

These formulations work with the variance form of risk:

```math
\\begin{align}
\\sigma^2 &= \\bm{w}^{\\intercal} \\mathbf{\\Sigma} \\bm{w}\\,.
\\end{align}
```

Where:

  - ``\\bm{w}`` is the `N×1` vector of portfolio weights.
  - ``\\mathbf{\\Sigma}`` is the `N×N` covariance matrix.
  - ``\\sigma^2`` is the portfolio variance.

# Behaviour

  - Produces a [`QuadExpr`](https://jump.dev/JuMP.jl/stable/api/JuMP/#QuadExpr) for `sd_risk` in the [`JuMP`](https://github.com/jump-dev/JuMP.jl) model.
  - Not compatible with [`NOC`](@ref) (Near Optimal Centering) optimisations because [`QuadExpr`](https://jump.dev/JuMP.jl/stable/api/JuMP/#QuadExpr) are not strictly convex.
  - May have different numerical stability properties compared to direct SD formulations.
  - Risk is expressed in terms of the variance (squared standard deviation).
"""
abstract type SDSquaredFormulation <: SDFormulation end

"""
    struct QuadSD <: SDSquaredFormulation end

# Description

Explicit quadratic formulation for variance-based [`Portfolio`](@ref) optimisation.

See also: [`SDFormulation`](@ref), [`SDSquaredFormulation`](@ref), [`SOCSD`](@ref), [`SimpleSD`](@ref), [`SD`](@ref).

# Mathematical Description

Implements the classical quadratic form of portfolio variance:

```math
\\begin{align}
\\underset{\\bm{w}}{\\mathrm{opt}} &\\qquad \\sigma^2\\\\
\\textrm{s.t.} &\\qquad \\sigma^2 = \\bm{w}^\\intercal \\mathbf{\\Sigma} \\bm{w}\\,.
\\end{align}
```

Where:

  - ``\\bm{w}`` is the `N×1` vector of portfolio weights.
  - ``\\mathbf{\\Sigma}`` is the `N×N` covariance matrix.
  - ``\\sigma^2`` is the portfolio variance.

# Behaviour

  - Produces a [`QuadExpr`](https://jump.dev/JuMP.jl/stable/api/JuMP/#QuadExpr) risk expression `sd_risk = dot(w, sigma, w)`.
  - Not compatible with [`NOC`](@ref) (Near Optimal Centering) optimisations because [`QuadExpr`](https://jump.dev/JuMP.jl/stable/api/JuMP/#QuadExpr) are not strictly convex.
  - No additional variables or constraints introduced.
  - Requires a solver capable of handling quadratic objectives.
  - Performance may degrade for large portfolios.

# Examples

```@example
# Using portfolio's built-in covariance
sd_risk = SD(; formulation = QuadSD())

# Custom configuration with specific covariance matrix
my_sigma = [1.0 0.2; 0.2 1.0]
sd_risk = SD(; settings = RMSettings(; scale = 2.0), formulation = QuadSD(),
             sigma = my_sigma)
```
"""
struct QuadSD <: SDSquaredFormulation end

"""
    struct SOCSD <: SDSquaredFormulation end

# Description

Second-Order Cone (SOC) formulation for variance-based [`Portfolio`](@ref) optimisation.

See also: [`SDFormulation`](@ref), [`SDSquaredFormulation`](@ref), [`QuadSD`](@ref), [`SimpleSD`](@ref), [`SD`](@ref).

# Mathematical Description

Reformulates the quadratic variance expression using second-order cone constraints:

```math
\\begin{align}
\\underset{\\bm{w}}{\\mathrm{opt}} &\\qquad \\sigma^2\\\\
\\textrm{s.t.} &\\qquad \\left\\lVert \\sqrt{\\mathbf{\\Sigma}} \\bm{w} \\right\\rVert_{2} \\leq \\sigma\\,.
\\end{align}
```

Where:

  - ``\\bm{w}`` is the `N×1` vector of portfolio weights.
  - ``\\mathbf{\\Sigma}`` is the `N×N` covariance matrix.
  - ``\\sigma^2`` is the portfolio variance.

# Behaviour

  - Uses [`SecondOrderCone`](https://jump.dev/JuMP.jl/stable/manual/constraints/#Second-order-cone-constraints) constraints.
  - Defines a standard deviation variable `dev`.
  - Produces a [`QuadExpr`](https://jump.dev/JuMP.jl/stable/api/JuMP/#QuadExpr) risk expression `sd_risk = dev^2`.
  - Not compatible with [`NOC`](@ref) (Near Optimal Centering) optimisations because [`QuadExpr`](https://jump.dev/JuMP.jl/stable/api/JuMP/#QuadExpr) are not strictly convex.
  - Often more numerically stable than direct quadratic formulation.
  - Better scaling properties for large portfolios.
  - Compatible with specialized conic solvers.
  - May introduce more variables but often leads to better solution times.
  - Particularly effective for large-scale problems.

# Examples

```@example
# Custom configuration with specific covariance matrix
# Using portfolio's built-in covariance
sd_risk = SD(; formulation = SOCSD())

my_sigma = [1.0 0.2; 0.2 1.0]
sd_risk = SD(; settings = RMSettings(; scale = 2.0), formulation = SOCSD(),
             sigma = my_sigma)
```

See also: [`SD`](@ref), [`SDSquaredFormulation`](@ref), [`QuadSD`](@ref), [`SimpleSD`](@ref).
"""
struct SOCSD <: SDSquaredFormulation end

"""
    struct SimpleSD <: SDFormulation end

# Description

Linear standard deviation formulation using Second-Order Cone constraints for [`Portfolio`](@ref) optimisations.

See also: [`SDFormulation`](@ref), [`SDSquaredFormulation`](@ref), [`SOCSD`](@ref), [`QuadSD`](@ref), [`SD`](@ref).

# Mathematical Description

Reformulates the affine standard deviation expression using second-order cone constraints:

```math
\\begin{align}
\\underset{\\bm{w}}{\\mathrm{opt}} &\\qquad \\sigma\\\\
\\textrm{s.t.} &\\qquad \\left\\lVert \\sqrt{\\mathbf{\\Sigma}} \\bm{w} \\right\\rVert_{2} \\leq \\sigma\\,.
\\end{align}
```

Where:

  - ``\\bm{w}`` is the `N×1` vector of portfolio weights.
  - ``\\mathbf{\\Sigma}`` is the `N×N` covariance matrix.
  - ``\\sigma`` is the portfolio standard deviation.

# Behaviour

  - Uses [`SecondOrderCone`](https://jump.dev/JuMP.jl/stable/manual/constraints/#Second-order-cone-constraints) constraints.
  - Defines a standard deviation variable `dev`.
  - Sets the [`AffExpr`](https://jump.dev/JuMP.jl/stable/api/JuMP/#AffExpr) risk expression `sd_risk = dev`.
  - Compatible with [`NOC`](@ref) (Near Optimal Centering) optimisations because [`AffExpr`](https://jump.dev/JuMP.jl/stable/api/JuMP/#AffExpr) are strictly convex.
  - Direct optimisation of standard deviation rather than variance.
  - Often better numerical properties than squared formulations.
  - Compatible with specialized conic solvers.
  - May provide more intuitive results as risk is in same units as returns.

# Examples

```@example
# Using portfolio's built-in covariance
sd_risk = SD(; formulation = SimpleSD())

# Custom configuration with specific covariance matrix
my_sigma = [1.0 0.2; 0.2 1.0]
sd_risk = SD(; settings = RMSettings(; scale = 2.0), formulation = SimpleSD(),
             sigma = my_sigma)
```
"""
struct SimpleSD <: SDFormulation end

"""
    mutable struct SD <: RiskMeasure

# Description

  - Measures the dispersion in the returns from the mean.

See also: [`RiskMeasure`](@ref), [`RMSettings`](@ref), [`SDFormulation`](@ref), [`SDSquaredFormulation`](@ref), [`SOCSD`](@ref), [`QuadSD`](@ref), [`SimpleSD`](@ref), [`MAD`](@ref), [`Portfolio`](@ref), [`HCPortfolio`](@ref), [`optimise!`](@ref), [`set_rm`](@ref), [`PortClass`](@ref), [`calc_risk`](@ref), [`_SD`](@ref).

## [`Portfolio`](@ref)

Implements portfolio Standard Deviation/Variance risk using configurable `formulation` strategies.

## [`HCPortfolio`](@ref)

Implements portfolio Standard Deviation risk.

# Fields

  - `settings::RMSettings = RMSettings()`: risk measure configuration settings.
  - `formulation::SDFormulation = SOCSD()`: strategy for standard deviation/variance calculation. Only affects [`Portfolio`](@ref) optimisations.
  - `sigma::Union{AbstractMatrix, Nothing} = nothing`: optional covariance matrix.

# Behaviour

## Covariance Matrix Usage

  - If `isnothing(sigma)`:

      + For [`Portfolio`](@ref): uses the covariance matrix `cov`, `fm_cov`, `bl_cov` or `blfm_cov`, depending on the `class::`[`PortClass`](@ref) parameter of [`optimise!`](@ref).
      + For [`HCPortfolio`](@ref): uses the covariance matrix `cov`.

  - If `sigma` provided: uses custom covariance matrix.

### Validation

  - When setting `sigma` at construction or runtime, the matrix must be square (`N×N`).

## Formulation Impact on [`Portfolio`](@ref) Optimisation

  - [`QuadSD`](@ref): Direct quadratic implementation of variance, [`QuadExpr`](https://jump.dev/JuMP.jl/stable/api/JuMP/#QuadExpr). Not compatible with [`NOC`](@ref) (Near Optimal Centering) optimisations because [`QuadExpr`](https://jump.dev/JuMP.jl/stable/api/JuMP/#QuadExpr) are not strictly convex.
  - [`SOCSD`](@ref): Second-order cone formulation of variance, [`QuadExpr`](https://jump.dev/JuMP.jl/stable/api/JuMP/#QuadExpr). Not compatible with [`NOC`](@ref) (Near Optimal Centering) optimisations because [`QuadExpr`](https://jump.dev/JuMP.jl/stable/api/JuMP/#QuadExpr) are not strictly convex.
  - [`SimpleSD`](@ref): Standard deviation Second-order cone constraints, [`AffExpr`](https://jump.dev/JuMP.jl/stable/api/JuMP/#AffExpr). Compatible with [`NOC`](@ref) (Near Optimal Centering) optimisations because [`AffExpr`](https://jump.dev/JuMP.jl/stable/api/JuMP/#AffExpr) are strictly convex.

# Examples

```@example
# Default settings
sd_risk = SD()

# Custom configuration with specific covariance matrix
my_sigma = [1.0 0.2; 0.2 1.0]
sd_risk = SD(; settings = RMSettings(; scale = 2.0), formulation = SOCSD(),
             sigma = my_sigma)

# Using portfolio's built-in covariance
sd_risk = SD(; formulation = QuadSD(), sigma = nothing)

# For an NOC optimisation
sd_risk = SD(; formulation = SimpleSD())
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
    return SD(settings, formulation, sigma)
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

# Description

Mean Absolute Deviation risk measure implementation.

  - Measures the dispersion in the returns from the mean.

See also: [`RiskMeasure`](@ref), [`RMSettings`](@ref), [`SD`](@ref), [`Portfolio`](@ref), [`HCPortfolio`](@ref), [`optimise!`](@ref), [`set_rm`](@ref), [`calc_risk`](@ref), [`_MAD`](@ref).

# Fields

  - `settings::RMSettings = RMSettings()`: configuration settings for the risk measure.
  - `w::Union{<:AbstractWeights, Nothing} = nothing`: optional `T×1` vector of weights for expected return calculation.
  - `mu::Union{<:AbstractVector, Nothing} = nothing`: optional `N×1` vector of expected asset returns.

# Behaviour

## [`Portfolio`](@ref) Optimisation

  - If `isnothing(mu)`: use the expected returns vector from the [`Portfolio`](@ref) instance.

## [`HCPortfolio`](@ref) Optimisation or in [`calc_risk`](@ref).

  - If `isnothing(w)`: computes the unweighted mean portfolio return.

# Examples

```@example
# Default settings
mad = MAD()

# Custom configuration
w = eweights(1:100, 0.3)  # Exponential weights for computing the portfolio mean return
mu = rand(10)             # Expected returns
mad = MAD(; settings = RMSettings(; scale = 2.0), w = w, mu = mu)
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

# Description

Semi Standard Deviation risk measure implementation.

  - Measures the standard deviation equal to or below the `target` return threshold.

See also: [`RiskMeasure`](@ref), [`RMSettings`](@ref), [`Portfolio`](@ref), [`HCPortfolio`](@ref), [`optimise!`](@ref), [`set_rm`](@ref), [`calc_risk`](@ref), [`_SSD`](@ref).

# Fields

  - `settings::RMSettings = RMSettings()`: configuration settings for the risk measure.
  - `target::T1 = 0.0`: minimum return threshold for downside classification.
  - `w::Union{<:AbstractWeights, Nothing} = nothing`: optional `T×1` vector of weights for expected return calculation.
  - `mu::Union{<:AbstractVector, Nothing} = nothing`: optional `N×1` vector of expected asset returns.

# Behaviour

## [`Portfolio`](@ref) Optimisation

  - If `isnothing(mu)`: use the expected returns vector from the [`Portfolio`](@ref) instance.

## [`HCPortfolio`](@ref) Optimisation or in [`calc_risk`](@ref).

  - If `isnothing(w)`: computes the unweighted mean portfolio return.

# Examples

```@example
# Default settings
ssd = SSD()

# Custom configuration with specific target
w = eweights(1:100, 0.3)  # Exponential weights for computing the portfolio mean return
mu = rand(10)             # Expected returns
ssd = SSD(; settings = RMSettings(; scale = 2.0), w = w, mu = mu)
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

# Description

First Lower Partial Moment (Omega ratio) risk measure.

  - Measures the dispersion equal to or below the `target` return threshold.

See also: [`RiskMeasure`](@ref), [`RMSettings`](@ref), [`SLPM`](@ref), [`Portfolio`](@ref), [`HCPortfolio`](@ref), [`optimise!`](@ref), [`set_rm`](@ref), [`calc_risk`](@ref), [`_FLPM`](@ref).

# Fields

  - `settings::RMSettings = RMSettings()`: configuration settings for the risk measure.
  - `target::T1 = 0.0`: minimum return threshold for downside classification.

# Examples

```@example
# Default settings
flpm = FLPM()

# Custom target return
flpm = FLPM(; target = 0.01)  # 1 % minimum return threshold
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

# Description

Second Lower Partial Moment (Sortino ratio) risk measure.

  - Measures the dispersion equal to or below the `target` return threshold.

See also: [`RiskMeasure`](@ref), [`RMSettings`](@ref), [`FLPM`](@ref), [`Portfolio`](@ref), [`HCPortfolio`](@ref), [`optimise!`](@ref), [`set_rm`](@ref), [`calc_risk`](@ref), [`_SLPM`](@ref).

# Fields

  - `settings::RMSettings = RMSettings()`: configuration settings for the risk measure.
  - `target::T1 = 0.0`: minimum return threshold for downside classification.

# Examples

```@example
# Default settings
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

# Description

Worst Realization/Return risk measure.

  - Useful for extremely conservative risk assessment.
  - ``\\mathrm{VaR}(\\bm{X},\\, \\alpha) \\leq \\mathrm{CVaR}(\\bm{X},\\, \\alpha) \\leq \\mathrm{EVaR}(\\bm{X},\\, \\alpha) \\leq \\mathrm{RLVaR}(\\bm{X},\\, \\alpha,\\, \\kappa) \\leq \\mathrm{WR}(\\bm{X})``.

See also: [`RiskMeasure`](@ref), [`RMSettings`](@ref), [`Portfolio`](@ref), [`HCPortfolio`](@ref), [`optimise!`](@ref), [`set_rm`](@ref), [`calc_risk`](@ref), [`_WR`](@ref), [`VaR`](@ref), [`CVaR`](@ref), [`EVaR`](@ref), [`RLVaR`](@ref).

# Fields

  - `settings::RMSettings = RMSettings()`: configuration settings for the risk measure.

# Examples

```@example
# Default settings
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

# Description

Conditional Value at Risk (Expected Shortfall) risk measure.

  - Measures expected loss in the worst `alpha %` of cases.
  - ``\\mathrm{VaR}(\\bm{X},\\, \\alpha) \\leq \\mathrm{CVaR}(\\bm{X},\\, \\alpha) \\leq \\mathrm{EVaR}(\\bm{X},\\, \\alpha) \\leq \\mathrm{RLVaR}(\\bm{X},\\, \\alpha,\\, \\kappa) \\leq \\mathrm{WR}(\\bm{X})``.

See also: [`RiskMeasure`](@ref), [`RMSettings`](@ref), [`Portfolio`](@ref), [`HCPortfolio`](@ref), [`optimise!`](@ref), [`set_rm`](@ref), [`calc_risk`](@ref), [`_CVaR`](@ref), [`VaR`](@ref), [`EVaR`](@ref), [`RLVaR`](@ref), [`WR`](@ref), [`DaR`](@ref), [`DaR_r`](@ref), [`CDaR`](@ref), [`CDaR_r`](@ref), [`EDaR`](@ref), [`EDaR_r`](@ref), [`RLDaR`](@ref), [`RLDaR_r`](@ref).

# Fields

  - `settings::RMSettings = RMSettings()`: configuration settings for the risk measure.
  - `alpha::T1 = 0.05`: significance level, `alpha ∈ (0, 1)`.

# Behaviour

## Validation

  - When setting `alpha` at construction or runtime, `alpha ∈ (0, 1)`.

# Examples

```@example
# Default settings
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

# Description

Entropic Value at Risk risk measure.

  - It is the upper bound of the Chernoff inequality for the [`VaR`](@ref) and [`CVaR`](@ref).
  - ``\\mathrm{VaR}(\\bm{X},\\, \\alpha) \\leq \\mathrm{CVaR}(\\bm{X},\\, \\alpha) \\leq \\mathrm{EVaR}(\\bm{X},\\, \\alpha) \\leq \\mathrm{RLVaR}(\\bm{X},\\, \\alpha,\\, \\kappa) \\leq \\mathrm{WR}(\\bm{X})``.

See also: [`RiskMeasure`](@ref), [`RMSettings`](@ref), [`Portfolio`](@ref), [`HCPortfolio`](@ref), [`optimise!`](@ref), [`set_rm`](@ref), [`calc_risk`](@ref), [`_EVaR`](@ref), [`VaR`](@ref), [`CVaR`](@ref), [`RLVaR`](@ref), [`WR`](@ref), [`DaR`](@ref), [`DaR_r`](@ref), [`CDaR`](@ref), [`CDaR_r`](@ref), [`EDaR`](@ref), [`EDaR_r`](@ref), [`RLDaR`](@ref), [`RLDaR_r`](@ref).

# Fields

  - `settings::RMSettings = RMSettings()`: configuration settings for the risk measure.
  - `alpha::T1 = 0.05`: significance level, `alpha ∈ (0, 1)`.
  - `solvers::Union{<:AbstractDict, Nothing}`: optional JuMP-compatible solvers for exponential cone problems.

# Behaviour

  - Requires solver capability for exponential cone problems.
  - Uses `solvers` from [`Portfolio`](@ref)/[`HCPortfolio`](@ref) if `solvers` is `nothing`.

## Validation

  - When setting `alpha` at construction or runtime, `alpha ∈ (0, 1)`.

# Examples

```@example
# Default settings
evar = EVaR()

# Custom configuration with specific solver
evar = EVaR(; alpha = 0.025,  # 2.5 % significance level
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

# Description

Relativistic Value at Risk risk measure.

  - It is a generalisation of the [`EVaR`](@ref).
  - ``\\mathrm{VaR}(\\bm{X},\\, \\alpha) \\leq \\mathrm{CVaR}(\\bm{X},\\, \\alpha) \\leq \\mathrm{EVaR}(\\bm{X},\\, \\alpha) \\leq \\mathrm{RLVaR}(\\bm{X},\\, \\alpha,\\, \\kappa) \\leq \\mathrm{WR}(\\bm{X})``.
  - ``\\lim_{\\kappa \\to 0} \\mathrm{RLVaR}(\\bm{X},\\, \\alpha,\\, \\kappa) \\approx \\mathrm{EVaR}(\\bm{X},\\, \\alpha)``
  - ``\\lim_{\\kappa \\to 1} \\mathrm{RLVaR}(\\bm{X},\\, \\alpha,\\, \\kappa) \\approx \\mathrm{WR}(\\bm{X})``

See also: [`RiskMeasure`](@ref), [`RMSettings`](@ref), [`Portfolio`](@ref), [`HCPortfolio`](@ref), [`optimise!`](@ref), [`set_rm`](@ref), [`calc_risk`](@ref), [`_RLVaR`](@ref), [`VaR`](@ref), [`CVaR`](@ref), [`EVaR`](@ref), [`WR`](@ref), [`DaR`](@ref), [`DaR_r`](@ref), [`CDaR`](@ref), [`CDaR_r`](@ref), [`EDaR`](@ref), [`EDaR_r`](@ref), [`RLDaR`](@ref), [`RLDaR_r`](@ref).

# Fields

  - `settings::RMSettings = RMSettings()`: configuration settings for the risk measure.
  - `alpha::T1 = 0.05`: significance level, `alpha ∈ (0, 1)`.
  - `kappa::T1 = 0.3`: significance level, `kappa ∈ (0, 1)`.
  - `solvers::Union{<:AbstractDict, Nothing}`: optional JuMP-compatible solvers for 3D power cone problems.

# Behaviour

  - Requires solver capability for 3D power cone problems.
  - Uses `solvers` from [`Portfolio`](@ref)/[`HCPortfolio`](@ref) if `solvers` is `nothing`.

## Validation

  - When setting `alpha` at construction or runtime, `alpha ∈ (0, 1)`.
  - When setting `kappa` at construction or runtime, `kappa ∈ (0, 1)`.

# Examples

```@example
# Default settings
rlvar = RLVaR()

# Custom configuration
rlvar = RLVaR(; alpha = 0.07,   # 7 % significance level
              kappa = 0.2,      # Deformation parameter
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

  - Measures the largest peak-to-trough decline in uncompounded returns.
  - ``\\mathrm{DaR}(\\bm{X},\\, \\alpha) \\leq \\mathrm{CDaR}(\\bm{X},\\, \\alpha) \\leq \\mathrm{EDaR}(\\bm{X},\\, \\alpha) \\leq \\mathrm{RLDaR}(\\bm{X},\\, \\alpha,\\, \\kappa) \\leq \\mathrm{MDD}(\\bm{X})``.

See also: [`RiskMeasure`](@ref), [`RMSettings`](@ref), [`Portfolio`](@ref), [`HCPortfolio`](@ref), [`optimise!`](@ref), [`set_rm`](@ref), [`calc_risk`](@ref), [`_MDD`](@ref).

# Fields

  - `settings::RMSettings = RMSettings()`: configuration settings for the risk measure.

# Examples

```@example
# Default settings
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

  - Measures the average of all peak-to-trough declines in uncompounded returns.
  - Provides a more balanced view than the maximum drawdown [`MDD`](@ref).

See also: [`RiskMeasure`](@ref), [`RMSettings`](@ref), [`Portfolio`](@ref), [`HCPortfolio`](@ref), [`optimise!`](@ref), [`set_rm`](@ref), [`calc_risk`](@ref), [`_ADD`](@ref), [`MDD`](@ref).

# Fields

  - `settings::RMSettings = RMSettings()`: configuration settings for the risk measure.

# Examples

```@example
# Default settings
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

  - Measures expected peak-to-trough loss in the worst `alpha %` of cases.
  - ``\\mathrm{DaR}(\\bm{X},\\, \\alpha) \\leq \\mathrm{CDaR}(\\bm{X},\\, \\alpha) \\leq \\mathrm{EDaR}(\\bm{X},\\, \\alpha) \\leq \\mathrm{RLDaR}(\\bm{X},\\, \\alpha,\\, \\kappa) \\leq \\mathrm{MDD}(\\bm{X})``.

See also: [`RiskMeasure`](@ref), [`RMSettings`](@ref), [`Portfolio`](@ref), [`HCPortfolio`](@ref), [`optimise!`](@ref), [`set_rm`](@ref), [`calc_risk`](@ref), [`_CDaR`](@ref), [`VaR`](@ref), [`EVaR`](@ref), [`RLVaR`](@ref), [`WR`](@ref), [`DaR`](@ref), [`DaR_r`](@ref), [`CDaR`](@ref), [`CDaR_r`](@ref), [`EDaR`](@ref), [`EDaR_r`](@ref), [`RLDaR`](@ref), [`RLDaR_r`](@ref).

# Fields

  - `settings::RMSettings = RMSettings()`: configuration settings for the risk measure.
  - `alpha::T1 = 0.05`: significance level, `alpha ∈ (0, 1)`.

# Behaviour

## Validation

  - When setting `alpha` at construction or runtime, `alpha ∈ (0, 1)`.

# Examples

```@example
# Default settings
cdar = CDaR()

# Custom significance level
cdar = CDaR(; settings = RMSettings(; scale = 1.0), alpha = 0.01) # 1 % significance level
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
    mutable struct UCI <: RiskMeasure

Ulcer Index risk measure.

  - Penalizes larger drawdowns more than smaller ones.

See also: [`RiskMeasure`](@ref), [`RMSettings`](@ref), [`Portfolio`](@ref), [`HCPortfolio`](@ref), [`optimise!`](@ref), [`set_rm`](@ref), [`calc_risk`](@ref), [`_UCI`](@ref).

# Fields

  - `settings::RMSettings = RMSettings()`: configuration settings for the risk measure.

# Examples

```@example
# Default settings
uci = UCI()

# Custom settings
uci = UCI(; settings = RMSettings(; scale = 1.5))
```
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

  - It is the upper bound of the Chernoff inequality for the [`DaR`](@ref) and [`CDaR`](@ref).
  - ``\\mathrm{DaR}(\\bm{X},\\, \\alpha) \\leq \\mathrm{CDaR}(\\bm{X},\\, \\alpha) \\leq \\mathrm{EDaR}(\\bm{X},\\, \\alpha) \\leq \\mathrm{RLDaR}(\\bm{X},\\, \\alpha,\\, \\kappa) \\leq \\mathrm{MDD}(\\bm{X})``.

See also: [`RiskMeasure`](@ref), [`RMSettings`](@ref), [`Portfolio`](@ref), [`HCPortfolio`](@ref), [`optimise!`](@ref), [`set_rm`](@ref), [`calc_risk`](@ref), [`_EDaR`](@ref), [`VaR`](@ref), [`CVaR`](@ref), [`RLVaR`](@ref), [`WR`](@ref), [`DaR`](@ref), [`DaR_r`](@ref), [`CDaR`](@ref), [`CDaR_r`](@ref), [`EDaR`](@ref), [`EDaR_r`](@ref), [`RLDaR`](@ref), [`RLDaR_r`](@ref).

# Fields

  - `settings::RMSettings = RMSettings()`: configuration settings for the risk measure.
  - `alpha::T1 = 0.05`: significance level, `alpha ∈ (0, 1)`.
  - `solvers::Union{<:AbstractDict, Nothing}`: optional JuMP-compatible solvers for exponential cone problems.

# Behaviour

  - Requires solver capability for exponential cone problems.
  - Uses `solvers` from [`Portfolio`](@ref)/[`HCPortfolio`](@ref) if `solvers` is `nothing`.

## Validation

  - When setting `alpha` at construction or runtime, `alpha ∈ (0, 1)`.

# Examples

```@example
# Default settings
edar = EDaR()

# Custom configuration with specific solver
edar = EDaR(; alpha = 0.025,  # 2.5 % significance level
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

# Description

Relativistic Drawdown at Risk risk measure.

  - It is a generalisation of the [`EDaR`](@ref).
  - ``\\mathrm{DaR}(\\bm{X},\\, \\alpha) \\leq \\mathrm{CDaR}(\\bm{X},\\, \\alpha) \\leq \\mathrm{EDaR}(\\bm{X},\\, \\alpha) \\leq \\mathrm{RLDaR}(\\bm{X},\\, \\alpha,\\, \\kappa) \\leq \\mathrm{MDD}(\\bm{X})``.
  - ``\\lim_{\\kappa \\to 0} \\mathrm{RLDaR}(\\bm{X},\\, \\alpha,\\, \\kappa) \\approx \\mathrm{EDaR}(\\bm{X},\\, \\alpha)``
  - ``\\lim_{\\kappa \\to 1} \\mathrm{RLDaR}(\\bm{X},\\, \\alpha,\\, \\kappa) \\approx \\mathrm{MDD}(\\bm{X})``

See also: [`RiskMeasure`](@ref), [`RMSettings`](@ref), [`Portfolio`](@ref), [`HCPortfolio`](@ref), [`optimise!`](@ref), [`set_rm`](@ref), [`calc_risk`](@ref), [`_RLDaR`](@ref), [`VaR`](@ref), [`CVaR`](@ref), [`EVaR`](@ref), [`WR`](@ref), [`DaR`](@ref), [`DaR_r`](@ref), [`CDaR`](@ref), [`CDaR_r`](@ref), [`EDaR`](@ref), [`EDaR_r`](@ref), [`RLDaR`](@ref), [`RLDaR_r`](@ref).

# Fields

  - `settings::RMSettings = RMSettings()`: configuration settings for the risk measure.
  - `alpha::T1 = 0.05`: significance level, `alpha ∈ (0, 1)`.
  - `kappa::T1 = 0.3`: significance level, `kappa ∈ (0, 1)`.
  - `solvers::Union{<:AbstractDict, Nothing}`: optional JuMP-compatible solvers for 3D power cone problems.

# Behaviour

  - Requires solver capability for 3D power cone problems.
  - Uses `solvers` from [`Portfolio`](@ref)/[`HCPortfolio`](@ref) if `solvers` is `nothing`.

## Validation

  - When setting `alpha` at construction or runtime, `alpha ∈ (0, 1)`.
  - When setting `kappa` at construction or runtime, `kappa ∈ (0, 1)`.

# Examples

```@example
# Default settings
rldar = RLDaR()

# Custom configuration
rldar = RLDaR(; alpha = 0.05, # 5 % significance level
              kappa = 0.3,    # 30 % Deformation parameter
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

# Description

Square Root Kurtosis risk measure implementation for portfolio optimisation.

See also: [`RiskMeasure`](@ref), [`RMSettings`](@ref), [`Portfolio`](@ref), [`HCPortfolio`](@ref), [`optimise!`](@ref), [`set_rm`](@ref), [`calc_risk`](@ref), [`_Kurt`](@ref), [`SKurt`](@ref).

# Fields

  - `settings::RMSettings = RMSettings()`: configuration settings for the risk measure.
  - `w::Union{<:AbstractWeights, Nothing} = nothing`: optional `T×1` vector of weights for expected return calculation.
  - `kt::Union{AbstractMatrix, Nothing} = nothing`: optional cokurtosis matrix.

# Behaviour

  - If `isnothing(kt)`: uses the semi cokurtosis matrix `skurt` from the [`Portfolio`](@ref)/[`HCPortfolio`](@ref) object.
  - If `kt` provided: uses custom semi cokurtosis matrix.

## Validation

  - When setting `kt` at construction or runtime, the matrix must be square (`N²×N²`).

# Examples

```@example
# Default settings
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

# Description

Square Root Semi Kurtosis risk measure implementation for portfolio optimisation.

  - Measures the kurtosis equal to or below the `target` return threshold.

See also: [`RiskMeasure`](@ref), [`RMSettings`](@ref), [`Portfolio`](@ref), [`HCPortfolio`](@ref), [`optimise!`](@ref), [`set_rm`](@ref), [`calc_risk`](@ref), [`_SKurt`](@ref), [`Kurt`](@ref).

# Fields

  - `settings::RMSettings = RMSettings()`: configuration settings for the risk measure.
  - `target::T1 = 0.0`: minimum return threshold for downside classification.
  - `w::Union{<:AbstractWeights, Nothing} = nothing`: optional `T×1` vector of weights for expected return calculation.
  - `kt::Union{AbstractMatrix, Nothing} = nothing`: optional cokurtosis matrix.

# Behaviour

  - If `isnothing(kt)`: uses the cokurtosis from matrix `skurt` from the [`Portfolio`](@ref)/[`HCPortfolio`](@ref) object.
  - If `kt` provided: uses custom cokurtosis matrix.

## Validation

  - When setting `kt` at construction or runtime, the matrix must be square (`N²×N²`).

# Examples

```@example
# Default settings
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
    mutable struct RG{T1 <: Real} <: RiskMeasure

# Description

Defines the Range risk measure.

  - Measures the best and worst returns, ``\\left[\\mathrm{WR}(\\bm{X}),\\, \\mathrm{WR}(-\\bm{X})\\right]``.

See also: [`RiskMeasure`](@ref), [`RMSettings`](@ref), [`Portfolio`](@ref), [`HCPortfolio`](@ref), [`optimise!`](@ref), [`set_rm`](@ref), [`calc_risk`](@ref), [`_RG`](@ref).

# Fields

  - `settings::RMSettings = RMSettings()`: configuration settings for the risk measure.

# Examples

```@example
# Default settings
rg = RG()

# Custom settings
rg = RG(; settings = RMSettings(; ub = 0.5))
```
"""
struct RG <: RiskMeasure
    settings::RMSettings
end
function RG(; settings::RMSettings = RMSettings())
    return RG(settings)
end

"""
    mutable struct CVaRRG{T1 <: Real, T2 <: Real} <: RiskMeasure

# Description

Defines the Conditional Value at Risk Range risk measure.

  - Measures the range between the expected loss in the worst `alpha %` of cases and expected gain in the best `beta %` of cases, ``\\left[\\mathrm{CVaR}(\\bm{X},\\, \\alpha),\\, \\mathrm{CVaR}(-\\bm{X},\\, \\beta)\\right]``.

See also: [`RiskMeasure`](@ref), [`RMSettings`](@ref), [`Portfolio`](@ref), [`HCPortfolio`](@ref), [`optimise!`](@ref), [`set_rm`](@ref), [`calc_risk`](@ref), [`_TGRG`](@ref), [`CVaR`](@ref), [`RG`](@ref).

# Fields

  - `settings::RMSettings = RMSettings()`: configuration settings for the risk measure.
  - `alpha::T1 = 0.05`: significance level of losses, `alpha ∈ (0, 1)`.
  - `alpha::T2 = 0.05`: significance level of gains, `beta ∈ (0, 1)`.

# Behaviour

## Validation

  - When setting `alpha` at construction or runtime, `alpha ∈ (0, 1)`.
  - When setting `beta` at construction or runtime, `beta ∈ (0, 1)`.

# Examples

```@example
# Default settings
cdar = CVaRRG()

# Custom significance level
cdar = CVaRRG(; settings = RMSettings(; scale = 1.0), #
              alpha = 0.01, # 1 % significance level losses 
              beta = 0.03)  # 3 % significance level gains
```
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
    mutable struct OWASettings{T1 <: AbstractVector{<:Real}}

# Description

Defines the settings for Ordered Weight Array (OWA) risk measures.

See also: [`RiskMeasure`](@ref), [`RMSettings`](@ref), [`Portfolio`](@ref), [`HCPortfolio`](@ref), [`optimise!`](@ref), [`set_rm`](@ref).

# Fields

  - `approx::Bool = true`: use the p-norm based approximation of the OWA risk measure optimisation.
  - `p::T1 = Float64[2, 3, 4, 10, 50]`: vector of the p-norm orders to be used in the approximation.

# Behaviour

  - `p` is only used when `approx == true`.

# Examples

```@example
# Default settings
owa = OWASettings()

# Use full risk measure formulation
owa = OWASettings(; approx = false)

# Use more p-norms
owa = OWASettings(; p = Float64[1, 2, 4, 8, 16, 32, 64, 128])
```
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
    struct GMD <: RiskMeasure

# Description

Defines the Gini Mean Difference risk measure.

See also: [`RiskMeasure`](@ref), [`RMSettings`](@ref), [`OWASettings`](@ref), [`Portfolio`](@ref), [`HCPortfolio`](@ref), [`optimise!`](@ref), [`set_rm`](@ref), [`calc_risk`](@ref), [`_GMD`](@ref).

# Fields

  - `settings::RMSettings = RMSettings()`: configuration settings for the risk measure.
  - `owa::OWASettings = OWASettings()`: OWA risk measure settings.

# Examples

```@example
# Default settings
gmd = GMD()

# Use full risk measure formulation
gmd = GMD(; owa = OWASettings(; approx = false))

# Use more p-norms and custom settings
gmd = GMD(; settings = RMSettings(; scale = 1.7),
          owa = OWASettings(; p = Float64[1, 2, 4, 8, 16, 32, 64, 128]))
```
"""
struct GMD <: RiskMeasure
    settings::RMSettings
    owa::OWASettings
end
function GMD(; settings::RMSettings = RMSettings(), owa::OWASettings = OWASettings())
    return GMD(settings, owa)
end

"""
    mutable struct TG{T1 <: Real, T2 <: Real, T3 <: Integer} <: RiskMeasure

# Description

Defines the Tail Gini Difference risk measure.

See also: [`RiskMeasure`](@ref), [`RMSettings`](@ref), [`OWASettings`](@ref), [`Portfolio`](@ref), [`HCPortfolio`](@ref), [`optimise!`](@ref), [`set_rm`](@ref), [`calc_risk`](@ref), [`_TG`](@ref).

# Fields

  - `settings::RMSettings = RMSettings()`: configuration settings for the risk measure.
  - `owa::OWASettings = OWASettings()`: OWA risk measure settings.
  - `alpha_i::T1 = 0.0001`: start value of the significance level of CVaR losses, `0 < alpha_i < alpha < 1`.
  - `alpha::T2 = 0.05`: end value of the significance level of CVaR losses, `0 < alpha_i < alpha < 1`.
  - `a_sim::T3 = 100`: number of CVaRs to approximate the Tail Gini losses, `a_sim > 0`.

# Behaviour

## Validation

  - When setting `alpha_i` at construction or runtime, `0 < alpha_i < alpha < 1`.
  - When setting `alpha` at construction or runtime, `0 < alpha_i < alpha < 1`.
  - When setting `a_sim` at construction or runtime, `a_sim > 0`.

# Examples

```@example
# Default settings
tg = TG()

# Use full risk measure formulation with custom parameters
tg = TG(; alpha = 0.07, owa = OWASettings(; approx = false))

# Use more p-norms and constrain risk without adding it to the problem's risk expression
tg = TG(; settings = RMSettings(; flag = false, ub = 0.1),
        owa = OWASettings(; p = Float64[1, 2, 4, 8, 16, 32, 64, 128]))
```
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
    mutable struct TGRG{T1 <: Real, T2 <: Real, T3 <: Integer, T4 <: Real, T5 <: Real, T6 <: Integer} <: RiskMeasure

# Description

Defines the Tail Gini Difference Range risk measure.

  - Measures the range between the worst `alpha %` tail gini of cases and best `beta %` tail gini of cases, ``\\left[\\mathrm{TG}(\\bm{X},\\, \\alpha),\\, \\mathrm{TG}(-\\bm{X},\\, \\beta)\\right]``.

See also: [`RiskMeasure`](@ref), [`RMSettings`](@ref), [`OWASettings`](@ref), [`Portfolio`](@ref), [`HCPortfolio`](@ref), [`optimise!`](@ref), [`set_rm`](@ref), [`calc_risk`](@ref), [`_TGRG`](@ref), [`TG`](@ref).

# Fields

  - `settings::RMSettings = RMSettings()`: configuration settings for the risk measure.
  - `owa::OWASettings = OWASettings()`: OWA risk measure settings.
  - `alpha_i::T1 = 0.0001`: start value of the significance level of CVaR losses, `0 < alpha_i < alpha < 1`.
  - `alpha::T2 = 0.05`: end value of the significance level of CVaR losses, `0 < alpha_i < alpha < 1`.
  - `a_sim::T3 = 100`: number of CVaRs to approximate the Tail Gini losses, `a_sim > 0`.
  - `beta_i::T4 = 0.0001`: start value of the significance level of CVaR gains, `0 < beta_i < beta < 1`.
  - `beta::T5 = 0.05`: end value of the significance level of CVaR gains, `0 < beta_i < beta < 1`.
  - `b_sim::T6 = 100`: number of CVaRs to approximate the Tail Gini gains, `b_sim > 0`.

# Behaviour

## Validation

  - When setting `alpha_i` at construction or runtime, `0 < alpha_i < alpha < 1`.
  - When setting `alpha` at construction or runtime, `0 < alpha_i < alpha < 1`.
  - When setting `a_sim` at construction or runtime, `a_sim > 0`.
  - When setting `beta_i` at construction or runtime, `0 < beta_i < beta < 1`.
  - When setting `beta` at construction or runtime, `0 < beta_i < beta < 1`.
  - When setting `b_sim` at construction or runtime, `b_sim > 0`.

# Examples

```@example
# Default settings
rtg = RTG()

# Use full risk measure formulation with custom parameters
rtg = RTG(; alpha = 0.07, b_sim = 200, owa = OWASettings(; approx = false))

# Use more p-norms and constrain risk without adding it to the problem's risk expression
rtg = RTG(; settings = RMSettings(; flag = false, ub = 0.1),
          owa = OWASettings(; p = Float64[1, 2, 4, 8, 16, 32, 64, 128]))
```
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
    mutable struct OWA <: RiskMeasure

# Description

Defines the generic Ordered Weight Array risk measure.

  - Uses a vector of ordered weights generated by [`owa_l_moment`](@ref) or [`owa_l_moment_crm`](@ref) for arbitrary L-moment optimisations.

See also: [`RiskMeasure`](@ref), [`RMSettings`](@ref), [`OWASettings`](@ref), [`Portfolio`](@ref), [`HCPortfolio`](@ref), [`optimise!`](@ref), [`set_rm`](@ref), [`calc_risk`](@ref), [`_OWA`](@ref), [`owa_l_moment`](@ref), [`owa_l_moment_crm`](@ref).

# Fields

  - `settings::RMSettings = RMSettings()`: configuration settings for the risk measure.
  - `owa::OWASettings = OWASettings()`: OWA risk measure settings.
  - `w::Union{<:AbstractWeights, Nothing} = nothing`: `T×1` ordered weight vector of arbitrary L-moments generated by [`owa_l_moment`](@ref) or [`owa_l_moment_crm`](@ref).

# Examples

```@example
# Default settings
w = owa_l_moment_crm(10)
owa = OWA(; w = w)

# Use full risk measure formulation with custom parameters
owa = OWA(; w = w, owa = OWASettings(; approx = false))

# Use more p-norms and constrain risk without adding it to the problem's risk expression
owa = OWA(; w = w, settings = RMSettings(; flag = false, ub = 0.1),
          owa = OWASettings(; p = Float64[1, 2, 4, 8, 16, 32, 64, 128]))
```
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
    struct dVar <: RiskMeasure

# Description

Define the Brownian Distance Variance risk measure.

  - Measures linear and non-linear relationships between variables.

See also: [`RiskMeasure`](@ref), [`RMSettings`](@ref), [`OWASettings`](@ref), [`Portfolio`](@ref), [`HCPortfolio`](@ref), [`optimise!`](@ref), [`set_rm`](@ref), [`calc_risk`](@ref), [`_dVar`](@ref).

# Fields

  - `settings::RMSettings = RMSettings()`: configuration settings for the risk measure.

# Examples

```@example
# Default settings
dvar = dVar()

# Custom settings
dvar = dVar(; settings = RMSettings(; ub = 0.5))
```
"""
struct dVar <: RiskMeasure
    settings::RMSettings
end
function dVar(; settings::RMSettings = RMSettings())
    return dVar(settings)
end

"""
    struct Skew <: RiskMeasure

# Description

Define the Quadratic Skewness risk measure.

See also: [`RiskMeasure`](@ref), [`RMSettings`](@ref), [`OWASettings`](@ref), [`Portfolio`](@ref), [`HCPortfolio`](@ref), [`optimise!`](@ref), [`set_rm`](@ref), [`calc_risk`](@ref), [`_Skew`](@ref).

# Fields

  - `settings::RMSettings = RMSettings()`: configuration settings for the risk measure.

# Examples

```@example
# Default settings
skew = Skew()

# Custom settings
skew = Skew(; settings = RMSettings(; ub = 0.5))
```
"""
struct Skew <: RiskMeasure
    settings::RMSettings
end
function Skew(; settings::RMSettings = RMSettings())
    return Skew(settings)
end

"""
    struct SSkew <: RiskMeasure

Define the Quadratic Semi Skewness risk measure.

See also: [`RiskMeasure`](@ref), [`RMSettings`](@ref), [`OWASettings`](@ref), [`Portfolio`](@ref), [`HCPortfolio`](@ref), [`optimise!`](@ref), [`set_rm`](@ref), [`calc_risk`](@ref), [`_Skew`](@ref).

# Fields

  - `settings::RMSettings = RMSettings()`: configuration settings for the risk measure.

# Examples

```@example
# Default settings
sskew = SSkew()

# Custom settings
sskew = SSkew(; settings = RMSettings(; ub = 0.5))
```
"""
struct SSkew <: RiskMeasure
    settings::RMSettings
end
function SSkew(; settings::RMSettings = RMSettings())
    return SSkew(settings)
end

"""
    mutable struct Variance{T1 <: Union{AbstractMatrix, Nothing}} <: HCRiskMeasure

# Description

Defines the Variance risk measure.

See also: [`HCRMSettings`](@ref), [`HCPortfolio`](@ref), [`optimise!`](@ref), [`calc_risk`](@ref), [`_Variance`](@ref).

# Fields

  - `settings::HCRMSettings = HCRMSettings()`: hierarchical risk measure configuration settings.
  - `sigma::Union{AbstractMatrix, Nothing} = nothing`: optional covariance matrix.

# Behaviour

  - If `isnothing(sigma)`: uses the covariance matrix `cov`.
  - If `sigma` provided: uses custom covariance matrix.

## Validation

  - When setting `sigma` at construction or runtime, the matrix must be square (`N×N`).

# Examples

```@example
# Default settings
variance = Variance()

# Custom settings
variance = Variance(; settings = HCRMSettings(; scale = 3))
```
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
const RMOWA = Union{GMD, TG, TGRG, OWA}

export RiskMeasure, HCRiskMeasure, RMSettings, HCRMSettings, QuadSD, SOCSD, SimpleSD, SD,
       MAD, SSD, FLPM, SLPM, WR, CVaR, EVaR, RLVaR, MDD, ADD, CDaR, UCI, EDaR, RLDaR, Kurt,
       SKurt, RG, CVaRRG, OWASettings, GMD, TG, TGRG, OWA, dVar, Skew, SSkew, Variance,
       SVariance, VaR, DaR, DaR_r, MDD_r, ADD_r, CDaR_r, UCI_r, EDaR_r, RLDaR_r, Equal
