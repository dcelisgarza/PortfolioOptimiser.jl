# Copywrite (c) 2025
# Author: Daniel Celis Garza <daniel.celis.garza@gmail.com>
# SPDX-License-Identifier: MIT

"""
    abstract type AbstractRiskMeasure end

Supertype for all risk measaures.

See also: [`RiskMeasure`](@ref), [`HCRiskMeasure`](@ref), [`NoOptRiskMeasure`](@ref).
"""
abstract type AbstractRiskMeasure end

"""
    abstract type RiskMeasure <: AbstractRiskMeasure end

Supertype for risk measures compatible with [`Trad`](@ref), [`RB`](@ref), [`NOC`](@ref), [`HRP`](@ref), [`HERC`](@ref), and [`NCO`](@ref) optimisation types.

See also: [`calc_risk`](@ref), [`RMSettings`](@ref), [`Trad`](@ref), [`RB`](@ref), [`NOC`](@ref), [`HRP`](@ref), [`HERC`](@ref), [`NCO`](@ref), [`set_rm`](@ref), [`set_rm_solvers!`](@ref), [`unset_rm_solvers!`](@ref).

# Implementation requirements

To ensure a risk measure can be used any of the above optimisation types, it must abide by a few rules.

  - Include a `settings::RMSettings` field, [`RMSettings`](@ref).

```julia
struct MyRiskMeasure <: RiskMeasure
    # Fields of MyRiskMeasure
    settings::RMSettings
end
```

  - Implement your measure's risk calculation method, [`calc_risk`](@ref). This will let the library use the risk function everywhere it needs to.

```julia
function calc_risk(my_risk::MyRiskMeasure, w::AbstractVector; kwargs...)
    # Risk measure calculation
end
```

  - A scalar [`JuMP`](https://github.com/jump-dev/JuMP.jl) model implementation of [`set_rm`](@ref). If appropriate, a vector equivalent.

```julia
# The scalar function.
function PortfolioOptimiser.set_rm(port, rm::MyRiskMeasure, type::Union{Trad, RB, NOC};
                                   kwargs...)
    # Get optimisation model.
    model = port.model

    ###
    # Variables, constraints, expressions, etc.
    ###

    # Define the risk expression for MyRiskMeasure
    @expression(model, MyRiskMeasure_risk, ...)

    # Define the key name for the upper bound.
    ub_key = "MyRiskMeasure_risk"

    # Set the upper bound on MyRiskMeasure_risk. 
    # If isinf(rm.settings.ub), no upper bound will be set.
    set_rm_risk_upper_bound(type, model, MyRiskMeasure_risk, rm.settings.ub, ub_key)

    # Add the risk to the risk expression.
    # If rm.settings.flag == true, MyRiskMeasure_risk will be added to the vector of risks.
    # This means it will form part of the objective function when the objective function 
    # is MinRisk, Utility or Sharpe.
    # If rm.settings.flag == false, MyRiskMeasure_risk will only be used as a risk 
    # constraint.
    set_risk_expression(model, MyRiskMeasure_risk, rm.settings.scale, rm.settings.flag)

    return nothing
end

# Vector equivalent if it's possible to provide multiple instances of MyRiskMeasure.
function PortfolioOptimiser.set_rm(port, rms::AbstractVector{<:MyRiskMeasure},
                                   type::Union{Trad, RB, NOC}; kwargs...)

    # Get optimisation model.
    model = port.model
    count = length(rms)

    ###
    # Variables, constraints, expressions, that can be initialised beforehand etc.
    ###

    # Define the risk expression for MyRiskMeasure.
    # It can also be defined in the iteration using the 
    # `model[key] = @expression(model, ..., ...) `
    # construct, then reference the `model[key]` expression.
    # Or use its anonymous version
    # `key = @expression(model, ..., ...)` 
    # and reference the `key` expression.
    @expression(model, MyRiskMeasure_risk[1:count], ...)

    for (i, rm) ∈ pairs(rms)
        ###
        # Variables, constraints, expressions, that must be set during each iteration.
        # If you want to register them in the model use:
        # `model[constraint_key] = @constraint(model, ..., ...)`
        # `model[expression_key] = @expression(model, ..., ...)`
        # `model[variable_key] = @variable(model, ..., ...)`
        # If you want them to be anonymous (i.e. not registered in the model) use:
        # `constraint_key = @constraint(model, ..., ...)`
        # `expression_key = @expression(model, ..., ...)`
        # `variable_key = @variable(model, ..., ...)`
        # If they were defined outside of the loop like MyRiskMeasure_risk 
        # but need to be modified then use:
        # `add_to_expression!(MyRiskMeasure_risk[i], ...).`
        ###

        # Define the key name for the upper bound.
        ub_key = "MyRiskMeasure_risk_\$(i)"

        # Set the upper bound on MyRiskMeasure_risk[i]. 
        # If isinf(rm.settings.ub), no upper bound will be set.
        set_rm_risk_upper_bound(type, model, MyRiskMeasure_risk[i], rm.settings.ub, ub_key)

        # Add the risk to the risk expression.
        # If rm.settings.flag == true, MyRiskMeasure_risk[i] will be added to the vector of
        # risks. This means it will form part of the objective function when the objective 
        # function is MinRisk, Utility or Sharpe.
        # If rm.settings.flag == false, MyRiskMeasure_risk[i] will only be used as a risk 
        # constraint.
        set_risk_expression(model, MyRiskMeasure_risk[i], rm.settings.scale,
                            rm.settings.flag)
    end

    return nothing
end
```

  - If calculating the risk requires solving an optimisation model it must have a `solvers::Union{<:AbstractDict, Nothing}` field to store [`JuMP`](https://github.com/jump-dev/JuMP.jl)-compatible solvers, and implement [`set_rm_solvers!`](@ref) and [`unset_rm_solvers!`](@ref).

```julia
struct MyRiskMeasure <: RiskMeasure
    # Fields of MyRiskMeasure
    solvers::Union{<:AbstractDict, Nothing}
end

function PortfolioOptimiser.set_rm_solvers!(rm::MyRiskMeasure, solvers)
    flag = false
    if isnothing(rm.solvers) || isempty(rm.solvers)
        rm.solvers = solvers
        flag = true
    end
    return flag
end

function PortfolioOptimiser.unset_rm_solvers!(rm::MyRiskMeasure, flag)
    if flag
        rm.solvers = nothing
    end
end
```
"""
abstract type RiskMeasure <: AbstractRiskMeasure end

"""
    abstract type HCRiskMeasure <: AbstractRiskMeasure end

Supertype for risk measures compatible with [`HRP`](@ref), [`HERC`](@ref), and [`NCO`](@ref) with any of the previous.

See also: [`calc_risk`](@ref), [`HCRMSettings`](@ref), [`HRP`](@ref), [`HERC`](@ref), [`NCO`](@ref), [`set_rm_solvers!`](@ref), [`unset_rm_solvers!`](@ref).

# Implementation requirements

To ensure a risk measure can be used by the above optimisation types, it must abide by a few rules.

  - Include a `settings::HCRMSettings` field, [`HCRMSettings`](@ref).

```julia
struct MyHCRiskMeasure <: HCRiskMeasure
    # Fields of MyHCRiskMeasure
    settings::HCRMSettings
end
```

  - Implement your measure's risk calculation method, [`calc_risk`](@ref). This will let the library use the risk function everywhere it needs to.

```julia
function calc_risk(my_risk::MyHCRiskMeasure, w::AbstractVector; kwargs...)
    # Risk measure calculation
end
```

  - If calculating the risk requires solving an optimisation model it must have a `solvers::Union{<:AbstractDict, Nothing}` field to store [`JuMP`](https://github.com/jump-dev/JuMP.jl)-compatible solvers, and implement [`set_rm_solvers!`](@ref) and [`unset_rm_solvers!`](@ref).

```julia
struct MyHCRiskMeasure <: RiskMeasure
    # Fields of MyHCRiskMeasure
    solvers::Union{<:AbstractDict, Nothing}
end

function PortfolioOptimiser.set_rm_solvers!(rm::MyHCRiskMeasure, solvers)
    flag = false
    if isnothing(rm.solvers) || isempty(rm.solvers)
        rm.solvers = solvers
        flag = true
    end
    return flag
end

function PortfolioOptimiser.unset_rm_solvers!(rm::MyHCRiskMeasure, flag)
    if flag
        rm.solvers = nothing
    end
end
```
"""
abstract type HCRiskMeasure <: AbstractRiskMeasure end

"""
    abstract type NoOptRiskMeasure <: AbstractRiskMeasure end

Abstract type for risk measures that cannot be used in optimisations but can be used as performance measurements via [`calc_risk`](@ref). This can be for two reasons:

 1. They can be negative, therefore unsuitable for [`HRP`](@ref), [`HERC`](@ref), and [`NCO`](@ref) with any of the previous.
 2. They have no known optimisation formulation, therefore unsuitable for [`Trad`](@ref), [`RB`](@ref), [`NOC`](@ref), and [`NCO`](@ref) with any of the previous.

See also: [`calc_risk`](@ref), [`Trad`](@ref), [`RB`](@ref), [`NOC`](@ref), [`HRP`](@ref), [`HERC`](@ref), [`NCO`](@ref).

# Implementation

  - Implement your measure's risk calculation method, [`calc_risk`](@ref). This will let the library use the risk function everywhere it needs to.

```julia
struct MyNoOptRiskMeasure <: NoOptRiskMeasure
    # Fields of MyNoOptRiskMeasure
end

function calc_risk(my_risk::MyNoOptRiskMeasure, w::AbstractVector; kwargs...)
    # Risk measure calculation
end
```
"""
abstract type NoOptRiskMeasure <: AbstractRiskMeasure end

"""
    mutable struct RMSettings{T1 <: Real, T2 <: Real}

Configuration settings for concrete subtypes of [`RiskMeasure`](@ref). Having this field makes it possible for a risk measure to be used in any optimisation types that take risk measures as parameters.

See also: [`calc_risk`](@ref), [`RiskMeasure`](@ref), [`Trad`](@ref), [`RB`](@ref), [`NOC`](@ref), [`HRP`](@ref), [`HERC`](@ref), [`NCO`](@ref), [`AbstractScalarisation`](@ref).

# Keyword Parameters

## In [`Trad`](@ref), [`RB`](@ref), [`NOC`](@ref), [`NCO`](@ref) with any of the previous

  - `flag::Bool = true`:

      + If `true`: it is included in the optimisation's risk vector.
      + If `false`: it is *not* included in the optimisation's risk vector, used when you want to constrain the upper bound of a risk measure without having that risk measure appear in the [`MinRisk`](@ref), [`Utility`](@ref), or [`Sharpe`](@ref) objective measures.

  - `scale::T1 = 1.0`: weight parameter of the risk measure in the [`AbstractScalarisation`](@ref) method being used.
  - `ub::T2 = Inf`: upper bound risk constraint.

## In [`HRP`](@ref), [`HERC`](@ref), [`NCO`](@ref) with any of the previous

  - `flag::Bool = true`: no effect, the risk cannot be bounded in these optimisations.
  - `scale::T1 = 1.0`: weight parameter of the risk measure in the [`AbstractScalarisation`](@ref) method being used.
  - `ub::T2 = Inf`: no effect, the risk cannot be bounded in these optimisations.

# Examples

```julia
using Clarabel

# Randomly generated normally distributed returns.
ret = [0.670643    1.94045   -0.0896267   0.851535    -0.268234
       1.33575    -0.541003   2.28744    -0.157588    -1.45177
       -1.91694    -0.167745   0.920495    0.00677243  -1.29112
       0.123141    1.59841   -0.185076    2.58911     -0.250747
       1.92782     1.01679    1.12107     1.09731     -0.99954
       2.07114    -0.513216  -0.532891    0.917748    -0.0346682
       -1.37424    -1.35272   -0.628216   -2.76234     -0.112378
       1.3831      1.14021   -0.577472    0.224504     1.28137
       -0.0577619  -0.10658   -0.637011    1.70933      1.84176
       1.6319      2.05059   -0.21469    -0.640715     1.39879]

# Instantiate portfolio instance.
port = Portfolio(; ret = ret, assets = 1:size(ret, 2),
                 solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                  :check_sol => (allow_local = true,
                                                                 allow_almost = true),
                                                  :params => Dict("verbose" => false))))

# Compute statistics.
asset_statistics!(port)

# Clusterise assets.
cluster_assets!(port)

# Traditional optimisation.
type = Trad(;
            rm = [
                  # Append to the risk vector as `sqrt(252) * sd_risk`
                  SD(; settings = RMSettings(; flag = true, scale = sqrt(252))),
                  # Append to the risk vector as `0.5 * cvar_risk`
                  CVaR(; settings = RMSettings(; scale = 0.5, flag = true)),
                  # Do not add to the risk vector but constrain the maximum
                  # CDaR to 100
                  CDaR(; settings = RMSettings(; flag = false, ub = 100))])
w1 = optimise!(port, type)
w1
#=
5×2 DataFrame
 Row │ tickers  weights     
     │ Int64    Float64     
─────┼──────────────────────
   1 │       1  3.11076e-11
   2 │       2  0.0165189
   3 │       3  0.525957
   4 │       4  0.0220971
   5 │       5  0.435427
=#

# Risk vector only contains the risk measures with `flag == true`, 
# multiplied by the scaling factor
port.model[:risk_vec]
#=
2-element Vector{JuMP.AbstractJuMPScalar}:
 15.874507866387544 sd_risk
 0.5 var + z_cvar[1] + z_cvar[2] + z_cvar[3] + z_cvar[4] + z_cvar[5] + z_cvar[6] + z_cvar[7] + z_cvar[8] + z_cvar[9] + z_cvar[10]
=#

# The risk expression is the sum of the entries in the vector because
# the default scalarisation is the sum of the expressions.
port.model[:risk]
#=
15.874507866387544 sd_risk + 0.5 var + z_cvar[1] + z_cvar[2] + z_cvar[3] + z_cvar[4] + z_cvar[5] + z_cvar[6] + z_cvar[7] + z_cvar[8] + z_cvar[9] + z_cvar[10]
=#

# Check that the CDaR is indeed constrained to be <= 100.
port.model[:cdar_risk_ub]
#=
dar + 2 z_cdar[1] + 2 z_cdar[2] + 2 z_cdar[3] + 2 z_cdar[4] + 2 z_cdar[5] + 2 z_cdar[6] + 2 z_cdar[7] + 2 z_cdar[8] + 2 z_cdar[9] + 2 z_cdar[10] <= 100
=#

# Hierarchical equal risk optimisation.
type = HERC(;
            rm = [
                  # Add to the risk calculation (via `AbstractScalarisation`)
                  # as `sqrt(252) * sd_risk`.
                  SD(; settings = RMSettings(; flag = true, scale = sqrt(252))),
                  # Add to the risk calculation (via `AbstractScalarisation`)
                  # as `0.5 * cvar_risk`.
                  CVaR(; settings = RMSettings(; scale = 0.5, flag = true)),
                  # Add to the risk calculation because `flag` and `ub` have no
                  # effect in hierarchical optimisations.
                  CDaR(; settings = RMSettings(; flag = false, ub = 100))])
w2 = optimise!(port, type)
w2
#=
5×2 DataFrame
 Row │ tickers  weights   
     │ Int64    Float64   
─────┼────────────────────
   1 │       1  0.0548607
   2 │       2  0.0634407
   3 │       3  0.695601
   4 │       4  0.0506031
   5 │       5  0.135495
=#

# There is no `JuMP` model for HRP and HERC so we can't see the internals. 
# The effects can be seen by changing coefficients.
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

Configuration settings for concrete subtypes of [`HCRiskMeasure`](@ref).

See also: [`calc_risk`](@ref), [`HCRiskMeasure`](@ref), [`HRP`](@ref), [`HERC`](@ref), [`NCO`](@ref), [`AbstractScalarisation`](@ref).

# Keyword Parameters

  - `scale::T1 = 1.0`: weight parameter of the risk measure in the [`AbstractScalarisation`](@ref) method being used.

```julia
# Randomly generated normally distributed returns.
ret = [0.670643    1.94045   -0.0896267   0.851535    -0.268234
       1.33575    -0.541003   2.28744    -0.157588    -1.45177
       -1.91694    -0.167745   0.920495    0.00677243  -1.29112
       0.123141    1.59841   -0.185076    2.58911     -0.250747
       1.92782     1.01679    1.12107     1.09731     -0.99954
       2.07114    -0.513216  -0.532891    0.917748    -0.0346682
       -1.37424    -1.35272   -0.628216   -2.76234     -0.112378
       1.3831      1.14021   -0.577472    0.224504     1.28137
       -0.0577619  -0.10658   -0.637011    1.70933      1.84176
       1.6319      2.05059   -0.21469    -0.640715     1.39879]

# Instantiate portfolio instance.
port = Portfolio(; ret = ret, assets = 1:size(ret, 2))

# Compute asset statistics.                                                
asset_statistics!(port)

# Clusterise assets.
cluster_assets!(port)

# Hierarchical equal risk optimisation.
type = HERC(;
            rm = [
                  # Add to the risk calculation (via `AbstractScalarisation`)
                  # as `0.5 * dar_risk`.
                  DaR(; settings = HCRMSettings(; scale = 0.5)),
                  # Add to the risk calculation (via `AbstractScalarisation`)
                  # as `3 * var_risk`.
                  VaR(; settings = HCRMSettings(; scale = 3))])
w = optimise!(port, type)
#=
5×2 DataFrame
 Row │ tickers  weights   
     │ Int64    Float64   
─────┼────────────────────
   1 │       1  0.035433
   2 │       2  0.0480019
   3 │       3  0.795106
   4 │       4  0.0245889
   5 │       5  0.0968701
=#

# There is no `JuMP` model for HRP and HERC so we can't see the internals. 
# The effects can be seen by changing coefficients.
```
"""
mutable struct HCRMSettings{T1 <: Real}
    scale::T1
end
function HCRMSettings(; scale::Real = 1.0)
    return HCRMSettings{typeof(scale)}(scale)
end

"""
    abstract type VarianceFormulation end

Abstract type for implementing various formulations of the [`Variance`](@ref) in [`Trad`](@ref), [`RB`](@ref), [`NOC`](@ref), [`NCO`](@ref) with any of the previous optimisation types.

See also: [`Variance`](@ref), [`Quad`](@ref), [`SOC`](@ref), [`Trad`](@ref), [`RB`](@ref), [`NOC`](@ref), [`NCO`](@ref).
"""
abstract type VarianceFormulation end

"""
    struct Quad <: VarianceFormulation end

Explicit quadratic formulation for the [`Variance`](@ref) as an optimisation model.

```math
\\begin{align}
\\underset{\\bm{w}}{\\mathrm{opt}} &\\qquad \\bm{w}^\\intercal \\mathbf{\\Sigma} \\bm{w}\\\\
\\end{align}
```

Where:

  - ``\\bm{w}``: is the `N×1` vector of asset weights.
  - ``\\mathbf{\\Sigma}``: is the `N×N` asset covariance matrix.

See also: [`VarianceFormulation`](@ref), [`SOC`](@ref), [`Variance`](@ref).

# Behaviour

  - Produces a [`QuadExpr`](https://jump.dev/JuMP.jl/stable/api/JuMP/#QuadExpr) risk expression `variance_risk = dot(w, sigma, w)`.
  - Not compatible with [`NOC`](@ref) optimisations because [`QuadExpr`](https://jump.dev/JuMP.jl/stable/api/JuMP/#QuadExpr) are not strictly convex.
  - No additional variables or constraints introduced.
  - Requires a solver capable of handling quadratic expressions.
  - Performance may degrade for large portfolios.

# Examples

```julia
using Clarabel, LinearAlgebra

# Randomly generated normally distributed returns.
ret = [0.670643    1.94045   -0.0896267   0.851535    -0.268234
       1.33575    -0.541003   2.28744    -0.157588    -1.45177
       -1.91694    -0.167745   0.920495    0.00677243  -1.29112
       0.123141    1.59841   -0.185076    2.58911     -0.250747
       1.92782     1.01679    1.12107     1.09731     -0.99954
       2.07114    -0.513216  -0.532891    0.917748    -0.0346682
       -1.37424    -1.35272   -0.628216   -2.76234     -0.112378
       1.3831      1.14021   -0.577472    0.224504     1.28137
       -0.0577619  -0.10658   -0.637011    1.70933      1.84176
       1.6319      2.05059   -0.21469    -0.640715     1.39879]

# Instantiate portfolio instance.
port = Portfolio(; ret = ret, assets = 1:size(ret, 2),
                 solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                  :check_sol => (allow_local = true,
                                                                 allow_almost = true),
                                                  :params => Dict("verbose" => false))))
# Compute asset statistics.                                                
asset_statistics!(port)

# Explicit quadratic expression for the variance.
w = optimise!(port, Trad(; rm = Variance(; formulation = Quad()), str_names = true))
w
#=
5×2 DataFrame
 Row │ tickers  weights    
     │ Int64    Float64    
─────┼─────────────────────
   1 │       1  3.5113e-10
   2 │       2  0.0183835
   3 │       3  0.520498
   4 │       4  0.0340474
   5 │       5  0.427071
=#

# Covariance matrix, sigma.
port.cov
#=
5×5 Matrix{Float64}:
 1.89335    0.649754   0.129808     0.602482     0.262616
 0.649754   1.41321   -0.152314     0.785941     0.367755
 0.129808  -0.152314   0.958713    -0.00842172  -0.855604
 0.602482   0.785941  -0.00842172   2.11576      0.113369
 0.262616   0.367755  -0.855604     0.113369     1.32352
=#

# Check that the variance risk is indeed `dot(w, sigma, w)`.
port.model[:variance_risk] == dot(port.model[:w], port.cov, port.model[:w]) # true
```
"""
struct Quad <: VarianceFormulation end

"""
    struct SOC <: VarianceFormulation end

Second-Order Cone (SOC) formulation for the [`Variance`](@ref). Reformulates the quadratic variance expression using a [MOI.SecondOrderCone](https://jump.dev/JuMP.jl/stable/tutorials/conic/tips_and_tricks/#Second-Order-Cone) cone constraint.

```math
\\begin{align}
\\underset{\\bm{w}}{\\mathrm{opt}} &\\qquad \\sigma^2\\nonumber\\\\
\\textrm{s.t.} &\\qquad \\left\\lVert \\mathbf{G} \\bm{w} \\right\\rVert_{2} \\leq \\sigma\\,.
\\end{align}
```

Where:

  - ``\\bm{w}``: is the `N×1` vector of asset weights.
  - ``\\mathbf{G}``: is a suitable factorisation of the `N×N` covariance matrix, such as the square root matrix, or the Cholesky factorisation.
  - ``\\sigma^2``: is the portfolio variance.
  - ``\\lVert \\cdot \\rVert_{2}``: is the L-2 norm, which is modelled as an [MOI.SecondOrderCone](https://jump.dev/JuMP.jl/stable/tutorials/conic/tips_and_tricks/#Second-Order-Cone).

See also: [`VarianceFormulation`](@ref), [`Quad`](@ref), [`Variance`](@ref).

# Behaviour

  - Uses a [`SecondOrderCone`](https://jump.dev/JuMP.jl/stable/manual/constraints/#Second-order-cone-constraints) constraint.
  - Defines a standard deviation variable `dev`.
  - Produces a [`QuadExpr`](https://jump.dev/JuMP.jl/stable/api/JuMP/#QuadExpr) risk expression `sd_risk = dev^2`.
  - Not compatible with [`NOC`](@ref) (Near Optimal Centering) optimisations because [`QuadExpr`](https://jump.dev/JuMP.jl/stable/api/JuMP/#QuadExpr) are not strictly convex.
  - Often more numerically stable than direct quadratic formulation.
  - Better scaling properties for large portfolios.
  - Compatible with specialised conic solvers.
  - May introduce more variables but often leads to better solution times.
  - Particularly effective for large-scale problems.

# Examples

```julia
using Clarabel

# Randomly generated normally distributed returns.
ret = [0.670643    1.94045   -0.0896267   0.851535    -0.268234
       1.33575    -0.541003   2.28744    -0.157588    -1.45177
       -1.91694    -0.167745   0.920495    0.00677243  -1.29112
       0.123141    1.59841   -0.185076    2.58911     -0.250747
       1.92782     1.01679    1.12107     1.09731     -0.99954
       2.07114    -0.513216  -0.532891    0.917748    -0.0346682
       -1.37424    -1.35272   -0.628216   -2.76234     -0.112378
       1.3831      1.14021   -0.577472    0.224504     1.28137
       -0.0577619  -0.10658   -0.637011    1.70933      1.84176
       1.6319      2.05059   -0.21469    -0.640715     1.39879]

# Instantiate portfolio instance.
port = Portfolio(; ret = ret, assets = 1:size(ret, 2),
                 solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                  :check_sol => (allow_local = true,
                                                                 allow_almost = true),
                                                  :params => Dict("verbose" => false))))
# Compute asset statistics.                                                
asset_statistics!(port)

# Explicit quadratic expression for the variance.
w = optimise!(port, Trad(; rm = Variance(; formulation = SOC()), str_names = true))
w
#=
5×2 DataFrame
 Row │ tickers  weights    
     │ Int64    Float64    
─────┼─────────────────────
   1 │       1  2.14218e-9
   2 │       2  0.0183819
   3 │       3  0.520499
   4 │       4  0.0340462
   5 │       5  0.427073
=#

# Covariance matrix, sigma.
port.cov
#=
5×5 Matrix{Float64}:
 1.89335    0.649754   0.129808     0.602482     0.262616
 0.649754   1.41321   -0.152314     0.785941     0.367755
 0.129808  -0.152314   0.958713    -0.00842172  -0.855604
 0.602482   0.785941  -0.00842172   2.11576      0.113369
 0.262616   0.367755  -0.855604     0.113369     1.32352
 =#

# Check that the variance risk exists as an SOC constraint.
port.model[:variance_risk]
#=
dev²
=#

port.model[:constr_dev_soc]
#=
constr_dev_soc : 
[dev, 
1.3339817521328776 w[1] + 0.23752111354642388 w[2] + 0.08772311968949179 w[3] + 0.1933954630317334 w[4] + 0.11104990914089145 w[5], 
0.23752111354642388 w[1] + 1.1172071395407432 w[2] - 0.05472328163488465 w[3] + 0.2914549020386699 w[4] + 0.14389651700778633 w[5],
0.08772311968949179 w[1] - 0.05472328163488465 w[2] + 0.8635663605757948 w[3] - 0.00013005161391315675 w[4] - 0.4497512686528594 w[5],
0.1933954630317334 w[1] + 0.2914549020386699 w[2] - 0.00013005161391315675 w[3] + 1.4117365158022712 w[4] + 0.020326110574790338 w[5],
0.11104990914089145 w[1] + 0.14389651700778633 w[2] - 0.4497512686528594 w[3] + 0.020326110574790338 w[4] + 1.0429726582846264 w[5]]
in MathOptInterface.SecondOrderCone(6)
=#
```
"""
struct SOC <: VarianceFormulation end

"""
    mutable struct SD <: RiskMeasure

Measures and computes the portfolio Standard Deviation (SD).

```math
\\begin{align}
\\mathrm{SD}(\\bm{w},\\, \\mathbf{\\Sigma}) &= \\left(\\bm{w}^\\intercal \\, \\mathbf{\\Sigma} \\, \\bm{w}\\right)^{1/2}\\,.
\\end{align}
```

Where:

  - ``\\bm{w}``: is the `N×1` vector of asset weights.
  - ``\\mathbf{\\Sigma}``: is the `N×N` asset covariance matrix.

See also: [`RiskMeasure`](@ref), [`RMSettings`](@ref), [`Variance`](@ref), [`SSD`](@ref), [`SVariance`](@ref), [`PortClass`](@ref), [`OptimType`](@ref), [`calc_risk`](@ref), [`optimise!`](@ref), [`set_rm`](@ref).

# Keyword Parameters

  - `settings::RMSettings = RMSettings()`: risk measure configuration settings.

  - `sigma::Union{<:AbstractMatrix, Nothing} = nothing`: optional covariance matrix.

      + If `nothing`: takes its value from the instance [`Portfolio`](@ref), the field depends on the [`PortClass`](@ref) parameter of the [`OptimType`](@ref) used.

# Behaviour

  - When setting `sigma` at construction or runtime, the matrix must be square (`N×N`).

## In [`Trad`](@ref), [`RB`](@ref), [`NOC`](@ref), [`NCO`](@ref) with any of the previous

  - Uses a [`SecondOrderCone`](https://jump.dev/JuMP.jl/stable/manual/constraints/#Second-order-cone-constraints) constraint.
  - Defines a standard deviation variable `sd_risk`.
  - Sets the [`AffExpr`](https://jump.dev/JuMP.jl/stable/api/JuMP/#AffExpr) risk expression `sd_risk = dev`.
  - Compatible with [`NOC`](@ref) (Near Optimal Centering) optimisations because [`AffExpr`](https://jump.dev/JuMP.jl/stable/api/JuMP/#AffExpr) are strictly convex.

# Functor

  - `(sd::SD)(w::AbstractVector)`: computes the standard deviation using the covariance matrix stored in the `N×N` matrix `sigma`, and `N×1` vector of asset weights, `w`.

# Examples

```julia
using Clarabel

# Randomly generated normally distributed returns.
ret = [0.670643    1.94045   -0.0896267   0.851535    -0.268234
       1.33575    -0.541003   2.28744    -0.157588    -1.45177
       -1.91694    -0.167745   0.920495    0.00677243  -1.29112
       0.123141    1.59841   -0.185076    2.58911     -0.250747
       1.92782     1.01679    1.12107     1.09731     -0.99954
       2.07114    -0.513216  -0.532891    0.917748    -0.0346682
       -1.37424    -1.35272   -0.628216   -2.76234     -0.112378
       1.3831      1.14021   -0.577472    0.224504     1.28137
       -0.0577619  -0.10658   -0.637011    1.70933      1.84176
       1.6319      2.05059   -0.21469    -0.640715     1.39879]

# Instantiate portfolio instance.
port = Portfolio(; ret = ret, assets = 1:size(ret, 2),
                 solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                  :check_sol => (allow_local = true,
                                                                 allow_almost = true),
                                                  :params => Dict("verbose" => false))))
# Risk measure
rm = SD()

# Compute asset statistics.                                                
asset_statistics!(port)

# Explicit quadratic expression for the variance.
w1 = optimise!(port, Trad(; rm = rm, str_names = true))
w1
#=
5×2 DataFrame
 Row │ tickers  weights     
     │ Int64    Float64     
─────┼──────────────────────
   1 │       1  1.42346e-10
   2 │       2  0.018383
   3 │       3  0.520498
   4 │       4  0.0340474
   5 │       5  0.427072
=#

# Compute the standard deviation.
r1 = calc_risk(port, :Trad; rm = rm) # 0.3612722213729872

# As a functor.
r1 == SD(; sigma = port.cov)(w1.weights) # true

# Covariance matrix, sigma.
port.cov
#=
5×5 Matrix{Float64}:
 1.89335    0.649754   0.129808     0.602482     0.262616
 0.649754   1.41321   -0.152314     0.785941     0.367755
 0.129808  -0.152314   0.958713    -0.00842172  -0.855604
 0.602482   0.785941  -0.00842172   2.11576      0.113369
 0.262616   0.367755  -0.855604     0.113369     1.32352
=#

# Check that the std risk exists as an SOC constraint.
port.model[:sd_risk]
#=
sd_risk
=#

port.model[:constr_sd_risk_soc]
#=
constr_sd_risk_soc : 
[sd_risk,
1.3339817521328776 w[1] + 0.23752111354642388 w[2] + 0.08772311968949179 w[3] + 0.1933954630317334 w[4] + 0.11104990914089145 w[5],
0.23752111354642388 w[1] + 1.1172071395407432 w[2] - 0.05472328163488465 w[3] + 0.2914549020386699 w[4] + 0.14389651700778633 w[5],
0.08772311968949179 w[1] - 0.05472328163488465 w[2] + 0.8635663605757948 w[3] - 0.00013005161391315675 w[4] - 0.4497512686528594 w[5],
0.1933954630317334 w[1] + 0.2914549020386699 w[2] - 0.00013005161391315675 w[3] + 1.4117365158022712 w[4] + 0.020326110574790338 w[5],
0.11104990914089145 w[1] + 0.14389651700778633 w[2] - 0.4497512686528594 w[3] + 0.020326110574790338 w[4] + 1.0429726582846264 w[5]]
in MathOptInterface.SecondOrderCone(6)
=#

# Clusterise assets.
cluster_assets!(port)

# Hierarchical optimisation, no JuMP model.
w2 = optimise!(port, HRP(; rm = rm))
w2
#=
5×2 DataFrame
 Row │ tickers  weights  
     │ Int64    Float64  
─────┼───────────────────
   1 │       1  0.199755
   2 │       2  0.229988
   3 │       3  0.198033
   4 │       4  0.133306
   5 │       5  0.238918
=#

# Compute the standard deviation.
r2 = calc_risk(port, :HRP; rm = rm) # 0.6547632732108616

# As a functor.
r2 == SD(; sigma = port.cov)(w2.weights) # true
```
"""
mutable struct SD <: RiskMeasure
    settings::RMSettings
    sigma::Union{<:AbstractMatrix, Nothing}
end
function SD(; settings::RMSettings = RMSettings(),
            sigma::Union{<:AbstractMatrix, Nothing} = nothing)
    if !isnothing(sigma)
        @smart_assert(size(sigma, 1) == size(sigma, 2))
    end
    return SD(settings, sigma)
end
function Base.setproperty!(obj::SD, sym::Symbol, val)
    if sym == :sigma
        if !isnothing(val)
            @smart_assert(size(val, 1) == size(val, 2))
        end
    end
    return setfield!(obj, sym, val)
end
function (sd::SD)(w::AbstractVector)
    return sqrt(dot(w, sd.sigma, w))
end

"""
    mutable struct MAD <: RiskMeasure

Measures and computes the portfolio Mean Absolute Deviation (MAD). In other words, it is the expected value of the absolute deviation from the expected value of the returns vector. This is a generalisation to accomodate the use of weighted means.

```math
\\begin{align}
\\mathrm{MAD}(\\bm{X}) &= \\mathbb{E}\\left(\\left\\lvert \\bm{X} - \\mathbb{E}\\left(\\bm{X}\\right) \\right\\rvert\\right)\\,.
\\end{align}
```

Where:

  - ``\\bm{X}``: is the `T×1` portfolio returns vector.
  - ``\\lvert \\cdot \\rvert``: is the absolute value.
  - ``\\mathbb{E}(\\cdot)``: is the expected value.

See also: [`RiskMeasure`](@ref), [`RMSettings`](@ref), [`Portfolio`](@ref), [`MuSimple`](@ref), [`NoKelly`](@ref), [`calc_risk`](@ref), [`optimise!`](@ref), [`set_rm`](@ref).

# Keyword Parameters

  - `settings::RMSettings = RMSettings()`: risk measure configuration settings.

  - `w1::Union{<:AbstractWeights, Nothing} = nothing`: optional, `T×1` vector of weights for computing the expected value of the returns vector (internal expected value).

      + `w1` has no effect in optimisations using [`JuMP`](https://github.com/jump-dev/JuMP.jl) models. However, it can be taken into account if `mu` paramter is computed with the [`MuSimple`](@ref) estimator and the would-be value of `w1` as the weight vector.
  - `w2::Union{<:AbstractWeights, Nothing} = nothing`: optional `T×1` vector of weights for computing the expected value of the absolute deviation (external expected value).
  - `mu::Union{<:AbstractVector, Nothing} = nothing`: optional, vector of expected returns.

      + If `nothing`: takes its value from the `mu` instance [`Portfolio`](@ref).
      + Only used in optimisations using [`JuMP`](https://github.com/jump-dev/JuMP.jl) models. Other optimisations, as well as the risk calculation, compute its value via the functor.

# Functor

  - `(mad::MAD)(x::AbstractVector)`: computes the mean absolute deviation using the internal values of `w1` and `w2`, for a `T×1` vector of portfolio returns, `x`.

!!! warning

    Using non-default `w1`, `w2` and/or `mu` in optimisations using [`JuMP`](https://github.com/jump-dev/JuMP.jl) may lead to the value computed by the functor to be different than that of `mad_risk`. This is because the [`JuMP`](https://github.com/jump-dev/JuMP.jl) model is based on slack constraints, and the functor computes the internal expected return.

# Examples

```julia
using Clarabel, StatsBase, JuMP

# Randomly generated normally distributed returns.
ret = [0.670643    1.94045   -0.0896267   0.851535    -0.268234
       1.33575    -0.541003   2.28744    -0.157588    -1.45177
       -1.91694    -0.167745   0.920495    0.00677243  -1.29112
       0.123141    1.59841   -0.185076    2.58911     -0.250747
       1.92782     1.01679    1.12107     1.09731     -0.99954
       2.07114    -0.513216  -0.532891    0.917748    -0.0346682
       -1.37424    -1.35272   -0.628216   -2.76234     -0.112378
       1.3831      1.14021   -0.577472    0.224504     1.28137
       -0.0577619  -0.10658   -0.637011    1.70933      1.84176
       1.6319      2.05059   -0.21469    -0.640715     1.39879]

# Instantiate portfolio instance.
port = Portfolio(; ret = ret, assets = 1:size(ret, 2),
                 solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                  :check_sol => (allow_local = true,
                                                                 allow_almost = true),
                                                  :params => Dict("verbose" => false))))

# Compute asset statistics.
asset_statistics!(port)

# Vanilla mean absolute deviation.
rm = MAD()

w1 = optimise!(port, Trad(; rm = rm, str_names = true))
w1
#=
5×2 DataFrame
 Row │ tickers  weights     
     │ Int64    Float64     
─────┼──────────────────────
   1 │       1  1.52867e-11 
   2 │       2  0.125902    
   3 │       3  0.558899    
   4 │       4  0.0528451   
   5 │       5  0.262353 
=#

r1 = calc_risk(port; rm = rm) # 0.2742955393862703
isapprox(r1, value(port.model[:mad_risk])) # true

# Exponential weights.
ew1 = eweights(1:size(ret, 1), 0.2; scale = true)
ew2 = eweights(1:size(ret, 1), 0.3; scale = true)

# Compute asset statistics, use ew1 in the `Trad` optimisation.
# This makes it consistent with the risk measure.
asset_statistics!(port; mu_type = MuSimple(; w = ew1))

# Mean absolute deviation with different weights.
# w1 has no effect, so we account for it in the computation
# of `port.mu` above.
rm = MAD(; w1 = ew1, w2 = ew2)

w2 = optimise!(port, Trad(; rm = rm, str_names = true))
w2
#=
5×2 DataFrame
 Row │ tickers  weights     
     │ Int64    Float64     
─────┼──────────────────────
   1 │       1  0.0868265
   2 │       2  1.62118e-10
   3 │       3  0.623665
   4 │       4  5.10551e-11
   5 │       5  0.289508
=#

# Using `w1` and `w2` may lead to inconsistent values between the
# functor and value in the model because the mean absolute 
# deviation is formulated with slack constraints.
r2_1 = calc_risk(port; rm = rm) # 0.23722684255394683
r2_2 = value(port.model[:mad_risk]) # 0.21882857231148417

# Custom mu (added some random noise).
custom_mu = port.mu + [-0.0025545471368230766, -0.0047554044723918795, 0.010574122455999866,
                       0.0021521845052968917, -0.004417767086053032]
rm = MAD(; mu = custom_mu)

w3 = optimise!(port, Trad(; rm = rm, str_names = true))
w3
#=
5×2 DataFrame
 Row │ tickers  weights     
     │ Int64    Float64     
─────┼──────────────────────
   1 │       1  0.129419
   2 │       2  9.10067e-11
   3 │       3  0.590633
   4 │       4  0.13825
   5 │       5  0.141698
=#

# Values don't match.
r3_1 = calc_risk(port; rm = rm) # 0.359772128074709
r3_2 = value(port.model[:mad_risk]) # 0.23232767711259888

# Clusterise assets.
cluster_assets!(port)

# Vanilla mean absolute deviation.
rm = MAD()

# Hierarchical optimisation, no JuMP model.
w4 = optimise!(port, HRP(; rm = rm))
w4
#=
5×2 DataFrame
 Row │ tickers  weights  
     │ Int64    Float64  
─────┼───────────────────
   1 │       1  0.179264
   2 │       2  0.222268
   3 │       3  0.216523
   4 │       4  0.160511
   5 │       5  0.221434
=#

# Compute the mean absolute deviation.
r4 = calc_risk(port, :HRP; rm = rm) # 0.46350089072293715
# As a functor.
r4 == rm(port.returns * w4.weights) # true

# Custom mu has no effect.
rm = MAD(; mu = custom_mu)

# Hierarchical optimisation, no JuMP model.
w5 = optimise!(port, HRP(; rm = rm))
w4.weights == w5.weights # true

# Compute the mean absolute deviation.
r5 = calc_risk(port, :HRP; rm = rm) # 0.46350089072293715

# w1 and w2 both have effects.
rm = MAD(; w1 = ew1, w2 = ew2)

# Hierarchical optimisation, no JuMP model.
w6 = optimise!(port, HRP(; rm = rm))
w6
#=
5×2 DataFrame
 Row │ tickers  weights  
     │ Int64    Float64  
─────┼───────────────────
   1 │       1  0.157301
   2 │       2  0.179251
   3 │       3  0.363034
   4 │       4  0.128884
   5 │       5  0.171529
=#

# Compute the mean absolute deviation.
r6 = calc_risk(port, :HRP; rm = rm) # 0.3704190560125852
```
"""
mutable struct MAD <: RiskMeasure
    settings::RMSettings
    w1::Union{<:AbstractWeights, Nothing}
    w2::Union{<:AbstractWeights, Nothing}
    mu::Union{<:AbstractVector, Nothing}
end
function MAD(; settings::RMSettings = RMSettings(),
             w1::Union{<:AbstractWeights, Nothing} = nothing,
             w2::Union{<:AbstractWeights, Nothing} = nothing,
             mu::Union{<:AbstractVector, Nothing} = nothing)
    return MAD(settings, w1, w2, mu)
end
function (mad::MAD)(x::AbstractVector)
    w1 = mad.w1
    w2 = mad.w2
    mu = isnothing(w1) ? mean(x) : mean(x, w1)
    return isnothing(w2) ? mean(abs.(x .- mu)) : mean(abs.(x .- mu), Weights(2 * w2))
end

"""
    mutable struct SSD{T1 <: Real} <: RiskMeasure

Measures and computes the portfolio Semi Standard Deviation (SD) below a `target` value. In other words, it measures the standard deviation below a `target` value while ignoring deviations above it.

```math
\\begin{align}
\\mathrm{SSD}(\\bm{X}) &= \\left(\\dfrac{1}{T-1} \\sum\\limits_{t=1}^{T}\\min\\left(X_{t} - \\mathbb{E}(\\bm{X}),\\, r\\right)^{2}\\right)^{1/2}\\,.
\\end{align}
```

See also: [`RiskMeasure`](@ref), [`RMSettings`](@ref), [`Portfolio`](@ref), [`SD`](@ref), [`Variance`](@ref), [`SVariance`](@ref), [`calc_risk`](@ref), [`optimise!`](@ref), [`set_rm`](@ref).

# Keyword Parameters

  - `settings::RMSettings = RMSettings()`: risk measure configuration settings.

  - `target::T1 <: Real = 0.0`: minimum return threshold for classifying downside returns. Only returns equal to or below this value are considered in the calculation. Must be in the same frequency as the returns.
  - `w::Union{<:AbstractWeights, Nothing} = nothing`: optional, `T×1` vector of weights for computing the expected value of the returns vector (internal expected value).

      + `w` has no effect in optimisations using [`JuMP`](https://github.com/jump-dev/JuMP.jl) models. However, it can be taken into account if `mu` paramter is computed with the [`MuSimple`](@ref) estimator and the would-be value of `w` as the weight vector.
  - `mu::Union{<:AbstractVector, Nothing} = nothing`: optional, vector of expected returns.

      + If `nothing`: takes its value from the `mu` instance [`Portfolio`](@ref).
      + Only used in optimisations using [`JuMP`](https://github.com/jump-dev/JuMP.jl) models. Other optimisations, as well as the risk calculation, compute its value via the functor.

# Functor

  - `(ssd::SSD)(x::AbstractVector)`: computes the semi standard deviation using the internal values of `target` and `w`, for a `T×1` vector of portfolio returns, `x`.

!!! warning

    Using non-default `target`, `w`, and/or `mu` in optimisations using [`JuMP`](https://github.com/jump-dev/JuMP.jl) may lead to the value computed by the functor to be different than that of `sdev_risk`. This is because the [`JuMP`](https://github.com/jump-dev/JuMP.jl) model is based on slack constraints, and the functor computes the internal expected return.

# Examples
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
function (ssd::SSD)(x::AbstractVector)
    T = length(x)
    w = ssd.w
    target = ssd.target
    mu = isnothing(w) ? mean(x) : mean(x, w)
    val = x .- mu
    return sqrt(sum(val[val .<= target] .^ 2) / (T - 1))
end

"""
    mutable struct FLPM{T1 <: Real} <: RiskMeasure

# Description

First Lower Partial Moment (Omega ratio) risk measure.

  - Measures the dispersion equal to or below the `target` return threshold.

```math
\\begin{align}
\\mathrm{FLPM}(\\bm{X},\\, r) &= \\dfrac{1}{T} \\sum\\limits_{t=1}^{T}\\max\\left(r - X_{t},\\, 0\\right)\\,.
\\end{align}
```

See also: [`RiskMeasure`](@ref), [`RMSettings`](@ref), [`SLPM`](@ref), [`Portfolio`](@ref), [`optimise!`](@ref), [`set_rm`](@ref), [`calc_risk(::FLPM, ::AbstractVector)`](@ref).

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
    ret_target::Union{<:Real, AbstractVector{<:Real}}
    target::T1
    w::Union{AbstractWeights, Nothing}
end
function FLPM(; settings::RMSettings = RMSettings(),
              ret_target::Union{<:Real, AbstractVector{<:Real}} = 0.0, target::Real = 0.0,
              w::Union{AbstractWeights, Nothing} = nothing)
    return FLPM{typeof(target)}(settings, ret_target, target, w)
end
function (flpm::FLPM)(x::AbstractVector)
    T = length(x)
    target = flpm.target
    target = if !isinf(target)
        target
    else
        w = flpm.w
        isnothing(w) ? mean(x) : mean(x, w)
    end
    val = x .- target
    return -sum(val[val .<= zero(target)]) / T
end

"""
    mutable struct SLPM{T1 <: Real} <: RiskMeasure

# Description

Second Lower Partial Moment (Sortino ratio) risk measure.

  - Measures the dispersion equal to or below the `target` return threshold.

```math
\\begin{align}
\\mathrm{SLPM}(\\bm{X},\\, r) &= \\left(\\dfrac{1}{T-1} \\sum\\limits_{t=1}^{T}\\max\\left(r - X_{t},\\, 0\\right)^{2}\\right)^{1/2}\\,.
\\end{align}
```

See also: [`RiskMeasure`](@ref), [`RMSettings`](@ref), [`FLPM`](@ref), [`Portfolio`](@ref), [`optimise!`](@ref), [`set_rm`](@ref), [`calc_risk(::SLPM, ::AbstractVector)`](@ref).

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
    ret_target::Union{<:Real, AbstractVector{<:Real}}
    target::T1
    w::Union{AbstractWeights, Nothing}
end
function SLPM(; settings::RMSettings = RMSettings(),
              ret_target::Union{<:Real, AbstractVector{<:Real}} = 0.0, target::Real = 0.0,
              w::Union{AbstractWeights, Nothing} = nothing)
    return SLPM{typeof(target)}(settings, ret_target, target, w)
end
function (slpm::SLPM)(x::AbstractVector)
    T = length(x)
    target = slpm.target
    target = if !isinf(target)
        target
    else
        w = slpm.w
        isnothing(w) ? mean(x) : mean(x, w)
    end
    val = x .- target
    val = val[val .<= zero(target)]
    return sqrt(dot(val, val) / (T - 1))
end

"""
    struct WR <: RiskMeasure

# Description

Worst Realization/Return risk measure.

  - Useful for extremely conservative risk assessment.
  - ``\\mathrm{VaR}(\\bm{X},\\, \\alpha) \\leq \\mathrm{CVaR}(\\bm{X},\\, \\alpha) \\leq \\mathrm{EVaR}(\\bm{X},\\, \\alpha) \\leq \\mathrm{RLVaR}(\\bm{X},\\, \\alpha,\\, \\kappa) \\leq \\mathrm{WR}(\\bm{X})``.

```math
\\begin{align}
\\mathrm{WR}(\\bm{X}) &= -\\min(\\bm{X})\\,.
\\end{align}
```

See also: [`RiskMeasure`](@ref), [`RMSettings`](@ref), [`Portfolio`](@ref), [`optimise!`](@ref), [`set_rm`](@ref), [`calc_risk(::WR, ::AbstractVector)`](@ref), [`VaR`](@ref), [`CVaR`](@ref), [`EVaR`](@ref), [`RLVaR`](@ref), [`DaR`](@ref), [`DaR_r`](@ref), [`CDaR`](@ref), [`CDaR_r`](@ref), [`EDaR`](@ref), [`EDaR_r`](@ref), [`RLDaR`](@ref), [`RLDaR_r`](@ref), [`MDD`](@ref), [`MDD_r`](@ref).

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
function (wr::WR)(x::AbstractVector)
    return -minimum(x)
end

"""
    mutable struct CVaR{T1 <: Real} <: RiskMeasure

# Description

Conditional Value at Risk (Expected Shortfall) risk measure.

  - Measures expected loss in the worst `alpha %` of cases.
  - ``\\mathrm{VaR}(\\bm{X},\\, \\alpha) \\leq \\mathrm{CVaR}(\\bm{X},\\, \\alpha) \\leq \\mathrm{EVaR}(\\bm{X},\\, \\alpha) \\leq \\mathrm{RLVaR}(\\bm{X},\\, \\alpha,\\, \\kappa) \\leq \\mathrm{WR}(\\bm{X})``.

```math
\\begin{align}
\\mathrm{CVaR}(\\bm{X},\\, \\alpha) &= \\mathrm{VaR}(\\bm{X},\\, \\alpha) + \\dfrac{1}{\\alpha T} \\sum\\limits_{t=1}^{T} \\max\\left(-X_{t} - \\mathrm{VaR}(\\bm{X},\\, \\alpha),\\, 0\\right)\\,.
\\end{align}
```

Where:

  - ``\\mathrm{VaR}(\\bm{X},\\, \\alpha)`` is the Value at Risk as defined in [`VaR`](@ref).

See also: [`RiskMeasure`](@ref), [`RMSettings`](@ref), [`Portfolio`](@ref), [`optimise!`](@ref), [`set_rm`](@ref), [`calc_risk(::CVaR, ::AbstractVector)`](@ref), [`VaR`](@ref), [`EVaR`](@ref), [`RLVaR`](@ref), [`WR`](@ref), [`DaR`](@ref), [`DaR_r`](@ref), [`CDaR`](@ref), [`CDaR_r`](@ref), [`EDaR`](@ref), [`EDaR_r`](@ref), [`RLDaR`](@ref), [`RLDaR_r`](@ref), [`MDD`](@ref), [`MDD_r`](@ref).

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
function (cvar::CVaR)(x::AbstractVector)
    alpha = cvar.alpha
    aT = alpha * length(x)
    idx = ceil(Int, aT)
    var = -partialsort!(x, idx)
    sum_var = 0.0
    for i ∈ 1:(idx - 1)
        sum_var += x[i] + var
    end
    return var - sum_var / aT
end

mutable struct DRCVaR{T1, T2, T3} <: RiskMeasure
    settings::RMSettings
    l::T1
    alpha::T2
    r::T3
end
function DRCVaR(; settings::RMSettings = RMSettings(), l::Real = 1.0, alpha::Real = 0.05,
                r::Real = 0.02)
    @smart_assert(zero(alpha) < alpha < one(alpha))
    return DRCVaR{typeof(l), typeof(alpha), typeof(r)}(settings, l, alpha, r)
end
function Base.setproperty!(obj::DRCVaR, sym::Symbol, val)
    if sym == :alpha
        @smart_assert(zero(val) < val < one(val))
    end
    return setfield!(obj, sym, val)
end
function (drcvar::DRCVaR)(x::AbstractVector)
    alpha = drcvar.alpha
    aT = alpha * length(x)
    idx = ceil(Int, aT)
    var = -partialsort!(x, idx)
    sum_var = 0.0
    for i ∈ 1:(idx - 1)
        sum_var += x[i] + var
    end
    return var - sum_var / aT
end

"""
    mutable struct EVaR{T1 <: Real} <: RiskMeasure

# Description

Entropic Value at Risk risk measure.

  - It is the upper bound of the Chernoff inequality for the [`VaR`](@ref) and [`CVaR`](@ref).
  - ``\\mathrm{VaR}(\\bm{X},\\, \\alpha) \\leq \\mathrm{CVaR}(\\bm{X},\\, \\alpha) \\leq \\mathrm{EVaR}(\\bm{X},\\, \\alpha) \\leq \\mathrm{RLVaR}(\\bm{X},\\, \\alpha,\\, \\kappa) \\leq \\mathrm{WR}(\\bm{X})``.

```math
\\begin{align}
\\mathrm{EVaR}(\\bm{X},\\alpha) &= \\underset{z > 0}{\\inf} \\left\\{\\mathrm{ERM}(\\bm{X},\\, z, \\,\\alpha)\\right\\}\\,.
\\end{align}
```

Where:

  - ``\\mathrm{ERM}(\\bm{X},\\, z, \\,\\alpha)`` is the entropic risk measure as defined in [`ERM`](@ref).

See also: [`RiskMeasure`](@ref), [`RMSettings`](@ref), [`Portfolio`](@ref), [`optimise!`](@ref), [`set_rm`](@ref), [`calc_risk(::EVaR, ::AbstractVector)`](@ref), [`VaR`](@ref), [`CVaR`](@ref), [`RLVaR`](@ref), [`WR`](@ref), [`DaR`](@ref), [`DaR_r`](@ref), [`CDaR`](@ref), [`CDaR_r`](@ref), [`EDaR`](@ref), [`EDaR_r`](@ref), [`RLDaR`](@ref), [`RLDaR_r`](@ref), [`MDD`](@ref), [`MDD_r`](@ref).

# Fields

  - `settings::RMSettings = RMSettings()`: configuration settings for the risk measure.
  - `alpha::T1 = 0.05`: significance level, `alpha ∈ (0, 1)`.
  - `solvers::Union{<:AbstractDict, Nothing}`: optional JuMP-compatible solvers for exponential cone problems.

# Behaviour

  - Requires solver capability for exponential cone problems.

  - When computing [`calc_risk(::EVaR, ::AbstractVector)`](@ref):

      + If `solvers` is `nothing`: uses `solvers` from [`Portfolio`](@ref)/.
      + If `solvers` is provided: use the solvers.

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
function (evar::EVaR)(x::AbstractVector)
    return ERM(x, evar.solvers, evar.alpha)
end

"""
    mutable struct RLVaR{T1 <: Real, T2 <: Real} <: RiskMeasure

# Description

Relativistic Value at Risk risk measure.

  - It is a generalisation of the [`EVaR`](@ref).
  - ``\\mathrm{VaR}(\\bm{X},\\, \\alpha) \\leq \\mathrm{CVaR}(\\bm{X},\\, \\alpha) \\leq \\mathrm{EVaR}(\\bm{X},\\, \\alpha) \\leq \\mathrm{RLVaR}(\\bm{X},\\, \\alpha,\\, \\kappa) \\leq \\mathrm{WR}(\\bm{X})``.
  - ``\\lim\\limits_{\\kappa \\to 0} \\mathrm{RLVaR}(\\bm{X},\\, \\alpha,\\, \\kappa) \\approx \\mathrm{EVaR}(\\bm{X},\\, \\alpha)``
  - ``\\lim\\limits_{\\kappa \\to 1} \\mathrm{RLVaR}(\\bm{X},\\, \\alpha,\\, \\kappa) \\approx \\mathrm{WR}(\\bm{X})``

```math
\\begin{align}
\\mathrm{RLVaR}(\\bm{X},\\, \\alpha,\\, \\kappa) &= \\mathrm{RRM}(\\bm{X},\\, \\alpha,\\, \\kappa)\\,.
\\end{align}
```

Where:

  - ``\\mathrm{RRM}(\\bm{X},\\, \\alpha,\\, \\kappa)`` is the Relativistic Risk Measure as defined in [`RRM`](@ref).

See also: [`RiskMeasure`](@ref), [`RMSettings`](@ref), [`Portfolio`](@ref), [`optimise!`](@ref), [`set_rm`](@ref), [`calc_risk(::RLVaR, ::AbstractVector)`](@ref), [`VaR`](@ref), [`CVaR`](@ref), [`EVaR`](@ref), [`WR`](@ref), [`DaR`](@ref), [`DaR_r`](@ref), [`CDaR`](@ref), [`CDaR_r`](@ref), [`EDaR`](@ref), [`EDaR_r`](@ref), [`RLDaR`](@ref), [`RLDaR_r`](@ref), [`MDD`](@ref), [`MDD_r`](@ref).

# Fields

  - `settings::RMSettings = RMSettings()`: configuration settings for the risk measure.
  - `alpha::T1 = 0.05`: significance level, `alpha ∈ (0, 1)`.
  - `kappa::T1 = 0.3`: significance level, `kappa ∈ (0, 1)`.
  - `solvers::Union{<:AbstractDict, Nothing}`: optional JuMP-compatible solvers for 3D power cone problems.

# Behaviour

  - Requires solver capability for 3D power cone problems.

  - When computing [`calc_risk(::RLVaR, ::AbstractVector)`](@ref):

      + If `solvers` is `nothing`: uses `solvers` from [`Portfolio`](@ref)/.
      + If `solvers` is provided: use the solvers.

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
function (rlvar::RLVaR)(x::AbstractVector)
    return RRM(x, rlvar.solvers, rlvar.alpha, rlvar.kappa)
end

"""
    struct MDD <: RiskMeasure

# Description

Maximum Drawdown (Calmar ratio) risk measure for uncompounded cumulative returns.

  - Measures the largest peak-to-trough decline.
  - ``\\mathrm{DaR}(\\bm{X},\\, \\alpha) \\leq \\mathrm{CDaR}(\\bm{X},\\, \\alpha) \\leq \\mathrm{EDaR}(\\bm{X},\\, \\alpha) \\leq \\mathrm{RLDaR}(\\bm{X},\\, \\alpha,\\, \\kappa) \\leq \\mathrm{MDD}(\\bm{X})``.

```math
\\begin{align}
\\mathrm{MDD_{a}}(\\bm{X}) &= \\max\\mathrm{DD_{a}}(\\bm{X})\\,.
\\end{align}
```

Where:

  - ``\\mathrm{DD_{a}}(\\bm{X})`` is the Drawdown of uncompounded cumulative returns as defined in [`DaR`](@ref).

See also: [`RiskMeasure`](@ref), [`RMSettings`](@ref), [`Portfolio`](@ref), [`optimise!`](@ref), [`set_rm`](@ref), [`calc_risk(::MDD, ::AbstractVector)`](@ref), [`VaR`](@ref), [`CVaR`](@ref), [`EVaR`](@ref), [`RLVaR`](@ref) [`WR`](@ref), [`DaR`](@ref), [`DaR_r`](@ref), [`CDaR`](@ref), [`CDaR_r`](@ref), [`EDaR`](@ref), [`EDaR_r`](@ref), [`RLDaR`](@ref), [`RLDaR_r`](@ref), [`MDD_r`](@ref).

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
function (dar::MDD)(x::AbstractVector)
    pushfirst!(x, 1)
    cs = cumsum(x)
    val = 0.0
    peak = -Inf
    for i ∈ cs
        if i > peak
            peak = i
        end
        dd = peak - i
        if dd > val
            val = dd
        end
    end
    popfirst!(x)
    return val
end

"""
    struct ADD <: RiskMeasure

# Description

Average Drawdown risk measure for uncompounded cumulative returns.

  - Measures the average of all peak-to-trough declines.
  - Provides a more balanced view than the maximum drawdown [`MDD`](@ref).

```math
\\begin{align}
\\mathrm{ADD_{a}}(\\bm{X}) &= \\dfrac{1}{T} \\sum\\limits_{j=1}^{T} \\mathrm{DD_{a}}(\\bm{X}, j)\\,.
\\end{align}
```

Where:

  - ``\\mathrm{DD_{a}}(\\bm{X}, j)`` is the Drawdown of uncompounded cumulative returns at time ``j`` as defined in [`DaR`](@ref).

See also: [`RiskMeasure`](@ref), [`RMSettings`](@ref), [`Portfolio`](@ref), [`optimise!`](@ref), [`set_rm`](@ref), [`calc_risk(::ADD, ::AbstractVector)`](@ref), [`ADD_r`](@ref), [`MDD`](@ref).

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
    w::Union{<:AbstractWeights, Nothing}
end
function ADD(; settings::RMSettings = RMSettings(),
             w::Union{<:AbstractWeights, Nothing} = nothing)
    return ADD(settings, w)
end
function (add::ADD)(x::AbstractVector)
    T = length(x)
    w = add.w
    pushfirst!(x, 1)
    cs = cumsum(x)
    val = 0.0
    peak = -Inf
    if isnothing(w)
        for i ∈ cs
            if i > peak
                peak = i
            end
            dd = peak - i
            if dd > 0
                val += dd
            end
        end
        popfirst!(x)
        return val / T
    else
        @smart_assert(length(w) == T)
        for i ∈ cs
            if i > peak
                peak = i
            end
            dd = peak - i
            if dd > 0
                val += dd * w[i]
            end
        end
        popfirst!(x)
        return val / sum(w)
    end
end

"""
    mutable struct CDaR{T1 <: Real} <: RiskMeasure

# Description

Conditional Drawdown at Risk risk measure for uncompounded cumulative returns.

  - Measures the expected peak-to-trough loss in the worst `alpha %` of cases.
  - ``\\mathrm{DaR}(\\bm{X},\\, \\alpha) \\leq \\mathrm{CDaR}(\\bm{X},\\, \\alpha) \\leq \\mathrm{EDaR}(\\bm{X},\\, \\alpha) \\leq \\mathrm{RLDaR}(\\bm{X},\\, \\alpha,\\, \\kappa) \\leq \\mathrm{MDD}(\\bm{X})``.

```math
\\begin{align}
\\mathrm{CDaR_{a}}(\\bm{X},\\, \\alpha) &= \\mathrm{DaR_{a}}(\\bm{X},\\, \\alpha) + \\dfrac{1}{\\alpha T} \\sum\\limits_{j=0}^{T} \\max\\left(\\mathrm{DD_{a}}(\\bm{X},\\, j) - \\mathrm{DaR_{a}}(\\bm{X},\\, \\alpha),\\, 0 \\right)\\,.
\\end{align}
```

Where:

  - ``\\mathrm{DD_{a}}(\\bm{X},\\, j)`` is the Drawdown of uncompounded cumulative returns at time ``j`` as defined in [`DaR`](@ref).
  - ``\\mathrm{DaR_{a}}(\\bm{X},\\, \\alpha)`` is the Drawdown at Risk of uncompounded cumulative returns as defined in [`DaR`](@ref).

See also: [`RiskMeasure`](@ref), [`RMSettings`](@ref), [`Portfolio`](@ref), [`optimise!`](@ref), [`set_rm`](@ref), [`calc_risk(::CDaR, ::AbstractVector)`](@ref), [`VaR`](@ref), [`CVaR`](@ref), [`EVaR`](@ref), [`RLVaR`](@ref) [`WR`](@ref), [`DaR`](@ref), [`DaR_r`](@ref), [`CDaR_r`](@ref), [`EDaR`](@ref), [`EDaR_r`](@ref), [`RLDaR`](@ref), [`RLDaR_r`](@ref), [`MDD`](@ref), [`MDD_r`](@ref).

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
function (cdar::CDaR)(x::AbstractVector)
    T = length(x)
    alpha = cdar.alpha
    aT = alpha * T
    idx = ceil(Int, aT)
    pushfirst!(x, 1)
    cs = cumsum(x)
    peak = -Inf
    dd = similar(cs)
    for (idx, i) ∈ pairs(cs)
        if i > peak
            peak = i
        end
        dd[idx] = i - peak
    end
    popfirst!(x)
    popfirst!(dd)
    var = -partialsort!(dd, idx)
    sum_var = 0.0
    for i ∈ 1:(idx - 1)
        sum_var += dd[i] + var
    end
    return var - sum_var / aT
end

"""
    mutable struct UCI <: RiskMeasure

# Description

Ulcer Index risk measure for uncompounded cumulative returns.

  - Penalizes larger drawdowns more than smaller ones.

```math
\\begin{align}
\\mathrm{UCI_{a}}(\\bm{X}) &= \\left(\\dfrac{1}{T} \\sum\\limits_{j=0}^{T} \\mathrm{DD_{a}}(\\bm{X},\\, j)^{2}\\right)^{1/2}\\,.
\\end{align}
```

Where:

  - ``\\mathrm{DD_{a}}(\\bm{X},\\, j)`` is the Drawdown of uncompounded cumulative returns at time ``j`` as defined in [`DaR`](@ref).

See also: [`RiskMeasure`](@ref), [`RMSettings`](@ref), [`Portfolio`](@ref), [`optimise!`](@ref), [`set_rm`](@ref), [`calc_risk(::UCI, ::AbstractVector)`](@ref), [`UCI_r`](@ref).

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
function (uci::UCI)(x::AbstractVector)
    T = length(x)
    pushfirst!(x, 1)
    cs = cumsum(x)
    val = 0.0
    peak = -Inf
    for i ∈ cs
        if i > peak
            peak = i
        end
        dd = peak - i
        if dd > 0
            val += dd^2
        end
    end
    popfirst!(x)
    return sqrt(val / T)
end

"""
    mutable struct EDaR{T1 <: Real} <: RiskMeasure

# Description

Entropic Drawdown at Risk risk measure for uncompounded cumulative returns.

  - It is the upper bound of the Chernoff inequality for the [`DaR`](@ref) and [`CDaR`](@ref).
  - ``\\mathrm{DaR}(\\bm{X},\\, \\alpha) \\leq \\mathrm{CDaR}(\\bm{X},\\, \\alpha) \\leq \\mathrm{EDaR}(\\bm{X},\\, \\alpha) \\leq \\mathrm{RLDaR}(\\bm{X},\\, \\alpha,\\, \\kappa) \\leq \\mathrm{MDD}(\\bm{X})``.

```math
\\begin{align}
\\mathrm{EDaR_{a}}(\\bm{X},\\alpha) &= \\underset{z > 0}{\\inf} \\left\\{\\mathrm{ERM}(\\mathrm{DD_{a}}(\\bm{X}),\\, z, \\,\\alpha)\\right\\}\\,.
\\end{align}
```

Where:

  - ``\\mathrm{ERM}(\\mathrm{DD_{a}}(\\bm{X}),\\, z, \\,\\alpha)`` is the Entropic Risk Measure as defined in [`ERM`](@ref), using the Drawdown of uncompounded cumulative returns as defined in [`DaR`](@ref).

See also: [`RiskMeasure`](@ref), [`RMSettings`](@ref), [`Portfolio`](@ref), [`optimise!`](@ref), [`set_rm`](@ref), [`calc_risk(::EDaR, ::AbstractVector)`](@ref), [`VaR`](@ref), [`CVaR`](@ref), [`EVaR`](@ref), [`RLVaR`](@ref) [`WR`](@ref), [`DaR`](@ref), [`DaR_r`](@ref), [`CDaR`](@ref), [`CDaR_r`](@ref), [`EDaR_r`](@ref), [`RLDaR`](@ref), [`RLDaR_r`](@ref), [`MDD`](@ref), [`MDD_r`](@ref).

# Fields

  - `settings::RMSettings = RMSettings()`: configuration settings for the risk measure.
  - `alpha::T1 = 0.05`: significance level, `alpha ∈ (0, 1)`.
  - `solvers::Union{<:AbstractDict, Nothing}`: optional JuMP-compatible solvers for exponential cone problems.

# Behaviour

  - Requires solver capability for exponential cone problems.

  - When computing [`calc_risk(::EDaR, ::AbstractVector)`](@ref):

      + If `solvers` is `nothing`: uses `solvers` from [`Portfolio`](@ref)/.
      + If `solvers` is provided: use the solvers.

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
function (edar::EDaR)(x::AbstractVector)
    pushfirst!(x, 1)
    cs = cumsum(x)
    peak = -Inf
    dd = similar(cs)
    for (idx, i) ∈ pairs(cs)
        if i > peak
            peak = i
        end
        dd[idx] = -(peak - i)
    end
    popfirst!(x)
    popfirst!(dd)
    return ERM(dd, edar.solvers, edar.alpha)
end

"""
    mutable struct RLDaR{T1 <: Real, T2 <: Real} <: RiskMeasure

# Description

Relativistic Drawdown at Risk risk measure for uncompounded cumulative returns.

  - It is a generalisation of the [`EDaR`](@ref).
  - ``\\mathrm{DaR}(\\bm{X},\\, \\alpha) \\leq \\mathrm{CDaR}(\\bm{X},\\, \\alpha) \\leq \\mathrm{EDaR}(\\bm{X},\\, \\alpha) \\leq \\mathrm{RLDaR}(\\bm{X},\\, \\alpha,\\, \\kappa) \\leq \\mathrm{MDD}(\\bm{X})``.
  - ``\\lim\\limits_{\\kappa \\to 0} \\mathrm{RLDaR}(\\bm{X},\\, \\alpha,\\, \\kappa) \\approx \\mathrm{EDaR}(\\bm{X},\\, \\alpha)``
  - ``\\lim\\limits_{\\kappa \\to 1} \\mathrm{RLDaR}(\\bm{X},\\, \\alpha,\\, \\kappa) \\approx \\mathrm{MDD}(\\bm{X})``

```math
\\begin{align}
\\mathrm{RLDaR_{a}}(\\bm{X},\\, \\alpha,\\, \\kappa) &= \\mathrm{RRM}(\\mathrm{DD_{a}}(\\bm{X}),\\, \\alpha,\\, \\kappa)\\,.
\\end{align}
```

Where:

  - ``\\mathrm{RRM}(\\mathrm{DD_{a}}(\\bm{X}),\\, \\alpha,\\, \\kappa)`` is the Relativistic Risk Measure as defined in [`RRM`](@ref), using the Drawdown of uncompounded cumulative returns as defined in [`DaR`](@ref).

See also: [`RiskMeasure`](@ref), [`RMSettings`](@ref), [`Portfolio`](@ref), [`optimise!`](@ref), [`set_rm`](@ref), [`calc_risk(::RLDaR, ::AbstractVector)`](@ref), [`VaR`](@ref), [`CVaR`](@ref), [`EVaR`](@ref), [`RLVaR`](@ref) [`WR`](@ref), [`DaR`](@ref), [`DaR_r`](@ref), [`CDaR`](@ref), [`CDaR_r`](@ref), [`EDaR`](@ref), [`EDaR_r`](@ref), [`RLDaR_r`](@ref), [`MDD`](@ref), [`MDD_r`](@ref).

# Fields

  - `settings::RMSettings = RMSettings()`: configuration settings for the risk measure.
  - `alpha::T1 = 0.05`: significance level, `alpha ∈ (0, 1)`.
  - `kappa::T1 = 0.3`: significance level, `kappa ∈ (0, 1)`.
  - `solvers::Union{<:AbstractDict, Nothing}`: optional JuMP-compatible solvers for 3D power cone problems.

# Behaviour

  - Requires solver capability for 3D power cone problems.

  - When computing [`calc_risk(::RLDaR, ::AbstractVector)`](@ref):

      + If `solvers` is `nothing`: uses `solvers` from [`Portfolio`](@ref)/.
      + If `solvers` is provided: use the solvers.

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
function (rldar::RLDaR)(x::AbstractVector)
    pushfirst!(x, 1)
    cs = cumsum(x)
    peak = -Inf
    dd = similar(cs)
    for (idx, i) ∈ pairs(cs)
        if i > peak
            peak = i
        end
        dd[idx] = i - peak
    end
    popfirst!(x)
    popfirst!(dd)
    return RRM(dd, rldar.solvers, rldar.alpha, rldar.kappa)
end

"""
    mutable struct Kurt <: RiskMeasure

# Description

Square Root Kurtosis risk measure implementation for portfolio optimisation.

```math
\\begin{align}
\\mathrm{Kurt}(\\bm{X}) &= \\left(\\dfrac{1}{T} \\sum\\limits_{t=1}^{T} \\left( X_{t} - \\mathbb{E}(\\bm{X}) \\right)^{4} \\right)^{1/2}\\,.
\\end{align}
```

See also: [`RiskMeasure`](@ref), [`RMSettings`](@ref), [`Portfolio`](@ref), [`optimise!`](@ref), [`set_rm`](@ref), [`calc_risk(::Kurt, ::AbstractVector)`](@ref), [`SKurt`](@ref).

# Fields

  - `settings::RMSettings = RMSettings()`: configuration settings for the risk measure.
  - `w::Union{<:AbstractWeights, Nothing} = nothing`: optional `T×1` vector of weights for expected return calculation.
  - `kt::Union{AbstractMatrix, Nothing} = nothing`: optional cokurtosis matrix.

# Behaviour

  - If `kt` is `nothing`: uses the semi cokurtosis matrix `skurt` from the [`Portfolio`](@ref)/object.
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
function (kurt::Kurt)(x::AbstractVector, scale::Bool = false)
    T = length(x)
    w = kurt.w
    mu = isnothing(w) ? mean(x) : mean(x, w)
    val = x .- mu
    kurt = sqrt(sum(val .^ 4) / T)
    return !scale ? kurt : kurt / 2
end

"""
    mutable struct SKurt{T1 <: Real} <: RiskMeasure

# Description

Square Root Semi Kurtosis risk measure implementation for portfolio optimisation.

  - Measures the kurtosis equal to or below the `target` return threshold.

```math
\\begin{align}
\\mathrm{SKurt}(\\bm{X}) &= \\left(\\dfrac{1}{T} \\sum\\limits_{t=1}^{T} \\min\\left( X_{t} - \\mathbb{E}(\\bm{X}),\\, r \\right)^{4} \\right)^{1/2}\\,.
\\end{align}
```

See also: [`RiskMeasure`](@ref), [`RMSettings`](@ref), [`Portfolio`](@ref), [`optimise!`](@ref), [`set_rm`](@ref), [`calc_risk(::SKurt, ::AbstractVector)`](@ref), [`Kurt`](@ref).

# Fields

  - `settings::RMSettings = RMSettings()`: configuration settings for the risk measure.
  - `target::T1 = 0.0`: minimum return threshold for downside classification.
  - `w::Union{<:AbstractWeights, Nothing} = nothing`: optional `T×1` vector of weights for expected return calculation.
  - `kt::Union{AbstractMatrix, Nothing} = nothing`: optional cokurtosis matrix.

# Behaviour

  - If `kt` is `nothing`: uses the cokurtosis from matrix `skurt` from the [`Portfolio`](@ref)/object.
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
function (skurt::SKurt)(x::AbstractVector, scale::Bool = false)
    T = length(x)
    w = skurt.w
    target = skurt.target
    mu = isnothing(w) ? mean(x) : mean(x, w)
    val = x .- mu
    skurt = sqrt(sum(val[val .<= target] .^ 4) / T)
    return !scale ? skurt : skurt / 2
end

"""
    mutable struct RG{T1 <: Real} <: RiskMeasure

# Description

Defines the Range risk measure.

  - Measures the best and worst returns, ``\\left[\\mathrm{WR}(\\bm{X}),\\, \\mathrm{WR}(-\\bm{X})\\right]``.

See also: [`RiskMeasure`](@ref), [`RMSettings`](@ref), [`Portfolio`](@ref), [`optimise!`](@ref), [`set_rm`](@ref), [`calc_risk(::RG, ::AbstractVector)`](@ref).

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
function (rg::RG)(x::AbstractVector)
    T = length(x)
    w = owa_rg(T)
    return dot(w, sort!(x))
end

"""
    mutable struct CVaRRG{T1 <: Real, T2 <: Real} <: RiskMeasure

# Description

Defines the Conditional Value at Risk Range risk measure.

  - Measures the range between the expected loss in the worst `alpha %` of cases and expected gain in the best `beta %` of cases, ``\\left[\\mathrm{CVaR}(\\bm{X},\\, \\alpha),\\, \\mathrm{CVaR}(-\\bm{X},\\, \\beta)\\right]``.

See also: [`RiskMeasure`](@ref), [`RMSettings`](@ref), [`Portfolio`](@ref), [`optimise!`](@ref), [`set_rm`](@ref), [`calc_risk(::CVaRRG, ::AbstractVector)`](@ref), [`CVaR`](@ref), [`RG`](@ref).

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
function (cvarrg::CVaRRG)(x::AbstractVector)
    T = length(x)
    w = owa_rcvar(T; alpha = cvarrg.alpha, beta = cvarrg.beta)
    return dot(w, sort!(x))
end

"""
    mutable struct OWASettings{T1 <: AbstractVector{<:Real}}

# Description

Defines the settings for Ordered Weight Array (OWA) risk measures.

See also: [`RiskMeasure`](@ref), [`RMSettings`](@ref), [`Portfolio`](@ref), [`optimise!`](@ref), [`set_rm`](@ref).

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

See also: [`RiskMeasure`](@ref), [`RMSettings`](@ref), [`OWASettings`](@ref), [`Portfolio`](@ref), [`optimise!`](@ref), [`set_rm`](@ref), [`calc_risk(::GMD, ::AbstractVector)`](@ref).

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
function (gmd::GMD)(x::AbstractVector)
    T = length(x)
    w = owa_gmd(T)
    return dot(w, sort!(x))
end

"""
    mutable struct TG{T1 <: Real, T2 <: Real, T3 <: Integer} <: RiskMeasure

# Description

Defines the Tail Gini Difference risk measure.

See also: [`RiskMeasure`](@ref), [`RMSettings`](@ref), [`OWASettings`](@ref), [`Portfolio`](@ref), [`optimise!`](@ref), [`set_rm`](@ref), [`calc_risk(::TG, ::AbstractVector)`](@ref).

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
function (tg::TG)(x::AbstractVector)
    T = length(x)
    w = owa_tg(T; alpha_i = tg.alpha_i, alpha = tg.alpha, a_sim = tg.a_sim)
    return dot(w, sort!(x))
end

"""
    mutable struct TGRG{T1 <: Real, T2 <: Real, T3 <: Integer, T4 <: Real, T5 <: Real, T6 <: Integer} <: RiskMeasure

# Description

Defines the Tail Gini Difference Range risk measure.

  - Measures the range between the worst `alpha %` tail gini of cases and best `beta %` tail gini of cases, ``\\left[\\mathrm{TG}(\\bm{X},\\, \\alpha),\\, \\mathrm{TG}(-\\bm{X},\\, \\beta)\\right]``.

See also: [`RiskMeasure`](@ref), [`RMSettings`](@ref), [`OWASettings`](@ref), [`Portfolio`](@ref), [`optimise!`](@ref), [`set_rm`](@ref), [`calc_risk(::TGRG, ::AbstractVector)`](@ref), [`TG`](@ref).

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
function (tgrg::TGRG)(x::AbstractVector)
    T = length(x)
    w = owa_rtg(T; alpha_i = tgrg.alpha_i, alpha = tgrg.alpha, a_sim = tgrg.a_sim,
                beta_i = tgrg.beta_i, beta = tgrg.beta, b_sim = tgrg.b_sim)
    return dot(w, sort!(x))
end

"""
    mutable struct OWA <: RiskMeasure

# Description

Defines the generic Ordered Weight Array risk measure.

  - Uses a vector of ordered weights generated by [`owa_l_moment`](@ref) or [`owa_l_moment_crm`](@ref) for arbitrary L-moment optimisations.

See also: [`RiskMeasure`](@ref), [`RMSettings`](@ref), [`OWASettings`](@ref), [`Portfolio`](@ref), [`optimise!`](@ref), [`set_rm`](@ref), [`calc_risk(::OWA, ::AbstractVector)`](@ref), [`owa_l_moment`](@ref), [`owa_l_moment_crm`](@ref).

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
function (owa::OWA)(x::AbstractVector)
    w = isnothing(owa.w) ? owa_gmd(length(x)) : owa.w
    return dot(w, sort!(x))
end

abstract type BDVarianceFormulation end
struct BDVAbsVal <: BDVarianceFormulation end
struct BDVIneq <: BDVarianceFormulation end
"""
    struct BDVariance <: RiskMeasure

# Description

Define the Brownian Distance Variance risk measure.

  - Measures linear and non-linear relationships between variables.

See also: [`RiskMeasure`](@ref), [`RMSettings`](@ref), [`OWASettings`](@ref), [`Portfolio`](@ref), [`optimise!`](@ref), [`set_rm`](@ref), [`calc_risk(::BDVariance, ::AbstractVector)`](@ref).

```math
\\begin{align}
\\mathrm{BDVariance}(\\bm{X}) &= \\mathrm{BDCov}(\\bm{X},\\, \\bm{X}) =  \\dfrac{1}{T^{2}} \\sum\\limits_{i=1}^{T}\\sum\\limits_{j=1}^{T} A_{i,\\,j}^2\\\\
\\mathrm{BDCov}(\\bm{X},\\, \\bm{Y}) &= \\dfrac{1}{T^{2}} \\sum\\limits_{i=1}^{T} \\sum\\limits_{j=1}^{T} A_{i,\\,j} B_{i,\\,j}\\\\
A_{i,\\,j} &= a_{i,\\,j} - \\bar{a}_{i\\,.} - \\bar{a}_{.\\,j} + \\bar{a}_{.\\,.}\\\\
B_{i,\\,j} &= b_{i,\\,j} - \\bar{b}_{i\\,.} - \\bar{b}_{.\\,j} + \\bar{b}_{.\\,.}\\\\
a_{i,\\,j} &= \\lVert X_{i} - X_{j} \\rVert_{2}, \\quad \\forall i,\\, j = 1,\\, \\ldots ,\\, T\\\\
b_{i,\\,j} &= \\lVert Y_{i} - Y_{j} \\rVert_{2}, \\quad \\forall i,\\, j = 1,\\, \\ldots ,\\, T\\,.
\\end{align}
```

where:

  - ``\\bm{X}`` and ``\\bm{Y}`` are random variables, they are equal in this case as they are the portfolio returns.
  - ``a_{i,\\,j}`` and ``b_{i,\\,j}`` are entries of a distance matrix where ``i`` and ``j`` are points in time. Each entry is defined as the Euclidean distance ``\\lVert \\cdot \\rVert_{2}`` between the value of the random variable at time ``i`` and its value at time ``j``.
  - ``\\bar{a}_{i,\\,\\cdot}`` and ``\\bar{b}_{i,\\,\\cdot}`` are the ``i``-th row means of their respective matrices.
  - ``\\bar{a}_{\\cdot,\\,j}`` and ``\\bar{b}_{\\cdot,\\,j}`` are the ``j``-th column means of their respective matrices.
  - ``\\bar{a}_{\\cdot,\\,\\cdot}`` and ``\\bar{b}_{\\cdot,\\,\\cdot}`` are the grand means of their respective matrices.
  - ``A_{i,\\,j}`` and ``B_{i,\\,j}`` are doubly centered distances.

# Fields

  - `settings::RMSettings = RMSettings()`: configuration settings for the risk measure.

# Examples

```@example
# Default settings
dvar = BDVariance()

# Custom settings
dvar = BDVariance(; settings = RMSettings(; ub = 0.5))
```
"""
struct BDVariance <: RiskMeasure
    settings::RMSettings
    formulation::BDVarianceFormulation
end
function BDVariance(; settings::RMSettings = RMSettings(),
                    formulation::BDVarianceFormulation = BDVAbsVal())
    return BDVariance(settings, formulation)
end
function (bdvariance::BDVariance)(x::AbstractVector)
    T = length(x)
    iT2 = inv(T^2)
    D = Matrix{eltype(x)}(undef, T, T)
    D .= x
    D .-= transpose(x)
    D .= abs.(D)
    return iT2 * (dot(D, D) + iT2 * sum(D)^2)
end

"""
    struct Skew <: RiskMeasure

# Description

Define the Quadratic Skewness risk measure.

```math
\\begin{align}
\\nu &= \\bm{w}^{\\intercal} \\mathbf{V} \\bm{w}\\\\
\\end{align}
```

Where:

  - ``\\bm{w}`` is the vector of asset weights.
  - ``\\mathbf{V}`` is the sum of the symmetric negative spectral slices of the coskewness.

See also: [`RiskMeasure`](@ref), [`RMSettings`](@ref), [`OWASettings`](@ref), [`Portfolio`](@ref), [`optimise!`](@ref), [`set_rm`](@ref), [`calc_risk(::Skew, ::AbstractVector)`](@ref).

# Fields

  - `settings::RMSettings = RMSettings()`: configuration settings for the risk measure.
  - `skew::Union{<:AbstractMatrix, Nothing}`: optional `N×N²` custom coskewness matrix.
  - `V::Union{Nothing, <:AbstractMatrix}`: optional `Na×Na` custom sum of the symmetric negative spectral slices of the coskewness.

# Behaviour

## Coskewness matrix usage

  - If `skew` is `nothing`:

      + With [`Portfolio`](@ref)/[`calc_risk`](@ref): no effect.
      + With : uses the portfolio coskewness matrix `skew` to generate the `V` matrix.

  - If `skew` provided:

      + With [`Portfolio`](@ref)/[`calc_risk`](@ref): no effect.
      + With : uses the custom coskew matrix to generate the `V` matrix.

## `V` matrix

  - If `V` is `nothing`:

      + With [`Portfolio`](@ref)/[`calc_risk`](@ref): uses the portfolio `V` matrix.
      + With : no effect.

  - If `V` provided:

      + With [`Portfolio`](@ref)/[`calc_risk`](@ref): uses the custom `V` matrix.
      + With : no effect.

## Validation

  - When setting `skew` at construction or runtime, the matrix must have dimensions (`N×N²`).
  - When setting `V` at construction or runtime, the matrix must be square (`N×N`).

# Examples

```@example
# Default settings
skew = Skew()

# Custom settings
skew = Skew(; settings = RMSettings(; ub = 0.5))
```
"""
mutable struct Skew <: RiskMeasure
    settings::RMSettings
    skew::Union{<:AbstractMatrix, Nothing}
    V::Union{<:AbstractMatrix, Nothing}
end
function Skew(; settings::RMSettings = RMSettings(),
              skew::Union{<:AbstractMatrix, Nothing} = nothing,
              V::Union{<:AbstractMatrix, Nothing} = nothing)
    if !isnothing(skew)
        @smart_assert(size(skew, 1)^2 == size(skew, 2))
    end
    if !isnothing(V)
        @smart_assert(size(V, 1) == size(V, 2))
    end
    return Skew(settings, skew, V)
end
function Base.setproperty!(obj::Skew, sym::Symbol, val)
    if sym == :skew
        if !isnothing(val)
            @smart_assert(size(val, 1)^2 == size(val, 2))
        end
    elseif sym == :V
        if !isnothing(val)
            @smart_assert(size(val, 1) == size(val, 2))
        end
    end
    return setfield!(obj, sym, val)
end
function (skew::Skew)(w::AbstractVector)
    return sqrt(dot(w, skew.V, w))
end

"""
    struct SSkew <: RiskMeasure

# Description

Define the Quadratic Semi Skewness risk measure.

```math
\\begin{align}
\\nu &= \\bm{w}^{\\intercal} \\mathbf{V} \\bm{w}\\\\
\\end{align}
```

Where:

  - ``\\bm{w}`` is the vector of asset weights.
  - ``\\mathbf{V}`` is the sum of the symmetric negative spectral slices of the semicoskewness.

See also: [`RiskMeasure`](@ref), [`RMSettings`](@ref), [`OWASettings`](@ref), [`Portfolio`](@ref), [`optimise!`](@ref), [`set_rm`](@ref), [`calc_risk(::SSkew, ::AbstractVector)`](@ref).

# Fields

  - `settings::RMSettings = RMSettings()`: configuration settings for the risk measure.
  - `skew::Union{<:AbstractMatrix, Nothing}`: optional `N×N²` custom semi coskewness matrix.
  - `V::Union{Nothing, <:AbstractMatrix}`: optional `Na×Na` custom sum of the symmetric negative spectral slices of the semi coskewness.

# Behaviour

## Coskewness matrix usage

  - If `skew` is `nothing`:

      + With [`Portfolio`](@ref)/[`calc_risk`](@ref): no effect.
      + With : uses the portfolio semi coskewness matrix `sskew` to generate the `V` matrix.

  - If `skew` provided:

      + With [`Portfolio`](@ref)/[`calc_risk`](@ref): no effect.
      + With : uses the custom semi coskew matrix to generate the `V` matrix.

## `V` matrix

  - If `V` is `nothing`:

      + With [`Portfolio`](@ref)/[`calc_risk`](@ref): uses the portfolio `SV` matrix.
      + With : no effect.

  - If `V` provided:

      + With [`Portfolio`](@ref)/[`calc_risk`](@ref): uses the custom `V` matrix.
      + With : no effect.

## Validation

  - When setting `skew` at construction or runtime, the matrix must have dimensions (`N×N²`).
  - When setting `V` at construction or runtime, the matrix must be square (`N×N`).

# Examples

```@example
# Default settings
sskew = SSkew()

# Custom settings
sskew = SSkew(; settings = RMSettings(; ub = 0.5))
```
"""
mutable struct SSkew <: RiskMeasure
    settings::RMSettings
    skew::Union{<:AbstractMatrix, Nothing}
    V::Union{Nothing, <:AbstractMatrix}
end
function SSkew(; settings::RMSettings = RMSettings(),
               skew::Union{<:AbstractMatrix, Nothing} = nothing,
               V::Union{<:AbstractMatrix, Nothing} = nothing)
    if !isnothing(skew)
        @smart_assert(size(skew, 1)^2 == size(skew, 2))
    end
    if !isnothing(V)
        @smart_assert(size(V, 1) == size(V, 2))
    end
    return SSkew(settings, skew, V)
end
function Base.setproperty!(obj::SSkew, sym::Symbol, val)
    if sym == :skew
        if !isnothing(val)
            @smart_assert(size(val, 1)^2 == size(val, 2))
        end
    elseif sym == :V
        if !isnothing(val)
            @smart_assert(size(val, 1) == size(val, 2))
        end
    end
    return setfield!(obj, sym, val)
end
function (sskew::SSkew)(w::AbstractVector)
    return sqrt(dot(w, sskew.V, w))
end

"""
    mutable struct Variance{T1 <: Union{<:AbstractMatrix, Nothing}} <: RiskMeasure

# Description

Defines the Variance risk measure.

```math
\\begin{align}
\\mathrm{Variance}(\\bm{w},\\, \\mathbf{\\Sigma}) &= \\bm{w}^\\intercal \\, \\mathbf{\\Sigma}\\, \\bm{w}\\,.
\\end{align}
```

See also: [`HCRMSettings`](@ref), [`optimise!`](@ref), [`calc_risk(::Variance, ::AbstractVector)`](@ref).

# Fields

  - `settings::HCRMSettings = HCRMSettings()`: hierarchical risk measure configuration settings.
  - `sigma::Union{<:AbstractMatrix, Nothing} = nothing`: optional covariance matrix.

# Behaviour

  - If `sigma` is `nothing`: uses the covariance matrix `cov` from the (or [`Portfolio`] when used in [`calc_risk`](@ref)) instance.
  - If `sigma` provided: uses custom covariance matrix.

## Validation

  - When setting `sigma` at construction or runtime, the matrix must be square (`N×N`).

# Examples

```@example
# Default settings
variance = Variance()

# Custom settings
variance = Variance(; settings = RMSettings(; scale = 3))
```
"""
mutable struct Variance <: RiskMeasure
    settings::RMSettings
    formulation::VarianceFormulation
    sigma::Union{<:AbstractMatrix, Nothing}
end
function Variance(; settings::RMSettings = RMSettings(),
                  formulation::VarianceFormulation = SOC(),
                  sigma::Union{<:AbstractMatrix, Nothing} = nothing)
    if !isnothing(sigma)
        @smart_assert(size(sigma, 1) == size(sigma, 2))
    end
    return Variance(settings, formulation, sigma)
end
function Base.setproperty!(obj::Variance, sym::Symbol, val)
    if sym == :sigma
        if !isnothing(val)
            @smart_assert(size(val, 1) == size(val, 2))
        end
    end
    return setfield!(obj, sym, val)
end
function (variance::Variance)(w::AbstractVector)
    return dot(w, variance.sigma, w)
end

"""
    mutable struct SVariance{T1 <: Real} <: RiskMeasure

# Description

Defines the Semi Variance risk measure.

```math
\\begin{align}
\\mathrm{SVariance}(\\bm{X}) &= \\dfrac{1}{T-1} \\sum\\limits_{t=1}^{T}\\min\\left(X_{t} - \\mathbb{E}(\\bm{X}),\\, r\\right)^{2}\\,.
\\end{align}
```

See also: [`HCRiskMeasure`](@ref), [`HCRMSettings`](@ref), [`optimise!`](@ref), [`calc_risk(::SVariance, ::AbstractVector)`](@ref).

# Fields

  - `settings::HCRMSettings = HCRMSettings()`: configuration settings for the risk measure.
  - `target::T1 = 0.0`: minimum return threshold for downside classification.
  - `w::Union{<:AbstractWeights, Nothing} = nothing`: optional `T×1` vector of weights for expected return calculation.

# Behaviour

  - If `w` is `nothing`: computes the unweighted mean portfolio return.

# Examples

```@example
# Default settings
svariance = SVariance()

# Custom configuration with specific target
w = eweights(1:100, 0.3)  # Exponential weights for computing the portfolio mean return
svariance = SVariance(; target = 0.02,  # 2 % return target
                      settings = HCRMSettings(; scale = 2.0), w = w)
```
"""
mutable struct SVariance{T1 <: Real} <: RiskMeasure
    settings::RMSettings
    formulation::VarianceFormulation
    target::T1
    w::Union{<:AbstractWeights, Nothing}
    mu::Union{<:AbstractVector, Nothing}
end
function SVariance(; settings::RMSettings = RMSettings(),
                   formulation::VarianceFormulation = SOC(), target::Real = 0.0,
                   w::Union{<:AbstractWeights, Nothing} = nothing,
                   mu::Union{<:AbstractVector, Nothing} = nothing)
    return SVariance{typeof(target)}(settings, formulation, target, w, mu)
end
function (svariance::SVariance)(x::AbstractVector)
    T = length(x)
    w = svariance.w
    target = svariance.target
    mu = isnothing(w) ? mean(x) : mean(x, w)
    val = x .- mu
    return sum(val[val .<= target] .^ 2) / (T - 1)
end

"""
```
abstract type WorstCaseSet end
```

Abstract type for subtyping worst case mean variance set types.
"""
abstract type WorstCaseSet end
abstract type WCSetMuSigma <: WorstCaseSet end
abstract type WCSetMu <: WorstCaseSet end

"""
```
struct Box <: WCSetMuSigma end
```

Box sets for worst case mean variance optimisation.
"""
struct Box <: WCSetMuSigma end

"""
```
struct Ellipse <: WCSetMuSigma end
```

Elliptical sets for worst case mean variance optimisation.
"""
struct Ellipse <: WCSetMuSigma end

"""
```
@kwdef mutable struct NoWC <: WorstCaseSet
    formulation::VarianceFormulation = SOC()
end
```

Use no set for worst case mean variance optimisation.

# Parameters

  - `formulation`: quadratic expression formulation of [`SD`](@ref) risk measure to use [`VarianceFormulation`](@ref).
"""
mutable struct NoWC <: WCSetMu
    formulation::VarianceFormulation
end
function NoWC(; formulation::VarianceFormulation = SOC())
    return NoWC(formulation)
end

mutable struct WCVariance{T1} <: RiskMeasure
    settings::RMSettings
    wc_set::WCSetMuSigma
    sigma::Union{<:AbstractMatrix, Nothing}
    cov_l::Union{AbstractMatrix{<:Real}, Nothing}
    cov_u::Union{AbstractMatrix{<:Real}, Nothing}
    cov_sigma::Union{AbstractMatrix{<:Real}, Nothing}
    k_sigma::T1
end
function WCVariance(; settings::RMSettings = RMSettings(), wc_set::WCSetMuSigma = Box(),
                    sigma::Union{<:AbstractMatrix{<:Real}, Nothing} = nothing,
                    cov_l::Union{AbstractMatrix{<:Real}, Nothing} = nothing,
                    cov_u::Union{AbstractMatrix{<:Real}, Nothing} = nothing,
                    cov_sigma::Union{AbstractMatrix{<:Real}, Nothing} = nothing,
                    k_sigma::Real = Inf)
    if !isnothing(sigma)
        @smart_assert(size(sigma, 1) == size(sigma, 2))
    end
    return WCVariance{typeof(k_sigma)}(settings, wc_set, sigma, cov_l, cov_u, cov_sigma,
                                       k_sigma)
end
function Base.setproperty!(obj::WCVariance, sym::Symbol, val)
    if sym ∈ (:sigma, :cov_l, :cov_u, :cov_mu, :cov_sigma)
        if !isnothing(val)
            @smart_assert(size(val, 1) == size(val, 2))
        end
    end
    return setfield!(obj, sym, val)
end

# ### Tracking

"""
```
abstract type TrackingErr end
```
"""
abstract type TrackingErr end

"""
```
struct NoTracking <: TrackingErr end
```
"""
struct NoTracking <: TrackingErr end

"""
```
@kwdef mutable struct TrackWeight{T1 <: Real, T2 <: AbstractVector{<:Real}} <: TrackingErr
    err::T1 = 0.0
    w::T2 = Vector{Float64}(undef, 0)
end
```
"""
mutable struct TrackWeight{T1 <: Real, T2 <: AbstractVector{<:Real}} <: TrackingErr
    err::T1
    w::T2
end
function TrackWeight(; err::Real = 0.0,
                     w::AbstractVector{<:Real} = Vector{Float64}(undef, 0))
    return TrackWeight{typeof(err), typeof(w)}(err, w)
end

"""
```
@kwdef mutable struct TrackRet{T1 <: Real, T2 <: AbstractVector{<:Real}} <: TrackingErr
    err::T1 = 0.0
    w::T2 = Vector{Float64}(undef, 0)
end
```
"""
mutable struct TrackRet{T1 <: Real, T2 <: AbstractVector{<:Real}} <: TrackingErr
    err::T1
    w::T2
end
function TrackRet(; err::Real = 0.0, w::AbstractVector{<:Real} = Vector{Float64}(undef, 0))
    return TrackRet{typeof(err), typeof(w)}(err, w)
end

# ### Turnover and rebalance

"""
```
abstract type AbstractTR end
```
"""
abstract type AbstractTR end

"""
```
struct NoTR <: AbstractTR end
```
"""
struct NoTR <: AbstractTR end

"""
```
@kwdef mutable struct TR{T1 <: Union{<:Real, <:AbstractVector{<:Real}},
                         T2 <: AbstractVector{<:Real}} <: AbstractTR
    val::T1 = 0.0
    w::T2 = Vector{Float64}(undef, 0)
end
```
"""
mutable struct TR{T1 <: Union{<:Real, <:AbstractVector{<:Real}},
                  T2 <: AbstractVector{<:Real}} <: AbstractTR
    val::T1
    w::T2
end
function TR(; val::Union{<:Real, <:AbstractVector{<:Real}} = 0.0,
            w::AbstractVector{<:Real} = Vector{Float64}(undef, 0))
    return TR{typeof(val), typeof(w)}(val, w)
end

mutable struct TrackingRM <: RiskMeasure
    settings::RMSettings
    tr::Union{TrackWeight, TrackRet}
end
function TrackingRM(; settings::RMSettings = RMSettings(),
                    tr::Union{TrackWeight, TrackRet} = TrackRet(;))
    return TrackingRM(settings, tr)
end
function (trackingRM::TrackingRM)(X::AbstractMatrix, w::AbstractVector)
    T = size(X, 1)
    tr = trackingRM.tr
    benchmark = tracking_error_benchmark(tr, X)
    return norm(X * w - benchmark) / sqrt(T - 1)
end

mutable struct TurnoverRM <: RiskMeasure
    settings::RMSettings
    tr::TR
end
function TurnoverRM(; settings::RMSettings = RMSettings(), tr::TR = TR(;))
    return TurnoverRM(settings, tr)
end
function (turnoverRM::TurnoverRM)(w::AbstractVector)
    benchmark = turnoverRM.tr.w
    return norm(benchmark - w, 1)
end
"""
    mutable struct VaR{T1 <: Real} <: HCRiskMeasure

# Description

Defines the Value at Risk risk measure.

  - Measures lower bound of the losses in the worst `alpha %` of cases.
  - ``\\mathrm{VaR}(\\bm{X},\\, \\alpha) \\leq \\mathrm{CVaR}(\\bm{X},\\, \\alpha) \\leq \\mathrm{EVaR}(\\bm{X},\\, \\alpha) \\leq \\mathrm{RLVaR}(\\bm{X},\\, \\alpha,\\, \\kappa) \\leq \\mathrm{WR}(\\bm{X})``.

```math
\\begin{align}
\\mathrm{VaR}(\\bm{X},\\, \\alpha) &= -\\underset{t \\in (0,\\, T)}{\\inf} \\left\\{ X_{t} \\in \\mathbb{R} \\, | \\, F_{\\bm{X}}(X_{t}) > \\alpha \\right\\}\\,.
\\end{align}
```

See also: [`HCRiskMeasure`](@ref), [`HCRMSettings`](@ref), [`optimise!`](@ref), [`calc_risk(::VaR, ::AbstractVector)`](@ref), [`CVaR`](@ref), [`EVaR`](@ref), [`RLVaR`](@ref) [`WR`](@ref), [`DaR`](@ref), [`DaR_r`](@ref), [`CDaR`](@ref), [`CDaR_r`](@ref), [`EDaR`](@ref), [`EDaR_r`](@ref), [`RLDaR`](@ref), [`RLDaR_r`](@ref), [`MDD`](@ref), [`MDD_r`](@ref).

# Fields

  - `settings::HCRMSettings = HCRMSettings()`: configuration settings for the risk measure.
  - `alpha::T1 = 0.05`: significance level, `alpha ∈ (0, 1)`.

# Behaviour

## Validation

  - When setting `alpha` at construction or runtime, `alpha ∈ (0, 1)`.

# Examples

```@example
# Default settings
var = VaR()

# Custom significance level
var = VaR(; settings = HCRMSettings(; scale = 1.0), alpha = 0.01)
```
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
function (var::VaR)(x::AbstractVector)
    alpha = var.alpha
    return -partialsort!(x, ceil(Int, alpha * length(x)))
end

"""
    mutable struct DaR{T1 <: Real} <: HCRiskMeasure

# Description

Defines the Drawdown at Risk for uncompounded cumulative returns risk measure.

  - Measures the lower bound of the peak-to-trough loss in the worst `alpha %` of cases.
  - ``\\mathrm{DaR}(\\bm{X},\\, \\alpha) \\leq \\mathrm{CDaR}(\\bm{X},\\, \\alpha) \\leq \\mathrm{EDaR}(\\bm{X},\\, \\alpha) \\leq \\mathrm{RLDaR}(\\bm{X},\\, \\alpha,\\, \\kappa) \\leq \\mathrm{MDD}(\\bm{X})``.

```math
\\begin{align}
\\mathrm{DaR_{a}}(\\bm{X},\\, \\alpha) &= \\underset{j \\in (0,\\, T)}{\\max} \\left\\{ \\mathrm{DD_{a}}(\\bm{X},\\, j) \\in \\mathbb{R} \\, | \\, F_{\\mathrm{DD}}\\left(\\mathrm{DD_{a}}(\\bm{X},\\, j)\\right) < 1 - \\alpha \\right\\}\\\\
\\mathrm{DD_{a}}(\\bm{X},\\, j) &= \\underset{t \\in (0,\\, j)}{\\max}\\left( \\sum\\limits_{i=0}^{t} X_{i} \\right) - \\sum\\limits_{i=0}^{j} X_{i}\\\\
\\mathrm{DD_{a}}(\\bm{X}) &= \\left\\{j \\in (0,\\,T) \\, | \\, \\mathrm{DD_{a}}(\\bm{X},\\, j)\\right\\}\\,.
\\end{align}
```

Where:

  - ``\\mathrm{DD_{a}}(\\bm{X})`` is the Drawdown of uncompounded cumulative returns.
  - ``\\mathrm{DD_{a}}(\\bm{X},\\, j)`` is the Drawdown of uncompounded cumulative returns at time ``j``.
  - ``\\mathrm{DaR_{a}}(\\bm{X},\\, \\alpha)`` the Drawdown at Risk of uncompounded cumulative returns.

See also: [`HCRiskMeasure`](@ref), [`HCRMSettings`](@ref), [`optimise!`](@ref), [`calc_risk(::DaR, ::AbstractVector)`](@ref), [`VaR`](@ref), [`CVaR`](@ref), [`EVaR`](@ref), [`RLVaR`](@ref) [`WR`](@ref), [`DaR_r`](@ref), [`CDaR`](@ref), [`CDaR_r`](@ref), [`EDaR`](@ref), [`EDaR_r`](@ref), [`RLDaR`](@ref), [`RLDaR_r`](@ref), [`MDD`](@ref), [`MDD_r`](@ref).

# Fields

  - `settings::HCRMSettings = HCRMSettings()`: configuration settings for the risk measure.
  - `alpha::T1 = 0.05`: significance level, `alpha ∈ (0, 1)`.

# Behaviour

## Validation

  - When setting `alpha` at construction or runtime, `alpha ∈ (0, 1)`.

# Examples

```@example
# Default settings
dar = DaR()

# Custom significance level
dar = DaR(; settings = HCRMSettings(; scale = 1.0), alpha = 0.01) # 1 % significance level
```
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
function (dar::DaR)(x::AbstractVector)
    T = length(x)
    alpha = dar.alpha
    pushfirst!(x, 1)
    cs = cumsum(x)
    peak = -Inf
    dd = similar(cs)
    for (idx, i) ∈ pairs(cs)
        if i > peak
            peak = i
        end
        dd[idx] = i - peak
    end
    popfirst!(x)
    popfirst!(dd)
    return -partialsort!(dd, ceil(Int, alpha * T))
end

"""
    mutable struct DaR_r{T1 <: Real} <: HCRiskMeasure

# Description

Defines the Drawdown at Risk for compounded cumulative returns risk measure.

  - Measures the lower bound of the peak-to-trough loss in the worst `alpha %` of cases.
  - ``\\mathrm{DaR_{r}}(\\bm{X},\\, \\alpha) \\leq \\mathrm{CDaR_{r}}(\\bm{X},\\, \\alpha) \\leq \\mathrm{EDaR_{r}}(\\bm{X},\\, \\alpha) \\leq \\mathrm{RLDaR_{r}}(\\bm{X},\\, \\alpha,\\, \\kappa) \\leq \\mathrm{MDD_{r}}(\\bm{X})``.

```math
\\begin{align}
\\mathrm{DaR_{r}}(\\bm{X},\\, \\alpha) &= \\underset{j \\in (0,\\, T)}{\\max} \\left\\{ \\mathrm{DD_{r}}(\\bm{X},\\, j) \\in \\mathbb{R} \\, | \\, F_{\\mathrm{DD}}\\left(\\mathrm{DD_{r}}(\\bm{X},\\, j)\\right) < 1 - \\alpha \\right\\}\\\\
\\mathrm{DD_{r}}(\\bm{X},\\, j) &= \\underset{t \\in (0,\\, j)}{\\max}\\left( \\prod\\limits_{i=0}^{t} \\left(1+X_{i}\\right) \\right) - \\prod\\limits_{i=0}^{j} \\left(1+X_{i}\\right)\\\\
\\mathrm{DD_{r}}(\\bm{X}) &= \\left\\{j \\in (0,\\,T) \\, | \\, \\mathrm{DD_{r}}(\\bm{X},\\, j)\\right\\}\\,.
\\end{align}
```

Where:

  - ``\\mathrm{DD_{r}}(\\bm{X})`` is the Drawdown of compounded cumulative returns.
  - ``\\mathrm{DD_{r}}(\\bm{X},\\, j)`` is the Drawdown of compounded cumulative returns at time ``j``.
  - ``\\mathrm{DaR_{r}}(\\bm{X},\\, \\alpha)`` the Drawdown at Risk of compounded cumulative returns.

See also: [`HCRiskMeasure`](@ref), [`HCRMSettings`](@ref), [`optimise!`](@ref), [`calc_risk(::DaR_r, ::AbstractVector)`](@ref), [`VaR`](@ref), [`CVaR`](@ref), [`EVaR`](@ref), [`RLVaR`](@ref) [`WR`](@ref), [`DaR`](@ref), [`CDaR`](@ref), [`CDaR_r`](@ref), [`EDaR`](@ref), [`EDaR_r`](@ref), [`RLDaR`](@ref), [`RLDaR_r`](@ref), [`MDD`](@ref), [`MDD_r`](@ref).

# Fields

  - `settings::HCRMSettings = HCRMSettings()`: configuration settings for the risk measure.
  - `alpha::T1 = 0.05`: significance level, `alpha ∈ (0, 1)`.

# Behaviour

## Validation

  - When setting `alpha` at construction or runtime, `alpha ∈ (0, 1)`.

# Examples

```@example
# Default settings
dar = DaR_r()

# Custom significance level
dar = DaR_r(; settings = HCRMSettings(; scale = 1.0), alpha = 0.01) # 1 % significance level
```
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
function (dar_r::DaR_r)(x::AbstractVector)
    T = length(x)
    alpha = dar_r.alpha
    x .= pushfirst!(x, 0) .+ 1
    cs = cumprod(x)
    peak = -Inf
    dd = similar(cs)
    for (idx, i) ∈ pairs(cs)
        if i > peak
            peak = i
        end
        dd[idx] = i / peak - 1
    end
    popfirst!(dd)
    return -partialsort!(dd, ceil(Int, alpha * T))
end

"""
    mutable struct MDD_r <: HCRiskMeasure

# Description

Maximum Drawdown (Calmar ratio) risk measure for compounded cumulative returns.

  - Measures the largest peak-to-trough decline.
  - ``\\mathrm{DaR_{r}}(\\bm{X},\\, \\alpha) \\leq \\mathrm{CDaR_{r}}(\\bm{X},\\, \\alpha) \\leq \\mathrm{EDaR_{r}}(\\bm{X},\\, \\alpha) \\leq \\mathrm{RLDaR_{r}}(\\bm{X},\\, \\alpha,\\, \\kappa) \\leq \\mathrm{MDD_{r}}(\\bm{X})``.

```math
\\begin{align}
\\mathrm{MDD_{r}}(\\bm{X}) &= \\max \\mathrm{DD_{r}}(\\bm{X})\\,.
\\end{align}
```

Where:

  - ``\\mathrm{DD_{r}}(\\bm{X})`` is the Drawdown of compounded cumulative returns as defined in [`DaR_r`](@ref).

See also: [`HCRiskMeasure`](@ref), [`HCRMSettings`](@ref), [`optimise!`](@ref), [`calc_risk(::MDD_r, ::AbstractVector)`](@ref), [`VaR`](@ref), [`CVaR`](@ref), [`EVaR`](@ref), [`RLVaR`](@ref) [`WR`](@ref), [`DaR`](@ref), [`DaR_r`](@ref), [`CDaR`](@ref), [`CDaR_r`](@ref), [`EDaR`](@ref), [`EDaR_r`](@ref), [`RLDaR`](@ref), [`RLDaR_r`](@ref), [`MDD`](@ref).

# Fields

  - `settings::HCRMSettings = HCRMSettings()`: configuration settings for the risk measure.

# Examples

```@example
# Default settings
mdd = MDD_r()

# Custom significance level
mdd = MDD_r(; settings = HCRMSettings(; scale = 1.0), alpha = 0.01) # 1 % significance level
```
"""
struct MDD_r <: HCRiskMeasure
    settings::HCRMSettings
end
function MDD_r(; settings::HCRMSettings = HCRMSettings())
    return MDD_r(settings)
end
function (mdd_r::MDD_r)(x::AbstractVector)
    x .= pushfirst!(x, 0) .+ 1
    cs = cumprod(x)
    val = 0.0
    peak = -Inf
    for i ∈ cs
        if i > peak
            peak = i
        end
        dd = 1 - i / peak
        if dd > val
            val = dd
        end
    end
    return val
end

"""
    mutable struct ADD_r <: HCRiskMeasure

# Description

Average Drawdown risk measure for uncompounded cumulative returns.

  - Measures the average of all peak-to-trough declines.
  - Provides a more balanced view than the maximum drawdown [`MDD_r`](@ref).

```math
\\begin{align}
\\mathrm{ADD_{r}}(\\bm{X}) &= \\dfrac{1}{T} \\sum\\limits_{j=0}^{T} \\mathrm{DD_{r}}(\\bm{X},\\, j)\\,.
\\end{align}
```

Where:

  - ``\\mathrm{DD_{r}}(\\bm{X},\\, j)`` is the Drawdown of compounded cumulative returns at time ``j`` as defined in [`DaR_r`](@ref).

See also: [`RiskMeasure`](@ref), [`RMSettings`](@ref), [`Portfolio`](@ref), [`optimise!`](@ref), [`calc_risk(::ADD_r, ::AbstractVector)`](@ref), [`ADD`](@ref), [`MDD_r`](@ref).

# Fields

  - `settings::HCRMSettings = HCRMSettings()`: configuration settings for the risk measure.

# Examples

```@example
# Default settings
add = ADD_r()

# Custom significance level
add = ADD_r(; settings = HCRMSettings(; scale = 1.0), alpha = 0.01) # 1 % significance level
```
"""
struct ADD_r <: HCRiskMeasure
    settings::HCRMSettings
    w::Union{<:AbstractWeights, Nothing}
end
function ADD_r(; settings::HCRMSettings = HCRMSettings(),
               w::Union{<:AbstractWeights, Nothing} = nothing)
    return ADD_r(settings, w)
end
function (add_r::ADD_r)(x::AbstractVector)
    T = length(x)
    w = add_r.w
    x .= pushfirst!(x, 0) .+ 1
    cs = cumprod(x)
    val = 0.0
    peak = -Inf
    if isnothing(w)
        for i ∈ cs
            if i > peak
                peak = i
            end
            dd = 1 - i / peak
            if dd > 0
                val += dd
            end
        end
        popfirst!(x)
        return val / T
    else
        @smart_assert(length(w) == T)
        for i ∈ cs
            if i > peak
                peak = i
            end
            dd = 1 - i / peak
            if dd > 0
                val += dd * w[i]
            end
        end
        popfirst!(x)
        return val / sum(w)
    end
end

"""
    mutable struct CDaR_r{T1 <: Real} <: HCRiskMeasure

# Description

Conditional Drawdown at Risk risk measure for compounded cumulative returns.

  - Measures the expected peak-to-trough loss in the worst `alpha %` of cases.
  - ``\\mathrm{DaR_{r}}(\\bm{X},\\, \\alpha) \\leq \\mathrm{CDaR_{r}}(\\bm{X},\\, \\alpha) \\leq \\mathrm{EDaR_{r}}(\\bm{X},\\, \\alpha) \\leq \\mathrm{RLDaR_{r}}(\\bm{X},\\, \\alpha,\\, \\kappa) \\leq \\mathrm{MDD_{r}}(\\bm{X})``.

```math
\\begin{align}
\\mathrm{CDaR_{r}}(\\bm{X},\\, \\alpha) &= \\mathrm{DaR_{r}}(\\bm{X},\\, \\alpha) + \\dfrac{1}{\\alpha T} \\sum\\limits_{j=0}^{T} \\max\\left(\\mathrm{DD_{r}}(\\bm{X},\\, j) - \\mathrm{DaR_{r}}(\\bm{X},\\, \\alpha),\\, 0 \\right)\\,.
\\end{align}
```

Where:

  - ``\\mathrm{DD_{r}}(\\bm{X},\\, j)`` is the Drawdown of compounded cumulative returns at time ``j`` as defined in [`DaR_r`](@ref).
  - ``\\mathrm{DaR_{r}}(\\bm{X},\\, \\alpha)`` the Drawdown at Risk of compounded cumulative returns as defined in [`DaR_r`](@ref).

See also: [`HCRiskMeasure`](@ref), [`HCRMSettings`](@ref), [`optimise!`](@ref), [`calc_risk(::CDaR_r, ::AbstractVector)`](@ref), [`VaR`](@ref), [`CVaR`](@ref), [`EVaR`](@ref), [`RLVaR`](@ref) [`WR`](@ref), [`DaR`](@ref), [`DaR_r`](@ref), [`CDaR`](@ref), [`EDaR`](@ref), [`EDaR_r`](@ref), [`RLDaR`](@ref), [`RLDaR_r`](@ref), [`MDD`](@ref), [`MDD_r`](@ref).

# Fields

  - `settings::HCRMSettings = HCRMSettings()`: configuration settings for the risk measure.
  - `alpha::T1 = 0.05`: significance level, `alpha ∈ (0, 1)`.

# Behaviour

## Validation

  - When setting `alpha` at construction or runtime, `alpha ∈ (0, 1)`.

# Examples

```@example
# Default settings
cdar_r = CDaR_r()

# Custom significance level
cdar_r = CDaR_r(; settings = HCRMSettings(; scale = 1.0), alpha = 0.01) # 1 % significance level
```
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
function (cdar_r::CDaR_r)(x::AbstractVector)
    T = length(x)
    alpha = cdar_r.alpha
    aT = alpha * T
    x .= pushfirst!(x, 0) .+ 1
    cs = cumprod(x)
    peak = -Inf
    dd = similar(cs)
    for (idx, i) ∈ pairs(cs)
        if i > peak
            peak = i
        end
        dd[idx] = i / peak - 1
    end
    popfirst!(dd)
    idx = ceil(Int, aT)
    var = -partialsort!(dd, idx)
    sum_var = 0.0
    for i ∈ 1:(idx - 1)
        sum_var += dd[i] + var
    end
    return var - sum_var / aT
end

"""
    mutable struct UCI_r{T1 <: Real} <: HCRiskMeasure

# Description

Ulcer Index risk measure for compounded cumulative returns.

  - Penalizes larger drawdowns more than smaller ones.

```math
\\begin{align}
\\mathrm{UCI_{r}}(\\bm{X}) &= \\left(\\dfrac{1}{T} \\sum\\limits_{j=0}^{T} \\mathrm{DD_{r}}(\\bm{X},\\, j)^{2}\\right)^{1/2}\\,.
\\end{align}
```

Where:

  - ``\\mathrm{DD_{r}}(\\bm{X},\\, j)`` is the Drawdown of compounded cumulative returns at time ``j`` as defined in [`DaR_r`](@ref).

See also: [`HCRiskMeasure`](@ref), [`HCRMSettings`](@ref), [`optimise!`](@ref), [`calc_risk(::UCI_r, ::AbstractVector)`](@ref), [`UCI`](@ref).

# Fields

  - `settings::HCRMSettings = HCRMSettings()`: configuration settings for the risk measure.

# Examples

```@example
# Default settings
uci_r = UCI_r()

# Custom settings
uci_r = UCI_r(; settings = HCRMSettings(; scale = 1.5))
```
"""
struct UCI_r <: HCRiskMeasure
    settings::HCRMSettings
end
function UCI_r(; settings::HCRMSettings = HCRMSettings())
    return UCI_r(settings)
end
function (uci_r::UCI_r)(x::AbstractVector)
    T = length(x)
    x .= pushfirst!(x, 0) .+ 1
    cs = cumprod(x)
    val = 0.0
    peak = -Inf
    for i ∈ cs
        if i > peak
            peak = i
        end
        dd = 1 - i / peak
        if dd > 0
            val += dd^2
        end
    end
    return sqrt(val / T)
end

"""
    mutable struct EDaR_r{T1 <: Real} <: HCRiskMeasure

# Description

Entropic Drawdown at Risk risk measure for compounded cumulative returns.

  - It is the upper bound of the Chernoff inequality for the [`DaR`](@ref) and [`CDaR`](@ref).
  - ``\\mathrm{DaR_{r}}(\\bm{X},\\, \\alpha) \\leq \\mathrm{CDaR_{r}}(\\bm{X},\\, \\alpha) \\leq \\mathrm{EDaR_{r}}(\\bm{X},\\, \\alpha) \\leq \\mathrm{RLDaR_{r}}(\\bm{X},\\, \\alpha,\\, \\kappa) \\leq \\mathrm{MDD_{r}}(\\bm{X})``.

```math
\\begin{align}
\\mathrm{RLDaR_{r}}(\\bm{X},\\, \\alpha,\\, \\kappa) &= \\mathrm{RRM}(\\mathrm{DD_{r}}(\\bm{X}),\\, \\alpha,\\, \\kappa)\\,.
\\end{align}
```

Where:

  - ``\\mathrm{RRM}(\\mathrm{DD_{r}}(\\bm{X}),\\, \\alpha,\\, \\kappa)`` is the Relativistic Risk Measure as defined in [`RRM`](@ref), using the Drawdown of compounded cumulative returns as defined in [`DaR_r`](@ref).

See also: [`HCRiskMeasure`](@ref), [`HCRMSettings`](@ref), [`Portfolio`](@ref), [`optimise!`](@ref), [`calc_risk(::EDaR_r, ::AbstractVector)`](@ref), [`VaR`](@ref), [`CVaR`](@ref), [`EVaR`](@ref), [`RLVaR`](@ref) [`WR`](@ref), [`DaR`](@ref), [`DaR_r`](@ref), [`CDaR`](@ref), [`CDaR_r`](@ref), [`EDaR_r`](@ref), [`RLDaR`](@ref), [`RLDaR_r`](@ref), [`MDD`](@ref), [`MDD_r`](@ref).

# Fields

  - `settings::HCRMSettings = HCRMSettings()`: configuration settings for the risk measure.
  - `alpha::T1 = 0.05`: significance level, `alpha ∈ (0, 1)`.
  - `solvers::Union{<:AbstractDict, Nothing}`: optional JuMP-compatible solvers for exponential cone problems.

# Behaviour

  - Requires solver capability for exponential cone problems.

  - When computing [`calc_risk(::EDaR_r, ::AbstractVector)`](@ref):

      + If `solvers` is `nothing`: uses `solvers` from [`Portfolio`](@ref)/.
      + If `solvers` is provided: use the solvers.

## Validation

  - When setting `alpha` at construction or runtime, `alpha ∈ (0, 1)`.

# Examples

```@example
# Default settings
edar_r = EDaR_r()

# Custom configuration with specific solver
edar_r = EDaR_r(; alpha = 0.025,  # 2.5 % significance level
                solvers = Dict("solver" => my_solver))
```
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
function (edar_r::EDaR_r)(x::AbstractVector)
    x .= pushfirst!(x, 0) .+ 1
    cs = cumprod(x)
    peak = -Inf
    dd = similar(cs)
    for (idx, i) ∈ pairs(cs)
        if i > peak
            peak = i
        end
        dd[idx] = i / peak - 1
    end
    popfirst!(dd)
    return ERM(dd, edar_r.solvers, edar_r.alpha)
end

"""
    mutable struct RLDaR_r{T1 <: Real} <: RiskMeasure

# Description

Relativistic Drawdown at Risk risk measure for compounded cumulative returns.

  - It is a generalisation of the [`EDaR`](@ref).
  - ``\\mathrm{DaR_{r}}(\\bm{X},\\, \\alpha) \\leq \\mathrm{CDaR_{r}}(\\bm{X},\\, \\alpha) \\leq \\mathrm{EDaR_{r}}(\\bm{X},\\, \\alpha) \\leq \\mathrm{RLDaR_{r}}(\\bm{X},\\, \\alpha,\\, \\kappa) \\leq \\mathrm{MDD_{r}}(\\bm{X})``.
  - ``\\lim\\limits_{\\kappa \\to 0} \\mathrm{RLDaR_{r}}(\\bm{X},\\, \\alpha,\\, \\kappa) \\approx \\mathrm{EDaR_{r}}(\\bm{X},\\, \\alpha)``
  - ``\\lim\\limits_{\\kappa \\to 1} \\mathrm{RLDaR_{r}}(\\bm{X},\\, \\alpha,\\, \\kappa) \\approx \\mathrm{MDD_{r}}(\\bm{X})``

```math
\\begin{align}
\\mathrm{RLDaR_{r}}(\\bm{X},\\, \\alpha,\\, \\kappa) &= \\mathrm{RRM}(\\mathrm{DD_{r}}(\\bm{X}),\\, \\alpha,\\, \\kappa)\\,.
\\end{align}
```

Where:

  - ``\\mathrm{RRM}(\\mathrm{DD_{r}}(\\bm{X}),\\, \\alpha,\\, \\kappa)`` is the Relativistic Risk Measure as defined in [`RRM`](@ref), using the Drawdown of compounded cumulative returns as defined in [`DaR_r`](@ref).

See also: [`RiskMeasure`](@ref), [`RMSettings`](@ref), [`Portfolio`](@ref), [`optimise!`](@ref), [`calc_risk(::RLDaR_r, ::AbstractVector)`](@ref), [`VaR`](@ref), [`CVaR`](@ref), [`EVaR`](@ref), [`RLVaR`](@ref) [`WR`](@ref), [`DaR`](@ref), [`DaR_r`](@ref), [`CDaR`](@ref), [`CDaR_r`](@ref), [`EDaR`](@ref), [`EDaR_r`](@ref), [`RLDaR_r`](@ref), [`MDD`](@ref), [`MDD_r`](@ref).

# Fields

  - `settings::RMSettings = RMSettings()`: configuration settings for the risk measure.
  - `alpha::T1 = 0.05`: significance level, `alpha ∈ (0, 1)`.
  - `kappa::T1 = 0.3`: significance level, `kappa ∈ (0, 1)`.
  - `solvers::Union{<:AbstractDict, Nothing}`: optional JuMP-compatible solvers for 3D power cone problems.

# Behaviour

  - Requires solver capability for 3D power cone problems.

  - When computing [`calc_risk(::RLDaR_r, ::AbstractVector)`](@ref):

      + If `solvers` is `nothing`: uses `solvers` from [`Portfolio`](@ref)/.
      + If `solvers` is provided: use the solvers.

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
function (rldar_r::RLDaR_r)(x::AbstractVector)
    x .= pushfirst!(x, 0) .+ 1
    cs = cumprod(x)
    peak = -Inf
    dd = similar(cs)
    for (idx, i) ∈ pairs(cs)
        if i > peak
            peak = i
        end
        dd[idx] = i / peak - 1
    end
    popfirst!(dd)
    return RRM(dd, rldar_r.solvers, rldar_r.alpha, rldar_r.kappa)
end

"""
    struct Equal <: HCRiskMeasure

# Description

Equal risk measure.

  - Risk is allocated evenly among a group of assets.

# Fields

  - `settings::RMSettings = RMSettings()`: configuration settings for the risk measure.

# Examples

```@example
# Default settings
equal = Equal()

# Custom configuration
equal = Equal(; settings = HCRMSettings(; scale = 3))
```
"""
mutable struct Equal <: HCRiskMeasure
    settings::HCRMSettings
end
function Equal(; settings::HCRMSettings = HCRMSettings())
    return Equal(settings)
end
function (equal::Equal)(w::AbstractVector, delta::Real = 0)
    return inv(length(w)) + delta
end

mutable struct TCM <: NoOptRiskMeasure
    settings::HCRMSettings
    w::Union{AbstractWeights, Nothing}
end
function TCM(; settings::HCRMSettings = HCRMSettings(;),
             w::Union{AbstractWeights, Nothing} = nothing)
    return TCM(settings, w)
end
function (tcm::TCM)(x::AbstractVector)
    T = length(x)
    w = tcm.w
    mu = isnothing(w) ? mean(x) : mean(x, w)
    val = x .- mu
    return sum(val .^ 3) / T
end
mutable struct TLPM{T1} <: HCRiskMeasure
    settings::HCRMSettings
    target::T1
    w::Union{AbstractWeights, Nothing}
end
function TLPM(; settings::HCRMSettings = HCRMSettings(;), target::Real = 0.0,
              w::Union{AbstractWeights, Nothing} = nothing)
    return TLPM{typeof(target)}(settings, target, w)
end
function (tlpm::TLPM)(x::AbstractVector)
    T = length(x)
    target = tlpm.target
    target = if !isinf(target)
        target
    else
        w = tlpm.w
        isnothing(w) ? mean(x) : mean(x, w)
    end
    val = x .- target
    return -sum(val[val .<= zero(target)] .^ 3) / T
end
mutable struct FTCM <: HCRiskMeasure
    settings::HCRMSettings
    w::Union{AbstractWeights, Nothing}
end
function FTCM(; settings::HCRMSettings = HCRMSettings(;),
              w::Union{AbstractWeights, Nothing} = nothing)
    return FTCM(settings, w)
end
function (ftcm::FTCM)(x::AbstractVector)
    T = length(x)
    w = ftcm.w
    mu = isnothing(w) ? mean(x) : mean(x, w)
    val = x .- mu
    return sum(val .^ 4) / T
end

mutable struct FTLPM{T1} <: HCRiskMeasure
    settings::HCRMSettings
    target::T1
    w::Union{AbstractWeights, Nothing}
end
function FTLPM(; settings::HCRMSettings = HCRMSettings(;), target::Real = 0.0,
               w::Union{AbstractWeights, Nothing} = nothing)
    return FTLPM{typeof(target)}(settings, target, w)
end
function (ftlpm::FTLPM)(x::AbstractVector)
    T = length(x)
    target = ftlpm.target
    target = if !isinf(target)
        target
    else
        w = ftlpm.w
        isnothing(w) ? mean(x) : mean(x, w)
    end
    val = x .- target
    return sum(val[val .<= zero(target)] .^ 4) / T
end

mutable struct Skewness <: NoOptRiskMeasure
    settings::HCRMSettings
    mean_w::Union{AbstractWeights, Nothing}
    ve::CovarianceEstimator
    std_w::Union{AbstractWeights, Nothing}
end
function Skewness(; settings::HCRMSettings = HCRMSettings(),
                  mean_w::Union{AbstractWeights, Nothing} = nothing,
                  ve::CovarianceEstimator = SimpleVariance(),
                  std_w::Union{AbstractWeights, Nothing} = nothing)
    return Skewness(settings, mean_w, ve, std_w)
end
function (skewness::Skewness)(x::AbstractVector)
    T = length(x)
    mean_w = skewness.mean_w
    ve = skewness.ve
    std_w = skewness.std_w
    mu = isnothing(mean_w) ? mean(x) : mean(x, mean_w)
    sigma = isnothing(std_w) ? std(x) : std(ve, x, std_w)
    val = x .- mu
    return sum(dot(val, val) * val) / T / sigma^3
end

mutable struct SSkewness{T1} <: HCRiskMeasure
    settings::HCRMSettings
    target::T1
    mean_w::Union{AbstractWeights, Nothing}
    ve::CovarianceEstimator
    std_w::Union{AbstractWeights, Nothing}
end
function SSkewness(; settings::HCRMSettings = HCRMSettings(), target::Real = 0.0,
                   mean_w::Union{AbstractWeights, Nothing} = nothing,
                   ve::CovarianceEstimator = SimpleVariance(),
                   std_w::Union{AbstractWeights, Nothing} = nothing)
    return SSkewness{typeof(target)}(settings, target, mean_w, ve, std_w)
end
function (sskewness::SSkewness)(x::AbstractVector)
    T = length(x)
    target = sskewness.target
    mean_w = sskewness.mean_w
    ve = sskewness.ve
    std_w = sskewness.std_w
    mu = isnothing(mean_w) ? mean(x) : mean(x, mean_w)
    val = x .- mu
    val .= val[val .<= target]
    sigma = isnothing(std_w) ? std(val) : std(ve, val, std_w; mean = zero(target))
    return sum(dot(val, val) * val) / T / sigma^3
end

mutable struct Kurtosis <: HCRiskMeasure
    settings::HCRMSettings
    mean_w::Union{AbstractWeights, Nothing}
    ve::CovarianceEstimator
    std_w::Union{AbstractWeights, Nothing}
end
function Kurtosis(; settings::HCRMSettings = HCRMSettings(),
                  mean_w::Union{AbstractWeights, Nothing} = nothing,
                  ve::CovarianceEstimator = SimpleVariance(),
                  std_w::Union{AbstractWeights, Nothing} = nothing)
    return Kurtosis(settings, mean_w, ve, std_w)
end
function (kurtosis::Kurtosis)(x::AbstractVector)
    T = length(x)
    mean_w = kurtosis.mean_w
    ve = kurtosis.ve
    std_w = kurtosis.std_w
    mu = isnothing(mean_w) ? mean(x) : mean(x, mean_w)
    sigma = isnothing(std_w) ? std(x) : std(ve, x, std_w)
    val = x .- mu
    return dot(val, val)^2 / T / sigma^4
end
mutable struct SKurtosis{T1} <: HCRiskMeasure
    settings::HCRMSettings
    target::T1
    mean_w::Union{AbstractWeights, Nothing}
    ve::CovarianceEstimator
    std_w::Union{AbstractWeights, Nothing}
end
function SKurtosis(; settings::HCRMSettings = HCRMSettings(), target::Real = 0.0,
                   mean_w::Union{AbstractWeights, Nothing} = nothing,
                   ve::CovarianceEstimator = SimpleVariance(),
                   std_w::Union{AbstractWeights, Nothing} = nothing)
    return SKurtosis{typeof(target)}(settings, target, mean_w, ve, std_w)
end
function (skurtosis::SKurtosis)(x::AbstractVector)
    T = length(x)
    target = skurtosis.target
    mean_w = skurtosis.mean_w
    ve = skurtosis.ve
    std_w = skurtosis.std_w
    mu = isnothing(mean_w) ? mean(x) : mean(x, mean_w)
    val = x .- mu
    val .= val[val .<= target]
    sigma = isnothing(std_w) ? std(val) : std(ve, val, std_w; mean = zero(target))
    return dot(val, val)^2 / T / sigma^4
end

const RMSolvers = Union{EVaR, EDaR, EDaR_r, RLVaR, RLDaR, RLDaR_r}
const RMSigma = Union{SD, Variance, WCVariance}
const RMSkew = Union{Skew, SSkew}
const RMOWA = Union{GMD, TG, TGRG, OWA}

export RiskMeasure, HCRiskMeasure, NoOptRiskMeasure, RMSettings, HCRMSettings,
       VarianceFormulation, Quad, SOC, SD, MAD, SSD, FLPM, SLPM, WR, CVaR, EVaR, RLVaR, MDD,
       ADD, CDaR, UCI, EDaR, RLDaR, Kurt, SKurt, RG, CVaRRG, OWASettings, GMD, TG, TGRG,
       OWA, BDVariance, Skew, SSkew, Variance, SVariance, VaR, DaR, DaR_r, MDD_r, ADD_r,
       CDaR_r, UCI_r, EDaR_r, RLDaR_r, Equal, BDVAbsVal, BDVIneq, WCVariance, DRCVaR, Box,
       Ellipse, NoWC, TrackingRM, TurnoverRM, NoTracking, TrackWeight, TrackRet, NoTR, TR
