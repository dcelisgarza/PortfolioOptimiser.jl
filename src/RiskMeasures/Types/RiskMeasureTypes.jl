# Copywrite (c) 2025
# Author: Daniel Celis Garza <daniel.celis.garza@gmail.com>
# SPDX-License-Identifier: MIT

"""
    mutable struct PortOptSolver

Structure for defining solver and solver parameters for solving [`JuMP`](https://github.com/jump-dev/JuMP.jl) models.

  - Functions and types which take a `PortOptSolver` also take a `<:AbstractVector{PortOptSolver}`.

      + If a vector is provided, it will iterated over until a solution is accepted or the vector runs out, at which point the optimisation will be considered to have failed.

See also: [`Portfolio`](@ref), [`RMSolvers`](@ref), [`OWAJTypes`](@ref).

# Keyword Arguments

  - `name::Union{Symbol, <:AbstractString} = ""`: optimiser name (for registering solutions).
  - `solver::Any = nothing`: defines the solver to be used. One can also use [`JuMP.optimizer_with_attributes`](https://jump.dev/JuMP.jl/stable/api/JuMP/#optimizer_with_attributes) to direcly provide a solver with attributes already attached.`JuMP` optimiser factory [`optimizer_factory`](https://jump.dev/JuMP.jl/stable/api/JuMP/#Model).
  - `check_sol::NamedTuple = (; allow_local = true, allow_almost = true)`: defines the keyword arguments passed on to [`JuMP.is_solved_and_feasible`](https://jump.dev/JuMP.jl/stable/api/JuMP/#is_solved_and_feasible) for accepting/rejecting solutions.
  - `solvers::Union{Nothing, <:AbstractDict} = nothing`:  defines solver-specific parameters/attributes.
  - `add_bridges::Bool = true`: value of the `add_bridges` kwarg of [`JuMP.set_optimizer`](https://jump.dev/JuMP.jl/stable/api/JuMP/#JuMP.set_optimizer).

# Examples

It's possible to define a single solver.

```jldoctest solvers
using JuMP, Clarabel, PortfolioOptimiser
solvers = PortOptSolver(; name = :Clarabel, solver = Clarabel.Optimizer,
                        check_sol = (allow_local = true, allow_almost = true),
                        params = Dict("verbose" => false), add_bridges = false)

# output

PortOptSolver(:Clarabel, Clarabel.MOIwrapper.Optimizer, (allow_local = true, allow_almost = true), Dict{String, Bool}("verbose" => 0), false)
```

Or a collection of solvers.

```jldoctest solvers
solvers = [PortOptSolver(; name = :Clarabel_1, solver = Clarabel.Optimizer,
                         check_sol = (allow_local = true, allow_almost = true),
                         params = Dict("verbose" => false, "max_step_fraction" => 0.8)),
           # Provide solver with pre-attached attributes and no arguments 
           # for the `JuMP.is_solved_and_feasible` function.
           PortOptSolver(; name = :Clarabel_2,
                         solver = JuMP.optimizer_with_attributes(Clarabel.Optimizer,
                                                                 "max_step_fraction" => 0.75),
                         # Do not add JuMP bridges to JuMP.Model()
                         add_bridges = false)]

# output

2-element Vector{PortOptSolver}:
 PortOptSolver(:Clarabel_1, Clarabel.MOIwrapper.Optimizer, (allow_local = true, allow_almost = true), Dict{String, Real}("verbose" => false, "max_step_fraction" => 0.8), true)
 PortOptSolver(:Clarabel_2, MathOptInterface.OptimizerWithAttributes(Clarabel.MOIwrapper.Optimizer, Pair{MathOptInterface.AbstractOptimizerAttribute, Any}[MathOptInterface.RawOptimizerAttribute("max_step_fraction") => 0.75]), (allow_local = true, allow_almost = true), nothing, false)
```
"""
mutable struct PortOptSolver
    name::Union{Symbol, <:AbstractString}
    solver::Any
    check_sol::NamedTuple
    params::Union{Nothing, <:AbstractDict}
    add_bridges::Bool
end
function PortOptSolver(; name::Union{Symbol, <:AbstractString} = "", solver::Any = nothing,
                       check_sol::NamedTuple = (; allow_local = true, allow_almost = true),
                       params::Union{Nothing, <:AbstractDict} = nothing,
                       add_bridges::Bool = true)
    return PortOptSolver(name, solver, check_sol, params, add_bridges)
end
function Base.isequal(A::PortOptSolver, B::PortOptSolver)
    for property ∈ propertynames(A)
        prop_A = getproperty(A, property)
        prop_B = getproperty(B, property)
        if !isequal(prop_A, prop_B)
            return false
        end
    end
    return true
end

"""
    abstract type AbstractRMSettings end

Abstract type for subtyping risk measure settings.

See also: [`RMSettings`](@ref), [`HCRMSettings`](@ref).
"""
abstract type AbstractRMSettings end

"""
    mutable struct RMSettings{T1 <: Real, T2 <: Real} <: AbstractRMSettings

Configuration settings for concrete subtypes of [`RiskMeasure`](@ref). Having this property makes it possible for a risk measure to be used in any optimisation types that take risk measures as parameters.

See also: [`RiskMeasure`](@ref), [`AbstractScalarisation`](@ref), [`calc_risk`](@ref).

# Keyword Arguments

## In optimisations which take risk measures and use [`JuMP`](https://github.com/jump-dev/JuMP.jl) models

  - `flag::Bool = true`:

      + If `true`: it is included in the optimisation's risk vector.
      + If `false`: it is *not* included in the optimisation's risk vector, used when you want to constrain the upper bound of a risk measure without having that risk measure appear in the [`MinRisk`](@ref), [`Utility`](@ref), or [`Sharpe`](@ref) objective measures.

  - `scale::T1 = 1.0`: weight parameter of the risk measure in the [`AbstractScalarisation`](@ref) method being used.
  - `ub::T2 = Inf`: upper bound risk constraint.

## In optimisations which take risk measures and do not use [`JuMP`](https://github.com/jump-dev/JuMP.jl) models

  - `flag::Bool = true`: no effect, the risk cannot be bounded in these optimisations.
  - `scale::T1 = 1.0`: weight parameter of the risk measure in the [`AbstractScalarisation`](@ref) method being used.
  - `ub::T2 = Inf`: no effect, the risk cannot be bounded in these optimisations.

# Examples
"""
mutable struct RMSettings{T1 <: Real, T2 <: Real} <: AbstractRMSettings
    flag::Bool
    scale::T1
    ub::T2
end
function RMSettings(; flag::Bool = true, scale::Real = 1.0, ub::Real = Inf)
    return RMSettings{typeof(scale), typeof(ub)}(flag, scale, ub)
end

"""
    mutable struct HCRMSettings{T1 <: Real} <: AbstractRMSettings

Configuration settings for concrete subtypes of [`HCRiskMeasure`](@ref).

See also: [`HCRiskMeasure`](@ref), [`AbstractScalarisation`](@ref), [`calc_risk`](@ref).

# Keyword Arguments

  - `scale::T1 = 1.0`: weight parameter of the risk measure in the [`AbstractScalarisation`](@ref) method being used.

# Examples
"""
mutable struct HCRMSettings{T1 <: Real} <: AbstractRMSettings
    scale::T1
end
function HCRMSettings(; scale::Real = 1.0)
    return HCRMSettings{typeof(scale)}(scale)
end

"""
    abstract type OWAFormulation end

Abstract type for subtyping Ordered Weight Array formulations in optimisations which use [`JuMP`](https://github.com/jump-dev/JuMP.jl) models.

See also: [`OWAExact`](@ref), [`OWAApprox`](@ref), [`RiskMeasureOWA`](@ref).
"""
abstract type OWAFormulation end

"""
    struct OWAExact <: OWAFormulation end

Type for dispatching the exact formulation of Ordered Weight Array risk measures.

See also: [`OWAApprox`](@ref), [`RiskMeasureOWA`](@ref).
"""
struct OWAExact <: OWAFormulation end

"""
    mutable struct OWAApprox{T1 <: AbstractVector{<:Real}} <: OWAFormulation

Type for dispatching and tuning the approximate formulation of Ordered Weight Array risk measures.

See also: [`OWAExact`](@ref), [`RiskMeasureOWA`](@ref).

# Keyword Arguments

  - `p::T1 = Float64[2, 3, 4, 10, 50]`: vector of the p-norm orders to be used in the approximation.

# Behaviour

  - Uses [`PowerCone`](https://jump.dev/JuMP.jl/stable/tutorials/conic/tips_and_tricks/#PowerCone) constraints.

# Examples
"""
mutable struct OWAApprox{T1 <: AbstractVector{<:Real}} <: OWAFormulation
    p::T1
end
function OWAApprox(; p::AbstractVector{<:Real} = Float64[2, 3, 4, 10, 50])
    return OWAApprox{typeof(p)}(p)
end

"""
    abstract type AbstractRiskMeasure end

Supertype for all risk measaures.

See also: [`RiskMeasure`](@ref), [`HCRiskMeasure`](@ref), [`NoOptRiskMeasure`](@ref).
"""
abstract type AbstractRiskMeasure end

"""
    abstract type RiskMeasure <: AbstractRiskMeasure end

Supertype for risk measures compatible with optimisations which accept risk measures.

See also: [`RiskMeasureSolvers`](@ref), [`RiskMeasureSigma`](@ref), [`RiskMeasureSkew`](@ref), [`RiskMeasureOWA`](@ref), [`RiskMeasureMu`](@ref), [`RiskMeasureTarget`](@ref), [`RMSettings`](@ref), [`calc_risk`](@ref), [`set_rm`](@ref).

# Implementation

To ensure a risk measure can be used any of the above optimisation types, it must abide by a few rules.

  - Implement: `Base.iterate`, `Base.Symbol`, `Base.length`, `Base.getindex`, and `Base.view`.

```julia
struct MyRiskMeasure <: RiskMeasure
    # Properties of MyRiskMeasure
end

Base.iterate(S::MyRiskMeasure, state = 1) = state > 1 ? nothing : (S, state + 1)
function Base.String(s::MyRiskMeasure)
    return "MyRiskMeasure"
end
function Base.Symbol(::MyRiskMeasure)
    return Symbol("MyRiskMeasure")
end
function Base.length(::MyRiskMeasure)
    return 1
end
function Base.getindex(S::MyRiskMeasure, ::Any)
    return S
end
function Base.view(S::MyRiskMeasure, ::Any)
    return S
end
```

  - Include a `settings::RMSettings` property, [`RMSettings`](@ref).

```julia
struct MyRiskMeasure <: RiskMeasure
    # Properties of MyRiskMeasure
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
    # The upper bound key will be `Symbol("MyRiskMeasure_risk_ub")`
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
        # The upper bound key will be `Symbol("MyRiskMeasure_risk_\$(i)_ub")`
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

  - If a risk measure is to be compatible with hierarchical optimisations that take risk measures as parameters, and it contains a properties which can/must be indexed/computed per asset, like a vector or matrix, it must implement [`set_custom_hc_rm!`](@ref) and [`unset_custom_hc_rm!`](@ref) which dispatches on the custom risk measure.

```julia
struct MyRiskMeasure{T1, T2, T3} <: RiskMeasure
    # Properties containing asset information (computable or indexable).
    indexable_vector::Vector{T1}
    indexable_matrix::Matrix{T1}
    computable_vector::Vector{T1}
    computable_matrix::Matrix{T1}
    computable_vector_args::T2
    computable_matrix_args::T3
end

# We have some computable properties, so we need to define the function to do so.
function compute_MyRiskMeasure_vec_mtx!(rm::MyRiskMeasure, args...)
    # Compute vector and matrix
    new_computable_vector = ...
    new_computable_matrix = ...

    rm.computable_vector = new_computable_vector
    rm.computable_matrix = new_computable_matrix

    return nothing
end

# port is the portfolio, sigma is the covariance matrix, cluster are the indices defining the cluster.
function set_custom_hc_rm!(rm::MyRiskMeasure, port, sigma, cluster)
    old_i_vector = rm.indexable_vector
    old_i_matrix = rm.indexable_matrix
    old_c_vector = rm.computable_vector
    old_c_matrix = rm.computable_matrix

    ###
    ###
    # These can be placed inside if statements that condition the indexing
    rm.indexable_vector = rm.indexable_vector[cluster]
    rm.indexable_matrix = rm.indexable_matrix[cluster, cluster]
    ###
    ###

    compute_MyRiskMeasure_vec_mtx!(rm, port, sigma, cluster)

    return Tuple(old_i_vector, old_i_matrix, old_c_vector, old_c_matrix)
end
function unset_custom_hc_rm!(rm::MyRiskMeasure, old_custom)
    rm.indexable_vector = old_custom[1]
    rm.indexable_matrix = old_custom[2]
    rm.computable_vector = old_custom[3]
    rm.computable_matrix = old_custom[4]

    return nothing
end
```

  - Similarly, if the risk measure is to be used in [`NCO`](@ref) optimisations, and it contains a properties which can/must be indexed/computed per asset, like a vector or matrix, it must implement [`pre_modify_intra_port!`](@ref), [`post_modify_intra_port!`](@ref), [`reset_intra_port!`](@ref), [`pre_modify_inter_port!`](@ref), [`post_modify_inter_port!`](@ref), [`reset_inter_port!`](@ref), which dispatch on custom structures that. The functions can then check for the custom risk measure and modify it as in the previous bullet point. See the function's docstrings for explanations on their arguments and use.

```julia
# Structures for dispatching on.
struct MyPreModify <: AbstractNCOModify
    # Custom properties.
end
struct MyPostModify <: AbstractNCOModify
    # Custom properties.
end

# Procedures for computing or modifying risk measures for the internal optimisations.
# Each cluster is treated as a single portfolio.
function pre_modify_intra_port!(pre_modify::MyPreModify, intra_port, internal_args, i,
                                cluster, cidx, idx_sq, Nc, special_rm_idx)
    # Modify intra-cluster portfolio pre computation of statistics.
    return pre_mod_output
end
function post_modify_intra_port!(post_modify::MyPostModify, intra_port, internal_args, i,
                                 cluster, cidx, idx_sq, Nc, special_rm_idx)
    # Modify intra-cluster portfolio post computation of statistics.
    return post_mod_output
end
function reset_intra_port!(pre_modify::MyPreModify, pre_mod_output,
                           post_modify::MyPostModify, post_mod_output, intra_port,
                           internal_args, i, cluster, cidx, idx_sq, Nc, special_rm_idx)
    # Reset any changes done to the optimisation arguments.
    return nothing
end

# Procedures for computing or modifying risk measures for the external optimisation.
# Each cluster is turned into a synthetic asset and a portfolio optimisation is
# performed on them.
function pre_modify_inter_port!(pre_modify::MyPreModify, inter_port, wi, external_args,
                                special_rm_idx)
    # Modify inter-cluster portfolio pre computation of statistics.
    return pre_mod_output
end
function post_modify_inter_port!(post_modify::MyPostModify, inter_port, wi, external_args,
                                 special_rm_idx)
    # Modify inter-cluster portfolio post computation of statistics.
    return post_mod_output
end
function reset_inter_port!(pre_modify::MyPreModify, pre_mod_output,
                           post_modify::MyPostModify, post_mod_output, inter_port, wi,
                           external_args, special_rm_idx)
    # Reset any changes done to the optimisation arguments.
    return nothing
end

# The NCO optimisation type would then be defined like so.
type = NCO(;
           internal = NCOArgs(; type = Trad(; rm = MyRiskMeasure()),
                              pre_modify = MyPreModify(), post_modify = MyPostModify()),
           # In case the external optimisation is to use something different.
           external = ...)
```
"""
abstract type RiskMeasure <: AbstractRiskMeasure end

"""
    abstract type HCRiskMeasure <: AbstractRiskMeasure end

Supertype for risk measures compatible with optimisations which accept risk measures and do not use [`JuMP`](https://github.com/jump-dev/JuMP.jl) models.

See also: [`HCRiskMeasureSolvers`](@ref), [`HCRiskMeasureMu`](@ref), [`HCRiskMeasureTarget`](@ref), [`HCRMSettings`](@ref), [`calc_risk`](@ref).

# Implementation

To ensure a risk measure can be used by the above optimisation types, it must abide by a few rules.

  - Implement: `Base.iterate`, `Base.Symbol`, `Base.length`, `Base.getindex`, and `Base.view`.

```julia
struct MyHCRiskMeasure <: HCRiskMeasure
    # Properties of MyHCRiskMeasure
end

Base.iterate(S::MyHCRiskMeasure, state = 1) = state > 1 ? nothing : (S, state + 1)
function Base.String(s::MyHCRiskMeasure)
    return "MyHCRiskMeasure"
end
function Base.Symbol(::MyHCRiskMeasure)
    return Symbol("MyHCRiskMeasure")
end
function Base.length(::MyHCRiskMeasure)
    return 1
end
function Base.getindex(S::MyHCRiskMeasure, ::Any)
    return S
end
function Base.view(S::MyHCRiskMeasure, ::Any)
    return S
end
```

  - Include a `settings::HCRMSettings` property, [`HCRMSettings`](@ref).

```julia
struct MyHCRiskMeasure <: HCRiskMeasure
    # Properties of MyHCRiskMeasure
    settings::HCRMSettings
end
```

  - Implement your measure's risk calculation method, [`calc_risk`](@ref). This will let the library use the risk function everywhere it needs to.

```julia
function calc_risk(my_risk::MyHCRiskMeasure, w::AbstractVector; kwargs...)
    # Risk measure calculation
end
```

  - If a risk measure is to be compatible with hierarchical optimisations that take risk measures as parameters, and it contains a properties which can/must be indexed/computed per asset, like a vector or matrix, it must implement [`set_custom_hc_rm!`](@ref) and [`unset_custom_hc_rm!`](@ref) which dispatches on the custom risk measure.

```julia
struct MyHCRiskMeasure{T1, T2, T3} <: HCRiskMeasure
    # Properties containing asset information (computable or indexable).
    indexable_vector::Vector{T1}
    indexable_matrix::Matrix{T1}
    computable_vector::Vector{T1}
    computable_matrix::Matrix{T1}
    computable_vector_args::T2
    computable_matrix_args::T3
end

# We have some computable properties, so we need to define the function to do so.
function compute_MyRiskMeasure_vec_mtx!(rm::MyHCRiskMeasure, args...)
    # Compute vector and matrix
    new_computable_vector = ...
    new_computable_matrix = ...

    rm.computable_vector = new_computable_vector
    rm.computable_matrix = new_computable_matrix

    return nothing
end

# port is the portfolio, sigma is the covariance matrix, cluster are the indices defining the cluster.
function set_custom_hc_rm!(rm::MyHCRiskMeasure, port, sigma, cluster)
    old_i_vector = rm.indexable_vector
    old_i_matrix = rm.indexable_matrix
    old_c_vector = rm.computable_vector
    old_c_matrix = rm.computable_matrix

    ###
    ###
    # These can be placed inside if statements that condition the indexing
    rm.indexable_vector = rm.indexable_vector[cluster]
    rm.indexable_matrix = rm.indexable_matrix[cluster, cluster]
    ###
    ###

    compute_MyRiskMeasure_vec_mtx!(rm, port, sigma, cluster)

    return Tuple(old_i_vector, old_i_matrix, old_c_vector, old_c_matrix)
end
function unset_custom_hc_rm!(rm::MyHCRiskMeasure, old_custom)
    rm.indexable_vector = old_custom[1]
    rm.indexable_matrix = old_custom[2]
    rm.computable_vector = old_custom[3]
    rm.computable_matrix = old_custom[4]

    return nothing
end
```
"""
abstract type HCRiskMeasure <: AbstractRiskMeasure end

"""
    abstract type NoOptRiskMeasure <: AbstractRiskMeasure end

Abstract type for risk measures that cannot be used in optimisations but can be used as performance measurements via [`calc_risk`](@ref). This can be for two reasons:

 1. They can be negative, therefore unsuitable for optimisations that accept risk measures and do not use [`JuMP`](https://github.com/jump-dev/JuMP.jl) models.
 2. They have no known optimisation formulation, therefore unsuitable for optimisations that accept risk measures and use [`JuMP`](https://github.com/jump-dev/JuMP.jl) models.

See also: [`calc_risk`](@ref).

# Implementation

  - Implement: [`calc_risk`](@ref) for the type. This will let the library use the risk function everywhere it needs to.

```julia
struct MyNoOptRiskMeasure <: NoOptRiskMeasure
    # Properties of MyNoOptRiskMeasure
end

function calc_risk(my_risk::MyNoOptRiskMeasure, w::AbstractVector; kwargs...)
    # Risk measure calculation
end
```
"""
abstract type NoOptRiskMeasure <: AbstractRiskMeasure end

"""
    abstract type RiskMeasureSolvers <: RiskMeasure end

Abstract type for subtyping [`RiskMeasure`](@ref) for which computing the risk requires solving an optimisation model.

See also: [`RiskMeasure`](@ref), [`EVaR`](@ref), [`EDaR`](@ref), [`RLVaR`](@ref), [`RLDaR`](@ref).

# Implementation

  - Concrete subtypes must contain the following properties:

      + `solvers::Union{Nothing, PortOptSolver, <:AbstractVector{PortOptSolver}}`: (optional) instance or `AbstractVector` of [`PortOptSolver`](@ref).

  - Ideally perform any necessary validation checks at instantiation and `setproperty!`.
"""
abstract type RiskMeasureSolvers <: RiskMeasure end

"""
    abstract type HCRiskMeasureSolvers <: HCRiskMeasure end

Abstract type for subtyping [`HCRiskMeasure`](@ref) for which computing the risk requires solving an optimisation model.

See also: [`HCRiskMeasure`](@ref), [`EDaR_r`](@ref), [`RLDaR_r`](@ref).

# Implementation

  - Concrete subtypes must contain the following properties:

      + `solvers::Union{Nothing, PortOptSolver, <:AbstractVector{PortOptSolver}}`: (optional) instance or `AbstractVector` of [`PortOptSolver`](@ref).

  - Ideally perform any necessary validation checks at instantiation and `setproperty!`.
"""
abstract type HCRiskMeasureSolvers <: HCRiskMeasure end

"""
    abstract type RiskMeasureSigma <: RiskMeasure end

Abstract type for subtyping [`RiskMeasure`](@ref) which can use an `N×N` covariance matrix.

See also: [`RiskMeasure`](@ref), [`Variance`](@ref), [`SD`](@ref), [`WCVariance`](@ref).

# Implementation

  - Concrete subtypes must contain the following properties:

      + `sigma::Union{<:AbstractMatrix, Nothing}`: property to store an `N×N` covariance matrix.

  - Ideally perform any necessary validation checks at instantiation and `setproperty!`.
"""
abstract type RiskMeasureSigma <: RiskMeasure end

abstract type RiskMeasureRCSigma <: RiskMeasureSigma end

"""
    abstract type RiskMeasureSkew <: RiskMeasure end

Abstract type for subtyping [`RiskMeasure`](@ref) which can use an `N×N²` coskew matrix and `N×N` matrix that stores the sum of the symmetric negative spectral slices of the coskewness.

See also: [`RiskMeasure`](@ref), [`NQSkew`](@ref), [`NQSSkew`](@ref).

# Implementation

  - Concrete subtypes must contain the following properties:

      + `skew::Union{<:AbstractMatrix, Nothing}`: property to store an `N×N²` coskew matrix.
      + `V::Union{<:AbstractMatrix, Nothing}`: property to store an `N×N` matrix that stores the sum of the symmetric negative spectral slices of the coskewness.

  - Ideally perform any necessary validation checks at instantiation and `setproperty!`.
"""
abstract type RiskMeasureSkew <: RiskMeasure end

"""
    abstract type RiskMeasureOWA <: RiskMeasure end

Abstract type for subtyping [`RiskMeasure`](@ref) which is implemented via an Ordered Weight Array formulation.

See also: [`RiskMeasure`](@ref), [`GMD`](@ref), [`TG`](@ref), [`TGRG`](@ref), [`OWA`](@ref).

# Implementation

  - Concrete subtypes must contain the following properties:

      + `formulation::OWAFormulation`: property to store the formulation dispatch type.

  - Ideally perform any necessary validation checks at instantiation and `setproperty!`.
"""
abstract type RiskMeasureOWA <: RiskMeasure end

"""
    abstract type RiskMeasureMu <: RiskMeasure end

Abstract type for subtyping [`RiskMeasure`](@ref) which can use a `T×1` [`AbstractWeights`](https://juliastats.org/StatsBase.jl/stable/weights/) vector for computing the weighted mean, and an `N×1` expected returns vector.

See also: [`RiskMeasure`](@ref), [`RiskMeasureTarget`](@ref), [`MAD`](@ref), [`SSD`](@ref), [`SVariance`](@ref), [`Kurt`](@ref), [`SKurt`](@ref).

# Implementation

  - Concrete subtypes must contain the following properties:

      + `mu::Union{<:AbstractVector, Nothing}`: property to store an `N×1` expected returns vector.
      + `w::Union{<:AbstractWeights, Nothing}`: property to store a `T×1` [`AbstractWeights`](https://juliastats.org/StatsBase.jl/stable/weights/) vector for computing the weighted mean.

  - Ideally perform any necessary validation checks at instantiation and `setproperty!`.
"""
abstract type RiskMeasureMu <: RiskMeasure end

"""
    abstract type HCRiskMeasureMu <: RiskMeasure end

Abstract type for subtyping [`HCRiskMeasure`](@ref) which can use a `T×1` [`AbstractWeights`](https://juliastats.org/StatsBase.jl/stable/weights/) vector for computing the weighted mean, and an `N×1` expected returns vector.

See also: [`HCRiskMeasure`](@ref), [`FTCM`](@ref).

# Implementation

  - Concrete subtypes must contain the following properties:

      + `mu::Union{<:AbstractVector, Nothing}`: property to store an `N×1` expected returns vector.
      + `w::Union{<:AbstractWeights, Nothing}`: property to store a `T×1` [`AbstractWeights`](https://juliastats.org/StatsBase.jl/stable/weights/) vector for computing the weighted mean.

  - Ideally perform any necessary validation checks at instantiation and `setproperty!`.
"""
abstract type HCRiskMeasureMu <: HCRiskMeasure end

"""
    abstract type NoOptRiskMeasureMu <: RiskMeasure end

Abstract type for subtyping [`NoOptRiskMeasure`](@ref) which can use a `T×1` [`AbstractWeights`](https://juliastats.org/StatsBase.jl/stable/weights/) vector for computing the weighted mean, and an `N×1` expected returns vector.

See also: [`NoOptRiskMeasure`](@ref), [`TCM`](@ref).

# Implementation

  - Concrete subtypes must contain the following properties:

      + `w::Union{<:AbstractWeights, Nothing}`: property to store a `T×1` [`AbstractWeights`](https://juliastats.org/StatsBase.jl/stable/weights/) vector for computing the weighted mean.
      + `mu::Union{<:AbstractVector, Nothing}`: property to store an `N×1` expected returns vector.

  - Ideally perform any necessary validation checks at instantiation and `setproperty!`.
"""
abstract type NoOptRiskMeasureMu <: NoOptRiskMeasure end

"""
    abstract type RiskMeasureTarget <: RiskMeasureMu end

Abstract type for subtyping [`RiskMeasure`](@ref) which can use a scalar or an `N×1` vector specifying the minimum return threshold for classifying downside returns, a `T×1` [`AbstractWeights`](https://juliastats.org/StatsBase.jl/stable/weights/) vector for computing the weighted mean, and an `N×1` expected returns vector.

See also: [`RiskMeasure`](@ref), [`FLPM`](@ref), [`SSD`](@ref), [`SVariance`](@ref).

# Implementation

  - Concrete subtypes must contain the following properties:

      + `target::Union{<:Real, <:AbstractVector{<:Real}, Nothing}`: scalar or `N×1` minimum return threshold for classifying downside returns. Only returns equal to or below this value are considered in the calculation. Must be in the same frequency as the returns.
      + `w::Union{<:AbstractWeights, Nothing}`: property to store a `T×1` [`AbstractWeights`](https://juliastats.org/StatsBase.jl/stable/weights/) vector for computing the weighted mean.
      + `mu::Union{<:AbstractVector, Nothing}`: property to store an `N×1` expected returns vector.

  - Ideally perform any necessary validation checks at instantiation and `setproperty!`.
"""
abstract type RiskMeasureTarget <: RiskMeasureMu end

"""
    abstract type HCRiskMeasureTarget <: RiskMeasureMu end

Abstract type for subtyping [`HCRiskMeasure`](@ref) which can use a scalar or `N×1` vector specifying the minimum return threshold for classifying downside returns, a `T×1` [`AbstractWeights`](https://juliastats.org/StatsBase.jl/stable/weights/) vector for computing the weighted mean, and an `N×1` expected returns vector.

See also: [`HCRiskMeasure`](@ref), [`TLPM`](@ref), [`FTLPM`](@ref).

# Implementation

  - Concrete subtypes must contain the following properties:

      + `target::Union{<:Real, <:AbstractVector{<:Real}, Nothing}`: scalar or `N×1` minimum return threshold for classifying downside returns. Only returns equal to or below this value are considered in the calculation. Must be in the same frequency as the returns.
      + `w::Union{<:AbstractWeights, Nothing}`: property to store a `T×1` [`AbstractWeights`](https://juliastats.org/StatsBase.jl/stable/weights/) vector for computing the weighted mean.
      + `mu::Union{<:AbstractVector, Nothing}`: property to store an `N×1` expected returns vector.

  - Ideally perform any necessary validation checks at instantiation and `setproperty!`.
"""
abstract type HCRiskMeasureTarget <: HCRiskMeasureMu end

"""
    abstract type VarianceFormulation end

Abstract type for implementing various formulations of the [`Variance`](@ref) and [`SVariance`](@ref) in optimisations which use [`JuMP`](https://github.com/jump-dev/JuMP.jl) models.

# [`Variance`](@ref)

  - If either `network_adj` or `cluster_adj` property of the [`Portfolio`](@ref) instance is [`SDP`](@ref), the formulation has no effect because this constraint type requires a [`PSDCone`](https://jump.dev/JuMP.jl/stable/tutorials/conic/tips_and_tricks/#Positive-Semidefinite-Cone) formulation of the variance.

See also: [`Variance`](@ref), [`Quad`](@ref), [`SOC`](@ref), [`Portfolio`](@ref), [`SDP`](@ref).
"""
abstract type VarianceFormulation end

"""
    struct Quad <: VarianceFormulation end

Explicit quadratic formulation for the [`Variance`](@ref) and [`SVariance`](@ref) as an optimisation model.

See also: [`VarianceFormulation`](@ref), [`SOC`](@ref), [`RSOC`](@ref), [`Variance`](@ref), [`SVariance`](@ref).

# General Behaviour

  - No additional variables or constraints introduced.
  - Performance may degrade for large portfolios.

# [`Variance`](@ref)

```math
\\begin{align}
\\underset{\\bm{w}}{\\mathrm{opt}} &\\qquad \\bm{w}^\\intercal \\mathbf{\\Sigma} \\bm{w}\\\\
\\end{align}
```

Where:

  - ``\\bm{w}``: is the `N×1` vector of asset weights.
  - ``\\mathbf{\\Sigma}``: is the `N×N` asset covariance matrix.

## Behaviour

  - Produces a [`QuadExpr`](https://jump.dev/JuMP.jl/stable/api/JuMP/#QuadExpr) risk expression `variance_risk = dot(w, sigma, w)`.

## Examples

# [`SVariance`](@ref)

```math
\\begin{align}
\\underset{\\bm{w}}{\\mathrm{opt}} &\\qquad \\dfrac{\\bm{y} \\cdot \\bm{y}}{T-1}\\nonumber\\\\
\\textrm{s.t.} &\\qquad \\bm{y} \\geq \\bm{0}\\nonumber\\\\
               &\\qquad (\\mathbf{X} - \\bm{\\mu}^\\intercal) \\bm{w} \\geq -\\bm{y}
\\end{align}
```

Where:

  - ``\\bm{w}``: is the `N×1` vector of asset weights.
  - ``\\mathbf{X}``: is the `T×N` matrix of asset returns.
  - ``\\bm{\\mu}``: is the `N×1` vector of expected returns.
  - ``\\bm{w}``: is the `N×1` vector of asset weights.
  - ``\\bm{y}``: is the `T×1` vector of deviations from the expected portfolio return that meets the minimum return threshold.

## Behaviour

  - Produces a [`QuadExpr`](https://jump.dev/JuMP.jl/stable/api/JuMP/#QuadExpr) risk expression `svariance_risk = dot(svariance, svariance)/(T-1)`.
"""
struct Quad <: VarianceFormulation end

"""
    struct SOC <: VarianceFormulation end

Second-Order Cone (SOC) formulation for the [`Variance`](@ref) and [`SVariance`](@ref). Formulates the quadratic variance/semi variance expression using a [SecondOrderCone](https://jump.dev/JuMP.jl/stable/tutorials/conic/tips_and_tricks/#Second-Order-Cone) cone constraint.

See also: [`VarianceFormulation`](@ref), [`Quad`](@ref), [`RSOC`](@ref), [`Variance`](@ref), [`SVariance`](@ref).

# General Behaviour

  - Uses [`SecondOrderCone`](https://jump.dev/JuMP.jl/stable/manual/constraints/#Second-order-cone-constraints) constraints.
  - Often more numerically stable than direct quadratic formulation.
  - Better scaling properties for large portfolios.
  - Compatible with specialised conic solvers.
  - Particularly effective for large-scale problems.

# [`Variance`](@ref)

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
  - ``\\lVert \\cdot \\rVert_{2}``: is the L-2 norm, which is modelled as an [SecondOrderCone](https://jump.dev/JuMP.jl/stable/tutorials/conic/tips_and_tricks/#Second-Order-Cone).

## Behaviour

  - Produces a [`QuadExpr`](https://jump.dev/JuMP.jl/stable/api/JuMP/#QuadExpr) risk expression `variance_risk = dev^2`.

## Examples

# [`SVariance`](@ref)

```math
\\begin{align}
\\underset{\\bm{w}}{\\mathrm{opt}} &\\qquad \\dfrac{v^{2}}{T-1}\\nonumber\\\\
\\textrm{s.t.} &\\qquad \\bm{y} \\geq \\bm{0}\\nonumber\\\\
               &\\qquad (\\mathbf{X} - \\bm{\\mu}^\\intercal) \\bm{w} \\geq -\\bm{y}\\nonumber\\\\
               &\\qquad (v,\\, \\bm{y}) \\in \\mathcal{K}_{\\textrm{soc}}
\\end{align}
```

Where:

  - ``\\bm{w}``: is the `N×1` vector of asset weights.
  - ``\\mathbf{X}``: is the `T×N` matrix of asset returns.
  - ``\\bm{\\mu}``: is the `N×1` vector of expected returns.
  - ``\\bm{w}``: is the `N×1` vector of asset weights.
  - ``\\bm{y}``: is the `T×1` vector of deviations from the expected portfolio return that meets the minimum return threshold.

## Behaviour

  - Produces a [`QuadExpr`](https://jump.dev/JuMP.jl/stable/api/JuMP/#QuadExpr) risk expression `svariance_risk = sdev^2/(T-1)`.
"""
struct SOC <: VarianceFormulation end

"""
    struct RSOC <: VarianceFormulation end

Rotated Second-Order Cone (RSOC) formulation for the [`Variance`](@ref) and [`SVariance`](@ref). Formulates the quadratic variance/semi variance expression using a [`RotatedSecondOrderCone`](https://jump.dev/JuMP.jl/stable/tutorials/conic/tips_and_tricks/#Rotated-Second-Order-Cone) constraint.

See also: [`VarianceFormulation`](@ref), [`Quad`](@ref), [`SOC`](@ref), [`Variance`](@ref), [`SVariance`](@ref).

# General Behaviour

  - Uses [`SecondOrderCone`](https://jump.dev/JuMP.jl/stable/manual/constraints/#Second-order-cone-constraints) constraints.
  - Uses [`RotatedSecondOrderCone`](https://jump.dev/JuMP.jl/stable/tutorials/conic/tips_and_tricks/#Rotated-Second-Order-Cone) constraints.
  - Compatible with specialised conic solvers.
  - Performance may degrade for large portfolios.

# [`Variance`](@ref)

```math
\\begin{align}
\\underset{\\bm{w}}{\\mathrm{opt}} &\\qquad \\dfrac{v}{T-1}\\nonumber\\\\
\\textrm{s.t.} &\\qquad \\bm{y} = (\\mathbf{X} - \\bm{\\mu}^\\intercal) \\bm{w}\\nonumber\\\\
               &\\qquad (v,\\, 0.5,\\, \\bm{y}) \\in \\mathcal{K}_{\\textrm{rsoc}}
\\end{align}
```

Where:

  - ``\\bm{w}``: is the `N×1` vector of asset weights.
  - ``\\bm{v}``: is the portfolio variance.
  - ``T``: is the number of observations.
  - ``\\bm{y}``: is the `T×1` vector of deviations from the expected portfolio return.
  - ``\\bm{\\mu}``: is the `N×1` vector of expected asset returns.
  - ``\\mathcal{K}_{\\textrm{rsoc}}``: is the [`RotatedSecondOrderCone`](https://jump.dev/JuMP.jl/stable/tutorials/conic/tips_and_tricks/#Rotated-Second-Order-Cone). Which in this formulation represents the sum of squares of a vector.

## Behaviour

  - Produces a [`QuadExpr`](https://jump.dev/JuMP.jl/stable/api/JuMP/#QuadExpr) risk expression `svariance_risk = tvariance/(T-1)`.

!!! warning

    Using `w` to compute the Variance risk of a portfolio optimised via an optimisation which uses [`JuMP`](https://github.com/jump-dev/JuMP.jl), you have to ensure that the value of `mu` used by the optimisation is consistent with the value of `w`---i.e. it was computed with [`MuSimple`](@ref) using `w`. Otherwise, the calculation will be inconsistent with the value of `:svariance_risk`. Alternatively, use the value of `mu` in both.

## Examples

# [`SVariance`](@ref)

```math
\\begin{align}
\\underset{\\bm{w}}{\\mathrm{opt}} &\\qquad \\dfrac{v}{T-1}\\nonumber\\\\
\\textrm{s.t.} &\\qquad \\bm{y} \\geq \\bm{0}\\nonumber\\\\
               &\\qquad (\\mathbf{X} - \\bm{\\mu}^\\intercal) \\bm{w} \\geq -\\bm{y}\\nonumber\\\\
               &\\qquad (v,\\, 0.5,\\, \\bm{y}) \\in \\mathcal{K}_{\\textrm{rsoc}}
\\end{align}
```

Where:

  - ``\\bm{w}``: is the `N×1` vector of asset weights.
  - ``\\mathbf{X}``: is the `T×N` matrix of asset returns.
  - ``\\bm{\\mu}``: is the `N×1` vector of expected returns.
  - ``\\bm{w}``: is the `N×1` vector of asset weights.
  - ``\\bm{y}``: is the `T×1` vector of deviations from the expected portfolio return that meets the minimum return threshold.

## Behaviour

  - Produces a [`QuadExpr`](https://jump.dev/JuMP.jl/stable/api/JuMP/#QuadExpr) risk expression `svariance_risk = tsvariance/(T-1)`.
"""
struct RSOC <: VarianceFormulation end

"""
    mutable struct Variance <: RiskMeasureSigma

Measures and computes the portfolio Variance (Variance).

```math
\\begin{align}
\\mathrm{Variance}(\\bm{w},\\, \\mathbf{\\Sigma}) &= \\bm{w}^\\intercal \\, \\mathbf{\\Sigma}\\, \\bm{w}\\,.
\\end{align}
```

Where:

  - ``\\bm{w}``: is the `N×1` vector of asset weights.
  - ``\\mathbf{\\Sigma}``: is the `N×N` asset covariance matrix.

See also: [`RiskMeasureSigma`](@ref), [`RMSettings`](@ref), [`SD`](@ref), [`PortClass`](@ref), [`OptimType`](@ref), [`NOC`](@ref), [`NoAdj`](@ref), [`IP`](@ref), [`SDP`](@ref), [`Portfolio`](@ref), [`calc_risk`](@ref), [`optimise!`](@ref), [`set_rm`](@ref).

# Keyword Arguments

  - `settings::RMSettings = RMSettings()`: risk measure configuration settings.

  - `sigma::Union{<:AbstractMatrix, Nothing} = nothing`: (optional) `N×N` covariance matrix.

      + If `nothing`: takes its value from the instance [`Portfolio`](@ref), the specific property depends on the [`PortClass`](@ref) parameter of the [`OptimType`](@ref) used.

# Behaviour in optimisations which take risk measures and use [`JuMP`](https://github.com/jump-dev/JuMP.jl) models

## [`NoAdj`](@ref), [`IP`](@ref) network and cluster constraints

  - Requires a solver that supports [`SecondOrderCone`](https://jump.dev/JuMP.jl/stable/tutorials/conic/tips_and_tricks/#Second-Order-Cone) constraints.
  - Requires a solver capable of handling quadratic expressions.
  - Defines the variance risk, `:variance_risk`, as a [`QuadExpr`](https://jump.dev/JuMP.jl/stable/api/JuMP/#QuadExpr).
  - If it exists, the upper bound is defined via the portfolio standard deviation with the key, `:dev_ub`.
  - Incompatible with [`NOC`](@ref) (Near Optimal Centering) optimisations because [`QuadExpr`](https://jump.dev/JuMP.jl/stable/api/JuMP/#QuadExpr) are strictly not convex.

## [`SDP`](@ref) network and/or cluster constraints

  - Requires a solver that supports [`PSDCone`](https://jump.dev/JuMP.jl/stable/tutorials/conic/tips_and_tricks/#Positive-Semidefinite-Cone) constraints.
  - Defines the variance risk, `:variance_risk`, as an [`AffExpr`](https://jump.dev/JuMP.jl/stable/api/JuMP/#AffExpr).
  - Compatible with [`NOC`](@ref) (Near Optimal Centering) optimisations because [`AffExpr`](https://jump.dev/JuMP.jl/stable/api/JuMP/#AffExpr) are strictly convex.
  - If it exists, the upper bound is defined via the portfolio variance with the key, `:variance_risk_ub`.

# Functor

  - `(variance::Variance)(w::AbstractVector)`: computes the Variance of an `N×1` vector of asset weights.

# Examples
"""
mutable struct Variance <: RiskMeasureRCSigma
    settings::RMSettings
    formulation::VarianceFormulation
    sigma::Union{<:AbstractMatrix, Nothing}
    a_rc::Union{<:AbstractMatrix, Nothing}
    b_rc::Union{<:AbstractVector, Nothing}
end
function Variance(; settings::RMSettings = RMSettings(),
                  formulation::VarianceFormulation = SOC(),
                  sigma::Union{<:AbstractMatrix, Nothing} = nothing,
                  a_rc::Union{<:AbstractMatrix, Nothing} = nothing,
                  b_rc::Union{<:AbstractVector, Nothing} = nothing)
    if !isnothing(sigma) && !isempty(sigma)
        @smart_assert(size(sigma, 1) == size(sigma, 2))
    end
    if !isnothing(a_rc) && !isnothing(b_rc) && !isempty(a_rc) && !isempty(b_rc)
        @smart_assert(size(a_rc, 1) == length(b_rc))
    end
    return Variance(settings, formulation, sigma, a_rc, b_rc)
end
function Base.setproperty!(obj::Variance, sym::Symbol, val)
    if sym == :sigma
        if !isnothing(val) && !isempty(val)
            @smart_assert(size(val, 1) == size(val, 2))
        end
    elseif sym == :a_rc
        if !isnothing(val) && !isnothing(obj.b_rc) && !isempty(val) && !isempty(obj.b_rc)
            @smart_assert(size(val, 1) == length(obj.b_rc))
        end
    elseif sym == :b_rc
        if !isnothing(val) && !isnothing(obj.a_rc) && !isempty(val) && !isempty(obj.a_rc)
            @smart_assert(size(obj.a_rc, 1) == length(val))
        end
    end
    return setfield!(obj, sym, val)
end
function (variance::Variance)(w::AbstractVector)
    return dot(w, variance.sigma, w)
end

"""
    mutable struct SD <: RiskMeasureSigma

Measures and computes the portfolio Standard Deviation (SD).

```math
\\begin{align}
\\mathrm{SD}(\\bm{w},\\, \\mathbf{\\Sigma}) &= \\left(\\bm{w}^\\intercal \\, \\mathbf{\\Sigma} \\, \\bm{w}\\right)^{1/2}\\,.
\\end{align}
```

Where:

  - ``\\bm{w}``: is the `N×1` vector of asset weights.
  - ``\\mathbf{\\Sigma}``: is the `N×N` asset covariance matrix.

See also: [`RiskMeasureSigma`](@ref), [`RMSettings`](@ref), [`Variance`](@ref), [`Portfolio`](@ref), [`PortClass`](@ref), [`OptimType`](@ref), [`calc_risk`](@ref), [`optimise!`](@ref), [`set_rm`](@ref).

# Keyword Arguments

  - `settings::RMSettings = RMSettings()`: risk measure configuration settings.

  - `sigma::Union{<:AbstractMatrix, Nothing} = nothing`: (optional) `N×N` covariance matrix.

      + If `nothing`: takes its value from the instance [`Portfolio`](@ref), the specific property depends on the [`PortClass`](@ref) parameter of the [`OptimType`](@ref) used.

# Behaviour in optimisations which take risk measures and use [`JuMP`](https://github.com/jump-dev/JuMP.jl) models

  - The SD risk is defined as a [`VariableRef`](https://jump.dev/JuMP.jl/stable/api/JuMP/#VariableRef) with the key, `:sd_risk`.
  - If it exists, the upper bound is defined via the portfolio variance with the key, `:sd_risk_ub`.
  - Requires a solver that supports [`SecondOrderCone`](https://jump.dev/JuMP.jl/stable/tutorials/conic/tips_and_tricks/#Second-Order-Cone) constraints.

# Functor

  - `(sd::SD)(w::AbstractVector)`: computes the SD risk of an `N×1` vector of asset weights.

# Examples
"""
mutable struct SD <: RiskMeasureSigma
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
    mutable struct MAD <: RiskMeasureMu

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

See also: [`RiskMeasureMu`](@ref), [`RMSettings`](@ref), [`Portfolio`](@ref), [`PortClass`](@ref), [`OptimType`](@ref), [`calc_risk`](@ref), [`optimise!`](@ref), [`set_rm`](@ref), [`calc_ret_mu`](@ref).

# Keyword Arguments

  - `settings::RMSettings = RMSettings()`: risk measure configuration settings.

  - `w::Union{<:AbstractWeights, Nothing} = nothing`: (optional, functor-exclusive) `T×1` vector of weights for computing the expected value of the returns vector (internal expected value) via [`calc_ret_mu`](@ref).
  - `we::Union{<:AbstractWeights, Nothing} = nothing`: (optional) `T×1` vector of weights for computing the expected value of the absolute deviation (external expected value).

      + If `isnothing(we)`: computes the unweighted mean.
      + Else: computes the weighted mean.
  - `mu::Union{<:AbstractVector{<:Real}, Nothing} = nothing`: (optional) `N×1` vector of weights for computing the expected value of the returns vector (internal expected value) via [`calc_ret_mu`](@ref) or [`set_rm`](@ref).

!!! warning

    Using `w` to compute the MAD risk of a portfolio optimised via an optimisation which uses [`JuMP`](https://github.com/jump-dev/JuMP.jl), you have to ensure that the value of `mu` used by the optimisation is consistent with the value of `w`---i.e. it was computed with [`MuSimple`](@ref) using `w`. Otherwise, the calculation will be inconsistent with the value of `:mad_risk`. Alternatively, use the value of `mu` in both.

# Behaviour in optimisations which take risk measures and use [`JuMP`](https://github.com/jump-dev/JuMP.jl) models

  - The MAD risk is defined as an [`AffExpr`](https://jump.dev/JuMP.jl/stable/api/JuMP/#AffExpr) with the key, `:mad_risk`.
  - If it exists, the upper bound is defined via the portfolio variance with the key, `:mad_risk_ub`.

# Functor

  - `(mad::MAD)(X::AbstractMatrix, w::AbstractVector, fees::Real = 0.0)`: computes the MAD risk of a `T×N` returns matrix, a `N×1` vector of asset weights `w`, and fees `fees`.

      + `fees`: must be consistent with the returns frequency.

# Examples
"""
mutable struct MAD <: RiskMeasureMu
    settings::RMSettings
    w::Union{<:AbstractWeights, Nothing}
    mu::Union{<:AbstractVector{<:Real}, Nothing}
    we::Union{<:AbstractWeights, Nothing}
end
function MAD(; settings::RMSettings = RMSettings(),
             w::Union{<:AbstractWeights, Nothing} = nothing,
             mu::Union{<:AbstractVector{<:Real}, Nothing} = nothing,
             we::Union{<:AbstractWeights, Nothing} = nothing)
    return MAD(settings, w, mu, we)
end
function (mad::MAD)(X::AbstractMatrix, w::AbstractVector, fees::Real = 0.0)
    x = X * w .- fees
    mu = calc_ret_mu(x, w, mad)
    we = mad.we
    return isnothing(we) ? mean(abs.(x .- mu)) : mean(abs.(x .- mu), we)
end

"""
    mutable struct SSD <: RiskMeasureTarget

Measures and computes the portfolio Semi Standard Deviation (SSD). Measures the standard deviation equal to or below the `target` return. If the target return is the risk-free rate, the risk-adjusted return ratio of this risk measure is commonly known as the Sortino ratio.

```math
\\begin{align}
\\mathrm{SSD}(\\bm{X}) &= \\left(\\dfrac{1}{T-1} \\sum\\limits_{t=1}^{T}\\min\\left(X_{t} - r,\\, 0\\right)^{2}\\right)^{1/2}\\,.
\\end{align}
```

Where:

  - ``\\bm{X}``: is the `T×1` vector of portfolio returns.
  - ``T``: is the number of observations.
  - ``r``: is the minimum acceptable return.
  - ``X_{t}``: is the `t`-th value of the portfolio returns vector.
  - ``\\mathbb{E}(\\cdot)``: is the expected value.

See also: [`RiskMeasureMu`](@ref), [`RMSettings`](@ref), [`Portfolio`](@ref), [`SD`](@ref), [`Variance`](@ref), [`SVariance`](@ref), [`calc_risk`](@ref), [`optimise!`](@ref), [`set_rm`](@ref).

# Keyword Arguments

  - `settings::RMSettings = RMSettings()`: risk measure configuration settings.
  - `target::target::Union{<:Real, <:AbstractVector{<:Real}, Nothing} = nothing`: minimum return threshold for classifying downside returns. Only returns equal to or below this value are considered in the calculation. Its value can be computed via [`calc_target_ret_mu`](@ref) or [`calc_rm_target`](@ref).
  - `w::Union{<:AbstractWeights, Nothing} = nothing`: (optional, functor-exclusive) `T×1` vector of weights for computing the expected value of the returns vector via [`calc_ret_mu`](@ref).
  - `mu::Union{<:AbstractVector{<:Real}, Nothing} = nothing`: (optional) `N×1` vector of weights for computing the expected value of the returns vector via [`calc_ret_mu`](@ref) or [`set_rm`](@ref).

!!! warning

    Using `w` to compute the SSD risk of a portfolio optimised via an optimisation which uses [`JuMP`](https://github.com/jump-dev/JuMP.jl), you have to ensure that the value of `mu` used by the optimisation is consistent with the value of `w`---i.e. it was computed with [`MuSimple`](@ref) using `w`. Otherwise, the calculation will be inconsistent with the value of `:ssd_risk`. Alternatively, use the value of `mu` in both.

# Behaviour in optimisations which take risk measures and use [`JuMP`](https://github.com/jump-dev/JuMP.jl) models

  - The SSD risk is defined as an [`AffExpr`](https://jump.dev/JuMP.jl/stable/api/JuMP/#AffExpr) with the key, `:ssd_risk`.
  - If it exists, the upper bound is defined via the portfolio variance with the key, `:ssd_risk_ub`.

# Functor

  - `(ssd::SSD)(X::AbstractMatrix, w::AbstractVector, fees::Real = 0.0)`: computes the SSD risk of a `T×N` returns matrix, a `N×1` vector of asset weights `w`, and fees `fees`.

      + `fees`: must be consistent with the returns frequency.

# Examples
"""
mutable struct SSD <: RiskMeasureTarget
    settings::RMSettings
    target::Union{<:Real, <:AbstractVector{<:Real}, Nothing}
    w::Union{<:AbstractWeights, Nothing}
    mu::Union{<:AbstractVector{<:Real}, Nothing}
end
function SSD(; settings::RMSettings = RMSettings(),
             target::Union{<:Real, <:AbstractVector{<:Real}, Nothing} = nothing,
             w::Union{<:AbstractWeights, Nothing} = nothing,
             mu::Union{<:AbstractVector{<:Real}, Nothing} = nothing)
    return SSD(settings, target, w, mu)
end
function (ssd::SSD)(X::AbstractMatrix, w::AbstractVector, fees::Real = 0.0)
    x = X * w .- fees
    T = length(x)
    mu = calc_target_ret_mu(x, w, ssd)
    val = x .- mu
    val = val[val .<= zero(eltype(val))]
    return sqrt(dot(val, val) / (T - 1))
end

"""
    mutable struct FLPM <: RiskMeasureTarget

Measures and computes the portfolio First Lower Partial Moment (FLPM). Measures the dispersion equal to or below the `target` return threshold. The risk-adjusted return ratio of this risk measure is commonly known as the Omega ratio.

```math
\\begin{align}
\\mathrm{FLPM}(\\bm{X},\\, r) &= \\dfrac{1}{T} \\sum\\limits_{t=1}^{T}\\max\\left(r - X_{t},\\, 0\\right)\\,.
\\end{align}
```

Where:

  - ``\\bm{X}``: is the `T×1` vector of portfolio returns.
  - ``r``: is the minimum acceptable return.
  - ``T``: is the number of observations.
  - ``X_{t}``: is the `t`-th value of the portfolio returns vector.

See also: [`RiskMeasureTarget`](@ref), [`RMSettings`](@ref), [`Portfolio`](@ref), [`calc_risk`](@ref), [`optimise!`](@ref), [`set_rm`](@ref).

# Keyword Arguments

  - `settings::RMSettings = RMSettings()`: risk measure configuration settings.
  - `target::T1 = 0.0`: minimum return threshold for classifying downside returns. Only returns equal to or below this value are considered in the calculation. Its value can be computed via [`calc_target_ret_mu`](@ref) or [`calc_rm_target`](@ref).
  - `w::Union{<:AbstractWeights, Nothing} = nothing`: (optional, functor-exclusive) `T×1` vector of weights for computing the expected value of the returns vector via [`calc_ret_mu`](@ref).
  - `mu::Union{<:AbstractVector{<:Real}, Nothing} = nothing`: (optional) `N×1` vector of weights for computing the expected value of the returns vector via [`calc_ret_mu`](@ref) or [`set_rm`](@ref).

!!! warning

    Using `w` to compute the FLPM risk of a portfolio optimised via an optimisation which uses [`JuMP`](https://github.com/jump-dev/JuMP.jl), you have to ensure that the value of `mu` used by the optimisation is consistent with the value of `w`---i.e. it was computed with [`MuSimple`](@ref) using `w`. Otherwise, the calculation will be inconsistent with the value of `:flpm_risk`. Alternatively, use the value of `mu` in both.

# Behaviour in optimisations which take risk measures and use [`JuMP`](https://github.com/jump-dev/JuMP.jl) models

  - The FLPM risk is defined as an [`AffExpr`](https://jump.dev/JuMP.jl/stable/api/JuMP/#AffExpr) with the key, `:flpm_risk`.
  - If it exists, the upper bound is defined via the portfolio variance with the key, `:flpm_risk_ub`.

# Functor

  - `(flpm::FLPM)(X::AbstractMatrix, w::AbstractVector, fees::Real = 0.0)`: computes the FLPM risk of a `T×N` returns matrix, a `N×1` vector of asset weights `w`, and fees `fees`.

      + `fees`: must be consistent with the returns frequency.

# Examples
"""
mutable struct FLPM <: RiskMeasureTarget
    settings::RMSettings
    target::Union{<:Real, <:AbstractVector{<:Real}, Nothing}
    w::Union{<:AbstractWeights, Nothing}
    mu::Union{<:AbstractVector{<:Real}, Nothing}
end
function FLPM(; settings::RMSettings = RMSettings(),
              target::Union{<:Real, <:AbstractVector{<:Real}, Nothing} = 0.0,
              w::Union{<:AbstractWeights, Nothing} = nothing,
              mu::Union{<:AbstractVector{<:Real}, Nothing} = nothing)
    return FLPM(settings, target, w, mu)
end
function (flpm::FLPM)(X::AbstractMatrix, w::AbstractVector, fees::Real = 0.0)
    x = X * w .- fees
    T = length(x)
    target = calc_target_ret_mu(x, w, flpm)
    val = x .- target
    val = val[val .<= zero(eltype(val))]
    return -sum(val) / T
end

"""
    struct WR <: RiskMeasure

Measures and computes the portfolio Worst Realization/Return (WR). It is the absolute value of the most extreme loss for the period. Best used in combination with other risk measures.

  - ``\\mathrm{VaR}(\\bm{X},\\, \\alpha) \\leq \\mathrm{CVaR}(\\bm{X},\\, \\alpha) \\leq \\mathrm{EVaR}(\\bm{X},\\, \\alpha) \\leq \\mathrm{RLVaR}(\\bm{X},\\, \\alpha,\\, \\kappa) \\leq \\mathrm{WR}(\\bm{X})``.

```math
\\begin{align}
\\mathrm{WR}(\\bm{X}) &= -\\min(\\bm{X})\\,.
\\end{align}
```

Where:

  - ``\\bm{X}``: is the `T×1` vector of portfolio returns.

See also: [`RiskMeasure`](@ref), [`RMSettings`](@ref), [`Portfolio`](@ref), [`VaR`](@ref), [`CVaR`](@ref), [`EVaR`](@ref), [`RLVaR`](@ref), [`calc_risk`](@ref), [`optimise!`](@ref), [`set_rm`](@ref).

# Keyword Arguments

  - `settings::RMSettings = RMSettings()`: risk measure configuration settings.

# Behaviour in optimisations which take risk measures and use [`JuMP`](https://github.com/jump-dev/JuMP.jl) models

  - The WR risk is defined as an [`AffExpr`](https://jump.dev/JuMP.jl/stable/api/JuMP/#AffExpr) with the key, `:wr_risk`.
  - If it exists, the upper bound is defined via the portfolio variance with the key, `:wr_risk_ub`.

# Functor

  - `(wr::WR)(x::AbstractVector)`: computes the WR risk of a `T×1` vector of portfolio returns `x`.

# Examples
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

Measures and computes the portfolio Conditional Value at Risk (CVaR). Also known as the Expected Shortfall, it is the weighted average of all tail losses up to the Value at Risk, which is the threshold below or equal to which are the worst `alpha %` of portfolio returns.

  - ``\\mathrm{VaR}(\\bm{X},\\, \\alpha) \\leq \\mathrm{CVaR}(\\bm{X},\\, \\alpha) \\leq \\mathrm{EVaR}(\\bm{X},\\, \\alpha) \\leq \\mathrm{RLVaR}(\\bm{X},\\, \\alpha,\\, \\kappa) \\leq \\mathrm{WR}(\\bm{X})``.

```math
\\begin{align}
\\mathrm{CVaR}(\\bm{X},\\, \\alpha) &= \\mathrm{VaR}(\\bm{X},\\, \\alpha) + \\dfrac{1}{\\alpha T} \\sum\\limits_{t=1}^{T} \\max\\left(-X_{t} - \\mathrm{VaR}(\\bm{X},\\, \\alpha),\\, 0\\right)\\,.
\\end{align}
```

Where:

  - ``\\bm{X}``: is the `T×1` vector of portfolio returns.
  - ``\\alpha``: is the significance level.
  - ``\\mathrm{VaR}(\\bm{X},\\, \\alpha)``: is the Value at Risk as defined in [`VaR`](@ref).
  - ``T``: is the number of observations.
  - ``X_{t}``: is the `t`-th value of the portfolio returns vector.

See also: [`RiskMeasure`](@ref), [`RMSettings`](@ref), [`Portfolio`](@ref), [`VaR`](@ref), [`WR`](@ref), [`EVaR`](@ref), [`RLVaR`](@ref), [`calc_risk`](@ref), [`optimise!`](@ref), [`set_rm`](@ref).

# Keyword Arguments

  - `settings::RMSettings = RMSettings()`: configuration settings for the risk measure.
  - `alpha::T1 = 0.05`: significance level, `alpha ∈ (0, 1)`.

# Behaviour in optimisations which take risk measures and use [`JuMP`](https://github.com/jump-dev/JuMP.jl) models

  - The CVaR is defined as an [`AffExpr`](https://jump.dev/JuMP.jl/stable/api/JuMP/#AffExpr) with the key, `:cvar_risk`.
  - If it exists, the upper bound is defined via the portfolio variance with the key, `:cvar_risk_ub`.

# Functor

  - `(cvar::CVaR)(x::AbstractVector)`: computes the CVaR of a `T×1` vector of portfolio returns `x`.

# Examples
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
    sum_var = zero(eltype(x))
    for i ∈ 1:(idx - 1)
        sum_var += x[i] + var
    end
    return var - sum_var / aT
end

"""
"""
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
    sum_var = zero(eltype(x))
    for i ∈ 1:(idx - 1)
        sum_var += x[i] + var
    end
    return var - sum_var / aT
end

"""
    mutable struct EVaR{T1 <: Real} <: RiskMeasureSolvers

Measures and computes the portfolio Entropic Value at Risk (EVaR). It is the upper bound of the Chernoff inequality for the [`VaR`](@ref) and [`CVaR`](@ref).

  - Uses [`ExponentialCone`](https://jump.dev/JuMP.jl/stable/tutorials/conic/tips_and_tricks/#Exponential-Cone) constraints.
  - ``\\mathrm{VaR}(\\bm{X},\\, \\alpha) \\leq \\mathrm{CVaR}(\\bm{X},\\, \\alpha) \\leq \\mathrm{EVaR}(\\bm{X},\\, \\alpha) \\leq \\mathrm{RLVaR}(\\bm{X},\\, \\alpha,\\, \\kappa) \\leq \\mathrm{WR}(\\bm{X})``.

```math
\\begin{align}
\\mathrm{EVaR}(\\bm{X},\\alpha) &= \\underset{z > 0}{\\inf} \\left\\{\\mathrm{ERM}(\\bm{X},\\, z, \\,\\alpha)\\right\\}\\,.
\\end{align}
```

Where:

  - ``\\mathrm{ERM}(\\bm{X},\\, z, \\,\\alpha)`` is the entropic risk measure as defined in [`ERM`](@ref).
  - ``z``: is the entropic value at risk.
  - ``\\bm{X}``: is the `T×1` vector of portfolio returns.
  - ``\\alpha``: is the significance level.

See also: [`RiskMeasureSolvers`](@ref), [`RMSettings`](@ref), [`Portfolio`](@ref), [`VaR`](@ref), [`WR`](@ref), [`CVaR`](@ref), [`RLVaR`](@ref), [`calc_risk`](@ref), [`optimise!`](@ref), [`set_rm`](@ref).

# Keyword Arguments

  - `settings::RMSettings = RMSettings()`: configuration settings for the risk measure.

  - `alpha::T1 = 0.05`: significance level, `alpha ∈ (0, 1)`.
  - `solvers::Union{Nothing, PortOptSolver, <:AbstractVector{PortOptSolver}} = nothing`: optional abstract dictionary containing the solvers, their settings, solution criteria, and other arguments.

      + If `isnothing(solvers)`: it takes its value from the `solvers` property of the instance of [`Portfolio`](@ref).

# Behaviour in optimisations which take risk measures and use [`JuMP`](https://github.com/jump-dev/JuMP.jl) models

  - The EVaR is defined as an [`AffExpr`](https://jump.dev/JuMP.jl/stable/api/JuMP/#AffExpr) with the key, `:evar_risk`.
  - If it exists, the upper bound is defined via the portfolio variance with the key, `:evar_risk_ub`.

# Functor

  - `(evar::EVaR)(x::AbstractVector)`: computes the EVaR of a `T×1` vector of portfolio returns `x`.

# Examples
"""
mutable struct EVaR{T1 <: Real} <: RiskMeasureSolvers
    settings::RMSettings
    alpha::T1
    solvers::Union{Nothing, PortOptSolver, <:AbstractVector{PortOptSolver}}
end
function EVaR(; settings::RMSettings = RMSettings(), alpha::Real = 0.05,
              solvers::Union{Nothing, PortOptSolver, <:AbstractVector{PortOptSolver}} = nothing)
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

mutable struct EVaRRG{T1 <: Real, T2 <: Real} <: RiskMeasureSolvers
    settings::RMSettings
    alpha::T1
    beta::T2
    solvers::Union{Nothing, PortOptSolver, <:AbstractVector{PortOptSolver}}
end
function EVaRRG(; settings::RMSettings = RMSettings(), alpha::Real = 0.05,
                beta::Real = 0.05,
                solvers::Union{Nothing, PortOptSolver, <:AbstractVector{PortOptSolver}} = nothing)
    @smart_assert(zero(alpha) < alpha < one(alpha))
    @smart_assert(zero(beta) < beta < one(beta))
    return EVaRRG{typeof(alpha), typeof(beta)}(settings, alpha, beta, solvers)
end
function Base.setproperty!(obj::EVaRRG, sym::Symbol, val)
    if sym ∈ (:alpha, :beta)
        @smart_assert(zero(val) < val < one(val))
    end
    return setfield!(obj, sym, val)
end
function (evarrg::EVaRRG)(x::AbstractVector)
    return ERM(x, evarrg.solvers, evarrg.alpha) + ERM(-x, evarrg.solvers, evarrg.beta)
end

"""
    mutable struct RLVaR{T1 <: Real, T2 <: Real} <: RiskMeasureSolvers

Measures and computes the portfolio Relativistic Value at Risk (RLVaR). It is a generalisation of the [`EVaR`](@ref).

  - Uses [`PowerCone`](https://jump.dev/JuMP.jl/stable/tutorials/conic/tips_and_tricks/#PowerCone) constraints.
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
  - ``\\bm{X}``: is the `T×1` vector of portfolio returns.
  - ``\\alpha``: is the significance level.
  - ``\\kappa``: is the relativistic deformation parameter.

See also: [`RiskMeasureSolvers`](@ref), [`RMSettings`](@ref), [`Portfolio`](@ref), [`VaR`](@ref), [`WR`](@ref), [`CVaR`](@ref), [`EVaR`](@ref), [`calc_risk`](@ref), [`optimise!`](@ref), [`set_rm`](@ref).

# Keyword Arguments

  - `settings::RMSettings = RMSettings()`: configuration settings for the risk measure.

  - `alpha::T1 = 0.05`: significance level, `alpha ∈ (0, 1)`.
  - `kappa::T1 = 0.3`: relativistic deformation level, `kappa ∈ (0, 1)`.
  - `solvers::Union{Nothing, PortOptSolver, <:AbstractVector{PortOptSolver}} = nothing`: optional abstract dictionary containing the solvers, their settings, solution criteria, and other arguments. In order to solve the problem, a solver must be compatible with [`ExponentialCone`](https://jump.dev/JuMP.jl/stable/tutorials/conic/tips_and_tricks/#Exponential-Cone).

      + If `isnothing(solvers)`: it takes its value from the `solvers` property of the instance of [`Portfolio`](@ref).

# Behaviour in optimisations which take risk measures and use [`JuMP`](https://github.com/jump-dev/JuMP.jl) models

  - The RLVaR is defined as an [`AffExpr`](https://jump.dev/JuMP.jl/stable/api/JuMP/#AffExpr) with the key, `:rlvar_risk`.
  - If it exists, the upper bound is defined via the portfolio variance with the key, `:rlvar_risk_ub`.

# Functor

  - `(rlvar::RLVaR)(x::AbstractVector)`: computes the RLVaR of a `T×1` vector of portfolio returns `x`.

# Examples
"""
mutable struct RLVaR{T1 <: Real, T2 <: Real} <: RiskMeasureSolvers
    settings::RMSettings
    alpha::T1
    kappa::T2
    solvers::Union{Nothing, PortOptSolver, <:AbstractVector{PortOptSolver}}
end
function RLVaR(; settings::RMSettings = RMSettings(), alpha::Real = 0.05, kappa::Real = 0.3,
               solvers::Union{Nothing, PortOptSolver, <:AbstractVector{PortOptSolver}} = nothing)
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

mutable struct RLVaRRG{T1 <: Real, T2 <: Real, T3 <: Real, T4 <: Real} <: RiskMeasureSolvers
    settings::RMSettings
    alpha::T1
    kappa_a::T2
    beta::T1
    kappa_b::T2
    solvers::Union{Nothing, PortOptSolver, <:AbstractVector{PortOptSolver}}
end
function RLVaRRG(; settings::RMSettings = RMSettings(), alpha::Real = 0.05,
                 kappa_a::Real = 0.3, beta::Real = 0.05, kappa_b::Real = 0.3,
                 solvers::Union{Nothing, PortOptSolver, <:AbstractVector{PortOptSolver}} = nothing)
    @smart_assert(zero(alpha) < alpha < one(alpha))
    @smart_assert(zero(kappa_a) < kappa_a < one(kappa_a))
    @smart_assert(zero(beta) < beta < one(beta))
    @smart_assert(zero(kappa_b) < kappa_b < one(kappa_b))
    return RLVaRRG{typeof(alpha), typeof(kappa_a), typeof(beta), typeof(kappa_b)}(settings,
                                                                                  alpha,
                                                                                  kappa_a,
                                                                                  beta,
                                                                                  kappa_b,
                                                                                  solvers)
end
function Base.setproperty!(obj::RLVaRRG, sym::Symbol, val)
    if sym ∈ (:alpha, :kappa_a, :beta, :kappa_b)
        @smart_assert(zero(val) < val < one(val))
    end
    return setfield!(obj, sym, val)
end
function (rlvarrg::RLVaRRG)(x::AbstractVector)
    return RRM(x, rlvarrg.solvers, rlvarrg.alpha, rlvarrg.kappa_a) +
           RRM(-x, rlvarrg.solvers, rlvarrg.beta, rlvarrg.kappa_b)
end

"""
    struct MDD <: RiskMeasure

Measures and computes the portfolio Maximum Drawdown of uncompounded cumulative returns (MDD). It measures the largest peak-to-trough decline. Best used in combination with other risk measures. The risk-adjusted return ratio of this risk measure is commonly known as the Calmar ratio.

  - ``\\mathrm{DaR}(\\bm{X},\\, \\alpha) \\leq \\mathrm{CDaR}(\\bm{X},\\, \\alpha) \\leq \\mathrm{EDaR}(\\bm{X},\\, \\alpha) \\leq \\mathrm{RLDaR}(\\bm{X},\\, \\alpha,\\, \\kappa) \\leq \\mathrm{MDD}(\\bm{X})``.

```math
\\begin{align}
\\mathrm{MDD_{a}}(\\bm{X}) &= \\max\\mathrm{DD_{a}}(\\bm{X})\\,.
\\end{align}
```

Where:

  - ``\\mathrm{DD_{a}}(\\bm{X})`` is the Drawdown of uncompounded cumulative returns as defined in [`DaR`](@ref).
  - ``\\bm{X}``: is the `T×1` vector of portfolio returns.

See also: [`RiskMeasure`](@ref), [`RMSettings`](@ref), [`Portfolio`](@ref), [`DaR`](@ref), [`CDaR`](@ref), [`EDaR`](@ref), [`RLDaR`](@ref), [`DaR_r`](@ref), [`CDaR_r`](@ref), [`EDaR_r`](@ref), [`RLDaR_r`](@ref), [`calc_risk`](@ref), [`optimise!`](@ref), [`set_rm`](@ref).

# Keyword Arguments

  - `settings::RMSettings = RMSettings()`: risk measure configuration settings.

# Behaviour in optimisations which take risk measures and use [`JuMP`](https://github.com/jump-dev/JuMP.jl) models

  - The MDD is defined as an [`AffExpr`](https://jump.dev/JuMP.jl/stable/api/JuMP/#AffExpr) with the key, `:mdd_risk`.
  - If it exists, the upper bound is defined via the portfolio variance with the key, `:mdd_risk_ub`.

# Functor

  - `(mdd::MDD)(x::AbstractVector)`: computes the MDD of a `T×1` vector of portfolio returns `x`.

# Examples
"""
struct MDD <: RiskMeasure
    settings::RMSettings
end
function MDD(; settings::RMSettings = RMSettings())
    return MDD(settings)
end
function (mdd::MDD)(x::AbstractVector)
    pushfirst!(x, 1)
    cs = cumsum(x)
    val = zero(eltype(x))
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

Measures and computes the portfolio Average (Expected) Drawdown of uncompounded cumulative returns (ADD). This is a generalisation to accomodate the use of weighted means.

```math
\\begin{align}
\\mathrm{ADD_{a}}(\\bm{X}) &= \\mathbb{E}\\lleft(\\mathrm{DD_{a}}(\\bm{X}, j)\\,\\forall j=1\\,\\ldots\\,T\\right)\\,.
\\end{align}
```

Where:

  - ``\\mathrm{DD_{a}}(\\bm{X}, j)`` is the Drawdown of uncompounded cumulative returns at time ``j`` as defined in [`DaR`](@ref).
  - ``T``: is the number of observations.
  - ``\\bm{X}``: is the `T×1` vector of portfolio returns.

See also: [`RiskMeasure`](@ref), [`RMSettings`](@ref), [`Portfolio`](@ref), [`DaR`](@ref), [`calc_risk`](@ref), [`optimise!`](@ref), [`set_rm`](@ref).

# Keyword Arguments

  - `w::Union{<:AbstractWeights, Nothing} = nothing`: optional `T×1` vector of weights for computing the expected value of the returns vector.

# Behaviour in optimisations which take risk measures and use [`JuMP`](https://github.com/jump-dev/JuMP.jl) models

  - The ADD is defined as an [`AffExpr`](https://jump.dev/JuMP.jl/stable/api/JuMP/#AffExpr) with the key, `:add_risk`.
  - If it exists, the upper bound is defined via the portfolio variance with the key, `:add_risk_ub`.

# Functor

  - `(add::ADD)(x::AbstractVector)`: computes the ADD of a `T×1` vector of portfolio returns `x`.

# Examples
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
    val = zero(eltype(x))
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
        for (idx, i) ∈ pairs(cs)
            if i > peak
                peak = i
            end
            dd = peak - i
            if dd > 0
                wi = isone(idx) ? 1 : w[idx - 1]
                val += dd * wi
            end
        end
        popfirst!(x)
        return val / sum(w)
    end
end

"""
    mutable struct CDaR{T1 <: Real} <: RiskMeasure

Measures and computes the portfolio Conditional Drawdown at Risk of uncompounded cumulative returns (CDaR). It is the weighted average of all drawdowns up to the Drawdown at Risk, which is the threshold below or equal to which are the worst `alpha %` of portfolio drawdowns.

  - ``\\mathrm{DaR}(\\bm{X},\\, \\alpha) \\leq \\mathrm{CDaR}(\\bm{X},\\, \\alpha) \\leq \\mathrm{EDaR}(\\bm{X},\\, \\alpha) \\leq \\mathrm{RLDaR}(\\bm{X},\\, \\alpha,\\, \\kappa) \\leq \\mathrm{MDD}(\\bm{X})``.

```math
\\begin{align}
\\mathrm{CDaR_{a}}(\\bm{X},\\, \\alpha) &= \\mathrm{DaR_{a}}(\\bm{X},\\, \\alpha) + \\dfrac{1}{\\alpha T} \\sum\\limits_{j=0}^{T} \\max\\left(\\mathrm{DD_{a}}(\\bm{X},\\, j) - \\mathrm{DaR_{a}}(\\bm{X},\\, \\alpha),\\, 0 \\right)\\,.
\\end{align}
```

Where:

  - ``\\mathrm{DD_{a}}(\\bm{X},\\, j)`` is the Drawdown of uncompounded cumulative returns at time ``j`` as defined in [`DaR`](@ref).
  - ``\\mathrm{DaR_{a}}(\\bm{X},\\, \\alpha)`` is the Drawdown at Risk of uncompounded cumulative returns as defined in [`DaR`](@ref).
  - ``\\bm{X}``: is the `T×1` vector of portfolio returns.
  - ``\\alpha``: is the significance level.
  - ``T``: is the number of observations.

See also: [`RiskMeasure`](@ref), [`RMSettings`](@ref), [`Portfolio`](@ref), [`DaR`](@ref), [`MDD`](@ref), [`EDaR`](@ref), [`RLDaR`](@ref), [`calc_risk`](@ref), [`optimise!`](@ref), [`set_rm`](@ref).

# Keyword Arguments

  - `settings::RMSettings = RMSettings()`: configuration settings for the risk measure.
  - `alpha::T1 = 0.05`: significance level, `alpha ∈ (0, 1)`.

# Behaviour in optimisations which take risk measures and use [`JuMP`](https://github.com/jump-dev/JuMP.jl) models

  - The CDaR is defined as an [`AffExpr`](https://jump.dev/JuMP.jl/stable/api/JuMP/#AffExpr) with the key, `:cdar_risk`.
  - If it exists, the upper bound is defined via the portfolio variance with the key, `:cdar_risk_ub`.

# Functor

  - `(cdar::CDaR)(x::AbstractVector)`: computes the CDaR of a `T×1` vector of portfolio returns `x`.

# Examples
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
    sum_var = zero(eltype(x))
    for i ∈ 1:(idx - 1)
        sum_var += dd[i] + var
    end
    return var - sum_var / aT
end

"""
    mutable struct UCI <: RiskMeasure

Measures and computes the portfolio Ulcer Index of uncompounded cumulative returns (UCI). It is the normalised L2-norm of the portfolio drawdowns of uncompounded cumulative returns.

```math
\\begin{align}
\\mathrm{UCI_{a}}(\\bm{X}) &= \\left(\\dfrac{1}{T} \\sum\\limits_{j=0}^{T} \\mathrm{DD_{a}}(\\bm{X},\\, j)^{2}\\right)^{1/2}\\,.
\\end{align}
```

Where:

  - ``\\mathrm{DD_{a}}(\\bm{X},\\, j)`` is the Drawdown of uncompounded cumulative returns at time ``j`` as defined in [`DaR`](@ref).
  - ``\\bm{X}``: is the `T×1` vector of portfolio returns.

See also: [`RiskMeasure`](@ref), [`RMSettings`](@ref), [`Portfolio`](@ref), [`DaR`](@ref), [`calc_risk`](@ref), [`optimise!`](@ref), [`set_rm`](@ref).

# Keyword Arguments

  - `settings::RMSettings = RMSettings()`: configuration settings for the risk measure.

# Behaviour in optimisations which take risk measures and use [`JuMP`](https://github.com/jump-dev/JuMP.jl) models

  - The UCI is defined as an [`AffExpr`](https://jump.dev/JuMP.jl/stable/api/JuMP/#AffExpr) with the key, `:uci_risk`.
  - If it exists, the upper bound is defined via the portfolio variance with the key, `:uci_risk_ub`.

# Functor

  - `(uci::UCI)(x::AbstractVector)`: computes the UCI of a `T×1` vector of portfolio returns `x`.

# Examples
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
    val = zero(eltype(x))
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
    mutable struct EDaR{T1 <: Real} <: RiskMeasureSolvers

Measures and computes the portfolio Entropic Drawdown at Risk of uncompounded cumulative returns (EDaR). It is the upper bound of the Chernoff inequality for the [`DaR`](@ref) and [`CDaR`](@ref).

  - Uses [`ExponentialCone`](https://jump.dev/JuMP.jl/stable/tutorials/conic/tips_and_tricks/#Exponential-Cone) constraints.
  - ``\\mathrm{DaR}(\\bm{X},\\, \\alpha) \\leq \\mathrm{CDaR}(\\bm{X},\\, \\alpha) \\leq \\mathrm{EDaR}(\\bm{X},\\, \\alpha) \\leq \\mathrm{RLDaR}(\\bm{X},\\, \\alpha,\\, \\kappa) \\leq \\mathrm{MDD}(\\bm{X})``.

```math
\\begin{align}
\\mathrm{EDaR_{a}}(\\bm{X},\\alpha) &= \\underset{z > 0}{\\inf} \\left\\{\\mathrm{ERM}(\\mathrm{DD_{a}}(\\bm{X}),\\, z, \\,\\alpha)\\right\\}\\,.
\\end{align}
```

Where:

  - ``\\mathrm{ERM}(\\mathrm{DD_{a}}(\\bm{X}),\\, z, \\,\\alpha)`` is the Entropic Risk Measure as defined in [`ERM`](@ref), using the Drawdown of uncompounded cumulative returns as defined in [`DaR`](@ref).
  - ``z``: is the entropic drawdown at risk.
  - ``\\bm{X}``: is the `T×1` vector of portfolio returns.
  - ``\\alpha``: is the significance level.

See also: [`RiskMeasureSolvers`](@ref), [`RMSettings`](@ref), [`Portfolio`](@ref), [`DaR`](@ref), [`MDD`](@ref), [`CDaR`](@ref), [`RLDaR`](@ref), [`calc_risk`](@ref), [`optimise!`](@ref), [`set_rm`](@ref).

# Keyword Arguments

  - `settings::RMSettings = RMSettings()`: configuration settings for the risk measure.

  - `alpha::T1 = 0.05`: significance level, `alpha ∈ (0, 1)`.
  - `solvers::Union{Nothing, PortOptSolver, <:AbstractVector{PortOptSolver}} = nothing`: optional abstract dictionary containing the solvers, their settings, solution criteria, and other arguments.

      + If `isnothing(solvers)`: it takes its value from the `solvers` property of the instance of [`Portfolio`](@ref).

# Behaviour in optimisations which take risk measures and use [`JuMP`](https://github.com/jump-dev/JuMP.jl) models

  - The EDaR is defined as an [`AffExpr`](https://jump.dev/JuMP.jl/stable/api/JuMP/#AffExpr) with the key, `:edar_risk`.
  - If it exists, the upper bound is defined via the portfolio variance with the key, `:edar_risk_ub`.

# Functor

  - `(edar::EDaR)(x::AbstractVector)`: computes the EDaR of a `T×1` vector of portfolio returns `x`.

# Examples
"""
mutable struct EDaR{T1 <: Real} <: RiskMeasureSolvers
    settings::RMSettings
    alpha::T1
    solvers::Union{Nothing, PortOptSolver, <:AbstractVector{PortOptSolver}}
end
function EDaR(; settings::RMSettings = RMSettings(), alpha::Real = 0.05,
              solvers::Union{Nothing, PortOptSolver, <:AbstractVector{PortOptSolver}} = nothing)
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
    mutable struct RLDaR{T1 <: Real, T2 <: Real} <: RiskMeasureSolvers

Measures and computes the portfolio Relativistic Drawdown at Risk of uncompounded cumulative returns (RLDaR). It is a generalisation of the [`EDaR`](@ref).

  - Uses [`PowerCone`](https://jump.dev/JuMP.jl/stable/tutorials/conic/tips_and_tricks/#PowerCone) constraints.
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
  - ``\\bm{X}``: is the `T×1` vector of portfolio returns.
  - ``\\alpha``: is the significance level.
  - ``\\kappa``: is the relativistic deformation parameter.

See also: [`RiskMeasureSolvers`](@ref), [`RMSettings`](@ref), [`Portfolio`](@ref), [`VaR`](@ref), [`WR`](@ref), [`CVaR`](@ref), [`EVaR`](@ref), [`calc_risk`](@ref), [`optimise!`](@ref), [`set_rm`](@ref).

# Keyword Arguments

  - `settings::RMSettings = RMSettings()`: configuration settings for the risk measure.

  - `alpha::T1 = 0.05`: significance level, `alpha ∈ (0, 1)`.
  - `kappa::T1 = 0.3`: relativistic deformation level, `kappa ∈ (0, 1)`.
  - `solvers::Union{Nothing, PortOptSolver, <:AbstractVector{PortOptSolver}} = nothing`: optional abstract dictionary containing the solvers, their settings, solution criteria, and other arguments.

      + If `isnothing(solvers)`: it takes its value from the `solvers` property of the instance of [`Portfolio`](@ref).

# Behaviour in optimisations which take risk measures and use [`JuMP`](https://github.com/jump-dev/JuMP.jl) models

  - The RLDaR is defined as an [`AffExpr`](https://jump.dev/JuMP.jl/stable/api/JuMP/#AffExpr) with the key, `:rldar_risk`.
  - If it exists, the upper bound is defined via the portfolio variance with the key, `:rldar_risk_ub`.

# Functor

  - `(rldar::RLDaR)(x::AbstractVector)`: computes the RLDaR of a `T×1` vector of portfolio returns `x`.

# Examples
"""
mutable struct RLDaR{T1 <: Real, T2 <: Real} <: RiskMeasureSolvers
    settings::RMSettings
    alpha::T1
    kappa::T2
    solvers::Union{Nothing, PortOptSolver, <:AbstractVector{PortOptSolver}}
end
function RLDaR(; settings = RMSettings(), alpha::Real = 0.05, kappa = 0.3,
               solvers::Union{Nothing, PortOptSolver, <:AbstractVector{PortOptSolver}} = nothing)
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
    mutable struct Kurt <: RiskMeasureMu

Measures and computes the portfolio Square Root Kurtosis (Kurt).

```math
\\begin{align}
\\mathrm{Kurt}(\\bm{X}) &= \\left(\\dfrac{1}{T} \\sum\\limits_{t=1}^{T} \\left(X_{t} - \\mathbb{E}(\\bm{X}) \\right)^{4} \\right)^{1/2}\\,.
\\end{align}
```

Where:

  - ``\\bm{X}``: is the `T×1` vector of portfolio returns.
  - ``T``: is the number of observations.
  - ``X_{t}``: is the `t`-th value of the portfolio returns vector.
  - ``\\mathbb{E}(\\cdot)``: is the expected value.

See also: [`RiskMeasureMu`](@ref), [`RMSettings`](@ref), [`SKurt`](@ref), [`Kurtosis`](@ref), [`SKurtosis`](@ref), [`calc_risk`](@ref), [`Portfolio`](@ref), [`optimise!`](@ref), [`set_rm`](@ref).

# Keyword Arguments

  - `settings::RMSettings = RMSettings()`: risk measure configuration settings.

  - `w::Union{<:AbstractWeights, Nothing} = nothing`: (optional, functor-exclusive) `T×1` vector of weights for computing the expected value of the returns vector via [`calc_ret_mu`](@ref).
  - `mu::Union{<:AbstractVector{<:Real}, Nothing} = nothing`: (optional, functor-exclusive) `N×1` vector of weights for computing the expected value of the returns vector via [`calc_ret_mu`](@ref).
  - `kt::Union{<:AbstractMatrix, Nothing} = nothing`: (optional, [`set_rm`](@ref) exclusive) `N²×N²` cokurtosis matrix.

      + If `nothing`: takes its value from the `kurt` instance [`Portfolio`](@ref).
      + `kt` has no effect in optimisations that don't use [`JuMP`](https://github.com/jump-dev/JuMP.jl) models, because the kurtosis is computed from the returns vector.

!!! warning

    Using `w` to compute the Kurt risk of a portfolio optimised via an optimisation which uses [`JuMP`](https://github.com/jump-dev/JuMP.jl), you have to ensure that the value of `kurt` in the instance of [`Portfolio`](@ref) used by the optimisation is consistent with the value of `w`---i.e. it was computed with [`MuSimple`](@ref) using `w`. Otherwise, the calculation will be inconsistent with the value of `:kurt_risk`.

# Behaviour in optimisations which take risk measures and use [`JuMP`](https://github.com/jump-dev/JuMP.jl) models

  - Uses [`PSDCone`](https://jump.dev/JuMP.jl/stable/tutorials/conic/tips_and_tricks/#Positive-Semidefinite-Cone) constraints.
  - The Kurt risk is defined as an [`AffExpr`](https://jump.dev/JuMP.jl/stable/api/JuMP/#AffExpr) with the key, `:kurt_risk`.
  - If it exists, the upper bound is defined via the portfolio variance with the key, `:kurt_risk_ub`.

# Functor

  - `(kurt::Kurt)(x::AbstractVector)`: computes the Kurt risk of a `T×1` vector of portfolio returns `x`.

# Examples
"""
mutable struct Kurt <: RiskMeasureMu
    settings::RMSettings
    w::Union{<:AbstractWeights, Nothing}
    mu::Union{<:AbstractVector{<:Real}, Nothing}
    kt::Union{<:AbstractMatrix, Nothing}
end
function Kurt(; settings::RMSettings = RMSettings(),
              w::Union{<:AbstractWeights, Nothing} = nothing,
              mu::Union{<:AbstractVector{<:Real}, Nothing} = nothing,
              kt::Union{<:AbstractMatrix, Nothing} = nothing)
    if !isnothing(kt)
        @smart_assert(size(kt, 1) == size(kt, 2))
    end
    return Kurt(settings, w, mu, kt)
end
function (kurt::Kurt)(X::AbstractMatrix, w::AbstractVector, fees::Real = 0.0;
                      scale::Bool = false)
    x = X * w .- fees
    T = length(x)
    mu = calc_ret_mu(x, w, kurt)
    val = x .- mu
    kurt = sqrt(sum(val .^ 4) / T)
    return !scale ? kurt : kurt / 2
end

"""
    mutable struct SKurt <: RiskMeasureMu

Measures and computes the portfolio Square Root Semi Kurtosis (SKurt). Measures the kurtosis equal to or below the mean portfolio returns.

```math
\\begin{align}
\\mathrm{SKurt}(\\bm{X}) &= \\left(\\dfrac{1}{T} \\sum\\limits_{t=1}^{T} \\min\\left(X_{t} - \\mathbb{E}(\\bm{X}),\\, 0 \\right)^{4} \\right)^{1/2}\\,.
\\end{align}
```

Where:

  - ``\\bm{X}``: is the `T×1` vector of portfolio returns.
  - ``T``: is the number of observations.
  - ``X_{t}``: is the `t`-th value of the portfolio returns vector.
  - ``\\mathbb{E}(\\cdot)``: is the expected value.

See also: [`RiskMeasureMu`](@ref), [`RMSettings`](@ref), [`Kurt`](@ref), [`Kurtosis`](@ref), [`SKurtosis`](@ref), [`Portfolio`](@ref), [`calc_risk`](@ref), [`optimise!`](@ref), [`set_rm`](@ref).

# Keyword Arguments

  - `settings::RMSettings = RMSettings()`: risk measure configuration settings.

  - `w::Union{<:AbstractWeights, Nothing} = nothing`: (optional, functor-exclusive) `T×1` vector of weights for computing the expected value of the returns vector via [`calc_ret_mu`](@ref).
  - `mu::Union{<:AbstractVector{<:Real}, Nothing} = nothing`: (optional, functor-exclusive) `N×1` vector of weights for computing the expected value of the returns vector via [`calc_ret_mu`](@ref).
  - `kt::Union{<:AbstractMatrix, Nothing} = nothing`: (optional, [`set_rm`](@ref) exclusive) `N²×N²` cokurtosis matrix.

      + If `nothing`: takes its value from the `skurt` instance [`Portfolio`](@ref).
      + `kt` has no effect in optimisations that don't use [`JuMP`](https://github.com/jump-dev/JuMP.jl) models, because the kurtosis is computed from the returns vector.

!!! warning

    Using `w` to compute the Kurt risk of a portfolio optimised via an optimisation which uses [`JuMP`](https://github.com/jump-dev/JuMP.jl), you have to ensure that the value of `skurt` in the instance of [`Portfolio`](@ref) used by the optimisation is consistent with the value of `w`---i.e. it was computed with [`MuSimple`](@ref) using `w`. Otherwise, the calculation will be inconsistent with the value of `:skurt_risk`.

# Behaviour in optimisations which take risk measures and use [`JuMP`](https://github.com/jump-dev/JuMP.jl) models

  - Uses [`PSDCone`](https://jump.dev/JuMP.jl/stable/tutorials/conic/tips_and_tricks/#Positive-Semidefinite-Cone) constraints.
  - The Kurt risk is defined as an [`AffExpr`](https://jump.dev/JuMP.jl/stable/api/JuMP/#AffExpr) with the key, `:skurt_risk`.
  - If it exists, the upper bound is defined via the portfolio variance with the key, `:skurt_risk_ub`.

# Functor

  - `(skurt::SKurt)(x::AbstractVector)`: computes the SKurt risk of a `T×1` vector of portfolio returns `x`.

# Examples
"""
mutable struct SKurt <: RiskMeasureMu
    settings::RMSettings
    w::Union{<:AbstractWeights, Nothing}
    mu::Union{<:AbstractVector{<:Real}, Nothing}
    kt::Union{<:AbstractMatrix, Nothing}
end
function SKurt(; settings::RMSettings = RMSettings(),
               w::Union{<:AbstractWeights, Nothing} = nothing,
               mu::Union{<:AbstractVector{<:Real}, Nothing} = nothing,
               kt::Union{<:AbstractMatrix, Nothing} = nothing)
    if !isnothing(kt)
        @smart_assert(size(kt, 1) == size(kt, 2))
    end
    return SKurt(settings, w, mu, kt)
end
function (skurt::SKurt)(X::AbstractMatrix, w::AbstractVector, fees::Real = 0.0;
                        scale::Bool = false)
    x = X * w .- fees
    T = length(x)
    mu = calc_ret_mu(x, w, skurt)
    val = x .- mu
    skurt = sqrt(sum(val[val .<= zero(eltype(val))] .^ 4) / T)
    return !scale ? skurt : skurt / 2
end

"""
    mutable struct RG <: RiskMeasure

Measures and computes the portfolio Range (RG). It is the difference between the best and worst return.

```math
\\begin{align}
\\mathrm{RG}(\\bm{X}) &= \\max(\\bm{X}) - \\min(\\bm{X})\\,.
\\end{align}
```

Where:

  - ``\\bm{X}``: is the `T×1` vector of portfolio returns.

See also: [`RiskMeasure`](@ref), [`RMSettings`](@ref), [`Portfolio`](@ref), [`WR`](@ref), [`calc_risk`](@ref), [`optimise!`](@ref), [`set_rm`](@ref).

# Keyword Arguments

  - `settings::RMSettings = RMSettings()`: risk measure configuration settings.

# Behaviour in optimisations which take risk measures and use [`JuMP`](https://github.com/jump-dev/JuMP.jl) models

  - The RG risk is defined as an [`AffExpr`](https://jump.dev/JuMP.jl/stable/api/JuMP/#AffExpr) with the key, `:rg_risk`.
  - If it exists, the upper bound is defined via the portfolio variance with the key, `:rg_risk_ub`.

# Functor

  - `(rg::RG)(x::AbstractVector)`: computes the RG risk of a `T×1` vector of portfolio returns `x`.

# Examples
"""
struct RG <: RiskMeasure
    settings::RMSettings
end
function RG(; settings::RMSettings = RMSettings())
    return RG(settings)
end
function (rg::RG)(x::AbstractVector)
    lo, hi = extrema(x)
    return hi - lo
end

"""
    mutable struct CVaRRG{T1 <: Real, T2 <: Real} <: RiskMeasure

Measures and computes the portfolio Conditional Value at Risk Range (CVaRRG). It measures the difference between the weighted average of all head gains up to the Value at Risk for gains, which is the threshold above or equal to which are the best `beta %` of portfolio returns. And the weighted average of all tail losses up to the Value at Risk, which is the threshold below or equal to which are the worst `alpha %` of portfolio returns.

```math
\\begin{align}
\\mathrm{CVaRRG}(\\bm{X},\\, \\alpha,\\,\\beta) &= \\mathrm{CVaR}(\\bm{X},\\, \\alpha) - \\mathrm{CVaR}(-\\bm{X},\\, \\beta)
\\end{align}
```

Where:

  - ``\\mathrm{CVaR}(\\bm{X},\\, \\alpha)``: is the Conditional Value at Risk as defined in [`CVaR`](@ref).
  - ``\\bm{X}``: is the `T×1` vector of portfolio returns.
  - ``\\alpha``: is the significance level of losses.
  - ``\\beta``: is the significance level of gains.

See also: See also: [`RiskMeasure`](@ref), [`RMSettings`](@ref), [`Portfolio`](@ref), [`CVaR`](@ref), [`calc_risk`](@ref), [`optimise!`](@ref), [`set_rm`](@ref).

# Keyword Arguments

  - `settings::RMSettings = RMSettings()`: configuration settings for the risk measure.
  - `alpha::T1 = 0.05`: significance level of losses, `alpha ∈ (0, 1)`.
  - `beta::T1 = 0.05`: significance level of gains, `beta ∈ (0, 1)`.

# Behaviour in optimisations which take risk measures and use [`JuMP`](https://github.com/jump-dev/JuMP.jl) models

  - The CVaRRG is defined as an [`AffExpr`](https://jump.dev/JuMP.jl/stable/api/JuMP/#AffExpr) with the key, `:cvarrg_risk`.
  - If it exists, the upper bound is defined via the portfolio variance with the key, `:cvarrg_risk_ub`.

# Functor

  - `(cvarrg::CVaRRG)(x::AbstractVector)`: computes the CVaRRG of a `T×1` vector of portfolio returns `x`.

# Examples
"""
mutable struct CVaRRG{T1 <: Real, T2 <: Real} <: RiskMeasure
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
    alpha = cvarrg.alpha
    aT = alpha * length(x)
    idx = ceil(Int, aT)
    var = -partialsort!(x, idx)
    sum_var = zero(eltype(x))
    for i ∈ 1:(idx - 1)
        sum_var += x[i] + var
    end
    loss = var - sum_var / aT

    beta = cvarrg.beta
    aT = beta * length(x)
    idx = ceil(Int, aT)
    var = -partialsort!(x, idx; rev = true)
    sum_var = zero(eltype(x))
    for i ∈ 1:(idx - 1)
        sum_var += x[i] + var
    end
    gain = var - sum_var / aT

    return loss - gain
end

"""
    mutable struct GMD <: RiskMeasureOWA

Measures and computes the portfolio Gini Mean Difference (GMD).

See also: See also: [`RiskMeasureOWA`](@ref), [`RMSettings`](@ref), [`Portfolio`](@ref), [`owa_gmd`](@ref), [`calc_risk`](@ref), [`optimise!`](@ref), [`set_rm`](@ref).

# Keyword Arguments

  - `settings::RMSettings = RMSettings()`: configuration settings for the risk measure.
  - `formulation::OWAFormulation = OWAApprox()`: OWA risk measure settings.

# Behaviour in optimisations which take risk measures and use [`JuMP`](https://github.com/jump-dev/JuMP.jl) models

  - The GMD risk is defined as an [`AffExpr`](https://jump.dev/JuMP.jl/stable/api/JuMP/#AffExpr) with the key, `:gmd_risk`.
  - If it exists, the upper bound is defined via the portfolio variance with the key, `:gmd_risk_ub`.

# Functor

  - `(gmd::GMD)(x::AbstractVector)`: computes the GMD risk of a `T×1` vector of portfolio returns `x`.

# Examples
"""
mutable struct GMD <: RiskMeasureOWA
    settings::RMSettings
    formulation::OWAFormulation
end
function GMD(; settings::RMSettings = RMSettings(),
             formulation::OWAFormulation = OWAApprox())
    return GMD(settings, formulation)
end
function (gmd::GMD)(x::AbstractVector)
    T = length(x)
    w = owa_gmd(T)
    return dot(w, sort!(x))
end

"""
    mutable struct TG{T1 <: Real, T2 <: Real, T3 <: Integer} <: RiskMeasureOWA

Measures and computes the portfolio Tail Gini (TG).

See also: See also: [`RiskMeasureOWA`](@ref), [`RMSettings`](@ref), [`Portfolio`](@ref), [`owa_tg`](@ref), [`calc_risk`](@ref), [`optimise!`](@ref), [`set_rm`](@ref).

# Keyword Arguments

  - `settings::RMSettings = RMSettings()`: configuration settings for the risk measure.
  - `formulation::OWAFormulation = OWAApprox()`: OWA risk measure settings.
  - `alpha_i::T1 = 0.0001`: start value of the significance level of CVaR losses, `0 < alpha_i < alpha < 1`.
  - `alpha::T2 = 0.05`: end value of the significance level of CVaR losses, `0 < alpha_i < alpha < 1`.
  - `a_sim::T3 = 100`: number of CVaRs to approximate the Tail Gini losses, `a_sim > 0`.

# Behaviour in optimisations which take risk measures and use [`JuMP`](https://github.com/jump-dev/JuMP.jl) models

  - The TG risk is defined as an [`AffExpr`](https://jump.dev/JuMP.jl/stable/api/JuMP/#AffExpr) with the key, `:tg_risk`.
  - If it exists, the upper bound is defined via the portfolio variance with the key, `:tg_risk_ub`.

# Functor

  - `(tg::TG)(x::AbstractVector)`: computes the TG risk of a `T×1` vector of portfolio returns `x`.

# Examples
"""
mutable struct TG{T1 <: Real, T2 <: Real, T3 <: Integer} <: RiskMeasureOWA
    settings::RMSettings
    formulation::OWAFormulation
    alpha_i::T1
    alpha::T2
    a_sim::T3
end
function TG(; settings::RMSettings = RMSettings(),
            formulation::OWAFormulation = OWAApprox(), alpha_i::Real = 0.0001,
            alpha::Real = 0.05, a_sim::Integer = 100)
    @smart_assert(zero(alpha) < alpha_i < alpha < one(alpha))
    @smart_assert(a_sim > zero(a_sim))
    return TG{typeof(alpha_i), typeof(alpha), typeof(a_sim)}(settings, formulation, alpha_i,
                                                             alpha, a_sim)
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
    mutable struct TGRG{T1 <: Real, T2 <: Real, T3 <: Integer, 
                        T4 <: Real, T5 <: Real, T6 <: Integer} <: RiskMeasureOWA

Measures and computes the portfolio Tail Gini Range (TGRG). It is the difference between the Head Gini and Tail Gini.

```math
\\begin{align}
\\mathrm{TGRG}(\\bm{X},\\, \\alpha,\\, \\beta) &= \\mathrm{TG}(\\bm{X},\\,\\alpha) - \\mathrm{TG}(-\\bm{X},\\,\\beta)\\,.
\\end{align}
```

Where:

  - ``\\mathrm{TG}``: is the Tail Gini defined in [`TG`](@ref).
  - ``\\bm{X}``: is the `T×1` vector of portfolio returns.
  - ``\\alpha``: is the significance level of losses.
  - ``\\beta``: is the significance level of gains.

See also: See also: [`RiskMeasureOWA`](@ref), [`RMSettings`](@ref), [`Portfolio`](@ref), [`owa_tgrg`](@ref), [`calc_risk`](@ref), [`optimise!`](@ref), [`set_rm`](@ref).

# Keyword Arguments

  - `settings::RMSettings = RMSettings()`: configuration settings for the risk measure.
  - `formulation::OWAFormulation = OWAApprox()`: OWA risk measure settings.
  - `alpha_i::T1 = 0.0001`: start value of the significance level of CVaR losses, `0 < alpha_i < alpha < 1`.
  - `alpha::T2 = 0.05`: end value of the significance level of CVaR losses, `0 < alpha_i < alpha < 1`.
  - `a_sim::T3 = 100`: number of CVaRs to approximate the Tail Gini losses, `a_sim > 0`.
  - `beta_i::T4 = 0.0001`: start value of the significance level of CVaR gains, `0 < beta_i < beta < 1`.
  - `beta::T5 = 0.05`: end value of the significance level of CVaR gains, `0 < beta_i < beta < 1`.
  - `b_sim::T6 = 100`: number of CVaRs to approximate the Tail Gini gains, `b_sim > 0`.

# Behaviour in optimisations which take risk measures and use [`JuMP`](https://github.com/jump-dev/JuMP.jl) models

  - The TGRG risk is defined as an [`AffExpr`](https://jump.dev/JuMP.jl/stable/api/JuMP/#AffExpr) with the key, `:tgrg_risk`.
  - If it exists, the upper bound is defined via the portfolio variance with the key, `:tgrg_risk_ub`.

# Functor

  - `(tgrg::TGRG)(x::AbstractVector)`: computes the TGRG risk of a `T×1` vector of portfolio returns `x`.

# Examples
"""
mutable struct TGRG{T1 <: Real, T2 <: Real, T3 <: Integer, T4 <: Real, T5 <: Real,
                    T6 <: Integer} <: RiskMeasureOWA
    settings::RMSettings
    formulation::OWAFormulation
    alpha_i::T1
    alpha::T2
    a_sim::T3
    beta_i::T4
    beta::T5
    b_sim::T6
end
function TGRG(; settings::RMSettings = RMSettings(),
              formulation::OWAFormulation = OWAApprox(), alpha_i = 0.0001,
              alpha::Real = 0.05, a_sim::Integer = 100, beta_i = 0.0001, beta::Real = 0.05,
              b_sim::Integer = 100)
    @smart_assert(zero(alpha) < alpha_i < alpha < one(alpha))
    @smart_assert(a_sim > zero(a_sim))
    @smart_assert(zero(beta) < beta_i < beta < one(beta))
    @smart_assert(b_sim > zero(b_sim))
    return TGRG{typeof(alpha_i), typeof(alpha), typeof(a_sim), typeof(beta_i), typeof(beta),
                typeof(b_sim)}(settings, formulation, alpha_i, alpha, a_sim, beta_i, beta,
                               b_sim)
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
    w = owa_tgrg(T; alpha_i = tgrg.alpha_i, alpha = tgrg.alpha, a_sim = tgrg.a_sim,
                 beta_i = tgrg.beta_i, beta = tgrg.beta, b_sim = tgrg.b_sim)
    return dot(w, sort!(x))
end

"""
    mutable struct OWA <: RiskMeasureOWA

# Description

Defines the generic Ordered Weight Array.

  - Uses a vector of ordered weights generated by [`owa_l_moment`](@ref) or [`owa_l_moment_crm`](@ref) for arbitrary L-moment optimisations.

See also: [`RiskMeasureOWA`](@ref), [`RMSettings`](@ref), [`Portfolio`](@ref), [`optimise!`](@ref), [`set_rm`](@ref), [`calc_risk(::OWA, ::AbstractVector)`](@ref), [`owa_l_moment`](@ref), [`owa_l_moment_crm`](@ref).

# Properties

  - `settings::RMSettings = RMSettings()`: configuration settings for the risk measure.
  - `formulation::OWAFormulation = OWAApprox()`: OWA risk measure settings.
  - `w::Union{<:AbstractWeights, Nothing} = nothing`: `T×1` ordered weight vector of arbitrary L-moments generated by [`owa_l_moment`](@ref) or [`owa_l_moment_crm`](@ref).

# Examples
"""
mutable struct OWA <: RiskMeasureOWA
    settings::RMSettings
    formulation::OWAFormulation
    w::Union{<:AbstractVector, Nothing}
end
function OWA(; settings::RMSettings = RMSettings(),
             formulation::OWAFormulation = OWAApprox(),
             w::Union{<:AbstractVector, Nothing} = nothing)
    return OWA(settings, formulation, w)
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

Measures and computes the portfolio Brownian Distance Variance (BDVariance). It captures linear and non-linear codependence between assets.

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

Where:

  - ``\\bm{X}`` and ``\\bm{Y}`` : are random variables, they are equal in this case as they are the portfolio returns.
  - ``a_{i,\\,j}`` and ``b_{i,\\,j}`` : are entries of a distance matrix where ``i`` and ``j`` are points in time. Each entry is defined as the Euclidean distance ``\\lVert \\cdot \\rVert_{2}`` between the value of the random variable at time ``i`` and its value at time ``j``.
  - ``\\bar{a}_{i\\,\\cdot}`` and ``\\bar{b}_{i\\,\\cdot}`` : are the ``i``-th row means of their respective matrices.
  - ``\\bar{a}_{\\cdot\\,j}`` and ``\\bar{b}_{\\cdot\\,j}`` : are the ``j``-th column means of their respective matrices.
  - ``\\bar{a}_{\\cdot\\,\\cdot}`` and ``\\bar{b}_{\\cdot\\,\\cdot}`` : are the grand means of their respective matrices.
  - ``A_{i,\\,j}`` and ``B_{i,\\,j}`` : are doubly centered distances.

See also: See also: [`RiskMeasure`](@ref), [`RMSettings`](@ref), [`Portfolio`](@ref), [`calc_risk`](@ref), [`optimise!`](@ref), [`set_rm`](@ref).

# Keyword Arguments

  - `settings::RMSettings = RMSettings()`: configuration settings for the risk measure.

# Behaviour in optimisations which take risk measures and use [`JuMP`](https://github.com/jump-dev/JuMP.jl) models

  - The BDVariance risk is defined as a [`QuadExpr`](https://jump.dev/JuMP.jl/stable/api/JuMP/#QuadExpr) with the key, `:bdvariance_risk`.
  - If it exists, the upper bound is defined via the portfolio variance with the key, `:bdvariance_risk_ub`.

# Functor

  - `(bdvariance::BDVariance)(x::AbstractVector)`: computes the BDVariance risk of a `T×1` vector of portfolio returns `x`.

# Examples
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
    mutable struct NQSkew <: RiskMeasureSkew

Measures and computes the portfolio Negative Quadratic Skewness (NQSkew).

```math
\\begin{align}
\\nu &= \\bm{w}^{\\intercal} \\mathbf{V} \\bm{w}\\\\
\\end{align}
```

Where:

  - ``\\bm{w}``: is the vector of asset weights.
  - ``\\mathbf{V}``: is the sum of the symmetric negative spectral slices of the coskewness.

See also: See also: [`RiskMeasureSkew`](@ref), [`RMSettings`](@ref), [`Portfolio`](@ref), [`calc_risk`](@ref), [`optimise!`](@ref), [`set_rm`](@ref).

# Keyword Arguments

  - `settings::RMSettings = RMSettings()`: configuration settings for the risk measure.

  - `skew::Union{<:AbstractMatrix, Nothing}`: (optional) `N×N²` custom coskewness matrix.

      + If `isnothing(skew)`: takes its value from the `skew` property of the instance of [`Portfolio`](@ref).
  - `V::Union{Nothing, <:AbstractMatrix}`: (optional) `Na×Na` custom sum of the symmetric negative spectral slices of the coskewness.

      + If `isnothing(V)`: takes its value from the `V` property of the instance of [`Portfolio`](@ref).

# Behaviour in optimisations which take risk measures and use [`JuMP`](https://github.com/jump-dev/JuMP.jl) models

  - The NQSkew risk is defined as an [`QuadExpr`](https://jump.dev/JuMP.jl/stable/api/JuMP/#AffExpr) with the key, `:nqskew_risk = t_nqskew^2`.
  - If it exists, the upper bound is defined via the portfolio variance with the key, `:t_nqskew_ub`.
  - Incompatible with [`NOC`](@ref) (Near Optimal Centering) optimisations because [`QuadExpr`](https://jump.dev/JuMP.jl/stable/api/JuMP/#QuadExpr) are strictly not convex.

# Functor

  - `(nqskew::NQSkew)(w::AbstractVector)`: computes the NQSkew risk of an `N×1` vector of asset weights.

# Examples
"""
mutable struct NQSkew <: RiskMeasureSkew
    settings::RMSettings
    skew::Union{<:AbstractMatrix, Nothing}
    V::Union{<:AbstractMatrix, Nothing}
end
function NQSkew(; settings::RMSettings = RMSettings(),
                skew::Union{<:AbstractMatrix, Nothing} = nothing,
                V::Union{<:AbstractMatrix, Nothing} = nothing)
    if !isnothing(skew)
        @smart_assert(size(skew, 1)^2 == size(skew, 2))
    end
    if !isnothing(V)
        @smart_assert(size(V, 1) == size(V, 2))
    end
    return NQSkew(settings, skew, V)
end
function Base.setproperty!(obj::NQSkew, sym::Symbol, val)
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
function (nqskew::NQSkew)(w::AbstractVector)
    return dot(w, nqskew.V, w)
end

"""
    mutable struct NQSSkew <: RiskMeasureSkew

Measures and computes the portfolio Negative Quadratic Semi Skewness (NQSSkew).

```math
\\begin{align}
\\nu &= \\bm{w}^{\\intercal} \\mathbf{V} \\bm{w}\\\\
\\end{align}
```

Where:

  - ``\\bm{w}``: is the vector of asset weights.
  - ``\\mathbf{V}``: is the sum of the symmetric negative spectral slices of the semi coskewness.

See also: See also: [`RiskMeasureSkew`](@ref), [`RMSettings`](@ref), [`Portfolio`](@ref), [`calc_risk`](@ref), [`optimise!`](@ref), [`set_rm`](@ref).

# Keyword Arguments

  - `settings::RMSettings = RMSettings()`: configuration settings for the risk measure.

  - `skew::Union{<:AbstractMatrix, Nothing}`: (optional) `N×N²` custom semi coskewness matrix.

      + If `isnothing(skew)`: takes its value from the `sskew` property of the instance of [`Portfolio`](@ref).
  - `V::Union{Nothing, <:AbstractMatrix}`: (optional) `Na×Na` custom sum of the symmetric negative spectral slices of the semi coskewness.

      + If `isnothing(V)`: takes its value from the `SV` property of the instance of [`Portfolio`](@ref).

# Behaviour in optimisations which take risk measures and use [`JuMP`](https://github.com/jump-dev/JuMP.jl) models

  - The NQSSkew risk is defined as an [`AffExpr`](https://jump.dev/JuMP.jl/stable/api/JuMP/#AffExpr) with the key, `:nqsskew_risk = t_nqsskew^2`.
  - If it exists, the upper bound is defined via the portfolio variance with the key, `:t_nqsskew_ub`.
  - Incompatible with [`NOC`](@ref) (Near Optimal Centering) optimisations because [`QuadExpr`](https://jump.dev/JuMP.jl/stable/api/JuMP/#QuadExpr) are strictly not convex.

# Functor

  - `(nqsskew::NQSSkew)(w::AbstractVector)`: computes the NQSSkew risk of an `N×1` vector of asset weights.

# Examples
"""
mutable struct NQSSkew <: RiskMeasureSkew
    settings::RMSettings
    skew::Union{<:AbstractMatrix, Nothing}
    V::Union{Nothing, <:AbstractMatrix}
end
function NQSSkew(; settings::RMSettings = RMSettings(),
                 skew::Union{<:AbstractMatrix, Nothing} = nothing,
                 V::Union{<:AbstractMatrix, Nothing} = nothing)
    if !isnothing(skew)
        @smart_assert(size(skew, 1)^2 == size(skew, 2))
    end
    if !isnothing(V)
        @smart_assert(size(V, 1) == size(V, 2))
    end
    return NQSSkew(settings, skew, V)
end
function Base.setproperty!(obj::NQSSkew, sym::Symbol, val)
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
function (nqsskew::NQSSkew)(w::AbstractVector)
    return dot(w, nqsskew.V, w)
end
"""
    mutable struct NSkew <: RiskMeasureSkew

Measures and computes the portfolio Negative Skewness (NSkew).

```math
\\begin{align}
\\nu &= \\left(\\bm{w}^{\\intercal} \\mathbf{V} \\bm{w}\\right)^{1/2}\\\\
\\end{align}
```

Where:

  - ``\\bm{w}``: is the vector of asset weights.
  - ``\\mathbf{V}``: is the sum of the symmetric negative spectral slices of the coskewness.

See also: See also: [`RiskMeasureSkew`](@ref), [`RMSettings`](@ref), [`Portfolio`](@ref), [`calc_risk`](@ref), [`optimise!`](@ref), [`set_rm`](@ref).

# Keyword Arguments

  - `settings::RMSettings = RMSettings()`: configuration settings for the risk measure.

  - `skew::Union{<:AbstractMatrix, Nothing}`: (optional) `N×N²` custom coskewness matrix.

      + If `isnothing(skew)`: takes its value from the `skew` property of the instance of [`Portfolio`](@ref).
  - `V::Union{Nothing, <:AbstractMatrix}`: (optional) `Na×Na` custom sum of the symmetric negative spectral slices of the coskewness.

      + If `isnothing(V)`: takes its value from the `V` property of the instance of [`Portfolio`](@ref).

# Behaviour in optimisations which take risk measures and use [`JuMP`](https://github.com/jump-dev/JuMP.jl) models

  - The NQSkew risk is defined as an [`AffExpr`](https://jump.dev/JuMP.jl/stable/api/JuMP/#AffExpr) with the key, `:nskew_risk`.
  - If it exists, the upper bound is defined via the portfolio variance with the key, `:nskew_risk_ub`.

# Functor

  - `(nskew::NSkew)(w::AbstractVector)`: computes the NSkew risk of an `N×1` vector of asset weights.

# Examples
"""
mutable struct NSkew <: RiskMeasureSkew
    settings::RMSettings
    skew::Union{<:AbstractMatrix, Nothing}
    V::Union{<:AbstractMatrix, Nothing}
end
function NSkew(; settings::RMSettings = RMSettings(),
               skew::Union{<:AbstractMatrix, Nothing} = nothing,
               V::Union{<:AbstractMatrix, Nothing} = nothing)
    if !isnothing(skew)
        @smart_assert(size(skew, 1)^2 == size(skew, 2))
    end
    if !isnothing(V)
        @smart_assert(size(V, 1) == size(V, 2))
    end
    return NSkew(settings, skew, V)
end
function Base.setproperty!(obj::NSkew, sym::Symbol, val)
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
function (nskew::NSkew)(w::AbstractVector)
    return sqrt(dot(w, nskew.V, w))
end

"""
    mutable struct NSSkew <: RiskMeasureSkew

Measures and computes the portfolio Negative Semi Skewness (NSSkew).

```math
\\begin{align}
\\nu &= \\left(\\bm{w}^{\\intercal} \\mathbf{V} \\bm{w}\\right)^{1/2}\\\\
\\end{align}
```

Where:

  - ``\\bm{w}``: is the vector of asset weights.
  - ``\\mathbf{V}``: is the sum of the symmetric negative spectral slices of the semi coskewness.

See also: See also: [`RiskMeasureSkew`](@ref), [`RMSettings`](@ref), [`Portfolio`](@ref), [`calc_risk`](@ref), [`optimise!`](@ref), [`set_rm`](@ref).

# Keyword Arguments

  - `settings::RMSettings = RMSettings()`: configuration settings for the risk measure.

  - `skew::Union{<:AbstractMatrix, Nothing}`: (optional) `N×N²` custom semi coskewness matrix.

      + If `isnothing(skew)`: takes its value from the `sskew` property of the instance of [`Portfolio`](@ref).
  - `V::Union{Nothing, <:AbstractMatrix}`: (optional) `Na×Na` custom sum of the symmetric negative spectral slices of the semi coskewness.

      + If `isnothing(V)`: takes its value from the `SV` property of the instance of [`Portfolio`](@ref).

# Behaviour in optimisations which take risk measures and use [`JuMP`](https://github.com/jump-dev/JuMP.jl) models

  - The NSSkew risk is defined as an [`AffExpr`](https://jump.dev/JuMP.jl/stable/api/JuMP/#AffExpr) with the key, `:nsskew_risk`.
  - If it exists, the upper bound is defined via the portfolio variance with the key, `:nsskew_risk_ub`.

# Functor

  - `(nsskew::NSSkew)(w::AbstractVector)`: computes the NSSkew risk of an `N×1` vector of asset weights.

# Examples
"""
mutable struct NSSkew <: RiskMeasureSkew
    settings::RMSettings
    skew::Union{<:AbstractMatrix, Nothing}
    V::Union{Nothing, <:AbstractMatrix}
end
function NSSkew(; settings::RMSettings = RMSettings(),
                skew::Union{<:AbstractMatrix, Nothing} = nothing,
                V::Union{<:AbstractMatrix, Nothing} = nothing)
    if !isnothing(skew)
        @smart_assert(size(skew, 1)^2 == size(skew, 2))
    end
    if !isnothing(V)
        @smart_assert(size(V, 1) == size(V, 2))
    end
    return NSSkew(settings, skew, V)
end
function Base.setproperty!(obj::NSSkew, sym::Symbol, val)
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
function (nsskew::NSSkew)(w::AbstractVector)
    return sqrt(dot(w, nsskew.V, w))
end

"""
    mutable struct SVariance <: RiskMeasureTarget

Measures and computes the portfolio Semi Variance (SVariance). Measures the variance equal to or below the target return.

```math
\\begin{align}
\\mathrm{SVariance}(\\bm{X}) &= \\dfrac{1}{T-1} \\sum\\limits_{t=1}^{T}\\min\\left(X_{t} - \\mathbb{E}(\\bm{X}),\\, 0\\right)^{2}\\,.
\\end{align}
```

Where:

  - ``\\bm{X}``: is the `T×1` vector of portfolio returns.
  - ``T``: is the number of observations.
  - ``X_{t}``: is the `t`-th value of the portfolio returns vector.
  - ``\\mathbb{E}(\\cdot)``: is the expected value.

See also: [`RiskMeasureMu`](@ref), [`RMSettings`](@ref), [`Portfolio`](@ref), [`SD`](@ref), [`Variance`](@ref), [`SSD`](@ref), [`NOC`](@ref), [`calc_risk`](@ref), [`optimise!`](@ref), [`set_rm`](@ref).

# Keyword Arguments

  - `settings::RMSettings = RMSettings()`: risk measure configuration settings.
  - `target::target::Union{<:Real, <:AbstractVector{<:Real}, Nothing} = nothing`: minimum return threshold for classifying downside returns. Only returns equal to or below this value are considered in the calculation. Its value can be computed via [`calc_target_ret_mu`](@ref) or [`calc_rm_target`](@ref).
  - `w::Union{<:AbstractWeights, Nothing} = nothing`: (optional, functor-exclusive) `T×1` vector of weights for computing the expected value of the returns vector via [`calc_ret_mu`](@ref).
  - `mu::Union{<:AbstractVector{<:Real}, Nothing} = nothing`: (optional) `N×1` vector of weights for computing the expected value of the returns vector via [`calc_ret_mu`](@ref) or [`set_rm`](@ref).

!!! warning

    Using `w` to compute the SVariance risk of a portfolio optimised via an optimisation which uses [`JuMP`](https://github.com/jump-dev/JuMP.jl), you have to ensure that the value of `mu` used by the optimisation is consistent with the value of `w`---i.e. it was computed with [`MuSimple`](@ref) using `w`. Otherwise, the calculation will be inconsistent with the value of `:svariance_risk`. Alternatively, use the value of `mu` in both.

# Behaviour in optimisations which take risk measures and use [`JuMP`](https://github.com/jump-dev/JuMP.jl) models

  - The SVariance risk is defined as a [`QuadExpr`](https://jump.dev/JuMP.jl/stable/api/JuMP/#QuadExpr) with the key, `:svariance_risk`.
  - If it exists, the upper bound is defined via the portfolio semi standard deviation with the key, `:sdev_ub`.
  - Incompatible with [`NOC`](@ref) (Near Optimal Centering) optimisations because [`QuadExpr`](https://jump.dev/JuMP.jl/stable/api/JuMP/#QuadExpr) are strictly not convex.

# Functor

  - `(svariance::SVariance)(X::AbstractMatrix, w::AbstractVector, fees::Real = 0.0)`: computes the SVariance risk of a `T×N` returns matrix, a `N×1` vector of asset weights `w`, and fees `fees`.

      + `fees`: must be consistent with the returns frequency.

# Examples
"""
mutable struct SVariance <: RiskMeasureTarget
    settings::RMSettings
    formulation::VarianceFormulation
    target::Union{<:Real, <:AbstractVector{<:Real}, Nothing}
    w::Union{<:AbstractWeights, Nothing}
    mu::Union{<:AbstractVector{<:Real}, Nothing}
end
function SVariance(; settings::RMSettings = RMSettings(),
                   formulation::VarianceFormulation = SOC(),
                   target::Union{<:Real, <:AbstractVector{<:Real}, Nothing} = nothing,
                   w::Union{<:AbstractWeights, Nothing} = nothing,
                   mu::Union{<:AbstractVector{<:Real}, Nothing} = nothing)
    return SVariance(settings, formulation, target, w, mu)
end
function (svariance::SVariance)(X::AbstractMatrix, w::AbstractVector, fees::Real = 0.0)
    x = X * w .- fees
    T = length(x)
    mu = calc_target_ret_mu(x, w, svariance)
    val = x .- mu
    val = val[val .<= zero(eltype(val))]
    return dot(val, val) / (T - 1)
end

"""
```
abstract type WorstCaseSet end
```

Abstract type for subtyping worst case mean variance set types.
"""
abstract type WorstCaseSet end

"""
"""
abstract type WCSetMuSigma <: WorstCaseSet end

"""
"""
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

"""
    mutable struct WCVariance{T1 <: Real} <: RiskMeasureSigma

Measures and computes the portfolio Variance with uncertainty sets.

  - Only has an effect on optimisations which take risk measures and use [`JuMP`](https://github.com/jump-dev/JuMP.jl) models.
  - It is not affected by [`VarianceFormulation`](@ref).
  - It otherwise acts the same as the [`Variance`](@ref) except for the fact that it always gets its value from the `cov` property of the [`Portfolio`](@ref) instance.

```math
\\begin{align}
\\mathrm{Variance}(\\bm{w},\\, \\mathbf{\\Sigma}) &= \\bm{w}^\\intercal \\, \\mathbf{\\Sigma}\\, \\bm{w}\\,.
\\end{align}
```

Where:

  - ``\\bm{w}``: is the `N×1` vector of asset weights.
  - ``\\mathbf{\\Sigma}``: is the `N×N` asset covariance matrix.

See also: [`RiskMeasureSigma`](@ref), [`RMSettings`](@ref), [`WCType`](@ref), [`WCSetMuSigma`](@ref), [`Box`](@ref), [`Ellipse`](@ref), [`Variance`](@ref), [`VarianceFormulation`](@ref), [`Portfolio`](@ref), [`wc_statistics!`](@ref), [`calc_risk`](@ref), [`optimise!`](@ref), [`set_rm`](@ref).

# Keyword Arguments

  - `settings::RMSettings = RMSettings()`: risk measure configuration settings.

  - `wc_set::WCSetMuSigma = Box()`: uncertainty set apply to the variance in the optimisation.

      + If `isa(wc_set, Box)`: uses `cov_l` and `cov_u`, either from the risk measure, or their corresponding instance of [`Portfolio`](@ref).
      + If `isa(wc_set, Ellipse)`: uses `cov_sigma` and `k_sigma`, either from the risk measure, or their corresponding instance of [`Portfolio`](@ref).
  - `sigma::Union{<:AbstractMatrix, Nothing} = nothing`: (optional) `N×N` covariance matrix.

      + If `nothing`: takes its value from the `cov` property of the instance [`Portfolio`](@ref).
  - `cov_l::Union{AbstractMatrix{<:Real}, Nothing}` = nothing: `N×N` lower percentile of the covariance matrix of the [`Box`](@ref) uncertainty set.

      + If `nothing`: takes its value from the `cov_l` property of the instance of [`Portfolio`](@ref).
  - `cov_u::Union{AbstractMatrix{<:Real}, Nothing}` = nothing: `N×N` upper percentile of the covariance matrix of the [`Box`](@ref) uncertainty set.

      + If `nothing`: takes its value from the `cov_u` property of the instance of [`Portfolio`](@ref).
  - `cov_sigma::Union{AbstractMatrix{<:Real}, Nothing}` = nothing: `N²×N²` covariance matrix of the estimation errors of the covariance of the [`Ellipse`](@ref) uncertainty set.

      + If `nothing`: takes its value from the `cov_sigma` property of the instance of [`Portfolio`](@ref).
  - `k_sigma::T1 = Inf`: square root of the size of the [`Ellipse`](@ref) uncertainty set.

      + If `nothing`: takes its value from the `k_sigma` property of the instance of [`Portfolio`](@ref).

# Behaviour in optimisations which take risk measures and use [`JuMP`](https://github.com/jump-dev/JuMP.jl) models

  - Uses [`PSDCone`](https://jump.dev/JuMP.jl/stable/tutorials/conic/tips_and_tricks/#Positive-Semidefinite-Cone) constraints.
  - The WCVariance risk is defined as an [`AffExpr`](https://jump.dev/JuMP.jl/stable/api/JuMP/#AffExpr) with the key, `:wcvariance_risk`.
  - If it exists, the upper bound is defined via the portfolio variance with the key, `:wcvariance_risk_ub`.

# Functor

  - `(variance::WCVariance)(w::AbstractVector)`: computes the WCVariance (which is equal to the variance) of an `N×1` vector of asset weights.

# Examples
"""
mutable struct WCVariance{T1 <: Real} <: RiskMeasureSigma
    settings::RMSettings
    wc_set::WCSetMuSigma
    sigma::Union{<:AbstractMatrix{<:Real}, Nothing}
    cov_l::Union{<:AbstractMatrix{<:Real}, Nothing}
    cov_u::Union{<:AbstractMatrix{<:Real}, Nothing}
    cov_sigma::Union{<:AbstractMatrix{<:Real}, Nothing}
    k_sigma::T1
end
function WCVariance(; settings::RMSettings = RMSettings(), wc_set::WCSetMuSigma = Box(),
                    sigma::Union{<:AbstractMatrix{<:Real}, Nothing} = nothing,
                    cov_l::Union{<:AbstractMatrix{<:Real}, Nothing} = nothing,
                    cov_u::Union{<:AbstractMatrix{<:Real}, Nothing} = nothing,
                    cov_sigma::Union{AbstractMatrix{<:Real}, Nothing} = nothing,
                    k_sigma::Real = Inf)
    if !isnothing(sigma)
        @smart_assert(size(sigma, 1) == size(sigma, 2))
    end
    if !isnothing(cov_l)
        @smart_assert(size(cov_l, 1) == size(cov_l, 2))
    end
    if !isnothing(cov_u)
        @smart_assert(size(cov_u, 1) == size(cov_u, 2))
    end
    if !isnothing(cov_sigma)
        @smart_assert(size(cov_sigma, 1) == size(cov_sigma, 2))
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
function (wcvariance::WCVariance)(w::AbstractVector)
    return dot(w, wcvariance.sigma, w)
end

mutable struct Fees
    long::Union{<:Real, <:AbstractVector{<:Real}}
    short::Union{<:Real, <:AbstractVector{<:Real}}
    fixed_long::Union{<:Real, <:AbstractVector{<:Real}}
    fixed_short::Union{<:Real, <:AbstractVector{<:Real}}
    tol_kwargs::NamedTuple
end
function Fees(; long::Union{<:Real, <:AbstractVector{<:Real}} = 0.0,
              short::Union{<:Real, <:AbstractVector{<:Real}} = 0.0,
              fixed_long::Union{<:Real, <:AbstractVector{<:Real}} = 0.0,
              fixed_short::Union{<:Real, <:AbstractVector{<:Real}} = 0.0,
              tol_kwargs::NamedTuple = (; atol = 1e-8))
    @smart_assert(all(long .>= zero(long)))
    @smart_assert(all(short .<= zero(short)))
    @smart_assert(all(fixed_long .>= zero(fixed_long)))
    @smart_assert(all(fixed_short .<= zero(fixed_short)))
    return Fees(long, short, fixed_long, fixed_short, tol_kwargs)
end
function Base.setproperty!(fees::Fees, sym::Symbol, val)
    if sym ∈ (:long, :fixed_long)
        @smart_assert(all(val .>= zero(val)))
    elseif sym ∈ (:short, :fixed_short)
        @smart_assert(all(val .<= zero(val)))
    end
    return setfield!(fees, sym, val)
end
function Base.isequal(A::Fees, B::Fees)
    for property ∈ propertynames(A)
        prop_A = getproperty(A, property)
        prop_B = getproperty(B, property)
        if !isequal(prop_A, prop_B)
            return false
        end
    end
    return true
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
    fees::Fees
    rebalance::AbstractTR
end
function TrackWeight(; err::Real = 0.0,
                     w::AbstractVector{<:Real} = Vector{Float64}(undef, 0),
                     fees::Fees = Fees(), rebalance::AbstractTR = NoTR())
    return TrackWeight{typeof(err), typeof(w)}(err, w, fees, rebalance)
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

"""
"""
mutable struct TrackingRM <: RiskMeasure
    settings::RMSettings
    tr::Union{TrackWeight, TrackRet}
end
function TrackingRM(; settings::RMSettings = RMSettings(),
                    tr::Union{TrackWeight, TrackRet} = TrackRet(;))
    return TrackingRM(settings, tr)
end
function (trackingRM::TrackingRM)(X::AbstractMatrix, w::AbstractVector, fees::Real = 0.0)
    T = size(X, 1)
    tr = trackingRM.tr
    benchmark = tracking_error_benchmark(tr, X)
    return norm(X * w - benchmark .- fees) / sqrt(T - 1)
end

"""
"""
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

```math
\\begin{align}
\\mathrm{CVaRRG}(\\bm{X},\\, \\alpha,\\,\\beta) &= \\mathrm{CVaR}(\\bm{X},\\, \\alpha) - \\mathrm{CVaR}(-\\bm{X},\\, \\beta)
\\end{align}
```

Where:

  - ``\\mathrm{CVaR}(\\bm{X},\\, \\alpha)``: is the Conditional Value at Risk as defined in [`CVaR`](@ref).
  - ``\\bm{X}``: is the `T×1` vector of portfolio returns.
  - ``\\alpha``: is the significance level of losses.
  - ``\\beta``: is the significance level of gains.

# Description

Defines the Value at Risk.

  - Measures lower bound of the losses in the worst `alpha %` of cases.
  - ``\\mathrm{VaR}(\\bm{X},\\, \\alpha) \\leq \\mathrm{CVaR}(\\bm{X},\\, \\alpha) \\leq \\mathrm{EVaR}(\\bm{X},\\, \\alpha) \\leq \\mathrm{RLVaR}(\\bm{X},\\, \\alpha,\\, \\kappa) \\leq \\mathrm{WR}(\\bm{X})``.

```math
\\begin{align}
\\mathrm{VaR}(\\bm{X},\\, \\alpha) &= -\\underset{t \\in (0,\\, T)}{\\inf} \\left\\{ X_{t} \\in \\mathbb{R} \\, | \\, F_{\\bm{X}}(X_{t}) > \\alpha \\right\\}\\,.
\\end{align}
```

See also: [`HCRiskMeasure`](@ref), [`HCRMSettings`](@ref), [`optimise!`](@ref), [`calc_risk(::VaR, ::AbstractVector)`](@ref), [`CVaR`](@ref), [`EVaR`](@ref), [`RLVaR`](@ref) [`WR`](@ref), [`DaR`](@ref), [`DaR_r`](@ref), [`CDaR`](@ref), [`CDaR_r`](@ref), [`EDaR`](@ref), [`EDaR_r`](@ref), [`RLDaR`](@ref), [`RLDaR_r`](@ref), [`MDD`](@ref), [`MDD_r`](@ref).

# Properties

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
    mutable struct VaRRG{T1 <: Real, T2 <: Real} <: HCRiskMeasure

Measures and computes the portfolio Value at Risk Range (VaRRG). It measures the difference between the threshold above or equal to which are the best `beta %` of portfolio returns. And the threshold below or equal to which are the worst `alpha %` of portfolio returns.

```math
\\begin{align}
\\mathrm{VaRRG}(\\bm{X},\\, \\alpha,\\,\\beta) &= \\mathrm{VaR}(\\bm{X},\\, \\alpha) - \\mathrm{VaR}(-\\bm{X},\\, \\beta)
\\end{align}
```

Where:

  - ``\\mathrm{VaR}(\\bm{X},\\, \\alpha)``: is the Value at Risk as defined in [`VaR`](@ref).
  - ``\\bm{X}``: is the `T×1` vector of portfolio returns.
  - ``\\alpha``: is the significance level of losses.
  - ``\\beta``: is the significance level of gains.

See also: See also: [`RiskMeasure`](@ref), [`RMSettings`](@ref), [`Portfolio`](@ref), [`VaR`](@ref), [`calc_risk`](@ref), [`optimise!`](@ref), [`set_rm`](@ref).

# Keyword Arguments

  - `settings::RMSettings = RMSettings()`: configuration settings for the risk measure.
  - `alpha::T1 = 0.05`: significance level of losses, `alpha ∈ (0, 1)`.
  - `beta::T1 = 0.05`: significance level of gains, `beta ∈ (0, 1)`.

# Functor

  - `(varrg::VaRRG)(x::AbstractVector)`: computes the VaRRG of a `T×1` vector of portfolio returns `x`.

# Examples
"""
mutable struct VaRRG{T1 <: Real, T2 <: Real} <: HCRiskMeasure
    settings::HCRMSettings
    alpha::T1
    beta::T2
end
function VaRRG(; settings::HCRMSettings = HCRMSettings(), alpha::Real = 0.05,
               beta::Real = 0.05)
    @smart_assert(zero(alpha) < alpha < one(alpha))
    @smart_assert(zero(beta) < beta < one(beta))
    return VaRRG{typeof(alpha), typeof(beta)}(settings, alpha, beta)
end
function Base.setproperty!(obj::VaRRG, sym::Symbol, val)
    if sym ∈ (:alpha, :beta)
        @smart_assert(zero(val) < val < one(val))
    end
    return setfield!(obj, sym, val)
end
function (varrg::VaRRG)(x::AbstractVector)
    T = length(x)
    alpha = varrg.alpha
    beta = varrg.beta
    loss = -partialsort!(x, ceil(Int, alpha * T))
    gain = partialsort!(x, ceil(Int, beta * T); rev = true)
    return loss + gain
end

"""
    mutable struct DaR{T1 <: Real} <: HCRiskMeasure

# Description

Defines the Drawdown at Risk for uncompounded cumulative returns.

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

# Properties

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

Defines the Drawdown at Risk for compounded cumulative returns.

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

# Properties

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

# Properties

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
    val = zero(eltype(x))
    peak = -Inf
    for i ∈ cs
        if i > peak
            peak = i
        end
        dd = one(eltype(x)) - i / peak
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

# Properties

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
    val = zero(eltype(x))
    peak = -Inf
    if isnothing(w)
        for i ∈ cs
            if i > peak
                peak = i
            end
            dd = one(eltype(x)) - i / peak
            if dd > 0
                val += dd
            end
        end
        popfirst!(x)
        return val / T
    else
        @smart_assert(length(w) == T)
        for (idx, i) ∈ pairs(cs)
            if i > peak
                peak = i
            end
            dd = one(eltype(x)) - i / peak
            if dd > 0
                wi = isone(idx) ? 1 : w[idx - 1]
                val += dd * wi
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

# Properties

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
    sum_var = zero(eltype(x))
    for i ∈ 1:(idx - 1)
        sum_var += dd[i] + var
    end
    return var - sum_var / aT
end

"""
    mutable struct UCI_r <: HCRiskMeasure

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

# Properties

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
    val = zero(eltype(x))
    peak = -Inf
    for i ∈ cs
        if i > peak
            peak = i
        end
        dd = one(eltype(x)) - i / peak
        if dd > 0
            val += dd^2
        end
    end
    return sqrt(val / T)
end

"""
    mutable struct EDaR_r{T1 <: Real} <: HCRiskMeasureSolvers

# Description

Entropic Drawdown at Risk risk measure for compounded cumulative returns.

  - Uses [`ExponentialCone`](https://jump.dev/JuMP.jl/stable/tutorials/conic/tips_and_tricks/#Exponential-Cone) constraints.
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

# Properties

  - `settings::HCRMSettings = HCRMSettings()`: configuration settings for the risk measure.
  - `alpha::T1 = 0.05`: significance level, `alpha ∈ (0, 1)`.
  - `solvers::Union{Nothing, PortOptSolver, <:AbstractVector{PortOptSolver}}`: optional JuMP-compatible solvers for exponential cone problems.

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
mutable struct EDaR_r{T1 <: Real} <: HCRiskMeasureSolvers
    settings::HCRMSettings
    alpha::T1
    solvers::Union{Nothing, PortOptSolver, <:AbstractVector{PortOptSolver}}
end
function EDaR_r(; settings::HCRMSettings = HCRMSettings(), alpha::Real = 0.05,
                solvers::Union{Nothing, PortOptSolver, <:AbstractVector{PortOptSolver}} = nothing)
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
    mutable struct RLDaR_r{T1 <: Real, T2 <: Real} <: HCRiskMeasureSolvers

# Description

Relativistic Drawdown at Risk risk measure for compounded cumulative returns. It is a generalisation of the [`EDaR_r`](@ref).

  - Uses [`PowerCone`](https://jump.dev/JuMP.jl/stable/tutorials/conic/tips_and_tricks/#PowerCone) constraints.
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

# Properties

  - `settings::RMSettings = RMSettings()`: configuration settings for the risk measure.
  - `alpha::T1 = 0.05`: significance level, `alpha ∈ (0, 1)`.
  - `kappa::T1 = 0.3`: significance level, `kappa ∈ (0, 1)`.
  - `solvers::Union{Nothing, PortOptSolver, <:AbstractVector{PortOptSolver}}`: optional JuMP-compatible solvers for 3D power cone problems.

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
mutable struct RLDaR_r{T1 <: Real, T2 <: Real} <: HCRiskMeasureSolvers
    settings::HCRMSettings
    alpha::T1
    kappa::T2
    solvers::Union{Nothing, PortOptSolver, <:AbstractVector{PortOptSolver}}
end
function RLDaR_r(; settings::HCRMSettings = HCRMSettings(), alpha::Real = 0.05, kappa = 0.3,
                 solvers::Union{Nothing, PortOptSolver, <:AbstractVector{PortOptSolver}} = nothing)
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

# Properties

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

#! Generalise this like the TCM
"""
"""
mutable struct TCM <: NoOptRiskMeasureMu
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

"""
"""
mutable struct TLPM <: HCRiskMeasureTarget
    settings::RMSettings
    target::Union{<:Real, <:AbstractVector{<:Real}, Nothing}
    w::Union{<:AbstractWeights, Nothing}
    mu::Union{<:AbstractVector{<:Real}, Nothing}
end
function TLPM(; settings::RMSettings = RMSettings(),
              target::Union{<:Real, <:AbstractVector{<:Real}, Nothing} = 0.0,
              w::Union{<:AbstractWeights, Nothing} = nothing,
              mu::Union{<:AbstractVector{<:Real}, Nothing} = nothing)
    return TLPM(settings, target, w, mu)
end
function (tlpm::TLPM)(X::AbstractMatrix, w::AbstractVector, fees::Real = 0.0)
    x = X * w .- fees
    T = length(x)
    target = calc_target_ret_mu(x, w, tlpm)
    val = x .- target
    val = val[val .<= zero(eltype(val))] .^ 3
    return -sum(val) / T
end

#! Generalise this like the FTLPM
"""
"""
mutable struct FTCM <: HCRiskMeasureMu
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

"""
"""
mutable struct FTLPM <: HCRiskMeasureTarget
    settings::RMSettings
    target::Union{<:Real, <:AbstractVector{<:Real}, Nothing}
    w::Union{<:AbstractWeights, Nothing}
    mu::Union{<:AbstractVector{<:Real}, Nothing}
end
function FTLPM(; settings::RMSettings = RMSettings(),
               target::Union{<:Real, <:AbstractVector{<:Real}, Nothing} = 0.0,
               w::Union{<:AbstractWeights, Nothing} = nothing,
               mu::Union{<:AbstractVector{<:Real}, Nothing} = nothing)
    return FTLPM(settings, target, w, mu)
end
function (ftlpm::FTLPM)(X::AbstractMatrix, w::AbstractVector, fees::Real = 0.0)
    x = X * w .- fees
    T = length(x)
    target = calc_target_ret_mu(x, w, ftlpm)
    val = x .- target
    val = val[val .<= zero(eltype(val))] .^ 4
    return sum(val) / T
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

mutable struct SSkewness{T1 <: Real} <: HCRiskMeasure
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

"""
    mutable struct Kurtosis <: HCRiskMeasure
"""
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
"""
    mutable struct SKurtosis{T1 <: Real} <: HCRiskMeasure
"""
mutable struct SKurtosis{T1 <: Real} <: HCRiskMeasure
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

"""
    const RMSolvers = Union{RiskMeasureSolvers, HCRiskMeasureSolvers}

Risk measures that use `solvers`.

See also: [`RiskMeasureSolvers`](@ref), [`HCRiskMeasureSolvers`](@ref).
"""
const RMSolvers = Union{RiskMeasureSolvers, HCRiskMeasureSolvers}

"""
    const RMSigma = Union{RiskMeasureSigma}

Risk measures that use `sigma`.

See also: [`RiskMeasureSigma`](@ref).
"""
const RMSigma = Union{RiskMeasureSigma}

"""
    const RMSkew = Union{RiskMeasureSkew}

Risk measures that use `skew` and `V`.

See also: [`RiskMeasureSkew`](@ref).
"""
const RMSkew = Union{RiskMeasureSkew}

"""
    const RMOWA = Union{RiskMeasureOWA}

Risk measures that use Ordered Weight Array formulations.

See also: [`RiskMeasureOWA`](@ref).
"""
const RMOWA = Union{RiskMeasureOWA}

"""
    const RMMu = Union{RiskMeasureMu, HCRiskMeasureMu, NoOptRiskMeasureMu}

Risk measures that use `mu` and `w`.

See also: [`RiskMeasureMu`](@ref), [`HCRiskMeasureMu`](@ref), [`NoOptRiskMeasureMu`](@ref), [`calc_ret_mu`](@ref).
"""
const RMMu = Union{RiskMeasureMu, HCRiskMeasureMu, NoOptRiskMeasureMu}

"""
    const RMTarget = Union{RiskMeasureTarget, HCRiskMeasureTarget}

Risk measures that use `target`, `mu` and `w`.

See also: [`RiskMeasureTarget`](@ref), [`HCRiskMeasureTarget`](@ref), [`calc_target_ret_mu`](@ref).
"""
const RMTarget = Union{RiskMeasureTarget, HCRiskMeasureTarget}

"""
    calc_ret_mu(x::AbstractVector, w::AbstractVector, rm::RMMu)

Computes the mean portfolio return for `rm`.

  - If `isnothing(rm.mu) || isempty(rm.mu)`, computes the mean return from `x`.

      + If `isnothing(rm.w)`: computes the unweighted mean.
      + Else: computes the weighted mean.

  - Else: computes the mean return as `dot(rm.mu, w)`.

See also: [`RMMu`](@ref).

# Positional Arguments

  - `x`: `T×1` vector of portfolio returns.
  - `w`: `N×1` vector of asset weights.
  - `rm`: [`RMMu`](@ref) risk measure.

# Returns

  - `mu::Real`: portfolio mean return.
"""
function calc_ret_mu(x::AbstractVector, w::AbstractVector, rm::RMMu)
    mu = rm.mu
    return mu = if isnothing(mu) || isempty(mu)
        wi = rm.w
        isnothing(wi) ? mean(x) : mean(x, wi)
    else
        dot(mu, w)
    end
end

"""
    calc_target_ret_mu(x::AbstractVector, w::AbstractVector, rm::RMTarget)

Computes the minimum acceptable portfolio return target for `rm`. Only returns equal to or below this value are accounted for in the calculation.

  - If `isnothing(rm.target) || isa(rm.target, AbstractVector) && isempty(rm.target)`, computes the mean acceptable return from `x` via [`calc_ret_mu`](@ref).
  - Else: returns `rm.target`.

See also: [`RMTarget`](@ref).

# Positional Arguments

  - `x`: `T×1` vector of portfolio returns.
  - `w`: `N×1` vector of asset weights.
  - `rm`: [`RMTarget`](@ref) risk measure.

# Returns

  - `target::Real`: minimum acceptable return target.
"""
function calc_target_ret_mu(x::AbstractVector, w::AbstractVector, rm::RMTarget)
    target = rm.target
    if isnothing(target) || isa(target, AbstractVector) && isempty(target)
        target = calc_ret_mu(x, w, rm)
    end
    return target
end

export RiskMeasure, HCRiskMeasure, NoOptRiskMeasure, RMSettings, HCRMSettings, Quad, SOC,
       RSOC, SD, MAD, SSD, FLPM, WR, CVaR, EVaR, EVaRRG, RLVaR, RLVaRRG, MDD, ADD, CDaR,
       UCI, EDaR, RLDaR, Kurt, SKurt, RG, CVaRRG, GMD, TG, TGRG, OWA, BDVariance, NQSkew,
       NQSSkew, NSkew, NSSkew, Variance, SVariance, VaR, VaRRG, DaR, DaR_r, MDD_r, ADD_r,
       CDaR_r, UCI_r, EDaR_r, RLDaR_r, Equal, BDVAbsVal, BDVIneq, WCVariance, DRCVaR, Box,
       Ellipse, NoWC, TrackingRM, TurnoverRM, NoTracking, TrackWeight, TrackRet, NoTR, TR,
       Kurtosis, SKurtosis, OWAApprox, OWAExact, RiskMeasureSigma, RiskMeasureMu,
       HCRiskMeasureMu, NoOptRiskMeasureMu, RiskMeasureTarget, HCRiskMeasureTarget,
       RiskMeasureSolvers, HCRiskMeasureSolvers, RiskMeasureOWA, RiskMeasureSkew, TCM, TLPM,
       FTCM, FTLPM, PortOptSolver, RMSolvers, RMSigma, RMSkew, RMOWA, RMMu, RMTarget, Fees
