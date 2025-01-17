"""
    abstract type AbstractOptimType end

Abstract type for the different types of optimisations.

See also: [`OptimType`](@ref) and [`HCOptimType`](@ref).
"""
abstract type AbstractOptimType end

"""
    abstract type OptimType <: AbstractOptimType end

Abstract type for optimisations that are not hierarchical.

See also: [`AbstractOptimType`](@ref), [`HCOptimType`](@ref), [`Trad`](@ref), [`RB`](@ref), [`RRB`](@ref), [`NOC`](@ref).
"""
abstract type OptimType <: AbstractOptimType end

"""
    abstract type HCOptimType <: AbstractOptimType end

Abstract type for hierarchical optimisations.

See also: [`AbstractOptimType`](@ref), [`OptimType`](@ref), [`HRP`](@ref), [`SchurHRP`](@ref), [`HERC`](@ref), [`NCO`](@ref).
"""
abstract type HCOptimType <: AbstractOptimType end

"""
    abstract type AbstractScalarisation end

Abstract type for scalarisation functions used when simultaneously optimising for multiple risk measures.

See also: [`ScalarSum`](@ref), [`ScalarMax`](@ref), [`ScalarLogSumExp`](@ref).
"""
abstract type AbstractScalarisation end

"""
    struct ScalarSum <: AbstractScalarisation end

Scalarises the risk measures as a weighted sum.

See also: [`AbstractScalarisation`](@ref).

```math
\\begin{align}
    r &= \\bm{r} \\cdot \\bm{w}
\\end{align}
```

Where:

  - ``r`` is the scalarised risk.
  - ``\\bm{r}`` is the vector of risk measures.
  - ``\\bm{w}`` is the corresponding vector of risk measure weights
  - ``\\cdot`` is the dot product.

# Examples

```julia
scalariser = ScalarSum()
```
"""
struct ScalarSum <: AbstractScalarisation end

"""
    struct ScalarMax <: AbstractScalarisation end

Scalarises the risk measures by taking the maximum of them.

See also: [`AbstractScalarisation`](@ref).

```math
\\begin{align}
    r &= \\max \\left( \\bm{r} \\odot \\bm{w} \\right)
\\end{align}
```

Where:

  - ``r`` is the scalarised risk.
  - ``\\bm{r}`` is the vector of risk measures.
  - ``\\bm{w}`` is the corresponding vector of risk measure weights.
  - ``\\odot`` is the Hadamard (element-wise) multiplication.

# Examples

```julia
scalariser = ScalarMax()
```
"""
struct ScalarMax <: AbstractScalarisation end

"""
    mutable struct ScalarLogSumExp{T1 <: Real} <: AbstractScalarisation end

Scalarises the risk measures as the log_sum_exp of the weighted risk measures.

See also: [`AbstractScalarisation`](@ref).

```math
\\begin{align}
    r &= \\frac{1}{\\gamma} \\log \\left( \\sum_{i = 1}^{N} \\exp(\\gamma r_i w_i) \\right)
\\end{align}
```

Where:

  - ``r`` is the scalarised risk.
  - ``r_i`` is the ``i``-th risk measure.
  - ``w_i`` is the weight of the ``i``-th risk measure.
  - ``\\gamma`` is a parameter that controls the shape of the scalarisation.

# Keyword Parameters

  - `gamma::Real = 1.0`: `gamma > 0`. As `gamma → 0`, the scalarisation approaches [`ScalarSum`](@ref). As `gamma → Inf`, the scalarisation approaches [`ScalarMax`](@ref).

# Examples

```julia
# Default constructor.
scalariser = ScalarLogSumExp()

# Approximate ScalarSum()
scalariser = ScalarLogSumExp(; gamma = 1e-6)

# Approximate ScalarMax()
scalariser = ScalarLogSumExp(; gamma = 1e6)
```
"""
mutable struct ScalarLogSumExp{T1 <: Real} <: AbstractScalarisation
    gamma::T1
end
function ScalarLogSumExp(; gamma::Real = 1.0)
    @smart_assert(zero(gamma) <= gamma)
    return ScalarLogSumExp{typeof(gamma)}(gamma)
end
function Base.setproperty!(obj::ScalarLogSumExp, sym::Symbol, val)
    if sym == :gamma
        @smart_assert(zero(val) <= val)
    end
    return setfield!(obj, sym, val)
end

"""
    mutable struct Trad{T1, T2} <: OptimType

Traditional optimisation type.

See also: [`OptimType`](@ref), [`RiskMeasure`](@ref), [`ObjectiveFunction`](@ref), [`RetType`](@ref), [`PortClass`](@ref), [`CustomConstraint`](@ref), [`CustomObjective`](@ref), [`AbstractScalarisation`](@ref).

# Keyword Parameters

  - `rm::Union{AbstractVector, <:RiskMeasure} = Variance()`: The risk measure(s) to be used.

      + If multiple instances of the same risk measure are used, they must be grouped in a single vector wrapped in another vector, see examples.

  - `obj::ObjectiveFunction = MinRisk()`: The objective function to be used.
  - `kelly::RetType = NoKelly()`: The Kelly criterion to be used.
  - `class::PortClass = Classic()`: The portfolio class to be used.
  - `w_ini::T1 = Vector{Float64}(undef, 0) where T1 <: AbstractVector`: The initial weights for the optimisation.

      + Irrelevant if the solver does not support them.
  - `custom_constr::CustomConstraint = NoCustomConstraint()`: Add custom constraints to the optimisation problem.
  - `custom_obj::CustomObjective = NoCustomObjective()`: Add custom terms to the objective function.
  - `ohf::T2 = 1.0 where T2 <: Real`: The optimal homogenisation factor.

      + Only used when `obj == Sharpe()`, and if `iszero(ohf)`, [`optimise!`](@ref) sets its value in the [`JuMP`](https://github.com/jump-dev/JuMP.jl) model by using a heuristic.
  - `scalarisation::AbstractScalarisation = ScalarSum()`: The scalarisation function to be used.

      + Only relevant when multiple risk measures are used.
  - `str_names::Bool = false`: Whether to use string names in the [`JuMP`](https://github.com/jump-dev/JuMP.jl) model.

# Examples

```julia
# Default constructor.
opt_type = Trad()

# Single risk measure.
opt_type = Trad(; rm = SD())

# Multiple risk measures.
opt_type = Trad(; rm = [Variance(), CVaR(; alpha = 0.15)])

# Incorrect use of multiple risk measures of the same type.
# This will produce a JuMP registration error when optimise! is called.
opt_type = Trad(; rm = [CVaR(), CVaR(; alpha = 0.2)])

# Correct use of multiple risk measures of the same type.
opt_type = Trad(; rm = [[CVaR(), CVaR(; alpha = 0.2)]])

# Incorrect use of multiple risk measures, some of the same type.
# This will produce a JuMP registration error regarding the CVaR
# risk measure when optimise! is called.
opt_type = Trad(; rm = [MAD(), CVaR(), CVaR(; alpha = 0.2)])

# Correct use of multiple risk measures, some of the same type.
opt_type = Trad(; rm = [MAD(), [CVaR(), CVaR(; alpha = 0.2)]])
```
"""
mutable struct Trad{T1, T2} <: OptimType
    rm::Union{AbstractVector, <:RiskMeasure}
    obj::ObjectiveFunction
    kelly::RetType
    class::PortClass
    w_ini::T1
    custom_constr::CustomConstraint
    custom_obj::CustomObjective
    ohf::T2
    scalarisation::AbstractScalarisation
    str_names::Bool
end
function Trad(; rm::Union{AbstractVector, <:RiskMeasure} = Variance(),
              obj::ObjectiveFunction = MinRisk(), kelly::RetType = NoKelly(),
              class::PortClass = Classic(),
              w_ini::AbstractVector = Vector{Float64}(undef, 0),
              custom_constr::CustomConstraint = NoCustomConstraint(),
              custom_obj::CustomObjective = NoCustomObjective(), ohf::Real = 1.0,
              scalarisation::AbstractScalarisation = ScalarSum(), str_names::Bool = false)
    return Trad{typeof(w_ini), typeof(ohf)}(rm, obj, kelly, class, w_ini, custom_constr,
                                            custom_obj, ohf, scalarisation, str_names)
end

"""
    mutable struct RB{T1} <: OptimType

Risk budget optimisation type. Allows the user to specify a risk budget vector specifying the maximum risk contribution per asset or factor. The asset weights are then optimised to meet the risk budget per asset/factor as optimally as possible.

See also: [`OptimType`](@ref), [`RiskMeasure`](@ref), [`RetType`](@ref), [`PortClass`](@ref), [`CustomConstraint`](@ref), [`CustomObjective`](@ref), [`AbstractScalarisation`](@ref).

# Keyword Parameters

  - `rm::Union{AbstractVector, <:RiskMeasure} = Variance()`: The risk measure(s) to be used.

      + If multiple instances of the same risk measure are used, they must be grouped in a single vector wrapped in another vector, see examples.

  - `kelly::RetType = NoKelly()`: The Kelly criterion to be used.
  - `class::PortClass = Classic()`: The portfolio class to be used.
  - `w_ini::T1 = Vector{Float64}(undef, 0) where T1 <: AbstractVector`: The initial weights for the optimisation.

      + Irrelevant if the solver does not support them.
  - `custom_constr::CustomConstraint = NoCustomConstraint()`: Add custom constraints to the optimisation problem.
  - `custom_obj::CustomObjective = NoCustomObjective()`: Add custom terms to the objective function.
  - `scalarisation::AbstractScalarisation = ScalarSum()`: The scalarisation function to be used.

      + Only relevant when multiple risk measures are used.
  - `str_names::Bool = false`: Whether to use string names in the [`JuMP`](https://github.com/jump-dev/JuMP.jl) model.

# Examples

```julia
# Default constructor.
opt_type = RB()

# Single risk measure.
opt_type = RB(; rm = SD())

# Multiple risk measures.
opt_type = RB(; rm = [Variance(), CVaR(; alpha = 0.15)])

# Incorrect use of multiple risk measures of the same type.
# This will produce a JuMP registration error when optimise! is called.
opt_type = RB(; rm = [CVaR(), CVaR(; alpha = 0.2)])

# Correct use of multiple risk measures of the same type.
opt_type = RB(; rm = [[CVaR(), CVaR(; alpha = 0.2)]])

# Incorrect use of multiple risk measures, some of the same type.
# This will produce a JuMP registration error regarding the CVaR
# risk measure when optimise! is called.
opt_type = RB(; rm = [MAD(), CVaR(), CVaR(; alpha = 0.2)])

# Correct use of multiple risk measures, some of the same type.
opt_type = RB(; rm = [MAD(), [CVaR(), CVaR(; alpha = 0.2)]])
```
"""
mutable struct RB{T1} <: OptimType
    rm::Union{AbstractVector, <:RiskMeasure}
    kelly::RetType
    class::PortClass
    w_ini::T1
    custom_constr::CustomConstraint
    custom_obj::CustomObjective
    scalarisation::AbstractScalarisation
    str_names::Bool
end
function RB(; rm::Union{AbstractVector, <:RiskMeasure} = Variance(),
            kelly::RetType = NoKelly(), class::PortClass = Classic(),
            w_ini::AbstractVector = Vector{Float64}(undef, 0),
            custom_constr::CustomConstraint = NoCustomConstraint(),
            custom_obj::CustomObjective = NoCustomObjective(),
            scalarisation::AbstractScalarisation = ScalarSum(), str_names::Bool = false)
    return RB{typeof(w_ini)}(rm, kelly, class, w_ini, custom_constr, custom_obj,
                             scalarisation, str_names)
end

"""
    abstract type RRBVersion end

Abstract type for the different versions of the relaxed risk budget optimisation.

See also: [`RRB`](@ref), [`BasicRRB`](@ref), [`RegRRB`](@ref), [`RegPenRRB`](@ref).
"""
abstract type RRBVersion end

"""
struct BasicRRB <: RRBVersion end

Basic relaxed risk budget optimisation version.

See also: [`RRBVersion`](@ref), [`RRB`](@ref).
"""
struct BasicRRB <: RRBVersion end

"""
    struct RegRRB <: RRBVersion end

Relaxed risk budget optimisation version with regularisation.

See also: [`RRBVersion`](@ref), [`RRB`](@ref).
"""
struct RegRRB <: RRBVersion end

"""
    mutable struct RegPenRRB{T1} <: RRBVersion

Relaxed risk budget optimisation version with regularisation and penalty.

See also: [`RRBVersion`](@ref), [`RRB`](@ref).

# Keyword Parameters

  - `penalty::Real = 1.0`: The penalty to be used.

# Examples

```julia
# Default constructor.
rrb_ver = RegPenRRB()

# Custom penalty.
rrb_ver = RegPenRRB(; penalty = 0.5)
```
"""
mutable struct RegPenRRB{T1} <: RRBVersion
    penalty::T1
end
function RegPenRRB(; penalty::Real = 1.0)
    return RegPenRRB(penalty)
end

"""
    mutable struct RRB{T1} <: OptimType

The relaxed risk budget optimisation only applies to the variance risk measure.

See also: [`OptimType`](@ref), [`RRBVersion`](@ref), [`RetType`](@ref), [`PortClass`](@ref), [`CustomConstraint`](@ref), [`CustomObjective`](@ref), [`AbstractScalarisation`](@ref).

# Keyword Parameters

  - `version::RRBVersion = BasicRRB()`: Relaxed risk budget optimisation version.

  - `kelly::RetType = NoKelly()`: The Kelly criterion to be used.
  - `class::PortClass = Classic()`: The portfolio class to be used.
  - `w_ini::T1 = Vector{Float64}(undef, 0) where T1 <: AbstractVector`: The initial weights for the optimisation.

      + Irrelevant if the solver does not support them.

      + `custom_constr::CustomConstraint = NoCustomConstraint()`: Add custom constraints to the optimisation problem.
      + `custom_obj::CustomObjective = NoCustomObjective()`: Add custom terms to the objective function.
  - `str_names::Bool = false`: Whether to use string names in the [`JuMP`](https://github.com/jump-dev/JuMP.jl) model.
"""
mutable struct RRB{T1} <: OptimType
    version::RRBVersion
    kelly::RetType
    class::PortClass
    w_ini::T1
    custom_constr::CustomConstraint
    custom_obj::CustomObjective
    str_names::Bool
end
function RRB(; version::RRBVersion = BasicRRB(), kelly::RetType = NoKelly(),
             class::PortClass = Classic(),
             w_ini::AbstractVector = Vector{Float64}(undef, 0),
             custom_constr::CustomConstraint = NoCustomConstraint(),
             custom_obj::CustomObjective = NoCustomObjective(), str_names::Bool = false,)
    return RRB{typeof(w_ini)}(version, kelly, class, w_ini, custom_constr, custom_obj,
                              str_names)
end
function Base.getproperty(obj::RRB, sym::Symbol)
    return if sym == :rm
        nothing
    else
        getfield(obj, sym)
    end
end

"""
    mutable struct NOC{T1, T2, T3, T4, T5, T6, T7, T8} <: OptimType

Near optimal centering optimisation type. This type of optimisation defines a near-optimal convex region around a point in the efficient frontier, and finds the portfolio which best fits the analytic centre of the region.

See also: [`OptimType`](@ref), [`RiskMeasure`](@ref), [`RetType`](@ref), [`PortClass`](@ref), [`CustomConstraint`](@ref), [`CustomObjective`](@ref), [`AbstractScalarisation`](@ref).

# Keyword Parameters

  - `flag::Bool = true`:

  - `bins::T1 = 20.0 where T1 <: Real`:
  - `w_opt::T2 = Vector{Float64}(undef, 0) where T2 <: AbstractVector{<:Real}`: Vector of weights of the efficient frontier portfolio.
  - `w_min::T3 = Vector{Float64}(undef, 0) where T3 <: AbstractVector{<:Real}`: Vector of weights of the minimal risk portfolio.
  - `w_max::T4 = Vector{Float64}(undef, 0) where T4 <: AbstractVector{<:Real}`: Vector of weights of the maxumal return portfolio.
  - `w_opt_ini::T6 = Vector{Float64}(undef, 0) where T6 <: AbstractVector{<:Real}`: The initial weights of the efficient frontier portfolio optimisation.

      + Irrelevant if the solver does not support them.
  - `w_min_ini::T6 = Vector{Float64}(undef, 0) where T6 <: AbstractVector{<:Real}`: The initial weights of the minimum risk optimisation.

      + Irrelevant if the solver does not support them.
  - `w_max_ini::T7 = Vector{Float64}(undef, 0) where T7 <: AbstractVector{<:Real}`: The initial weights of the maximum return optimisation.

      + Irrelevant if the solver does not support them.
  - `rm::Union{AbstractVector, <:RiskMeasure} = Variance()`: The risk measure(s) to be used.

      + If multiple instances of the same risk measure are used, they must be grouped in a single vector wrapped in another vector, see examples.
  - `obj::ObjectiveFunction = MinRisk()`:
  - `kelly::RetType = NoKelly()`:
  - `class::PortClass = Classic()`:
  - `w_ini::T8 = Vector{Float64}(undef, 0) where T8 <: AbstractVector{<:Real}`: The initial weights for the optimisation of the near optimal centering portfolio.

      + Irrelevant if the solver does not support them.
  - `custom_constr::CustomConstraint = NoCustomConstraint()`: Add custom constraints to the optimisation problem.
  - `custom_obj::CustomObjective = NoCustomObjective()`: Add custom terms to the objective function.
  - `ohf::T9 = 1.0 where T9 <: Real`:
  - `scalarisation::AbstractScalarisation = ScalarSum()`: The scalarisation function to be used.

      + Only relevant when multiple risk measures are used.
  - `str_names::Bool = false`: Whether to use string names in the [`JuMP`](https://github.com/jump-dev/JuMP.jl) model.
"""
mutable struct NOC{T1, T2, T3, T4, T5, T6, T7, T8, T9} <: OptimType
    flag::Bool
    bins::T1
    w_opt::T2
    w_min::T3
    w_max::T4
    w_opt_ini::T5
    w_min_ini::T6
    w_max_ini::T7
    rm::Union{AbstractVector, <:RiskMeasure}
    obj::ObjectiveFunction
    kelly::RetType
    class::PortClass
    w_ini::T8
    custom_constr::CustomConstraint
    custom_obj::CustomObjective
    ohf::T9
    scalarisation::AbstractScalarisation
    str_names::Bool
end
function NOC(; flag::Bool = true, bins::Real = 20.0,
             w_opt::AbstractVector{<:Real} = Vector{Float64}(undef, 0),
             w_min::AbstractVector{<:Real} = Vector{Float64}(undef, 0),
             w_max::AbstractVector{<:Real} = Vector{Float64}(undef, 0),
             w_opt_ini::AbstractVector{<:Real} = Vector{Float64}(undef, 0),
             w_min_ini::AbstractVector{<:Real} = Vector{Float64}(undef, 0),
             w_max_ini::AbstractVector{<:Real} = Vector{Float64}(undef, 0),
             rm::Union{AbstractVector, <:RiskMeasure} = Variance(),
             obj::ObjectiveFunction = MinRisk(), kelly::RetType = NoKelly(),
             class::PortClass = Classic(),
             w_ini::AbstractVector{<:Real} = Vector{Float64}(undef, 0),
             custom_constr::CustomConstraint = NoCustomConstraint(),
             custom_obj::CustomObjective = NoCustomObjective(), ohf::Real = 1.0,
             scalarisation::AbstractScalarisation = ScalarSum(), str_names::Bool = false)
    return NOC{typeof(bins), typeof(w_opt), typeof(w_min), typeof(w_max), typeof(w_opt_ini),
               typeof(w_min_ini), typeof(w_max_ini), typeof(w_ini), typeof(ohf)}(flag, bins,
                                                                                 w_opt,
                                                                                 w_min,
                                                                                 w_max,
                                                                                 w_opt_ini,
                                                                                 w_min_ini,
                                                                                 w_max_ini,
                                                                                 rm, obj,
                                                                                 kelly,
                                                                                 class,
                                                                                 w_ini,
                                                                                 custom_constr,
                                                                                 custom_obj,
                                                                                 ohf,
                                                                                 scalarisation,
                                                                                 str_names)
end

abstract type HCOptWeightFinaliser end
mutable struct HWF{T1} <: HCOptWeightFinaliser
    max_iter::T1
end
function HWF(; max_iter::Integer = 100)
    return HWF{typeof(max_iter)}(max_iter)
end
mutable struct JWF{T1 <: Integer} <: HCOptWeightFinaliser
    type::T1
end
function JWF(; type::Integer = 1)
    @smart_assert(type ∈ (1, 2, 3, 4))
    return JWF{typeof(type)}(type)
end
function Base.setproperty!(obj::JWF, sym::Symbol, val)
    if sym == :type
        @smart_assert(val ∈ (1, 2, 3, 4))
    end
    return setfield!(obj, sym, val)
end

"""
```
struct HRP <: HCOptimType end
```
"""
mutable struct HRP <: HCOptimType
    rm::Union{AbstractVector, <:AbstractRiskMeasure}
    class::PortClass
    scalarisation::AbstractScalarisation
    finaliser::HCOptWeightFinaliser
end
function HRP(; rm::Union{AbstractVector, <:AbstractRiskMeasure} = Variance(),
             class::PortClass = Classic(),
             scalarisation::AbstractScalarisation = ScalarSum(),
             finaliser::HCOptWeightFinaliser = HWF())
    return HRP(rm, class, scalarisation, finaliser)
end

mutable struct SchurParams{T1, T2, T3, T4}
    rm::RMSigma
    gamma::T1
    prop_coef::T2
    tol::T3
    max_iter::T4
end
function SchurParams(; rm::RMSigma = Variance(;), gamma::Real = 0.5, prop_coef::Real = 0.5,
                     tol::Real = 1e-2, max_iter::Integer = 10)
    @smart_assert(zero(gamma) <= gamma <= one(gamma))
    @smart_assert(zero(prop_coef) <= prop_coef <= one(prop_coef))
    @smart_assert(zero(tol) < tol)
    @smart_assert(zero(max_iter) < max_iter)
    return SchurParams{typeof(gamma), typeof(prop_coef), typeof(tol), typeof(max_iter)}(rm,
                                                                                        gamma,
                                                                                        prop_coef,
                                                                                        tol,
                                                                                        max_iter)
end
function Base.setproperty!(obj::SchurParams, sym::Symbol, val)
    if sym ∈ (:gamma, :prop_coef)
        @smart_assert(zero(val) <= val <= one(val))
    elseif sym ∈ (:tol, :max_iter)
        @smart_assert(zero(val) < val)
    end
    return setfield!(obj, sym, val)
end

"""
    mutable struct SchurHRP <: HCOptimType
"""
mutable struct SchurHRP <: HCOptimType
    params::Union{AbstractVector, <:SchurParams}
    class::PortClass
    finaliser::HCOptWeightFinaliser
end
function SchurHRP(; params::Union{AbstractVector, <:SchurParams} = SchurParams(),
                  class::PortClass = Classic(), finaliser::HCOptWeightFinaliser = HWF())
    return SchurHRP(params, class, finaliser)
end

"""
```
struct HERC <: HCOptimType end
```
"""
mutable struct HERC <: HCOptimType
    rm::Union{AbstractVector, <:AbstractRiskMeasure}
    rm_o::Union{AbstractVector, <:AbstractRiskMeasure}
    class::PortClass
    class_o::PortClass
    scalarisation::AbstractScalarisation
    scalarisation_o::AbstractScalarisation
    finaliser::HCOptWeightFinaliser
end
function HERC(; rm::Union{AbstractVector, <:AbstractRiskMeasure} = Variance(),
              rm_o::Union{AbstractVector, <:AbstractRiskMeasure} = rm,
              class::PortClass = Classic(), class_o::PortClass = class,
              scalarisation::AbstractScalarisation = ScalarSum(),
              scalarisation_o::AbstractScalarisation = scalarisation,
              finaliser::HCOptWeightFinaliser = HWF())
    return HERC(rm, rm_o, class, class_o, scalarisation, scalarisation_o, finaliser)
end

abstract type AbstractNCOModify end
struct NoNCOModify <: AbstractNCOModify end

mutable struct NCOArgs
    type::AbstractOptimType
    pre_modify::AbstractNCOModify
    post_modify::AbstractNCOModify
    port_kwargs::NamedTuple
    stats_kwargs::NamedTuple
    wc_kwargs::NamedTuple
    factor_kwargs::NamedTuple
    cluster_kwargs::NamedTuple
end
function NCOArgs(; type::AbstractOptimType = Trad(),
                 pre_modify::AbstractNCOModify = NoNCOModify(),
                 post_modify::AbstractNCOModify = NoNCOModify(),
                 port_kwargs::NamedTuple = (;), stats_kwargs::NamedTuple = (;),
                 wc_kwargs::NamedTuple = (;), factor_kwargs::NamedTuple = (;),
                 cluster_kwargs::NamedTuple = (;))
    return NCOArgs(type, pre_modify, post_modify, port_kwargs, stats_kwargs, wc_kwargs,
                   factor_kwargs, cluster_kwargs)
end
"""
```
mutable struct NCO <: HCOptimType
    internal::NCOArgs
    external::NCOArgs
    finaliser::HCOptWeightFinaliser
end
```
"""
mutable struct NCO <: HCOptimType
    internal::Union{AbstractVector{<:NCOArgs}, NCOArgs}
    external::NCOArgs
    finaliser::HCOptWeightFinaliser
end
function NCO(; internal::Union{AbstractVector{<:NCOArgs}, NCOArgs} = NCOArgs(;),
             external::NCOArgs = internal, finaliser::HCOptWeightFinaliser = HWF())
    return NCO(internal, external, finaliser)
end
function Base.getproperty(nco::NCO, sym::Symbol)
    if sym ∈
       (:rm, :obj, :kelly, :class, :scalarisation, :w_ini, :custom_constr, :custom_obj,
        :str_names)
        type = nco.internal.type
        isa(type, NCO) ? getproperty(type, sym) : getfield(type, sym)
    elseif sym ∈
           (:rm_o, :obj_o, :kelly_o, :class_o, :scalarisation_o, :w_ini_o, :custom_constr_o,
            :custom_obj_o, :str_names_o)
        type = nco.external.type
        if isa(type, NCO)
            getproperty(type, sym)
        else
            str_sym = string(sym)
            sym = contains(str_sym, "_o") ? Symbol(str_sym[1:(end - 2)]) : sym
            getfield(type, sym)
        end
    else
        getfield(nco, sym)
    end
end

for (op, name) ∈ zip((Trad, RB, RRB, NOC, HRP, HERC, NCO, SchurHRP),
                     ("Trad", "RB", "RRB", "NOC", "HRP", "HERC", "NCO", "SchurHRP"))
    eval(quote
             function Base.String(::$op)
                 return $name
             end
             function Base.Symbol(s::$op)
                 return Symbol($name)
             end
         end)
end

export Trad, RB, BasicRRB, RegRRB, RegPenRRB, RRB, NOC, HRP, HERC, NCO, NCOArgs, SchurHRP,
       SchurParams, HWF, JWF, ScalarSum, ScalarMax, ScalarLogSumExp
