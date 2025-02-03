# Copywrite (c) 2025
# Author: Daniel Celis Garza <daniel.celis.garza@gmail.com>
# SPDX-License-Identifier: MIT

"""
```
abstract type OWATypes end
```

Abstract type for subtyping Ordered Weight Array (OWA) types for computing the weights used to combine L-moments higher than 2 [OWAL](@cite) in [`owa_l_moment_crm`](@ref).
"""
abstract type OWATypes end

"""
    abstract type OWAJTypes <: OWATypes end
"""
abstract type OWAJTypes <: OWATypes end

"""
```
@kwdef mutable struct CRRA{T1 <: Real} <: OWATypes
    g::T1 = 0.5
end
```

Normalised Constant Relative Risk Aversion Coefficients.

# Parameters

  - `g`: Risk aversion coefficient.
"""
mutable struct CRRA{T1 <: Real} <: OWATypes
    g::T1
end
function CRRA(; g::Real = 0.5)
    @smart_assert(zero(g) < g < one(g))
    return CRRA{typeof(g)}(g)
end
function Base.setproperty!(obj::CRRA, sym::Symbol, val)
    if sym == :g
        @smart_assert(zero(val) < val < one(val))
    end
    return setfield!(obj, sym, val)
end

"""
```
@kwdef mutable struct MaxEntropy{T1 <: Real, T2 <: AbstractDict} <: OWATypes
    max_phi::T1 = 0.5
    solvers::T2 = Dict()
end
```

Maximum Entropy. Solver must support `MOI.RelativeEntropyCone` and `MOI.NormOneCone`.

# Parameters

  - `max_phi`: Maximum weight constraint of the L-moments.
"""
mutable struct MaxEntropy{T1 <: Real, T2 <: Real, T3 <: Real,
                          T4 <: Union{PortOptSolver, <:AbstractVector{PortOptSolver}}} <:
               OWAJTypes
    max_phi::T1
    scale_constr::T2
    scale_obj::T3
    solvers::T4
end
function MaxEntropy(; max_phi::Real = 0.5, scale_constr::Real = 1.0, scale_obj::Real = 1.0,
                    solvers::Union{PortOptSolver, <:AbstractVector{PortOptSolver}} = PortOptSolver())
    @smart_assert(zero(max_phi) < max_phi < one(max_phi))
    @smart_assert(scale_constr > zero(scale_constr))
    @smart_assert(scale_obj > zero(scale_obj))
    return MaxEntropy{typeof(max_phi), typeof(scale_constr), typeof(scale_obj),
                      Union{PortOptSolver, <:AbstractVector{PortOptSolver}}}(max_phi,
                                                                             scale_constr,
                                                                             scale_obj,
                                                                             solvers)
end

"""
```
@kwdef mutable struct MinSumSq{T1 <: Real} <: OWAJTypes
    max_phi::T1 = 0.5
end
```

Minimum Sum of Squares. Solver must support `MOI.SecondOrderCone`.

# Parameters

  - `max_phi`: Maximum weight constraint of the L-moments.
"""
mutable struct MinSumSq{T1 <: Real, T2 <: Real, T3 <: Real,
                        T4 <: Union{PortOptSolver, <:AbstractVector{PortOptSolver}}} <:
               OWAJTypes
    max_phi::T1
    scale_constr::T2
    scale_obj::T3
    solvers::T4
end
function MinSumSq(; max_phi::Real = 0.5, scale_constr::Real = 1.0, scale_obj::Real = 1.0,
                  solvers::Union{PortOptSolver, <:AbstractVector{PortOptSolver}} = PortOptSolver())
    @smart_assert(zero(max_phi) < max_phi < one(max_phi))
    @smart_assert(scale_constr > zero(scale_constr))
    @smart_assert(scale_obj > zero(scale_obj))
    return MinSumSq{typeof(max_phi), typeof(scale_constr), typeof(scale_obj),
                    Union{PortOptSolver, <:AbstractVector{PortOptSolver}}}(max_phi,
                                                                           scale_constr,
                                                                           scale_obj,
                                                                           solvers)
end

"""
```
@kwdef mutable struct MinSqDist{T1 <: Real} <: OWAJTypes
    max_phi::T1 = 0.5
end
```

Minimum Square Distance. Solver must support `MOI.SecondOrderCone`.

# Parameters

  - `max_phi`: Maximum weight constraint of the L-moments.
"""
mutable struct MinSqDist{T1 <: Real, T2 <: Real, T3 <: Real,
                         T4 <: Union{PortOptSolver, <:AbstractVector{PortOptSolver}}} <:
               OWAJTypes
    max_phi::T1
    scale_constr::T2
    scale_obj::T3
    solvers::T4
end
function MinSqDist(; max_phi::Real = 0.5, scale_constr::Real = 1.0, scale_obj::Real = 1.0,
                   solvers::Union{PortOptSolver, <:AbstractVector{PortOptSolver}} = PortOptSolver())
    @smart_assert(zero(max_phi) < max_phi < one(max_phi))
    @smart_assert(scale_constr > zero(scale_constr))
    @smart_assert(scale_obj > zero(scale_obj))
    return MinSqDist{typeof(max_phi), typeof(scale_constr), typeof(scale_obj),
                     Union{PortOptSolver, <:AbstractVector{PortOptSolver}}}(max_phi,
                                                                            scale_constr,
                                                                            scale_obj,
                                                                            solvers)
end
function Base.setproperty!(obj::OWAJTypes, sym::Symbol, val)
    if sym == :max_phi
        @smart_assert(zero(val) < val < one(val))
    elseif sym in (:scale_constr, :scale_obj)
        @smart_assert(val > zero(val))
    end
    return setfield!(obj, sym, val)
end

export CRRA, MaxEntropy, MinSumSq, MinSqDist
