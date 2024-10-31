"""
```
abstract type OWAMethods end
```

Abstract type for subtyping Ordered Weight Array (OWA) methods for computing the weights used to combine L-moments higher than 2 [OWAL](@cite) in [`owa_l_moment_crm`](@ref).
"""
abstract type OWAMethods end

"""
```
@kwdef mutable struct CRRA{T1 <: Real} <: OWAMethods
    g::T1 = 0.5
end
```

Normalised Constant Relative Risk Aversion Coefficients.

# Parameters

  - `g`: Risk aversion coefficient.
"""
mutable struct CRRA{T1 <: Real} <: OWAMethods
    g::T1
end
function CRRA(; g::Real = 0.5)
    @smart_assert(zero(g) < g < one(g))
    return CRRA{typeof(g)}(g)
end

"""
```
@kwdef mutable struct MaxEntropy{T1 <: Real, T2 <: AbstractDict} <: OWAMethods
    max_phi::T1 = 0.5
    solvers::T2 = Dict()
end
```

Maximum Entropy. Solver must support `MOI.RelativeEntropyCone` and `MOI.NormOneCone`.

# Parameters

  - `max_phi`: Maximum weight constraint of the L-moments.
"""
mutable struct MaxEntropy{T1 <: Real, T2 <: AbstractDict} <: OWAMethods
    max_phi::T1
    solvers::T2
end
function MaxEntropy(; max_phi::Real = 0.5, solvers::AbstractDict = Dict())
    @smart_assert(zero(max_phi) < max_phi < one(max_phi))
    return MaxEntropy{typeof(max_phi), typeof(solvers)}(max_phi, solvers)
end

"""
```
@kwdef mutable struct MinSumSq{T1 <: Real} <: OWAMethods
    max_phi::T1 = 0.5
end
```

Minimum Sum of Squares. Solver must support `MOI.SecondOrderCone`.

# Parameters

  - `max_phi`: Maximum weight constraint of the L-moments.
"""
mutable struct MinSumSq{T1 <: Real, T2 <: AbstractDict} <: OWAMethods
    max_phi::T1
    solvers::T2
end
function MinSumSq(; max_phi::Real = 0.5, solvers::AbstractDict = Dict())
    @smart_assert(zero(max_phi) < max_phi < one(max_phi))
    return MinSumSq{typeof(max_phi), typeof(solvers)}(max_phi, solvers)
end

"""
```
@kwdef mutable struct MinSqDist{T1 <: Real} <: OWAMethods
    max_phi::T1 = 0.5
end
```

Minimum Square Distance. Solver must support `MOI.SecondOrderCone`.

# Parameters

  - `max_phi`: Maximum weight constraint of the L-moments.
"""
mutable struct MinSqDist{T1 <: Real, T2 <: AbstractDict} <: OWAMethods
    max_phi::T1
    solvers::T2
end
function MinSqDist(; max_phi::Real = 0.5, solvers::AbstractDict = Dict())
    @smart_assert(zero(max_phi) < max_phi < one(max_phi))
    return MinSqDist{typeof(max_phi), typeof(solvers)}(max_phi, solvers)
end

export CRRA, MaxEntropy, MinSumSq, MinSqDist
