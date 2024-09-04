# ## Ordered Weight Array statistics

"""
```
OWAMethods = (:CRRA, :E, :SS, :SD)
```

Methods for computing the weights used to combine L-moments higher than 2 [OWAL](@cite).

  - `:CRRA:` Normalised Constant Relative Risk Aversion Coefficients.
  - `:E`: Maximum Entropy. Solver must support `MOI.RelativeEntropyCone` and `MOI.NormOneCone`.
  - `:SS`: Minimum Sum of Squares. Solver must support `MOI.SecondOrderCone`.
  - `:SD`: Minimum Square Distance. Solver must support `MOI.SecondOrderCone`.
"""
abstract type OWAMethods end

mutable struct CRRA{T1 <: Real} <: OWAMethods
    g::T1
end
function CRRA(; g::Real = 0.5)
    @smart_assert(zero(g) < g < one(g))
    return CRRA{typeof(g)}(g)
end

mutable struct MaxEntropy{T1 <: Real} <: OWAMethods
    max_phi::T1
end
function MaxEntropy(; max_phi::Real = 0.5)
    @smart_assert(zero(max_phi) < max_phi < one(max_phi))
    return MaxEntropy{typeof(max_phi)}(max_phi)
end

mutable struct MinSumSq{T1 <: Real} <: OWAMethods
    max_phi::T1
end
function MinSumSq(; max_phi::Real = 0.5)
    @smart_assert(zero(max_phi) < max_phi < one(max_phi))
    return MinSumSq{typeof(max_phi)}(max_phi)
end

mutable struct MinSqDist{T1 <: Real} <: OWAMethods
    max_phi::T1
end
function MinSqDist(; max_phi::Real = 0.5)
    @smart_assert(zero(max_phi) < max_phi < one(max_phi))
    return MinSqDist{typeof(max_phi)}(max_phi)
end

export CRRA, MaxEntropy, MinSumSq, MinSqDist
