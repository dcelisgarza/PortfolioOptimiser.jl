# Copywrite (c) 2025
# Author: Daniel Celis Garza <daniel.celis.garza@gmail.com>
# SPDX-License-Identifier: MIT

"""
```
abstract type ObjectiveFunction end
```
"""
abstract type ObjectiveFunction end

"""
```
struct MinRisk <: ObjectiveFunction end
```
"""
struct MinRisk <: ObjectiveFunction end

"""
```
@kwdef mutable struct Utility{T1 <: Real} <: ObjectiveFunction
    l::T1 = 2.0
end
```
"""
mutable struct Utility{T1 <: Real} <: ObjectiveFunction
    l::T1
end
function Utility(; l::Real = 2.0)
    return Utility(l)
end

"""
```
@kwdef mutable struct Sharpe{T1 <: Real} <: ObjectiveFunction
    rf::T1 = 0.0
end
```

Maximum risk-adjusted return (Sharpe) ratio objective function.

# Parameters

  - `rf`: risk free rate.
  - `ohf::T2 = 1.0 where T2 <: Real`: The optimal homogenisation factor.
"""
mutable struct Sharpe{T1 <: Real, T2 <: Real} <: ObjectiveFunction
    rf::T1
    ohf::T2
end
function Sharpe(; rf::Real = 0.0, ohf::Real = 1.0)
    @smart_assert(ohf >= zero(ohf))
    return Sharpe{typeof(rf), typeof(ohf)}(rf, ohf)
end
function Base.setproperty!(obj::Sharpe, sym::Symbol, val)
    if sym == :ohf
        @smart_assert(val >= zero(val))
    end
    return setfield!(obj, sym, val)
end

"""
```
struct MaxRet <: ObjectiveFunction end
```
"""
struct MaxRet <: ObjectiveFunction end

"""
    abstract type CustomObjective end
"""
abstract type CustomObjective end
"""
    struct NoCustomObjective <: CustomObjective end
"""
struct NoCustomObjective <: CustomObjective end

export MinRisk, Utility, Sharpe, MaxRet
