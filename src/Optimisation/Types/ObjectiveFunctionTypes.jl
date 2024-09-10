abstract type ObjectiveFunction end
struct MinRisk <: ObjectiveFunction end
@kwdef mutable struct Utility{T1 <: Real} <: ObjectiveFunction
    l::T1 = 2.0
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
"""
mutable struct Sharpe{T1 <: Real} <: ObjectiveFunction
    rf::T1
end
function Sharpe(; rf::Real = 0.0)
    return Sharpe{typeof(rf)}(rf)
end

struct MaxRet <: ObjectiveFunction end

export MinRisk, Utility, Sharpe, MaxRet
