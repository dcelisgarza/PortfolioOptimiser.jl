"""
```
abstract type PosdefFix end
```

Abstract type for subtyping methods for fixing non positive definite matrices in [`posdef_fix!`](@ref).
"""
abstract type PosdefFix end

"""
```
struct NoPosdef <: PosdefFix end
```

Non positive definite matrices will not be fixed in [`posdef_fix!`](@ref).
"""
struct NoPosdef <: PosdefFix end

"""
```
@kwdef mutable struct PosdefNearest <: PosdefFix
    method::NearestCorrelationMatrix.NCMAlgorithm = NearestCorrelationMatrix.Newton(;
                                                                                    tau = 1e-12)
end
```

Defines which method from [`NearestCorrelationMatrix`](https://github.com/adknudson/NearestCorrelationMatrix.jl) to use in [`posdef_fix!`](@ref).
"""
mutable struct PosdefNearest <: PosdefFix
    method::NearestCorrelationMatrix.NCMAlgorithm
end
function PosdefNearest(;
                       method::NearestCorrelationMatrix.NCMAlgorithm = NearestCorrelationMatrix.Newton(;
                                                                                                       tau = 1e-12))
    return PosdefNearest(method)
end

export NoPosdef, PosdefNearest
