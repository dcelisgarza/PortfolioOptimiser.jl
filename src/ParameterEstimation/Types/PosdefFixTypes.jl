"""
```
abstract type AbstractPosdefFix end
```

Abstract type for subtyping methods for fixing non positive definite matrices in [`posdef_fix!`](@ref).
"""
abstract type AbstractPosdefFix end

"""
```
struct NoPosdef <: AbstractPosdefFix end
```

Non positive definite matrices will not be fixed in [`posdef_fix!`](@ref).
"""
struct NoPosdef <: AbstractPosdefFix end

"""
```
@kwdef mutable struct PosdefNearest <: AbstractPosdefFix
    method::NearestCorrelationMatrix.NCMAlgorithm = NearestCorrelationMatrix.Newton(;
                                                                                    tau = 1e-12)
end
```

Defines which method from [`NearestCorrelationMatrix`](https://github.com/adknudson/NearestCorrelationMatrix.jl) to use in [`posdef_fix!`](@ref).
"""
mutable struct PosdefNearest <: AbstractPosdefFix
    method::NearestCorrelationMatrix.NCMAlgorithm
end
function PosdefNearest(;
                       method::NearestCorrelationMatrix.NCMAlgorithm = NearestCorrelationMatrix.Newton(;
                                                                                                       tau = 1e-12))
    return PosdefNearest(method)
end

export NoPosdef, PosdefNearest
