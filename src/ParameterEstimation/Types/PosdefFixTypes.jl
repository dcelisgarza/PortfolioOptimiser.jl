# Copywrite (c) 2025
# Author: Daniel Celis Garza <daniel.celis.garza@gmail.com>
# SPDX-License-Identifier: MIT

"""
```
abstract type AbstractPosdefFix end
```

Abstract type for subtyping types for fixing non positive definite matrices in [`posdef_fix!`](@ref).
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
    type::NearestCorrelationMatrix.NCMAlgorithm = NearestCorrelationMatrix.Newton(;
                                                                                    tau = 1e-12)
end
```

Defines which type from [`NearestCorrelationMatrix`](https://github.com/adknudson/NearestCorrelationMatrix.jl) to use in [`posdef_fix!`](@ref).
"""
mutable struct PosdefNearest <: AbstractPosdefFix
    type::NearestCorrelationMatrix.NCMAlgorithm
end
function PosdefNearest(;
                       type::NearestCorrelationMatrix.NCMAlgorithm = NearestCorrelationMatrix.Newton(;
                                                                                                     tau = 1e-12))
    return PosdefNearest(type)
end

export NoPosdef, PosdefNearest
