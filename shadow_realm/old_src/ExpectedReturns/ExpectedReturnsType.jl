"""
```
abstract type AbstractReturnModel end
struct MRet <: AbstractReturnModel end
struct EMRet <: AbstractReturnModel end
struct CAPMRet <: AbstractReturnModel end
struct ECAPMRet <: AbstractReturnModel end
```

Return types, used to dispatch on [`ret_model`](@ref).

  - `MRet`: Mean returns.
  - `EMRet`: Exponentially weighted mean returns.
  - `CAPMRet`: Capital Asset Pricing Model (CAPM) returns.
  - `ECAPMRet`: Exponentially weighted Capital Asset Pricing Model (ECAPM) returns.
"""
abstract type AbstractReturnModel end
struct MRet <: AbstractReturnModel end
struct EMRet <: AbstractReturnModel end
struct CAPMRet <: AbstractReturnModel end
struct ECAPMRet <: AbstractReturnModel end
