"""
```
abstract type AbstractFixPosDef end
struct SFix <: AbstractFixPosDef end
struct DFix <: AbstractFixPosDef end
```

Types for fixing non-positive definite matrices in [`risk_matrix`](@ref).

- `SFix`: fix via the spectral method (eigenvalue decomposition).
- `DFix`: fix by adding a multiple of the identity matrix (damping factor).
"""
abstract type AbstractFixPosDef end
struct SFix <: AbstractFixPosDef end
struct DFix <: AbstractFixPosDef end

"""
```
abstract type AbstractRiskModel end
struct Cov <: AbstractRiskModel end
struct SCov <: AbstractRiskModel end
struct ECov <: AbstractRiskModel end
struct ESCov <: AbstractRiskModel end
```

Risk models for dispatch on [`risk_matrix`](@ref).

- `Cov`: Mean returns.
- `SCov`: Exponentially weighted mean returns.
- `ECov`: Capital Asset Pricing Model (CAPM) returns.
- `ESCov`: Exponentially weighted Capital Asset Pricing Model (ECAPM) returns.
"""
abstract type AbstractRiskModel end
struct Cov <: AbstractRiskModel end
struct SCov <: AbstractRiskModel end
struct ECov <: AbstractRiskModel end
struct ESCov <: AbstractRiskModel end
