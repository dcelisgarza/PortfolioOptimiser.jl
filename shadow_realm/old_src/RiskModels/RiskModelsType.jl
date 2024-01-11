"""
```
abstract type AbstractFixPosDef end
struct SFix <: AbstractFixPosDef end
struct DFix <: AbstractFixPosDef end
struct FFix <: AbstractFixPosDef end
```

Types for fixing non-positive definite matrices in [`cov`](@ref).

  - `SFix`: fix via the spectral method (eigenvalue decomposition).
  - `DFix`: fix by adding a multiple of the identity matrix (damping factor).
  - `FFix`: fix by changing bad eigenvalues their sum devided by the difference between the number of good eigenvalues minus bad ones.
"""
abstract type AbstractFixPosDef end
struct SFix <: AbstractFixPosDef end
struct DFix <: AbstractFixPosDef end
struct FFix <: AbstractFixPosDef end

"""
```
abstract type AbstractRiskModel end
struct Cov <: AbstractRiskModel end
struct SCov <: AbstractRiskModel end
struct ECov <: AbstractRiskModel end
struct ESCov <: AbstractRiskModel end
struct CustomCov <: AbstractRiskModel end
```

Risk models for dispatch on [`cov`](@ref).

  - `Cov`: Mean returns.
  - `SCov`: Exponentially weighted mean returns.
  - `ECov`: Capital Asset Pricing Model (CAPM) returns.
  - `ESCov`: Exponentially weighted Capital Asset Pricing Model (ECAPM) returns.
  - `CustomCov`: Custom covariance matrix.
  - `CustomSCov`: Custom semicovariance matrix.
"""
abstract type AbstractRiskModel end
struct Cov <: AbstractRiskModel end
struct SCov <: AbstractRiskModel end
struct ECov <: AbstractRiskModel end
struct ESCov <: AbstractRiskModel end
struct CustomCov <: AbstractRiskModel end
struct CustomSCov <: AbstractRiskModel end
