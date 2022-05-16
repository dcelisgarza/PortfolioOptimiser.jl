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

Risk models for dispatch.
"""
abstract type AbstractRiskModel end
struct Cov <: AbstractRiskModel end
struct SCov <: AbstractRiskModel end
struct ECov <: AbstractRiskModel end
struct ESCov <: AbstractRiskModel end
