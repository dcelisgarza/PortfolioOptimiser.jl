abstract type AbstractFixPosDef end
struct SpecFix <: AbstractFixPosDef end
struct DiagFix <: AbstractFixPosDef end

abstract type AbstractRiskModel end
struct SampleCov <: AbstractRiskModel end
struct SemiCov <: AbstractRiskModel end
struct ExpCov <: AbstractRiskModel end
struct ExpSemiCov <: AbstractRiskModel end
