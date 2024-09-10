"""
```
abstract type WorstCaseSet end
```

Abstract type for subtyping worst case mean variance set types.
"""
abstract type WorstCaseSet end

"""
```
struct Box <: WorstCaseSet end
```

Box sets for worst case mean variance optimisation.
"""
struct Box <: WorstCaseSet end

"""
```
struct Ellipse <: WorstCaseSet end
```

Elliptical sets for worst case mean variance optimisation.
"""
struct Ellipse <: WorstCaseSet end

"""
```
@kwdef mutable struct NoWC <: WorstCaseSet
    formulation::SDSquaredFormulation = SOCSD()
end
```

Use no set for worst case mean variance optimisation.

# Parameters

  - `formulation`: quadratic expression formulation of [`SD`](@ref) risk measure to use [`SDSquaredFormulation`](@ref).
"""
mutable struct NoWC <: WorstCaseSet
    formulation::SDSquaredFormulation
end
function NoWC(; formulation::SDSquaredFormulation = SOCSD())
    return NoWC(formulation)
end

"""
```
abstract type WorstCaseMethod end
```

Abstract type for subtyping methods for computing worst case mean variance sets.
"""
abstract type WorstCaseMethod end

"""
```
abstract type WorstCaseArchMethod <: WorstCaseMethod end
```

Abstract type for subtyping methods for computing worst case mean variance for bootstrapping with [`arch`](https://pypi.org/project/arch/).
"""
abstract type WorstCaseArchMethod <: WorstCaseMethod end

"""
```
struct StationaryBS <: WorstCaseArchMethod end
```

[Stationary](https://bashtage.github.io/arch/bootstrap/generated/arch.bootstrap.StationaryBootstrap.html#arch.bootstrap.StationaryBootstrap) bootstrap.
"""
struct StationaryBS <: WorstCaseArchMethod end

"""
```
struct CircularBS <: WorstCaseArchMethod end
```

[Circular block](https://bashtage.github.io/arch/bootstrap/generated/arch.bootstrap.CircularBlockBootstrap.html#arch.bootstrap.CircularBlockBootstrap) bootstrap.
"""
struct CircularBS <: WorstCaseArchMethod end

"""
```
struct MovingBS <: WorstCaseArchMethod end
```

[Moving block](https://bashtage.github.io/arch/bootstrap/generated/arch.bootstrap.MovingBlockBootstrap.html#arch.bootstrap.MovingBlockBootstrap) bootstrap.
"""
struct MovingBS <: WorstCaseArchMethod end

"""
```
@kwdef mutable struct ArchWC{T1 <: Integer, T2 <: Integer, T3 <: Real} <: WorstCaseMethod
    bootstrap::WorstCaseArchMethod = StationaryBS()
    n_sim::T1 = 3_000
    block_size::T2 = 3
    q::T3 = 0.05
    seed::Union{<:Integer, Nothing} = nothing
end
```
"""
mutable struct ArchWC{T1 <: Integer, T2 <: Integer, T3 <: Real} <: WorstCaseMethod
    bootstrap::WorstCaseArchMethod
    n_sim::T1
    block_size::T2
    q::T3
    seed::Union{<:Integer, Nothing}
end
function ArchWC(; bootstrap::WorstCaseArchMethod = StationaryBS(), n_sim::Integer = 3_000,
                block_size::Integer = 3, q::Real = 0.05,
                seed::Union{<:Integer, Nothing} = nothing)
    return ArchWC{typeof(n_sim), typeof(block_size), typeof(q)}(bootstrap, n_sim,
                                                                block_size, q, seed)
end

"""
```
@kwdef mutable struct NormalWC{T1 <: Integer, T2 <: Real} <: WorstCaseMethod
    n_sim::T1 = 3_000
    q::T2 = 0.05
    rng::AbstractRNG = Random.default_rng()
    seed::Union{<:Integer, Nothing} = nothing
end
```
"""
mutable struct NormalWC{T1 <: Integer, T2 <: Real} <: WorstCaseMethod
    n_sim::T1
    q::T2
    rng::AbstractRNG
    seed::Union{<:Integer, Nothing}
end
function NormalWC(; n_sim::Integer = 3_000, q::Real = 0.05,
                  rng::AbstractRNG = Random.default_rng(),
                  seed::Union{<:Integer, Nothing} = nothing)
    return NormalWC{typeof(n_sim), typeof(q)}(n_sim, q, rng, seed)
end

"""
```
@kwdef mutable struct DeltaWC{T1 <: Real, T2 <: Real} <: WorstCaseMethod
    dcov::T1 = 0.1
    dmu::T2 = 0.1
end
```
"""
mutable struct DeltaWC{T1 <: Real, T2 <: Real} <: WorstCaseMethod
    dcov::T1
    dmu::T2
end
function DeltaWC(; dcov::Real = 0.1, dmu::Real = 0.1)
    return DeltaWC{typeof(dcov), typeof(dmu)}(dcov, dmu)
end

"""
```
abstract type WorstCaseKMethod end
```
"""
abstract type WorstCaseKMethod end

"""
```
struct KNormalWC <: WorstCaseKMethod end
```
"""
struct KNormalWC <: WorstCaseKMethod end

"""
```
struct KGeneralWC <: WorstCaseKMethod end
```
"""
struct KGeneralWC <: WorstCaseKMethod end

"""
```
@kwdef mutable struct WCType
    cov_type::PortfolioOptimiserCovCor = PortCovCor(;)
    mu_type::MeanEstimator = MuSimple(;)
    box::WorstCaseMethod = NormalWC(;)
    ellipse::WorstCaseMethod = NormalWC(;)
    k_sigma::Union{<:Real, WorstCaseKMethod} = KNormalWC(;)
    k_mu::Union{<:Real, WorstCaseKMethod} = KNormalWC(;)
    posdef::PosdefFix = PosdefNearest(;)
    diagonal::Bool = false
end
```
"""
mutable struct WCType
    cov_type::PortfolioOptimiserCovCor
    mu_type::MeanEstimator
    box::WorstCaseMethod
    ellipse::WorstCaseMethod
    k_sigma::Union{<:Real, WorstCaseKMethod}
    k_mu::Union{<:Real, WorstCaseKMethod}
    posdef::PosdefFix
    diagonal::Bool
end
function WCType(; cov_type::PortfolioOptimiserCovCor = PortCovCor(;),
                mu_type::MeanEstimator = MuSimple(;), box::WorstCaseMethod = NormalWC(;),
                ellipse::WorstCaseMethod = NormalWC(;),
                k_sigma::Union{<:Real, WorstCaseKMethod} = KNormalWC(;),
                k_mu::Union{<:Real, WorstCaseKMethod} = KNormalWC(;),
                posdef::PosdefFix = PosdefNearest(;), diagonal::Bool = false)
    return WCType(cov_type, mu_type, box, ellipse, k_sigma, k_mu, posdef, diagonal)
end

export Box, Ellipse, NoWC, StationaryBS, CircularBS, MovingBS, ArchWC, NormalWC, DeltaWC,
       KNormalWC, KGeneralWC, WCType
