# Copywrite (c) 2025
# Author: Daniel Celis Garza <daniel.celis.garza@gmail.com>
# SPDX-License-Identifier: MIT

"""
```
abstract type WorstCaseType end
```

Abstract type for subtyping types for computing worst case mean variance sets.
"""
abstract type WorstCaseType end

"""
```
abstract type WorstCaseArchType <: WorstCaseType end
```

Abstract type for subtyping types for computing worst case mean variance for bootstrapping with [`arch`](https://pypi.org/project/arch/).
"""
abstract type WorstCaseArchType <: WorstCaseType end

"""
```
struct StationaryBS <: WorstCaseArchType end
```

[Stationary](https://bashtage.github.io/arch/bootstrap/generated/arch.bootstrap.StationaryBootstrap.html#arch.bootstrap.StationaryBootstrap) bootstrap.
"""
struct StationaryBS <: WorstCaseArchType end

"""
```
struct CircularBS <: WorstCaseArchType end
```

[Circular block](https://bashtage.github.io/arch/bootstrap/generated/arch.bootstrap.CircularBlockBootstrap.html#arch.bootstrap.CircularBlockBootstrap) bootstrap.
"""
struct CircularBS <: WorstCaseArchType end

"""
```
struct MovingBS <: WorstCaseArchType end
```

[Moving block](https://bashtage.github.io/arch/bootstrap/generated/arch.bootstrap.MovingBlockBootstrap.html#arch.bootstrap.MovingBlockBootstrap) bootstrap.
"""
struct MovingBS <: WorstCaseArchType end

"""
```
@kwdef mutable struct ArchWC{T1 <: Integer, T2 <: Integer, T3 <: Real} <: WorstCaseType
    bootstrap::WorstCaseArchType = StationaryBS()
    n_sim::T1 = 3_000
    block_size::T2 = 3
    q::T3 = 0.05
    seed::Union{<:Integer, Nothing} = nothing
end
```
"""
mutable struct ArchWC{T1 <: Integer, T2 <: Integer, T3 <: Real} <: WorstCaseType
    bootstrap::WorstCaseArchType
    n_sim::T1
    block_size::T2
    q::T3
    seed::Union{<:Integer, Nothing}
end
function ArchWC(; bootstrap::WorstCaseArchType = StationaryBS(), n_sim::Integer = 3_000,
                block_size::Integer = 3, q::Real = 0.05,
                seed::Union{<:Integer, Nothing} = nothing)
    return ArchWC{typeof(n_sim), typeof(block_size), typeof(q)}(bootstrap, n_sim,
                                                                block_size, q, seed)
end

"""
```
@kwdef mutable struct NormalWC{T1 <: Integer, T2 <: Real} <: WorstCaseType
    n_sim::T1 = 3_000
    q::T2 = 0.05
    rng::AbstractRNG = Random.default_rng()
    seed::Union{<:Integer, Nothing} = nothing
end
```
"""
mutable struct NormalWC{T1 <: Integer, T2 <: Real} <: WorstCaseType
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
@kwdef mutable struct DeltaWC{T1 <: Real, T2 <: Real} <: WorstCaseType
    dcov::T1 = 0.1
    dmu::T2 = 0.1
end
```
"""
mutable struct DeltaWC{T1 <: Real, T2 <: Real} <: WorstCaseType
    dcov::T1
    dmu::T2
end
function DeltaWC(; dcov::Real = 0.1, dmu::Real = 0.1)
    return DeltaWC{typeof(dcov), typeof(dmu)}(dcov, dmu)
end

"""
```
abstract type WorstCaseKType end
```
"""
abstract type WorstCaseKType end

"""
```
struct KNormalWC <: WorstCaseKType end
```
"""
struct KNormalWC <: WorstCaseKType end

"""
```
struct KGeneralWC <: WorstCaseKType end
```
"""
struct KGeneralWC <: WorstCaseKType end

"""
```
@kwdef mutable struct WCType
    cov_type::PortfolioOptimiserCovCor = PortCovCor(;)
    mu_type::MeanEstimator = MuSimple(;)
    box::WorstCaseType = NormalWC(;)
    ellipse::WorstCaseType = NormalWC(;)
    k_sigma::Union{<:Real, WorstCaseKType} = KNormalWC(;)
    k_mu::Union{<:Real, WorstCaseKType} = KNormalWC(;)
    posdef::AbstractPosdefFix = PosdefNearest(;)
    diagonal::Bool = false
end
```
"""
mutable struct WCType
    cov_type::PortfolioOptimiserCovCor
    mu_type::MeanEstimator
    box::WorstCaseType
    ellipse::WorstCaseType
    k_sigma::Union{<:Real, WorstCaseKType}
    k_mu::Union{<:Real, WorstCaseKType}
    posdef::AbstractPosdefFix
    diagonal::Bool
end
function WCType(; cov_type::PortfolioOptimiserCovCor = PortCovCor(;),
                mu_type::MeanEstimator = MuSimple(;), box::WorstCaseType = NormalWC(;),
                ellipse::WorstCaseType = NormalWC(;),
                k_sigma::Union{<:Real, WorstCaseKType} = KNormalWC(;),
                k_mu::Union{<:Real, WorstCaseKType} = KNormalWC(;),
                posdef::AbstractPosdefFix = PosdefNearest(;), diagonal::Bool = false)
    return WCType(cov_type, mu_type, box, ellipse, k_sigma, k_mu, posdef, diagonal)
end

export StationaryBS, CircularBS, MovingBS, ArchWC, NormalWC, DeltaWC, KNormalWC, KGeneralWC,
       WCType
