abstract type WorstCaseSet end
struct Box <: WorstCaseSet end
struct Ellipse <: WorstCaseSet end
@kwdef mutable struct NoWC <: WorstCaseSet
    formulation::SDSquaredFormulation = SOCSD()
end

abstract type WorstCaseMethod end
abstract type WorstCaseArchMethod <: WorstCaseMethod end
struct StationaryBS <: WorstCaseArchMethod end
struct CircularBS <: WorstCaseArchMethod end
struct MovingBS <: WorstCaseArchMethod end

@kwdef mutable struct ArchWC <: WorstCaseMethod
    bootstrap::WorstCaseArchMethod = StationaryBS()
    n_sim::Integer = 3_000
    block_size::Integer = 3
    q::Real = 0.05
    seed::Union{<:Integer, Nothing} = nothing
end

@kwdef mutable struct NormalWC <: WorstCaseMethod
    n_sim::Integer = 3_000
    q::Real = 0.05
    rng::AbstractRNG = Random.default_rng()
    seed::Union{<:Integer, Nothing} = nothing
end

@kwdef mutable struct DeltaWC <: WorstCaseMethod
    dcov::Real = 0.1
    dmu::Real = 0.1
end

abstract type WorstCaseKMethod end
struct KNormalWC <: WorstCaseKMethod end
struct KGeneralWC <: WorstCaseKMethod end

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

export Box, Ellipse, NoWC, StationaryBS, CircularBS, MovingBS, ArchWC, NormalWC, DeltaWC,
       KNormalWC, KGeneralWC, WCType
