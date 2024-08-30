# # Parameter estimation

# ## Postitive definite matrices

abstract type PosdefFix end
struct NoPosdef <: PosdefFix end
@kwdef mutable struct PosdefNearest <: PosdefFix
    method::NearestCorrelationMatrix.NCMAlgorithm = NearestCorrelationMatrix.Newton(;
                                                                                    tau = 1e-12)
end

# ## Matrix denoising

abstract type Denoise end

struct NoDenoise <: Denoise end
function denoise!(::NoDenoise, ::PosdefFix, X::AbstractMatrix, q::Real)
    return nothing
end

mutable struct Fixed{T1, T2, T3, T4} <: Denoise
    detone::Bool
    mkt_comp::T1
    kernel::T2
    m::T3
    n::T4
    args::Tuple
    kwargs::NamedTuple
end
function Fixed(; detone::Bool = false, mkt_comp::Integer = 1,
               kernel = AverageShiftedHistograms.Kernels.gaussian, m::Integer = 10,
               n::Integer = 1000, args::Tuple = (), kwargs::NamedTuple = (;))
    return Fixed{typeof(mkt_comp), typeof(kernel), typeof(m), typeof(n)}(detone, mkt_comp,
                                                                         kernel, m, n, args,
                                                                         kwargs)
end

mutable struct Spectral{T1, T2, T3, T4} <: Denoise
    detone::Bool
    mkt_comp::T1
    kernel::T2
    m::T3
    n::T4
    args::Tuple
    kwargs::NamedTuple
end
function Spectral(; detone::Bool = false, mkt_comp::Integer = 1,
                  kernel = AverageShiftedHistograms.Kernels.gaussian, m::Integer = 10,
                  n::Integer = 1000, args::Tuple = (), kwargs::NamedTuple = (;))
    return Spectral{typeof(mkt_comp), typeof(kernel), typeof(m), typeof(n)}(detone,
                                                                            mkt_comp,
                                                                            kernel, m, n,
                                                                            args, kwargs)
end

mutable struct Shrink{T1, T2, T3, T4, T5} <: Denoise
    detone::Bool
    alpha::T1
    mkt_comp::T2
    kernel::T3
    m::T4
    n::T5
    args::Tuple
    kwargs::NamedTuple
end
function Shrink(; alpha::Real = 0.0, detone::Bool = false, mkt_comp::Integer = 1,
                kernel = AverageShiftedHistograms.Kernels.gaussian, m::Integer = 10,
                n::Integer = 1000, args::Tuple = (), kwargs::NamedTuple = (;))
    @smart_assert(zero(alpha) <= alpha <= one(alpha))
    return Shrink{typeof(alpha), typeof(mkt_comp), typeof(kernel), typeof(m), typeof(n)}(detone,
                                                                                         alpha,
                                                                                         mkt_comp,
                                                                                         kernel,
                                                                                         m,
                                                                                         n,
                                                                                         args,
                                                                                         kwargs)
end

# ## Distances

abstract type DistanceMethod end

@kwdef mutable struct DistanceMLP <: DistanceMethod
    absolute::Bool = false
end

@kwdef mutable struct DistanceSqMLP <: DistanceMethod
    absolute::Bool = false
    distance::Distances.UnionMetric = Distances.Euclidean()
    args::Tuple = ()
    kwargs::NamedTuple = (;)
end

struct DistanceLog <: DistanceMethod end

struct DistanceDefault <: DistanceMethod end

abstract type AbstractBins end
abstract type AstroBins <: AbstractBins end
struct Knuth <: AstroBins end
struct Freedman <: AstroBins end
struct Scott <: AstroBins end
struct HGR <: AbstractBins end
mutable struct DistanceVarInfo <: DistanceMethod
    bins::Union{<:Integer, <:AbstractBins}
    normalise::Bool
end
function DistanceVarInfo(; bins::Union{<:Integer, <:AbstractBins} = HGR(),
                         normalise::Bool = true)
    if isa(bins, Integer)
        @smart_assert(bins > zero(bins))
    end
    return DistanceVarInfo(bins, normalise)
end

# ## Clustering

abstract type HClustAlg end
@kwdef mutable struct HAC <: HClustAlg
    linkage::Symbol = :ward
end

abstract type DBHTSimilarity end
struct DBHTExp <: DBHTSimilarity end
struct DBHTMaxDist <: DBHTSimilarity end
abstract type DBHTRootMethod end
struct UniqueDBHT <: DBHTRootMethod end
struct EqualDBHT <: DBHTRootMethod end
mutable struct DBHT <: HClustAlg
    distance::DistanceMethod
    similarity::DBHTSimilarity
    root_method::DBHTRootMethod
end
function DBHT(; distance::DistanceMethod = DistanceMLP(),
              similarity::DBHTSimilarity = DBHTMaxDist(),
              root_method::DBHTRootMethod = UniqueDBHT())
    return DBHT(distance, similarity, root_method)
end

abstract type NumClusterMethod end
struct TwoDiff <: NumClusterMethod end
@kwdef mutable struct StdSilhouette <: NumClusterMethod
    metric::Union{Distances.SemiMetric, Nothing} = nothing
end
@kwdef mutable struct HCOpt{T1 <: Integer, T2 <: Integer}
    branchorder::Symbol = :optimal
    k_method::NumClusterMethod = TwoDiff()
    k::T1 = 0
    max_k::T2 = 0
end

struct ClusterNode{tid, tl, tr, td, tcnt}
    id::tid
    left::tl
    right::tr
    dist::td
    count::tcnt

    function ClusterNode(id, left::Union{ClusterNode, Nothing} = nothing,
                         right::Union{ClusterNode, Nothing} = nothing, dist::Real = 0.0,
                         count::Int = 1)
        icount = isnothing(left) ? count : (left.count + right.count)

        return new{typeof(id), typeof(left), typeof(right), typeof(dist), typeof(count)}(id,
                                                                                         left,
                                                                                         right,
                                                                                         dist,
                                                                                         icount)
    end
end

# ## Covariance, correlation, kurt and skew

abstract type PortfolioOptimiserCovCor <: StatsBase.CovarianceEstimator end
abstract type CorPearson <: PortfolioOptimiserCovCor end
abstract type CorRank <: PortfolioOptimiserCovCor end

@kwdef mutable struct CovFull <: CorPearson
    absolute::Bool = false
    ce::StatsBase.CovarianceEstimator = StatsBase.SimpleCovariance(; corrected = true)
    w::Union{<:AbstractWeights, Nothing} = nothing
end

@kwdef mutable struct SimpleVariance <: StatsBase.CovarianceEstimator
    corrected = true
end

@kwdef mutable struct CovSemi <: CorPearson
    absolute::Bool = false
    ce::StatsBase.CovarianceEstimator = StatsBase.SimpleCovariance(; corrected = true)
    target::Union{<:Real, AbstractVector{<:Real}} = 0.0
    w::Union{<:AbstractWeights, Nothing} = nothing
end

@kwdef mutable struct CorSpearman <: CorRank
    absolute::Bool = false
end

@kwdef mutable struct CorKendall <: CorRank
    absolute::Bool = false
end

mutable struct CorMutualInfo <: PortfolioOptimiserCovCor
    bins::Union{<:Integer, <:AbstractBins}
    normalise::Bool
    ve::StatsBase.CovarianceEstimator
    std_w::Union{<:AbstractWeights, Nothing}
end
function CorMutualInfo(; bins::Union{<:Integer, <:AbstractBins} = HGR(),
                       normalise::Bool = true,
                       ve::StatsBase.CovarianceEstimator = SimpleVariance(),
                       std_w::Union{<:AbstractWeights, Nothing} = nothing)
    if isa(bins, Integer)
        @smart_assert(bins > zero(bins))
    end
    return CorMutualInfo(bins, normalise, ve, std_w)
end

@kwdef mutable struct CorDistance <: PortfolioOptimiserCovCor
    distance::Distances.UnionMetric = Distances.Euclidean()
    dist_args::Tuple = ()
    dist_kwargs::NamedTuple = (;)
    mean_w1::Union{<:AbstractWeights, Nothing} = nothing
    mean_w2::Union{<:AbstractWeights, Nothing} = nothing
    mean_w3::Union{<:AbstractWeights, Nothing} = nothing
end

mutable struct CorLTD <: PortfolioOptimiserCovCor
    alpha::Real
    ve::StatsBase.CovarianceEstimator
    std_w::Union{<:AbstractWeights, Nothing}
end
function CorLTD(; alpha::Real = 0.05, ve::StatsBase.CovarianceEstimator = SimpleVariance(),
                std_w::Union{<:AbstractWeights, Nothing} = nothing)
    @smart_assert(zero(alpha) < alpha < one(alpha))
    return CorLTD(alpha, ve, std_w)
end

abstract type CorGerber <: PortfolioOptimiserCovCor end
abstract type CorGerberBasic <: CorGerber end
abstract type CorSB <: CorGerber end
abstract type CorGerberSB <: CorGerber end

mutable struct CorGerber0{T1 <: Real} <: CorGerberBasic
    normalise::Bool
    threshold::T1
    ve::StatsBase.CovarianceEstimator
    std_w::Union{<:AbstractWeights, Nothing}
    mean_w::Union{<:AbstractWeights, Nothing}
    posdef::PosdefFix
end
function CorGerber0(; normalise::Bool = false, threshold::Real = 0.5,
                    ve::StatsBase.CovarianceEstimator = SimpleVariance(),
                    std_w::Union{<:AbstractWeights, Nothing} = nothing,
                    mean_w::Union{<:AbstractWeights, Nothing} = nothing,
                    posdef::PosdefFix = PosdefNearest())
    @smart_assert(zero(threshold) < threshold < one(threshold))
    return CorGerber0{typeof(threshold)}(normalise, threshold, ve, std_w, mean_w, posdef)
end

mutable struct CorGerber1{T1 <: Real} <: CorGerberBasic
    normalise::Bool
    threshold::T1
    ve::StatsBase.CovarianceEstimator
    std_w::Union{<:AbstractWeights, Nothing}
    mean_w::Union{<:AbstractWeights, Nothing}
    posdef::PosdefFix
end
function CorGerber1(; normalise::Bool = false, threshold::Real = 0.5,
                    ve::StatsBase.CovarianceEstimator = SimpleVariance(),
                    std_w::Union{<:AbstractWeights, Nothing} = nothing,
                    mean_w::Union{<:AbstractWeights, Nothing} = nothing,
                    posdef::PosdefFix = PosdefNearest())
    @smart_assert(zero(threshold) < threshold < one(threshold))
    return CorGerber1{typeof(threshold)}(normalise, threshold, ve, std_w, mean_w, posdef)
end

mutable struct CorGerber2{T1 <: Real} <: CorGerberBasic
    normalise::Bool
    threshold::T1
    ve::StatsBase.CovarianceEstimator
    std_w::Union{<:AbstractWeights, Nothing}
    mean_w::Union{<:AbstractWeights, Nothing}
    posdef::PosdefFix
end
function CorGerber2(; normalise::Bool = false, threshold::Real = 0.5,
                    ve::StatsBase.CovarianceEstimator = SimpleVariance(),
                    std_w::Union{<:AbstractWeights, Nothing} = nothing,
                    mean_w::Union{<:AbstractWeights, Nothing} = nothing,
                    posdef::PosdefFix = PosdefNearest())
    @smart_assert(zero(threshold) < threshold < one(threshold))
    return CorGerber2{typeof(threshold)}(normalise, threshold, ve, std_w, mean_w, posdef)
end
function Base.setproperty!(obj::CorGerberBasic, sym::Symbol, val)
    if sym == :threshold
        @smart_assert(zero(val) < val < one(val))
    end
    return setfield!(obj, sym, val)
end

mutable struct CorSB0{T1, T2, T3, T4, T5} <: CorSB
    normalise::Bool
    threshold::T1
    c1::T2
    c2::T3
    c3::T4
    n::T5
    ve::StatsBase.CovarianceEstimator
    std_w::Union{<:AbstractWeights, Nothing}
    mean_w::Union{<:AbstractWeights, Nothing}
    posdef::PosdefFix
end
function CorSB0(; normalise::Bool = false, threshold::Real = 0.5, c1::Real = 0.5,
                c2::Real = 0.5, c3::Real = 4.0, n::Real = 2.0,
                ve::StatsBase.CovarianceEstimator = SimpleVariance(),
                std_w::Union{<:AbstractWeights, Nothing} = nothing,
                mean_w::Union{<:AbstractWeights, Nothing} = nothing,
                posdef::PosdefFix = PosdefNearest())
    @smart_assert(zero(threshold) < threshold < one(threshold))
    @smart_assert(zero(c1) < c1 <= one(c1))
    @smart_assert(zero(c2) < c2 <= one(c2))
    @smart_assert(c3 > c2)
    return CorSB0{typeof(threshold), typeof(c1), typeof(c2), typeof(c3), typeof(n)}(normalise,
                                                                                    threshold,
                                                                                    c1, c2,
                                                                                    c3, n,
                                                                                    ve,
                                                                                    std_w,
                                                                                    mean_w,
                                                                                    posdef)
end

mutable struct CorSB1{T1, T2, T3, T4, T5} <: CorSB
    normalise::Bool
    threshold::T1
    c1::T2
    c2::T3
    c3::T4
    n::T5
    ve::StatsBase.CovarianceEstimator
    std_w::Union{<:AbstractWeights, Nothing}
    mean_w::Union{<:AbstractWeights, Nothing}
    posdef::PosdefFix
end
function CorSB1(; normalise::Bool = false, threshold::Real = 0.5, c1::Real = 0.5,
                c2::Real = 0.5, c3::Real = 4.0, n::Real = 2.0,
                ve::StatsBase.CovarianceEstimator = SimpleVariance(),
                std_w::Union{<:AbstractWeights, Nothing} = nothing,
                mean_w::Union{<:AbstractWeights, Nothing} = nothing,
                posdef::PosdefFix = PosdefNearest())
    @smart_assert(zero(threshold) < threshold < one(threshold))
    @smart_assert(zero(c1) < c1 <= one(c1))
    @smart_assert(zero(c2) < c2 <= one(c2))
    @smart_assert(c3 > c2)
    return CorSB1{typeof(threshold), typeof(c1), typeof(c2), typeof(c3), typeof(n)}(normalise,
                                                                                    threshold,
                                                                                    c1, c2,
                                                                                    c3, n,
                                                                                    ve,
                                                                                    std_w,
                                                                                    mean_w,
                                                                                    posdef)
end

mutable struct CorGerberSB0{T1, T2, T3, T4, T5} <: CorSB
    normalise::Bool
    threshold::T1
    c1::T2
    c2::T3
    c3::T4
    n::T5
    ve::StatsBase.CovarianceEstimator
    std_w::Union{<:AbstractWeights, Nothing}
    mean_w::Union{<:AbstractWeights, Nothing}
    posdef::PosdefFix
end
function CorGerberSB0(; normalise::Bool = false, threshold::Real = 0.5, c1::Real = 0.5,
                      c2::Real = 0.5, c3::Real = 4.0, n::Real = 2.0,
                      ve::StatsBase.CovarianceEstimator = SimpleVariance(),
                      std_w::Union{<:AbstractWeights, Nothing} = nothing,
                      mean_w::Union{<:AbstractWeights, Nothing} = nothing,
                      posdef::PosdefFix = PosdefNearest())
    @smart_assert(zero(threshold) < threshold < one(threshold))
    @smart_assert(zero(c1) < c1 <= one(c1))
    @smart_assert(zero(c2) < c2 <= one(c2))
    @smart_assert(c3 > c2)
    return CorGerberSB0{typeof(threshold), typeof(c1), typeof(c2), typeof(c3), typeof(n)}(normalise,
                                                                                          threshold,
                                                                                          c1,
                                                                                          c2,
                                                                                          c3,
                                                                                          n,
                                                                                          ve,
                                                                                          std_w,
                                                                                          mean_w,
                                                                                          posdef)
end

mutable struct CorGerberSB1{T1, T2, T3, T4, T5} <: CorSB
    normalise::Bool
    threshold::T1
    c1::T2
    c2::T3
    c3::T4
    n::T5
    ve::StatsBase.CovarianceEstimator
    std_w::Union{<:AbstractWeights, Nothing}
    mean_w::Union{<:AbstractWeights, Nothing}
    posdef::PosdefFix
end
function CorGerberSB1(; normalise::Bool = false, threshold::Real = 0.5, c1::Real = 0.5,
                      c2::Real = 0.5, c3::Real = 4.0, n::Real = 2.0,
                      ve::StatsBase.CovarianceEstimator = SimpleVariance(),
                      std_w::Union{<:AbstractWeights, Nothing} = nothing,
                      mean_w::Union{<:AbstractWeights, Nothing} = nothing,
                      posdef::PosdefFix = PosdefNearest())
    @smart_assert(zero(threshold) < threshold < one(threshold))
    @smart_assert(zero(c1) < c1 <= one(c1))
    @smart_assert(zero(c2) < c2 <= one(c2))
    @smart_assert(c3 > c2)
    return CorGerberSB1{typeof(threshold), typeof(c1), typeof(c2), typeof(c3), typeof(n)}(normalise,
                                                                                          threshold,
                                                                                          c1,
                                                                                          c2,
                                                                                          c3,
                                                                                          n,
                                                                                          ve,
                                                                                          std_w,
                                                                                          mean_w,
                                                                                          posdef)
end
function Base.setproperty!(obj::CorSB, sym::Symbol, val)
    if sym == :threshold
        @smart_assert(zero(val) < val < one(val))
    elseif sym ∈ (:c1, :c2)
        @smart_assert(zero(val) < val <= one(val) && val < obj.c3)
    elseif sym == :c3
        @smart_assert(val > obj.c2)
    end
    return setfield!(obj, sym, val)
end

abstract type AbstractJLoGo end
struct NoJLoGo <: AbstractJLoGo end
@kwdef mutable struct JLoGo <: AbstractJLoGo
    DBHT::DBHT = DBHT(;)
end

abstract type KurtEstimator end

@kwdef mutable struct KurtFull <: KurtEstimator
    posdef::PosdefFix = PosdefNearest(;)
    denoise::Denoise = NoDenoise(;)
    jlogo::AbstractJLoGo = NoJLoGo(;)
end

@kwdef mutable struct KurtSemi <: KurtEstimator
    target::Union{<:Real, AbstractVector{<:Real}} = 0.0
    posdef::PosdefFix = PosdefNearest(;)
    denoise::Denoise = NoDenoise(;)
    jlogo::AbstractJLoGo = NoJLoGo(;)
end

abstract type SkewEstimator end

struct SkewFull <: SkewEstimator end

@kwdef mutable struct SkewSemi <: SkewEstimator
    target::Union{<:Real, AbstractVector{<:Real}} = 0.0
end

"""
```
@kwdef mutable struct PortCovCor <: PortfolioOptimiserCovCor
    ce::CovarianceEstimator = CovFull(;)
    posdef::PosdefFix = PosdefNearest(;)
    denoise::Denoise = NoDenoise(;)
    jlogo::AbstractJLoGo = NoJLoGo(;)
end
```
"""
mutable struct PortCovCor <: PortfolioOptimiserCovCor
    ce::CovarianceEstimator
    posdef::PosdefFix
    denoise::Denoise
    jlogo::AbstractJLoGo
end
function PortCovCor(; ce::CovarianceEstimator = CovFull(;),
                    posdef::PosdefFix = PosdefNearest(;), denoise::Denoise = NoDenoise(;),
                    jlogo::AbstractJLoGo = NoJLoGo(;))
    return PortCovCor(ce, posdef, denoise, jlogo)
end

# ## Mean estimator

"""
```
abstract type MeanEstimator end
```
"""
abstract type MeanEstimator end

"""
```
abstract type MeanTarget end
```
"""
abstract type MeanTarget end

"""
```
struct GM <: MeanTarget end
```
"""
struct GM <: MeanTarget end

"""
```
struct VW <: MeanTarget end
```
"""
struct VW <: MeanTarget end

"""
```
struct SE <: MeanTarget end
```
"""
struct SE <: MeanTarget end

"""
```
@kwdef mutable struct MuSimple <: MeanEstimator
    w::Union{<:AbstractWeights, Nothing} = nothing
end
```
"""
mutable struct MuSimple <: MeanEstimator
    w::Union{<:AbstractWeights, Nothing}
end
function MuSimple(; w::Union{<:AbstractWeights, Nothing} = nothing)
    return MuSimple(w)
end

"""
```
@kwdef mutable struct MuJS{T1} <: MeanEstimator
    target::MeanTarget = GM()
    w::Union{<:AbstractWeights, Nothing} = nothing
    sigma::T1 = Matrix{Float64}(undef, 0, 0)
end
```
"""
mutable struct MuJS{T1} <: MeanEstimator
    target::MeanTarget
    w::Union{<:AbstractWeights, Nothing}
    sigma::T1
end
function MuJS(; target::MeanTarget = GM(), w::Union{<:AbstractWeights, Nothing} = nothing,
              sigma::AbstractMatrix = Matrix{Float64}(undef, 0, 0))
    return MuJS{typeof(sigma)}(target, w, sigma)
end

"""
```
@kwdef mutable struct MuBS{T1} <: MeanEstimator
    target::MeanTarget = GM()
    w::Union{<:AbstractWeights, Nothing} = nothing
    sigma::T1 = Matrix{Float64}(undef, 0, 0)
end
```
"""
mutable struct MuBS{T1} <: MeanEstimator
    target::MeanTarget
    w::Union{<:AbstractWeights, Nothing}
    sigma::T1
end
function MuBS(; target::MeanTarget = GM(), w::Union{<:AbstractWeights, Nothing} = nothing,
              sigma::AbstractMatrix = Matrix{Float64}(undef, 0, 0))
    return MuBS{typeof(sigma)}(target, w, sigma)
end

"""
```
@kwdef mutable struct MuBOP{T1} <: MeanEstimator
    target::MeanTarget = GM()
    w::Union{<:AbstractWeights, Nothing} = nothing
    sigma::T1 = Matrix{Float64}(undef, 0, 0)
end
```
"""
mutable struct MuBOP{T1} <: MeanEstimator
    target::MeanTarget
    w::Union{<:AbstractWeights, Nothing}
    sigma::T1
end
function MuBOP(; target::MeanTarget = GM(), w::Union{<:AbstractWeights, Nothing} = nothing,
               sigma::AbstractMatrix = Matrix{Float64}(undef, 0, 0))
    return MuBOP{typeof(sigma)}(target, w, sigma)
end

# ## Worst case statistics

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

# ## Regression statistics

abstract type RegressionType end
abstract type StepwiseRegression <: RegressionType end
abstract type RegressionCriteria end
abstract type MinValRegressionCriteria <: RegressionCriteria end
abstract type MaxValRegressionCriteria <: RegressionCriteria end
struct AIC <: MinValRegressionCriteria end
struct AICC <: MinValRegressionCriteria end
struct BIC <: MinValRegressionCriteria end
struct RSq <: MaxValRegressionCriteria end
struct AdjRSq <: MaxValRegressionCriteria end

mutable struct PVal{T1 <: Real} <: RegressionCriteria
    threshold::T1
end
function PVal(; threshold::Real = 0.05)
    @smart_assert(zero(threshold) < threshold < one(threshold))
    return PVal{typeof(threshold)}(threshold)
end

abstract type DimensionReductionTarget end
@kwdef mutable struct PCATarget <: DimensionReductionTarget
    kwargs::NamedTuple = (;)
end

@kwdef mutable struct FReg <: StepwiseRegression
    criterion::RegressionCriteria = PVal(;)
end

@kwdef mutable struct BReg <: StepwiseRegression
    criterion::RegressionCriteria = PVal(;)
end

@kwdef mutable struct DRR <: RegressionType
    ve::StatsBase.CovarianceEstimator = SimpleVariance(;)
    std_w::Union{<:AbstractWeights, Nothing} = nothing
    mean_w::Union{<:AbstractWeights, Nothing} = nothing
    pcr::DimensionReductionTarget = PCATarget(;)
end

@kwdef mutable struct FactorType
    error::Bool = true
    B::Union{Nothing, DataFrame} = nothing
    method::RegressionType = FReg(;)
    ve::StatsBase.CovarianceEstimator = SimpleVariance(;)
    var_w::Union{<:AbstractWeights, Nothing} = nothing
end

# ## Black Litterman

abstract type BlackLitterman end
abstract type BlackLittermanFactor <: BlackLitterman end

@kwdef mutable struct BLType{T1 <: Real} <: BlackLitterman
    eq::Bool = true
    delta::Union{<:Real, Nothing} = 1.0
    rf::T1 = 0.0
    posdef::PosdefFix = PosdefNearest()
    denoise::Denoise = NoDenoise()
    jlogo::AbstractJLoGo = NoJLoGo()
end

@kwdef mutable struct BBLType{T1 <: Real} <: BlackLittermanFactor
    constant::Bool = true
    error::Bool = true
    delta::Union{<:Real, Nothing} = 1.0
    rf::T1 = 0.0
    ve::StatsBase.CovarianceEstimator = SimpleVariance()
    var_w::Union{<:AbstractWeights, Nothing} = nothing
end

@kwdef mutable struct ABLType{T1 <: Real} <: BlackLittermanFactor
    constant::Bool = true
    eq::Bool = true
    delta::Union{<:Real, Nothing} = 1.0
    rf::T1 = 0.0
    posdef::PosdefFix = PosdefNearest()
    denoise::Denoise = NoDenoise()
    jlogo::AbstractJLoGo = NoJLoGo()
end

# ## Ordered Weight Array statistics

"""
```
OWAMethods = (:CRRA, :E, :SS, :SD)
```

Methods for computing the weights used to combine L-moments higher than 2 [OWAL](@cite).

  - `:CRRA:` Normalised Constant Relative Risk Aversion Coefficients.
  - `:E`: Maximum Entropy. Solver must support `MOI.RelativeEntropyCone` and `MOI.NormOneCone`.
  - `:SS`: Minimum Sum of Squares. Solver must support `MOI.SecondOrderCone`.
  - `:SD`: Minimum Square Distance. Solver must support `MOI.SecondOrderCone`.
"""
abstract type OWAMethods end

mutable struct CRRA{T1 <: Real} <: OWAMethods
    g::T1
end
function CRRA(; g::Real = 0.5)
    @smart_assert(zero(g) < g < one(g))
    return CRRA{typeof(g)}(g)
end

mutable struct MaxEntropy{T1 <: Real} <: OWAMethods
    max_phi::T1
end
function MaxEntropy(; max_phi::Real = 0.5)
    @smart_assert(zero(max_phi) < max_phi < one(max_phi))
    return MaxEntropy{typeof(max_phi)}(max_phi)
end

mutable struct MinSumSq{T1 <: Real} <: OWAMethods
    max_phi::T1
end
function MinSumSq(; max_phi::Real = 0.5)
    @smart_assert(zero(max_phi) < max_phi < one(max_phi))
    return MinSumSq{typeof(max_phi)}(max_phi)
end

mutable struct MinSqDist{T1 <: Real} <: OWAMethods
    max_phi::T1
end
function MinSqDist(; max_phi::Real = 0.5)
    @smart_assert(zero(max_phi) < max_phi < one(max_phi))
    return MinSqDist{typeof(max_phi)}(max_phi)
end

# ## Network statistics

abstract type CentralityType end

@kwdef mutable struct DegreeCentrality <: CentralityType
    args::Tuple = ()
    kwargs::NamedTuple = (;)
end

abstract type TreeType end
@kwdef mutable struct KruskalTree <: TreeType
    args::Tuple = ()
    kwargs::NamedTuple = (;)
end

abstract type NetworkType end

@kwdef mutable struct TMFG{T1 <: Integer} <: NetworkType
    similarity::DBHTSimilarity = DBHTMaxDist()
    steps::T1 = 1
    centrality::CentralityType = DegreeCentrality()
end

@kwdef mutable struct MST{T1 <: Integer} <: NetworkType
    tree::TreeType = KruskalTree()
    steps::T1 = 1
    centrality::CentralityType = DegreeCentrality()
end

export NoPosdef, PosdefNearest, NoDenoise, Fixed, Spectral, Shrink, DistanceMLP,
       DistanceSqMLP, DistanceLog, DistanceDefault, Knuth, Freedman, Scott, HGR,
       DistanceVarInfo, HAC, DBHTExp, DBHTMaxDist, UniqueDBHT, EqualDBHT, DBHT, TwoDiff,
       StdSilhouette, HCOpt, ClusterNode, CovFull, SimpleVariance, CovSemi, CorSpearman,
       CorKendall, CorMutualInfo, CorDistance, CorLTD, CorGerber0, CorGerber1, CorGerber2,
       CorSB0, CorSB1, CorGerberSB0, CorGerberSB1, NoJLoGo, JLoGo, KurtFull, KurtSemi,
       SkewFull, SkewSemi, PortCovCor, GM, VW, SE, MuSimple, MuJS, MuBS, MuBOP, Box,
       Ellipse, NoWC, StationaryBS, CircularBS, MovingBS, ArchWC, NormalWC, DeltaWC,
       KNormalWC, KGeneralWC, WCType, AIC, AICC, BIC, RSq, AdjRSq, PVal, PCATarget, FReg,
       BReg, DRR, FactorType, BLType, BBLType, ABLType, CRRA, MaxEntropy, MinSumSq,
       MinSqDist, DegreeCentrality, KruskalTree, TMFG, MST
