module PortfolioOptimiser

@doc let
    path = joinpath(dirname(@__DIR__), "README.md")
    include_dependency(path)
    read(path, String)
end PortfolioOptimiser

using AverageShiftedHistograms, Clustering, DataFrames, Dates, Distances, Distributions,
      GLM, JuMP, LinearAlgebra, MultivariateStats, NearestCorrelationMatrix, Optim, Graphs,
      SimpleWeightedGraphs, PyCall, Random, SmartAsserts, SparseArrays, Statistics,
      StatsBase, TimeSeries

include("./RiskMeasureTypes.jl")
include("./ParameterEstimationTypes.jl")
include("./OptimisationTypes.jl")
include("./PortfolioTypes.jl")
include("./Misc.jl")
include("./OWA.jl")
include("./DBHTs.jl")
include("./ParameterEstimation.jl")
include("./Constraints.jl")
include("./RiskMeasures.jl")
include("./HClustering.jl")
include("./NetworkConstraints.jl")
include("./Statistics.jl")
include("./Optimisation.jl")
include("./PortfolioOptimiserPlotsExtDefinitions.jl")
end
