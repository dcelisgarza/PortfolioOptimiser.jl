module PortfolioOptimiser
using AverageShiftedHistograms, Clustering, DataFrames, Dates, Distances, Distributions,
      GLM, JuMP, LinearAlgebra, MultivariateStats, NearestCorrelationMatrix, Optim, Graphs,
      SimpleWeightedGraphs, PyCall, Random, SmartAsserts, SparseArrays, Statistics,
      StatsBase, TimeSeries

include("Constants.jl")
include("Types.jl")
include("DBHTs.jl")
include("OWA.jl")
include("Portfolio.jl")
include("Statistics.jl")
include("Risk_measures.jl")
include("Constraints.jl")
include("ExtDefinitions.jl")

# Version 2.0
include("Types_v2.jl")
include("ParameterEstimation.jl")
include("RiskMeasures_v2.jl")
include("Optimisation.jl")
include("Statistics_v2.jl")
include("HClustering.jl")
include("NetworkConstraints.jl")
include("PortfolioRisk.jl")

end
