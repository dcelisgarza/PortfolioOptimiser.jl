module PortfolioOptimiser
using AverageShiftedHistograms, Clustering, DataFrames, Dates, Distances, Distributions,
      GLM, JuMP, LinearAlgebra, MultivariateStats, NearestCorrelationMatrix, Optim, Graphs,
      SimpleWeightedGraphs, PyCall, Random, SmartAsserts, SparseArrays, Statistics,
      StatsBase, TimeSeries

# Types
## Risk measures
include("./RiskMeasures/Types/RiskMeasureTypes.jl")
## Parameter estimation
include("./ParameterEstimation/Types/ParameterEstimationTypes.jl")
## Optimisation
include("./Optimisation/Types/OptimisationTypes.jl")
## Portfolio
include("./Portfolio/Types/PortfolioTypes.jl")

# Functions
## Risk measures
include("./RiskMeasures/Functions/RiskMeasureFunctions.jl")
## Parameter estimation
include("./ParameterEstimation/Functions/ParameterEstimationFunctions.jl")
## Portfolio
include("./Portfolio/Functions/PortfolioFunctions.jl")
## Constraints
include("./Constraints/Constraints.jl")
## Optimisation
include("./Optimisation/Functions/OptimisationFunctions.jl")

# Extensions
include("./Extensions/PortfolioOptimiserPlotsExtDefinitions.jl")
end
