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
## Constraints
include("./Constraints/Constraints.jl")
## Parameter estimation
include("./ParameterEstimation/Functions/ParameterEstimationFunctions.jl")
## Risk measures
include("./RiskMeasures/Functions/RiskMeasureFunctions.jl")
## Portfolio
include("./Portfolio/Functions/PortfolioFunctions.jl")
## Optimisation
include("./Optimisation/Functions/OptimisationFunctions.jl")

# Extensions
include("./Extensions/PortfolioOptimiserPlotsExtDefinitions.jl")
end
