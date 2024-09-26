module PortfolioOptimiser
using AverageShiftedHistograms, Clustering, DataFrames, Dates, Distances, Distributions,
      GLM, JuMP, LinearAlgebra, MultivariateStats, NearestCorrelationMatrix, Optim, Graphs,
      SimpleWeightedGraphs, PythonCall, Random, SmartAsserts, SparseArrays, Statistics,
      StatsBase, TimeSeries

# Types
## Risk measures
include("./RiskMeasures/Types/RiskMeasureTypes.jl")
## Parameter estimation
include("./ParameterEstimation/Types/ParameterEstimationTypes.jl")
## Constraints
include("./Constraints/Types/ConstraintTypes.jl")
## Optimisation
include("./Optimisation/Types/OptimisationTypes.jl")
## Portfolio
include("./Portfolio/Types/PortfolioTypes.jl")

# Functions
## Risk measures
include("./RiskMeasures/Functions/RiskMeasureFunctions.jl")
## Parameter estimation
include("./ParameterEstimation/Functions/ParameterEstimationFunctions.jl")
## Constraints
include("./Constraints/Functions/ConstraintFunctions.jl")
## Optimisation
include("./Optimisation/Functions/OptimisationFunctions.jl")
## Portfolio
include("./Portfolio/Functions/PortfolioFunctions.jl")

# Extensions
include("./Extensions/PortfolioOptimiserPlotsExtDefinitions.jl")
end
