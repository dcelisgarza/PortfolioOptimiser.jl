# Copywrite (c) 2025
# Author: Daniel Celis Garza <daniel.celis.garza@gmail.com>
# SPDX-License-Identifier: MIT

module PortfolioOptimiser
using AverageShiftedHistograms, Clustering, DataFrames, Dates, Distances, Distributions,
      GLM, JuMP, LinearAlgebra, MultivariateStats, NearestCorrelationMatrix, Optim, Graphs,
      SimpleWeightedGraphs, PythonCall, Random, SmartAsserts, SparseArrays, Statistics,
      StatsBase, TimeSeries

# Turn readme into PortfolioOptimiser's docs.
@doc let
    path = joinpath(dirname(@__DIR__), "README.md")
    include_dependency(path)
    read(path, String)
end PortfolioOptimiser

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
## Type utility functions
include("./Utils/Utils.jl")

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

# https://en.wikipedia.org/wiki/The_Power_of_10:_Rules_for_Developing_Safety-Critical_Code
# https://www.youtube.com/watch?v=WwkuAqObplU
