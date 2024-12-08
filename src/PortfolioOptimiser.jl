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

# """
#     function_name(arg1::Type1, arg2::Type2; kwarg1::Type3 = default1, kwarg2::Type4 = default2)

# # Description
# Brief description of what the function does and its key features.
#   - Key feature or characteristic 1
#   - Key feature or characteristic 2
#   - Mathematical relationships if applicable: ``\\mathrm{equation}``
#   - Additional important notes

# See also: [`RelatedFunction1`](@ref), [`RelatedFunction2`](@ref)

# # Arguments
#   - `arg1::Type1`: description of first positional argument
#   - `arg2::Type2`: description of second positional argument

# # Keywords
#   - `kwarg1::Type3 = default1`: description of first keyword argument
#   - `kwarg2::Type4 = default2`: description of second keyword argument

# # Returns
#   - `ReturnType`: description of what is returned

# # Behaviour
#   - Important behavior note 1
#   - Important behavior note 2

# ## Validation
#   - Validation check 1 (e.g., `arg1 > 0`)
#   - Validation check 2 (e.g., `kwarg1 ∈ (0, 1)`)

# # Examples
# ```julia
# # Basic usage
# result = function_name(arg1, arg2)

# # Using keyword arguments
# result = function_name(arg1, arg2; kwarg1 = custom1, kwarg2 = custom2)
# ```
# """
