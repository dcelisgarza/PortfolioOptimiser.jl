module PortfolioOptimiser

# Risk models
include("./RiskModels/RiskModels.jl")

# Expected returns
include("./ExpectedReturns/ExpectedReturns.jl")

# Optimisers
## Base functionality
include("./BaseOptimiser/BaseOptimiser.jl")
## Efficient frontier
include("./EfficientFrontierOptimiser/EfficientFrontierOptimiser.jl")
## Near optimal centering
include("./NearOptCenteringOptimiser/NearOptCenteringOptimiser.jl")
## Hierarchical risk parity
include("./HierarchicalOptimiser/HierarchicalOptimiser.jl")
## Critical line
include("./CriticalLineOptimiser/CriticalLineOptimiser.jl")
## Black-Litterman
include("./BlackLittermanOptimiser/BlackLittermanOptimiser.jl")

# Asset allocation
include("./AssetAllocation/AssetAllocation.jl")

# Version 2.0
using SparseArrays, Random, DataFrames, JuMP, Dates, Distributions, PyCall, Distances
include("./Portfolio/Definitions.jl")
include("./Portfolio/DBHTs.jl")
include("./Portfolio/OWA.jl")
include("./Portfolio/Codependence.jl")
include("./Portfolio/Aux_functions.jl")

include("./Portfolio/Portfolio.jl")
include("./Portfolio/HCPortfolio.jl")

include("./Portfolio/Asset_statistics.jl")
include("./Portfolio/Risk_measures.jl")
include("./Portfolio/Portfolio_risk_setup.jl")
include("./Portfolio/Portfolio_optim_setup.jl")

end
