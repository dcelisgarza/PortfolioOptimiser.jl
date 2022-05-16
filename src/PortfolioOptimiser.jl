module PortfolioOptimiser
# Expected returns
include("./ExpectedReturns/ExpectedReturns.jl")

# Risk models
include("./RiskModels/RiskModels.jl")

# Optimisers
## Base functionality
include("./BaseOptimiser/BaseOptimiser.jl")
## Efficient frontier
include("./EfficientFrontierOptimiser/EfficientFrontierOptimiser.jl")
## Hierarchical risk parity
include("./HierarchicalOptimiser/HierarchicalOptimiser.jl")
## Critical line
include("./CriticalLineOptimiser/CriticalLineOptimiser.jl")
## Black-Litterman
include("./BlackLittermanOptimiser/BlackLittermanOptimiser.jl")

# Asset allocation
include("./AssetAllocation/AssetAllocation.jl")

end
