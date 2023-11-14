module PortfolioOptimiser

# # Risk models
# include("./RiskModels/RiskModels.jl")

# # Expected returns
# include("./ExpectedReturns/ExpectedReturns.jl")

# # Optimisers
# ## Base functionality
# include("./BaseOptimiser/BaseOptimiser.jl")
# ## Efficient frontier
# include("./EfficientFrontierOptimiser/EfficientFrontierOptimiser.jl")
# ## Near optimal centering
# include("./NearOptCenteringOptimiser/NearOptCenteringOptimiser.jl")
# ## Hierarchical risk parity
# include("./HierarchicalOptimiser/HierarchicalOptimiser.jl")
# ## Critical line
# include("./CriticalLineOptimiser/CriticalLineOptimiser.jl")
# ## Black-Litterman
# include("./BlackLittermanOptimiser/BlackLittermanOptimiser.jl")

# # Asset allocation
# include("./AssetAllocation/AssetAllocation.jl")

# Version 2.0
using Clustering,
    DataFrames,
    Dates,
    Distances,
    Distributions,
    GLM,
    JuMP,
    LinearAlgebra,
    MultivariateStats,
    PyCall,
    Random,
    SparseArrays,
    Statistics,
    StatsBase,
    NearestCorrelationMatrix,
    Optim,
    AverageShiftedHistograms

include("Definitions.jl")
include("DBHTs.jl")
include("Constraint_functions.jl")
include("OWA.jl")
include("Codependence.jl")
include("Aux_functions.jl")

include("Portfolio.jl")
include("Portfolio_risk_setup.jl")
include("Portfolio_optim_setup.jl")

include("HCPortfolio.jl")
include("HCPortfolio_optim_setup.jl")

include("Asset_statistics.jl")
include("Risk_measures.jl")
include("Portfolio_allocation.jl")

include("../old_src/BaseOptimiser/BaseOptimiser.jl")
include("../old_src/RiskModels/RiskModels.jl")
include("../old_src/ExpectedReturns/ExpectedReturns.jl")
include("../old_src/AssetAllocation/AssetAllocation.jl")

end
