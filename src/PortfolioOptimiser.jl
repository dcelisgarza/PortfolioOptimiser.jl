module PortfolioOptimiser
using AverageShiftedHistograms,
    Clustering,
    DataFrames,
    Dates,
    Distances,
    Distributions,
    GLM,
    JuMP,
    LinearAlgebra,
    MultivariateStats,
    NearestCorrelationMatrix,
    Optim,
    StatsPlots,
    PyCall,
    Random,
    SparseArrays,
    Statistics,
    StatsBase

include("Definitions.jl")
include("DBHTs.jl") # Type checked.
include("Constraint_functions.jl") # Type checked.
include("OWA.jl") # Type checked.
include("Codependence.jl")
include("Aux_functions.jl")

include("Portfolio.jl")
include("HCPortfolio.jl")

include("Asset_statistics.jl")
include("Risk_measures.jl")
include("Portfolio_risk_setup.jl")
include("Portfolio_optim_setup.jl")
include("HCPortfolio_optim_setup.jl")
include("Portfolio_allocation.jl")

include("Plotting.jl")

end
