using SafeTestsets

# @safetestset "Risk measures" begin
#     include("_RiskMeasureTests.jl")
# end

# @safetestset "Constraint functions" begin
#     include("ConstraintTests.jl")
# end

# @safetestset "Stats" begin
#     include("StatTests.jl")
# end

# @safetestset "Codependence and distance functions" begin
#     include("CodepDistTests.jl")
# end

# @safetestset "Log Tests" begin
#     include("LogTests.jl")
# end

# @safetestset "Clustering" begin
#     include("ClusteringTests.jl")
# end

# @safetestset "Portfolio allocation" begin
#     include("PortfolioAllocationTests.jl")
# end

# @safetestset "OWA weights" begin
#     include("OWAWeightsTests.jl")
# end

# @safetestset "Hierarchical optimisations" begin
#     include("HierarchicalOptimisationTests.jl")
# end

# @safetestset "Risk parity optimisations" begin
#     include("RiskParityTests.jl")
# end

# @safetestset "Worst case optimisations" begin
#     include("WorstCaseOptimisationTests.jl")
# end

# @safetestset "Traditional optimisations" begin
#     include("TraditionalOptimisationTests.jl")
# end

@safetestset "Misc" begin
    include("MiscTests.jl")
end

@safetestset "Entropic and Relativistic RM tests" begin
    include("EntrRelRMs.jl")
end

@safetestset "Portfolio Traditional Optimisation Constraints" begin
    include("PortfolioTradConstraintTests.jl")
end

@safetestset "Portfolio Classic RP OWA Optimisation" begin
    include("PortfolioClassicRPOWATests.jl")
end

@safetestset "Efficient Frontier" begin
    include("PortfolioEfficientFrontierTests.jl")
end

@safetestset "Portfolio Classic Traditional Optimisation" begin
    include("PortfolioClassicTradTests.jl")
end

@safetestset "Portfolio Classic WC Optimisation" begin
    include("PortfolioClassicWCTests.jl")
end

@safetestset "Asset Statistics" begin
    include("AssetStatisticsTests.jl")
end

@safetestset "Asset Allocation" begin
    include("AssetAllocationTests.jl")
end

@safetestset "BL Statistics" begin
    include("BLStatisticsTests.jl")
end

@safetestset "OWA Weights" begin
    include("OWAWeightsTests.jl")
end

@safetestset "Misc Statistics" begin
    include("MiscStatisticsTests.jl")
end

@safetestset "Factor Statistics" begin
    include("FactorStatisticsTests.jl")
end

@safetestset "WC Statistics" begin
    include("WCStatisticsTests.jl")
end

@safetestset "DBHT Clustering" begin
    include("DBHTClusteringTests.jl")
end

@safetestset "Constraint Functions" begin
    include("ConstraintTests.jl")
end

@safetestset "Portfolio Traditional Class Picking" begin
    include("PortfolioTradClassPickingTests.jl")
end

@safetestset "Portfolio Classic Traditional OWA Optimisation" begin
    include("PortfolioClassicTradOWATests.jl")
end

@safetestset "Portfolio Classic RP Optimisation" begin
    include("PortfolioClassicRPTests.jl")
end

@safetestset "HCPortfolio Optimisation" begin
    include("HCPortfolioOptimisationTests.jl")
end

@safetestset "HCPortfolio OWA Optimisation" begin
    include("HCPortfolioOWAOptimisationTests.jl")
end

@safetestset "Plotting" begin
    include("PlottingTests.jl")
end
