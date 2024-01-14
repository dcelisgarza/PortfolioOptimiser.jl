using SafeTestsets

# @safetestset "Misc" begin
#     include("MiscTests.jl")
# end

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

# @safetestset "Plotting" begin
#     include("PlottingTests.jl")
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
@safetestset "Constraint Functions" begin
    include("ConstraintTests.jl")
end

@safetestset "Asset Statistics" begin
    include("AssetStatisticsTests.jl")
end

@safetestset "Misc Statistics" begin
    include("MiscStatisticsTests.jl")
end

@safetestset "BL Statistics" begin end

@safetestset "Factor Statistics" begin
    include("FactorStatisticsTests.jl")
end

@safetestset "WC Statistics" begin
    include("WCStatisticsTests.jl")
end

@safetestset "DBHT Clustering" begin
    include("DBHTClusteringTests.jl")
end

@safetestset "Portfolio Classic WC Optimisation" begin
    include("PortfolioClassicWCTests.jl")
end

@safetestset "Portfolio Classic RP OWA Optimisation" begin
    include("PortfolioClassicRPOWATests.jl")
end

@safetestset "Portfolio Classic RP Optimisation" begin
    include("PortfolioClassicRPTests.jl")
end

@safetestset "Portfolio Classic Traditional Optimisation" begin
    include("PortfolioClassicTradTests.jl")
end

@safetestset "Efficient Frontier" begin
    include("PortfolioEfficientFrontierTests.jl")
end

@safetestset "Portfolio Classic Traditional OWA Optimisation" begin
    include("PortfolioClassicTradOWATests.jl")
end

@safetestset "HCPortfolio Optimisation" begin
    include("HCPortfolioOptimisationTests.jl")
end

@safetestset "HCPortfolio OWA Optimisation" begin
    include("HCPortfolioOWAOptimisationTests.jl")
end
