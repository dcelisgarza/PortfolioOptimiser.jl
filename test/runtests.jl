using SafeTestsets

@safetestset "Entropic and Relativistic RM tests" begin
    include("EntrRelRMs.jl")
end

@safetestset "Trad optimisation rm vectors" begin
    include("TradOptimisationVectorTests_v2.jl")
end

@safetestset "Efficient frontier v2" begin
    include("EfficientFrontierTests_v2.jl")
end

@safetestset "Plotting v2" begin
    include("PlottingTests_v2.jl")
end

@safetestset "Trad optimisation v2" begin
    include("TradOptimisationTests_v2.jl")
end

@safetestset "Risk measure tests V2" begin
    include("RiskMeasureTests_v2.jl")
end

@safetestset "Network constraint tests" begin
    include("NetworkConstraintTests.jl")
end

@safetestset "Clustering tests V2" begin
    include("ClusteringTests_v2.jl")
end

@safetestset "BL Stats V2 tests" begin
    include("BLStatsTests_v2.jl")
end

@safetestset "Stats V2 tests" begin
    include("StatsTest_v2_1.jl")
    include("StatsTest_v2_2.jl")
end

@safetestset "WC Stats V2 tests" begin
    include("WCStatsTest_v2.jl")
end

@safetestset "Factor Stats V2 tests" begin
    include("FactorStatsTest_v2.jl")
end

@safetestset "HCPortfolio Optimisation" begin
    include("HCPortfolioOptimisationTests.jl")
end

@safetestset "Asset Statistics" begin
    include("AssetStatisticsTests1.jl")
    include("AssetStatisticsTests2.jl")
end

@safetestset "Portfolio Classic Traditional Optimisation" begin
    include("PortfolioClassicTradTests.jl")
end

@safetestset "Misc" begin
    include("MiscTests.jl")
end

@safetestset "Portfolio Traditional Optimisation Constraints" begin
    include("PortfolioTradConstraintTests.jl")
end

@safetestset "Constraint Functions" begin
    include("ConstraintTests.jl")
end

@safetestset "Efficient Frontier" begin
    include("PortfolioEfficientFrontierTests.jl")
end

@safetestset "Asset Allocation" begin
    include("AssetAllocationTests.jl")
end

@safetestset "BL Statistics" begin
    include("BLStatisticsTests.jl")
end

@safetestset "Factor Statistics" begin
    include("FactorStatisticsTests.jl")
end

@safetestset "OWA Weights" begin
    include("OWAWeightsTests.jl")
end

@safetestset "Misc Statistics" begin
    include("MiscStatisticsTests.jl")
end

@safetestset "DBHT Clustering" begin
    include("DBHTClusteringTests.jl")
end

@safetestset "Portfolio Traditional Class Picking" begin
    include("PortfolioTradClassPickingTests.jl")
end

@safetestset "Portfolio Classic RP OWA Optimisation" begin
    include("PortfolioClassicRPOWATests.jl")
end

@safetestset "HCPortfolio OWA Optimisation" begin
    include("HCPortfolioOWAOptimisationTests.jl")
end

@safetestset "Portfolio Classic Traditional OWA Optimisation" begin
    include("PortfolioClassicTradOWATests.jl")
end

@safetestset "Portfolio Classic WC Optimisation" begin
    include("PortfolioClassicWCTests.jl")
end

@safetestset "WC Statistics" begin
    include("WCStatisticsTests.jl")
end

@safetestset "Portfolio Classic RP Optimisation" begin
    include("PortfolioClassicRPTests.jl")
end

@safetestset "Plotting" begin
    include("PlottingTests.jl")
end
