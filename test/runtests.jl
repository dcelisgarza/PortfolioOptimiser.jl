using SafeTestsets

@safetestset "Entropic and Relativistic RM tests" begin
    include("EntrRelRMs.jl")
end

@safetestset "Asset Statistics" begin
    include("AssetStatisticsTests2.jl")
    include("AssetStatisticsTests1.jl")
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

@safetestset "Plotting" begin
    include("PlottingTests.jl")
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

@safetestset "Portfolio Classic RP Optimisation" begin
    include("PortfolioClassicRPTests.jl")
end

@safetestset "Portfolio Classic RP OWA Optimisation" begin
    include("PortfolioClassicRPOWATests.jl")
end

@safetestset "HCPortfolio OWA Optimisation" begin
    include("HCPortfolioOWAOptimisationTests.jl")
end

@safetestset "HCPortfolio Optimisation" begin
    include("HCPortfolioOptimisationTests.jl")
end

@safetestset "Portfolio Classic Traditional Optimisation" begin
    include("PortfolioClassicTradTests.jl")
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
