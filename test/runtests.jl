using SafeTestsets

@safetestset "Trad optimisation rm vectors" begin
    include("TradOptimisationVectorTests.jl")
end

@safetestset "Trad optimisation v2" begin
    include("TradOptimisationTests.jl")
end

@safetestset "Risk measure tests V2" begin
    include("RiskMeasureTests.jl")
end

@safetestset "Network constraint tests" begin
    include("NetworkConstraintTests.jl")
end

@safetestset "Clustering tests V2" begin
    include("ClusteringTests.jl")
end

@safetestset "BL Stats V2 tests" begin
    include("BLStatsTests.jl")
end

@safetestset "Stats V2 tests" begin
    include("StatsTest.jl")
end

@safetestset "WC Stats V2 tests" begin
    include("WCStatsTest.jl")
end

@safetestset "Factor Stats V2 tests" begin
    include("FactorStatsTest.jl")
end

@safetestset "Entropic and Relativistic RM tests" begin
    include("EntrRelRMs.jl")
end

@safetestset "Constraint Functions" begin
    include("ConstraintTests.jl")
end

@safetestset "Type tests" begin
    include("TypeTests.jl")
end

@safetestset "Misc Statistics" begin
    include("MiscStatisticsTests.jl")
end

@safetestset "Allocation tests" begin
    include("AllocationTests.jl")
end

@safetestset "OWA Weights" begin
    include("OWAWeightsTests.jl")
end

@safetestset "DBHT Clustering" begin
    include("DBHTClusteringTests.jl")
end

@safetestset "Efficient frontier v2" begin
    include("EfficientFrontierTests.jl")
end

@safetestset "Near Optimal Centering" begin
    include("NearOptimalCenteringTests.jl")
end

@safetestset "HC Optimistaion" begin
    include("HCOptimisationTests.jl")
end

@safetestset "Constraints" begin
    include("ConstrainedOptimisationTests.jl")
end

@safetestset "Portfolio classes" begin
    include("PortfolioClassesTests.jl")
end

@safetestset "RP optimisation" begin
    include("RPOptimisationTests.jl")
end

@safetestset "RRP Optimisation" begin
    include("RRPOptimisationTests.jl")
end

@safetestset "WC Optimisation" begin
    include("WCOptimisationTests.jl")
end

@safetestset "Plotting" begin
    include("PlottingTests.jl")
end
