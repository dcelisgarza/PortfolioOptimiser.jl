using SafeTestsets

@safetestset "Constrained optimisation" begin
    include("ConstrainedOptimisationTests.jl")
end

@safetestset "HC optimisation" begin
    include("HCOptimisationTests.jl")
end

@safetestset "RRP optimisation" begin
    include("RRPOptimisationTests.jl")
end

@safetestset "RP optimisation" begin
    include("RPOptimisationTests.jl")
end

@safetestset "NOC optimisation" begin
    include("NOCOptimisationTests.jl")
end

@safetestset "Objective scalarisation tests" begin
    include("ObjectiveScalarisationTests.jl")
end

@safetestset "Constraint Functions" begin
    include("ConstraintTests.jl")
end

@safetestset "Stats tests" begin
    include("StatsTest.jl")
end

@safetestset "Allocation tests" begin
    include("AllocationTests.jl")
end

@safetestset "Portfolio classes" begin
    include("PortfolioClassesTests.jl")
end

@safetestset "Factor Stats tests" begin
    include("FactorStatsTest.jl")
end

@safetestset "WC Stats tests" begin
    include("WCStatsTest.jl")
end

@safetestset "Clustering tests" begin
    include("ClusteringTests.jl")
end

@safetestset "DBHT Clustering" begin
    include("DBHTClusteringTests.jl")
end

@safetestset "Network constraint tests" begin
    include("NetworkConstraintTests.jl")
end

@safetestset "BL Stats tests" begin
    include("BLStatsTests.jl")
end

@safetestset "Trad optimisation" begin
    include("TradOptimisationTests.jl")
end

@safetestset "Risk measure tests" begin
    include("RiskMeasureTests.jl")
end

@safetestset "Type tests" begin
    include("TypeTests.jl")
end

@safetestset "WC optimisation" begin
    include("WCOptimisationTests.jl")
end

@safetestset "Trad optimisation rm vector" begin
    include("TradOptimisationVectorTests.jl")
end

@safetestset "OWA Weights" begin
    include("OWAWeightsTests.jl")
end

@safetestset "Plotting" begin
    include("PlottingTests.jl")
end

@safetestset "Efficient frontier" begin
    include("EfficientFrontierTests.jl")
end

@safetestset "Entropic and Relativistic RM tests" begin
    include("EntrRelRMs.jl")
end

@safetestset "Misc Statistics" begin
    include("MiscStatisticsTests.jl")
end
