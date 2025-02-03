using SafeTestsets

@safetestset "Trad optimisation rm vector" begin
    include("TradOptimisationVectorTests.jl")
end

@safetestset "NOC optimisation" begin
    include("NOCOptimisationTests.jl")
end

@safetestset "Constrained optimisation" begin
    include("ConstrainedOptimisationTests.jl")
end

@safetestset "Entropic and Relativistic RM tests" begin
    include("EntrRelRMs.jl")
end

@safetestset "Constraint Functions" begin
    include("ConstraintTests.jl")
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

@safetestset "Type tests" begin
    include("TypeTests.jl")
end

@safetestset "OWA Weights" begin
    include("OWAWeightsTests.jl")
end

@safetestset "Misc Statistics" begin
    include("MiscStatisticsTests.jl")
end

@safetestset "Stats tests" begin
    include("StatsTest.jl")
end

@safetestset "Risk measure tests" begin
    include("RiskMeasureTests.jl")
end

@safetestset "Plotting" begin
    include("PlottingTests.jl")
end

@safetestset "Objective scalarisation tests" begin
    include("ObjectiveScalarisationTests.jl")
end

@safetestset "Portfolio classes" begin
    include("PortfolioClassesTests.jl")
end

@safetestset "HC optimisation" begin
    include("HCOptimisationTests.jl")
end

@safetestset "Trad optimisation" begin
    include("TradOptimisationTests.jl")
end

@safetestset "WC optimisation" begin
    include("WCOptimisationTests.jl")
end

@safetestset "Allocation tests" begin
    include("AllocationTests.jl")
end

@safetestset "RRB optimisation" begin
    include("RRBOptimisationTests.jl")
end

@safetestset "RB optimisation" begin
    include("RBOptimisationTests.jl")
end

@safetestset "Efficient frontier" begin
    include("EfficientFrontierTests.jl")
end
