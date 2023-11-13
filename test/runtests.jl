using SafeTestsets

@safetestset "Misc" begin
    include("MiscTests.jl")
end

@safetestset "Risk measures" begin
    include("RiskMeasureTests.jl")
end

@safetestset "Constraint functions" begin
    include("ConstraintTests.jl")
end

@safetestset "Asset stats" begin
    include("AssetStatTests.jl")
end

@safetestset "Codependence and distance functions" begin
    include("CodepDistTests.jl")
end

@safetestset "Clustering" begin
    include("ClusteringTests.jl")
end

@safetestset "Portfolio allocation" begin
    include("PortfolioAllocationTests.jl")
end

@safetestset "OWA weights" begin
    include("OWAWeightsTests.jl")
end

@safetestset "Hierarchical optimisations" begin
    include("HierarchicalOptimisationTests.jl")
end

@safetestset "Risk parity optimisations" begin
    include("RiskParityTests.jl")
end

@safetestset "Worst case optimisations" begin
    include("WorstCaseOptimisationTests.jl")
end

@safetestset "Traditional optimisations" begin
    include("TraditionalOptimisationTests.jl")
end
