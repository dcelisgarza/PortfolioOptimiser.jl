using SafeTestsets

@safetestset "Risk functions" begin
    include("RiskMeasureTests.jl")
end

@safetestset "Codependence and distance functions" begin
    include("CodepDistTests.jl")
end

@safetestset "Clustering" begin
    include("ClusteringTests.jl")
end

@safetestset "Hierarchical optimisations" begin
    include("HierarchicalOptimisationTests.jl")
end

@safetestset "Traditional optimisations" begin
    include("TraditionalOptimisationTests.jl")
end

# @safetestset "Expected returns" begin
#     include("ExpectedReturnsTests.jl")
# end

# @safetestset "Risk models" begin
#     include("RiskModelsTests.jl")
# end

# @safetestset "Black Litterman" begin
#     include("BlackLittermanTests.jl")
# end

# @safetestset "Efficient Frontier" begin
#     include("EfficientFrontierTests.jl")
# end

# @safetestset "Hierarchical Risk Parity" begin
#     include("HRPOptTests.jl")
# end

# @safetestset "Critical Line" begin
#     include("CriticalLineTests.jl")
# end

# @safetestset "Custom optimiser" begin
#     include("CustomOptimiserTests.jl")
# end