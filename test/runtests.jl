using SafeTestsets

# @safetestset "Misc" begin
#     include("MiscTests.jl")
# end

# @safetestset "Risk measures" begin
#     include("RiskMeasureTests.jl")
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
@safetestset "HCPortfolio Optimisation Tests" begin
    include("HCPortfolioOptimisationTests.jl")
end

@safetestset "Portfolio Classic Traditional Optimisation" begin
    include("PortfolioClassicTradTests.jl")
end

@safetestset "Efficient Frontier" begin
    include("PortfolioEfficientFrontierTests.jl")
end
