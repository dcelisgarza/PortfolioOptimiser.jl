using SafeTestsets

@safetestset "Expected returns" begin
    include("ExpectedReturnsTests.jl")
end

@safetestset "Risk models" begin
    include("RiskModelsTests.jl")
end

@safetestset "Black Litterman" begin
    include("BlackLittermanTests.jl")
end

@safetestset "Efficient Frontier" begin
    include("EfficientFrontierTests.jl")
end

@safetestset "Hierarchical Risk Parity" begin
    include("HRPOptTests.jl")
end

@safetestset "Critical Line" begin
    include("CriticalLineTests.jl")
end

@safetestset "Custom optimiser" begin
    include("CustomOptimiserTests.jl")
end