using Test, PortfolioOptimiser

@testset "Miscelaneous" begin
    d = PortfolioOptimiser.duplication_matrix(11)
    l = PortfolioOptimiser.elimination_matrix(11)
    @test PortfolioOptimiser.summation_matrix(11) == transpose(d) * d * l
end
