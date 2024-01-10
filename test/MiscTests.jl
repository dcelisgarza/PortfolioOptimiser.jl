using Test,
      PortfolioOptimiser,
      DataFrames,
      TimeSeries,
      CSV,
      Dates,
      ECOS,
      SCS,
      Clarabel,
      COSMO,
      OrderedCollections,
      LinearAlgebra,
      StatsBase,
      HiGHS,
      Logging

@testset "Miscelaneous" begin
    d = PortfolioOptimiser.duplication_matrix(11)
    l = PortfolioOptimiser.elimination_matrix(11)
    @test PortfolioOptimiser.summation_matrix(11) == transpose(d) * d * l

    a = ClusterNode(1, nothing, nothing, 1, 1)
    b = ClusterNode(1, nothing, nothing, 1, 1)
    c = ClusterNode(1, nothing, nothing, 2, 1)
    @test a == b
    @test a < c
    @test c > b
end

@testset "Portfolio setfield" begin end
