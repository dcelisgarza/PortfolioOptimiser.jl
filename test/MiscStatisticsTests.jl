using CovarianceEstimation, CSV, HiGHS, LinearAlgebra, OrderedCollections,
      PortfolioOptimiser, SparseArrays, Statistics, StatsBase, Test, TimeSeries, Logging

Logging.disable_logging(Logging.Warn)

prices = TimeArray(CSV.File("./assets/stock_prices.csv"); timestamp = :date)

rf = 1.0329^(1 / 252) - 1
l = 2.0

@testset "Misc Statistics" begin
    portfolio = Portfolio(; prices = prices)
    asset_statistics!(portfolio; calc_kurt = false)

    simret1 = PortfolioOptimiser.cov_returns(portfolio.cov; iters = 5, len = 100,
                                             seed = 123456789)
    simret2 = PortfolioOptimiser.cov_returns(portfolio.cov; iters = 2, len = 100,
                                             seed = 123456789)

    sm1 = PortfolioOptimiser.summation_matrix(1)
    sm2 = PortfolioOptimiser.summation_matrix(2)
    sm3 = PortfolioOptimiser.summation_matrix(3)
    sm4 = PortfolioOptimiser.summation_matrix(5)
    sm5 = PortfolioOptimiser.summation_matrix(7)
    sm6 = PortfolioOptimiser.summation_matrix(13)

    smt1 = sparse([1], [1], [1.0], 1, 1)
    smt2 = sparse([1, 2, 3], [1, 2, 4], [1.0, 2.0, 1.0], 3, 4)
    smt3 = sparse([1, 2, 3, 4, 5, 6], [1, 2, 3, 5, 6, 9], [1.0, 2.0, 2.0, 1.0, 2.0, 1.0], 6,
                  9)
    smt4 = sparse([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                  [1, 2, 3, 4, 5, 7, 8, 9, 10, 13, 14, 15, 19, 20, 25],
                  [1.0, 2.0, 2.0, 2.0, 2.0, 1.0, 2.0, 2.0, 2.0, 1.0, 2.0, 2.0, 1.0, 2.0,
                   1.0], 15, 25)
    smt5 = sparse([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                   21, 22, 23, 24, 25, 26, 27, 28],
                  [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 17, 18, 19, 20, 21, 25, 26,
                   27, 28, 33, 34, 35, 41, 42, 49],
                  [1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0,
                   2.0, 2.0, 2.0, 2.0, 1.0, 2.0, 2.0, 2.0, 1.0, 2.0, 2.0, 1.0, 2.0, 1.0],
                  28, 49)
    smt6 = sparse([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                   21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38,
                   39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56,
                   57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74,
                   75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91],
                  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21,
                   22, 23, 24, 25, 26, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 43, 44,
                   45, 46, 47, 48, 49, 50, 51, 52, 57, 58, 59, 60, 61, 62, 63, 64, 65, 71,
                   72, 73, 74, 75, 76, 77, 78, 85, 86, 87, 88, 89, 90, 91, 99, 100, 101,
                   102, 103, 104, 113, 114, 115, 116, 117, 127, 128, 129, 130, 141, 142,
                   143, 155, 156, 169],
                  [1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0,
                   2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0, 2.0, 2.0,
                   2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                   2.0, 2.0, 2.0, 2.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0,
                   2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                   1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0, 2.0, 2.0, 2.0, 2.0, 1.0, 2.0, 2.0,
                   2.0, 1.0, 2.0, 2.0, 1.0, 2.0, 1.0], 91, 169)

    @test isapprox(cov(simret1), portfolio.cov)
    @test isapprox(cov(simret2), portfolio.cov)
    @test isapprox(sm1, smt1)
    @test isapprox(sm2, smt2)
    @test isapprox(sm3, smt3)
    @test isapprox(sm4, smt4)
    @test isapprox(sm5, smt5)
    @test isapprox(sm6, smt6)
end
