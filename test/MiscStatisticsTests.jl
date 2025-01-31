using CSV, LinearAlgebra, PortfolioOptimiser, SparseArrays, Statistics, StatsBase, Test,
      TimeSeries

path = joinpath(@__DIR__, "assets/stock_prices.csv")
prices = TimeArray(CSV.File(path); timestamp = :date)

rf = 1.0329^(1 / 252) - 1
l = 2.0

@testset "Misc Statistics" begin
    portfolio = Portfolio(; prices = prices)
    asset_statistics!(portfolio)

    simret1 = PortfolioOptimiser.cov_returns(portfolio.cov; iters = 5, len = 100,
                                             seed = 123456789)
    simret2 = PortfolioOptimiser.cov_returns(portfolio.cov; iters = 2, len = 100,
                                             seed = 123456789)
    @test isapprox(cov(simret1), portfolio.cov)
    @test isapprox(cov(simret2), portfolio.cov)

    d1 = PortfolioOptimiser.elimination_matrix(1)
    d2 = PortfolioOptimiser.elimination_matrix(2)
    d3 = PortfolioOptimiser.elimination_matrix(3)
    d4 = PortfolioOptimiser.elimination_matrix(5)
    d5 = PortfolioOptimiser.elimination_matrix(7)
    d6 = PortfolioOptimiser.elimination_matrix(13)

    dt1 = sparse([1], [1], [1], 1, 1)
    dt2 = sparse([1, 2, 3], [1, 2, 4], [1, 1, 1], 3, 4)
    dt3 = sparse([1, 2, 3, 4, 5, 6], [1, 2, 3, 5, 6, 9], [1, 1, 1, 1, 1, 1], 6, 9)
    dt4 = sparse([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                 [1, 2, 3, 4, 5, 7, 8, 9, 10, 13, 14, 15, 19, 20, 25],
                 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 15, 25)
    dt5 = sparse([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                  22, 23, 24, 25, 26, 27, 28],
                 [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 17, 18, 19, 20, 21, 25, 26,
                  27, 28, 33, 34, 35, 41, 42, 49],
                 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                  1, 1, 1], 28, 49)
    dt6 = sparse([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                  22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
                  40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
                  58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75,
                  76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91],
                 [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22,
                  23, 24, 25, 26, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 43, 44, 45,
                  46, 47, 48, 49, 50, 51, 52, 57, 58, 59, 60, 61, 62, 63, 64, 65, 71, 72,
                  73, 74, 75, 76, 77, 78, 85, 86, 87, 88, 89, 90, 91, 99, 100, 101, 102,
                  103, 104, 113, 114, 115, 116, 117, 127, 128, 129, 130, 141, 142, 143, 155,
                  156, 169],
                 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 91, 169)
    @test isapprox(d1, dt1)
    @test isapprox(d2, dt2)
    @test isapprox(d3, dt3)
    @test isapprox(d4, dt4)
    @test isapprox(d5, dt5)
    @test isapprox(d6, dt6)

    l1 = PortfolioOptimiser.elimination_matrix(1)
    l2 = PortfolioOptimiser.elimination_matrix(2)
    l3 = PortfolioOptimiser.elimination_matrix(3)
    l4 = PortfolioOptimiser.elimination_matrix(5)
    l5 = PortfolioOptimiser.elimination_matrix(7)
    l6 = PortfolioOptimiser.elimination_matrix(13)

    lt1 = sparse([1], [1], [1], 1, 1)
    lt2 = sparse([1, 2, 3], [1, 2, 4], [1, 1, 1], 3, 4)
    lt3 = sparse([1, 2, 3, 4, 5, 6], [1, 2, 3, 5, 6, 9], [1, 1, 1, 1, 1, 1], 6, 9)
    lt4 = sparse([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                 [1, 2, 3, 4, 5, 7, 8, 9, 10, 13, 14, 15, 19, 20, 25],
                 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 15, 25)
    lt5 = sparse([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                  22, 23, 24, 25, 26, 27, 28],
                 [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 17, 18, 19, 20, 21, 25, 26,
                  27, 28, 33, 34, 35, 41, 42, 49],
                 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                  1, 1, 1], 28, 49)
    lt6 = sparse([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                  22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
                  40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
                  58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75,
                  76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91],
                 [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22,
                  23, 24, 25, 26, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 43, 44, 45,
                  46, 47, 48, 49, 50, 51, 52, 57, 58, 59, 60, 61, 62, 63, 64, 65, 71, 72,
                  73, 74, 75, 76, 77, 78, 85, 86, 87, 88, 89, 90, 91, 99, 100, 101, 102,
                  103, 104, 113, 114, 115, 116, 117, 127, 128, 129, 130, 141, 142, 143, 155,
                  156, 169],
                 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 91, 169)
    @test isapprox(l1, lt1)
    @test isapprox(l2, lt2)
    @test isapprox(l3, lt3)
    @test isapprox(l4, lt4)
    @test isapprox(l5, lt5)
    @test isapprox(l6, lt6)

    s1 = PortfolioOptimiser.summation_matrix(1)
    s2 = PortfolioOptimiser.summation_matrix(2)
    s3 = PortfolioOptimiser.summation_matrix(3)
    s4 = PortfolioOptimiser.summation_matrix(5)
    s5 = PortfolioOptimiser.summation_matrix(7)
    s6 = PortfolioOptimiser.summation_matrix(13)

    st1 = sparse([1], [1], [1.0], 1, 1)
    st2 = sparse([1, 2, 3], [1, 2, 4], [1.0, 2.0, 1.0], 3, 4)
    st3 = sparse([1, 2, 3, 4, 5, 6], [1, 2, 3, 5, 6, 9], [1.0, 2.0, 2.0, 1.0, 2.0, 1.0], 6,
                 9)
    st4 = sparse([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                 [1, 2, 3, 4, 5, 7, 8, 9, 10, 13, 14, 15, 19, 20, 25],
                 [1.0, 2.0, 2.0, 2.0, 2.0, 1.0, 2.0, 2.0, 2.0, 1.0, 2.0, 2.0, 1.0, 2.0,
                  1.0], 15, 25)
    st5 = sparse([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                  22, 23, 24, 25, 26, 27, 28],
                 [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 17, 18, 19, 20, 21, 25, 26,
                  27, 28, 33, 34, 35, 41, 42, 49],
                 [1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0, 2.0,
                  2.0, 2.0, 2.0, 1.0, 2.0, 2.0, 2.0, 1.0, 2.0, 2.0, 1.0, 2.0, 1.0], 28, 49)
    st6 = sparse([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                  22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
                  40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
                  58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75,
                  76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91],
                 [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22,
                  23, 24, 25, 26, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 43, 44, 45,
                  46, 47, 48, 49, 50, 51, 52, 57, 58, 59, 60, 61, 62, 63, 64, 65, 71, 72,
                  73, 74, 75, 76, 77, 78, 85, 86, 87, 88, 89, 90, 91, 99, 100, 101, 102,
                  103, 104, 113, 114, 115, 116, 117, 127, 128, 129, 130, 141, 142, 143, 155,
                  156, 169],
                 [1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0, 2.0,
                  2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0, 2.0, 2.0, 2.0, 2.0,
                  2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                  2.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0, 2.0, 2.0, 2.0, 2.0,
                  2.0, 2.0, 2.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0, 2.0, 2.0, 2.0, 2.0,
                  2.0, 1.0, 2.0, 2.0, 2.0, 2.0, 1.0, 2.0, 2.0, 2.0, 1.0, 2.0, 2.0, 1.0, 2.0,
                  1.0], 91, 169)
    @test isapprox(s1, st1)
    @test isapprox(s2, st2)
    @test isapprox(s3, st3)
    @test isapprox(s4, st4)
    @test isapprox(s5, st5)
    @test isapprox(s6, st6)

    N = rand([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
              73, 79, 83, 89, 97], 5)

    for i ∈ N
        d = PortfolioOptimiser.duplication_matrix(i)
        l = PortfolioOptimiser.elimination_matrix(i)
        s = PortfolioOptimiser.summation_matrix(i)
        d2, l2, s2 = PortfolioOptimiser.dup_elim_sum_matrices(i)

        @test isequal(d, d2)
        @test isequal(l, l2)
        @test isequal(s, s2)
    end

    A = rand(10, 25)
    @test_throws DimensionMismatch PortfolioOptimiser.block_vec_pq(A, 3, 5)
    @test_throws DimensionMismatch PortfolioOptimiser.block_vec_pq(A, 2, 3)
end
