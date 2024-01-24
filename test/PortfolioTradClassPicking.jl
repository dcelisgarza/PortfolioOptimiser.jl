using COSMO, CSV, Clarabel, DataFrames, Graphs, HiGHS, JuMP, LinearAlgebra,
      OrderedCollections, Pajarito, PortfolioOptimiser, Statistics, Test, TimeSeries,
      Logging, GLPK

prices = TimeArray(CSV.File("./assets/stock_prices.csv"); timestamp = :date)

rf = 1.0329^(1 / 252) - 1
l = 2.0

@testset "Traditional Portfolio Classes" begin
    portfolio = Portfolio(; prices = prices,
                          solvers = Dict(:Clarabel => Dict(:solver => (Clarabel.Optimizer),
                                                           :params => Dict("verbose" => false,
                                                                           "max_step_fraction" => 0.75))))
    asset_statistics!(portfolio; calc_kurt = false)

    w1 = optimise!(portfolio; hist = 5)
    w2 = optimise!(portfolio; class = :FM, hist = 1)
    portfolio.mu_fm = portfolio.mu
    w3 = optimise!(portfolio; class = :FM, hist = 2)
    @test_throws AssertionError optimise!(portfolio, class = :FM, hist = 3)

    portfolio.returns_fm = portfolio.returns
    portfolio.cov_fm = portfolio.cov
    w4 = optimise!(portfolio; class = :FM, hist = 1)
    w5 = optimise!(portfolio; class = :FM, hist = 2)

    portfolio = Portfolio(; prices = prices,
                          solvers = Dict(:Clarabel => Dict(:solver => (Clarabel.Optimizer),
                                                           :params => Dict("verbose" => false,
                                                                           "max_step_fraction" => 0.75))))
    asset_statistics!(portfolio; calc_kurt = false)

    w6 = optimise!(portfolio; hist = 5)
    @test_throws DimensionMismatch optimise!(portfolio, class = :BL, hist = 1)
    portfolio.mu_bl = portfolio.mu
    w7 = optimise!(portfolio; class = :BL, hist = 2)
    @test_throws AssertionError optimise!(portfolio, class = :BL, hist = 3)

    portfolio.cov_bl = portfolio.cov
    w8 = optimise!(portfolio; class = :BL, hist = 1)
    w9 = optimise!(portfolio; class = :BL, hist = 2)

    portfolio = Portfolio(; prices = prices,
                          solvers = Dict(:Clarabel => Dict(:solver => (Clarabel.Optimizer),
                                                           :params => Dict("verbose" => false,
                                                                           "max_step_fraction" => 0.75))))
    asset_statistics!(portfolio; calc_kurt = false)

    w10 = optimise!(portfolio; hist = 5)
    w11 = optimise!(portfolio; class = :BLFM, hist = 1)
    portfolio.mu_bl_fm = portfolio.mu
    w12 = optimise!(portfolio; class = :BLFM, hist = 2)
    @test_throws DimensionMismatch optimise!(portfolio, class = :BLFM, hist = 3)

    portfolio.cov_bl_fm = portfolio.cov
    portfolio.returns_fm = portfolio.returns
    w13 = optimise!(portfolio; class = :BLFM, hist = 1)
    w14 = optimise!(portfolio; class = :BLFM, hist = 2)
    @test_throws DimensionMismatch optimise!(portfolio, class = :BLFM, hist = 3)
    portfolio.cov_fm = portfolio.cov
    w15 = optimise!(portfolio; class = :BLFM, hist = 3)

    @test isempty(w2)
    @test isapprox(w1.weights, w3.weights)
    @test isapprox(w1.weights, w4.weights)
    @test isapprox(w1.weights, w5.weights)
    @test isapprox(w6.weights, w7.weights)
    @test isapprox(w6.weights, w8.weights)
    @test isapprox(w6.weights, w9.weights)
    @test isempty(w11)
    @test isapprox(w10.weights, w12.weights)
    @test isapprox(w10.weights, w13.weights)
    @test isapprox(w10.weights, w14.weights)
    @test isapprox(w10.weights, w15.weights)
end
