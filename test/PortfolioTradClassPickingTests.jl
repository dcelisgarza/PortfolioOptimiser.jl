using CSV, Clarabel, DataFrames, Graphs, HiGHS, JuMP, LinearAlgebra, OrderedCollections,
      Pajarito, PortfolioOptimiser, Statistics, Test, TimeSeries, Logging

Logging.disable_logging(Logging.Warn)
prices = TimeArray(CSV.File("./assets/stock_prices.csv"); timestamp = :date)

rf = 1.0329^(1 / 252) - 1
l = 2.0

@testset "Traditional Portfolio Classes" begin
    portfolio = Portfolio(; prices = prices,
                          solvers = Dict(:Clarabel => Dict(:solver => (Clarabel.Optimizer),
                                                           :params => Dict("verbose" => false,
                                                                           "max_step_fraction" => 0.75))))
    asset_statistics!(portfolio; calc_kurt = false)

    opt = OptimiseOpt(; hist = 5)
    w1 = optimise!(portfolio, opt)
    opt.class = :FM
    opt.hist = 1
    w2 = optimise!(portfolio, opt)
    portfolio.mu_fm = portfolio.mu
    opt.hist = 2
    w3 = optimise!(portfolio, opt)
    opt.hist = 3
    @test_throws AssertionError optimise!(portfolio, opt)

    portfolio.returns_fm = portfolio.returns
    portfolio.cov_fm = portfolio.cov
    opt.hist = 1
    w4 = optimise!(portfolio, opt)
    opt.hist = 2
    w5 = optimise!(portfolio, opt)

    portfolio = Portfolio(; prices = prices,
                          solvers = Dict(:Clarabel => Dict(:solver => (Clarabel.Optimizer),
                                                           :params => Dict("verbose" => false,
                                                                           "max_step_fraction" => 0.75))))
    asset_statistics!(portfolio; calc_kurt = false)
    opt.class = :Classic
    opt.hist = 5
    w6 = optimise!(portfolio, opt)
    opt.class = :BL
    opt.hist = 1
    @test_throws DimensionMismatch optimise!(portfolio, opt)
    portfolio.mu_bl = portfolio.mu
    opt.hist = 2
    w7 = optimise!(portfolio, opt)
    opt.hist = 3
    @test_throws AssertionError optimise!(portfolio, opt)

    portfolio.cov_bl = portfolio.cov
    opt.hist = 1
    w8 = optimise!(portfolio, opt)
    opt.hist = 2
    w9 = optimise!(portfolio, opt)

    portfolio = Portfolio(; prices = prices,
                          solvers = Dict(:Clarabel => Dict(:solver => (Clarabel.Optimizer),
                                                           :params => Dict("verbose" => false,
                                                                           "max_step_fraction" => 0.75))))
    asset_statistics!(portfolio; calc_kurt = false)

    opt.class = :Classic
    opt.hist = 5
    w10 = optimise!(portfolio, opt)
    opt.class = :BLFM
    opt.hist = 1
    w11 = optimise!(portfolio, opt)
    portfolio.mu_bl_fm = portfolio.mu
    opt.hist = 2
    w12 = optimise!(portfolio, opt)
    opt.hist = 3
    @test_throws DimensionMismatch optimise!(portfolio, opt)

    portfolio.cov_bl_fm = portfolio.cov
    portfolio.returns_fm = portfolio.returns
    opt.hist = 1
    w13 = optimise!(portfolio, opt)
    opt.hist = 2
    w14 = optimise!(portfolio, opt)
    opt.hist = 3
    @test_throws DimensionMismatch optimise!(portfolio, opt)
    portfolio.cov_fm = portfolio.cov
    opt.hist = 3
    w15 = optimise!(portfolio, opt)

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
