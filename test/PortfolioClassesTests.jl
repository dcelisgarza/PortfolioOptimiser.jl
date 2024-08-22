using CSV, TimeSeries, DataFrames, StatsBase, Statistics, LinearAlgebra, Test, Clarabel,
      PortfolioOptimiser

prices = TimeArray(CSV.File("./assets/stock_prices.csv"); timestamp = :date)

rf = 1.0329^(1 / 252) - 1
l = 2.0

@testset "Portfolio classes" begin
    portfolio = Portfolio2(; prices = prices,
                           solvers = Dict(:Clarabel => Dict(:solver => (Clarabel.Optimizer),
                                                            :params => Dict("verbose" => false))))

    asset_statistics2!(portfolio)

    portfolio.bl_mu = portfolio.mu
    portfolio.bl_cov = portfolio.cov

    obj = MinRisk()
    w1 = optimise2!(portfolio; obj = obj, class = Classic2())
    class = BL2(; type = 1)
    w2 = optimise2!(portfolio; obj = obj, class = class)
    class = BL2(; type = 2)
    w3 = optimise2!(portfolio; obj = obj, class = class)
    @test isapprox(w1.weights, w2.weights)
    @test isapprox(w1.weights, w3.weights)

    obj = SR(; rf = rf)
    w4 = optimise2!(portfolio; obj = obj, class = Classic2())
    class = BL2(; type = 1)
    w5 = optimise2!(portfolio; obj = obj, class = class)
    class = BL2(; type = 2)
    w6 = optimise2!(portfolio; obj = obj, class = class)
    @test isapprox(w4.weights, w5.weights)
    @test isapprox(w4.weights, w6.weights)

    portfolio.fm_mu = portfolio.mu
    portfolio.fm_cov = portfolio.cov
    portfolio.fm_returns = portfolio.returns

    obj = MinRisk()
    w7 = optimise2!(portfolio; obj = obj, class = Classic2())
    class = FM2(; type = 1)
    w8 = optimise2!(portfolio; obj = obj, class = class)
    class = FM2(; type = 2)
    w9 = optimise2!(portfolio; obj = obj, class = class)
    @test isapprox(w7.weights, w8.weights)
    @test isapprox(w7.weights, w9.weights)

    obj = SR(; rf = rf)
    w10 = optimise2!(portfolio; obj = obj, class = Classic2())
    class = FM2(; type = 1)
    w11 = optimise2!(portfolio; obj = obj, class = class)
    class = FM2(; type = 2)
    w12 = optimise2!(portfolio; obj = obj, class = class)
    @test isapprox(w10.weights, w11.weights)
    @test isapprox(w10.weights, w12.weights)

    portfolio.blfm_mu = portfolio.mu
    portfolio.blfm_cov = portfolio.cov

    obj = MinRisk()
    w13 = optimise2!(portfolio; obj = obj, class = Classic2())
    class = FM2(; type = 1)
    w14 = optimise2!(portfolio; obj = obj, class = class)
    class = FM2(; type = 2)
    w15 = optimise2!(portfolio; obj = obj, class = class)
    @test isapprox(w13.weights, w14.weights)
    @test isapprox(w13.weights, w15.weights)

    obj = SR(; rf = rf)
    w16 = optimise2!(portfolio; obj = obj, class = Classic2())
    class = FM2(; type = 1)
    w17 = optimise2!(portfolio; obj = obj, class = class)
    class = FM2(; type = 2)
    w18 = optimise2!(portfolio; obj = obj, class = class)
    @test isapprox(w16.weights, w17.weights)
    @test isapprox(w16.weights, w18.weights)
end
