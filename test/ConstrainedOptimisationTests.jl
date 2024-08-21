using CSV, TimeSeries, DataFrames, StatsBase, Statistics, LinearAlgebra, Test, Clarabel,
      PortfolioOptimiser

prices = TimeArray(CSV.File("./assets/stock_prices.csv"); timestamp = :date)
rf = 1.0329^(1 / 252) - 1
l = 2.0

@testset "Rebalance constraints" begin
    portfolio = Portfolio2(; prices = prices,
                           solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                            :params => Dict("verbose" => false,
                                                                            "max_step_fraction" => 0.7))))
    asset_statistics2!(portfolio)

    w1 = optimise2!(portfolio; obj = MinRisk())
    r1 = calc_risk(portfolio)
    ret1 = dot(portfolio.mu, w1.weights)
    sr1 = sharpe_ratio(portfolio)

    w2 = optimise2!(portfolio; obj = Util(; l = l))
    r2 = calc_risk(portfolio)
    ret2 = dot(portfolio.mu, w2.weights)
    sr2 = sharpe_ratio(portfolio)

    w3 = optimise2!(portfolio; obj = SR(; rf = rf))
    r3 = calc_risk(portfolio)
    ret3 = dot(portfolio.mu, w3.weights)
    sr3 = sharpe_ratio(portfolio)

    w4 = optimise2!(portfolio; obj = MaxRet())
    r4 = calc_risk(portfolio)
    ret4 = dot(portfolio.mu, w4.weights)
    sr4 = sharpe_ratio(portfolio)

    @test r1 < r3 < r2 < r4
    @test ret1 < ret3 < ret2 < ret4
    @test sr1 < sr4 < sr2 < sr3

    portfolio.rebalance = 0
    portfolio.rebalance_weights = w3.weights
    w5 = optimise2!(portfolio; obj = MinRisk())
    @test isapprox(w1.weights, w5.weights)
    portfolio.rebalance_weights = w1.weights
    w6 = optimise2!(portfolio; obj = Util(; l = l))
    w7 = optimise2!(portfolio; obj = SR(; rf = rf))
    w8 = optimise2!(portfolio; obj = MaxRet())
    @test isapprox(w2.weights, w6.weights)
    @test isapprox(w3.weights, w7.weights)
    @test isapprox(w4.weights, w8.weights)

    portfolio.rebalance = 1e10
    portfolio.rebalance_weights = w3.weights
    w9 = optimise2!(portfolio; obj = MinRisk())
    @test isapprox(w9.weights, w3.weights)
    portfolio.rebalance_weights = w1.weights
    w10 = optimise2!(portfolio; obj = Util(; l = l))
    w11 = optimise2!(portfolio; obj = SR(; rf = rf))
    w12 = optimise2!(portfolio; obj = MaxRet())
    @test isapprox(w10.weights, w1.weights)
    @test isapprox(w11.weights, w1.weights)
    @test isapprox(w12.weights, w1.weights)

    portfolio.rebalance = 1e-4
    portfolio.rebalance_weights = w3.weights
    w13 = optimise2!(portfolio; obj = MinRisk())
    @test !isapprox(w13.weights, w1.weights)
    @test !isapprox(w13.weights, w3.weights)
    portfolio.rebalance_weights = w1.weights
    w14 = optimise2!(portfolio; obj = Util(; l = l))
    w15 = optimise2!(portfolio; obj = SR(; rf = rf))
    w16 = optimise2!(portfolio; obj = MaxRet())
    @test !isapprox(w14.weights, w1.weights)
    @test !isapprox(w14.weights, w2.weights)
    @test !isapprox(w15.weights, w1.weights)
    @test !isapprox(w15.weights, w3.weights)
    @test !isapprox(w16.weights, w1.weights)
    @test !isapprox(w16.weights, w4.weights)

    portfolio.rebalance = [0.0005174248858061537, 0.0001378289720696607,
                           1.182008035855453e-5, 0.0009118233964947257,
                           0.0008043804574686568, 0.0005568104999737413,
                           0.0001433167617425195, 0.0008152431443894213,
                           0.0006805053356229013, 8.922295760840915e-5,
                           0.0008525847915972609, 0.0009046977862414844,
                           0.0009820771255260512, 0.0005494961009926494,
                           3.971977944267568e-5, 0.0006942164994964002,
                           0.000742647266054625, 0.0004077250418932119,
                           0.00031612114608380824, 0.00028833648463458153]
    portfolio.rebalance_weights = w3.weights
    w13 = optimise2!(portfolio; obj = MinRisk())
    @test !isapprox(w13.weights, w3.weights)
    @test !isapprox(w13.weights, w1.weights)
    portfolio.rebalance_weights = w1.weights
    w14 = optimise2!(portfolio; obj = Util(; l = l))
    w15 = optimise2!(portfolio; obj = SR(; rf = rf))
    w16 = optimise2!(portfolio; obj = MaxRet())
    @test !isapprox(w14.weights, w1.weights)
    @test !isapprox(w14.weights, w2.weights)
    @test !isapprox(w15.weights, w1.weights)
    @test !isapprox(w15.weights, w3.weights)
    @test !isapprox(w16.weights, w1.weights)
    @test !isapprox(w16.weights, w4.weights)

    @test_throws AssertionError portfolio.rebalance = rand(19)
    @test_throws AssertionError portfolio.rebalance = rand(21)
end