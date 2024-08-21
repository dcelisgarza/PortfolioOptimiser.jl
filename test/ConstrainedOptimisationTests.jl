using CSV, TimeSeries, DataFrames, StatsBase, Statistics, LinearAlgebra, Test, GLPK,
      Pajarito, JuMP, Clarabel, PortfolioOptimiser

prices = TimeArray(CSV.File("./assets/stock_prices.csv"); timestamp = :date)
rf = 1.0329^(1 / 252) - 1
l = 2.0

@testset "Rebalance" begin
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

    @test_throws AssertionError portfolio.rebalance = 1:19
    @test_throws AssertionError portfolio.rebalance = 1:21
end

@testset "Turnover" begin
    portfolio = Portfolio2(; prices = prices,
                           solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                            :params => Dict("verbose" => false,
                                                                            "max_step_fraction" => 0.75))))
    asset_statistics2!(portfolio)

    w1 = optimise2!(portfolio; obj = SR(; rf = rf))
    to1 = 0.05
    tow1 = copy(w1.weights)
    portfolio.turnover = to1
    portfolio.turnover_weights = tow1
    w2 = optimise2!(portfolio; obj = MinRisk())
    @test all(abs.(w2.weights - tow1) .<= to1)

    portfolio.turnover = Inf
    w3 = optimise2!(portfolio; obj = MinRisk())
    to2 = 0.031
    tow2 = copy(w3.weights)
    portfolio.turnover = to2
    portfolio.turnover_weights = tow2
    w4 = optimise2!(portfolio; obj = SR(; rf = rf))
    @test all(abs.(w4.weights - tow2) .<= to2)

    portfolio.turnover = Inf
    w5 = optimise2!(portfolio; obj = SR(; rf = rf))
    to3 = range(; start = 0.001, stop = 0.003, length = 20)
    tow3 = copy(w5.weights)
    portfolio.turnover = to3
    portfolio.turnover_weights = tow3
    w6 = optimise2!(portfolio; obj = MinRisk())
    @test all(abs.(w6.weights - tow3) .<= to3)

    portfolio.turnover = Inf
    w7 = optimise2!(portfolio; obj = MinRisk())
    to4 = 0.031
    tow4 = copy(w7.weights)
    portfolio.turnover = to4
    portfolio.turnover_weights = tow4
    w8 = optimise2!(portfolio; obj = SR(; rf = rf))
    @test all(abs.(w8.weights - tow4) .<= to2)

    @test_throws AssertionError portfolio.turnover = 1:19
    @test_throws AssertionError portfolio.turnover = 1:21
end

@testset "Tracking" begin
    portfolio = Portfolio2(; prices = prices,
                           solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                            :params => Dict("verbose" => false,
                                                                            "max_step_fraction" => 0.75))))
    asset_statistics2!(portfolio)
    T = size(portfolio.returns, 1)

    w1 = optimise2!(portfolio; obj = SR(; rf = rf))
    te1 = 0.0005
    tw1 = copy(w1.weights)
    portfolio.kind_tracking_err = TrackWeight()
    portfolio.tracking_err = te1
    portfolio.tracking_err_weights = tw1
    w2 = optimise2!(portfolio; obj = MinRisk())
    @test norm(portfolio.returns * (w2.weights - tw1), 2) / sqrt(T - 1) <= te1

    w3 = optimise2!(portfolio; obj = MinRisk())
    te2 = 0.0003
    tw2 = copy(w3.weights)
    portfolio.kind_tracking_err = TrackWeight()
    portfolio.tracking_err = te2
    portfolio.tracking_err_weights = tw2
    w4 = optimise2!(portfolio; obj = SR(; rf = rf))
    @test norm(portfolio.returns * (w4.weights - tw2), 2) / sqrt(T - 1) <= te2

    @test_throws AssertionError portfolio.tracking_err_weights = 1:19
    @test_throws AssertionError portfolio.tracking_err_weights = 1:21

    portfolio.tracking_err = Inf
    w5 = optimise2!(portfolio; obj = SR(; rf = rf))
    te3 = 0.007
    tw3 = copy(w5.weights)
    portfolio.kind_tracking_err = TrackRet()
    portfolio.tracking_err = te3
    tw3 = vec(mean(portfolio.returns; dims = 2))
    portfolio.tracking_err_returns = tw3
    w6 = optimise2!(portfolio; obj = MinRisk())
    @test norm(portfolio.returns * w6.weights - tw3, 2) / sqrt(T - 1) <= te3

    portfolio.tracking_err = Inf
    w7 = optimise2!(portfolio; obj = MinRisk())
    te4 = 0.0024
    tw4 = copy(w7.weights)
    portfolio.kind_tracking_err = TrackRet()
    portfolio.tracking_err = te4
    tw4 = vec(mean(portfolio.returns; dims = 2))
    portfolio.tracking_err_returns = tw4
    w8 = optimise2!(portfolio; obj = SR(; rf = rf))
    @test norm(portfolio.returns * w8.weights - tw4, 2) / sqrt(T - 1) <= te4

    @test_throws AssertionError portfolio.tracking_err_returns = 1:(T - 1)
    @test_throws AssertionError portfolio.tracking_err_returns = 1:(T + 1)
end

@testset "Min and max number of effective assets" begin
    portfolio = Portfolio2(; prices = prices,
                           solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                            :params => Dict("verbose" => false,
                                                                            "max_step_fraction" => 0.75))))
    asset_statistics2!(portfolio)

    w1 = optimise2!(portfolio; obj = MinRisk())
    portfolio.num_assets_l = 12
    w2 = optimise2!(portfolio; obj = MinRisk())
    @test count(w2.weights .>= 2e-2) >= 12
    @test count(w2.weights .>= 2e-2) > count(w1.weights .>= 2e-2)
    @test !isapprox(w1.weights, w2.weights)

    portfolio.num_assets_l = 0
    w3 = optimise2!(portfolio; obj = SR(; rf = rf))
    portfolio.num_assets_l = 8
    w4 = optimise2!(portfolio; obj = SR(; rf = rf))
    @test count(w4.weights .>= 2e-2) >= 8
    @test count(w4.weights .>= 2e-2) > count(w3.weights .>= 2e-2)
    @test !isapprox(w3.weights, w4.weights)

    portfolio = Portfolio2(; prices = prices,
                           solvers = Dict(:PClGL => Dict(:solver => optimizer_with_attributes(Pajarito.Optimizer,
                                                                                              "verbose" => false,
                                                                                              "oa_solver" => optimizer_with_attributes(GLPK.Optimizer,
                                                                                                                                       MOI.Silent() => true),
                                                                                              "conic_solver" => optimizer_with_attributes(Clarabel.Optimizer,
                                                                                                                                          "verbose" => false,
                                                                                                                                          "max_step_fraction" => 0.75)))))
    asset_statistics2!(portfolio)

    w5 = optimise2!(portfolio; obj = MinRisk())
    portfolio.num_assets_u = 5
    w6 = optimise2!(portfolio; obj = MinRisk())
    @test count(w6.weights .>= 2e-2) <= 5
    @test count(w6.weights .>= 2e-2) < count(w5.weights .>= 2e-2)
    @test !isapprox(w5.weights, w6.weights)

    portfolio.num_assets_u = 0
    w7 = optimise2!(portfolio; obj = SR(; rf = rf))
    portfolio.num_assets_u = 3
    w8 = optimise2!(portfolio; obj = SR(; rf = rf))
    @test count(w8.weights .>= 2e-2) <= 3
    @test count(w8.weights .>= 2e-2) < count(w7.weights .>= 2e-2)
    @test !isapprox(w7.weights, w8.weights)
end
