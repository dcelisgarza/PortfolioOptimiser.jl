using CSV, TimeSeries, DataFrames, StatsBase, Statistics, LinearAlgebra, Test, GLPK,
      Pajarito, JuMP, Clarabel, PortfolioOptimiser

prices = TimeArray(CSV.File("./assets/stock_prices.csv"); timestamp = :date)
rf = 1.0329^(1 / 252) - 1
l = 2.0

@testset "Rebalance Trad" begin
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

    sr5 = sharpe_ratio(portfolio; kelly = true)
    @test isapprox(dot(portfolio.mu, w4.weights) / calc_risk(portfolio), sr4)
    @test isapprox(1 / size(portfolio.returns, 1) *
                   sum(log.(1 .+ portfolio.returns * w4.weights)) / calc_risk(portfolio),
                   sr5)

    portfolio.rebalance = TR(; val = 0, w = w3.weights)
    w5 = optimise2!(portfolio; obj = MinRisk())
    @test isapprox(w1.weights, w5.weights)
    portfolio.rebalance.w = w1.weights
    w6 = optimise2!(portfolio; obj = Util(; l = l))
    w7 = optimise2!(portfolio; obj = SR(; rf = rf))
    w8 = optimise2!(portfolio; obj = MaxRet())
    @test isapprox(w2.weights, w6.weights)
    @test isapprox(w3.weights, w7.weights)
    @test isapprox(w4.weights, w8.weights)

    portfolio.rebalance = TR(; val = 1e10, w = w3.weights)
    w9 = optimise2!(portfolio; obj = MinRisk())
    @test isapprox(w9.weights, w3.weights)
    portfolio.rebalance.w = w1.weights
    w10 = optimise2!(portfolio; obj = Util(; l = l))
    w11 = optimise2!(portfolio; obj = SR(; rf = rf))
    w12 = optimise2!(portfolio; obj = MaxRet())
    @test isapprox(w10.weights, w1.weights)
    @test isapprox(w11.weights, w1.weights)
    @test isapprox(w12.weights, w1.weights)

    portfolio.rebalance = TR(; val = 1e-4, w = w3.weights)
    w13 = optimise2!(portfolio; obj = MinRisk())
    @test !isapprox(w13.weights, w1.weights)
    @test !isapprox(w13.weights, w3.weights)
    portfolio.rebalance.w = w1.weights
    w14 = optimise2!(portfolio; obj = Util(; l = l))
    w15 = optimise2!(portfolio; obj = SR(; rf = rf))
    w16 = optimise2!(portfolio; obj = MaxRet())
    @test !isapprox(w14.weights, w1.weights)
    @test !isapprox(w14.weights, w2.weights)
    @test !isapprox(w15.weights, w1.weights)
    @test !isapprox(w15.weights, w3.weights)
    @test !isapprox(w16.weights, w1.weights)
    @test !isapprox(w16.weights, w4.weights)

    portfolio.rebalance = TR(;
                             val = [0.0005174248858061537, 0.0001378289720696607,
                                    1.182008035855453e-5, 0.0009118233964947257,
                                    0.0008043804574686568, 0.0005568104999737413,
                                    0.0001433167617425195, 0.0008152431443894213,
                                    0.0006805053356229013, 8.922295760840915e-5,
                                    0.0008525847915972609, 0.0009046977862414844,
                                    0.0009820771255260512, 0.0005494961009926494,
                                    3.971977944267568e-5, 0.0006942164994964002,
                                    0.000742647266054625, 0.0004077250418932119,
                                    0.00031612114608380824, 0.00028833648463458153],
                             w = w3.weights)
    w13 = optimise2!(portfolio; obj = MinRisk())
    @test !isapprox(w13.weights, w3.weights)
    @test !isapprox(w13.weights, w1.weights)
    portfolio.rebalance.w = w1.weights
    w14 = optimise2!(portfolio; obj = Util(; l = l))
    w15 = optimise2!(portfolio; obj = SR(; rf = rf))
    w16 = optimise2!(portfolio; obj = MaxRet())
    @test !isapprox(w14.weights, w1.weights)
    @test !isapprox(w14.weights, w2.weights)
    @test !isapprox(w15.weights, w1.weights)
    @test !isapprox(w15.weights, w3.weights)
    @test !isapprox(w16.weights, w1.weights)
    @test !isapprox(w16.weights, w4.weights)

    @test_throws AssertionError portfolio.rebalance = TR(; val = 1:19)
    @test_throws AssertionError portfolio.rebalance = TR(; val = 1:21)
    @test_throws AssertionError portfolio.rebalance = TR(; w = 1:19)
    @test_throws AssertionError portfolio.rebalance = TR(; w = 1:21)
end

@testset "Rebalance WC" begin
    portfolio = Portfolio2(; prices = prices,
                           solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                            :params => Dict("verbose" => false,
                                                                            "max_step_fraction" => 0.7))))
    asset_statistics2!(portfolio)
    wc_statistics2!(portfolio)

    w1 = optimise2!(portfolio; type = WC2(), obj = MinRisk())
    r1 = calc_risk(portfolio; type = :WC2)
    ret1 = dot(portfolio.mu, w1.weights)
    sr1 = sharpe_ratio(portfolio; type = :WC2)

    w2 = optimise2!(portfolio; type = WC2(), obj = Util(; l = l))
    r2 = calc_risk(portfolio; type = :WC2)
    ret2 = dot(portfolio.mu, w2.weights)
    sr2 = sharpe_ratio(portfolio; type = :WC2)

    w3 = optimise2!(portfolio; type = WC2(), obj = SR(; rf = rf))
    r3 = calc_risk(portfolio; type = :WC2)
    ret3 = dot(portfolio.mu, w3.weights)
    sr3 = sharpe_ratio(portfolio; type = :WC2)

    w4 = optimise2!(portfolio; type = WC2(), obj = MaxRet())
    r4 = calc_risk(portfolio; type = :WC2)
    ret4 = dot(portfolio.mu, w4.weights)
    sr4 = sharpe_ratio(portfolio; type = :WC2)

    @test r1 < r2 < r3 < r4
    @test ret1 < ret2 < ret3 < ret4
    @test sr1 < sr4 < sr3 < sr2

    sr5 = sharpe_ratio(portfolio; type = :WC2, kelly = true)
    @test isapprox(dot(portfolio.mu, w4.weights) / calc_risk(portfolio; type = :WC2), sr4)
    @test isapprox(1 / size(portfolio.returns, 1) *
                   sum(log.(1 .+ portfolio.returns * w4.weights)) /
                   calc_risk(portfolio; type = :WC2), sr5)

    portfolio.rebalance = TR(; val = 0, w = w3.weights)
    w5 = optimise2!(portfolio; type = WC2(), obj = MinRisk())
    @test isapprox(w1.weights, w5.weights)
    portfolio.rebalance.w = w1.weights
    w6 = optimise2!(portfolio; type = WC2(), obj = Util(; l = l))
    w7 = optimise2!(portfolio; type = WC2(), obj = SR(; rf = rf))
    w8 = optimise2!(portfolio; type = WC2(), obj = MaxRet())
    @test isapprox(w2.weights, w6.weights)
    @test isapprox(w3.weights, w7.weights)
    @test isapprox(w4.weights, w8.weights)

    portfolio.rebalance = TR(; val = 1e10, w = w3.weights)
    w9 = optimise2!(portfolio; type = WC2(), obj = MinRisk())
    @test isapprox(w9.weights, w3.weights)
    portfolio.rebalance.w = w1.weights
    w10 = optimise2!(portfolio; type = WC2(), obj = Util(; l = l))
    w11 = optimise2!(portfolio; type = WC2(), obj = SR(; rf = rf))
    w12 = optimise2!(portfolio; type = WC2(), obj = MaxRet())
    @test isapprox(w10.weights, w1.weights, rtol = 5.0e-8)
    @test isapprox(w11.weights, w1.weights, rtol = 5.0e-8)
    @test isapprox(w12.weights, w1.weights, rtol = 1.0e-6)

    portfolio.rebalance = TR(; val = 1e-4, w = w3.weights)
    w13 = optimise2!(portfolio; type = WC2(), obj = MinRisk())
    @test !isapprox(w13.weights, w1.weights)
    @test !isapprox(w13.weights, w3.weights)
    portfolio.rebalance.w = w1.weights
    w14 = optimise2!(portfolio; type = WC2(), obj = Util(; l = l))
    w15 = optimise2!(portfolio; type = WC2(), obj = SR(; rf = rf))
    w16 = optimise2!(portfolio; type = WC2(), obj = MaxRet())
    @test !isapprox(w14.weights, w1.weights)
    @test !isapprox(w14.weights, w2.weights)
    @test !isapprox(w15.weights, w1.weights)
    @test !isapprox(w15.weights, w3.weights)
    @test !isapprox(w16.weights, w1.weights)
    @test !isapprox(w16.weights, w4.weights)

    portfolio.rebalance = TR(;
                             val = [0.0005174248858061537, 0.0001378289720696607,
                                    1.182008035855453e-5, 0.0009118233964947257,
                                    0.0008043804574686568, 0.0005568104999737413,
                                    0.0001433167617425195, 0.0008152431443894213,
                                    0.0006805053356229013, 8.922295760840915e-5,
                                    0.0008525847915972609, 0.0009046977862414844,
                                    0.0009820771255260512, 0.0005494961009926494,
                                    3.971977944267568e-5, 0.0006942164994964002,
                                    0.000742647266054625, 0.0004077250418932119,
                                    0.00031612114608380824, 0.00028833648463458153],
                             w = w3.weights)
    w13 = optimise2!(portfolio; type = WC2(), obj = MinRisk())
    @test !isapprox(w13.weights, w3.weights)
    @test !isapprox(w13.weights, w1.weights)
    portfolio.rebalance.w = w1.weights
    w14 = optimise2!(portfolio; type = WC2(), obj = Util(; l = l))
    w15 = optimise2!(portfolio; type = WC2(), obj = SR(; rf = rf))
    w16 = optimise2!(portfolio; type = WC2(), obj = MaxRet())
    @test !isapprox(w14.weights, w1.weights)
    @test !isapprox(w14.weights, w2.weights)
    @test !isapprox(w15.weights, w1.weights)
    @test !isapprox(w15.weights, w3.weights)
    @test !isapprox(w16.weights, w1.weights)
    @test !isapprox(w16.weights, w4.weights)

    @test_throws AssertionError portfolio.rebalance = TR(; val = 1:19)
    @test_throws AssertionError portfolio.rebalance = TR(; val = 1:21)
    @test_throws AssertionError portfolio.rebalance = TR(; w = 1:19)
    @test_throws AssertionError portfolio.rebalance = TR(; w = 1:21)
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
    portfolio.turnover = TR(; val = to1, w = tow1)
    w2 = optimise2!(portfolio; obj = MinRisk())
    @test all(abs.(w2.weights - tow1) .<= to1)

    portfolio.turnover = NoTR()
    w3 = optimise2!(portfolio; obj = MinRisk())
    to2 = 0.031
    tow2 = copy(w3.weights)
    portfolio.turnover = TR(; val = to2, w = tow2)
    w4 = optimise2!(portfolio; obj = SR(; rf = rf))
    @test all(abs.(w4.weights - tow2) .<= to2)

    portfolio.turnover = NoTR()
    w5 = optimise2!(portfolio; obj = SR(; rf = rf))
    to3 = range(; start = 0.001, stop = 0.003, length = 20)
    tow3 = copy(w5.weights)
    portfolio.turnover = TR(; val = to3, w = tow3)
    w6 = optimise2!(portfolio; obj = MinRisk())
    @test all(abs.(w6.weights - tow3) .<= to3)

    portfolio.turnover = NoTR()
    w7 = optimise2!(portfolio; obj = MinRisk())
    to4 = 0.031
    tow4 = copy(w7.weights)
    portfolio.turnover = TR(; val = to4, w = tow4)
    w8 = optimise2!(portfolio; obj = SR(; rf = rf))
    @test all(abs.(w8.weights - tow4) .<= to2)

    @test_throws AssertionError portfolio.turnover = TR(; val = 1:19)
    @test_throws AssertionError portfolio.turnover = TR(; val = 1:21)
    @test_throws AssertionError portfolio.turnover = TR(; w = 1:19)
    @test_throws AssertionError portfolio.turnover = TR(; w = 1:21)
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
    portfolio.tracking_err = TrackWeight(; err = te1, w = tw1)
    w2 = optimise2!(portfolio; obj = MinRisk())
    @test norm(portfolio.returns * (w2.weights - tw1), 2) / sqrt(T - 1) <= te1

    w3 = optimise2!(portfolio; obj = MinRisk())
    te2 = 0.0003
    tw2 = copy(w3.weights)
    portfolio.tracking_err = TrackWeight(; err = te2, w = tw2)
    w4 = optimise2!(portfolio; obj = SR(; rf = rf))
    @test norm(portfolio.returns * (w4.weights - tw2), 2) / sqrt(T - 1) <= te2
    @test_throws AssertionError portfolio.tracking_err = TrackWeight(; err = te2, w = 1:19)
    @test_throws AssertionError portfolio.tracking_err = TrackWeight(; err = te2, w = 1:21)

    portfolio.tracking_err = NoTracking()
    w5 = optimise2!(portfolio; obj = SR(; rf = rf))
    te3 = 0.007
    tw3 = vec(mean(portfolio.returns; dims = 2))
    portfolio.tracking_err = TrackRet(; err = te3, w = tw3)
    w6 = optimise2!(portfolio; obj = MinRisk())
    @test norm(portfolio.returns * w6.weights - tw3, 2) / sqrt(T - 1) <= te3

    portfolio.tracking_err = NoTracking()
    w7 = optimise2!(portfolio; obj = MinRisk())
    te4 = 0.0024
    tw4 = vec(mean(portfolio.returns; dims = 2))
    portfolio.tracking_err = TrackRet(; err = te4, w = tw4)
    w8 = optimise2!(portfolio; obj = SR(; rf = rf))
    @test norm(portfolio.returns * w8.weights - tw4, 2) / sqrt(T - 1) <= te4

    @test_throws AssertionError portfolio.tracking_err = TrackRet(; err = te2,
                                                                  w = 1:(T - 1))
    @test_throws AssertionError portfolio.tracking_err = TrackRet(; err = te2,
                                                                  w = 1:(T + 1))

    portfolio.tracking_err = TrackRet(; err = te2, w = 1:T)
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

    portfolio.num_assets_l = 0
    portfolio.short = true
    portfolio.short_u = 0.2
    portfolio.long_u = 0.8

    w5 = optimise2!(portfolio; obj = MinRisk())
    @test isapprox(sum(w5.weights), portfolio.sum_short_long)
    @test sum(w5.weights[w5.weights .< 0]) <= portfolio.short_u
    @test sum(w5.weights[w5.weights .>= 0]) <= portfolio.long_u
    portfolio.num_assets_l = 17
    w6 = optimise2!(portfolio; obj = MinRisk())
    @test isapprox(sum(w6.weights), portfolio.sum_short_long)
    @test sum(w6.weights[w6.weights .< 0]) <= portfolio.short_u
    @test sum(w6.weights[w6.weights .>= 0]) <= portfolio.long_u
    @test count(abs.(w6.weights) .>= 4e-3) >= 17
    @test count(abs.(w6.weights) .>= 4e-3) > count(abs.(w5.weights) .>= 4e-3)
    @test !isapprox(w5.weights, w6.weights)

    portfolio.num_assets_l = 0
    w7 = optimise2!(portfolio; obj = SR(; rf = rf))
    @test isapprox(sum(w7.weights), portfolio.sum_short_long)
    portfolio.num_assets_l = 13
    w8 = optimise2!(portfolio; obj = SR(; rf = rf))
    @test isapprox(sum(w8.weights), portfolio.sum_short_long)
    @test count(abs.(w8.weights) .>= 4e-3) >= 13
    @test count(abs.(w8.weights) .>= 4e-3) > count(abs.(w7.weights) .>= 4e-3)
    @test !isapprox(w7.weights, w8.weights)

    portfolio = Portfolio2(; prices = prices,
                           solvers = Dict(:PClGL => Dict(:solver => optimizer_with_attributes(Pajarito.Optimizer,
                                                                                              "verbose" => false,
                                                                                              "oa_solver" => optimizer_with_attributes(GLPK.Optimizer,
                                                                                                                                       MOI.Silent() => true),
                                                                                              "conic_solver" => optimizer_with_attributes(Clarabel.Optimizer,
                                                                                                                                          "verbose" => false,
                                                                                                                                          "max_step_fraction" => 0.75)))))
    asset_statistics2!(portfolio)

    w9 = optimise2!(portfolio; obj = MinRisk())
    portfolio.num_assets_u = 5
    w10 = optimise2!(portfolio; obj = MinRisk())
    @test count(w10.weights .>= 2e-2) <= 5
    @test count(w10.weights .>= 2e-2) < count(w9.weights .>= 2e-2)
    @test !isapprox(w9.weights, w10.weights)

    portfolio.num_assets_u = 0
    w11 = optimise2!(portfolio; obj = SR(; rf = rf))
    portfolio.num_assets_u = 3
    w12 = optimise2!(portfolio; obj = SR(; rf = rf))
    @test count(w12.weights .>= 2e-2) <= 3
    @test count(w12.weights .>= 2e-2) < count(w11.weights .>= 2e-2)
    @test !isapprox(w11.weights, w12.weights)

    portfolio.num_assets_u = 0
    portfolio.short = true
    portfolio.short_u = 0.2
    portfolio.long_u = 0.8

    w13 = optimise2!(portfolio; obj = MinRisk())
    @test isapprox(sum(w13.weights), portfolio.sum_short_long)
    @test sum(w13.weights[w13.weights .< 0]) <= portfolio.short_u
    @test sum(w13.weights[w13.weights .>= 0]) <= portfolio.long_u
    portfolio.num_assets_u = 7
    w14 = optimise2!(portfolio; obj = MinRisk())
    @test isapprox(sum(w14.weights), portfolio.sum_short_long)
    @test sum(w14.weights[w14.weights .< 0]) <= portfolio.short_u
    @test sum(w14.weights[w14.weights .>= 0]) <= portfolio.long_u
    @test count(abs.(w14.weights) .>= 2e-2) <= 7
    @test count(abs.(w14.weights) .>= 2e-2) < count(abs.(w13.weights) .>= 2e-2)
    @test !isapprox(w13.weights, w14.weights)

    portfolio.num_assets_u = 0
    w15 = optimise2!(portfolio; obj = SR(; rf = rf))
    @test isapprox(sum(w15.weights), portfolio.sum_short_long)
    @test sum(w15.weights[w15.weights .< 0]) <= portfolio.short_u
    @test sum(w15.weights[w15.weights .>= 0]) <= portfolio.long_u
    portfolio.num_assets_u = 4
    w16 = optimise2!(portfolio; obj = SR(; rf = rf))
    @test isapprox(sum(w16.weights), portfolio.sum_short_long)
    @test sum(w16.weights[w16.weights .< 0]) <= portfolio.short_u
    @test abs(sum(w16.weights[w16.weights .>= 0]) - portfolio.long_u) <= 20 * eps()
    @test count(abs.(w16.weights) .>= 2e-2) >= 4
    @test count(abs.(w16.weights) .>= 2e-2) < count(abs.(w15.weights) .>= 2e-2)
    @test !isapprox(w15.weights, w16.weights)

    @test_throws AssertionError portfolio.num_assets_l = -1
    @test_throws AssertionError portfolio.num_assets_u = -1
end

@testset "Linear" begin
    portfolio = Portfolio2(; prices = prices,
                           solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                            :params => Dict("verbose" => false,
                                                                            "max_step_fraction" => 0.75))))
    asset_statistics2!(portfolio)

    asset_sets = DataFrame("Asset" => portfolio.assets,
                           "PDBHT" => [1, 2, 1, 1, 1, 3, 2, 2, 3, 3, 3, 4, 4, 3, 3, 4, 2, 2,
                                       3, 1],
                           "SPDBHT" => [1, 1, 1, 1, 1, 2, 3, 4, 2, 3, 3, 2, 3, 3, 3, 3, 1,
                                        4, 2, 1],
                           "Pward" => [1, 1, 1, 1, 1, 2, 3, 2, 2, 2, 2, 4, 4, 2, 3, 4, 1, 2,
                                       2, 1],
                           "SPward" => [1, 1, 1, 1, 1, 2, 2, 3, 2, 2, 2, 4, 3, 2, 2, 3, 1,
                                        2, 2, 1],
                           "G2DBHT" => [1, 2, 1, 1, 1, 3, 2, 3, 4, 3, 4, 3, 3, 4, 4, 3, 2,
                                        3, 4, 1],
                           "G2ward" => [1, 1, 1, 1, 1, 2, 3, 4, 2, 2, 4, 2, 3, 3, 3, 2, 1,
                                        4, 2, 2])
    constraints = DataFrame(:Enabled => [true, true, true, true, true],
                            :Type => ["Each Asset in Subset", "Each Asset in Subset",
                                      "Asset", "Subset", "Asset"],
                            :Set => ["G2DBHT", "G2DBHT", "", "G2ward", ""],
                            :Position => [2, 3, "AAPL", 2, "MA"],
                            :Sign => [">=", "<=", ">=", "<=", ">="],
                            :Weight => [0.03, 0.2, 0.032, "", ""],
                            :Relative_Type => ["", "", "", "Asset", "Subset"],
                            :Relative_Set => ["", "", "", "", "G2ward"],
                            :Relative_Position => ["", "", "", "MA", 3],
                            :Factor => ["", "", "", 2.2, 5])

    A, B = asset_constraints(constraints, asset_sets)
    portfolio.a_mtx_ineq = A
    portfolio.b_vec_ineq = B

    w1 = optimise2!(portfolio; obj = MinRisk())
    @test all(w1.weights[asset_sets.G2DBHT .== 2] .>= 0.03)
    @test all(w1.weights[asset_sets.G2DBHT .== 3] .<= 0.2)
    @test all(w1.weights[w1.tickers .== "AAPL"] .>= 0.032)
    @test sum(w1.weights[asset_sets.G2ward .== 2]) <=
          w1.weights[w1.tickers .== "MA"][1] * 2.2
    @test w1.weights[w1.tickers .== "MA"][1] >= sum(w1.weights[asset_sets.G2ward .== 3]) * 5

    w2 = optimise2!(portfolio; obj = SR(; rf = rf))
    @test all(w2.weights[asset_sets.G2DBHT .== 2] .>= 0.03)
    @test all(w2.weights[asset_sets.G2DBHT .== 3] .<= 0.2)
    @test all(w2.weights[w2.tickers .== "AAPL"] .>= 0.032)
    @test sum(w2.weights[asset_sets.G2ward .== 2]) <=
          w2.weights[w2.tickers .== "MA"][1] * 2.2
    @test w2.weights[w2.tickers .== "MA"][1] >= sum(w2.weights[asset_sets.G2ward .== 3]) * 5

    @test_throws AssertionError portfolio.a_mtx_ineq = rand(13, 19)
    @test_throws AssertionError portfolio.a_mtx_ineq = rand(13, 21)
end

@testset "Network and Dendrogram SD" begin
    portfolio = Portfolio2(; prices = prices,
                           solvers = Dict(:PClGL => Dict(:solver => optimizer_with_attributes(Pajarito.Optimizer,
                                                                                              "verbose" => false,
                                                                                              "oa_solver" => optimizer_with_attributes(GLPK.Optimizer,
                                                                                                                                       MOI.Silent() => true),
                                                                                              "conic_solver" => optimizer_with_attributes(Clarabel.Optimizer,
                                                                                                                                          "verbose" => false,
                                                                                                                                          "max_step_fraction" => 0.75)))))
    asset_statistics2!(portfolio)
    wc_statistics2!(portfolio,
                    WCType(; box = WorstCaseNormal(; seed = 123456789),
                           ellipse = WorstCaseNormal(; seed = 123456789)))

    A = centrality_vector2(portfolio)
    B = connection_matrix2(portfolio)
    C = cluster_matrix2(portfolio)

    rm = SD2()
    obj = MinRisk()
    w1 = optimise2!(portfolio; obj = obj, rm = rm)
    wt = [0.00791850924098073, 0.030672065216453506, 0.010501402809199123,
          0.027475241969187995, 0.012272329269527841, 0.03339587076426262,
          1.4321532289072258e-6, 0.13984297866711365, 2.4081081597353397e-6,
          5.114425959766348e-5, 0.2878111114337346, 1.5306036912879562e-6,
          1.1917690994187655e-6, 0.12525446872321966, 6.630910273840812e-6,
          0.015078706008184504, 8.254970801614574e-5, 0.1930993918762575,
          3.0369340845779798e-6, 0.11652799957572683]
    @test isapprox(w1.weights, wt)

    wc1 = optimise2!(portfolio; type = WC2(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w1.weights, wc1.weights)

    portfolio.a_vec_cent = A
    portfolio.b_cent = minimum(A)
    w2 = optimise2!(portfolio; obj = obj, rm = rm)
    wt = [3.5426430661259544e-11, 0.07546661314221123, 0.021373033743425803,
          0.027539611826011074, 0.023741467255115698, 0.12266515815974924,
          2.517181275812244e-6, 0.22629893782442134, 3.37118211246211e-10,
          0.02416530860824232, 3.3950118687352606e-10, 1.742541411959769e-6,
          1.5253730188343444e-6, 5.304686980295978e-11, 0.024731789616991084,
          3.3711966852218057e-10, 7.767147353183488e-11, 0.30008056166009706,
          1.171415437554966e-10, 0.1539317317710031]
    @test isapprox(w2.weights, wt)

    portfolio.network_method = IP2(; A = B)
    w3 = optimise2!(portfolio; obj = obj, rm = rm)
    wt = [0.0, 0.0, 0.08531373474615221, 0.0, 0.0, 0.0, 0.0, 0.2553257507374485, 0.0, 0.0,
          0.0, 0.0, 0.0, 0.0, 0.04299908482109457, 0.0, 0.0, 0.3859132985782231, 0.0,
          0.2304481311170818]
    @test isapprox(w3.weights, wt, rtol = 0.01)

    wc3 = optimise2!(portfolio; type = WC2(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w3.weights, wc3.weights, rtol = 0.01)

    portfolio.network_method = SDP2(; A = B)
    w4 = optimise2!(portfolio; obj = obj, rm = rm)
    wt = [4.2907851108373285e-11, 0.0754418669962984, 0.021396172655536325,
          0.027531481813488985, 0.02375148897766012, 0.12264432042703496,
          3.777946382632432e-6, 0.2263233585633904, 3.8909758570393176e-10,
          0.024184506959405477, 3.8945719747726095e-10, 2.597992728770223e-6,
          2.2916747013451583e-6, 5.942491440558677e-11, 0.02472298577516722,
          3.89238712058158e-10, 9.010863437870188e-11, 0.30009565874345184,
          1.32909841405546e-10, 0.15389948998160896]
    @test isapprox(w4.weights, wt)

    wc4 = optimise2!(portfolio; type = WC2(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w4.weights, wc4.weights)

    portfolio.network_method = IP2(; A = C)
    w5 = optimise2!(portfolio; obj = obj, rm = rm)
    wt = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
          0.053218458202620444, 0.0, 0.0, 0.5647264843921129, 0.0, 0.38205505740526663]
    @test isapprox(w5.weights, wt)

    wc5 = optimise2!(portfolio; type = WC2(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w5.weights, wc5.weights)

    portfolio.network_method = SDP2(; A = C)
    w6 = optimise2!(portfolio; obj = obj, rm = rm)
    wt = [6.011468296535483e-11, 9.424732042613459e-6, 5.527537353284497e-6,
          2.903730060970158e-6, 4.265000229139505e-6, 5.05868769590198e-6,
          3.009791891015968e-6, 1.5268395403568086e-5, 5.657961006483616e-10,
          2.68155059410767e-6, 5.662844616891034e-10, 3.196842024458429e-6,
          2.419924540093777e-6, 8.983883474709306e-11, 0.05335383109918724,
          5.659272794304194e-10, 1.3052770418372395e-10, 0.5616352353647922,
          1.9587068208203376e-10, 0.38495717516982575]
    @test isapprox(w6.weights, wt)

    wc6 = optimise2!(portfolio; type = WC2(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w6.weights, wc6.weights)

    portfolio.a_vec_cent = []
    portfolio.b_cent = Inf

    portfolio.network_method = IP2(; A = B)
    w7 = optimise2!(portfolio; obj = obj, rm = rm)
    wt = [0.0, 0.0, 0.0, 0.05362285516498196, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
          0.4241445775236924, 0.0, 0.0, 0.0, 0.0, 0.027024889332849245, 0.0,
          0.30343700156503567, 0.0, 0.19177067641344075]
    @test isapprox(w7.weights, wt)

    wc7 = optimise2!(portfolio; type = WC2(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w7.weights, wc7.weights, rtol = 0.0005)

    portfolio.network_method = SDP2(; A = B)
    w8 = optimise2!(portfolio; obj = obj, rm = rm)
    wt = [1.1008281068185902e-5, 0.055457116453828725, 0.012892903094396706,
          0.03284102649502972, 0.014979204379343122, 0.057886691097825904,
          1.0224197847100319e-6, 9.70550406073092e-6, 2.045810229626667e-6,
          0.012049795193184915, 0.3735114108030411, 1.330358085433759e-6,
          1.0648304534905729e-6, 9.906438750370314e-6, 0.007878565110236062,
          0.022521836037796082, 3.1895242150194783e-6, 0.26190144912467656,
          1.3462532236761872e-6, 0.14803938279076995]
    @test isapprox(w8.weights, wt, rtol = 5.0e-8)

    wc8 = optimise2!(portfolio; type = WC2(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w8.weights, wc8.weights)

    portfolio.network_method = IP2(; A = C)
    w9 = optimise2!(portfolio; obj = obj, rm = rm)
    wt = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6083352374839345, 0.0, 0.0,
          0.0, 0.0, 0.05780303235300369, 0.0, 0.0, 0.0, 0.3338617301630618]
    @test isapprox(w9.weights, wt)

    wc9 = optimise2!(portfolio; type = WC2(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w9.weights, wc9.weights)

    portfolio.network_method = SDP2(; A = C)
    w10 = optimise2!(portfolio; obj = obj, rm = rm)
    wt = [6.5767880723988875e-6, 4.623865766176694e-6, 3.533665790240184e-6,
          2.181035748649495e-6, 2.5215556700206228e-6, 1.858111674425038e-6,
          2.5125294721249436e-6, 2.7192384937770815e-6, 8.752492794952926e-7,
          9.63854668493286e-7, 0.6119910196342336, 2.4053903351549724e-6,
          9.311964129813141e-7, 9.706505481488379e-6, 1.3337324741660364e-5,
          0.05681677180817997, 4.294323401245429e-5, 1.2805451158943426e-5,
          1.6336794237581935e-6, 0.33108007988138427]
    @test isapprox(w10.weights, wt)

    wc10 = optimise2!(portfolio; type = WC2(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w10.weights, wc10.weights)

    portfolio = Portfolio2(; prices = prices,
                           solvers = Dict(:PClGL => Dict(:solver => optimizer_with_attributes(Pajarito.Optimizer,
                                                                                              "verbose" => false,
                                                                                              "oa_solver" => optimizer_with_attributes(GLPK.Optimizer,
                                                                                                                                       MOI.Silent() => true),
                                                                                              "conic_solver" => optimizer_with_attributes(Clarabel.Optimizer,
                                                                                                                                          "verbose" => false,
                                                                                                                                          "max_step_fraction" => 0.75)))))
    asset_statistics2!(portfolio)
    obj = SR(; rf = rf)

    A = centrality_vector2(portfolio; network_type = TMFG())
    B = connection_matrix2(portfolio; network_type = TMFG())
    C = cluster_matrix2(portfolio; hclust_alg = DBHT())

    w11 = optimise2!(portfolio; obj = obj, rm = rm)
    wt = [5.2456519308516375e-9, 1.3711772920494845e-8, 1.828307590553689e-8,
          1.1104551657521415e-8, 0.5180586238310904, 1.7475553033958716e-9,
          0.06365101880957791, 1.1504018896467816e-8, 1.0491794751235226e-8,
          5.650081752041375e-9, 9.262635444153261e-9, 1.4051602963640009e-9,
          9.923106257589635e-10, 3.050812431838031e-9, 9.966076078173243e-10,
          0.1432679499627637, 0.19649701265872604, 1.0778757180749977e-8,
          0.07852527921167819, 1.1301377011579277e-8]
    @test isapprox(w11.weights, wt)

    wc11 = optimise2!(portfolio; type = WC2(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w11.weights, wc11.weights, rtol = 5.0e-6)

    portfolio.a_vec_cent = A
    portfolio.b_cent = minimum(A)

    w12 = optimise2!(portfolio; obj = obj, rm = rm)
    wt = [3.580964533638559e-11, 9.277053125069126e-10, 9.835000020421294e-10,
          0.48362433627647977, 1.3062621693742353e-9, 3.4185103123124565e-11,
          0.28435553885288534, 0.17984723049407092, 1.693432131576565e-10,
          2.621850918060399e-10, 0.05217287791621345, 5.161470450551339e-9,
          2.4492072342885186e-9, 4.790090083985224e-11, 2.6687808589346464e-9,
          1.0539691276230515e-9, 2.0827050059613154e-10, 8.48619997995145e-10,
          2.329804826102029e-10, 7.01603778985698e-11]
    @test isapprox(w12.weights, wt)

    wc12 = optimise2!(portfolio; type = WC2(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w12.weights, wc12.weights, rtol = 5.0e-5)

    portfolio.network_method = IP2(; A = B)
    w13 = optimise2!(portfolio; obj = obj, rm = rm)
    wt = [1.721507261658482e-11, 4.611331427286785e-11, 4.6537999918192326e-11,
          5.4302470941152905e-11, 4.508258159401719e-11, 2.510366932792115e-11,
          0.9999999992889718, 5.3842283750049385e-11, 9.81183158048394e-12,
          3.589613596601703e-11, 5.3385756391682404e-11, 4.8553725558375136e-11,
          4.299842533772006e-11, 1.755101403265066e-11, 4.285647343183246e-11,
          4.6446497580083914e-11, 2.1013218648897535e-11, 4.559832123949408e-11,
          3.102165424744005e-11, 2.7697895657875245e-11]
    @test isapprox(w13.weights, wt)

    wc13 = optimise2!(portfolio; type = WC2(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w13.weights, wc13.weights)

    portfolio.network_method = SDP2(; A = B)
    w14 = optimise2!(portfolio; obj = obj, rm = rm)
    wt = [5.024048627217141e-14, 1.5886394555170722e-12, 1.5924178422101003e-12,
          0.24554600920016964, 1.6332399340098677e-12, 1.0616523261700116e-13,
          0.0959435974734496, 0.2562392110969168, 2.7829378542014683e-13,
          4.877264338339266e-13, 0.402271160351581, 1.2141431002683076e-8,
          4.523802241069811e-9, 5.3729255370564864e-14, 5.20282725457044e-9,
          1.5947986553082236e-12, 3.43551721221936e-13, 1.5824894417145518e-12,
          3.8307499177570936e-13, 1.280439184666322e-13]
    @test isapprox(w14.weights, wt)

    wc14 = optimise2!(portfolio; type = WC2(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w14.weights, wc14.weights, rtol = 5.0e-5)

    portfolio.network_method = IP2(; A = C)
    w15 = optimise2!(portfolio; obj = obj, rm = rm)
    wt = [3.5488808869733083e-12, 8.281743679661074e-12, 8.80902177574244e-12,
          9.955072746064901e-12, 1.1134822019504797e-11, 4.7417114086610015e-12,
          0.6074274195608048, 1.0253105313724553e-11, 1.4148918790428058e-12,
          6.3499711667509184e-12, 0.39257258031345454, 9.215307590460059e-12,
          7.61409793255415e-12, 3.5761100061677828e-12, 7.3543478626708e-12,
          9.27354997599413e-12, 3.8374099382272584e-12, 8.458519745181501e-12,
          6.62153510029776e-12, 5.3005368572386036e-12]
    @test isapprox(w15.weights, wt, rtol = 1.0e-7)

    wc15 = optimise2!(portfolio; type = WC2(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w15.weights, wc15.weights, rtol = 0.005)

    portfolio.network_method = SDP2(; A = C)
    w16 = optimise2!(portfolio; obj = obj, rm = rm)
    wt = [1.645653643573042e-14, 5.24765733463725e-13, 5.259882343242254e-13,
          0.38135427736460104, 5.383168184665497e-13, 3.606530943112124e-14,
          2.659687073118818e-8, 2.387877721283499e-8, 9.190837103177029e-14,
          1.6141883247690533e-13, 0.6186456659519168, 3.0002908666789655e-9,
          1.5771657468870285e-9, 1.747936659731388e-14, 1.6271322639469356e-9,
          5.267534803071819e-13, 1.1362061261344631e-13, 5.231419643587768e-13,
          1.2683767418078167e-13, 4.256536051656617e-14]
    @test isapprox(w16.weights, wt)

    wc16 = optimise2!(portfolio; type = WC2(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w16.weights, wc16.weights, rtol = 5.0e-6)

    portfolio.a_vec_cent = []
    portfolio.b_cent = Inf

    portfolio.network_method = IP2(; A = B)
    w17 = optimise2!(portfolio; obj = obj, rm = rm)
    wt = [2.0966673826999653e-12, 2.198260339765037e-12, 1.9669534265126797e-12,
          2.0537353944645557e-12, 0.762282392772933, 2.61589719142697e-12,
          1.6397732202694488e-12, 2.573524529021814e-12, 2.1635532147945916e-12,
          2.3920718749431715e-12, 2.6031535089914636e-12, 2.254849729224801e-12,
          2.3351897528966937e-12, 2.5746406212787903e-12, 2.4339077077727048e-12,
          0.23771760718605286, 2.077497433772476e-12, 2.544464994811904e-12,
          2.2242733110585934e-12, 2.2657939494042705e-12]
    @test isapprox(w17.weights, wt)

    wc17 = optimise2!(portfolio; type = WC2(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w17.weights, wc17.weights, rtol = 0.0005)

    portfolio.network_method = SDP2(; A = B)
    w18 = optimise2!(portfolio; obj = obj, rm = rm)
    wt = [1.131030668352105e-7, 9.736849167799003e-7, 1.5073134798603708e-7,
          9.724848430006577e-8, 0.3050368084415617, 4.717356198432155e-8,
          0.03491168150833796, 1.1696087757981777e-6, 2.1133994869894878e-7,
          1.38898043984553e-7, 0.23158972602737993, 2.930159759606465e-8,
          1.841227833023016e-8, 1.3415748037100702e-7, 2.038375787580353e-8,
          0.10856264505102015, 2.399490931217591e-7, 0.2072142228794291,
          2.522693174355702e-7, 0.11268131983060002]
    @test isapprox(w18.weights, wt, rtol = 5.0e-8)

    wc18 = optimise2!(portfolio; type = WC2(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w18.weights, wc18.weights, rtol = 5.0e-5)

    portfolio.network_method = IP2(; A = C)
    w19 = optimise2!(portfolio; obj = obj, rm = rm)
    wt = [1.1904763791512141e-12, 1.2340531692956696e-12, 1.067454109369536e-12,
          1.127653866268004e-12, 0.617653720874792, 1.6368429734597877e-12,
          9.929978443062197e-13, 1.5549069160715418e-12, 1.0890925109349555e-12,
          1.3843599449712156e-12, 1.5689472056216027e-12, 1.400584981008052e-12,
          1.5941278979112304e-12, 1.5487191642726196e-12, 1.6035009203180108e-12,
          0.1719148482846596, 1.1171304302527388e-12, 1.5058581547861255e-12,
          0.210431430817618, 1.3135957217697496e-12]
    @test isapprox(w19.weights, wt)

    wc19 = optimise2!(portfolio; type = WC2(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w19.weights, wc19.weights, rtol = 0.005)

    portfolio.network_method = SDP2(; A = C)
    w20 = optimise2!(portfolio; obj = obj, rm = rm)
    wt = [1.9469265893415583e-7, 1.9264487242540467e-7, 2.232335248226593e-7,
          1.1246896197634285e-7, 0.40746626496017013, 8.182096647686593e-8,
          5.006408130895852e-8, 2.231759312172926e-7, 3.1463420211396267e-7,
          7.356649686989611e-6, 0.3255255880379563, 4.241919388356267e-8,
          2.851355716950413e-8, 2.082232132989454e-7, 3.1366090964049265e-8,
          0.15590260393465316, 0.010877836307400067, 2.4616586233596695e-7,
          0.10021813719562453, 2.6349139195481583e-7]
    @test isapprox(w20.weights, wt, rtol = 5.0e-6)

    wc20 = optimise2!(portfolio; type = WC2(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w20.weights, wc20.weights, rtol = 0.0001)
end

@testset "Network and Dendrogram SD short" begin
    portfolio = Portfolio2(; prices = prices,
                           solvers = Dict(:PClGL => Dict(:solver => optimizer_with_attributes(Pajarito.Optimizer,
                                                                                              "verbose" => false,
                                                                                              "oa_solver" => optimizer_with_attributes(GLPK.Optimizer,
                                                                                                                                       MOI.Silent() => true),
                                                                                              "conic_solver" => optimizer_with_attributes(Clarabel.Optimizer,
                                                                                                                                          "verbose" => false,
                                                                                                                                          "max_step_fraction" => 0.75)))))
    asset_statistics2!(portfolio)
    wc_statistics2!(portfolio,
                    WCType(; box = WorstCaseNormal(; seed = 123456789),
                           ellipse = WorstCaseNormal(; seed = 123456789)))
    portfolio.short = true
    portfolio.short_u = 0.22
    portfolio.long_u = 0.88
    ssl1 = portfolio.sum_short_long

    A = centrality_vector2(portfolio)
    B = connection_matrix2(portfolio)
    C = cluster_matrix2(portfolio)

    rm = SD2()
    obj = MinRisk()
    w1 = optimise2!(portfolio; obj = obj, rm = rm)
    wt = [0.0023732456580253273, 0.02478417662625044, 0.011667587612994237,
          0.021835859830789766, 0.008241287060153578, 0.03550224737381907,
          -0.006408569798470374, 0.09320297606692711, -0.0072069893239688695,
          0.012269888650642718, 0.18722784970990347, -0.01395210113745165,
          -0.006026909793594887, 0.0962770551826336, 0.0005110433722419729,
          0.016784621725152556, 0.009614188864666265, 0.13436280943955117,
          -0.042344878358121694, 0.08128461123785628]
    @test isapprox(w1.weights, wt)

    wc1 = optimise2!(portfolio; type = WC2(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w1.weights, wc1.weights)

    portfolio.a_vec_cent = A
    portfolio.b_cent = minimum(A)
    w2 = optimise2!(portfolio; obj = obj, rm = rm)
    wt = [0.0064520166328966245, 0.022489884719278264, 0.01029468309633522,
          0.020902946113760288, 0.00711671166632188, 0.03337768007882898,
          -0.006266220580382531, 0.09139178524192762, -0.02237340947148962,
          0.010954134035283628, 0.18619094237390604, -0.014102818325104587,
          -0.0055880934328339325, 0.09574848799807273, 0.0002468303788802295,
          0.01712117146324995, 0.014137258176396965, 0.13074888415899583,
          -0.01805679382768158, 0.07921391950335803]
    @test isapprox(w2.weights, wt)

    portfolio.network_method = IP2(; A = B)
    w3 = optimise2!(portfolio; obj = obj, rm = rm)
    wt = [0.0, -0.0, -0.0, 0.031931693272627654, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0,
          0.31510656939207515, -0.0, -0.0, -0.0, 0.0, 0.02489343060792462, -0.0,
          0.17590654848391327, -0.0, 0.11216175824345923]
    @test isapprox(w3.weights, wt, rtol = 0.05)

    wc3 = optimise2!(portfolio; type = WC2(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w3.weights, wc3.weights, rtol = 0.05)

    portfolio.network_method = SDP2(; A = B)
    w4 = optimise2!(portfolio; obj = obj, rm = rm)
    wt = [0.022589148447524018, 0.036812012135242114, 0.006833793601670234,
          0.013202059102917668, 0.005824555380015791, 0.03984269080339122,
          -0.006834168984417891, 1.9327468236390324e-5, -2.1177840385696868e-5,
          0.011500602752026409, 0.25265179781193275, -0.014411712546749403,
          -1.628444737600106e-5, 1.195721590601671e-5, 0.007515505990608158,
          0.019532808277500043, 4.833328704956932e-6, 0.16879897503212193,
          4.3131102914277385e-6, 0.09613896336083992]
    @test isapprox(w4.weights, wt)

    wc4 = optimise2!(portfolio; type = WC2(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w4.weights, wc4.weights)

    portfolio.network_method = IP2(; A = C)
    w5 = optimise2!(portfolio; obj = obj, rm = rm)
    wt = [-0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, 0.0, -0.0, -0.0, 0.33999999999999975,
          -0.0, 0.0, -0.0, 0.02739848834500888, -0.0, -0.0, 0.0, -0.0, 0.2926015116549916]
    @test isapprox(w5.weights, wt)

    wc5 = optimise2!(portfolio; type = WC2(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w5.weights, wc5.weights)

    portfolio.network_method = SDP2(; A = C)
    w6 = optimise2!(portfolio; obj = obj, rm = rm)
    wt = [1.3733290417870086e-6, 2.6763589401080658e-6, 2.9538542317039173e-6,
          0.0005900343176583919, 1.3887640713916724e-6, 1.3598728144018154e-6,
          8.979482586127353e-7, 3.591109234764808e-6, 6.310852949253323e-8,
          6.324340049812946e-7, 0.32590968729693326, 2.4443660959673622e-6,
          1.414358800261893e-7, 1.9293289965058625e-6, 0.01628800189959108,
          0.014069026506391764, 7.282630829588722e-6, 0.05452643821874623,
          -2.6476800486090808e-6, 0.24859272489979878]
    @test isapprox(w6.weights, wt)

    wc6 = optimise2!(portfolio; type = WC2(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w6.weights, wc6.weights)

    portfolio.a_vec_cent = []
    portfolio.b_cent = Inf

    portfolio.network_method = IP2(; A = B)
    w7 = optimise2!(portfolio; obj = obj, rm = rm)
    wt = [0.0, 0.0, -0.0, 0.0351778790727926, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0,
          0.28016082470423853, -0.0, -0.0, 0.0, -0.0, 0.017762653107240345, -0.0,
          0.20060384392516015, -0.0, 0.12629479919056838]
    @test isapprox(w7.weights, wt, rtol = 0.05)

    wc7 = optimise2!(portfolio; type = WC2(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w7.weights, wc7.weights, rtol = 0.05)

    portfolio.network_method = SDP2(; A = B)
    w8 = optimise2!(portfolio; obj = obj, rm = rm)
    wt = [3.353883669642267e-6, 0.038122465245421025, 0.012782948631217341,
          0.024251278321675323, 0.011026952156532094, 0.04193024239831407,
          -0.007723026091322901, 6.2145857665693815e-6, -3.39569902695355e-5,
          0.012648769570655168, 0.2453658971390818, -0.014844139342231548,
          -2.384652077201505e-6, 6.1640233295210345e-6, 0.006967498867791508,
          0.01709226942230279, 1.1245332165782372e-6, 0.17380674670205037,
          6.802767311385322e-7, 0.09859090131814645]
    @test isapprox(w8.weights, wt)

    wc8 = optimise2!(portfolio; type = WC2(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w8.weights, wc8.weights)

    portfolio.network_method = IP2(; A = C)
    w9 = optimise2!(portfolio; obj = obj, rm = rm)
    wt = [-0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, 0.40635001415589167,
          -0.0, -0.0, -0.0, 0.0, 0.03772335678460417, 0.0, -0.0, -0.0, 0.21592662905950424]
    @test isapprox(w9.weights, wt)

    wc9 = optimise2!(portfolio; type = WC2(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w9.weights, wc9.weights)

    portfolio.network_method = SDP2(; A = C)
    w10 = optimise2!(portfolio; obj = obj, rm = rm)
    wt = [1.645587229073337e-6, 1.0964650874070903e-6, 8.099324032340491e-7,
          4.750334438676555e-7, 5.708947010377271e-7, 5.026059745915548e-7,
          3.5554835828654246e-7, 8.862257420249835e-7, 1.6505843417208158e-7,
          2.4331260596876106e-7, 0.4039667416480195, 2.493579100224862e-7,
          7.165427252726094e-8, 2.0949825706215104e-6, 2.1348296632699007e-6,
          0.037534143273519546, 1.0029864740757473e-5, 2.983369616639739e-6,
          3.6844804532910475e-7, 0.21847443190766216]
    @test isapprox(w10.weights, wt)

    wc10 = optimise2!(portfolio; type = WC2(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w10.weights, wc10.weights)

    portfolio = Portfolio2(; prices = prices,
                           solvers = Dict(:PClGL => Dict(:solver => optimizer_with_attributes(Pajarito.Optimizer,
                                                                                              "verbose" => false,
                                                                                              "oa_solver" => optimizer_with_attributes(GLPK.Optimizer,
                                                                                                                                       MOI.Silent() => true),
                                                                                              "conic_solver" => optimizer_with_attributes(Clarabel.Optimizer,
                                                                                                                                          "verbose" => false,
                                                                                                                                          "max_step_fraction" => 0.75)))))
    asset_statistics2!(portfolio)
    wc_statistics2!(portfolio,
                    WCType(; box = WorstCaseNormal(; seed = 123456789),
                           ellipse = WorstCaseNormal(; seed = 123456789)))
    portfolio.short = true
    portfolio.short_u = 0.27
    portfolio.long_u = 0.81
    ssl2 = portfolio.sum_short_long

    obj = SR(; rf = rf)

    A = centrality_vector2(portfolio; network_type = TMFG())
    B = connection_matrix2(portfolio; network_type = TMFG())
    C = cluster_matrix2(portfolio; hclust_alg = DBHT())

    w11 = optimise2!(portfolio; obj = obj, rm = rm)
    wt = [1.3566877119847358e-9, 2.0446966457964675e-8, 0.002262512310895483,
          1.4309648003585085e-7, 0.2734983203742686, -0.10965529909211759,
          0.03892353698664605, 0.01819498928305758, 5.900712314743839e-9,
          1.0611222523062497e-8, 0.039700687003729467, -0.05489428919436992,
          -0.028441439852089117, 6.842779119067803e-10, -0.07700892954655005,
          0.10494492437890107, 0.14627523180964974, 6.830986121056796e-8,
          0.18619935145952116, 1.5367224934418446e-7]
    @test isapprox(w11.weights, wt)

    wc11 = optimise2!(portfolio; type = WC2(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w11.weights, wc11.weights, rtol = 5.0e-5)

    portfolio.a_vec_cent = A
    portfolio.b_cent = minimum(A)

    w12 = optimise2!(portfolio; obj = obj, rm = rm)
    wt = [2.1497119336493176e-10, 0.0088070179228199, 0.029397048718257417,
          0.019022694413270754, 0.2828940619983948, -0.12224670576806218,
          0.04353925700174413, 0.04156276108416902, 1.3097764426973018e-7,
          1.4902868835689554e-7, 0.09605820879721724, -0.04855626421146047,
          -0.029205822936867836, 1.055694863453504e-9, -0.0699910697420536,
          0.11603391472918954, 0.04641077698276719, 0.04812947034812958,
          0.07616841032531034, 0.001975959060175573]
    @test isapprox(w12.weights, wt)

    wc12 = optimise2!(portfolio; type = WC2(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w12.weights, wc12.weights, rtol = 5e-5)

    portfolio.network_method = IP2(; A = B)
    w13 = optimise2!(portfolio; obj = obj, rm = rm)
    wt = [1.5269128820151116e-10, 5.0153063190116795e-11, 5.600508563045624e-11,
          -0.1499999969120081, 5.516614800822251e-11, 8.547041188790406e-11,
          3.5801273917717314e-11, 1.256273861351736e-11, 2.409947687735074e-10,
          0.6899999950792018, 1.3622829299878125e-11, 4.821233895750819e-11,
          6.845347024420311e-11, 1.1561363674032775e-10, 6.517378250466465e-11,
          5.779913743888119e-11, 2.8740320583775717e-10, 4.066302469648558e-11,
          3.2976327105495704e-10, 1.1724076184220898e-10]
    @test isapprox(w13.weights, wt)

    wc13 = optimise2!(portfolio; type = WC2(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w13.weights, wc13.weights)

    portfolio.network_method = SDP2(; A = B)
    w14 = optimise2!(portfolio; obj = obj, rm = rm)
    wt = [6.120770021032183e-8, 1.6354124623024568e-6, -0.00014073619896361552,
          -1.5382765037317844e-5, 0.20801384458722627, -1.2971430048721393e-7,
          0.01815677072722214, 2.2037258513423353e-7, 6.723276350257066e-8,
          6.337618763433898e-8, 0.14038784849839608, -0.04511790824985411,
          -0.005307605149301374, 3.927932051708812e-8, -0.058853373768001094,
          0.06663138474637784, 7.950726727959057e-7, 0.12731744850449067,
          0.08892422332791153, 7.335001414362041e-7]
    @test isapprox(w14.weights, wt, rtol = 5.0e-8)

    wc14 = optimise2!(portfolio; type = WC2(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w14.weights, wc14.weights, rtol = 0.0001)

    portfolio.network_method = IP2(; A = C)
    w15 = optimise2!(portfolio; obj = obj, rm = rm)
    wt = [7.297197051787683e-13, 9.751495210303498e-13, 9.25146423310274e-13,
          9.90656747259057e-13, 0.5686937869795202, 6.347117711378158e-13,
          1.3067533184269999e-12, 1.0554342404566512e-12, 8.12208116452551e-13,
          9.444489676262045e-13, 1.0265787555201995e-12, 6.195482895301479e-13,
          -0.10244889726663442, 7.588262617099839e-13, 4.79429725214721e-13,
          1.2471770398993632e-12, 7.867540101053476e-13, 1.0021245942023042e-12,
          0.07375511027191975, 8.998095142341619e-13]
    @test isapprox(w15.weights, wt)

    wc15 = optimise2!(portfolio; type = WC2(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w15.weights, wc15.weights, rtol = 0.0005)

    portfolio.network_method = SDP2(; A = C)
    w16 = optimise2!(portfolio; obj = obj, rm = rm)
    wt = [1.4274592086142099e-8, 1.522783678899051e-8, 2.7938717235026846e-8,
          1.4561550826598941e-8, 0.2368909369265324, -0.0001506005786567205,
          -1.0081165111908498e-8, 1.7224029270673118e-5, 4.5248679067995964e-8,
          -0.00022728261938254014, 0.119886342638237, -2.4823635865134306e-7,
          -3.413454237398767e-6, -1.0451687138374718e-8, -2.8921310398193577e-6,
          0.08684122100032984, 0.0035296823817011595, 7.622590527125835e-9,
          0.09321873846044709, 1.8724204259728512e-7]
    @test isapprox(w16.weights, wt)

    wc16 = optimise2!(portfolio; type = WC2(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w16.weights, wc16.weights, rtol = 0.0005)

    portfolio.a_vec_cent = []
    portfolio.b_cent = Inf

    portfolio.network_method = IP2(; A = B)
    w17 = optimise2!(portfolio; obj = obj, rm = rm)
    wt = [-0.0, -0.0, -0.0, -0.0, 0.8099999999999998, -0.0, -0.0, -0.0, -0.0, -0.0, 0.0,
          -0.0, -0.0, -0.0, -0.27, -0.0, 0.0, -0.0, 0.0, -0.0]
    @test isapprox(w17.weights, wt)

    wc17 = optimise2!(portfolio; type = WC2(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w17.weights, wc17.weights)

    portfolio.network_method = SDP2(; A = B)
    w18 = optimise2!(portfolio; obj = obj, rm = rm)
    wt = [3.7372455923638106e-8, 4.0946527532379914e-7, 5.741510057885535e-8,
          3.4014209639844175e-8, 0.1986252065725683, -8.486609959409557e-8,
          0.023951161167138676, 1.8655603296653543e-7, 3.192609402220337e-8,
          4.19296704875058e-8, 0.18694537859607946, -0.055626176371317476,
          -2.153219858731183e-8, 1.6743729639380584e-8, -0.05376684092336848,
          0.08378904151649921, 8.526537485802294e-8, 0.1560810784728213,
          9.345153848505544e-8, 2.632283952726144e-7]
    @test isapprox(w18.weights, wt)

    wc18 = optimise2!(portfolio; type = WC2(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w18.weights, wc18.weights, rtol = 0.0001)

    portfolio.network_method = IP2(; A = C)
    w19 = optimise2!(portfolio; obj = obj, rm = rm)
    wt = [-0.0, -0.0, -0.0, -0.0, 0.6011773438073398, -0.27, 0.0, -0.0, -0.0, 0.0, 0.0,
          -0.0, -0.0, -0.0, -0.0, 0.20882265619266027, 0.0, -0.0, 0.0, -0.0]
    @test isapprox(w19.weights, wt, rtol = 0.0005)

    wc19 = optimise2!(portfolio; type = WC2(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w19.weights, wc19.weights, rtol = 0.0005)

    portfolio.network_method = SDP2(; A = C)
    w20 = optimise2!(portfolio; obj = obj, rm = rm)
    wt = [1.9089390697715466e-8, 2.229133765919515e-8, 3.281824986291854e-8,
          1.8749370001662676e-8, 0.2360852915904294, -8.445636169630471e-6,
          -1.2148952122020842e-9, 1.2285635879456866e-6, 5.003357016527931e-8,
          -0.0002758503591244658, 0.08672896660586178, -2.649201973767617e-7,
          -5.065177762345351e-7, -1.1746904762175866e-8, -9.7236488086004e-7,
          0.0852459526649671, 0.0035191828565809746, 1.6956403200522112e-8,
          0.128705177720426, 9.28197738427405e-8]
    @test isapprox(w20.weights, wt)

    wc20 = optimise2!(portfolio; type = WC2(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w20.weights, wc20.weights, rtol = 0.001)
end

@testset "Network and Dendrogram upper dev" begin
    portfolio = Portfolio2(; prices = prices,
                           solvers = Dict(:PClGL => Dict(:solver => optimizer_with_attributes(Pajarito.Optimizer,
                                                                                              "verbose" => false,
                                                                                              "oa_solver" => optimizer_with_attributes(GLPK.Optimizer,
                                                                                                                                       MOI.Silent() => true),
                                                                                              "conic_solver" => optimizer_with_attributes(Clarabel.Optimizer,
                                                                                                                                          "verbose" => false,
                                                                                                                                          "max_step_fraction" => 0.75)))))
    asset_statistics2!(portfolio)
    B = connection_matrix2(portfolio; network_type = TMFG())

    rm = SD2()
    w1 = optimise2!(portfolio; obj = SR(; rf = rf), rm = rm)
    r1 = calc_risk(portfolio; rm = rm)

    rm.settings.ub = r1
    portfolio.network_method = IP2(; A = B)
    w2 = optimise2!(portfolio; obj = MinRisk(), rm = rm)
    r2 = calc_risk(portfolio; rm = rm)
    w3 = optimise2!(portfolio; obj = Util(; l = l), rm = rm)
    r3 = calc_risk(portfolio; rm = rm)
    w4 = optimise2!(portfolio; obj = SR(; rf = rf), rm = rm)
    r4 = calc_risk(portfolio; rm = rm)
    w5 = optimise2!(portfolio; obj = MaxRet(), rm = rm)
    r5 = calc_risk(portfolio; rm = rm)
    @test r2 <= r1
    @test r3 <= r1 || abs(r3 - r1) < 5e-8
    @test r4 <= r1
    @test r5 <= r1 || abs(r5 - r1) < 5e-8

    portfolio.network_method = SDP2(; A = B)
    w6 = optimise2!(portfolio; obj = MinRisk(), rm = rm)
    r6 = calc_risk(portfolio; rm = rm)
    w7 = optimise2!(portfolio; obj = Util(; l = l), rm = rm)
    r7 = calc_risk(portfolio; rm = rm)
    w8 = optimise2!(portfolio; obj = SR(; rf = rf), rm = rm)
    r8 = calc_risk(portfolio; rm = rm)
    w9 = optimise2!(portfolio; obj = MaxRet(), rm = rm)
    r9 = calc_risk(portfolio; rm = rm)
    @test r6 <= r1
    @test r7 <= r1
    @test r8 <= r1
    @test r9 <= r1

    portfolio = Portfolio2(; prices = prices,
                           solvers = Dict(:PClGL => Dict(:solver => optimizer_with_attributes(Pajarito.Optimizer,
                                                                                              "verbose" => false,
                                                                                              "oa_solver" => optimizer_with_attributes(GLPK.Optimizer,
                                                                                                                                       MOI.Silent() => true),
                                                                                              "conic_solver" => optimizer_with_attributes(Clarabel.Optimizer,
                                                                                                                                          "verbose" => false,
                                                                                                                                          "max_step_fraction" => 0.75)))))
    asset_statistics2!(portfolio)
    rm = [[SD2(), SD2()]]
    w10 = optimise2!(portfolio; obj = SR(; rf = rf), rm = rm)
    r10 = calc_risk(portfolio; rm = rm[1][1])

    rm[1][1].settings.ub = r10
    portfolio.network_method = IP2(; A = B)
    w11 = optimise2!(portfolio; obj = MinRisk(), rm = rm)
    r11 = calc_risk(portfolio; rm = rm[1][1])
    w12 = optimise2!(portfolio; obj = Util(; l = l), rm = rm)
    r12 = calc_risk(portfolio; rm = rm[1][1])
    w13 = optimise2!(portfolio; obj = SR(; rf = rf), rm = rm)
    r13 = calc_risk(portfolio; rm = rm[1][1])
    w14 = optimise2!(portfolio; obj = MaxRet(), rm = rm)
    r14 = calc_risk(portfolio; rm = rm[1][1])
    @test r11 <= r10
    @test r12 <= r10 || abs(r12 - r10) < 5e-7
    @test r13 <= r10 || abs(r13 - r10) < 1e-10
    @test r14 <= r10 || abs(r14 - r10) < 5e-7

    portfolio.network_method = SDP2(; A = B)
    w15 = optimise2!(portfolio; obj = MinRisk(), rm = rm)
    r15 = calc_risk(portfolio; rm = rm[1][1])
    w16 = optimise2!(portfolio; obj = Util(; l = l), rm = rm)
    r16 = calc_risk(portfolio; rm = rm[1][1])
    w17 = optimise2!(portfolio; obj = SR(; rf = rf), rm = rm)
    r17 = calc_risk(portfolio; rm = rm[1][1])
    w18 = optimise2!(portfolio; obj = MaxRet(), rm = rm)
    r18 = calc_risk(portfolio; rm = rm[1][1])
    @test r15 <= r10
    @test r16 <= r10
    @test r17 <= r10
    @test r18 <= r10

    portfolio = portfolio = Portfolio2(; prices = prices,
                                       solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                                        :params => Dict("verbose" => false))))
    asset_statistics2!(portfolio)

    rm = [SD2(; settings = RiskMeasureSettings(; flag = false)), CDaR2()]

    portfolio.network_method = SDP2(; A = B)
    w19 = optimise2!(portfolio; kelly = AKelly(), obj = SR(; rf = rf), rm = rm)
    w20 = optimise2!(portfolio; kelly = EKelly(), obj = SR(; rf = rf), rm = rm)
    @test isapprox(w19.weights, w20.weights)

    w21 = optimise2!(portfolio; type = RP2(), rm = rm)
    portfolio.network_method = IP2(; A = B)
    w22 = optimise2!(portfolio; type = RP2(), rm = rm)
    portfolio.network_method = NoNtwk()
    w23 = optimise2!(portfolio; type = RP2(), rm = rm)
    @test isapprox(w21.weights, w22.weights)
    @test isapprox(w21.weights, w23.weights)
    @test isapprox(w22.weights, w23.weights)
end

@testset "Network and Dendrogram non SD" begin
    portfolio = Portfolio2(; prices = prices,
                           solvers = Dict(:PClGL => Dict(:solver => optimizer_with_attributes(Pajarito.Optimizer,
                                                                                              "verbose" => false,
                                                                                              "oa_solver" => optimizer_with_attributes(GLPK.Optimizer,
                                                                                                                                       MOI.Silent() => true),
                                                                                              "conic_solver" => optimizer_with_attributes(Clarabel.Optimizer,
                                                                                                                                          "verbose" => false,
                                                                                                                                          "max_step_fraction" => 0.75)))))
    asset_statistics2!(portfolio)

    A = centrality_vector2(portfolio; network_type = TMFG())
    B = connection_matrix2(portfolio; network_type = TMFG())
    C = cluster_matrix2(portfolio; hclust_alg = DBHT())

    rm = CDaR2()
    obj = MinRisk()
    w1 = optimise2!(portfolio; obj = obj, rm = rm)
    wt = [0.0, 5.74250749576219e-17, 0.0, 0.0, 0.0034099011531647325, 0.0, 0.0,
          0.07904282391685276, 0.0, 0.0, 0.3875931702023311, 0.0, 0.0,
          2.1342373233586235e-18, 0.0005545152741455163, 0.09598828930573457,
          0.26790888256716056, 0.0, 0.000656017191282443, 0.16484640038932813]
    @test isapprox(w1.weights, wt)

    portfolio.a_vec_cent = A
    portfolio.b_cent = minimum(A)
    w2 = optimise2!(portfolio; obj = obj, rm = rm)
    wt = [0.0, 0.0, 0.0, 0.051723138879499364, 0.0, 0.0, 0.05548699473292653,
          0.25666738948843576, 0.0, 0.0, 0.5073286323951223, 0.12879384450401607, 0.0, 0.0,
          0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    @test isapprox(w2.weights, wt)

    portfolio.network_method = IP2(; A = B)
    w3 = optimise2!(portfolio; obj = obj, rm = rm)
    wt = [0.0, 0.0, 0.0, 0.13808597886475105, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
          0.861914021135249, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    @test isapprox(w3.weights, wt)

    portfolio.network_method = SDP2(; A = B)
    w4 = optimise2!(portfolio; obj = obj, rm = rm)
    wt = [6.916528681040479e-11, 2.861831047852453e-11, 1.7205226727441286e-11,
          0.0706145671003988, 3.0536778285536253e-9, 6.76023773113509e-11,
          0.07499015177704357, 0.26806573786222676, 7.209644329535395e-11,
          6.263090324526159e-11, 0.41654672020086003, 0.16978281921567007,
          6.213186671810222e-11, 6.969908419111133e-11, 2.3053255039158856e-12,
          8.392552900081856e-11, 7.289286270405327e-11, 4.247286241831121e-11,
          7.335225686899845e-11, 6.602480341871697e-11]
    @test isapprox(w4.weights, wt)

    portfolio.network_method = IP2(; A = C)
    w5 = optimise2!(portfolio; obj = obj, rm = rm)
    wt = [0.0, 0.0, 0.0, 1.5501398315533836e-16, 6.703062176969295e-17, 0.0, 0.0,
          0.23899933639488402, 0.0, 0.0, 0.7610006636051158, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
          0.0, 0.0, 0.0]
    @test isapprox(w5.weights, wt)

    portfolio.network_method = SDP2(; A = C)
    w6 = optimise2!(portfolio; obj = obj, rm = rm)
    wt = [2.2433050416665602e-11, 7.367350133776243e-12, 1.028483946149567e-12,
          0.05175588225120212, 9.626152454135536e-10, 2.188948515818977e-11,
          0.05231788616170087, 0.2808759207113746, 2.3399824709341706e-11,
          2.0271505690336202e-11, 0.47619910136821797, 0.13885120828718264,
          8.152746608851178e-12, 2.264714662939521e-11, 3.124732104779394e-12,
          4.578061717214521e-11, 2.365333386726047e-11, 1.2764462709179066e-11,
          2.380632701160386e-11, 2.138756409983987e-11]
    @test isapprox(w6.weights, wt)

    portfolio.a_vec_cent = []
    portfolio.b_cent = Inf

    portfolio.network_method = IP2(; A = B)
    w7 = optimise2!(portfolio; obj = obj, rm = rm)
    wt = [0.0, 0.0, 0.0, 0.0, 0.37297150326295364, 0.0, 0.0, 0.0, 0.0, 0.0,
          0.6270284967370463, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    @test isapprox(w7.weights, wt)

    portfolio.network_method = SDP2(; A = B)
    w8 = optimise2!(portfolio; obj = obj, rm = rm)
    wt = [5.333299624468313e-11, 0.05638785424268776, 1.801388361336222e-10,
          5.248225672626687e-11, 0.08206405829302137, 1.7756997488326776e-11,
          1.9025293508818056e-12, 0.13507762926776312, 1.3391306712204848e-10,
          3.7873704079316313e-11, 0.3175440779337929, 0.020298213899228854,
          4.164020888733719e-11, 1.174242267435215e-10, 4.7848187362837886e-11,
          0.12477101930240944, 0.07961825077560757, 0.03009048638763158,
          8.237622588564551e-11, 0.15414840913116812]
    @test isapprox(w8.weights, wt)

    portfolio.network_method = IP2(; A = C)
    w9 = optimise2!(portfolio; obj = obj, rm = rm)
    wt = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4887092443113238, 0.0, 0.0,
          0.0, 0.0, 0.06771233053987868, 0.4435784251487976, 0.0, 0.0, 0.0]
    @test isapprox(w9.weights, wt)

    portfolio.network_method = SDP2(; A = C)
    w10 = optimise2!(portfolio; obj = obj, rm = rm)
    wt = [8.875598045563749e-11, 6.786115952224246e-11, 1.403756249532236e-10,
          3.8200133576780337e-11, 0.026405191197268175, 2.7118263798158232e-11,
          4.287027282530246e-12, 0.08423696478830205, 1.4264703610446034e-10,
          2.3728313538121046e-10, 0.3865644514064455, 0.009574217972314737,
          4.844735058003537e-11, 1.8846516237664424e-10, 1.2713658919263283e-11,
          0.14421172180536518, 0.2209166214591706, 2.7179501902437763e-9,
          0.0006754849417179316, 0.12741534271531094]
    @test isapprox(w10.weights, wt)

    portfolio = Portfolio2(; prices = prices,
                           solvers = Dict(:PClGL => Dict(:solver => optimizer_with_attributes(Pajarito.Optimizer,
                                                                                              "verbose" => false,
                                                                                              "oa_solver" => optimizer_with_attributes(GLPK.Optimizer,
                                                                                                                                       MOI.Silent() => true),
                                                                                              "conic_solver" => optimizer_with_attributes(Clarabel.Optimizer,
                                                                                                                                          "verbose" => false,
                                                                                                                                          "max_step_fraction" => 0.75)))))
    asset_statistics2!(portfolio)
    obj = SR(; rf = rf)

    A = centrality_vector2(portfolio; network_type = TMFG())
    B = connection_matrix2(portfolio; network_type = TMFG())
    C = cluster_matrix2(portfolio; hclust_alg = DBHT())

    w11 = optimise2!(portfolio; obj = obj, rm = rm)
    wt = [0.0, 0.0, 0.07233520665470376, 0.0, 0.3107248736916702, 0.0, 0.0,
          0.12861270774687708, 0.0, 0.0, 0.16438408898657855, 0.0, 0.0, 0.0, 0.0,
          0.2628826637767333, 0.0, 0.0, 0.0, 0.061060459143437176]
    @test isapprox(w11.weights, wt)

    portfolio.a_vec_cent = A
    portfolio.b_cent = minimum(A)

    w12 = optimise2!(portfolio; obj = obj, rm = rm)
    wt = [0.0, 0.0, 0.0, 0.13626926674571754, 1.515977496012877e-16, 0.0,
          0.2529009327082102, 0.0, 0.0, 0.0, 0.4850451479068739, 0.12578465263919827, 0.0,
          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    @test isapprox(w12.weights, wt)

    portfolio.network_method = IP2(; A = B)
    w13 = optimise2!(portfolio; obj = obj, rm = rm)
    wt = [0.0, 0.0, 0.0, 0.0, 1.4641796690079654e-17, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    @test isapprox(w13.weights, wt)

    portfolio.network_method = SDP2(; A = B)
    w14 = optimise2!(portfolio; obj = obj, rm = rm)
    wt = [3.9876730635706574e-10, 9.058885429652311e-11, 1.5225868210279566e-11,
          0.15463223546511348, 1.787917922937287e-8, 3.89830536537106e-10,
          0.2677212654853749, 0.05396055329565413, 4.2501077924645286e-10,
          3.3984441830986676e-10, 0.353757964593812, 0.16992795957233114,
          2.6507986578852124e-11, 4.0584551279201494e-10, 2.2729721750159754e-10,
          3.426736037949514e-12, 4.169695539917153e-10, 1.7568326295730658e-10,
          4.272760538261085e-10, 3.662610153313769e-10]
    @test isapprox(w14.weights, wt)

    portfolio.network_method = IP2(; A = C)
    w15 = optimise2!(portfolio; obj = obj, rm = rm)
    wt = [0.0, 0.0, 0.0, 0.0, 6.313744038982638e-18, 0.0, 0.28939681985466575, 0.0, 0.0,
          0.0, 0.7106031801453343, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    @test isapprox(w15.weights, wt)

    portfolio.network_method = SDP2(; A = C)
    w16 = optimise2!(portfolio; obj = obj, rm = rm)
    wt = [3.0604243778282573e-10, 8.137264759368085e-11, 2.0929602956552645e-11,
          0.1477170963925791, 1.3249679846056793e-8, 2.982689000604614e-10,
          0.29216838667211203, 7.376556540494882e-9, 3.164565128243966e-10,
          2.618365745527757e-10, 0.4227708041709823, 0.13734368926853047,
          4.137476730712234e-11, 3.121559703851553e-10, 1.8649684695770075e-10,
          1.3020396700053977e-11, 2.9600184890250864e-10, 1.4425665165982614e-10,
          3.0895233240149184e-10, 2.8239402646536884e-10]
    @test isapprox(w16.weights, wt)

    portfolio.a_vec_cent = []
    portfolio.b_cent = Inf

    portfolio.network_method = IP2(; A = B)
    w17 = optimise2!(portfolio; obj = obj, rm = rm)
    wt = [0.0, 0.0, 0.0, 0.0, 0.7390777009270599, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
          0.0, 0.0, 0.26092229907294007, 0.0, 0.0, 0.0, 0.0]
    @test isapprox(w17.weights, wt)

    portfolio.network_method = SDP2(; A = B)
    w18 = optimise2!(portfolio; obj = obj, rm = rm)
    wt = [1.4680691018542715e-9, 4.859275738822192e-9, 0.015992802521425674,
          7.293948583023383e-10, 0.2988850375190448, 6.520933919634565e-11,
          2.128743388395745e-9, 0.16470653869543705, 4.624274795849735e-10,
          4.162708987445042e-10, 0.13583336339414312, 1.525142217002498e-11,
          2.494845225126199e-10, 1.0043814255652154e-9, 9.599421959136733e-11,
          0.2560211483665152, 4.277938501034575e-9, 1.880279259853607e-9,
          2.4445455761302985e-9, 0.12856108940616845]
    @test isapprox(w18.weights, wt)

    portfolio.network_method = IP2(; A = C)
    w19 = optimise2!(portfolio; obj = obj, rm = rm)
    wt = [0.0, 0.0, 0.0, 0.0, 0.3708533501847804, 0.0, 0.0, 0.0, 0.0, 0.0,
          0.3290504062456894, 0.0, 0.0, 0.0, 0.0, 0.3000962435695302, 0.0, 0.0, 0.0, 0.0]
    @test isapprox(w19.weights, wt)

    portfolio.network_method = SDP2(; A = C)
    w20 = optimise2!(portfolio; obj = obj, rm = rm)
    wt = [8.477768420059939e-10, 6.203676538852534e-10, 0.05312672191525725,
          3.525182424405551e-10, 0.3338885551569258, 2.412376047604793e-11,
          1.2535190055486572e-9, 0.10330170626208342, 2.6852440963129434e-10,
          3.0123040045012183e-10, 0.2215936600067496, 3.15192092477848e-11,
          1.19545885901302e-10, 5.880051207030092e-10, 9.099593094339936e-11,
          0.2880893428896174, 1.7533990524576498e-9, 6.973233642519765e-10,
          2.7663596217040672e-9, 4.0541581354667404e-9]
    @test isapprox(w20.weights, wt)
end

@testset "Network and Dendrogram non SD Short" begin
    portfolio = Portfolio2(; prices = prices,
                           solvers = Dict(:PClGL => Dict(:solver => optimizer_with_attributes(Pajarito.Optimizer,
                                                                                              "verbose" => false,
                                                                                              "oa_solver" => optimizer_with_attributes(GLPK.Optimizer,
                                                                                                                                       MOI.Silent() => true),
                                                                                              "conic_solver" => optimizer_with_attributes(Clarabel.Optimizer,
                                                                                                                                          "verbose" => false,
                                                                                                                                          "max_step_fraction" => 0.75)))))
    asset_statistics2!(portfolio)

    A = centrality_vector2(portfolio)
    B = connection_matrix2(portfolio)
    C = cluster_matrix2(portfolio)

    portfolio.short = true
    portfolio.short_u = 0.13
    portfolio.long_u = 1

    rm = CDaR2()
    obj = MinRisk()
    w1 = optimise2!(portfolio; obj = obj, rm = rm)
    wt = [0.0, -0.0033927423088478525, 0.0, -0.0017047316992726633, 0.02620966174092891,
          -0.048951276707458066, -0.02831053066644215, 0.0797084661157905,
          -0.006091550656354475, -0.01772949018065388, 0.34963750962788587,
          -0.009810028561542944, 0.006616206609090757, -1.7025233012195975e-16,
          -0.01400964921942798, 0.09738313723086256, 0.16435744872368938,
          0.01643568935570537, 0.10356751838407428, 0.15608436221197253]
    @test isapprox(w1.weights, wt)

    portfolio.a_vec_cent = A
    portfolio.b_cent = minimum(A)
    w2 = optimise2!(portfolio; obj = obj, rm = rm)
    wt = [-0.09486167717442834, 0.04019416692914485, 0.037577223179041455,
          -5.0655785389293176e-17, 0.12801741187716315, -7.979727989493313e-17,
          -0.01998095976345068, 0.13355629824656456, 0.019271038397774838, 0.0,
          0.28797176886025655, 0.0, 0.008845283123909234, 0.0, -0.015157363062120899,
          0.10734222426525322, 0.0, 0.061474266941201526, 4.119968255444917e-17,
          0.17575031817969053]
    @test isapprox(w2.weights, wt)

    portfolio.network_method = IP2(; A = B, penalty = 0.5)
    w3 = optimise2!(portfolio; obj = obj, rm = rm)
    wt = [-0.0, 0.0, 0.0, -0.0, 0.25423835281660595, -0.10786045982487914, -0.0,
          0.3290798306064528, -0.0, -0.0, 2.220446049250313e-16, -0.0, 0.0, -0.0,
          0.02916010168515354, 0.13000000000000062, -0.0, 0.0, -0.0, 0.23538217471666556]
    @test isapprox(w3.weights, wt)

    portfolio.network_method = SDP2(; A = B, penalty = 0.5)
    w4 = optimise2!(portfolio; obj = obj, rm = rm)
    wt = [9.272435109943138e-12, 0.06109011163471849, 0.07388719114292448,
          0.019692527431470045, 0.10331812359040753, 0.08117018361193389,
          0.0233844026939607, 0.12195682880133857, 1.687054000923196e-11,
          0.034579519160302545, 0.02795355513256586, 0.05841592129946603,
          7.482492434260051e-12, 0.029412363249472755, 0.0020709632770030445,
          0.04322171854228879, -8.422304421932733e-13, 0.08680666625945271,
          -3.585326191328158e-11, 0.10303992417576462]
    @test isapprox(w4.weights, wt)

    portfolio.network_method = IP2(; A = C, penalty = 0.5)
    w5 = optimise2!(portfolio; obj = obj, rm = rm)
    wt = [-0.0, -9.824982518806696e-17, -2.881746754291979e-16, -6.328762489489334e-17,
          0.3345846888561428, -0.0, -0.0, 0.4054153111438571, -0.0, -0.0, 0.0, 0.0, 0.0,
          -0.0, -0.0, 0.13000000000000017, -0.0, 0.0, -0.0, 0.0]
    @test isapprox(w5.weights, wt)

    portfolio.network_method = SDP2(; A = C, penalty = 0.5)
    w6 = optimise2!(portfolio; obj = obj, rm = rm)
    wt = [1.6792221684864492e-11, 7.752448098385652e-10, 0.02435598503520649,
          8.231084843588821e-11, 0.1847117617985036, 1.4177904571247094e-10,
          0.022357207427965185, 0.2751582759071143, 3.067483602511589e-11,
          9.505278645317588e-11, 0.030652168623148388, 0.07645421112479128,
          0.000723625668406513, 2.8213000163156964e-11, 0.03653126520849825,
          0.09934783116298963, 1.0942656031997705e-11, 0.005474421241494843,
          5.436551647491036e-12, 0.11423324561543377]
    @test isapprox(w6.weights, wt)

    portfolio.a_vec_cent = []
    portfolio.b_cent = Inf

    portfolio.network_method = IP2(; A = B, penalty = 0.5)
    w7 = optimise2!(portfolio; obj = obj, rm = rm)
    wt = [-0.0, -0.0, -0.0, 3.428633578839961e-17, 0.1119976188327676, -0.0, -0.0, 0.0, 0.0,
          0.0, 0.3750880542218098, -0.0, -0.0, -0.0, -0.0, 0.1412763749809966, -0.0,
          0.09588426469923997, 0.0, 0.14575368726518567]
    @test isapprox(w7.weights, wt)

    portfolio.network_method = SDP2(; A = B, penalty = 0.5)
    w8 = optimise2!(portfolio; obj = obj, rm = rm)
    wt = [6.766929954587163e-11, 0.05845600849974435, 0.06857721044042771,
          0.01863708044634632, 0.09512946481942187, 0.07994346654887266,
          0.02554585983141323, 0.07233232435238945, 5.714153552243313e-11,
          0.032112301662597306, 0.08806729718852686, 0.046990727441064695,
          2.302281082192624e-11, 0.041083140039804476, 0.001683242401425925,
          0.057989539665591214, 6.10236089865971e-11, 0.0822728003096611,
          3.5401881596356065e-11, 0.10117953610845386]
    @test isapprox(w8.weights, wt)

    portfolio.network_method = IP2(; A = C, penalty = 0.5)
    w9 = optimise2!(portfolio; obj = obj, rm = rm)
    wt = [0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 4.8338121699908494e-17, 0.0, -0.0, -0.0,
          0.425177042550851, -0.0, -1.0139121127832994e-15, -0.0, 2.2724003756919815e-16,
          0.05890972756969809, 0.3859132298794514, 0.0, -0.0, 0.0]
    @test isapprox(w9.weights, wt)

    portfolio.network_method = SDP2(; A = C, penalty = 0.5)
    w10 = optimise2!(portfolio; obj = obj, rm = rm)
    wt = [3.136684399611615e-10, 6.421446869383163e-11, 0.015548662828311242,
          5.302443988456493e-11, 0.07187792498761011, 3.443518625849509e-11,
          0.014549230934352904, 0.10906579897534709, 2.8604004872211283e-11,
          2.3132104858134452e-11, 0.23095115947815376, 0.044332984975470155,
          1.7809249411779717e-10, 4.072456628263281e-11, 0.005860251167461893,
          0.16234482039541365, 0.07541584072146104, 6.086150708285326e-11,
          5.436944458782648e-11, 0.14005332468529116]
    @test isapprox(w10.weights, wt)

    portfolio = Portfolio2(; prices = prices,
                           solvers = Dict(:PClGL => Dict(:solver => optimizer_with_attributes(Pajarito.Optimizer,
                                                                                              "verbose" => false,
                                                                                              "oa_solver" => optimizer_with_attributes(GLPK.Optimizer,
                                                                                                                                       MOI.Silent() => true),
                                                                                              "conic_solver" => optimizer_with_attributes(Clarabel.Optimizer,
                                                                                                                                          "verbose" => false,
                                                                                                                                          "max_step_fraction" => 0.75)))))
    asset_statistics2!(portfolio)
    obj = SR(; rf = rf)

    A = centrality_vector2(portfolio; network_type = TMFG())
    B = connection_matrix2(portfolio; network_type = TMFG())
    C = cluster_matrix2(portfolio; hclust_alg = DBHT())

    portfolio.short = true
    portfolio.short_u = 0.18
    portfolio.long_u = 0.95

    w11 = optimise2!(portfolio; obj = obj, rm = rm)
    wt = [0.0, 0.0, 0.06455652837124166, 0.0, 0.17806945118716286, -0.1534166903019555,
          6.554560797119506e-17, 0.12351856122756986, 0.0, 0.0, 0.21867000759778857, 0.0,
          -0.023295149141940166, 0.0, -0.0032881605561044004, 0.1770017931069312,
          0.006143447950208067, 0.0, 0.04071674553919559, 0.1413234650199022]
    @test isapprox(w11.weights, wt)

    portfolio.a_vec_cent = A
    portfolio.b_cent = minimum(A)

    w12 = optimise2!(portfolio; obj = obj, rm = rm)
    wt = [0.0, 0.0, 0.07328869996428454, 0.0, 0.19593630860627667, -0.1516035031548817,
          -2.228175066036319e-17, 0.1290693464408622, 0.0, 0.0, 0.20897763141227396, 0.0,
          -0.02192543103531701, 0.0, -0.006471065809801198, 0.18010601289300754,
          7.366923752795404e-17, 0.0, 0.025951685743898865, 0.13667031493939605]
    @test isapprox(w12.weights, wt)

    portfolio.network_method = IP2(; A = B, penalty = 0.5)
    w13 = optimise2!(portfolio; obj = obj, rm = rm)
    wt = [-0.0, -0.0, 0.0, -0.0, 0.6900000000000023, -0.0, -0.0, 0.0, -0.0, -0.0,
          0.19949253798767527, -0.0, -0.11949253798767445, -0.0, -0.0, 0.0, 0.0, -0.0, 0.0,
          0.0]
    @test isapprox(w13.weights, wt)

    portfolio.network_method = SDP2(; A = B, penalty = 0.5)
    w14 = optimise2!(portfolio; obj = obj, rm = rm)
    wt = [3.514727061450158e-11, 0.10214522082786334, 1.5973615263620542e-9,
          8.750301284842235e-10, 0.23010239598778515, -0.025107535823589502,
          0.022839041934182035, 0.1484093093837071, -2.6374371040425877e-10,
          -3.153936811484647e-11, 0.12599682872769954, -6.592141286737157e-10,
          -0.03303460329907297, -1.0047674687409329e-10, -0.05977664458614574,
          0.17110148065348418, -7.483961562407158e-11, 5.423167113303609e-10,
          -2.6450191488394353e-10, 0.08732450453854651]
    @test isapprox(w14.weights, wt)

    portfolio.network_method = IP2(; A = C, penalty = 0.5)
    w15 = optimise2!(portfolio; obj = obj, rm = rm)
    wt = [-0.0, -0.0, 0.0, -0.0, 0.6899999999999927, -0.0, -0.0, 0.0, -0.0, -0.0,
          0.1994925379876753, -0.0, -0.11949253798766485, -0.0, -0.0, 0.0, -0.0, -0.0, 0.0,
          0.0]
    @test isapprox(w15.weights, wt)

    portfolio.network_method = SDP2(; A = C, penalty = 0.5)
    w16 = optimise2!(portfolio; obj = obj, rm = rm)
    wt = [3.1345471980832356e-10, 2.469910273202323e-10, 7.946447462097655e-10,
          1.7210857874043897e-10, 0.354561355808636, -3.665814844245118e-10,
          8.522943700016685e-10, 2.9715927145570816e-10, 8.376446627301666e-10,
          1.3758357758118232e-10, 0.16641976055535695, -2.8469632582700114e-10,
          -0.014268236579253484, 4.2616226395158317e-10, -0.01533295006816714,
          0.272938209489049, 4.78233132976098e-10, 1.5817077710699714e-10,
          0.005681856375922364, 3.5528715646533134e-10]
    @test isapprox(w16.weights, wt)

    portfolio.a_vec_cent = []
    portfolio.b_cent = Inf

    portfolio.network_method = IP2(; A = B, penalty = 0.5)
    w17 = optimise2!(portfolio; obj = obj, rm = rm)
    wt = [-0.0, -0.0, 0.0, -0.0, 0.6751155811095629, -0.0, -0.0, 0.0, -0.0, -0.0,
          0.208898224965609, -0.0, -0.11401380607517049, -0.0, -0.0, 0.0, 0.0, -0.0, -0.0,
          0.0]
    @test isapprox(w17.weights, wt)

    portfolio.network_method = SDP2(; A = B, penalty = 0.5)
    w18 = optimise2!(portfolio; obj = obj, rm = rm)
    wt = [9.396464919780623e-11, 0.09919154701647347, 6.368353684023622e-10,
          3.465978583702456e-10, 0.22064663460078363, -1.0614315396176674e-9,
          0.020025141314969998, 0.1266686482714095, 5.6795111561929726e-12,
          1.1754682802203758e-11, 0.12107267096223644, -4.262330938609224e-10,
          -0.04188688600867918, 1.5637094635344307e-10, -0.06521868018990389,
          0.16284169111496827, 6.247282655579725e-11, 4.814732694037535e-10,
          1.7210187767534034e-10, 0.12665923243815538]
    @test isapprox(w18.weights, wt)

    portfolio.network_method = IP2(; A = C, penalty = 0.5)
    w19 = optimise2!(portfolio; obj = obj, rm = rm)
    wt = [-0.0, -0.0, 0.0, -0.0, 0.5385298461146729, -0.0, 0.0, 0.0, -0.0, -0.0, 0.0,
          -0.18000000000000382, -0.0, 0.0, 3.8379398333663466e-15, 0.411470153885331, -0.0,
          -0.0, -0.0, 0.0]
    @test isapprox(w19.weights, wt)

    portfolio.network_method = SDP2(; A = C, penalty = 0.5)
    w20 = optimise2!(portfolio; obj = obj, rm = rm)
    wt = [1.4172669439373994e-10, 1.1299075883203906e-10, 5.359434439554113e-10,
          8.26042302579247e-11, 0.3649711359888737, -6.90608391192042e-11,
          4.0350791884642237e-10, 2.0028585646891752e-10, 2.1735756891215644e-10,
          1.0421359032360126e-10, 0.11284986034990291, -6.494847118187474e-11,
          -0.027835392456788028, 1.767593240750613e-10, -0.015067449302819187,
          0.2566729326870001, 1.7852927573841912e-10, 7.917164427601336e-11,
          0.07840891046398553, 1.7076400157132798e-10]
    @test isapprox(w20.weights, wt, rtol = 5.0e-8)
end