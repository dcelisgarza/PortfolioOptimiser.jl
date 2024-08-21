using CSV, TimeSeries, DataFrames, StatsBase, Statistics, LinearAlgebra, Test, GLPK,
      Pajarito, JuMP, Clarabel, PortfolioOptimiser

prices = TimeArray(CSV.File("./test/assets/stock_prices.csv"); timestamp = :date)
rf = 1.0329^(1 / 252) - 1
l = 2.0

TrackWeight()
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
    portfolio.tracking_err = TrackWeight(; e = te1, w = tw1)
    w2 = optimise2!(portfolio; obj = MinRisk())
    @test norm(portfolio.returns * (w2.weights - tw1), 2) / sqrt(T - 1) <= te1

    w3 = optimise2!(portfolio; obj = MinRisk())
    te2 = 0.0003
    tw2 = copy(w3.weights)
    portfolio.tracking_err = TrackWeight(; e = te2, w = tw2)
    w4 = optimise2!(portfolio; obj = SR(; rf = rf))
    @test norm(portfolio.returns * (w4.weights - tw2), 2) / sqrt(T - 1) <= te2
    @test_throws AssertionError portfolio.tracking_err = TrackWeight(; e = te2, w = 1:19)
    @test_throws AssertionError portfolio.tracking_err = TrackWeight(; e = te2, w = 1:21)

    portfolio.tracking_err = NoTracking()
    w5 = optimise2!(portfolio; obj = SR(; rf = rf))
    te3 = 0.007
    tw3 = vec(mean(portfolio.returns; dims = 2))
    portfolio.tracking_err = TrackRet(; e = te3, w = tw3)
    w6 = optimise2!(portfolio; obj = MinRisk())
    @test norm(portfolio.returns * w6.weights - tw3, 2) / sqrt(T - 1) <= te3

    portfolio.tracking_err = NoTracking()
    w7 = optimise2!(portfolio; obj = MinRisk())
    te4 = 0.0024
    tw4 = vec(mean(portfolio.returns; dims = 2))
    portfolio.tracking_err = TrackRet(; e = te4, w = tw4)
    w8 = optimise2!(portfolio; obj = SR(; rf = rf))
    @test norm(portfolio.returns * w8.weights - tw4, 2) / sqrt(T - 1) <= te4

    @test_throws AssertionError portfolio.tracking_err = TrackRet(; e = te2, w = 1:(T - 1))
    @test_throws AssertionError portfolio.tracking_err = TrackRet(; e = te2, w = 1:(T + 1))
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

# # @testset "Network and Dendrogram" begin
# portfolio = Portfolio2(; prices = prices,
#                        solvers = Dict(:PClGL => Dict(:solver => optimizer_with_attributes(Pajarito.Optimizer,
#                                                                                           "verbose" => false,
#                                                                                           "oa_solver" => optimizer_with_attributes(GLPK.Optimizer,
#                                                                                                                                    MOI.Silent() => true),
#                                                                                           "conic_solver" => optimizer_with_attributes(Clarabel.Optimizer,
#                                                                                                                                       "verbose" => false,
#                                                                                                                                       "max_step_fraction" => 0.75)))))
# asset_statistics2!(portfolio)

# A = centrality_vector2(portfolio)
# B = connection_matrix2(portfolio)
# C = cluster_matrix2(portfolio)

# rm = SD2()
# obj = MinRisk()
# w1 = optimise2!(portfolio; obj = obj, rm = rm)
# wt = [0.00791850924098073, 0.030672065216453506, 0.010501402809199123, 0.027475241969187995,
#       0.012272329269527841, 0.03339587076426262, 1.4321532289072258e-6, 0.13984297866711365,
#       2.4081081597353397e-6, 5.114425959766348e-5, 0.2878111114337346,
#       1.5306036912879562e-6, 1.1917690994187655e-6, 0.12525446872321966,
#       6.630910273840812e-6, 0.015078706008184504, 8.254970801614574e-5, 0.1930993918762575,
#       3.0369340845779798e-6, 0.11652799957572683]
# @test isapprox(w1.weights, wt)

# portfolio.a_vec_cent = A
# portfolio.b_cent = minimum(A)
# w2 = optimise2!(portfolio; obj = obj, rm = rm)
# wt = [3.5426430661259544e-11, 0.07546661314221123, 0.021373033743425803,
#       0.027539611826011074, 0.023741467255115698, 0.12266515815974924, 2.517181275812244e-6,
#       0.22629893782442134, 3.37118211246211e-10, 0.02416530860824232,
#       3.3950118687352606e-10, 1.742541411959769e-6, 1.5253730188343444e-6,
#       5.304686980295978e-11, 0.024731789616991084, 3.3711966852218057e-10,
#       7.767147353183488e-11, 0.30008056166009706, 1.171415437554966e-10, 0.1539317317710031]
# @test isapprox(w2.weights, wt)

# portfolio.network_method = IP2()
# w3 = optimise!(portfolio, opt)
# # end
