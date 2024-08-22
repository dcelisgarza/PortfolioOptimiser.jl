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
    @test isapprox(w3.weights, wc3.weights)

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

    w7 = optimise2!(portfolio; obj = obj, rm = rm)
    wt = [5.2456519308516375e-9, 1.3711772920494845e-8, 1.828307590553689e-8,
          1.1104551657521415e-8, 0.5180586238310904, 1.7475553033958716e-9,
          0.06365101880957791, 1.1504018896467816e-8, 1.0491794751235226e-8,
          5.650081752041375e-9, 9.262635444153261e-9, 1.4051602963640009e-9,
          9.923106257589635e-10, 3.050812431838031e-9, 9.966076078173243e-10,
          0.1432679499627637, 0.19649701265872604, 1.0778757180749977e-8,
          0.07852527921167819, 1.1301377011579277e-8]
    @test isapprox(w7.weights, wt)

    wc7 = optimise2!(portfolio; type = WC2(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w7.weights, wc7.weights, rtol = 5.0e-6)

    portfolio.a_vec_cent = A
    portfolio.b_cent = minimum(A)

    w8 = optimise2!(portfolio; obj = obj, rm = rm)
    wt = [3.580964533638559e-11, 9.277053125069126e-10, 9.835000020421294e-10,
          0.48362433627647977, 1.3062621693742353e-9, 3.4185103123124565e-11,
          0.28435553885288534, 0.17984723049407092, 1.693432131576565e-10,
          2.621850918060399e-10, 0.05217287791621345, 5.161470450551339e-9,
          2.4492072342885186e-9, 4.790090083985224e-11, 2.6687808589346464e-9,
          1.0539691276230515e-9, 2.0827050059613154e-10, 8.48619997995145e-10,
          2.329804826102029e-10, 7.01603778985698e-11]
    @test isapprox(w8.weights, wt)

    wc8 = optimise2!(portfolio; type = WC2(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w8.weights, wc8.weights, rtol = 5.0e-5)

    portfolio.network_method = IP2(; A = B)
    w9 = optimise2!(portfolio; obj = obj, rm = rm)
    wt = [1.721507261658482e-11, 4.611331427286785e-11, 4.6537999918192326e-11,
          5.4302470941152905e-11, 4.508258159401719e-11, 2.510366932792115e-11,
          0.9999999992889718, 5.3842283750049385e-11, 9.81183158048394e-12,
          3.589613596601703e-11, 5.3385756391682404e-11, 4.8553725558375136e-11,
          4.299842533772006e-11, 1.755101403265066e-11, 4.285647343183246e-11,
          4.6446497580083914e-11, 2.1013218648897535e-11, 4.559832123949408e-11,
          3.102165424744005e-11, 2.7697895657875245e-11]
    @test isapprox(w9.weights, wt)

    wc9 = optimise2!(portfolio; type = WC2(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w9.weights, wc9.weights)

    portfolio.network_method = SDP2(; A = B)
    w10 = optimise2!(portfolio; obj = obj, rm = rm)
    wt = [5.024048627217141e-14, 1.5886394555170722e-12, 1.5924178422101003e-12,
          0.24554600920016964, 1.6332399340098677e-12, 1.0616523261700116e-13,
          0.0959435974734496, 0.2562392110969168, 2.7829378542014683e-13,
          4.877264338339266e-13, 0.402271160351581, 1.2141431002683076e-8,
          4.523802241069811e-9, 5.3729255370564864e-14, 5.20282725457044e-9,
          1.5947986553082236e-12, 3.43551721221936e-13, 1.5824894417145518e-12,
          3.8307499177570936e-13, 1.280439184666322e-13]
    @test isapprox(w10.weights, wt)

    wc10 = optimise2!(portfolio; type = WC2(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w10.weights, wc10.weights, rtol = 5.0e-5)

    portfolio.network_method = IP2(; A = C)
    w11 = optimise2!(portfolio; obj = obj, rm = rm)
    wt = [3.5488808869733083e-12, 8.281743679661074e-12, 8.80902177574244e-12,
          9.955072746064901e-12, 1.1134822019504797e-11, 4.7417114086610015e-12,
          0.6074274195608048, 1.0253105313724553e-11, 1.4148918790428058e-12,
          6.3499711667509184e-12, 0.39257258031345454, 9.215307590460059e-12,
          7.61409793255415e-12, 3.5761100061677828e-12, 7.3543478626708e-12,
          9.27354997599413e-12, 3.8374099382272584e-12, 8.458519745181501e-12,
          6.62153510029776e-12, 5.3005368572386036e-12]
    @test isapprox(w11.weights, wt, rtol = 1.0e-7)

    wc11 = optimise2!(portfolio; type = WC2(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w11.weights, wc11.weights, rtol = 0.005)

    portfolio.network_method = SDP2(; A = C)
    w12 = optimise2!(portfolio; obj = obj, rm = rm)
    wt = [1.645653643573042e-14, 5.24765733463725e-13, 5.259882343242254e-13,
          0.38135427736460104, 5.383168184665497e-13, 3.606530943112124e-14,
          2.659687073118818e-8, 2.387877721283499e-8, 9.190837103177029e-14,
          1.6141883247690533e-13, 0.6186456659519168, 3.0002908666789655e-9,
          1.5771657468870285e-9, 1.747936659731388e-14, 1.6271322639469356e-9,
          5.267534803071819e-13, 1.1362061261344631e-13, 5.231419643587768e-13,
          1.2683767418078167e-13, 4.256536051656617e-14]
    @test isapprox(w12.weights, wt)

    wc12 = optimise2!(portfolio; type = WC2(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w12.weights, wc12.weights, rtol = 5.0e-6)

    portfolio.a_vec_cent = []
    portfolio.b_cent = Inf

    portfolio.network_method = IP2(; A = B)
    w13 = optimise2!(portfolio; obj = obj, rm = rm)
    wt = [2.0966673826999653e-12, 2.198260339765037e-12, 1.9669534265126797e-12,
          2.0537353944645557e-12, 0.762282392772933, 2.61589719142697e-12,
          1.6397732202694488e-12, 2.573524529021814e-12, 2.1635532147945916e-12,
          2.3920718749431715e-12, 2.6031535089914636e-12, 2.254849729224801e-12,
          2.3351897528966937e-12, 2.5746406212787903e-12, 2.4339077077727048e-12,
          0.23771760718605286, 2.077497433772476e-12, 2.544464994811904e-12,
          2.2242733110585934e-12, 2.2657939494042705e-12]
    @test isapprox(w13.weights, wt)

    wc13 = optimise2!(portfolio; type = WC2(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w13.weights, wc13.weights, rtol = 0.0005)

    portfolio.network_method = SDP2(; A = B)
    w14 = optimise2!(portfolio; obj = obj, rm = rm)
    wt = [1.131030668352105e-7, 9.736849167799003e-7, 1.5073134798603708e-7,
          9.724848430006577e-8, 0.3050368084415617, 4.717356198432155e-8,
          0.03491168150833796, 1.1696087757981777e-6, 2.1133994869894878e-7,
          1.38898043984553e-7, 0.23158972602737993, 2.930159759606465e-8,
          1.841227833023016e-8, 1.3415748037100702e-7, 2.038375787580353e-8,
          0.10856264505102015, 2.399490931217591e-7, 0.2072142228794291,
          2.522693174355702e-7, 0.11268131983060002]
    @test isapprox(w14.weights, wt)

    wc14 = optimise2!(portfolio; type = WC2(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w14.weights, wc14.weights, rtol = 5.0e-5)

    portfolio.network_method = IP2(; A = C)
    w15 = optimise2!(portfolio; obj = obj, rm = rm)
    wt = [1.1904763791512141e-12, 1.2340531692956696e-12, 1.067454109369536e-12,
          1.127653866268004e-12, 0.617653720874792, 1.6368429734597877e-12,
          9.929978443062197e-13, 1.5549069160715418e-12, 1.0890925109349555e-12,
          1.3843599449712156e-12, 1.5689472056216027e-12, 1.400584981008052e-12,
          1.5941278979112304e-12, 1.5487191642726196e-12, 1.6035009203180108e-12,
          0.1719148482846596, 1.1171304302527388e-12, 1.5058581547861255e-12,
          0.210431430817618, 1.3135957217697496e-12]
    @test isapprox(w15.weights, wt)

    wc15 = optimise2!(portfolio; type = WC2(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w15.weights, wc15.weights, rtol = 0.005)

    portfolio.network_method = SDP2(; A = C)
    w16 = optimise2!(portfolio; obj = obj, rm = rm)
    wt = [1.9469265893415583e-7, 1.9264487242540467e-7, 2.232335248226593e-7,
          1.1246896197634285e-7, 0.40746626496017013, 8.182096647686593e-8,
          5.006408130895852e-8, 2.231759312172926e-7, 3.1463420211396267e-7,
          7.356649686989611e-6, 0.3255255880379563, 4.241919388356267e-8,
          2.851355716950413e-8, 2.082232132989454e-7, 3.1366090964049265e-8,
          0.15590260393465316, 0.010877836307400067, 2.4616586233596695e-7,
          0.10021813719562453, 2.6349139195481583e-7]
    @test isapprox(w16.weights, wt, rtol = 5.0e-6)

    wc16 = optimise2!(portfolio; type = WC2(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w16.weights, wc16.weights, rtol = 0.0001)
end
