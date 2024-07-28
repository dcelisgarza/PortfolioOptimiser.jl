using CSV, Clarabel, DataFrames, Graphs, HiGHS, JuMP, LinearAlgebra, OrderedCollections,
      Pajarito, PortfolioOptimiser, Statistics, Test, TimeSeries, GLPK, Distances

import Distances: pairwise, UnionMetric

struct POCorDist <: Distances.UnionMetric end
function Distances.pairwise(::POCorDist, mtx, i)
    return sqrt.(clamp!((1 .- mtx) / 2, 0, 1))
end
dbht_d(corr, dist) = 2 .- (dist .^ 2) / 2

prices = TimeArray(CSV.File("./assets/stock_prices.csv"); timestamp = :date)

rf = 1.0329^(1 / 252) - 1
l = 2.0

solvers = Dict(:PClGL => Dict(:solver => optimizer_with_attributes(Pajarito.Optimizer,
                                                                   "verbose" => false,
                                                                   "oa_solver" => optimizer_with_attributes(GLPK.Optimizer,
                                                                                                            MOI.Silent() => true),
                                                                   "conic_solver" => optimizer_with_attributes(Clarabel.Optimizer,
                                                                                                               "verbose" => false,
                                                                                                               "max_step_fraction" => 0.75))))

@testset "Rebalancing constraints" begin
    portfolio = Portfolio(; prices = prices,
                          solvers = OrderedDict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                                  :params => Dict("verbose" => false,
                                                                                  "max_step_fraction" => 0.7))))
    asset_statistics!(portfolio; calc_kurt = false)

    portfolio.rebalance = Inf
    portfolio.rebalance_weights = []
    w1 = optimise!(portfolio, OptimiseOpt(; obj = :Min_Risk, rf = rf))
    r1 = calc_risk(portfolio)
    sr1 = sharpe_ratio(portfolio)
    w2 = optimise!(portfolio, OptimiseOpt(; obj = :Utility, rf = rf))
    r2 = calc_risk(portfolio)
    w3 = optimise!(portfolio, OptimiseOpt(; obj = :Sharpe, rf = rf))
    r3 = calc_risk(portfolio)
    sr3 = sharpe_ratio(portfolio)
    w4 = optimise!(portfolio, OptimiseOpt(; obj = :Max_Ret, rf = rf))
    r4 = calc_risk(portfolio)

    portfolio.rebalance = 0
    portfolio.rebalance_weights = w3.weights
    w5 = optimise!(portfolio, OptimiseOpt(; obj = :Min_Risk, rf = rf))
    portfolio.rebalance_weights = w1.weights
    w6 = optimise!(portfolio, OptimiseOpt(; obj = :Utility, rf = rf))
    w7 = optimise!(portfolio, OptimiseOpt(; obj = :Sharpe, rf = rf))
    w8 = optimise!(portfolio, OptimiseOpt(; obj = :Max_Ret, rf = rf))

    portfolio.rebalance = 1e10
    portfolio.rebalance_weights = w3.weights
    w9 = optimise!(portfolio, OptimiseOpt(; obj = :Min_Risk, rf = rf))
    portfolio.rebalance_weights = w1.weights
    w10 = optimise!(portfolio, OptimiseOpt(; obj = :Utility, rf = rf))
    w11 = optimise!(portfolio, OptimiseOpt(; obj = :Sharpe, rf = rf))
    w12 = optimise!(portfolio, OptimiseOpt(; obj = :Max_Ret, rf = rf))

    portfolio.rebalance_weights = []
    w13 = optimise!(portfolio, OptimiseOpt(; obj = :Min_Risk, rf = rf))

    portfolio.rebalance = 1e-4
    portfolio.rebalance_weights = w3.weights
    w14 = optimise!(portfolio, OptimiseOpt(; obj = :Min_Risk, rf = rf))
    r14 = calc_risk(portfolio)
    sr14 = sharpe_ratio(portfolio)

    portfolio.rebalance = 5e-4
    portfolio.rebalance_weights = w1.weights
    w15 = optimise!(portfolio, OptimiseOpt(; obj = :Utility, rf = rf))
    r15 = calc_risk(portfolio)
    w16 = optimise!(portfolio, OptimiseOpt(; obj = :Sharpe, rf = rf))
    r16 = calc_risk(portfolio)
    sr16 = sharpe_ratio(portfolio)
    w17 = optimise!(portfolio, OptimiseOpt(; obj = :Max_Ret, rf = rf))
    r17 = calc_risk(portfolio)

    @test isapprox(w1.weights, w5.weights)
    @test isapprox(w2.weights, w6.weights)
    @test isapprox(w3.weights, w7.weights)
    @test isapprox(w4.weights, w8.weights)
    @test isapprox(w3.weights, w9.weights)
    @test isapprox(w1.weights, w10.weights)
    @test isapprox(w1.weights, w11.weights)
    @test isapprox(w1.weights, w12.weights, rtol = 1.0e-7)
    @test isapprox(w1.weights, w13.weights)
    @test !isapprox(w1.weights, w14.weights)
    @test !isapprox(w3.weights, w14.weights)
    @test !isapprox(w1.weights, w15.weights)
    @test !isapprox(w2.weights, w15.weights)
    @test !isapprox(w1.weights, w16.weights)
    @test !isapprox(w3.weights, w16.weights)
    @test !isapprox(w1.weights, w17.weights)
    @test !isapprox(w4.weights, w17.weights)
    @test r1 < r14 < r3
    @test sr1 < sr14 < sr3
    @test r1 < r15 < r2
    @test r1 < r3 < r16
    @test sr1 < sr16 < sr3
    @test r1 < r17 < r4
end

@testset "Network and Dendrogram Constraints $(:CDaR)" begin
    portfolio = Portfolio(; prices = prices, solvers = solvers)
    asset_statistics!(portfolio; calc_kurt = false)

    rm = :CDaR
    obj = :Min_Risk
    opt = OptimiseOpt(; obj = obj, rm = rm, l = l, rf = rf)
    CV = centrality_vector(portfolio;
                           cor_opt = CorOpt(; dist = DistOpt(; method = POCorDist())),
                           network_opt = NetworkOpt(; method = :TMFG,
                                                    tmfg_genfunc = GenericFunction(;
                                                                                   func = dbht_d)))
    B_1 = connection_matrix(portfolio;
                            cor_opt = CorOpt(; dist = DistOpt(; method = POCorDist())),
                            network_opt = NetworkOpt(; method = :TMFG,
                                                     tmfg_genfunc = GenericFunction(;
                                                                                    func = dbht_d)))
    L_A = cluster_matrix(portfolio;
                         cor_opt = CorOpt(; dist = DistOpt(; method = POCorDist())),
                         cluster_opt = ClusterOpt(; linkage = :DBHT,
                                                  genfunc = GenericFunction(;
                                                                            func = dbht_d)))

    w1 = optimise!(portfolio, opt)

    portfolio.a_vec_cent = CV
    portfolio.b_cent = minimum(CV)
    w2 = optimise!(portfolio, opt)

    portfolio.network_method = :IP
    portfolio.network_ip = B_1
    w3 = optimise!(portfolio, opt)

    portfolio.network_method = :SDP
    portfolio.network_sdp = B_1
    w4 = optimise!(portfolio, opt)

    portfolio.network_method = :IP
    portfolio.network_ip = L_A
    w5 = optimise!(portfolio, opt)

    portfolio.network_method = :SDP
    portfolio.network_sdp = L_A
    w6 = optimise!(portfolio, opt)

    portfolio.a_vec_cent = []
    portfolio.b_cent = Inf

    portfolio.network_method = :IP
    portfolio.network_ip = B_1
    w7 = optimise!(portfolio, opt)

    portfolio.network_method = :SDP
    portfolio.network_sdp = B_1
    w8 = optimise!(portfolio, opt)

    portfolio.network_method = :IP
    portfolio.network_ip = L_A
    w9 = optimise!(portfolio, opt)

    portfolio.network_method = :SDP
    portfolio.network_sdp = L_A
    w10 = optimise!(portfolio, opt)

    portfolio = Portfolio(; prices = prices, solvers = solvers)
    asset_statistics!(portfolio; calc_kurt = false)

    rm = :CDaR
    obj = :Utility
    opt = OptimiseOpt(; obj = obj, rm = rm, l = l, rf = rf)
    CV = centrality_vector(portfolio;
                           cor_opt = CorOpt(; dist = DistOpt(; method = POCorDist())),
                           network_opt = NetworkOpt(; method = :TMFG,
                                                    tmfg_genfunc = GenericFunction(;
                                                                                   func = dbht_d)))
    B_1 = connection_matrix(portfolio;
                            cor_opt = CorOpt(; dist = DistOpt(; method = POCorDist())),
                            network_opt = NetworkOpt(; method = :TMFG,
                                                     tmfg_genfunc = GenericFunction(;
                                                                                    func = dbht_d)))
    L_A = cluster_matrix(portfolio;
                         cor_opt = CorOpt(; dist = DistOpt(; method = POCorDist())),
                         cluster_opt = ClusterOpt(; linkage = :DBHT,
                                                  genfunc = GenericFunction(;
                                                                            func = dbht_d)))

    w11 = optimise!(portfolio, opt)

    portfolio.a_vec_cent = CV
    portfolio.b_cent = minimum(CV)
    w12 = optimise!(portfolio, opt)

    portfolio.network_method = :IP
    portfolio.network_ip = B_1
    w13 = optimise!(portfolio, opt)

    portfolio.network_method = :SDP
    portfolio.network_sdp = B_1
    w14 = optimise!(portfolio, opt)

    portfolio.network_method = :IP
    portfolio.network_ip = L_A
    w15 = optimise!(portfolio, opt)

    portfolio.network_method = :SDP
    portfolio.network_sdp = L_A
    w16 = optimise!(portfolio, opt)

    portfolio.a_vec_cent = []
    portfolio.b_cent = Inf

    portfolio.network_method = :IP
    portfolio.network_ip = B_1
    w17 = optimise!(portfolio, opt)

    portfolio.network_method = :SDP
    portfolio.network_sdp = B_1
    w18 = optimise!(portfolio, opt)

    portfolio.network_method = :IP
    portfolio.network_ip = L_A
    w19 = optimise!(portfolio, opt)

    portfolio.network_method = :SDP
    portfolio.network_sdp = L_A
    w20 = optimise!(portfolio, opt)

    portfolio = Portfolio(; prices = prices, solvers = solvers)
    asset_statistics!(portfolio; calc_kurt = false)

    rm = :CDaR
    obj = :Sharpe
    opt = OptimiseOpt(; obj = obj, rm = rm, l = l, rf = rf)
    CV = centrality_vector(portfolio;
                           cor_opt = CorOpt(; dist = DistOpt(; method = POCorDist())),
                           network_opt = NetworkOpt(; method = :TMFG,
                                                    tmfg_genfunc = GenericFunction(;
                                                                                   func = dbht_d)))
    B_1 = connection_matrix(portfolio;
                            cor_opt = CorOpt(; dist = DistOpt(; method = POCorDist())),
                            network_opt = NetworkOpt(; method = :TMFG,
                                                     tmfg_genfunc = GenericFunction(;
                                                                                    func = dbht_d)))
    L_A = cluster_matrix(portfolio;
                         cor_opt = CorOpt(; dist = DistOpt(; method = POCorDist())),
                         cluster_opt = ClusterOpt(; linkage = :DBHT,
                                                  genfunc = GenericFunction(;
                                                                            func = dbht_d)))

    w21 = optimise!(portfolio, opt)

    portfolio.a_vec_cent = CV
    portfolio.b_cent = minimum(CV)
    w22 = optimise!(portfolio, opt)

    portfolio.network_method = :IP
    portfolio.network_ip = B_1
    w23 = optimise!(portfolio, opt)

    portfolio.network_method = :SDP
    portfolio.network_sdp = B_1
    w24 = optimise!(portfolio, opt)

    portfolio.network_method = :IP
    portfolio.network_ip = L_A
    w25 = optimise!(portfolio, opt)

    portfolio.network_method = :SDP
    portfolio.network_sdp = L_A
    w26 = optimise!(portfolio, opt)

    portfolio.a_vec_cent = []
    portfolio.b_cent = Inf

    portfolio.network_method = :IP
    portfolio.network_ip = B_1
    w27 = optimise!(portfolio, opt)

    portfolio.network_method = :SDP
    portfolio.network_sdp = B_1
    w28 = optimise!(portfolio, opt)

    portfolio.network_method = :IP
    portfolio.network_ip = L_A
    w29 = optimise!(portfolio, opt)

    portfolio.network_method = :SDP
    portfolio.network_sdp = L_A
    w30 = optimise!(portfolio, opt)

    portfolio = Portfolio(; prices = prices, solvers = solvers)
    asset_statistics!(portfolio; calc_kurt = false)

    rm = :CDaR
    obj = :Max_Ret
    opt = OptimiseOpt(; obj = obj, rm = rm, l = l, rf = rf)
    CV = centrality_vector(portfolio;
                           cor_opt = CorOpt(; dist = DistOpt(; method = POCorDist())),
                           network_opt = NetworkOpt(; method = :TMFG,
                                                    tmfg_genfunc = GenericFunction(;
                                                                                   func = dbht_d)))
    B_1 = connection_matrix(portfolio;
                            cor_opt = CorOpt(; dist = DistOpt(; method = POCorDist())),
                            network_opt = NetworkOpt(; method = :TMFG,
                                                     tmfg_genfunc = GenericFunction(;
                                                                                    func = dbht_d)))
    L_A = cluster_matrix(portfolio;
                         cor_opt = CorOpt(; dist = DistOpt(; method = POCorDist())),
                         cluster_opt = ClusterOpt(; linkage = :DBHT,
                                                  genfunc = GenericFunction(;
                                                                            func = dbht_d)))

    w31 = optimise!(portfolio, opt)

    portfolio.a_vec_cent = CV
    portfolio.b_cent = minimum(CV)
    w32 = optimise!(portfolio, opt)

    portfolio.network_method = :IP
    portfolio.network_ip = B_1
    w33 = optimise!(portfolio, opt)

    portfolio.network_method = :SDP
    portfolio.network_sdp = B_1
    w34 = optimise!(portfolio, opt)

    portfolio.network_method = :IP
    portfolio.network_ip = L_A
    w35 = optimise!(portfolio, opt)

    portfolio.network_method = :SDP
    portfolio.network_sdp = L_A
    w36 = optimise!(portfolio, opt)

    portfolio.a_vec_cent = []
    portfolio.b_cent = Inf

    portfolio.network_method = :IP
    portfolio.network_ip = B_1
    w37 = optimise!(portfolio, opt)

    portfolio.network_method = :SDP
    portfolio.network_sdp = B_1
    w38 = optimise!(portfolio, opt)

    portfolio.network_method = :IP
    portfolio.network_ip = L_A
    w39 = optimise!(portfolio, opt)

    portfolio.network_method = :SDP
    portfolio.network_sdp = L_A
    w40 = optimise!(portfolio, opt)

    portfolio = Portfolio(; prices = prices, solvers = solvers)
    asset_statistics!(portfolio; calc_kurt = false)
    portfolio.network_penalty = 0.5

    rm = :CDaR
    obj = :Min_Risk
    opt = OptimiseOpt(; obj = obj, rm = rm, l = l, rf = rf)
    CV = centrality_vector(portfolio;
                           cor_opt = CorOpt(; dist = DistOpt(; method = POCorDist())),
                           network_opt = NetworkOpt(; method = :TMFG,
                                                    tmfg_genfunc = GenericFunction(;
                                                                                   func = dbht_d)))
    B_1 = connection_matrix(portfolio;
                            cor_opt = CorOpt(; dist = DistOpt(; method = POCorDist())),
                            network_opt = NetworkOpt(; method = :TMFG,
                                                     tmfg_genfunc = GenericFunction(;
                                                                                    func = dbht_d)))
    L_A = cluster_matrix(portfolio;
                         cor_opt = CorOpt(; dist = DistOpt(; method = POCorDist())),
                         cluster_opt = ClusterOpt(; linkage = :DBHT,
                                                  genfunc = GenericFunction(;
                                                                            func = dbht_d)))

    w41 = optimise!(portfolio, opt)

    portfolio.a_vec_cent = CV
    portfolio.b_cent = minimum(CV)
    w42 = optimise!(portfolio, opt)

    portfolio.network_method = :IP
    portfolio.network_ip = B_1
    w43 = optimise!(portfolio, opt)

    portfolio.network_method = :SDP
    portfolio.network_sdp = B_1
    w44 = optimise!(portfolio, opt)

    portfolio.network_method = :IP
    portfolio.network_ip = L_A
    w45 = optimise!(portfolio, opt)

    portfolio.network_method = :SDP
    portfolio.network_sdp = L_A
    w46 = optimise!(portfolio, opt)

    portfolio.a_vec_cent = []
    portfolio.b_cent = Inf

    portfolio.network_method = :IP
    portfolio.network_ip = B_1
    w47 = optimise!(portfolio, opt)

    portfolio.network_method = :SDP
    portfolio.network_sdp = B_1
    w48 = optimise!(portfolio, opt)

    portfolio.network_method = :IP
    portfolio.network_ip = L_A
    w49 = optimise!(portfolio, opt)

    portfolio.network_method = :SDP
    portfolio.network_sdp = L_A
    w50 = optimise!(portfolio, opt)

    portfolio = Portfolio(; prices = prices, solvers = solvers)
    asset_statistics!(portfolio; calc_kurt = false)
    portfolio.network_penalty = 0.5

    rm = :CDaR
    obj = :Utility
    opt = OptimiseOpt(; obj = obj, rm = rm, l = l, rf = rf)
    CV = centrality_vector(portfolio;
                           cor_opt = CorOpt(; dist = DistOpt(; method = POCorDist())),
                           network_opt = NetworkOpt(; method = :TMFG,
                                                    tmfg_genfunc = GenericFunction(;
                                                                                   func = dbht_d)))
    B_1 = connection_matrix(portfolio;
                            cor_opt = CorOpt(; dist = DistOpt(; method = POCorDist())),
                            network_opt = NetworkOpt(; method = :TMFG,
                                                     tmfg_genfunc = GenericFunction(;
                                                                                    func = dbht_d)))
    L_A = cluster_matrix(portfolio;
                         cor_opt = CorOpt(; dist = DistOpt(; method = POCorDist())),
                         cluster_opt = ClusterOpt(; linkage = :DBHT,
                                                  genfunc = GenericFunction(;
                                                                            func = dbht_d)))

    w51 = optimise!(portfolio, opt)

    portfolio.a_vec_cent = CV
    portfolio.b_cent = minimum(CV)
    w52 = optimise!(portfolio, opt)

    portfolio.network_method = :IP
    portfolio.network_ip = B_1
    w53 = optimise!(portfolio, opt)

    portfolio.network_method = :SDP
    portfolio.network_sdp = B_1
    w54 = optimise!(portfolio, opt)

    portfolio.network_method = :IP
    portfolio.network_ip = L_A
    w55 = optimise!(portfolio, opt)

    portfolio.network_method = :SDP
    portfolio.network_sdp = L_A
    w56 = optimise!(portfolio, opt)

    portfolio.a_vec_cent = []
    portfolio.b_cent = Inf

    portfolio.network_method = :IP
    portfolio.network_ip = B_1
    w57 = optimise!(portfolio, opt)

    portfolio.network_method = :SDP
    portfolio.network_sdp = B_1
    w58 = optimise!(portfolio, opt)

    portfolio.network_method = :IP
    portfolio.network_ip = L_A
    w59 = optimise!(portfolio, opt)

    portfolio.network_method = :SDP
    portfolio.network_sdp = L_A
    w60 = optimise!(portfolio, opt)

    portfolio = Portfolio(; prices = prices, solvers = solvers)
    asset_statistics!(portfolio; calc_kurt = false)
    portfolio.network_penalty = 0.5

    rm = :CDaR
    obj = :Sharpe
    opt = OptimiseOpt(; obj = obj, rm = rm, l = l, rf = rf)
    CV = centrality_vector(portfolio;
                           cor_opt = CorOpt(; dist = DistOpt(; method = POCorDist())),
                           network_opt = NetworkOpt(; method = :TMFG,
                                                    tmfg_genfunc = GenericFunction(;
                                                                                   func = dbht_d)))
    B_1 = connection_matrix(portfolio;
                            cor_opt = CorOpt(; dist = DistOpt(; method = POCorDist())),
                            network_opt = NetworkOpt(; method = :TMFG,
                                                     tmfg_genfunc = GenericFunction(;
                                                                                    func = dbht_d)))
    L_A = cluster_matrix(portfolio;
                         cor_opt = CorOpt(; dist = DistOpt(; method = POCorDist())),
                         cluster_opt = ClusterOpt(; linkage = :DBHT,
                                                  genfunc = GenericFunction(;
                                                                            func = dbht_d)))

    w61 = optimise!(portfolio, opt)

    portfolio.a_vec_cent = CV
    portfolio.b_cent = minimum(CV)
    w62 = optimise!(portfolio, opt)

    portfolio.network_method = :IP
    portfolio.network_ip = B_1
    w63 = optimise!(portfolio, opt)

    portfolio.network_method = :SDP
    portfolio.network_sdp = B_1
    w64 = optimise!(portfolio, opt)

    portfolio.network_method = :IP
    portfolio.network_ip = L_A
    w65 = optimise!(portfolio, opt)

    portfolio.network_method = :SDP
    portfolio.network_sdp = L_A
    w66 = optimise!(portfolio, opt)

    portfolio.a_vec_cent = []
    portfolio.b_cent = Inf

    portfolio.network_method = :IP
    portfolio.network_ip = B_1
    w67 = optimise!(portfolio, opt)

    portfolio.network_method = :SDP
    portfolio.network_sdp = B_1
    w68 = optimise!(portfolio, opt)

    portfolio.network_method = :IP
    portfolio.network_ip = L_A
    w69 = optimise!(portfolio, opt)

    portfolio.network_method = :SDP
    portfolio.network_sdp = L_A
    w70 = optimise!(portfolio, opt)

    portfolio = Portfolio(; prices = prices, solvers = solvers)
    asset_statistics!(portfolio; calc_kurt = false)
    portfolio.network_penalty = 0.5

    rm = :CDaR
    obj = :Max_Ret
    opt = OptimiseOpt(; obj = obj, rm = rm, l = l, rf = rf)
    CV = centrality_vector(portfolio;
                           cor_opt = CorOpt(; dist = DistOpt(; method = POCorDist())),
                           network_opt = NetworkOpt(; method = :TMFG,
                                                    tmfg_genfunc = GenericFunction(;
                                                                                   func = dbht_d)))
    B_1 = connection_matrix(portfolio;
                            cor_opt = CorOpt(; dist = DistOpt(; method = POCorDist())),
                            network_opt = NetworkOpt(; method = :TMFG,
                                                     tmfg_genfunc = GenericFunction(;
                                                                                    func = dbht_d)))
    L_A = cluster_matrix(portfolio;
                         cor_opt = CorOpt(; dist = DistOpt(; method = POCorDist())),
                         cluster_opt = ClusterOpt(; linkage = :DBHT,
                                                  genfunc = GenericFunction(;
                                                                            func = dbht_d)))

    w71 = optimise!(portfolio, opt)

    portfolio.a_vec_cent = CV
    portfolio.b_cent = minimum(CV)
    w72 = optimise!(portfolio, opt)

    portfolio.network_method = :IP
    portfolio.network_ip = B_1
    w73 = optimise!(portfolio, opt)

    portfolio.network_method = :SDP
    portfolio.network_sdp = B_1
    w74 = optimise!(portfolio, opt)

    portfolio.network_method = :IP
    portfolio.network_ip = L_A
    w75 = optimise!(portfolio, opt)

    portfolio.network_method = :SDP
    portfolio.network_sdp = L_A
    w76 = optimise!(portfolio, opt)

    portfolio.a_vec_cent = []
    portfolio.b_cent = Inf

    portfolio.network_method = :IP
    portfolio.network_ip = B_1
    w77 = optimise!(portfolio, opt)

    portfolio.network_method = :SDP
    portfolio.network_sdp = B_1
    w78 = optimise!(portfolio, opt)

    portfolio.network_method = :IP
    portfolio.network_ip = L_A
    w79 = optimise!(portfolio, opt)

    portfolio.network_method = :SDP
    portfolio.network_sdp = L_A
    w80 = optimise!(portfolio, opt)

    w1t = [0.0, 0.0, 0.0, 0.0, 0.0034099011531648123, 0.0, 0.0, 0.0790428239168528, 0.0,
           0.0, 0.38759317020233125, 0.0, 0.0, 0.0, 0.000554515274145546,
           0.0959882893057346, 0.26790888256716067, 0.0, 0.0006560171912823603,
           0.16484640038932794]
    w2t = [0.0, 0.0, 0.0, 0.0517231388794993, 1.0853285495591566e-17, 0.0,
           0.05548699473292642, 0.2566673894884358, 0.0, 0.0, 0.5073286323951223,
           0.12879384450401618, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    w3t = [0.0, 0.0, 0.0, 0.1380859788647512, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
           0.8619140211352487, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    w4t = [6.916534741199183e-11, 2.8618337810291045e-11, 1.7205254095654967e-11,
           0.07061456710040678, 3.0536835743353447e-9, 6.760242032946457e-11,
           0.0749901517770178, 0.26806573786220506, 7.209653720639389e-11,
           6.263094071793627e-11, 0.4165467202009079, 0.16978281921565505,
           6.213183961911815e-11, 6.969913668329987e-11, 2.3053457824367047e-12,
           8.392549703807868e-11, 7.289297163434109e-11, 4.24728884907784e-11,
           7.335236005568172e-11, 6.602485142184087e-11]
    w5t = [0.0, 0.0, 0.0, 2.2660061040532335e-16, 3.1871886633789018e-18, 0.0,
           5.414233441434456e-17, 0.23899933639488408, 0.0, 0.0, 0.7610006636051158, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    w6t = [2.2437454319616068e-11, 7.367925928057877e-12, 1.0309757583889708e-12,
           0.05175588225206795, 9.626705291019906e-10, 2.1894616561097622e-11,
           0.052317886159437714, 0.2808759207143959, 2.3403280740204143e-11,
           2.0276367859252104e-11, 0.47619910136345883, 0.1388512082902021,
           8.155818589736122e-12, 2.26522600759685e-11, 3.1262348556340355e-12,
           4.579867802449056e-11, 2.3656252089037902e-11, 1.2767151006645863e-11,
           2.380829218608891e-11, 2.1391808011831222e-11]
    w7t = [0.0, 0.0, 5.920765310267057e-17, 0.0, 0.37297150326295364, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.6270284967370462, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    w8t = [5.333278572147394e-11, 0.05638785424214076, 1.8014070508602034e-10,
           5.2482313928648487e-11, 0.0820640582931353, 1.7756938142535277e-11,
           1.9026995337541293e-12, 0.13507762926843872, 1.339080959908328e-10,
           3.787326943211817e-11, 0.3175440779328707, 0.020298213899402687,
           4.163925315365446e-11, 1.174229289652941e-10, 4.784818255603212e-11,
           0.12477101930299711, 0.07961825077575638, 0.03009048638710793,
           8.237676509605876e-11, 0.15414840913146652]
    w9t = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4887092443113236, 0.0, 0.0,
           0.0, 0.0, 0.06771233053987874, 0.4435784251487976, 0.0, 0.0, 0.0]
    w10t = [8.875593213438965e-11, 6.786109213687967e-11, 1.4037575625933866e-10,
            3.8199967324374995e-11, 0.02640519119812921, 2.7117944596220678e-11,
            4.288392782851217e-12, 0.08423696478561357, 1.4264688819150728e-10,
            2.372852183733598e-10, 0.38656445140741746, 0.009574217971642852,
            4.8447681491384566e-11, 1.8846595318466704e-10, 1.2713734073799868e-11,
            0.14421172180595543, 0.2209166214560235, 2.7179524314502213e-9,
            0.0006754849428707257, 0.1274153427182363]
    w11t = [0.0, 0.0, 0.0, 0.0, 0.003559479269687892, 0.0, 0.0, 0.0799730685504724, 0.0,
            0.0, 0.3871598884133303, 0.0, 0.0, 0.0, 0.00022182912532979019,
            0.09652576896214482, 0.2659397480673419, 0.0, 0.0019205861925266662,
            0.16469963141916624]
    w12t = [0.0, 0.0, 0.0, 0.0523709534650843, 6.511971297354939e-17, 0.0,
            0.05695993250300767, 0.2532224974039182, 0.0, 0.0, 0.5082815392792128,
            0.1291650773487769, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    w13t = [0.0, 0.0, 0.0, 0.13808597886475102, 2.7755575615628914e-17, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.861914021135249, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    w14t = [1.044662547922087e-10, 5.6539479401170696e-11, 4.725050326199467e-11,
            0.0572708519970354, 4.30945100568106e-9, 1.027956998640308e-10,
            0.06334909256506384, 0.26639825454780713, 1.0786076329442599e-10,
            9.693009821232081e-11, 0.4668330300322677, 0.14614876501469856,
            4.16267588363442e-11, 1.0517003141350241e-10, 2.5736057408558917e-11,
            4.5042184740120697e-10, 1.0880272028698694e-10, 7.597190961290156e-11,
            1.0930840075846332e-10, 1.0079585815632804e-10]
    w15t = [0.0, 0.0, 0.0, 0.0, 2.0555053175707343e-17, 0.0, 0.0, 0.23899933639488455, 0.0,
            0.0, 0.7610006636051155, 2.9926097858580835e-17, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0]
    w16t = [5.4783144267735684e-11, 2.7164764801617607e-11, 1.9185821948629592e-11,
            0.05213986634242332, 2.2346788271117297e-9, 5.387532878644971e-11,
            0.0542933556956247, 0.2681570751783776, 5.664822722409223e-11,
            5.070405628780779e-11, 0.4893894154774715, 0.1360202842774923,
            6.378801350105508e-12, 5.5211067013043996e-11, 8.085507812824697e-12,
            2.557605661440101e-10, 5.713924037794297e-11, 3.870221265330855e-11,
            5.7431582637833805e-11, 5.286126045118138e-11]
    w17t = [0.0, 0.0, 0.0, 0.0, 0.37297150326295375, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.6270284967370463, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    w18t = [5.045783366213136e-11, 0.027087558541133032, 1.2490463860059793e-10,
            2.3179354430033212e-11, 0.031974200738443176, 1.2232533405103717e-11,
            1.7470181805933803e-11, 0.09327461173474633, 1.3933388322477516e-10,
            1.7771948756462316e-11, 0.36898570967277466, 0.007183433710346759,
            7.127037788297474e-12, 2.2503960566329528e-10, 1.883751349496794e-11,
            0.11991644921560822, 0.19463651907696827, 0.029194687029793633,
            1.4014767002437255e-10, 0.12774682950368368]
    w19t = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.5048215439088678e-16, 0.0, 0.0,
            0.48870924431132384, 0.0, 0.0, 0.0, 0.0, 0.06771233053987892,
            0.4435784251487972, 0.0, 0.0, 5.154710590957425e-18]
    w20t = [8.253055625624691e-11, 2.265333612074869e-10, 1.4325320510272744e-10,
            5.124561734755736e-11, 0.008648188171162315, 1.896728582187503e-11,
            2.0738498832222256e-11, 0.07790581528688661, 1.5254232150729298e-10,
            8.794660253558486e-11, 0.3900258922237101, 4.524557755050906e-10,
            1.0235386014486964e-11, 3.2160411713832515e-10, 2.678664206462518e-11,
            0.12122208843330948, 0.2493037722429426, 0.004925686431219488,
            0.0013470285435718318, 0.14662152707235823]
    w21t = [0.0, 0.0, 0.07233520665470372, 0.0, 0.3107248736916702, 0.0, 0.0,
            0.12861270774687708, 0.0, 0.0, 0.1643840889865787, 0.0, 0.0, 0.0, 0.0,
            0.26288266377673325, 0.0, 0.0, 0.0, 0.06106045914343715]
    w22t = [0.0, 0.0, 0.0, 0.13626926674571752, 3.031954992025752e-17, 0.0,
            0.25290093270820974, 0.0, 0.0, 0.0, 0.48504514790687475, 0.125784652639198, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    w23t = [0.0, 0.0, 0.0, 0.0, 2.1783103199597982e-16, 0.0, 0.9999999999999991, 0.0, 0.0,
            0.0, 3.974611617249547e-16, 2.0601893316598876e-16, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0]
    w24t = [3.9876685656966745e-10, 9.058825654099517e-11, 1.5225238539063207e-11,
            0.15463223546864613, 1.7879156639428104e-8, 3.898300817015904e-10,
            0.2677212654869171, 0.05396055328845373, 4.2501034256593807e-10,
            3.398439384006062e-10, 0.3537579645962041, 0.16992795957209442,
            2.650732952741712e-11, 4.0584506626825864e-10, 2.2729667067761047e-10,
            3.4261005281370425e-12, 4.169691127080938e-10, 1.756827020590759e-10,
            4.2727561790302304e-10, 3.6626054963002114e-10]
    w25t = [0.0, 0.0, 0.0, 0.0, 6.313744038982638e-18, 0.0, 0.28939681985466575, 0.0, 0.0,
            0.0, 0.7106031801453343, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    w26t = [3.060462713388461e-10, 8.137840970909115e-11, 2.093581353623592e-11,
            0.14771709638997196, 1.3249869608796294e-8, 2.9827281123433514e-10,
            0.2921683866742857, 7.376479209607696e-9, 3.164602530501976e-10,
            2.618408136944159e-10, 0.4227708041708252, 0.13734368926895515,
            4.138125182364551e-11, 3.121597526506129e-10, 1.865018938350479e-10,
            1.3013875215717378e-11, 2.9600577500177714e-10, 1.4426192369895814e-10,
            3.089561412862967e-10, 2.823980670384955e-10]
    w27t = [0.0, 0.0, 0.0, 0.0, 0.7390777009270597, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.26092229907294023, 0.0, 0.0, 0.0, 0.0]
    w28t = [1.4680691018542715e-9, 4.859275738822192e-9, 0.015992802521425674,
            7.293948583023383e-10, 0.2988850375190448, 6.520933919634565e-11,
            2.128743388395745e-9, 0.16470653869543705, 4.624274795849735e-10,
            4.162708987445042e-10, 0.13583336339414312, 1.525142217002498e-11,
            2.494845225126199e-10, 1.0043814255652154e-9, 9.599421959136733e-11,
            0.2560211483665152, 4.277938501034575e-9, 1.880279259853607e-9,
            2.4445455761302985e-9, 0.12856108940616845]
    w29t = [0.0, 0.0, 0.0, 0.0, 0.37085335018478044, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.3290504062456894, 0.0, 0.0, 0.0, 0.0, 0.3000962435695302, 0.0, 0.0, 0.0, 0.0]
    w30t = [8.477676296638692e-10, 6.203619332148419e-10, 0.05312672189873553,
            3.525167655282879e-10, 0.33388855514882815, 2.4119211866760604e-11,
            1.2535040237468848e-9, 0.10330170622398843, 2.685246230866603e-10,
            3.0122990800851947e-10, 0.22159366006722425, 3.151455292776752e-11,
            1.1953995044141562e-10, 5.880004722184381e-10, 9.09904014906927e-11,
            0.2880893428920373, 1.7533763684708367e-9, 6.973164143794418e-10,
            2.7663224833929877e-9, 4.054101687985871e-9]
    w31t = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0]
    w32t = [0.0, 0.0, 0.0, 1.3528869751691398e-15, 0.0, 0.0, 0.9999999999999978, 0.0, 0.0,
            0.0, 9.889489766620252e-16, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    w33t = [0.0, 0.0, 0.0, 2.777132124193826e-15, 1.564886636111088e-17, 0.0,
            0.9999999999999947, 2.3505735879776856e-15, 0.0, 0.0, 2.0482117593296807e-16,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    w34t = [1.0281467676909825e-11, 1.187843668428613e-10, 1.4771882459712553e-10,
            0.15139448120550697, 3.1975461709679893e-10, 1.1703545334501572e-12,
            0.1608860077652859, 0.145521229458573, 2.5416288345688954e-11,
            2.5700717469472062e-11, 0.14506228245658642, 0.13806623634649423,
            0.12899222872350294, 1.2102153630587951e-11, 0.13007753307129755,
            1.3295486471969757e-10, 2.949644553944116e-11, 1.1477386849283491e-10,
            3.2350676124759535e-11, 2.248620229115514e-12]
    w35t = [0.0, 0.0, 0.0, 2.5346276720679334e-16, 1.1102230246251565e-16, 0.0,
            0.9999999999999939, 9.238751117774938e-16, 0.0, 0.0, 4.833286823928312e-15, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    w36t = [1.2705502938212128e-11, 1.2775333062061562e-10, 1.4817637107411689e-10,
            8.870770528393561e-8, 3.621922402976174e-10, 4.657960714644175e-12,
            0.3492479486074557, 1.0257908295131268e-8, 2.7671354140161673e-11,
            2.3825598933165473e-11, 0.3333954896557578, 2.4301757336849687e-8,
            0.3173564236164806, 1.507761022723158e-11, 1.3772464547471515e-8,
            1.7632643839853688e-10, 3.153068029100323e-11, 1.1630557696918374e-10,
            3.422455413401805e-11, 2.3077035956923223e-14]
    w37t = [0.0, 0.0, 0.0, 0.0, 1.932161143874555e-16, 0.0, 0.9999999999999998, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    w38t = [3.621329113468944e-10, 4.642067108379984e-9, 0.12994305384371968,
            0.1291627004290325, 1.859917952899618e-8, 5.267372540852563e-10,
            0.13864454544108298, 4.671761912470198e-8, 3.379932422513672e-10,
            2.4518139966590085e-9, 0.12281782523192106, 0.1158371561091733,
            7.993254387075122e-10, 6.521437796957827e-10, 0.10782932291536423,
            0.13173382352421667, 1.7328529810946067e-10, 0.12403148247918894,
            1.8439793595295232e-10, 1.4579605040215111e-8]
    w39t = [0.0, 0.0, 0.0, 0.0, 9.778825117965047e-17, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    w40t = [8.821695294777969e-9, 9.390891568028275e-9, 1.1453997375961597e-8,
            1.0555860544879462e-8, 1.551358747656118e-5, 7.608809353542062e-9,
            0.3388982404644372, 6.556130573611959e-9, 2.9056436537261766e-7,
            1.4782457963268473e-8, 2.2844429484614942e-8, 8.364527552747015e-9,
            4.348375629290353e-9, 1.1867153799932726e-8, 5.370396415973305e-9,
            0.3320042360584475, 1.1844714953983606e-8, 6.94882467729285e-9,
            0.3290815702080288, 8.358979467165728e-9]
    w41t = [0.0, 0.0, 0.0, 0.0, 0.0034099011531648123, 0.0, 0.0, 0.0790428239168528, 0.0,
            0.0, 0.38759317020233125, 0.0, 0.0, 0.0, 0.000554515274145546,
            0.0959882893057346, 0.26790888256716067, 0.0, 0.0006560171912823603,
            0.16484640038932794]
    w42t = [0.0, 0.0, 0.0, 0.0517231388794993, 1.0853285495591566e-17, 0.0,
            0.05548699473292642, 0.2566673894884358, 0.0, 0.0, 0.5073286323951223,
            0.12879384450401618, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    w43t = [0.0, 0.0, 0.0, 0.1380859788647512, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.8619140211352487, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    w44t = [5.149778311551971e-11, 2.4798751785860862e-12, 6.996081193565089e-10,
            0.14014631390762305, 1.5247863930734077e-9, 4.932213714657232e-11,
            0.13499505966078165, 0.21013438149593003, 5.420271549080447e-11,
            4.56334091518674e-11, 0.26706479999587784, 0.2453420578817712,
            0.002317382061699604, 5.175236011552128e-11, 2.3018232653008603e-9,
            5.3186426124386946e-11, 5.455682214418857e-11, 4.905971318108076e-12,
            5.49542402978741e-11, 4.76071398325532e-11]
    w45t = [0.0, 0.0, 0.0, 2.2660061040532335e-16, 3.1871886633789018e-18, 0.0,
            5.414233441434456e-17, 0.23899933639488408, 0.0, 0.0, 0.7610006636051158, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    w46t = [1.2394684853119162e-10, 2.263459219698208e-11, 6.742962250256897e-11,
            0.056507144893220625, 3.5114265245884664e-10, 1.1319451572354543e-10,
            0.07292885204694775, 0.28340360409306736, 1.336000911507412e-10,
            1.0159817693410076e-10, 0.38778598616527254, 0.1178488626720078,
            0.08152554357747022, 1.2542423220323185e-10, 4.921374942744655e-11,
            5.060121574593095e-9, 1.3535885468592743e-10, 2.242299103745142e-11,
            1.3651345460114417e-10, 1.0941217898247998e-10]
    w47t = [0.0, 0.0, 5.920765310267057e-17, 0.0, 0.37297150326295364, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.6270284967370462, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    w48t = [6.544970352321704e-11, 0.10178810405224831, 0.004112920935146684,
            0.0031639790031650875, 0.1530529999471956, 2.051853078939224e-11,
            0.014700198697066904, 0.1902839333114586, 2.706174002994138e-11,
            0.006575121290324204, 0.21040590074211363, 0.004140118886384232,
            2.3663886591314264e-11, 2.8342248346527903e-11, 0.02751075137205449,
            0.11821237566813832, 1.1286146419967076e-11, 1.5812546032508732e-10,
            6.128688173726229e-12, 0.16605359575412765]
    w49t = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4887092443113236, 0.0, 0.0,
            0.0, 0.0, 0.06771233053987874, 0.4435784251487976, 0.0, 0.0, 0.0]
    w50t = [1.5137302348582236e-10, 7.516757623743089e-11, 1.5732751128070879e-10,
            4.4879258940551166e-11, 0.10762107786841804, 4.965221525304349e-11,
            1.7650334038433817e-11, 0.09015085333758943, 7.437982028586447e-11,
            0.08190502196737333, 0.3585641373041588, 8.45012547336593e-11,
            8.055713680586459e-10, 6.266080826369522e-11, 2.4844808192707154e-11,
            0.20228189160080448, 0.013928469201256573, 1.3523703526424697e-10,
            1.7275538163833918e-10, 0.14554854686439914]
    w51t = [0.0, 0.0, 0.0, 0.0, 0.003559479269687892, 0.0, 0.0, 0.0799730685504724, 0.0,
            0.0, 0.3871598884133303, 0.0, 0.0, 0.0, 0.00022182912532979019,
            0.09652576896214482, 0.2659397480673419, 0.0, 0.0019205861925266662,
            0.16469963141916624]
    w52t = [0.0, 0.0, 0.0, 0.0523709534650843, 6.511971297354939e-17, 0.0,
            0.05695993250300767, 0.2532224974039182, 0.0, 0.0, 0.5082815392792128,
            0.1291650773487769, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    w53t = [0.0, 0.0, 0.0, 0.13808597886475102, 2.7755575615628914e-17, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.861914021135249, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    w54t = [4.427547682445635e-11, 1.2323388377239161e-11, 3.3789734127576494e-11,
            0.131348383651554, 1.890595631440567e-9, 4.3083631171441386e-11,
            0.11672047631962672, 0.22786079185199815, 4.582338939575489e-11,
            4.085822971817704e-11, 0.28975686608149864, 0.2343134784209093,
            8.471714805912585e-10, 4.4393685317616355e-11, 4.776188865720527e-10,
            4.3443093722436156e-11, 4.616716202230953e-11, 1.6316282801049807e-11,
            4.6368902972223686e-11, 4.218402495241013e-11]
    w55t = [0.0, 0.0, 0.0, 0.0, 2.0555053175707343e-17, 0.0, 0.0, 0.23899933639488455, 0.0,
            0.0, 0.7610006636051155, 2.9926097858580835e-17, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0]
    w56t = [1.7710672780195416e-10, 1.588513675393295e-11, 7.48740327757445e-11,
            0.06479406534073119, 7.659434179473154e-10, 1.6943407458889466e-10,
            0.0703313747755134, 0.2946514189340479, 1.8560588763769303e-10,
            1.5927415800127664e-10, 0.3915338882392439, 0.16681475643262836,
            0.01187448700487319, 1.7856927597881676e-10, 5.572416269029598e-11,
            6.946264208884656e-9, 1.8741852240203464e-10, 2.6430490814768294e-12,
            1.8825779770736008e-10, 1.6596156666547942e-10]
    w57t = [0.0, 0.0, 0.0, 0.0, 0.37297150326295375, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.6270284967370463, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    w58t = [1.0675624983732587e-10, 0.10287620722699994, 0.007419930667590516,
            0.005575733023148208, 0.1403841692167293, 3.199121980349859e-11,
            4.6306802673266263e-10, 0.18278277233059223, 8.631146826284374e-11,
            4.630387025867777e-10, 0.23500367383413973, 0.007030151281493114,
            5.022902604510499e-11, 6.927670538297251e-11, 0.006391407528962509,
            0.12391311923336361, 4.114531115043418e-11, 0.01987312469120852,
            1.9281611083696664e-11, 0.168749709634674]
    w59t = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.5048215439088678e-16, 0.0, 0.0,
            0.48870924431132384, 0.0, 0.0, 0.0, 0.0, 0.06771233053987892,
            0.4435784251487972, 0.0, 0.0, 5.154710590957425e-18]
    w60t = [2.788422003574095e-10, 1.6207940040589565e-10, 3.176807280585392e-10,
            1.021170891828994e-10, 0.08201299787452024, 9.44563568899901e-11,
            2.939598790408913e-11, 0.07466701851075265, 1.606771680114918e-10,
            0.05156485374361989, 0.3841363850280125, 1.981784636232484e-10,
            5.922874403607585e-10, 1.4968348989132215e-10, 5.083795147161331e-11,
            0.18843210751566997, 0.0920240940602742, 4.010448666029233e-10,
            4.0440096852321036e-10, 0.1271625403254685]
    w61t = [0.0, 0.0, 0.07233520665470372, 0.0, 0.3107248736916702, 0.0, 0.0,
            0.12861270774687708, 0.0, 0.0, 0.1643840889865787, 0.0, 0.0, 0.0, 0.0,
            0.26288266377673325, 0.0, 0.0, 0.0, 0.06106045914343715]
    w62t = [0.0, 0.0, 0.0, 0.13626926674571752, 3.031954992025752e-17, 0.0,
            0.25290093270820974, 0.0, 0.0, 0.0, 0.48504514790687475, 0.125784652639198, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    w63t = [0.0, 0.0, 0.0, 0.0, 2.1783103199597982e-16, 0.0, 0.9999999999999991, 0.0, 0.0,
            0.0, 3.974611617249547e-16, 2.0601893316598876e-16, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0]
    w64t = [6.998082927042676e-10, 1.5424340303099734e-10, 3.143052708446116e-11,
            0.19802517820583498, 2.9154150617875177e-8, 6.835660500782289e-10,
            0.3528275164814403, 0.10872075946065324, 6.950007745275945e-10,
            6.061445866384829e-10, 0.18177164937436036, 0.15865486058401645,
            1.209391620092009e-10, 7.071154007623483e-10, 4.4477427881280717e-10,
            2.7661165383299815e-10, 6.353206628572158e-10, 3.486548879707198e-10,
            6.847397491821611e-10, 6.511943842367757e-10]
    w65t = [0.0, 0.0, 0.0, 0.0, 6.313744038982638e-18, 0.0, 0.28939681985466575, 0.0, 0.0,
            0.0, 0.7106031801453343, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    w66t = [1.2566993443586423e-9, 3.739004426867521e-10, 7.218734047107632e-10,
            2.8711361498849682e-8, 4.7464518217059254e-8, 1.2427794466288645e-9,
            0.6479054824063863, 3.015466260821723e-9, 1.3090889757290767e-9,
            1.0544879652450842e-9, 0.35209438820418454, 3.230992063103231e-8,
            6.290664753103031e-11, 1.3012694178751276e-9, 7.562546119123085e-10,
            6.002565736656838e-9, 1.251419407261467e-9, 1.2486482949865245e-10,
            1.3315318986275207e-9, 1.0985203449597362e-9]
    w67t = [0.0, 0.0, 0.0, 0.0, 0.7390777009270597, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.26092229907294023, 0.0, 0.0, 0.0, 0.0]
    w68t = [2.7588373786033397e-10, 0.06230489766807689, 8.845864664436178e-10,
            5.350493552040951e-10, 0.2617450599704982, 7.669972307999461e-12,
            0.06747640663988178, 0.1314516270147797, 2.7995440420307e-10,
            2.334051199517679e-10, 0.11816977777706833, 7.419102269372113e-11,
            1.0248689533597765e-10, 7.199470300436598e-10, 2.1211052329000083e-11,
            0.21351023865750965, 2.963036099128278e-10, 1.2911351324417137e-9,
            3.2069817026981996e-10, 0.14534198722966346]
    w69t = [0.0, 0.0, 0.0, 0.0, 0.37085335018478044, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.3290504062456894, 0.0, 0.0, 0.0, 0.0, 0.3000962435695302, 0.0, 0.0, 0.0, 0.0]
    w70t = [2.83528025811891e-10, 1.9889363251220856e-10, 6.896952788121541e-10,
            2.0393982050817282e-10, 0.42840939255621546, 5.521528707519844e-11,
            0.020618414768207645, 3.0570366279259903e-10, 6.848434859603265e-10,
            2.5749926919453344e-10, 0.17530289231029145, 2.3978342380909192e-11,
            1.176120201518638e-10, 3.7881468006641544e-10, 1.0774042279343509e-10,
            0.3083078080171925, 3.4513551919100005e-10, 1.5361236524199915e-10,
            0.06736148825030497, 2.915763875007706e-10]
    w71t = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0]
    w72t = [0.0, 0.0, 0.0, 1.3528869751691398e-15, 0.0, 0.0, 0.9999999999999978, 0.0, 0.0,
            0.0, 9.889489766620252e-16, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    w73t = [0.0, 0.0, 0.0, 2.777132124193826e-15, 1.564886636111088e-17, 0.0,
            0.9999999999999947, 2.3505735879776856e-15, 0.0, 0.0, 2.0482117593296807e-16,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    w74t = [2.790097700306141e-11, 1.320285521917956e-10, 8.114088362470686e-10,
            0.1437099072732515, 3.0959084833408986e-10, 1.9823574024363207e-11,
            0.14466189300591933, 0.14312268150542642, 4.355719152823437e-11,
            1.5448017724435915e-11, 0.14307601420706664, 0.14237702719286968,
            0.14147060549885723, 3.0907863758647446e-11, 0.14158186945898285,
            2.3040636971923577e-10, 4.79573313389037e-11, 1.2271426282336827e-10,
            5.1031871387289024e-11, 1.4850891024797537e-11]
    w75t = [0.0, 0.0, 0.0, 2.5346276720679334e-16, 1.1102230246251565e-16, 0.0,
            0.9999999999999939, 9.238751117774938e-16, 0.0, 0.0, 4.833286823928312e-15, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    w76t = [2.8261968808837346e-11, 2.5313674193475505e-10, 2.691356518671941e-10,
            6.951238488601809e-8, 3.8825564022105e-10, 2.987651974774495e-12,
            0.3349164337831479, 3.509608637672499e-8, 7.318076963586879e-11,
            8.504661668156891e-11, 0.3333523027604628, 7.603126438747696e-8,
            0.3317310383283473, 3.623114179918471e-11, 4.1786761099433656e-8,
            1.1495113881521925e-9, 8.14169314433141e-11, 2.3609915960227954e-10,
            8.898658347321343e-11, 9.294935163160804e-12]
    w77t = [0.0, 0.0, 0.0, 0.0, 1.932161143874555e-16, 0.0, 0.9999999999999998, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    w78t = [3.310716861138238e-10, 1.3784145362205136e-8, 0.1254961986602289,
            0.12541329858112535, 6.168866828239069e-9, 3.5791503711104764e-10,
            0.12636375257965587, 3.1023664056976957e-7, 2.6354635887713034e-10,
            2.3289709408976344e-8, 0.12478242449722923, 0.12408556968228424,
            9.78830418898426e-9, 4.203854032463786e-10, 0.12328237793117948,
            0.12567295402982664, 1.631526610039635e-10, 0.12490305460146264,
            1.638252734757663e-10, 4.469444906125927e-9]
    w79t = [0.0, 0.0, 0.0, 0.0, 9.778825117965047e-17, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    w80t = [1.096538144845483e-8, 1.1710442485058386e-8, 1.4062410854845497e-8,
            1.3175553084222845e-8, 1.451345404572765e-5, 9.329995185086374e-9,
            0.3338765516751027, 8.324444955693995e-9, 3.8240618584207635e-7,
            1.792424380322421e-8, 2.4187729920932632e-8, 1.0148440911844875e-8,
            5.323940829425975e-9, 1.4500361899132632e-8, 6.783329792094791e-9,
            0.3332006936384892, 1.4659126849647355e-8, 8.72334605226523e-9,
            0.3329076786122852, 1.0395143371166035e-8]

    @test isapprox(w1.weights, w1t)
    @test isapprox(w2.weights, w2t)
    @test isapprox(w3.weights, w3t)
    @test isapprox(w4.weights, w4t)
    @test isapprox(w5.weights, w5t)
    @test isapprox(w6.weights, w6t)
    @test isapprox(w7.weights, w7t)
    @test isapprox(w8.weights, w8t)
    @test isapprox(w9.weights, w9t)
    @test isapprox(w10.weights, w10t)

    @test isapprox(w11.weights, w11t)
    @test isapprox(w12.weights, w12t)
    @test isapprox(w13.weights, w13t)
    @test isapprox(w14.weights, w14t)
    @test isapprox(w15.weights, w15t)
    @test isapprox(w16.weights, w16t)
    @test isapprox(w17.weights, w17t)
    @test isapprox(w18.weights, w18t)
    @test isapprox(w19.weights, w19t)
    @test isapprox(w20.weights, w20t)

    @test isapprox(w21.weights, w21t)
    @test isapprox(w22.weights, w22t)
    @test isapprox(w23.weights, w23t)
    @test isapprox(w24.weights, w24t)
    @test isapprox(w25.weights, w25t)
    @test isapprox(w26.weights, w26t)
    @test isapprox(w27.weights, w27t)
    @test isapprox(w28.weights, w28t)
    @test isapprox(w29.weights, w29t)
    @test isapprox(w30.weights, w30t)

    @test isapprox(w31.weights, w31t)
    @test isapprox(w32.weights, w32t)
    @test isapprox(w33.weights, w33t)
    @test isapprox(w34.weights, w34t)
    @test isapprox(w35.weights, w35t)
    @test isapprox(w36.weights, w36t)
    @test isapprox(w37.weights, w37t)
    @test isapprox(w38.weights, w38t)
    @test isapprox(w39.weights, w39t)
    @test isapprox(w40.weights, w40t)

    @test isapprox(w41.weights, w41t)
    @test isapprox(w42.weights, w42t)
    @test isapprox(w43.weights, w43t)
    @test isapprox(w44.weights, w44t)
    @test isapprox(w45.weights, w45t)
    @test isapprox(w46.weights, w46t, rtol = 1.0e-7)
    @test isapprox(w47.weights, w47t)
    @test isapprox(w48.weights, w48t)
    @test isapprox(w49.weights, w49t)
    @test isapprox(w50.weights, w50t)

    @test isapprox(w41.weights, w1.weights)
    @test isapprox(w42.weights, w2.weights)
    @test isapprox(w43.weights, w3.weights)
    @test !isapprox(w44.weights, w4.weights)
    @test isapprox(w45.weights, w5.weights)
    @test !isapprox(w46.weights, w6.weights)
    @test isapprox(w47.weights, w7.weights)
    @test !isapprox(w48.weights, w8.weights)
    @test isapprox(w49.weights, w9.weights)
    @test !isapprox(w50.weights, w10.weights)

    @test isapprox(w51.weights, w51t)
    @test isapprox(w52.weights, w52t)
    @test isapprox(w53.weights, w53t)
    @test isapprox(w54.weights, w54t)
    @test isapprox(w55.weights, w55t)
    @test isapprox(w56.weights, w56t)
    @test isapprox(w57.weights, w57t)
    @test isapprox(w58.weights, w58t)
    @test isapprox(w59.weights, w59t)
    @test isapprox(w60.weights, w60t, rtol = 1.0e-7)

    @test isapprox(w51.weights, w11.weights)
    @test isapprox(w52.weights, w12.weights)
    @test isapprox(w53.weights, w13.weights)
    @test !isapprox(w54.weights, w14.weights)
    @test isapprox(w55.weights, w15.weights)
    @test !isapprox(w56.weights, w16.weights)
    @test isapprox(w57.weights, w17.weights)
    @test !isapprox(w58.weights, w18.weights)
    @test isapprox(w59.weights, w19.weights)
    @test !isapprox(w60.weights, w20.weights)

    @test isapprox(w61.weights, w61t)
    @test isapprox(w62.weights, w62t)
    @test isapprox(w63.weights, w63t)
    @test isapprox(w64.weights, w64t)
    @test isapprox(w65.weights, w65t)
    @test isapprox(w66.weights, w66t)
    @test isapprox(w67.weights, w67t)
    @test isapprox(w68.weights, w68t)
    @test isapprox(w69.weights, w69t)
    @test isapprox(w70.weights, w70t)

    @test isapprox(w61.weights, w21.weights)
    @test isapprox(w62.weights, w22.weights)
    @test isapprox(w63.weights, w23.weights)
    @test !isapprox(w64.weights, w24.weights)
    @test isapprox(w65.weights, w25.weights)
    @test !isapprox(w66.weights, w26.weights)
    @test isapprox(w67.weights, w27.weights)
    @test !isapprox(w68.weights, w28.weights)
    @test isapprox(w69.weights, w29.weights)
    @test !isapprox(w70.weights, w30.weights)

    @test isapprox(w71.weights, w71t)
    @test isapprox(w72.weights, w72t)
    @test isapprox(w73.weights, w73t)
    @test isapprox(w74.weights, w74t)
    @test isapprox(w75.weights, w75t)
    @test isapprox(w76.weights, w76t)
    @test isapprox(w77.weights, w77t)
    @test isapprox(w78.weights, w78t)
    @test isapprox(w79.weights, w79t)
    @test isapprox(w80.weights, w80t)

    @test isapprox(w71.weights, w31.weights)
    @test isapprox(w72.weights, w32.weights)
    @test isapprox(w73.weights, w33.weights)
    @test !isapprox(w74.weights, w34.weights)
    @test isapprox(w75.weights, w35.weights)
    @test !isapprox(w76.weights, w36.weights)
    @test isapprox(w77.weights, w37.weights)
    @test !isapprox(w78.weights, w38.weights)
    @test isapprox(w79.weights, w39.weights)
    @test !isapprox(w80.weights, w40.weights)
end

@testset "Network and Dendrogram Constraints Short $(:CDaR)" begin
    portfolio = Portfolio(; prices = prices, solvers = solvers)
    asset_statistics!(portfolio; calc_kurt = false)
    portfolio.short = true
    portfolio.short_u = 0.13
    portfolio.long_u = 1
    ssl1 = portfolio.sum_short_long

    rm = :CDaR
    obj = :Min_Risk
    opt = OptimiseOpt(; obj = obj, rm = rm, l = l, rf = rf)
    CV = centrality_vector(portfolio;
                           cor_opt = CorOpt(; dist = DistOpt(; method = POCorDist())),
                           network_opt = NetworkOpt(; method = :MST,
                                                    tmfg_genfunc = GenericFunction(;
                                                                                   func = dbht_d)))
    B_1 = connection_matrix(portfolio;
                            cor_opt = CorOpt(; dist = DistOpt(; method = POCorDist())),
                            network_opt = NetworkOpt(; method = :MST,
                                                     tmfg_genfunc = GenericFunction(;
                                                                                    func = dbht_d)))
    L_A = cluster_matrix(portfolio;
                         cor_opt = CorOpt(; dist = DistOpt(; method = POCorDist())),
                         cluster_opt = ClusterOpt(; linkage = :ward,
                                                  genfunc = GenericFunction(;
                                                                            func = dbht_d)))

    w1 = optimise!(portfolio, opt)

    portfolio.a_vec_cent = CV
    portfolio.b_cent = minimum(CV)
    w2 = optimise!(portfolio, opt)

    portfolio.network_method = :IP
    portfolio.network_ip = B_1
    w3 = optimise!(portfolio, opt)

    portfolio.network_method = :SDP
    portfolio.network_sdp = B_1
    w4 = optimise!(portfolio, opt)

    portfolio.network_method = :IP
    portfolio.network_ip = L_A
    w5 = optimise!(portfolio, opt)

    portfolio.network_method = :SDP
    portfolio.network_sdp = L_A
    w6 = optimise!(portfolio, opt)

    portfolio.a_vec_cent = []
    portfolio.b_cent = Inf

    portfolio.network_method = :IP
    portfolio.network_ip = B_1
    w7 = optimise!(portfolio, opt)

    portfolio.network_method = :SDP
    portfolio.network_sdp = B_1
    w8 = optimise!(portfolio, opt)

    portfolio.network_method = :IP
    portfolio.network_ip = L_A
    w9 = optimise!(portfolio, opt)

    portfolio.network_method = :SDP
    portfolio.network_sdp = L_A
    w10 = optimise!(portfolio, opt)

    portfolio = Portfolio(; prices = prices, solvers = solvers)
    asset_statistics!(portfolio; calc_kurt = false)
    portfolio.short = true
    portfolio.short_u = 0.07
    portfolio.long_u = 0.57
    ssl2 = portfolio.sum_short_long

    rm = :CDaR
    obj = :Utility
    opt = OptimiseOpt(; obj = obj, rm = rm, l = l, rf = rf)
    CV = centrality_vector(portfolio;
                           cor_opt = CorOpt(; dist = DistOpt(; method = POCorDist())),
                           network_opt = NetworkOpt(; method = :TMFG,
                                                    tmfg_genfunc = GenericFunction(;
                                                                                   func = dbht_d)))
    B_1 = connection_matrix(portfolio;
                            cor_opt = CorOpt(; dist = DistOpt(; method = POCorDist())),
                            network_opt = NetworkOpt(; method = :TMFG,
                                                     tmfg_genfunc = GenericFunction(;
                                                                                    func = dbht_d)))
    L_A = cluster_matrix(portfolio;
                         cor_opt = CorOpt(; dist = DistOpt(; method = POCorDist())),
                         cluster_opt = ClusterOpt(; linkage = :DBHT,
                                                  genfunc = GenericFunction(;
                                                                            func = dbht_d)))

    w11 = optimise!(portfolio, opt)

    portfolio.a_vec_cent = CV
    portfolio.b_cent = minimum(CV)
    w12 = optimise!(portfolio, opt)

    portfolio.network_method = :IP
    portfolio.network_ip = B_1
    w13 = optimise!(portfolio, opt)

    portfolio.network_method = :SDP
    portfolio.network_sdp = B_1
    w14 = optimise!(portfolio, opt)

    portfolio.network_method = :IP
    portfolio.network_ip = L_A
    w15 = optimise!(portfolio, opt)

    portfolio.network_method = :SDP
    portfolio.network_sdp = L_A
    w16 = optimise!(portfolio, opt)

    portfolio.a_vec_cent = []
    portfolio.b_cent = Inf

    portfolio.network_method = :IP
    portfolio.network_ip = B_1
    w17 = optimise!(portfolio, opt)

    portfolio.network_method = :SDP
    portfolio.network_sdp = B_1
    w18 = optimise!(portfolio, opt)

    portfolio.network_method = :IP
    portfolio.network_ip = L_A
    w19 = optimise!(portfolio, opt)

    portfolio.network_method = :SDP
    portfolio.network_sdp = L_A
    w20 = optimise!(portfolio, opt)

    portfolio = Portfolio(; prices = prices, solvers = solvers)
    asset_statistics!(portfolio; calc_kurt = false)
    portfolio.short = true
    portfolio.short_u = 0.18
    portfolio.long_u = 0.95
    ssl3 = portfolio.sum_short_long

    rm = :CDaR
    obj = :Sharpe
    opt = OptimiseOpt(; obj = obj, rm = rm, l = l, rf = rf)
    CV = centrality_vector(portfolio;
                           cor_opt = CorOpt(; dist = DistOpt(; method = POCorDist())),
                           network_opt = NetworkOpt(; method = :TMFG,
                                                    tmfg_genfunc = GenericFunction(;
                                                                                   func = dbht_d)))
    B_1 = connection_matrix(portfolio;
                            cor_opt = CorOpt(; dist = DistOpt(; method = POCorDist())),
                            network_opt = NetworkOpt(; method = :TMFG,
                                                     tmfg_genfunc = GenericFunction(;
                                                                                    func = dbht_d)))
    L_A = cluster_matrix(portfolio;
                         cor_opt = CorOpt(; dist = DistOpt(; method = POCorDist())),
                         cluster_opt = ClusterOpt(; linkage = :DBHT,
                                                  genfunc = GenericFunction(;
                                                                            func = dbht_d)))

    w21 = optimise!(portfolio, opt)

    portfolio.a_vec_cent = CV
    portfolio.b_cent = minimum(CV)
    w22 = optimise!(portfolio, opt)

    portfolio.network_method = :IP
    portfolio.network_ip = B_1
    w23 = optimise!(portfolio, opt)

    portfolio.network_method = :SDP
    portfolio.network_sdp = B_1
    w24 = optimise!(portfolio, opt)

    portfolio.network_method = :IP
    portfolio.network_ip = L_A
    w25 = optimise!(portfolio, opt)

    portfolio.network_method = :SDP
    portfolio.network_sdp = L_A
    w26 = optimise!(portfolio, opt)

    portfolio.a_vec_cent = []
    portfolio.b_cent = Inf

    portfolio.network_method = :IP
    portfolio.network_ip = B_1
    w27 = optimise!(portfolio, opt)

    portfolio.network_method = :SDP
    portfolio.network_sdp = B_1
    w28 = optimise!(portfolio, opt)

    portfolio.network_method = :IP
    portfolio.network_ip = L_A
    w29 = optimise!(portfolio, opt)

    portfolio.network_method = :SDP
    portfolio.network_sdp = L_A
    w30 = optimise!(portfolio, opt)

    portfolio = Portfolio(; prices = prices, solvers = solvers)
    asset_statistics!(portfolio; calc_kurt = false)
    portfolio.short = true
    portfolio.short_u = 0.22
    portfolio.long_u = 0.83
    ssl4 = portfolio.sum_short_long

    rm = :CDaR
    obj = :Max_Ret
    opt = OptimiseOpt(; obj = obj, rm = rm, l = l, rf = rf)
    CV = centrality_vector(portfolio;
                           cor_opt = CorOpt(; dist = DistOpt(; method = POCorDist())),
                           network_opt = NetworkOpt(; method = :TMFG,
                                                    tmfg_genfunc = GenericFunction(;
                                                                                   func = dbht_d)))
    B_1 = connection_matrix(portfolio;
                            cor_opt = CorOpt(; dist = DistOpt(; method = POCorDist())),
                            network_opt = NetworkOpt(; method = :TMFG,
                                                     tmfg_genfunc = GenericFunction(;
                                                                                    func = dbht_d)))
    L_A = cluster_matrix(portfolio;
                         cor_opt = CorOpt(; dist = DistOpt(; method = POCorDist())),
                         cluster_opt = ClusterOpt(; linkage = :DBHT,
                                                  genfunc = GenericFunction(;
                                                                            func = dbht_d)))

    w31 = optimise!(portfolio, opt)

    portfolio.a_vec_cent = CV
    portfolio.b_cent = minimum(CV)
    w32 = optimise!(portfolio, opt)

    portfolio.network_method = :IP
    portfolio.network_ip = B_1
    w33 = optimise!(portfolio, opt)

    portfolio.network_method = :SDP
    portfolio.network_sdp = B_1
    w34 = optimise!(portfolio, opt)

    portfolio.network_method = :IP
    portfolio.network_ip = L_A
    w35 = optimise!(portfolio, opt)

    portfolio.network_method = :SDP
    portfolio.network_sdp = L_A
    w36 = optimise!(portfolio, opt)

    portfolio.a_vec_cent = []
    portfolio.b_cent = Inf

    portfolio.network_method = :IP
    portfolio.network_ip = B_1
    w37 = optimise!(portfolio, opt)

    portfolio.network_method = :SDP
    portfolio.network_sdp = B_1
    w38 = optimise!(portfolio, opt)

    portfolio.network_method = :IP
    portfolio.network_ip = L_A
    w39 = optimise!(portfolio, opt)

    portfolio.network_method = :SDP
    portfolio.network_sdp = L_A
    w40 = optimise!(portfolio, opt)

    portfolio = Portfolio(; prices = prices, solvers = solvers)
    asset_statistics!(portfolio; calc_kurt = false)
    portfolio.short = true
    portfolio.short_u = 0.13
    portfolio.long_u = 1
    ssl5 = portfolio.sum_short_long
    portfolio.network_penalty = 0.5

    rm = :CDaR
    obj = :Min_Risk
    opt = OptimiseOpt(; obj = obj, rm = rm, l = l, rf = rf)
    CV = centrality_vector(portfolio;
                           cor_opt = CorOpt(; dist = DistOpt(; method = POCorDist())),
                           network_opt = NetworkOpt(; method = :MST,
                                                    tmfg_genfunc = GenericFunction(;
                                                                                   func = dbht_d)))
    B_1 = connection_matrix(portfolio;
                            cor_opt = CorOpt(; dist = DistOpt(; method = POCorDist())),
                            network_opt = NetworkOpt(; method = :MST,
                                                     tmfg_genfunc = GenericFunction(;
                                                                                    func = dbht_d)))
    L_A = cluster_matrix(portfolio;
                         cor_opt = CorOpt(; dist = DistOpt(; method = POCorDist())),
                         cluster_opt = ClusterOpt(; linkage = :ward,
                                                  genfunc = GenericFunction(;
                                                                            func = dbht_d)))

    w41 = optimise!(portfolio, opt)

    portfolio.a_vec_cent = CV
    portfolio.b_cent = minimum(CV)
    w42 = optimise!(portfolio, opt)

    portfolio.network_method = :IP
    portfolio.network_ip = B_1
    w43 = optimise!(portfolio, opt)

    portfolio.network_method = :SDP
    portfolio.network_sdp = B_1
    w44 = optimise!(portfolio, opt)

    portfolio.network_method = :IP
    portfolio.network_ip = L_A
    w45 = optimise!(portfolio, opt)

    portfolio.network_method = :SDP
    portfolio.network_sdp = L_A
    w46 = optimise!(portfolio, opt)

    portfolio.a_vec_cent = []
    portfolio.b_cent = Inf

    portfolio.network_method = :IP
    portfolio.network_ip = B_1
    w47 = optimise!(portfolio, opt)

    portfolio.network_method = :SDP
    portfolio.network_sdp = B_1
    w48 = optimise!(portfolio, opt)

    portfolio.network_method = :IP
    portfolio.network_ip = L_A
    w49 = optimise!(portfolio, opt)

    portfolio.network_method = :SDP
    portfolio.network_sdp = L_A
    w50 = optimise!(portfolio, opt)

    portfolio = Portfolio(; prices = prices, solvers = solvers)
    asset_statistics!(portfolio; calc_kurt = false)
    portfolio.network_penalty = 0.5

    portfolio.short = true
    portfolio.short_u = 0.07
    portfolio.long_u = 0.57
    ssl6 = portfolio.sum_short_long

    rm = :CDaR
    obj = :Utility
    opt = OptimiseOpt(; obj = obj, rm = rm, l = l, rf = rf)
    CV = centrality_vector(portfolio;
                           cor_opt = CorOpt(; dist = DistOpt(; method = POCorDist())),
                           network_opt = NetworkOpt(; method = :TMFG,
                                                    tmfg_genfunc = GenericFunction(;
                                                                                   func = dbht_d)))
    B_1 = connection_matrix(portfolio;
                            cor_opt = CorOpt(; dist = DistOpt(; method = POCorDist())),
                            network_opt = NetworkOpt(; method = :TMFG,
                                                     tmfg_genfunc = GenericFunction(;
                                                                                    func = dbht_d)))
    L_A = cluster_matrix(portfolio;
                         cor_opt = CorOpt(; dist = DistOpt(; method = POCorDist())),
                         cluster_opt = ClusterOpt(; linkage = :DBHT,
                                                  genfunc = GenericFunction(;
                                                                            func = dbht_d)))

    w51 = optimise!(portfolio, opt)

    portfolio.a_vec_cent = CV
    portfolio.b_cent = minimum(CV)
    w52 = optimise!(portfolio, opt)

    portfolio.network_method = :IP
    portfolio.network_ip = B_1
    w53 = optimise!(portfolio, opt)

    portfolio.network_method = :SDP
    portfolio.network_sdp = B_1
    w54 = optimise!(portfolio, opt)

    portfolio.network_method = :IP
    portfolio.network_ip = L_A
    w55 = optimise!(portfolio, opt)

    portfolio.network_method = :SDP
    portfolio.network_sdp = L_A
    w56 = optimise!(portfolio, opt)

    portfolio.a_vec_cent = []
    portfolio.b_cent = Inf

    portfolio.network_method = :IP
    portfolio.network_ip = B_1
    w57 = optimise!(portfolio, opt)

    portfolio.network_method = :SDP
    portfolio.network_sdp = B_1
    w58 = optimise!(portfolio, opt)

    portfolio.network_method = :IP
    portfolio.network_ip = L_A
    w59 = optimise!(portfolio, opt)

    portfolio.network_method = :SDP
    portfolio.network_sdp = L_A
    w60 = optimise!(portfolio, opt)

    portfolio = Portfolio(; prices = prices, solvers = solvers)
    asset_statistics!(portfolio; calc_kurt = false)
    portfolio.network_penalty = 0.5

    portfolio.short = true
    portfolio.short_u = 0.18
    portfolio.long_u = 0.95
    ssl7 = portfolio.sum_short_long

    rm = :CDaR
    obj = :Sharpe
    opt = OptimiseOpt(; obj = obj, rm = rm, l = l, rf = rf)
    CV = centrality_vector(portfolio;
                           cor_opt = CorOpt(; dist = DistOpt(; method = POCorDist())),
                           network_opt = NetworkOpt(; method = :TMFG,
                                                    tmfg_genfunc = GenericFunction(;
                                                                                   func = dbht_d)))
    B_1 = connection_matrix(portfolio;
                            cor_opt = CorOpt(; dist = DistOpt(; method = POCorDist())),
                            network_opt = NetworkOpt(; method = :TMFG,
                                                     tmfg_genfunc = GenericFunction(;
                                                                                    func = dbht_d)))
    L_A = cluster_matrix(portfolio;
                         cor_opt = CorOpt(; dist = DistOpt(; method = POCorDist())),
                         cluster_opt = ClusterOpt(; linkage = :DBHT,
                                                  genfunc = GenericFunction(;
                                                                            func = dbht_d)))

    w61 = optimise!(portfolio, opt)

    portfolio.a_vec_cent = CV
    portfolio.b_cent = minimum(CV)
    w62 = optimise!(portfolio, opt)

    portfolio.network_method = :IP
    portfolio.network_ip = B_1
    w63 = optimise!(portfolio, opt)

    portfolio.network_method = :SDP
    portfolio.network_sdp = B_1
    w64 = optimise!(portfolio, opt)

    portfolio.network_method = :IP
    portfolio.network_ip = L_A
    w65 = optimise!(portfolio, opt)

    portfolio.network_method = :SDP
    portfolio.network_sdp = L_A
    w66 = optimise!(portfolio, opt)

    portfolio.a_vec_cent = []
    portfolio.b_cent = Inf

    portfolio.network_method = :IP
    portfolio.network_ip = B_1
    w67 = optimise!(portfolio, opt)

    portfolio.network_method = :SDP
    portfolio.network_sdp = B_1
    w68 = optimise!(portfolio, opt)

    portfolio.network_method = :IP
    portfolio.network_ip = L_A
    w69 = optimise!(portfolio, opt)

    portfolio.network_method = :SDP
    portfolio.network_sdp = L_A
    w70 = optimise!(portfolio, opt)

    portfolio = Portfolio(; prices = prices, solvers = solvers)
    asset_statistics!(portfolio; calc_kurt = false)
    portfolio.network_penalty = 0.5

    portfolio.short = true
    portfolio.short_u = 0.22
    portfolio.long_u = 0.83
    ssl8 = portfolio.sum_short_long

    rm = :CDaR
    obj = :Max_Ret
    opt = OptimiseOpt(; obj = obj, rm = rm, l = l, rf = rf)
    CV = centrality_vector(portfolio;
                           cor_opt = CorOpt(; dist = DistOpt(; method = POCorDist())),
                           network_opt = NetworkOpt(; method = :TMFG,
                                                    tmfg_genfunc = GenericFunction(;
                                                                                   func = dbht_d)))
    B_1 = connection_matrix(portfolio;
                            cor_opt = CorOpt(; dist = DistOpt(; method = POCorDist())),
                            network_opt = NetworkOpt(; method = :TMFG,
                                                     tmfg_genfunc = GenericFunction(;
                                                                                    func = dbht_d)))
    L_A = cluster_matrix(portfolio;
                         cor_opt = CorOpt(; dist = DistOpt(; method = POCorDist())),
                         cluster_opt = ClusterOpt(; linkage = :DBHT,
                                                  genfunc = GenericFunction(;
                                                                            func = dbht_d)))

    w71 = optimise!(portfolio, opt)

    portfolio.a_vec_cent = CV
    portfolio.b_cent = minimum(CV)
    w72 = optimise!(portfolio, opt)

    portfolio.network_method = :IP
    portfolio.network_ip = B_1
    w73 = optimise!(portfolio, opt)

    portfolio.network_method = :SDP
    portfolio.network_sdp = B_1
    w74 = optimise!(portfolio, opt)

    portfolio.network_method = :IP
    portfolio.network_ip = L_A
    w75 = optimise!(portfolio, opt)

    portfolio.network_method = :SDP
    portfolio.network_sdp = L_A
    w76 = optimise!(portfolio, opt)

    portfolio.a_vec_cent = []
    portfolio.b_cent = Inf

    portfolio.network_method = :IP
    portfolio.network_ip = B_1
    w77 = optimise!(portfolio, opt)

    portfolio.network_method = :SDP
    portfolio.network_sdp = B_1
    w78 = optimise!(portfolio, opt)

    portfolio.network_method = :IP
    portfolio.network_ip = L_A
    w79 = optimise!(portfolio, opt)

    portfolio.network_method = :SDP
    portfolio.network_sdp = L_A
    w80 = optimise!(portfolio, opt)

    w1t = [0.0, -0.0033927423088478525, 0.0, -0.0017047316992726633, 0.02620966174092891,
           -0.048951276707458066, -0.02831053066644215, 0.0797084661157905,
           -0.006091550656354475, -0.01772949018065388, 0.34963750962788587,
           -0.009810028561542944, 0.006616206609090757, -1.7025233012195975e-16,
           -0.01400964921942798, 0.09738313723086256, 0.16435744872368938,
           0.01643568935570537, 0.10356751838407428, 0.15608436221197253]
    w2t = [-0.09486167717442834, 0.04019416692914485, 0.037577223179041455,
           -5.0655785389293176e-17, 0.12801741187716315, -7.979727989493313e-17,
           -0.01998095976345068, 0.13355629824656456, 0.019271038397774838, 0.0,
           0.28797176886025655, 0.0, 0.008845283123909234, 0.0, -0.015157363062120899,
           0.10734222426525322, 0.0, 0.061474266941201526, 4.119968255444917e-17,
           0.17575031817969053]
    w3t = [-0.0, 0.0, 0.0, -0.0, 0.2542383528166068, -0.10786045982487963, -0.0,
           0.32907983060645274, -0.0, -2.0147579982279472e-17, 0.0, -0.0, 0.0, -0.0,
           0.029160101685153927, 0.13000000000000078, -0.0, 0.0, -0.0, 0.2353821747166649]
    w4t = [-0.01932528177598594, 0.04728373640057564, 0.08111180936616852,
           0.006702806870499849, 0.11062369340855788, 0.0022197632349840145,
           -0.025844621013543234, 0.08086336919259877, 0.027815773004550008,
           -0.006292839131009731, 0.27379360801846886, -0.0027801322519064387,
           0.009407464925989706, 0.005426314483965923, -0.01835147439447884,
           0.08781127608943395, -1.7739446729829628e-12, 0.06594215311122797,
           -0.035382906790888555, 0.17897548725256543]
    w5t = [-0.0, 0.0, 0.0, -0.0, 0.3345846888561426, -0.0, -0.0, 0.40541531114385687, -0.0,
           -0.0, 0.0, 0.0, -2.4870639382614985e-17, -0.0, 5.551115123125783e-17,
           0.13000000000000017, -0.0, 0.0, -0.0, -0.0]
    w6t = [-7.214654486461352e-11, 0.05224000652772797, 2.1084109420890238e-9,
           9.239927363752105e-11, 0.13600471870502973, 9.464164366878283e-12,
           -0.005257289864896337, 0.18836405182042718, 2.983819859555014e-11,
           3.7121011068347645e-11, 0.17386537310870234, 0.03260563547262639,
           0.006186175461892385, 5.762553937706617e-12, -5.549855776621395e-11,
           0.10277892136547706, -6.089456621935936e-12, 0.04618434314867764,
           -0.024440715712457622, 0.16146877781753147]
    w7t = [0.0, 0.0, 0.0, 0.0, 0.11199761883276776, -0.0, 1.9544851218808546e-17, -0.0, 0.0,
           0.0, 0.37508805422181013, -0.0, -0.0, -0.0, -0.0, 0.1412763749809966, 0.0,
           0.0958842646992405, -0.0, 0.14575368726518473]
    w8t = [7.952976789652419e-10, -0.0032720635053790576, 0.058653102262301825,
           -0.01041387756328327, 0.07503911419326144, -0.005393437034271601,
           -0.03820502798676035, 0.05334438344735125, -0.02198920577331149,
           -0.017281798058379805, 0.3051995774692396, -0.008898292571899221,
           0.007234081372116345, 0.01796539565164849, -0.024546284632572215,
           0.10516121679278256, 0.07890511508155898, 0.04604809897163522,
           0.09382272741895784, 0.1586271736697057]
    w9t = [-0.0, -0.0, 0.0, -0.0, -0.0, -0.0, -0.0, 0.0, -0.0, -0.0, 0.4251770425508514,
           -0.0, -0.0, -0.0, -0.0, 0.0589097275696944, 0.38591322987945387, 0.0, -0.0, 0.0]
    w10t = [4.521390531422383e-11, 1.7222479995698984e-11, 2.970473941417862e-10,
            1.2421731115968197e-11, 0.0274015058322884, 6.871539719718925e-12,
            -0.015740286257514956, 0.09047904287159925, 2.224127542215651e-11,
            6.369205629458061e-12, 0.3203382195808221, 0.0030473760125333746,
            0.0011262657610696697, 3.9424638392788905e-11, -0.0012491276129846269,
            0.10756160676442085, 0.2136041354049521, 4.058797194495132e-10,
            0.006019217695542393, 0.11741204309457973]
    w11t = [0.0, -0.003017422928335126, 0.0, -0.0005578189854038648, 0.014262826621635541,
            -0.026715129365893625, -0.016293232318117676, 0.04647310508003153,
            -0.0007856937176750969, -0.009357177603515744, 0.20100258315237746,
            -0.005501225617356324, 0.0035875951435311317, 4.817656010306793e-17,
            -0.0077722994637025485, 0.05659395572134987, 0.09433614865616102,
            0.008921359362670873, 0.05573433156787839, 0.08908809469436405]
    w12t = [0.0, -0.0021150569579432375, 2.9191758302543264e-17, 7.546047120499111e-17,
            0.02373249705660922, -0.031359033712414726, -0.01659550850242915,
            0.05200925317209252, -1.0262974121720264e-16, -0.0069393108741671525,
            0.20271106577475675, -0.00523818457046277, 0.00489100250981936,
            -4.004416292395704e-17, -0.007752905382582943, 0.05987427889337347,
            0.08208706271127576, 0.014526197775378596, 0.047831039424574534,
            0.08233760268211991]
    w13t = [0.0, -0.0, -0.0, 0.0, -2.3592239273284576e-16, -0.0, -0.0, -0.0, 0.0, -0.0,
            -0.0, -0.0, -0.0, 0.0, -0.0, -0.0, 0.0, -0.0, 0.0, 0.5000000000000002]
    w14t = [-9.645189965701146e-11, -0.0050399738515790335, 0.006240067141756696,
            -8.168595626982305e-10, 0.029922546594095524, -0.003831780764119934,
            -0.021099425556261774, 0.05300324681810716, -2.0512296574749853e-11,
            -0.023221187629459766, 0.18171475659600628, -0.008043660465761797,
            0.006854039888432041, 0.0005197421875196578, -0.008763961959371598,
            0.059893570181895374, 0.08872379651923099, 0.028529078164246643,
            0.036750231402480134, 0.07784891566660697]
    w15t = [-0.0, -0.0, 0.0, -0.0, 0.0, -0.0, -0.0, 0.0, -0.0, -0.0, 0.2560219841653259,
            -0.0, -8.149713241787042e-18, -0.0, -0.0, 0.08697526781400827,
            0.1570027480206658, 0.0, -0.0, 0.0]
    w16t = [2.8831025892719243e-11, -1.9595939475020135e-11, 1.2707150920122393e-10,
            -2.4340997938577834e-11, 0.02078233953214821, -0.011979674327820608,
            -0.01575163660481671, 0.053302205584505, -3.174982024667296e-12,
            -0.012376352878341641, 0.19464981229443953, -2.613129613085263e-10,
            0.004730622738422459, 6.765929338311763e-10, -0.007796473446100772,
            0.06178525447486814, 0.09405919607948578, 0.00826872962692379,
            0.036543088534655, 0.07378288786756096]
    w17t = [0.0, 0.0, 2.183222085500859e-17, -0.0, 0.181926392116249, -0.0, -0.0,
            -1.5869981902154903e-16, -0.0, -0.0, 0.32681877979942786, -0.0,
            -0.008745171915676905, 1.5869981902154903e-16, -0.0, 0.0, 0.0, -0.0, -0.0, 0.0]
    w18t = [-3.0873947281827094e-11, -0.010519572022264678, 0.0013399784326198429,
            0.0003691217989469454, 0.030433150524283224, -0.0009857124049867387,
            -0.022335271788733742, 0.05459788751045089, -2.722526787374905e-11,
            -0.021466509029366244, 0.1814328231131001, -0.007994824786606158,
            0.005852848474010549, 2.156674191179492e-9, -0.006698105330729311,
            0.05843815154434452, 0.08844809757370012, 0.02337687884589162,
            0.0460729868052739, 0.07963806864148995]
    w19t = [-0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, 0.0, -0.0, -0.0, 0.24435462215566234,
            -0.0, 1.1102230246251565e-16, 0.0, -0.0, 0.033856165269938565,
            0.22178921257439893, -0.0, -0.0, 0.0]
    w20t = [1.7526883929328617e-11, -1.1160191625521576e-10, 3.6807992630451534e-11,
            -5.5608904845136703e-11, 0.013625973707120929, -0.0154208661688382,
            -0.01470060989101075, 0.04990657111651833, -1.861354881695333e-11,
            -0.016221904950683222, 0.19553262179187778, -4.2908898950760014e-11,
            0.0036252074941067365, 3.4730541572141054e-10, -0.006673377140946396,
            0.05618945185118478, 0.10212298014248992, 0.004866684577518749,
            0.04517960584344184, 0.08196766145431224]
    w21t = [0.0, 0.0, 0.06455652837124166, 0.0, 0.17806945118716286, -0.1534166903019555,
            6.554560797119506e-17, 0.12351856122756986, 0.0, 0.0, 0.21867000759778857, 0.0,
            -0.023295149141940166, 0.0, -0.0032881605561044004, 0.1770017931069312,
            0.006143447950208067, 0.0, 0.04071674553919559, 0.1413234650199022]
    w22t = [0.0, 0.0, 0.07328869996428454, 0.0, 0.19593630860627667, -0.1516035031548817,
            -2.228175066036319e-17, 0.1290693464408622, 0.0, 0.0, 0.20897763141227396, 0.0,
            -0.02192543103531701, 0.0, -0.006471065809801198, 0.18010601289300754,
            7.366923752795404e-17, 0.0, 0.025951685743898865, 0.13667031493939605]
    w23t = [-0.0, -0.0, 0.0, -0.0, 0.6900000000000023, -0.0, -0.0, 0.0, -0.0, -0.0,
            0.19949253798767527, -0.0, -0.11949253798767445, -0.0, -0.0, 0.0, 0.0, -0.0,
            0.0, 0.0]
    w24t = [2.741356653806159e-10, 0.0327985135192352, 0.05482942296464975,
            1.6807444866887212e-9, 0.23098209876494905, -0.1272968878569344,
            1.2333109128769504e-10, 0.16767135103688993, 2.6900253889629564e-10,
            -1.81025212106666e-10, 0.11192694227345162, -0.0009046010074898403,
            -0.03158635862052409, 4.729287488649668e-10, -0.02021214736044242,
            0.21547631352003682, 2.330039172547154e-9, 1.6683155317454398e-9,
            0.016107281158490215, 0.12020806497021609]
    w25t = [-0.0, -0.0, 0.0, -0.0, 0.6899999999999881, -0.0, -0.0, 0.0, -0.0, -0.0,
            0.19949253798767191, -0.0, -0.11949253798766571, -0.0, -0.0, 0.0, -0.0, -0.0,
            0.0, 0.0]
    w26t = [4.367553933241219e-10, 1.7121297950096687e-9, 0.0791226899227516,
            3.0269068839537553e-10, 0.21739124744272845, -0.08376014653096171,
            -4.84842340329126e-10, 0.0715288510460991, 3.0607249175175775e-10,
            -3.0075276191920114e-11, 0.18251643617281296, -0.016971041079266443,
            -0.042283058661629604, 6.061838715163352e-10, -0.036985746017378626,
            0.31226363823511005, 8.316214223912442e-9, 6.995880831888516e-10,
            0.008871428458185506, 0.07830568914683192]
    w27t = [-0.0, -0.0, 0.0, -0.0, 0.6751155811095971, -0.0, -0.0, 0.0, -0.0, -0.0,
            0.208898224965582, -0.0, -0.11401380607517626, -0.0, -0.0, 0.0, 0.0, -0.0, -0.0,
            0.0]
    w28t = [3.123975164353941e-10, 0.03350488448786665, 0.0534748818006825,
            2.821085138067214e-9, 0.23689108605426548, -0.12555326614879248,
            9.934697729258627e-11, 0.1723411763832003, 3.262860547598395e-10,
            -1.3867719497983212e-10, 0.11452460513756973, -0.0018222047596648128,
            -0.0290903140132882, 5.836695085279428e-10, -0.023534209145378505,
            0.23091435018211504, 2.4461690447150134e-9, 2.1936759123102853e-9,
            2.718201580262619e-8, 0.10834897419545532]
    w29t = [-0.0, -0.0, 0.0, -0.0, 0.5385298461146968, -0.0, 0.0, 0.0, -0.0, -0.0, 0.0,
            -0.18, -0.0, 0.0, 0.0, 0.41147015388533015, -0.0, -0.0, -0.0, 0.0]
    w30t = [5.955950581033087e-10, 2.3983008245798362e-9, 0.05613753626575293,
            4.4297597474810196e-10, 0.24310704612051393, -0.08870686013747185,
            -7.333621475329736e-10, 0.10741465743450865, 3.7159128965983553e-10,
            -4.4370653646708964e-11, 0.1903247053265503, -0.005210219430165392,
            -0.0457147528808303, 8.55858618798722e-10, -0.04036815651430539,
            0.31161717203174516, 6.770405211463369e-9, 1.03400275498263e-9,
            1.0337735212242514e-8, 0.04139884975496985]
    w31t = [0.0, 0.0, 0.0, 0.0, 8.881784197001252e-16, 0.0, 0.8299999999999991, 0.0, 0.0,
            0.0, 0.0, 0.0, -0.22, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    w32t = [0.0, 0.0, 0.0, 0.0, 0.7959999999999998, 0.0, -1.2210381468724101e-16, 0.0, 0.0,
            0.0, 0.0, 0.0, -0.21999999999999986, 0.0, -1.515142386685086e-16,
            3.190790756317167e-16, -4.1901986586950697e-16, 0.0, 0.034000000000000336, 0.0]
    w33t = [-0.0, -0.0, 0.0, 1.722299551653608e-17, 0.3300000000000011, 0.2799999999999992,
            0.0, -1.1102230246251417e-17, 0.0, 0.0, -2.317705389555333e-16,
            -2.4719822535691137e-16, -0.0, 0.0, 1.397810916546131e-16, -0.0, 0.0, -0.0, 0.0,
            0.0]
    w34t = [1.8121498283334406e-9, 1.9538378670831258e-7, 0.09223164870599383,
            0.08411821812673427, 0.006540587245225673, 2.7379243920377502e-9,
            0.09910753522724745, 1.4987394920924547e-7, 0.13515200288733867,
            0.006193043766760279, 0.08329074250246365, 3.410014493453135e-8,
            1.1369560578888378e-9, 2.0210600890551512e-7, 0.004195094996182217,
            1.88274497115238e-7, 8.575988996631515e-10, 0.09236731587276072,
            2.7511520649373592e-9, 0.006803031635125004]
    w35t = [0.0, -0.0, -0.0, 0.0, 0.796, -0.0, 0.0, -0.0, -0.0, 0.0, -0.0, -0.0, -0.22,
            -0.0, -0.0, 0.0, 0.0, -0.0, 0.034, 0.0]
    w36t = [9.586460941954497e-10, 1.2043085831747494e-9, 1.6562835271854915e-9,
            1.5132558557620894e-9, 9.76151345475116e-7, 1.2778314005351548e-9,
            0.21116800645796965, 9.247329637416388e-10, 0.1380760068933674,
            1.9080612642175524e-9, 0.05728822279762091, 1.9334564108707354e-9,
            5.661181147009026e-10, 2.2187327511919097e-9, 9.158421160721371e-10,
            0.2034667527956946, 9.039756726783958e-10, 8.97333140457683e-10,
            1.705878478198533e-8, 9.6663922868371e-10]
    w37t = [0.0, 0.0, 0.0, 2.7755575615628914e-17, 1.6930901125533637e-15, 0.0,
            0.8299999999999983, 0.0, 0.0, 0.0, -1.1934897514720433e-15, 0.0,
            -0.21999999999999903, 0.0, 1.3877787807814457e-16, 4.218847493575595e-15, 0.0,
            0.0, 0.0, 0.0]
    w38t = [1.0443645301878027e-10, 5.247754628365762e-9, 0.08119210339693894,
            0.08041442447941809, 1.3500569046457257e-8, 1.3887941993224242e-10,
            0.08989356382126025, 1.560892126233196e-8, 1.0931887494596336e-10,
            5.193444098620796e-10, 0.0740689592711821, 0.06708506399842139,
            1.0567486907866775e-10, 1.9961769401352203e-10, 0.05907971109551509,
            0.08298370492423734, 4.327413003528916e-11, 0.07528242283018084,
            4.8413586818070956e-11, 1.0556641714105229e-8]
    w39t = [0.0, 2.7755575615628914e-17, 2.7755575615628914e-17, 2.7755575615628914e-17,
            0.0, 0.0, 0.830000000000001, 0.0, 4.163336342344337e-16, 0.0, 0.0, 0.0, -0.22,
            0.0, 0.0, -1.1379786002407855e-15, 2.7755575615628914e-17,
            2.7755575615628914e-17, 0.0, 2.7755575615628914e-17]
    w40t = [9.388191265148332e-10, 1.0075452389260085e-9, 1.4584866887566497e-9,
            1.134497958854917e-9, 2.4212632810185523e-6, 7.356060425589322e-10,
            0.20890910103572308, 6.946773726156755e-10, 3.193279846714939e-8,
            1.5916945723303897e-9, 6.635601956789222e-9, 8.183703394954517e-10,
            4.3533621873608734e-10, 1.1794037375579287e-9, 5.134275041810117e-10,
            0.20200499915360604, 1.288221441886696e-9, 7.307977650031418e-10,
            0.19908342655616054, 8.959469274964959e-10]
    w41t = [0.0, -0.0033927423088478525, 0.0, -0.0017047316992726633, 0.02620966174092891,
            -0.048951276707458066, -0.02831053066644215, 0.0797084661157905,
            -0.006091550656354475, -0.01772949018065388, 0.34963750962788587,
            -0.009810028561542944, 0.006616206609090757, -1.7025233012195975e-16,
            -0.01400964921942798, 0.09738313723086256, 0.16435744872368938,
            0.01643568935570537, 0.10356751838407428, 0.15608436221197253]
    w42t = [-0.09486167717442834, 0.04019416692914485, 0.037577223179041455,
            -5.0655785389293176e-17, 0.12801741187716315, -7.979727989493313e-17,
            -0.01998095976345068, 0.13355629824656456, 0.019271038397774838, 0.0,
            0.28797176886025655, 0.0, 0.008845283123909234, 0.0, -0.015157363062120899,
            0.10734222426525322, 0.0, 0.061474266941201526, 4.119968255444917e-17,
            0.17575031817969053]
    w43t = [-0.0, 0.0, 0.0, -0.0, 0.2542383528166068, -0.10786045982487963, -0.0,
            0.32907983060645274, -0.0, -2.0147579982279472e-17, 0.0, -0.0, 0.0, -0.0,
            0.029160101685153927, 0.13000000000000078, -0.0, 0.0, -0.0, 0.2353821747166649]
    w44t = [9.272379413696209e-12, 0.06109011163457147, 0.07388719114315925,
            0.019692527430815995, 0.1033181235897267, 0.08117018361137503,
            0.02338440269447632, 0.12195682880299138, 1.6870437788041508e-11,
            0.03457951915997033, 0.02795355513077693, 0.05841592130018157,
            7.482456877351314e-12, 0.029412363249929775, 0.0020709632768409862,
            0.04322171854316308, -8.422242390311495e-13, 0.08680666625948201,
            -3.58530874609073e-11, 0.10303992417560913]
    w45t = [-0.0, 0.0, 0.0, -0.0, 0.3345846888561426, -0.0, -0.0, 0.40541531114385687, -0.0,
            -0.0, 0.0, 0.0, -2.4870639382614985e-17, -0.0, 5.551115123125783e-17,
            0.13000000000000017, -0.0, 0.0, -0.0, -0.0]
    w46t = [1.6772123784853272e-11, 7.752239769909168e-10, 0.024355985034754033,
            8.227953173267867e-11, 0.18471176179842455, 1.4174654445697787e-10,
            0.022357207428156892, 0.2751582759067159, 3.065339706808394e-11,
            9.502208782908329e-11, 0.030652168623109988, 0.07645421112481837,
            0.0007236256680371044, 2.8188842001055798e-11, 0.03653126520853406,
            0.09934783116326011, 1.0928241691744877e-11, 0.005474421241443925,
            5.4282952234581425e-12, 0.11423324561650182]
    w47t = [0.0, 0.0, 0.0, 0.0, 0.11199761883276776, -0.0, 1.9544851218808546e-17, -0.0,
            0.0, 0.0, 0.37508805422181013, -0.0, -0.0, -0.0, -0.0, 0.1412763749809966, 0.0,
            0.0958842646992405, -0.0, 0.14575368726518473]
    w48t = [6.767065787779005e-11, 0.05845600850004039, 0.06857721044022666,
            0.018637080446546616, 0.09512946481878305, 0.07994346654886235,
            0.025545859831804, 0.07233232435258317, 5.714359090647327e-11,
            0.032112301662442985, 0.08806729718778292, 0.04699072744148142,
            2.302327613832333e-11, 0.04108314004036545, 0.0016832424014023952,
            0.057989539665363896, 6.102487921981212e-11, 0.08227280030983276,
            3.540297561534863e-11, 0.10117953610821653]
    w49t = [-0.0, -0.0, 0.0, -0.0, -0.0, -0.0, -0.0, 0.0, -0.0, -0.0, 0.4251770425508514,
            -0.0, -0.0, -0.0, -0.0, 0.0589097275696944, 0.38591322987945387, 0.0, -0.0, 0.0]
    w50t = [3.1366745962878835e-10, 6.421354185263541e-11, 0.015548662828215137,
            5.302349120647804e-11, 0.0718779249876566, 3.443486328538297e-11,
            0.014549230934361859, 0.10906579897532855, 2.8603746709819275e-11,
            2.3131893924215492e-11, 0.23095115947824363, 0.044332984975426426,
            1.7809258175941517e-10, 4.072425414215351e-11, 0.0058602511674883265,
            0.16234482039542594, 0.07541584072148488, 6.08611551099025e-11,
            5.436905753460773e-11, 0.14005332468524623]
    w51t = [0.0, -0.003017422928335126, 0.0, -0.0005578189854038648, 0.014262826621635541,
            -0.026715129365893625, -0.016293232318117676, 0.04647310508003153,
            -0.0007856937176750969, -0.009357177603515744, 0.20100258315237746,
            -0.005501225617356324, 0.0035875951435311317, 4.817656010306793e-17,
            -0.0077722994637025485, 0.05659395572134987, 0.09433614865616102,
            0.008921359362670873, 0.05573433156787839, 0.08908809469436405]
    w52t = [0.0, -0.0021150569579432375, 2.9191758302543264e-17, 7.546047120499111e-17,
            0.02373249705660922, -0.031359033712414726, -0.01659550850242915,
            0.05200925317209252, -1.0262974121720264e-16, -0.0069393108741671525,
            0.20271106577475675, -0.00523818457046277, 0.00489100250981936,
            -4.004416292395704e-17, -0.007752905382582943, 0.05987427889337347,
            0.08208706271127576, 0.014526197775378596, 0.047831039424574534,
            0.08233760268211991]
    w53t = [0.0, -0.0, -0.0, 0.0, -2.3592239273284576e-16, -0.0, -0.0, -0.0, 0.0, -0.0,
            -0.0, -0.0, -0.0, 0.0, -0.0, -0.0, 0.0, -0.0, 0.0, 0.5000000000000002]
    w54t = [0.010908485283292416, 0.004999163332856768, 6.274610272615418e-10,
            -2.877505728191141e-10, 0.019419368027987334, 3.363269195697643e-11,
            -0.008009455654976261, 0.06251307669008833, 0.008690491230390321,
            6.422270227832934e-11, 0.1576724342901207, 0.0059069274244321025,
            0.0016466554354329429, 7.896925981547204e-10, 0.003233313743018466,
            0.046494394372719246, 0.1275073736487822, 2.5468758863498447e-11,
            1.8588836351741396e-9, 0.05901776906424442]
    w55t = [-0.0, -0.0, 0.0, -0.0, 0.0, -0.0, -0.0, 0.0, -0.0, -0.0, 0.2560219841653259,
            -0.0, -8.149713241787042e-18, -0.0, -0.0, 0.08697526781400827,
            0.1570027480206658, 0.0, -0.0, 0.0]
    w56t = [1.293339538571538e-10, 1.3119044423334432e-11, 4.1324337407554035e-11,
            8.128395264876261e-12, 0.004135586895420447, 1.9675611277732737e-11,
            -2.429828286194846e-12, 0.03231412012197525, 2.40905011736792e-11,
            0.008103449345537962, 0.19094970511516976, 4.718209542934128e-11,
            2.281557343693795e-11, 3.224552246043005e-11, 5.1293017695592876e-12,
            0.06616898576702318, 0.11711765760427535, 3.953457010792507e-11,
            0.014474765616092396, 0.066735729154355]
    w57t = [0.0, 0.0, 2.183222085500859e-17, -0.0, 0.181926392116249, -0.0, -0.0,
            -1.5869981902154903e-16, -0.0, -0.0, 0.32681877979942786, -0.0,
            -0.008745171915676905, 1.5869981902154903e-16, -0.0, 0.0, 0.0, -0.0, -0.0, 0.0]
    w58t = [2.958438629696401e-11, 0.04575491953539697, 0.0004820271027547595,
            0.0002773957565989856, 0.06681819160451097, 1.1647671767122028e-11,
            -1.1221845475965489e-11, 0.07920753237466555, 5.0823604024479756e-11,
            8.320390188629942e-11, 0.12758286564217977, 0.0005899568031571467,
            3.527132103689566e-11, 3.739032472085168e-11, 0.004128213010937313,
            0.060702451625124505, 5.099068198569209e-11, 0.022300346322857812,
            1.531388561082186e-11, 0.09215609991881214]
    w59t = [-0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, 0.0, -0.0, -0.0, 0.24435462215566234,
            -0.0, 1.1102230246251565e-16, 0.0, -0.0, 0.033856165269938565,
            0.22178921257439893, -0.0, -0.0, 0.0]
    w60t = [1.197119223629002e-10, 1.0964813614536123e-10, 1.6625233481677722e-10,
            7.036588322795268e-11, 0.025392719609852527, 5.1716455986397435e-11,
            5.604600196426509e-12, 0.03567773430806398, 1.2694413905819024e-10,
            0.01108234845961798, 0.20075178385073517, 9.03052454929875e-10,
            4.338676855285493e-11, 1.3394744000844658e-10, 2.4006013284169077e-11,
            0.08680970956003806, 0.08544740107386053, 6.568466058879663e-10,
            3.989110076363996e-10, 0.054838300327437806]
    w61t = [0.0, 0.0, 0.06455652837124166, 0.0, 0.17806945118716286, -0.1534166903019555,
            6.554560797119506e-17, 0.12351856122756986, 0.0, 0.0, 0.21867000759778857, 0.0,
            -0.023295149141940166, 0.0, -0.0032881605561044004, 0.1770017931069312,
            0.006143447950208067, 0.0, 0.04071674553919559, 0.1413234650199022]
    w62t = [0.0, 0.0, 0.07328869996428454, 0.0, 0.19593630860627667, -0.1516035031548817,
            -2.228175066036319e-17, 0.1290693464408622, 0.0, 0.0, 0.20897763141227396, 0.0,
            -0.02192543103531701, 0.0, -0.006471065809801198, 0.18010601289300754,
            7.366923752795404e-17, 0.0, 0.025951685743898865, 0.13667031493939605]
    w63t = [-0.0, -0.0, 0.0, -0.0, 0.6900000000000023, -0.0, -0.0, 0.0, -0.0, -0.0,
            0.19949253798767527, -0.0, -0.11949253798767445, -0.0, -0.0, 0.0, 0.0, -0.0,
            0.0, 0.0]
    w64t = [3.514705859068149e-11, 0.10214522076326914, 1.5973969676769913e-9,
            8.750509707972237e-10, 0.23010239599369686, -0.025107535873741628,
            0.022839041959543116, 0.1484093094529992, -2.6374884760198716e-10,
            -3.153944096601185e-11, 0.1259968286749299, -6.592245864531316e-10,
            -0.033034603266437215, -1.0048020292583086e-10, -0.05977664457598265,
            0.1711014805850215, -7.484390185339501e-11, 5.423264358124763e-10,
            -2.645110205956699e-10, 0.08732450463112845]
    w65t = [-0.0, -0.0, 0.0, -0.0, 0.6899999999999881, -0.0, -0.0, 0.0, -0.0, -0.0,
            0.19949253798767191, -0.0, -0.11949253798766571, -0.0, -0.0, 0.0, -0.0, -0.0,
            0.0, 0.0]
    w66t = [3.1346018670849086e-10, 2.4699533473449294e-10, 7.946585372882473e-10,
            1.7211158903222362e-10, 0.354561355804607, -3.6658791888432367e-10,
            8.523092152390044e-10, 2.971644471268001e-10, 8.376592645525147e-10,
            1.3758599124820434e-10, 0.16641976056066163, -2.8470131081273173e-10,
            -0.014268236577701886, 4.2616973821312224e-10, -0.01533295007045943,
            0.27293820948799796, 4.782414806471252e-10, 1.5817354331413705e-10,
            0.005681856376361206, 3.552933477277175e-10]
    w67t = [-0.0, -0.0, 0.0, -0.0, 0.6751155811095971, -0.0, -0.0, 0.0, -0.0, -0.0,
            0.208898224965582, -0.0, -0.11401380607517626, -0.0, -0.0, 0.0, 0.0, -0.0, -0.0,
            0.0]
    w68t = [9.396447803845082e-11, 0.09919154707338323, 6.368346293116606e-10,
            3.4659752542329914e-10, 0.22064663466240597, -1.061430905867918e-9,
            0.020025141282779442, 0.1266686483319935, 5.679461315964227e-12,
            1.1754652331706741e-11, 0.12107267108578944, -4.262326626042898e-10,
            -0.04188688600627883, 1.5637092276435317e-10, -0.0652186801900443,
            0.16284169105163024, 6.247272381306623e-11, 4.81472955329701e-10,
            1.7210168938233205e-10, 0.1266592322287559]
    w69t = [-0.0, -0.0, 0.0, -0.0, 0.5385298461146968, -0.0, 0.0, 0.0, -0.0, -0.0, 0.0,
            -0.18, -0.0, 0.0, 0.0, 0.41147015388533015, -0.0, -0.0, -0.0, 0.0]
    w70t = [1.41726398168247e-10, 1.1299052444425518e-10, 5.359423143640224e-10,
            8.260406045447116e-11, 0.36497113598185654, -6.906069341391924e-11,
            4.035070873116829e-10, 2.0028543669526173e-10, 2.1735713433761467e-10,
            1.0421338716830917e-10, 0.11284986038577438, -6.494833825625461e-11,
            -0.027835392432297344, 1.7675897843157214e-10, -0.015067449302544307,
            0.2566729326825135, 1.785289060258331e-10, 7.917147902202065e-11,
            0.07840891041485702, 1.707636450614387e-10]
    w71t = [0.0, 0.0, 0.0, 0.0, 8.881784197001252e-16, 0.0, 0.8299999999999991, 0.0, 0.0,
            0.0, 0.0, 0.0, -0.22, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    w72t = [0.0, 0.0, 0.0, 0.0, 0.7959999999999998, 0.0, -1.2210381468724101e-16, 0.0, 0.0,
            0.0, 0.0, 0.0, -0.21999999999999986, 0.0, -1.515142386685086e-16,
            3.190790756317167e-16, -4.1901986586950697e-16, 0.0, 0.034000000000000336, 0.0]
    w73t = [-0.0, -0.0, 0.0, 1.722299551653608e-17, 0.3300000000000011, 0.2799999999999992,
            0.0, -1.1102230246251417e-17, 0.0, 0.0, -2.317705389555333e-16,
            -2.4719822535691137e-16, -0.0, 0.0, 1.397810916546131e-16, -0.0, 0.0, -0.0, 0.0,
            0.0]
    w74t = [2.2594248153239878e-10, 0.05056245427150847, 0.09016607489794702,
            0.08107397458595665, 8.570900866809083e-9, 2.120718807857173e-10,
            2.071209857183213e-9, 4.394476961593897e-9, 1.59553380057758e-8,
            0.05544201879614573, 0.045210654597117185, 0.04481746688838601,
            0.03452777999237384, 6.017183442494671e-9, 0.07894326936896161,
            1.4266089152736319e-9, 1.4565259484448497e-10, 0.05034315831282506,
            0.07891309981409753, 9.455295887711091e-9]
    w75t = [0.0, -0.0, -0.0, 0.0, 0.796, -0.0, 0.0, -0.0, -0.0, 0.0, -0.0, -0.0, -0.22,
            -0.0, -0.0, 0.0, 0.0, -0.0, 0.034, 0.0]
    w76t = [4.225325247315316e-9, 5.198354408340478e-9, 6.387224751420317e-9,
            6.261094652394527e-9, 4.450565546124522e-6, 5.5433483931899384e-9,
            0.2041130559415384, 3.9980598588196765e-9, 0.1380925438177799,
            7.812543535725312e-9, 0.06444311820226425, 8.000899706646038e-9,
            2.5310445399637496e-9, 9.266895648718285e-9, 4.109376070857174e-9,
            0.20334667226231262, 4.027038710705652e-9, 3.985990710895623e-9,
            8.360233183101534e-8, 4.261031793856425e-9]
    w77t = [0.0, 0.0, 0.0, 2.7755575615628914e-17, 1.6930901125533637e-15, 0.0,
            0.8299999999999983, 0.0, 0.0, 0.0, -1.1934897514720433e-15, 0.0,
            -0.21999999999999903, 0.0, 1.3877787807814457e-16, 4.218847493575595e-15, 0.0,
            0.0, 0.0, 0.0]
    w78t = [1.1182057041359085e-10, 2.995946324855838e-9, 0.07674520130300927,
            0.07666521247748433, 2.9757070740658692e-9, 1.0934128476258304e-10,
            0.0776133722313396, 5.642797683376111e-8, 8.884501566635404e-11,
            3.6488867777397636e-9, 0.0760319669001114, 0.07533524505618218,
            2.1219674373801457e-9, 1.4977332982630043e-10, 0.07453266281027018,
            0.07692306072059908, 5.5296485556325755e-11, 0.07615320747980886,
            5.331579488169772e-11, 2.2823181964765645e-9]
    w79t = [0.0, 2.7755575615628914e-17, 2.7755575615628914e-17, 2.7755575615628914e-17,
            0.0, 0.0, 0.830000000000001, 0.0, 4.163336342344337e-16, 0.0, 0.0, 0.0, -0.22,
            0.0, 0.0, -1.1379786002407855e-15, 2.7755575615628914e-17,
            2.7755575615628914e-17, 0.0, 2.7755575615628914e-17]
    w80t = [4.250124696499757e-9, 4.678156603162632e-9, 5.3720677452260745e-9,
            5.151213859318981e-9, 8.23318093256969e-6, 3.544983662264096e-9,
            0.20388296087146449, 3.272926289372493e-9, 1.30304133031642e-7,
            6.752179566818858e-9, 1.4717627067313058e-8, 3.967909754190531e-9,
            2.1981477405946624e-9, 5.336831864011093e-9, 2.8127069564404503e-9,
            0.20320061551825436, 5.6625306666524776e-9, 3.4887434173951917e-9,
            0.2029079848247328, 4.094332913765017e-9]

    @test isapprox(w1.weights, w1t)
    @test isapprox(w2.weights, w2t)
    @test isapprox(w3.weights, w3t, rtol = 0.01)
    @test isapprox(w4.weights, w4t)
    @test isapprox(w5.weights, w5t, rtol = 0.01)
    @test isapprox(w6.weights, w6t)
    @test isapprox(w7.weights, w7t, rtol = 0.01)
    @test isapprox(w8.weights, w8t)
    @test isapprox(w9.weights, w9t, rtol = 1.0e-7)
    @test isapprox(w10.weights, w10t)

    @test isapprox(sum(w1.weights), ssl1)
    @test isapprox(sum(w2.weights), ssl1)
    @test isapprox(sum(w3.weights), ssl1)
    @test isapprox(sum(w4.weights), ssl1)
    @test isapprox(sum(w5.weights), ssl1)
    @test isapprox(sum(w6.weights), ssl1)
    @test isapprox(sum(w7.weights), ssl1)
    @test isapprox(sum(w8.weights), ssl1)
    @test isapprox(sum(w9.weights), ssl1)
    @test isapprox(sum(w10.weights), ssl1)

    @test isapprox(w11.weights, w11t)
    @test isapprox(w12.weights, w12t)
    @test isapprox(w13.weights, w13t)
    @test isapprox(w14.weights, w14t)
    @test isapprox(w15.weights, w15t)
    @test isapprox(w16.weights, w16t)
    @test isapprox(w17.weights, w17t)
    @test isapprox(w18.weights, w18t)
    @test isapprox(w19.weights, w19t)
    @test isapprox(w20.weights, w20t)

    @test isapprox(sum(w11.weights), ssl2)
    @test isapprox(sum(w12.weights), ssl2)
    @test isapprox(sum(w13.weights), ssl2)
    @test isapprox(sum(w14.weights), ssl2)
    @test isapprox(sum(w15.weights), ssl2)
    @test isapprox(sum(w16.weights), ssl2)
    @test isapprox(sum(w17.weights), ssl2)
    @test isapprox(sum(w18.weights), ssl2)
    @test isapprox(sum(w19.weights), ssl2)
    @test isapprox(sum(w20.weights), ssl2)

    @test isapprox(w21.weights, w21t)
    @test isapprox(w22.weights, w22t)
    @test isapprox(w23.weights, w23t)
    @test isapprox(w24.weights, w24t)
    @test isapprox(w25.weights, w25t)
    @test isapprox(w26.weights, w26t)
    @test isapprox(w27.weights, w27t)
    @test isapprox(w28.weights, w28t)
    @test isapprox(w29.weights, w29t)
    @test isapprox(w30.weights, w30t)

    @test isapprox(sum(w21.weights), ssl3)
    @test isapprox(sum(w22.weights), ssl3)
    @test isapprox(sum(w23.weights), ssl3)
    @test isapprox(sum(w24.weights), ssl3)
    @test isapprox(sum(w25.weights), ssl3)
    @test isapprox(sum(w26.weights), ssl3)
    @test isapprox(sum(w27.weights), ssl3)
    @test isapprox(sum(w28.weights), ssl3)
    @test isapprox(sum(w29.weights), ssl3)
    @test isapprox(sum(w30.weights), ssl3)

    @test isapprox(w31.weights, w31t)
    @test isapprox(w32.weights, w32t)
    @test isapprox(w33.weights, w33t)
    @test isapprox(w34.weights, w34t)
    @test isapprox(w35.weights, w35t)
    @test isapprox(w36.weights, w36t)
    @test isapprox(w37.weights, w37t)
    @test isapprox(w38.weights, w38t)
    @test isapprox(w39.weights, w39t)
    @test isapprox(w40.weights, w40t)

    @test isapprox(sum(w31.weights), ssl4)
    @test isapprox(sum(w32.weights), ssl4)
    @test isapprox(sum(w33.weights), ssl4)
    @test isapprox(sum(w34.weights), ssl4)
    @test isapprox(sum(w35.weights), ssl4)
    @test isapprox(sum(w36.weights), ssl4)
    @test isapprox(sum(w37.weights), ssl4)
    @test isapprox(sum(w38.weights), ssl4)
    @test isapprox(sum(w39.weights), ssl4)
    @test isapprox(sum(w40.weights), ssl4)

    @test isapprox(w41.weights, w41t)
    @test isapprox(w42.weights, w42t)
    @test isapprox(w43.weights, w43t)
    @test isapprox(w44.weights, w44t)
    @test isapprox(w45.weights, w45t)
    @test isapprox(w46.weights, w46t)
    @test isapprox(w47.weights, w47t)
    @test isapprox(w48.weights, w48t)
    @test isapprox(w49.weights, w49t)
    @test isapprox(w50.weights, w50t, rtol = 1.0e-5)

    @test isapprox(sum(w41.weights), ssl5)
    @test isapprox(sum(w42.weights), ssl5)
    @test isapprox(sum(w43.weights), ssl5)
    @test isapprox(sum(w44.weights), ssl5)
    @test isapprox(sum(w45.weights), ssl5)
    @test isapprox(sum(w46.weights), ssl5)
    @test isapprox(sum(w47.weights), ssl5)
    @test isapprox(sum(w48.weights), ssl5)
    @test isapprox(sum(w49.weights), ssl5)
    @test isapprox(sum(w50.weights), ssl5)

    @test isapprox(w41.weights, w1.weights)
    @test isapprox(w42.weights, w2.weights)
    @test isapprox(w43.weights, w3.weights)
    @test !isapprox(w44.weights, w4.weights)
    @test isapprox(w45.weights, w5.weights)
    @test !isapprox(w46.weights, w6.weights)
    @test isapprox(w47.weights, w7.weights)
    @test !isapprox(w48.weights, w8.weights)
    @test isapprox(w49.weights, w9.weights)
    @test !isapprox(w50.weights, w10.weights)

    @test isapprox(w51.weights, w51t)
    @test isapprox(w52.weights, w52t)
    @test isapprox(w53.weights, w53t)
    @test isapprox(w54.weights, w54t)
    @test isapprox(w55.weights, w55t)
    @test isapprox(w56.weights, w56t)
    @test isapprox(w57.weights, w57t)
    @test isapprox(w58.weights, w58t)
    @test isapprox(w59.weights, w59t)
    @test isapprox(w60.weights, w60t)

    @test isapprox(sum(w51.weights), ssl6)
    @test isapprox(sum(w52.weights), ssl6)
    @test isapprox(sum(w53.weights), ssl6)
    @test isapprox(sum(w54.weights), ssl6)
    @test isapprox(sum(w55.weights), ssl6)
    @test isapprox(sum(w56.weights), ssl6)
    @test isapprox(sum(w57.weights), ssl6)
    @test isapprox(sum(w58.weights), ssl6)
    @test isapprox(sum(w59.weights), ssl6)
    @test isapprox(sum(w60.weights), ssl6)

    @test isapprox(w51.weights, w11.weights)
    @test isapprox(w52.weights, w12.weights)
    @test isapprox(w53.weights, w13.weights)
    @test !isapprox(w54.weights, w14.weights)
    @test isapprox(w55.weights, w15.weights)
    @test !isapprox(w56.weights, w16.weights)
    @test isapprox(w57.weights, w17.weights)
    @test !isapprox(w58.weights, w18.weights)
    @test isapprox(w59.weights, w19.weights)
    @test !isapprox(w60.weights, w20.weights)

    @test isapprox(w61.weights, w61t)
    @test isapprox(w62.weights, w62t)
    @test isapprox(w63.weights, w63t)
    @test isapprox(w64.weights, w64t)
    @test isapprox(w65.weights, w65t)
    @test isapprox(w66.weights, w66t)
    @test isapprox(w67.weights, w67t)
    @test isapprox(w68.weights, w68t)
    @test isapprox(w69.weights, w69t)
    @test isapprox(w70.weights, w70t)

    @test isapprox(sum(w61.weights), ssl7)
    @test isapprox(sum(w62.weights), ssl7)
    @test isapprox(sum(w63.weights), ssl7)
    @test isapprox(sum(w64.weights), ssl7)
    @test isapprox(sum(w65.weights), ssl7)
    @test isapprox(sum(w66.weights), ssl7)
    @test isapprox(sum(w67.weights), ssl7)
    @test isapprox(sum(w68.weights), ssl7)
    @test isapprox(sum(w69.weights), ssl7)
    @test isapprox(sum(w70.weights), ssl7)

    @test isapprox(w61.weights, w21.weights)
    @test isapprox(w62.weights, w22.weights)
    @test isapprox(w63.weights, w23.weights)
    @test !isapprox(w64.weights, w24.weights)
    @test isapprox(w65.weights, w25.weights)
    @test !isapprox(w66.weights, w26.weights)
    @test isapprox(w67.weights, w27.weights)
    @test !isapprox(w68.weights, w28.weights)
    @test isapprox(w69.weights, w29.weights)
    @test !isapprox(w70.weights, w30.weights)

    @test isapprox(w71.weights, w71t)
    @test isapprox(w72.weights, w72t)
    @test isapprox(w73.weights, w73t)
    @test isapprox(w74.weights, w74t)
    @test isapprox(w75.weights, w75t)
    @test isapprox(w76.weights, w76t, rtol = 5.0e-8)
    @test isapprox(w77.weights, w77t)
    @test isapprox(w78.weights, w78t)
    @test isapprox(w79.weights, w79t)
    @test isapprox(w80.weights, w80t, rtol = 5.0e-6)

    @test isapprox(sum(w71.weights), ssl8)
    @test isapprox(sum(w72.weights), ssl8)
    @test isapprox(sum(w73.weights), ssl8)
    @test isapprox(sum(w74.weights), ssl8)
    @test isapprox(sum(w75.weights), ssl8)
    @test isapprox(sum(w76.weights), ssl8)
    @test isapprox(sum(w77.weights), ssl8)
    @test isapprox(sum(w78.weights), ssl8)
    @test isapprox(sum(w79.weights), ssl8)
    @test isapprox(sum(w80.weights), ssl8)

    @test isapprox(w71.weights, w31.weights)
    @test isapprox(w72.weights, w32.weights)
    @test isapprox(w73.weights, w33.weights)
    @test !isapprox(w74.weights, w34.weights)
    @test isapprox(w75.weights, w35.weights)
    @test !isapprox(w76.weights, w36.weights)
    @test isapprox(w77.weights, w37.weights)
    @test !isapprox(w78.weights, w38.weights)
    @test isapprox(w79.weights, w39.weights)
    @test !isapprox(w80.weights, w40.weights)
end

@testset "Network and Dendrogram Constraints $(:SD)" begin
    portfolio = Portfolio(; prices = prices, solvers = solvers)
    asset_statistics!(portfolio; calc_kurt = false)

    rm = :SD
    obj = :Min_Risk
    opt = OptimiseOpt(; obj = obj, rm = rm, l = l, rf = rf)
    CV = centrality_vector(portfolio;
                           cor_opt = CorOpt(; dist = DistOpt(; method = POCorDist())))
    B_1 = connection_matrix(portfolio;
                            cor_opt = CorOpt(; dist = DistOpt(; method = POCorDist())),
                            network_opt = NetworkOpt(; method = :MST))
    L_A = cluster_matrix(portfolio;
                         cor_opt = CorOpt(; dist = DistOpt(; method = POCorDist())),
                         cluster_opt = ClusterOpt(; linkage = :ward,
                                                  genfunc = GenericFunction(;
                                                                            func = dbht_d)))

    w1 = optimise!(portfolio, opt)

    portfolio.a_vec_cent = CV
    portfolio.b_cent = minimum(CV)
    w2 = optimise!(portfolio, opt)

    portfolio.network_method = :IP
    portfolio.network_ip = B_1
    w3 = optimise!(portfolio, opt)

    portfolio.network_method = :SDP
    portfolio.network_sdp = B_1
    w4 = optimise!(portfolio, opt)

    portfolio.network_method = :IP
    portfolio.network_ip = L_A
    w5 = optimise!(portfolio, opt)

    portfolio.network_method = :SDP
    portfolio.network_sdp = L_A
    w6 = optimise!(portfolio, opt)

    portfolio.a_vec_cent = []
    portfolio.b_cent = Inf

    portfolio.network_method = :IP
    portfolio.network_ip = B_1
    w7 = optimise!(portfolio, opt)

    portfolio.network_method = :SDP
    portfolio.network_sdp = B_1
    w8 = optimise!(portfolio, opt)

    portfolio.network_method = :IP
    portfolio.network_ip = L_A
    w9 = optimise!(portfolio, opt)

    portfolio.network_method = :SDP
    portfolio.network_sdp = L_A
    w10 = optimise!(portfolio, opt)

    portfolio = Portfolio(; prices = prices, solvers = solvers)
    asset_statistics!(portfolio; calc_kurt = false)

    rm = :SD
    obj = :Utility
    opt = OptimiseOpt(; obj = obj, rm = rm, l = l, rf = rf)
    CV = centrality_vector(portfolio;
                           cor_opt = CorOpt(; dist = DistOpt(; method = POCorDist())),
                           network_opt = NetworkOpt(; method = :TMFG,
                                                    tmfg_genfunc = GenericFunction(;
                                                                                   func = dbht_d)))
    B_1 = connection_matrix(portfolio;
                            cor_opt = CorOpt(; dist = DistOpt(; method = POCorDist())),
                            network_opt = NetworkOpt(; method = :TMFG,
                                                     tmfg_genfunc = GenericFunction(;
                                                                                    func = dbht_d)))
    L_A = cluster_matrix(portfolio;
                         cor_opt = CorOpt(; dist = DistOpt(; method = POCorDist())),
                         cluster_opt = ClusterOpt(; linkage = :DBHT,
                                                  genfunc = GenericFunction(;
                                                                            func = dbht_d)))

    w11 = optimise!(portfolio, opt)

    portfolio.a_vec_cent = CV
    portfolio.b_cent = minimum(CV)
    w12 = optimise!(portfolio, opt)

    portfolio.network_method = :IP
    portfolio.network_ip = B_1
    w13 = optimise!(portfolio, opt)

    portfolio.network_method = :SDP
    portfolio.network_sdp = B_1
    w14 = optimise!(portfolio, opt)

    portfolio.network_method = :IP
    portfolio.network_ip = L_A
    w15 = optimise!(portfolio, opt)

    portfolio.network_method = :SDP
    portfolio.network_sdp = L_A
    w16 = optimise!(portfolio, opt)

    portfolio.a_vec_cent = []
    portfolio.b_cent = Inf

    portfolio.network_method = :IP
    portfolio.network_ip = B_1
    w17 = optimise!(portfolio, opt)

    portfolio.network_method = :SDP
    portfolio.network_sdp = B_1
    w18 = optimise!(portfolio, opt)

    portfolio.network_method = :IP
    portfolio.network_ip = L_A
    w19 = optimise!(portfolio, opt)

    portfolio.network_method = :SDP
    portfolio.network_sdp = L_A
    w20 = optimise!(portfolio, opt)

    portfolio = Portfolio(; prices = prices, solvers = solvers)
    asset_statistics!(portfolio; calc_kurt = false)

    rm = :SD
    obj = :Sharpe
    opt = OptimiseOpt(; obj = obj, rm = rm, l = l, rf = rf)
    CV = centrality_vector(portfolio;
                           cor_opt = CorOpt(; dist = DistOpt(; method = POCorDist())),
                           network_opt = NetworkOpt(; method = :TMFG,
                                                    tmfg_genfunc = GenericFunction(;
                                                                                   func = dbht_d)))
    B_1 = connection_matrix(portfolio;
                            cor_opt = CorOpt(; dist = DistOpt(; method = POCorDist())),
                            network_opt = NetworkOpt(; method = :TMFG,
                                                     tmfg_genfunc = GenericFunction(;
                                                                                    func = dbht_d)))
    L_A = cluster_matrix(portfolio;
                         cor_opt = CorOpt(; dist = DistOpt(; method = POCorDist())),
                         cluster_opt = ClusterOpt(; linkage = :DBHT,
                                                  genfunc = GenericFunction(;
                                                                            func = dbht_d)))

    w21 = optimise!(portfolio, opt)

    portfolio.a_vec_cent = CV
    portfolio.b_cent = minimum(CV)
    w22 = optimise!(portfolio, opt)

    portfolio.network_method = :IP
    portfolio.network_ip = B_1
    w23 = optimise!(portfolio, opt)

    portfolio.network_method = :SDP
    portfolio.network_sdp = B_1
    w24 = optimise!(portfolio, opt)

    portfolio.network_method = :IP
    portfolio.network_ip = L_A
    w25 = optimise!(portfolio, opt)

    portfolio.network_method = :SDP
    portfolio.network_sdp = L_A
    w26 = optimise!(portfolio, opt)

    portfolio.a_vec_cent = []
    portfolio.b_cent = Inf

    portfolio.network_method = :IP
    portfolio.network_ip = B_1
    w27 = optimise!(portfolio, opt)

    portfolio.network_method = :SDP
    portfolio.network_sdp = B_1
    w28 = optimise!(portfolio, opt)

    portfolio.network_method = :IP
    portfolio.network_ip = L_A
    w29 = optimise!(portfolio, opt)

    portfolio.network_method = :SDP
    portfolio.network_sdp = L_A
    w30 = optimise!(portfolio, opt)

    portfolio = Portfolio(; prices = prices, solvers = solvers)
    asset_statistics!(portfolio; calc_kurt = false)

    rm = :SD
    obj = :Max_Ret
    opt = OptimiseOpt(; obj = obj, rm = rm, l = l, rf = rf)
    CV = centrality_vector(portfolio;
                           cor_opt = CorOpt(; dist = DistOpt(; method = POCorDist())),
                           network_opt = NetworkOpt(; method = :TMFG,
                                                    tmfg_genfunc = GenericFunction(;
                                                                                   func = dbht_d)))
    B_1 = connection_matrix(portfolio;
                            cor_opt = CorOpt(; dist = DistOpt(; method = POCorDist())),
                            network_opt = NetworkOpt(; method = :TMFG,
                                                     tmfg_genfunc = GenericFunction(;
                                                                                    func = dbht_d)))
    L_A = cluster_matrix(portfolio;
                         cor_opt = CorOpt(; dist = DistOpt(; method = POCorDist())),
                         cluster_opt = ClusterOpt(; linkage = :DBHT,
                                                  genfunc = GenericFunction(;
                                                                            func = dbht_d)))

    w31 = optimise!(portfolio, opt)

    portfolio.a_vec_cent = CV
    portfolio.b_cent = minimum(CV)
    w32 = optimise!(portfolio, opt)

    portfolio.network_method = :IP
    portfolio.network_ip = B_1
    w33 = optimise!(portfolio, opt)

    portfolio.network_method = :SDP
    portfolio.network_sdp = B_1
    w34 = optimise!(portfolio, opt)

    portfolio.network_method = :IP
    portfolio.network_ip = L_A
    w35 = optimise!(portfolio, opt)

    portfolio.network_method = :SDP
    portfolio.network_sdp = L_A
    w36 = optimise!(portfolio, opt)

    portfolio.a_vec_cent = []
    portfolio.b_cent = Inf

    portfolio.network_method = :IP
    portfolio.network_ip = B_1
    w37 = optimise!(portfolio, opt)

    portfolio.network_method = :SDP
    portfolio.network_sdp = B_1
    w38 = optimise!(portfolio, opt)

    portfolio.network_method = :IP
    portfolio.network_ip = L_A
    w39 = optimise!(portfolio, opt)

    portfolio.network_method = :SDP
    portfolio.network_sdp = L_A
    w40 = optimise!(portfolio, opt)

    w1t = [0.00791850924098073, 0.030672065216453506, 0.010501402809199123,
           0.027475241969187995, 0.012272329269527841, 0.03339587076426262,
           1.4321532289072258e-6, 0.13984297866711365, 2.4081081597353397e-6,
           5.114425959766348e-5, 0.2878111114337346, 1.5306036912879562e-6,
           1.1917690994187655e-6, 0.12525446872321966, 6.630910273840812e-6,
           0.015078706008184504, 8.254970801614574e-5, 0.1930993918762575,
           3.0369340845779798e-6, 0.11652799957572683]
    w2t = [3.5426430661259544e-11, 0.07546661314221123, 0.021373033743425803,
           0.027539611826011074, 0.023741467255115698, 0.12266515815974924,
           2.517181275812244e-6, 0.22629893782442134, 3.37118211246211e-10,
           0.02416530860824232, 3.3950118687352606e-10, 1.742541411959769e-6,
           1.5253730188343444e-6, 5.304686980295978e-11, 0.024731789616991084,
           3.3711966852218057e-10, 7.767147353183488e-11, 0.30008056166009706,
           1.171415437554966e-10, 0.1539317317710031]
    w3t = [0.0, 0.0, 0.08531373477060625, 0.0, 0.0, 0.0, 0.0, 0.2553257507395435, 0.0, 0.0,
           5.551115123125783e-17, 0.0, 0.0, 0.0, 0.042999084822445105, 0.0, 0.0,
           0.38591329853627393, 0.0, 0.23044813113113127]
    w4t = [4.2907851108373285e-11, 0.0754418669962984, 0.021396172655536325,
           0.027531481813488985, 0.02375148897766012, 0.12264432042703496,
           3.777946382632432e-6, 0.2263233585633904, 3.8909758570393176e-10,
           0.024184506959405477, 3.8945719747726095e-10, 2.597992728770223e-6,
           2.2916747013451583e-6, 5.942491440558677e-11, 0.02472298577516722,
           3.89238712058158e-10, 9.010863437870188e-11, 0.30009565874345184,
           1.32909841405546e-10, 0.15389948998160896]
    w5t = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
           0.053218458202620444, 0.0, 0.0, 0.5647264843921129, 0.0, 0.38205505740526663]
    w6t = [6.011468296535483e-11, 9.424732042613459e-6, 5.527537353284497e-6,
           2.903730060970158e-6, 4.265000229139505e-6, 5.05868769590198e-6,
           3.009791891015968e-6, 1.5268395403568086e-5, 5.657961006483616e-10,
           2.68155059410767e-6, 5.662844616891034e-10, 3.196842024458429e-6,
           2.419924540093777e-6, 8.983883474709306e-11, 0.05335383109918724,
           5.659272794304194e-10, 1.3052770418372395e-10, 0.5616352353647922,
           1.9587068208203376e-10, 0.38495717516982575]
    w7t = [0.0, 0.0, 0.0, 0.05362285528694267, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
           0.4241445775272498, 0.0, 0.0, 0.0, 0.0, 0.02702488928992336, 0.0,
           0.3034370015606549, 0.0, 0.19177067633522926]
    w8t = [1.1008281068185902e-5, 0.055457116453828725, 0.012892903094396706,
           0.03284102649502972, 0.014979204379343122, 0.057886691097825904,
           1.0224197847100319e-6, 9.70550406073092e-6, 2.045810229626667e-6,
           0.012049795193184915, 0.3735114108030411, 1.330358085433759e-6,
           1.0648304534905729e-6, 9.906438750370314e-6, 0.007878565110236062,
           0.022521836037796082, 3.1895242150194783e-6, 0.26190144912467656,
           1.3462532236761872e-6, 0.14803938279076995]
    w9t = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6083352375051165, 0.0, 0.0,
           0.0, 0.0, 0.0578030323485863, 0.0, 0.0, 0.0, 0.33386173014629716]
    w10t = [6.5767880723988875e-6, 4.623865766176694e-6, 3.533665790240184e-6,
            2.181035748649495e-6, 2.5215556700206228e-6, 1.858111674425038e-6,
            2.5125294721249436e-6, 2.7192384937770815e-6, 8.752492794952926e-7,
            9.63854668493286e-7, 0.6119910196342336, 2.4053903351549724e-6,
            9.311964129813141e-7, 9.706505481488379e-6, 1.3337324741660364e-5,
            0.05681677180817997, 4.294323401245429e-5, 1.2805451158943426e-5,
            1.6336794237581935e-6, 0.33108007988138427]
    w11t = [1.213462967397082e-7, 1.9380369577075775e-7, 2.535435909493459e-7,
            2.01415606821647e-7, 0.7741878767667363, 4.576078711246336e-8,
            0.10998669529571711, 1.1949104304770047e-7, 2.5377660584396696e-7,
            1.1474781030976249e-7, 1.0972151062076415e-7, 4.216073184323918e-8,
            3.0871518191419254e-8, 6.865709075114092e-8, 3.052542028090083e-8,
            0.11582231440207927, 7.938918380642579e-7, 1.2800156024200203e-7,
            4.4673807713760866e-7, 1.5908228364211473e-7]
    w12t = [5.883853770665528e-11, 1.3060642855112434e-9, 1.301061529798232e-9,
            0.3917666856975877, 1.3098519102344834e-9, 7.956507457157616e-11,
            0.21166320819473436, 0.20931941976220955, 2.273559804279206e-10,
            3.6637561839382273e-10, 0.18725030230174003, 1.8998931162824488e-7,
            9.064072407075894e-8, 5.729783545761994e-11, 9.543040293061169e-8,
            1.3057416024452685e-9, 2.7196597566046784e-10, 1.3214884830931058e-9,
            2.9606943663580964e-10, 8.161341000042588e-11]
    w13t = [0.0, 0.0, 0.0, 0.5221634099560039, 1.1102230246251565e-16, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.4778365900439961, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    w14t = [2.7039946953296587e-11, 5.304815002281447e-10, 5.312907530858779e-10,
            0.39174140576279676, 5.364912966852598e-10, 3.2595953616397375e-11,
            0.21166437097929608, 0.20932312570878908, 9.329097923092044e-11,
            1.587497803742671e-10, 0.1872709236586728, 8.769818453443376e-8,
            4.067205991501162e-8, 2.722555750549475e-11, 4.225876855666987e-8,
            5.326193576385418e-10, 1.100125614453946e-10, 5.288670917390021e-10,
            1.1950917222761676e-10, 3.32586012916347e-11]
    w15t = [0.0, 0.0, 0.0, 0.5221634438462894, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.4778365561537106, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    w16t = [3.181577804133192e-11, 7.410247413753799e-10, 7.42562355400477e-10,
            0.42977864615510014, 7.48674144591813e-10, 5.159514835577257e-11,
            0.05180804975958271, 7.796097271775099e-7, 1.296211874745331e-10,
            2.2542619108251165e-10, 0.5184121925711523, 1.9231509760578446e-7,
            6.755584073688425e-8, 3.206251898648789e-11, 6.746863673969513e-8,
            7.43544013609555e-10, 1.5596696897421446e-10, 7.384192478079624e-10,
            1.7169610373651742e-10, 5.245402734999976e-11]
    w17t = [0.0, 0.0, 0.0, 0.0, 0.8446617044700805, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.15533829552991954, 0.0, 0.0, 0.0, 0.0]
    w18t = [1.6606069518619783e-7, 2.2569375787121754e-7, 2.2360042487437665e-7,
            1.504981891990987e-7, 0.7741843310935411, 5.8031861445512475e-8,
            0.10999501737171542, 1.535615052543095e-7, 3.304005997767513e-7,
            1.477203844994807e-7, 1.4017814406853923e-7, 5.4322270748683715e-8,
            4.030051858877857e-8, 8.720973486396945e-8, 3.929473616299964e-8,
            0.1158177362616575, 3.2117894566063483e-7, 1.6454413556457108e-7,
            4.057699559196934e-7, 2.0690722623701084e-7]
    w19t = [0.0, 0.0, 0.0, 0.0, 0.8446617596520674, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.15533824034793267, 0.0, 0.0, 0.0, 0.0]
    w20t = [1.8460941461947824e-7, 1.9874083398189004e-7, 2.461002657842926e-7,
            1.601344417983893e-7, 0.8425793883145211, 6.66257124267573e-8,
            1.3400489474042612e-7, 1.2847388610565929e-7, 5.707510804798721e-7,
            1.852699283786712e-7, 1.7402169390193984e-7, 6.113254557309974e-8,
            4.509051933411562e-8, 1.030888422984093e-7, 4.498178692052351e-8,
            0.1574162193807334, 4.3032204436845463e-7, 1.4778919608274027e-7,
            1.3181071433238867e-6, 1.9306051549892292e-7]
    w21t = [5.2456519308516375e-9, 1.3711772920494845e-8, 1.828307590553689e-8,
            1.1104551657521415e-8, 0.5180586238310904, 1.7475553033958716e-9,
            0.06365101880957791, 1.1504018896467816e-8, 1.0491794751235226e-8,
            5.650081752041375e-9, 9.262635444153261e-9, 1.4051602963640009e-9,
            9.923106257589635e-10, 3.050812431838031e-9, 9.966076078173243e-10,
            0.1432679499627637, 0.19649701265872604, 1.0778757180749977e-8,
            0.07852527921167819, 1.1301377011579277e-8]
    w22t = [3.580964533638559e-11, 9.277053125069126e-10, 9.835000020421294e-10,
            0.48362433627647977, 1.3062621693742353e-9, 3.4185103123124565e-11,
            0.28435553885288534, 0.17984723049407092, 1.693432131576565e-10,
            2.621850918060399e-10, 0.05217287791621345, 5.161470450551339e-9,
            2.4492072342885186e-9, 4.790090083985224e-11, 2.6687808589346464e-9,
            1.0539691276230515e-9, 2.0827050059613154e-10, 8.48619997995145e-10,
            2.329804826102029e-10, 7.01603778985698e-11]
    w23t = [1.721820848503481e-11, 4.612136611876603e-11, 4.654661610120085e-11,
            5.4311966167325844e-11, 4.509276429538823e-11, 2.5105203761711743e-11,
            0.9999999992888641, 5.385076217256054e-11, 9.813277452536767e-12,
            3.590165511089574e-11, 5.3394014993982584e-11, 4.855963134775048e-11,
            4.3001784409793634e-11, 1.7552373893068702e-11, 4.2859679156734115e-11,
            4.645550624708972e-11, 2.1016260978513294e-11, 4.56055004014961e-11,
            3.102658186332549e-11, 2.770273726611377e-11]
    w24t = [5.024048627217141e-14, 1.5886394555170722e-12, 1.5924178422101003e-12,
            0.24554600920016964, 1.6332399340098677e-12, 1.0616523261700116e-13,
            0.0959435974734496, 0.2562392110969168, 2.7829378542014683e-13,
            4.877264338339266e-13, 0.402271160351581, 1.2141431002683076e-8,
            4.523802241069811e-9, 5.3729255370564864e-14, 5.20282725457044e-9,
            1.5947986553082236e-12, 3.43551721221936e-13, 1.5824894417145518e-12,
            3.8307499177570936e-13, 1.280439184666322e-13]
    w25t = [3.545654174928171e-12, 8.274610083444574e-12, 8.800101440916067e-12,
            9.946196524871394e-12, 1.1119487502408103e-11, 4.745336872225506e-12,
            0.6074274098748014, 1.0247200829383612e-11, 1.4145276752792622e-12,
            6.3466740561637836e-12, 0.3925725899995186, 9.215306610570093e-12,
            7.619896892506436e-12, 3.5776900678392008e-12, 7.360318525875661e-12,
            9.263220573598715e-12, 3.836345504038822e-12, 8.453631152518605e-12,
            6.617594557178819e-12, 5.2961689568170635e-12]
    w26t = [1.645653643573042e-14, 5.24765733463725e-13, 5.259882343242254e-13,
            0.38135427736460104, 5.383168184665497e-13, 3.606530943112124e-14,
            2.659687073118818e-8, 2.387877721283499e-8, 9.190837103177029e-14,
            1.6141883247690533e-13, 0.6186456659519168, 3.0002908666789655e-9,
            1.5771657468870285e-9, 1.747936659731388e-14, 1.6271322639469356e-9,
            5.267534803071819e-13, 1.1362061261344631e-13, 5.231419643587768e-13,
            1.2683767418078167e-13, 4.256536051656617e-14]
    w27t = [2.0965112097230834e-12, 2.198052346644996e-12, 1.9667657639650937e-12,
            2.0535413229602297e-12, 0.7622823927636417, 2.6157871252109093e-12,
            1.6394667710985665e-12, 2.5732973354729354e-12, 2.1633358406031507e-12,
            2.3918722896606665e-12, 2.602927046141792e-12, 2.2548132490543633e-12,
            2.3351802256510217e-12, 2.5744706316910684e-12, 2.433900476478536e-12,
            0.23771760719534718, 2.0772615592741314e-12, 2.544234451135619e-12,
            2.224029066401844e-12, 2.2655943185569015e-12]
    w28t = [1.131030668352105e-7, 9.736849167799003e-7, 1.5073134798603708e-7,
            9.724848430006577e-8, 0.3050368084415617, 4.717356198432155e-8,
            0.03491168150833796, 1.1696087757981777e-6, 2.1133994869894878e-7,
            1.38898043984553e-7, 0.23158972602737993, 2.930159759606465e-8,
            1.841227833023016e-8, 1.3415748037100702e-7, 2.038375787580353e-8,
            0.10856264505102015, 2.399490931217591e-7, 0.2072142228794291,
            2.522693174355702e-7, 0.11268131983060002]
    w29t = [1.1908280728767136e-12, 1.2343666441622383e-12, 1.067722183959735e-12,
            1.127939911690136e-12, 0.617653720906891, 1.637422528832974e-12,
            9.930865447414695e-13, 1.5553115843760996e-12, 1.089388229132098e-12,
            1.3847504587766668e-12, 1.5693615122529316e-12, 1.4011570899113892e-12,
            1.5947815925789765e-12, 1.5492021300042962e-12, 1.6041737741776276e-12,
            0.17191484828430395, 1.1173707571922096e-12, 1.5062504939730769e-12,
            0.2104314307858681, 1.313942784993723e-12]
    w30t = [1.9469265893415583e-7, 1.9264487242540467e-7, 2.232335248226593e-7,
            1.1246896197634285e-7, 0.40746626496017013, 8.182096647686593e-8,
            5.006408130895852e-8, 2.231759312172926e-7, 3.1463420211396267e-7,
            7.356649686989611e-6, 0.3255255880379563, 4.241919388356267e-8,
            2.851355716950413e-8, 2.082232132989454e-7, 3.1366090964049265e-8,
            0.15590260393465316, 0.010877836307400067, 2.4616586233596695e-7,
            0.10021813719562453, 2.6349139195481583e-7]
    w31t = [2.2425872064182503e-8, 2.3869345714744183e-8, 3.0249972071127925e-8,
            2.7441036408966832e-8, 2.674016472879105e-6, 1.1117945427350462e-8,
            0.9999969549278999, 1.6603745200809882e-8, 2.5223224538501096e-8,
            1.8088400365786554e-8, 1.61723807198772e-8, 1.1616638624454757e-8,
            8.681387036441865e-9, 1.377196283532058e-8, 8.992921448080973e-9,
            4.0392434474846235e-8, 3.1600448886835504e-8, 1.7426103712222823e-8,
            2.6166436896960802e-8, 2.1215370935214236e-8]
    w32t = [2.4447563022700497e-11, 6.271371558238493e-10, 6.25970201328042e-10,
            1.1870571500221265e-7, 6.302325624910133e-10, 4.3945990717418966e-11,
            0.9999997071900648, 4.664954454669818e-8, 1.0974183300160644e-10,
            1.844117024760375e-10, 4.5095359587902474e-8, 3.1024339810950786e-8,
            2.3004878405757456e-8, 2.411519915644356e-11, 2.447681491748793e-8,
            6.275711416521617e-10, 1.3328743730741433e-10, 6.307996409010861e-10,
            1.4680063961382647e-10, 4.48220316729067e-11]
    w33t = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0]
    w34t = [1.3246227905683136e-11, 2.912606564707933e-10, 2.916192555434721e-10,
            9.980620896184629e-8, 2.9323527291316115e-10, 1.9667466338176123e-11,
            0.9999997395553143, 4.543573060908011e-8, 5.1182282717830506e-11,
            8.822774943211234e-11, 4.362248281595419e-8, 2.8471498776704633e-8,
            2.0311719390079024e-8, 1.3305733413113982e-11, 2.1004583232052386e-8,
            2.9199459591554877e-10, 6.121952676631645e-11, 2.9050651836188416e-10,
            6.711445542965078e-11, 1.9882143325837112e-11]
    w35t = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0]
    w36t = [1.2557724728916152e-11, 2.769718041267331e-10, 2.773407581462474e-10,
            9.890455271830441e-8, 2.788562357986681e-10, 1.8749891201471724e-11,
            0.9999997504514712, 4.315859810355631e-8, 4.867961780107973e-11,
            8.394633368220956e-11, 4.076049769347624e-8, 2.642852909375246e-8,
            1.8980981890458434e-8, 1.2613991936471078e-11, 1.961064113176213e-8,
            2.77690482967976e-10, 5.824693952681687e-11, 2.7625654501795365e-10,
            6.386029407422011e-11, 1.8957657926482805e-11]
    w37t = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0]
    w38t = [4.910637552518462e-8, 5.24833257576739e-8, 6.776561958929454e-8,
            6.075155756492678e-8, 6.682444984068555e-6, 2.430166834945202e-8,
            0.9999924954534366, 3.602499996090785e-8, 5.57734751142365e-8,
            3.93072824995048e-8, 3.5008363314987464e-8, 2.5473165201502467e-8,
            1.9469578539831567e-8, 2.985739806406903e-8, 2.0003831062078893e-8,
            9.328183771420692e-8, 7.129519652374865e-8, 3.77666920150283e-8,
            5.809228298980444e-8, 4.6338929545222386e-8]
    w39t = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0]
    w40t = [4.5185477633595806e-8, 4.836189973291979e-8, 6.276527813067999e-8,
            5.614309351450809e-8, 6.657654187634732e-6, 2.2296226914261664e-8,
            0.9999925841099468, 3.3024194056178085e-8, 5.144593057852878e-8,
            3.6059580987408606e-8, 3.2086884409557315e-8, 2.3360554445902674e-8,
            1.7910387243157e-8, 2.735848179347183e-8, 1.8394564419415435e-8,
            8.691652185192217e-8, 6.607234122605335e-8, 3.463377306051872e-8,
            5.361816698370993e-8, 4.260250852934645e-8]

    @test isapprox(w1.weights, w1t)
    @test isapprox(w2.weights, w2t)
    @test isapprox(w3.weights, w3t, rtol = 0.01)
    @test isapprox(w4.weights, w4t)
    @test isapprox(w5.weights, w5t)
    @test isapprox(w6.weights, w6t)
    @test isapprox(w7.weights, w7t)
    @test isapprox(w8.weights, w8t)
    @test isapprox(w9.weights, w9t)
    @test isapprox(w10.weights, w10t)
    @test isapprox(w11.weights, w11t)
    @test isapprox(w12.weights, w12t)
    @test isapprox(w13.weights, w13t)
    @test isapprox(w14.weights, w14t)
    @test isapprox(w15.weights, w15t)
    @test isapprox(w16.weights, w16t)
    @test isapprox(w17.weights, w17t)
    @test isapprox(w18.weights, w18t)
    @test isapprox(w19.weights, w19t, rtol = 5.0e-6)
    @test isapprox(w20.weights, w20t)
    @test isapprox(w21.weights, w21t)
    @test isapprox(w22.weights, w22t)
    @test isapprox(w23.weights, w23t)
    @test isapprox(w24.weights, w24t, rtol = 1.0e-7)
    @test isapprox(w25.weights, w25t, rtol = 1.0e-5)
    @test isapprox(w26.weights, w26t)
    @test isapprox(w27.weights, w27t)
    @test isapprox(w28.weights, w28t, rtol = 1.0e-7)
    @test isapprox(w29.weights, w29t)
    @test isapprox(w30.weights, w30t, rtol = 5.0e-6)
    @test isapprox(w31.weights, w31t)
    @test isapprox(w32.weights, w32t)
    @test isapprox(w33.weights, w33t)
    @test isapprox(w34.weights, w34t)
    @test isapprox(w35.weights, w35t)
    @test isapprox(w36.weights, w36t)
    @test isapprox(w37.weights, w37t)
    @test isapprox(w38.weights, w38t)
    @test isapprox(w39.weights, w39t)
    @test isapprox(w40.weights, w40t)
end

@testset "Network and Dendrogram Constraints Short $(:SD)" begin
    portfolio = Portfolio(; prices = prices, solvers = solvers)
    asset_statistics!(portfolio; calc_kurt = false)
    portfolio.short = true
    portfolio.short_u = 0.22
    portfolio.long_u = 0.88
    ssl1 = portfolio.sum_short_long

    rm = :SD
    obj = :Min_Risk
    opt = OptimiseOpt(; obj = obj, rm = rm, l = l, rf = rf)
    CV = centrality_vector(portfolio;
                           cor_opt = CorOpt(; dist = DistOpt(; method = POCorDist())))
    B_1 = connection_matrix(portfolio;
                            cor_opt = CorOpt(; dist = DistOpt(; method = POCorDist())),
                            network_opt = NetworkOpt(; method = :MST))
    L_A = cluster_matrix(portfolio;
                         cor_opt = CorOpt(; dist = DistOpt(; method = POCorDist())),
                         cluster_opt = ClusterOpt(; linkage = :ward,
                                                  genfunc = GenericFunction(;
                                                                            func = dbht_d)))

    w1 = optimise!(portfolio, opt)

    portfolio.a_vec_cent = CV
    portfolio.b_cent = minimum(CV)
    w2 = optimise!(portfolio, opt)

    portfolio.network_method = :IP
    portfolio.network_ip = B_1
    w3 = optimise!(portfolio, opt)

    portfolio.network_method = :SDP
    portfolio.network_sdp = B_1
    w4 = optimise!(portfolio, opt)

    portfolio.network_method = :IP
    portfolio.network_ip = L_A
    w5 = optimise!(portfolio, opt)

    portfolio.network_method = :SDP
    portfolio.network_sdp = L_A
    w6 = optimise!(portfolio, opt)

    portfolio.a_vec_cent = []
    portfolio.b_cent = Inf

    portfolio.network_method = :IP
    portfolio.network_ip = B_1
    w7 = optimise!(portfolio, opt)

    portfolio.network_method = :SDP
    portfolio.network_sdp = B_1
    w8 = optimise!(portfolio, opt)

    portfolio.network_method = :IP
    portfolio.network_ip = L_A
    w9 = optimise!(portfolio, opt)

    portfolio.network_method = :SDP
    portfolio.network_sdp = L_A
    w10 = optimise!(portfolio, opt)

    portfolio = Portfolio(; prices = prices, solvers = solvers)
    asset_statistics!(portfolio; calc_kurt = false)
    portfolio.short = true
    portfolio.short_u = 0.32
    portfolio.long_u = 0.97
    ssl2 = portfolio.sum_short_long

    rm = :SD
    obj = :Utility
    opt = OptimiseOpt(; obj = obj, rm = rm, l = l, rf = rf)
    CV = centrality_vector(portfolio;
                           cor_opt = CorOpt(; dist = DistOpt(; method = POCorDist())),
                           network_opt = NetworkOpt(; method = :TMFG,
                                                    tmfg_genfunc = GenericFunction(;
                                                                                   func = dbht_d)))
    B_1 = connection_matrix(portfolio;
                            cor_opt = CorOpt(; dist = DistOpt(; method = POCorDist())),
                            network_opt = NetworkOpt(; method = :TMFG,
                                                     tmfg_genfunc = GenericFunction(;
                                                                                    func = dbht_d)))
    L_A = cluster_matrix(portfolio;
                         cor_opt = CorOpt(; dist = DistOpt(; method = POCorDist())),
                         cluster_opt = ClusterOpt(; linkage = :DBHT,
                                                  genfunc = GenericFunction(;
                                                                            func = dbht_d)))

    w11 = optimise!(portfolio, opt)

    portfolio.a_vec_cent = CV
    portfolio.b_cent = minimum(CV)
    w12 = optimise!(portfolio, opt)

    portfolio.network_method = :IP
    portfolio.network_ip = B_1
    w13 = optimise!(portfolio, opt)

    portfolio.network_method = :SDP
    portfolio.network_sdp = B_1
    w14 = optimise!(portfolio, opt)

    portfolio.network_method = :IP
    portfolio.network_ip = L_A
    w15 = optimise!(portfolio, opt)

    portfolio.network_method = :SDP
    portfolio.network_sdp = L_A
    w16 = optimise!(portfolio, opt)

    portfolio.a_vec_cent = []
    portfolio.b_cent = Inf

    portfolio.network_method = :IP
    portfolio.network_ip = B_1
    w17 = optimise!(portfolio, opt)

    portfolio.network_method = :SDP
    portfolio.network_sdp = B_1
    w18 = optimise!(portfolio, opt)

    portfolio.network_method = :IP
    portfolio.network_ip = L_A
    w19 = optimise!(portfolio, opt)

    portfolio.network_method = :SDP
    portfolio.network_sdp = L_A
    w20 = optimise!(portfolio, opt)

    portfolio = Portfolio(; prices = prices, solvers = solvers)
    asset_statistics!(portfolio; calc_kurt = false)
    portfolio.short = true
    portfolio.short_u = 0.27
    portfolio.long_u = 0.81
    ssl3 = portfolio.sum_short_long

    rm = :SD
    obj = :Sharpe
    opt = OptimiseOpt(; obj = obj, rm = rm, l = l, rf = rf)
    CV = centrality_vector(portfolio;
                           cor_opt = CorOpt(; dist = DistOpt(; method = POCorDist())),
                           network_opt = NetworkOpt(; method = :TMFG,
                                                    tmfg_genfunc = GenericFunction(;
                                                                                   func = dbht_d)))
    B_1 = connection_matrix(portfolio;
                            cor_opt = CorOpt(; dist = DistOpt(; method = POCorDist())),
                            network_opt = NetworkOpt(; method = :TMFG,
                                                     tmfg_genfunc = GenericFunction(;
                                                                                    func = dbht_d)))
    L_A = cluster_matrix(portfolio;
                         cor_opt = CorOpt(; dist = DistOpt(; method = POCorDist())),
                         cluster_opt = ClusterOpt(; linkage = :DBHT,
                                                  genfunc = GenericFunction(;
                                                                            func = dbht_d)))

    w21 = optimise!(portfolio, opt)

    portfolio.a_vec_cent = CV
    portfolio.b_cent = minimum(CV)
    w22 = optimise!(portfolio, opt)

    portfolio.network_method = :IP
    portfolio.network_ip = B_1
    w23 = optimise!(portfolio, opt)

    portfolio.network_method = :SDP
    portfolio.network_sdp = B_1
    w24 = optimise!(portfolio, opt)

    portfolio.network_method = :IP
    portfolio.network_ip = L_A
    w25 = optimise!(portfolio, opt)

    portfolio.network_method = :SDP
    portfolio.network_sdp = L_A
    w26 = optimise!(portfolio, opt)

    portfolio.a_vec_cent = []
    portfolio.b_cent = Inf

    portfolio.network_method = :IP
    portfolio.network_ip = B_1
    w27 = optimise!(portfolio, opt)

    portfolio.network_method = :SDP
    portfolio.network_sdp = B_1
    w28 = optimise!(portfolio, opt)

    portfolio.network_method = :IP
    portfolio.network_ip = L_A
    w29 = optimise!(portfolio, opt)

    portfolio.network_method = :SDP
    portfolio.network_sdp = L_A
    w30 = optimise!(portfolio, opt)

    portfolio = Portfolio(; prices = prices, solvers = solvers)
    asset_statistics!(portfolio; calc_kurt = false)
    portfolio.short = true
    portfolio.short_u = 0.42
    portfolio.long_u = 0.69
    ssl4 = portfolio.sum_short_long

    rm = :SD
    obj = :Max_Ret
    opt = OptimiseOpt(; obj = obj, rm = rm, l = l, rf = rf)
    CV = centrality_vector(portfolio;
                           cor_opt = CorOpt(; dist = DistOpt(; method = POCorDist())),
                           network_opt = NetworkOpt(; method = :TMFG,
                                                    tmfg_genfunc = GenericFunction(;
                                                                                   func = dbht_d)))
    B_1 = connection_matrix(portfolio;
                            cor_opt = CorOpt(; dist = DistOpt(; method = POCorDist())),
                            network_opt = NetworkOpt(; method = :TMFG,
                                                     tmfg_genfunc = GenericFunction(;
                                                                                    func = dbht_d)))
    L_A = cluster_matrix(portfolio;
                         cor_opt = CorOpt(; dist = DistOpt(; method = POCorDist())),
                         cluster_opt = ClusterOpt(; linkage = :DBHT,
                                                  genfunc = GenericFunction(;
                                                                            func = dbht_d)))

    w31 = optimise!(portfolio, opt)

    portfolio.a_vec_cent = CV
    portfolio.b_cent = minimum(CV)
    w32 = optimise!(portfolio, opt)

    portfolio.network_method = :IP
    portfolio.network_ip = B_1
    w33 = optimise!(portfolio, opt)

    portfolio.network_method = :SDP
    portfolio.network_sdp = B_1
    w34 = optimise!(portfolio, opt)

    portfolio.network_method = :IP
    portfolio.network_ip = L_A
    w35 = optimise!(portfolio, opt)

    portfolio.network_method = :SDP
    portfolio.network_sdp = L_A
    w36 = optimise!(portfolio, opt)

    portfolio.a_vec_cent = []
    portfolio.b_cent = Inf

    portfolio.network_method = :IP
    portfolio.network_ip = B_1
    w37 = optimise!(portfolio, opt)

    portfolio.network_method = :SDP
    portfolio.network_sdp = B_1
    w38 = optimise!(portfolio, opt)

    portfolio.network_method = :IP
    portfolio.network_ip = L_A
    w39 = optimise!(portfolio, opt)

    portfolio.network_method = :SDP
    portfolio.network_sdp = L_A
    w40 = optimise!(portfolio, opt)

    w1t = [0.0023732456580253273, 0.02478417662625044, 0.011667587612994237,
           0.021835859830789766, 0.008241287060153578, 0.03550224737381907,
           -0.006408569798470374, 0.09320297606692711, -0.0072069893239688695,
           0.012269888650642718, 0.18722784970990347, -0.01395210113745165,
           -0.006026909793594887, 0.0962770551826336, 0.0005110433722419729,
           0.016784621725152556, 0.009614188864666265, 0.13436280943955117,
           -0.042344878358121694, 0.08128461123785628]
    w2t = [0.0064520166328966245, 0.022489884719278264, 0.01029468309633522,
           0.020902946113760288, 0.00711671166632188, 0.03337768007882898,
           -0.006266220580382531, 0.09139178524192762, -0.02237340947148962,
           0.010954134035283628, 0.18619094237390604, -0.014102818325104587,
           -0.0055880934328339325, 0.09574848799807273, 0.0002468303788802295,
           0.01712117146324995, 0.014137258176396965, 0.13074888415899583,
           -0.01805679382768158, 0.07921391950335803]
    w3t = [-0.0, -0.0, -0.0, 0.03197389529022984, -0.0, -0.0, -0.0, 0.0, -0.0, -0.0,
           0.315129618927561, -0.0, -0.0, -0.0, -0.0, 0.024870381072438885, -0.0,
           0.17763584837581356, -0.0, 0.11039025633395685]
    w4t = [0.022589148447524018, 0.036812012135242114, 0.006833793601670234,
           0.013202059102917668, 0.005824555380015791, 0.03984269080339122,
           -0.006834168984417891, 1.9327468236390324e-5, -2.1177840385696868e-5,
           0.011500602752026409, 0.25265179781193275, -0.014411712546749403,
           -1.628444737600106e-5, 1.195721590601671e-5, 0.007515505990608158,
           0.019532808277500043, 4.833328704956932e-6, 0.16879897503212193,
           4.3131102914277385e-6, 0.09613896336083992]
    w5t = [-0.0, -0.0, -0.0, -0.0, -0.0, 0.0, 0.0, 0.0, -0.0, 0.0, 0.33999999999999986, 0.0,
           0.0, -0.0, 0.027398488452002878, -0.0, -0.0, 0.0, -0.0, 0.2926015115479973]
    w6t = [1.3733290417870086e-6, 2.6763589401080658e-6, 2.9538542317039173e-6,
           0.0005900343176583919, 1.3887640713916724e-6, 1.3598728144018154e-6,
           8.979482586127353e-7, 3.591109234764808e-6, 6.310852949253323e-8,
           6.324340049812946e-7, 0.32590968729693326, 2.4443660959673622e-6,
           1.414358800261893e-7, 1.9293289965058625e-6, 0.01628800189959108,
           0.014069026506391764, 7.282630829588722e-6, 0.05452643821874623,
           -2.6476800486090808e-6, 0.24859272489979878]
    w7t = [-0.0, -0.0, -0.0, 0.03508434384495715, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0,
           0.278751394655428, -0.0, -0.0, -0.0, -0.0, 0.019879402507731042, -0.0,
           0.19982769140819875, -0.0, 0.1264571675836849]
    w8t = [3.353883669642267e-6, 0.038122465245421025, 0.012782948631217341,
           0.024251278321675323, 0.011026952156532094, 0.04193024239831407,
           -0.007723026091322901, 6.2145857665693815e-6, -3.39569902695355e-5,
           0.012648769570655168, 0.2453658971390818, -0.014844139342231548,
           -2.384652077201505e-6, 6.1640233295210345e-6, 0.006967498867791508,
           0.01709226942230279, 1.1245332165782372e-6, 0.17380674670205037,
           6.802767311385322e-7, 0.09859090131814645]
    w9t = [0.0, -0.0, -0.0, -0.0, -0.0, -0.0, 0.0, 0.0, -0.0, 0.0, 0.406349998639137, 0.0,
           -0.0, -0.0, 0.0, 0.03772336548546998, 0.0, 0.0, 0.0, 0.21592663587539304]
    w10t = [1.645587229073337e-6, 1.0964650874070903e-6, 8.099324032340491e-7,
            4.750334438676555e-7, 5.708947010377271e-7, 5.026059745915548e-7,
            3.5554835828654246e-7, 8.862257420249835e-7, 1.6505843417208158e-7,
            2.4331260596876106e-7, 0.4039667416480195, 2.493579100224862e-7,
            7.165427252726094e-8, 2.0949825706215104e-6, 2.1348296632699007e-6,
            0.037534143273519546, 1.0029864740757473e-5, 2.983369616639739e-6,
            3.6844804532910475e-7, 0.21847443190766216]
    w11t = [8.294334951081222e-9, 3.14296793373891e-8, 4.663656382344745e-8,
            4.221790199391347e-8, 0.7041478436807951, -1.9066351113570122e-6,
            0.13029927740768363, 6.232524299474378e-9, 7.238461777482583e-8,
            1.1681167186690599e-8, 3.352968112786475e-9, -0.007526966687991981,
            -0.09199605698101808, -2.1008447864211535e-8, -0.2204746277613093,
            0.13555195839535814, 1.3053597742630056e-7, 8.789576461621622e-9,
            1.1658674838131411e-7, 2.144798199524771e-8]
    w12t = [1.3534550067046187e-8, 3.067930788232628e-8, 4.550603111363069e-8,
            3.7661602560749386e-8, 0.6948969601883085, -4.698052533521243e-7,
            0.12550825609473534, 4.6360462674977965e-9, 1.4405340220939713e-7,
            1.3352289582690065e-8, 1.8407821140087247e-9, -0.00775156914611299,
            -0.0918827580379383, -1.3964939720823826e-8, -0.22036479617897983,
            0.12904285402370896, 1.461538316465404e-6, 8.741805014843763e-9,
            0.02054974965348771, 2.5628850428915752e-8]
    w13t = [0.0, -0.0, -0.0, -0.0, 0.44999999999999996, 0.2, -0.0, -0.0, 0.0, 0.0, -0.0,
            -0.0, -0.0, 0.0, -0.0, -0.0, 0.0, 0.0, 0.0, 0.0]
    w14t = [1.484166426020456e-7, 1.4523270236063422e-7, 1.1006646933081909e-7,
            2.897643663323473e-5, 0.794501000567825, -8.010674225456363e-7,
            0.1354126203604132, 7.040831964072188e-8, 3.0328423364072293e-7,
            1.2638676111482872e-7, 4.8569914128961147e-8, -0.018505750363012833,
            -0.07247815112306952, -3.213111301030334e-8, -0.2290127400326522,
            0.018508943131871566, 1.1176691860884785e-6, 1.030383765215055e-7,
            0.021543489724163423, 2.714237582376665e-7]
    w15t = [-0.0, -0.0, -0.0, -0.0, 0.8899999999999999, -0.0, 0.0, -0.0, -0.0,
            0.08000000000000018, -0.0, -0.0, -0.0, -0.0, -0.32, 0.0, -0.0, -0.0, 0.0, 0.0]
    w16t = [7.428943590471719e-8, 1.020360637792936e-7, 1.2239435375260194e-7,
            8.498252198780311e-8, 0.8678142498830479, -1.1408369933240329e-6,
            9.473082605724163e-8, 1.525819855387539e-7, 3.1037180730958768e-6,
            8.506340308943326e-8, 2.116405594632223e-8, -2.995410269222867e-7,
            -0.05659675756397001, -8.181788397570242e-8, -0.2633999358324448,
            0.09418105858527617, 4.936279277404523e-7, 2.805843333970848e-8,
            0.007998239929023123, 3.045478917187851e-7]
    w17t = [-0.0, 0.0, 0.0, 0.0, 0.97, -0.0, 0.0, -8.664441010460355e-19, -0.0, 0.0,
            8.664441010460355e-19, -0.0, -0.0, -0.0, -0.32, 0.0, 0.0, 0.0, 0.0, -0.0]
    w18t = [1.1108248516192008e-7, 1.2362553314410222e-7, 8.291777804748123e-8,
            4.3030275502347674e-5, 0.7977223566406114, -1.2788647083388429e-6,
            0.14680472252800947, 6.847903750675513e-8, 1.7342972344650687e-7,
            9.529193262584644e-8, 4.8908117813880195e-8, -0.02098440697053509,
            -0.06870881715943246, -8.772960703816946e-8, -0.23030301823356059,
            0.025425705241483847, 3.5004123809385323e-7, 8.621439297825369e-8,
            4.7077884360697684e-7, 1.835031538884187e-7]
    w19t = [0.0, 0.0, 0.0, 0.0, 0.8097834582530379, -0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0,
            -0.0, 0.0, -0.32, 0.1602165417469622, 0.0, 0.0, 0.0, 0.0]
    w20t = [1.2021065886974759e-7, 1.6289663394445494e-7, 1.9887740957356065e-7,
            1.3330439131452247e-7, 0.8698438782269206, -2.980388732329605e-6,
            1.5243726292027774e-7, 1.245449704452252e-7, 0.0023779795995715345,
            1.2477028944350633e-7, 4.33901596696634e-8, -4.87547602528004e-7,
            -0.05371055491222226, -1.7825259085033198e-7, -0.26628278090984436,
            0.09777148988348058, 4.6723741078436704e-7, 5.796431896598628e-8,
            1.7450323664395916e-6, 3.036351473646106e-7]
    w21t = [1.3566877119847358e-9, 2.0446966457964675e-8, 0.002262512310895483,
            1.4309648003585085e-7, 0.2734983203742686, -0.10965529909211759,
            0.03892353698664605, 0.01819498928305758, 5.900712314743839e-9,
            1.0611222523062497e-8, 0.039700687003729467, -0.05489428919436992,
            -0.028441439852089117, 6.842779119067803e-10, -0.07700892954655005,
            0.10494492437890107, 0.14627523180964974, 6.830986121056796e-8,
            0.18619935145952116, 1.5367224934418446e-7]
    w22t = [2.1497119336493176e-10, 0.0088070179228199, 0.029397048718257417,
            0.019022694413270754, 0.2828940619983948, -0.12224670576806218,
            0.04353925700174413, 0.04156276108416902, 1.3097764426973018e-7,
            1.4902868835689554e-7, 0.09605820879721724, -0.04855626421146047,
            -0.029205822936867836, 1.055694863453504e-9, -0.0699910697420536,
            0.11603391472918954, 0.04641077698276719, 0.04812947034812958,
            0.07616841032531034, 0.001975959060175573]
    w23t = [1.4988232349255807e-10, 4.9467867905753305e-11, 5.4907673216223237e-11,
            -0.14999999697368063, 5.304710120142331e-11, 8.578998276132632e-11,
            3.4109029177491014e-11, 1.3429010692376603e-11, 2.3650859205921446e-10,
            0.6899999951626685, 1.4540348898984051e-11, 4.948760838562393e-11,
            7.003203779715464e-11, 1.1439808651529528e-10, 6.673875412023378e-11,
            5.641615523806144e-11, 2.821405028363235e-10, 4.077315772545405e-11,
            3.241008874911552e-10, 1.1524033839101851e-10]
    w24t = [6.120770021032183e-8, 1.6354124623024568e-6, -0.00014073619896361552,
            -1.5382765037317844e-5, 0.20801384458722627, -1.2971430048721393e-7,
            0.01815677072722214, 2.2037258513423353e-7, 6.723276350257066e-8,
            6.337618763433898e-8, 0.14038784849839608, -0.04511790824985411,
            -0.005307605149301374, 3.927932051708812e-8, -0.058853373768001094,
            0.06663138474637784, 7.950726727959057e-7, 0.12731744850449067,
            0.08892422332791153, 7.335001414362041e-7]
    w25t = [7.219639252119365e-13, 9.652815699818791e-13, 9.157634028111738e-13,
            9.8068376434984e-13, 0.568693785803048, 6.27301596906793e-13,
            1.2944896305482662e-12, 1.0447072852174023e-12, 8.039729547464033e-13,
            9.347867775055693e-13, 1.0160545637642792e-12, 6.122729728569126e-13,
            -0.10244889619698481, 7.505753909878498e-13, 4.730673990744027e-13,
            1.23523567731067e-12, 7.788087233970368e-13, 9.918816210207571e-13,
            0.07375511037889951, 8.905987022953166e-13]
    w26t = [1.4274592086142099e-8, 1.522783678899051e-8, 2.7938717235026846e-8,
            1.4561550826598941e-8, 0.2368909369265324, -0.0001506005786567205,
            -1.0081165111908498e-8, 1.7224029270673118e-5, 4.5248679067995964e-8,
            -0.00022728261938254014, 0.119886342638237, -2.4823635865134306e-7,
            -3.413454237398767e-6, -1.0451687138374718e-8, -2.8921310398193577e-6,
            0.08684122100032984, 0.0035296823817011595, 7.622590527125835e-9,
            0.09321873846044709, 1.8724204259728512e-7]
    w27t = [-0.0, -0.0, 0.0, 0.0, 0.81, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0,
            -0.0, -0.26999999999999996, -0.0, 0.0, -0.0, 0.0, -4.940238843152641e-16]
    w28t = [3.7372455923638106e-8, 4.0946527532379914e-7, 5.741510057885535e-8,
            3.4014209639844175e-8, 0.1986252065725683, -8.486609959409557e-8,
            0.023951161167138676, 1.8655603296653543e-7, 3.192609402220337e-8,
            4.19296704875058e-8, 0.18694537859607946, -0.055626176371317476,
            -2.153219858731183e-8, 1.6743729639380584e-8, -0.05376684092336848,
            0.08378904151649921, 8.526537485802294e-8, 0.1560810784728213,
            9.345153848505544e-8, 2.632283952726144e-7]
    w29t = [-0.0, -0.0, 0.0, 0.0, 0.601177343810864, -0.27, 0.0, -0.0, -0.0, 0.0, 0.0, -0.0,
            -0.0, -0.0, -0.0, 0.2088226561891345, -0.0, -0.0, -0.0, -0.0]
    w30t = [1.9089390697715466e-8, 2.229133765919515e-8, 3.281824986291854e-8,
            1.8749370001662676e-8, 0.2360852915904294, -8.445636169630471e-6,
            -1.2148952122020842e-9, 1.2285635879456866e-6, 5.003357016527931e-8,
            -0.0002758503591244658, 0.08672896660586178, -2.649201973767617e-7,
            -5.065177762345351e-7, -1.1746904762175866e-8, -9.7236488086004e-7,
            0.0852459526649671, 0.0035191828565809746, 1.6956403200522112e-8,
            0.128705177720426, 9.28197738427405e-8]
    w31t = [1.634032259124207e-9, 1.956618587379166e-9, 3.3556231312196225e-9,
            2.7876601754016495e-9, 2.943159354953637e-7, -6.461049201951197e-9,
            0.6899996361457548, -1.6841719863356268e-10, 2.3533711879678234e-9,
            3.987796673066491e-10, -3.544377042870169e-10, -4.124215986514049e-9,
            -0.41999985907917853, -1.698636143063043e-9, -8.373729471778457e-8,
            5.152026258425221e-9, 3.6144495140376544e-9, 1.3823291921040307e-10,
            2.503980335160312e-9, 1.2667652765437487e-9]
    w32t = [7.566873085534017e-9, 4.825565610664524e-9, 7.691301424793246e-9,
            5.161090844202675e-9, 0.539999362510004, -4.504534684418618e-9,
            4.1043614683303156e-7, 2.946688973782728e-10, 2.0575770043224483e-8,
            2.563567537152254e-9, -2.8742801767929926e-12, -5.70461405660151e-9,
            -0.41999973231096854, 5.357471771959318e-10, -1.7998526701142086e-7,
            1.194418269337657e-8, 2.3473551192669888e-7, 1.4874512489845639e-9,
            0.14999984679079434, 5.389582884375121e-9]
    w33t = [0.5475000004927484, 3.639348698090457e-11, 3.62971186667658e-11,
            4.835937319318533e-11, 3.698595086050992e-11, -8.344032758362837e-12,
            4.9532465938771604e-11, 4.9279769108881156e-11, -5.937463255392753e-11,
            1.528196028842464e-11, 4.938309063262805e-11, 4.6783545983278276e-11,
            -0.2775000007198744, -2.60403536325353e-11, 4.7698153813061054e-11,
            3.594328832734966e-11, -7.450626428687431e-11, 3.696492658646363e-11,
            -8.601888277065173e-11, -7.494655060185238e-12]
    w34t = [3.8167525545898195e-8, 2.410824596817611e-8, 3.866893427606442e-8,
            2.613302366792217e-8, 0.5399956074700393, -2.4227004510666e-8,
            3.1218352299989366e-6, 9.831598665296148e-10, 1.0049811904563169e-7,
            1.2810619583955812e-8, -6.236703705143353e-10, -3.2517158191717513e-8,
            -0.41999851237277896, 2.3482533868862432e-9, -1.01786381712679e-6,
            6.135268722882911e-8, 1.1740060411361075e-6, 6.935175296325583e-9,
            0.14999934500728115, 2.7280093759660284e-8]
    w35t = [0.0, 0.0, 0.0, 0.0, 0.5399999999999998, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            -0.42, 0.0, 0.0, 0.0, 0.0, 0.0, 0.14999999999999997, 0.0]
    w36t = [2.7594592956224582e-8, 1.793388362324446e-8, 2.8538303281084573e-8,
            1.9625541435892716e-8, 0.5399963579240279, -1.8381797978260847e-8,
            2.654652248368342e-6, 9.060944725051436e-10, 7.06587922753743e-8,
            9.398541984942123e-9, -3.0083799530960525e-10, -2.402919617976049e-8,
            -0.41999883525774595, 1.2963131070627324e-9, -8.123321902671323e-7,
            4.445741821085269e-8, 9.207285208125567e-7, 5.207361154302872e-9,
            0.14999951142688414, 1.995324458963913e-8]
    w37t = [2.7755575615628914e-16, 0.0, 0.0, 0.0, 0.0, 0.0, 0.69, 0.0, 0.0, 0.0,
            1.3322676295501878e-15, 0.0, -0.42, 0.0, 1.1102230246251565e-16, 0.0, 0.0, 0.0,
            0.0, 0.0]
    w38t = [3.926524175515487e-8, 4.777581996957669e-8, 8.002238487719638e-8,
            6.600835291098098e-8, 5.866790132949834e-6, -1.6231482847658006e-7,
            0.6899922382160447, -4.70762297257171e-9, 5.4861661634507464e-8,
            9.12192394367729e-9, -9.740109441939511e-9, -1.1847727451568363e-7,
            -0.41999545369718655, -4.565487118723881e-8, -2.9143969717876916e-6,
            1.2419886189767282e-7, 8.762106045147688e-8, 2.9968956877677843e-9,
            6.051390905057652e-8, 3.1596575076291346e-8]
    w39t = [2.7755575615628914e-16, 0.0, 0.0, 0.0, 0.0, 0.0, 0.69, 0.0, 0.0,
            1.3322676295501878e-15, 0.0, 0.0, -0.4200000000000013, 0.0,
            1.3877787807814457e-15, 0.0, 0.0, 0.0, 0.0, 0.0]
    w40t = [1.7620733723874953e-8, 2.1537177684751873e-8, 3.63370440124854e-8,
            2.998215973623228e-8, 3.32010387480701e-6, -7.900947951268348e-8,
            0.6899957728941339, -3.1281315063185747e-9, 2.479895828654508e-8,
            3.6016546702866596e-9, -5.270156878103679e-9, -5.739774826048532e-8,
            -0.41999750775768, -2.2282474068662558e-8, -1.6868280378868658e-6,
            5.373829891771601e-8, 3.9320350356177794e-8, 5.414735910641679e-10,
            2.7056343354506067e-8, 1.4141504975069959e-8]

    @test isapprox(w1.weights, w1t)
    @test isapprox(w2.weights, w2t)
    @test isapprox(w3.weights, w3t, rtol = 0.01)
    @test isapprox(w4.weights, w4t)
    @test isapprox(w5.weights, w5t, rtol = 0.01)
    @test isapprox(w6.weights, w6t)
    @test isapprox(w7.weights, w7t, rtol = 0.05)
    @test isapprox(w8.weights, w8t)
    @test isapprox(w9.weights, w9t, rtol = 1.0e-7)
    @test isapprox(w10.weights, w10t)

    @test isapprox(sum(w1.weights), ssl1)
    @test isapprox(sum(w2.weights), ssl1)
    @test isapprox(sum(w3.weights), ssl1)
    @test isapprox(sum(w4.weights), ssl1)
    @test isapprox(sum(w5.weights), ssl1)
    @test isapprox(sum(w6.weights), ssl1)
    @test isapprox(sum(w7.weights), ssl1)
    @test isapprox(sum(w8.weights), ssl1)
    @test isapprox(sum(w9.weights), ssl1)
    @test isapprox(sum(w10.weights), ssl1)

    @test isapprox(w11.weights, w11t)
    @test isapprox(w12.weights, w12t)
    @test isapprox(w13.weights, w13t)
    @test isapprox(w14.weights, w14t)
    @test isapprox(w15.weights, w15t)
    @test isapprox(w16.weights, w16t)
    @test isapprox(w17.weights, w17t)
    @test isapprox(w18.weights, w18t)
    @test isapprox(w19.weights, w19t, rtol = 1.0e-3)
    @test isapprox(w20.weights, w20t)

    @test isapprox(sum(w11.weights), ssl2)
    @test isapprox(sum(w12.weights), ssl2)
    @test isapprox(sum(w13.weights), ssl2)
    @test isapprox(sum(w14.weights), ssl2)
    @test isapprox(sum(w15.weights), ssl2)
    @test isapprox(sum(w16.weights), ssl2)
    @test isapprox(sum(w17.weights), ssl2)
    @test isapprox(sum(w18.weights), ssl2)
    @test isapprox(sum(w19.weights), ssl2)
    @test isapprox(sum(w20.weights), ssl2)

    @test isapprox(w21.weights, w21t)
    @test isapprox(w22.weights, w22t)
    @test isapprox(w23.weights, w23t)
    @test isapprox(w24.weights, w24t, rtol = 1.0e-6)
    @test isapprox(w25.weights, w25t, rtol = 0.001)
    @test isapprox(w26.weights, w26t)
    @test isapprox(w27.weights, w27t)
    @test isapprox(w28.weights, w28t)
    @test isapprox(w29.weights, w29t, rtol = 0.001)
    @test isapprox(w30.weights, w30t)

    @test isapprox(sum(w21.weights), ssl3)
    @test isapprox(sum(w22.weights), ssl3)
    @test isapprox(sum(w23.weights), ssl3)
    @test isapprox(sum(w24.weights), ssl3)
    @test isapprox(sum(w25.weights), ssl3)
    @test isapprox(sum(w26.weights), ssl3)
    @test isapprox(sum(w27.weights), ssl3)
    @test isapprox(sum(w28.weights), ssl3)
    @test isapprox(sum(w29.weights), ssl3)
    @test isapprox(sum(w30.weights), ssl3)

    @test isapprox(w31.weights, w31t)
    @test isapprox(w32.weights, w32t)
    @test isapprox(w33.weights, w33t)
    @test isapprox(w34.weights, w34t)
    @test isapprox(w35.weights, w35t)
    @test isapprox(w36.weights, w36t)
    @test isapprox(w37.weights, w37t)
    @test isapprox(w38.weights, w38t)
    @test isapprox(w39.weights, w39t)
    @test isapprox(w40.weights, w40t)

    @test isapprox(sum(w31.weights), ssl4)
    @test isapprox(sum(w32.weights), ssl4)
    @test isapprox(sum(w33.weights), ssl4)
    @test isapprox(sum(w34.weights), ssl4)
    @test isapprox(sum(w35.weights), ssl4)
    @test isapprox(sum(w36.weights), ssl4)
    @test isapprox(sum(w37.weights), ssl4)
    @test isapprox(sum(w38.weights), ssl4)
    @test isapprox(sum(w39.weights), ssl4)
    @test isapprox(sum(w40.weights), ssl4)
end

@testset "Network and Dendrogram Upper Dev Constraints" begin
    portfolio = Portfolio(; prices = prices, solvers = solvers)
    asset_statistics!(portfolio; calc_kurt = false)

    rm = :SD
    L_A = cluster_matrix(portfolio; cor_opt = CorOpt(;),
                         cluster_opt = ClusterOpt(; linkage = :ward))

    portfolio.network_method = :None
    portfolio.network_ip = L_A
    portfolio.network_sdp = L_A
    opt = OptimiseOpt(; obj = :Sharpe, rm = rm, l = l, rf = rf)
    w1 = optimise!(portfolio, opt)
    r1 = calc_risk(portfolio; rm = rm)

    portfolio.sd_u = r1
    portfolio.network_method = :IP
    opt.obj = :Min_Risk
    w2 = optimise!(portfolio, opt)
    r2 = calc_risk(portfolio; rm = rm)
    opt.obj = :Utility
    w3 = optimise!(portfolio, opt)
    r3 = calc_risk(portfolio; rm = rm)
    opt.obj = :Sharpe
    w4 = optimise!(portfolio, opt)
    r4 = calc_risk(portfolio; rm = rm)
    opt.obj = :Max_Ret
    w5 = optimise!(portfolio, opt)
    r5 = calc_risk(portfolio; rm = rm)

    portfolio.network_method = :SDP
    opt.obj = :Min_Risk
    w6 = optimise!(portfolio, opt)
    r6 = calc_risk(portfolio; rm = rm)
    opt.obj = :Utility
    w7 = optimise!(portfolio, opt)
    r7 = calc_risk(portfolio; rm = rm)
    opt.obj = :Sharpe
    w8 = optimise!(portfolio, opt)
    r8 = calc_risk(portfolio; rm = rm)
    opt.obj = :Max_Ret
    w9 = optimise!(portfolio, opt)
    r9 = calc_risk(portfolio; rm = rm)

    @test r2 <= r1 + sqrt(eps())
    @test r3 <= r1 + length(w5.weights) * sqrt(eps())
    @test r4 <= r1 + sqrt(eps())
    @test r5 <= r1 + length(w5.weights) * sqrt(eps())

    @test r6 <= r1 + sqrt(eps())
    @test r7 <= r1 + sqrt(eps())
    @test r8 <= r1 + sqrt(eps())
    @test r9 <= r1 + length(w5.weights) * sqrt(eps())
end

@testset "Turnover" begin
    portfolio = Portfolio(; prices = prices,
                          solvers = OrderedDict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                                  :params => Dict("verbose" => false,
                                                                                  "max_step_fraction" => 0.75))))
    asset_statistics!(portfolio)

    rm = :SD
    opt = OptimiseOpt(; rm = rm, l = l, rf = rf)
    opt.obj = :Sharpe
    w1 = optimise!(portfolio, opt)
    to1 = 0.05
    tow1 = copy(w1.weights)
    portfolio.turnover = to1
    portfolio.turnover_weights = tow1

    opt.obj = :Min_Risk
    w2 = optimise!(portfolio, opt)

    portfolio.turnover = Inf
    opt.obj = :Min_Risk
    w3 = optimise!(portfolio, opt)
    to2 = 0.031
    tow2 = copy(w3.weights)
    portfolio.turnover = to2
    portfolio.turnover_weights = tow2

    opt.obj = :Sharpe
    w4 = optimise!(portfolio, opt)

    portfolio.turnover = Inf
    opt.obj = :Sharpe
    w5 = optimise!(portfolio, opt)
    to3 = range(; start = 0.001, stop = 0.003, length = 20)
    tow3 = copy(w5.weights)
    portfolio.turnover = to3
    portfolio.turnover_weights = tow3

    opt.obj = :Min_Risk
    w6 = optimise!(portfolio, opt)

    portfolio.turnover = Inf
    opt.obj = :Min_Risk
    w7 = optimise!(portfolio, opt)
    to4 = [fill(0.01, 5); fill(0.03, 5); fill(0.07, 5); fill(0.04, 5)]
    tow4 = copy(w7.weights)
    portfolio.turnover = to4
    portfolio.turnover_weights = tow4

    opt.obj = :Sharpe
    w8 = optimise!(portfolio, opt)

    @test all(abs.(w2.weights - tow1) .<= to1)
    @test all(abs.(w4.weights - tow2) .<= to2)
    @test all(abs.(w6.weights - tow3) .<= to3)
    @test all(abs.(w8.weights - tow4) .<= to4)
end

@testset "Tracking Error" begin
    portfolio = Portfolio(; prices = prices,
                          solvers = OrderedDict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                                  :params => Dict("verbose" => false,
                                                                                  "max_step_fraction" => 0.75))))
    asset_statistics!(portfolio)

    T = size(portfolio.returns, 1)
    rm = :SD

    opt = OptimiseOpt(; rm = rm, l = l, rf = rf)

    opt.obj = :Sharpe
    w1 = optimise!(portfolio, opt)
    portfolio.kind_tracking_err = :Weights
    te1 = 0.0005
    tw1 = copy(w1.weights)
    portfolio.tracking_err = te1
    portfolio.tracking_err_weights = tw1

    opt.obj = :Min_Risk
    w2 = optimise!(portfolio, opt)

    portfolio.tracking_err = Inf
    opt.obj = :Min_Risk
    w3 = optimise!(portfolio, opt)
    portfolio.kind_tracking_err = :Weights
    te2 = 0.0003
    tw2 = copy(w3.weights)
    portfolio.tracking_err = te2
    portfolio.tracking_err_weights = tw2

    opt.obj = :Sharpe
    w4 = optimise!(portfolio, opt)

    portfolio.tracking_err = Inf
    opt.obj = :Sharpe
    w5 = optimise!(portfolio, opt)
    portfolio.kind_tracking_err = :Returns
    te3 = 0.007
    tw3 = vec(mean(portfolio.returns; dims = 2))
    portfolio.tracking_err = te3
    portfolio.tracking_err_returns = tw3

    opt.obj = :Min_Risk
    w6 = optimise!(portfolio, opt)

    portfolio.tracking_err = Inf
    opt.obj = :Min_Risk
    w7 = optimise!(portfolio, opt)
    portfolio.kind_tracking_err = :Returns
    te4 = 0.0024
    tw4 = vec(mean(portfolio.returns; dims = 2))
    portfolio.tracking_err = te4
    portfolio.tracking_err_returns = tw4

    opt.obj = :Sharpe
    w8 = optimise!(portfolio, opt)

    @test norm(portfolio.returns * (w2.weights - tw1), 2) / sqrt(T - 1) <= te1
    @test norm(portfolio.returns * (w4.weights - tw2), 2) / sqrt(T - 1) <= te2

    @test norm(portfolio.returns * w6.weights - tw3, 2) / sqrt(T - 1) <= te3
    @test norm(portfolio.returns * w8.weights - tw4, 2) / sqrt(T - 1) <= te4
end

@testset "Minimum Number of Effective Assets" begin
    portfolio = Portfolio(; prices = prices,
                          solvers = OrderedDict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                                  :params => Dict("verbose" => false,
                                                                                  "max_step_fraction" => 0.75))))
    asset_statistics!(portfolio)

    rm = :SD
    opt = OptimiseOpt(; obj = :Min_Risk, rm = rm, l = l, rf = rf)

    w1 = optimise!(portfolio, opt)
    portfolio.num_assets_l = 8
    w2 = optimise!(portfolio, opt)

    portfolio.num_assets_l = 0
    opt.obj = :Sharpe
    w3 = optimise!(portfolio, opt)
    portfolio.num_assets_l = 6
    w4 = optimise!(portfolio, opt)

    @test count(w2.weights .>= 2e-2) >= 8
    @test count(w2.weights .>= 2e-2) >= count(w1.weights .>= 2e-2)

    @test count(w4.weights .>= 2e-2) >= 6
    @test count(w4.weights .>= 2e-2) >= count(w3.weights .>= 2e-2)
end

@testset "Linear Constraints" begin
    portfolio = Portfolio(; prices = prices,
                          solvers = OrderedDict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                                  :params => Dict("verbose" => false,
                                                                                  "max_step_fraction" => 0.75))))
    asset_statistics!(portfolio)

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

    rm = :SD

    opt = OptimiseOpt(; obj = :Min_Risk, rm = rm, l = l, rf = rf)
    w1 = optimise!(portfolio, opt)

    @test all(w1.weights[asset_sets.G2DBHT .== 2] .>= 0.03)
    @test all(w1.weights[asset_sets.G2DBHT .== 3] .<= 0.2)
    @test all(w1.weights[w1.tickers .== "AAPL"] .>= 0.032)
    @test sum(w1.weights[asset_sets.G2ward .== 2]) <=
          w1.weights[w1.tickers .== "MA"][1] * 2.2
    @test w1.weights[w1.tickers .== "MA"][1] * 5 >= sum(w1.weights[asset_sets.G2ward .== 3])

    opt.obj = :Sharpe
    w2 = optimise!(portfolio, opt)

    @test all(w2.weights[asset_sets.G2DBHT .== 2] .>= 0.03)
    @test all(w2.weights[asset_sets.G2DBHT .== 3] .<= 0.2)
    @test all(w2.weights[w2.tickers .== "AAPL"] .>= 0.032)
    @test sum(w2.weights[asset_sets.G2ward .== 2]) <=
          w2.weights[w2.tickers .== "MA"][1] * 2.2
    @test w2.weights[w2.tickers .== "MA"][1] * 5 >= sum(w2.weights[asset_sets.G2ward .== 3])
end

@testset "Max Number of Assets" begin
    portfolio = Portfolio(; prices = prices, solvers = solvers)
    asset_statistics!(portfolio; calc_kurt = false)

    rm = :SSD
    opt = OptimiseOpt(; obj = :Min_Risk, rm = rm, l = l, rf = rf)
    w1 = optimise!(portfolio, opt)
    portfolio.num_assets_u = 5
    w2 = optimise!(portfolio, opt)
    sort!(w1, :weights; rev = true)
    sort!(w2, :weights; rev = true)

    portfolio.num_assets_u = 0
    opt.obj = :Sharpe
    w3 = optimise!(portfolio, opt)
    portfolio.num_assets_u = 4
    w4 = optimise!(portfolio, opt)
    sort!(w3, :weights; rev = true)
    sort!(w4, :weights; rev = true)

    portfolio = Portfolio(; prices = prices, solvers = solvers, short = true)
    asset_statistics!(portfolio; calc_kurt = false)

    rm = :CVaR
    opt.obj = :Min_Risk
    opt.rm = rm
    w5 = optimise!(portfolio, opt)
    portfolio.num_assets_u = 7
    w6 = optimise!(portfolio, opt)
    sort!(w5, :weights; rev = true)
    sort!(w6, :weights; rev = true)

    portfolio.num_assets_u = 0
    opt.obj = :Sharpe
    w7 = optimise!(portfolio, opt)
    portfolio.num_assets_u = 5
    w8 = optimise!(portfolio, opt)
    sort!(w7, :weights; rev = true)
    sort!(w8, :weights; rev = true)

    @test isempty(setdiff(w1.tickers[1:5], w2.tickers[1:5]))
    @test isempty(setdiff(w3.tickers[1:4], w4.tickers[1:4]))

    @test isempty(setdiff(w5.tickers[1:6], w6.tickers[1:6]))
    @test isempty(setdiff(w5.tickers[end], w6.tickers[end]))

    @test isempty(setdiff(w7.tickers[1:3], w8.tickers[1:3]))
    @test isempty(setdiff(w7.tickers[(end - 1):end], w8.tickers[(end - 1):end]))
end
