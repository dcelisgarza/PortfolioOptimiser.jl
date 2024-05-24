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
    w5t = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.025097613514509735, 0.21816860550265443, 0.0,
           0.0, 0.7567337809828358, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    w6t = [2.3619106009815383e-11, 8.067170593344577e-12, 4.423147858553386e-14,
           0.057396447628007746, 1.0340165057802283e-9, 2.3043416936860277e-11,
           0.06229570827632909, 0.2777483134388886, 2.463565829761638e-11,
           2.1453101919945665e-11, 0.4524895944858641, 0.1500699348762852,
           1.829573487583943e-11, 2.3847410229820496e-11, 1.8578953098309855e-12,
           2.96210027764348e-11, 2.4914663824441573e-11, 1.362590109572021e-11,
           2.5070091894760514e-11, 2.2513435462821127e-11]
    w7t = [0.0, 0.0, 5.920765310267057e-17, 0.0, 0.37297150326295364, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.6270284967370462, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    w8t = [5.333278572147394e-11, 0.05638785424214076, 1.8014070508602034e-10,
           5.2482313928648487e-11, 0.0820640582931353, 1.7756938142535277e-11,
           1.9026995337541293e-12, 0.13507762926843872, 1.339080959908328e-10,
           3.787326943211817e-11, 0.3175440779328707, 0.020298213899402687,
           4.163925315365446e-11, 1.174229289652941e-10, 4.784818255603212e-11,
           0.12477101930299711, 0.07961825077575638, 0.03009048638710793,
           8.237676509605876e-11, 0.15414840913146652]
    w9t = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.42755737470037836, 0.0, 0.0,
           0.0, 0.0, 0.09032779150803771, 0.2604585803196508, 0.0, 0.0, 0.22165625347193316]
    w10t = [9.017441384157346e-11, 2.088379275484815e-10, 1.8968073569426466e-10,
            3.577739521553439e-11, 0.009299135419976502, 2.452405443850451e-11,
            8.52706779843429e-12, 0.08339117958209727, 4.348088362371337e-11,
            5.6232077774939575e-11, 0.37536836807170226, 3.298737906237926e-10,
            3.802952991600171e-11, 1.0621836612448752e-10, 3.898555752314655e-11,
            0.12493267988536692, 0.24826840670117062, 0.01928380593452991,
            1.2946554242917746e-10, 0.1394564231053492]
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
    w15t = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.025097613514509676, 0.2181686055026544, 0.0,
            0.0, 0.756733780982836, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    w16t = [1.3069722357580953e-10, 7.768635026233496e-11, 4.778050838113835e-11,
            0.05227246158487569, 5.800628022441388e-9, 1.2858865985884664e-10,
            0.054706131372740004, 0.26790346508301155, 1.347791839833452e-10,
            1.2219830230156443e-10, 0.489560802764437, 0.13555713189363028,
            3.8388435236906645e-11, 1.3169592822978464e-10, 4.3639226929235135e-11,
            1.497031536512631e-10, 1.359056127346353e-10, 9.6636856476867e-11,
            1.3654802224540096e-10, 1.264299586842682e-10]
    w17t = [0.0, 0.0, 0.0, 0.0, 0.37297150326295375, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.6270284967370463, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    w18t = [5.045783366213136e-11, 0.027087558541133032, 1.2490463860059793e-10,
            2.3179354430033212e-11, 0.031974200738443176, 1.2232533405103717e-11,
            1.7470181805933803e-11, 0.09327461173474633, 1.3933388322477516e-10,
            1.7771948756462316e-11, 0.36898570967277466, 0.007183433710346759,
            7.127037788297474e-12, 2.2503960566329528e-10, 1.883751349496794e-11,
            0.11991644921560822, 0.19463651907696827, 0.029194687029793633,
            1.4014767002437255e-10, 0.12774682950368368]
    w19t = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.1319557613440377e-17, 0.0, 0.0,
            0.42755737470037847, 0.0, 0.0, 0.0, 0.0, 0.09032779150803776,
            0.2604585803196507, 0.0, 0.0, 0.22165625347193313]
    w20t = [7.002850555311706e-11, 4.0075871231773496e-10, 1.3398586493238303e-10,
            2.651290498759254e-11, 0.00909315736569001, 9.464343969310413e-12,
            2.651848100633e-11, 0.08073698516358772, 4.7856495632505926e-11,
            2.3824457826965448e-11, 0.38481073542095723, 1.8747421374644458e-10,
            2.31303007584795e-11, 1.5897491275302298e-10, 6.301389671216925e-11,
            0.11782349280106136, 0.25264750917470596, 0.006736429552082139,
            1.8740667255688688e-10, 0.14815168916296595]
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
    w25t = [0.0, 0.0, 0.0, 0.05826291361186471, 2.924313212638887e-18, 0.0,
            0.2139337899569756, 0.0, 0.0, 0.0, 0.7278032964311597, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0]
    w26t = [3.914041661350895e-10, 1.0440931371700415e-10, 3.264463277917618e-11,
            0.1667587533566986, 1.725739915555789e-8, 3.820516009534159e-10,
            0.29703706361374943, 4.189953148342988e-8, 4.1406809383116163e-10,
            3.350766056715591e-10, 0.3854434089460482, 0.1507607112555558,
            2.667191147764504e-11, 3.972605988399306e-10, 2.3012435863041758e-10,
            1.788330370642148e-11, 3.9267905882420467e-10, 1.8432284446990112e-10,
            4.034153263478759e-10, 3.5900544083880114e-10]
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
    w30t = [3.104859263283459e-9, 5.050352501105057e-9, 0.02071817208809596,
            1.1375971992012339e-9, 0.2689527788782444, 3.863881258316245e-11,
            5.758719238069159e-9, 0.14545053207011435, 8.230630297276717e-10,
            9.456057052230571e-10, 0.19916721421639413, 4.569420935877387e-11,
            3.7302034354419284e-10, 1.9844257962369145e-9, 2.5077748192276306e-10,
            0.2555182268699173, 0.11019303322639727, 2.522962579725147e-9,
            5.654682470867937e-9, 1.4960437804888155e-8]
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
    w36t = [9.979213806048484e-12, 1.0757807645909353e-10, 1.2496589288244604e-10,
            0.25481893654259824, 3.0255861277061964e-10, 2.6014208790094435e-12,
            0.2643025023935577, 1.7678179390633747e-8, 2.2936027821783945e-11,
            2.1434806104161215e-11, 0.24846377546128248, 1.7503560233365565e-8,
            0.23241474102146278, 1.1901247547186953e-11, 8.511128014474434e-9,
            1.2594927864345297e-10, 2.6328036540339196e-11, 1.022354956256432e-10,
            2.8758576589219076e-11, 1.0045201911579455e-12]
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
    w40t = [1.5817742449215597e-9, 1.9977695466256318e-9, 2.108622145985894e-9,
            1.8670653083290913e-9, 0.25381137711209945, 1.596688743085009e-9,
            0.2543089467747431, 1.1759327685411096e-9, 7.712572788247258e-8,
            2.6426798922450897e-9, 5.661446465420896e-9, 1.7066726136326424e-9,
            8.404287803833489e-10, 2.4044925788096207e-9, 1.155147011245612e-9,
            0.24740169810315635, 4.301014276350286e-8, 1.2543258908823085e-9,
            0.24447783039500992, 1.4860744428790223e-9]
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
    w45t = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.025097613514509735, 0.21816860550265443, 0.0,
            0.0, 0.7567337809828358, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    w46t = [8.066176662948276e-11, 7.846726522211354e-11, 8.830767008646231e-11,
            0.14400449191792639, 1.6067698894706017e-9, 7.239080424857936e-11,
            0.16096092858719221, 0.20357959165707923, 8.874424293796657e-11,
            6.796790608722297e-11, 0.22526752432228056, 0.2311571938945118,
            0.035030265303532857, 8.242544866913844e-11, 3.036068731700161e-11,
            1.8483786806634138e-9, 9.041213795783903e-11, 2.330828731505939e-11,
            9.080470343979167e-11, 6.847755397204912e-11]
    w47t = [0.0, 0.0, 5.920765310267057e-17, 0.0, 0.37297150326295364, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.6270284967370462, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    w48t = [6.544970352321704e-11, 0.10178810405224831, 0.004112920935146684,
            0.0031639790031650875, 0.1530529999471956, 2.051853078939224e-11,
            0.014700198697066904, 0.1902839333114586, 2.706174002994138e-11,
            0.006575121290324204, 0.21040590074211363, 0.004140118886384232,
            2.3663886591314264e-11, 2.8342248346527903e-11, 0.02751075137205449,
            0.11821237566813832, 1.1286146419967076e-11, 1.5812546032508732e-10,
            6.128688173726229e-12, 0.16605359575412765]
    w49t = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.42755737470037836, 0.0, 0.0,
            0.0, 0.0, 0.09032779150803771, 0.2604585803196508, 0.0, 0.0,
            0.22165625347193316]
    w50t = [3.4799103456914945e-10, 3.832365350339127e-10, 4.771366849426316e-10,
            8.413787205893765e-11, 0.035311043145816734, 8.207488951237227e-11,
            5.4414076576700516e-11, 0.12708392636824645, 6.68048499888268e-11,
            0.03191755066470304, 0.29800203290856325, 8.412019286218417e-11,
            0.007423632662406353, 1.0744827336499246e-10, 6.859821213836867e-11,
            0.15243449812562618, 0.24469036498291835, 2.600053655860802e-10,
            1.172317238562962e-10, 0.10313694900852015]
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
    w55t = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.025097613514509676, 0.2181686055026544, 0.0,
            0.0, 0.756733780982836, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    w56t = [1.5319633778915382e-10, 1.3844162024732992e-10, 9.556606857917e-11,
            0.11834011631260989, 4.250816973425614e-9, 1.4460555001627373e-10,
            0.1247479289356113, 0.24173757178606575, 1.6504171762051182e-10,
            1.316610847086899e-10, 0.2831378183828014, 0.22309878326858223,
            0.008937773184009717, 1.5512333643344602e-10, 5.745269587498984e-11,
            2.355417652769651e-9, 1.6720022453072754e-10, 9.061492730943852e-12,
            1.6730031163976047e-10, 1.394346153617014e-10]
    w57t = [0.0, 0.0, 0.0, 0.0, 0.37297150326295375, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.6270284967370463, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    w58t = [1.0675624983732587e-10, 0.10287620722699994, 0.007419930667590516,
            0.005575733023148208, 0.1403841692167293, 3.199121980349859e-11,
            4.6306802673266263e-10, 0.18278277233059223, 8.631146826284374e-11,
            4.630387025867777e-10, 0.23500367383413973, 0.007030151281493114,
            5.022902604510499e-11, 6.927670538297251e-11, 0.006391407528962509,
            0.12391311923336361, 4.114531115043418e-11, 0.01987312469120852,
            1.9281611083696664e-11, 0.168749709634674]
    w59t = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.1319557613440377e-17, 0.0, 0.0,
            0.42755737470037847, 0.0, 0.0, 0.0, 0.0, 0.09032779150803776,
            0.2604585803196507, 0.0, 0.0, 0.22165625347193313]
    w60t = [1.042186474854199e-10, 1.3485254620491336e-10, 1.8303991954359516e-10,
            3.914802519406392e-11, 0.032009499095408796, 2.933009110105721e-11,
            1.7975745557757888e-11, 0.11600178830341451, 3.1938830869442e-11,
            0.0021331662793370254, 0.3186868005600354, 1.3203064509914337e-10,
            0.000664183488778848, 5.108672640287638e-11, 2.686899739811538e-11,
            0.15685331808087816, 0.24691361133786774, 3.0238587900862603e-10,
            5.831035450226437e-11, 0.1267376317430931]
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
    w65t = [0.0, 0.0, 0.0, 0.05826291361186471, 2.924313212638887e-18, 0.0,
            0.2139337899569756, 0.0, 0.0, 0.0, 0.7278032964311597, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0]
    w66t = [6.887066199553351e-10, 1.3942219296218284e-10, 5.617973600675453e-11,
            0.28887871559482536, 2.8511681818350362e-8, 6.727791125001964e-10,
            0.4576951603196201, 4.805197643914377e-9, 6.712327218039465e-10,
            5.972884880908119e-10, 0.1560728864837939, 0.09735319790583159,
            1.0817057217327353e-11, 7.029527191800699e-10, 4.224100615252325e-10,
            1.8682750141714914e-10, 6.319770730638095e-10, 3.165175267171814e-10,
            6.608697025929005e-10, 6.210689882882156e-10]
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
    w70t = [9.245516220601118e-10, 5.186782148635348e-9, 3.4039386184143264e-9,
            6.646594911931452e-10, 0.3308627214855685, 8.939476826782104e-11,
            0.03086646430540743, 1.1623232349494082e-9, 1.8599693781754214e-9,
            5.926674468334273e-10, 0.17144513835131672, 2.5251341793341505e-10,
            2.6526771610019613e-10, 1.5828200789884177e-9, 7.165889559341803e-11,
            0.2798483559644734, 0.1833483301534829, 6.156055011640072e-10,
            0.0036289718718038083, 1.1957949612101392e-9]
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
    w76t = [2.808953057569564e-11, 2.5368126242568577e-10, 2.956740604573578e-10,
            0.25047657021708464, 4.935701606610306e-10, 4.171446712890793e-12,
            0.2514337884599485, 6.99770525499966e-8, 6.919181492056242e-11,
            8.478928096414263e-11, 0.24984650892745455, 5.963818484767332e-8,
            0.2482429677886559, 3.629199687830896e-11, 3.240177346545273e-8,
            8.994660792508148e-10, 7.844041301241544e-11, 2.520509892668899e-10,
            8.474789959378888e-11, 9.68046702397924e-12]
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
    w80t = [6.599607268267841e-9, 8.195744953282097e-9, 8.430704428049894e-9,
            8.083985319376354e-9, 0.2503810770458226, 6.149491257259549e-9,
            0.25043074292510137, 5.188426666896991e-9, 3.1598863643306385e-7,
            1.0016342060996323e-8, 1.5929624978851616e-8, 6.427401837850428e-9,
            3.357288470442662e-9, 9.070608332101324e-9, 5.0995642204099986e-9,
            0.24974023357410505, 1.7924824571124688e-7, 5.404587550281227e-9,
            0.24944734715916728, 6.105544260199474e-9]

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
    @test isapprox(w46.weights, w46t)
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
    @test isapprox(w60.weights, w60t)

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

    w1t = [0.0, -0.003392742308847771, 0.0, -0.001704731699272089, 0.026209661740928515,
           -0.04895127670745864, -0.028310530666442145, 0.07970846611579031,
           -0.006091550656354434, -0.01772949018065377, 0.3496375096278862,
           -0.009810028561543104, 0.0066162066090907625, 0.0, -0.01400964921942805,
           0.09738313723086277, 0.16435744872368885, 0.016435689355704775,
           0.10356751838407473, 0.1560843622119731]
    w2t = [-0.09486167717442828, 0.04019416692914493, 0.03757722317904152, 0.0,
           0.12801741187716337, 0.0, -0.019980959763450823, 0.13355629824656448,
           0.019271038397774748, 0.0, 0.28797176886025694, 1.0099015437560113e-17,
           0.008845283123909222, 0.0, -0.01515736306212094, 0.10734222426525301, 0.0,
           0.06147426694120132, 2.550023892786934e-17, 0.17575031817969047]
    w3t = [-0.0, -0.0, 0.0, 0.0, 0.2542383528166067, -0.10786045982487949, -0.0,
           0.32907983060645285, -0.0, -0.0, -0.0, -0.0, 0.0, -0.0, 0.029160101685153927,
           0.13000000000000084, -0.0, 0.0, -0.0, 0.23538217471666467]
    w4t = [-0.01932528177619994, 0.04728373641008076, 0.08111180935451298,
           0.006702806869866796, 0.11062369341271694, 0.0022197632312653077,
           -0.025844621007448453, 0.08086336920504056, 0.027815772993299463,
           -0.006292839135299517, 0.2737936079939405, -0.0027801322473058913,
           0.009407464924822933, 0.005426314499081213, -0.018351474396010235,
           0.0878112760881544, -1.774073039859027e-12, 0.06594215310933166,
           -0.03538290678964342, 0.17897548726156803]
    w5t = [-0.0, -0.0, -0.0, -0.0, 0.37577751021966854, -0.0, -0.06097567215251032,
           0.42519816193284143, -0.0, -0.0, 0.0, 0.0, -0.0, -0.0, 0.0, 0.13, -0.0, 0.0,
           -0.0, 0.0]
    w6t = [-4.499564055721756e-11, 0.030677855781885515, 0.02293345031363999,
           7.607951788007616e-11, 0.14056975471684863, 8.303248098948132e-12,
           -0.013472790425564199, 0.1813898557718623, 2.219218031085124e-11,
           2.70699503592878e-11, 0.17650253193647833, 0.035113112906152634,
           0.010757646659120794, 4.180375388190422e-12, -0.004890040907711415,
           0.10239680091612628, -4.399904468481569e-12, 0.048918684553956704,
           -0.024816555455095007, 0.16391969314386942]
    w7t = [0.0, -0.0, 0.0, 0.0, 0.11199761883276771, -0.0, -0.0, -0.0, 0.0, -0.0,
           0.3750880542218099, -0.0, -0.0, -0.0, -0.0, 0.14127637498099654, 0.0,
           0.09588426469924055, -0.0, 0.1457536872651849]
    w8t = [7.952973670583862e-10, -0.0032720635053855407, 0.05865310226246447,
           -0.010413877563161094, 0.07503911419346931, -0.005393437034375811,
           -0.03820502798683336, 0.05334438344736868, -0.02198920577315415,
           -0.017281798058430213, 0.3051995774693994, -0.008898292571920983,
           0.007234081372159881, 0.017965395651689326, -0.024546284632603232,
           0.10516121679282586, 0.07890511508125506, 0.04604809897154898,
           0.09382272741870916, 0.1586271736696768]
    w9t = [-0.0, -0.0, 0.0, -0.0, -0.0, -0.0, -0.03380540048487986, 0.0, -0.0, -0.0,
           0.44152390203521386, -0.0, -0.0, -0.0, -1.4672583285548087e-16,
           0.06721017342340446, 0.3950713250262614, 0.0, -0.0, 0.0]
    w10t = [6.089397389448996e-11, 3.5227528823980896e-11, 4.5084676704830544e-10,
            3.637074282327953e-11, 0.02835782061401647, 1.2336223195123247e-11,
            -0.019654575305172915, 0.0902574998428759, 2.968080327244569e-11,
            8.165633367768792e-12, 0.32504197158931264, 0.0016944403868190952,
            0.0034463276065987633, 5.1397945656816304e-11, -0.005504064448278817,
            0.10784797431586648, 0.2089457756148421, 4.6179782520234376e-10,
            0.01187259732483055, 0.117694231311572]
    w11t = [0.0, -0.003017422928334872, 0.0, -0.0005578189854036299, 0.014262826621635725,
            -0.026715129365893816, -0.01629323231811754, 0.04647310508003174,
            -0.000785693717675302, -0.009357177603515936, 0.20100258315237743,
            -0.00550122561735638, 0.0035875951435311256, 0.0, -0.00777229946370252,
            0.056593955721350044, 0.09433614865616086, 0.008921359362670706,
            0.055734331567878256, 0.08908809469436402]
    w12t = [0.0, -0.0021150569579430887, 0.0, 9.367506770274758e-17, 0.02373249705660925,
            -0.03135903371241482, -0.016595508502429055, 0.0520092531720927, 0.0,
            -0.006939310874167318, 0.20271106577475637, -0.0052381845704629,
            0.004891002509819227, 0.0, -0.007752905382582925, 0.05987427889337356,
            0.0820870627112757, 0.014526197775378705, 0.04783103942457454,
            0.08233760268211984]
    w13t = [0.0, -0.0, -0.0, 0.0, -6.691950947171401e-19, 4.740429376046475e-18, 0.0, -0.0,
            0.0, 0.0, -0.0, -0.0, -1.758364082434267e-16, 0.0, -0.0, -0.0, 0.0, -0.0, 0.0,
            0.5000000000000001]
    w14t = [-9.643912558097658e-11, -0.0050399738494296496, 0.006240067134787434,
            -8.168713118117357e-10, 0.029922546610572653, -0.003831780758886675,
            -0.02109942555425765, 0.05300324682462914, -2.0522565326285698e-11,
            -0.023221187645030297, 0.18171475659440825, -0.008043660464453165,
            0.006854039888749915, 0.000519742200943334, -0.008763961954351902,
            0.059893570187057384, 0.08872379649745515, 0.02852907814457536,
            0.03675023141956626, 0.07784891565749662]
    w15t = [-0.0, -0.0, -0.0, -0.0, 0.0, -0.0, -0.0, 0.0, -0.0, -0.0, 0.2198594562930118,
            -0.0, -0.0, -0.0, -0.0, 0.04758011770644055, 0.12578976738198605, -0.0, -0.0,
            0.10677065861856155]
    w16t = [9.973616692597187e-12, -0.002111523607882587, 0.000765893258734339,
            -1.0628022427938349e-11, 0.02425749115891634, -0.007975124564156498,
            -0.017473000366884976, 0.05601612405866506, -6.596743906345443e-12,
            -0.02586654793029544, 0.18283138659726084, -0.006000210657826323,
            0.007119862742960104, 2.1543631484089202e-10, -0.007905201070381078,
            0.059856769049511795, 0.10060434125699806, 0.03222776391222757,
            0.03053338446890243, 0.07311859148506465]
    w17t = [0.0, 0.0, -0.0, -0.0, 0.181926392116249, -0.0, -0.0, -3.9674954755387275e-17,
            -0.0, -0.0, 0.32681877979942786, -0.0, -0.008745171915676898,
            3.9674954755387275e-17, -0.0, 0.0, 0.0, -0.0, -0.0, 0.0]
    w18t = [-3.089049106471567e-11, -0.010519572030844835, 0.0013399784161230476,
            0.0003691217993650725, 0.030433150523738264, -0.0009857124073506323,
            -0.022335271789932047, 0.05459788751909203, -2.7119100290369388e-11,
            -0.02146650901934323, 0.18143282311778505, -0.007994824784190365,
            0.005852848473598183, 2.1566807472818993e-9, -0.006698105331070265,
            0.058438151545986625, 0.08844809757806639, 0.023376878840100652,
            0.046072986812801385, 0.07963806863740361]
    w19t = [-0.0, -0.0, 0.0, -0.0, -0.0, -0.0, -0.0, 0.0, -0.0, -0.0, 0.21377868735018923,
            -0.0, -0.0, 0.0, -0.0, 0.04516389575401886, 0.1302292901598254, -0.0, 0.0,
            0.11082812673596654]
    w20t = [7.533991949505759e-11, -0.013844343769093286, 3.890634897683969e-10,
            -0.0014387668398351675, 0.020598543097712488, -1.1181087071326063e-8,
            -0.01948444596873307, 0.05541665064916829, -1.518476001319919e-10,
            -0.02596502813319883, 0.1808175102318874, -0.004579705973773067,
            0.005825776868587391, 4.6007708009610746e-9, -0.0046876799876500235,
            0.05688024983624939, 0.11207946723761839, 0.014303668090091598,
            0.044422750864556736, 0.07965536006417218]
    w21t = [0.0, 0.0, 0.06455652837124198, 0.0, 0.178069451187163, -0.15341669030195554,
            0.0, 0.12351856122756995, 0.0, 0.0, 0.21867000759778846, 0.0,
            -0.023295149141940086, 0.0, -0.003288160556104364, 0.17700179310693104,
            0.006143447950208203, 0.0, 0.04071674553919557, 0.14132346501990195]
    w22t = [0.0, 0.0, 0.07328869996428451, 0.0, 0.19593630860627578, -0.15160350315488128,
            0.0, 0.1290693464408609, 0.0, 0.0, 0.20897763141227568, 0.0,
            -0.021925431035317158, 0.0, -0.00647106580980153, 0.18010601289300818, 0.0, 0.0,
            0.02595168574389885, 0.13667031493939596]
    w23t = [-0.0, -0.0, 0.0, -0.0, 0.6900000000000017, -0.0, -0.0, 0.0, -0.0, -0.0,
            0.19949253798763725, -0.0, -0.11949253798763816, -0.0, -0.0, 0.0, 0.0, -0.0,
            0.0, 0.0]
    w24t = [2.7416060722916885e-10, 0.03279851446016573, 0.05482942294157594,
            1.6808968847828969e-9, 0.23098209829527744, -0.127296887172888,
            1.233417693451104e-10, 0.1676713504911736, 2.6902695953232987e-10,
            -1.81041595332319e-10, 0.11192694235047511, -0.0009046010539312808,
            -0.03158635899513159, 4.729715392434103e-10, -0.020212147621465146,
            0.21547631362842085, 2.3302505410358413e-9, 1.6684667259725497e-9,
            0.016107280866013464, 0.12020806517224032]
    w25t = [-0.0, -5.0660410896093355e-18, 0.0, 5.0660410896093355e-18, 0.562887057319936,
            4.665392816308511e-18, -4.963989925462136e-15, 0.0, -0.0, -0.0, 0.0,
            -0.15111111111111833, -0.0, -0.0, -0.0, 0.3871129426800691,
            -0.028888888888885078, -0.0, -0.0, -0.0]
    w26t = [3.2196598974343385e-10, 0.08211794085495151, 0.047313624197963255,
            1.4975831321446326e-10, 0.2036334202489531, -0.09721783795441537,
            -6.273765541117185e-10, 0.13077993557255166, -1.0968239871717082e-10,
            -1.1812032582377953e-10, 0.18977508840124324, -1.7587662331978057e-9,
            -0.04297554331983498, 2.2122198451716471e-10, -0.03980661086810857,
            0.2516731057038056, 0.04379913020071229, 6.755838340752206e-10,
            5.730078593378238e-10, 0.0009077476345858865]
    w27t = [-0.0, -0.0, 0.0, -0.0, 0.6751155811095628, -0.0, -0.0, 0.0, -0.0, -0.0,
            0.20889822496559668, -0.0, -0.11401380607515949, -0.0, -0.0, 0.0, 0.0, -0.0,
            -0.0, 0.0]
    w28t = [3.1240843093976506e-10, 0.033504884619729336, 0.053474881454892605,
            2.8211825936305724e-9, 0.23689108616800225, -0.12555326600681038,
            9.935047744100234e-11, 0.1723411766492513, 3.2629699220372303e-10,
            -1.386819691236479e-10, 0.1145246049508838, -0.001822204816870733,
            -0.029090314054612768, 5.836891106122332e-10, -0.0235342091892772,
            0.23091435041977204, 2.446252744569892e-9, 2.1937503046158836e-9,
            2.718293623101887e-8, 0.1083489739778547]
    w29t = [-0.0, -0.0, 0.0, -0.0, 0.4910940264104013, -0.0, 0.012858457525192869, 0.0,
            -0.0, -0.0, 0.0, -0.18, -0.0, -0.0, -0.0, 0.44604751606440585, -0.0, -0.0, -0.0,
            0.0]
    w30t = [3.5580824832385636e-10, 3.2380724065020703e-7, 0.05338838061869483,
            2.1624334859243212e-10, 0.1697928078624459, -0.11732497229755073,
            -3.8203256668381455e-7, 0.14244541866605412, -2.2515409565257293e-11,
            -3.7148720331266884e-10, 0.1726014765517114, -6.417565103169377e-9,
            -0.047662320845948666, 4.970558358819477e-10, -0.015012308860930395,
            0.22116851653847186, 0.18959365386568477, 7.726193821067852e-10,
            2.779326852320861e-9, 0.001009408317207001]
    w31t = [0.0, 0.0, 0.0, 0.0, -2.2529049373126247e-17, 0.0, 0.83, 0.0, 0.0, 0.0, 0.0, 0.0,
            -0.22, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    w32t = [0.0, 0.0, 0.0, 0.0, 0.7960000000000002, 0.0, -1.5265566588595902e-16, 0.0, 0.0,
            0.0, 0.0, 0.0, -0.22000000000000014, 0.0, 1.3877787807814457e-16, 0.0, 0.0, 0.0,
            0.03399999999999996, 0.0]
    w33t = [0.0, -0.0, -0.0, -0.0, 0.3299999999999994, 0.2800000000000006, 0.0, 0.0, 0.0,
            0.0, 1.755623758819901e-17, 1.755623758819901e-17, -0.0, 0.0,
            -1.755623758819901e-17, 0.0, 0.0, 0.0, 0.0, 0.0]
    w34t = [1.8121482049613785e-9, 1.953836774701269e-7, 0.09223164870592435,
            0.08411821812658415, 0.006540587245736241, 2.7379226621427814e-9,
            0.09910753522728317, 1.498738704274719e-7, 0.13515200288712242,
            0.006193043767241372, 0.08329074250198215, 3.410012873147804e-8,
            1.1369549411300035e-9, 2.021059040306791e-7, 0.004195094996550774,
            1.882744012018949e-7, 8.575974710203657e-10, 0.09236731587227337,
            2.7511505156397633e-9, 0.006803031635546478]
    w35t = [0.0, 0.0, -0.0, 0.0, 0.7959999999999998, -0.0, 0.0, -0.0, -0.0, 0.0, -0.0, -0.0,
            -0.22, -0.0, -0.0, 0.0, 0.0, -0.0, 0.03400000000000017, 0.0]
    w36t = [1.1273258778815825e-9, 2.983693996539084e-8, 2.7479268657668387e-9,
            1.78195435384964e-9, 0.1574016738887992, 1.31210090874932e-9,
            0.15871383470321934, 1.0841429502413197e-9, 0.1230851225656321,
            1.920726950160699e-9, 0.0197976669580578, 2.037605109612841e-9,
            6.116418453873342e-10, 2.307642503480735e-9, 9.42234758252193e-10,
            0.1510015733739434, 2.5148759920165294e-8, 1.0464689592213275e-9,
            5.543442462146499e-8, 1.1704526190250803e-9]
    w37t = [0.0, 0.0, 0.0, 2.7755575615628914e-17, -5.551115123125783e-17, 0.0,
            0.8299999999999998, 0.0, 0.0, 0.0, -1.1379786002407855e-15, 0.0,
            -0.2199999999999989, 0.0, 0.0, 4.107825191113079e-15, 0.0, 0.0, 0.0, 0.0]
    w38t = [1.0443712948059114e-10, 5.2477565540742585e-9, 0.08119210339683554,
            0.08041442447942818, 1.3500573400013555e-8, 1.3888073320012455e-10,
            0.08989356382159856, 1.5608927894431535e-8, 1.0932005070533145e-10,
            5.193463903930771e-10, 0.07406895927054744, 0.06708506399932766,
            1.0567702389032397e-10, 1.996190844828679e-10, 0.059079711095228464,
            0.08298370492415096, 4.3274643309951834e-11, 0.07528242283001076,
            4.841408887583296e-11, 1.0556645375862988e-8]
    w39t = [0.0, 0.0, 0.0, 0.0, 1.6653345369377348e-16, 0.0, 0.8299999999999998, 0.0, 0.0,
            1.942890293094024e-16, 0.0, 0.0, -0.22, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    w40t = [1.9381826431140206e-10, 2.999818731738452e-10, 4.103426637087461e-10,
            2.2744288067996139e-10, 0.15630910339305995, 1.4535607893445071e-10,
            0.1568131916194722, 1.519235201570981e-10, 6.389880371828789e-9,
            2.9179083390573893e-10, 2.348814252571861e-9, 1.608557856222224e-10,
            7.615077834915424e-11, 2.3938293811515366e-10, 9.817487864588622e-11,
            0.1499008017355208, 5.5489995150045505e-9, 1.4872584558656299e-10,
            0.14697688632627273, 1.9403397490250221e-10]
    w41t = [0.0, -0.003392742308847771, 0.0, -0.001704731699272089, 0.026209661740928515,
            -0.04895127670745864, -0.028310530666442145, 0.07970846611579031,
            -0.006091550656354434, -0.01772949018065377, 0.3496375096278862,
            -0.009810028561543104, 0.0066162066090907625, 0.0, -0.01400964921942805,
            0.09738313723086277, 0.16435744872368885, 0.016435689355704775,
            0.10356751838407473, 0.1560843622119731]
    w42t = [-0.09486167717442828, 0.04019416692914493, 0.03757722317904152, 0.0,
            0.12801741187716337, 0.0, -0.019980959763450823, 0.13355629824656448,
            0.019271038397774748, 0.0, 0.28797176886025694, 1.0099015437560113e-17,
            0.008845283123909222, 0.0, -0.01515736306212094, 0.10734222426525301, 0.0,
            0.06147426694120132, 2.550023892786934e-17, 0.17575031817969047]
    w43t = [-0.0, -0.0, 0.0, 0.0, 0.2542383528166067, -0.10786045982487949, -0.0,
            0.32907983060645285, -0.0, -0.0, -0.0, -0.0, 0.0, -0.0, 0.029160101685153927,
            0.13000000000000084, -0.0, 0.0, -0.0, 0.23538217471666467]
    w44t = [9.271630518019008e-12, 0.06109011163374341, 0.07388719114467274,
            0.019692527427610754, 0.10331812358710908, 0.08117018360404031,
            0.023384402697719586, 0.12195682881828308, 1.6869831402670816e-11,
            0.034579519154434864, 0.0279535551175954, 0.058415921305426353,
            7.482407199562929e-12, 0.0294123632537224, 0.0020709632754864218,
            0.043221718548742204, -8.420657157961142e-13, 0.08680666625664661,
            -3.5849856607860955e-11, 0.10303992417783474]
    w45t = [-0.0, -0.0, -0.0, -0.0, 0.37577751021966854, -0.0, -0.06097567215251032,
            0.42519816193284143, -0.0, -0.0, 0.0, 0.0, -0.0, -0.0, 0.0, 0.13, -0.0, 0.0,
            -0.0, 0.0]
    w46t = [3.012837218605047e-11, 3.7507519975889906e-10, 5.188596103929877e-10,
            8.720148171734737e-11, 0.14384555524146925, 7.013623147638555e-11,
            0.07485022181879199, 0.24773939153900298, 3.178846134958025e-11,
            7.048698031118573e-11, 0.01966163671652463, 0.08856704199858074,
            9.041854010060457e-11, 3.818058519030339e-11, 0.0676662863111033,
            0.1103383628541097, 3.015261568346702e-11, 6.956971448724868e-10,
            1.8368482196615338e-11, 0.11733150146392567]
    w47t = [0.0, -0.0, 0.0, 0.0, 0.11199761883276771, -0.0, -0.0, -0.0, 0.0, -0.0,
            0.3750880542218099, -0.0, -0.0, -0.0, -0.0, 0.14127637498099654, 0.0,
            0.09588426469924055, -0.0, 0.1457536872651849]
    w48t = [6.769250370239776e-11, 0.05845600849927483, 0.06857721044007528,
            0.018637080446924643, 0.09512946482010036, 0.07994346654990074,
            0.0255458598305162, 0.07233232435088978, 5.7033932361977025e-11,
            0.03211230166228922, 0.08806729719101165, 0.04699072744010763,
            2.3315039720774004e-11, 0.04108314003881863, 0.0016832424016449622,
            0.05798953966566996, 6.097942366786586e-11, 0.08227280031037314,
            3.5395232965443286e-11, 0.10117953610798705]
    w49t = [-0.0, -0.0, 0.0, -0.0, -0.0, -0.0, -0.03380540048487986, 0.0, -0.0, -0.0,
            0.44152390203521386, -0.0, -0.0, -0.0, -1.4672583285548087e-16,
            0.06721017342340446, 0.3950713250262614, 0.0, -0.0, 0.0]
    w50t = [2.1014063331246616e-9, 1.1373539554822593e-10, 3.5443049815757925e-10,
            6.395587061347731e-11, 0.03864847109756799, 4.834804519361183e-11,
            0.07043822680094357, 0.12089020149556272, 5.121202000743714e-11,
            4.0472110532060826e-11, 0.17110546289126932, 0.07062955046440907,
            4.1666623418175826e-11, 6.832704120785326e-11, 0.04016682517855291,
            0.13350447210022412, 0.10340911779701802, 1.3933271479984958e-10,
            8.336899197294065e-11, 0.12120766906819612]
    w51t = [0.0, -0.003017422928334872, 0.0, -0.0005578189854036299, 0.014262826621635725,
            -0.026715129365893816, -0.01629323231811754, 0.04647310508003174,
            -0.000785693717675302, -0.009357177603515936, 0.20100258315237743,
            -0.00550122561735638, 0.0035875951435311256, 0.0, -0.00777229946370252,
            0.056593955721350044, 0.09433614865616086, 0.008921359362670706,
            0.055734331567878256, 0.08908809469436402]
    w52t = [0.0, -0.0021150569579430887, 0.0, 9.367506770274758e-17, 0.02373249705660925,
            -0.03135903371241482, -0.016595508502429055, 0.0520092531720927, 0.0,
            -0.006939310874167318, 0.20271106577475637, -0.0052381845704629,
            0.004891002509819227, 0.0, -0.007752905382582925, 0.05987427889337356,
            0.0820870627112757, 0.014526197775378705, 0.04783103942457454,
            0.08233760268211984]
    w53t = [0.0, -0.0, -0.0, 0.0, -6.691950947171401e-19, 4.740429376046475e-18, 0.0, -0.0,
            0.0, 0.0, -0.0, -0.0, -1.758364082434267e-16, 0.0, -0.0, -0.0, 0.0, -0.0, 0.0,
            0.5000000000000001]
    w54t = [0.010908485275843953, 0.004999163328775131, 6.274396323951957e-10,
            -2.8767443124621546e-10, 0.019419368032426595, 3.382848453189391e-11,
            -0.00800945565640827, 0.06251307669202077, 0.008690491233018096,
            6.422162170316477e-11, 0.15767243429095218, 0.00590692742299124,
            0.0016466554361123198, 7.897736770922339e-10, 0.003233313742678891,
            0.04649439437391507, 0.12750737364874276, 2.5572627983679824e-11,
            1.8590441541635798e-9, 0.05901776906672541]
    w55t = [-0.0, -0.0, -0.0, -0.0, 0.0, -0.0, -0.0, 0.0, -0.0, -0.0, 0.2198594562930118,
            -0.0, -0.0, -0.0, -0.0, 0.04758011770644055, 0.12578976738198605, -0.0, -0.0,
            0.10677065861856155]
    w56t = [1.3109714607156008e-10, 1.1357561430115851e-10, 1.7585285129402294e-10,
            4.301003554466401e-11, 0.005216014865391346, 3.579636871556504e-11,
            -0.001551752910840366, 0.0535526206687511, 4.7332876263148247e-11,
            2.370404293588521e-10, 0.16936793060582078, 1.6792688363353818e-9,
            7.678157094220432e-11, 9.985884207789293e-11, 3.87649220324801e-11,
            0.06447733098180772, 0.1360997517819511, 0.006552709711679462,
            1.7829205692466388e-10, 0.06628539143876752]
    w57t = [0.0, 0.0, -0.0, -0.0, 0.181926392116249, -0.0, -0.0, -3.9674954755387275e-17,
            -0.0, -0.0, 0.32681877979942786, -0.0, -0.008745171915676898,
            3.9674954755387275e-17, -0.0, 0.0, 0.0, -0.0, -0.0, 0.0]
    w58t = [2.9582947785857556e-11, 0.045754919534512, 0.0004820271021414232,
            0.0002773957562477908, 0.06681819160523143, 1.164750367131227e-11,
            -1.1221500122380731e-11, 0.0792075323745304, 5.0823442924804755e-11,
            8.32029906709523e-11, 0.12758286564203336, 0.0005899568024840472,
            3.5270546527460813e-11, 3.73900972319007e-11, 0.004128213011533481,
            0.06070245162490041, 5.0989064048308504e-11, 0.022300346324049675,
            1.5313539399465177e-11, 0.09215609991933713]
    w59t = [-0.0, -0.0, 0.0, -0.0, -0.0, -0.0, -0.0, 0.0, -0.0, -0.0, 0.21377868735018923,
            -0.0, -0.0, 0.0, -0.0, 0.04516389575401886, 0.1302292901598254, -0.0, 0.0,
            0.11082812673596654]
    w60t = [4.351192335277095e-11, 8.486331684486964e-11, 7.546636588035156e-11,
            2.0330326318467156e-11, 0.010871174870787148, 1.1275050869788192e-11,
            -0.003155848739032144, 0.05458914972390995, 1.3592520668814048e-11,
            9.616925631887336e-11, 0.17352275985947085, 0.0008828278226669918,
            7.378123551773738e-11, 3.224032907263678e-11, 2.1410615811242994e-11,
            0.06643442997117671, 0.12513685069855257, 0.007099270843369595,
            3.640071761237825e-11, 0.06461938444005637]
    w61t = [0.0, 0.0, 0.06455652837124198, 0.0, 0.178069451187163, -0.15341669030195554,
            0.0, 0.12351856122756995, 0.0, 0.0, 0.21867000759778846, 0.0,
            -0.023295149141940086, 0.0, -0.003288160556104364, 0.17700179310693104,
            0.006143447950208203, 0.0, 0.04071674553919557, 0.14132346501990195]
    w62t = [0.0, 0.0, 0.07328869996428451, 0.0, 0.19593630860627578, -0.15160350315488128,
            0.0, 0.1290693464408609, 0.0, 0.0, 0.20897763141227568, 0.0,
            -0.021925431035317158, 0.0, -0.00647106580980153, 0.18010601289300818, 0.0, 0.0,
            0.02595168574389885, 0.13667031493939596]
    w63t = [-0.0, -0.0, 0.0, -0.0, 0.6900000000000017, -0.0, -0.0, 0.0, -0.0, -0.0,
            0.19949253798763725, -0.0, -0.11949253798763816, -0.0, -0.0, 0.0, 0.0, -0.0,
            0.0, 0.0]
    w64t = [3.514750492267543e-11, 0.10214522080573049, 1.597406036616589e-9,
            8.750555705113527e-10, 0.23010239600015692, -0.025107535839082554,
            0.022839041940755388, 0.14840930939238076, -2.63750519241461e-10,
            -3.153990002739095e-11, 0.12599682871937234, -6.592300796551849e-10,
            -0.033034603297895634, -1.0048043879523312e-10, -0.05977664458678473,
            0.17110148065192135, -7.484365217953592e-11, 5.423303529748546e-10,
            -2.6451121785500766e-10, 0.08732450455786189]
    w65t = [-0.0, -5.0660410896093355e-18, 0.0, 5.0660410896093355e-18, 0.562887057319936,
            4.665392816308511e-18, -4.963989925462136e-15, 0.0, -0.0, -0.0, 0.0,
            -0.15111111111111833, -0.0, -0.0, -0.0, 0.3871129426800691,
            -0.028888888888885078, -0.0, -0.0, -0.0]
    w66t = [1.8912838027016928e-10, 0.11186702544104837, 9.642820192698134e-10,
            1.6634653074734197e-10, 0.28429472444022086, -1.1817732641283577e-9,
            0.04395107030861871, 3.187673684990668e-10, 2.4043649410537723e-10,
            7.596312767259209e-11, 0.15202955989083144, -2.684945614964219e-10,
            -0.04192482243044426, 2.561726091221542e-10, -0.00317836870343084,
            0.21410113035169862, 0.00885967845556758, 1.3621716317872475e-10,
            1.056128589780399e-9, 2.9271511811487536e-10]
    w67t = [-0.0, -0.0, 0.0, -0.0, 0.6751155811095628, -0.0, -0.0, 0.0, -0.0, -0.0,
            0.20889822496559668, -0.0, -0.11401380607515949, -0.0, -0.0, 0.0, 0.0, -0.0,
            -0.0, 0.0]
    w68t = [9.397263321364191e-11, 0.09919154706754962, 6.368409304847823e-10,
            3.4659912616090777e-10, 0.22064663461074588, -1.0614510903600328e-9,
            0.020025141305240964, 0.12666864829527658, 5.680696360019536e-12,
            1.1755153585122928e-11, 0.12107267101747536, -4.262475933360104e-10,
            -0.04188688601302829, 1.563765014658337e-10, -0.06521868019008682,
            0.16284169107985866, 6.24790643979899e-11, 4.814802706555206e-10,
            1.7210741701012835e-10, 0.126659232347375]
    w69t = [-0.0, -0.0, 0.0, -0.0, 0.4910940264104013, -0.0, 0.012858457525192869, 0.0,
            -0.0, -0.0, 0.0, -0.18, -0.0, -0.0, -0.0, 0.44604751606440585, -0.0, -0.0, -0.0,
            0.0]
    w70t = [1.6974829666717566e-10, 1.0082180073560484e-9, 9.65774553660695e-10,
            1.2879053175416931e-10, 0.27923305689807, -5.440139862345934e-10,
            1.7559116578590743e-9, 2.711321080631578e-10, 2.6149877285920984e-10,
            8.34869368815444e-11, 0.1321064798026845, -2.1082947498475423e-10,
            -0.050426240607246944, 2.613177760924842e-10, -0.0007980562114940349,
            0.19920369561928547, 0.18964773028735682, 1.2284415089112795e-10,
            0.021033329674863793, 2.626010576491403e-10]
    w71t = [0.0, 0.0, 0.0, 0.0, -2.2529049373126247e-17, 0.0, 0.83, 0.0, 0.0, 0.0, 0.0, 0.0,
            -0.22, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    w72t = [0.0, 0.0, 0.0, 0.0, 0.7960000000000002, 0.0, -1.5265566588595902e-16, 0.0, 0.0,
            0.0, 0.0, 0.0, -0.22000000000000014, 0.0, 1.3877787807814457e-16, 0.0, 0.0, 0.0,
            0.03399999999999996, 0.0]
    w73t = [0.0, -0.0, -0.0, -0.0, 0.3299999999999994, 0.2800000000000006, 0.0, 0.0, 0.0,
            0.0, 1.755623758819901e-17, 1.755623758819901e-17, -0.0, 0.0,
            -1.755623758819901e-17, 0.0, 0.0, 0.0, 0.0, 0.0]
    w74t = [2.2468952807735533e-10, 0.050562454311270474, 0.09016607500556278,
            0.08107397437937414, 8.571788455864894e-9, 2.1053512029003874e-10,
            2.070499474446891e-9, 4.393779191232077e-9, 1.5958175108944467e-8,
            0.05544201883187869, 0.04521065466270187, 0.044817466934343214,
            0.034527779849848506, 6.018106669878948e-9, 0.07894326937005992,
            1.4253350296227537e-9, 1.4428522398845262e-10, 0.050343158395440055,
            0.07891309978641055, 9.45591603813389e-9]
    w75t = [0.0, 0.0, -0.0, 0.0, 0.7959999999999998, -0.0, 0.0, -0.0, -0.0, 0.0, -0.0, -0.0,
            -0.22, -0.0, -0.0, 0.0, 0.0, -0.0, 0.03400000000000017, 0.0]
    w76t = [1.6604781884406825e-9, 2.6253857622375046e-8, 2.42027507505511e-9,
            2.593165177466118e-9, 0.1529906411908314, 1.9828392414560096e-9,
            0.15312073839508317, 1.619277533246167e-9, 0.12352268543167817,
            2.3303450589816268e-9, 0.028016109507989672, 2.9009081280876434e-9,
            9.73653987986173e-10, 3.0400520343381018e-9, 1.7655406807812698e-9,
            0.1523496959876855, 3.227058335651183e-8, 1.659082728293909e-9,
            4.632124016871946e-8, 1.695433216256918e-9]
    w77t = [0.0, 0.0, 0.0, 2.7755575615628914e-17, -5.551115123125783e-17, 0.0,
            0.8299999999999998, 0.0, 0.0, 0.0, -1.1379786002407855e-15, 0.0,
            -0.2199999999999989, 0.0, 0.0, 4.107825191113079e-15, 0.0, 0.0, 0.0, 0.0]
    w78t = [1.1176928006023076e-10, 2.995851513358725e-9, 0.0767452013083987,
            0.07666521247095669, 2.9756605239053962e-9, 1.0928836177191118e-10,
            0.07761337222589743, 5.642793921672302e-8, 8.880701322916989e-11,
            3.648723084125519e-9, 0.07603196690070452, 0.07533524506679641,
            2.121806092195131e-9, 1.497276308531424e-10, 0.07453266280803049,
            0.07692306071869554, 5.5267509126534306e-11, 0.07615320748011697,
            5.3292273981351296e-11, 2.2822711905323473e-9]
    w79t = [0.0, 0.0, 0.0, 0.0, 1.6653345369377348e-16, 0.0, 0.8299999999999998, 0.0, 0.0,
            1.942890293094024e-16, 0.0, 0.0, -0.22, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    w80t = [1.5301379346160968e-9, 1.1599217598222888e-9, 1.9519333729514195e-9,
            1.884819511925185e-9, 0.1528812912330765, 1.2845904202338365e-9,
            0.15293067206531155, 1.2279013918134221e-9, 5.3917280100417234e-8,
            2.091621092391012e-9, 5.137105751966751e-9, 1.3421973330264116e-9,
            7.797347777262099e-10, 1.877467859064439e-9, 1.1449977903013299e-9,
            0.1522402134830058, 5.2502394791291094e-8, 1.289986226698016e-9,
            0.1519476926651265, 1.4313896035704728e-9]

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
    @test isapprox(w50.weights, w50t)

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
    @test isapprox(w76.weights, w76t)
    @test isapprox(w77.weights, w77t)
    @test isapprox(w78.weights, w78t)
    @test isapprox(w79.weights, w79t)
    @test isapprox(w80.weights, w80t)

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
    w3t = [3.9517277944273235e-13, 2.8175454087012248e-12, 0.08557388122551701,
           2.923044739569266e-12, 2.8860883669300447e-12, 3.013990305140833e-12,
           1.6497089962249406e-12, 0.25924212630559207, 1.945102100853769e-12,
           2.8682173814339863e-12, 2.5066088936427084e-12, 3.0291972157017444e-12,
           1.6132842764301432e-12, 1.7011559368376116e-12, 0.0435030685421342,
           1.921938211101941e-12, 3.284788568401419e-13, 0.3837286036513077,
           1.607905642849369e-12, 0.2279523202442415]
    w4t = [4.290787155109147e-11, 0.07544186699634876, 0.021396172655484873,
           0.027531481813503244, 0.02375148897763694, 0.12264432042707833,
           3.7779463731968698e-6, 0.22632335856335578, 3.8909751127216796e-10,
           0.02418450695936337, 3.8945712296570245e-10, 2.5979927223109437e-6,
           2.291674695643033e-6, 5.942488053271556e-11, 0.02472298577518245,
           3.892386375947259e-10, 9.010864797681229e-11, 0.30009565874343674,
           1.3290984747432744e-10, 0.1538994899816738]
    w5t = [3.907003246117169e-12, 2.301201807994552e-12, 2.9602208991537512e-12,
           9.473301280126003e-13, 3.0277756384220105e-12, 1.2765388182957773e-12,
           4.646261820964922e-12, 1.5517034164068801e-12, 3.714094341126798e-12,
           1.8110280722972018e-12, 9.052674435805542e-13, 6.547290379885876e-12,
           0.0043641446233057095, 2.036772311509158e-12, 0.05248599404102016,
           4.421157501803325e-12, 1.5695522925002238e-12, 0.5604466984137566,
           1.1158068900423167e-12, 0.3827031628791785]
    w6t = [2.547112635903074e-11, 4.01901198789686e-6, 2.4084561946195076e-6,
           1.2336586422546944e-6, 1.7781762247025576e-6, 2.1633922564164977e-6,
           1.3229748185654915e-6, 6.29437403194288e-6, 2.3996795995873316e-10,
           1.15755137975054e-6, 2.401748723676182e-10, 1.448842496872421e-5,
           0.004374890511245133, 3.813629090542969e-11, 0.0524739635479751,
           2.4001731879662444e-10, 5.5350440318706916e-11, 0.5604571460251726,
           8.309165102271217e-11, 0.3826591329728926]
    w7t = [0.0, 0.0, 0.0, 0.05362285557460243, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
           0.42414457744261835, 0.0, 0.0, 0.0, 0.0, 0.027024889194692054, 0.0,
           0.3034370015158859, 0.0, 0.19177067627220137]
    w8t = [1.1008281068042757e-5, 0.055457116453828906, 0.012892903094396474,
           0.03284102649502979, 0.014979204379343075, 0.05788669109782618,
           1.0224197846968107e-6, 9.705504060604912e-6, 2.0458102296001954e-6,
           0.012049795193184627, 0.37351141080304123, 1.3303580854165434e-6,
           1.0648304534767993e-6, 9.906438750241614e-6, 0.007878565110236033,
           0.022521836037796145, 3.1895242149782774e-6, 0.26190144912467667,
           1.3462532236588775e-6, 0.14803938279077022]
    w9t = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5931779718994062, 0.0, 0.0,
           0.0, 0.02433672134677551, 0.05339236823580681, 0.0, 0.0, 0.0,
           0.32909293851801136]
    w10t = [1.6016566197250686e-5, 1.1481607285816556e-5, 8.007540526406977e-6,
            4.447839793158355e-6, 5.070283063020247e-6, 4.179903041407502e-6,
            1.622827348638267e-5, 6.990722659388864e-6, 2.028453244085152e-6,
            2.3437681751855885e-6, 0.5956880256595543, 5.664323511449351e-6,
            2.25988006187764e-6, 1.7702146674931902e-5, 0.02350402920236911,
            0.05218633084468716, 6.981194915944795e-5, 2.3947834281194956e-5,
            3.7102298476765392e-6, 0.3284217229723807]
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
    w13t = [0.0, 0.0, 0.0, 0.5221634101217328, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.47783658987826716, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    w14t = [2.7039950921233163e-11, 5.304814952504774e-10, 5.3129074809144e-10,
            0.39174140576271615, 5.364912915934331e-10, 3.2595950170065766e-11,
            0.2116643709792961, 0.20932312570881034, 9.329098548843035e-11,
            1.587497769904578e-10, 0.18727092365873096, 8.769818494491118e-8,
            4.067206010635532e-8, 2.722556146908735e-11, 4.2258768755406594e-8,
            5.326193526193868e-10, 1.1001256941293843e-10, 5.288670867914201e-10,
            1.1950918195302962e-10, 3.32585978316583e-11]
    w15t = [0.0, 0.0, 0.0, 0.42641947441117506, 0.0, 0.0, 0.2135309391746484, 0.0, 0.0, 0.0,
            0.3600495864141765, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    w16t = [3.1703780442621356e-11, 6.253642949326953e-10, 6.267630976576822e-10,
            0.4232344428724752, 6.323891270351866e-10, 3.862504951142336e-11,
            0.21316313195164788, 9.489251548229374e-7, 1.0998892197735694e-10,
            1.8723956893005518e-10, 0.3636012636443419, 1.0514306294813026e-7,
            5.126977383195584e-8, 3.1905905494459565e-11, 5.234790294036524e-8,
            6.279401001994162e-10, 1.2980765943851301e-10, 6.234275617003164e-10,
            1.4107615024583291e-10, 3.940943260227995e-11]
    w17t = [0.0, 0.0, 0.0, 0.0, 0.8446617044761973, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.1553382955238028, 0.0, 0.0, 0.0, 0.0]
    w18t = [1.6606069500686817e-7, 2.2569375762762836e-7, 2.2360042463316183e-7,
            1.504981890365041e-7, 0.7741843310935376, 5.803186138228797e-8,
            0.10999501737171634, 1.5356150508828583e-7, 3.30400599420469e-7,
            1.4772038433976735e-7, 1.4017814391691792e-7, 5.432227068945168e-8,
            4.0300518544647105e-8, 8.720973476934716e-8, 3.9294736119938764e-8,
            0.11581773626166317, 3.2117894531448195e-7, 1.6454413538672828e-7,
            4.0576995548256315e-7, 2.0690722601358115e-7]
    w19t = [0.0, 0.0, 0.0, 0.0, 0.7764286452522803, 0.0, 0.1093312370040934, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.11424011774362632, 0.0, 0.0, 0.0, 0.0]
    w20t = [1.756102069538767e-7, 2.494719209887536e-7, 2.366783048369117e-7,
            1.579956579456619e-7, 0.774188976862582, 6.352849238127664e-8,
            0.10999236653671997, 1.2612571282258627e-7, 3.7140895041790373e-7,
            1.6095963372252927e-7, 1.5385133100170124e-7, 5.9390111595927995e-8,
            4.394474525678509e-8, 9.560066114336652e-8, 4.2939547571731136e-8,
            0.11581500168274747, 7.176412577743592e-7, 1.4219749591082674e-7,
            6.711725868359599e-7, 1.8640133339391738e-7]
    w21t = [5.246410071901244e-9, 1.371360542346673e-8, 1.8285488392067687e-8,
            1.110605328291695e-8, 0.5180586237153811, 1.7478694817037212e-9,
            0.0636510187959932, 1.1505571260279971e-8, 1.0493218714510297e-8,
            5.6508912112153355e-9, 9.263903401136402e-9, 1.405431016593253e-9,
            9.925289466655718e-10, 3.051292019814939e-9, 9.968264754684221e-10,
            0.14326794993193862, 0.1964970125442071, 1.0780217541288174e-8,
            0.07852527947026894, 1.1302903668605104e-8]
    w22t = [3.5808405398921816e-11, 9.276706869254714e-10, 9.834639922535403e-10,
            0.4836243365433993, 1.3062195823765053e-9, 3.418329060735444e-11,
            0.2843555390371597, 0.1798472306419059, 1.6933692536736022e-10,
            2.6217486963401034e-10, 0.05217287731790357, 5.161223457836004e-9,
            2.4490882267948745e-9, 4.789935654992571e-11, 2.6686514879435067e-9,
            1.0539315646745956e-9, 2.0826274262412784e-10, 8.48587463407438e-10,
            2.3297183068360905e-10, 7.015762023056445e-11]
    w23t = [1.7196397334204643e-11, 4.606593374437138e-11, 4.64867629337321e-11,
            5.4246590455306016e-11, 4.5020103512641234e-11, 2.5097631786779733e-11,
            0.9999999992895859, 5.3793433157832775e-11, 9.803720811728841e-12,
            3.586446445775199e-11, 5.3338347317324754e-11, 4.8521881380882465e-11,
            4.298341610534519e-11, 1.754475227463304e-11, 4.284238173766535e-11,
            4.639252991662063e-11, 2.0996136296889343e-11, 4.5556927003130936e-11,
            3.099329144509537e-11, 2.7669358878738413e-11]
    w24t = [5.0238006615330125e-14, 1.5886390713546995e-12, 1.592417456178811e-12,
            0.2455460091994101, 1.6332395279430773e-12, 1.0616694749223187e-13,
            0.09594359747334343, 0.2562392110961138, 2.782891287371722e-13,
            4.877272742931374e-13, 0.4022711603532498, 1.214143102636416e-8,
            4.52380224989683e-9, 5.3726774007724484e-14, 5.202827264727149e-9,
            1.5947982680787817e-12, 3.4354565542982846e-13, 1.5824890605846971e-12,
            3.8306752939941e-13, 1.2804562266847693e-13]
    w25t = [6.884835405638841e-13, 1.718174429717088e-12, 1.7761127542873972e-12,
            0.5484412380560161, 2.242687531631097e-12, 1.062208567301459e-12,
            0.31695473543556413, 2.231535608256878e-12, 3.039289230758525e-13,
            1.3496361906476665e-12, 0.13460402648383143, 1.9534371071268215e-12,
            1.6811833254664016e-12, 8.250376842665847e-13, 1.6369509265433267e-12,
            1.943301498014961e-12, 8.279097935059882e-13, 1.828118574364412e-12,
            1.3876742080544613e-12, 1.1316971369230826e-12]
    w26t = [5.006409320159508e-14, 1.5761501981872816e-12, 1.580316713777149e-12,
            0.29608061132108704, 1.622007104300589e-12, 1.0431997645482868e-13,
            0.10772454470419436, 1.2357722126285712e-7, 2.759129738380603e-13,
            4.832884332099062e-13, 0.5961947045795979, 7.642156435823121e-9,
            4.095472342817988e-9, 5.3652727586692836e-14, 4.070524791520612e-9,
            1.5828011266161295e-12, 3.405852504535674e-13, 1.5698894615454598e-12,
            3.799212579043351e-13, 1.2670601569139691e-13]
    w27t = [2.0965977638306498e-12, 2.1981604499420718e-12, 1.966863117040418e-12,
            2.053642221497893e-12, 0.7622823927610672, 2.6158618916505694e-12,
            1.639606583672681e-12, 2.5734175543887085e-12, 2.163447200553989e-12,
            2.3919794322207437e-12, 2.6030473410682516e-12, 2.2548548141498357e-12,
            2.335212242026824e-12, 2.5745684006720025e-12, 2.433932750296958e-12,
            0.23771760719791996, 2.077379133081505e-12, 2.5443556078827168e-12,
            2.224151708789596e-12, 2.26569997497047e-12]
    w28t = [1.1308465207681977e-7, 9.73526620770959e-7, 1.5070680511830079e-7,
            9.723264849814074e-8, 0.3050368066970302, 4.716588328541566e-8,
            0.03491168223394641, 1.1694185116945428e-6, 2.1130552122418197e-7,
            1.388754338920844e-7, 0.2315897120042744, 2.9296826516677692e-8,
            1.8409280518957694e-8, 1.3413563861820296e-7, 2.0380438936355492e-8,
            0.10856264372446413, 2.399100164983763e-7, 0.20721423382368512,
            2.522282543298217e-7, 0.11268132584006768]
    w29t = [1.0375024119083457e-12, 1.0617622166099404e-12, 9.177574267465612e-13,
            9.531680530284932e-13, 0.5951748369199624, 1.4453007674796509e-12,
            0.07413615945442659, 1.3572186736124512e-12, 9.386566653775516e-13,
            1.1955529773392841e-12, 1.3659627360604914e-12, 1.2424478886153598e-12,
            1.4239823756887318e-12, 1.3552118993089236e-12, 1.4059755908622179e-12,
            0.16178888374869185, 9.569143707890807e-13, 1.3103849912556126e-12,
            0.16890011985780365, 1.1477423566140866e-12]
    w30t = [1.062884073041064e-7, 2.2302956928097245e-7, 1.2596982724195794e-7,
            7.66502474844137e-8, 0.29018352760142707, 4.5178287895631315e-8,
            1.0893631985401922e-7, 1.2308798316707335e-7, 1.4369795886798624e-7,
            1.4948325261566892e-7, 0.28877538102863004, 2.487798539661704e-8,
            1.665892926640866e-8, 1.0915914270469649e-7, 1.8560189624197693e-8,
            0.11499079031551857, 0.30604802422510774, 1.654792900032147e-7,
            6.837543409236383e-7, 1.5601758497791475e-7]
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
    w34t = [1.3246243248116402e-11, 2.912606295766421e-10, 2.916192286405692e-10,
            9.980619692203968e-8, 2.932352459636008e-10, 1.9667452487020827e-11,
            0.9999997395553456, 4.543572511490105e-8, 5.1182307313605876e-11,
            8.82277343116623e-11, 4.362247754045141e-8, 2.8471495329503985e-8,
            2.031171692850768e-8, 1.3305748755337724e-11, 2.100458068675933e-8,
            2.919945690005951e-10, 6.121955871947987e-11, 2.905064914894068e-10,
            6.71144950632644e-11, 1.9882129470666975e-11]
    w35t = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0]
    w36t = [1.327716513160142e-11, 2.9178042292306077e-10, 2.9214789610861196e-10,
            9.990881483779031e-8, 2.937489329605167e-10, 1.9702517207211864e-11,
            0.999999739052093, 4.5598788193780474e-8, 5.128087949029023e-11,
            8.84003814073117e-11, 4.3694935731962784e-8, 2.8535481767875758e-8,
            2.0359691723930134e-8, 1.3336166011714468e-11, 2.1054530505348053e-8,
            2.925144811578067e-10, 6.132793825994158e-11, 2.9100250688322563e-10,
            6.722601244400006e-11, 1.9918857232838557e-11]
    w37t = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0]
    w38t = [4.910637550468277e-8, 5.2483325735763995e-8, 6.776561956101155e-8,
            6.075155753956884e-8, 6.682444981297579e-6, 2.4301668339295417e-8,
            0.9999924954534397, 3.60249999458616e-8, 5.57734750909545e-8,
            3.930728248308984e-8, 3.500836330036527e-8, 2.5473165190857268e-8,
            1.9469578531690528e-8, 2.9857398051595346e-8, 2.0003831053714972e-8,
            9.328183767528413e-8, 7.129519649399392e-8, 3.776669199925575e-8,
            5.809228296555559e-8, 4.633892952587484e-8]
    w39t = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0]
    w40t = [4.910949593434607e-8, 5.248901161842497e-8, 6.778287658615582e-8,
            6.076819968816433e-8, 6.6836712603933515e-6, 2.4303539832281232e-8,
            0.999992494167897, 3.6031745793528076e-8, 5.577361124407241e-8,
            3.931113806657303e-8, 3.5012154717031964e-8, 2.5474521949750526e-8,
            1.9471164699389783e-8, 2.9859844874834594e-8, 2.0004782732255165e-8,
            9.329471334987515e-8, 7.127041702834e-8, 3.7774597867159985e-8,
            5.8081823089469546e-8, 4.634720362428834e-8]

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
    @test isapprox(w19.weights, w19t)
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
    @test isapprox(w30.weights, w30t, rtol = 1.0e-7)
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

    w1t = [0.0023732458314435797, 0.02478417654578371, 0.011667587595578982,
           0.02183585978966643, 0.008241287070974379, 0.03550224724861833,
           -0.006408569769741798, 0.09320297600893762, -0.007206989959465249,
           0.01226988858431616, 0.1872278496426652, -0.013952101084696744,
           -0.006026909759383776, 0.09627705493690875, 0.0005110433923910731,
           0.01678462171195537, 0.009614188803624819, 0.13436280929845829,
           -0.04234487700762211, 0.081284611119587]
    w2t = [0.006452016632567143, 0.022489884720976502, 0.010294683096246317,
           0.020902946114202372, 0.007116711665155562, 0.03337768008427154,
           -0.006266220580894063, 0.09139178524224749, -0.022373409482123394,
           0.010954134038201172, 0.18619094236730954, -0.014102818327074284,
           -0.0055880934331768006, 0.09574848800246745, 0.00024683037856816767,
           0.017121171465051255, 0.014137258184550324, 0.13074888415894076,
           -0.018056793831845837, 0.07921391950435887]
    w3t = [-0.0, -0.0, -0.0, 0.03197389529230635, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0,
           0.31512961892161206, -0.0, -0.0, -0.0, -0.0, 0.02487038107838771, -0.0,
           0.17763584836953158, -0.0, 0.11039025633816249]
    w4t = [0.022589148447530488, 0.03681201213524642, 0.006833793601672919,
           0.013202059102917465, 0.005824555380017889, 0.039842690803400506,
           -0.006834168984416324, 1.932746823818211e-5, -2.1177840386818776e-5,
           0.011500602752034086, 0.2526517978119041, -0.014411712546746532,
           -1.628444737605379e-5, 1.195721590794361e-5, 0.00751550599061037,
           0.019532808277501732, 4.833328705774014e-6, 0.16879897503211036,
           4.313110291669844e-6, 0.09613896336083594]
    w5t = [-0.0, -0.0, -0.0, -0.0, -0.0, 0.0, 0.0, -0.0, -0.0, 0.0, 0.3223916787635588, 0.0,
           -0.0, -0.0, 0.02719549071911309, 0.017608321236440988, -0.0, -0.0, -0.0,
           0.2928045092808871]
    w6t = [1.2615193933414136e-6, 3.084699616258613e-6, 2.107999320385742e-6,
           2.0045868293388076e-5, 1.2464522070471497e-6, 1.425176826287472e-6,
           0.0004288479884293975, 3.2396669146255933e-6, -1.0567507887280546e-8,
           6.328841933690417e-7, 0.31179751387650373, 1.4280072237100239e-5,
           6.335432111070736e-7, 1.4115851556406504e-6, 0.021338173039848925,
           0.028008567425219174, 5.4050961036738605e-5, 0.05718605896015916,
           -4.81371780894261e-6, 0.24114224256675132]
    w7t = [0.0, -0.0, -0.0, 0.03505933668518188, -0.0, -0.0, -0.0, 0.0, -0.0, -0.0,
           0.27868799505082664, -0.0, -0.0, -0.0, 0.0, 0.01989895478083778, -0.0,
           0.19977893006219768, -0.0, 0.12657478342095604]
    w8t = [3.353883664404619e-6, 0.03812246524541973, 0.012782948631180398,
           0.024251278321704522, 0.011026952156491108, 0.041930242398256504,
           -0.007723026091328561, 6.214585756847854e-6, -3.395699021109978e-5,
           0.012648769570592432, 0.24536589713917978, -0.014844139342288459,
           -2.384652073053903e-6, 6.1640233198501206e-6, 0.0069674988677969054,
           0.01709226942229845, 1.124533214898152e-6, 0.17380674670207089,
           6.802767300693614e-7, 0.0985909013182244]
    w9t = [0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, 0.0, -0.0, 0.0, 0.3893078883432557, 0.0,
           -0.0, -0.0, 0.016399018676705762, 0.035742746503397343, 0.0, 0.0, -0.0,
           0.21855034647664115]
    w10t = [1.6533746597640781e-6, 1.0177845198689362e-6, 7.324863106213353e-7,
            3.881455624184704e-7, 4.772989178963352e-7, 4.7087347899385827e-7,
            7.925736575335058e-7, 8.773752631332231e-7, 1.460032959748696e-7,
            2.3088730150873863e-7, 0.39317612261589835, 2.284184777116435e-7,
            6.966280741557097e-8, 1.4216122454314906e-6, 0.015541709090405573,
            0.03448456080079451, 6.2557383346042255e-6, 2.5670457385759268e-6,
            3.27903950646482e-7, 0.2167799503083795]
    w11t = [8.294313028544018e-9, 3.14295949256877e-8, 4.66364384606612e-8,
            4.2217788428882957e-8, 0.7041478436885059, -1.9066288844162577e-6,
            0.13029927740061617, 6.232507885707777e-9, 7.23844225628642e-8,
            1.1681136028255033e-8, 3.3529594725190564e-9, -0.007526966693933902,
            -0.0919960569729057, -2.1008390426730607e-8, -0.22047462777076263,
            0.13555195839705692, 1.3053562539444425e-7, 8.78955317178912e-9,
            1.1658643395775117e-7, 2.1447924514401987e-8]
    w12t = [1.353457449307234e-8, 3.067936321700962e-8, 4.5506113132043295e-8,
            3.766167047439087e-8, 0.6948969601915064, -4.6980608921945596e-7,
            0.12550825609306346, 4.636054739015448e-9, 1.44053660285692e-7,
            1.3352313703787318e-8, 1.8407855566855775e-9, -0.007751569148802047,
            -0.0918827580349486, -1.3964964769825242e-8, -0.22036479617763619,
            0.1290428540202922, 1.4615408947070008e-6, 8.741820860292878e-9,
            0.020549749651431116, 2.562889662689068e-8]
    w13t = [0.0, -0.0, -0.0, -0.0, 0.4499999999999999, 0.2, -0.0, -0.0, 0.0, 0.0, -0.0,
            -0.0, -0.0, 0.0, -0.0, -0.0, 0.0, 0.0, 0.0, 0.0]
    w14t = [1.4841664254667114e-7, 1.4523270230804128e-7, 1.1006646928928847e-7,
            2.8976436632931236e-5, 0.7945010005677562, -8.010674222730887e-7,
            0.13541262036041035, 7.040831961577511e-8, 3.0328423353475175e-7,
            1.2638676107031148e-7, 4.856991411160282e-8, -0.01850575036302242,
            -0.07247815112304379, -3.213111299921966e-8, -0.22901274003266947,
            0.018508943131945278, 1.11766918571744e-6, 1.0303837648505954e-7,
            0.02154348972416336, 2.714237581422936e-7]
    w15t = [0.0, -0.0, 0.0, -0.0, 0.96, -0.0, 0.0, 0.0, -0.0, 0.0, 0.0, -0.0,
            -0.08833197273361398, -0.0, -0.2316680272663859, 0.0, 0.010000000000000064,
            -0.0, 0.0, 0.0]
    w16t = [8.241834070562598e-8, 1.7134143221840816e-7, 1.4686417142284546e-7,
            1.0617361383248306e-7, 0.7892644488069224, -6.961127387795805e-7,
            0.1389928181285743, 4.473233692765534e-8, 7.000035135188417e-7,
            1.172414320752293e-7, 1.9635728131616883e-8, -2.7725136618988303e-7,
            -0.08119760259573504, -7.55243449473203e-8, -0.2387992469253642,
            0.016279807695197324, 0.01779963880763364, 3.963118835550958e-8,
            0.007659536482345969, 2.2044711844497758e-7]
    w17t = [-0.0, -0.0, -0.0, -0.0, 0.97, -0.0, 0.0, -0.0, 0.0, -0.0, -0.0, -0.0, -0.0,
            -0.0, -0.32, 0.0, 0.0, -0.0, 0.0, 0.0]
    w18t = [1.1108248516742848e-7, 1.2362553314973753e-7, 8.291777803238013e-8,
            4.3030275502086306e-5, 0.7977223566406112, -1.278864708404284e-6,
            0.1468047225280131, 6.847903750975949e-8, 1.7342972345479683e-7,
            9.529193263012322e-8, 4.8908117815923066e-8, -0.020984406970518347,
            -0.06870881715945286, -8.772960704310079e-8, -0.23030301823355667,
            0.02542570524148051, 3.500412381090025e-7, 8.621439298205231e-8,
            4.70778843627457e-7, 1.8350315389648818e-7]
    w19t = [0.0, 0.0, 0.0, 0.0, 0.816497810804542, -0.0, 0.15350218919545805, 0.0, 0.0, 0.0,
            0.0, -0.0, -0.08567577370570562, 0.0, -0.2343242262942946, 0.0, 0.0, 0.0, 0.0,
            0.0]
    w20t = [3.829303656333224e-8, 5.211392790356772e-8, 7.261503009983882e-8,
            5.632652722566254e-8, 0.8106947984384285, -4.847910993847807e-7,
            0.15136908878038238, 8.049341088759719e-8, 0.00021027201498456663,
            4.413991557538945e-8, 2.2613849312151955e-8, -1.2001855827674826e-7,
            -0.08195188135016043, -4.94779445821025e-8, -0.2380464436839114,
            0.007723449702010213, 2.7173950970348796e-7, 2.747835567468853e-8,
            5.674159128684433e-7, 1.3715639292334853e-7]
    w21t = [1.3563709018845515e-9, 2.04421924461791e-8, 0.0022625126293570152,
            1.4306321988575245e-7, 0.2734983202574751, -0.10965529913244607,
            0.03892353698113018, 0.018194989301722808, 5.899334060155333e-9,
            1.0608744413741611e-8, 0.03970068699329653, -0.05489428920309655,
            -0.028441439842784383, 6.841180695444902e-10, -0.077008929520643,
            0.10494492437645861, 0.1462752316809572, 6.829391732507569e-8,
            0.18619935149422748, 1.5363644814229243e-7]
    w22t = [2.1497220872992086e-10, 0.00880701792467768, 0.02939704873532587,
            0.01902269441946229, 0.28289406199758904, -0.12224670576120043,
            0.04353925699909916, 0.04156276109954055, 1.309782594866215e-7,
            1.4902938771277945e-7, 0.09605820880040905, -0.048556264213814794,
            -0.02920582293777555, 1.0556998337432043e-9, -0.06999106974458105,
            0.11603391473132217, 0.046410776994123236, 0.04812947035730913,
            0.0761684103300507, 0.001975958990143702]
    w23t = [1.5021228401414956e-10, 4.9511757583987165e-11, 5.504677745110165e-11,
            -0.1499999969655728, 5.3466077081909765e-11, 8.546144964443675e-11,
            3.4466859511859894e-11, 1.3158602347597899e-11, 2.370455220660836e-10,
            0.6899999951538286, 1.4253029249609113e-11, 4.9005006990537455e-11,
            6.940911153353816e-11, 1.1440050515382358e-10, 6.61289087915012e-11,
            5.662765621494298e-11, 2.827591009354492e-10, 4.062829614873586e-11,
            3.2470973773948236e-10, 1.154510705546119e-10]
    w24t = [6.120770021211894e-8, 1.6354124623821429e-6, -0.00014073619896366336,
            -1.5382765037330814e-5, 0.2080138445872139, -1.2971430049535733e-7,
            0.018156770727218413, 2.2037258514540825e-7, 6.723276350650806e-8,
            6.337618763810101e-8, 0.1403878484984017, -0.04511790824984821,
            -0.005307605149301033, 3.927932051978494e-8, -0.058853373767992184,
            0.06663138474637213, 7.950726728448752e-7, 0.12731744850448984,
            0.08892422332791322, 7.33500141477371e-7]
    w25t = [-0.0, -0.0, 0.0, 0.0, 0.47051750921911656, -0.27, -0.0, -0.0, -0.0, 0.0, -0.0,
            -0.0, -0.0, -0.0, -0.0, 0.16698249078088656, 0.17249999999999863, 0.0, -0.0,
            0.0]
    w26t = [-1.4974268896904724e-9, 2.1299660069719237e-5, 7.138328419233795e-8,
            5.556498546627508e-8, 0.1956610520113446, -0.031893375562866595,
            0.0012055689244354576, 3.853805211331286e-6, 1.745285420148969e-7,
            0.0005851792883326295, 0.1627324904413541, -8.676390111160785e-7,
            -0.00021822659434104827, -3.519333880318415e-8, -0.00027353974022040813,
            0.0775405561417694, 0.13331690518163633, 0.0012622121417600453,
            1.8421735137028658e-7, 5.644293712819668e-5]
    w27t = [-0.0, -0.0, 0.0, 0.0, 0.81, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0,
            -0.0, -0.2700000000000002, -0.0, 0.0, 0.0, 0.0, 2.607011823064403e-16]
    w28t = [3.7372455923785476e-8, 4.094652753242599e-7, 5.7415100578892254e-8,
            3.401420963990992e-8, 0.1986252065725694, -8.486609959408817e-8,
            0.02395116116714053, 1.8655603296624098e-7, 3.192609402225675e-8,
            4.192967048756701e-8, 0.18694537859607632, -0.05562617637132034,
            -2.1532198587287727e-8, 1.6743729639447377e-8, -0.05376684092336866,
            0.08378904151649995, 8.526537485818498e-8, 0.1560810784728238,
            9.345153848513423e-8, 2.632283952728975e-7]
    w29t = [3.4283567996084206e-13, 3.7990822572287465e-13, 3.7142600801290104e-13,
            3.70503543736892e-13, 0.35749479955024777, -0.2699999981703773,
            4.517813144165653e-13, 4.152307197389584e-13, 3.8325919009880227e-13,
            3.8641173847782796e-13, 4.226784154498111e-13, 2.408333518927719e-13,
            2.1260972137250578e-13, 3.654487441426555e-13, 2.1222999600239372e-13,
            0.12305942548004342, 0.32944577313432416, 4.12098690158791e-13,
            4.102519513097008e-13, 3.846261776058408e-13]
    w30t = [1.4803515151077643e-8, -0.0006879584023924411, 4.2888040863450947e-8,
            2.2929765196701626e-8, 0.1795493808424352, -0.03138600771274758,
            -2.2584024525876738e-5, 4.9272317910895146e-8, 2.3423822987329342e-8,
            0.0009506016249741461, 0.11355195250620045, -1.0293753264822257e-6,
            -0.001573453741525871, -8.840766922004543e-9, -0.012141047904287173,
            0.07393091023345466, 0.21614904295648268, 0.0008895717027130405,
            2.41803868760394e-7, 0.0007902350139812155]
    w31t = [1.6340056117624302e-9, 1.9565866790646583e-9, 3.3555684076755637e-9,
            2.787614713976566e-9, 2.9431115157487124e-7, -6.460943820285891e-9,
            0.6899996361516727, -1.684144492089768e-10, 2.3533328091781668e-9,
            3.987731658411245e-10, -3.5443192099261323e-10, -4.124148718389558e-9,
            -0.41999985908147647, -1.698608436071608e-9, -8.37359295713311e-8,
            5.1519422412486e-9, 3.6143905696730985e-9, 1.3823066718871278e-10,
            2.5039395003112434e-9, 1.266744619081732e-9]
    w32t = [7.566327322459596e-9, 4.825217563466855e-9, 7.690746685595653e-9,
            5.160718596762832e-9, 0.5399993625561607, -4.504209798563395e-9,
            4.1040653527350213e-7, 2.9464764102897943e-10, 2.0574286015166503e-8,
            2.563382637201182e-9, -2.87407645308971e-12, -5.704202616324874e-9,
            -0.4199997323304285, 5.357085337765291e-10, -1.7997228500301755e-7,
            1.1943321213600445e-8, 2.3471857991397815e-7, 1.4873439634158403e-9,
            0.14999984680182982, 5.389194158287023e-9]
    w33t = [0.5475000004927487, 3.639348752267374e-11, 3.6297119207090926e-11,
            4.8359373913081175e-11, 3.6985951411054806e-11, -8.344032882552616e-12,
            4.95324666760957e-11, 4.9279769842503945e-11, -5.937463343784977e-11,
            1.5281960515915713e-11, 4.938309136779165e-11, 4.6783546679767e-11,
            -0.27750000071987474, -2.604035402019287e-11, 4.769815452319877e-11,
            3.594328886239922e-11, -7.450626539605664e-11, 3.696492713675703e-11,
            -8.601888405120908e-11, -7.494655171783378e-12]
    w34t = [3.816752554116215e-8, 2.4108245965184602e-8, 3.866893427126608e-8,
            2.6133023664679506e-8, 0.5399956074700397, -2.422700450765979e-8,
            3.121835229611604e-6, 9.83159866407741e-10, 1.0049811903316101e-7,
            1.2810619582366151e-8, -6.236703704368565e-10, -3.25171581876826e-8,
            -0.41999851237277913, 2.3482533865948143e-9, -1.0178638170004879e-6,
            6.135268722121618e-8, 1.1740060409904206e-6, 6.935175295465014e-9,
            0.14999934500728127, 2.728009375627522e-8]
    w35t = [0.0, 0.0, 0.0, 0.0, 0.5399999999999998, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            -0.42, 0.0, 0.0, 0.0, 0.0, 0.0, 0.15000000000000008, 0.0]
    w36t = [3.849468175432524e-8, 2.445316423070237e-8, 3.908919536507999e-8,
            2.6533158522369478e-8, 0.5399956707625192, -2.4680120813639767e-8,
            3.060459441432979e-6, 1.0494725271113753e-9, 1.0051942097386338e-7,
            1.2990342396495733e-8, -5.857340820358566e-10, -3.3070776790323165e-8,
            -0.41999853487787814, 2.2724790965929263e-9, -9.888639567736495e-7,
            6.163032720726743e-8, 1.152322725585961e-6, 7.076234095294047e-9,
            0.14999935677040055, 2.7654903692297257e-8]
    w37t = [2.7755575615628914e-16, 0.0, 0.0, 0.0, 0.0, 0.0, 0.69, 0.0, 0.0, 0.0,
            1.3322676295501878e-15, 0.0, -0.42, 0.0, 1.1102230246251565e-16, 0.0, 0.0, 0.0,
            0.0, 0.0]
    w38t = [3.926524175521384e-8, 4.7775819969648566e-8, 8.002238487731654e-8,
            6.600835291108059e-8, 5.8667901329586775e-6, -1.6231482847682396e-7,
            0.6899922382160447, -4.707622972579072e-9, 5.486166163459034e-8,
            9.12192394369125e-9, -9.740109441953985e-9, -1.1847727451586164e-7,
            -0.41999545369718655, -4.5654871187307614e-8, -2.9143969717920704e-6,
            1.241988618978603e-7, 8.762106045160896e-8, 2.996895687772333e-9,
            6.051390905066756e-8, 3.159657507633913e-8]
    w39t = [5.551115123125783e-16, 0.0, 0.0, 0.0, 0.0, -5.551115123125783e-17, 0.69, 0.0,
            0.0, 0.0, 0.0, 0.0, -0.4200000000000001, 0.0, 1.609823385706477e-15, 0.0, 0.0,
            0.0, 0.0, 0.0]
    w40t = [3.951392432594568e-8, 4.805543516590613e-8, 8.043134711972166e-8,
            6.636881254430819e-8, 5.8786262903503805e-6, -1.6293660747398834e-7,
            0.6899922200206299, -4.6553498878000645e-9, 5.5131224962268105e-8,
            9.278398379036754e-9, -9.679229731696501e-9, -1.1895407248941689e-7,
            -0.4199954320943935, -4.573654367645275e-8, -2.9307137991371913e-6,
            1.2477757268334885e-7, 8.722378007378553e-8, 3.116428882240821e-9,
            6.01725330932627e-8, 3.2053618284583953e-8]

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
    @test isapprox(w19.weights, w19t, rtol = 1.0e-7)
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
