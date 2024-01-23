using COSMO, CSV, Clarabel, HiGHS, JuMP, LinearAlgebra, OrderedCollections, Pajarito,
      PortfolioOptimiser, Statistics, Test, TimeSeries, Logging, GLPK

prices = TimeArray(CSV.File("./test/assets/stock_prices.csv"); timestamp = :date)

rf = 1.0329^(1 / 252) - 1
l = 2.0

solvers = Dict(:PClGL => Dict(:solver => optimizer_with_attributes(Pajarito.Optimizer,
                                                                   "verbose" => false,
                                                                   "oa_solver" => optimizer_with_attributes(GLPK.Optimizer,
                                                                                                            MOI.Silent() => true),
                                                                   "conic_solver" => optimizer_with_attributes(Clarabel.Optimizer,
                                                                                                               "verbose" => false,
                                                                                                               "max_step_fraction" => 0.75))))
@testset "Network and Dendrogram Constraints" begin
    portfolio = Portfolio(; prices = prices, solvers = solvers)
    asset_statistics!(portfolio)

    rm = :SD
    obj = :Min_Risk
    CV = centrality_vector(portfolio, CorOpt(;))
    B_1 = connection_matrix(portfolio, CorOpt(;); method = :MST, steps = 1)
    L_A = cluster_matrix(portfolio, CorOpt(;); linkage = :ward)

    w1 = optimise!(portfolio; obj = obj, rm = rm)

    portfolio.a_vec_cent = CV
    portfolio.b_cent = minimum(CV)
    w2 = optimise!(portfolio; obj = obj, rm = rm)

    portfolio.network_method = :IP
    portfolio.network_ip = B_1
    w3 = optimise!(portfolio; obj = obj, rm = rm)

    portfolio.network_method = :SDP
    portfolio.network_sdp = B_1
    w4 = optimise!(portfolio; obj = obj, rm = rm)

    portfolio.network_method = :IP
    portfolio.network_ip = L_A
    w5 = optimise!(portfolio; obj = obj, rm = rm)

    portfolio.network_method = :SDP
    portfolio.network_sdp = L_A
    w6 = optimise!(portfolio; obj = obj, rm = rm)

    portfolio.a_vec_cent = []
    portfolio.b_cent = Inf

    portfolio.network_method = :IP
    portfolio.network_ip = B_1
    w7 = optimise!(portfolio; obj = obj, rm = rm)

    portfolio.network_method = :SDP
    portfolio.network_sdp = B_1
    w8 = optimise!(portfolio; obj = obj, rm = rm)

    portfolio.network_method = :IP
    portfolio.network_ip = L_A
    w9 = optimise!(portfolio; obj = obj, rm = rm)

    portfolio.network_method = :SDP
    portfolio.network_sdp = L_A
    w10 = optimise!(portfolio; obj = obj, rm = rm)

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
    w3t = [0.0, 0.0, 0.08531373476784498, 0.0, 0.0, 0.0, 0.0, 0.2553257507386259, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.04299908481416786, 0.0, 0.0, 0.38591329855014755, 0.0,
           0.23044813112921367]
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
end
