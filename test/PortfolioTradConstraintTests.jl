using COSMO, CSV, Clarabel, Graphs, HiGHS, JuMP, LinearAlgebra, OrderedCollections,
      Pajarito, PortfolioOptimiser, Statistics, Test, TimeSeries, GLPK

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
@testset "Network and Dendrogram Constraints $(:SD)" begin
    portfolio = Portfolio(; prices = prices, solvers = solvers)
    asset_statistics!(portfolio; calc_kurt = false)

    rm = :SD
    obj = :Min_Risk
    CV = centrality_vector(portfolio, CorOpt(;))
    B_1 = connection_matrix(portfolio, CorOpt(;); method = :MST, steps = 1)
    L_A = cluster_matrix(portfolio, CorOpt(;); linkage = :ward)

    w1 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio.a_vec_cent = CV
    portfolio.b_cent = minimum(CV)
    w2 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio.network_method = :IP
    portfolio.network_ip = B_1
    w3 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio.network_method = :SDP
    portfolio.network_sdp = B_1
    w4 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio.network_method = :IP
    portfolio.network_ip = L_A
    w5 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio.network_method = :SDP
    portfolio.network_sdp = L_A
    w6 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio.a_vec_cent = []
    portfolio.b_cent = Inf

    portfolio.network_method = :IP
    portfolio.network_ip = B_1
    w7 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio.network_method = :SDP
    portfolio.network_sdp = B_1
    w8 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio.network_method = :IP
    portfolio.network_ip = L_A
    w9 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio.network_method = :SDP
    portfolio.network_sdp = L_A
    w10 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio = Portfolio(; prices = prices, solvers = solvers)
    asset_statistics!(portfolio; calc_kurt = false)

    rm = :SD
    obj = :Utility
    CV = centrality_vector(portfolio, CorOpt(;); method = :TMFG)
    B_1 = connection_matrix(portfolio, CorOpt(;); method = :TMFG, steps = 1)
    L_A = cluster_matrix(portfolio, CorOpt(;); linkage = :DBHT)

    w11 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio.a_vec_cent = CV
    portfolio.b_cent = minimum(CV)
    w12 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio.network_method = :IP
    portfolio.network_ip = B_1
    w13 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio.network_method = :SDP
    portfolio.network_sdp = B_1
    w14 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio.network_method = :IP
    portfolio.network_ip = L_A
    w15 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio.network_method = :SDP
    portfolio.network_sdp = L_A
    w16 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio.a_vec_cent = []
    portfolio.b_cent = Inf

    portfolio.network_method = :IP
    portfolio.network_ip = B_1
    w17 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio.network_method = :SDP
    portfolio.network_sdp = B_1
    w18 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio.network_method = :IP
    portfolio.network_ip = L_A
    w19 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio.network_method = :SDP
    portfolio.network_sdp = L_A
    w20 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio = Portfolio(; prices = prices, solvers = solvers)
    asset_statistics!(portfolio; calc_kurt = false)

    rm = :SD
    obj = :Sharpe
    CV = centrality_vector(portfolio, CorOpt(;); method = :TMFG)
    B_1 = connection_matrix(portfolio, CorOpt(;); method = :TMFG, steps = 1)
    L_A = cluster_matrix(portfolio, CorOpt(;); linkage = :DBHT)

    w21 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio.a_vec_cent = CV
    portfolio.b_cent = minimum(CV)
    w22 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio.network_method = :IP
    portfolio.network_ip = B_1
    w23 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio.network_method = :SDP
    portfolio.network_sdp = B_1
    w24 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio.network_method = :IP
    portfolio.network_ip = L_A
    w25 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio.network_method = :SDP
    portfolio.network_sdp = L_A
    w26 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio.a_vec_cent = []
    portfolio.b_cent = Inf

    portfolio.network_method = :IP
    portfolio.network_ip = B_1
    w27 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio.network_method = :SDP
    portfolio.network_sdp = B_1
    w28 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio.network_method = :IP
    portfolio.network_ip = L_A
    w29 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio.network_method = :SDP
    portfolio.network_sdp = L_A
    w30 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio = Portfolio(; prices = prices, solvers = solvers)
    asset_statistics!(portfolio; calc_kurt = false)

    rm = :SD
    obj = :Max_Ret
    CV = centrality_vector(portfolio, CorOpt(;); method = :TMFG)
    B_1 = connection_matrix(portfolio, CorOpt(;); method = :TMFG, steps = 1)
    L_A = cluster_matrix(portfolio, CorOpt(;); linkage = :DBHT)

    w31 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio.a_vec_cent = CV
    portfolio.b_cent = minimum(CV)
    w32 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio.network_method = :IP
    portfolio.network_ip = B_1
    w33 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio.network_method = :SDP
    portfolio.network_sdp = B_1
    w34 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio.network_method = :IP
    portfolio.network_ip = L_A
    w35 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio.network_method = :SDP
    portfolio.network_sdp = L_A
    w36 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio.a_vec_cent = []
    portfolio.b_cent = Inf

    portfolio.network_method = :IP
    portfolio.network_ip = B_1
    w37 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio.network_method = :SDP
    portfolio.network_sdp = B_1
    w38 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio.network_method = :IP
    portfolio.network_ip = L_A
    w39 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio.network_method = :SDP
    portfolio.network_sdp = L_A
    w40 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

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
    CV = centrality_vector(portfolio, CorOpt(;))
    B_1 = connection_matrix(portfolio, CorOpt(;); method = :MST, steps = 1)
    L_A = cluster_matrix(portfolio, CorOpt(;); linkage = :ward)

    w1 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio.a_vec_cent = CV
    portfolio.b_cent = minimum(CV)
    w2 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio.network_method = :IP
    portfolio.network_ip = B_1
    w3 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio.network_method = :SDP
    portfolio.network_sdp = B_1
    w4 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio.network_method = :IP
    portfolio.network_ip = L_A
    w5 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio.network_method = :SDP
    portfolio.network_sdp = L_A
    w6 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio.a_vec_cent = []
    portfolio.b_cent = Inf

    portfolio.network_method = :IP
    portfolio.network_ip = B_1
    w7 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio.network_method = :SDP
    portfolio.network_sdp = B_1
    w8 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio.network_method = :IP
    portfolio.network_ip = L_A
    w9 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio.network_method = :SDP
    portfolio.network_sdp = L_A
    w10 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio = Portfolio(; prices = prices, solvers = solvers)
    asset_statistics!(portfolio; calc_kurt = false)
    portfolio.short = true
    portfolio.short_u = 0.32
    portfolio.long_u = 0.97
    ssl2 = portfolio.sum_short_long

    rm = :SD
    obj = :Utility
    CV = centrality_vector(portfolio, CorOpt(;); method = :TMFG)
    B_1 = connection_matrix(portfolio, CorOpt(;); method = :TMFG, steps = 1)
    L_A = cluster_matrix(portfolio, CorOpt(;); linkage = :DBHT)

    w11 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio.a_vec_cent = CV
    portfolio.b_cent = minimum(CV)
    w12 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio.network_method = :IP
    portfolio.network_ip = B_1
    w13 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio.network_method = :SDP
    portfolio.network_sdp = B_1
    w14 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio.network_method = :IP
    portfolio.network_ip = L_A
    w15 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio.network_method = :SDP
    portfolio.network_sdp = L_A
    w16 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio.a_vec_cent = []
    portfolio.b_cent = Inf

    portfolio.network_method = :IP
    portfolio.network_ip = B_1
    w17 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio.network_method = :SDP
    portfolio.network_sdp = B_1
    w18 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio.network_method = :IP
    portfolio.network_ip = L_A
    w19 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio.network_method = :SDP
    portfolio.network_sdp = L_A
    w20 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio = Portfolio(; prices = prices, solvers = solvers)
    asset_statistics!(portfolio; calc_kurt = false)
    portfolio.short = true
    portfolio.short_u = 0.27
    portfolio.long_u = 0.81
    ssl3 = portfolio.sum_short_long

    rm = :SD
    obj = :Sharpe
    CV = centrality_vector(portfolio, CorOpt(;); method = :TMFG)
    B_1 = connection_matrix(portfolio, CorOpt(;); method = :TMFG, steps = 1)
    L_A = cluster_matrix(portfolio, CorOpt(;); linkage = :DBHT)

    w21 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio.a_vec_cent = CV
    portfolio.b_cent = minimum(CV)
    w22 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio.network_method = :IP
    portfolio.network_ip = B_1
    w23 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio.network_method = :SDP
    portfolio.network_sdp = B_1
    w24 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio.network_method = :IP
    portfolio.network_ip = L_A
    w25 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio.network_method = :SDP
    portfolio.network_sdp = L_A
    w26 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio.a_vec_cent = []
    portfolio.b_cent = Inf

    portfolio.network_method = :IP
    portfolio.network_ip = B_1
    w27 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio.network_method = :SDP
    portfolio.network_sdp = B_1
    w28 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio.network_method = :IP
    portfolio.network_ip = L_A
    w29 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio.network_method = :SDP
    portfolio.network_sdp = L_A
    w30 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio = Portfolio(; prices = prices, solvers = solvers)
    asset_statistics!(portfolio; calc_kurt = false)
    portfolio.short = true
    portfolio.short_u = 0.42
    portfolio.long_u = 0.69
    ssl4 = portfolio.sum_short_long

    rm = :SD
    obj = :Max_Ret
    CV = centrality_vector(portfolio, CorOpt(;); method = :TMFG)
    B_1 = connection_matrix(portfolio, CorOpt(;); method = :TMFG, steps = 1)
    L_A = cluster_matrix(portfolio, CorOpt(;); linkage = :DBHT)

    w31 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio.a_vec_cent = CV
    portfolio.b_cent = minimum(CV)
    w32 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio.network_method = :IP
    portfolio.network_ip = B_1
    w33 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio.network_method = :SDP
    portfolio.network_sdp = B_1
    w34 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio.network_method = :IP
    portfolio.network_ip = L_A
    w35 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio.network_method = :SDP
    portfolio.network_sdp = L_A
    w36 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio.a_vec_cent = []
    portfolio.b_cent = Inf

    portfolio.network_method = :IP
    portfolio.network_ip = B_1
    w37 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio.network_method = :SDP
    portfolio.network_sdp = B_1
    w38 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio.network_method = :IP
    portfolio.network_ip = L_A
    w39 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio.network_method = :SDP
    portfolio.network_sdp = L_A
    w40 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

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

    @test isapprox(w31.weights, w31t)

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

@testset "Network and Dendrogram Constraints $(:CDaR)" begin
    portfolio = Portfolio(; prices = prices, solvers = solvers)
    asset_statistics!(portfolio; calc_kurt = false)

    rm = :CDaR
    obj = :Min_Risk
    CV = centrality_vector(portfolio, CorOpt(;); method = :TMFG)
    B_1 = connection_matrix(portfolio, CorOpt(;); method = :TMFG, steps = 1)
    L_A = cluster_matrix(portfolio, CorOpt(;); linkage = :DBHT)

    w1 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio.a_vec_cent = CV
    portfolio.b_cent = minimum(CV)
    w2 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio.network_method = :IP
    portfolio.network_ip = B_1
    w3 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio.network_method = :SDP
    portfolio.network_sdp = B_1
    w4 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio.network_method = :IP
    portfolio.network_ip = L_A
    w5 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio.network_method = :SDP
    portfolio.network_sdp = L_A
    w6 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio.a_vec_cent = []
    portfolio.b_cent = Inf

    portfolio.network_method = :IP
    portfolio.network_ip = B_1
    w7 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio.network_method = :SDP
    portfolio.network_sdp = B_1
    w8 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio.network_method = :IP
    portfolio.network_ip = L_A
    w9 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio.network_method = :SDP
    portfolio.network_sdp = L_A
    w10 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio = Portfolio(; prices = prices, solvers = solvers)
    asset_statistics!(portfolio; calc_kurt = false)

    rm = :CDaR
    obj = :Utility
    CV = centrality_vector(portfolio, CorOpt(;); method = :TMFG)
    B_1 = connection_matrix(portfolio, CorOpt(;); method = :TMFG, steps = 1)
    L_A = cluster_matrix(portfolio, CorOpt(;); linkage = :DBHT)

    w11 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio.a_vec_cent = CV
    portfolio.b_cent = minimum(CV)
    w12 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio.network_method = :IP
    portfolio.network_ip = B_1
    w13 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio.network_method = :SDP
    portfolio.network_sdp = B_1
    w14 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio.network_method = :IP
    portfolio.network_ip = L_A
    w15 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio.network_method = :SDP
    portfolio.network_sdp = L_A
    w16 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio.a_vec_cent = []
    portfolio.b_cent = Inf

    portfolio.network_method = :IP
    portfolio.network_ip = B_1
    w17 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio.network_method = :SDP
    portfolio.network_sdp = B_1
    w18 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio.network_method = :IP
    portfolio.network_ip = L_A
    w19 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio.network_method = :SDP
    portfolio.network_sdp = L_A
    w20 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio = Portfolio(; prices = prices, solvers = solvers)
    asset_statistics!(portfolio; calc_kurt = false)

    rm = :CDaR
    obj = :Sharpe
    CV = centrality_vector(portfolio, CorOpt(;); method = :TMFG)
    B_1 = connection_matrix(portfolio, CorOpt(;); method = :TMFG, steps = 1)
    L_A = cluster_matrix(portfolio, CorOpt(;); linkage = :DBHT)

    w21 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio.a_vec_cent = CV
    portfolio.b_cent = minimum(CV)
    w22 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio.network_method = :IP
    portfolio.network_ip = B_1
    w23 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio.network_method = :SDP
    portfolio.network_sdp = B_1
    w24 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio.network_method = :IP
    portfolio.network_ip = L_A
    w25 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio.network_method = :SDP
    portfolio.network_sdp = L_A
    w26 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio.a_vec_cent = []
    portfolio.b_cent = Inf

    portfolio.network_method = :IP
    portfolio.network_ip = B_1
    w27 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio.network_method = :SDP
    portfolio.network_sdp = B_1
    w28 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio.network_method = :IP
    portfolio.network_ip = L_A
    w29 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio.network_method = :SDP
    portfolio.network_sdp = L_A
    w30 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio = Portfolio(; prices = prices, solvers = solvers)
    asset_statistics!(portfolio; calc_kurt = false)

    rm = :CDaR
    obj = :Max_Ret
    CV = centrality_vector(portfolio, CorOpt(;); method = :TMFG)
    B_1 = connection_matrix(portfolio, CorOpt(;); method = :TMFG, steps = 1)
    L_A = cluster_matrix(portfolio, CorOpt(;); linkage = :DBHT)

    w31 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio.a_vec_cent = CV
    portfolio.b_cent = minimum(CV)
    w32 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio.network_method = :IP
    portfolio.network_ip = B_1
    w33 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio.network_method = :SDP
    portfolio.network_sdp = B_1
    w34 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio.network_method = :IP
    portfolio.network_ip = L_A
    w35 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio.network_method = :SDP
    portfolio.network_sdp = L_A
    w36 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio.a_vec_cent = []
    portfolio.b_cent = Inf

    portfolio.network_method = :IP
    portfolio.network_ip = B_1
    w37 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio.network_method = :SDP
    portfolio.network_sdp = B_1
    w38 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio.network_method = :IP
    portfolio.network_ip = L_A
    w39 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio.network_method = :SDP
    portfolio.network_sdp = L_A
    w40 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio = Portfolio(; prices = prices, solvers = solvers)
    asset_statistics!(portfolio; calc_kurt = false)
    portfolio.network_penalty = 0.5

    rm = :CDaR
    obj = :Min_Risk
    CV = centrality_vector(portfolio, CorOpt(;); method = :TMFG)
    B_1 = connection_matrix(portfolio, CorOpt(;); method = :TMFG, steps = 1)
    L_A = cluster_matrix(portfolio, CorOpt(;); linkage = :DBHT)

    w41 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio.a_vec_cent = CV
    portfolio.b_cent = minimum(CV)
    w42 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio.network_method = :IP
    portfolio.network_ip = B_1
    w43 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio.network_method = :SDP
    portfolio.network_sdp = B_1
    w44 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio.network_method = :IP
    portfolio.network_ip = L_A
    w45 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio.network_method = :SDP
    portfolio.network_sdp = L_A
    w46 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio.a_vec_cent = []
    portfolio.b_cent = Inf

    portfolio.network_method = :IP
    portfolio.network_ip = B_1
    w47 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio.network_method = :SDP
    portfolio.network_sdp = B_1
    w48 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio.network_method = :IP
    portfolio.network_ip = L_A
    w49 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio.network_method = :SDP
    portfolio.network_sdp = L_A
    w50 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio = Portfolio(; prices = prices, solvers = solvers)
    asset_statistics!(portfolio; calc_kurt = false)
    portfolio.network_penalty = 0.5

    rm = :CDaR
    obj = :Utility
    CV = centrality_vector(portfolio, CorOpt(;); method = :TMFG)
    B_1 = connection_matrix(portfolio, CorOpt(;); method = :TMFG, steps = 1)
    L_A = cluster_matrix(portfolio, CorOpt(;); linkage = :DBHT)

    w51 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio.a_vec_cent = CV
    portfolio.b_cent = minimum(CV)
    w52 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio.network_method = :IP
    portfolio.network_ip = B_1
    w53 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio.network_method = :SDP
    portfolio.network_sdp = B_1
    w54 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio.network_method = :IP
    portfolio.network_ip = L_A
    w55 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio.network_method = :SDP
    portfolio.network_sdp = L_A
    w56 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio.a_vec_cent = []
    portfolio.b_cent = Inf

    portfolio.network_method = :IP
    portfolio.network_ip = B_1
    w57 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio.network_method = :SDP
    portfolio.network_sdp = B_1
    w58 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio.network_method = :IP
    portfolio.network_ip = L_A
    w59 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio.network_method = :SDP
    portfolio.network_sdp = L_A
    w60 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio = Portfolio(; prices = prices, solvers = solvers)
    asset_statistics!(portfolio; calc_kurt = false)
    portfolio.network_penalty = 0.5

    rm = :CDaR
    obj = :Sharpe
    CV = centrality_vector(portfolio, CorOpt(;); method = :TMFG)
    B_1 = connection_matrix(portfolio, CorOpt(;); method = :TMFG, steps = 1)
    L_A = cluster_matrix(portfolio, CorOpt(;); linkage = :DBHT)

    w61 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio.a_vec_cent = CV
    portfolio.b_cent = minimum(CV)
    w62 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio.network_method = :IP
    portfolio.network_ip = B_1
    w63 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio.network_method = :SDP
    portfolio.network_sdp = B_1
    w64 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio.network_method = :IP
    portfolio.network_ip = L_A
    w65 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio.network_method = :SDP
    portfolio.network_sdp = L_A
    w66 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio.a_vec_cent = []
    portfolio.b_cent = Inf

    portfolio.network_method = :IP
    portfolio.network_ip = B_1
    w67 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio.network_method = :SDP
    portfolio.network_sdp = B_1
    w68 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio.network_method = :IP
    portfolio.network_ip = L_A
    w69 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio.network_method = :SDP
    portfolio.network_sdp = L_A
    w70 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio = Portfolio(; prices = prices, solvers = solvers)
    asset_statistics!(portfolio; calc_kurt = false)
    portfolio.network_penalty = 0.5

    rm = :CDaR
    obj = :Max_Ret
    CV = centrality_vector(portfolio, CorOpt(;); method = :TMFG)
    B_1 = connection_matrix(portfolio, CorOpt(;); method = :TMFG, steps = 1)
    L_A = cluster_matrix(portfolio, CorOpt(;); linkage = :DBHT)

    w71 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio.a_vec_cent = CV
    portfolio.b_cent = minimum(CV)
    w72 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio.network_method = :IP
    portfolio.network_ip = B_1
    w73 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio.network_method = :SDP
    portfolio.network_sdp = B_1
    w74 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio.network_method = :IP
    portfolio.network_ip = L_A
    w75 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio.network_method = :SDP
    portfolio.network_sdp = L_A
    w76 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio.a_vec_cent = []
    portfolio.b_cent = Inf

    portfolio.network_method = :IP
    portfolio.network_ip = B_1
    w77 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio.network_method = :SDP
    portfolio.network_sdp = B_1
    w78 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio.network_method = :IP
    portfolio.network_ip = L_A
    w79 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

    portfolio.network_method = :SDP
    portfolio.network_sdp = L_A
    w80 = optimise!(portfolio; obj = obj, rm = rm, l = l, rf = rf)

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
