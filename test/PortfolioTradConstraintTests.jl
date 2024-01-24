using COSMO, CSV, Clarabel, HiGHS, JuMP, LinearAlgebra, OrderedCollections, Pajarito,
      PortfolioOptimiser, Statistics, Test, TimeSeries, GLPK

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
    @test isapprox(w3.weights, w3t)
    @test isapprox(w4.weights, w4t)
    @test isapprox(w5.weights, w5t)
    @test isapprox(w6.weights, w6t)
    @test isapprox(w7.weights, w7t)
    @test isapprox(w8.weights, w8t)
    @test isapprox(w9.weights, w9t)
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
end
