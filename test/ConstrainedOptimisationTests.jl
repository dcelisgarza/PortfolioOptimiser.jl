using CSV, TimeSeries, DataFrames, StatsBase, Statistics, LinearAlgebra, Test, Pajarito,
      JuMP, Clarabel, PortfolioOptimiser, HiGHS

prices = TimeArray(CSV.File("./assets/stock_prices.csv"); timestamp = :date)
rf = 1.0329^(1 / 252) - 1
l = 2.0

@testset "Network and Dendrogram SD" begin
    portfolio = Portfolio(; prices = prices,
                          solvers = Dict(:PClGL => Dict(:solver => optimizer_with_attributes(Pajarito.Optimizer,
                                                                                             "verbose" => false,
                                                                                             "oa_solver" => optimizer_with_attributes(HiGHS.Optimizer,
                                                                                                                                      MOI.Silent() => true),
                                                                                             "conic_solver" => optimizer_with_attributes(Clarabel.Optimizer,
                                                                                                                                         "verbose" => false,
                                                                                                                                         "max_step_fraction" => 0.75)))))
    asset_statistics!(portfolio)
    wc_statistics!(portfolio,
                   WCType(; box = NormalWC(; seed = 123456789),
                          ellipse = NormalWC(; seed = 123456789)))

    A = centrality_vector(portfolio)
    B = connection_matrix(portfolio)
    C = cluster_matrix(portfolio)

    rm = SD()
    obj = MinRisk()
    w1 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [0.00791850924098073, 0.030672065216453506, 0.010501402809199123,
          0.027475241969187995, 0.012272329269527841, 0.03339587076426262,
          1.4321532289072258e-6, 0.13984297866711365, 2.4081081597353397e-6,
          5.114425959766348e-5, 0.2878111114337346, 1.5306036912879562e-6,
          1.1917690994187655e-6, 0.12525446872321966, 6.630910273840812e-6,
          0.015078706008184504, 8.254970801614574e-5, 0.1930993918762575,
          3.0369340845779798e-6, 0.11652799957572683]
    @test isapprox(w1.weights, wt)

    wc1 = optimise!(portfolio; type = WC(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w1.weights, wc1.weights)

    portfolio.a_vec_cent = A
    portfolio.b_cent = minimum(A)
    w2 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [3.5426430661259544e-11, 0.07546661314221123, 0.021373033743425803,
          0.027539611826011074, 0.023741467255115698, 0.12266515815974924,
          2.517181275812244e-6, 0.22629893782442134, 3.37118211246211e-10,
          0.02416530860824232, 3.3950118687352606e-10, 1.742541411959769e-6,
          1.5253730188343444e-6, 5.304686980295978e-11, 0.024731789616991084,
          3.3711966852218057e-10, 7.767147353183488e-11, 0.30008056166009706,
          1.171415437554966e-10, 0.1539317317710031]
    @test isapprox(w2.weights, wt)

    portfolio.network_adj = IP(; A = B)
    w3 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [5.603687159886085e-13, 1.3820161690005612e-12, 1.5853880637229228e-12,
          1.33602425322658e-12, 0.07121859232520031, 1.119571110222061e-12,
          1.041437600224815e-13, 0.26930632966134743, 6.131616834263862e-13,
          1.0353405330340386e-12, 1.0102504371932946e-12, 1.3075720486452749e-12,
          1.1733675314147727e-13, 4.761261290370584e-13, 7.973896995771363e-13,
          5.188187662626333e-13, 5.626224717027035e-15, 0.41769670819791543,
          7.996868449487614e-13, 0.24177836980276793]
    @test isapprox(w3.weights, wt, rtol = 0.01)

    wc3 = optimise!(portfolio; type = WC(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w3.weights, wc3.weights, rtol = 0.01)

    portfolio.network_adj = SDP(; A = B)
    w4 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [4.2907851108373285e-11, 0.0754418669962984, 0.021396172655536325,
          0.027531481813488985, 0.02375148897766012, 0.12264432042703496,
          3.777946382632432e-6, 0.2263233585633904, 3.8909758570393176e-10,
          0.024184506959405477, 3.8945719747726095e-10, 2.597992728770223e-6,
          2.2916747013451583e-6, 5.942491440558677e-11, 0.02472298577516722,
          3.89238712058158e-10, 9.010863437870188e-11, 0.30009565874345184,
          1.32909841405546e-10, 0.15389948998160896]
    @test isapprox(w4.weights, wt)

    wc4 = optimise!(portfolio; type = WC(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w4.weights, wc4.weights)

    portfolio.network_adj = IP(; A = C)
    w5 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [2.600332585507959e-12, 2.49058898193321e-12, 2.5578162901552497e-12,
          1.5466046737425697e-12, 1.4967933865374272e-12, 3.0881407771151395e-13,
          4.123817493960925e-12, 3.252723921673227e-12, 1.632009812408571e-12,
          4.1272715410990253e-13, 1.3219081818309027e-12, 2.8170581035055335e-12,
          3.983969550512609e-12, 1.5360670864105517e-12, 0.05337095972065508,
          2.0129031207422168e-13, 8.038123010019403e-13, 0.5616133973621348,
          1.3058311743272652e-12, 0.38501564288481793]
    @test isapprox(w5.weights, wt)

    wc5 = optimise!(portfolio; type = WC(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w5.weights, wc5.weights)

    portfolio.network_adj = SDP(; A = C)
    w6 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [6.011468296535483e-11, 9.424732042613459e-6, 5.527537353284497e-6,
          2.903730060970158e-6, 4.265000229139505e-6, 5.05868769590198e-6,
          3.009791891015968e-6, 1.5268395403568086e-5, 5.657961006483616e-10,
          2.68155059410767e-6, 5.662844616891034e-10, 3.196842024458429e-6,
          2.419924540093777e-6, 8.983883474709306e-11, 0.05335383109918724,
          5.659272794304194e-10, 1.3052770418372395e-10, 0.5616352353647922,
          1.9587068208203376e-10, 0.38495717516982575]
    @test isapprox(w6.weights, wt)

    wc6 = optimise!(portfolio; type = WC(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w6.weights, wc6.weights)

    portfolio.a_vec_cent = []
    portfolio.b_cent = Inf

    portfolio.network_adj = IP(; A = B)
    w7 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [4.436079292890281e-13, 4.339657195356424e-13, 4.3353028423648065e-13,
          2.099632742038836e-14, 3.139151563704902e-13, 5.251815182737052e-13,
          9.708678618251658e-14, 7.40139180161491e-13, 2.82644625737616e-13,
          4.0939722177208954e-13, 0.43964473499450424, 6.45713115023492e-13,
          2.4339291393877985e-13, 4.979300009984306e-13, 9.848526731370591e-13,
          1.5980123190589848e-13, 4.0414638067893824e-13, 0.3300904730120406,
          3.428158408862516e-13, 0.2302647919864759]
    @test isapprox(w7.weights, wt)

    wc7 = optimise!(portfolio; type = WC(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w7.weights, wc7.weights, rtol = 0.0005)

    portfolio.network_adj = SDP(; A = B)
    w8 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [1.1008281068185902e-5, 0.055457116453828725, 0.012892903094396706,
          0.03284102649502972, 0.014979204379343122, 0.057886691097825904,
          1.0224197847100319e-6, 9.70550406073092e-6, 2.045810229626667e-6,
          0.012049795193184915, 0.3735114108030411, 1.330358085433759e-6,
          1.0648304534905729e-6, 9.906438750370314e-6, 0.007878565110236062,
          0.022521836037796082, 3.1895242150194783e-6, 0.26190144912467656,
          1.3462532236761872e-6, 0.14803938279076995]
    @test isapprox(w8.weights, wt, rtol = 5.0e-8)

    wc8 = optimise!(portfolio; type = WC(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w8.weights, wc8.weights)

    portfolio.network_adj = IP(; A = C)
    w9 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [1.0787207720863677e-12, 8.294709114613152e-14, 7.954607618722517e-13,
          3.3664271487071197e-13, 9.683130672209493e-13, 7.208016377361811e-13,
          5.283852777701427e-12, 4.1303200648255883e-13, 1.698528524551865e-12,
          8.083770937152382e-13, 0.6254998770683314, 2.693559025098044e-12,
          4.872090994188645e-12, 3.2453165851283277e-13, 1.0091301616417753e-12,
          0.06614823334410497, 0.30835188956514026, 2.463468634892428e-13,
          9.106489132241315e-13, 1.8060640077175145e-13]
    @test isapprox(w9.weights, wt)

    wc9 = optimise!(portfolio; type = WC(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w9.weights, wc9.weights)

    portfolio.network_adj = SDP(; A = C)
    w10 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [6.5767880723988875e-6, 4.623865766176694e-6, 3.533665790240184e-6,
          2.181035748649495e-6, 2.5215556700206228e-6, 1.858111674425038e-6,
          2.5125294721249436e-6, 2.7192384937770815e-6, 8.752492794952926e-7,
          9.63854668493286e-7, 0.6119910196342336, 2.4053903351549724e-6,
          9.311964129813141e-7, 9.706505481488379e-6, 1.3337324741660364e-5,
          0.05681677180817997, 4.294323401245429e-5, 1.2805451158943426e-5,
          1.6336794237581935e-6, 0.33108007988138427]
    @test isapprox(w10.weights, wt)

    wc10 = optimise!(portfolio; type = WC(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w10.weights, wc10.weights)

    portfolio = Portfolio(; prices = prices,
                          solvers = Dict(:PClGL => Dict(:solver => optimizer_with_attributes(Pajarito.Optimizer,
                                                                                             "verbose" => false,
                                                                                             "oa_solver" => optimizer_with_attributes(HiGHS.Optimizer,
                                                                                                                                      MOI.Silent() => true),
                                                                                             "conic_solver" => optimizer_with_attributes(Clarabel.Optimizer,
                                                                                                                                         "verbose" => false,
                                                                                                                                         "max_step_fraction" => 0.75)))))
    asset_statistics!(portfolio)
    obj = Sharpe(; rf = rf)

    A = centrality_vector(portfolio; network_type = TMFG())
    B = connection_matrix(portfolio; network_type = TMFG())
    C = cluster_matrix(portfolio; hclust_alg = DBHT())

    w11 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [5.2456519308516375e-9, 1.3711772920494845e-8, 1.828307590553689e-8,
          1.1104551657521415e-8, 0.5180586238310904, 1.7475553033958716e-9,
          0.06365101880957791, 1.1504018896467816e-8, 1.0491794751235226e-8,
          5.650081752041375e-9, 9.262635444153261e-9, 1.4051602963640009e-9,
          9.923106257589635e-10, 3.050812431838031e-9, 9.966076078173243e-10,
          0.1432679499627637, 0.19649701265872604, 1.0778757180749977e-8,
          0.07852527921167819, 1.1301377011579277e-8]
    @test isapprox(w11.weights, wt)

    wc11 = optimise!(portfolio; type = WC(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w11.weights, wc11.weights, rtol = 5.0e-6)

    portfolio.a_vec_cent = A
    portfolio.b_cent = minimum(A)

    w12 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [3.580964533638559e-11, 9.277053125069126e-10, 9.835000020421294e-10,
          0.48362433627647977, 1.3062621693742353e-9, 3.4185103123124565e-11,
          0.28435553885288534, 0.17984723049407092, 1.693432131576565e-10,
          2.621850918060399e-10, 0.05217287791621345, 5.161470450551339e-9,
          2.4492072342885186e-9, 4.790090083985224e-11, 2.6687808589346464e-9,
          1.0539691276230515e-9, 2.0827050059613154e-10, 8.48619997995145e-10,
          2.329804826102029e-10, 7.01603778985698e-11]
    @test isapprox(w12.weights, wt)

    wc12 = optimise!(portfolio; type = WC(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w12.weights, wc12.weights, rtol = 5.0e-5)

    portfolio.network_adj = IP(; A = B)
    w13 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [1.721507261658482e-11, 4.611331427286785e-11, 4.6537999918192326e-11,
          5.4302470941152905e-11, 4.508258159401719e-11, 2.510366932792115e-11,
          0.9999999992889718, 5.3842283750049385e-11, 9.81183158048394e-12,
          3.589613596601703e-11, 5.3385756391682404e-11, 4.8553725558375136e-11,
          4.299842533772006e-11, 1.755101403265066e-11, 4.285647343183246e-11,
          4.6446497580083914e-11, 2.1013218648897535e-11, 4.559832123949408e-11,
          3.102165424744005e-11, 2.7697895657875245e-11]
    @test isapprox(w13.weights, wt)

    wc13 = optimise!(portfolio; type = WC(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w13.weights, wc13.weights)

    portfolio.network_adj = SDP(; A = B)
    w14 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [5.024048627217141e-14, 1.5886394555170722e-12, 1.5924178422101003e-12,
          0.24554600920016964, 1.6332399340098677e-12, 1.0616523261700116e-13,
          0.0959435974734496, 0.2562392110969168, 2.7829378542014683e-13,
          4.877264338339266e-13, 0.402271160351581, 1.2141431002683076e-8,
          4.523802241069811e-9, 5.3729255370564864e-14, 5.20282725457044e-9,
          1.5947986553082236e-12, 3.43551721221936e-13, 1.5824894417145518e-12,
          3.8307499177570936e-13, 1.280439184666322e-13]
    @test isapprox(w14.weights, wt)

    wc14 = optimise!(portfolio; type = WC(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w14.weights, wc14.weights, rtol = 5.0e-5)

    portfolio.network_adj = IP(; A = C)
    w15 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [3.5488808869733083e-12, 8.281743679661074e-12, 8.80902177574244e-12,
          9.955072746064901e-12, 1.1134822019504797e-11, 4.7417114086610015e-12,
          0.6074274195608048, 1.0253105313724553e-11, 1.4148918790428058e-12,
          6.3499711667509184e-12, 0.39257258031345454, 9.215307590460059e-12,
          7.61409793255415e-12, 3.5761100061677828e-12, 7.3543478626708e-12,
          9.27354997599413e-12, 3.8374099382272584e-12, 8.458519745181501e-12,
          6.62153510029776e-12, 5.3005368572386036e-12]
    @test isapprox(w15.weights, wt, rtol = 1.0e-7)

    wc15 = optimise!(portfolio; type = WC(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w15.weights, wc15.weights, rtol = 0.005)

    portfolio.network_adj = SDP(; A = C)
    w16 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [1.645653643573042e-14, 5.24765733463725e-13, 5.259882343242254e-13,
          0.38135427736460104, 5.383168184665497e-13, 3.606530943112124e-14,
          2.659687073118818e-8, 2.387877721283499e-8, 9.190837103177029e-14,
          1.6141883247690533e-13, 0.6186456659519168, 3.0002908666789655e-9,
          1.5771657468870285e-9, 1.747936659731388e-14, 1.6271322639469356e-9,
          5.267534803071819e-13, 1.1362061261344631e-13, 5.231419643587768e-13,
          1.2683767418078167e-13, 4.256536051656617e-14]
    @test isapprox(w16.weights, wt)

    wc16 = optimise!(portfolio; type = WC(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w16.weights, wc16.weights, rtol = 5.0e-6)

    portfolio.a_vec_cent = []
    portfolio.b_cent = Inf

    portfolio.network_adj = IP(; A = B)
    w17 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [2.0966673826999653e-12, 2.198260339765037e-12, 1.9669534265126797e-12,
          2.0537353944645557e-12, 0.762282392772933, 2.61589719142697e-12,
          1.6397732202694488e-12, 2.573524529021814e-12, 2.1635532147945916e-12,
          2.3920718749431715e-12, 2.6031535089914636e-12, 2.254849729224801e-12,
          2.3351897528966937e-12, 2.5746406212787903e-12, 2.4339077077727048e-12,
          0.23771760718605286, 2.077497433772476e-12, 2.544464994811904e-12,
          2.2242733110585934e-12, 2.2657939494042705e-12]
    @test isapprox(w17.weights, wt)

    wc17 = optimise!(portfolio; type = WC(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w17.weights, wc17.weights, rtol = 0.0005)

    portfolio.network_adj = SDP(; A = B)
    w18 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [1.131030668352105e-7, 9.736849167799003e-7, 1.5073134798603708e-7,
          9.724848430006577e-8, 0.3050368084415617, 4.717356198432155e-8,
          0.03491168150833796, 1.1696087757981777e-6, 2.1133994869894878e-7,
          1.38898043984553e-7, 0.23158972602737993, 2.930159759606465e-8,
          1.841227833023016e-8, 1.3415748037100702e-7, 2.038375787580353e-8,
          0.10856264505102015, 2.399490931217591e-7, 0.2072142228794291,
          2.522693174355702e-7, 0.11268131983060002]
    @test isapprox(w18.weights, wt, rtol = 5.0e-8)

    wc18 = optimise!(portfolio; type = WC(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w18.weights, wc18.weights, rtol = 5.0e-5)

    portfolio.network_adj = IP(; A = C)
    w19 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [1.1904763791512141e-12, 1.2340531692956696e-12, 1.067454109369536e-12,
          1.127653866268004e-12, 0.617653720874792, 1.6368429734597877e-12,
          9.929978443062197e-13, 1.5549069160715418e-12, 1.0890925109349555e-12,
          1.3843599449712156e-12, 1.5689472056216027e-12, 1.400584981008052e-12,
          1.5941278979112304e-12, 1.5487191642726196e-12, 1.6035009203180108e-12,
          0.1719148482846596, 1.1171304302527388e-12, 1.5058581547861255e-12,
          0.210431430817618, 1.3135957217697496e-12]
    @test isapprox(w19.weights, wt)

    wc19 = optimise!(portfolio; type = WC(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w19.weights, wc19.weights, rtol = 0.005)

    portfolio.network_adj = SDP(; A = C)
    w20 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [1.9469265893415583e-7, 1.9264487242540467e-7, 2.232335248226593e-7,
          1.1246896197634285e-7, 0.40746626496017013, 8.182096647686593e-8,
          5.006408130895852e-8, 2.231759312172926e-7, 3.1463420211396267e-7,
          7.356649686989611e-6, 0.3255255880379563, 4.241919388356267e-8,
          2.851355716950413e-8, 2.082232132989454e-7, 3.1366090964049265e-8,
          0.15590260393465316, 0.010877836307400067, 2.4616586233596695e-7,
          0.10021813719562453, 2.6349139195481583e-7]
    @test isapprox(w20.weights, wt, rtol = 1.0e-5)

    wc20 = optimise!(portfolio; type = WC(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w20.weights, wc20.weights, rtol = 0.0001)
end

@testset "Network and Dendrogram SD short" begin
    portfolio = Portfolio(; prices = prices, short_budget = 10.0,
                          solvers = Dict(:PClGL => Dict(:check_sol => (allow_local = true,
                                                                       allow_almost = true),
                                                        :solver => optimizer_with_attributes(Pajarito.Optimizer,
                                                                                             "verbose" => false,
                                                                                             "oa_solver" => optimizer_with_attributes(HiGHS.Optimizer,
                                                                                                                                      MOI.Silent() => true),
                                                                                             "conic_solver" => optimizer_with_attributes(Clarabel.Optimizer,
                                                                                                                                         "verbose" => false,
                                                                                                                                         "max_step_fraction" => 0.75)))))
    asset_statistics!(portfolio)
    wc_statistics!(portfolio,
                   WCType(; box = NormalWC(; seed = 123456789),
                          ellipse = NormalWC(; seed = 123456789)))
    portfolio.short = true
    portfolio.short_budget = 0.22
    portfolio.short_u = 0.22
    portfolio.long_u = 0.88
    portfolio.budget = portfolio.long_u - portfolio.short_u

    A = centrality_vector(portfolio)
    B = connection_matrix(portfolio)
    C = cluster_matrix(portfolio)

    rm = SD()
    obj = MinRisk()
    w1 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [0.0023732456580253273, 0.02478417662625044, 0.011667587612994237,
          0.021835859830789766, 0.008241287060153578, 0.03550224737381907,
          -0.006408569798470374, 0.09320297606692711, -0.0072069893239688695,
          0.012269888650642718, 0.18722784970990347, -0.01395210113745165,
          -0.006026909793594887, 0.0962770551826336, 0.0005110433722419729,
          0.016784621725152556, 0.009614188864666265, 0.13436280943955117,
          -0.042344878358121694, 0.08128461123785628]
    @test isapprox(w1.weights, wt)

    wc1 = optimise!(portfolio; type = WC(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w1.weights, wc1.weights)

    portfolio.a_vec_cent = A
    portfolio.b_cent = minimum(A)
    w2 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [0.0064520166328966245, 0.022489884719278264, 0.01029468309633522,
          0.020902946113760288, 0.00711671166632188, 0.03337768007882898,
          -0.006266220580382531, 0.09139178524192762, -0.02237340947148962,
          0.010954134035283628, 0.18619094237390604, -0.014102818325104587,
          -0.0055880934328339325, 0.09574848799807273, 0.0002468303788802295,
          0.01712117146324995, 0.014137258176396965, 0.13074888415899583,
          -0.01805679382768158, 0.07921391950335803]
    @test isapprox(w2.weights, wt)

    portfolio.network_adj = IP(; A = B)
    w3 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [0.006062713502026987, -1.82444535940986e-11, -1.8668641666100138e-11,
          -1.8497828216051628e-11, -1.8790386852871776e-11, -1.8031722495722727e-11,
          -1.9061197724922234e-11, -1.7599957286747152e-11, -1.7049358857324967e-11,
          -1.8223294421307198e-11, 0.32181185959658903, -1.8908757892656348e-11,
          -1.9168697126217532e-11, -1.3928253949800599e-11, -1.8349718535583803e-11,
          -1.6677394168784684e-11, -6.267257105255323e-12, 0.3321254271795085,
          -2.669852770519402e-12, -1.79874967199412e-11]
    @test isapprox(w3.weights, wt, rtol = 0.5)

    wc3 = optimise!(portfolio; type = WC(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w3.weights, wc3.weights, rtol = 0.05)

    portfolio.network_adj = SDP(; A = B)
    w4 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [0.022589148447524018, 0.036812012135242114, 0.006833793601670234,
          0.013202059102917668, 0.005824555380015791, 0.03984269080339122,
          -0.006834168984417891, 1.9327468236390324e-5, -2.1177840385696868e-5,
          0.011500602752026409, 0.25265179781193275, -0.014411712546749403,
          -1.628444737600106e-5, 1.195721590601671e-5, 0.007515505990608158,
          0.019532808277500043, 4.833328704956932e-6, 0.16879897503212193,
          4.3131102914277385e-6, 0.09613896336083992]
    @test isapprox(w4.weights, wt)

    wc4 = optimise!(portfolio; type = WC(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w4.weights, wc4.weights)

    portfolio.network_adj = IP(; A = C)
    w5 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [-5.293989582770398e-11, -2.169360184726156e-11, -2.302603024389023e-11,
          -2.1745518271876716e-11, -2.3125965561871784e-11, -2.1701344936140427e-11,
          -2.296700740543876e-11, -1.9944292316184238e-11, -3.429967459242985e-11,
          -2.2415059313673134e-11, 0.34000000102143574, 0.017848743137050257,
          -2.3535780083971463e-11, -4.146383770552791e-11, -2.220301457889096e-11,
          -3.3471016659862616e-11, -6.139513723264243e-11, -1.9956735333122212e-11,
          -7.7723124936471e-11, 0.30215125638512175]
    @test isapprox(w5.weights, wt, rtol = 0.05)

    wc5 = optimise!(portfolio; type = WC(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w5.weights, wc5.weights, rtol = 0.05)

    portfolio.network_adj = SDP(; A = C)
    w6 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [1.3733290417870086e-6, 2.6763589401080658e-6, 2.9538542317039173e-6,
          0.0005900343176583919, 1.3887640713916724e-6, 1.3598728144018154e-6,
          8.979482586127353e-7, 3.591109234764808e-6, 6.310852949253323e-8,
          6.324340049812946e-7, 0.32590968729693326, 2.4443660959673622e-6,
          1.414358800261893e-7, 1.9293289965058625e-6, 0.01628800189959108,
          0.014069026506391764, 7.282630829588722e-6, 0.05452643821874623,
          -2.6476800486090808e-6, 0.24859272489979878]
    @test isapprox(w6.weights, wt)

    wc6 = optimise!(portfolio; type = WC(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w6.weights, wc6.weights)

    portfolio.a_vec_cent = []
    portfolio.b_cent = Inf

    portfolio.network_adj = IP(; A = B)
    w7 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [-2.7979705637303613e-12, -2.674164661233833e-12, -2.7195659781236553e-12,
          0.061139565783507016, -2.6186012114462426e-12, -2.683596584182768e-12,
          -0.004020157322809187, -2.4927467651482086e-12, -2.8185652409874115e-12,
          -2.7054992575150434e-12, 0.3414093186030684, -2.348342770957414e-12,
          0.003703016949216757, -2.5024479096446906e-12, -2.0714448417457455e-12,
          -2.3965106324102836e-12, -2.7913315961420834e-12, 0.25776825602578524,
          -2.7430718962487568e-12, -2.4043436382078565e-12]
    @test isapprox(w7.weights, wt, rtol = 0.05)

    wc7 = optimise!(portfolio; type = WC(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w7.weights, wc7.weights, rtol = 0.25)

    portfolio.network_adj = SDP(; A = B)
    w8 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [3.353883669642267e-6, 0.038122465245421025, 0.012782948631217341,
          0.024251278321675323, 0.011026952156532094, 0.04193024239831407,
          -0.007723026091322901, 6.2145857665693815e-6, -3.39569902695355e-5,
          0.012648769570655168, 0.2453658971390818, -0.014844139342231548,
          -2.384652077201505e-6, 6.1640233295210345e-6, 0.006967498867791508,
          0.01709226942230279, 1.1245332165782372e-6, 0.17380674670205037,
          6.802767311385322e-7, 0.09859090131814645]
    @test isapprox(w8.weights, wt)

    wc8 = optimise!(portfolio; type = WC(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w8.weights, wc8.weights)

    portfolio.network_adj = IP(; A = C)
    w9 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [0.18027575650144695, -3.877823575232523e-11, -3.971956651770663e-11,
          -3.92894492300436e-11, -4.007720189936552e-11, -3.7916771387879246e-11,
          0.007415441174848875, -3.645286098861366e-11, -3.9683707230051737e-11,
          -3.8639346247339255e-11, 0.4723088029811972, -3.9779839964217435e-11,
          -4.0420809059123584e-11, -3.7352503635664154e-11, -3.8858812193696976e-11,
          -3.822646548151099e-11, -3.8817465094503005e-11, -3.691522018311384e-11,
          -3.8915290842731694e-11, -3.764869556000491e-11]
    @test isapprox(w9.weights, wt, rtol = 0.1)

    wc9 = optimise!(portfolio; type = WC(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w9.weights, wc9.weights, rtol = 0.1)

    portfolio.network_adj = SDP(; A = C)
    w10 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [1.645587229073337e-6, 1.0964650874070903e-6, 8.099324032340491e-7,
          4.750334438676555e-7, 5.708947010377271e-7, 5.026059745915548e-7,
          3.5554835828654246e-7, 8.862257420249835e-7, 1.6505843417208158e-7,
          2.4331260596876106e-7, 0.4039667416480195, 2.493579100224862e-7,
          7.165427252726094e-8, 2.0949825706215104e-6, 2.1348296632699007e-6,
          0.037534143273519546, 1.0029864740757473e-5, 2.983369616639739e-6,
          3.6844804532910475e-7, 0.21847443190766216]
    @test isapprox(w10.weights, wt)

    wc10 = optimise!(portfolio; type = WC(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w10.weights, wc10.weights)

    portfolio = Portfolio(; prices = prices,
                          solvers = Dict(:PClGL => Dict(:solver => optimizer_with_attributes(Pajarito.Optimizer,
                                                                                             "verbose" => false,
                                                                                             "oa_solver" => optimizer_with_attributes(HiGHS.Optimizer,
                                                                                                                                      MOI.Silent() => true),
                                                                                             "conic_solver" => optimizer_with_attributes(Clarabel.Optimizer,
                                                                                                                                         "verbose" => false,
                                                                                                                                         "max_step_fraction" => 0.75)))))
    asset_statistics!(portfolio)
    wc_statistics!(portfolio,
                   WCType(; box = NormalWC(; seed = 123456789),
                          ellipse = NormalWC(; seed = 123456789)))
    portfolio.short = true
    portfolio.short_budget = 0.27
    portfolio.short_u = 0.27
    portfolio.long_u = 0.81
    portfolio.budget = portfolio.long_u - portfolio.short_u

    obj = Sharpe(; rf = rf)

    A = centrality_vector(portfolio; network_type = TMFG())
    B = connection_matrix(portfolio; network_type = TMFG())
    C = cluster_matrix(portfolio; hclust_alg = DBHT())

    w11 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [1.3566877119847358e-9, 2.0446966457964675e-8, 0.002262512310895483,
          1.4309648003585085e-7, 0.2734983203742686, -0.10965529909211759,
          0.03892353698664605, 0.01819498928305758, 5.900712314743839e-9,
          1.0611222523062497e-8, 0.039700687003729467, -0.05489428919436992,
          -0.028441439852089117, 6.842779119067803e-10, -0.07700892954655005,
          0.10494492437890107, 0.14627523180964974, 6.830986121056796e-8,
          0.18619935145952116, 1.5367224934418446e-7]
    @test isapprox(w11.weights, wt)

    wc11 = optimise!(portfolio; type = WC(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w11.weights, wc11.weights, rtol = 5.0e-5)

    portfolio.a_vec_cent = A
    portfolio.b_cent = minimum(A)

    w12 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [2.1497119336493176e-10, 0.0088070179228199, 0.029397048718257417,
          0.019022694413270754, 0.2828940619983948, -0.12224670576806218,
          0.04353925700174413, 0.04156276108416902, 1.3097764426973018e-7,
          1.4902868835689554e-7, 0.09605820879721724, -0.04855626421146047,
          -0.029205822936867836, 1.055694863453504e-9, -0.0699910697420536,
          0.11603391472918954, 0.04641077698276719, 0.04812947034812958,
          0.07616841032531034, 0.001975959060175573]
    @test isapprox(w12.weights, wt)

    wc12 = optimise!(portfolio; type = WC(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w12.weights, wc12.weights, rtol = 5e-5)

    portfolio.network_adj = IP(; A = B)
    w13 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [1.5269128820151116e-10, 5.0153063190116795e-11, 5.600508563045624e-11,
          -0.1499999969120081, 5.516614800822251e-11, 8.547041188790406e-11,
          3.5801273917717314e-11, 1.256273861351736e-11, 2.409947687735074e-10,
          0.6899999950792018, 1.3622829299878125e-11, 4.821233895750819e-11,
          6.845347024420311e-11, 1.1561363674032775e-10, 6.517378250466465e-11,
          5.779913743888119e-11, 2.8740320583775717e-10, 4.066302469648558e-11,
          3.2976327105495704e-10, 1.1724076184220898e-10]
    @test isapprox(w13.weights, wt)

    wc13 = optimise!(portfolio; type = WC(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w13.weights, wc13.weights)

    portfolio.network_adj = SDP(; A = B)
    w14 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [6.120770021032183e-8, 1.6354124623024568e-6, -0.00014073619896361552,
          -1.5382765037317844e-5, 0.20801384458722627, -1.2971430048721393e-7,
          0.01815677072722214, 2.2037258513423353e-7, 6.723276350257066e-8,
          6.337618763433898e-8, 0.14038784849839608, -0.04511790824985411,
          -0.005307605149301374, 3.927932051708812e-8, -0.058853373768001094,
          0.06663138474637784, 7.950726727959057e-7, 0.12731744850449067,
          0.08892422332791153, 7.335001414362041e-7]
    @test isapprox(w14.weights, wt, rtol = 5.0e-8)

    wc14 = optimise!(portfolio; type = WC(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w14.weights, wc14.weights, rtol = 0.0001)

    portfolio.network_adj = IP(; A = C)
    w15 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [7.297197051787683e-13, 9.751495210303498e-13, 9.25146423310274e-13,
          9.90656747259057e-13, 0.5686937869795202, 6.347117711378158e-13,
          1.3067533184269999e-12, 1.0554342404566512e-12, 8.12208116452551e-13,
          9.444489676262045e-13, 1.0265787555201995e-12, 6.195482895301479e-13,
          -0.10244889726663442, 7.588262617099839e-13, 4.79429725214721e-13,
          1.2471770398993632e-12, 7.867540101053476e-13, 1.0021245942023042e-12,
          0.07375511027191975, 8.998095142341619e-13]
    @test isapprox(w15.weights, wt)

    wc15 = optimise!(portfolio; type = WC(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w15.weights, wc15.weights, rtol = 0.0005)

    portfolio.network_adj = SDP(; A = C)
    w16 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [1.4274592086142099e-8, 1.522783678899051e-8, 2.7938717235026846e-8,
          1.4561550826598941e-8, 0.2368909369265324, -0.0001506005786567205,
          -1.0081165111908498e-8, 1.7224029270673118e-5, 4.5248679067995964e-8,
          -0.00022728261938254014, 0.119886342638237, -2.4823635865134306e-7,
          -3.413454237398767e-6, -1.0451687138374718e-8, -2.8921310398193577e-6,
          0.08684122100032984, 0.0035296823817011595, 7.622590527125835e-9,
          0.09321873846044709, 1.8724204259728512e-7]
    @test isapprox(w16.weights, wt)

    wc16 = optimise!(portfolio; type = WC(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w16.weights, wc16.weights, rtol = 0.0005)

    portfolio.a_vec_cent = []
    portfolio.b_cent = Inf

    portfolio.network_adj = IP(; A = B)
    w17 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [-0.0, -0.0, -0.0, -0.0, 0.8099999999999998, -0.0, -0.0, -0.0, -0.0, -0.0, 0.0,
          -0.0, -0.0, -0.0, -0.27, -0.0, 0.0, -0.0, 0.0, -0.0]
    @test isapprox(w17.weights, wt)

    wc17 = optimise!(portfolio; type = WC(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w17.weights, wc17.weights, rtol = 5.0e-7)

    portfolio.network_adj = SDP(; A = B)
    w18 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [3.7372455923638106e-8, 4.0946527532379914e-7, 5.741510057885535e-8,
          3.4014209639844175e-8, 0.1986252065725683, -8.486609959409557e-8,
          0.023951161167138676, 1.8655603296653543e-7, 3.192609402220337e-8,
          4.19296704875058e-8, 0.18694537859607946, -0.055626176371317476,
          -2.153219858731183e-8, 1.6743729639380584e-8, -0.05376684092336848,
          0.08378904151649921, 8.526537485802294e-8, 0.1560810784728213,
          9.345153848505544e-8, 2.632283952726144e-7]
    @test isapprox(w18.weights, wt)

    wc18 = optimise!(portfolio; type = WC(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w18.weights, wc18.weights, rtol = 0.0001)

    portfolio.network_adj = IP(; A = C)
    w19 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [-0.0, -0.0, -0.0, -0.0, 0.6011773438073398, -0.27, 0.0, -0.0, -0.0, 0.0, 0.0,
          -0.0, -0.0, -0.0, -0.0, 0.20882265619266027, 0.0, -0.0, 0.0, -0.0]
    @test isapprox(w19.weights, wt, rtol = 0.0005)

    wc19 = optimise!(portfolio; type = WC(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w19.weights, wc19.weights, rtol = 0.0005)

    portfolio.network_adj = SDP(; A = C)
    w20 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [1.9089390697715466e-8, 2.229133765919515e-8, 3.281824986291854e-8,
          1.8749370001662676e-8, 0.2360852915904294, -8.445636169630471e-6,
          -1.2148952122020842e-9, 1.2285635879456866e-6, 5.003357016527931e-8,
          -0.0002758503591244658, 0.08672896660586178, -2.649201973767617e-7,
          -5.065177762345351e-7, -1.1746904762175866e-8, -9.7236488086004e-7,
          0.0852459526649671, 0.0035191828565809746, 1.6956403200522112e-8,
          0.128705177720426, 9.28197738427405e-8]
    @test isapprox(w20.weights, wt)

    wc20 = optimise!(portfolio; type = WC(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w20.weights, wc20.weights, rtol = 0.001)
end

@testset "Network and Dendrogram upper dev" begin
    portfolio = Portfolio(; prices = prices,
                          solvers = Dict(:PClGL => Dict(:check_sol => (allow_local = true,
                                                                       allow_almost = true),
                                                        :solver => optimizer_with_attributes(Pajarito.Optimizer,
                                                                                             "verbose" => false,
                                                                                             "oa_solver" => optimizer_with_attributes(HiGHS.Optimizer,
                                                                                                                                      MOI.Silent() => true),
                                                                                             "conic_solver" => optimizer_with_attributes(Clarabel.Optimizer,
                                                                                                                                         "verbose" => false,
                                                                                                                                         "max_step_fraction" => 0.75)))))
    asset_statistics!(portfolio)
    B = connection_matrix(portfolio; network_type = TMFG())

    rm = SD()
    w1 = optimise!(portfolio; obj = Sharpe(; rf = rf), rm = rm)
    r1 = calc_risk(portfolio; rm = rm)

    rm.settings.ub = r1
    portfolio.network_adj = IP(; A = B)
    w2 = optimise!(portfolio; obj = MinRisk(), rm = rm)
    r2 = calc_risk(portfolio; rm = rm)
    w3 = optimise!(portfolio; obj = Utility(; l = l), rm = rm)
    r3 = calc_risk(portfolio; rm = rm)
    w4 = optimise!(portfolio; obj = Sharpe(; rf = rf), rm = rm)
    r4 = calc_risk(portfolio; rm = rm)
    w5 = optimise!(portfolio; obj = MaxRet(), rm = rm)
    r5 = calc_risk(portfolio; rm = rm)
    @test r2 <= r1
    @test r3 <= r1 || abs(r3 - r1) < 5e-8
    @test r4 <= r1
    @test r5 <= r1 || abs(r5 - r1) < 5e-8

    portfolio.network_adj = SDP(; A = B)
    w6 = optimise!(portfolio; obj = MinRisk(), rm = rm)
    r6 = calc_risk(portfolio; rm = rm)
    w7 = optimise!(portfolio; obj = Utility(; l = l), rm = rm)
    r7 = calc_risk(portfolio; rm = rm)
    w8 = optimise!(portfolio; obj = Sharpe(; rf = rf), rm = rm)
    r8 = calc_risk(portfolio; rm = rm)
    w9 = optimise!(portfolio; obj = MaxRet(), rm = rm)
    r9 = calc_risk(portfolio; rm = rm)
    @test r6 <= r1
    @test r7 <= r1
    @test r8 <= r1
    @test r9 <= r1

    portfolio = Portfolio(; prices = prices,
                          solvers = Dict(:PClGL => Dict(:solver => optimizer_with_attributes(Pajarito.Optimizer,
                                                                                             "verbose" => false,
                                                                                             "oa_solver" => optimizer_with_attributes(HiGHS.Optimizer,
                                                                                                                                      MOI.Silent() => true),
                                                                                             "conic_solver" => optimizer_with_attributes(Clarabel.Optimizer,
                                                                                                                                         "verbose" => false,
                                                                                                                                         "max_step_fraction" => 0.75)))))
    asset_statistics!(portfolio)
    rm = [[SD(), SD()]]
    w10 = optimise!(portfolio; obj = Sharpe(; rf = rf), rm = rm)
    r10 = calc_risk(portfolio; rm = rm[1][1])

    rm[1][1].settings.ub = r10
    portfolio.network_adj = IP(; A = B)
    w11 = optimise!(portfolio; obj = MinRisk(), rm = rm)
    r11 = calc_risk(portfolio; rm = rm[1][1])
    w12 = optimise!(portfolio; obj = Utility(; l = l), rm = rm)
    r12 = calc_risk(portfolio; rm = rm[1][1])
    w13 = optimise!(portfolio; obj = Sharpe(; rf = rf), rm = rm)
    r13 = calc_risk(portfolio; rm = rm[1][1])
    w14 = optimise!(portfolio; obj = MaxRet(), rm = rm)
    r14 = calc_risk(portfolio; rm = rm[1][1])
    @test r11 <= r10
    @test r12 <= r10 || abs(r12 - r10) < 5e-7
    @test r13 <= r10 || abs(r13 - r10) < 1e-10
    @test r14 <= r10 || abs(r14 - r10) < 5e-7

    portfolio.network_adj = SDP(; A = B)
    w15 = optimise!(portfolio; obj = MinRisk(), rm = rm)
    r15 = calc_risk(portfolio; rm = rm[1][1])
    w16 = optimise!(portfolio; obj = Utility(; l = l), rm = rm)
    r16 = calc_risk(portfolio; rm = rm[1][1])
    w17 = optimise!(portfolio; obj = Sharpe(; rf = rf), rm = rm)
    r17 = calc_risk(portfolio; rm = rm[1][1])
    w18 = optimise!(portfolio; obj = MaxRet(), rm = rm)
    r18 = calc_risk(portfolio; rm = rm[1][1])
    @test r15 <= r10
    @test r16 <= r10
    @test r17 <= r10
    @test r18 <= r10

    portfolio = portfolio = Portfolio(; prices = prices,
                                      solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                                       :check_sol => (allow_local = true,
                                                                                      allow_almost = true),
                                                                       :params => Dict("verbose" => false))))
    asset_statistics!(portfolio)

    rm = [SD(; settings = RMSettings(; flag = false)), CDaR()]

    portfolio.network_adj = SDP(; A = B)
    w19 = optimise!(portfolio; kelly = AKelly(), obj = Sharpe(; rf = rf), rm = rm)
    w20 = optimise!(portfolio; kelly = EKelly(), obj = Sharpe(; rf = rf), rm = rm)
    @test isapprox(w19.weights, w20.weights)

    w21 = optimise!(portfolio; type = RP(), rm = rm)
    portfolio.network_adj = IP(; A = B)
    w22 = optimise!(portfolio; type = RP(), rm = rm)
    portfolio.network_adj = SDP(; A = B)
    w23 = optimise!(portfolio; type = RP(), rm = rm)
    @test isapprox(w21.weights, w22.weights)
    @test isapprox(w21.weights, w23.weights)
    @test isapprox(w22.weights, w23.weights)
end

@testset "Network and Dendrogram non SD" begin
    portfolio = Portfolio(; prices = prices,
                          solvers = Dict(:PClGL => Dict(:solver => optimizer_with_attributes(Pajarito.Optimizer,
                                                                                             "verbose" => false,
                                                                                             "oa_solver" => optimizer_with_attributes(HiGHS.Optimizer,
                                                                                                                                      MOI.Silent() => true),
                                                                                             "conic_solver" => optimizer_with_attributes(Clarabel.Optimizer,
                                                                                                                                         "verbose" => false,
                                                                                                                                         "max_step_fraction" => 0.75)))))
    asset_statistics!(portfolio)

    A = centrality_vector(portfolio; network_type = TMFG())
    B = connection_matrix(portfolio; network_type = TMFG())
    C = cluster_matrix(portfolio; hclust_alg = DBHT())

    rm = CDaR()
    obj = MinRisk()
    w1 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [0.0, 5.74250749576219e-17, 0.0, 0.0, 0.0034099011531647325, 0.0, 0.0,
          0.07904282391685276, 0.0, 0.0, 0.3875931702023311, 0.0, 0.0,
          2.1342373233586235e-18, 0.0005545152741455163, 0.09598828930573457,
          0.26790888256716056, 0.0, 0.000656017191282443, 0.16484640038932813]
    @test isapprox(w1.weights, wt)

    portfolio.a_vec_cent = A
    portfolio.b_cent = minimum(A)
    w2 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [0.0, 0.0, 0.0, 0.051723138879499364, 0.0, 0.0, 0.05548699473292653,
          0.25666738948843576, 0.0, 0.0, 0.5073286323951223, 0.12879384450401607, 0.0, 0.0,
          0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    @test isapprox(w2.weights, wt)

    portfolio.network_adj = IP(; A = B)
    w3 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [0.0, 0.0, 0.0, 0.13808597886475105, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
          0.861914021135249, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    @test isapprox(w3.weights, wt)

    portfolio.network_adj = SDP(; A = B)
    w4 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [6.916528681040479e-11, 2.861831047852453e-11, 1.7205226727441286e-11,
          0.0706145671003988, 3.0536778285536253e-9, 6.76023773113509e-11,
          0.07499015177704357, 0.26806573786222676, 7.209644329535395e-11,
          6.263090324526159e-11, 0.41654672020086003, 0.16978281921567007,
          6.213186671810222e-11, 6.969908419111133e-11, 2.3053255039158856e-12,
          8.392552900081856e-11, 7.289286270405327e-11, 4.247286241831121e-11,
          7.335225686899845e-11, 6.602480341871697e-11]
    @test isapprox(w4.weights, wt)

    portfolio.network_adj = IP(; A = C)
    w5 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [0.0, 0.0, 0.0, 1.5501398315533836e-16, 6.703062176969295e-17, 0.0, 0.0,
          0.23899933639488402, 0.0, 0.0, 0.7610006636051158, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
          0.0, 0.0, 0.0]
    @test isapprox(w5.weights, wt)

    portfolio.network_adj = SDP(; A = C)
    w6 = optimise!(portfolio; obj = obj, rm = rm)
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

    portfolio.network_adj = IP(; A = B)
    w7 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [0.0, 0.0, 0.0, 0.0, 0.37297150326295364, 0.0, 0.0, 0.0, 0.0, 0.0,
          0.6270284967370463, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    @test isapprox(w7.weights, wt)

    portfolio.network_adj = SDP(; A = B)
    w8 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [5.333299624468313e-11, 0.05638785424268776, 1.801388361336222e-10,
          5.248225672626687e-11, 0.08206405829302137, 1.7756997488326776e-11,
          1.9025293508818056e-12, 0.13507762926776312, 1.3391306712204848e-10,
          3.7873704079316313e-11, 0.3175440779337929, 0.020298213899228854,
          4.164020888733719e-11, 1.174242267435215e-10, 4.7848187362837886e-11,
          0.12477101930240944, 0.07961825077560757, 0.03009048638763158,
          8.237622588564551e-11, 0.15414840913116812]
    @test isapprox(w8.weights, wt)

    portfolio.network_adj = IP(; A = C)
    w9 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4887092443113238, 0.0, 0.0,
          0.0, 0.0, 0.06771233053987868, 0.4435784251487976, 0.0, 0.0, 0.0]
    @test isapprox(w9.weights, wt)

    portfolio.network_adj = SDP(; A = C)
    w10 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [8.875598045563749e-11, 6.786115952224246e-11, 1.403756249532236e-10,
          3.8200133576780337e-11, 0.026405191197268175, 2.7118263798158232e-11,
          4.287027282530246e-12, 0.08423696478830205, 1.4264703610446034e-10,
          2.3728313538121046e-10, 0.3865644514064455, 0.009574217972314737,
          4.844735058003537e-11, 1.8846516237664424e-10, 1.2713658919263283e-11,
          0.14421172180536518, 0.2209166214591706, 2.7179501902437763e-9,
          0.0006754849417179316, 0.12741534271531094]
    @test isapprox(w10.weights, wt)

    portfolio = Portfolio(; prices = prices,
                          solvers = Dict(:PClGL => Dict(:solver => optimizer_with_attributes(Pajarito.Optimizer,
                                                                                             "verbose" => false,
                                                                                             "oa_solver" => optimizer_with_attributes(HiGHS.Optimizer,
                                                                                                                                      MOI.Silent() => true),
                                                                                             "conic_solver" => optimizer_with_attributes(Clarabel.Optimizer,
                                                                                                                                         "verbose" => false,
                                                                                                                                         "max_step_fraction" => 0.75)))))
    asset_statistics!(portfolio)
    obj = Sharpe(; rf = rf)

    A = centrality_vector(portfolio; network_type = TMFG())
    B = connection_matrix(portfolio; network_type = TMFG())
    C = cluster_matrix(portfolio; hclust_alg = DBHT())

    w11 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [0.0, 0.0, 0.07233520665470376, 0.0, 0.3107248736916702, 0.0, 0.0,
          0.12861270774687708, 0.0, 0.0, 0.16438408898657855, 0.0, 0.0, 0.0, 0.0,
          0.2628826637767333, 0.0, 0.0, 0.0, 0.061060459143437176]
    @test isapprox(w11.weights, wt)

    portfolio.a_vec_cent = A
    portfolio.b_cent = minimum(A)

    w12 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [0.0, 0.0, 0.0, 0.13626926674571754, 1.515977496012877e-16, 0.0,
          0.2529009327082102, 0.0, 0.0, 0.0, 0.4850451479068739, 0.12578465263919827, 0.0,
          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    @test isapprox(w12.weights, wt)

    portfolio.network_adj = IP(; A = B)
    w13 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [0.0, 0.0, 0.0, 0.0, 1.4641796690079654e-17, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    @test isapprox(w13.weights, wt)

    portfolio.network_adj = SDP(; A = B)
    w14 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [3.9876730635706574e-10, 9.058885429652311e-11, 1.5225868210279566e-11,
          0.15463223546511348, 1.787917922937287e-8, 3.89830536537106e-10,
          0.2677212654853749, 0.05396055329565413, 4.2501077924645286e-10,
          3.3984441830986676e-10, 0.353757964593812, 0.16992795957233114,
          2.6507986578852124e-11, 4.0584551279201494e-10, 2.2729721750159754e-10,
          3.426736037949514e-12, 4.169695539917153e-10, 1.7568326295730658e-10,
          4.272760538261085e-10, 3.662610153313769e-10]
    @test isapprox(w14.weights, wt)

    portfolio.network_adj = IP(; A = C)
    w15 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [0.0, 0.0, 0.0, 0.0, 6.313744038982638e-18, 0.0, 0.28939681985466575, 0.0, 0.0,
          0.0, 0.7106031801453343, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    @test isapprox(w15.weights, wt)

    portfolio.network_adj = SDP(; A = C)
    w16 = optimise!(portfolio; obj = obj, rm = rm)
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

    portfolio.network_adj = IP(; A = B)
    w17 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [0.0, 0.0, 0.0, 0.0, 0.7390777009270599, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
          0.0, 0.0, 0.26092229907294007, 0.0, 0.0, 0.0, 0.0]
    @test isapprox(w17.weights, wt)

    portfolio.network_adj = SDP(; A = B)
    w18 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [1.4680691018542715e-9, 4.859275738822192e-9, 0.015992802521425674,
          7.293948583023383e-10, 0.2988850375190448, 6.520933919634565e-11,
          2.128743388395745e-9, 0.16470653869543705, 4.624274795849735e-10,
          4.162708987445042e-10, 0.13583336339414312, 1.525142217002498e-11,
          2.494845225126199e-10, 1.0043814255652154e-9, 9.599421959136733e-11,
          0.2560211483665152, 4.277938501034575e-9, 1.880279259853607e-9,
          2.4445455761302985e-9, 0.12856108940616845]
    @test isapprox(w18.weights, wt)

    portfolio.network_adj = IP(; A = C)
    w19 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [0.0, 0.0, 0.0, 0.0, 0.3708533501847804, 0.0, 0.0, 0.0, 0.0, 0.0,
          0.3290504062456894, 0.0, 0.0, 0.0, 0.0, 0.3000962435695302, 0.0, 0.0, 0.0, 0.0]
    @test isapprox(w19.weights, wt)

    portfolio.network_adj = SDP(; A = C)
    w20 = optimise!(portfolio; obj = obj, rm = rm)
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
    portfolio = Portfolio(; prices = prices,
                          solvers = Dict(:PClGL => Dict(:solver => optimizer_with_attributes(Pajarito.Optimizer,
                                                                                             "verbose" => false,
                                                                                             "oa_solver" => optimizer_with_attributes(HiGHS.Optimizer,
                                                                                                                                      MOI.Silent() => true),
                                                                                             "conic_solver" => optimizer_with_attributes(Clarabel.Optimizer,
                                                                                                                                         "verbose" => false,
                                                                                                                                         "max_step_fraction" => 0.75)))))
    asset_statistics!(portfolio)

    A = centrality_vector(portfolio)
    B = connection_matrix(portfolio)
    C = cluster_matrix(portfolio)

    portfolio.short = true
    portfolio.short_u = 0.13
    portfolio.short_budget = 0.13
    portfolio.long_u = 1
    portfolio.budget = portfolio.long_u - portfolio.short_u

    rm = CDaR()
    obj = MinRisk()
    w1 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [0.0, -0.0033927423088478525, 0.0, -0.0017047316992726633, 0.02620966174092891,
          -0.048951276707458066, -0.02831053066644215, 0.0797084661157905,
          -0.006091550656354475, -0.01772949018065388, 0.34963750962788587,
          -0.009810028561542944, 0.006616206609090757, -1.7025233012195975e-16,
          -0.01400964921942798, 0.09738313723086256, 0.16435744872368938,
          0.01643568935570537, 0.10356751838407428, 0.15608436221197253]
    @test isapprox(w1.weights, wt)

    portfolio.a_vec_cent = A
    portfolio.b_cent = minimum(A)
    w2 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [-0.09486167717442834, 0.04019416692914485, 0.037577223179041455,
          -5.0655785389293176e-17, 0.12801741187716315, -7.979727989493313e-17,
          -0.01998095976345068, 0.13355629824656456, 0.019271038397774838, 0.0,
          0.28797176886025655, 0.0, 0.008845283123909234, 0.0, -0.015157363062120899,
          0.10734222426525322, 0.0, 0.061474266941201526, 4.119968255444917e-17,
          0.17575031817969053]
    @test isapprox(w2.weights, wt)

    portfolio.network_adj = IP(; A = B)
    w3 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [-0.0, 0.0, 0.0, -0.0, 0.25423835281660595, -0.10786045982487914, -0.0,
          0.3290798306064528, -0.0, -0.0, 2.220446049250313e-16, -0.0, 0.0, -0.0,
          0.02916010168515354, 0.13000000000000062, -0.0, 0.0, -0.0, 0.23538217471666556]
    @test isapprox(w3.weights, wt)

    portfolio.network_adj = SDP(; A = B, penalty = 0.5)
    w4 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [9.272435109943138e-12, 0.06109011163471849, 0.07388719114292448,
          0.019692527431470045, 0.10331812359040753, 0.08117018361193389,
          0.0233844026939607, 0.12195682880133857, 1.687054000923196e-11,
          0.034579519160302545, 0.02795355513256586, 0.05841592129946603,
          7.482492434260051e-12, 0.029412363249472755, 0.0020709632770030445,
          0.04322171854228879, -8.422304421932733e-13, 0.08680666625945271,
          -3.585326191328158e-11, 0.10303992417576462]
    @test isapprox(w4.weights, wt)

    portfolio.network_adj = IP(; A = C)
    w5 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [-0.0, -9.824982518806696e-17, -2.881746754291979e-16, -6.328762489489334e-17,
          0.3345846888561428, -0.0, -0.0, 0.4054153111438571, -0.0, -0.0, 0.0, 0.0, 0.0,
          -0.0, -0.0, 0.13000000000000017, -0.0, 0.0, -0.0, 0.0]
    @test isapprox(w5.weights, wt)

    portfolio.network_adj = SDP(; A = C, penalty = 0.5)
    w6 = optimise!(portfolio; obj = obj, rm = rm)
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

    portfolio.network_adj = IP(; A = B)
    w7 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [-0.0, -0.0, -0.0, 3.428633578839961e-17, 0.1119976188327676, -0.0, -0.0, 0.0, 0.0,
          0.0, 0.3750880542218098, -0.0, -0.0, -0.0, -0.0, 0.1412763749809966, -0.0,
          0.09588426469923997, 0.0, 0.14575368726518567]
    @test isapprox(w7.weights, wt)

    portfolio.network_adj = SDP(; A = B, penalty = 0.5)
    w8 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [6.766929954587163e-11, 0.05845600849974435, 0.06857721044042771,
          0.01863708044634632, 0.09512946481942187, 0.07994346654887266,
          0.02554585983141323, 0.07233232435238945, 5.714153552243313e-11,
          0.032112301662597306, 0.08806729718852686, 0.046990727441064695,
          2.302281082192624e-11, 0.041083140039804476, 0.001683242401425925,
          0.057989539665591214, 6.10236089865971e-11, 0.0822728003096611,
          3.5401881596356065e-11, 0.10117953610845386]
    @test isapprox(w8.weights, wt)

    portfolio.network_adj = IP(; A = C)
    w9 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 4.8338121699908494e-17, 0.0, -0.0, -0.0,
          0.425177042550851, -0.0, -1.0139121127832994e-15, -0.0, 2.2724003756919815e-16,
          0.05890972756969809, 0.3859132298794514, 0.0, -0.0, 0.0]
    @test isapprox(w9.weights, wt)

    portfolio.network_adj = SDP(; A = C, penalty = 0.5)
    w10 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [3.136684399611615e-10, 6.421446869383163e-11, 0.015548662828311242,
          5.302443988456493e-11, 0.07187792498761011, 3.443518625849509e-11,
          0.014549230934352904, 0.10906579897534709, 2.8604004872211283e-11,
          2.3132104858134452e-11, 0.23095115947815376, 0.044332984975470155,
          1.7809249411779717e-10, 4.072456628263281e-11, 0.005860251167461893,
          0.16234482039541365, 0.07541584072146104, 6.086150708285326e-11,
          5.436944458782648e-11, 0.14005332468529116]
    @test isapprox(w10.weights, wt)

    portfolio = Portfolio(; prices = prices,
                          solvers = Dict(:PClGL => Dict(:solver => optimizer_with_attributes(Pajarito.Optimizer,
                                                                                             "verbose" => false,
                                                                                             "oa_solver" => optimizer_with_attributes(HiGHS.Optimizer,
                                                                                                                                      MOI.Silent() => true),
                                                                                             "conic_solver" => optimizer_with_attributes(Clarabel.Optimizer,
                                                                                                                                         "verbose" => false,
                                                                                                                                         "max_step_fraction" => 0.75)))))
    asset_statistics!(portfolio)
    obj = Sharpe(; rf = rf)

    A = centrality_vector(portfolio; network_type = TMFG())
    B = connection_matrix(portfolio; network_type = TMFG())
    C = cluster_matrix(portfolio; hclust_alg = DBHT())

    portfolio.short = true
    portfolio.short_u = 0.18
    portfolio.short_budget = 0.18
    portfolio.long_u = 0.95
    portfolio.budget = portfolio.long_u - portfolio.short_u

    w11 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [0.0, 0.0, 0.06455652837124166, 0.0, 0.17806945118716286, -0.1534166903019555,
          6.554560797119506e-17, 0.12351856122756986, 0.0, 0.0, 0.21867000759778857, 0.0,
          -0.023295149141940166, 0.0, -0.0032881605561044004, 0.1770017931069312,
          0.006143447950208067, 0.0, 0.04071674553919559, 0.1413234650199022]
    @test isapprox(w11.weights, wt)

    portfolio.a_vec_cent = A
    portfolio.b_cent = minimum(A)

    w12 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [0.0, 0.0, 0.07328869996428454, 0.0, 0.19593630860627667, -0.1516035031548817,
          -2.228175066036319e-17, 0.1290693464408622, 0.0, 0.0, 0.20897763141227396, 0.0,
          -0.02192543103531701, 0.0, -0.006471065809801198, 0.18010601289300754,
          7.366923752795404e-17, 0.0, 0.025951685743898865, 0.13667031493939605]
    @test isapprox(w12.weights, wt)

    portfolio.network_adj = IP(; A = B)
    w13 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [-0.0, -0.0, 0.0, -0.0, 0.6900000000000023, -0.0, -0.0, 0.0, -0.0, -0.0,
          0.19949253798767527, -0.0, -0.11949253798767445, -0.0, -0.0, 0.0, 0.0, -0.0, 0.0,
          0.0]
    @test isapprox(w13.weights, wt)

    portfolio.network_adj = SDP(; A = B, penalty = 0.5)
    w14 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [3.514727061450158e-11, 0.10214522082786334, 1.5973615263620542e-9,
          8.750301284842235e-10, 0.23010239598778515, -0.025107535823589502,
          0.022839041934182035, 0.1484093093837071, -2.6374371040425877e-10,
          -3.153936811484647e-11, 0.12599682872769954, -6.592141286737157e-10,
          -0.03303460329907297, -1.0047674687409329e-10, -0.05977664458614574,
          0.17110148065348418, -7.483961562407158e-11, 5.423167113303609e-10,
          -2.6450191488394353e-10, 0.08732450453854651]
    @test isapprox(w14.weights, wt)

    portfolio.network_adj = IP(; A = C)
    w15 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [-0.0, -0.0, 0.0, -0.0, 0.6899999999999927, -0.0, -0.0, 0.0, -0.0, -0.0,
          0.1994925379876753, -0.0, -0.11949253798766485, -0.0, -0.0, 0.0, -0.0, -0.0, 0.0,
          0.0]
    @test isapprox(w15.weights, wt)

    portfolio.network_adj = SDP(; A = C, penalty = 0.5)
    w16 = optimise!(portfolio; obj = obj, rm = rm)
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

    portfolio.network_adj = IP(; A = B)
    w17 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [-0.0, -0.0, 0.0, -0.0, 0.6751155811095629, -0.0, -0.0, 0.0, -0.0, -0.0,
          0.208898224965609, -0.0, -0.11401380607517049, -0.0, -0.0, 0.0, 0.0, -0.0, -0.0,
          0.0]
    @test isapprox(w17.weights, wt)

    portfolio.network_adj = SDP(; A = B, penalty = 0.5)
    w18 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [9.396464919780623e-11, 0.09919154701647347, 6.368353684023622e-10,
          3.465978583702456e-10, 0.22064663460078363, -1.0614315396176674e-9,
          0.020025141314969998, 0.1266686482714095, 5.6795111561929726e-12,
          1.1754682802203758e-11, 0.12107267096223644, -4.262330938609224e-10,
          -0.04188688600867918, 1.5637094635344307e-10, -0.06521868018990389,
          0.16284169111496827, 6.247282655579725e-11, 4.814732694037535e-10,
          1.7210187767534034e-10, 0.12665923243815538]
    @test isapprox(w18.weights, wt)

    portfolio.network_adj = IP(; A = C)
    w19 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [-0.0, -0.0, 0.0, -0.0, 0.5385298461146729, -0.0, 0.0, 0.0, -0.0, -0.0, 0.0,
          -0.18000000000000382, -0.0, 0.0, 3.8379398333663466e-15, 0.411470153885331, -0.0,
          -0.0, -0.0, 0.0]
    @test isapprox(w19.weights, wt)

    portfolio.network_adj = SDP(; A = C, penalty = 0.5)
    w20 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [1.4172669439373994e-10, 1.1299075883203906e-10, 5.359434439554113e-10,
          8.26042302579247e-11, 0.3649711359888737, -6.90608391192042e-11,
          4.0350791884642237e-10, 2.0028585646891752e-10, 2.1735756891215644e-10,
          1.0421359032360126e-10, 0.11284986034990291, -6.494847118187474e-11,
          -0.027835392456788028, 1.767593240750613e-10, -0.015067449302819187,
          0.2566729326870001, 1.7852927573841912e-10, 7.917164427601336e-11,
          0.07840891046398553, 1.7076400157132798e-10]
    @test isapprox(w20.weights, wt, rtol = 5.0e-8)
end

@testset "Cluster and Dendrogram SD" begin
    portfolio = Portfolio(; prices = prices,
                          solvers = Dict(:PClGL => Dict(:solver => optimizer_with_attributes(Pajarito.Optimizer,
                                                                                             "verbose" => false,
                                                                                             "oa_solver" => optimizer_with_attributes(HiGHS.Optimizer,
                                                                                                                                      MOI.Silent() => true),
                                                                                             "conic_solver" => optimizer_with_attributes(Clarabel.Optimizer,
                                                                                                                                         "verbose" => false,
                                                                                                                                         "max_step_fraction" => 0.75)))))
    asset_statistics!(portfolio)
    wc_statistics!(portfolio,
                   WCType(; box = NormalWC(; seed = 123456789),
                          ellipse = NormalWC(; seed = 123456789)))

    A = centrality_vector(portfolio)
    B = connection_matrix(portfolio)
    C = cluster_matrix(portfolio)

    rm = SD()
    obj = MinRisk()
    w1 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [0.00791850924098073, 0.030672065216453506, 0.010501402809199123,
          0.027475241969187995, 0.012272329269527841, 0.03339587076426262,
          1.4321532289072258e-6, 0.13984297866711365, 2.4081081597353397e-6,
          5.114425959766348e-5, 0.2878111114337346, 1.5306036912879562e-6,
          1.1917690994187655e-6, 0.12525446872321966, 6.630910273840812e-6,
          0.015078706008184504, 8.254970801614574e-5, 0.1930993918762575,
          3.0369340845779798e-6, 0.11652799957572683]
    @test isapprox(w1.weights, wt)

    wc1 = optimise!(portfolio; type = WC(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w1.weights, wc1.weights)

    portfolio.a_vec_cent = A
    portfolio.b_cent = minimum(A)
    w2 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [3.5426430661259544e-11, 0.07546661314221123, 0.021373033743425803,
          0.027539611826011074, 0.023741467255115698, 0.12266515815974924,
          2.517181275812244e-6, 0.22629893782442134, 3.37118211246211e-10,
          0.02416530860824232, 3.3950118687352606e-10, 1.742541411959769e-6,
          1.5253730188343444e-6, 5.304686980295978e-11, 0.024731789616991084,
          3.3711966852218057e-10, 7.767147353183488e-11, 0.30008056166009706,
          1.171415437554966e-10, 0.1539317317710031]
    @test isapprox(w2.weights, wt)

    portfolio.cluster_adj = IP(; A = B)
    w3 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [5.603687159886085e-13, 1.3820161690005612e-12, 1.5853880637229228e-12,
          1.33602425322658e-12, 0.07121859232520031, 1.119571110222061e-12,
          1.041437600224815e-13, 0.26930632966134743, 6.131616834263862e-13,
          1.0353405330340386e-12, 1.0102504371932946e-12, 1.3075720486452749e-12,
          1.1733675314147727e-13, 4.761261290370584e-13, 7.973896995771363e-13,
          5.188187662626333e-13, 5.626224717027035e-15, 0.41769670819791543,
          7.996868449487614e-13, 0.24177836980276793]
    @test isapprox(w3.weights, wt)

    wc3 = optimise!(portfolio; type = WC(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w3.weights, wc3.weights, rtol = 0.01)

    portfolio.cluster_adj = SDP(; A = B)
    w4 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [4.2907851108373285e-11, 0.0754418669962984, 0.021396172655536325,
          0.027531481813488985, 0.02375148897766012, 0.12264432042703496,
          3.777946382632432e-6, 0.2263233585633904, 3.8909758570393176e-10,
          0.024184506959405477, 3.8945719747726095e-10, 2.597992728770223e-6,
          2.2916747013451583e-6, 5.942491440558677e-11, 0.02472298577516722,
          3.89238712058158e-10, 9.010863437870188e-11, 0.30009565874345184,
          1.32909841405546e-10, 0.15389948998160896]
    @test isapprox(w4.weights, wt)

    wc4 = optimise!(portfolio; type = WC(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w4.weights, wc4.weights)

    portfolio.cluster_adj = IP(; A = C)
    w5 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [2.600332585507959e-12, 2.49058898193321e-12, 2.5578162901552497e-12,
          1.5466046737425697e-12, 1.4967933865374272e-12, 3.0881407771151395e-13,
          4.123817493960925e-12, 3.252723921673227e-12, 1.632009812408571e-12,
          4.1272715410990253e-13, 1.3219081818309027e-12, 2.8170581035055335e-12,
          3.983969550512609e-12, 1.5360670864105517e-12, 0.05337095972065508,
          2.0129031207422168e-13, 8.038123010019403e-13, 0.5616133973621348,
          1.3058311743272652e-12, 0.38501564288481793]
    @test isapprox(w5.weights, wt)

    wc5 = optimise!(portfolio; type = WC(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w5.weights, wc5.weights)

    portfolio.cluster_adj = SDP(; A = C)
    w6 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [6.011468296535483e-11, 9.424732042613459e-6, 5.527537353284497e-6,
          2.903730060970158e-6, 4.265000229139505e-6, 5.05868769590198e-6,
          3.009791891015968e-6, 1.5268395403568086e-5, 5.657961006483616e-10,
          2.68155059410767e-6, 5.662844616891034e-10, 3.196842024458429e-6,
          2.419924540093777e-6, 8.983883474709306e-11, 0.05335383109918724,
          5.659272794304194e-10, 1.3052770418372395e-10, 0.5616352353647922,
          1.9587068208203376e-10, 0.38495717516982575]
    @test isapprox(w6.weights, wt)

    wc6 = optimise!(portfolio; type = WC(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w6.weights, wc6.weights)

    portfolio.a_vec_cent = []
    portfolio.b_cent = Inf

    portfolio.cluster_adj = IP(; A = B)
    w7 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [4.436079292890281e-13, 4.339657195356424e-13, 4.3353028423648065e-13,
          2.099632742038836e-14, 3.139151563704902e-13, 5.251815182737052e-13,
          9.708678618251658e-14, 7.40139180161491e-13, 2.82644625737616e-13,
          4.0939722177208954e-13, 0.43964473499450424, 6.45713115023492e-13,
          2.4339291393877985e-13, 4.979300009984306e-13, 9.848526731370591e-13,
          1.5980123190589848e-13, 4.0414638067893824e-13, 0.3300904730120406,
          3.428158408862516e-13, 0.2302647919864759]
    @test isapprox(w7.weights, wt)

    wc7 = optimise!(portfolio; type = WC(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w7.weights, wc7.weights, rtol = 0.0005)

    portfolio.cluster_adj = SDP(; A = B)
    w8 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [1.1008281068185902e-5, 0.055457116453828725, 0.012892903094396706,
          0.03284102649502972, 0.014979204379343122, 0.057886691097825904,
          1.0224197847100319e-6, 9.70550406073092e-6, 2.045810229626667e-6,
          0.012049795193184915, 0.3735114108030411, 1.330358085433759e-6,
          1.0648304534905729e-6, 9.906438750370314e-6, 0.007878565110236062,
          0.022521836037796082, 3.1895242150194783e-6, 0.26190144912467656,
          1.3462532236761872e-6, 0.14803938279076995]
    @test isapprox(w8.weights, wt, rtol = 5.0e-8)

    wc8 = optimise!(portfolio; type = WC(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w8.weights, wc8.weights)

    portfolio.cluster_adj = IP(; A = C)
    w9 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [1.0787207720863677e-12, 8.294709114613152e-14, 7.954607618722517e-13,
          3.3664271487071197e-13, 9.683130672209493e-13, 7.208016377361811e-13,
          5.283852777701427e-12, 4.1303200648255883e-13, 1.698528524551865e-12,
          8.083770937152382e-13, 0.6254998770683314, 2.693559025098044e-12,
          4.872090994188645e-12, 3.2453165851283277e-13, 1.0091301616417753e-12,
          0.06614823334410497, 0.30835188956514026, 2.463468634892428e-13,
          9.106489132241315e-13, 1.8060640077175145e-13]
    @test isapprox(w9.weights, wt)

    wc9 = optimise!(portfolio; type = WC(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w9.weights, wc9.weights)

    portfolio.cluster_adj = SDP(; A = C)
    w10 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [6.5767880723988875e-6, 4.623865766176694e-6, 3.533665790240184e-6,
          2.181035748649495e-6, 2.5215556700206228e-6, 1.858111674425038e-6,
          2.5125294721249436e-6, 2.7192384937770815e-6, 8.752492794952926e-7,
          9.63854668493286e-7, 0.6119910196342336, 2.4053903351549724e-6,
          9.311964129813141e-7, 9.706505481488379e-6, 1.3337324741660364e-5,
          0.05681677180817997, 4.294323401245429e-5, 1.2805451158943426e-5,
          1.6336794237581935e-6, 0.33108007988138427]
    @test isapprox(w10.weights, wt)

    wc10 = optimise!(portfolio; type = WC(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w10.weights, wc10.weights)

    portfolio = Portfolio(; prices = prices,
                          solvers = Dict(:PClGL => Dict(:solver => optimizer_with_attributes(Pajarito.Optimizer,
                                                                                             "verbose" => false,
                                                                                             "oa_solver" => optimizer_with_attributes(HiGHS.Optimizer,
                                                                                                                                      MOI.Silent() => true),
                                                                                             "conic_solver" => optimizer_with_attributes(Clarabel.Optimizer,
                                                                                                                                         "verbose" => false,
                                                                                                                                         "max_step_fraction" => 0.75)))))
    asset_statistics!(portfolio)
    obj = Sharpe(; rf = rf)

    A = centrality_vector(portfolio; network_type = TMFG())
    B = connection_matrix(portfolio; network_type = TMFG())
    C = cluster_matrix(portfolio; hclust_alg = DBHT())

    w11 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [5.2456519308516375e-9, 1.3711772920494845e-8, 1.828307590553689e-8,
          1.1104551657521415e-8, 0.5180586238310904, 1.7475553033958716e-9,
          0.06365101880957791, 1.1504018896467816e-8, 1.0491794751235226e-8,
          5.650081752041375e-9, 9.262635444153261e-9, 1.4051602963640009e-9,
          9.923106257589635e-10, 3.050812431838031e-9, 9.966076078173243e-10,
          0.1432679499627637, 0.19649701265872604, 1.0778757180749977e-8,
          0.07852527921167819, 1.1301377011579277e-8]
    @test isapprox(w11.weights, wt)

    wc11 = optimise!(portfolio; type = WC(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w11.weights, wc11.weights, rtol = 5.0e-6)

    portfolio.a_vec_cent = A
    portfolio.b_cent = minimum(A)

    w12 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [3.580964533638559e-11, 9.277053125069126e-10, 9.835000020421294e-10,
          0.48362433627647977, 1.3062621693742353e-9, 3.4185103123124565e-11,
          0.28435553885288534, 0.17984723049407092, 1.693432131576565e-10,
          2.621850918060399e-10, 0.05217287791621345, 5.161470450551339e-9,
          2.4492072342885186e-9, 4.790090083985224e-11, 2.6687808589346464e-9,
          1.0539691276230515e-9, 2.0827050059613154e-10, 8.48619997995145e-10,
          2.329804826102029e-10, 7.01603778985698e-11]
    @test isapprox(w12.weights, wt)

    wc12 = optimise!(portfolio; type = WC(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w12.weights, wc12.weights, rtol = 5.0e-5)

    portfolio.cluster_adj = IP(; A = B)
    w13 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [1.721507261658482e-11, 4.611331427286785e-11, 4.6537999918192326e-11,
          5.4302470941152905e-11, 4.508258159401719e-11, 2.510366932792115e-11,
          0.9999999992889718, 5.3842283750049385e-11, 9.81183158048394e-12,
          3.589613596601703e-11, 5.3385756391682404e-11, 4.8553725558375136e-11,
          4.299842533772006e-11, 1.755101403265066e-11, 4.285647343183246e-11,
          4.6446497580083914e-11, 2.1013218648897535e-11, 4.559832123949408e-11,
          3.102165424744005e-11, 2.7697895657875245e-11]
    @test isapprox(w13.weights, wt)

    wc13 = optimise!(portfolio; type = WC(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w13.weights, wc13.weights)

    portfolio.cluster_adj = SDP(; A = B)
    w14 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [5.024048627217141e-14, 1.5886394555170722e-12, 1.5924178422101003e-12,
          0.24554600920016964, 1.6332399340098677e-12, 1.0616523261700116e-13,
          0.0959435974734496, 0.2562392110969168, 2.7829378542014683e-13,
          4.877264338339266e-13, 0.402271160351581, 1.2141431002683076e-8,
          4.523802241069811e-9, 5.3729255370564864e-14, 5.20282725457044e-9,
          1.5947986553082236e-12, 3.43551721221936e-13, 1.5824894417145518e-12,
          3.8307499177570936e-13, 1.280439184666322e-13]
    @test isapprox(w14.weights, wt)

    wc14 = optimise!(portfolio; type = WC(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w14.weights, wc14.weights, rtol = 5.0e-5)

    portfolio.cluster_adj = IP(; A = C)
    w15 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [3.5488808869733083e-12, 8.281743679661074e-12, 8.80902177574244e-12,
          9.955072746064901e-12, 1.1134822019504797e-11, 4.7417114086610015e-12,
          0.6074274195608048, 1.0253105313724553e-11, 1.4148918790428058e-12,
          6.3499711667509184e-12, 0.39257258031345454, 9.215307590460059e-12,
          7.61409793255415e-12, 3.5761100061677828e-12, 7.3543478626708e-12,
          9.27354997599413e-12, 3.8374099382272584e-12, 8.458519745181501e-12,
          6.62153510029776e-12, 5.3005368572386036e-12]
    @test isapprox(w15.weights, wt, rtol = 1.0e-7)

    wc15 = optimise!(portfolio; type = WC(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w15.weights, wc15.weights, rtol = 0.005)

    portfolio.cluster_adj = SDP(; A = C)
    w16 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [1.645653643573042e-14, 5.24765733463725e-13, 5.259882343242254e-13,
          0.38135427736460104, 5.383168184665497e-13, 3.606530943112124e-14,
          2.659687073118818e-8, 2.387877721283499e-8, 9.190837103177029e-14,
          1.6141883247690533e-13, 0.6186456659519168, 3.0002908666789655e-9,
          1.5771657468870285e-9, 1.747936659731388e-14, 1.6271322639469356e-9,
          5.267534803071819e-13, 1.1362061261344631e-13, 5.231419643587768e-13,
          1.2683767418078167e-13, 4.256536051656617e-14]
    @test isapprox(w16.weights, wt)

    wc16 = optimise!(portfolio; type = WC(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w16.weights, wc16.weights, rtol = 5.0e-6)

    portfolio.a_vec_cent = []
    portfolio.b_cent = Inf

    portfolio.cluster_adj = IP(; A = B)
    w17 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [2.0966673826999653e-12, 2.198260339765037e-12, 1.9669534265126797e-12,
          2.0537353944645557e-12, 0.762282392772933, 2.61589719142697e-12,
          1.6397732202694488e-12, 2.573524529021814e-12, 2.1635532147945916e-12,
          2.3920718749431715e-12, 2.6031535089914636e-12, 2.254849729224801e-12,
          2.3351897528966937e-12, 2.5746406212787903e-12, 2.4339077077727048e-12,
          0.23771760718605286, 2.077497433772476e-12, 2.544464994811904e-12,
          2.2242733110585934e-12, 2.2657939494042705e-12]
    @test isapprox(w17.weights, wt)

    wc17 = optimise!(portfolio; type = WC(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w17.weights, wc17.weights, rtol = 0.0005)

    portfolio.cluster_adj = SDP(; A = B)
    w18 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [1.131030668352105e-7, 9.736849167799003e-7, 1.5073134798603708e-7,
          9.724848430006577e-8, 0.3050368084415617, 4.717356198432155e-8,
          0.03491168150833796, 1.1696087757981777e-6, 2.1133994869894878e-7,
          1.38898043984553e-7, 0.23158972602737993, 2.930159759606465e-8,
          1.841227833023016e-8, 1.3415748037100702e-7, 2.038375787580353e-8,
          0.10856264505102015, 2.399490931217591e-7, 0.2072142228794291,
          2.522693174355702e-7, 0.11268131983060002]
    @test isapprox(w18.weights, wt, rtol = 5.0e-8)

    wc18 = optimise!(portfolio; type = WC(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w18.weights, wc18.weights, rtol = 5.0e-5)

    portfolio.cluster_adj = IP(; A = C)
    w19 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [1.1904763791512141e-12, 1.2340531692956696e-12, 1.067454109369536e-12,
          1.127653866268004e-12, 0.617653720874792, 1.6368429734597877e-12,
          9.929978443062197e-13, 1.5549069160715418e-12, 1.0890925109349555e-12,
          1.3843599449712156e-12, 1.5689472056216027e-12, 1.400584981008052e-12,
          1.5941278979112304e-12, 1.5487191642726196e-12, 1.6035009203180108e-12,
          0.1719148482846596, 1.1171304302527388e-12, 1.5058581547861255e-12,
          0.210431430817618, 1.3135957217697496e-12]
    @test isapprox(w19.weights, wt)

    wc19 = optimise!(portfolio; type = WC(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w19.weights, wc19.weights, rtol = 0.005)

    portfolio.cluster_adj = SDP(; A = C)
    w20 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [1.9469265893415583e-7, 1.9264487242540467e-7, 2.232335248226593e-7,
          1.1246896197634285e-7, 0.40746626496017013, 8.182096647686593e-8,
          5.006408130895852e-8, 2.231759312172926e-7, 3.1463420211396267e-7,
          7.356649686989611e-6, 0.3255255880379563, 4.241919388356267e-8,
          2.851355716950413e-8, 2.082232132989454e-7, 3.1366090964049265e-8,
          0.15590260393465316, 0.010877836307400067, 2.4616586233596695e-7,
          0.10021813719562453, 2.6349139195481583e-7]
    @test isapprox(w20.weights, wt, rtol = 1.0e-5)

    wc20 = optimise!(portfolio; type = WC(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w20.weights, wc20.weights, rtol = 0.0001)
end

@testset "Cluster and Dendrogram SD short" begin
    portfolio = Portfolio(; prices = prices, short_budget = 10.0,
                          solvers = Dict(:PClGL => Dict(:check_sol => (allow_local = true,
                                                                       allow_almost = true),
                                                        :solver => optimizer_with_attributes(Pajarito.Optimizer,
                                                                                             "verbose" => false,
                                                                                             "oa_solver" => optimizer_with_attributes(HiGHS.Optimizer,
                                                                                                                                      MOI.Silent() => true),
                                                                                             "conic_solver" => optimizer_with_attributes(Clarabel.Optimizer,
                                                                                                                                         "verbose" => false,
                                                                                                                                         "max_step_fraction" => 0.75)))))
    asset_statistics!(portfolio)
    wc_statistics!(portfolio,
                   WCType(; box = NormalWC(; seed = 123456789),
                          ellipse = NormalWC(; seed = 123456789)))
    portfolio.short = true
    portfolio.short_budget = 0.22
    portfolio.short_u = 0.22
    portfolio.long_u = 0.88
    portfolio.budget = portfolio.long_u - portfolio.short_u

    A = centrality_vector(portfolio)
    B = connection_matrix(portfolio)
    C = cluster_matrix(portfolio)

    rm = SD()
    obj = MinRisk()
    w1 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [0.0023732456580253273, 0.02478417662625044, 0.011667587612994237,
          0.021835859830789766, 0.008241287060153578, 0.03550224737381907,
          -0.006408569798470374, 0.09320297606692711, -0.0072069893239688695,
          0.012269888650642718, 0.18722784970990347, -0.01395210113745165,
          -0.006026909793594887, 0.0962770551826336, 0.0005110433722419729,
          0.016784621725152556, 0.009614188864666265, 0.13436280943955117,
          -0.042344878358121694, 0.08128461123785628]
    @test isapprox(w1.weights, wt)

    wc1 = optimise!(portfolio; type = WC(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w1.weights, wc1.weights)

    portfolio.a_vec_cent = A
    portfolio.b_cent = minimum(A)
    w2 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [0.0064520166328966245, 0.022489884719278264, 0.01029468309633522,
          0.020902946113760288, 0.00711671166632188, 0.03337768007882898,
          -0.006266220580382531, 0.09139178524192762, -0.02237340947148962,
          0.010954134035283628, 0.18619094237390604, -0.014102818325104587,
          -0.0055880934328339325, 0.09574848799807273, 0.0002468303788802295,
          0.01712117146324995, 0.014137258176396965, 0.13074888415899583,
          -0.01805679382768158, 0.07921391950335803]
    @test isapprox(w2.weights, wt)

    portfolio.cluster_adj = IP(; A = B)
    w3 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [0.006062713502026987, -1.82444535940986e-11, -1.8668641666100138e-11,
          -1.8497828216051628e-11, -1.8790386852871776e-11, -1.8031722495722727e-11,
          -1.9061197724922234e-11, -1.7599957286747152e-11, -1.7049358857324967e-11,
          -1.8223294421307198e-11, 0.32181185959658903, -1.8908757892656348e-11,
          -1.9168697126217532e-11, -1.3928253949800599e-11, -1.8349718535583803e-11,
          -1.6677394168784684e-11, -6.267257105255323e-12, 0.3321254271795085,
          -2.669852770519402e-12, -1.79874967199412e-11]
    @test isapprox(w3.weights, wt, rtol = 0.5)

    wc3 = optimise!(portfolio; type = WC(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w3.weights, wc3.weights, rtol = 0.05)

    portfolio.cluster_adj = SDP(; A = B)
    w4 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [0.022589148447524018, 0.036812012135242114, 0.006833793601670234,
          0.013202059102917668, 0.005824555380015791, 0.03984269080339122,
          -0.006834168984417891, 1.9327468236390324e-5, -2.1177840385696868e-5,
          0.011500602752026409, 0.25265179781193275, -0.014411712546749403,
          -1.628444737600106e-5, 1.195721590601671e-5, 0.007515505990608158,
          0.019532808277500043, 4.833328704956932e-6, 0.16879897503212193,
          4.3131102914277385e-6, 0.09613896336083992]
    @test isapprox(w4.weights, wt)

    wc4 = optimise!(portfolio; type = WC(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w4.weights, wc4.weights)

    portfolio.cluster_adj = IP(; A = C)
    w5 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [-5.293989582770398e-11, -2.169360184726156e-11, -2.302603024389023e-11,
          -2.1745518271876716e-11, -2.3125965561871784e-11, -2.1701344936140427e-11,
          -2.296700740543876e-11, -1.9944292316184238e-11, -3.429967459242985e-11,
          -2.2415059313673134e-11, 0.34000000102143574, 0.017848743137050257,
          -2.3535780083971463e-11, -4.146383770552791e-11, -2.220301457889096e-11,
          -3.3471016659862616e-11, -6.139513723264243e-11, -1.9956735333122212e-11,
          -7.7723124936471e-11, 0.30215125638512175]
    @test isapprox(w5.weights, wt, rtol = 0.05)

    wc5 = optimise!(portfolio; type = WC(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w5.weights, wc5.weights)

    portfolio.cluster_adj = SDP(; A = C)
    w6 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [1.3733290417870086e-6, 2.6763589401080658e-6, 2.9538542317039173e-6,
          0.0005900343176583919, 1.3887640713916724e-6, 1.3598728144018154e-6,
          8.979482586127353e-7, 3.591109234764808e-6, 6.310852949253323e-8,
          6.324340049812946e-7, 0.32590968729693326, 2.4443660959673622e-6,
          1.414358800261893e-7, 1.9293289965058625e-6, 0.01628800189959108,
          0.014069026506391764, 7.282630829588722e-6, 0.05452643821874623,
          -2.6476800486090808e-6, 0.24859272489979878]
    @test isapprox(w6.weights, wt)

    wc6 = optimise!(portfolio; type = WC(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w6.weights, wc6.weights)

    portfolio.a_vec_cent = []
    portfolio.b_cent = Inf

    portfolio.cluster_adj = IP(; A = B)
    w7 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [-2.7979705637303613e-12, -2.674164661233833e-12, -2.7195659781236553e-12,
          0.061139565783507016, -2.6186012114462426e-12, -2.683596584182768e-12,
          -0.004020157322809187, -2.4927467651482086e-12, -2.8185652409874115e-12,
          -2.7054992575150434e-12, 0.3414093186030684, -2.348342770957414e-12,
          0.003703016949216757, -2.5024479096446906e-12, -2.0714448417457455e-12,
          -2.3965106324102836e-12, -2.7913315961420834e-12, 0.25776825602578524,
          -2.7430718962487568e-12, -2.4043436382078565e-12]
    @test isapprox(w7.weights, wt)

    wc7 = optimise!(portfolio; type = WC(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w7.weights, wc7.weights, rtol = 0.25)

    portfolio.cluster_adj = SDP(; A = B)
    w8 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [3.353883669642267e-6, 0.038122465245421025, 0.012782948631217341,
          0.024251278321675323, 0.011026952156532094, 0.04193024239831407,
          -0.007723026091322901, 6.2145857665693815e-6, -3.39569902695355e-5,
          0.012648769570655168, 0.2453658971390818, -0.014844139342231548,
          -2.384652077201505e-6, 6.1640233295210345e-6, 0.006967498867791508,
          0.01709226942230279, 1.1245332165782372e-6, 0.17380674670205037,
          6.802767311385322e-7, 0.09859090131814645]
    @test isapprox(w8.weights, wt)

    wc8 = optimise!(portfolio; type = WC(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w8.weights, wc8.weights)

    portfolio.cluster_adj = IP(; A = C)
    w9 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [0.18027575650144695, -3.877823575232523e-11, -3.971956651770663e-11,
          -3.92894492300436e-11, -4.007720189936552e-11, -3.7916771387879246e-11,
          0.007415441174848875, -3.645286098861366e-11, -3.9683707230051737e-11,
          -3.8639346247339255e-11, 0.4723088029811972, -3.9779839964217435e-11,
          -4.0420809059123584e-11, -3.7352503635664154e-11, -3.8858812193696976e-11,
          -3.822646548151099e-11, -3.8817465094503005e-11, -3.691522018311384e-11,
          -3.8915290842731694e-11, -3.764869556000491e-11]
    @test isapprox(w9.weights, wt, rtol = 0.1)

    wc9 = optimise!(portfolio; type = WC(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w9.weights, wc9.weights)

    portfolio.cluster_adj = SDP(; A = C)
    w10 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [1.645587229073337e-6, 1.0964650874070903e-6, 8.099324032340491e-7,
          4.750334438676555e-7, 5.708947010377271e-7, 5.026059745915548e-7,
          3.5554835828654246e-7, 8.862257420249835e-7, 1.6505843417208158e-7,
          2.4331260596876106e-7, 0.4039667416480195, 2.493579100224862e-7,
          7.165427252726094e-8, 2.0949825706215104e-6, 2.1348296632699007e-6,
          0.037534143273519546, 1.0029864740757473e-5, 2.983369616639739e-6,
          3.6844804532910475e-7, 0.21847443190766216]
    @test isapprox(w10.weights, wt)

    wc10 = optimise!(portfolio; type = WC(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w10.weights, wc10.weights)

    portfolio = Portfolio(; prices = prices,
                          solvers = Dict(:PClGL => Dict(:solver => optimizer_with_attributes(Pajarito.Optimizer,
                                                                                             "verbose" => false,
                                                                                             "oa_solver" => optimizer_with_attributes(HiGHS.Optimizer,
                                                                                                                                      MOI.Silent() => true),
                                                                                             "conic_solver" => optimizer_with_attributes(Clarabel.Optimizer,
                                                                                                                                         "verbose" => false,
                                                                                                                                         "max_step_fraction" => 0.75)))))
    asset_statistics!(portfolio)
    wc_statistics!(portfolio,
                   WCType(; box = NormalWC(; seed = 123456789),
                          ellipse = NormalWC(; seed = 123456789)))
    portfolio.short = true
    portfolio.short_budget = 0.27
    portfolio.short_u = 0.27
    portfolio.long_u = 0.81
    portfolio.budget = portfolio.long_u - portfolio.short_u

    obj = Sharpe(; rf = rf)

    A = centrality_vector(portfolio; network_type = TMFG())
    B = connection_matrix(portfolio; network_type = TMFG())
    C = cluster_matrix(portfolio; hclust_alg = DBHT())

    w11 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [1.3566877119847358e-9, 2.0446966457964675e-8, 0.002262512310895483,
          1.4309648003585085e-7, 0.2734983203742686, -0.10965529909211759,
          0.03892353698664605, 0.01819498928305758, 5.900712314743839e-9,
          1.0611222523062497e-8, 0.039700687003729467, -0.05489428919436992,
          -0.028441439852089117, 6.842779119067803e-10, -0.07700892954655005,
          0.10494492437890107, 0.14627523180964974, 6.830986121056796e-8,
          0.18619935145952116, 1.5367224934418446e-7]
    @test isapprox(w11.weights, wt)

    wc11 = optimise!(portfolio; type = WC(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w11.weights, wc11.weights, rtol = 5.0e-5)

    portfolio.a_vec_cent = A
    portfolio.b_cent = minimum(A)

    w12 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [2.1497119336493176e-10, 0.0088070179228199, 0.029397048718257417,
          0.019022694413270754, 0.2828940619983948, -0.12224670576806218,
          0.04353925700174413, 0.04156276108416902, 1.3097764426973018e-7,
          1.4902868835689554e-7, 0.09605820879721724, -0.04855626421146047,
          -0.029205822936867836, 1.055694863453504e-9, -0.0699910697420536,
          0.11603391472918954, 0.04641077698276719, 0.04812947034812958,
          0.07616841032531034, 0.001975959060175573]
    @test isapprox(w12.weights, wt)

    wc12 = optimise!(portfolio; type = WC(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w12.weights, wc12.weights, rtol = 5e-5)

    portfolio.cluster_adj = IP(; A = B)
    w13 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [1.5269128820151116e-10, 5.0153063190116795e-11, 5.600508563045624e-11,
          -0.1499999969120081, 5.516614800822251e-11, 8.547041188790406e-11,
          3.5801273917717314e-11, 1.256273861351736e-11, 2.409947687735074e-10,
          0.6899999950792018, 1.3622829299878125e-11, 4.821233895750819e-11,
          6.845347024420311e-11, 1.1561363674032775e-10, 6.517378250466465e-11,
          5.779913743888119e-11, 2.8740320583775717e-10, 4.066302469648558e-11,
          3.2976327105495704e-10, 1.1724076184220898e-10]
    @test isapprox(w13.weights, wt)

    wc13 = optimise!(portfolio; type = WC(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w13.weights, wc13.weights)

    portfolio.cluster_adj = SDP(; A = B)
    w14 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [6.120770021032183e-8, 1.6354124623024568e-6, -0.00014073619896361552,
          -1.5382765037317844e-5, 0.20801384458722627, -1.2971430048721393e-7,
          0.01815677072722214, 2.2037258513423353e-7, 6.723276350257066e-8,
          6.337618763433898e-8, 0.14038784849839608, -0.04511790824985411,
          -0.005307605149301374, 3.927932051708812e-8, -0.058853373768001094,
          0.06663138474637784, 7.950726727959057e-7, 0.12731744850449067,
          0.08892422332791153, 7.335001414362041e-7]
    @test isapprox(w14.weights, wt, rtol = 5.0e-8)

    wc14 = optimise!(portfolio; type = WC(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w14.weights, wc14.weights, rtol = 0.0001)

    portfolio.cluster_adj = IP(; A = C)
    w15 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [7.297197051787683e-13, 9.751495210303498e-13, 9.25146423310274e-13,
          9.90656747259057e-13, 0.5686937869795202, 6.347117711378158e-13,
          1.3067533184269999e-12, 1.0554342404566512e-12, 8.12208116452551e-13,
          9.444489676262045e-13, 1.0265787555201995e-12, 6.195482895301479e-13,
          -0.10244889726663442, 7.588262617099839e-13, 4.79429725214721e-13,
          1.2471770398993632e-12, 7.867540101053476e-13, 1.0021245942023042e-12,
          0.07375511027191975, 8.998095142341619e-13]
    @test isapprox(w15.weights, wt)

    wc15 = optimise!(portfolio; type = WC(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w15.weights, wc15.weights, rtol = 0.0005)

    portfolio.cluster_adj = SDP(; A = C)
    w16 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [1.4274592086142099e-8, 1.522783678899051e-8, 2.7938717235026846e-8,
          1.4561550826598941e-8, 0.2368909369265324, -0.0001506005786567205,
          -1.0081165111908498e-8, 1.7224029270673118e-5, 4.5248679067995964e-8,
          -0.00022728261938254014, 0.119886342638237, -2.4823635865134306e-7,
          -3.413454237398767e-6, -1.0451687138374718e-8, -2.8921310398193577e-6,
          0.08684122100032984, 0.0035296823817011595, 7.622590527125835e-9,
          0.09321873846044709, 1.8724204259728512e-7]
    @test isapprox(w16.weights, wt)

    wc16 = optimise!(portfolio; type = WC(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w16.weights, wc16.weights, rtol = 0.0005)

    portfolio.a_vec_cent = []
    portfolio.b_cent = Inf

    portfolio.cluster_adj = IP(; A = B)
    w17 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [-0.0, -0.0, -0.0, -0.0, 0.8099999999999998, -0.0, -0.0, -0.0, -0.0, -0.0, 0.0,
          -0.0, -0.0, -0.0, -0.27, -0.0, 0.0, -0.0, 0.0, -0.0]
    @test isapprox(w17.weights, wt)

    wc17 = optimise!(portfolio; type = WC(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w17.weights, wc17.weights, rtol = 5.0e-7)

    portfolio.cluster_adj = SDP(; A = B)
    w18 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [3.7372455923638106e-8, 4.0946527532379914e-7, 5.741510057885535e-8,
          3.4014209639844175e-8, 0.1986252065725683, -8.486609959409557e-8,
          0.023951161167138676, 1.8655603296653543e-7, 3.192609402220337e-8,
          4.19296704875058e-8, 0.18694537859607946, -0.055626176371317476,
          -2.153219858731183e-8, 1.6743729639380584e-8, -0.05376684092336848,
          0.08378904151649921, 8.526537485802294e-8, 0.1560810784728213,
          9.345153848505544e-8, 2.632283952726144e-7]
    @test isapprox(w18.weights, wt)

    wc18 = optimise!(portfolio; type = WC(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w18.weights, wc18.weights, rtol = 0.0001)

    portfolio.cluster_adj = IP(; A = C)
    w19 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [-0.0, -0.0, -0.0, -0.0, 0.6011773438073398, -0.27, 0.0, -0.0, -0.0, 0.0, 0.0,
          -0.0, -0.0, -0.0, -0.0, 0.20882265619266027, 0.0, -0.0, 0.0, -0.0]
    @test isapprox(w19.weights, wt, rtol = 0.0005)

    wc19 = optimise!(portfolio; type = WC(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w19.weights, wc19.weights, rtol = 0.0005)

    portfolio.cluster_adj = SDP(; A = C)
    w20 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [1.9089390697715466e-8, 2.229133765919515e-8, 3.281824986291854e-8,
          1.8749370001662676e-8, 0.2360852915904294, -8.445636169630471e-6,
          -1.2148952122020842e-9, 1.2285635879456866e-6, 5.003357016527931e-8,
          -0.0002758503591244658, 0.08672896660586178, -2.649201973767617e-7,
          -5.065177762345351e-7, -1.1746904762175866e-8, -9.7236488086004e-7,
          0.0852459526649671, 0.0035191828565809746, 1.6956403200522112e-8,
          0.128705177720426, 9.28197738427405e-8]
    @test isapprox(w20.weights, wt)

    wc20 = optimise!(portfolio; type = WC(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w20.weights, wc20.weights, rtol = 0.001)
end

@testset "Cluster and Dendrogram upper dev" begin
    portfolio = Portfolio(; prices = prices,
                          solvers = Dict(:PClGL => Dict(:check_sol => (allow_local = true,
                                                                       allow_almost = true),
                                                        :solver => optimizer_with_attributes(Pajarito.Optimizer,
                                                                                             "verbose" => false,
                                                                                             "oa_solver" => optimizer_with_attributes(HiGHS.Optimizer,
                                                                                                                                      MOI.Silent() => true),
                                                                                             "conic_solver" => optimizer_with_attributes(Clarabel.Optimizer,
                                                                                                                                         "verbose" => false,
                                                                                                                                         "max_step_fraction" => 0.75)))))
    asset_statistics!(portfolio)
    B = connection_matrix(portfolio; network_type = TMFG())

    rm = SD()
    w1 = optimise!(portfolio; obj = Sharpe(; rf = rf), rm = rm)
    r1 = calc_risk(portfolio; rm = rm)

    rm.settings.ub = r1
    portfolio.cluster_adj = IP(; A = B)
    w2 = optimise!(portfolio; obj = MinRisk(), rm = rm)
    r2 = calc_risk(portfolio; rm = rm)
    w3 = optimise!(portfolio; obj = Utility(; l = l), rm = rm)
    r3 = calc_risk(portfolio; rm = rm)
    w4 = optimise!(portfolio; obj = Sharpe(; rf = rf), rm = rm)
    r4 = calc_risk(portfolio; rm = rm)
    w5 = optimise!(portfolio; obj = MaxRet(), rm = rm)
    r5 = calc_risk(portfolio; rm = rm)
    @test r2 <= r1
    @test r3 <= r1 || abs(r3 - r1) < 5e-8
    @test r4 <= r1
    @test r5 <= r1 || abs(r5 - r1) < 5e-8

    portfolio.cluster_adj = SDP(; A = B)
    w6 = optimise!(portfolio; obj = MinRisk(), rm = rm)
    r6 = calc_risk(portfolio; rm = rm)
    w7 = optimise!(portfolio; obj = Utility(; l = l), rm = rm)
    r7 = calc_risk(portfolio; rm = rm)
    w8 = optimise!(portfolio; obj = Sharpe(; rf = rf), rm = rm)
    r8 = calc_risk(portfolio; rm = rm)
    w9 = optimise!(portfolio; obj = MaxRet(), rm = rm)
    r9 = calc_risk(portfolio; rm = rm)
    @test r6 <= r1
    @test r7 <= r1
    @test r8 <= r1
    @test r9 <= r1

    portfolio = Portfolio(; prices = prices,
                          solvers = Dict(:PClGL => Dict(:solver => optimizer_with_attributes(Pajarito.Optimizer,
                                                                                             "verbose" => false,
                                                                                             "oa_solver" => optimizer_with_attributes(HiGHS.Optimizer,
                                                                                                                                      MOI.Silent() => true),
                                                                                             "conic_solver" => optimizer_with_attributes(Clarabel.Optimizer,
                                                                                                                                         "verbose" => false,
                                                                                                                                         "max_step_fraction" => 0.75)))))
    asset_statistics!(portfolio)
    rm = [[SD(), SD()]]
    w10 = optimise!(portfolio; obj = Sharpe(; rf = rf), rm = rm)
    r10 = calc_risk(portfolio; rm = rm[1][1])

    rm[1][1].settings.ub = r10
    portfolio.cluster_adj = IP(; A = B)
    w11 = optimise!(portfolio; obj = MinRisk(), rm = rm)
    r11 = calc_risk(portfolio; rm = rm[1][1])
    w12 = optimise!(portfolio; obj = Utility(; l = l), rm = rm)
    r12 = calc_risk(portfolio; rm = rm[1][1])
    w13 = optimise!(portfolio; obj = Sharpe(; rf = rf), rm = rm)
    r13 = calc_risk(portfolio; rm = rm[1][1])
    w14 = optimise!(portfolio; obj = MaxRet(), rm = rm)
    r14 = calc_risk(portfolio; rm = rm[1][1])
    @test r11 <= r10
    @test r12 <= r10 || abs(r12 - r10) < 5e-7
    @test r13 <= r10 || abs(r13 - r10) < 1e-10
    @test r14 <= r10 || abs(r14 - r10) < 5e-7

    portfolio.cluster_adj = SDP(; A = B)
    w15 = optimise!(portfolio; obj = MinRisk(), rm = rm)
    r15 = calc_risk(portfolio; rm = rm[1][1])
    w16 = optimise!(portfolio; obj = Utility(; l = l), rm = rm)
    r16 = calc_risk(portfolio; rm = rm[1][1])
    w17 = optimise!(portfolio; obj = Sharpe(; rf = rf), rm = rm)
    r17 = calc_risk(portfolio; rm = rm[1][1])
    w18 = optimise!(portfolio; obj = MaxRet(), rm = rm)
    r18 = calc_risk(portfolio; rm = rm[1][1])
    @test r15 <= r10
    @test r16 <= r10
    @test r17 <= r10
    @test r18 <= r10

    portfolio = portfolio = Portfolio(; prices = prices,
                                      solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                                       :check_sol => (allow_local = true,
                                                                                      allow_almost = true),
                                                                       :params => Dict("verbose" => false))))
    asset_statistics!(portfolio)

    rm = [SD(; settings = RMSettings(; flag = false)), CDaR()]

    portfolio.cluster_adj = SDP(; A = B)
    w19 = optimise!(portfolio; kelly = AKelly(), obj = Sharpe(; rf = rf), rm = rm)
    w20 = optimise!(portfolio; kelly = EKelly(), obj = Sharpe(; rf = rf), rm = rm)
    @test isapprox(w19.weights, w20.weights)

    w21 = optimise!(portfolio; type = RP(), rm = rm)
    portfolio.cluster_adj = IP(; A = B)
    w22 = optimise!(portfolio; type = RP(), rm = rm)
    portfolio.cluster_adj = SDP(; A = B)
    w23 = optimise!(portfolio; type = RP(), rm = rm)
    @test isapprox(w21.weights, w22.weights)
    @test isapprox(w21.weights, w23.weights)
    @test isapprox(w22.weights, w23.weights)
end

@testset "Cluster and Dendrogram non SD" begin
    portfolio = Portfolio(; prices = prices,
                          solvers = Dict(:PClGL => Dict(:solver => optimizer_with_attributes(Pajarito.Optimizer,
                                                                                             "verbose" => false,
                                                                                             "oa_solver" => optimizer_with_attributes(HiGHS.Optimizer,
                                                                                                                                      MOI.Silent() => true),
                                                                                             "conic_solver" => optimizer_with_attributes(Clarabel.Optimizer,
                                                                                                                                         "verbose" => false,
                                                                                                                                         "max_step_fraction" => 0.75)))))
    asset_statistics!(portfolio)

    A = centrality_vector(portfolio; network_type = TMFG())
    B = connection_matrix(portfolio; network_type = TMFG())
    C = cluster_matrix(portfolio; hclust_alg = DBHT())

    rm = CDaR()
    obj = MinRisk()
    w1 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [0.0, 5.74250749576219e-17, 0.0, 0.0, 0.0034099011531647325, 0.0, 0.0,
          0.07904282391685276, 0.0, 0.0, 0.3875931702023311, 0.0, 0.0,
          2.1342373233586235e-18, 0.0005545152741455163, 0.09598828930573457,
          0.26790888256716056, 0.0, 0.000656017191282443, 0.16484640038932813]
    @test isapprox(w1.weights, wt)

    portfolio.a_vec_cent = A
    portfolio.b_cent = minimum(A)
    w2 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [0.0, 0.0, 0.0, 0.051723138879499364, 0.0, 0.0, 0.05548699473292653,
          0.25666738948843576, 0.0, 0.0, 0.5073286323951223, 0.12879384450401607, 0.0, 0.0,
          0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    @test isapprox(w2.weights, wt)

    portfolio.cluster_adj = IP(; A = B)
    w3 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [0.0, 0.0, 0.0, 0.13808597886475105, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
          0.861914021135249, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    @test isapprox(w3.weights, wt)

    portfolio.cluster_adj = SDP(; A = B)
    w4 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [6.916528681040479e-11, 2.861831047852453e-11, 1.7205226727441286e-11,
          0.0706145671003988, 3.0536778285536253e-9, 6.76023773113509e-11,
          0.07499015177704357, 0.26806573786222676, 7.209644329535395e-11,
          6.263090324526159e-11, 0.41654672020086003, 0.16978281921567007,
          6.213186671810222e-11, 6.969908419111133e-11, 2.3053255039158856e-12,
          8.392552900081856e-11, 7.289286270405327e-11, 4.247286241831121e-11,
          7.335225686899845e-11, 6.602480341871697e-11]
    @test isapprox(w4.weights, wt)

    portfolio.cluster_adj = IP(; A = C)
    w5 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [0.0, 0.0, 0.0, 1.5501398315533836e-16, 6.703062176969295e-17, 0.0, 0.0,
          0.23899933639488402, 0.0, 0.0, 0.7610006636051158, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
          0.0, 0.0, 0.0]
    @test isapprox(w5.weights, wt)

    portfolio.cluster_adj = SDP(; A = C)
    w6 = optimise!(portfolio; obj = obj, rm = rm)
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

    portfolio.cluster_adj = IP(; A = B)
    w7 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [0.0, 0.0, 0.0, 0.0, 0.37297150326295364, 0.0, 0.0, 0.0, 0.0, 0.0,
          0.6270284967370463, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    @test isapprox(w7.weights, wt)

    portfolio.cluster_adj = SDP(; A = B)
    w8 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [5.333299624468313e-11, 0.05638785424268776, 1.801388361336222e-10,
          5.248225672626687e-11, 0.08206405829302137, 1.7756997488326776e-11,
          1.9025293508818056e-12, 0.13507762926776312, 1.3391306712204848e-10,
          3.7873704079316313e-11, 0.3175440779337929, 0.020298213899228854,
          4.164020888733719e-11, 1.174242267435215e-10, 4.7848187362837886e-11,
          0.12477101930240944, 0.07961825077560757, 0.03009048638763158,
          8.237622588564551e-11, 0.15414840913116812]
    @test isapprox(w8.weights, wt)

    portfolio.cluster_adj = IP(; A = C)
    w9 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4887092443113238, 0.0, 0.0,
          0.0, 0.0, 0.06771233053987868, 0.4435784251487976, 0.0, 0.0, 0.0]
    @test isapprox(w9.weights, wt)

    portfolio.cluster_adj = SDP(; A = C)
    w10 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [8.875598045563749e-11, 6.786115952224246e-11, 1.403756249532236e-10,
          3.8200133576780337e-11, 0.026405191197268175, 2.7118263798158232e-11,
          4.287027282530246e-12, 0.08423696478830205, 1.4264703610446034e-10,
          2.3728313538121046e-10, 0.3865644514064455, 0.009574217972314737,
          4.844735058003537e-11, 1.8846516237664424e-10, 1.2713658919263283e-11,
          0.14421172180536518, 0.2209166214591706, 2.7179501902437763e-9,
          0.0006754849417179316, 0.12741534271531094]
    @test isapprox(w10.weights, wt)

    portfolio = Portfolio(; prices = prices,
                          solvers = Dict(:PClGL => Dict(:solver => optimizer_with_attributes(Pajarito.Optimizer,
                                                                                             "verbose" => false,
                                                                                             "oa_solver" => optimizer_with_attributes(HiGHS.Optimizer,
                                                                                                                                      MOI.Silent() => true),
                                                                                             "conic_solver" => optimizer_with_attributes(Clarabel.Optimizer,
                                                                                                                                         "verbose" => false,
                                                                                                                                         "max_step_fraction" => 0.75)))))
    asset_statistics!(portfolio)
    obj = Sharpe(; rf = rf)

    A = centrality_vector(portfolio; network_type = TMFG())
    B = connection_matrix(portfolio; network_type = TMFG())
    C = cluster_matrix(portfolio; hclust_alg = DBHT())

    w11 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [0.0, 0.0, 0.07233520665470376, 0.0, 0.3107248736916702, 0.0, 0.0,
          0.12861270774687708, 0.0, 0.0, 0.16438408898657855, 0.0, 0.0, 0.0, 0.0,
          0.2628826637767333, 0.0, 0.0, 0.0, 0.061060459143437176]
    @test isapprox(w11.weights, wt)

    portfolio.a_vec_cent = A
    portfolio.b_cent = minimum(A)

    w12 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [0.0, 0.0, 0.0, 0.13626926674571754, 1.515977496012877e-16, 0.0,
          0.2529009327082102, 0.0, 0.0, 0.0, 0.4850451479068739, 0.12578465263919827, 0.0,
          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    @test isapprox(w12.weights, wt)

    portfolio.cluster_adj = IP(; A = B)
    w13 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [0.0, 0.0, 0.0, 0.0, 1.4641796690079654e-17, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    @test isapprox(w13.weights, wt)

    portfolio.cluster_adj = SDP(; A = B)
    w14 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [3.9876730635706574e-10, 9.058885429652311e-11, 1.5225868210279566e-11,
          0.15463223546511348, 1.787917922937287e-8, 3.89830536537106e-10,
          0.2677212654853749, 0.05396055329565413, 4.2501077924645286e-10,
          3.3984441830986676e-10, 0.353757964593812, 0.16992795957233114,
          2.6507986578852124e-11, 4.0584551279201494e-10, 2.2729721750159754e-10,
          3.426736037949514e-12, 4.169695539917153e-10, 1.7568326295730658e-10,
          4.272760538261085e-10, 3.662610153313769e-10]
    @test isapprox(w14.weights, wt)

    portfolio.cluster_adj = IP(; A = C)
    w15 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [0.0, 0.0, 0.0, 0.0, 6.313744038982638e-18, 0.0, 0.28939681985466575, 0.0, 0.0,
          0.0, 0.7106031801453343, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    @test isapprox(w15.weights, wt)

    portfolio.cluster_adj = SDP(; A = C)
    w16 = optimise!(portfolio; obj = obj, rm = rm)
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

    portfolio.cluster_adj = IP(; A = B)
    w17 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [0.0, 0.0, 0.0, 0.0, 0.7390777009270599, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
          0.0, 0.0, 0.26092229907294007, 0.0, 0.0, 0.0, 0.0]
    @test isapprox(w17.weights, wt)

    portfolio.cluster_adj = SDP(; A = B)
    w18 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [1.4680691018542715e-9, 4.859275738822192e-9, 0.015992802521425674,
          7.293948583023383e-10, 0.2988850375190448, 6.520933919634565e-11,
          2.128743388395745e-9, 0.16470653869543705, 4.624274795849735e-10,
          4.162708987445042e-10, 0.13583336339414312, 1.525142217002498e-11,
          2.494845225126199e-10, 1.0043814255652154e-9, 9.599421959136733e-11,
          0.2560211483665152, 4.277938501034575e-9, 1.880279259853607e-9,
          2.4445455761302985e-9, 0.12856108940616845]
    @test isapprox(w18.weights, wt)

    portfolio.cluster_adj = IP(; A = C)
    w19 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [0.0, 0.0, 0.0, 0.0, 0.3708533501847804, 0.0, 0.0, 0.0, 0.0, 0.0,
          0.3290504062456894, 0.0, 0.0, 0.0, 0.0, 0.3000962435695302, 0.0, 0.0, 0.0, 0.0]
    @test isapprox(w19.weights, wt)

    portfolio.cluster_adj = SDP(; A = C)
    w20 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [8.477768420059939e-10, 6.203676538852534e-10, 0.05312672191525725,
          3.525182424405551e-10, 0.3338885551569258, 2.412376047604793e-11,
          1.2535190055486572e-9, 0.10330170626208342, 2.6852440963129434e-10,
          3.0123040045012183e-10, 0.2215936600067496, 3.15192092477848e-11,
          1.19545885901302e-10, 5.880051207030092e-10, 9.099593094339936e-11,
          0.2880893428896174, 1.7533990524576498e-9, 6.973233642519765e-10,
          2.7663596217040672e-9, 4.0541581354667404e-9]
    @test isapprox(w20.weights, wt)
end

@testset "Cluster and Dendrogram non SD Short" begin
    portfolio = Portfolio(; prices = prices,
                          solvers = Dict(:PClGL => Dict(:solver => optimizer_with_attributes(Pajarito.Optimizer,
                                                                                             "verbose" => false,
                                                                                             "oa_solver" => optimizer_with_attributes(HiGHS.Optimizer,
                                                                                                                                      MOI.Silent() => true),
                                                                                             "conic_solver" => optimizer_with_attributes(Clarabel.Optimizer,
                                                                                                                                         "verbose" => false,
                                                                                                                                         "max_step_fraction" => 0.75)))))
    asset_statistics!(portfolio)

    A = centrality_vector(portfolio)
    B = connection_matrix(portfolio)
    C = cluster_matrix(portfolio)

    portfolio.short = true
    portfolio.short_budget = 0.13
    portfolio.short_u = 0.13
    portfolio.long_u = 1
    portfolio.budget = portfolio.long_u - portfolio.short_u

    rm = CDaR()
    obj = MinRisk()
    w1 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [0.0, -0.0033927423088478525, 0.0, -0.0017047316992726633, 0.02620966174092891,
          -0.048951276707458066, -0.02831053066644215, 0.0797084661157905,
          -0.006091550656354475, -0.01772949018065388, 0.34963750962788587,
          -0.009810028561542944, 0.006616206609090757, -1.7025233012195975e-16,
          -0.01400964921942798, 0.09738313723086256, 0.16435744872368938,
          0.01643568935570537, 0.10356751838407428, 0.15608436221197253]
    @test isapprox(w1.weights, wt)

    portfolio.a_vec_cent = A
    portfolio.b_cent = minimum(A)
    w2 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [-0.09486167717442834, 0.04019416692914485, 0.037577223179041455,
          -5.0655785389293176e-17, 0.12801741187716315, -7.979727989493313e-17,
          -0.01998095976345068, 0.13355629824656456, 0.019271038397774838, 0.0,
          0.28797176886025655, 0.0, 0.008845283123909234, 0.0, -0.015157363062120899,
          0.10734222426525322, 0.0, 0.061474266941201526, 4.119968255444917e-17,
          0.17575031817969053]
    @test isapprox(w2.weights, wt)

    portfolio.cluster_adj = IP(; A = B)
    w3 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [-0.0, 0.0, 0.0, -0.0, 0.25423835281660595, -0.10786045982487914, -0.0,
          0.3290798306064528, -0.0, -0.0, 2.220446049250313e-16, -0.0, 0.0, -0.0,
          0.02916010168515354, 0.13000000000000062, -0.0, 0.0, -0.0, 0.23538217471666556]
    @test isapprox(w3.weights, wt)

    portfolio.cluster_adj = SDP(; A = B, penalty = 0.5)
    w4 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [9.272435109943138e-12, 0.06109011163471849, 0.07388719114292448,
          0.019692527431470045, 0.10331812359040753, 0.08117018361193389,
          0.0233844026939607, 0.12195682880133857, 1.687054000923196e-11,
          0.034579519160302545, 0.02795355513256586, 0.05841592129946603,
          7.482492434260051e-12, 0.029412363249472755, 0.0020709632770030445,
          0.04322171854228879, -8.422304421932733e-13, 0.08680666625945271,
          -3.585326191328158e-11, 0.10303992417576462]
    @test isapprox(w4.weights, wt)

    portfolio.cluster_adj = IP(; A = C)
    w5 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [-0.0, -9.824982518806696e-17, -2.881746754291979e-16, -6.328762489489334e-17,
          0.3345846888561428, -0.0, -0.0, 0.4054153111438571, -0.0, -0.0, 0.0, 0.0, 0.0,
          -0.0, -0.0, 0.13000000000000017, -0.0, 0.0, -0.0, 0.0]
    @test isapprox(w5.weights, wt)

    portfolio.cluster_adj = SDP(; A = C, penalty = 0.5)
    w6 = optimise!(portfolio; obj = obj, rm = rm)
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

    portfolio.cluster_adj = IP(; A = B)
    w7 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [-0.0, -0.0, -0.0, 3.428633578839961e-17, 0.1119976188327676, -0.0, -0.0, 0.0, 0.0,
          0.0, 0.3750880542218098, -0.0, -0.0, -0.0, -0.0, 0.1412763749809966, -0.0,
          0.09588426469923997, 0.0, 0.14575368726518567]
    @test isapprox(w7.weights, wt)

    portfolio.cluster_adj = SDP(; A = B, penalty = 0.5)
    w8 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [6.766929954587163e-11, 0.05845600849974435, 0.06857721044042771,
          0.01863708044634632, 0.09512946481942187, 0.07994346654887266,
          0.02554585983141323, 0.07233232435238945, 5.714153552243313e-11,
          0.032112301662597306, 0.08806729718852686, 0.046990727441064695,
          2.302281082192624e-11, 0.041083140039804476, 0.001683242401425925,
          0.057989539665591214, 6.10236089865971e-11, 0.0822728003096611,
          3.5401881596356065e-11, 0.10117953610845386]
    @test isapprox(w8.weights, wt)

    portfolio.cluster_adj = IP(; A = C)
    w9 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 4.8338121699908494e-17, 0.0, -0.0, -0.0,
          0.425177042550851, -0.0, -1.0139121127832994e-15, -0.0, 2.2724003756919815e-16,
          0.05890972756969809, 0.3859132298794514, 0.0, -0.0, 0.0]
    @test isapprox(w9.weights, wt)

    portfolio.cluster_adj = SDP(; A = C, penalty = 0.5)
    w10 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [3.136684399611615e-10, 6.421446869383163e-11, 0.015548662828311242,
          5.302443988456493e-11, 0.07187792498761011, 3.443518625849509e-11,
          0.014549230934352904, 0.10906579897534709, 2.8604004872211283e-11,
          2.3132104858134452e-11, 0.23095115947815376, 0.044332984975470155,
          1.7809249411779717e-10, 4.072456628263281e-11, 0.005860251167461893,
          0.16234482039541365, 0.07541584072146104, 6.086150708285326e-11,
          5.436944458782648e-11, 0.14005332468529116]
    @test isapprox(w10.weights, wt)

    portfolio = Portfolio(; prices = prices,
                          solvers = Dict(:PClGL => Dict(:solver => optimizer_with_attributes(Pajarito.Optimizer,
                                                                                             "verbose" => false,
                                                                                             "oa_solver" => optimizer_with_attributes(HiGHS.Optimizer,
                                                                                                                                      MOI.Silent() => true),
                                                                                             "conic_solver" => optimizer_with_attributes(Clarabel.Optimizer,
                                                                                                                                         "verbose" => false,
                                                                                                                                         "max_step_fraction" => 0.75)))))
    asset_statistics!(portfolio)
    obj = Sharpe(; rf = rf)

    A = centrality_vector(portfolio; network_type = TMFG())
    B = connection_matrix(portfolio; network_type = TMFG())
    C = cluster_matrix(portfolio; hclust_alg = DBHT())

    portfolio.short = true
    portfolio.short_budget = 0.18
    portfolio.short_u = 0.18
    portfolio.long_u = 0.95
    portfolio.budget = portfolio.long_u - portfolio.short_u

    w11 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [0.0, 0.0, 0.06455652837124166, 0.0, 0.17806945118716286, -0.1534166903019555,
          6.554560797119506e-17, 0.12351856122756986, 0.0, 0.0, 0.21867000759778857, 0.0,
          -0.023295149141940166, 0.0, -0.0032881605561044004, 0.1770017931069312,
          0.006143447950208067, 0.0, 0.04071674553919559, 0.1413234650199022]
    @test isapprox(w11.weights, wt)

    portfolio.a_vec_cent = A
    portfolio.b_cent = minimum(A)

    w12 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [0.0, 0.0, 0.07328869996428454, 0.0, 0.19593630860627667, -0.1516035031548817,
          -2.228175066036319e-17, 0.1290693464408622, 0.0, 0.0, 0.20897763141227396, 0.0,
          -0.02192543103531701, 0.0, -0.006471065809801198, 0.18010601289300754,
          7.366923752795404e-17, 0.0, 0.025951685743898865, 0.13667031493939605]
    @test isapprox(w12.weights, wt)

    portfolio.cluster_adj = IP(; A = B)
    w13 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [-0.0, -0.0, 0.0, -0.0, 0.6900000000000023, -0.0, -0.0, 0.0, -0.0, -0.0,
          0.19949253798767527, -0.0, -0.11949253798767445, -0.0, -0.0, 0.0, 0.0, -0.0, 0.0,
          0.0]
    @test isapprox(w13.weights, wt)

    portfolio.cluster_adj = SDP(; A = B, penalty = 0.5)
    w14 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [3.514727061450158e-11, 0.10214522082786334, 1.5973615263620542e-9,
          8.750301284842235e-10, 0.23010239598778515, -0.025107535823589502,
          0.022839041934182035, 0.1484093093837071, -2.6374371040425877e-10,
          -3.153936811484647e-11, 0.12599682872769954, -6.592141286737157e-10,
          -0.03303460329907297, -1.0047674687409329e-10, -0.05977664458614574,
          0.17110148065348418, -7.483961562407158e-11, 5.423167113303609e-10,
          -2.6450191488394353e-10, 0.08732450453854651]
    @test isapprox(w14.weights, wt)

    portfolio.cluster_adj = IP(; A = C)
    w15 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [-0.0, -0.0, 0.0, -0.0, 0.6899999999999927, -0.0, -0.0, 0.0, -0.0, -0.0,
          0.1994925379876753, -0.0, -0.11949253798766485, -0.0, -0.0, 0.0, -0.0, -0.0, 0.0,
          0.0]
    @test isapprox(w15.weights, wt)

    portfolio.cluster_adj = SDP(; A = C, penalty = 0.5)
    w16 = optimise!(portfolio; obj = obj, rm = rm)
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

    portfolio.cluster_adj = IP(; A = B)
    w17 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [-0.0, -0.0, 0.0, -0.0, 0.6751155811095629, -0.0, -0.0, 0.0, -0.0, -0.0,
          0.208898224965609, -0.0, -0.11401380607517049, -0.0, -0.0, 0.0, 0.0, -0.0, -0.0,
          0.0]
    @test isapprox(w17.weights, wt)

    portfolio.cluster_adj = SDP(; A = B, penalty = 0.5)
    w18 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [9.396464919780623e-11, 0.09919154701647347, 6.368353684023622e-10,
          3.465978583702456e-10, 0.22064663460078363, -1.0614315396176674e-9,
          0.020025141314969998, 0.1266686482714095, 5.6795111561929726e-12,
          1.1754682802203758e-11, 0.12107267096223644, -4.262330938609224e-10,
          -0.04188688600867918, 1.5637094635344307e-10, -0.06521868018990389,
          0.16284169111496827, 6.247282655579725e-11, 4.814732694037535e-10,
          1.7210187767534034e-10, 0.12665923243815538]
    @test isapprox(w18.weights, wt)

    portfolio.cluster_adj = IP(; A = C)
    w19 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [-0.0, -0.0, 0.0, -0.0, 0.5385298461146729, -0.0, 0.0, 0.0, -0.0, -0.0, 0.0,
          -0.18000000000000382, -0.0, 0.0, 3.8379398333663466e-15, 0.411470153885331, -0.0,
          -0.0, -0.0, 0.0]
    @test isapprox(w19.weights, wt)

    portfolio.cluster_adj = SDP(; A = C, penalty = 0.5)
    w20 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [1.4172669439373994e-10, 1.1299075883203906e-10, 5.359434439554113e-10,
          8.26042302579247e-11, 0.3649711359888737, -6.90608391192042e-11,
          4.0350791884642237e-10, 2.0028585646891752e-10, 2.1735756891215644e-10,
          1.0421359032360126e-10, 0.11284986034990291, -6.494847118187474e-11,
          -0.027835392456788028, 1.767593240750613e-10, -0.015067449302819187,
          0.2566729326870001, 1.7852927573841912e-10, 7.917164427601336e-11,
          0.07840891046398553, 1.7076400157132798e-10]
    @test isapprox(w20.weights, wt, rtol = 5.0e-8)
end

@testset "Cluster + Network and Dendrogram SD" begin
    portfolio = Portfolio(; prices = prices,
                          solvers = Dict(:PClGL => Dict(:solver => optimizer_with_attributes(Pajarito.Optimizer,
                                                                                             "verbose" => false,
                                                                                             "oa_solver" => optimizer_with_attributes(HiGHS.Optimizer,
                                                                                                                                      MOI.Silent() => true),
                                                                                             "conic_solver" => optimizer_with_attributes(Clarabel.Optimizer,
                                                                                                                                         "verbose" => false,
                                                                                                                                         "max_step_fraction" => 0.75)))))
    asset_statistics!(portfolio)
    wc_statistics!(portfolio,
                   WCType(; box = NormalWC(; seed = 123456789),
                          ellipse = NormalWC(; seed = 123456789)))

    A = centrality_vector(portfolio)
    B = connection_matrix(portfolio)
    C = cluster_matrix(portfolio)

    rm = SD()
    obj = MinRisk()
    w1 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [0.00791850924098073, 0.030672065216453506, 0.010501402809199123,
          0.027475241969187995, 0.012272329269527841, 0.03339587076426262,
          1.4321532289072258e-6, 0.13984297866711365, 2.4081081597353397e-6,
          5.114425959766348e-5, 0.2878111114337346, 1.5306036912879562e-6,
          1.1917690994187655e-6, 0.12525446872321966, 6.630910273840812e-6,
          0.015078706008184504, 8.254970801614574e-5, 0.1930993918762575,
          3.0369340845779798e-6, 0.11652799957572683]
    @test isapprox(w1.weights, wt)

    wc1 = optimise!(portfolio; type = WC(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w1.weights, wc1.weights)

    portfolio.a_vec_cent = A
    portfolio.b_cent = minimum(A)
    w2 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [3.5426430661259544e-11, 0.07546661314221123, 0.021373033743425803,
          0.027539611826011074, 0.023741467255115698, 0.12266515815974924,
          2.517181275812244e-6, 0.22629893782442134, 3.37118211246211e-10,
          0.02416530860824232, 3.3950118687352606e-10, 1.742541411959769e-6,
          1.5253730188343444e-6, 5.304686980295978e-11, 0.024731789616991084,
          3.3711966852218057e-10, 7.767147353183488e-11, 0.30008056166009706,
          1.171415437554966e-10, 0.1539317317710031]
    @test isapprox(w2.weights, wt)
    @test isapprox(portfolio.b_cent, average_centrality(portfolio))

    portfolio.network_adj = IP(; A = B)
    portfolio.cluster_adj = IP(; A = C)
    w3 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [7.825727871557501e-12, 3.806030349693684e-11, 3.806345657743864e-11,
          7.834553546814647e-12, 3.78443800872837e-11, 3.758227260416078e-11,
          3.836700851910193e-11, 6.492150028704859e-12, 2.8327452397747455e-11,
          3.7710540677982595e-11, 2.8685817566372707e-11, 3.754875311121983e-11,
          3.88463148157298e-11, 1.775892388168607e-11, 0.053371401966537065,
          2.8746558919801268e-11, 3.064769046738463e-12, 0.5616138799977659,
          2.2082610124167892e-11, 0.3850147175808554]
    @test isapprox(w3.weights, wt)
    @test isapprox(portfolio.b_cent, average_centrality(portfolio))

    wc3 = optimise!(portfolio; type = WC(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w3.weights, wc3.weights)
    @test isapprox(portfolio.b_cent, average_centrality(portfolio))

    portfolio.network_adj = SDP(; A = B)
    portfolio.cluster_adj = SDP(; A = C)
    w4 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [5.909619388249067e-11, 9.253002289952112e-6, 5.41349519287628e-6,
          2.8449858084242797e-6, 4.182099210497449e-6, 4.9681774294981124e-6,
          2.697080811635782e-6, 1.5085206223431674e-5, 5.557471468919298e-10,
          2.6285153469086896e-6, 5.562294450987228e-10, 3.0537614518427532e-6,
          2.3009437863731846e-6, 8.818755387403218e-11, 0.053355215375154925,
          5.55895821466581e-10, 1.2822097636254883e-10, 0.5616323163167606,
          1.9234490511708156e-10, 0.38496003890481084]
    @test isapprox(w4.weights, wt)
    @test isapprox(portfolio.b_cent, average_centrality(portfolio))

    wc4 = optimise!(portfolio; type = WC(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w4.weights, wc4.weights)
    @test isapprox(portfolio.b_cent, average_centrality(portfolio))

    portfolio.network_adj = SDP(; A = B)
    portfolio.cluster_adj = IP(; A = C)
    w5 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [5.114622431555773e-14, 2.4455869446189113e-13, 2.4286395283431467e-13,
          2.4440704278222516e-13, 2.4299088335904587e-13, 2.4457662594403974e-13,
          2.4345301712496775e-13, 2.466897545799507e-13, 1.7678156747959555e-13,
          2.435376356615091e-13, 1.8349553059279157e-13, 0.00986655726313585,
          2.4152370921992144e-13, 1.1797457514775457e-13, 2.4746223609473365e-13,
          1.7926218652603644e-13, 1.1389891895170576e-14, 0.5902137542951235,
          1.4724177676685395e-13, 0.3999196884384313]
    @test isapprox(w5.weights, wt)
    @test isapprox(portfolio.b_cent, average_centrality(portfolio))

    wc5 = optimise!(portfolio; type = WC(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w5.weights, wc5.weights, rtol = 0.05)
    @test isapprox(portfolio.b_cent, average_centrality(portfolio))

    portfolio.network_adj = IP(; A = B)
    portfolio.cluster_adj = SDP(; A = C)
    w6 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [1.1624257456781998e-13, 5.591685437277831e-13, 5.596939271223075e-13,
          5.591718788185254e-13, 5.580563537967945e-13, 5.583544173587496e-13,
          5.72049631663973e-13, 5.556893374619858e-13, 4.168286783821205e-13,
          5.601683654125956e-13, 4.085406943442685e-13, 5.749624584192578e-13,
          5.729720414234623e-13, 2.622107163651453e-13, 5.659043281857597e-13,
          4.1968266619184597e-13, 3.340400013769717e-14, 0.5939747163881034,
          3.30726579670635e-13, 0.4060252836037128]
    @test isapprox(w6.weights, wt)
    @test isapprox(portfolio.b_cent, average_centrality(portfolio))

    wc6 = optimise!(portfolio; type = WC(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w6.weights, wc6.weights)
    @test isapprox(portfolio.b_cent, average_centrality(portfolio))

    portfolio.a_vec_cent = []
    portfolio.b_cent = Inf

    portfolio.network_adj = IP(; A = B)
    portfolio.cluster_adj = IP(; A = C)
    w7 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [5.197341385259319e-11, 5.1803327957307115e-11, 5.2003773513759864e-11,
          5.1995606743083794e-11, 5.2098251771110276e-11, 5.171993695676165e-11,
          5.1745079520810936e-11, 5.143525069676673e-11, 5.182266118210948e-11,
          5.168377355178661e-11, 0.625501515921244, 5.193395920525499e-11,
          5.147906698225565e-11, 5.1667889792416134e-11, 5.2174055439356466e-11,
          0.06615026333717866, 0.30834821986107064, 5.1507096579397706e-11,
          5.1809728900942206e-11, 5.1653867694434894e-11]
    @test isapprox(w7.weights, wt)

    wc7 = optimise!(portfolio; type = WC(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w7.weights, wc7.weights)

    portfolio.network_adj = SDP(; A = B)
    portfolio.cluster_adj = SDP(; A = C)
    w8 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [4.8985467719266855e-6, 3.62082005436798e-6, 2.6003609630782663e-6,
          1.6420744621590142e-6, 1.8367337434577754e-6, 1.3834676822080563e-6,
          1.4429039295754813e-6, 2.0642252162438465e-6, 6.48535112584547e-7,
          7.247120962787134e-7, 0.6119951330629139, 1.808807880906616e-6,
          7.110319137821802e-7, 4.42849778767468e-6, 6.815015220177915e-6,
          0.05683659753712471, 2.365322304205034e-5, 9.070833691582269e-6,
          9.850186034815847e-7, 0.33109993459179005]
    @test isapprox(w8.weights, wt)

    wc8 = optimise!(portfolio; type = WC(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w8.weights, wc8.weights)

    portfolio.network_adj = IP(; A = B)
    portfolio.cluster_adj = SDP(; A = C)
    w9 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [1.2544372156438123e-12, 1.2547018087342703e-12, 1.2544553604862145e-12,
          1.2526712061134886e-12, 1.2534254783129386e-12, 1.253573757923605e-12,
          1.2710481435665748e-12, 1.2506928489112165e-12, 1.2578074248522432e-12,
          1.25541361650739e-12, 0.6374450270670242, 1.2725603250393488e-12,
          1.273191157993477e-12, 1.2515917231645316e-12, 1.2686359908016167e-12,
          1.2684283414521342e-12, 1.255334951961596e-12, 1.2510564694926944e-12,
          1.256126016734311e-12, 0.36255497291032074]
    @test isapprox(w9.weights, wt)

    wc9 = optimise!(portfolio; type = WC(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w9.weights, wc9.weights)

    portfolio.network_adj = SDP(; A = B)
    portfolio.cluster_adj = IP(; A = C)
    w10 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [2.2001469743749958e-13, 2.2077198163370508e-13, 2.198678853470032e-13,
          2.2028650556530205e-13, 2.1980141600894831e-13, 2.2117194146494213e-13,
          0.015445831964940747, 2.229019960870065e-13, 2.1841459368578927e-13,
          2.2017511371522347e-13, 2.2382858066395749e-13, 2.180741084136083e-13,
          2.192536991018764e-13, 2.2250985782954765e-13, 2.2261215975968928e-13,
          2.2072099477418895e-13, 2.306383089388056e-13, 0.5831641853034233,
          2.2617130370758232e-13, 0.4013899827278688]
    @test isapprox(w10.weights, wt, rtol = 1.2)

    wc10 = optimise!(portfolio; type = WC(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w10.weights, wc10.weights, rtol = 1.3)

    portfolio = Portfolio(; prices = prices,
                          solvers = Dict(:PClGL => Dict(:solver => optimizer_with_attributes(Pajarito.Optimizer,
                                                                                             "verbose" => false,
                                                                                             "oa_solver" => optimizer_with_attributes(HiGHS.Optimizer,
                                                                                                                                      MOI.Silent() => true),
                                                                                             "conic_solver" => optimizer_with_attributes(Clarabel.Optimizer,
                                                                                                                                         "verbose" => false,
                                                                                                                                         "max_step_fraction" => 0.75)))))
    asset_statistics!(portfolio)
    obj = Sharpe(; rf = rf)

    network_type = TMFG()
    A = centrality_vector(portfolio; network_type = network_type)
    B = connection_matrix(portfolio; network_type = network_type)
    C = cluster_matrix(portfolio; hclust_alg = DBHT())

    w11 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [5.2456519308516375e-9, 1.3711772920494845e-8, 1.828307590553689e-8,
          1.1104551657521415e-8, 0.5180586238310904, 1.7475553033958716e-9,
          0.06365101880957791, 1.1504018896467816e-8, 1.0491794751235226e-8,
          5.650081752041375e-9, 9.262635444153261e-9, 1.4051602963640009e-9,
          9.923106257589635e-10, 3.050812431838031e-9, 9.966076078173243e-10,
          0.1432679499627637, 0.19649701265872604, 1.0778757180749977e-8,
          0.07852527921167819, 1.1301377011579277e-8]
    @test isapprox(w11.weights, wt)

    wc11 = optimise!(portfolio; type = WC(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w11.weights, wc11.weights, rtol = 5.0e-6)

    portfolio.a_vec_cent = A
    portfolio.b_cent = minimum(A)

    w12 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [3.580964533638559e-11, 9.277053125069126e-10, 9.835000020421294e-10,
          0.48362433627647977, 1.3062621693742353e-9, 3.4185103123124565e-11,
          0.28435553885288534, 0.17984723049407092, 1.693432131576565e-10,
          2.621850918060399e-10, 0.05217287791621345, 5.161470450551339e-9,
          2.4492072342885186e-9, 4.790090083985224e-11, 2.6687808589346464e-9,
          1.0539691276230515e-9, 2.0827050059613154e-10, 8.48619997995145e-10,
          2.329804826102029e-10, 7.01603778985698e-11]
    @test isapprox(w12.weights, wt)
    @test isapprox(portfolio.b_cent,
                   average_centrality(portfolio; network_type = network_type))

    wc12 = optimise!(portfolio; type = WC(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w12.weights, wc12.weights, rtol = 5.0e-5)
    @test isapprox(portfolio.b_cent,
                   average_centrality(portfolio; network_type = network_type))

    portfolio.cluster_adj = IP(; A = B)
    portfolio.cluster_adj = IP(; A = C)
    w13 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [3.5488808869733083e-12, 8.281743679661074e-12, 8.80902177574244e-12,
          9.955072746064901e-12, 1.1134822019504797e-11, 4.7417114086610015e-12,
          0.6074274195608048, 1.0253105313724553e-11, 1.4148918790428058e-12,
          6.3499711667509184e-12, 0.39257258031345454, 9.215307590460059e-12,
          7.61409793255415e-12, 3.5761100061677828e-12, 7.3543478626708e-12,
          9.27354997599413e-12, 3.8374099382272584e-12, 8.458519745181501e-12,
          6.62153510029776e-12, 5.3005368572386036e-12]
    @test isapprox(w13.weights, wt, rtol = 1.0e-7)
    @test isapprox(portfolio.b_cent,
                   average_centrality(portfolio; network_type = network_type))

    wc13 = optimise!(portfolio; type = WC(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w13.weights, wc13.weights, rtol = 1.0e-5)
    @test isapprox(portfolio.b_cent,
                   average_centrality(portfolio; network_type = network_type))

    portfolio.network_adj = SDP(; A = B)
    portfolio.cluster_adj = SDP(; A = C)
    w14 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [1.7130940506947144e-14, 5.472152155764947e-13, 5.484832393170935e-13,
          0.3813543675333245, 5.61298558468235e-13, 3.76667203405565e-14,
          2.7631956774756272e-8, 2.4645270594479162e-8, 9.586191060960748e-14,
          1.6837532851455495e-13, 0.618645573730773, 3.120649394135478e-9,
          1.6414007635973589e-9, 1.8200628342042637e-14, 1.693240653689941e-9,
          5.492876185862781e-13, 1.1845691238532438e-13, 5.455316921642039e-13,
          1.3225726310349946e-13, 4.4394449945462336e-14]
    @test isapprox(w14.weights, wt)
    @test isapprox(portfolio.b_cent,
                   average_centrality(portfolio; network_type = network_type))

    wc14 = optimise!(portfolio; type = WC(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w14.weights, wc14.weights, rtol = 5.0e-6)
    @test isapprox(portfolio.b_cent,
                   average_centrality(portfolio; network_type = network_type))

    portfolio.network_adj = IP(; A = B)
    portfolio.cluster_adj = SDP(; A = C)
    w15 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [1.1205964905960483e-15, 3.2803647004915935e-15, 3.3859019317511605e-15,
          0.38135127828825593, 4.291434611264293e-15, 2.762746037305381e-15,
          5.018110136333574e-15, 3.982513247994438e-15, 6.373793831546046e-16,
          2.492301114872751e-15, 0.6186487217116904, 4.1385804585537326e-15,
          4.144045359826403e-15, 1.5797528011511813e-15, 4.184761787981488e-15,
          3.518227610193536e-15, 1.59927853883942e-15, 3.223117527272749e-15,
          2.5183160151224432e-15, 1.7723567961783658e-15]
    @test isapprox(w15.weights, wt, rtol = 5.0e-7)
    @test isapprox(portfolio.b_cent,
                   average_centrality(portfolio; network_type = network_type))

    wc15 = optimise!(portfolio; type = WC(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w15.weights, wc15.weights, rtol = 1.0e-6)
    @test isapprox(portfolio.b_cent,
                   average_centrality(portfolio; network_type = network_type))

    portfolio.network_adj = SDP(; A = B)
    portfolio.cluster_adj = IP(; A = C)
    w16 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [1.0910778825706367e-15, 3.161394669980802e-15, 3.264088553405086e-15,
          0.38135132868675464, 4.111867541106487e-15, 2.625839871137247e-15,
          4.8150698284604255e-15, 3.8292329839591e-15, 5.948736324756478e-16,
          2.402175454242592e-15, 0.6186486713131937, 3.968888569780334e-15,
          3.940598616340146e-15, 1.5056949104399084e-15, 3.981704392022144e-15,
          3.3882773131640016e-15, 1.5186674763459897e-15, 3.1021374213375176e-15,
          2.399022600928559e-15, 1.71568552640857e-15]
    @test isapprox(w16.weights, wt)
    @test isapprox(portfolio.b_cent,
                   average_centrality(portfolio; network_type = network_type))

    wc16 = optimise!(portfolio; type = WC(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w16.weights, wc16.weights, rtol = 5.0e-6)
    @test isapprox(portfolio.b_cent,
                   average_centrality(portfolio; network_type = network_type))

    portfolio.a_vec_cent = []
    portfolio.b_cent = Inf

    portfolio.network_adj = IP(; A = B)
    portfolio.cluster_adj = IP(; A = C)
    w17 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [2.0966673826999653e-12, 2.198260339765037e-12, 1.9669534265126797e-12,
          2.0537353944645557e-12, 0.762282392772933, 2.61589719142697e-12,
          1.6397732202694488e-12, 2.573524529021814e-12, 2.1635532147945916e-12,
          2.3920718749431715e-12, 2.6031535089914636e-12, 2.254849729224801e-12,
          2.3351897528966937e-12, 2.5746406212787903e-12, 2.4339077077727048e-12,
          0.23771760718605286, 2.077497433772476e-12, 2.544464994811904e-12,
          2.2242733110585934e-12, 2.2657939494042705e-12]
    @test isapprox(w17.weights, wt)

    wc17 = optimise!(portfolio; type = WC(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w17.weights, wc17.weights, rtol = 0.0005)

    portfolio.network_adj = SDP(; A = B)
    portfolio.cluster_adj = SDP(; A = C)
    w18 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [1.9429668083392023e-7, 1.8953039886559254e-7, 2.2663313286802088e-7,
          1.1285789421938343e-7, 0.4172071333310411, 8.189670543963489e-8,
          5.3331240834762504e-8, 1.6014263723499125e-7, 2.428717976450698e-7,
          2.857532755915146e-7, 0.42083278525974904, 4.217900919215878e-8,
          3.290880791493152e-8, 2.2190598070649336e-7, 3.086752925527749e-8,
          0.16195597634253983, 7.493345525821349e-7, 2.3801140452326176e-7,
          9.862095406523452e-7, 2.5633608163451095e-7]
    @test isapprox(w18.weights, wt, rtol = 5.0e-8)

    wc18 = optimise!(portfolio; type = WC(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w18.weights, wc18.weights, rtol = 5.0e-5)

    portfolio.network_adj = IP(; A = B)
    portfolio.cluster_adj = SDP(; A = C)
    w19 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [7.516835373992114e-15, 7.515391914082471e-15, 7.582195074433755e-15,
          7.5363518859814e-15, 0.4882257537416507, 7.406381311672584e-15,
          8.45848560796015e-15, 7.436019933863325e-15, 7.516040038347184e-15,
          7.468824022093513e-15, 0.5117742462582129, 7.40452338495203e-15,
          7.577331097777585e-15, 7.40966187118098e-15, 7.542271326105534e-15,
          7.793739835627654e-15, 7.61017974899799e-15, 7.454425214104263e-15,
          7.522563207652192e-15, 7.514210635173347e-15]
    @test isapprox(w19.weights, wt, rtol = 5.0e-7)

    wc19 = optimise!(portfolio; type = WC(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w19.weights, wc19.weights, rtol = 1.0)

    portfolio.network_adj = SDP(; A = B)
    portfolio.cluster_adj = IP(; A = C)
    w20 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [4.306686593227565e-15, 4.3022182649128705e-15, 4.324425280755364e-15,
          4.307926180599051e-15, 0.4172073233039717, 4.368924067985393e-15,
          4.6684967426995244e-15, 4.271166688479858e-15, 4.301638623911586e-15,
          4.288865051718612e-15, 0.42085264277266976, 4.341406512522108e-15,
          4.5565532543739786e-15, 4.281361632836901e-15, 4.538133214696471e-15,
          0.16194003392328452, 4.3343395991262894e-15, 4.2808806310558155e-15,
          4.3034132757055466e-15, 4.308466946921385e-15]
    @test isapprox(w20.weights, wt, rtol = 5.0e-8)

    wc20 = optimise!(portfolio; type = WC(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w20.weights, wc20.weights, rtol = 1e-5)
end

@testset "Cluster + Network and Dendrogram SD short" begin
    portfolio = Portfolio(; prices = prices, short_budget = 10.0,
                          solvers = Dict(:PClGL => Dict(:check_sol => (allow_local = true,
                                                                       allow_almost = true),
                                                        :solver => optimizer_with_attributes(Pajarito.Optimizer,
                                                                                             "verbose" => false,
                                                                                             "oa_solver" => optimizer_with_attributes(HiGHS.Optimizer,
                                                                                                                                      MOI.Silent() => true),
                                                                                             "conic_solver" => optimizer_with_attributes(Clarabel.Optimizer,
                                                                                                                                         "verbose" => false,
                                                                                                                                         "max_step_fraction" => 0.75)))))
    asset_statistics!(portfolio)
    wc_statistics!(portfolio,
                   WCType(; box = NormalWC(; seed = 123456789),
                          ellipse = NormalWC(; seed = 123456789)))
    portfolio.short = true
    portfolio.short_budget = 0.22
    portfolio.short_u = 0.22
    portfolio.long_u = 0.88
    portfolio.budget = portfolio.long_u - portfolio.short_u

    A = centrality_vector(portfolio)
    B = connection_matrix(portfolio)
    C = cluster_matrix(portfolio)

    rm = SD()
    obj = MinRisk()
    w1 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [0.0023732456580253273, 0.02478417662625044, 0.011667587612994237,
          0.021835859830789766, 0.008241287060153578, 0.03550224737381907,
          -0.006408569798470374, 0.09320297606692711, -0.0072069893239688695,
          0.012269888650642718, 0.18722784970990347, -0.01395210113745165,
          -0.006026909793594887, 0.0962770551826336, 0.0005110433722419729,
          0.016784621725152556, 0.009614188864666265, 0.13436280943955117,
          -0.042344878358121694, 0.08128461123785628]
    @test isapprox(w1.weights, wt)

    wc1 = optimise!(portfolio; type = WC(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w1.weights, wc1.weights)

    portfolio.a_vec_cent = A
    portfolio.b_cent = minimum(A)
    w2 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [0.0064520166328966245, 0.022489884719278264, 0.01029468309633522,
          0.020902946113760288, 0.00711671166632188, 0.03337768007882898,
          -0.006266220580382531, 0.09139178524192762, -0.02237340947148962,
          0.010954134035283628, 0.18619094237390604, -0.014102818325104587,
          -0.0055880934328339325, 0.09574848799807273, 0.0002468303788802295,
          0.01712117146324995, 0.014137258176396965, 0.13074888415899583,
          -0.01805679382768158, 0.07921391950335803]
    @test isapprox(w2.weights, wt)
    @test isapprox(portfolio.b_cent, average_centrality(portfolio))

    portfolio.network_adj = IP(; A = B)
    portfolio.cluster_adj = IP(; A = C)
    w3 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [-5.289552410167509e-12, -2.4997622993523034e-12, -2.6512967994053205e-12,
          -2.4547780606726618e-12, -2.6461597546995527e-12, -2.5283221365738343e-12,
          -2.393948898681117e-12, -2.381332384949783e-12, -3.3950276982279217e-12,
          -2.5699476192293362e-12, 0.34000000011239045, 0.017854881294650442,
          -2.462514166900806e-12, -4.077219875721461e-12, -2.4135263339641155e-12,
          -3.3275495702341715e-12, -6.429029515090798e-12, -2.3877495454258555e-12,
          -9.321489076489813e-12, 0.30214511865218857]
    @test isapprox(w3.weights, wt)
    @test isapprox(portfolio.b_cent, average_centrality(portfolio))

    wc3 = optimise!(portfolio; type = WC(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w3.weights, wc3.weights, rtol = 0.05)
    @test isapprox(portfolio.b_cent, average_centrality(portfolio))

    portfolio.network_adj = SDP(; A = B)
    portfolio.cluster_adj = SDP(; A = C)
    w4 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [8.860278617521361e-7, 1.8219036928649961e-6, 1.9512172097817713e-6,
          0.000599483417716642, 9.362218017661003e-7, 9.064785845011358e-7,
          4.590695260155553e-7, 2.4142204504366963e-6, 7.06771740264861e-8,
          4.247489806068837e-7, 0.32592401548242295, 1.191916760551887e-6,
          8.516871868616484e-8, 9.590413892182947e-7, 0.016290534854604282,
          0.014063779590494504, 2.032895184160878e-6, 0.054489031741986174,
          -9.558286536711943e-8, 0.2486191109083065]
    @test isapprox(w4.weights, wt)
    @test isapprox(portfolio.b_cent, average_centrality(portfolio))

    wc4 = optimise!(portfolio; type = WC(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w4.weights, wc4.weights)
    @test isapprox(portfolio.b_cent, average_centrality(portfolio))

    portfolio.network_adj = IP(; A = B)
    portfolio.cluster_adj = SDP(; A = C)
    w5 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [-2.078204924849056e-11, -8.230675490985097e-12, -8.231562101631194e-12,
          -8.228283332273618e-12, -8.229272495886211e-12, -8.230184812199766e-12,
          -8.238216481817404e-12, -8.224653650606202e-12, -1.193210612781556e-11,
          -8.233645238811708e-12, 0.34000000044629464, 0.017854374625966764,
          -8.243354685882791e-12, -1.6109050232757797e-11, -8.23196150628087e-12,
          -1.1928819036423143e-11, -2.5909833849309036e-11, -8.224848166181637e-12,
          -3.737181191329823e-11, 0.30214562514231924]
    @test isapprox(w5.weights, wt)
    @test isapprox(portfolio.b_cent, average_centrality(portfolio))

    wc5 = optimise!(portfolio; type = WC(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w5.weights, wc5.weights)
    @test isapprox(portfolio.b_cent, average_centrality(portfolio))

    portfolio.network_adj = SDP(; A = B)
    portfolio.cluster_adj = IP(; A = C)
    w6 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [-4.770023832350059e-12, -5.635586402064825e-12, -5.6366780159866025e-12,
          -5.631727890316738e-12, -5.634530984171582e-12, -5.638567825410028e-12,
          -5.64294455541182e-12, -5.6290505345048275e-12, -5.705922428379741e-12,
          -5.6389432908303395e-12, 0.6525712951809018, 0.0855715286932582,
          -5.651156843948871e-12, -5.399077480788593e-12, -5.643065761197434e-12,
          -5.7043784998364415e-12, -0.07814282378260566, -5.631205533345361e-12,
          -2.327254410869679e-12, -5.635809731291216e-12]
    @test isapprox(w6.weights, wt)
    @test isapprox(portfolio.b_cent, average_centrality(portfolio))

    wc6 = optimise!(portfolio; type = WC(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w6.weights, wc6.weights)
    @test isapprox(portfolio.b_cent, average_centrality(portfolio))

    portfolio.a_vec_cent = []
    portfolio.b_cent = Inf

    portfolio.network_adj = IP(; A = B)
    portfolio.cluster_adj = IP(; A = C)
    w7 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [0.17086295092148623, -3.303826507423286e-12, -3.505235465309721e-12,
          -3.356743994009819e-12, -3.4850791736746247e-12, -3.3013413775197147e-12,
          -3.2370166855925405e-12, -3.1578736677517037e-12, -3.4083700754515744e-12,
          -3.3550498434371606e-12, 0.4652962477712207, 0.023840801366298098,
          -6.05252238559656e-12, -3.184791337088874e-12, -3.2296925614543096e-12,
          -3.3191254331698436e-12, -3.2953623304255398e-12, -3.1594902698825325e-12,
          -3.33106584098666e-12, -3.3221422104639324e-12]
    @test isapprox(w7.weights, wt)

    wc7 = optimise!(portfolio; type = WC(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w7.weights, wc7.weights)

    portfolio.network_adj = SDP(; A = B)
    portfolio.cluster_adj = SDP(; A = C)
    w8 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [1.6756974218841814e-6, 1.1533951915350161e-6, 8.124709341410984e-7,
          4.882016439038582e-7, 5.7198752241851e-7, 5.075039687578658e-7,
          2.789172307833894e-7, 9.013540249376955e-7, 1.642402794489107e-7,
          2.4602912179626235e-7, 0.40395258257675787, 2.565850666542391e-7,
          7.268139489094041e-8, 1.504622558734627e-6, 1.3751843464661264e-6,
          0.03754373324797069, 7.834700298200092e-6, 2.8878537026527065e-6,
          3.488321141470234e-7, 0.21848260391845015]
    @test isapprox(w8.weights, wt)

    wc8 = optimise!(portfolio; type = WC(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w8.weights, wc8.weights)

    portfolio.network_adj = IP(; A = B)
    portfolio.cluster_adj = SDP(; A = C)
    w9 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [-2.0577697872082403e-11, -2.057850795384142e-11, -2.0577279189428685e-11,
          -2.0574079620839933e-11, -2.0575347316619462e-11, -2.0573674385393817e-11,
          -2.0595346753387385e-11, -2.0566597722485122e-11, -2.0587896535957797e-11,
          -2.0578170497739385e-11, 0.43316919790962216, -2.058276882879673e-11,
          -2.0582222622414205e-11, -2.0573136298393467e-11, -2.058252087315015e-11,
          -2.0576275312274478e-11, 0.226830802460788, -2.057078644668308e-11,
          -2.0583533708920115e-11, -2.0574209051813953e-11]
    @test isapprox(w9.weights, wt, rtol = 0.5)

    wc9 = optimise!(portfolio; type = WC(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w9.weights, wc9.weights, rtol = 0.5)

    portfolio.network_adj = SDP(; A = B)
    portfolio.cluster_adj = IP(; A = C)
    w10 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [-1.536480343749736e-11, -1.5364452719983088e-11, -1.53669762443374e-11,
          -1.536028956113302e-11, -1.5364661989815024e-11, -1.53706430140619e-11,
          -1.537822672795077e-11, -1.53546497735517e-11, -1.537099012388435e-11,
          -1.5370169798859178e-11, 0.6078269008618088, 0.05217309941480593,
          -1.5384760352562456e-11, -1.5358671746614204e-11, -1.5377825792048056e-11,
          -1.536753000649473e-11, -1.5365131679664878e-11, -1.5360350385833858e-11,
          -1.537078312633907e-11, -1.5363101712520454e-11]
    @test isapprox(w10.weights, wt)

    wc10 = optimise!(portfolio; type = WC(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w10.weights, wc10.weights)

    portfolio = Portfolio(; prices = prices,
                          solvers = Dict(:PClGL => Dict(:solver => optimizer_with_attributes(Pajarito.Optimizer,
                                                                                             "verbose" => false,
                                                                                             "oa_solver" => optimizer_with_attributes(HiGHS.Optimizer,
                                                                                                                                      MOI.Silent() => true),
                                                                                             "conic_solver" => optimizer_with_attributes(Clarabel.Optimizer,
                                                                                                                                         "verbose" => false,
                                                                                                                                         "max_step_fraction" => 0.75)))))
    asset_statistics!(portfolio)
    wc_statistics!(portfolio,
                   WCType(; box = NormalWC(; seed = 123456789),
                          ellipse = NormalWC(; seed = 123456789)))
    portfolio.short = true
    portfolio.short_budget = 0.27
    portfolio.short_u = 0.27
    portfolio.long_u = 0.81
    portfolio.budget = portfolio.long_u - portfolio.short_u

    obj = Sharpe(; rf = rf)
    network_type = TMFG()
    A = centrality_vector(portfolio; network_type = network_type)
    B = connection_matrix(portfolio; network_type = network_type)
    C = cluster_matrix(portfolio; hclust_alg = DBHT())

    w11 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [1.3566877119847358e-9, 2.0446966457964675e-8, 0.002262512310895483,
          1.4309648003585085e-7, 0.2734983203742686, -0.10965529909211759,
          0.03892353698664605, 0.01819498928305758, 5.900712314743839e-9,
          1.0611222523062497e-8, 0.039700687003729467, -0.05489428919436992,
          -0.028441439852089117, 6.842779119067803e-10, -0.07700892954655005,
          0.10494492437890107, 0.14627523180964974, 6.830986121056796e-8,
          0.18619935145952116, 1.5367224934418446e-7]
    @test isapprox(w11.weights, wt)

    wc11 = optimise!(portfolio; type = WC(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w11.weights, wc11.weights, rtol = 5.0e-5)

    portfolio.a_vec_cent = A
    portfolio.b_cent = minimum(A)

    w12 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [2.1497119336493176e-10, 0.0088070179228199, 0.029397048718257417,
          0.019022694413270754, 0.2828940619983948, -0.12224670576806218,
          0.04353925700174413, 0.04156276108416902, 1.3097764426973018e-7,
          1.4902868835689554e-7, 0.09605820879721724, -0.04855626421146047,
          -0.029205822936867836, 1.055694863453504e-9, -0.0699910697420536,
          0.11603391472918954, 0.04641077698276719, 0.04812947034812958,
          0.07616841032531034, 0.001975959060175573]
    @test isapprox(w12.weights, wt)
    @test isapprox(portfolio.b_cent,
                   average_centrality(portfolio; network_type = network_type))

    wc12 = optimise!(portfolio; type = WC(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w12.weights, wc12.weights, rtol = 5e-5)
    @test isapprox(portfolio.b_cent,
                   average_centrality(portfolio; network_type = network_type))

    portfolio.network_adj = IP(; A = B)
    portfolio.cluster_adj = IP(; A = C)
    w13 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [1.5269128820151116e-10, 5.0153063190116795e-11, 5.600508563045624e-11,
          -0.1499999969120081, 5.516614800822251e-11, 8.547041188790406e-11,
          3.5801273917717314e-11, 1.256273861351736e-11, 2.409947687735074e-10,
          0.6899999950792018, 1.3622829299878125e-11, 4.821233895750819e-11,
          6.845347024420311e-11, 1.1561363674032775e-10, 6.517378250466465e-11,
          5.779913743888119e-11, 2.8740320583775717e-10, 4.066302469648558e-11,
          3.2976327105495704e-10, 1.1724076184220898e-10]
    @test isapprox(w13.weights, wt)
    @test isapprox(portfolio.b_cent,
                   average_centrality(portfolio; network_type = network_type))

    wc13 = optimise!(portfolio; type = WC(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w13.weights, wc13.weights)
    @test isapprox(portfolio.b_cent,
                   average_centrality(portfolio; network_type = network_type))

    portfolio.network_adj = SDP(; A = B)
    portfolio.cluster_adj = SDP(; A = C)
    w14 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [3.9226732613422946e-8, 4.1378325011325356e-8, -8.746772919211594e-5,
          -5.830649729199822e-8, 0.24605534441653706, -2.444664623236618e-7,
          1.1179465039707675e-8, 1.9673138196197508e-8, 8.971833108922652e-8,
          2.5956115822091557e-8, 0.14618224870911645, -3.182942523156022e-7,
          -0.010483763375852995, 3.3646696541465275e-8, -3.641559331499138e-7,
          0.060764433503729, 4.1493812695216685e-7, 3.3724592528147524e-8,
          0.097569400182017, 8.007526693050935e-8]
    @test isapprox(w14.weights, wt)
    @test isapprox(portfolio.b_cent,
                   average_centrality(portfolio; network_type = network_type))

    wc14 = optimise!(portfolio; type = WC(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w14.weights, wc14.weights, rtol = 0.0005)
    @test isapprox(portfolio.b_cent,
                   average_centrality(portfolio; network_type = network_type))

    portfolio.network_adj = IP(; A = B)
    portfolio.cluster_adj = SDP(; A = C)
    w15 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [2.5854260768969554e-16, 2.429369047954835e-16, 2.2107051349605093e-16,
          -0.14999999999999536, 2.251897550471393e-16, 2.2960294016237113e-16,
          1.8851294007089998e-16, 2.8548355651830353e-16, 4.671639690568098e-16,
          0.6899999999999902, 2.808050013242852e-16, 2.1205944823914323e-16,
          1.5294737258345109e-16, 2.40054004142672e-16, 1.6134142201477707e-16,
          2.1346354730278497e-16, 6.26058470243299e-16, 2.8068803939673705e-16,
          6.780177166985372e-16, 2.0206034249856564e-16]
    @test isapprox(w15.weights, wt)
    @test isapprox(portfolio.b_cent,
                   average_centrality(portfolio; network_type = network_type))

    wc15 = optimise!(portfolio; type = WC(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w15.weights, wc15.weights)
    @test isapprox(portfolio.b_cent,
                   average_centrality(portfolio; network_type = network_type))

    portfolio.network_adj = SDP(; A = B)
    portfolio.cluster_adj = IP(; A = C)
    w16 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [1.1090393091590862e-15, 1.1109594462365323e-15, 1.1472251481809554e-15,
          1.1349263016935252e-15, 0.34793141491098084, 6.838898048105854e-16,
          1.1375534936239241e-15, 9.201489865610157e-16, 1.1294560043852071e-15,
          1.0009646731472268e-15, 9.009008394263488e-16, 7.209357874694432e-16,
          4.115169441387524e-16, 8.187342841150198e-16, 4.499256568804305e-16,
          0.1080685850890053, 1.1115986295346577e-15, 9.633757622308786e-16,
          0.08399999999999802, 1.0873218010131636e-15]
    @test isapprox(w16.weights, wt)
    @test isapprox(portfolio.b_cent,
                   average_centrality(portfolio; network_type = network_type))

    wc16 = optimise!(portfolio; type = WC(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w16.weights, wc16.weights, rtol = 5.0e-7)
    @test isapprox(portfolio.b_cent,
                   average_centrality(portfolio; network_type = network_type))

    portfolio.a_vec_cent = []
    portfolio.b_cent = Inf

    portfolio.network_adj = IP(; A = B)
    portfolio.cluster_adj = IP(; A = C)
    w17 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [-0.0, -0.0, -0.0, -0.0, 0.8099999999999998, -0.0, -0.0, -0.0, -0.0, -0.0, 0.0,
          -0.0, -0.0, -0.0, -0.27, -0.0, 0.0, -0.0, 0.0, -0.0]
    @test isapprox(w17.weights, wt, rtol = 5.0e-8)

    wc17 = optimise!(portfolio; type = WC(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w17.weights, wc17.weights, rtol = 5.0e-7)

    portfolio.network_adj = SDP(; A = B)
    portfolio.cluster_adj = SDP(; A = C)
    w18 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [7.017281710184748e-9, 6.686407772536674e-9, 3.5438945128797934e-8,
          1.993959996194112e-8, 0.24448653858401273, -1.7203965755150878e-6,
          -2.4812836252360975e-7, -4.291664458966747e-9, 2.9621506257754306e-8,
          4.3967271314828195e-10, 0.20436645032561826, -3.116614069447712e-7,
          -0.00023860884054438986, 4.883426464894677e-9, -0.0014918033184182409,
          0.09284780015050982, 3.0271051194004177e-5, -2.5049451606485735e-7,
          1.6113587414565013e-6, 1.71634571927112e-7]
    @test isapprox(w18.weights, wt)

    wc18 = optimise!(portfolio; type = WC(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w18.weights, wc18.weights, rtol = 0.00005)

    portfolio.network_adj = IP(; A = B)
    portfolio.cluster_adj = SDP(; A = C)
    w19 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [1.3159480238575351e-15, 1.3161225950935655e-15, 1.3166098995143563e-15,
          1.316510767965194e-15, 0.2815062379020216, 1.312966753461982e-15,
          1.3186259626789107e-15, 1.315167994096295e-15, 1.3160060456322172e-15,
          1.3152087842963287e-15, 0.2846086094275412, 1.3128011843382446e-15,
          -0.02611484732958527, 1.3141787261837519e-15, 1.311433987986354e-15,
          1.3170785084395453e-15, 1.3166508979077908e-15, 1.3152264399741363e-15,
          1.3161610190181426e-15, 1.3159078785579517e-15]
    @test isapprox(w19.weights, wt, rtol = 5.0e-6)

    wc19 = optimise!(portfolio; type = WC(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w19.weights, wc19.weights, rtol = 5.0e-5)

    portfolio.network_adj = SDP(; A = B)
    portfolio.cluster_adj = IP(; A = C)
    w20 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [1.0600689683895216e-15, 1.058254958331794e-15, 1.0530565895145853e-15,
          1.0551136825774455e-15, 0.243039705717105, 1.072060052114558e-15,
          1.027980663893521e-15, 1.0682913971726322e-15, 1.0569594998418415e-15,
          1.0658688571876108e-15, 0.20487489455208943, 1.0719803149756893e-15,
          1.0722057495043067e-15, 1.0719578829918942e-15, 1.0713884492758524e-15,
          0.0920853997307875, 1.052312516938842e-15, 1.0670473518529165e-15,
          1.0561410633592871e-15, 1.0613532078139645e-15]
    @test isapprox(w20.weights, wt, rtol = 1.0e-6)

    wc20 = optimise!(portfolio; type = WC(; mu = NoWC(), cov = NoWC()), obj = obj)
    @test isapprox(w20.weights, wc20.weights, rtol = 1.0e-5)
end

@testset "Cluster + Network and Dendrogram upper dev" begin
    portfolio = Portfolio(; prices = prices,
                          solvers = Dict(:PClGL => Dict(:check_sol => (allow_local = true,
                                                                       allow_almost = true),
                                                        :solver => optimizer_with_attributes(Pajarito.Optimizer,
                                                                                             "verbose" => false,
                                                                                             "oa_solver" => optimizer_with_attributes(HiGHS.Optimizer,
                                                                                                                                      MOI.Silent() => true),
                                                                                             "conic_solver" => optimizer_with_attributes(Clarabel.Optimizer,
                                                                                                                                         "verbose" => false,
                                                                                                                                         "max_step_fraction" => 0.75)))))
    asset_statistics!(portfolio)

    C = cluster_matrix(portfolio)
    B = connection_matrix(portfolio; network_type = TMFG())

    rm = SD()
    w1 = optimise!(portfolio; obj = Sharpe(; rf = rf), rm = rm)
    r1 = calc_risk(portfolio; rm = rm)

    rm.settings.ub = r1
    portfolio.network_adj = IP(; A = B)
    portfolio.cluster_adj = IP(; A = C)
    w2 = optimise!(portfolio; obj = MinRisk(), rm = rm)
    r2 = calc_risk(portfolio; rm = rm)
    w3 = optimise!(portfolio; obj = Utility(; l = l), rm = rm)
    r3 = calc_risk(portfolio; rm = rm)
    w4 = optimise!(portfolio; obj = Sharpe(; rf = rf), rm = rm)
    r4 = calc_risk(portfolio; rm = rm)
    w5 = optimise!(portfolio; obj = MaxRet(), rm = rm)
    r5 = calc_risk(portfolio; rm = rm)
    @test r2 <= r1
    @test r3 <= r1 || abs(r3 - r1) < 5e-8
    @test r4 <= r1
    @test r5 <= r1 || abs(r5 - r1) < 5e-8

    portfolio.network_adj = SDP(; A = B)
    portfolio.cluster_adj = SDP(; A = C)
    w6 = optimise!(portfolio; obj = MinRisk(), rm = rm)
    r6 = calc_risk(portfolio; rm = rm)
    w7 = optimise!(portfolio; obj = Utility(; l = l), rm = rm)
    r7 = calc_risk(portfolio; rm = rm)
    w8 = optimise!(portfolio; obj = Sharpe(; rf = rf), rm = rm)
    r8 = calc_risk(portfolio; rm = rm)
    w9 = optimise!(portfolio; obj = MaxRet(), rm = rm)
    r9 = calc_risk(portfolio; rm = rm)
    @test r6 <= r1
    @test r7 <= r1
    @test r8 <= r1
    @test r9 <= r1

    portfolio.network_adj = IP(; A = B)
    portfolio.cluster_adj = SDP(; A = C)
    w2 = optimise!(portfolio; obj = MinRisk(), rm = rm)
    r2 = calc_risk(portfolio; rm = rm)
    w3 = optimise!(portfolio; obj = Utility(; l = l), rm = rm)
    r3 = calc_risk(portfolio; rm = rm)
    w4 = optimise!(portfolio; obj = Sharpe(; rf = rf), rm = rm)
    r4 = calc_risk(portfolio; rm = rm)
    w5 = optimise!(portfolio; obj = MaxRet(), rm = rm)
    r5 = calc_risk(portfolio; rm = rm)
    @test r2 <= r1
    @test r3 <= r1 || abs(r3 - r1) < 5e-8
    @test r4 <= r1
    @test r5 <= r1 || abs(r5 - r1) < 5e-8

    portfolio.network_adj = SDP(; A = B)
    portfolio.cluster_adj = IP(; A = C)
    w2 = optimise!(portfolio; obj = MinRisk(), rm = rm)
    r2 = calc_risk(portfolio; rm = rm)
    w3 = optimise!(portfolio; obj = Utility(; l = l), rm = rm)
    r3 = calc_risk(portfolio; rm = rm)
    w4 = optimise!(portfolio; obj = Sharpe(; rf = rf), rm = rm)
    r4 = calc_risk(portfolio; rm = rm)
    w5 = optimise!(portfolio; obj = MaxRet(), rm = rm)
    r5 = calc_risk(portfolio; rm = rm)
    @test r2 <= r1
    @test r3 <= r1 || abs(r3 - r1) < 5e-8
    @test r4 <= r1
    @test r5 <= r1 || abs(r5 - r1) < 5e-8

    portfolio = Portfolio(; prices = prices,
                          solvers = Dict(:PClGL => Dict(:solver => optimizer_with_attributes(Pajarito.Optimizer,
                                                                                             "verbose" => false,
                                                                                             "oa_solver" => optimizer_with_attributes(HiGHS.Optimizer,
                                                                                                                                      MOI.Silent() => true),
                                                                                             "conic_solver" => optimizer_with_attributes(Clarabel.Optimizer,
                                                                                                                                         "verbose" => false,
                                                                                                                                         "max_step_fraction" => 0.75)))))
    asset_statistics!(portfolio)
    rm = [[SD(), SD()]]
    w10 = optimise!(portfolio; obj = Sharpe(; rf = rf), rm = rm)
    r10 = calc_risk(portfolio; rm = rm[1][1])

    rm[1][1].settings.ub = r10
    portfolio.network_adj = IP(; A = B)
    portfolio.cluster_adj = IP(; A = C)
    w11 = optimise!(portfolio; obj = MinRisk(), rm = rm)
    r11 = calc_risk(portfolio; rm = rm[1][1])
    w12 = optimise!(portfolio; obj = Utility(; l = l), rm = rm)
    r12 = calc_risk(portfolio; rm = rm[1][1])
    w13 = optimise!(portfolio; obj = Sharpe(; rf = rf), rm = rm)
    r13 = calc_risk(portfolio; rm = rm[1][1])
    w14 = optimise!(portfolio; obj = MaxRet(), rm = rm)
    r14 = calc_risk(portfolio; rm = rm[1][1])
    @test r11 <= r10
    @test r12 <= r10 || abs(r12 - r10) < 5e-7
    @test r13 <= r10 || abs(r13 - r10) < 1e-10
    @test r14 <= r10 || abs(r14 - r10) < 5e-7

    portfolio.network_adj = SDP(; A = B)
    portfolio.cluster_adj = SDP(; A = C)
    w15 = optimise!(portfolio; obj = MinRisk(), rm = rm)
    r15 = calc_risk(portfolio; rm = rm[1][1])
    w16 = optimise!(portfolio; obj = Utility(; l = l), rm = rm)
    r16 = calc_risk(portfolio; rm = rm[1][1])
    w17 = optimise!(portfolio; obj = Sharpe(; rf = rf), rm = rm)
    r17 = calc_risk(portfolio; rm = rm[1][1])
    w18 = optimise!(portfolio; obj = MaxRet(), rm = rm)
    r18 = calc_risk(portfolio; rm = rm[1][1])
    @test r15 <= r10
    @test r16 <= r10
    @test r17 <= r10
    @test r18 <= r10

    portfolio.network_adj = SDP(; A = B)
    portfolio.cluster_adj = IP(; A = C)
    w11 = optimise!(portfolio; obj = MinRisk(), rm = rm)
    r11 = calc_risk(portfolio; rm = rm[1][1])
    w12 = optimise!(portfolio; obj = Utility(; l = l), rm = rm)
    r12 = calc_risk(portfolio; rm = rm[1][1])
    w13 = optimise!(portfolio; obj = Sharpe(; rf = rf), rm = rm)
    r13 = calc_risk(portfolio; rm = rm[1][1])
    w14 = optimise!(portfolio; obj = MaxRet(), rm = rm)
    r14 = calc_risk(portfolio; rm = rm[1][1])
    @test r11 <= r10
    @test r12 <= r10 || abs(r12 - r10) < 5e-7
    @test r13 <= r10 || abs(r13 - r10) < 1e-10
    @test r14 <= r10 || abs(r14 - r10) < 5e-7

    portfolio.network_adj = IP(; A = B)
    portfolio.cluster_adj = SDP(; A = C)
    w11 = optimise!(portfolio; obj = MinRisk(), rm = rm)
    r11 = calc_risk(portfolio; rm = rm[1][1])
    w12 = optimise!(portfolio; obj = Utility(; l = l), rm = rm)
    r12 = calc_risk(portfolio; rm = rm[1][1])
    w13 = optimise!(portfolio; obj = Sharpe(; rf = rf), rm = rm)
    r13 = calc_risk(portfolio; rm = rm[1][1])
    w14 = optimise!(portfolio; obj = MaxRet(), rm = rm)
    r14 = calc_risk(portfolio; rm = rm[1][1])
    @test r11 <= r10
    @test r12 <= r10 || abs(r12 - r10) < 5e-7
    @test r13 <= r10 || abs(r13 - r10) < 1e-10
    @test r14 <= r10 || abs(r14 - r10) < 5e-7

    portfolio = portfolio = Portfolio(; prices = prices,
                                      solvers = Dict(:PClGL => Dict(:solver => optimizer_with_attributes(Pajarito.Optimizer,
                                                                                                         "verbose" => false,
                                                                                                         "oa_solver" => optimizer_with_attributes(HiGHS.Optimizer,
                                                                                                                                                  MOI.Silent() => true),
                                                                                                         "conic_solver" => optimizer_with_attributes(Clarabel.Optimizer,
                                                                                                                                                     "verbose" => false,
                                                                                                                                                     "max_step_fraction" => 0.75)))))
    asset_statistics!(portfolio)

    rm = [SD(; settings = RMSettings(; flag = false)), CDaR()]

    portfolio.network_adj = IP(; A = B)
    portfolio.cluster_adj = IP(; A = C)
    w19 = optimise!(portfolio; kelly = AKelly(), obj = Sharpe(; rf = rf), rm = rm)
    w20 = optimise!(portfolio; kelly = EKelly(), obj = Sharpe(; rf = rf), rm = rm)
    @test !isapprox(w19.weights, w20.weights)

    portfolio.network_adj = SDP(; A = B)
    portfolio.cluster_adj = SDP(; A = C)
    w19 = optimise!(portfolio; kelly = AKelly(), obj = Sharpe(; rf = rf), rm = rm)
    w20 = optimise!(portfolio; kelly = EKelly(), obj = Sharpe(; rf = rf), rm = rm)
    @test isapprox(w19.weights, w20.weights)

    portfolio.network_adj = IP(; A = B)
    portfolio.cluster_adj = SDP(; A = C)
    w19 = optimise!(portfolio; kelly = AKelly(), obj = Sharpe(; rf = rf), rm = rm)
    w20 = optimise!(portfolio; kelly = EKelly(), obj = Sharpe(; rf = rf), rm = rm)
    @test isapprox(w19.weights, w20.weights)

    portfolio.network_adj = SDP(; A = B)
    portfolio.cluster_adj = IP(; A = C)
    w19 = optimise!(portfolio; kelly = AKelly(), obj = Sharpe(; rf = rf), rm = rm)
    w20 = optimise!(portfolio; kelly = EKelly(), obj = Sharpe(; rf = rf), rm = rm)
    @test isapprox(w19.weights, w20.weights)

    w21 = optimise!(portfolio; type = RP(), rm = rm)
    portfolio.network_adj = IP(; A = B)
    portfolio.cluster_adj = IP(; A = C)
    w22 = optimise!(portfolio; type = RP(), rm = rm)
    portfolio.network_adj = SDP(; A = B)
    portfolio.cluster_adj = SDP(; A = C)
    w23 = optimise!(portfolio; type = RP(), rm = rm)
    portfolio.network_adj = IP(; A = B)
    portfolio.cluster_adj = SDP(; A = C)
    w24 = optimise!(portfolio; type = RP(), rm = rm)
    portfolio.network_adj = SDP(; A = B)
    portfolio.cluster_adj = IP(; A = C)
    w25 = optimise!(portfolio; type = RP(), rm = rm)
    @test isapprox(w21.weights, w22.weights)
    @test isapprox(w21.weights, w23.weights)
    @test isapprox(w21.weights, w24.weights)
    @test isapprox(w21.weights, w25.weights)
    @test isapprox(w22.weights, w23.weights)
    @test isapprox(w22.weights, w24.weights)
    @test isapprox(w22.weights, w25.weights)
    @test isapprox(w23.weights, w24.weights)
    @test isapprox(w23.weights, w25.weights)
    @test isapprox(w24.weights, w25.weights)
end

@testset "Cluster + Network and Dendrogram non SD" begin
    portfolio = Portfolio(; prices = prices,
                          solvers = Dict(:PClGL => Dict(:solver => optimizer_with_attributes(Pajarito.Optimizer,
                                                                                             "verbose" => false,
                                                                                             "oa_solver" => optimizer_with_attributes(HiGHS.Optimizer,
                                                                                                                                      MOI.Silent() => true),
                                                                                             "conic_solver" => optimizer_with_attributes(Clarabel.Optimizer,
                                                                                                                                         "verbose" => false,
                                                                                                                                         "max_step_fraction" => 0.75)))))
    asset_statistics!(portfolio)

    network_type = TMFG()
    hclust_alg = DBHT()
    A = centrality_vector(portfolio; network_type = network_type)
    B = connection_matrix(portfolio; network_type = network_type)
    C = cluster_matrix(portfolio; hclust_alg = hclust_alg)

    rm = CDaR()
    obj = MinRisk()
    w1 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [0.0, 5.74250749576219e-17, 0.0, 0.0, 0.0034099011531647325, 0.0, 0.0,
          0.07904282391685276, 0.0, 0.0, 0.3875931702023311, 0.0, 0.0,
          2.1342373233586235e-18, 0.0005545152741455163, 0.09598828930573457,
          0.26790888256716056, 0.0, 0.000656017191282443, 0.16484640038932813]
    @test isapprox(w1.weights, wt)

    portfolio.a_vec_cent = A
    portfolio.b_cent = minimum(A)
    w2 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [0.0, 0.0, 0.0, 0.051723138879499364, 0.0, 0.0, 0.05548699473292653,
          0.25666738948843576, 0.0, 0.0, 0.5073286323951223, 0.12879384450401607, 0.0, 0.0,
          0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    @test isapprox(w2.weights, wt)
    @test isapprox(portfolio.b_cent,
                   average_centrality(portfolio; network_type = network_type))

    portfolio.network_adj = IP(; A = B)
    portfolio.cluster_adj = IP(; A = C)
    w3 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [0.0, 0.0, 0.0, 0.13808597886475105, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
          0.861914021135249, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    @test isapprox(w3.weights, wt)
    @test isapprox(portfolio.b_cent,
                   average_centrality(portfolio; network_type = network_type))

    portfolio.network_adj = SDP(; A = B)
    portfolio.cluster_adj = SDP(; A = C)
    w4 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [3.792185015188745e-11, 2.8718827511977774e-12, 1.682546044639531e-11,
          0.055644822127312234, 1.4054810718974508e-9, 3.672871917081098e-11,
          0.05821329934128988, 0.28917703917177284, 3.974532177143144e-11,
          3.379379499842924e-11, 0.4477961123577317, 0.1491687248032702,
          1.751122052954301e-10, 3.827881313356045e-11, 1.4217499113064038e-11,
          2.682309981271483e-10, 4.0157929977533265e-11, 1.2996404576431402e-11,
          4.045036300945231e-11, 3.5810915956455e-11]
    @test isapprox(w4.weights, wt)
    @test isapprox(portfolio.b_cent,
                   average_centrality(portfolio; network_type = network_type))

    portfolio.network_adj = IP(; A = B)
    portfolio.cluster_adj = SDP(; A = C)
    w5 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [3.929029769104475e-11, 4.682013352703612e-11, 5.092732493262497e-11,
          0.1449179066607989, 5.7550438454396217e-11, 1.6570233137821554e-12,
          3.13482303429845e-11, 4.2626964462304147e-11, 5.27712769561164e-13,
          3.8228126544181934e-11, 0.855082092724509, 4.366460022365986e-11,
          2.3679864316262534e-11, 6.710995318107745e-12, 2.917029199773922e-11,
          5.279515377095721e-11, 2.1073589876410156e-11, 4.3165677301681555e-11,
          3.75716865330263e-11, 4.78840296392686e-11]
    @test isapprox(w5.weights, wt)
    @test isapprox(portfolio.b_cent,
                   average_centrality(portfolio; network_type = network_type))

    portfolio.network_adj = SDP(; A = B)
    portfolio.cluster_adj = IP(; A = C)
    w6 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [2.4927190181490636e-11, 4.0578979518480365e-11, 3.9784690723582584e-11,
          3.868170472764502e-11, 4.821161714905576e-11, 8.008509037765195e-12,
          2.4060606651834297e-11, 0.26864562583896623, 7.010576119575405e-12,
          2.4127930283578015e-11, 0.7313543736736025, 4.221706518277982e-11,
          4.107751700256068e-12, 1.4455653900291778e-11, 1.2911452035859952e-11,
          4.422125447685239e-11, 1.5103712284960505e-11, 2.9596235997434e-11,
          3.1284574553505805e-11, 3.8141943577276535e-11]
    @test isapprox(w6.weights, wt)
    @test isapprox(portfolio.b_cent,
                   average_centrality(portfolio; network_type = network_type))

    portfolio.a_vec_cent = []
    portfolio.b_cent = Inf

    portfolio.network_adj = IP(; A = B)
    portfolio.cluster_adj = IP(; A = C)
    w7 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [0.0, 0.0, 0.0, 0.0, 0.37297150326295364, 0.0, 0.0, 0.0, 0.0, 0.0,
          0.6270284967370463, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    @test isapprox(w7.weights, wt)

    portfolio.network_adj = SDP(; A = B)
    portfolio.cluster_adj = SDP(; A = C)
    w8 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [5.582273103776846e-11, 3.0858854706578295e-11, 7.446562879169474e-11,
          2.1940357315002827e-11, 0.044390981566184345, 1.8708160646542956e-11,
          1.0641846725984274e-12, 0.08248112336017129, 3.461307368978628e-11,
          0.022166689687188608, 0.3926071515577202, 0.009380638407106484,
          8.659486533726753e-11, 5.2871866333127325e-11, 9.663218769033281e-12,
          0.1609958578336364, 0.17631431106420714, 2.1144913303259385e-10,
          1.1177012129293661e-10, 0.11166324581396334]
    @test isapprox(w8.weights, wt)

    portfolio.network_adj = IP(; A = B)
    portfolio.cluster_adj = SDP(; A = C)
    w9 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [2.6169977244650817e-11, 2.8351812201915143e-11, 2.8569969548923387e-11,
          2.7394349028459374e-11, 0.387317313049007, 1.799470571553872e-11,
          1.9156221690478035e-11, 3.1669987334775954e-11, 3.139988360612296e-11,
          3.193499760440153e-11, 0.6126826865093625, 2.2375962881375716e-12,
          1.0127637010294365e-11, 2.889002150481588e-11, 2.9742263515167183e-12,
          3.4977174860285824e-11, 3.198488249483316e-11, 2.8398528457030627e-11,
          3.139299653853306e-11, 2.8005535798813038e-11]
    @test isapprox(w9.weights, wt)

    portfolio.network_adj = SDP(; A = B)
    portfolio.cluster_adj = IP(; A = C)
    w10 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [6.069624774965556e-12, 3.5043373343299148e-12, 5.789268142616743e-12,
          5.684915857686101e-13, 6.205836818060747e-12, 2.9975804482698397e-13,
          5.903453593195174e-12, 7.652335711064563e-12, 1.9963522492931897e-12,
          9.469859700004268e-13, 0.4651153889743581, 6.363420731152238e-13,
          5.245783887557819e-12, 3.8900216779580126e-12, 4.4321450635667534e-12,
          0.13318037188082973, 0.40170423907680675, 5.1045766855186574e-12,
          2.9754392877625265e-12, 6.784585044649918e-12]
    @test isapprox(w10.weights, wt)

    portfolio = Portfolio(; prices = prices,
                          solvers = Dict(:PClGL => Dict(:solver => optimizer_with_attributes(Pajarito.Optimizer,
                                                                                             "verbose" => false,
                                                                                             "oa_solver" => optimizer_with_attributes(HiGHS.Optimizer,
                                                                                                                                      MOI.Silent() => true),
                                                                                             "conic_solver" => optimizer_with_attributes(Clarabel.Optimizer,
                                                                                                                                         "verbose" => false,
                                                                                                                                         "max_step_fraction" => 0.75)))))
    asset_statistics!(portfolio)
    obj = Sharpe(; rf = rf)

    network_type = TMFG()
    hclust_alg = DBHT()
    A = centrality_vector(portfolio; network_type = network_type)
    B = connection_matrix(portfolio; network_type = network_type)
    C = cluster_matrix(portfolio; hclust_alg = hclust_alg)

    w11 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [0.0, 0.0, 0.07233520665470376, 0.0, 0.3107248736916702, 0.0, 0.0,
          0.12861270774687708, 0.0, 0.0, 0.16438408898657855, 0.0, 0.0, 0.0, 0.0,
          0.2628826637767333, 0.0, 0.0, 0.0, 0.061060459143437176]
    @test isapprox(w11.weights, wt)

    portfolio.a_vec_cent = A
    portfolio.b_cent = minimum(A)

    w12 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [0.0, 0.0, 0.0, 0.13626926674571754, 1.515977496012877e-16, 0.0,
          0.2529009327082102, 0.0, 0.0, 0.0, 0.4850451479068739, 0.12578465263919827, 0.0,
          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    @test isapprox(w12.weights, wt)
    @test isapprox(portfolio.b_cent,
                   average_centrality(portfolio; network_type = network_type))

    portfolio.network_adj = IP(; A = B)
    portfolio.cluster_adj = IP(; A = C)
    w13 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [0.0, 0.0, 0.0, 0.0, 1.4641796690079654e-17, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    @test isapprox(w13.weights, wt)
    @test isapprox(portfolio.b_cent,
                   average_centrality(portfolio; network_type = network_type))

    portfolio.network_adj = SDP(; A = B)
    portfolio.cluster_adj = SDP(; A = C)
    w14 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [1.0555822112534945e-9, 1.6677992671851285e-10, 6.004374613767746e-11,
          0.10737241991699309, 4.66363324726794e-8, 1.0309364104387801e-9,
          0.40730879313155793, 1.7573481469721425e-8, 1.1145911266106817e-9,
          8.985297977885696e-10, 0.31579172259060667, 0.16952699043101732,
          1.8214691171394823e-11, 1.0767769773867006e-9, 6.160144955445939e-10,
          8.250745941774404e-11, 1.0856076792946e-9, 4.3010967164772414e-10,
          1.1171412115733564e-9, 9.671757140826077e-10]
    @test isapprox(w14.weights, wt)
    @test isapprox(portfolio.b_cent,
                   average_centrality(portfolio; network_type = network_type),
                   rtol = 5.0e-8)

    portfolio.network_adj = IP(; A = B)
    portfolio.cluster_adj = SDP(; A = C)
    w15 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [8.246377966553579e-12, 2.1662261942703532e-11, 2.3149428998649142e-11,
          1.329632066983559e-11, 2.680611928129831e-11, 8.104922374272414e-12,
          0.9999999997517751, 8.337979380555776e-12, 5.900718176167758e-12,
          3.366428532847471e-12, 1.42849940859823e-11, 1.587886257869507e-11,
          7.738975752062779e-12, 8.701085526720765e-12, 1.0088996897558722e-11,
          2.3569252799878376e-11, 2.205428491447811e-12, 1.646598734550664e-11,
          8.739933058412532e-12, 2.1680696844699123e-11]
    @test isapprox(w15.weights, wt)
    @test isapprox(portfolio.b_cent,
                   average_centrality(portfolio; network_type = network_type))

    portfolio.network_adj = SDP(; A = B)
    portfolio.cluster_adj = IP(; A = C)
    w16 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [2.855211948093276e-12, 7.742507590769276e-12, 8.380999836236588e-12,
          4.178721679406631e-12, 9.735206090932517e-12, 3.161333860346923e-12,
          0.3407924052513351, 2.039044524659916e-12, 2.0979812131207165e-12,
          2.6054937533189115e-13, 0.6592075946673086, 5.241370039777376e-12,
          2.955813931602209e-12, 3.277613565981399e-12, 3.807378356131966e-12,
          8.291507128590018e-12, 5.047994358641981e-13, 5.7675316626487636e-12,
          3.086567074607075e-12, 7.9721616287409e-12]
    @test isapprox(w16.weights, wt)

    portfolio.a_vec_cent = []
    portfolio.b_cent = Inf

    portfolio.network_adj = IP(; A = B)
    portfolio.cluster_adj = IP(; A = C)
    w17 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [0.0, 0.0, 0.0, 0.0, 0.7390777009270599, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
          0.0, 0.0, 0.26092229907294007, 0.0, 0.0, 0.0, 0.0]
    @test isapprox(w17.weights, wt)

    portfolio.network_adj = SDP(; A = B)
    portfolio.cluster_adj = SDP(; A = C)
    w18 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [1.1114280032389733e-9, 5.465932432253086e-10, 0.07127361978937344,
          4.2735924027111676e-10, 0.3773893830292412, 8.307478542440443e-12,
          5.829914399286376e-8, 9.309839371077408e-8, 3.738324087545115e-10,
          4.5667669251599526e-10, 0.270188375284781, 4.3615424798530115e-11,
          1.5207758687711383e-10, 7.454797671399692e-10, 1.2696115141860878e-10,
          0.28114846019148243, 1.6002454364725072e-9, 6.525631843804948e-10,
          1.6899787070323335e-9, 2.372465944664811e-9]
    @test isapprox(w18.weights, wt)

    portfolio.network_adj = IP(; A = B)
    portfolio.cluster_adj = SDP(; A = C)
    w19 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [5.232209676225395e-12, 1.8323313827626035e-13, 7.06504320407363e-12,
          1.3995607505212688e-12, 0.6920481196398304, 2.75356252824143e-12,
          1.1137509668436481e-11, 9.021452984645777e-12, 2.828151083804309e-12,
          1.5044051688912587e-12, 9.129965289153754e-12, 3.849409091142576e-12,
          5.020088727615075e-12, 1.5926323703224481e-12, 4.100181820688043e-12,
          0.30795188028544296, 2.9226926071196326e-12, 5.659859191061064e-13,
          2.2083369632134006e-12, 4.2122813951640084e-12]
    @test isapprox(w19.weights, wt)

    portfolio.network_adj = SDP(; A = B)
    portfolio.cluster_adj = IP(; A = C)
    w20 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [9.496514636476357e-13, 2.8751708258013117e-13, 2.14620992398969e-12,
          1.5686768457173072e-13, 0.399034177463875, 1.0013597728003799e-12,
          1.2221529732267048e-12, 2.3218308880471126e-12, 3.6848115002656865e-13,
          2.607930234667157e-13, 0.30061960201631427, 1.1438619102674228e-12,
          1.4673247510707675e-12, 1.34071412066232e-13, 1.3474929726168846e-12,
          0.30034622050259596, 1.176042991202119e-12, 6.470293022392075e-13,
          8.384206546555804e-13, 1.7455173969110805e-12]
    @test isapprox(w20.weights, wt)
end

@testset "Cluster + Network and Dendrogram non SD Short" begin
    portfolio = Portfolio(; prices = prices,
                          solvers = Dict(:PClGL => Dict(:solver => optimizer_with_attributes(Pajarito.Optimizer,
                                                                                             "verbose" => false,
                                                                                             "oa_solver" => optimizer_with_attributes(HiGHS.Optimizer,
                                                                                                                                      MOI.Silent() => true),
                                                                                             "conic_solver" => optimizer_with_attributes(Clarabel.Optimizer,
                                                                                                                                         "verbose" => false,
                                                                                                                                         "max_step_fraction" => 0.75)))))
    asset_statistics!(portfolio)

    A = centrality_vector(portfolio)
    B = connection_matrix(portfolio)
    C = cluster_matrix(portfolio)

    portfolio.short = true
    portfolio.short_budget = 0.13
    portfolio.short_u = 0.13
    portfolio.long_u = 1
    portfolio.budget = portfolio.long_u - portfolio.short_u

    rm = CDaR()
    obj = MinRisk()
    w1 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [0.0, -0.0033927423088478525, 0.0, -0.0017047316992726633, 0.02620966174092891,
          -0.048951276707458066, -0.02831053066644215, 0.0797084661157905,
          -0.006091550656354475, -0.01772949018065388, 0.34963750962788587,
          -0.009810028561542944, 0.006616206609090757, -1.7025233012195975e-16,
          -0.01400964921942798, 0.09738313723086256, 0.16435744872368938,
          0.01643568935570537, 0.10356751838407428, 0.15608436221197253]
    @test isapprox(w1.weights, wt)

    portfolio.a_vec_cent = A
    portfolio.b_cent = minimum(A)
    w2 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [-0.09486167717442834, 0.04019416692914485, 0.037577223179041455,
          -5.0655785389293176e-17, 0.12801741187716315, -7.979727989493313e-17,
          -0.01998095976345068, 0.13355629824656456, 0.019271038397774838, 0.0,
          0.28797176886025655, 0.0, 0.008845283123909234, 0.0, -0.015157363062120899,
          0.10734222426525322, 0.0, 0.061474266941201526, 4.119968255444917e-17,
          0.17575031817969053]
    @test isapprox(w2.weights, wt)
    @test isapprox(portfolio.b_cent, average_centrality(portfolio))

    portfolio.network_adj = IP(; A = B)
    portfolio.cluster_adj = IP(; A = C)
    w3 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [0.0, 0.0, 0.0, 0.0, 0.334584688856143, 0.0, -2.376571162088226e-16,
          0.4054153111438569, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.13000000000000034, 0.0,
          0.0, -5.551115123125783e-17, 0.0]
    @test isapprox(w3.weights, wt)
    @test isapprox(portfolio.b_cent, average_centrality(portfolio))

    portfolio.network_adj = SDP(; A = B)
    portfolio.cluster_adj = SDP(; A = C)
    w4 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [-6.693267202841269e-12, 0.040432422336115055, 0.023578638541304663,
          5.7717379693667485e-11, 0.1463980040465671, 2.6100459375455476e-11,
          -8.171664935319021e-13, 0.21316996235268962, 1.7556757990572343e-11,
          3.868808986004283e-11, 0.13235733407279987, 0.057815001165258756,
          0.0029786153690263707, 1.0547814233890217e-11, 1.972659470805425e-8,
          0.1075790749125807, -6.173977620653348e-12, 0.01518763048117749,
          -0.018322734829876286, 0.14882603168883576]
    @test isapprox(w4.weights, wt)
    @test isapprox(portfolio.b_cent, average_centrality(portfolio))

    portfolio.network_adj = IP(; A = B)
    portfolio.cluster_adj = SDP(; A = C)
    w5 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [1.5206895831159078e-12, 5.849077752014655e-12, 4.953373341546518e-12,
          1.9114935820043825e-12, 0.1511900919941239, 7.112737528223264e-13,
          -3.5043538914858695e-12, 0.23861103836384287, 1.2816497977538522e-13,
          3.6994064368453596e-13, 9.375507882623025e-12, 7.699910572432064e-12,
          7.788130139700947e-12, 2.791231532587461e-12, 0.003761054484780868,
          0.1299999999386743, 5.6730843280878846e-12, 0.09171651526487172,
          3.1626721351127575e-12, 0.25472129990527614]
    @test isapprox(w5.weights, wt)
    @test isapprox(portfolio.b_cent, average_centrality(portfolio))

    portfolio.network_adj = SDP(; A = B)
    portfolio.cluster_adj = IP(; A = C)
    w6 = optimise!(portfolio; obj = obj, rm = rm)
    if !isempty(w6)
        wt = [-3.436744679402206e-11, 3.239661533633809e-11, 2.4840447922545817e-11,
              -9.972810205467658e-12, 0.3351029495116913, 7.839734082797505e-12,
              -3.5168375152895714e-11, 0.4048970498835798, -2.1168226953999462e-11,
              7.405357389041877e-12, 2.7811407918484925e-11, 3.871855933414875e-11,
              3.031055896740596e-11, -1.7227555861709482e-11, 3.5161380685179906e-11,
              0.13000000053320468, -3.482577397178112e-11, 2.7638894619405443e-11,
              -4.384228255547239e-11, 3.597490549371302e-11]
        @test isapprox(w6.weights, wt)
        @test isapprox(portfolio.b_cent, average_centrality(portfolio))
    end

    portfolio.a_vec_cent = []
    portfolio.b_cent = Inf

    portfolio.network_adj = IP(; A = B)
    portfolio.cluster_adj = IP(; A = C)
    w7 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 6.938893903907228e-18, 0.0, 0.0, 0.0,
          0.42517704255085137, -0.0, 0.0, 0.0, -0.0, 0.05890972756969503,
          0.3859132298794536, 0.0, 0.0, -0.0]
    @test isapprox(w7.weights, wt)

    portfolio.network_adj = SDP(; A = B)
    portfolio.cluster_adj = SDP(; A = C)
    w8 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [8.005099809395322e-11, 5.54741874344066e-11, 3.7161008165908413e-10,
          3.399830567579541e-11, 0.024515641359833074, 2.3077913541693312e-11,
          -8.557321355409364e-11, 0.0884095404422948, 3.0571544254790295e-11,
          1.3442669404688171e-11, 0.31877875453680166, 0.013279538620306513,
          1.6185295915456725e-10, 4.9030713005730264e-11, 1.3508583854644475e-10,
          0.12732969672456004, 0.196986324697881, 9.532758916856292e-11,
          8.305964627126601e-11, 0.10070050257131365]
    @test isapprox(w8.weights, wt)

    portfolio.network_adj = IP(; A = B)
    portfolio.cluster_adj = SDP(; A = C)
    w9 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [8.926756900155627e-12, 8.095386355809985e-12, 1.0856706932843284e-11,
          1.011084318300073e-11, 0.1226277184829006, 2.497233097267528e-12,
          -3.214161084729858e-12, 1.2897612868436275e-11, 9.428324905742709e-12,
          9.003416136030217e-12, 0.3678587947424985, -9.312778743405145e-12,
          -7.686747028134787e-12, 9.122868863091762e-12, -7.765072733362852e-12,
          0.15489065361947696, 1.177966233551351e-11, 0.07584182985544369,
          1.0823632945824571e-11, 0.14878100322411653]
    @test isapprox(w9.weights, wt)

    portfolio.network_adj = SDP(; A = B)
    portfolio.cluster_adj = IP(; A = C)
    w10 = optimise!(portfolio; obj = obj, rm = rm)
    if !isempty(w10)
        wt = [5.714715683777237e-12, 3.5136317105336944e-12, 5.8369943427409435e-12,
              -1.19148080834951e-12, 6.001951208464746e-12, -2.114951066063337e-13,
              -6.265586756394273e-12, 7.283842094661988e-12, 1.830378102116846e-12,
              6.837378299528731e-13, 0.40465038840179224, 7.866958277926574e-13,
              -5.225243165180584e-12, 4.436829788196232e-12, -4.227652520920422e-12,
              0.11586692349160559, 0.34948268807236954, 5.458758514235868e-12,
              3.231966844365002e-12, 6.574567542422419e-12]
        @test isapprox(w10.weights, wt)
    end

    portfolio = Portfolio(; prices = prices,
                          solvers = Dict(:PClGL => Dict(:solver => optimizer_with_attributes(Pajarito.Optimizer,
                                                                                             "verbose" => false,
                                                                                             "oa_solver" => optimizer_with_attributes(HiGHS.Optimizer,
                                                                                                                                      MOI.Silent() => true),
                                                                                             "conic_solver" => optimizer_with_attributes(Clarabel.Optimizer,
                                                                                                                                         "verbose" => false,
                                                                                                                                         "max_step_fraction" => 0.75)))))
    asset_statistics!(portfolio)
    obj = Sharpe(; rf = rf)

    network_type = TMFG()
    hclust_alg = DBHT()
    A = centrality_vector(portfolio; network_type = network_type)
    B = connection_matrix(portfolio; network_type = network_type)
    C = cluster_matrix(portfolio; hclust_alg = hclust_alg)

    portfolio.short = true
    portfolio.short_budget = 0.18
    portfolio.short_u = 0.18
    portfolio.long_u = 0.95
    portfolio.budget = portfolio.long_u - portfolio.short_u

    w11 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [0.0, 0.0, 0.06455652837124166, 0.0, 0.17806945118716286, -0.1534166903019555,
          6.554560797119506e-17, 0.12351856122756986, 0.0, 0.0, 0.21867000759778857, 0.0,
          -0.023295149141940166, 0.0, -0.0032881605561044004, 0.1770017931069312,
          0.006143447950208067, 0.0, 0.04071674553919559, 0.1413234650199022]
    @test isapprox(w11.weights, wt)

    portfolio.a_vec_cent = A
    portfolio.b_cent = minimum(A)

    w12 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [0.0, 0.0, 0.07328869996428454, 0.0, 0.19593630860627667, -0.1516035031548817,
          -2.228175066036319e-17, 0.1290693464408622, 0.0, 0.0, 0.20897763141227396, 0.0,
          -0.02192543103531701, 0.0, -0.006471065809801198, 0.18010601289300754,
          7.366923752795404e-17, 0.0, 0.025951685743898865, 0.13667031493939605]
    @test isapprox(w12.weights, wt)
    @test isapprox(portfolio.b_cent,
                   average_centrality(portfolio; network_type = network_type))

    portfolio.network_adj = IP(; A = B)
    portfolio.cluster_adj = IP(; A = C)
    w13 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [-0.0, -0.0, 0.0, -0.0, 0.6900000000000023, -0.0, -0.0, 0.0, -0.0, -0.0,
          0.19949253798767527, -0.0, -0.11949253798767445, -0.0, -0.0, 0.0, 0.0, -0.0, 0.0,
          0.0]
    @test isapprox(w13.weights, wt)
    @test isapprox(portfolio.b_cent,
                   average_centrality(portfolio; network_type = network_type))

    portfolio.network_adj = SDP(; A = B)
    portfolio.cluster_adj = SDP(; A = C)
    w14 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [4.20344835013978e-10, 1.4168776430128298e-9, 0.09696500469093917,
          3.1176128226326995e-10, 0.2922997525244137, -0.026459097591948585,
          -4.417632975372617e-10, 0.028779635749055912, 9.546290432471772e-11,
          -2.495670414711786e-12, 0.19271665010436986, -0.01954598124555238,
          -0.06439906902193586, 6.034104243703727e-10, -0.06959581904337238,
          0.3351515315112699, 8.394462035273617e-9, 4.681439604848753e-10,
          0.004087347335763035, 3.3720793479888645e-8]
    @test isapprox(w14.weights, wt)
    @test isapprox(portfolio.b_cent,
                   average_centrality(portfolio; network_type = network_type))

    portfolio.network_adj = IP(; A = B)
    portfolio.cluster_adj = SDP(; A = C)
    w15 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [-0.0, -0.0, 0.0, -0.0, 0.6899999999999927, -0.0, -0.0, 0.0, -0.0, -0.0,
          0.1994925379876753, -0.0, -0.11949253798766485, -0.0, -0.0, 0.0, -0.0, -0.0, 0.0,
          0.0]
    @test isapprox(w15.weights, wt)
    @test isapprox(portfolio.b_cent,
                   average_centrality(portfolio; network_type = network_type))

    # portfolio.network_adj = SDP(; A = B)
    # portfolio.cluster_adj = IP(; A = C)
    # w16 = optimise!(portfolio; obj = obj, rm = CDaR())
    # wt = [3.1345471980832356e-10, 2.469910273202323e-10, 7.946447462097655e-10,
    #       1.7210857874043897e-10, 0.354561355808636, -3.665814844245118e-10,
    #       8.522943700016685e-10, 2.9715927145570816e-10, 8.376446627301666e-10,
    #       1.3758357758118232e-10, 0.16641976055535695, -2.8469632582700114e-10,
    #       -0.014268236579253484, 4.2616226395158317e-10, -0.01533295006816714,
    #       0.272938209489049, 4.78233132976098e-10, 1.5817077710699714e-10, 0.005681856375922364,
    #       3.5528715646533134e-10]
    # @test isapprox(w16.weights, wt)
    # @test isapprox(portfolio.b_cent, average_centrality(portfolio; network_type = network_type))

    portfolio.a_vec_cent = []
    portfolio.b_cent = Inf

    portfolio.network_adj = IP(; A = B)
    portfolio.cluster_adj = IP(; A = C)
    w17 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [-0.0, -0.0, 0.0, -0.0, 0.6751155811095629, -0.0, -0.0, 0.0, -0.0, -0.0,
          0.208898224965609, -0.0, -0.11401380607517049, -0.0, -0.0, 0.0, 0.0, -0.0, -0.0,
          0.0]
    @test isapprox(w17.weights, wt)

    portfolio.network_adj = SDP(; A = B)
    portfolio.cluster_adj = SDP(; A = C)
    w18 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [2.550290644841355e-10, 9.771388050317999e-10, 0.05961068291021875,
          1.6026865333539806e-10, 0.2904304663922716, -0.05362659027424702,
          -1.63659061741679e-10, 0.07230890224668701, 5.1369564031811e-11,
          2.0424468150220464e-11, 0.19375985463838347, -6.891317947200625e-8,
          -0.059887761717970445, 3.446707017015399e-10, -0.06648555112878939,
          0.33389004862092864, 1.94796711892746e-9, 2.9491701612043247e-10,
          3.675177048642511e-9, 9.662393532595498e-9]
    @test isapprox(w18.weights, wt)

    portfolio.network_adj = IP(; A = B)
    portfolio.cluster_adj = SDP(; A = C)
    w19 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [3.0982423795801523e-13, 1.0897190081325992e-12, 1.1439057096976537e-12,
          8.335502746643442e-13, 0.6169500695767203, -3.861790690926653e-13,
          1.8380185480850114e-16, 1.20684882427598e-12, 5.868291924756247e-13,
          4.4628734778063407e-13, 0.2509320307207596, -2.37722960041735e-13,
          -0.0978821003077978, 3.4547568626542086e-13, -6.747069769061212e-13,
          1.654492336692321e-12, 1.2431657638900952e-12, 6.722716586546247e-13,
          1.0218382105346152e-12, 1.0622255787841832e-12]
    @test isapprox(w19.weights, wt)

    portfolio.network_adj = SDP(; A = B)
    portfolio.cluster_adj = IP(; A = C)
    w20 = optimise!(portfolio; obj = obj, rm = rm)
    wt = [1.2305818195374707e-12, 6.638995180407921e-13, 1.4818099637685315e-12,
          6.446873700377288e-13, 0.5385298458083386, 9.688920546141059e-14,
          1.859881899987937e-12, 1.7499757900946622e-12, -1.6444783912761784e-13,
          3.17879317664001e-13, 1.6480292361775621e-12, -0.17999999988660298,
          -8.95627519573147e-13, 9.75176970539232e-13, -3.014371991930602e-13,
          0.41147015406512866, 9.498790292467675e-13, 8.291346273853208e-13,
          7.802681766331766e-13, 1.2690237115010107e-12]
    @test isapprox(w20.weights, wt)
end

@testset "Rebalance Trad" begin
    portfolio = Portfolio(; prices = prices,
                          solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                           :check_sol => (allow_local = true,
                                                                          allow_almost = true),
                                                           :params => Dict("verbose" => false,
                                                                           "max_step_fraction" => 0.7))))
    asset_statistics!(portfolio)

    w1 = optimise!(portfolio; obj = MinRisk())
    r1 = calc_risk(portfolio)
    ret1 = dot(portfolio.mu, w1.weights)
    sr1 = sharpe_ratio(portfolio)

    w2 = optimise!(portfolio; obj = Utility(; l = l))
    r2 = calc_risk(portfolio)
    ret2 = dot(portfolio.mu, w2.weights)
    sr2 = sharpe_ratio(portfolio)

    w3 = optimise!(portfolio; obj = Sharpe(; rf = rf))
    r3 = calc_risk(portfolio)
    ret3 = dot(portfolio.mu, w3.weights)
    sr3 = sharpe_ratio(portfolio)

    w4 = optimise!(portfolio; obj = MaxRet())
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
    w5 = optimise!(portfolio; obj = MinRisk())
    @test isapprox(w1.weights, w5.weights)
    portfolio.rebalance.w = w1.weights
    w6 = optimise!(portfolio; obj = Utility(; l = l))
    w7 = optimise!(portfolio; obj = Sharpe(; rf = rf))
    w8 = optimise!(portfolio; obj = MaxRet())
    @test isapprox(w2.weights, w6.weights)
    @test isapprox(w3.weights, w7.weights)
    @test isapprox(w4.weights, w8.weights)

    portfolio.rebalance = TR(; val = 1e10, w = w3.weights)
    w9 = optimise!(portfolio; obj = MinRisk())
    @test isapprox(w9.weights, w3.weights)
    portfolio.rebalance.w = w1.weights
    w10 = optimise!(portfolio; obj = Utility(; l = l))
    w11 = optimise!(portfolio; obj = Sharpe(; rf = rf))
    w12 = optimise!(portfolio; obj = MaxRet())
    @test isapprox(w10.weights, w1.weights)
    @test isapprox(w11.weights, w1.weights)
    @test isapprox(w12.weights, w1.weights)

    portfolio.rebalance = TR(; val = 1e-4, w = w3.weights)
    w13 = optimise!(portfolio; obj = MinRisk())
    @test !isapprox(w13.weights, w1.weights)
    @test !isapprox(w13.weights, w3.weights)
    portfolio.rebalance.w = w1.weights
    w14 = optimise!(portfolio; obj = Utility(; l = l))
    w15 = optimise!(portfolio; obj = Sharpe(; rf = rf))
    w16 = optimise!(portfolio; obj = MaxRet())
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
    w13 = optimise!(portfolio; obj = MinRisk())
    @test !isapprox(w13.weights, w3.weights)
    @test !isapprox(w13.weights, w1.weights)
    portfolio.rebalance.w = w1.weights
    w14 = optimise!(portfolio; obj = Utility(; l = l))
    w15 = optimise!(portfolio; obj = Sharpe(; rf = rf))
    w16 = optimise!(portfolio; obj = MaxRet())
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
    portfolio = Portfolio(; prices = prices,
                          solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                           :check_sol => (allow_local = true,
                                                                          allow_almost = true),
                                                           :params => Dict("verbose" => false,
                                                                           "max_step_fraction" => 0.7))))
    asset_statistics!(portfolio)
    wc_statistics!(portfolio)

    w1 = optimise!(portfolio; type = WC(), obj = MinRisk())
    r1 = calc_risk(portfolio; type = :WC)
    ret1 = dot(portfolio.mu, w1.weights)
    sr1 = sharpe_ratio(portfolio; type = :WC)

    w2 = optimise!(portfolio; type = WC(), obj = Utility(; l = l))
    r2 = calc_risk(portfolio; type = :WC)
    ret2 = dot(portfolio.mu, w2.weights)
    sr2 = sharpe_ratio(portfolio; type = :WC)

    w3 = optimise!(portfolio; type = WC(), obj = Sharpe(; rf = rf))
    r3 = calc_risk(portfolio; type = :WC)
    ret3 = dot(portfolio.mu, w3.weights)
    sr3 = sharpe_ratio(portfolio; type = :WC)

    w4 = optimise!(portfolio; type = WC(), obj = MaxRet())
    r4 = calc_risk(portfolio; type = :WC)
    ret4 = dot(portfolio.mu, w4.weights)
    sr4 = sharpe_ratio(portfolio; type = :WC)

    @test r1 < r2 < r3 < r4
    @test ret1 < ret2 < ret3 < ret4
    @test sr1 < sr4 < sr3 < sr2

    sr5 = sharpe_ratio(portfolio; type = :WC, kelly = true)
    @test isapprox(dot(portfolio.mu, w4.weights) / calc_risk(portfolio; type = :WC), sr4)
    @test isapprox(1 / size(portfolio.returns, 1) *
                   sum(log.(1 .+ portfolio.returns * w4.weights)) /
                   calc_risk(portfolio; type = :WC), sr5)

    portfolio.rebalance = TR(; val = 0, w = w3.weights)
    w5 = optimise!(portfolio; type = WC(), obj = MinRisk())
    @test isapprox(w1.weights, w5.weights)
    portfolio.rebalance.w = w1.weights
    w6 = optimise!(portfolio; type = WC(), obj = Utility(; l = l))
    w7 = optimise!(portfolio; type = WC(), obj = Sharpe(; rf = rf))
    w8 = optimise!(portfolio; type = WC(), obj = MaxRet())
    @test isapprox(w2.weights, w6.weights)
    @test isapprox(w3.weights, w7.weights)
    @test isapprox(w4.weights, w8.weights)

    portfolio.rebalance = TR(; val = 1e10, w = w3.weights)
    w9 = optimise!(portfolio; type = WC(), obj = MinRisk())
    @test isapprox(w9.weights, w3.weights)
    portfolio.rebalance.w = w1.weights
    w10 = optimise!(portfolio; type = WC(), obj = Utility(; l = l))
    w11 = optimise!(portfolio; type = WC(), obj = Sharpe(; rf = rf))
    w12 = optimise!(portfolio; type = WC(), obj = MaxRet())
    @test isapprox(w10.weights, w1.weights, rtol = 1.0e-7)
    @test isapprox(w11.weights, w1.weights, rtol = 5.0e-8)
    @test isapprox(w12.weights, w1.weights, rtol = 5.0e-6)

    portfolio.rebalance = TR(; val = 1e-4, w = w3.weights)
    w13 = optimise!(portfolio; type = WC(), obj = MinRisk())
    @test !isapprox(w13.weights, w1.weights)
    @test !isapprox(w13.weights, w3.weights)
    portfolio.rebalance.w = w1.weights
    w14 = optimise!(portfolio; type = WC(), obj = Utility(; l = l))
    w15 = optimise!(portfolio; type = WC(), obj = Sharpe(; rf = rf))
    w16 = optimise!(portfolio; type = WC(), obj = MaxRet())
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
    w13 = optimise!(portfolio; type = WC(), obj = MinRisk())
    @test !isapprox(w13.weights, w3.weights)
    @test !isapprox(w13.weights, w1.weights)
    portfolio.rebalance.w = w1.weights
    w14 = optimise!(portfolio; type = WC(), obj = Utility(; l = l))
    w15 = optimise!(portfolio; type = WC(), obj = Sharpe(; rf = rf))
    w16 = optimise!(portfolio; type = WC(), obj = MaxRet())
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
    portfolio = Portfolio(; prices = prices,
                          solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                           :check_sol => (allow_local = true,
                                                                          allow_almost = true),
                                                           :params => Dict("verbose" => false,
                                                                           "max_step_fraction" => 0.75))))
    asset_statistics!(portfolio)

    w1 = optimise!(portfolio; obj = Sharpe(; rf = rf))
    to1 = 0.05
    tow1 = copy(w1.weights)
    portfolio.turnover = TR(; val = to1, w = tow1)
    w2 = optimise!(portfolio; obj = MinRisk())
    @test all(abs.(w2.weights - tow1) .<= to1)

    portfolio.turnover = NoTR()
    w3 = optimise!(portfolio; obj = MinRisk())
    to2 = 0.031
    tow2 = copy(w3.weights)
    portfolio.turnover = TR(; val = to2, w = tow2)
    w4 = optimise!(portfolio; obj = Sharpe(; rf = rf))
    @test all(abs.(w4.weights - tow2) .<= to2)

    portfolio.turnover = NoTR()
    w5 = optimise!(portfolio; obj = Sharpe(; rf = rf))
    to3 = range(; start = 0.001, stop = 0.003, length = 20)
    tow3 = copy(w5.weights)
    portfolio.turnover = TR(; val = to3, w = tow3)
    w6 = optimise!(portfolio; obj = MinRisk())
    @test all(abs.(w6.weights - tow3) .<= to3)

    portfolio.turnover = NoTR()
    w7 = optimise!(portfolio; obj = MinRisk())
    to4 = 0.031
    tow4 = copy(w7.weights)
    portfolio.turnover = TR(; val = to4, w = tow4)
    w8 = optimise!(portfolio; obj = Sharpe(; rf = rf))
    @test all(abs.(w8.weights - tow4) .<= to2)

    @test_throws AssertionError portfolio.turnover = TR(; val = 1:19)
    @test_throws AssertionError portfolio.turnover = TR(; val = 1:21)
    @test_throws AssertionError portfolio.turnover = TR(; w = 1:19)
    @test_throws AssertionError portfolio.turnover = TR(; w = 1:21)
end

@testset "Tracking" begin
    portfolio = Portfolio(; prices = prices,
                          solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                           :check_sol => (allow_local = true,
                                                                          allow_almost = true),
                                                           :params => Dict("verbose" => false,
                                                                           "max_step_fraction" => 0.75))))
    asset_statistics!(portfolio)
    T = size(portfolio.returns, 1)

    w1 = optimise!(portfolio; obj = Sharpe(; rf = rf))
    te1 = 0.0005
    tw1 = copy(w1.weights)
    portfolio.tracking_err = TrackWeight(; err = te1, w = tw1)
    w2 = optimise!(portfolio; obj = MinRisk())
    @test norm(portfolio.returns * (w2.weights - tw1), 2) / sqrt(T - 1) <= te1

    w3 = optimise!(portfolio; obj = MinRisk())
    te2 = 0.0003
    tw2 = copy(w3.weights)
    portfolio.tracking_err = TrackWeight(; err = te2, w = tw2)
    w4 = optimise!(portfolio; obj = Sharpe(; rf = rf))
    @test norm(portfolio.returns * (w4.weights - tw2), 2) / sqrt(T - 1) <= te2
    @test_throws AssertionError portfolio.tracking_err = TrackWeight(; err = te2, w = 1:19)
    @test_throws AssertionError portfolio.tracking_err = TrackWeight(; err = te2, w = 1:21)

    portfolio.tracking_err = NoTracking()
    w5 = optimise!(portfolio; obj = Sharpe(; rf = rf))
    te3 = 0.007
    tw3 = vec(mean(portfolio.returns; dims = 2))
    portfolio.tracking_err = TrackRet(; err = te3, w = tw3)
    w6 = optimise!(portfolio; obj = MinRisk())
    @test norm(portfolio.returns * w6.weights - tw3, 2) / sqrt(T - 1) <= te3

    portfolio.tracking_err = NoTracking()
    w7 = optimise!(portfolio; obj = MinRisk())
    te4 = 0.0024
    tw4 = vec(mean(portfolio.returns; dims = 2))
    portfolio.tracking_err = TrackRet(; err = te4, w = tw4)
    w8 = optimise!(portfolio; obj = Sharpe(; rf = rf))
    @test norm(portfolio.returns * w8.weights - tw4, 2) / sqrt(T - 1) <= te4

    @test_throws AssertionError portfolio.tracking_err = TrackRet(; err = te2,
                                                                  w = 1:(T - 1))
    @test_throws AssertionError portfolio.tracking_err = TrackRet(; err = te2,
                                                                  w = 1:(T + 1))

    portfolio.tracking_err = TrackRet(; err = te2, w = 1:T)
end

@testset "Min and max number of effective assets" begin
    portfolio = Portfolio(; prices = prices,
                          solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                           :check_sol => (allow_local = true,
                                                                          allow_almost = true),
                                                           :params => Dict("verbose" => false,
                                                                           "max_step_fraction" => 0.75))))
    asset_statistics!(portfolio)

    w1 = optimise!(portfolio; obj = MinRisk())
    portfolio.num_assets_l = 12
    w2 = optimise!(portfolio; obj = MinRisk())
    @test count(w2.weights .>= 2e-2) >= 12
    @test count(w2.weights .>= 2e-2) > count(w1.weights .>= 2e-2)
    @test !isapprox(w1.weights, w2.weights)
    @test isapprox(portfolio.num_assets_l, floor(Int, number_effective_assets(portfolio)))

    portfolio.num_assets_l = 0
    w3 = optimise!(portfolio; obj = Sharpe(; rf = rf))
    portfolio.num_assets_l = 8
    w4 = optimise!(portfolio; obj = Sharpe(; rf = rf))
    @test count(w4.weights .>= 2e-2) >= 8
    @test count(w4.weights .>= 2e-2) > count(w3.weights .>= 2e-2)
    @test !isapprox(w3.weights, w4.weights)
    @test isapprox(portfolio.num_assets_l, floor(Int, number_effective_assets(portfolio)))

    portfolio.num_assets_l = 0
    portfolio.short = true
    portfolio.short_u = 0.2
    portfolio.long_u = 0.8
    portfolio.budget = portfolio.long_u - portfolio.short_u

    w5 = optimise!(portfolio; obj = MinRisk())
    @test isapprox(sum(w5.weights), portfolio.budget)
    @test sum(w5.weights[w5.weights .< 0]) <= portfolio.short_budget
    @test sum(w5.weights[w5.weights .>= 0]) <= portfolio.budget + portfolio.short_budget

    portfolio.num_assets_l = 17
    w6 = optimise!(portfolio; obj = MinRisk())
    @test isapprox(sum(w6.weights), portfolio.budget)
    @test sum(w6.weights[w6.weights .< 0]) <= portfolio.short_budget
    @test sum(w6.weights[w6.weights .>= 0]) <= portfolio.budget + portfolio.short_budget
    @test count(abs.(w6.weights) .>= 4e-3) >= 17
    @test count(abs.(w6.weights) .>= 4e-3) > count(abs.(w5.weights) .>= 4e-3)
    @test !isapprox(w5.weights, w6.weights)
    @test isapprox(portfolio.num_assets_l, floor(Int, number_effective_assets(portfolio)))

    portfolio.num_assets_l = 0
    w7 = optimise!(portfolio; obj = Sharpe(; rf = rf))
    @test isapprox(sum(w7.weights), portfolio.budget)
    portfolio.num_assets_l = 13
    w8 = optimise!(portfolio; obj = Sharpe(; rf = rf))
    @test isapprox(sum(w8.weights), portfolio.budget)
    @test count(abs.(w8.weights) .>= 4e-3) >= 13
    @test count(abs.(w8.weights) .>= 4e-3) > count(abs.(w7.weights) .>= 4e-3)
    @test !isapprox(w7.weights, w8.weights)
    @test isapprox(portfolio.num_assets_l, floor(Int, number_effective_assets(portfolio)))

    portfolio = Portfolio(; prices = prices,
                          solvers = Dict(:PClGL => Dict(:solver => optimizer_with_attributes(Pajarito.Optimizer,
                                                                                             "verbose" => false,
                                                                                             "oa_solver" => optimizer_with_attributes(HiGHS.Optimizer,
                                                                                                                                      MOI.Silent() => true),
                                                                                             "conic_solver" => optimizer_with_attributes(Clarabel.Optimizer,
                                                                                                                                         "verbose" => false,
                                                                                                                                         "max_step_fraction" => 0.75)))))
    asset_statistics!(portfolio)

    w9 = optimise!(portfolio; obj = MinRisk())
    portfolio.num_assets_u = 5
    w10 = optimise!(portfolio; obj = MinRisk())
    @test count(w10.weights .>= 2e-2) <= 5
    @test count(w10.weights .>= 2e-2) < count(w9.weights .>= 2e-2)
    @test !isapprox(w9.weights, w10.weights)

    portfolio.num_assets_u = 0
    w11 = optimise!(portfolio; obj = Sharpe(; rf = rf))
    portfolio.num_assets_u = 3
    w12 = optimise!(portfolio; obj = Sharpe(; rf = rf))
    @test count(w12.weights .>= 2e-2) <= 3
    @test count(w12.weights .>= 2e-2) < count(w11.weights .>= 2e-2)
    @test !isapprox(w11.weights, w12.weights)

    portfolio.num_assets_u = 0
    portfolio.short = true
    portfolio.short_u = 0.2
    portfolio.long_u = 0.8
    portfolio.budget = portfolio.long_u - portfolio.short_u

    w13 = optimise!(portfolio; obj = MinRisk())
    @test isapprox(sum(w13.weights), portfolio.budget)
    @test sum(w13.weights[w13.weights .< 0]) <= portfolio.short_budget
    @test sum(w13.weights[w13.weights .>= 0]) <= portfolio.budget + portfolio.short_budget
    portfolio.num_assets_u = 7
    w14 = optimise!(portfolio; obj = MinRisk())
    @test isapprox(sum(w14.weights), portfolio.budget)
    @test sum(w14.weights[w14.weights .< 0]) <= portfolio.short_budget
    @test sum(w14.weights[w14.weights .>= 0]) <= portfolio.budget + portfolio.short_budget
    @test count(abs.(w14.weights) .>= 2e-2) <= 7
    @test count(abs.(w14.weights) .>= 2e-2) < count(abs.(w13.weights) .>= 2e-2)
    @test !isapprox(w13.weights, w14.weights)

    portfolio.num_assets_u = 0
    w15 = optimise!(portfolio; obj = Sharpe(; rf = rf))
    @test isapprox(sum(w15.weights), portfolio.budget)
    @test sum(w15.weights[w15.weights .< 0]) <= portfolio.short_budget
    @test sum(w15.weights[w15.weights .>= 0]) <= portfolio.budget + portfolio.short_budget
    portfolio.num_assets_u = 4
    w16 = optimise!(portfolio; obj = Sharpe(; rf = rf))
    @test isapprox(sum(w16.weights), portfolio.budget)
    @test sum(w16.weights[w16.weights .< 0]) <= portfolio.short_budget
    @test abs(sum(w16.weights[w16.weights .>= 0]) <=
              portfolio.budget + portfolio.short_budget)
    @test count(abs.(w16.weights) .>= 2e-2) >= 4
    @test count(abs.(w16.weights) .>= 2e-2) < count(abs.(w15.weights) .>= 2e-2)
    @test !isapprox(w15.weights, w16.weights)

    @test_throws AssertionError portfolio.num_assets_l = -1
    @test_throws AssertionError portfolio.num_assets_u = -1
end

@testset "Linear" begin
    portfolio = Portfolio(; prices = prices,
                          solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                           :check_sol => (allow_local = true,
                                                                          allow_almost = true),
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

    w1 = optimise!(portfolio; obj = MinRisk())
    @test all(w1.weights[asset_sets.G2DBHT .== 2] .>= 0.03)
    @test all(w1.weights[asset_sets.G2DBHT .== 3] .<= 0.2)
    @test all(w1.weights[w1.tickers .== "AAPL"] .>= 0.032)
    @test sum(w1.weights[asset_sets.G2ward .== 2]) <=
          w1.weights[w1.tickers .== "MA"][1] * 2.2
    @test w1.weights[w1.tickers .== "MA"][1] >= sum(w1.weights[asset_sets.G2ward .== 3]) * 5

    w2 = optimise!(portfolio; obj = Sharpe(; rf = rf))
    @test all(w2.weights[asset_sets.G2DBHT .== 2] .>= 0.03)
    @test all(w2.weights[asset_sets.G2DBHT .== 3] .<= 0.2)
    @test all(w2.weights[w2.tickers .== "AAPL"] .>= 0.032)
    @test sum(w2.weights[asset_sets.G2ward .== 2]) <=
          w2.weights[w2.tickers .== "MA"][1] * 2.2
    @test w2.weights[w2.tickers .== "MA"][1] >= sum(w2.weights[asset_sets.G2ward .== 3]) * 5

    @test_throws AssertionError portfolio.a_mtx_ineq = rand(13, 19)
    @test_throws AssertionError portfolio.a_mtx_ineq = rand(13, 21)
end
