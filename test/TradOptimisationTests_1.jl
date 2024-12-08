@testset "Fail optimisation" begin
    portfolio = Portfolio(; prices = prices[(end - 200):end],
                          solvers = Dict(:HiGHS => Dict(:solver => HiGHS.Optimizer,
                                                        :params => Dict("log_to_console" => false))))
    asset_statistics!(portfolio)
    optimise!(portfolio, Trad())

    @test !isempty(portfolio.fail)
    @test haskey(portfolio.fail, :HiGHS_Trad)
    @test haskey(portfolio.fail[:HiGHS_Trad], :JuMP_error)
    @test length(keys(portfolio.fail[:HiGHS_Trad])) == 1

    portfolio.solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                               :check_sol => (allow_local = true,
                                                              allow_almost = true),
                                               :params => Dict("verbose" => false,
                                                               "max_step_fraction" => 0.75,
                                                               "max_iter" => 100,
                                                               "equilibrate_max_iter" => 20)))
    rm = [[OWA(; owa = OWASettings(; approx = false)),
           OWA(; owa = OWASettings(; approx = true))]]
    optimise!(portfolio, Trad(; rm = rm))
    @test !isempty(portfolio.fail)
    @test haskey(portfolio.fail, :Clarabel_Trad)
    @test length(keys(portfolio.fail[:Clarabel_Trad])) == 6

    portfolio.solvers = Dict(:Clarabel1 => Dict(:solver => Clarabel.Optimizer,
                                                :check_sol => (allow_local = true,
                                                               allow_almost = true),
                                                :params => Dict("verbose" => false,
                                                                "max_iter" => 1)),
                             :Clarabel2 => Dict(:solver => Clarabel.Optimizer,
                                                :check_sol => (allow_local = true,
                                                               allow_almost = true),
                                                :params => Dict("verbose" => false)))
    optimise!(portfolio, Trad(; rm = SD()))
    @test !isempty(portfolio.fail)
    @test haskey(portfolio.fail, :Clarabel1_Trad)
    @test length(keys(portfolio.fail[:Clarabel1_Trad])) == 6
end

@testset "Variance" begin
    portfolio = Portfolio(; prices = prices,
                          solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                           :check_sol => (allow_local = true,
                                                                          allow_almost = true),
                                                           :params => Dict("verbose" => false))))
    asset_statistics!(portfolio)
    rm = Variance()

    obj = MinRisk()
    w1 = optimise!(portfolio, Trad(; rm = rm, kelly = NoKelly(), obj = obj))
    r1 = calc_risk(portfolio; type = :Trad, rm = rm)
    ret1 = dot(portfolio.mu, w1.weights)
    wt = [0.0078093616583851146, 0.030698578999046943, 0.010528316771324561,
          0.027486806578381814, 0.012309038313357737, 0.03341430871186881,
          1.1079055085166888e-7, 0.13985416268183082, 2.0809271230580642e-7,
          6.32554715643476e-6, 0.28784123553791136, 1.268075347971748e-7,
          8.867081236591187e-8, 0.12527141708492384, 3.264070667606171e-7,
          0.015079837844627948, 1.5891383112101438e-5, 0.1931406057562605,
          2.654124109083291e-7, 0.11654298695072397]
    riskt = 5.936075960042635e-5
    rett = 0.0003482663810696356
    @test isapprox(w1.weights, wt)
    @test isapprox(r1, riskt)
    @test isapprox(ret1, rett)

    w2 = optimise!(portfolio, Trad(; rm = rm, kelly = AKelly(), obj = obj))
    wt = [0.0078093616583851146, 0.030698578999046943, 0.010528316771324561,
          0.027486806578381814, 0.012309038313357737, 0.03341430871186881,
          1.1079055085166888e-7, 0.13985416268183082, 2.0809271230580642e-7,
          6.32554715643476e-6, 0.28784123553791136, 1.268075347971748e-7,
          8.867081236591187e-8, 0.12527141708492384, 3.264070667606171e-7,
          0.015079837844627948, 1.5891383112101438e-5, 0.1931406057562605,
          2.654124109083291e-7, 0.11654298695072397]
    @test isapprox(w2.weights, wt)

    w3 = optimise!(portfolio, Trad(; rm = rm, kelly = EKelly(), obj = obj))
    wt = [0.007893834004010548, 0.030693397218184384, 0.01050943911162566,
          0.027487590678529683, 0.0122836015907984, 0.03341312720689581,
          2.654460293680794e-8, 0.13984817931920596, 4.861252776141605e-8,
          3.309876672531783e-7, 0.2878217133212084, 3.116270780466401e-8,
          2.2390519612760115e-8, 0.12528318523437287, 9.334010636386356e-8,
          0.015085761770714442, 7.170234777802365e-7, 0.19312554465008394,
          6.179100704970626e-8, 0.11655329404175341]
    @test isapprox(w3.weights, wt)

    obj = Utility(; l = l)
    w4 = optimise!(portfolio, Trad(; rm = rm, kelly = NoKelly(), obj = obj))
    r2 = calc_risk(portfolio; type = :Trad, rm = rm)
    ret2 = dot(portfolio.mu, w4.weights)
    wt = [1.065553054256496e-9, 1.906979877393637e-9, 2.1679869440360567e-9,
          1.70123972526289e-9, 0.7741855142171694, 3.9721744242294547e-10,
          0.10998135534654405, 1.3730517031876334e-9, 1.5832262577152926e-9,
          1.0504881447825781e-9, 1.2669287896045939e-9, 4.038975120701348e-10,
          6.074001448526581e-10, 2.654358762537183e-10, 6.574536682273354e-10,
          0.1158331072870088, 3.0452991740231055e-9, 1.3663094482455795e-9,
          2.4334674474942e-9, 1.8573424305703526e-9]
    riskt = 0.00025903630381171117
    rett = 0.0017268228943243054
    @test isapprox(w4.weights, wt)
    @test isapprox(r2, riskt)
    @test isapprox(ret2, rett)

    w5 = optimise!(portfolio, Trad(; rm = rm, kelly = AKelly(), obj = obj))
    wt = [8.870505348946403e-9, 2.0785636066552727e-8, 2.798137197308657e-8,
          1.971091339190289e-8, 0.7185881405281792, 4.806880649144299e-10,
          0.09860828964210354, 1.1235098321720224e-8, 2.977172777854582e-8,
          8.912749778026878e-9, 9.63062128166912e-9, 1.0360544993920464e-9,
          2.180352541614548e-9, 2.689800139816139e-9, 2.3063944199708073e-9,
          0.15518499560246005, 0.027618271886178034, 1.246121371211767e-8,
          1.2842725621709964e-7, 1.586069567397408e-8]
    @test isapprox(w5.weights, wt)

    w6 = optimise!(portfolio, Trad(; rm = rm, kelly = EKelly(), obj = obj))
    wt = [4.076718644400776e-9, 7.97059286965588e-9, 1.0440947901329591e-8,
          7.702064328487365e-9, 0.7207180459688085, 1.1223657131481055e-9,
          0.09835526321809454, 4.730656228269007e-9, 1.133279001422807e-8,
          4.082347527215102e-9, 4.208721035741791e-9, 9.44650173900743e-10,
          5.699105612892133e-10, 2.079988023687919e-9, 5.463657962369361e-10,
          0.15505269860336068, 0.02587387715494727, 5.1544684301828716e-9,
          4.384014669868401e-8, 6.25205503896986e-9]
    @test isapprox(w6.weights, wt, rtol = 5.0e-7)

    obj = Sharpe(; rf = rf)
    w7 = optimise!(portfolio, Trad(; rm = rm, kelly = NoKelly(), obj = obj))
    r3 = calc_risk(portfolio; type = :Trad, rm = rm)
    ret3 = dot(portfolio.mu, w7.weights)
    wt = [3.435681413150756e-10, 9.269743242817558e-10, 1.174114891347931e-9,
          7.447407606568295e-10, 0.5180552669298059, 1.08151797431462e-10,
          0.0636508036260703, 7.872264113421062e-10, 7.841830201959634e-10,
          3.9005509625957585e-10, 6.479557895235057e-10, 8.472023236127232e-11,
          5.766670106753152e-11, 1.988136246095318e-10, 5.935811276550078e-11,
          0.14326634942881586, 0.1964867973307653, 7.554937254824565e-10,
          0.0785407748474901, 7.740298948228655e-10]
    riskt = 0.00017320867441528544
    rett = 0.0014788430765515807
    @test isapprox(w7.weights, wt)
    @test isapprox(r3, riskt)
    @test isapprox(ret3, rett)

    w8 = optimise!(portfolio, Trad(; rm = rm, kelly = AKelly(), obj = obj))
    wt = [3.953983055390625e-7, 1.2130189854003184e-6, 1.5626692963148418e-6,
          9.099292074973729e-7, 0.4886931254919885, 1.1962647602545863e-7,
          0.05842798708631173, 1.2992007103695783e-6, 7.700590186769794e-7,
          4.5424814443255545e-7, 9.701300232241762e-7, 9.241777891765365e-8,
          6.70735556712963e-8, 2.2704051659598955e-7, 6.523754123014794e-8,
          0.14023347164242564, 0.21376781701297304, 1.095611158897337e-6,
          0.09886732333044786, 1.0337751344638193e-6]
    @test isapprox(w8.weights, wt)

    w9 = optimise!(portfolio, Trad(; rm = rm, kelly = EKelly(), obj = obj))
    wt = [1.7647799778761598e-7, 7.43791350184855e-7, 9.974860927108156e-7,
          4.726318325437073e-7, 0.48902202810838025, 5.25817986396541e-8,
          0.05826321657926799, 8.395640963247789e-7, 1.7634626972511063e-7,
          2.0620668133652036e-7, 5.77393877243638e-7, 4.130907666540682e-8,
          3.032865903519244e-8, 9.975084610937565e-8, 2.927622589710415e-8,
          0.140132597154028, 0.2135052983305069, 6.899386614850263e-7, 0.09907112697494824,
          5.99769403061889e-7]
    @test isapprox(w9.weights, wt, rtol = 0.001)

    obj = MaxRet()
    w10 = optimise!(portfolio, Trad(; rm = rm, kelly = NoKelly(), obj = obj))
    r4 = calc_risk(portfolio; type = :Trad, rm = rm)
    ret4 = dot(portfolio.mu, w10.weights)
    wt = [8.719239065561217e-10, 8.89979881250899e-10, 9.962391574545242e-10,
          9.646258079031587e-10, 8.401034275877383e-9, 4.828906105645332e-10,
          0.9999999782316086, 6.87727997367879e-10, 9.021263551270326e-10,
          7.350693996493505e-10, 6.753002461228969e-10, 5.009649350579108e-10,
          3.7428368039997965e-10, 5.884337547691459e-10, 3.9326986484718143e-10,
          8.842556785821523e-10, 9.784139669171374e-10, 7.146277206720297e-10,
          9.044289592659792e-10, 8.2279511460579e-10]
    riskt = 0.0016481855568877633
    rett = 0.0018453756308089402
    @test isapprox(w10.weights, wt)
    @test isapprox(r4, riskt)
    @test isapprox(ret4, rett)

    w11 = optimise!(portfolio, Trad(; rm = rm, kelly = AKelly(), obj = obj))
    wt = [3.997304959875111e-10, 6.096555731047457e-10, 1.0828360159599854e-9,
          8.390684135793784e-10, 0.8503431998881756, 5.837264433583887e-10,
          0.14965678808511193, 8.520932897976487e-13, 7.66160958682953e-10,
          1.0247860261071675e-10, 5.1627700971086255e-11, 5.483183958203547e-10,
          7.565204185674542e-10, 3.16106264753721e-10, 7.638502459889708e-10,
          2.447496129413098e-9, 1.372927322256315e-9, 6.541563185875491e-11,
          9.248420166125226e-10, 3.9509971643490626e-10]
    @test isapprox(w11.weights, wt)

    w12 = optimise!(portfolio, Trad(; rm = rm, kelly = EKelly(), obj = obj))
    wt = [2.633743649985236e-9, 2.9505379229172413e-9, 3.665208013478754e-9,
          3.305907804738003e-9, 0.8533949389480296, 1.1597137196070284e-9,
          0.14660501531709985, 2.0097483499520764e-9, 3.1965650000368884e-9,
          2.1746159802304897e-9, 1.9315555659189856e-9, 1.2031777907298662e-9,
          8.570011599717695e-10, 1.5424724997016623e-9, 8.7105116377961e-10,
          5.975487549944421e-9, 4.106592066239195e-9, 2.115816975845725e-9,
          3.4223392213476532e-9, 2.6133360746565418e-9]
    @test isapprox(w12.weights, wt, rtol = 1.0e-7)

    # Risk upper bound
    obj = MaxRet()
    rm.settings.ub = r1
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio; type = :Trad, rm = rm) <= r1 ||
          abs(calc_risk(portfolio; type = :Trad, rm = rm) - r1) < 1e-9

    rm.settings.ub = r2
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test abs(calc_risk(portfolio; type = :Trad, rm = rm) - r2) <= 5e-10

    rm.settings.ub = r3
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test abs(calc_risk(portfolio; type = :Trad, rm = rm) - r3) <= 1e-10

    rm.settings.ub = r4
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio; type = :Trad, rm = rm) <= r4

    obj = Sharpe(; rf = rf)
    rm.settings.ub = r1
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio; type = :Trad, rm = rm) <= r1 ||
          abs(calc_risk(portfolio; type = :Trad, rm = rm) - r1) < 1e-10

    rm.settings.ub = r2
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio; type = :Trad, rm = rm) <= r2

    rm.settings.ub = r3
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio; type = :Trad, rm = rm) <= r3

    rm.settings.ub = r4
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio; type = :Trad, rm = rm) <= r4

    # Ret lower bound
    rm.settings.ub = Inf
    obj = MinRisk()
    portfolio.mu_l = ret1
    w13 = optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test dot(portfolio.mu, w13.weights) >= ret1

    portfolio.mu_l = ret2
    w14 = optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test dot(portfolio.mu, w14.weights) >= ret2

    portfolio.mu_l = ret3
    w15 = optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test dot(portfolio.mu, w15.weights) >= ret3

    portfolio.mu_l = ret4
    w16 = optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test dot(portfolio.mu, w16.weights) >= ret4

    obj = Sharpe(; rf = rf)
    portfolio.mu_l = ret1
    w17 = optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test dot(portfolio.mu, w17.weights) >= ret1

    portfolio.mu_l = ret2
    w18 = optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test dot(portfolio.mu, w18.weights) >= ret2

    portfolio.mu_l = ret3
    w19 = optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test dot(portfolio.mu, w19.weights) >= ret3

    portfolio.mu_l = ret4
    w20 = optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test dot(portfolio.mu, w20.weights) >= ret4
end

@testset "Variance formulations" begin
    portfolio = Portfolio(; prices = prices,
                          solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                           :check_sol => (allow_local = true,
                                                                          allow_almost = true),
                                                           :params => Dict("verbose" => false))))
    asset_statistics!(portfolio)

    obj = MinRisk()
    rm = Variance(; formulation = SOC())
    w1 = optimise!(portfolio, Trad(; rm = rm, kelly = NoKelly(), obj = obj))
    r1 = calc_risk(portfolio; type = :Trad, rm = rm)
    ret1 = dot(portfolio.mu, w1.weights)
    wt = [0.0078093616583851146, 0.030698578999046943, 0.010528316771324561,
          0.027486806578381814, 0.012309038313357737, 0.03341430871186881,
          1.1079055085166888e-7, 0.13985416268183082, 2.0809271230580642e-7,
          6.32554715643476e-6, 0.28784123553791136, 1.268075347971748e-7,
          8.867081236591187e-8, 0.12527141708492384, 3.264070667606171e-7,
          0.015079837844627948, 1.5891383112101438e-5, 0.1931406057562605,
          2.654124109083291e-7, 0.11654298695072397]
    riskt = 5.936075960042635e-5
    rett = 0.0003482663810696356
    @test isapprox(w1.weights, wt)
    @test isapprox(r1, riskt)
    @test isapprox(ret1, rett)

    rm = Variance(; formulation = Quad())
    w2 = optimise!(portfolio, Trad(; rm = rm, kelly = NoKelly(), obj = obj))
    r2 = calc_risk(portfolio; type = :Trad, rm = rm)
    ret2 = dot(portfolio.mu, w2.weights)
    wt = [0.007931704468101724, 0.030580315769072094, 0.010456593899158567,
          0.02742093947332831, 0.012239100149899967, 0.033302579036721264,
          4.162859647692109e-6, 0.13981730989327237, 5.788675176975045e-6,
          0.00033179125802216067, 0.2877731391009407, 3.3160181115934523e-6,
          2.9562651082726095e-6, 0.12512918292552744, 3.227218484382772e-5,
          0.015034147440717766, 0.0005108903689970619, 0.19299166064346063,
          6.739278073914423e-6, 0.1164254102918178]
    riskt = 5.936214450112821e-5
    rett = 0.0003484772957329131
    @test isapprox(w2.weights, wt)
    @test isapprox(r2, riskt)
    @test isapprox(ret2, rett)

    @test isapprox(w1.weights, w2.weights, rtol = 0.005)
    @test isapprox(r1, r2, rtol = 5.0e-5)
    @test isapprox(ret1, ret2, rtol = 0.001)

    obj = Utility(; l = l)
    rm = Variance(; formulation = SOC())
    w3 = optimise!(portfolio, Trad(; rm = rm, kelly = NoKelly(), obj = obj))
    r3 = calc_risk(portfolio; type = :Trad, rm = rm)
    ret3 = dot(portfolio.mu, w3.weights)
    wt = [1.065553054256496e-9, 1.906979877393637e-9, 2.1679869440360567e-9,
          1.70123972526289e-9, 0.7741855142171694, 3.9721744242294547e-10,
          0.10998135534654405, 1.3730517031876334e-9, 1.5832262577152926e-9,
          1.0504881447825781e-9, 1.2669287896045939e-9, 4.038975120701348e-10,
          6.074001448526581e-10, 2.654358762537183e-10, 6.574536682273354e-10,
          0.1158331072870088, 3.0452991740231055e-9, 1.3663094482455795e-9,
          2.4334674474942e-9, 1.8573424305703526e-9]
    riskt = 0.00025903630381171117
    rett = 0.0017268228943243054
    @test isapprox(w3.weights, wt)
    @test isapprox(r3, riskt)
    @test isapprox(ret3, rett)

    rm = Variance(; formulation = Quad())
    w4 = optimise!(portfolio, Trad(; rm = rm, kelly = NoKelly(), obj = obj))
    r4 = calc_risk(portfolio; type = :Trad, rm = rm)
    ret4 = dot(portfolio.mu, w4.weights)
    wt = [6.852081519398486e-11, 1.9688305552023357e-10, 2.779044358366435e-10,
          2.0509456674893866e-10, 0.7741909329958984, 2.003910403124917e-11,
          0.1099865880247063, 9.121516622151838e-11, 3.6704849024948924e-10,
          8.806774623589372e-11, 7.781246849842617e-11, 1.813903586007653e-11,
          1.2669422405822381e-11, 3.710822071187361e-11, 1.2738399524886383e-11,
          0.11582247405793253, 2.2605095246705475e-9, 1.0496320870713167e-10,
          9.506590849948595e-10, 1.320900856147283e-10]
    riskt = 0.0002590398491928102
    rett = 0.0017268299955487294
    @test isapprox(w4.weights, wt)
    @test isapprox(r4, riskt)
    @test isapprox(ret4, rett)

    @test isapprox(w3.weights, w4.weights, rtol = 5e-5)
    @test isapprox(r3, r4, rtol = 5.0e-5)
    @test isapprox(ret3, ret4, rtol = 5.0e-6)

    obj = Sharpe(; rf = rf)
    rm = Variance(; formulation = SOC())
    w5 = optimise!(portfolio, Trad(; rm = rm, kelly = NoKelly(), obj = obj))
    r5 = calc_risk(portfolio; type = :Trad, rm = rm)
    ret5 = dot(portfolio.mu, w5.weights)
    wt = [3.435681413150756e-10, 9.269743242817558e-10, 1.174114891347931e-9,
          7.447407606568295e-10, 0.5180552669298059, 1.08151797431462e-10,
          0.0636508036260703, 7.872264113421062e-10, 7.841830201959634e-10,
          3.9005509625957585e-10, 6.479557895235057e-10, 8.472023236127232e-11,
          5.766670106753152e-11, 1.988136246095318e-10, 5.935811276550078e-11,
          0.14326634942881586, 0.1964867973307653, 7.554937254824565e-10,
          0.0785407748474901, 7.740298948228655e-10]
    riskt = 0.00017320867441528544
    rett = 0.0014788430765515807
    @test isapprox(w5.weights, wt)
    @test isapprox(r5, riskt)
    @test isapprox(ret5, rett)

    rm = Variance(; formulation = Quad())
    w6 = optimise!(portfolio, Trad(; rm = rm, kelly = NoKelly(), obj = obj))
    r6 = calc_risk(portfolio; type = :Trad, rm = rm)
    ret6 = dot(portfolio.mu, w6.weights)
    wt = [2.2723291920423494e-10, 8.274438575006142e-10, 1.525985513481683e-9,
          7.273695858012778e-10, 0.5180580411000183, 6.940884975732612e-11,
          0.06365095030024893, 5.333993060298542e-10, 3.9178175629371144e-10,
          2.433346403837124e-10, 3.8877926150779944e-10, 5.601703894979824e-11,
          4.159446993235074e-11, 1.275650437326318e-10, 3.926119276017696e-11,
          0.1432677858447418, 0.19649634671149677, 4.6010816369377544e-10,
          0.07852686985456356, 5.296488926346786e-10]
    riskt = 0.00017320984043392098
    rett = 0.0014788476224523953
    @test isapprox(w6.weights, wt)
    @test isapprox(r6, riskt)
    @test isapprox(ret6, rett)

    @test isapprox(w5.weights, w6.weights, rtol = 5.0e-5)
    @test isapprox(r5, r6, rtol = 1.0e-5)
    @test isapprox(ret5, ret6, rtol = 5.0e-6)

    obj = MaxRet()
    rm = Variance(; formulation = SOC())
    w7 = optimise!(portfolio, Trad(; rm = rm, kelly = NoKelly(), obj = obj))
    r7 = calc_risk(portfolio; type = :Trad, rm = rm)
    ret7 = dot(portfolio.mu, w7.weights)
    wt = [8.719239065561217e-10, 8.89979881250899e-10, 9.962391574545242e-10,
          9.646258079031587e-10, 8.401034275877383e-9, 4.828906105645332e-10,
          0.9999999782316086, 6.87727997367879e-10, 9.021263551270326e-10,
          7.350693996493505e-10, 6.753002461228969e-10, 5.009649350579108e-10,
          3.7428368039997965e-10, 5.884337547691459e-10, 3.9326986484718143e-10,
          8.842556785821523e-10, 9.784139669171374e-10, 7.146277206720297e-10,
          9.044289592659792e-10, 8.2279511460579e-10]
    riskt = 0.0016481855568877633
    rett = 0.0018453756308089402
    @test isapprox(w7.weights, wt)
    @test isapprox(r7, riskt)
    @test isapprox(ret7, rett)

    rm = Variance(; formulation = Quad())
    w8 = optimise!(portfolio, Trad(; rm = rm, kelly = NoKelly(), obj = obj))
    r8 = calc_risk(portfolio; type = :Trad, rm = rm)
    ret8 = dot(portfolio.mu, w8.weights)
    wt = [8.719239065561217e-10, 8.89979881250899e-10, 9.962391574545242e-10,
          9.646258079031587e-10, 8.401034275877383e-9, 4.828906105645332e-10,
          0.9999999782316086, 6.87727997367879e-10, 9.021263551270326e-10,
          7.350693996493505e-10, 6.753002461228969e-10, 5.009649350579108e-10,
          3.7428368039997965e-10, 5.884337547691459e-10, 3.9326986484718143e-10,
          8.842556785821523e-10, 9.784139669171374e-10, 7.146277206720297e-10,
          9.044289592659792e-10, 8.2279511460579e-10]
    riskt = 0.0016481855568877633
    rett = 0.0018453756308089402
    @test isapprox(w8.weights, wt)
    @test isapprox(r8, riskt)
    @test isapprox(ret8, rett)

    @test isapprox(w7.weights, w8.weights)
    @test isapprox(r7, r8)
    @test isapprox(ret7, ret8)

    obj = MinRisk()
    rm = Variance(; formulation = SOC())
    w9 = optimise!(portfolio,
                   Trad(; rm = rm, kelly = AKelly(; formulation = SOC()), obj = obj))
    r9 = calc_risk(portfolio; type = :Trad, rm = rm)
    ret9 = dot(portfolio.mu, w9.weights)
    wt = [0.0078093616583851146, 0.030698578999046943, 0.010528316771324561,
          0.027486806578381814, 0.012309038313357737, 0.03341430871186881,
          1.1079055085166888e-7, 0.13985416268183082, 2.0809271230580642e-7,
          6.32554715643476e-6, 0.28784123553791136, 1.268075347971748e-7,
          8.867081236591187e-8, 0.12527141708492384, 3.264070667606171e-7,
          0.015079837844627948, 1.5891383112101438e-5, 0.1931406057562605,
          2.654124109083291e-7, 0.11654298695072397]
    riskt = 5.936075960042635e-5
    rett = 0.0003482663810696356
    @test isapprox(w9.weights, wt)
    @test isapprox(r9, riskt)
    @test isapprox(ret9, rett)

    rm = Variance(; formulation = Quad())
    w10 = optimise!(portfolio,
                    Trad(; rm = rm, kelly = AKelly(; formulation = SOC()), obj = obj))
    r10 = calc_risk(portfolio; type = :Trad, rm = rm)
    ret10 = dot(portfolio.mu, w10.weights)
    wt = [0.007931704468101724, 0.030580315769072094, 0.010456593899158567,
          0.02742093947332831, 0.012239100149899967, 0.033302579036721264,
          4.162859647692109e-6, 0.13981730989327237, 5.788675176975045e-6,
          0.00033179125802216067, 0.2877731391009407, 3.3160181115934523e-6,
          2.9562651082726095e-6, 0.12512918292552744, 3.227218484382772e-5,
          0.015034147440717766, 0.0005108903689970619, 0.19299166064346063,
          6.739278073914423e-6, 0.1164254102918178]
    riskt = 5.936214450112821e-5
    rett = 0.0003484772957329131
    @test isapprox(w10.weights, wt)
    @test isapprox(r10, riskt)
    @test isapprox(ret10, rett)

    @test isapprox(w9.weights, w10.weights, rtol = 0.005)
    @test isapprox(r9, r10, rtol = 5.0e-5)
    @test isapprox(ret9, ret10, rtol = 0.001)

    obj = Utility(; l = l)
    rm = Variance(; formulation = SOC())
    w11 = optimise!(portfolio,
                    Trad(; rm = rm, kelly = AKelly(; formulation = SOC()), obj = obj))
    r11 = calc_risk(portfolio; type = :Trad, rm = rm)
    ret11 = dot(portfolio.mu, w11.weights)
    wt = [8.870505348946403e-9, 2.0785636066552727e-8, 2.798137197308657e-8,
          1.971091339190289e-8, 0.7185881405281792, 4.806880649144299e-10,
          0.09860828964210354, 1.1235098321720224e-8, 2.977172777854582e-8,
          8.912749778026878e-9, 9.63062128166912e-9, 1.0360544993920464e-9,
          2.180352541614548e-9, 2.689800139816139e-9, 2.3063944199708073e-9,
          0.15518499560246005, 0.027618271886178034, 1.246121371211767e-8,
          1.2842725621709964e-7, 1.586069567397408e-8]
    riskt = 0.00023831960644703196
    rett = 0.0016792722833452185
    @test isapprox(w11.weights, wt)
    @test isapprox(r11, riskt)
    @test isapprox(ret11, rett)

    rm = Variance(; formulation = Quad())
    w12 = optimise!(portfolio,
                    Trad(; rm = rm, kelly = AKelly(; formulation = SOC()), obj = obj))
    r12 = calc_risk(portfolio; type = :Trad, rm = rm)
    ret12 = dot(portfolio.mu, w12.weights)
    wt = [1.6868171199009505e-9, 6.518606011133161e-9, 9.86352279770984e-9,
          6.150045875747458e-9, 0.718630983702095, 3.7119199852091685e-10,
          0.09861482354309821, 2.790532675752989e-9, 1.2728774591077898e-8,
          2.1871465657675766e-9, 2.2693974383129075e-9, 2.958986484663317e-10,
          2.0351357726703993e-10, 7.649820299793997e-10, 1.9650537328495345e-10,
          0.15517370021004367, 0.027580337279790704, 3.3046232245984773e-9,
          1.0179775163080922e-7, 4.1356629132822375e-9]
    riskt = 0.00023833463446320128
    rett = 0.0016793098997042355
    @test isapprox(w12.weights, wt)
    @test isapprox(r12, riskt)
    @test isapprox(ret12, rett)

    @test isapprox(w11.weights, w12.weights, rtol = 0.0001)
    @test isapprox(r11, r12, rtol = 0.0001)
    @test isapprox(ret11, ret12, rtol = 5.0e-5)

    obj = Sharpe(; rf = rf)
    rm = Variance(; formulation = SOC())
    w13 = optimise!(portfolio,
                    Trad(; rm = rm, kelly = AKelly(; formulation = SOC()), obj = obj))
    r13 = calc_risk(portfolio; type = :Trad, rm = rm)
    ret13 = dot(portfolio.mu, w13.weights)
    wt = [3.953983055390625e-7, 1.2130189854003184e-6, 1.5626692963148418e-6,
          9.099292074973729e-7, 0.4886931254919885, 1.1962647602545863e-7,
          0.05842798708631173, 1.2992007103695783e-6, 7.700590186769794e-7,
          4.5424814443255545e-7, 9.701300232241762e-7, 9.241777891765365e-8,
          6.70735556712963e-8, 2.2704051659598955e-7, 6.523754123014794e-8,
          0.14023347164242564, 0.21376781701297304, 1.095611158897337e-6,
          0.09886732333044786, 1.0337751344638193e-6]
    riskt = 0.00016559358507428642
    rett = 0.0014479293174607377
    @test isapprox(w13.weights, wt)
    @test isapprox(r13, riskt)
    @test isapprox(ret13, rett)

    rm = Variance(; formulation = Quad())
    w14 = optimise!(portfolio,
                    Trad(; rm = rm, kelly = AKelly(; formulation = SOC()), obj = obj))
    r14 = calc_risk(portfolio; type = :Trad, rm = rm)
    ret14 = dot(portfolio.mu, w14.weights)
    wt = [1.669718396236806e-8, 6.027044058934097e-8, 8.739577024175596e-8,
          4.301880600905147e-8, 0.48873017518452627, 5.254152037070482e-9,
          0.05843208512186604, 6.917260938265724e-8, 2.8809473152698165e-8,
          1.8499861374074213e-8, 4.4781660705625404e-8, 4.139805510566116e-9,
          2.9926847605727475e-9, 9.45059691996542e-9, 2.9559088456557273e-9,
          0.14024309499313364, 0.21373264542041767, 5.2284896629923713e-8,
          0.09886150576581468, 4.779039174782547e-8]
    riskt = 0.00016560238556231427
    rett = 0.0014479684757221218
    @test isapprox(w14.weights, wt, rtol = 5.0e-7)
    @test isapprox(r14, riskt, rtol = 1.0e-7)
    @test isapprox(ret14, rett, rtol = 5.0e-8)

    @test isapprox(w13.weights, w14.weights, rtol = 0.0001)
    @test isapprox(r13, r14, rtol = 0.0001)
    @test isapprox(ret13, ret14, rtol = 5.0e-5)

    obj = MaxRet()
    rm = Variance(; formulation = SOC())
    w15 = optimise!(portfolio,
                    Trad(; rm = rm, kelly = AKelly(; formulation = SOC()), obj = obj))
    r15 = calc_risk(portfolio; type = :Trad, rm = rm)
    ret15 = dot(portfolio.mu, w15.weights)
    wt = [3.997304959875111e-10, 6.096555731047457e-10, 1.0828360159599854e-9,
          8.390684135793784e-10, 0.8503431998881756, 5.837264433583887e-10,
          0.14965678808511193, 8.520932897976487e-13, 7.66160958682953e-10,
          1.0247860261071675e-10, 5.1627700971086255e-11, 5.483183958203547e-10,
          7.565204185674542e-10, 3.16106264753721e-10, 7.638502459889708e-10,
          2.447496129413098e-9, 1.372927322256315e-9, 6.541563185875491e-11,
          9.248420166125226e-10, 3.9509971643490626e-10]
    riskt = 0.000307068809623536
    rett = 0.001803059901755384
    @test isapprox(w15.weights, wt)
    @test isapprox(r15, riskt)
    @test isapprox(ret15, rett)

    rm = Variance(; formulation = Quad())
    w16 = optimise!(portfolio,
                    Trad(; rm = rm, kelly = AKelly(; formulation = SOC()), obj = obj))
    r16 = calc_risk(portfolio; type = :Trad, rm = rm)
    ret16 = dot(portfolio.mu, w16.weights)
    wt = [3.1491090891677036e-10, 3.9252510945045457e-10, 5.011436739931957e-10,
          4.814228838168751e-10, 0.8503245220331475, 4.987452504286351e-11,
          0.14967547181371002, 1.8257330907415293e-10, 4.5000351561986515e-10,
          2.195999249689875e-10, 1.8725349256894292e-10, 7.858729597392012e-11,
          2.333547428526958e-10, 1.351809534814174e-10, 1.8382016146366064e-10,
          1.1430290850714406e-9, 6.064115688191264e-10, 2.1745240525978677e-10,
          4.80092693730735e-10, 2.959062049654606e-10]
    riskt = 0.0003070706709250669
    rett = 0.0018030608400966524
    @test isapprox(w16.weights, wt)
    @test isapprox(r16, riskt)
    @test isapprox(ret16, rett)

    @test isapprox(w15.weights, w16.weights, rtol = 5.0e-5)
    @test isapprox(r15, r16, rtol = 1.0e-5)
    @test isapprox(ret15, ret16, rtol = 1.0e-6)

    obj = MinRisk()
    rm = Variance(; formulation = SOC())
    w17 = optimise!(portfolio,
                    Trad(; rm = rm, kelly = AKelly(; formulation = Quad()), obj = obj))
    r17 = calc_risk(portfolio; type = :Trad, rm = rm)
    ret17 = dot(portfolio.mu, w17.weights)
    wt = [0.0078093616583851146, 0.030698578999046943, 0.010528316771324561,
          0.027486806578381814, 0.012309038313357737, 0.03341430871186881,
          1.1079055085166888e-7, 0.13985416268183082, 2.0809271230580642e-7,
          6.32554715643476e-6, 0.28784123553791136, 1.268075347971748e-7,
          8.867081236591187e-8, 0.12527141708492384, 3.264070667606171e-7,
          0.015079837844627948, 1.5891383112101438e-5, 0.1931406057562605,
          2.654124109083291e-7, 0.11654298695072397]
    riskt = 5.936075960042635e-5
    rett = 0.0003482663810696356
    @test isapprox(w17.weights, wt)
    @test isapprox(r17, riskt)
    @test isapprox(ret17, rett)

    rm = Variance(; formulation = Quad())
    w18 = optimise!(portfolio,
                    Trad(; rm = rm, kelly = AKelly(; formulation = Quad()), obj = obj))
    r18 = calc_risk(portfolio; type = :Trad, rm = rm)
    ret18 = dot(portfolio.mu, w18.weights)
    wt = [0.007931704468101724, 0.030580315769072094, 0.010456593899158567,
          0.02742093947332831, 0.012239100149899967, 0.033302579036721264,
          4.162859647692109e-6, 0.13981730989327237, 5.788675176975045e-6,
          0.00033179125802216067, 0.2877731391009407, 3.3160181115934523e-6,
          2.9562651082726095e-6, 0.12512918292552744, 3.227218484382772e-5,
          0.015034147440717766, 0.0005108903689970619, 0.19299166064346063,
          6.739278073914423e-6, 0.1164254102918178]
    riskt = 5.936214450112821e-5
    rett = 0.0003484772957329131
    @test isapprox(w18.weights, wt)
    @test isapprox(r18, riskt)
    @test isapprox(ret18, rett)

    @test isapprox(w17.weights, w18.weights, rtol = 0.005)
    @test isapprox(r17, r18, rtol = 5.0e-5)
    @test isapprox(ret17, ret18, rtol = 0.001)

    obj = Utility(; l = l)
    rm = Variance(; formulation = SOC())
    w19 = optimise!(portfolio,
                    Trad(; rm = rm, kelly = AKelly(; formulation = Quad()), obj = obj))
    r19 = calc_risk(portfolio; type = :Trad, rm = rm)
    ret19 = dot(portfolio.mu, w19.weights)
    wt = [8.870505348946403e-9, 2.0785636066552727e-8, 2.798137197308657e-8,
          1.971091339190289e-8, 0.7185881405281792, 4.806880649144299e-10,
          0.09860828964210354, 1.1235098321720224e-8, 2.977172777854582e-8,
          8.912749778026878e-9, 9.63062128166912e-9, 1.0360544993920464e-9,
          2.180352541614548e-9, 2.689800139816139e-9, 2.3063944199708073e-9,
          0.15518499560246005, 0.027618271886178034, 1.246121371211767e-8,
          1.2842725621709964e-7, 1.586069567397408e-8]
    riskt = 0.00023831960644703196
    rett = 0.0016792722833452185
    @test isapprox(w19.weights, wt)
    @test isapprox(r19, riskt)
    @test isapprox(ret19, rett)

    rm = Variance(; formulation = Quad())
    w20 = optimise!(portfolio,
                    Trad(; rm = rm, kelly = AKelly(; formulation = Quad()), obj = obj))
    r20 = calc_risk(portfolio; type = :Trad, rm = rm)
    ret20 = dot(portfolio.mu, w20.weights)
    wt = [1.6868171199009505e-9, 6.518606011133161e-9, 9.86352279770984e-9,
          6.150045875747458e-9, 0.718630983702095, 3.7119199852091685e-10,
          0.09861482354309821, 2.790532675752989e-9, 1.2728774591077898e-8,
          2.1871465657675766e-9, 2.2693974383129075e-9, 2.958986484663317e-10,
          2.0351357726703993e-10, 7.649820299793997e-10, 1.9650537328495345e-10,
          0.15517370021004367, 0.027580337279790704, 3.3046232245984773e-9,
          1.0179775163080922e-7, 4.1356629132822375e-9]
    riskt = 0.00023833463446320128
    rett = 0.0016793098997042355
    @test isapprox(w20.weights, wt)
    @test isapprox(r20, riskt)
    @test isapprox(ret20, rett)

    @test isapprox(w19.weights, w20.weights, rtol = 0.0001)
    @test isapprox(r19, r20, rtol = 0.0001)
    @test isapprox(ret19, ret20, rtol = 5.0e-5)

    obj = Sharpe(; rf = rf)
    rm = Variance(; formulation = SOC())
    w21 = optimise!(portfolio,
                    Trad(; rm = rm, kelly = AKelly(; formulation = Quad()), obj = obj))
    r21 = calc_risk(portfolio; type = :Trad, rm = rm)
    ret21 = dot(portfolio.mu, w21.weights)
    wt = [3.953983055390625e-7, 1.2130189854003184e-6, 1.5626692963148418e-6,
          9.099292074973729e-7, 0.4886931254919885, 1.1962647602545863e-7,
          0.05842798708631173, 1.2992007103695783e-6, 7.700590186769794e-7,
          4.5424814443255545e-7, 9.701300232241762e-7, 9.241777891765365e-8,
          6.70735556712963e-8, 2.2704051659598955e-7, 6.523754123014794e-8,
          0.14023347164242564, 0.21376781701297304, 1.095611158897337e-6,
          0.09886732333044786, 1.0337751344638193e-6]
    riskt = 0.00016559358507428642
    rett = 0.0014479293174607377
    @test isapprox(w21.weights, wt)
    @test isapprox(r21, riskt)
    @test isapprox(ret21, rett)

    rm = Variance(; formulation = Quad())
    w22 = optimise!(portfolio,
                    Trad(; rm = rm, kelly = AKelly(; formulation = Quad()), obj = obj))
    r22 = calc_risk(portfolio; type = :Trad, rm = rm)
    ret22 = dot(portfolio.mu, w22.weights)
    wt = [1.669718396236806e-8, 6.027044058934097e-8, 8.739577024175596e-8,
          4.301880600905147e-8, 0.48873017518452627, 5.254152037070482e-9,
          0.05843208512186604, 6.917260938265724e-8, 2.8809473152698165e-8,
          1.8499861374074213e-8, 4.4781660705625404e-8, 4.139805510566116e-9,
          2.9926847605727475e-9, 9.45059691996542e-9, 2.9559088456557273e-9,
          0.14024309499313364, 0.21373264542041767, 5.2284896629923713e-8,
          0.09886150576581468, 4.779039174782547e-8]
    riskt = 0.00016560238556231427
    rett = 0.0014479684757221218
    @test isapprox(w22.weights, wt, rtol = 5.0e-7)
    @test isapprox(r22, riskt, rtol = 1.0e-7)
    @test isapprox(ret22, rett, rtol = 5.0e-8)

    @test isapprox(w21.weights, w22.weights, rtol = 0.0001)
    @test isapprox(r21, r22, rtol = 0.0001)
    @test isapprox(ret21, ret22, rtol = 5.0e-5)

    obj = MaxRet()
    rm = Variance(; formulation = SOC())
    w23 = optimise!(portfolio,
                    Trad(; rm = rm, kelly = AKelly(; formulation = Quad()), obj = obj))
    r23 = calc_risk(portfolio; type = :Trad, rm = rm)
    ret23 = dot(portfolio.mu, w23.weights)
    wt = [3.997304959875111e-10, 6.096555731047457e-10, 1.0828360159599854e-9,
          8.390684135793784e-10, 0.8503431998881756, 5.837264433583887e-10,
          0.14965678808511193, 8.520932897976487e-13, 7.66160958682953e-10,
          1.0247860261071675e-10, 5.1627700971086255e-11, 5.483183958203547e-10,
          7.565204185674542e-10, 3.16106264753721e-10, 7.638502459889708e-10,
          2.447496129413098e-9, 1.372927322256315e-9, 6.541563185875491e-11,
          9.248420166125226e-10, 3.9509971643490626e-10]
    riskt = 0.000307068809623536
    rett = 0.001803059901755384
    @test isapprox(w23.weights, wt)
    @test isapprox(r23, riskt)
    @test isapprox(ret23, rett)

    rm = Variance(; formulation = Quad())
    w24 = optimise!(portfolio,
                    Trad(; rm = rm, kelly = AKelly(; formulation = Quad()), obj = obj))
    r24 = calc_risk(portfolio; type = :Trad, rm = rm)
    ret24 = dot(portfolio.mu, w24.weights)
    wt = [3.1491090891677036e-10, 3.9252510945045457e-10, 5.011436739931957e-10,
          4.814228838168751e-10, 0.8503245220331475, 4.987452504286351e-11,
          0.14967547181371002, 1.8257330907415293e-10, 4.5000351561986515e-10,
          2.195999249689875e-10, 1.8725349256894292e-10, 7.858729597392012e-11,
          2.333547428526958e-10, 1.351809534814174e-10, 1.8382016146366064e-10,
          1.1430290850714406e-9, 6.064115688191264e-10, 2.1745240525978677e-10,
          4.80092693730735e-10, 2.959062049654606e-10]
    riskt = 0.0003070706709250669
    rett = 0.0018030608400966524
    @test isapprox(w24.weights, wt)
    @test isapprox(r24, riskt)
    @test isapprox(ret24, rett)

    @test isapprox(w23.weights, w24.weights, rtol = 5.0e-5)
    @test isapprox(r23, r24, rtol = 1.0e-5)
    @test isapprox(ret23, ret24, rtol = 1.0e-6)

    obj = MinRisk()
    rm = Variance(; formulation = SOC())
    w25 = optimise!(portfolio, Trad(; rm = rm, kelly = EKelly(), obj = obj))
    r25 = calc_risk(portfolio; type = :Trad, rm = rm)
    ret25 = dot(portfolio.mu, w25.weights)
    wt = [0.007893834004010548, 0.030693397218184384, 0.01050943911162566,
          0.027487590678529683, 0.0122836015907984, 0.03341312720689581,
          2.654460293680794e-8, 0.13984817931920596, 4.861252776141605e-8,
          3.309876672531783e-7, 0.2878217133212084, 3.116270780466401e-8,
          2.2390519612760115e-8, 0.12528318523437287, 9.334010636386356e-8,
          0.015085761770714442, 7.170234777802365e-7, 0.19312554465008394,
          6.179100704970626e-8, 0.11655329404175341]
    riskt = 5.936072842537098e-5
    rett = 0.0003482418640932403
    @test isapprox(w25.weights, wt)
    @test isapprox(r25, riskt)
    @test isapprox(ret25, rett)

    rm = Variance(; formulation = Quad())
    w26 = optimise!(portfolio, Trad(; rm = rm, kelly = EKelly(), obj = obj))
    r26 = calc_risk(portfolio; type = :Trad, rm = rm)
    ret26 = dot(portfolio.mu, w26.weights)
    wt = [0.007903849541683496, 0.030674182061093153, 0.010497013909674717,
          0.027478080045244463, 0.01227276983521539, 0.03339506695222121,
          9.108585123593556e-8, 0.1398431330227047, 1.088800120401102e-7,
          5.459480997321388e-5, 0.28781541622794565, 6.401091570273526e-8,
          6.921020873995449e-8, 0.1252625404795329, 2.4246045907605043e-6,
          0.015077019228467355, 8.393931222812146e-5, 0.19310429397772946,
          1.152440704144604e-7, 0.11653522756063725]
    riskt = 5.936085534818755e-5
    rett = 0.00034827715166856516
    @test isapprox(w26.weights, wt, rtol = 5.0e-7)
    @test isapprox(r26, riskt)
    @test isapprox(ret26, rett, rtol = 5.0e-7)

    @test isapprox(w25.weights, w26.weights, rtol = 0.0005)
    @test isapprox(r25, r26, rtol = 5.0e-6)
    @test isapprox(ret25, ret26, rtol = 0.0005)

    obj = Utility(; l = l)
    rm = Variance(; formulation = SOC())
    w27 = optimise!(portfolio, Trad(; rm = rm, kelly = EKelly(), obj = obj))
    r27 = calc_risk(portfolio; type = :Trad, rm = rm)
    ret27 = dot(portfolio.mu, w27.weights)
    wt = [4.076718644400776e-9, 7.97059286965588e-9, 1.0440947901329591e-8,
          7.702064328487365e-9, 0.7207180459688085, 1.1223657131481055e-9,
          0.09835526321809454, 4.730656228269007e-9, 1.133279001422807e-8,
          4.082347527215102e-9, 4.208721035741791e-9, 9.44650173900743e-10,
          5.699105612892133e-10, 2.079988023687919e-9, 5.463657962369361e-10,
          0.15505269860336068, 0.02587387715494727, 5.1544684301828716e-9,
          4.384014669868401e-8, 6.25205503896986e-9]
    riskt = 0.00023889947439128677
    rett = 0.0016807187826177266
    @test isapprox(w27.weights, wt, rtol = 5.0e-7)
    @test isapprox(r27, riskt, rtol = 5.0e-7)
    @test isapprox(ret27, rett, rtol = 5.0e-8)

    rm = Variance(; formulation = Quad())
    w28 = optimise!(portfolio, Trad(; rm = rm, kelly = EKelly(), obj = obj))
    r28 = calc_risk(portfolio; type = :Trad, rm = rm)
    ret28 = dot(portfolio.mu, w28.weights)
    wt = [4.906137545253611e-10, 9.649391630853278e-10, 1.2701034801874472e-9,
          9.359495578646146e-10, 0.7207182393129586, 1.5333174883536851e-10,
          0.09835528520857263, 5.720441631202269e-10, 1.418133301927009e-9,
          4.955762282237675e-10, 5.097611222203524e-10, 1.3589683804670167e-10,
          9.65359446446974e-11, 2.607474831260744e-10, 9.240398341698498e-11,
          0.1550525996289181, 0.02587386065523803, 6.236903050825014e-10,
          6.42190516047178e-9, 7.526805986772741e-10]
    riskt = 0.00023889953965714231
    rett = 0.0016807189690271362
    @test isapprox(w28.weights, wt, rtol = 5.0e-7)
    @test isapprox(r28, riskt, rtol = 5.0e-7)
    @test isapprox(ret28, rett, rtol = 5.0e-8)

    @test isapprox(w27.weights, w28.weights, rtol = 1.0e-6)
    @test isapprox(r27, r28, rtol = 5.0e-7)
    @test isapprox(ret27, ret28, rtol = 5.0e-7)

    obj = Sharpe(; rf = rf)
    rm = Variance(; formulation = SOC())
    w29 = optimise!(portfolio, Trad(; rm = rm, kelly = EKelly(), obj = obj))
    r29 = calc_risk(portfolio; type = :Trad, rm = rm)
    ret29 = dot(portfolio.mu, w29.weights)
    wt = [1.7647799778761598e-7, 7.43791350184855e-7, 9.974860927108156e-7,
          4.726318325437073e-7, 0.48902202810838025, 5.25817986396541e-8,
          0.05826321657926799, 8.395640963247789e-7, 1.7634626972511063e-7,
          2.0620668133652036e-7, 5.77393877243638e-7, 4.130907666540682e-8,
          3.032865903519244e-8, 9.975084610937565e-8, 2.927622589710415e-8,
          0.140132597154028, 0.2135052983305069, 6.899386614850263e-7, 0.09907112697494824,
          5.99769403061889e-7]
    riskt = 0.00016561198996065772
    rett = 0.001448007912353208
    @test isapprox(w29.weights, wt, rtol = 0.001)
    @test isapprox(r29, riskt, rtol = 0.0005)
    @test isapprox(ret29, rett, rtol = 0.0005)

    rm = Variance(; formulation = Quad())
    w30 = optimise!(portfolio, Trad(; rm = rm, kelly = EKelly(), obj = obj))
    r30 = calc_risk(portfolio; type = :Trad, rm = rm)
    ret30 = dot(portfolio.mu, w30.weights)
    wt = [1.8065035676955756e-7, 6.209371911900467e-7, 8.219298442320451e-7,
          4.28759812352543e-7, 0.4892265348048135, 5.4889283577166394e-8,
          0.05831047678326769, 9.468124994336092e-7, 3.317011259802482e-7,
          2.0530407527981482e-7, 5.897602489929196e-7, 4.3652542841594486e-8,
          3.1516232812474066e-8, 1.0230480040716144e-7, 3.056839308404574e-8,
          0.14027987133379333, 0.2135545115699679, 6.541854618300521e-7,
          0.09862302976752084, 5.3276876801969e-7]
    riskt = 0.00016567991769890275
    rett = 0.0014482954242715832
    @test isapprox(w30.weights, wt, rtol = 5.0e-7)
    @test isapprox(r30, riskt, rtol = 1.0e-7)
    @test isapprox(ret30, rett, rtol = 5.0e-8)

    @test isapprox(w29.weights, w30.weights, rtol = 0.005)
    @test isapprox(r29, r30, rtol = 0.001)
    @test isapprox(ret29, ret30, rtol = 0.0005)

    obj = MaxRet()
    rm = Variance(; formulation = SOC())
    w31 = optimise!(portfolio, Trad(; rm = rm, kelly = EKelly(), obj = obj))
    r31 = calc_risk(portfolio; type = :Trad, rm = rm)
    ret31 = dot(portfolio.mu, w31.weights)
    wt = [2.633743649985236e-9, 2.9505379229172413e-9, 3.665208013478754e-9,
          3.305907804738003e-9, 0.8533949389480296, 1.1597137196070284e-9,
          0.14660501531709985, 2.0097483499520764e-9, 3.1965650000368884e-9,
          2.1746159802304897e-9, 1.9315555659189856e-9, 1.2031777907298662e-9,
          8.570011599717695e-10, 1.5424724997016623e-9, 8.7105116377961e-10,
          5.975487549944421e-9, 4.106592066239195e-9, 2.115816975845725e-9,
          3.4223392213476532e-9, 2.6133360746565418e-9]
    riskt = 0.0003067814463463604
    rett = 0.0018029079979477232
    @test isapprox(w31.weights, wt, rtol = 1.0e-7)
    @test isapprox(r31, riskt)
    @test isapprox(ret31, rett)

    rm = Variance(; formulation = Quad())
    w32 = optimise!(portfolio, Trad(; rm = rm, kelly = EKelly(), obj = obj))
    r32 = calc_risk(portfolio; type = :Trad, rm = rm)
    ret32 = dot(portfolio.mu, w32.weights)
    wt = [2.633743649985236e-9, 2.9505379229172413e-9, 3.665208013478754e-9,
          3.305907804738003e-9, 0.8533949389480296, 1.1597137196070284e-9,
          0.14660501531709985, 2.0097483499520764e-9, 3.1965650000368884e-9,
          2.1746159802304897e-9, 1.9315555659189856e-9, 1.2031777907298662e-9,
          8.570011599717695e-10, 1.5424724997016623e-9, 8.7105116377961e-10,
          5.975487549944421e-9, 4.106592066239195e-9, 2.115816975845725e-9,
          3.4223392213476532e-9, 2.6133360746565418e-9]
    riskt = 0.0003067814463463604
    rett = 0.0018029079979477232
    @test isapprox(w32.weights, wt, rtol = 1.0e-7)
    @test isapprox(r32, riskt)
    @test isapprox(ret32, rett)

    @test isapprox(w31.weights, w32.weights)
    @test isapprox(r31, r32)
    @test isapprox(ret31, ret32)

    obj = MaxRet()
    rm = Variance(; formulation = SOC())
    rm.settings.ub = r1
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test abs(calc_risk(portfolio; rm = rm, type = :Trad) - r1) <= 1e-11

    rm = Variance(; formulation = Quad())
    rm.settings.ub = r1
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test abs(calc_risk(portfolio; rm = rm, type = :Trad) - r1) <= 1e-11

    obj = Sharpe(; rf = rf)
    rm = Variance(; formulation = SOC())
    rm.settings.ub = r1
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test abs(calc_risk(portfolio; rm = rm, type = :Trad) - r1) <= 1e-11

    rm = Variance(; formulation = Quad())
    rm.settings.ub = r1
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test abs(calc_risk(portfolio; rm = rm, type = :Trad) - r1) <= 1e-11
end

@testset "Approx Kelly Formulations, non SD rm" begin
    portfolio = Portfolio(; prices = prices,
                          solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                           :check_sol => (allow_local = true,
                                                                          allow_almost = true),
                                                           :params => Dict("verbose" => false))))
    asset_statistics!(portfolio)

    obj = MinRisk()
    rm = CVaR()
    w1 = optimise!(portfolio,
                   Trad(; rm = rm, kelly = AKelly(; formulation = SOC()), obj = obj))
    wt = [9.204098576587342e-11, 0.04242033344850073, 9.851178943961487e-11,
          2.742446224128441e-10, 0.007574028452634096, 1.0444829577591807e-10,
          1.1253114859467353e-11, 0.09464947950883103, 4.304498662179531e-11,
          5.769494146605776e-11, 0.3040110652564133, 5.2022107005624686e-11,
          2.8850407100827043e-11, 0.06564166930507728, 9.716009959308051e-11,
          0.02937161116309481, 1.2139968762793123e-10, 0.3663101127840467,
          5.621906727997288e-11, 0.0900216990445119]
    @test isapprox(w1.weights, wt)

    w2 = optimise!(portfolio,
                   Trad(; rm = rm, kelly = AKelly(; formulation = Quad()), obj = obj))
    wt = [9.204098576587342e-11, 0.04242033344850073, 9.851178943961487e-11,
          2.742446224128441e-10, 0.007574028452634096, 1.0444829577591807e-10,
          1.1253114859467353e-11, 0.09464947950883103, 4.304498662179531e-11,
          5.769494146605776e-11, 0.3040110652564133, 5.2022107005624686e-11,
          2.8850407100827043e-11, 0.06564166930507728, 9.716009959308051e-11,
          0.02937161116309481, 1.2139968762793123e-10, 0.3663101127840467,
          5.621906727997288e-11, 0.0900216990445119]
    @test isapprox(w2.weights, wt)
    @test isapprox(w1.weights, w2.weights)

    obj = Sharpe(; rf = rf)
    w3 = optimise!(portfolio,
                   Trad(; rm = rm, kelly = AKelly(; formulation = SOC()), obj = obj))
    wt = [3.2592887766535115e-9, 4.791059887642716e-9, 5.249797493797327e-9,
          2.8711763368198212e-9, 0.5622701776345497, 1.0280292558806207e-9,
          0.04192142982970925, 1.0255141075230176e-8, 4.121252048908375e-9,
          2.2276807756373176e-9, 1.0367076788039217e-8, 9.40935771913547e-10,
          6.248034137050829e-10, 1.9096958890220713e-9, 6.454243382878365e-10,
          0.20851352690754332, 0.1872947839377768, 1.039921510320994e-8,
          1.6146206905377305e-8, 6.853637223964302e-9]
    @test isapprox(w3.weights, wt)

    w4 = optimise!(portfolio,
                   Trad(; rm = rm, kelly = AKelly(; formulation = Quad()), obj = obj))
    wt = [3.2592887766535115e-9, 4.791059887642716e-9, 5.249797493797327e-9,
          2.8711763368198212e-9, 0.5622701776345497, 1.0280292558806207e-9,
          0.04192142982970925, 1.0255141075230176e-8, 4.121252048908375e-9,
          2.2276807756373176e-9, 1.0367076788039217e-8, 9.40935771913547e-10,
          6.248034137050829e-10, 1.9096958890220713e-9, 6.454243382878365e-10,
          0.20851352690754332, 0.1872947839377768, 1.039921510320994e-8,
          1.6146206905377305e-8, 6.853637223964302e-9]
    @test isapprox(w4.weights, wt)
    @test isapprox(w3.weights, w4.weights)

    obj = MinRisk()
    rm = [CDaR(), Variance()]
    w5 = optimise!(portfolio, Trad(; rm = rm, kelly = AKelly(), obj = obj);)
    wt = [3.0140842192047997e-12, 1.787134110937184e-10, 5.592260476575972e-12,
          6.419005790016778e-13, 0.003409910046771097, 1.0130887237699018e-12,
          4.34349179775756e-12, 0.07904283645064122, 6.078039691287873e-12,
          8.069382194823101e-13, 0.3875931628052873, 2.0960478141272157e-11,
          3.079574073139989e-11, 4.9577938775743915e-11, 0.0005545143505126257,
          0.0959882887370522, 0.26790886767993305, 1.0297127257935643e-10,
          0.0006560189794797459, 0.16484640054581412]
    @test isapprox(w5.weights, wt)

    rm = [CDaR(), [Variance(), Variance()]]
    w6 = optimise!(portfolio, Trad(; rm = rm, kelly = AKelly(), obj = obj);)
    wt = [1.5768872930601726e-12, 8.677442278134462e-11, 2.9153706602582253e-12,
          3.6098097550578777e-13, 0.0036748952209461295, 4.520865389736217e-13,
          2.1207996542890634e-12, 0.07937575544792427, 3.0766704137737114e-12,
          3.4211526762454803e-13, 0.3873790549560513, 1.00837174295667e-11,
          1.5385954223103806e-11, 2.4839232852770276e-11, 0.0005657309313687338,
          0.09595979975394403, 0.26758637922488315, 5.3731343771365196e-11,
          0.0006505694386978859, 0.16480781482452508]
    @test isapprox(w6.weights, wt)
    @test isapprox(w5.weights, w6.weights, rtol = 0.005)

    obj = Sharpe(; rf = rf)
    rm = [CDaR(), Variance()]
    w7 = optimise!(portfolio, Trad(; rm = rm, kelly = AKelly(), obj = obj);)
    wt = [1.9888249507425084e-9, 3.5254543512471607e-9, 0.07884297116011559,
          1.0943427727912886e-9, 0.3121450921967167, 5.660343936949406e-10,
          7.966292041656362e-10, 0.14128662850497914, 9.626714885639305e-10,
          1.014377343775238e-9, 0.14387791153074742, 7.30717929638123e-10,
          3.783672796046428e-10, 1.7117775686710675e-9, 5.891624301226878e-10,
          0.23039408053454055, 9.484822775890805e-9, 2.3929879127048082e-9,
          2.6525141607518525e-9, 0.093453288184216]
    @test isapprox(w7.weights, wt, rtol = 5.0e-7)

    rm = [CDaR(), [Variance(), Variance()]]
    w8 = optimise!(portfolio, Trad(; rm = rm, kelly = AKelly(), obj = obj);)
    wt = [6.405400069198978e-9, 1.1376329592965537e-8, 0.08236939537917244,
          3.55534346999143e-9, 0.3180379618984808, 1.7984037387886066e-9,
          2.938963715650467e-9, 0.1448806429704737, 3.099287057022163e-9,
          3.244420822479567e-9, 0.14122128106967594, 2.29410413584785e-9,
          1.1929603858874258e-9, 5.52374869284604e-9, 1.8466639966855422e-9,
          0.22849806673608258, 3.382991843367897e-8, 7.746952804630744e-9,
          8.952814714588622e-9, 0.08499255814080282]
    @test isapprox(w8.weights, wt)
    @test isapprox(w7.weights, w8.weights, rtol = 0.05)

    obj = MinRisk()
    rm = [CVaR(), Variance(; settings = RMSettings(; flag = false))]
    w9 = optimise!(portfolio,
                   Trad(; rm = rm, kelly = AKelly(; formulation = Quad()), obj = obj))
    wt = [9.204153687833781e-11, 0.04242033344850122, 9.85123787201143e-11,
          2.742462482595728e-10, 0.007574028452637937, 1.0444892006359854e-10,
          1.1253189456551386e-11, 0.09464947950883604, 4.304524873576078e-11,
          5.769528999444338e-11, 0.3040110652564113, 5.2022422074226226e-11,
          2.8850585492519713e-11, 0.06564166930506558, 9.716068090161583e-11,
          0.029371611163095106, 1.2140041190522942e-10, 0.366310112784046,
          5.621940710019265e-11, 0.09002169904451053]
    @test isapprox(w9.weights, wt)

    rm = [CVaR(),
          [Variance(; settings = RMSettings(; flag = false)),
           Variance(; settings = RMSettings(; flag = false))]]
    w10 = optimise!(portfolio,
                    Trad(; rm = rm, kelly = AKelly(; formulation = Quad()), obj = obj))
    wt = [9.204153687833781e-11, 0.04242033344850122, 9.85123787201143e-11,
          2.742462482595728e-10, 0.007574028452637937, 1.0444892006359854e-10,
          1.1253189456551386e-11, 0.09464947950883604, 4.304524873576078e-11,
          5.769528999444338e-11, 0.3040110652564113, 5.2022422074226226e-11,
          2.8850585492519713e-11, 0.06564166930506558, 9.716068090161583e-11,
          0.029371611163095106, 1.2140041190522942e-10, 0.366310112784046,
          5.621940710019265e-11, 0.09002169904451053]
    @test isapprox(w10.weights, wt)
    @test isapprox(w9.weights, w10.weights)

    obj = Sharpe(; rf = rf)
    rm = [CVaR(), Variance(; settings = RMSettings(; flag = false))]
    w11 = optimise!(portfolio, Trad(; rm = rm, kelly = AKelly(), obj = obj))
    wt = [3.260162507546655e-9, 4.792306152533942e-9, 5.251125060455357e-9,
          2.871937096039082e-9, 0.5622701776208648, 1.0283013870249794e-9,
          0.041921429790140866, 1.025778862473136e-8, 4.1224267746471814e-9,
          2.228299691233297e-9, 1.0369206792942252e-8, 9.411813765003053e-10,
          6.249610865941219e-10, 1.9102195169153173e-9, 6.455896646042745e-10,
          0.20851352691681913, 0.18729478395901494, 1.0402139991509507e-8,
          1.6151939224864303e-8, 6.855575116847886e-9]
    @test isapprox(w11.weights, wt)

    rm = [CVaR(),
          [Variance(; settings = RMSettings(; flag = false)),
           Variance(; settings = RMSettings(; flag = false))]]
    w12 = optimise!(portfolio, Trad(; rm = rm, kelly = AKelly(), obj = obj))
    wt = [7.603408154002885e-9, 1.1152364863644904e-8, 1.2188541976061238e-8,
          6.796575566980297e-9, 0.5622701801805305, 2.407503224790042e-9,
          0.04192109919331281, 2.3687180096337898e-8, 9.973686891036133e-9,
          5.287396587637505e-9, 2.3249158909945905e-8, 2.2064009298341627e-9,
          1.4608281579287036e-9, 4.475233426819531e-9, 1.5063635973708286e-9,
          0.20851329093797066, 0.18729523309857254, 2.509298144519644e-8,
          4.3438025154331117e-8, 1.606396453121718e-8]
    @test isapprox(w12.weights, wt)
    @test isapprox(w11.weights, w12.weights, rtol = 1e-6)
end

@testset "SD" begin
    portfolio = Portfolio(; prices = prices,
                          solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                           :check_sol => (allow_local = true,
                                                                          allow_almost = true),
                                                           :params => Dict("verbose" => false))))
    asset_statistics!(portfolio)

    rm = SD()

    obj = MinRisk()
    w1 = optimise!(portfolio, Trad(; rm = rm, kelly = NoKelly(), obj = obj))
    r1 = calc_risk(portfolio; type = :Trad, rm = rm)
    ret1 = dot(portfolio.mu, w1.weights)
    wt = [0.007892956618378178, 0.03069135078310372, 0.010510053328054379,
          0.02748716488934741, 0.012284726286598107, 0.033412747896745434,
          3.4510983561959808e-9, 0.1398498186599616, 6.463477389090647e-9,
          1.2183912701620326e-7, 0.28783457711461696, 3.971402286172399e-9,
          2.9059474605121267e-9, 0.12527565747300592, 1.1534530258372196e-8,
          0.015085797229230475, 3.1071292428662565e-7, 0.19312938283717607,
          8.183117329711524e-9, 0.11654529782215743]
    riskt = 0.007704591083785214
    rett = 0.00034824217501046187
    @test isapprox(w1.weights, wt)
    @test isapprox(r1, riskt)
    @test isapprox(ret1, rett)

    obj = Utility(; l = l)
    w2 = optimise!(portfolio, Trad(; rm = rm, kelly = NoKelly(), obj = obj))
    r2 = calc_risk(portfolio; type = :Trad, rm = rm)
    ret2 = dot(portfolio.mu, w2.weights)
    wt = [3.3845525734762743e-7, 0.03245411192485674, 0.014041141771232574,
          0.028634625422379358, 0.030290985036508705, 0.005376551745731238,
          4.200673931391037e-9, 0.13652895353559166, 6.6304529461540396e-9,
          0.0006160259140775326, 0.29064360722877763, 2.67074552134372e-9,
          1.9721256449954764e-9, 0.11178585267210861, 5.669569610426783e-9,
          0.020423298923273724, 0.015543439516805872, 0.19712337633735869,
          9.611449035566669e-9, 0.11653766076102358]
    riskt = 0.0077229723137168345
    rett = 0.0004207388247934707
    @test isapprox(w2.weights, wt)
    @test isapprox(r2, riskt)
    @test isapprox(ret2, rett)

    obj = Sharpe(; rf = rf)
    w3 = optimise!(portfolio, Trad(; rm = rm, kelly = NoKelly(), obj = obj))
    r3 = calc_risk(portfolio; type = :Trad, rm = rm)
    ret3 = dot(portfolio.mu, w3.weights)
    wt = [9.288153197917532e-9, 2.5287441861372433e-8, 3.262334495626996e-8,
          2.042021085171968e-8, 0.518057899726503, 2.824107456066348e-9,
          0.06365089409237853, 2.1380549780544252e-8, 2.025773500381765e-8,
          1.0387505704405014e-8, 1.727503562974513e-8, 2.2309535984252933e-9,
          1.6025905812586714e-9, 5.242822500115322e-9, 1.5543835762675563e-9,
          0.1432677690591166, 0.1964967492101948, 2.0271309849859462e-8,
          0.07852647637780057, 2.0887861949385024e-8]
    riskt = 0.013160919559594407
    rett = 0.0014788474312895242
    @test isapprox(w3.weights, wt)
    @test isapprox(r3, riskt)
    @test isapprox(ret3, rett)

    obj = MaxRet()
    w4 = optimise!(portfolio, Trad(; rm = rm, kelly = NoKelly(), obj = obj))
    r4 = calc_risk(portfolio; type = :Trad, rm = rm)
    ret4 = dot(portfolio.mu, w4.weights)
    wt = [8.719239065561217e-10, 8.89979881250899e-10, 9.962391574545242e-10,
          9.646258079031587e-10, 8.401034275877383e-9, 4.828906105645332e-10,
          0.9999999782316086, 6.87727997367879e-10, 9.021263551270326e-10,
          7.350693996493505e-10, 6.753002461228969e-10, 5.009649350579108e-10,
          3.7428368039997965e-10, 5.884337547691459e-10, 3.9326986484718143e-10,
          8.842556785821523e-10, 9.784139669171374e-10, 7.146277206720297e-10,
          9.044289592659792e-10, 8.2279511460579e-10]
    riskt = 0.040597851628968784
    rett = 0.0018453756308089402
    @test isapprox(w4.weights, wt)
    @test isapprox(r4, riskt)
    @test isapprox(ret4, rett)

    obj = MinRisk()
    w5 = optimise!(portfolio,
                   Trad(; rm = rm, kelly = AKelly(; formulation = SOC()), obj = obj))
    r5 = calc_risk(portfolio; type = :Trad, rm = rm)
    ret5 = dot(portfolio.mu, w5.weights)
    wt = [0.007906622824521838, 0.0306895984127092, 0.010507018213921474,
          0.027486712348764736, 0.012278391699589107, 0.03341107486462988,
          1.327144418389713e-8, 0.1398483157890693, 2.402388441885521e-8,
          8.7926866323909e-7, 0.2878224640775026, 1.4738175403912173e-8,
          1.0360543938119554e-8, 0.12528372442400512, 4.35276992990535e-8,
          0.015084986855759729, 1.6666832696184048e-6, 0.19312428641477594,
          3.0429690349691976e-8, 0.11655412177138057]
    riskt = 0.007704591293677231
    rett = 0.0003482372557706726
    @test isapprox(w5.weights, wt)
    @test isapprox(r5, riskt)
    @test isapprox(ret5, rett)

    obj = Utility(; l = l)
    w6 = optimise!(portfolio,
                   Trad(; rm = rm, kelly = AKelly(; formulation = SOC()), obj = obj))
    r6 = calc_risk(portfolio; type = :Trad, rm = rm)
    ret6 = dot(portfolio.mu, w6.weights)
    wt = [2.640214395791425e-7, 0.032453826203609225, 0.01403599132742926,
          0.028634298225380724, 0.030229200150996755, 0.005494459391852277,
          2.5973081619687135e-9, 0.13654410089119845, 4.086320037511638e-9,
          0.0005854739366724334, 0.2906338058853499, 1.708323945861404e-9,
          1.2723713776231143e-9, 0.11184501222840455, 3.4219077389509126e-9,
          0.020404710183255025, 0.01547480604161841, 0.19711662680433542,
          5.903480064430017e-9, 0.11654740571874671]
    riskt = 0.007722830216304192
    rett = 0.0004204541017196187
    @test isapprox(w6.weights, wt)
    @test isapprox(r6, riskt)
    @test isapprox(ret6, rett)

    obj = Sharpe(; rf = rf)
    w7 = optimise!(portfolio,
                   Trad(; rm = rm, kelly = AKelly(; formulation = SOC()), obj = obj))
    r7 = calc_risk(portfolio; type = :Trad, rm = rm)
    ret7 = dot(portfolio.mu, w7.weights)
    wt = [1.8458276350518707e-7, 1.1242055118543265e-6, 1.7350829904800117e-6,
          6.09277335798391e-7, 0.48862429310867145, 5.986068358297053e-8,
          0.058371436862563957, 1.4692544571231515e-6, 3.4142035041849293e-7,
          1.922150865329865e-7, 7.5820062323825e-7, 4.336284981809023e-8,
          2.992265608855721e-8, 1.0399449339249306e-7, 3.216088378118804e-8,
          0.1402474230976331, 0.21381304326115025, 9.558248701649777e-7, 0.0989354081841314,
          7.561202941486916e-7]
    riskt = 0.012867316042419063
    rett = 0.0014478208808539196
    @test isapprox(w7.weights, wt, rtol = 5.0e-8)
    @test isapprox(r7, riskt)
    @test isapprox(ret7, rett)

    obj = MaxRet()
    w8 = optimise!(portfolio,
                   Trad(; rm = rm, kelly = AKelly(; formulation = SOC()), obj = obj))
    r8 = calc_risk(portfolio; type = :Trad, rm = rm)
    ret8 = dot(portfolio.mu, w8.weights)
    wt = [8.200717678498626e-12, 1.165944066433638e-11, 1.9491550737530498e-11,
          1.5564279602540793e-11, 0.8503256615648446, 7.843186626655801e-12,
          0.14967433822357495, 1.4191391225848078e-12, 1.431718486635305e-11,
          3.2006772615212402e-12, 5.682201673569556e-13, 7.367687860519602e-12,
          1.105979506497775e-11, 3.670722791006216e-12, 1.0886466989257992e-11,
          4.468015906039003e-11, 2.4293660105933157e-11, 2.5727569981528378e-12,
          1.6799258000949892e-11, 7.985555398181619e-12]
    riskt = 0.017523428903989537
    rett = 0.0018030607904410188
    @test isapprox(w8.weights, wt)
    @test isapprox(r8, riskt)
    @test isapprox(ret8, rett)

    obj = MinRisk()
    w9 = optimise!(portfolio,
                   Trad(; rm = rm, kelly = AKelly(; formulation = Quad()), obj = obj))
    r9 = calc_risk(portfolio; type = :Trad, rm = rm)
    ret9 = dot(portfolio.mu, w9.weights)
    wt = [0.007906622824521838, 0.0306895984127092, 0.010507018213921474,
          0.027486712348764736, 0.012278391699589107, 0.03341107486462988,
          1.327144418389713e-8, 0.1398483157890693, 2.402388441885521e-8,
          8.7926866323909e-7, 0.2878224640775026, 1.4738175403912173e-8,
          1.0360543938119554e-8, 0.12528372442400512, 4.35276992990535e-8,
          0.015084986855759729, 1.6666832696184048e-6, 0.19312428641477594,
          3.0429690349691976e-8, 0.11655412177138057]
    riskt = 0.007704591293677231
    rett = 0.0003482372557706726
    @test isapprox(w9.weights, wt)
    @test isapprox(r9, riskt)
    @test isapprox(ret9, rett)

    obj = Utility(; l = l)
    w10 = optimise!(portfolio,
                    Trad(; rm = rm, kelly = AKelly(; formulation = Quad()), obj = obj))
    r10 = calc_risk(portfolio; type = :Trad, rm = rm)
    ret10 = dot(portfolio.mu, w10.weights)
    wt = [2.744898111887545e-7, 0.03245376880429415, 0.0140359794436688,
          0.028634276757516522, 0.03022915072666002, 0.005494724832370341,
          2.737068173582613e-9, 0.13654410280331405, 4.313314153767035e-9,
          0.0005851751493369103, 0.290633700515106, 1.801499106458113e-9,
          1.3407046445457633e-9, 0.11184510920447174, 3.6137483918526165e-9,
          0.020404709063953468, 0.015474870825434297, 0.1971165775419533,
          6.233889543783176e-9, 0.11654755980188516]
    riskt = 0.007722830055349047
    rett = 0.00042045377532140305
    @test isapprox(w10.weights, wt)
    @test isapprox(r10, riskt)
    @test isapprox(ret10, rett)

    obj = Sharpe(; rf = rf)
    w11 = optimise!(portfolio,
                    Trad(; rm = rm, kelly = AKelly(; formulation = Quad()), obj = obj))
    r11 = calc_risk(portfolio; type = :Trad, rm = rm)
    ret11 = dot(portfolio.mu, w11.weights)
    wt = [1.8458276350518707e-7, 1.1242055118543265e-6, 1.7350829904800117e-6,
          6.09277335798391e-7, 0.48862429310867145, 5.986068358297053e-8,
          0.058371436862563957, 1.4692544571231515e-6, 3.4142035041849293e-7,
          1.922150865329865e-7, 7.5820062323825e-7, 4.336284981809023e-8,
          2.992265608855721e-8, 1.0399449339249306e-7, 3.216088378118804e-8,
          0.1402474230976331, 0.21381304326115025, 9.558248701649777e-7, 0.0989354081841314,
          7.561202941486916e-7]
    riskt = 0.012867316042419063
    rett = 0.0014478208808539196
    @test isapprox(w11.weights, wt, rtol = 5.0e-8)
    @test isapprox(r11, riskt)
    @test isapprox(ret11, rett)

    obj = MaxRet()
    w12 = optimise!(portfolio,
                    Trad(; rm = rm, kelly = AKelly(; formulation = Quad()), obj = obj))
    r12 = calc_risk(portfolio; type = :Trad, rm = rm)
    ret12 = dot(portfolio.mu, w12.weights)
    wt = [9.327601044474791e-11, 1.217521270051073e-10, 1.2469188778283442e-10,
          1.4001482690699294e-10, 0.8503245365370284, 2.0602657811834836e-11,
          0.14967546152840527, 5.394646310425498e-11, 1.49623554681287e-10,
          6.90823191379229e-11, 6.208186399055279e-11, 2.826959813733456e-11,
          4.071757440562292e-11, 4.467430554304692e-11, 1.320580036318213e-10,
          3.720933633992451e-10, 1.6285487328809039e-10, 7.234379443637997e-11,
          1.5192008851655943e-10, 9.456303107536222e-11]
    riskt = 0.017523432073549134
    rett = 0.0018030608442894938
    @test isapprox(w12.weights, wt)
    @test isapprox(r12, riskt)
    @test isapprox(ret12, rett)

    obj = MinRisk()
    w13 = optimise!(portfolio, Trad(; rm = rm, kelly = EKelly(), obj = obj))
    r13 = calc_risk(portfolio; type = :Trad, rm = rm)
    ret13 = dot(portfolio.mu, w13.weights)
    wt = [0.00790409633874847, 0.030689928037939188, 0.010507998487432462,
          0.02748686384854246, 0.012279538174146456, 0.033412706024557044,
          5.863252541677311e-10, 0.1398487299449627, 1.0456745320204646e-9,
          2.356042145726172e-8, 0.2878234001518533, 6.520325820725347e-10,
          4.78341311046709e-10, 0.125283393778506, 2.0064216192084034e-9,
          0.015084912392169235, 4.354723284523309e-8, 0.1931247757099064,
          1.3224469461845564e-9, 0.11655358391233962]
    riskt = 0.007704591029545491
    rett = 0.0003482360682112058
    @test isapprox(w13.weights, wt)
    @test isapprox(r13, riskt)
    @test isapprox(ret13, rett)

    obj = Utility(; l = l)
    w14 = optimise!(portfolio, Trad(; rm = rm, kelly = EKelly(), obj = obj))
    r14 = calc_risk(portfolio; type = :Trad, rm = rm)
    ret14 = dot(portfolio.mu, w14.weights)
    wt = [2.836963234129393e-8, 0.032455792649287205, 0.014033059013617231,
          0.02863185443671385, 0.03022338148892875, 0.005503606381703397,
          1.0363728488707515e-9, 0.13654489926552815, 1.6684665131812563e-9,
          0.000592676681830992, 0.2906339250165604, 6.987305189376417e-10,
          5.245214029217624e-10, 0.11184763715457732, 1.441953964683239e-9,
          0.020401719061588303, 0.015468273452356863, 0.19711624998221147,
          2.413380037921624e-9, 0.11654688926203842]
    riskt = 0.0077228169828044355
    rett = 0.0004204275685221765
    @test isapprox(w14.weights, wt, rtol = 5.0e-5)
    @test isapprox(r14, riskt, rtol = 5.0e-7)
    @test isapprox(ret14, rett, rtol = 1.0e-5)

    obj = Sharpe(; rf = rf)
    w15 = optimise!(portfolio, Trad(; rm = rm, kelly = EKelly(), obj = obj))
    r15 = calc_risk(portfolio; type = :Trad, rm = rm)
    ret15 = dot(portfolio.mu, w15.weights)
    wt = [3.1327630095042823e-7, 1.0196082287391234e-6, 1.3356589900729345e-6,
          7.431349458373225e-7, 0.4895003985767998, 9.47038780059679e-8,
          0.05835377215690013, 1.159618235789627e-6, 6.501483098601631e-7,
          3.5366805691866187e-7, 8.290334151035846e-7, 7.536156665827754e-8,
          5.434235900536822e-8, 1.765842834020984e-7, 5.283885973094895e-8,
          0.1403322526828587, 0.21340504034222177, 9.399379925400543e-7,
          0.09839987796764298, 8.603581539853723e-7]
    riskt = 0.012874349687070808
    rett = 0.0014485863654269517
    @test isapprox(w15.weights, wt)
    @test isapprox(r15, riskt)
    @test isapprox(ret15, rett)

    obj = MaxRet()
    w48 = optimise!(portfolio, Trad(; rm = rm, kelly = EKelly(), obj = obj))
    r48 = calc_risk(portfolio; type = :Trad, rm = rm)
    ret48 = dot(portfolio.mu, w48.weights)
    wt = [2.633743649985236e-9, 2.9505379229172413e-9, 3.665208013478754e-9,
          3.305907804738003e-9, 0.8533949389480296, 1.1597137196070284e-9,
          0.14660501531709985, 2.0097483499520764e-9, 3.1965650000368884e-9,
          2.1746159802304897e-9, 1.9315555659189856e-9, 1.2031777907298662e-9,
          8.570011599717695e-10, 1.5424724997016623e-9, 8.7105116377961e-10,
          5.975487549944421e-9, 4.106592066239195e-9, 2.115816975845725e-9,
          3.4223392213476532e-9, 2.6133360746565418e-9]
    riskt = 0.017515177599623715
    rett = 0.0018029079979477232
    @test isapprox(w48.weights, wt, rtol = 1.0e-7)
    @test isapprox(r48, riskt)
    @test isapprox(ret48, rett)
end

@testset "MAD" begin
    portfolio = Portfolio(; prices = prices,
                          solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                           :check_sol => (allow_local = true,
                                                                          allow_almost = true),
                                                           :params => Dict("verbose" => false))))
    asset_statistics!(portfolio)
    rm = MAD()

    obj = MinRisk()
    w1 = optimise!(portfolio, Trad(; rm = rm, kelly = NoKelly(), obj = obj))
    r1 = calc_risk(portfolio; type = :Trad, rm = rm)
    ret1 = dot(portfolio.mu, w1.weights)
    wt = [0.014061868668321136, 0.042374931066480125, 0.016863352606484652,
          0.0020208147131289722, 0.017683877638866075, 0.054224070727585096,
          1.5203458994558942e-10, 0.15821654686783665, 1.5215634326982302e-10,
          3.2940703692302626e-10, 0.23689726473347567, 3.6543867640883435e-11,
          3.206825794173305e-11, 0.12783200362381844, 0.00035095362264346173,
          0.0009122872782575071, 0.043949384701932145, 0.1827242897900749,
          2.05020287223712e-10, 0.10188835305386476]
    riskt = 0.005627573037670243
    rett = 0.0003490123937090518
    @test isapprox(w1.weights, wt)
    @test isapprox(r1, riskt)
    @test isapprox(ret1, rett)

    w2 = optimise!(portfolio, Trad(; rm = rm, kelly = AKelly(), obj = obj))
    wt = [0.014061863385848581, 0.04237498184231898, 0.01686336630144988,
          0.002020817224301023, 0.017683873550852366, 0.05422408515584866,
          1.387753856939975e-10, 0.15821654369918353, 1.270567497872515e-10,
          2.926848054198672e-10, 0.23689726734186892, 2.9521547821185666e-11,
          2.5593796366110068e-11, 0.12783199137044676, 0.0003509560143046383,
          0.0009122807490557987, 0.04394942057368978, 0.18272429271723745,
          1.6372602178857848e-10, 0.1018882592962354]
    @test isapprox(w2.weights, wt)

    w3 = optimise!(portfolio, Trad(; rm = rm, kelly = EKelly(), obj = obj))
    wt = [0.014061797876338947, 0.04237519499239779, 0.016863459312604988,
          0.0020208328757590683, 0.017683833434707426, 0.054224138533348794,
          7.863915186715287e-10, 0.1582165054573331, 7.588190190840074e-10,
          1.7607068078312442e-9, 0.23689721053079502, 2.122315275831388e-10,
          1.896930448019363e-10, 0.12783196608816202, 0.0003509968449856307,
          0.0009122578712989519, 0.04394953194662374, 0.18272435509275958,
          9.790161415272732e-10, 0.1018879144560269]
    @test isapprox(w3.weights, wt)

    obj = Utility(; l = l)
    w4 = optimise!(portfolio, Trad(; rm = rm, kelly = NoKelly(), obj = obj))
    r2 = calc_risk(portfolio; type = :Trad, rm = rm)
    ret2 = dot(portfolio.mu, w4.weights)
    wt = [0.01647361822610948, 0.046524427215187444, 0.016357566973298703,
          0.013095667846448401, 0.038235416839893316, 0.010541877382994481,
          3.191194569039483e-10, 0.14372001930723385, 2.8417177519271143e-10,
          0.014100839489151853, 0.22271602252730396, 2.1961732060886273e-11,
          1.8748442633614797e-11, 0.10528245850706676, 7.626784696794497e-10,
          0.009373212289272542, 0.07294307090505608, 0.19175135885323213,
          2.0308455199850216e-9, 0.09888444020022567]
    riskt = 0.0056539990640096975
    rett = 0.00046515773122973215
    @test isapprox(w4.weights, wt)
    @test isapprox(r2, riskt)
    @test isapprox(ret2, rett)

    w5 = optimise!(portfolio, Trad(; rm = rm, kelly = AKelly(), obj = obj))
    wt = [0.016473743338205324, 0.04652437618414195, 0.016357674926649513,
          0.013095479709898642, 0.03823535011138278, 0.01054187676476419,
          2.8157800290108366e-9, 0.14372003323261792, 1.3745792698107461e-9,
          0.014100589809353248, 0.22271656597585632, 4.707772428526545e-10,
          5.098291787576843e-10, 0.10528250113760682, 6.308826520315478e-9,
          0.009373211821600945, 0.07294282014031918, 0.1917512597670182,
          1.7651570025781863e-8, 0.09888448794922271]
    @test isapprox(w5.weights, wt)

    w6 = optimise!(portfolio, Trad(; rm = rm, kelly = EKelly(), obj = obj))
    wt = [0.016473615847189163, 0.04652432174038814, 0.01635760018049102,
          0.01309554984323759, 0.03823530856590744, 0.010541756213120158,
          3.211172147123902e-9, 0.14371997790310523, 1.93305065009931e-9,
          0.014100650655050954, 0.22271650699370943, 2.0293945194487924e-10,
          1.729239196814806e-10, 0.1052827953629525, 5.4015200175889305e-9,
          0.00937319477093643, 0.07294290064314823, 0.1917511803078885,
          1.521930968656527e-8, 0.0988846148319594]
    @test isapprox(w6.weights, wt, rtol = 5.0e-6)

    obj = Sharpe(; rf = rf)
    w7 = optimise!(portfolio, Trad(; rm = rm, kelly = NoKelly(), obj = obj))
    r3 = calc_risk(portfolio; type = :Trad, rm = rm)
    ret3 = dot(portfolio.mu, w7.weights)
    wt = [7.934194869562142e-9, 2.040909734049381e-8, 2.0779015907725986e-8,
          1.1275212895546124e-8, 0.6622287168576108, 2.313094271042932e-9,
          0.04258591096562903, 1.189757129417062e-8, 2.2184758888576742e-8,
          1.0275636987162422e-8, 1.8861513364090324e-8, 1.610638743736056e-9,
          1.1200721055135385e-9, 4.334325691487995e-9, 1.244142335605315e-9,
          0.13436790162356135, 0.08741391009411406, 1.7249393491131058e-8,
          0.0734033914311675, 1.753924918834114e-8]
    riskt = 0.009899090393769974
    rett = 0.0015742126888891649
    @test isapprox(w7.weights, wt)
    @test isapprox(r3, riskt)
    @test isapprox(ret3, rett)

    w8 = optimise!(portfolio, Trad(; rm = rm, kelly = AKelly(), obj = obj))
    wt = [2.16570599053621e-11, 5.83494553105452e-11, 7.207206628852935e-11,
          3.132639273740643e-11, 0.5792409295185824, 6.645072983634004e-12,
          0.038919410356779975, 5.0266422946023506e-11, 4.789327545224995e-11,
          2.5687050054593087e-11, 8.786936422936622e-11, 4.521257915653591e-12,
          3.330649010405318e-12, 1.2548831669696809e-11, 3.607888280248482e-12,
          0.13600806842772567, 0.13940515527614972, 6.632264807794442e-11,
          0.10642643587312858, 5.5536260955281533e-11]
    @test isapprox(w8.weights, wt)

    w9 = optimise!(portfolio, Trad(; rm = rm, kelly = EKelly(), obj = obj))
    wt = [1.2411833797983574e-8, 3.31862551772405e-8, 4.060486183974115e-8,
          1.7892869078383173e-8, 0.579198528089336, 3.8009092878956895e-9,
          0.03882817486729307, 2.7980696087232757e-8, 2.780606165978267e-8,
          1.4683737826469131e-8, 4.8598317913787205e-8, 2.5983064214690757e-9,
          1.914415563042893e-9, 7.175434995935641e-9, 2.073773731335218e-9,
          0.1359806178950477, 0.13965987829953413, 3.7067483961690605e-8,
          0.10633249112117524, 3.1932656386940196e-8]
    @test isapprox(w9.weights, wt, rtol = 5.0e-5)

    obj = MaxRet()
    w10 = optimise!(portfolio, Trad(; rm = rm, kelly = NoKelly(), obj = obj))
    r4 = calc_risk(portfolio; type = :Trad, rm = rm)
    ret4 = dot(portfolio.mu, w10.weights)
    wt = [1.2971056777827964e-8, 1.3930988564983735e-8, 1.66544480132768e-8,
          1.6052005598315065e-8, 2.5272650716621174e-7, 1.010529781274771e-8,
          0.9999995082509356, 1.1380213103425267e-8, 1.4878768493855394e-8,
          1.2066244486141283e-8, 1.135216260978634e-8, 1.00771678859298e-8,
          9.057732065596785e-9, 1.08055110966454e-8, 9.429257146092314e-9,
          2.331510717112652e-8, 1.7720775475614838e-8, 1.1669132650930886e-8,
          1.5021261060062696e-8, 1.25354272189245e-8]
    riskt = 0.02624326616973302
    rett = 0.0018453753062770842
    @test isapprox(w10.weights, wt)
    @test isapprox(r4, riskt)
    @test isapprox(ret4, rett)

    w11 = optimise!(portfolio, Trad(; rm = rm, kelly = AKelly(), obj = obj))
    wt = [6.24782988116139e-10, 9.366363865924148e-10, 1.6428209739519796e-9,
          1.2834078728926378e-9, 0.8503450577361338, 8.327210654169724e-10,
          0.14965492407526051, 1.8834252112272695e-11, 1.1714888587578576e-9,
          1.7599099927857186e-10, 5.734283990075868e-11, 7.855145138821309e-10,
          1.109109882692946e-9, 4.456991350071304e-10, 1.1055539398977906e-9,
          3.790544357778616e-9, 2.0750030440064227e-9, 1.2070096217342874e-10,
          1.4018002145268373e-9, 6.106531961743963e-10]
    @test isapprox(w11.weights, wt)

    w12 = optimise!(portfolio, Trad(; rm = rm, kelly = EKelly(), obj = obj))
    wt = [2.775098189701407e-9, 3.107340228620143e-9, 3.8612959055128815e-9,
          3.4816222831382753e-9, 0.8533961476837788, 1.2201521175256111e-9,
          0.14660380416690774, 2.115409842210539e-9, 3.3650154898594905e-9,
          2.288480892151623e-9, 2.0327587042036143e-9, 1.265769420446084e-9,
          9.014114382699417e-10, 1.6231741881735537e-9, 9.160741159459735e-10,
          6.291722300786765e-9, 4.323978569824584e-9, 2.22721385821838e-9,
          3.6019745771746776e-9, 2.750821392544799e-9]
    @test isapprox(w12.weights, wt, rtol = 5.0e-6)

    # Risk upper bound
    obj = MaxRet()
    rm.settings.ub = r1
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio; type = :Trad, rm = rm) <= r1 ||
          abs(calc_risk(portfolio; type = :Trad, rm = rm) - r1) < 1e-9

    rm.settings.ub = r2
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio; type = :Trad, rm = rm) <= r2

    rm.settings.ub = r3
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio; type = :Trad, rm = rm) <= r3

    rm.settings.ub = r4
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio; type = :Trad, rm = rm) <= r4

    obj = Sharpe(; rf = rf)
    rm.settings.ub = r1
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio; type = :Trad, rm = rm) <= r1 ||
          abs(calc_risk(portfolio; type = :Trad, rm = rm) - r1) < 1e-10

    rm.settings.ub = r2
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio; type = :Trad, rm = rm) <= r2

    rm.settings.ub = r3
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio; type = :Trad, rm = rm) <= r3

    rm.settings.ub = r4
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio; type = :Trad, rm = rm) <= r4

    # Ret lower bound
    rm.settings.ub = Inf
    obj = MinRisk()
    portfolio.mu_l = ret1
    w13 = optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test dot(portfolio.mu, w13.weights) >= ret1

    portfolio.mu_l = ret2
    w14 = optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test dot(portfolio.mu, w14.weights) >= ret2

    portfolio.mu_l = ret3
    w15 = optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test dot(portfolio.mu, w15.weights) >= ret3

    portfolio.mu_l = ret4
    w16 = optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test dot(portfolio.mu, w16.weights) >= ret4 ||
          abs(dot(portfolio.mu, w16.weights) - ret4) < 1e-10

    obj = Sharpe(; rf = rf)
    portfolio.mu_l = ret1
    w17 = optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test dot(portfolio.mu, w17.weights) >= ret1

    portfolio.mu_l = ret2
    w18 = optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test dot(portfolio.mu, w18.weights) >= ret2

    portfolio.mu_l = ret3
    w19 = optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test dot(portfolio.mu, w19.weights) >= ret3

    portfolio.mu_l = ret4
    w20 = optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test dot(portfolio.mu, w20.weights) >= ret4 ||
          abs(dot(portfolio.mu, w16.weights) - ret4) < 1e-10
end

@testset "SSD" begin
    portfolio = Portfolio(; prices = prices,
                          solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                           :check_sol => (allow_local = true,
                                                                          allow_almost = true),
                                                           :params => Dict("verbose" => false))))
    asset_statistics!(portfolio)
    rm = SSD()

    obj = MinRisk()
    w1 = optimise!(portfolio, Trad(; rm = rm, kelly = NoKelly(), obj = obj))
    r1 = calc_risk(portfolio; type = :Trad, rm = rm)
    ret1 = dot(portfolio.mu, w1.weights)
    wt = [2.954283441277634e-8, 0.04973740293196223, 1.5004834875433086e-8,
          0.002978185203626395, 0.00255077171396876, 0.02013428421720317,
          8.938505323199939e-10, 0.12809490679767346, 2.5514571986823903e-9,
          3.4660313800221236e-9, 0.29957738105080456, 3.6587132183584753e-9,
          1.61047759821642e-9, 0.1206961339634279, 0.012266097184153368,
          0.009663325635394784, 1.859820936315932e-8, 0.22927479857319558,
          3.22169253589993e-9, 0.12502663418048846]
    riskt = 0.005538773213915548
    rett = 0.00031286022410236273
    @test isapprox(w1.weights, wt)
    @test isapprox(r1, riskt)
    @test isapprox(ret1, rett)

    w2 = optimise!(portfolio, Trad(; rm = rm, kelly = AKelly(), obj = obj))
    wt = [1.328560973467273e-8, 0.049738240342198585, 6.729638177718146e-9,
          0.0029785880061378384, 0.002551194236638699, 0.02013119386894698,
          3.7241820112123204e-10, 0.1280950127330832, 1.1193171935929993e-9,
          1.5313119883776913e-9, 0.2995770952417976, 1.6188140270762791e-9,
          6.955937873270665e-10, 0.12069621604817828, 0.012266360319875118,
          0.009662882398733882, 8.34506433718684e-9, 0.22927554413792217,
          1.4212376700688693e-9, 0.12502763754748242]
    @test isapprox(w2.weights, wt)

    w3 = optimise!(portfolio, Trad(; rm = rm, kelly = EKelly(), obj = obj))
    wt = [2.5190879083102043e-8, 0.049737774196137965, 1.2868124743982513e-8,
          0.0029783638332052057, 0.0025509173649045694, 0.020133768670666973,
          9.047615243063875e-10, 0.1280950304816767, 2.3106803351985052e-9,
          3.086425946402379e-9, 0.2995771396012498, 3.251604912185191e-9,
          1.5132089628845487e-9, 0.12069584048889491, 0.012266113001483253,
          0.009662762604571083, 1.5907937864516914e-8, 0.2292749772563436,
          2.8790402254376556e-9, 0.12502724458820227]
    @test isapprox(w3.weights, wt)

    obj = Utility(; l = l)
    w4 = optimise!(portfolio, Trad(; rm = rm, kelly = NoKelly(), obj = obj))
    r2 = calc_risk(portfolio; type = :Trad, rm = rm)
    ret2 = dot(portfolio.mu, w4.weights)
    wt = [2.541867415091494e-9, 0.05530813817602699, 2.41244845892351e-9,
          0.004275662662643133, 0.03390817389617674, 2.317666518349058e-9,
          2.2319148968228334e-10, 0.1252175035167737, 7.110662324391565e-10,
          8.147053974005637e-10, 0.29837139405224006, 3.8477613032148593e-10,
          1.94163891883498e-10, 0.10724362095308528, 0.0027603552142120838,
          0.02112020545265115, 0.005356541418379395, 0.2289888503445687,
          1.076090818037532e-9, 0.11744954363726649]
    riskt = 0.005561860483104124
    rett = 0.00041086155751295247
    @test isapprox(w4.weights, wt)
    @test isapprox(r2, riskt)
    @test isapprox(ret2, rett)

    w5 = optimise!(portfolio, Trad(; rm = rm, kelly = AKelly(), obj = obj))
    wt = [5.310366383671854e-10, 0.055185496756761133, 4.958251729556289e-10,
          0.004440395111615488, 0.033778656762619236, 4.711290484606268e-10,
          1.4995099027754112e-10, 0.12528998427061558, 8.369500406278012e-12,
          2.3638500512437794e-11, 0.2983971162306186, 1.031575927125493e-10,
          1.5846481226439663e-10, 0.10741032813109379, 0.002725697131388786,
          0.021069728462151726, 0.005363035530923987, 0.2288344439497473,
          9.738524268054006e-11, 0.11750511562350689]
    @test isapprox(w5.weights, wt)

    w6 = optimise!(portfolio, Trad(; rm = rm, kelly = EKelly(), obj = obj))
    wt = [5.433753359097147e-9, 0.05518898222258595, 5.182392774858997e-9,
          0.004437472362032297, 0.033766432408213885, 5.009853443063153e-9,
          6.038676037602799e-10, 0.12529115606897243, 1.6077266838664619e-9,
          1.8347984451301988e-9, 0.2984003465672304, 9.359423722016316e-10,
          5.436022541896717e-10, 0.107415956756865, 0.002730640442805002,
          0.021065115761000137, 0.00535251787335741, 0.22884010334158902,
          2.3574094419155423e-9, 0.11751125268600204]
    @test isapprox(w6.weights, wt, rtol = 5.0e-6)

    obj = Sharpe(; rf = rf)
    w7 = optimise!(portfolio, Trad(; rm = rm, kelly = NoKelly(), obj = obj))
    r3 = calc_risk(portfolio; type = :Trad, rm = rm)
    ret3 = dot(portfolio.mu, w7.weights)
    wt = [1.8955665771231532e-8, 4.4148508725600023e-8, 3.537890454211662e-8,
          2.1966271358039556e-8, 0.6666203563586275, 6.130148331498872e-9,
          0.03792018451465443, 3.563315827678111e-8, 4.349162854829938e-8,
          1.8479882644634467e-8, 4.552310886494339e-8, 4.8863225987358126e-9,
          3.315774614641478e-9, 1.2573247089938602e-8, 3.5165001620600556e-9,
          0.1718521246394113, 0.10257058901854942, 4.7654011023485184e-8,
          0.021036366772688796, 3.7042935949165386e-8]
    riskt = 0.00981126385893784
    rett = 0.0015868900032431047
    @test isapprox(w7.weights, wt)
    @test isapprox(r3, riskt)
    @test isapprox(ret3, rett)

    w8 = optimise!(portfolio, Trad(; rm = rm, kelly = AKelly(), obj = obj))
    wt = [1.3339702140673612e-8, 3.5118233468488585e-8, 2.6116387870556337e-8,
          1.615808488462772e-8, 0.6064956769018914, 4.431283062086683e-9,
          0.03599985869128995, 3.301696172838813e-8, 2.6380682145221292e-8,
          1.3318860470718295e-8, 4.536059984210661e-8, 3.49538583445088e-9,
          2.5171281609815222e-9, 8.971379973355786e-9, 2.6175762313284876e-9,
          0.16669074639103687, 0.13807422918691778, 4.381405154981034e-8,
          0.052739184604051505, 2.9568495081235402e-8]
    @test isapprox(w8.weights, wt, rtol = 5.0e-5)

    w9 = optimise!(portfolio, Trad(; rm = rm, kelly = EKelly(), obj = obj))
    wt = [2.3080013239943724e-7, 5.877291398266554e-7, 4.4741431178311566e-7,
          2.8030179924506093e-7, 0.6080716522300086, 7.731283829297768e-8,
          0.03582104852484281, 5.413845554751583e-7, 4.788054759666056e-7,
          2.2931051547170714e-7, 7.43547157424213e-7, 6.176732835770858e-8,
          4.43439456239147e-8, 1.5413584235956952e-7, 4.6104328305770496e-8,
          0.16672109094189144, 0.13724565495175012, 7.191616534200629e-7,
          0.05213541754161184, 4.936908712842138e-7]
    @test isapprox(w9.weights, wt, rtol = 5.0e-8)

    obj = MaxRet()
    w10 = optimise!(portfolio, Trad(; rm = rm, kelly = NoKelly(), obj = obj))
    r4 = calc_risk(portfolio; type = :Trad, rm = rm)
    ret4 = dot(portfolio.mu, w10.weights)
    wt = [1.8748605656928656e-8, 1.8320719487450265e-8, 1.538653450499897e-8,
          1.6989819644791176e-8, 3.3008818023484547e-7, 1.4569564034395335e-8,
          0.9999993779344156, 1.8172395224611005e-8, 1.8057718124898897e-8,
          1.882604735063403e-8, 1.7974701566096796e-8, 1.5301460205927646e-8,
          1.2337854122082459e-8, 1.675271979100736e-8, 1.2582988670103514e-8,
          8.789449903631792e-9, 1.4346521744321203e-8, 1.8465753178372508e-8,
          1.7524551334228423e-8, 1.882999955421456e-8]
    riskt = 0.025704341997146034
    rett = 0.0018453751965893277
    @test isapprox(w10.weights, wt)
    @test isapprox(r4, riskt)
    @test isapprox(ret4, rett)

    w11 = optimise!(portfolio, Trad(; rm = rm, kelly = AKelly(), obj = obj))
    wt = [1.530417939183873e-10, 2.8612308735562203e-10, 5.505656045721553e-10,
          4.5496556945069153e-10, 0.8503475681489141, 3.8305099849033153e-10,
          0.1496524260843989, 7.250692125835878e-11, 3.862370077496164e-10,
          1.7288373304633485e-11, 9.181697560252934e-11, 3.404667655856662e-10,
          3.876849785161851e-10, 2.3474971482803645e-10, 4.1508395195208027e-10,
          6.184903168160772e-10, 7.002854291572307e-10, 3.1063304351626673e-11,
          4.812277872084391e-10, 1.6203868518158558e-10]
    @test isapprox(w11.weights, wt)

    w12 = optimise!(portfolio, Trad(; rm = rm, kelly = EKelly(), obj = obj))
    wt = [3.1790430632771992e-9, 3.566808124524501e-9, 4.438843489656868e-9,
          4.000536222933051e-9, 0.8533944501261447, 1.3808396419652132e-9,
          0.1466054945600572, 2.422388961759021e-9, 3.865692844688943e-9,
          2.626349396091125e-9, 2.3301832079786294e-9, 1.449134769754441e-9,
          1.0301518916040134e-9, 1.8537259113199025e-9, 1.050557173488852e-9,
          7.299687191109068e-9, 4.9744003185052485e-9, 2.5486253915331206e-9,
          4.137759971609911e-9, 3.1590704947489513e-9]
    @test isapprox(w12.weights, wt, rtol = 5.0e-5)

    # Risk upper bound
    obj = MaxRet()
    rm.settings.ub = r1
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio; type = :Trad, rm = rm) <= r1 ||
          abs(calc_risk(portfolio; type = :Trad, rm = rm) - r1) < 1e-9

    rm.settings.ub = r2
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio; type = :Trad, rm = rm) <= r2

    rm.settings.ub = r3
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio; type = :Trad, rm = rm) <= r3

    rm.settings.ub = r4
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio; type = :Trad, rm = rm) <= r4

    obj = Sharpe(; rf = rf)
    rm.settings.ub = r1 * 1.001
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio; type = :Trad, rm = rm) <= r1 * 1.001 ||
          abs(calc_risk(portfolio; type = :Trad, rm = rm) - r1 * 1.001) < 1e-8

    rm.settings.ub = r2
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio; type = :Trad, rm = rm) <= r2

    rm.settings.ub = r3
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio; type = :Trad, rm = rm) <= r3

    rm.settings.ub = r4
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio; type = :Trad, rm = rm) <= r4

    # Ret lower bound
    rm.settings.ub = Inf
    obj = MinRisk()
    portfolio.mu_l = ret1
    w13 = optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test dot(portfolio.mu, w13.weights) >= ret1

    portfolio.mu_l = ret2
    w14 = optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test dot(portfolio.mu, w14.weights) >= ret2

    portfolio.mu_l = ret3
    w15 = optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test dot(portfolio.mu, w15.weights) >= ret3

    portfolio.mu_l = ret4
    w16 = optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test dot(portfolio.mu, w16.weights) >= ret4 ||
          abs(dot(portfolio.mu, w16.weights) - ret4) < 1e-10

    obj = Sharpe(; rf = rf)
    portfolio.mu_l = ret1
    w17 = optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test dot(portfolio.mu, w17.weights) >= ret1

    portfolio.mu_l = ret2
    w18 = optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test dot(portfolio.mu, w18.weights) >= ret2

    portfolio.mu_l = ret3
    w19 = optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test dot(portfolio.mu, w19.weights) >= ret3

    portfolio.mu_l = ret4
    w20 = optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test dot(portfolio.mu, w20.weights) >= ret4 ||
          abs(dot(portfolio.mu, w16.weights) - ret4) < 1e-10
end

@testset "FLPM" begin
    portfolio = Portfolio(; prices = prices,
                          solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                           :check_sol => (allow_local = true,
                                                                          allow_almost = true),
                                                           :params => Dict("verbose" => false))))
    asset_statistics!(portfolio)
    rm = FLPM(; target = rf)

    obj = MinRisk()
    w1 = optimise!(portfolio, Trad(; rm = rm, kelly = NoKelly(), obj = obj))
    r1 = calc_risk(portfolio; type = :Trad, rm = rm)
    ret1 = dot(portfolio.mu, w1.weights)
    wt = [0.004266070614517317, 0.04362165239521167, 0.01996043023729806,
          0.007822722623891595, 0.060525786877357816, 2.187204740032422e-8,
          0.00039587162942815576, 0.13089236100375287, 7.734531969787049e-9,
          0.0118785975269765, 0.2066094523343813, 6.469640939191796e-10,
          6.246750358607508e-10, 0.08329494463798208, 1.6616489736084757e-9,
          0.013888127426323596, 0.0873465246195096, 0.19210093372199202,
          0.03721303157281544, 0.10018346023869455]
    riskt = 0.00265115220934628
    rett = 0.0005443423420749122
    @test isapprox(w1.weights, wt)
    @test isapprox(r1, riskt)
    @test isapprox(ret1, rett)

    w2 = optimise!(portfolio, Trad(; rm = rm, kelly = AKelly(), obj = obj))
    wt = [0.00426610239591103, 0.04362178853662362, 0.019960457154756087,
          0.007822768508828756, 0.06052527699807257, 4.506186395476567e-10,
          0.0003959578648306813, 0.1308926705227917, 1.8600603149618773e-10,
          0.011878924975411324, 0.2066096337189848, 1.5259036251172595e-11,
          1.466852985609556e-11, 0.08329480736550154, 4.0265597843607914e-11,
          0.013888097943043014, 0.08734723725306878, 0.19210036041488635,
          0.037213199462145435, 0.10018271617832634]
    @test isapprox(w2.weights, wt)

    w3 = optimise!(portfolio, Trad(; rm = rm, kelly = EKelly(), obj = obj))
    wt = [0.00426608716920985, 0.04362173641258839, 0.019960444305697746,
          0.007822767144185885, 0.06052541421131514, 3.4024764351128286e-9,
          0.00039592869717005593, 0.13089257106535185, 1.263061291396848e-9,
          0.0118788599251899, 0.20660956985823367, 1.2640723567497645e-10,
          1.224400506824836e-10, 0.08329482612251998, 2.9820592088268585e-10,
          0.013888098156328821, 0.08734710673926958, 0.19210049684132757,
          0.037213161399705486, 0.1001829267393152]
    @test isapprox(w3.weights, wt)

    obj = Utility(; l = l)
    w4 = optimise!(portfolio, Trad(; rm = rm, kelly = NoKelly(), obj = obj))
    r2 = calc_risk(portfolio; type = :Trad, rm = rm)
    ret2 = dot(portfolio.mu, w4.weights)
    wt = [5.651779189600572e-9, 0.042138947590740154, 0.012527625908380429,
          0.00720584542391769, 0.09868086575102923, 4.666011527039072e-10,
          0.002772974287825817, 0.10921142006880304, 1.453665644745307e-9,
          4.587434723327683e-9, 0.20154713458183793, 1.034055452971105e-10,
          1.1243437198410302e-10, 0.026542856504844895, 2.1611055686908496e-10,
          0.041731161486272504, 0.11080106522064202, 0.17232533003113762,
          0.06800096884126186, 0.10651379171187556]
    riskt = 0.0026842735895541213
    rett = 0.0006732529128667895
    @test isapprox(w4.weights, wt)
    @test isapprox(r2, riskt)
    @test isapprox(ret2, rett)

    w5 = optimise!(portfolio, Trad(; rm = rm, kelly = AKelly(), obj = obj))
    wt = [7.025580903339914e-9, 0.03801909891801268, 0.015508772801692194,
          0.007920175117119383, 0.09985335454405923, 2.5401009772523425e-10,
          0.0024873633006595353, 0.10503236887549107, 8.540598148869465e-10,
          4.653369605256221e-9, 0.203007892809087, 7.000508823160763e-10,
          6.910321683931141e-10, 0.035031103036583196, 5.676257759160685e-10,
          0.03798687241385029, 0.10539179463937084, 0.1777681655105699, 0.06295784178579301,
          0.10903518150198238]
    @test isapprox(w5.weights, wt)

    w6 = optimise!(portfolio, Trad(; rm = rm, kelly = EKelly(), obj = obj))
    wt = [1.2872334765954068e-8, 0.03767953016874954, 0.015470385826160825,
          0.0081170648205414, 0.09892766032152121, 1.0417936519167395e-9,
          0.0025348401228208013, 0.10487955021910977, 2.8917086673976536e-9,
          8.851185748932032e-9, 0.20310090288185126, 3.0551749498443944e-10,
          3.161080085781991e-10, 0.03555872480929, 5.151922419610025e-10,
          0.03730024048937033, 0.10541030477402333, 0.17862267733167023,
          0.06248696950024344, 0.10991112194080724]
    @test isapprox(w6.weights, wt, rtol = 5.0e-5)

    obj = Sharpe(; rf = rf)
    w7 = optimise!(portfolio, Trad(; rm = rm, kelly = NoKelly(), obj = obj))
    r3 = calc_risk(portfolio; type = :Trad, rm = rm)
    ret3 = dot(portfolio.mu, w7.weights)
    wt = [5.791704589818663e-10, 1.4777512342996448e-9, 1.4920733133812998e-9,
          8.941347428424144e-10, 0.6999099125632519, 2.145377355161713e-10,
          0.029295630576512924, 1.1027104693788755e-9, 1.8864271969797675e-9,
          8.43330450121613e-10, 1.4937081011622384e-9, 1.4856958187000145e-10,
          1.0768233412852032e-10, 3.8855123608537257e-10, 1.2149887816181597e-10,
          0.15181164107816766, 0.04226710946215913, 1.3947899372714116e-9,
          0.07671569251341252, 1.6615602330924226e-9]
    riskt = 0.00431255671125957
    rett = 0.0015948388159746803
    @test isapprox(w7.weights, wt)
    @test isapprox(r3, riskt)
    @test isapprox(ret3, rett)

    w8 = optimise!(portfolio, Trad(; rm = rm, kelly = AKelly(), obj = obj))
    wt = [1.8429959334822882e-11, 4.579361301517624e-11, 4.772355702946762e-11,
          2.6650970992935517e-11, 0.6058883405722568, 6.314300546560684e-12,
          0.028234427676223084, 4.0188462017382376e-11, 4.7607659743264474e-11,
          2.1544747364296088e-11, 6.674241399057538e-11, 4.351442761481561e-12,
          3.299167356187561e-12, 1.1413908220300238e-11, 3.5811703032948647e-12,
          0.14977766642855195, 0.10888035658229393, 5.643682502844824e-11,
          0.10721920828369275, 5.6903389356087466e-11]
    @test isapprox(w8.weights, wt)

    w9 = optimise!(portfolio, Trad(; rm = rm, kelly = EKelly(), obj = obj))
    wt = [5.833714539784639e-9, 1.4477767592579734e-8, 1.6175278895116947e-8,
          8.533045693838208e-9, 0.6066263690850154, 1.984257444670923e-9,
          0.028290380035531127, 1.2664384171896574e-8, 1.5359011055744673e-8,
          7.119832059573429e-9, 1.9623395261372662e-8, 1.3445768599438299e-9,
          1.0210352375021523e-9, 3.589684135816939e-9, 1.1156509645967702e-9,
          0.14977632640507282, 0.10820593602438255, 1.827062842472936e-8,
          0.10710084398310481, 1.735463099578888e-8]
    @test isapprox(w9.weights, wt, rtol = 1.0e-4)

    obj = MaxRet()
    w10 = optimise!(portfolio, Trad(; rm = rm, kelly = NoKelly(), obj = obj))
    r4 = calc_risk(portfolio; type = :Trad, rm = rm)
    ret4 = dot(portfolio.mu, w10.weights)
    wt = [1.3019302358014325e-8, 1.3979706940965055e-8, 1.670240049189312e-8,
          1.6104356899062238e-8, 2.5495489803481445e-7, 1.0170483295381582e-8,
          0.9999995050048601, 1.1436904494135492e-8, 1.492977755510578e-8,
          1.212324914648131e-8, 1.1410343061057653e-8, 1.0141009010483304e-8,
          9.120987910382183e-9, 1.0868081833333878e-8, 9.494553530172169e-9,
          2.3389178339261675e-8, 1.777099881408979e-8, 1.1725146345613458e-8,
          1.5069511913545612e-8, 1.2584249870605286e-8]
    riskt = 0.012237371871856062
    rett = 0.001845375304551891
    @test isapprox(w10.weights, wt)
    @test isapprox(r4, riskt)
    @test isapprox(ret4, rett)

    w11 = optimise!(portfolio, Trad(; rm = rm, kelly = AKelly(), obj = obj))
    wt = [6.184818023388107e-10, 9.270209690896355e-10, 1.625727765129299e-9,
          1.2702031382871368e-9, 0.8503448667005368, 8.234230282459137e-10,
          0.14965511529944078, 1.8812024811068833e-11, 1.1594526094497495e-9,
          1.7436144865199502e-10, 5.659330075801571e-11, 7.768388767333282e-10,
          1.0972252733499826e-9, 4.4072771043489173e-10, 1.0935450396733876e-9,
          3.753052447991426e-9, 2.05335213852063e-9, 1.1963791294032173e-10,
          1.3871915795314854e-9, 6.043753072340663e-10]
    @test isapprox(w11.weights, wt)

    w12 = optimise!(portfolio, Trad(; rm = rm, kelly = EKelly(), obj = obj))
    wt = [3.4833527152512497e-9, 3.906326367133753e-9, 4.849219223866047e-9,
          4.376719460303863e-9, 0.8533925847628662, 1.5383414338773639e-9,
          0.14660735464542013, 2.6635923145571535e-9, 4.235728108730784e-9,
          2.8833481317072016e-9, 2.560891985886492e-9, 1.5963183961895322e-9,
          1.1374687562546555e-9, 2.045290779367009e-9, 1.1563631712747345e-9,
          7.91550317404802e-9, 5.439211345857847e-9, 2.8038212912536403e-9,
          4.5367310838932605e-9, 3.463485923768432e-9]
    @test isapprox(w12.weights, wt, rtol = 5.0e-6)

    # Risk upper bound
    obj = MaxRet()
    rm.settings.ub = r1
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio; type = :Trad, rm = rm) <= r1 ||
          abs(calc_risk(portfolio; type = :Trad, rm = rm) - r1) < 1e-9

    rm.settings.ub = r2
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio; type = :Trad, rm = rm) <= r2

    rm.settings.ub = r3
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio; type = :Trad, rm = rm) <= r3

    rm.settings.ub = r4
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio; type = :Trad, rm = rm) <= r4

    obj = Sharpe(; rf = rf)
    rm.settings.ub = r1
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio; type = :Trad, rm = rm) <= r1 ||
          abs(calc_risk(portfolio; type = :Trad, rm = rm) - r1) < 1e-10

    rm.settings.ub = r2
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio; type = :Trad, rm = rm) <= r2

    rm.settings.ub = r3
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio; type = :Trad, rm = rm) <= r3

    rm.settings.ub = r4
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio; type = :Trad, rm = rm) <= r4

    # Ret lower bound
    rm.settings.ub = Inf
    obj = MinRisk()
    portfolio.mu_l = ret1
    w13 = optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test dot(portfolio.mu, w13.weights) >= ret1

    portfolio.mu_l = ret2
    w14 = optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test dot(portfolio.mu, w14.weights) >= ret2

    portfolio.mu_l = ret3
    w15 = optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test dot(portfolio.mu, w15.weights) >= ret3

    portfolio.mu_l = ret4
    w16 = optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test dot(portfolio.mu, w16.weights) >= ret4 ||
          abs(dot(portfolio.mu, w16.weights) - ret4) < 1e-10

    obj = Sharpe(; rf = rf)
    portfolio.mu_l = ret1
    w17 = optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test dot(portfolio.mu, w17.weights) >= ret1

    portfolio.mu_l = ret2
    w18 = optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test dot(portfolio.mu, w18.weights) >= ret2

    portfolio.mu_l = ret3
    w19 = optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test dot(portfolio.mu, w19.weights) >= ret3

    portfolio.mu_l = ret4
    w20 = optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test dot(portfolio.mu, w20.weights) >= ret4 ||
          abs(dot(portfolio.mu, w16.weights) - ret4) < 1e-10
end

@testset "SLPM" begin
    portfolio = Portfolio(; prices = prices,
                          solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                           :check_sol => (allow_local = true,
                                                                          allow_almost = true),
                                                           :params => Dict("verbose" => false))))
    asset_statistics!(portfolio)
    rm = SLPM(; target = rf)

    obj = MinRisk()
    w1 = optimise!(portfolio, Trad(; rm = rm, kelly = NoKelly(), obj = obj))
    r1 = calc_risk(portfolio; type = :Trad, rm = rm)
    ret1 = dot(portfolio.mu, w1.weights)
    wt = [1.1230805069911956e-8, 0.05524472463362737, 1.0686041548307544e-8,
          0.0043185378999763225, 0.033597348034736865, 1.0487157577222361e-8,
          1.1738886913269633e-9, 0.12478148562530009, 3.4647395424618816e-9,
          3.8805677196069256e-9, 0.3005648369145803, 2.0034183913036616e-9,
          1.0927362747553375e-9, 0.10661826438516031, 0.003123732919975542,
          0.021391817407374183, 0.003595424842043441, 0.22964898912299475,
          5.129978967782806e-9, 0.117114789064897]
    riskt = 0.005418882634929856
    rett = 0.0004088880259308715
    @test isapprox(w1.weights, wt)
    @test isapprox(r1, riskt)
    @test isapprox(ret1, rett)

    w2 = optimise!(portfolio, Trad(; rm = rm, kelly = AKelly(), obj = obj))
    wt = [3.553751623019336e-9, 0.0552444506714033, 3.3792972175226587e-9,
          0.004319267700904996, 0.033597486252238504, 3.3178770246587074e-9,
          3.530263895833413e-10, 0.12478100673912003, 1.082173144802101e-9,
          1.2142739606639318e-9, 0.3005650138904127, 6.172236686305243e-10,
          3.272670368272328e-10, 0.10661801005424905, 0.003123621247828574,
          0.02139180763720946, 0.0035949221271092354, 0.22965076223179196,
          1.6119446394023392e-9, 0.11711363599089746]
    @test isapprox(w2.weights, wt)

    w3 = optimise!(portfolio, Trad(; rm = rm, kelly = EKelly(), obj = obj))
    wt = [8.960032986198268e-9, 0.05524502934743527, 8.531490753416422e-9,
          0.004319426319869003, 0.03359614618594921, 8.379240639976445e-9,
          1.0539120789014738e-9, 0.12478035258171694, 2.855174321889051e-9,
          3.1807196110268267e-9, 0.30056524386869504, 1.706724783203312e-9,
          9.907199378298648e-10, 0.10661840631750075, 0.003124219404851094,
          0.021391583878468307, 0.0035949097819517437, 0.22965048214875944,
          4.163566475384283e-9, 0.11711416034322167]
    @test isapprox(w3.weights, wt)

    obj = Utility(; l = l)
    w4 = optimise!(portfolio, Trad(; rm = rm, kelly = NoKelly(), obj = obj))
    r2 = calc_risk(portfolio; type = :Trad, rm = rm)
    ret2 = dot(portfolio.mu, w4.weights)
    wt = [1.892614163740768e-9, 0.05510469378866011, 2.336572603808684e-9,
          0.0004235261366778282, 0.06590298875275914, 9.60649390058247e-10,
          4.495790137196085e-10, 0.11928868236657098, 1.3194622743579184e-9,
          1.2532033673577644e-9, 0.2937381531255968, 3.557016906443741e-10,
          1.75711616979717e-10, 0.07604922161635236, 1.3912072068726968e-9,
          0.03452489138234348, 0.02823700959313513, 0.22320354704027148,
          2.8171948818718627e-9, 0.10352727324573653]
    riskt = 0.005439630405063694
    rett = 0.0004936947590309835
    @test isapprox(w4.weights, wt)
    @test isapprox(r2, riskt)
    @test isapprox(ret2, rett)

    w5 = optimise!(portfolio, Trad(; rm = rm, kelly = AKelly(), obj = obj))
    wt = [1.1179343918984971e-10, 0.054976790588896596, 1.7713391003665344e-10,
          0.000633725765454258, 0.06550332883993616, 2.3043194331626305e-11,
          9.784822751076497e-11, 0.11941492846830355, 2.6615472963293778e-11,
          1.8716678076476947e-11, 0.2937827494968045, 1.1133950578913522e-10,
          1.37189305281988e-10, 0.0764029089634452, 3.804971672707788e-11,
          0.03435573176652704, 0.028126826103343827, 0.2230959623916138,
          2.3934818713608943e-10, 0.10370704663459744]
    @test isapprox(w5.weights, wt)

    w6 = optimise!(portfolio, Trad(; rm = rm, kelly = EKelly(), obj = obj))
    wt = [1.909018619028115e-9, 0.05498057942444095, 2.339665296590162e-9,
          0.000631953942326558, 0.06548845319058122, 1.0190004075879495e-9,
          5.259478961923537e-10, 0.11941671482755227, 1.3463607876661425e-9,
          1.2948590823945186e-9, 0.29378582770159867, 4.3653724439217223e-10,
          2.6599708561332144e-10, 0.07641754715464191, 1.4214351227433604e-9,
          0.03435049194054096, 0.028114027913869674, 0.2231003981170177,
          2.7492193028254847e-9, 0.10371399247938931]
    @test isapprox(w6.weights, wt, rtol = 5.0e-8)

    obj = Sharpe(; rf = rf)
    w7 = optimise!(portfolio, Trad(; rm = rm, kelly = NoKelly(), obj = obj))
    r3 = calc_risk(portfolio; type = :Trad, rm = rm)
    ret3 = dot(portfolio.mu, w7.weights)
    wt = [1.9163296529161612e-8, 4.442773083721428e-8, 3.487971564859802e-8,
          2.1724751159534335e-8, 0.6654321579931172, 6.209329060743242e-9,
          0.03807260797466121, 3.6518531572866855e-8, 4.3161980618677364e-8,
          1.8351785923597892e-8, 4.6197788418506723e-8, 5.02002507806016e-9,
          3.3979241671533133e-9, 1.283560044201211e-8, 3.5855450291876262e-9,
          0.17459230136257337, 0.10412390410574306, 4.8448629929346825e-8,
          0.017778647586382573, 3.705488830957971e-8]
    riskt = 0.00909392522496688
    rett = 0.0015869580721210722
    @test isapprox(w7.weights, wt)
    @test isapprox(r3, riskt)
    @test isapprox(ret3, rett)

    w8 = optimise!(portfolio, Trad(; rm = rm, kelly = AKelly(), obj = obj))
    wt = [7.1313222696196286e-9, 1.8556253618894194e-8, 1.3523168143938587e-8,
          8.449599974284862e-9, 0.610572312454996, 2.3820744436346198e-9,
          0.036338308826069636, 1.7584912138118536e-8, 1.4023893271882696e-8,
          7.017765982042944e-9, 2.3612750200409217e-8, 1.9058395740420574e-9,
          1.3684166095297231e-9, 4.846459856703243e-9, 1.4176608092570215e-9,
          0.16949805603533355, 0.13682289651084628, 2.2953669955066264e-8,
          0.046768265951569875, 1.544739788558272e-8]
    @test isapprox(w8.weights, wt, rtol = 5.0e-7)

    w9 = optimise!(portfolio, Trad(; rm = rm, kelly = EKelly(), obj = obj))
    wt = [2.9562281849363614e-7, 7.203175143415471e-7, 5.527354915875717e-7,
          3.562290393419334e-7, 0.6120717582322253, 1.0106091093835403e-7,
          0.036164963256371814, 6.730610166304067e-7, 5.529573171292939e-7,
          2.821980090718655e-7, 9.319291317992295e-7, 8.419509011177982e-8,
          6.018370176805596e-8, 1.9488520525642497e-7, 6.199288625890055e-8,
          0.16951003917104113, 0.13598079153678486, 8.914833982446512e-7,
          0.04626609570920224, 5.932428436544369e-7]
    @test isapprox(w9.weights, wt, rtol = 5.0e-8)

    obj = MaxRet()
    w10 = optimise!(portfolio, Trad(; rm = rm, kelly = NoKelly(), obj = obj))
    r4 = calc_risk(portfolio; type = :Trad, rm = rm)
    ret4 = dot(portfolio.mu, w10.weights)
    wt = [8.874552417228264e-9, 1.0569760898904627e-8, 2.0363934073123463e-8,
          1.547715765009297e-8, 4.267804159375552e-7, 2.8607270157041685e-9,
          0.9999993860106617, 4.216763165728999e-9, 1.2576645584153888e-8,
          5.097663390006654e-9, 3.971251769425353e-9, 2.9266677130332086e-9,
          2.874302327436276e-9, 3.1933824668323967e-9, 2.8301314578492617e-9,
          4.222188374473967e-8, 2.2946307691638603e-8, 4.6341884676924975e-9,
          1.3929168448114664e-8, 7.644434060912258e-9]
    riskt = 0.024842158070968706
    rett = 0.0018453754259793952
    @test isapprox(w10.weights, wt)
    @test isapprox(r4, riskt)
    @test isapprox(ret4, rett)

    w11 = optimise!(portfolio, Trad(; rm = rm, kelly = AKelly(), obj = obj))
    wt = [7.210116063134277e-11, 1.176606228072633e-10, 2.1258483097832237e-10,
          1.6536274651174382e-10, 0.8503141905762418, 1.273544562974361e-10,
          0.14968580705048062, 1.992517244424829e-12, 1.5128034960181728e-10,
          1.850938653797299e-11, 1.2078508927108743e-11, 1.1804552928693339e-10,
          1.5680771674497215e-10, 6.957317957422098e-11, 1.6101143219670432e-10,
          4.425491709419713e-10, 2.7404957392346757e-10, 9.559038664076399e-12,
          1.8545492666656794e-10, 7.730240246200862e-11]
    @test isapprox(w11.weights, wt)

    w12 = optimise!(portfolio, Trad(; rm = rm, kelly = EKelly(), obj = obj))
    wt = [3.7036695360161674e-9, 4.1472927479416145e-9, 5.15386195452294e-9,
          4.6470329264037e-9, 0.8533935407333886, 1.6277480467467953e-9,
          0.14660639500167905, 2.8231597428747422e-9, 4.491287116935e-9,
          3.054355737304291e-9, 2.7129571744811642e-9, 1.689249615135807e-9,
          1.202919191281741e-9, 2.1660529624022714e-9, 1.222630516718091e-9,
          8.399883420534573e-9, 5.771484370394195e-9, 2.972308021065827e-9,
          4.807518056050933e-9, 3.6715213560850315e-9]
    @test isapprox(w12.weights, wt, rtol = 5.0e-6)

    # Risk upper bound
    obj = MaxRet()
    rm.settings.ub = r1
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio; type = :Trad, rm = rm) <= r1 ||
          abs(calc_risk(portfolio; type = :Trad, rm = rm) - r1) < 1e-9

    rm.settings.ub = r2
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio; type = :Trad, rm = rm) <= r2

    rm.settings.ub = r3
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio; type = :Trad, rm = rm) <= r3

    rm.settings.ub = r4
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio; type = :Trad, rm = rm) <= r4

    obj = Sharpe(; rf = rf)
    rm.settings.ub = r1 * (1.000001)
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio; type = :Trad, rm = rm) <= r1 ||
          abs(calc_risk(portfolio; type = :Trad, rm = rm) - r1) < 1e-8

    rm.settings.ub = r2
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio; type = :Trad, rm = rm) <= r2

    rm.settings.ub = r3
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio; type = :Trad, rm = rm) <= r3

    rm.settings.ub = r4
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio; type = :Trad, rm = rm) <= r4

    # Ret lower bound
    rm.settings.ub = Inf
    obj = MinRisk()
    portfolio.mu_l = ret1
    w13 = optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test dot(portfolio.mu, w13.weights) >= ret1

    portfolio.mu_l = ret2
    w14 = optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test dot(portfolio.mu, w14.weights) >= ret2

    portfolio.mu_l = ret3
    w15 = optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test dot(portfolio.mu, w15.weights) >= ret3

    portfolio.mu_l = ret4
    w16 = optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test dot(portfolio.mu, w16.weights) >= ret4 ||
          abs(dot(portfolio.mu, w16.weights) - ret4) < 1e-10

    obj = Sharpe(; rf = rf)
    portfolio.mu_l = ret1
    w17 = optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test dot(portfolio.mu, w17.weights) >= ret1

    portfolio.mu_l = ret2
    w18 = optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test dot(portfolio.mu, w18.weights) >= ret2

    portfolio.mu_l = ret3
    w19 = optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test dot(portfolio.mu, w19.weights) >= ret3

    portfolio.mu_l = ret4
    w20 = optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test dot(portfolio.mu, w20.weights) >= ret4 ||
          abs(dot(portfolio.mu, w16.weights) - ret4) < 1e-10
end
