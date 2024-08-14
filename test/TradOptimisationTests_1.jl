@testset "SD" begin
    portfolio = Portfolio2(; prices = prices,
                           solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                            :params => Dict("verbose" => false))))
    asset_statistics2!(portfolio)
    rm = SD2()

    obj = MinRisk()
    w1 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r1 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret1 = dot(portfolio.mu, w1.weights)
    wt = [0.0078093616583851146, 0.030698578999046943, 0.010528316771324561,
          0.027486806578381814, 0.012309038313357737, 0.03341430871186881,
          1.1079055085166888e-7, 0.13985416268183082, 2.0809271230580642e-7,
          6.32554715643476e-6, 0.28784123553791136, 1.268075347971748e-7,
          8.867081236591187e-8, 0.12527141708492384, 3.264070667606171e-7,
          0.015079837844627948, 1.5891383112101438e-5, 0.1931406057562605,
          2.654124109083291e-7, 0.11654298695072397]
    riskt = 0.007704593409157056
    rett = 0.0003482663810696356
    @test isapprox(w1.weights, wt)
    @test isapprox(r1, riskt)
    @test isapprox(ret1, rett)

    w2 = optimise2!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [0.0078093616583851146, 0.030698578999046943, 0.010528316771324561,
          0.027486806578381814, 0.012309038313357737, 0.03341430871186881,
          1.1079055085166888e-7, 0.13985416268183082, 2.0809271230580642e-7,
          6.32554715643476e-6, 0.28784123553791136, 1.268075347971748e-7,
          8.867081236591187e-8, 0.12527141708492384, 3.264070667606171e-7,
          0.015079837844627948, 1.5891383112101438e-5, 0.1931406057562605,
          2.654124109083291e-7, 0.11654298695072397]
    @test isapprox(w2.weights, wt)

    w3 = optimise2!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    wt = [0.007893834004010548, 0.030693397218184384, 0.01050943911162566,
          0.027487590678529683, 0.0122836015907984, 0.03341312720689581,
          2.654460293680794e-8, 0.13984817931920596, 4.861252776141605e-8,
          3.309876672531783e-7, 0.2878217133212084, 3.116270780466401e-8,
          2.2390519612760115e-8, 0.12528318523437287, 9.334010636386356e-8,
          0.015085761770714442, 7.170234777802365e-7, 0.19312554465008394,
          6.179100704970626e-8, 0.11655329404175341]
    @test isapprox(w3.weights, wt)

    obj = Util(; l = l)
    w4 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r2 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret2 = dot(portfolio.mu, w4.weights)
    wt = [1.065553054256496e-9, 1.906979877393637e-9, 2.1679869440360567e-9,
          1.70123972526289e-9, 0.7741855142171694, 3.9721744242294547e-10,
          0.10998135534654405, 1.3730517031876334e-9, 1.5832262577152926e-9,
          1.0504881447825781e-9, 1.2669287896045939e-9, 4.038975120701348e-10,
          6.074001448526581e-10, 2.654358762537183e-10, 6.574536682273354e-10,
          0.1158331072870088, 3.0452991740231055e-9, 1.3663094482455795e-9,
          2.4334674474942e-9, 1.8573424305703526e-9]
    riskt = 0.01609460480445889
    rett = 0.0017268228943243054
    @test isapprox(w4.weights, wt)
    @test isapprox(r2, riskt)
    @test isapprox(ret2, rett)

    w5 = optimise2!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [8.870505348946403e-9, 2.0785636066552727e-8, 2.798137197308657e-8,
          1.971091339190289e-8, 0.7185881405281792, 4.806880649144299e-10,
          0.09860828964210354, 1.1235098321720224e-8, 2.977172777854582e-8,
          8.912749778026878e-9, 9.63062128166912e-9, 1.0360544993920464e-9,
          2.180352541614548e-9, 2.689800139816139e-9, 2.3063944199708073e-9,
          0.15518499560246005, 0.027618271886178034, 1.246121371211767e-8,
          1.2842725621709964e-7, 1.586069567397408e-8]
    @test isapprox(w5.weights, wt)

    w6 = optimise2!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    wt = [4.176019357134568e-9, 8.185339020442324e-9, 1.0737258331407902e-8,
          7.901762479846926e-9, 0.7207182132289714, 1.1540211071681835e-9,
          0.09835523884681871, 4.849354486370478e-9, 1.1755943684787842e-8,
          4.185213130955141e-9, 4.314456480234504e-9, 9.722540074689786e-10,
          5.895848876745837e-10, 2.1334187374036406e-9, 5.661079854916932e-10,
          0.15505269633577082, 0.025873730012408114, 5.286907987963515e-9,
          4.83569668574532e-8, 6.41142230104617e-9]
    @test isapprox(w6.weights, wt, rtol = 1.0e-6)

    obj = SR(; rf = rf)
    w7 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r3 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret3 = dot(portfolio.mu, w7.weights)
    wt = [3.435681413150756e-10, 9.269743242817558e-10, 1.174114891347931e-9,
          7.447407606568295e-10, 0.5180552669298059, 1.08151797431462e-10,
          0.0636508036260703, 7.872264113421062e-10, 7.841830201959634e-10,
          3.9005509625957585e-10, 6.479557895235057e-10, 8.472023236127232e-11,
          5.766670106753152e-11, 1.988136246095318e-10, 5.935811276550078e-11,
          0.14326634942881586, 0.1964867973307653, 7.554937254824565e-10,
          0.0785407748474901, 7.740298948228655e-10]
    riskt = 0.013160876658207102
    rett = 0.0014788430765515807
    @test isapprox(w7.weights, wt)
    @test isapprox(r3, riskt)
    @test isapprox(ret3, rett)

    w8 = optimise2!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [1.0672947772539922e-8, 3.6603580846259566e-8, 4.190924498057212e-8,
          2.579795624783031e-8, 0.45247454726503317, 3.139203265461306e-9,
          0.05198581042386962, 1.1201379704294516e-7, 1.799097939748088e-8,
          1.2844577033392204e-8, 5.1484053193477936e-8, 2.3241091705338425e-9,
          1.699312214555245e-9, 6.26319015273334e-9, 1.6636900367102399e-9,
          0.13648649205020114, 0.2350741185365231, 4.844604537439258e-8,
          0.12397862173181687, 3.713986941437853e-8]
    @test isapprox(w8.weights, wt, rtol = 1.0e-7)

    w9 = optimise2!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    wt = [1.6553521389413233e-7, 5.275915958538328e-7, 6.846007363199405e-7,
          3.914933469758698e-7, 0.48926168709100504, 4.9332881037102406e-8,
          0.0583064644410985, 5.594366962947531e-7, 3.1357711474708337e-7,
          1.895896838004368e-7, 4.1299427275337544e-7, 3.811276445091462e-8,
          2.7731552876975723e-8, 9.393138539482288e-8, 2.6831018704067043e-8,
          0.1402408063077745, 0.2134138585246757, 4.713662104500069e-7, 0.09877278790316771,
          4.4360780483006885e-7]
    @test isapprox(w9.weights, wt, rtol = 1e-2)

    obj = MaxRet()
    w10 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r4 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret4 = dot(portfolio.mu, w10.weights)
    wt = [8.719239065561217e-10, 8.89979881250899e-10, 9.962391574545242e-10,
          9.646258079031587e-10, 8.401034275877383e-9, 4.828906105645332e-10,
          0.9999999782316086, 6.87727997367879e-10, 9.021263551270326e-10,
          7.350693996493505e-10, 6.753002461228969e-10, 5.009649350579108e-10,
          3.7428368039997965e-10, 5.884337547691459e-10, 3.9326986484718143e-10,
          8.842556785821523e-10, 9.784139669171374e-10, 7.146277206720297e-10,
          9.044289592659792e-10, 8.2279511460579e-10]
    riskt = 0.040597851628968784
    rett = 0.0018453756308089402
    @test isapprox(w10.weights, wt)
    @test isapprox(r4, riskt)
    @test isapprox(ret4, rett)

    w11 = optimise2!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [3.997304959875111e-10, 6.096555731047457e-10, 1.0828360159599854e-9,
          8.390684135793784e-10, 0.8503431998881756, 5.837264433583887e-10,
          0.14965678808511193, 8.520932897976487e-13, 7.66160958682953e-10,
          1.0247860261071675e-10, 5.1627700971086255e-11, 5.483183958203547e-10,
          7.565204185674542e-10, 3.16106264753721e-10, 7.638502459889708e-10,
          2.447496129413098e-9, 1.372927322256315e-9, 6.541563185875491e-11,
          9.248420166125226e-10, 3.9509971643490626e-10]
    @test isapprox(w11.weights, wt)

    w12 = optimise2!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    wt = [1.2461220626745066e-9, 1.3954805691188813e-9, 1.7339340756162538e-9,
          1.5635882299558742e-9, 0.853395149768853, 5.48096579085184e-10,
          0.14660482860584584, 9.501446622854747e-10, 1.5113651288469943e-9,
          1.027931406345638e-9, 9.130613494698656e-10, 5.686010690200261e-10,
          4.0494468011345616e-10, 7.290999439594515e-10, 4.1154424470964885e-10,
          2.82566220199723e-9, 1.9419703441146337e-9, 1.0003454025331967e-9,
          1.6178718912419106e-9, 1.2355373241783204e-9]
    @test isapprox(w12.weights, wt, rtol = 1.0e-5)

    # Risk upper bound
    obj = MaxRet()
    rm.settings.ub = r1
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r1 ||
          abs(calc_risk(portfolio; type = :Trad2, rm = rm) - r1) < 1e-9

    rm.settings.ub = r2
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r2

    rm.settings.ub = r3
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r3

    rm.settings.ub = r4
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r4

    obj = SR(; rf = rf)
    rm.settings.ub = r1
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r1 ||
          abs(calc_risk(portfolio; type = :Trad2, rm = rm) - r1) < 1e-10

    rm.settings.ub = r2
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r2

    rm.settings.ub = r3
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r3

    rm.settings.ub = r4
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r4

    # Ret lower bound
    rm.settings.ub = Inf
    obj = MinRisk()
    portfolio.mu_l = ret1
    w13 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w13.weights) >= ret1

    portfolio.mu_l = ret2
    w14 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w14.weights) >= ret2

    portfolio.mu_l = ret3
    w15 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w15.weights) >= ret3

    portfolio.mu_l = ret4
    w16 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w16.weights) >= ret4

    obj = SR(; rf = rf)
    portfolio.mu_l = ret1
    w17 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w17.weights) >= ret1

    portfolio.mu_l = ret2
    w18 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w18.weights) >= ret2

    portfolio.mu_l = ret3
    w19 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w19.weights) >= ret3

    portfolio.mu_l = ret4
    w20 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w20.weights) >= ret4
end

@testset "SD formulations" begin
    portfolio = Portfolio2(; prices = prices,
                           solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                            :params => Dict("verbose" => false))))
    asset_statistics2!(portfolio)

    obj = MinRisk()
    rm = SD2(; formulation = SOCSD())
    w1 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r1 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret1 = dot(portfolio.mu, w1.weights)
    wt = [0.0078093616583851146, 0.030698578999046943, 0.010528316771324561,
          0.027486806578381814, 0.012309038313357737, 0.03341430871186881,
          1.1079055085166888e-7, 0.13985416268183082, 2.0809271230580642e-7,
          6.32554715643476e-6, 0.28784123553791136, 1.268075347971748e-7,
          8.867081236591187e-8, 0.12527141708492384, 3.264070667606171e-7,
          0.015079837844627948, 1.5891383112101438e-5, 0.1931406057562605,
          2.654124109083291e-7, 0.11654298695072397]
    riskt = 0.007704593409157056
    rett = 0.0003482663810696356
    @test isapprox(w1.weights, wt)
    @test isapprox(r1, riskt)
    @test isapprox(ret1, rett)

    rm = SD2(; formulation = QuadSD())
    w2 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r2 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret2 = dot(portfolio.mu, w2.weights)
    wt = [0.007931704468101724, 0.030580315769072094, 0.010456593899158567,
          0.02742093947332831, 0.012239100149899967, 0.033302579036721264,
          4.162859647692109e-6, 0.13981730989327237, 5.788675176975045e-6,
          0.00033179125802216067, 0.2877731391009407, 3.3160181115934523e-6,
          2.9562651082726095e-6, 0.12512918292552744, 3.227218484382772e-5,
          0.015034147440717766, 0.0005108903689970619, 0.19299166064346063,
          6.739278073914423e-6, 0.1164254102918178]
    riskt = 0.007704683283635234
    rett = 0.0003484772957329131
    @test isapprox(w2.weights, wt)
    @test isapprox(r2, riskt)
    @test isapprox(ret2, rett)

    rm = SD2(; formulation = SimpleSD())
    w3 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r3 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret3 = dot(portfolio.mu, w3.weights)
    wt = [0.007892956618378178, 0.03069135078310372, 0.010510053328054379,
          0.02748716488934741, 0.012284726286598107, 0.033412747896745434,
          3.4510983561959808e-9, 0.1398498186599616, 6.463477389090647e-9,
          1.2183912701620326e-7, 0.28783457711461696, 3.971402286172399e-9,
          2.9059474605121267e-9, 0.12527565747300592, 1.1534530258372196e-8,
          0.015085797229230475, 3.1071292428662565e-7, 0.19312938283717607,
          8.183117329711524e-9, 0.11654529782215743]
    riskt = 0.007704591083785214
    rett = 0.00034824217501046187
    @test isapprox(w3.weights, wt)
    @test isapprox(r3, riskt)
    @test isapprox(ret3, rett)

    @test isapprox(w1.weights, w2.weights, rtol = 0.005)
    @test isapprox(r1, r2, rtol = 5.0e-5)
    @test isapprox(ret1, ret2, rtol = 0.001)

    @test isapprox(w1.weights, w3.weights, rtol = 0.005)
    @test isapprox(r1, r3, rtol = 5.0e-5)
    @test isapprox(ret1, ret3, rtol = 0.0001)

    @test isapprox(w2.weights, w3.weights, rtol = 0.005)
    @test isapprox(r2, r3, rtol = 5.0e-5)
    @test isapprox(ret2, ret3, rtol = 0.001)

    obj = Util(; l = l)
    rm = SD2(; formulation = SOCSD())
    w4 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r4 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret4 = dot(portfolio.mu, w4.weights)
    wt = [1.065553054256496e-9, 1.906979877393637e-9, 2.1679869440360567e-9,
          1.70123972526289e-9, 0.7741855142171694, 3.9721744242294547e-10,
          0.10998135534654405, 1.3730517031876334e-9, 1.5832262577152926e-9,
          1.0504881447825781e-9, 1.2669287896045939e-9, 4.038975120701348e-10,
          6.074001448526581e-10, 2.654358762537183e-10, 6.574536682273354e-10,
          0.1158331072870088, 3.0452991740231055e-9, 1.3663094482455795e-9,
          2.4334674474942e-9, 1.8573424305703526e-9]
    riskt = 0.01609460480445889
    rett = 0.0017268228943243054
    @test isapprox(w4.weights, wt)
    @test isapprox(r4, riskt)
    @test isapprox(ret4, rett)

    rm = SD2(; formulation = QuadSD())
    w5 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r5 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret5 = dot(portfolio.mu, w5.weights)
    wt = [6.852081519398486e-11, 1.9688305552023357e-10, 2.779044358366435e-10,
          2.0509456674893866e-10, 0.7741909329958984, 2.003910403124917e-11,
          0.1099865880247063, 9.121516622151838e-11, 3.6704849024948924e-10,
          8.806774623589372e-11, 7.781246849842617e-11, 1.813903586007653e-11,
          1.2669422405822381e-11, 3.710822071187361e-11, 1.2738399524886383e-11,
          0.11582247405793253, 2.2605095246705475e-9, 1.0496320870713167e-10,
          9.506590849948595e-10, 1.320900856147283e-10]
    riskt = 0.016094714945994234
    rett = 0.0017268299955487294
    @test isapprox(w5.weights, wt)
    @test isapprox(r5, riskt)
    @test isapprox(ret5, rett)

    rm = SD2(; formulation = SimpleSD())
    w6 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r6 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret6 = dot(portfolio.mu, w6.weights)
    wt = [3.3845525734762743e-7, 0.03245411192485674, 0.014041141771232574,
          0.028634625422379358, 0.030290985036508705, 0.005376551745731238,
          4.200673931391037e-9, 0.13652895353559166, 6.6304529461540396e-9,
          0.0006160259140775326, 0.29064360722877763, 2.67074552134372e-9,
          1.9721256449954764e-9, 0.11178585267210861, 5.669569610426783e-9,
          0.020423298923273724, 0.015543439516805872, 0.19712337633735869,
          9.611449035566669e-9, 0.11653766076102358]
    riskt = 0.0077229723137168345
    rett = 0.0004207388247934707
    @test isapprox(w6.weights, wt)
    @test isapprox(r6, riskt)
    @test isapprox(ret6, rett)

    @test isapprox(w4.weights, w5.weights, rtol = 5e-5)
    @test isapprox(r4, r5, rtol = 1.0e-5)
    @test isapprox(ret4, ret5, rtol = 5.0e-6)

    @test !isapprox(w4.weights, w6.weights)
    @test !isapprox(r4, r6)
    @test !isapprox(ret4, ret6)

    @test !isapprox(w5.weights, w6.weights)
    @test !isapprox(r5, r6)
    @test !isapprox(ret5, ret6)

    obj = SR(; rf = rf)
    rm = SD2(; formulation = SOCSD())
    w7 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r7 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret7 = dot(portfolio.mu, w7.weights)
    wt = [3.435681413150756e-10, 9.269743242817558e-10, 1.174114891347931e-9,
          7.447407606568295e-10, 0.5180552669298059, 1.08151797431462e-10,
          0.0636508036260703, 7.872264113421062e-10, 7.841830201959634e-10,
          3.9005509625957585e-10, 6.479557895235057e-10, 8.472023236127232e-11,
          5.766670106753152e-11, 1.988136246095318e-10, 5.935811276550078e-11,
          0.14326634942881586, 0.1964867973307653, 7.554937254824565e-10,
          0.0785407748474901, 7.740298948228655e-10]
    riskt = 0.013160876658207102
    rett = 0.0014788430765515807
    @test isapprox(w7.weights, wt)
    @test isapprox(r7, riskt)
    @test isapprox(ret7, rett)

    rm = SD2(; formulation = QuadSD())
    w8 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r8 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret8 = dot(portfolio.mu, w8.weights)
    wt = [2.2723291920423494e-10, 8.274438575006142e-10, 1.525985513481683e-9,
          7.273695858012778e-10, 0.5180580411000183, 6.940884975732612e-11,
          0.06365095030024893, 5.333993060298542e-10, 3.9178175629371144e-10,
          2.433346403837124e-10, 3.8877926150779944e-10, 5.601703894979824e-11,
          4.159446993235074e-11, 1.275650437326318e-10, 3.926119276017696e-11,
          0.1432677858447418, 0.19649634671149677, 4.6010816369377544e-10,
          0.07852686985456356, 5.296488926346786e-10]
    riskt = 0.013160920956832975
    rett = 0.0014788476224523953
    @test isapprox(w8.weights, wt)
    @test isapprox(r8, riskt)
    @test isapprox(ret8, rett)

    rm = SD2(; formulation = SimpleSD())
    w9 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r9 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret9 = dot(portfolio.mu, w9.weights)
    wt = [9.288153197917532e-9, 2.5287441861372433e-8, 3.262334495626996e-8,
          2.042021085171968e-8, 0.518057899726503, 2.824107456066348e-9,
          0.06365089409237853, 2.1380549780544252e-8, 2.025773500381765e-8,
          1.0387505704405014e-8, 1.727503562974513e-8, 2.2309535984252933e-9,
          1.6025905812586714e-9, 5.242822500115322e-9, 1.5543835762675563e-9,
          0.1432677690591166, 0.1964967492101948, 2.0271309849859462e-8,
          0.07852647637780057, 2.0887861949385024e-8]
    riskt = 0.013160919559594407
    rett = 0.0014788474312895242
    @test isapprox(w9.weights, wt, rtol = 5.0e-8)
    @test isapprox(r9, riskt)
    @test isapprox(ret9, rett)

    @test isapprox(w7.weights, w8.weights, rtol = 5.0e-5)
    @test isapprox(r7, r8, rtol = 5.0e-6)
    @test isapprox(ret7, ret8, rtol = 5.0e-6)

    @test isapprox(w7.weights, w9.weights, rtol = 5.0e-5)
    @test isapprox(r7, r9, rtol = 5.0e-6)
    @test isapprox(ret7, ret9, rtol = 5.0e-6)

    @test isapprox(w8.weights, w9.weights, rtol = 5.0e-6)
    @test isapprox(r8, r9, rtol = 5.0e-7)
    @test isapprox(ret8, ret9, rtol = 5.0e-7)

    obj = MaxRet()
    rm = SD2(; formulation = SOCSD())
    w10 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r10 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret10 = dot(portfolio.mu, w10.weights)
    wt = [8.719239065561217e-10, 8.89979881250899e-10, 9.962391574545242e-10,
          9.646258079031587e-10, 8.401034275877383e-9, 4.828906105645332e-10,
          0.9999999782316086, 6.87727997367879e-10, 9.021263551270326e-10,
          7.350693996493505e-10, 6.753002461228969e-10, 5.009649350579108e-10,
          3.7428368039997965e-10, 5.884337547691459e-10, 3.9326986484718143e-10,
          8.842556785821523e-10, 9.784139669171374e-10, 7.146277206720297e-10,
          9.044289592659792e-10, 8.2279511460579e-10]
    riskt = 0.040597851628968784
    rett = 0.0018453756308089402
    @test isapprox(w10.weights, wt)
    @test isapprox(r10, riskt)
    @test isapprox(ret10, rett)

    rm = SD2(; formulation = QuadSD())
    w11 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r11 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret11 = dot(portfolio.mu, w11.weights)
    wt = [8.719239065561217e-10, 8.89979881250899e-10, 9.962391574545242e-10,
          9.646258079031587e-10, 8.401034275877383e-9, 4.828906105645332e-10,
          0.9999999782316086, 6.87727997367879e-10, 9.021263551270326e-10,
          7.350693996493505e-10, 6.753002461228969e-10, 5.009649350579108e-10,
          3.7428368039997965e-10, 5.884337547691459e-10, 3.9326986484718143e-10,
          8.842556785821523e-10, 9.784139669171374e-10, 7.146277206720297e-10,
          9.044289592659792e-10, 8.2279511460579e-10]
    riskt = 0.040597851628968784
    rett = 0.0018453756308089402
    @test isapprox(w11.weights, wt)
    @test isapprox(r11, riskt)
    @test isapprox(ret11, rett)

    rm = SD2(; formulation = SimpleSD())
    w12 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r12 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret12 = dot(portfolio.mu, w12.weights)
    wt = [8.719239065561217e-10, 8.89979881250899e-10, 9.962391574545242e-10,
          9.646258079031587e-10, 8.401034275877383e-9, 4.828906105645332e-10,
          0.9999999782316086, 6.87727997367879e-10, 9.021263551270326e-10,
          7.350693996493505e-10, 6.753002461228969e-10, 5.009649350579108e-10,
          3.7428368039997965e-10, 5.884337547691459e-10, 3.9326986484718143e-10,
          8.842556785821523e-10, 9.784139669171374e-10, 7.146277206720297e-10,
          9.044289592659792e-10, 8.2279511460579e-10]
    riskt = 0.040597851628968784
    rett = 0.0018453756308089402
    @test isapprox(w12.weights, wt)
    @test isapprox(r12, riskt)
    @test isapprox(ret12, rett)

    @test isapprox(w10.weights, w11.weights)
    @test isapprox(r10, r11)
    @test isapprox(ret10, ret11)

    @test isapprox(w10.weights, w12.weights)
    @test isapprox(r10, r12)
    @test isapprox(ret10, ret12)

    @test isapprox(w11.weights, w12.weights)
    @test isapprox(r11, r12)
    @test isapprox(ret11, ret12)

    obj = MinRisk()
    rm = SD2(; formulation = SOCSD())
    w13 = optimise2!(portfolio; rm = rm, kelly = AKelly(; formulation = SOCSD()), obj = obj)
    r13 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret13 = dot(portfolio.mu, w13.weights)
    wt = [0.0078093616583851146, 0.030698578999046943, 0.010528316771324561,
          0.027486806578381814, 0.012309038313357737, 0.03341430871186881,
          1.1079055085166888e-7, 0.13985416268183082, 2.0809271230580642e-7,
          6.32554715643476e-6, 0.28784123553791136, 1.268075347971748e-7,
          8.867081236591187e-8, 0.12527141708492384, 3.264070667606171e-7,
          0.015079837844627948, 1.5891383112101438e-5, 0.1931406057562605,
          2.654124109083291e-7, 0.11654298695072397]
    riskt = 0.007704593409157056
    rett = 0.0003482663810696356
    @test isapprox(w13.weights, wt)
    @test isapprox(r13, riskt)
    @test isapprox(ret13, rett)

    rm = SD2(; formulation = QuadSD())
    w14 = optimise2!(portfolio; rm = rm, kelly = AKelly(; formulation = SOCSD()), obj = obj)
    r14 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret14 = dot(portfolio.mu, w14.weights)
    wt = [0.007931704468101724, 0.030580315769072094, 0.010456593899158567,
          0.02742093947332831, 0.012239100149899967, 0.033302579036721264,
          4.162859647692109e-6, 0.13981730989327237, 5.788675176975045e-6,
          0.00033179125802216067, 0.2877731391009407, 3.3160181115934523e-6,
          2.9562651082726095e-6, 0.12512918292552744, 3.227218484382772e-5,
          0.015034147440717766, 0.0005108903689970619, 0.19299166064346063,
          6.739278073914423e-6, 0.1164254102918178]
    riskt = 0.007704683283635234
    rett = 0.0003484772957329131
    @test isapprox(w14.weights, wt)
    @test isapprox(r14, riskt)
    @test isapprox(ret14, rett)

    rm = SD2(; formulation = SimpleSD())
    w15 = optimise2!(portfolio; rm = rm, kelly = AKelly(; formulation = SOCSD()), obj = obj)
    r15 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret15 = dot(portfolio.mu, w15.weights)
    wt = [0.007892956618378178, 0.03069135078310372, 0.010510053328054379,
          0.02748716488934741, 0.012284726286598107, 0.033412747896745434,
          3.4510983561959808e-9, 0.1398498186599616, 6.463477389090647e-9,
          1.2183912701620326e-7, 0.28783457711461696, 3.971402286172399e-9,
          2.9059474605121267e-9, 0.12527565747300592, 1.1534530258372196e-8,
          0.015085797229230475, 3.1071292428662565e-7, 0.19312938283717607,
          8.183117329711524e-9, 0.11654529782215743]
    riskt = 0.007704591083785214
    rett = 0.00034824217501046187
    @test isapprox(w15.weights, wt)
    @test isapprox(r15, riskt)
    @test isapprox(ret15, rett)

    @test isapprox(w13.weights, w14.weights, rtol = 0.005)
    @test isapprox(r13, r14, rtol = 5.0e-5)
    @test isapprox(ret13, ret14, rtol = 0.001)

    @test isapprox(w13.weights, w15.weights, rtol = 0.0005)
    @test isapprox(r13, r15, rtol = 5.0e-7)
    @test isapprox(ret13, ret15, rtol = 0.0001)

    @test isapprox(w14.weights, w15.weights, rtol = 0.005)
    @test isapprox(r14, r15, rtol = 5.0e-5)
    @test isapprox(ret14, ret15, rtol = 0.001)

    obj = Util(; l = l)
    rm = SD2(; formulation = SOCSD())
    w16 = optimise2!(portfolio; rm = rm, kelly = AKelly(; formulation = SOCSD()), obj = obj)
    r16 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret16 = dot(portfolio.mu, w16.weights)
    wt = [8.870505348946403e-9, 2.0785636066552727e-8, 2.798137197308657e-8,
          1.971091339190289e-8, 0.7185881405281792, 4.806880649144299e-10,
          0.09860828964210354, 1.1235098321720224e-8, 2.977172777854582e-8,
          8.912749778026878e-9, 9.63062128166912e-9, 1.0360544993920464e-9,
          2.180352541614548e-9, 2.689800139816139e-9, 2.3063944199708073e-9,
          0.15518499560246005, 0.027618271886178034, 1.246121371211767e-8,
          1.2842725621709964e-7, 1.586069567397408e-8]
    riskt = 0.015437603649758339
    rett = 0.0016792722833452185
    @test isapprox(w16.weights, wt)
    @test isapprox(r16, riskt)
    @test isapprox(ret16, rett)

    rm = SD2(; formulation = QuadSD())
    w17 = optimise2!(portfolio; rm = rm, kelly = AKelly(; formulation = SOCSD()), obj = obj)
    r17 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret17 = dot(portfolio.mu, w17.weights)
    wt = [1.6868171199009505e-9, 6.518606011133161e-9, 9.86352279770984e-9,
          6.150045875747458e-9, 0.718630983702095, 3.7119199852091685e-10,
          0.09861482354309821, 2.790532675752989e-9, 1.2728774591077898e-8,
          2.1871465657675766e-9, 2.2693974383129075e-9, 2.958986484663317e-10,
          2.0351357726703993e-10, 7.649820299793997e-10, 1.9650537328495345e-10,
          0.15517370021004367, 0.027580337279790704, 3.3046232245984773e-9,
          1.0179775163080922e-7, 4.1356629132822375e-9]
    riskt = 0.01543809037618323
    rett = 0.0016793098997042355
    @test isapprox(w17.weights, wt)
    @test isapprox(r17, riskt)
    @test isapprox(ret17, rett)

    rm = SD2(; formulation = SimpleSD())
    w18 = optimise2!(portfolio; rm = rm, kelly = AKelly(; formulation = SOCSD()), obj = obj)
    r18 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret18 = dot(portfolio.mu, w18.weights)
    wt = [0.00013252521373139774, 0.032348823937503775, 0.013768728500928654,
          0.028574435773051073, 0.02706641022737482, 0.01116868693439196,
          2.7658196890296503e-8, 0.13726587064660747, 4.403989680287055e-8,
          4.667120708760286e-5, 0.29018499771794004, 1.9210424282030228e-8,
          1.4462185135147743e-8, 0.1146653670353858, 4.603743952550462e-8,
          0.01938229311906383, 0.011904698959685265, 0.19667027309919696,
          6.172737257419162e-8, 0.11682000449253614]
    riskt = 0.007716433466493668
    rett = 0.0004062057200139071
    @test isapprox(w18.weights, wt)
    @test isapprox(r18, riskt)
    @test isapprox(ret18, rett)

    @test isapprox(w16.weights, w17.weights, rtol = 0.0001)
    @test isapprox(r16, r17, rtol = 5.0e-5)
    @test isapprox(ret16, ret17, rtol = 5.0e-5)

    @test !isapprox(w16.weights, w18.weights)
    @test !isapprox(r16, r18)
    @test !isapprox(ret16, ret18)

    @test !isapprox(w17.weights, w18.weights)
    @test !isapprox(r17, r18)
    @test !isapprox(ret17, ret18)

    obj = SR(; rf = rf)
    rm = SD2(; formulation = SOCSD())
    w19 = optimise2!(portfolio; rm = rm, kelly = AKelly(; formulation = SOCSD()), obj = obj)
    r19 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret19 = dot(portfolio.mu, w19.weights)
    wt = [1.0672947772539922e-8, 3.6603580846259566e-8, 4.190924498057212e-8,
          2.579795624783031e-8, 0.45247454726503317, 3.139203265461306e-9,
          0.05198581042386962, 1.1201379704294516e-7, 1.799097939748088e-8,
          1.2844577033392204e-8, 5.1484053193477936e-8, 2.3241091705338425e-9,
          1.699312214555245e-9, 6.26319015273334e-9, 1.6636900367102399e-9,
          0.13648649205020114, 0.2350741185365231, 4.844604537439258e-8,
          0.12397862173181687, 3.713986941437853e-8]
    riskt = 0.012532646375473184
    rett = 0.0014098003867777677
    @test isapprox(w19.weights, wt, rtol = 5.0e-8)
    @test isapprox(r19, riskt)
    @test isapprox(ret19, rett)

    rm = SD2(; formulation = QuadSD())
    w20 = optimise2!(portfolio; rm = rm, kelly = AKelly(; formulation = SOCSD()), obj = obj)
    r20 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret20 = dot(portfolio.mu, w20.weights)
    wt = [8.235412254567274e-8, 3.3869214614352685e-7, 4.3387088917280187e-7,
          2.172031674215957e-7, 0.45250197384117946, 2.390192776356049e-8,
          0.05199034420584679, 8.848846811240671e-7, 1.3444231577151073e-7,
          9.837931270837787e-8, 4.369321262087557e-7, 1.774237715854489e-8,
          1.3029937877791496e-8, 4.7466504823938394e-8, 1.263306083191649e-8,
          0.13648904728986502, 0.23504422368312097, 4.3376556992927087e-7,
          0.12397092672117808, 3.0896067026143913e-7]
    riskt = 0.012532862921884226
    rett = 0.0014098257752725304
    @test isapprox(w20.weights, wt, rtol = 1.0e-7)
    @test isapprox(r20, riskt)
    @test isapprox(ret20, rett)

    rm = SD2(; formulation = SimpleSD())
    w21 = optimise2!(portfolio; rm = rm, kelly = AKelly(; formulation = SOCSD()), obj = obj)
    r21 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret21 = dot(portfolio.mu, w21.weights)
    wt = [1.1880683879690429e-7, 6.814589470902193e-7, 9.465092715178632e-7,
          3.421023543582475e-7, 0.452492943281118, 3.6575939684051314e-8,
          0.05198997115208025, 4.034408585281609e-6, 1.7729081972465543e-7,
          1.3222313684672454e-7, 1.0668680875725833e-6, 2.7121698741803906e-8,
          1.9413619089801955e-8, 6.881490909702032e-8, 1.959765228165462e-8,
          0.13648814335285792, 0.2350514754242049, 1.0322471086607908e-6,
          0.12396817976239216, 5.835883779337245e-7]
    riskt = 0.012532777032115257
    rett = 0.0014098152452307682
    @test isapprox(w21.weights, wt, rtol = 5.0e-7)
    @test isapprox(r21, riskt, rtol = 5.0e-8)
    @test isapprox(ret21, rett, rtol = 5.0e-8)

    @test isapprox(w19.weights, w20.weights, rtol = 0.0001)
    @test isapprox(r19, r20, rtol = 5.0e-5)
    @test isapprox(ret19, ret20, rtol = 5.0e-5)

    @test isapprox(w19.weights, w21.weights, rtol = 0.0001)
    @test isapprox(r19, r21, rtol = 5.0e-5)
    @test isapprox(ret19, ret21, rtol = 5.0e-5)

    @test isapprox(w20.weights, w21.weights, rtol = 5.0e-5)
    @test isapprox(r20, r21, rtol = 1.0e-5)
    @test isapprox(ret20, ret21, rtol = 1.0e-5)

    obj = MaxRet()
    rm = SD2(; formulation = SOCSD())
    w22 = optimise2!(portfolio; rm = rm, kelly = AKelly(; formulation = SOCSD()), obj = obj)
    r22 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret22 = dot(portfolio.mu, w22.weights)
    wt = [3.997304959875111e-10, 6.096555731047457e-10, 1.0828360159599854e-9,
          8.390684135793784e-10, 0.8503431998881756, 5.837264433583887e-10,
          0.14965678808511193, 8.520932897976487e-13, 7.66160958682953e-10,
          1.0247860261071675e-10, 5.1627700971086255e-11, 5.483183958203547e-10,
          7.565204185674542e-10, 3.16106264753721e-10, 7.638502459889708e-10,
          2.447496129413098e-9, 1.372927322256315e-9, 6.541563185875491e-11,
          9.248420166125226e-10, 3.9509971643490626e-10]
    riskt = 0.017523378944242916
    rett = 0.001803059901755384
    @test isapprox(w22.weights, wt)
    @test isapprox(r22, riskt)
    @test isapprox(ret22, rett)

    rm = SD2(; formulation = QuadSD())
    w23 = optimise2!(portfolio; rm = rm, kelly = AKelly(; formulation = SOCSD()), obj = obj)
    r23 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret23 = dot(portfolio.mu, w23.weights)
    wt = [3.1491090891677036e-10, 3.9252510945045457e-10, 5.011436739931957e-10,
          4.814228838168751e-10, 0.8503245220331475, 4.987452504286351e-11,
          0.14967547181371002, 1.8257330907415293e-10, 4.5000351561986515e-10,
          2.195999249689875e-10, 1.8725349256894292e-10, 7.858729597392012e-11,
          2.333547428526958e-10, 1.351809534814174e-10, 1.8382016146366064e-10,
          1.1430290850714406e-9, 6.064115688191264e-10, 2.1745240525978677e-10,
          4.80092693730735e-10, 2.959062049654606e-10]
    riskt = 0.01752343205325563
    rett = 0.0018030608400966524
    @test isapprox(w23.weights, wt)
    @test isapprox(r23, riskt)
    @test isapprox(ret23, rett)

    rm = SD2(; formulation = SimpleSD())
    w24 = optimise2!(portfolio; rm = rm, kelly = AKelly(; formulation = SOCSD()), obj = obj)
    r24 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret24 = dot(portfolio.mu, w24.weights)
    wt = [1.1217675271775678e-7, 0.03320414215533587, 0.012524704290247737,
          0.026110249232198696, 0.08360000446030601, 3.904449262956923e-8,
          0.0006541477156547997, 0.12525073220221242, 8.303542345798751e-8,
          4.35268081045318e-7, 0.27786098907764745, 1.2425286059678063e-8,
          9.096986741561476e-9, 0.047202238450064625, 1.6006757884557573e-8,
          0.037876877445336625, 0.06717815089023799, 0.19031170742002357,
          5.248015934696513e-7, 0.09822482480536034]
    riskt = 0.007909092769906272
    rett = 0.0005709556368833311
    @test isapprox(w24.weights, wt)
    @test isapprox(r24, riskt)
    @test isapprox(ret24, rett)

    @test isapprox(w22.weights, w23.weights, rtol = 5.0e-5)
    @test isapprox(r22, r23, rtol = 5.0e-6)
    @test isapprox(ret22, ret23, rtol = 1.0e-6)

    @test !isapprox(w22.weights, w24.weights)
    @test !isapprox(r22, r24)
    @test !isapprox(ret22, ret24)

    @test !isapprox(w23.weights, w24.weights)
    @test !isapprox(r23, r24)
    @test !isapprox(ret23, ret24)

    obj = MinRisk()
    rm = SD2(; formulation = SOCSD())
    w25 = optimise2!(portfolio; rm = rm, kelly = AKelly(; formulation = QuadSD()),
                     obj = obj)
    r25 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret25 = dot(portfolio.mu, w25.weights)
    wt = [0.0078093616583851146, 0.030698578999046943, 0.010528316771324561,
          0.027486806578381814, 0.012309038313357737, 0.03341430871186881,
          1.1079055085166888e-7, 0.13985416268183082, 2.0809271230580642e-7,
          6.32554715643476e-6, 0.28784123553791136, 1.268075347971748e-7,
          8.867081236591187e-8, 0.12527141708492384, 3.264070667606171e-7,
          0.015079837844627948, 1.5891383112101438e-5, 0.1931406057562605,
          2.654124109083291e-7, 0.11654298695072397]
    riskt = 0.007704593409157056
    rett = 0.0003482663810696356
    @test isapprox(w25.weights, wt)
    @test isapprox(r25, riskt)
    @test isapprox(ret25, rett)

    rm = SD2(; formulation = QuadSD())
    w26 = optimise2!(portfolio; rm = rm, kelly = AKelly(; formulation = QuadSD()),
                     obj = obj)
    r26 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret26 = dot(portfolio.mu, w26.weights)
    wt = [0.007931704468101724, 0.030580315769072094, 0.010456593899158567,
          0.02742093947332831, 0.012239100149899967, 0.033302579036721264,
          4.162859647692109e-6, 0.13981730989327237, 5.788675176975045e-6,
          0.00033179125802216067, 0.2877731391009407, 3.3160181115934523e-6,
          2.9562651082726095e-6, 0.12512918292552744, 3.227218484382772e-5,
          0.015034147440717766, 0.0005108903689970619, 0.19299166064346063,
          6.739278073914423e-6, 0.1164254102918178]
    riskt = 0.007704683283635234
    rett = 0.0003484772957329131
    @test isapprox(w26.weights, wt)
    @test isapprox(r26, riskt)
    @test isapprox(ret26, rett)

    rm = SD2(; formulation = SimpleSD())
    w27 = optimise2!(portfolio; rm = rm, kelly = AKelly(; formulation = QuadSD()),
                     obj = obj)
    r27 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret27 = dot(portfolio.mu, w27.weights)
    wt = [0.007892956618378178, 0.03069135078310372, 0.010510053328054379,
          0.02748716488934741, 0.012284726286598107, 0.033412747896745434,
          3.4510983561959808e-9, 0.1398498186599616, 6.463477389090647e-9,
          1.2183912701620326e-7, 0.28783457711461696, 3.971402286172399e-9,
          2.9059474605121267e-9, 0.12527565747300592, 1.1534530258372196e-8,
          0.015085797229230475, 3.1071292428662565e-7, 0.19312938283717607,
          8.183117329711524e-9, 0.11654529782215743]
    riskt = 0.007704591083785214
    rett = 0.00034824217501046187
    @test isapprox(w27.weights, wt)
    @test isapprox(r27, riskt)
    @test isapprox(ret27, rett)

    @test isapprox(w25.weights, w26.weights, rtol = 0.005)
    @test isapprox(r25, r26, rtol = 5.0e-5)
    @test isapprox(ret25, ret26, rtol = 0.001)

    @test isapprox(w25.weights, w27.weights, rtol = 0.0005)
    @test isapprox(r25, r27, rtol = 5.0e-7)
    @test isapprox(ret25, ret27, rtol = 0.0001)

    @test isapprox(w26.weights, w27.weights, rtol = 0.005)
    @test isapprox(r26, r27, rtol = 5.0e-5)
    @test isapprox(ret26, ret27, rtol = 0.001)

    obj = Util(; l = l)
    rm = SD2(; formulation = SOCSD())
    w28 = optimise2!(portfolio; rm = rm, kelly = AKelly(; formulation = QuadSD()),
                     obj = obj)
    r28 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret28 = dot(portfolio.mu, w28.weights)
    wt = [8.870505348946403e-9, 2.0785636066552727e-8, 2.798137197308657e-8,
          1.971091339190289e-8, 0.7185881405281792, 4.806880649144299e-10,
          0.09860828964210354, 1.1235098321720224e-8, 2.977172777854582e-8,
          8.912749778026878e-9, 9.63062128166912e-9, 1.0360544993920464e-9,
          2.180352541614548e-9, 2.689800139816139e-9, 2.3063944199708073e-9,
          0.15518499560246005, 0.027618271886178034, 1.246121371211767e-8,
          1.2842725621709964e-7, 1.586069567397408e-8]
    riskt = 0.015437603649758339
    rett = 0.0016792722833452185
    @test isapprox(w28.weights, wt)
    @test isapprox(r28, riskt)
    @test isapprox(ret28, rett)

    rm = SD2(; formulation = QuadSD())
    w29 = optimise2!(portfolio; rm = rm, kelly = AKelly(; formulation = QuadSD()),
                     obj = obj)
    r29 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret29 = dot(portfolio.mu, w29.weights)
    wt = [1.6868171199009505e-9, 6.518606011133161e-9, 9.86352279770984e-9,
          6.150045875747458e-9, 0.718630983702095, 3.7119199852091685e-10,
          0.09861482354309821, 2.790532675752989e-9, 1.2728774591077898e-8,
          2.1871465657675766e-9, 2.2693974383129075e-9, 2.958986484663317e-10,
          2.0351357726703993e-10, 7.649820299793997e-10, 1.9650537328495345e-10,
          0.15517370021004367, 0.027580337279790704, 3.3046232245984773e-9,
          1.0179775163080922e-7, 4.1356629132822375e-9]
    riskt = 0.01543809037618323
    rett = 0.0016793098997042355
    @test isapprox(w29.weights, wt)
    @test isapprox(r29, riskt)
    @test isapprox(ret29, rett)

    rm = SD2(; formulation = SimpleSD())
    w30 = optimise2!(portfolio; rm = rm, kelly = AKelly(; formulation = QuadSD()),
                     obj = obj)
    r30 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret30 = dot(portfolio.mu, w30.weights)
    wt = [0.00013252521373139774, 0.032348823937503775, 0.013768728500928654,
          0.028574435773051073, 0.02706641022737482, 0.01116868693439196,
          2.7658196890296503e-8, 0.13726587064660747, 4.403989680287055e-8,
          4.667120708760286e-5, 0.29018499771794004, 1.9210424282030228e-8,
          1.4462185135147743e-8, 0.1146653670353858, 4.603743952550462e-8,
          0.01938229311906383, 0.011904698959685265, 0.19667027309919696,
          6.172737257419162e-8, 0.11682000449253614]
    riskt = 0.007716433466493668
    rett = 0.0004062057200139071
    @test isapprox(w30.weights, wt)
    @test isapprox(r30, riskt)
    @test isapprox(ret30, rett)

    @test isapprox(w28.weights, w29.weights, rtol = 0.0001)
    @test isapprox(r28, r29, rtol = 5.0e-5)
    @test isapprox(ret28, ret29, rtol = 5.0e-5)

    @test !isapprox(w28.weights, w30.weights)
    @test !isapprox(r28, r30)
    @test !isapprox(ret28, ret30)

    @test !isapprox(w29.weights, w30.weights)
    @test !isapprox(r29, r30)
    @test !isapprox(ret29, ret30)

    obj = SR(; rf = rf)
    rm = SD2(; formulation = SOCSD())
    w31 = optimise2!(portfolio; rm = rm, kelly = AKelly(; formulation = QuadSD()),
                     obj = obj)
    r31 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret31 = dot(portfolio.mu, w31.weights)
    wt = [1.0672947772539922e-8, 3.6603580846259566e-8, 4.190924498057212e-8,
          2.579795624783031e-8, 0.45247454726503317, 3.139203265461306e-9,
          0.05198581042386962, 1.1201379704294516e-7, 1.799097939748088e-8,
          1.2844577033392204e-8, 5.1484053193477936e-8, 2.3241091705338425e-9,
          1.699312214555245e-9, 6.26319015273334e-9, 1.6636900367102399e-9,
          0.13648649205020114, 0.2350741185365231, 4.844604537439258e-8,
          0.12397862173181687, 3.713986941437853e-8]
    riskt = 0.012532646375473184
    rett = 0.0014098003867777677
    @test isapprox(w31.weights, wt, rtol = 5.0e-8)
    @test isapprox(r31, riskt)
    @test isapprox(ret31, rett)

    rm = SD2(; formulation = QuadSD())
    w32 = optimise2!(portfolio; rm = rm, kelly = AKelly(; formulation = QuadSD()),
                     obj = obj)
    r32 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret32 = dot(portfolio.mu, w32.weights)
    wt = [8.235412254567274e-8, 3.3869214614352685e-7, 4.3387088917280187e-7,
          2.172031674215957e-7, 0.45250197384117946, 2.390192776356049e-8,
          0.05199034420584679, 8.848846811240671e-7, 1.3444231577151073e-7,
          9.837931270837787e-8, 4.369321262087557e-7, 1.774237715854489e-8,
          1.3029937877791496e-8, 4.7466504823938394e-8, 1.263306083191649e-8,
          0.13648904728986502, 0.23504422368312097, 4.3376556992927087e-7,
          0.12397092672117808, 3.0896067026143913e-7]
    riskt = 0.012532862921884226
    rett = 0.0014098257752725304
    @test isapprox(w32.weights, wt, rtol = 1.0e-7)
    @test isapprox(r32, riskt)
    @test isapprox(ret32, rett)

    rm = SD2(; formulation = SimpleSD())
    w33 = optimise2!(portfolio; rm = rm, kelly = AKelly(; formulation = QuadSD()),
                     obj = obj)
    r33 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret33 = dot(portfolio.mu, w33.weights)
    wt = [1.1880683879690429e-7, 6.814589470902193e-7, 9.465092715178632e-7,
          3.421023543582475e-7, 0.452492943281118, 3.6575939684051314e-8,
          0.05198997115208025, 4.034408585281609e-6, 1.7729081972465543e-7,
          1.3222313684672454e-7, 1.0668680875725833e-6, 2.7121698741803906e-8,
          1.9413619089801955e-8, 6.881490909702032e-8, 1.959765228165462e-8,
          0.13648814335285792, 0.2350514754242049, 1.0322471086607908e-6,
          0.12396817976239216, 5.835883779337245e-7]
    riskt = 0.012532777032115257
    rett = 0.0014098152452307682
    @test isapprox(w33.weights, wt, rtol = 5.0e-7)
    @test isapprox(r33, riskt, rtol = 5.0e-8)
    @test isapprox(ret33, rett, rtol = 5.0e-8)

    @test isapprox(w31.weights, w32.weights, rtol = 0.0001)
    @test isapprox(r31, r32, rtol = 5.0e-5)
    @test isapprox(ret31, ret32, rtol = 5.0e-5)

    @test isapprox(w31.weights, w33.weights, rtol = 0.0001)
    @test isapprox(r31, r33, rtol = 5.0e-5)
    @test isapprox(ret31, ret33, rtol = 5.0e-5)

    @test isapprox(w32.weights, w33.weights, rtol = 5.0e-5)
    @test isapprox(r32, r33, rtol = 1.0e-5)
    @test isapprox(ret32, ret33, rtol = 1.0e-5)

    obj = MaxRet()
    rm = SD2(; formulation = SOCSD())
    w34 = optimise2!(portfolio; rm = rm, kelly = AKelly(; formulation = QuadSD()),
                     obj = obj)
    r34 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret34 = dot(portfolio.mu, w34.weights)
    wt = [3.997304959875111e-10, 6.096555731047457e-10, 1.0828360159599854e-9,
          8.390684135793784e-10, 0.8503431998881756, 5.837264433583887e-10,
          0.14965678808511193, 8.520932897976487e-13, 7.66160958682953e-10,
          1.0247860261071675e-10, 5.1627700971086255e-11, 5.483183958203547e-10,
          7.565204185674542e-10, 3.16106264753721e-10, 7.638502459889708e-10,
          2.447496129413098e-9, 1.372927322256315e-9, 6.541563185875491e-11,
          9.248420166125226e-10, 3.9509971643490626e-10]
    riskt = 0.017523378944242916
    rett = 0.001803059901755384
    @test isapprox(w34.weights, wt)
    @test isapprox(r34, riskt)
    @test isapprox(ret34, rett)

    rm = SD2(; formulation = QuadSD())
    w35 = optimise2!(portfolio; rm = rm, kelly = AKelly(; formulation = QuadSD()),
                     obj = obj)
    r35 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret35 = dot(portfolio.mu, w35.weights)
    wt = [3.1491090891677036e-10, 3.9252510945045457e-10, 5.011436739931957e-10,
          4.814228838168751e-10, 0.8503245220331475, 4.987452504286351e-11,
          0.14967547181371002, 1.8257330907415293e-10, 4.5000351561986515e-10,
          2.195999249689875e-10, 1.8725349256894292e-10, 7.858729597392012e-11,
          2.333547428526958e-10, 1.351809534814174e-10, 1.8382016146366064e-10,
          1.1430290850714406e-9, 6.064115688191264e-10, 2.1745240525978677e-10,
          4.80092693730735e-10, 2.959062049654606e-10]
    riskt = 0.01752343205325563
    rett = 0.0018030608400966524
    @test isapprox(w35.weights, wt)
    @test isapprox(r35, riskt)
    @test isapprox(ret35, rett)

    rm = SD2(; formulation = SimpleSD())
    w36 = optimise2!(portfolio; rm = rm, kelly = AKelly(; formulation = QuadSD()),
                     obj = obj)
    r36 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret36 = dot(portfolio.mu, w36.weights)
    wt = [1.1217675271775678e-7, 0.03320414215533587, 0.012524704290247737,
          0.026110249232198696, 0.08360000446030601, 3.904449262956923e-8,
          0.0006541477156547997, 0.12525073220221242, 8.303542345798751e-8,
          4.35268081045318e-7, 0.27786098907764745, 1.2425286059678063e-8,
          9.096986741561476e-9, 0.047202238450064625, 1.6006757884557573e-8,
          0.037876877445336625, 0.06717815089023799, 0.19031170742002357,
          5.248015934696513e-7, 0.09822482480536034]
    riskt = 0.007909092769906272
    rett = 0.0005709556368833311
    @test isapprox(w36.weights, wt)
    @test isapprox(r36, riskt)
    @test isapprox(ret36, rett)

    @test isapprox(w34.weights, w35.weights, rtol = 5.0e-5)
    @test isapprox(r34, r35, rtol = 5.0e-6)
    @test isapprox(ret34, ret35, rtol = 1.0e-6)

    @test !isapprox(w34.weights, w36.weights)
    @test !isapprox(r34, r36)
    @test !isapprox(ret34, ret36)

    @test !isapprox(w35.weights, w36.weights)
    @test !isapprox(r35, r36)
    @test !isapprox(ret35, ret36)

    @test isapprox(w13.weights, w25.weights)
    @test isapprox(w14.weights, w26.weights)
    @test isapprox(w15.weights, w27.weights)
    @test isapprox(w16.weights, w28.weights)
    @test isapprox(w17.weights, w29.weights)
    @test isapprox(w18.weights, w30.weights)
    @test isapprox(w19.weights, w31.weights)
    @test isapprox(w20.weights, w32.weights)
    @test isapprox(w21.weights, w33.weights)
    @test isapprox(w22.weights, w34.weights)
    @test isapprox(w23.weights, w35.weights)
    @test isapprox(w24.weights, w36.weights)

    obj = MinRisk()
    rm = SD2(; formulation = SOCSD())
    w37 = optimise2!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    r37 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret37 = dot(portfolio.mu, w37.weights)
    wt = [0.007893834004010548, 0.030693397218184384, 0.01050943911162566,
          0.027487590678529683, 0.0122836015907984, 0.03341312720689581,
          2.654460293680794e-8, 0.13984817931920596, 4.861252776141605e-8,
          3.309876672531783e-7, 0.2878217133212084, 3.116270780466401e-8,
          2.2390519612760115e-8, 0.12528318523437287, 9.334010636386356e-8,
          0.015085761770714442, 7.170234777802365e-7, 0.19312554465008394,
          6.179100704970626e-8, 0.11655329404175341]
    riskt = 0.007704591386009447
    rett = 0.0003482418640932403
    @test isapprox(w37.weights, wt)
    @test isapprox(r37, riskt)
    @test isapprox(ret37, rett)

    rm = SD2(; formulation = QuadSD())
    w38 = optimise2!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    r38 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret38 = dot(portfolio.mu, w38.weights)
    wt = [0.0079038447398669, 0.03067417723293006, 0.01049701116722492,
          0.027478077515798748, 0.012272768984669821, 0.033395062173063324,
          9.107944911795474e-8, 0.13984313136066523, 1.0887079574279266e-7,
          5.461103669713283e-5, 0.2878154142627347, 6.40042257524692e-8,
          6.920299340996158e-8, 0.12526253439190477, 2.42483090318101e-6,
          0.015077016639342558, 8.396730564557102e-5, 0.19310428804051907,
          1.1523196443551639e-7, 0.11653522192860562]
    riskt = 0.007704599625058708
    rett = 0.0003482771661845515
    @test isapprox(w38.weights, wt, rtol = 5.0e-7)
    @test isapprox(r38, riskt, rtol = 5.0e-7)
    @test isapprox(ret38, rett, rtol = 5.0e-7)

    rm = SD2(; formulation = SimpleSD())
    w39 = optimise2!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    r39 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret39 = dot(portfolio.mu, w39.weights)
    wt = [0.00790409633874847, 0.030689928037939188, 0.010507998487432462,
          0.02748686384854246, 0.012279538174146456, 0.033412706024557044,
          5.863252541677311e-10, 0.1398487299449627, 1.0456745320204646e-9,
          2.356042145726172e-8, 0.2878234001518533, 6.520325820725347e-10,
          4.78341311046709e-10, 0.125283393778506, 2.0064216192084034e-9,
          0.015084912392169235, 4.354723284523309e-8, 0.1931247757099064,
          1.3224469461845564e-9, 0.11655358391233962]
    riskt = 0.007704591029545491
    rett = 0.0003482360682112058
    @test isapprox(w39.weights, wt)
    @test isapprox(r39, riskt)
    @test isapprox(ret39, rett)

    @test isapprox(w37.weights, w38.weights, rtol = 0.0005)
    @test isapprox(r37, r38, rtol = 5.0e-6)
    @test isapprox(ret37, ret38, rtol = 0.0005)

    @test isapprox(w37.weights, w39.weights, rtol = 5.0e-5)
    @test isapprox(r37, r39, rtol = 5.0e-8)
    @test isapprox(ret37, ret39, rtol = 5.0e-5)

    @test isapprox(w38.weights, w39.weights, rtol = 0.0005)
    @test isapprox(r38, r39, rtol = 5.0e-6)
    @test isapprox(ret38, ret39, rtol = 0.0005)

    obj = Util(; l = l)
    rm = SD2(; formulation = SOCSD())
    w40 = optimise2!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    r40 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret40 = dot(portfolio.mu, w40.weights)
    wt = [4.176019357134568e-9, 8.185339020442324e-9, 1.0737258331407902e-8,
          7.901762479846926e-9, 0.7207182132289714, 1.1540211071681835e-9,
          0.09835523884681871, 4.849354486370478e-9, 1.1755943684787842e-8,
          4.185213130955141e-9, 4.314456480234504e-9, 9.722540074689786e-10,
          5.895848876745837e-10, 2.1334187374036406e-9, 5.661079854916932e-10,
          0.15505269633577082, 0.025873730012408114, 5.286907987963515e-9,
          4.83569668574532e-8, 6.41142230104617e-9]
    riskt = 0.01545637469747925
    rett = 0.0016807188921860062
    @test isapprox(w40.weights, wt, rtol = 5.0e-7)
    @test isapprox(r40, riskt, rtol = 5.0e-8)
    @test isapprox(ret40, rett, rtol = 5.0e-8)

    rm = SD2(; formulation = QuadSD())
    w41 = optimise2!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    r41 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret41 = dot(portfolio.mu, w41.weights)
    wt = [1.283441637415495e-9, 2.440610108113494e-9, 3.179201148967639e-9,
          2.3666779313029497e-9, 0.7207182073299757, 4.4698968146679326e-10,
          0.09835514401060054, 1.4766118010562706e-9, 3.447005483088427e-9,
          1.2851976401129992e-9, 1.322396214137293e-9, 3.760213159114525e-10,
          2.629298030829916e-10, 7.047260359879429e-10, 2.639057995063778e-10,
          0.15505231182978302, 0.025874301218806878, 1.6021560458923619e-9,
          1.3224736594162053e-8, 1.9282264449537997e-9]
    riskt = 0.015456372955099407
    rett = 0.0016807187764718914
    @test isapprox(w41.weights, wt, rtol = 1.0e-6)
    @test isapprox(r41, riskt, rtol = 5.0e-7)
    @test isapprox(ret41, rett, rtol = 5.0e-7)

    rm = SD2(; formulation = SimpleSD())
    w42 = optimise2!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    r42 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret42 = dot(portfolio.mu, w42.weights)
    wt = [6.214736363600313e-8, 0.03245587850018982, 0.014032726256566989,
          0.028632075999491004, 0.030223671140647047, 0.005499489002845217,
          1.0593687302923256e-9, 0.13654440716481103, 1.6177970718811873e-9,
          0.0005979353759582741, 0.29063387999605544, 6.581419567102793e-10,
          4.996472030374174e-10, 0.11184818873333964, 1.4562892699089074e-9,
          0.020401419608021897, 0.015467143766117195, 0.1971159808699044,
          2.3417136618323754e-9, 0.11654713380573052]
    riskt = 0.007722818747684341
    rett = 0.0004204311133053862
    @test isapprox(w42.weights, wt, rtol = 5.0e-6)
    @test isapprox(r42, riskt, rtol = 5.0e-8)
    @test isapprox(ret42, rett, rtol = 1.0e-6)

    @test isapprox(w40.weights, w41.weights, rtol = 1.0e-6)
    @test isapprox(r40, r41, rtol = 5.0e-7)
    @test isapprox(ret40, ret41, rtol = 5.0e-7)

    @test !isapprox(w40.weights, w42.weights)
    @test !isapprox(r40, r42)
    @test !isapprox(ret40, ret42)

    @test !isapprox(w41.weights, w42.weights)
    @test !isapprox(r41, r42)
    @test !isapprox(ret41, ret42)

    obj = SR(; rf = rf)
    rm = SD2(; formulation = SOCSD())
    w43 = optimise2!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    r43 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret43 = dot(portfolio.mu, w43.weights)
    wt = [1.6553521389413233e-7, 5.275915958538328e-7, 6.846007363199405e-7,
          3.914933469758698e-7, 0.48926168709100504, 4.9332881037102406e-8,
          0.0583064644410985, 5.594366962947531e-7, 3.1357711474708337e-7,
          1.895896838004368e-7, 4.1299427275337544e-7, 3.811276445091462e-8,
          2.7731552876975723e-8, 9.393138539482288e-8, 2.6831018704067043e-8,
          0.1402408063077745, 0.2134138585246757, 4.713662104500069e-7, 0.09877278790316771,
          4.4360780483006885e-7]
    riskt = 0.012871646142553771
    rett = 0.0014482928429808686
    @test isapprox(w43.weights, wt, rtol = 0.005)
    @test isapprox(r43, riskt, rtol = 0.0005)
    @test isapprox(ret43, rett, rtol = 0.0005)

    rm = SD2(; formulation = QuadSD())
    w44 = optimise2!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    r44 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret44 = dot(portfolio.mu, w44.weights)
    wt = [1.8066657889667037e-7, 6.211224047267681e-7, 8.218680436150428e-7,
          4.2901188167305846e-7, 0.4892264519150767, 5.4896861574378756e-8,
          0.058310462447603094, 9.466957685428644e-7, 3.318230094204569e-7,
          2.0527150618206202e-7, 5.898130855796297e-7, 4.3654365903048476e-8,
          3.15206149069227e-8, 1.0230356420007934e-7, 3.057132673768531e-8,
          0.14027982350011933, 0.21355443907528876, 6.542230656143559e-7,
          0.09862324690617728, 5.327136572668713e-7]
    riskt = 0.012871670066043072
    rett = 0.0014482953082515207
    @test isapprox(w44.weights, wt, rtol = 5.0e-6)
    @test isapprox(r44, riskt, rtol = 1.0e-7)
    @test isapprox(ret44, rett, rtol = 1.0e-7)

    rm = SD2(; formulation = SimpleSD())
    w45 = optimise2!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    r45 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret45 = dot(portfolio.mu, w45.weights)
    wt = [3.132767774963053e-7, 1.0196096155442822e-6, 1.3356606891735424e-6,
          7.431360887646086e-7, 0.489500399278415, 9.470402658035639e-8,
          0.05835377136437526, 1.1596196497474775e-6, 6.501493831429562e-7,
          3.536685987904172e-7, 8.290345179546054e-7, 7.536168556631919e-8,
          5.434244468509493e-8, 1.7658455778729018e-7, 5.283894371229725e-8,
          0.14033225214128342, 0.21340504200938865, 9.399392154726395e-7,
          0.09839987692102177, 8.603593215001693e-7]
    riskt = 0.012874349686716243
    rett = 0.0014485863653838002
    @test isapprox(w45.weights, wt)
    @test isapprox(r45, riskt)
    @test isapprox(ret45, rett)

    @test isapprox(w43.weights, w44.weights, rtol = 0.005)
    @test isapprox(r43, r44, rtol = 0.0005)
    @test isapprox(ret43, ret44, rtol = 0.0005)

    @test isapprox(w43.weights, w45.weights, rtol = 0.005)
    @test isapprox(r43, r45, rtol = 0.001)
    @test isapprox(ret43, ret45, rtol = 0.001)

    @test isapprox(w44.weights, w45.weights, rtol = 0.001)
    @test isapprox(r44, r45, rtol = 0.0005)
    @test isapprox(ret44, ret45, rtol = 0.0005)

    obj = MaxRet()
    rm = SD2(; formulation = SOCSD())
    w46 = optimise2!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    r46 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret46 = dot(portfolio.mu, w46.weights)
    wt = [1.2461220626745066e-9, 1.3954805691188813e-9, 1.7339340756162538e-9,
          1.5635882299558742e-9, 0.853395149768853, 5.48096579085184e-10,
          0.14660482860584584, 9.501446622854747e-10, 1.5113651288469943e-9,
          1.027931406345638e-9, 9.130613494698656e-10, 5.686010690200261e-10,
          4.0494468011345616e-10, 7.290999439594515e-10, 4.1154424470964885e-10,
          2.82566220199723e-9, 1.9419703441146337e-9, 1.0003454025331967e-9,
          1.6178718912419106e-9, 1.2355373241783204e-9]
    riskt = 0.017515177411134022
    rett = 0.001802908016949115
    @test isapprox(w46.weights, wt, rtol = 1.0e-5)
    @test isapprox(r46, riskt, rtol = 1.0e-6)
    @test isapprox(ret46, rett, rtol = 5.0e-7)

    rm = SD2(; formulation = QuadSD())
    w47 = optimise2!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    r47 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret47 = dot(portfolio.mu, w47.weights)
    wt = [1.2461220626745066e-9, 1.3954805691188813e-9, 1.7339340756162538e-9,
          1.5635882299558742e-9, 0.853395149768853, 5.48096579085184e-10,
          0.14660482860584584, 9.501446622854747e-10, 1.5113651288469943e-9,
          1.027931406345638e-9, 9.130613494698656e-10, 5.686010690200261e-10,
          4.0494468011345616e-10, 7.290999439594515e-10, 4.1154424470964885e-10,
          2.82566220199723e-9, 1.9419703441146337e-9, 1.0003454025331967e-9,
          1.6178718912419106e-9, 1.2355373241783204e-9]
    riskt = 0.017515177411134022
    rett = 0.001802908016949115
    @test isapprox(w47.weights, wt, rtol = 1.0e-5)
    @test isapprox(r47, riskt, rtol = 1.0e-6)
    @test isapprox(ret47, rett, rtol = 5.0e-7)

    rm = SD2(; formulation = SimpleSD())
    w48 = optimise2!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    r48 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret48 = dot(portfolio.mu, w48.weights)
    wt = [1.2461220626745066e-9, 1.3954805691188813e-9, 1.7339340756162538e-9,
          1.5635882299558742e-9, 0.853395149768853, 5.48096579085184e-10,
          0.14660482860584584, 9.501446622854747e-10, 1.5113651288469943e-9,
          1.027931406345638e-9, 9.130613494698656e-10, 5.686010690200261e-10,
          4.0494468011345616e-10, 7.290999439594515e-10, 4.1154424470964885e-10,
          2.82566220199723e-9, 1.9419703441146337e-9, 1.0003454025331967e-9,
          1.6178718912419106e-9, 1.2355373241783204e-9]
    riskt = 0.017515177411134022
    rett = 0.001802908016949115
    @test isapprox(w48.weights, wt, rtol = 1.0e-5)
    @test isapprox(r48, riskt, rtol = 1.0e-6)
    @test isapprox(ret48, rett, rtol = 5.0e-7)

    @test isapprox(w46.weights, w47.weights)
    @test isapprox(r46, r47)
    @test isapprox(ret46, ret47)

    @test isapprox(w46.weights, w48.weights)
    @test isapprox(r46, r48)
    @test isapprox(ret46, ret48)

    @test isapprox(w47.weights, w48.weights)
    @test isapprox(r47, r48)
    @test isapprox(ret47, ret48)
end

@testset "Approx Kelly Formulations, non SD rm" begin
    portfolio = Portfolio2(; prices = prices,
                           solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                            :params => Dict("verbose" => false))))
    asset_statistics2!(portfolio)

    obj = MinRisk()
    rm = CVaR2()
    w1 = optimise2!(portfolio; rm = rm, kelly = AKelly(; formulation = SOCSD()), obj = obj)
    wt = [9.204098576587342e-11, 0.04242033344850073, 9.851178943961487e-11,
          2.742446224128441e-10, 0.007574028452634096, 1.0444829577591807e-10,
          1.1253114859467353e-11, 0.09464947950883103, 4.304498662179531e-11,
          5.769494146605776e-11, 0.3040110652564133, 5.2022107005624686e-11,
          2.8850407100827043e-11, 0.06564166930507728, 9.716009959308051e-11,
          0.02937161116309481, 1.2139968762793123e-10, 0.3663101127840467,
          5.621906727997288e-11, 0.0900216990445119]
    @test isapprox(w1.weights, wt)

    w2 = optimise2!(portfolio; rm = rm, kelly = AKelly(; formulation = QuadSD()), obj = obj)
    wt = [9.204098576587342e-11, 0.04242033344850073, 9.851178943961487e-11,
          2.742446224128441e-10, 0.007574028452634096, 1.0444829577591807e-10,
          1.1253114859467353e-11, 0.09464947950883103, 4.304498662179531e-11,
          5.769494146605776e-11, 0.3040110652564133, 5.2022107005624686e-11,
          2.8850407100827043e-11, 0.06564166930507728, 9.716009959308051e-11,
          0.02937161116309481, 1.2139968762793123e-10, 0.3663101127840467,
          5.621906727997288e-11, 0.0900216990445119]
    @test isapprox(w2.weights, wt)
    @test isapprox(w1.weights, w2.weights)

    obj = SR(; rf = rf)
    w3 = optimise2!(portfolio; rm = rm, kelly = AKelly(; formulation = SOCSD()), obj = obj)
    wt = [2.304102546507255e-9, 3.4058580184231964e-9, 3.264755599420317e-9,
          2.168529122297692e-9, 0.5593117496370377, 7.139134073206089e-10,
          0.029474976465948034, 4.9115201797004046e-8, 2.799982453416685e-9,
          1.6115667456355964e-9, 8.831047243202884e-9, 6.334407262324075e-10,
          4.1948103829488986e-10, 1.3615475408342457e-9, 4.3425958678632566e-10,
          0.20296977157601848, 0.20824336902427606, 2.80725022409301e-8,
          2.143306519713471e-8, 6.727466637284185e-9]
    @test isapprox(w3.weights, wt)

    w4 = optimise2!(portfolio; rm = rm, kelly = AKelly(; formulation = QuadSD()), obj = obj)
    wt = [2.304102546507255e-9, 3.4058580184231964e-9, 3.264755599420317e-9,
          2.168529122297692e-9, 0.5593117496370377, 7.139134073206089e-10,
          0.029474976465948034, 4.9115201797004046e-8, 2.799982453416685e-9,
          1.6115667456355964e-9, 8.831047243202884e-9, 6.334407262324075e-10,
          4.1948103829488986e-10, 1.3615475408342457e-9, 4.3425958678632566e-10,
          0.20296977157601848, 0.20824336902427606, 2.80725022409301e-8,
          2.143306519713471e-8, 6.727466637284185e-9]
    @test isapprox(w4.weights, wt)
    @test isapprox(w3.weights, w4.weights)

    obj = MinRisk()
    rm = [CDaR2(), SD2()]
    w5 = optimise2!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [3.0140842192047997e-12, 1.787134110937184e-10, 5.592260476575972e-12,
          6.419005790016778e-13, 0.003409910046771097, 1.0130887237699018e-12,
          4.34349179775756e-12, 0.07904283645064122, 6.078039691287873e-12,
          8.069382194823101e-13, 0.3875931628052873, 2.0960478141272157e-11,
          3.079574073139989e-11, 4.9577938775743915e-11, 0.0005545143505126257,
          0.0959882887370522, 0.26790886767993305, 1.0297127257935643e-10,
          0.0006560189794797459, 0.16484640054581412]
    @test isapprox(w5.weights, wt)

    rm = [CDaR2(), [SD2(), SD2()]]
    w6 = optimise2!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [1.5768872930601726e-12, 8.677442278134462e-11, 2.9153706602582253e-12,
          3.6098097550578777e-13, 0.0036748952209461295, 4.520865389736217e-13,
          2.1207996542890634e-12, 0.07937575544792427, 3.0766704137737114e-12,
          3.4211526762454803e-13, 0.3873790549560513, 1.00837174295667e-11,
          1.5385954223103806e-11, 2.4839232852770276e-11, 0.0005657309313687338,
          0.09595979975394403, 0.26758637922488315, 5.3731343771365196e-11,
          0.0006505694386978859, 0.16480781482452508]
    @test isapprox(w6.weights, wt)
    @test isapprox(w5.weights, w6.weights, rtol = 0.005)

    obj = SR(; rf = rf)
    rm = [CDaR2(), SD2()]
    w7 = optimise2!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [3.2427985505997448e-9, 6.731930327482895e-9, 0.06161721134212049,
          1.830266389068953e-9, 0.2668191260139166, 9.181506532413515e-10,
          1.2790346865644012e-9, 0.13906662770111541, 1.7852815645935868e-9,
          1.6731632092264042e-9, 0.18011362815460516, 1.1457113224256441e-9,
          6.270163379885616e-10, 2.737948937004369e-9, 9.22605081713148e-10,
          0.18045703255664325, 1.0301433006689299e-7, 4.598776005906912e-9,
          6.223032176387515e-9, 0.17192623750155375]
    @test isapprox(w7.weights, wt, rtol = 5.0e-5)

    rm = [CDaR2(), [SD2(), SD2()]]
    w8 = optimise2!(portfolio; rm = rm, kelly = AKelly(), obj = obj, str_names = true)
    wt = [9.07154948272134e-10, 1.900234492744791e-9, 0.06131325918116843,
          5.157312360844364e-10, 0.27108984023293636, 2.54127458556343e-10,
          3.548383610373354e-10, 0.14267665002427293, 4.897235118614095e-10,
          4.73363322142555e-10, 0.17325358427382076, 3.1826399542224086e-10,
          1.7170941060594656e-10, 7.728025175298101e-10, 2.5473980916498663e-10,
          0.17975392607084145, 3.048826586846149e-8, 1.3362334336357378e-9,
          1.6428340862018751e-9, 0.1719127003369377]
    @test isapprox(w8.weights, wt, rtol = 5.0e-8)
    @test isapprox(w7.weights, w8.weights, rtol = 0.05)

    obj = MinRisk()
    rm = [CVaR2(), SD2(; settings = RiskMeasureSettings(; flag = false))]
    w9 = optimise2!(portfolio; rm = rm, kelly = AKelly(; formulation = QuadSD()), obj = obj)
    wt = [9.204153687833781e-11, 0.04242033344850122, 9.85123787201143e-11,
          2.742462482595728e-10, 0.007574028452637937, 1.0444892006359854e-10,
          1.1253189456551386e-11, 0.09464947950883604, 4.304524873576078e-11,
          5.769528999444338e-11, 0.3040110652564113, 5.2022422074226226e-11,
          2.8850585492519713e-11, 0.06564166930506558, 9.716068090161583e-11,
          0.029371611163095106, 1.2140041190522942e-10, 0.366310112784046,
          5.621940710019265e-11, 0.09002169904451053]
    @test isapprox(w9.weights, wt)

    rm = [CVaR2(),
          [SD2(; settings = RiskMeasureSettings(; flag = false)),
           SD2(; settings = RiskMeasureSettings(; flag = false))]]
    w10 = optimise2!(portfolio; rm = rm, kelly = AKelly(; formulation = QuadSD()),
                     obj = obj)
    wt = [9.204153687833781e-11, 0.04242033344850122, 9.85123787201143e-11,
          2.742462482595728e-10, 0.007574028452637937, 1.0444892006359854e-10,
          1.1253189456551386e-11, 0.09464947950883604, 4.304524873576078e-11,
          5.769528999444338e-11, 0.3040110652564113, 5.2022422074226226e-11,
          2.8850585492519713e-11, 0.06564166930506558, 9.716068090161583e-11,
          0.029371611163095106, 1.2140041190522942e-10, 0.366310112784046,
          5.621940710019265e-11, 0.09002169904451053]
    @test isapprox(w10.weights, wt)
    @test isapprox(w9.weights, w10.weights)

    obj = SR(; rf = rf)
    rm = [CVaR2(), SD2(; settings = RiskMeasureSettings(; flag = false))]
    w11 = optimise2!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [2.304102546507255e-9, 3.4058580184231964e-9, 3.264755599420317e-9,
          2.168529122297692e-9, 0.5593117496370377, 7.139134073206089e-10,
          0.029474976465948034, 4.9115201797004046e-8, 2.799982453416685e-9,
          1.6115667456355964e-9, 8.831047243202884e-9, 6.334407262324075e-10,
          4.1948103829488986e-10, 1.3615475408342457e-9, 4.3425958678632566e-10,
          0.20296977157601848, 0.20824336902427606, 2.80725022409301e-8,
          2.143306519713471e-8, 6.727466637284185e-9]
    @test isapprox(w11.weights, wt)

    rm = [CVaR2(),
          [SD2(; settings = RiskMeasureSettings(; flag = false)),
           SD2(; settings = RiskMeasureSettings(; flag = false))]]
    w12 = optimise2!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [3.196566252540456e-9, 4.80237219121404e-9, 4.464886416453463e-9,
          3.0491060989617767e-9, 0.559311725652781, 1.0002259978345894e-9,
          0.029474910968437143, 6.846301937152314e-8, 3.688080874836413e-9,
          2.200427095805256e-9, 1.2783401736778063e-8, 8.785857782221683e-10,
          5.886927386343963e-10, 1.9134842812094356e-9, 6.096542976272067e-10,
          0.2029697140029372, 0.20824346355748677, 3.939026699943966e-8,
          2.8910471099005946e-8, 9.87911662830787e-9]
    @test isapprox(w12.weights, wt, rtol = 5.0e-6)
    @test isapprox(w11.weights, w12.weights, rtol = 5e-6)
end

@testset "MAD" begin
    portfolio = Portfolio2(; prices = prices,
                           solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                            :params => Dict("verbose" => false))))
    asset_statistics2!(portfolio)
    rm = MAD2()

    obj = MinRisk()
    w1 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r1 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret1 = dot(portfolio.mu, w1.weights)
    wt = [0.014061915969795006, 0.04237496202139695, 0.01686336647933223,
          0.002020806523507274, 0.01768380555270159, 0.05422405215837249,
          2.9350570130142624e-10, 0.15821651684232851, 3.0060399538100176e-10,
          7.086259738110947e-10, 0.23689725720512037, 7.61312046632753e-11,
          6.545365843921615e-11, 0.12783204733233253, 0.0003509663915665695,
          0.0009122945557616327, 0.0439493643547516, 0.18272429223715872,
          4.105696610811196e-10, 0.10188835052098438]
    riskt = 0.005627573038796034
    rett = 0.0003490122974688338
    @test isapprox(w1.weights, wt)
    @test isapprox(r1, riskt)
    @test isapprox(ret1, rett)

    w2 = optimise2!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [0.01406192655834688, 0.04237501824390749, 0.016863390017404657,
          0.0020208011519371604, 0.01768375954179011, 0.054224042598668906,
          3.860281087148557e-10, 0.15821648846151862, 3.724862305158064e-10,
          8.917492418807677e-10, 0.23689726979743447, 9.589227849839197e-11,
          8.366729059719944e-11, 0.12783207480092484, 0.0003509794124345412,
          0.0009122918961555292, 0.04394937135411301, 0.18272429219207284,
          4.845293759724182e-10, 0.10188829165893845]
    @test isapprox(w2.weights, wt)

    w3 = optimise2!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    wt = [0.01406182186298033, 0.042375224999262856, 0.016863463035875152,
          0.002020826242971577, 0.017683796862612216, 0.05422412374300622,
          6.732690796579932e-10, 0.15821648252680418, 6.380663170694303e-10,
          1.5182889035485862e-9, 0.23689723775690535, 1.7588752969970232e-10,
          1.5574094499941487e-10, 0.12783199407018078, 0.0003510013637440437,
          0.0009122576900588412, 0.04394953137073167, 0.18272434607988453,
          8.246849358651642e-10, 0.10188788840904445]
    @test isapprox(w3.weights, wt)

    obj = Util(; l = l)
    w4 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r2 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret2 = dot(portfolio.mu, w4.weights)
    wt = [0.009637742224105319, 0.05653095216430771, 0.008693798837803784,
          0.010136614699778262, 0.07101740258202667, 1.626746387899197e-10,
          0.00018717126494052322, 0.14166839870576114, 2.0925633115430728e-10,
          0.011541334039355096, 0.2033965474136089, 1.5032151500330365e-11,
          1.7660928669155618e-11, 0.0776986622152643, 5.6076568461019626e-11,
          0.018226673642440187, 0.08534261194338741, 0.17173762426959616,
          0.03444968686639962, 0.09973477867052433]
    riskt = 0.005726370460509949
    rett = 0.0005627824531830065
    @test isapprox(w4.weights, wt)
    @test isapprox(r2, riskt)
    @test isapprox(ret2, rett)

    w5 = optimise2!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [0.009549562791630994, 0.056131884099740126, 0.009058885157954702,
          0.010221145667272422, 0.07101604294130621, 6.158699951966029e-10,
          0.00017307955558545985, 0.1418306549018017, 1.0502171408125738e-9,
          0.01160511059804641, 0.20331669980076691, 4.561412945913223e-10,
          4.377177861936833e-10, 0.0778263850914995, 1.3357521459019656e-10,
          0.01821185094370143, 0.08514627046381432, 0.17137747164304867,
          0.034630959754866464, 0.09990399389544322]
    @test isapprox(w5.weights, wt)

    w6 = optimise2!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    wt = [0.009654581165225669, 0.05668321904461835, 0.008731332496008505,
          0.010033306687767333, 0.07088447280879365, 2.4996977644532297e-9,
          0.00027321456023309383, 0.14173741676314647, 3.441171402254621e-9,
          0.01168506376484469, 0.20349044392989254, 2.9458239145194473e-10,
          3.320396727480843e-10, 0.0779288019164822, 9.597391595541974e-10,
          0.018198160717452304, 0.08532649443354709, 0.17148712466334665,
          0.03410246739707899, 0.09978389212433202]
    @test isapprox(w6.weights, wt, rtol = 0.0001)

    obj = SR(; rf = rf)
    w7 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r3 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret3 = dot(portfolio.mu, w7.weights)
    wt = [3.372797021213529e-8, 8.679493419672167e-8, 8.730378695790742e-8,
          4.654016314992972e-8, 0.6621196971211604, 1.000789705313297e-8,
          0.04256386189823906, 5.0027909676906887e-8, 9.072276529043811e-8,
          4.296795445352721e-8, 7.991647846958404e-8, 7.108969143618601e-9,
          5.039720687490243e-9, 1.839999189017112e-8, 5.602046740184832e-9,
          0.1343671243475813, 0.08752271182145684, 7.258944630234996e-8,
          0.07342589536656563, 7.269496276800682e-8]
    riskt = 0.009898352231115614
    rett = 0.0015741047141763708
    @test isapprox(w7.weights, wt, rtol = 1.0e-7)
    @test isapprox(r3, riskt)
    @test isapprox(ret3, rett)

    w8 = optimise2!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [5.795290443004618e-10, 1.7735397255154201e-9, 2.4415842553008867e-9,
          7.40231745041612e-10, 0.5088109495587692, 1.6329984594581992e-10,
          0.03160667052061857, 3.0214145196781185e-9, 1.040688531056621e-9,
          6.436583441219632e-10, 0.0028683462933828748, 1.0487960563547365e-10,
          7.78113271299916e-11, 3.1960362958775614e-10, 8.512258385515617e-11,
          0.13826727169736175, 0.18565374607217008, 4.6031374812460635e-9,
          0.1327929979115769, 2.3516201199681936e-9]
    @test isapprox(w8.weights, wt, rtol = 1e-7)

    w9 = optimise2!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    wt = [1.1698348692452301e-8, 3.055254703791635e-8, 3.7017682303678675e-8,
          1.688244767018444e-8, 0.5791986473166304, 3.6199007069253584e-9,
          0.03882577851852925, 2.4972285601281053e-8, 2.70885146227947e-8,
          1.3757807183441849e-8, 4.270065859227207e-8, 2.513254702967732e-9,
          1.8448570882468499e-9, 6.737568054533749e-9, 1.9980748786998393e-9,
          0.13597861132741978, 0.13966589702878732, 3.314399410321976e-8,
          0.10633078216653338, 2.9114158748935705e-8]
    @test isapprox(w9.weights, wt, rtol = 1.0e-6)

    obj = MaxRet()
    w10 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r4 = calc_risk(portfolio; type = :Trad2, rm = rm)
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

    w11 = optimise2!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [6.24782988116139e-10, 9.366363865924148e-10, 1.6428209739519796e-9,
          1.2834078728926378e-9, 0.8503450577361338, 8.327210654169724e-10,
          0.14965492407526051, 1.8834252112272695e-11, 1.1714888587578576e-9,
          1.7599099927857186e-10, 5.734283990075868e-11, 7.855145138821309e-10,
          1.109109882692946e-9, 4.456991350071304e-10, 1.1055539398977906e-9,
          3.790544357778616e-9, 2.0750030440064227e-9, 1.2070096217342874e-10,
          1.4018002145268373e-9, 6.106531961743963e-10]
    @test isapprox(w11.weights, wt)

    w12 = optimise2!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    wt = [6.757093243027148e-9, 7.664031309161123e-9, 9.442178874312985e-9,
          8.584900291197654e-9, 0.8533948964966961, 3.0825110840326666e-9,
          0.1466049837308141, 5.287317105823606e-9, 8.388979848897765e-9,
          5.749959272436911e-9, 5.103146714031344e-9, 3.205393470435072e-9,
          2.2935857006942965e-9, 4.081417604988999e-9, 2.3373314078613168e-9,
          1.5619143210865274e-8, 1.0720641149855979e-8, 5.557609448709485e-9,
          9.024863752987753e-9, 6.872386298380374e-9]
    @test isapprox(w12.weights, wt, rtol = 1.0e-6)

    # Risk upper bound
    obj = MaxRet()
    rm.settings.ub = r1
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r1 ||
          abs(calc_risk(portfolio; type = :Trad2, rm = rm) - r1) < 1e-9

    rm.settings.ub = r2
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r2

    rm.settings.ub = r3
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r3

    rm.settings.ub = r4
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r4

    obj = SR(; rf = rf)
    rm.settings.ub = r1
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r1 ||
          abs(calc_risk(portfolio; type = :Trad2, rm = rm) - r1) < 1e-10

    rm.settings.ub = r2
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r2

    rm.settings.ub = r3
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r3

    rm.settings.ub = r4
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r4

    # Ret lower bound
    rm.settings.ub = Inf
    obj = MinRisk()
    portfolio.mu_l = ret1
    w13 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w13.weights) >= ret1

    portfolio.mu_l = ret2
    w14 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w14.weights) >= ret2

    portfolio.mu_l = ret3
    w15 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w15.weights) >= ret3

    portfolio.mu_l = ret4
    w16 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w16.weights) >= ret4 ||
          abs(dot(portfolio.mu, w16.weights) - ret4) < 1e-10

    obj = SR(; rf = rf)
    portfolio.mu_l = ret1
    w17 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w17.weights) >= ret1

    portfolio.mu_l = ret2
    w18 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w18.weights) >= ret2

    portfolio.mu_l = ret3
    w19 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w19.weights) >= ret3

    portfolio.mu_l = ret4
    w20 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w20.weights) >= ret4 ||
          abs(dot(portfolio.mu, w16.weights) - ret4) < 1e-10
end

@testset "SSD" begin
    portfolio = Portfolio2(; prices = prices,
                           solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                            :params => Dict("verbose" => false))))
    asset_statistics2!(portfolio)
    rm = SSD2()

    obj = MinRisk()
    w1 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r1 = calc_risk(portfolio; type = :Trad2, rm = rm)
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

    w2 = optimise2!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [1.328560973467273e-8, 0.049738240342198585, 6.729638177718146e-9,
          0.0029785880061378384, 0.002551194236638699, 0.02013119386894698,
          3.7241820112123204e-10, 0.1280950127330832, 1.1193171935929993e-9,
          1.5313119883776913e-9, 0.2995770952417976, 1.6188140270762791e-9,
          6.955937873270665e-10, 0.12069621604817828, 0.012266360319875118,
          0.009662882398733882, 8.34506433718684e-9, 0.22927554413792217,
          1.4212376700688693e-9, 0.12502763754748242]
    @test isapprox(w2.weights, wt)

    w3 = optimise2!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    wt = [2.5190879083102043e-8, 0.049737774196137965, 1.2868124743982513e-8,
          0.0029783638332052057, 0.0025509173649045694, 0.020133768670666973,
          9.047615243063875e-10, 0.1280950304816767, 2.3106803351985052e-9,
          3.086425946402379e-9, 0.2995771396012498, 3.251604912185191e-9,
          1.5132089628845487e-9, 0.12069584048889491, 0.012266113001483253,
          0.009662762604571083, 1.5907937864516914e-8, 0.2292749772563436,
          2.8790402254376556e-9, 0.12502724458820227]
    @test isapprox(w3.weights, wt)

    obj = Util(; l = l)
    w4 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r2 = calc_risk(portfolio; type = :Trad2, rm = rm)
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

    w5 = optimise2!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [5.310366383671854e-10, 0.055185496756761133, 4.958251729556289e-10,
          0.004440395111615488, 0.033778656762619236, 4.711290484606268e-10,
          1.4995099027754112e-10, 0.12528998427061558, 8.369500406278012e-12,
          2.3638500512437794e-11, 0.2983971162306186, 1.031575927125493e-10,
          1.5846481226439663e-10, 0.10741032813109379, 0.002725697131388786,
          0.021069728462151726, 0.005363035530923987, 0.2288344439497473,
          9.738524268054006e-11, 0.11750511562350689]
    @test isapprox(w5.weights, wt)

    w6 = optimise2!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    wt = [2.2895674946199656e-9, 0.05518903108297828, 2.1836060733181756e-9,
          0.0044374629070126565, 0.033766396900666275, 2.110869140807151e-9,
          2.544160805002686e-10, 0.12529117631048525, 6.773571588439321e-10,
          7.730249106515778e-10, 0.2984002871957823, 3.943554094652132e-10,
          2.2905478935733912e-10, 0.10741594289180213, 0.0027306359963794984,
          0.02106510870973968, 0.005352552882926046, 0.2288401489338605,
          9.93196861710178e-10, 0.11751124628291945]
    @test isapprox(w6.weights, wt, rtol = 1.0e-6)

    obj = SR(; rf = rf)
    w7 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r3 = calc_risk(portfolio; type = :Trad2, rm = rm)
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

    w8 = optimise2!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [1.1686179891013613e-8, 3.7143021536201827e-8, 2.301711086719213e-8,
          1.389322574147696e-8, 0.566054323906044, 3.798547414540005e-9,
          0.02904414098555804, 5.047834297132999e-8, 1.9806257217757055e-8,
          1.1826573374560461e-8, 1.380045456737115e-7, 2.885310238914747e-9,
          2.101052458639916e-9, 8.171076474830977e-9, 2.189455584169703e-9,
          0.164249355676866, 0.1629106816795142, 7.659812295832371e-8, 0.07774106421141706,
          3.1941778285711126e-8]
    @test isapprox(w8.weights, wt, rtol = 5e-7)

    w9 = optimise2!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    wt = [2.3079619465821097e-7, 5.87719458710357e-7, 4.4740668517610773e-7,
          2.8029700495576643e-7, 0.6080716728391783, 7.731151243684516e-8,
          0.03582104571289915, 5.413757919342572e-7, 4.787969851327624e-7,
          2.2930662515647874e-7, 7.435354777738109e-7, 6.176625839188381e-8,
          4.43431803073848e-8, 1.5413323851090924e-7, 4.610353331379092e-8,
          0.16672108503331592, 0.13724564769278624, 7.191502039920105e-7,
          0.0521354129969113, 4.936827587037762e-7]
    @test isapprox(w9.weights, wt)

    obj = MaxRet()
    w10 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r4 = calc_risk(portfolio; type = :Trad2, rm = rm)
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

    w11 = optimise2!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [1.530417939183873e-10, 2.8612308735562203e-10, 5.505656045721553e-10,
          4.5496556945069153e-10, 0.8503475681489141, 3.8305099849033153e-10,
          0.1496524260843989, 7.250692125835878e-11, 3.862370077496164e-10,
          1.7288373304633485e-11, 9.181697560252934e-11, 3.404667655856662e-10,
          3.876849785161851e-10, 2.3474971482803645e-10, 4.1508395195208027e-10,
          6.184903168160772e-10, 7.002854291572307e-10, 3.1063304351626673e-11,
          4.812277872084391e-10, 1.6203868518158558e-10]
    @test isapprox(w11.weights, wt)

    w12 = optimise2!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    wt = [2.4198206282739137e-9, 2.7523773137294424e-9, 3.4600331962305982e-9,
          3.1089170165506684e-9, 0.8533948120145702, 9.622378649372175e-10,
          0.14660514506394484, 1.8389581632770744e-9, 2.9995402556500355e-9,
          2.0242460469329184e-9, 1.7817450751121457e-9, 1.0985267193180098e-9,
          7.712326645258262e-10, 1.3809357105377707e-9, 8.057118379005105e-10,
          6.0398082300262525e-9, 3.896463510594108e-9, 1.9253519524735756e-9,
          3.2097224949880373e-9, 2.445856342069014e-9]
    @test isapprox(w12.weights, wt, rtol = 1.0e-6)

    # Risk upper bound
    obj = MaxRet()
    rm.settings.ub = r1
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r1 ||
          abs(calc_risk(portfolio; type = :Trad2, rm = rm) - r1) < 1e-9

    rm.settings.ub = r2
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r2

    rm.settings.ub = r3
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r3

    rm.settings.ub = r4
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r4

    obj = SR(; rf = rf)
    rm.settings.ub = r1
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r1 ||
          abs(calc_risk(portfolio; type = :Trad2, rm = rm) - r1) < 1e-8

    rm.settings.ub = r2
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r2

    rm.settings.ub = r3
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r3

    rm.settings.ub = r4
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r4

    # Ret lower bound
    rm.settings.ub = Inf
    obj = MinRisk()
    portfolio.mu_l = ret1
    w13 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w13.weights) >= ret1

    portfolio.mu_l = ret2
    w14 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w14.weights) >= ret2

    portfolio.mu_l = ret3
    w15 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w15.weights) >= ret3

    portfolio.mu_l = ret4
    w16 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w16.weights) >= ret4 ||
          abs(dot(portfolio.mu, w16.weights) - ret4) < 1e-10

    obj = SR(; rf = rf)
    portfolio.mu_l = ret1
    w17 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w17.weights) >= ret1

    portfolio.mu_l = ret2
    w18 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w18.weights) >= ret2

    portfolio.mu_l = ret3
    w19 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w19.weights) >= ret3

    portfolio.mu_l = ret4
    w20 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w20.weights) >= ret4 ||
          abs(dot(portfolio.mu, w16.weights) - ret4) < 1e-10
end

@testset "FLPM" begin
    portfolio = Portfolio2(; prices = prices,
                           solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                            :params => Dict("verbose" => false))))
    asset_statistics2!(portfolio)
    rm = FLPM2(; target = rf)

    obj = MinRisk()
    w1 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r1 = calc_risk(portfolio; type = :Trad2, rm = rm)
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

    w2 = optimise2!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [0.00426610239591103, 0.04362178853662362, 0.019960457154756087,
          0.007822768508828756, 0.06052527699807257, 4.506186395476567e-10,
          0.0003959578648306813, 0.1308926705227917, 1.8600603149618773e-10,
          0.011878924975411324, 0.2066096337189848, 1.5259036251172595e-11,
          1.466852985609556e-11, 0.08329480736550154, 4.0265597843607914e-11,
          0.013888097943043014, 0.08734723725306878, 0.19210036041488635,
          0.037213199462145435, 0.10018271617832634]
    @test isapprox(w2.weights, wt)

    w3 = optimise2!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    wt = [0.00426608716920985, 0.04362173641258839, 0.019960444305697746,
          0.007822767144185885, 0.06052541421131514, 3.4024764351128286e-9,
          0.00039592869717005593, 0.13089257106535185, 1.263061291396848e-9,
          0.0118788599251899, 0.20660956985823367, 1.2640723567497645e-10,
          1.224400506824836e-10, 0.08329482612251998, 2.9820592088268585e-10,
          0.013888098156328821, 0.08734710673926958, 0.19210049684132757,
          0.037213161399705486, 0.1001829267393152]
    @test isapprox(w3.weights, wt)

    obj = Util(; l = l)
    w4 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r2 = calc_risk(portfolio; type = :Trad2, rm = rm)
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

    w5 = optimise2!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [7.025580903339914e-9, 0.03801909891801268, 0.015508772801692194,
          0.007920175117119383, 0.09985335454405923, 2.5401009772523425e-10,
          0.0024873633006595353, 0.10503236887549107, 8.540598148869465e-10,
          4.653369605256221e-9, 0.203007892809087, 7.000508823160763e-10,
          6.910321683931141e-10, 0.035031103036583196, 5.676257759160685e-10,
          0.03798687241385029, 0.10539179463937084, 0.1777681655105699, 0.06295784178579301,
          0.10903518150198238]
    @test isapprox(w5.weights, wt)

    w6 = optimise2!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    wt = [1.2872334765954068e-8, 0.03767953016874954, 0.015470385826160825,
          0.0081170648205414, 0.09892766032152121, 1.0417936519167395e-9,
          0.0025348401228208013, 0.10487955021910977, 2.8917086673976536e-9,
          8.851185748932032e-9, 0.20310090288185126, 3.0551749498443944e-10,
          3.161080085781991e-10, 0.03555872480929, 5.151922419610025e-10,
          0.03730024048937033, 0.10541030477402333, 0.17862267733167023,
          0.06248696950024344, 0.10991112194080724]
    @test isapprox(w6.weights, wt, rtol = 5.0e-5)

    obj = SR(; rf = rf)
    w7 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r3 = calc_risk(portfolio; type = :Trad2, rm = rm)
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

    w8 = optimise2!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [5.8071005976616953e-11, 1.541427902953817e-10, 1.7913932166632589e-10,
          7.782119293420651e-11, 0.5637246749896364, 1.8931437657935694e-11,
          0.026029768151395943, 1.9414490982830922e-10, 1.259249686191089e-10,
          7.208065397311031e-11, 1.1199492525822813e-9, 1.215314166911757e-11,
          9.41885824068697e-12, 3.5763702301181766e-11, 1.0455689250552838e-11,
          0.14917727800463884, 0.12978984415928974, 3.9155754084852714e-10,
          0.13127843200243125, 2.330533005896693e-10]
    @test isapprox(w8.weights, wt)

    w9 = optimise2!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    wt = [5.450446471466287e-9, 1.3512126713442444e-8, 1.5151665151427853e-8,
          7.976975650100931e-9, 0.6065958578622958, 1.8554190775039432e-9,
          0.028288084366048082, 1.1813144626604747e-8, 1.4369857514985968e-8,
          6.665738536162035e-9, 1.8203193050019052e-8, 1.257919516215051e-9,
          9.552968174589085e-10, 3.3556538128122655e-9, 1.0433283283023054e-9,
          0.14977639360164002, 0.1082338279016761, 1.701911449061477e-8,
          0.10710570148059825, 1.6157861947293164e-8]
    @test isapprox(w9.weights, wt, rtol = 5.0e-5)

    obj = MaxRet()
    w10 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r4 = calc_risk(portfolio; type = :Trad2, rm = rm)
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

    w11 = optimise2!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [6.184818023388107e-10, 9.270209690896355e-10, 1.625727765129299e-9,
          1.2702031382871368e-9, 0.8503448667005368, 8.234230282459137e-10,
          0.14965511529944078, 1.8812024811068833e-11, 1.1594526094497495e-9,
          1.7436144865199502e-10, 5.659330075801571e-11, 7.768388767333282e-10,
          1.0972252733499826e-9, 4.4072771043489173e-10, 1.0935450396733876e-9,
          3.753052447991426e-9, 2.05335213852063e-9, 1.1963791294032173e-10,
          1.3871915795314854e-9, 6.043753072340663e-10]
    @test isapprox(w11.weights, wt)

    w12 = optimise2!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    wt = [1.0436536369860602e-9, 1.1989953779953557e-9, 1.464622032811774e-9,
          1.3427482165971476e-9, 0.8533952746506102, 4.935314543235269e-10,
          0.14660470645477996, 8.379594796066728e-10, 1.3262258615336055e-9,
          9.158610967401738e-10, 8.121781642332045e-10, 5.143517373531351e-10,
          3.696751986345881e-10, 6.50560988974405e-10, 3.7768409279100334e-10,
          2.4590065299397058e-9, 1.6858266276584893e-9, 8.793941840416185e-10,
          1.4336339581766902e-9, 1.0887011785012557e-9]
    @test isapprox(w12.weights, wt, rtol = 1.0e-6)

    # Risk upper bound
    obj = MaxRet()
    rm.settings.ub = r1
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r1 ||
          abs(calc_risk(portfolio; type = :Trad2, rm = rm) - r1) < 1e-9

    rm.settings.ub = r2
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r2

    rm.settings.ub = r3
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r3

    rm.settings.ub = r4
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r4

    obj = SR(; rf = rf)
    rm.settings.ub = r1
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r1 ||
          abs(calc_risk(portfolio; type = :Trad2, rm = rm) - r1) < 1e-10

    rm.settings.ub = r2
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r2

    rm.settings.ub = r3
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r3

    rm.settings.ub = r4
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r4

    # Ret lower bound
    rm.settings.ub = Inf
    obj = MinRisk()
    portfolio.mu_l = ret1
    w13 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w13.weights) >= ret1

    portfolio.mu_l = ret2
    w14 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w14.weights) >= ret2

    portfolio.mu_l = ret3
    w15 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w15.weights) >= ret3

    portfolio.mu_l = ret4
    w16 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w16.weights) >= ret4 ||
          abs(dot(portfolio.mu, w16.weights) - ret4) < 1e-10

    obj = SR(; rf = rf)
    portfolio.mu_l = ret1
    w17 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w17.weights) >= ret1

    portfolio.mu_l = ret2
    w18 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w18.weights) >= ret2

    portfolio.mu_l = ret3
    w19 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w19.weights) >= ret3

    portfolio.mu_l = ret4
    w20 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w20.weights) >= ret4 ||
          abs(dot(portfolio.mu, w16.weights) - ret4) < 1e-10
end

@testset "SLPM" begin
    portfolio = Portfolio2(; prices = prices,
                           solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                            :params => Dict("verbose" => false))))
    asset_statistics2!(portfolio)
    rm = SLPM2(; target = rf)

    obj = MinRisk()
    w1 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r1 = calc_risk(portfolio; type = :Trad2, rm = rm)
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

    w2 = optimise2!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [3.553751623019336e-9, 0.0552444506714033, 3.3792972175226587e-9,
          0.004319267700904996, 0.033597486252238504, 3.3178770246587074e-9,
          3.530263895833413e-10, 0.12478100673912003, 1.082173144802101e-9,
          1.2142739606639318e-9, 0.3005650138904127, 6.172236686305243e-10,
          3.272670368272328e-10, 0.10661801005424905, 0.003123621247828574,
          0.02139180763720946, 0.0035949221271092354, 0.22965076223179196,
          1.6119446394023392e-9, 0.11711363599089746]
    @test isapprox(w2.weights, wt)

    w3 = optimise2!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    wt = [8.960032986198268e-9, 0.05524502934743527, 8.531490753416422e-9,
          0.004319426319869003, 0.03359614618594921, 8.379240639976445e-9,
          1.0539120789014738e-9, 0.12478035258171694, 2.855174321889051e-9,
          3.1807196110268267e-9, 0.30056524386869504, 1.706724783203312e-9,
          9.907199378298648e-10, 0.10661840631750075, 0.003124219404851094,
          0.021391583878468307, 0.0035949097819517437, 0.22965048214875944,
          4.163566475384283e-9, 0.11711416034322167]
    @test isapprox(w3.weights, wt)

    obj = Util(; l = l)
    w4 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r2 = calc_risk(portfolio; type = :Trad2, rm = rm)
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

    w5 = optimise2!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [1.1179343918984971e-10, 0.054976790588896596, 1.7713391003665344e-10,
          0.000633725765454258, 0.06550332883993616, 2.3043194331626305e-11,
          9.784822751076497e-11, 0.11941492846830355, 2.6615472963293778e-11,
          1.8716678076476947e-11, 0.2937827494968045, 1.1133950578913522e-10,
          1.37189305281988e-10, 0.0764029089634452, 3.804971672707788e-11,
          0.03435573176652704, 0.028126826103343827, 0.2230959623916138,
          2.3934818713608943e-10, 0.10370704663459744]
    @test isapprox(w5.weights, wt)

    w6 = optimise2!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    wt = [2.909374492519179e-9, 0.054980576422244756, 3.565867766679885e-9,
          0.0006319533873603366, 0.06548847846241447, 1.553456588968246e-9,
          8.009894415787075e-10, 0.11941669348400365, 2.0520045922813665e-9,
          1.972894188888827e-9, 0.29378579258366055, 6.654042713665044e-10,
          4.054332929406695e-10, 0.0764175574974796, 2.168836496219484e-9,
          0.03435048211474188, 0.028114016486312555, 0.22310042002755265,
          4.189571372445089e-9, 0.10371400925039707]
    @test isapprox(w6.weights, wt, rtol = 5.0e-7)

    obj = SR(; rf = rf)
    w7 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r3 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret3 = dot(portfolio.mu, w7.weights)
    wt = [1.9161992029945534e-8, 4.442467014160941e-8, 3.487731975169222e-8,
          2.172326848650473e-8, 0.6654321506924412, 6.20892532022181e-9,
          0.03807260712526902, 3.6516022610300514e-8, 4.3159008520930105e-8,
          1.8350537901763542e-8, 4.619460482355355e-8, 5.0197040711936325e-9,
          3.3977158843464672e-9, 1.2834736295215969e-8, 3.5853236437253736e-9,
          0.17459230019672953, 0.10412390455189192, 4.844528935490425e-8,
          0.017778656482209734, 3.7052339729479755e-8]
    riskt = 0.00909392522496688
    rett = 0.0015869580721210722
    @test isapprox(w7.weights, wt)
    @test isapprox(r3, riskt)
    @test isapprox(ret3, rett)

    w8 = optimise2!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [1.3248921152324648e-8, 4.085757001568184e-8, 2.5052363399873344e-8,
          1.543168580522427e-8, 0.5737596378553825, 4.3480211803596126e-9,
          0.029566570892699606, 5.302664319377606e-8, 2.2678880685535206e-8,
          1.3224044271395122e-8, 1.1753423586262955e-7, 3.3659140729627245e-9,
          2.441172159950741e-9, 9.365453914041405e-9, 2.532713913601762e-9,
          0.1671874274459401, 0.15962776043775584, 7.614872045448292e-8,
          0.06985816973821751, 3.437366426339058e-8]
    @test isapprox(w8.weights, wt, rtol = 1e-6)

    w9 = optimise2!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    wt = [2.956369706502196e-7, 7.203516650576015e-7, 5.527631754013819e-7,
          3.5624630813868357e-7, 0.6120717648980301, 1.0106563076307954e-7,
          0.03616496177959474, 6.730914491132973e-7, 5.529892319617771e-7,
          2.8221149925910375e-7, 9.319704068880028e-7, 8.419900388670793e-8,
          6.018647280784777e-8, 1.9489436736726285e-7, 6.199574934187237e-8,
          0.1695100352935956, 0.13598076586849808, 8.915232281848942e-7, 0.0462661197643809,
          5.932707417147085e-7]
    @test isapprox(w9.weights, wt, rtol = 1.0e-7)

    obj = MaxRet()
    w10 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r4 = calc_risk(portfolio; type = :Trad2, rm = rm)
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

    w11 = optimise2!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [7.210116063134277e-11, 1.176606228072633e-10, 2.1258483097832237e-10,
          1.6536274651174382e-10, 0.8503141905762418, 1.273544562974361e-10,
          0.14968580705048062, 1.992517244424829e-12, 1.5128034960181728e-10,
          1.850938653797299e-11, 1.2078508927108743e-11, 1.1804552928693339e-10,
          1.5680771674497215e-10, 6.957317957422098e-11, 1.6101143219670432e-10,
          4.425491709419713e-10, 2.7404957392346757e-10, 9.559038664076399e-12,
          1.8545492666656794e-10, 7.730240246200862e-11]
    @test isapprox(w11.weights, wt)

    w12 = optimise2!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    wt = [2.4129797561271965e-9, 2.701958395018328e-9, 3.3576629746257995e-9,
          3.0274737839898444e-9, 0.8533951483772857, 1.0606695596971499e-9,
          0.14660480975447185, 1.8393387540596498e-9, 2.9260550631872547e-9,
          1.989898450820525e-9, 1.7675044799733574e-9, 1.100567325980679e-9,
          7.837356706523964e-10, 1.4112632681808324e-9, 7.965335460056496e-10,
          5.471987951034473e-9, 3.760034250575122e-9, 1.9365276473399495e-9,
          3.1320916121578367e-9, 2.3919599353675564e-9]
    @test isapprox(w12.weights, wt, rtol = 1.0e-7)

    # Risk upper bound
    obj = MaxRet()
    rm.settings.ub = r1
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r1 ||
          abs(calc_risk(portfolio; type = :Trad2, rm = rm) - r1) < 1e-9

    rm.settings.ub = r2
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r2

    rm.settings.ub = r3
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r3

    rm.settings.ub = r4
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r4

    obj = SR(; rf = rf)
    rm.settings.ub = r1 * (1.000001)
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r1 ||
          abs(calc_risk(portfolio; type = :Trad2, rm = rm) - r1) < 1e-8

    rm.settings.ub = r2
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r2

    rm.settings.ub = r3
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r3

    rm.settings.ub = r4
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r4

    # Ret lower bound
    rm.settings.ub = Inf
    obj = MinRisk()
    portfolio.mu_l = ret1
    w13 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w13.weights) >= ret1

    portfolio.mu_l = ret2
    w14 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w14.weights) >= ret2

    portfolio.mu_l = ret3
    w15 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w15.weights) >= ret3

    portfolio.mu_l = ret4
    w16 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w16.weights) >= ret4 ||
          abs(dot(portfolio.mu, w16.weights) - ret4) < 1e-10

    obj = SR(; rf = rf)
    portfolio.mu_l = ret1
    w17 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w17.weights) >= ret1

    portfolio.mu_l = ret2
    w18 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w18.weights) >= ret2

    portfolio.mu_l = ret3
    w19 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w19.weights) >= ret3

    portfolio.mu_l = ret4
    w20 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w20.weights) >= ret4 ||
          abs(dot(portfolio.mu, w16.weights) - ret4) < 1e-10
end
