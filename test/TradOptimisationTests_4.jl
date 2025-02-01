@testset "GMD" begin
    portfolio = Portfolio(; prices = prices[(end - 200):end],
                          solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                           :check_sol => (allow_local = true,
                                                                          allow_almost = true),
                                                           :params => Dict("verbose" => false,
                                                                           "max_step_fraction" => 0.75,
                                                                           "max_iter" => 100,
                                                                           "equilibrate_max_iter" => 20))))
    asset_statistics!(portfolio)
    rm = GMD(; formulation = OWAExact())

    obj = MinRisk()
    w1 = optimise!(portfolio, Trad(; rm = rm, kelly = NoKelly(), obj = obj))
    r1 = calc_risk(portfolio, :Trad; rm = rm)
    ret1 = dot(portfolio.mu, w1.weights)
    wt = [2.860021085782162e-10, 1.2840309896218056e-9, 6.706876349584523e-10,
          0.011313354703334149, 0.0411883749620692, 0.021164889458692743,
          9.703898006880258e-11, 0.047274318622891866, 5.765740916268004e-10,
          4.570883347699393e-10, 0.07079108352937005, 2.3692512642792012e-11,
          6.86255931091155e-11, 0.29261593911138273, 2.6280037229769384e-11,
          0.014804770495317407, 0.055360406768988304, 0.22572329075836045,
          0.008097367714011027, 0.21166620038556183]
    riskt = 0.007973013868952676
    rett = 0.00029353687778951485
    @test isapprox(w1.weights, wt, rtol = 5.0e-6)
    @test isapprox(r1, riskt)
    @test isapprox(ret1, rett, rtol = 1.0e-6)

    w2 = optimise!(portfolio, Trad(; rm = rm, kelly = AKelly(), obj = obj))
    wt = [5.250702616813578e-10, 2.3649724723993027e-9, 1.2279491916198084e-9,
          0.01136623473742439, 0.0411504017712101, 0.02111853666359142,
          1.7823656285034675e-10, 0.04726442039789345, 1.0571802875720308e-9,
          8.38512874634768e-10, 0.07085086111257587, 4.4206112295333334e-11,
          1.2541463432371463e-10, 0.29253129575787246, 4.881312822213951e-11,
          0.014796400914986178, 0.05529639972264038, 0.22572814198254204,
          0.008121076155090038, 0.2117762243738181]
    @test isapprox(w2.weights, wt, rtol = 1.0e-6)

    w3 = optimise!(portfolio, Trad(; rm = rm, kelly = EKelly(), obj = obj))
    wt = [5.323839510142863e-10, 2.229192435645185e-9, 1.1712440903830328e-9,
          0.011344721661516587, 0.04116337911244441, 0.02111829239021385,
          2.1641304918219377e-10, 0.047255888185545984, 1.005291628173614e-9,
          8.091650681427752e-10, 0.07082525289366046, 9.26602870015525e-11,
          1.6964206457835295e-10, 0.29255890224533354, 9.639341758984514e-11,
          0.014800853997633014, 0.055321508358536234, 0.2257384135285236,
          0.008118509287753244, 0.211754272016453]
    @test isapprox(w3.weights, wt, rtol = 5.0e-6)

    obj = Utility(; l = l)
    w4 = optimise!(portfolio, Trad(; rm = rm, kelly = NoKelly(), obj = obj))
    r2 = calc_risk(portfolio, :Trad; rm = rm)
    ret2 = dot(portfolio.mu, w4.weights)
    wt = [4.395442944440727e-10, 2.1495615257047445e-9, 8.700158416075159e-10,
          0.006589303508967539, 0.05330845589356504, 7.632737386668005e-10,
          9.831583865942048e-11, 0.06008335161013898, 1.3797121555924764e-9,
          3.0160588760343415e-9, 0.06394372429204083, 4.486199590323317e-11,
          1.0975948346699443e-10, 0.24314317639043878, 5.90094645737884e-11,
          0.018483948217979402, 0.08594753958504305, 0.216033164253688, 0.05666930366548,
          0.19579802365254514]
    riskt = 0.008023853567234033
    rett = 0.0005204190203575007
    @test isapprox(w4.weights, wt, rtol = 1.0e-6)
    @test isapprox(r2, riskt)
    @test isapprox(ret2, rett, rtol = 5.0e-8)

    w5 = optimise!(portfolio, Trad(; rm = rm, kelly = AKelly(), obj = obj))
    wt = [2.1362512461147959e-10, 1.0473426830174378e-9, 4.232904776563241e-10,
          0.006623775536322704, 0.053261985949698866, 3.7469349330098336e-10,
          4.858498678278164e-11, 0.05983738243189082, 6.626203326440813e-10,
          1.432599593268026e-9, 0.06407652576443064, 2.2558984339781933e-11,
          5.3690642468248475e-11, 0.24323109220603928, 2.96486691492619e-11,
          0.018454184890627775, 0.08594373007217042, 0.21613806337839309,
          0.05652374385950317, 0.1959095116022682]
    @test isapprox(w5.weights, wt, rtol = 5.0e-6)

    w6 = optimise!(portfolio, Trad(; rm = rm, kelly = EKelly(), obj = obj))
    wt = [5.937788173616308e-10, 2.7035923440469955e-9, 1.1184215485237802e-9,
          0.006640033798649065, 0.05321801224326745, 9.963309538802194e-10,
          1.8376772950076455e-10, 0.05964942314439764, 1.7115613152193195e-9,
          3.697169350159616e-9, 0.06413044858686795, 1.187496479279526e-10,
          1.9736634980204724e-10, 0.24337338790443755, 1.362995356534818e-10,
          0.018577394541992927, 0.0858912259147116, 0.216090928601388, 0.056732714784793,
          0.19569641902245713]
    @test isapprox(w6.weights, wt, rtol = 0.005)

    obj = Sharpe(; rf = rf)
    w7 = optimise!(portfolio, Trad(; rm = rm, kelly = NoKelly(), obj = obj))
    r3 = calc_risk(portfolio, :Trad; rm = rm)
    ret3 = dot(portfolio.mu, w7.weights)
    wt = [1.0467205635658914e-12, 4.278912675065536e-11, 3.374093991319961e-12,
          3.8277639532838844e-11, 0.21079296400188735, 2.4364045721705662e-11,
          2.2390341906149484e-11, 0.01106006895313273, 0.0215579625569257,
          1.5262610412144913e-10, 2.1500846791960303e-11, 1.6225951152430356e-11,
          2.272395485557818e-11, 1.011545112756681e-11, 1.8686814475829852e-11,
          0.09771827802310681, 0.3985981897681632, 5.508078066235759e-11,
          0.26027253624006286, 2.7519584077987032e-11]
    riskt = 0.010781870841767164
    rett = 0.001659046007257454
    @test isapprox(w7.weights, wt)
    @test isapprox(r3, riskt)
    @test isapprox(ret3, rett)

    w8 = optimise!(portfolio, Trad(; rm = rm, kelly = AKelly(), obj = obj))
    wt = [1.0920644565612393e-10, 2.5303831383802556e-10, 1.2376306976438313e-10,
          2.369564354076823e-10, 0.20616254959454816, 3.202182206033137e-11,
          3.971613464382764e-11, 0.015569075031954824, 0.01601461718150225,
          6.237669799218814e-10, 1.8776259246956673e-10, 6.016603087271084e-11,
          3.871362662544616e-11, 1.4873125031397894e-10, 5.163281350655045e-11,
          0.09845408940920172, 0.39772987513121644, 3.0052638214680857e-10,
          0.266069791233974, 2.1160070552220802e-10]
    @test isapprox(w8.weights, wt)

    w9 = optimise!(portfolio, Trad(; rm = rm, kelly = EKelly(), obj = obj))
    wt = [3.14905351889511e-7, 7.08763348846685e-7, 3.538581511420651e-7,
          6.876632506556505e-7, 0.20716142258876596, 9.623479730547593e-8,
          1.2169267976274955e-7, 0.015607749851993332, 0.012493885188245756,
          1.6035402196119575e-6, 4.965350735695629e-7, 1.8143160353564277e-7,
          1.1505668719872142e-7, 4.0369243757166665e-7, 1.5047808406371426e-7,
          0.098197585753003, 0.39812133395490806, 7.72480185603894e-7, 0.26841146641647695,
          5.499147363025267e-7]
    @test isapprox(w9.weights, wt, rtol = 0.005)

    obj = MaxRet()
    w10 = optimise!(portfolio, Trad(; rm = rm, kelly = NoKelly(), obj = obj))
    r4 = calc_risk(portfolio, :Trad; rm = rm)
    ret4 = dot(portfolio.mu, w10.weights)
    wt = [8.713294565329801e-10, 1.4655494663368051e-9, 9.339861464983668e-10,
          2.1822916966782698e-9, 0.9999999341529429, 3.1514150951638473e-10,
          4.623635446692933e-10, 1.3314213108887623e-9, 5.686161252843585e-9,
          1.371922705853848e-9, 7.146550790353018e-10, 5.740256406691589e-10,
          3.664169883135698e-10, 7.000039902341564e-10, 4.5030867400627253e-10,
          6.996286227848405e-9, 3.6287634044926756e-8, 9.000571133076656e-10,
          3.4765700831013475e-9, 7.60932312745152e-10]
    riskt = 0.017170975178026397
    rett = 0.0019086272315059051
    @test isapprox(w10.weights, wt)
    @test isapprox(r4, riskt)
    @test isapprox(ret4, rett)

    w11 = optimise!(portfolio, Trad(; rm = rm, kelly = AKelly(), obj = obj))
    wt = [5.995668682294329e-12, 1.778824643654798e-11, 7.422520403984214e-12,
          2.585579633200434e-11, 0.8577737311670789, 5.0842729898340874e-12,
          1.891888971224657e-12, 1.7873758325289183e-11, 9.127375624650092e-11,
          1.8171912621743217e-11, 3.850582248871377e-12, 2.5508491084881004e-13,
          4.021631383731875e-12, 3.3391228496613597e-12, 2.0662267001955667e-12,
          1.856978015488069e-10, 0.14222626837308944, 8.073644265708836e-12,
          5.656088636719738e-11, 4.608688065458597e-12]
    @test isapprox(w11.weights, wt)

    w12 = optimise!(portfolio, Trad(; rm = rm, kelly = EKelly(), obj = obj))
    wt = [4.40859451333247e-9, 7.864888444121469e-9, 4.612603900429012e-9,
          1.0349688665135194e-8, 0.892636677290951, 1.4440413646088198e-9,
          1.968066871845967e-9, 7.631330480424597e-9, 3.4295274358342415e-8,
          7.454362493254684e-9, 3.6914967896364635e-9, 2.6763326762619786e-9,
          1.5289402772999891e-9, 3.5264095931927543e-9, 1.8905286042675526e-9,
          6.13629141224752e-8, 0.10736313857685581, 4.827944431112196e-9,
          2.0764556150948145e-8, 3.834219335175887e-9]
    @test isapprox(w12.weights, wt, rtol = 1.0e-5)

    # Risk upper bound
    obj = MaxRet()
    rm.settings.ub = r1
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio, :Trad; rm = rm) <= r1 ||
          abs(calc_risk(portfolio, :Trad; rm = rm) - r1) < 5e-7

    rm.settings.ub = r2
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio, :Trad; rm = rm) <= r2 ||
          abs(calc_risk(portfolio, :Trad; rm = rm) - r2) < 5e-8

    rm.settings.ub = r3
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio, :Trad; rm = rm) <= r3 ||
          abs(calc_risk(portfolio, :Trad; rm = rm) - r3) < 5e-9

    rm.settings.ub = r4
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio, :Trad; rm = rm) <= r4 ||
          abs(calc_risk(portfolio, :Trad; rm = rm) - r4) < 1e-9

    obj = Sharpe(; rf = rf)
    rm.settings.ub = r1
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio, :Trad; rm = rm) <= r1 ||
          abs(calc_risk(portfolio, :Trad; rm = rm) - r1) < 1e-7

    rm.settings.ub = r2
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio, :Trad; rm = rm) <= r2 ||
          abs(calc_risk(portfolio, :Trad; rm = rm) - r2) < 1e-8

    rm.settings.ub = r3
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio, :Trad; rm = rm) <= r3

    rm.settings.ub = r4
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio, :Trad; rm = rm) <= r4

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
    @test dot(portfolio.mu, w15.weights) >= ret3 ||
          abs(dot(portfolio.mu, w15.weights) - ret3) < 1e-10

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
          abs(dot(portfolio.mu, w20.weights) - ret4) < 1e-10

    portfolio.mu_l = Inf
    rm = GMD(; formulation = OWAApprox())

    obj = MinRisk()
    w21 = optimise!(portfolio, Trad(; rm = rm, kelly = NoKelly(), obj = obj))
    r5 = calc_risk(portfolio, :Trad; rm = rm)
    ret5 = dot(portfolio.mu, w21.weights)
    wt = [8.140413690800091e-11, 3.1314237286265923e-10, 2.459150025774704e-10,
          0.012753536931243696, 0.04110182515536095, 0.020275520509108896,
          3.1381157788844864e-11, 0.04814435394212773, 1.5778666784488445e-10,
          1.3636895882485786e-10, 0.07183001206796316, 1.183150790173959e-11,
          2.6559559771998244e-11, 0.28566671687963346, 1.2814464403991672e-11,
          0.014433686947862142, 0.05034883084048245, 0.2318074436879354,
          0.010375874622091419, 0.21326219739898694]
    riskt = 0.007973655098253875
    rett = 0.00029600408640478203
    @test isapprox(w21.weights, wt, rtol = 1.0e-6)
    @test isapprox(r5, riskt)
    @test isapprox(ret5, rett, rtol = 5.0e-7)
    @test isapprox(w21.weights, w1.weights, rtol = 0.05)
    @test isapprox(r5, r1, rtol = 0.0001)
    @test isapprox(ret5, ret1, rtol = 0.01)

    w22 = optimise!(portfolio, Trad(; rm = rm, kelly = AKelly(), obj = obj))
    wt = [7.034395130489708e-11, 2.7181892930716457e-10, 2.122776472288477e-10,
          0.012753547485918709, 0.04110187352399651, 0.020275504313749547,
          2.710851288819668e-11, 0.04814436792707842, 1.3637204039503427e-10,
          1.178684164111055e-10, 0.07182994288989382, 1.022132387349719e-11,
          2.2978381925349025e-11, 0.2856670935500346, 1.1076568106888852e-11,
          0.01443368980013432, 0.050348616318303056, 0.23180736012161507,
          0.01037579411940822, 0.21326220906980192]
    @test isapprox(w22.weights, wt, rtol = 1.0e-6)
    @test isapprox(w22.weights, w2.weights, rtol = 0.05)

    w23 = optimise!(portfolio, Trad(; rm = rm, kelly = EKelly(), obj = obj))
    wt = [3.283193871157042e-9, 1.2623384893600218e-8, 9.6875124882564e-9,
          0.012749885648440006, 0.04110312287321009, 0.020276168437349404,
          1.309441552615906e-9, 0.04814729990808006, 6.253786503484984e-9,
          5.441711051356393e-9, 0.07182907888394237, 5.450069196191549e-10,
          1.128244946994874e-9, 0.28566055946302527, 5.846012985314102e-10,
          0.014433653744394552, 0.05035097906706464, 0.23180795291629838,
          0.010378189265173865, 0.21326306893613764]
    @test isapprox(w23.weights, wt, rtol = 5.0e-5)
    @test isapprox(w23.weights, w3.weights, rtol = 0.05)

    obj = Utility(; l = l)
    w24 = optimise!(portfolio, Trad(; rm = rm, kelly = NoKelly(), obj = obj))
    r6 = calc_risk(portfolio, :Trad; rm = rm)
    ret6 = dot(portfolio.mu, w24.weights)
    wt = [5.8110157601432665e-11, 2.3417285930233157e-10, 1.2115696693370317e-10,
          0.007251917893291007, 0.054753388198703026, 9.627022738622659e-11,
          1.7969586951807185e-11, 0.05983396638901407, 1.69498742783923e-10,
          4.5590939260668677e-10, 0.06474252983328635, 1.1016105690368296e-11,
          2.069658947082804e-11, 0.24014344916696204, 1.2899382126862342e-11,
          0.019952580776195442, 0.0809902232434041, 0.21692365742361994,
          0.05867570592101953, 0.19673257995680443]
    riskt = 0.008024389698787715
    rett = 0.0005208059567492945
    @test isapprox(w24.weights, wt, rtol = 1.0e-6)
    @test isapprox(r6, riskt)
    @test isapprox(ret6, rett, rtol = 5.0e-7)
    @test isapprox(w24.weights, w4.weights, rtol = 0.05)
    @test isapprox(r6, r2, rtol = 0.0001)
    @test isapprox(ret6, ret2, rtol = 0.001)

    w25 = optimise!(portfolio, Trad(; rm = rm, kelly = AKelly(), obj = obj))
    wt = [2.0736631602331747e-10, 8.655223246760442e-10, 4.422112699075041e-10,
          0.007279964293440632, 0.054710041436977594, 3.526068558092772e-10,
          5.867546955347469e-11, 0.05980210019357443, 6.19706182580293e-10,
          1.6819963957422548e-9, 0.06488958369738956, 3.289042608836966e-11,
          6.854894781971836e-11, 0.24024294736395768, 3.981157068271533e-11,
          0.019958752569611955, 0.08084283517203607, 0.21706489736597484,
          0.058375208492143185, 0.19683366504555833]
    @test isapprox(w25.weights, wt, rtol = 1.0e-5)
    @test isapprox(w25.weights, w5.weights, rtol = 0.05)

    w26 = optimise!(portfolio, Trad(; rm = rm, kelly = EKelly(), obj = obj))
    wt = [7.005014909612338e-10, 2.7467026366131667e-9, 1.4017364500267405e-9,
          0.007283940938477996, 0.054704072897809736, 1.1078365447758096e-9,
          2.421590740893963e-10, 0.05979334185468173, 1.951592606355225e-9,
          4.921544922778384e-9, 0.06489521904235875, 1.6565473623071294e-10,
          2.716506287275471e-10, 0.24026430507652788, 1.8698567442688306e-10,
          0.019956465632820667, 0.08082675417550453, 0.21707369519738892,
          0.05835340693325342, 0.19684878455481156]
    @test isapprox(w26.weights, wt, rtol = 1.0e-6)
    @test isapprox(w26.weights, w6.weights, rtol = 0.05)

    obj = Sharpe(; rf = rf)
    w27 = optimise!(portfolio, Trad(; rm = rm, kelly = NoKelly(), obj = obj))
    r7 = calc_risk(portfolio, :Trad; rm = rm)
    ret7 = dot(portfolio.mu, w27.weights)
    wt = [2.3165483637216146e-9, 5.277960972301135e-9, 2.6144623655189783e-9,
          4.937613666793425e-9, 0.20912077984038888, 7.170818923270425e-10,
          8.655513259240756e-10, 0.008622102737487099, 0.02233840818634442,
          1.3008110844344438e-8, 3.928695536801523e-9, 1.2667027584232391e-9,
          8.584328349856967e-10, 3.1283197358020426e-9, 1.1293172346711742e-9,
          0.0954810207995227, 0.4040142134083853, 6.303883709457116e-9, 0.2604234242519083,
          4.423282169992254e-9]
    riskt = 0.010796650473566557
    rett = 0.0016610969770519217
    @test isapprox(w27.weights, wt, rtol = 5.0e-5)
    @test isapprox(r7, riskt, rtol = 1.0e-6)
    @test isapprox(ret7, rett, rtol = 1.0e-6)
    @test isapprox(w27.weights, w7.weights, rtol = 0.05)
    @test isapprox(r7, r3, rtol = 0.005)
    @test isapprox(ret7, ret3, rtol = 0.005)

    w28 = optimise!(portfolio, Trad(; rm = rm, kelly = AKelly(), obj = obj))
    wt = [2.801693829174791e-9, 6.4864422622523155e-9, 3.170070310537541e-9,
          5.947838867798853e-9, 0.20410322971309705, 8.704675205133046e-10,
          1.0479250127313452e-9, 0.013630508416909019, 0.01331618628758413,
          1.628346473911696e-8, 4.87430778001188e-9, 1.5425845581195923e-9,
          1.0374826039966627e-9, 3.859230960634676e-9, 1.3608523183479668e-9,
          0.09642016219761229, 0.40186863168291026, 7.857753161602685e-9, 0.27066121900057,
          5.561203485491739e-9]
    @test isapprox(w28.weights, wt, rtol = 5.0e-4)
    @test isapprox(w28.weights, w8.weights, rtol = 0.05)

    w29 = optimise!(portfolio, Trad(; rm = rm, kelly = EKelly(), obj = obj))
    wt = [6.719217698200277e-9, 1.537906070128829e-8, 7.56738270970034e-9,
          1.4446662815228481e-8, 0.20424520764711784, 2.0779558045850886e-9,
          2.540005853981756e-9, 0.013628782997443988, 0.012993333960935351,
          3.658840993142483e-8, 1.1219204711372385e-8, 3.742555065696061e-9,
          2.4762518863217168e-9, 9.00088123867838e-9, 3.2575941221762144e-9,
          0.0964393881748356, 0.40171928756537423, 1.7818008021791055e-8,
          0.2709738541541385, 1.266696407069806e-8]
    @test isapprox(w29.weights, wt, rtol = 0.0005)
    @test isapprox(w29.weights, w9.weights, rtol = 0.05)

    obj = MaxRet()
    w30 = optimise!(portfolio, Trad(; rm = rm, kelly = NoKelly(), obj = obj))
    r8 = calc_risk(portfolio, :Trad; rm = rm)
    ret8 = dot(portfolio.mu, w30.weights)
    wt = [3.003468043166542e-8, 4.989680902144069e-8, 3.236147859233924e-8,
          6.925347727164586e-8, 0.999998549648788, 8.880071192002583e-9,
          1.4192005624622162e-8, 4.580488074279412e-8, 1.4087036390860031e-7,
          4.706727710273597e-8, 2.4012363049655272e-8, 1.8498467909574943e-8,
          1.0678075715091602e-8, 2.3437721208674016e-8, 1.3737369143792671e-8,
          1.6398960158911142e-7, 6.023666956203431e-7, 3.110222932878985e-8,
          9.835275183965494e-8, 2.581489271209344e-8]
    riskt = 0.017170960084852898
    rett = 0.0019086263079852075
    @test isapprox(w30.weights, wt)
    @test isapprox(r8, riskt)
    @test isapprox(ret8, rett)

    w31 = optimise!(portfolio, Trad(; rm = rm, kelly = AKelly(), obj = obj))
    wt = [2.1877877724598778e-10, 5.423634965391811e-10, 2.5253431909718113e-10,
          7.867829458604338e-10, 0.8577730688248231, 8.617306754355166e-11,
          1.881760474509161e-11, 5.340082262051755e-10, 2.6334834681217134e-9,
          5.48174866618407e-10, 1.399929067726362e-10, 4.649356854273397e-11,
          7.430736924775098e-11, 1.2724883443711166e-10, 2.7593276139962158e-11,
          5.004679883728293e-9, 0.1422269180687958, 2.5760424370343565e-10,
          1.6464003712654701e-9, 1.6094394207994044e-10]
    @test isapprox(w31.weights, wt)
    @test isapprox(w31.weights, w11.weights, rtol = 5.0e-6)

    w32 = optimise!(portfolio, Trad(; rm = rm, kelly = EKelly(), obj = obj))
    wt = [7.495244566211183e-9, 1.3356715214376525e-8, 7.779717687780013e-9,
          1.7599686482177285e-8, 0.8926338954347407, 2.5033970331565238e-9,
          3.377725979808072e-9, 1.2898997612987054e-8, 5.788946814686859e-8,
          1.2506319355284187e-8, 6.269268628602493e-9, 4.546684802206718e-9,
          2.6316119362164115e-9, 5.991019267313653e-9, 3.2331818591355224e-9,
          1.019630349555436e-7, 0.10736579479363274, 8.18324477422888e-9,
          3.505072621551823e-8, 6.4955820348114385e-9]
    @test isapprox(w32.weights, wt, rtol = 5.0e-6)
    @test isapprox(w32.weights, w12.weights, rtol = 1.0e-5)

    # Risk upper bound
    obj = MaxRet()
    rm.settings.ub = r5 * 1.01
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio, :Trad; rm = rm) <= r5 * 1.01

    rm.settings.ub = r6
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio, :Trad; rm = rm) <= r6

    rm.settings.ub = r7
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio, :Trad; rm = rm) <= r7

    rm.settings.ub = r8
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio, :Trad; rm = rm) <= r8

    obj = Sharpe(; rf = rf)
    rm.settings.ub = r5 * 1.01
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio, :Trad; rm = rm) <= r5 * 1.01

    rm.settings.ub = r6
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio, :Trad; rm = rm) <= r6

    rm.settings.ub = r7
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio, :Trad; rm = rm) <= r7

    rm.settings.ub = r8
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio, :Trad; rm = rm) <= r8

    # Ret lower bound
    rm.settings.ub = Inf
    obj = MinRisk()
    portfolio.mu_l = ret5
    w33 = optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test dot(portfolio.mu, w33.weights) >= ret5

    portfolio.mu_l = ret6
    w34 = optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test dot(portfolio.mu, w34.weights) >= ret6

    portfolio.mu_l = ret7
    w35 = optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test dot(portfolio.mu, w35.weights) >= ret7

    portfolio.mu_l = ret8
    w36 = optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test dot(portfolio.mu, w16.weights) >= ret8

    obj = Sharpe(; rf = rf)
    portfolio.mu_l = ret5
    w37 = optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test dot(portfolio.mu, w17.weights) >= ret1

    portfolio.mu_l = ret6
    w38 = optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test dot(portfolio.mu, w18.weights) >= ret6

    portfolio.mu_l = ret7
    w39 = optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test dot(portfolio.mu, w39.weights) >= ret7

    portfolio.mu_l = ret8
    w40 = optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test dot(portfolio.mu, w40.weights) >= ret8
end

@testset "TG" begin
    portfolio = Portfolio(; prices = prices[(end - 200):end],
                          solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                           :check_sol => (allow_local = true,
                                                                          allow_almost = true),
                                                           :params => Dict("verbose" => false,
                                                                           "max_step_fraction" => 0.75,
                                                                           "max_iter" => 100,
                                                                           "equilibrate_max_iter" => 20))))
    asset_statistics!(portfolio)
    rm = TG(; formulation = OWAExact())

    obj = MinRisk()
    w1 = optimise!(portfolio, Trad(; rm = rm, kelly = NoKelly(), obj = obj))
    r1 = calc_risk(portfolio, :Trad; rm = rm)
    ret1 = dot(portfolio.mu, w1.weights)
    wt = [5.969325155308689e-13, 0.19890625725641747, 5.01196330972591e-12,
          0.05490289350205719, 4.555898520654742e-11, 1.0828830360388467e-12,
          3.3254529493732914e-12, 0.10667398877393132, 1.0523685498886423e-12,
          1.8363867378603495e-11, 0.0486780391076569, 0.09089397416963123,
          3.698655012660736e-12, 8.901903375667794e-12, 2.0443054631623374e-12,
          0.13418199806574263, 6.531476985844397e-12, 0.13829581359726595,
          1.724738947709949e-12, 0.22746703542940375]
    riskt = 0.02356383470533441
    rett = 0.0005937393209710076
    @test isapprox(w1.weights, wt)
    @test isapprox(r1, riskt)
    @test isapprox(ret1, rett)

    w2 = optimise!(portfolio, Trad(; rm = rm, kelly = AKelly(), obj = obj))
    wt = [7.856868439678794e-13, 0.1989062583681603, 4.838778513927888e-12,
          0.054902892206828, 4.38111226861629e-11, 1.2803205819789809e-12,
          3.5416120724104257e-12, 0.10667398932256585, 1.3155352269357149e-12,
          1.858482119276245e-11, 0.048678037555666076, 0.09089397446667812,
          3.926648516381695e-12, 9.106515699454836e-12, 1.7422547807424147e-12,
          0.1341819985507733, 6.519246165182908e-12, 0.1382958141359801,
          1.543400238197745e-12, 0.22746703529635232]
    @test isapprox(w2.weights, wt)

    w3 = optimise!(portfolio, Trad(; rm = rm, kelly = EKelly(), obj = obj))
    wt = [3.558276949223149e-11, 0.19890649251730394, 8.339684482191303e-11,
          0.054903006138987936, 4.435282205930281e-10, 3.131042671094137e-11,
          1.1925439551200015e-11, 0.10667398512265627, 3.205009634146099e-11,
          2.005456583753527e-10, 0.04867780055964718, 0.09089395600647165,
          8.839854474688133e-12, 1.1690613364791602e-10, 5.6683355297695216e-11,
          0.13418187280163776, 1.0595822785383569e-10, 0.13829579900743022,
          5.7396745888772046e-11, 0.22746708666174134]
    @test isapprox(w3.weights, wt)

    obj = Utility(; l = l)
    w4 = optimise!(portfolio, Trad(; rm = rm, kelly = NoKelly(), obj = obj))
    r2 = calc_risk(portfolio, :Trad; rm = rm)
    ret2 = dot(portfolio.mu, w4.weights)
    wt = [2.3531676655820754e-12, 0.22146555850379093, 9.410028484870327e-12,
          0.05148290829152357, 8.0605983841845e-10, 4.9113218142397595e-12,
          8.131364518006022e-12, 0.11536086211760316, 2.271272286048353e-12,
          5.43251102138864e-11, 7.650522053564507e-10, 0.09513372436124735,
          8.884714671260909e-12, 1.3757061851213529e-11, 1.6291810113089304e-13,
          0.1432756744154531, 2.0775280711374397e-11, 0.14179358412385565,
          4.251506235430576e-12, 0.2314876864861804]
    riskt = 0.02356604972729362
    rett = 0.000633639299164722
    @test isapprox(w4.weights, wt)
    @test isapprox(r2, riskt)
    @test isapprox(ret2, rett)

    w5 = optimise!(portfolio, Trad(; rm = rm, kelly = AKelly(), obj = obj))
    wt = [7.89044712826026e-13, 0.22386204538926974, 3.552778484583874e-12,
          0.06247360288493247, 2.681553554799882e-10, 1.6414072098675413e-12,
          2.8859413630889454e-12, 0.11120554432711087, 9.715362217793387e-13,
          2.0393816091257544e-11, 3.5256904279870844e-10, 0.09247718794641346,
          3.1599126086787316e-12, 5.745570168144382e-12, 9.982334651996505e-14,
          0.13672420196765245, 6.549771094991111e-12, 0.1389988715113014,
          1.5187241475848804e-12, 0.234258545305287]
    @test isapprox(w5.weights, wt)

    w6 = optimise!(portfolio, Trad(; rm = rm, kelly = EKelly(), obj = obj))
    wt = [9.020812440261157e-12, 0.2238691771668263, 2.2902607458718274e-11,
          0.06250624343467237, 8.920948226738776e-10, 6.036964724458708e-12,
          2.2145160275491298e-12, 0.11119320768009527, 9.038009482320266e-12,
          7.6435766143155e-11, 8.44799817150313e-10, 0.09246929426507262,
          1.3660110791138543e-12, 2.810362809747963e-11, 1.1492651852637795e-11,
          0.13670473474202013, 3.690296159262254e-11, 0.13899056687783914,
          1.7157900326346932e-11, 0.2342667738759077]
    @test isapprox(w6.weights, wt, rtol = 5.0e-6)

    obj = Sharpe(; rf = rf)
    w7 = optimise!(portfolio, Trad(; rm = rm, kelly = NoKelly(), obj = obj))
    r3 = calc_risk(portfolio, :Trad; rm = rm)
    ret3 = dot(portfolio.mu, w7.weights)
    wt = [1.0342189964374973e-11, 1.400649279934334e-11, 1.0025902458601371e-11,
          6.108757972397652e-12, 0.35723529048659247, 1.2069564550657953e-11,
          1.2830347528094887e-11, 1.1781378314488614e-12, 6.5343202013864566e-12,
          2.895398409002917e-12, 9.118697983089066e-12, 1.0966314618191202e-11,
          1.3147762207425575e-11, 9.611939363025545e-12, 1.263250173104243e-11,
          0.21190800224274572, 0.4308567071123932, 7.712262968799656e-12,
          1.2412631429194858e-11, 6.675313779730199e-12]
    riskt = 0.03208927225110264
    rett = 0.0017686555804583674
    @test isapprox(w7.weights, wt)
    @test isapprox(r3, riskt)
    @test isapprox(ret3, rett)

    w8 = optimise!(portfolio, Trad(; rm = rm, kelly = AKelly(), obj = obj))
    wt = [1.2959197244759468e-11, 8.084356150422197e-11, 1.3926601479162983e-11,
          2.8785342853445117e-11, 0.32204068843816375, 3.214346139905109e-12,
          4.692943169531694e-12, 6.33836412905834e-11, 1.484504201576286e-10,
          3.804633552654395e-11, 1.9639729850630742e-11, 1.1407234782655726e-11,
          4.23415076667216e-12, 1.7187957086567557e-11, 5.342675786892118e-12,
          0.20669809375923995, 0.47126121713540664, 2.38746093868233e-11,
          1.625803285907461e-10, 2.8620587870227428e-11]
    @test isapprox(w8.weights, wt)

    w9 = optimise!(portfolio, Trad(; rm = rm, kelly = EKelly(), obj = obj))
    wt = [1.3801649259171437e-7, 8.592740180284502e-7, 1.4750175001348523e-7,
          2.93858579076417e-7, 0.33034161966700004, 4.25979385472981e-8,
          5.669708973524571e-8, 6.511546089373594e-7, 1.0217092350189244e-6,
          3.8588865642171967e-7, 2.0759750030993866e-7, 1.2423343866980414e-7,
          5.2774005027449543e-8, 1.8191259827821858e-7, 6.320894661292222e-8,
          0.20770466885250968, 0.46194751935242984, 2.4840026016179655e-7,
          1.412838309788135e-6, 3.0446463326205604e-7]
    @test isapprox(w9.weights, wt, rtol = 0.005)

    obj = MaxRet()
    w10 = optimise!(portfolio, Trad(; rm = rm, kelly = NoKelly(), obj = obj))
    r4 = calc_risk(portfolio, :Trad; rm = rm)
    ret4 = dot(portfolio.mu, w10.weights)
    wt = [8.730661626481806e-10, 1.4687309386680775e-9, 9.358654898947756e-10,
          2.186677678966605e-9, 0.9999999341280966, 3.156239683565371e-10,
          4.629682444037334e-10, 1.3343331058979577e-9, 5.69212053674475e-9,
          1.3749327860458163e-9, 7.161024888414711e-10, 5.748069307464309e-10,
          3.666660703394924e-10, 7.0140810552241e-10, 4.5086981398858524e-10,
          7.002103638459066e-9, 3.6268802221190136e-8, 9.019721807104911e-10,
          3.482364214437503e-9, 7.624890097546668e-10]
    riskt = 0.041844368667445314
    rett = 0.0019086272314632886
    @test isapprox(w10.weights, wt)
    @test isapprox(r4, riskt)
    @test isapprox(ret4, rett)

    w11 = optimise!(portfolio, Trad(; rm = rm, kelly = AKelly(), obj = obj))
    wt = [6.036096739587215e-12, 1.790094245743028e-11, 7.472431823401233e-12,
          2.6015061923204717e-11, 0.8577737313404759, 5.106563274225794e-12,
          1.900666972805193e-12, 1.798805120124093e-11, 9.180990810209664e-11,
          1.8285780713416517e-11, 3.878646392138336e-12, 2.6114409753401695e-13,
          4.039407801796174e-12, 3.3645983430819403e-12, 2.0794146894700064e-12,
          1.8675565751693112e-10, 0.14222626819696105, 8.128255004448968e-12,
          5.689786193015622e-11, 4.642665800172863e-12]
    @test isapprox(w11.weights, wt)

    w12 = optimise!(portfolio, Trad(; rm = rm, kelly = EKelly(), obj = obj))
    wt = [4.65046216906908e-9, 8.296253929547001e-9, 4.865391327970577e-9,
          1.0916852112918646e-8, 0.8926331516236782, 1.5238336619771253e-9,
          2.076243111622752e-9, 8.049225239703472e-9, 3.617749936955201e-8,
          7.86223613409023e-9, 3.89394397749308e-9, 2.8230386682124775e-9,
          1.6131006163903666e-9, 3.7199163646151806e-9, 1.9944778806862605e-9,
          6.473044506145028e-8, 0.10736665414274231, 5.092714368879538e-9,
          2.1903498344282783e-8, 4.044447060145975e-9]
    @test isapprox(w12.weights, wt, rtol = 5.0e-6)

    # Risk upper bound
    obj = MaxRet()
    rm.settings.ub = r1
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio, :Trad; rm = rm) <= r1 ||
          abs(calc_risk(portfolio, :Trad; rm = rm) - r1) < 1e-8

    rm.settings.ub = r2
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio, :Trad; rm = rm) <= r2 ||
          abs(calc_risk(portfolio, :Trad; rm = rm) - r2) < 5e-10

    rm.settings.ub = r3
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio, :Trad; rm = rm) <= r3

    rm.settings.ub = r4
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio, :Trad; rm = rm) <= r4

    obj = Sharpe(; rf = rf)
    rm.settings.ub = r1
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio, :Trad; rm = rm) <= r1 ||
          abs(calc_risk(portfolio, :Trad; rm = rm) - r1) < 5e-9

    rm.settings.ub = r2
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio, :Trad; rm = rm) <= r2 ||
          abs(calc_risk(portfolio, :Trad; rm = rm) - r2) < 5e-10

    rm.settings.ub = r3
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio, :Trad; rm = rm) <= r3

    rm.settings.ub = r4
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio, :Trad; rm = rm) <= r4

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
    @test dot(portfolio.mu, w15.weights) >= ret3 ||
          abs(dot(portfolio.mu, w15.weights) - ret3) < 1e-10

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
          abs(dot(portfolio.mu, w20.weights) - ret4) < 1e-10

    portfolio.mu_l = Inf
    rm = TG(; formulation = OWAApprox())

    obj = MinRisk()
    w21 = optimise!(portfolio, Trad(; rm = rm, kelly = NoKelly(), obj = obj))
    r5 = calc_risk(portfolio, :Trad; rm = rm)
    ret5 = dot(portfolio.mu, w21.weights)
    wt = [2.1793176066144965e-10, 0.24657578304895264, 5.476679022837874e-10,
          0.041100700961631355, 1.745099342448546e-9, 2.0168069945246096e-10,
          7.629903785304511e-11, 0.11651659826585649, 1.9351039549294452e-10,
          1.0580904330304497e-9, 5.933842241374943e-8, 0.09732932587810658,
          5.619716498021303e-11, 6.782716439749812e-10, 3.4982414551865584e-10,
          0.14662256588896547, 5.824103810697346e-10, 0.12694239844522773,
          3.400868203663799e-10, 0.22491256212576752]
    riskt = 0.023582366401225324
    rett = 0.0006432235211782866
    @test isapprox(w21.weights, wt, rtol = 5.0e-5)
    @test isapprox(r5, riskt, rtol = 5.0e-7)
    @test isapprox(ret5, rett, rtol = 5.0e-6)
    @test isapprox(w21.weights, w1.weights, rtol = 0.25)
    @test isapprox(r5, r1, rtol = 0.001)
    @test isapprox(ret5, ret1, rtol = 0.1)

    w22 = optimise!(portfolio, Trad(; rm = rm, kelly = AKelly(), obj = obj))
    wt = [2.132183776397635e-10, 0.24657765819583657, 5.355024854123618e-10,
          0.04109911857321909, 1.7029194284107386e-9, 1.974679372397374e-10,
          7.446166082524666e-11, 0.11651653133071553, 1.8952844225356302e-10,
          1.0314865628216637e-9, 6.295221219719734e-8, 0.09732926904417218,
          5.470853792167843e-11, 6.745287801746886e-10, 3.428894296510097e-10,
          0.1466228087872955, 5.711384009324977e-10, 0.12694288445100937,
          3.3336697140072015e-10, 0.2249116607443225]
    @test isapprox(w22.weights, wt, rtol = 5.0e-5)
    @test isapprox(w22.weights, w2.weights, rtol = 0.25)

    w23 = optimise!(portfolio, Trad(; rm = rm, kelly = EKelly(), obj = obj))
    wt = [1.927837301458917e-10, 0.2465727935166203, 4.5776197117687873e-10,
          0.0411034353688984, 1.4160012847338463e-9, 1.7931052490904626e-10,
          7.492504213032373e-11, 0.11651631583960065, 1.7182805746854656e-10,
          8.718361635547206e-10, 6.868975884764904e-8, 0.09732867628494225,
          5.796922323295118e-11, 6.094684782004051e-10, 3.067454763778319e-10,
          0.14662102857422757, 5.026263032275627e-10, 0.12694436503613155,
          2.957487773764835e-10, 0.22491331155281538]
    @test isapprox(w23.weights, wt, rtol = 5.0e-5)
    @test isapprox(w23.weights, w3.weights, rtol = 0.25)

    obj = Utility(; l = l)
    w24 = optimise!(portfolio, Trad(; rm = rm, kelly = NoKelly(), obj = obj))
    r6 = calc_risk(portfolio, :Trad; rm = rm)
    ret6 = dot(portfolio.mu, w24.weights)
    wt = [4.031429185256284e-10, 0.24853621593089034, 1.0564435079928515e-9,
          0.041227339264945304, 3.187793853320347e-8, 2.993481845211221e-10,
          1.316251677752441e-10, 0.11775482753120951, 4.0247404407692655e-10,
          2.4553272342813325e-9, 8.117272668206023e-9, 0.09806143571966024,
          9.667832808949718e-11, 1.0479300493959856e-9, 5.16355132377166e-10,
          0.14808310631257574, 1.3560319804684306e-9, 0.12226287288483399,
          7.148464022467191e-10, 0.22407415388047078]
    riskt = 0.023585657006845066
    rett = 0.0006461184324927211
    @test isapprox(w24.weights, wt, rtol = 5.0e-5)
    @test isapprox(r6, riskt, rtol = 1.0e-7)
    @test isapprox(ret6, rett, rtol = 5.0e-6)
    @test isapprox(w24.weights, w4.weights, rtol = 0.1)
    @test isapprox(r6, r2, rtol = 0.001)
    @test isapprox(ret6, ret2, rtol = 0.05)

    w25 = optimise!(portfolio, Trad(; rm = rm, kelly = AKelly(), obj = obj))
    wt = [1.507133941066285e-10, 0.24827976788672826, 3.73990694940837e-10,
          0.041308890243378256, 7.675438081297324e-9, 1.1154812723593603e-10,
          4.951489768730557e-11, 0.11762133998644565, 1.4637086998562598e-10,
          8.756011450879805e-10, 6.943850562800141e-9, 0.09794572252106948,
          3.5923970291357996e-11, 4.3057334069665553e-10, 2.002929167365126e-10,
          0.14789685246583284, 5.19138038484229e-10, 0.12278739931801369,
          2.6551070295904287e-10, 0.22416000980006515]
    @test isapprox(w25.weights, wt, rtol = 1.0e-5)
    @test isapprox(w25.weights, w5.weights, rtol = 0.1)

    w26 = optimise!(portfolio, Trad(; rm = rm, kelly = EKelly(), obj = obj))
    wt = [1.8080300946247023e-10, 0.24828292527938153, 4.68121589287319e-10,
          0.04130630822882187, 1.2359535796409274e-8, 1.3734305268211817e-10,
          6.559527877655717e-11, 0.11762157323389853, 1.814435581688314e-10,
          1.0697620823533454e-9, 2.1021006172059525e-9, 0.09794654012223465,
          5.079896932410669e-11, 4.387939949600902e-10, 2.2506554283303623e-10,
          0.14789817630497223, 5.74829787739748e-10, 0.12278545945634213,
          3.126744048715974e-10, 0.2241589992074813]
    @test isapprox(w26.weights, wt, rtol = 5.0e-6)
    @test isapprox(w26.weights, w6.weights, rtol = 0.1)

    obj = Sharpe(; rf = rf)
    w27 = optimise!(portfolio, Trad(; rm = rm, kelly = NoKelly(), obj = obj))
    r7 = calc_risk(portfolio, :Trad; rm = rm)
    ret7 = dot(portfolio.mu, w27.weights)
    wt = [2.611247264961084e-10, 1.8600657774984216e-9, 2.7626934335103416e-10,
          5.497788498901174e-10, 0.3290378114659103, 8.224811202970948e-11,
          1.068509794521808e-10, 1.0698183121278107e-9, 1.6147966531606507e-9,
          7.115598011937777e-10, 3.984269424697578e-10, 2.4827950070497723e-10,
          1.0188959979493673e-10, 3.5208451215754276e-10, 1.1948530978332868e-10,
          0.20499862760973586, 0.4659635498088353, 4.740736136016396e-10,
          2.3087600736393448e-9, 5.80006498885295e-10]
    riskt = 0.03204363189094902
    rett = 0.00176571179911687
    @test isapprox(w27.weights, wt, rtol = 0.0001)
    @test isapprox(r7, riskt, rtol = 5.0e-6)
    @test isapprox(ret7, rett, rtol = 5.0e-6)
    @test isapprox(w27.weights, w7.weights, rtol = 0.1)
    @test isapprox(r7, r3, rtol = 0.005)
    @test isapprox(ret7, ret3, rtol = 0.005)

    w28 = optimise!(portfolio, Trad(; rm = rm, kelly = AKelly(), obj = obj))
    wt = [1.869325822862471e-10, 1.4328061947056766e-9, 1.983624339909501e-10,
          3.9049540188495594e-10, 0.2993249703403086, 5.91655997016558e-11,
          7.591779016922828e-11, 9.543812870273496e-10, 1.629098331489297e-9,
          5.646270702560161e-10, 3.0669712261634527e-10, 1.7813852329496214e-10,
          7.36556294671998e-11, 2.624444093298053e-10, 8.577535879309574e-11,
          0.20566930367997732, 0.4950057161442303, 3.6134043743821436e-10,
          2.615852679683726e-9, 4.5979294701178386e-10]
    @test isapprox(w28.weights, wt, rtol = 1.0e-5)
    @test isapprox(w28.weights, w8.weights, rtol = 0.1)

    w29 = optimise!(portfolio, Trad(; rm = rm, kelly = EKelly(), obj = obj))
    wt = [9.221062320825582e-8, 7.047478800634529e-7, 9.781708009684867e-8,
          1.916840895248151e-7, 0.2993421988101789, 2.9430761608320223e-8,
          3.7627543652153636e-8, 4.689749347827317e-7, 8.149066126733291e-7,
          2.7810815645071277e-7, 1.5112599835099536e-7, 8.779800183945856e-8,
          3.6587065526347325e-8, 1.2935364553010815e-7, 4.243295514320968e-8,
          0.20608699061385133, 0.4945659352314525, 1.7836651758245156e-7,
          1.3084916989859224e-6, 2.2568095228872053e-7]
    @test isapprox(w29.weights, wt, rtol = 0.001)
    @test isapprox(w29.weights, w9.weights, rtol = 0.1)

    obj = MaxRet()
    w30 = optimise!(portfolio, Trad(; rm = rm, kelly = NoKelly(), obj = obj))
    r8 = calc_risk(portfolio, :Trad; rm = rm)
    ret8 = dot(portfolio.mu, w30.weights)
    wt = [3.003468043166542e-8, 4.989680902144069e-8, 3.236147859233924e-8,
          6.925347727164586e-8, 0.999998549648788, 8.880071192002583e-9,
          1.4192005624622162e-8, 4.580488074279412e-8, 1.4087036390860031e-7,
          4.706727710273597e-8, 2.4012363049655272e-8, 1.8498467909574943e-8,
          1.0678075715091602e-8, 2.3437721208674016e-8, 1.3737369143792671e-8,
          1.6398960158911142e-7, 6.023666956203431e-7, 3.110222932878985e-8,
          9.835275183965494e-8, 2.581489271209344e-8]
    riskt = 0.041844336232849194
    rett = 0.0019086263079852075
    @test isapprox(w30.weights, wt)
    @test isapprox(r8, riskt)
    @test isapprox(ret8, rett)

    w31 = optimise!(portfolio, Trad(; rm = rm, kelly = AKelly(), obj = obj))
    wt = [2.1877877724598778e-10, 5.423634965391811e-10, 2.5253431909718113e-10,
          7.867829458604338e-10, 0.8577730688248231, 8.617306754355166e-11,
          1.881760474509161e-11, 5.340082262051755e-10, 2.6334834681217134e-9,
          5.48174866618407e-10, 1.399929067726362e-10, 4.649356854273397e-11,
          7.430736924775098e-11, 1.2724883443711166e-10, 2.7593276139962158e-11,
          5.004679883728293e-9, 0.1422269180687958, 2.5760424370343565e-10,
          1.6464003712654701e-9, 1.6094394207994044e-10]
    @test isapprox(w31.weights, wt)
    @test isapprox(w31.weights, w11.weights, rtol = 5.0e-6)

    w32 = optimise!(portfolio, Trad(; rm = rm, kelly = EKelly(), obj = obj))
    wt = [7.495244566211183e-9, 1.3356715214376525e-8, 7.779717687780013e-9,
          1.7599686482177285e-8, 0.8926338954347407, 2.5033970331565238e-9,
          3.377725979808072e-9, 1.2898997612987054e-8, 5.788946814686859e-8,
          1.2506319355284187e-8, 6.269268628602493e-9, 4.546684802206718e-9,
          2.6316119362164115e-9, 5.991019267313653e-9, 3.2331818591355224e-9,
          1.019630349555436e-7, 0.10736579479363274, 8.18324477422888e-9,
          3.505072621551823e-8, 6.4955820348114385e-9]
    @test isapprox(w32.weights, wt, rtol = 5.0e-6)
    @test isapprox(w32.weights, w12.weights, rtol = 5.0e-6)

    # Risk upper bound
    obj = MaxRet()
    rm.settings.ub = r5 * 1.001
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio, :Trad; rm = rm) <= r5 * 1.001

    rm.settings.ub = r6 * 1.001
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio, :Trad; rm = rm) <= r6 * 1.001

    rm.settings.ub = r7
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio, :Trad; rm = rm) <= r7

    rm.settings.ub = r8
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio, :Trad; rm = rm) <= r8

    obj = Sharpe(; rf = rf)
    rm.settings.ub = r5 * 1.001
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio, :Trad; rm = rm) <= r5 * 1.001

    rm.settings.ub = r6 * 1.001
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio, :Trad; rm = rm) <= r6 * 1.001

    rm.settings.ub = r7
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio, :Trad; rm = rm) <= r7

    rm.settings.ub = r8
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio, :Trad; rm = rm) <= r8

    # Ret lower bound
    rm.settings.ub = Inf
    obj = MinRisk()
    portfolio.mu_l = ret5
    w33 = optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test dot(portfolio.mu, w33.weights) >= ret5

    portfolio.mu_l = ret6
    w34 = optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test dot(portfolio.mu, w34.weights) >= ret6

    portfolio.mu_l = ret7
    w35 = optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test dot(portfolio.mu, w35.weights) >= ret7

    portfolio.mu_l = ret8
    w36 = optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test dot(portfolio.mu, w16.weights) >= ret8

    obj = Sharpe(; rf = rf)
    portfolio.mu_l = ret5
    w37 = optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test dot(portfolio.mu, w17.weights) >= ret1

    portfolio.mu_l = ret6
    w38 = optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test dot(portfolio.mu, w18.weights) >= ret6

    portfolio.mu_l = ret7
    w39 = optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test dot(portfolio.mu, w39.weights) >= ret7

    portfolio.mu_l = ret8
    w40 = optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test dot(portfolio.mu, w40.weights) >= ret8
end

@testset "TGRG" begin
    portfolio = Portfolio(; prices = prices[(end - 200):end],
                          solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                           :check_sol => (allow_local = true,
                                                                          allow_almost = true),
                                                           :params => Dict("verbose" => false,
                                                                           "max_step_fraction" => 0.75,
                                                                           "max_iter" => 150,
                                                                           "equilibrate_max_iter" => 20))))
    asset_statistics!(portfolio)
    rm = TGRG(; formulation = OWAExact())

    obj = MinRisk()
    w1 = optimise!(portfolio, Trad(; rm = rm, kelly = NoKelly(), obj = obj))
    r1 = calc_risk(portfolio, :Trad; rm = rm)
    ret1 = dot(portfolio.mu, w1.weights)
    wt = [2.5214700615756507e-11, 0.0977217336226819, 7.926139285725633e-12,
          7.457000940478333e-11, 0.017418083280552568, 0.041201250865648346,
          3.651208574394543e-11, 0.0605532233732396, 4.0061964805479585e-11,
          3.485215673060734e-11, 0.22830023908770422, 5.948224415946499e-11,
          8.091952729934292e-11, 0.13930598099477526, 4.6100247383814126e-11,
          0.0362263423975006, 2.1886131636373804e-10, 0.24663412209234842,
          3.368068554202892e-12, 0.13263902365768074]
    riskt = 0.040929059924343876
    rett = 0.0002030737318767868
    @test isapprox(w1.weights, wt, rtol = 5.0e-6)
    @test isapprox(r1, riskt)
    @test isapprox(ret1, rett, rtol = 5.0e-7)

    w2 = optimise!(portfolio, Trad(; rm = rm, kelly = AKelly(), obj = obj))
    wt = [1.2108720101946655e-11, 0.09772217254221559, 4.81863826084196e-12,
          3.02585838958066e-11, 0.01741776379164449, 0.04120109573339351,
          1.6887091610159788e-11, 0.060553113693647444, 1.814783613740796e-11,
          1.550224070179354e-11, 0.2283000074934528, 2.2736746484279212e-11,
          3.5127248380557346e-11, 0.13930594113966846, 2.080394368295553e-11,
          0.036225995972047495, 8.850605909822465e-11, 0.24663483687605506,
          2.880208709756556e-14, 0.1326390724929492]
    @test isapprox(w2.weights, wt, rtol = 1.0e-5)

    w3 = optimise!(portfolio, Trad(; rm = rm, kelly = EKelly(), obj = obj))
    wt = [8.811924635330943e-11, 0.09772140868720845, 1.122654799152096e-10,
          2.37572471816488e-10, 0.017418373373283072, 0.04120131178042838,
          7.127051950074236e-11, 0.060553314717056085, 6.589675795931631e-11,
          7.494015206341305e-11, 0.2283004114667184, 2.1131509261245617e-10,
          6.938595551965339e-12, 0.13930602059627178, 5.7562377876372754e-11,
          0.03622666368397424, 4.344115727549587e-10, 0.24663350637558398,
          1.2919699945413943e-10, 0.13263898782998645]
    @test isapprox(w3.weights, wt, rtol = 1.0e-5)

    obj = Utility(; l = l)
    w4 = optimise!(portfolio, Trad(; rm = rm, kelly = NoKelly(), obj = obj))
    r2 = calc_risk(portfolio, :Trad; rm = rm)
    ret2 = dot(portfolio.mu, w4.weights)
    wt = [7.026805478010537e-12, 0.07691042016208616, 1.4421489729308237e-12,
          3.605941949944227e-12, 0.018736328513127485, 0.02727677352489846,
          7.406419741602451e-12, 0.06300039975110802, 9.651040072665584e-12,
          7.87028804146206e-12, 0.23138910163058715, 1.8363064032313167e-12,
          1.772932503046444e-11, 0.12343688817492235, 1.1346887350045238e-11,
          0.04767277729217605, 9.551667214175191e-11, 0.2743410772339696,
          6.373792606352843e-13, 0.1372362335530555]
    riskt = 0.04094394403447746
    rett = 0.0002672659151039052
    @test isapprox(w4.weights, wt, rtol = 1.0e-5)
    @test isapprox(r2, riskt)
    @test isapprox(ret2, rett, rtol = 5.0e-6)

    w5 = optimise!(portfolio, Trad(; rm = rm, kelly = AKelly(), obj = obj))
    wt = [1.600055346649941e-11, 0.07691021608020968, 1.8469418148082205e-12,
          9.803684650795123e-12, 0.01873640699156462, 0.027276835283812476,
          1.6894403771567032e-11, 0.06300044191624866, 2.2762867658148485e-11,
          1.8229280293511624e-11, 0.23138920733402632, 6.781298043889894e-12,
          4.316209189530447e-11, 0.1234368682215077, 2.6839601081935933e-11,
          0.04767288463330064, 2.459971361881845e-10, 0.2743409166508848,
          2.24390877556761e-13, 0.1372362224799031]
    @test isapprox(w5.weights, wt, rtol = 5.0e-6)

    w6 = optimise!(portfolio, Trad(; rm = rm, kelly = EKelly(), obj = obj))
    wt = [1.0787202920705292e-10, 0.07690948400647307, 1.6879056417409415e-10,
          2.5245792909003697e-10, 0.018736500329638677, 0.027276963891749564,
          1.000057881290897e-10, 0.06300063043099476, 7.954318508906627e-11,
          1.013061699807833e-10, 0.23138969387684377, 1.9924484204092705e-10,
          6.358240878581979e-12, 0.12343664483254023, 6.105478255907702e-11,
          0.047673346540890435, 1.4270229200577038e-9, 0.2743405274999266,
          1.7633723380245867e-10, 0.13723620591094907]
    @test isapprox(w6.weights, wt, rtol = 5.0e-6)

    obj = Sharpe(; rf = rf)
    w7 = optimise!(portfolio, Trad(; rm = rm, kelly = NoKelly(), obj = obj))
    r3 = calc_risk(portfolio, :Trad; rm = rm)
    ret3 = dot(portfolio.mu, w7.weights)
    wt = [4.8650129756778804e-11, 2.1051127604426756e-11, 4.7364768826598876e-11,
          4.001588817352124e-11, 0.1416419921187176, 5.571902073317586e-11,
          5.631693464060233e-11, 0.0249378701969849, 1.7997595945540913e-10,
          2.6619900499982886e-11, 4.154321525692782e-11, 5.2543818628034845e-11,
          5.781072172554138e-11, 4.516910259420552e-11, 5.695227967987287e-11,
          0.21453454572847452, 0.618885583612367, 3.483200049567782e-11,
          7.540575027899836e-9, 3.831617883258372e-11]
    riskt = 0.058407554117453894
    rett = 0.0017136727125023712
    @test isapprox(w7.weights, wt)
    @test isapprox(r3, riskt)
    @test isapprox(ret3, rett)

    w8 = optimise!(portfolio, Trad(; rm = rm, kelly = AKelly(), obj = obj))
    wt = [1.654206037447342e-10, 5.358158451039841e-10, 1.748314075924528e-10,
          2.9147892540376374e-10, 0.12240639032349754, 3.9584770668606736e-11,
          6.4153791283446e-11, 0.08640359838798325, 4.2188574847853614e-9,
          4.592854281143285e-10, 3.1732874227066445e-10, 1.2465979434902665e-10,
          4.931653503148915e-11, 2.504134044719627e-10, 5.6885470515011877e-11,
          0.18988672243361443, 0.5839704059313315, 4.1162063264733596e-10,
          0.017332875396264127, 3.6765633747301796e-10]
    @test isapprox(w8.weights, wt, rtol = 5.0e-5)

    w9 = optimise!(portfolio, Trad(; rm = rm, kelly = EKelly(), obj = obj))
    wt = [6.819222451299357e-7, 1.9455839168785423e-6, 7.223321480278138e-7,
          1.2141889319095752e-6, 0.13446074251326615, 2.0241616270705986e-7,
          2.983949165841309e-7, 0.027218556131528564, 1.8934383723013564e-5,
          1.7669965118151768e-6, 1.1574660838236174e-6, 5.140068292903971e-7,
          2.383435351241923e-7, 9.405418558136564e-7, 2.741611974536715e-7,
          0.20138995968439916, 0.6200582566977366, 1.4773399348650805e-6,
          0.01684079661400841, 1.3202810686765458e-6]
    @test isapprox(w9.weights, wt, rtol = 0.05)

    obj = MaxRet()
    w10 = optimise!(portfolio, Trad(; rm = rm, kelly = NoKelly(), obj = obj))
    r4 = calc_risk(portfolio, :Trad; rm = rm)
    ret4 = dot(portfolio.mu, w10.weights)
    wt = [8.742294125473961e-10, 1.4707486744372402e-9, 9.371030740356124e-10,
          2.189422648260022e-9, 0.9999999340817113, 3.1597937318578487e-10,
          4.6333260236071825e-10, 1.336207603310839e-9, 5.696795440054946e-9,
          1.3768680167520074e-9, 7.171481417281956e-10, 5.752604171160508e-10,
          3.667148376429282e-10, 7.02423046380329e-10, 4.511960450499541e-10,
          7.007335179502708e-9, 3.628430937215484e-8, 9.033006700630679e-10,
          3.486313790911996e-9, 7.636002597011294e-10]
    riskt = 0.09407701111033144
    rett = 0.0019086272314304555
    @test isapprox(w10.weights, wt)
    @test isapprox(r4, riskt)
    @test isapprox(ret4, rett)

    w11 = optimise!(portfolio, Trad(; rm = rm, kelly = AKelly(), obj = obj))
    wt = [6.081744743317995e-12, 1.8027870228623893e-11, 7.528771234816698e-12,
          2.619433427851579e-11, 0.8577737315543389, 5.131178140286494e-12,
          1.910345129583652e-12, 1.8116791122017028e-11, 9.24125719338025e-11,
          1.841401037679729e-11, 3.910406972249195e-12, 2.6816959315973815e-13,
          4.0589539810848404e-12, 3.393483997501398e-12, 2.094088472931994e-12,
          1.879427893567236e-10, 0.1422262679800275, 8.189891650530504e-12,
          5.727689069582678e-11, 4.6811169005055095e-12]
    @test isapprox(w11.weights, wt)

    w12 = optimise!(portfolio, Trad(; rm = rm, kelly = EKelly(), obj = obj))
    wt = [6.557930724596975e-9, 1.169719539259437e-8, 6.857789315099148e-9,
          1.53923396788463e-8, 0.8926328782837357, 2.15295445251154e-9,
          2.928350914069381e-9, 1.1344549994935971e-8, 5.1072724712584e-8,
          1.1075784478191067e-8, 5.48983431519763e-9, 3.980052489203433e-9,
          2.275195783161705e-9, 5.244427324021107e-9, 2.812394513654006e-9,
          9.134466109979566e-8, 0.10736684771470409, 7.178445745346023e-9,
          3.0896176410926564e-8, 5.700752843300637e-9]
    @test isapprox(w12.weights, wt, rtol = 1.0e-6)

    # Risk upper bound
    obj = MaxRet()
    rm.settings.ub = r1
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio, :Trad; rm = rm) <= r1 ||
          abs(calc_risk(portfolio, :Trad; rm = rm) - r1) < 5e-7

    rm.settings.ub = r2
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio, :Trad; rm = rm) <= r2 ||
          abs(calc_risk(portfolio, :Trad; rm = rm) - r2) < 5e-9

    rm.settings.ub = r3
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio, :Trad; rm = rm) <= r3

    rm.settings.ub = r4
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio, :Trad; rm = rm) <= r4

    obj = Sharpe(; rf = rf)
    rm.settings.ub = r1
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test abs(calc_risk(portfolio, :Trad; rm = rm) - r1) < 5e-8

    rm.settings.ub = r2
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio, :Trad; rm = rm) <= r2 ||
          abs(calc_risk(portfolio, :Trad; rm = rm) - r2) < 1e-8

    rm.settings.ub = r3
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio, :Trad; rm = rm) <= r3

    rm.settings.ub = r4
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio, :Trad; rm = rm) <= r4

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
    @test dot(portfolio.mu, w15.weights) >= ret3 ||
          abs(dot(portfolio.mu, w15.weights) - ret3) < 1e-10

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
          abs(dot(portfolio.mu, w20.weights) - ret4) < 1e-10

    portfolio.mu_l = Inf
    rm = TGRG(; formulation = OWAApprox())

    obj = MinRisk()
    w21 = optimise!(portfolio, Trad(; rm = rm, kelly = NoKelly(), obj = obj))
    r5 = calc_risk(portfolio, :Trad; rm = rm)
    ret5 = dot(portfolio.mu, w21.weights)
    wt = [1.7462960892040426e-10, 0.07595900140326077, 2.2462862078716012e-10,
          4.1694976470230874e-10, 0.016158519860181603, 0.06416220550267758,
          1.4375745138077692e-10, 0.06291504138456387, 1.3574503250418147e-10,
          1.3385423993741634e-10, 0.2346916256683561, 3.767441280618164e-10,
          9.696691150876087e-12, 0.12787666960665323, 1.0856406928222544e-10,
          0.03254510693876332, 8.971270388896471e-10, 0.25270338097861256,
          2.5981779164819596e-10, 0.1329884457754166]
    riskt = 0.040954129005633125
    rett = 9.918670892592328e-5
    @test isapprox(w21.weights, wt, rtol = 5.0e-5)
    @test isapprox(r5, riskt, rtol = 5.0e-7)
    @test isapprox(ret5, rett, rtol = 0.0005)
    @test isapprox(w21.weights, w1.weights, rtol = 0.1)
    @test isapprox(r5, r1, rtol = 0.001)
    @test isapprox(ret5, ret1, rtol = 1.0)

    w22 = optimise!(portfolio, Trad(; rm = rm, kelly = AKelly(), obj = obj))
    wt = [8.63858885172251e-11, 0.07595815692971593, 1.1146006722904144e-10,
          2.083710210535686e-10, 0.016161780014724153, 0.06416452075134822,
          7.084957696805093e-11, 0.06291525556520872, 6.717412739040257e-11,
          6.663036112475917e-11, 0.23469041209683966, 1.8433568020564578e-10,
          4.903202008412172e-12, 0.1278789341587736, 5.3617252529952105e-11,
          0.03254586329159728, 4.4193332845829306e-10, 0.25269674790959884,
          1.283304145893871e-10, 0.1329883278582027]
    @test isapprox(w22.weights, wt, rtol = 5.0e-5)
    @test isapprox(w22.weights, w2.weights, rtol = 0.1)

    w23 = optimise!(portfolio, Trad(; rm = rm, kelly = EKelly(), obj = obj))
    wt = [1.6793518717256488e-10, 0.07596106036994572, 2.1253995610133782e-10,
          3.8596872185479985e-10, 0.016157340901643096, 0.06416129228334888,
          1.4005228196843185e-10, 0.0629155704419926, 1.330078309091806e-10,
          1.3091495562998193e-10, 0.23469088663504892, 3.4942070098402143e-10,
          1.9416952951307485e-11, 0.12787699755749554, 1.0822833740344731e-10,
          0.03254538498044053, 8.215938138656856e-10, 0.25269966151676576,
          2.448764833032413e-10, 0.1329918025993639]
    @test isapprox(w23.weights, wt, rtol = 5.0e-5)
    @test isapprox(w23.weights, w3.weights, rtol = 0.1)

    obj = Utility(; l = l)
    w24 = optimise!(portfolio, Trad(; rm = rm, kelly = NoKelly(), obj = obj))
    r6 = calc_risk(portfolio, :Trad; rm = rm)
    ret6 = dot(portfolio.mu, w24.weights)
    wt = [3.0304053951267564e-10, 0.07453304868365346, 4.1828999142831036e-10,
          8.214117407719228e-10, 0.013255403794685544, 0.030458667234735934,
          2.4353769672299006e-10, 0.06755531886068894, 2.271514127114169e-10,
          2.386649231216626e-10, 0.24079384312943225, 5.051439605773995e-10,
          1.4108661976498046e-11, 0.115304129360715, 1.4376597446132064e-10,
          0.05209000398439176, 8.023524255067688e-9, 0.26862239171868946,
          4.905563191324257e-10, 0.13738718180381207]
    riskt = 0.04095506618348751
    rett = 0.0002523116021427951
    @test isapprox(w24.weights, wt, rtol = 5.0e-5)
    @test isapprox(r6, riskt, rtol = 5.0e-7)
    @test isapprox(ret6, rett, rtol = 0.0001)
    @test isapprox(w24.weights, w4.weights, rtol = 0.05)
    @test isapprox(r6, r2, rtol = 0.0005)
    @test isapprox(ret6, ret2, rtol = 0.1)

    w25 = optimise!(portfolio, Trad(; rm = rm, kelly = AKelly(), obj = obj))
    wt = [1.8353611267988513e-10, 0.0745184295592935, 2.5408791927800754e-10,
          4.918663126231065e-10, 0.013399887290100344, 0.030570224814760637,
          1.4708830314950738e-10, 0.06751035369476728, 1.3798915593816255e-10,
          1.4529379799881571e-10, 0.24065941788228978, 3.041009095939534e-10,
          9.092271824077073e-12, 0.11545001889277848, 8.723588651223558e-11,
          0.05192208558815138, 4.809366300730343e-9, 0.2685936663484324,
          2.9665893516290464e-10, 0.13737590906311037]
    @test isapprox(w25.weights, wt, rtol = 5.0e-5)
    @test isapprox(w25.weights, w5.weights, rtol = 0.05)

    portfolio.solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                               :check_sol => (allow_local = true,
                                                              allow_almost = true),
                                               :params => Dict("verbose" => false,
                                                               "max_step_fraction" => 0.65,
                                                               "max_iter" => 200,
                                                               "equilibrate_max_iter" => 20)))
    w26 = optimise!(portfolio, Trad(; rm = rm, kelly = EKelly(), obj = obj))
    wt = [1.1021014280037167e-10, 0.07451709909662046, 1.4914922139122829e-10,
          2.638451535490444e-10, 0.013398398692832728, 0.030570877454752877,
          9.169620634465952e-11, 0.06750925204409298, 8.339622172675383e-11,
          8.490355487034885e-11, 0.2406585676041461, 1.8729991674281185e-10,
          9.390461254664664e-12, 0.11544816636449991, 5.5056401314852075e-11,
          0.05192177370561512, 2.9148413261716578e-9, 0.26859767349044694,
          1.7601536995316106e-10, 0.13737818742118874]
    @test isapprox(w26.weights, wt, rtol = 5.0e-5)
    @test isapprox(w26.weights, w6.weights, rtol = 0.05)

    portfolio.solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                               :check_sol => (allow_local = true,
                                                              allow_almost = true),
                                               :params => Dict("verbose" => false,
                                                               "max_step_fraction" => 0.75,
                                                               "max_iter" => 150,
                                                               "equilibrate_max_iter" => 20)))
    obj = Sharpe(; rf = rf)
    w27 = optimise!(portfolio, Trad(; rm = rm, kelly = NoKelly(), obj = obj))
    r7 = calc_risk(portfolio, :Trad; rm = rm)
    ret7 = dot(portfolio.mu, w27.weights)
    wt = [8.219545880021218e-11, 2.565438728252597e-10, 8.650606098099197e-11,
          1.4114233483488025e-10, 0.13198940023993627, 2.4610410018955212e-11,
          3.6184278866044613e-11, 0.050797988372461154, 2.1819447478665915e-9,
          2.0811378789218257e-10, 1.510716122221882e-10, 6.433466526981136e-11,
          2.911151147549101e-11, 1.1914850620333808e-10, 3.229941562811719e-11,
          0.2040331478373737, 0.6131790914570013, 1.9178100101472346e-10,
          3.6831330182416174e-7, 1.7493820048093999e-10]
    riskt = 0.05762039329563678
    rett = 0.0016919367978591121
    @test isapprox(w27.weights, wt, rtol = 1.0e-5)
    @test isapprox(r7, riskt, rtol = 5.0e-6)
    @test isapprox(ret7, rett, rtol = 5.0e-6)
    @test isapprox(w27.weights, w7.weights, rtol = 0.05)
    @test isapprox(r7, r3, rtol = 0.05)
    @test isapprox(ret7, ret3, rtol = 0.05)

    w28 = optimise!(portfolio, Trad(; rm = rm, kelly = AKelly(), obj = obj))
    wt = [9.574307101948423e-11, 3.0355265397207857e-10, 1.0055856980898536e-10,
          1.6435915900306982e-10, 0.12952559215215598, 2.883929506160503e-11,
          4.180752892868291e-11, 0.05962696437901661, 2.3733804378045624e-9,
          2.518939224188598e-10, 1.8073275928122368e-10, 7.438203510683199e-11,
          3.406773904276162e-11, 1.4214121534051272e-10, 3.7916194561772323e-11,
          0.19604082980586632, 0.600752260675684, 2.302394346160292e-10,
          0.014054348718009799, 2.096532708364886e-10]
    @test isapprox(w28.weights, wt, rtol = 1.0e-5)
    @test isapprox(w28.weights, w8.weights, rtol = 0.1)

    w29 = optimise!(portfolio, Trad(; rm = rm, kelly = EKelly(), obj = obj))
    wt = [8.93750764235055e-9, 2.825606017133729e-8, 9.382468767567432e-9,
          1.5246770144175973e-8, 0.12935162072446502, 2.7586031089035915e-9,
          3.954169057701503e-9, 0.0602979412472781, 2.0247784667332573e-7,
          2.3323605705757623e-8, 1.686122207745861e-8, 6.9761738732298636e-9,
          3.243383113111965e-9, 1.3275656401974732e-8, 3.591645866502726e-9,
          0.1959803430288302, 0.6005064746738639, 2.1464400533055073e-8,
          0.013863240996211855, 1.957983787246729e-8]
    @test isapprox(w29.weights, wt, rtol = 0.0005)
    @test isapprox(w29.weights, w9.weights, rtol = 0.1)

    obj = MaxRet()
    w30 = optimise!(portfolio, Trad(; rm = rm, kelly = NoKelly(), obj = obj))
    r8 = calc_risk(portfolio, :Trad; rm = rm)
    ret8 = dot(portfolio.mu, w30.weights)
    wt = [3.00845786348597e-8, 4.998342351476689e-8, 3.241872407636413e-8,
          6.939670670550576e-8, 0.9999985466871499, 8.894593817511603e-9,
          1.4224027755782415e-8, 4.588284791539874e-8, 1.4117228008197312e-7,
          4.715433551011111e-8, 2.4048842977890437e-8, 1.8537296912282917e-8,
          1.0709853495700978e-8, 2.347278852314067e-8, 1.3767219866865825e-8,
          1.643632180618663e-7, 6.036553493957491e-7, 3.1149941840168985e-8,
          9.854270619900907e-8, 2.5854114672658464e-8]
    riskt = 0.094076924625055
    rett = 0.0019086263061293103
    @test isapprox(w30.weights, wt)
    @test isapprox(r8, riskt)
    @test isapprox(ret8, rett)

    w31 = optimise!(portfolio, Trad(; rm = rm, kelly = AKelly(), obj = obj))
    wt = [2.190334412895664e-10, 5.426275417067833e-10, 2.527422968656065e-10,
          7.872036601656902e-10, 0.8577730698594613, 8.642574311796287e-11,
          1.8708367749634052e-11, 5.343213504436128e-10, 2.634345870526317e-9,
          5.486611558308938e-10, 1.4018106002552162e-10, 4.6749606324967964e-11,
          7.408098151878835e-11, 1.2735938731434327e-10, 2.7441807943994096e-11,
          5.005105120519997e-9, 0.14222691702943244, 2.5781871823641354e-10,
          1.6471762350765557e-9, 1.6112397258611544e-10]
    @test isapprox(w31.weights, wt)
    @test isapprox(w31.weights, w11.weights, rtol = 5.0e-6)

    w32 = optimise!(portfolio, Trad(; rm = rm, kelly = EKelly(), obj = obj))
    wt = [6.988817379941981e-9, 1.246112573221949e-8, 7.27450216352786e-9,
          1.6424993437502256e-8, 0.8926373681917745, 2.3127374052792367e-9,
          3.1453814825319106e-9, 1.2059839067651109e-8, 5.372980370297169e-8,
          1.1724298488561742e-8, 5.8500245552980346e-9, 4.241841445606058e-9,
          2.4497152835755314e-9, 5.592179770640874e-9, 3.015253977137906e-9,
          9.491532014568073e-8, 0.10736234326349421, 7.644458300255699e-9,
          3.264578032704871e-8, 6.068658483137429e-9]
    @test isapprox(w32.weights, wt, rtol = 1.0e-5)
    @test isapprox(w32.weights, w12.weights, rtol = 1.0e-5)

    # Risk upper bound
    obj = MaxRet()
    rm.settings.ub = r5 * 1.001
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio, :Trad; rm = rm) <= r5 * 1.001

    rm.settings.ub = r6 * 1.001
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio, :Trad; rm = rm) <= r6 * 1.001

    rm.settings.ub = r7
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio, :Trad; rm = rm) <= r7

    rm.settings.ub = r8
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio, :Trad; rm = rm) <= r8

    obj = Sharpe(; rf = rf)
    rm.settings.ub = r5 * 1.001
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio, :Trad; rm = rm) <= r5 * 1.001

    rm.settings.ub = r6 * 1.001
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio, :Trad; rm = rm) <= r6 * 1.001

    rm.settings.ub = r7
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio, :Trad; rm = rm) <= r7

    rm.settings.ub = r8
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio, :Trad; rm = rm) <= r8

    # Ret lower bound
    rm.settings.ub = Inf
    obj = MinRisk()
    portfolio.mu_l = ret5
    w33 = optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test dot(portfolio.mu, w33.weights) >= ret5

    portfolio.mu_l = ret6
    w34 = optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test dot(portfolio.mu, w34.weights) >= ret6

    portfolio.mu_l = ret7
    w35 = optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test dot(portfolio.mu, w35.weights) >= ret7 ||
          abs(dot(portfolio.mu, w35.weights) - ret7) < 1e-10

    portfolio.mu_l = ret8
    w36 = optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test dot(portfolio.mu, w16.weights) >= ret8

    obj = Sharpe(; rf = rf)
    portfolio.mu_l = ret5
    w37 = optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test dot(portfolio.mu, w17.weights) >= ret1

    portfolio.mu_l = ret6
    w38 = optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test dot(portfolio.mu, w18.weights) >= ret6

    portfolio.mu_l = ret7
    w39 = optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test dot(portfolio.mu, w39.weights) >= ret7

    portfolio.mu_l = ret8
    w40 = optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test dot(portfolio.mu, w40.weights) >= ret8
end

@testset "OWA" begin
    portfolio = Portfolio(; prices = prices[(end - 200):end],
                          solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                           :check_sol => (allow_local = true,
                                                                          allow_almost = true),
                                                           :params => Dict("verbose" => false,
                                                                           "max_step_fraction" => 0.75,
                                                                           "max_iter" => 100,
                                                                           "equilibrate_max_iter" => 20))))
    asset_statistics!(portfolio)
    rm = OWA(; formulation = OWAExact())

    obj = MinRisk()
    w1 = optimise!(portfolio, Trad(; rm = rm, kelly = NoKelly(), obj = obj))
    r1 = calc_risk(portfolio, :Trad; rm = rm)
    ret1 = dot(portfolio.mu, w1.weights)
    wt = [2.860021085782162e-10, 1.2840309896218056e-9, 6.706876349584523e-10,
          0.011313354703334149, 0.0411883749620692, 0.021164889458692743,
          9.703898006880258e-11, 0.047274318622891866, 5.765740916268004e-10,
          4.570883347699393e-10, 0.07079108352937005, 2.3692512642792012e-11,
          6.86255931091155e-11, 0.29261593911138273, 2.6280037229769384e-11,
          0.014804770495317407, 0.055360406768988304, 0.22572329075836045,
          0.008097367714011027, 0.21166620038556183]
    riskt = 0.007973013868952676
    rett = 0.00029353687778951485
    @test isapprox(w1.weights, wt, rtol = 5.0e-6)
    @test isapprox(r1, riskt)
    @test isapprox(ret1, rett, rtol = 1.0e-6)

    w2 = optimise!(portfolio, Trad(; rm = rm, kelly = AKelly(), obj = obj))
    wt = [5.250702616813578e-10, 2.3649724723993027e-9, 1.2279491916198084e-9,
          0.01136623473742439, 0.0411504017712101, 0.02111853666359142,
          1.7823656285034675e-10, 0.04726442039789345, 1.0571802875720308e-9,
          8.38512874634768e-10, 0.07085086111257587, 4.4206112295333334e-11,
          1.2541463432371463e-10, 0.29253129575787246, 4.881312822213951e-11,
          0.014796400914986178, 0.05529639972264038, 0.22572814198254204,
          0.008121076155090038, 0.2117762243738181]
    @test isapprox(w2.weights, wt, rtol = 1.0e-6)

    w3 = optimise!(portfolio, Trad(; rm = rm, kelly = EKelly(), obj = obj))
    wt = [5.323839510142863e-10, 2.229192435645185e-9, 1.1712440903830328e-9,
          0.011344721661516587, 0.04116337911244441, 0.02111829239021385,
          2.1641304918219377e-10, 0.047255888185545984, 1.005291628173614e-9,
          8.091650681427752e-10, 0.07082525289366046, 9.26602870015525e-11,
          1.6964206457835295e-10, 0.29255890224533354, 9.639341758984514e-11,
          0.014800853997633014, 0.055321508358536234, 0.2257384135285236,
          0.008118509287753244, 0.211754272016453]
    @test isapprox(w3.weights, wt, rtol = 5.0e-6)

    obj = Utility(; l = l)
    w4 = optimise!(portfolio, Trad(; rm = rm, kelly = NoKelly(), obj = obj))
    r2 = calc_risk(portfolio, :Trad; rm = rm)
    ret2 = dot(portfolio.mu, w4.weights)
    wt = [4.395442944440727e-10, 2.1495615257047445e-9, 8.700158416075159e-10,
          0.006589303508967539, 0.05330845589356504, 7.632737386668005e-10,
          9.831583865942048e-11, 0.06008335161013898, 1.3797121555924764e-9,
          3.0160588760343415e-9, 0.06394372429204083, 4.486199590323317e-11,
          1.0975948346699443e-10, 0.24314317639043878, 5.90094645737884e-11,
          0.018483948217979402, 0.08594753958504305, 0.216033164253688, 0.05666930366548,
          0.19579802365254514]
    riskt = 0.008023853567234033
    rett = 0.0005204190203575007
    @test isapprox(w4.weights, wt, rtol = 1.0e-6)
    @test isapprox(r2, riskt)
    @test isapprox(ret2, rett, rtol = 5.0e-8)

    w5 = optimise!(portfolio, Trad(; rm = rm, kelly = AKelly(), obj = obj))
    wt = [2.1362512461147959e-10, 1.0473426830174378e-9, 4.232904776563241e-10,
          0.006623775536322704, 0.053261985949698866, 3.7469349330098336e-10,
          4.858498678278164e-11, 0.05983738243189082, 6.626203326440813e-10,
          1.432599593268026e-9, 0.06407652576443064, 2.2558984339781933e-11,
          5.3690642468248475e-11, 0.24323109220603928, 2.96486691492619e-11,
          0.018454184890627775, 0.08594373007217042, 0.21613806337839309,
          0.05652374385950317, 0.1959095116022682]
    @test isapprox(w5.weights, wt, rtol = 5.0e-6)

    w6 = optimise!(portfolio, Trad(; rm = rm, kelly = EKelly(), obj = obj))
    wt = [5.937788173616308e-10, 2.7035923440469955e-9, 1.1184215485237802e-9,
          0.006640033798649065, 0.05321801224326745, 9.963309538802194e-10,
          1.8376772950076455e-10, 0.05964942314439764, 1.7115613152193195e-9,
          3.697169350159616e-9, 0.06413044858686795, 1.187496479279526e-10,
          1.9736634980204724e-10, 0.24337338790443755, 1.362995356534818e-10,
          0.018577394541992927, 0.0858912259147116, 0.216090928601388, 0.056732714784793,
          0.19569641902245713]
    @test isapprox(w6.weights, wt, rtol = 0.005)

    obj = Sharpe(; rf = rf)
    w7 = optimise!(portfolio, Trad(; rm = rm, kelly = NoKelly(), obj = obj))
    r3 = calc_risk(portfolio, :Trad; rm = rm)
    ret3 = dot(portfolio.mu, w7.weights)
    wt = [1.0467205635658914e-12, 4.278912675065536e-11, 3.374093991319961e-12,
          3.8277639532838844e-11, 0.21079296400188735, 2.4364045721705662e-11,
          2.2390341906149484e-11, 0.01106006895313273, 0.0215579625569257,
          1.5262610412144913e-10, 2.1500846791960303e-11, 1.6225951152430356e-11,
          2.272395485557818e-11, 1.011545112756681e-11, 1.8686814475829852e-11,
          0.09771827802310681, 0.3985981897681632, 5.508078066235759e-11,
          0.26027253624006286, 2.7519584077987032e-11]
    riskt = 0.010781870841767164
    rett = 0.001659046007257454
    @test isapprox(w7.weights, wt)
    @test isapprox(r3, riskt)
    @test isapprox(ret3, rett)

    w8 = optimise!(portfolio, Trad(; rm = rm, kelly = AKelly(), obj = obj))
    wt = [1.0920644565612393e-10, 2.5303831383802556e-10, 1.2376306976438313e-10,
          2.369564354076823e-10, 0.20616254959454816, 3.202182206033137e-11,
          3.971613464382764e-11, 0.015569075031954824, 0.01601461718150225,
          6.237669799218814e-10, 1.8776259246956673e-10, 6.016603087271084e-11,
          3.871362662544616e-11, 1.4873125031397894e-10, 5.163281350655045e-11,
          0.09845408940920172, 0.39772987513121644, 3.0052638214680857e-10,
          0.266069791233974, 2.1160070552220802e-10]
    @test isapprox(w8.weights, wt)

    w9 = optimise!(portfolio, Trad(; rm = rm, kelly = EKelly(), obj = obj))
    wt = [3.14905351889511e-7, 7.08763348846685e-7, 3.538581511420651e-7,
          6.876632506556505e-7, 0.20716142258876596, 9.623479730547593e-8,
          1.2169267976274955e-7, 0.015607749851993332, 0.012493885188245756,
          1.6035402196119575e-6, 4.965350735695629e-7, 1.8143160353564277e-7,
          1.1505668719872142e-7, 4.0369243757166665e-7, 1.5047808406371426e-7,
          0.098197585753003, 0.39812133395490806, 7.72480185603894e-7, 0.26841146641647695,
          5.499147363025267e-7]
    @test isapprox(w9.weights, wt, rtol = 0.005)

    obj = MaxRet()
    w10 = optimise!(portfolio, Trad(; rm = rm, kelly = NoKelly(), obj = obj))
    r4 = calc_risk(portfolio, :Trad; rm = rm)
    ret4 = dot(portfolio.mu, w10.weights)
    wt = [8.713294565329801e-10, 1.4655494663368051e-9, 9.339861464983668e-10,
          2.1822916966782698e-9, 0.9999999341529429, 3.1514150951638473e-10,
          4.623635446692933e-10, 1.3314213108887623e-9, 5.686161252843585e-9,
          1.371922705853848e-9, 7.146550790353018e-10, 5.740256406691589e-10,
          3.664169883135698e-10, 7.000039902341564e-10, 4.5030867400627253e-10,
          6.996286227848405e-9, 3.6287634044926756e-8, 9.000571133076656e-10,
          3.4765700831013475e-9, 7.60932312745152e-10]
    riskt = 0.017170975178026397
    rett = 0.0019086272315059051
    @test isapprox(w10.weights, wt)
    @test isapprox(r4, riskt)
    @test isapprox(ret4, rett)

    w11 = optimise!(portfolio, Trad(; rm = rm, kelly = AKelly(), obj = obj))
    wt = [5.995668682294329e-12, 1.778824643654798e-11, 7.422520403984214e-12,
          2.585579633200434e-11, 0.8577737311670789, 5.0842729898340874e-12,
          1.891888971224657e-12, 1.7873758325289183e-11, 9.127375624650092e-11,
          1.8171912621743217e-11, 3.850582248871377e-12, 2.5508491084881004e-13,
          4.021631383731875e-12, 3.3391228496613597e-12, 2.0662267001955667e-12,
          1.856978015488069e-10, 0.14222626837308944, 8.073644265708836e-12,
          5.656088636719738e-11, 4.608688065458597e-12]
    @test isapprox(w11.weights, wt)

    w12 = optimise!(portfolio, Trad(; rm = rm, kelly = EKelly(), obj = obj))
    wt = [4.40859451333247e-9, 7.864888444121469e-9, 4.612603900429012e-9,
          1.0349688665135194e-8, 0.892636677290951, 1.4440413646088198e-9,
          1.968066871845967e-9, 7.631330480424597e-9, 3.4295274358342415e-8,
          7.454362493254684e-9, 3.6914967896364635e-9, 2.6763326762619786e-9,
          1.5289402772999891e-9, 3.5264095931927543e-9, 1.8905286042675526e-9,
          6.13629141224752e-8, 0.10736313857685581, 4.827944431112196e-9,
          2.0764556150948145e-8, 3.834219335175887e-9]
    @test isapprox(w12.weights, wt, rtol = 1.0e-5)

    # Risk upper bound
    obj = MaxRet()
    rm.settings.ub = r1
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio, :Trad; rm = rm) <= r1 ||
          abs(calc_risk(portfolio, :Trad; rm = rm) - r1) < 5e-7

    rm.settings.ub = r2
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio, :Trad; rm = rm) <= r2 ||
          abs(calc_risk(portfolio, :Trad; rm = rm) - r2) < 5e-8

    rm.settings.ub = r3
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio, :Trad; rm = rm) <= r3 ||
          abs(calc_risk(portfolio, :Trad; rm = rm) - r3) < 5e-9

    rm.settings.ub = r4
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio, :Trad; rm = rm) <= r4 ||
          abs(calc_risk(portfolio, :Trad; rm = rm) - r4) < 1e-9

    obj = Sharpe(; rf = rf)
    rm.settings.ub = r1
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio, :Trad; rm = rm) <= r1 ||
          abs(calc_risk(portfolio, :Trad; rm = rm) - r1) < 1e-7

    rm.settings.ub = r2
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio, :Trad; rm = rm) <= r2 ||
          abs(calc_risk(portfolio, :Trad; rm = rm) - r2) < 1e-8

    rm.settings.ub = r3
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio, :Trad; rm = rm) <= r3

    rm.settings.ub = r4
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio, :Trad; rm = rm) <= r4

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
    @test dot(portfolio.mu, w15.weights) >= ret3 ||
          abs(dot(portfolio.mu, w15.weights) - ret3) < 1e-10

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
          abs(dot(portfolio.mu, w20.weights) - ret4) < 1e-10

    portfolio.mu_l = Inf
    rm = OWA(; formulation = OWAApprox())

    obj = MinRisk()
    w21 = optimise!(portfolio, Trad(; rm = rm, kelly = NoKelly(), obj = obj))
    r5 = calc_risk(portfolio, :Trad; rm = rm)
    ret5 = dot(portfolio.mu, w21.weights)
    wt = [8.140413690800091e-11, 3.1314237286265923e-10, 2.459150025774704e-10,
          0.012753536931243696, 0.04110182515536095, 0.020275520509108896,
          3.1381157788844864e-11, 0.04814435394212773, 1.5778666784488445e-10,
          1.3636895882485786e-10, 0.07183001206796316, 1.183150790173959e-11,
          2.6559559771998244e-11, 0.28566671687963346, 1.2814464403991672e-11,
          0.014433686947862142, 0.05034883084048245, 0.2318074436879354,
          0.010375874622091419, 0.21326219739898694]
    riskt = 0.007973655098253875
    rett = 0.00029600408640478203
    @test isapprox(w21.weights, wt, rtol = 1.0e-6)
    @test isapprox(r5, riskt)
    @test isapprox(ret5, rett, rtol = 5.0e-7)
    @test isapprox(w21.weights, w1.weights, rtol = 0.05)
    @test isapprox(r5, r1, rtol = 0.0001)
    @test isapprox(ret5, ret1, rtol = 0.01)

    w22 = optimise!(portfolio, Trad(; rm = rm, kelly = AKelly(), obj = obj))
    wt = [7.034395130489708e-11, 2.7181892930716457e-10, 2.122776472288477e-10,
          0.012753547485918709, 0.04110187352399651, 0.020275504313749547,
          2.710851288819668e-11, 0.04814436792707842, 1.3637204039503427e-10,
          1.178684164111055e-10, 0.07182994288989382, 1.022132387349719e-11,
          2.2978381925349025e-11, 0.2856670935500346, 1.1076568106888852e-11,
          0.01443368980013432, 0.050348616318303056, 0.23180736012161507,
          0.01037579411940822, 0.21326220906980192]
    @test isapprox(w22.weights, wt, rtol = 1.0e-6)
    @test isapprox(w22.weights, w2.weights, rtol = 0.05)

    w23 = optimise!(portfolio, Trad(; rm = rm, kelly = EKelly(), obj = obj))
    wt = [3.283193871157042e-9, 1.2623384893600218e-8, 9.6875124882564e-9,
          0.012749885648440006, 0.04110312287321009, 0.020276168437349404,
          1.309441552615906e-9, 0.04814729990808006, 6.253786503484984e-9,
          5.441711051356393e-9, 0.07182907888394237, 5.450069196191549e-10,
          1.128244946994874e-9, 0.28566055946302527, 5.846012985314102e-10,
          0.014433653744394552, 0.05035097906706464, 0.23180795291629838,
          0.010378189265173865, 0.21326306893613764]
    @test isapprox(w23.weights, wt, rtol = 5.0e-5)
    @test isapprox(w23.weights, w3.weights, rtol = 0.05)

    obj = Utility(; l = l)
    w24 = optimise!(portfolio, Trad(; rm = rm, kelly = NoKelly(), obj = obj))
    r6 = calc_risk(portfolio, :Trad; rm = rm)
    ret6 = dot(portfolio.mu, w24.weights)
    wt = [5.8110157601432665e-11, 2.3417285930233157e-10, 1.2115696693370317e-10,
          0.007251917893291007, 0.054753388198703026, 9.627022738622659e-11,
          1.7969586951807185e-11, 0.05983396638901407, 1.69498742783923e-10,
          4.5590939260668677e-10, 0.06474252983328635, 1.1016105690368296e-11,
          2.069658947082804e-11, 0.24014344916696204, 1.2899382126862342e-11,
          0.019952580776195442, 0.0809902232434041, 0.21692365742361994,
          0.05867570592101953, 0.19673257995680443]
    riskt = 0.008024389698787715
    rett = 0.0005208059567492945
    @test isapprox(w24.weights, wt, rtol = 1.0e-6)
    @test isapprox(r6, riskt)
    @test isapprox(ret6, rett, rtol = 5.0e-7)
    @test isapprox(w24.weights, w4.weights, rtol = 0.05)
    @test isapprox(r6, r2, rtol = 0.0001)
    @test isapprox(ret6, ret2, rtol = 0.001)

    w25 = optimise!(portfolio, Trad(; rm = rm, kelly = AKelly(), obj = obj))
    wt = [2.0736631602331747e-10, 8.655223246760442e-10, 4.422112699075041e-10,
          0.007279964293440632, 0.054710041436977594, 3.526068558092772e-10,
          5.867546955347469e-11, 0.05980210019357443, 6.19706182580293e-10,
          1.6819963957422548e-9, 0.06488958369738956, 3.289042608836966e-11,
          6.854894781971836e-11, 0.24024294736395768, 3.981157068271533e-11,
          0.019958752569611955, 0.08084283517203607, 0.21706489736597484,
          0.058375208492143185, 0.19683366504555833]
    @test isapprox(w25.weights, wt, rtol = 1.0e-5)
    @test isapprox(w25.weights, w5.weights, rtol = 0.05)

    w26 = optimise!(portfolio, Trad(; rm = rm, kelly = EKelly(), obj = obj))
    wt = [7.005014909612338e-10, 2.7467026366131667e-9, 1.4017364500267405e-9,
          0.007283940938477996, 0.054704072897809736, 1.1078365447758096e-9,
          2.421590740893963e-10, 0.05979334185468173, 1.951592606355225e-9,
          4.921544922778384e-9, 0.06489521904235875, 1.6565473623071294e-10,
          2.716506287275471e-10, 0.24026430507652788, 1.8698567442688306e-10,
          0.019956465632820667, 0.08082675417550453, 0.21707369519738892,
          0.05835340693325342, 0.19684878455481156]
    @test isapprox(w26.weights, wt, rtol = 1.0e-6)
    @test isapprox(w26.weights, w6.weights, rtol = 0.05)

    obj = Sharpe(; rf = rf)
    w27 = optimise!(portfolio, Trad(; rm = rm, kelly = NoKelly(), obj = obj))
    r7 = calc_risk(portfolio, :Trad; rm = rm)
    ret7 = dot(portfolio.mu, w27.weights)
    wt = [2.3165483637216146e-9, 5.277960972301135e-9, 2.6144623655189783e-9,
          4.937613666793425e-9, 0.20912077984038888, 7.170818923270425e-10,
          8.655513259240756e-10, 0.008622102737487099, 0.02233840818634442,
          1.3008110844344438e-8, 3.928695536801523e-9, 1.2667027584232391e-9,
          8.584328349856967e-10, 3.1283197358020426e-9, 1.1293172346711742e-9,
          0.0954810207995227, 0.4040142134083853, 6.303883709457116e-9, 0.2604234242519083,
          4.423282169992254e-9]
    riskt = 0.010796650473566557
    rett = 0.0016610969770519217
    @test isapprox(w27.weights, wt, rtol = 5.0e-5)
    @test isapprox(r7, riskt, rtol = 1.0e-6)
    @test isapprox(ret7, rett, rtol = 1.0e-6)
    @test isapprox(w27.weights, w7.weights, rtol = 0.05)
    @test isapprox(r7, r3, rtol = 0.005)
    @test isapprox(ret7, ret3, rtol = 0.005)

    w28 = optimise!(portfolio, Trad(; rm = rm, kelly = AKelly(), obj = obj))
    wt = [2.801693829174791e-9, 6.4864422622523155e-9, 3.170070310537541e-9,
          5.947838867798853e-9, 0.20410322971309705, 8.704675205133046e-10,
          1.0479250127313452e-9, 0.013630508416909019, 0.01331618628758413,
          1.628346473911696e-8, 4.87430778001188e-9, 1.5425845581195923e-9,
          1.0374826039966627e-9, 3.859230960634676e-9, 1.3608523183479668e-9,
          0.09642016219761229, 0.40186863168291026, 7.857753161602685e-9, 0.27066121900057,
          5.561203485491739e-9]
    @test isapprox(w28.weights, wt, rtol = 5.0e-4)
    @test isapprox(w28.weights, w8.weights, rtol = 0.05)

    w29 = optimise!(portfolio, Trad(; rm = rm, kelly = EKelly(), obj = obj))
    wt = [6.719217698200277e-9, 1.537906070128829e-8, 7.56738270970034e-9,
          1.4446662815228481e-8, 0.20424520764711784, 2.0779558045850886e-9,
          2.540005853981756e-9, 0.013628782997443988, 0.012993333960935351,
          3.658840993142483e-8, 1.1219204711372385e-8, 3.742555065696061e-9,
          2.4762518863217168e-9, 9.00088123867838e-9, 3.2575941221762144e-9,
          0.0964393881748356, 0.40171928756537423, 1.7818008021791055e-8,
          0.2709738541541385, 1.266696407069806e-8]
    @test isapprox(w29.weights, wt, rtol = 0.0005)
    @test isapprox(w29.weights, w9.weights, rtol = 0.05)

    obj = MaxRet()
    w30 = optimise!(portfolio, Trad(; rm = rm, kelly = NoKelly(), obj = obj))
    r8 = calc_risk(portfolio, :Trad; rm = rm)
    ret8 = dot(portfolio.mu, w30.weights)
    wt = [3.003468043166542e-8, 4.989680902144069e-8, 3.236147859233924e-8,
          6.925347727164586e-8, 0.999998549648788, 8.880071192002583e-9,
          1.4192005624622162e-8, 4.580488074279412e-8, 1.4087036390860031e-7,
          4.706727710273597e-8, 2.4012363049655272e-8, 1.8498467909574943e-8,
          1.0678075715091602e-8, 2.3437721208674016e-8, 1.3737369143792671e-8,
          1.6398960158911142e-7, 6.023666956203431e-7, 3.110222932878985e-8,
          9.835275183965494e-8, 2.581489271209344e-8]
    riskt = 0.017170960084852898
    rett = 0.0019086263079852075
    @test isapprox(w30.weights, wt)
    @test isapprox(r8, riskt)
    @test isapprox(ret8, rett)

    w31 = optimise!(portfolio, Trad(; rm = rm, kelly = AKelly(), obj = obj))
    wt = [2.1877877724598778e-10, 5.423634965391811e-10, 2.5253431909718113e-10,
          7.867829458604338e-10, 0.8577730688248231, 8.617306754355166e-11,
          1.881760474509161e-11, 5.340082262051755e-10, 2.6334834681217134e-9,
          5.48174866618407e-10, 1.399929067726362e-10, 4.649356854273397e-11,
          7.430736924775098e-11, 1.2724883443711166e-10, 2.7593276139962158e-11,
          5.004679883728293e-9, 0.1422269180687958, 2.5760424370343565e-10,
          1.6464003712654701e-9, 1.6094394207994044e-10]
    @test isapprox(w31.weights, wt)
    @test isapprox(w31.weights, w11.weights, rtol = 5.0e-6)

    w32 = optimise!(portfolio, Trad(; rm = rm, kelly = EKelly(), obj = obj))
    wt = [7.495244566211183e-9, 1.3356715214376525e-8, 7.779717687780013e-9,
          1.7599686482177285e-8, 0.8926338954347407, 2.5033970331565238e-9,
          3.377725979808072e-9, 1.2898997612987054e-8, 5.788946814686859e-8,
          1.2506319355284187e-8, 6.269268628602493e-9, 4.546684802206718e-9,
          2.6316119362164115e-9, 5.991019267313653e-9, 3.2331818591355224e-9,
          1.019630349555436e-7, 0.10736579479363274, 8.18324477422888e-9,
          3.505072621551823e-8, 6.4955820348114385e-9]
    @test isapprox(w32.weights, wt, rtol = 5.0e-6)
    @test isapprox(w32.weights, w12.weights, rtol = 1.0e-5)

    # Risk upper bound
    obj = MaxRet()
    rm.settings.ub = r5 * 1.01
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio, :Trad; rm = rm) <= r5 * 1.01

    rm.settings.ub = r6
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio, :Trad; rm = rm) <= r6

    rm.settings.ub = r7
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio, :Trad; rm = rm) <= r7

    rm.settings.ub = r8
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio, :Trad; rm = rm) <= r8

    obj = Sharpe(; rf = rf)
    rm.settings.ub = r5 * 1.01
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio, :Trad; rm = rm) <= r5 * 1.01

    rm.settings.ub = r6
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio, :Trad; rm = rm) <= r6

    rm.settings.ub = r7
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio, :Trad; rm = rm) <= r7

    rm.settings.ub = r8
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio, :Trad; rm = rm) <= r8

    # Ret lower bound
    rm.settings.ub = Inf
    obj = MinRisk()
    portfolio.mu_l = ret5
    w33 = optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test dot(portfolio.mu, w33.weights) >= ret5

    portfolio.mu_l = ret6
    w34 = optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test dot(portfolio.mu, w34.weights) >= ret6

    portfolio.mu_l = ret7
    w35 = optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test dot(portfolio.mu, w35.weights) >= ret7

    portfolio.mu_l = ret8
    w36 = optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test dot(portfolio.mu, w16.weights) >= ret8

    obj = Sharpe(; rf = rf)
    portfolio.mu_l = ret5
    w37 = optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test dot(portfolio.mu, w17.weights) >= ret1

    portfolio.mu_l = ret6
    w38 = optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test dot(portfolio.mu, w18.weights) >= ret6

    portfolio.mu_l = ret7
    w39 = optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test dot(portfolio.mu, w39.weights) >= ret7

    portfolio.mu_l = ret8
    w40 = optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test dot(portfolio.mu, w40.weights) >= ret8

    portfolio = Portfolio(; prices = prices[(end - 200):end],
                          solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                           :check_sol => (allow_local = true,
                                                                          allow_almost = true),
                                                           :params => Dict("verbose" => false,
                                                                           "max_step_fraction" => 0.75,
                                                                           "max_iter" => 100,
                                                                           "equilibrate_max_iter" => 20))))
    asset_statistics!(portfolio)
    rm = OWA(; w = owa_tg(200), formulation = OWAExact())

    obj = MinRisk()
    w1 = optimise!(portfolio, Trad(; rm = rm, kelly = NoKelly(), obj = obj))
    r1 = calc_risk(portfolio, :Trad; rm = rm)
    ret1 = dot(portfolio.mu, w1.weights)
    wt = [5.969325155308689e-13, 0.19890625725641747, 5.01196330972591e-12,
          0.05490289350205719, 4.555898520654742e-11, 1.0828830360388467e-12,
          3.3254529493732914e-12, 0.10667398877393132, 1.0523685498886423e-12,
          1.8363867378603495e-11, 0.0486780391076569, 0.09089397416963123,
          3.698655012660736e-12, 8.901903375667794e-12, 2.0443054631623374e-12,
          0.13418199806574263, 6.531476985844397e-12, 0.13829581359726595,
          1.724738947709949e-12, 0.22746703542940375]
    riskt = 0.02356383470533441
    rett = 0.0005937393209710076
    @test isapprox(w1.weights, wt)
    @test isapprox(r1, riskt)
    @test isapprox(ret1, rett)

    w2 = optimise!(portfolio, Trad(; rm = rm, kelly = AKelly(), obj = obj))
    wt = [7.856868439678794e-13, 0.1989062583681603, 4.838778513927888e-12,
          0.054902892206828, 4.38111226861629e-11, 1.2803205819789809e-12,
          3.5416120724104257e-12, 0.10667398932256585, 1.3155352269357149e-12,
          1.858482119276245e-11, 0.048678037555666076, 0.09089397446667812,
          3.926648516381695e-12, 9.106515699454836e-12, 1.7422547807424147e-12,
          0.1341819985507733, 6.519246165182908e-12, 0.1382958141359801,
          1.543400238197745e-12, 0.22746703529635232]
    @test isapprox(w2.weights, wt)

    w3 = optimise!(portfolio, Trad(; rm = rm, kelly = EKelly(), obj = obj))
    wt = [3.558276949223149e-11, 0.19890649251730394, 8.339684482191303e-11,
          0.054903006138987936, 4.435282205930281e-10, 3.131042671094137e-11,
          1.1925439551200015e-11, 0.10667398512265627, 3.205009634146099e-11,
          2.005456583753527e-10, 0.04867780055964718, 0.09089395600647165,
          8.839854474688133e-12, 1.1690613364791602e-10, 5.6683355297695216e-11,
          0.13418187280163776, 1.0595822785383569e-10, 0.13829579900743022,
          5.7396745888772046e-11, 0.22746708666174134]
    @test isapprox(w3.weights, wt)

    obj = Utility(; l = l)
    w4 = optimise!(portfolio, Trad(; rm = rm, kelly = NoKelly(), obj = obj))
    r2 = calc_risk(portfolio, :Trad; rm = rm)
    ret2 = dot(portfolio.mu, w4.weights)
    wt = [2.3531676655820754e-12, 0.22146555850379093, 9.410028484870327e-12,
          0.05148290829152357, 8.0605983841845e-10, 4.9113218142397595e-12,
          8.131364518006022e-12, 0.11536086211760316, 2.271272286048353e-12,
          5.43251102138864e-11, 7.650522053564507e-10, 0.09513372436124735,
          8.884714671260909e-12, 1.3757061851213529e-11, 1.6291810113089304e-13,
          0.1432756744154531, 2.0775280711374397e-11, 0.14179358412385565,
          4.251506235430576e-12, 0.2314876864861804]
    riskt = 0.02356604972729362
    rett = 0.000633639299164722
    @test isapprox(w4.weights, wt)
    @test isapprox(r2, riskt)
    @test isapprox(ret2, rett)

    w5 = optimise!(portfolio, Trad(; rm = rm, kelly = AKelly(), obj = obj))
    wt = [7.89044712826026e-13, 0.22386204538926974, 3.552778484583874e-12,
          0.06247360288493247, 2.681553554799882e-10, 1.6414072098675413e-12,
          2.8859413630889454e-12, 0.11120554432711087, 9.715362217793387e-13,
          2.0393816091257544e-11, 3.5256904279870844e-10, 0.09247718794641346,
          3.1599126086787316e-12, 5.745570168144382e-12, 9.982334651996505e-14,
          0.13672420196765245, 6.549771094991111e-12, 0.1389988715113014,
          1.5187241475848804e-12, 0.234258545305287]
    @test isapprox(w5.weights, wt)

    w6 = optimise!(portfolio, Trad(; rm = rm, kelly = EKelly(), obj = obj))
    wt = [9.020812440261157e-12, 0.2238691771668263, 2.2902607458718274e-11,
          0.06250624343467237, 8.920948226738776e-10, 6.036964724458708e-12,
          2.2145160275491298e-12, 0.11119320768009527, 9.038009482320266e-12,
          7.6435766143155e-11, 8.44799817150313e-10, 0.09246929426507262,
          1.3660110791138543e-12, 2.810362809747963e-11, 1.1492651852637795e-11,
          0.13670473474202013, 3.690296159262254e-11, 0.13899056687783914,
          1.7157900326346932e-11, 0.2342667738759077]
    @test isapprox(w6.weights, wt, rtol = 5.0e-6)

    obj = Sharpe(; rf = rf)
    w7 = optimise!(portfolio, Trad(; rm = rm, kelly = NoKelly(), obj = obj))
    r3 = calc_risk(portfolio, :Trad; rm = rm)
    ret3 = dot(portfolio.mu, w7.weights)
    wt = [1.0342189964374973e-11, 1.400649279934334e-11, 1.0025902458601371e-11,
          6.108757972397652e-12, 0.35723529048659247, 1.2069564550657953e-11,
          1.2830347528094887e-11, 1.1781378314488614e-12, 6.5343202013864566e-12,
          2.895398409002917e-12, 9.118697983089066e-12, 1.0966314618191202e-11,
          1.3147762207425575e-11, 9.611939363025545e-12, 1.263250173104243e-11,
          0.21190800224274572, 0.4308567071123932, 7.712262968799656e-12,
          1.2412631429194858e-11, 6.675313779730199e-12]
    riskt = 0.03208927225110264
    rett = 0.0017686555804583674
    @test isapprox(w7.weights, wt)
    @test isapprox(r3, riskt)
    @test isapprox(ret3, rett)

    w8 = optimise!(portfolio, Trad(; rm = rm, kelly = AKelly(), obj = obj))
    wt = [1.2959197244759468e-11, 8.084356150422197e-11, 1.3926601479162983e-11,
          2.8785342853445117e-11, 0.32204068843816375, 3.214346139905109e-12,
          4.692943169531694e-12, 6.33836412905834e-11, 1.484504201576286e-10,
          3.804633552654395e-11, 1.9639729850630742e-11, 1.1407234782655726e-11,
          4.23415076667216e-12, 1.7187957086567557e-11, 5.342675786892118e-12,
          0.20669809375923995, 0.47126121713540664, 2.38746093868233e-11,
          1.625803285907461e-10, 2.8620587870227428e-11]
    @test isapprox(w8.weights, wt)

    w9 = optimise!(portfolio, Trad(; rm = rm, kelly = EKelly(), obj = obj))
    wt = [1.3801649259171437e-7, 8.592740180284502e-7, 1.4750175001348523e-7,
          2.93858579076417e-7, 0.33034161966700004, 4.25979385472981e-8,
          5.669708973524571e-8, 6.511546089373594e-7, 1.0217092350189244e-6,
          3.8588865642171967e-7, 2.0759750030993866e-7, 1.2423343866980414e-7,
          5.2774005027449543e-8, 1.8191259827821858e-7, 6.320894661292222e-8,
          0.20770466885250968, 0.46194751935242984, 2.4840026016179655e-7,
          1.412838309788135e-6, 3.0446463326205604e-7]
    @test isapprox(w9.weights, wt, rtol = 0.005)

    obj = MaxRet()
    w10 = optimise!(portfolio, Trad(; rm = rm, kelly = NoKelly(), obj = obj))
    r4 = calc_risk(portfolio, :Trad; rm = rm)
    ret4 = dot(portfolio.mu, w10.weights)
    wt = [8.730661626481806e-10, 1.4687309386680775e-9, 9.358654898947756e-10,
          2.186677678966605e-9, 0.9999999341280966, 3.156239683565371e-10,
          4.629682444037334e-10, 1.3343331058979577e-9, 5.69212053674475e-9,
          1.3749327860458163e-9, 7.161024888414711e-10, 5.748069307464309e-10,
          3.666660703394924e-10, 7.0140810552241e-10, 4.5086981398858524e-10,
          7.002103638459066e-9, 3.6268802221190136e-8, 9.019721807104911e-10,
          3.482364214437503e-9, 7.624890097546668e-10]
    riskt = 0.041844368667445314
    rett = 0.0019086272314632886
    @test isapprox(w10.weights, wt)
    @test isapprox(r4, riskt)
    @test isapprox(ret4, rett)

    w11 = optimise!(portfolio, Trad(; rm = rm, kelly = AKelly(), obj = obj))
    wt = [6.036096739587215e-12, 1.790094245743028e-11, 7.472431823401233e-12,
          2.6015061923204717e-11, 0.8577737313404759, 5.106563274225794e-12,
          1.900666972805193e-12, 1.798805120124093e-11, 9.180990810209664e-11,
          1.8285780713416517e-11, 3.878646392138336e-12, 2.6114409753401695e-13,
          4.039407801796174e-12, 3.3645983430819403e-12, 2.0794146894700064e-12,
          1.8675565751693112e-10, 0.14222626819696105, 8.128255004448968e-12,
          5.689786193015622e-11, 4.642665800172863e-12]
    @test isapprox(w11.weights, wt)

    w12 = optimise!(portfolio, Trad(; rm = rm, kelly = EKelly(), obj = obj))
    wt = [4.65046216906908e-9, 8.296253929547001e-9, 4.865391327970577e-9,
          1.0916852112918646e-8, 0.8926331516236782, 1.5238336619771253e-9,
          2.076243111622752e-9, 8.049225239703472e-9, 3.617749936955201e-8,
          7.86223613409023e-9, 3.89394397749308e-9, 2.8230386682124775e-9,
          1.6131006163903666e-9, 3.7199163646151806e-9, 1.9944778806862605e-9,
          6.473044506145028e-8, 0.10736665414274231, 5.092714368879538e-9,
          2.1903498344282783e-8, 4.044447060145975e-9]
    @test isapprox(w12.weights, wt, rtol = 5.0e-6)

    # Risk upper bound
    obj = MaxRet()
    rm.settings.ub = r1
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio, :Trad; rm = rm) <= r1 ||
          abs(calc_risk(portfolio, :Trad; rm = rm) - r1) < 1e-8

    rm.settings.ub = r2
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio, :Trad; rm = rm) <= r2 ||
          abs(calc_risk(portfolio, :Trad; rm = rm) - r2) < 5e-10

    rm.settings.ub = r3
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio, :Trad; rm = rm) <= r3

    rm.settings.ub = r4
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio, :Trad; rm = rm) <= r4

    obj = Sharpe(; rf = rf)
    rm.settings.ub = r1
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio, :Trad; rm = rm) <= r1 ||
          abs(calc_risk(portfolio, :Trad; rm = rm) - r1) < 5e-9

    rm.settings.ub = r2
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio, :Trad; rm = rm) <= r2 ||
          abs(calc_risk(portfolio, :Trad; rm = rm) - r2) < 5e-10

    rm.settings.ub = r3
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio, :Trad; rm = rm) <= r3

    rm.settings.ub = r4
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio, :Trad; rm = rm) <= r4

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
    @test dot(portfolio.mu, w15.weights) >= ret3 ||
          abs(dot(portfolio.mu, w15.weights) - ret3) < 1e-10

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
          abs(dot(portfolio.mu, w20.weights) - ret4) < 1e-10

    portfolio.mu_l = Inf
    rm = OWA(; w = owa_tg(200), formulation = OWAApprox())

    obj = MinRisk()
    w21 = optimise!(portfolio, Trad(; rm = rm, kelly = NoKelly(), obj = obj))
    r5 = calc_risk(portfolio, :Trad; rm = rm)
    ret5 = dot(portfolio.mu, w21.weights)
    wt = [2.1793176066144965e-10, 0.24657578304895264, 5.476679022837874e-10,
          0.041100700961631355, 1.745099342448546e-9, 2.0168069945246096e-10,
          7.629903785304511e-11, 0.11651659826585649, 1.9351039549294452e-10,
          1.0580904330304497e-9, 5.933842241374943e-8, 0.09732932587810658,
          5.619716498021303e-11, 6.782716439749812e-10, 3.4982414551865584e-10,
          0.14662256588896547, 5.824103810697346e-10, 0.12694239844522773,
          3.400868203663799e-10, 0.22491256212576752]
    riskt = 0.023582366401225324
    rett = 0.0006432235211782866
    @test isapprox(w21.weights, wt, rtol = 5.0e-5)
    @test isapprox(r5, riskt, rtol = 5.0e-7)
    @test isapprox(ret5, rett, rtol = 5.0e-6)
    @test isapprox(w21.weights, w1.weights, rtol = 0.25)
    @test isapprox(r5, r1, rtol = 0.001)
    @test isapprox(ret5, ret1, rtol = 0.1)

    w22 = optimise!(portfolio, Trad(; rm = rm, kelly = AKelly(), obj = obj))
    wt = [2.132183776397635e-10, 0.24657765819583657, 5.355024854123618e-10,
          0.04109911857321909, 1.7029194284107386e-9, 1.974679372397374e-10,
          7.446166082524666e-11, 0.11651653133071553, 1.8952844225356302e-10,
          1.0314865628216637e-9, 6.295221219719734e-8, 0.09732926904417218,
          5.470853792167843e-11, 6.745287801746886e-10, 3.428894296510097e-10,
          0.1466228087872955, 5.711384009324977e-10, 0.12694288445100937,
          3.3336697140072015e-10, 0.2249116607443225]
    @test isapprox(w22.weights, wt, rtol = 5.0e-5)
    @test isapprox(w22.weights, w2.weights, rtol = 0.25)

    w23 = optimise!(portfolio, Trad(; rm = rm, kelly = EKelly(), obj = obj))
    wt = [1.927837301458917e-10, 0.2465727935166203, 4.5776197117687873e-10,
          0.0411034353688984, 1.4160012847338463e-9, 1.7931052490904626e-10,
          7.492504213032373e-11, 0.11651631583960065, 1.7182805746854656e-10,
          8.718361635547206e-10, 6.868975884764904e-8, 0.09732867628494225,
          5.796922323295118e-11, 6.094684782004051e-10, 3.067454763778319e-10,
          0.14662102857422757, 5.026263032275627e-10, 0.12694436503613155,
          2.957487773764835e-10, 0.22491331155281538]
    @test isapprox(w23.weights, wt, rtol = 5.0e-5)
    @test isapprox(w23.weights, w3.weights, rtol = 0.25)

    obj = Utility(; l = l)
    w24 = optimise!(portfolio, Trad(; rm = rm, kelly = NoKelly(), obj = obj))
    r6 = calc_risk(portfolio, :Trad; rm = rm)
    ret6 = dot(portfolio.mu, w24.weights)
    wt = [4.031429185256284e-10, 0.24853621593089034, 1.0564435079928515e-9,
          0.041227339264945304, 3.187793853320347e-8, 2.993481845211221e-10,
          1.316251677752441e-10, 0.11775482753120951, 4.0247404407692655e-10,
          2.4553272342813325e-9, 8.117272668206023e-9, 0.09806143571966024,
          9.667832808949718e-11, 1.0479300493959856e-9, 5.16355132377166e-10,
          0.14808310631257574, 1.3560319804684306e-9, 0.12226287288483399,
          7.148464022467191e-10, 0.22407415388047078]
    riskt = 0.023585657006845066
    rett = 0.0006461184324927211
    @test isapprox(w24.weights, wt, rtol = 5.0e-5)
    @test isapprox(r6, riskt, rtol = 1.0e-7)
    @test isapprox(ret6, rett, rtol = 5.0e-6)
    @test isapprox(w24.weights, w4.weights, rtol = 0.1)
    @test isapprox(r6, r2, rtol = 0.001)
    @test isapprox(ret6, ret2, rtol = 0.05)

    w25 = optimise!(portfolio, Trad(; rm = rm, kelly = AKelly(), obj = obj))
    wt = [1.507133941066285e-10, 0.24827976788672826, 3.73990694940837e-10,
          0.041308890243378256, 7.675438081297324e-9, 1.1154812723593603e-10,
          4.951489768730557e-11, 0.11762133998644565, 1.4637086998562598e-10,
          8.756011450879805e-10, 6.943850562800141e-9, 0.09794572252106948,
          3.5923970291357996e-11, 4.3057334069665553e-10, 2.002929167365126e-10,
          0.14789685246583284, 5.19138038484229e-10, 0.12278739931801369,
          2.6551070295904287e-10, 0.22416000980006515]
    @test isapprox(w25.weights, wt, rtol = 1.0e-5)
    @test isapprox(w25.weights, w5.weights, rtol = 0.1)

    w26 = optimise!(portfolio, Trad(; rm = rm, kelly = EKelly(), obj = obj))
    wt = [1.8080300946247023e-10, 0.24828292527938153, 4.68121589287319e-10,
          0.04130630822882187, 1.2359535796409274e-8, 1.3734305268211817e-10,
          6.559527877655717e-11, 0.11762157323389853, 1.814435581688314e-10,
          1.0697620823533454e-9, 2.1021006172059525e-9, 0.09794654012223465,
          5.079896932410669e-11, 4.387939949600902e-10, 2.2506554283303623e-10,
          0.14789817630497223, 5.74829787739748e-10, 0.12278545945634213,
          3.126744048715974e-10, 0.2241589992074813]
    @test isapprox(w26.weights, wt, rtol = 5.0e-6)
    @test isapprox(w26.weights, w6.weights, rtol = 0.1)

    obj = Sharpe(; rf = rf)
    w27 = optimise!(portfolio, Trad(; rm = rm, kelly = NoKelly(), obj = obj))
    r7 = calc_risk(portfolio, :Trad; rm = rm)
    ret7 = dot(portfolio.mu, w27.weights)
    wt = [2.611247264961084e-10, 1.8600657774984216e-9, 2.7626934335103416e-10,
          5.497788498901174e-10, 0.3290378114659103, 8.224811202970948e-11,
          1.068509794521808e-10, 1.0698183121278107e-9, 1.6147966531606507e-9,
          7.115598011937777e-10, 3.984269424697578e-10, 2.4827950070497723e-10,
          1.0188959979493673e-10, 3.5208451215754276e-10, 1.1948530978332868e-10,
          0.20499862760973586, 0.4659635498088353, 4.740736136016396e-10,
          2.3087600736393448e-9, 5.80006498885295e-10]
    riskt = 0.03204363189094902
    rett = 0.00176571179911687
    @test isapprox(w27.weights, wt, rtol = 0.0001)
    @test isapprox(r7, riskt, rtol = 5.0e-6)
    @test isapprox(ret7, rett, rtol = 5.0e-6)
    @test isapprox(w27.weights, w7.weights, rtol = 0.1)
    @test isapprox(r7, r3, rtol = 0.005)
    @test isapprox(ret7, ret3, rtol = 0.005)

    w28 = optimise!(portfolio, Trad(; rm = rm, kelly = AKelly(), obj = obj))
    wt = [1.869325822862471e-10, 1.4328061947056766e-9, 1.983624339909501e-10,
          3.9049540188495594e-10, 0.2993249703403086, 5.91655997016558e-11,
          7.591779016922828e-11, 9.543812870273496e-10, 1.629098331489297e-9,
          5.646270702560161e-10, 3.0669712261634527e-10, 1.7813852329496214e-10,
          7.36556294671998e-11, 2.624444093298053e-10, 8.577535879309574e-11,
          0.20566930367997732, 0.4950057161442303, 3.6134043743821436e-10,
          2.615852679683726e-9, 4.5979294701178386e-10]
    @test isapprox(w28.weights, wt, rtol = 1.0e-5)
    @test isapprox(w28.weights, w8.weights, rtol = 0.1)

    w29 = optimise!(portfolio, Trad(; rm = rm, kelly = EKelly(), obj = obj))
    wt = [9.221062320825582e-8, 7.047478800634529e-7, 9.781708009684867e-8,
          1.916840895248151e-7, 0.2993421988101789, 2.9430761608320223e-8,
          3.7627543652153636e-8, 4.689749347827317e-7, 8.149066126733291e-7,
          2.7810815645071277e-7, 1.5112599835099536e-7, 8.779800183945856e-8,
          3.6587065526347325e-8, 1.2935364553010815e-7, 4.243295514320968e-8,
          0.20608699061385133, 0.4945659352314525, 1.7836651758245156e-7,
          1.3084916989859224e-6, 2.2568095228872053e-7]
    @test isapprox(w29.weights, wt, rtol = 0.001)
    @test isapprox(w29.weights, w9.weights, rtol = 0.1)

    obj = MaxRet()
    w30 = optimise!(portfolio, Trad(; rm = rm, kelly = NoKelly(), obj = obj))
    r8 = calc_risk(portfolio, :Trad; rm = rm)
    ret8 = dot(portfolio.mu, w30.weights)
    wt = [3.003468043166542e-8, 4.989680902144069e-8, 3.236147859233924e-8,
          6.925347727164586e-8, 0.999998549648788, 8.880071192002583e-9,
          1.4192005624622162e-8, 4.580488074279412e-8, 1.4087036390860031e-7,
          4.706727710273597e-8, 2.4012363049655272e-8, 1.8498467909574943e-8,
          1.0678075715091602e-8, 2.3437721208674016e-8, 1.3737369143792671e-8,
          1.6398960158911142e-7, 6.023666956203431e-7, 3.110222932878985e-8,
          9.835275183965494e-8, 2.581489271209344e-8]
    riskt = 0.041844336232849194
    rett = 0.0019086263079852075
    @test isapprox(w30.weights, wt)
    @test isapprox(r8, riskt)
    @test isapprox(ret8, rett)

    w31 = optimise!(portfolio, Trad(; rm = rm, kelly = AKelly(), obj = obj))
    wt = [2.1877877724598778e-10, 5.423634965391811e-10, 2.5253431909718113e-10,
          7.867829458604338e-10, 0.8577730688248231, 8.617306754355166e-11,
          1.881760474509161e-11, 5.340082262051755e-10, 2.6334834681217134e-9,
          5.48174866618407e-10, 1.399929067726362e-10, 4.649356854273397e-11,
          7.430736924775098e-11, 1.2724883443711166e-10, 2.7593276139962158e-11,
          5.004679883728293e-9, 0.1422269180687958, 2.5760424370343565e-10,
          1.6464003712654701e-9, 1.6094394207994044e-10]
    @test isapprox(w31.weights, wt)
    @test isapprox(w31.weights, w11.weights, rtol = 5.0e-6)

    w32 = optimise!(portfolio, Trad(; rm = rm, kelly = EKelly(), obj = obj))
    wt = [7.495244566211183e-9, 1.3356715214376525e-8, 7.779717687780013e-9,
          1.7599686482177285e-8, 0.8926338954347407, 2.5033970331565238e-9,
          3.377725979808072e-9, 1.2898997612987054e-8, 5.788946814686859e-8,
          1.2506319355284187e-8, 6.269268628602493e-9, 4.546684802206718e-9,
          2.6316119362164115e-9, 5.991019267313653e-9, 3.2331818591355224e-9,
          1.019630349555436e-7, 0.10736579479363274, 8.18324477422888e-9,
          3.505072621551823e-8, 6.4955820348114385e-9]
    @test isapprox(w32.weights, wt, rtol = 5.0e-6)
    @test isapprox(w32.weights, w12.weights, rtol = 5.0e-6)

    # Risk upper bound
    obj = MaxRet()
    rm.settings.ub = r5 * 1.001
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio, :Trad; rm = rm) <= r5 * 1.001

    rm.settings.ub = r6 * 1.001
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio, :Trad; rm = rm) <= r6 * 1.001

    rm.settings.ub = r7
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio, :Trad; rm = rm) <= r7

    rm.settings.ub = r8
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio, :Trad; rm = rm) <= r8

    obj = Sharpe(; rf = rf)
    rm.settings.ub = r5 * 1.001
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio, :Trad; rm = rm) <= r5 * 1.001

    rm.settings.ub = r6 * 1.001
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio, :Trad; rm = rm) <= r6 * 1.001

    rm.settings.ub = r7
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio, :Trad; rm = rm) <= r7

    rm.settings.ub = r8
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio, :Trad; rm = rm) <= r8

    # Ret lower bound
    rm.settings.ub = Inf
    obj = MinRisk()
    portfolio.mu_l = ret5
    w33 = optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test dot(portfolio.mu, w33.weights) >= ret5

    portfolio.mu_l = ret6
    w34 = optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test dot(portfolio.mu, w34.weights) >= ret6

    portfolio.mu_l = ret7
    w35 = optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test dot(portfolio.mu, w35.weights) >= ret7

    portfolio.mu_l = ret8
    w36 = optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test dot(portfolio.mu, w16.weights) >= ret8

    obj = Sharpe(; rf = rf)
    portfolio.mu_l = ret5
    w37 = optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test dot(portfolio.mu, w17.weights) >= ret1

    portfolio.mu_l = ret6
    w38 = optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test dot(portfolio.mu, w18.weights) >= ret6

    portfolio.mu_l = ret7
    w39 = optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test dot(portfolio.mu, w39.weights) >= ret7

    portfolio.mu_l = ret8
    w40 = optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test dot(portfolio.mu, w40.weights) >= ret8
end

@testset "All negative expected returns" begin
    path = joinpath(@__DIR__, "assets/stock_prices3.csv")
    prices = TimeArray(CSV.File(path); timestamp = :date)
    portfolio = Portfolio(; prices = prices,
                          solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                           :check_sol => (allow_local = true,
                                                                          allow_almost = true),
                                                           :params => Dict("verbose" => false,
                                                                           "max_step_fraction" => 0.75))))

    asset_statistics!(portfolio; set_kurt = false, set_skurt = false, set_skew = false,
                      set_sskew = false)
    obj = Sharpe(; rf = 3.5 / 100 / 252)
    w1 = optimise!(portfolio, Trad(; obj = obj))
    wt = [0.004167110904562332, 0.0194731557988081, 0.018485712797445205,
          0.04390022593613357, 0.004422372076608825, 0.03030102917315649,
          0.7652011327490768, 0.07019028482746328, 0.04385897573674533]
    @test isapprox(w1.weights, wt)

    portfolio.l2 = 0.1
    w2 = optimise!(portfolio, Trad(; obj = obj))
    wt = [0.09035196773333391, 0.11398449552304651, 0.11381392960375913,
          0.11755739519087734, 0.09223364272674096, 0.11582484774604462,
          0.12039044433099177, 0.1182121523391645, 0.11763112480604135]
    @test isapprox(w2.weights, wt)

    w3 = optimise!(portfolio, Trad(; obj = obj, kelly = EKelly()))
    wt = [0.08373523261171062, 0.11445906511775665, 0.11428659252691513,
          0.11993149415789296, 0.08628997612519118, 0.1168088505947512, 0.12386205654049225,
          0.12050873207101519, 0.12011800025427488]
    @test isapprox(w3.weights, wt)
end
