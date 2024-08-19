@testset "GMD" begin
    portfolio = Portfolio2(; prices = prices[(end - 200):end],
                           solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                            :params => Dict("verbose" => false,
                                                                            "max_step_fraction" => 0.75,
                                                                            "max_iter" => 100,
                                                                            "equilibrate_max_iter" => 20))))
    asset_statistics2!(portfolio)
    rm = GMD2(; owa = OWASettings(; approx = false))

    obj = MinRisk()
    w1 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r1 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret1 = dot(portfolio.mu, w1.weights)
    wt = [2.859922116198051e-10, 1.2839851061414282e-9, 6.706646698191418e-10,
          0.011313263237504052, 0.041188384886560354, 0.021164906062678113,
          9.703581331630189e-11, 0.04727424011116681, 5.765539509407631e-10,
          4.5707249638450353e-10, 0.07079102697763319, 2.3691876986431332e-11,
          6.862345293974742e-11, 0.2926160718937385, 2.627931180682999e-11,
          0.014804760223130658, 0.055360563578446625, 0.22572350787381906,
          0.008097074123824198, 0.2116661975415996]
    riskt = 0.007973013868952676
    rett = 0.00029353687778951485
    @test isapprox(w1.weights, wt, rtol = 5.0e-7)
    @test isapprox(r1, riskt)
    @test isapprox(ret1, rett, rtol = 1.0e-6)

    w2 = optimise2!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [5.252616779346811e-10, 2.365839628970636e-9, 1.22839420711326e-9,
          0.011366311199872714, 0.04115038403356847, 0.021118569382081374,
          1.783012445570223e-10, 0.04726445861603414, 1.0575646680853942e-9,
          8.388180967197924e-10, 0.07085096688461949, 4.42223650770833e-11,
          1.254597253903546e-10, 0.29253116010000035, 4.8831001454051914e-11,
          0.014796386353461968, 0.05529636268909762, 0.2257280181516915,
          0.008121109451917503, 0.21177626672496228]
    @test isapprox(w2.weights, wt, rtol = 5.0e-7)

    w3 = optimise2!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    wt = [5.322679956671707e-10, 2.22871430513242e-9, 1.1710008341832608e-9,
          0.01134477276755037, 0.04116336482965813, 0.02111819928448936,
          2.1636233704677454e-10, 0.0472560347538972, 1.0050858820636962e-9,
          8.089975732994674e-10, 0.07082525480344888, 9.263513620208697e-11,
          1.6960069074338912e-10, 0.2925587830137476, 9.636779050056544e-11,
          0.014800843066311553, 0.05532148371288869, 0.22573823250579925,
          0.008118735919623265, 0.21175428902155313]
    @test isapprox(w3.weights, wt, rtol = 1.0e-6)

    obj = Util(; l = l)
    w4 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r2 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret2 = dot(portfolio.mu, w4.weights)
    wt = [4.395442944440727e-10, 2.1495615257047445e-9, 8.700158416075159e-10,
          0.006589303508967539, 0.05330845589356504, 7.632737386668005e-10,
          9.831583865942048e-11, 0.06008335161013898, 1.3797121555924764e-9,
          3.0160588760343415e-9, 0.06394372429204083, 4.486199590323317e-11,
          1.0975948346699443e-10, 0.24314317639043878, 5.90094645737884e-11,
          0.018483948217979402, 0.08594753958504305, 0.216033164253688, 0.05666930366548,
          0.19579802365254514]
    riskt = 0.008023853567234033
    rett = 0.0005204188067663664
    @test isapprox(w4.weights, wt, rtol = 1.0e-6)
    @test isapprox(r2, riskt)
    @test isapprox(ret2, rett, rtol = 5.0e-8)

    w5 = optimise2!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [2.1362512461147959e-10, 1.0473426830174378e-9, 4.232904776563241e-10,
          0.006623775536322704, 0.053261985949698866, 3.7469349330098336e-10,
          4.858498678278164e-11, 0.05983738243189082, 6.626203326440813e-10,
          1.432599593268026e-9, 0.06407652576443064, 2.2558984339781933e-11,
          5.3690642468248475e-11, 0.24323109220603928, 2.96486691492619e-11,
          0.018454184890627775, 0.08594373007217042, 0.21613806337839309,
          0.05652374385950317, 0.1959095116022682]
    @test isapprox(w5.weights, wt, rtol = 5.0e-6)

    w6 = optimise2!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    wt = [5.8420951269574795e-9, 2.4391916450367587e-8, 1.0071995347751536e-8,
          0.006659057409806418, 0.05312958085464967, 8.334452892012202e-9,
          1.929663610765478e-9, 0.059635994176648714, 2.1986603285155345e-8,
          3.117814112069695e-8, 0.06410822943092967, 1.5583858339937482e-9,
          2.0082967075430054e-9, 0.2433291222014175, 1.5338984655275193e-9,
          0.018559739720678686, 0.0859661153175324, 0.21612757832310056,
          0.05665122109537618, 0.19583325263441156]
    @test isapprox(w6.weights, wt, rtol = 0.005)

    obj = SR(; rf = rf)
    w7 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r3 = calc_risk(portfolio; type = :Trad2, rm = rm)
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

    w8 = optimise2!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [1.0154805469670173e-10, 2.381730742924776e-10, 1.1584205630210449e-10,
          2.1757905443264182e-10, 0.20152993689645365, 2.9930412807375975e-11,
          3.64391888559604e-11, 0.02863690188953056, 0.00580640153065922,
          6.433774573403564e-10, 1.8439563108773455e-10, 5.5270018825247095e-11,
          3.6021609124707604e-11, 1.4359882175835754e-10, 4.8136096688424435e-11,
          0.09660151868641804, 0.3822570707288953, 3.075257417034269e-10,
          0.2851681678956018, 2.14604154394175e-10]
    @test isapprox(w8.weights, wt)

    w9 = optimise2!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    wt = [2.1413560639408036e-7, 4.824835233541495e-7, 2.406914746153931e-7,
          4.666374251657947e-7, 0.2071309575719716, 6.549070937928946e-8,
          8.258348668841516e-8, 0.01582848177484498, 0.012569567871964202,
          1.0990851999992477e-6, 3.3970785361076e-7, 1.231246854165843e-7,
          7.829817011460847e-8, 2.7575597570933386e-7, 1.0245173736381925e-7,
          0.09828155080487408, 0.3973840510410685, 5.298880345954879e-7, 0.2688009136340711,
          3.76967323193072e-7]
    @test isapprox(w9.weights, wt, rtol = 0.001)

    obj = MaxRet()
    w10 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r4 = calc_risk(portfolio; type = :Trad2, rm = rm)
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

    w11 = optimise2!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [5.995668682294329e-12, 1.778824643654798e-11, 7.422520403984214e-12,
          2.585579633200434e-11, 0.8577737311670789, 5.0842729898340874e-12,
          1.891888971224657e-12, 1.7873758325289183e-11, 9.127375624650092e-11,
          1.8171912621743217e-11, 3.850582248871377e-12, 2.5508491084881004e-13,
          4.021631383731875e-12, 3.3391228496613597e-12, 2.0662267001955667e-12,
          1.856978015488069e-10, 0.14222626837308944, 8.073644265708836e-12,
          5.656088636719738e-11, 4.608688065458597e-12]
    @test isapprox(w11.weights, wt)

    w12 = optimise2!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    wt = [9.62957942044306e-9, 1.7177497817937903e-8, 1.0071080871110762e-8,
          2.2600658264306737e-8, 0.8926312059112995, 3.1604103611572677e-9,
          4.3001093397322324e-9, 1.666218101253023e-8, 7.49780862909425e-8,
          1.6269274961255667e-8, 8.062707576138389e-9, 5.845979308416873e-9,
          3.341497686912086e-9, 7.701378850984419e-9, 4.129869947133951e-9,
          1.3414099735463856e-7, 0.10736839173350264, 1.0542586207839114e-8,
          4.536866275330717e-8, 8.37263993939097e-9]
    @test isapprox(w12.weights, wt, rtol = 5.0e-6)

    # Risk upper bound
    obj = MaxRet()
    rm.settings.ub = r1
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r1 ||
          abs(calc_risk(portfolio; type = :Trad2, rm = rm) - r1) < 5e-7

    rm.settings.ub = r2
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r2 ||
          abs(calc_risk(portfolio; type = :Trad2, rm = rm) - r2) < 5e-8

    rm.settings.ub = r3
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r3 ||
          abs(calc_risk(portfolio; type = :Trad2, rm = rm) - r3) < 5e-9

    rm.settings.ub = r4
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r4 ||
          abs(calc_risk(portfolio; type = :Trad2, rm = rm) - r4) < 1e-9

    obj = SR(; rf = rf)
    rm.settings.ub = r1
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r1 ||
          abs(calc_risk(portfolio; type = :Trad2, rm = rm) - r1) < 1e-7

    rm.settings.ub = r2
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r2 ||
          abs(calc_risk(portfolio; type = :Trad2, rm = rm) - r2) < 1e-8

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
    @test dot(portfolio.mu, w15.weights) >= ret3 ||
          abs(dot(portfolio.mu, w15.weights) - ret3) < 1e-10

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
          abs(dot(portfolio.mu, w20.weights) - ret4) < 1e-10

    portfolio.mu_l = Inf
    rm = GMD2(; owa = OWASettings())

    obj = MinRisk()
    w21 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r5 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret5 = dot(portfolio.mu, w21.weights)
    wt = [5.046569074460872e-11, 1.9446427133014524e-10, 1.5235479475306312e-10,
          0.012753556176921463, 0.041101880902372545, 0.020275435571924925,
          1.9450297598820413e-11, 0.04814436030268956, 9.780643160848096e-11,
          8.453394562121676e-11, 0.0718300148873752, 7.332574672625954e-12,
          1.6469076767775372e-11, 0.2856668045661774, 7.943393191933155e-12,
          0.014433628410132425, 0.05034866468945478, 0.23180755976747225,
          0.010375821895801248, 0.21326227219885777]
    riskt = 0.007973655098253875
    rett = 0.0002960041138012962
    @test isapprox(w21.weights, wt, rtol = 5.0e-6)
    @test isapprox(r5, riskt)
    @test isapprox(ret5, rett, rtol = 1.0e-6)
    @test isapprox(w21.weights, w1.weights, rtol = 0.05)
    @test isapprox(r5, r1, rtol = 0.0001)
    @test isapprox(ret5, ret1, rtol = 0.01)

    w22 = optimise2!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [6.085958730777325e-11, 2.368595308570172e-10, 1.8274079976986026e-10,
          0.012753559162711264, 0.04110189431996443, 0.020275527798981435,
          2.3425679309561966e-11, 0.04814426851732857, 1.1780033540776392e-10,
          1.0185698129397273e-10, 0.07183004439341859, 8.83400173557485e-12,
          1.9904438997433556e-11, 0.28566668880855195, 9.586292536707711e-12,
          0.014433659588712983, 0.050348734289227565, 0.23180750860480773,
          0.010375871927472354, 0.2132622418269556]
    @test isapprox(w22.weights, wt, rtol = 1.0e-6)
    @test isapprox(w22.weights, w2.weights, rtol = 0.05)

    w23 = optimise2!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    wt = [3.5257931606306207e-9, 1.3520364528236652e-8, 1.035605265254761e-8,
          0.012750804249588226, 0.041099729362825785, 0.02027745868985658,
          1.4073123777014541e-9, 0.048146705349223565, 6.684729921496123e-9,
          5.825310937899827e-9, 0.0718279647358401, 5.895262260980076e-10,
          1.2113369870119272e-9, 0.2856609915588955, 6.31932520036972e-10,
          0.01443337291334594, 0.050356148115682245, 0.23180458433517367,
          0.010378788680789746, 0.21326340825641937]
    @test isapprox(w23.weights, wt, rtol = 5.0e-5)
    @test isapprox(w23.weights, w3.weights, rtol = 0.05)

    obj = Util(; l = l)
    w24 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r6 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret6 = dot(portfolio.mu, w24.weights)
    wt = [5.7077823480874523e-11, 2.2979249808299912e-10, 1.1918317763741275e-10,
          0.007251925367368971, 0.05475338034046398, 9.483225869723537e-11,
          1.764714083719282e-11, 0.05983388783711289, 1.6658748928695604e-10,
          4.5304609055236015e-10, 0.06474254056499605, 1.0812568575742713e-11,
          2.0332472270971213e-11, 0.2401434357261637, 1.265342197262604e-11,
          0.019952573814651816, 0.08099015670705892, 0.21692370891990106,
          0.05867573585124927, 0.1967326536890683]
    riskt = 0.008024389698787715
    rett = 0.0005208058261359645
    @test isapprox(w24.weights, wt, rtol = 1.0e-6)
    @test isapprox(r6, riskt)
    @test isapprox(ret6, rett, rtol = 1.0e-7)
    @test isapprox(w24.weights, w4.weights, rtol = 0.05)
    @test isapprox(r6, r2, rtol = 0.0001)
    @test isapprox(ret6, ret2, rtol = 0.001)

    w25 = optimise2!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [2.0555226226463499e-10, 8.582006816920231e-10, 4.380531604344673e-10,
          0.007279467671437155, 0.054711192884278, 3.4897483970291476e-10,
          5.817408610429655e-11, 0.05980275462460995, 6.141055750628744e-10,
          1.6620416611085704e-9, 0.06488986087388922, 3.264876608092541e-11,
          6.795048142488927e-11, 0.24024239414062037, 3.951178809096139e-11,
          0.019958448469100377, 0.08084194043242174, 0.2170637586767573,
          0.058376885399118915, 0.19683329250255382]
    @test isapprox(w25.weights, wt, rtol = 5.0e-6)
    @test isapprox(w25.weights, w5.weights, rtol = 0.05)

    w26 = optimise2!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    wt = [7.343013549725222e-10, 2.8814547912921296e-9, 1.4729077903885225e-9,
          0.007283938786415164, 0.054704074988421954, 1.1664799677394659e-9,
          2.5366014692458325e-10, 0.05979332821166699, 2.0487026360819167e-9,
          5.189468631037292e-9, 0.06489522311639705, 1.7292554019928242e-10,
          2.8497265717059424e-10, 0.24026432699966493, 1.9543407410733274e-10,
          0.019956465494764018, 0.08082674094028784, 0.21707367770984815,
          0.05835339372972919, 0.19684881562249712]
    @test isapprox(w26.weights, wt, rtol = 1.0e-6)
    @test isapprox(w26.weights, w6.weights, rtol = 0.05)

    obj = SR(; rf = rf)
    w27 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r7 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret7 = dot(portfolio.mu, w27.weights)
    wt = [2.8630910311281004e-9, 6.522667183136477e-9, 3.2312642303224394e-9,
          6.1021180209421834e-9, 0.20912149394556778, 8.862751181668007e-10,
          1.0696851012842994e-9, 0.008619909225730223, 0.022327617734427343,
          1.607926764101964e-8, 4.856182910997696e-9, 1.5653323119111666e-9,
          1.0609698078389692e-9, 3.866791245627318e-9, 1.3958882366277933e-9,
          0.09548197220265975, 0.4040165574840697, 7.793230561888217e-9,
          0.26043238664623924, 5.468542510279709e-9]
    riskt = 0.010796650621936266
    rett = 0.0016610970146746496
    @test isapprox(w27.weights, wt, rtol = 5.0e-5)
    @test isapprox(r7, riskt, rtol = 5.0e-6)
    @test isapprox(ret7, rett, rtol = 1.0e-6)
    @test isapprox(w27.weights, w7.weights, rtol = 0.05)
    @test isapprox(r7, r3, rtol = 0.005)
    @test isapprox(ret7, ret3, rtol = 0.005)

    w28 = optimise2!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [2.1637878014410733e-9, 5.103165771742511e-9, 2.464752962698745e-9,
          4.502318468707382e-9, 0.19864989443943193, 6.784553823706463e-10,
          7.948150192946989e-10, 0.026307833960794825, 0.002644812882983666,
          1.413785835248894e-8, 4.062428337636719e-9, 1.1671706795014532e-9,
          8.029611871397618e-10, 3.1361580973496052e-9, 1.0530943509630832e-9,
          0.09468140613232137, 0.38692982998084924, 6.826031894899593e-9, 0.29078617090203,
          4.8085907234947415e-9]
    @test isapprox(w28.weights, wt, rtol = 0.0001)
    @test isapprox(w28.weights, w8.weights, rtol = 0.05)

    w29 = optimise2!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    wt = [9.766000625765347e-9, 2.2356909854732345e-8, 1.0999572910841263e-8,
          2.0992718586594702e-8, 0.20424685102403345, 3.0206324924799093e-9,
          3.691586679161349e-9, 0.013610921809396461, 0.0129480513890126,
          5.32503156262045e-8, 1.6319216662991532e-8, 5.439655785968068e-9,
          3.6000288893835075e-9, 1.3089163588373258e-8, 4.735473682745068e-9,
          0.09643685013934791, 0.40173945163361796, 2.5925271202608417e-8,
          0.27101766238662417, 1.843142091277025e-8]
    @test isapprox(w29.weights, wt, rtol = 1.0e-5)
    @test isapprox(w29.weights, w9.weights, rtol = 0.05)

    obj = MaxRet()
    w30 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r8 = calc_risk(portfolio; type = :Trad2, rm = rm)
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

    w31 = optimise2!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [2.1877877724598778e-10, 5.423634965391811e-10, 2.5253431909718113e-10,
          7.867829458604338e-10, 0.8577730688248231, 8.617306754355166e-11,
          1.881760474509161e-11, 5.340082262051755e-10, 2.6334834681217134e-9,
          5.48174866618407e-10, 1.399929067726362e-10, 4.649356854273397e-11,
          7.430736924775098e-11, 1.2724883443711166e-10, 2.7593276139962158e-11,
          5.004679883728293e-9, 0.1422269180687958, 2.5760424370343565e-10,
          1.6464003712654701e-9, 1.6094394207994044e-10]
    @test isapprox(w31.weights, wt)
    @test isapprox(w31.weights, w11.weights, rtol = 5.0e-6)

    w32 = optimise2!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    wt = [9.582076829858595e-9, 1.7100009371334017e-8, 1.001142853853447e-8,
          2.2544068403868613e-8, 0.8926288017342953, 3.1237536310195203e-9,
          4.303736780951003e-9, 1.6595082389931922e-8, 7.312445684301714e-8,
          1.619345152951524e-8, 8.028153283280456e-9, 5.818797312570166e-9,
          3.3477297942616325e-9, 7.678312506638265e-9, 4.134196407903452e-9,
          1.2969240847583332e-7, 0.1073708034653235, 1.0508357801476938e-8,
          4.467105946746357e-8, 8.34330196221366e-9]
    @test isapprox(w32.weights, wt, rtol = 1.0e-5)
    @test isapprox(w32.weights, w12.weights, rtol = 5.0e-6)

    # Risk upper bound
    obj = MaxRet()
    rm.settings.ub = r5 * 1.01
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r5 * 1.01

    rm.settings.ub = r6
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r6

    rm.settings.ub = r7
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r7

    rm.settings.ub = r8
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r8

    obj = SR(; rf = rf)
    rm.settings.ub = r5 * 1.01
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r5 * 1.01

    rm.settings.ub = r6
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r6

    rm.settings.ub = r7
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r7

    rm.settings.ub = r8
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r8

    # Ret lower bound
    rm.settings.ub = Inf
    obj = MinRisk()
    portfolio.mu_l = ret5
    w33 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w33.weights) >= ret5

    portfolio.mu_l = ret6
    w34 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w34.weights) >= ret6

    portfolio.mu_l = ret7
    w35 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w35.weights) >= ret7

    portfolio.mu_l = ret8
    w36 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w16.weights) >= ret8

    obj = SR(; rf = rf)
    portfolio.mu_l = ret5
    w37 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w17.weights) >= ret1

    portfolio.mu_l = ret6
    w38 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w18.weights) >= ret6

    portfolio.mu_l = ret7
    w39 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w39.weights) >= ret7

    portfolio.mu_l = ret8
    w40 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w40.weights) >= ret8
end

@testset "TG" begin
    portfolio = Portfolio2(; prices = prices[(end - 200):end],
                           solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                            :params => Dict("verbose" => false,
                                                                            "max_step_fraction" => 0.75,
                                                                            "max_iter" => 100,
                                                                            "equilibrate_max_iter" => 20))))
    asset_statistics2!(portfolio)
    rm = TG2(; owa = OWASettings(; approx = false))

    obj = MinRisk()
    w1 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r1 = calc_risk(portfolio; type = :Trad2, rm = rm)
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

    w2 = optimise2!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [7.856868439678794e-13, 0.1989062583681603, 4.838778513927888e-12,
          0.054902892206828, 4.38111226861629e-11, 1.2803205819789809e-12,
          3.5416120724104257e-12, 0.10667398932256585, 1.3155352269357149e-12,
          1.858482119276245e-11, 0.048678037555666076, 0.09089397446667812,
          3.926648516381695e-12, 9.106515699454836e-12, 1.7422547807424147e-12,
          0.1341819985507733, 6.519246165182908e-12, 0.1382958141359801,
          1.543400238197745e-12, 0.22746703529635232]
    @test isapprox(w2.weights, wt)

    w3 = optimise2!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    wt = [3.558276949223149e-11, 0.19890649251730394, 8.339684482191303e-11,
          0.054903006138987936, 4.435282205930281e-10, 3.131042671094137e-11,
          1.1925439551200015e-11, 0.10667398512265627, 3.205009634146099e-11,
          2.005456583753527e-10, 0.04867780055964718, 0.09089395600647165,
          8.839854474688133e-12, 1.1690613364791602e-10, 5.6683355297695216e-11,
          0.13418187280163776, 1.0595822785383569e-10, 0.13829579900743022,
          5.7396745888772046e-11, 0.22746708666174134]
    @test isapprox(w3.weights, wt)

    obj = Util(; l = l)
    w4 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r2 = calc_risk(portfolio; type = :Trad2, rm = rm)
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

    w5 = optimise2!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [7.89044712826026e-13, 0.22386204538926974, 3.552778484583874e-12,
          0.06247360288493247, 2.681553554799882e-10, 1.6414072098675413e-12,
          2.8859413630889454e-12, 0.11120554432711087, 9.715362217793387e-13,
          2.0393816091257544e-11, 3.5256904279870844e-10, 0.09247718794641346,
          3.1599126086787316e-12, 5.745570168144382e-12, 9.982334651996505e-14,
          0.13672420196765245, 6.549771094991111e-12, 0.1389988715113014,
          1.5187241475848804e-12, 0.234258545305287]
    @test isapprox(w5.weights, wt)

    w6 = optimise2!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    wt = [1.5708143372764898e-11, 0.22386943618920468, 3.969633399094772e-11,
          0.06250736036178277, 1.6133359173747837e-9, 1.0522575814542826e-11,
          3.8744621511970435e-12, 0.11119278945041582, 1.580927840121846e-11,
          1.3219024491425533e-10, 1.5237878704653776e-9, 0.0924690199294926,
          2.401058607850628e-12, 4.893126822172125e-11, 1.9999252446027412e-11,
          0.13670405758437285, 6.506459695614286e-11, 0.13899027797551924,
          3.0005175806374896e-11, 0.23426705498788586]
    @test isapprox(w6.weights, wt)

    obj = SR(; rf = rf)
    w7 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r3 = calc_risk(portfolio; type = :Trad2, rm = rm)
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

    w8 = optimise2!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [1.4660656272634796e-11, 1.2428783197184728e-10, 1.5746190204444947e-11,
          3.2489360402560526e-11, 0.32203999284596063, 3.6317215204099666e-12,
          5.244821655318297e-12, 8.62868500308722e-11, 2.1422593568347197e-10,
          4.552475520185754e-11, 2.370426923395792e-11, 1.3304916542187571e-11,
          4.8158800441891724e-12, 2.035115318343551e-11, 6.0157929919227965e-12,
          0.2066979969754921, 0.4712620092597057, 2.832844143707883e-11,
          2.430037202998556e-10, 3.721948981068307e-11]
    @test isapprox(w8.weights, wt)

    w9 = optimise2!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    wt = [1.391278800411977e-7, 8.623873248243568e-7, 1.4869928251260138e-7,
          2.96406065227947e-7, 0.3311202398853064, 4.295933053107905e-8,
          5.719534983666903e-8, 6.562728741790354e-7, 1.0250794423331498e-6,
          3.881969439919271e-7, 2.0917400536327976e-7, 1.252249175150365e-7,
          5.3226307614060094e-8, 1.833481197458791e-7, 6.37517188423901e-8,
          0.2078010153009358, 0.4610725195700459, 2.503427358110119e-7,
          1.4171807402340881e-6, 3.0667067330212224e-7]
    @test isapprox(w9.weights, wt)

    obj = MaxRet()
    w10 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r4 = calc_risk(portfolio; type = :Trad2, rm = rm)
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

    w11 = optimise2!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [6.036096739587215e-12, 1.790094245743028e-11, 7.472431823401233e-12,
          2.6015061923204717e-11, 0.8577737313404759, 5.106563274225794e-12,
          1.900666972805193e-12, 1.798805120124093e-11, 9.180990810209664e-11,
          1.8285780713416517e-11, 3.878646392138336e-12, 2.6114409753401695e-13,
          4.039407801796174e-12, 3.3645983430819403e-12, 2.0794146894700064e-12,
          1.8675565751693112e-10, 0.14222626819696105, 8.128255004448968e-12,
          5.689786193015622e-11, 4.642665800172863e-12]
    @test isapprox(w11.weights, wt)

    w12 = optimise2!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    wt = [8.957786502418264e-9, 1.5977656821401683e-8, 9.366331715285868e-9,
          2.102411418302442e-8, 0.8926311270031554, 2.942399215868535e-9,
          4.0003407568025345e-9, 1.5494573479566973e-8, 6.977521159918088e-8,
          1.5126598278011455e-8, 7.499156033895163e-9, 5.43708765295699e-9,
          3.108410245384283e-9, 7.163474439558288e-9, 3.841546341499991e-9,
          1.2472897993653294e-7, 0.10736849875485037, 9.805062757918749e-9,
          4.220649118074376e-8, 7.786773164597008e-9]
    @test isapprox(w12.weights, wt)

    # Risk upper bound
    obj = MaxRet()
    rm.settings.ub = r1
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r1 ||
          abs(calc_risk(portfolio; type = :Trad2, rm = rm) - r1) < 5e-10

    rm.settings.ub = r2
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r2 ||
          abs(calc_risk(portfolio; type = :Trad2, rm = rm) - r2) < 1e-10

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
          abs(calc_risk(portfolio; type = :Trad2, rm = rm) - r1) < 5e-9

    rm.settings.ub = r2
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r2 ||
          abs(calc_risk(portfolio; type = :Trad2, rm = rm) - r2) < 5e-10

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
    @test dot(portfolio.mu, w15.weights) >= ret3 ||
          abs(dot(portfolio.mu, w15.weights) - ret3) < 1e-10

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
          abs(dot(portfolio.mu, w20.weights) - ret4) < 1e-10

    portfolio.mu_l = Inf
    rm = TG2(; owa = OWASettings())

    obj = MinRisk()
    w21 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r5 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret5 = dot(portfolio.mu, w21.weights)
    wt = [1.424802287200348e-10, 0.24657565369133722, 3.660767298307013e-10,
          0.041101497481424845, 1.1765742472564645e-9, 1.3206711379272908e-10,
          4.948146901460501e-11, 0.11651694980250114, 1.27384858611194e-10,
          7.083586776439093e-10, 2.7784344366481497e-8, 0.09732942313209282,
          3.6499869966418024e-11, 4.3139455041429935e-10, 2.246994554915304e-10,
          0.14662262356233058, 3.7795902929060657e-10, 0.12694138139146063,
          2.229837567112617e-10, 0.22491243915854847]
    riskt = 0.023582366401225324
    rett = 0.000643224255466324
    @test isapprox(w21.weights, wt)
    @test isapprox(r5, riskt)
    @test isapprox(ret5, rett)
    @test isapprox(w21.weights, w1.weights, rtol = 0.25)
    @test isapprox(r5, r1, rtol = 0.001)
    @test isapprox(ret5, ret1, rtol = 0.1)

    w22 = optimise2!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [1.7145199698665806e-10, 0.2465793219892653, 4.301883839209905e-10,
          0.041100104500154715, 1.3674270711524961e-9, 1.5877584929573976e-10,
          5.988796424560396e-11, 0.11651809360509256, 1.523565552521047e-10,
          8.285519694899257e-10, 5.126970982587606e-8, 0.0973303915245183,
          4.399358422139485e-11, 5.431910026552069e-10, 2.75950471495248e-10,
          0.14662426603995923, 4.5944136999024105e-10, 0.12693544083332275,
          2.6804335690610397e-10, 0.22491232547871773]
    @test isapprox(w22.weights, wt)
    @test isapprox(w22.weights, w2.weights, rtol = 0.25)

    w23 = optimise2!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    wt = [3.957086335116054e-10, 0.24657634199068815, 9.468086929687249e-10,
          0.041099110306559036, 2.9394736684185725e-9, 3.6820828608700706e-10,
          1.5360187847252765e-10, 0.11651425698724804, 3.534506440127111e-10,
          1.8044411442000496e-9, 1.2960592292489882e-7, 0.09732806521814777,
          1.1896739793544472e-10, 1.236063078780606e-9, 6.259061296541714e-10,
          0.14662075751429432, 1.0282527239415776e-9, 0.12694826993421615,
          6.071981939490457e-10, 0.22491305786484317]
    @test isapprox(w23.weights, wt)
    @test isapprox(w23.weights, w3.weights, rtol = 0.25)

    obj = Util(; l = l)
    w24 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r6 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret6 = dot(portfolio.mu, w24.weights)
    wt = [2.0301833133540474e-10, 0.24852949875451355, 5.263484037968663e-10,
          0.04123289271029369, 1.523903625802092e-8, 1.5052654443309095e-10,
          6.624257626595854e-11, 0.11775475939426301, 2.0193317805448293e-10,
          1.2258124002876573e-9, 4.734012882640558e-9, 0.0980601025673535,
          4.855031408636756e-11, 5.375534373649509e-10, 2.6194404690686127e-10,
          0.1480800800607471, 6.884282753351088e-10, 0.12226540572157801,
          3.602123192531965e-10, 0.22407723654763223]
    riskt = 0.023585651701255057
    rett = 0.0006461160935175212
    @test isapprox(w24.weights, wt)
    @test isapprox(r6, riskt)
    @test isapprox(ret6, rett)
    @test isapprox(w24.weights, w4.weights, rtol = 0.1)
    @test isapprox(r6, r2, rtol = 0.001)
    @test isapprox(ret6, ret2, rtol = 0.05)

    w25 = optimise2!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [1.4161173737240054e-10, 0.24827698978459728, 3.5044000211710806e-10,
          0.041310748778789715, 7.100907394687908e-9, 1.0478351216193068e-10,
          4.6533513371635504e-11, 0.11762139887711939, 1.373722231604787e-10,
          8.20825347052686e-10, 6.683973349533642e-9, 0.09794540067397356,
          3.374497117705696e-11, 4.0625315698953255e-10, 1.885448260250963e-10,
          0.1478961037274142, 4.884602515672759e-10, 0.1227879148875347,
          2.4943170734828623e-10, 0.2241614265176891]
    @test isapprox(w25.weights, wt)
    @test isapprox(w25.weights, w5.weights, rtol = 0.1)

    w26 = optimise2!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    wt = [1.6845611584802501e-10, 0.2482833189147694, 4.3578770240337657e-10,
          0.04130616160474544, 1.1212885167185478e-8, 1.2791734209293923e-10,
          6.110256120529568e-11, 0.11762168667866996, 1.6895486364991856e-10,
          9.972022019812701e-10, 1.9762907324450302e-9, 0.09794664463602874,
          4.730827627325885e-11, 4.094865141480286e-10, 2.0982478299677356e-10,
          0.14789835042895502, 5.353742783198788e-10, 0.12278499583922883,
          2.912433385002021e-10, 0.22415882525576855]
    @test isapprox(w26.weights, wt)
    @test isapprox(w26.weights, w6.weights, rtol = 0.1)

    obj = SR(; rf = rf)
    w27 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r7 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret7 = dot(portfolio.mu, w27.weights)
    wt = [2.465085712389692e-10, 1.7591592054961841e-9, 2.6079839244395587e-10,
          5.190986820248767e-10, 0.3290475106649799, 7.765809158263677e-11,
          1.0087407386198554e-10, 1.0105472171462993e-9, 1.5219079686966533e-9,
          6.719673558205139e-10, 3.7631629567205446e-10, 2.3450024422381644e-10,
          9.619358604737496e-11, 3.325560088347231e-10, 1.1281942009082258e-10,
          0.20499762407557243, 0.46595485476474174, 4.4765819636375454e-10,
          2.177943962387867e-9, 5.481986913612393e-10]
    riskt = 0.03204366080645988
    rett = 0.00176571347903363
    @test isapprox(w27.weights, wt)
    @test isapprox(r7, riskt)
    @test isapprox(ret7, rett)
    @test isapprox(w27.weights, w7.weights, rtol = 0.1)
    @test isapprox(r7, r3, rtol = 0.005)
    @test isapprox(ret7, ret3, rtol = 0.005)

    w28 = optimise2!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [3.043911005150265e-10, 3.003199792680628e-9, 3.2260854035399194e-10,
          6.269047825687732e-10, 0.28925750034744363, 9.730401276044429e-11,
          1.227522626699953e-10, 1.8307825942400746e-9, 3.049544959096665e-9,
          9.699190113124327e-10, 5.345618956621474e-10, 2.99642107488554e-10,
          1.2150341416070546e-10, 4.4802804065845196e-10, 1.4014881021995218e-10,
          0.20684370520042217, 0.50389877563481, 6.21517343083002e-10, 5.475179599898194e-9,
          8.493359047675665e-10]
    @test isapprox(w28.weights, wt)
    @test isapprox(w28.weights, w8.weights, rtol = 0.1)

    w29 = optimise2!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    wt = [5.454025832765033e-8, 4.1758134700817797e-7, 5.7856741240063795e-8,
          1.1344449439291107e-7, 0.29960823571224127, 1.741045381363305e-8,
          2.225774590051608e-8, 2.7736698054689297e-7, 4.801971409642234e-7,
          1.6454217169284146e-7, 8.939515395915099e-8, 5.194131178647426e-8,
          2.1640167566461715e-8, 7.651996443785722e-8, 2.5104913248942918e-8,
          0.20599080438422707, 0.494398079909095, 1.0545667236281758e-7,
          7.711081586585951e-7, 1.3363076080133797e-7]
    @test isapprox(w29.weights, wt)
    @test isapprox(w29.weights, w9.weights, rtol = 0.1)

    obj = MaxRet()
    w30 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r8 = calc_risk(portfolio; type = :Trad2, rm = rm)
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

    w31 = optimise2!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [2.1877877724598778e-10, 5.423634965391811e-10, 2.5253431909718113e-10,
          7.867829458604338e-10, 0.8577730688248231, 8.617306754355166e-11,
          1.881760474509161e-11, 5.340082262051755e-10, 2.6334834681217134e-9,
          5.48174866618407e-10, 1.399929067726362e-10, 4.649356854273397e-11,
          7.430736924775098e-11, 1.2724883443711166e-10, 2.7593276139962158e-11,
          5.004679883728293e-9, 0.1422269180687958, 2.5760424370343565e-10,
          1.6464003712654701e-9, 1.6094394207994044e-10]
    @test isapprox(w31.weights, wt)
    @test isapprox(w31.weights, w11.weights, rtol = 5.0e-6)

    w32 = optimise2!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    wt = [9.582076829858595e-9, 1.7100009371334017e-8, 1.001142853853447e-8,
          2.2544068403868613e-8, 0.8926288017342953, 3.1237536310195203e-9,
          4.303736780951003e-9, 1.6595082389931922e-8, 7.312445684301714e-8,
          1.619345152951524e-8, 8.028153283280456e-9, 5.818797312570166e-9,
          3.3477297942616325e-9, 7.678312506638265e-9, 4.134196407903452e-9,
          1.2969240847583332e-7, 0.1073708034653235, 1.0508357801476938e-8,
          4.467105946746357e-8, 8.34330196221366e-9]
    @test isapprox(w32.weights, wt, rtol = 1.0e-5)
    @test isapprox(w32.weights, w12.weights, rtol = 5.0e-6)

    # Risk upper bound
    obj = MaxRet()
    rm.settings.ub = r5 * 1.001
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r5 * 1.001

    rm.settings.ub = r6 * 1.001
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r6 * 1.001

    rm.settings.ub = r7
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r7

    rm.settings.ub = r8
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r8

    obj = SR(; rf = rf)
    rm.settings.ub = r5 * 1.001
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r5 * 1.001

    rm.settings.ub = r6 * 1.001
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r6 * 1.001

    rm.settings.ub = r7
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r7

    rm.settings.ub = r8
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r8

    # Ret lower bound
    rm.settings.ub = Inf
    obj = MinRisk()
    portfolio.mu_l = ret5
    w33 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w33.weights) >= ret5

    portfolio.mu_l = ret6
    w34 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w34.weights) >= ret6

    portfolio.mu_l = ret7
    w35 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w35.weights) >= ret7

    portfolio.mu_l = ret8
    w36 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w16.weights) >= ret8

    obj = SR(; rf = rf)
    portfolio.mu_l = ret5
    w37 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w17.weights) >= ret1

    portfolio.mu_l = ret6
    w38 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w18.weights) >= ret6

    portfolio.mu_l = ret7
    w39 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w39.weights) >= ret7

    portfolio.mu_l = ret8
    w40 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w40.weights) >= ret8
end
