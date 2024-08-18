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
