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
end
