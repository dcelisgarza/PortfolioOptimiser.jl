@testset "RDaR vec" begin
    portfolio = Portfolio2(; prices = prices,
                           solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                            :params => Dict("verbose" => false,
                                                                            "max_step_fraction" => 0.75))))
    asset_statistics2!(portfolio)

    obj = MinRisk()
    wt0 = [0.05620454939911102, 3.025963157432751e-10, 1.55525199050249e-9,
           1.8034522386040858e-10, 0.0299591051407541, 4.1576811387096e-10,
           6.991555930029523e-10, 0.02066796438128093, 2.7039382408143445e-10,
           5.654715437689442e-10, 0.44148521326051954, 4.732606403168718e-10,
           0.021758018837756656, 1.393894367130224e-10, 1.2836530696174218e-10,
           0.14345074805893418, 0.15965735285676014, 0.06193188215818479,
           7.588611460321727e-10, 0.0648851604178394]
    riskt0 = 0.07576350913162658
    rett0 = 0.0005794990185578756

    rm = RDaR2(; settings = RiskMeasureSettings(; scale = 2.0))
    w1 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r1 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret1 = dot(portfolio.mu, w1.weights)
    wt = [0.05620451802549699, 4.007740730902365e-10, 7.066606385657234e-10,
          2.1196153188913592e-10, 0.029959050493386215, 6.036969701991338e-10,
          4.697814151904008e-10, 0.020667984925591232, 3.6401574968367325e-10,
          8.438952076627693e-10, 0.4414853183129637, 7.326021745407056e-10,
          0.021758037911551346, 1.6799908582927857e-10, 1.7729741297312878e-10,
          0.1434506295847512, 0.15965736559867008, 0.06193186998516469,
          9.679323699212284e-10, 0.06488521951580782]
    riskt = 0.07576351003435194
    rett = 0.0005794988152389773
    @test isapprox(w1.weights, wt0, rtol = 1.0e-6)
    @test isapprox(r1, riskt0, rtol = 1.0e-7)
    @test isapprox(ret1, rett0, rtol = 1.0e-6)
    @test isapprox(w1.weights, wt, rtol = 5.0e-7)
    @test isapprox(r1, riskt, rtol = 1.0e-7)
    @test isapprox(ret1, rett, rtol = 5.0e-7)

    rm = [[RDaR2(), RDaR2()]]
    w2 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r2 = calc_risk(portfolio; type = :Trad2, rm = rm[1][1])
    ret2 = dot(portfolio.mu, w2.weights)
    wt = [0.05620459491457741, 2.7975000273075594e-10, 1.0175488234371036e-9,
          8.670333880132651e-11, 0.029959019582913242, 2.7815780083323167e-10,
          1.6189382015384345e-10, 0.02066785568037724, 1.6936331910940167e-10,
          3.661140990834138e-10, 0.441485198826831, 3.046007499019945e-10,
          0.021758015205556966, 8.308419667764708e-11, 6.424295730191501e-11,
          0.1434509134177822, 0.1596575035553573, 0.061931060753152856,
          6.510983802614346e-10, 0.06488583460089434]
    riskt = 0.07576350806840577
    rett = 0.000579499327291706
    @test isapprox(w2.weights, wt0, rtol = 5.0e-6)
    @test isapprox(r2, riskt0)
    @test isapprox(ret2, rett0, rtol = 1.0e-6)
    @test isapprox(w2.weights, wt, rtol = 1.0e-5)
    @test isapprox(r2, riskt)
    @test isapprox(ret2, rett, rtol = 5.0e-6)

    obj = SR(; rf = rf)
    wt0 = [1.620183105236905e-8, 1.1140035827148705e-8, 0.04269821778198461,
           4.539811056982736e-9, 0.27528471583276515, 2.5884954318029336e-9,
           4.359695142864356e-9, 0.09149825388403467, 3.6147826445679577e-9,
           4.148787837320512e-9, 0.3004550564031882, 2.15879252345473e-9,
           1.3866951024663507e-9, 6.227749990512301e-9, 2.022396335125738e-9,
           0.2900636413773165, 1.8039723473841205e-8, 9.883423967231107e-9,
           8.029172726894051e-9, 2.0379317768180928e-8]
    riskt0 = 0.09342425156101017
    rett0 = 0.0009783083257672756

    rm = RDaR2(; settings = RiskMeasureSettings(; scale = 2.0))
    w3 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r3 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret3 = dot(portfolio.mu, w3.weights)
    wt = [4.576043121892937e-9, 3.072072587595365e-9, 0.042683400714392035,
          1.2618080628355493e-9, 0.27528467067527007, 7.21323223142053e-10,
          1.2532959128584345e-9, 0.09150588890011835, 1.002477468680859e-9,
          1.1528501466570201e-9, 0.3004564423167713, 6.009595662382212e-10,
          3.85071188705413e-10, 1.71870616716845e-9, 5.617511221281938e-10,
          0.2900695654667778, 5.017383938689553e-9, 2.742108560690652e-9,
          2.236891867802922e-9, 5.6239275397434465e-9]
    riskt = 0.09342369025329782
    rett = 0.000978303316192452
    @test isapprox(w3.weights, wt0, rtol = 5.0e-5)
    @test isapprox(r3, riskt0, rtol = 1.0e-5)
    @test isapprox(ret3, rett0, rtol = 1.0e-5)
    @test isapprox(w3.weights, wt)
    @test isapprox(r3, riskt)
    @test isapprox(ret3, rett)

    rm = [[RDaR2(), RDaR2()]]
    w4 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r4 = calc_risk(portfolio; type = :Trad2, rm = rm[1][1])
    ret4 = dot(portfolio.mu, w4.weights)
    wt = [5.6609919645419106e-9, 3.839867350537999e-9, 0.042680382839781775,
          1.5675061171185566e-9, 0.27528531309356663, 8.923105887122472e-10,
          1.4234767921687202e-9, 0.09150763575475925, 1.2457956685196475e-9,
          1.4312488154769366e-9, 0.3004563852539165, 7.424399985387603e-10,
          4.754309123583926e-10, 2.151287655244273e-9, 6.948169497458978e-10,
          0.2900702434978636, 6.220430434686813e-9, 3.4141868907099685e-9,
          2.766715041544124e-9, 7.033607026338685e-9]
    riskt = 0.09342363756654541
    rett = 0.0009783028412136859
    @test isapprox(w4.weights, wt0, rtol = 5.0e-5)
    @test isapprox(r4, riskt0, rtol = 1.0e-5)
    @test isapprox(ret4, rett0, rtol = 1.0e-5)
    @test isapprox(w4.weights, wt)
    @test isapprox(r4, riskt, rtol = 5.0e-8)
    @test isapprox(ret4, rett)

    # Risk upper bound
    obj = MaxRet()
    rm = RDaR2(; settings = RiskMeasureSettings(; scale = 2.0))
    rm.settings.ub = r1
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r1 ||
          abs(calc_risk(portfolio; type = :Trad2, rm = rm) - r1) < 5e-7

    rm = [[RDaR2(), RDaR2()]]
    rm[1][1].settings.ub = r2
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm[1][1]) <= r2 ||
          abs(calc_risk(portfolio; type = :Trad2, rm = rm[1][1]) - r2) < 5e-7

    obj = SR(; rf = rf)
    rm = RDaR2(; settings = RiskMeasureSettings(; scale = 2.0))
    rm.settings.ub = r1 * 1.000001
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r1 * 1.000001

    rm = [[RDaR2(), RDaR2()]]
    rm[1][1].settings.ub = r2
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm[1][1]) <= r2 ||
          abs(calc_risk(portfolio; type = :Trad2, rm = rm[1][1]) - r2) < 5e-7

    # Ret lower bound
    obj = MinRisk()
    rm = RDaR2(; settings = RiskMeasureSettings(; scale = 2.0))
    rm.settings.ub = Inf
    portfolio.mu_l = ret1
    w5 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w5.weights) >= ret1

    rm = [[RDaR2(), RDaR2()]]
    portfolio.mu_l = ret2
    w6 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w6.weights) >= ret2

    obj = SR(; rf = rf)
    rm = RDaR2(; settings = RiskMeasureSettings(; scale = 2.0))
    portfolio.mu_l = ret1
    w7 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w7.weights) >= ret1

    rm = [[RDaR2(), RDaR2()]]
    portfolio.mu_l = ret2
    w8 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w8.weights) >= ret2

    portfolio.mu_l = Inf
    obj = MinRisk()
    wt0 = [0.037682575145149035, 9.56278555310282e-9, 2.903645459870996e-9,
           1.1985145087686667e-9, 0.015069295206235518, 1.6728544107938776e-9,
           3.388725311447959e-10, 0.03749724397027287, 2.0700836960731254e-9,
           2.2024819594850123e-9, 0.4198578113688092, 2.938587276054813e-9,
           0.013791892244920912, 3.6397355643491843e-9, 1.9217193756673735e-9,
           0.13381043923267996, 0.20615475605389819, 0.037331096485967066,
           4.8493986688419375e-9, 0.09880485699338833]
    riskt0 = 0.06867235340781119
    rett0 = 0.0005981290826955536
    wt1 = [1.0824864155062309e-8, 2.4957286615186985e-8, 0.2115530098577518,
           8.141340696603367e-9, 0.3057708568267008, 3.0450387238744273e-9,
           7.075013781037969e-9, 0.13351415577278983, 6.5941684811104185e-9,
           6.571189642973626e-9, 0.11759965423964155, 3.080567800710732e-9,
           1.8450882352298185e-9, 1.0313678582796207e-8, 2.80786527899511e-9,
           0.20775317077172692, 0.023808954822869766, 1.4675901738462066e-8,
           3.4286429938206516e-8, 6.349008574376557e-8]
    riskt1 = 0.045712870660457844
    rett1 = 0.0010916439887987248
    rm = [[RDaR2(), RDaR2(; alpha = 0.75)]]
    w9 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r9 = calc_risk(portfolio; type = :Trad2, rm = rm[1][1])
    ret9 = dot(portfolio.mu, w9.weights)
    wt = [0.05205188084014378, 2.3435119303377945e-9, 3.449182487041259e-9,
          1.5778403025918072e-10, 0.028033949353693376, 3.1687827107106907e-10,
          5.726731728935765e-11, 0.03245625705141398, 3.3895314711219634e-10,
          5.034893952100365e-10, 0.41549163931923916, 3.812499371863707e-10,
          0.018672776482202273, 5.986347671977716e-10, 4.25234559338416e-10,
          0.14239995164320318, 0.20032802686772627, 0.0626517022655744,
          1.6881299660136958e-9, 0.04791380591648782]
    riskt = 0.07612556586542824
    rett = 0.0006031053529244473
    @test !isapprox(w9.weights, wt0)
    @test !isapprox(r9, riskt0)
    @test !isapprox(ret9, rett0)
    @test !isapprox(w9.weights, wt1)
    @test !isapprox(r9, riskt1)
    @test !isapprox(ret9, rett1)
    @test isapprox(w9.weights, wt, rtol = 1.0e-6)
    @test isapprox(r9, riskt, rtol = 1.0e-7)
    @test isapprox(ret9, rett, rtol = 5.0e-7)
end

@testset "Kurt vec" begin
    portfolio = Portfolio2(; prices = prices,
                           solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                            :params => Dict("verbose" => false,
                                                                            "max_step_fraction" => 0.75))))
    asset_statistics2!(portfolio)

    obj = MinRisk()
    wt0 = [1.2206987375321365e-8, 0.039710885748646646, 1.5461868222416377e-8,
           0.0751622837839456, 1.5932470737314812e-8, 0.011224033304772868,
           8.430423528936607e-9, 0.12881590610168409, 8.394156714410349e-9,
           3.2913090335193894e-8, 0.3694591285102107, 8.438604213700625e-9,
           4.8695034432287695e-9, 0.039009375331486955, 2.101727257447083e-8,
           0.02018194607274747, 4.58542212019818e-8, 0.18478029998119072,
           1.0811006933572572e-8, 0.1316559568357098]
    riskt0 = 0.00013888508487777416
    rett0 = 0.0004070691284861725

    rm = Kurt2(; settings = RiskMeasureSettings(; scale = 2.0))
    w1 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r1 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret1 = dot(portfolio.mu, w1.weights)
    wt = [1.9365481462888576e-8, 0.03971054609946532, 2.452322304695446e-8,
          0.07516202017809688, 2.5257467674280395e-8, 0.011225005844683996,
          1.348206693674745e-8, 0.12881599306755828, 1.3379093396514524e-8,
          5.1975368649868716e-8, 0.3694593056093741, 1.3472200632609833e-8,
          7.800476776261508e-9, 0.039009117679769986, 3.3395763437422323e-8,
          0.020181767062364027, 7.23884265551776e-8, 0.18478009568307874,
          1.7181993269128916e-8, 0.13165585655404677]
    riskt = 0.00013888508788699341
    rett = 0.0004070678718055879
    @test isapprox(w1.weights, wt0, rtol = 5.0e-6)
    @test isapprox(r1, riskt0, rtol = 5.0e-8)
    @test isapprox(ret1, rett0, rtol = 5.0e-6)
    @test isapprox(w1.weights, wt)
    @test isapprox(r1, riskt)
    @test isapprox(ret1, rett)

    rm = [[Kurt2(), Kurt2(; kt = portfolio.kurt)]]
    w2 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r2 = calc_risk(portfolio; type = :Trad2, rm = rm[1][1])
    ret2 = dot(portfolio.mu, w2.weights)
    wt = [1.0969876707703097e-8, 0.039710884441438374, 1.4108531089332219e-8,
          0.07516282814469828, 1.4564842087742093e-8, 0.01122143842431493,
          7.730955021557689e-9, 0.12881579211780614, 7.504081006319637e-9,
          3.0700104388706474e-8, 0.36945933610612347, 7.556986708322137e-9,
          4.3379982052485495e-9, 0.03900910314432929, 1.9590959044426742e-8,
          0.020181769263493434, 4.2995478392453554e-8, 0.1847817086767035,
          9.702887803139182e-9, 0.13165696991839212]
    riskt = 0.00013888508442463864
    rett = 0.0004070721138966786
    @test isapprox(w2.weights, wt0, rtol = 1.0e-5)
    @test isapprox(r2, riskt0, rtol = 5.0e-9)
    @test isapprox(ret2, rett0, rtol = 1.0e-5)
    @test isapprox(w2.weights, wt)
    @test isapprox(r2, riskt)
    @test isapprox(ret2, rett)

    obj = SR(; rf = rf)
    wt0 = [3.6858654756755185e-8, 0.007745692921941757, 7.659402306115789e-7,
           0.06255101954920105, 0.21591225958933066, 1.164996778478312e-8,
           0.04812416132599465, 0.1265864284132457, 7.028702221705754e-8,
           6.689307437956413e-8, 0.19709510156536242, 1.0048124974067658e-8,
           5.6308070308085104e-9, 2.6819277429081264e-8, 6.042626796051015e-9,
           0.09193445125562229, 0.20844322199597035, 0.03380290484590025,
           0.00779744381804698, 6.3145495978703785e-6]
    riskt0 = 0.00020556674631177048
    rett0 = 0.0009657710513568699

    rm = Kurt2(; settings = RiskMeasureSettings(; scale = 2.0))
    w3 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r3 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret3 = dot(portfolio.mu, w3.weights)
    wt = [1.5027584021534503e-8, 0.007744363746657039, 3.117152029012665e-7,
          0.06255191220895817, 0.21591436828861665, 4.744420020761665e-9,
          0.04812430920999633, 0.1265859720979338, 2.853911923671977e-8,
          2.7127451637983628e-8, 0.19709349563207804, 4.108608833649343e-9,
          2.298334582961534e-9, 1.0898630330334143e-8, 2.4689254804443724e-9,
          0.09193518784707086, 0.20844467805906633, 0.03380616015796153,
          0.007796600457494015, 2.5453658904060404e-6]
    riskt = 0.00020556750435209815
    rett = 0.000965774220431758
    @test isapprox(w3.weights, wt0, rtol = 5.0e-5)
    @test isapprox(r3, riskt0, rtol = 5.0e-6)
    @test isapprox(ret3, rett0, rtol = 5.0e-6)
    @test isapprox(w3.weights, wt)
    @test isapprox(r3, riskt)
    @test isapprox(ret3, rett)

    rm = [[Kurt2(; kt = portfolio.kurt), Kurt2()]]
    w4 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r4 = calc_risk(portfolio; type = :Trad2, rm = rm[1][1])
    ret4 = dot(portfolio.mu, w4.weights)
    wt = [2.8272958797553797e-8, 0.007741632284935734, 6.138503193588016e-7,
          0.06255245774514491, 0.2159137611854654, 9.028739192135674e-9,
          0.04812430647421746, 0.12658626033628126, 5.4298681099351515e-8,
          5.159246243919232e-8, 0.19709655710981566, 7.761476325770073e-9,
          4.319595101590778e-9, 2.065447514926452e-8, 4.652141138077654e-9,
          0.09193526020743648, 0.20844596187182207, 0.033804156151533, 0.007793787514144101,
          5.0646883552937275e-6]
    riskt = 0.00020556710349365
    rett = 0.000965772536611109
    @test isapprox(w4.weights, wt0, rtol = 5.0e-5)
    @test isapprox(r4, riskt0, rtol = 5.0e-6)
    @test isapprox(ret4, rett0, rtol = 5.0e-6)
    @test isapprox(w4.weights, wt, rtol = 5.0e-8)
    @test isapprox(r4, riskt)
    @test isapprox(ret4, rett)

    # Risk upper bound
    obj = MaxRet()
    rm = Kurt2(; settings = RiskMeasureSettings(; scale = 2.0))
    rm.settings.ub = r1
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r1 ||
          abs(calc_risk(portfolio; type = :Trad2, rm = rm) - r1) < 5e-9

    rm = [[Kurt2(), Kurt2(; kt = portfolio.kurt)]]
    rm[1][1].settings.ub = r2
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm[1][1]) <= r2 ||
          abs(calc_risk(portfolio; type = :Trad2, rm = rm[1][1]) - r2) < 5e-9

    obj = SR(; rf = rf)
    rm = Kurt2(; settings = RiskMeasureSettings(; scale = 2.0))
    rm.settings.ub = r1
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r1 ||
          abs(calc_risk(portfolio; type = :Trad2, rm = rm) - r1) < 1e-10

    rm = [[Kurt2(; kt = portfolio.kurt), Kurt2()]]
    rm[1][1].settings.ub = r2
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm[1][1]) <= r2 ||
          abs(calc_risk(portfolio; type = :Trad2, rm = rm[1][1]) - r2) < 1e-10

    # Ret lower bound
    obj = MinRisk()
    rm = Kurt2(; settings = RiskMeasureSettings(; scale = 2.0))
    rm.settings.ub = Inf
    portfolio.mu_l = ret1
    w5 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w5.weights) >= ret1

    rm = [[Kurt2(), Kurt2(; kt = portfolio.kurt)]]
    portfolio.mu_l = ret2
    w6 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w6.weights) >= ret2

    obj = SR(; rf = rf)
    rm = Kurt2(; settings = RiskMeasureSettings(; scale = 2.0))
    portfolio.mu_l = ret1
    w7 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w7.weights) >= ret1

    rm = [[Kurt2(), Kurt2()]]
    portfolio.mu_l = ret2
    w8 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w8.weights) >= ret2
end

@testset "Kurt Reduced vec" begin
    portfolio = Portfolio2(; prices = prices, max_num_assets_kurt = 1,
                           solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                            :params => Dict("verbose" => false,
                                                                            "max_step_fraction" => 0.75))))
    asset_statistics2!(portfolio)

    obj = MinRisk()
    wt0 = [6.258921377348572e-7, 3.680102939415165e-6, 7.486004924604795e-7,
           0.03498034415709424, 7.917668013849855e-7, 1.2350801714348813e-6,
           3.0460581400971164e-7, 0.08046935867869996, 5.699668814229727e-7,
           8.85393394989514e-7, 0.6357270365898279, 5.979410095673201e-7,
           2.7962721329567345e-7, 1.8757932499250933e-6, 5.070585649379864e-7,
           0.00127706653757055, 1.2457133780976972e-6, 0.11108722055351541,
           6.616561323348826e-7, 0.13644496428511088]
    riskt0 = 0.0001588490319818568
    rett0 = 0.0003563680681010386

    rm = Kurt2(; settings = RiskMeasureSettings(; scale = 2.0))
    w1 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r1 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret1 = dot(portfolio.mu, w1.weights)
    wt = [9.078227512107962e-7, 5.462117858218533e-6, 1.0945241720142207e-6,
          0.03498458878597766, 1.1612565960494257e-6, 1.8123811712208276e-6,
          4.445447317098628e-7, 0.08047324759831426, 8.314810843838215e-7,
          1.2904761285370553e-6, 0.635723153162971, 8.816994588882366e-7,
          4.063501043239381e-7, 2.771627693198632e-6, 7.418287368656995e-7,
          0.0013101640352412046, 1.8277486723920982e-6, 0.111070076160227,
          9.628069199669719e-7, 0.1364181735911897]
    riskt = 0.00015884800413573525
    rett = 0.0003563894347703385
    @test isapprox(w1.weights, wt0, rtol = 0.0001)
    @test isapprox(r1, riskt0, rtol = 1.0e-5)
    @test isapprox(ret1, rett0, rtol = 0.0001)
    @test isapprox(w1.weights, wt)
    @test isapprox(r1, riskt)
    @test isapprox(ret1, rett)

    rm = [[Kurt2(), Kurt2(; kt = portfolio.kurt)]]
    w2 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r2 = calc_risk(portfolio; type = :Trad2, rm = rm[1][1])
    ret2 = dot(portfolio.mu, w2.weights)
    wt = [7.264246709961181e-7, 4.508344724375973e-6, 8.817903761500179e-7,
          0.03498229679687887, 9.357327303319221e-7, 1.4628151689184126e-6,
          3.5641605656117595e-7, 0.08047177176332428, 6.707850608084554e-7,
          1.0386845135548448e-6, 0.6357295944050206, 7.136038891337252e-7,
          3.257129307780298e-7, 2.255215917977421e-6, 5.957596233161254e-7,
          0.0012954615162864534, 1.4828126235100654e-6, 0.11107629254006336,
          7.757927049082309e-7, 0.13642785308743507]
    riskt = 0.00015884916839850122
    rett = 0.0003563784967964951
    @test isapprox(w2.weights, wt0, rtol = 5.0e-5)
    @test isapprox(r2, riskt0, rtol = 1.0e-6)
    @test isapprox(ret2, rett0, rtol = 5.0e-5)
    @test isapprox(w2.weights, wt)
    @test isapprox(r2, riskt)
    @test isapprox(ret2, rett)

    obj = SR(; rf = rf)
    wt0 = [2.1869958623435718e-8, 1.2027489763838995e-7, 8.307710723713282e-8,
           0.04981239216610764, 0.19930305948126495, 9.890114269259893e-9,
           0.041826524860577356, 0.10466467192338667, 3.332719673064681e-8,
           4.0257776120217934e-8, 0.23960266714976633, 8.639963684421896e-9,
           4.958509460440708e-9, 2.0027087137090508e-8, 5.256105741068125e-9,
           0.08392271758019651, 0.28086715326117795, 2.469501355591167e-7,
           1.0503063749371977e-7, 1.1401803293988891e-7]
    riskt0 = 0.00020482110250140048
    rett0 = 0.0009552956056983061

    rm = Kurt2(; settings = RiskMeasureSettings(; scale = 2.0))
    w3 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r3 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret3 = dot(portfolio.mu, w3.weights)
    wt = [9.447488454850529e-9, 5.157345717076827e-8, 3.572354058709873e-8,
          0.04981231717568401, 0.19930319986110112, 4.406095669814159e-9,
          0.04182661887540398, 0.10466363493869778, 1.4369370030038918e-8,
          1.7422213062598217e-8, 0.23960404002939426, 3.837832221844872e-9,
          2.223339920873821e-9, 8.790458001535558e-9, 2.3567516802668404e-9,
          0.08392271006689143, 0.28086712953018445, 1.0544124398984358e-7,
          4.498269030396397e-8, 4.894816173318005e-8]
    riskt = 0.00020482113513366557
    rett = 0.0009552957011081039
    @test isapprox(w3.weights, wt0, rtol = 5.0e-6)
    @test isapprox(r3, riskt0, rtol = 5.0e-7)
    @test isapprox(ret3, rett0, rtol = 5.0e-7)
    @test isapprox(w3.weights, wt)
    @test isapprox(r3, riskt)
    @test isapprox(ret3, rett, rtol = 5.0e-7)

    rm = [[Kurt2(; kt = portfolio.kurt), Kurt2()]]
    w4 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r4 = calc_risk(portfolio; type = :Trad2, rm = rm[1][1])
    ret4 = dot(portfolio.mu, w4.weights)
    wt = [1.889615058125512e-8, 1.0282087989092793e-7, 7.181371604310739e-8,
          0.04981218606917156, 0.19930293309182331, 9.010676842451078e-9,
          0.04182643137274975, 0.10466444731259547, 2.8808624395019052e-8,
          3.505874113199112e-8, 0.23960282650661302, 7.896228141715727e-9,
          4.441526411289592e-9, 1.7880274301701496e-8, 4.723909524610223e-9,
          0.0839226895349512, 0.28086779190349964, 2.06187642888939e-7,
          8.899283446363545e-8, 9.767739147432175e-8]
    riskt = 0.00020482111480875584
    rett = 0.0009552955342588195
    @test isapprox(w4.weights, wt0, rtol = 5.0e-6)
    @test isapprox(r4, riskt0, rtol = 1.0e-7)
    @test isapprox(ret4, rett0, rtol = 1.0e-7)
    @test isapprox(w4.weights, wt, rtol = 5.0e-8)
    @test isapprox(r4, riskt)
    @test isapprox(ret4, rett)

    # Risk upper bound
    obj = MaxRet()
    rm = Kurt2(; settings = RiskMeasureSettings(; scale = 1.0))
    rm.settings.ub = r1
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r1 ||
          abs(calc_risk(portfolio; type = :Trad2, rm = rm) - r1) < 5e-5

    rm = [[Kurt2(), Kurt2(; kt = portfolio.kurt)]]
    rm[1][1].settings.ub = r2
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm[1][1]) <= r2 ||
          abs(calc_risk(portfolio; type = :Trad2, rm = rm[1][1]) - r2) < 5e-5

    obj = SR(; rf = rf)
    rm = Kurt2(; settings = RiskMeasureSettings(; scale = 2.0))
    rm.settings.ub = r1
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r1 ||
          abs(calc_risk(portfolio; type = :Trad2, rm = rm) - r1) < 5e-5

    rm = [[Kurt2(; kt = portfolio.kurt), Kurt2()]]
    rm[1][1].settings.ub = r2
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm[1][1]) <= r2 ||
          abs(calc_risk(portfolio; type = :Trad2, rm = rm[1][1]) - r2) < 5e-5

    # Ret lower bound
    obj = MinRisk()
    rm = Kurt2(; settings = RiskMeasureSettings(; scale = 2.0))
    rm.settings.ub = Inf
    portfolio.mu_l = ret1
    w5 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w5.weights) >= ret1

    rm = [[Kurt2(), Kurt2(; kt = portfolio.kurt)]]
    portfolio.mu_l = ret2
    w6 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w6.weights) >= ret2

    obj = SR(; rf = rf)
    rm = Kurt2(; settings = RiskMeasureSettings(; scale = 2.0))
    portfolio.mu_l = ret1
    w7 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w7.weights) >= ret1

    rm = [[Kurt2(), Kurt2()]]
    portfolio.mu_l = ret2
    w8 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w8.weights) >= ret2
end

@testset "SKurt vec" begin
    portfolio = Portfolio2(; prices = prices,
                           solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                            :params => Dict("verbose" => false,
                                                                            "max_step_fraction" => 0.75))))
    asset_statistics2!(portfolio)

    obj = MinRisk()
    wt0 = [1.84452954864755e-8, 0.09964222365209786, 1.3574917720415848e-8,
           3.86114747805695e-8, 1.3966729610624488e-8, 3.110357119905182e-7,
           6.064715171940905e-9, 0.13536486151149585, 1.1522227247871875e-8,
           1.636443293768575e-8, 0.39008850330648476, 1.8792073784410723e-8,
           8.525841353694015e-9, 0.021330458258521926, 3.206796675755174e-8,
           5.682782161200388e-8, 1.3934759941686095e-7, 0.19923258780375303,
           1.7400011528901028e-8, 0.15434066292082727]
    riskt0 = 0.0001095671204061385
    rett0 = 0.0003976490301536407

    rm = SKurt2(; settings = RiskMeasureSettings(; scale = 2.0))
    w1 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r1 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret1 = dot(portfolio.mu, w1.weights)
    wt = [9.77511650461382e-9, 0.09964296868660863, 7.189912318509742e-9,
          2.057806665088293e-8, 7.397069984908799e-9, 1.6694954473336397e-7,
          3.2211014665824285e-9, 0.13536475558016456, 6.107934289282906e-9,
          8.68450180905975e-9, 0.39008811581758646, 9.995996169334164e-9,
          4.523120144772703e-9, 0.02133035809675054, 1.708691386523893e-8,
          3.030815366117899e-8, 7.450641576648867e-8, 0.1992330859779131,
          9.226346198844044e-9, 0.15434034029078322]
    riskt = 0.00010956712036235774
    rett = 0.000397649412448154
    @test isapprox(w1.weights, wt0, rtol = 5.0e-6)
    @test isapprox(r1, riskt0, rtol = 5.0e-10)
    @test isapprox(ret1, rett0, rtol = 1.0e-6)
    @test isapprox(w1.weights, wt)
    @test isapprox(r1, riskt)
    @test isapprox(ret1, rett)

    rm = [[SKurt2(), SKurt2(; kt = portfolio.skurt)]]
    w2 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r2 = calc_risk(portfolio; type = :Trad2, rm = rm[1][1])
    ret2 = dot(portfolio.mu, w2.weights)
    wt = [4.848370567114622e-9, 0.09964418060675992, 3.534421901349859e-9,
          1.0714719154481452e-8, 3.6417689889476594e-9, 9.13421901209692e-8,
          1.568600983497844e-9, 0.13536472338980743, 3.0095904595235706e-9,
          4.308615350569908e-9, 0.3900875607774929, 5.02351639137919e-9,
          2.2068996721699858e-9, 0.021328617510922788, 8.883526651372444e-9,
          1.591917099510743e-8, 4.027650274878825e-8, 0.19923447468779643,
          4.598966358451933e-9, 0.15434024315036032]
    riskt = 0.00010956712275679024
    rett = 0.0003976507154245343
    @test isapprox(w2.weights, wt0, rtol = 1.0e-5)
    @test isapprox(r2, riskt0, rtol = 5.0e-8)
    @test isapprox(ret2, rett0, rtol = 5.0e-6)
    @test isapprox(w2.weights, wt)
    @test isapprox(r2, riskt)
    @test isapprox(ret2, rett)

    obj = SR(; rf = rf)
    wt0 = [8.263501742935076e-8, 3.4292121981081834e-5, 8.553642485927413e-8,
           1.0517936993748617e-7, 0.32096393208295665, 1.9848129848992033e-8,
           0.004686373576551095, 0.05273663584046675, 6.751336821499605e-8,
           5.166788731416254e-8, 0.24476417791090582, 1.7957314779078257e-8,
           9.390528572513579e-9, 5.4617904908192196e-8, 1.060832570103225e-8,
           0.135854021084771, 0.24095692751428416, 1.9017748026664851e-6,
           2.5600086153798754e-7, 9.771381477598014e-7]
    riskt0 = 0.0001695775787535489
    rett0 = 0.0010654948366056365

    rm = SKurt2(; settings = RiskMeasureSettings(; scale = 2.0))
    w3 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r3 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret3 = dot(portfolio.mu, w3.weights)
    wt = [4.069681359032733e-8, 1.6380870112351286e-5, 4.231110258351172e-8,
          5.193650991297926e-8, 0.32096785698385566, 9.84261938828389e-9,
          0.004686172725258817, 0.0527377161428981, 3.33194480396312e-8,
          2.5495359403493753e-8, 0.24476542707397542, 8.891400081295185e-9,
          4.6595955535350376e-9, 2.6897540978006548e-8, 5.264297995099994e-9,
          0.13585674367245396, 0.24096795098345958, 9.044278818277224e-7,
          1.2492510861551688e-7, 4.72880308192895e-7]
    riskt = 0.00016957876787065916
    rett = 0.001065501597545198
    @test isapprox(w3.weights, wt0, rtol = 5.0e-5)
    @test isapprox(r3, riskt0, rtol = 1.0e-5)
    @test isapprox(ret3, rett0, rtol = 1.0e-5)
    @test isapprox(w3.weights, wt)
    @test isapprox(r3, riskt)
    @test isapprox(ret3, rett)

    rm = [[SKurt2(; kt = portfolio.skurt), SKurt2()]]
    w4 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r4 = calc_risk(portfolio; type = :Trad2, rm = rm[1][1])
    ret4 = dot(portfolio.mu, w4.weights)
    wt = [2.792086458477168e-8, 1.23270227624102e-5, 2.8693839205641713e-8,
          3.545993562585731e-8, 0.32096938025347765, 6.724625281454166e-9,
          0.00468443526306139, 0.052736851509426384, 2.268649071119944e-8,
          1.732887618775518e-8, 0.2447670673554815, 6.044143966674832e-9,
          3.136982365560315e-9, 1.8518351716171052e-8, 3.5579477845468585e-9,
          0.13585763652163776, 0.2409710323367473, 6.623091797731533e-7,
          8.819187783048938e-8, 3.4916429056298737e-7]
    riskt = 0.0001695789159594149
    rett = 0.001065502038777707
    @test isapprox(w4.weights, wt0, rtol = 0.0001)
    @test isapprox(r4, riskt0, rtol = 1.0e-5)
    @test isapprox(ret4, rett0, rtol = 1.0e-5)
    @test isapprox(w4.weights, wt, rtol = 5.0e-8)
    @test isapprox(r4, riskt)
    @test isapprox(ret4, rett)

    # Risk upper bound
    obj = MaxRet()
    rm = SKurt2(; settings = RiskMeasureSettings(; scale = 2.0))
    rm.settings.ub = r1 * 1.05
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r1 * 1.05

    rm = [[SKurt2(), SKurt2(; kt = portfolio.skurt)]]
    rm[1][1].settings.ub = r2 * 1.05
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm[1][1]) <= r2 * 1.05

    obj = SR(; rf = rf)
    rm = SKurt2(; settings = RiskMeasureSettings(; scale = 2.0))
    rm.settings.ub = r1 * 1.05
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r1 * 1.05

    rm = [[SKurt2(; kt = portfolio.skurt), SKurt2()]]
    rm[1][1].settings.ub = r2 * 1.05
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm[1][1]) <= r2 * 1.05

    # Ret lower bound
    obj = MinRisk()
    rm = SKurt2(; settings = RiskMeasureSettings(; scale = 2.0))
    rm.settings.ub = Inf
    portfolio.mu_l = ret1
    w5 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w5.weights) >= ret1

    rm = [[SKurt2(), SKurt2(; kt = portfolio.skurt)]]
    portfolio.mu_l = ret2
    w6 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w6.weights) >= ret2

    obj = SR(; rf = rf)
    rm = SKurt2(; settings = RiskMeasureSettings(; scale = 2.0))
    portfolio.mu_l = ret1
    w7 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w7.weights) >= ret1

    rm = [[SKurt2(), SKurt2()]]
    portfolio.mu_l = ret2
    w8 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w8.weights) >= ret2
end

@testset "Kurt Reduced vec" begin
    portfolio = Portfolio2(; prices = prices, max_num_assets_kurt = 1,
                           solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                            :params => Dict("verbose" => false,
                                                                            "max_step_fraction" => 0.75))))
    asset_statistics2!(portfolio)

    obj = MinRisk()
    wt0 = [9.537055656561948e-7, 0.08065159914970102, 6.97429322507607e-7,
           1.636243884801297e-6, 6.970775807421223e-7, 3.475223837047256e-6,
           2.8901276220144585e-7, 0.1269025346766842, 6.27357228262538e-7,
           8.136543409694392e-7, 0.4622659726775211, 1.0093804242816126e-6,
           4.237381686192652e-7, 2.0106199669190677e-5, 1.10186359354052e-6,
           2.157409439859392e-6, 3.9496933962571675e-6, 0.1701191581088843,
           8.973744700260093e-7, 0.16002190002352545]
    riskt0 = 0.00011018128303928912
    rett0 = 0.00039273793904369474

    rm = SKurt2(; settings = RiskMeasureSettings(; scale = 2.0))
    w1 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r1 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret1 = dot(portfolio.mu, w1.weights)
    wt = [1.3948063260252035e-6, 0.08064485664148535, 1.0158432752067733e-6,
          2.4048200905373573e-6, 1.0164757142897614e-6, 5.158293253802848e-6,
          4.18871458719602e-7, 0.1269018022353016, 9.130060654317273e-7,
          1.1887888375594998e-6, 0.4622649000635114, 1.481269227818027e-6,
          6.159580734436594e-7, 3.0149866076906225e-5, 1.6173319512074959e-6,
          3.1762526671495673e-6, 5.82108981364211e-6, 0.17011000773243246,
          1.3109850587851905e-6, 0.16002074966937865]
    riskt = 0.00011018136745154176
    rett = 0.0003927313904011752
    @test isapprox(w1.weights, wt0, rtol = 5.0e-5)
    @test isapprox(r1, riskt0, rtol = 1.0e-6)
    @test isapprox(ret1, rett0, rtol = 5.0e-5)
    @test isapprox(w1.weights, wt)
    @test isapprox(r1, riskt)
    @test isapprox(ret1, rett)

    rm = [[SKurt2(), SKurt2(; kt = portfolio.skurt)]]
    w2 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r2 = calc_risk(portfolio; type = :Trad2, rm = rm[1][1])
    ret2 = dot(portfolio.mu, w2.weights)
    wt = [1.0227877108077565e-6, 0.08065089015954513, 7.412454395587232e-7,
          1.7926511341998345e-6, 7.421665823389713e-7, 3.895216715124378e-6,
          3.0372995890587404e-7, 0.12690257696880766, 6.671932962454392e-7,
          8.704876640970151e-7, 0.46226464259981564, 1.095617957541829e-6,
          4.48210203315133e-7, 2.3231717150112222e-5, 1.1992336087187594e-6,
          2.37256803277216e-6, 4.38077559768376e-6, 0.17011527424167755,
          9.62332736457258e-7, 0.160022890096366]
    riskt = 0.00011018125526225448
    rett = 0.00039273677236197595
    @test isapprox(w2.weights, wt0, rtol = 1.0e-5)
    @test isapprox(r2, riskt0, rtol = 5.0e-7)
    @test isapprox(ret2, rett0, rtol = 5.0e-6)
    @test isapprox(w2.weights, wt)
    @test isapprox(r2, riskt)
    @test isapprox(ret2, rett)

    obj = SR(; rf = rf)
    wt0 = [9.329112161227789e-8, 1.8997236041327231e-6, 9.183667269717346e-8,
           1.1278547529605994e-7, 0.323307535710129, 2.5473368939712553e-8,
           6.050657719406614e-6, 0.04416525712754569, 7.422093025293419e-8,
           6.013968073613049e-8, 0.2607281038602079, 2.2840291563131592e-8,
           1.2053370397405268e-8, 6.734594412605242e-8, 1.3612437340851914e-8,
           0.13505441433828766, 0.23673428116977457, 1.0078714608278738e-6,
           2.4778495858369644e-7, 6.28157019375726e-7]
    riskt0 = 0.00016836304121264654
    rett0 = 0.0010573999779863363

    rm = SKurt2(; settings = RiskMeasureSettings(; scale = 2.0))
    w3 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r3 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret3 = dot(portfolio.mu, w3.weights)
    wt = [4.230729448715929e-8, 8.604210005296255e-7, 4.1583103977396935e-8,
          5.115018261414783e-8, 0.323309384326836, 1.1873568750735587e-8,
          2.7823172796899216e-6, 0.04416280873383693, 3.36501986500584e-8,
          2.7395698894902835e-8, 0.2607296328157974, 1.0619855015995494e-8,
          5.64125598956425e-9, 3.100449049844335e-8, 6.374003984232665e-9,
          0.13505509615224126, 0.23673832203822642, 4.546558831978202e-7,
          1.1212147423080774e-7, 2.8481777148442364e-7]
    riskt = 0.00016836324309220775
    rett = 0.0010574002540951438
    @test isapprox(w3.weights, wt0, rtol = 5.0e-5)
    @test isapprox(r3, riskt0, rtol = 5.0e-6)
    @test isapprox(ret3, rett0, rtol = 5.0e-7)
    @test isapprox(w3.weights, wt, rtol = 5.0e-8)
    @test isapprox(r3, riskt)
    @test isapprox(ret3, rett)

    rm = [[SKurt2(; kt = portfolio.skurt), SKurt2()]]
    w4 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r4 = calc_risk(portfolio; type = :Trad2, rm = rm[1][1])
    ret4 = dot(portfolio.mu, w4.weights)
    wt = [6.889293196205733e-8, 1.4187204176623265e-6, 6.787465002725962e-8,
          8.428703867270279e-8, 0.3233074396290324, 1.9600581450462458e-8,
          4.941656152738128e-6, 0.04416439744292931, 5.5090564307047604e-8,
          4.4900297456426566e-8, 0.26072958577715877, 1.7585177593330183e-8,
          9.107886906610544e-9, 5.1174855030000144e-8, 1.0328781974379036e-8,
          0.13505473870980508, 0.23673566348915104, 7.321441827025678e-7,
          1.8195614298011009e-7, 4.7163226174391176e-7]
    riskt = 0.00016836291133139275
    rett = 0.00105739893823015
    @test isapprox(w4.weights, wt0, rtol = 1.0e-5)
    @test isapprox(r4, riskt0, rtol = 1.0e-6)
    @test isapprox(ret4, rett0, rtol = 1.0e-6)
    @test isapprox(w4.weights, wt, rtol = 5.0e-8)
    @test isapprox(r4, riskt)
    @test isapprox(ret4, rett)

    # Risk upper bound
    obj = MaxRet()
    rm = SKurt2(; settings = RiskMeasureSettings(; scale = 1.0))
    rm.settings.ub = r1
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r1

    rm = [[SKurt2(), SKurt2(; kt = portfolio.skurt)]]
    rm[1][1].settings.ub = r2
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm[1][1]) <= r2

    obj = SR(; rf = rf)
    rm = SKurt2(; settings = RiskMeasureSettings(; scale = 2.0))
    rm.settings.ub = r1
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r1

    rm = [[SKurt2(; kt = portfolio.skurt), SKurt2()]]
    rm[1][1].settings.ub = r2
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm[1][1]) <= r2

    # Ret lower bound
    obj = MinRisk()
    rm = SKurt2(; settings = RiskMeasureSettings(; scale = 2.0))
    rm.settings.ub = Inf
    portfolio.mu_l = ret1
    w5 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w5.weights) >= ret1

    rm = [[SKurt2(), SKurt2(; kt = portfolio.skurt)]]
    portfolio.mu_l = ret2
    w6 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w6.weights) >= ret2

    obj = SR(; rf = rf)
    rm = SKurt2(; settings = RiskMeasureSettings(; scale = 2.0))
    portfolio.mu_l = ret1
    w7 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w7.weights) >= ret1

    rm = [[SKurt2(), SKurt2()]]
    portfolio.mu_l = ret2
    w8 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w8.weights) >= ret2
end

@testset "Add Skew and SSkew to SD" begin
    portfolio = Portfolio2(; prices = prices,
                           solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                            :params => Dict("verbose" => false))))
    asset_statistics2!(portfolio)

    rm = Skew2(; settings = RiskMeasureSettings(; scale = 1.0))
    obj = MinRisk()
    w1 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    rm.settings.scale = 0.99
    w2 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    @test isapprox(w1.weights, w2.weights, rtol = 5e-6)

    rm = SSkew2(; settings = RiskMeasureSettings(; scale = 1.0))
    obj = MinRisk()
    w1 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    rm.settings.scale = 0.99
    w2 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    @test isapprox(w1.weights, w2.weights, rtol = 1e-4)
end
