@testset "RLDaR vec" begin
    portfolio = OmniPortfolio(; prices = prices,
                              solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                               :check_sol => (allow_local = true,
                                                                              allow_almost = true),
                                                               :params => Dict("verbose" => false,
                                                                               "max_step_fraction" => 0.75))))
    asset_statistics!(portfolio)

    obj = MinRisk()
    wt0 = [0.056204547878128584, 3.0259800260869496e-10, 1.555253414894205e-9,
           1.8034538925884771e-10, 0.02995909477889712, 4.1576845487613254e-10,
           6.991483640848045e-10, 0.02066795658490318, 2.7039446504260417e-10,
           5.654729719892038e-10, 0.4414852157629007, 4.732604971945372e-10,
           0.02175802110770559, 1.3939034660782987e-10, 1.2836606577463974e-10,
           0.1434507494122351, 0.15965737951388045, 0.06193182311007122,
           7.588623991638673e-10, 0.0648852063624177]
    riskt0 = 0.07576351356344437
    rett0 = 0.0005794990295959008

    rm = RLDaR(; settings = RMSettings(; scale = 1.0))
    w1 = optimise!(portfolio, Trad(; rm = rm, kelly = NoKelly(), obj = obj))
    r1 = calc_risk(portfolio; type = :Trad, rm = rm)
    ret1 = dot(portfolio.mu, w1.weights)
    wt = [0.05620452658327181, 3.4764146928942893e-10, 5.734877372759312e-10,
          1.8344046550600199e-10, 0.02995900962804213, 5.248155758807402e-10,
          3.9527963735244927e-10, 0.020667941663096986, 3.1588175694268626e-10,
          7.337263048736264e-10, 0.44148529173460954, 6.374978708090692e-10,
          0.021758041273862418, 1.4544127465459826e-10, 1.5403052159158215e-10,
          0.1434506868365832, 0.15965748652335518, 0.061931528622712555,
          8.378548679507754e-10, 0.06488548228536865]
    riskt = 0.07576351003435194
    rett = 0.0005794988152389773
    @test isapprox(w1.weights, wt0, rtol = 1.0e-6)
    @test isapprox(r1, riskt0, rtol = 1.0e-7)
    @test isapprox(ret1, rett0, rtol = 1.0e-6)
    @test isapprox(w1.weights, wt, rtol = 5.0e-6)
    @test isapprox(r1, riskt, rtol = 1.0e-7)
    @test isapprox(ret1, rett, rtol = 5.0e-7)

    rm = [[RLDaR(), RLDaR()]]
    w2 = optimise!(portfolio, Trad(; rm = rm, kelly = NoKelly(), obj = obj))
    r2 = calc_risk(portfolio; type = :Trad, rm = rm[1][1])
    ret2 = dot(portfolio.mu, w2.weights)
    wt = [0.05620453278532932, 1.3747585252970507e-10, 4.6203396083366557e-10,
          4.421854922127261e-11, 0.029959316084278997, 1.4948180138934594e-10,
          8.762194884412426e-11, 0.020668220448327352, 8.842559346430314e-11,
          1.9764220880249978e-10, 0.441485102795407, 1.6596232626444434e-10,
          0.021757966639808116, 3.9799915681603316e-11, 3.248264940373496e-11,
          0.14345051114496193, 0.15965660128950152, 0.061934130689017576,
          3.306287153631385e-10, 0.06488361638759452]
    riskt = 0.07576351461846881
    rett = 0.0005794983351148754
    @test isapprox(w2.weights, wt0, rtol = 1.0e-5)
    @test isapprox(r2, riskt0, rtol = 1.0e-7)
    @test isapprox(ret2, rett0, rtol = 5.0e-6)
    @test isapprox(w2.weights, wt, rtol = 5.0e-5)
    @test isapprox(r2, riskt, rtol = 1.0e-7)
    @test isapprox(ret2, rett, rtol = 5.0e-6)

    obj = Sharpe(; rf = rf)
    wt0 = [1.620183105236905e-8, 1.1140035827148705e-8, 0.04269821778198461,
           4.539811056982736e-9, 0.27528471583276515, 2.5884954318029336e-9,
           4.359695142864356e-9, 0.09149825388403467, 3.6147826445679577e-9,
           4.148787837320512e-9, 0.3004550564031882, 2.15879252345473e-9,
           1.3866951024663507e-9, 6.227749990512301e-9, 2.022396335125738e-9,
           0.2900636413773165, 1.8039723473841205e-8, 9.883423967231107e-9,
           8.029172726894051e-9, 2.0379317768180928e-8]
    riskt0 = 0.09342425156101017
    rett0 = 0.0009783083257672756

    rm = RLDaR(; settings = RMSettings(; scale = 2.0))
    w3 = optimise!(portfolio, Trad(; rm = rm, kelly = NoKelly(), obj = obj))
    r3 = calc_risk(portfolio; type = :Trad, rm = rm)
    ret3 = dot(portfolio.mu, w3.weights)
    wt = [4.5760069600096634e-9, 3.0720467393845974e-9, 0.042683406788272905,
          1.2617977893516534e-9, 0.2752846738091935, 7.213173902108917e-10,
          1.25328591908192e-9, 0.09150588461561442, 1.0024693294562603e-9,
          1.152840800797083e-9, 0.3004564390827643, 6.009547018080882e-10,
          3.850680801557246e-10, 1.7186919251208136e-9, 5.617465208139532e-10,
          0.29006956377774407, 5.017343427577103e-9, 2.7420862717901794e-9,
          2.236873754105225e-9, 5.623881210968573e-9]
    riskt = 0.09342369025329782
    rett = 0.000978303316192452
    @test isapprox(w3.weights, wt0, rtol = 5.0e-5)
    @test isapprox(r3, riskt0, rtol = 1.0e-5)
    @test isapprox(ret3, rett0, rtol = 1.0e-5)
    @test isapprox(w3.weights, wt, rtol = 5.0e-8)
    @test isapprox(r3, riskt, rtol = 1.0e-7)
    @test isapprox(ret3, rett)

    rm = [[RLDaR(), RLDaR()]]
    w4 = optimise!(portfolio, Trad(; rm = rm, kelly = NoKelly(), obj = obj))
    r4 = calc_risk(portfolio; type = :Trad, rm = rm[1][1])
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
    rm = RLDaR(; settings = RMSettings(; scale = 2.0))
    rm.settings.ub = r1
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test abs(calc_risk(portfolio; type = :Trad, rm = rm) - r1) < 5e-6

    rm = [[RLDaR(), RLDaR()]]
    rm[1][1].settings.ub = r2
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio; type = :Trad, rm = rm[1][1]) <= r2 ||
          abs(calc_risk(portfolio; type = :Trad, rm = rm[1][1]) - r2) < 5e-7

    obj = Sharpe(; rf = rf)
    rm = RLDaR(; settings = RMSettings(; scale = 2.0))
    rm.settings.ub = r1 * 1.000001
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio; type = :Trad, rm = rm) <= r1 * 1.000001

    rm = [[RLDaR(), RLDaR()]]
    rm[1][1].settings.ub = r2
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio; type = :Trad, rm = rm[1][1]) <= r2 ||
          abs(calc_risk(portfolio; type = :Trad, rm = rm[1][1]) - r2) < 5e-7

    # Ret lower bound
    obj = MinRisk()
    rm = RLDaR(; settings = RMSettings(; scale = 2.0))
    rm.settings.ub = Inf
    portfolio.mu_l = ret1
    w5 = optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test dot(portfolio.mu, w5.weights) >= ret1

    rm = [[RLDaR(), RLDaR()]]
    portfolio.mu_l = ret2
    w6 = optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test dot(portfolio.mu, w6.weights) >= ret2

    obj = Sharpe(; rf = rf)
    rm = RLDaR(; settings = RMSettings(; scale = 2.0))
    portfolio.mu_l = ret1
    w7 = optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test dot(portfolio.mu, w7.weights) >= ret1

    rm = [[RLDaR(), RLDaR()]]
    portfolio.mu_l = ret2
    w8 = optimise!(portfolio, Trad(; rm = rm, obj = obj))
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
    rm = [[RLDaR(), RLDaR(; alpha = 0.75)]]
    w9 = optimise!(portfolio, Trad(; rm = rm, kelly = NoKelly(), obj = obj))
    r9 = calc_risk(portfolio; type = :Trad, rm = rm[1][1])
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
    @test isapprox(w9.weights, wt, rtol = 5.0e-6)
    @test isapprox(r9, riskt, rtol = 5.0e-7)
    @test isapprox(ret9, rett, rtol = 5.0e-7)
end

@testset "Kurt vec" begin
    portfolio = OmniPortfolio(; prices = prices,
                              solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                               :check_sol => (allow_local = true,
                                                                              allow_almost = true),
                                                               :params => Dict("verbose" => false,
                                                                               "max_step_fraction" => 0.75))))
    asset_statistics!(portfolio)

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

    rm = Kurt(; settings = RMSettings(; scale = 2.0))
    w1 = optimise!(portfolio, Trad(; rm = rm, kelly = NoKelly(), obj = obj))
    r1 = calc_risk(portfolio; type = :Trad, rm = rm)
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

    rm = [[Kurt(), Kurt(; kt = portfolio.kurt)]]
    w2 = optimise!(portfolio, Trad(; rm = rm, kelly = NoKelly(), obj = obj))
    r2 = calc_risk(portfolio; type = :Trad, rm = rm[1][1])
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

    obj = Sharpe(; rf = rf)
    wt0 = [3.6858654756755185e-8, 0.007745692921941757, 7.659402306115789e-7,
           0.06255101954920105, 0.21591225958933066, 1.164996778478312e-8,
           0.04812416132599465, 0.1265864284132457, 7.028702221705754e-8,
           6.689307437956413e-8, 0.19709510156536242, 1.0048124974067658e-8,
           5.6308070308085104e-9, 2.6819277429081264e-8, 6.042626796051015e-9,
           0.09193445125562229, 0.20844322199597035, 0.03380290484590025,
           0.00779744381804698, 6.3145495978703785e-6]
    riskt0 = 0.00020556674631177048
    rett0 = 0.0009657710513568699

    rm = Kurt(; settings = RMSettings(; scale = 2.0))
    w3 = optimise!(portfolio, Trad(; rm = rm, kelly = NoKelly(), obj = obj))
    r3 = calc_risk(portfolio; type = :Trad, rm = rm)
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

    rm = [[Kurt(; kt = portfolio.kurt), Kurt()]]
    w4 = optimise!(portfolio, Trad(; rm = rm, kelly = NoKelly(), obj = obj))
    r4 = calc_risk(portfolio; type = :Trad, rm = rm[1][1])
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
    rm = Kurt(; settings = RMSettings(; scale = 2.0))
    rm.settings.ub = r1
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio; type = :Trad, rm = rm) <= r1 ||
          abs(calc_risk(portfolio; type = :Trad, rm = rm) - r1) < 5e-9

    rm = [[Kurt(), Kurt(; kt = portfolio.kurt)]]
    rm[1][1].settings.ub = r2
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio; type = :Trad, rm = rm[1][1]) <= r2 ||
          abs(calc_risk(portfolio; type = :Trad, rm = rm[1][1]) - r2) < 5e-9

    obj = Sharpe(; rf = rf)
    rm = Kurt(; settings = RMSettings(; scale = 2.0))
    rm.settings.ub = r1
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio; type = :Trad, rm = rm) <= r1 ||
          abs(calc_risk(portfolio; type = :Trad, rm = rm) - r1) < 1e-10

    rm = [[Kurt(; kt = portfolio.kurt), Kurt()]]
    rm[1][1].settings.ub = r2
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio; type = :Trad, rm = rm[1][1]) <= r2 ||
          abs(calc_risk(portfolio; type = :Trad, rm = rm[1][1]) - r2) < 1e-10

    # Ret lower bound
    obj = MinRisk()
    rm = Kurt(; settings = RMSettings(; scale = 2.0))
    rm.settings.ub = Inf
    portfolio.mu_l = ret1
    w5 = optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test dot(portfolio.mu, w5.weights) >= ret1

    rm = [[Kurt(), Kurt(; kt = portfolio.kurt)]]
    portfolio.mu_l = ret2
    w6 = optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test dot(portfolio.mu, w6.weights) >= ret2

    obj = Sharpe(; rf = rf)
    rm = Kurt(; settings = RMSettings(; scale = 2.0))
    portfolio.mu_l = ret1
    w7 = optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test dot(portfolio.mu, w7.weights) >= ret1

    rm = [[Kurt(), Kurt()]]
    portfolio.mu_l = ret2
    w8 = optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test dot(portfolio.mu, w8.weights) >= ret2
end

@testset "Kurt Reduced vec" begin
    portfolio = OmniPortfolio(; prices = prices, max_num_assets_kurt = 1,
                              solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                               :check_sol => (allow_local = true,
                                                                              allow_almost = true),
                                                               :params => Dict("verbose" => false,
                                                                               "max_step_fraction" => 0.75))))
    asset_statistics!(portfolio)

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

    rm = Kurt(; settings = RMSettings(; scale = 2.0))
    w1 = optimise!(portfolio, Trad(; rm = rm, kelly = NoKelly(), obj = obj))
    r1 = calc_risk(portfolio; type = :Trad, rm = rm)
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

    rm = [[Kurt(), Kurt(; kt = portfolio.kurt)]]
    w2 = optimise!(portfolio, Trad(; rm = rm, kelly = NoKelly(), obj = obj))
    r2 = calc_risk(portfolio; type = :Trad, rm = rm[1][1])
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

    obj = Sharpe(; rf = rf)
    wt0 = [2.1869958623435718e-8, 1.2027489763838995e-7, 8.307710723713282e-8,
           0.04981239216610764, 0.19930305948126495, 9.890114269259893e-9,
           0.041826524860577356, 0.10466467192338667, 3.332719673064681e-8,
           4.0257776120217934e-8, 0.23960266714976633, 8.639963684421896e-9,
           4.958509460440708e-9, 2.0027087137090508e-8, 5.256105741068125e-9,
           0.08392271758019651, 0.28086715326117795, 2.469501355591167e-7,
           1.0503063749371977e-7, 1.1401803293988891e-7]
    riskt0 = 0.00020482110250140048
    rett0 = 0.0009552956056983061

    rm = Kurt(; settings = RMSettings(; scale = 2.0))
    w3 = optimise!(portfolio, Trad(; rm = rm, kelly = NoKelly(), obj = obj))
    r3 = calc_risk(portfolio; type = :Trad, rm = rm)
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
    @test isapprox(w3.weights, wt, rtol = 5.0e-8)
    @test isapprox(r3, riskt)
    @test isapprox(ret3, rett, rtol = 5.0e-7)

    rm = [[Kurt(; kt = portfolio.kurt), Kurt()]]
    w4 = optimise!(portfolio, Trad(; rm = rm, kelly = NoKelly(), obj = obj))
    r4 = calc_risk(portfolio; type = :Trad, rm = rm[1][1])
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
    rm = Kurt(; settings = RMSettings(; scale = 1.0))
    rm.settings.ub = r1
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio; type = :Trad, rm = rm) <= r1 ||
          abs(calc_risk(portfolio; type = :Trad, rm = rm) - r1) < 5e-5

    rm = [[Kurt(), Kurt(; kt = portfolio.kurt)]]
    rm[1][1].settings.ub = r2
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio; type = :Trad, rm = rm[1][1]) <= r2 ||
          abs(calc_risk(portfolio; type = :Trad, rm = rm[1][1]) - r2) < 5e-5

    obj = Sharpe(; rf = rf)
    rm = Kurt(; settings = RMSettings(; scale = 2.0))
    rm.settings.ub = r1
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio; type = :Trad, rm = rm) <= r1 ||
          abs(calc_risk(portfolio; type = :Trad, rm = rm) - r1) < 5e-5

    rm = [[Kurt(; kt = portfolio.kurt), Kurt()]]
    rm[1][1].settings.ub = r2
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio; type = :Trad, rm = rm[1][1]) <= r2 ||
          abs(calc_risk(portfolio; type = :Trad, rm = rm[1][1]) - r2) < 5e-5

    # Ret lower bound
    obj = MinRisk()
    rm = Kurt(; settings = RMSettings(; scale = 2.0))
    rm.settings.ub = Inf
    portfolio.mu_l = ret1
    w5 = optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test dot(portfolio.mu, w5.weights) >= ret1

    rm = [[Kurt(), Kurt(; kt = portfolio.kurt)]]
    portfolio.mu_l = ret2
    w6 = optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test dot(portfolio.mu, w6.weights) >= ret2

    obj = Sharpe(; rf = rf)
    rm = Kurt(; settings = RMSettings(; scale = 2.0))
    portfolio.mu_l = ret1
    w7 = optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test dot(portfolio.mu, w7.weights) >= ret1

    rm = [[Kurt(), Kurt()]]
    portfolio.mu_l = ret2
    w8 = optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test dot(portfolio.mu, w8.weights) >= ret2
end

@testset "SKurt vec" begin
    portfolio = OmniPortfolio(; prices = prices,
                              solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                               :check_sol => (allow_local = true,
                                                                              allow_almost = true),
                                                               :params => Dict("verbose" => false,
                                                                               "max_step_fraction" => 0.75))))
    asset_statistics!(portfolio)

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

    rm = SKurt(; settings = RMSettings(; scale = 2.0))
    w1 = optimise!(portfolio, Trad(; rm = rm, kelly = NoKelly(), obj = obj))
    r1 = calc_risk(portfolio; type = :Trad, rm = rm)
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

    rm = [[SKurt(), SKurt(; kt = portfolio.skurt)]]
    w2 = optimise!(portfolio, Trad(; rm = rm, kelly = NoKelly(), obj = obj))
    r2 = calc_risk(portfolio; type = :Trad, rm = rm[1][1])
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

    obj = Sharpe(; rf = rf)
    wt0 = [8.263501742935076e-8, 3.4292121981081834e-5, 8.553642485927413e-8,
           1.0517936993748617e-7, 0.32096393208295665, 1.9848129848992033e-8,
           0.004686373576551095, 0.05273663584046675, 6.751336821499605e-8,
           5.166788731416254e-8, 0.24476417791090582, 1.7957314779078257e-8,
           9.390528572513579e-9, 5.4617904908192196e-8, 1.060832570103225e-8,
           0.135854021084771, 0.24095692751428416, 1.9017748026664851e-6,
           2.5600086153798754e-7, 9.771381477598014e-7]
    riskt0 = 0.0001695775787535489
    rett0 = 0.0010654948366056365

    rm = SKurt(; settings = RMSettings(; scale = 2.0))
    w3 = optimise!(portfolio, Trad(; rm = rm, kelly = NoKelly(), obj = obj))
    r3 = calc_risk(portfolio; type = :Trad, rm = rm)
    ret3 = dot(portfolio.mu, w3.weights)
    wt = [4.072562380759179e-8, 1.6401601258952214e-5, 4.2341099056426447e-8,
          5.197329699483326e-8, 0.32096785143338913, 9.849563533550322e-9,
          0.0046861732786994785, 0.05273771402529875, 3.334301705047341e-8,
          2.551337120345484e-8, 0.2447654292337497, 8.897677423341471e-9,
          4.662881441791528e-9, 2.6916523931691957e-8, 5.268010487161288e-9,
          0.13585674018866847, 0.24096793744666997, 9.050668480349218e-7,
          1.2501358531547884e-7, 4.7322076760474195e-7]
    riskt = 0.00016957876787065916
    rett = 0.001065501597545198
    @test isapprox(w3.weights, wt0, rtol = 5.0e-5)
    @test isapprox(r3, riskt0, rtol = 1.0e-5)
    @test isapprox(ret3, rett0, rtol = 1.0e-5)
    @test isapprox(w3.weights, wt)
    @test isapprox(r3, riskt)
    @test isapprox(ret3, rett)

    rm = [[SKurt(; kt = portfolio.skurt), SKurt()]]
    w4 = optimise!(portfolio, Trad(; rm = rm, kelly = NoKelly(), obj = obj))
    r4 = calc_risk(portfolio; type = :Trad, rm = rm[1][1])
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
    rm = SKurt(; settings = RMSettings(; scale = 2.0))
    rm.settings.ub = r1 * 1.05
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio; type = :Trad, rm = rm) <= r1 * 1.05

    rm = [[SKurt(), SKurt(; kt = portfolio.skurt)]]
    rm[1][1].settings.ub = r2 * 1.05
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio; type = :Trad, rm = rm[1][1]) <= r2 * 1.05

    obj = Sharpe(; rf = rf)
    rm = SKurt(; settings = RMSettings(; scale = 2.0))
    rm.settings.ub = r1 * 1.05
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio; type = :Trad, rm = rm) <= r1 * 1.05

    rm = [[SKurt(; kt = portfolio.skurt), SKurt()]]
    rm[1][1].settings.ub = r2 * 1.05
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio; type = :Trad, rm = rm[1][1]) <= r2 * 1.05

    # Ret lower bound
    obj = MinRisk()
    rm = SKurt(; settings = RMSettings(; scale = 2.0))
    rm.settings.ub = Inf
    portfolio.mu_l = ret1
    w5 = optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test dot(portfolio.mu, w5.weights) >= ret1

    rm = [[SKurt(), SKurt(; kt = portfolio.skurt)]]
    portfolio.mu_l = ret2
    w6 = optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test dot(portfolio.mu, w6.weights) >= ret2

    obj = Sharpe(; rf = rf)
    rm = SKurt(; settings = RMSettings(; scale = 2.0))
    portfolio.mu_l = ret1
    w7 = optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test dot(portfolio.mu, w7.weights) >= ret1

    rm = [[SKurt(), SKurt()]]
    portfolio.mu_l = ret2
    w8 = optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test dot(portfolio.mu, w8.weights) >= ret2
end

@testset "Kurt Reduced vec" begin
    portfolio = OmniPortfolio(; prices = prices, max_num_assets_kurt = 1,
                              solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                               :check_sol => (allow_local = true,
                                                                              allow_almost = true),
                                                               :params => Dict("verbose" => false,
                                                                               "max_step_fraction" => 0.75))))
    asset_statistics!(portfolio)

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

    rm = SKurt(; settings = RMSettings(; scale = 2.0))
    w1 = optimise!(portfolio, Trad(; rm = rm, kelly = NoKelly(), obj = obj))
    r1 = calc_risk(portfolio; type = :Trad, rm = rm)
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

    rm = [[SKurt(), SKurt(; kt = portfolio.skurt)]]
    w2 = optimise!(portfolio, Trad(; rm = rm, kelly = NoKelly(), obj = obj))
    r2 = calc_risk(portfolio; type = :Trad, rm = rm[1][1])
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

    obj = Sharpe(; rf = rf)
    wt0 = [9.329112161227789e-8, 1.8997236041327231e-6, 9.183667269717346e-8,
           1.1278547529605994e-7, 0.323307535710129, 2.5473368939712553e-8,
           6.050657719406614e-6, 0.04416525712754569, 7.422093025293419e-8,
           6.013968073613049e-8, 0.2607281038602079, 2.2840291563131592e-8,
           1.2053370397405268e-8, 6.734594412605242e-8, 1.3612437340851914e-8,
           0.13505441433828766, 0.23673428116977457, 1.0078714608278738e-6,
           2.4778495858369644e-7, 6.28157019375726e-7]
    riskt0 = 0.00016836304121264654
    rett0 = 0.0010573999779863363

    rm = SKurt(; settings = RMSettings(; scale = 2.0))
    w3 = optimise!(portfolio, Trad(; rm = rm, kelly = NoKelly(), obj = obj))
    r3 = calc_risk(portfolio; type = :Trad, rm = rm)
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

    rm = [[SKurt(; kt = portfolio.skurt), SKurt()]]
    w4 = optimise!(portfolio, Trad(; rm = rm, kelly = NoKelly(), obj = obj))
    r4 = calc_risk(portfolio; type = :Trad, rm = rm[1][1])
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
    rm = SKurt(; settings = RMSettings(; scale = 1.0))
    rm.settings.ub = r1
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio; type = :Trad, rm = rm) <= r1

    rm = [[SKurt(), SKurt(; kt = portfolio.skurt)]]
    rm[1][1].settings.ub = r2
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio; type = :Trad, rm = rm[1][1]) <= r2

    obj = Sharpe(; rf = rf)
    rm = SKurt(; settings = RMSettings(; scale = 2.0))
    rm.settings.ub = r1
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio; type = :Trad, rm = rm) <= r1

    rm = [[SKurt(; kt = portfolio.skurt), SKurt()]]
    rm[1][1].settings.ub = r2
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio; type = :Trad, rm = rm[1][1]) <= r2

    # Ret lower bound
    obj = MinRisk()
    rm = SKurt(; settings = RMSettings(; scale = 2.0))
    rm.settings.ub = Inf
    portfolio.mu_l = ret1
    w5 = optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test dot(portfolio.mu, w5.weights) >= ret1

    rm = [[SKurt(), SKurt(; kt = portfolio.skurt)]]
    portfolio.mu_l = ret2
    w6 = optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test dot(portfolio.mu, w6.weights) >= ret2

    obj = Sharpe(; rf = rf)
    rm = SKurt(; settings = RMSettings(; scale = 2.0))
    portfolio.mu_l = ret1
    w7 = optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test dot(portfolio.mu, w7.weights) >= ret1

    rm = [[SKurt(), SKurt()]]
    portfolio.mu_l = ret2
    w8 = optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test dot(portfolio.mu, w8.weights) >= ret2
end

@testset "TG vec" begin
    portfolio = OmniPortfolio(; prices = prices[(end - 200):end],
                              solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                               :check_sol => (allow_local = true,
                                                                              allow_almost = true),
                                                               :params => Dict("verbose" => false,
                                                                               "max_step_fraction" => 0.75,
                                                                               "max_iter" => 100,
                                                                               "equilibrate_max_iter" => 20))))
    asset_statistics!(portfolio)

    obj = MinRisk()
    wt0 = [5.969325155308689e-13, 0.19890625725641747, 5.01196330972591e-12,
           0.05490289350205719, 4.555898520654742e-11, 1.0828830360388467e-12,
           3.3254529493732914e-12, 0.10667398877393132, 1.0523685498886423e-12,
           1.8363867378603495e-11, 0.0486780391076569, 0.09089397416963123,
           3.698655012660736e-12, 8.901903375667794e-12, 2.0443054631623374e-12,
           0.13418199806574263, 6.531476985844397e-12, 0.13829581359726595,
           1.724738947709949e-12, 0.22746703542940375]
    riskt0 = 0.02356383470533441
    rett0 = 0.0005937393209710076

    rm = TG(; settings = RMSettings(; scale = 2.0), owa = OWASettings(; approx = false))
    w1 = optimise!(portfolio, Trad(; rm = rm, kelly = NoKelly(), obj = obj))
    r1 = calc_risk(portfolio; type = :Trad, rm = rm)
    ret1 = dot(portfolio.mu, w1.weights)
    wt = [9.117129296000855e-13, 0.1989062604779038, 5.091046781890198e-12,
          0.054902892230445206, 4.656142975466791e-11, 1.5118793197837656e-12,
          3.8540198560422116e-12, 0.10667398953978909, 1.3175608677583888e-12,
          1.9772109268754522e-11, 0.04867803522723395, 0.09089397449417429,
          4.256470959584207e-12, 9.23111680963713e-12, 1.7711791864713846e-12,
          0.1341819980026756, 6.777051511565845e-12, 0.13829581436956526,
          1.6817643269228358e-12, 0.22746703555547537]
    riskt = 0.023563834706007303
    rett = 0.0005937393233563302
    @test isapprox(w1.weights, wt0)
    @test isapprox(r1, riskt0)
    @test isapprox(ret1, rett0)
    @test isapprox(w1.weights, wt)
    @test isapprox(r1, riskt)
    @test isapprox(ret1, rett)

    rm = [[TG(; owa = OWASettings(; approx = false)),
           TG(; owa = OWASettings(; approx = false))]]
    w2 = optimise!(portfolio, Trad(; rm = rm, kelly = NoKelly(), obj = obj))
    r2 = calc_risk(portfolio; type = :Trad, rm = rm[1][1])
    ret2 = dot(portfolio.mu, w2.weights)
    wt = [4.641528551675654e-12, 0.19890627715965065, 2.6164624191846925e-13,
          0.054902895298161516, 4.196501505287773e-11, 5.113631855168327e-12,
          6.787750163857296e-12, 0.106673990078965, 5.062864104596882e-12,
          1.0352546837427235e-11, 0.04867801775229248, 0.09089397404658377,
          7.108498848500437e-12, 2.7425743138904577e-12, 2.6840570215544006e-12,
          0.13418199203670245, 1.0078682496776611e-12, 0.13829581534257337,
          2.844310734614091e-12, 0.2274670381944986]
    riskt = 0.023563834710486266
    rett = 0.0005937393357118465
    @test isapprox(w2.weights, wt0, rtol = 1.0e-7)
    @test isapprox(r2, riskt0)
    @test isapprox(ret2, rett0, rtol = 5.0e-8)
    @test isapprox(w2.weights, wt)
    @test isapprox(r2, riskt)
    @test isapprox(ret2, rett)

    obj = Sharpe(; rf = rf)
    wt0 = [1.0342189964374973e-11, 1.400649279934334e-11, 1.0025902458601371e-11,
           6.108757972397652e-12, 0.35723529048659247, 1.2069564550657953e-11,
           1.2830347528094887e-11, 1.1781378314488614e-12, 6.5343202013864566e-12,
           2.895398409002917e-12, 9.118697983089066e-12, 1.0966314618191202e-11,
           1.3147762207425575e-11, 9.611939363025545e-12, 1.263250173104243e-11,
           0.21190800224274572, 0.4308567071123932, 7.712262968799656e-12,
           1.2412631429194858e-11, 6.675313779730199e-12]
    riskt0 = 0.03208927225110264
    rett0 = 0.0017686555804583674

    rm = TG(; settings = RMSettings(; scale = 2.0), owa = OWASettings(; approx = false))
    w3 = optimise!(portfolio, Trad(; rm = rm, kelly = NoKelly(), obj = obj))
    r3 = calc_risk(portfolio; type = :Trad, rm = rm)
    ret3 = dot(portfolio.mu, w3.weights)
    wt = [1.7628494451185628e-11, 6.309210814522966e-12, 1.73340851042146e-11,
          1.2094716457413075e-11, 0.35723528912382363, 2.055184073410796e-11,
          2.047384063050713e-11, 5.9981511062419914e-12, 4.013121626457183e-12,
          1.0610718968359735e-11, 1.6004699009244607e-11, 1.832523859937056e-11,
          2.0738137429563127e-11, 1.6615403276744508e-11, 2.021575771586549e-11,
          0.21190800221923103, 0.4308567084114284, 1.4469393378858674e-11,
          1.1144282063101474e-11, 1.299013644991039e-11]
    riskt = 0.03208927224682404
    rett = 0.001768655580040797
    @test isapprox(w3.weights, wt0)
    @test isapprox(r3, riskt0)
    @test isapprox(ret3, rett0)
    @test isapprox(w3.weights, wt)
    @test isapprox(r3, riskt)
    @test isapprox(ret3, rett)

    rm = [[TG(; owa = OWASettings(; approx = false)),
           TG(; owa = OWASettings(; approx = false))]]
    w4 = optimise!(portfolio, Trad(; rm = rm, kelly = NoKelly(), obj = obj))
    r4 = calc_risk(portfolio; type = :Trad, rm = rm[1][1])
    ret4 = dot(portfolio.mu, w4.weights)
    wt = [9.27189739926205e-12, 2.41888502907255e-12, 9.261436640083589e-12,
          6.688973543852134e-12, 0.3572352895737421, 9.602174585541168e-12,
          9.742071453613786e-12, 5.758080242369661e-12, 3.6338662109179993e-12,
          7.021702688461076e-12, 8.794481459242355e-12, 9.356771227591303e-12,
          9.630742521058502e-12, 8.932139581595437e-12, 9.675622394371964e-12,
          0.21190800382409697, 0.4308567064744369, 8.521841437957333e-12,
          1.4535712298415591e-12, 7.959924261484958e-12]
    riskt = 0.03208927224100061
    rett = 0.001768655580065531
    @test isapprox(w4.weights, wt0)
    @test isapprox(r4, riskt0)
    @test isapprox(ret4, rett0)
    @test isapprox(w4.weights, wt)
    @test isapprox(r4, riskt)
    @test isapprox(ret4, rett)

    # Risk upper bound
    obj = MaxRet()
    rm = TG(; settings = RMSettings(; scale = 2.0), owa = OWASettings(; approx = false))
    rm.settings.ub = r1
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test abs(calc_risk(portfolio; type = :Trad, rm = rm) - r1) < 1e-8

    rm = [[TG(; owa = OWASettings(; approx = false)),
           TG(; owa = OWASettings(; approx = false))]]
    rm[1][1].settings.ub = r2
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test abs(calc_risk(portfolio; type = :Trad, rm = rm[1][1]) - r2) < 1e-8

    obj = Sharpe(; rf = rf)
    rm = TG(; settings = RMSettings(; scale = 2.0), owa = OWASettings(; approx = false))
    rm.settings.ub = r1
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio; type = :Trad, rm = rm) <= r1 ||
          abs(calc_risk(portfolio; type = :Trad, rm = rm) - r1) < 1e-9

    rm = [[TG(; owa = OWASettings(; approx = false)),
           TG(; owa = OWASettings(; approx = false))]]
    rm[1][1].settings.ub = r2
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio; type = :Trad, rm = rm[1][1]) <= r2 ||
          abs(calc_risk(portfolio; type = :Trad, rm = rm[1][1]) - r2) < 5e-6

    # Ret lower bound
    obj = MinRisk()
    rm = TG(; settings = RMSettings(; scale = 2.0), owa = OWASettings(; approx = false))
    rm.settings.ub = Inf
    portfolio.mu_l = ret1
    w5 = optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test dot(portfolio.mu, w5.weights) >= ret1

    rm = [[TG(; owa = OWASettings(; approx = false)),
           TG(; owa = OWASettings(; approx = false))]]
    portfolio.mu_l = ret2
    w6 = optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test dot(portfolio.mu, w6.weights) >= ret2

    obj = Sharpe(; rf = rf)
    rm = TG(; settings = RMSettings(; scale = 2.0), owa = OWASettings(; approx = false))
    portfolio.mu_l = ret1
    w7 = optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test dot(portfolio.mu, w7.weights) >= ret1

    rm = [[TG(; owa = OWASettings(; approx = false)),
           TG(; owa = OWASettings(; approx = false))]]
    portfolio.mu_l = ret2
    w8 = optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test dot(portfolio.mu, w8.weights) >= ret2

    portfolio.mu_l = Inf
    obj = MinRisk()
    wt0 = [5.969325155308689e-13, 0.19890625725641747, 5.01196330972591e-12,
           0.05490289350205719, 4.555898520654742e-11, 1.0828830360388467e-12,
           3.3254529493732914e-12, 0.10667398877393132, 1.0523685498886423e-12,
           1.8363867378603495e-11, 0.0486780391076569, 0.09089397416963123,
           3.698655012660736e-12, 8.901903375667794e-12, 2.0443054631623374e-12,
           0.13418199806574263, 6.531476985844397e-12, 0.13829581359726595,
           1.724738947709949e-12, 0.22746703542940375]
    riskt0 = 0.02356383470533441
    rett0 = 0.0005937393209710076
    wt1 = [1.2836541099300374e-10, 7.632300965664097e-10, 1.919887858998074e-10,
           4.801955740158814e-10, 0.0982889108245419, 5.651545334295017e-11,
           1.6576927251449667e-11, 0.07900741935773047, 6.319231416335898e-10,
           8.34058889694369e-8, 0.043496746354597604, 2.57787733628572e-11,
           7.687212211788456e-11, 0.11246278997924469, 5.4971523683089116e-11,
           0.03112348236478457, 0.1477185644358598, 0.17254181002118, 0.158714558264814,
           0.1566456325649403]
    riskt1 = 0.0052910845613359445
    rett1 = 0.0008709132492808145

    rm = [[TG(; owa = OWASettings(; approx = false)),
           TG(; alpha = 0.75, owa = OWASettings(; approx = false))]]
    w9 = optimise!(portfolio, Trad(; rm = rm, kelly = NoKelly(), obj = obj))
    r9 = calc_risk(portfolio; type = :Trad, rm = rm[1][1])
    ret9 = dot(portfolio.mu, w9.weights)
    wt = [2.0365478576275784e-12, 0.26110976994697394, 3.4935250075823384e-11,
          0.014232056343286167, 0.025589079693026068, 4.6955664317896684e-12,
          1.280870684858824e-11, 0.09709844547134337, 5.464466441696193e-13,
          1.1641073623944162e-10, 0.08852689155201673, 0.022082626782455907,
          1.591050398876822e-11, 6.63033240411461e-10, 2.00094227838181e-12,
          0.0702929792386146, 1.5518347340153248e-10, 0.23109956724678077,
          2.3551644968730975e-11, 0.18996858269438938]
    riskt = 0.02402876910565921
    rett = 0.0006220186938847865
    @test !isapprox(w9.weights, wt0)
    @test !isapprox(r9, riskt0)
    @test !isapprox(ret9, rett0)
    @test !isapprox(w9.weights, wt1)
    @test !isapprox(r9, riskt1)
    @test !isapprox(ret9, rett1)
    @test isapprox(w9.weights, wt, rtol = 0.0005)
    @test isapprox(r9, riskt, rtol = 5.0e-6)
    @test isapprox(ret9, rett, rtol = 5.0e-5)

    obj = MinRisk()
    wt0 = [2.1793176066144965e-10, 0.24657578304895264, 5.476679022837874e-10,
           0.041100700961631355, 1.745099342448546e-9, 2.0168069945246096e-10,
           7.629903785304511e-11, 0.11651659826585649, 1.9351039549294452e-10,
           1.0580904330304497e-9, 5.933842241374943e-8, 0.09732932587810658,
           5.619716498021303e-11, 6.782716439749812e-10, 3.4982414551865584e-10,
           0.14662256588896547, 5.824103810697346e-10, 0.12694239844522773,
           3.400868203663799e-10, 0.22491256212576752]
    riskt0 = 0.023582366401225324
    rett0 = 0.0006432235211782866

    rm = TG(; settings = RMSettings(; scale = 2.0), owa = OWASettings(; approx = true))
    w1 = optimise!(portfolio, Trad(; rm = rm, kelly = NoKelly(), obj = obj))
    r1 = calc_risk(portfolio; type = :Trad, rm = rm)
    ret1 = dot(portfolio.mu, w1.weights)
    wt = [3.9936340457424127e-10, 0.24657622800276421, 1.0029426420816195e-9,
          0.04109935113614855, 3.213182208801116e-9, 3.696223300650158e-10,
          1.3886189785425067e-10, 0.11651467765066632, 3.541515759525849e-10,
          1.9220249209401027e-9, 1.3472001756117787e-7, 0.09732827863666377,
          1.0160138214212973e-10, 1.255233136605034e-9, 6.484292746108698e-10,
          0.14662114332298815, 1.0783577922601826e-9, 0.12694764616541818,
          6.249160198927854e-10, 0.2249125292566467]
    riskt = 0.02358236418515349
    rett = 0.0006432212632505017
    @test isapprox(w1.weights, wt0, rtol = 5.0e-5)
    @test isapprox(r1, riskt0, rtol = 1.0e-7)
    @test isapprox(ret1, rett0, rtol = 5.0e-6)
    @test isapprox(w1.weights, wt, rtol = 5.0e-5)
    @test isapprox(r1, riskt, rtol = 5.0e-7)
    @test isapprox(ret1, rett, rtol = 1.0e-5)

    rm = [[TG(; owa = OWASettings(; approx = true)),
           TG(; owa = OWASettings(; approx = true))]]
    w2 = optimise!(portfolio, Trad(; rm = rm, kelly = NoKelly(), obj = obj))
    r2 = calc_risk(portfolio; type = :Trad, rm = rm[1][1])
    ret2 = dot(portfolio.mu, w2.weights)
    wt = [8.042383456053024e-11, 0.24657433721313826, 2.1254182969089954e-10,
          0.04110003318794624, 6.863865002714227e-10, 7.379687542758122e-11,
          2.3884451190172904e-11, 0.1165140233749262, 7.120153544400378e-11,
          4.1657467153520344e-10, 4.1826336336684103e-8, 0.09732779797208628,
          1.5892620143699496e-11, 2.651961620714389e-10, 1.3255150556956351e-10,
          0.14662024611748817, 2.2753025869209643e-10, 0.1269506375292963,
          1.298716222696399e-10, 0.22491288044293023]
    riskt = 0.023582361499711504
    rett = 0.0006432198776886395
    @test isapprox(w2.weights, wt0, rtol = 5.0e-5)
    @test isapprox(r2, riskt0, rtol = 5.0e-7)
    @test isapprox(ret2, rett0, rtol = 1.0e-5)
    @test isapprox(w2.weights, wt, rtol = 5.0e-5)
    @test isapprox(r2, riskt, rtol = 5.0e-7)
    @test isapprox(ret2, rett, rtol = 1.0e-5)

    obj = Sharpe(; rf = rf)
    wt0 = [2.465085712389692e-10, 1.7591592054961841e-9, 2.6079839244395587e-10,
           5.190986820248767e-10, 0.3290475106649799, 7.765809158263677e-11,
           1.0087407386198554e-10, 1.0105472171462993e-9, 1.5219079686966533e-9,
           6.719673558205139e-10, 3.7631629567205446e-10, 2.3450024422381644e-10,
           9.619358604737496e-11, 3.325560088347231e-10, 1.1281942009082258e-10,
           0.20499762407557243, 0.46595485476474174, 4.4765819636375454e-10,
           2.177943962387867e-9, 5.481986913612393e-10]
    riskt0 = 0.03204366080645988
    rett0 = 0.00176571347903363

    rm = TG(; settings = RMSettings(; scale = 2.0), owa = OWASettings(; approx = true))
    w3 = optimise!(portfolio, Trad(; rm = rm, kelly = NoKelly(), obj = obj))
    r3 = calc_risk(portfolio; type = :Trad, rm = rm)
    ret3 = dot(portfolio.mu, w3.weights)
    wt = [1.30958811238442e-10, 9.393064643496553e-10, 1.3859460639303259e-10,
          2.7641929970970515e-10, 0.32908074699916545, 4.0831848300802844e-11,
          5.321357998359013e-11, 5.390945651126132e-10, 8.118459844241153e-10,
          3.581643614310909e-10, 2.003119941264806e-10, 1.245657705529936e-10,
          5.073302868223789e-11, 1.7689645810943782e-10, 5.959970853453841e-11,
          0.2049920451953837, 0.46592720221227896, 2.383524310980431e-10,
          1.162120177366155e-9, 2.9216288025431493e-10]
    riskt = 0.032043725010131384
    rett = 0.0017657171780919168
    @test isapprox(w3.weights, wt0, rtol = 0.0001)
    @test isapprox(r3, riskt0, rtol = 5.0e-6)
    @test isapprox(ret3, rett0, rtol = 5.0e-6)
    @test isapprox(w3.weights, wt, rtol = 5.0e-5)
    @test isapprox(r3, riskt, rtol = 5.0e-6)
    @test isapprox(ret3, rett, rtol = 5.0e-6)

    rm = [[TG(; owa = OWASettings(; approx = true)),
           TG(; owa = OWASettings(; approx = true))]]
    w4 = optimise!(portfolio, Trad(; rm = rm, kelly = NoKelly(), obj = obj))
    r4 = calc_risk(portfolio; type = :Trad, rm = rm[1][1])
    ret4 = dot(portfolio.mu, w4.weights)
    wt = [7.969157789262045e-11, 5.766162636150547e-10, 8.43578632654925e-11,
          1.6869812629460985e-10, 0.3290723319262457, 2.457968693355534e-11,
          3.214627364480214e-11, 3.2981665272915827e-10, 4.961175585483804e-10,
          2.1886744520342848e-10, 1.2227967428248629e-10, 7.58694703847239e-11,
          3.063754370941499e-11, 1.0792869660473871e-10, 3.605814395871514e-11,
          0.20499368159023673, 0.4659339830651274, 1.4554016002709475e-10,
          7.104314686454891e-10, 1.7875349405006732e-10]
    riskt = 0.03204374129311252
    rett = 0.0017657180543370445
    @test isapprox(w4.weights, wt0, rtol = 0.0001)
    @test isapprox(r4, riskt0, rtol = 5.0e-6)
    @test isapprox(ret4, rett0, rtol = 5.0e-6)
    @test isapprox(w4.weights, wt, rtol = 5.0e-5)
    @test isapprox(r4, riskt, rtol = 5.0e-6)
    @test isapprox(ret4, rett, rtol = 5.0e-6)

    # Risk upper bound
    obj = MaxRet()
    rm = TG(; settings = RMSettings(; scale = 2.0), owa = OWASettings(; approx = true))
    rm.settings.ub = r1 * 1.001
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio; type = :Trad, rm = rm) <= r1 * 1.001

    rm = [[TG(; owa = OWASettings(; approx = true)),
           TG(; owa = OWASettings(; approx = true))]]
    rm[1][1].settings.ub = r2 * 1.001
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio; type = :Trad, rm = rm[1][1]) <= r2 * 1.001

    obj = Sharpe(; rf = rf)
    rm = TG(; settings = RMSettings(; scale = 2.0), owa = OWASettings(; approx = true))
    rm.settings.ub = r1 * 1.001
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio; type = :Trad, rm = rm) <= r1 * 1.001

    rm = [[TG(; owa = OWASettings(; approx = true)),
           TG(; owa = OWASettings(; approx = true))]]
    rm[1][1].settings.ub = r2 * 1.005
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    if !Sys.isapple()
        @test calc_risk(portfolio; type = :Trad, rm = rm[1][1]) <= r2 * 1.005
    end

    # Ret lower bound
    obj = MinRisk()
    rm = TG(; settings = RMSettings(; scale = 2.0), owa = OWASettings(; approx = true))
    rm.settings.ub = Inf
    portfolio.mu_l = ret1
    w5 = optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test dot(portfolio.mu, w5.weights) >= ret1

    rm = [[TG(; owa = OWASettings(; approx = true)),
           TG(; owa = OWASettings(; approx = true))]]
    portfolio.mu_l = ret2
    w6 = optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test dot(portfolio.mu, w6.weights) >= ret2

    obj = Sharpe(; rf = rf)
    rm = TG(; settings = RMSettings(; scale = 2.0), owa = OWASettings(; approx = true))
    portfolio.mu_l = ret1
    w7 = optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test dot(portfolio.mu, w7.weights) >= ret1

    rm = [[TG(; owa = OWASettings(; approx = true)),
           TG(; owa = OWASettings(; approx = true))]]
    portfolio.mu_l = ret2
    w8 = optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test dot(portfolio.mu, w8.weights) >= ret2

    portfolio.mu_l = Inf
    obj = MinRisk()
    wt0 = [1.424802287200348e-10, 0.24657565369133722, 3.660767298307013e-10,
           0.041101497481424845, 1.1765742472564645e-9, 1.3206711379272908e-10,
           4.948146901460501e-11, 0.11651694980250114, 1.27384858611194e-10,
           7.083586776439093e-10, 2.7784344366481497e-8, 0.09732942313209282,
           3.6499869966418024e-11, 4.3139455041429935e-10, 2.246994554915304e-10,
           0.14662262356233058, 3.7795902929060657e-10, 0.12694138139146063,
           2.229837567112617e-10, 0.22491243915854847]
    riskt0 = 0.023582366401225324
    rett0 = 0.000643224255466324
    wt1 = [7.300441198975879e-11, 3.733446529544342e-10, 1.0543513189637169e-10,
           2.3029529249627563e-10, 0.09845917917186836, 3.71261966988816e-11,
           1.86649050658783e-11, 0.0769519240514099, 3.0699196883209783e-10,
           0.00018304528861999298, 0.04462511459102036, 2.307293409612758e-11,
           4.744860664868727e-11, 0.11542959315117003, 3.730300447537313e-11,
           0.03289876924459179, 0.14723423110092668, 0.16996234638096502,
           0.15838244526051007, 0.15587335050623066]
    riskt1 = 0.005291194452999399
    rett1 = 0.0008696210533165451

    rm = [[TG(; owa = OWASettings(; approx = true)),
           TG(; alpha = 0.75, owa = OWASettings(; approx = true))]]
    w9 = optimise!(portfolio, Trad(; rm = rm, kelly = NoKelly(), obj = obj))
    r9 = calc_risk(portfolio; type = :Trad, rm = rm[1][1])
    ret9 = dot(portfolio.mu, w9.weights)
    wt = [7.50920171037976e-11, 0.2609567509058036, 1.8054952886985706e-10,
          0.020467743861486545, 0.008657062382432874, 5.3521376788735344e-11,
          2.4094798872980635e-11, 0.0937608422302937, 6.760517792773822e-11,
          4.262590556341936e-10, 0.08288409866568487, 0.024366883942363048,
          1.3548303594176315e-11, 2.7453073588686673e-9, 7.47791397994732e-11,
          0.07072019989759804, 7.695475396700068e-10, 0.24384420472955967,
          1.5820130362962118e-10, 0.19434220879627206]
    riskt = 0.024020116307912565
    rett = 0.0005994859757037167
    @test !isapprox(w9.weights, wt0)
    @test !isapprox(r9, riskt0)
    @test !isapprox(ret9, rett0)
    @test !isapprox(w9.weights, wt1)
    @test !isapprox(r9, riskt1)
    @test !isapprox(ret9, rett1)
    @test isapprox(w9.weights, wt, rtol = 5.0e-5)
    @test isapprox(r9, riskt, rtol = 1.0e-6)
    @test isapprox(ret9, rett, rtol = 1.0e-5)

    wt0 = [5.969325155308689e-13, 0.19890625725641747, 5.01196330972591e-12,
           0.05490289350205719, 4.555898520654742e-11, 1.0828830360388467e-12,
           3.3254529493732914e-12, 0.10667398877393132, 1.0523685498886423e-12,
           1.8363867378603495e-11, 0.0486780391076569, 0.09089397416963123,
           3.698655012660736e-12, 8.901903375667794e-12, 2.0443054631623374e-12,
           0.13418199806574263, 6.531476985844397e-12, 0.13829581359726595,
           1.724738947709949e-12, 0.22746703542940375]
    riskt0 = 0.02356383470533441
    rett0 = 0.0005937393209710076
    wt1 = [1.424802287200348e-10, 0.24657565369133722, 3.660767298307013e-10,
           0.041101497481424845, 1.1765742472564645e-9, 1.3206711379272908e-10,
           4.948146901460501e-11, 0.11651694980250114, 1.27384858611194e-10,
           7.083586776439093e-10, 2.7784344366481497e-8, 0.09732942313209282,
           3.6499869966418024e-11, 4.3139455041429935e-10, 2.246994554915304e-10,
           0.14662262356233058, 3.7795902929060657e-10, 0.12694138139146063,
           2.229837567112617e-10, 0.22491243915854847]
    riskt1 = 0.023582366401225324
    rett1 = 0.000643224255466324

    rm = [[TG(; settings = RMSettings(; scale = 10), owa = OWASettings(; approx = true)),
           TG(; settings = RMSettings(; scale = 10), owa = OWASettings(; approx = false))]]
    w10 = optimise!(portfolio, Trad(; rm = rm, kelly = NoKelly(), obj = obj))
    @test all(JuMP.value.(portfolio.model[:tga][:, 1]) .== 0)
    @test all(JuMP.value.(portfolio.model[:tga][:, 2]) .!= 0)
    @test JuMP.value(portfolio.model[:tg_t][1]) != 0
    @test JuMP.value(portfolio.model[:tg_t][2]) == 0

    rm = [[TG(; settings = RMSettings(; scale = 10), owa = OWASettings(; approx = false)),
           TG(; settings = RMSettings(; scale = 10), owa = OWASettings(; approx = true))]]
    w11 = optimise!(portfolio, Trad(; rm = rm, kelly = NoKelly(), obj = obj))
    @test all(JuMP.value.(portfolio.model[:tga][:, 1]) .!= 0)
    @test all(JuMP.value.(portfolio.model[:tga][:, 2]) .== 0)
    @test JuMP.value(portfolio.model[:tg_t][1]) == 0
    @test JuMP.value(portfolio.model[:tg_t][2]) != 0
end

@testset "TGRG vec" begin
    portfolio = OmniPortfolio(; prices = prices[(end - 200):end],
                              solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                               :check_sol => (allow_local = true,
                                                                              allow_almost = true),
                                                               :params => Dict("verbose" => false,
                                                                               "max_step_fraction" => 0.75,
                                                                               "max_iter" => 100,
                                                                               "equilibrate_max_iter" => 20))))
    asset_statistics!(portfolio)

    obj = MinRisk()
    wt0 = [1.1206142630614112e-11, 0.09772215128420475, 3.5338858649323828e-12,
           3.307433371895369e-11, 0.017417781863929577, 0.041201102000678634,
           1.6220089767187403e-11, 0.060553118743468366, 1.7795758713115547e-11,
           1.5483766349422575e-11, 0.22830001756787208, 2.6382663454855723e-11,
           3.59484391773399e-11, 0.13930594414393907, 2.0476463882653826e-11,
           0.0362260148933746, 9.71096463830921e-11, 0.24663479935480356,
           1.4791298973742443e-12, 0.13263906986901897]
    riskt0 = 0.040929059924343876
    rett0 = 0.00020307377724072516

    rm = TGRG(; settings = RMSettings(; scale = 2.0), owa = OWASettings(; approx = false))
    w1 = optimise!(portfolio, Trad(; rm = rm, kelly = NoKelly(), obj = obj))
    r1 = calc_risk(portfolio; type = :Trad, rm = rm)
    ret1 = dot(portfolio.mu, w1.weights)
    wt = [1.2225786446788121e-11, 0.09772217146590369, 5.673701473996134e-12,
          2.838678422820901e-11, 0.017417765814194564, 0.04120108956321286,
          1.7020460093659138e-11, 0.06055311540007961, 1.8215357850080928e-11,
          1.5993348924342457e-11, 0.22830001315390086, 2.2090204516394868e-11,
          3.4812544761278037e-11, 0.1393059408057302, 2.1043316789830012e-11,
          0.03622600298167753, 8.466516237025695e-11, 0.24663482910403023,
          6.705648025817591e-13, 0.13263907145047318]
    riskt = 0.04092905991714879
    rett = 0.00020307380112565102
    @test isapprox(w1.weights, wt0, rtol = 1.0e-5)
    @test isapprox(r1, riskt0)
    @test isapprox(ret1, rett0, rtol = 1.0e-6)
    @test isapprox(w1.weights, wt, rtol = 1.0e-5)
    @test isapprox(r1, riskt)
    @test isapprox(ret1, rett, rtol = 5.0e-6)

    rm = [[TGRG(; owa = OWASettings(; approx = false)),
           TGRG(; owa = OWASettings(; approx = false))]]
    w2 = optimise!(portfolio, Trad(; rm = rm, kelly = NoKelly(), obj = obj))
    r2 = calc_risk(portfolio; type = :Trad, rm = rm[1][1])
    ret2 = dot(portfolio.mu, w2.weights)
    wt = [6.5365499852512134e-12, 0.09772240434883234, 5.526476222151655e-12,
          1.326819254329726e-12, 0.01741759983339092, 0.04120101056973128,
          7.242479098728069e-12, 0.060553055589138934, 7.397133279605934e-12,
          6.8918853524363896e-12, 0.22829988406019494, 1.97108943741777e-12,
          9.58777480808517e-12, 0.13930592194943237, 7.624103666891728e-12,
          0.03622581778020346, 7.017382775264764e-12, 0.24663520816747836,
          4.979828035590969e-12, 0.13263909763549586]
    riskt = 0.040929059853266406
    rett = 0.00020307381171069216
    @test isapprox(w2.weights, wt0, rtol = 5.0e-6)
    @test isapprox(r2, riskt0)
    @test isapprox(ret2, rett0, rtol = 5.0e-7)
    @test isapprox(w2.weights, wt)
    @test isapprox(r2, riskt)
    @test isapprox(ret2, rett)

    obj = Sharpe(; rf = rf)
    wt0 = [4.8650129756778804e-11, 2.1051127604426756e-11, 4.7364768826598876e-11,
           4.001588817352124e-11, 0.1416419921187176, 5.571902073317586e-11,
           5.631693464060233e-11, 0.0249378701969849, 1.7997595945540913e-10,
           2.6619900499982886e-11, 4.154321525692782e-11, 5.2543818628034845e-11,
           5.781072172554138e-11, 4.516910259420552e-11, 5.695227967987287e-11,
           0.21453454572847452, 0.618885583612367, 3.483200049567782e-11,
           7.540575027899836e-9, 3.831617883258372e-11]
    riskt0 = 0.058407554117453894
    rett0 = 0.0017136727125023712

    rm = TGRG(; settings = RMSettings(; scale = 2.0), owa = OWASettings(; approx = false))
    w3 = optimise!(portfolio, Trad(; rm = rm, kelly = NoKelly(), obj = obj))
    r3 = calc_risk(portfolio; type = :Trad, rm = rm)
    ret3 = dot(portfolio.mu, w3.weights)
    wt = [3.4001037191815174e-11, 2.2355333787010714e-11, 3.3554316072973305e-11,
          2.8898887740881055e-11, 0.14164200133309077, 3.802902157957741e-11,
          3.7592926537025763e-11, 0.02493785949633177, 9.123469329072154e-11,
          2.5210980548228365e-11, 3.013523976362732e-11, 3.588577750596173e-11,
          3.812814290863558e-11, 3.199132311231273e-11, 3.787576693587153e-11,
          0.214534550564929, 0.6188855840604209, 2.7033319448566177e-11,
          4.0048021948108925e-9, 2.8498470326997418e-11]
    riskt = 0.05840755452044075
    rett = 0.0017136727239630205
    @test isapprox(w3.weights, wt0, rtol = 5.0e-8)
    @test isapprox(r3, riskt0)
    @test isapprox(ret3, rett0)
    @test isapprox(w3.weights, wt)
    @test isapprox(r3, riskt)
    @test isapprox(ret3, rett)

    rm = [[TGRG(; owa = OWASettings(; approx = false)),
           TGRG(; owa = OWASettings(; approx = false))]]
    w4 = optimise!(portfolio, Trad(; rm = rm, kelly = NoKelly(), obj = obj))
    r4 = calc_risk(portfolio; type = :Trad, rm = rm[1][1])
    ret4 = dot(portfolio.mu, w4.weights)
    wt = [1.8358434637862024e-11, 1.4083716698386552e-11, 1.821936894908436e-11,
          1.467819735551388e-11, 0.14164200346063474, 1.9381692081589777e-11,
          1.931121647829508e-11, 0.02493786121383846, 1.6419443753251767e-11,
          1.5082872446497197e-11, 1.7346712009358093e-11, 1.8967135874484723e-11,
          1.9393937194312647e-11, 1.7806575530914132e-11, 1.9401450625637668e-11,
          0.21453455281975672, 0.6188855809423065, 1.650621034495274e-11,
          1.3017199444830673e-9, 1.6786729002273376e-11]
    riskt = 0.05840755449786701
    rett = 0.00171367272384763
    @test isapprox(w4.weights, wt0, rtol = 5.0e-8)
    @test isapprox(r4, riskt0)
    @test isapprox(ret4, rett0)
    @test isapprox(w4.weights, wt)
    @test isapprox(r4, riskt)
    @test isapprox(ret4, rett)

    # Risk upper bound
    obj = MaxRet()
    rm = TGRG(; settings = RMSettings(; scale = 2.0), owa = OWASettings(; approx = false))
    rm.settings.ub = r1
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio; type = :Trad, rm = rm) <= r1 ||
          abs(calc_risk(portfolio; type = :Trad, rm = rm) - r1) < 5e-7

    rm = [[TGRG(; owa = OWASettings(; approx = false)),
           TGRG(; owa = OWASettings(; approx = false))]]
    rm[1][1].settings.ub = r2
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio; type = :Trad, rm = rm[1][1]) <= r2 ||
          abs(calc_risk(portfolio; type = :Trad, rm = rm[1][1]) - r2) < 5e-9

    obj = Sharpe(; rf = rf)
    rm = TGRG(; settings = RMSettings(; scale = 2.0), owa = OWASettings(; approx = false))
    rm.settings.ub = r1
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio; type = :Trad, rm = rm) <= r1 ||
          abs(calc_risk(portfolio; type = :Trad, rm = rm) - r1) < 5e-9

    rm = [[TGRG(; owa = OWASettings(; approx = false)),
           TGRG(; owa = OWASettings(; approx = false))]]
    rm[1][1].settings.ub = r2
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio; type = :Trad, rm = rm[1][1]) <= r2 ||
          abs(calc_risk(portfolio; type = :Trad, rm = rm[1][1]) - r2) < 5e-5

    # Ret lower bound
    obj = MinRisk()
    rm = TGRG(; settings = RMSettings(; scale = 2.0), owa = OWASettings(; approx = false))
    rm.settings.ub = Inf
    portfolio.mu_l = ret1
    w5 = optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test dot(portfolio.mu, w5.weights) >= ret1

    rm = [[TGRG(; owa = OWASettings(; approx = false)),
           TGRG(; owa = OWASettings(; approx = false))]]
    portfolio.mu_l = ret2
    w6 = optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test dot(portfolio.mu, w6.weights) >= ret2

    obj = Sharpe(; rf = rf)
    rm = TGRG(; settings = RMSettings(; scale = 2.0), owa = OWASettings(; approx = false))
    portfolio.mu_l = ret1
    w7 = optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test dot(portfolio.mu, w7.weights) >= ret1

    rm = [[TGRG(; owa = OWASettings(; approx = false)),
           TGRG(; owa = OWASettings(; approx = false))]]
    portfolio.mu_l = ret2
    w8 = optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test dot(portfolio.mu, w8.weights) >= ret2

    portfolio.mu_l = Inf
    obj = MinRisk()
    wt0 = [1.1206142630614112e-11, 0.09772215128420475, 3.5338858649323828e-12,
           3.307433371895369e-11, 0.017417781863929577, 0.041201102000678634,
           1.6220089767187403e-11, 0.060553118743468366, 1.7795758713115547e-11,
           1.5483766349422575e-11, 0.22830001756787208, 2.6382663454855723e-11,
           3.59484391773399e-11, 0.13930594414393907, 2.0476463882653826e-11,
           0.0362260148933746, 9.71096463830921e-11, 0.24663479935480356,
           1.4791298973742443e-12, 0.13263906986901897]
    riskt0 = 0.040929059924343876
    rett0 = 0.00020307377724072516
    wt1 = [1.0009153496459239e-10, 9.749900346647964e-11, 5.874551709857877e-11,
           5.5624273682545086e-11, 0.1063233767596915, 0.13310564164072994,
           5.386813608798474e-11, 0.07245409142850337, 3.3361351110701744e-10,
           8.12579663572646e-12, 0.1820105683187836, 1.602733364456541e-11,
           2.447576840031422e-11, 0.20383023498792277, 1.1981432205433173e-11,
           0.03264070951504424, 0.018170189957485795, 0.15729142359636483,
           0.02869810997957648, 0.06547565305584528]
    riskt1 = 0.02105727354969537
    rett1 = -1.5010409711686289e-5

    rm = [[TGRG(; owa = OWASettings(; approx = false)),
           TGRG(; alpha = 0.75, owa = OWASettings(; approx = false))]]
    w9 = optimise!(portfolio, Trad(; rm = rm, kelly = NoKelly(), obj = obj))
    r9 = calc_risk(portfolio; type = :Trad, rm = rm[1][1])
    ret9 = dot(portfolio.mu, w9.weights)
    wt = [1.6916066220987235e-11, 0.05805570333905351, 2.622481909992745e-11,
          1.111708235686416e-11, 0.08643136754204687, 0.10355430344702476,
          3.368389615174491e-11, 0.06137059798951525, 2.13353503436033e-11,
          3.852317738835321e-11, 0.21247686482854392, 4.367438940550122e-11,
          6.261241963177479e-11, 0.19241699869507678, 4.875570562353115e-11,
          0.02403732695302407, 3.074666561844692e-10, 0.18479462947064082,
          2.2524154065461607e-11, 0.07686220710223998]
    riskt = 0.04110659227915137
    rett = 2.4539426389906168e-5
    @test !isapprox(w9.weights, wt0)
    @test !isapprox(r9, riskt0)
    @test !isapprox(ret9, rett0)
    @test !isapprox(w9.weights, wt1)
    @test !isapprox(r9, riskt1)
    @test !isapprox(ret9, rett1)
    @test isapprox(w9.weights, wt, rtol = 0.0005)
    @test isapprox(r9, riskt, rtol = 1.0e-5)
    @test isapprox(ret9, rett, rtol = 0.005)

    obj = MinRisk()
    wt0 = [1.5554085500133127e-10, 0.07595979756754843, 1.9905556106887235e-10,
           3.6624716029666307e-10, 0.01615916526158566, 0.06416386867818721,
           1.2848115418862524e-10, 0.06291523433064322, 1.2090701833560976e-10,
           1.1827681587213582e-10, 0.23469041892380466, 3.4112403394282124e-10,
           8.432006697948113e-12, 0.12787798399269393, 9.671619751668736e-11,
           0.03254472622225042, 8.037294265403677e-10, 0.2526997892456903,
           2.3204571420906694e-10, 0.1329890132070401]
    riskt0 = 0.04095412710520921
    rett0 = 9.918096973651042e-5

    rm = TGRG(; settings = RMSettings(; scale = 2.0), owa = OWASettings(; approx = true))
    w1 = optimise!(portfolio, Trad(; rm = rm, kelly = NoKelly(), obj = obj))
    r1 = calc_risk(portfolio; type = :Trad, rm = rm)
    ret1 = dot(portfolio.mu, w1.weights)
    wt = [3.3301109472583185e-10, 0.075961449911884, 4.2523213691384906e-10,
          7.808100738453494e-10, 0.016156431516432097, 0.06416164555491817,
          2.753910854995426e-10, 0.06291545503958582, 2.5854477065134333e-10,
          2.517118252037656e-10, 0.23469199923390635, 7.383755124641802e-10,
          1.6800347698484544e-11, 0.12787714956541735, 2.0661048896877232e-10,
          0.03254557184006468, 1.7323396348860434e-9, 0.2527044694088422,
          4.979944105701631e-10, 0.1329858224121279]
    riskt = 0.0409541313496074
    rett = 9.918825863420109e-5
    @test isapprox(w1.weights, wt0, rtol = 5.0e-5)
    @test isapprox(r1, riskt0, rtol = 5.0e-7)
    @test isapprox(ret1, rett0, rtol = 0.0005)
    @test isapprox(w1.weights, wt, rtol = 5.0e-5)
    @test isapprox(r1, riskt, rtol = 5.0e-7)
    @test isapprox(ret1, rett, rtol = 5.0e-4)

    rm = [[TGRG(; owa = OWASettings(; approx = true)),
           TGRG(; owa = OWASettings(; approx = true))]]
    w2 = optimise!(portfolio, Trad(; rm = rm, kelly = NoKelly(), obj = obj))
    r2 = calc_risk(portfolio; type = :Trad, rm = rm[1][1])
    ret2 = dot(portfolio.mu, w2.weights)
    wt = [7.323822930638743e-11, 0.07596151470774425, 9.55960574020943e-11,
          1.8327629367515005e-10, 0.016156792637999056, 0.06416316949550527,
          5.927097359363786e-11, 0.06291582582113614, 5.530818965062581e-11,
          5.3852117894745915e-11, 0.23469089552199487, 1.6994549953094993e-10,
          2.735419948620365e-12, 0.12787816303390517, 4.2823718261181944e-11,
          0.03254522569030388, 4.0980036594507556e-10, 0.2526991516500957,
          1.1277598292365151e-10, 0.13298926018269275]
    riskt = 0.04095412686801266
    rett = 9.918158737157122e-5
    @test isapprox(w2.weights, wt0, rtol = 5.0e-5)
    @test isapprox(r2, riskt0, rtol = 1.0e-7)
    @test isapprox(ret2, rett0, rtol = 5.0e-4)
    @test isapprox(w2.weights, wt, rtol = 5.0e-5)
    @test isapprox(r2, riskt, rtol = 1.0e-7)
    @test isapprox(ret2, rett, rtol = 0.0005)

    obj = Sharpe(; rf = rf)
    wt0 = [7.488945193149933e-11, 2.3375420206676627e-10, 7.883173363915309e-11,
           1.2873379811554355e-10, 0.13199511266235572, 2.242050372262956e-11,
           3.2978647447839584e-11, 0.05078658196320988, 2.022756210960327e-9,
           1.8972428888719845e-10, 1.3751973663049892e-10, 5.861630633467697e-11,
           2.653461970910614e-11, 1.0851576920778706e-10, 2.944625350285356e-11,
           0.2040363490588814, 0.6131815993946066, 1.746246172291531e-10,
           3.53442317783505e-7, 1.5928264894663242e-10]
    riskt0 = 0.05762076056624533
    rett0 = 0.0016919468952059898

    rm = TGRG(; settings = RMSettings(; scale = 2.0), owa = OWASettings(; approx = true))
    w3 = optimise!(portfolio, Trad(; rm = rm, kelly = NoKelly(), obj = obj))
    r3 = calc_risk(portfolio; type = :Trad, rm = rm)
    ret3 = dot(portfolio.mu, w3.weights)
    wt = [3.197410530058853e-11, 1.0054386559292895e-10, 3.367520875992358e-11,
          5.524058741281638e-11, 0.13199371243430094, 9.298756281513519e-12,
          1.3860820762394248e-11, 0.05079826530871056, 8.666305133741033e-10,
          8.153906386704847e-11, 5.898285864674876e-11, 2.4934124186595022e-11,
          1.1074907971371647e-11, 4.6473367532648086e-11, 1.2335858659101685e-11,
          0.20403395887938955, 0.6131739136725372, 7.499168834495462e-11,
          1.4821515231405806e-7, 6.835373017357717e-11]
    riskt = 0.05762060785164221
    rett = 0.0016919427200940596
    @test isapprox(w3.weights, wt0, rtol = 5.0e-5)
    @test isapprox(r3, riskt0, rtol = 1.0e-5)
    @test isapprox(ret3, rett0, rtol = 1.0e-5)
    @test isapprox(w3.weights, wt, rtol = 5.0e-5)
    @test isapprox(r3, riskt, rtol = 1.0e-5)
    @test isapprox(ret3, rett, rtol = 1.0e-5)

    rm = [[TGRG(; owa = OWASettings(; approx = true)),
           TGRG(; owa = OWASettings(; approx = true))]]
    w4 = optimise!(portfolio, Trad(; rm = rm, kelly = NoKelly(), obj = obj))
    r4 = calc_risk(portfolio; type = :Trad, rm = rm[1][1])
    ret4 = dot(portfolio.mu, w4.weights)
    wt = [1.920952850023999e-11, 6.085892910191842e-11, 2.023832029467552e-11,
          3.3267688987184784e-11, 0.13199086919382808, 5.464986273964041e-12,
          8.22790112679589e-12, 0.05080126587360025, 5.320191454400441e-10,
          4.9252846530668396e-11, 3.567780455377237e-11, 1.495497893752875e-11,
          6.539192097272326e-12, 2.8048046083673616e-11, 7.29885141443001e-12,
          0.20403351443951037, 0.6131742431946052, 4.540399772874772e-11,
          1.0639061416564161e-7, 4.137965898695189e-11]
    riskt = 0.0576202987933357
    rett = 0.0016919342172824025
    @test isapprox(w4.weights, wt0, rtol = 5.0e-5)
    @test isapprox(r4, riskt0, rtol = 1.0e-5)
    @test isapprox(ret4, rett0, rtol = 1.0e-5)
    @test isapprox(w4.weights, wt, rtol = 5.0e-5)
    @test isapprox(r4, riskt, rtol = 5.0e-6)
    @test isapprox(ret4, rett, rtol = 5.0e-6)

    # Risk upper bound
    obj = MaxRet()
    rm = TGRG(; settings = RMSettings(; scale = 2.0), owa = OWASettings(; approx = true))
    rm.settings.ub = r1 * 1.001
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio; type = :Trad, rm = rm) <= r1 * 1.001

    rm = [[TGRG(; owa = OWASettings(; approx = true)),
           TGRG(; owa = OWASettings(; approx = true))]]
    rm[1][1].settings.ub = r2 * 1.001
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio; type = :Trad, rm = rm[1][1]) <= r2 * 1.001

    obj = Sharpe(; rf = rf)
    rm = TGRG(; settings = RMSettings(; scale = 2.0), owa = OWASettings(; approx = true))
    rm.settings.ub = r1 * 1.001
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio; type = :Trad, rm = rm) <= r1 * 1.001

    rm = [[TGRG(; owa = OWASettings(; approx = true)),
           TGRG(; owa = OWASettings(; approx = true))]]
    rm[1][1].settings.ub = r2 * 1.1
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio; type = :Trad, rm = rm[1][1]) <= r2 * 1.1

    # Ret lower bound
    obj = MinRisk()
    rm = TGRG(; settings = RMSettings(; scale = 2.0), owa = OWASettings(; approx = true))
    rm.settings.ub = Inf
    portfolio.mu_l = ret1
    w5 = optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test dot(portfolio.mu, w5.weights) >= ret1

    rm = [[TGRG(; owa = OWASettings(; approx = true)),
           TGRG(; owa = OWASettings(; approx = true))]]
    portfolio.mu_l = ret2
    w6 = optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test dot(portfolio.mu, w6.weights) >= ret2

    obj = Sharpe(; rf = rf)
    rm = TGRG(; settings = RMSettings(; scale = 2.0), owa = OWASettings(; approx = true))
    portfolio.mu_l = ret1
    w7 = optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test dot(portfolio.mu, w7.weights) >= ret1

    rm = [[TGRG(; owa = OWASettings(; approx = true)),
           TGRG(; owa = OWASettings(; approx = true))]]
    portfolio.mu_l = ret2
    w8 = optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test dot(portfolio.mu, w8.weights) >= ret2

    portfolio.mu_l = Inf
    obj = MinRisk()
    wt0 = [1.1206142630614112e-11, 0.09772215128420475, 3.5338858649323828e-12,
           3.307433371895369e-11, 0.017417781863929577, 0.041201102000678634,
           1.6220089767187403e-11, 0.060553118743468366, 1.7795758713115547e-11,
           1.5483766349422575e-11, 0.22830001756787208, 2.6382663454855723e-11,
           3.59484391773399e-11, 0.13930594414393907, 2.0476463882653826e-11,
           0.0362260148933746, 9.71096463830921e-11, 0.24663479935480356,
           1.4791298973742443e-12, 0.13263906986901897]
    riskt0 = 0.040929059924343876
    rett0 = 0.00020307377724072516
    wt1 = [1.0009153496459239e-10, 9.749900346647964e-11, 5.874551709857877e-11,
           5.5624273682545086e-11, 0.1063233767596915, 0.13310564164072994,
           5.386813608798474e-11, 0.07245409142850337, 3.3361351110701744e-10,
           8.12579663572646e-12, 0.1820105683187836, 1.602733364456541e-11,
           2.447576840031422e-11, 0.20383023498792277, 1.1981432205433173e-11,
           0.03264070951504424, 0.018170189957485795, 0.15729142359636483,
           0.02869810997957648, 0.06547565305584528]
    riskt1 = 0.02105727354969537
    rett1 = -1.5010409711686289e-5

    rm = [[TGRG(; owa = OWASettings(; approx = true)),
           TGRG(; alpha = 0.75, owa = OWASettings(; approx = true))]]
    w9 = optimise!(portfolio, Trad(; rm = rm, kelly = NoKelly(), obj = obj))
    r9 = calc_risk(portfolio; type = :Trad, rm = rm[1][1])
    ret9 = dot(portfolio.mu, w9.weights)
    wt = [5.143525613322085e-7, 0.04231423029651418, 4.041239877448205e-7,
          7.063506210198864e-7, 0.08226090060742469, 0.11855341397775944,
          3.32584854280333e-7, 0.06178037457301832, 4.222684064214522e-7,
          2.3060682983166463e-7, 0.2080649272519033, 1.9298287401659178e-7,
          1.2518398945064707e-8, 0.19108573855742744, 1.2714920670545827e-7,
          0.015234659398129147, 7.650188267617955e-6, 0.19157811812378625,
          8.321957322172731e-7, 0.08911621189229724]
    riskt = 0.041128724239980215
    rett = -5.9772444643978466e-5
    @test !isapprox(w9.weights, wt0)
    @test !isapprox(r9, riskt0)
    @test !isapprox(ret9, rett0)
    @test !isapprox(w9.weights, wt1)
    @test !isapprox(r9, riskt1)
    @test !isapprox(ret9, rett1)
    @test isapprox(w9.weights, wt, rtol = 0.005)
    @test isapprox(r9, riskt, rtol = 5.0e-5)
    @test isapprox(ret9, rett, rtol = 0.05)

    rm = [[TGRG(; owa = OWASettings(; approx = true)),
           TGRG(; owa = OWASettings(; approx = false))]]
    w10 = optimise!(portfolio, Trad(; rm = rm, kelly = NoKelly(), obj = obj))
    @test all(value.(portfolio.model[:rtga][:, 1]) .== 0)
    @test all(value.(portfolio.model[:rtga][:, 2]) .!= 0)
    @test value(portfolio.model[:rltg_t][1]) != 0
    @test value(portfolio.model[:rltg_t][2]) == 0

    rm = [[TGRG(; owa = OWASettings(; approx = false)),
           TGRG(; owa = OWASettings(; approx = true))]]
    w11 = optimise!(portfolio, Trad(; rm = rm, kelly = NoKelly(), obj = obj))
    @test all(value.(portfolio.model[:rtga][:, 1]) .!= 0)
    @test all(value.(portfolio.model[:rtga][:, 2]) .== 0)
    @test value(portfolio.model[:rltg_t][1]) == 0
    @test value(portfolio.model[:rltg_t][2]) != 0
end

@testset "OWA vec" begin
    portfolio = OmniPortfolio(; prices = prices[(end - 200):end],
                              solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                               :check_sol => (allow_local = true,
                                                                              allow_almost = true),
                                                               :params => Dict("verbose" => false,
                                                                               "max_step_fraction" => 0.75,
                                                                               "max_iter" => 200,
                                                                               "equilibrate_max_iter" => 20))))
    asset_statistics!(portfolio)

    obj = MinRisk()
    wt0 = [5.969325155308689e-13, 0.19890625725641747, 5.01196330972591e-12,
           0.05490289350205719, 4.555898520654742e-11, 1.0828830360388467e-12,
           3.3254529493732914e-12, 0.10667398877393132, 1.0523685498886423e-12,
           1.8363867378603495e-11, 0.0486780391076569, 0.09089397416963123,
           3.698655012660736e-12, 8.901903375667794e-12, 2.0443054631623374e-12,
           0.13418199806574263, 6.531476985844397e-12, 0.13829581359726595,
           1.724738947709949e-12, 0.22746703542940375]
    riskt0 = 0.02356383470533441
    rett0 = 0.0005937393209710076

    rm = OWA(; w = owa_tg(200), settings = RMSettings(; scale = 2.0),
             owa = OWASettings(; approx = false))
    w1 = optimise!(portfolio, Trad(; rm = rm, kelly = NoKelly(), obj = obj))
    r1 = calc_risk(portfolio; type = :Trad, rm = rm)
    ret1 = dot(portfolio.mu, w1.weights)
    wt = [9.117129296000855e-13, 0.1989062604779038, 5.091046781890198e-12,
          0.054902892230445206, 4.656142975466791e-11, 1.5118793197837656e-12,
          3.8540198560422116e-12, 0.10667398953978909, 1.3175608677583888e-12,
          1.9772109268754522e-11, 0.04867803522723395, 0.09089397449417429,
          4.256470959584207e-12, 9.23111680963713e-12, 1.7711791864713846e-12,
          0.1341819980026756, 6.777051511565845e-12, 0.13829581436956526,
          1.6817643269228358e-12, 0.22746703555547537]
    riskt = 0.023563834706007303
    rett = 0.0005937393233563302
    @test isapprox(w1.weights, wt0)
    @test isapprox(r1, riskt0)
    @test isapprox(ret1, rett0)
    @test isapprox(w1.weights, wt)
    @test isapprox(r1, riskt)
    @test isapprox(ret1, rett)

    rm = [[OWA(; w = owa_tg(200), owa = OWASettings(; approx = false)),
           OWA(; w = owa_tg(200), owa = OWASettings(; approx = false))]]
    w2 = optimise!(portfolio, Trad(; rm = rm, kelly = NoKelly(), obj = obj))
    r2 = calc_risk(portfolio; type = :Trad, rm = rm[1][1])
    ret2 = dot(portfolio.mu, w2.weights)
    wt = [4.641528551675654e-12, 0.19890627715965065, 2.6164624191846925e-13,
          0.054902895298161516, 4.196501505287773e-11, 5.113631855168327e-12,
          6.787750163857296e-12, 0.106673990078965, 5.062864104596882e-12,
          1.0352546837427235e-11, 0.04867801775229248, 0.09089397404658377,
          7.108498848500437e-12, 2.7425743138904577e-12, 2.6840570215544006e-12,
          0.13418199203670245, 1.0078682496776611e-12, 0.13829581534257337,
          2.844310734614091e-12, 0.2274670381944986]
    riskt = 0.023563834710486266
    rett = 0.0005937393357118465
    @test isapprox(w2.weights, wt0, rtol = 1.0e-7)
    @test isapprox(r2, riskt0)
    @test isapprox(ret2, rett0, rtol = 5.0e-8)
    @test isapprox(w2.weights, wt)
    @test isapprox(r2, riskt)
    @test isapprox(ret2, rett)

    obj = Sharpe(; rf = rf)
    wt0 = [1.0342189964374973e-11, 1.400649279934334e-11, 1.0025902458601371e-11,
           6.108757972397652e-12, 0.35723529048659247, 1.2069564550657953e-11,
           1.2830347528094887e-11, 1.1781378314488614e-12, 6.5343202013864566e-12,
           2.895398409002917e-12, 9.118697983089066e-12, 1.0966314618191202e-11,
           1.3147762207425575e-11, 9.611939363025545e-12, 1.263250173104243e-11,
           0.21190800224274572, 0.4308567071123932, 7.712262968799656e-12,
           1.2412631429194858e-11, 6.675313779730199e-12]
    riskt0 = 0.03208927225110264
    rett0 = 0.0017686555804583674

    rm = OWA(; w = owa_tg(200), settings = RMSettings(; scale = 2.0),
             owa = OWASettings(; approx = false))
    w3 = optimise!(portfolio, Trad(; rm = rm, kelly = NoKelly(), obj = obj))
    r3 = calc_risk(portfolio; type = :Trad, rm = rm)
    ret3 = dot(portfolio.mu, w3.weights)
    wt = [1.7628494451185628e-11, 6.309210814522966e-12, 1.73340851042146e-11,
          1.2094716457413075e-11, 0.35723528912382363, 2.055184073410796e-11,
          2.047384063050713e-11, 5.9981511062419914e-12, 4.013121626457183e-12,
          1.0610718968359735e-11, 1.6004699009244607e-11, 1.832523859937056e-11,
          2.0738137429563127e-11, 1.6615403276744508e-11, 2.021575771586549e-11,
          0.21190800221923103, 0.4308567084114284, 1.4469393378858674e-11,
          1.1144282063101474e-11, 1.299013644991039e-11]
    riskt = 0.03208927224682404
    rett = 0.001768655580040797
    @test isapprox(w3.weights, wt0)
    @test isapprox(r3, riskt0)
    @test isapprox(ret3, rett0)
    @test isapprox(w3.weights, wt)
    @test isapprox(r3, riskt)
    @test isapprox(ret3, rett)

    rm = [[OWA(; w = owa_tg(200), owa = OWASettings(; approx = false)),
           OWA(; w = owa_tg(200), owa = OWASettings(; approx = false))]]
    w4 = optimise!(portfolio, Trad(; rm = rm, kelly = NoKelly(), obj = obj))
    r4 = calc_risk(portfolio; type = :Trad, rm = rm[1][1])
    ret4 = dot(portfolio.mu, w4.weights)
    wt = [9.27189739926205e-12, 2.41888502907255e-12, 9.261436640083589e-12,
          6.688973543852134e-12, 0.3572352895737421, 9.602174585541168e-12,
          9.742071453613786e-12, 5.758080242369661e-12, 3.6338662109179993e-12,
          7.021702688461076e-12, 8.794481459242355e-12, 9.356771227591303e-12,
          9.630742521058502e-12, 8.932139581595437e-12, 9.675622394371964e-12,
          0.21190800382409697, 0.4308567064744369, 8.521841437957333e-12,
          1.4535712298415591e-12, 7.959924261484958e-12]
    riskt = 0.03208927224100061
    rett = 0.001768655580065531
    @test isapprox(w4.weights, wt0)
    @test isapprox(r4, riskt0)
    @test isapprox(ret4, rett0)
    @test isapprox(w4.weights, wt)
    @test isapprox(r4, riskt)
    @test isapprox(ret4, rett)

    # Risk upper bound
    obj = MaxRet()
    rm = OWA(; w = owa_tg(200), settings = RMSettings(; scale = 2.0),
             owa = OWASettings(; approx = false))
    rm.settings.ub = r1
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test abs(calc_risk(portfolio; type = :Trad, rm = rm) - r1) < 1e-8

    rm = [[OWA(; w = owa_tg(200), owa = OWASettings(; approx = false)),
           OWA(; w = owa_tg(200), owa = OWASettings(; approx = false))]]
    rm[1][1].settings.ub = r2
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test abs(calc_risk(portfolio; type = :Trad, rm = rm[1][1]) - r2) < 1e-8

    obj = Sharpe(; rf = rf)
    rm = OWA(; w = owa_tg(200), settings = RMSettings(; scale = 2.0),
             owa = OWASettings(; approx = false))
    rm.settings.ub = r1
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio; type = :Trad, rm = rm) <= r1 ||
          abs(calc_risk(portfolio; type = :Trad, rm = rm) - r1) < 1e-9

    rm = [[OWA(; w = owa_tg(200), owa = OWASettings(; approx = false)),
           OWA(; w = owa_tg(200), owa = OWASettings(; approx = false))]]
    rm[1][1].settings.ub = r2
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio; type = :Trad, rm = rm[1][1]) <= r2 ||
          abs(calc_risk(portfolio; type = :Trad, rm = rm[1][1]) - r2) < 5e-6

    # Ret lower bound
    obj = MinRisk()
    rm = OWA(; w = owa_tg(200), settings = RMSettings(; scale = 2.0),
             owa = OWASettings(; approx = false))
    rm.settings.ub = Inf
    portfolio.mu_l = ret1
    w5 = optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test dot(portfolio.mu, w5.weights) >= ret1

    rm = [[OWA(; w = owa_tg(200), owa = OWASettings(; approx = false)),
           OWA(; w = owa_tg(200), owa = OWASettings(; approx = false))]]
    portfolio.mu_l = ret2
    w6 = optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test dot(portfolio.mu, w6.weights) >= ret2

    obj = Sharpe(; rf = rf)
    rm = OWA(; w = owa_tg(200), settings = RMSettings(; scale = 2.0),
             owa = OWASettings(; approx = false))
    portfolio.mu_l = ret1
    w7 = optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test dot(portfolio.mu, w7.weights) >= ret1

    rm = [[OWA(; w = owa_tg(200), owa = OWASettings(; approx = false)),
           OWA(; w = owa_tg(200), owa = OWASettings(; approx = false))]]
    portfolio.mu_l = ret2
    w8 = optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test dot(portfolio.mu, w8.weights) >= ret2

    portfolio.mu_l = Inf
    obj = MinRisk()
    wt0 = [5.969325155308689e-13, 0.19890625725641747, 5.01196330972591e-12,
           0.05490289350205719, 4.555898520654742e-11, 1.0828830360388467e-12,
           3.3254529493732914e-12, 0.10667398877393132, 1.0523685498886423e-12,
           1.8363867378603495e-11, 0.0486780391076569, 0.09089397416963123,
           3.698655012660736e-12, 8.901903375667794e-12, 2.0443054631623374e-12,
           0.13418199806574263, 6.531476985844397e-12, 0.13829581359726595,
           1.724738947709949e-12, 0.22746703542940375]
    riskt0 = 0.02356383470533441
    rett0 = 0.0005937393209710076
    wt1 = [1.2836541099300374e-10, 7.632300965664097e-10, 1.919887858998074e-10,
           4.801955740158814e-10, 0.0982889108245419, 5.651545334295017e-11,
           1.6576927251449667e-11, 0.07900741935773047, 6.319231416335898e-10,
           8.34058889694369e-8, 0.043496746354597604, 2.57787733628572e-11,
           7.687212211788456e-11, 0.11246278997924469, 5.4971523683089116e-11,
           0.03112348236478457, 0.1477185644358598, 0.17254181002118, 0.158714558264814,
           0.1566456325649403]
    riskt1 = 0.0052910845613359445
    rett1 = 0.0008709132492808145

    rm = [[OWA(; w = owa_tg(200), owa = OWASettings(; approx = false)),
           OWA(; w = owa_tg(200; alpha = 0.75), owa = OWASettings(; approx = false))]]
    w9 = optimise!(portfolio, Trad(; rm = rm, kelly = NoKelly(), obj = obj))
    r9 = calc_risk(portfolio; type = :Trad, rm = rm[1][1])
    ret9 = dot(portfolio.mu, w9.weights)
    wt = [2.0365478576275784e-12, 0.26110976994697394, 3.4935250075823384e-11,
          0.014232056343286167, 0.025589079693026068, 4.6955664317896684e-12,
          1.280870684858824e-11, 0.09709844547134337, 5.464466441696193e-13,
          1.1641073623944162e-10, 0.08852689155201673, 0.022082626782455907,
          1.591050398876822e-11, 6.63033240411461e-10, 2.00094227838181e-12,
          0.0702929792386146, 1.5518347340153248e-10, 0.23109956724678077,
          2.3551644968730975e-11, 0.18996858269438938]
    riskt = 0.02402876910565921
    rett = 0.0006220186938847865
    @test !isapprox(w9.weights, wt0)
    @test !isapprox(r9, riskt0)
    @test !isapprox(ret9, rett0)
    @test !isapprox(w9.weights, wt1)
    @test !isapprox(r9, riskt1)
    @test !isapprox(ret9, rett1)
    @test isapprox(w9.weights, wt, rtol = 0.0005)
    @test isapprox(r9, riskt, rtol = 5.0e-6)
    @test isapprox(ret9, rett, rtol = 5.0e-5)

    obj = MinRisk()
    wt0 = [1.424802287200348e-10, 0.24657565369133722, 3.660767298307013e-10,
           0.041101497481424845, 1.1765742472564645e-9, 1.3206711379272908e-10,
           4.948146901460501e-11, 0.11651694980250114, 1.27384858611194e-10,
           7.083586776439093e-10, 2.7784344366481497e-8, 0.09732942313209282,
           3.6499869966418024e-11, 4.3139455041429935e-10, 2.246994554915304e-10,
           0.14662262356233058, 3.7795902929060657e-10, 0.12694138139146063,
           2.229837567112617e-10, 0.22491243915854847]
    riskt0 = 0.023582366401225324
    rett0 = 0.000643224255466324

    rm = OWA(; w = owa_tg(200), settings = RMSettings(; scale = 2.0),
             owa = OWASettings(; approx = true))
    w1 = optimise!(portfolio, Trad(; rm = rm, kelly = NoKelly(), obj = obj))
    r1 = calc_risk(portfolio; type = :Trad, rm = rm)
    ret1 = dot(portfolio.mu, w1.weights)
    wt = [3.9936340457424127e-10, 0.24657622800276421, 1.0029426420816195e-9,
          0.04109935113614855, 3.213182208801116e-9, 3.696223300650158e-10,
          1.3886189785425067e-10, 0.11651467765066632, 3.541515759525849e-10,
          1.9220249209401027e-9, 1.3472001756117787e-7, 0.09732827863666377,
          1.0160138214212973e-10, 1.255233136605034e-9, 6.484292746108698e-10,
          0.14662114332298815, 1.0783577922601826e-9, 0.12694764616541818,
          6.249160198927854e-10, 0.2249125292566467]
    riskt = 0.02358236418515349
    rett = 0.0006432212632505017
    @test isapprox(w1.weights, wt0, rtol = 5.0e-5)
    @test isapprox(r1, riskt0, rtol = 1.0e-7)
    @test isapprox(ret1, rett0, rtol = 5.0e-6)
    @test isapprox(w1.weights, wt, rtol = 5.0e-5)
    @test isapprox(r1, riskt, rtol = 5.0e-7)
    @test isapprox(ret1, rett, rtol = 1.0e-5)

    rm = [[OWA(; w = owa_tg(200), owa = OWASettings(; approx = true)),
           OWA(; w = owa_tg(200), owa = OWASettings(; approx = true))]]
    w2 = optimise!(portfolio, Trad(; rm = rm, kelly = NoKelly(), obj = obj))
    r2 = calc_risk(portfolio; type = :Trad, rm = rm[1][1])
    ret2 = dot(portfolio.mu, w2.weights)
    wt = [8.747751866852216e-11, 0.24657591640394882, 2.2999033865361832e-10,
          0.04110030975228419, 7.40840581578713e-10, 8.022226425110321e-11,
          2.6020918127856197e-11, 0.11651563958306337, 7.73467651675735e-11,
          4.5063979465362114e-10, 4.875300149443852e-8, 0.0973286614947302,
          1.7302556481268366e-11, 2.905426791825603e-10, 1.447990828774433e-10,
          0.14662146100473505, 2.4813093060591616e-10, 0.12694473083454902,
          1.4123632339801168e-10, 0.22491322963913815]
    riskt = 0.023582366176196835
    rett = 0.0006432220772657842
    @test isapprox(w2.weights, wt0, rtol = 5.0e-5)
    @test isapprox(r2, riskt0, rtol = 1.0e-7)
    @test isapprox(ret2, rett0, rtol = 5.0e-6)
    @test isapprox(w2.weights, wt, rtol = 5.0e-5)
    @test isapprox(r2, riskt, rtol = 1.0e-7)
    @test isapprox(ret2, rett, rtol = 5.0e-6)

    obj = Sharpe(; rf = rf)
    wt0 = [2.465085712389692e-10, 1.7591592054961841e-9, 2.6079839244395587e-10,
           5.190986820248767e-10, 0.3290475106649799, 7.765809158263677e-11,
           1.0087407386198554e-10, 1.0105472171462993e-9, 1.5219079686966533e-9,
           6.719673558205139e-10, 3.7631629567205446e-10, 2.3450024422381644e-10,
           9.619358604737496e-11, 3.325560088347231e-10, 1.1281942009082258e-10,
           0.20499762407557243, 0.46595485476474174, 4.4765819636375454e-10,
           2.177943962387867e-9, 5.481986913612393e-10]
    riskt0 = 0.03204366080645988
    rett0 = 0.00176571347903363

    rm = OWA(; w = owa_tg(200), settings = RMSettings(; scale = 2.0),
             owa = OWASettings(; approx = true))
    w3 = optimise!(portfolio, Trad(; rm = rm, kelly = NoKelly(), obj = obj))
    r3 = calc_risk(portfolio; type = :Trad, rm = rm)
    ret3 = dot(portfolio.mu, w3.weights)
    wt = [1.30958811238442e-10, 9.393064643496553e-10, 1.3859460639303259e-10,
          2.7641929970970515e-10, 0.32908074699916545, 4.0831848300802844e-11,
          5.321357998359013e-11, 5.390945651126132e-10, 8.118459844241153e-10,
          3.581643614310909e-10, 2.003119941264806e-10, 1.245657705529936e-10,
          5.073302868223789e-11, 1.7689645810943782e-10, 5.959970853453841e-11,
          0.2049920451953837, 0.46592720221227896, 2.383524310980431e-10,
          1.162120177366155e-9, 2.9216288025431493e-10]
    riskt = 0.032043769959883034
    rett = 0.0017657196603620788
    @test isapprox(w3.weights, wt0, rtol = 0.0001)
    @test isapprox(r3, riskt0, rtol = 5.0e-6)
    @test isapprox(ret3, rett0, rtol = 5.0e-6)
    @test isapprox(w3.weights, wt, rtol = 5.0e-5)
    @test isapprox(r3, riskt, rtol = 5.0e-6)
    @test isapprox(ret3, rett, rtol = 5.0e-6)

    rm = [[OWA(; w = owa_tg(200), owa = OWASettings(; approx = true)),
           OWA(; w = owa_tg(200), owa = OWASettings(; approx = true))]]
    w4 = optimise!(portfolio, Trad(; rm = rm, kelly = NoKelly(), obj = obj))
    r4 = calc_risk(portfolio; type = :Trad, rm = rm[1][1])
    ret4 = dot(portfolio.mu, w4.weights)
    wt = [7.969157789262045e-11, 5.766162636150547e-10, 8.43578632654925e-11,
          1.6869812629460985e-10, 0.3290723319262457, 2.457968693355534e-11,
          3.214627364480214e-11, 3.2981665272915827e-10, 4.961175585483804e-10,
          2.1886744520342848e-10, 1.2227967428248629e-10, 7.58694703847239e-11,
          3.063754370941499e-11, 1.0792869660473871e-10, 3.605814395871514e-11,
          0.20499368159023673, 0.4659339830651274, 1.4554016002709475e-10,
          7.104314686454891e-10, 1.7875349405006732e-10]
    riskt = 0.03204374129311252
    rett = 0.0017657180543370445
    @test isapprox(w4.weights, wt0, rtol = 0.0001)
    @test isapprox(r4, riskt0, rtol = 5.0e-6)
    @test isapprox(ret4, rett0, rtol = 5.0e-6)
    @test isapprox(w4.weights, wt, rtol = 5.0e-5)
    @test isapprox(r4, riskt, rtol = 5.0e-6)
    @test isapprox(ret4, rett, rtol = 5.0e-6)

    # Risk upper bound
    obj = MaxRet()
    rm = OWA(; w = owa_tg(200), settings = RMSettings(; scale = 2.0),
             owa = OWASettings(; approx = true))
    rm.settings.ub = r1 * 1.001
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio; type = :Trad, rm = rm) <= r1 * 1.001

    rm = [[OWA(; w = owa_tg(200), owa = OWASettings(; approx = true)),
           OWA(; w = owa_tg(200), owa = OWASettings(; approx = true))]]
    rm[1][1].settings.ub = r2 * 1.001
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio; type = :Trad, rm = rm[1][1]) <= r2 * 1.001

    obj = Sharpe(; rf = rf)
    rm = OWA(; w = owa_tg(200), settings = RMSettings(; scale = 2.0),
             owa = OWASettings(; approx = true))
    rm.settings.ub = r1 * 1.001
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test calc_risk(portfolio; type = :Trad, rm = rm) <= r1 * 1.001

    rm = [[OWA(; w = owa_tg(200), owa = OWASettings(; approx = true)),
           OWA(; w = owa_tg(200), owa = OWASettings(; approx = true))]]
    rm[1][1].settings.ub = r2 * 1.01
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    if !Sys.isapple()
        @test calc_risk(portfolio; type = :Trad, rm = rm[1][1]) <= r2 * 1.01
    end

    # Ret lower bound
    obj = MinRisk()
    rm = OWA(; w = owa_tg(200), settings = RMSettings(; scale = 2.0),
             owa = OWASettings(; approx = true))
    rm.settings.ub = Inf
    portfolio.mu_l = ret1
    w5 = optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test dot(portfolio.mu, w5.weights) >= ret1

    rm = [[OWA(; w = owa_tg(200), owa = OWASettings(; approx = true)),
           OWA(; w = owa_tg(200), owa = OWASettings(; approx = true))]]
    portfolio.mu_l = ret2
    w6 = optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test dot(portfolio.mu, w6.weights) >= ret2

    obj = Sharpe(; rf = rf)
    rm = OWA(; w = owa_tg(200), settings = RMSettings(; scale = 2.0),
             owa = OWASettings(; approx = true))
    portfolio.mu_l = ret1
    w7 = optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test dot(portfolio.mu, w7.weights) >= ret1

    rm = [[OWA(; w = owa_tg(200), owa = OWASettings(; approx = true)),
           OWA(; w = owa_tg(200), owa = OWASettings(; approx = true))]]
    portfolio.mu_l = ret2
    w8 = optimise!(portfolio, Trad(; rm = rm, obj = obj))
    @test dot(portfolio.mu, w8.weights) >= ret2

    portfolio.mu_l = Inf
    obj = MinRisk()
    wt0 = [1.424802287200348e-10, 0.24657565369133722, 3.660767298307013e-10,
           0.041101497481424845, 1.1765742472564645e-9, 1.3206711379272908e-10,
           4.948146901460501e-11, 0.11651694980250114, 1.27384858611194e-10,
           7.083586776439093e-10, 2.7784344366481497e-8, 0.09732942313209282,
           3.6499869966418024e-11, 4.3139455041429935e-10, 2.246994554915304e-10,
           0.14662262356233058, 3.7795902929060657e-10, 0.12694138139146063,
           2.229837567112617e-10, 0.22491243915854847]
    riskt0 = 0.023582366401225324
    rett0 = 0.000643224255466324
    wt1 = [7.300441198975879e-11, 3.733446529544342e-10, 1.0543513189637169e-10,
           2.3029529249627563e-10, 0.09845917917186836, 3.71261966988816e-11,
           1.86649050658783e-11, 0.0769519240514099, 3.0699196883209783e-10,
           0.00018304528861999298, 0.04462511459102036, 2.307293409612758e-11,
           4.744860664868727e-11, 0.11542959315117003, 3.730300447537313e-11,
           0.03289876924459179, 0.14723423110092668, 0.16996234638096502,
           0.15838244526051007, 0.15587335050623066]
    riskt1 = 0.005291194452999399
    rett1 = 0.0008696210533165451

    rm = [[OWA(; w = owa_tg(200), owa = OWASettings(; approx = true)),
           OWA(; w = owa_tg(200; alpha = 0.75), owa = OWASettings(; approx = true))]]
    w9 = optimise!(portfolio, Trad(; rm = rm, kelly = NoKelly(), obj = obj))
    r9 = calc_risk(portfolio; type = :Trad, rm = rm[1][1])
    ret9 = dot(portfolio.mu, w9.weights)
    wt = [7.50920171037976e-11, 0.2609567509058036, 1.8054952886985706e-10,
          0.020467743861486545, 0.008657062382432874, 5.3521376788735344e-11,
          2.4094798872980635e-11, 0.0937608422302937, 6.760517792773822e-11,
          4.262590556341936e-10, 0.08288409866568487, 0.024366883942363048,
          1.3548303594176315e-11, 2.7453073588686673e-9, 7.47791397994732e-11,
          0.07072019989759804, 7.695475396700068e-10, 0.24384420472955967,
          1.5820130362962118e-10, 0.19434220879627206]
    riskt = 0.024020116307912565
    rett = 0.0005994819474003006
    @test !isapprox(w9.weights, wt0)
    @test !isapprox(r9, riskt0)
    @test !isapprox(ret9, rett0)
    @test !isapprox(w9.weights, wt1)
    @test !isapprox(r9, riskt1)
    @test !isapprox(ret9, rett1)
    @test isapprox(w9.weights, wt, rtol = 5.0e-5)
    @test isapprox(r9, riskt, rtol = 1.0e-6)
    @test isapprox(ret9, rett, rtol = 5.0e-6)

    wt0 = [5.969325155308689e-13, 0.19890625725641747, 5.01196330972591e-12,
           0.05490289350205719, 4.555898520654742e-11, 1.0828830360388467e-12,
           3.3254529493732914e-12, 0.10667398877393132, 1.0523685498886423e-12,
           1.8363867378603495e-11, 0.0486780391076569, 0.09089397416963123,
           3.698655012660736e-12, 8.901903375667794e-12, 2.0443054631623374e-12,
           0.13418199806574263, 6.531476985844397e-12, 0.13829581359726595,
           1.724738947709949e-12, 0.22746703542940375]
    riskt0 = 0.02356383470533441
    rett0 = 0.0005937393209710076
    wt1 = [1.424802287200348e-10, 0.24657565369133722, 3.660767298307013e-10,
           0.041101497481424845, 1.1765742472564645e-9, 1.3206711379272908e-10,
           4.948146901460501e-11, 0.11651694980250114, 1.27384858611194e-10,
           7.083586776439093e-10, 2.7784344366481497e-8, 0.09732942313209282,
           3.6499869966418024e-11, 4.3139455041429935e-10, 2.246994554915304e-10,
           0.14662262356233058, 3.7795902929060657e-10, 0.12694138139146063,
           2.229837567112617e-10, 0.22491243915854847]
    riskt1 = 0.023582366401225324
    rett1 = 0.000643224255466324

    rm = [[OWA(; settings = RMSettings(; scale = 10), owa = OWASettings(; approx = true)),
           OWA(; settings = RMSettings(; scale = 10), owa = OWASettings(; approx = false))]]
    w10 = optimise!(portfolio, Trad(; rm = rm, kelly = NoKelly(), obj = obj))
    @test all(value.(portfolio.model[:owa_a][:, 1]) .== 0)
    @test all(value.(portfolio.model[:owa_a][:, 2]) .!= 0)
    @test value(portfolio.model[:owa_t][1]) != 0
    @test value(portfolio.model[:owa_t][2]) == 0

    rm = [[OWA(; settings = RMSettings(; scale = 10), owa = OWASettings(; approx = false)),
           OWA(; settings = RMSettings(; scale = 10), owa = OWASettings(; approx = true))]]
    w11 = optimise!(portfolio, Trad(; rm = rm, kelly = NoKelly(), obj = obj))
    @test all(value.(portfolio.model[:owa_a][:, 1]) .!= 0)
    @test all(value.(portfolio.model[:owa_a][:, 2]) .== 0)
    @test value(portfolio.model[:owa_t][1]) == 0
    @test value(portfolio.model[:owa_t][2]) != 0
end

@testset "Skew vec" begin
    portfolio = OmniPortfolio(; prices = prices,
                              solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                               :check_sol => (allow_local = true,
                                                                              allow_almost = true),
                                                               :params => Dict("verbose" => false,
                                                                               "max_step_fraction" => 0.75))))
    asset_statistics!(portfolio)
    rm = Skew()

    obj = MinRisk()
    w1 = optimise!(portfolio, Trad(; rm = [[rm]], kelly = NoKelly(), obj = obj))
    r1 = calc_risk(portfolio; type = :Trad, rm = rm)
    ret1 = dot(portfolio.mu, w1.weights)
    wt = [2.409023468372493e-6, 0.17210378292297784, 5.103256374511177e-7,
          4.2722770517309445e-7, 3.5848058134265105e-6, 6.972041664934061e-7,
          3.2915769657012085e-7, 0.1415418122674741, 4.3289050477603765e-7,
          4.5431144777227054e-7, 0.07897482123611543, 0.023295191901219474,
          2.0444083999934734e-6, 3.3398275530097316e-6, 0.1761574592680367,
          0.042496745295449355, 3.003590887382274e-6, 0.23119283730811144,
          6.400097708092224e-7, 0.1342194770175644]
    riskt = 0.0016553752647584506
    rett = 0.0001952238162305396
    @test isapprox(w1.weights, wt)
    @test isapprox(r1, riskt)
    @test isapprox(ret1, rett)

    rm.V = 2 * portfolio.V
    w2 = optimise!(portfolio, Trad(; rm = [[rm]], kelly = NoKelly(), obj = obj))
    @test isapprox(w2.weights, wt, rtol = 5.0e-5)
end

@testset "SSkew vec" begin
    portfolio = OmniPortfolio(; prices = prices,
                              solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                               :check_sol => (allow_local = true,
                                                                              allow_almost = true),
                                                               :params => Dict("verbose" => false,
                                                                               "max_step_fraction" => 0.75))))
    asset_statistics!(portfolio)
    rm = SSkew()

    obj = MinRisk()
    w1 = optimise!(portfolio, Trad(; rm = [[rm]], kelly = NoKelly(), obj = obj))
    r1 = calc_risk(portfolio; type = :Trad, rm = rm)
    ret1 = dot(portfolio.mu, w1.weights)
    wt = [7.610819221245223e-7, 2.0842051720782308e-5, 6.71139244607462e-7,
          8.839948442444747e-7, 2.1807270401133766e-6, 1.0842040956915447e-6,
          2.7830339110192037e-7, 0.1280975813408656, 4.697915971310934e-7,
          7.433008049916792e-7, 0.4996629306075607, 1.043320338349008e-6,
          4.5083144725397534e-7, 0.026586616541717394, 2.6481651179988687e-5,
          0.013510133781810273, 3.2563765357090695e-6, 0.21130849297469684,
          6.622336342182562e-7, 0.1207744357455529]
    riskt = 0.0033523757385970935
    rett = 0.0003452673005217105
    @test isapprox(w1.weights, wt)
    @test isapprox(r1, riskt)
    @test isapprox(ret1, rett)

    rm.V = 2 * portfolio.SV
    w2 = optimise!(portfolio, Trad(; rm = [[rm]], kelly = NoKelly(), obj = obj))
    @test isapprox(w2.weights, wt, rtol = 0.0001)
end

@testset "Add Skew and SSkew to SD" begin
    portfolio = OmniPortfolio(; prices = prices,
                              solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                               :check_sol => (allow_local = true,
                                                                              allow_almost = true),
                                                               :params => Dict("verbose" => false))))
    asset_statistics!(portfolio)

    rm = Skew(; settings = RMSettings(; scale = 1.0))
    obj = MinRisk()
    w1 = optimise!(portfolio, Trad(; rm = rm, kelly = NoKelly(), obj = obj))
    rm.settings.scale = 0.99
    w2 = optimise!(portfolio, Trad(; rm = rm, kelly = NoKelly(), obj = obj))
    @test isapprox(w1.weights, w2.weights, rtol = 5e-6)

    rm = SSkew(; settings = RMSettings(; scale = 1.0))
    obj = MinRisk()
    w1 = optimise!(portfolio, Trad(; rm = rm, kelly = NoKelly(), obj = obj))
    rm.settings.scale = 0.99
    w2 = optimise!(portfolio, Trad(; rm = rm, kelly = NoKelly(), obj = obj))
    @test isapprox(w1.weights, w2.weights, rtol = 1e-4)
end

@testset "Get value of z" begin
    portfolio = OmniPortfolio(; prices = prices,
                              solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                               :check_sol => (allow_local = true,
                                                                              allow_almost = true),
                                                               :params => Dict("verbose" => false,
                                                                               "max_step_fraction" => 0.75))))
    asset_statistics!(portfolio)

    obj = MinRisk()

    rm = EVaR()
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    z1 = get_z(portfolio, rm)
    @test isapprox(z1, 0.004652651226719961, rtol = 5.0e-6)

    rm = EDaR()
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    z2 = get_z(portfolio, rm)
    @test isapprox(z2, 0.00916553108191174, rtol = 5.0e-5)

    rm = RLVaR()
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    z3 = get_z(portfolio, rm)
    @test isapprox(z3, 0.0018050418972062146, rtol = 5.0e-6)

    rm = RLDaR()
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    z4 = get_z(portfolio, rm)
    @test isapprox(z4, 0.003567369182292617, rtol = 5.0e-8)

    rm = [[EVaR(), EVaR()]]
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    z5 = get_z(portfolio, rm[1])
    @test isapprox(z5[1], z5[2])

    rm = [[EDaR(), EDaR()]]
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    z6 = get_z(portfolio, rm[1])
    @test isapprox(z6[1], z6[2], rtol = 1.0e-5)

    rm = [[RLVaR(), RLVaR()]]
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    z7 = get_z(portfolio, rm[1])
    @test isapprox(z7[1], z7[2], rtol = 1.0e-5)

    rm = [[RLDaR(), RLDaR()]]
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    z8 = get_z(portfolio, rm[1])
    @test isapprox(z8[1], z8[2], rtol = 1.0e-5)

    @test isapprox(z1, z5[1], rtol = 1.0e-5)
    @test isapprox(z1, z5[2], rtol = 1.0e-5)

    @test isapprox(z2, z6[1], rtol = 5.0e-5)
    @test isapprox(z2, z6[2], rtol = 5.0e-5)

    @test isapprox(z3, z7[1], rtol = 5.0e-6)
    @test isapprox(z3, z7[2], rtol = 1.0e-5)

    @test isapprox(z4, z8[1], rtol = 5.0e-6)
    @test isapprox(z4, z8[2], rtol = 1.0e-5)

    obj = Sharpe(; rf = rf)

    rm = EVaR()
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    z9 = get_z(portfolio, rm)
    @test isapprox(z9, 0.006484925949235588, rtol = 5.0e-6)

    rm = EDaR()
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    z10 = get_z(portfolio, rm)
    @test isapprox(z10, 0.01165505512394213, rtol = 5.0e-7)

    rm = RLVaR()
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    z11 = get_z(portfolio, rm)
    @test isapprox(z11, 0.002530053705676598)

    rm = RLDaR()
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    z12 = get_z(portfolio, rm)
    @test isapprox(z12, 0.0041978310601217435, rtol = 5.0e-5)

    rm = [[EVaR(), EVaR()]]
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    z13 = get_z(portfolio, rm[1])
    @test isapprox(z13[1], z13[2])

    rm = [[EDaR(), EDaR()]]
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    z14 = get_z(portfolio, rm[1])
    @test isapprox(z14[1], z14[2], rtol = 5.0e-7)

    rm = [[RLVaR(), RLVaR()]]
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    z15 = get_z(portfolio, rm[1])
    @test isapprox(z15[1], z15[2], rtol = 5.0e-8)

    rm = [[RLDaR(), RLDaR()]]
    optimise!(portfolio, Trad(; rm = rm, obj = obj))
    z16 = get_z(portfolio, rm[1])
    @test isapprox(z16[1], z16[2])

    @test isapprox(z9, z13[1], rtol = 0.0001)
    @test isapprox(z9, z13[2], rtol = 0.0001)

    @test isapprox(z10, z14[1], rtol = 0.0001)
    @test isapprox(z10, z14[2], rtol = 0.0001)

    @test isapprox(z11, z15[1], rtol = 5.0e-6)
    @test isapprox(z11, z15[2], rtol = 5.0e-6)

    @test isapprox(z12, z16[1], rtol = 0.0001)
    @test isapprox(z12, z16[2], rtol = 0.0001)

    @test z1 < z9
    @test z2 < z10
    @test z3 < z11
    @test z4 < z12
end
