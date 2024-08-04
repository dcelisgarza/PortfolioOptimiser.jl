using CSV, TimeSeries, DataFrames, StatsBase, Statistics, LinearAlgebra, Test, Clarabel,
      PortfolioOptimiser

prices = TimeArray(CSV.File("./assets/stock_prices.csv"); timestamp = :date)
rf = 1.0329^(1 / 252) - 1
l = 2.0

@testset "SD vec" begin
    portfolio = Portfolio2(; prices = prices,
                           solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                            :params => Dict("verbose" => false))))
    asset_statistics2!(portfolio)

    obj = MinRisk()
    wt0 = [0.0078093616583851146, 0.030698578999046943, 0.010528316771324561,
           0.027486806578381814, 0.012309038313357737, 0.03341430871186881,
           1.1079055085166888e-7, 0.13985416268183082, 2.0809271230580642e-7,
           6.32554715643476e-6, 0.28784123553791136, 1.268075347971748e-7,
           8.867081236591187e-8, 0.12527141708492384, 3.264070667606171e-7,
           0.015079837844627948, 1.5891383112101438e-5, 0.1931406057562605,
           2.654124109083291e-7, 0.11654298695072397]
    riskt0 = 0.007704593409157056
    rett0 = 0.0003482663810696356

    rm = SD2(; settings = RiskMeasureSettings(; scale = 2.0))
    w1 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r1 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret1 = dot(portfolio.mu, w1.weights)
    wt = [0.007803574652793466, 0.030696393880215295, 0.01053164984154553,
          0.027487607855267902, 0.012312334265071122, 0.033412250816636166,
          1.299471553518265e-7, 0.13985442592481906, 2.4156277302018273e-7,
          7.864374081465627e-6, 0.2878401132986823, 1.4735956218041721e-7,
          1.0321921819409948e-7, 0.12527069535291313, 3.955555293421062e-7,
          0.01507898235254538, 1.9492437441821213e-5, 0.19314048155138044,
          3.077617211885732e-7, 0.1165428079906476]
    riskt = 0.007704593875620669
    rett = 0.0003482746054188019
    @test isapprox(w1.weights, wt0, rtol = 5.0e-5)
    @test isapprox(r1, riskt0, rtol = 1.0e-7)
    @test isapprox(ret1, rett0, rtol = 5.0e-5)
    @test isapprox(w1.weights, wt)
    @test isapprox(r1, riskt)
    @test isapprox(ret1, rett)

    rm = [[SD2(), SD2()]]
    w2 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r2 = calc_risk(portfolio; type = :Trad2, rm = rm[1][1])
    ret2 = dot(portfolio.mu, w2.weights)
    wt = [0.007915388871475906, 0.030683958752011704, 0.010508205609248093,
          0.027475849952762574, 0.012282359540077049, 0.03341303914251413,
          2.175127184907252e-7, 0.13985531485392316, 3.857034710616735e-7,
          7.272686859215288e-6, 0.2878127342449225, 2.4118452837030857e-7,
          1.7754786906244115e-7, 0.12527046149549506, 9.208995835372493e-7,
          0.015081632349546324, 1.6957320970673963e-5, 0.1931387421643356,
          4.891535906473207e-7, 0.11653565101409688]
    riskt = 0.007704594747672459
    rett = 0.0003482443267403384
    @test isapprox(w2.weights, wt0, rtol = 0.0005)
    @test isapprox(r2, riskt0, rtol = 5.0e-7)
    @test isapprox(ret2, rett0, rtol = 0.0001)
    @test isapprox(w2.weights, wt)
    @test isapprox(r2, riskt)
    @test isapprox(ret2, rett)

    obj = SR(; rf = rf)
    wt0 = [3.435681413150756e-10, 9.269743242817558e-10, 1.174114891347931e-9,
           7.447407606568295e-10, 0.5180552669298059, 1.08151797431462e-10,
           0.0636508036260703, 7.872264113421062e-10, 7.841830201959634e-10,
           3.9005509625957585e-10, 6.479557895235057e-10, 8.472023236127232e-11,
           5.766670106753152e-11, 1.988136246095318e-10, 5.935811276550078e-11,
           0.14326634942881586, 0.1964867973307653, 7.554937254824565e-10,
           0.0785407748474901, 7.740298948228655e-10]
    riskt0 = 0.013160876658207102
    rett0 = 0.0014788430765515807

    rm = SD2(; settings = RiskMeasureSettings(; scale = 2.0))
    w3 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r3 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret3 = dot(portfolio.mu, w3.weights)
    wt = [8.52841592569558e-11, 2.253252149545565e-10, 2.8722458050462655e-10,
          1.8137873748868258e-10, 0.5180597467729909, 2.6576392522330683e-11,
          0.06365110534863692, 1.9165429044549566e-10, 1.8545427508419136e-10,
          9.460111853647939e-11, 1.5765293405072457e-10, 2.09537661975462e-11,
          1.5085382035750038e-11, 4.892760285281127e-11, 1.4795089265880615e-11,
          0.1432681044346168, 0.19649528591245916, 1.8353028354843352e-10,
          0.07852575562500307, 1.878492923799339e-10]
    riskt = 0.013160937359892306
    rett = 0.0014788493063661143
    @test isapprox(w3.weights, wt0, rtol = 5.0e-5)
    @test isapprox(r3, riskt0, rtol = 5.0e-6)
    @test isapprox(ret3, rett0, rtol = 5.0e-6)
    @test isapprox(w3.weights, wt)
    @test isapprox(r3, riskt)
    @test isapprox(ret3, rett)

    rm = [[SD2(), SD2()]]
    w4 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r4 = calc_risk(portfolio; type = :Trad2, rm = rm[1][1])
    ret4 = dot(portfolio.mu, w4.weights)
    wt = [9.858016009804939e-11, 2.5694132694162927e-10, 3.282857496757523e-10,
          2.0744206575853424e-10, 0.5180585632618029, 3.068747672161786e-11,
          0.0636510186847423, 2.199009910896005e-10, 2.1682604479247158e-10,
          1.0977711348502211e-10, 1.823471135999993e-10, 2.4290756164631022e-11,
          1.7518442692620923e-11, 5.6805730694743005e-11, 1.7107354743957206e-11,
          0.14326797917919648, 0.19649623796071977, 2.1120264112672722e-10,
          0.07852619872038856, 2.1543716097380204e-10]
    riskt = 0.013160926757877493
    rett = 0.0014788482184721376
    @test isapprox(w4.weights, wt0, rtol = 5.0e-5)
    @test isapprox(r4, riskt0, rtol = 5.0e-6)
    @test isapprox(ret4, rett0, rtol = 5.0e-6)
    @test isapprox(w4.weights, wt)
    @test isapprox(r4, riskt)
    @test isapprox(ret4, rett)

    rm = [[SD2(; formulation = QuadSD()), SD2(; formulation = SimpleSD())]]
    w9 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r9 = calc_risk(portfolio; type = :Trad2, rm = rm[1][1])
    ret9 = dot(portfolio.mu, w9.weights)
    wt = [0.007880024993803471, 0.03069715970077981, 0.010506316128987503,
          0.027491209570591545, 0.012285931087288456, 0.03341338952912436,
          7.598476262379018e-9, 0.13984935908109208, 1.390559757879338e-8,
          3.9788368331534336e-7, 0.28782740688266556, 8.693025434887618e-9,
          6.317718226870099e-9, 0.12528014529233403, 2.2283390396918905e-8,
          0.01508412915677593, 1.0132673276690782e-6, 0.19313003507359347,
          1.7638155638061054e-8, 0.1165534059155893]
    riskt = 0.007704591179178414
    rett = 0.0003482414668032645
    @test isapprox(w9.weights, wt0, rtol = 5.0e-4)
    @test isapprox(r9, riskt0, rtol = 5.0e-7)
    @test isapprox(ret9, rett0, rtol = 1.0e-4)
    @test isapprox(w9.weights, wt)
    @test isapprox(r9, riskt)
    @test isapprox(ret9, rett)

    rm = [[SD2(; formulation = SimpleSD()), SD2(; formulation = QuadSD())]]
    w10 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r10 = calc_risk(portfolio; type = :Trad2, rm = rm[1][1])
    ret10 = dot(portfolio.mu, w10.weights)
    wt = [0.00788002499379505, 0.030697159700781393, 0.010506316128987525,
          0.02749120957059247, 0.012285931087289991, 0.0334133895291245,
          7.59847626776494e-9, 0.13984935908109256, 1.390559758887827e-8,
          3.9788368367743396e-7, 0.2878274068826678, 8.69302544109695e-9,
          6.317718231315523e-9, 0.12528014529233325, 2.2283390411837418e-8,
          0.015084129156775265, 1.013267328639817e-6, 0.19313003507359447,
          1.7638155650901756e-8, 0.11655340591558987]
    riskt = 0.0077045911791784145
    rett = 0.0003482414668032652
    @test isapprox(w10.weights, wt0, rtol = 0.0005)
    @test isapprox(r10, riskt0, rtol = 5.0e-7)
    @test isapprox(ret10, rett0, rtol = 0.0001)
    @test isapprox(w10.weights, wt)
    @test isapprox(r10, riskt)
    @test isapprox(ret10, rett)

    # Risk upper bound
    obj = MaxRet()
    rm = SD2(; settings = RiskMeasureSettings(; scale = 2.0))
    rm.settings.ub = r1
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r1 ||
          abs(calc_risk(portfolio; type = :Trad2, rm = rm) - r1) < 5e-10

    rm = [[SD2(), SD2()]]
    rm[1][1].settings.ub = r2
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm[1][1]) <= r2 ||
          abs(calc_risk(portfolio; type = :Trad2, rm = rm[1][1]) - r2) < 5e-10

    obj = SR(; rf = rf)
    rm = SD2(; settings = RiskMeasureSettings(; scale = 2.0))
    rm.settings.ub = r1
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r1 ||
          abs(calc_risk(portfolio; type = :Trad2, rm = rm) - r1) < 1e-10

    rm = [[SD2(), SD2()]]
    rm[1][1].settings.ub = r2
    optimise2!(portfolio; rm = rm, obj = obj)
    calc_risk(portfolio; type = :Trad2, rm = rm[1][1])
    @test calc_risk(portfolio; type = :Trad2, rm = rm[1][1]) <= r2 ||
          abs(calc_risk(portfolio; type = :Trad2, rm = rm[1][1]) - r2) < 1e-10

    # Ret lower bound
    obj = MinRisk()

    rm = SD2(; settings = RiskMeasureSettings(; scale = 2.0))
    rm.settings.ub = Inf
    portfolio.mu_l = ret1
    w5 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w5.weights) >= ret1

    rm = [[SD2(), SD2()]]
    portfolio.mu_l = ret2
    w6 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w6.weights) >= ret2

    obj = SR(; rf = rf)
    rm = SD2(; settings = RiskMeasureSettings(; scale = 2.0))
    portfolio.mu_l = ret1
    w7 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w7.weights) >= ret1

    rm = [[SD2(), SD2()]]
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

    rm = SD2()
    obj = MinRisk()
    w1 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)

    rm = [SD2(), Skew2(; settings = RiskMeasureSettings(; scale = 0.0))]
    w2 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    @test isapprox(w1.weights, w2.weights, rtol = 3e-4)

    rm = [SD2(), Skew2(; settings = RiskMeasureSettings(; scale = 2))]
    w = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    wt = [0.0026354275749322443, 0.05520947741035115, 2.88292486305246e-7,
          0.011217444462648793, 0.015540235791726633, 0.007294887210979084,
          3.899501931693162e-8, 0.1384846686508059, 5.619219962894404e-8,
          1.4264636900253708e-7, 0.2855982912592649, 8.550398887290524e-8,
          4.0185944557342566e-8, 0.11727545683980922, 0.005180482430773081,
          0.016745180622565338, 0.0077834334790627055, 0.20483183287545345,
          7.577961384734552e-8, 0.1322024537960061]
    @test isapprox(w.weights, wt)

    rm = [SD2(), Skew2(; settings = RiskMeasureSettings(; scale = 8))]
    w = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    wt = [8.413927417714243e-7, 0.09222294098507178, 1.6168463788897735e-7,
          2.0594000236277162e-7, 0.008523442957645658, 3.007500480370547e-7,
          6.833538384822706e-8, 0.13619418248362034, 9.979458409901339e-8,
          1.5596045505028015e-7, 0.26494454649109994, 3.4315324995498946e-6,
          1.2825613036862424e-7, 0.0783181629157472, 0.02532294038010334,
          0.01907855067328539, 0.012932625739071507, 0.21592581988533274,
          1.422385714567375e-7, 0.14653125160396763]
    @test isapprox(w.weights, wt)

    rm = [SD2(), SSkew2(; settings = RiskMeasureSettings(; scale = 0.0))]
    w2 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    @test isapprox(w1.weights, w2.weights, rtol = 3e-4)

    rm = [SD2(), SSkew2(; settings = RiskMeasureSettings(; scale = 2))]
    w = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    wt = [1.384008546759806e-6, 0.0316494420628888, 1.4466615477601905e-6,
          0.015775935372681668, 0.010442899482149982, 0.009851951563574745,
          1.6845564712725654e-7, 0.1404230153792723, 2.93065068940981e-7,
          5.00892434748868e-7, 0.32532989744017604, 3.1063572739077716e-7,
          1.7332147477485165e-7, 0.1184225153788876, 1.25268476291211e-6,
          0.014302557449595256, 2.0736860865331673e-6, 0.2083923849842472,
          3.9292677008851197e-7, 0.12540140454845938]
    @test isapprox(w.weights, wt)

    rm = [SD2(), SSkew2(; settings = RiskMeasureSettings(; scale = 8))]
    w = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    wt = [1.5365064547283644e-7, 0.02437633461456116, 1.299823053389551e-7,
          3.5309417854060804e-7, 3.017455267702621e-6, 2.6113474157486046e-7,
          3.341100393369674e-8, 0.13768001144500144, 5.584135855499354e-8,
          1.1036943651763183e-7, 0.38090454974359306, 7.778862184342059e-8,
          4.2133989399698356e-8, 0.09436174496182163, 3.53865987048023e-7,
          0.013926597934786485, 4.4441759524485204e-7, 0.21941318550910147,
          7.686992020172018e-8, 0.12933246577608337]
    @test isapprox(w.weights, wt)
end
