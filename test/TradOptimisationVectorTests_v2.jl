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

    rm = [[SD2(; formulation = QuadSD()), SD2(; formulation = SimpleSD())]]
    w9 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r9 = calc_risk(portfolio; type = :Trad2, rm = rm[1][1])
    ret9 = dot(portfolio.mu, w9.weights)
    wt = [0.007893411266173175, 0.03069787875733664, 0.010517467279741203,
          0.02749153027861106, 0.012315934864436422, 0.033357226892062115,
          1.5186481591738784e-8, 0.13984281248182642, 2.6392792428112547e-8,
          4.3321340905586297e-7, 0.28782917621099047, 1.622410061275207e-8,
          1.2780832333541773e-8, 0.1252610993835163, 6.694587666856286e-8,
          0.015097144456616973, 6.797374193485821e-7, 0.1931368287482932,
          3.321982647315455e-8, 0.11655820567965758]
    riskt = 0.007704591319283172
    rett = 0.0003483682842520989
    @test isapprox(w9.weights, wt0, rtol = 5.0e-4)
    @test isapprox(r9, riskt0, rtol = 5.0e-7)
    @test isapprox(ret9, rett0, rtol = 5.0e-4)
    @test isapprox(w9.weights, wt)
    @test isapprox(r9, riskt)
    @test isapprox(ret9, rett)

    rm = [[SD2(; formulation = SimpleSD()), SD2(; formulation = QuadSD())]]
    w10 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r10 = calc_risk(portfolio; type = :Trad2, rm = rm[1][1])
    ret10 = dot(portfolio.mu, w10.weights)
    wt = [0.007893411258991168, 0.030697878760864626, 0.010517467283966332,
          0.027491530280761234, 0.01231593487986923, 0.033357226869771425,
          1.5186461875055205e-8, 0.13984281247966734, 2.6392759414051237e-8,
          4.332127216826158e-7, 0.2878291762142109, 1.6224079654211018e-8,
          1.278081655504334e-8, 0.12526109937507104, 6.694578390760774e-8,
          0.015097144461265135, 6.797363006658334e-7, 0.1931368287542989,
          3.32197849377567e-8, 0.11655820568255393]
    riskt = 0.007704591319282906
    rett = 0.0003483682843049291
    @test isapprox(w10.weights, wt0, rtol = 0.0005)
    @test isapprox(r10, riskt0, rtol = 5.0e-7)
    @test isapprox(ret10, rett0, rtol = 0.0005)
    @test isapprox(w10.weights, wt)
    @test isapprox(r10, riskt)
    @test isapprox(ret10, rett)
end

@testset "MAD vec" begin
    portfolio = Portfolio2(; prices = prices,
                           solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                            :params => Dict("verbose" => false))))
    asset_statistics2!(portfolio)

    obj = MinRisk()
    wt0 = [0.014061915969795006, 0.04237496202139695, 0.01686336647933223,
           0.002020806523507274, 0.01768380555270159, 0.05422405215837249,
           2.9350570130142624e-10, 0.15821651684232851, 3.0060399538100176e-10,
           7.086259738110947e-10, 0.23689725720512037, 7.61312046632753e-11,
           6.545365843921615e-11, 0.12783204733233253, 0.0003509663915665695,
           0.0009122945557616327, 0.0439493643547516, 0.18272429223715872,
           4.105696610811196e-10, 0.10188835052098438]
    riskt0 = 0.005627573038796034
    rett0 = 0.0003490122974688338

    rm = MAD2(; mu = portfolio.mu, settings = RiskMeasureSettings(; scale = 2.0))
    w1 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r1 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret1 = dot(portfolio.mu, w1.weights)
    wt = [0.01406186866835325, 0.04237493107143327, 0.01686335260831174,
          0.0020208147122650153, 0.017683877634995713, 0.05422407072490352,
          1.520384422389592e-10, 0.15821654686210973, 1.521614191247427e-10,
          3.2941544413819765e-10, 0.23689726473708203, 3.655003054856631e-11,
          3.207149176530337e-11, 0.12783200363024275, 0.0003509536239723649,
          0.0009122872781545095, 0.0439493847011208, 0.18272428979193125,
          2.0502674476911855e-10, 0.10188835304786052]
    riskt = 0.00562757303767025
    rett = 0.0003490123937016413
    @test isapprox(w1.weights, wt0, rtol = 5.0e-7)
    @test isapprox(r1, riskt0)
    @test isapprox(ret1, rett0, rtol = 5.0e-7)
    @test isapprox(w1.weights, wt)
    @test isapprox(r1, riskt)
    @test isapprox(ret1, rett)

    rm = [[MAD2(; mu = portfolio.mu), MAD2()]]
    w2 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r2 = calc_risk(portfolio; type = :Trad2, rm = rm[1][1])
    ret2 = dot(portfolio.mu, w2.weights)
    wt = [0.014061881328717746, 0.042374912207995226, 0.016863341624819422,
          0.0020208132843084273, 0.017683882600877637, 0.05422406849292218,
          2.7972352995564857e-11, 0.15821655478063953, 2.609884816742614e-11,
          6.442742240765734e-11, 0.23689727240879593, 5.712793017812015e-12,
          4.881340926115715e-12, 0.12783200028731787, 0.00035094692825357994,
          0.0009122899577910392, 0.043949380082220386, 0.18272427968205707,
          3.4528375107134664e-11, 0.10188837616966281]
    riskt = 0.005627573036802626
    rett = 0.00034901240817621353
    @test isapprox(w2.weights, wt0, rtol = 5.0e-7)
    @test isapprox(r2, riskt0, rtol = 5.0e-10)
    @test isapprox(ret2, rett0, rtol = 5.0e-7)
    @test isapprox(w2.weights, wt)
    @test isapprox(r2, riskt)
    @test isapprox(ret2, rett)

    obj = SR(; rf = rf)
    wt0 = [3.372797021213529e-8, 8.679493419672167e-8, 8.730378695790742e-8,
           4.654016314992972e-8, 0.6621196971211604, 1.000789705313297e-8,
           0.04256386189823906, 5.0027909676906887e-8, 9.072276529043811e-8,
           4.296795445352721e-8, 7.991647846958404e-8, 7.108969143618601e-9,
           5.039720687490243e-9, 1.839999189017112e-8, 5.602046740184832e-9,
           0.1343671243475813, 0.08752271182145684, 7.258944630234996e-8,
           0.07342589536656563, 7.269496276800682e-8]
    riskt0 = 0.009898352231115614
    rett0 = 0.0015741047141763708

    rm = MAD2(; mu = portfolio.mu, settings = RiskMeasureSettings(; scale = 2.0))
    w3 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r3 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret3 = dot(portfolio.mu, w3.weights)
    wt = [7.934063396858117e-9, 2.0408768052563734e-8, 2.0778681518746626e-8,
          1.1275026485105553e-8, 0.6622287174731025, 2.313052237883684e-9,
          0.042585911131941455, 1.1897377032234231e-8, 2.2184407674858158e-8,
          1.0275470757089733e-8, 1.8861206457621417e-8, 1.6106078520734577e-9,
          1.1200492715092873e-9, 4.33425147093016e-9, 1.2441173248135942e-9,
          0.13436790158853482, 0.08741390923117279, 1.7249113608568275e-8,
          0.07340339155009168, 1.7538963712621776e-8]
    riskt = 0.009899090397890692
    rett = 0.0015742126894920563
    @test isapprox(w3.weights, wt0, rtol = 0.0005)
    @test isapprox(r3, riskt0, rtol = 0.0001)
    @test isapprox(ret3, rett0, rtol = 0.0001)
    @test isapprox(w3.weights, wt)
    @test isapprox(r3, riskt)
    @test isapprox(ret3, rett)

    rm = [[MAD2(), MAD2()]]
    w4 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r4 = calc_risk(portfolio; type = :Trad2, rm = rm[1][1])
    ret4 = dot(portfolio.mu, w4.weights)
    wt = [2.6906266986058425e-9, 6.91960213688136e-9, 7.059695895617844e-9,
          3.803375295535123e-9, 0.6622644675673581, 7.535013022066731e-10,
          0.04259456854168396, 4.007362895264134e-9, 7.44521731286276e-9,
          3.47632460287596e-9, 6.412667642777305e-9, 5.140769640424289e-10,
          3.425780027847076e-10, 1.447560219564652e-9, 3.8751030966899683e-10,
          0.1343658953007059, 0.08736747254275348, 5.8655775842608014e-9,
          0.07340753897007853, 5.951743163590262e-9]
    riskt = 0.009899326753550985
    rett = 0.0015742472514148917
    @test isapprox(w4.weights, wt0, rtol = 0.0005)
    @test isapprox(r4, riskt0, rtol = 0.0001)
    @test isapprox(ret4, rett0, rtol = 0.0001)
    @test isapprox(w4.weights, wt)
    @test isapprox(r4, riskt)
    @test isapprox(ret4, rett)

    # Risk upper bound
    obj = MaxRet()
    rm = MAD2(; settings = RiskMeasureSettings(; scale = 2.0))
    rm.settings.ub = r1
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r1 ||
          abs(calc_risk(portfolio; type = :Trad2, rm = rm) - r1) < 5e-10

    rm = [[MAD2(), MAD2()]]
    rm[1][2].settings.ub = r2
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm[1][2]) <= r2 ||
          abs(calc_risk(portfolio; type = :Trad2, rm = rm[1][2]) - r2) < 5e-10

    obj = SR(; rf = rf)
    rm = MAD2(; settings = RiskMeasureSettings(; scale = 2.0))
    rm.settings.ub = r1
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r1 ||
          abs(calc_risk(portfolio; type = :Trad2, rm = rm) - r1) < 1e-10

    rm = [[MAD2(), MAD2(; mu = portfolio.mu)]]
    rm[1][1].settings.ub = r2
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm[1][2]) <= r2 ||
          abs(calc_risk(portfolio; type = :Trad2, rm = rm[1][2]) - r2) < 1e-10

    # Ret lower bound
    obj = MinRisk()
    rm = MAD2(; settings = RiskMeasureSettings(; scale = 2.0))
    portfolio.mu_l = ret1
    w5 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w5.weights) >= ret1

    rm = [[MAD2(), MAD2()]]
    portfolio.mu_l = ret2
    w6 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w6.weights) >= ret2

    obj = SR(; rf = rf)
    rm = MAD2(; settings = RiskMeasureSettings(; scale = 2.0))
    portfolio.mu_l = ret1
    w7 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w7.weights) >= ret1

    rm = [[MAD2(), MAD2()]]
    portfolio.mu_l = ret2
    w8 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w8.weights) >= ret2
end

@testset "SSD vec" begin
    portfolio = Portfolio2(; prices = prices,
                           solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                            :params => Dict("verbose" => false))))
    asset_statistics2!(portfolio)

    obj = MinRisk()
    wt0 = [2.954283441277634e-8, 0.04973740293196223, 1.5004834875433086e-8,
           0.002978185203626395, 0.00255077171396876, 0.02013428421720317,
           8.938505323199939e-10, 0.12809490679767346, 2.5514571986823903e-9,
           3.4660313800221236e-9, 0.29957738105080456, 3.6587132183584753e-9,
           1.61047759821642e-9, 0.1206961339634279, 0.012266097184153368,
           0.009663325635394784, 1.859820936315932e-8, 0.22927479857319558,
           3.22169253589993e-9, 0.12502663418048846]
    riskt0 = 0.005538773213915548
    rett0 = 0.00031286022410236273

    rm = SSD2(; mu = portfolio.mu, settings = RiskMeasureSettings(; scale = 2.0))
    w1 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r1 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret1 = dot(portfolio.mu, w1.weights)
    wt = [6.960433324737852e-9, 0.04973824078712541, 3.5139017334540466e-9,
          0.0029776760402027854, 0.0025509094502779977, 0.020132620937815832,
          1.722851693536219e-10, 0.1280944627600755, 5.649870022984124e-10,
          7.815575397825495e-10, 0.2995763729808538, 8.277153645130121e-10,
          3.422304597374197e-10, 0.12069668523130198, 0.012266251816130945,
          0.009663132326196127, 4.363236723893739e-9, 0.22927646722931586,
          7.237055363002736e-10, 0.12502716219065096]
    riskt = 0.005538773198516764
    rett = 0.0003128617100692991
    @test isapprox(w1.weights, wt0, rtol = 1.0e-5)
    @test isapprox(r1, riskt0)
    @test isapprox(ret1, rett0, rtol = 5.0e-6)
    @test isapprox(w1.weights, wt)
    @test isapprox(r1, riskt)
    @test isapprox(ret1, rett)

    rm = [[SSD2(; mu = portfolio.mu), SSD2()]]
    w2 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r2 = calc_risk(portfolio; type = :Trad2, rm = rm[1][1])
    ret2 = dot(portfolio.mu, w2.weights)
    wt = [2.8356011614873756e-9, 0.04973815922174698, 1.420550272865242e-9,
          0.002977544107776336, 0.002550734790594938, 0.020132684505438647,
          4.909123033041351e-11, 0.1280945084922063, 2.1026595958143468e-10,
          2.9914680494510455e-10, 0.299577047726477, 3.180957691080518e-10,
          1.1884040347081485e-10, 0.12069653344893792, 0.012266348181321605,
          0.009663148702489027, 1.769272482205502e-9, 0.2292763924783607,
          2.7540831039969104e-10, 0.125026891048378]
    riskt = 0.00553877319575872
    rett = 0.000312861063993284
    @test isapprox(w2.weights, wt0, rtol = 1.0e-5)
    @test isapprox(r2, riskt0)
    @test isapprox(ret2, rett0, rtol = 5.0e-6)
    @test isapprox(w2.weights, wt)
    @test isapprox(r2, riskt)
    @test isapprox(ret2, rett)

    obj = SR(; rf = rf)
    wt0 = [1.8955665771231532e-8, 4.4148508725600023e-8, 3.537890454211662e-8,
           2.1966271358039556e-8, 0.6666203563586275, 6.130148331498872e-9,
           0.03792018451465443, 3.563315827678111e-8, 4.349162854829938e-8,
           1.8479882644634467e-8, 4.552310886494339e-8, 4.8863225987358126e-9,
           3.315774614641478e-9, 1.2573247089938602e-8, 3.5165001620600556e-9,
           0.1718521246394113, 0.10257058901854942, 4.7654011023485184e-8,
           0.021036366772688796, 3.7042935949165386e-8]
    riskt0 = 0.00981126385893784
    rett0 = 0.0015868900032431047

    rm = SSD2(; mu = portfolio.mu, settings = RiskMeasureSettings(; scale = 2.0))
    w3 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r3 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret3 = dot(portfolio.mu, w3.weights)
    wt = [2.0576174316494517e-8, 4.8814610184186155e-8, 3.898419830241749e-8,
          2.3953470996597847e-8, 0.6665776024206931, 6.2050881889442745e-9,
          0.0379164098146924, 3.926735636240487e-8, 4.808098102422677e-8,
          2.0044157078411763e-8, 5.035190667246275e-8, 4.811139222707983e-9,
          3.0517811832662023e-9, 1.3423714141373433e-8, 3.277007061804338e-9,
          0.17183757235202618, 0.1026005130461382, 5.273819953852542e-8,
          0.021067487940363432, 4.0846302694850526e-8]
    riskt = 0.009810971129515471
    rett = 0.0015868464822755095
    @test isapprox(w3.weights, wt0, rtol = 0.0001)
    @test isapprox(r3, riskt0, rtol = 5.0e-5)
    @test isapprox(ret3, rett0, rtol = 5.0e-5)
    @test isapprox(w3.weights, wt)
    @test isapprox(r3, riskt)
    @test isapprox(ret3, rett)

    rm = [[SSD2(), SSD2()]]
    w4 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r4 = calc_risk(portfolio; type = :Trad2, rm = rm[1][1])
    ret4 = dot(portfolio.mu, w4.weights)
    wt = [5.851576083911889e-9, 1.4150127488740236e-8, 1.1258800460572914e-8,
          6.843371359487446e-9, 0.6666414693122927, 1.6304079796247095e-9,
          0.03792573395556936, 1.1348845531303547e-8, 1.3931340738360478e-8,
          5.696455136245695e-9, 1.4611398493896516e-8, 1.22081039352903e-9,
          7.040733859924109e-10, 3.751457820958834e-9, 7.701384680513084e-10,
          0.17184489903975814, 0.10255677095513442, 1.531136895896822e-8,
          0.021031007844837638, 1.1812235386219602e-8]
    riskt = 0.0098114054666367
    rett = 0.001586911132321079
    @test isapprox(w4.weights, wt0, rtol = 5.0e-5)
    @test isapprox(r4, riskt0, rtol = 5.0e-5)
    @test isapprox(ret4, rett0, rtol = 5.0e-5)
    @test isapprox(w4.weights, wt)
    @test isapprox(r4, riskt)
    @test isapprox(ret4, rett)

    # Risk upper bound
    obj = MaxRet()
    rm = SSD2(; settings = RiskMeasureSettings(; scale = 2.0))
    rm.settings.ub = r1
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r1 ||
          abs(calc_risk(portfolio; type = :Trad2, rm = rm) - r1) < 5e-10

    rm = [[SSD2(), SSD2()]]
    rm[1][2].settings.ub = r2
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm[1][2]) <= r2 ||
          abs(calc_risk(portfolio; type = :Trad2, rm = rm[1][2]) - r2) < 5e-10

    obj = SR(; rf = rf)
    rm = SSD2(; settings = RiskMeasureSettings(; scale = 2.0))
    rm.settings.ub = r1
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r1 ||
          abs(calc_risk(portfolio; type = :Trad2, rm = rm) - r1) < 5e-10

    rm = [[SSD2(), SSD2(; mu = portfolio.mu)]]
    rm[1][1].settings.ub = r2
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm[1][2]) <= r2 ||
          abs(calc_risk(portfolio; type = :Trad2, rm = rm[1][2]) - r2) < 1e-10

    # Ret lower bound
    obj = MinRisk()
    rm = SSD2(; settings = RiskMeasureSettings(; scale = 2.0))
    portfolio.mu_l = ret1
    w5 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w5.weights) >= ret1

    rm = [[SSD2(), SSD2()]]
    portfolio.mu_l = ret2
    w6 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w6.weights) >= ret2

    obj = SR(; rf = rf)
    rm = SSD2(; settings = RiskMeasureSettings(; scale = 2.0))
    portfolio.mu_l = ret1
    w7 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w7.weights) >= ret1

    rm = [[SSD2(), SSD2()]]
    portfolio.mu_l = ret2
    w8 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w8.weights) >= ret2
end

@testset "FLPM vec" begin
    portfolio = Portfolio2(; prices = prices,
                           solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                            :params => Dict("verbose" => false))))
    asset_statistics2!(portfolio)

    obj = MinRisk()
    wt0 = [0.004266070614517317, 0.04362165239521167, 0.01996043023729806,
           0.007822722623891595, 0.060525786877357816, 2.187204740032422e-8,
           0.00039587162942815576, 0.13089236100375287, 7.734531969787049e-9,
           0.0118785975269765, 0.2066094523343813, 6.469640939191796e-10,
           6.246750358607508e-10, 0.08329494463798208, 1.6616489736084757e-9,
           0.013888127426323596, 0.0873465246195096, 0.19210093372199202,
           0.03721303157281544, 0.10018346023869455]
    riskt0 = 0.00265115220934628
    rett0 = 0.0005443423420749122

    rm = FLPM2(; target = rf, settings = RiskMeasureSettings(; scale = 2.0))
    w1 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r1 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret1 = dot(portfolio.mu, w1.weights)
    wt = [0.004266104426490134, 0.04362180013313201, 0.019960458730624693,
          0.007822768526557906, 0.06052525935724189, 2.5189992639074766e-10,
          0.00039596144449614276, 0.1308926842708566, 9.6512578139821e-11,
          0.011878930319190989, 0.2066096392346393, 7.720672274633286e-12,
          7.51612634574704e-12, 0.08329480308432323, 2.1089055580562457e-11,
          0.013888098428249715, 0.0873472499865774, 0.19210034328732165,
          0.037213207502595255, 0.10018269088296457]
    riskt = 0.002651152194543036
    rett = 0.0005443422035677568
    @test isapprox(w1.weights, wt0, rtol = 5.0e-6)
    @test isapprox(r1, riskt0)
    @test isapprox(ret1, rett0, rtol = 5.0e-7)
    @test isapprox(w1.weights, wt)
    @test isapprox(r1, riskt)
    @test isapprox(ret1, rett)

    rm = [[FLPM2(; target = rf), FLPM2(; target = rf)]]
    w2 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r2 = calc_risk(portfolio; type = :Trad2, rm = rm[1][1])
    ret2 = dot(portfolio.mu, w2.weights)
    wt = [0.0042661025692006084, 0.04362179398245574, 0.01996045752230989,
          0.007822766802592616, 0.06052528167274898, 4.706273071756809e-10,
          0.00039595776676766293, 0.13089267108182642, 1.6748614479253928e-10,
          0.01187892180394578, 0.20660963124085852, 9.920087603758116e-12,
          9.375656474763803e-12, 0.08329480850019082, 3.084958963640792e-11,
          0.013888099208033144, 0.08734722283656324, 0.19210036515047657,
          0.037213198389437985, 0.10018272078433316]
    riskt = 0.0026511521949200375
    rett = 0.0005443422112328603
    @test isapprox(w2.weights, wt0, rtol = 5.0e-6)
    @test isapprox(r2, riskt0)
    @test isapprox(ret2, rett0, rtol = 5.0e-7)
    @test isapprox(w2.weights, wt)
    @test isapprox(r2, riskt)
    @test isapprox(ret2, rett)

    obj = SR(; rf = rf)
    wt0 = [5.791704589818663e-10, 1.4777512342996448e-9, 1.4920733133812998e-9,
           8.941347428424144e-10, 0.6999099125632519, 2.145377355161713e-10,
           0.029295630576512924, 1.1027104693788755e-9, 1.8864271969797675e-9,
           8.43330450121613e-10, 1.4937081011622384e-9, 1.4856958187000145e-10,
           1.0768233412852032e-10, 3.8855123608537257e-10, 1.2149887816181597e-10,
           0.15181164107816766, 0.04226710946215913, 1.3947899372714116e-9,
           0.07671569251341252, 1.6615602330924226e-9]
    riskt0 = 0.00431255671125957
    rett0 = 0.0015948388159746803

    rm = FLPM2(; target = rf, settings = RiskMeasureSettings(; scale = 2.0))
    w3 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r3 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret3 = dot(portfolio.mu, w3.weights)
    wt = [2.4294572589841835e-9, 6.179960460836824e-9, 6.31476619346495e-9,
          3.7439642316023034e-9, 0.6999119242791928, 8.710519436213725e-10,
          0.029296946476234622, 4.639277689542512e-9, 7.91380032353166e-9,
          3.5241131557338203e-9, 6.273574072038055e-9, 5.920209240175243e-10,
          4.1797092904728947e-10, 1.606091921430325e-9, 4.741980163108361e-10,
          0.15181127705381728, 0.04226505979073052, 5.908053503080119e-9,
          0.07671473450029871, 7.011425553901324e-9]
    riskt = 0.004312564863804427
    rett = 0.001594841568930236
    @test isapprox(w3.weights, wt0, rtol = 5.0e-6)
    @test isapprox(r3, riskt0, rtol = 5.0e-6)
    @test isapprox(ret3, rett0, rtol = 5.0e-6)
    @test isapprox(w3.weights, wt)
    @test isapprox(r3, riskt)
    @test isapprox(ret3, rett)

    rm = [[FLPM2(; target = rf), FLPM2(; target = rf)]]
    w4 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r4 = calc_risk(portfolio; type = :Trad2, rm = rm[1][1])
    ret4 = dot(portfolio.mu, w4.weights)
    wt = [1.930685273810068e-9, 5.041092410026878e-9, 5.0821904314687215e-9,
          3.0141097622462803e-9, 0.6999128903418748, 6.729280633063908e-10,
          0.0292971145193568, 3.752169432584665e-9, 6.427834285996224e-9,
          2.848042833354374e-9, 5.1116853330762796e-9, 4.4397481371217813e-10,
          3.0284273137954636e-10, 1.2734195267495197e-9, 3.5073967972756645e-10,
          0.1518112155629529, 0.04226425292801926, 4.772062448540593e-9, 0.0767144799375144,
          5.686504884167509e-9]
    riskt = 0.004312567606261896
    rett = 0.0015948425039936168
    @test isapprox(w4.weights, wt0, rtol = 1.0e-5)
    @test isapprox(r4, riskt0, rtol = 5.0e-6)
    @test isapprox(ret4, rett0, rtol = 5.0e-6)
    @test isapprox(w4.weights, wt)
    @test isapprox(r4, riskt)
    @test isapprox(ret4, rett)

    # Risk upper bound
    obj = MaxRet()
    rm = FLPM2(; target = rf, settings = RiskMeasureSettings(; scale = 2.0))
    rm.settings.ub = r1
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r1 ||
          abs(calc_risk(portfolio; type = :Trad2, rm = rm) - r1) < 5e-10

    rm = [[FLPM2(; target = rf), FLPM2(; target = rf)]]
    rm[1][2].settings.ub = r2
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm[1][2]) <= r2 ||
          abs(calc_risk(portfolio; type = :Trad2, rm = rm[1][2]) - r2) < 1e-10

    obj = SR(; rf = rf)
    rm = FLPM2(; target = rf, settings = RiskMeasureSettings(; scale = 2.0))
    rm.settings.ub = r1
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r1 ||
          abs(calc_risk(portfolio; type = :Trad2, rm = rm) - r1) < 1e-10

    rm = [[FLPM2(; target = rf), FLPM2(; target = rf)]]
    rm[1][1].settings.ub = r2
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm[1][2]) <= r2 ||
          abs(calc_risk(portfolio; type = :Trad2, rm = rm[1][2]) - r2) < 1e-10

    # Ret lower bound
    obj = MinRisk()
    rm = FLPM2(; target = rf, settings = RiskMeasureSettings(; scale = 2.0))
    portfolio.mu_l = ret1
    w5 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w5.weights) >= ret1

    rm = [[FLPM2(; target = rf), FLPM2(; target = rf)]]
    portfolio.mu_l = ret2
    w6 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w6.weights) >= ret2

    obj = SR(; rf = rf)
    rm = FLPM2(; target = rf, settings = RiskMeasureSettings(; scale = 2.0))
    portfolio.mu_l = ret1
    w7 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w7.weights) >= ret1

    rm = [[FLPM2(; target = rf), FLPM2(; target = rf)]]
    portfolio.mu_l = ret2
    w8 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w8.weights) >= ret2
end

@testset "SLPM vec" begin
    portfolio = Portfolio2(; prices = prices,
                           solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                            :params => Dict("verbose" => false))))
    asset_statistics2!(portfolio)

    obj = MinRisk()
    wt0 = [1.1230805069911956e-8, 0.05524472463362737, 1.0686041548307544e-8,
           0.0043185378999763225, 0.033597348034736865, 1.0487157577222361e-8,
           1.1738886913269633e-9, 0.12478148562530009, 3.4647395424618816e-9,
           3.8805677196069256e-9, 0.3005648369145803, 2.0034183913036616e-9,
           1.0927362747553375e-9, 0.10661826438516031, 0.003123732919975542,
           0.021391817407374183, 0.003595424842043441, 0.22964898912299475,
           5.129978967782806e-9, 0.117114789064897]
    riskt0 = 0.005418882634929856
    rett0 = 0.0004088880259308715

    rm = SLPM2(; target = rf, settings = RiskMeasureSettings(; scale = 2.0))
    w1 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r1 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret1 = dot(portfolio.mu, w1.weights)
    wt = [3.040143822805962e-9, 0.05524527357462312, 2.8904547881370996e-9,
          0.004319294128856247, 0.03359597405643318, 2.8370407574090003e-9,
          2.7806479192171846e-10, 0.12478090085231208, 9.072566192158078e-10,
          1.021020094762195e-9, 0.300564774698862, 5.060902381563956e-10,
          2.55959987894066e-10, 0.10661908344799811, 0.003123925457394359,
          0.021391494122960396, 0.0035942208283433153, 0.2296509297212885,
          1.3643323906700271e-9, 0.11711411601056511]
    riskt = 0.00541888262069998
    rett = 0.0004088848980081816
    @test isapprox(w1.weights, wt0, rtol = 1.0e-5)
    @test isapprox(r1, riskt0)
    @test isapprox(ret1, rett0, rtol = 1.0e-5)
    @test isapprox(w1.weights, wt)
    @test isapprox(r1, riskt)
    @test isapprox(ret1, rett)

    rm = [[SLPM2(; target = rf), SLPM2(; target = rf)]]
    w2 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r2 = calc_risk(portfolio; type = :Trad2, rm = rm[1][1])
    ret2 = dot(portfolio.mu, w2.weights)
    wt = [1.5772354512388244e-9, 0.05524497981981157, 1.4978362635708959e-9,
          0.004319206332176008, 0.03359652816036835, 1.469949360823367e-9,
          1.1590126503590421e-10, 0.12478064380973253, 4.488202183056923e-10,
          5.089794342103587e-10, 0.30056494474637707, 2.365524936008412e-10,
          1.0420149303155579e-10, 0.10661897606833388, 0.0031241411356547535,
          0.021391577450620487, 0.0035943808427583647, 0.22965043507108537,
          6.906570587468534e-10, 0.11711417991294854]
    riskt = 0.005418882618158122
    rett = 0.0004088853956912218
    @test isapprox(w2.weights, wt0, rtol = 1.0e-5)
    @test isapprox(r2, riskt0)
    @test isapprox(ret2, rett0, rtol = 1.0e-5)
    @test isapprox(w2.weights, wt)
    @test isapprox(r2, riskt)
    @test isapprox(ret2, rett)

    obj = SR(; rf = rf)
    wt0 = [1.9161992029945534e-8, 4.442467014160941e-8, 3.487731975169222e-8,
           2.172326848650473e-8, 0.6654321506924412, 6.20892532022181e-9,
           0.03807260712526902, 3.6516022610300514e-8, 4.3159008520930105e-8,
           1.8350537901763542e-8, 4.619460482355355e-8, 5.0197040711936325e-9,
           3.3977158843464672e-9, 1.2834736295215969e-8, 3.5853236437253736e-9,
           0.17459230019672953, 0.10412390455189192, 4.844528935490425e-8,
           0.017778656482209734, 3.7052339729479755e-8]
    riskt0 = 0.00909392522496688
    rett0 = 0.0015869580721210722

    rm = SLPM2(; target = rf, settings = RiskMeasureSettings(; scale = 2.0))
    w3 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r3 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret3 = dot(portfolio.mu, w3.weights)
    wt = [1.7409926223114184e-8, 4.1094850431678315e-8, 3.214993409523547e-8,
          1.981327667997557e-8, 0.6654856600867949, 5.261353658182919e-9,
          0.03807572382780905, 3.3669849549290687e-8, 3.99137400781023e-8,
          1.6646247419788346e-8, 4.272685524345408e-8, 4.1464718731632056e-9,
          2.6250518036446776e-9, 1.1473167572867645e-8, 2.8011113883488952e-9,
          0.17459676505538677, 0.10403024524587386, 4.484132712601738e-8,
          0.0178112570297131, 3.418125916935718e-8]
    riskt = 0.009094178823863376
    rett = 0.0015869987550458655
    @test isapprox(w3.weights, wt0, rtol = 0.0005)
    @test isapprox(r3, riskt0, rtol = 5.0e-5)
    @test isapprox(ret3, rett0, rtol = 5.0e-5)
    @test isapprox(w3.weights, wt)
    @test isapprox(r3, riskt)
    @test isapprox(ret3, rett)

    rm = [[SLPM2(; target = rf), SLPM2(; target = rf)]]
    w4 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r4 = calc_risk(portfolio; type = :Trad2, rm = rm[1][1])
    ret4 = dot(portfolio.mu, w4.weights)
    wt = [7.0510286395162275e-9, 1.696232255596058e-8, 1.3217576346424488e-8,
          8.057141924814791e-9, 0.6654328349852813, 1.969482493957357e-9,
          0.03807304491622934, 1.3858841344062226e-8, 1.6470995111912925e-8,
          6.73440898283744e-9, 1.7655839895664892e-8, 1.5029864096051593e-9,
          8.666534914772722e-10, 4.569053938020522e-9, 9.402169109446613e-10,
          0.17458629963057767, 0.10411120599203362, 1.853604857419994e-8,
          0.017796472012105692, 1.4071175755687053e-8]
    riskt = 0.009093909446301915
    rett = 0.0015869556220572164
    @test isapprox(w4.weights, wt0, rtol = 5.0e-5)
    @test isapprox(r4, riskt0, rtol = 5.0e-6)
    @test isapprox(ret4, rett0, rtol = 5.0e-6)
    @test isapprox(w4.weights, wt)
    @test isapprox(r4, riskt)
    @test isapprox(ret4, rett)

    # Risk upper bound
    obj = MaxRet()
    rm = SLPM2(; target = rf, settings = RiskMeasureSettings(; scale = 2.0))
    rm.settings.ub = r1
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r1 ||
          abs(calc_risk(portfolio; type = :Trad2, rm = rm) - r1) < 5e-10

    rm = [[SLPM2(; target = rf), SLPM2(; target = rf)]]
    rm[1][2].settings.ub = r2
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm[1][2]) <= r2 ||
          abs(calc_risk(portfolio; type = :Trad2, rm = rm[1][2]) - r2) < 5e-10

    obj = SR(; rf = rf)
    rm = SLPM2(; target = rf, settings = RiskMeasureSettings(; scale = 2.0))
    rm.settings.ub = r1
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r1 ||
          abs(calc_risk(portfolio; type = :Trad2, rm = rm) - r1) < 1e-9

    rm = [[SLPM2(; target = rf), SLPM2(; target = rf)]]
    rm[1][1].settings.ub = r2
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm[1][2]) <= r2 ||
          abs(calc_risk(portfolio; type = :Trad2, rm = rm[1][2]) - r2) < 1e-10

    # Ret lower bound
    obj = MinRisk()
    rm = SLPM2(; target = rf, settings = RiskMeasureSettings(; scale = 2.0))
    portfolio.mu_l = ret1
    w5 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w5.weights) >= ret1

    rm = [[SLPM2(; target = rf), SLPM2(; target = rf)]]
    portfolio.mu_l = ret2
    w6 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w6.weights) >= ret2

    obj = SR(; rf = rf)
    rm = SLPM2(; target = rf, settings = RiskMeasureSettings(; scale = 2.0))
    portfolio.mu_l = ret1
    w7 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w7.weights) >= ret1

    rm = [[SLPM2(; target = rf), SLPM2(; target = rf)]]
    portfolio.mu_l = ret2
    w8 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w8.weights) >= ret2
end

@testset "WR settings" begin
    portfolio = Portfolio2(; prices = prices,
                           solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                            :params => Dict("verbose" => false))))
    asset_statistics2!(portfolio)

    obj = MinRisk()
    wt0 = [2.8612942682723194e-12, 0.22119703870515675, 2.828265759908966e-12,
           2.1208855227168895e-12, 3.697781891063451e-12, 3.24353226480368e-12,
           0.02918541183751788, 4.420452260557843e-12, 2.3374667530908414e-12,
           2.8919479333342058e-12, 0.5455165570312099, 1.4490684503326206e-12,
           1.9114786537154165e-12, 2.7506310060540026e-12, 3.640894035272413e-11,
           2.7909315847715066e-12, 1.694217734278189e-12, 3.798024068784819e-12,
           2.7258514165515688e-12, 0.20410099234818463]
    riskt0 = 0.03217605105120276
    rett0 = 0.0005011526784679896

    rm = WR2(; settings = RiskMeasureSettings(; scale = 2.0))
    w1 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r1 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret1 = dot(portfolio.mu, w1.weights)
    wt = [8.845836103523569e-12, 0.22119703964820553, 1.008777572455134e-11,
          9.023936172238157e-12, 2.7036274583341965e-11, 4.907010806919218e-12,
          0.029185411855919705, 9.380589563172966e-12, 7.978672948061364e-12,
          1.417495794019674e-11, 0.5455165564726506, 5.130324134567864e-12,
          3.0788105346959306e-12, 8.467302050439833e-12, 4.359556713964813e-10,
          8.183695280547008e-12, 1.2074582253455018e-12, 1.2166776481143779e-11,
          1.0054533157340265e-11, 0.20410099144754437]
    riskt = 0.032176051054907874
    rett = 0.0005011526780890267
    @test isapprox(w1.weights, wt0)
    @test isapprox(r1, riskt0)
    @test isapprox(ret1, rett0)
    @test isapprox(w1.weights, wt)
    @test isapprox(r1, riskt)
    @test isapprox(ret1, rett)

    obj = SR(; rf = rf)
    wt0 = [6.957399908772388e-9, 2.039271731107526e-8, 5.497898695084438e-9,
           1.1584017088731345e-8, 0.3797661371235164, 1.9162230097305403e-9,
           0.17660512608552742, 1.0666210782547244e-8, 1.0225338760635262e-8,
           0.04075088574289245, 0.05638221165264284, 2.089109162284139e-9,
           1.23279550928153e-9, 9.013331222315118e-9, 2.1778889815995123e-9,
           0.15854733523481268, 0.18794817199402036, 1.1268704949879534e-8,
           3.4599644297968083e-8, 4.545308055104026e-9]
    riskt0 = 0.04173382316607199
    rett0 = 0.0014131701721435356

    rm = WR2(; settings = RiskMeasureSettings(; scale = 2.0))
    w2 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r2 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret2 = dot(portfolio.mu, w2.weights)
    wt = [1.4359790974133853e-8, 4.4601310941731896e-8, 1.0973527016789228e-8,
          2.2757611493809034e-8, 0.37976615216872034, 3.5095229542488296e-9,
          0.17660511008948512, 2.5356317289911655e-8, 2.0545010295564603e-8,
          0.0407508367688423, 0.056382180959740104, 3.703877049386887e-9,
          2.074181254821784e-9, 1.8466263261921645e-8, 3.978425067250442e-9,
          0.15854728169942325, 0.18794810403065457, 2.6543710875879104e-8,
          1.2892383426602799e-7, 8.489751613190112e-9]
    riskt = 0.041733824947418285
    rett = 0.0014131701464381698
    @test isapprox(w2.weights, wt0, rtol = 5.0e-7)
    @test isapprox(r2, riskt0, rtol = 5.0e-8)
    @test isapprox(ret2, rett0, rtol = 5.0e-8)
    @test isapprox(w2.weights, wt)
    @test isapprox(r2, riskt)
    @test isapprox(ret2, rett)

    # Risk upper bound
    obj = MaxRet()
    rm = WR2(; settings = RiskMeasureSettings(; scale = 2.0))
    rm.settings.ub = r1
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r1

    obj = SR(; rf = rf)
    rm = WR2(; settings = RiskMeasureSettings(; scale = 2.0))
    rm.settings.ub = r1
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r1 ||
          abs(calc_risk(portfolio; type = :Trad2, rm = rm) - r1) < 1e-10

    # Ret lower bound
    obj = MinRisk()
    rm = WR2(; settings = RiskMeasureSettings(; scale = 2.0))
    portfolio.mu_l = ret1
    w3 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w3.weights) >= ret1

    obj = SR(; rf = rf)
    rm = WR2(; settings = RiskMeasureSettings(; scale = 2.0))
    portfolio.mu_l = ret1
    w4 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w4.weights) >= ret1
end

@testset "RG settings" begin
    portfolio = Portfolio2(; prices = prices,
                           solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                            :params => Dict("verbose" => false))))
    asset_statistics2!(portfolio)

    obj = MinRisk()
    wt0 = [3.4835106277243952e-12, 0.12689181906634667, 6.919061275326071e-12,
           0.2504942755637169, 1.6084529746393874e-11, 1.4834264958811167e-11,
           1.7580530855156377e-12, 0.1053793586345684, 3.0155464926527536e-12,
           0.01026273777456012, 0.39071437611839677, 1.3196103594515992e-12,
           4.2161735386690386e-14, 4.265414596070901e-12, 0.013789122525542109,
           4.212243490314627e-11, 1.869748261234954e-11, 1.0033734598934388e-11,
           4.125432454379667e-12, 0.10246831019016761]
    riskt0 = 0.06215928170399987
    rett0 = 0.0005126268312684577

    rm = RG2(; settings = RiskMeasureSettings(; scale = 2.0))
    w1 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r1 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret1 = dot(portfolio.mu, w1.weights)
    wt = [9.352397414179783e-12, 0.12689181824759477, 2.2029288056634106e-11,
          0.25049427304023353, 9.563307103750518e-11, 7.448614457002062e-11,
          7.256594026628282e-12, 0.10537935700244638, 1.6363194661272212e-12,
          0.010262744178469656, 0.3907143796567699, 9.358948411243617e-12,
          1.772311334920759e-11, 1.4852936118325254e-11, 0.013789123195856956,
          2.3911086155376745e-10, 8.91833520024116e-11, 4.177666344792136e-11,
          7.88924594585403e-12, 0.10246830404833977]
    riskt = 0.062159281714750256
    rett = 0.0005126268272393026
    @test isapprox(w1.weights, wt0, rtol = 5.0e-8)
    @test isapprox(r1, riskt0)
    @test isapprox(ret1, rett0)
    @test isapprox(w1.weights, wt)
    @test isapprox(r1, riskt)
    @test isapprox(ret1, rett)

    obj = SR(; rf = rf)
    wt0 = [6.09724566157312e-10, 1.2600001874338616e-9, 1.1376809473114097e-9,
           0.3052627178260572, 0.25494610585828753, 1.8310456179966306e-10,
           0.09576793051248834, 3.7212437966398855e-9, 1.284058570384776e-9,
           2.9231767008643053e-9, 8.664801052496888e-10, 2.2919525949753356e-10,
           9.024587888396662e-11, 4.047257690673743e-10, 8.93417076405552e-11,
           0.12660213909219864, 0.1910488629796444, 1.5541667982439667e-9,
           0.02225468123935576, 0.0041175481388230385]
    riskt0 = 0.08445623506377935
    rett0 = 0.0012690611588731012

    rm = RG2(; settings = RiskMeasureSettings(; scale = 2.0))
    w2 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r2 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret2 = dot(portfolio.mu, w2.weights)
    wt = [1.9186184378951531e-10, 4.1666419611390246e-10, 3.511235237907537e-10,
          0.30526275516106566, 0.25494606948317644, 4.2094209049480684e-11,
          0.09576792399737531, 1.9610366442325496e-9, 4.012091480090772e-10,
          1.3263741799530412e-9, 2.409260936031573e-10, 6.510401565462342e-11,
          1.143360838330506e-11, 1.2972124220616567e-10, 9.446355170047803e-12,
          0.12660221848662798, 0.1910492804390715, 5.236180388098645e-10,
          0.022254317257956593, 0.004117429504113589]
    riskt = 0.08445624038422006
    rett = 0.0012690612327898097
    @test isapprox(w2.weights, wt0, rtol = 5.0e-6)
    @test isapprox(r2, riskt0, rtol = 1.0e-7)
    @test isapprox(ret2, rett0, rtol = 1.0e-7)
    @test isapprox(w2.weights, wt)
    @test isapprox(r2, riskt)
    @test isapprox(ret2, rett)

    # Risk upper bound
    obj = MaxRet()
    rm = RG2(; settings = RiskMeasureSettings(; scale = 2.0))
    rm.settings.ub = r1
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r1 ||
          abs(calc_risk(portfolio; type = :Trad2, rm = rm) - r1) < 1e-10

    obj = SR(; rf = rf)
    rm = RG2(; settings = RiskMeasureSettings(; scale = 2.0))
    rm.settings.ub = r1
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r1 ||
          abs(calc_risk(portfolio; type = :Trad2, rm = rm) - r1) < 1e-9

    # Ret lower bound
    obj = MinRisk()
    rm = RG2(; settings = RiskMeasureSettings(; scale = 2.0))
    portfolio.mu_l = ret1
    w3 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w3.weights) >= ret1

    obj = SR(; rf = rf)
    rm = RG2(; settings = RiskMeasureSettings(; scale = 2.0))
    portfolio.mu_l = ret1
    w4 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w4.weights) >= ret1
end

@testset "CVaR vec" begin
    portfolio = Portfolio2(; prices = prices,
                           solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                            :params => Dict("verbose" => false))))
    asset_statistics2!(portfolio)

    obj = MinRisk()
    wt0 = [9.965769176302831e-11, 0.04242033378148941, 9.902259604479418e-11,
           2.585550936025974e-10, 0.007574028506215674, 1.1340405766435789e-10,
           1.3814642470526227e-11, 0.09464947974750273, 4.637745432335755e-11,
           6.484701166044592e-11, 0.3040110652312709, 5.940889071027648e-11,
           3.420745138676034e-11, 0.06564166947730173, 9.544192184784114e-11,
           0.029371611149186894, 1.241093002048221e-10, 0.36631011287979914,
           5.953639120278758e-11, 0.09002169815885094]
    riskt0 = 0.01704950212555889
    rett0 = 0.0003860990591135937

    rm = CVaR2(; settings = RiskMeasureSettings(; scale = 2.0))
    w1 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r1 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret1 = dot(portfolio.mu, w1.weights)
    wt = [9.039113116707146e-12, 0.042420332743419456, 1.0023954281481145e-11,
          2.1672854590187125e-11, 0.00757402788043966, 1.0078660053389585e-11,
          1.8709032936343172e-12, 0.09464947864323794, 4.741082204917204e-12,
          6.176307280995208e-12, 0.30401106573635855, 5.484012104258149e-12,
          3.239252777004022e-12, 0.06564167088258994, 5.8615621940737234e-12,
          0.029371611142210794, 1.2268140738879004e-11, 0.36631011301179545,
          5.969284110086278e-12, 0.09002169986352294]
    riskt = 0.017049502122244473
    rett = 0.0003860990576923258
    @test isapprox(w1.weights, wt0)
    @test isapprox(r1, riskt0)
    @test isapprox(ret1, rett0)
    @test isapprox(w1.weights, wt)
    @test isapprox(r1, riskt)
    @test isapprox(ret1, rett)

    rm = [[CVaR2(), CVaR2()]]
    w2 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r2 = calc_risk(portfolio; type = :Trad2, rm = rm[1][1])
    ret2 = dot(portfolio.mu, w2.weights)
    wt = [6.766537894788691e-11, 0.042420332799196256, 6.967389797629564e-11,
          2.1360695921491254e-10, 0.0075740287628112725, 8.19623669734998e-11,
          1.5113072840935442e-12, 0.09464947986321764, 2.81310170882108e-11,
          4.327764716110422e-11, 0.30401106764498176, 3.7179405135745806e-11,
          1.571027388599003e-11, 0.06564166770873361, 7.96462088600085e-11,
          0.029371611131516446, 9.001224179664519e-11, 0.3663101122441516,
          3.796007805345114e-11, 0.09002169907905451]
    riskt = 0.01704950212496479
    rett = 0.0003860990597740532
    @test isapprox(w2.weights, wt0)
    @test isapprox(r2, riskt0)
    @test isapprox(ret2, rett0)
    @test isapprox(w2.weights, wt)
    @test isapprox(r2, riskt)
    @test isapprox(ret2, rett)

    obj = SR(; rf = rf)
    wt0 = [2.305962223730381e-9, 3.061529980299523e-9, 3.6226755773356135e-9,
           1.968988878444111e-9, 0.562845489616387, 6.289605285168684e-10,
           0.044341929816432854, 5.465596947274736e-9, 3.128822366888805e-9,
           1.6003971393612084e-9, 4.52394176361636e-9, 5.75356193927518e-10,
           3.1728380155852195e-10, 1.240519587265295e-9, 3.422838872379099e-10,
           0.20959173183485763, 0.18322079783245407, 6.034806498955341e-9,
           1.1803331196573864e-8, 4.279412029260546e-9]
    riskt0 = 0.03005421217653932
    rett0 = 0.0015191213711409513

    rm = CVaR2(; settings = RiskMeasureSettings(; scale = 2.0))
    w3 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r3 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret3 = dot(portfolio.mu, w3.weights)
    wt = [3.444383766463055e-9, 4.681173768158096e-9, 5.383635423471024e-9,
          2.9506547246714714e-9, 0.562845458681874, 7.747566327165435e-10,
          0.04434192222277232, 8.27974795943376e-9, 5.136873560884811e-9,
          2.3563335954686833e-9, 6.933886486199326e-9, 7.083559915716843e-10,
          2.7014332763187937e-10, 1.7583334343556281e-9, 3.2277175403322157e-10,
          0.20959175250003312, 0.18322078610659337, 9.288019586921007e-9,
          2.160467846262691e-8, 6.594978823541082e-9]
    riskt = 0.03005421146854565
    rett = 0.0015191213329156926
    @test isapprox(w3.weights, wt0, rtol = 1.0e-7)
    @test isapprox(r3, riskt0, rtol = 5.0e-8)
    @test isapprox(ret3, rett0, rtol = 5.0e-8)
    @test isapprox(w3.weights, wt)
    @test isapprox(r3, riskt)
    @test isapprox(ret3, rett)

    rm = [[CVaR2(), CVaR2()]]
    w4 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r4 = calc_risk(portfolio; type = :Trad2, rm = rm[1][1])
    ret4 = dot(portfolio.mu, w4.weights)
    wt = [2.01075695009106e-9, 2.711928519041927e-9, 3.231881995803488e-9,
          1.6761038821670713e-9, 0.5628461675597376, 3.676056299307345e-10,
          0.044341983367491865, 5.350560682418396e-9, 2.9509107325912622e-9,
          1.3168086570787986e-9, 4.370920940444675e-9, 3.2323899039202986e-10,
          6.220548863283214e-11, 9.708605488937601e-10, 8.586434059514893e-11,
          0.20959190235690242, 0.18321989840550656, 5.9724387387015885e-9,
          1.2809706071570516e-8, 4.0985694976215465e-9]
    riskt = 0.03005422530208694
    rett = 0.0015191219778014082
    @test isapprox(w4.weights, wt0, rtol = 5.0e-6)
    @test isapprox(r4, riskt0, rtol = 5.0e-7)
    @test isapprox(ret4, rett0, rtol = 5.0e-7)
    @test isapprox(w4.weights, wt)
    @test isapprox(r4, riskt)
    @test isapprox(ret4, rett)

    # Risk upper bound
    obj = MaxRet()
    rm = CVaR2(; settings = RiskMeasureSettings(; scale = 2.0))
    rm.settings.ub = r1
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r1 ||
          abs(calc_risk(portfolio; type = :Trad2, rm = rm) - r1) < 1e-10

    rm = [[CVaR2(), CVaR2()]]
    rm[1][1].settings.ub = r2
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm[1][1]) <= r2 ||
          abs(calc_risk(portfolio; type = :Trad2, rm = rm[1][1]) - r2) < 1e-10

    obj = SR(; rf = rf)
    rm = CVaR2(; settings = RiskMeasureSettings(; scale = 2.0))
    rm.settings.ub = r1
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r1 ||
          abs(calc_risk(portfolio; type = :Trad2, rm = rm) - r1) < 1e-10

    rm = [[CVaR2(), CVaR2()]]
    rm[1][1].settings.ub = r2
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm[1][1]) <= r2 ||
          abs(calc_risk(portfolio; type = :Trad2, rm = rm[1][1]) - r2) < 5e-10

    # Ret lower bound
    obj = MinRisk()
    rm = CVaR2(; settings = RiskMeasureSettings(; scale = 2.0))
    rm.settings.ub = Inf
    portfolio.mu_l = ret1
    w5 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w5.weights) >= ret1

    rm = [[CVaR2(), CVaR2()]]
    portfolio.mu_l = ret2
    w6 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w6.weights) >= ret2

    obj = SR(; rf = rf)
    rm = CVaR2(; settings = RiskMeasureSettings(; scale = 2.0))
    portfolio.mu_l = ret1
    w7 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w7.weights) >= ret1

    rm = [[CVaR2(), CVaR2()]]
    portfolio.mu_l = ret2
    w8 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w8.weights) >= ret2

    obj = MinRisk()
    wt0 = [9.965769176302831e-11, 0.04242033378148941, 9.902259604479418e-11,
           2.585550936025974e-10, 0.007574028506215674, 1.1340405766435789e-10,
           1.3814642470526227e-11, 0.09464947974750273, 4.637745432335755e-11,
           6.484701166044592e-11, 0.3040110652312709, 5.940889071027648e-11,
           3.420745138676034e-11, 0.06564166947730173, 9.544192184784114e-11,
           0.029371611149186894, 1.241093002048221e-10, 0.36631011287979914,
           5.953639120278758e-11, 0.09002169815885094]
    riskt0 = 0.01704950212555889
    rett0 = 0.0003860990591135937
    wt1 = [1.24800727271117e-9, 0.0031953680013384602, 0.05121723147271688,
           0.03892554835725213, 0.07798435887801634, 2.4531915465534646e-10,
           0.01180294635260589, 0.13459628736139845, 4.098470671003605e-10,
           0.026312591423892855, 0.20350781672620089, 4.85493902933692e-11,
           5.881796544290361e-11, 0.06481644902677358, 7.228076721322575e-11,
           0.033078859159137594, 0.06402843787550846, 0.15619687739656893,
           4.931193895559528e-9, 0.1343372209545739]
    riskt1 = 0.002583003989695373
    rett1 = 0.0006032530367580909

    rm = [[CVaR2(), CVaR2(; alpha = 0.75)]]
    w9 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r9 = calc_risk(portfolio; type = :Trad2, rm = rm[1][1])
    ret9 = dot(portfolio.mu, w9.weights)
    wt = [3.9922885670943495e-11, 0.030425596544942, 5.4468155257367006e-11,
          0.006558936627595747, 0.03690984338233109, 3.177251961522887e-11,
          1.520238704109624e-12, 0.11048861990131845, 1.488042060838733e-11,
          1.9921891193477953e-11, 0.269162194704224, 1.4132073165301486e-11,
          4.636343021525174e-12, 0.045417445113116454, 3.5070799021628147e-11,
          0.0309689986422629, 6.629204406905901e-11, 0.3651087267454197,
          2.2927416896953245e-11, 0.10495963803324489]
    riskt = 0.017084509125516244
    rett = 0.0004430763018669782
    @test !isapprox(w9.weights, wt0)
    @test !isapprox(r9, riskt0)
    @test !isapprox(ret9, rett0)
    @test !isapprox(w9.weights, wt1)
    @test !isapprox(r9, riskt1)
    @test !isapprox(ret9, rett1)
    @test isapprox(w9.weights, wt)
    @test isapprox(r9, riskt)
    @test isapprox(ret9, rett)
end

@testset "RCVaR vec" begin
    portfolio = Portfolio2(; prices = prices,
                           solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                            :params => Dict("verbose" => false))))
    asset_statistics2!(portfolio)

    obj = MinRisk()
    wt0 = [0.022526773522155673, 0.02205967648382663, 2.319996990805667e-11,
           0.029966893048161407, 0.006773557016985066, 0.021160245217902482,
           5.016724411235469e-13, 0.11191320782878157, 2.6370102192581807e-12,
           1.2019874588666696e-11, 0.31895383693063833, 2.262605967359396e-12,
           1.135073650138578e-12, 0.09089853669776103, 7.539013998246915e-13,
           0.045111723951082434, 3.636547744318311e-11, 0.23185661850827086,
           3.242869688736363e-12, 0.09877893071231604]
    riskt0 = 0.03439845483008025
    rett0 = 0.00038005938396668074

    rm = RCVaR2(; settings = RiskMeasureSettings(; scale = 2.0))
    w1 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r1 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret1 = dot(portfolio.mu, w1.weights)
    wt = [0.022526773640711432, 0.022059676336958167, 8.938452301128973e-12,
          0.029966893035187715, 0.00677355712531329, 0.021160245126991578,
          4.3190754830541274e-14, 0.11191320780277068, 8.03314283567928e-13,
          3.927142579247319e-12, 0.31895383706573777, 6.177190594675038e-13,
          1.808344216177238e-13, 0.09089853660073993, 1.449393920033099e-13,
          0.04511172395664449, 1.3137744337484285e-11, 0.23185661859564424,
          1.0387625845466213e-12, 0.09877893068446858]
    riskt = 0.03439845482982849
    rett = 0.00038005938418307687
    @test isapprox(w1.weights, wt0)
    @test isapprox(r1, riskt0)
    @test isapprox(ret1, rett0)
    @test isapprox(w1.weights, wt)
    @test isapprox(r1, riskt)
    @test isapprox(ret1, rett)

    rm = [[RCVaR2(), RCVaR2()]]
    w2 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r2 = calc_risk(portfolio; type = :Trad2, rm = rm[1][1])
    ret2 = dot(portfolio.mu, w2.weights)
    wt = [0.02252677314188004, 0.022059676656782594, 2.6823401769473206e-11,
          0.029966893154714246, 0.006773556915623245, 0.0211602451562918,
          1.0024625362393218e-12, 0.11191320780941605, 1.7215677936067835e-12,
          1.3083084957964856e-11, 0.3189538370403967, 1.1508209206841141e-12,
          2.278938834875411e-13, 0.09089853658130546, 5.510129758693874e-13,
          0.04511172388241891, 4.793512444410647e-11, 0.23185661883791073,
          2.5179377145817183e-12, 0.09877893072824685]
    riskt = 0.03439845483021759
    rett = 0.0003800593838823556
    @test isapprox(w2.weights, wt0)
    @test isapprox(r2, riskt0)
    @test isapprox(ret2, rett0)
    @test isapprox(w2.weights, wt)
    @test isapprox(r2, riskt)
    @test isapprox(ret2, rett)

    obj = SR(; rf = rf)
    wt0 = [4.9839432322477e-10, 8.448693289397689e-10, 1.173858895147354e-9,
           9.375514999679668e-10, 0.573440952697266, 7.451535889801865e-11,
           0.05607344635024722, 1.8370189606252853e-9, 8.256113918546924e-10,
           4.3173297057509325e-10, 1.0216619066483312e-9, 4.6589215379003155e-11,
           6.2787877132137955e-12, 2.0536944124409375e-10, 1.550936856433007e-12,
           0.14189709250803045, 0.2285884636440746, 1.020124064623182e-9,
           3.463719056083844e-8, 1.2380641433681093e-9]
    riskt0 = 0.0642632537835233
    rett0 = 0.0015273688513609762

    rm = RCVaR2(; settings = RiskMeasureSettings(; scale = 2.0))
    w3 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r3 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret3 = dot(portfolio.mu, w3.weights)
    wt = [8.343316180258173e-11, 1.496491747142568e-10, 2.1768238664130146e-10,
          1.698811473553839e-10, 0.573440992341175, 3.208233622765092e-14,
          0.056073452248728055, 3.4276599595629554e-10, 1.460784851394298e-10,
          7.062066256789735e-11, 1.8667511086255285e-10, 5.503441681425758e-12,
          1.3461900170168065e-11, 2.5861749274836016e-11, 1.492004653857825e-11,
          0.14189712780051628, 0.2285884209516441, 1.871280643112202e-10,
          4.817191473748767e-9, 2.2705189368060486e-10]
    riskt = 0.06426325594594744
    rett = 0.0015273689006402935
    @test isapprox(w3.weights, wt0, rtol = 5.0e-7)
    @test isapprox(r3, riskt0, rtol = 5.0e-8)
    @test isapprox(ret3, rett0, rtol = 5.0e-8)
    @test isapprox(w3.weights, wt)
    @test isapprox(r3, riskt)
    @test isapprox(ret3, rett)

    rm = [[RCVaR2(), RCVaR2()]]
    w4 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r4 = calc_risk(portfolio; type = :Trad2, rm = rm[1][1])
    ret4 = dot(portfolio.mu, w4.weights)
    wt = [4.92144376503163e-11, 9.290218071961544e-11, 1.185480711900962e-10,
          9.897815143178041e-11, 0.5734410119998661, 6.383567558330417e-12,
          0.05607344516546397, 2.1998976815919417e-10, 8.660050377597577e-11,
          3.9036065395056645e-11, 1.1458038022187725e-10, 7.7671052270593e-12,
          1.2061372005987416e-11, 1.1576413727845234e-11, 1.4674607053811892e-11,
          0.14189710274873424, 0.22858843394418632, 1.1098823396573785e-10,
          5.017583250056151e-9, 1.408653957301166e-10]
    riskt = 0.06426325621576456
    rett = 0.0015273689067783166
    @test isapprox(w4.weights, wt0, rtol = 5.0e-7)
    @test isapprox(r4, riskt0, rtol = 5.0e-8)
    @test isapprox(ret4, rett0, rtol = 5.0e-8)
    @test isapprox(w4.weights, wt)
    @test isapprox(r4, riskt)
    @test isapprox(ret4, rett)

    # Risk upper bound
    obj = MaxRet()
    rm = RCVaR2(; settings = RiskMeasureSettings(; scale = 2.0))
    rm.settings.ub = r1
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r1 ||
          abs(calc_risk(portfolio; type = :Trad2, rm = rm) - r1) < 1e-10

    rm = [[RCVaR2(), RCVaR2()]]
    rm[1][1].settings.ub = r2
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm[1][1]) <= r2 ||
          abs(calc_risk(portfolio; type = :Trad2, rm = rm[1][1]) - r2) < 5e-10

    obj = SR(; rf = rf)
    rm = RCVaR2(; settings = RiskMeasureSettings(; scale = 2.0))
    rm.settings.ub = r1
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r1 ||
          abs(calc_risk(portfolio; type = :Trad2, rm = rm) - r1) < 1e-10

    rm = [[RCVaR2(), RCVaR2()]]
    rm[1][1].settings.ub = r2
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm[1][1]) <= r2 ||
          abs(calc_risk(portfolio; type = :Trad2, rm = rm[1][1]) - r2) < 5e-9

    # Ret lower bound
    obj = MinRisk()
    rm = RCVaR2(; settings = RiskMeasureSettings(; scale = 2.0))
    rm.settings.ub = Inf
    portfolio.mu_l = ret1
    w5 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w5.weights) >= ret1

    rm = [[RCVaR2(), RCVaR2()]]
    portfolio.mu_l = ret2
    w6 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w6.weights) >= ret2

    obj = SR(; rf = rf)
    rm = RCVaR2(; settings = RiskMeasureSettings(; scale = 2.0))
    portfolio.mu_l = ret1
    w7 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w7.weights) >= ret1

    rm = [[RCVaR2(), RCVaR2()]]
    portfolio.mu_l = ret2
    w8 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w8.weights) >= ret2

    obj = MinRisk()
    wt0 = [0.022526773522155673, 0.02205967648382663, 2.319996990805667e-11,
           0.029966893048161407, 0.006773557016985066, 0.021160245217902482,
           5.016724411235469e-13, 0.11191320782878157, 2.6370102192581807e-12,
           1.2019874588666696e-11, 0.31895383693063833, 2.262605967359396e-12,
           1.135073650138578e-12, 0.09089853669776103, 7.539013998246915e-13,
           0.045111723951082434, 3.636547744318311e-11, 0.23185661850827086,
           3.242869688736363e-12, 0.09877893071231604]
    riskt0 = 0.03439845483008025
    rett0 = 0.00038005938396668074
    wt1 = [0.01590329676292731, 0.01465252459266188, 0.030008643239972217,
           0.02098287397708189, 0.015112683503680367, 0.017363189384944563,
           2.751262227188605e-12, 0.1430733590868079, 3.3402944727161174e-12,
           0.008948752482605552, 0.2673754466933422, 9.550573196820747e-13,
           1.012181954005461e-12, 0.14438819043344578, 1.6615965867537844e-11,
           0.012604756017322135, 0.02186818922796837, 0.16489946603400776,
           5.016676674735644e-12, 0.12281862853354067]
    riskt1 = 0.006200586303619828
    rett1 = 0.0003800593838902877

    rm = [[RCVaR2(), RCVaR2(; alpha = 0.75, beta = 0.75)]]
    w9 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r9 = calc_risk(portfolio; type = :Trad2, rm = rm[1][1])
    ret9 = dot(portfolio.mu, w9.weights)
    wt = [0.023921654687474096, 0.021361003970205204, 1.89630785414371e-10,
          0.028653888646049154, 0.010423847101409285, 0.026411355825713634,
          1.0311054581713778e-12, 0.1153769742490529, 4.130425762090548e-12,
          2.238644898704979e-11, 0.31222807465278246, 2.194028866169219e-12,
          2.943418905472722e-13, 0.09049709748637362, 5.894198002252489e-13,
          0.0447708552060712, 1.6172080291765744e-10, 0.22800461298404587,
          5.3162828407181034e-12, 0.0983506348035288]
    riskt = 0.034402085686936854
    rett = 0.0003800593841288208
    @test !isapprox(w9.weights, wt0)
    @test !isapprox(r9, riskt0)
    @test isapprox(ret9, rett0)
    @test !isapprox(w9.weights, wt1)
    @test !isapprox(r9, riskt1)
    @test isapprox(ret9, rett1)
    @test isapprox(w9.weights, wt)
    @test isapprox(r9, riskt)
    @test isapprox(ret9, rett)
end

@testset "EVaR vec" begin
    portfolio = Portfolio2(; prices = prices,
                           solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                            :params => Dict("verbose" => false,
                                                                            "max_step_fraction" => 0.75))))
    asset_statistics2!(portfolio)

    obj = MinRisk()
    wt0 = [1.2500112838486011e-8, 0.15679318740948175, 1.0361131210275206e-8,
           0.01670757435974688, 1.501287503554962e-8, 6.061816596694208e-8,
           0.014452439886462113, 0.15570664400078943, 9.522408711497533e-9,
           1.059085220479153e-8, 0.452447219917494, 8.305434093731495e-9,
           1.2081476879327763e-8, 2.3952270923291378e-8, 0.004794389308245565,
           1.7142647790367886e-7, 0.01841950032750946, 1.9211685081872636e-7,
           1.2233795563491733e-8, 0.18067850606841873]
    riskt0 = 0.024507972823062964
    rett0 = 0.00046038550243244597

    rm = EVaR2(; settings = RiskMeasureSettings(; scale = 2.0))
    w1 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r1 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret1 = dot(portfolio.mu, w1.weights)
    wt = [1.8441527671949737e-8, 0.15678963327741566, 1.539241427355876e-8,
          0.01670907677535256, 2.1850309377019758e-8, 8.853820413355456e-8,
          0.014452766691861008, 0.1557073793762452, 1.4234518846301731e-8,
          1.5675159966159678e-8, 0.45244617608937443, 1.234154972997298e-8,
          1.7658533220335193e-8, 3.4307770213779566e-8, 0.004793921880979019,
          2.4825608613098985e-7, 0.01841892399524742, 2.9676248299537406e-7,
          1.8136029042261154e-8, 0.18068132031893921]
    riskt = 0.02450797316477554
    rett = 0.00046038661273523995
    @test isapprox(w1.weights, wt0, rtol = 1.0e-5)
    @test isapprox(r1, riskt0)
    @test isapprox(ret1, rett0, rtol = 5.0e-6)
    @test isapprox(w1.weights, wt, rtol = 5.0e-6)
    @test isapprox(r1, riskt)
    @test isapprox(ret1, rett, rtol = 1.0e-7)

    rm = [[EVaR2(), EVaR2()]]
    w2 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r2 = calc_risk(portfolio; type = :Trad2, rm = rm[1][1])
    ret2 = dot(portfolio.mu, w2.weights)
    wt = [6.570316448702579e-9, 0.15679009674996885, 5.406277443034958e-9,
          0.016708948798958748, 7.964909261054618e-9, 3.283395409927604e-8,
          0.014451195748914906, 0.15570714273972625, 5.021500907220597e-9,
          5.562864899760625e-9, 0.4524480293888993, 4.371830795915324e-9,
          6.3793118346395444e-9, 1.3294682356091756e-8, 0.004794185620271175,
          1.0087949849792273e-7, 0.018420628563446687, 1.408033478312917e-7,
          6.4194578330347835e-9, 0.18067943688186178]
    riskt = 0.024507972507204205
    rett = 0.0004603842983759555
    @test isapprox(w2.weights, wt0, rtol = 1.0e-5)
    @test isapprox(r2, riskt0)
    @test isapprox(ret2, rett0, rtol = 1.0e-5)
    @test isapprox(w2.weights, wt, rtol = 1.0e-5)
    @test isapprox(r2, riskt)
    @test isapprox(ret2, rett, rtol = 5.0e-6)

    obj = SR(; rf = rf)
    wt0 = [1.0750748140777434e-8, 3.269490337304986e-8, 1.1161941451849754e-8,
           1.3795466025857643e-8, 0.5351874067614019, 2.6718249477546367e-9,
           0.1390764348877217, 1.41282558079161e-8, 1.0656060597300996e-8,
           7.83717309959956e-9, 1.794801260303159e-8, 2.6229370477942236e-9,
           1.8308405319956406e-9, 6.011246604979923e-9, 1.9381716976717685e-9,
           0.18358697053484188, 0.14214899271554252, 1.7344741890623557e-8,
           3.394097823954422e-8, 9.767190110912097e-9]
    riskt0 = 0.03754976868195822
    rett0 = 0.0015728602397846448

    rm = EVaR2(; settings = RiskMeasureSettings(; scale = 2.0))
    w3 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r3 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret3 = dot(portfolio.mu, w3.weights)
    wt = [1.0522315051828238e-8, 3.216594965328698e-8, 1.0917886168415742e-8,
          1.3472730969756074e-8, 0.5351883278837632, 2.63105137211939e-9,
          0.13907379409736423, 1.3852500641187822e-8, 1.042434164848934e-8,
          7.680810551644923e-9, 1.763631480248168e-8, 2.584483740964515e-9,
          1.8037930584769864e-9, 5.888264850076767e-9, 1.909828395787896e-9,
          0.18358571813954144, 0.14215196861546997, 1.6996127968851362e-8,
          3.3224046351623283e-8, 9.553415963779003e-9]
    riskt = 0.037549725210212895
    rett = 0.0015728585717447157
    @test isapprox(w3.weights, wt0, rtol = 1.0e-5)
    @test isapprox(r3, riskt0, rtol = 5.0e-6)
    @test isapprox(ret3, rett0, rtol = 5.0e-6)
    @test isapprox(w3.weights, wt, rtol = 1.0e-6)
    @test isapprox(r3, riskt, rtol = 5.0e-8)
    @test isapprox(ret3, rett, rtol = 5.0e-8)

    rm = [[EVaR2(), EVaR2()]]
    w4 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r4 = calc_risk(portfolio; type = :Trad2, rm = rm[1][1])
    ret4 = dot(portfolio.mu, w4.weights)
    wt = [7.408129309103847e-9, 2.2226353224832677e-8, 7.662541118677215e-9,
          9.508961911843281e-9, 0.5351802096343967, 1.832774579865685e-9,
          0.13907850307936856, 9.849118333123395e-9, 7.404826969572652e-9,
          5.521915986669233e-9, 1.2683235516209414e-8, 1.793266887925894e-9,
          1.249847569505163e-9, 4.188639800236319e-9, 1.33210852749895e-9,
          0.18358384144359496, 0.1421573109272211, 1.2036422134317067e-8,
          2.354195614586394e-8, 6.6753205478192035e-9]
    riskt = 0.037549654583478594
    rett = 0.0015728558698142113
    @test isapprox(w4.weights, wt0, rtol = 5.0e-5)
    @test isapprox(r4, riskt0, rtol = 5.0e-6)
    @test isapprox(ret4, rett0, rtol = 5.0e-6)
    @test isapprox(w4.weights, wt, rtol = 5.0e-7)
    @test isapprox(r4, riskt, rtol = 5.0e-8)
    @test isapprox(ret4, rett, rtol = 5.0e-8)

    # Risk upper bound
    obj = MaxRet()
    rm = EVaR2(; settings = RiskMeasureSettings(; scale = 2.0))
    rm.settings.ub = r1
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r1 ||
          abs(calc_risk(portfolio; type = :Trad2, rm = rm) - r1) < 3e-6

    rm = [[EVaR2(), EVaR2()]]
    rm[1][1].settings.ub = r2
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm[1][1]) <= r2 ||
          abs(calc_risk(portfolio; type = :Trad2, rm = rm[1][1]) - r2) < 1e-7

    obj = SR(; rf = rf)
    rm = EVaR2(; settings = RiskMeasureSettings(; scale = 2.0))
    rm.settings.ub = r1 * 1.000001
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r1 * 1.000001 ||
          abs(calc_risk(portfolio; type = :Trad2, rm = rm) - r1 * 1.000001) < 5e-8

    rm = [[EVaR2(), EVaR2()]]
    rm[1][1].settings.ub = r2
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm[1][1]) <= r2 ||
          abs(calc_risk(portfolio; type = :Trad2, rm = rm[1][1]) - r2) < 1e-6

    # Ret lower bound
    obj = MinRisk()
    rm = EVaR2(; settings = RiskMeasureSettings(; scale = 2.0))
    rm.settings.ub = Inf
    portfolio.mu_l = ret1
    w5 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w5.weights) >= ret1

    rm = [[EVaR2(), EVaR2()]]
    portfolio.mu_l = ret2
    w6 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w6.weights) >= ret2

    obj = SR(; rf = rf)
    rm = EVaR2(; settings = RiskMeasureSettings(; scale = 2.0))
    portfolio.mu_l = ret1
    w7 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w7.weights) >= ret1

    rm = [[EVaR2(), EVaR2()]]
    portfolio.mu_l = ret2
    w8 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w8.weights) >= ret2

    obj = MinRisk()
    wt0 = [1.2500112838486011e-8, 0.15679318740948175, 1.0361131210275206e-8,
           0.01670757435974688, 1.501287503554962e-8, 6.061816596694208e-8,
           0.014452439886462113, 0.15570664400078943, 9.522408711497533e-9,
           1.059085220479153e-8, 0.452447219917494, 8.305434093731495e-9,
           1.2081476879327763e-8, 2.3952270923291378e-8, 0.004794389308245565,
           1.7142647790367886e-7, 0.01841950032750946, 1.9211685081872636e-7,
           1.2233795563491733e-8, 0.18067850606841873]
    riskt0 = 0.024507972823062964
    rett0 = 0.00046038550243244597
    wt1 = [2.693454399344831e-7, 0.06778951325183796, 2.998462681307733e-7,
           0.019151949797203326, 0.03790168847064121, 1.6004314356972936e-7,
           3.0355097010339537e-7, 0.12652184881356043, 1.8592295403425746e-7,
           2.244311228292717e-7, 0.3266552053601428, 7.353131910367664e-8,
           5.180091016405146e-8, 0.03437460177203133, 1.342453537898445e-7,
           0.029150478773176166, 0.04836586492635733, 0.1784915067032569,
           3.10539984332275e-7, 0.13159532887432668]
    riskt1 = 0.005938598537038723
    rett1 = 0.000497886452599781

    rm = [[EVaR2(), EVaR2(; alpha = 0.75)]]
    w9 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r9 = calc_risk(portfolio; type = :Trad2, rm = rm[1][1])
    ret9 = dot(portfolio.mu, w9.weights)
    wt = [1.720023847602269e-8, 0.12640956928663982, 1.515595196248127e-8,
          0.025684825897665967, 3.528211797696264e-8, 4.011757964162281e-8,
          0.0074388492154825415, 0.14684958279284363, 1.2574039834912915e-8,
          1.4472196554338671e-8, 0.4363570237552876, 8.62694416616101e-9,
          9.418474842872053e-9, 3.502189010834034e-8, 2.8721986541908246e-7,
          0.007078604027300184, 0.04096220880329674, 0.028899498269273783,
          1.7363358390649193e-8, 0.18031934549955236]
    riskt = 0.024558345252505446
    rett = 0.00047267221858361203
    @test !isapprox(w9.weights, wt0)
    @test !isapprox(r9, riskt0)
    @test !isapprox(ret9, rett0)
    @test !isapprox(w9.weights, wt1)
    @test !isapprox(r9, riskt1)
    @test !isapprox(ret9, rett1)
    @test isapprox(w9.weights, wt, rtol = 5.0e-5)
    @test isapprox(r9, riskt, rtol = 5.0e-7)
    @test isapprox(ret9, rett, rtol = 5.0e-6)
end

@testset "RVaR vec" begin
    portfolio = Portfolio2(; prices = prices,
                           solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                            :params => Dict("verbose" => false,
                                                                            "max_step_fraction" => 0.75))))
    asset_statistics2!(portfolio)

    obj = MinRisk()
    wt0 = [5.102262457628692e-9, 0.21104494400160803, 5.341766086210288e-9,
           2.5458382238901392e-8, 1.697696472902229e-8, 7.287515616039478e-9,
           0.03667031714382797, 0.061041476346139684, 4.093926758298615e-9,
           4.303160140655642e-9, 0.49353591974074, 2.1672264824822902e-9,
           3.926886474939328e-9, 4.083625597792755e-9, 1.043237724356759e-8,
           1.1621198331723714e-8, 2.5232405645111758e-8, 1.1835541180026409e-8,
           4.9679012600678e-9, 0.19770719993654404]
    riskt0 = 0.028298069755304314
    rett0 = 0.000508233446626652

    rm = RVaR2(; settings = RiskMeasureSettings(; scale = 2.0))
    w1 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r1 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret1 = dot(portfolio.mu, w1.weights)
    wt = [9.913473240823125e-9, 0.21103953264400618, 1.0688310273591342e-8,
          5.3264509831530596e-8, 3.142657152701095e-8, 1.403919028467466e-8,
          0.036670780896199455, 0.06103588681258062, 8.016530923475137e-9,
          8.463197939501447e-9, 0.49353807292516894, 4.108809303828205e-9,
          7.564021128858248e-9, 7.920935556677459e-9, 1.981074290701079e-8,
          2.478297427657827e-8, 4.913423015144346e-8, 2.0111898684763165e-8,
          1.0424638218871013e-8, 0.19771544705201047]
    riskt = 0.028298069311286384
    rett = 0.0005082343814003915
    @test isapprox(w1.weights, wt0, rtol = 5.0e-5)
    @test isapprox(r1, riskt0, rtol = 5.0e-7)
    @test isapprox(ret1, rett0, rtol = 5.0e-6)
    @test isapprox(w1.weights, wt, rtol = 5.0e-6)
    @test isapprox(r1, riskt, rtol = 5.0e-7)
    @test isapprox(ret1, rett, rtol = 5.0e-7)

    rm = [[RVaR2(), RVaR2()]]
    w2 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r2 = calc_risk(portfolio; type = :Trad2, rm = rm[1][1])
    ret2 = dot(portfolio.mu, w2.weights)
    wt = [1.4097272321966204e-9, 0.2110508645527885, 1.5432657325410157e-9,
          8.224555108428983e-9, 8.188544359356975e-9, 2.0467382718505696e-9,
          0.03667013219084804, 0.06104862385038757, 1.1545205788850546e-9,
          1.2014975170275297e-9, 0.49353309590382577, 5.516848747794179e-10,
          1.0882481401019833e-9, 1.1919443513103505e-9, 2.9945904499140644e-9,
          4.9758627720463646e-9, 8.859577318771023e-9, 6.6972920206520355e-9,
          1.4672079720070243e-9, 0.1976972319068934]
    riskt = 0.028298068766299395
    rett = 0.0005082326485814695
    @test isapprox(w2.weights, wt0, rtol = 5.0e-5)
    @test isapprox(r2, riskt0, rtol = 5.0e-8)
    @test isapprox(ret2, rett0, rtol = 5.0e-6)
    @test isapprox(w2.weights, wt, rtol = 5.0e-6)
    @test isapprox(r2, riskt, rtol = 5.0e-8)
    @test isapprox(ret2, rett, rtol = 5.0e-6)

    obj = SR(; rf = rf)
    wt0 = [9.496500669050249e-9, 2.64615310020192e-8, 7.273118042494954e-9,
           1.4049587952157727e-8, 0.5059944415194525, 2.377003832919441e-9,
           0.17234053237874894, 1.8314836691951746e-8, 1.2375544635066102e-8,
           4.317304792347554e-8, 1.9197414728022034e-6, 2.401462046149522e-9,
           1.6115997522673463e-9, 9.360121102571334e-9, 2.354326688306667e-9,
           0.1824768715252159, 0.1391859057572847, 2.2814940892439545e-8,
           1.5125718216815985e-7, 5.757021876600399e-9]
    riskt0 = 0.04189063415633535
    rett0 = 0.0015775582433052353

    rm = RVaR2(; settings = RiskMeasureSettings(; scale = 2.0))
    w3 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r3 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret3 = dot(portfolio.mu, w3.weights)
    wt = [3.6964402195787056e-9, 1.0392450005092712e-8, 2.784310764430045e-9,
          5.445375391935735e-9, 0.5059942976555293, 9.241124324143136e-10,
          0.17233974981261022, 7.2220562389865515e-9, 4.808425821851268e-9,
          1.8097317315860523e-8, 7.660889317153167e-7, 9.325396069739327e-10,
          6.252333216504217e-10, 3.6899451990998283e-9, 9.22741438260991e-10,
          0.1824769815798792, 0.13918807405363534, 8.997042436389308e-9,
          6.00576896628795e-8, 2.213734428185603e-9]
    riskt = 0.041890646919977646
    rett = 0.0015775584188519136
    @test isapprox(w3.weights, wt0, rtol = 5.0e-6)
    @test isapprox(r3, riskt0, rtol = 5.0e-7)
    @test isapprox(ret3, rett0, rtol = 5.0e-7)
    @test isapprox(w3.weights, wt, rtol = 1.0e-7)
    @test isapprox(r3, riskt, rtol = 5.0e-8)
    @test isapprox(ret3, rett)

    rm = [[RVaR2(), RVaR2()]]
    w4 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r4 = calc_risk(portfolio; type = :Trad2, rm = rm[1][1])
    ret4 = dot(portfolio.mu, w4.weights)
    wt = [4.01862578529583e-9, 1.135001153879239e-8, 3.02615601845224e-9,
          5.928312901255709e-9, 0.5059937381144556, 1.0038057612200312e-9,
          0.17234033480262054, 7.901337339049742e-9, 5.218986111329988e-9,
          6.09101653261886e-8, 1.666755623388464e-6, 1.0117836491356133e-9,
          6.78668148934042e-10, 4.0325203692118824e-9, 1.0053482463189763e-9,
          0.18247685711761946, 0.1391872175678591, 9.840172994354298e-9,
          6.731122999590008e-8, 2.4046977629845584e-9]
    riskt = 0.041890627807408765
    rett = 0.0015775577512983736
    @test isapprox(w4.weights, wt0, rtol = 5.0e-6)
    @test isapprox(r4, riskt0, rtol = 5.0e-7)
    @test isapprox(ret4, rett0, rtol = 5.0e-7)
    @test isapprox(w4.weights, wt, rtol = 5.0e-8)
    @test isapprox(r4, riskt, rtol = 5.0e-7)
    @test isapprox(ret4, rett)

    # Risk upper bound
    obj = MaxRet()
    rm = RVaR2(; settings = RiskMeasureSettings(; scale = 2.0))
    rm.settings.ub = r1
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r1 ||
          abs(calc_risk(portfolio; type = :Trad2, rm = rm) - r1) < 2e-6

    rm = [[RVaR2(), RVaR2()]]
    rm[1][1].settings.ub = r2
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm[1][1]) <= r2 ||
          abs(calc_risk(portfolio; type = :Trad2, rm = rm[1][1]) - r2) < 2e-6

    obj = SR(; rf = rf)
    rm = RVaR2(; settings = RiskMeasureSettings(; scale = 1.0))
    rm.settings.ub = r1 * 1.000001
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r1 ||
          abs(calc_risk(portfolio; type = :Trad2, rm = rm) - r1) < 1e-8

    rm = [[RVaR2(), RVaR2()]]
    rm[1][1].settings.ub = r2
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm[1][1]) <= r2 ||
          abs(calc_risk(portfolio; type = :Trad2, rm = rm[1][1]) - r2) < 1e-7

    # Ret lower bound
    obj = MinRisk()
    rm = RVaR2(; settings = RiskMeasureSettings(; scale = 2.0))
    rm.settings.ub = Inf
    portfolio.mu_l = ret1
    w5 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w5.weights) >= ret1

    rm = [[RVaR2(), RVaR2()]]
    portfolio.mu_l = ret2
    w6 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w6.weights) >= ret2

    obj = SR(; rf = rf)
    rm = RVaR2(; settings = RiskMeasureSettings(; scale = 2.0))
    portfolio.mu_l = ret1
    w7 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w7.weights) >= ret1

    rm = [[RVaR2(), RVaR2()]]
    portfolio.mu_l = ret2
    w8 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w8.weights) >= ret2

    obj = MinRisk()
    wt0 = [5.102262457628692e-9, 0.21104494400160803, 5.341766086210288e-9,
           2.5458382238901392e-8, 1.697696472902229e-8, 7.287515616039478e-9,
           0.03667031714382797, 0.061041476346139684, 4.093926758298615e-9,
           4.303160140655642e-9, 0.49353591974074, 2.1672264824822902e-9,
           3.926886474939328e-9, 4.083625597792755e-9, 1.043237724356759e-8,
           1.1621198331723714e-8, 2.5232405645111758e-8, 1.1835541180026409e-8,
           4.9679012600678e-9, 0.19770719993654404]
    riskt0 = 0.028298069755304314
    rett0 = 0.000508233446626652
    wt1 = [2.4600397213078317e-8, 0.09562005510508859, 2.2484210868221866e-8,
           0.027854316374187026, 0.005002688335990861, 2.071409133341056e-8,
           0.003718233146430421, 0.12627281447779795, 1.6884777126154073e-8,
           2.0178587919597117e-8, 0.3766707075734549, 8.63395850422015e-9,
           6.7424532903562576e-9, 5.9543651135995095e-8, 2.0829009309447037e-8,
           0.02859254921089233, 0.0782650660292682, 0.09185260641270176,
           2.482440325337885e-8, 0.16615073789864804]
    riskt1 = 0.007924109456497849
    rett1 = 0.00050823276045652

    rm = [[RVaR2(), RVaR2(; alpha = 0.75)]]
    w9 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r9 = calc_risk(portfolio; type = :Trad2, rm = rm[1][1])
    ret9 = dot(portfolio.mu, w9.weights)
    wt = [1.2119402634609286e-9, 0.18881779134736748, 1.2072579301196785e-9,
          0.012052906035833965, 3.6663470936394974e-9, 2.3037857899137735e-9,
          0.029076569761242545, 0.09158832917729257, 1.0065101194158649e-9,
          1.1549345611832656e-9, 0.46705257183229987, 5.936779422053456e-10,
          1.0273863326472856e-9, 1.1104755188389804e-9, 1.949288729732805e-9,
          1.2452633157976303e-8, 0.02506812878017088, 2.625635742204549e-9,
          1.3273133081705558e-9, 0.18634367142860617]
    riskt = 0.0283718773971774
    rett = 0.0005082326592020647
    @test !isapprox(w9.weights, wt0)
    @test !isapprox(r9, riskt0)
    @test !isapprox(ret9, rett0)
    @test !isapprox(w9.weights, wt1)
    @test !isapprox(r9, riskt1)
    @test !isapprox(ret9, rett1)
    @test isapprox(w9.weights, wt, rtol = 5.0e-5)
    @test isapprox(r9, riskt, rtol = 5.0e-8)
    @test isapprox(ret9, rett, rtol = 5.0e-6)
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
