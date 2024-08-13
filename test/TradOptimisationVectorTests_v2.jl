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
