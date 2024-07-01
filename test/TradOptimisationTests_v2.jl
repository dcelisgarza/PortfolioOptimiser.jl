using CSV, TimeSeries, DataFrames, StatsBase, Statistics, LinearAlgebra, Test, Clarabel,
      PortfolioOptimiser

prices = TimeArray(CSV.File("./assets/stock_prices.csv"); timestamp = :date)
rf = 1.0329^(1 / 252) - 1
l = 2.0

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
    calc_risk(portfolio; type = :Trad2, rm = rm)
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
    wt = wt = [0.01406192655834688, 0.04237501824390749, 0.016863390017404657,
               0.0020208011519371604, 0.01768375954179011, 0.054224042598668906,
               3.860281087148557e-10, 0.15821648846151862, 3.724862305158064e-10,
               8.917492418807677e-10, 0.23689726979743447, 9.589227849839197e-11,
               8.366729059719944e-11, 0.12783207480092484, 0.0003509794124345412,
               0.0009122918961555292, 0.04394937135411301, 0.18272429219207284,
               4.845293759724182e-10, 0.10188829165893845]
    @test isapprox(w2.weights, wt)

    w3 = optimise2!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    wt = wt = [0.01406182186298033, 0.042375224999262856, 0.016863463035875152,
               0.002020826242971577, 0.017683796862612216, 0.05422412374300622,
               6.732690796579932e-10, 0.15821648252680418, 6.380663170694303e-10,
               1.5182889035485862e-9, 0.23689723775690535, 1.7588752969970232e-10,
               1.5574094499941487e-10, 0.12783199407018078, 0.0003510013637440437,
               0.0009122576900588412, 0.04394953137073167, 0.18272434607988453,
               8.246849358651642e-10, 0.10188788840904445]
    @test isapprox(w3.weights, wt)
end
#=
################
port = Portfolio(; prices = prices,
                 solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                  :params => Dict("verbose" => false))))
asset_statistics!(port)

r = :MAD
opt = OptimiseOpt(; rf = rf, l = l, class = :Classic, type = :Trad, rm = r, obj = :Min_Risk,
                  kelly = :None)
opt.obj = :Min_Risk
opt.kelly = :Exact
@time _w = optimise!(port, opt)
println("wt = $(_w.weights)")
println("riskt = $(calc_risk(port; type = :Trad, rm = r, rf = rf))")
println("rett = $(dot(port.mu, _w.weights))")
################

obj = Util(; l = l)
w4 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
r2 = calc_risk(portfolio; type = :Trad2, rm = rm)
ret2 = dot(portfolio.mu, w4.weights)
wt = [1.065553054256496e-9, 1.906979877393637e-9, 2.1679869440360567e-9,
      1.70123972526289e-9, 0.7741855142171694, 3.9721744242294547e-10, 0.10998135534654405,
      1.3730517031876334e-9, 1.5832262577152926e-9, 1.0504881447825781e-9,
      1.2669287896045939e-9, 4.038975120701348e-10, 6.074001448526581e-10,
      2.654358762537183e-10, 6.574536682273354e-10, 0.1158331072870088,
      3.0452991740231055e-9, 1.3663094482455795e-9, 2.4334674474942e-9,
      1.8573424305703526e-9]
riskt = 0.01609460480445889
rett = 0.0017268228943243054
@test isapprox(w4.weights, wt)
@test isapprox(r2, riskt)
@test isapprox(ret2, rett)

w5 = optimise2!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
wt = [8.870505348946403e-9, 2.0785636066552727e-8, 2.798137197308657e-8,
      1.971091339190289e-8, 0.7185881405281792, 4.806880649144299e-10, 0.09860828964210354,
      1.1235098321720224e-8, 2.977172777854582e-8, 8.912749778026878e-9,
      9.63062128166912e-9, 1.0360544993920464e-9, 2.180352541614548e-9,
      2.689800139816139e-9, 2.3063944199708073e-9, 0.15518499560246005,
      0.027618271886178034, 1.246121371211767e-8, 1.2842725621709964e-7,
      1.586069567397408e-8]
@test isapprox(w5.weights, wt)

w6 = optimise2!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
wt = [4.176019357134568e-9, 8.185339020442324e-9, 1.0737258331407902e-8,
      7.901762479846926e-9, 0.7207182132289714, 1.1540211071681835e-9, 0.09835523884681871,
      4.849354486370478e-9, 1.1755943684787842e-8, 4.185213130955141e-9,
      4.314456480234504e-9, 9.722540074689786e-10, 5.895848876745837e-10,
      2.1334187374036406e-9, 5.661079854916932e-10, 0.15505269633577082,
      0.025873730012408114, 5.286907987963515e-9, 4.83569668574532e-8, 6.41142230104617e-9]
@test isapprox(w6.weights, wt, rtol = 1.0e-6)

obj = SR(; rf = rf)
w7 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
r3 = calc_risk(portfolio; type = :Trad2, rm = rm)
ret3 = dot(portfolio.mu, w7.weights)
wt = [3.435681413150756e-10, 9.269743242817558e-10, 1.174114891347931e-9,
      7.447407606568295e-10, 0.5180552669298059, 1.08151797431462e-10, 0.0636508036260703,
      7.872264113421062e-10, 7.841830201959634e-10, 3.9005509625957585e-10,
      6.479557895235057e-10, 8.472023236127232e-11, 5.766670106753152e-11,
      1.988136246095318e-10, 5.935811276550078e-11, 0.14326634942881586, 0.1964867973307653,
      7.554937254824565e-10, 0.0785407748474901, 7.740298948228655e-10]
riskt = 0.013160876658207102
rett = 0.0014788430765515807
@test isapprox(w7.weights, wt)
@test isapprox(r3, riskt)
@test isapprox(ret3, rett)

w8 = optimise2!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
wt = [1.0672947772539922e-8, 3.6603580846259566e-8, 4.190924498057212e-8,
      2.579795624783031e-8, 0.45247454726503317, 3.139203265461306e-9, 0.05198581042386962,
      1.1201379704294516e-7, 1.799097939748088e-8, 1.2844577033392204e-8,
      5.1484053193477936e-8, 2.3241091705338425e-9, 1.699312214555245e-9,
      6.26319015273334e-9, 1.6636900367102399e-9, 0.13648649205020114, 0.2350741185365231,
      4.844604537439258e-8, 0.12397862173181687, 3.713986941437853e-8]
@test isapprox(w8.weights, wt, rtol = 1.0e-7)

w9 = optimise2!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
wt = [1.6553521389413233e-7, 5.275915958538328e-7, 6.846007363199405e-7,
      3.914933469758698e-7, 0.48926168709100504, 4.9332881037102406e-8, 0.0583064644410985,
      5.594366962947531e-7, 3.1357711474708337e-7, 1.895896838004368e-7,
      4.1299427275337544e-7, 3.811276445091462e-8, 2.7731552876975723e-8,
      9.393138539482288e-8, 2.6831018704067043e-8, 0.1402408063077745, 0.2134138585246757,
      4.713662104500069e-7, 0.09877278790316771, 4.4360780483006885e-7]
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
      8.390684135793784e-10, 0.8503431998881756, 5.837264433583887e-10, 0.14965678808511193,
      8.520932897976487e-13, 7.66160958682953e-10, 1.0247860261071675e-10,
      5.1627700971086255e-11, 5.483183958203547e-10, 7.565204185674542e-10,
      3.16106264753721e-10, 7.638502459889708e-10, 2.447496129413098e-9,
      1.372927322256315e-9, 6.541563185875491e-11, 9.248420166125226e-10,
      3.9509971643490626e-10]
@test isapprox(w11.weights, wt)

w12 = optimise2!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
wt = [1.2461220626745066e-9, 1.3954805691188813e-9, 1.7339340756162538e-9,
      1.5635882299558742e-9, 0.853395149768853, 5.48096579085184e-10, 0.14660482860584584,
      9.501446622854747e-10, 1.5113651288469943e-9, 1.027931406345638e-9,
      9.130613494698656e-10, 5.686010690200261e-10, 4.0494468011345616e-10,
      7.290999439594515e-10, 4.1154424470964885e-10, 2.82566220199723e-9,
      1.9419703441146337e-9, 1.0003454025331967e-9, 1.6178718912419106e-9,
      1.2355373241783204e-9]
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
calc_risk(portfolio; type = :Trad2, rm = rm)
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
# end
=#
