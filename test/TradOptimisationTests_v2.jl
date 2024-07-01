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
    wt = [0.01406192655834688, 0.04237501824390749, 0.016863390017404657,
          0.0020208011519371604, 0.01768375954179011, 0.054224042598668906,
          3.860281087148557e-10, 0.15821648846151862, 3.724862305158064e-10,
          8.917492418807677e-10, 0.23689726979743447, 9.589227849839197e-11,
          8.366729059719944e-11, 0.12783207480092484, 0.0003509794124345412,
          0.0009122918961555292, 0.04394937135411301, 0.18272429219207284,
          4.845293759724182e-10, 0.10188829165893845]
    @test isapprox(w2.weights, wt)

    w3 = optimise2!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    wt = [0.01406182186298033, 0.042375224999262856, 0.016863463035875152,
          0.002020826242971577, 0.017683796862612216, 0.05422412374300622,
          6.732690796579932e-10, 0.15821648252680418, 6.380663170694303e-10,
          1.5182889035485862e-9, 0.23689723775690535, 1.7588752969970232e-10,
          1.5574094499941487e-10, 0.12783199407018078, 0.0003510013637440437,
          0.0009122576900588412, 0.04394953137073167, 0.18272434607988453,
          8.246849358651642e-10, 0.10188788840904445]
    @test isapprox(w3.weights, wt)

    obj = Util(; l = l)
    w4 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r2 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret2 = dot(portfolio.mu, w4.weights)
    wt = [0.009637742224105319, 0.05653095216430771, 0.008693798837803784,
          0.010136614699778262, 0.07101740258202667, 1.626746387899197e-10,
          0.00018717126494052322, 0.14166839870576114, 2.0925633115430728e-10,
          0.011541334039355096, 0.2033965474136089, 1.5032151500330365e-11,
          1.7660928669155618e-11, 0.0776986622152643, 5.6076568461019626e-11,
          0.018226673642440187, 0.08534261194338741, 0.17173762426959616,
          0.03444968686639962, 0.09973477867052433]
    riskt = 0.005726370460509949
    rett = 0.0005627824531830065
    @test isapprox(w4.weights, wt)
    @test isapprox(r2, riskt)
    @test isapprox(ret2, rett)

    w5 = optimise2!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [0.009549562791630994, 0.056131884099740126, 0.009058885157954702,
          0.010221145667272422, 0.07101604294130621, 6.158699951966029e-10,
          0.00017307955558545985, 0.1418306549018017, 1.0502171408125738e-9,
          0.01160511059804641, 0.20331669980076691, 4.561412945913223e-10,
          4.377177861936833e-10, 0.0778263850914995, 1.3357521459019656e-10,
          0.01821185094370143, 0.08514627046381432, 0.17137747164304867,
          0.034630959754866464, 0.09990399389544322]
    @test isapprox(w5.weights, wt)

    w6 = optimise2!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    wt = [0.009654581165225669, 0.05668321904461835, 0.008731332496008505,
          0.010033306687767333, 0.07088447280879365, 2.4996977644532297e-9,
          0.00027321456023309383, 0.14173741676314647, 3.441171402254621e-9,
          0.01168506376484469, 0.20349044392989254, 2.9458239145194473e-10,
          3.320396727480843e-10, 0.0779288019164822, 9.597391595541974e-10,
          0.018198160717452304, 0.08532649443354709, 0.17148712466334665,
          0.03410246739707899, 0.09978389212433202]
    @test isapprox(w6.weights, wt)

    obj = SR(; rf = rf)
    w7 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r3 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret3 = dot(portfolio.mu, w7.weights)
    wt = [3.372797021213529e-8, 8.679493419672167e-8, 8.730378695790742e-8,
          4.654016314992972e-8, 0.6621196971211604, 1.000789705313297e-8,
          0.04256386189823906, 5.0027909676906887e-8, 9.072276529043811e-8,
          4.296795445352721e-8, 7.991647846958404e-8, 7.108969143618601e-9,
          5.039720687490243e-9, 1.839999189017112e-8, 5.602046740184832e-9,
          0.1343671243475813, 0.08752271182145684, 7.258944630234996e-8,
          0.07342589536656563, 7.269496276800682e-8]
    riskt = 0.009898352231115614
    rett = 0.0015741047141763708
    @test isapprox(w7.weights, wt)
    @test isapprox(r3, riskt)
    @test isapprox(ret3, rett)

    w8 = optimise2!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [5.795290443004618e-10, 1.7735397255154201e-9, 2.4415842553008867e-9,
          7.40231745041612e-10, 0.5088109495587692, 1.6329984594581992e-10,
          0.03160667052061857, 3.0214145196781185e-9, 1.040688531056621e-9,
          6.436583441219632e-10, 0.0028683462933828748, 1.0487960563547365e-10,
          7.78113271299916e-11, 3.1960362958775614e-10, 8.512258385515617e-11,
          0.13826727169736175, 0.18565374607217008, 4.6031374812460635e-9,
          0.1327929979115769, 2.3516201199681936e-9]
    @test isapprox(w8.weights, wt, rtol = 1e-7)

    w9 = optimise2!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    wt = [1.1698348692452301e-8, 3.055254703791635e-8, 3.7017682303678675e-8,
          1.688244767018444e-8, 0.5791986473166304, 3.6199007069253584e-9,
          0.03882577851852925, 2.4972285601281053e-8, 2.70885146227947e-8,
          1.3757807183441849e-8, 4.270065859227207e-8, 2.513254702967732e-9,
          1.8448570882468499e-9, 6.737568054533749e-9, 1.9980748786998393e-9,
          0.13597861132741978, 0.13966589702878732, 3.314399410321976e-8,
          0.10633078216653338, 2.9114158748935705e-8]
    @test isapprox(w9.weights, wt)

    obj = MaxRet()
    w10 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r4 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret4 = dot(portfolio.mu, w10.weights)
    wt = [1.2971056777827964e-8, 1.3930988564983735e-8, 1.66544480132768e-8,
          1.6052005598315065e-8, 2.5272650716621174e-7, 1.010529781274771e-8,
          0.9999995082509356, 1.1380213103425267e-8, 1.4878768493855394e-8,
          1.2066244486141283e-8, 1.135216260978634e-8, 1.00771678859298e-8,
          9.057732065596785e-9, 1.08055110966454e-8, 9.429257146092314e-9,
          2.331510717112652e-8, 1.7720775475614838e-8, 1.1669132650930886e-8,
          1.5021261060062696e-8, 1.25354272189245e-8]
    riskt = 0.02624326616973302
    rett = 0.0018453753062770842
    @test isapprox(w10.weights, wt)
    @test isapprox(r4, riskt)
    @test isapprox(ret4, rett)

    w11 = optimise2!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [6.24782988116139e-10, 9.366363865924148e-10, 1.6428209739519796e-9,
          1.2834078728926378e-9, 0.8503450577361338, 8.327210654169724e-10,
          0.14965492407526051, 1.8834252112272695e-11, 1.1714888587578576e-9,
          1.7599099927857186e-10, 5.734283990075868e-11, 7.855145138821309e-10,
          1.109109882692946e-9, 4.456991350071304e-10, 1.1055539398977906e-9,
          3.790544357778616e-9, 2.0750030440064227e-9, 1.2070096217342874e-10,
          1.4018002145268373e-9, 6.106531961743963e-10]
    @test isapprox(w11.weights, wt)

    w12 = optimise2!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    wt = [6.757093243027148e-9, 7.664031309161123e-9, 9.442178874312985e-9,
          8.584900291197654e-9, 0.8533948964966961, 3.0825110840326666e-9,
          0.1466049837308141, 5.287317105823606e-9, 8.388979848897765e-9,
          5.749959272436911e-9, 5.103146714031344e-9, 3.205393470435072e-9,
          2.2935857006942965e-9, 4.081417604988999e-9, 2.3373314078613168e-9,
          1.5619143210865274e-8, 1.0720641149855979e-8, 5.557609448709485e-9,
          9.024863752987753e-9, 6.872386298380374e-9]
    @test isapprox(w12.weights, wt)

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
          abs(dot(portfolio.mu, w16.weights) - ret4) < 1e-10
end
