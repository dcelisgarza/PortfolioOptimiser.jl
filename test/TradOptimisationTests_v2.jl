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
    @test isapprox(w12.weights, wt, rtol = 1.0e-5)

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
    @test isapprox(w6.weights, wt, rtol = 0.0001)

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
    @test isapprox(w7.weights, wt, rtol = 1.0e-7)
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
    @test isapprox(w9.weights, wt, rtol = 1.0e-6)

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

@testset "SSD" begin
    portfolio = Portfolio2(; prices = prices,
                           solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                            :params => Dict("verbose" => false))))
    asset_statistics2!(portfolio)
    rm = SSD2()

    obj = MinRisk()
    w1 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r1 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret1 = dot(portfolio.mu, w1.weights)
    wt = [2.954283441277634e-8, 0.04973740293196223, 1.5004834875433086e-8,
          0.002978185203626395, 0.00255077171396876, 0.02013428421720317,
          8.938505323199939e-10, 0.12809490679767346, 2.5514571986823903e-9,
          3.4660313800221236e-9, 0.29957738105080456, 3.6587132183584753e-9,
          1.61047759821642e-9, 0.1206961339634279, 0.012266097184153368,
          0.009663325635394784, 1.859820936315932e-8, 0.22927479857319558,
          3.22169253589993e-9, 0.12502663418048846]
    riskt = 0.005538773213915548
    rett = 0.00031286022410236273
    @test isapprox(w1.weights, wt)
    @test isapprox(r1, riskt)
    @test isapprox(ret1, rett)

    w2 = optimise2!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [1.328560973467273e-8, 0.049738240342198585, 6.729638177718146e-9,
          0.0029785880061378384, 0.002551194236638699, 0.02013119386894698,
          3.7241820112123204e-10, 0.1280950127330832, 1.1193171935929993e-9,
          1.5313119883776913e-9, 0.2995770952417976, 1.6188140270762791e-9,
          6.955937873270665e-10, 0.12069621604817828, 0.012266360319875118,
          0.009662882398733882, 8.34506433718684e-9, 0.22927554413792217,
          1.4212376700688693e-9, 0.12502763754748242]
    @test isapprox(w2.weights, wt)

    w3 = optimise2!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    wt = [2.5190879083102043e-8, 0.049737774196137965, 1.2868124743982513e-8,
          0.0029783638332052057, 0.0025509173649045694, 0.020133768670666973,
          9.047615243063875e-10, 0.1280950304816767, 2.3106803351985052e-9,
          3.086425946402379e-9, 0.2995771396012498, 3.251604912185191e-9,
          1.5132089628845487e-9, 0.12069584048889491, 0.012266113001483253,
          0.009662762604571083, 1.5907937864516914e-8, 0.2292749772563436,
          2.8790402254376556e-9, 0.12502724458820227]
    @test isapprox(w3.weights, wt)

    obj = Util(; l = l)
    w4 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r2 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret2 = dot(portfolio.mu, w4.weights)
    wt = [2.541867415091494e-9, 0.05530813817602699, 2.41244845892351e-9,
          0.004275662662643133, 0.03390817389617674, 2.317666518349058e-9,
          2.2319148968228334e-10, 0.1252175035167737, 7.110662324391565e-10,
          8.147053974005637e-10, 0.29837139405224006, 3.8477613032148593e-10,
          1.94163891883498e-10, 0.10724362095308528, 0.0027603552142120838,
          0.02112020545265115, 0.005356541418379395, 0.2289888503445687,
          1.076090818037532e-9, 0.11744954363726649]
    riskt = 0.005561860483104124
    rett = 0.00041086155751295247
    @test isapprox(w4.weights, wt)
    @test isapprox(r2, riskt)
    @test isapprox(ret2, rett)

    w5 = optimise2!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [5.310366383671854e-10, 0.055185496756761133, 4.958251729556289e-10,
          0.004440395111615488, 0.033778656762619236, 4.711290484606268e-10,
          1.4995099027754112e-10, 0.12528998427061558, 8.369500406278012e-12,
          2.3638500512437794e-11, 0.2983971162306186, 1.031575927125493e-10,
          1.5846481226439663e-10, 0.10741032813109379, 0.002725697131388786,
          0.021069728462151726, 0.005363035530923987, 0.2288344439497473,
          9.738524268054006e-11, 0.11750511562350689]
    @test isapprox(w5.weights, wt)

    w6 = optimise2!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    wt = [2.2895674946199656e-9, 0.05518903108297828, 2.1836060733181756e-9,
          0.0044374629070126565, 0.033766396900666275, 2.110869140807151e-9,
          2.544160805002686e-10, 0.12529117631048525, 6.773571588439321e-10,
          7.730249106515778e-10, 0.2984002871957823, 3.943554094652132e-10,
          2.2905478935733912e-10, 0.10741594289180213, 0.0027306359963794984,
          0.02106510870973968, 0.005352552882926046, 0.2288401489338605,
          9.93196861710178e-10, 0.11751124628291945]
    @test isapprox(w6.weights, wt, rtol = 1.0e-6)

    obj = SR(; rf = rf)
    w7 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r3 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret3 = dot(portfolio.mu, w7.weights)
    wt = [1.8955665771231532e-8, 4.4148508725600023e-8, 3.537890454211662e-8,
          2.1966271358039556e-8, 0.6666203563586275, 6.130148331498872e-9,
          0.03792018451465443, 3.563315827678111e-8, 4.349162854829938e-8,
          1.8479882644634467e-8, 4.552310886494339e-8, 4.8863225987358126e-9,
          3.315774614641478e-9, 1.2573247089938602e-8, 3.5165001620600556e-9,
          0.1718521246394113, 0.10257058901854942, 4.7654011023485184e-8,
          0.021036366772688796, 3.7042935949165386e-8]
    riskt = 0.00981126385893784
    rett = 0.0015868900032431047
    @test isapprox(w7.weights, wt)
    @test isapprox(r3, riskt)
    @test isapprox(ret3, rett)

    w8 = optimise2!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [1.1686179891013613e-8, 3.7143021536201827e-8, 2.301711086719213e-8,
          1.389322574147696e-8, 0.566054323906044, 3.798547414540005e-9,
          0.02904414098555804, 5.047834297132999e-8, 1.9806257217757055e-8,
          1.1826573374560461e-8, 1.380045456737115e-7, 2.885310238914747e-9,
          2.101052458639916e-9, 8.171076474830977e-9, 2.189455584169703e-9,
          0.164249355676866, 0.1629106816795142, 7.659812295832371e-8, 0.07774106421141706,
          3.1941778285711126e-8]
    @test isapprox(w8.weights, wt, rtol = 5e-7)

    w9 = optimise2!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    wt = [2.3079619465821097e-7, 5.87719458710357e-7, 4.4740668517610773e-7,
          2.8029700495576643e-7, 0.6080716728391783, 7.731151243684516e-8,
          0.03582104571289915, 5.413757919342572e-7, 4.787969851327624e-7,
          2.2930662515647874e-7, 7.435354777738109e-7, 6.176625839188381e-8,
          4.43431803073848e-8, 1.5413323851090924e-7, 4.610353331379092e-8,
          0.16672108503331592, 0.13724564769278624, 7.191502039920105e-7,
          0.0521354129969113, 4.936827587037762e-7]
    @test isapprox(w9.weights, wt)

    obj = MaxRet()
    w10 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r4 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret4 = dot(portfolio.mu, w10.weights)
    wt = [1.8748605656928656e-8, 1.8320719487450265e-8, 1.538653450499897e-8,
          1.6989819644791176e-8, 3.3008818023484547e-7, 1.4569564034395335e-8,
          0.9999993779344156, 1.8172395224611005e-8, 1.8057718124898897e-8,
          1.882604735063403e-8, 1.7974701566096796e-8, 1.5301460205927646e-8,
          1.2337854122082459e-8, 1.675271979100736e-8, 1.2582988670103514e-8,
          8.789449903631792e-9, 1.4346521744321203e-8, 1.8465753178372508e-8,
          1.7524551334228423e-8, 1.882999955421456e-8]
    riskt = 0.025704341997146034
    rett = 0.0018453751965893277
    @test isapprox(w10.weights, wt)
    @test isapprox(r4, riskt)
    @test isapprox(ret4, rett)

    w11 = optimise2!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [1.530417939183873e-10, 2.8612308735562203e-10, 5.505656045721553e-10,
          4.5496556945069153e-10, 0.8503475681489141, 3.8305099849033153e-10,
          0.1496524260843989, 7.250692125835878e-11, 3.862370077496164e-10,
          1.7288373304633485e-11, 9.181697560252934e-11, 3.404667655856662e-10,
          3.876849785161851e-10, 2.3474971482803645e-10, 4.1508395195208027e-10,
          6.184903168160772e-10, 7.002854291572307e-10, 3.1063304351626673e-11,
          4.812277872084391e-10, 1.6203868518158558e-10]
    @test isapprox(w11.weights, wt)

    w12 = optimise2!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    wt = [2.4198206282739137e-9, 2.7523773137294424e-9, 3.4600331962305982e-9,
          3.1089170165506684e-9, 0.8533948120145702, 9.622378649372175e-10,
          0.14660514506394484, 1.8389581632770744e-9, 2.9995402556500355e-9,
          2.0242460469329184e-9, 1.7817450751121457e-9, 1.0985267193180098e-9,
          7.712326645258262e-10, 1.3809357105377707e-9, 8.057118379005105e-10,
          6.0398082300262525e-9, 3.896463510594108e-9, 1.9253519524735756e-9,
          3.2097224949880373e-9, 2.445856342069014e-9]
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
          abs(calc_risk(portfolio; type = :Trad2, rm = rm) - r1) < 1e-8

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

@testset "FLPM" begin
    portfolio = Portfolio2(; prices = prices,
                           solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                            :params => Dict("verbose" => false))))
    asset_statistics2!(portfolio)
    rm = FLPM2(; target = rf)

    obj = MinRisk()
    w1 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r1 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret1 = dot(portfolio.mu, w1.weights)
    wt = [0.004266070614517317, 0.04362165239521167, 0.01996043023729806,
          0.007822722623891595, 0.060525786877357816, 2.187204740032422e-8,
          0.00039587162942815576, 0.13089236100375287, 7.734531969787049e-9,
          0.0118785975269765, 0.2066094523343813, 6.469640939191796e-10,
          6.246750358607508e-10, 0.08329494463798208, 1.6616489736084757e-9,
          0.013888127426323596, 0.0873465246195096, 0.19210093372199202,
          0.03721303157281544, 0.10018346023869455]
    riskt = 0.00265115220934628
    rett = 0.0005443423420749122
    @test isapprox(w1.weights, wt)
    @test isapprox(r1, riskt)
    @test isapprox(ret1, rett)

    w2 = optimise2!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [0.00426610239591103, 0.04362178853662362, 0.019960457154756087,
          0.007822768508828756, 0.06052527699807257, 4.506186395476567e-10,
          0.0003959578648306813, 0.1308926705227917, 1.8600603149618773e-10,
          0.011878924975411324, 0.2066096337189848, 1.5259036251172595e-11,
          1.466852985609556e-11, 0.08329480736550154, 4.0265597843607914e-11,
          0.013888097943043014, 0.08734723725306878, 0.19210036041488635,
          0.037213199462145435, 0.10018271617832634]
    @test isapprox(w2.weights, wt)

    w3 = optimise2!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    wt = [0.00426608716920985, 0.04362173641258839, 0.019960444305697746,
          0.007822767144185885, 0.06052541421131514, 3.4024764351128286e-9,
          0.00039592869717005593, 0.13089257106535185, 1.263061291396848e-9,
          0.0118788599251899, 0.20660956985823367, 1.2640723567497645e-10,
          1.224400506824836e-10, 0.08329482612251998, 2.9820592088268585e-10,
          0.013888098156328821, 0.08734710673926958, 0.19210049684132757,
          0.037213161399705486, 0.1001829267393152]
    @test isapprox(w3.weights, wt)

    obj = Util(; l = l)
    w4 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r2 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret2 = dot(portfolio.mu, w4.weights)
    wt = [5.651779189600572e-9, 0.042138947590740154, 0.012527625908380429,
          0.00720584542391769, 0.09868086575102923, 4.666011527039072e-10,
          0.002772974287825817, 0.10921142006880304, 1.453665644745307e-9,
          4.587434723327683e-9, 0.20154713458183793, 1.034055452971105e-10,
          1.1243437198410302e-10, 0.026542856504844895, 2.1611055686908496e-10,
          0.041731161486272504, 0.11080106522064202, 0.17232533003113762,
          0.06800096884126186, 0.10651379171187556]
    riskt = 0.0026842735895541213
    rett = 0.0006732529128667895
    @test isapprox(w4.weights, wt)
    @test isapprox(r2, riskt)
    @test isapprox(ret2, rett)

    w5 = optimise2!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [7.025580903339914e-9, 0.03801909891801268, 0.015508772801692194,
          0.007920175117119383, 0.09985335454405923, 2.5401009772523425e-10,
          0.0024873633006595353, 0.10503236887549107, 8.540598148869465e-10,
          4.653369605256221e-9, 0.203007892809087, 7.000508823160763e-10,
          6.910321683931141e-10, 0.035031103036583196, 5.676257759160685e-10,
          0.03798687241385029, 0.10539179463937084, 0.1777681655105699, 0.06295784178579301,
          0.10903518150198238]
    @test isapprox(w5.weights, wt)

    w6 = optimise2!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    wt = [1.2872334765954068e-8, 0.03767953016874954, 0.015470385826160825,
          0.0081170648205414, 0.09892766032152121, 1.0417936519167395e-9,
          0.0025348401228208013, 0.10487955021910977, 2.8917086673976536e-9,
          8.851185748932032e-9, 0.20310090288185126, 3.0551749498443944e-10,
          3.161080085781991e-10, 0.03555872480929, 5.151922419610025e-10,
          0.03730024048937033, 0.10541030477402333, 0.17862267733167023,
          0.06248696950024344, 0.10991112194080724]
    @test isapprox(w6.weights, wt, rtol = 5.0e-5)

    obj = SR(; rf = rf)
    w7 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r3 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret3 = dot(portfolio.mu, w7.weights)
    wt = [5.791704589818663e-10, 1.4777512342996448e-9, 1.4920733133812998e-9,
          8.941347428424144e-10, 0.6999099125632519, 2.145377355161713e-10,
          0.029295630576512924, 1.1027104693788755e-9, 1.8864271969797675e-9,
          8.43330450121613e-10, 1.4937081011622384e-9, 1.4856958187000145e-10,
          1.0768233412852032e-10, 3.8855123608537257e-10, 1.2149887816181597e-10,
          0.15181164107816766, 0.04226710946215913, 1.3947899372714116e-9,
          0.07671569251341252, 1.6615602330924226e-9]
    riskt = 0.00431255671125957
    rett = 0.0015948388159746803
    @test isapprox(w7.weights, wt)
    @test isapprox(r3, riskt)
    @test isapprox(ret3, rett)

    w8 = optimise2!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [5.8071005976616953e-11, 1.541427902953817e-10, 1.7913932166632589e-10,
          7.782119293420651e-11, 0.5637246749896364, 1.8931437657935694e-11,
          0.026029768151395943, 1.9414490982830922e-10, 1.259249686191089e-10,
          7.208065397311031e-11, 1.1199492525822813e-9, 1.215314166911757e-11,
          9.41885824068697e-12, 3.5763702301181766e-11, 1.0455689250552838e-11,
          0.14917727800463884, 0.12978984415928974, 3.9155754084852714e-10,
          0.13127843200243125, 2.330533005896693e-10]
    @test isapprox(w8.weights, wt)

    w9 = optimise2!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    wt = [5.450446471466287e-9, 1.3512126713442444e-8, 1.5151665151427853e-8,
          7.976975650100931e-9, 0.6065958578622958, 1.8554190775039432e-9,
          0.028288084366048082, 1.1813144626604747e-8, 1.4369857514985968e-8,
          6.665738536162035e-9, 1.8203193050019052e-8, 1.257919516215051e-9,
          9.552968174589085e-10, 3.3556538128122655e-9, 1.0433283283023054e-9,
          0.14977639360164002, 0.1082338279016761, 1.701911449061477e-8,
          0.10710570148059825, 1.6157861947293164e-8]
    @test isapprox(w9.weights, wt, rtol = 5.0e-5)

    obj = MaxRet()
    w10 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r4 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret4 = dot(portfolio.mu, w10.weights)
    wt = [1.3019302358014325e-8, 1.3979706940965055e-8, 1.670240049189312e-8,
          1.6104356899062238e-8, 2.5495489803481445e-7, 1.0170483295381582e-8,
          0.9999995050048601, 1.1436904494135492e-8, 1.492977755510578e-8,
          1.212324914648131e-8, 1.1410343061057653e-8, 1.0141009010483304e-8,
          9.120987910382183e-9, 1.0868081833333878e-8, 9.494553530172169e-9,
          2.3389178339261675e-8, 1.777099881408979e-8, 1.1725146345613458e-8,
          1.5069511913545612e-8, 1.2584249870605286e-8]
    riskt = 0.012237371871856062
    rett = 0.001845375304551891
    @test isapprox(w10.weights, wt)
    @test isapprox(r4, riskt)
    @test isapprox(ret4, rett)

    w11 = optimise2!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [6.184818023388107e-10, 9.270209690896355e-10, 1.625727765129299e-9,
          1.2702031382871368e-9, 0.8503448667005368, 8.234230282459137e-10,
          0.14965511529944078, 1.8812024811068833e-11, 1.1594526094497495e-9,
          1.7436144865199502e-10, 5.659330075801571e-11, 7.768388767333282e-10,
          1.0972252733499826e-9, 4.4072771043489173e-10, 1.0935450396733876e-9,
          3.753052447991426e-9, 2.05335213852063e-9, 1.1963791294032173e-10,
          1.3871915795314854e-9, 6.043753072340663e-10]
    @test isapprox(w11.weights, wt)

    w12 = optimise2!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    wt = wt = [1.0436536369860602e-9, 1.1989953779953557e-9, 1.464622032811774e-9,
               1.3427482165971476e-9, 0.8533952746506102, 4.935314543235269e-10,
               0.14660470645477996, 8.379594796066728e-10, 1.3262258615336055e-9,
               9.158610967401738e-10, 8.121781642332045e-10, 5.143517373531351e-10,
               3.696751986345881e-10, 6.50560988974405e-10, 3.7768409279100334e-10,
               2.4590065299397058e-9, 1.6858266276584893e-9, 8.793941840416185e-10,
               1.4336339581766902e-9, 1.0887011785012557e-9]
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

@testset "SLPM" begin
    portfolio = Portfolio2(; prices = prices,
                           solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                            :params => Dict("verbose" => false))))
    asset_statistics2!(portfolio)
    rm = SLPM2(; target = rf)

    obj = MinRisk()
    w1 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r1 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret1 = dot(portfolio.mu, w1.weights)
    wt = [1.1230805069911956e-8, 0.05524472463362737, 1.0686041548307544e-8,
          0.0043185378999763225, 0.033597348034736865, 1.0487157577222361e-8,
          1.1738886913269633e-9, 0.12478148562530009, 3.4647395424618816e-9,
          3.8805677196069256e-9, 0.3005648369145803, 2.0034183913036616e-9,
          1.0927362747553375e-9, 0.10661826438516031, 0.003123732919975542,
          0.021391817407374183, 0.003595424842043441, 0.22964898912299475,
          5.129978967782806e-9, 0.117114789064897]
    riskt = 0.005418882634929856
    rett = 0.0004088880259308715
    @test isapprox(w1.weights, wt)
    @test isapprox(r1, riskt)
    @test isapprox(ret1, rett)

    w2 = optimise2!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [3.553751623019336e-9, 0.0552444506714033, 3.3792972175226587e-9,
          0.004319267700904996, 0.033597486252238504, 3.3178770246587074e-9,
          3.530263895833413e-10, 0.12478100673912003, 1.082173144802101e-9,
          1.2142739606639318e-9, 0.3005650138904127, 6.172236686305243e-10,
          3.272670368272328e-10, 0.10661801005424905, 0.003123621247828574,
          0.02139180763720946, 0.0035949221271092354, 0.22965076223179196,
          1.6119446394023392e-9, 0.11711363599089746]
    @test isapprox(w2.weights, wt)

    w3 = optimise2!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    wt = [8.960032986198268e-9, 0.05524502934743527, 8.531490753416422e-9,
          0.004319426319869003, 0.03359614618594921, 8.379240639976445e-9,
          1.0539120789014738e-9, 0.12478035258171694, 2.855174321889051e-9,
          3.1807196110268267e-9, 0.30056524386869504, 1.706724783203312e-9,
          9.907199378298648e-10, 0.10661840631750075, 0.003124219404851094,
          0.021391583878468307, 0.0035949097819517437, 0.22965048214875944,
          4.163566475384283e-9, 0.11711416034322167]
    @test isapprox(w3.weights, wt)

    obj = Util(; l = l)
    w4 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r2 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret2 = dot(portfolio.mu, w4.weights)
    wt = [1.892614163740768e-9, 0.05510469378866011, 2.336572603808684e-9,
          0.0004235261366778282, 0.06590298875275914, 9.60649390058247e-10,
          4.495790137196085e-10, 0.11928868236657098, 1.3194622743579184e-9,
          1.2532033673577644e-9, 0.2937381531255968, 3.557016906443741e-10,
          1.75711616979717e-10, 0.07604922161635236, 1.3912072068726968e-9,
          0.03452489138234348, 0.02823700959313513, 0.22320354704027148,
          2.8171948818718627e-9, 0.10352727324573653]
    riskt = 0.005439630405063694
    rett = 0.0004936947590309835
    @test isapprox(w4.weights, wt)
    @test isapprox(r2, riskt)
    @test isapprox(ret2, rett)

    w5 = optimise2!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [1.1179343918984971e-10, 0.054976790588896596, 1.7713391003665344e-10,
          0.000633725765454258, 0.06550332883993616, 2.3043194331626305e-11,
          9.784822751076497e-11, 0.11941492846830355, 2.6615472963293778e-11,
          1.8716678076476947e-11, 0.2937827494968045, 1.1133950578913522e-10,
          1.37189305281988e-10, 0.0764029089634452, 3.804971672707788e-11,
          0.03435573176652704, 0.028126826103343827, 0.2230959623916138,
          2.3934818713608943e-10, 0.10370704663459744]
    @test isapprox(w5.weights, wt)

    w6 = optimise2!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    wt = [2.909374492519179e-9, 0.054980576422244756, 3.565867766679885e-9,
          0.0006319533873603366, 0.06548847846241447, 1.553456588968246e-9,
          8.009894415787075e-10, 0.11941669348400365, 2.0520045922813665e-9,
          1.972894188888827e-9, 0.29378579258366055, 6.654042713665044e-10,
          4.054332929406695e-10, 0.0764175574974796, 2.168836496219484e-9,
          0.03435048211474188, 0.028114016486312555, 0.22310042002755265,
          4.189571372445089e-9, 0.10371400925039707]
    @test isapprox(w6.weights, wt, rtol = 5.0e-7)

    obj = SR(; rf = rf)
    w7 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r3 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret3 = dot(portfolio.mu, w7.weights)
    wt = [1.9161992029945534e-8, 4.442467014160941e-8, 3.487731975169222e-8,
          2.172326848650473e-8, 0.6654321506924412, 6.20892532022181e-9,
          0.03807260712526902, 3.6516022610300514e-8, 4.3159008520930105e-8,
          1.8350537901763542e-8, 4.619460482355355e-8, 5.0197040711936325e-9,
          3.3977158843464672e-9, 1.2834736295215969e-8, 3.5853236437253736e-9,
          0.17459230019672953, 0.10412390455189192, 4.844528935490425e-8,
          0.017778656482209734, 3.7052339729479755e-8]
    riskt = 0.00909392522496688
    rett = 0.0015869580721210722
    @test isapprox(w7.weights, wt)
    @test isapprox(r3, riskt)
    @test isapprox(ret3, rett)

    w8 = optimise2!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [1.3248921152324648e-8, 4.085757001568184e-8, 2.5052363399873344e-8,
          1.543168580522427e-8, 0.5737596378553825, 4.3480211803596126e-9,
          0.029566570892699606, 5.302664319377606e-8, 2.2678880685535206e-8,
          1.3224044271395122e-8, 1.1753423586262955e-7, 3.3659140729627245e-9,
          2.441172159950741e-9, 9.365453914041405e-9, 2.532713913601762e-9,
          0.1671874274459401, 0.15962776043775584, 7.614872045448292e-8,
          0.06985816973821751, 3.437366426339058e-8]
    @test isapprox(w8.weights, wt, rtol = 1e-6)

    w9 = optimise2!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    wt = [2.956369706502196e-7, 7.203516650576015e-7, 5.527631754013819e-7,
          3.5624630813868357e-7, 0.6120717648980301, 1.0106563076307954e-7,
          0.03616496177959474, 6.730914491132973e-7, 5.529892319617771e-7,
          2.8221149925910375e-7, 9.319704068880028e-7, 8.419900388670793e-8,
          6.018647280784777e-8, 1.9489436736726285e-7, 6.199574934187237e-8,
          0.1695100352935956, 0.13598076586849808, 8.915232281848942e-7, 0.0462661197643809,
          5.932707417147085e-7]
    @test isapprox(w9.weights, wt, rtol = 1.0e-7)

    obj = MaxRet()
    w10 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r4 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret4 = dot(portfolio.mu, w10.weights)
    wt = [8.874552417228264e-9, 1.0569760898904627e-8, 2.0363934073123463e-8,
          1.547715765009297e-8, 4.267804159375552e-7, 2.8607270157041685e-9,
          0.9999993860106617, 4.216763165728999e-9, 1.2576645584153888e-8,
          5.097663390006654e-9, 3.971251769425353e-9, 2.9266677130332086e-9,
          2.874302327436276e-9, 3.1933824668323967e-9, 2.8301314578492617e-9,
          4.222188374473967e-8, 2.2946307691638603e-8, 4.6341884676924975e-9,
          1.3929168448114664e-8, 7.644434060912258e-9]
    riskt = 0.024842158070968706
    rett = 0.0018453754259793952
    @test isapprox(w10.weights, wt)
    @test isapprox(r4, riskt)
    @test isapprox(ret4, rett)

    w11 = optimise2!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [7.210116063134277e-11, 1.176606228072633e-10, 2.1258483097832237e-10,
          1.6536274651174382e-10, 0.8503141905762418, 1.273544562974361e-10,
          0.14968580705048062, 1.992517244424829e-12, 1.5128034960181728e-10,
          1.850938653797299e-11, 1.2078508927108743e-11, 1.1804552928693339e-10,
          1.5680771674497215e-10, 6.957317957422098e-11, 1.6101143219670432e-10,
          4.425491709419713e-10, 2.7404957392346757e-10, 9.559038664076399e-12,
          1.8545492666656794e-10, 7.730240246200862e-11]
    @test isapprox(w11.weights, wt)

    w12 = optimise2!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    wt = [2.4129797561271965e-9, 2.701958395018328e-9, 3.3576629746257995e-9,
          3.0274737839898444e-9, 0.8533951483772857, 1.0606695596971499e-9,
          0.14660480975447185, 1.8393387540596498e-9, 2.9260550631872547e-9,
          1.989898450820525e-9, 1.7675044799733574e-9, 1.100567325980679e-9,
          7.837356706523964e-10, 1.4112632681808324e-9, 7.965335460056496e-10,
          5.471987951034473e-9, 3.760034250575122e-9, 1.9365276473399495e-9,
          3.1320916121578367e-9, 2.3919599353675564e-9]
    @test isapprox(w12.weights, wt, rtol = 1.0e-7)

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
    rm.settings.ub = r1 * (1.000001)
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r1 ||
          abs(calc_risk(portfolio; type = :Trad2, rm = rm) - r1) < 1e-8

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

@testset "WR" begin
    portfolio = Portfolio2(; prices = prices,
                           solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                            :params => Dict("verbose" => false))))
    asset_statistics2!(portfolio)
    rm = WR2()

    obj = MinRisk()
    w1 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r1 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret1 = dot(portfolio.mu, w1.weights)
    wt = [2.8612942682723194e-12, 0.22119703870515675, 2.828265759908966e-12,
          2.1208855227168895e-12, 3.697781891063451e-12, 3.24353226480368e-12,
          0.02918541183751788, 4.420452260557843e-12, 2.3374667530908414e-12,
          2.8919479333342058e-12, 0.5455165570312099, 1.4490684503326206e-12,
          1.9114786537154165e-12, 2.7506310060540026e-12, 3.640894035272413e-11,
          2.7909315847715066e-12, 1.694217734278189e-12, 3.798024068784819e-12,
          2.7258514165515688e-12, 0.20410099234818463]
    riskt = 0.03217605105120276
    rett = 0.0005011526784679896
    @test isapprox(w1.weights, wt)
    @test isapprox(r1, riskt)
    @test isapprox(ret1, rett)

    w2 = optimise2!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [8.31617952752212e-13, 0.22119703875062227, 6.832160545751061e-13,
          6.96193291795788e-13, 4.036019076776619e-12, 4.3471614749828785e-13,
          0.029185411841700936, 9.078156634695366e-13, 6.665895218019583e-13,
          1.8373133578563436e-12, 0.5455165570616287, 2.7833273599028926e-13,
          5.493760560329698e-13, 4.481197759509806e-13, 3.32575216803394e-11,
          1.1556828005777454e-12, 4.0624744116434915e-12, 6.860861190643671e-13,
          8.372817139938184e-13, 0.20410099229467973]
    @test isapprox(w2.weights, wt)

    w3 = optimise2!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    wt = [2.635110175804277e-10, 0.22119708420320755, 3.820489210458886e-10,
          1.4187173357103828e-9, 2.6363330885995266e-9, 5.189538035605022e-10,
          0.02918540888981707, 7.808080527361188e-10, 3.6109547229216355e-10,
          5.363142216800803e-10, 0.5455165396306245, 1.2773165351656046e-10,
          3.490104383331551e-10, 1.9841993233744004e-10, 1.7436321318876394e-8,
          5.702494806667545e-10, 1.2872101855890893e-9, 3.0511723902483676e-10,
          4.117916217763004e-10, 0.2041009396927171]
    @test isapprox(w3.weights, wt)

    obj = Util(; l = l)
    w4 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r2 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret2 = dot(portfolio.mu, w4.weights)
    wt = [4.619204869507631e-11, 0.2211970401400894, 4.3701774419007517e-11,
          8.353407625940733e-11, 1.358657025936017e-10, 1.0578194758385017e-10,
          0.029185412435982716, 1.8405841561907286e-10, 3.740225068356339e-11,
          6.44640015460276e-11, 0.5455165540499841, 1.992397665103746e-11,
          4.891135996048911e-11, 3.833815555309313e-11, 6.709113288205949e-10,
          1.1391046858565562e-10, 3.582241940912807e-11, 5.843933123280907e-11,
          4.670206096914456e-11, 0.20410099163998444]
    riskt = 0.03217605106792421
    rett = 0.0005011526792393935
    @test isapprox(w4.weights, wt)
    @test isapprox(r2, riskt)
    @test isapprox(ret2, rett)

    w5 = optimise2!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [6.780104323028494e-11, 0.22119703859779105, 6.276295210763981e-11,
          6.945685168435587e-11, 1.272876224890368e-10, 5.0924754939132304e-11,
          0.029185411803606478, 4.2025073814241164e-11, 5.136255227405268e-11,
          8.72807827590185e-11, 0.5455165573896047, 4.3304404498191726e-11,
          3.360166563092323e-11, 5.6057018320553576e-11, 2.6187078514462165e-10,
          3.8203524552350216e-11, 5.947703697380478e-11, 7.050017028250539e-11,
          6.195177931996556e-11, 0.20410099102512988]
    @test isapprox(w5.weights, wt)

    w6 = optimise2!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    wt = [1.1929038472765297e-10, 0.22119705255880093, 1.580011094715284e-10,
          4.847576119579443e-10, 1.120981268022772e-9, 1.9690189492108395e-10,
          0.02918541304709744, 1.803814386688887e-10, 1.5236190246315402e-10,
          2.4029948591279455e-10, 0.5455165527590767, 3.7615507614648974e-11,
          1.209905706287236e-10, 6.875025162230649e-11, 1.725560852711778e-9,
          1.970196651568895e-10, 6.270793283913616e-10, 1.034939719979999e-10,
          2.0441141224195639e-10, 0.20410097589712836]
    @test isapprox(w6.weights, wt, rtol = 1.0e-7)

    obj = SR(; rf = rf)
    w7 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r3 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret3 = dot(portfolio.mu, w7.weights)
    wt = [6.957399908772388e-9, 2.039271731107526e-8, 5.497898695084438e-9,
          1.1584017088731345e-8, 0.3797661371235164, 1.9162230097305403e-9,
          0.17660512608552742, 1.0666210782547244e-8, 1.0225338760635262e-8,
          0.04075088574289245, 0.05638221165264284, 2.089109162284139e-9,
          1.23279550928153e-9, 9.013331222315118e-9, 2.1778889815995123e-9,
          0.15854733523481268, 0.18794817199402036, 1.1268704949879534e-8,
          3.4599644297968083e-8, 4.545308055104026e-9]
    riskt = 0.04173382316607199
    rett = 0.0014131701721435356
    @test isapprox(w7.weights, wt)
    @test isapprox(r3, riskt)
    @test isapprox(ret3, rett)

    w8 = optimise2!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [2.346041541755758e-8, 2.790563286730082e-7, 2.0230169044344763e-8,
          4.148314099306561e-8, 0.3797663868865331, 6.721680190226824e-9,
          0.17660488556016016, 4.094018803728532e-8, 3.2462164530088474e-8,
          0.040750676423078724, 0.0563819780260819, 8.094382876600906e-9,
          4.282569514075461e-9, 6.146028749441822e-8, 7.254495678879109e-9,
          0.15854739341595234, 0.187947992353654, 3.965732629982512e-8, 9.90370453631504e-8,
          2.319434573454084e-8]
    @test isapprox(w8.weights, wt)

    w9 = optimise2!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    wt = [1.3449084787703177e-7, 4.694914111656391e-7, 1.0656708224145255e-7,
          2.1792722845632984e-7, 0.37976702290626235, 3.7648369976100124e-8,
          0.1766045318438728, 2.2074894042984511e-7, 1.9142894435540533e-7,
          0.04075031946476531, 0.056380347301646885, 4.0148020588606196e-8,
          2.4311369893956776e-8, 1.85275632936872e-7, 3.9552840747762915e-8,
          0.1585475070017042, 0.18794747366731218, 2.450350381229234e-7,
          7.912653792134936e-7, 9.39233301941589e-8]
    @test isapprox(w9.weights, wt)

    obj = MaxRet()
    w10 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r4 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret4 = dot(portfolio.mu, w10.weights)
    wt = [1.5566238289527526e-9, 1.570956878493887e-9, 1.608762274371583e-9,
          1.6258884417989995e-9, 1.4600423065623333e-7, 1.0857056194353091e-9,
          0.9999998292073036, 1.3820011237558862e-9, 1.5713320679343279e-9,
          1.430630625805485e-9, 1.3707526492120582e-9, 1.0935734943467402e-9,
          8.72584405607747e-10, 1.247652840328386e-9, 9.237093680924222e-10,
          1.3747133091550288e-9, 1.577564048696993e-9, 1.4136760653056833e-9,
          1.5661129788821178e-9, 1.51622582562161e-9]
    riskt = 0.24229070767001235
    rett = 0.001845375606854123
    @test isapprox(w10.weights, wt)
    @test isapprox(r4, riskt)
    @test isapprox(ret4, rett)

    w11 = optimise2!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [3.5443437027516e-11, 5.2350950229862524e-11, 9.066935701147142e-11,
          7.120923208257929e-11, 0.8503259395366981, 4.355477349277696e-11,
          0.149674059466214, 2.4290625152597366e-12, 6.509078740819352e-11,
          1.0998068341006642e-11, 1.701927079115916e-12, 4.105730646991286e-11,
          5.880134786712474e-11, 2.270001553981933e-11, 5.835110319164303e-11,
          2.0849838100796641e-10, 1.1412283881505805e-10, 8.007551744972036e-12,
          7.751897551597984e-11, 3.458306521316913e-11]
    @test isapprox(w11.weights, wt)

    w12 = optimise2!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    wt = [2.4427026397905277e-9, 2.953342366847875e-9, 3.488355282693272e-9,
          3.3039868175714513e-9, 0.8533950216137636, 1.3232043581911538e-9,
          0.14660493036607455, 2.166597089120636e-9, 3.3970019446108583e-9,
          2.4108748932365233e-9, 2.131897673067489e-9, 1.3895315918812704e-9,
          1.013781789198535e-9, 1.7170881348384435e-9, 1.04480062674174e-9,
          6.196346478260153e-9, 4.233214436766981e-9, 2.2605081913012243e-9,
          3.736694833735987e-9, 2.810232763884542e-9]
    @test isapprox(w12.weights, wt, rtol = 5.0e-7)

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

@testset "CVaR" begin
    portfolio = Portfolio2(; prices = prices,
                           solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                            :params => Dict("verbose" => false))))
    asset_statistics2!(portfolio)
    rm = CVaR2()

    obj = MinRisk()
    w1 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r1 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret1 = dot(portfolio.mu, w1.weights)
    wt = [9.965769176302831e-11, 0.04242033378148941, 9.902259604479418e-11,
          2.585550936025974e-10, 0.007574028506215674, 1.1340405766435789e-10,
          1.3814642470526227e-11, 0.09464947974750273, 4.637745432335755e-11,
          6.484701166044592e-11, 0.3040110652312709, 5.940889071027648e-11,
          3.420745138676034e-11, 0.06564166947730173, 9.544192184784114e-11,
          0.029371611149186894, 1.241093002048221e-10, 0.36631011287979914,
          5.953639120278758e-11, 0.09002169815885094]
    riskt = 0.01704950212555889
    rett = 0.0003860990591135937
    @test isapprox(w1.weights, wt)
    @test isapprox(r1, riskt)
    @test isapprox(ret1, rett)

    w2 = optimise2!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [9.204153687833781e-11, 0.04242033344850122, 9.85123787201143e-11,
          2.742462482595728e-10, 0.007574028452637937, 1.0444892006359854e-10,
          1.1253189456551386e-11, 0.09464947950883604, 4.304524873576078e-11,
          5.769528999444338e-11, 0.3040110652564113, 5.2022422074226226e-11,
          2.8850585492519713e-11, 0.06564166930506558, 9.716068090161583e-11,
          0.029371611163095106, 1.2140041190522942e-10, 0.366310112784046,
          5.621940710019265e-11, 0.09002169904451053]
    @test isapprox(w2.weights, wt)

    w3 = optimise2!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    wt = [1.0779002913123264e-9, 0.0424202478835117, 1.1394636976177199e-9,
          3.2034009048719517e-9, 0.007574041437995398, 1.214157664740724e-9,
          2.0866351379515966e-10, 0.09464947248139145, 5.475138966935703e-10,
          7.320881150782838e-10, 0.304010968264335, 7.160260081039788e-10,
          4.2716139435542236e-10, 0.06564164417407643, 1.1869177983842032e-9,
          0.02937161641119216, 1.343287701885978e-9, 0.3663101613158242,
          6.666278889397393e-10, 0.09002183556846477]
    @test isapprox(w3.weights, wt)

    obj = Util(; l = l)
    w4 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r2 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret2 = dot(portfolio.mu, w4.weights)
    wt = [3.59415027814126e-12, 0.03635916778167782, 3.552811195795529e-12,
          1.2653345986437583e-11, 0.017177347037601237, 3.337778093941137e-12,
          2.5795636217644956e-13, 0.09164255461978174, 1.734150257351883e-12,
          2.0375788095235085e-12, 0.32258428294244085, 1.6071720149229441e-12,
          7.478631306913877e-13, 0.03955071531394016, 3.4490687806596575e-12,
          0.030919378393966135, 4.608903322785033e-12, 0.3733298428934439,
          2.2844085399535255e-12, 0.08843671097728309]
    riskt = 0.017056094628321805
    rett = 0.0004072001780847205
    @test isapprox(w4.weights, wt)
    @test isapprox(r2, riskt)
    @test isapprox(ret2, rett)

    w5 = optimise2!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [1.2446651084459793e-10, 0.03635920624448488, 1.2099878252057046e-10,
          4.3353705390596115e-10, 0.017177327412953393, 1.1523416518156477e-10,
          1.1529023942129099e-11, 0.091642549012939, 6.068436102923306e-11,
          7.000020425265046e-11, 0.32258415691878095, 5.729097510879122e-11,
          2.6990419802016365e-11, 0.039550772304670946, 1.2030281101532894e-10,
          0.030919377851809874, 1.525456685804843e-10, 0.373329855951351,
          7.83789622418283e-11, 0.08843675293105117]
    @test isapprox(w5.weights, wt)

    w6 = optimise2!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    wt = [1.8770987535438225e-10, 0.03635922666523639, 1.9543535804493658e-10,
          5.920383859192107e-10, 0.017177311523088035, 1.753401786031763e-10,
          4.144490215167871e-11, 0.0916425570130048, 1.0276360004231398e-10,
          1.1514827254900474e-10, 0.3225840596074403, 9.619163005772905e-11,
          5.6449361466469335e-11, 0.0395507982905123, 1.8272605765002315e-10,
          0.030919377989654204, 2.4645160197705085e-10, 0.3733298632135168,
          1.2924986342087758e-10, 0.08843680357659808]
    @test isapprox(w6.weights, wt, rtol = 5.0e-7)

    obj = SR(; rf = rf)
    w7 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r3 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret3 = dot(portfolio.mu, w7.weights)
    wt = [2.305962223730381e-9, 3.061529980299523e-9, 3.6226755773356135e-9,
          1.968988878444111e-9, 0.562845489616387, 6.289605285168684e-10,
          0.044341929816432854, 5.465596947274736e-9, 3.128822366888805e-9,
          1.6003971393612084e-9, 4.52394176361636e-9, 5.75356193927518e-10,
          3.1728380155852195e-10, 1.240519587265295e-9, 3.422838872379099e-10,
          0.20959173183485763, 0.18322079783245407, 6.034806498955341e-9,
          1.1803331196573864e-8, 4.279412029260546e-9]
    riskt = 0.03005421217653932
    rett = 0.0015191213711409513
    @test isapprox(w7.weights, wt)
    @test isapprox(r3, riskt)
    @test isapprox(ret3, rett)

    w8 = optimise2!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [2.304102546507255e-9, 3.4058580184231964e-9, 3.264755599420317e-9,
          2.168529122297692e-9, 0.5593117496370377, 7.139134073206089e-10,
          0.029474976465948034, 4.9115201797004046e-8, 2.799982453416685e-9,
          1.6115667456355964e-9, 8.831047243202884e-9, 6.334407262324075e-10,
          4.1948103829488986e-10, 1.3615475408342457e-9, 4.3425958678632566e-10,
          0.20296977157601848, 0.20824336902427606, 2.80725022409301e-8,
          2.143306519713471e-8, 6.727466637284185e-9]
    @test isapprox(w8.weights, wt)

    w9 = optimise2!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    wt = [6.64088087345985e-8, 9.538860932028128e-8, 1.0453296168596982e-7,
          5.7868546146211784e-8, 0.5622705387604069, 2.1054154746516597e-8,
          0.041923329563299166, 2.0823113529748093e-7, 8.937992205180785e-8,
          4.70377466776035e-8, 1.957973941516057e-7, 1.911064451024525e-8,
          1.2600789308841253e-8, 3.954227708856903e-8, 1.3000474018720965e-8,
          0.20851348525481309, 0.18729086427243372, 2.3196503370395504e-7,
          4.399321628916103e-7, 1.402983868138997e-7]
    @test isapprox(w9.weights, wt)

    obj = MaxRet()
    w10 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r4 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret4 = dot(portfolio.mu, w10.weights)
    wt = [1.2979096100520238e-8, 1.3939595287583712e-8, 1.6664693909263272e-8,
          1.6061904799427314e-8, 2.5289048640527673e-7, 1.0111632690844616e-8,
          0.9999995079388162, 1.1387303215317185e-8, 1.4887953777768852e-8,
          1.2073758025755345e-8, 1.1359241573256225e-8, 1.0083482858838886e-8,
          9.063418984706977e-9, 1.0812267724771105e-8, 9.4351700847673e-9,
          2.3329395161801666e-8, 1.7731658764712164e-8, 1.1676397354409723e-8,
          1.503052388589023e-8, 1.2543203151639056e-8]
    riskt = 0.08082926752528491
    rett = 0.001845375306063485
    @test isapprox(w10.weights, wt)
    @test isapprox(r4, riskt)
    @test isapprox(ret4, rett)

    w11 = optimise2!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [1.622468763587675e-11, 2.4305243326642927e-11, 4.2602118130674744e-11,
          3.3294464132139124e-11, 0.8503255339884118, 2.152943423043018e-11,
          0.14967446554001587, 5.218627470261068e-13, 3.039577793482502e-11,
          4.597121407190349e-12, 1.4530843735638327e-12, 2.030952150941086e-11,
          2.8702503009729792e-11, 1.1510274994891938e-11, 2.8604495580622442e-11,
          9.834475691645756e-11, 5.3801390739755065e-11, 3.1622056975748866e-12,
          3.635795488049805e-11, 1.5855285676325412e-11]
    @test isapprox(w11.weights, wt)

    w12 = optimise2!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    wt = [2.4358139548252643e-9, 2.9443323823815057e-9, 3.4781287468046434e-9,
          3.294150521464759e-9, 0.8533951211371982, 1.3187368397190805e-9,
          0.14660483098984345, 2.1595613343429504e-9, 3.3865724969917085e-9,
          2.4029865798428798e-9, 2.12499035870577e-9, 1.3849425461316505e-9,
          1.0104473225853474e-9, 1.7114267991768829e-9, 1.0412585823551084e-9,
          6.179367364251867e-9, 4.220597690045663e-9, 2.253256560559402e-9,
          3.724982057897676e-9, 2.80140636669256e-9]
    @test isapprox(w12.weights, wt, rtol = 1.0e-5)

    # Risk upper bound
    obj = MaxRet()
    rm.settings.ub = r1
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r1 ||
          abs(calc_risk(portfolio; type = :Trad2, rm = rm) - r1) < 1e-9

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
          abs(calc_risk(portfolio; type = :Trad2, rm = rm) - r1) < 1e-10

    rm.settings.ub = r2
    optimise2!(portfolio; rm = rm, obj = obj)
    calc_risk(portfolio; type = :Trad2, rm = rm)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r2 ||
          abs(calc_risk(portfolio; type = :Trad2, rm = rm) - r2) < 1e-10

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
          abs(dot(portfolio.mu, w16.weights) - ret4) < 1e-10
end

@testset "EVaR" begin
    portfolio = Portfolio2(; prices = prices,
                           solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                            :params => Dict("verbose" => false,
                                                                            "max_step_fraction" => 0.75))))
    asset_statistics2!(portfolio)
    rm = EVaR2()

    obj = MinRisk()
    w1 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r1 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret1 = dot(portfolio.mu, w1.weights)
    wt = [1.2500112838486011e-8, 0.15679318740948175, 1.0361131210275206e-8,
          0.01670757435974688, 1.501287503554962e-8, 6.061816596694208e-8,
          0.014452439886462113, 0.15570664400078943, 9.522408711497533e-9,
          1.059085220479153e-8, 0.452447219917494, 8.305434093731495e-9,
          1.2081476879327763e-8, 2.3952270923291378e-8, 0.004794389308245565,
          1.7142647790367886e-7, 0.01841950032750946, 1.9211685081872636e-7,
          1.2233795563491733e-8, 0.18067850606841873]
    riskt = 0.024507972823062964
    rett = 0.00046038550243244597
    @test isapprox(w1.weights, wt)
    @test isapprox(r1, riskt)
    @test isapprox(ret1, rett)

    w2 = optimise2!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [1.1175842274627593e-8, 0.15678962525657866, 9.269168746643073e-9,
          0.016707074814442415, 1.3452573176413182e-8, 5.436355786124658e-8,
          0.014451271346349, 0.15570865514232668, 8.500395881111563e-9,
          9.498747909712649e-9, 0.4524457444044333, 7.375543927729117e-9,
          1.0692599637688104e-8, 2.157786807878457e-8, 0.0047941761453754485,
          1.5347677599844998e-7, 0.018424665778813513, 1.7044737317526053e-7,
          1.0938910978010825e-8, 0.18067830634232349]
    @test isapprox(w2.weights, wt)

    w3 = optimise2!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    wt = [7.744149706504591e-10, 0.15679410893441026, 7.274156507917334e-10,
          0.016706213514055972, 1.5810201387153583e-9, 1.6952606171928384e-9,
          0.014453448797547326, 0.15570508057409402, 5.943675126172893e-10,
          6.193547072116074e-10, 0.4524506451372982, 3.8768620409309906e-10,
          4.1492798285227507e-10, 9.744373592147397e-10, 0.004792324631608908,
          4.6408869326508805e-9, 0.01841570061795783, 4.774631989432649e-9,
          7.63777788259464e-10, 0.18068245984484557]
    @test isapprox(w3.weights, wt)

    obj = Util(; l = l)
    w4 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r2 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret2 = dot(portfolio.mu, w4.weights)
    wt = [1.4475965511299271e-8, 0.15139053006670328, 1.2531364593722484e-8,
          0.02042309439335846, 2.1361563540103295e-8, 4.012521850613263e-8,
          0.01722100416172815, 0.1532392038577736, 1.1517423032461546e-8,
          1.2693072417282946e-8, 0.4464468008721702, 9.157260802561859e-9,
          1.1830960942068228e-8, 2.3761186886959645e-8, 1.4669914215262773e-7,
          6.144682912498466e-7, 0.027626609709476953, 1.631595856184254e-7,
          1.4944878185822058e-8, 0.18365166021287574]
    riskt = 0.024511874696597793
    rett = 0.0004794326659607715
    @test isapprox(w4.weights, wt)
    @test isapprox(r2, riskt)
    @test isapprox(ret2, rett)

    w5 = optimise2!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [5.4262370368099214e-9, 0.15109830119103693, 4.5753033355549586e-9,
          0.020595549491597333, 7.964018113907466e-9, 1.5349176697692086e-8,
          0.017131627832874405, 0.15324759220272538, 4.0337811229377734e-9,
          4.598713475553217e-9, 0.44642374758452036, 3.14646860021372e-9,
          4.116666022232492e-9, 8.820226856534203e-9, 5.280033483159844e-8,
          3.006208023615729e-7, 0.02786083166919229, 5.994243143472186e-8,
          5.387502906521676e-9, 0.18364187324639056]
    @test isapprox(w5.weights, wt, rtol = 5e-5)

    w6 = optimise2!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    wt = [2.2974521636472227e-10, 0.1511117107080329, 2.3772089567528915e-10,
          0.02058706960215576, 8.800029560475762e-10, 3.33220242180068e-10,
          0.01713205476125333, 0.15324961307673107, 1.971297743563386e-10,
          1.906147975235925e-10, 0.4464267013369022, 1.1241175348079011e-10,
          1.1533719330242375e-10, 2.646022456979656e-10, 1.0697230346457774e-9,
          5.0201710049166106e-9, 0.027855440141728288, 1.2652809114102488e-9,
          2.5056480589466414e-10, 0.18363740020667155]
    @test isapprox(w6.weights, wt)

    obj = SR(; rf = rf)
    w7 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r3 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret3 = dot(portfolio.mu, w7.weights)
    wt = [1.0750748140777434e-8, 3.269490337304986e-8, 1.1161941451849754e-8,
          1.3795466025857643e-8, 0.5351874067614019, 2.6718249477546367e-9,
          0.1390764348877217, 1.41282558079161e-8, 1.0656060597300996e-8,
          7.83717309959956e-9, 1.794801260303159e-8, 2.6229370477942236e-9,
          1.8308405319956406e-9, 6.011246604979923e-9, 1.9381716976717685e-9,
          0.18358697053484188, 0.14214899271554252, 1.7344741890623557e-8,
          3.394097823954422e-8, 9.767190110912097e-9]
    riskt = 0.03754976868195822
    rett = 0.0015728602397846448
    @test isapprox(w7.weights, wt)
    @test isapprox(r3, riskt)
    @test isapprox(ret3, rett)

    w8 = optimise2!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [1.9979357266227307e-8, 1.3433190080573913e-7, 1.9289466264215865e-8,
          2.6567127395872726e-8, 0.4887145767904928, 4.687357399750469e-9,
          0.11042519617829799, 6.329803422124635e-8, 1.9120266630287126e-8,
          1.626296836809088e-8, 1.1949389923023293e-7, 4.433503242518708e-9,
          2.9962250121467016e-9, 1.218049680765375e-8, 3.3390094757338038e-9,
          0.1801155318875559, 0.22074406904792815, 5.5588214838936996e-8,
          1.0546407189021632e-7, 1.9063826338160836e-8]
    @test isapprox(w8.weights, wt, rtol = 5e-7)

    w9 = optimise2!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    wt = [8.22397665559172e-8, 3.826190817815927e-7, 9.2494531582376e-8,
          1.116490369754742e-7, 0.5137842676302625, 1.9493350548570787e-8,
          0.11863275644718071, 1.5404503430726993e-7, 8.610208570285454e-8,
          6.013966797550019e-8, 2.0590880890358276e-7, 1.8543652214974115e-8,
          1.2825779526983523e-8, 4.58672619265406e-8, 1.3527189010888518e-8,
          0.18159860062783653, 0.18598228856404617, 1.809789890792263e-7,
          5.372930554387161e-7, 8.300338256936208e-8]
    @test isapprox(w9.weights, wt)

    obj = MaxRet()
    w10 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r4 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret4 = dot(portfolio.mu, w10.weights)
    wt = [3.4349721061540534e-9, 3.654412215248722e-9, 4.5795242614345734e-9,
          4.166579635738961e-9, 2.721474418332461e-7, 1.6176234392665889e-9,
          0.9999996718661417, 2.521138576067204e-9, 3.865486493372323e-9,
          2.762737441868102e-9, 2.4450889056720585e-9, 1.7125484610603246e-9,
          1.2415710732838844e-9, 2.0536581804908253e-9, 1.281393702188933e-9,
          5.965051328690624e-9, 4.7776301196822434e-9, 2.6493753543706035e-9,
          4.0069166797873965e-9, 3.2507084446056985e-9]
    riskt = 0.14084653100897324
    rett = 0.0018453755649407056
    @test isapprox(w10.weights, wt)
    @test isapprox(r4, riskt)
    @test isapprox(ret4, rett)

    w11 = optimise2!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [3.714812761322751e-11, 4.773899536284892e-11, 7.269098817719502e-11,
          6.02689343593572e-11, 0.8503244596483243, 1.3212530602986929e-11,
          0.14967553959927737, 1.4434873773786082e-11, 5.6011672428045124e-11,
          2.0563699040590723e-11, 1.1468895310076392e-11, 1.223933138963094e-11,
          2.5822375180978835e-11, 1.5505601538916262e-12, 2.4392582540848822e-11,
          1.499362309475636e-10, 8.733241804718736e-11, 1.8229290852346547e-11,
          6.3532171924317e-11, 3.5824978876083113e-11]
    @test isapprox(w11.weights, wt)

    w12 = optimise2!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    wt = [2.630332111490297e-9, 2.8208813441505202e-9, 3.4946060056339277e-9,
          3.1476772348502665e-9, 0.8533950846247867, 1.2168423662193806e-9,
          0.1466048715410871, 1.8953132288109335e-9, 3.003500110840219e-9,
          1.9905919789926412e-9, 1.7863560639864318e-9, 1.1568945873970863e-9,
          8.405498765740556e-10, 1.4870828390487235e-9, 8.482710333148072e-10,
          6.0247478526723576e-9, 3.882589512102091e-9, 2.013333761943543e-9,
          3.2065860439340593e-9, 2.3879703550881473e-9]
    @test isapprox(w12.weights, wt)

    # Risk upper bound
    obj = MaxRet()
    rm.settings.ub = r1
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r1 ||
          abs(calc_risk(portfolio; type = :Trad2, rm = rm) - r1) < 5e-6

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
          abs(dot(portfolio.mu, w16.weights) - ret4) <= 1e-10

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

@testset "RVaR" begin
    portfolio = Portfolio2(; prices = prices,
                           solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                            :params => Dict("verbose" => false,
                                                                            "max_step_fraction" => 0.75))))
    asset_statistics2!(portfolio)
    rm = RVaR2()

    obj = MinRisk()
    w1 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r1 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret1 = dot(portfolio.mu, w1.weights)
    wt = [5.102262457628692e-9, 0.21104494400160803, 5.341766086210288e-9,
          2.5458382238901392e-8, 1.697696472902229e-8, 7.287515616039478e-9,
          0.03667031714382797, 0.061041476346139684, 4.093926758298615e-9,
          4.303160140655642e-9, 0.49353591974074, 2.1672264824822902e-9,
          3.926886474939328e-9, 4.083625597792755e-9, 1.043237724356759e-8,
          1.1621198331723714e-8, 2.5232405645111758e-8, 1.1835541180026409e-8,
          4.9679012600678e-9, 0.19770719993654404]
    riskt = 0.028298069755304314
    rett = 0.000508233446626652
    @test isapprox(w1.weights, wt)
    @test isapprox(r1, riskt)
    @test isapprox(ret1, rett)

    w2 = optimise2!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [1.6668841800133678e-9, 0.21104236787122874, 1.6385410773041922e-9,
          2.2326054634855798e-8, 3.0730029625651843e-9, 5.094787825563139e-9,
          0.036670498818371776, 0.06104319657703143, 1.3672919980122889e-9,
          1.647423381582632e-9, 0.49353551089576253, 9.658287207358622e-10,
          2.6914612707146334e-9, 1.5502409858734394e-9, 6.688040133571025e-9,
          5.702498460496523e-9, 1.74920185632266e-8, 2.8842309091216022e-9,
          1.6939432548395604e-9, 0.19770834935535725]
    @test isapprox(w2.weights, wt, rtol = 1e-5)

    w3 = optimise2!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    wt = [1.502698952160907e-10, 0.21104666398587518, 1.523573282203632e-10,
          1.1502118020356136e-9, 5.349304212959444e-10, 3.127959428432429e-10,
          0.03667019153803494, 0.06104496281800859, 1.2258127777323074e-10,
          1.3664808057443733e-10, 0.4935347100598141, 7.369173246087541e-11,
          1.7234948097211898e-10, 1.2626925782413405e-10, 5.002396290612599e-10,
          4.3599827752737464e-10, 1.0080007692050082e-9, 4.2016982389760224e-10,
          1.5724233733436136e-10, 0.19770346614451104]
    @test isapprox(w3.weights, wt)

    obj = Util(; l = l)
    w4 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r2 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret2 = dot(portfolio.mu, w4.weights)
    wt = [8.228671862587609e-9, 0.21291612525433196, 9.212632024402435e-9,
          6.77271672761038e-8, 5.647944186841271e-8, 9.525583076397586e-9,
          0.03942301849563698, 0.05652834541257296, 7.053510326818946e-9,
          6.90167215724061e-9, 0.49014702731909277, 3.2234218542855514e-9,
          5.408738468267058e-9, 5.859453487190136e-9, 1.1154624806837896e-8,
          2.753228972694376e-8, 6.081480482292825e-8, 2.3832551410618976e-8,
          8.947215219262381e-9, 0.20098517161658686]
    riskt = 0.028299654429795203
    rett = 0.0005145957821535152
    @test isapprox(w4.weights, wt)
    @test isapprox(r2, riskt)
    @test isapprox(ret2, rett)

    w5 = optimise2!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [1.409645126108176e-9, 0.2128167193323565, 1.4049598588911328e-9,
          2.6279990910417456e-8, 2.883095904491254e-9, 3.015914001474748e-9,
          0.039303621052848764, 0.05687420123824051, 1.1031129606041379e-9,
          1.3172803496684389e-9, 0.49017424862907016, 7.234519166886258e-10,
          1.6289081525934723e-9, 1.2832030143670635e-9, 3.453683769844097e-9,
          5.307424148337824e-9, 1.9685243970616102e-8, 3.2980253868381807e-9,
          1.40489155857031e-9, 0.20083113554865312]
    @test isapprox(w5.weights, wt, rtol = 5e-6)

    w6 = optimise2!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    wt = [8.332438351175533e-11, 0.2128135847060895, 9.145907630512437e-11,
          1.2184106241802656e-9, 4.493973710574867e-10, 1.5387170447945842e-10,
          0.03930172627407741, 0.05687333392336399, 7.36092924910324e-11,
          7.847268644655777e-11, 0.49017736599631556, 4.0892372405286e-11,
          8.781756269721712e-11, 6.49246211802165e-11, 1.9465341973994675e-10,
          2.772073232494246e-10, 8.093463891866714e-10, 1.6289435907455095e-10,
          9.568356005578712e-11, 0.20083398521818877]
    @test isapprox(w6.weights, wt)

    obj = SR(; rf = rf)
    w7 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r3 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret3 = dot(portfolio.mu, w7.weights)
    wt = [9.496500669050249e-9, 2.64615310020192e-8, 7.273118042494954e-9,
          1.4049587952157727e-8, 0.5059944415194525, 2.377003832919441e-9,
          0.17234053237874894, 1.8314836691951746e-8, 1.2375544635066102e-8,
          4.317304792347554e-8, 1.9197414728022034e-6, 2.401462046149522e-9,
          1.6115997522673463e-9, 9.360121102571334e-9, 2.354326688306667e-9,
          0.1824768715252159, 0.1391859057572847, 2.2814940892439545e-8,
          1.5125718216815985e-7, 5.757021876600399e-9]
    riskt = 0.04189063415633535
    rett = 0.0015775582433052353
    @test isapprox(w7.weights, wt)
    @test isapprox(r3, riskt)
    @test isapprox(ret3, rett)

    w8 = optimise2!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [1.1238164057580054e-8, 7.502987655712217e-8, 8.562395613769455e-9,
          1.7540789268869764e-8, 0.45025091826713454, 2.7589235102909653e-9,
          0.14690695026744388, 3.007054255105741e-8, 1.2072989519488072e-8,
          2.454812073912125e-8, 0.051268956491519475, 2.776064180360977e-9,
          1.7374807154043764e-9, 1.0436142612069122e-8, 2.4318724561358117e-9,
          0.16956901232837346, 0.182003873492148, 2.480966141363538e-8,
          5.707536077061003e-8, 8.0649967819549e-9]
    @test isapprox(w8.weights, wt, rtol = 5e-7)

    w9 = optimise2!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    wt = [1.3388885824612102e-7, 5.044363200075856e-7, 1.0333859856380586e-7,
          2.0019055857457234e-7, 0.4815462218543765, 3.2400656880734604e-8,
          0.15041378144727896, 3.452256577699966e-7, 1.6487301487356635e-7,
          3.1217276266993253e-7, 0.02247517567944038, 3.1543005887384424e-8,
          2.0866510024902028e-8, 1.1730943478258464e-7, 2.855782321665766e-8,
          0.17739887304211172, 0.16816024420968856, 3.5420797416356593e-7,
          3.2692426375081392e-6, 8.551329085377921e-8]
    @test isapprox(w9.weights, wt)

    obj = MaxRet()
    w10 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r4 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret4 = dot(portfolio.mu, w10.weights)
    wt = [3.831289441927501e-8, 4.0774049973893135e-8, 5.116921734398324e-8,
          4.653543598014566e-8, 3.0359943766315674e-6, 1.8025259997572137e-8,
          0.9999963392770433, 2.8089814289843758e-8, 4.3130715980449956e-8,
          3.078127147470436e-8, 2.7243225373031e-8, 1.906666528886154e-8,
          1.3841167393289765e-8, 2.2873601944820206e-8, 1.4288780218455534e-8,
          6.670750148901588e-8, 5.3397980334051206e-8, 2.9520022848729233e-8,
          4.472247993754918e-8, 3.624849569013439e-8]
    riskt = 0.1896251879021564
    rett = 0.0018453747093886835
    @test isapprox(w10.weights, wt)
    @test isapprox(r4, riskt)
    @test isapprox(ret4, rett)

    w11 = optimise2!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [1.8450452798034504e-10, 2.3719138287141804e-10, 3.6153220965024256e-10,
          2.9963202552729554e-10, 0.8503242801043888, 6.5628112212073e-11,
          0.14967571615535655, 7.122636503209286e-11, 2.781951307363639e-10,
          1.0170880062937885e-10, 5.654165034078128e-11, 6.098567027533659e-11,
          1.303890693422527e-10, 8.227265244624121e-12, 1.2077210891122954e-10,
          7.456041785896548e-10, 4.3436034951865134e-10, 9.022345600511273e-11,
          3.1568713917821245e-10, 1.778453418556971e-10]
    @test isapprox(w11.weights, wt)

    w12 = optimise2!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    wt = [2.7740594068572653e-9, 2.9895068386912447e-9, 3.7070853891264206e-9,
          3.337656158169103e-9, 0.8533950905998182, 1.2851548910520028e-9,
          0.14660486291889813, 2.0131312391534185e-9, 3.18366579397459e-9,
          2.1205442212122336e-9, 1.9033528476129074e-9, 1.2334871766943601e-9,
          8.945098419298842e-10, 1.578723339062802e-9, 9.052768476844037e-10,
          6.355287878151272e-9, 4.119720692605665e-9, 2.134978898877822e-9,
          3.400093873856053e-9, 2.5450484421453455e-9]
    @test isapprox(w12.weights, wt)

    # Risk upper bound
    obj = MaxRet()
    rm.settings.ub = r1
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r1 ||
          abs(calc_risk(portfolio; type = :Trad2, rm = rm) - r1) < 5e-6

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
    rm.settings.ub = r1 * (1.000001)
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r1 ||
          abs(calc_risk(portfolio; type = :Trad2, rm = rm) - r1) < 1e-7

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
          abs(dot(portfolio.mu, w16.weights) - ret4) <= 1e-10

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

#=
################
port = Portfolio(; prices = prices,
                 solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                  :params => Dict("verbose" => false,
                                                                  "max_step_fraction" => 0.75))))
asset_statistics!(port)

r = :RVaR
opt = OptimiseOpt(; rf = rf, l = l, class = :Classic, type = :Trad, rm = r, obj = :Min_Risk,
                  kelly = :None)
opt.obj = :Max_Ret
opt.kelly = :Exact
@time _w = optimise!(port, opt)
println("wt = $(_w.weights)")
println("riskt = $(calc_risk(port; type = :Trad, rm = r, rf = rf))")
println("rett = $(dot(port.mu, _w.weights))")
################
=#
