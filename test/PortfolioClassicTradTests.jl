using CSV, Clarabel, HiGHS, LinearAlgebra, OrderedCollections, PortfolioOptimiser,
      Statistics, Test, TimeSeries, Logging

Logging.disable_logging(Logging.Warn)

prices = TimeArray(CSV.File("./assets/stock_prices.csv"); timestamp = :date)

rf = 1.0329^(1 / 252) - 1
l = 2.0

@testset "Test adding skewness to another measure" begin
    portfolio = Portfolio(; prices = prices,
                          solvers = OrderedDict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                                  :params => Dict("verbose" => false))))
    asset_statistics!(portfolio)

    rm = :Skew
    opt = OptimiseOpt(; rf = rf, l = l, class = :Classic, type = :Trad, rm = rm,
                      obj = :Min_Risk, kelly = :None)

    w1 = optimise!(portfolio, opt)
    portfolio.skew_factor = 0
    w2 = optimise!(portfolio, opt)
    portfolio.skew_factor = 8
    w3 = optimise!(portfolio, opt)
    opt.rm = :SD
    portfolio.skew_factor = Inf
    w4 = optimise!(portfolio, opt)
    portfolio.skew_factor = 0
    w5 = optimise!(portfolio, opt)
    portfolio.skew_factor = 2
    w6 = optimise!(portfolio, opt)
    portfolio.skew_factor = 8
    w7 = optimise!(portfolio, opt)
    portfolio.skew_factor = Inf

    portfolio.sskew_factor = Inf
    opt.rm = :SSkew
    w8 = optimise!(portfolio, opt)
    portfolio.sskew_factor = 0
    w9 = optimise!(portfolio, opt)
    portfolio.sskew_factor = 8
    w10 = optimise!(portfolio, opt)
    opt.rm = :SD
    portfolio.sskew_factor = Inf
    w11 = optimise!(portfolio, opt)
    portfolio.sskew_factor = 0
    w12 = optimise!(portfolio, opt)
    portfolio.sskew_factor = 2
    w13 = optimise!(portfolio, opt)
    portfolio.sskew_factor = 8
    w14 = optimise!(portfolio, opt)

    w6t = [0.0026354275749322443, 0.05520947741035115, 2.88292486305246e-7,
           0.011217444462648793, 0.015540235791726633, 0.007294887210979084,
           3.899501931693162e-8, 0.1384846686508059, 5.619219962894404e-8,
           1.4264636900253708e-7, 0.2855982912592649, 8.550398887290524e-8,
           4.0185944557342566e-8, 0.11727545683980922, 0.005180482430773081,
           0.016745180622565338, 0.0077834334790627055, 0.20483183287545345,
           7.577961384734552e-8, 0.1322024537960061]
    w7t = [8.413927417714243e-7, 0.09222294098507178, 1.6168463788897735e-7,
           2.0594000236277162e-7, 0.008523442957645658, 3.007500480370547e-7,
           6.833538384822706e-8, 0.13619418248362034, 9.979458409901339e-8,
           1.5596045505028015e-7, 0.26494454649109994, 3.4315324995498946e-6,
           1.2825613036862424e-7, 0.0783181629157472, 0.02532294038010334,
           0.01907855067328539, 0.012932625739071507, 0.21592581988533274,
           1.422385714567375e-7, 0.14653125160396763]
    w12t = [0.0078093616583851146, 0.030698578999046943, 0.010528316771324561,
            0.027486806578381814, 0.012309038313357737, 0.03341430871186881,
            1.1079055085166888e-7, 0.13985416268183082, 2.0809271230580642e-7,
            6.32554715643476e-6, 0.28784123553791136, 1.268075347971748e-7,
            8.867081236591187e-8, 0.12527141708492384, 3.264070667606171e-7,
            0.015079837844627948, 1.5891383112101438e-5, 0.1931406057562605,
            2.654124109083291e-7, 0.11654298695072397]
    w13t = [1.384008546759806e-6, 0.0316494420628888, 1.4466615477601905e-6,
            0.015775935372681668, 0.010442899482149982, 0.009851951563574745,
            1.6845564712725654e-7, 0.1404230153792723, 2.93065068940981e-7,
            5.00892434748868e-7, 0.32532989744017604, 3.1063572739077716e-7,
            1.7332147477485165e-7, 0.1184225153788876, 1.25268476291211e-6,
            0.014302557449595256, 2.0736860865331673e-6, 0.2083923849842472,
            3.9292677008851197e-7, 0.12540140454845938]

    @test isapprox(w1.weights, w2.weights)
    @test isapprox(w1.weights, w3.weights)
    @test isapprox(w4.weights, w5.weights)
    @test isapprox(w6.weights, w6t)
    @test !isapprox(w6.weights, w1.weights)
    @test !isapprox(w6.weights, w4.weights)
    @test isapprox(w7.weights, w7t)
    @test !isapprox(w7.weights, w1.weights)
    @test !isapprox(w7.weights, w4.weights)
    @test !isapprox(w7.weights, w6.weights)
    @test isapprox(w8.weights, w9.weights)
    @test isapprox(w8.weights, w10.weights)
    @test isapprox(w11.weights, w12.weights)
    @test isapprox(w12.weights, w12t)
    @test !isapprox(w12.weights, w8.weights)
    @test !isapprox(w12.weights, w10.weights)
    @test isapprox(w13.weights, w13t)
    @test !isapprox(w13.weights, w8.weights)
    @test !isapprox(w13.weights, w10.weights)
    @test !isapprox(w13.weights, w12.weights)
    @test !isapprox(w6.weights, w12.weights)
    @test !isapprox(w7.weights, w13.weights)
end

@testset "$(:Classic), $(:Trad), $(:Skew)" begin
    portfolio = Portfolio(; prices = prices,
                          solvers = OrderedDict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                                  :params => Dict("verbose" => false))))
    asset_statistics!(portfolio)

    rm = :Skew
    opt = OptimiseOpt(; rf = rf, l = l, class = :Classic, type = :Trad, rm = rm,
                      obj = :Min_Risk, kelly = :None)

    w1 = optimise!(portfolio, opt)
    risk1 = calc_risk(portfolio; type = :Trad, rm = rm, rf = rf)
    opt.kelly = :Approx
    w2 = optimise!(portfolio, opt)
    opt.kelly = :Exact
    w3 = optimise!(portfolio, opt)
    opt.obj = :Utility
    opt.kelly = :None
    w4 = optimise!(portfolio, opt)
    opt.kelly = :Approx
    w5 = optimise!(portfolio, opt)
    opt.kelly = :Exact
    w6 = optimise!(portfolio, opt)
    opt.obj = :Sharpe
    opt.kelly = :None
    w7 = optimise!(portfolio, opt)
    risk7 = calc_risk(portfolio; type = :Trad, rm = rm, rf = rf)
    ret7 = dot(portfolio.mu, w7.weights)
    opt.kelly = :Approx
    w8 = optimise!(portfolio, opt)
    risk8 = calc_risk(portfolio; type = :Trad, rm = rm, rf = rf)
    ret8 = dot(portfolio.mu, w8.weights)
    opt.kelly = :Exact
    w9 = optimise!(portfolio, opt)
    risk9 = calc_risk(portfolio; type = :Trad, rm = rm, rf = rf)
    ret9 = dot(portfolio.mu, w9.weights)
    opt.obj = :Max_Ret
    opt.kelly = :None
    w10 = optimise!(portfolio, opt)
    opt.kelly = :Approx
    w11 = optimise!(portfolio, opt)
    opt.kelly = :Exact
    w12 = optimise!(portfolio, opt)
    setproperty!(portfolio, :mu_l, ret7)
    opt.obj = :Min_Risk
    opt.kelly = :None
    w13 = optimise!(portfolio, opt)
    setproperty!(portfolio, :mu_l, ret8)
    w14 = optimise!(portfolio, opt)
    setproperty!(portfolio, :mu_l, ret9)
    w15 = optimise!(portfolio, opt)
    setproperty!(portfolio, :mu_l, Inf)
    rmf = Symbol(lowercase(string(rm)) * "_u")
    setproperty!(portfolio, rmf, risk7)
    opt.obj = :Max_Ret
    opt.kelly = :None
    w16 = optimise!(portfolio, opt)
    setproperty!(portfolio, rmf, risk8)
    w17 = optimise!(portfolio, opt)
    setproperty!(portfolio, rmf, risk9)
    w18 = optimise!(portfolio, opt)
    setproperty!(portfolio, rmf, risk1)
    opt.obj = :Sharpe
    w19 = optimise!(portfolio, opt)
    setproperty!(portfolio, rmf, Inf)
    opt.sd_cone = false
    opt.kelly = :None
    opt.obj = :Min_Risk
    w20 = optimise!(portfolio, opt)
    risk20 = calc_risk(portfolio; type = :Trad, rm = rm, rf = rf)
    opt.obj = :Sharpe
    w21 = optimise!(portfolio, opt)
    risk21 = calc_risk(portfolio; type = :Trad, rm = rm, rf = rf)
    setproperty!(portfolio, rmf, 5)
    w22 = optimise!(portfolio, opt)
    risk22 = calc_risk(portfolio; type = :Trad, rm = rm, rf = rf)
    opt.obj = :Max_Ret
    setproperty!(portfolio, rmf, risk22)
    w23 = optimise!(portfolio, opt)

    w1t = [7.177018027625965e-6, 0.17212244407623012, 1.6776693950871295e-6,
           1.3963993738400464e-6, 9.1359350316075e-6, 2.3408365225251785e-6,
           1.0530241862671654e-6, 0.14156916979780365, 1.4235989499310422e-6,
           1.5011317568697727e-6, 0.07889701131958901, 0.023263445852354615,
           6.502350106065425e-6, 1.0724416935573918e-5, 0.176179498697996,
           0.042430373181700995, 9.111326427800912e-6, 0.23124986735787248,
           2.14887779917171e-6, 0.1342339971319409]

    w2t = [2.3764400256418007e-6, 0.17211106246566843, 5.233204537511742e-7,
           4.3352303461348296e-7, 3.278007213209933e-6, 7.290189341254957e-7,
           3.1888963334740083e-7, 0.1415518641160209, 4.3921105755806157e-7,
           4.6461776590596254e-7, 0.0789511670299032, 0.023284417289375792,
           2.0725966787613766e-6, 3.399092050618401e-6, 0.176168674125215,
           0.04247435910871821, 2.962379184770527e-6, 0.23121455091579368,
           6.66550703166691e-7, 0.13422624130256938]

    w3t = [1.2752392804209853e-5, 0.17225794522472923, 8.650675052742634e-7,
           7.672514249718594e-7, 1.4722959877027843e-5, 3.281130913651751e-6,
           6.655478664239528e-7, 0.14161686018169342, 8.213745220695056e-7,
           8.815668541716947e-7, 0.07883478543711876, 0.02313827588562926,
           1.5526006390681e-5, 1.6712979344748186e-5, 0.1760274073272452,
           0.04252222994427097, 1.206892326016289e-5, 0.23123708452557817,
           1.1531381957227768e-6, 0.134285193134776]

    w4t = [2.7878815351385812e-12, 3.410625057278025e-12, 5.401169527619885e-12,
           4.461178083722922e-12, 0.7646199234854653, 1.7914614290379853e-12,
           0.235380076462061, 5.374308218216221e-13, 3.722978521946705e-12,
           1.042582983337915e-12, 4.592294722688388e-13, 1.0936669918110351e-12,
           2.827048677526319e-12, 5.427031884224676e-13, 2.4151149276532487e-12,
           8.5966602678171e-12, 6.008333631830634e-12, 8.938756593773768e-13,
           4.144337259366646e-12, 2.3377888629606306e-12]

    w5t = [1.5079638098408455e-11, 2.1768062732827557e-11, 3.236339014797277e-11,
           2.6436699845860637e-11, 0.8609404210298733, 1.1352241322467653e-11,
           0.1390595785958096, 4.520291657895208e-12, 2.4745878901440058e-11,
           6.780557339661817e-12, 2.720339976466312e-12, 1.0313286385332716e-11,
           1.641962876623788e-11, 4.340436998236362e-12, 1.613794313257212e-11,
           8.762118669850096e-11, 4.2568739494228917e-11, 6.259564235793339e-12,
           2.962001019353363e-11, 1.5268788369712192e-11]

    w6t = [9.930053407384181e-10, 1.162995199569192e-9, 1.4317810771057834e-9,
           1.2812619296601035e-9, 0.863830788083523, 3.222170639319314e-10,
           0.13616919439978564, 7.220488633343297e-10, 1.2411854213495685e-9,
           7.819410072997375e-10, 6.761310949386647e-10, 3.463203746512828e-10,
           1.8763189085353233e-10, 4.979912133191178e-10, 1.9465267085070128e-10,
           2.856072064469606e-9, 1.6943016272720905e-9, 7.665148670974287e-10,
           1.3643703583356066e-9, 9.962694370304531e-10]

    w7t = [3.261419722536554e-11, 0.05423799556479565, 9.001106132301878e-12,
           1.0162505478457725e-11, 0.46437258870521564, 4.166614904373509e-12,
           0.005519348362025475, 2.3418260685542927e-11, 1.2380681162033663e-11,
           8.844274080071102e-12, 1.511583312343952e-11, 8.536574779138005e-12,
           2.8413071839632174e-12, 9.150342027748148e-12, 3.808948397142347e-12,
           0.44526561548834903, 0.03060445165082648, 2.7835574939997012e-11,
           2.555633880550943e-11, 3.535514731731058e-11]

    w8t = [3.594531663135267e-8, 0.10853533823634272, 7.842161707780163e-9,
           8.557153548572493e-9, 0.4070397304821956, 3.708466763238307e-9,
           0.0006282251273458405, 4.881365826709703e-8, 1.1575057122155202e-8,
           8.052899050280961e-9, 2.0019259932639926e-8, 7.557833820043143e-9,
           2.4576414747868304e-9, 9.470645551489532e-9, 3.3939950478241578e-9,
           0.38347566328583044, 0.100320718547955, 6.688678532807718e-8,
           3.216050026841628e-8, 5.7878955851834096e-8]

    w9t = [1.1041564118786488e-7, 0.07956873956844326, 2.7193844584687208e-8,
           3.0014356677034905e-8, 0.4459925255861044, 1.2285711382746525e-8,
           0.0076429563022691336, 1.2109735680875165e-7, 3.958459501261688e-8,
           2.722860368234154e-8, 5.784832728154244e-8, 2.2993805465658803e-8,
           8.030102762211087e-9, 2.9427760671514516e-8, 1.0896653505810534e-8,
           0.3914909146714699, 0.0753039536478863, 1.4166266570802747e-7,
           1.0843805035133106e-7, 1.6310635193074948e-7]

    w10t = [6.81331305467003e-10, 7.205814794576859e-10, 8.34831518150026e-10,
            7.792734175024859e-10, 6.5221427794168475e-9, 3.851584221109272e-10,
            0.9999999826921956, 5.543292497730712e-10, 7.327905841519223e-10,
            5.851608086559763e-10, 5.486649547830487e-10, 4.018453181148886e-10,
            3.1184313896564674e-10, 4.735544180082318e-10, 3.259069237278748e-10,
            6.839528904427972e-10, 8.017682409155311e-10, 5.766375493341149e-10,
            7.266472278040796e-10, 6.613839980521307e-10]

    w11t = [1.0849923639381655e-11, 1.5129664219201716e-11, 2.4725183728446014e-11,
            1.9934427107494966e-11, 0.8503239914634373, 8.99110014311602e-12,
            0.1496760082713066, 2.6122226938813372e-12, 1.8414042654753048e-11,
            4.7601087016406025e-12, 1.616770681751022e-12, 8.239461491675383e-12,
            1.277128077394691e-11, 3.680185496772935e-12, 1.2546692883146315e-11,
            5.429666862274197e-11, 3.052481770917738e-11, 4.044465608609783e-12,
            2.1445767918280156e-11, 1.0673540425047092e-11]

    w12t = [3.926721468533251e-10, 4.3968586501052974e-10, 5.46384280967213e-10,
            4.926529944301778e-10, 0.8533951460581576, 1.7265404013765942e-10,
            0.14660484712865535, 2.993238527901115e-10, 4.761537766943826e-10,
            3.2381368387445846e-10, 2.8762740946406504e-10, 1.791029353299881e-10,
            1.2755046888940372e-10, 2.2967420881801438e-10, 1.296256534413122e-10,
            8.90341189536715e-10, 6.118629770732845e-10, 3.151441285209569e-10,
            5.096830586409891e-10, 3.8923445889639607e-10]

    @test isapprox(w1.weights, w1t)
    @test isapprox(w2.weights, w2t)
    @test isapprox(w3.weights, w3t)
    @test isapprox(w2.weights, w3.weights, rtol = 2e-3)
    @test isapprox(w4.weights, w4t)
    @test isapprox(w5.weights, w5t)
    @test isapprox(w6.weights, w6t, rtol = 1.0e-6)
    @test isapprox(w5.weights, w6.weights, rtol = 5e-3)
    @test isapprox(w7.weights, w7t)
    @test isapprox(w8.weights, w8t)
    @test isapprox(w9.weights, w9t, rtol = 0.0005)
    @test isapprox(w8.weights, w9.weights, rtol = 1e-1)
    @test isapprox(w10.weights, w10t)
    @test isapprox(w11.weights, w11t)
    @test isapprox(w12.weights, w12t, rtol = 1.0e-5)
    @test isapprox(w11.weights, w12.weights, rtol = 6e-3)
    @test isapprox(w13.weights, w7.weights, rtol = 2e-3)
    @test isapprox(w14.weights, w8.weights, rtol = 8e-2)
    @test isapprox(w15.weights, w9.weights, rtol = 9e-2)
    @test isapprox(w16.weights, w7.weights, rtol = 5e-5)
    @test isapprox(w17.weights, w8.weights, rtol = 9e-2)
    @test isapprox(w18.weights, w9.weights, rtol = 9e-2)
    @test isapprox(w13.weights, w16.weights, rtol = 1e-3)
    @test isapprox(w14.weights, w17.weights, rtol = 6e-3)
    @test isapprox(w15.weights, w18.weights, rtol = 7e-3)
    @test isapprox(w19.weights, w1.weights, rtol = 8e-3)
    @test isapprox(w20.weights, w1.weights, rtol = 0.002)
    @test isapprox(w21.weights, w7.weights, rtol = 0.0000008)
    @test isapprox(w22.weights, w7.weights, rtol = 1.0e-6)
    @test isapprox(w23.weights, w22.weights, rtol = 2.0e-4)
end

@testset "$(:Classic), $(:Trad), $(:SSkew)" begin
    portfolio = Portfolio(; prices = prices,
                          solvers = OrderedDict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                                  :params => Dict("verbose" => false))))
    asset_statistics!(portfolio)

    rm = :SSkew
    opt = OptimiseOpt(; rf = rf, l = l, class = :Classic, type = :Trad, rm = rm,
                      obj = :Min_Risk, kelly = :None)

    w1 = optimise!(portfolio, opt)
    risk1 = calc_risk(portfolio; type = :Trad, rm = rm, rf = rf)
    opt.kelly = :Approx
    w2 = optimise!(portfolio, opt)
    opt.kelly = :Exact
    w3 = optimise!(portfolio, opt)
    opt.obj = :Utility
    opt.kelly = :None
    w4 = optimise!(portfolio, opt)
    opt.kelly = :Approx
    w5 = optimise!(portfolio, opt)
    opt.kelly = :Exact
    w6 = optimise!(portfolio, opt)
    opt.obj = :Sharpe
    opt.kelly = :None
    w7 = optimise!(portfolio, opt)
    risk7 = calc_risk(portfolio; type = :Trad, rm = rm, rf = rf)
    ret7 = dot(portfolio.mu, w7.weights)
    opt.kelly = :Approx
    w8 = optimise!(portfolio, opt)
    risk8 = calc_risk(portfolio; type = :Trad, rm = rm, rf = rf)
    ret8 = dot(portfolio.mu, w8.weights)
    opt.kelly = :Exact
    w9 = optimise!(portfolio, opt)
    risk9 = calc_risk(portfolio; type = :Trad, rm = rm, rf = rf)
    ret9 = dot(portfolio.mu, w9.weights)
    opt.obj = :Max_Ret
    opt.kelly = :None
    w10 = optimise!(portfolio, opt)
    opt.kelly = :Approx
    w11 = optimise!(portfolio, opt)
    opt.kelly = :Exact
    w12 = optimise!(portfolio, opt)
    setproperty!(portfolio, :mu_l, ret7)
    opt.obj = :Min_Risk
    opt.kelly = :None
    w13 = optimise!(portfolio, opt)
    setproperty!(portfolio, :mu_l, ret8)
    w14 = optimise!(portfolio, opt)
    setproperty!(portfolio, :mu_l, ret9)
    w15 = optimise!(portfolio, opt)
    setproperty!(portfolio, :mu_l, Inf)
    rmf = Symbol(lowercase(string(rm)) * "_u")
    setproperty!(portfolio, rmf, risk7)
    opt.obj = :Max_Ret
    opt.kelly = :None
    w16 = optimise!(portfolio, opt)
    setproperty!(portfolio, rmf, risk8)
    w17 = optimise!(portfolio, opt)
    setproperty!(portfolio, rmf, risk9)
    w18 = optimise!(portfolio, opt)
    setproperty!(portfolio, rmf, risk1)
    opt.obj = :Sharpe
    w19 = optimise!(portfolio, opt)
    setproperty!(portfolio, rmf, Inf)
    opt.sd_cone = false
    opt.kelly = :None
    opt.obj = :Min_Risk
    w20 = optimise!(portfolio, opt)
    risk20 = calc_risk(portfolio; type = :Trad, rm = rm, rf = rf)
    opt.obj = :Sharpe
    w21 = optimise!(portfolio, opt)
    risk21 = calc_risk(portfolio; type = :Trad, rm = rm, rf = rf)
    setproperty!(portfolio, rmf, 5)
    w22 = optimise!(portfolio, opt)
    risk22 = calc_risk(portfolio; type = :Trad, rm = rm, rf = rf)
    opt.obj = :Max_Ret
    setproperty!(portfolio, rmf, risk22)
    w23 = optimise!(portfolio, opt)

    w1t = [1.6044126474808694e-6, 6.387919740147824e-5, 1.414858054840168e-6,
           1.6813775168781085e-6, 2.0469710540271765e-6, 2.116339953996922e-6,
           5.874226424834042e-7, 0.12812799249943374, 1.0974501778662702e-6,
           1.7086142924218442e-6, 0.4997282356314232, 1.3273919118913505e-6,
           9.728050097433342e-7, 0.026495764704323903, 0.00015023141200918842,
           0.013458959792367066, 2.946452848774722e-6, 0.21129032992645078,
           1.5467266618233742e-6, 0.12066555601381827]

    w2t = [1.6741408439788703e-6, 5.4181619830474946e-5, 1.4772206315003974e-6,
           1.7971431052953999e-6, 2.6482468158924754e-6, 2.246087598334583e-6,
           5.917923438339016e-7, 0.12812195127012185, 1.1231990478506012e-6,
           1.7619259095515968e-6, 0.4996871899259452, 1.5623277400739484e-6,
           9.939378327706881e-7, 0.026506706873806737, 0.00016017104781535386,
           0.01342123105488912, 3.830684541612858e-6, 0.21131359319071602,
           1.588305210829121e-6, 0.12071368000525379]

    w3t = [3.489633903940562e-7, 1.7888508369111969e-6, 2.995002162203492e-7,
           4.112941353222211e-7, 9.653114721218626e-7, 5.805832585591465e-7,
           1.1934743183331442e-7, 0.12809302541637513, 2.1171680462919454e-7,
           3.4553885285982423e-7, 0.4996976419506717, 5.298775488614621e-7,
           2.4362677762407146e-7, 0.026556767350414054, 2.1774976358111058e-5,
           0.013498675647276996, 1.390959461315096e-6, 0.21132712354190702,
           2.95624993387981e-7, 0.120797459921817]

    w4t = [7.052084116651649e-11, 1.1011456306472704e-10, 2.1349200076014008e-10,
           1.5432621008187172e-10, 0.8964274785082526, 1.739778466305075e-10,
           0.10357251888949207, 3.345459809184482e-11, 1.2749228017926342e-10,
           1.3909874692879892e-11, 4.59595184139058e-11, 1.589986592408071e-10,
           2.1698240512924053e-10, 1.0784688621030566e-10, 2.1379411467057645e-10,
           4.6422703030242035e-10, 2.6128939335401695e-10, 1.9650064748911383e-11,
           1.5795421790524158e-10, 5.826476193012757e-11]

    w5t = [4.03951510116872e-11, 6.56162536985027e-11, 1.0181240275254102e-10,
           7.787365194556045e-11, 0.8832591266867846, 4.5587652947206215e-11,
           0.11674087205513205, 1.0713501752351081e-11, 7.651053985807375e-11,
           1.5882559822214765e-11, 4.861705899662827e-12, 4.362777191010949e-11,
           6.282059377511313e-11, 2.1620509050839107e-11, 6.211225286087416e-11,
           3.305784345455929e-10, 1.414471943424064e-10, 1.5975960718954705e-11,
           9.638055375788389e-11, 4.4266689812649654e-11]

    w6t = [2.8238666402252405e-9, 3.371665986257128e-9, 4.117552009808937e-9,
           3.6284380274929153e-9, 0.8863750465930669, 1.0787392894025073e-9,
           0.11362490022726426, 2.2271610067177074e-9, 3.6282149372105794e-9,
           2.3550646211777028e-9, 2.1142360390207808e-9, 1.1154190580161686e-9,
           7.299776385580322e-10, 1.5627173054426554e-9, 7.4968996330788e-10,
           9.390662252777206e-9, 4.979035544732365e-9, 2.3346607208328445e-9,
           4.03722329867881e-9, 2.93534455731154e-9]

    w7t = [1.9208950784660518e-10, 3.2176497330463307e-10, 2.4046864614545796e-10,
           1.8337883326718426e-10, 0.8086229923872494, 7.988007958328952e-11,
           0.0043547547791044595, 3.386058392712632e-10, 1.9622464070952902e-10,
           1.6355298824641334e-10, 3.1940809699051557e-10, 7.012901224488476e-11,
           4.874296829843509e-11, 1.3641899280890118e-10, 5.2647081948336176e-11,
           0.18702224906335355, 5.376052134706978e-10, 2.9754670175260314e-10,
           3.153043649354607e-10, 2.765246466590213e-10]

    w8t = [1.5640157344076053e-8, 3.582061272235252e-8, 2.0967647646182747e-8,
           1.434265151901145e-8, 0.7615402768341621, 5.8799811509887846e-9,
           0.01795493375117831, 6.82072789192539e-8, 1.7161833781871267e-8,
           1.3937346954368635e-8, 5.331381082832199e-8, 4.989596015087121e-9,
           3.52895190702238e-9, 1.1379989369077216e-8, 3.725672571225845e-9,
           0.22050422976420622, 1.7751905872404807e-7, 4.198766806450605e-8,
           4.217111232106812e-8, 2.907708355516215e-8]

    w9t = [5.553592439719559e-8, 1.1663023650724744e-7, 7.633484799626114e-8,
           5.326938535959939e-8, 0.7729653491506971, 2.083948647969848e-8,
           0.020610259502484927, 1.520542775416144e-7, 6.471991580131279e-8,
           4.987187222564725e-8, 1.2708076393696777e-7, 1.8038849954898925e-8,
           1.2669331427630937e-8, 3.910989923638194e-8, 1.3355344444324209e-8,
           0.2064228749643929, 3.6443089845109477e-7, 1.1604485798825767e-7,
           1.4158630547059004e-7, 9.481022777802042e-8]

    w10t = [4.3422955344620045e-9, 4.482608372372868e-9, 5.2151136335546805e-9,
            4.997973799097022e-9, 1.479369787463968e-7, 2.2326679314415314e-9,
            0.999999785026607, 3.327627412942966e-9, 4.59970993047624e-9,
            3.582706666614823e-9, 3.263665683434885e-9, 2.3305636210161792e-9,
            1.7153315191007083e-9, 2.77851132735056e-9, 1.7865323395456848e-9,
            4.945270403094947e-9, 5.191706124856787e-9, 3.4811228279287047e-9,
            4.650003219981197e-9, 4.113003750877124e-9]

    w11t = [5.619586642787703e-12, 8.276729930516474e-12, 1.4303142954340837e-11,
            1.1318044807100396e-11, 0.8503236954639315, 7.1381171385102224e-12,
            0.14967630438138319, 4.2876945553380653e-13, 1.0193743602633687e-11,
            1.732551494667518e-12, 4.5941464265574075e-14, 6.44661322933117e-12,
            9.08817518586835e-12, 3.5643431740902176e-12, 8.77780354909866e-12,
            3.082287714886444e-11, 1.7759280864727746e-11, 1.3797700206848313e-12,
            1.2160188596177151e-11, 5.629568200254058e-12]

    w12t = [2.4708329877025105e-9, 2.825658691624536e-9, 3.4617711868402514e-9,
            3.1646001687314595e-9, 0.8533932860581867, 1.1538651284884013e-9,
            0.14660666955105, 1.965825799273989e-9, 3.1130503740613393e-9,
            2.144622849616904e-9, 1.9023680945779106e-9, 1.2012465199275993e-9,
            8.619052713077644e-10, 1.5231616903722065e-9, 8.799719630980316e-10,
            5.779499550493376e-9, 3.964911074788187e-9, 2.064194312614277e-9,
            3.359371733545459e-9, 2.5539059160475118e-9]

    @test isapprox(w1.weights, w1t)
    @test isapprox(w2.weights, w2t)
    @test isapprox(w3.weights, w3t)
    @test isapprox(w2.weights, w3.weights, rtol = 5e-4)
    @test isapprox(w4.weights, w4t)
    @test isapprox(w5.weights, w5t)
    @test isapprox(w6.weights, w6t, rtol = 1.0e-5)
    @test isapprox(w5.weights, w6.weights, rtol = 5e-3)
    @test isapprox(w7.weights, w7t)
    @test isapprox(w8.weights, w8t)
    @test isapprox(w9.weights, w9t, rtol = 0.0005)
    @test isapprox(w8.weights, w9.weights, rtol = 3e-2)
    @test isapprox(w10.weights, w10t)
    @test isapprox(w11.weights, w11t)
    @test isapprox(w12.weights, w12t, rtol = 1.0e-5)
    @test isapprox(w11.weights, w12.weights, rtol = 6e-3)
    @test isapprox(w13.weights, w7.weights, rtol = 7e-4)
    @test isapprox(w14.weights, w8.weights, rtol = 4e-2)
    @test isapprox(w15.weights, w9.weights, rtol = 4e-2)
    @test isapprox(w16.weights, w7.weights, rtol = 4e-5)
    @test isapprox(w17.weights, w8.weights, rtol = 4e-2)
    @test isapprox(w18.weights, w9.weights, rtol = 4e-2)
    @test isapprox(w13.weights, w16.weights, rtol = 8e-4)
    @test isapprox(w14.weights, w17.weights, rtol = 4e-3)
    @test isapprox(w15.weights, w18.weights, rtol = 4e-3)
    @test isapprox(w19.weights, w1.weights, rtol = 6e-3)
    @test isapprox(w20.weights, w1.weights, rtol = 0.005)
    @test isapprox(w21.weights, w7.weights, rtol = 0.0000009)
    @test isapprox(w22.weights, w7.weights, rtol = 9.0e-7)
    @test isapprox(w23.weights, w22.weights, rtol = 2.0e-4)
end

@testset "$(:Classic), $(:Trad), $(:DVar)" begin
    portfolio = Portfolio(; prices = prices[(end - 50):end],
                          solvers = OrderedDict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                                  :params => Dict("verbose" => false,
                                                                                  "max_step_fraction" => 0.75))))
    asset_statistics!(portfolio)

    rm = :DVar
    opt = OptimiseOpt(; rf = rf, l = l, class = :Classic, type = :Trad, rm = rm,
                      obj = :Min_Risk, kelly = :None)
    opt.obj = :Min_Risk
    w1 = optimise!(portfolio, opt)
    risk1 = calc_risk(portfolio; type = :Trad, rm = rm, rf = rf)
    opt.kelly = :Approx
    w2 = optimise!(portfolio, opt)
    opt.kelly = :Exact
    w3 = optimise!(portfolio, opt)
    opt.obj = :Utility
    opt.kelly = :None
    w4 = optimise!(portfolio, opt)
    opt.kelly = :Approx
    w5 = optimise!(portfolio, opt)
    opt.kelly = :Exact
    w6 = optimise!(portfolio, opt)
    opt.obj = :Sharpe
    opt.kelly = :None
    w7 = optimise!(portfolio, opt)
    risk7 = calc_risk(portfolio; type = :Trad, rm = rm, rf = rf)
    ret7 = dot(portfolio.mu, w7.weights)
    opt.kelly = :Approx
    w8 = optimise!(portfolio, opt)
    risk8 = calc_risk(portfolio; type = :Trad, rm = rm, rf = rf)
    ret8 = dot(portfolio.mu, w8.weights)
    opt.kelly = :Exact
    w9 = optimise!(portfolio, opt)
    risk9 = calc_risk(portfolio; type = :Trad, rm = rm, rf = rf)
    ret9 = dot(portfolio.mu, w9.weights)
    opt.obj = :Max_Ret
    opt.kelly = :None
    w10 = optimise!(portfolio, opt)
    opt.kelly = :Approx
    w11 = optimise!(portfolio, opt)
    opt.kelly = :Exact
    w12 = optimise!(portfolio, opt)
    setproperty!(portfolio, :mu_l, ret7)
    opt.obj = :Min_Risk
    opt.kelly = :None
    w13 = optimise!(portfolio, opt)
    setproperty!(portfolio, :mu_l, ret8)
    w14 = optimise!(portfolio, opt)
    setproperty!(portfolio, :mu_l, ret9)
    w15 = optimise!(portfolio, opt)
    setproperty!(portfolio, :mu_l, Inf)
    rmf = Symbol(lowercase(string(rm)) * "_u")
    setproperty!(portfolio, rmf, risk7)
    opt.obj = :Max_Ret
    opt.kelly = :None
    w16 = optimise!(portfolio, opt)
    setproperty!(portfolio, rmf, risk8)
    w17 = optimise!(portfolio, opt)
    setproperty!(portfolio, rmf, risk9)
    w18 = optimise!(portfolio, opt)
    setproperty!(portfolio, rmf, 1)
    opt.obj = :Sharpe
    w19 = optimise!(portfolio, opt)
    setproperty!(portfolio, rmf, Inf)

    w1t = [7.181638681175778e-9, 2.183033586036383e-8, 1.5218044065340246e-8,
           4.2604530005585394e-8, 1.294056562795285e-7, 2.1086905490413199e-7,
           4.365828792120837e-8, 0.03488916642364215, 1.0517213388849292e-8,
           1.326745253044503e-8, 0.3163054062669858, 5.210873942304675e-9,
           2.1795660933612152e-9, 0.070986242138395, 6.637018341865444e-9,
           0.0010206495217824942, 2.78297757102955e-8, 1.4641218140925397e-7,
           1.680728155414907e-8, 0.5767978360202839]

    w2t = [5.087795911623077e-9, 1.4563464110653952e-8, 1.0581227629131406e-8,
           2.8967790479164322e-8, 8.099525135277588e-8, 1.2684621730557623e-7,
           2.9321625771381956e-8, 0.03489495209701263, 7.287795036102504e-9,
           9.077216038198508e-9, 0.31630029659380093, 3.7623478183642865e-9,
           1.6303205179387497e-9, 0.07099422169169607, 4.8004495913202385e-9,
           0.0010135983452668872, 1.8533015465654853e-8, 8.91788117310633e-8,
           1.1432974748043883e-8, 0.57679648920592]

    w3t = [6.723491246438328e-9, 1.4697270479092619e-8, 1.1713176307264514e-8,
           2.483387400182778e-8, 4.861848872483968e-8, 7.910053289706015e-8,
           2.6665269282945067e-8, 0.034901114616923246, 8.91824199472646e-9,
           1.0768082602311042e-8, 0.31629556740006426, 4.523920767134701e-9,
           1.7677913859901154e-9, 0.07100254912218402, 5.530855118185789e-9,
           0.0010056016839992874, 1.755773644696841e-8, 5.3963040359234354e-8,
           1.2526841605902122e-8, 0.576794839268216]

    w4t = [1.1100397929454455e-9, 3.4387184900318225e-9, 7.394057306671417e-10,
           7.589139950533886e-10, 1.5576792729297947e-9, 6.034252300244166e-10,
           1.1553324968794632e-9, 7.507168975683432e-10, 9.063942740124959e-10,
           7.013590770530115e-10, 2.1159033714573355e-9, 0.08806276986342787,
           0.0837832784661892, 4.2507266585085384e-10, 1.016449775809084e-9,
           7.534501518741969e-10, 1.9034758848819247e-8, 5.336603795000105e-10,
           8.315835510446377e-10, 0.828153915237519]

    w5t = [9.513535701703655e-10, 1.6476842050611988e-9, 4.328346076015453e-10,
           5.056791094826032e-10, 7.115178442420491e-10, 4.1403433426886756e-10,
           7.814855946005831e-10, 5.557313944859753e-10, 6.253126151187478e-10,
           4.4406966079766157e-10, 1.152443895915974e-9, 0.07704055975254906,
           0.07386876060943966, 2.0475453363933794e-10, 5.892633471077494e-10,
           3.9606490150140533e-10, 1.149315400404046e-8, 2.8970440474326456e-10,
           4.668005705258435e-10, 0.8490906579761227]

    w6t = [5.70768280991311e-10, 4.343028200618946e-9, 8.47226406683893e-10,
           6.202018223537769e-10, 2.325373002834198e-9, 4.845114742059602e-10,
           3.6864663019857525e-10, 4.358301126567282e-10, 9.197474719844687e-10,
           8.49925473531239e-10, 2.7707702545826414e-9, 0.07736676337609043,
           0.07393381109381487, 9.269925743738179e-10, 1.2646110635926894e-9,
           1.2203751648629535e-9, 1.760923131213385e-8, 9.829402826502637e-10,
           1.2518563109824393e-9, 0.8486993877380589]

    w7t = [1.978669956865214e-12, 9.263395380076963e-12, 2.6636503923018716e-12,
           2.386050089830185e-12, 9.429454382736214e-12, 1.650875679061161e-12,
           1.2502489426661446e-12, 7.919453109756513e-13, 2.9743935979747758e-12,
           2.7648934882256243e-12, 3.744297679221852e-12, 0.4841047817596029,
           0.49717956285476433, 2.5711566888209673e-12, 4.105867376644003e-12,
           2.939943592845087e-12, 1.0489956911810742e-11, 2.837915154773032e-12,
           3.4426838661236216e-12, 0.018715655320347368]

    w8t = [1.0527174500968367e-9, 7.315420256170622e-9, 1.4068990722754566e-9,
           1.167208734220835e-9, 6.133012287704447e-9, 7.739487869618192e-10,
           6.090331304746666e-10, 6.964600822226045e-10, 1.6103550673636972e-9,
           1.4512217948003605e-9, 2.4009885938451686e-9, 0.24097700574026484,
           0.25881732392928103, 1.318160254528228e-9, 2.5111105288906926e-9,
           1.658231619467009e-9, 1.1651872752832763e-8, 1.5200213303384298e-9,
           2.0483736166365492e-9, 0.5002056250054189]

    w9t = [3.0038272203728177e-9, 2.2925634424496407e-8, 4.07970171533653e-9,
           3.38429316420835e-9, 2.0061018700747915e-8, 2.1760104607088733e-9,
           1.6924255086762817e-9, 1.949572516136007e-9, 4.6687666008379666e-9,
           4.20250902795816e-9, 7.078739234163238e-9, 0.27649474451422085,
           0.29120961177271076, 3.79939308595352e-9, 7.441258863449038e-9,
           4.817879461494422e-9, 3.46406720346918e-8, 4.425808975739151e-9,
           6.011695148777275e-9, 0.43229550735386224]

    w10t = [1.595081606192407e-8, 2.9382334431414644e-8, 1.7658467922714496e-8,
            1.6582896735815844e-8, 2.7054675489076104e-8, 1.4110650817598288e-8,
            1.2211197645107672e-8, 1.3699678443896811e-8, 1.9274099938707518e-8,
            1.861445082317701e-8, 2.1964968347874875e-8, 9.698462231820156e-8,
            0.999999532231261, 1.7731042737188638e-8, 2.6080622753652606e-8,
            2.0173338470245485e-8, 2.7700515356613562e-8, 1.9503708210601273e-8,
            2.1275426991479796e-8, 3.1815225407544205e-8]

    w11t = [3.6284539135707244e-10, 1.5010073244676765e-9, 5.156882524413499e-10,
            4.264998516185897e-10, 1.3395647008117877e-9, 1.0835244933809214e-10,
            1.1804438923295997e-10, 1.416739433084119e-10, 6.508592529702165e-10,
            5.805156454024769e-10, 8.742627947410693e-10, 8.485198818271674e-9,
            0.9999999780400666, 3.6460737171271087e-10, 1.099700685469474e-9,
            7.024995339703052e-10, 1.477660769170928e-9, 6.118389402353163e-10,
            8.236442718585597e-10, 1.77546915593609e-9]

    w12t = [6.656263910656658e-9, 1.418855067294687e-8, 7.399905926320237e-9,
            6.8643886948896626e-9, 1.3255951140357994e-8, 5.275356601279071e-9,
            4.53900528865935e-9, 4.968400565900951e-9, 8.411533216795753e-9,
            7.838930818078597e-9, 9.303159310851257e-9, 6.95139472785675e-8,
            0.9999997676151651, 7.053486428756268e-9, 1.1475834772577401e-8,
            8.42106083234053e-9, 1.402405425782804e-8, 7.968238485908183e-9,
            9.351503373662934e-9, 1.587526336615388e-8]

    @test isapprox(w1.weights, w1t)
    @test isapprox(w2.weights, w2t)
    @test isapprox(w3.weights, w3t)
    @test isapprox(w2.weights, w3.weights, rtol = 3e-5)
    @test isapprox(w4.weights, w4t)
    @test isapprox(w5.weights, w5t)
    @test isapprox(w6.weights, w6t, rtol = 5.0e-5)
    @test isapprox(w5.weights, w6.weights, rtol = 7e-4)
    @test isapprox(w7.weights, w7t)
    @test isapprox(w8.weights, w8t)
    @test isapprox(w9.weights, w9t, rtol = 5.0e-5)
    @test isapprox(w8.weights, w9.weights, rtol = 2e-1)
    @test isapprox(w10.weights, w10t)
    @test isapprox(w11.weights, w11t)
    @test isapprox(w12.weights, w12t, rtol = 5.0e-5)
    @test isapprox(w11.weights, w12.weights, rtol = 3e-7)
    @test isapprox(w13.weights, w7.weights, rtol = 2e-7)
    @test isapprox(w14.weights, w8.weights, rtol = 2e-2)
    @test isapprox(w15.weights, w9.weights, rtol = 2e-2)
    @test isapprox(w16.weights, w7.weights, rtol = 3e-5)
    @test isapprox(w17.weights, w8.weights, rtol = 2e-2)
    @test isapprox(w18.weights, w9.weights, rtol = 2e-2)
    @test isapprox(w13.weights, w16.weights, rtol = 3e-5)
    @test isapprox(w14.weights, w17.weights, rtol = 6e-5)
    @test isapprox(w15.weights, w18.weights, rtol = 6e-5)
    @test isapprox(w19.weights, w7.weights, rtol = 9e-6)
end
@testset "Mu Lower bound Sharpe" begin
    portfolio = Portfolio(; prices = prices,
                          solvers = OrderedDict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                                  :params => Dict("verbose" => false,
                                                                                  "max_step_fraction" => 0.75))))
    asset_statistics!(portfolio)

    opt = OptimiseOpt(; rf = rf, l = l, class = :Classic, type = :Trad, rm = :SD,
                      obj = :Sharpe, kelly = :None)

    w1 = optimise!(portfolio, opt)
    r1 = dot(portfolio.mu, w1.weights)
    opt.obj = :Max_Ret
    w2 = optimise!(portfolio, opt)
    r2 = dot(portfolio.mu, w2.weights)

    r3 = (r1 + r2) / 2
    portfolio.mu_l = r3
    opt.obj = :Sharpe
    w3 = optimise!(portfolio, opt)
    r4 = dot(portfolio.mu, w2.weights)

    portfolio.mu_l = Inf
    T = size(portfolio.returns, 1)
    opt.kelly = :Exact
    w5 = optimise!(portfolio, opt)
    r5 = sum(log.(1 .+ portfolio.returns * w5.weights)) / T

    opt.obj = :Max_Ret
    w6 = optimise!(portfolio, opt)
    r6 = sum(log.(1 .+ portfolio.returns * w6.weights)) / T

    r7 = (r5 + r6) / 2
    portfolio.mu_l = r7
    opt.obj = :Sharpe
    w7 = optimise!(portfolio, opt)
    r8 = sum(log.(1 .+ portfolio.returns * w7.weights)) / T

    @test r4 >= r3
    @test r4 <= r2
    @test r4 >= r1

    @test r8 >= r5
    @test r8 <= r6
    @test r8 >= r7
end

@testset "MAD and LPM targets" begin
    portfolio = Portfolio(; prices = prices,
                          solvers = OrderedDict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                                  :params => Dict("verbose" => false,
                                                                                  "max_step_fraction" => 0.75))))
    asset_statistics!(portfolio)
    portfolio.msv_target = rf

    opt = OptimiseOpt(; rf = rf, l = l, class = :Classic, type = :Trad, rm = :MAD,
                      obj = :Sharpe, kelly = :None)
    w1 = optimise!(portfolio, opt)
    opt.rm = :FLPM
    w2 = optimise!(portfolio, opt)

    opt.rm = :SSD
    w3 = optimise!(portfolio, opt)
    opt.rm = :SLPM
    w4 = optimise!(portfolio, opt)

    portfolio.msv_target = 0
    portfolio.lpm_target = 0

    opt.rm = :MAD
    w5 = optimise!(portfolio, opt)
    opt.rm = :FLPM
    w6 = optimise!(portfolio, opt)

    opt.rm = :SSD
    w7 = optimise!(portfolio, opt)
    opt.rm = :SLPM
    w8 = optimise!(portfolio, opt)

    @test isapprox(w1.weights, w2.weights)
    @test isapprox(w3.weights, w4.weights)
    @test isapprox(w5.weights, w6.weights)
    @test isapprox(w7.weights, w8.weights)
end

@testset "$(:Classic), $(:Trad), $(:SD)" begin
    portfolio = Portfolio(; prices = prices,
                          solvers = OrderedDict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                                  :params => Dict("verbose" => false,
                                                                                  "max_step_fraction" => 0.75))))
    asset_statistics!(portfolio)

    rm = :SD
    opt = OptimiseOpt(; rf = rf, l = l, class = :Classic, type = :Trad, rm = rm,
                      obj = :Min_Risk, kelly = :None)

    w1 = optimise!(portfolio, opt)
    risk1 = calc_risk(portfolio; type = :Trad, rm = rm, rf = rf)
    opt.kelly = :Approx
    w2 = optimise!(portfolio, opt)
    opt.kelly = :Exact
    w3 = optimise!(portfolio, opt)
    opt.obj = :Utility
    opt.kelly = :None
    w4 = optimise!(portfolio, opt)
    opt.kelly = :Approx
    w5 = optimise!(portfolio, opt)
    opt.kelly = :Exact
    w6 = optimise!(portfolio, opt)
    opt.obj = :Sharpe
    opt.kelly = :None
    w7 = optimise!(portfolio, opt)
    risk7 = calc_risk(portfolio; type = :Trad, rm = rm, rf = rf)
    ret7 = dot(portfolio.mu, w7.weights)
    opt.kelly = :Approx
    w8 = optimise!(portfolio, opt)
    risk8 = calc_risk(portfolio; type = :Trad, rm = rm, rf = rf)
    ret8 = dot(portfolio.mu, w8.weights)
    opt.kelly = :Exact
    w9 = optimise!(portfolio, opt)
    risk9 = calc_risk(portfolio; type = :Trad, rm = rm, rf = rf)
    ret9 = dot(portfolio.mu, w9.weights)
    opt.obj = :Max_Ret
    opt.kelly = :None
    w10 = optimise!(portfolio, opt)
    opt.kelly = :Approx
    w11 = optimise!(portfolio, opt)
    opt.kelly = :Exact
    w12 = optimise!(portfolio, opt)
    setproperty!(portfolio, :mu_l, ret7)
    opt.obj = :Min_Risk
    opt.kelly = :None
    w13 = optimise!(portfolio, opt)
    setproperty!(portfolio, :mu_l, ret8)
    w14 = optimise!(portfolio, opt)
    setproperty!(portfolio, :mu_l, ret9)
    w15 = optimise!(portfolio, opt)
    setproperty!(portfolio, :mu_l, Inf)
    rmf = Symbol(lowercase(string(rm)) * "_u")
    setproperty!(portfolio, rmf, risk7)
    opt.obj = :Max_Ret
    opt.kelly = :None
    w16 = optimise!(portfolio, opt)
    setproperty!(portfolio, rmf, risk8)
    w17 = optimise!(portfolio, opt)
    setproperty!(portfolio, rmf, risk9)
    w18 = optimise!(portfolio, opt)
    setproperty!(portfolio, rmf, risk1)
    opt.obj = :Sharpe
    w19 = optimise!(portfolio, opt)
    setproperty!(portfolio, rmf, Inf)

    w1t = [0.007911813464310113, 0.03068510213747275, 0.010505366137939913,
           0.02748375285330511, 0.012276170276067139, 0.03340727036207522,
           3.968191684113028e-7, 0.1398469015680947, 6.62690268489184e-7,
           1.412505225860106e-5, 0.2878192785440307, 4.2200065419448415e-7,
           3.296864762413375e-7, 0.12527560333674628, 1.836077875288932e-6,
           0.015083599354027831, 2.2788913817547395e-5, 0.19311714086150886,
           8.333796539106878e-7, 0.11654660648424862]

    w2t = [0.007911813464310113, 0.03068510213747275, 0.010505366137939913,
           0.02748375285330511, 0.012276170276067139, 0.03340727036207522,
           3.968191684113028e-7, 0.1398469015680947, 6.62690268489184e-7,
           1.412505225860106e-5, 0.2878192785440307, 4.2200065419448415e-7,
           3.296864762413375e-7, 0.12527560333674628, 1.836077875288932e-6,
           0.015083599354027831, 2.2788913817547395e-5, 0.19311714086150886,
           8.333796539106878e-7, 0.11654660648424862]

    w3t = [0.00791011483676178, 0.03068925469836609, 0.010506700416679362,
           0.02748642042968038, 0.012277366650756448, 0.03341099795660443,
           8.158411219169124e-8, 0.1398481565836097, 1.4218817713969255e-7,
           2.2275104721893052e-6, 0.28782177097087014, 9.528243270444993e-8,
           7.326778892658585e-8, 0.12528224568926083, 3.716016297026811e-7,
           0.015085206659951809, 3.449985338978694e-6, 0.19312271585991778,
           1.7981604361408237e-7, 0.1165524280115459]

    w4t = [1.8806825315287365e-9, 3.862038288374664e-9, 5.3002004387495145e-9,
           4.036002575991236e-9, 0.7741908142503021, 3.8445966380599916e-10,
           0.10998661007165159, 1.852701338887806e-9, 5.530542601292167e-9,
           1.7145172711134676e-9, 1.5657439073700846e-9, 4.404614975595981e-10,
           8.278998697027929e-10, 3.393489762947003e-10, 7.56536345977105e-10,
           0.11582251334241239, 1.8378733654274228e-8, 2.095243020357391e-9,
           1.0434852414339229e-8, 2.9356695212202287e-9]

    w5t = [4.000173954884603e-9, 9.797941061152537e-9, 1.3496791594413377e-8,
           9.407090882112671e-9, 0.7186309516095013, 4.5372657047985664e-10,
           0.09861483106122985, 5.019344235849704e-9, 1.511043562253717e-8,
           4.033307336574645e-9, 4.239178636456144e-9, 6.689136834623081e-10,
           1.2974747157266286e-9, 1.0370396145841418e-9, 1.2435269744525334e-9,
           0.15517374173124648, 0.02758032279672288, 5.654612702089748e-9,
           7.008784891048097e-8, 7.253892996532703e-9]

    w6t = [6.839076880744838e-9, 1.232816753920913e-8, 1.5586533315004005e-8,
           1.1729481303080301e-8, 0.7207181011796364, 2.2663390521198906e-9,
           0.0983552572118593, 7.511729485889375e-9, 1.685175480021761e-8,
           6.274697474333583e-9, 6.564355053847367e-9, 1.6967378612547427e-9,
           1.0682360568465446e-9, 3.55682404547078e-9, 1.029413856489014e-9,
           0.15505254746798722, 0.025873920718209042, 8.2315065963093e-9,
           6.252067885998374e-8, 9.36677584782878e-9]

    w7t = [3.0475825682563666e-9, 8.756567654479468e-9, 1.183597736460028e-8,
           7.037545660104337e-9, 0.5180580294593814, 8.497531491525025e-10,
           0.06365095465210915, 7.0711477926675866e-9, 6.793919774975346e-9,
           3.406672135170839e-9, 5.600435165445702e-9, 7.43907631470485e-10,
           5.560792321598317e-10, 1.6569788566556483e-9, 5.015125409988519e-10,
           0.14326778797036815, 0.19649629723668247, 6.666401729130281e-9,
           0.07852685909966249, 7.057315174215836e-9]

    w8t = [1.8936644903543066e-7, 7.76000979748512e-7, 1.0308400105528235e-6,
           4.859599976321728e-7, 0.452499304930526, 5.6605105096373026e-8,
           0.05199219524185312, 2.624818374913499e-6, 3.0810565235958007e-7,
           2.2165976127282862e-7, 9.982287659814127e-7, 4.316509358322131e-8,
           3.194570649487603e-8, 1.0930559915177949e-7, 3.121923591207061e-8,
           0.13648884750262485, 0.23504401124035085, 9.923220711882843e-7,
           0.12396703810935396, 7.034324884005354e-7]
    w9t = [2.387216613685937e-8, 7.828771156369864e-8, 1.0229870690822666e-7,
           5.6299119725801325e-8, 0.4894148286726404, 7.444140272727556e-9,
           0.05833657453105973, 8.106665070266756e-8, 5.276461180379594e-8,
           2.7180318459157044e-8, 5.86669235243501e-8, 6.0358107251716945e-9,
           4.40402551166592e-9, 1.3582073904098907e-8, 4.27712094853996e-9,
           0.14032584660655578, 0.2133774892572834, 6.897142501094063e-8,
           0.09854461196095957, 6.382069575305657e-8]

    w10t = [2.242587206880573e-8, 2.3869345719665024e-8, 3.0249972077364615e-8,
            2.7441036414624337e-8, 2.674016473443787e-6, 1.111794542964317e-8,
            0.999996954927899, 1.6603745204232977e-8, 2.5223224543701134e-8,
            1.8088400369515617e-8, 1.6172380723211374e-8, 1.16166386268502e-8,
            8.681387038232673e-9, 1.3771962838160079e-8, 8.992921449936008e-9,
            4.039243448317557e-8, 3.1600448893350835e-8, 1.7426103715815354e-8,
            2.6166436902355393e-8, 2.121537093958793e-8]

    w11t = [1.3172620135686553e-10, 1.9064388092607144e-10, 3.2584225087336927e-10,
            2.59317046047275e-10, 0.8503241090172801, 1.3130387717999635e-10,
            0.1496758875427982, 1.4044287784039595e-11, 2.309729885121847e-10,
            4.123955829597335e-11, 3.434408199187934e-12, 1.1134275889489673e-10,
            2.0702691150761263e-10, 6.564442934076178e-11, 1.225576898169225e-10,
            7.582849306597439e-10, 4.0835176373099187e-10, 3.60117150622103e-11,
            2.749916443740321e-10, 1.271854038475352e-10]

    w12t = [2.7645873503234275e-9, 2.973259803593348e-9, 3.6878114973229714e-9,
            3.321370576117124e-9, 0.8533950505923042, 1.2742245971629924e-9,
            0.14660490317164734, 1.9962609910135675e-9, 3.1697472598073793e-9,
            2.1019702300844745e-9, 1.8844964419676663e-9, 1.2186378467195185e-9,
            8.840525159658307e-10, 1.5657402848907941e-9, 8.942385792282134e-10,
            6.363468927866918e-9, 4.103435435690254e-9, 2.1221723839725065e-9,
            3.387040113458199e-9, 2.5235336172152343e-9]

    @test isapprox(w1.weights, w1t)
    @test isapprox(w2.weights, w2t)
    @test isapprox(w3.weights, w3t)
    @test isapprox(w2.weights, w3.weights, rtol = 1e-4)
    @test isapprox(w4.weights, w4t)
    @test isapprox(w5.weights, w5t)
    @test isapprox(w6.weights, w6t, rtol = 1e-6)
    @test isapprox(w5.weights, w6.weights, rtol = 1e-2)
    @test isapprox(w7.weights, w7t)
    @test isapprox(w8.weights, w8t)
    @test isapprox(w9.weights, w9t, rtol = 1e-5)
    @test isapprox(w8.weights, w9.weights, rtol = 1e-1)
    @test isapprox(w10.weights, w10t)
    @test isapprox(w11.weights, w11t)
    @test isapprox(w12.weights, w12t, rtol = 1e-6)
    @test isapprox(w11.weights, w12.weights, rtol = 1e-2)
    @test isapprox(w13.weights, w7.weights, rtol = 1e-5)
    @test isapprox(w14.weights, w8.weights, rtol = 1e-5)
    @test isapprox(w15.weights, w9.weights, rtol = 1e-3)
    @test isapprox(w16.weights, w7.weights, rtol = 1e-5)
    @test isapprox(w17.weights, w8.weights, rtol = 1e-4)
    @test isapprox(w18.weights, w9.weights, rtol = 1e-3)
    @test isapprox(w13.weights, w16.weights, rtol = 1e-4)
    @test isapprox(w14.weights, w17.weights, rtol = 1e-5)
    @test isapprox(w15.weights, w18.weights, rtol = 1e-5)
    @test isapprox(w19.weights, w1.weights, rtol = 1e-2)
end

@testset "$(:Classic), $(:Trad), $(:MAD)" begin
    portfolio = Portfolio(; prices = prices,
                          solvers = OrderedDict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                                  :params => Dict("verbose" => false,
                                                                                  "max_step_fraction" => 0.75))))
    asset_statistics!(portfolio)

    rm = :MAD
    opt = OptimiseOpt(; rf = rf, l = l, class = :Classic, type = :Trad, rm = rm,
                      obj = :Min_Risk, kelly = :None)

    w1 = optimise!(portfolio, opt)
    risk1 = calc_risk(portfolio; type = :Trad, rm = rm, rf = rf)
    opt.kelly = :Approx
    w2 = optimise!(portfolio, opt)
    opt.kelly = :Exact
    w3 = optimise!(portfolio, opt)
    opt.obj = :Utility
    opt.kelly = :None
    w4 = optimise!(portfolio, opt)
    opt.kelly = :Approx
    w5 = optimise!(portfolio, opt)
    opt.kelly = :Exact
    w6 = optimise!(portfolio, opt)
    opt.obj = :Sharpe
    opt.kelly = :None
    w7 = optimise!(portfolio, opt)
    risk7 = calc_risk(portfolio; type = :Trad, rm = rm, rf = rf)
    ret7 = dot(portfolio.mu, w7.weights)
    opt.kelly = :Approx
    w8 = optimise!(portfolio, opt)
    risk8 = calc_risk(portfolio; type = :Trad, rm = rm, rf = rf)
    ret8 = dot(portfolio.mu, w8.weights)
    opt.kelly = :Exact
    w9 = optimise!(portfolio, opt)
    risk9 = calc_risk(portfolio; type = :Trad, rm = rm, rf = rf)
    ret9 = dot(portfolio.mu, w9.weights)
    opt.obj = :Max_Ret
    opt.kelly = :None
    w10 = optimise!(portfolio, opt)
    opt.kelly = :Approx
    w11 = optimise!(portfolio, opt)
    opt.kelly = :Exact
    w12 = optimise!(portfolio, opt)
    setproperty!(portfolio, :mu_l, ret7)
    opt.obj = :Min_Risk
    opt.kelly = :None
    w13 = optimise!(portfolio, opt)
    setproperty!(portfolio, :mu_l, ret8)
    w14 = optimise!(portfolio, opt)
    setproperty!(portfolio, :mu_l, ret9)
    w15 = optimise!(portfolio, opt)
    setproperty!(portfolio, :mu_l, Inf)
    rmf = Symbol(lowercase(string(rm)) * "_u")
    setproperty!(portfolio, rmf, risk7)
    opt.obj = :Max_Ret
    opt.kelly = :None
    w16 = optimise!(portfolio, opt)
    setproperty!(portfolio, rmf, risk8)
    w17 = optimise!(portfolio, opt)
    setproperty!(portfolio, rmf, risk9)
    w18 = optimise!(portfolio, opt)
    setproperty!(portfolio, rmf, risk1)
    opt.obj = :Sharpe
    w19 = optimise!(portfolio, opt)
    setproperty!(portfolio, rmf, Inf)

    w1t = [0.014061763213837397, 0.04237575385355662, 0.01686367745456826,
           0.0020208625490112752, 0.017683591293599046, 0.054224216082713245,
           2.3867129053495714e-9, 0.15821633781770636, 3.663198950992048e-9,
           1.2649839996351797e-8, 0.23689716002716457, 7.93586467840601e-10,
           6.049503766739469e-10, 0.12783198610229046, 0.0003510941636230447,
           0.0009122213826153329, 0.04394979334424017, 0.18272445401585058,
           6.313784959827286e-9, 0.10188706228715007]

    w2t = [0.014061752856837131, 0.04237578891874663, 0.016863690055886074,
           0.002020864183965532, 0.017683588216924864, 0.054224227833375285,
           2.429437994037918e-9, 0.1582163369456717, 3.1632003770504287e-9,
           1.0927844041886865e-8, 0.2368971535868154, 7.068169136095108e-10,
           5.920404944790139e-10, 0.1278319758420608, 0.00035110130472155846,
           0.0009122170031755927, 0.04394981566186466, 0.1827244653614497,
           5.119296838076666e-9, 0.10188699928986843]

    w3t = [0.014061730021956926, 0.04237681086380523, 0.016864110455008855,
           0.0020208888808363407, 0.017683080541691185, 0.05422435324481363,
           5.030901763872749e-9, 0.1582160486026472, 5.03495250609392e-9,
           1.2440276395608508e-8, 0.236897025647305, 1.7141578448104994e-9,
           1.5470716256507083e-9, 0.12783204379901414, 0.00035130719202668934,
           0.0009121497716635496, 0.04395022638702546, 0.18272466457782433,
           6.5525582825696e-9, 0.10188552769446294]

    w4t = [0.009636077396557733, 0.056522878655191816, 0.008701105968682678,
           0.010138087680654306, 0.07101726645597514, 2.9797322967830606e-9,
           0.0001869270397029465, 0.1416719477864801, 3.6178511627857463e-9,
           0.011542526970877365, 0.20339473896586552, 1.6827404471768724e-10,
           2.0799570288327862e-10, 0.07770129172150758, 7.499137953151453e-10,
           0.018226433419953343, 0.08533894547207829, 0.171730231622432,
           0.034453336654939326, 0.09973819646533492]

    w5t = [0.00954001462989987, 0.05608884354388105, 0.009099080956435662,
           0.010229893132460477, 0.07101511634469561, 2.565766175478752e-9,
           0.00017197124742815046, 0.14184884914672502, 2.926730409653829e-9,
           0.011612816167616218, 0.2033085543007606, 1.2765261822267684e-9,
           1.2968716833134146e-9, 0.07784157797062317, 1.0738275904388476e-11,
           0.018210092020721925, 0.0851246938503036, 0.17133671710044868,
           0.03464895781294488, 0.09992281369842247]

    w6t = [0.0096541945646638, 0.05668225606733838, 0.008734348582497732,
           0.0100326150536412, 0.07088276444739741, 4.276962566683111e-9,
           0.00027417427393046617, 0.14173946581018104, 5.338070235053351e-9,
           0.011687234549490623, 0.20349109883888222, 5.528641142269042e-10,
           5.939594758177769e-10, 0.0779325440019444, 1.5506089092558476e-9,
           0.018197720512202992, 0.08532495894732768, 0.17148151873340392,
           0.03409940792900322, 0.09978568537562955]

    w7t = [9.259127402925153e-9, 2.2980515040591824e-8, 2.3804308460720392e-8,
           1.318154641226635e-8, 0.6622515657528877, 2.830715027119353e-9,
           0.04259149391317827, 1.3386835891152789e-8, 2.732658446888359e-8,
           1.1778099607993433e-8, 2.0403898522298196e-8, 2.0621159463481253e-9,
           1.463960975170087e-9, 5.1526331089249376e-9, 1.5995178505492494e-9,
           0.1343663509523453, 0.08738455173733144, 1.893470517579942e-8,
           0.07340584371041851, 1.9769274963389775e-8]

    w8t = [2.865116810636884e-10, 8.566189001237221e-10, 1.1842084933511273e-9,
           3.9422398849504083e-10, 0.5088019188786548, 8.2865359095358e-11,
           0.03160692589837326, 1.2597713673351562e-9, 6.058515329727305e-10,
           3.2048345020222015e-10, 0.002869078356749755, 5.646121575420229e-11,
           4.159999898477776e-11, 1.57942004849256e-10, 4.480374565906585e-11,
           0.13826750831479348, 0.18567334197059143, 1.942405557940144e-9,
           0.1327812182610064, 1.086083910510438e-9]

    w9t = [6.5627975823617265e-9, 1.708219707871963e-8, 2.0352779640863774e-8,
           9.634211952954316e-9, 0.579196619439176, 2.0683739588925613e-9,
           0.03882329697879108, 1.3557085658699045e-8, 1.6099151485333377e-8,
           7.715829334727869e-9, 2.3048766179716736e-8, 1.4766939853267195e-9,
           1.080936816201994e-9, 3.808855941796661e-9, 1.163216634210376e-9,
           0.1359799531885089, 0.13967447490489168, 1.816820112558695e-8,
           0.10632549752410367, 1.6145431300136472e-8]

    w10t = [1.139103743825721e-7, 1.2165114028756902e-7, 1.5829912077419795e-7,
            1.4121491399200763e-7, 2.2449189761828184e-5, 5.9461623380797726e-8,
            0.9999756137337178, 8.487797249360217e-8, 1.2947233335875742e-7,
            9.194803754104378e-8, 8.268440835833502e-8, 6.173917064519365e-8,
            4.8030429477916233e-8, 7.155970725833292e-8, 4.9508511479068365e-8,
            2.2473430854167673e-7, 1.6687673060228055e-7, 8.865268379188848e-8,
            1.3486060100000475e-7, 1.0759445301174015e-7]

    w11t = [5.590140426506262e-10, 8.047430053051441e-10, 1.355052546821628e-9,
            1.0888618435428773e-9, 0.8503233835946482, 5.281966476534393e-10,
            0.1496766019623376, 8.414986256120907e-11, 9.784820820672162e-10,
            1.9988763752742648e-10, 3.6196486354996156e-11, 4.4873471202598566e-10,
            8.448387109276434e-10, 2.5476375354819673e-10, 5.571346334333825e-10,
            3.1351155738599574e-9, 1.6938060660241428e-9, 1.7181114897362135e-10,
            1.1543622525787854e-9, 5.478633204877463e-10]

    w12t = [2.5742355639191003e-9, 2.7707095338095783e-9, 3.4398915027060286e-9,
            3.0946403611644163e-9, 0.853395112703566, 1.1885293695688552e-9,
            0.14660484419495157, 1.8615532784981483e-9, 2.9537909540856124e-9,
            1.9597683145029672e-9, 1.7573815678039295e-9, 1.1370240886156044e-9,
            8.260556319078895e-10, 1.4588631844041763e-9, 8.344564898483304e-10,
            5.929979283721413e-9, 3.8282649737436145e-9, 1.9764014752773826e-9,
            3.157429415745761e-9, 2.3525073356166437e-9]

    @test isapprox(w1.weights, w1t)
    @test isapprox(w2.weights, w2t)
    @test isapprox(w3.weights, w3t)
    @test isapprox(w2.weights, w3.weights, rtol = 1e-5)
    @test isapprox(w4.weights, w4t)
    @test isapprox(w5.weights, w5t)
    @test isapprox(w6.weights, w6t, rtol = 0.0001)
    @test isapprox(w5.weights, w6.weights, rtol = 1e-2)
    @test isapprox(w7.weights, w7t)
    @test isapprox(w8.weights, w8t, rtol = 1e-7)
    @test isapprox(w9.weights, w9t, rtol = 1e-4)
    @test isapprox(w8.weights, w9.weights, atol = 1e-1)
    @test isapprox(w10.weights, w10t)
    @test isapprox(w11.weights, w11t)
    @test isapprox(w12.weights, w12t, rtol = 1e-6)
    @test isapprox(w11.weights, w12.weights, rtol = 1e-2)
    @test isapprox(w13.weights, w7.weights, rtol = 1e-5)
    @test isapprox(w14.weights, w8.weights, rtol = 1e-1)
    @test isapprox(w15.weights, w9.weights, rtol = 1e-1)
    @test isapprox(w16.weights, w7.weights, rtol = 1e-5)
    @test isapprox(w17.weights, w8.weights, rtol = 1e-1)
    @test isapprox(w18.weights, w9.weights, rtol = 1e-1)
    @test isapprox(w13.weights, w16.weights, rtol = 1e-5)
    @test isapprox(w14.weights, w17.weights, rtol = 1e-3)
    @test isapprox(w15.weights, w18.weights, rtol = 1e-3)
    @test isapprox(w19.weights, w1.weights, rtol = 2e-1)
end

@testset "$(:Classic), $(:Trad), $(:SSD)" begin
    portfolio = Portfolio(; prices = prices,
                          solvers = OrderedDict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                                  :params => Dict("verbose" => false,
                                                                                  "max_step_fraction" => 0.75))))
    asset_statistics!(portfolio)

    rm = :SSD
    opt = OptimiseOpt(; rf = rf, l = l, class = :Classic, type = :Trad, rm = rm,
                      obj = :Min_Risk, kelly = :None)

    w1 = optimise!(portfolio, opt)
    risk1 = calc_risk(portfolio; type = :Trad, rm = rm, rf = rf)
    opt.kelly = :Approx
    w2 = optimise!(portfolio, opt)
    opt.kelly = :Exact
    w3 = optimise!(portfolio, opt)
    opt.obj = :Utility
    opt.kelly = :None
    w4 = optimise!(portfolio, opt)
    opt.kelly = :Approx
    w5 = optimise!(portfolio, opt)
    opt.kelly = :Exact
    w6 = optimise!(portfolio, opt)
    opt.obj = :Sharpe
    opt.kelly = :None
    w7 = optimise!(portfolio, opt)
    risk7 = calc_risk(portfolio; type = :Trad, rm = rm, rf = rf)
    ret7 = dot(portfolio.mu, w7.weights)
    opt.kelly = :Approx
    w8 = optimise!(portfolio, opt)
    risk8 = calc_risk(portfolio; type = :Trad, rm = rm, rf = rf)
    ret8 = dot(portfolio.mu, w8.weights)
    opt.kelly = :Exact
    w9 = optimise!(portfolio, opt)
    risk9 = calc_risk(portfolio; type = :Trad, rm = rm, rf = rf)
    ret9 = dot(portfolio.mu, w9.weights)
    opt.obj = :Max_Ret
    opt.kelly = :None
    w10 = optimise!(portfolio, opt)
    opt.kelly = :Approx
    w11 = optimise!(portfolio, opt)
    opt.kelly = :Exact
    w12 = optimise!(portfolio, opt)
    setproperty!(portfolio, :mu_l, ret7)
    opt.obj = :Min_Risk
    opt.kelly = :None
    w13 = optimise!(portfolio, opt)
    setproperty!(portfolio, :mu_l, ret8)
    w14 = optimise!(portfolio, opt)
    setproperty!(portfolio, :mu_l, ret9)
    w15 = optimise!(portfolio, opt)
    setproperty!(portfolio, :mu_l, Inf)
    rmf = Symbol(lowercase(string(rm)) * "_u")
    setproperty!(portfolio, rmf, risk7)
    opt.obj = :Max_Ret
    opt.kelly = :None
    w16 = optimise!(portfolio, opt)
    setproperty!(portfolio, rmf, risk8)
    w17 = optimise!(portfolio, opt)
    setproperty!(portfolio, rmf, risk9)
    w18 = optimise!(portfolio, opt)
    setproperty!(portfolio, rmf, risk1)
    opt.obj = :Sharpe
    w19 = optimise!(portfolio, opt)
    setproperty!(portfolio, rmf, Inf)

    w1t = [1.394350488871705e-8, 0.049737895947338455, 6.600055984590935e-9,
           0.0029778225534967816, 0.0025509406974169035, 0.020133111426581966,
           3.96407814766883e-10, 0.12809466424134056, 1.091051157683022e-9,
           1.6828147395583045e-9, 0.29957700458521846, 1.4416242671178354e-9,
           6.7051791521021e-10, 0.12069625891775575, 0.012266264709254888,
           0.009663136606939291, 9.254652171638183e-9, 0.22927585651100002,
           1.4643960792067457e-9, 0.12502700725863183]

    w2t = [9.746174068955984e-9, 0.04973789569850327, 4.7750975410853275e-9,
           0.0029778067761385626, 0.002550940877551505, 0.02013309085193254,
           2.89170586675408e-10, 0.12809466159008298, 8.077887003456104e-10,
           1.216195010127782e-9, 0.29957701278968896, 1.0860556690394684e-9,
           4.98783966477131e-10, 0.1206962245419088, 0.012266274829176258,
           0.009663136328150185, 6.42805307973352e-9, 0.22927589624624395,
           1.0587280733807273e-9, 0.12502703356457634]

    w3t = [2.535385452843278e-8, 0.04973791171761704, 1.3356096979722265e-8,
           0.002977888213474425, 0.0025509065906121338, 0.020133153061788464,
           1.5429006918608676e-9, 0.1280946708020185, 3.311367123034686e-9,
           4.4616832340809484e-9, 0.2995769790601962, 4.324345104112863e-9,
           2.3318855924543267e-9, 0.12069636820390635, 0.012266230547308361,
           0.00966313789227606, 1.6527732522078397e-8, 0.22927574890075175,
           4.062144167891954e-9, 0.1250269297380408]

    w4t = [2.3442083139305346e-9, 0.05530879616102099, 2.2432983413301295e-9,
           0.0042754028543387935, 0.033908145414542276, 1.975947149297289e-9,
           2.2852878216316096e-10, 0.12521698450276408, 6.468480453075671e-10,
           7.878643523121792e-10, 0.29837138348123077, 3.5156738668474465e-10,
           1.861724586982361e-10, 0.10724336067545101, 0.0027603786840747434,
           0.02112052308165311, 0.005355660646086244, 0.22898943165976857,
           1.0040455850648044e-9, 0.11744992307058902]

    w5t = [2.731097314139538e-10, 0.05518564827160002, 2.4715374705672034e-10,
           0.004440204616525222, 0.033778842744289314, 2.892231064130228e-10,
           1.567469116811235e-10, 0.12529015632354162, 5.832780512799746e-11,
           5.0445449112832445e-11, 0.29839769018352, 9.666613329275674e-11,
           1.46557226947914e-10, 0.10741008220940933, 0.0027253648392198174,
           0.02106984860692222, 0.0053631774992532934, 0.2288343448370201,
           4.645031105397796e-12, 0.11750463854582387]

    w6t = [3.6963998014182935e-9, 0.05518903048141686, 3.4599618567368905e-9,
           0.0044374647398944455, 0.03376640323328681, 3.2510090714661645e-9,
           6.035569216006214e-10, 0.12529118190304275, 1.2618185082003576e-9,
           1.426408649803653e-9, 0.2984002799241621, 7.20856652878251e-10,
           4.411560345980172e-10, 0.10741594769521565, 0.002730634438652209,
           0.02106510515327547, 0.005352551419669141, 0.22884014402081754,
           1.8378569753810924e-9, 0.11751124029154254]

    w7t = [3.6633520206614757e-9, 7.97412245318715e-9, 6.634878572730317e-9,
           4.325994932493507e-9, 0.6666007050803732, 1.2187225161796183e-9,
           0.03791876510948157, 6.335333878784392e-9, 7.972707882636176e-9,
           3.5077686177729347e-9, 7.991725521354389e-9, 1.0154257280076322e-9,
           6.872634730766401e-10, 2.3746418322360103e-9, 7.203415342153691e-10,
           0.17184313837916457, 0.1025928710370009, 8.407168803543539e-9,
           0.021044450856490815, 6.708041316141942e-9]

    w8t = [6.394795458287015e-9, 2.117099021138723e-8, 1.3412130484585112e-8,
           7.931377874754428e-9, 0.5660579919475337, 2.0800052766669152e-9,
           0.029043969015690303, 2.6888534953967605e-8, 1.1783199925120606e-8,
           6.371578929988278e-9, 7.517943958325472e-8, 1.6440978843757989e-9,
           1.197870953556057e-9, 4.294880525042789e-9, 1.2389946291903091e-9,
           0.16425007174420087, 0.1629081622443728, 4.266854113163681e-8,
           0.07773956532220085, 1.746956364298778e-8]

    w9t = [1.691732873281931e-7, 4.1902811252473786e-7, 3.2433701622935056e-7,
           2.0739127262420673e-7, 0.6079810913616143, 5.8253204203366666e-8,
           0.03580115851913769, 3.8006144372105117e-7, 3.4322252514798573e-7,
           1.6628893025466347e-7, 5.199176790478948e-7, 4.788628324839719e-8,
           3.4435993172487995e-8, 1.1183475431767843e-7, 3.5668106101310785e-8,
           0.16670483358415644, 0.13731644935712478, 5.09438279984467e-7,
           0.05219279055456652, 3.4968651256721096e-7]

    w10t = [1.1485876083980298e-7, 1.2286765161003074e-7, 1.6067522891592387e-7,
            1.432742280764689e-7, 2.0215920126734213e-5, 5.868758772304026e-8,
            0.9999778332231143, 8.449637852338254e-8, 1.3130622753084616e-7,
            9.220063030215003e-8, 8.219286864317247e-8, 6.148495243723356e-8,
            4.796247629799614e-8, 7.093757944889581e-8, 4.9097257233201166e-8,
            2.2817632264582585e-7, 1.6931479760380464e-7, 8.84478903741339e-8,
            1.366687610663341e-7, 1.0820715964815496e-7]

    w11t = [4.4372756954496346e-10, 6.219272838830198e-10, 1.0871321089702985e-9,
            8.941565667803825e-10, 0.8503215309121464, 4.0728045462917795e-10,
            0.1496784575050638, 2.0016153643590607e-11, 7.817855997414043e-10,
            1.2437789214942929e-10, 7.213926439417411e-11, 3.915637626349398e-10,
            6.984324771309397e-10, 2.631203634530896e-10, 5.869885908282255e-10,
            2.5199832741308024e-9, 1.338999260842608e-9, 5.713691782390282e-11,
            8.926917142521769e-10, 3.8133044804180164e-10]

    w12t = [2.8243326567075264e-9, 2.980258201518003e-9, 3.648296676120534e-9,
            3.3087547488694207e-9, 0.8533950941534197, 1.3065937988602602e-9,
            0.14660485977130017, 2.003799928761554e-9, 3.144096791619841e-9,
            2.0870913964972194e-9, 1.8792225125246225e-9, 1.226177591622567e-9,
            8.891085588928079e-10, 1.5806196269766107e-9, 8.961420041745974e-10,
            6.277091776636327e-9, 4.030404030118061e-9, 2.143059400219315e-9,
            3.358758052088439e-9, 2.491472323754614e-9]

    @test isapprox(w1.weights, w1t)
    @test isapprox(w2.weights, w2t)
    @test isapprox(w3.weights, w3t)
    @test isapprox(w2.weights, w3.weights, rtol = 1e-6)
    @test isapprox(w4.weights, w4t)
    @test isapprox(w5.weights, w5t)
    @test isapprox(w6.weights, w6t, rtol = 1e-7)
    @test isapprox(w5.weights, w6.weights, rtol = 1e-4)
    @test isapprox(w7.weights, w7t)
    @test isapprox(w8.weights, w8t, rtol = 1.0e-7)
    @test isapprox(w9.weights, w9t, rtol = 1e-4)
    @test isapprox(w8.weights, w9.weights, rtol = 1e-1)
    @test isapprox(w10.weights, w10t)
    @test isapprox(w11.weights, w11t)
    @test isapprox(w12.weights, w12t, rtol = 1e-7)
    @test isapprox(w11.weights, w12.weights, rtol = 1e-2)
    @test isapprox(w13.weights, w7.weights, rtol = 1e-5)
    @test isapprox(w14.weights, w8.weights, rtol = 1e-1)
    @test isapprox(w15.weights, w9.weights, rtol = 1e-1)
    @test isapprox(w16.weights, w7.weights, rtol = 1e-5)
    @test isapprox(w17.weights, w8.weights, rtol = 1e-1)
    @test isapprox(w18.weights, w9.weights, rtol = 1e-1)
    @test isapprox(w13.weights, w16.weights, rtol = 1e-5)
    @test isapprox(w14.weights, w17.weights, rtol = 1e-3)
    @test isapprox(w15.weights, w18.weights, rtol = 1e-3)
    @test isapprox(w19.weights, w1.weights, rtol = 1e-3)
end

@testset "$(:Classic), $(:Trad), $(:FLPM)" begin
    portfolio = Portfolio(; prices = prices,
                          solvers = OrderedDict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                                  :params => Dict("verbose" => false,
                                                                                  "max_step_fraction" => 0.75))))
    asset_statistics!(portfolio)

    rm = :FLPM
    opt = OptimiseOpt(; rf = rf, l = l, class = :Classic, type = :Trad, rm = rm,
                      obj = :Min_Risk, kelly = :None)

    w1 = optimise!(portfolio, opt)
    risk1 = calc_risk(portfolio; type = :Trad, rm = rm, rf = rf)
    opt.kelly = :Approx
    w2 = optimise!(portfolio, opt)
    opt.kelly = :Exact
    w3 = optimise!(portfolio, opt)
    opt.obj = :Utility
    opt.kelly = :None
    w4 = optimise!(portfolio, opt)
    opt.kelly = :Approx
    w5 = optimise!(portfolio, opt)
    opt.kelly = :Exact
    w6 = optimise!(portfolio, opt)
    opt.obj = :Sharpe
    opt.kelly = :None
    w7 = optimise!(portfolio, opt)
    risk7 = calc_risk(portfolio; type = :Trad, rm = rm, rf = rf)
    ret7 = dot(portfolio.mu, w7.weights)
    opt.kelly = :Approx
    w8 = optimise!(portfolio, opt)
    risk8 = calc_risk(portfolio; type = :Trad, rm = rm, rf = rf)
    ret8 = dot(portfolio.mu, w8.weights)
    opt.kelly = :Exact
    w9 = optimise!(portfolio, opt)
    risk9 = calc_risk(portfolio; type = :Trad, rm = rm, rf = rf)
    ret9 = dot(portfolio.mu, w9.weights)
    opt.obj = :Max_Ret
    opt.kelly = :None
    w10 = optimise!(portfolio, opt)
    opt.kelly = :Approx
    w11 = optimise!(portfolio, opt)
    opt.kelly = :Exact
    w12 = optimise!(portfolio, opt)
    setproperty!(portfolio, :mu_l, ret7)
    opt.obj = :Min_Risk
    opt.kelly = :None
    w13 = optimise!(portfolio, opt)
    setproperty!(portfolio, :mu_l, ret8)
    w14 = optimise!(portfolio, opt)
    setproperty!(portfolio, :mu_l, ret9)
    w15 = optimise!(portfolio, opt)
    setproperty!(portfolio, :mu_l, Inf)
    rmf = Symbol(lowercase(string(rm)) * "_u")
    setproperty!(portfolio, rmf, risk7)
    opt.obj = :Max_Ret
    opt.kelly = :None
    w16 = optimise!(portfolio, opt)
    setproperty!(portfolio, rmf, risk8)
    w17 = optimise!(portfolio, opt)
    setproperty!(portfolio, rmf, risk9)
    w18 = optimise!(portfolio, opt)
    setproperty!(portfolio, rmf, risk1)
    opt.obj = :Sharpe
    w19 = optimise!(portfolio, opt)
    setproperty!(portfolio, rmf, Inf)

    w1t = [0.004266046503603202, 0.043621461545148975, 0.019960422989327348,
           0.007822756750752771, 0.06052586949560898, 4.0345183053999036e-8,
           0.00039583799738645854, 0.13089220651521072, 1.3627505331294202e-8,
           0.01187856290823966, 0.20660940008866496, 6.180519691106305e-10,
           5.042819710864134e-10, 0.08329493079429666, 1.8160563449462696e-9,
           0.013888103353452178, 0.08734668352615117, 0.19210100327920146,
           0.0372130179843601, 0.10018363935751669]

    w2t = [0.004266038406989898, 0.04362141374057654, 0.01996041608700697,
           0.007822753833683773, 0.06052594102685525, 3.612859481405071e-8,
           0.0003958251843227585, 0.13089214317134815, 1.1947644986105599e-8,
           0.011878533868833297, 0.2066093811568535, 6.082982926003325e-10,
           5.544216597521101e-10, 0.083294966133445, 2.034567074874004e-9,
           0.013888101932479597, 0.08734663027830582, 0.1921010869730084,
           0.037212981112549906, 0.10018373582021434]

    w3t = [0.004266009604713132, 0.04362129116013896, 0.01996037621721602,
           0.007822727640623991, 0.06052631504374161, 2.8031851589952083e-8,
           0.00039576459421439834, 0.13089190822567823, 1.114264609745513e-8,
           0.011878362355194717, 0.20660924668882039, 1.4584139025061787e-9,
           1.3601211573659023e-9, 0.08329506673094936, 2.9164985265422323e-9,
           0.013888113597890274, 0.08734622171236156, 0.1921014930512477,
           0.03721282568166173, 0.10018423278601674]

    w4t = [1.3946345703391873e-8, 0.04213859506687942, 0.012527983311869585,
           0.007205812506697446, 0.09868196462765234, 8.657980218448944e-10,
           0.002772634291894064, 0.1092113824473775, 3.601092313897938e-9,
           1.2349635284211694e-8, 0.2015471802053637, 1.583383447405798e-10,
           1.9010385369085806e-10, 0.02654475238042611, 3.72281218977062e-10,
           0.04173081725737542, 0.11079911217187115, 0.1723272726602518,
           0.06799949425703258, 0.10651296733171428]

    w5t = [5.841786285792397e-9, 0.03801562362967448, 0.015511754308399954,
           0.007919818513720612, 0.09985215446523957, 9.673549201842328e-12,
           0.0024877489488354374, 0.1050289522683147, 6.980808305348869e-10,
           4.092961477385218e-9, 0.20300827982740322, 5.772785080444301e-10,
           5.570272334452517e-10, 0.03503779268561045, 1.8540707556019774e-10,
           0.03798362119810868, 0.10538923116465836, 0.17777130622911355,
           0.0629549443408573, 0.10903876045784872]

    w6t = [1.0845347856914958e-8, 0.037685133985301295, 0.015473444699338519,
           0.008118756931064931, 0.09893115345910755, 9.64736380826414e-10,
           0.0025330757931410707, 0.10486216352332926, 2.6047015834936564e-9,
           7.700350532863383e-9, 0.20310278424763056, 3.0268647554552327e-10,
           3.024395010078847e-10, 0.0355519575425661, 4.752808201333804e-10,
           0.03730321185963623, 0.10541402070980747, 0.1786236714519576,
           0.06248099688485916, 0.10991960571671706]

    w7t = [4.012395522980917e-9, 9.795014596668397e-9, 1.021051159482409e-8,
           6.105308927681581e-9, 0.6999045875537633, 1.4553846281077864e-9,
           0.02929533509874303, 7.146944663300512e-9, 1.3171977288831124e-8,
           5.575542620503757e-9, 9.396495126732703e-9, 1.0445189519913473e-9,
           7.533264814163661e-10, 2.622411683213462e-9, 8.309436055001469e-10,
           0.15181155191354562, 0.04227264588275489, 9.085900119356214e-9,
           0.07671578769781048, 1.0646706888927673e-8]

    w8t = [1.6704606434760852e-10, 4.5039920116288666e-10, 5.370330127992086e-10,
           2.394136694374759e-10, 0.5637246963905376, 5.4482958004525573e-11,
           0.026029771670945778, 5.074146570381013e-10, 4.4071340641401345e-10,
           2.0515563403397109e-10, 3.0129094144403086e-9, 3.740126956304181e-11,
           2.843171837240471e-11, 1.0095673103350372e-10, 3.1024271719048774e-11,
           0.14917729111548997, 0.12978983681715497, 1.0276842198752544e-9,
           0.13127839651981474, 6.459909068665672e-10]

    w9t = [5.513677262615502e-9, 1.3319306029473428e-8, 1.445211952716753e-8,
           8.048285173219424e-9, 0.6065957128792241, 1.899895339491111e-9,
           0.028288103620619738, 1.1211637167620422e-8, 1.5010751784462062e-8,
           6.549897771257781e-9, 1.749689668249809e-8, 1.3414664661068578e-9,
           1.008238949289819e-9, 3.3893250664741667e-9, 1.0935767675604547e-9,
           0.1497761499074153, 0.10823323886404156, 1.6084711454039298e-8,
           0.10710666267669791, 1.5632215990894862e-8]

    w10t = [1.139342832777196e-7, 1.216775588491462e-7, 1.583370741783028e-7,
            1.4124741887434891e-7, 2.2449677331304052e-5, 5.947269477975765e-8,
            0.9999756128272367, 8.489347599154916e-8, 1.2950108288968402e-7,
            9.196535598865469e-8, 8.269936707363312e-8, 6.175045961808709e-8,
            4.8040494339997626e-8, 7.157236827561222e-8, 4.951884124586472e-8,
            2.2479084911015693e-7, 1.6691743892807265e-7, 8.866918314721361e-8,
            1.348911339100182e-7, 1.0761635157039321e-7]

    w11t = [5.592940960122782e-10, 8.051361938960762e-10, 1.3555250476576508e-9,
            1.089331283958596e-9, 0.850323382369204, 5.281736431720078e-10,
            0.1496766031811145, 8.438798557233486e-11, 9.7904156480956e-10,
            2.0020970096415077e-10, 3.637688376357992e-11, 4.489199537756185e-10,
            8.448954864437396e-10, 2.547478199098988e-10, 5.579366440452939e-10,
            3.13608043731199e-9, 1.6943988689226196e-9, 1.7203692251059194e-10,
            1.1549548202484472e-9, 5.482342721063637e-10]

    w12t = [2.8597303785264986e-9, 3.078157385006265e-9, 3.821589665374536e-9,
            3.438222641385142e-9, 0.8533950109766304, 1.3202413632655985e-9,
            0.14660494113891653, 2.068198575405446e-9, 3.2816386735455967e-9,
            2.177422503950781e-9, 1.9525167916694454e-9, 1.263207669208692e-9,
            9.177086762501277e-10, 1.6207504402008501e-9, 9.270556823416673e-10,
            6.587333413347233e-9, 4.253150491743481e-9, 2.195759329554425e-9,
            3.5079448927359886e-9, 2.6138245754126665e-9]

    @test isapprox(w1.weights, w1t)
    @test isapprox(w2.weights, w2t)
    @test isapprox(w3.weights, w3t)
    @test isapprox(w2.weights, w3.weights, rtol = 1e-5)
    @test isapprox(w4.weights, w4t)
    @test isapprox(w5.weights, w5t)
    @test isapprox(w6.weights, w6t, rtol = 1e-4)
    @test isapprox(w5.weights, w6.weights, rtol = 1e-2)
    @test isapprox(w7.weights, w7t)
    @test isapprox(w8.weights, w8t)
    @test isapprox(w9.weights, w9t, rtol = 1e-4)
    @test isapprox(w8.weights, w9.weights, rtol = 1e-1)
    @test isapprox(w10.weights, w10t)
    @test isapprox(w11.weights, w11t)
    @test isapprox(w12.weights, w12t, rtol = 1e-6)
    @test isapprox(w11.weights, w12.weights, rtol = 1e-2)
    @test isapprox(w13.weights, w7.weights, rtol = 1e-5)
    @test isapprox(w14.weights, w8.weights, rtol = 1e-1)
    @test isapprox(w15.weights, w9.weights, rtol = 1e-1)
    @test isapprox(w16.weights, w7.weights, rtol = 1e-5)
    @test isapprox(w17.weights, w8.weights, rtol = 1e-1)
    @test isapprox(w18.weights, w9.weights, rtol = 1e-1)
    @test isapprox(w13.weights, w16.weights, rtol = 1e-5)
    @test isapprox(w14.weights, w17.weights, rtol = 1e-4)
    @test isapprox(w15.weights, w18.weights, rtol = 1e-3)
    @test isapprox(w19.weights, w1.weights, rtol = 1e-4)
end

@testset "$(:Classic), $(:Trad), $(:SLPM)" begin
    portfolio = Portfolio(; prices = prices,
                          solvers = OrderedDict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                                  :params => Dict("verbose" => false,
                                                                                  "max_step_fraction" => 0.75))))
    asset_statistics!(portfolio)

    rm = :SLPM
    opt = OptimiseOpt(; rf = rf, l = l, class = :Classic, type = :Trad, rm = rm,
                      obj = :Min_Risk, kelly = :None)

    w1 = optimise!(portfolio, opt)
    risk1 = calc_risk(portfolio; type = :Trad, rm = rm, rf = rf)
    opt.kelly = :Approx
    w2 = optimise!(portfolio, opt)
    opt.kelly = :Exact
    w3 = optimise!(portfolio, opt)
    opt.obj = :Utility
    opt.kelly = :None
    w4 = optimise!(portfolio, opt)
    opt.kelly = :Approx
    w5 = optimise!(portfolio, opt)
    opt.kelly = :Exact
    w6 = optimise!(portfolio, opt)
    opt.obj = :Sharpe
    opt.kelly = :None
    w7 = optimise!(portfolio, opt)
    risk7 = calc_risk(portfolio; type = :Trad, rm = rm, rf = rf)
    ret7 = dot(portfolio.mu, w7.weights)
    opt.kelly = :Approx
    w8 = optimise!(portfolio, opt)
    risk8 = calc_risk(portfolio; type = :Trad, rm = rm, rf = rf)
    ret8 = dot(portfolio.mu, w8.weights)
    opt.kelly = :Exact
    w9 = optimise!(portfolio, opt)
    risk9 = calc_risk(portfolio; type = :Trad, rm = rm, rf = rf)
    ret9 = dot(portfolio.mu, w9.weights)
    opt.obj = :Max_Ret
    opt.kelly = :None
    w10 = optimise!(portfolio, opt)
    opt.kelly = :Approx
    w11 = optimise!(portfolio, opt)
    opt.kelly = :Exact
    w12 = optimise!(portfolio, opt)
    setproperty!(portfolio, :mu_l, ret7)
    opt.obj = :Min_Risk
    opt.kelly = :None
    w13 = optimise!(portfolio, opt)
    setproperty!(portfolio, :mu_l, ret8)
    w14 = optimise!(portfolio, opt)
    setproperty!(portfolio, :mu_l, ret9)
    w15 = optimise!(portfolio, opt)
    setproperty!(portfolio, :mu_l, Inf)
    rmf = Symbol(lowercase(string(rm)) * "_u")
    setproperty!(portfolio, rmf, risk7)
    opt.obj = :Max_Ret
    opt.kelly = :None
    w16 = optimise!(portfolio, opt)
    setproperty!(portfolio, rmf, risk8)
    w17 = optimise!(portfolio, opt)
    setproperty!(portfolio, rmf, risk9)
    w18 = optimise!(portfolio, opt)
    setproperty!(portfolio, rmf, risk1)
    opt.obj = :Sharpe
    w19 = optimise!(portfolio, opt)
    setproperty!(portfolio, rmf, Inf)

    w1t = [5.766882799545668e-9, 0.05524530843954044, 5.351639326485838e-9,
           0.004319542520506111, 0.03359647871518277, 4.912676762096846e-9,
           5.630810074797396e-10, 0.1247807805565492, 1.5696246816367933e-9,
           1.97211147076299e-9, 0.3005645006808748, 8.526182535263627e-10,
           4.81893631976952e-10, 0.106618715763935, 0.003124122237380697,
           0.02139164452732346, 0.003594091295751131, 0.22965139768181642,
           2.5529020812733613e-9, 0.11711339355771]

    w2t = [4.021634615249811e-9, 0.05524532610725361, 3.825750373543256e-9,
           0.004319538078000414, 0.03359646920623926, 3.498808069445511e-9,
           4.151554805196016e-10, 0.1247808003007947, 1.1584234365750364e-9,
           1.4264784061906804e-9, 0.30056452350243695, 6.341079784184448e-10,
           3.4935642659484896e-10, 0.10661869350352154, 0.003124129694510038,
           0.021391636381936932, 0.0035940418666616535, 0.2296514061440249,
           1.8247471494050523e-9, 0.11711341806015815]

    w3t = [1.1253420971468167e-8, 0.055245273549437444, 1.0516057137279787e-8,
           0.004319557472773191, 0.03359652322567405, 9.90331548475738e-9,
           2.110502719541927e-9, 0.12478069608436998, 4.2240224301503164e-9,
           4.694877715093118e-9, 0.3005644723289171, 2.4758886719149437e-9,
           1.5298332098736224e-9, 0.10661878666024294, 0.0031240959279081474,
           0.021391685026921726, 0.0035941932519648256, 0.22965134365379075,
           6.081911703684676e-9, 0.11711332002816967]

    w4t = [1.1368746242613526e-9, 0.0551050860578228, 1.453433912689154e-9,
           0.0004237975147989165, 0.06590335344711847, 5.461538829876965e-10,
           3.0649525901391164e-10, 0.11928808875496356, 7.757607025809469e-10,
           7.780918748102178e-10, 0.2937385787823647, 2.2065208914441143e-10,
           1.1589470842548216e-10, 0.07604818120102969, 7.428148331136416e-10,
           0.03452492530750511, 0.02823617279076166, 0.22320471114886994,
           1.763495741887641e-9, 0.10352709715509763]

    w5t = [2.6299772986419097e-11, 0.05497684342750627, 4.166462526119003e-11,
           0.0006337759947072918, 0.06550297576540559, 7.37689000861037e-12,
           6.159929911659955e-11, 0.11941531441774188, 8.411516117791819e-12,
           1.53724297591133e-11, 0.29378254436545476, 5.216451373118523e-11,
           6.770186262664022e-11, 0.07640300366525808, 3.9983720134880635e-11,
           0.034355575245763464, 0.02812692802786668, 0.22309576225589772,
           5.5484170265411745e-11, 0.1037072764583394]

    w6t = [1.636578097996472e-9, 0.05498060213065741, 1.9945581169052047e-9,
           0.0006319442684968718, 0.06548844904743839, 8.38849294502362e-10,
           6.602585498793782e-10, 0.1194167179641163, 1.1975540030833824e-9,
           1.1386649524694993e-9, 0.2937858416148934, 3.9490576846855183e-10,
           2.4950935723077373e-10, 0.07641754464706908, 1.0246815353765307e-9,
           0.034350503435374796, 0.028114023680122082, 0.22310037316006634,
           2.3960169713245868e-9, 0.10371398852018866]

    w7t = [3.2656053026749458e-9, 7.068556226444792e-9, 5.813255866456905e-9,
           3.7848800723812255e-9, 0.6654376552808184, 1.0783218793449591e-9,
           0.0380722528505957, 5.695693706920986e-9, 7.02206353232648e-9,
           3.07726233117816e-9, 7.093496541942908e-9, 9.083523397986405e-10,
           6.123027831254452e-10, 2.129221861280047e-9, 6.382252967288842e-10,
           0.17459113358431838, 0.10411293923415325, 7.47768560795025e-9,
           0.017785957468977102, 5.916213811100413e-9]

    w8t = [6.069692264540452e-9, 1.9282915104476062e-8, 1.2103645766798553e-8,
           7.3583011837289236e-9, 0.5737583812211344, 1.9992767852555444e-9,
           0.029566394428607624, 2.3429525932943068e-8, 1.1125590353620928e-8,
           5.970025526959925e-9, 5.258287443276183e-8, 1.6052248309830733e-9,
           1.1652000292949317e-9, 4.128218585152892e-9, 1.2012620465024395e-9,
           0.16718994028264156, 0.15962930086006144, 3.499509041849996e-8,
           0.06985578457856859, 1.561214320309629e-8]

    w9t = [2.0539089574565041e-7, 4.979601606716827e-7, 3.816844739679543e-7,
           2.4701671807002115e-7, 0.6119592444409852, 7.114033110036842e-8,
           0.03614495307009323, 4.524974191219566e-7, 4.07253008719744e-7,
           1.985011501177067e-7, 6.029932825212801e-7, 5.926866316972883e-8,
           4.2497528084595413e-8, 1.365779063768648e-7, 4.390712813305422e-8,
           0.16951045465270273, 0.13609808996850242, 5.953538422406526e-7,
           0.04628290483988222, 4.109853261884776e-7]

    w10t = [1.1984144316766492e-7, 1.28373065051865e-7, 1.686558656878764e-7,
            1.5010742141568907e-7, 2.0606808081031894e-5, 6.064196727656754e-8,
            0.9999773568509776, 8.759065757919374e-8, 1.3734468998997524e-7,
            9.57353512442505e-8, 8.515788246765282e-8, 6.35162602074274e-8,
            4.957427631858815e-8, 7.334183143471595e-8, 5.077095601817701e-8,
            2.4023232664947415e-7, 1.778570021974521e-7, 9.17705800401458e-8,
            1.4307324549903535e-7, 1.1275611922443606e-7]

    w11t = [6.199048946409833e-10, 8.559871336140005e-10, 1.4950753903857514e-9,
            1.2247215833941066e-9, 0.850320606721586, 5.41440004603452e-10,
            0.1496793776832693, 2.332993267009514e-11, 1.0609488232051006e-9,
            1.7246013104296854e-10, 8.792761934730706e-11, 5.269981886531533e-10,
            8.476967747260268e-10, 3.4295627231377317e-10, 7.927566939928397e-10,
            3.3474476679829137e-9, 1.8338786520566145e-9, 8.566614126931458e-11,
            1.2143967068809675e-9, 5.215516606938051e-10]

    w12t = [3.2953340796183815e-9, 3.4671481951300583e-9, 4.241258470281023e-9,
            3.847437640117419e-9, 0.853395064254099, 1.5285472994141022e-9,
            0.14660488214870265, 2.329917157200875e-9, 3.6530553335829964e-9,
            2.4220752763401977e-9, 2.182344444020592e-9, 1.4273907167298974e-9,
            1.0358267197507206e-9, 1.8404086919639226e-9, 1.0436488985807215e-9,
            7.314996153330884e-9, 4.682190580102188e-9, 2.4935331858759923e-9,
            3.901722675299207e-9, 2.890362735629566e-9]

    @test isapprox(w1.weights, w1t)
    @test isapprox(w2.weights, w2t)
    @test isapprox(w3.weights, w3t)
    @test isapprox(w2.weights, w3.weights, rtol = 1e-6)
    @test isapprox(w4.weights, w4t)
    @test isapprox(w5.weights, w5t)
    @test isapprox(w6.weights, w6t)
    @test isapprox(w5.weights, w6.weights, rtol = 1e-4)
    @test isapprox(w7.weights, w7t)
    @test isapprox(w8.weights, w8t, rtol = 1e-7)
    @test isapprox(w9.weights, w9t, rtol = 1e-4)
    @test isapprox(w8.weights, w9.weights, rtol = 1e-1)
    @test isapprox(w10.weights, w10t)
    @test isapprox(w11.weights, w11t)
    @test isapprox(w12.weights, w12t, rtol = 1e-6)
    @test isapprox(w11.weights, w12.weights, rtol = 1e-2)
    @test isapprox(w13.weights, w7.weights, rtol = 1e-6)
    @test isapprox(w14.weights, w8.weights, rtol = 1e-1)
    @test isapprox(w15.weights, w9.weights, rtol = 1e-1)
    @test isapprox(w16.weights, w7.weights, rtol = 1e-5)
    @test isapprox(w17.weights, w8.weights, rtol = 1e-1)
    @test isapprox(w18.weights, w9.weights, rtol = 1e-1)
    @test isapprox(w13.weights, w16.weights, rtol = 1e-5)
    @test isapprox(w14.weights, w17.weights, rtol = 1e-3)
    @test isapprox(w15.weights, w18.weights, rtol = 1e-3)
    @test isapprox(w19.weights, w1.weights, rtol = 1e-3)
end

@testset "$(:Classic), $(:Trad), $(:WR)" begin
    portfolio = Portfolio(; prices = prices,
                          solvers = OrderedDict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                                  :params => Dict("verbose" => false,
                                                                                  "max_step_fraction" => 0.75))))
    asset_statistics!(portfolio)

    rm = :WR
    opt = OptimiseOpt(; rf = rf, l = l, class = :Classic, type = :Trad, rm = rm,
                      obj = :Min_Risk, kelly = :None)

    w1 = optimise!(portfolio, opt)
    risk1 = calc_risk(portfolio; type = :Trad, rm = rm, rf = rf)
    opt.kelly = :Approx
    w2 = optimise!(portfolio, opt)
    opt.kelly = :Exact
    w3 = optimise!(portfolio, opt)
    opt.obj = :Utility
    opt.kelly = :None
    w4 = optimise!(portfolio, opt)
    opt.kelly = :Approx
    w5 = optimise!(portfolio, opt)
    opt.kelly = :Exact
    w6 = optimise!(portfolio, opt)
    opt.obj = :Sharpe
    opt.kelly = :None
    w7 = optimise!(portfolio, opt)
    risk7 = calc_risk(portfolio; type = :Trad, rm = rm, rf = rf)
    ret7 = dot(portfolio.mu, w7.weights)
    opt.kelly = :Approx
    w8 = optimise!(portfolio, opt)
    risk8 = calc_risk(portfolio; type = :Trad, rm = rm, rf = rf)
    ret8 = dot(portfolio.mu, w8.weights)
    opt.kelly = :Exact
    w9 = optimise!(portfolio, opt)
    risk9 = calc_risk(portfolio; type = :Trad, rm = rm, rf = rf)
    ret9 = dot(portfolio.mu, w9.weights)
    opt.obj = :Max_Ret
    opt.kelly = :None
    w10 = optimise!(portfolio, opt)
    opt.kelly = :Approx
    w11 = optimise!(portfolio, opt)
    opt.kelly = :Exact
    w12 = optimise!(portfolio, opt)
    setproperty!(portfolio, :mu_l, ret7)
    opt.obj = :Min_Risk
    opt.kelly = :None
    w13 = optimise!(portfolio, opt)
    setproperty!(portfolio, :mu_l, ret8)
    w14 = optimise!(portfolio, opt)
    setproperty!(portfolio, :mu_l, ret9)
    w15 = optimise!(portfolio, opt)
    setproperty!(portfolio, :mu_l, Inf)
    rmf = Symbol(lowercase(string(rm)) * "_u")
    setproperty!(portfolio, rmf, risk7)
    opt.obj = :Max_Ret
    opt.kelly = :None
    w16 = optimise!(portfolio, opt)
    setproperty!(portfolio, rmf, risk8)
    w17 = optimise!(portfolio, opt)
    setproperty!(portfolio, rmf, risk9)
    w18 = optimise!(portfolio, opt)
    setproperty!(portfolio, rmf, risk1)
    opt.obj = :Sharpe
    w19 = optimise!(portfolio, opt)
    setproperty!(portfolio, rmf, Inf)

    w1t = [1.5302228351003913e-11, 0.22119703968343155, 1.5042802504566466e-11,
           1.149655345278045e-10, 3.5524668523177065e-11, 4.610749744263935e-11,
           0.02918541165092378, 7.025091033726773e-11, 1.452337436127022e-11,
           2.235138683404488e-11, 0.5455165565497712, 1.0448155077916786e-11,
           2.7150611313308743e-11, 1.2643628210669704e-11, 5.702825772004901e-10,
           3.0034901185213575e-11, 1.229864177463597e-10, 1.9411492003804697e-11,
           1.631580481195887e-11, 0.2041009909725315]

    w2t = [4.8183148132402435e-11, 0.2211970420118015, 4.543989969150727e-11,
           3.6490937181511005e-10, 1.2061528170317823e-10, 1.583724126560376e-10,
           0.02918541134375931, 2.647463154513871e-10, 4.044537069039929e-11,
           7.004318834052582e-11, 0.5455165551108696, 2.5394328031947243e-11,
           7.612250719463419e-11, 4.2145011656152064e-11, 1.78315731459658e-9,
           1.0024051727804193e-10, 3.6659422059826397e-10, 8.10276573583226e-11,
           4.82746139763759e-11, 0.2041009878978584]

    w3t = [6.052046916462879e-10, 0.22119705515294769, 6.424098444604305e-10,
           1.5424395136280537e-9, 1.5518248762287603e-9, 8.927755117426529e-10,
           0.029185409451782888, 8.657982940794159e-10, 5.879280367906113e-10,
           6.882917114504467e-10, 0.5455165482402493, 3.563222102560007e-10,
           5.626809098979851e-10, 4.228608998520786e-10, 7.235485263145381e-9,
           8.569826888158532e-10, 1.414724069583225e-9, 5.041990263253413e-10,
           6.135603914871631e-10, 0.20410096781153206]

    w4t = [1.2914431968018099e-11, 0.2211970403633371, 2.0358466367337004e-11,
           3.398879426663391e-10, 8.750768903808466e-11, 8.224731529585248e-11,
           0.02918541164641545, 1.3715955735214613e-10, 2.242350456488422e-11,
           4.2254445798876156e-11, 0.5455165561727514, 8.376570115102258e-12,
           4.7307262659151097e-11, 1.5452395801064293e-11, 5.081408824494903e-10,
           5.648552693030804e-11, 3.7563121626528117e-10, 1.8070940226359078e-11,
           2.6502439903891977e-11, 0.20410099001677537]

    w5t = [4.0777670249502093e-11, 0.2211970394568426, 3.999966498578144e-11,
           4.396392821801329e-10, 1.468247473902192e-10, 1.1707443220853845e-10,
           0.029185411509662387, 2.3062296403314715e-10, 3.5211521729044234e-11,
           6.566327827423724e-11, 0.545516556238355, 2.1642958075178246e-11,
           5.81386469196181e-11, 3.305228435244028e-11, 4.785901710871034e-10,
           9.686431346861199e-11, 4.648017162700928e-10, 7.04912182157167e-11,
           4.301253744281239e-11, 0.2041009904127327]

    w6t = [1.8243130837241829e-10, 0.22119704275966878, 1.9676475495716242e-10,
           5.641726005165918e-10, 6.962065005475656e-10, 2.5329822169433217e-10,
           0.02918541154016518, 2.9047486494574945e-10, 1.6870547027377955e-10,
           2.0932007348136436e-10, 0.5455165547034347, 9.865662457626997e-11,
           1.6019179444587639e-10, 1.3612545083356438e-10, 8.360816783741759e-10,
           2.83702408019619e-10, 4.981451224337243e-10, 1.7658411784243953e-10,
           1.783847965358073e-10, 0.20410098606748553]

    w7t = [6.4402403918195814e-9, 1.721322400623156e-8, 5.903014452645576e-9,
           1.0738911770044676e-8, 0.37976639184596345, 1.8519414025157515e-9,
           0.1766051757091611, 7.794574028214419e-9, 9.681568235488115e-9,
           0.04075087739644037, 0.05638209794984671, 2.0512967590664425e-9,
           1.2046311183066474e-9, 7.996033718433275e-9, 2.0054719874347095e-9,
           0.15854735629934594, 0.18794798809768362, 9.127762865027043e-9,
           2.5979627339195894e-8, 4.713260904050864e-9]

    w8t = [1.2987966435494406e-8, 1.0274074138557696e-7, 1.2809717942310205e-8,
           2.509716420823527e-8, 0.37976634492649053, 3.842660604764721e-9,
           0.17660501684308846, 2.0316117966709345e-8, 1.9725818724432325e-8,
           0.04075077162476984, 0.056382133138374785, 4.4706516442605335e-9,
           2.529304168327545e-9, 2.29187283467979e-8, 3.871552345567762e-9,
           0.15854732853406348, 0.1879480546724281, 2.1444934255450605e-8,
           8.495414608883858e-8, 1.2551280877660048e-8]

    w9t = [1.1771510068649787e-7, 4.1969207615922344e-7, 1.1167960483371123e-7,
           2.709520992005699e-7, 0.3797874980435765, 3.349027947459708e-8,
           0.17660104085294917, 1.7059278409813566e-7, 2.1712657786105016e-7,
           0.040741554472707074, 0.05637710851153043, 3.6269792712197094e-8,
           2.2517557318621907e-8, 1.4856583508347104e-7, 3.413719624294643e-8,
           0.15854981054187542, 0.18793730676485773, 2.1058454235298844e-7,
           3.79635666353943e-6, 9.113239404480115e-8]

    w10t = [2.306326363105418e-9, 2.4676826774710196e-9, 3.2173662688029383e-9,
            2.8713914044086282e-9, 4.449957006275181e-7, 1.1952192350439894e-9,
            0.9999995157416278, 1.713795716123049e-9, 2.6335252107919382e-9,
            1.8587469795978376e-9, 1.6691580852897476e-9, 1.2365889365893327e-9,
            9.563960602724664e-10, 1.4418153484765036e-9, 9.888815088419938e-10,
            4.6005991814570895e-9, 3.3948203360257317e-9, 1.791198481437484e-9,
            2.7418766922088886e-9, 2.1772831997661477e-9]

    w11t = [9.083404201338528e-12, 1.2969176812179506e-11, 2.1582413496144878e-11,
            1.7467087197925577e-11, 0.8503244546368753, 8.103672899340972e-12,
            0.14967554513252176, 1.6661614314600896e-12, 1.5768824434944537e-11,
            3.5405782731913797e-12, 8.710829422952142e-13, 6.861152189314967e-12,
            1.2958811894123454e-11, 3.760688702416905e-12, 9.223005031547282e-12,
            4.939614173717909e-11, 2.6876005397793104e-11, 3.0272626414501494e-12,
            1.848249518446453e-11, 8.964811324601124e-12]

    w12t = [2.6100943009435e-9, 2.809814476396465e-9, 3.4889916445724065e-9,
            3.138692005456562e-9, 0.8533950789889978, 1.2049749726675938e-9,
            0.14660487729706478, 1.8875976565926353e-9, 2.9958634577839283e-9,
            1.9874793758098272e-9, 1.7821139823816611e-9, 1.1529195697938039e-9,
            8.375155200622696e-10, 1.4793733255296196e-9, 8.46219785069479e-10,
            6.016445336363821e-9, 3.88326473681706e-9, 2.0041854469694113e-9,
            3.2023164187645093e-9, 2.3860755735083145e-9]

    @test isapprox(w1.weights, w1t)
    @test isapprox(w2.weights, w2t)
    @test isapprox(w3.weights, w3t)
    @test isapprox(w2.weights, w3.weights, rtol = 1e-7)
    @test isapprox(w4.weights, w4t)
    @test isapprox(w5.weights, w5t)
    @test isapprox(w6.weights, w6t)
    @test isapprox(w5.weights, w6.weights, rtol = 1e-8)
    @test isapprox(w7.weights, w7t)
    @test isapprox(w8.weights, w8t)
    @test isapprox(w9.weights, w9t, rtol = 0.0001)
    @test isapprox(w8.weights, w9.weights, rtol = 1e-4)
    @test isapprox(w10.weights, w10t)
    @test isapprox(w11.weights, w11t)
    @test isapprox(w12.weights, w12t, rtol = 1.0e-7)
    @test isapprox(w11.weights, w12.weights, rtol = 1e-2)
    @test isapprox(w13.weights, w7.weights, rtol = 1e-6)
    @test isapprox(w14.weights, w8.weights, rtol = 1e-6)
    @test isapprox(w15.weights, w9.weights, rtol = 1e-4)
    @test isapprox(w16.weights, w7.weights, rtol = 1e-6)
    @test isapprox(w17.weights, w8.weights, rtol = 1e-6)
    @test isapprox(w18.weights, w9.weights, rtol = 1e-4)
    @test isapprox(w13.weights, w16.weights, rtol = 1e-6)
    @test isapprox(w14.weights, w17.weights, rtol = 1e-6)
    @test isapprox(w15.weights, w18.weights, rtol = 1e-4)
    @test isapprox(w19.weights, w1.weights, rtol = 1e-8)
end

@testset "$(:Classic), $(:Trad), $(:CVaR)" begin
    portfolio = Portfolio(; prices = prices,
                          solvers = OrderedDict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                                  :params => Dict("verbose" => false,
                                                                                  "max_step_fraction" => 0.75))))
    asset_statistics!(portfolio)

    rm = :CVaR
    opt = OptimiseOpt(; rf = rf, l = l, class = :Classic, type = :Trad, rm = rm,
                      obj = :Min_Risk, kelly = :None)

    w1 = optimise!(portfolio, opt)
    risk1 = calc_risk(portfolio; type = :Trad, rm = rm, rf = rf)
    opt.kelly = :Approx
    w2 = optimise!(portfolio, opt)
    opt.kelly = :Exact
    w3 = optimise!(portfolio, opt)
    opt.obj = :Utility
    opt.kelly = :None
    w4 = optimise!(portfolio, opt)
    opt.kelly = :Approx
    w5 = optimise!(portfolio, opt)
    opt.kelly = :Exact
    w6 = optimise!(portfolio, opt)
    opt.obj = :Sharpe
    opt.kelly = :None
    w7 = optimise!(portfolio, opt)
    risk7 = calc_risk(portfolio; type = :Trad, rm = rm, rf = rf)
    ret7 = dot(portfolio.mu, w7.weights)
    opt.kelly = :Approx
    w8 = optimise!(portfolio, opt)
    risk8 = calc_risk(portfolio; type = :Trad, rm = rm, rf = rf)
    ret8 = dot(portfolio.mu, w8.weights)
    opt.kelly = :Exact
    w9 = optimise!(portfolio, opt)
    risk9 = calc_risk(portfolio; type = :Trad, rm = rm, rf = rf)
    ret9 = dot(portfolio.mu, w9.weights)
    opt.obj = :Max_Ret
    opt.kelly = :None
    w10 = optimise!(portfolio, opt)
    opt.kelly = :Approx
    w11 = optimise!(portfolio, opt)
    opt.kelly = :Exact
    w12 = optimise!(portfolio, opt)
    setproperty!(portfolio, :mu_l, ret7)
    opt.obj = :Min_Risk
    opt.kelly = :None
    w13 = optimise!(portfolio, opt)
    setproperty!(portfolio, :mu_l, ret8)
    w14 = optimise!(portfolio, opt)
    setproperty!(portfolio, :mu_l, ret9)
    w15 = optimise!(portfolio, opt)
    setproperty!(portfolio, :mu_l, Inf)
    rmf = Symbol(lowercase(string(rm)) * "_u")
    setproperty!(portfolio, rmf, risk7)
    opt.obj = :Max_Ret
    opt.kelly = :None
    w16 = optimise!(portfolio, opt)
    setproperty!(portfolio, rmf, risk8)
    w17 = optimise!(portfolio, opt)
    setproperty!(portfolio, rmf, risk9)
    w18 = optimise!(portfolio, opt)
    setproperty!(portfolio, rmf, risk1)
    opt.obj = :Sharpe
    w19 = optimise!(portfolio, opt)
    setproperty!(portfolio, rmf, Inf)

    w1t = [1.0917910452416447e-10, 0.04242032550545622, 1.1117901206969251e-10,
           2.4088084808091204e-10, 0.007574028854500408, 1.1946344076887745e-10,
           2.1177436868338872e-11, 0.09464947742115848, 5.658892712214821e-11,
           7.479870941291559e-11, 0.30401106305449943, 6.734804514187451e-11,
           3.5729381580507716e-11, 0.0656416661485033, 8.008722037727663e-11,
           0.029371611741159806, 1.4242578458283341e-10, 0.36631011582253764,
           7.299643607410934e-11, 0.09002171032033043]

    w2t = [1.7024952644903223e-10, 0.042420322488037975, 1.730973072010344e-10,
           3.773046294100106e-10, 0.0075740296009401305, 1.8617526668454714e-10,
           3.3319428280617314e-11, 0.09464947666059295, 8.360172861235745e-11,
           1.170003901630541e-10, 0.30401105825340946, 1.0137749899224049e-10,
           5.3842629307320194e-11, 0.0656416636435107, 1.3308402198855074e-10,
           0.029371612102473556, 2.2702477730232083e-10, 0.3663101182954551,
           1.0973693661316229e-10, 0.09002171718976597]

    w3t = [1.0879273278110467e-9, 0.04242028020004519, 1.145162515769511e-9,
           2.4737251584714044e-9, 0.0075740379783315916, 1.1069043025409862e-9,
           4.60107581863361e-10, 0.09464946929287248, 6.606987853325169e-10,
           7.768598533839002e-10, 0.30401102086106024, 6.872797392158463e-10,
           4.1660419049643057e-10, 0.06564163010594888, 8.824381971730573e-10,
           0.029371615803228158, 1.4870880967094485e-9, 0.36631014260213807,
           7.785118576844406e-10, 0.09002179119306775]

    w4t = [2.698019055400323e-10, 0.03635924479049503, 2.677619444743686e-10,
           7.985214641068182e-10, 0.017177306575757743, 2.416569816374551e-10,
           3.634867658227212e-11, 0.091642549689489, 1.3594890043427558e-10,
           1.6096545847137964e-10, 0.3225840157899133, 1.2112029321153645e-10,
           5.532800251851602e-11, 0.039550826065936406, 2.2692286888074573e-10,
           0.03091937779637299, 3.571613328255802e-10, 0.37332986901370324,
           1.8040087599469393e-10, 0.0884368074263937]

    w5t = [1.071858185401761e-10, 0.036359191039169104, 1.0528306788057081e-10,
           3.2767565668907875e-10, 0.01717733720636712, 9.94981355651191e-11,
           1.3805650656124311e-11, 0.0916425620279538, 5.3024957380642035e-11,
           6.468917879575973e-11, 0.3225841785565634, 4.960040427129864e-11,
           2.4514555449223887e-11, 0.03955075197222998, 9.427868038640871e-11,
           0.03091937764467206, 1.3322446515063434e-10, 0.3733298493019755,
           6.987184976717326e-11, 0.08843675110841663]

    w6t = [4.699789610198439e-10, 0.036359251819189475, 4.756764922244709e-10,
           1.324613592470362e-9, 0.01717731850441682, 4.0318522009740097e-10,
           1.5681714840975024e-10, 0.09164260171009447, 2.765343432591495e-10,
           3.0138179120283394e-10, 0.3225838494903846, 2.3145827906444672e-10,
           1.3594943144943183e-10, 0.03955085831889792, 3.980162483956832e-10,
           0.030919373991586525, 6.370396129096171e-10, 0.37332986246897387,
           3.4229434952125315e-10, 0.08843687854351082]

    w7t = [3.802075015666319e-9, 5.1805719364096775e-9, 5.859219087411186e-9,
           3.4073919251831195e-9, 0.5628459293676846, 1.1010179911468395e-9,
           0.0443421614793685, 8.510617458294086e-9, 5.387394765197944e-9,
           2.685248329088635e-9, 7.195766531356565e-9, 1.0156576904930125e-9,
           5.80192606857298e-10, 2.1103859927301306e-9, 6.308370560763638e-10,
           0.20959188864184677, 0.1832199367673754, 9.328546020721074e-9,
           1.9900800546721317e-8, 7.0480018578192434e-9]

    w8t = [1.5090749848603494e-8, 2.3395642242860216e-8, 2.3125545411750963e-8,
           1.5151043915823873e-8, 0.5593119505801409, 4.611638970033272e-9,
           0.02947532536975665, 2.52065213326967e-7, 1.928953016450107e-8,
           1.0615158689174018e-8, 5.5479980569419174e-8, 4.107293456123312e-9,
           2.7538852170268156e-9, 8.69532580104164e-9, 2.8480376395749858e-9,
           0.2029695637876332, 0.20824236515528147, 1.6079151046145378e-7,
           1.5413778421770426e-7, 4.29488479449105e-8]

    w9t = [8.981202705142185e-8, 1.270266843603467e-7, 1.4329594312565794e-7,
           8.313206011341467e-8, 0.5622802604361524, 2.905655153015814e-8,
           0.041914230984294636, 2.3565534693115868e-7, 1.1954721880205166e-7,
           6.554555496394908e-8, 2.1550685857022667e-7, 2.655146004917418e-8,
           1.7729401946328358e-8, 5.239972268339452e-8, 1.831338752381326e-8,
           0.20850172287099394, 0.18730161219541538, 2.621746508246125e-7,
           5.172570409158214e-7, 1.7050923421811878e-7]

    w10t = [2.8480425931663927e-8, 3.0415809102700086e-8, 3.957871009075327e-8,
            3.530723614800688e-8, 5.612848741145925e-6, 1.4866888335537356e-8,
            0.9999939028339723, 2.1221606397558113e-8, 3.237130075404036e-8,
            2.2989297822974406e-8, 2.0673160934084982e-8, 1.5436331740229477e-8,
            1.2008806117094413e-8, 1.7891709647828548e-8, 1.2378363277772456e-8,
            5.618914773297307e-8, 4.1723324578723495e-8, 2.2165377789582972e-8,
            3.3718500949974336e-8, 2.6901289289519924e-8]

    w11t = [3.495905040148807e-11, 5.032616430793468e-11, 8.474079906140368e-11,
            6.809405076476044e-11, 0.8503243449205248, 3.303173578227459e-11,
            0.14967565417625428, 5.262530509452157e-12, 6.119123223034577e-11,
            1.250039664061075e-11, 2.263678250159422e-12, 2.806243270267671e-11,
            5.283354244939076e-11, 1.5932081537413547e-11, 3.4841410726571836e-11,
            1.9605987332184607e-10, 1.0592537183769043e-10, 1.0744589483472177e-11,
            7.219024092675466e-11, 3.4261723153174966e-11]

    w12t = [2.5735820108753144e-9, 2.7700094042055317e-9, 3.439012204627773e-9,
            3.0938675761521603e-9, 0.8533950721450337, 1.1882265690550022e-9,
            0.14660488476436898, 1.8610871769928931e-9, 2.9530323234192727e-9,
            1.959276311303801e-9, 1.7569429941886365e-9, 1.1367381576802053e-9,
            8.258470037692067e-10, 1.4584966588720545e-9, 8.342516258680642e-10,
            5.928475813214658e-9, 3.82729705632758e-9, 1.97590070809681e-9,
            3.1566277327108637e-9, 2.3519257212828204e-9]

    @test isapprox(w1.weights, w1t)
    @test isapprox(w2.weights, w2t)
    @test isapprox(w3.weights, w3t)
    @test isapprox(w2.weights, w3.weights, rtol = 1e-6)
    @test isapprox(w4.weights, w4t)
    @test isapprox(w5.weights, w5t)
    @test isapprox(w6.weights, w6t, rtol = 1.0e-7)
    @test isapprox(w5.weights, w6.weights, rtol = 1e-6)
    @test isapprox(w7.weights, w7t)
    @test isapprox(w8.weights, w8t)
    @test isapprox(w9.weights, w9t)
    @test isapprox(w8.weights, w9.weights, rtol = 1e-1)
    @test isapprox(w10.weights, w10t)
    @test isapprox(w11.weights, w11t)
    @test isapprox(w12.weights, w12t, rtol = 1e-7)
    @test isapprox(w11.weights, w12.weights, rtol = 1e-2)
    @test isapprox(w13.weights, w7.weights, rtol = 1e-6)
    @test isapprox(w14.weights, w8.weights, rtol = 1e-5)
    @test isapprox(w15.weights, w9.weights, rtol = 1e-4)
    @test isapprox(w16.weights, w7.weights, rtol = 1e-6)
    @test isapprox(w17.weights, w8.weights, rtol = 1e-5)
    @test isapprox(w18.weights, w9.weights, rtol = 1e-4)
    @test isapprox(w13.weights, w16.weights, rtol = 1e-6)
    @test isapprox(w14.weights, w17.weights, rtol = 1e-6)
    @test isapprox(w15.weights, w18.weights, rtol = 1e-5)
    @test isapprox(w19.weights, w1.weights, rtol = 1e-5)
end

@testset "$(:Classic), $(:Trad), $(:EVaR)" begin
    portfolio = Portfolio(; prices = prices,
                          solvers = OrderedDict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                                  :params => Dict("verbose" => false,
                                                                                  "max_step_fraction" => 0.75))))
    asset_statistics!(portfolio)

    rm = :EVaR
    opt = OptimiseOpt(; rf = rf, l = l, class = :Classic, type = :Trad, rm = rm,
                      obj = :Min_Risk, kelly = :None)

    w1 = optimise!(portfolio, opt)
    risk1 = calc_risk(portfolio; type = :Trad, rm = rm, rf = rf)
    opt.kelly = :Approx
    w2 = optimise!(portfolio, opt)
    opt.kelly = :Exact
    w3 = optimise!(portfolio, opt)
    opt.obj = :Utility
    opt.kelly = :None
    w4 = optimise!(portfolio, opt)
    opt.kelly = :Approx
    w5 = optimise!(portfolio, opt)
    opt.kelly = :Exact
    w6 = optimise!(portfolio, opt)
    opt.obj = :Sharpe
    opt.kelly = :None
    w7 = optimise!(portfolio, opt)
    risk7 = calc_risk(portfolio; type = :Trad, rm = rm, rf = rf)
    ret7 = dot(portfolio.mu, w7.weights)
    opt.kelly = :Approx
    w8 = optimise!(portfolio, opt)
    risk8 = calc_risk(portfolio; type = :Trad, rm = rm, rf = rf)
    ret8 = dot(portfolio.mu, w8.weights)
    opt.kelly = :Exact
    w9 = optimise!(portfolio, opt)
    risk9 = calc_risk(portfolio; type = :Trad, rm = rm, rf = rf)
    ret9 = dot(portfolio.mu, w9.weights)
    opt.obj = :Max_Ret
    opt.kelly = :None
    w10 = optimise!(portfolio, opt)
    opt.kelly = :Approx
    w11 = optimise!(portfolio, opt)
    opt.kelly = :Exact
    w12 = optimise!(portfolio, opt)
    setproperty!(portfolio, :mu_l, ret7)
    opt.obj = :Min_Risk
    opt.kelly = :None
    w13 = optimise!(portfolio, opt)
    setproperty!(portfolio, :mu_l, ret8)
    w14 = optimise!(portfolio, opt)
    setproperty!(portfolio, :mu_l, ret9)
    w15 = optimise!(portfolio, opt)
    setproperty!(portfolio, :mu_l, Inf)
    rmf = Symbol(lowercase(string(rm)) * "_u")
    setproperty!(portfolio, rmf, risk7)
    opt.obj = :Max_Ret
    opt.kelly = :None
    w16 = optimise!(portfolio, opt)
    setproperty!(portfolio, rmf, risk8)
    w17 = optimise!(portfolio, opt)
    setproperty!(portfolio, rmf, risk9)
    w18 = optimise!(portfolio, opt)
    setproperty!(portfolio, rmf, risk1)
    opt.obj = :Sharpe
    w19 = optimise!(portfolio, opt)
    setproperty!(portfolio, rmf, Inf)

    w1t = [1.215760277810166e-8, 0.15679037218654726, 1.0089401815603605e-8,
           0.0167098231782758, 1.4503329606354254e-8, 5.9299561320595284e-8,
           0.014451125278323807, 0.15570654410659587, 9.321254055304116e-9,
           1.0323796429819983e-8, 0.45244932118334397, 8.15896746119881e-9,
           1.1858350511875477e-8, 2.2968134857700408e-8, 0.004793473928182738,
           1.6801773705866435e-7, 0.018418699599600505, 1.8659794160143353e-7,
           1.1906313135012431e-8, 0.18068011533673956]

    w2t = [1.3006571440000197e-8, 0.1567891709789269, 1.0800524712031763e-8,
           0.01670838352427139, 1.532774250301635e-8, 6.339013646477005e-8,
           0.01445122563587082, 0.15570695555217035, 9.939156743847192e-9,
           1.1020427965087136e-8, 0.4524484672852396, 8.641078548344099e-9,
           1.2503731931382423e-8, 2.48161692263119e-8, 0.004793465082527265,
           1.8069125609126623e-7, 0.018422134979702607, 1.9858874691831332e-7,
           1.2732862618003511e-8, 0.18067963550288602]
    w3t = [1.2333347938018269e-9, 0.1567938764578563, 1.1807339961271223e-9,
           0.01670531985998916, 2.920983295648933e-9, 2.3725739174696647e-9,
           0.01445427066945339, 0.15570561296431598, 9.264147840540523e-10,
           1.0094571083720776e-9, 0.4524496895695898, 6.0114071216114e-10,
           5.949374172933385e-10, 1.8539785685453273e-9, 0.004792796287046998,
           6.581473879085626e-9, 0.018416917961844134, 1.015068386777272e-8,
           1.2414495809404807e-9, 0.18068148556274233]

    w4t = [2.5755276351943506e-8, 0.1513893221334751, 2.2340700410417753e-8,
           0.02042365036145328, 3.84905402032807e-8, 7.09983960119684e-8,
           0.017220660868011294, 0.1532395020116679, 2.050531639004377e-8,
           2.2585890935578955e-8, 0.44644574066104875, 1.6266145662320583e-8,
           2.1009975727630527e-8, 4.2153800864230896e-8, 2.5882137580405836e-7,
           1.0921630572169067e-6, 0.027628201728423105, 3.021740989454302e-7,
           2.6623620093004507e-8, 0.18365096234772604]

    w5t = [4.654464936575532e-9, 0.15110573700451743, 3.932894882289678e-9,
           0.020589950386288525, 7.025817936470814e-9, 1.3167381206329623e-8,
           0.017130896290434905, 0.1532465509633357, 3.441103256516102e-9,
           3.9733423346730945e-9, 0.4464282781413881, 2.6815700352058374e-9,
           3.511331342474142e-9, 8.183626557234464e-9, 4.391947260276608e-8,
           2.5098249936721764e-7, 0.02787023932756787, 5.475946905534737e-8,
           4.635737701825992e-9, 0.18362794301775628]

    w6t = [3.5810013562008097e-10, 0.1511121133102664, 3.5807053893297125e-10,
           0.020587272036606468, 1.3069704913963028e-9, 5.001885560965721e-10,
           0.017132440414091295, 0.15325009811352616, 3.018150551633827e-10,
           2.93004705255767e-10, 0.44642681211480706, 1.7042613018387576e-10,
           1.7194192596473736e-10, 3.649869117750076e-10, 1.7039208285029884e-9,
           7.607002922841725e-9, 0.027855389576678508, 1.7211908314388229e-9,
           3.8039286184652447e-10, 0.18363585919601216]

    w7t = [1.0750813604735852e-8, 3.269509079351244e-8, 1.1161958355618464e-8,
           1.3795474360886804e-8, 0.5351877377745937, 2.6718281126769987e-9,
           0.13907652548471258, 1.4128484886659633e-8, 1.0656086916773938e-8,
           7.837197682601661e-9, 1.7948234909357196e-8, 2.62293931824833e-9,
           1.8308426969298138e-9, 6.011261123705567e-9, 1.9381737009512553e-9,
           0.18358587252263112, 0.14214966911600826, 1.7345006490563085e-8,
           3.3941485986849895e-8, 9.767175298889203e-9]

    w8t = [1.4784428555700969e-8, 9.956294021978918e-8, 1.426966573613237e-8,
           1.9583346288557528e-8, 0.4887150416179051, 3.477056999890348e-9,
           0.11042415725395371, 4.7358963919519425e-8, 1.4103948150060762e-8,
           1.2046964205776984e-8, 9.03784021263289e-8, 3.2896343254856622e-9,
           2.2226992140983426e-9, 9.03464776782426e-9, 2.475275002468014e-9,
           0.1801136994525949, 0.22074663672331457, 4.1305456873074073e-8,
           7.693192994148603e-8, 1.4126872365982037e-8]

    w9t = [1.0208856361119273e-7, 4.6073968191462595e-7, 1.1449238329739809e-7,
           1.384675682321963e-7, 0.5138191268604309, 2.4127564708797755e-8,
           0.11865622568608232, 1.888084872199739e-7, 1.0706031936085872e-7,
           7.479265759573994e-8, 2.5041281021401765e-7, 2.294716268102065e-8,
           1.5859365384147223e-8, 5.6899366409145524e-8, 1.6738237988771136e-8,
           0.18152113968094294, 0.18600099056631475, 2.2055552253345187e-7,
           6.206801628722678e-7, 1.0253637518149154e-7]

    w10t = [3.4346718528402118e-9, 3.6540927977462333e-9, 4.579123920270781e-9,
            4.1662154436625805e-9, 2.721231234070582e-7, 1.6174815174245242e-9,
            0.999999671895358, 2.5209180162301834e-9, 3.865148615759323e-9,
            2.7624958231731586e-9, 2.444874971510488e-9, 1.7123982681270434e-9,
            1.2414618683892697e-9, 2.05347831671404e-9, 1.2812810374245841e-9,
            5.964529411215915e-9, 4.777212426602421e-9, 2.6491436225104996e-9,
            4.006566441929497e-9, 3.2504242770109577e-9]

    w11t = [3.714811036268255e-11, 4.7738975632490153e-11, 7.269096260019431e-11,
            6.026891169308541e-11, 0.8503244596483465, 1.3212536057062485e-11,
            0.14967553959925534, 1.4434861848996695e-11, 5.6011650759493475e-11,
            2.0563685677384248e-11, 1.1468884081351451e-11, 1.2239337067734528e-11,
            2.582237766818944e-11, 1.5505683335796168e-12, 2.4392585368992762e-11,
            1.4993618728562185e-10, 8.733238904362575e-11, 1.822927803678296e-11,
            6.353214849610202e-11, 3.5824961937731867e-11]

    w12t = [2.6591686415143532e-9, 2.8618811807331272e-9, 3.5533392772841433e-9,
            3.1966028069895103e-9, 0.8533950748072571, 1.2278970852183128e-9,
            0.14660488067055402, 1.92255421810582e-9, 3.0509850549622057e-9,
            2.0239132351816554e-9, 1.8148909878169361e-9, 1.174360803965346e-9,
            8.531646834056681e-10, 1.5068285343817963e-9, 8.619332548777916e-10,
            6.127919381209295e-9, 3.954583600570231e-9, 2.0412911983481386e-9,
            3.2612382379377608e-9, 2.429636796346731e-9]

    @test isapprox(w1.weights, w1t, rtol = 1e-4)
    @test isapprox(w2.weights, w2t, rtol = 1e-4)
    @test isapprox(w3.weights, w3t, rtol = 1e-5)
    @test isapprox(w2.weights, w3.weights, rtol = 1e-4)
    @test isapprox(w4.weights, w4t, rtol = 1e-5)
    @test isapprox(w5.weights, w5t, rtol = 1e-4)
    @test isapprox(w6.weights, w6t, rtol = 1e-5)
    @test isapprox(w5.weights, w6.weights, rtol = 1e-4)
    @test isapprox(w7.weights, w7t, rtol = 1e-6)
    @test isapprox(w8.weights, w8t, rtol = 1e-5)
    @test isapprox(w9.weights, w9t, rtol = 1e-4)
    @test isapprox(w8.weights, w9.weights, rtol = 1e-1)
    @test isapprox(w10.weights, w10t)
    @test isapprox(w11.weights, w11t)
    @test isapprox(w12.weights, w12t, rtol = 1e-7)
    @test isapprox(w11.weights, w12.weights, rtol = 1e-2)
    @test isapprox(w13.weights, w7.weights, rtol = 1e-4)
    @test isapprox(w14.weights, w8.weights, rtol = 1e-1)
    @test isapprox(w15.weights, w9.weights, rtol = 1e-1)
    @test isapprox(w16.weights, w7.weights, rtol = 1e-4)
    @test isapprox(w17.weights, w8.weights, rtol = 1e-1)
    @test isapprox(w18.weights, w9.weights, rtol = 1e-1)
    @test isapprox(w13.weights, w16.weights, rtol = 1e-4)
    @test isapprox(w14.weights, w17.weights, rtol = 1e-2)
    @test isapprox(w15.weights, w18.weights, rtol = 1e-2)
    @test isapprox(w19.weights, w1.weights, rtol = 1e0)
end

@testset "$(:Classic), $(:Trad), $(:RVaR)" begin
    portfolio = Portfolio(; prices = prices,
                          solvers = OrderedDict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                                  :params => Dict("verbose" => false,
                                                                                  "max_step_fraction" => 0.65))))
    asset_statistics!(portfolio)

    rm = :RVaR
    opt = OptimiseOpt(; rf = rf, l = l, class = :Classic, type = :Trad, rm = rm,
                      obj = :Min_Risk, kelly = :None)

    w1 = optimise!(portfolio, opt)
    risk1 = calc_risk(portfolio; type = :Trad, rm = rm, rf = rf)
    opt.kelly = :Approx
    w2 = optimise!(portfolio, opt)
    opt.kelly = :Exact
    w3 = optimise!(portfolio, opt)
    opt.obj = :Utility
    opt.kelly = :None
    w4 = optimise!(portfolio, opt)
    opt.kelly = :Approx
    w5 = optimise!(portfolio, opt)
    opt.kelly = :Exact
    w6 = optimise!(portfolio, opt)
    opt.obj = :Sharpe
    opt.kelly = :None
    w7 = optimise!(portfolio, opt)
    risk7 = calc_risk(portfolio; type = :Trad, rm = rm, rf = rf)
    ret7 = dot(portfolio.mu, w7.weights)
    opt.kelly = :Approx
    w8 = optimise!(portfolio, opt)
    risk8 = calc_risk(portfolio; type = :Trad, rm = rm, rf = rf)
    ret8 = dot(portfolio.mu, w8.weights)
    opt.kelly = :Exact
    w9 = optimise!(portfolio, opt)
    risk9 = calc_risk(portfolio; type = :Trad, rm = rm, rf = rf)
    ret9 = dot(portfolio.mu, w9.weights)
    opt.obj = :Max_Ret
    opt.kelly = :None
    w10 = optimise!(portfolio, opt)
    opt.kelly = :Approx
    w11 = optimise!(portfolio, opt)
    opt.kelly = :Exact
    w12 = optimise!(portfolio, opt)
    setproperty!(portfolio, :mu_l, ret7)
    opt.obj = :Min_Risk
    opt.kelly = :None
    w13 = optimise!(portfolio, opt)
    setproperty!(portfolio, :mu_l, ret8)
    w14 = optimise!(portfolio, opt)
    setproperty!(portfolio, :mu_l, ret9)
    w15 = optimise!(portfolio, opt)
    setproperty!(portfolio, :mu_l, Inf)
    rmf = Symbol(lowercase(string(rm)) * "_u")
    setproperty!(portfolio, rmf, risk7)
    opt.obj = :Max_Ret
    opt.kelly = :None
    w16 = optimise!(portfolio, opt)
    setproperty!(portfolio, rmf, risk8)
    w17 = optimise!(portfolio, opt)
    setproperty!(portfolio, rmf, risk9)
    w18 = optimise!(portfolio, opt)
    setproperty!(portfolio, rmf, risk1 + 1e-6 * risk1)
    opt.obj = :Sharpe
    w19 = optimise!(portfolio, opt)
    setproperty!(portfolio, rmf, Inf)

    w1t = [3.6383870609065e-9, 0.21104601990219268, 3.786816455647481e-9,
           2.100015622831307e-8, 1.0251709183992321e-8, 5.8717062605485755e-9,
           0.036670319635292575, 0.06104294164051898, 2.955951610001794e-9,
           3.2263214734567926e-9, 0.4935354622595077, 1.5929895760319034e-9,
           3.134723198998334e-9, 3.344775056827488e-9, 8.60298097599189e-9,
           8.750990940638024e-9, 1.9889845649056345e-8, 1.0109583835098379e-8,
           3.8869186924010954e-9, 0.19770514651863197]

    w2t = [2.9423292942390505e-9, 0.21104675510164206, 3.188885854062719e-9,
           1.9034585108307795e-8, 8.261419735635841e-9, 5.274542800185119e-9,
           0.036670346928692824, 0.061044106051299146, 2.4722458553624584e-9,
           2.6586336491614765e-9, 0.49353498217160474, 1.3597681116677926e-9,
           2.8394130334764704e-9, 2.8490968199552002e-9, 7.92510930295889e-9,
           7.255070417123828e-9, 1.739866563559235e-8, 7.497638449810587e-9,
           3.110205943022469e-9, 0.19770371567915115]

    w3t = [2.877260077060643e-10, 0.2110466940428392, 2.9587719846653324e-10,
           1.9693632430753494e-9, 9.584829843060796e-10, 5.364300640140843e-10,
           0.03667019406031719, 0.06104535523814454, 2.3244331140901786e-10,
           2.5296582287142204e-10, 0.4935344576777325, 1.3343595743420152e-10,
           2.9207189538429934e-10, 2.4060370134094297e-10, 8.085827866358879e-10,
           7.550151113153446e-10, 1.7595953181592781e-9, 7.665926429801792e-10,
           3.036925206217448e-10, 0.19770328938808793]

    w4t = [3.4469321749778263e-9, 0.2129248557061676, 3.473320124721614e-9,
           7.817037066367329e-8, 9.354474626147698e-9, 9.268240982013965e-9,
           0.0394228377015491, 0.05653482340878542, 2.8687710235954115e-9,
           3.4413257053792345e-9, 0.49014410147547605, 1.9576286507358135e-9,
           5.207571911585772e-9, 3.07242280612367e-9, 1.2418365446646256e-8,
           1.3357449889942468e-8, 4.8691411070442944e-8, 6.392890554739148e-9,
           3.601279691432997e-9, 0.2009731769855666]

    w5t = [1.5573500076701032e-9, 0.2128146235318881, 1.5683172884292565e-9,
           2.885317624357063e-8, 3.516241824259499e-9, 3.5055544812995142e-9,
           0.03930346892441097, 0.0568702770689884, 1.2551505141230292e-9,
           1.4949153527071272e-9, 0.49017591995431586, 8.227836974448982e-10,
           1.9433771468168774e-9, 1.4567237690151316e-9, 4.487399870673571e-9,
           5.200596170043698e-9, 1.8387306663046683e-8, 3.3804978701684465e-9,
           1.5972262715921246e-9, 0.2008356314937795]

    w6t = [1.0619909548587346e-10, 0.21281367841974347, 1.1524569485795181e-10,
           1.4589107281956207e-9, 5.463309730466734e-10, 1.8720102058164852e-10,
           0.03930179796772753, 0.05687387606454392, 9.078063490377604e-11,
           9.719793743390038e-11, 0.49017728815700234, 5.026158910520679e-11,
           1.0576317745751681e-10, 8.670334843394393e-11, 2.3104682351230082e-10,
           3.7630700393057046e-10, 1.085408467200128e-9, 2.243756066901045e-10,
           1.2948713630714168e-10, 0.2008333544997635]

    w7t = [1.1068806347501084e-8, 3.193376771607641e-8, 8.10831099595818e-9,
           1.606592240287098e-8, 0.5059935387642095, 2.707772481392125e-9,
           0.17234038010079739, 2.297512913474259e-8, 1.388414078370297e-8,
           4.145747074673398e-8, 2.2611924053804935e-6, 2.7150345620033065e-9,
           1.822732557306626e-9, 1.0768577287274038e-8, 2.670267621659695e-9,
           0.18247679771830816, 0.13918662761269435, 2.8233634640202082e-8,
           1.9371339157063408e-7, 6.486626402503413e-9]

    w8t = [1.227867193702248e-8, 8.151041294909433e-8, 9.395565375106247e-9,
           1.918313547773279e-8, 0.45025074050762437, 3.012202768004055e-9,
           0.14690703630783494, 3.2792492415857553e-8, 1.3244759272266374e-8,
           2.6853315565966835e-8, 0.05126907412087959, 3.028325226247328e-9,
           1.8988812283510856e-9, 1.135574003388173e-8, 2.6509078911470524e-9,
           0.16956898343582735, 0.18200384658816218, 2.7150707575397735e-8,
           6.587388922758553e-8, 8.810664659576033e-9]

    w9t = [1.0848651712837185e-7, 4.0564072103309447e-7, 8.356439525875819e-8,
           1.6179445045656012e-7, 0.48151244459008047, 2.6259951881436033e-8,
           0.1503990137440082, 2.790596930210792e-7, 1.3301310467480588e-7,
           2.500866424281245e-7, 0.022514976922673888, 2.541232713642439e-8,
           1.6897001090232652e-8, 9.385458044772982e-8, 2.3077915826192684e-8,
           0.1773873110733191, 0.16818168351573232, 2.863287507833691e-7,
           2.6083024418669695e-6, 6.837569297751946e-8]

    w10t = [7.666597750828601e-8, 8.157127843442902e-8, 1.0184363896361215e-7,
            9.28815377441142e-8, 4.2401613461599374e-6, 3.5120028228998205e-8,
            0.999994519686426, 5.592681189724083e-8, 8.621532034254276e-8,
            6.144017201555716e-8, 5.418521078022546e-8, 3.7271760553546944e-8,
            2.655511315188288e-8, 4.5150576157442046e-8, 2.7465285473116317e-8,
            1.3099365177974647e-7, 1.0611889537495103e-7, 5.886343636622202e-8,
            8.935059022872039e-8, 7.25329428928066e-8]

    w11t = [1.710642967692236e-10, 2.2139640415498829e-10, 3.3824726531052144e-10,
            2.796452055666223e-10, 0.8503243950858105, 6.977197041461292e-11,
            0.1496756013843621, 6.729061148694712e-11, 2.60676155944983e-10,
            9.49689967499947e-11, 5.398374022397536e-11, 6.296727146583125e-11,
            1.21024812264786e-10, 8.66565833089963e-12, 1.1851981219546946e-10,
            7.05998247243594e-10, 4.0805729162716726e-10, 8.461555643373638e-11,
            2.9667761792323574e-10, 1.6625645816129323e-10]

    w12t = [2.316484490250889e-9, 2.5532772408139914e-9, 3.182385662543267e-9,
            2.8626131643053575e-9, 0.8533950907874841, 1.0178141110232858e-9,
            0.1466048696858697, 1.718349149341377e-9, 2.7462620697750734e-9,
            1.8426000367001627e-9, 1.6404986875871444e-9, 1.0282577751076847e-9,
            7.327367338384384e-10, 1.3232995395132197e-9, 7.451576702379923e-10,
            5.277799251075205e-9, 3.553936067639395e-9, 1.8182087275622266e-9,
            2.944559715112548e-9, 2.222406038190457e-9]

    @test isapprox(w1.weights, w1t, rtol = 1e-5)
    @test isapprox(w2.weights, w2t, rtol = 1e-6)
    @test isapprox(w3.weights, w3t, rtol = 1e-7)
    @test isapprox(w2.weights, w3.weights, rtol = 1e-5)
    @test isapprox(w4.weights, w4t, rtol = 1e-6)
    @test isapprox(w5.weights, w5t, rtol = 1e-7)
    @test isapprox(w6.weights, w6t, rtol = 1e-6)
    @test isapprox(w5.weights, w6.weights, rtol = 1e-5)
    @test isapprox(w7.weights, w7t, rtol = 1.0e-7)
    @test isapprox(w8.weights, w8t, rtol = 1e-4)
    @test isapprox(w9.weights, w9t, rtol = 1e-4)
    @test isapprox(w8.weights, w9.weights, rtol = 1e-1)
    @test isapprox(w10.weights, w10t)
    @test isapprox(w11.weights, w11t)
    @test isapprox(w12.weights, w12t, rtol = 1e-6)
    @test isapprox(w11.weights, w12.weights, rtol = 1e-2)
    @test isapprox(w13.weights, w7.weights, rtol = 1e-5)
    @test isapprox(w14.weights, w8.weights, rtol = 1e-1)
    @test isapprox(w15.weights, w9.weights, rtol = 1e-1)
    @test isapprox(w16.weights, w7.weights, rtol = 1e-4)
    @test isapprox(w17.weights, w8.weights, rtol = 1e-1)
    @test isapprox(w18.weights, w9.weights, rtol = 1e-1)
    @test isapprox(w13.weights, w16.weights, rtol = 1e-5)
    @test isapprox(w14.weights, w17.weights, rtol = 1e-2)
    @test isapprox(w15.weights, w18.weights, rtol = 1e-2)
    @test isapprox(w19.weights, w1.weights, rtol = 1e-2)
end

@testset "$(:Classic), $(:Trad), $(:MDD)" begin
    portfolio = Portfolio(; prices = prices,
                          solvers = OrderedDict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                                  :params => Dict("verbose" => false,
                                                                                  "max_step_fraction" => 0.75))))
    asset_statistics!(portfolio)

    rm = :MDD
    opt = OptimiseOpt(; rf = rf, l = l, class = :Classic, type = :Trad, rm = rm,
                      obj = :Min_Risk, kelly = :None)

    w1 = optimise!(portfolio, opt)
    risk1 = calc_risk(portfolio; type = :Trad, rm = rm, rf = rf)
    opt.kelly = :Approx
    w2 = optimise!(portfolio, opt)
    opt.kelly = :Exact
    w3 = optimise!(portfolio, opt)
    opt.obj = :Utility
    opt.kelly = :None
    w4 = optimise!(portfolio, opt)
    opt.kelly = :Approx
    w5 = optimise!(portfolio, opt)
    opt.kelly = :Exact
    w6 = optimise!(portfolio, opt)
    opt.obj = :Sharpe
    opt.kelly = :None
    w7 = optimise!(portfolio, opt)
    risk7 = calc_risk(portfolio; type = :Trad, rm = rm, rf = rf)
    ret7 = dot(portfolio.mu, w7.weights)
    opt.kelly = :Approx
    w8 = optimise!(portfolio, opt)
    risk8 = calc_risk(portfolio; type = :Trad, rm = rm, rf = rf)
    ret8 = dot(portfolio.mu, w8.weights)
    opt.kelly = :Exact
    w9 = optimise!(portfolio, opt)
    risk9 = calc_risk(portfolio; type = :Trad, rm = rm, rf = rf)
    ret9 = dot(portfolio.mu, w9.weights)
    opt.obj = :Max_Ret
    opt.kelly = :None
    w10 = optimise!(portfolio, opt)
    opt.kelly = :Approx
    w11 = optimise!(portfolio, opt)
    opt.kelly = :Exact
    w12 = optimise!(portfolio, opt)
    setproperty!(portfolio, :mu_l, ret7)
    opt.obj = :Min_Risk
    opt.kelly = :None
    w13 = optimise!(portfolio, opt)
    setproperty!(portfolio, :mu_l, ret8)
    w14 = optimise!(portfolio, opt)
    setproperty!(portfolio, :mu_l, ret9)
    w15 = optimise!(portfolio, opt)
    setproperty!(portfolio, :mu_l, Inf)
    rmf = Symbol(lowercase(string(rm)) * "_u")
    setproperty!(portfolio, rmf, risk7)
    opt.obj = :Max_Ret
    opt.kelly = :None
    w16 = optimise!(portfolio, opt)
    setproperty!(portfolio, rmf, risk8)
    w17 = optimise!(portfolio, opt)
    setproperty!(portfolio, rmf, risk9)
    w18 = optimise!(portfolio, opt)
    setproperty!(portfolio, rmf, risk1)
    opt.obj = :Sharpe
    w19 = optimise!(portfolio, opt)
    setproperty!(portfolio, rmf, Inf)

    w1t = [0.0377381016811754, 1.1200569878579588e-11, 2.5346081903301402e-11,
           3.3892860925846467e-12, 0.067670285905209, 3.406616375784729e-11,
           7.0594200743628e-11, 1.978746628123619e-9, 1.2487228359728224e-11,
           6.736329552903881e-11, 0.4911343239711766, 2.7696701331426515e-11,
           0.02869986911957876, 5.145977511412783e-12, 9.825203329679707e-12,
           0.1454387869171815, 0.09520044944324468, 0.10299333992719902,
           5.416355995326648e-11, 0.031124840735209925]

    w2t = [0.03773809824707895, 4.7950055503122635e-12, 1.7512419976284742e-11,
           8.830922537142664e-12, 0.06767028626913404, 5.139387169962604e-11,
           1.5905292117348912e-10, 4.431681248604511e-9, 5.566762528210568e-12,
           1.0926399124593914e-10, 0.4911343247919767, 3.8720115575498385e-11,
           0.028699868269157304, 2.6325272898761313e-11, 3.2537753294099414e-11,
           0.1454387858203395, 0.09520045496590665, 0.10299334046308291,
           7.475396852526468e-11, 0.031124836212889816]

    w3t = [0.0377381105946706, 3.515881253854551e-11, 2.357341494392936e-10,
           2.559678033913933e-11, 0.06767028183074263, 7.783217404030185e-11,
           2.3923170656167073e-10, 2.9515493486199157e-9, 4.1184253345251125e-11,
           1.1344540362687383e-10, 0.4911343223544339, 6.716667348006059e-11,
           0.028699868315736374, 2.264050607030489e-11, 1.7950773709630438e-11,
           0.1454387874690095, 0.09520044732491449, 0.10299332822716616,
           9.114327674461133e-11, 0.031124849964692478]

    w4t = [0.037738097788721364, 5.626447701425248e-12, 2.068109015485723e-12,
           1.0352773875396406e-11, 0.06767028841987822, 1.5753302684443457e-11,
           1.043159457210935e-10, 1.2497886691974153e-9, 5.113390259122275e-12,
           3.8904722930687346e-11, 0.4911343232238213, 9.437166064946229e-12,
           0.02869986938523649, 1.944129041397662e-11, 2.082103656049831e-11,
           0.1454387869942569, 0.09520045084137742, 0.10299334654561362,
           2.3619899407858453e-11, 0.031124835295851968]

    w5t = [0.0377381008605952, 2.1395064127499257e-11, 4.56268497506332e-11,
           7.148263563923392e-12, 0.06767028828542994, 6.78369667138314e-11,
           1.0716413512416832e-10, 1.62987846066615e-9, 2.553557808684936e-11,
           1.2241849684871454e-10, 0.49113431969118, 5.567949502506581e-11,
           0.02869986808689812, 4.591809651924742e-12, 1.2257034909431584e-11,
           0.14543878956006676, 0.09520045253667762, 0.10299334531821301,
           9.979589582025746e-11, 0.031124833461611292]

    w6t = [0.0377381064423748, 2.2562315685094713e-11, 8.56029528762664e-11,
           1.7106146706284752e-11, 0.06767028509664332, 4.682338748599671e-11,
           1.9323906468788836e-10, 1.1186716254178256e-9, 2.6159896885607444e-11,
           6.838778331226203e-11, 0.4911343214805795, 4.007981875449744e-11,
           0.028699868994417097, 1.4182955093346131e-11, 1.0755467400201429e-11,
           0.14543878849203562, 0.09520044661792577, 0.10299333685233825,
           6.223288088756335e-11, 0.031124844317881293]

    w7t = [4.892325460504359e-9, 2.3620744806861124e-9, 2.416375993540523e-7,
           9.449041516295317e-10, 0.28539562041854605, 3.5238346161000153e-10,
           3.069601041000791e-9, 0.07964022360209898, 6.116688867974749e-10,
           7.647888674416564e-10, 0.3337090408880962, 2.1521588791802522e-10,
           6.334107711301085e-12, 1.0779808634446087e-9, 1.9538752835653e-10,
           0.30125484690320103, 3.985302853433573e-9, 2.104166972119901e-9,
           1.806659650134201e-9, 4.161664297087937e-9]

    w8t = [4.6723263531254646e-8, 1.3113687908729208e-8, 0.003281659568549761,
           6.709007878694467e-9, 0.28652423341497996, 3.7799800035547e-9,
           2.330856921113036e-8, 0.07216110179261974, 5.3175348613050524e-9,
           6.081023281396519e-9, 0.3429541541961689, 3.019445915913519e-9,
           2.0311295020036126e-9, 7.4245780738037804e-9, 2.726540364018463e-9,
           0.2950786382758394, 3.848475275732074e-8, 1.5101860040861237e-8,
           1.378913244546831e-8, 2.5141336444393144e-8]

    w9t = [1.3507642533445513e-7, 6.228521548436306e-8, 0.011943598042801758,
           3.0068293604911325e-8, 0.2887265202453238, 1.650755335235124e-8,
           8.475378431285043e-7, 0.06348061565479772, 2.233311587450199e-8,
           2.6140813302542165e-8, 0.35123695275072575, 1.4015736174326238e-8,
           8.64500222884532e-9, 3.370292173399281e-8, 1.3354045998767818e-8,
           0.28461079133695205, 1.0631629278754787e-7, 5.6609548077446006e-8,
           5.054511418619518e-8, 9.883147764515989e-8]

    w10t = [1.6659582987187668e-8, 1.748896200801003e-8, 8.398111881754062e-8,
            1.976860094668443e-8, 8.038624101722981e-5, 7.651179700048689e-9,
            0.9999192072483999, 1.4278742389853873e-8, 1.8409804420583514e-8,
            1.3238423567902734e-8, 1.2664504635066056e-8, 8.088410227872565e-9,
            5.8215448239161296e-9, 9.894326492642074e-9, 6.023921411537698e-9,
            1.0159376396877056e-7, 2.3101742441311244e-8, 1.2756696592544408e-8,
            1.9324369590668562e-8, 1.576488793769761e-8]

    w11t = [3.539836162718825e-11, 3.221953648664385e-11, 2.5781639181400178e-11,
            2.8113465819272007e-11, 0.8503240434403975, 4.695833920374047e-11,
            0.14967595592182592, 4.167646866074926e-11, 2.8344090818571735e-11,
            3.996463700031924e-11, 4.194297435645168e-11, 4.5386353973682577e-11,
            4.783095072688438e-11, 4.467501087164152e-11, 4.527566219651595e-11,
            1.61421821798192e-11, 1.7708013368013927e-11, 3.937990564631631e-11,
            2.5979109008401075e-11, 3.4999581658780914e-11]

    w12t = [2.6729502263208933e-9, 2.909649534210988e-9, 3.5331620352017617e-9,
            3.188289682208487e-9, 0.8533950400928205, 1.300276858911552e-9,
            0.1466049148411774, 1.988525027566319e-9, 3.072992367841461e-9,
            2.0823133012912945e-9, 1.875054545788815e-9, 1.2409112305387383e-9,
            9.275490306960879e-10, 1.5661689288996204e-9, 9.209670141567376e-10,
            5.9438351916223865e-9, 3.955783810914124e-9, 2.1047111610602355e-9,
            3.280151598193955e-9, 2.502710360668684e-9]

    @test isapprox(w1.weights, w1t)
    @test isapprox(w2.weights, w2t)
    @test isapprox(w3.weights, w3t)
    @test isapprox(w2.weights, w3.weights, rtol = 1e-7)
    @test isapprox(w4.weights, w4t)
    @test isapprox(w5.weights, w5t)
    @test isapprox(w6.weights, w6t)
    @test isapprox(w5.weights, w6.weights, rtol = 1e-7)
    @test isapprox(w7.weights, w7t)
    @test isapprox(w8.weights, w8t)
    @test isapprox(w9.weights, w9t, rtol = 1e-4)
    @test isapprox(w8.weights, w9.weights, rtol = 1e-1)
    @test isapprox(w10.weights, w10t)
    @test isapprox(w11.weights, w11t)
    @test isapprox(w12.weights, w12t, rtol = 1e-6)
    @test isapprox(w11.weights, w12.weights, rtol = 1e-2)
    @test isapprox(w13.weights, w7.weights, rtol = 1e-5)
    @test isapprox(w14.weights, w8.weights, rtol = 1e-1)
    @test isapprox(w15.weights, w9.weights, rtol = 1e-1)
    @test isapprox(w16.weights, w7.weights, rtol = 1e-5)
    @test isapprox(w17.weights, w8.weights, rtol = 1e-1)
    @test isapprox(w18.weights, w9.weights, rtol = 1e-1)
    @test isapprox(w13.weights, w16.weights, rtol = 1e-6)
    @test isapprox(w14.weights, w17.weights, rtol = 1e-2)
    @test isapprox(w15.weights, w18.weights, rtol = 1e-2)
    @test isapprox(w19.weights, w1.weights, rtol = 1e-5)
end

@testset "$(:Classic), $(:Trad), $(:ADD)" begin
    portfolio = Portfolio(; prices = prices,
                          solvers = OrderedDict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                                  :params => Dict("verbose" => false,
                                                                                  "max_step_fraction" => 0.75))))
    asset_statistics!(portfolio)

    rm = :ADD
    opt = OptimiseOpt(; rf = rf, l = l, class = :Classic, type = :Trad, rm = rm,
                      obj = :Min_Risk, kelly = :None)

    w1 = optimise!(portfolio, opt)
    risk1 = calc_risk(portfolio; type = :Trad, rm = rm, rf = rf)
    opt.kelly = :Approx
    w2 = optimise!(portfolio, opt)
    opt.kelly = :Exact
    w3 = optimise!(portfolio, opt)
    opt.obj = :Utility
    opt.kelly = :None
    w4 = optimise!(portfolio, opt)
    opt.kelly = :Approx
    w5 = optimise!(portfolio, opt)
    opt.kelly = :Exact
    w6 = optimise!(portfolio, opt)
    opt.obj = :Sharpe
    opt.kelly = :None
    w7 = optimise!(portfolio, opt)
    risk7 = calc_risk(portfolio; type = :Trad, rm = rm, rf = rf)
    ret7 = dot(portfolio.mu, w7.weights)
    opt.kelly = :Approx
    w8 = optimise!(portfolio, opt)
    risk8 = calc_risk(portfolio; type = :Trad, rm = rm, rf = rf)
    ret8 = dot(portfolio.mu, w8.weights)
    opt.kelly = :Exact
    w9 = optimise!(portfolio, opt)
    risk9 = calc_risk(portfolio; type = :Trad, rm = rm, rf = rf)
    ret9 = dot(portfolio.mu, w9.weights)
    opt.obj = :Max_Ret
    opt.kelly = :None
    w10 = optimise!(portfolio, opt)
    opt.kelly = :Approx
    w11 = optimise!(portfolio, opt)
    opt.kelly = :Exact
    w12 = optimise!(portfolio, opt)
    setproperty!(portfolio, :mu_l, ret7)
    opt.obj = :Min_Risk
    opt.kelly = :None
    w13 = optimise!(portfolio, opt)
    setproperty!(portfolio, :mu_l, ret8)
    w14 = optimise!(portfolio, opt)
    setproperty!(portfolio, :mu_l, ret9)
    w15 = optimise!(portfolio, opt)
    setproperty!(portfolio, :mu_l, Inf)
    rmf = Symbol(lowercase(string(rm)) * "_u")
    setproperty!(portfolio, rmf, risk7)
    opt.obj = :Max_Ret
    opt.kelly = :None
    w16 = optimise!(portfolio, opt)
    setproperty!(portfolio, rmf, risk8)
    w17 = optimise!(portfolio, opt)
    setproperty!(portfolio, rmf, risk9)
    w18 = optimise!(portfolio, opt)
    setproperty!(portfolio, rmf, risk1)
    opt.obj = :Sharpe
    w19 = optimise!(portfolio, opt)
    setproperty!(portfolio, rmf, Inf)

    w1t = [1.658867109370025e-9, 0.04440526789586851, 0.15876310140117367,
           4.786120156590545e-10, 0.06788824555596652, 9.406377552206124e-10,
           5.633647612600515e-10, 0.06197612030177127, 6.838564922740126e-10,
           9.41023594121528e-10, 0.21027705608097105, 2.5569840425410465e-9,
           0.001212203514973533, 0.01721503423748183, 1.885383269183345e-10,
           0.06907119869801609, 0.10146329053910068, 0.022503670310607772,
           0.1303621006875258, 0.11486270276465917]

    w2t = [2.9546197309498427e-10, 0.044412487099918234, 0.15877836007075094,
           8.262600524566952e-11, 0.06789444067626153, 1.5774159966490687e-10,
           1.1169193908548297e-10, 0.061987202427287794, 1.1847209661231412e-10,
           1.6344588468884824e-10, 0.21025763539759812, 4.587593288791562e-10,
           0.0012130793412735766, 0.017215982637797774, 2.0490560202680597e-11,
           0.06906957886270527, 0.1014613511336555, 0.022482582236001403,
           0.13036122597379143, 0.11486607273426885]

    w3t = [7.938306054361492e-10, 0.04441100724016392, 0.1587752096080819,
           2.97931516762739e-10, 0.06789317466620222, 4.967320027648943e-10,
           6.588108120914333e-10, 0.06198503883356246, 3.7093735367645567e-10,
           4.786883837155837e-10, 0.2102614862270133, 1.044518846372774e-9,
           0.0012128645736156581, 0.017215818727124545, 1.7029252204428173e-10,
           0.06906992186554593, 0.10146201857751622, 0.02248675775282967,
           0.13036126069142137, 0.1148654369251809]

    w4t = [4.021114768591322e-10, 0.04241219267735608, 0.15457806744390729,
           1.1249859837396645e-10, 0.06853005727165204, 1.72519856923856e-10,
           1.5404289624380286e-10, 0.0647789180494693, 1.6005807804400596e-10,
           2.4095700898524293e-10, 0.20764907203706887, 3.9616592823426475e-10,
           0.0007314223964568549, 0.018524405647912967, 9.11031171934365e-12,
           0.07144516364544452, 0.10791271411091963, 0.01630496921811774,
           0.13059135751262207, 0.11654165834160858]

    w5t = [3.921606293348361e-10, 0.042412197038969784, 0.15457806960678974,
           2.3190291991119676e-10, 0.06853005911678414, 3.7274082018941094e-11,
           2.553396863412027e-10, 0.06477895442296683, 6.000149885365848e-11,
           1.0134112817076893e-10, 0.20764901834146207, 5.455726899589358e-10,
           0.0007314711091918342, 0.018524451772162186, 3.0259183861816564e-10,
           0.07144514134880234, 0.10791275003486642, 0.016304998139771636,
           0.1305912732187585, 0.11654161392329017]

    w6t = [2.5976742265504515e-10, 0.04241218281488972, 0.15457804750666868,
           1.0357964400316277e-10, 0.06853005512798623, 1.503991627652277e-10,
           2.2248790093301881e-10, 0.06477892477728506, 1.2544236474792164e-10,
           1.692810625708977e-10, 0.2076490720103722, 2.474802743709651e-10,
           0.0007314159726606864, 0.018524402754752714, 5.617714525428631e-11,
           0.07144517348103258, 0.10791275268838114, 0.016304951154709438,
           0.13059135492459267, 0.11654166545205402]

    w7t = [9.564872974711584e-9, 5.7339670555083325e-8, 0.22303024168623314,
           1.515431601555135e-8, 0.23131276392529762, 3.594812162144443e-9,
           0.0024132848105890155, 0.06938079241017829, 1.1530974218270493e-8,
           1.0263855053385585e-8, 0.09197047516463074, 3.2887248368439274e-9,
           1.6928114665239044e-9, 1.2773739691815457e-8, 2.5251340775314692e-9,
           0.14135016926776034, 0.14559430335011078, 2.0500941272628907e-8,
           0.09494773308743353, 8.806791436337999e-8]

    w8t = [1.3769435079831538e-10, 1.2814870478488616e-9, 0.23659136870999053,
           1.7404404863906068e-10, 0.20617017158098644, 5.742894425207952e-11,
           0.0015585418951312156, 0.07372211287680558, 1.4832607638254242e-10,
           1.5236406915212554e-10, 0.13344052584803504, 5.1400009573103774e-11,
           3.1974521586877296e-11, 2.055032225779995e-10, 4.0839438548595924e-11,
           0.1326892973707173, 0.1439100113182151, 3.051941525665131e-10,
           0.07191796640979169, 1.404071116827574e-9]

    w9t = [3.5610882120608872e-9, 2.084403425207242e-8, 0.2169518203681985,
           5.524677487770747e-9, 0.23650600750340378, 1.5076879610356949e-9,
           0.0025865086874570196, 0.06889626918299784, 4.331536120278762e-9,
           4.139851168335594e-9, 0.09239049046287896, 1.412000145364046e-9,
           8.501530417062027e-10, 4.643390457492644e-9, 1.1355303271995159e-9,
           0.1408958730232913, 0.14604825808208508, 8.27699472973074e-9,
           0.09572466740842041, 4.905432339140721e-8]

    w10t = [6.509552105512051e-8, 6.947144693820443e-8, 9.594304133014301e-8,
            7.945690876372323e-8, 0.0002807779790487726, 3.031506173615717e-8,
            0.9997179764556234, 4.8655663070948615e-8, 7.294086894739173e-8,
            5.250806729457889e-8, 4.676943046511983e-8, 3.2080982132331165e-8,
            2.2966599435685053e-8, 3.8909639005585543e-8, 2.3786624206100866e-8,
            2.8615627299161437e-7, 9.192652328337918e-8, 5.0433638519792876e-8,
            7.633808962726245e-8, 6.181094888873405e-8]

    w11t = [7.930914195345808e-10, 7.607771842189218e-10, 6.136954733870341e-10,
            7.105794828735389e-10, 0.8503244335948051, 9.832202603863963e-10,
            0.1496755522542633, 8.858512936320987e-10, 7.319822174654019e-10,
            8.9338230540246e-10, 8.9517845689364e-10, 9.627624396940222e-10,
            9.984456075862567e-10, 9.456730938777202e-10, 9.576920709058473e-10,
            1.6831563157438143e-10, 5.196207575489219e-10, 8.706454084281897e-10,
            6.591313671322272e-10, 8.008867882414274e-10]

    w12t = [2.6164587113268375e-9, 2.857869118898357e-9, 3.432590485956772e-9,
            3.133851285334089e-9, 0.8533950197131018, 1.282713568266341e-9,
            0.14660493604494526, 1.969663275786142e-9, 3.0289997855180384e-9,
            2.057304348546363e-9, 1.854884475587887e-9, 1.2353107763271118e-9,
            9.232228155766545e-10, 1.55332316100827e-9, 9.187022345375131e-10,
            5.737312851552629e-9, 3.874665402771516e-9, 2.0779633060684033e-9,
            3.227062163001105e-9, 2.460055099149407e-9]

    @test isapprox(w1.weights, w1t)
    @test isapprox(w2.weights, w2t)
    @test isapprox(w3.weights, w3t)
    @test isapprox(w2.weights, w3.weights, rtol = 1e-4)
    @test isapprox(w4.weights, w4t)
    @test isapprox(w5.weights, w5t)
    @test isapprox(w6.weights, w6t, rtol = 1e-7)
    @test isapprox(w5.weights, w6.weights, rtol = 1e-6)
    @test isapprox(w7.weights, w7t)
    @test isapprox(w8.weights, w8t)
    @test isapprox(w9.weights, w9t, rtol = 1e-5)
    @test isapprox(w8.weights, w9.weights, atol = 1e-1)
    @test isapprox(w10.weights, w10t)
    @test isapprox(w11.weights, w11t)
    @test isapprox(w12.weights, w12t, rtol = 1e-6)
    @test isapprox(w11.weights, w12.weights, rtol = 1e-2)
    @test isapprox(w13.weights, w7.weights, rtol = 1e-4)
    @test isapprox(w14.weights, w8.weights, rtol = 1e-1)
    @test isapprox(w15.weights, w9.weights, rtol = 1e-1)
    @test isapprox(w16.weights, w7.weights, rtol = 1e-5)
    @test isapprox(w17.weights, w8.weights, rtol = 1e-1)
    @test isapprox(w18.weights, w9.weights, rtol = 1e-1)
    @test isapprox(w13.weights, w16.weights, rtol = 1e-5)
    @test isapprox(w14.weights, w17.weights, rtol = 1e-3)
    @test isapprox(w15.weights, w18.weights, rtol = 1e-3)
    @test isapprox(w19.weights, w1.weights, rtol = 1e-3)
end

@testset "$(:Classic), $(:Trad), $(:CDaR)" begin
    portfolio = Portfolio(; prices = prices,
                          solvers = OrderedDict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                                  :params => Dict("verbose" => false,
                                                                                  "max_step_fraction" => 0.75))))
    asset_statistics!(portfolio)

    rm = :CDaR
    opt = OptimiseOpt(; rf = rf, l = l, class = :Classic, type = :Trad, rm = rm,
                      obj = :Min_Risk, kelly = :None)

    w1 = optimise!(portfolio, opt)
    risk1 = calc_risk(portfolio; type = :Trad, rm = rm, rf = rf)
    opt.kelly = :Approx
    w2 = optimise!(portfolio, opt)
    opt.kelly = :Exact
    w3 = optimise!(portfolio, opt)
    opt.obj = :Utility
    opt.kelly = :None
    w4 = optimise!(portfolio, opt)
    opt.kelly = :Approx
    w5 = optimise!(portfolio, opt)
    opt.kelly = :Exact
    w6 = optimise!(portfolio, opt)
    opt.obj = :Sharpe
    opt.kelly = :None
    w7 = optimise!(portfolio, opt)
    risk7 = calc_risk(portfolio; type = :Trad, rm = rm, rf = rf)
    ret7 = dot(portfolio.mu, w7.weights)
    opt.kelly = :Approx
    w8 = optimise!(portfolio, opt)
    risk8 = calc_risk(portfolio; type = :Trad, rm = rm, rf = rf)
    ret8 = dot(portfolio.mu, w8.weights)
    opt.kelly = :Exact
    w9 = optimise!(portfolio, opt)
    risk9 = calc_risk(portfolio; type = :Trad, rm = rm, rf = rf)
    ret9 = dot(portfolio.mu, w9.weights)
    opt.obj = :Max_Ret
    opt.kelly = :None
    w10 = optimise!(portfolio, opt)
    opt.kelly = :Approx
    w11 = optimise!(portfolio, opt)
    opt.kelly = :Exact
    w12 = optimise!(portfolio, opt)
    setproperty!(portfolio, :mu_l, ret7)
    opt.obj = :Min_Risk
    opt.kelly = :None
    w13 = optimise!(portfolio, opt)
    setproperty!(portfolio, :mu_l, ret8)
    w14 = optimise!(portfolio, opt)
    setproperty!(portfolio, :mu_l, ret9)
    w15 = optimise!(portfolio, opt)
    setproperty!(portfolio, :mu_l, Inf)
    rmf = Symbol(lowercase(string(rm)) * "_u")
    setproperty!(portfolio, rmf, risk7)
    opt.obj = :Max_Ret
    opt.kelly = :None
    w16 = optimise!(portfolio, opt)
    setproperty!(portfolio, rmf, risk8)
    w17 = optimise!(portfolio, opt)
    setproperty!(portfolio, rmf, risk9)
    w18 = optimise!(portfolio, opt)
    setproperty!(portfolio, rmf, risk1)
    opt.obj = :Sharpe
    w19 = optimise!(portfolio, opt)
    setproperty!(portfolio, rmf, Inf)

    w1t = [1.1760269161470477e-11, 4.51896814609307e-10, 2.8194904298860603e-11,
           2.449667278432988e-12, 0.0034099023246082208, 4.093302275581014e-12,
           1.5423101424735247e-11, 0.07904283094279253, 1.9458418637841482e-11,
           2.8270174222165505e-12, 0.3875931677297157, 6.139851118171879e-11,
           8.345514723821662e-11, 1.5236413923661703e-10, 0.0005545109736418541,
           0.09598829145444249, 0.2679088659263076, 3.5541175305583344e-10,
           0.0006560252623130244, 0.16484640419744556]

    w2t = [7.012800082751424e-12, 4.619212287248136e-10, 2.0656468028906088e-11,
           2.046603633441899e-12, 0.003409902426221898, 9.202914356270217e-12,
           2.045599355249358e-11, 0.07904283127070877, 1.547774827911816e-11,
           7.588878368281477e-12, 0.38759316760783663, 5.743982661180284e-11,
           7.952650368554001e-11, 1.5141942547021284e-10, 0.0005545107663803482,
           0.09598829156717693, 0.2679088645382247, 3.8299623957123007e-10,
           0.0006560259050787777, 0.1648464047026272]

    w3t = [6.615480034093807e-11, 9.007666075308696e-10, 2.3729682679034493e-10,
           4.059424831719802e-11, 0.0034099026082622403, 3.064216146422458e-11,
           3.843083028853642e-11, 0.07904283116151786, 7.123137462241196e-11,
           3.2614570567834254e-11, 0.38759317074994043, 1.4765405420424953e-10,
           1.6274602374633934e-10, 2.9104033416483276e-10, 0.0005545068989028069,
           0.09598829049874072, 0.2679088593452675, 4.890855081865546e-10,
           0.0006560269641851877, 0.16484640926492586]

    w4t = [2.7516810481229006e-12, 6.022413914687849e-10, 1.6653163257824942e-11,
           7.075598423308588e-12, 0.0035594778263146193, 1.5451916395354444e-11,
           2.8349564592097567e-11, 0.07997306459976396, 1.2555683175689363e-11,
           1.3971338123767737e-11, 0.3871598951354419, 5.965579149043487e-11,
           6.414929569013956e-11, 1.5666844101987548e-10, 0.00022182233592094708,
           0.09652576670482815, 0.26593973648571356, 3.5011626347884844e-10,
           0.0019205892824167372, 0.16469964629996]

    w5t = [9.244049822534778e-12, 3.9446389235177096e-10, 2.0268786502242605e-11,
           2.6673723439778847e-12, 0.0035594781414829274, 2.5964904344141702e-12,
           1.1452236094077253e-11, 0.07997306594745791, 1.584366794271913e-11,
           1.7218035060140847e-12, 0.38715989255559247, 4.8085840158486825e-11,
           5.2186541505803215e-11, 1.1545626620589032e-10, 0.00022182516991004964,
           0.09652576729385567, 0.26593974272504417, 2.2039312053697949e-10,
           0.0019205872662845678, 0.1646996400059921]

    w6t = [2.402258858050201e-11, 3.2891192981880564e-10, 7.296806542601198e-11,
           1.505605397362157e-11, 0.0035594777823382325, 1.1122696652413812e-11,
           1.5052771912294595e-11, 0.07997306748495751, 2.641070785571999e-11,
           1.185069951393151e-11, 0.38715989283923846, 5.2063196969647463e-11,
           5.3337593285609885e-11, 1.0430615212282018e-10, 0.00022182352521205273,
           0.09652576935320581, 0.26593973676015886, 1.6201917406730117e-10,
           0.0019205918913253596, 0.16469963948644215]

    w7t = [1.0183926119455649e-9, 1.5396821165797216e-9, 0.07233516307568606,
           3.826621409788533e-10, 0.3107275054983851, 6.45427344133195e-11,
           5.584504285845458e-10, 0.1286151655884055, 2.996423933821242e-10,
           3.2586987432543617e-10, 0.16437851379378257, 1.7148118285990425e-10,
           5.6929824062815345e-11, 7.389934535320326e-10, 6.741905114083056e-11,
           0.26288213034459407, 4.4326074687963796e-9, 1.1140396114972316e-9,
           1.3229167472856395e-9, 0.06106150960551708]

    w8t = [6.9278112142706525e-9, 1.3808803736604177e-8, 0.06188556414535693,
           4.040723586318945e-9, 0.2561799771565045, 1.970866113351878e-9,
           6.025760331006358e-9, 0.1358961109475559, 3.7945832255902746e-9,
           3.6417482532404362e-9, 0.1825105987011374, 2.4776630901329517e-9,
           1.3478611175697586e-9, 5.886460489984224e-9, 1.9693335738338864e-9,
           0.17526675529483798, 1.8464918991668192e-7, 9.966941643803265e-9,
           1.3289625031335693e-8, 0.18826073395723605]

    w9t = [4.1085195899104635e-8, 6.776292328232836e-8, 0.07723233575536713,
           2.2126011344526693e-8, 0.28355893448054137, 1.1522110480122366e-8,
           5.035679301378742e-8, 0.12101291760885102, 1.9242047572508437e-8,
           2.0336124569434956e-8, 0.1625029824328639, 1.4697344361741197e-8,
           7.673629751161474e-9, 3.416827661009607e-8, 1.1972854973796668e-8,
           0.2371765815501717, 1.7244976802821185e-7, 4.721175958821201e-8,
           5.2447637766646616e-8, 0.11851567511972762]

    w10t = [1.2188118577218307e-8, 1.2907082924304506e-8, 2.213720505470441e-8,
            1.4865191679647488e-8, 6.827517229939409e-5, 5.6806877113057485e-9,
            0.9999314784669739, 9.27199397363275e-9, 1.3611753959598207e-8,
            9.81501537310985e-9, 8.844263557232336e-9, 6.00000086986027e-9,
            4.3002142182094945e-9, 7.30094603149862e-9, 4.475767087611672e-9,
            6.230792922550653e-8, 1.730928459805808e-8, 9.4551245335049e-9,
            1.4321320214800878e-8, 1.1568827163714552e-8]

    w11t = [4.195065318119874e-11, 3.982139469473178e-11, 3.2345691969181575e-11,
            3.7027795569977416e-11, 0.8503244615708407, 5.246440737144838e-11,
            0.14967553767946676, 4.710327236473579e-11, 3.8074242631169895e-11,
            4.720265984081426e-11, 4.7562820599568226e-11, 5.1208671141323496e-11,
            5.333725870904038e-11, 5.027017479107912e-11, 5.07703283043863e-11,
            1.2349598943969907e-11, 2.6170906354444942e-11, 4.599637513326967e-11,
            3.394320819768132e-11, 4.2092912261392836e-11]

    w12t = [2.6506474045310002e-9, 2.8975793306712963e-9, 3.508966985045771e-9,
            3.1832862621456444e-9, 0.8533732043830846, 1.2785783868919741e-9,
            0.14662675074701076, 1.9812046683998626e-9, 3.0705239458182888e-9,
            2.074795442693323e-9, 1.866487321295243e-9, 1.233531291455467e-9,
            9.184664047037696e-10, 1.5565062423455896e-9, 9.157051626123145e-10,
            5.926371526760078e-9, 3.949275538623232e-9, 2.0928966058801505e-9,
            3.274967840465919e-9, 2.4901142655300988e-9]

    @test isapprox(w1.weights, w1t)
    @test isapprox(w2.weights, w2t)
    @test isapprox(w3.weights, w3t)
    @test isapprox(w2.weights, w3.weights, rtol = 1e-7)
    @test isapprox(w4.weights, w4t)
    @test isapprox(w5.weights, w5t)
    @test isapprox(w6.weights, w6t, rtol = 1.0e-7)
    @test isapprox(w5.weights, w6.weights, rtol = 1e-7)
    @test isapprox(w7.weights, w7t)
    @test isapprox(w8.weights, w8t)
    @test isapprox(w9.weights, w9t, rtol = 1e-5)
    @test isapprox(w8.weights, w9.weights, rtol = 3e-1)
    @test isapprox(w10.weights, w10t)
    @test isapprox(w11.weights, w11t)
    @test isapprox(w12.weights, w12t, rtol = 1e-4)
    @test isapprox(w11.weights, w12.weights, rtol = 1e-2)
    @test isapprox(w13.weights, w7.weights, rtol = 1e-5)
    @test isapprox(w14.weights, w8.weights, rtol = 1e-4)
    @test isapprox(w15.weights, w9.weights, rtol = 1e-1)
    @test isapprox(w16.weights, w7.weights, rtol = 1e-5)
    @test isapprox(w17.weights, w8.weights, rtol = 1e-4)
    @test isapprox(w18.weights, w9.weights, rtol = 1e-1)
    @test isapprox(w13.weights, w16.weights, rtol = 1e-5)
    @test isapprox(w14.weights, w17.weights, rtol = 1e-6)
    @test isapprox(w15.weights, w18.weights, rtol = 1e-3)
    @test isapprox(w19.weights, w1.weights, rtol = 1e-4)
end

@testset "$(:Classic), $(:Trad), $(:UCI)" begin
    portfolio = Portfolio(; prices = prices,
                          solvers = OrderedDict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                                  :params => Dict("verbose" => false,
                                                                                  "max_step_fraction" => 0.75))))
    asset_statistics!(portfolio)

    rm = :UCI
    opt = OptimiseOpt(; rf = rf, l = l, class = :Classic, type = :Trad, rm = rm,
                      obj = :Min_Risk, kelly = :None)

    w1 = optimise!(portfolio, opt)
    risk1 = calc_risk(portfolio; type = :Trad, rm = rm, rf = rf)
    opt.kelly = :Approx
    w2 = optimise!(portfolio, opt)
    opt.kelly = :Exact
    w3 = optimise!(portfolio, opt)
    opt.obj = :Utility
    opt.kelly = :None
    w4 = optimise!(portfolio, opt)
    opt.kelly = :Approx
    w5 = optimise!(portfolio, opt)
    opt.kelly = :Exact
    w6 = optimise!(portfolio, opt)
    opt.obj = :Sharpe
    opt.kelly = :None
    w7 = optimise!(portfolio, opt)
    risk7 = calc_risk(portfolio; type = :Trad, rm = rm, rf = rf)
    ret7 = dot(portfolio.mu, w7.weights)
    opt.kelly = :Approx
    w8 = optimise!(portfolio, opt)
    risk8 = calc_risk(portfolio; type = :Trad, rm = rm, rf = rf)
    ret8 = dot(portfolio.mu, w8.weights)
    opt.kelly = :Exact
    w9 = optimise!(portfolio, opt)
    risk9 = calc_risk(portfolio; type = :Trad, rm = rm, rf = rf)
    ret9 = dot(portfolio.mu, w9.weights)
    opt.obj = :Max_Ret
    opt.kelly = :None
    w10 = optimise!(portfolio, opt)
    opt.kelly = :Approx
    w11 = optimise!(portfolio, opt)
    opt.kelly = :Exact
    w12 = optimise!(portfolio, opt)
    setproperty!(portfolio, :mu_l, ret7)
    opt.obj = :Min_Risk
    opt.kelly = :None
    w13 = optimise!(portfolio, opt)
    setproperty!(portfolio, :mu_l, ret8)
    w14 = optimise!(portfolio, opt)
    setproperty!(portfolio, :mu_l, ret9)
    w15 = optimise!(portfolio, opt)
    setproperty!(portfolio, :mu_l, Inf)
    rmf = Symbol(lowercase(string(rm)) * "_u")
    setproperty!(portfolio, rmf, risk7)
    opt.obj = :Max_Ret
    opt.kelly = :None
    w16 = optimise!(portfolio, opt)
    setproperty!(portfolio, rmf, risk8)
    w17 = optimise!(portfolio, opt)
    setproperty!(portfolio, rmf, risk9)
    w18 = optimise!(portfolio, opt)
    setproperty!(portfolio, rmf, risk1 + 1e-6 * risk1)
    opt.obj = :Sharpe
    w19 = optimise!(portfolio, opt)
    setproperty!(portfolio, rmf, Inf)

    w1t = [1.1582610619993254e-9, 0.025012031153354795, 0.09201905647247656,
           2.4725280175282603e-10, 0.01940460141965807, 3.248470115263675e-10,
           2.241729738092696e-10, 0.06508416266874355, 7.094166285479311e-10,
           4.678882101877213e-10, 0.2853657602865979, 1.4540433421342913e-9,
           1.1859347446814042e-9, 0.008371105129174886, 2.515571799428369e-10,
           0.09716533696621195, 0.17512906576943255, 0.024590148078092143,
           0.05728417228728839, 0.1505745537455952]

    w2t = [7.784027707075817e-10, 0.025012110446565015, 0.09201913739932868,
           1.3192672206712626e-10, 0.019404634673634007, 1.5912173753626132e-10,
           3.820611725654994e-11, 0.06508415641282228, 4.5644825722858323e-10,
           2.772590436525527e-10, 0.2853656842035142, 9.506831652427965e-10,
           6.958161100494381e-10, 0.008371191534910654, 9.400722202727143e-11,
           0.09716524824000686, 0.17512906803512085, 0.024590169228223192,
           0.05728401925428821, 0.1505745769897148]

    w3t = [3.3157554274495315e-10, 0.02501223132102398, 0.09201934664259247,
           1.271693261565579e-10, 0.019404706159597128, 1.4627748469164312e-10,
           3.7298104720989985e-10, 0.06508419151791228, 2.1468318007124789e-10,
           1.7712943419181205e-10, 0.2853655309015868, 3.449938348998477e-10,
           3.0490478826230125e-10, 0.0083712542975607, 1.1905896477424563e-10,
           0.09716510922022549, 0.17512902078887851, 0.02459014641712313,
           0.05728384544490632, 0.15057461514981957]

    w4t = [3.2971412359929465e-10, 0.02399497508890937, 0.09614539616130102,
           1.5162778377743103e-11, 0.019578463866494714, 3.336507489284487e-11,
           3.1061276876896947e-11, 0.06625965123714526, 1.7434645574938298e-10,
           9.118673981267884e-11, 0.28380866879643013, 4.025821520660614e-10,
           2.3103428439635038e-10, 0.0015418711740064976, 1.1832699483551404e-11,
           0.10123975650957402, 0.17626248033257308, 0.02047764286534096,
           0.06087184051397425, 0.1498192521339652]

    w5t = [1.330815308043571e-10, 0.024111764817956096, 0.09588132573430211,
           2.2240944270615014e-10, 0.01966706184693453, 1.8817549981397476e-10,
           3.6309205543919376e-10, 0.06618957601631957, 2.8748369925713323e-11,
           1.342454844466748e-10, 0.283893078077539, 2.1949452478313269e-10,
           2.504102499707302e-10, 0.001921989258750942, 2.216529587761545e-10,
           0.10102337161162057, 0.17613365894985925, 0.020639404836029353,
           0.06070355437435705, 0.14983521271502145]

    w6t = [1.0553569482950676e-10, 0.024111024567765194, 0.09587702639505548,
           4.0844844816005457e-11, 0.019666507377059217, 4.524916339799068e-11,
           1.0353540990548643e-10, 0.06618843922137391, 6.802278948236294e-11,
           5.6122626064496073e-11, 0.28389601405925674, 1.0561627817754676e-10,
           8.22102312688968e-11, 0.0019256887225310405, 3.727891926423515e-11,
           0.10102227096126733, 0.17613341687855733, 0.020642251349448342,
           0.06070194543416285, 0.14983541438910664]

    w7t = [5.957569918068336e-9, 1.3980735962960395e-8, 0.20539044230848347,
           6.832776734031752e-9, 0.28579716189660503, 1.717136596270128e-9,
           5.7198725705352006e-8, 0.14359015272253803, 5.735523940211634e-9,
           4.82854673110931e-9, 0.07575264601248562, 1.7276509424310319e-9,
           8.591348958592942e-10, 7.193325697407817e-9, 1.207476622256161e-9,
           0.18477267663197217, 0.10469657485541108, 1.0760967198017963e-8,
           1.0010342880180476e-7, 1.2746950496085876e-7]

    w8t = [1.0763689193617502e-8, 2.548230317000341e-8, 0.19292156743774386,
           9.939608290267437e-9, 0.23084390026204613, 3.775049472236702e-9,
           7.459958683988046e-8, 0.13727321764317305, 1.030390784566238e-8,
           8.886015220885793e-9, 0.11254433471598915, 3.6960426850612505e-9,
           2.506075454340359e-9, 1.3536795498907395e-8, 2.7439525543711344e-9,
           0.15375773443660654, 0.129483966691421, 2.040045088496055e-8,
           0.013828480733690555, 0.02934661144585244]

    w9t = [1.466695873302529e-7, 3.030648146210805e-7, 0.19375781127591588,
           1.6637649343284934e-7, 0.2832099847277843, 5.104932050860402e-8,
           5.2445196644840765e-6, 0.1384885175894236, 1.3870331272656116e-7,
           1.208492676797603e-7, 0.09626608900274275, 4.976799481171291e-8,
           3.2072330928261895e-8, 1.655953465371966e-7, 3.86242395082163e-8,
           0.17236377894405977, 0.11589946390661375, 2.5671450718047966e-7,
           3.443637416156945e-6, 4.196909164160141e-6]

    w10t = [1.8716937835303494e-8, 1.9973382366074817e-8, 2.776360025909665e-8,
            2.281819139394947e-8, 8.289532590513477e-5, 8.757629733885615e-9,
            0.9999167477418127, 1.3988967091471027e-8, 2.096133875021668e-8,
            1.5100556123300817e-8, 1.34498122214326e-8, 9.250957345881481e-9,
            6.63179929323081e-9, 1.1194609458623365e-8, 6.896037268319963e-9,
            8.087082879543003e-8, 2.6353358186238957e-8, 1.4507752106925198e-8,
            2.1915001477455265e-8, 1.7781522466660712e-8]

    w11t = [8.797277287891711e-10, 8.498129353717714e-10, 6.824691436540582e-10,
            7.743725565481004e-10, 0.8503246140759131, 1.0567063810003947e-9,
            0.14967536972693438, 1.020411288353996e-9, 8.046805752527794e-10,
            1.0120065952513828e-9, 1.04334502628843e-9, 1.1116587514978695e-9,
            1.2197894881070559e-9, 1.0806269064942662e-9, 1.1470587981672498e-9,
            2.991041511686047e-10, 5.711981170907676e-10, 1.0013030396837345e-9,
            7.401985123156343e-10, 9.026824699621393e-10]

    w12t = [2.8822789773616786e-9, 3.0812211217073126e-9, 3.663003850269969e-9,
            3.3556987149655877e-9, 0.8533950923341356, 1.4246550456641444e-9,
            0.14660486019808447, 2.122487167835777e-9, 3.2291368199463732e-9,
            2.187508997583982e-9, 1.983764682537369e-9, 1.3370712605579514e-9,
            1.0066426618965392e-9, 1.690454368407241e-9, 9.83509727143943e-10,
            6.121774341821932e-9, 4.098731346923496e-9, 2.25668425689169e-9,
            3.4368233394079237e-9, 2.606333176196267e-9]

    @test isapprox(w1.weights, w1t)
    @test isapprox(w2.weights, w2t)
    @test isapprox(w3.weights, w3t)
    @test isapprox(w2.weights, w3.weights, rtol = 1e-6)
    @test isapprox(w4.weights, w4t)
    @test isapprox(w5.weights, w5t)
    @test isapprox(w6.weights, w6t, rtol = 1e-6)
    @test isapprox(w5.weights, w6.weights, rtol = 1e-4)
    @test isapprox(w7.weights, w7t)
    @test isapprox(w8.weights, w8t, rtol = 1e-6)
    @test isapprox(w9.weights, w9t, rtol = 1e-3)
    @test isapprox(w8.weights, w9.weights, rtol = 2e-1)
    @test isapprox(w10.weights, w10t)
    @test isapprox(w11.weights, w11t)
    @test isapprox(w12.weights, w12t, rtol = 1e-5)
    @test isapprox(w11.weights, w12.weights, rtol = 1e-2)
    @test isapprox(w13.weights, w7.weights, rtol = 1e-5)
    @test isapprox(w14.weights, w8.weights, rtol = 1e-1)
    @test isapprox(w15.weights, w9.weights, rtol = 1e-1)
    @test isapprox(w16.weights, w7.weights, rtol = 1e-4)
    @test isapprox(w17.weights, w8.weights, rtol = 1e-1)
    @test isapprox(w18.weights, w9.weights, rtol = 1e-1)
    @test isapprox(w13.weights, w16.weights, rtol = 1e-4)
    @test isapprox(w14.weights, w17.weights, rtol = 1e-3)
    @test isapprox(w15.weights, w18.weights, rtol = 1e-3)
    @test isapprox(w19.weights, w1.weights, rtol = 1e-2)
end

@testset "$(:Classic), $(:Trad), $(:EDaR)" begin
    portfolio = Portfolio(; prices = prices,
                          solvers = OrderedDict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                                  :params => Dict("verbose" => false,
                                                                                  "max_step_fraction" => 0.75))))
    asset_statistics!(portfolio)

    rm = :EDaR
    opt = OptimiseOpt(; rf = rf, l = l, class = :Classic, type = :Trad, rm = rm,
                      obj = :Min_Risk, kelly = :None)

    w1 = optimise!(portfolio, opt)
    risk1 = calc_risk(portfolio; type = :Trad, rm = rm, rf = rf)
    opt.kelly = :Approx
    w2 = optimise!(portfolio, opt)
    opt.kelly = :Exact
    w3 = optimise!(portfolio, opt)
    opt.obj = :Utility
    opt.kelly = :None
    w4 = optimise!(portfolio, opt)
    opt.kelly = :Approx
    w5 = optimise!(portfolio, opt)
    opt.kelly = :Exact
    w6 = optimise!(portfolio, opt)
    opt.obj = :Sharpe
    opt.kelly = :None
    w7 = optimise!(portfolio, opt)
    risk7 = calc_risk(portfolio; type = :Trad, rm = rm, rf = rf)
    ret7 = dot(portfolio.mu, w7.weights)
    opt.kelly = :Approx
    w8 = optimise!(portfolio, opt)
    risk8 = calc_risk(portfolio; type = :Trad, rm = rm, rf = rf)
    ret8 = dot(portfolio.mu, w8.weights)
    opt.kelly = :Exact
    w9 = optimise!(portfolio, opt)
    risk9 = calc_risk(portfolio; type = :Trad, rm = rm, rf = rf)
    ret9 = dot(portfolio.mu, w9.weights)
    opt.obj = :Max_Ret
    opt.kelly = :None
    w10 = optimise!(portfolio, opt)
    opt.kelly = :Approx
    w11 = optimise!(portfolio, opt)
    opt.kelly = :Exact
    w12 = optimise!(portfolio, opt)
    setproperty!(portfolio, :mu_l, ret7)
    opt.obj = :Min_Risk
    opt.kelly = :None
    w13 = optimise!(portfolio, opt)
    setproperty!(portfolio, :mu_l, ret8)
    w14 = optimise!(portfolio, opt)
    setproperty!(portfolio, :mu_l, ret9)
    w15 = optimise!(portfolio, opt)
    setproperty!(portfolio, :mu_l, Inf)
    rmf = Symbol(lowercase(string(rm)) * "_u")
    setproperty!(portfolio, rmf, risk7)
    opt.obj = :Max_Ret
    opt.kelly = :None
    w16 = optimise!(portfolio, opt)
    setproperty!(portfolio, rmf, risk8)
    w17 = optimise!(portfolio, opt)
    setproperty!(portfolio, rmf, risk9)
    w18 = optimise!(portfolio, opt)
    setproperty!(portfolio, rmf, risk1 + 1e-4 * risk1)
    opt.obj = :Sharpe
    w19 = optimise!(portfolio, opt)
    setproperty!(portfolio, rmf, Inf)

    w1t = [0.03767983578726278, 1.034587023018692e-8, 3.065960787549633e-9,
           1.2796160190708413e-9, 0.015066695491251836, 1.797630143233225e-9,
           3.44637525187613e-10, 0.037499932108150544, 2.222787738100075e-9,
           2.363844610555352e-9, 0.4198553467362975, 3.1381609433203413e-9,
           0.013791306408964418, 3.905273319087641e-9, 2.0611297291823003e-9,
           0.13380750692888574, 0.20616056779866462, 0.0373307129680977,
           5.2198171518340995e-9, 0.09880806002769674]

    w2t = [0.03767686608403224, 7.1969529700714165e-9, 2.194093700979918e-9,
           8.919555809888013e-10, 0.015072822101735576, 1.2476055178922415e-9,
           2.423497456693741e-10, 0.03750270701181329, 1.5476053376142818e-9,
           1.6408861795347482e-9, 0.4198534460319029, 2.158307662576109e-9,
           0.013791605988109662, 2.709652966379883e-9, 1.4218114242082266e-9,
           0.13380987412888742, 0.20615034515153494, 0.03733723426179557,
           3.680279248512641e-9, 0.0988050743086882]

    w3t = [0.03767849698103619, 2.1468116914137282e-10, 3.918589887637564e-10,
           3.737414546488954e-11, 0.015071058490491764, 4.4614172881236245e-11,
           1.453658523203974e-10, 0.03749882025991621, 5.415043561244281e-11,
           5.790349126317654e-11, 0.41985663254475397, 7.062206515245646e-11,
           0.013791835199234905, 9.168387751420108e-11, 4.754583153661536e-11,
           0.13380928315627105, 0.2061518508577956, 0.03733639451629273,
           1.3130765376882578e-10, 0.0988056267070999]

    w4t = [0.03711637964446376, 9.635614299658612e-9, 2.7643293798921003e-9,
           1.1357557874321587e-9, 0.016055774203689446, 1.5859520752670138e-9,
           2.836491777874425e-10, 0.037612060634633016, 2.003325601667589e-9,
           2.113723938506527e-9, 0.41979100585813817, 2.7035448810053766e-9,
           0.013864561907214181, 3.56165225582795e-9, 1.8026957481409973e-9,
           0.1341161997979314, 0.2050801737363802, 0.037871275235355856,
           4.876620229708579e-9, 0.09849253651533062]
    w5t = [0.037100852546472454, 2.859828224208465e-9, 1.0181967089884713e-9,
           3.8866744744221173e-10, 0.016075512991623363, 5.36263573310557e-10,
           1.3215730721531076e-10, 0.03763426942833204, 6.589963844074449e-10,
           7.029068546811892e-10, 0.4197811067598082, 8.986903125450899e-10,
           0.013861284191330274, 1.123889290809053e-9, 5.933712867833036e-10,
           0.1341228522681763, 0.20502753855204256, 0.0379053384328581,
           1.497177745019913e-9, 0.09849123441921166]

    w6t = [0.03710000096546551, 6.777859448908837e-11, 7.278559975936294e-11,
           1.1342354532619916e-11, 0.016072946413646264, 1.3914889066854262e-11,
           4.343509884700505e-11, 0.03763427489983173, 1.7047191083277878e-11,
           1.824846620132219e-11, 0.41978072526963717, 2.1595757812650208e-11,
           0.013861415827868326, 2.8773973224118135e-11, 1.4630238435788975e-11,
           0.13412031543424935, 0.20503244600823683, 0.03790534545633962,
           4.1391898007306127e-11, 0.09849252937378115]

    w7t = [1.304559190511154e-8, 1.4274870449537091e-8, 0.10584952164876094,
           4.777581874154085e-9, 0.2712724249718687, 2.6358497952069248e-9,
           6.969956590749234e-9, 0.08852141238814072, 4.063081674947575e-9,
           4.421055495125228e-9, 0.2612419804947476, 2.5287249385311452e-9,
           1.4969217849283207e-9, 7.16326209776733e-9, 2.295589654298989e-9,
           0.2731145017235489, 2.8480257360987228e-8, 1.0590860111130809e-8,
           1.0028476627527744e-8, 4.600085295556881e-8]

    w8t = [1.3943194782982129e-8, 1.7380194629758627e-8, 0.09397026051518141,
           4.750726953917328e-9, 0.26905909762191393, 2.542463445177757e-9,
           4.919168536415354e-9, 0.084967620081315, 4.463883069906198e-9,
           4.6427951522823615e-9, 0.30747518818826547, 2.369643587139357e-9,
           1.489251423540626e-9, 7.458249187647291e-9, 2.07596702778992e-9,
           0.24452733999580603, 9.005841127000335e-8, 1.4264772787038934e-8,
           1.249657555740849e-8, 3.1074222074783046e-7]

    w9t = [4.367744261984825e-8, 5.6718772989180197e-8, 0.11563996519587906,
           1.756513510986065e-8, 0.2785793627765359, 9.133100823355754e-9,
           3.143157709942596e-7, 0.09130776922495325, 1.4825557029393895e-8,
           1.5838330246351703e-8, 0.26201845861216916, 8.831154139781117e-9,
           5.168589731127955e-9, 2.575161799194872e-8, 8.271663125703772e-9,
           0.25245356301361194, 1.1952132397471034e-7, 3.823748029881371e-8,
           3.801151280037912e-8, 1.653093989024109e-7]

    w10t = [3.6548095821590484e-8, 3.86135372002626e-8, 1.0055208096181307e-7,
            4.3605962550967795e-8, 0.00013523507157440338, 1.6797142080480717e-8,
            0.9998640245676909, 2.8629149668365944e-8, 4.0381237337227926e-8,
            2.915470146368117e-8, 2.6691037290402214e-8, 1.7760355413360968e-8,
            1.2662589272860673e-8, 2.1702773589086086e-8, 1.3149002745996455e-8,
            1.581362795110165e-7, 5.0792381184707626e-8, 2.8089443453424042e-8,
            4.2420844231628124e-8, 3.4674120923247266e-8]

    w11t = [7.185450461102519e-11, 6.851670170644865e-11, 5.955264428870439e-11,
            6.374430917408214e-11, 0.8503244542941133, 8.633998243068413e-11,
            0.14967554440704783, 7.945903730098282e-11, 6.591794801321178e-11,
            7.78312035011345e-11, 8.048309720883013e-11, 8.709770959530669e-11,
            9.086616695507754e-11, 8.431811658389171e-11, 9.052179847698563e-11,
            2.638129192671608e-11, 5.279762852820883e-11, 7.827108286600944e-11,
            6.250821359030034e-11, 7.237737587245349e-11]

    w12t = [2.295570291520812e-9, 2.5128468051443085e-9, 3.0644428614202233e-9,
            2.7596171784190625e-9, 0.85339501038672, 1.1038154660045194e-9,
            0.14660495067330162, 1.7112237641254399e-9, 2.6606577810977123e-9,
            1.7979965758303158e-9, 1.6160885219086092e-9, 1.0627374395028326e-9,
            7.911545048410588e-10, 1.3427358190857723e-9, 7.86444152051084e-10,
            5.1744053906025614e-9, 3.438314547730402e-9, 1.8110289028783654e-9,
            2.844260675789349e-9, 2.166637608289391e-9]

    @test isapprox(w1.weights, w1t, rtol = 1e-4)
    @test isapprox(w2.weights, w2t, rtol = 1e-4)
    @test isapprox(w3.weights, w3t, rtol = 1e-5)
    @test isapprox(w2.weights, w3.weights, rtol = 1e-4)
    @test isapprox(w4.weights, w4t, rtol = 1e-5)
    @test isapprox(w5.weights, w5t, rtol = 1e-4)
    @test isapprox(w6.weights, w6t, rtol = 1e-6)
    @test isapprox(w5.weights, w6.weights, rtol = 1e-4)
    @test isapprox(w7.weights, w7t, rtol = 1e-6)
    @test isapprox(w8.weights, w8t, rtol = 1e-5)
    @test isapprox(w9.weights, w9t, rtol = 1e-4)
    @test isapprox(w8.weights, w9.weights, rtol = 2e-1)
    @test isapprox(w10.weights, w10t)
    @test isapprox(w11.weights, w11t)
    @test isapprox(w12.weights, w12t, rtol = 1e-6)
    @test isapprox(w11.weights, w12.weights, rtol = 1e-2)
    @test isapprox(w13.weights, w7.weights, rtol = 1e-4)
    @test isapprox(w14.weights, w8.weights, rtol = 1e-1)
    @test isapprox(w15.weights, w9.weights, rtol = 1e-1)
    @test isapprox(w16.weights, w7.weights, rtol = 1e-4)
    @test isapprox(w17.weights, w8.weights, rtol = 1e-1)
    @test isapprox(w18.weights, w9.weights, rtol = 1e-1)
    @test isapprox(w13.weights, w16.weights, rtol = 1e-4)
    @test isapprox(w14.weights, w17.weights, rtol = 1e-2)
    @test isapprox(w15.weights, w18.weights, rtol = 1e-2)
    @test isapprox(w19.weights, w1.weights, rtol = 1e-1)
end

@testset "$(:Classic), $(:Trad), $(:RDaR)" begin
    portfolio = Portfolio(; prices = prices,
                          solvers = OrderedDict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                                  :params => Dict("verbose" => false,
                                                                                  "max_step_fraction" => 0.75))))
    asset_statistics!(portfolio)

    rm = :RDaR
    opt = OptimiseOpt(; rf = rf, l = l, class = :Classic, type = :Trad, rm = rm,
                      obj = :Min_Risk, kelly = :None)

    w1 = optimise!(portfolio, opt)
    risk1 = calc_risk(portfolio; type = :Trad, rm = rm, rf = rf)
    opt.kelly = :Approx
    w2 = optimise!(portfolio, opt)
    opt.kelly = :Exact
    w3 = optimise!(portfolio, opt)
    opt.obj = :Utility
    opt.kelly = :None
    w4 = optimise!(portfolio, opt)
    opt.kelly = :Approx
    w5 = optimise!(portfolio, opt)
    opt.kelly = :Exact
    w6 = optimise!(portfolio, opt)
    opt.obj = :Sharpe
    opt.kelly = :None
    w7 = optimise!(portfolio, opt)
    risk7 = calc_risk(portfolio; type = :Trad, rm = rm, rf = rf)
    ret7 = dot(portfolio.mu, w7.weights)
    opt.kelly = :Approx
    w8 = optimise!(portfolio, opt)
    risk8 = calc_risk(portfolio; type = :Trad, rm = rm, rf = rf)
    ret8 = dot(portfolio.mu, w8.weights)
    opt.kelly = :Exact
    w9 = optimise!(portfolio, opt)
    risk9 = calc_risk(portfolio; type = :Trad, rm = rm, rf = rf)
    ret9 = dot(portfolio.mu, w9.weights)
    opt.obj = :Max_Ret
    opt.kelly = :None
    w10 = optimise!(portfolio, opt)
    opt.kelly = :Approx
    w11 = optimise!(portfolio, opt)
    opt.kelly = :Exact
    w12 = optimise!(portfolio, opt)
    setproperty!(portfolio, :mu_l, ret7)
    opt.obj = :Min_Risk
    opt.kelly = :None
    w13 = optimise!(portfolio, opt)
    setproperty!(portfolio, :mu_l, ret8)
    w14 = optimise!(portfolio, opt)
    setproperty!(portfolio, :mu_l, ret9)
    w15 = optimise!(portfolio, opt)
    setproperty!(portfolio, :mu_l, Inf)
    rmf = Symbol(lowercase(string(rm)) * "_u")
    setproperty!(portfolio, rmf, risk7)
    opt.obj = :Max_Ret
    opt.kelly = :None
    w16 = optimise!(portfolio, opt)
    setproperty!(portfolio, rmf, risk8)
    w17 = optimise!(portfolio, opt)
    setproperty!(portfolio, rmf, risk9)
    w18 = optimise!(portfolio, opt)
    setproperty!(portfolio, rmf, risk1 + 1e-6 * risk1)
    opt.obj = :Sharpe
    w19 = optimise!(portfolio, opt)
    setproperty!(portfolio, rmf, Inf)

    w1t = [0.05620460167041587, 3.5674808843719795e-10, 3.271082179464376e-9,
           2.490299621074662e-10, 0.029959567746799393, 4.423376540870037e-10,
           1.6528267282388329e-9, 0.020668244367877297, 3.187332097652016e-10,
           6.107384735920315e-10, 0.44148524192715666, 4.898409268534613e-10,
           0.021757933169939566, 1.794065278475042e-10, 1.4097693073321668e-10,
           0.14345061711918758, 0.15965607631146994, 0.06193469673331933,
           9.916624953475415e-10, 0.06488301225045126]

    w2t = [0.05620456421651178, 2.638399683779679e-10, 1.4744208537848776e-9,
           1.6368303893099755e-10, 0.02995912528274963, 3.581526395436185e-10,
           6.789514969342704e-10, 0.020667970360103472, 2.3784712341326505e-10,
           4.806308939931586e-10, 0.44148518555838034, 3.9923438601136615e-10,
           0.02175800975188342, 1.2416746582919976e-10, 1.1142780686875671e-10,
           0.143450779133829, 0.15965729515495541, 0.06193194296607907,
           6.710905964208492e-10, 0.06488512261206149]

    w3t = [0.05620462178848656, 3.458608272530347e-11, 2.388243069009394e-10,
           2.312759187678553e-11, 0.02995917820866244, 4.215789277568693e-11,
           1.0583089715868429e-10, 0.02066799627023981, 3.061420093807828e-11,
           5.3847498915330365e-11, 0.4414850516423621, 4.624568806692328e-11,
           0.02175797625715154, 2.0132211987961882e-11, 1.8489960530634206e-11,
           0.14345091837132742, 0.1596571166046852, 0.061932042676021065,
           8.292947172809268e-11, 0.06488509748427795]

    w4t = [0.05663668356281749, 4.4949054199031623e-10, 4.045654101089874e-9,
           3.069248155336226e-10, 0.030740877018997373, 5.42297568076841e-10,
           2.394136391926208e-9, 0.02093665900542826, 3.938689196722233e-10,
           7.55961628824867e-10, 0.4409673103487479, 5.946730137499104e-10,
           0.02151852528028207, 2.073889971677028e-10, 1.61467916739268e-10,
           0.14456220296720357, 0.15845833782620006, 0.062075484476346456,
           1.2756349574715012e-9, 0.06410390838647798]

    w5t = [0.05662335935829186, 2.1234400285437444e-10, 8.718157963696351e-10,
           1.2792813945082466e-10, 0.030730664412930722, 2.8045828112716073e-10,
           2.9401609598595923e-10, 0.020948208687072558, 1.8698517419949056e-10,
           3.6969262035856406e-10, 0.4409549980619708, 3.1033067305774504e-10,
           0.0215219614563657, 1.0286107090484641e-10, 9.488852598320492e-11,
           0.144530352420999, 0.15849653978157696, 0.06213320908675631,
           5.014577869329655e-10, 0.06406070338125802]

    w6t = [0.056623286488486815, 1.0057871595562474e-11, 8.270865827083622e-11,
           6.506749671621905e-12, 0.030730278091613387, 1.1517652544168507e-11,
           3.4762553287097994e-11, 0.02094789516748728, 8.50939507991036e-12,
           1.4577445381644612e-11, 0.4409552897479191, 1.2314188102883017e-11,
           0.021522050704693773, 5.558771208618245e-12, 4.940556600817432e-12,
           0.14453020436659747, 0.1584969814905925, 0.062132002396094746,
           2.3904127481772756e-11, 0.06406201133115703]

    w7t = [1.5358710124577505e-8, 1.0760310357592999e-8, 0.04269897219126458,
           4.343578842822139e-9, 0.27528464407422565, 2.4700494803767716e-9,
           3.976356782955665e-9, 0.09149783712958341, 3.4628322806127026e-9,
           3.970496960583462e-9, 0.30045494891686153, 2.060496836986878e-9,
           1.3228630085825473e-9, 6.000633932212436e-9, 1.9326892678486254e-9,
           0.2900634878372457, 1.7309744832129924e-8, 9.495284979535135e-9,
           7.678009111803233e-9, 1.9708762273843586e-8]

    w8t = [1.962834925887632e-8, 8.964810817889151e-9, 0.04795062114752034,
           4.007510010366182e-9, 0.27537573561834244, 2.2974010177070404e-9,
           4.523630695305993e-9, 0.08982849401246491, 3.34286827405443e-9,
           3.804218809837961e-9, 0.31413374419736845, 1.890461385247909e-9,
           1.3017401141088974e-9, 4.697745484191045e-9, 1.6483568156094035e-9,
           0.2727112836444813, 2.4292185956230826e-8, 1.1313244102112346e-8,
           8.025516589971239e-9, 2.164178342340015e-8]

    w9t = [8.208693176935992e-8, 7.805511432248174e-8, 0.06303870176606124,
           2.5558474689990926e-8, 0.27838132788267295, 1.3738612784045754e-8,
           2.1381718104006003e-8, 0.08134895683958954, 2.0204995619255107e-8,
           2.3210640864143486e-8, 0.303005884358898, 1.1408413762027243e-8,
           7.3305856588301025e-9, 3.724756285820412e-8, 1.1379294154467483e-8,
           0.2742244717110522, 1.1271559001513796e-7, 5.6670245896529826e-8,
           4.665724977908473e-8, 1.097962958695238e-7]

    w10t = [1.8157885612868624e-8, 1.9512929617724308e-8, 4.4301238362163684e-8,
            2.216719261082652e-8, 6.368492940827653e-5, 8.456518437934178e-9,
            0.9999359109050406, 1.437353625745112e-8, 2.0185902360410413e-8,
            1.4608360376550806e-8, 1.329298842628497e-8, 8.935597861543337e-9,
            6.428257584828559e-9, 1.0872696388136073e-8, 6.6347497266490586e-9,
            1.1646296907815419e-7, 2.6820262300752665e-8, 1.4057969354280807e-8,
            2.1459905434908744e-8, 1.743659126901538e-8]

    w11t = [3.8937837964339e-11, 3.725947416965801e-11, 3.1122280007927656e-11,
            3.37338993027047e-11, 0.8503244492606654, 4.707487325539199e-11,
            0.14967555003705155, 4.4141449409532015e-11, 3.567005049127002e-11,
            4.3301567001333876e-11, 4.4789729324825794e-11, 4.8033229316058155e-11,
            5.0231861342749336e-11, 4.69121482177809e-11, 5.0018915242051374e-11,
            8.98917808811706e-12, 2.5997500023601423e-11, 4.337255438905853e-11,
            3.313292394128641e-11, 3.9563780671640486e-11]

    w12t = [2.0604485455094505e-9, 2.436927344747036e-9, 2.9731507459665057e-9,
            2.6886066881519344e-9, 0.8533947466568395, 9.35084945958532e-10,
            0.1466052148922998, 1.6815481315556726e-9, 2.651554266117528e-9,
            1.8467409944158816e-9, 1.6332134928216206e-9, 1.0587762371231237e-9,
            8.427832371402091e-10, 1.2911788038560076e-9, 8.123908483180113e-10,
            5.164050278245789e-9, 3.4918973168527626e-9, 1.7672451300781401e-9,
            2.897337529574461e-9, 2.21792617295559e-9]

    @test isapprox(w1.weights, w1t, rtol = 1e-5)
    @test isapprox(w2.weights, w2t, rtol = 1e-4)
    @test isapprox(w3.weights, w3t, rtol = 1.0e-6)
    @test isapprox(w2.weights, w3.weights, rtol = 1e-4)
    @test isapprox(w4.weights, w4t, rtol = 1.0e-5)
    @test isapprox(w5.weights, w5t, rtol = 1.0e-5)
    @test isapprox(w6.weights, w6t, rtol = 1.0e-6)
    @test isapprox(w5.weights, w6.weights, rtol = 1e-5)
    @test isapprox(w7.weights, w7t)
    @test isapprox(w8.weights, w8t, rtol = 1.0e-4)
    @test isapprox(w9.weights, w9t, rtol = 0.1)
    @test isapprox(w8.weights, w9.weights, rtol = 1e-1)
    @test isapprox(w10.weights, w10t, rtol = 1.0e-7)
    @test isapprox(w11.weights, w11t)
    @test isapprox(w12.weights, w12t, rtol = 1.0e-6)
    @test isapprox(w11.weights, w12.weights, rtol = 1e-2)
    @test isapprox(w13.weights, w7.weights, rtol = 5e-5)
    @test isapprox(w14.weights, w8.weights, rtol = 1e-1)
    @test isapprox(w15.weights, w9.weights, rtol = 1e-1)
    @test isapprox(w16.weights, w7.weights, rtol = 5e-5)
    @test isapprox(w17.weights, w8.weights, rtol = 1e-1)
    @test isapprox(w18.weights, w9.weights, rtol = 1e-1)
    @test isapprox(w13.weights, w16.weights, rtol = 1e-5)
    @test isapprox(w14.weights, w17.weights, rtol = 1e-2)
    @test isapprox(w15.weights, w18.weights, rtol = 1e-2)
    @test isapprox(w19.weights, w1.weights, rtol = 1e-2)
end

@testset "$(:Classic), $(:Trad), Full $(:Kurt)" begin
    portfolio = Portfolio(; prices = prices,
                          solvers = OrderedDict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                                  :params => Dict("verbose" => false,
                                                                                  "max_step_fraction" => 0.75))))
    asset_statistics!(portfolio)

    rm = :Kurt
    opt = OptimiseOpt(; rf = rf, l = l, class = :Classic, type = :Trad, rm = rm,
                      obj = :Min_Risk, kelly = :None)

    w1 = optimise!(portfolio, opt)
    risk1 = calc_risk(portfolio; type = :Trad, rm = rm, rf = rf)
    opt.kelly = :Approx
    w2 = optimise!(portfolio, opt)
    opt.kelly = :Exact
    w3 = optimise!(portfolio, opt)
    opt.obj = :Utility
    opt.kelly = :None
    w4 = optimise!(portfolio, opt)
    opt.kelly = :Approx
    w5 = optimise!(portfolio, opt)
    opt.kelly = :Exact
    w6 = optimise!(portfolio, opt)
    opt.obj = :Sharpe
    opt.kelly = :None
    w7 = optimise!(portfolio, opt)
    risk7 = calc_risk(portfolio; type = :Trad, rm = rm, rf = rf)
    ret7 = dot(portfolio.mu, w7.weights)
    opt.kelly = :Approx
    w8 = optimise!(portfolio, opt)
    risk8 = calc_risk(portfolio; type = :Trad, rm = rm, rf = rf)
    ret8 = dot(portfolio.mu, w8.weights)
    opt.obj = :Max_Ret
    opt.kelly = :None
    w10 = optimise!(portfolio, opt)
    opt.kelly = :Approx
    w11 = optimise!(portfolio, opt)
    setproperty!(portfolio, :mu_l, ret7)
    opt.obj = :Min_Risk
    opt.kelly = :None
    w13 = optimise!(portfolio, opt)
    setproperty!(portfolio, :mu_l, ret8)
    w14 = optimise!(portfolio, opt)
    setproperty!(portfolio, :mu_l, Inf)
    rmf = Symbol(lowercase(string(rm)) * "_u")
    setproperty!(portfolio, rmf, risk7)
    opt.obj = :Max_Ret
    w16 = optimise!(portfolio, opt)
    setproperty!(portfolio, rmf, risk8)
    w17 = optimise!(portfolio, opt)
    setproperty!(portfolio, rmf, risk1 + 1e-4 * risk1)
    opt.obj = :Sharpe
    w18 = optimise!(portfolio, opt)
    setproperty!(portfolio, rmf, Inf)

    w1t = [1.2207374662075011e-8, 0.03971088565985921, 1.5462358749843204e-8,
           0.07516228373134295, 1.5932976249989572e-8, 0.011224033461961283,
           8.430690935829477e-9, 0.128815906121176, 8.394423005573743e-9,
           3.29141345013612e-8, 0.3694591286005053, 8.438871933381344e-9,
           4.8696579139321915e-9, 0.03900937526506312, 2.101793927311059e-8,
           0.02018194601668498, 4.585567635948199e-8, 0.18478029997048878,
           1.0811349893382834e-8, 0.13165595683746487]

    w2t = [3.3462134865962766e-8, 0.03970968579442602, 4.2432207376996394e-8,
           0.07516162014582399, 4.375804802956574e-8, 0.0112260536265873,
           2.3416646784015625e-8, 0.1288161872960105, 2.3088102301512903e-8,
           9.03508386966651e-8, 0.36946012385409505, 2.3265192365737076e-8,
           1.3532565442693126e-8, 0.039008405416776956, 5.8146463321969216e-8,
           0.02018128366591476, 1.2572063742436965e-7, 0.18478011330737115,
           2.9674042202445336e-8, 0.1316560200461156]

    w3t = [1.1191171763430908e-7, 0.039710753795231014, 1.289768725651975e-7,
           0.07516129212867434, 1.29178570402263e-7, 0.011225943329681175,
           6.84179579835758e-8, 0.12881622734316683, 7.243989136052092e-8,
           2.6021795337900513e-7, 0.36945958980838456, 7.765620996826182e-8,
           4.6086159408315595e-8, 0.03900927774930172, 1.7821385144435033e-7,
           0.020181734598398574, 3.4379020180745197e-7, 0.1847790413543707,
           9.215463437879879e-8, 0.13165463084877063]

    w4t = [6.488257629546953e-9, 4.212509532760617e-8, 0.012824070122186799,
           0.02084004300406732, 0.3535763653540333, 1.970991126948177e-9,
           0.0863510565350794, 4.974079450006908e-7, 1.1571650403744032e-8,
           9.868829125865894e-9, 2.0167487739863782e-8, 1.799041212079837e-9,
           1.1714958080503707e-9, 3.6613705005986247e-9, 1.1209037353862583e-9,
           0.13915737487846996, 0.3050735943714213, 2.243500811419314e-8,
           0.08217685583635567, 2.0110310483841686e-8]

    w5t = [5.6837710346497385e-9, 4.956360824743668e-8, 0.010938887497562425,
           0.02251547139088467, 0.33660050751160714, 1.7156396316106738e-9,
           0.07880524840944743, 0.044375992812347095, 1.0111038021377374e-8,
           9.163868118016054e-9, 3.403353149644027e-8, 1.499572162395962e-9,
           9.999749990487708e-10, 3.310461670486626e-9, 9.638229102965985e-10,
           0.13238856008120647, 0.2908997454166324, 2.9820895853988096e-8,
           0.08347541834595142, 2.1668176794356163e-8]

    w6t = [5.988176293166298e-9, 3.9132432114588405e-8, 0.010740572705887724,
           0.022447064856100554, 0.33659134675090197, 1.9224780725630647e-9,
           0.07876384138833782, 0.044884451761521914, 9.54336070376902e-9,
           8.688563886208845e-9, 2.9408326101688127e-8, 1.5729688657188628e-9,
           1.024911612199555e-9, 3.539143533509802e-9, 9.9852325614194e-10,
           0.13242338028357242, 0.29082173354357455, 2.6496331661406445e-8,
           0.08332746169093166, 1.870395525974221e-8]

    w7t = [3.684471815229458e-8, 0.007745692671391506, 7.656541334757149e-7,
           0.06255101978519823, 0.2159122600775436, 1.164556494545623e-8,
           0.04812416128754939, 0.1265864287977592, 7.026046774095028e-8,
           6.686781031886239e-8, 0.19709510148146508, 1.0044327356115857e-8,
           5.628678247009387e-9, 2.6809145191775967e-8, 6.040342473700855e-9,
           0.0919344514120299, 0.2084432227195518, 0.03380290565348507,
           0.007797443469422402, 6.3128494160389445e-6]

    w8t = [1.175528742211673e-8, 0.011976076948066442, 1.146873660115615e-7,
           0.06099287570402978, 0.1930026726196843, 4.37147463002243e-9,
           0.04149627803275983, 0.12397068314603107, 2.216723739151983e-8,
           2.1711712355060224e-8, 0.22978042682357938, 3.6591585583072032e-9,
           2.22190046457137e-9, 1.0066322101408625e-8, 2.427798364557446e-9,
           0.08320878878970762, 0.18057603110772905, 0.059101699159070174,
           4.457808427763218e-7, 0.01589382882024229]

    w10t = [5.931184119367502e-9, 6.317449636723593e-9, 7.964568609408255e-9,
            7.2214774823451605e-9, 8.520484030306939e-7, 2.8273843121043952e-9,
            0.9999990507158887, 4.3580731359089834e-9, 6.7051952832165315e-9,
            4.775594065770581e-9, 4.227574060868775e-9, 2.9808106334097823e-9,
            2.180603250031813e-9, 3.5647655852469804e-9, 2.254720160512395e-9,
            1.0485825514734632e-8, 8.30810820610955e-9, 4.577363947745535e-9,
            6.9469366036575476e-9, 5.608073622713495e-9]

    w11t = [3.928665478740454e-12, 4.988144568107438e-12, 7.843111455508564e-12,
            6.441290907069037e-12, 0.8503244345703825, 1.936080209775467e-12,
            0.149675565351375, 1.1551289353140768e-12, 5.971245234638931e-12,
            2.006245627095128e-12, 9.5243233235081e-13, 1.4183849506808721e-12,
            2.7164170996670568e-12, 3.905314670872243e-13, 2.725358028455437e-12,
            1.464656205251539e-11, 9.166048835993442e-12, 1.625528193550047e-12,
            6.6448418029847165e-12, 3.6862487579985345e-12]

    @test isapprox(w1.weights, w1t, rtol = 1.0e-8)
    @test isapprox(w2.weights, w2t, rtol = 1.0e-8)
    @test isapprox(w3.weights, w3t, rtol = 1.0e-8)
    @test isapprox(w2.weights, w3.weights, rtol = 1e-5)
    @test isapprox(w4.weights, w4t, rtol = 1.0e-8)
    @test isapprox(w5.weights, w5t, rtol = 1.0e-8)
    @test isapprox(w6.weights, w6t, rtol = 0.001)
    @test isapprox(w5.weights, w6.weights, rtol = 1e-2)
    @test isapprox(w7.weights, w7t, rtol = 1.0e-7)
    @test isapprox(w8.weights, w8t, rtol = 1.0e-6)
    @test isapprox(w10.weights, w10t, rtol = 1.0e-9)
    @test isapprox(w11.weights, w11t, rtol = 1.0e-10)
    @test isapprox(w13.weights, w7.weights, rtol = 1e-4)
    @test isapprox(w14.weights, w8.weights, rtol = 1e-1)
    @test isapprox(w16.weights, w7.weights, rtol = 1e-1)
    @test isapprox(w17.weights, w8.weights, rtol = 1e-1)
    @test isapprox(w13.weights, w16.weights, rtol = 1e-1)
    @test isapprox(w14.weights, w17.weights, rtol = 1e-1)
    @test isapprox(w18.weights, w1.weights, rtol = 1e-1)
end

@testset "$(:Classic), $(:Trad), Reduced $(:Kurt)" begin
    portfolio = Portfolio(; prices = prices,
                          solvers = OrderedDict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                                  :params => Dict("verbose" => false,
                                                                                  "max_step_fraction" => 0.75))),
                          max_num_assets_kurt = 1)
    asset_statistics!(portfolio)

    rm = :Kurt
    opt = OptimiseOpt(; rf = rf, l = l, class = :Classic, type = :Trad, rm = rm,
                      obj = :Min_Risk, kelly = :None)

    w1 = optimise!(portfolio, opt)
    opt.kelly = :Approx
    w2 = optimise!(portfolio, opt)
    opt.kelly = :Exact
    w3 = optimise!(portfolio, opt)
    opt.obj = :Utility
    opt.kelly = :None
    w4 = optimise!(portfolio, opt)
    opt.kelly = :Approx
    w5 = optimise!(portfolio, opt)
    opt.kelly = :Exact
    opt.obj = :Sharpe
    opt.kelly = :None
    w7 = optimise!(portfolio, opt)
    risk7 = calc_risk(portfolio; type = :Trad, rm = :Kurt, rf = rf)
    ret7 = dot(portfolio.mu, w7.weights)
    opt.obj = :Max_Ret
    opt.kelly = :None
    w10 = optimise!(portfolio, opt)
    opt.kelly = :Approx
    w11 = optimise!(portfolio, opt)
    setproperty!(portfolio, :mu_l, ret7)
    opt.obj = :Min_Risk
    opt.kelly = :None
    w13 = optimise!(portfolio, opt)
    setproperty!(portfolio, :mu_l, Inf)

    w1t = [6.258921653709612e-7, 3.6801031030531044e-6, 7.486005254331526e-7,
           0.034980344157996356, 7.917668371432098e-7, 1.2350802253901254e-6,
           3.0460582768201714e-7, 0.08046935867943861, 5.699669067592361e-7,
           8.853934346583818e-7, 0.6357270365895509, 5.979410366260909e-7,
           2.796272257944517e-7, 1.8757933336123485e-6, 5.070585868816226e-7,
           0.001277066543368786, 1.2457134330236783e-6, 0.11108722055002894,
           6.616561616213671e-7, 0.13644496428081343]

    w2t = [6.651646613846569e-7, 3.928958252714088e-6, 7.951215771629839e-7,
           0.03498130373756166, 8.407837895806342e-7, 1.315189508353321e-6,
           3.2080443032823865e-7, 0.08046930933588052, 6.034153408631512e-7,
           9.410031323705422e-7, 0.6357234132230779, 6.327667032885894e-7,
           2.9391855576182194e-7, 2.001627925583568e-6, 5.362111352460991e-7,
           0.001277889238101153, 1.3259046122088047e-6, 0.1110872373659002,
           7.019310252816386e-7, 0.13644594429882836]

    w3t = [7.981051299642011e-8, 4.105480259250049e-7, 8.863453036212545e-8,
           0.034971480825376364, 9.311920189177616e-8, 1.694283046379795e-7,
           3.577469992710369e-8, 0.08046249483334252, 6.693424513074424e-8,
           1.0588939013790742e-7, 0.6357235016751249, 7.285057471685483e-8,
           3.4257265466937416e-8, 2.3850773878472746e-7, 6.334662647470646e-8,
           0.0011915157930652035, 1.4424093006489007e-7, 0.11113298715900677,
           7.791840424179946e-8, 0.13651633845363342]

    w4t = [1.6331898672791201e-7, 4.739269614649452e-7, 8.799602729380153e-7,
           2.68281028137023e-6, 0.33250417011284233, 6.252670605108193e-8,
           0.08180392349674485, 7.710747907071424e-7, 2.963448053295334e-7,
           2.658102052584168e-7, 4.0340013760256616e-7, 5.870525244168889e-8,
           3.865053009926538e-8, 1.0933111970401291e-7, 3.72375735954745e-8,
           0.12820198647082293, 0.45748114819069624, 4.127937506828679e-7,
           1.7817988851940717e-6, 3.340386345213016e-7]

    w5t = [9.358597882700263e-8, 3.2853349227603195e-7, 6.036593171285032e-7,
           1.4009187937286994e-5, 0.3219311277450799, 3.771800717294446e-8,
           0.07599801296981785, 7.256094732777976e-6, 1.7414044343986163e-7,
           1.7420308012621616e-7, 4.800036129933735e-7, 3.3973421651095024e-8,
           2.2531129156532274e-8, 6.78475871005808e-8, 2.209024751214574e-8,
           0.1277167419936065, 0.4743249136646003, 4.0320782458402484e-7,
           5.268419453500141e-6, 2.2843063007240183e-7]

    w7t = [2.1878148272919998e-8, 1.2031998466130417e-7, 8.310823488565457e-8,
           0.04981239243380165, 0.1993030598433479, 9.893818607829968e-9,
           0.0418265246728685, 0.10466467053565787, 3.333967932286556e-8,
           4.0272858398190096e-8, 0.23960267250695264, 8.643199964337484e-9,
           4.960366925219853e-9, 2.0034588388061512e-8, 5.258074682256516e-9,
           0.0839227178720721, 0.28086714825267073, 2.4704290295415187e-7,
           1.0506999839860491e-7, 1.1406077340552796e-7]

    w10t = [4.9364659950303746e-8, 5.2763900916927184e-8, 6.809010621885798e-8,
            6.105779651359053e-8, 6.716487105118319e-6, 2.4448091827621114e-8,
            0.9999924571180464, 3.623677859414453e-8, 5.6060946409167296e-8,
            3.9533922857375574e-8, 3.521614029150337e-8, 2.5626154701481916e-8,
            1.9584959319965945e-8, 3.003684886880722e-8, 2.01226412941958e-8,
            9.369902614945842e-8, 7.159574645532308e-8, 3.798727753206114e-8,
            5.837684637511342e-8, 4.659300415145668e-8]

    w11t = [2.0892470193804777e-10, 2.8224507340252007e-10, 4.3716271915222594e-10,
            3.620316284498617e-10, 0.8503240818444223, 1.1465670506881569e-10,
            0.14967591348957518, 6.943761636530554e-11, 3.352850325645301e-10,
            1.0468382741061257e-10, 5.110600610899828e-11, 1.1040923740659723e-10,
            2.236508758681182e-10, 4.166792719541042e-11, 1.5966495841276126e-10,
            9.379277376716377e-10, 5.37800894432283e-10, 9.381810823039038e-11,
            3.873271298356599e-10, 2.0820241494720741e-10]

    @test isapprox(w1.weights, w1t)
    @test isapprox(w2.weights, w2t)
    @test isapprox(w3.weights, w3t)
    @test isapprox(w2.weights, w3.weights, rtol = 0.0005)
    @test isapprox(w4.weights, w4t)
    @test isapprox(w5.weights, w5t)
    @test isapprox(w7.weights, w7t, rtol = 1.0e-7)
    @test isapprox(w10.weights, w10t)
    @test isapprox(w11.weights, w11t)
    @test isapprox(w13.weights, w7.weights, rtol = 0.0001)
end

@testset "$(:Classic), $(:Trad), Full $(:SKurt)" begin
    portfolio = Portfolio(; prices = prices,
                          solvers = OrderedDict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                                  :params => Dict("verbose" => false,
                                                                                  "max_step_fraction" => 0.75))))
    asset_statistics!(portfolio)

    rm = :SKurt
    opt = OptimiseOpt(; rf = rf, l = l, class = :Classic, type = :Trad, rm = rm,
                      obj = :Min_Risk, kelly = :None)

    w1 = optimise!(portfolio, opt)
    risk1 = calc_risk(portfolio; type = :Trad, rm = rm, rf = rf)
    opt.kelly = :Approx
    w2 = optimise!(portfolio, opt)
    opt.kelly = :Exact
    w3 = optimise!(portfolio, opt)
    opt.obj = :Utility
    opt.kelly = :None
    w4 = optimise!(portfolio, opt)
    opt.kelly = :Approx
    w5 = optimise!(portfolio, opt)
    opt.kelly = :Exact
    w6 = optimise!(portfolio, opt)
    opt.obj = :Sharpe
    opt.kelly = :None
    w7 = optimise!(portfolio, opt)
    risk7 = calc_risk(portfolio; type = :Trad, rm = rm, rf = rf)
    ret7 = dot(portfolio.mu, w7.weights)
    opt.kelly = :Approx
    w8 = optimise!(portfolio, opt)
    risk8 = calc_risk(portfolio; type = :Trad, rm = rm, rf = rf)
    ret8 = dot(portfolio.mu, w8.weights)
    opt.obj = :Max_Ret
    opt.kelly = :None
    w10 = optimise!(portfolio, opt)
    opt.kelly = :Approx
    w11 = optimise!(portfolio, opt)
    setproperty!(portfolio, :mu_l, ret7)
    opt.obj = :Min_Risk
    opt.kelly = :None
    w13 = optimise!(portfolio, opt)
    setproperty!(portfolio, :mu_l, ret8)
    w14 = optimise!(portfolio, opt)
    setproperty!(portfolio, :mu_l, Inf)
    rmf = Symbol(lowercase(string(rm)) * "_u")
    setproperty!(portfolio, rmf, risk7)
    opt.obj = :Max_Ret
    w16 = optimise!(portfolio, opt)
    setproperty!(portfolio, rmf, risk8)
    w17 = optimise!(portfolio, opt)
    setproperty!(portfolio, rmf, risk1 + 4e-2 * risk1)
    opt.obj = :Sharpe
    w18 = optimise!(portfolio, opt)
    setproperty!(portfolio, rmf, Inf)

    w1t = [1.7415671832877554e-8, 0.09964232195454342, 1.2826906157095241e-8,
           3.620519374703137e-8, 1.3195188875301837e-8, 2.911115531964785e-7,
           5.7616625258389416e-9, 0.1353648589359337, 1.0911585316802027e-8,
           1.5454998047298212e-8, 0.3900883050668621, 1.768537270943559e-8,
           8.060055145517512e-9, 0.02133094950520555, 3.009127529968494e-8,
           5.326395426982443e-8, 1.30293148280847e-7, 0.199232395517277,
           1.643394214993775e-8, 0.15434051030967075]

    w2t = [1.3173453247258147e-8, 0.09964275927272323, 9.701561077554268e-9,
           2.7490187518814357e-8, 9.980597919022525e-9, 2.212186051622533e-7,
           4.388523413574402e-9, 0.13536479083531627, 8.265820121727917e-9,
           1.170106187371425e-8, 0.3900880830855678, 1.3465686865426654e-8,
           6.169401403427077e-9, 0.021330812821569744, 2.2945557702874025e-8,
           4.0433473924038086e-8, 9.881233242340814e-8, 0.19923267501462183,
           1.2443915204476195e-8, 0.15434037878002327]

    w3t = [1.203677292111802e-7, 0.09964032518435083, 8.430494669073566e-8,
           2.1639361574292794e-7, 8.4260708196597e-8, 1.6851797041539608e-6,
           3.707022997795726e-8, 0.13536464661477562, 7.166188959360897e-8,
           1.014707880221743e-7, 0.3900887134394191, 1.188229185875413e-7,
           5.718696005408646e-8, 0.021332112290831606, 1.9521980695312645e-7,
           3.097279618914747e-7, 7.074644372188145e-7, 0.19923007868895815,
           1.0636135294894942e-7, 0.15434022828861546]

    w4t = [2.405150276647129e-9, 5.016952234021194e-9, 3.4959105703349184e-9,
           2.4492362502394727e-9, 0.6383912639288716, 5.735756043558714e-10,
           0.10348429323972896, 2.3097401452051595e-9, 2.5819663895077334e-9,
           1.3946303332289194e-9, 2.3142580818523254e-9, 5.487465170545815e-10,
           3.386337247426007e-10, 1.0447926276434799e-9, 3.578311278172937e-10,
           0.19207064194651963, 0.06605376289867351, 2.7026739563722534e-9,
           7.595805200570531e-9, 2.8563033379292407e-9]

    w5t = [2.2797342319151626e-9, 5.77189526756379e-9, 3.3878816270315285e-9,
           2.5115240056460785e-9, 0.5728216422772073, 5.516554271642011e-10,
           0.08065973551515868, 2.9387655878145927e-9, 2.6229454349156457e-9,
           1.4499388221302601e-9, 2.961543281479944e-9, 5.058788194912443e-10,
           3.156082507772624e-10, 1.057901371004259e-9, 3.3114501286437446e-10,
           0.18372057214293466, 0.16279800567260477, 3.363269561555129e-9,
           1.1190680968452987e-8, 3.151726922761528e-9]

    w6t = [9.719846101660443e-7, 9.414453254795493e-7, 7.366782784843086e-7,
           6.601768027385454e-7, 0.5736118379766, 5.475979094163312e-7, 0.08041348221469191,
           5.298936836477535e-7, 3.068416994102572e-7, 8.930574609009536e-7,
           1.535061858646997e-6, 3.6335161865262885e-7, 6.566911171879743e-7,
           6.14894099791105e-7, 1.0113926635114196e-6, 0.18384332753031996,
           0.162118130379122, 8.741290723993682e-7, 1.512005560236791e-6,
           1.0666975054430286e-6]

    w7t = [7.077740786502278e-8, 2.8995072121003874e-5, 7.324120908262872e-8,
           9.014685437808509e-8, 0.3209653044472713, 1.7039921288539243e-8,
           0.004686827926912032, 0.05273757705900058, 5.795360291451094e-8,
           4.438118459659057e-8, 0.2447634182942204, 1.544477562980185e-8,
           8.089789267817478e-9, 4.6835863487950934e-8, 9.128252537330769e-9,
           0.1358547146503151, 0.2409600845568891, 1.5995825276373024e-6,
           2.1810886342639211e-7, 8.272630184620377e-7]

    w8t = [8.029101550920666e-9, 0.0163303208494206, 8.4661788285626e-9,
           1.1909101200703231e-8, 0.28264982816771445, 2.275805657436956e-9,
           0.0006810019447891123, 0.07099645271648265, 6.9405290124144486e-9,
           5.561557657963495e-9, 0.26894671534012754, 2.018809605444504e-9,
           1.1281814323428742e-9, 6.2941071929937846e-9, 1.2739315420323108e-9,
           0.12089026799972692, 0.21860946619237137, 0.013239568096715172,
           2.445527027269246e-8, 0.007656300340078226]

    w10t = [1.0025388976055293e-9, 1.0668671447191466e-9, 1.3519393202152327e-9,
            1.2282465330364543e-9, 1.005177430408014e-7, 4.791684152377937e-10,
            0.9999998830179009, 7.341499980325647e-10, 1.1371524504118676e-9,
            8.074220981577865e-10, 7.11789134528336e-10, 5.053597547452678e-10,
            3.720070059848833e-10, 6.014948684937246e-10, 3.8399651339014223e-10,
            1.787507624730215e-9, 1.404677100071108e-9, 7.700998592963003e-10,
            1.1737535372835695e-9, 9.461859039839322e-10]

    w11t = [3.874221086619612e-12, 5.104908090067462e-12, 8.520494500197468e-12,
            7.290959009200468e-12, 0.8503243999802813, 2.3379452125355536e-12,
            0.14967559993564175, 1.0705730358034575e-12, 6.66117020876977e-12,
            2.305310714767095e-12, 6.777032301231088e-13, 1.6337900308145626e-12,
            2.9324367282130846e-12, 7.923883883764425e-13, 3.029696894312117e-12,
            1.585046203658579e-11, 9.70832317790743e-12, 1.3214127598773591e-12,
            7.1618937601443205e-12, 3.803361046983932e-12]

    @test isapprox(w1.weights, w1t, rtol = 1.0e-5)
    @test isapprox(w2.weights, w2t, rtol = 1.0e-5)
    @test isapprox(w3.weights, w3t, rtol = 1.0e-6)
    @test isapprox(w2.weights, w3.weights, rtol = 1e-5)
    @test isapprox(w4.weights, w4t, rtol = 1.0e-6)
    @test isapprox(w5.weights, w5t, rtol = 1.0e-6)
    if !isempty(w6)
        @test isapprox(w6.weights, w6t, rtol = 0.0001)
        @test isapprox(w5.weights, w6.weights, rtol = 1e-2)
    end
    @test isapprox(w7.weights, w7t, rtol = 5.0e-5)
    @test isapprox(w8.weights, w8t, rtol = 1.0e-5)
    @test isapprox(w10.weights, w10t, rtol = 1.0e-7)
    @test isapprox(w11.weights, w11t, rtol = 1.0e-7)
    @test isapprox(w13.weights, w7.weights, rtol = 1e-4)
    @test isapprox(w14.weights, w8.weights, rtol = 1e-1)
    @test isapprox(w16.weights, w7.weights, rtol = 1e-1)
    @test isapprox(w17.weights, w8.weights, rtol = 1e-1)
    @test isapprox(w13.weights, w16.weights, rtol = 1e-1)
    @test isapprox(w14.weights, w17.weights, rtol = 1e-1)
    @test isapprox(w18.weights, w1.weights, rtol = 2e-1)
end

@testset "$(:Classic), $(:Trad), Reduced $(:SKurt)" begin
    portfolio = Portfolio(; prices = prices,
                          solvers = OrderedDict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                                  :params => Dict("verbose" => false,
                                                                                  "max_step_fraction" => 0.75))),
                          max_num_assets_kurt = 1)
    asset_statistics!(portfolio)

    rm = :SKurt
    opt = OptimiseOpt(; rf = rf, l = l, class = :Classic, type = :Trad, rm = rm,
                      obj = :Min_Risk, kelly = :None)

    w1 = optimise!(portfolio, opt)
    opt.kelly = :Approx
    w2 = optimise!(portfolio, opt)
    opt.kelly = :Exact
    w3 = optimise!(portfolio, opt)
    opt.obj = :Utility
    opt.kelly = :None
    w4 = optimise!(portfolio, opt)
    opt.kelly = :Approx
    w5 = optimise!(portfolio, opt)
    opt.kelly = :Exact
    opt.obj = :Sharpe
    opt.kelly = :None
    w7 = optimise!(portfolio, opt)
    risk7 = calc_risk(portfolio; type = :Trad, rm = :Kurt, rf = rf)
    ret7 = dot(portfolio.mu, w7.weights)
    opt.obj = :Max_Ret
    opt.kelly = :None
    w10 = optimise!(portfolio, opt)
    opt.kelly = :Approx
    w11 = optimise!(portfolio, opt)
    setproperty!(portfolio, :mu_l, ret7)
    opt.obj = :Min_Risk
    opt.kelly = :None
    w13 = optimise!(portfolio, opt)
    setproperty!(portfolio, :mu_l, Inf)

    w1t = [9.53705592757537e-7, 0.08065159914855387, 6.974293424842483e-7,
           1.6362439335233703e-6, 6.97077600776025e-7, 3.475223941560657e-6,
           2.890127705918288e-7, 0.12690253467700394, 6.273572462322351e-7,
           8.136543644995557e-7, 0.462265972677861, 1.009380453920336e-6,
           4.2373818121359034e-7, 2.0106200273473065e-5, 1.1018636244579568e-6,
           2.157409506365442e-6, 3.949693509750946e-6, 0.17011915810851502,
           8.973744955725058e-7, 0.16002190002322905]

    w2t = [1.0280900236544283e-6, 0.0806515184187096, 7.508175412886482e-7,
           1.7607417560736309e-6, 7.503921095213154e-7, 3.7461196533230673e-6,
           3.092490915598872e-7, 0.12690208017303703, 6.738162750594779e-7,
           8.759116032501992e-7, 0.4622644456200876, 1.0839908244464866e-6,
           4.532816894653588e-7, 2.1752847116319123e-5, 1.183559275324526e-6,
           2.3209737430561147e-6, 4.250432566957934e-6, 0.1701182582121211,
           9.657095999763854e-7, 0.16002179164317523]

    w3t = [1.2779803016602962e-7, 0.08067090781696645, 8.873074067270016e-8,
           2.0264874035824251e-7, 8.770967101288204e-8, 4.782122226538125e-7,
           3.5886492811386174e-8, 0.12690302318378593, 7.928056174743125e-8,
           1.0345366605272408e-7, 0.4622604106323459, 1.3078820216733013e-7,
           5.508635055834974e-8, 2.1490004675268424e-6, 1.4202451693270379e-7,
           2.660556142206975e-7, 4.770252753395685e-7, 0.17013513300107347,
           1.1366636627326855e-7, 0.16002598799890957]

    w4t = [2.351831218818561e-7, 4.5486436237674446e-7, 3.289992459233192e-7,
           2.3649353238336663e-7, 0.6546702534218488, 6.070770916505342e-8,
           0.10183768792876238, 2.2938785029609432e-7, 2.5049214012753256e-7,
           1.4230718791947422e-7, 2.3118132664771802e-7, 5.86594072963264e-8,
           3.6106226410924454e-8, 1.0854896459171796e-7, 3.821049909517469e-8,
           0.19381294831163215, 0.04967546347760286, 2.6587343329424463e-7,
           6.969988060815927e-7, 2.728463401933757e-7]

    w5t = [2.309743299138018e-7, 5.148216665773432e-7, 3.2223881095891444e-7,
           2.4930853488648735e-7, 0.582936039521658, 6.590069080963773e-8,
           0.07877810955000863, 2.9539172535462957e-7, 2.6077691909368537e-7,
           1.5720694556734463e-7, 2.9865408420429057e-7, 6.097583592180474e-8,
           3.8615636270009715e-8, 1.1939996595105053e-7, 4.090835060852274e-8,
           0.18421305306408892, 0.154068542006734, 3.3260661875264105e-7,
           9.635346690461754e-7, 3.0454272669510413e-7]

    w7t = [9.329022023707404e-8, 1.899703874705677e-6, 9.183578489064419e-8,
           1.127843873400207e-7, 0.32330753577000576, 2.547312313016181e-8,
           6.050607477067122e-6, 0.044165256984155805, 7.422021388003195e-8,
           6.013910015827217e-8, 0.26072810405200914, 2.284007108846321e-8,
           1.2053254336149004e-8, 6.734529353443216e-8, 1.3612306242433804e-8,
           0.1350544143521361, 0.23673428114212236, 1.0078611572415905e-6,
           2.4778255312650613e-7, 6.281507540774648e-7]

    w10t = [4.9361186803657026e-8, 5.276019912068542e-8, 6.808537979965752e-8,
            6.105353959598766e-8, 6.716105964584807e-6, 2.444636594459631e-8,
            0.999992457557, 3.623421006815773e-8, 5.6057026670956243e-8,
            3.953112345795217e-8, 3.521364358449262e-8, 2.562434532175363e-8,
            1.9583592576006807e-8, 3.003472007744737e-8, 2.0121234588388678e-8,
            9.369260216436811e-8, 7.159078914671406e-8, 3.7984586248700956e-8,
            5.837277025344171e-8, 4.658972012276225e-8]

    w11t = [2.0892582150438786e-10, 2.822466459887792e-10, 4.3716523727498276e-10,
            3.6203373177553037e-10, 0.8503240818502349, 1.1465722422462249e-10,
            0.14967591348373613, 6.94378771085796e-11, 3.352869895823194e-10,
            1.0468436967641833e-10, 5.110616040530915e-11, 1.1040973718844364e-10,
            2.2365217864583654e-10, 4.1668192227228446e-11, 1.5966585909281757e-10,
            9.379337226336307e-10, 5.378040741377087e-10, 9.381853915336232e-11,
            3.873293703071198e-10, 2.0820349967377268e-10]

    @test isapprox(w1.weights, w1t)
    @test isapprox(w2.weights, w2t)
    @test isapprox(w3.weights, w3t)
    @test isapprox(w2.weights, w3.weights, rtol = 0.0001)
    @test isapprox(w4.weights, w4t)
    @test isapprox(w5.weights, w5t)
    @test isapprox(w7.weights, w7t, rtol = 1.0e-7)
    @test isapprox(w10.weights, w10t)
    @test isapprox(w11.weights, w11t)
    @test isapprox(w13.weights, w7.weights, rtol = 0.0005)
end
