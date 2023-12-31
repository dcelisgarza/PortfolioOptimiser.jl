using COSMO,
    CSV,
    Clarabel,
    HiGHS,
    LinearAlgebra,
    OrderedCollections,
    PortfolioOptimiser,
    Statistics,
    Test,
    TimeSeries

prices = TimeArray(CSV.File("./assets/stock_prices.csv"); timestamp = :date)

rf = 1.0329^(1 / 252) - 1
l = 2.0

@testset "$(:Classic), $(:Trad), $(:GMD), blank $(:OWA) and owa_w = owa_gmd(T)" begin
    portfolio = Portfolio(
        prices = prices[(end - 200):end],
        solvers = OrderedDict(
            :Clarabel => Dict(
                :solver => Clarabel.Optimizer,
                :params => Dict(
                    "verbose" => false,
                    "max_step_fraction" => 0.75,
                    "max_iter" => 100,
                    "equilibrate_max_iter" => 20,
                ),
            ),
            :COSMO =>
                Dict(:solver => COSMO.Optimizer, :params => Dict("verbose" => false)),
        ),
    )
    asset_statistics!(portfolio)

    w1 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = :Classic,
        type = :Trad,
        rm = :GMD,
        obj = :Min_Risk,
        kelly = :None,
    )
    risk1 = calc_risk(portfolio; type = :Trad, rm = :GMD, rf = rf)
    w2 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = :Classic,
        type = :Trad,
        rm = :GMD,
        obj = :Min_Risk,
        kelly = :Approx,
    )
    w3 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = :Classic,
        type = :Trad,
        rm = :GMD,
        obj = :Min_Risk,
        kelly = :Exact,
    )
    w4 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = :Classic,
        type = :Trad,
        rm = :GMD,
        obj = :Utility,
        kelly = :None,
    )
    w5 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = :Classic,
        type = :Trad,
        rm = :GMD,
        obj = :Utility,
        kelly = :Approx,
    )
    w6 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = :Classic,
        type = :Trad,
        rm = :GMD,
        obj = :Utility,
        kelly = :Exact,
    )
    w7 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = :Classic,
        type = :Trad,
        rm = :GMD,
        obj = :Sharpe,
        kelly = :None,
    )
    risk7 = calc_risk(portfolio; type = :Trad, rm = :GMD, rf = rf)
    ret7 = dot(portfolio.mu, w7.weights)
    w8 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = :Classic,
        type = :Trad,
        rm = :GMD,
        obj = :Sharpe,
        kelly = :Approx,
    )
    risk8 = calc_risk(portfolio; type = :Trad, rm = :GMD, rf = rf)
    ret8 = dot(portfolio.mu, w8.weights)
    w9 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = :Classic,
        type = :Trad,
        rm = :GMD,
        obj = :Sharpe,
        kelly = :Exact,
    )
    risk9 = calc_risk(portfolio; type = :Trad, rm = :GMD, rf = rf)
    ret9 = dot(portfolio.mu, w9.weights)
    w10 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = :Classic,
        type = :Trad,
        rm = :GMD,
        obj = :Max_Ret,
        kelly = :None,
    )
    w11 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = :Classic,
        type = :Trad,
        rm = :GMD,
        obj = :Max_Ret,
        kelly = :Approx,
    )
    w12 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = :Classic,
        type = :Trad,
        rm = :GMD,
        obj = :Max_Ret,
        kelly = :Exact,
    )
    setproperty!(portfolio, :mu_l, ret7)
    w13 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = :Classic,
        type = :Trad,
        rm = :GMD,
        obj = :Min_Risk,
        kelly = :None,
    )
    setproperty!(portfolio, :mu_l, ret8)
    w14 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = :Classic,
        type = :Trad,
        rm = :GMD,
        obj = :Min_Risk,
        kelly = :None,
    )
    setproperty!(portfolio, :mu_l, ret9)
    w15 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = :Classic,
        type = :Trad,
        rm = :GMD,
        obj = :Min_Risk,
        kelly = :None,
    )
    setproperty!(portfolio, :mu_l, Inf)
    rmf = Symbol(lowercase(string(:GMD)) * "_u")
    setproperty!(portfolio, rmf, risk7)
    w16 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = :Classic,
        type = :Trad,
        rm = :GMD,
        obj = :Max_Ret,
        kelly = :None,
    )
    setproperty!(portfolio, rmf, risk8)
    w17 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = :Classic,
        type = :Trad,
        rm = :GMD,
        obj = :Max_Ret,
        kelly = :None,
    )
    setproperty!(portfolio, rmf, risk9)
    w18 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = :Classic,
        type = :Trad,
        rm = :GMD,
        obj = :Max_Ret,
        kelly = :None,
    )
    setproperty!(portfolio, rmf, risk1)
    w19 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = :Classic,
        type = :Trad,
        rm = :GMD,
        obj = :Sharpe,
        kelly = :None,
    )
    setproperty!(portfolio, rmf, Inf)
    w20 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = :Classic,
        type = :Trad,
        rm = :OWA,
        obj = :Min_Risk,
        kelly = :None,
    )
    risk20 = calc_risk(portfolio; type = :Trad, rm = :OWA, rf = rf)
    w21 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = :Classic,
        type = :Trad,
        rm = :OWA,
        obj = :Min_Risk,
        kelly = :Approx,
    )
    w22 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = :Classic,
        type = :Trad,
        rm = :OWA,
        obj = :Min_Risk,
        kelly = :Exact,
    )
    w23 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = :Classic,
        type = :Trad,
        rm = :OWA,
        obj = :Utility,
        kelly = :None,
    )
    w24 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = :Classic,
        type = :Trad,
        rm = :OWA,
        obj = :Utility,
        kelly = :Approx,
    )
    w25 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = :Classic,
        type = :Trad,
        rm = :OWA,
        obj = :Utility,
        kelly = :Exact,
    )
    w26 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = :Classic,
        type = :Trad,
        rm = :OWA,
        obj = :Sharpe,
        kelly = :None,
    )
    risk26 = calc_risk(portfolio; type = :Trad, rm = :OWA, rf = rf)
    ret26 = dot(portfolio.mu, w26.weights)
    w27 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = :Classic,
        type = :Trad,
        rm = :OWA,
        obj = :Sharpe,
        kelly = :Approx,
    )
    risk27 = calc_risk(portfolio; type = :Trad, rm = :OWA, rf = rf)
    ret27 = dot(portfolio.mu, w27.weights)
    w28 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = :Classic,
        type = :Trad,
        rm = :OWA,
        obj = :Sharpe,
        kelly = :Exact,
    )
    risk28 = calc_risk(portfolio; type = :Trad, rm = :OWA, rf = rf)
    ret28 = dot(portfolio.mu, w28.weights)
    w29 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = :Classic,
        type = :Trad,
        rm = :OWA,
        obj = :Max_Ret,
        kelly = :None,
    )
    w30 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = :Classic,
        type = :Trad,
        rm = :OWA,
        obj = :Max_Ret,
        kelly = :Approx,
    )
    w31 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = :Classic,
        type = :Trad,
        rm = :OWA,
        obj = :Max_Ret,
        kelly = :Exact,
    )
    setproperty!(portfolio, :mu_l, ret26)
    w32 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = :Classic,
        type = :Trad,
        rm = :OWA,
        obj = :Min_Risk,
        kelly = :None,
    )
    setproperty!(portfolio, :mu_l, ret27)
    w33 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = :Classic,
        type = :Trad,
        rm = :OWA,
        obj = :Min_Risk,
        kelly = :None,
    )
    setproperty!(portfolio, :mu_l, ret28)
    w34 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = :Classic,
        type = :Trad,
        rm = :OWA,
        obj = :Min_Risk,
        kelly = :None,
    )
    setproperty!(portfolio, :mu_l, Inf)
    rmf = Symbol(lowercase(string(:OWA)) * "_u")
    setproperty!(portfolio, rmf, risk26)
    w35 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = :Classic,
        type = :Trad,
        rm = :OWA,
        obj = :Max_Ret,
        kelly = :None,
    )
    setproperty!(portfolio, rmf, risk27)
    w36 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = :Classic,
        type = :Trad,
        rm = :OWA,
        obj = :Max_Ret,
        kelly = :None,
    )
    setproperty!(portfolio, rmf, risk28)
    w37 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = :Classic,
        type = :Trad,
        rm = :OWA,
        obj = :Max_Ret,
        kelly = :None,
    )
    setproperty!(portfolio, rmf, risk20)
    w38 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = :Classic,
        type = :Trad,
        rm = :OWA,
        obj = :Sharpe,
        kelly = :None,
    )
    setproperty!(portfolio, rmf, Inf)
    portfolio.owa_w = owa_gmd(size(portfolio.returns, 1))
    w39 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = :Classic,
        type = :Trad,
        rm = :OWA,
        obj = :Min_Risk,
        kelly = :None,
    )
    risk39 = calc_risk(portfolio; type = :Trad, rm = :OWA, rf = rf)
    w40 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = :Classic,
        type = :Trad,
        rm = :OWA,
        obj = :Min_Risk,
        kelly = :Approx,
    )
    w41 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = :Classic,
        type = :Trad,
        rm = :OWA,
        obj = :Min_Risk,
        kelly = :Exact,
    )
    w42 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = :Classic,
        type = :Trad,
        rm = :OWA,
        obj = :Utility,
        kelly = :None,
    )
    w43 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = :Classic,
        type = :Trad,
        rm = :OWA,
        obj = :Utility,
        kelly = :Approx,
    )
    w44 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = :Classic,
        type = :Trad,
        rm = :OWA,
        obj = :Utility,
        kelly = :Exact,
    )
    w45 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = :Classic,
        type = :Trad,
        rm = :OWA,
        obj = :Sharpe,
        kelly = :None,
    )
    risk45 = calc_risk(portfolio; type = :Trad, rm = :OWA, rf = rf)
    ret45 = dot(portfolio.mu, w45.weights)
    w46 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = :Classic,
        type = :Trad,
        rm = :OWA,
        obj = :Sharpe,
        kelly = :Approx,
    )
    risk46 = calc_risk(portfolio; type = :Trad, rm = :OWA, rf = rf)
    ret46 = dot(portfolio.mu, w46.weights)
    w47 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = :Classic,
        type = :Trad,
        rm = :OWA,
        obj = :Sharpe,
        kelly = :Exact,
    )
    risk47 = calc_risk(portfolio; type = :Trad, rm = :OWA, rf = rf)
    ret47 = dot(portfolio.mu, w47.weights)
    w48 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = :Classic,
        type = :Trad,
        rm = :OWA,
        obj = :Max_Ret,
        kelly = :None,
    )
    w49 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = :Classic,
        type = :Trad,
        rm = :OWA,
        obj = :Max_Ret,
        kelly = :Approx,
    )
    w50 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = :Classic,
        type = :Trad,
        rm = :OWA,
        obj = :Max_Ret,
        kelly = :Exact,
    )
    setproperty!(portfolio, :mu_l, ret45)
    w51 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = :Classic,
        type = :Trad,
        rm = :OWA,
        obj = :Min_Risk,
        kelly = :None,
    )
    setproperty!(portfolio, :mu_l, ret46)
    w52 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = :Classic,
        type = :Trad,
        rm = :OWA,
        obj = :Min_Risk,
        kelly = :None,
    )
    setproperty!(portfolio, :mu_l, ret47)
    w53 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = :Classic,
        type = :Trad,
        rm = :OWA,
        obj = :Min_Risk,
        kelly = :None,
    )
    setproperty!(portfolio, :mu_l, Inf)
    rmf = Symbol(lowercase(string(:OWA)) * "_u")
    setproperty!(portfolio, rmf, risk45)
    w54 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = :Classic,
        type = :Trad,
        rm = :OWA,
        obj = :Max_Ret,
        kelly = :None,
    )
    setproperty!(portfolio, rmf, risk46)
    w55 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = :Classic,
        type = :Trad,
        rm = :OWA,
        obj = :Max_Ret,
        kelly = :None,
    )
    setproperty!(portfolio, rmf, risk47)
    w56 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = :Classic,
        type = :Trad,
        rm = :OWA,
        obj = :Max_Ret,
        kelly = :None,
    )
    setproperty!(portfolio, rmf, risk39)
    w57 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = :Classic,
        type = :Trad,
        rm = :OWA,
        obj = :Sharpe,
        kelly = :None,
    )
    setproperty!(portfolio, rmf, Inf)

    w1t = [
        2.859922116198051e-10,
        1.2839851061414282e-9,
        6.706646698191418e-10,
        0.011313263237504052,
        0.041188384886560354,
        0.021164906062678113,
        9.703581331630189e-11,
        0.04727424011116681,
        5.765539509407631e-10,
        4.5707249638450353e-10,
        0.07079102697763319,
        2.3691876986431332e-11,
        6.862345293974742e-11,
        0.2926160718937385,
        2.627931180682999e-11,
        0.014804760223130658,
        0.055360563578446625,
        0.22572350787381906,
        0.008097074123824198,
        0.2116661975415996,
    ]

    w2t = [
        5.251492167108595e-10,
        2.365324942455108e-9,
        1.228133964759497e-9,
        0.01136625390653424,
        0.04115041304677448,
        0.021118575976097856,
        1.7826397676563226e-10,
        0.04726442195474365,
        1.0573388621330249e-9,
        8.386390023197234e-10,
        0.07085089072254845,
        4.421333531677392e-11,
        1.2543433104522538e-10,
        0.2925312243040699,
        4.882105192830059e-11,
        0.014796401887494866,
        0.055296423856387264,
        0.225728025547612,
        0.008121179213586627,
        0.21177618317283206,
    ]

    w3t = [
        5.354026704869777e-10,
        2.2436905049917003e-9,
        1.1821239557765714e-9,
        0.011356430502054034,
        0.04115626893621375,
        0.021118125043514093,
        2.1573924944334443e-10,
        0.047258599507084435,
        1.0190748261693879e-9,
        8.182234303696856e-10,
        0.07083743989900024,
        9.061256814168461e-11,
        1.6821477600735128e-10,
        0.2925476241867142,
        9.45240785865248e-11,
        0.01479824618994179,
        0.05530574095746841,
        0.22573566536305198,
        0.008120334785801188,
        0.21176551826154985,
    ]

    w4t = [
        4.395442944440727e-10,
        2.1495615257047445e-9,
        8.700158416075159e-10,
        0.006589303508967539,
        0.05330845589356504,
        7.632737386668005e-10,
        9.831583865942048e-11,
        0.06008335161013898,
        1.3797121555924764e-9,
        3.0160588760343415e-9,
        0.06394372429204083,
        4.486199590323317e-11,
        1.0975948346699443e-10,
        0.24314317639043878,
        5.90094645737884e-11,
        0.018483948217979402,
        0.08594753958504305,
        0.216033164253688,
        0.05666930366548,
        0.19579802365254514,
    ]

    w5t = [
        2.1373923858202174e-10,
        1.047935128182663e-9,
        4.2352337620182766e-10,
        0.006623632023945508,
        0.053262035174195796,
        3.748985425782847e-10,
        4.8604916317016427e-11,
        0.059837358756352356,
        6.629955827787793e-10,
        1.4334114636079199e-9,
        0.06407645387087887,
        2.2564457261760664e-11,
        5.371370812720617e-11,
        0.24323140594664305,
        2.9658041881066845e-11,
        0.018454146656654288,
        0.0859437656795087,
        0.21613799090173264,
        0.05652392850317017,
        0.19590927817587422,
    ]

    w6t = [
        1.7127252443308418e-7,
        6.934858519404509e-7,
        2.866291591186648e-7,
        0.006553485493782127,
        0.05314502242400377,
        2.3075679946487433e-7,
        5.766434552633233e-8,
        0.059965342332728655,
        7.108722767488626e-7,
        8.499172115788923e-7,
        0.06397266225548641,
        4.906525899110893e-8,
        6.017663441983441e-8,
        0.243323504645496,
        4.6587059930614004e-8,
        0.018708368507070104,
        0.08575936499218657,
        0.2163840887167888,
        0.05625377780709815,
        0.19593122639823732,
    ]

    w7t = [
        1.0960279114854144e-12,
        4.2713859884519724e-11,
        3.322171869553163e-12,
        3.8205053405909246e-11,
        0.2107929632511357,
        2.439956520619061e-11,
        2.2427021896796488e-11,
        0.011060068605326014,
        0.021557962662139638,
        1.524860659816748e-10,
        2.143824873504224e-11,
        1.626628185083569e-11,
        2.276043092971382e-11,
        1.0059568286002983e-11,
        1.872567893305395e-11,
        0.0977182789979517,
        0.39859819200001834,
        5.499837267441688e-11,
        0.26027253402707684,
        2.7453438274980142e-11,
    ]

    w8t = [
        1.0155020749727522e-10,
        2.38178191529736e-10,
        1.1584451510196998e-10,
        2.1758373547817112e-10,
        0.20152993710189268,
        2.993101288477597e-11,
        3.6439928804387634e-11,
        0.028636902063586874,
        0.005806401898518205,
        6.433916064904551e-10,
        1.843996026263977e-10,
        5.527116705376367e-11,
        3.602234132572901e-11,
        1.4360190351319453e-10,
        4.813709296729719e-11,
        0.09660151833030618,
        0.38225707069143156,
        3.075324109719233e-10,
        0.28516816754177193,
        2.1460877457707472e-10,
    ]

    w9t = [
        5.671157677840885e-7,
        1.2733743334667473e-6,
        6.370657936156067e-7,
        1.239290084842503e-6,
        0.20705790824767295,
        1.7306287519061096e-7,
        2.1949092659034066e-7,
        0.01502662025379091,
        0.01270587595162983,
        2.877209394665577e-6,
        8.889750437093147e-7,
        3.269795768965802e-7,
        2.0703480742609938e-7,
        7.234077461496872e-7,
        2.7060737069570363e-7,
        0.09794013285762235,
        0.39947829026120407,
        1.3837803825062758e-6,
        0.26777939992827077,
        9.85105705745203e-7,
    ]

    w10t = [
        8.713294565329801e-10,
        1.4655494663368051e-9,
        9.339861464983668e-10,
        2.1822916966782698e-9,
        0.9999999341529429,
        3.1514150951638473e-10,
        4.623635446692933e-10,
        1.3314213108887623e-9,
        5.686161252843585e-9,
        1.371922705853848e-9,
        7.146550790353018e-10,
        5.740256406691589e-10,
        3.664169883135698e-10,
        7.000039902341564e-10,
        4.5030867400627253e-10,
        6.996286227848405e-9,
        3.6287634044926756e-8,
        9.000571133076656e-10,
        3.4765700831013475e-9,
        7.60932312745152e-10,
    ]

    w11t = [
        5.995643893118203e-12,
        1.7788172572403522e-11,
        7.42248967697158e-12,
        2.5855688894534932e-11,
        0.8577737311672241,
        5.084251669919802e-12,
        1.8918809363559223e-12,
        1.7873684105427732e-11,
        9.127337657198234e-11,
        1.8171837160854858e-11,
        3.850566386470494e-12,
        2.5508401127966545e-13,
        4.0216144858012615e-12,
        3.339109115577017e-12,
        2.0662179397071288e-12,
        1.8569702892630115e-10,
        0.14222626837294614,
        8.073610828892573e-12,
        5.65606511499808e-11,
        4.608669048088745e-12,
    ]

    w12t = [
        6.8783457384567556e-9,
        1.2274774655426258e-8,
        7.210231160923748e-9,
        1.6157723542114e-8,
        0.8926318884419521,
        2.236669895286051e-9,
        3.0721278132094936e-9,
        1.1931574111287059e-8,
        5.313990853277208e-8,
        1.1675553033694841e-8,
        5.76375880463489e-9,
        4.1812752119557695e-9,
        2.3871108169602047e-9,
        5.509568787249582e-9,
        2.9557632944416428e-9,
        9.517716377203498e-8,
        0.10736782516025675,
        7.544504043725093e-9,
        3.230881102698668e-8,
        5.9929266940059665e-9,
    ]

    w13t = [
        2.111064693976598e-12,
        4.199098436247614e-11,
        2.4277750791679688e-12,
        3.8079944685205154e-11,
        0.21084481684472242,
        2.6325394341890063e-11,
        2.398425045782114e-11,
        0.011087215230686672,
        0.021727930888863614,
        1.5980784508945338e-10,
        2.2534788169784417e-11,
        1.7758263471786547e-11,
        2.4160204726553847e-11,
        1.030058571203951e-11,
        2.000500776826796e-11,
        0.09769732518814264,
        0.39851241713315694,
        5.853448351366192e-11,
        0.26013029423727785,
        2.9129230394432605e-11,
    ]

    w14t = [
        8.942270766111967e-13,
        4.997065086540317e-11,
        4.597515424452085e-12,
        4.14025186234731e-11,
        0.2045787142958964,
        2.757625635103089e-11,
        2.5526588174105618e-11,
        0.028823839199628453,
        0.00935681866157209,
        2.2178037048090422e-10,
        3.349076787579454e-11,
        1.87998796931707e-11,
        2.5264985600201933e-11,
        1.653886341218741e-11,
        2.0604214115929157e-11,
        0.09537027357843764,
        0.377524299391081,
        8.51485669294363e-11,
        0.2843460542557699,
        4.601891415928919e-11,
    ]

    w15t = [
        5.1306740699054454e-12,
        3.498203888295839e-11,
        9.443660977510647e-13,
        3.0051209746434546e-11,
        0.2096190099044751,
        2.6830393435136183e-11,
        2.4866423157271724e-11,
        0.015388952005987784,
        0.01947723659851483,
        1.4698330462316705e-10,
        1.836237237661002e-11,
        1.931290540546062e-11,
        2.4901861449564757e-11,
        6.724696953439446e-12,
        2.113917582584407e-11,
        0.09756081719654436,
        0.3941356326147162,
        5.2721875186497566e-11,
        0.26381835124131175,
        2.5498813689609923e-11,
    ]

    w16t = [
        8.645417462703117e-13,
        9.74931569493778e-12,
        1.7691244640820176e-12,
        8.874979261575232e-12,
        0.21078639257923237,
        3.922009126537224e-12,
        3.4338064345481404e-12,
        0.011063602423968855,
        0.02153048309961715,
        3.387355507162292e-11,
        5.799061166068984e-12,
        2.2147166479480898e-12,
        3.476808868173112e-12,
        3.3237370929321917e-12,
        2.640299454582965e-12,
        0.09773989516924422,
        0.3986125551970178,
        1.3091156994666361e-11,
        0.26026707143077404,
        7.112393143958492e-12,
    ]

    w17t = [
        2.196665141984919e-12,
        1.2060486285149243e-11,
        3.2469345250787356e-12,
        1.035882803391085e-11,
        0.20454337993180025,
        2.8837388491504335e-12,
        2.4643498937450153e-12,
        0.02883361320119504,
        0.009228447548408824,
        4.5440170588235665e-11,
        8.78290112077899e-12,
        1.1948258696922626e-12,
        2.427952696676779e-12,
        5.51739407846932e-12,
        1.524788450375429e-12,
        0.09537008234204332,
        0.3777286103721778,
        1.876506344052584e-11,
        0.2842958664763445,
        1.116624337339785e-11,
    ]

    w18t = [
        9.09711465126155e-13,
        9.449290266913617e-12,
        1.7739420562277521e-12,
        8.37151312097126e-12,
        0.20966915944272185,
        3.5479502740177604e-12,
        3.0757608336758066e-12,
        0.015378180154910333,
        0.01950214340438561,
        3.35180762679845e-11,
        5.785868839429511e-12,
        1.9479838142418733e-12,
        3.1193764890204444e-12,
        3.3382615997835853e-12,
        2.3158677660472516e-12,
        0.09758505196452177,
        0.3941957379297834,
        1.3034815581554832e-11,
        0.2636697270062483,
        7.240147916177639e-12,
    ]

    w19t = [
        1.853222332200707e-9,
        7.77154342441663e-9,
        4.361453531340065e-9,
        0.011268503190433975,
        0.041024263447110715,
        0.019990798875061184,
        6.32380113701166e-10,
        0.04768163200097348,
        3.919590544731981e-9,
        3.2835408002817512e-9,
        0.07074337720718103,
        1.626354865702736e-10,
        4.78095986084816e-10,
        0.29162850516446387,
        1.824495731211796e-10,
        0.014716325550611356,
        0.05627433067844324,
        0.2257134112824113,
        0.009011591493696645,
        0.2119472384647015,
    ]

    @test isapprox(w1.weights, w1t, rtol = 1.0e-6)
    @test isapprox(w2.weights, w2t, rtol = 1.0e-6)
    @test isapprox(w3.weights, w3t, rtol = 1.0e-5)
    @test isapprox(w2.weights, w3.weights, rtol = 0.001)
    @test isapprox(w4.weights, w4t, rtol = 1.0e-6)
    @test isapprox(w5.weights, w5t, rtol = 1.0e-5)
    @test isapprox(w6.weights, w6t, rtol = 0.01)
    @test isapprox(w5.weights, w6.weights, rtol = 0.01)
    @test isapprox(w7.weights, w7t)
    @test isapprox(w8.weights, w8t)
    @test isapprox(w9.weights, w9t, rtol = 0.01)
    @test isapprox(w8.weights, w9.weights, rtol = 0.1)
    @test isapprox(w10.weights, w10t)
    @test isapprox(w11.weights, w11t)
    @test isapprox(w12.weights, w12t, rtol = 1.0e-5)
    @test isapprox(w11.weights, w12.weights, rtol = 0.1)
    @test isapprox(w13.weights, w7.weights, rtol = 0.001)
    @test isapprox(w14.weights, w8.weights, rtol = 0.1)
    @test isapprox(w15.weights, w9.weights, rtol = 0.1)
    @test isapprox(w16.weights, w7.weights, rtol = 0.001)
    @test isapprox(w17.weights, w8.weights, rtol = 0.1)
    @test isapprox(w18.weights, w9.weights, rtol = 0.1)
    @test isapprox(w13.weights, w16.weights, rtol = 0.001)
    @test isapprox(w14.weights, w17.weights, rtol = 0.01)
    @test isapprox(w15.weights, w18.weights, rtol = 0.001)
    @test isapprox(w19.weights, w1.weights, rtol = 0.01)
    @test isapprox(w1.weights, w20.weights)
    @test isapprox(w2.weights, w21.weights)
    @test isapprox(w3.weights, w22.weights)
    @test isapprox(w4.weights, w23.weights)
    @test isapprox(w5.weights, w24.weights)
    @test isapprox(w6.weights, w25.weights)
    @test isapprox(w7.weights, w26.weights)
    @test isapprox(w8.weights, w27.weights)
    @test isapprox(w9.weights, w28.weights)
    @test isapprox(w10.weights, w29.weights)
    @test isapprox(w11.weights, w30.weights)
    @test isapprox(w12.weights, w31.weights)
    @test isapprox(w13.weights, w32.weights)
    @test isapprox(w14.weights, w33.weights)
    @test isapprox(w15.weights, w34.weights)
    @test isapprox(w16.weights, w35.weights)
    @test isapprox(w17.weights, w36.weights)
    @test isapprox(w18.weights, w37.weights)
    @test isapprox(w19.weights, w38.weights)
    @test isapprox(w1.weights, w39.weights)
    @test isapprox(w2.weights, w40.weights)
    @test isapprox(w3.weights, w41.weights)
    @test isapprox(w5.weights, w43.weights)
    @test isapprox(w6.weights, w44.weights)
    @test isapprox(w7.weights, w45.weights)
    @test isapprox(w8.weights, w46.weights)
    @test isapprox(w9.weights, w47.weights)
    @test isapprox(w10.weights, w48.weights)
    @test isapprox(w11.weights, w49.weights)
    @test isapprox(w12.weights, w50.weights)
    @test isapprox(w13.weights, w51.weights)
    @test isapprox(w14.weights, w52.weights)
    @test isapprox(w15.weights, w53.weights)
    @test isapprox(w16.weights, w54.weights)
    @test isapprox(w17.weights, w55.weights)
    @test isapprox(w18.weights, w56.weights)
    @test isapprox(w19.weights, w57.weights)
end