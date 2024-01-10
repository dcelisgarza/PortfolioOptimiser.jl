using COSMO, CSV, Clarabel, HiGHS, LinearAlgebra, OrderedCollections,
      PortfolioOptimiser,
      Statistics, Test, TimeSeries, Logging

Logging.disable_logging(Logging.Warn)

prices = TimeArray(CSV.File("./assets/stock_prices.csv"); timestamp = :date)

rf = 1.0329^(1 / 252) - 1
l = 2.0

@testset "$(:Classic), $(:Trad), $(:GMD), blank $(:OWA) and owa_w = owa_gmd(T)" begin
    portfolio = Portfolio(; prices = prices[(end - 200):end],
                          solvers = OrderedDict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                                  :params => Dict("verbose" => false,
                                                                                  "max_step_fraction" => 0.75,
                                                                                  "max_iter" => 100,
                                                                                  "equilibrate_max_iter" => 20)),
                                                :COSMO => Dict(:solver => COSMO.Optimizer,
                                                               :params => Dict("verbose" => false))))
    asset_statistics!(portfolio)

    w1 = opt_port!(portfolio; rf = rf, l = l, class = :Classic, type = :Trad,
                   rm = :GMD,
                   obj = :Min_Risk, kelly = :None,)
    risk1 = calc_risk(portfolio; type = :Trad, rm = :GMD, rf = rf)
    w2 = opt_port!(portfolio; rf = rf, l = l, class = :Classic, type = :Trad,
                   rm = :GMD,
                   obj = :Min_Risk, kelly = :Approx,)
    w3 = opt_port!(portfolio; rf = rf, l = l, class = :Classic, type = :Trad,
                   rm = :GMD,
                   obj = :Min_Risk, kelly = :Exact,)
    w4 = opt_port!(portfolio; rf = rf, l = l, class = :Classic, type = :Trad,
                   rm = :GMD,
                   obj = :Utility, kelly = :None,)
    w5 = opt_port!(portfolio; rf = rf, l = l, class = :Classic, type = :Trad,
                   rm = :GMD,
                   obj = :Utility, kelly = :Approx,)
    w6 = opt_port!(portfolio; rf = rf, l = l, class = :Classic, type = :Trad,
                   rm = :GMD,
                   obj = :Utility, kelly = :Exact,)
    w7 = opt_port!(portfolio; rf = rf, l = l, class = :Classic, type = :Trad,
                   rm = :GMD,
                   obj = :Sharpe, kelly = :None,)
    risk7 = calc_risk(portfolio; type = :Trad, rm = :GMD, rf = rf)
    ret7 = dot(portfolio.mu, w7.weights)
    w8 = opt_port!(portfolio; rf = rf, l = l, class = :Classic, type = :Trad,
                   rm = :GMD,
                   obj = :Sharpe, kelly = :Approx,)
    risk8 = calc_risk(portfolio; type = :Trad, rm = :GMD, rf = rf)
    ret8 = dot(portfolio.mu, w8.weights)
    w9 = opt_port!(portfolio; rf = rf, l = l, class = :Classic, type = :Trad,
                   rm = :GMD,
                   obj = :Sharpe, kelly = :Exact,)
    risk9 = calc_risk(portfolio; type = :Trad, rm = :GMD, rf = rf)
    ret9 = dot(portfolio.mu, w9.weights)
    w10 = opt_port!(portfolio; rf = rf, l = l, class = :Classic, type = :Trad,
                    rm = :GMD,
                    obj = :Max_Ret, kelly = :None,)
    w11 = opt_port!(portfolio; rf = rf, l = l, class = :Classic, type = :Trad,
                    rm = :GMD,
                    obj = :Max_Ret, kelly = :Approx,)
    w12 = opt_port!(portfolio; rf = rf, l = l, class = :Classic, type = :Trad,
                    rm = :GMD,
                    obj = :Max_Ret, kelly = :Exact,)
    setproperty!(portfolio, :mu_l, ret7)
    w13 = opt_port!(portfolio; rf = rf, l = l, class = :Classic, type = :Trad,
                    rm = :GMD,
                    obj = :Min_Risk, kelly = :None,)
    setproperty!(portfolio, :mu_l, ret8)
    w14 = opt_port!(portfolio; rf = rf, l = l, class = :Classic, type = :Trad,
                    rm = :GMD,
                    obj = :Min_Risk, kelly = :None,)
    setproperty!(portfolio, :mu_l, ret9)
    w15 = opt_port!(portfolio; rf = rf, l = l, class = :Classic, type = :Trad,
                    rm = :GMD,
                    obj = :Min_Risk, kelly = :None,)
    setproperty!(portfolio, :mu_l, Inf)
    rmf = Symbol(lowercase(string(:GMD)) * "_u")
    setproperty!(portfolio, rmf, risk7)
    w16 = opt_port!(portfolio; rf = rf, l = l, class = :Classic, type = :Trad,
                    rm = :GMD,
                    obj = :Max_Ret, kelly = :None,)
    setproperty!(portfolio, rmf, risk8)
    w17 = opt_port!(portfolio; rf = rf, l = l, class = :Classic, type = :Trad,
                    rm = :GMD,
                    obj = :Max_Ret, kelly = :None,)
    setproperty!(portfolio, rmf, risk9)
    w18 = opt_port!(portfolio; rf = rf, l = l, class = :Classic, type = :Trad,
                    rm = :GMD,
                    obj = :Max_Ret, kelly = :None,)
    setproperty!(portfolio, rmf, risk1)
    w19 = opt_port!(portfolio; rf = rf, l = l, class = :Classic, type = :Trad,
                    rm = :GMD,
                    obj = :Sharpe, kelly = :None,)
    setproperty!(portfolio, rmf, Inf)
    w20 = opt_port!(portfolio; rf = rf, l = l, class = :Classic, type = :Trad,
                    rm = :OWA,
                    obj = :Min_Risk, kelly = :None,)
    risk20 = calc_risk(portfolio; type = :Trad, rm = :OWA, rf = rf)
    w21 = opt_port!(portfolio; rf = rf, l = l, class = :Classic, type = :Trad,
                    rm = :OWA,
                    obj = :Min_Risk, kelly = :Approx,)
    w22 = opt_port!(portfolio; rf = rf, l = l, class = :Classic, type = :Trad,
                    rm = :OWA,
                    obj = :Min_Risk, kelly = :Exact,)
    w23 = opt_port!(portfolio; rf = rf, l = l, class = :Classic, type = :Trad,
                    rm = :OWA,
                    obj = :Utility, kelly = :None,)
    w24 = opt_port!(portfolio; rf = rf, l = l, class = :Classic, type = :Trad,
                    rm = :OWA,
                    obj = :Utility, kelly = :Approx,)
    w25 = opt_port!(portfolio; rf = rf, l = l, class = :Classic, type = :Trad,
                    rm = :OWA,
                    obj = :Utility, kelly = :Exact,)
    w26 = opt_port!(portfolio; rf = rf, l = l, class = :Classic, type = :Trad,
                    rm = :OWA,
                    obj = :Sharpe, kelly = :None,)
    risk26 = calc_risk(portfolio; type = :Trad, rm = :OWA, rf = rf)
    ret26 = dot(portfolio.mu, w26.weights)
    w27 = opt_port!(portfolio; rf = rf, l = l, class = :Classic, type = :Trad,
                    rm = :OWA,
                    obj = :Sharpe, kelly = :Approx,)
    risk27 = calc_risk(portfolio; type = :Trad, rm = :OWA, rf = rf)
    ret27 = dot(portfolio.mu, w27.weights)
    w28 = opt_port!(portfolio; rf = rf, l = l, class = :Classic, type = :Trad,
                    rm = :OWA,
                    obj = :Sharpe, kelly = :Exact,)
    risk28 = calc_risk(portfolio; type = :Trad, rm = :OWA, rf = rf)
    ret28 = dot(portfolio.mu, w28.weights)
    w29 = opt_port!(portfolio; rf = rf, l = l, class = :Classic, type = :Trad,
                    rm = :OWA,
                    obj = :Max_Ret, kelly = :None,)
    w30 = opt_port!(portfolio; rf = rf, l = l, class = :Classic, type = :Trad,
                    rm = :OWA,
                    obj = :Max_Ret, kelly = :Approx,)
    w31 = opt_port!(portfolio; rf = rf, l = l, class = :Classic, type = :Trad,
                    rm = :OWA,
                    obj = :Max_Ret, kelly = :Exact,)
    setproperty!(portfolio, :mu_l, ret26)
    w32 = opt_port!(portfolio; rf = rf, l = l, class = :Classic, type = :Trad,
                    rm = :OWA,
                    obj = :Min_Risk, kelly = :None,)
    setproperty!(portfolio, :mu_l, ret27)
    w33 = opt_port!(portfolio; rf = rf, l = l, class = :Classic, type = :Trad,
                    rm = :OWA,
                    obj = :Min_Risk, kelly = :None,)
    setproperty!(portfolio, :mu_l, ret28)
    w34 = opt_port!(portfolio; rf = rf, l = l, class = :Classic, type = :Trad,
                    rm = :OWA,
                    obj = :Min_Risk, kelly = :None,)
    setproperty!(portfolio, :mu_l, Inf)
    rmf = Symbol(lowercase(string(:OWA)) * "_u")
    setproperty!(portfolio, rmf, risk26)
    w35 = opt_port!(portfolio; rf = rf, l = l, class = :Classic, type = :Trad,
                    rm = :OWA,
                    obj = :Max_Ret, kelly = :None,)
    setproperty!(portfolio, rmf, risk27)
    w36 = opt_port!(portfolio; rf = rf, l = l, class = :Classic, type = :Trad,
                    rm = :OWA,
                    obj = :Max_Ret, kelly = :None,)
    setproperty!(portfolio, rmf, risk28)
    w37 = opt_port!(portfolio; rf = rf, l = l, class = :Classic, type = :Trad,
                    rm = :OWA,
                    obj = :Max_Ret, kelly = :None,)
    setproperty!(portfolio, rmf, risk20)
    w38 = opt_port!(portfolio; rf = rf, l = l, class = :Classic, type = :Trad,
                    rm = :OWA,
                    obj = :Sharpe, kelly = :None,)
    setproperty!(portfolio, rmf, Inf)
    portfolio.owa_w = owa_gmd(size(portfolio.returns, 1))
    w39 = opt_port!(portfolio; rf = rf, l = l, class = :Classic, type = :Trad,
                    rm = :OWA,
                    obj = :Min_Risk, kelly = :None,)
    risk39 = calc_risk(portfolio; type = :Trad, rm = :OWA, rf = rf)
    w40 = opt_port!(portfolio; rf = rf, l = l, class = :Classic, type = :Trad,
                    rm = :OWA,
                    obj = :Min_Risk, kelly = :Approx,)
    w41 = opt_port!(portfolio; rf = rf, l = l, class = :Classic, type = :Trad,
                    rm = :OWA,
                    obj = :Min_Risk, kelly = :Exact,)
    w42 = opt_port!(portfolio; rf = rf, l = l, class = :Classic, type = :Trad,
                    rm = :OWA,
                    obj = :Utility, kelly = :None,)
    w43 = opt_port!(portfolio; rf = rf, l = l, class = :Classic, type = :Trad,
                    rm = :OWA,
                    obj = :Utility, kelly = :Approx,)
    w44 = opt_port!(portfolio; rf = rf, l = l, class = :Classic, type = :Trad,
                    rm = :OWA,
                    obj = :Utility, kelly = :Exact,)
    w45 = opt_port!(portfolio; rf = rf, l = l, class = :Classic, type = :Trad,
                    rm = :OWA,
                    obj = :Sharpe, kelly = :None,)
    risk45 = calc_risk(portfolio; type = :Trad, rm = :OWA, rf = rf)
    ret45 = dot(portfolio.mu, w45.weights)
    w46 = opt_port!(portfolio; rf = rf, l = l, class = :Classic, type = :Trad,
                    rm = :OWA,
                    obj = :Sharpe, kelly = :Approx,)
    risk46 = calc_risk(portfolio; type = :Trad, rm = :OWA, rf = rf)
    ret46 = dot(portfolio.mu, w46.weights)
    w47 = opt_port!(portfolio; rf = rf, l = l, class = :Classic, type = :Trad,
                    rm = :OWA,
                    obj = :Sharpe, kelly = :Exact,)
    risk47 = calc_risk(portfolio; type = :Trad, rm = :OWA, rf = rf)
    ret47 = dot(portfolio.mu, w47.weights)
    w48 = opt_port!(portfolio; rf = rf, l = l, class = :Classic, type = :Trad,
                    rm = :OWA,
                    obj = :Max_Ret, kelly = :None,)
    w49 = opt_port!(portfolio; rf = rf, l = l, class = :Classic, type = :Trad,
                    rm = :OWA,
                    obj = :Max_Ret, kelly = :Approx,)
    w50 = opt_port!(portfolio; rf = rf, l = l, class = :Classic, type = :Trad,
                    rm = :OWA,
                    obj = :Max_Ret, kelly = :Exact,)
    setproperty!(portfolio, :mu_l, ret45)
    w51 = opt_port!(portfolio; rf = rf, l = l, class = :Classic, type = :Trad,
                    rm = :OWA,
                    obj = :Min_Risk, kelly = :None,)
    setproperty!(portfolio, :mu_l, ret46)
    w52 = opt_port!(portfolio; rf = rf, l = l, class = :Classic, type = :Trad,
                    rm = :OWA,
                    obj = :Min_Risk, kelly = :None,)
    setproperty!(portfolio, :mu_l, ret47)
    w53 = opt_port!(portfolio; rf = rf, l = l, class = :Classic, type = :Trad,
                    rm = :OWA,
                    obj = :Min_Risk, kelly = :None,)
    setproperty!(portfolio, :mu_l, Inf)
    rmf = Symbol(lowercase(string(:OWA)) * "_u")
    setproperty!(portfolio, rmf, risk45)
    w54 = opt_port!(portfolio; rf = rf, l = l, class = :Classic, type = :Trad,
                    rm = :OWA,
                    obj = :Max_Ret, kelly = :None,)
    setproperty!(portfolio, rmf, risk46)
    w55 = opt_port!(portfolio; rf = rf, l = l, class = :Classic, type = :Trad,
                    rm = :OWA,
                    obj = :Max_Ret, kelly = :None,)
    setproperty!(portfolio, rmf, risk47)
    w56 = opt_port!(portfolio; rf = rf, l = l, class = :Classic, type = :Trad,
                    rm = :OWA,
                    obj = :Max_Ret, kelly = :None,)
    setproperty!(portfolio, rmf, risk39)
    w57 = opt_port!(portfolio; rf = rf, l = l, class = :Classic, type = :Trad,
                    rm = :OWA,
                    obj = :Sharpe, kelly = :None,)
    setproperty!(portfolio, rmf, Inf)

    w1t = [2.859922116198051e-10, 1.2839851061414282e-9, 6.706646698191418e-10,
           0.011313263237504052, 0.041188384886560354, 0.021164906062678113,
           9.703581331630189e-11, 0.04727424011116681, 5.765539509407631e-10,
           4.5707249638450353e-10, 0.07079102697763319, 2.3691876986431332e-11,
           6.862345293974742e-11, 0.2926160718937385, 2.627931180682999e-11,
           0.014804760223130658, 0.055360563578446625, 0.22572350787381906,
           0.008097074123824198, 0.2116661975415996]

    w2t = [5.251492167108595e-10, 2.365324942455108e-9, 1.228133964759497e-9,
           0.01136625390653424, 0.04115041304677448, 0.021118575976097856,
           1.7826397676563226e-10, 0.04726442195474365, 1.0573388621330249e-9,
           8.386390023197234e-10, 0.07085089072254845, 4.421333531677392e-11,
           1.2543433104522538e-10, 0.2925312243040699, 4.882105192830059e-11,
           0.014796401887494866, 0.055296423856387264, 0.225728025547612,
           0.008121179213586627, 0.21177618317283206]

    w3t = [5.354026704869777e-10, 2.2436905049917003e-9, 1.1821239557765714e-9,
           0.011356430502054034, 0.04115626893621375, 0.021118125043514093,
           2.1573924944334443e-10, 0.047258599507084435, 1.0190748261693879e-9,
           8.182234303696856e-10, 0.07083743989900024, 9.061256814168461e-11,
           1.6821477600735128e-10, 0.2925476241867142, 9.45240785865248e-11,
           0.01479824618994179, 0.05530574095746841, 0.22573566536305198,
           0.008120334785801188, 0.21176551826154985]

    w4t = [4.395442944440727e-10, 2.1495615257047445e-9, 8.700158416075159e-10,
           0.006589303508967539, 0.05330845589356504, 7.632737386668005e-10,
           9.831583865942048e-11, 0.06008335161013898, 1.3797121555924764e-9,
           3.0160588760343415e-9, 0.06394372429204083, 4.486199590323317e-11,
           1.0975948346699443e-10, 0.24314317639043878, 5.90094645737884e-11,
           0.018483948217979402, 0.08594753958504305, 0.216033164253688,
           0.05666930366548,
           0.19579802365254514]

    w5t = [2.1373923858202174e-10, 1.047935128182663e-9, 4.2352337620182766e-10,
           0.006623632023945508, 0.053262035174195796, 3.748985425782847e-10,
           4.8604916317016427e-11, 0.059837358756352356, 6.629955827787793e-10,
           1.4334114636079199e-9, 0.06407645387087887, 2.2564457261760664e-11,
           5.371370812720617e-11, 0.24323140594664305, 2.9658041881066845e-11,
           0.018454146656654288, 0.0859437656795087, 0.21613799090173264,
           0.05652392850317017, 0.19590927817587422]

    w6t = [1.7127252443308418e-7, 6.934858519404509e-7, 2.866291591186648e-7,
           0.006553485493782127, 0.05314502242400377, 2.3075679946487433e-7,
           5.766434552633233e-8, 0.059965342332728655, 7.108722767488626e-7,
           8.499172115788923e-7, 0.06397266225548641, 4.906525899110893e-8,
           6.017663441983441e-8, 0.243323504645496, 4.6587059930614004e-8,
           0.018708368507070104, 0.08575936499218657, 0.2163840887167888,
           0.05625377780709815, 0.19593122639823732]

    w7t = [1.0960279114854144e-12, 4.2713859884519724e-11,
           3.322171869553163e-12,
           3.8205053405909246e-11, 0.2107929632511357, 2.439956520619061e-11,
           2.2427021896796488e-11, 0.011060068605326014, 0.021557962662139638,
           1.524860659816748e-10, 2.143824873504224e-11, 1.626628185083569e-11,
           2.276043092971382e-11, 1.0059568286002983e-11, 1.872567893305395e-11,
           0.0977182789979517, 0.39859819200001834, 5.499837267441688e-11,
           0.26027253402707684, 2.7453438274980142e-11]

    w8t = [1.0155020749727522e-10, 2.38178191529736e-10, 1.1584451510196998e-10,
           2.1758373547817112e-10, 0.20152993710189268, 2.993101288477597e-11,
           3.6439928804387634e-11, 0.028636902063586874, 0.005806401898518205,
           6.433916064904551e-10, 1.843996026263977e-10, 5.527116705376367e-11,
           3.602234132572901e-11, 1.4360190351319453e-10, 4.813709296729719e-11,
           0.09660151833030618, 0.38225707069143156, 3.075324109719233e-10,
           0.28516816754177193, 2.1460877457707472e-10]

    w9t = [5.671157677840885e-7, 1.2733743334667473e-6, 6.370657936156067e-7,
           1.239290084842503e-6, 0.20705790824767295, 1.7306287519061096e-7,
           2.1949092659034066e-7, 0.01502662025379091, 0.01270587595162983,
           2.877209394665577e-6, 8.889750437093147e-7, 3.269795768965802e-7,
           2.0703480742609938e-7, 7.234077461496872e-7, 2.7060737069570363e-7,
           0.09794013285762235, 0.39947829026120407, 1.3837803825062758e-6,
           0.26777939992827077, 9.85105705745203e-7]

    w10t = [8.713294565329801e-10, 1.4655494663368051e-9, 9.339861464983668e-10,
            2.1822916966782698e-9, 0.9999999341529429, 3.1514150951638473e-10,
            4.623635446692933e-10, 1.3314213108887623e-9, 5.686161252843585e-9,
            1.371922705853848e-9, 7.146550790353018e-10, 5.740256406691589e-10,
            3.664169883135698e-10, 7.000039902341564e-10,
            4.5030867400627253e-10,
            6.996286227848405e-9, 3.6287634044926756e-8, 9.000571133076656e-10,
            3.4765700831013475e-9, 7.60932312745152e-10]

    w11t = [5.995643893118203e-12, 1.7788172572403522e-11, 7.42248967697158e-12,
            2.5855688894534932e-11, 0.8577737311672241, 5.084251669919802e-12,
            1.8918809363559223e-12, 1.7873684105427732e-11,
            9.127337657198234e-11,
            1.8171837160854858e-11, 3.850566386470494e-12,
            2.5508401127966545e-13,
            4.0216144858012615e-12, 3.339109115577017e-12,
            2.0662179397071288e-12,
            1.8569702892630115e-10, 0.14222626837294614, 8.073610828892573e-12,
            5.65606511499808e-11, 4.608669048088745e-12]

    w12t = [6.8783457384567556e-9, 1.2274774655426258e-8, 7.210231160923748e-9,
            1.6157723542114e-8, 0.8926318884419521, 2.236669895286051e-9,
            3.0721278132094936e-9, 1.1931574111287059e-8, 5.313990853277208e-8,
            1.1675553033694841e-8, 5.76375880463489e-9, 4.1812752119557695e-9,
            2.3871108169602047e-9, 5.509568787249582e-9, 2.9557632944416428e-9,
            9.517716377203498e-8, 0.10736782516025675, 7.544504043725093e-9,
            3.230881102698668e-8, 5.9929266940059665e-9]

    w13t = [2.111064693976598e-12, 4.199098436247614e-11,
            2.4277750791679688e-12,
            3.8079944685205154e-11, 0.21084481684472242, 2.6325394341890063e-11,
            2.398425045782114e-11, 0.011087215230686672, 0.021727930888863614,
            1.5980784508945338e-10, 2.2534788169784417e-11,
            1.7758263471786547e-11,
            2.4160204726553847e-11, 1.030058571203951e-11,
            2.000500776826796e-11,
            0.09769732518814264, 0.39851241713315694, 5.853448351366192e-11,
            0.26013029423727785, 2.9129230394432605e-11]

    w14t = [8.942270766111967e-13, 4.997065086540317e-11, 4.597515424452085e-12,
            4.14025186234731e-11, 0.2045787142958964, 2.757625635103089e-11,
            2.5526588174105618e-11, 0.028823839199628453, 0.00935681866157209,
            2.2178037048090422e-10, 3.349076787579454e-11, 1.87998796931707e-11,
            2.5264985600201933e-11, 1.653886341218741e-11,
            2.0604214115929157e-11,
            0.09537027357843764, 0.377524299391081, 8.51485669294363e-11,
            0.2843460542557699, 4.601891415928919e-11]

    w15t = [5.1306740699054454e-12, 3.498203888295839e-11,
            9.443660977510647e-13,
            3.0051209746434546e-11, 0.2096190099044751, 2.6830393435136183e-11,
            2.4866423157271724e-11, 0.015388952005987784, 0.01947723659851483,
            1.4698330462316705e-10, 1.836237237661002e-11,
            1.931290540546062e-11,
            2.4901861449564757e-11, 6.724696953439446e-12,
            2.113917582584407e-11,
            0.09756081719654436, 0.3941356326147162, 5.2721875186497566e-11,
            0.26381835124131175, 2.5498813689609923e-11]

    w16t = [8.645417462703117e-13, 9.74931569493778e-12, 1.7691244640820176e-12,
            8.874979261575232e-12, 0.21078639257923237, 3.922009126537224e-12,
            3.4338064345481404e-12, 0.011063602423968855, 0.02153048309961715,
            3.387355507162292e-11, 5.799061166068984e-12,
            2.2147166479480898e-12,
            3.476808868173112e-12, 3.3237370929321917e-12,
            2.640299454582965e-12,
            0.09773989516924422, 0.3986125551970178, 1.3091156994666361e-11,
            0.26026707143077404, 7.112393143958492e-12]

    w17t = [2.196665141984919e-12, 1.2060486285149243e-11,
            3.2469345250787356e-12,
            1.035882803391085e-11, 0.20454337993180025, 2.8837388491504335e-12,
            2.4643498937450153e-12, 0.02883361320119504, 0.009228447548408824,
            4.5440170588235665e-11, 8.78290112077899e-12,
            1.1948258696922626e-12,
            2.427952696676779e-12, 5.51739407846932e-12, 1.524788450375429e-12,
            0.09537008234204332, 0.3777286103721778, 1.876506344052584e-11,
            0.2842958664763445, 1.116624337339785e-11]

    w18t = [9.09711465126155e-13, 9.449290266913617e-12, 1.7739420562277521e-12,
            8.37151312097126e-12, 0.20966915944272185, 3.5479502740177604e-12,
            3.0757608336758066e-12, 0.015378180154910333, 0.01950214340438561,
            3.35180762679845e-11, 5.785868839429511e-12, 1.9479838142418733e-12,
            3.1193764890204444e-12, 3.3382615997835853e-12,
            2.3158677660472516e-12,
            0.09758505196452177, 0.3941957379297834, 1.3034815581554832e-11,
            0.2636697270062483, 7.240147916177639e-12]

    w19t = [1.853222332200707e-9, 7.77154342441663e-9, 4.361453531340065e-9,
            0.011268503190433975, 0.041024263447110715, 0.019990798875061184,
            6.32380113701166e-10, 0.04768163200097348, 3.919590544731981e-9,
            3.2835408002817512e-9, 0.07074337720718103, 1.626354865702736e-10,
            4.78095986084816e-10, 0.29162850516446387, 1.824495731211796e-10,
            0.014716325550611356, 0.05627433067844324, 0.2257134112824113,
            0.009011591493696645, 0.2119472384647015]

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

@testset "$(:Classic), $(:Trad), $(:RG)" begin
    portfolio = Portfolio(; prices = prices[(end - 200):end],
                          solvers = OrderedDict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                                  :params => Dict("verbose" => false,
                                                                                  "max_step_fraction" => 0.75)),
                                                :COSMO => Dict(:solver => COSMO.Optimizer,
                                                               :params => Dict("verbose" => false))))
    asset_statistics!(portfolio)

    w1 = opt_port!(portfolio; rf = rf, l = l, class = :Classic, type = :Trad,
                   rm = :RG,
                   obj = :Min_Risk, kelly = :None,)
    risk1 = calc_risk(portfolio; type = :Trad, rm = :RG, rf = rf)
    w2 = opt_port!(portfolio; rf = rf, l = l, class = :Classic, type = :Trad,
                   rm = :RG,
                   obj = :Min_Risk, kelly = :Approx,)
    w3 = opt_port!(portfolio; rf = rf, l = l, class = :Classic, type = :Trad,
                   rm = :RG,
                   obj = :Min_Risk, kelly = :Exact,)
    w4 = opt_port!(portfolio; rf = rf, l = l, class = :Classic, type = :Trad,
                   rm = :RG,
                   obj = :Utility, kelly = :None,)
    w5 = opt_port!(portfolio; rf = rf, l = l, class = :Classic, type = :Trad,
                   rm = :RG,
                   obj = :Utility, kelly = :Approx,)
    w6 = opt_port!(portfolio; rf = rf, l = l, class = :Classic, type = :Trad,
                   rm = :RG,
                   obj = :Utility, kelly = :Exact,)
    w7 = opt_port!(portfolio; rf = rf, l = l, class = :Classic, type = :Trad,
                   rm = :RG,
                   obj = :Sharpe, kelly = :None,)
    risk7 = calc_risk(portfolio; type = :Trad, rm = :RG, rf = rf)
    ret7 = dot(portfolio.mu, w7.weights)
    w8 = opt_port!(portfolio; rf = rf, l = l, class = :Classic, type = :Trad,
                   rm = :RG,
                   obj = :Sharpe, kelly = :Approx,)
    risk8 = calc_risk(portfolio; type = :Trad, rm = :RG, rf = rf)
    ret8 = dot(portfolio.mu, w8.weights)
    w10 = opt_port!(portfolio; rf = rf, l = l, class = :Classic, type = :Trad,
                    rm = :RG,
                    obj = :Max_Ret, kelly = :None,)
    w11 = opt_port!(portfolio; rf = rf, l = l, class = :Classic, type = :Trad,
                    rm = :RG,
                    obj = :Max_Ret, kelly = :Approx,)
    w12 = opt_port!(portfolio; rf = rf, l = l, class = :Classic, type = :Trad,
                    rm = :RG,
                    obj = :Max_Ret, kelly = :Exact,)
    setproperty!(portfolio, :mu_l, ret7)
    w13 = opt_port!(portfolio; rf = rf, l = l, class = :Classic, type = :Trad,
                    rm = :RG,
                    obj = :Min_Risk, kelly = :None,)
    setproperty!(portfolio, :mu_l, ret8)
    w14 = opt_port!(portfolio; rf = rf, l = l, class = :Classic, type = :Trad,
                    rm = :RG,
                    obj = :Min_Risk, kelly = :None,)
    setproperty!(portfolio, :mu_l, Inf)
    rmf = Symbol(lowercase(string(:RG)) * "_u")
    setproperty!(portfolio, rmf, risk7)
    w16 = opt_port!(portfolio; rf = rf, l = l, class = :Classic, type = :Trad,
                    rm = :RG,
                    obj = :Max_Ret, kelly = :None,)
    setproperty!(portfolio, rmf, risk8)
    w17 = opt_port!(portfolio; rf = rf, l = l, class = :Classic, type = :Trad,
                    rm = :RG,
                    obj = :Max_Ret, kelly = :None,)
    setproperty!(portfolio, rmf, risk1)
    w19 = opt_port!(portfolio; rf = rf, l = l, class = :Classic, type = :Trad,
                    rm = :RG,
                    obj = :Sharpe, kelly = :None,)
    setproperty!(portfolio, rmf, Inf)

    w1t = [4.911237220109838e-14, 0.12080622849296571, 1.059356361390728e-11,
           0.010186673770960653, 1.1761029979686987e-13, 0.07374195232209402,
           1.5137783944058707e-12, 0.03614126086809069, 1.158366863783836e-12,
           9.738440344047313e-13, 0.35461674321178027, 5.117183922702972e-13,
           1.581681376477715e-12, 0.009915762320151547, 8.103594452583275e-13,
           1.6237275286009668e-12, 0.11690923379335506, 2.2061802929879864e-14,
           4.174977331409494e-13, 0.27768214520122886]

    w2t = [3.8103948068553603e-13, 0.12080622901533082, 5.1433180277351925e-11,
           0.010186673839978315, 5.399490769239643e-13, 0.07374195247988,
           7.267436278938843e-12, 0.03614126091815042, 5.422321295162276e-12,
           3.362385235212633e-12, 0.35461674317653374, 2.451603864906192e-12,
           7.655069818679018e-12, 0.009915762420370688, 2.910952848464599e-12,
           5.8610965884333956e-12, 0.11690923305024577, 1.0025670895460215e-12,
           2.132566907425521e-12, 0.27768214500908994]

    w3t = [2.223145952146828e-11, 0.12080623046744454, 1.4967781202316547e-10,
           0.010186674186813583, 3.0930164067004603e-11, 0.07374195295478396,
           6.8488712449068216e-12, 0.036141261027605835, 9.997394356672985e-12,
           1.4065772019548506e-11, 0.35461674318829717, 1.6559905037978058e-11,
           2.922918291003705e-12, 0.00991576259625932, 1.517620136606096e-11,
           2.7038223888288796e-11, 0.1169092306848456, 2.5453279592674617e-11,
           1.8265927438892204e-11, 0.27768214455478196]

    w4t = [1.5218960826317059e-13, 0.12276734050506966, 0.008565562635440845,
           3.8086405878606735e-12, 8.614088923893088e-14, 0.06024115368418488,
           2.511737723477254e-12, 0.03543856388632871, 1.8199328483712365e-12,
           1.1663940571002837e-12, 0.3596410911054393, 1.136225975160405e-12,
           2.857215093924921e-12, 0.007480617549000438, 1.4575209615511937e-12,
           3.5506933165276716e-12, 0.1305269569777664, 1.1525649478062172e-12,
           4.308283867966185e-13, 0.27533871363663986]

    w5t = [1.8226000390737606e-13, 0.12276734045092537, 0.008565562620567194,
           2.439747767851362e-12, 3.897627319435204e-14, 0.06024115368617171,
           1.5539549841691534e-12, 0.03543856388303625, 1.0025228847537055e-12,
           8.586356828254743e-13, 0.359641091104997, 7.931026024779409e-13,
           1.6527895569897798e-12, 0.0074806175426397874, 9.597065199308827e-13,
           1.5313576071902954e-12, 0.1305269570442035, 8.070790829056648e-13,
           3.506082479997246e-13, 0.27533871365528845]

    w6t = [1.7464672063730016e-11, 0.12276734154675845, 0.00856556301132976,
           3.7359576555297145e-11, 2.1138690074548852e-11, 0.06024115357599186,
           4.192209212498068e-12, 0.03543856392784425, 7.212662899636176e-12,
           1.1224471750101623e-11, 0.35964109124697374, 1.077203011880652e-11,
           1.2789525771363605e-12, 0.0074806175654811785, 8.8814720123182e-12,
           1.757551175289691e-11, 0.13052695573583922, 2.8944405947489043e-11,
           1.524546600846002e-11, 0.2753387132084914]

    w7t = [5.772112458682092e-11, 4.590320818820348e-11, 4.371244099615786e-11,
           4.7650025204031245e-11, 0.11690000617850771, 6.398985501559165e-11,
           5.854782252033428e-11, 1.9433718592164295e-11, 4.224295985554396e-11,
           3.771496141321701e-11, 5.3698729859692593e-11, 6.377096516993624e-11,
           6.748336078292134e-11, 4.916620489731109e-11, 6.346748119531207e-11,
           8.943374502389502e-10, 0.8830999920967746, 4.923907187473908e-11,
           1.4911114748916195e-11, 5.1727215933774835e-11]

    w8t = [1.8342066357613254e-11, 3.855541712193138e-11, 3.459103850680907e-11,
           3.5993353508408624e-11, 0.11690051277619719, 3.759314018825843e-12,
           1.1625985278167321e-11, 1.2709738884155148e-10,
           7.277657506826433e-11,
           6.305252275646322e-11, 3.076041440864706e-11, 1.1152570932849577e-11,
           2.664385927602493e-12, 3.832417831853527e-11, 7.454348101949232e-12,
           3.07006342205312e-8, 0.883099455814034, 3.337740761909485e-11,
           1.4488962939829924e-10, 3.4717931922136874e-11]

    w10t = [8.923307373487656e-10, 1.5032949673873539e-9, 9.565584921031364e-10,
            2.234246280605458e-9, 0.9999999335066485, 3.2137424889799944e-10,
            4.69426067980136e-10, 1.3663520558679047e-9, 5.769405035482146e-9,
            1.4079050962626806e-9, 7.328658337490239e-10, 5.833011280316127e-10,
            3.6923029326302933e-10, 7.176786838885732e-10,
            4.5692176226819763e-10,
            7.084879516281728e-9, 3.637437458194181e-8, 9.236728792495098e-10,
            3.5491131514079583e-9, 7.804206038609059e-10]

    w11t = [6.570639760206459e-12, 1.934271719617747e-11, 8.128186692450276e-12,
            2.803744923730736e-11, 0.8577737352454997, 5.336375969196261e-12,
            1.9852511833111874e-12, 1.945348035496646e-11,
            9.850455954637638e-11,
            1.974312495312116e-11, 4.259176650582746e-12,
            3.6610253951453656e-13,
            4.2126095690719996e-12, 3.7156001481586565e-12,
            2.2245822753049234e-12,
            1.9976461196558243e-10, 0.14222626425777468, 8.844921761594041e-12,
            6.113579297990577e-11, 5.100463871065917e-12]

    w12t = [4.502088679248276e-9, 8.026789256301228e-9, 4.708313192247162e-9,
            1.0563193089964122e-8, 0.8926330429712972, 1.4761874423354312e-9,
            2.0119573802827614e-9, 7.788743446209408e-9, 3.498564106619679e-8,
            7.6060043167538e-9, 3.767980708528448e-9, 2.731684532585433e-9,
            1.5623831944020235e-9, 3.602618817538397e-9, 1.933744486474113e-9,
            6.263510821188843e-8, 0.1073667690973204, 4.928736269105768e-9,
            2.1186236622823188e-8, 3.9139717053691975e-9]

    w13t = [3.8700315197280095e-11, 2.6468299550616142e-11,
            3.172118949612173e-11,
            2.1122012132925015e-11, 0.11689998076987461, 1.1003410852202503e-10,
            6.653085923479279e-11, 5.020131416919809e-10,
            1.4367963673865566e-11,
            2.4246212954331247e-11, 4.22603301015764e-11, 6.064273494058325e-11,
            1.0098194839107344e-10, 4.328698990119692e-11, 7.37654253121017e-11,
            6.501569856375878e-11, 0.8831000179206832, 3.499952488922403e-11,
            1.2697087064115542e-11, 4.058834322235273e-11]

    w14t = [3.738839027042623e-12, 3.2337171283963043e-12,
            3.3575407286489112e-12,
            3.0442943129012785e-12, 0.11690046813294093, 5.876630941309367e-12,
            4.581904873513019e-12, 1.0303094844951605e-10,
            2.6007307380729437e-12,
            2.913710232098732e-12, 3.7412265432400904e-12,
            4.4386659062532255e-12,
            5.6472280265669996e-12, 3.5626174776613003e-12,
            4.81121876856421e-12,
            1.237372182680866e-11, 0.8830995316910026, 3.5017501661296126e-12,
            2.0487925663320552e-12, 3.552997348756605e-12]

    w16t = [2.716771124469647e-13, 2.761023205879752e-11, 6.499891412700421e-12,
            5.234265601174752e-13, 0.11689987313735063, 7.326004421112206e-14,
            1.114164841753301e-13, 1.1051256563558323e-8, 8.590668010168357e-13,
            1.8059003017468587e-11, 3.33280873366803e-9, 8.885990476213998e-13,
            1.637173511141264e-12, 3.2713035039324393e-9,
            1.0328174762506177e-12,
            5.1410430202148184e-11, 0.8831001062797235, 3.106277948976559e-12,
            5.935577387170622e-12, 2.809538081525264e-9]

    w17t = [9.885535330912956e-14, 2.26052143179359e-12, 1.0856986953716035e-12,
            2.756617675579683e-13, 0.11690045035043688, 4.424660598138524e-13,
            1.3781201487338703e-13, 1.7003557870422732e-8,
            6.065511211705249e-13,
            2.6860035324289382e-12, 2.3323564060638427e-10,
            3.0946574882827813e-13,
            5.906492016133197e-13, 1.3189967249032733e-10, 4.1690156524589e-13,
            9.971365585305323e-12, 0.8830995320699504, 6.782377837103799e-13,
            2.8918272754598856e-12, 1.8846741284501473e-10]

    w19t = [2.675767996540718e-14, 0.12080622903411996, 2.9558241375349435e-9,
            0.010186670224881579, 7.129435002294713e-14, 0.0737419476147184,
            1.7480658940579254e-13, 0.03614126061360424, 1.2876257080423246e-13,
            5.909690439506736e-14, 0.3546167449562341, 7.393224284873788e-14,
            1.96903680920803e-13, 0.009915761455550983, 8.64193563842051e-14,
            2.793069991230167e-13, 0.11690923870554527, 1.1229025156636944e-14,
            4.96570759364685e-14, 0.2776821444383632]

    @test isapprox(w1.weights, w1t)
    @test isapprox(w2.weights, w2t)
    @test isapprox(w3.weights, w3t)
    @test isapprox(w2.weights, w3.weights, rtol = 1e-8)
    @test isapprox(w4.weights, w4t)
    @test isapprox(w5.weights, w5t)
    @test isapprox(w6.weights, w6t)
    @test isapprox(w5.weights, w6.weights, rtol = 1e-8)
    @test isapprox(w7.weights, w7t)
    @test isapprox(w8.weights, w8t)
    @test isapprox(w10.weights, w10t)
    @test isapprox(w11.weights, w11t)
    @test isapprox(w12.weights, w12t, rtol = 1.0e-5)
    @test isapprox(w11.weights, w12.weights, rtol = 1e-1)
    @test isapprox(w13.weights, w7.weights, rtol = 1e-7)
    @test isapprox(w14.weights, w8.weights, rtol = 1e-6)
    @test isapprox(w16.weights, w7.weights, rtol = 1e-6)
    @test isapprox(w17.weights, w8.weights, rtol = 1e-6)
    @test isapprox(w13.weights, w16.weights, rtol = 1e-6)
    @test isapprox(w14.weights, w17.weights, rtol = 1e-7)
    @test isapprox(w19.weights, w1.weights, rtol = 1e-7)
end

@testset "$(:Classic), $(:Trad), $(:RCVaR)" begin
    portfolio = Portfolio(; prices = prices[(end - 200):end],
                          solvers = OrderedDict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                                  :params => Dict("verbose" => false,
                                                                                  "max_step_fraction" => 0.75)),
                                                :COSMO => Dict(:solver => COSMO.Optimizer,
                                                               :params => Dict("verbose" => false))))
    asset_statistics!(portfolio)

    w1 = opt_port!(portfolio; rf = rf, l = l, class = :Classic, type = :Trad,
                   rm = :RCVaR,
                   obj = :Min_Risk, kelly = :None,)
    risk1 = calc_risk(portfolio; type = :Trad, rm = :RCVaR, rf = rf)
    w2 = opt_port!(portfolio; rf = rf, l = l, class = :Classic, type = :Trad,
                   rm = :RCVaR,
                   obj = :Min_Risk, kelly = :Approx,)
    w3 = opt_port!(portfolio; rf = rf, l = l, class = :Classic, type = :Trad,
                   rm = :RCVaR,
                   obj = :Min_Risk, kelly = :Exact,)
    w4 = opt_port!(portfolio; rf = rf, l = l, class = :Classic, type = :Trad,
                   rm = :RCVaR,
                   obj = :Utility, kelly = :None,)
    w5 = opt_port!(portfolio; rf = rf, l = l, class = :Classic, type = :Trad,
                   rm = :RCVaR,
                   obj = :Utility, kelly = :Approx,)
    w6 = opt_port!(portfolio; rf = rf, l = l, class = :Classic, type = :Trad,
                   rm = :RCVaR,
                   obj = :Utility, kelly = :Exact,)
    w7 = opt_port!(portfolio; rf = rf, l = l, class = :Classic, type = :Trad,
                   rm = :RCVaR,
                   obj = :Sharpe, kelly = :None,)
    risk7 = calc_risk(portfolio; type = :Trad, rm = :RCVaR, rf = rf)
    ret7 = dot(portfolio.mu, w7.weights)
    w8 = opt_port!(portfolio; rf = rf, l = l, class = :Classic, type = :Trad,
                   rm = :RCVaR,
                   obj = :Sharpe, kelly = :Approx,)
    risk8 = calc_risk(portfolio; type = :Trad, rm = :RCVaR, rf = rf)
    ret8 = dot(portfolio.mu, w8.weights)
    w9 = opt_port!(portfolio; rf = rf, l = l, class = :Classic, type = :Trad,
                   rm = :RCVaR,
                   obj = :Sharpe, kelly = :Exact,)
    risk9 = calc_risk(portfolio; type = :Trad, rm = :RCVaR, rf = rf)
    ret9 = dot(portfolio.mu, w9.weights)
    w10 = opt_port!(portfolio; rf = rf, l = l, class = :Classic, type = :Trad,
                    rm = :RCVaR,
                    obj = :Max_Ret, kelly = :None,)
    w11 = opt_port!(portfolio; rf = rf, l = l, class = :Classic, type = :Trad,
                    rm = :RCVaR,
                    obj = :Max_Ret, kelly = :Approx,)
    w12 = opt_port!(portfolio; rf = rf, l = l, class = :Classic, type = :Trad,
                    rm = :RCVaR,
                    obj = :Max_Ret, kelly = :Exact,)
    setproperty!(portfolio, :mu_l, ret7)
    w13 = opt_port!(portfolio; rf = rf, l = l, class = :Classic, type = :Trad,
                    rm = :RCVaR,
                    obj = :Min_Risk, kelly = :None,)
    setproperty!(portfolio, :mu_l, ret8)
    w14 = opt_port!(portfolio; rf = rf, l = l, class = :Classic, type = :Trad,
                    rm = :RCVaR,
                    obj = :Min_Risk, kelly = :None,)
    setproperty!(portfolio, :mu_l, ret9)
    w15 = opt_port!(portfolio; rf = rf, l = l, class = :Classic, type = :Trad,
                    rm = :RCVaR,
                    obj = :Min_Risk, kelly = :None,)
    setproperty!(portfolio, :mu_l, Inf)
    rmf = Symbol(lowercase(string(:RCVaR)) * "_u")
    setproperty!(portfolio, rmf, risk7)
    w16 = opt_port!(portfolio; rf = rf, l = l, class = :Classic, type = :Trad,
                    rm = :RCVaR,
                    obj = :Max_Ret, kelly = :None,)
    setproperty!(portfolio, rmf, risk8)
    w17 = opt_port!(portfolio; rf = rf, l = l, class = :Classic, type = :Trad,
                    rm = :RCVaR,
                    obj = :Max_Ret, kelly = :None,)
    setproperty!(portfolio, rmf, risk9)
    w18 = opt_port!(portfolio; rf = rf, l = l, class = :Classic, type = :Trad,
                    rm = :RCVaR,
                    obj = :Max_Ret, kelly = :None,)
    setproperty!(portfolio, rmf, risk1 + 1e-2 * risk1)
    w19 = opt_port!(portfolio; rf = rf, l = l, class = :Classic, type = :Trad,
                    rm = :RCVaR,
                    obj = :Sharpe, kelly = :None,)
    setproperty!(portfolio, rmf, Inf)

    w1t = [1.885489815120511e-12, 0.09028989786900095, 2.691337115564653e-12,
           0.014128521705363658, 0.01054553270756888, 0.1233433168149076,
           3.4451101679610427e-13, 0.04829573394884391, 8.208904050352963e-13,
           1.2133205175868203e-12, 0.21255173652142656, 2.3177250729737917e-12,
           2.404071634065065e-12, 0.21099540606427278, 7.315418249553558e-13,
           0.016491659891644805, 2.067664194996877e-11, 0.18907332086976483,
           2.8338456705002445e-12, 0.08428487357128672]

    w2t = [7.029847379563586e-13, 0.09028989748308971, 2.2407972206654733e-12,
           0.014128521537723353, 0.01054553373092779, 0.12334331657175238,
           7.953514196185144e-13, 0.04829573408103305, 1.2121189494696365e-13,
           1.7012521384426009e-12, 0.21255173713103248, 1.2707120219263577e-12,
           2.170467365462205e-12, 0.21099540630254562, 6.258205714546469e-13,
           0.01649166037161498, 1.6210746725772437e-11, 0.18907331972763766,
           2.736018099673632e-12, 0.08428487303406755]

    w3t = [4.0439884009001815e-11, 0.09028990815588868, 4.627064674269077e-11,
           0.01412852812411127, 0.010545504865982989, 0.12334332020074194,
           2.6906358046549096e-11, 0.04829572932784399, 2.7646342909088797e-11,
           3.497567425156123e-11, 0.2125517199941872, 4.542864456976733e-11,
           1.1728487715209849e-11, 0.21099539847154825, 3.1192714589419127e-11,
           0.01649164534700353, 1.4899110586748804e-10, 0.18907335482735677,
           4.3820551497255055e-11, 0.08428489022793507]

    w4t = [1.3042019472879314e-12, 0.0897519650303308, 3.012854872493291e-12,
           0.014161334732600912, 0.011510713787084816, 0.12331586179062769,
           1.3583201147992276e-12, 0.04842259555559938, 1.141375140730255e-12,
           3.4715755994336264e-12, 0.2125908029944935, 1.2171286409783937e-12,
           2.8971542526531803e-12, 0.2114119823958323, 8.121206672593077e-13,
           0.01678480793306278, 2.2306758276053457e-10, 0.18809446169029106,
           6.872443983064102e-12, 0.08395547384492195]

    w5t = [1.1412556329852153e-12, 0.08975194328269624, 2.717161499046862e-12,
           0.014161389193827444, 0.011510810405915037, 0.12331580771188659,
           1.5113814653193585e-12, 0.04842261656903503, 1.6238599237533107e-14,
           2.114107047781725e-12, 0.2125907633892074, 9.494443922714283e-13,
           3.2406759790989303e-12, 0.21141201593053796, 1.0181646820472085e-12,
           0.01678483873213622, 2.1329818950484845e-10, 0.18809436758240414,
           5.010773343341856e-12, 0.0839554469713365]

    w6t = [4.002866048076541e-8, 0.08956174036907162, 5.456496597171497e-8,
           0.01461595513557362, 0.012326195603033616, 0.12284465891853509,
           1.7365043632890118e-8, 0.048599442621491344, 3.6882175978059996e-8,
           5.1078935978603024e-8, 0.2122516405273212, 3.8019737338871854e-8,
           6.6575139552254436e-9, 0.21170525251962352, 2.0238104749958166e-8,
           0.017048555325479928, 1.1404201046791317e-6, 0.18730740839776847,
           7.526405630225357e-8, 0.08373767006280242]

    w7t = [4.2693352519551793e-11, 8.867167773284327e-12,
           4.2700994025679096e-11,
           3.4041741120708815e-11, 0.0877176151449881, 5.035905549140716e-11,
           5.059109343669759e-11, 0.12008424635963133, 9.891198745498483e-10,
           1.0576692025145878e-11, 3.213045171035503e-11,
           4.6546193005104476e-11,
           5.162314268681543e-11, 3.772772657321096e-11, 5.104864427195964e-11,
           0.1512089568494227, 0.52926907626048, 2.5614249764684694e-11,
           0.11172010388038414, 3.145336765831645e-11]

    w8t = [1.3858425728870938e-11, 5.244526978327792e-11,
           1.3582656326233242e-11,
           2.2991761561261977e-11, 0.0871269994618649, 3.2502340244073916e-12,
           4.987861644563195e-12, 0.11919782595349114, 5.537639176958171e-9,
           5.5128702241480615e-11, 3.152318059722877e-11, 9.833773598615321e-12,
           4.0316229073976485e-12, 2.3002573566255762e-11, 4.58143734930654e-12,
           0.15025836355557748, 0.5287598025525965, 3.9891213750471954e-11,
           0.11465700262759881, 3.2123556488563507e-11]

    w9t = [1.985814524659342e-6, 5.977446641435807e-6, 1.974751874279346e-6,
           3.4983394774290887e-6, 0.09939555442120176, 5.849823220175371e-7,
           8.477466902236184e-7, 0.009387123667727471, 0.012664893982154672,
           6.161310956131769e-6, 3.6498768085680014e-6, 1.5117740541627101e-6,
           7.076282937476447e-7, 2.758555876344199e-6, 7.984571242564781e-7,
           0.14357227105283799, 0.5697620116652903, 4.647280566834289e-6,
           0.16517938710141586, 3.654144161831504e-6]

    w10t = [8.735302946157558e-10, 1.4694340605677882e-9, 9.363444121843388e-10,
            2.1876506754441007e-9, 0.999999934079193, 3.1580386249575923e-10,
            4.6313075159652074e-10, 1.335000928516461e-9, 5.6950382785808335e-9,
            1.375619583416357e-9, 7.165610541687079e-10, 5.749923656028609e-10,
            3.666762655574611e-10, 7.018552422000728e-10,
            4.5101400196236804e-10,
            7.0059703434465785e-9, 3.6302592807549995e-8, 9.025128280807362e-10,
            3.4841124221393946e-9, 7.629668820258387e-10]

    w11t = [6.012916296192793e-12, 1.7807646763699447e-11,
            7.440376945637007e-12,
            2.587584387096029e-11, 0.8577737319718369, 5.068259676883461e-12,
            1.8804674354775047e-12, 1.789472629356861e-11,
            9.128597681060186e-11,
            1.819259062171437e-11, 3.8670768776829065e-12,
            2.7111115509351316e-13,
            4.00482212048949e-12, 3.356117138708273e-12, 2.0553290896105107e-12,
            1.8565364640276052e-10, 0.14222626756819667, 8.092204255179985e-12,
            5.6581231155764054e-11, 4.626108511204044e-12]

    w12t = [5.267818042337735e-9, 9.393521844527004e-9, 5.506676096216506e-9,
            1.2359848570706553e-8, 0.8926326010317032, 1.732456915685432e-9,
            2.3557119319779647e-9, 9.111699081942208e-9, 4.096064286031918e-8,
            8.891589226590329e-9, 4.410817054271919e-9, 3.200618733198522e-9,
            1.8314614821228676e-9, 4.214721747128061e-9, 2.262937390475276e-9,
            7.319752039835533e-8, 0.10736717913270283, 5.766275258919584e-9,
            2.4791195044730602e-8, 4.580082262376339e-9]

    w13t = [4.44377627466711e-12, 1.153051112397103e-12, 4.277395814303216e-12,
            3.4524181628424036e-12, 0.08771756335211468, 4.8129649890867725e-12,
            4.74536279957673e-12, 0.1200842185529602, 4.888812411581831e-11,
            1.2527232053427513e-12, 2.9777632848519678e-12,
            4.791997887837139e-12,
            4.6108658628520116e-12, 3.821179693593815e-12,
            4.770865597886005e-12,
            0.15120891877759712, 0.5292691262555218, 3.0378769282551643e-12,
            0.11172017296382873, 9.410024668919459e-13]

    w14t = [2.7496085764195365e-12, 5.518254644820851e-13,
            2.683228370706734e-12,
            2.2536437056243538e-12, 0.08825799527307868, 3.019885556355915e-12,
            2.9891579441813048e-12, 0.12071728737987927, 1.738949508186961e-11,
            7.133340530476957e-13, 1.6616048748672291e-12,
            2.9573741726550625e-12,
            2.9380094661954363e-12, 2.23817801615805e-12,
            3.0084424552985255e-12,
            0.1513731469723375, 0.5287160830386103, 1.6102691596939627e-12,
            0.11093548728874388, 5.863529750908513e-13]

    w15t = [4.904488573750729e-12, 2.071106728993754e-12, 4.800133024741897e-12,
            3.915793695045729e-12, 0.09178330614142587, 4.723084691031301e-12,
            4.837403928921659e-12, 0.0009767817347373915, 6.496610443683043e-11,
            2.3988177382041532e-12, 3.893853552308335e-12,
            5.039285982466147e-12,
            4.729786847833998e-12, 4.45452517438696e-12, 4.8980359771502755e-12,
            0.14141381903028388, 0.5743520466758162, 3.98631737830706e-12,
            0.19147404629527873, 2.839376537835085e-12]

    w16t = [2.5331499710763255e-13, 1.942139684520031e-12,
            2.797498374891446e-13,
            7.10310260273347e-14, 0.08771758250749578, 6.882339655405197e-13,
            6.201429363256583e-13, 0.12008424696163275, 1.6361505841694846e-11,
            1.6057558309462293e-12, 5.658350518833768e-10,
            4.055982305386779e-13,
            6.614076506701964e-13, 4.240025021643277e-12, 6.462429177509855e-13,
            0.15120892456303342, 0.5292691017319702, 1.4024604250170038e-10,
            0.11172014345931404, 4.2697820209948237e-11]

    w17t = [1.6216819060655524e-13, 3.602736795125626e-12,
            2.002307574155093e-13,
            7.66888202319162e-14, 0.0881713453630755, 5.181892735823799e-13,
            4.796904823523065e-13, 0.1206158001292373, 8.808315999173303e-12,
            2.284474458865395e-12, 2.9537581692501615e-9, 2.811308300914961e-13,
            5.282854348160106e-13, 2.1270364701539336e-11,
            5.105401816811959e-13,
            0.15134681025157934, 0.5288047384122242, 7.567913219398542e-10,
            0.1110613018671969, 2.2741444375492125e-10]

    w18t = [6.017964139448665e-13, 4.162833539796467e-12, 6.255737396544022e-13,
            3.6934545672165233e-13, 0.09161103798114185, 1.5396470048295767e-12,
            1.376237525199013e-12, 4.413917662714404e-8, 0.0016588467476908635,
            2.3365692904110845e-12, 8.101137999978144e-10, 8.55247252035555e-13,
            1.5006222484359331e-12, 2.524735133715067e-11,
            1.4530377927239903e-12,
            0.1414989808108703, 0.5755694299610409, 5.909926376158509e-10,
            0.189661658609003, 3.099016152969422e-10]

    w19t = [1.6110206476361473e-13, 0.06360841984194797, 1.0678594859382793e-12,
            0.015485826016009672, 0.059166685383026926, 0.0421889436710687,
            4.4297172420271715e-12, 0.06565785421138116, 1.274079507781254e-12,
            8.653414596076305e-13, 0.21371386666915232, 8.606033133303287e-13,
            6.75941338405136e-12, 0.20637251783502342, 4.746594002690692e-12,
            0.05906555903416136, 0.04638502538108881, 0.14881895925409996,
            9.237186691754156e-12, 0.07953634267363793]

    @test isapprox(w1.weights, w1t)
    @test isapprox(w2.weights, w2t)
    @test isapprox(w3.weights, w3t)
    @test isapprox(w2.weights, w3.weights, rtol = 1e-6)
    @test isapprox(w4.weights, w4t)
    @test isapprox(w5.weights, w5t)
    @test isapprox(w6.weights, w6t, rtol = 0.01)
    @test isapprox(w5.weights, w6.weights, rtol = 1e-2)
    @test isapprox(w7.weights, w7t)
    @test isapprox(w8.weights, w8t)
    @test isapprox(w9.weights, w9t, rtol = 0.0001)
    @test isapprox(w8.weights, w9.weights, rtol = 1e0)
    @test isapprox(w10.weights, w10t)
    @test isapprox(w11.weights, w11t)
    @test isapprox(w12.weights, w12t, rtol = 1.0e-6)
    @test isapprox(w11.weights, w12.weights, rtol = 1e-1)
    @test isapprox(w13.weights, w7.weights, rtol = 1e-6)
    @test isapprox(w14.weights, w8.weights, rtol = 1e-2)
    @test isapprox(w15.weights, w9.weights, rtol = 1e-1)
    @test isapprox(w16.weights, w7.weights, rtol = 1e-6)
    @test isapprox(w17.weights, w8.weights, rtol = 1e-2)
    @test isapprox(w18.weights, w9.weights, rtol = 1e-1)
    @test isapprox(w13.weights, w16.weights, rtol = 1e-7)
    @test isapprox(w14.weights, w17.weights, rtol = 1e-3)
    @test isapprox(w15.weights, w18.weights, rtol = 1e-2)
    @test isapprox(w19.weights, w1.weights, rtol = 1e0)
end

@testset "$(:Classic), $(:Trad), $(:TG)" begin
    portfolio = Portfolio(; prices = prices[(end - 200):end],
                          solvers = OrderedDict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                                  :params => Dict("verbose" => false,
                                                                                  "max_step_fraction" => 0.75)),
                                                :COSMO => Dict(:solver => COSMO.Optimizer,
                                                               :params => Dict("verbose" => false))))
    asset_statistics!(portfolio)

    w1 = opt_port!(portfolio; rf = rf, l = l, class = :Classic, type = :Trad,
                   rm = :TG,
                   obj = :Min_Risk, kelly = :None,)
    risk1 = calc_risk(portfolio; type = :Trad, rm = :TG, rf = rf)
    w2 = opt_port!(portfolio; rf = rf, l = l, class = :Classic, type = :Trad,
                   rm = :TG,
                   obj = :Min_Risk, kelly = :Approx,)
    w3 = opt_port!(portfolio; rf = rf, l = l, class = :Classic, type = :Trad,
                   rm = :TG,
                   obj = :Min_Risk, kelly = :Exact,)
    w4 = opt_port!(portfolio; rf = rf, l = l, class = :Classic, type = :Trad,
                   rm = :TG,
                   obj = :Utility, kelly = :None,)
    w5 = opt_port!(portfolio; rf = rf, l = l, class = :Classic, type = :Trad,
                   rm = :TG,
                   obj = :Utility, kelly = :Approx,)
    w6 = opt_port!(portfolio; rf = rf, l = l, class = :Classic, type = :Trad,
                   rm = :TG,
                   obj = :Utility, kelly = :Exact,)
    w7 = opt_port!(portfolio; rf = rf, l = l, class = :Classic, type = :Trad,
                   rm = :TG,
                   obj = :Sharpe, kelly = :None,)
    risk7 = calc_risk(portfolio; type = :Trad, rm = :TG, rf = rf)
    ret7 = dot(portfolio.mu, w7.weights)
    w8 = opt_port!(portfolio; rf = rf, l = l, class = :Classic, type = :Trad,
                   rm = :TG,
                   obj = :Sharpe, kelly = :Approx,)
    risk8 = calc_risk(portfolio; type = :Trad, rm = :TG, rf = rf)
    ret8 = dot(portfolio.mu, w8.weights)
    w9 = opt_port!(portfolio; rf = rf, l = l, class = :Classic, type = :Trad,
                   rm = :TG,
                   obj = :Sharpe, kelly = :Exact,)
    risk9 = calc_risk(portfolio; type = :Trad, rm = :TG, rf = rf)
    ret9 = dot(portfolio.mu, w9.weights)
    w10 = opt_port!(portfolio; rf = rf, l = l, class = :Classic, type = :Trad,
                    rm = :TG,
                    obj = :Max_Ret, kelly = :None,)
    w11 = opt_port!(portfolio; rf = rf, l = l, class = :Classic, type = :Trad,
                    rm = :TG,
                    obj = :Max_Ret, kelly = :Approx,)
    w12 = opt_port!(portfolio; rf = rf, l = l, class = :Classic, type = :Trad,
                    rm = :TG,
                    obj = :Max_Ret, kelly = :Exact,)
    setproperty!(portfolio, :mu_l, ret7)
    w13 = opt_port!(portfolio; rf = rf, l = l, class = :Classic, type = :Trad,
                    rm = :TG,
                    obj = :Min_Risk, kelly = :None,)
    setproperty!(portfolio, :mu_l, ret8)
    w14 = opt_port!(portfolio; rf = rf, l = l, class = :Classic, type = :Trad,
                    rm = :TG,
                    obj = :Min_Risk, kelly = :None,)
    setproperty!(portfolio, :mu_l, ret9)
    w15 = opt_port!(portfolio; rf = rf, l = l, class = :Classic, type = :Trad,
                    rm = :TG,
                    obj = :Min_Risk, kelly = :None,)
    setproperty!(portfolio, :mu_l, Inf)
    rmf = Symbol(lowercase(string(:TG)) * "_u")
    setproperty!(portfolio, rmf, risk7)
    w16 = opt_port!(portfolio; rf = rf, l = l, class = :Classic, type = :Trad,
                    rm = :TG,
                    obj = :Max_Ret, kelly = :None,)
    setproperty!(portfolio, rmf, risk8)
    w17 = opt_port!(portfolio; rf = rf, l = l, class = :Classic, type = :Trad,
                    rm = :TG,
                    obj = :Max_Ret, kelly = :None,)
    setproperty!(portfolio, rmf, risk9)
    w18 = opt_port!(portfolio; rf = rf, l = l, class = :Classic, type = :Trad,
                    rm = :TG,
                    obj = :Max_Ret, kelly = :None,)
    setproperty!(portfolio, rmf, risk1)
    w19 = opt_port!(portfolio; rf = rf, l = l, class = :Classic, type = :Trad,
                    rm = :TG,
                    obj = :Sharpe, kelly = :None,)
    setproperty!(portfolio, rmf, Inf)

    w1t = [5.969325155308689e-13, 0.19890625725641747, 5.01196330972591e-12,
           0.05490289350205719, 4.555898520654742e-11, 1.0828830360388467e-12,
           3.3254529493732914e-12, 0.10667398877393132, 1.0523685498886423e-12,
           1.8363867378603495e-11, 0.0486780391076569, 0.09089397416963123,
           3.698655012660736e-12, 8.901903375667794e-12, 2.0443054631623374e-12,
           0.13418199806574263, 6.531476985844397e-12, 0.13829581359726595,
           1.724738947709949e-12, 0.22746703542940375]

    w2t = [7.83309142057733e-13, 0.19890625833071449, 4.831870290348725e-12,
           0.05490289219899938, 4.374760714538073e-11, 1.2774333103612622e-12,
           3.535077619663961e-12, 0.10667398931900413, 1.3121776240169205e-12,
           1.8553793304558087e-11, 0.04867803758995198, 0.09089397446685778,
           3.919625007947312e-12, 9.092499319944059e-12, 1.7407386183632267e-12,
           0.13418199857050314, 6.510498478609928e-12, 0.13829581413563036,
           1.5426320356269584e-12, 0.22746703529149154]

    w3t = [3.5570068819038745e-11, 0.19890649268403118, 8.336556035166604e-11,
           0.05490300615115322, 4.4255254018763954e-10, 3.1388042099786696e-11,
           1.1939283146646753e-11, 0.10667398523145416, 3.19679816057631e-11,
           2.003425066287439e-10, 0.048677800386075394, 0.09089395604491374,
           8.85943982524158e-12, 1.1676830497110102e-10, 5.6678363746223674e-11,
           0.13418187266618783, 1.056082969456107e-10, 0.13829579898429772,
           5.7277639547135284e-11, 0.2274670866695688]

    w4t = [2.3531676655820754e-12, 0.22146555850379093, 9.410028484870327e-12,
           0.05148290829152357, 8.0605983841845e-10, 4.9113218142397595e-12,
           8.131364518006022e-12, 0.11536086211760316, 2.271272286048353e-12,
           5.43251102138864e-11, 7.650522053564507e-10, 0.09513372436124735,
           8.884714671260909e-12, 1.3757061851213529e-11,
           1.6291810113089304e-13,
           0.1432756744154531, 2.0775280711374397e-11, 0.14179358412385565,
           4.251506235430576e-12, 0.2314876864861804]

    w5t = [7.876741054601859e-13, 0.22386204520499708, 3.551136120852785e-12,
           0.062473602045942406, 2.681412014274187e-10, 1.6398169690036202e-12,
           2.8836268962671105e-12, 0.11120554463921713, 9.69896654536334e-13,
           2.037905902633807e-11, 3.525244052955003e-10, 0.09247718814867852,
           3.1574898462203464e-12, 5.739501373413488e-12,
           1.0045259045269914e-13,
           0.1367242024665253, 6.543734045634463e-12, 0.13899887173200867,
           1.5190248956445046e-12, 0.23425854509469402]

    w6t = [1.2941247551525297e-11, 0.22386925954250705, 3.2824653702191277e-11,
           0.0625065786029998, 1.3620372143823865e-9, 8.717706191359317e-12,
           3.1852031182937095e-12, 0.11119308337310815, 1.2968996068172452e-11,
           1.0934304821426853e-10, 1.2800883116924593e-9, 0.09246921064626823,
           1.969493077754324e-12, 4.049153686277518e-11, 1.65539530029089e-11,
           0.13670452814172337, 5.3419099417948596e-11, 0.138990478667524,
           2.4600231942469032e-11, 0.23426685806672873]

    w7t = [1.034203540495142e-11, 1.4007889383681966e-11,
           1.0025731760906318e-11,
           6.108387515365813e-12, 0.3572352904863753, 1.2069497838089613e-11,
           1.2830319739722155e-11, 1.1788800032033873e-12,
           6.535335074248242e-12,
           2.8948639761902753e-12, 9.118481022316377e-12,
           1.0966191842706305e-11,
           1.3147750680814202e-11, 9.61174753780057e-12, 1.2632463845017768e-11,
           0.21190800224252684, 0.43085670711282753, 7.711974266816198e-12,
           1.2413946392925463e-11, 6.674972177548544e-12]

    w8t = [1.4718118126007435e-11, 1.2474877074375374e-10,
           1.580793386862312e-11,
           3.2618621177143654e-11, 0.32203999245931664, 3.645398056694916e-12,
           5.265099829843491e-12, 8.660993875158394e-11, 2.1492657816370373e-10,
           4.569980763473106e-11, 2.379471084814323e-11, 1.3355963768881334e-11,
           4.8340543797950154e-12, 2.042924205474397e-11, 6.038969225874817e-12,
           0.20669799685131962, 0.4712620097671732, 2.8437238842777484e-11,
           2.4390014647907217e-10, 3.735984832594157e-11]

    w9t = [9.049607283737652e-8, 5.431276130993444e-7, 9.670954936738661e-8,
           1.9238834045719207e-7, 0.32906337290031373, 2.7962734129007014e-8,
           3.717570143464828e-8, 4.1921441814268555e-7, 6.609279521185226e-7,
           2.523123468483196e-7, 1.3533584694438362e-7, 8.133860462210069e-8,
           3.460134867768275e-8, 1.1892230983700056e-7, 4.150758024032568e-8,
           0.20763789936753316, 0.4632947293695633, 1.6231243261885348e-7,
           9.07663022787599e-7, 1.9636671559329504e-7]

    w10t = [8.730661626481806e-10, 1.4687309386680775e-9, 9.358654898947756e-10,
            2.186677678966605e-9, 0.9999999341280966, 3.156239683565371e-10,
            4.629682444037334e-10, 1.3343331058979577e-9, 5.69212053674475e-9,
            1.3749327860458163e-9, 7.161024888414711e-10, 5.748069307464309e-10,
            3.666660703394924e-10, 7.0140810552241e-10, 4.5086981398858524e-10,
            7.002103638459066e-9, 3.6268802221190136e-8, 9.019721807104911e-10,
            3.482364214437503e-9, 7.624890097546668e-10]

    w11t = [5.98855212549632e-12, 1.7739824023891267e-11, 7.410312574194724e-12,
            2.5779997088827038e-11, 0.857773731868385, 5.054838680220514e-12,
            1.8752566177088636e-12, 1.7825960548207287e-11,
            9.096345201998647e-11,
            1.8124063387139953e-11, 3.8501508467546374e-12,
            2.674046232448817e-13,
            3.994147030923467e-12, 3.3407504614825583e-12,
            2.047533640542789e-12,
            1.8501728724970428e-10, 0.14222626767329194, 8.059325025309758e-12,
            5.637848418414877e-11, 4.6056332303540835e-12]

    w12t = [5.2804930141649576e-9, 9.416211338192126e-9, 5.520001335041304e-9,
            1.2389747433355727e-8, 0.8926328276594367, 1.7366619884827056e-9,
            2.36142330144021e-9, 9.133777915513395e-9, 4.105626060606478e-8,
            8.913137607797025e-9, 4.421552283984949e-9, 3.208536147187476e-9,
            1.8359894346800377e-9, 4.224879347145791e-9, 2.2684200090728395e-9,
            7.336516777793194e-8, 0.10736695198679332, 5.780217003330638e-9,
            2.4850102158041525e-8, 4.591191059247631e-9]

    w13t = [3.4396140094111085e-12, 5.702331297877075e-12,
            3.3850739092860734e-12,
            2.1449219455862455e-12, 0.3572352911645771, 4.158207199076786e-12,
            4.030094976593963e-12, 1.6243295230400244e-12,
            2.463859780243482e-12,
            1.0962514895832545e-12, 2.7142265167093685e-12,
            3.2469604150886593e-12,
            4.124825680360781e-12, 2.672732115046088e-12, 3.929789757857018e-12,
            0.21190800498277654, 0.43085670379690205, 2.433400486793345e-12,
            7.374176308416663e-12, 1.2035320830453566e-12]

    w14t = [3.257745229963087e-12, 1.0130827408116335e-8, 3.192673279104055e-12,
            2.036266055028359e-12, 0.32204005948679654, 3.999534104117489e-12,
            3.922912154529824e-12, 8.565478472093807e-12, 3.382005532265066e-12,
            1.6154896835195683e-13, 1.8633120777140204e-12,
            2.9655484946338515e-12,
            3.931677496001916e-12, 2.173876526701325e-12, 3.813058190024838e-12,
            0.2066980124008367, 0.47126191792157196, 1.6331021369502729e-12,
            1.339801780572528e-11, 1.670664150519693e-12]

    w15t = [2.7465751160880876e-12, 2.1177644537355916e-9,
            2.6919131619233422e-12,
            1.7754153207760756e-12, 0.32919375360177344, 3.3653950108839433e-12,
            3.2914905879479745e-12, 5.5100400491825204e-12,
            2.2748945420127287e-12,
            3.8881028270373756e-13, 1.6959554376446352e-12,
            2.550772718640682e-12,
            3.3032659915241093e-12, 1.945487563770423e-12,
            3.2085635894749945e-12,
            0.20775698021279215, 0.4630492640217384, 1.4889310873384205e-12,
            8.998414125961183e-12, 6.955715851970014e-13]

    w16t = [9.979862067124343e-15, 4.203228929046188e-11, 8.980440460800805e-15,
            6.446959689097406e-15, 0.3572352914112592, 2.0781543002891548e-14,
            1.7246066770888985e-14, 8.153114867404073e-13,
            5.611409506218815e-14,
            1.7847952880714525e-14, 4.2082544141755004e-15,
            5.40968849423465e-15,
            1.9828797974959275e-14, 4.947950070792405e-15,
            1.7773371270726178e-14,
            0.21190800509326468, 0.4308567034510944, 1.0447138423251913e-14,
            1.0125147915616658e-13, 1.232779785619679e-12]

    w17t = [9.310433196319792e-13, 6.519217348949254e-8, 3.7382112176040974e-13,
            5.9708001503694214e-12, 0.3220400916566543, 2.4038594993022703e-12,
            2.5347579086540625e-12, 4.5441541601200795e-9,
            5.312619251171737e-12,
            9.141386937821757e-12, 2.5774849468146276e-11,
            2.0184955678972354e-11,
            2.583332491176395e-12, 3.809356677370173e-12, 1.824190275114758e-12,
            0.20669801970721083, 0.47126181304128306, 2.932901748617543e-11,
            1.6116519846779716e-11, 5.732233621279043e-9]

    w18t = [5.22559179306702e-12, 1.949910286726206e-7, 3.3288860762198112e-12,
            1.891245018197559e-11, 0.3293096665369873, 1.2616124768857992e-11,
            1.2641987148578779e-11, 1.089540208970257e-8,
            3.6235616774174153e-11,
            3.447243519612428e-11, 7.278589141121017e-11,
            5.3221882905407935e-11,
            1.2736246057904183e-11, 1.1554060860433626e-11,
            1.0265589268338373e-11,
            0.2077741546167127, 0.46291595767439997, 8.501954099643547e-11,
            9.753022929512629e-11, 1.4818922832047022e-8]

    w19t = [7.55657374634104e-12, 0.19891044921795567, 2.491805468101704e-11,
            0.054902604651253704, 1.5885531756094824e-9, 1.1863426477811066e-11,
            2.186647458232334e-11, 0.10667546168194031, 1.054290697369371e-11,
            1.326199568511875e-10, 0.04866926165556854, 0.09089466315035939,
            2.3725316835207252e-11, 4.042940031217252e-11,
            1.3879617478706681e-12,
            0.13418338454436574, 2.2390768113754715e-11, 0.1382963370480211,
            3.4856870773993155e-12, 0.22746783616119595]

    @test isapprox(w1.weights, w1t)
    @test isapprox(w2.weights, w2t)
    @test isapprox(w3.weights, w3t)
    @test isapprox(w2.weights, w3.weights, rtol = 1e-6)
    @test isapprox(w4.weights, w4t)
    @test isapprox(w5.weights, w5t)
    @test isapprox(w6.weights, w6t, rtol = 1.0e-6)
    @test isapprox(w5.weights, w6.weights, rtol = 1e-3)
    @test isapprox(w7.weights, w7t)
    @test isapprox(w8.weights, w8t)
    @test isapprox(w9.weights, w9t, rtol = 0.01)
    @test isapprox(w8.weights, w9.weights, rtol = 1e-1)
    @test isapprox(w10.weights, w10t)
    @test isapprox(w11.weights, w11t)
    @test isapprox(w12.weights, w12t, rtol = 1.0e-5)
    @test isapprox(w11.weights, w12.weights, rtol = 1e-1)
    @test isapprox(w13.weights, w7.weights, rtol = 1e-8)
    @test isapprox(w14.weights, w8.weights, rtol = 1e-6)
    @test isapprox(w15.weights, w9.weights, rtol = 0.01)
    @test isapprox(w16.weights, w7.weights, rtol = 1.0e-5)
    @test isapprox(w17.weights, w8.weights, rtol = 1e-6)
    @test isapprox(w18.weights, w9.weights, rtol = 0.01)
    @test isapprox(w13.weights, w16.weights, rtol = 1.0e-5)
    @test isapprox(w14.weights, w17.weights, rtol = 1e-6)
    @test isapprox(w15.weights, w18.weights, rtol = 1e-3)
    @test isapprox(w19.weights, w1.weights, rtol = 1e-4)
end

@testset "$(:Classic), $(:Trad), $(:RTG)" begin
    portfolio = Portfolio(; prices = prices[(end - 200):end],
                          solvers = OrderedDict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                                  :params => Dict("verbose" => false,
                                                                                  "max_step_fraction" => 0.75)),
                                                :COSMO => Dict(:solver => COSMO.Optimizer,
                                                               :params => Dict("verbose" => false))))
    asset_statistics!(portfolio)

    w1 = opt_port!(portfolio; rf = rf, l = l, class = :Classic, type = :Trad,
                   rm = :RTG,
                   obj = :Min_Risk, kelly = :None,)
    risk1 = calc_risk(portfolio; type = :Trad, rm = :RTG, rf = rf)
    w2 = opt_port!(portfolio; rf = rf, l = l, class = :Classic, type = :Trad,
                   rm = :RTG,
                   obj = :Min_Risk, kelly = :Approx,)
    w3 = opt_port!(portfolio; rf = rf, l = l, class = :Classic, type = :Trad,
                   rm = :RTG,
                   obj = :Min_Risk, kelly = :Exact,)
    w4 = opt_port!(portfolio; rf = rf, l = l, class = :Classic, type = :Trad,
                   rm = :RTG,
                   obj = :Utility, kelly = :None,)
    w5 = opt_port!(portfolio; rf = rf, l = l, class = :Classic, type = :Trad,
                   rm = :RTG,
                   obj = :Utility, kelly = :Approx,)
    w6 = opt_port!(portfolio; rf = rf, l = l, class = :Classic, type = :Trad,
                   rm = :RTG,
                   obj = :Utility, kelly = :Exact,)
    w7 = opt_port!(portfolio; rf = rf, l = l, class = :Classic, type = :Trad,
                   rm = :RTG,
                   obj = :Sharpe, kelly = :None,)
    risk7 = calc_risk(portfolio; type = :Trad, rm = :RTG, rf = rf)
    ret7 = dot(portfolio.mu, w7.weights)
    w8 = opt_port!(portfolio; rf = rf, l = l, class = :Classic, type = :Trad,
                   rm = :RTG,
                   obj = :Sharpe, kelly = :Approx,)
    risk8 = calc_risk(portfolio; type = :Trad, rm = :RTG, rf = rf)
    ret8 = dot(portfolio.mu, w8.weights)
    w9 = opt_port!(portfolio; rf = rf, l = l, class = :Classic, type = :Trad,
                   rm = :RTG,
                   obj = :Sharpe, kelly = :Exact,)
    risk9 = calc_risk(portfolio; type = :Trad, rm = :RTG, rf = rf)
    ret9 = dot(portfolio.mu, w9.weights)
    w10 = opt_port!(portfolio; rf = rf, l = l, class = :Classic, type = :Trad,
                    rm = :RTG,
                    obj = :Max_Ret, kelly = :None,)
    w11 = opt_port!(portfolio; rf = rf, l = l, class = :Classic, type = :Trad,
                    rm = :RTG,
                    obj = :Max_Ret, kelly = :Approx,)
    w12 = opt_port!(portfolio; rf = rf, l = l, class = :Classic, type = :Trad,
                    rm = :RTG,
                    obj = :Max_Ret, kelly = :Exact,)
    setproperty!(portfolio, :mu_l, ret7)
    w13 = opt_port!(portfolio; rf = rf, l = l, class = :Classic, type = :Trad,
                    rm = :RTG,
                    obj = :Min_Risk, kelly = :None,)
    setproperty!(portfolio, :mu_l, ret8)
    w14 = opt_port!(portfolio; rf = rf, l = l, class = :Classic, type = :Trad,
                    rm = :RTG,
                    obj = :Min_Risk, kelly = :None,)
    setproperty!(portfolio, :mu_l, ret9)
    w15 = opt_port!(portfolio; rf = rf, l = l, class = :Classic, type = :Trad,
                    rm = :RTG,
                    obj = :Min_Risk, kelly = :None,)
    setproperty!(portfolio, :mu_l, Inf)
    rmf = Symbol(lowercase(string(:RTG)) * "_u")
    setproperty!(portfolio, rmf, risk7)
    w16 = opt_port!(portfolio; rf = rf, l = l, class = :Classic, type = :Trad,
                    rm = :RTG,
                    obj = :Max_Ret, kelly = :None,)
    setproperty!(portfolio, rmf, risk8)
    w17 = opt_port!(portfolio; rf = rf, l = l, class = :Classic, type = :Trad,
                    rm = :RTG,
                    obj = :Max_Ret, kelly = :None,)
    setproperty!(portfolio, rmf, risk9)
    w18 = opt_port!(portfolio; rf = rf, l = l, class = :Classic, type = :Trad,
                    rm = :RTG,
                    obj = :Max_Ret, kelly = :None,)
    setproperty!(portfolio, rmf, risk1)
    w19 = opt_port!(portfolio; rf = rf, l = l, class = :Classic, type = :Trad,
                    rm = :RTG,
                    obj = :Sharpe, kelly = :None,)
    setproperty!(portfolio, rmf, Inf)

    w1t = [1.1206142630614112e-11, 0.09772215128420475, 3.5338858649323828e-12,
           3.307433371895369e-11, 0.017417781863929577, 0.041201102000678634,
           1.6220089767187403e-11, 0.060553118743468366, 1.7795758713115547e-11,
           1.5483766349422575e-11, 0.22830001756787208, 2.6382663454855723e-11,
           3.59484391773399e-11, 0.13930594414393907, 2.0476463882653826e-11,
           0.0362260148933746, 9.71096463830921e-11, 0.24663479935480356,
           1.4791298973742443e-12, 0.13263906986901897]

    w2t = [3.133895413047744e-11, 0.09772167604116666, 1.2455268180095731e-11,
           7.840719879567121e-11, 0.017418117169965327, 0.041201275182607314,
           4.3715932416052955e-11, 0.060553238510597074, 4.6981827189762015e-11,
           4.012322407176324e-11, 0.22830027271191833, 5.893318699116201e-11,
           9.091918390609722e-11, 0.1393059827415553, 5.385714087496309e-11,
           0.036226381315616035, 2.2932266834945375e-10, 0.24663403746828325,
           1.0356157915867175e-13, 0.13263901817213244]

    w3t = [1.4587347004162206e-11, 0.09772230568561657, 1.8550133212740332e-11,
           3.915311549856396e-11, 0.01741768366195373, 0.04120103554990599,
           1.178197456081433e-11, 0.06055308145258582, 1.0909759733378663e-11,
           1.2480862380469432e-11, 0.22829993453436156, 3.517050936225545e-11,
           1.099023841251717e-12, 0.13930593379386513, 9.51967882379049e-12,
           0.036225908143233815, 7.234022717397937e-11, 0.24663503026515377,
           2.1424797938930822e-11, 0.1326390866663063]

    w4t = [7.026805478010537e-12, 0.07691042016208616, 1.4421489729308237e-12,
           3.605941949944227e-12, 0.018736328513127485, 0.02727677352489846,
           7.406419741602451e-12, 0.06300039975110802, 9.651040072665584e-12,
           7.87028804146206e-12, 0.23138910163058715, 1.8363064032313167e-12,
           1.772932503046444e-11, 0.12343688817492235, 1.1346887350045238e-11,
           0.04767277729217605, 9.551667214175191e-11, 0.2743410772339696,
           6.373792606352843e-13, 0.1372362335530555]

    w5t = [3.4706517214133838e-12, 0.0769105134835127, 4.332238200934135e-13,
           2.0798243234787587e-12, 0.01873630615765449, 0.02727674226360157,
           3.662553846864922e-12, 0.06300037490664813, 4.921659104661623e-12,
           3.948558209931358e-12, 0.23138904026457519, 1.4125611823881814e-12,
           9.32600479211128e-12, 0.12343690440083245, 5.795540631821292e-12,
           0.04767271820335058, 5.279314116397832e-11, 0.27434116640204886,
           8.883155341591128e-15, 0.13723623382992328]

    w6t = [1.1582555456119735e-11, 0.07691048840193924, 1.8165976834316422e-11,
           2.4036634425084372e-11, 0.018736314535723074, 0.027276734997086285,
           1.1164739905107386e-11, 0.06300037717964192, 8.533819483771081e-12,
           1.0639516305899386e-11, 0.23138904745439162, 2.1844733214823184e-11,
           1.0274078493009247e-12, 0.12343689433900468, 6.561907834635063e-12,
           0.047672729286724246, 1.3270678366594373e-10, 0.2743411792778638,
           1.9047327419495057e-11, 0.13723623426231363]

    w7t = [4.865125156825178e-11, 2.1052614079822636e-11,
           4.7365907624153683e-11,
           4.001712405491339e-11, 0.1416419921188952, 5.572004917204688e-11,
           5.631795517134003e-11, 0.024937870196849492, 1.79971814557232e-10,
           2.6621313364886448e-11, 4.154443093866612e-11,
           5.2544888997749557e-11,
           5.7811722520210246e-11, 4.517027037812361e-11, 5.695329181673672e-11,
           0.21453454572858413, 0.6188855836123038, 3.483330484227522e-11,
           7.540474114074475e-9, 3.8317437143151725e-11]

    w8t = [2.7581567308792713e-11, 9.305269420569393e-11,
           2.8856463333280447e-11,
           4.9209914529880345e-11, 0.11879417202822751, 6.49213529026326e-12,
           1.0558365268373762e-11, 0.11578411598281094, 8.11697335377414e-10,
           7.623263804780261e-11, 5.5430873896552634e-11, 2.083692541531304e-11,
           8.228898768789633e-12, 4.304777213202721e-11, 9.268065127424773e-12,
           0.18392223680167313, 0.5660084495482668, 7.095328732116712e-11,
           0.01549102426416752, 6.340719531059086e-11]

    w9t = [4.452192067125962e-7, 1.287606811636711e-6, 4.7070532250350305e-7,
           7.836131334480091e-7, 0.13083957327844015, 1.3272190447713796e-7,
           1.95023880053568e-7, 0.046757382144710716, 1.2607203758700611e-5,
           1.1565419811781638e-6, 7.726780248137755e-7, 3.3853707145607237e-7,
           1.5622419866868718e-7, 6.248082052347081e-7, 1.7863550934265617e-7,
           0.1988979404666623, 0.6074976195595244, 9.865617645770584e-7,
           0.01598646464700567, 8.838228839058677e-7]

    w10t = [8.742294125473961e-10, 1.4707486744372402e-9, 9.371030740356124e-10,
            2.189422648260022e-9, 0.9999999340817113, 3.1597937318578487e-10,
            4.6333260236071825e-10, 1.336207603310839e-9, 5.696795440054946e-9,
            1.3768680167520074e-9, 7.171481417281956e-10, 5.752604171160508e-10,
            3.667148376429282e-10, 7.02423046380329e-10, 4.511960450499541e-10,
            7.007335179502708e-9, 3.628430937215484e-8, 9.033006700630679e-10,
            3.486313790911996e-9, 7.636002597011294e-10]

    w11t = [6.034984815892211e-12, 1.7868956473930124e-11,
            7.467599094385243e-12,
            2.596243132845663e-11, 0.8577737320666471, 5.080228610064883e-12,
            1.8850929459354704e-12, 1.795690659535625e-11,
            9.157707978562806e-11,
            1.825453605253934e-11, 3.8824404267478314e-12,
            2.7453150914440054e-13,
            4.014309200232137e-12, 3.3700780411065767e-12,
            2.0623302151245345e-12,
            1.862276586845158e-10, 0.1422262674719025, 8.121980053374418e-12,
            5.6764282879759227e-11, 4.6446850307836085e-12]

    w12t = [5.2646377055566815e-9, 9.387752688951843e-9, 5.503480930338795e-9,
            1.235216674736526e-8, 0.8926328836450478, 1.7312474138778515e-9,
            2.354226867903479e-9, 9.106311409818638e-9, 4.0935783548154107e-8,
            8.886647767038408e-9, 4.408090320821839e-9, 3.198559055116633e-9,
            1.8302608070763718e-9, 4.2122112930226e-9, 2.2615788647672336e-9,
            7.316435095402851e-8, 0.1073668966411457, 5.7628136176508805e-9,
            2.4776389791662633e-8, 4.577296653036629e-9]

    w13t = [2.595599936537348e-11, 1.651019365128729e-11,
            2.5534866643685744e-11,
            2.2331322397629907e-11, 0.1416419980914485, 2.8547848491104438e-11,
            2.7956901030692e-11, 0.024937875012701664, 5.018205148061643e-11,
            1.8276516401964547e-11, 2.2228415745472723e-11,
            2.6483256941600314e-11,
            2.852286792224843e-11, 2.2890050038139587e-11,
            2.8047760729549417e-11,
            0.21453455083719497, 0.6188855733657549, 2.058371397242547e-11,
            2.3093663502351488e-9, 1.9481925188446284e-11]

    w14t = [9.511453385748187e-12, 5.158782452595663e-12, 9.355679577321975e-12,
            8.22983530798536e-12, 0.12110207620316887, 1.0596327432720495e-11,
            1.0374452970729877e-11, 0.12035458528085836, 1.3443395894894593e-11,
            6.173547869257024e-12, 7.423348583160891e-12, 9.811854273805244e-12,
            1.055058152032995e-11, 8.121094107257048e-12,
            1.0423920490319705e-11,
            0.19438514114756852, 0.5641581939077709, 6.83787341129961e-12,
            3.3283477786307378e-9, 6.273481559993505e-12]

    w15t = [7.955590598945476e-12, 4.741151065966018e-12, 7.806242374200578e-12,
            6.7811623433498785e-12, 0.12905162388524555, 8.814266159476679e-12,
            8.60688199371375e-12, 0.05087945538764261, 1.2940951279389932e-11,
            5.360477650229075e-12, 6.6585498610750066e-12,
            8.127137297753299e-12,
            8.80723832819855e-12, 6.888195928846777e-12, 8.630917689837923e-12,
            0.20914744600224697, 0.6109214742545982, 6.143790071406996e-12,
            3.563450885739955e-10, 5.658901090638695e-12]

    w16t = [2.8587696467818527e-13, 1.2467345240121674e-11,
            2.2151480175468828e-13,
            2.9286845680027484e-13, 0.1416419802719567, 9.221202005185037e-13,
            8.149711440468206e-13, 0.024937901687576456, 1.078452322684367e-11,
            1.1220924972901308e-12, 3.174456033772842e-10,
            4.819035187133998e-13,
            9.187561246840649e-13, 1.0508896919184018e-10,
            8.691685150417825e-13,
            0.21453454552082565, 0.6188855710931238, 4.7710933265881826e-11,
            3.349885945516861e-10, 5.921021367708042e-10]

    w17t = [5.022809091179894e-12, 3.278883980116286e-10, 4.242900142157342e-12,
            5.228105269875839e-12, 0.12109989810844203, 1.850103245573456e-11,
            1.652832955852485e-11, 0.12029753058162629, 1.549175844600153e-10,
            2.6176720920853117e-11, 1.5556586266721366e-8,
            8.796490585995089e-12,
            1.8675975278631834e-11, 2.869279167039059e-9, 1.791874739805025e-11,
            0.19439728866440398, 0.564205084513498, 1.5677625914936408e-9,
            1.1071292377501081e-7, 6.682158054054204e-8]

    w18t = [8.992162154254699e-13, 7.263481368968066e-11, 6.839052956845666e-13,
            9.941690919243274e-13, 0.12913503414082383, 3.0409506206984005e-12,
            2.7098510229949162e-12, 0.05030090545900891, 3.1945339872689706e-11,
            3.907803554444016e-12, 2.498095575579715e-9, 1.5461452921547135e-12,
            3.0823931861879485e-12, 5.3187482531673e-10, 2.9037874349796552e-12,
            0.20927015347378355, 0.6112938998801234, 3.2771749676076714e-10,
            8.397114124343423e-10, 2.7245127873750213e-9]

    w19t = [1.349735065468948e-10, 0.09766834612143344, 8.255178832253601e-11,
            1.7624896122680538e-11, 0.017466195123683765, 0.041194656278016314,
            1.5181689396767334e-10, 0.060564045880682656,
            1.7672406247051208e-10,
            1.5929697935895283e-10, 0.22831789328907265, 9.215376723987625e-11,
            3.028225240948744e-10, 0.1393079661617446, 1.9962145792093938e-10,
            0.036274396509238534, 1.9707575661813782e-8, 0.24657062542506455,
            4.2844788845583406e-11, 0.132635854143057]

    @test isapprox(w1.weights, w1t, rtol = 1.0e-5)
    @test isapprox(w2.weights, w2t, rtol = 1.0e-5)
    @test isapprox(w3.weights, w3t, rtol = 0.0001)
    @test isapprox(w2.weights, w3.weights, rtol = 0.0001)
    @test isapprox(w4.weights, w4t, rtol = 1.0e-5)
    @test isapprox(w5.weights, w5t, rtol = 1.0e-5)
    if !isempty(w6)
        @test isapprox(w6.weights, w6t)
        @test isapprox(w5.weights, w6.weights, rtol = 1e-7)
    end
    @test isapprox(w7.weights, w7t)
    @test isapprox(w8.weights, w8t)
    @test isapprox(w9.weights, w9t, rtol = 0.1)
    @test isapprox(w8.weights, w9.weights, rtol = 1e0)
    @test isapprox(w10.weights, w10t)
    @test isapprox(w11.weights, w11t)
    @test isapprox(w12.weights, w12t, rtol = 1.0e-6)
    @test isapprox(w11.weights, w12.weights, rtol = 1e-1)
    @test isapprox(w13.weights, w7.weights, rtol = 1e-7)
    @test isapprox(w14.weights, w8.weights, rtol = 1e-1)
    @test isapprox(w15.weights, w9.weights, rtol = 1e-1)
    @test isapprox(w16.weights, w7.weights, rtol = 1e-7)
    @test isapprox(w17.weights, w8.weights, rtol = 1e-1)
    @test isapprox(w18.weights, w9.weights, rtol = 1e-1)
    @test isapprox(w13.weights, w16.weights, rtol = 1e-7)
    @test isapprox(w14.weights, w17.weights, rtol = 1e-3)
    @test isapprox(w15.weights, w18.weights, rtol = 1e-2)
    @test isapprox(w19.weights, w1.weights, rtol = 1e-3)
end
