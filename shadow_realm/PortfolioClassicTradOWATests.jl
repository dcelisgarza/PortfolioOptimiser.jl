using CSV, Clarabel, HiGHS, LinearAlgebra, OrderedCollections, PortfolioOptimiser,
      Statistics, Test, TimeSeries

prices = TimeArray(CSV.File("./assets/stock_prices.csv"); timestamp = :date)

rf = 1.0329^(1 / 252) - 1
l = 2.0

@testset "$(:Classic), $(:Trad), $(:GMD), blank $(:OWA) and owa_w = owa_gmd(T)" begin
    portfolio = Portfolio(; prices = prices[(end - 200):end],
                          solvers = OrderedDict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                                  :params => Dict("verbose" => false,
                                                                                  "max_step_fraction" => 0.75,
                                                                                  "max_iter" => 100,
                                                                                  "equilibrate_max_iter" => 20))))
    asset_statistics!(portfolio)

    w1 = optimise!(portfolio,
                   OptimiseOpt(; owa_approx = false, rf = rf, l = l, class = :Classic,
                               type = :Trad, rm = :GMD, obj = :Min_Risk, kelly = :None))
    risk1 = calc_risk(portfolio; type = :Trad, rm = :GMD, rf = rf)
    w2 = optimise!(portfolio,
                   OptimiseOpt(; owa_approx = false, rf = rf, l = l, class = :Classic,
                               type = :Trad, rm = :GMD, obj = :Min_Risk, kelly = :Approx))
    w3 = optimise!(portfolio,
                   OptimiseOpt(; owa_approx = false, rf = rf, l = l, class = :Classic,
                               type = :Trad, rm = :GMD, obj = :Min_Risk, kelly = :Exact))
    w4 = optimise!(portfolio,
                   OptimiseOpt(; owa_approx = false, rf = rf, l = l, class = :Classic,
                               type = :Trad, rm = :GMD, obj = :Utility, kelly = :None))
    w5 = optimise!(portfolio,
                   OptimiseOpt(; owa_approx = false, rf = rf, l = l, class = :Classic,
                               type = :Trad, rm = :GMD, obj = :Utility, kelly = :Approx))
    w6 = optimise!(portfolio,
                   OptimiseOpt(; owa_approx = false, rf = rf, l = l, class = :Classic,
                               type = :Trad, rm = :GMD, obj = :Utility, kelly = :Exact))
    w7 = optimise!(portfolio,
                   OptimiseOpt(; owa_approx = false, rf = rf, l = l, class = :Classic,
                               type = :Trad, rm = :GMD, obj = :Sharpe, kelly = :None))
    risk7 = calc_risk(portfolio; type = :Trad, rm = :GMD, rf = rf)
    ret7 = dot(portfolio.mu, w7.weights)
    w8 = optimise!(portfolio,
                   OptimiseOpt(; owa_approx = false, rf = rf, l = l, class = :Classic,
                               type = :Trad, rm = :GMD, obj = :Sharpe, kelly = :Approx))
    risk8 = calc_risk(portfolio; type = :Trad, rm = :GMD, rf = rf)
    ret8 = dot(portfolio.mu, w8.weights)
    w9 = optimise!(portfolio,
                   OptimiseOpt(; owa_approx = false, rf = rf, l = l, class = :Classic,
                               type = :Trad, rm = :GMD, obj = :Sharpe, kelly = :Exact))
    risk9 = calc_risk(portfolio; type = :Trad, rm = :GMD, rf = rf)
    ret9 = dot(portfolio.mu, w9.weights)
    w10 = optimise!(portfolio,
                    OptimiseOpt(; owa_approx = false, rf = rf, l = l, class = :Classic,
                                type = :Trad, rm = :GMD, obj = :Max_Ret, kelly = :None))
    w11 = optimise!(portfolio,
                    OptimiseOpt(; owa_approx = false, rf = rf, l = l, class = :Classic,
                                type = :Trad, rm = :GMD, obj = :Max_Ret, kelly = :Approx))
    w12 = optimise!(portfolio,
                    OptimiseOpt(; owa_approx = false, rf = rf, l = l, class = :Classic,
                                type = :Trad, rm = :GMD, obj = :Max_Ret, kelly = :Exact))
    setproperty!(portfolio, :mu_l, ret7)
    w13 = optimise!(portfolio,
                    OptimiseOpt(; owa_approx = false, rf = rf, l = l, class = :Classic,
                                type = :Trad, rm = :GMD, obj = :Min_Risk, kelly = :None))
    setproperty!(portfolio, :mu_l, ret8)
    w14 = optimise!(portfolio,
                    OptimiseOpt(; owa_approx = false, rf = rf, l = l, class = :Classic,
                                type = :Trad, rm = :GMD, obj = :Min_Risk, kelly = :None))
    setproperty!(portfolio, :mu_l, ret9)
    w15 = optimise!(portfolio,
                    OptimiseOpt(; owa_approx = false, rf = rf, l = l, class = :Classic,
                                type = :Trad, rm = :GMD, obj = :Min_Risk, kelly = :None))
    setproperty!(portfolio, :mu_l, Inf)
    rmf = Symbol(lowercase(string(:GMD)) * "_u")
    setproperty!(portfolio, rmf, risk7)
    w16 = optimise!(portfolio,
                    OptimiseOpt(; owa_approx = false, rf = rf, l = l, class = :Classic,
                                type = :Trad, rm = :GMD, obj = :Max_Ret, kelly = :None))
    setproperty!(portfolio, rmf, risk8)
    w17 = optimise!(portfolio,
                    OptimiseOpt(; owa_approx = false, rf = rf, l = l, class = :Classic,
                                type = :Trad, rm = :GMD, obj = :Max_Ret, kelly = :None))
    setproperty!(portfolio, rmf, risk9)
    w18 = optimise!(portfolio,
                    OptimiseOpt(; owa_approx = false, rf = rf, l = l, class = :Classic,
                                type = :Trad, rm = :GMD, obj = :Max_Ret, kelly = :None))
    setproperty!(portfolio, rmf, risk1)
    w19 = optimise!(portfolio,
                    OptimiseOpt(; owa_approx = false, rf = rf, l = l, class = :Classic,
                                type = :Trad, rm = :GMD, obj = :Sharpe, kelly = :None))
    setproperty!(portfolio, rmf, Inf)
    w20 = optimise!(portfolio,
                    OptimiseOpt(; owa_approx = false, rf = rf, l = l, class = :Classic,
                                type = :Trad, rm = :OWA, obj = :Min_Risk, kelly = :None))
    risk20 = calc_risk(portfolio; type = :Trad, rm = :OWA, rf = rf)
    w21 = optimise!(portfolio,
                    OptimiseOpt(; owa_approx = false, rf = rf, l = l, class = :Classic,
                                type = :Trad, rm = :OWA, obj = :Min_Risk, kelly = :Approx))
    w22 = optimise!(portfolio,
                    OptimiseOpt(; owa_approx = false, rf = rf, l = l, class = :Classic,
                                type = :Trad, rm = :OWA, obj = :Min_Risk, kelly = :Exact))
    w23 = optimise!(portfolio,
                    OptimiseOpt(; owa_approx = false, rf = rf, l = l, class = :Classic,
                                type = :Trad, rm = :OWA, obj = :Utility, kelly = :None))
    w24 = optimise!(portfolio,
                    OptimiseOpt(; owa_approx = false, rf = rf, l = l, class = :Classic,
                                type = :Trad, rm = :OWA, obj = :Utility, kelly = :Approx))
    w25 = optimise!(portfolio,
                    OptimiseOpt(; owa_approx = false, rf = rf, l = l, class = :Classic,
                                type = :Trad, rm = :OWA, obj = :Utility, kelly = :Exact))
    w26 = optimise!(portfolio,
                    OptimiseOpt(; owa_approx = false, rf = rf, l = l, class = :Classic,
                                type = :Trad, rm = :OWA, obj = :Sharpe, kelly = :None))
    risk26 = calc_risk(portfolio; type = :Trad, rm = :OWA, rf = rf)
    ret26 = dot(portfolio.mu, w26.weights)
    w27 = optimise!(portfolio,
                    OptimiseOpt(; owa_approx = false, rf = rf, l = l, class = :Classic,
                                type = :Trad, rm = :OWA, obj = :Sharpe, kelly = :Approx))
    risk27 = calc_risk(portfolio; type = :Trad, rm = :OWA, rf = rf)
    ret27 = dot(portfolio.mu, w27.weights)
    w28 = optimise!(portfolio,
                    OptimiseOpt(; owa_approx = false, rf = rf, l = l, class = :Classic,
                                type = :Trad, rm = :OWA, obj = :Sharpe, kelly = :Exact))
    risk28 = calc_risk(portfolio; type = :Trad, rm = :OWA, rf = rf)
    ret28 = dot(portfolio.mu, w28.weights)
    w29 = optimise!(portfolio,
                    OptimiseOpt(; owa_approx = false, rf = rf, l = l, class = :Classic,
                                type = :Trad, rm = :OWA, obj = :Max_Ret, kelly = :None))
    w30 = optimise!(portfolio,
                    OptimiseOpt(; owa_approx = false, rf = rf, l = l, class = :Classic,
                                type = :Trad, rm = :OWA, obj = :Max_Ret, kelly = :Approx))
    w31 = optimise!(portfolio,
                    OptimiseOpt(; owa_approx = false, rf = rf, l = l, class = :Classic,
                                type = :Trad, rm = :OWA, obj = :Max_Ret, kelly = :Exact))
    setproperty!(portfolio, :mu_l, ret26)
    w32 = optimise!(portfolio,
                    OptimiseOpt(; owa_approx = false, rf = rf, l = l, class = :Classic,
                                type = :Trad, rm = :OWA, obj = :Min_Risk, kelly = :None))
    setproperty!(portfolio, :mu_l, ret27)
    w33 = optimise!(portfolio,
                    OptimiseOpt(; owa_approx = false, rf = rf, l = l, class = :Classic,
                                type = :Trad, rm = :OWA, obj = :Min_Risk, kelly = :None))
    setproperty!(portfolio, :mu_l, ret28)
    w34 = optimise!(portfolio,
                    OptimiseOpt(; owa_approx = false, rf = rf, l = l, class = :Classic,
                                type = :Trad, rm = :OWA, obj = :Min_Risk, kelly = :None))
    setproperty!(portfolio, :mu_l, Inf)
    rmf = Symbol(lowercase(string(:OWA)) * "_u")
    setproperty!(portfolio, rmf, risk26)
    w35 = optimise!(portfolio,
                    OptimiseOpt(; owa_approx = false, rf = rf, l = l, class = :Classic,
                                type = :Trad, rm = :OWA, obj = :Max_Ret, kelly = :None))
    setproperty!(portfolio, rmf, risk27)
    w36 = optimise!(portfolio,
                    OptimiseOpt(; owa_approx = false, rf = rf, l = l, class = :Classic,
                                type = :Trad, rm = :OWA, obj = :Max_Ret, kelly = :None))
    setproperty!(portfolio, rmf, risk28)
    w37 = optimise!(portfolio,
                    OptimiseOpt(; owa_approx = false, rf = rf, l = l, class = :Classic,
                                type = :Trad, rm = :OWA, obj = :Max_Ret, kelly = :None))
    setproperty!(portfolio, rmf, risk20)
    w38 = optimise!(portfolio,
                    OptimiseOpt(; owa_approx = false, rf = rf, l = l, class = :Classic,
                                type = :Trad, rm = :OWA, obj = :Sharpe, kelly = :None))
    setproperty!(portfolio, rmf, Inf)
    portfolio.owa_w = owa_gmd(size(portfolio.returns, 1))
    w39 = optimise!(portfolio,
                    OptimiseOpt(; owa_approx = false, rf = rf, l = l, class = :Classic,
                                type = :Trad, rm = :OWA, obj = :Min_Risk, kelly = :None))
    risk39 = calc_risk(portfolio; type = :Trad, rm = :OWA, rf = rf)
    w40 = optimise!(portfolio,
                    OptimiseOpt(; owa_approx = false, rf = rf, l = l, class = :Classic,
                                type = :Trad, rm = :OWA, obj = :Min_Risk, kelly = :Approx))
    w41 = optimise!(portfolio,
                    OptimiseOpt(; owa_approx = false, rf = rf, l = l, class = :Classic,
                                type = :Trad, rm = :OWA, obj = :Min_Risk, kelly = :Exact))
    w42 = optimise!(portfolio,
                    OptimiseOpt(; owa_approx = false, rf = rf, l = l, class = :Classic,
                                type = :Trad, rm = :OWA, obj = :Utility, kelly = :None))
    w43 = optimise!(portfolio,
                    OptimiseOpt(; owa_approx = false, rf = rf, l = l, class = :Classic,
                                type = :Trad, rm = :OWA, obj = :Utility, kelly = :Approx))
    w44 = optimise!(portfolio,
                    OptimiseOpt(; owa_approx = false, rf = rf, l = l, class = :Classic,
                                type = :Trad, rm = :OWA, obj = :Utility, kelly = :Exact))
    w45 = optimise!(portfolio,
                    OptimiseOpt(; owa_approx = false, rf = rf, l = l, class = :Classic,
                                type = :Trad, rm = :OWA, obj = :Sharpe, kelly = :None))
    risk45 = calc_risk(portfolio; type = :Trad, rm = :OWA, rf = rf)
    ret45 = dot(portfolio.mu, w45.weights)
    w46 = optimise!(portfolio,
                    OptimiseOpt(; owa_approx = false, rf = rf, l = l, class = :Classic,
                                type = :Trad, rm = :OWA, obj = :Sharpe, kelly = :Approx))
    risk46 = calc_risk(portfolio; type = :Trad, rm = :OWA, rf = rf)
    ret46 = dot(portfolio.mu, w46.weights)
    w47 = optimise!(portfolio,
                    OptimiseOpt(; owa_approx = false, rf = rf, l = l, class = :Classic,
                                type = :Trad, rm = :OWA, obj = :Sharpe, kelly = :Exact))
    risk47 = calc_risk(portfolio; type = :Trad, rm = :OWA, rf = rf)
    ret47 = dot(portfolio.mu, w47.weights)
    w48 = optimise!(portfolio,
                    OptimiseOpt(; owa_approx = false, rf = rf, l = l, class = :Classic,
                                type = :Trad, rm = :OWA, obj = :Max_Ret, kelly = :None))
    w49 = optimise!(portfolio,
                    OptimiseOpt(; owa_approx = false, rf = rf, l = l, class = :Classic,
                                type = :Trad, rm = :OWA, obj = :Max_Ret, kelly = :Approx))
    w50 = optimise!(portfolio,
                    OptimiseOpt(; owa_approx = false, rf = rf, l = l, class = :Classic,
                                type = :Trad, rm = :OWA, obj = :Max_Ret, kelly = :Exact))
    setproperty!(portfolio, :mu_l, ret45)
    w51 = optimise!(portfolio,
                    OptimiseOpt(; owa_approx = false, rf = rf, l = l, class = :Classic,
                                type = :Trad, rm = :OWA, obj = :Min_Risk, kelly = :None))
    setproperty!(portfolio, :mu_l, ret46)
    w52 = optimise!(portfolio,
                    OptimiseOpt(; owa_approx = false, rf = rf, l = l, class = :Classic,
                                type = :Trad, rm = :OWA, obj = :Min_Risk, kelly = :None))
    setproperty!(portfolio, :mu_l, ret47)
    w53 = optimise!(portfolio,
                    OptimiseOpt(; owa_approx = false, rf = rf, l = l, class = :Classic,
                                type = :Trad, rm = :OWA, obj = :Min_Risk, kelly = :None))
    setproperty!(portfolio, :mu_l, Inf)
    rmf = Symbol(lowercase(string(:OWA)) * "_u")
    setproperty!(portfolio, rmf, risk45)
    w54 = optimise!(portfolio,
                    OptimiseOpt(; owa_approx = false, rf = rf, l = l, class = :Classic,
                                type = :Trad, rm = :OWA, obj = :Max_Ret, kelly = :None))
    setproperty!(portfolio, rmf, risk46)
    w55 = optimise!(portfolio,
                    OptimiseOpt(; owa_approx = false, rf = rf, l = l, class = :Classic,
                                type = :Trad, rm = :OWA, obj = :Max_Ret, kelly = :None))
    setproperty!(portfolio, rmf, risk47)
    w56 = optimise!(portfolio,
                    OptimiseOpt(; owa_approx = false, rf = rf, l = l, class = :Classic,
                                type = :Trad, rm = :OWA, obj = :Max_Ret, kelly = :None))
    setproperty!(portfolio, rmf, risk39)
    w57 = optimise!(portfolio,
                    OptimiseOpt(; owa_approx = false, rf = rf, l = l, class = :Classic,
                                type = :Trad, rm = :OWA, obj = :Sharpe, kelly = :None))
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

    w3t = [5.323890319967473e-10, 2.2292112184369823e-9, 1.1712519600144323e-9,
           0.011344717229265862, 0.04116339712572708, 0.02111825161481731,
           2.164157814422413e-10, 0.047256020508289955, 1.0052989476585146e-9,
           8.091710915406835e-10, 0.07082523408083218, 9.26621100982225e-11,
           1.6964447909481636e-10, 0.29255890184817535, 9.639519884799085e-11,
           0.014800789769721878, 0.05532156457201481, 0.22573814910460713,
           0.00811880253571274, 0.21175416528839586]

    w4t = [4.395442944440727e-10, 2.1495615257047445e-9, 8.700158416075159e-10,
           0.006589303508967539, 0.05330845589356504, 7.632737386668005e-10,
           9.831583865942048e-11, 0.06008335161013898, 1.3797121555924764e-9,
           3.0160588760343415e-9, 0.06394372429204083, 4.486199590323317e-11,
           1.0975948346699443e-10, 0.24314317639043878, 5.90094645737884e-11,
           0.018483948217979402, 0.08594753958504305, 0.216033164253688, 0.05666930366548,
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

    w7t = [1.0960279114854144e-12, 4.2713859884519724e-11, 3.322171869553163e-12,
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
            3.664169883135698e-10, 7.000039902341564e-10, 4.5030867400627253e-10,
            6.996286227848405e-9, 3.6287634044926756e-8, 9.000571133076656e-10,
            3.4765700831013475e-9, 7.60932312745152e-10]

    w11t = [5.995643893118203e-12, 1.7788172572403522e-11, 7.42248967697158e-12,
            2.5855688894534932e-11, 0.8577737311672241, 5.084251669919802e-12,
            1.8918809363559223e-12, 1.7873684105427732e-11, 9.127337657198234e-11,
            1.8171837160854858e-11, 3.850566386470494e-12, 2.5508401127966545e-13,
            4.0216144858012615e-12, 3.339109115577017e-12, 2.0662179397071288e-12,
            1.8569702892630115e-10, 0.14222626837294614, 8.073610828892573e-12,
            5.65606511499808e-11, 4.608669048088745e-12]

    w12t = [6.8783457384567556e-9, 1.2274774655426258e-8, 7.210231160923748e-9,
            1.6157723542114e-8, 0.8926318884419521, 2.236669895286051e-9,
            3.0721278132094936e-9, 1.1931574111287059e-8, 5.313990853277208e-8,
            1.1675553033694841e-8, 5.76375880463489e-9, 4.1812752119557695e-9,
            2.3871108169602047e-9, 5.509568787249582e-9, 2.9557632944416428e-9,
            9.517716377203498e-8, 0.10736782516025675, 7.544504043725093e-9,
            3.230881102698668e-8, 5.9929266940059665e-9]

    w13t = [2.111064693976598e-12, 4.199098436247614e-11, 2.4277750791679688e-12,
            3.8079944685205154e-11, 0.21084481684472242, 2.6325394341890063e-11,
            2.398425045782114e-11, 0.011087215230686672, 0.021727930888863614,
            1.5980784508945338e-10, 2.2534788169784417e-11, 1.7758263471786547e-11,
            2.4160204726553847e-11, 1.030058571203951e-11, 2.000500776826796e-11,
            0.09769732518814264, 0.39851241713315694, 5.853448351366192e-11,
            0.26013029423727785, 2.9129230394432605e-11]

    w14t = [8.942270766111967e-13, 4.997065086540317e-11, 4.597515424452085e-12,
            4.14025186234731e-11, 0.2045787142958964, 2.757625635103089e-11,
            2.5526588174105618e-11, 0.028823839199628453, 0.00935681866157209,
            2.2178037048090422e-10, 3.349076787579454e-11, 1.87998796931707e-11,
            2.5264985600201933e-11, 1.653886341218741e-11, 2.0604214115929157e-11,
            0.09537027357843764, 0.377524299391081, 8.51485669294363e-11,
            0.2843460542557699, 4.601891415928919e-11]

    w15t = [5.1306740699054454e-12, 3.498203888295839e-11, 9.443660977510647e-13,
            3.0051209746434546e-11, 0.2096190099044751, 2.6830393435136183e-11,
            2.4866423157271724e-11, 0.015388952005987784, 0.01947723659851483,
            1.4698330462316705e-10, 1.836237237661002e-11, 1.931290540546062e-11,
            2.4901861449564757e-11, 6.724696953439446e-12, 2.113917582584407e-11,
            0.09756081719654436, 0.3941356326147162, 5.2721875186497566e-11,
            0.26381835124131175, 2.5498813689609923e-11]

    w16t = [8.645417462703117e-13, 9.74931569493778e-12, 1.7691244640820176e-12,
            8.874979261575232e-12, 0.21078639257923237, 3.922009126537224e-12,
            3.4338064345481404e-12, 0.011063602423968855, 0.02153048309961715,
            3.387355507162292e-11, 5.799061166068984e-12, 2.2147166479480898e-12,
            3.476808868173112e-12, 3.3237370929321917e-12, 2.640299454582965e-12,
            0.09773989516924422, 0.3986125551970178, 1.3091156994666361e-11,
            0.26026707143077404, 7.112393143958492e-12]

    w17t = [2.196665141984919e-12, 1.2060486285149243e-11, 3.2469345250787356e-12,
            1.035882803391085e-11, 0.20454337993180025, 2.8837388491504335e-12,
            2.4643498937450153e-12, 0.02883361320119504, 0.009228447548408824,
            4.5440170588235665e-11, 8.78290112077899e-12, 1.1948258696922626e-12,
            2.427952696676779e-12, 5.51739407846932e-12, 1.524788450375429e-12,
            0.09537008234204332, 0.3777286103721778, 1.876506344052584e-11,
            0.2842958664763445, 1.116624337339785e-11]

    w18t = [9.09711465126155e-13, 9.449290266913617e-12, 1.7739420562277521e-12,
            8.37151312097126e-12, 0.20966915944272185, 3.5479502740177604e-12,
            3.0757608336758066e-12, 0.015378180154910333, 0.01950214340438561,
            3.35180762679845e-11, 5.785868839429511e-12, 1.9479838142418733e-12,
            3.1193764890204444e-12, 3.3382615997835853e-12, 2.3158677660472516e-12,
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
                                                                                  "max_step_fraction" => 0.75))))
    asset_statistics!(portfolio)

    w1 = optimise!(portfolio,
                   OptimiseOpt(; rf = rf, l = l, class = :Classic, type = :Trad, rm = :RG,
                               obj = :Min_Risk, kelly = :None))
    risk1 = calc_risk(portfolio; type = :Trad, rm = :RG, rf = rf)
    w2 = optimise!(portfolio,
                   OptimiseOpt(; rf = rf, l = l, class = :Classic, type = :Trad, rm = :RG,
                               obj = :Min_Risk, kelly = :Approx))
    w3 = optimise!(portfolio,
                   OptimiseOpt(; rf = rf, l = l, class = :Classic, type = :Trad, rm = :RG,
                               obj = :Min_Risk, kelly = :Exact))
    w4 = optimise!(portfolio,
                   OptimiseOpt(; rf = rf, l = l, class = :Classic, type = :Trad, rm = :RG,
                               obj = :Utility, kelly = :None))
    w5 = optimise!(portfolio,
                   OptimiseOpt(; rf = rf, l = l, class = :Classic, type = :Trad, rm = :RG,
                               obj = :Utility, kelly = :Approx))
    w6 = optimise!(portfolio,
                   OptimiseOpt(; rf = rf, l = l, class = :Classic, type = :Trad, rm = :RG,
                               obj = :Utility, kelly = :Exact))
    w7 = optimise!(portfolio,
                   OptimiseOpt(; rf = rf, l = l, class = :Classic, type = :Trad, rm = :RG,
                               obj = :Sharpe, kelly = :None))
    risk7 = calc_risk(portfolio; type = :Trad, rm = :RG, rf = rf)
    ret7 = dot(portfolio.mu, w7.weights)
    w8 = optimise!(portfolio,
                   OptimiseOpt(; rf = rf, l = l, class = :Classic, type = :Trad, rm = :RG,
                               obj = :Sharpe, kelly = :Approx))
    risk8 = calc_risk(portfolio; type = :Trad, rm = :RG, rf = rf)
    ret8 = dot(portfolio.mu, w8.weights)
    w10 = optimise!(portfolio,
                    OptimiseOpt(; rf = rf, l = l, class = :Classic, type = :Trad, rm = :RG,
                                obj = :Max_Ret, kelly = :None))
    w11 = optimise!(portfolio,
                    OptimiseOpt(; rf = rf, l = l, class = :Classic, type = :Trad, rm = :RG,
                                obj = :Max_Ret, kelly = :Approx))
    w12 = optimise!(portfolio,
                    OptimiseOpt(; rf = rf, l = l, class = :Classic, type = :Trad, rm = :RG,
                                obj = :Max_Ret, kelly = :Exact))
    setproperty!(portfolio, :mu_l, ret7)
    w13 = optimise!(portfolio,
                    OptimiseOpt(; rf = rf, l = l, class = :Classic, type = :Trad, rm = :RG,
                                obj = :Min_Risk, kelly = :None))
    setproperty!(portfolio, :mu_l, ret8)
    w14 = optimise!(portfolio,
                    OptimiseOpt(; rf = rf, l = l, class = :Classic, type = :Trad, rm = :RG,
                                obj = :Min_Risk, kelly = :None))
    setproperty!(portfolio, :mu_l, Inf)
    rmf = Symbol(lowercase(string(:RG)) * "_u")
    setproperty!(portfolio, rmf, risk7)
    w16 = optimise!(portfolio,
                    OptimiseOpt(; rf = rf, l = l, class = :Classic, type = :Trad, rm = :RG,
                                obj = :Max_Ret, kelly = :None))
    setproperty!(portfolio, rmf, risk8)
    w17 = optimise!(portfolio,
                    OptimiseOpt(; rf = rf, l = l, class = :Classic, type = :Trad, rm = :RG,
                                obj = :Max_Ret, kelly = :None))
    setproperty!(portfolio, rmf, risk1)
    w19 = optimise!(portfolio,
                    OptimiseOpt(; rf = rf, l = l, class = :Classic, type = :Trad, rm = :RG,
                                obj = :Sharpe, kelly = :None))
    setproperty!(portfolio, rmf, Inf)

    w1t = [2.7873911403633515e-10, 0.12080623340248343, 1.8131427072844702e-9,
           0.010186667805044073, 4.361163529695424e-10, 0.0737419471409221,
           1.0324812889613347e-10, 0.036141260913597116, 1.2619746566994148e-10,
           1.893194223114784e-10, 0.35461674653492997, 1.7401570030245953e-10,
           4.131266459289963e-11, 0.009915761176833628, 1.8908182495338059e-10,
           3.0931929857659314e-10, 0.1169092357922682, 3.5540996583874167e-10,
           2.2725245876536357e-10, 0.2776821429907665]

    w2t = [5.329640173952366e-10, 0.12080623620629036, 3.1422460341597667e-9,
           0.010186664556761694, 7.324346548231717e-10, 0.07374194450064485,
           1.52326203839161e-10, 0.03614126141900872, 2.4637859031752326e-10,
           3.673096675965166e-10, 0.3546167475039565, 3.385339383286379e-10,
           6.289963548046649e-11, 0.009915760986671612, 3.0421946963355653e-10,
           6.935238582276558e-10, 0.1169092353416341, 6.686492609495671e-10,
           4.3718443443274295e-10, 0.27768214180636236]

    w3t = [2.076025281986918e-9, 0.12080625405763794, 1.144003364538009e-8,
           0.010186640966112541, 2.719158284030966e-9, 0.0737419213927987,
           8.363254382858614e-10, 0.036141260942683065, 1.0921242657300607e-9,
           1.288316557730707e-9, 0.35461676322188374, 1.462021562029873e-9,
           5.246566238680777e-10, 0.009915754060003609, 1.3471424004856027e-9,
           2.2376942698140677e-9, 0.11690924401143217, 2.2495212737991297e-9,
           1.689626179164416e-9, 0.27768213238480244]

    w4t = [1.5218960826317059e-13, 0.12276734050506966, 0.008565562635440845,
           3.8086405878606735e-12, 8.614088923893088e-14, 0.06024115368418488,
           2.511737723477254e-12, 0.03543856388632871, 1.8199328483712365e-12,
           1.1663940571002837e-12, 0.3596410911054393, 1.136225975160405e-12,
           2.857215093924921e-12, 0.007480617549000438, 1.4575209615511937e-12,
           3.5506933165276716e-12, 0.1305269569777664, 1.1525649478062172e-12,
           4.308283867966185e-13, 0.27533871363663986]

    w5t = [3.8469299909613237e-10, 0.12276733572333637, 0.008565559103327281,
           7.979175710868809e-10, 5.719809681238536e-10, 0.06024115566318636,
           9.654531303513558e-11, 0.03543856374099673, 1.766090131202888e-10,
           2.75065746301491e-10, 0.35964108982847987, 2.1499762950220632e-10,
           4.4227765489593413e-11, 0.007480617088333502, 1.728421560322919e-10,
           3.8394748696727555e-10, 0.13052696110232667, 7.525129803251529e-10,
           3.4507561596552835e-10, 0.27533871353359807]

    w6t = [6.090284743262061e-10, 0.12276733352254886, 0.008565559638736744,
           1.0888027934621796e-9, 7.918017825629352e-10, 0.06024115439888005,
           2.0566858501052024e-10, 0.03543856355676416, 3.137257898649479e-10,
           4.4232030120876955e-10, 0.35964109048911164, 3.6710204171317164e-10,
           1.2712937521523104e-10, 0.0074806155785989165, 3.1581320225464667e-10,
           5.159807855497086e-10, 0.13052696287377566, 9.264097283313578e-10,
           5.255636229044951e-10, 0.27533871371223756]

    w7t = [1.18516216356278e-9, 1.871483475477399e-9, 2.317719350237808e-9,
           1.702517364383894e-9, 0.11689985768701226, 1.4699386614407516e-10,
           7.071961137798103e-10, 6.565006268107687e-9, 2.5853905687389923e-9,
           3.6336316693028458e-9, 1.7485120296440887e-9, 4.968966197946203e-10,
           6.644833362087502e-11, 2.175061607519454e-9, 3.7607621648443904e-10,
           7.189123745434342e-8, 0.8831000336375262, 1.9703400234320536e-9,
           7.344611684478502e-9, 1.8911768647737335e-9]

    w8t = [2.790527639179805e-8, 5.543615377220252e-8, 4.9045803721595204e-8,
           5.215548716539013e-8, 0.1168964493805144, 8.42389080920686e-9,
           1.8860824041093717e-8, 2.067215103117495e-7, 1.4293623221013188e-7,
           9.677804460916861e-8, 4.429091078619187e-8, 1.7583573348789464e-8,
           7.0066712271744735e-9, 5.5495443743094726e-8, 1.3338199885368256e-8,
           4.9547786853397585e-6, 0.8830973995586079, 4.883746552555989e-8,
           3.009886855272595e-7, 5.0478019289997525e-8]

    w10t = [3.5735255938507273e-9, 6.014669176456152e-9, 3.830651842153097e-9,
            8.926759812698012e-9, 0.9999997354907106, 1.2840422327970916e-9,
            1.8780190636412462e-9, 5.468141856403255e-9, 2.2967859741234005e-8,
            5.634048738753463e-9, 2.9344474269947232e-9, 2.3349815655301264e-9,
            1.4762414402715817e-9, 2.873570205701677e-9, 1.8278609989590683e-9,
            2.8185993897629944e-8, 1.4432076123351943e-7, 3.6988385456507575e-9,
            1.4153821555894728e-8, 3.1250544067940643e-9]

    w11t = [1.052334247856691e-10, 3.0959642642328976e-10, 1.3014157375465758e-10,
            4.4874693967048003e-10, 0.8577734832078435, 8.535317849418292e-11,
            3.1719154449594665e-11, 3.113390522804682e-10, 1.5764762143315813e-9,
            3.1598157693495734e-10, 6.81829979173464e-11, 5.895936714784277e-12,
            6.73928980673523e-11, 5.948202736887693e-11, 3.557293279583177e-11,
            3.197491150546457e-9, 0.14222650884196003, 1.4157151319690525e-10,
            9.783727180253945e-10, 8.164671462536028e-11]

    w12t = [4.502088679248276e-9, 8.026789256301228e-9, 4.708313192247162e-9,
            1.0563193089964122e-8, 0.8926330429712972, 1.4761874423354312e-9,
            2.0119573802827614e-9, 7.788743446209408e-9, 3.498564106619679e-8,
            7.6060043167538e-9, 3.767980708528448e-9, 2.731684532585433e-9,
            1.5623831944020235e-9, 3.602618817538397e-9, 1.933744486474113e-9,
            6.263510821188843e-8, 0.1073667690973204, 4.928736269105768e-9,
            2.1186236622823188e-8, 3.9139717053691975e-9]

    w13t = [4.2997999317991255e-11, 1.0552726405075377e-10, 1.2716766728941808e-10,
            8.144443561649235e-11, 0.11689983438378539, 1.276658725654182e-11,
            2.975584833791701e-11, 5.3723045084201284e-8, 1.0938597844891808e-10,
            1.9837869121739862e-10, 1.0718250381304394e-10, 1.5008024445501298e-11,
            1.9712958003110112e-11, 1.9384123034350407e-10, 2.9483295351083227e-12,
            1.0343107158708465e-9, 0.8831001092870253, 1.0156798795584252e-10,
            2.9463032853557197e-10, 1.2951774420947656e-10]

    w14t = [3.704326624238894e-11, 9.650246955319167e-11, 1.1130048756112682e-10,
            7.764157932034172e-11, 0.11689772834729636, 1.6610101922148432e-11,
            2.1256409057793786e-11, 2.656678874799244e-6, 1.083886297372283e-10,
            1.779195275387366e-10, 9.006943338891786e-11, 9.593603460050384e-12,
            2.2468197975879327e-11, 1.8662646059728973e-10, 1.7275135000673278e-12,
            1.1370432842988742e-9, 0.8830996124023996, 8.837900269743824e-11,
            2.7858318000076414e-10, 1.1027615038497895e-10]

    w16t = [2.716771124469647e-13, 2.761023205879752e-11, 6.499891412700421e-12,
            5.234265601174752e-13, 0.11689987313735063, 7.326004421112206e-14,
            1.114164841753301e-13, 1.1051256563558323e-8, 8.590668010168357e-13,
            1.8059003017468587e-11, 3.33280873366803e-9, 8.885990476213998e-13,
            1.637173511141264e-12, 3.2713035039324393e-9, 1.0328174762506177e-12,
            5.1410430202148184e-11, 0.8831001062797235, 3.106277948976559e-12,
            5.935577387170622e-12, 2.809538081525264e-9]

    w17t = [8.664913238144214e-10, 2.39727793359431e-9, 1.931544046702729e-9,
            1.5592665756061173e-9, 0.11689851463986643, 2.7399085939490286e-10,
            7.135674337793571e-10, 1.6240089180303066e-6, 1.986102281742648e-9,
            3.893819514097891e-9, 2.110203282680541e-9, 6.225415749507053e-10,
            2.016790963714184e-10, 3.489950017371612e-9, 4.504797239867769e-10,
            5.886288436758647e-8, 0.8830997683151328, 1.6842173274678652e-9,
            5.694241497827903e-9, 6.297825912878991e-9]

    w19t = [4.823712292694709e-13, 0.12080624408602351, 6.886975202494395e-8,
            0.010186591889464898, 1.3717873652983117e-12, 0.07374184372253122,
            1.3062255259320368e-13, 0.03614125520227511, 3.162380788462724e-14,
            1.8156376415172378e-12, 0.35461678362299054, 1.0858932787950072e-13,
            2.5332069326954925e-13, 0.009915742682836222, 8.968798342345741e-14,
            6.280506213540923e-13, 0.1169093434903966, 8.652177917102329e-13,
            4.596999215860765e-13, 0.27768212642749324]

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
    @test isapprox(w10.weights, w10t)
    @test isapprox(w11.weights, w11t)
    @test isapprox(w12.weights, w12t, rtol = 1.0e-5)
    @test isapprox(w11.weights, w12.weights, rtol = 1e-1)
    @test isapprox(w13.weights, w7.weights, rtol = 1e-6)
    @test isapprox(w14.weights, w8.weights, rtol = 1e-5)
    @test isapprox(w16.weights, w7.weights, rtol = 1e-6)
    @test isapprox(w17.weights, w8.weights, rtol = 1e-5)
    @test isapprox(w13.weights, w16.weights, rtol = 1e-6)
    @test isapprox(w14.weights, w17.weights, rtol = 1e-5)
    @test isapprox(w19.weights, w1.weights, rtol = 1e-6)
end

@testset "$(:Classic), $(:Trad), $(:RCVaR)" begin
    portfolio = Portfolio(; prices = prices[(end - 200):end],
                          solvers = OrderedDict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                                  :params => Dict("verbose" => false,
                                                                                  "max_step_fraction" => 0.75))))
    asset_statistics!(portfolio)

    w1 = optimise!(portfolio,
                   OptimiseOpt(; rf = rf, l = l, class = :Classic, type = :Trad,
                               rm = :RCVaR, obj = :Min_Risk, kelly = :None))
    risk1 = calc_risk(portfolio; type = :Trad, rm = :RCVaR, rf = rf)
    w2 = optimise!(portfolio,
                   OptimiseOpt(; rf = rf, l = l, class = :Classic, type = :Trad,
                               rm = :RCVaR, obj = :Min_Risk, kelly = :Approx))
    w3 = optimise!(portfolio,
                   OptimiseOpt(; rf = rf, l = l, class = :Classic, type = :Trad,
                               rm = :RCVaR, obj = :Min_Risk, kelly = :Exact))
    w4 = optimise!(portfolio,
                   OptimiseOpt(; rf = rf, l = l, class = :Classic, type = :Trad,
                               rm = :RCVaR, obj = :Utility, kelly = :None))
    w5 = optimise!(portfolio,
                   OptimiseOpt(; rf = rf, l = l, class = :Classic, type = :Trad,
                               rm = :RCVaR, obj = :Utility, kelly = :Approx))
    w6 = optimise!(portfolio,
                   OptimiseOpt(; rf = rf, l = l, class = :Classic, type = :Trad,
                               rm = :RCVaR, obj = :Utility, kelly = :Exact))
    w7 = optimise!(portfolio,
                   OptimiseOpt(; rf = rf, l = l, class = :Classic, type = :Trad,
                               rm = :RCVaR, obj = :Sharpe, kelly = :None))
    risk7 = calc_risk(portfolio; type = :Trad, rm = :RCVaR, rf = rf)
    ret7 = dot(portfolio.mu, w7.weights)
    w8 = optimise!(portfolio,
                   OptimiseOpt(; rf = rf, l = l, class = :Classic, type = :Trad,
                               rm = :RCVaR, obj = :Sharpe, kelly = :Approx))
    risk8 = calc_risk(portfolio; type = :Trad, rm = :RCVaR, rf = rf)
    ret8 = dot(portfolio.mu, w8.weights)
    w9 = optimise!(portfolio,
                   OptimiseOpt(; rf = rf, l = l, class = :Classic, type = :Trad,
                               rm = :RCVaR, obj = :Sharpe, kelly = :Exact))
    risk9 = calc_risk(portfolio; type = :Trad, rm = :RCVaR, rf = rf)
    ret9 = dot(portfolio.mu, w9.weights)
    w10 = optimise!(portfolio,
                    OptimiseOpt(; rf = rf, l = l, class = :Classic, type = :Trad,
                                rm = :RCVaR, obj = :Max_Ret, kelly = :None))
    w11 = optimise!(portfolio,
                    OptimiseOpt(; rf = rf, l = l, class = :Classic, type = :Trad,
                                rm = :RCVaR, obj = :Max_Ret, kelly = :Approx))
    w12 = optimise!(portfolio,
                    OptimiseOpt(; rf = rf, l = l, class = :Classic, type = :Trad,
                                rm = :RCVaR, obj = :Max_Ret, kelly = :Exact))
    setproperty!(portfolio, :mu_l, ret7)
    w13 = optimise!(portfolio,
                    OptimiseOpt(; rf = rf, l = l, class = :Classic, type = :Trad,
                                rm = :RCVaR, obj = :Min_Risk, kelly = :None))
    setproperty!(portfolio, :mu_l, ret8)
    w14 = optimise!(portfolio,
                    OptimiseOpt(; rf = rf, l = l, class = :Classic, type = :Trad,
                                rm = :RCVaR, obj = :Min_Risk, kelly = :None))
    setproperty!(portfolio, :mu_l, ret9)
    w15 = optimise!(portfolio,
                    OptimiseOpt(; rf = rf, l = l, class = :Classic, type = :Trad,
                                rm = :RCVaR, obj = :Min_Risk, kelly = :None))
    setproperty!(portfolio, :mu_l, Inf)
    rmf = Symbol(lowercase(string(:RCVaR)) * "_u")
    setproperty!(portfolio, rmf, risk7)
    w16 = optimise!(portfolio,
                    OptimiseOpt(; rf = rf, l = l, class = :Classic, type = :Trad,
                                rm = :RCVaR, obj = :Max_Ret, kelly = :None))
    setproperty!(portfolio, rmf, risk8)
    w17 = optimise!(portfolio,
                    OptimiseOpt(; rf = rf, l = l, class = :Classic, type = :Trad,
                                rm = :RCVaR, obj = :Max_Ret, kelly = :None))
    setproperty!(portfolio, rmf, risk9)
    w18 = optimise!(portfolio,
                    OptimiseOpt(; rf = rf, l = l, class = :Classic, type = :Trad,
                                rm = :RCVaR, obj = :Max_Ret, kelly = :None))
    setproperty!(portfolio, rmf, risk1 + 1e-2 * risk1)
    w19 = optimise!(portfolio,
                    OptimiseOpt(; rf = rf, l = l, class = :Classic, type = :Trad,
                                rm = :RCVaR, obj = :Sharpe, kelly = :None))
    setproperty!(portfolio, rmf, Inf)

    w1t = [6.845747701403724e-10, 0.09028989659085701, 8.803720112748341e-10,
           0.014128522895800116, 0.010545528153169432, 0.12334331655357939,
           4.345854984819632e-10, 0.04829573470186048, 4.815137741771747e-10,
           5.897135290682549e-10, 0.21255172785657894, 8.404609379948049e-10,
           7.645868959975233e-11, 0.21099540524125587, 4.556569127838046e-10,
           0.01649165838789583, 2.341719690812881e-9, 0.1890733240797836,
           7.693170629537897e-10, 0.08428487798484642]

    w2t = [6.606668435134837e-10, 0.09028989518103345, 8.11227346544289e-10,
           0.014128523214352028, 0.010545531466864568, 0.12334331528459883,
           4.280364169633914e-10, 0.048295734817207524, 4.4913620994417615e-10,
           5.795924298087613e-10, 0.21255173154186055, 7.250557230529528e-10,
           7.801320534601268e-11, 0.21099540544897213, 4.1585489853979857e-10,
           0.016491661019900013, 2.1221742006578195e-9, 0.18907331876679415,
           7.257394732809656e-10, 0.08428487626291999]

    w3t = [1.3230340767314797e-9, 0.09028990010517651, 1.5297689458910403e-9,
           0.014128524063068144, 0.01054551439182247, 0.12334331867873195,
           8.901183861171882e-10, 0.048295733977400536, 9.30490932134608e-10,
           1.1482748522606057e-9, 0.21255171483851182, 1.5010791250075835e-9,
           2.468751578874056e-10, 0.21099540270382092, 8.985035786912332e-10,
           0.016491653065857627, 4.193343215099625e-9, 0.18907333798946094,
           1.4294488905315346e-9, 0.08428488609521179]

    w4t = [1.8239835602353191e-9, 0.08975186856764639, 2.8615698635644866e-9,
           0.01416155933202109, 0.011511111941462962, 0.12331558736735213,
           7.594786551100019e-10, 0.048422689197105975, 1.3446305865645918e-9,
           1.8421961718357357e-9, 0.21259063417080554, 1.7854325937231454e-9,
           1.4190768222348424e-10, 0.2114121066533548, 8.379537139254804e-10,
           0.016784953793638018, 2.1069790413780844e-8, 0.18809407875859815,
           2.7566607825302103e-9, 0.08395537499441087]

    w5t = [4.991221172190947e-10, 0.08975186435372354, 7.507841086923247e-10,
           0.014161582368343265, 0.011511154326239975, 0.12331560535425769,
           2.0974152133618524e-10, 0.04842269307680932, 3.5195754190536525e-10,
           5.06045656059165e-10, 0.2125906226879721, 4.737691951030925e-10,
           3.6010073059109173e-11, 0.2114121288000131, 2.2258633863620673e-10,
           0.016784952089139035, 5.985700543116538e-9, 0.18809403423387117,
           7.500798619796658e-10, 0.08395535292383388]

    w6t = [4.002866048076541e-8, 0.08956174036907162, 5.456496597171497e-8,
           0.01461595513557362, 0.012326195603033616, 0.12284465891853509,
           1.7365043632890118e-8, 0.048599442621491344, 3.6882175978059996e-8,
           5.1078935978603024e-8, 0.2122516405273212, 3.8019737338871854e-8,
           6.6575139552254436e-9, 0.21170525251962352, 2.0238104749958166e-8,
           0.017048555325479928, 1.1404201046791317e-6, 0.18730740839776847,
           7.526405630225357e-8, 0.08373767006280242]

    w7t = [1.8059286082193725e-9, 6.994818748246583e-9, 1.7842995821148485e-9,
           3.3595718970751018e-9, 0.08771766389576391, 4.46581008623452e-10,
           6.658222332060181e-10, 0.12008345890196084, 1.0186649390349922e-7,
           6.659247014437244e-9, 3.783077910679677e-9, 1.383457310792321e-9,
           5.358428750491345e-10, 2.8435302781640896e-9, 5.761194433603441e-10,
           0.15120894388888653, 0.5292694320233682, 4.7042836850768194e-9,
           0.11172036003671369, 3.844232290506081e-9]

    w8t = [6.176008955527821e-9, 2.192831158102339e-8, 6.091433087291915e-9,
           1.0272885948132517e-8, 0.08712899639687167, 1.8650136728396854e-9,
           2.5842671296386086e-9, 0.11919957082563445, 2.3467679096126403e-7,
           2.238842744711071e-8, 1.3187271368835547e-8, 4.598967748247975e-9,
           2.195523706408022e-9, 9.79882599284304e-9, 2.4104142498726256e-9,
           0.15025983881114163, 0.5287592084871203, 1.6636072505324326e-8,
           0.11465201723716868, 1.3431848858828358e-8]

    w9t = [5.294864628753274e-8, 1.7349360556061008e-7, 5.2581349398223304e-8,
           9.060515218840288e-8, 0.08771738147416151, 1.5900281595277838e-8,
           2.2537230235620554e-8, 0.12005614262452197, 2.8467090680218895e-6,
           1.775987675556279e-7, 1.0088290961864575e-7, 3.958456396951245e-8,
           1.9196144616729544e-8, 7.692297056873234e-8, 2.1035383715167923e-8,
           0.15120016749930282, 0.5292705070559302, 1.2727462011998397e-7,
           0.11175188068176627, 1.0339362389181894e-7]

    w10t = [1.4115926241721963e-8, 2.3748314095614455e-8, 1.5131016418845087e-8,
            3.530102852398287e-8, 0.9999989447155134, 5.0879965971967655e-9,
            7.453248832759362e-9, 2.1582379212697126e-8, 9.137810649526648e-8,
            2.2238226198180427e-8, 1.1584908476125956e-8, 9.258567528354886e-9,
            5.878398289460337e-9, 1.1345927905923476e-8, 7.25572744893209e-9,
            1.1228489740455761e-7, 5.786095062554166e-7, 1.4596248617109073e-8,
            5.6097939331601985e-8, 1.233612269666117e-8]

    w11t = [1.0037999113218163e-10, 2.9641025795494843e-10, 1.2417887293015335e-10,
            4.302152607534612e-10, 0.8577734733912407, 8.322921401873196e-11,
            3.090965789250156e-11, 2.9795878487247924e-10, 1.5148785782594854e-9,
            3.0267649397054186e-10, 6.477029447160244e-11, 5.0122243441388885e-12,
            6.576290215850661e-11, 5.633878269442262e-11, 3.4178678330380736e-11,
            3.0774230611369046e-9, 0.1422265189723813, 1.3507052929128343e-10,
            9.394626769018034e-10, 7.75220812974745e-11]

    w12t = [4.780638291230175e-9, 8.526336767613021e-9, 5.001194016171701e-9,
            1.121959568720941e-8, 0.8926329066311905, 1.5666181764298914e-9,
            2.1340778537911874e-9, 8.272411311422974e-9, 3.7211106520488176e-8,
            8.080216556459967e-9, 4.001896809393193e-9, 2.900248089141273e-9,
            1.6572849087397765e-9, 3.823932624969129e-9, 2.050136403856455e-9,
            6.661738965663469e-8, 0.10736689361334668, 5.2345851809117e-9,
            2.2521159073907848e-8, 4.156634979308465e-9]

    w13t = [7.1225604903281e-11, 3.89346920257538e-10, 7.441316769482621e-11,
            1.7616726662201127e-10, 0.08771741819188573, 2.7368941051846073e-11,
            8.940832850246839e-12, 0.12008326700540226, 4.329933751596991e-9,
            3.6655362863808624e-10, 1.9364738146358e-10, 3.6455610440793596e-11,
            1.8696963313940842e-11, 1.3437844004619931e-10, 1.4233089709444487e-11,
            0.15120880083006258, 0.5292695130113965, 2.574637293655506e-10,
            0.11172099466095674, 2.0147073944192543e-10]

    w14t = [2.7496085764195365e-12, 5.518254644820851e-13, 2.683228370706734e-12,
            2.2536437056243538e-12, 0.08825799527307868, 3.019885556355915e-12,
            2.9891579441813048e-12, 0.12071728737987927, 1.738949508186961e-11,
            7.133340530476957e-13, 1.6616048748672291e-12, 2.9573741726550625e-12,
            2.9380094661954363e-12, 2.23817801615805e-12, 3.0084424552985255e-12,
            0.1513731469723375, 0.5287160830386103, 1.6102691596939627e-12,
            0.11093548728874388, 5.863529750908513e-13]

    w15t = [4.904488573750729e-12, 2.071106728993754e-12, 4.800133024741897e-12,
            3.915793695045729e-12, 0.09178330614142587, 4.723084691031301e-12,
            4.837403928921659e-12, 0.0009767817347373915, 6.496610443683043e-11,
            2.3988177382041532e-12, 3.893853552308335e-12, 5.039285982466147e-12,
            4.729786847833998e-12, 4.45452517438696e-12, 4.8980359771502755e-12,
            0.14141381903028388, 0.5743520466758162, 3.98631737830706e-12,
            0.19147404629527873, 2.839376537835085e-12]

    w16t = [7.258550303162134e-10, 3.510882669681817e-9, 6.950239505552793e-10,
            1.0609124257825676e-9, 0.08771778791840867, 2.0738014860049013e-10,
            2.8312640337460793e-10, 0.12008375612593049, 2.032593825607953e-8,
            3.021490433133468e-9, 9.570405521537858e-9, 5.380706152519343e-10,
            2.3209931783627336e-10, 2.103112203820464e-9, 2.516769264462635e-10,
            0.15120889278610558, 0.5292690063020407, 1.6500103551629395e-8,
            0.11172049348167974, 4.359757243300342e-9]

    w17t = [1.6216819060655524e-13, 3.602736795125626e-12, 2.002307574155093e-13,
            7.66888202319162e-14, 0.0881713453630755, 5.181892735823799e-13,
            4.796904823523065e-13, 0.1206158001292373, 8.808315999173303e-12,
            2.284474458865395e-12, 2.9537581692501615e-9, 2.811308300914961e-13,
            5.282854348160106e-13, 2.1270364701539336e-11, 5.105401816811959e-13,
            0.15134681025157934, 0.5288047384122242, 7.567913219398542e-10,
            0.1110613018671969, 2.2741444375492125e-10]

    w18t = [6.017964139448665e-13, 4.162833539796467e-12, 6.255737396544022e-13,
            3.6934545672165233e-13, 0.09161103798114185, 1.5396470048295767e-12,
            1.376237525199013e-12, 4.413917662714404e-8, 0.0016588467476908635,
            2.3365692904110845e-12, 8.101137999978144e-10, 8.55247252035555e-13,
            1.5006222484359331e-12, 2.524735133715067e-11, 1.4530377927239903e-12,
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
    @test isapprox(w8.weights, w8t, rtol = 1.0e-7)
    @test isapprox(w9.weights, w9t, rtol = 0.0001)
    @test isapprox(w8.weights, w9.weights, rtol = 1e0)
    @test isapprox(w10.weights, w10t)
    @test isapprox(w11.weights, w11t)
    @test isapprox(w12.weights, w12t, rtol = 1.0e-6)
    @test isapprox(w11.weights, w12.weights, rtol = 1e-1)
    @test isapprox(w13.weights, w7.weights, rtol = 1e-5)
    @test isapprox(w14.weights, w8.weights, rtol = 1e-2)
    @test isapprox(w15.weights, w9.weights, rtol = 1e-1)
    @test isapprox(w16.weights, w7.weights, rtol = 1e-6)
    @test isapprox(w17.weights, w8.weights, rtol = 1e-2)
    @test isapprox(w18.weights, w9.weights, rtol = 1e-1)
    @test isapprox(w13.weights, w16.weights, rtol = 1e-5)
    @test isapprox(w14.weights, w17.weights, rtol = 1e-3)
    @test isapprox(w15.weights, w18.weights, rtol = 1e-2)
    @test isapprox(w19.weights, w1.weights, rtol = 1e0)
end

@testset "$(:Classic), $(:Trad), $(:TG)" begin
    portfolio = Portfolio(; prices = prices[(end - 200):end],
                          solvers = OrderedDict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                                  :params => Dict("verbose" => false,
                                                                                  "max_step_fraction" => 0.75))))
    asset_statistics!(portfolio)

    w1 = optimise!(portfolio,
                   OptimiseOpt(; owa_approx = false, rf = rf, l = l, class = :Classic,
                               type = :Trad, rm = :TG, obj = :Min_Risk, kelly = :None))
    risk1 = calc_risk(portfolio; type = :Trad, rm = :TG, rf = rf)
    w2 = optimise!(portfolio,
                   OptimiseOpt(; owa_approx = false, rf = rf, l = l, class = :Classic,
                               type = :Trad, rm = :TG, obj = :Min_Risk, kelly = :Approx))
    w3 = optimise!(portfolio,
                   OptimiseOpt(; owa_approx = false, rf = rf, l = l, class = :Classic,
                               type = :Trad, rm = :TG, obj = :Min_Risk, kelly = :Exact))
    w4 = optimise!(portfolio,
                   OptimiseOpt(; owa_approx = false, rf = rf, l = l, class = :Classic,
                               type = :Trad, rm = :TG, obj = :Utility, kelly = :None))
    w5 = optimise!(portfolio,
                   OptimiseOpt(; owa_approx = false, rf = rf, l = l, class = :Classic,
                               type = :Trad, rm = :TG, obj = :Utility, kelly = :Approx))
    w6 = optimise!(portfolio,
                   OptimiseOpt(; owa_approx = false, rf = rf, l = l, class = :Classic,
                               type = :Trad, rm = :TG, obj = :Utility, kelly = :Exact))
    w7 = optimise!(portfolio,
                   OptimiseOpt(; owa_approx = false, rf = rf, l = l, class = :Classic,
                               type = :Trad, rm = :TG, obj = :Sharpe, kelly = :None))
    risk7 = calc_risk(portfolio; type = :Trad, rm = :TG, rf = rf)
    ret7 = dot(portfolio.mu, w7.weights)
    w8 = optimise!(portfolio,
                   OptimiseOpt(; owa_approx = false, rf = rf, l = l, class = :Classic,
                               type = :Trad, rm = :TG, obj = :Sharpe, kelly = :Approx))
    risk8 = calc_risk(portfolio; type = :Trad, rm = :TG, rf = rf)
    ret8 = dot(portfolio.mu, w8.weights)
    w9 = optimise!(portfolio,
                   OptimiseOpt(; owa_approx = false, rf = rf, l = l, class = :Classic,
                               type = :Trad, rm = :TG, obj = :Sharpe, kelly = :Exact))
    risk9 = calc_risk(portfolio; type = :Trad, rm = :TG, rf = rf)
    ret9 = dot(portfolio.mu, w9.weights)
    w10 = optimise!(portfolio,
                    OptimiseOpt(; owa_approx = false, rf = rf, l = l, class = :Classic,
                                type = :Trad, rm = :TG, obj = :Max_Ret, kelly = :None))
    w11 = optimise!(portfolio,
                    OptimiseOpt(; owa_approx = false, rf = rf, l = l, class = :Classic,
                                type = :Trad, rm = :TG, obj = :Max_Ret, kelly = :Approx))
    w12 = optimise!(portfolio,
                    OptimiseOpt(; owa_approx = false, rf = rf, l = l, class = :Classic,
                                type = :Trad, rm = :TG, obj = :Max_Ret, kelly = :Exact))
    setproperty!(portfolio, :mu_l, ret7)
    w13 = optimise!(portfolio,
                    OptimiseOpt(; owa_approx = false, rf = rf, l = l, class = :Classic,
                                type = :Trad, rm = :TG, obj = :Min_Risk, kelly = :None))
    setproperty!(portfolio, :mu_l, ret8)
    w14 = optimise!(portfolio,
                    OptimiseOpt(; owa_approx = false, rf = rf, l = l, class = :Classic,
                                type = :Trad, rm = :TG, obj = :Min_Risk, kelly = :None))
    setproperty!(portfolio, :mu_l, ret9)
    w15 = optimise!(portfolio,
                    OptimiseOpt(; owa_approx = false, rf = rf, l = l, class = :Classic,
                                type = :Trad, rm = :TG, obj = :Min_Risk, kelly = :None))
    setproperty!(portfolio, :mu_l, Inf)
    rmf = Symbol(lowercase(string(:TG)) * "_u")
    setproperty!(portfolio, rmf, risk7)
    w16 = optimise!(portfolio,
                    OptimiseOpt(; owa_approx = false, rf = rf, l = l, class = :Classic,
                                type = :Trad, rm = :TG, obj = :Max_Ret, kelly = :None))
    setproperty!(portfolio, rmf, risk8)
    w17 = optimise!(portfolio,
                    OptimiseOpt(; owa_approx = false, rf = rf, l = l, class = :Classic,
                                type = :Trad, rm = :TG, obj = :Max_Ret, kelly = :None))
    setproperty!(portfolio, rmf, risk9)
    w18 = optimise!(portfolio,
                    OptimiseOpt(; owa_approx = false, rf = rf, l = l, class = :Classic,
                                type = :Trad, rm = :TG, obj = :Max_Ret, kelly = :None))
    setproperty!(portfolio, rmf, risk1)
    w19 = optimise!(portfolio,
                    OptimiseOpt(; owa_approx = false, rf = rf, l = l, class = :Classic,
                                type = :Trad, rm = :TG, obj = :Sharpe, kelly = :None))
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
           8.884714671260909e-12, 1.3757061851213529e-11, 1.6291810113089304e-13,
           0.1432756744154531, 2.0775280711374397e-11, 0.14179358412385565,
           4.251506235430576e-12, 0.2314876864861804]

    w5t = [7.876741054601859e-13, 0.22386204520499708, 3.551136120852785e-12,
           0.062473602045942406, 2.681412014274187e-10, 1.6398169690036202e-12,
           2.8836268962671105e-12, 0.11120554463921713, 9.69896654536334e-13,
           2.037905902633807e-11, 3.525244052955003e-10, 0.09247718814867852,
           3.1574898462203464e-12, 5.739501373413488e-12, 1.0045259045269914e-13,
           0.1367242024665253, 6.543734045634463e-12, 0.13899887173200867,
           1.5190248956445046e-12, 0.23425854509469402]

    w6t = [1.2941247551525297e-11, 0.22386925954250705, 3.2824653702191277e-11,
           0.0625065786029998, 1.3620372143823865e-9, 8.717706191359317e-12,
           3.1852031182937095e-12, 0.11119308337310815, 1.2968996068172452e-11,
           1.0934304821426853e-10, 1.2800883116924593e-9, 0.09246921064626823,
           1.969493077754324e-12, 4.049153686277518e-11, 1.65539530029089e-11,
           0.13670452814172337, 5.3419099417948596e-11, 0.138990478667524,
           2.4600231942469032e-11, 0.23426685806672873]

    w7t = [1.034203540495142e-11, 1.4007889383681966e-11, 1.0025731760906318e-11,
           6.108387515365813e-12, 0.3572352904863753, 1.2069497838089613e-11,
           1.2830319739722155e-11, 1.1788800032033873e-12, 6.535335074248242e-12,
           2.8948639761902753e-12, 9.118481022316377e-12, 1.0966191842706305e-11,
           1.3147750680814202e-11, 9.61174753780057e-12, 1.2632463845017768e-11,
           0.21190800224252684, 0.43085670711282753, 7.711974266816198e-12,
           1.2413946392925463e-11, 6.674972177548544e-12]

    w8t = [1.4718118126007435e-11, 1.2474877074375374e-10, 1.580793386862312e-11,
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
            1.8752566177088636e-12, 1.7825960548207287e-11, 9.096345201998647e-11,
            1.8124063387139953e-11, 3.8501508467546374e-12, 2.674046232448817e-13,
            3.994147030923467e-12, 3.3407504614825583e-12, 2.047533640542789e-12,
            1.8501728724970428e-10, 0.14222626767329194, 8.059325025309758e-12,
            5.637848418414877e-11, 4.6056332303540835e-12]

    w12t = [5.2804930141649576e-9, 9.416211338192126e-9, 5.520001335041304e-9,
            1.2389747433355727e-8, 0.8926328276594367, 1.7366619884827056e-9,
            2.36142330144021e-9, 9.133777915513395e-9, 4.105626060606478e-8,
            8.913137607797025e-9, 4.421552283984949e-9, 3.208536147187476e-9,
            1.8359894346800377e-9, 4.224879347145791e-9, 2.2684200090728395e-9,
            7.336516777793194e-8, 0.10736695198679332, 5.780217003330638e-9,
            2.4850102158041525e-8, 4.591191059247631e-9]

    w13t = [3.4396140094111085e-12, 5.702331297877075e-12, 3.3850739092860734e-12,
            2.1449219455862455e-12, 0.3572352911645771, 4.158207199076786e-12,
            4.030094976593963e-12, 1.6243295230400244e-12, 2.463859780243482e-12,
            1.0962514895832545e-12, 2.7142265167093685e-12, 3.2469604150886593e-12,
            4.124825680360781e-12, 2.672732115046088e-12, 3.929789757857018e-12,
            0.21190800498277654, 0.43085670379690205, 2.433400486793345e-12,
            7.374176308416663e-12, 1.2035320830453566e-12]

    w14t = [3.257745229963087e-12, 1.0130827408116335e-8, 3.192673279104055e-12,
            2.036266055028359e-12, 0.32204005948679654, 3.999534104117489e-12,
            3.922912154529824e-12, 8.565478472093807e-12, 3.382005532265066e-12,
            1.6154896835195683e-13, 1.8633120777140204e-12, 2.9655484946338515e-12,
            3.931677496001916e-12, 2.173876526701325e-12, 3.813058190024838e-12,
            0.2066980124008367, 0.47126191792157196, 1.6331021369502729e-12,
            1.339801780572528e-11, 1.670664150519693e-12]

    w15t = [2.7465751160880876e-12, 2.1177644537355916e-9, 2.6919131619233422e-12,
            1.7754153207760756e-12, 0.32919375360177344, 3.3653950108839433e-12,
            3.2914905879479745e-12, 5.5100400491825204e-12, 2.2748945420127287e-12,
            3.8881028270373756e-13, 1.6959554376446352e-12, 2.550772718640682e-12,
            3.3032659915241093e-12, 1.945487563770423e-12, 3.2085635894749945e-12,
            0.20775698021279215, 0.4630492640217384, 1.4889310873384205e-12,
            8.998414125961183e-12, 6.955715851970014e-13]

    w16t = [9.979862067124343e-15, 4.203228929046188e-11, 8.980440460800805e-15,
            6.446959689097406e-15, 0.3572352914112592, 2.0781543002891548e-14,
            1.7246066770888985e-14, 8.153114867404073e-13, 5.611409506218815e-14,
            1.7847952880714525e-14, 4.2082544141755004e-15, 5.40968849423465e-15,
            1.9828797974959275e-14, 4.947950070792405e-15, 1.7773371270726178e-14,
            0.21190800509326468, 0.4308567034510944, 1.0447138423251913e-14,
            1.0125147915616658e-13, 1.232779785619679e-12]

    w17t = [3.678618986325083e-12, 2.442450604648056e-7, 1.477872015468837e-12,
            2.3580280414647715e-11, 0.32204025141683756, 9.496261076848765e-12,
            1.0012058508959932e-11, 1.7942676120279416e-8, 2.097857337390307e-11,
            3.6099140941494994e-11, 1.0178666530460167e-10, 7.971441610381944e-11,
            1.0205219097098085e-11, 1.5041278767336882e-11, 7.206176859525398e-12,
            0.20669805829008434, 0.4712614049716827, 1.158255427666937e-10,
            6.363851362161488e-11, 2.2634918251252982e-8]

    w18t = [5.22559179306702e-12, 1.949910286726206e-7, 3.3288860762198112e-12,
            1.891245018197559e-11, 0.3293096665369873, 1.2616124768857992e-11,
            1.2641987148578779e-11, 1.089540208970257e-8, 3.6235616774174153e-11,
            3.447243519612428e-11, 7.278589141121017e-11, 5.3221882905407935e-11,
            1.2736246057904183e-11, 1.1554060860433626e-11, 1.0265589268338373e-11,
            0.2077741546167127, 0.46291595767439997, 8.501954099643547e-11,
            9.753022929512629e-11, 1.4818922832047022e-8]

    w19t = [7.55657374634104e-12, 0.19891044921795567, 2.491805468101704e-11,
            0.054902604651253704, 1.5885531756094824e-9, 1.1863426477811066e-11,
            2.186647458232334e-11, 0.10667546168194031, 1.054290697369371e-11,
            1.326199568511875e-10, 0.04866926165556854, 0.09089466315035939,
            2.3725316835207252e-11, 4.042940031217252e-11, 1.3879617478706681e-12,
            0.13418338454436574, 2.2390768113754715e-11, 0.1382963370480211,
            3.4856870773993155e-12, 0.22746783616119595]

    @test isapprox(w1.weights, w1t)
    @test isapprox(w2.weights, w2t)
    @test isapprox(w3.weights, w3t)
    @test isapprox(w2.weights, w3.weights, rtol = 1e-6)
    @test isapprox(w4.weights, w4t)
    @test isapprox(w5.weights, w5t)
    @test isapprox(w6.weights, w6t, rtol = 5.0e-6)
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
    @test isapprox(w17.weights, w8.weights, rtol = 1e-5)
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
                                                                                  "max_step_fraction" => 0.75))))
    asset_statistics!(portfolio)

    w1 = optimise!(portfolio,
                   OptimiseOpt(; owa_approx = false, rf = rf, l = l, class = :Classic,
                               type = :Trad, rm = :RTG, obj = :Min_Risk, kelly = :None))
    risk1 = calc_risk(portfolio; type = :Trad, rm = :RTG, rf = rf)
    w2 = optimise!(portfolio,
                   OptimiseOpt(; owa_approx = false, rf = rf, l = l, class = :Classic,
                               type = :Trad, rm = :RTG, obj = :Min_Risk, kelly = :Approx))
    w3 = optimise!(portfolio,
                   OptimiseOpt(; owa_approx = false, rf = rf, l = l, class = :Classic,
                               type = :Trad, rm = :RTG, obj = :Min_Risk, kelly = :Exact))
    w4 = optimise!(portfolio,
                   OptimiseOpt(; owa_approx = false, rf = rf, l = l, class = :Classic,
                               type = :Trad, rm = :RTG, obj = :Utility, kelly = :None))
    w5 = optimise!(portfolio,
                   OptimiseOpt(; owa_approx = false, rf = rf, l = l, class = :Classic,
                               type = :Trad, rm = :RTG, obj = :Utility, kelly = :Approx))
    w6 = optimise!(portfolio,
                   OptimiseOpt(; owa_approx = false, rf = rf, l = l, class = :Classic,
                               type = :Trad, rm = :RTG, obj = :Utility, kelly = :Exact))
    w7 = optimise!(portfolio,
                   OptimiseOpt(; owa_approx = false, rf = rf, l = l, class = :Classic,
                               type = :Trad, rm = :RTG, obj = :Sharpe, kelly = :None))
    risk7 = calc_risk(portfolio; type = :Trad, rm = :RTG, rf = rf)
    ret7 = dot(portfolio.mu, w7.weights)
    w8 = optimise!(portfolio,
                   OptimiseOpt(; owa_approx = false, rf = rf, l = l, class = :Classic,
                               type = :Trad, rm = :RTG, obj = :Sharpe, kelly = :Approx))
    risk8 = calc_risk(portfolio; type = :Trad, rm = :RTG, rf = rf)
    ret8 = dot(portfolio.mu, w8.weights)
    w9 = optimise!(portfolio,
                   OptimiseOpt(; owa_approx = false, rf = rf, l = l, class = :Classic,
                               type = :Trad, rm = :RTG, obj = :Sharpe, kelly = :Exact))
    risk9 = calc_risk(portfolio; type = :Trad, rm = :RTG, rf = rf)
    ret9 = dot(portfolio.mu, w9.weights)
    w10 = optimise!(portfolio,
                    OptimiseOpt(; owa_approx = false, rf = rf, l = l, class = :Classic,
                                type = :Trad, rm = :RTG, obj = :Max_Ret, kelly = :None))
    w11 = optimise!(portfolio,
                    OptimiseOpt(; owa_approx = false, rf = rf, l = l, class = :Classic,
                                type = :Trad, rm = :RTG, obj = :Max_Ret, kelly = :Approx))
    w12 = optimise!(portfolio,
                    OptimiseOpt(; owa_approx = false, rf = rf, l = l, class = :Classic,
                                type = :Trad, rm = :RTG, obj = :Max_Ret, kelly = :Exact))
    setproperty!(portfolio, :mu_l, ret7)
    w13 = optimise!(portfolio,
                    OptimiseOpt(; owa_approx = false, rf = rf, l = l, class = :Classic,
                                type = :Trad, rm = :RTG, obj = :Min_Risk, kelly = :None))
    setproperty!(portfolio, :mu_l, ret8)
    w14 = optimise!(portfolio,
                    OptimiseOpt(; owa_approx = false, rf = rf, l = l, class = :Classic,
                                type = :Trad, rm = :RTG, obj = :Min_Risk, kelly = :None))
    setproperty!(portfolio, :mu_l, ret9)
    w15 = optimise!(portfolio,
                    OptimiseOpt(; owa_approx = false, rf = rf, l = l, class = :Classic,
                                type = :Trad, rm = :RTG, obj = :Min_Risk, kelly = :None))
    setproperty!(portfolio, :mu_l, Inf)
    rmf = Symbol(lowercase(string(:RTG)) * "_u")
    setproperty!(portfolio, rmf, risk7)
    w16 = optimise!(portfolio,
                    OptimiseOpt(; owa_approx = false, rf = rf, l = l, class = :Classic,
                                type = :Trad, rm = :RTG, obj = :Max_Ret, kelly = :None))
    setproperty!(portfolio, rmf, risk8)
    w17 = optimise!(portfolio,
                    OptimiseOpt(; owa_approx = false, rf = rf, l = l, class = :Classic,
                                type = :Trad, rm = :RTG, obj = :Max_Ret, kelly = :None))
    setproperty!(portfolio, rmf, risk9)
    w18 = optimise!(portfolio,
                    OptimiseOpt(; owa_approx = false, rf = rf, l = l, class = :Classic,
                                type = :Trad, rm = :RTG, obj = :Max_Ret, kelly = :None))
    setproperty!(portfolio, rmf, risk1)
    w19 = optimise!(portfolio,
                    OptimiseOpt(; owa_approx = false, rf = rf, l = l, class = :Classic,
                                type = :Trad, rm = :RTG, obj = :Sharpe, kelly = :None))
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

    w6t = [3.1489679903033217e-12, 0.07691055111611911, 4.92763990515081e-12,
           6.50181306011904e-12, 0.01873630823951751, 0.02727673625141555,
           3.0360696746111825e-12, 0.06300036315524085, 2.3145853146646083e-12,
           2.8909858946755216e-12, 0.23138900986462824, 5.9436082507459575e-12,
           2.673210160676305e-13, 0.12343692042276387, 1.7836189693367114e-12,
           0.04767269068668512, 3.630207022299048e-11, 0.27434118640442534,
           5.162613006794215e-12, 0.13723623378692507]

    w7t = [4.865125156825178e-11, 2.1052614079822636e-11, 4.7365907624153683e-11,
           4.001712405491339e-11, 0.1416419921188952, 5.572004917204688e-11,
           5.631795517134003e-11, 0.024937870196849492, 1.79971814557232e-10,
           2.6621313364886448e-11, 4.154443093866612e-11, 5.2544888997749557e-11,
           5.7811722520210246e-11, 4.517027037812361e-11, 5.695329181673672e-11,
           0.21453454572858413, 0.6188855836123038, 3.483330484227522e-11,
           7.540474114074475e-9, 3.8317437143151725e-11]

    w8t = [2.7581567308792713e-11, 9.305269420569393e-11, 2.8856463333280447e-11,
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

    w11t = [6.034984815892211e-12, 1.7868956473930124e-11, 7.467599094385243e-12,
            2.596243132845663e-11, 0.8577737320666471, 5.080228610064883e-12,
            1.8850929459354704e-12, 1.795690659535625e-11, 9.157707978562806e-11,
            1.825453605253934e-11, 3.8824404267478314e-12, 2.7453150914440054e-13,
            4.014309200232137e-12, 3.3700780411065767e-12, 2.0623302151245345e-12,
            1.862276586845158e-10, 0.1422262674719025, 8.121980053374418e-12,
            5.6764282879759227e-11, 4.6446850307836085e-12]

    w12t = [9.210272510062331e-9, 1.6438981633302013e-8, 9.660753455889674e-9,
            2.1637170360081433e-8, 0.8926306591100491, 2.988220387163655e-9,
            4.1084226648484844e-9, 1.598020232517557e-8, 7.12190753331888e-8,
            1.5648054768916137e-8, 7.717168334257566e-9, 5.593150315535965e-9,
            3.1898663719233132e-9, 7.3754559447192124e-9, 3.952111657171382e-9,
            1.2768673785766315e-7, 0.10736895706111364, 1.0104796499819663e-8,
            4.3293111299095845e-8, 8.025285499915455e-9]

    w13t = [2.595599936537348e-11, 1.651019365128729e-11, 2.5534866643685744e-11,
            2.2331322397629907e-11, 0.1416419980914485, 2.8547848491104438e-11,
            2.7956901030692e-11, 0.024937875012701664, 5.018205148061643e-11,
            1.8276516401964547e-11, 2.2228415745472723e-11, 2.6483256941600314e-11,
            2.852286792224843e-11, 2.2890050038139587e-11, 2.8047760729549417e-11,
            0.21453455083719497, 0.6188855733657549, 2.058371397242547e-11,
            2.3093663502351488e-9, 1.9481925188446284e-11]

    w14t = [9.511453385748187e-12, 5.158782452595663e-12, 9.355679577321975e-12,
            8.22983530798536e-12, 0.12110207620316887, 1.0596327432720495e-11,
            1.0374452970729877e-11, 0.12035458528085836, 1.3443395894894593e-11,
            6.173547869257024e-12, 7.423348583160891e-12, 9.811854273805244e-12,
            1.055058152032995e-11, 8.121094107257048e-12, 1.0423920490319705e-11,
            0.19438514114756852, 0.5641581939077709, 6.83787341129961e-12,
            3.3283477786307378e-9, 6.273481559993505e-12]

    w15t = [7.955590598945476e-12, 4.741151065966018e-12, 7.806242374200578e-12,
            6.7811623433498785e-12, 0.12905162388524555, 8.814266159476679e-12,
            8.60688199371375e-12, 0.05087945538764261, 1.2940951279389932e-11,
            5.360477650229075e-12, 6.6585498610750066e-12, 8.127137297753299e-12,
            8.80723832819855e-12, 6.888195928846777e-12, 8.630917689837923e-12,
            0.20914744600224697, 0.6109214742545982, 6.143790071406996e-12,
            3.563450885739955e-10, 5.658901090638695e-12]

    w16t = [2.8587696467818527e-13, 1.2467345240121674e-11, 2.2151480175468828e-13,
            2.9286845680027484e-13, 0.1416419802719567, 9.221202005185037e-13,
            8.149711440468206e-13, 0.024937901687576456, 1.078452322684367e-11,
            1.1220924972901308e-12, 3.174456033772842e-10, 4.819035187133998e-13,
            9.187561246840649e-13, 1.0508896919184018e-10, 8.691685150417825e-13,
            0.21453454552082565, 0.6188855710931238, 4.7710933265881826e-11,
            3.349885945516861e-10, 5.921021367708042e-10]

    w17t = [5.022809091179894e-12, 3.278883980116286e-10, 4.242900142157342e-12,
            5.228105269875839e-12, 0.12109989810844203, 1.850103245573456e-11,
            1.652832955852485e-11, 0.12029753058162629, 1.549175844600153e-10,
            2.6176720920853117e-11, 1.5556586266721366e-8, 8.796490585995089e-12,
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
            1.5181689396767334e-10, 0.060564045880682656, 1.7672406247051208e-10,
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
    @test isapprox(w6.weights, w6t)
    @test isapprox(w5.weights, w6.weights, rtol = 1e-5)
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
