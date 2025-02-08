using CSV, TimeSeries, DataFrames, StatsBase, Statistics, LinearAlgebra, Test, Clarabel,
      PortfolioOptimiser

assets_path = joinpath(@__DIR__, "assets/stock_prices.csv")
factors_path = joinpath(@__DIR__, "assets/factor_prices.csv")
prices = TimeArray(CSV.File(assets_path); timestamp = :date)
factors = TimeArray(CSV.File(factors_path); timestamp = :date)

rf = 1.0329^(1 / 252) - 1
l = 2.0

@testset "Portfolio Trad classes" begin
    portfolio = Portfolio(; prices = prices,
                          solvers = PortOptSolver(; name = :Clarabel,
                                                  solver = Clarabel.Optimizer,
                                                  check_sol = (; allow_local = true,
                                                               allow_almost = true),
                                                  params = Dict("verbose" => false)))

    asset_statistics!(portfolio)

    portfolio.bl_mu = portfolio.mu
    portfolio.bl_cov = portfolio.cov

    obj = MinRisk()
    w1 = optimise!(portfolio, Trad(; class = Classic()))
    class = BL(; type = 1)
    w2 = optimise!(portfolio, Trad(; class = class))
    class = BL(; type = 2)
    w3 = optimise!(portfolio, Trad(; class = class))
    @test isapprox(w1.weights, w2.weights)
    @test isapprox(w1.weights, w3.weights)

    obj = Sharpe(; rf = rf)
    w4 = optimise!(portfolio, Trad(; class = Classic()))
    class = BL(; type = 1)
    w5 = optimise!(portfolio, Trad(; class = class))
    class = BL(; type = 2)
    w6 = optimise!(portfolio, Trad(; class = class))
    @test isapprox(w4.weights, w5.weights)
    @test isapprox(w4.weights, w6.weights)

    portfolio.fm_mu = portfolio.mu
    portfolio.fm_cov = portfolio.cov
    portfolio.fm_returns = portfolio.returns

    obj = MinRisk()
    w7 = optimise!(portfolio, Trad(; class = Classic()))
    class = FM(; type = 1)
    w8 = optimise!(portfolio, Trad(; class = class))
    class = FM(; type = 2)
    w9 = optimise!(portfolio, Trad(; class = class))
    @test isapprox(w7.weights, w8.weights)
    @test isapprox(w7.weights, w9.weights)

    obj = Sharpe(; rf = rf)
    w10 = optimise!(portfolio, Trad(; class = Classic()))
    class = FM(; type = 1)
    w11 = optimise!(portfolio, Trad(; class = class))
    class = FM(; type = 2)
    w12 = optimise!(portfolio, Trad(; class = class))
    @test isapprox(w10.weights, w11.weights)
    @test isapprox(w10.weights, w12.weights)

    portfolio.blfm_mu = portfolio.mu
    portfolio.blfm_cov = portfolio.cov

    obj = MinRisk()
    w13 = optimise!(portfolio, Trad(; class = Classic()))
    class = BLFM(; type = 1)
    w14 = optimise!(portfolio, Trad(; class = class))
    class = BLFM(; type = 2)
    w15 = optimise!(portfolio, Trad(; class = class))
    class = BLFM(; type = 3)
    w16 = optimise!(portfolio, Trad(; class = class))
    @test isapprox(w13.weights, w14.weights)
    @test isapprox(w13.weights, w15.weights)
    @test isapprox(w13.weights, w16.weights)

    obj = Sharpe(; rf = rf)
    w17 = optimise!(portfolio, Trad(; class = Classic()))
    class = BLFM(; type = 1)
    w18 = optimise!(portfolio, Trad(; class = class))
    class = BLFM(; type = 2)
    w19 = optimise!(portfolio, Trad(; class = class))
    class = BLFM(; type = 3)
    w20 = optimise!(portfolio, Trad(; class = class))
    @test isapprox(w17.weights, w18.weights)
    @test isapprox(w17.weights, w19.weights)
    @test isapprox(w17.weights, w20.weights)
end

@testset "Portfolio RB classes" begin
    portfolio = Portfolio(; prices = prices, f_prices = factors,
                          solvers = PortOptSolver(; name = :Clarabel,
                                                  solver = Clarabel.Optimizer,
                                                  check_sol = (; allow_local = true,
                                                               allow_almost = true),
                                                  params = Dict("verbose" => false,
                                                                "max_step_fraction" => 0.75)))

    asset_statistics!(portfolio)
    factor_statistics!(portfolio;
                       factor_type = FactorType(;
                                                type = PCAReg(;
                                                              target = PCATarget(;
                                                                                 kwargs = (;
                                                                                           pratio = 0.95)))))

    portfolio.short = true
    w1 = optimise!(portfolio, RB(; class = FC()))
    frc1 = factor_risk_contribution(portfolio, :RB)
    frc1_l, frc1_h = extrema(frc1[1:3])
    wt = [-0.21775727135097175, 0.267838585540792, 0.22580781962994076, 0.13008899739167523,
          -0.27877115882781617, -0.33139043626001263, -0.05165584583737949,
          0.13461824474865575, -0.3735804844140108, -0.9529846280913601, 0.3065924339842286,
          -0.09511483993695352, -0.1549592508550557, 0.23633506590609307,
          0.2981846144310902, 0.12937548510518276, 0.34093250561536237, 0.3861636853250784,
          0.3057194948009992, 0.6945569830944616]
    frct = [0.00028179302976080956, 0.00028176957440110553, 0.00028179342571403067,
            3.5583246627616816e-14]
    @test isapprox(w1.weights, wt; rtol = 5e-5)
    @test isapprox(frc1, frct, rtol = 1.0e-4)
    @test isapprox(frc1_h / frc1_l, 1, rtol = 0.0005)

    portfolio.f_risk_budget = 1:3
    w2 = optimise!(portfolio, RB(; class = FC()))
    frc2 = factor_risk_contribution(portfolio, :RB)
    frc2_l, frc2_h = extrema(frc2[1:3])
    wt = [-0.060386739304374173, 0.2023298205406452, 0.27318569546482635,
          0.14622836290146649, -0.303835220037279, -0.3214072601122666,
          -0.030083063614480467, 0.04697389149993996, -0.44740154575335267,
          -0.9617229031455009, 0.446757168225814, -0.07727608365256784,
          -0.14070790742287034, 0.1455290122867876, 0.258314932956942, 0.15309541530292148,
          0.07798353855573913, 0.5004482730703521, 0.37172196533236285, 0.7202526469048937]
    frct = [0.00013139593879266183, 0.0002627836837684036, 0.0003941803332018199,
            -2.4827716734937985e-14]
    @test isapprox(w2.weights, wt, rtol = 5.0e-5)
    @test isapprox(frc2, frct, rtol = 0.0001)
    @test isapprox(frc2_h / frc2_l, 3, rtol = 0.0001)

    portfolio.f_risk_budget = []
    w3 = optimise!(portfolio, RB(; class = FC(false)))
    frc3 = factor_risk_contribution(portfolio, :RB)
    frc3_l, frc3_h = extrema(frc3[1:3])
    wt = [0.08170925623690223, 0.10455256562057738, 0.12404101030136445,
          0.14432452511356408, 0.04731376886051586, -0.0076143489186165534,
          0.027007618319482684, 0.05771838540012703, -0.03808478102921201,
          -0.2187339997550548, 0.10862332618463623, 0.006138589444627486,
          -0.2976440236043342, 0.08817318632356322, 0.2962661696660294, 0.11069001473074284,
          0.08049007724311617, 0.11163960809196481, 0.028974237886975698,
          0.1444148138830274]
    frct = [0.00020822470113906105, 0.00020821744054522453, 0.0002082380049283071,
            -2.754072633534196e-20]
    @test isapprox(w3.weights, wt, rtol = 5.0e-5)
    @test isapprox(frc3, frct, rtol = 1.0e-4)
    @test isapprox(frc3_h / frc3_l, 1, rtol = 1.0e-4)

    portfolio.f_risk_budget = 1:3
    w4 = optimise!(portfolio, RB(; class = FC(false)))
    frc4 = factor_risk_contribution(portfolio, :RB)
    frc4_l, frc4_h = extrema(frc4[1:3])
    wt = [0.1290819501060403, 0.09077164796746803, 0.1620757594996783, 0.1688745651621904,
          0.04946219709713012, -0.00918736247644377, 0.05732417003612202,
          0.045024919014252436, -0.07688815882011989, -0.25765292367695425,
          0.14184893672430976, 0.030222457803378677, -0.29937346008181615,
          0.06365774751567693, 0.21859306734833434, 0.13235175600851004,
          0.03606036882994791, 0.14254006442897596, 0.008853420198605088,
          0.1663588773147133]
    frct = [0.0001006946863819786, 0.0002013881130711076, 0.0003020885947592278,
            -2.78584310746006e-20]
    @test isapprox(w4.weights, wt, rtol = 5.0e-5)
    @test isapprox(frc4, frct, rtol = 0.0001)
    @test isapprox(frc4_h / frc4_l, 3, rtol = 5.0e-4)

    factor_statistics!(portfolio; factor_type = FactorType(; type = BReg()))
    w5 = optimise!(portfolio, RB(; class = FC()))
    frc5 = factor_risk_contribution(portfolio, :RB)
    frc5_l, frc5_h = extrema(frc5[1:5])
    wt = [-2.3015557979838155, 1.015632612233884, 1.4947894574786151, -0.5829581771148582,
          -0.46358313943370977, -0.07532758341121848, -0.5191395483112032,
          1.1904680595031902, -0.4105315628779078, -0.8833258301042758, -0.9185414083981849,
          -0.8999018752834927, -0.6700017068291841, 1.0150795925421592, 0.8242328200110127,
          0.49533510792764995, 2.835029907168694, -0.8499295716783859, 0.9995078653621676,
          -0.2952792208011381]
    frct = [0.0017566137529628331, 0.0017565784749225815, 0.0017565435848182192,
            0.0017563104607473269, 0.0017564454358236965, -3.781663218703131e-13]
    @test isapprox(w5.weights, wt, rtol = 5.0e-5)
    @test isapprox(frc5, frct, rtol = 0.0001)
    @test isapprox(frc5_h / frc5_l, 1, rtol = 0.0005)

    portfolio.f_risk_budget = 1:5
    w6 = optimise!(portfolio, RB(; class = FC()))
    frc6 = factor_risk_contribution(portfolio, :RB)
    frc6_l, frc6_h = extrema(frc6[1:5])
    wt = [-2.2578187217827628, 0.9179169624221084, 1.203465037545498, -0.36341097260129385,
          -0.43200523630742893, -0.5355634581626013, -0.4610505223031706, 1.089644973934692,
          -0.139101166606894, -0.6979495331662376, -0.791296745855257, -0.766139247758445,
          -0.5799578737345131, 1.0421239982507846, 0.6556645399439347, 0.3204082187316696,
          2.6064091841203694, -0.8933923431775181, 1.1777896422706118, -0.09573673576354759]
    frct = [0.00047270623387185855, 0.0009453745948397381, 0.0014180418088610214,
            0.0018903394856425782, 0.0023631033351667967, -9.550342529756068e-13]
    @test isapprox(w6.weights, wt, rtol = 5.0e-5)
    @test isapprox(frc6, frct, rtol = 0.0001)
    @test isapprox(frc6_h / frc6_l, 5, rtol = 0.0005)

    portfolio.f_risk_budget = []
    w7 = optimise!(portfolio, RB(; class = FC(false)))
    frc7 = factor_risk_contribution(portfolio, :RB)
    frc7_l, frc7_h = extrema(frc7[1:5])
    wt = [-0.3004127266568237, 0.4837127843439659, 0.15400712675524886, 0.37928131658434344,
          -0.11093636742715317, -0.019227147416553236, -0.7239019664321605,
          0.3023582570333168, 0.4543035226073654, 0.014682326977849769,
          -0.037435652744221726, -0.6759655256126349, -1.5070329235093687,
          0.5341445413096939, 0.8826395761710898, 0.19713651658098424, 0.6968875983538265,
          -0.07513469093810905, 0.4181682213818354, -0.06727478736249529]
    frct = [0.0023003469349628385, 0.0023005265140464857, 0.002300340214825507,
            0.0023003822099879312, 0.0023003617938229634, -7.696361878734756e-19]
    @test isapprox(w7.weights, wt, rtol = 1.0e-4)
    @test isapprox(frc7, frct, rtol = 0.0005)
    @test isapprox(frc7_h / frc7_l, 1, rtol = 0.0005)

    portfolio.f_risk_budget = 1:5
    w8 = optimise!(portfolio, RB(; class = FC(false)))
    frc8 = factor_risk_contribution(portfolio, :RB)
    frc8_l, frc8_h = extrema(frc8[1:5])
    wt = [-0.17748034020963993, 0.28804865930885837, 0.07839868288225765,
          0.23226186278725133, -0.04543783296007117, -0.14449206629982284,
          -0.29649913182717624, 0.18005290202848567, 0.3577874247387566,
          0.08038791157573967, -0.035186052510478044, -0.2768651016063091,
          -0.8029814798116754, 0.31808053039175743, 0.4493154420154725, 0.10035407824020558,
          0.4149932457688494, -0.0588541309995302, 0.2948342911881, 0.04328110529896874]
    frct = [0.00021075657803768937, 0.0004215087944519139, 0.000632263808507996,
            0.0008429002349035797, 0.0010536625518982787, -2.0954666844811407e-19]
    @test isapprox(w8.weights, wt, rtol = 5.0e-5)
    @test isapprox(frc8, frct, rtol = 1.0e-4)
    @test isapprox(frc8_h / frc8_l, 5, rtol = 1e-3)

    portfolio.f_risk_budget = [0.2, 0.2, 0.2, 0.2, 0.2]
    portfolio.f_risk_budget[1] = 20
    portfolio.f_risk_budget /= sum(portfolio.f_risk_budget)
    w9 = optimise!(portfolio, RB(; class = FC()))
    rc9 = factor_risk_contribution(portfolio, :RB)
    lrc9, hrc9 = extrema(rc9[1:4])
    @test isapprox(hrc9 / lrc9, 100, rtol = 0.001)
    @test isapprox(sum(portfolio.f_risk_budget), 1)

    portfolio.loadings = DataFrame()
    rc10 = factor_risk_contribution(portfolio, :RB)
    @test isapprox(rc9, rc10)
end
