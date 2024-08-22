using CSV, TimeSeries, DataFrames, StatsBase, Statistics, LinearAlgebra, Test, Clarabel,
      PortfolioOptimiser

prices = TimeArray(CSV.File("./assets/stock_prices.csv"); timestamp = :date)
factors = TimeArray(CSV.File("./assets/factor_prices.csv"); timestamp = :date)

rf = 1.0329^(1 / 252) - 1
l = 2.0

@testset "Portfolio Trad classes" begin
    portfolio = Portfolio2(; prices = prices,
                           solvers = Dict(:Clarabel => Dict(:solver => (Clarabel.Optimizer),
                                                            :params => Dict("verbose" => false))))

    asset_statistics2!(portfolio)

    portfolio.bl_mu = portfolio.mu
    portfolio.bl_cov = portfolio.cov

    obj = MinRisk()
    w1 = optimise2!(portfolio; obj = obj, class = Classic2())
    class = BL2(; type = 1)
    w2 = optimise2!(portfolio; obj = obj, class = class)
    class = BL2(; type = 2)
    w3 = optimise2!(portfolio; obj = obj, class = class)
    @test isapprox(w1.weights, w2.weights)
    @test isapprox(w1.weights, w3.weights)

    obj = SR(; rf = rf)
    w4 = optimise2!(portfolio; obj = obj, class = Classic2())
    class = BL2(; type = 1)
    w5 = optimise2!(portfolio; obj = obj, class = class)
    class = BL2(; type = 2)
    w6 = optimise2!(portfolio; obj = obj, class = class)
    @test isapprox(w4.weights, w5.weights)
    @test isapprox(w4.weights, w6.weights)

    portfolio.fm_mu = portfolio.mu
    portfolio.fm_cov = portfolio.cov
    portfolio.fm_returns = portfolio.returns

    obj = MinRisk()
    w7 = optimise2!(portfolio; obj = obj, class = Classic2())
    class = FM2(; type = 1)
    w8 = optimise2!(portfolio; obj = obj, class = class)
    class = FM2(; type = 2)
    w9 = optimise2!(portfolio; obj = obj, class = class)
    @test isapprox(w7.weights, w8.weights)
    @test isapprox(w7.weights, w9.weights)

    obj = SR(; rf = rf)
    w10 = optimise2!(portfolio; obj = obj, class = Classic2())
    class = FM2(; type = 1)
    w11 = optimise2!(portfolio; obj = obj, class = class)
    class = FM2(; type = 2)
    w12 = optimise2!(portfolio; obj = obj, class = class)
    @test isapprox(w10.weights, w11.weights)
    @test isapprox(w10.weights, w12.weights)

    portfolio.blfm_mu = portfolio.mu
    portfolio.blfm_cov = portfolio.cov

    obj = MinRisk()
    w13 = optimise2!(portfolio; obj = obj, class = Classic2())
    class = BLFM2(; type = 1)
    w14 = optimise2!(portfolio; obj = obj, class = class)
    class = BLFM2(; type = 2)
    w15 = optimise2!(portfolio; obj = obj, class = class)
    class = BLFM2(; type = 3)
    w16 = optimise2!(portfolio; obj = obj, class = class)
    @test isapprox(w13.weights, w14.weights)
    @test isapprox(w13.weights, w15.weights)
    @test isapprox(w13.weights, w16.weights)

    obj = SR(; rf = rf)
    w17 = optimise2!(portfolio; obj = obj, class = Classic2())
    class = BLFM2(; type = 1)
    w18 = optimise2!(portfolio; obj = obj, class = class)
    class = BLFM2(; type = 2)
    w19 = optimise2!(portfolio; obj = obj, class = class)
    class = BLFM2(; type = 3)
    w20 = optimise2!(portfolio; obj = obj, class = class)
    @test isapprox(w17.weights, w18.weights)
    @test isapprox(w17.weights, w19.weights)
    @test isapprox(w17.weights, w20.weights)
end

@testset "Portfolio RP classes" begin
    portfolio = Portfolio2(; prices = prices, f_prices = factors,
                           solvers = Dict(:Clarabel => Dict(:solver => (Clarabel.Optimizer),
                                                            :params => Dict("verbose" => false,
                                                                            "max_step_fraction" => 0.75))))

    asset_statistics2!(portfolio)
    factor_statistics2!(portfolio;
                        factor_type = FactorType(;
                                                 method = DimensionReductionReg(;
                                                                                pcr = PCATarget(;
                                                                                                kwargs = (;
                                                                                                          pratio = 0.95)))))
    w1 = optimise2!(portfolio; type = RP2(), obj = obj, class = FC2())
    wt = [-0.21775727135097175, 0.267838585540792, 0.22580781962994076, 0.13008899739167523,
          -0.27877115882781617, -0.33139043626001263, -0.05165584583737949,
          0.13461824474865575, -0.3735804844140108, -0.9529846280913601, 0.3065924339842286,
          -0.09511483993695352, -0.1549592508550557, 0.23633506590609307,
          0.2981846144310902, 0.12937548510518276, 0.34093250561536237, 0.3861636853250784,
          0.3057194948009992, 0.6945569830944616]
    @test isapprox(w1.weights, wt; rtol = 1e-5)

    portfolio.f_risk_budget = 1:3
    w2 = optimise2!(portfolio; type = RP2(), obj = obj, class = FC2())
    wt = [-0.060386739304374173, 0.2023298205406452, 0.27318569546482635,
          0.14622836290146649, -0.303835220037279, -0.3214072601122666,
          -0.030083063614480467, 0.04697389149993996, -0.44740154575335267,
          -0.9617229031455009, 0.446757168225814, -0.07727608365256784,
          -0.14070790742287034, 0.1455290122867876, 0.258314932956942, 0.15309541530292148,
          0.07798353855573913, 0.5004482730703521, 0.37172196533236285, 0.7202526469048937]
    @test isapprox(w2.weights, wt, rtol = 5.0e-5)

    portfolio.f_risk_budget = []
    w3 = optimise2!(portfolio; type = RP2(), obj = obj, class = FC2(false))
    wt = [0.08170925623690223, 0.10455256562057738, 0.12404101030136445,
          0.14432452511356408, 0.04731376886051586, -0.0076143489186165534,
          0.027007618319482684, 0.05771838540012703, -0.03808478102921201,
          -0.2187339997550548, 0.10862332618463623, 0.006138589444627486,
          -0.2976440236043342, 0.08817318632356322, 0.2962661696660294, 0.11069001473074284,
          0.08049007724311617, 0.11163960809196481, 0.028974237886975698,
          0.1444148138830274]
    @test isapprox(w3.weights, wt)

    portfolio.f_risk_budget = 1:3
    w4 = optimise2!(portfolio; type = RP2(), obj = obj, class = FC2(false))
    wt = [0.1290819501060403, 0.09077164796746803, 0.1620757594996783, 0.1688745651621904,
          0.04946219709713012, -0.00918736247644377, 0.05732417003612202,
          0.045024919014252436, -0.07688815882011989, -0.25765292367695425,
          0.14184893672430976, 0.030222457803378677, -0.29937346008181615,
          0.06365774751567693, 0.21859306734833434, 0.13235175600851004,
          0.03606036882994791, 0.14254006442897596, 0.008853420198605088,
          0.1663588773147133]
    @test isapprox(w4.weights, wt)

    factor_statistics2!(portfolio; factor_type = FactorType(; method = BackwardReg()))
    w5 = optimise2!(portfolio; type = RP2(), obj = obj, class = FC2())
    wt = [-2.3015557979838155, 1.015632612233884, 1.4947894574786151, -0.5829581771148582,
          -0.46358313943370977, -0.07532758341121848, -0.5191395483112032,
          1.1904680595031902, -0.4105315628779078, -0.8833258301042758, -0.9185414083981849,
          -0.8999018752834927, -0.6700017068291841, 1.0150795925421592, 0.8242328200110127,
          0.49533510792764995, 2.835029907168694, -0.8499295716783859, 0.9995078653621676,
          -0.2952792208011381]
    @test isapprox(w5.weights, wt, rtol = 5.0e-5)

    portfolio.f_risk_budget = 1:5
    w6 = optimise2!(portfolio; type = RP2(), obj = obj, class = FC2())
    wt = [-2.2578187217827628, 0.9179169624221084, 1.203465037545498, -0.36341097260129385,
          -0.43200523630742893, -0.5355634581626013, -0.4610505223031706, 1.089644973934692,
          -0.139101166606894, -0.6979495331662376, -0.791296745855257, -0.766139247758445,
          -0.5799578737345131, 1.0421239982507846, 0.6556645399439347, 0.3204082187316696,
          2.6064091841203694, -0.8933923431775181, 1.1777896422706118, -0.09573673576354759]
    @test isapprox(w6.weights, wt, rtol = 5.0e-5)

    portfolio.f_risk_budget = []
    w7 = optimise2!(portfolio; type = RP2(), obj = obj, class = FC2(false))
    wt = [-0.3004127266568237, 0.4837127843439659, 0.15400712675524886, 0.37928131658434344,
          -0.11093636742715317, -0.019227147416553236, -0.7239019664321605,
          0.3023582570333168, 0.4543035226073654, 0.014682326977849769,
          -0.037435652744221726, -0.6759655256126349, -1.5070329235093687,
          0.5341445413096939, 0.8826395761710898, 0.19713651658098424, 0.6968875983538265,
          -0.07513469093810905, 0.4181682213818354, -0.06727478736249529]
    @test isapprox(w7.weights, wt, rtol = 5.0e-5)

    portfolio.f_risk_budget = 1:5
    w8 = optimise2!(portfolio; type = RP2(), obj = obj, class = FC2(false))
    wt = [-0.17747736716970441, 0.28804797250825886, 0.07839477443617672,
          0.2322728530398622, -0.04543617818471239, -0.14449404709946034,
          -0.29648833378894945, 0.18005247272448766, 0.35778751558889327,
          0.0804013508328198, -0.0351935915138625, -0.2768550186089918, -0.802962082939696,
          0.31807977198552323, 0.4492930420833068, 0.10034907524156604, 0.4149922562915483,
          -0.05886306962212894, 0.29482394535061346, 0.04327465884444936]
    @test isapprox(w8.weights, wt, rtol = 5.0e-5)
end
