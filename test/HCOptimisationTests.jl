using CSV, TimeSeries, DataFrames, StatsBase, Statistics, LinearAlgebra, Test, Clarabel,
      PortfolioOptimiser

prices = TimeArray(CSV.File("./assets/stock_prices.csv"); timestamp = :date)
rf = 1.0329^(1 / 252) - 1
l = 2.0

@testset "HRP and HERC risk scale" begin
    portfolio = HCPortfolio(; prices = prices,
                            solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                             :params => Dict("verbose" => false,
                                                                             "max_step_fraction" => 0.75))))

    asset_statistics!(portfolio)

    w1 = optimise!(portfolio; rm = SD(), type = HRP(), cluster = true)
    w2 = optimise!(portfolio; rm = CDaR(), type = HRP(), cluster = false)
    w3 = optimise!(portfolio; rm = [SD(), CDaR()], type = HRP(), cluster = false)
    w4 = optimise!(portfolio;
                   rm = [SD(), CDaR(; settings = RiskMeasureSettings(; scale = 1e-10))],
                   type = HRP(), cluster = false)
    w5 = optimise!(portfolio;
                   rm = [SD(), CDaR(; settings = RiskMeasureSettings(; scale = 1e10))],
                   type = HRP(), cluster = false)

    w6 = optimise!(portfolio;
                   rm = [CDaR(), SD(; settings = RiskMeasureSettings(; scale = 1e-10))],
                   type = HRP(), cluster = false)
    w7 = optimise!(portfolio;
                   rm = [CDaR(), SD(; settings = RiskMeasureSettings(; scale = 1e10))],
                   type = HRP(), cluster = false)
    @test isapprox(w1.weights, w4.weights)
    @test isapprox(w2.weights, w5.weights)
    @test isapprox(w2.weights, w6.weights)
    @test isapprox(w1.weights, w7.weights)
    @test !isapprox(w1.weights, w3.weights)
    @test !isapprox(w2.weights, w3.weights)

    w1 = optimise!(portfolio; rm = SD(), type = HERC(), cluster = false)
    w2 = optimise!(portfolio; rm = CDaR(), type = HERC(), cluster = false)
    w3 = optimise!(portfolio; rm = [SD(), CDaR()], type = HERC(), cluster = false)
    w4 = optimise!(portfolio;
                   rm = [SD(), CDaR(; settings = RiskMeasureSettings(; scale = 1e-10))],
                   type = HERC(), cluster = false)
    w5 = optimise!(portfolio;
                   rm = [CDaR(), SD(; settings = RiskMeasureSettings(; scale = 1e-10))],
                   type = HERC(), cluster = false)
    @test isapprox(w1.weights, w4.weights)
    @test isapprox(w2.weights, w5.weights)
    @test !isapprox(w1.weights, w3.weights)
    @test !isapprox(w2.weights, w3.weights)
end

@testset "NCO vector rm" begin
    portfolio = HCPortfolio(; prices = prices,
                            solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                             :params => Dict("verbose" => false,
                                                                             "max_step_fraction" => 0.75))))

    asset_statistics!(portfolio)

    w1 = optimise!(portfolio; rm = SD(), type = NCO(), cluster = true)
    w2 = optimise!(portfolio; rm = [[SD(), SD()]], type = NCO(), cluster = false)
    w3 = optimise!(portfolio; rm = [SD(; settings = RiskMeasureSettings(; scale = 2))],
                   type = NCO(), cluster = false)
    @test isapprox(w1.weights, w2.weights, rtol = 5.0e-5)
    @test isapprox(w1.weights, w3.weights, rtol = 1.0e-5)
    @test isapprox(w2.weights, w3.weights, rtol = 1.0e-5)

    w4 = optimise!(portfolio; rm = [SD(), CDaR()], type = NCO(), cluster = true)
    w5 = optimise!(portfolio; rm = [[SD(), SD()], [CDaR(), CDaR()]], type = NCO(),
                   cluster = false)
    w6 = optimise!(portfolio;
                   rm = [SD(; settings = RiskMeasureSettings(; scale = 2)),
                         CDaR(; settings = RiskMeasureSettings(; scale = 2))], type = NCO(),
                   cluster = true)
    @test isapprox(w4.weights, w5.weights, rtol = 5.0e-5)
    @test isapprox(w4.weights, w6.weights, rtol = 5.0e-9)
    @test isapprox(w5.weights, w6.weights, rtol = 5.0e-5)
end

@testset "HERC and NCO mixed parameters" begin
    portfolio = HCPortfolio(; prices = prices,
                            solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                             :params => Dict("verbose" => false,
                                                                             "max_step_fraction" => 0.75))))

    asset_statistics!(portfolio)
    hclust_alg = DBHT(; root_method = EqualDBHT())
    hclust_opt = HCType()

    w1 = optimise!(portfolio; cluster = true, hclust_alg = hclust_alg,
                   hclust_opt = hclust_opt, rm = SD(), rmo = CDaR(),
                   type = NCO(; opt_kwargs = (; obj = Sharpe(; rf = rf), kelly = EKelly()),
                              opt_kwargs_o = (; obj = Utility(; l = 10 * l),
                                              kelly = NoKelly())))
    wt = [8.749535903078065e-9, 0.021318202697098224, 0.010058873436520996,
          0.0006391710067219409, 0.23847337816224912, 5.078733996478347e-9,
          0.03433819538761402, 0.00922817630772192, 3.9969791626882106e-8,
          2.3591257653925676e-8, 5.272610870475486e-8, 4.559863014598392e-9,
          2.7523314119460396e-9, 9.64460078901257e-9, 3.0307822675469274e-9,
          0.11722640236879024, 0.4120447177289777, 0.01690611807209831, 0.12415863804756225,
          0.015607976681640013]
    @test isapprox(w1.weights, wt, rtol = 5.0e-5)

    w2 = optimise!(portfolio; cluster = false, hclust_alg = hclust_alg,
                   hclust_opt = hclust_opt, rm = SD(), rmo = CDaR(),
                   type = NCO(; opt_kwargs = (; obj = Sharpe(; rf = rf), kelly = NoKelly()),
                              opt_kwargs_o = (; obj = Utility(; l = 10 * l),
                                              kelly = EKelly())))
    wt = [3.1657435876961315e-10, 0.011350048898726263, 0.005901925005901058,
          4.145858254573504e-9, 0.24703174384250007, 1.1554588711505922e-10,
          0.03571166800872377, 2.0093176507817576e-9, 1.1359578457903453e-9,
          5.740514558667733e-10, 9.252049263004662e-10, 1.1349813784657876e-10,
          6.974954102659066e-11, 2.3407404085962843e-10, 6.928260874063481e-11,
          0.1292316915165453, 0.44715804867316167, 2.5109012694019742e-9,
          0.12361485036662549, 1.1467800437648667e-8]
    @test isapprox(w2.weights, wt)

    w3 = optimise!(portfolio; cluster = false, hclust_alg = hclust_alg,
                   hclust_opt = hclust_opt, rm = SD(), rmo = CDaR(),
                   type = NCO(;
                              opt_kwargs = (; obj = Utility(; l = 10 * l),
                                            kelly = EKelly()),
                              opt_kwargs_o = (; obj = Sharpe(; rf = rf), kelly = NoKelly())))
    wt = [6.4791954821063925e-9, 0.029812930861426164, 0.010894696408080332,
          0.011896393137335998, 0.044384433675000466, 2.746543990901563e-9,
          0.005106858724009885, 0.06946926268324362, 3.8491579305047204e-9,
          0.02819781049675312, 0.28810639904355484, 9.707590311179332e-10,
          4.695579190317573e-10, 0.040824363280960715, 5.178841305932862e-10,
          0.049118688613542884, 0.20174937118162223, 0.10295500372922885,
          0.06140247037868661, 0.05608130275345569]
    @test isapprox(w3.weights, wt)

    w4 = optimise!(portfolio; cluster = false, hclust_alg = hclust_alg,
                   hclust_opt = hclust_opt, rm = SD(), rmo = CDaR(),
                   type = NCO(;
                              opt_kwargs = (; obj = Utility(; l = 10 * l),
                                            kelly = NoKelly()),
                              opt_kwargs_o = (; obj = Sharpe(; rf = rf), kelly = EKelly())))
    wt = [2.0554538223172535e-8, 0.029818986226667964, 0.010923608025930946,
          0.01186842576568594, 0.04541568191043348, 9.983988311413258e-9,
          0.005261250564448335, 0.06930507652058161, 2.1332962817997016e-8,
          0.027639989232562466, 0.2870831936646783, 2.7962815225575495e-10,
          3.641935755054462e-9, 0.03800725786958468, 3.3607876962123727e-9,
          0.049529929060475084, 0.2034415850017132, 0.10271915204925444,
          0.06299503778536726, 0.05599076716877515]
    @test isapprox(w4.weights, wt, rtol = 5.0e-5)
    @test !isapprox(w1.weights, w2.weights)
    @test !isapprox(w1.weights, w3.weights)
    @test !isapprox(w1.weights, w4.weights)
    @test !isapprox(w2.weights, w3.weights)
    @test !isapprox(w2.weights, w4.weights)
    @test !isapprox(w3.weights, w4.weights)

    hclust_alg = DBHT()
    w5 = optimise!(portfolio; cluster = true, hclust_alg = hclust_alg,
                   hclust_opt = hclust_opt, rmo = SD(), rm = CDaR(), type = HERC())
    wt = [0.10871059727246735, 0.05431039601849186, 0.10533650868181384,
          0.027317993835046576, 0.07431929926304212, 0.00954218610227609,
          0.024606833580412473, 0.04020099981391352, 0.022469670005659467,
          0.05391899731269113, 0.04317380104646033, 0.006286394179643389,
          0.006060907562898212, 0.0320291710414021, 0.005842729905950518,
          0.044283643115509565, 0.10749469436263087, 0.09602771642660826,
          0.04410905860746655, 0.09395840186561562]
    @test isapprox(w5.weights, wt)

    w6 = optimise!(portfolio; cluster = false, hclust_alg = hclust_alg,
                   hclust_opt = hclust_opt, rm = SD(), rmo = CDaR(), type = HERC())
    wt = [0.08320752059200986, 0.08290256524137433, 0.07517557492907619,
          0.06023885608558014, 0.06626202578072789, 0.024707098983435642,
          0.029699159972552684, 0.0942847206692912, 0.019894956146041556,
          0.02777710488606625, 0.03221349389416141, 0.01243771863240931,
          0.009827277935290812, 0.027588252827801342, 0.010488817689171098,
          0.0192723640402875, 0.09460246164880029, 0.10914949211014122,
          0.024315592933065365, 0.09595494500271598]
    @test isapprox(w6.weights, wt)
    @test !isapprox(w5.weights, w6.weights)
end

@testset "Weight bounds" begin
    portfolio = HCPortfolio(; prices = prices,
                            solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                             :params => Dict("verbose" => false,
                                                                             "max_step_fraction" => 0.75))))

    asset_statistics!(portfolio)
    hclust_alg = HAC()
    hclust_opt = HCType()

    asset_sets = DataFrame("Asset" => portfolio.assets,
                           "PDBHT" => [1, 2, 1, 1, 1, 3, 2, 2, 3, 3, 3, 4, 4, 3, 3, 4, 2, 2,
                                       3, 1],
                           "SPDBHT" => [1, 1, 1, 1, 1, 2, 3, 4, 2, 3, 3, 2, 3, 3, 3, 3, 1,
                                        4, 2, 1],
                           "Pward" => [1, 1, 1, 1, 1, 2, 3, 2, 2, 2, 2, 4, 4, 2, 3, 4, 1, 2,
                                       2, 1],
                           "SPward" => [1, 1, 1, 1, 1, 2, 2, 3, 2, 2, 2, 4, 3, 2, 2, 3, 1,
                                        2, 2, 1],
                           "G2DBHT" => [1, 2, 1, 1, 1, 3, 2, 3, 4, 3, 4, 3, 3, 4, 4, 3, 2,
                                        3, 4, 1],
                           "G2ward" => [1, 1, 1, 1, 1, 2, 3, 4, 2, 2, 4, 2, 3, 3, 3, 2, 1,
                                        4, 2, 2])
    constraints = DataFrame("Enabled" => [true, true, true, true, true, true, false],
                            "Type" => ["Asset", "Asset", "All Assets", "All Assets",
                                       "Each Asset in Subset", "Each Asset in Subset",
                                       "Asset"],
                            "Set" => ["", "", "", "", "PDBHT", "Pward", ""],
                            "Position" => ["WMT", "T", "", "", 3, 2, "AAPL"],
                            "Sign" => [">=", "<=", ">=", "<=", ">=", "<=", ">="],
                            "Weight" => [0.05, 0.04, 0.02, 0.07, 0.04, 0.08, 0.2])
    w_min, w_max = hrp_constraints(constraints, asset_sets)
    N = length(w_min)

    portfolio.w_min = w_min
    portfolio.w_max = w_max

    w1 = optimise!(portfolio; hclust_alg = hclust_alg, hclust_opt = hclust_opt,
                   type = HRP(), rm = CDaR(), cluster = true)
    wt = [0.06365058210843497, 0.053794323634285435, 0.061675036869342476,
          0.021736783382546202, 0.05913547381912344, 0.06999999999999998, 0.02, 0.07, 0.04,
          0.04, 0.04, 0.02, 0.02, 0.06999999999999999, 0.04, 0.060007800186267495, 0.07,
          0.07, 0.04, 0.07]
    @test isapprox(w1.weights, wt)
    @test all(abs.(w1.weights .- w_min) .>= -eps() * N)
    @test all(w1.weights .<= w_max)

    w2 = optimise!(portfolio; hclust_alg = hclust_alg, hclust_opt = hclust_opt,
                   type = HERC(), rm = CDaR(), cluster = false)
    wt = [0.07, 0.04776234586617294, 0.07, 0.024024340928672654, 0.06535883249174655, 0.04,
          0.033027702141039166, 0.05, 0.04, 0.04443141920819796, 0.04, 0.02, 0.02,
          0.04973697996776029, 0.04, 0.0684078172625897, 0.07, 0.0687551470785739,
          0.06849541505524677, 0.07]
    @test isapprox(w2.weights, wt)
    @test all(abs.(w2.weights .- w_min) .>= -eps() * N)
    @test all(w2.weights .<= w_max)

    w3 = optimise!(portfolio; hclust_alg = hclust_alg, hclust_opt = hclust_opt,
                   type = NCO(), rm = CDaR(), cluster = false)
    wt = [0.08235294117647059, 0.023529411764705882, 0.08235294117647059,
          0.023529411764705882, 0.023529411764705882, 0.047058823529411764,
          0.023529411764705882, 0.058823529411764705, 0.047058823529411764,
          0.047058823529411764, 0.047058823529411764, 0.023529411764705882,
          0.023529411764705882, 0.047058823529411764, 0.047058823529411764,
          0.023529411764705882, 0.08235294117647059, 0.08235294117647059,
          0.08235294117647059, 0.08235294117647059]
    @test isapprox(w3.weights, wt)
    @test all(w3.weights .>= w_min)
    @test !all(w3.weights .<= w_max)

    portfolio.w_min = 0.03
    portfolio.w_max = 0.07
    w4 = optimise!(portfolio; hclust_alg = hclust_alg, hclust_opt = hclust_opt,
                   type = HRP(), rm = CDaR(), cluster = false)
    wt = [0.06365058210843497, 0.053794323634285435, 0.061675036869342476, 0.03,
          0.04087225720166965, 0.04480131688257841, 0.030000000000000002, 0.07, 0.03,
          0.0518685746904485, 0.07, 0.030000000000000006, 0.03, 0.06999999999999998, 0.03,
          0.05154040230787044, 0.07, 0.07, 0.03179750630537009, 0.07]
    @test isapprox(w4.weights, wt)
    @test all(w4.weights .>= 0.03)
    @test all(w4.weights .- 0.07 .<= eps() * N)

    w5 = optimise!(portfolio; hclust_alg = hclust_alg, hclust_opt = hclust_opt,
                   type = HERC(), rm = CDaR(), cluster = false)
    wt = [0.07, 0.04731823124674532, 0.07, 0.03, 0.06475109825064297, 0.03,
          0.0344529244661744, 0.03, 0.03385856962021572, 0.04311499951757443,
          0.06505672527156531, 0.03, 0.03, 0.04826336645397842, 0.03, 0.07, 0.07,
          0.06671806091164698, 0.0664660242614564, 0.07]
    @test isapprox(w5.weights, wt)
    @test all(abs.(w5.weights .- 0.03) .>= -eps() * N)
    @test all(w5.weights .<= 0.07)

    w6 = optimise!(portfolio; hclust_alg = hclust_alg, hclust_opt = hclust_opt,
                   type = NCO(), rm = CDaR(), cluster = false)
    wt = [0.07954545454545454, 0.03409090909090909, 0.07954545454545454,
          0.03409090909090909, 0.03409090909090909, 0.03409090909090909,
          0.03409090909090909, 0.03409090909090909, 0.03409090909090909,
          0.03409090909090909, 0.07954545454545454, 0.03409090909090909,
          0.03409090909090909, 0.03409090909090909, 0.03409090909090909,
          0.03409090909090909, 0.07954545454545454, 0.07954545454545454,
          0.07954545454545454, 0.07954545454545454]
    @test isapprox(w6.weights, wt)
    @test all(w6.weights .>= 0.03)
    @test !all(w6.weights .<= 0.07)

    portfolio.w_min = 0
    portfolio.w_max = 1
    w7 = optimise!(portfolio; hclust_alg = hclust_alg, hclust_opt = hclust_opt,
                   type = HRP(), rm = CDaR(), cluster = false)
    w8 = optimise!(portfolio; hclust_alg = hclust_alg, hclust_opt = hclust_opt,
                   type = HERC(), rm = CDaR(), cluster = false)
    w9 = optimise!(portfolio; hclust_alg = hclust_alg, hclust_opt = hclust_opt,
                   type = NCO(), rm = CDaR(), cluster = false)

    portfolio.w_min = Float64[]
    portfolio.w_max = Float64[]
    w10 = optimise!(portfolio; hclust_alg = hclust_alg, hclust_opt = hclust_opt,
                    type = HRP(), rm = CDaR(), cluster = false)
    w11 = optimise!(portfolio; hclust_alg = hclust_alg, hclust_opt = hclust_opt,
                    type = HERC(), rm = CDaR(), cluster = false)
    w12 = optimise!(portfolio; hclust_alg = hclust_alg, hclust_opt = hclust_opt,
                    type = NCO(), rm = CDaR(), cluster = false)

    @test isapprox(w7.weights, w10.weights)
    @test isapprox(w8.weights, w11.weights)
    @test isapprox(w9.weights, w12.weights)
end

@testset "Shorting with NCO" begin
    portfolio = HCPortfolio(; prices = prices,
                            solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                             :params => Dict("verbose" => false,
                                                                             "max_step_fraction" => 0.75))))

    asset_statistics!(portfolio)
    hclust_alg = HAC()
    hclust_opt = HCType()

    portfolio.w_min = -0.2
    portfolio.w_max = 0.8
    w1 = optimise!(portfolio; hclust_alg = hclust_alg, hclust_opt = hclust_opt,
                   cluster = true,
                   type = NCO(; opt_kwargs = (; obj = Sharpe()),
                              port_kwargs = (; short = true)))
    wt = [-0.087301210473091, 0.011322088745794164, 0.013258683926176839,
          0.004456088735487908, 0.22464713851216647, -0.04347232765939294,
          0.04304321233588089, 0.02538694929635001, 3.6391344665779835e-10,
          4.348234024497777e-8, 0.03768812903884124, -1.1037524427811691e-10,
          -0.013820021862556863, -4.136140736693239e-11, -0.01540643428627949,
          0.10308907870563483, 0.18282205944978064, 0.03429683638117709,
          0.11998968304453421, 2.4149790185098565e-9]
    @test isapprox(w1.weights, wt)
    @test isapprox(w1.weights, wt)
    @test all(w1.weights[w1.weights .>= 0] .<= 0.8)
    @test all(w1.weights[w1.weights .<= 0] .>= -0.2)
    @test sum(w1.weights[w1.weights .>= 0]) <= 0.8
    @test sum(abs.(w1.weights[w1.weights .<= 0])) <= 0.2

    w2 = optimise!(portfolio; hclust_alg = hclust_alg, hclust_opt = hclust_opt,
                   cluster = false,
                   type = NCO(; opt_kwargs = (; obj = Sharpe()),
                              port_kwargs = (; short = true, short_u = 0.3, long_u = 0.6)))
    wt = [-0.016918145881010444, 0.001712686850666274, 0.003151233891742689,
          0.000877434644273003, 0.02890364374221545, -0.031938949651537776,
          0.007380063055004761, 0.006474271417208838, -1.2243771111394412e-12,
          0.0035435277501784123, 0.0162240125533059, -0.0014765958830886081,
          -0.004205149992108177, -0.005246447380365774, -0.006973410234892313,
          0.01793024969296401, 0.023139966869064287, 0.012461571215558315,
          0.03566741209372675, -0.0007073747516812343]
    @test isapprox(w2.weights, wt)
    @test isapprox(w2.weights, wt)
    @test all(w2.weights[w2.weights .>= 0] .<= 0.8)
    @test all(w2.weights[w2.weights .<= 0] .>= -0.2)
    @test sum(w2.weights[w2.weights .>= 0]) <= 0.6
    @test sum(abs.(w2.weights[w2.weights .<= 0])) <= 0.3

    w3 = optimise!(portfolio; hclust_alg = hclust_alg, hclust_opt = hclust_opt,
                   cluster = false,
                   type = NCO(; opt_kwargs = (; obj = Sharpe()),
                              port_kwargs_o = (; short = true, short_u = 0.6, long_u = 0.4)))
    wt = [1.0339757656912846e-25, 3.308722450212111e-24, 8.271806125530277e-25,
          1.6543612251060553e-24, 5.551115123125783e-17, -7.754818242684634e-26,
          4.163336342344337e-17, -2.7755575615628914e-17, -8.271806125530277e-25,
          -1.2407709188295415e-24, -8.673617379884035e-19, 1.925929944387236e-34,
          3.611118645726067e-35, -2.0679515313825692e-25, 1.2924697071141057e-26,
          1.0339757656912846e-25, 2.7755575615628914e-17, -1.3877787807814457e-17, -0.2,
          4.1359030627651384e-25]
    @test isapprox(w3.weights, wt)
    @test all(w3.weights[w3.weights .>= 0] .<= 0.8)
    @test all(w3.weights[w3.weights .<= 0] .>= -0.2)
    @test sum(w3.weights[w3.weights .>= 0]) <= 0.6
    @test sum(abs.(w3.weights[w3.weights .<= 0])) <= 0.3

    w4 = optimise!(portfolio; hclust_alg = hclust_alg, hclust_opt = hclust_opt,
                   cluster = false,
                   type = NCO(; opt_kwargs = (; obj = Sharpe()),
                              port_kwargs = (; short = true, short_u = 0.3, long_u = 0.6),
                              port_kwargs_o = (; short = true)))
    wt = [-0.045115057672989595, 0.004567165136595039, 0.008403290748699927,
          0.0023398258212850554, 0.07707638671287502, -0.08517053748341849,
          0.019680161942876372, 0.017264724808840448, -3.2650058244204215e-12,
          0.009449407897343706, 0.043264036055620944, -0.003937587780318715,
          -0.01121372977736988, -0.013990527181995529, -0.018595754764415592,
          0.0478139841087731, 0.06170658103975119, 0.03323085855661183, 0.09511310458886758,
          -0.0018863327543673799]
    @test isapprox(w4.weights, wt)
    @test all(w4.weights[w4.weights .>= 0] .<= 0.8)
    @test all(w4.weights[w4.weights .<= 0] .>= -0.2)
    @test sum(w4.weights[w4.weights .>= 0]) <= 0.6
    @test sum(abs.(w4.weights[w4.weights .<= 0])) <= 0.3

    w5 = optimise!(portfolio; hclust_alg = hclust_alg, hclust_opt = hclust_opt,
                   cluster = false,
                   type = NCO(; opt_kwargs = (; obj = Sharpe()),
                              port_kwargs_o = (; short = true, short_u = 0.6, long_u = 0.4),
                              port_kwargs = (; short = true)))
    wt = [-0.004706878592862724, 0.0006104348021668806, 0.0007148470817697388,
          0.00024025175095859297, 0.012111937526189324, 0.028034645035369427,
          0.02059326767499224, -0.016371658716560954, -2.3468226455651654e-10,
          -2.8041102000015e-8, -0.024304503037642658, -5.280709377299777e-11,
          -0.006611946321967141, 2.6673344541851835e-11, -0.007370937436053205,
          0.04932115604168774, 0.009856922180849811, -0.022117509817183564, -0.2,
          1.302045296146306e-10]
    @test isapprox(w5.weights, wt)
    @test all(w5.weights[w5.weights .>= 0] .<= 0.8)
    @test all(w5.weights[w5.weights .<= 0] .>= -0.2)
    @test sum(w5.weights[w5.weights .>= 0]) <= 0.4
    @test sum(abs.(w5.weights[w5.weights .<= 0])) <= 0.6
end

@testset "NCO" begin
    portfolio = HCPortfolio(; prices = prices,
                            solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                             :params => Dict("verbose" => false,
                                                                             :check_sol => (allow_local = true,
                                                                                            allow_almost = true),
                                                                             "max_step_fraction" => 0.75))))

    asset_statistics!(portfolio)
    hclust_alg = DBHT()
    hclust_opt = HCType()

    rm = SD()
    w1 = optimise!(portfolio; rm = rm, cluster = true, hclust_alg = hclust_alg,
                   hclust_opt = hclust_opt, type = NCO(; opt_kwargs = (; obj = MinRisk())))
    w2 = optimise!(portfolio; rm = rm, cluster = false, hclust_alg = hclust_alg,
                   hclust_opt = hclust_opt,
                   type = NCO(; opt_kwargs = (; obj = Utility(; l = l))))
    w3 = optimise!(portfolio; rm = rm, cluster = false, hclust_alg = hclust_alg,
                   hclust_opt = hclust_opt,
                   type = NCO(; opt_kwargs = (; obj = Sharpe(; rf = rf))))
    w4 = optimise!(portfolio; rm = rm, cluster = false, hclust_alg = hclust_alg,
                   hclust_opt = hclust_opt, type = NCO(; opt_kwargs = (; obj = MaxRet())))
    w5 = optimise!(portfolio; rm = rm, cluster = false, hclust_alg = hclust_alg,
                   hclust_opt = hclust_opt, type = NCO(; opt_kwargs = (; type = RP())))
    w1t = [0.013937085585839061, 0.04359528939613315, 0.010055244196525516,
           0.018529417015299345, 0.0060008557954075595, 0.06377945741010448,
           8.830053665226257e-7, 0.13066772968460527, 7.275669461246384e-6,
           1.1522825766672644e-5, 0.2351158704607046, 0.005999396004399579,
           3.2750802699731945e-7, 0.11215293274059893, 5.127280882607535e-7,
           4.000811842370495e-6, 0.037474931372346656, 0.18536090811244715,
           0.04037088344151298, 0.09693547623552375]
    w2t = [3.536321744413884e-9, 8.35986848490829e-9, 1.0627312978295631e-8,
           8.235768003812635e-9, 0.8703125909491561, 1.092557591202154e-18,
           0.12797913827326654, 4.176996018696875e-9, 2.695529928425216e-15,
           0.0007074965433076892, 5.925437102470992e-15, 7.374444588643583e-19,
           3.061344318207294e-12, 2.4076449883278634e-16, 8.597463762416993e-17,
           0.000999064400805917, 1.481241841822526e-6, 4.572656300248877e-9,
           1.827392812927106e-7, 6.3403468197462965e-9]
    w3t = [1.0268536101998741e-9, 4.422568918003183e-9, 4.380602489535938e-9,
           2.956064338817857e-9, 0.4781398601946685, 1.4512115881232926e-11,
           0.06499664007565499, 3.099354900052221e-9, 2.1921024761819662e-10,
           0.028340464844810755, 4.464581793782242e-10, 1.6080765752686552e-11,
           5.332341752653171e-11, 3.3839229356309946e-11, 8.259354624250968e-12,
           0.11005168755732588, 0.27213167660281556, 2.658802782758464e-9,
           0.04633964788062631, 3.5081676586779137e-9]
    w4t = [1.881251684938379e-8, 1.9822795312812863e-8, 2.414660854891215e-8,
           2.2334550044634408e-8, 1.4029799495224622e-6, 3.0636948499419235e-15,
           0.9999980761970937, 1.4630313582860776e-8, 3.638630221309437e-13,
           4.2808125305880613e-14, 6.995666270496466e-15, 3.1932948088578136e-15,
           6.779742676063459e-15, 4.669386166635582e-15, 2.0459303239459237e-15,
           2.2160699849552174e-7, 2.5006723377423327e-8, 1.5281438581326256e-8,
           1.4121190570852067e-7, 1.7968672959672945e-8]
    w5t = [0.03594354904543136, 0.03891177139524374, 0.03393897922702459,
           0.03191180338166861, 0.03154669000464689, 0.05674674429270935,
           0.02049786890241706, 0.05976698398595266, 0.04257331056600222,
           0.11964711125701236, 0.08477538305842097, 0.03508483105116085, 0.043256028579427,
           0.060481128224384596, 0.029559513397421338, 0.08211257818515973,
           0.04003780013424942, 0.05824274268779625, 0.048456009444482875,
           0.04650917317938824]
    @test isapprox(w1.weights, w1t)
    @test isapprox(w2.weights, w2t)
    @test isapprox(w3.weights, w3t)
    @test isapprox(w4.weights, w4t)
    @test isapprox(w5.weights, w5t, rtol = 5.0e-5)

    rm = MAD()
    w6 = optimise!(portfolio; rm = rm, cluster = false, hclust_alg = hclust_alg,
                   hclust_opt = hclust_opt, type = NCO(; opt_kwargs = (; obj = MinRisk())))
    w7 = optimise!(portfolio; rm = rm, cluster = false, hclust_alg = hclust_alg,
                   hclust_opt = hclust_opt,
                   type = NCO(; opt_kwargs = (; obj = Utility(; l = l))))
    w8 = optimise!(portfolio; rm = rm, cluster = false, hclust_alg = hclust_alg,
                   hclust_opt = hclust_opt,
                   type = NCO(; opt_kwargs = (; obj = Sharpe(; rf = rf))))
    w9 = optimise!(portfolio; rm = rm, cluster = false, hclust_alg = hclust_alg,
                   hclust_opt = hclust_opt, type = NCO(; opt_kwargs = (; obj = MaxRet())))
    w10 = optimise!(portfolio; rm = rm, cluster = false, hclust_alg = hclust_alg,
                    hclust_opt = hclust_opt, type = NCO(; opt_kwargs = (; type = RP())))
    w1t = [0.04059732374917473, 0.045600643372826564, 0.0008508736496987,
           0.006437852670865818, 0.002841800035783749, 0.05930362559033975,
           3.682328495715141e-9, 0.15293340032515007, 0.005759786022651817,
           5.950241253633107e-8, 0.1987549961908217, 2.0545559421162275e-9,
           5.32244045270275e-10, 0.10473908918633106, 2.2579839445845772e-10,
           2.187558110026064e-8, 0.056505427326554836, 0.19033548239854162,
           0.04096913911515197, 0.09437047249318714]
    w2t = [0.019135840114057178, 0.04319664563133952, 6.989818506731821e-8,
           0.010479643822795195, 0.044710601991195954, 0.023524270507582417,
           0.0027081241614039355, 0.15841866168441052, 0.004902231961856615,
           0.010598283527252079, 0.17605850844462068, 2.835425217252423e-10,
           6.674569801134214e-12, 0.07184677071156634, 1.9086830233118526e-11,
           0.00482767668105368, 0.07950214846405654, 0.1805692837235624,
           0.06741985554913642, 0.10210138281662147]
    w3t = [1.7711462588275667e-9, 6.253014446612376e-9, 4.5219875754910405e-9,
           3.3567391619124057e-9, 0.626542835057035, 4.244688293643053e-12,
           0.0410106345340863, 3.5790876174967654e-9, 6.627977950621314e-10,
           4.951758424642855e-9, 1.6115364648536767e-10, 4.154724183802552e-12,
           2.0959897320947726e-12, 9.858343606151467e-12, 1.5411541422962657e-12,
           0.13259044906595502, 0.1453951017104159, 3.983210451398619e-9,
           0.05446094499291796, 5.376799673344027e-9]
    w4t = [1.0697281800757452e-7, 1.1311775297941919e-7, 1.3780259688825507e-7,
           1.268055655932939e-7, 8.50822923994295e-6, 3.204283203701939e-13,
           0.9999862377026062, 8.136173647277906e-8, 4.024041566311586e-11,
           6.94984039611021e-12, 8.035455863957799e-13, 3.4976521519678484e-13,
           1.137123602399115e-12, 5.126371707340376e-13, 2.1878366160025683e-13,
           2.7532186682419475e-6, 1.431604095722679e-7, 8.5069527979556e-8,
           1.6044460686057254e-6, 1.0206247702601848e-7]
    w5t = [0.038482572880612374, 0.04124259860035183, 0.036310167752490984,
           0.03068785441299921, 0.03272027360303007, 0.0569582645840768,
           0.01903357698244691, 0.06024480428552155, 0.041510930079800556,
           0.10991717839342878, 0.08278034364647732, 0.0346366457548056,
           0.039589293259183145, 0.058854479193534, 0.026495842632932323,
           0.08543930360245622, 0.043229183172491234, 0.06156421651978086,
           0.0479372991636419, 0.0523651714799385]
    @test isapprox(w6.weights, w1t)
    @test isapprox(w7.weights, w2t)
    @test isapprox(w8.weights, w3t)
    @test isapprox(w9.weights, w4t)
    @test isapprox(w10.weights, w5t, rtol = 1.0e-5)

    rm = SSD()
    w11 = optimise!(portfolio; rm = rm, cluster = false, hclust_alg = hclust_alg,
                    hclust_opt = hclust_opt, type = NCO(; opt_kwargs = (; obj = MinRisk())))
    w12 = optimise!(portfolio; rm = rm, cluster = false, hclust_alg = hclust_alg,
                    hclust_opt = hclust_opt,
                    type = NCO(; opt_kwargs = (; obj = Utility(; l = l))))
    w13 = optimise!(portfolio; rm = rm, cluster = false, hclust_alg = hclust_alg,
                    hclust_opt = hclust_opt,
                    type = NCO(; opt_kwargs = (; obj = Sharpe(; rf = rf))))
    w14 = optimise!(portfolio; rm = rm, cluster = false, hclust_alg = hclust_alg,
                    hclust_opt = hclust_opt, type = NCO(; opt_kwargs = (; obj = MaxRet())))
    w15 = optimise!(portfolio; rm = rm, cluster = false, hclust_alg = hclust_alg,
                    hclust_opt = hclust_opt, type = NCO(; opt_kwargs = (; type = RP())))
    w1t = [7.121013120801824e-6, 0.061607987382417705, 1.4912756768376158e-9,
           0.009681807066376894, 0.009020659057053952, 0.059478722069370014,
           1.3179217642229542e-9, 0.11544692256445871, 1.5887557238653072e-9,
           7.210694103149243e-10, 0.24458547584840604, 0.009620637405741693,
           1.9394792891670725e-11, 0.12245874714177621, 5.062813106756657e-10,
           2.792879935708652e-10, 0.029931480605545943, 0.2164485615370284,
           0.03511997743484373, 0.08659189494987324]
    w2t = [2.505966264249908e-9, 0.06462231159494408, 6.312604456450302e-10,
           0.008290344916341286, 0.025721971719375243, 0.03177714442542715,
           3.100206836273914e-9, 0.12015839241220808, 5.753553216235469e-10,
           8.579825113335541e-10, 0.23290956247196232, 0.007185501507127438,
           4.740916222504663e-12, 0.10213169998624982, 1.0583051387480673e-10,
           3.598662360154047e-10, 0.0423881614274072, 0.21855640410841734,
           0.058728073140139496, 0.0875304241491915]
    w3t = [3.376211111543454e-9, 9.027836247432593e-9, 6.624231451997948e-9,
           4.456498283828095e-9, 0.6369289883278642, 1.2292588648044406e-15,
           0.041553493600651026, 8.454859532581521e-9, 3.335199701461747e-14,
           0.018745128572890082, 2.92766141429812e-14, 1.3113761446096982e-15,
           1.1080502561018842e-11, 3.1664656065420014e-15, 4.805195416637262e-16,
           0.14899701281311017, 0.15377296833875437, 1.0423893797290456e-8,
           2.357572635201043e-6, 8.399415392270166e-9]
    w4t = [3.169138468128157e-7, 3.361556961865357e-7, 4.168074813823495e-7,
           3.81870053665129e-7, 2.07656774406466e-5, 1.4691116695512883e-13,
           0.9999738665006, 2.3428043709562976e-7, 1.785834933043541e-11,
           2.4544969294036332e-12, 3.7784102629307113e-13, 1.6211460037242008e-13,
           4.1719665878120677e-13, 2.3542525005240866e-13, 1.0286912965828565e-13,
           1.7063596294143836e-6, 4.329282002176093e-7, 2.458728199437001e-7,
           9.967089115920972e-7, 2.9990312775869386e-7]
    w5t = [0.03596400232271158, 0.03926422387565698, 0.03173743270063896,
           0.030431800778782426, 0.031103389684032656, 0.055774400848792,
           0.02012838723950023, 0.05939418239844191, 0.04130776241819249,
           0.11737585383647636, 0.08621821890143008, 0.034941660524425426,
           0.047599879413746715, 0.060741559517526224, 0.03121118004196171,
           0.08601968722906522, 0.038070323373188976, 0.058782688335425784,
           0.04855866178072313, 0.045374704779281204]
    @test isapprox(w11.weights, w1t)
    @test isapprox(w12.weights, w2t)
    @test isapprox(w13.weights, w3t)
    @test isapprox(w14.weights, w4t)
    @test isapprox(w15.weights, w5t, rtol = 5.0e-5)

    rm = FLPM(; target = rf)
    w16 = optimise!(portfolio; rm = rm, cluster = false, hclust_alg = hclust_alg,
                    hclust_opt = hclust_opt, type = NCO(; opt_kwargs = (; obj = MinRisk())))
    w17 = optimise!(portfolio; rm = rm, cluster = false, hclust_alg = hclust_alg,
                    hclust_opt = hclust_opt,
                    type = NCO(; opt_kwargs = (; obj = Utility(; l = l))))
    w18 = optimise!(portfolio; rm = rm, cluster = false, hclust_alg = hclust_alg,
                    hclust_opt = hclust_opt,
                    type = NCO(; opt_kwargs = (; obj = Sharpe(; rf = rf))))
    w19 = optimise!(portfolio; rm = rm, cluster = false, hclust_alg = hclust_alg,
                    hclust_opt = hclust_opt, type = NCO(; opt_kwargs = (; obj = MaxRet())))
    w20 = optimise!(portfolio; rm = rm, cluster = false, hclust_alg = hclust_alg,
                    hclust_opt = hclust_opt, type = NCO(; opt_kwargs = (; type = RP())))
    w1t = [0.016087285360179032, 0.03929456125680425, 2.654669650722757e-8,
           0.009017145085378016, 0.04852710668341847, 0.022953259582093585,
           0.00113984978745799, 0.1456189117438582, 0.003637704859085449,
           0.014589771845411147, 0.17831030715846177, 1.2587617733789283e-9,
           2.1033525434368566e-11, 0.07185635646074559, 1.4705271853683893e-10,
           0.006896966942305214, 0.07311971752422704, 0.188190292120055,
           0.07385752447075629, 0.10690321114621834]
    w2t = [8.979792049815534e-9, 0.04319677159738373, 0.010553521731731526,
           0.011677797169390153, 0.07462748970960995, 2.4993569347447565e-9,
           0.0051395703992312735, 0.1393795754745042, 0.003947096902484896,
           0.013483746896893172, 0.16322712734000247, 7.880340298212877e-11,
           3.061723277788993e-12, 0.044267239650700875, 4.5346573333915216e-12,
           0.007528180962873085, 0.08925135943043455, 0.18949901379313963,
           0.08920985591556879, 0.11501164146050306]
    w3t = [4.70569028136048e-10, 1.3574602442249478e-9, 1.2171599066627055e-9,
           7.480173861691719e-10, 0.6552130462915482, 2.118918252960233e-12,
           0.037643941367949886, 8.865144930684747e-10, 3.4692335767765595e-10,
           8.838695321277905e-9, 7.803488985844738e-11, 2.036777057328308e-12,
           1.1596869403076888e-12, 4.9035154832726276e-12, 8.157197134641697e-13,
           0.1480747059921399, 0.11655387517352318, 1.0532930709570267e-9,
           0.042514414741941266, 1.4251951319177548e-9]
    w4t = [1.0701351024251811e-7, 1.1315899899963757e-7, 1.3784659823204616e-7,
           1.2684858651859715e-7, 8.51228975107345e-6, 3.206232650541434e-13,
           0.9999862315870663, 8.139959013478876e-8, 4.026227025375141e-11,
           6.954578268366241e-12, 8.040123708309039e-13, 3.4997831995900936e-13,
           1.1378946431508748e-12, 5.129468913623559e-13, 2.1891375482044552e-13,
           2.7542962972698258e-6, 1.432048377198796e-7, 8.510789499184016e-8,
           1.6050937773801256e-6, 1.0210252984489028e-7]
    w5t = [0.040183831915019924, 0.0434257369874625, 0.037998059938328436,
           0.0317472517897774, 0.03729164492388537, 0.049675475199063394,
           0.02023805337478654, 0.060464070967644855, 0.04287003746789862,
           0.10829254014258542, 0.08072109742895471, 0.031429506535522345,
           0.03707944602714823, 0.05568975882884374, 0.023775343473611613,
           0.08922090474464904, 0.045103020475796764, 0.061381817706047864,
           0.05045637642477227, 0.05295602564820106]
    @test isapprox(w16.weights, w1t)
    @test isapprox(w17.weights, w2t)
    @test isapprox(w18.weights, w3t, rtol = 5.0e-5)
    @test isapprox(w19.weights, w4t)
    @test isapprox(w20.weights, w5t, rtol = 5.0e-6)

    rm = SLPM(; target = rf)
    w21 = optimise!(portfolio; rm = rm, cluster = false, hclust_alg = hclust_alg,
                    hclust_opt = hclust_opt, type = NCO(; opt_kwargs = (; obj = MinRisk())))
    w22 = optimise!(portfolio; rm = rm, cluster = false, hclust_alg = hclust_alg,
                    hclust_opt = hclust_opt,
                    type = NCO(; opt_kwargs = (; obj = Utility(; l = l))))
    w23 = optimise!(portfolio; rm = rm, cluster = false, hclust_alg = hclust_alg,
                    hclust_opt = hclust_opt,
                    type = NCO(; opt_kwargs = (; obj = Sharpe(; rf = rf))))
    w24 = optimise!(portfolio; rm = rm, cluster = false, hclust_alg = hclust_alg,
                    hclust_opt = hclust_opt, type = NCO(; opt_kwargs = (; obj = MaxRet())))
    w25 = optimise!(portfolio; rm = rm, cluster = false, hclust_alg = hclust_alg,
                    hclust_opt = hclust_opt, type = NCO(; opt_kwargs = (; type = RP())))
    w1t = [5.889098113924682e-9, 0.06587618283494774, 1.5750531512406229e-9,
           0.008307801198717074, 0.025055535680020887, 0.031007017694062042,
           6.622189397331921e-9, 0.11895069126754958, 5.227655044085646e-10,
           1.0425338132041355e-9, 0.2342323891676781, 0.007211234975414065,
           5.564426798819609e-12, 0.10219179342324933, 1.0936729193190541e-10,
           4.3817799370420595e-10, 0.03983727337898306, 0.22064984507420685,
           0.059770315866896725, 0.08690990323352493]
    w2t = [1.3126689349585126e-9, 0.0683982344676695, 6.641537637834876e-10,
           0.005752802760099117, 0.04380613474339462, 0.006467953935425663,
           0.001194218858958455, 0.12252327476998784, 4.934350529800896e-10,
           5.070902306913777e-10, 0.22365749118830133, 0.005234400227547068,
           5.170702817426386e-19, 0.08421644705458561, 5.950345115515079e-11,
           2.284721290071047e-10, 0.05254721479863481, 0.2198834885600653,
           0.07900905406850742, 0.08730928130149963]
    w3t = [2.0146575390229065e-9, 5.441471606993735e-9, 3.941196024805785e-9,
           2.6032258222664272e-9, 0.6332890387116795, 4.488189568435251e-17,
           0.04170782181673219, 5.14894387005817e-9, 1.1896470812621962e-15,
           0.019835822335139475, 1.0623827899795569e-15, 4.8011381381596766e-17,
           1.1768078009951815e-11, 1.160757066681121e-16, 1.762740290937585e-17,
           0.1503590248352136, 0.15480805233160272, 6.412266352587129e-9,
           2.0941020989105202e-7, 4.9858908525016265e-9]
    w4t = [3.590022276972911e-7, 3.7987197002485197e-7, 4.669822555594107e-7,
           4.2943174301178453e-7, 2.4464659432521346e-5, 1.5740255311730082e-13,
           0.999969595458026, 2.6835123811842975e-7, 1.905346857630097e-11,
           2.6864639955607387e-12, 4.034356965950875e-13, 1.7357945988136158e-13,
           4.589237697745928e-13, 2.5158220448101467e-13, 1.1050657216281399e-13,
           1.8543896347411247e-6, 4.842533754688074e-7, 2.8119376275198016e-7,
           1.0759705564579637e-6, 3.4041248239753164e-7]
    w5t = [0.03651367301522128, 0.04020193126064295, 0.032440500747771495,
           0.03111511622029599, 0.033302134728400365, 0.0520110878935054, 0.020990313244494,
           0.05987395308080474, 0.041723833306903, 0.1172587304056323, 0.08535502205077988,
           0.033461836634349776, 0.045698746344584214, 0.0586004642617431,
           0.029201398506925658, 0.08770180110993163, 0.03939357780871859,
           0.059462235315846254, 0.04953271898182618, 0.04616092508162323]
    @test isapprox(w21.weights, w1t)
    @test isapprox(w22.weights, w2t)
    @test isapprox(w23.weights, w3t)
    @test isapprox(w24.weights, w4t)
    @test isapprox(w25.weights, w5t, rtol = 5.0e-5)

    rm = WR()
    w26 = optimise!(portfolio; rm = rm, cluster = false, hclust_alg = hclust_alg,
                    hclust_opt = hclust_opt, type = NCO(; opt_kwargs = (; obj = MinRisk())))
    w27 = optimise!(portfolio; rm = rm, cluster = false, hclust_alg = hclust_alg,
                    hclust_opt = hclust_opt,
                    type = NCO(; opt_kwargs = (; obj = Utility(; l = l))))
    w28 = optimise!(portfolio; rm = rm, cluster = false, hclust_alg = hclust_alg,
                    hclust_opt = hclust_opt,
                    type = NCO(; opt_kwargs = (; obj = Sharpe(; rf = rf))))
    w29 = optimise!(portfolio; rm = rm, cluster = false, hclust_alg = hclust_alg,
                    hclust_opt = hclust_opt, type = NCO(; opt_kwargs = (; obj = MaxRet())))
    w30 = optimise!(portfolio; rm = rm, cluster = false, hclust_alg = hclust_alg,
                    hclust_opt = hclust_opt, type = NCO(; opt_kwargs = (; type = RP())))
    w1t = [1.9747133396600235e-11, 0.28983188049620723, 2.2178320507471025e-11,
           2.3903571001566017e-9, 8.896231855698167e-11, 1.5285685327330832e-10,
           0.15054354915956833, 0.24258042615731373, 3.844939099024774e-20,
           7.052877593110094e-12, 5.164211395727411e-10, 5.2576871331120963e-20,
           1.4150944216989741e-12, 4.725567845330821e-20, 1.8628408346794537e-10,
           2.3466054659808233e-12, 2.582105425683964e-10, 9.538296837205346e-11,
           1.3506383389403547e-19, 0.3170441404456957]
    w2t = [1.141783373321512e-11, 0.28983188080893435, 8.285628164196383e-12,
           2.486875189454384e-9, 7.827728961310542e-11, 7.65738747390303e-11,
           0.15054354925197597, 0.24258042607837926, 2.3157763218355765e-20,
           1.2910398707352923e-12, 2.587019605725475e-10, 4.3942020776527333e-20,
           2.59035166323434e-13, 2.282609616740485e-20, 9.331929681572565e-11,
           4.295496663376929e-13, 2.666566597352273e-10, 1.0315442168985836e-10,
           1.8563920743873889e-19, 0.3170441404754686]
    w3t = [4.402292614352855e-9, 5.118496511304611e-8, 4.2768740184544925e-9,
           1.1806247921427134e-8, 0.5977202415770091, 5.162354964911019e-11,
           0.1966217669950393, 5.8367058605025425e-8, 1.3635578134809385e-9,
           0.028193636458432692, 0.02200162915850006, 7.512726824369969e-11,
           1.4929495461767638e-11, 1.457593147146519e-10, 2.3261104388259625e-11,
           0.011827130525631805, 0.012053559096343503, 0.10170500941174024,
           0.029876892099256584, 2.966349923966149e-9]
    w4t = [2.469260820591323e-9, 2.6086071830057514e-9, 3.1707427364237383e-9,
           2.9195261819125968e-9, 2.0428633058799499e-7, 7.36816518037098e-18,
           0.9999997520392511, 1.8969261383610923e-9, 9.079017212568097e-16,
           1.5459544528864506e-16, 1.8269672717801768e-17, 7.96162684815466e-18,
           2.4680933724024175e-17, 1.1734067342084149e-17, 4.944690968749843e-18,
           1.4275068260027735e-8, 3.292675858335107e-9, 1.981003055125446e-9,
           8.699313274091695e-9, 2.3612935845290182e-9]
    w5t = [0.03119363952440829, 0.05706544449319162, 0.03036361882099508,
           0.03989035207471949, 0.035078474806200605, 0.04895936407693799,
           0.027872256279024486, 0.03485249409732566, 0.039052219914194025,
           0.1579067332918187, 0.05717436110021421, 0.03859948119378035,
           0.07112283583631317, 0.03764608078209501, 0.04678339289819475,
           0.08546347524280776, 0.042331906477832595, 0.029078372453158717,
           0.04127271389202918, 0.048292782744758324]
    @test isapprox(w26.weights, w1t)
    @test isapprox(w27.weights, w2t)
    @test isapprox(w28.weights, w3t)
    @test isapprox(w29.weights, w4t)
    @test isapprox(w30.weights, w5t, rtol = 5.0e-5)

    rm = CVaR()
    w31 = optimise!(portfolio; rm = rm, cluster = false, hclust_alg = hclust_alg,
                    hclust_opt = hclust_opt, type = NCO(; opt_kwargs = (; obj = MinRisk())))
    w32 = optimise!(portfolio; rm = rm, cluster = false, hclust_alg = hclust_alg,
                    hclust_opt = hclust_opt,
                    type = NCO(; opt_kwargs = (; obj = Utility(; l = l))))
    w33 = optimise!(portfolio; rm = rm, cluster = false, hclust_alg = hclust_alg,
                    hclust_opt = hclust_opt,
                    type = NCO(; opt_kwargs = (; obj = Sharpe(; rf = rf))))
    w34 = optimise!(portfolio; rm = rm, cluster = false, hclust_alg = hclust_alg,
                    hclust_opt = hclust_opt, type = NCO(; opt_kwargs = (; obj = MaxRet())))
    w35 = optimise!(portfolio; rm = rm, cluster = false, hclust_alg = hclust_alg,
                    hclust_opt = hclust_opt, type = NCO(; opt_kwargs = (; type = RP())))
    w1t = [3.664859661161272e-10, 0.0762203721651536, 6.441778816535088e-11,
           0.03138093007735649, 0.027626085711376314, 0.02300652225536527,
           5.844065587830882e-11, 0.13924934619748974, 5.805427791614419e-11,
           1.0937171459308286e-10, 0.21362141189989498, 0.020888781049182764,
           1.980200934283383e-12, 0.10087916256515272, 7.26276933420973e-11,
           4.9799871178339534e-11, 0.014862212321435466, 0.28800164310949844,
           1.9196981699887677e-10, 0.06426353167494636]
    w2t = [1.5789168760766128e-10, 0.07694081488688548, 4.000845881532486e-11,
           0.03378345968044639, 0.027373933478449422, 0.022402731945984695,
           3.630403456173319e-11, 0.14125562537830114, 6.968211398613603e-11,
           2.3479950495644183e-10, 0.21025793521634595, 0.020157628966454504,
           1.2002752293501236e-19, 0.09889797348307146, 5.720839942268613e-11,
           1.0812540256939718e-10, 0.01696672848323463, 0.28844538572619727,
           5.044951919517896e-10, 0.06351778154611436]
    w3t = [7.550077262684875e-10, 1.3306533988959414e-9, 1.1093698572328e-9,
           6.709440849073084e-10, 0.5763326871574024, 2.5772533724661772e-18,
           0.073479023506859, 2.4127955663343554e-9, 5.071595466915273e-17,
           0.029877775100693744, 2.1496819981795957e-17, 2.552000792062951e-18,
           8.626232362752768e-13, 3.4515273144533547e-18, 3.189132459473233e-18,
           0.14923907048182963, 0.17107141857002867, 2.3942026704968923e-9,
           1.3484144929514954e-8, 3.0252058041577887e-9]
    w4t = [2.6743919524671433e-8, 2.8280194662924554e-8, 3.445156994729403e-8,
           3.170223879757576e-8, 2.127112149105521e-6, 1.2520005503949514e-15,
           0.999997376413533, 2.0340977818722356e-8, 1.5723018780652422e-13,
           2.7153417265778693e-14, 3.1396703010908747e-15, 1.3666277289161734e-15,
           4.44280727298183e-15, 2.00301248454596e-15, 8.548473940793335e-16,
           1.7209117914321296e-7, 3.5791059101939696e-8, 2.1267950571210223e-8,
           1.0028872937847331e-7, 2.551630147362986e-8]
    w5t = [0.03442967428619515, 0.03622525747192852, 0.028649166159358926,
           0.028820218103751707, 0.03064487884114398, 0.051338672771389905,
           0.021297430232675357, 0.060896416149046416, 0.038724170445278865,
           0.1180505650585966, 0.08629247235515324, 0.03352163294024383,
           0.05457013256128135, 0.06613966646033598, 0.032097566038840224,
           0.08807616383700578, 0.03507071652584228, 0.06546119522707869,
           0.046780168185706385, 0.04291383634914687]
    @test isapprox(w31.weights, w1t)
    @test isapprox(w32.weights, w2t)
    @test isapprox(w33.weights, w3t)
    @test isapprox(w34.weights, w4t)
    @test isapprox(w35.weights, w5t, rtol = 5.0e-5)

    rm = EVaR()
    w36 = optimise!(portfolio; rm = rm, cluster = false, hclust_alg = hclust_alg,
                    hclust_opt = hclust_opt, type = NCO(; opt_kwargs = (; obj = MinRisk())))
    w37 = optimise!(portfolio; rm = rm, cluster = false, hclust_alg = hclust_alg,
                    hclust_opt = hclust_opt,
                    type = NCO(; opt_kwargs = (; obj = Utility(; l = l))))
    w38 = optimise!(portfolio; rm = rm, cluster = false, hclust_alg = hclust_alg,
                    hclust_opt = hclust_opt,
                    type = NCO(; opt_kwargs = (; obj = Sharpe(; rf = rf))))
    w39 = optimise!(portfolio; rm = rm, cluster = false, hclust_alg = hclust_alg,
                    hclust_opt = hclust_opt, type = NCO(; opt_kwargs = (; obj = MaxRet())))
    w40 = optimise!(portfolio; rm = rm, cluster = false, hclust_alg = hclust_alg,
                    hclust_opt = hclust_opt, type = NCO(; opt_kwargs = (; type = RP())))
    w1t = [1.3785795089567192e-8, 0.1360835835861814, 9.796765065975505e-9,
           0.03979459018377941, 1.6617234434301553e-8, 0.05740929288219373,
           0.04390484034087705, 0.14120918087952655, 7.207773383460931e-9,
           1.0722818337316549e-8, 0.18078058609599582, 0.015910536201136125,
           1.6767734194460038e-9, 0.03652307629311764, 0.019099392747369295,
           3.1989632090928942e-9, 0.04716014099993904, 0.14062385947717723,
           0.010742934281030967, 0.13075792302555284]
    w2t = [1.539901271958622e-8, 0.14215345217490227, 1.1592191304365884e-8,
           0.04131131228118324, 2.2160183751107717e-8, 0.04471285118969558,
           0.04800165632918703, 0.1485469036262668, 1.8677812696498027e-8,
           1.9790701451501078e-8, 0.16069375067272876, 0.013510808701786338,
           3.0208950773188386e-9, 0.031079294902898496, 0.01396492760596718,
           5.921045055010991e-9, 0.059404804181166615, 0.14015472932859627,
           0.019139880880730844, 0.13732553156304858]
    w3t = [1.5891310507284716e-8, 0.0028908394423298495, 1.8841580627264263e-8,
           2.617260936163577e-8, 0.5396359787197313, 4.835893610840537e-16,
           0.14358082220438478, 6.771720274363291e-8, 8.677275256467885e-15,
           0.052940532835842276, 2.308873038393805e-14, 6.065872532873969e-16,
           1.7609411452779525e-10, 1.0863174347982087e-15, 3.057560324418493e-16,
           0.04214272260157403, 0.21880825158555614, 4.3109349482186617e-7,
           2.807036157422449e-7, 1.2014639494904744e-8]
    w4t = [3.616322845273605e-9, 3.83928189118373e-9, 4.759314749439659e-9,
           4.353846320931957e-9, 2.1245024409034897e-7, 2.0442498818823596e-17,
           0.9999997322303765, 2.6610724308919717e-9, 1.979188082489676e-15,
           1.9282537358428684e-16, 5.044432852800647e-17, 2.2636681407881077e-17,
           3.279210237999571e-17, 3.27175189089651e-17, 1.3934272006895045e-17,
           1.5771366511244315e-8, 4.950988636850216e-9, 2.7967080191875943e-9,
           9.144276028456238e-9, 3.4261996722832646e-9]
    w5t = [0.03244158253456704, 0.04235778407533121, 0.029184278892081254,
           0.03478319296608296, 0.02982893493239512, 0.052958781124194984,
           0.028243170894659806, 0.05136667760741622, 0.039891921160963774,
           0.14504526807786997, 0.07537259886425227, 0.03409029432604595,
           0.06829477209996813, 0.04839288789589798, 0.0385877771971674,
           0.08429331145127741, 0.03766316906044964, 0.04276953973125722,
           0.04287636716886095, 0.04155768993926082]
    @test isapprox(w36.weights, w1t, rtol = 5.0e-5)
    @test isapprox(w37.weights, w2t, rtol = 1.0e-5)
    @test isapprox(w38.weights, w3t, rtol = 5.0e-5)
    @test isapprox(w39.weights, w4t)
    @test isapprox(w40.weights, w5t, rtol = 5.0e-5)

    portfolio.solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                               :params => Dict("verbose" => false,
                                                               "max_step_fraction" => 0.65)))
    rm = RVaR()
    w41 = optimise!(portfolio; rm = rm, cluster = false, hclust_alg = hclust_alg,
                    hclust_opt = hclust_opt, type = NCO(; opt_kwargs = (; obj = MinRisk())))
    w42 = optimise!(portfolio; rm = rm, cluster = false, hclust_alg = hclust_alg,
                    hclust_opt = hclust_opt,
                    type = NCO(; opt_kwargs = (; obj = Utility(; l = l))))
    w43 = optimise!(portfolio; rm = rm, cluster = false, hclust_alg = hclust_alg,
                    hclust_opt = hclust_opt,
                    type = NCO(; opt_kwargs = (; obj = Sharpe(; rf = rf))))
    w44 = optimise!(portfolio; rm = rm, cluster = false, hclust_alg = hclust_alg,
                    hclust_opt = hclust_opt, type = NCO(; opt_kwargs = (; obj = MaxRet())))
    w45 = optimise!(portfolio; rm = rm, cluster = false, hclust_alg = hclust_alg,
                    hclust_opt = hclust_opt, type = NCO(; opt_kwargs = (; type = RP())))
    w1t = [2.529276503881537e-9, 0.21191459984592514, 2.4711225945029506e-9,
           0.09271239767205439, 7.05231706834582e-9, 0.029526571282439483,
           0.09814897392077784, 0.1951501120999505, 7.101329052225676e-10,
           5.451261076442535e-9, 0.07966445647137076, 0.0057273682001028935,
           9.66807484714734e-10, 2.7329296163754835e-9, 0.022440471292527384,
           1.6725018652286136e-9, 0.0896834273475343, 0.00010556324695181721,
           2.573094273901442e-9, 0.174926032460922]
    w2t = [3.2045890629594967e-9, 0.21575828346428821, 3.048365395458857e-9,
           0.09371783451172656, 1.2254224870649566e-8, 0.0251871911716709,
           0.10039772625098634, 0.19855611913117432, 1.5143763672179137e-9,
           1.0777719587061035e-8, 0.06876464539810181, 0.005185954591406701,
           1.896307599653251e-9, 3.787598431493779e-9, 0.018877883803262885,
           3.30735777657766e-9, 0.09601955141542645, 6.174466512333312e-8,
           1.0543645953386197e-8, 0.17753469818310574]
    w3t = [7.408349638706929e-9, 0.0038186592000146514, 5.835089914154846e-9,
           2.0063286898793054e-8, 0.5334258282960406, 1.8376899941977034e-11,
           0.1792828310954398, 0.0009339923167643227, 2.6910367020951073e-10,
           0.03645665813298778, 0.001470808950642995, 2.4826140427686114e-11,
           1.218766536400032e-10, 4.4366478375481055e-11, 1.1469569402445713e-11,
           0.018265611493780018, 0.13315781240098476, 0.08717716213664908,
           0.006010598751580895, 3.428369265483797e-9]
    w4t = [7.899825126828249e-8, 8.398383302840335e-8, 1.045159719223709e-7,
           9.545583379702279e-8, 3.920187162969964e-6, 1.1063451637644887e-14,
           0.9999948829831524, 5.7757250340023774e-8, 9.7629952535239e-13,
           9.821169011242402e-14, 2.7346804461857104e-14, 1.2254614413840212e-14,
           1.699822475848904e-14, 1.774075055765404e-14, 7.513493183055607e-15,
           3.358367969634551e-7, 1.0880889674288217e-7, 6.077133035237832e-8,
           1.959289193361942e-7, 7.477143351754478e-8]
    w5t = [0.03136070689246619, 0.047254371717960104, 0.029317542180578023,
           0.03807351023889607, 0.03218469553679128, 0.05212310046400624,
           0.030413012215974103, 0.04311656527524499, 0.03950391685866444,
           0.1567530853540612, 0.06469336743911681, 0.035572582903316, 0.0721152246935304,
           0.041529293632031145, 0.04167307947370621, 0.08552272108784345,
           0.039119677573665544, 0.035231385132331074, 0.04143766055839026,
           0.04300450077142654]
    @test isapprox(w41.weights, w1t, rtol = 5.0e-6)
    @test isapprox(w42.weights, w2t, rtol = 1.0e-6)
    @test isapprox(w43.weights, w3t, rtol = 5.0e-6)
    @test isapprox(w44.weights, w4t)
    @test isapprox(w45.weights, w5t, rtol = 1.0e-7)

    portfolio.solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                               :params => Dict("verbose" => false,
                                                               "max_step_fraction" => 0.75)))
    rm = MDD()
    w46 = optimise!(portfolio; rm = rm, cluster = false, hclust_alg = hclust_alg,
                    hclust_opt = hclust_opt, type = NCO(; opt_kwargs = (; obj = MinRisk())))
    w47 = optimise!(portfolio; rm = rm, cluster = false, hclust_alg = hclust_alg,
                    hclust_opt = hclust_opt,
                    type = NCO(; opt_kwargs = (; obj = Utility(; l = l))))
    w48 = optimise!(portfolio; rm = rm, cluster = false, hclust_alg = hclust_alg,
                    hclust_opt = hclust_opt,
                    type = NCO(; opt_kwargs = (; obj = Sharpe(; rf = rf))))
    w49 = optimise!(portfolio; rm = rm, cluster = false, hclust_alg = hclust_alg,
                    hclust_opt = hclust_opt, type = NCO(; opt_kwargs = (; obj = MaxRet())))
    w50 = optimise!(portfolio; rm = rm, cluster = false, hclust_alg = hclust_alg,
                    hclust_opt = hclust_opt, type = NCO(; opt_kwargs = (; type = RP())))
    w1t = [0.09306804224438855, 2.965538175980197e-9, 5.120616924704069e-10,
           0.012498074010119991, 3.7922350595309275e-11, 4.389967746479165e-12,
           1.571554053547651e-10, 0.01890604423249791, 1.3485844134776075e-10,
           8.710330716011039e-11, 0.4415479849421009, 0.05315150363540031,
           3.889652755318321e-21, 3.4792816837581925e-11, 2.5797454703026665e-11,
           1.6944084292719147e-11, 0.02308802678879938, 1.0073993561399218e-10,
           0.29524210224876657, 0.06249821782062269]
    w2t = [0.09810134108090621, 0.011388753754271286, 6.437047367157054e-10,
           0.011075489907070782, 3.148550335615237e-11, 1.4799393254155678e-11,
           1.318931439966537e-10, 0.01841991202305588, 1.0435131278098484e-10,
           8.614741934900651e-11, 0.41790394179337464, 0.05030533996568464,
           1.4970245155556828e-20, 3.110891490720659e-11, 3.954733749975306e-11,
           1.675812001360921e-11, 0.029370582327353862, 2.2607295727840045e-10,
           0.279432445332238, 0.08400219249017586]
    w3t = [8.193729881939037e-10, 2.409987309001213e-9, 0.5264670437187233,
           2.517270977151848e-10, 0.32265001048427105, 3.8303851614533396e-20,
           5.6956445630163935e-9, 0.05614503811357049, 1.0037399347394833e-17,
           8.813801105337669e-9, 2.179845581501531e-9, 1.786411989931377e-18,
           1.0632090753928366e-10, 3.451301121073617e-18, 2.064195656295622e-18,
           0.09473786627587708, 1.5456125575809285e-8, 5.141044640297729e-10,
           4.000999373837102e-9, 1.1596291575584218e-9]
    w4t = [1.8648307834296094e-8, 2.0121129931139206e-8, 1.2732227882968977e-7,
           2.248819943852892e-8, 8.505310254468229e-5, 8.750827164319476e-16,
           0.9999131027072671, 3.013461011003404e-8, 9.978547267173414e-13,
           1.6150100575048547e-14, 8.277584052238747e-14, 9.184146757828695e-16,
           4.191095842512731e-15, 1.4250045646643808e-15, 5.48091714601181e-16,
           1.1339473486081746e-6, 3.3121983434285156e-8, 1.4447566564553116e-8,
           4.2555952904300184e-7, 1.839812964815229e-8]
    w5t = [0.05659875480807075, 0.05765686028748787, 0.04814562158947152,
           0.032543952358064475, 0.031507038395257314, 0.037360792840002946,
           0.02039180126487961, 0.0618246311824005, 0.06456187917628958,
           0.08093640733937797, 0.10365488779001777, 0.02906988764065906,
           0.02760574340527159, 0.052682257268469534, 0.017789884077172368,
           0.06786092277422026, 0.043104486102002494, 0.047256307941542106,
           0.07426879053007172, 0.045179093229270634]
    @test isapprox(w46.weights, w1t)
    @test isapprox(w47.weights, w2t)
    @test isapprox(w48.weights, w3t)
    @test isapprox(w49.weights, w4t)
    @test isapprox(w50.weights, w5t, rtol = 5.0e-5)

    rm = ADD()
    w51 = optimise!(portfolio; rm = rm, cluster = false, hclust_alg = hclust_alg,
                    hclust_opt = hclust_opt, type = NCO(; opt_kwargs = (; obj = MinRisk())))
    w52 = optimise!(portfolio; rm = rm, cluster = false, hclust_alg = hclust_alg,
                    hclust_opt = hclust_opt,
                    type = NCO(; opt_kwargs = (; obj = Utility(; l = l))))
    w53 = optimise!(portfolio; rm = rm, cluster = false, hclust_alg = hclust_alg,
                    hclust_opt = hclust_opt,
                    type = NCO(; opt_kwargs = (; obj = Sharpe(; rf = rf))))
    w54 = optimise!(portfolio; rm = rm, cluster = false, hclust_alg = hclust_alg,
                    hclust_opt = hclust_opt, type = NCO(; opt_kwargs = (; obj = MaxRet())))
    w55 = optimise!(portfolio; rm = rm, cluster = false, hclust_alg = hclust_alg,
                    hclust_opt = hclust_opt, type = NCO(; opt_kwargs = (; type = RP())))
    w1t = [4.858382022674378e-10, 0.053481853549220186, 0.12301036094781877,
           1.1572606679018196e-10, 0.04087812887359, 0.002519516122229487,
           0.00042880092524383425, 0.13309997405160942, 6.61472699038919e-10,
           2.3028686401319528e-9, 0.09131414340068031, 0.017720396275592475,
           1.939245823037983e-18, 0.030718209317324146, 6.280352017591961e-11,
           1.049984884248537e-9, 0.18695553460116063, 0.03885091677025856,
           0.1588747870328061, 0.1221473734537721]
    w2t = [4.480528858292739e-10, 0.0564096307290648, 0.12339719843847467,
           7.733392551531556e-11, 0.04562833019774078, 0.002410781036645187,
           0.0008040452742058028, 0.13166519284950062, 8.034784301568693e-10,
           1.2569589667840443e-8, 0.08737372027691452, 0.016955689339267627,
           5.984793645609076e-18, 0.029392551406924643, 1.3827803193332287e-10,
           5.753665011415402e-9, 0.18848310019120762, 0.037265002327532964,
           0.1520188951238704, 0.12819584301825235]
    w3t = [1.02062048455541e-9, 0.0397811102944998, 0.18540374464442913,
           1.07360599890668e-9, 0.16404765384401954, 9.053644415456022e-11,
           0.03456210501430832, 0.030821893166634237, 2.849030118902176e-10,
           6.553423268609104e-9, 0.018734132135076267, 1.1980574003163722e-10,
           6.892277210265349e-11, 1.966174767706324e-10, 7.538937153221054e-12,
           0.12494570939916305, 0.2110201401051579, 2.550949677130291e-9,
           0.06327027556315082, 0.127413223866637]
    w4t = [4.9820377641708154e-8, 5.448876375975373e-8, 1.1275893036261378e-7,
           6.041464983120543e-8, 0.00029562183155109137, 1.335587798324464e-13,
           0.9996693811475731, 3.922730079076675e-8, 3.3886517178782046e-11,
           1.1415914845860577e-11, 3.4714407547192225e-12, 1.348472107004945e-13,
           2.5854586251445246e-12, 1.9381150104653553e-13, 8.26353796979984e-14,
           2.760589173723265e-5, 1.2528267644249627e-7, 3.8541662805900445e-8,
           6.862903785415222e-6, 4.763908748236392e-8]
    w5t = [0.03941964489419902, 0.05878789380633039, 0.06336122678465943,
           0.02644427349083025, 0.04079783708796677, 0.025118942413556285,
           0.02001067299965853, 0.06274910163100314, 0.053259359498068336,
           0.08018126523435438, 0.0694165490338133, 0.021434679840840902,
           0.01606475726803423, 0.0381974113604881, 0.01001038091191075,
           0.07031962100674591, 0.07179495883084511, 0.05835523485418758,
           0.0924829280168134, 0.08179326103569423]
    @test isapprox(w51.weights, w1t)
    @test isapprox(w52.weights, w2t)
    @test isapprox(w53.weights, w3t)
    @test isapprox(w54.weights, w4t)
    @test isapprox(w55.weights, w5t, rtol = 0.0001)

    rm = CDaR()
    w56 = optimise!(portfolio; rm = rm, cluster = false, hclust_alg = hclust_alg,
                    hclust_opt = hclust_opt, type = NCO(; opt_kwargs = (; obj = MinRisk())))
    w57 = optimise!(portfolio; rm = rm, cluster = false, hclust_alg = hclust_alg,
                    hclust_opt = hclust_opt,
                    type = NCO(; opt_kwargs = (; obj = Utility(; l = l))))
    w58 = optimise!(portfolio; rm = rm, cluster = false, hclust_alg = hclust_alg,
                    hclust_opt = hclust_opt,
                    type = NCO(; opt_kwargs = (; obj = Sharpe(; rf = rf))))
    w59 = optimise!(portfolio; rm = rm, cluster = false, hclust_alg = hclust_alg,
                    hclust_opt = hclust_opt, type = NCO(; opt_kwargs = (; obj = MaxRet())))
    w60 = optimise!(portfolio; rm = rm, cluster = false, hclust_alg = hclust_alg,
                    hclust_opt = hclust_opt, type = NCO(; opt_kwargs = (; type = RP())))
    w1t = [0.011004045505975613, 0.028251183077981752, 0.014422806875229753,
           4.958526061344831e-12, 0.009984293676628531, 2.3450739422880165e-11,
           6.031901028176002e-12, 0.09832765390526335, 1.2991801782199283e-10,
           2.2291710613941976e-10, 0.24797181174174213, 0.03612792745275501,
           1.056769979910633e-20, 0.03754084144061125, 1.8643272263494993e-11,
           5.689309987552144e-11, 0.09739889660050584, 0.03323021449469062,
           0.2194393677703557, 0.16630095699544786]
    w2t = [0.0071197508044408345, 0.024699895722335642, 0.018315740975714777,
           7.151885014623421e-12, 0.008356484733585173, 1.9089526941426207e-11,
           8.18850804197304e-12, 0.0962626345923765, 1.4039247614012587e-10,
           0.0001446322897575077, 0.25239579994504224, 0.036772480103140416,
           1.5733180507973053e-14, 0.03821064073749522, 3.108510515937393e-11,
           3.6913199117386076e-5, 0.09920775900938193, 0.02887885603246143,
           0.2233543807284419, 0.16624403092078577]
    w3t = [2.86011130169537e-10, 7.279305864441031e-10, 0.13990028972380286,
           5.829103756078371e-11, 0.33756478162542447, 1.4168493517274713e-19,
           3.0069380177645344e-10, 0.2116115968069058, 2.183525693820217e-18,
           6.38781643234779e-9, 6.179520517424919e-10, 4.2408209986768714e-19,
           1.2329702172663889e-10, 6.815529955852682e-19, 5.692773425625955e-19,
           0.24570860427285218, 5.763612679405827e-9, 1.5603214133107417e-10,
           1.4350232532914917e-9, 0.0652147117143545]
    w4t = [1.4702102374546112e-8, 1.6162920494912485e-8, 6.002515523179071e-8,
           1.7472955087571066e-8, 5.886754774299193e-5, 1.2612840120920024e-15,
           0.9999340453627157, 1.4420797125010015e-8, 3.9098351660212424e-13,
           7.836497422018281e-14, 4.745002163035154e-14, 1.3315777617840017e-15,
           1.8301083230580586e-14, 1.891380812749e-15, 7.840115067588502e-16,
           5.655799059022777e-6, 5.258192981293426e-8, 1.1388574409740394e-8,
           1.2300009429028432e-6, 1.4534564600702838e-8]
    w5t = [0.044621929656327136, 0.05974669646641222, 0.04673199751476315,
           0.02202711999783714, 0.034928496248394274, 0.030602363998901337,
           0.014815135143633454, 0.07371498589589642, 0.060020783382639276,
           0.08929648987351317, 0.09357047699806313, 0.021425090153187682,
           0.023300153316325536, 0.052589757331902416, 0.014189312841471253,
           0.0698335955428971, 0.046206226891693514, 0.04462051809754638,
           0.08090941082100037, 0.07684945982759507]
    @test isapprox(w56.weights, w1t)
    @test isapprox(w57.weights, w2t)
    @test isapprox(w58.weights, w3t)
    @test isapprox(w59.weights, w4t)
    @test isapprox(w60.weights, w5t, rtol = 5.0e-5)

    rm = UCI()
    w61 = optimise!(portfolio; rm = rm, cluster = false, hclust_alg = hclust_alg,
                    hclust_opt = hclust_opt, type = NCO(; opt_kwargs = (; obj = MinRisk())))
    w62 = optimise!(portfolio; rm = rm, cluster = false, hclust_alg = hclust_alg,
                    hclust_opt = hclust_opt,
                    type = NCO(; opt_kwargs = (; obj = Utility(; l = l))))
    w63 = optimise!(portfolio; rm = rm, cluster = false, hclust_alg = hclust_alg,
                    hclust_opt = hclust_opt,
                    type = NCO(; opt_kwargs = (; obj = Sharpe(; rf = rf))))
    w64 = optimise!(portfolio; rm = rm, cluster = false, hclust_alg = hclust_alg,
                    hclust_opt = hclust_opt, type = NCO(; opt_kwargs = (; obj = MaxRet())))
    w65 = optimise!(portfolio; rm = rm, cluster = false, hclust_alg = hclust_alg,
                    hclust_opt = hclust_opt, type = NCO(; opt_kwargs = (; type = RP())))
    w1t = [6.339221068451245e-10, 0.04651817931744516, 0.11017171572957145,
           2.3217212095726008e-12, 0.033480752472860545, 0.0017843135154844325,
           5.8154031303529526e-11, 0.13022303335424218, 7.373381378336765e-10,
           1.1215711483301387e-9, 0.11774178258081092, 0.02223575914998876,
           7.341828128086774e-19, 0.043728900575733115, 1.0805077202807164e-10,
           4.553292084477986e-10, 0.16441923747282433, 0.039638908228437054,
           0.15238444697953965, 0.13767296750637523]
    w2t = [5.814772490139419e-10, 0.04733086058026881, 0.11032398217999186,
           7.911405751610749e-11, 0.0354118560544708, 0.0010377953218527578,
           7.406664618709402e-11, 0.12696018297146391, 5.092177808027191e-10,
           1.107907204934204e-9, 0.12267498220182824, 0.022906839231032764,
           1.1268406472864974e-18, 0.045767441267552396, 1.255645628894548e-10,
           4.497819877549505e-10, 0.15935092313972446, 0.03748067867036363,
           0.15748652668727153, 0.13326792876704946]
    w3t = [8.828039841744664e-10, 3.9890928005496905e-9, 0.21226108486649664,
           4.1713658186614307e-10, 0.23673047785519769, 5.31587609110726e-17,
           0.0048039761977919015, 0.14638945477130516, 1.9486228895037198e-16,
           7.819440549878561e-9, 1.7393651143857764e-8, 1.0399149605948582e-16,
           1.4235659372375644e-10, 1.2848385093506377e-16, 1.5415688482482198e-17,
           0.17050521930616136, 0.14821155354495932, 1.2548664876774095e-9,
           5.439085290965003e-8, 0.08109814716788621]
    w4t = [5.995207906173523e-8, 6.50199354244668e-8, 1.2333856267120065e-7,
           7.21540711292467e-8, 0.00028068604118484296, 2.976124635366768e-13,
           0.9996800252578011, 4.7195114552088204e-8, 7.140974557928591e-11,
           6.564785028259721e-12, 1.328849866404678e-11, 3.081157588741628e-13,
           1.9906229052224903e-12, 4.2254457795983015e-13, 2.1273367817217905e-13,
           2.903657834747259e-5, 1.4092370999366812e-7, 4.646053553777311e-8,
           9.63956639466522e-6, 5.7417768867966126e-8]
    w5t = [0.042621616770917956, 0.05926643442160448, 0.05891985619040837,
           0.023392613122276185, 0.03818231982232573, 0.028188514974334427,
           0.017007819570827774, 0.07027563215510384, 0.055681856290461615,
           0.08452765001572406, 0.0785253874525583, 0.020242825802686264,
           0.018215706072420134, 0.04174002486233878, 0.011185883589400338,
           0.07256987057081475, 0.054917332512358356, 0.05039480814874462,
           0.09140224204282815, 0.0827416056118659]
    @test isapprox(w61.weights, w1t)
    @test isapprox(w62.weights, w2t)
    @test isapprox(w63.weights, w3t)
    @test isapprox(w64.weights, w4t)
    @test isapprox(w65.weights, w5t, rtol = 0.0001)

    rm = EDaR()
    w66 = optimise!(portfolio; rm = rm, cluster = false, hclust_alg = hclust_alg,
                    hclust_opt = hclust_opt, type = NCO(; opt_kwargs = (; obj = MinRisk())))
    w67 = optimise!(portfolio; rm = rm, cluster = false, hclust_alg = hclust_alg,
                    hclust_opt = hclust_opt,
                    type = NCO(; opt_kwargs = (; obj = Utility(; l = l))))
    w68 = optimise!(portfolio; rm = rm, cluster = false, hclust_alg = hclust_alg,
                    hclust_opt = hclust_opt,
                    type = NCO(; opt_kwargs = (; obj = Sharpe(; rf = rf))))
    w69 = optimise!(portfolio; rm = rm, cluster = false, hclust_alg = hclust_alg,
                    hclust_opt = hclust_opt, type = NCO(; opt_kwargs = (; obj = MaxRet())))
    w70 = optimise!(portfolio; rm = rm, cluster = false, hclust_alg = hclust_alg,
                    hclust_opt = hclust_opt, type = NCO(; opt_kwargs = (; type = RP())))
    w1t = [0.0726312361265007, 0.033384488741601014, 5.103751149075006e-9,
           3.014518528946845e-10, 4.724506670658462e-9, 1.253054866996639e-9,
           2.205095228542095e-10, 0.08141932914872951, 3.6881820699728994e-9,
           0.014557007444885766, 0.30653075374144134, 0.03842013687740744,
           8.468257870550366e-13, 0.005408493904766368, 6.545441982099049e-11,
           0.00395626991922374, 0.0855973005973881, 0.027809798353575423,
           0.25683830395676144, 0.0734468658299618]
    w2t = [0.07188718659948104, 0.03231379397613412, 7.862356773595368e-9,
           3.304805536050558e-10, 5.767616571417321e-9, 2.062324338537918e-9,
           2.3934067182581034e-10, 0.0806054983964276, 6.055078255784101e-9,
           0.015565533049874125, 0.3095095853423995, 0.03864336921086802,
           3.77609702984125e-12, 0.004388544698817729, 8.319748741533143e-11,
           0.004260470801984409, 0.08586493178147266, 0.02554875960399025,
           0.25989466505205777, 0.07151763908232191]
    w3t = [6.093553106747296e-9, 1.272321786628477e-8, 0.27138040478969777,
           2.0182828498984105e-9, 0.32321508213237843, 9.69583013515116e-17,
           5.175335729487244e-9, 0.18025501290038848, 2.685327727945996e-16,
           8.047029989704588e-9, 8.307726394072881e-9, 1.2217500016022025e-16,
           3.251856167488813e-10, 1.4737564848479143e-16, 2.4867715817468116e-17,
           0.22514939802530334, 2.5334475905700583e-8, 4.060556815673562e-9,
           1.7026345176235333e-8, 1.3040521844213984e-8]
    w4t = [3.2180585982358536e-8, 3.4792072301916835e-8, 1.969392408013349e-7,
           3.867877460090811e-8, 0.0001500291497766489, 6.176351673914708e-16,
           0.9998467424145034, 3.33470605537889e-8, 1.324537001996197e-13,
           2.1097260351535536e-14, 2.445689262829524e-14, 6.466977169358908e-16,
           3.3579852244224604e-15, 9.733564009400666e-16, 3.8534927984276727e-16,
           1.9407013400120445e-6, 5.301555337937153e-8, 2.4899700085263258e-8,
           8.425325908851574e-7, 3.134861737584489e-8]
    w5t = [0.04974172130529782, 0.05733881414227115, 0.050193916021615895,
           0.024534811168682846, 0.03152704024343209, 0.03377228228437697,
           0.017031241107275923, 0.08007434154998308, 0.05828257100946734,
           0.08635967294626462, 0.10082262710759267, 0.022763953645553357,
           0.024761931829072263, 0.05119987144439959, 0.015212168140680805,
           0.06700267065826623, 0.04438899718688257, 0.048204655053479324,
           0.07874539707869038, 0.058041316076715144]
    @test isapprox(w66.weights, w1t, rtol = 0.0001)
    @test isapprox(w67.weights, w2t, rtol = 5.0e-5)
    @test isapprox(w68.weights, w3t, rtol = 5.0e-6)
    @test isapprox(w69.weights, w4t)
    @test isapprox(w70.weights, w5t, rtol = 1.0e-5)

    rm = RDaR()
    w71 = optimise!(portfolio; rm = rm, cluster = false, hclust_alg = hclust_alg,
                    hclust_opt = hclust_opt, type = NCO(; opt_kwargs = (; obj = MinRisk())))
    w72 = optimise!(portfolio; rm = rm, cluster = false, hclust_alg = hclust_alg,
                    hclust_opt = hclust_opt,
                    type = NCO(; opt_kwargs = (; obj = Utility(; l = l))))
    w73 = optimise!(portfolio; rm = rm, cluster = false, hclust_alg = hclust_alg,
                    hclust_opt = hclust_opt,
                    type = NCO(; opt_kwargs = (; obj = Sharpe(; rf = rf))))
    w74 = optimise!(portfolio; rm = rm, cluster = false, hclust_alg = hclust_alg,
                    hclust_opt = hclust_opt, type = NCO(; opt_kwargs = (; obj = MaxRet())))
    w75 = optimise!(portfolio; rm = rm, cluster = false, hclust_alg = hclust_alg,
                    hclust_opt = hclust_opt, type = NCO(; opt_kwargs = (; type = RP())))
    w1t = [0.08378986882917888, 0.014222281895084562, 1.4640996523981236e-9,
           7.944954882144362e-11, 9.314212297173224e-10, 9.137860256305013e-11,
           1.8245207791188445e-10, 0.05641447999298819, 4.5648697820494774e-10,
           0.016986220828475156, 0.36206160895224154, 0.04411178251130963,
           1.1583898248023923e-12, 2.1145226385254962e-10, 1.7184793041218614e-11,
           0.004536260375593694, 0.07530115554166356, 0.020468216399785202,
           0.27991576632666104, 0.04219235491193489]
    w2t = [0.08360220460231052, 0.013583941175582762, 3.328795807287142e-9,
           1.87637917696734e-10, 2.2529865423853684e-9, 1.4232546558507974e-10,
           5.772690975589539e-10, 0.05733549479288826, 7.495380224782043e-10,
           0.016330706513896252, 0.36302988958155197, 0.0442327939969194,
           2.342722863753897e-12, 3.371032260258337e-10, 3.742234048075495e-11,
           0.004391597514939037, 0.07796335375207128, 0.017645665083984987,
           0.28088239780352764, 0.04100194756690676]
    w3t = [1.3515804012708479e-8, 2.2763927813695442e-8, 0.40422535276540883,
           4.209877377089907e-9, 0.3228420613543988, 9.382007684504622e-17,
           8.095862116122522e-9, 0.11630847933030657, 2.534082850464609e-16,
           9.151566224458459e-9, 7.013242330109768e-9, 1.212367854592803e-16,
           3.30324485838165e-10, 1.453262655206002e-16, 2.602876462067114e-17,
           0.15662397918383003, 2.8333194936724385e-8, 7.55051684815826e-9,
           1.2915452762775174e-8, 1.3486286320212747e-8]
    w4t = [1.7112615903132398e-8, 1.9024128895262603e-8, 8.647128018693435e-8,
           2.074472300676763e-8, 5.638464427967927e-5, 1.4958618984198002e-15,
           0.9999376754888423, 1.9517941371648137e-8, 2.8227576374538363e-13,
           5.238654504944856e-14, 5.833839585906032e-14, 1.55417915111153e-15,
           8.633146151296927e-15, 2.2806487733165333e-15, 9.281470736582364e-16,
           4.351938694057633e-6, 4.3900989019528236e-8, 1.32625726321481e-8,
           1.3507587848113968e-6, 1.7134740265231196e-8]
    w5t = [0.0560731811241318, 0.056704867975412944, 0.05028698047424797,
           0.028411275849565994, 0.031438244608776644, 0.03500694355093412,
           0.019634352772634988, 0.07479997517345825, 0.060089392631509704,
           0.08300309103129284, 0.10109742147029181, 0.026075081772755374,
           0.025610504206661496, 0.05006530433128356, 0.016088633338122934,
           0.06695558429275707, 0.04410955238183049, 0.05010270397113676,
           0.07553122561040937, 0.048915683432785906]
    @test isapprox(w71.weights, w1t, rtol = 5.0e-6)
    @test isapprox(w72.weights, w2t, rtol = 1.0e-6)
    @test isapprox(w73.weights, w3t, rtol = 5.0e-5)
    @test isapprox(w74.weights, w4t, rtol = 5.0e-8)
    @test isapprox(w75.weights, w5t, rtol = 5.0e-7)

    rm = Kurt()
    w76 = optimise!(portfolio; rm = rm, cluster = false, hclust_alg = hclust_alg,
                    hclust_opt = hclust_opt, type = NCO(; opt_kwargs = (; obj = MinRisk())))
    w77 = optimise!(portfolio; rm = rm, cluster = false, hclust_alg = hclust_alg,
                    hclust_opt = hclust_opt,
                    type = NCO(; opt_kwargs = (; obj = Utility(; l = l))))
    w78 = optimise!(portfolio; rm = rm, cluster = false, hclust_alg = hclust_alg,
                    hclust_opt = hclust_opt,
                    type = NCO(; opt_kwargs = (; obj = Sharpe(; rf = rf))))
    w79 = optimise!(portfolio; rm = rm, cluster = false, hclust_alg = hclust_alg,
                    hclust_opt = hclust_opt, type = NCO(; opt_kwargs = (; obj = MaxRet())))
    w80 = optimise!(portfolio; rm = rm, cluster = false, hclust_alg = hclust_alg,
                    hclust_opt = hclust_opt, type = NCO(; opt_kwargs = (; type = RP())))
    w1t = [2.5964437003610723e-8, 0.05105919106221655, 3.467171217276289e-8,
           0.052000269807801806, 2.0285416272445106e-8, 0.07566811426967662,
           0.0033034082607419535, 0.10667421688825035, 8.976013131180923e-8,
           2.4171113199481545e-8, 0.24860400348802075, 0.019037062444765506,
           1.5750411960203906e-9, 0.0975551387747684, 0.00019572743265402633,
           7.1657474133847724e-9, 0.026531138487239897, 0.18667818859089166,
           0.030269149949859745, 0.10242418694951412]
    w2t = [3.953279520738277e-9, 0.00608819450228333, 0.04888864431745161,
           0.03295859019620168, 0.31265900510311656, 2.238222706366946e-12,
           0.08633806381834545, 0.0778256129435594, 2.7427326002014e-11,
           0.036740103394666095, 0.0003690172576887811, 2.434913299244783e-12,
           3.4263659558102143e-10, 5.507335911411046e-12, 1.2895587770429228e-12,
           0.015108385527328301, 0.3825640663960669, 4.1130118915494924e-8,
           0.0004602340413968748, 3.703696274306287e-8]
    w3t = [2.9598248638838536e-8, 0.052132577592122115, 0.0066097662323385455,
           0.05284859892679505, 0.19124176638059454, 3.480278905600524e-10,
           0.05283328141583669, 0.1548121674960447, 5.916601577362687e-9,
           0.014889206408669914, 0.039372299120703544, 4.090534381910998e-10,
           5.66228430288467e-11, 1.0642901651394402e-9, 1.7585538169352688e-10,
           0.007156534493788735, 0.23161347982429942, 0.1157559447850554,
           0.049891598775869773, 0.030842740979181667]
    w4t = [8.005212570019472e-9, 8.467651966208832e-9, 1.0409572690089251e-8,
           9.543499190499942e-9, 8.295037015919763e-7, 1.3864613933349198e-17,
           0.9999990580669282, 6.007004624742323e-9, 1.6875079758803041e-15,
           5.123664164923582e-16, 3.1369581021316564e-17, 1.479521351471662e-17,
           9.819487055093616e-17, 2.0663973250659508e-17, 9.593483918575565e-18,
           2.6526189478950564e-8, 1.0708614452849208e-8, 6.2904542992933856e-9,
           1.889382404852362e-8, 7.577344478069048e-9]
    w5t = [0.03168837196142575, 0.037466468353004136, 0.031234374626791957,
           0.034549860557835925, 0.02792805782006853, 0.05605746091442722,
           0.023582207306295348, 0.057710978979585055, 0.04214105060168191,
           0.14962449271688594, 0.08550672279401512, 0.03422542718867126,
           0.04716059334901583, 0.056821918435025244, 0.032524555724384924,
           0.07606724180403476, 0.037144680642580016, 0.05150614556134447,
           0.0457072853415405, 0.041352105321386175]
    @test isapprox(w76.weights, w1t)
    @test isapprox(w77.weights, w2t, rtol = 1.0e-5)
    @test isapprox(w78.weights, w3t, rtol = 5.0e-7)
    @test isapprox(w79.weights, w4t)
    @test isapprox(w80.weights, w5t, rtol = 0.0005)

    w81 = optimise!(portfolio; rm = rm, cluster = false, hclust_alg = hclust_alg,
                    hclust_opt = hclust_opt,
                    type = NCO(; port_kwargs = (; max_num_assets_kurt = 1),
                               opt_kwargs = (; obj = MinRisk())))
    w82 = optimise!(portfolio; rm = rm, cluster = false, hclust_alg = hclust_alg,
                    hclust_opt = hclust_opt,
                    type = NCO(; port_kwargs = (; max_num_assets_kurt = 1),
                               opt_kwargs = (; obj = Utility(; l = l))))
    w83 = optimise!(portfolio; rm = rm, cluster = false, hclust_alg = hclust_alg,
                    hclust_opt = hclust_opt,
                    type = NCO(; port_kwargs = (; max_num_assets_kurt = 1),
                               opt_kwargs = (; obj = Sharpe(; rf = rf))))
    w84 = optimise!(portfolio; rm = rm, cluster = false, hclust_alg = hclust_alg,
                    hclust_opt = hclust_opt,
                    type = NCO(; port_kwargs = (; max_num_assets_kurt = 1),
                               opt_kwargs = (; obj = MaxRet())))
    w85 = optimise!(portfolio; rm = rm, cluster = false, hclust_alg = hclust_alg,
                    hclust_opt = hclust_opt,
                    type = NCO(; port_kwargs = (; max_num_assets_kurt = 1),
                               opt_kwargs = (; type = RP())))
    w1t = [4.727734278507026e-7, 0.006370060826488009, 5.590615606579003e-7,
           0.022998554451883063, 4.441627320343446e-7, 0.013284150263804119,
           5.970490492158233e-7, 0.0838931742703437, 2.412597531934661e-6,
           0.028510160124660774, 0.39492774392365976, 3.0722599041649877e-6,
           0.0018580695417469018, 0.057307904829193484, 1.8574497963603708e-6,
           0.008452191534703831, 1.3349875580765725e-6, 0.29222362353377535,
           3.998796068464042e-6, 0.09015961756211228]
    w2t = [4.3990242011970524e-7, 2.68235933214361e-6, 0.015110805656676652,
           0.009958675957496851, 0.3029373494428155, 8.039949480792974e-9,
           0.07776805252660571, 0.04001839753856603, 6.656269293972443e-8,
           0.03482818348818368, 0.013959764437193727, 9.339642428603718e-9,
           2.370312473712396e-8, 1.931902916975401e-8, 5.170246727299067e-9,
           0.01432251436244497, 0.4749552504521827, 2.2524168627126217e-6,
           0.01613343955720331, 2.059767330569216e-6]
    w3t = [4.322981465908775e-8, 0.014279088725597362, 6.820123885843372e-7,
           0.045925731089117114, 0.19013221470570524, 4.2567479660387834e-10,
           0.04520685479375434, 0.15571106714099356, 3.693702001115422e-9,
           0.016647289867665293, 0.05440784065269151, 4.748212271845846e-10,
           3.5938039941799915e-11, 1.0287431862135368e-9, 2.2432366681170457e-10,
           0.008001676389674889, 0.302875793180106, 0.09936564449080217,
           0.06032660878980307, 0.007119459048683397]
    w4t = [1.4257790350208626e-7, 1.501901652202739e-7, 1.8207593705908865e-7,
           1.678371539142802e-7, 1.0924852196866645e-5, 6.828857577556047e-14,
           0.9999864170854778, 1.0997181195175208e-7, 7.2415276505681555e-12,
           1.0320229965507217e-12, 1.5463236063731608e-13, 7.481093944549463e-14,
           1.9558227015596087e-13, 1.0424092698175282e-13, 4.836269527424305e-14,
           9.1384537765449e-7, 1.8900370311272044e-7, 1.1468234362776633e-7,
           5.517095420553719e-7, 1.3615946791535004e-7]
    w5t = [0.03157505508763204, 0.037473833994199506, 0.031164352830716866,
           0.03450873395242757, 0.027881506416966306, 0.05590865992464623,
           0.02352762384702192, 0.05776656715170088, 0.042027202277771854,
           0.14954092268057667, 0.086427388245954, 0.03414306069677871,
           0.047136137544626214, 0.05678369997557464, 0.03244108703071972,
           0.07602516032528432, 0.037016336561212136, 0.051840096275088836,
           0.045501808443562414, 0.04131076673753918]
    @test isapprox(w81.weights, w1t)
    @test isapprox(w82.weights, w2t)
    @test isapprox(w83.weights, w3t, rtol = 5.0e-7)
    @test isapprox(w84.weights, w4t)
    @test isapprox(w85.weights, w5t, rtol = 5.0e-5)

    rm = SKurt()
    w86 = optimise!(portfolio; rm = rm, cluster = false, hclust_alg = hclust_alg,
                    hclust_opt = hclust_opt, type = NCO(; opt_kwargs = (; obj = MinRisk())))
    w87 = optimise!(portfolio; rm = rm, cluster = false, hclust_alg = hclust_alg,
                    hclust_opt = hclust_opt,
                    type = NCO(; opt_kwargs = (; obj = Utility(; l = l))))
    w88 = optimise!(portfolio; rm = rm, cluster = false, hclust_alg = hclust_alg,
                    hclust_opt = hclust_opt,
                    type = NCO(; opt_kwargs = (; obj = Sharpe(; rf = rf))))
    w89 = optimise!(portfolio; rm = rm, cluster = false, hclust_alg = hclust_alg,
                    hclust_opt = hclust_opt, type = NCO(; opt_kwargs = (; obj = MaxRet())))
    w90 = optimise!(portfolio; rm = rm, cluster = false, hclust_alg = hclust_alg,
                    hclust_opt = hclust_opt, type = NCO(; opt_kwargs = (; type = RP())))
    w1t = [7.551916314181692e-9, 0.08663346505996936, 4.136135363410196e-9,
           0.005142916560864078, 4.818370239459004e-9, 0.08295324360196754,
           3.871908480663506e-9, 0.10161306822488796, 2.29356993539279e-8,
           1.0037393362885237e-8, 0.24803693892730824, 0.015639781543736138,
           6.322005863046181e-10, 0.11800230131930538, 3.9490169313941265e-7,
           3.2331232148000544e-9, 0.02791297634122693, 0.22534657397026278,
           2.663260304644401e-6, 0.08871561907172698]
    w2t = [1.648507594855836e-9, 7.106093993965583e-9, 2.6821898563545493e-9,
           1.7545831990829108e-9, 0.6557595753775312, 2.1295202363763833e-17,
           0.11251956687680768, 2.1063621510857507e-9, 1.5771333214892832e-16,
           1.0774661320180257e-8, 4.6619729840497305e-9, 2.4874285544821247e-17,
           2.5431768681261486e-17, 4.859461690491088e-17, 1.2603971741332548e-17,
           6.488225359675868e-9, 0.2317208043270109, 2.3331588025652267e-9,
           1.1862146674966285e-8, 2.00074810762526e-9]
    w3t = [5.993027021019961e-8, 0.09420488273482014, 7.120316107104118e-8,
           1.0079237479976533e-7, 0.329256358792681, 7.141475440137918e-15,
           0.03140252925073406, 0.14456953987300492, 6.169747768331338e-14,
           1.3915512545670295e-5, 5.019541284270236e-7, 9.156922521884978e-15,
           7.842948155187376e-14, 2.0719874961296496e-14, 3.4714079165152544e-15,
           9.11055470989817e-6, 0.256154299347926, 0.14438723907581813,
           7.795770495741892e-7, 6.114005955833654e-7]
    w4t = [2.2584896800420005e-9, 2.3868994846955344e-9, 2.935218991892342e-9,
           2.7071881211103927e-9, 1.8108760930357788e-7, 9.121782217691314e-18,
           0.9999997816102653, 1.710765968766494e-9, 1.103542029729593e-15,
           1.8029071739145351e-16, 2.1210296214776463e-17, 9.92890257439193e-18,
           3.540313902777023e-17, 1.3940277230261195e-17, 6.320073872927453e-18,
           1.1006635562617111e-8, 3.022630420456309e-9, 1.7903861779628587e-9,
           7.330238630607758e-9, 2.15367117108765e-9]
    w5t = [0.03371262104831552, 0.0388620407559719, 0.029516120865239113,
           0.03149690911309137, 0.029034868047156617, 0.05575791265829153,
           0.02201380907911114, 0.0549030438553775, 0.04043465605632881, 0.1317092632242419,
           0.08283775502355642, 0.033223382918018206, 0.059150908015103934,
           0.05744126469981375, 0.036058022488107334, 0.08781271342160586,
           0.03715886265808057, 0.05201465381699867, 0.04553081997712453,
           0.04133037227846529]
    @test isapprox(w86.weights, w1t)
    @test isapprox(w87.weights, w2t)
    @test isapprox(w88.weights, w3t, rtol = 1.0e-6)
    @test isapprox(w89.weights, w4t)
    @test isapprox(w90.weights, w5t, rtol = 0.0005)

    w91 = optimise!(portfolio; rm = rm, cluster = false, hclust_alg = hclust_alg,
                    hclust_opt = hclust_opt,
                    type = NCO(; port_kwargs = (; max_num_assets_kurt = 1),
                               opt_kwargs = (; obj = MinRisk())))
    w92 = optimise!(portfolio; rm = rm, cluster = false, hclust_alg = hclust_alg,
                    hclust_opt = hclust_opt,
                    type = NCO(; port_kwargs = (; max_num_assets_kurt = 1),
                               opt_kwargs = (; obj = Utility(; l = l))))
    w93 = optimise!(portfolio; rm = rm, cluster = false, hclust_alg = hclust_alg,
                    hclust_opt = hclust_opt,
                    type = NCO(; port_kwargs = (; max_num_assets_kurt = 1),
                               opt_kwargs = (; obj = Sharpe(; rf = rf))))
    w94 = optimise!(portfolio; rm = rm, cluster = false, hclust_alg = hclust_alg,
                    hclust_opt = hclust_opt,
                    type = NCO(; port_kwargs = (; max_num_assets_kurt = 1),
                               opt_kwargs = (; obj = MaxRet())))
    w95 = optimise!(portfolio; rm = rm, cluster = false, hclust_alg = hclust_alg,
                    hclust_opt = hclust_opt,
                    type = NCO(; port_kwargs = (; max_num_assets_kurt = 1),
                               opt_kwargs = (; type = RP())))
    w1t = [1.152779741425078e-6, 0.08722539114523502, 6.854407426076873e-7,
           3.068918009983621e-5, 7.947208980494585e-7, 0.07717690266402272,
           6.435899245721492e-7, 0.0992603766984917, 3.1849804524079635e-6,
           9.959820314972761e-6, 0.26146441197895043, 0.013901477451006361,
           6.275239789520174e-7, 0.11800807686124645, 7.900563306723513e-6,
           3.2082426080004494e-6, 0.01639944828857723, 0.23943934954364213,
           1.773320106936254e-5, 0.08704798532569098]
    w2t = [4.799202175901544e-7, 1.8280944270898892e-6, 7.595365144944402e-7,
           5.085253781995741e-7, 0.6551703729606982, 3.466969933911942e-12,
           0.11226680205762056, 6.061781116266855e-7, 2.737695046663116e-11,
           7.0071951311506616e-6, 2.965757363371859e-6, 4.278542317628517e-12,
           2.9672655930878333e-12, 8.338079411439662e-12, 2.1209149820549077e-12,
           4.219685395680112e-6, 0.23253561330191852, 6.697314693652168e-7,
           7.5909005770071146e-6, 5.7610662828549e-7]
    w3t = [1.4518351245675877e-7, 0.09197889683503516, 1.6088413279446193e-7,
           2.164447588017958e-7, 0.3284649568935163, 3.0229792699402393e-15,
           0.030240702590955418, 0.14449323297302258, 2.238936275130768e-14,
           1.18838878980427e-5, 3.416196988117059e-7, 3.742661555350317e-15,
           1.0318659559836205e-13, 7.713455523976681e-15, 1.538036493396456e-15,
           7.780594389307788e-6, 0.2610768094494007, 0.14372329309308265,
           5.195789822544405e-7, 1.0599714729475227e-6]
    w4t = [1.425655589790209e-7, 1.5017716054062437e-7, 1.8206021615075052e-7,
           1.6782265374424046e-7, 1.092403664876886e-5, 6.828252078066556e-14,
           0.9999864181337756, 1.0996221181681709e-7, 7.240886787846119e-12,
           1.0319293597830236e-12, 1.546186577305212e-13, 7.480430987253226e-14,
           1.9556440822813114e-13, 1.0423168172753032e-13, 4.835841039747908e-14,
           9.137650148276677e-7, 1.8898737032469956e-7, 1.1467234132136319e-7,
           5.516604676940009e-7, 1.361476615343836e-7]
    w5t = [0.03371286032622024, 0.03885323133887401, 0.029510251830648974,
           0.03149106262531467, 0.02902215708107005, 0.05575554476634823,
           0.022009754271010297, 0.05490862119132229, 0.040420250028205616,
           0.13170745728250258, 0.08289599265231477, 0.03321679831097424,
           0.0591519685748792, 0.0574450639058379, 0.03604162471190545, 0.08781851736381807,
           0.037163179992558235, 0.05202355312248844, 0.045529962818564744,
           0.04132214780514196]
    @test isapprox(w91.weights, w1t)
    @test isapprox(w92.weights, w2t)
    @test isapprox(w93.weights, w3t, rtol = 5.0e-6)
    @test isapprox(w94.weights, w4t)
    @test isapprox(w95.weights, w5t, rtol = 5.0e-5)

    rm = Skew()
    w96 = optimise!(portfolio; rm = rm, cluster = false, hclust_alg = hclust_alg,
                    hclust_opt = hclust_opt, type = NCO(; opt_kwargs = (; obj = MinRisk())))
    w97 = optimise!(portfolio; rm = rm, cluster = false, hclust_alg = hclust_alg,
                    hclust_opt = hclust_opt,
                    type = NCO(; opt_kwargs = (; obj = Utility(; l = l))))
    w98 = optimise!(portfolio; rm = rm, cluster = false, hclust_alg = hclust_alg,
                    hclust_opt = hclust_opt,
                    type = NCO(; opt_kwargs = (; obj = Sharpe(; rf = rf))))
    w99 = optimise!(portfolio; rm = rm, cluster = false, hclust_alg = hclust_alg,
                    hclust_opt = hclust_opt, type = NCO(; opt_kwargs = (; obj = MaxRet())))
    w100 = optimise!(portfolio; rm = rm, cluster = false, hclust_alg = hclust_alg,
                     hclust_opt = hclust_opt, type = NCO(; opt_kwargs = (; type = RP())))
    w1t = [9.512843515038461e-7, 0.02679372963723403, 7.481688741710024e-8,
           8.677981868535554e-8, 0.00024374710251416658, 1.6424163064416956e-10,
           9.925303140364328e-8, 0.018337706834624035, 4.199681081018799e-10,
           0.044847315863413596, 2.880663555289258e-5, 1.6699881321927315e-7,
           0.8850048893895833, 9.222351514908303e-10, 1.29959155424575e-5,
           1.2870890276640751e-6, 1.9217385711076424e-7, 0.02235979951041215,
           2.3395793459459278e-5, 0.002344753415432111]
    w2t = [2.892066058287496e-12, 6.021136122811683e-12, 1.6595810841870378e-11,
           1.2518518518980756e-11, 0.7759709237514477, 3.261400626300735e-22,
           0.22402907609740005, 1.1209584266409124e-11, 5.2093420567391296e-20,
           1.211847041348591e-21, 5.555320744491218e-22, 2.463831250808569e-22,
           1.936357517318602e-22, 5.1125064758334945e-23, 3.530451287716877e-22,
           5.1336508446549383e-11, 1.9893431894515837e-11, 9.98151946790108e-12,
           2.0436673289400196e-11, 2.671258551762943e-13]
    w3t = [1.0310068513022651e-9, 0.04247516664987991, 3.385746665892482e-10,
           4.4084678285666815e-10, 0.3154602214189142, 2.1215416105607568e-19,
           2.7889852555415977e-9, 3.2946681133034345e-9, 2.40803099868887e-18,
           0.14355268678846272, 1.6790419477890336e-18, 2.860421097787122e-19,
           2.7315391405613452e-11, 5.216927699995645e-19, 1.8561334858300276e-19,
           0.49370302263693644, 1.1329559074445164e-9, 0.004808890578177884,
           1.5115378626916289e-9, 1.3617380041860026e-9]
    w4t = [6.1346171755586554e-9, 6.599014059111091e-9, 8.090716526383227e-9,
           7.396914264079482e-9, 4.896724588071184e-7, 7.36517620191656e-18,
           0.9999994421566508, 4.737854348345277e-9, 8.928320134773704e-16,
           6.371336604145683e-17, 1.810543949163722e-17, 7.707398990930562e-18,
           1.3862608503512784e-17, 1.1718109870037684e-17, 5.412442015541036e-18,
           1.015661101908201e-8, 8.50118026514471e-9, 4.925362320042241e-9,
           5.8237163204458354e-9, 5.804903047063175e-9]
    w5t = [0.02587283980010857, 0.03264463657770643, 0.014158174674160097,
           0.015798390683037792, 0.021562504541245746, 0.019702542832291847,
           0.011175680915325443, 0.04053333087337281, 0.02613557862432773,
           0.13685040056428732, 0.05399833667801738, 0.01325291505118253,
           0.3224398305305101, 0.03177231426544304, 0.03083051325239583,
           0.07583058468110429, 0.02019743390256818, 0.039927409011560895,
           0.03821626595613802, 0.029100316585215984]
    @test isapprox(w96.weights, w1t)
    @test isapprox(w97.weights, w2t)
    @test isapprox(w98.weights, w3t)
    @test isapprox(w99.weights, w4t)
    @test isapprox(w100.weights, w5t, rtol = 1.0e-7)

    rm = SSkew()
    w101 = optimise!(portfolio; rm = rm, cluster = false, hclust_alg = hclust_alg,
                     hclust_opt = hclust_opt,
                     type = NCO(; opt_kwargs = (; obj = MinRisk())))
    w102 = optimise!(portfolio; rm = rm, cluster = false, hclust_alg = hclust_alg,
                     hclust_opt = hclust_opt,
                     type = NCO(; opt_kwargs = (; obj = Utility(; l = l))))
    w103 = optimise!(portfolio; rm = rm, cluster = false, hclust_alg = hclust_alg,
                     hclust_opt = hclust_opt,
                     type = NCO(; opt_kwargs = (; obj = Sharpe(; rf = rf))))
    w104 = optimise!(portfolio; rm = rm, cluster = false, hclust_alg = hclust_alg,
                     hclust_opt = hclust_opt, type = NCO(; opt_kwargs = (; obj = MaxRet())))
    w105 = optimise!(portfolio; rm = rm, cluster = false, hclust_alg = hclust_alg,
                     hclust_opt = hclust_opt, type = NCO(; opt_kwargs = (; type = RP())))
    w1t = [1.0892430594871065e-6, 0.031681259608003894, 6.385841396167424e-7,
           1.0208999440372263e-6, 3.6209703749382284e-6, 9.63334977859699e-6,
           5.195032617345709e-6, 0.16790560775238103, 1.6957722439003348e-6,
           4.9105984264575884e-6, 0.32690571795454637, 0.0064336998061633,
           1.1748062417479142e-8, 0.0780837391349781, 8.910688360428937e-7,
           1.284611349160674e-6, 2.3889090964349337e-6, 0.2998587061455184,
           9.651938161685354e-6, 0.08908923687231873]
    w2t = [8.15581000775344e-12, 2.1122924956231996e-11, 8.468440027459267e-11,
           6.290782166354827e-11, 0.7906420617811222, 3.770466458146047e-21,
           0.20935793754156667, 7.732354627171445e-11, 5.613257710234412e-19,
           2.4647140872454105e-19, 7.937696118180958e-21, 3.0134893967449533e-21,
           3.3302575799562404e-20, 7.080800527624176e-22, 4.270636058014689e-21,
           2.0043380508726013e-10, 9.698265399054417e-11, 5.504855806811231e-11,
           6.228588325811293e-11, 8.36586909877896e-12]
    w3t = [3.7861438714093765e-10, 7.882775220627101e-10, 4.736696170348332e-10,
           3.718007853005574e-10, 0.7671857043598811, 4.915030082168658e-19,
           0.07214396701662078, 8.737251526759338e-10, 6.322008478869382e-18,
           0.02001621273852103, 8.063255787638913e-18, 5.530049009448801e-19,
           3.6820467470084965e-11, 1.1316668921573435e-18, 2.8096469370461567e-19,
           0.14065410623907945, 3.1380239299922003e-9, 9.710437538871152e-10,
           2.04328809940668e-9, 5.706340470933306e-10]
    w4t = [4.808971411822285e-9, 5.0889067778196335e-9, 6.296951167461911e-9,
           5.8264815984649646e-9, 3.6863311384148367e-7, 2.074772259621311e-16,
           0.9999994962043447, 3.680607079860537e-9, 2.4513295260098905e-14,
           2.7849215145189317e-15, 4.759207596372842e-16, 2.1346320729387184e-16,
           4.421019750282232e-16, 3.178085787059933e-16, 1.3941144415954463e-16,
           5.739878651350115e-8, 6.535615364916281e-9, 3.8593171144659175e-9,
           3.7086577022239837e-8, 4.580298321968766e-9]
    w5t = [0.0339264619542522, 0.038638756632197464, 0.03015595919880873,
           0.029485590325139945, 0.03172755919967277, 0.05301160166381721,
           0.02380386625242652, 0.06207621701820662, 0.040253889426890024,
           0.12465166616658238, 0.09072083579161366, 0.031034596824263225,
           0.04653553578551491, 0.05952392664850132, 0.03166034045137151,
           0.08586716691912194, 0.0376304878982509, 0.058963225878939075,
           0.04715290372556418, 0.043179412238865456]
    @test isapprox(w101.weights, w1t, rtol = 5.0e-8)
    @test isapprox(w102.weights, w2t)
    @test isapprox(w103.weights, w3t)
    @test isapprox(w104.weights, w4t)
    @test isapprox(w105.weights, w5t, rtol = 5.0e-6)

    portfolio = HCPortfolio(; prices = prices[(end - 25):end],
                            solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                             :params => Dict("verbose" => false,
                                                                             "max_step_fraction" => 0.75))))

    asset_statistics!(portfolio)
    hclust_alg = HAC()
    rm = DVar()
    w106 = optimise!(portfolio; rm = rm, cluster = true, hclust_alg = hclust_alg,
                     hclust_opt = hclust_opt,
                     type = NCO(; opt_kwargs = (; obj = MinRisk())))
    w107 = optimise!(portfolio; rm = rm, cluster = false, hclust_alg = hclust_alg,
                     hclust_opt = hclust_opt,
                     type = NCO(; opt_kwargs = (; obj = Utility(; l = l))))
    w108 = optimise!(portfolio; rm = rm, cluster = false, hclust_alg = hclust_alg,
                     hclust_opt = hclust_opt,
                     type = NCO(; opt_kwargs = (; obj = Sharpe(; rf = rf))))
    w109 = optimise!(portfolio; rm = rm, cluster = false, hclust_alg = hclust_alg,
                     hclust_opt = hclust_opt, type = NCO(; opt_kwargs = (; obj = MaxRet())))
    w110 = optimise!(portfolio; rm = rm, cluster = false, hclust_alg = hclust_alg,
                     hclust_opt = hclust_opt, type = NCO(; opt_kwargs = (; type = RP())))
    w1t = [6.834840718346733e-15, 2.12661660856925e-13, 1.593754486266787e-8,
           3.127395868133335e-15, 2.712330218049574e-15, 1.4202097663079517e-7,
           0.011911258450762945, 0.08622274106594424, 6.220288317998889e-15,
           1.5954206758683104e-8, 0.5915112370358865, 1.270102811146502e-7,
           2.798803537238241e-9, 0.12907377152965582, 5.382298008325306e-10,
           4.833357981034712e-8, 5.7197896167821745e-8, 0.18128028398701804,
           3.069005217470668e-14, 2.9813895163443943e-7]
    w2t = [6.087104999022211e-11, 6.944744755943647e-11, 3.263961815450165e-17,
           2.6519049543234252e-11, 9.866211341474767e-11, 9.058094855826689e-10,
           5.489577389446619e-17, 4.138342706591664e-9, 7.494642823615315e-11,
           0.38935388178047176, 1.8060001347245284e-8, 1.3887640961260867e-9,
           0.25563209903556644, 4.882594874351539e-8, 5.206559677256252e-17,
           1.0012989237254488e-9, 6.504136027864861e-11, 0.02296138843873617,
           6.173439151305568e-11, 0.33205255596783845]
    w3t = [6.99927459172966e-13, 6.351719788969979e-13, 2.1243277127479536e-22,
           6.139347729676585e-13, 6.850737459567674e-13, 1.0400658443295194e-11,
           2.174169572685971e-22, 2.2775316023728395e-11, 7.122348068461382e-13,
           2.0708953804686408e-10, 2.3833834551615515e-11, 1.7237103485625813e-11,
           0.7895758048443616, 1.313760737481902e-10, 1.5378284119183503e-22,
           1.3897709953031573e-11, 6.997871517832699e-13, 3.449570741368316e-11,
           6.80238557191563e-13, 0.21042419468980597]
    w4t = [1.8453590228798314e-14, 3.6180280497773147e-14, 7.8985614511812e-14,
           1.9647369528216504e-14, 1.7956248044052224e-14, 3.7312789562441875e-8,
           3.561384400477929e-14, 4.600416928385553e-8, 1.834673946393253e-14,
           6.0139985334607e-8, 4.555393762100468e-8, 4.373109505228164e-8,
           0.9999990477978071, 2.930842604452552e-7, 6.945292174907096e-13,
           4.136941245290735e-8, 3.309976190391047e-14, 4.9824511263715374e-8,
           2.9235556315226516e-14, 3.3518104994705405e-7]
    w5t = [0.035115457178974414, 0.04430473724919071, 0.06147944704688781,
           0.029650594706968226, 0.03213846887456715, 0.052933228620072874,
           0.0662082270653987, 0.058398810568459386, 0.03536903358021121,
           0.042837814742357594, 0.07939049207576095, 0.040167522274443784,
           0.019064516076979485, 0.11336435302557887, 0.044398381724289546,
           0.04225460418266382, 0.04025282445992181, 0.06571777187539703,
           0.04012680907502655, 0.056826905596850164]
    @test isapprox(w106.weights, w1t)
    @test isapprox(w107.weights, w2t)
    @test isapprox(w108.weights, w3t)
    @test isapprox(w109.weights, w4t)
    @test isapprox(w110.weights, w5t, rtol = 1.0e-5)
end

@testset "HRP" begin
    portfolio = HCPortfolio(; prices = prices,
                            solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                             :params => Dict("verbose" => false,
                                                                             "max_step_fraction" => 0.75))))

    asset_statistics!(portfolio)
    hclust_alg = DBHT()
    hclust_opt = HCType()
    type = HRP()
    w1 = optimise!(portfolio; rm = SD(), cluster = true, type = type,
                   hclust_alg = hclust_alg, hclust_opt = hclust_opt)
    wt = [0.03692879524929352, 0.06173471900996101, 0.05788485449055379,
          0.02673494374797732, 0.051021461747188426, 0.06928891082994745,
          0.024148698656475776, 0.047867702815293824, 0.03125471206249845,
          0.0647453556976089, 0.10088112288028349, 0.038950479132407664,
          0.025466032983548218, 0.046088041928035575, 0.017522279227008376,
          0.04994166864441962, 0.076922254387969, 0.05541444481863126, 0.03819947378487138,
          0.07900404790602693]
    @test isapprox(w1.weights, wt)

    w2 = optimise!(portfolio; rm = MAD(), cluster = false, type = type,
                   hclust_alg = hclust_alg, hclust_opt = hclust_opt)
    wt = [0.03818384697661099, 0.0622147651480733, 0.06006655971419491,
          0.025623934919921716, 0.05562388497425944, 0.0707725175955155,
          0.025813212789699617, 0.0505495225915194, 0.030626076629624386,
          0.06238080360591265, 0.09374966122500297, 0.04170897361395994, 0.0238168503353984,
          0.04472819614772452, 0.016127737407569318, 0.05166088174144377,
          0.07470415476814536, 0.05398093542616027, 0.03829328495650196, 0.0793741994327616]
    @test isapprox(w2.weights, wt)

    w3 = optimise!(portfolio; rm = SSD(), cluster = false, type = type,
                   hclust_alg = hclust_alg, hclust_opt = hclust_opt)
    wt = [0.03685118468964678, 0.060679678711961914, 0.05670047906355952,
          0.026116775736648376, 0.05374470957245603, 0.07054365460247647,
          0.025708179197816795, 0.04589886455582043, 0.03028786387008358,
          0.0634613352386214, 0.10260291820653494, 0.038697426540672014,
          0.027823289120792207, 0.04746469792344547, 0.01859405658396672,
          0.052617311519562025, 0.07250127037171229, 0.05655029966386585,
          0.03752315313496762, 0.07563285169538961]
    @test isapprox(w3.weights, wt)

    w4 = optimise!(portfolio; rm = FLPM(; target = rf), cluster = false, type = type,
                   hclust_alg = hclust_alg, hclust_opt = hclust_opt)
    wt = [0.039336538617540336, 0.06395492117890568, 0.06276820986537449,
          0.026249574972701337, 0.06242254734451266, 0.06144221460126541,
          0.026935168312631458, 0.05003386507226852, 0.031591718694359686,
          0.062032838812258684, 0.08971950983943798, 0.03794006123084816,
          0.021868414535268534, 0.04077527283143069, 0.014126622482894119,
          0.053377350594462684, 0.07977242881987609, 0.05414859107042937,
          0.04026174326124142, 0.0812424078622927]
    @test isapprox(w4.weights, wt)

    w5 = optimise!(portfolio; rm = SLPM(; target = rf), cluster = false, type = type,
                   hclust_alg = hclust_alg, hclust_opt = hclust_opt)
    wt = [0.03752644591926642, 0.06162667128035028, 0.058061904976983145,
          0.02658920517037755, 0.056905956197316004, 0.06572780540966205,
          0.026284445157011994, 0.04570757302464124, 0.030832527351205462,
          0.06360209500884424, 0.10038131818271241, 0.03703703080055575,
          0.02666888092950761, 0.04516285946254531, 0.01726932945234202,
          0.053613204742961086, 0.07517335594118413, 0.05681780552493213,
          0.03852165318888371, 0.07648993227871743]
    @test isapprox(w5.weights, wt)

    w6 = optimise!(portfolio; rm = WR(), cluster = false, type = type,
                   hclust_alg = hclust_alg, hclust_opt = hclust_opt)
    wt = [0.05025200880802368, 0.055880855583038826, 0.05675550980104009,
          0.03045034754117226, 0.046067750416348405, 0.054698528388037695,
          0.02387947919504387, 0.03257444044988314, 0.02852023603441257,
          0.06948149861799151, 0.10142635777300404, 0.02403801691456373, 0.0376186609725536,
          0.0489137277286356, 0.023542345142850748, 0.05182849839922806,
          0.12037415047550741, 0.06259799738419287, 0.03040731610608419,
          0.050692274268387696]
    @test isapprox(w6.weights, wt)

    w7 = optimise!(portfolio; rm = CVaR(), cluster = false, type = type,
                   hclust_alg = hclust_alg, hclust_opt = hclust_opt)
    wt = [0.03557588495882196, 0.06070721124578364, 0.055279033097091264,
          0.02659326453716366, 0.053554687039556674, 0.06617470037072191,
          0.0262740407048504, 0.04720203604560211, 0.02912774720142737, 0.06321093829691052,
          0.10241165286325946, 0.03903551330484258, 0.029865777325086773,
          0.04683826432227621, 0.01880807994439806, 0.05468860348477102,
          0.07181506647247264, 0.0597967463434774, 0.035892100831017446,
          0.07714865161046885]
    @test isapprox(w7.weights, wt)

    w8 = optimise!(portfolio; rm = EVaR(), cluster = false, type = type,
                   hclust_alg = hclust_alg, hclust_opt = hclust_opt)
    wt = [0.04298217683837017, 0.058906176713889875, 0.0574016331486885,
          0.03020450525284909, 0.0498349903258857, 0.060682817777209165,
          0.024220257509343413, 0.03449220187183893, 0.028790726943045256,
          0.06651915441075427, 0.1102093216711328, 0.026320946186565213, 0.0350510342557288,
          0.04934072008028962, 0.021946092606624574, 0.05048203603428413,
          0.09622970122852144, 0.06481150812979522, 0.03384174474956397,
          0.057732254265619835]
    @test isapprox(w8.weights, wt, rtol = 5.0e-7)

    w9 = optimise!(portfolio; rm = RVaR(), cluster = false, type = type,
                   hclust_alg = hclust_alg, hclust_opt = hclust_opt)
    wt = [0.04746219128450875, 0.05772052034931984, 0.057894023300070915,
          0.030899831490444826, 0.047780697436069414, 0.056496204023026735,
          0.02359615954618457, 0.032640369565769795, 0.028784216596900798,
          0.06836029362987105, 0.10450059856679983, 0.02457130647918195,
          0.03686881087305981, 0.048996934209302734, 0.0227965377657541,
          0.05134221596812637, 0.11116746180000622, 0.06399587050591148,
          0.031665735616057476, 0.052460020993633505]
    @test isapprox(w9.weights, wt, rtol = 1.0e-7)

    w10 = optimise!(portfolio; rm = MDD(), cluster = false, type = type,
                    hclust_alg = hclust_alg, hclust_opt = hclust_opt)
    wt = [0.06921730086136976, 0.04450738292959614, 0.08140658224086754,
          0.018672719501981218, 0.053316921973365496, 0.02979463558079051,
          0.021114260172381653, 0.029076952717003283, 0.026064071405076886,
          0.07316497416971823, 0.11123520904148268, 0.018576814647316645,
          0.013991436918012332, 0.05469682002741476, 0.012327710930761682,
          0.08330184420498327, 0.07332958900751246, 0.060105980885656114,
          0.049926342635393445, 0.07617245014931591]
    @test isapprox(w10.weights, wt)

    w11 = optimise!(portfolio; rm = ADD(), cluster = false, type = type,
                    hclust_alg = hclust_alg, hclust_opt = hclust_opt)
    wt = [0.05548273429645013, 0.03986073873648723, 0.09599318831154405,
          0.014621119350699974, 0.08212398553813219, 0.024721555415213262,
          0.017757328803291575, 0.024648542319161523, 0.04053046304894661,
          0.06980177121920025, 0.07760681699761883, 0.01027851528431043,
          0.00777588366272761, 0.021591614519396285, 0.0034064806929941845,
          0.06991915064946408, 0.12960468534402678, 0.062161540752826025,
          0.08184195485513887, 0.07027193020237012]
    @test isapprox(w11.weights, wt)

    w12 = optimise!(portfolio; rm = CDaR(), cluster = false, type = type,
                    hclust_alg = hclust_alg, hclust_opt = hclust_opt)
    wt = [0.06282520890507177, 0.03706824571002148, 0.08139181669684024,
          0.015787409071561908, 0.05742532060680956, 0.022473205432788673,
          0.018010510386343107, 0.028690055814249292, 0.031947471924376705,
          0.07945270069786842, 0.10413378630449394, 0.015162575734861551,
          0.011642750066910313, 0.04691738037936903, 0.008558622422573802,
          0.08506702725550051, 0.07867872568683776, 0.06853164241553955,
          0.06271444623431467, 0.08352109825366769]
    @test isapprox(w12.weights, wt)

    w13 = optimise!(portfolio; rm = UCI(), cluster = false, type = type,
                    hclust_alg = hclust_alg, hclust_opt = hclust_opt)
    wt = [0.056222127504695026, 0.03676156065449049, 0.09101083070457834,
          0.014331694009042086, 0.06981598298437515, 0.02449761185808867,
          0.017540488246254482, 0.025783651111609372, 0.03897771757340407,
          0.07792753098849892, 0.09069642716675666, 0.011419250355786286,
          0.009579825814905836, 0.03155851948684042, 0.005020862393829734,
          0.07891456352044066, 0.10481408027151863, 0.06296146942913426, 0.0767246147903293,
          0.0754411911354216]
    @test isapprox(w13.weights, wt)

    w14 = optimise!(portfolio; rm = EDaR(), cluster = false, type = type,
                    hclust_alg = hclust_alg, hclust_opt = hclust_opt)
    wt = [0.06368313885096026, 0.03964177645523946, 0.08253885356726247,
          0.01658630133298382, 0.055878822739450135, 0.025248628049312456,
          0.019039909877211127, 0.028999411295580237, 0.030719786342128182,
          0.07837842859043447, 0.10479226890649017, 0.01598505374262056,
          0.012462054946453145, 0.0503667610779172, 0.009778781140263914,
          0.08402144368015278, 0.07578137459372361, 0.0651773824278426, 0.05986313372528741,
          0.0810566886586861]
    @test isapprox(w14.weights, wt, rtol = 1.0e-7)

    w15 = optimise!(portfolio; rm = RDaR(), cluster = false, type = type,
                    hclust_alg = hclust_alg, hclust_opt = hclust_opt)
    wt = [0.06626057100148959, 0.042384460290778865, 0.08229559342069401,
          0.017571400016583083, 0.054533052228267126, 0.02751643858660513,
          0.02021038180000024, 0.02882973184000429, 0.028405177937391907,
          0.0760417757563507, 0.10819526671824652, 0.01714761122061998,
          0.013186013461541538, 0.05250570179214284, 0.01096704678455384,
          0.08422872826614602, 0.07476238567443566, 0.0621153409498924,
          0.054747390549717716, 0.07809593170453868]
    @test isapprox(w15.weights, wt)

    w16 = optimise!(portfolio; rm = Kurt(), cluster = false, type = type,
                    hclust_alg = hclust_alg, hclust_opt = hclust_opt)
    wt = [0.029481521953684118, 0.05982735944410474, 0.03557911826015237,
          0.028474368121041888, 0.025828635808911045, 0.05929340372085539,
          0.00400811258716055, 0.04856579375773367, 0.028712479333209244,
          0.0657391879397804, 0.16216057544981644, 0.011489226379288793,
          0.009094526296241098, 0.08053893478316498, 0.012724404070883372,
          0.021335119185650623, 0.0855020913192837, 0.11267092582579596,
          0.03851809818436267, 0.08045611757887891]
    @test isapprox(w16.weights, wt)

    w17 = optimise!(portfolio; rm = SKurt(), cluster = false, type = type,
                    hclust_alg = hclust_alg, hclust_opt = hclust_opt)
    wt = [0.041695764288341326, 0.04968730030116381, 0.04519046227953908,
          0.02228610752801017, 0.036916599549008534, 0.06695518729855642,
          0.008601057788011419, 0.03832875671733724, 0.023603809923922775,
          0.05953659154802947, 0.15949334007294313, 0.010345061127586078,
          0.01551106764201746, 0.07844578132980796, 0.01457102407508962,
          0.034738374409788796, 0.09106192787421133, 0.1097918256490279, 0.0352173258407637,
          0.058022634756843806]
    @test isapprox(w17.weights, wt)

    w18 = optimise!(portfolio; rm = Skew(), cluster = false, type = type,
                    hclust_alg = hclust_alg, hclust_opt = hclust_opt)
    wt = [4.18164303203965e-6, 0.11905148244884775, 0.13754035964680367,
          4.18164303203965e-6, 0.13754035964680367, 1.658199358060021e-6,
          0.028721218707620826, 1.1383557170375464e-10, 8.355416012626526e-16,
          1.8128635217880256e-6, 0.08532736996421339, 0.007291462719384035,
          0.10926054871992093, 0.00021830908182616843, 0.2370505968817962,
          0.10926054871992093, 0.028721218707620826, 3.684313367457248e-6,
          4.824826616435831e-11, 1.0059308456231142e-6]
    @test isapprox(w18.weights, wt, rtol = 5.0e-5)

    w19 = optimise!(portfolio; rm = SSkew(), cluster = false, type = type,
                    hclust_alg = hclust_alg, hclust_opt = hclust_opt)
    wt = [0.03073561997154973, 0.06271788007944147, 0.059839304178611136,
          0.019020703463231065, 0.054114343500047124, 0.07478195087196789,
          0.015025906923294915, 0.037423537539572865, 0.02195663364580543,
          0.07120487867873436, 0.14238956696393482, 0.022992175030255378,
          0.02427916555268004, 0.040541588877064376, 0.010704183865328891,
          0.053000134123666075, 0.07802885174200914, 0.06680453879901624,
          0.030056876315924044, 0.08438215987786517]
    @test isapprox(w19.weights, wt)

    hclust_alg = HAC()
    w20 = optimise!(portfolio; rm = DVar(), cluster = true, type = type,
                    hclust_alg = hclust_alg, hclust_opt = hclust_opt)
    wt = [0.034616203287809656, 0.05339682894540566, 0.0277781553250953,
          0.04211296639608739, 0.05429852976387857, 0.07556907274344477,
          0.012918417022839884, 0.0639503232740957, 0.025141332254925024,
          0.05293567401349993, 0.12067634196508455, 0.0237681969964296,
          0.008823211105214881, 0.08942720087787832, 0.016861747398793574,
          0.03288067246055758, 0.07170919770494971, 0.0812323656620027, 0.03793954600227262,
          0.07396401679973474]
    @test isapprox(w20.weights, wt)

    w21 = optimise!(portfolio; rm = Variance(), cluster = false, type = type,
                    hclust_alg = hclust_alg, hclust_opt = hclust_opt)
    wt = [0.03360421525593201, 0.053460257098811775, 0.027429766590708997,
          0.04363053745921338, 0.05279180956461212, 0.07434468966041922,
          0.012386194792256387, 0.06206960160806503, 0.025502890164538234,
          0.0542097204834031, 0.12168116639250848, 0.023275086688903004,
          0.009124639465879256, 0.08924750757276853, 0.017850423121104797,
          0.032541204698588386, 0.07175228284814082, 0.08318399209117079,
          0.03809545426566615, 0.07381856017730955]
    @test isapprox(w21.weights, wt)

    w22 = optimise!(portfolio; rm = Equal(), cluster = false, type = type,
                    hclust_alg = hclust_alg, hclust_opt = hclust_opt)
    wt = [0.05, 0.05000000000000001, 0.05, 0.04999999999999999, 0.04999999999999999,
          0.05000000000000001, 0.05000000000000001, 0.05, 0.05, 0.05000000000000001,
          0.04999999999999999, 0.04999999999999999, 0.05, 0.04999999999999999, 0.05,
          0.04999999999999999, 0.04999999999999999, 0.05, 0.05, 0.04999999999999999]
    @test isapprox(w22.weights, wt)

    w23 = optimise!(portfolio; rm = VaR(), cluster = false, type = type,
                    hclust_alg = hclust_alg, hclust_opt = hclust_opt)
    wt = [0.03403717939776512, 0.05917176810341441, 0.029281627698577888,
          0.04641795773733506, 0.06002080678226616, 0.07336909341225178,
          0.032392046758139795, 0.05383361136188931, 0.02967264551919885,
          0.05284095110693102, 0.09521120378350553, 0.04261320776949539,
          0.01642971834566978, 0.07971670573840162, 0.021147864266696195,
          0.050803448662520866, 0.06563107532995796, 0.05200824347813839,
          0.034112945793535646, 0.07128789895430926]
    @test isapprox(w23.weights, wt)

    w24 = optimise!(portfolio; rm = DaR(), cluster = false, type = type,
                    hclust_alg = hclust_alg, hclust_opt = hclust_opt)
    wt = [0.057660532702867646, 0.04125282312548106, 0.05922785740065619,
          0.03223509018499126, 0.10384639449297382, 0.02480551517770064,
          0.030513023620218654, 0.037564939241009405, 0.02260728843785307,
          0.05164502511939685, 0.08286147806113833, 0.01696931547695469,
          0.00418787314771699, 0.06308172456210165, 0.007685315296496237,
          0.0694006227527249, 0.09119842783020647, 0.09248808273195344,
          0.044542968010494795, 0.06622570262706394]
    @test isapprox(w24.weights, wt)

    w25 = optimise!(portfolio; rm = DaR_r(), cluster = false, type = type,
                    hclust_alg = hclust_alg, hclust_opt = hclust_opt)
    wt = [0.05820921825149978, 0.04281381004716568, 0.061012991673039425,
          0.037261284520695284, 0.10225280827507012, 0.03171774621579708,
          0.029035913030571646, 0.03777811595747374, 0.021478656618637317,
          0.05210098246314737, 0.08336111731195082, 0.0224554330786605,
          0.008018398540074134, 0.060923985355596684, 0.009489132849269694,
          0.06345531419794387, 0.0883323204747975, 0.08122137323769983,
          0.039962166984027804, 0.06911923091688174]
    @test isapprox(w25.weights, wt)

    w26 = optimise!(portfolio; rm = MDD_r(), cluster = false, type = type,
                    hclust_alg = hclust_alg, hclust_opt = hclust_opt)
    wt = [0.054750291482807016, 0.05626561538767235, 0.044927537922610235,
          0.037029800949262545, 0.06286000734501392, 0.038964408950160866,
          0.033959752237812064, 0.04828571977339982, 0.01937785058895953,
          0.05279062837229332, 0.10144994531515807, 0.02678668518638289,
          0.010335110450233378, 0.07434568383344507, 0.01192345099055306,
          0.05711412672965212, 0.06785463405921893, 0.09152771438398988,
          0.032677346602089874, 0.07677368943928506]
    @test isapprox(w26.weights, wt)

    w27 = optimise!(portfolio; rm = ADD_r(), cluster = false, type = type,
                    hclust_alg = hclust_alg, hclust_opt = hclust_opt)
    wt = [0.05216718533133331, 0.04708278318245931, 0.07158823871210457,
          0.029511984859220904, 0.12867669205129983, 0.03859596234440916,
          0.022402839001206605, 0.026020360199894607, 0.030904469636691148,
          0.054239105004183275, 0.07390114503735276, 0.019880342325388017,
          0.00476824739544596, 0.03991945523395664, 0.0054480039590591904,
          0.059645486519252305, 0.11611402694084015, 0.06374794181935534,
          0.059893035563107475, 0.055492694883439435]
    @test isapprox(w27.weights, wt)

    w28 = optimise!(portfolio; rm = CDaR_r(), cluster = false, type = type,
                    hclust_alg = hclust_alg, hclust_opt = hclust_opt)
    wt = [0.055226953695215866, 0.047934679854074826, 0.05395157314076186,
          0.036982877818185954, 0.08463665680704126, 0.033820937721103415,
          0.03026563101495272, 0.04118023926554901, 0.02122427296619385,
          0.05288666614109493, 0.09090289747478623, 0.02380032651263032,
          0.008673469487824291, 0.06662055705080387, 0.010138402407857017,
          0.060505269563432176, 0.08319151689594376, 0.08557053458440436,
          0.038056791259321536, 0.07442974633882277]
    @test isapprox(w28.weights, wt)

    w29 = optimise!(portfolio; rm = UCI_r(), cluster = false, type = type,
                    hclust_alg = hclust_alg, hclust_opt = hclust_opt)
    wt = [0.05276898612268049, 0.04614128625526376, 0.06552577565455027, 0.0320518739448942,
          0.1111735166748811, 0.039154188455666976, 0.025443440854672504,
          0.03239581981059534, 0.027120969094009423, 0.05445935205465669,
          0.08028963951603105, 0.02075298500986429, 0.0058131044419531056,
          0.04948420047533672, 0.006850658262213453, 0.06288836730781266,
          0.1025592250558978, 0.07454814435900761, 0.050977607512044276,
          0.05960085913796832]
    @test isapprox(w29.weights, wt, rtol = 1.0e-7)

    w30 = optimise!(portfolio; rm = EDaR_r(), cluster = false, type = type,
                    hclust_alg = hclust_alg, hclust_opt = hclust_opt)
    wt = [0.055560528232119676, 0.05252278252766485, 0.04995930852013699,
          0.036469686665161705, 0.07360443788124264, 0.035484250062529706,
          0.03187063749286862, 0.04362148704778408, 0.021541875055852516,
          0.05435689568806405, 0.09401418942704692, 0.024881512156616555,
          0.009270444002026255, 0.06971378686582017, 0.01078203039784278,
          0.05939090630801511, 0.07729538472655453, 0.08736656785459875,
          0.03800312958595158, 0.07429015950210269]
    @test isapprox(w30.weights, wt, rtol = 1.0e-7)

    w31 = optimise!(portfolio; rm = RDaR_r(), cluster = false, type = type,
                    hclust_alg = hclust_alg, hclust_opt = hclust_opt)
    wt = [0.055263520718878155, 0.05470886033764548, 0.04729258737064928,
          0.03659066826504964, 0.06763240307582324, 0.03707186354483175,
          0.03301695115922358, 0.04590027827321353, 0.020741988491439888,
          0.05389295853521278, 0.09778300148886963, 0.025642106200246102,
          0.00971015663966544, 0.07161140382368784, 0.011262111599036555,
          0.05878170332530692, 0.07257021270894684, 0.08962515902013113,
          0.035691384041053154, 0.07521068138108908]
    @test isapprox(w31.weights, wt, rtol = 5.0e-8)
end

@testset "HERC" begin
    portfolio = HCPortfolio(; prices = prices,
                            solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                             :params => Dict("verbose" => false,
                                                                             "max_step_fraction" => 0.75))))

    asset_statistics!(portfolio)
    hclust_alg = DBHT()
    hclust_opt = HCType()
    type = HERC()
    w1 = optimise!(portfolio; rm = SD(), cluster = true, type = type,
                   hclust_alg = hclust_alg, hclust_opt = hclust_opt)
    wt = [0.07698450455880468, 0.07670235654605057, 0.06955326092716153,
          0.055733645924582964, 0.061306348146154536, 0.026630781912285295,
          0.0274779863650835, 0.08723325075122691, 0.021443967931440197,
          0.050919570266336714, 0.034721621146346957, 0.013406113465936382,
          0.018014864090601306, 0.029736261019180265, 0.01130547202588631,
          0.03532911363416089, 0.08752722816710831, 0.10098629923304404,
          0.026208793387783046, 0.08877856050082561]
    @test isapprox(w1.weights, wt)

    w2 = optimise!(portfolio; rm = MAD(), cluster = false, type = type,
                   hclust_alg = hclust_alg, hclust_opt = hclust_opt)
    wt = [0.0797145490968226, 0.07455683155558127, 0.070334401927198, 0.05349383521987859,
          0.06513229159697068, 0.026966237124386894, 0.029595506806349627,
          0.09315224544455303, 0.020640678885583085, 0.047472627309343436,
          0.032958383881734096, 0.014663096865841139, 0.017152340172727078,
          0.029195401656427377, 0.010527045845271121, 0.03720496223361964,
          0.08565021870448736, 0.09947562485956811, 0.025808052654604105,
          0.08630566815905283]
    @test isapprox(w2.weights, wt)

    w3 = optimise!(portfolio; rm = SSD(), cluster = false, type = type,
                   hclust_alg = hclust_alg, hclust_opt = hclust_opt)
    wt = [0.07837975159477181, 0.07443043390995964, 0.06801950779661087,
          0.055548455549923444, 0.0644736826243098, 0.02739674778042435,
          0.029790140405552036, 0.08281583015461562, 0.021850035315318022,
          0.05161041880598439, 0.035414428250310895, 0.013356805631372573,
          0.02005310749097011, 0.03085778614009161, 0.012088382453633814,
          0.03792292849371053, 0.08401306865550462, 0.10203433260227453,
          0.027069661454433437, 0.08287449489022782]
    @test isapprox(w3.weights, wt)

    w4 = optimise!(portfolio; rm = FLPM(; target = rf), cluster = false, type = type,
                   hclust_alg = hclust_alg, hclust_opt = hclust_opt)
    wt = [0.08074577526070313, 0.0758057372636474, 0.0723712035349317, 0.05388227729039704,
          0.0719726576355769, 0.023357700765219442, 0.030259361503984363,
          0.09069744656455722, 0.020432554010122735, 0.045981867419973635,
          0.03127471464903142, 0.013225268293206591, 0.01541923284313144,
          0.026677624195275505, 0.00924248201367476, 0.03763591530781149,
          0.08961751171161211, 0.09815629750095696, 0.02604005979180501,
          0.08720431244438119]
    @test isapprox(w4.weights, wt)

    w5 = optimise!(portfolio; rm = SLPM(; target = rf), cluster = false, type = type,
                   hclust_alg = hclust_alg, hclust_opt = hclust_opt)
    wt = [0.07895203150464367, 0.07513349407323168, 0.06901069326760172,
          0.05594112932547382, 0.06763676613416904, 0.025518328924508955,
          0.030140432803640472, 0.08170116913851065, 0.021723048018546902,
          0.05096388176098718, 0.0345040083464426, 0.012730715665077913,
          0.018968393009821956, 0.029479897263328572, 0.01127249390583484,
          0.03813269633505064, 0.0862014575477909, 0.1015604380650207, 0.02714041935142286,
          0.08328850555889497]
    @test isapprox(w5.weights, wt)

    w6 = optimise!(portfolio; rm = WR(), cluster = false, type = type,
                   hclust_alg = hclust_alg, hclust_opt = hclust_opt)
    wt = [0.09799469653109426, 0.07931044806875052, 0.07697883394077526,
          0.05938016484003881, 0.062482774302560766, 0.024174945904638732,
          0.021508124985828447, 0.05117445754264079, 0.023413964946189022,
          0.07738784752841718, 0.02842727729869877, 0.006737256345835118,
          0.024299987114092203, 0.030479447210160812, 0.014669862619463056,
          0.033478912079376394, 0.10842038272038522, 0.0983414761742503,
          0.02496318167060388, 0.05637595817620051]
    @test isapprox(w6.weights, wt)

    w7 = optimise!(portfolio; rm = CVaR(), cluster = false, type = type,
                   hclust_alg = hclust_alg, hclust_opt = hclust_opt)
    wt = [0.07701588327689891, 0.07520448896018464, 0.06541770521947644,
          0.05757000169964883, 0.06337709857771338, 0.025926148616162506,
          0.030149122854382444, 0.08115573202397372, 0.021487151979485357,
          0.053917172483500996, 0.03629492645395641, 0.013834276128551553,
          0.02188007667915268, 0.030097549937467067, 0.012085783569165435,
          0.04006561840657164, 0.08240686258336018, 0.10281015669469054,
          0.02647712574837549, 0.08282711810728184]
    @test isapprox(w7.weights, wt)

    w8 = optimise!(portfolio; rm = EVaR(), cluster = false, type = type,
                   hclust_alg = hclust_alg, hclust_opt = hclust_opt)
    wt = [0.08699441730392812, 0.07712958717175648, 0.07366521524310121,
          0.06113285849402033, 0.06395471849180502, 0.025641689880494402,
          0.024561083982710123, 0.05538393000042802, 0.023397235003817983,
          0.06889218377773806, 0.032808220114320844, 0.0078354841769452,
          0.024451109097834427, 0.03131096206947751, 0.01392669730115599,
          0.03521555916294673, 0.09758384164962133, 0.10406746553380812,
          0.027502023704060772, 0.06454571784002953]
    @test isapprox(w8.weights, wt, rtol = 5.0e-7)

    w9 = optimise!(portfolio; rm = RVaR(), cluster = false, type = type,
                   hclust_alg = hclust_alg, hclust_opt = hclust_opt)
    wt = [0.09291265811652286, 0.07830058121386262, 0.07669383422811399,
          0.06048994792338911, 0.0632964281903878, 0.024835265241643308,
          0.022359240252220428, 0.05137247534829683, 0.023653937183085144,
          0.07457499324960506, 0.02985537439802565, 0.007019917248757973,
          0.024398011725401916, 0.03105422675169787, 0.01444843160813441,
          0.033975817433097084, 0.10534002288596125, 0.10072270393057964,
          0.026021876211112602, 0.058674256860104405]
    @test isapprox(w9.weights, wt, rtol = 5.0e-7)

    w10 = optimise!(portfolio; rm = MDD(), cluster = false, type = type,
                    hclust_alg = hclust_alg, hclust_opt = hclust_opt)
    wt = [0.11942828911415344, 0.06790696087036098, 0.09708343396881959, 0.0322181147701282,
          0.06358441456867103, 0.010918315222441926, 0.029080597027890735,
          0.05067903550079908, 0.019797098917560686, 0.032160771893579174,
          0.04334920387136981, 0.0072395254377313496, 0.004408762771082226,
          0.030601955352742596, 0.0068971479386845886, 0.02624877427854606,
          0.100996587649217, 0.10476039799498128, 0.03792180923631091, 0.11471880361492931]
    @test isapprox(w10.weights, wt)

    w11 = optimise!(portfolio; rm = ADD(), cluster = false, type = type,
                    hclust_alg = hclust_alg, hclust_opt = hclust_opt)
    wt = [0.10441410900507704, 0.06337257015393993, 0.1388818998669592,
          0.027515787911662204, 0.1188160882745767, 0.009101280957417223,
          0.02353511836320865, 0.03744216548824482, 0.015577834189637181,
          0.01767347121199722, 0.025141762273107857, 0.0033298619605358548,
          0.0018130367073894856, 0.014126651605875161, 0.002228743288689212,
          0.016302454122944926, 0.17177480035356854, 0.09442597723364114,
          0.03145585583242577, 0.08307053119910199]
    @test isapprox(w11.weights, wt)

    w12 = optimise!(portfolio; rm = CDaR(), cluster = false, type = type,
                    hclust_alg = hclust_alg, hclust_opt = hclust_opt)
    wt = [0.11749818113343935, 0.05870055826126679, 0.11385134924830828,
          0.029526234501200906, 0.08032687433988389, 0.008852903280265263,
          0.026595918536857853, 0.04345063385165379, 0.02084656630749761,
          0.029413320573451408, 0.04005512791398602, 0.005832294513804399,
          0.0033062821268677552, 0.02971553377155059, 0.005420678468821243,
          0.02415714416132541, 0.11618399112873753, 0.10379008396251159,
          0.040922826850160604, 0.10155349706840978]
    @test isapprox(w12.weights, wt)

    w13 = optimise!(portfolio; rm = UCI(), cluster = false, type = type,
                    hclust_alg = hclust_alg, hclust_opt = hclust_opt)
    wt = [0.10987855273964697, 0.060729473692400754, 0.13188113586803757,
          0.028009359764792043, 0.10116830124988405, 0.00894518902343889,
          0.02515249217180455, 0.040355519105512605, 0.018050336913773852,
          0.021728169704026638, 0.031807104420907735, 0.0040047144063036725,
          0.0023194434166183132, 0.020203823344076888, 0.0032143655180704735,
          0.019106596337899368, 0.15029999715584982, 0.09854472399816692,
          0.03553068862836474, 0.08907001254042422]
    @test isapprox(w13.weights, wt)

    w14 = optimise!(portfolio; rm = EDaR(), cluster = false, type = type,
                    hclust_alg = hclust_alg, hclust_opt = hclust_opt)
    wt = [0.1177308542531208, 0.06254095321442733, 0.10519293286493245,
          0.030663052419603157, 0.07121563960435068, 0.009654306131028573,
          0.02799873014876406, 0.046417936073046444, 0.020983248093459222,
          0.03096550906654087, 0.04162305259732029, 0.006349196745551905,
          0.0037357551287625196, 0.030460451517300507, 0.00591394170374592,
          0.02518714132643887, 0.11143867125608797, 0.10432624097459507,
          0.040889704525289054, 0.10671268235563422]
    @test isapprox(w14.weights, wt, rtol = 5.0e-8)

    w15 = optimise!(portfolio; rm = RDaR(), cluster = false, type = type,
                    hclust_alg = hclust_alg, hclust_opt = hclust_opt)
    wt = [0.1183325674687939, 0.0652819692615648, 0.10043845905125683, 0.03138018351723441,
          0.06655539507649458, 0.01031117836998939, 0.02870698570180175,
          0.048504100847745896, 0.020662238620582297, 0.03196981829418142,
          0.04288859208876169, 0.006797311243319078, 0.004081222092532856,
          0.030487185781576095, 0.00636796350460292, 0.026069755474491547,
          0.10619308223996969, 0.10450491799042878, 0.03982385359055927, 0.1106432197841129]
    @test isapprox(w15.weights, wt)

    w16 = optimise!(portfolio; rm = Kurt(), cluster = false, type = type,
                    hclust_alg = hclust_alg, hclust_opt = hclust_opt)
    wt = [0.053466892258046884, 0.10328667521887797, 0.054404919323835864,
          0.05164034525881938, 0.03949521281426285, 0.02166132115612342,
          0.005325106752082301, 0.06987092753743517, 0.019722628041309125,
          0.058905384535264414, 0.04756653865024671, 0.00337013314805977,
          0.005397558580038606, 0.03531833344651656, 0.005579968832386527,
          0.012662292885364586, 0.11359655047116007, 0.16209808354457939,
          0.02645811650511682, 0.11017301104047358]
    @test isapprox(w16.weights, wt)

    w17 = optimise!(portfolio; rm = SKurt(), cluster = false, type = type,
                    hclust_alg = hclust_alg, hclust_opt = hclust_opt)
    wt = [0.09195751022271098, 0.07943849543837476, 0.06646961117264624, 0.0491506750340217,
          0.05429977685689461, 0.026897931456943396, 0.010291696410768226,
          0.05363160327577018, 0.01926078898577857, 0.06072467774497688,
          0.047686505861964906, 0.0030930433701959084, 0.009416070036669964,
          0.037424766409940795, 0.006951516870884198, 0.021088101344911066,
          0.10896121609215866, 0.1536264710999023, 0.028737457378644478,
          0.07089208493584212]
    @test isapprox(w17.weights, wt)

    w18 = optimise!(portfolio; rm = Skew(), cluster = false, type = type,
                    hclust_alg = hclust_alg, hclust_opt = hclust_opt)
    wt = [0.05013622008238557, 0.05843253495835018, 0.0326340699568634, 0.03810785845352964,
          0.04151978371454936, 0.020776349162705425, 0.014419467673840766,
          0.06450029254562262, 0.02337036891691671, 0.0768704564650501, 0.03525278374131375,
          0.007885328106569226, 0.23227643708915915, 0.029736303978302396,
          0.019940256048870408, 0.05591386120058815, 0.04641584792557699,
          0.06294077176658942, 0.036661874867119304, 0.0522091333460975]
    @test isapprox(w18.weights, wt)

    w19 = optimise!(portfolio; rm = SSkew(), cluster = false, type = type,
                    hclust_alg = hclust_alg, hclust_opt = hclust_opt)
    wt = [0.06664919970362215, 0.065688537645969, 0.05827905623198537, 0.04923706156165528,
          0.055949261956495855, 0.030046317963353407, 0.027319842253653435,
          0.08126769722698507, 0.02377683386943261, 0.08094977263933717,
          0.04476608242725907, 0.012406273321936936, 0.028493003361406897,
          0.03436511159723589, 0.014308519726637069, 0.05423891622864567,
          0.07452547794485952, 0.09337434382384531, 0.028961215530580075,
          0.07539747498510434]
    @test isapprox(w19.weights, wt)

    hclust_alg = HAC()
    w20 = optimise!(portfolio; rm = DVar(), cluster = true, type = type,
                    hclust_alg = hclust_alg, hclust_opt = hclust_opt)
    wt = [0.11814156791628717, 0.1127352219498846, 0.0948040083033778, 0.05910201284574831,
          0.0762034280232549, 0.021900080015085997, 0.005240769029403859,
          0.02492257943111223, 0.013821146124791104, 0.015571892677105771,
          0.036219350442415844, 0.012370220354411125, 0.004197597088141013,
          0.02684034894442968, 0.008021889189561834, 0.01711283206712795,
          0.1478163099569289, 0.03165769900663641, 0.020856810764390717, 0.1524642358699047]
    @test isapprox(w20.weights, wt)

    w21 = optimise!(portfolio; rm = Variance(), cluster = false, type = type,
                    hclust_alg = hclust_alg, hclust_opt = hclust_opt)
    wt = [0.11540105921544155, 0.1145567201312018, 0.09419723372475931,
          0.060483660234210215, 0.07318364748176763, 0.02168713926476103,
          0.005201942574220557, 0.023749598552788367, 0.014061923482203057,
          0.015961970718579934, 0.03686671320264766, 0.012132318435028086,
          0.00441044709602619, 0.02704002898133752, 0.00862810493627861,
          0.016962353908264397, 0.14917275390007956, 0.03182856610967728,
          0.021005280556336624, 0.15346853749439043]
    @test isapprox(w21.weights, wt)

    w22 = optimise!(portfolio; rm = Equal(), cluster = false, type = type,
                    hclust_alg = hclust_alg, hclust_opt = hclust_opt)
    wt = [0.09923664122137407, 0.09923664122137407, 0.09923664122137407,
          0.09923664122137407, 0.09923664122137407, 0.023487962419260135,
          0.02348796241926013, 0.023487962419260135, 0.023487962419260135,
          0.023487962419260135, 0.023487962419260135, 0.02348796241926013,
          0.02348796241926013, 0.023487962419260135, 0.02348796241926013,
          0.02348796241926013, 0.09923664122137407, 0.023487962419260135,
          0.023487962419260135, 0.09923664122137407]
    @test isapprox(w22.weights, wt)

    w23 = optimise!(portfolio; rm = VaR(), cluster = false, type = type,
                    hclust_alg = hclust_alg, hclust_opt = hclust_opt)
    wt = [0.10414300257102177, 0.09871202573006761, 0.08959251861207183,
          0.07555514734087271, 0.09769669156091847, 0.022943300926461043,
          0.016563538745652133, 0.028652238501006684, 0.019401391461582398,
          0.017501979293453335, 0.030610780860929044, 0.02395924092174919,
          0.013446069288029138, 0.025629238086958404, 0.01730739640456328,
          0.028564197108681265, 0.11491557263229325, 0.027680710219061735,
          0.02230467164850204, 0.12482028808612478]
    @test isapprox(w23.weights, wt)

    w24 = optimise!(portfolio; rm = DaR(), cluster = false, type = type,
                    hclust_alg = hclust_alg, hclust_opt = hclust_opt)
    wt = [0.14733227960307277, 0.06614386458233755, 0.15133705565662284,
          0.035584333675786964, 0.11463609164605132, 0.005518410268992302,
          0.013171538035678308, 0.010620417669043803, 0.014023858241571917,
          0.017001289131758357, 0.026781482271749004, 0.007217550381915265,
          0.0035803904597587304, 0.020388510168558176, 0.0065705021611778035,
          0.029518131826492447, 0.16035186159841677, 0.026148373666210922,
          0.027631100950263884, 0.11644295800454063]
    @test isapprox(w24.weights, wt)

    w25 = optimise!(portfolio; rm = DaR_r(), cluster = false, type = type,
                    hclust_alg = hclust_alg, hclust_opt = hclust_opt)
    wt = [0.14774355938966005, 0.0684328493553202, 0.15485994881153173, 0.0411979500128519,
          0.11305584705890495, 0.007216566584313541, 0.011970976561070937,
          0.011775533919110818, 0.013661517786887336, 0.016951592576994145,
          0.02646193157189205, 0.008427695298573155, 0.006529147510212775,
          0.019339548023735872, 0.007726723460706738, 0.02381526338245954,
          0.15152867379279225, 0.025316906660827858, 0.025417970255231653,
          0.11856979798692265]
    @test isapprox(w25.weights, wt)

    w26 = optimise!(portfolio; rm = MDD_r(), cluster = false, type = type,
                    hclust_alg = hclust_alg, hclust_opt = hclust_opt)
    wt = [0.15079446061615842, 0.08603152475897777, 0.12374041606663375,
          0.050529282163624914, 0.0857760767414279, 0.009923344774006223,
          0.0134887577916876, 0.016005392064562644, 0.0148967949294955,
          0.020046408045779833, 0.029228399800324625, 0.0099793139298191,
          0.008133340149925354, 0.02141948291605225, 0.00938330394572054,
          0.02127772796435491, 0.12849798956868397, 0.030338927541390587,
          0.025120832103469992, 0.14538822412790406]
    @test isapprox(w26.weights, wt)

    w27 = optimise!(portfolio; rm = ADD_r(), cluster = false, type = type,
                    hclust_alg = hclust_alg, hclust_opt = hclust_opt)
    wt = [0.12371584689868499, 0.07328365011343618, 0.16977338386193674,
          0.03354181126220188, 0.14624734118080304, 0.009376911249269603,
          0.008150033847862812, 0.008167198562468324, 0.012903517155053858,
          0.011746529657682171, 0.021870250127331976, 0.005917245694323535,
          0.0032768170628257485, 0.011813734015517456, 0.003743956814916424,
          0.01775306443497017, 0.19872759251615896, 0.020009027345803686,
          0.025007088649056427, 0.09497499954969608]
    @test isapprox(w27.weights, wt)

    w28 = optimise!(portfolio; rm = CDaR_r(), cluster = false, type = type,
                    hclust_alg = hclust_alg, hclust_opt = hclust_opt)
    wt = [0.1482304036785224, 0.07559110451590667, 0.14480725317353932, 0.04480939008007434,
          0.10254791389098332, 0.00791620073213515, 0.011996465335688086,
          0.012913340180870978, 0.014391268171680365, 0.01805225079803855,
          0.0270384713414632, 0.008561039338370812, 0.006787624092846017,
          0.01981584825797337, 0.0079340412211216, 0.021763902803450537, 0.1447239549586514,
          0.026833292915610436, 0.025804676072481453, 0.12948155844059195]
    @test isapprox(w28.weights, wt)

    w29 = optimise!(portfolio; rm = UCI_r(), cluster = false, type = type,
                    hclust_alg = hclust_alg, hclust_opt = hclust_opt)
    wt = [0.13177358314927293, 0.07247391549233687, 0.16362956503582105,
          0.03659195084440874, 0.126920998889531, 0.00897565819404876, 0.010308036952094911,
          0.010070348969290802, 0.013930178878301605, 0.014394572348790095,
          0.025070306799710288, 0.006958517882519902, 0.004449124903473799,
          0.01545135954194139, 0.005243228396110122, 0.021086596858518908,
          0.17918434540380382, 0.02317354007700532, 0.02618369531594223,
          0.10413047606707744]
    @test isapprox(w29.weights, wt)

    w30 = optimise!(portfolio; rm = EDaR_r(), cluster = false, type = type,
                    hclust_alg = hclust_alg, hclust_opt = hclust_opt)
    wt = [0.1488274121958717, 0.07965043465691322, 0.1338236845244531, 0.04685245137285603,
          0.09455930834611771, 0.008714107998123267, 0.012663731601521146,
          0.014222803628373831, 0.015035762837750218, 0.01910358970690992,
          0.028184597388505987, 0.009107245853358044, 0.007272820589254927,
          0.020899558111554476, 0.008458685760279014, 0.021738533486077228,
          0.14067164133836854, 0.028485905052245995, 0.026525362442459087,
          0.13520236310900652]
    @test isapprox(w30.weights, wt, rtol = 1.0e-7)

    w31 = optimise!(portfolio; rm = RDaR_r(), cluster = false, type = type,
                    hclust_alg = hclust_alg, hclust_opt = hclust_opt)
    wt = [0.14968299216723, 0.08274800409125811, 0.12809346731596793, 0.0484163890815167,
          0.08949048752314671, 0.009338809076442186, 0.013100309780119947,
          0.015211346871070416, 0.015288347829970085, 0.01979313346063202,
          0.029044996716584604, 0.009477734654729942, 0.00761013719713636,
          0.021271110083134277, 0.008826450239538248, 0.021726662479268907,
          0.13497973924683532, 0.029701767255421313, 0.02630713511281022,
          0.13989097981718657]
    @test isapprox(w31.weights, wt)
end
