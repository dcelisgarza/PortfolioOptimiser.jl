using CSV, TimeSeries, DataFrames, StatsBase, Statistics, LinearAlgebra, Test, Clarabel,
      HiGHS, PortfolioOptimiser

path = joinpath(@__DIR__, "assets/stock_prices.csv")
prices = TimeArray(CSV.File(path); timestamp = :date)
rf = 1.0329^(1 / 252) - 1
l = 2.0

@testset "NCO with NOC" begin
    portfolio = Portfolio(; prices = prices,
                          solvers = PortOptSolver(; name = :Clarabel,
                                                  solver = Clarabel.Optimizer,
                                                  check_sol = (; allow_local = true,
                                                               allow_almost = true),
                                                  params = Dict("verbose" => false,
                                                                "max_step_fraction" => 0.75)))

    asset_statistics!(portfolio)
    clust_alg = DBHT()
    clust_opt = ClustOpt()
    cluster_assets!(portfolio; clust_alg = clust_alg, clust_opt = clust_opt)

    rm = SD()

    w1 = optimise!(portfolio,
                   NCO(; internal = NCOArgs(; type = NOC(; rm = rm, obj = MinRisk()))))
    r1 = calc_risk(portfolio, :NCO; rm = rm)
    wt = [0.03790461179159279, 0.043161358136428865, 0.037260347201576303,
          0.034058321573411314, 0.039786708000230105, 0.04830524198186707,
          0.022018351649423864, 0.06838696522512083, 0.02901431193405299,
          0.033984474731822575, 0.24449429847122825, 0.016129719077845203,
          0.001567497883473154, 0.0938319041806258, 0.008463292862160748,
          0.017921585913734644, 0.04771244028387784, 0.06793056625302624,
          0.05466122991582493, 0.05340677293267646]
    @test isapprox(w1.weights, wt, rtol = 0.0001)

    w2 = optimise!(portfolio,
                   NCO(; internal = NCOArgs(; type = NOC(; rm = rm, obj = MinRisk())),
                       external = NCOArgs(; type = NOC(; rm = rm, obj = MinRisk()))))
    r2 = calc_risk(portfolio, :NCO; rm = rm)
    wt = [0.03790461179159279, 0.043161358136428865, 0.037260347201576303,
          0.034058321573411314, 0.039786708000230105, 0.04830524198186707,
          0.022018351649423864, 0.06838696522512083, 0.02901431193405299,
          0.033984474731822575, 0.24449429847122825, 0.016129719077845203,
          0.001567497883473154, 0.0938319041806258, 0.008463292862160748,
          0.017921585913734644, 0.04771244028387784, 0.06793056625302624,
          0.05466122991582493, 0.05340677293267646]
    @test isapprox(w2.weights, wt, rtol = 0.0001)
    @test isapprox(w1.weights, w2.weights)
    @test isapprox(r1, r2)

    w3 = optimise!(portfolio,
                   NCO(; internal = NCOArgs(; type = NOC(; rm = rm, obj = MinRisk())),
                       external = NCOArgs(; type = Trad(; rm = rm, obj = MinRisk()))))
    r3 = calc_risk(portfolio, :NCO; rm = rm)
    wt = [0.03522433336548412, 0.0401157706666858, 0.03462972131292158,
          0.031651690697440636, 0.0369719851326808, 0.05546841978364625,
          0.020461077462647193, 0.0635737174567101, 0.03331656969190076,
          0.007610345048626629, 0.28075054462444293, 0.01852171205148576,
          0.00035102424943079825, 0.10774646605597438, 0.009718347579722019,
          0.004013305839129753, 0.044354167846025354, 0.0631237428140239,
          0.0627672338651757, 0.04962982445584546]
    @test isapprox(w3.weights, wt, rtol = 5.0e-5)

    w4 = optimise!(portfolio,
                   NCO(; internal = NCOArgs(; type = Trad(; rm = rm, obj = MinRisk())),
                       external = NCOArgs(; type = NOC(; rm = rm, obj = MinRisk()))))
    r4 = calc_risk(portfolio, :NCO; rm = rm)
    wt = [0.01461872888777548, 0.04572847655644966, 0.010546474390906085,
          0.019436203450122144, 0.006292956348233892, 0.04523997280439374,
          5.223619467757096e-8, 0.1370593489066906, 7.343036894502557e-8,
          0.07737762257975038, 0.1667709209757637, 0.0042547896186773725,
          0.0021994796705050255, 0.07955142439272779, 5.128161080628527e-9,
          0.02686619252126284, 0.03930893693930172, 0.19442814653414167,
          0.02864262263018601, 0.10167757199838735]
    @test isapprox(w4.weights, wt, rtol = 0.0001)

    @test r4 < r3 < r2 == r1
end

@testset "NOC" begin
    portfolio = Portfolio(; prices = prices,
                          solvers = PortOptSolver(; name = :Clarabel,
                                                  solver = Clarabel.Optimizer,
                                                  check_sol = (; allow_local = true,
                                                               allow_almost = true),
                                                  params = Dict("verbose" => false,
                                                                "max_step_fraction" => 0.75)))
    asset_statistics!(portfolio)
    rm = SD()

    obj = Sharpe(; rf = rf)

    kelly = NoKelly()
    w1 = optimise!(portfolio, Trad(; rm = rm, kelly = kelly, obj = MinRisk())).weights
    w2 = optimise!(portfolio, Trad(; rm = rm, kelly = kelly, obj = MaxRet())).weights
    w3 = optimise!(portfolio, Trad(; rm = rm, kelly = kelly, obj = obj)).weights

    w4 = optimise!(portfolio,
                   NOC(; w_min = w1, w_max = w2, w_opt = w3, rm = rm, kelly = kelly,
                       obj = obj))
    w5 = optimise!(portfolio, NOC(; rm = rm, kelly = kelly, obj = obj))
    @test isapprox(w5.weights, w4.weights)

    w6 = optimise!(portfolio,
                   NOC(; w_min = w1, w_max = w2, w_opt = w3, bins = 0.5, rm = rm,
                       kelly = kelly, obj = obj))
    w7 = optimise!(portfolio,
                   NOC(; w_min = w1, w_max = w2, w_opt = w3, bins = 1, rm = rm,
                       kelly = kelly, obj = obj))
    w8 = optimise!(portfolio,
                   NOC(; w_min = w1, w_max = w2, w_opt = w3, bins = 4, rm = rm,
                       kelly = kelly, obj = obj))
    w9 = optimise!(portfolio,
                   NOC(; w_min = w1, w_max = w2, w_opt = w3, bins = 32, rm = rm,
                       kelly = kelly, obj = obj))
    w10 = optimise!(portfolio,
                    NOC(; w_min = w1, w_max = w2, w_opt = w3, bins = 512, rm = rm,
                        kelly = kelly, obj = obj))
    w11 = optimise!(portfolio,
                    NOC(; w_min = w1, w_max = w2, w_opt = w3, bins = 16384, rm = rm,
                        kelly = kelly, obj = obj))
    w12 = optimise!(portfolio,
                    NOC(; w_min = w1, w_max = w2, w_opt = w3, bins = 1.048576e6, rm = rm,
                        kelly = kelly, obj = obj))

    @test rmsd(w3, w6.weights) >
          rmsd(w3, w7.weights) >
          rmsd(w3, w8.weights) >
          rmsd(w3, w4.weights) >
          rmsd(w3, w9.weights) >
          rmsd(w3, w10.weights) >
          rmsd(w3, w11.weights) >
          rmsd(w3, w12.weights)

    @test isapprox(w3, w6.weights, rtol = 1)
    @test isapprox(w3, w7.weights, rtol = 1)
    @test isapprox(w3, w8.weights, rtol = 1)
    @test isapprox(w3, w9.weights, rtol = 1)
    @test isapprox(w3, w10.weights, rtol = 0.1)
    @test isapprox(w3, w11.weights, rtol = 0.005)
    @test isapprox(w3, w12.weights, rtol = 5.0e-5)

    kelly = EKelly()
    w13 = optimise!(portfolio, Trad(; rm = rm, kelly = kelly, obj = MinRisk())).weights
    w14 = optimise!(portfolio, Trad(; rm = rm, kelly = kelly, obj = MaxRet())).weights
    w15 = optimise!(portfolio, Trad(; rm = rm, kelly = kelly, obj = obj)).weights

    w16 = optimise!(portfolio,
                    NOC(; w_min = w13, w_max = w14, w_opt = w15, rm = rm, kelly = kelly,
                        obj = obj))
    w17 = optimise!(portfolio, NOC(; rm = rm, kelly = kelly, obj = obj))
    @test isapprox(w17.weights, w16.weights)

    w18 = optimise!(portfolio,
                    NOC(; w_min = w13, w_max = w14, w_opt = w15, bins = 0.5, rm = rm,
                        kelly = kelly, obj = obj))
    w19 = optimise!(portfolio,
                    NOC(; w_min = w13, w_max = w14, w_opt = w15, bins = 1, rm = rm,
                        kelly = kelly, obj = obj))
    w20 = optimise!(portfolio,
                    NOC(; w_min = w13, w_max = w14, w_opt = w15, bins = 4, rm = rm,
                        kelly = kelly, obj = obj))
    w21 = optimise!(portfolio,
                    NOC(; w_min = w13, w_max = w14, w_opt = w15, bins = 32, rm = rm,
                        kelly = kelly, obj = obj))
    w22 = optimise!(portfolio,
                    NOC(; w_min = w13, w_max = w14, w_opt = w15, bins = 512, rm = rm,
                        kelly = kelly, obj = obj))
    w23 = optimise!(portfolio,
                    NOC(; w_min = w13, w_max = w14, w_opt = w15, bins = 16384, rm = rm,
                        kelly = kelly, obj = obj))
    w24 = optimise!(portfolio,
                    NOC(; w_min = w13, w_max = w14, w_opt = w15, bins = 1.048576e6, rm = rm,
                        kelly = kelly, obj = obj))

    @test rmsd(w15, w18.weights) >
          rmsd(w15, w19.weights) >
          rmsd(w15, w20.weights) >
          rmsd(w15, w16.weights) >
          rmsd(w15, w21.weights) >
          rmsd(w15, w22.weights) >
          rmsd(w15, w23.weights) >
          rmsd(w15, w24.weights)

    @test isapprox(w15, w18.weights, rtol = 1)
    @test isapprox(w15, w19.weights, rtol = 1)
    @test isapprox(w15, w20.weights, rtol = 0.5)
    @test isapprox(w15, w21.weights, rtol = 0.5)
    @test isapprox(w15, w22.weights, rtol = 0.05)
    @test isapprox(w15, w23.weights, rtol = 0.005)
    @test isapprox(w15, w24.weights, rtol = 0.0005)
end

@testset "NOC scale and vec" begin
    portfolio = Portfolio(; prices = prices,
                          solvers = PortOptSolver(; name = :Clarabel,
                                                  solver = Clarabel.Optimizer,
                                                  check_sol = (; allow_local = true,
                                                               allow_almost = true),
                                                  params = Dict("verbose" => false,
                                                                "max_step_fraction" => 0.75)))
    asset_statistics!(portfolio)

    kelly = NoKelly()
    obj = Sharpe(; rf = rf)

    rm = SD(;)
    w1 = optimise!(portfolio, NOC(; rm = rm, kelly = kelly, obj = obj))

    rm = SD(; settings = RMSettings(; scale = 5))
    w2 = optimise!(portfolio, NOC(; rm = rm, kelly = kelly, obj = obj))
    @test isapprox(w2.weights, w1.weights, rtol = 0.0001)

    rm = [SD(;)]
    w3 = optimise!(portfolio, NOC(; rm = rm, kelly = kelly, obj = obj))
    @test isapprox(w3.weights, w1.weights)
    @test isapprox(w3.weights, w2.weights, rtol = 0.0001)

    rm = [[SD(;)]]
    w4 = optimise!(portfolio, NOC(; rm = rm, kelly = kelly, obj = obj))
    @test isapprox(w4.weights, w1.weights, rtol = 5.0e-5)
    @test isapprox(w4.weights, w2.weights, rtol = 5.0e-5)
    @test isapprox(w4.weights, w3.weights, rtol = 5.0e-5)

    rm = [[SD(;), SD(;), SD(; settings = RMSettings(; scale = 0.75)),
           SD(; settings = RMSettings(; scale = 2.25))]]
    w5 = optimise!(portfolio, NOC(; rm = rm, kelly = kelly, obj = obj))
    @test isapprox(w5.weights, w1.weights, rtol = 1.0e-4)
    @test isapprox(w5.weights, w2.weights, rtol = 5.0e-5)
    @test isapprox(w5.weights, w3.weights, rtol = 1.0e-4)
    @test isapprox(w5.weights, w4.weights, rtol = 1.0e-4)
end

@testset "NOC vec convergence" begin
    portfolio = Portfolio(; prices = prices,
                          solvers = PortOptSolver(; name = :Clarabel,
                                                  solver = Clarabel.Optimizer,
                                                  check_sol = (; allow_local = true,
                                                               allow_almost = true),
                                                  params = Dict("verbose" => false,
                                                                "max_step_fraction" => 0.75)))
    asset_statistics!(portfolio)
    rm = [SD(;), CVaR()]

    obj = Sharpe(; rf = rf)

    kelly = NoKelly()
    w1 = optimise!(portfolio, Trad(; rm = rm, kelly = kelly, obj = MinRisk())).weights
    w2 = optimise!(portfolio, Trad(; rm = rm, kelly = kelly, obj = MaxRet())).weights
    w3 = optimise!(portfolio, Trad(; rm = rm, kelly = kelly, obj = obj)).weights

    w4 = optimise!(portfolio,
                   NOC(; w_min = w1, w_max = w2, w_opt = w3, rm = rm, kelly = kelly,
                       obj = obj))
    w5 = optimise!(portfolio, NOC(; rm = rm, kelly = kelly, obj = obj))
    @test isapprox(w5.weights, w4.weights)

    w6 = optimise!(portfolio,
                   NOC(; w_min = w1, w_max = w2, w_opt = w3, bins = 0.5, rm = rm,
                       kelly = kelly, obj = obj);)
    w7 = optimise!(portfolio,
                   NOC(; w_min = w1, w_max = w2, w_opt = w3, bins = 1, rm = rm,
                       kelly = kelly, obj = obj))
    w8 = optimise!(portfolio,
                   NOC(; w_min = w1, w_max = w2, w_opt = w3, bins = 4, rm = rm,
                       kelly = kelly, obj = obj))
    w9 = optimise!(portfolio,
                   NOC(; w_min = w1, w_max = w2, w_opt = w3, bins = 32, rm = rm,
                       kelly = kelly, obj = obj))
    w10 = optimise!(portfolio,
                    NOC(; w_min = w1, w_max = w2, w_opt = w3, bins = 512, rm = rm,
                        kelly = kelly, obj = obj);)
    w11 = optimise!(portfolio,
                    NOC(; w_min = w1, w_max = w2, w_opt = w3, bins = 16384, rm = rm,
                        kelly = kelly, obj = obj);)
    w12 = optimise!(portfolio,
                    NOC(; w_min = w1, w_max = w2, w_opt = w3, bins = 1.048576e6, rm = rm,
                        kelly = kelly, obj = obj);)

    @test rmsd(w3, w6.weights) >
          rmsd(w3, w7.weights) >
          rmsd(w3, w8.weights) >
          rmsd(w3, w4.weights) >
          rmsd(w3, w9.weights) >
          rmsd(w3, w10.weights) >
          rmsd(w3, w11.weights) >
          rmsd(w3, w12.weights)

    @test isapprox(w3, w6.weights, rtol = 1)
    @test isapprox(w3, w7.weights, rtol = 1)
    @test isapprox(w3, w8.weights, rtol = 1)
    @test isapprox(w3, w9.weights, rtol = 0.5)
    @test isapprox(w3, w10.weights, rtol = 0.05)
    @test isapprox(w3, w11.weights, rtol = 0.005)
    @test isapprox(w3, w12.weights, rtol = 0.001)

    rm = [SD(; settings = RMSettings(; scale = 7.3)),
          [CVaR(; alpha = 0.1, settings = RMSettings(; scale = 5)),
           CVaR(; settings = RMSettings(; scale = 1.6))]]
    w13 = optimise!(portfolio, Trad(; rm = rm, kelly = kelly, obj = MinRisk())).weights
    w14 = optimise!(portfolio, Trad(; rm = rm, kelly = kelly, obj = MaxRet())).weights
    w15 = optimise!(portfolio, Trad(; rm = rm, kelly = kelly, obj = obj)).weights

    w16 = optimise!(portfolio,
                    NOC(; w_min = w13, w_max = w14, w_opt = w15, rm = rm, kelly = kelly,
                        obj = obj))
    w17 = optimise!(portfolio, NOC(; rm = rm, kelly = kelly, obj = obj))
    @test isapprox(w17.weights, w16.weights)

    w18 = optimise!(portfolio,
                    NOC(; w_min = w13, w_max = w14, w_opt = w15, bins = 0.5, rm = rm,
                        kelly = kelly, obj = obj))
    w19 = optimise!(portfolio,
                    NOC(; w_min = w13, w_max = w14, w_opt = w15, bins = 1, rm = rm,
                        kelly = kelly, obj = obj))
    w20 = optimise!(portfolio,
                    NOC(; w_min = w13, w_max = w14, w_opt = w15, bins = 4, rm = rm,
                        kelly = kelly, obj = obj))
    w21 = optimise!(portfolio,
                    NOC(; w_min = w13, w_max = w14, w_opt = w15, bins = 32, rm = rm,
                        kelly = kelly, obj = obj))
    w22 = optimise!(portfolio,
                    NOC(; w_min = w13, w_max = w14, w_opt = w15, bins = 512, rm = rm,
                        kelly = kelly, obj = obj))
    w23 = optimise!(portfolio,
                    NOC(; w_min = w13, w_max = w14, w_opt = w15, bins = 16384, rm = rm,
                        kelly = kelly, obj = obj))
    w24 = optimise!(portfolio,
                    NOC(; w_min = w13, w_max = w14, w_opt = w15, bins = 1.048576e6, rm = rm,
                        kelly = kelly, obj = obj))

    @test rmsd(w15, w18.weights) >
          rmsd(w15, w19.weights) >
          rmsd(w15, w20.weights) >
          rmsd(w15, w16.weights) >
          rmsd(w15, w21.weights) >
          rmsd(w15, w22.weights) >
          rmsd(w15, w23.weights) >
          rmsd(w15, w24.weights)

    @test isapprox(w15, w18.weights, rtol = 1)
    @test isapprox(w15, w19.weights, rtol = 1)
    @test isapprox(w15, w20.weights, rtol = 1)
    @test isapprox(w15, w21.weights, rtol = 0.5)
    @test isapprox(w15, w22.weights, rtol = 0.05)
    @test isapprox(w15, w23.weights, rtol = 0.005)
    @test isapprox(w15, w24.weights, rtol = 1.0e-4)
end
