using CSV, TimeSeries, DataFrames, StatsBase, Statistics, LinearAlgebra, Test, Clarabel,
      HiGHS, PortfolioOptimiser

path = joinpath(@__DIR__, "assets/stock_prices.csv")
prices = TimeArray(CSV.File(path); timestamp = :date)
rf = 1.0329^(1 / 252) - 1
l = 2.0

@testset "NOC" begin
    portfolio = OmniPortfolio(; prices = prices,
                              solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                               :check_sol => (allow_local = true,
                                                                              allow_almost = true),
                                                               :params => Dict("verbose" => false,
                                                                               "max_step_fraction" => 0.75))))
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
    portfolio = OmniPortfolio(; prices = prices,
                              solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                               :check_sol => (allow_local = true,
                                                                              allow_almost = true),
                                                               :params => Dict("verbose" => false,
                                                                               "max_step_fraction" => 0.75))))
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
    portfolio = OmniPortfolio(; prices = prices,
                              solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                               :check_sol => (allow_local = true,
                                                                              allow_almost = true),
                                                               :params => Dict("verbose" => false,
                                                                               "max_step_fraction" => 0.75))))
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

#=
@testset "NCO with NOC" begin
    portfolio = HCPortfolio(; prices = prices,
                            solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                             :check_sol => (allow_local = true,
                                                                            allow_almost = true),
                                                             :params => Dict("verbose" => false,
                                                                             "max_step_fraction" => 0.75))))

    asset_statistics!(portfolio)
    clust_alg = DBHT()
    clust_opt = ClustOpt()
    cluster_assets!(portfolio; clust_alg = clust_alg, clust_opt = clust_opt)

    rm = SD(; formulation = SimpleSD())
    w1 = optimise!(portfolio; rm = rm, type = NCO(; opt_kwargs = (; obj = MinRisk())))
    r1 = calc_risk(portfolio; type = :NCO, rm = rm)
    wt = [0.013937212184142983, 0.04359664137131615, 0.010054803841639296,
          0.018530098862770877, 0.005999582355222919, 0.06378065191340355,
          4.980097342952298e-8, 0.13066972116367842, 1.0352430629888919e-7,
          7.530964844955846e-7, 0.23511857767960168, 0.005998528266296141,
          2.1406969513306348e-8, 0.11215395133737208, 7.229833188751728e-9,
          2.6148173677333317e-7, 0.03747637698611914, 0.18536401855581047,
          0.04038121666787219, 0.09693742227445049]
    @test isapprox(w1.weights, wt)

    w2 = optimise!(portfolio; rm = rm,
                   type = NCO(; opt_kwargs = (; type = NOC(), obj = MinRisk()),
                              opt_kwargs_o = (; type = NOC(), obj = MinRisk())))
    r2 = calc_risk(portfolio; type = :NCO, rm = rm)
    wt = [0.037906487242410136, 0.04316011941931974, 0.03726451142954405,
          0.03405602529282125, 0.03978456604176708, 0.048303222781681324,
          0.02201580957530569, 0.06839899510994614, 0.02901348249835985,
          0.03398372441620662, 0.24448459499777064, 0.016129492800245913,
          0.0015677990391979831, 0.0938277928127947, 0.008463074925084095,
          0.01792139606465465, 0.047723392186772226, 0.06792721680295467,
          0.054660405663482355, 0.05340789089968085]
    @test isapprox(w2.weights, wt, rtol = 1.0e-4)

    w3 = optimise!(portfolio; rm = rm,
                   type = NCO(; opt_kwargs = (; type = NOC(), obj = MinRisk()),
                              opt_kwargs_o = (; type = Trad(), obj = MinRisk())))
    r3 = calc_risk(portfolio; type = :NCO, rm = rm)
    wt = [0.035228326506436196, 0.04011078022707072, 0.03463170742642874,
          0.03164990654118643, 0.03697371569867293, 0.055468215884126544,
          0.020460353476242633, 0.06356648446573684, 0.03331715811268141,
          0.007610351200197044, 0.28074988613021806, 0.01852203925997262,
          0.00035109457555329305, 0.10774561132688958, 0.009718433676978827,
          0.0040133363953719125, 0.044351649658250544, 0.0631280381351374,
          0.06276838287497882, 0.04963452842786952]
    @test isapprox(w3.weights, wt, rtol = 5.0e-5)

    w4 = optimise!(portfolio; rm = rm,
                   type = NCO(; opt_kwargs = (; type = Trad(), obj = MinRisk()),
                              opt_kwargs_o = (; type = NOC(), obj = MinRisk())))
    r4 = calc_risk(portfolio; type = :NCO, rm = rm)
    wt = [0.01461876240896566, 0.045728581413237906, 0.010546498574292788,
          0.01943624801792809, 0.00629297077817429, 0.04524235707343006,
          5.223631445675487e-8, 0.1370596631879483, 7.343423892425297e-8,
          0.07736424339190633, 0.16677971025476615, 0.004255013857608455,
          0.0021990993635029516, 0.07955561696814764, 5.128431348599965e-9,
          0.026861547149326963, 0.03930902707589028, 0.194428592363798, 0.02864413217393731,
          0.10167780514815394]
    @test isapprox(w4.weights, wt, rtol = 5.0e-5)

    @test r1 < r4 < r3 < r2
end
=#
