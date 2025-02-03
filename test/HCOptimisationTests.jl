using CSV, TimeSeries, DataFrames, StatsBase, Statistics, LinearAlgebra, Test, Clarabel,
      PortfolioOptimiser

path = joinpath(@__DIR__, "assets/stock_prices.csv")
f_path = joinpath(@__DIR__, "assets/factor_prices.csv")
prices = TimeArray(CSV.File(path); timestamp = :date)
factors = TimeArray(CSV.File(f_path); timestamp = :date)

rf = 1.0329^(1 / 252) - 1
l = 2.0

@testset "Kurt, skew and cov via rm" begin
    portfolio = Portfolio(; prices = prices,
                          solvers = PortOptSolver(; name = :Clarabel,
                                                  check_sol = (; allow_local = true,
                                                               allow_almost = true),
                                                  solver = Clarabel.Optimizer,
                                                  params = ["verbose" => false,
                                                            "max_step_fraction" => 0.75]))

    asset_statistics!(portfolio; set_mu = false, set_cov = false, set_kurt = false,
                      set_skurt = false)
    cluster_assets!(portfolio)

    sigma = cov(PortCovCor(), portfolio.returns)
    kt = cokurt(KurtFull(), portfolio.returns, vec(mean(portfolio.returns; dims = 1)))
    skt = cokurt(KurtSemi(), portfolio.returns, vec(mean(portfolio.returns; dims = 1)))
    skew, V = coskew(SkewFull(), portfolio.returns, vec(mean(portfolio.returns; dims = 1)))
    sskew, SV = coskew(SkewSemi(), portfolio.returns,
                       vec(mean(portfolio.returns; dims = 1)))

    rm = Variance(; sigma = sigma)
    w1 = optimise!(portfolio, HRP(; rm = rm))
    @test isapprox(rm[2].sigma, sigma)
    rm = Kurt(; kt = kt)
    w2 = optimise!(portfolio, HRP(; rm = rm))
    @test isapprox(rm[2].kt, kt)
    rm = SKurt(; kt = skt)
    w3 = optimise!(portfolio, HRP(; rm = rm))
    @test isapprox(rm[2].kt, skt)

    rm = Variance(; sigma = sigma)
    w4 = optimise!(portfolio, HERC(; rm = rm))
    @test isapprox(rm[2].sigma, sigma)
    rm = Kurt(; kt = kt)
    w5 = optimise!(portfolio, HERC(; rm = rm))
    @test isapprox(rm[2].kt, kt)
    rm = SKurt(; kt = skt)
    w6 = optimise!(portfolio, HERC(; rm = rm))
    @test isapprox(rm[2].kt, skt)

    rm = Variance(; sigma = sigma)
    w7 = optimise!(portfolio,
                   NCO(; internal = NCOArgs(; type = Trad(; rm = rm, kelly = AKelly()))))
    @test isapprox(rm[2].sigma, sigma)
    rm = Kurt(; kt = kt)
    w8 = optimise!(portfolio,
                   NCO(; internal = NCOArgs(; type = Trad(; rm = rm, kelly = AKelly()))))
    @test isapprox(rm[2].kt, kt)
    rm = SKurt(; kt = skt)
    w9 = optimise!(portfolio,
                   NCO(; internal = NCOArgs(; type = Trad(; rm = rm, kelly = AKelly()))))
    @test isapprox(rm[2].kt, skt)

    rm = Skew(; V = 2 * V)
    w19 = optimise!(portfolio, HRP(; rm = rm))
    @test isapprox(rm[2].V, 2 * V)
    rm = SSkew(; V = 2 * SV)
    w20 = optimise!(portfolio, HRP(; rm = rm))
    @test isapprox(rm[2].V, 2 * SV)

    rm = Skew(; skew = 2 * skew)
    w19_1 = optimise!(portfolio, HRP(; rm = rm))
    @test isapprox(rm[2].skew, 2 * skew)
    @test isapprox(w19_1.weights, w19.weights, rtol = 1.0e-5)
    rm = SSkew(; skew = 2 * sskew)
    w20_1 = optimise!(portfolio, HRP(; rm = rm))
    @test isapprox(rm[2].skew, 2 * sskew)
    @test isapprox(w20_1.weights, w20.weights)

    rm = Skew(; V = 2 * V)
    w21 = optimise!(portfolio, HERC(; rm = rm))
    @test isapprox(rm[2].V, 2 * V)
    rm = SSkew(; V = 2 * SV)
    w22 = optimise!(portfolio, HERC(; rm = rm))
    @test isapprox(rm[2].V, 2 * SV)

    rm = Skew(; skew = 2 * skew)
    w21_1 = optimise!(portfolio, HERC(; rm = rm))
    @test isapprox(rm[2].skew, 2 * skew)
    @test isapprox(w21_1.weights, w21.weights)
    rm = SSkew(; skew = 2 * sskew)
    w22_1 = optimise!(portfolio, HERC(; rm = rm))
    @test isapprox(rm[2].skew, 2 * sskew)
    @test isapprox(w22_1.weights, w22.weights)

    rm = Skew(; V = V)
    w23 = optimise!(portfolio, NCO(; internal = NCOArgs(; type = Trad(; rm = rm))))
    @test isapprox(rm[2].V, V)
    rm = SSkew(; V = SV)
    w24 = optimise!(portfolio, NCO(; internal = NCOArgs(; type = Trad(; rm = rm))))
    @test isapprox(rm[2].V, SV)

    rm = Skew(; skew = 2 * skew)
    w23_1 = optimise!(portfolio, NCO(; internal = NCOArgs(; type = Trad(; rm = rm))))
    @test isapprox(rm[2].skew, 2 * skew)
    @test isapprox(w23_1.weights, w23.weights, rtol = 0.0001)
    rm = SSkew(; skew = 2 * sskew)
    w24_1 = optimise!(portfolio, NCO(; internal = NCOArgs(; type = Trad(; rm = rm))))
    @test isapprox(rm[2].skew, 2 * sskew)
    @test isapprox(w24_1.weights, w24.weights, rtol = 5.0e-5)

    asset_statistics!(portfolio; set_mu = false)

    w10 = optimise!(portfolio, HRP(; rm = Variance()))
    w11 = optimise!(portfolio, HRP(; rm = Kurt()))
    w12 = optimise!(portfolio, HRP(; rm = SKurt()))

    w13 = optimise!(portfolio, HERC(; rm = Variance()))
    w14 = optimise!(portfolio, HERC(; rm = Kurt()))
    w15 = optimise!(portfolio, HERC(; rm = SKurt()))

    w16 = optimise!(portfolio, NCO(; internal = NCOArgs(; type = Trad(; rm = Variance()))))
    w17 = optimise!(portfolio, NCO(; internal = NCOArgs(; type = Trad(; rm = Kurt()))))
    w18 = optimise!(portfolio, NCO(; internal = NCOArgs(; type = Trad(; rm = SKurt()))))

    w25 = optimise!(portfolio, HRP(; rm = Skew()))
    w26 = optimise!(portfolio, HRP(; rm = SSkew()))

    w27 = optimise!(portfolio, HERC(; rm = Skew()))
    w28 = optimise!(portfolio, HERC(; rm = SSkew()))

    w29 = optimise!(portfolio, NCO(; internal = NCOArgs(; type = Trad(; rm = Skew()))))
    w30 = optimise!(portfolio, NCO(; internal = NCOArgs(; type = Trad(; rm = SSkew()))))

    w1t = [0.03360421525593201, 0.053460257098811775, 0.027429766590708997,
           0.04363053745921338, 0.05279180956461212, 0.07434468966041922,
           0.012386194792256387, 0.06206960160806503, 0.025502890164538234,
           0.0542097204834031, 0.12168116639250848, 0.023275086688903004,
           0.009124639465879256, 0.08924750757276853, 0.017850423121104797,
           0.032541204698588386, 0.07175228284814082, 0.08318399209117079,
           0.03809545426566615, 0.07381856017730955]
    w2t = [0.02660166910791369, 0.06492641981073898, 0.027068370735117964,
           0.06101719342516449, 0.04666674917794706, 0.05829299778714021,
           0.006263434504353046, 0.04283800489412812, 0.028202565852669725,
           0.06457170610231176, 0.11833087994096461, 0.017593034185449084,
           0.01117615084978861, 0.08786112240590857, 0.02834636634602395,
           0.026942339849853326, 0.07415964743263134, 0.09938265800881307,
           0.03783404379529308, 0.07192464578778932]
    w3t = [0.03619626162744469, 0.05131541777566552, 0.02616372964484003,
           0.05351895109216385, 0.05912568036772633, 0.06166840901616903,
           0.014638304434756218, 0.03554570413131339, 0.02541229367305363,
           0.06409818387741549, 0.11154452907251113, 0.014164606720477545,
           0.015124625807739454, 0.08754107413382982, 0.026249557930258086,
           0.041079820488903906, 0.0829249585424891, 0.10181983671784, 0.03791561741640956,
           0.05395243752899324]
    w4t = [0.11540105921544158, 0.11455672013120183, 0.09419723372475934,
           0.06048366023421023, 0.07318364748176764, 0.021687139264761034,
           0.005201942574220558, 0.023749598552788374, 0.014061923482203061,
           0.015961970718579938, 0.036866713202647665, 0.01213231843502809,
           0.004410447096026191, 0.027040028981337526, 0.008628104936278611,
           0.0169623539082644, 0.14917275390007959, 0.031828566109677284,
           0.021005280556336627, 0.15346853749439046]
    w5t = [0.0779069998861783, 0.15049977013223184, 0.07927380598657693,
           0.07524552489013532, 0.05754876354838589, 0.017342386871012454,
           0.0035601126581858342, 0.013389722810757232, 0.015790239345986116,
           0.019222425762048703, 0.03808250242180403, 0.009413070969783905,
           0.006144844627076254, 0.02827640095700075, 0.015585331593998221,
           0.014415373404347924, 0.1655223647919237, 0.031063683899916917,
           0.02118277500263851, 0.16053390044001106]
    w6t = [0.12759061505036243, 0.1102205406291484, 0.09222627440800182, 0.0681963315725192,
           0.0753406862526989, 0.02151709753852694, 0.009454030160307273,
           0.014203775868994323, 0.015407737800930445, 0.02066343748542382,
           0.038146992810429145, 0.008582183611601027, 0.011113580975034482,
           0.029938077226805145, 0.019288185461586883, 0.02488982356691207,
           0.15118317736271694, 0.04068638321005349, 0.022988632951770532,
           0.09836243605617674]
    w7t = [0.022029677103937727, 0.03596539871198957, 0.004846414444560177,
           0.011259810102186373, 4.2160202875649446e-7, 0.053874859288704356,
           2.0846532743763247e-7, 0.14208891427636758, 3.6952885140262937e-6,
           0.02768874747299815, 0.2568475483615421, 5.645008467263602e-7,
           8.370868274128559e-8, 0.12222926890974696, 3.685260826109718e-7,
           8.804302258675684e-7, 0.054325688993184094, 0.19259307114726434,
           4.260112763443916e-6, 0.07624011855304699]
    w8t = [0.007321893263657814, 0.037651219157370845, 0.006916321147796381,
           0.03143428593615366, 7.986448718355135e-9, 0.05094433678849883,
           2.978278956345824e-9, 0.12379622169702287, 4.808228410842133e-8,
           0.048518419848816774, 0.32386577634426567, 9.393388818929017e-9,
           4.503077196079472e-9, 0.08107854772763415, 9.31098756001244e-9,
           1.1083355587896159e-8, 0.05245542545719536, 0.17232798490472695,
           7.848453282270715e-8, 0.063689395904507]
    w9t = [0.022318495992249606, 0.047226178420357236, 7.004353866419689e-9,
           0.016791162176251907, 4.763247606318575e-9, 0.024081210716552424,
           4.765774878415544e-9, 0.12898160524336555, 1.1643699768300647e-8,
           2.9335490684322297e-8, 0.3316526000943647, 7.028016818913455e-9,
           4.260115858157482e-9, 0.07247742611942459, 9.139181292286824e-9,
           1.3829158777444936e-8, 0.07805973127571302, 0.2163479664040439,
           2.264596968564213e-8, 0.062063509142667846]
    w10t = [0.03360421525593201, 0.053460257098811775, 0.027429766590708997,
            0.04363053745921338, 0.05279180956461212, 0.07434468966041922,
            0.012386194792256387, 0.06206960160806503, 0.025502890164538234,
            0.0542097204834031, 0.12168116639250848, 0.023275086688903004,
            0.009124639465879256, 0.08924750757276853, 0.017850423121104797,
            0.032541204698588386, 0.07175228284814082, 0.08318399209117079,
            0.03809545426566615, 0.07381856017730955]
    w11t = [0.02660166910791369, 0.06492641981073898, 0.027068370735117964,
            0.06101719342516449, 0.04666674917794706, 0.05829299778714021,
            0.006263434504353046, 0.04283800489412812, 0.028202565852669725,
            0.06457170610231176, 0.11833087994096461, 0.017593034185449084,
            0.01117615084978861, 0.08786112240590857, 0.02834636634602395,
            0.026942339849853326, 0.07415964743263134, 0.09938265800881307,
            0.03783404379529308, 0.07192464578778932]
    w12t = [0.03619626162744469, 0.05131541777566552, 0.02616372964484003,
            0.05351895109216385, 0.05912568036772633, 0.06166840901616903,
            0.014638304434756218, 0.03554570413131339, 0.02541229367305363,
            0.06409818387741549, 0.11154452907251113, 0.014164606720477545,
            0.015124625807739454, 0.08754107413382982, 0.026249557930258086,
            0.041079820488903906, 0.0829249585424891, 0.10181983671784, 0.03791561741640956,
            0.05395243752899324]
    w13t = [0.11540105921544158, 0.11455672013120183, 0.09419723372475934,
            0.06048366023421023, 0.07318364748176764, 0.021687139264761034,
            0.005201942574220558, 0.023749598552788374, 0.014061923482203061,
            0.015961970718579938, 0.036866713202647665, 0.01213231843502809,
            0.004410447096026191, 0.027040028981337526, 0.008628104936278611,
            0.0169623539082644, 0.14917275390007959, 0.031828566109677284,
            0.021005280556336627, 0.15346853749439046]
    w14t = [0.0779069998861783, 0.15049977013223184, 0.07927380598657693,
            0.07524552489013532, 0.05754876354838589, 0.017342386871012454,
            0.0035601126581858342, 0.013389722810757232, 0.015790239345986116,
            0.019222425762048703, 0.03808250242180403, 0.009413070969783905,
            0.006144844627076254, 0.02827640095700075, 0.015585331593998221,
            0.014415373404347924, 0.1655223647919237, 0.031063683899916917,
            0.02118277500263851, 0.16053390044001106]
    w15t = [0.12759061505036243, 0.1102205406291484, 0.09222627440800182,
            0.0681963315725192, 0.0753406862526989, 0.02151709753852694,
            0.009454030160307273, 0.014203775868994323, 0.015407737800930445,
            0.02066343748542382, 0.038146992810429145, 0.008582183611601027,
            0.011113580975034482, 0.029938077226805145, 0.019288185461586883,
            0.02488982356691207, 0.15118317736271694, 0.04068638321005349,
            0.022988632951770532, 0.09836243605617674]
    w16t = [0.022029677103937727, 0.03596539871198957, 0.004846414444560177,
            0.011259810102186373, 4.2160202875649446e-7, 0.053874859288704356,
            2.0846532743763247e-7, 0.14208891427636758, 3.6952885140262937e-6,
            0.02768874747299815, 0.2568475483615421, 5.645008467263602e-7,
            8.370868274128559e-8, 0.12222926890974696, 3.685260826109718e-7,
            8.804302258675684e-7, 0.054325688993184094, 0.19259307114726434,
            4.260112763443916e-6, 0.07624011855304699]
    w17t = [0.007321893263657814, 0.037651219157370845, 0.006916321147796381,
            0.03143428593615366, 7.986448718355135e-9, 0.05094433678849883,
            2.978278956345824e-9, 0.12379622169702287, 4.808228410842133e-8,
            0.048518419848816774, 0.32386577634426567, 9.393388818929017e-9,
            4.503077196079472e-9, 0.08107854772763415, 9.31098756001244e-9,
            1.1083355587896159e-8, 0.05245542545719536, 0.17232798490472695,
            7.848453282270715e-8, 0.063689395904507]
    w18t = [0.022318495992249606, 0.047226178420357236, 7.004353866419689e-9,
            0.016791162176251907, 4.763247606318575e-9, 0.024081210716552424,
            4.765774878415544e-9, 0.12898160524336555, 1.1643699768300647e-8,
            2.9335490684322297e-8, 0.3316526000943647, 7.028016818913455e-9,
            4.260115858157482e-9, 0.07247742611942459, 9.139181292286824e-9,
            1.3829158777444936e-8, 0.07805973127571302, 0.2163479664040439,
            2.264596968564213e-8, 0.062063509142667846]

    @test isapprox(w1.weights, w1t)
    @test isapprox(w2.weights, w2t)
    @test isapprox(w3.weights, w3t)
    @test isapprox(w4.weights, w4t)
    @test isapprox(w5.weights, w5t)
    @test isapprox(w6.weights, w6t)
    @test isapprox(w7.weights, w7t)
    @test isapprox(w8.weights, w8t)
    @test isapprox(w9.weights, w9t)
    @test isapprox(w10.weights, w10t)
    @test isapprox(w11.weights, w11t)
    @test isapprox(w12.weights, w12t)
    @test isapprox(w13.weights, w13t)
    @test isapprox(w14.weights, w14t)
    @test isapprox(w15.weights, w15t)
    @test isapprox(w16.weights, w16t)
    @test isapprox(w17.weights, w17t)
    @test isapprox(w18.weights, w18t)

    @test isapprox(w1.weights, w10.weights)
    @test isapprox(w2.weights, w11.weights)
    @test isapprox(w3.weights, w12.weights)
    @test isapprox(w4.weights, w13.weights)
    @test isapprox(w5.weights, w14.weights)
    @test isapprox(w6.weights, w15.weights)
    @test isapprox(w7.weights, w16.weights)
    @test isapprox(w8.weights, w17.weights)
    @test isapprox(w9.weights, w18.weights)

    @test isapprox(w19.weights, w25.weights)
    @test isapprox(w20.weights, w26.weights)
    @test isapprox(w21.weights, w27.weights)
    @test isapprox(w22.weights, w28.weights)
    @test isapprox(w23.weights, w29.weights)
    @test isapprox(w24.weights, w30.weights)
end

@testset "Kurt, skew and cov via rm vecs" begin
    portfolio = Portfolio(; prices = prices,
                          solvers = PortOptSolver(; name = :Clarabel,
                                                  check_sol = (; allow_local = true,
                                                               allow_almost = true),
                                                  solver = Clarabel.Optimizer,
                                                  params = ["verbose" => false,
                                                            "max_step_fraction" => 0.75]))

    asset_statistics!(portfolio; set_mu = false, set_cov = false, set_kurt = false,
                      set_skurt = false)
    cluster_assets!(portfolio)

    sigma = cov(PortCovCor(), portfolio.returns)
    kt = cokurt(KurtFull(), portfolio.returns, vec(mean(portfolio.returns; dims = 1)))
    skt = cokurt(KurtSemi(), portfolio.returns, vec(mean(portfolio.returns; dims = 1)))
    skew, V = coskew(SkewFull(), portfolio.returns, vec(mean(portfolio.returns; dims = 1)))
    sskew, SV = coskew(SkewSemi(), portfolio.returns,
                       vec(mean(portfolio.returns; dims = 1)))

    rm = [CVaR(), Variance(; sigma = sigma)]
    w1 = optimise!(portfolio, HRP(; rm = rm))
    @test isapprox(rm[2].sigma, sigma)
    rm = [CVaR(), Kurt(; kt = kt)]
    w2 = optimise!(portfolio, HRP(; rm = rm))
    @test isapprox(rm[2].kt, kt)
    rm = [CVaR(), SKurt(; kt = skt)]
    w3 = optimise!(portfolio, HRP(; rm = rm))
    @test isapprox(rm[2].kt, skt)

    type = HERC()
    rm = [CVaR(), Variance(; sigma = sigma)]
    w4 = optimise!(portfolio, HERC(; rm = rm))
    @test isapprox(rm[2].sigma, sigma)
    rm = [CVaR(), Kurt(; kt = kt)]
    w5 = optimise!(portfolio, HERC(; rm = rm))
    @test isapprox(rm[2].kt, kt)
    rm = [CVaR(), SKurt(; kt = skt)]
    w6 = optimise!(portfolio, HERC(; rm = rm))
    @test isapprox(rm[2].kt, skt)

    rm = [CVaR(), Variance(; sigma = sigma)]
    w7 = optimise!(portfolio,
                   NCO(; internal = NCOArgs(; type = Trad(; rm = rm, kelly = AKelly()))))
    @test isapprox(rm[2].sigma, sigma)
    rm = [CVaR(), Kurt(; kt = kt)]
    w8 = optimise!(portfolio,
                   NCO(; internal = NCOArgs(; type = Trad(; rm = rm, kelly = AKelly()))))
    @test isapprox(rm[2].kt, kt)
    rm = [CVaR(), SKurt(; kt = skt)]
    w9 = optimise!(portfolio,
                   NCO(; internal = NCOArgs(; type = Trad(; rm = rm, kelly = AKelly()))))
    @test isapprox(rm[2].kt, skt)

    rm = [CVaR(), Skew(; V = 2 * V)]
    w19 = optimise!(portfolio, HRP(; rm = rm))
    @test isapprox(rm[2].V, 2 * V)
    rm = [CVaR(), SSkew(; V = 2 * SV)]
    w20 = optimise!(portfolio, HRP(; rm = rm))
    @test isapprox(rm[2].V, 2 * SV)

    rm = [CVaR(), Skew(; skew = 2 * skew)]
    w19_1 = optimise!(portfolio, HRP(; rm = rm))
    @test isapprox(rm[2].skew, 2 * skew)
    @test isapprox(w19_1.weights, w19.weights, rtol = 1.0e-2)
    rm = SSkew(; skew = 2 * sskew)
    w20_1 = optimise!(portfolio, HRP(; rm = rm))
    @test isapprox(rm[2].skew, 2 * sskew)
    @test isapprox(w20_1.weights, w20.weights, rtol = 0.25)

    rm = [CVaR(), Skew(; V = 2 * V)]
    w21 = optimise!(portfolio, HERC(; rm = rm))
    @test isapprox(rm[2].V, 2 * V)
    rm = [CVaR(), SSkew(; V = 2 * SV)]
    w22 = optimise!(portfolio, HERC(; rm = rm))
    @test isapprox(rm[2].V, 2 * SV)

    rm = [CVaR(), Skew(; skew = 2 * skew)]
    w21_1 = optimise!(portfolio, HERC(; rm = rm))
    @test isapprox(rm[2].skew, 2 * skew)
    @test isapprox(w21_1.weights, w21.weights, rtol = 0.005)
    rm = [CVaR(), SSkew(; skew = 2 * sskew)]
    w22_1 = optimise!(portfolio, HERC(; rm = rm))
    @test isapprox(rm[2].skew, 2 * sskew)
    @test isapprox(w22_1.weights, w22.weights, rtol = 0.0001)

    rm = [CVaR(), Skew(; V = V)]
    w23 = optimise!(portfolio, NCO(; internal = NCOArgs(; type = Trad(; rm = rm))))
    @test isapprox(rm[2].V, V)
    rm = [CVaR(), SSkew(; V = SV)]
    w24 = optimise!(portfolio, NCO(; internal = NCOArgs(; type = Trad(; rm = rm))))
    @test isapprox(rm[2].V, SV)

    rm = [CVaR(), Skew(; skew = 2 * skew)]
    w23_1 = optimise!(portfolio, NCO(; internal = NCOArgs(; type = Trad(; rm = rm))))
    @test isapprox(rm[2].skew, 2 * skew)
    @test isapprox(w23_1.weights, w23.weights)
    rm = [CVaR(), SSkew(; skew = 2 * sskew)]
    w24_1 = optimise!(portfolio, NCO(; internal = NCOArgs(; type = Trad(; rm = rm))))
    @test isapprox(rm[2].skew, 2 * sskew)
    @test isapprox(w24_1.weights, w24.weights, rtol = 5.0e-7)

    asset_statistics!(portfolio; set_mu = false)

    w10 = optimise!(portfolio, HRP(; rm = [CVaR(), Variance()]))
    w11 = optimise!(portfolio, HRP(; rm = [CVaR(), Kurt()]))
    w12 = optimise!(portfolio, HRP(; rm = [CVaR(), SKurt()]))

    w13 = optimise!(portfolio, HERC(; rm = [CVaR(), Variance()]))
    w14 = optimise!(portfolio, HERC(; rm = [CVaR(), Kurt()]))
    w15 = optimise!(portfolio, HERC(; rm = [CVaR(), SKurt()]))

    w16 = optimise!(portfolio,
                    NCO(; internal = NCOArgs(; type = Trad(; rm = [CVaR(), Variance()]))))
    w17 = optimise!(portfolio,
                    NCO(; internal = NCOArgs(; type = Trad(; rm = [CVaR(), Kurt()]))))
    w18 = optimise!(portfolio,
                    NCO(; internal = NCOArgs(; type = Trad(; rm = [CVaR(), SKurt()]))))

    w25 = optimise!(portfolio, HRP(; rm = [CVaR(), Skew()]))
    w26 = optimise!(portfolio, HRP(; rm = [CVaR(), SSkew()]))

    w27 = optimise!(portfolio, HERC(; rm = [CVaR(), Skew()]))
    w28 = optimise!(portfolio, HERC(; rm = [CVaR(), SSkew()]))

    w29 = optimise!(portfolio,
                    NCO(; internal = NCOArgs(; type = Trad(; rm = [CVaR(), Skew()]))))
    w30 = optimise!(portfolio,
                    NCO(; internal = NCOArgs(; type = Trad(; rm = [CVaR(), SSkew()]))))

    w1t = [0.03191912278931461, 0.057356295271335865, 0.027105018721271124,
           0.05267914713061161, 0.058041971708577646, 0.0687270036161218,
           0.0324315957522821, 0.04638616161250091, 0.027892267829731383,
           0.060520016455435845, 0.094935286129952, 0.03975409164565936,
           0.01966229512217808, 0.0786785077523546, 0.02491129594684013,
           0.050640608771182725, 0.06740723996617975, 0.05878060126385216,
           0.03441194496315267, 0.06775952755146564]
    w2t = [0.031782319126716495, 0.05749933035537043, 0.02713019601034664,
           0.053074172578444045, 0.057835331095642206, 0.06853940685148799,
           0.030794537612400565, 0.04622772049193054, 0.027981501141492324,
           0.06077208014051075, 0.09505571712442011, 0.039260560602957906,
           0.01998139477043791, 0.07872181322024784, 0.025915681021648294,
           0.050450536771833004, 0.06746606098556457, 0.05921471984759456,
           0.034524458763489516, 0.0677724614874643]
    w3t = [0.03194670689341285, 0.05730918908520062, 0.02708991492837433,
           0.052716740096836495, 0.05803703667140571, 0.06864198052108295,
           0.032284520595141135, 0.04618471740412094, 0.027895012894372923,
           0.06063278738464075, 0.09495240159766592, 0.039078063497759025,
           0.019770500116274247, 0.0787016645721313, 0.02503386566347605,
           0.05109239747543731, 0.06754205892242685, 0.0590782045004118,
           0.03444690517818773, 0.06756533200164111]
    w4t = [0.10495180563866828, 0.10333053496331085, 0.0874136349322691,
           0.06677252048856239, 0.07649764524739523, 0.022809626039749935,
           0.0150199714841608, 0.024802991100789686, 0.01685750680153158,
           0.018825934014723093, 0.03533593632796397, 0.02602617794420506,
           0.013422963319329372, 0.027454570966629673, 0.020450844198465653,
           0.03488631519802314, 0.12393921027720352, 0.03233244324979774,
           0.022696496503164056, 0.12617287130405686]
    w5t = [0.08776281428633192, 0.11918075036227804, 0.080439216866369, 0.07325554831445297,
           0.06926892222733473, 0.020917620865680837, 0.012899671242049965,
           0.019676176097564595, 0.018108996057486716, 0.020954777951957108,
           0.03681125369462506, 0.022361260521871946, 0.015098566636273967,
           0.02872420686382621, 0.02772208238823209, 0.031165457346710855,
           0.13086879686988448, 0.03261616381919247, 0.023255464189665224,
           0.12891225339821194]
    w6t = [0.11344394048001268, 0.10393035686880636, 0.08866616255700727,
           0.07185246358073227, 0.07922665991633335, 0.022095168158564004,
           0.016075914858829293, 0.019396308942691246, 0.017113859278123512,
           0.020684961230844765, 0.034896407553425525, 0.01839251715421756,
           0.016639897723177047, 0.028100240568006595, 0.024902073498028143,
           0.03387109815098045, 0.1283694647434153, 0.035783410045234935,
           0.02306584028502785, 0.10349325440654183]
    w7t = [1.0867447852189117e-9, 0.03446388384236452, 5.1375611798750196e-11,
           0.006283064281311594, 1.923302868654452e-11, 1.7932673256642006e-10,
           0.0023177706992643927, 0.12008548912883997, 1.382579541830863e-10,
           2.1607486579317353e-10, 0.3281069515655058, 0.004174837684588885,
           0.0021095146997856585, 0.05651475179397661, 0.005139943943266227,
           0.012735737270880242, 0.05053064075548222, 0.319706746279398,
           1.6661472184346249e-10, 0.05783066619770816]
    w8t = [8.854143778398498e-11, 0.03508093869266307, 2.0010122433257614e-11,
           0.006341940283209653, 7.644244056003908e-12, 4.336854694431796e-11,
           0.0022778192727704524, 0.12006380069543163, 2.98086285748754e-11,
           5.548501753723077e-11, 0.328047695212308, 0.004270566975308162,
           0.0021776491237066238, 0.05650454265850678, 0.005127306238146291,
           0.012642505638374107, 0.05111467804930803, 0.31964900723382866,
           4.0863997750808095e-11, 0.056701549640716685]
    w9t = [3.924115602413179e-11, 0.03515332725626445, 6.5700755472056325e-12,
           0.006355026233155235, 2.5933267820754263e-12, 4.693194605571264e-11,
           0.002309499935670216, 0.12003533549184661, 3.198490331648681e-11,
           5.214367672231314e-11, 0.3279699204088096, 0.0041599397698217095,
           0.0021019874510341237, 0.056491146154459444, 0.005121602617565053,
           0.012690291188108618, 0.05122015085471356, 0.3195732238415225,
           4.274378944007153e-11, 0.05681854857482001]
    w10t = [0.03191912278931461, 0.057356295271335865, 0.027105018721271124,
            0.05267914713061161, 0.058041971708577646, 0.0687270036161218,
            0.0324315957522821, 0.04638616161250091, 0.027892267829731383,
            0.060520016455435845, 0.094935286129952, 0.03975409164565936,
            0.01966229512217808, 0.0786785077523546, 0.02491129594684013,
            0.050640608771182725, 0.06740723996617975, 0.05878060126385216,
            0.03441194496315267, 0.06775952755146564]
    w11t = [0.031782319126716495, 0.05749933035537043, 0.02713019601034664,
            0.053074172578444045, 0.057835331095642206, 0.06853940685148799,
            0.030794537612400565, 0.04622772049193054, 0.027981501141492324,
            0.06077208014051075, 0.09505571712442011, 0.039260560602957906,
            0.01998139477043791, 0.07872181322024784, 0.025915681021648294,
            0.050450536771833004, 0.06746606098556457, 0.05921471984759456,
            0.034524458763489516, 0.0677724614874643]
    w12t = [0.03194670689341285, 0.05730918908520062, 0.02708991492837433,
            0.052716740096836495, 0.05803703667140571, 0.06864198052108295,
            0.032284520595141135, 0.04618471740412094, 0.027895012894372923,
            0.06063278738464075, 0.09495240159766592, 0.039078063497759025,
            0.019770500116274247, 0.0787016645721313, 0.02503386566347605,
            0.05109239747543731, 0.06754205892242685, 0.0590782045004118,
            0.03444690517818773, 0.06756533200164111]
    w13t = [0.10495180563866828, 0.10333053496331085, 0.0874136349322691,
            0.06677252048856239, 0.07649764524739523, 0.022809626039749935,
            0.0150199714841608, 0.024802991100789686, 0.01685750680153158,
            0.018825934014723093, 0.03533593632796397, 0.02602617794420506,
            0.013422963319329372, 0.027454570966629673, 0.020450844198465653,
            0.03488631519802314, 0.12393921027720352, 0.03233244324979774,
            0.022696496503164056, 0.12617287130405686]
    w14t = [0.08776281428633192, 0.11918075036227804, 0.080439216866369,
            0.07325554831445297, 0.06926892222733473, 0.020917620865680837,
            0.012899671242049965, 0.019676176097564595, 0.018108996057486716,
            0.020954777951957108, 0.03681125369462506, 0.022361260521871946,
            0.015098566636273967, 0.02872420686382621, 0.02772208238823209,
            0.031165457346710855, 0.13086879686988448, 0.03261616381919247,
            0.023255464189665224, 0.12891225339821194]
    w15t = [0.11344394048001268, 0.10393035686880636, 0.08866616255700727,
            0.07185246358073227, 0.07922665991633335, 0.022095168158564004,
            0.016075914858829293, 0.019396308942691246, 0.017113859278123512,
            0.020684961230844765, 0.034896407553425525, 0.01839251715421756,
            0.016639897723177047, 0.028100240568006595, 0.024902073498028143,
            0.03387109815098045, 0.1283694647434153, 0.035783410045234935,
            0.02306584028502785, 0.10349325440654183]
    w16t = [1.0867447822227728e-9, 0.03446388384236512, 5.1375611662156214e-11,
            0.006283064281311699, 1.923302863145884e-11, 1.79326722610662e-10,
            0.002317770699264646, 0.12008548912883966, 1.3825794656517615e-10,
            2.1607485374730153e-10, 0.328106951565505, 0.004174837684588534,
            0.0021095146997854794, 0.05651475179397642, 0.005139943943266248,
            0.012735737270880525, 0.05053064075548302, 0.3197067462793973,
            1.6661471261214182e-10, 0.05783066619770856]
    w17t = [8.854143778398498e-11, 0.03508093869266307, 2.0010122433257614e-11,
            0.006341940283209653, 7.644244056003908e-12, 4.336854694431796e-11,
            0.0022778192727704524, 0.12006380069543163, 2.98086285748754e-11,
            5.548501753723077e-11, 0.328047695212308, 0.004270566975308162,
            0.0021776491237066238, 0.05650454265850678, 0.005127306238146291,
            0.012642505638374107, 0.05111467804930803, 0.31964900723382866,
            4.0863997750808095e-11, 0.056701549640716685]
    w18t = [3.924115602413179e-11, 0.03515332725626445, 6.5700755472056325e-12,
            0.006355026233155235, 2.5933267820754263e-12, 4.693194605571264e-11,
            0.002309499935670216, 0.12003533549184661, 3.198490331648681e-11,
            5.214367672231314e-11, 0.3279699204088096, 0.0041599397698217095,
            0.0021019874510341237, 0.056491146154459444, 0.005121602617565053,
            0.012690291188108618, 0.05122015085471356, 0.3195732238415225,
            4.274378944007153e-11, 0.05681854857482001]

    @test isapprox(w1.weights, w1t)
    @test isapprox(w2.weights, w2t)
    @test isapprox(w3.weights, w3t)
    @test isapprox(w4.weights, w4t)
    @test isapprox(w5.weights, w5t)
    @test isapprox(w6.weights, w6t)
    @test isapprox(w7.weights, w7t)
    @test isapprox(w8.weights, w8t)
    @test isapprox(w9.weights, w9t)
    @test isapprox(w10.weights, w10t)
    @test isapprox(w11.weights, w11t)
    @test isapprox(w12.weights, w12t)
    @test isapprox(w13.weights, w13t)
    @test isapprox(w14.weights, w14t)
    @test isapprox(w15.weights, w15t)
    @test isapprox(w16.weights, w16t)
    @test isapprox(w17.weights, w17t)
    @test isapprox(w18.weights, w18t)

    @test isapprox(w1.weights, w10.weights)
    @test isapprox(w2.weights, w11.weights)
    @test isapprox(w3.weights, w12.weights)
    @test isapprox(w4.weights, w13.weights)
    @test isapprox(w5.weights, w14.weights)
    @test isapprox(w6.weights, w15.weights)
    @test isapprox(w7.weights, w16.weights)
    @test isapprox(w8.weights, w17.weights)
    @test isapprox(w9.weights, w18.weights)

    @test isapprox(w19.weights, w25.weights)
    @test isapprox(w20.weights, w26.weights)
    @test isapprox(w21.weights, w27.weights)
    @test isapprox(w22.weights, w28.weights)
    @test isapprox(w23.weights, w29.weights)
    @test isapprox(w24.weights, w30.weights)
end

@testset "Tracking and Turnover rms" begin
    portfolio = Portfolio(; prices = prices,
                          solvers = PortOptSolver(; name = :Clarabel,
                                                  check_sol = (; allow_local = true,
                                                               allow_almost = true),
                                                  solver = Clarabel.Optimizer,
                                                  params = ["verbose" => false,
                                                            "max_step_fraction" => 0.75]))

    asset_statistics!(portfolio)
    cluster_assets!(portfolio)
    N = size(portfolio.returns, 2)
    w1 = optimise!(portfolio, Trad(; rm = CVaR(), obj = MinRisk()))
    w2 = optimise!(portfolio, HRP(; rm = CVaR()))
    w3 = optimise!(portfolio, HERC(; rm = CVaR()))
    benchmark = portfolio.returns * w1.weights
    n2 = norm(benchmark - portfolio.returns * w2.weights)
    n3 = norm(benchmark - portfolio.returns * w3.weights)
    n2_1 = norm(w1.weights - w2.weights, 1)
    n3_1 = norm(w1.weights - w3.weights, 1)

    rm = TrackingRM(; tr = TrackWeight(; w = w1.weights))
    w4 = optimise!(portfolio, HRP(; rm = rm))
    w5 = optimise!(portfolio, HERC(; rm = rm))
    n4 = norm(benchmark - portfolio.returns * w4.weights)
    n5 = norm(benchmark - portfolio.returns * w5.weights)

    @test n4 <= n2
    @test n5 <= n3

    rm = TrackingRM(; tr = TrackRet(; w = benchmark))
    w6 = optimise!(portfolio, HRP(; rm = rm))
    w7 = optimise!(portfolio, HERC(; rm = rm))
    n6 = norm(benchmark - portfolio.returns * w6.weights)
    n7 = norm(benchmark - portfolio.returns * w7.weights)

    @test n6 <= n2
    @test n7 <= n3

    rm = TurnoverRM(; tr = TR(; w = w1.weights))
    w8 = optimise!(portfolio, HRP(; rm = rm))
    w9 = optimise!(portfolio, HERC(; rm = rm))
    n8 = norm(w1.weights - w8.weights, 1)
    n9 = norm(w1.weights - w9.weights, 1)

    @test n8 >= n2_1
    @test n9 >= n3_1

    w10 = optimise!(portfolio, Trad(; rm = CVaR(), obj = Sharpe(; rf = rf)))
    w11 = optimise!(portfolio,
                    NCO(;
                        internal = NCOArgs(; type = Trad(; obj = MinRisk(), rm = CVaR()))))

    benchmark = portfolio.returns * w10.weights
    n11 = norm(benchmark - portfolio.returns * w11.weights)
    n11_1 = norm(w10.weights - w11.weights, 1)

    rm = TrackingRM(; tr = TrackWeight(; w = w10.weights))
    w12 = optimise!(portfolio, NCO(; internal = NCOArgs(; type = Trad(; rm = rm))))
    n12 = norm(benchmark - portfolio.returns * w12.weights)

    rm = TrackingRM(; tr = TrackRet(; w = benchmark))
    w13 = optimise!(portfolio, NCO(; internal = NCOArgs(; type = Trad(; rm = rm))))
    n13 = norm(benchmark - portfolio.returns * w13.weights)

    rm = TurnoverRM(; tr = TR(; w = w10.weights))
    w14 = optimise!(portfolio, NCO(; internal = NCOArgs(; type = Trad(; rm = rm))))
    n14 = norm(w10.weights - w14.weights, 1)

    @test n12 <= n11
    @test n13 <= n11
    @test n14 <= n11_1
end

@testset "Tracking and Turnover vector rms" begin
    portfolio = Portfolio(; prices = prices,
                          solvers = PortOptSolver(; name = :Clarabel,
                                                  check_sol = (; allow_local = true,
                                                               allow_almost = true),
                                                  solver = Clarabel.Optimizer,
                                                  params = ["verbose" => false,
                                                            "max_step_fraction" => 0.75]))
    asset_statistics!(portfolio)
    cluster_assets!(portfolio)
    N = size(portfolio.returns, 2)
    w1 = optimise!(portfolio, Trad(; rm = CVaR(), obj = MinRisk()))
    w2 = optimise!(portfolio, HRP(; rm = CVaR()))
    w3 = optimise!(portfolio, HERC(; rm = CVaR()))
    benchmark = portfolio.returns * w1.weights
    n2 = norm(benchmark - portfolio.returns * w2.weights)
    n3 = norm(benchmark - portfolio.returns * w3.weights)
    n2_1 = norm(w1.weights - w2.weights, 1)
    n3_1 = norm(w1.weights - w3.weights, 1)

    rm = TrackingRM(; tr = TrackWeight(; w = w1.weights))
    w4 = optimise!(portfolio, HRP(; rm = [rm]))
    w5 = optimise!(portfolio, HERC(; rm = [rm]))
    n4 = norm(benchmark - portfolio.returns * w4.weights)
    n5 = norm(benchmark - portfolio.returns * w5.weights)

    @test n4 <= n2
    @test n5 <= n3

    rm = TrackingRM(; tr = TrackRet(; w = benchmark))
    w6 = optimise!(portfolio, HRP(; rm = [rm]))
    w7 = optimise!(portfolio, HERC(; rm = [rm]))
    n6 = norm(benchmark - portfolio.returns * w6.weights)
    n7 = norm(benchmark - portfolio.returns * w7.weights)

    @test n6 <= n2
    @test n7 <= n3

    rm = TurnoverRM(; tr = TR(; w = w1.weights))
    w8 = optimise!(portfolio, HRP(; rm = [rm]))
    w9 = optimise!(portfolio, HERC(; rm = [rm]))
    n8 = norm(w1.weights - w8.weights, 1)
    n9 = norm(w1.weights - w9.weights, 1)

    @test n8 >= n2_1
    @test n9 >= n3_1

    w10 = optimise!(portfolio, Trad(; rm = CVaR(), obj = Sharpe(; rf = rf)))
    w11 = optimise!(portfolio,
                    NCO(;
                        internal = NCOArgs(; type = Trad(; obj = MinRisk(), rm = CVaR()))))

    benchmark = portfolio.returns * w10.weights
    n11 = norm(benchmark - portfolio.returns * w11.weights)
    n11_1 = norm(w10.weights - w11.weights, 1)

    rm = TrackingRM(; tr = TrackWeight(; w = w10.weights))
    w12 = optimise!(portfolio, NCO(; internal = NCOArgs(; type = Trad(; rm = [[rm]]))))
    n12 = norm(benchmark - portfolio.returns * w12.weights)

    rm = TrackingRM(; tr = TrackRet(; w = benchmark))
    w13 = optimise!(portfolio, NCO(; internal = NCOArgs(; type = Trad(; rm = [[rm]]))))
    n13 = norm(benchmark - portfolio.returns * w13.weights)

    rm = TurnoverRM(; tr = TR(; w = w10.weights))
    w14 = optimise!(portfolio, NCO(; internal = NCOArgs(; type = Trad(; rm = [[rm]]))))
    n14 = norm(w10.weights - w14.weights, 1)

    @test n12 <= n11
    @test n13 <= n11
    @test n14 <= n11_1
end

@testset "Weight bounds" begin
    portfolio = Portfolio(; prices = prices,
                          solvers = PortOptSolver(; name = :Clarabel,
                                                  check_sol = (; allow_local = true,
                                                               allow_almost = true),
                                                  solver = Clarabel.Optimizer,
                                                  params = ["verbose" => false,
                                                            "max_step_fraction" => 0.75]))

    asset_statistics!(portfolio)
    clust_alg = HAC()
    clust_opt = ClustOpt()

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
    cluster_assets!(portfolio; clust_alg = clust_alg, clust_opt = clust_opt)

    w1 = optimise!(portfolio, HRP(; rm = CDaR()))
    wt = [0.05927474378621105, 0.050096081527465364, 0.057435012961977804,
          0.033075532524845755, 0.06999999999999999, 0.039999999999999994,
          0.03361343145360156, 0.04999999999999999, 0.039999999999999994,
          0.054959447747864446, 0.039999999999999994, 0.019999999999999997,
          0.019999999999999997, 0.06999999999999999, 0.039999999999999994,
          0.06816457729305708, 0.06999999999999999, 0.06999999999999999,
          0.04338117270497694, 0.06999999999999999]
    @test isapprox(w1.weights, wt, rtol = 0.001)
    @test all(w1.weights .>= w_min .- sqrt(eps()) * N)
    @test all(w1.weights .<= w_max .+ sqrt(eps()) * N)

    w1_1 = optimise!(portfolio, HRP(; rm = CDaR(), finaliser = JWF(; version = ROJWF())))
    wt = [0.0699999955113408, 0.04727872130012514, 0.05697281708956848,
          0.031215392670285426, 0.07000000042435879, 0.03999999989232963,
          0.03172304062599841, 0.05000000010823203, 0.039999999870180866,
          0.05186857920340135, 0.04000000048200124, 0.01999999983736234,
          0.019999999905450372, 0.06999999975022839, 0.03999999988912165,
          0.06999999953348054, 0.07000000043404352, 0.0700000004734982, 0.04094145261427832,
          0.07000000038471464]
    @test isapprox(w1_1.weights, wt)
    @test all(w1_1.weights .>= w_min .- sqrt(eps()) * N)
    @test all(w1_1.weights .<= w_max .+ sqrt(eps()) * N)

    w1_2 = optimise!(portfolio, HRP(; rm = CDaR(), finaliser = JWF(; version = RSJWF())))
    wt = [0.05968793997944266, 0.04995503884836061, 0.057722733480619286,
          0.032382091987901866, 0.06999999499105145, 0.039999999886375995,
          0.032927994132740473, 0.050000003282147244, 0.039999999465947005,
          0.055089747927454406, 0.04000000020988046, 0.02000000150415533,
          0.01999999952239743, 0.0699999736687402, 0.03999999953466318, 0.06928599178691963,
          0.06999999522555336, 0.06999999628285072, 0.042948506934650264,
          0.06999999134814851]
    @test isapprox(w1_2.weights, wt)
    @test all(w1_2.weights .>= w_min .- sqrt(eps()) * N)
    @test all(w1_2.weights .<= w_max .+ sqrt(eps()) * N)

    w1_3 = optimise!(portfolio, HRP(; rm = CDaR(), finaliser = JWF(; version = AOJWF())))
    wt = [0.057364912846114065, 0.048752191394273905, 0.05564151063694583,
          0.03278388203530661, 0.0700000009978663, 0.04180146612390559, 0.03328620020944123,
          0.05173717215728229, 0.04182990540805701, 0.05336997481187418,
          0.04000000116060582, 0.021884926824836937, 0.021860455167492077,
          0.06912766742554988, 0.04188898670906466, 0.06557792195238114,
          0.07000000099529383, 0.07000000104674822, 0.043092821182330106,
          0.07000000091463027]
    @test isapprox(w1_3.weights, wt)
    @test all(w1_3.weights .>= w_min .- sqrt(eps()) * N)
    @test all(w1_3.weights .<= w_max .+ sqrt(eps()) * N)

    w1_4 = optimise!(portfolio, HRP(; rm = CDaR(), finaliser = JWF(; version = AOJWF())))
    wt = [0.057364912846114065, 0.048752191394273905, 0.05564151063694583,
          0.03278388203530661, 0.0700000009978663, 0.04180146612390559, 0.03328620020944123,
          0.05173717215728229, 0.04182990540805701, 0.05336997481187418,
          0.04000000116060582, 0.021884926824836937, 0.021860455167492077,
          0.06912766742554988, 0.04188898670906466, 0.06557792195238114,
          0.07000000099529383, 0.07000000104674822, 0.043092821182330106,
          0.07000000091463027]
    @test isapprox(w1_4.weights, wt)
    @test all(w1_4.weights .>= w_min .- sqrt(eps()) * N)
    @test all(w1_4.weights .<= w_max .+ sqrt(eps()) * N)

    w2 = optimise!(portfolio, HERC(; rm = CDaR()))
    wt = [0.07, 0.07, 0.07, 0.07, 0.07, 0.04, 0.02, 0.05, 0.04, 0.04, 0.04, 0.02, 0.02,
          0.04, 0.04, 0.059848065481384424, 0.07, 0.06015193451861548, 0.04, 0.07]
    @test isapprox(w2.weights, wt, rtol = 0.05)
    @test all(w2.weights .>= w_min .- sqrt(eps()) * N)
    @test all(w2.weights .<= w_max .+ sqrt(eps()) * N)

    w3 = optimise!(portfolio, NCO(; internal = NCOArgs(; type = Trad(; rm = CDaR()))))
    wt = [0.07, 0.027692307692307686, 0.07, 0.027692307692307686, 0.027692307692307686,
          0.05538461538461537, 0.027692307692307686, 0.06923076923076922,
          0.05538461538461537, 0.05538461538461537, 0.04, 0.027692307692307686,
          0.027692307692307686, 0.05538461538461537, 0.05538461538461537,
          0.027692307692307686, 0.07, 0.07, 0.07, 0.07]
    @test isapprox(w3.weights, wt)
    @test all(w3.weights .>= w_min .- sqrt(eps()) * N)
    @test all(w3.weights .<= w_max .+ sqrt(eps()) * N)

    portfolio.w_min = 0.03
    portfolio.w_max = 0.07
    w4 = optimise!(portfolio, HRP(; rm = CDaR()))
    wt = [0.05751902789250354, 0.04861223729749336, 0.0557337898326743, 0.03209583637708745,
          0.07, 0.03, 0.0326178027578793, 0.04184798638718673, 0.03, 0.053331550775954466,
          0.07, 0.03, 0.03, 0.07, 0.03, 0.0661455448334159, 0.07, 0.07, 0.04209622384580504,
          0.07]
    @test isapprox(w4.weights, wt, rtol = 0.0001)
    @test all(w4.weights .>= 0.03 .- sqrt(eps()) * N)
    @test all(w4.weights .<= 0.07 .+ sqrt(eps()) * N)

    w5 = optimise!(portfolio, HERC(; rm = CDaR()))
    wt = [0.07, 0.07, 0.07, 0.07, 0.07, 0.03923076923076922, 0.03923076923076922,
          0.03923076923076922, 0.03923076923076922, 0.03923076923076922,
          0.03923076923076922, 0.03923076923076922, 0.03923076923076922,
          0.03923076923076922, 0.03923076923076922, 0.03923076923076922, 0.07,
          0.03923076923076922, 0.03923076923076922, 0.07]
    @test isapprox(w5.weights, wt)
    @test all(w5.weights .>= 0.03 .- sqrt(eps()) * N)
    @test all(w5.weights .<= 0.07 .+ sqrt(eps()) * N)

    w6 = optimise!(portfolio, NCO(; internal = NCOArgs(; type = Trad(; rm = CDaR()))))
    wt = [0.07, 0.03923076923076922, 0.07, 0.03923076923076922, 0.03923076923076922,
          0.03923076923076922, 0.03923076923076922, 0.03923076923076922,
          0.03923076923076922, 0.03923076923076922, 0.07, 0.03923076923076922,
          0.03923076923076922, 0.03923076923076922, 0.03923076923076922,
          0.03923076923076922, 0.07, 0.07, 0.07, 0.07]
    @test isapprox(w6.weights, wt)
    @test all(w6.weights .>= 0.03 .- sqrt(eps()) * N)
    @test all(w6.weights .<= 0.07 .+ sqrt(eps()) * N)

    portfolio.w_min = 0
    portfolio.w_max = 1
    w7 = optimise!(portfolio, HRP(; rm = CDaR()))
    w8 = optimise!(portfolio, HERC(; rm = CDaR()))
    w9 = optimise!(portfolio, NCO(; internal = NCOArgs(; type = Trad(; rm = CDaR()))))

    portfolio.w_min = Float64[]
    portfolio.w_max = Float64[]
    w10 = optimise!(portfolio, HRP(; rm = CDaR()))
    w11 = optimise!(portfolio, HERC(; rm = CDaR()))
    w12 = optimise!(portfolio, NCO(; internal = NCOArgs(; type = Trad(; rm = CDaR()))))

    @test isapprox(w7.weights, w10.weights)
    @test isapprox(w8.weights, w11.weights)
    @test isapprox(w9.weights, w12.weights)

    portfolio.short = true
    portfolio.w_min = -0.03
    portfolio.w_max = 0.15
    w13 = optimise!(portfolio, NCO(; internal = NCOArgs(; type = Trad(; rm = CDaR()))))
    wt = [0.14999999999999997, -0.029999999999999995, 0.14999999999999997,
          -0.029999999999999995, -0.029999999999999995, 0.08531350787535882,
          -0.029999999999999995, 0.06666858517236564, -0.029999999999999995,
          0.02877758773517308, 0.14999999999999997, -0.029999999999999995,
          0.011366859578601332, -0.029999999999999995, -0.002126540361498923,
          -0.029999999999999995, 0.14999999999999997, 0.14999999999999997,
          0.14999999999999997, 0.14999999999999997]
    @test isapprox(w13.weights, wt)
    @test all(w13.weights .>= -0.03 .- sqrt(eps()) * N)
    @test all(w13.weights .<= 0.15 .+ sqrt(eps()) * N)

    w13_1 = optimise!(portfolio,
                      NCO(; internal = NCOArgs(; type = Trad(; rm = CDaR())),
                          finaliser = JWF(; version = ROJWF())))
    wt = [0.14999999964859664, 0.040370839007783065, 0.15000000073197584,
          -0.02178769216507173, -0.023260205089099935, 0.033031224830834156,
          -0.0164904095128501, 0.02581238398129523, -0.02790858030435506,
          0.011141951519275791, 0.15000000127972346, -0.015621800729896732,
          0.004400959505832409, -0.0300000006984973, -0.0008233424499980418,
          -0.028865334196442202, 0.15000000135442945, 0.15000000063290642,
          0.1500000012930362, 0.1500000013605241]
    @test isapprox(w13_1.weights, wt)
    @test all(w13_1.weights .>= -0.03 .- sqrt(eps()) * N)
    @test all(w13_1.weights .<= 0.15 .+ sqrt(eps()) * N)

    w13_2 = optimise!(portfolio,
                      NCO(; internal = NCOArgs(; type = Trad(; rm = CDaR())),
                          finaliser = JWF(; version = RSJWF())))
    wt = [0.13676723172008057, -0.01814258524880257, 0.15000000091672347,
          -0.015969227439151374, -0.01662868456316418, 0.04640430587383024,
          -0.013157308466385966, 0.0339789313130122, -0.018361692436895456,
          0.01266356855846837, 0.15000000199165217, -0.012630585063809275,
          0.004638358134100107, -0.020094631031805647, -0.0008150335191599346,
          -0.01865265779471798, 0.15000000215459158, 0.15000000079998788,
          0.15000000198042754, 0.15000000212101786]
    @test isapprox(w13_2.weights, wt)
    @test all(w13_2.weights .>= -0.03 .- sqrt(eps()) * N)
    @test all(w13_2.weights .<= 0.15 .+ sqrt(eps()) * N)

    w13_3 = optimise!(portfolio,
                      NCO(; internal = NCOArgs(; type = Trad(; rm = CDaR())),
                          finaliser = JWF(; version = AOJWF())))
    wt = [0.07838596818283826, -0.006187041798976532, 0.12177627795109004,
          -0.0032051818156493862, -0.003615734268447658, 0.03949767754142414,
          -0.0013308304367659017, 0.032465424447549365, -0.004639591503695699,
          0.01853613665854565, 0.15000000066240204, -0.0009597608731757931,
          0.01252119349577821, -0.005646339312839102, 0.008216174275317654,
          -0.004804737712312188, 0.15000000074763667, 0.11899036229938285,
          0.15000000066671793, 0.15000000079317793]
    @test isapprox(w13_3.weights, wt)
    @test all(w13_3.weights .>= -0.03 .- sqrt(eps()) * N)
    @test all(w13_3.weights .<= 0.15 .+ sqrt(eps()) * N)

    w13_4 = optimise!(portfolio,
                      NCO(; internal = NCOArgs(; type = Trad(; rm = CDaR())),
                          finaliser = JWF(; version = ASJWF())))
    wt = [0.08733749701146554, -0.029999998790760967, 0.13151207133940515,
          -0.006893196150548155, -0.008365701783481633, 0.04792568071491608,
          -0.0015959420419500764, 0.040706832292976836, -0.01301405232550986,
          0.026036380233699363, 0.15000000170124103, -0.0007273381878140592,
          0.01929537991901988, -0.020936402501242252, 0.014071075235170587,
          -0.01397079992208166, 0.1500000018747393, 0.12861850782574158,
          0.15000000167230898, 0.1500000018827043]
    @test isapprox(w13_4.weights, wt)
    @test all(w13_4.weights .>= -0.03 .- sqrt(eps()) * N)
    @test all(w13_4.weights .<= 0.15 .+ sqrt(eps()) * N)
end

@testset "Schur HRP" begin
    portfolio = Portfolio(; prices = prices,
                          solvers = PortOptSolver(; name = :Clarabel,
                                                  solver = Clarabel.Optimizer,
                                                  check_sol = (; allow_local = true,
                                                               allow_almost = true),
                                                  params = "verbose" => false))

    asset_statistics!(portfolio)
    cluster_assets!(portfolio)

    w1 = optimise!(portfolio, SchurHRP(; params = SchurParams(; gamma = 0, prop_coef = 0)))
    w2 = optimise!(portfolio,
                   SchurHRP(; params = [SchurParams(; gamma = 0, prop_coef = 0)]))
    w3 = optimise!(portfolio,
                   SchurHRP(; params = SchurParams(; gamma = 0, prop_coef = 0.5)))
    w4 = optimise!(portfolio,
                   SchurHRP(; params = [SchurParams(; gamma = 0, prop_coef = 0.5)]))
    w5 = optimise!(portfolio, SchurHRP(; params = SchurParams(; gamma = 0, prop_coef = 1)))
    w6 = optimise!(portfolio,
                   SchurHRP(; params = [SchurParams(; gamma = 0, prop_coef = 1)]))
    wt = [0.03360421525593201, 0.053460257098811775, 0.027429766590708997,
          0.04363053745921338, 0.05279180956461212, 0.07434468966041922,
          0.012386194792256387, 0.06206960160806503, 0.025502890164538234,
          0.0542097204834031, 0.12168116639250848, 0.023275086688903004,
          0.009124639465879256, 0.08924750757276853, 0.017850423121104797,
          0.032541204698588386, 0.07175228284814082, 0.08318399209117079,
          0.03809545426566615, 0.07381856017730955]
    @test isapprox(wt, w1.weights)
    @test isapprox(w1.weights, w2.weights)
    @test isapprox(w1.weights, w3.weights)
    @test isapprox(w1.weights, w4.weights)
    @test isapprox(w1.weights, w5.weights)
    @test isapprox(w1.weights, w6.weights)

    w7 = optimise!(portfolio,
                   SchurHRP(; params = SchurParams(; gamma = 0.5, prop_coef = 0)))
    w8 = optimise!(portfolio,
                   SchurHRP(; params = [SchurParams(; gamma = 0.5, prop_coef = 0)]))
    wt = [0.04817241487998945, 0.0766365072042637, 0.039321200813807244,
          0.059715527896678174, 0.07225422743688918, 0.07869738002561813,
          0.0021336961067667005, 0.06570361713930474, 0.022599942168831988,
          0.048039125762185546, 0.11832529232461884, 0.02201138591116128,
          0.0015718473696351143, 0.08678612915927499, 0.0030749862210669495,
          0.030774408027284685, 0.05045796814141483, 0.08805420087901901,
          0.0337591174076653, 0.05191102512452411]
    @test isapprox(w7.weights, wt)

    w9 = optimise!(portfolio,
                   SchurHRP(; params = SchurParams(; gamma = 0.5, prop_coef = 0.5)))
    w10 = optimise!(portfolio,
                    SchurHRP(; params = [SchurParams(; gamma = 0.5, prop_coef = 0.5)]))
    wt = [0.03112165095243884, 0.1128077064846914, 0.02644397619176801, 0.03788942838988588,
          0.09057698759305942, 0.09706490207066822, 0.004698829761509681,
          0.04805914732787866, 0.018715161800093114, 0.048525116568153, 0.0797586054385343,
          0.032309215223349605, 0.0037641760733464298, 0.15393849110819147,
          0.0038323324937069407, 0.04620574903573764, 0.05497269088562942,
          0.043731272288269195, 0.02644293794408931, 0.039141622368999415]
    @test isapprox(w9.weights, wt)

    w11 = optimise!(portfolio,
                    SchurHRP(; params = SchurParams(; gamma = 0.5, prop_coef = 1)))
    w12 = optimise!(portfolio,
                    SchurHRP(; params = [SchurParams(; gamma = 0.5, prop_coef = 1)]))
    wt = [0.026934099434388202, 0.09457804872676835, 0.02706522286326548,
          0.02986436181624784, 0.03085880861935407, 0.15782233995032932,
          0.034380987584464756, 0.060060821427826136, 0.022339882221851893,
          0.05463528645648568, 0.10102027723890357, 0.026581383263657554,
          0.026618650389921793, 0.06847489832446459, 0.014426158044887356,
          0.044911132499694084, 0.053249856068170794, 0.03228032642407734,
          0.026624250997674282, 0.06727320764756689]
    @test isapprox(w11.weights, wt)

    @test isapprox(w7.weights, w8.weights)
    @test isapprox(w9.weights, w10.weights)
    @test isapprox(w11.weights, w12.weights)
    @test !isapprox(w1.weights, w7.weights)
    @test !isapprox(w1.weights, w9.weights)
    @test !isapprox(w1.weights, w11.weights)
    @test !isapprox(w7.weights, w9.weights)
    @test !isapprox(w7.weights, w11.weights)
    @test !isapprox(w9.weights, w11.weights)

    w13 = optimise!(portfolio, SchurHRP(; params = SchurParams(; gamma = 1, prop_coef = 0)))
    w14 = optimise!(portfolio,
                    SchurHRP(; params = [SchurParams(; gamma = 1, prop_coef = 0)]))
    wt = [0.04817241487998945, 0.0766365072042637, 0.039321200813807244,
          0.059715527896678174, 0.07225422743688918, 0.07869738002561813,
          0.0021336961067667005, 0.06570361713930474, 0.022599942168831988,
          0.048039125762185546, 0.11832529232461884, 0.02201138591116128,
          0.0015718473696351143, 0.08678612915927499, 0.0030749862210669495,
          0.030774408027284685, 0.05045796814141483, 0.08805420087901901,
          0.0337591174076653, 0.05191102512452411]
    @test isapprox(w13.weights, wt)

    w15 = optimise!(portfolio,
                    SchurHRP(; params = SchurParams(; gamma = 1, prop_coef = 0.5)))
    w16 = optimise!(portfolio,
                    SchurHRP(; params = [SchurParams(; gamma = 1, prop_coef = 0.5)]))
    wt = [0.03112165095243884, 0.1128077064846914, 0.02644397619176801, 0.03788942838988588,
          0.09057698759305942, 0.09706490207066822, 0.004698829761509681,
          0.04805914732787866, 0.018715161800093114, 0.048525116568153, 0.0797586054385343,
          0.032309215223349605, 0.0037641760733464298, 0.15393849110819147,
          0.0038323324937069407, 0.04620574903573764, 0.05497269088562942,
          0.043731272288269195, 0.02644293794408931, 0.039141622368999415]
    @test isapprox(w15.weights, wt)

    w17 = optimise!(portfolio, SchurHRP(; params = SchurParams(; gamma = 1, prop_coef = 1)))
    w18 = optimise!(portfolio,
                    SchurHRP(; params = [SchurParams(; gamma = 1, prop_coef = 1)]))
    wt = [0.026934099434388202, 0.09457804872676835, 0.02706522286326548,
          0.02986436181624784, 0.03085880861935407, 0.15782233995032932,
          0.034380987584464756, 0.060060821427826136, 0.022339882221851893,
          0.05463528645648568, 0.10102027723890357, 0.026581383263657554,
          0.026618650389921793, 0.06847489832446459, 0.014426158044887356,
          0.044911132499694084, 0.053249856068170794, 0.03228032642407734,
          0.026624250997674282, 0.06727320764756689]
    @test isapprox(w17.weights, wt)

    @test isapprox(w13.weights, w14.weights)
    @test isapprox(w15.weights, w16.weights)
    @test isapprox(w17.weights, w18.weights)

    @test !isapprox(w1.weights, w13.weights)
    @test !isapprox(w1.weights, w15.weights)
    @test !isapprox(w1.weights, w17.weights)
    @test isapprox(w7.weights, w13.weights)
    @test !isapprox(w7.weights, w15.weights)
    @test !isapprox(w7.weights, w17.weights)
    @test !isapprox(w9.weights, w13.weights)
    @test isapprox(w9.weights, w15.weights)
    @test !isapprox(w9.weights, w17.weights)
    @test !isapprox(w11.weights, w13.weights)
    @test !isapprox(w11.weights, w15.weights)
    @test isapprox(w11.weights, w17.weights)

    wh = optimise!(portfolio, HRP())
    wt = optimise!(portfolio, Trad())

    @test isapprox(wh.weights, w1.weights)
    @test StatsBase.rmsd(wh.weights, w1.weights) <=
          StatsBase.rmsd(wh.weights, w7.weights) <=
          StatsBase.rmsd(wh.weights, w13.weights) <=
          StatsBase.rmsd(wh.weights, w17.weights) <=
          StatsBase.rmsd(wh.weights, w11.weights) <=
          StatsBase.rmsd(wh.weights, w9.weights) <=
          StatsBase.rmsd(wh.weights, w15.weights)
end

@testset "Test failures" begin
    portfolio = Portfolio(; prices = prices,
                          solvers = PortOptSolver(; name = :Clarabel,
                                                  solver = Clarabel.Optimizer,
                                                  check_sol = (; allow_local = true,
                                                               allow_almost = true),
                                                  params = ["verbose" => false,
                                                            "max_step_fraction" => 0.75,
                                                            "max_iter" => 1]))

    asset_statistics!(portfolio)
    cluster_assets!(portfolio)

    @test_throws ErrorException optimise!(portfolio, HRP(; rm = RLVaR()))

    w1 = optimise!(portfolio, NCO(; internal = NCOArgs(; type = Trad(; rm = RLVaR()))))
    @test isempty(w1)
    @test haskey(portfolio.fail, :inter)
    @test haskey(portfolio.fail, :intra)
    @test portfolio.k == size(Matrix(portfolio.fail[:inter][:Clarabel_Trad][:port]), 1)
    @test length(keys(portfolio.fail[:intra])) == 3
    num_assets = 0
    for val ∈ values(portfolio.fail[:intra])
        num_assets += size(Matrix(val[:Clarabel_Trad][:port]), 1)
    end
    @test num_assets == 20

    asset_statistics!(portfolio; set_kurt = false, set_skurt = false, set_skew = false,
                      set_sskew = false)
    cluster_assets!(portfolio)
    portfolio.cov[3, :] .= portfolio.cov[:, 3] .= NaN
    @test isempty(optimise!(portfolio, HRP()))
    @test !isempty(portfolio.fail)
end

@testset "NCO HCOptimType and OptimType" begin
    portfolio = Portfolio(; prices = prices,
                          solvers = PortOptSolver(; name = :Clarabel,
                                                  check_sol = (; allow_local = true,
                                                               allow_almost = true),
                                                  solver = Clarabel.Optimizer,
                                                  params = ["verbose" => false,
                                                            "max_step_fraction" => 0.75]))

    asset_statistics!(portfolio; cor_type = PortCovCor(; ce = CovLTD()))
    cluster_assets!(portfolio)
    sd = SD()

    w1 = optimise!(portfolio, NCO(; internal = NCOArgs(; type = HRP(; rm = sd))))
    wt = [0.023212410493909395, 0.04180507918898045, 0.02097173779429894,
          0.01915222066801841, 0.019617512785329, 0.027315103218872105,
          0.010288389735636377, 0.28698504210338543, 0.018692638550135976,
          0.024753790610823476, 0.04069430485974347, 0.08801583260757373,
          0.05317874944885358, 0.032218256478400265, 0.07437971688459784,
          0.10407153529171204, 0.028007972576333588, 0.03470276984884458,
          0.022846121725202524, 0.029090815129348833]
    @test isapprox(w1.weights, wt)

    w2 = optimise!(portfolio, NCO(; internal = NCOArgs(; type = HERC(; rm = sd))))
    wt = [0.03502677104953508, 0.034898398023022054, 0.03164566895904809,
          0.02535795569178541, 0.02789344989235047, 0.028583381104026976,
          0.009422007575348099, 0.1299161111096586, 0.023016264028061697,
          0.024521990540045523, 0.03726744986476854, 0.0968547793732363,
          0.10704219566485063, 0.03191655746795579, 0.14971710111225556,
          0.07389688676967159, 0.03982354889701095, 0.034627489210387746,
          0.02813045190138087, 0.03044154176560006]
    @test isapprox(w2.weights, wt)

    w3 = optimise!(portfolio,
                   NCO(;
                       internal = NCOArgs(;
                                          type = NCO(;
                                                     internal = NCOArgs(;
                                                                        type = Trad(;
                                                                                    rm = sd))))))
    wt = [0.014969752163027872, 0.019968828399564154, 0.006074862573602332,
          0.005398360397220456, 0.00133313673639315, 0.029624908720316784,
          1.0560450839914053e-7, 0.10411919273781294, 1.0533484919594905e-7,
          0.01627282847655014, 0.2707826283771348, 0.013353759302369013,
          6.795169084887368e-8, 0.11853226528417016, 1.5158507899309045e-7,
          0.024600407558151988, 0.0341094108244004, 0.19553510703461152,
          1.502558631131416e-7, 0.14532397068268374]
    @test isapprox(w3.weights, wt)

    w4 = optimise!(portfolio, NCO(; internal = NCOArgs(; type = Trad(; rm = sd))))
    wt = [0.0076957598498797125, 0.03659063445979562, 0.006133615499820552,
          0.025442339540252306, 0.007681726065167578, 0.02729123533740041,
          3.501968970304363e-8, 0.10682621097472172, 5.685197838125375e-8,
          0.007422393087381324, 0.2903400496262563, 0.013680459185368518,
          1.8064755834194454e-8, 0.11417940060244011, 4.029844446707608e-8,
          0.02299469197046356, 0.0057248401157924626, 0.19842743263305943,
          7.920501974914871e-8, 0.1295689816123123]
    @test isapprox(w4.weights, wt)

    w5 = optimise!(portfolio, NCO(; internal = NCOArgs(; type = RB(; rm = sd))))
    wt = [0.02951361849950664, 0.031104849715876133, 0.02793314036244879,
          0.02656314996044489, 0.02698369883073134, 0.03534642287440858,
          0.01670180592241015, 0.16962120247500095, 0.024968721727516394,
          0.030553868252817862, 0.053419634990817616, 0.08054389490707248,
          0.07156673974826498, 0.03978806514517936, 0.10009831144259478,
          0.09468921373208417, 0.030878020651025625, 0.043903261583117,
          0.028415495601899526, 0.037406883576782755]
    @test isapprox(w5.weights, wt, rtol = 0.0001)

    w6 = optimise!(portfolio, NCO(; internal = NCOArgs(; type = RRB())))
    wt = [0.02951385510830519, 0.031104277103431124, 0.02793205044129874,
          0.02656246758388314, 0.02698306510931954, 0.03534824108098223,
          0.01670179407097666, 0.1696243734884042, 0.024968807965950374,
          0.030553423930877767, 0.05342433481508315, 0.08054081604902462,
          0.07156220430381516, 0.03979154648185254, 0.10009210597029748,
          0.09468690756324473, 0.030877944326196508, 0.04390805560754567,
          0.028416442085015408, 0.037407286914495776]
    @test isapprox(w6.weights, wt, rtol = 5.0e-7)

    w7 = optimise!(portfolio,
                   NCO(;
                       internal = NCOArgs(;
                                          wc_kwargs = (;
                                                       wc_type = WCType(;
                                                                        box = NormalWC(;
                                                                                       seed = 123456789))),
                                          type = Trad(; rm = WCVariance(),
                                                      kelly = NoKelly(; wc_set = Box())))))
    wt = [0.009564348753807156, 0.032031527936866075, 0.0016889115186933953,
          0.01401424498276791, 5.5889348150773855e-5, 0.024504649120182027,
          7.512500556359199e-7, 0.10101644729467935, 2.443365598774725e-6,
          0.00032692976149317196, 0.3070815922395533, 0.01060870480523307,
          3.4327288853152524e-6, 0.11778932427833638, 7.992047408512792e-6,
          0.019630375733016587, 0.015381429099365612, 0.20950354632439225,
          5.105839271293184e-6, 0.13678235357224355]
    @test isapprox(w7.weights, wt)

    w8 = optimise!(portfolio, NCO(; internal = NCOArgs(; type = NOC(; rm = sd))))
    wt = [0.04657186846076186, 0.05063463283102664, 0.045222349659841156,
          0.04218487498031347, 0.048437867106046084, 0.04814986123210316,
          0.026145163082748998, 0.13968419663046647, 0.03881009297676032,
          0.047447146364048905, 0.09443844541079913, 0.020608863650339575,
          0.0005637870530288164, 0.06040728591141588, 0.0014574390260837919,
          0.052935774664661824, 0.05223809459315094, 0.07475074326336023,
          0.04606492182569711, 0.06324659127734569]
    @test isapprox(w8.weights, wt, rtol = 0.0001)
end

@testset "NCO factors" begin
    portfolio = Portfolio(; prices = prices,
                          solvers = PortOptSolver(; name = :Clarabel,
                                                  check_sol = (; allow_local = true,
                                                               allow_almost = true),
                                                  solver = Clarabel.Optimizer,
                                                  params = ["verbose" => false,
                                                            "max_step_fraction" => 0.75]))

    asset_statistics!(portfolio)
    cluster_assets!(portfolio)

    f_returns = Matrix(DataFrame(percentchange(factors))[!, 2:end])
    f_assets = colnames(factors)

    w1 = optimise!(portfolio,
                   NCO(;
                       internal = NCOArgs(; type = Trad(),
                                          port_kwargs = (; f_assets = f_assets,
                                                         f_ret = f_returns))))
    w2 = optimise!(portfolio,
                   NCO(;
                       internal = NCOArgs(; type = Trad(; class = FM(), obj = MinRisk()),
                                          port_kwargs = (; f_assets = f_assets,
                                                         f_ret = f_returns),
                                          factor_kwargs = (; mu_type = MuSimple()))))
    wt = [0.05109914899016135, 0.050215140451460485, 0.04135353593026219,
          0.02681788291627889, 0.0322776857129373, 0.060200449755958466,
          0.014445885728257432, 0.06587623061278468, 0.0390439498991421,
          0.044348093011938955, 0.10223317402175182, 0.0337502443959161,
          0.012280071607071886, 0.07520118305412048, 0.024178566447285516,
          0.04723839078418083, 0.06545813938183885, 0.08828152320800649,
          0.05832385767940247, 0.06737684641124382]
    @test isapprox(w2.weights, wt)
    @test !isapprox(w1.weights, w2.weights)

    w3 = optimise!(portfolio,
                   NCO(;
                       internal = NCOArgs(; type = RB(;),
                                          port_kwargs = (; f_assets = f_assets,
                                                         f_ret = f_returns))))

    portfolio.w_min = -Inf
    portfolio.w_max = Inf
    portfolio.short = true

    w4 = optimise!(portfolio,
                   NCO(;
                       internal = NCOArgs(; type = RB(; class = FC(; flag = false)),
                                          port_kwargs = (; f_assets = f_assets,
                                                         f_ret = f_returns),
                                          factor_kwargs = (; mu_type = MuSimple()))))

    wt = [-0.1256695060143601, 0.14480536591446272, 0.1516947592682884,
          -0.17109327653920026, -0.2374367933118665, -0.6356860179202806,
          -1.5874882850425185, 0.39974898319587926, 0.42291836086357226, -0.555325557293549,
          -0.2024907556149184, -1.4823655726361797, -1.7261986983011786, 0.7061944970950512,
          2.4107426627195694, 3.5003685389094503, 0.2086218660061672, -0.8048228091510529,
          0.3659645611354176, 0.21751767672665606]
    @test isapprox(w4.weights, wt, rtol = 0.05)
    @test !isapprox(w3.weights, w4.weights)
end

@testset "NCO" begin
    portfolio = Portfolio(; prices = prices,
                          solvers = PortOptSolver(; name = :Clarabel,
                                                  check_sol = (; allow_local = true,
                                                               allow_almost = true),
                                                  solver = Clarabel.Optimizer,
                                                  params = ["verbose" => false,
                                                            "max_step_fraction" => 0.75]))

    asset_statistics!(portfolio)
    clust_alg = DBHT()
    clust_opt = ClustOpt()
    cluster_assets!(portfolio; clust_alg = clust_alg, clust_opt = clust_opt)

    rm = Variance()
    w1 = optimise!(portfolio,
                   NCO(; internal = NCOArgs(; type = Trad(; rm = rm, obj = MinRisk()))))
    w2 = optimise!(portfolio,
                   NCO(;
                       internal = NCOArgs(; type = Trad(; rm = rm, obj = Utility(; l = l)))))
    w3 = optimise!(portfolio,
                   NCO(;
                       internal = NCOArgs(;
                                          type = Trad(; rm = rm, obj = Sharpe(; rf = rf)))))
    w4 = optimise!(portfolio,
                   NCO(; internal = NCOArgs(; type = Trad(; rm = rm, obj = MaxRet()))))
    w5 = optimise!(portfolio, NCO(; internal = NCOArgs(; type = RB(; rm = rm))))
    w1t = [0.013937085585839061, 0.04359528939613315, 0.010055244196525516,
           0.018529417015299345, 0.0060008557954075595, 0.06377945741010448,
           8.830053665226257e-7, 0.13066772968460527, 7.275669461246384e-6,
           1.1522825766672644e-5, 0.2351158704607046, 0.005999396004399579,
           3.2750802699731945e-7, 0.11215293274059893, 5.127280882607535e-7,
           4.000811842370495e-6, 0.037474931372346656, 0.18536090811244715,
           0.04037088344151298, 0.09693547623552375]
    w2t = [1.4130328347864308e-8, 3.340702175929791e-8, 4.246858404517726e-8,
           3.2911083951213625e-8, 0.8703104182142611, 4.372050354384955e-18,
           0.12797886299501354, 1.6690708257671003e-8, 1.0786083316193095e-14,
           0.0007066153745743747, 2.3710326355760465e-14, 2.9510811113524065e-18,
           3.0575293462043653e-12, 9.634091319602901e-16, 3.4402431905565705e-16,
           0.0009978200918112112, 5.917294076308012e-6, 1.8271910477876723e-8,
           1.8281126799512493e-7, 2.533626547634067e-8]
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
    w6 = optimise!(portfolio,
                   NCO(; internal = NCOArgs(; type = Trad(; rm = rm, obj = MinRisk()))))
    w7 = optimise!(portfolio,
                   NCO(;
                       internal = NCOArgs(; type = Trad(; rm = rm, obj = Utility(; l = l)))))
    w8 = optimise!(portfolio,
                   NCO(;
                       internal = NCOArgs(;
                                          type = Trad(; rm = rm, obj = Sharpe(; rf = rf)))))
    w9 = optimise!(portfolio,
                   NCO(; internal = NCOArgs(; type = Trad(; rm = rm, obj = MaxRet()))))
    w10 = optimise!(portfolio, NCO(; internal = NCOArgs(; type = RB(; rm = rm))))
    w1t = [0.04059732452480461, 0.045600642166333714, 0.0008508832264865543,
           0.006437853767255612, 0.0028417982769144107, 0.05930364215554,
           3.9051667245469696e-9, 0.15293341427728768, 0.005759784188884512,
           1.652344063919435e-8, 0.19875502188644184, 1.0514248695659101e-9,
           1.4780072332606018e-10, 0.1047391054319265, 9.539443511561501e-11,
           6.074709477576578e-9, 0.05650542360887961, 0.19033546520370817,
           0.04096914490424977, 0.09437046858335037]
    w2t = [0.021007904715709336, 0.04431479750515221, 0.003306656556282061,
           0.008618330291953587, 0.034539688784537184, 0.04052261321637723,
           2.0651483846297533e-9, 0.15142712053644636, 0.014256524135172746,
           0.008941938346859975, 0.19513582085940215, 2.108754372098641e-10,
           3.500860845543409e-5, 0.08578220696546696, 1.4377785984834448e-11,
           0.0035592330900482503, 0.062174337116155554, 0.18004235846230465,
           0.04875777829777634, 0.09757768022149844]
    w3t = [1.6488442933157733e-9, 6.094597446158297e-9, 4.352701116356979e-9,
           3.216675055419697e-9, 0.6265423411545956, 3.55564504920547e-12,
           0.041010603937969514, 3.4532909517510285e-9, 4.992008759782647e-10,
           2.4410459827490494e-9, 1.6731654390844092e-10, 3.2851721535374462e-12,
           3.53007818516829e-12, 9.26245662419354e-12, 9.394680382529403e-13,
           0.13258943072391433, 0.1453950018580843, 3.851867293138941e-9,
           0.05446259135029774, 5.229026147704124e-9]
    w4t = [1.0697281800757452e-7, 1.1311775297941919e-7, 1.3780259688825507e-7,
           1.268055655932939e-7, 8.50822923994295e-6, 3.204283203701939e-13,
           0.9999862377026062, 8.136173647277906e-8, 4.024041566311586e-11,
           6.94984039611021e-12, 8.035455863957799e-13, 3.4976521519678484e-13,
           1.137123602399115e-12, 5.126371707340376e-13, 2.1878366160025683e-13,
           2.7532186682419475e-6, 1.431604095722679e-7, 8.5069527979556e-8,
           1.6044460686057254e-6, 1.0206247702601848e-7]
    w5t = [0.0384820255947251, 0.04124424217098382, 0.036313192676561065,
           0.030686739825600513, 0.032719507454256655, 0.056962822893488835,
           0.01903250914065054, 0.060249581877158384, 0.04150918291963332,
           0.10990582270170848, 0.08278646424688123, 0.034636409979171445,
           0.03958507278580084, 0.0588551505009234, 0.026494341575903916,
           0.08543088238865078, 0.043231785208553426, 0.06156935974506197,
           0.047936585368150555, 0.05236832094613573]
    @test isapprox(w6.weights, w1t)
    @test isapprox(w7.weights, w2t)
    @test isapprox(w8.weights, w3t)
    @test isapprox(w9.weights, w4t)
    @test isapprox(w10.weights, w5t, rtol = 5.0e-5)

    rm = SSD()
    w11 = optimise!(portfolio,
                    NCO(; internal = NCOArgs(; type = Trad(; rm = rm, obj = MinRisk()))))
    w12 = optimise!(portfolio,
                    NCO(;
                        internal = NCOArgs(;
                                           type = Trad(; rm = rm, obj = Utility(; l = l)))))
    w13 = optimise!(portfolio,
                    NCO(;
                        internal = NCOArgs(;
                                           type = Trad(; rm = rm, obj = Sharpe(; rf = rf)))))
    w14 = optimise!(portfolio,
                    NCO(; internal = NCOArgs(; type = Trad(; rm = rm, obj = MaxRet()))))
    w15 = optimise!(portfolio, NCO(; internal = NCOArgs(; type = RB(; rm = rm))))
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
    w16 = optimise!(portfolio,
                    NCO(; internal = NCOArgs(; type = Trad(; rm = rm, obj = MinRisk()))))
    w17 = optimise!(portfolio,
                    NCO(;
                        internal = NCOArgs(;
                                           type = Trad(; rm = rm, obj = Utility(; l = l)))))
    w18 = optimise!(portfolio,
                    NCO(;
                        internal = NCOArgs(;
                                           type = Trad(; rm = rm, obj = Sharpe(; rf = rf)))))
    w19 = optimise!(portfolio,
                    NCO(; internal = NCOArgs(; type = Trad(; rm = rm, obj = MaxRet()))))
    w20 = optimise!(portfolio, NCO(; internal = NCOArgs(; type = RB(; rm = rm))))
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
    w5t = [0.04018453570301376, 0.04342141273452162, 0.03800007001346819,
           0.03174624368911299, 0.037297611610450795, 0.04966618492167956,
           0.02023619079312285, 0.06045982005363625, 0.0428777395286891,
           0.10829686831504556, 0.08072312203473789, 0.03142696468192543,
           0.03708021422319735, 0.0556911215795128, 0.023773858320579242,
           0.0892294571140047, 0.045098034877630935, 0.06138013297454211,
           0.050457652969473174, 0.05295276386165584]
    @test isapprox(w16.weights, w1t)
    @test isapprox(w17.weights, w2t)
    @test isapprox(w18.weights, w3t, rtol = 5.0e-5)
    @test isapprox(w19.weights, w4t)
    @test isapprox(w20.weights, w5t, rtol = 5.0e-5)

    rm = SLPM(; target = rf)
    w21 = optimise!(portfolio,
                    NCO(; internal = NCOArgs(; type = Trad(; rm = rm, obj = MinRisk()))))
    w22 = optimise!(portfolio,
                    NCO(;
                        internal = NCOArgs(;
                                           type = Trad(; rm = rm, obj = Utility(; l = l)))))
    w23 = optimise!(portfolio,
                    NCO(;
                        internal = NCOArgs(;
                                           type = Trad(; rm = rm, obj = Sharpe(; rf = rf)))))
    w24 = optimise!(portfolio,
                    NCO(; internal = NCOArgs(; type = Trad(; rm = rm, obj = MaxRet()))))
    w25 = optimise!(portfolio, NCO(; internal = NCOArgs(; type = RB(; rm = rm))))
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
    w26 = optimise!(portfolio,
                    NCO(; internal = NCOArgs(; type = Trad(; rm = rm, obj = MinRisk()))))
    w27 = optimise!(portfolio,
                    NCO(;
                        internal = NCOArgs(;
                                           type = Trad(; rm = rm, obj = Utility(; l = l)))))
    w28 = optimise!(portfolio,
                    NCO(;
                        internal = NCOArgs(;
                                           type = Trad(; rm = rm, obj = Sharpe(; rf = rf)))))
    w29 = optimise!(portfolio,
                    NCO(; internal = NCOArgs(; type = Trad(; rm = rm, obj = MaxRet()))))
    w30 = optimise!(portfolio, NCO(; internal = NCOArgs(; type = RB(; rm = rm))))
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
    w31 = optimise!(portfolio,
                    NCO(; internal = NCOArgs(; type = Trad(; rm = rm, obj = MinRisk()))))
    w32 = optimise!(portfolio,
                    NCO(;
                        internal = NCOArgs(;
                                           type = Trad(; rm = rm, obj = Utility(; l = l)))))
    w33 = optimise!(portfolio,
                    NCO(;
                        internal = NCOArgs(;
                                           type = Trad(; rm = rm, obj = Sharpe(; rf = rf)))))
    w34 = optimise!(portfolio,
                    NCO(; internal = NCOArgs(; type = Trad(; rm = rm, obj = MaxRet()))))
    w35 = optimise!(portfolio, NCO(; internal = NCOArgs(; type = RB(; rm = rm))))
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
    w36 = optimise!(portfolio,
                    NCO(; internal = NCOArgs(; type = Trad(; rm = rm, obj = MinRisk()))))
    w37 = optimise!(portfolio,
                    NCO(;
                        internal = NCOArgs(;
                                           type = Trad(; rm = rm, obj = Utility(; l = l)))))
    w38 = optimise!(portfolio,
                    NCO(;
                        internal = NCOArgs(;
                                           type = Trad(; rm = rm, obj = Sharpe(; rf = rf)))))
    w39 = optimise!(portfolio,
                    NCO(; internal = NCOArgs(; type = Trad(; rm = rm, obj = MaxRet()))))
    w40 = optimise!(portfolio, NCO(; internal = NCOArgs(; type = RB(; rm = rm))))
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

    portfolio.solvers = PortOptSolver(; name = :Clarabel, solver = Clarabel.Optimizer,
                                      check_sol = (; allow_local = true,
                                                   allow_almost = true),
                                      params = ["verbose" => false,
                                                "max_step_fraction" => 0.65])

    rm = RLVaR()
    w41 = optimise!(portfolio,
                    NCO(; internal = NCOArgs(; type = Trad(; rm = rm, obj = MinRisk()))))
    w42 = optimise!(portfolio,
                    NCO(;
                        internal = NCOArgs(;
                                           type = Trad(; rm = rm, obj = Utility(; l = l)))))
    w43 = optimise!(portfolio,
                    NCO(;
                        internal = NCOArgs(;
                                           type = Trad(; rm = rm, obj = Sharpe(; rf = rf)))))
    w44 = optimise!(portfolio,
                    NCO(; internal = NCOArgs(; type = Trad(; rm = rm, obj = MaxRet()))))
    w45 = optimise!(portfolio, NCO(; internal = NCOArgs(; type = RB(; rm = rm))))
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
    w3t = [7.407935084500628e-9, 0.0038186792670059123, 5.834775791503411e-9,
           2.0062367407514654e-8, 0.5334271300529785, 1.8393891760014976e-11,
           0.17928326973218717, 0.0009340083525981799, 2.6935237631265187e-10,
           0.0364510267932581, 0.00147220170010373, 2.4849095476847513e-11,
           1.3665815061576729e-10, 4.440750509444722e-11, 1.1480174305845337e-11,
           0.018261857522027844, 0.1331581349879938, 0.08717736372707571,
           0.006016290626369066, 3.428182292139245e-9]
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
    @test isapprox(w42.weights, w2t, rtol = 1.0e-5)
    @test isapprox(w43.weights, w3t, rtol = 5.0e-5)
    @test isapprox(w44.weights, w4t, rtol = 5.0e-6)
    @test isapprox(w45.weights, w5t, rtol = 5.0e-6)

    portfolio.solvers = PortOptSolver(; name = :Clarabel, solver = Clarabel.Optimizer,
                                      check_sol = (; allow_local = true,
                                                   allow_almost = true),
                                      params = ["verbose" => false,
                                                "max_step_fraction" => 0.75])

    rm = MDD()
    w46 = optimise!(portfolio,
                    NCO(; internal = NCOArgs(; type = Trad(; rm = rm, obj = MinRisk()))))
    w47 = optimise!(portfolio,
                    NCO(;
                        internal = NCOArgs(;
                                           type = Trad(; rm = rm, obj = Utility(; l = l)))))
    w48 = optimise!(portfolio,
                    NCO(;
                        internal = NCOArgs(;
                                           type = Trad(; rm = rm, obj = Sharpe(; rf = rf)))))
    w49 = optimise!(portfolio,
                    NCO(; internal = NCOArgs(; type = Trad(; rm = rm, obj = MaxRet()))))
    w50 = optimise!(portfolio, NCO(; internal = NCOArgs(; type = RB(; rm = rm))))
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
    w51 = optimise!(portfolio,
                    NCO(; internal = NCOArgs(; type = Trad(; rm = rm, obj = MinRisk()))))
    w52 = optimise!(portfolio,
                    NCO(;
                        internal = NCOArgs(;
                                           type = Trad(; rm = rm, obj = Utility(; l = l)))))
    w53 = optimise!(portfolio,
                    NCO(;
                        internal = NCOArgs(;
                                           type = Trad(; rm = rm, obj = Sharpe(; rf = rf)))))
    w54 = optimise!(portfolio,
                    NCO(; internal = NCOArgs(; type = Trad(; rm = rm, obj = MaxRet()))))
    w55 = optimise!(portfolio, NCO(; internal = NCOArgs(; type = RB(; rm = rm))))
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
    w56 = optimise!(portfolio,
                    NCO(; internal = NCOArgs(; type = Trad(; rm = rm, obj = MinRisk()))))
    w57 = optimise!(portfolio,
                    NCO(;
                        internal = NCOArgs(;
                                           type = Trad(; rm = rm, obj = Utility(; l = l)))))
    w58 = optimise!(portfolio,
                    NCO(;
                        internal = NCOArgs(;
                                           type = Trad(; rm = rm, obj = Sharpe(; rf = rf)))))
    w59 = optimise!(portfolio,
                    NCO(; internal = NCOArgs(; type = Trad(; rm = rm, obj = MaxRet()))))
    w60 = optimise!(portfolio, NCO(; internal = NCOArgs(; type = RB(; rm = rm))))
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
    w61 = optimise!(portfolio,
                    NCO(; internal = NCOArgs(; type = Trad(; rm = rm, obj = MinRisk()))))
    w62 = optimise!(portfolio,
                    NCO(;
                        internal = NCOArgs(;
                                           type = Trad(; rm = rm, obj = Utility(; l = l)))))
    w63 = optimise!(portfolio,
                    NCO(;
                        internal = NCOArgs(;
                                           type = Trad(; rm = rm, obj = Sharpe(; rf = rf)))))
    w64 = optimise!(portfolio,
                    NCO(; internal = NCOArgs(; type = Trad(; rm = rm, obj = MaxRet()))))
    w65 = optimise!(portfolio, NCO(; internal = NCOArgs(; type = RB(; rm = rm))))
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
    w66 = optimise!(portfolio,
                    NCO(; internal = NCOArgs(; type = Trad(; rm = rm, obj = MinRisk()))))
    w67 = optimise!(portfolio,
                    NCO(;
                        internal = NCOArgs(;
                                           type = Trad(; rm = rm, obj = Utility(; l = l)))))
    w68 = optimise!(portfolio,
                    NCO(;
                        internal = NCOArgs(;
                                           type = Trad(; rm = rm, obj = Sharpe(; rf = rf)))))
    w69 = optimise!(portfolio,
                    NCO(; internal = NCOArgs(; type = Trad(; rm = rm, obj = MaxRet()))))
    w70 = optimise!(portfolio, NCO(; internal = NCOArgs(; type = RB(; rm = rm))))
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
    @test isapprox(w68.weights, w3t, rtol = 5.0e-5)
    @test isapprox(w69.weights, w4t)
    @test isapprox(w70.weights, w5t, rtol = 5.0e-5)

    rm = RLDaR()
    w71 = optimise!(portfolio,
                    NCO(; internal = NCOArgs(; type = Trad(; rm = rm, obj = MinRisk()))))
    w72 = optimise!(portfolio,
                    NCO(;
                        internal = NCOArgs(;
                                           type = Trad(; rm = rm, obj = Utility(; l = l)))))
    w73 = optimise!(portfolio,
                    NCO(;
                        internal = NCOArgs(;
                                           type = Trad(; rm = rm, obj = Sharpe(; rf = rf)))))
    w74 = optimise!(portfolio,
                    NCO(; internal = NCOArgs(; type = Trad(; rm = rm, obj = MaxRet()))))
    w75 = optimise!(portfolio, NCO(; internal = NCOArgs(; type = RB(; rm = rm))))
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
    @test isapprox(w72.weights, w2t, rtol = 5.0e-6)
    @test isapprox(w73.weights, w3t, rtol = 5.0e-5)
    @test isapprox(w74.weights, w4t, rtol = 5.0e-6)
    @test isapprox(w75.weights, w5t, rtol = 5.0e-6)

    rm = Kurt()
    w76 = optimise!(portfolio,
                    NCO(; internal = NCOArgs(; type = Trad(; rm = rm, obj = MinRisk()))))
    w77 = optimise!(portfolio,
                    NCO(;
                        external = NCOArgs(;
                                           type = Trad(; rm = rm, obj = Utility(; l = l))),
                        internal = NCOArgs(;
                                           type = Trad(; rm = [Kurt(; kt = portfolio.kurt)],
                                                       obj = Utility(; l = l)))))
    w78 = optimise!(portfolio,
                    NCO(;
                        external = NCOArgs(;
                                           type = Trad(; rm = rm, obj = Sharpe(; rf = rf))),
                        internal = NCOArgs(;
                                           type = Trad(; rm = [Kurt(; kt = portfolio.kurt)],
                                                       obj = Sharpe(; rf = rf)))))
    w79 = optimise!(portfolio,
                    NCO(;
                        external = NCOArgs(;
                                           type = Trad(; rm = Kurt(; kt = portfolio.kurt),
                                                       obj = MaxRet())),
                        internal = NCOArgs(;
                                           type = Trad(; rm = [Kurt(; kt = portfolio.kurt)],
                                                       obj = MaxRet()))))
    w80 = optimise!(portfolio, NCO(; internal = NCOArgs(; type = RB(; rm = rm))))
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
    @test isapprox(w76.weights, w1t, rtol = 5.0e-7)
    @test isapprox(w77.weights, w2t, rtol = 5.0e-5)
    @test isapprox(w78.weights, w3t, rtol = 5.0e-7)
    @test isapprox(w79.weights, w4t)
    @test isapprox(w80.weights, w5t, rtol = 0.0005)

    w81 = optimise!(portfolio,
                    NCO(;
                        internal = NCOArgs(;
                                           type = Trad(; rm = Kurt(; kt = portfolio.kurt),
                                                       obj = MinRisk()),
                                           port_kwargs = (; max_num_assets_kurt = 1)),
                        external = NCOArgs(; type = Trad(; rm = rm, obj = MinRisk()),
                                           port_kwargs = (; max_num_assets_kurt = 1))))
    w82 = optimise!(portfolio,
                    NCO(;
                        internal = NCOArgs(;
                                           type = Trad(; rm = [rm], obj = Utility(; l = l)),
                                           port_kwargs = (; max_num_assets_kurt = 1)),
                        external = NCOArgs(;
                                           type = Trad(; rm = [rm], obj = Utility(; l = l)),
                                           port_kwargs = (; max_num_assets_kurt = 1))))
    w83 = optimise!(portfolio,
                    NCO(;
                        internal = NCOArgs(;
                                           type = Trad(; rm = [Kurt(; kt = portfolio.kurt)],
                                                       obj = Sharpe(; rf = rf)),
                                           port_kwargs = (; max_num_assets_kurt = 1)),
                        external = NCOArgs(;
                                           type = Trad(; rm = rm, obj = Sharpe(; rf = rf)),
                                           port_kwargs = (; max_num_assets_kurt = 1))))
    w84 = optimise!(portfolio,
                    NCO(;
                        internal = NCOArgs(; type = Trad(; rm = rm, obj = MaxRet()),
                                           port_kwargs = (; max_num_assets_kurt = 1))))
    w85 = optimise!(portfolio,
                    NCO(;
                        internal = NCOArgs(; type = RB(; rm = rm),
                                           port_kwargs = (; max_num_assets_kurt = 1))))
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
    w5t = [0.03157804982018643, 0.03747680727891143, 0.031165822488220623,
           0.03450967076494636, 0.027883709488852788, 0.05591130187352058,
           0.023528893250812543, 0.05776503882705131, 0.04202791902100599,
           0.14953183942071227, 0.08641474530359258, 0.03414215898987479,
           0.04713467723576176, 0.05678828061160823, 0.03244144965344055,
           0.07602249414320358, 0.0370236010877867, 0.051834001667857996,
           0.04550276305236839, 0.041316776020285176]
    @test isapprox(w81.weights, w1t)
    @test isapprox(w82.weights, w2t)
    @test isapprox(w83.weights, w3t, rtol = 5.0e-7)
    @test isapprox(w84.weights, w4t)
    @test isapprox(w85.weights, w5t, rtol = 5.0e-5)

    rm = SKurt()
    w86 = optimise!(portfolio,
                    NCO(; internal = NCOArgs(; type = Trad(; rm = rm, obj = MinRisk()))))
    w87 = optimise!(portfolio,
                    NCO(;
                        internal = NCOArgs(;
                                           type = Trad(; rm = SKurt(; kt = portfolio.skurt),
                                                       obj = Utility(; l = l))),
                        external = NCOArgs(;
                                           type = Trad(; rm = SKurt(; kt = portfolio.skurt),
                                                       obj = Utility(; l = l)))))
    w88 = optimise!(portfolio,
                    NCO(;
                        internal = NCOArgs(;
                                           type = Trad(;
                                                       rm = [SKurt(; kt = portfolio.skurt)],
                                                       obj = Sharpe(; rf = rf)))))
    w89 = optimise!(portfolio,
                    NCO(;
                        internal = NCOArgs(;
                                           type = Trad(;
                                                       rm = [SKurt(; kt = portfolio.skurt)],
                                                       obj = MaxRet())),
                        external = NCOArgs(; type = Trad(; rm = rm, obj = MaxRet()))))
    w90 = optimise!(portfolio, NCO(; internal = NCOArgs(; type = RB(; rm = rm))))
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
    @test isapprox(w87.weights, w2t, rtol = 5.0e-7)
    @test isapprox(w88.weights, w3t, rtol = 5.0e-6)
    @test isapprox(w89.weights, w4t)
    @test isapprox(w90.weights, w5t, rtol = 0.0005)

    w91 = optimise!(portfolio,
                    NCO(;
                        internal = NCOArgs(;
                                           type = Trad(; rm = SKurt(; kt = portfolio.skurt),
                                                       obj = MinRisk()),
                                           port_kwargs = (; max_num_assets_kurt = 1)),
                        external = NCOArgs(; type = Trad(; rm = rm, obj = MinRisk()),
                                           port_kwargs = (; max_num_assets_kurt = 1))))
    w92 = optimise!(portfolio,
                    NCO(;
                        internal = NCOArgs(;
                                           type = Trad(; rm = [rm], obj = Utility(; l = l)),
                                           port_kwargs = (; max_num_assets_kurt = 1)),
                        external = NCOArgs(;
                                           type = Trad(; rm = [rm], obj = Utility(; l = l)),
                                           port_kwargs = (; max_num_assets_kurt = 1))))
    w93 = optimise!(portfolio,
                    NCO(;
                        internal = NCOArgs(;
                                           type = Trad(;
                                                       rm = [SKurt(; kt = portfolio.skurt)],
                                                       obj = Sharpe(; rf = rf)),
                                           port_kwargs = (; max_num_assets_kurt = 1)),
                        external = NCOArgs(;
                                           type = Trad(; rm = SKurt(; kt = portfolio.skurt),
                                                       obj = Sharpe(; rf = rf)),
                                           port_kwargs = (; max_num_assets_kurt = 1))))
    w94 = optimise!(portfolio,
                    NCO(;
                        internal = NCOArgs(; type = Trad(; rm = rm, obj = MaxRet()),
                                           port_kwargs = (; max_num_assets_kurt = 1))))
    w95 = optimise!(portfolio,
                    NCO(;
                        internal = NCOArgs(; type = RB(; rm = rm),
                                           port_kwargs = (; max_num_assets_kurt = 1))))
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
    w5t = [0.03371492622823459, 0.03885680909202028, 0.0295127116904648,
           0.03149315617014066, 0.029024572323534302, 0.055761432358667914,
           0.02201107558542142, 0.05490639185688602, 0.04042054368876062,
           0.13169734668204794, 0.08288881287730689, 0.033215260143894476,
           0.05914956966941689, 0.057450290453859534, 0.036041024444193694,
           0.0878154083390974, 0.037166407656602715, 0.0520191294135219,
           0.04552878788429187, 0.04132634344163601]
    @test isapprox(w91.weights, w1t)
    @test isapprox(w92.weights, w2t)
    @test isapprox(w93.weights, w3t, rtol = 5.0e-6)
    @test isapprox(w94.weights, w4t)
    @test isapprox(w95.weights, w5t, rtol = 5.0e-5)

    rm = Skew()
    w96 = optimise!(portfolio,
                    NCO(; internal = NCOArgs(; type = Trad(; rm = rm, obj = MinRisk()))))
    w97 = optimise!(portfolio,
                    NCO(;
                        internal = NCOArgs(;
                                           type = Trad(; rm = rm, obj = Utility(; l = l)))))
    w98 = optimise!(portfolio,
                    NCO(;
                        internal = NCOArgs(;
                                           type = Trad(; rm = rm, obj = Sharpe(; rf = rf)))))
    w99 = optimise!(portfolio,
                    NCO(; internal = NCOArgs(; type = Trad(; rm = rm, obj = MaxRet()))))
    w100 = optimise!(portfolio, NCO(; internal = NCOArgs(; type = RB(; rm = rm))))
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
    w5t = [0.025897739886635695, 0.0326561283379726, 0.014168996668904858,
           0.015810701080819185, 0.021577620843748826, 0.01972115889178686,
           0.011183145022198507, 0.04055388604205418, 0.026161010017240596,
           0.13683318786889617, 0.05401366367572873, 0.013262684492999192,
           0.3221636111063777, 0.031804433196940284, 0.030844131872616542,
           0.07582332972244796, 0.020213630860370833, 0.039948060105306306,
           0.038237112051432244, 0.029125768255522586]
    @test isapprox(w96.weights, w1t)
    @test isapprox(w97.weights, w2t, rtol = 0.0001)
    @test isapprox(w98.weights, w3t)
    @test isapprox(w99.weights, w4t)
    @test isapprox(w100.weights, w5t, rtol = 5.0e-7)

    rm = SSkew()
    w101 = optimise!(portfolio,
                     NCO(; internal = NCOArgs(; type = Trad(; rm = rm, obj = MinRisk()))))
    w102 = optimise!(portfolio,
                     NCO(;
                         internal = NCOArgs(;
                                            type = Trad(; rm = rm, obj = Utility(; l = l)))))
    w103 = optimise!(portfolio,
                     NCO(;
                         internal = NCOArgs(;
                                            type = Trad(; rm = rm, obj = Sharpe(; rf = rf)))))
    w104 = optimise!(portfolio,
                     NCO(; internal = NCOArgs(; type = Trad(; rm = rm, obj = MaxRet()))))
    w105 = optimise!(portfolio, NCO(; internal = NCOArgs(; type = RB(; rm = rm))))
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
    w5t = [0.033929154306477985, 0.038640214714792745, 0.030156418325532557,
           0.029484313087808987, 0.03172662386112446, 0.05301869378579685,
           0.023800595405927695, 0.06206195710022938, 0.04025962153252248,
           0.12466431877947569, 0.09069730279461317, 0.031035927494324978,
           0.04653954212676665, 0.059522058769001955, 0.031662911168944985,
           0.08587925177605744, 0.03763306930178136, 0.05894723169753126,
           0.04715977534634794, 0.04318101862494148]
    @test isapprox(w101.weights, w1t, rtol = 5.0e-8)
    @test isapprox(w102.weights, w2t, rtol = 5.0e-5)
    @test isapprox(w103.weights, w3t)
    @test isapprox(w104.weights, w4t)
    @test isapprox(w105.weights, w5t, rtol = 5.0e-6)

    portfolio = Portfolio(; prices = prices[(end - 25):end],
                          solvers = PortOptSolver(; name = :Clarabel,
                                                  check_sol = (; allow_local = true,
                                                               allow_almost = true),
                                                  solver = Clarabel.Optimizer,
                                                  params = ["verbose" => false,
                                                            "max_step_fraction" => 0.75]))

    asset_statistics!(portfolio)
    clust_alg = HAC()
    cluster_assets!(portfolio; clust_alg = clust_alg, clust_opt = clust_opt)
    rm = BDVariance()
    w106 = optimise!(portfolio,
                     NCO(; internal = NCOArgs(; type = Trad(; rm = rm, obj = MinRisk()))))
    w107 = optimise!(portfolio,
                     NCO(;
                         internal = NCOArgs(;
                                            type = Trad(; rm = rm, obj = Utility(; l = l)))))
    w108 = optimise!(portfolio,
                     NCO(;
                         internal = NCOArgs(;
                                            type = Trad(; rm = rm, obj = Sharpe(; rf = rf)))))
    w109 = optimise!(portfolio,
                     NCO(; internal = NCOArgs(; type = Trad(; rm = rm, obj = MaxRet()))))
    w110 = optimise!(portfolio, NCO(; internal = NCOArgs(; type = RB(; rm = rm))))
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
    w5t = [0.03511450632064662, 0.04430345595107029, 0.06148437221308192,
           0.029650071751310066, 0.03213766138980017, 0.052932648277201955,
           0.06621288541172803, 0.058396547728577546, 0.035368505906975364,
           0.042838902258038596, 0.07938844096835036, 0.040166785803924646,
           0.01906452173742589, 0.11336283742206299, 0.04440054812302461,
           0.042255677018139745, 0.04025377718314333, 0.06571441493632324,
           0.040126856265433684, 0.056826583333741]
    @test isapprox(w106.weights, w1t)
    @test isapprox(w107.weights, w2t)
    @test isapprox(w108.weights, w3t)
    @test isapprox(w109.weights, w4t)
    @test isapprox(w110.weights, w5t, rtol = 5.0e-5)
end

@testset "HRP and HERC risk scale" begin
    portfolio = Portfolio(; prices = prices,
                          solvers = PortOptSolver(; name = :Clarabel,
                                                  check_sol = (; allow_local = true,
                                                               allow_almost = true),
                                                  solver = Clarabel.Optimizer,
                                                  params = ["verbose" => false,
                                                            "max_step_fraction" => 0.75]))

    asset_statistics!(portfolio)
    cluster_assets!(portfolio)

    w1 = optimise!(portfolio, HRP(; rm = SD()))
    w2 = optimise!(portfolio, HRP(; rm = CDaR()))
    w3 = optimise!(portfolio, HRP(; rm = [SD(; sigma = portfolio.cov), CDaR()]))
    w4 = optimise!(portfolio,
                   HRP(; rm = [SD(), CDaR(; settings = RMSettings(; scale = 1e-10))]))
    w5 = optimise!(portfolio,
                   HRP(; rm = [SD(), CDaR(; settings = RMSettings(; scale = 1e10))]))
    w6 = optimise!(portfolio,
                   HRP(;
                       rm = [CDaR(),
                             SD(; sigma = portfolio.cov,
                                settings = RMSettings(; scale = 1e-10))]))
    w7 = optimise!(portfolio,
                   HRP(; rm = [CDaR(), SD(; settings = RMSettings(; scale = 1e10))]))
    @test isapprox(w1.weights, w4.weights)
    @test isapprox(w2.weights, w5.weights)
    @test isapprox(w2.weights, w6.weights)
    @test isapprox(w1.weights, w7.weights)
    @test !isapprox(w1.weights, w3.weights)
    @test !isapprox(w2.weights, w3.weights)

    w1 = optimise!(portfolio, HERC(; rm = SD()))
    w2 = optimise!(portfolio, HERC(; rm = CDaR()))
    w3 = optimise!(portfolio, HERC(; rm = [SD(; sigma = portfolio.cov), CDaR()]))
    w4 = optimise!(portfolio,
                   HERC(; rm = [SD(), CDaR(; settings = RMSettings(; scale = 1e-10))]))
    w5 = optimise!(portfolio,
                   HERC(;
                        rm = [CDaR(),
                              SD(; sigma = portfolio.cov,
                                 settings = RMSettings(; scale = 1e-10))]))
    @test isapprox(w1.weights, w4.weights)
    @test isapprox(w2.weights, w5.weights)
    @test !isapprox(w1.weights, w3.weights)
    @test !isapprox(w2.weights, w3.weights)
end

@testset "HRP" begin
    portfolio = Portfolio(; prices = prices,
                          solvers = PortOptSolver(; name = :Clarabel,
                                                  check_sol = (; allow_local = true,
                                                               allow_almost = true),
                                                  solver = Clarabel.Optimizer,
                                                  params = ["verbose" => false,
                                                            "max_step_fraction" => 0.75]))

    asset_statistics!(portfolio)
    clust_alg = DBHT()
    clust_opt = ClustOpt()
    cluster_assets!(portfolio; clust_alg = clust_alg, clust_opt = clust_opt)

    w1 = optimise!(portfolio, HRP(; rm = SD()))
    wt = [0.03692879524929352, 0.06173471900996101, 0.05788485449055379,
          0.02673494374797732, 0.051021461747188426, 0.06928891082994745,
          0.024148698656475776, 0.047867702815293824, 0.03125471206249845,
          0.0647453556976089, 0.10088112288028349, 0.038950479132407664,
          0.025466032983548218, 0.046088041928035575, 0.017522279227008376,
          0.04994166864441962, 0.076922254387969, 0.05541444481863126, 0.03819947378487138,
          0.07900404790602693]
    @test isapprox(w1.weights, wt)

    w2 = optimise!(portfolio, HRP(; rm = MAD()))
    wt = [0.03818384697661099, 0.0622147651480733, 0.06006655971419491,
          0.025623934919921716, 0.05562388497425944, 0.0707725175955155,
          0.025813212789699617, 0.0505495225915194, 0.030626076629624386,
          0.06238080360591265, 0.09374966122500297, 0.04170897361395994, 0.0238168503353984,
          0.04472819614772452, 0.016127737407569318, 0.05166088174144377,
          0.07470415476814536, 0.05398093542616027, 0.03829328495650196, 0.0793741994327616]
    @test isapprox(w2.weights, wt)

    w3 = optimise!(portfolio, HRP(; rm = SSD()))
    wt = [0.03685118468964678, 0.060679678711961914, 0.05670047906355952,
          0.026116775736648376, 0.05374470957245603, 0.07054365460247647,
          0.025708179197816795, 0.04589886455582043, 0.03028786387008358,
          0.0634613352386214, 0.10260291820653494, 0.038697426540672014,
          0.027823289120792207, 0.04746469792344547, 0.01859405658396672,
          0.052617311519562025, 0.07250127037171229, 0.05655029966386585,
          0.03752315313496762, 0.07563285169538961]
    @test isapprox(w3.weights, wt)

    w4 = optimise!(portfolio, HRP(; rm = FLPM(; target = rf)))
    wt = [0.039336538617540336, 0.06395492117890568, 0.06276820986537449,
          0.026249574972701337, 0.06242254734451266, 0.06144221460126541,
          0.026935168312631458, 0.05003386507226852, 0.031591718694359686,
          0.062032838812258684, 0.08971950983943798, 0.03794006123084816,
          0.021868414535268534, 0.04077527283143069, 0.014126622482894119,
          0.053377350594462684, 0.07977242881987609, 0.05414859107042937,
          0.04026174326124142, 0.0812424078622927]
    @test isapprox(w4.weights, wt)

    w5 = optimise!(portfolio, HRP(; rm = SLPM(; target = rf)))
    wt = [0.03752644591926642, 0.06162667128035028, 0.058061904976983145,
          0.02658920517037755, 0.056905956197316004, 0.06572780540966205,
          0.026284445157011994, 0.04570757302464124, 0.030832527351205462,
          0.06360209500884424, 0.10038131818271241, 0.03703703080055575,
          0.02666888092950761, 0.04516285946254531, 0.01726932945234202,
          0.053613204742961086, 0.07517335594118413, 0.05681780552493213,
          0.03852165318888371, 0.07648993227871743]
    @test isapprox(w5.weights, wt)

    w6 = optimise!(portfolio, HRP(; rm = WR()))
    wt = [0.05025200880802368, 0.055880855583038826, 0.05675550980104009,
          0.03045034754117226, 0.046067750416348405, 0.054698528388037695,
          0.02387947919504387, 0.03257444044988314, 0.02852023603441257,
          0.06948149861799151, 0.10142635777300404, 0.02403801691456373, 0.0376186609725536,
          0.0489137277286356, 0.023542345142850748, 0.05182849839922806,
          0.12037415047550741, 0.06259799738419287, 0.03040731610608419,
          0.050692274268387696]
    @test isapprox(w6.weights, wt)

    w7 = optimise!(portfolio, HRP(; rm = CVaR()))
    wt = [0.03557588495882196, 0.06070721124578364, 0.055279033097091264,
          0.02659326453716366, 0.053554687039556674, 0.06617470037072191,
          0.0262740407048504, 0.04720203604560211, 0.02912774720142737, 0.06321093829691052,
          0.10241165286325946, 0.03903551330484258, 0.029865777325086773,
          0.04683826432227621, 0.01880807994439806, 0.05468860348477102,
          0.07181506647247264, 0.0597967463434774, 0.035892100831017446,
          0.07714865161046885]
    @test isapprox(w7.weights, wt)

    w8 = optimise!(portfolio, HRP(; rm = EVaR()))
    wt = [0.04298217683837017, 0.058906176713889875, 0.0574016331486885,
          0.03020450525284909, 0.0498349903258857, 0.060682817777209165,
          0.024220257509343413, 0.03449220187183893, 0.028790726943045256,
          0.06651915441075427, 0.1102093216711328, 0.026320946186565213, 0.0350510342557288,
          0.04934072008028962, 0.021946092606624574, 0.05048203603428413,
          0.09622970122852144, 0.06481150812979522, 0.03384174474956397,
          0.057732254265619835]
    @test isapprox(w8.weights, wt, rtol = 5.0e-7)

    w9 = optimise!(portfolio, HRP(; rm = RLVaR()))
    wt = [0.04746219128450875, 0.05772052034931984, 0.057894023300070915,
          0.030899831490444826, 0.047780697436069414, 0.056496204023026735,
          0.02359615954618457, 0.032640369565769795, 0.028784216596900798,
          0.06836029362987105, 0.10450059856679983, 0.02457130647918195,
          0.03686881087305981, 0.048996934209302734, 0.0227965377657541,
          0.05134221596812637, 0.11116746180000622, 0.06399587050591148,
          0.031665735616057476, 0.052460020993633505]
    @test isapprox(w9.weights, wt, rtol = 5.0e-7)

    w10 = optimise!(portfolio, HRP(; rm = MDD()))
    wt = [0.06921730086136976, 0.04450738292959614, 0.08140658224086754,
          0.018672719501981218, 0.053316921973365496, 0.02979463558079051,
          0.021114260172381653, 0.029076952717003283, 0.026064071405076886,
          0.07316497416971823, 0.11123520904148268, 0.018576814647316645,
          0.013991436918012332, 0.05469682002741476, 0.012327710930761682,
          0.08330184420498327, 0.07332958900751246, 0.060105980885656114,
          0.049926342635393445, 0.07617245014931591]
    @test isapprox(w10.weights, wt)

    w11 = optimise!(portfolio, HRP(; rm = ADD()))
    wt = [0.05548273429645013, 0.03986073873648723, 0.09599318831154405,
          0.014621119350699974, 0.08212398553813219, 0.024721555415213262,
          0.017757328803291575, 0.024648542319161523, 0.04053046304894661,
          0.06980177121920025, 0.07760681699761883, 0.01027851528431043,
          0.00777588366272761, 0.021591614519396285, 0.0034064806929941845,
          0.06991915064946408, 0.12960468534402678, 0.062161540752826025,
          0.08184195485513887, 0.07027193020237012]
    @test isapprox(w11.weights, wt)

    w12 = optimise!(portfolio, HRP(; rm = CDaR()))
    wt = [0.06282520890507177, 0.03706824571002148, 0.08139181669684024,
          0.015787409071561908, 0.05742532060680956, 0.022473205432788673,
          0.018010510386343107, 0.028690055814249292, 0.031947471924376705,
          0.07945270069786842, 0.10413378630449394, 0.015162575734861551,
          0.011642750066910313, 0.04691738037936903, 0.008558622422573802,
          0.08506702725550051, 0.07867872568683776, 0.06853164241553955,
          0.06271444623431467, 0.08352109825366769]
    @test isapprox(w12.weights, wt)

    w13 = optimise!(portfolio, HRP(; rm = UCI()))
    wt = [0.056222127504695026, 0.03676156065449049, 0.09101083070457834,
          0.014331694009042086, 0.06981598298437515, 0.02449761185808867,
          0.017540488246254482, 0.025783651111609372, 0.03897771757340407,
          0.07792753098849892, 0.09069642716675666, 0.011419250355786286,
          0.009579825814905836, 0.03155851948684042, 0.005020862393829734,
          0.07891456352044066, 0.10481408027151863, 0.06296146942913426, 0.0767246147903293,
          0.0754411911354216]
    @test isapprox(w13.weights, wt)

    w14 = optimise!(portfolio, HRP(; rm = EDaR()))
    wt = [0.06368313885096026, 0.03964177645523946, 0.08253885356726247,
          0.01658630133298382, 0.055878822739450135, 0.025248628049312456,
          0.019039909877211127, 0.028999411295580237, 0.030719786342128182,
          0.07837842859043447, 0.10479226890649017, 0.01598505374262056,
          0.012462054946453145, 0.0503667610779172, 0.009778781140263914,
          0.08402144368015278, 0.07578137459372361, 0.0651773824278426, 0.05986313372528741,
          0.0810566886586861]
    @test isapprox(w14.weights, wt, rtol = 5.0e-7)

    w15 = optimise!(portfolio, HRP(; rm = RLDaR()))
    wt = [0.06626057100148959, 0.042384460290778865, 0.08229559342069401,
          0.017571400016583083, 0.054533052228267126, 0.02751643858660513,
          0.02021038180000024, 0.02882973184000429, 0.028405177937391907,
          0.0760417757563507, 0.10819526671824652, 0.01714761122061998,
          0.013186013461541538, 0.05250570179214284, 0.01096704678455384,
          0.08422872826614602, 0.07476238567443566, 0.0621153409498924,
          0.054747390549717716, 0.07809593170453868]
    @test isapprox(w15.weights, wt, rtol = 5.0e-8)

    w16 = optimise!(portfolio, HRP(; rm = Kurt()))
    wt = [0.029481521953684118, 0.05982735944410474, 0.03557911826015237,
          0.028474368121041888, 0.025828635808911045, 0.05929340372085539,
          0.00400811258716055, 0.04856579375773367, 0.028712479333209244,
          0.0657391879397804, 0.16216057544981644, 0.011489226379288793,
          0.009094526296241098, 0.08053893478316498, 0.012724404070883372,
          0.021335119185650623, 0.0855020913192837, 0.11267092582579596,
          0.03851809818436267, 0.08045611757887891]
    @test isapprox(w16.weights, wt)

    w17 = optimise!(portfolio, HRP(; rm = SKurt()))
    wt = [0.041695764288341326, 0.04968730030116381, 0.04519046227953908,
          0.02228610752801017, 0.036916599549008534, 0.06695518729855642,
          0.008601057788011419, 0.03832875671733724, 0.023603809923922775,
          0.05953659154802947, 0.15949334007294313, 0.010345061127586078,
          0.01551106764201746, 0.07844578132980796, 0.01457102407508962,
          0.034738374409788796, 0.09106192787421133, 0.1097918256490279, 0.0352173258407637,
          0.058022634756843806]
    @test isapprox(w17.weights, wt)

    w18 = optimise!(portfolio, HRP(; rm = Skew()))
    wt = [4.18164303203965e-6, 0.11905148244884775, 0.13754035964680367,
          4.18164303203965e-6, 0.13754035964680367, 1.658199358060021e-6,
          0.028721218707620826, 1.1383557170375464e-10, 8.355416012626526e-16,
          1.8128635217880256e-6, 0.08532736996421339, 0.007291462719384035,
          0.10926054871992093, 0.00021830908182616843, 0.2370505968817962,
          0.10926054871992093, 0.028721218707620826, 3.684313367457248e-6,
          4.824826616435831e-11, 1.0059308456231142e-6]
    @test isapprox(w18.weights, wt, rtol = 5.0e-5)

    w19 = optimise!(portfolio, HRP(; rm = SSkew()))
    wt = [0.03073561997154973, 0.06271788007944147, 0.059839304178611136,
          0.019020703463231065, 0.054114343500047124, 0.07478195087196789,
          0.015025906923294915, 0.037423537539572865, 0.02195663364580543,
          0.07120487867873436, 0.14238956696393482, 0.022992175030255378,
          0.02427916555268004, 0.040541588877064376, 0.010704183865328891,
          0.053000134123666075, 0.07802885174200914, 0.06680453879901624,
          0.030056876315924044, 0.08438215987786517]
    @test isapprox(w19.weights, wt)

    clust_alg = HAC()
    cluster_assets!(portfolio; clust_alg = clust_alg, clust_opt = clust_opt)
    w20 = optimise!(portfolio, HRP(; rm = BDVariance()))
    wt = [0.034616203287809656, 0.05339682894540566, 0.0277781553250953,
          0.04211296639608739, 0.05429852976387857, 0.07556907274344477,
          0.012918417022839884, 0.0639503232740957, 0.025141332254925024,
          0.05293567401349993, 0.12067634196508455, 0.0237681969964296,
          0.008823211105214881, 0.08942720087787832, 0.016861747398793574,
          0.03288067246055758, 0.07170919770494971, 0.0812323656620027, 0.03793954600227262,
          0.07396401679973474]
    @test isapprox(w20.weights, wt)

    w21 = optimise!(portfolio, HRP(; rm = Variance()))
    wt = [0.03360421525593201, 0.053460257098811775, 0.027429766590708997,
          0.04363053745921338, 0.05279180956461212, 0.07434468966041922,
          0.012386194792256387, 0.06206960160806503, 0.025502890164538234,
          0.0542097204834031, 0.12168116639250848, 0.023275086688903004,
          0.009124639465879256, 0.08924750757276853, 0.017850423121104797,
          0.032541204698588386, 0.07175228284814082, 0.08318399209117079,
          0.03809545426566615, 0.07381856017730955]
    @test isapprox(w21.weights, wt)

    w22 = optimise!(portfolio, HRP(; rm = SVariance()))
    wt = [0.03352132634395958, 0.05070696394146035, 0.025245292010558144,
          0.041199581487040476, 0.05550265797278166, 0.07527829480450478,
          0.01465919892361806, 0.059327687253569454, 0.024511978344932878,
          0.05322045263668202, 0.12280140260349257, 0.022293827005819677,
          0.009725576750592858, 0.09323358575910765, 0.01704823566709182,
          0.037255645640231386, 0.06932731082207985, 0.090058219354306, 0.03762181352060564,
          0.06746094915756515]
    @test isapprox(w22.weights, wt)

    w23 = optimise!(portfolio, HRP(; rm = Equal()))
    wt = [0.05, 0.05000000000000001, 0.05, 0.04999999999999999, 0.04999999999999999,
          0.05000000000000001, 0.05000000000000001, 0.05, 0.05, 0.05000000000000001,
          0.04999999999999999, 0.04999999999999999, 0.05, 0.04999999999999999, 0.05,
          0.04999999999999999, 0.04999999999999999, 0.05, 0.05, 0.04999999999999999]
    @test isapprox(w23.weights, wt)

    w24 = optimise!(portfolio, HRP(; rm = VaR()))
    wt = [0.03403717939776512, 0.05917176810341441, 0.029281627698577888,
          0.04641795773733506, 0.06002080678226616, 0.07336909341225178,
          0.032392046758139795, 0.05383361136188931, 0.02967264551919885,
          0.05284095110693102, 0.09521120378350553, 0.04261320776949539,
          0.01642971834566978, 0.07971670573840162, 0.021147864266696195,
          0.050803448662520866, 0.06563107532995796, 0.05200824347813839,
          0.034112945793535646, 0.07128789895430926]
    @test isapprox(w24.weights, wt)

    w25 = optimise!(portfolio, HRP(; rm = DaR()))
    wt = [0.057660532702867646, 0.04125282312548106, 0.05922785740065619,
          0.03223509018499126, 0.10384639449297382, 0.02480551517770064,
          0.030513023620218654, 0.037564939241009405, 0.02260728843785307,
          0.05164502511939685, 0.08286147806113833, 0.01696931547695469,
          0.00418787314771699, 0.06308172456210165, 0.007685315296496237,
          0.0694006227527249, 0.09119842783020647, 0.09248808273195344,
          0.044542968010494795, 0.06622570262706394]
    @test isapprox(w25.weights, wt)

    w26 = optimise!(portfolio, HRP(; rm = DaR_r()))
    wt = [0.05820921825149978, 0.04281381004716568, 0.061012991673039425,
          0.037261284520695284, 0.10225280827507012, 0.03171774621579708,
          0.029035913030571646, 0.03777811595747374, 0.021478656618637317,
          0.05210098246314737, 0.08336111731195082, 0.0224554330786605,
          0.008018398540074134, 0.060923985355596684, 0.009489132849269694,
          0.06345531419794387, 0.0883323204747975, 0.08122137323769983,
          0.039962166984027804, 0.06911923091688174]
    @test isapprox(w26.weights, wt)

    w27 = optimise!(portfolio, HRP(; rm = MDD_r()))
    wt = [0.054750291482807016, 0.05626561538767235, 0.044927537922610235,
          0.037029800949262545, 0.06286000734501392, 0.038964408950160866,
          0.033959752237812064, 0.04828571977339982, 0.01937785058895953,
          0.05279062837229332, 0.10144994531515807, 0.02678668518638289,
          0.010335110450233378, 0.07434568383344507, 0.01192345099055306,
          0.05711412672965212, 0.06785463405921893, 0.09152771438398988,
          0.032677346602089874, 0.07677368943928506]
    @test isapprox(w27.weights, wt)

    w28 = optimise!(portfolio, HRP(; rm = ADD_r()))
    wt = [0.05216718533133331, 0.04708278318245931, 0.07158823871210457,
          0.029511984859220904, 0.12867669205129983, 0.03859596234440916,
          0.022402839001206605, 0.026020360199894607, 0.030904469636691148,
          0.054239105004183275, 0.07390114503735276, 0.019880342325388017,
          0.00476824739544596, 0.03991945523395664, 0.0054480039590591904,
          0.059645486519252305, 0.11611402694084015, 0.06374794181935534,
          0.059893035563107475, 0.055492694883439435]
    @test isapprox(w28.weights, wt)

    w29 = optimise!(portfolio, HRP(; rm = CDaR_r()))
    wt = [0.055226953695215866, 0.047934679854074826, 0.05395157314076186,
          0.036982877818185954, 0.08463665680704126, 0.033820937721103415,
          0.03026563101495272, 0.04118023926554901, 0.02122427296619385,
          0.05288666614109493, 0.09090289747478623, 0.02380032651263032,
          0.008673469487824291, 0.06662055705080387, 0.010138402407857017,
          0.060505269563432176, 0.08319151689594376, 0.08557053458440436,
          0.038056791259321536, 0.07442974633882277]
    @test isapprox(w29.weights, wt)

    w30 = optimise!(portfolio, HRP(; rm = UCI_r()))
    wt = [0.05276898612268049, 0.04614128625526376, 0.06552577565455027, 0.0320518739448942,
          0.1111735166748811, 0.039154188455666976, 0.025443440854672504,
          0.03239581981059534, 0.027120969094009423, 0.05445935205465669,
          0.08028963951603105, 0.02075298500986429, 0.0058131044419531056,
          0.04948420047533672, 0.006850658262213453, 0.06288836730781266,
          0.1025592250558978, 0.07454814435900761, 0.050977607512044276,
          0.05960085913796832]
    @test isapprox(w30.weights, wt, rtol = 1.0e-7)

    w31 = optimise!(portfolio, HRP(; rm = EDaR_r()))
    wt = [0.055560528232119676, 0.05252278252766485, 0.04995930852013699,
          0.036469686665161705, 0.07360443788124264, 0.035484250062529706,
          0.03187063749286862, 0.04362148704778408, 0.021541875055852516,
          0.05435689568806405, 0.09401418942704692, 0.024881512156616555,
          0.009270444002026255, 0.06971378686582017, 0.01078203039784278,
          0.05939090630801511, 0.07729538472655453, 0.08736656785459875,
          0.03800312958595158, 0.07429015950210269]
    @test isapprox(w31.weights, wt, rtol = 5.0e-7)

    w32 = optimise!(portfolio, HRP(; rm = RLDaR_r()))
    wt = [0.055263520718878155, 0.05470886033764548, 0.04729258737064928,
          0.03659066826504964, 0.06763240307582324, 0.03707186354483175,
          0.03301695115922358, 0.04590027827321353, 0.020741988491439888,
          0.05389295853521278, 0.09778300148886963, 0.025642106200246102,
          0.00971015663966544, 0.07161140382368784, 0.011262111599036555,
          0.05878170332530692, 0.07257021270894684, 0.08962515902013113,
          0.035691384041053154, 0.07521068138108908]
    @test isapprox(w32.weights, wt, rtol = 5.0e-8)
end

@testset "HERC" begin
    portfolio = Portfolio(; prices = prices,
                          solvers = PortOptSolver(; name = :Clarabel,
                                                  check_sol = (; allow_local = true,
                                                               allow_almost = true),
                                                  solver = Clarabel.Optimizer,
                                                  params = ["verbose" => false,
                                                            "max_step_fraction" => 0.75]))

    asset_statistics!(portfolio)
    clust_alg = DBHT()
    clust_opt = ClustOpt()
    cluster_assets!(portfolio; clust_alg = clust_alg, clust_opt = clust_opt)

    w1 = optimise!(portfolio, HERC(; rm = SD()))
    wt = [0.07698450455880468, 0.07670235654605057, 0.06955326092716153,
          0.055733645924582964, 0.061306348146154536, 0.026630781912285295,
          0.0274779863650835, 0.08723325075122691, 0.021443967931440197,
          0.050919570266336714, 0.034721621146346957, 0.013406113465936382,
          0.018014864090601306, 0.029736261019180265, 0.01130547202588631,
          0.03532911363416089, 0.08752722816710831, 0.10098629923304404,
          0.026208793387783046, 0.08877856050082561]
    @test isapprox(w1.weights, wt)

    w2 = optimise!(portfolio, HERC(; rm = MAD()))
    wt = [0.0797145490968226, 0.07455683155558127, 0.070334401927198, 0.05349383521987859,
          0.06513229159697068, 0.026966237124386894, 0.029595506806349627,
          0.09315224544455303, 0.020640678885583085, 0.047472627309343436,
          0.032958383881734096, 0.014663096865841139, 0.017152340172727078,
          0.029195401656427377, 0.010527045845271121, 0.03720496223361964,
          0.08565021870448736, 0.09947562485956811, 0.025808052654604105,
          0.08630566815905283]
    @test isapprox(w2.weights, wt)

    w3 = optimise!(portfolio, HERC(; rm = SSD()))
    wt = [0.07837975159477181, 0.07443043390995964, 0.06801950779661087,
          0.055548455549923444, 0.0644736826243098, 0.02739674778042435,
          0.029790140405552036, 0.08281583015461562, 0.021850035315318022,
          0.05161041880598439, 0.035414428250310895, 0.013356805631372573,
          0.02005310749097011, 0.03085778614009161, 0.012088382453633814,
          0.03792292849371053, 0.08401306865550462, 0.10203433260227453,
          0.027069661454433437, 0.08287449489022782]
    @test isapprox(w3.weights, wt)

    w4 = optimise!(portfolio, HERC(; rm = FLPM(; target = rf)))
    wt = [0.08074577526070313, 0.0758057372636474, 0.0723712035349317, 0.05388227729039704,
          0.0719726576355769, 0.023357700765219442, 0.030259361503984363,
          0.09069744656455722, 0.020432554010122735, 0.045981867419973635,
          0.03127471464903142, 0.013225268293206591, 0.01541923284313144,
          0.026677624195275505, 0.00924248201367476, 0.03763591530781149,
          0.08961751171161211, 0.09815629750095696, 0.02604005979180501,
          0.08720431244438119]
    @test isapprox(w4.weights, wt)

    w5 = optimise!(portfolio, HERC(; rm = SLPM(; target = rf)))
    wt = [0.07895203150464367, 0.07513349407323168, 0.06901069326760172,
          0.05594112932547382, 0.06763676613416904, 0.025518328924508955,
          0.030140432803640472, 0.08170116913851065, 0.021723048018546902,
          0.05096388176098718, 0.0345040083464426, 0.012730715665077913,
          0.018968393009821956, 0.029479897263328572, 0.01127249390583484,
          0.03813269633505064, 0.0862014575477909, 0.1015604380650207, 0.02714041935142286,
          0.08328850555889497]
    @test isapprox(w5.weights, wt)

    w6 = optimise!(portfolio, HERC(; rm = WR()))
    wt = [0.09799469653109426, 0.07931044806875052, 0.07697883394077526,
          0.05938016484003881, 0.062482774302560766, 0.024174945904638732,
          0.021508124985828447, 0.05117445754264079, 0.023413964946189022,
          0.07738784752841718, 0.02842727729869877, 0.006737256345835118,
          0.024299987114092203, 0.030479447210160812, 0.014669862619463056,
          0.033478912079376394, 0.10842038272038522, 0.0983414761742503,
          0.02496318167060388, 0.05637595817620051]
    @test isapprox(w6.weights, wt)

    w7 = optimise!(portfolio, HERC(; rm = CVaR()))
    wt = [0.07701588327689891, 0.07520448896018464, 0.06541770521947644,
          0.05757000169964883, 0.06337709857771338, 0.025926148616162506,
          0.030149122854382444, 0.08115573202397372, 0.021487151979485357,
          0.053917172483500996, 0.03629492645395641, 0.013834276128551553,
          0.02188007667915268, 0.030097549937467067, 0.012085783569165435,
          0.04006561840657164, 0.08240686258336018, 0.10281015669469054,
          0.02647712574837549, 0.08282711810728184]
    @test isapprox(w7.weights, wt)

    w8 = optimise!(portfolio, HERC(; rm = EVaR()))
    wt = [0.08699441730392812, 0.07712958717175648, 0.07366521524310121,
          0.06113285849402033, 0.06395471849180502, 0.025641689880494402,
          0.024561083982710123, 0.05538393000042802, 0.023397235003817983,
          0.06889218377773806, 0.032808220114320844, 0.0078354841769452,
          0.024451109097834427, 0.03131096206947751, 0.01392669730115599,
          0.03521555916294673, 0.09758384164962133, 0.10406746553380812,
          0.027502023704060772, 0.06454571784002953]
    @test isapprox(w8.weights, wt, rtol = 5.0e-7)

    w9 = optimise!(portfolio, HERC(; rm = RLVaR()))
    wt = [0.09291265811652286, 0.07830058121386262, 0.07669383422811399,
          0.06048994792338911, 0.0632964281903878, 0.024835265241643308,
          0.022359240252220428, 0.05137247534829683, 0.023653937183085144,
          0.07457499324960506, 0.02985537439802565, 0.007019917248757973,
          0.024398011725401916, 0.03105422675169787, 0.01444843160813441,
          0.033975817433097084, 0.10534002288596125, 0.10072270393057964,
          0.026021876211112602, 0.058674256860104405]
    @test isapprox(w9.weights, wt, rtol = 5.0e-7)

    w10 = optimise!(portfolio, HERC(; rm = MDD()))
    wt = [0.11942828911415344, 0.06790696087036098, 0.09708343396881959, 0.0322181147701282,
          0.06358441456867103, 0.010918315222441926, 0.029080597027890735,
          0.05067903550079908, 0.019797098917560686, 0.032160771893579174,
          0.04334920387136981, 0.0072395254377313496, 0.004408762771082226,
          0.030601955352742596, 0.0068971479386845886, 0.02624877427854606,
          0.100996587649217, 0.10476039799498128, 0.03792180923631091, 0.11471880361492931]
    @test isapprox(w10.weights, wt)

    w11 = optimise!(portfolio, HERC(; rm = ADD()))
    wt = [0.10441410900507704, 0.06337257015393993, 0.1388818998669592,
          0.027515787911662204, 0.1188160882745767, 0.009101280957417223,
          0.02353511836320865, 0.03744216548824482, 0.015577834189637181,
          0.01767347121199722, 0.025141762273107857, 0.0033298619605358548,
          0.0018130367073894856, 0.014126651605875161, 0.002228743288689212,
          0.016302454122944926, 0.17177480035356854, 0.09442597723364114,
          0.03145585583242577, 0.08307053119910199]
    @test isapprox(w11.weights, wt)

    w12 = optimise!(portfolio, HERC(; rm = CDaR()))
    wt = [0.11749818113343935, 0.05870055826126679, 0.11385134924830828,
          0.029526234501200906, 0.08032687433988389, 0.008852903280265263,
          0.026595918536857853, 0.04345063385165379, 0.02084656630749761,
          0.029413320573451408, 0.04005512791398602, 0.005832294513804399,
          0.0033062821268677552, 0.02971553377155059, 0.005420678468821243,
          0.02415714416132541, 0.11618399112873753, 0.10379008396251159,
          0.040922826850160604, 0.10155349706840978]
    @test isapprox(w12.weights, wt)

    w13 = optimise!(portfolio, HERC(; rm = UCI()))
    wt = [0.10987855273964697, 0.060729473692400754, 0.13188113586803757,
          0.028009359764792043, 0.10116830124988405, 0.00894518902343889,
          0.02515249217180455, 0.040355519105512605, 0.018050336913773852,
          0.021728169704026638, 0.031807104420907735, 0.0040047144063036725,
          0.0023194434166183132, 0.020203823344076888, 0.0032143655180704735,
          0.019106596337899368, 0.15029999715584982, 0.09854472399816692,
          0.03553068862836474, 0.08907001254042422]
    @test isapprox(w13.weights, wt)

    w14 = optimise!(portfolio, HERC(; rm = EDaR()))
    wt = [0.1177308542531208, 0.06254095321442733, 0.10519293286493245,
          0.030663052419603157, 0.07121563960435068, 0.009654306131028573,
          0.02799873014876406, 0.046417936073046444, 0.020983248093459222,
          0.03096550906654087, 0.04162305259732029, 0.006349196745551905,
          0.0037357551287625196, 0.030460451517300507, 0.00591394170374592,
          0.02518714132643887, 0.11143867125608797, 0.10432624097459507,
          0.040889704525289054, 0.10671268235563422]
    @test isapprox(w14.weights, wt, rtol = 1.0e-7)

    w15 = optimise!(portfolio, HERC(; rm = RLDaR()))
    wt = [0.11833256405204345, 0.06528196976061985, 0.1004384592603245,
          0.031380183757123425, 0.06655539558564487, 0.010311178543611854,
          0.02870698592124496, 0.04850410121854156, 0.020662238837009742,
          0.03196981907376369, 0.04288859204254551, 0.006797311314496403,
          0.004081222192046016, 0.030487186099345632, 0.006367963571314287,
          0.026069756110107468, 0.10619308226658083, 0.1045049159464766,
          0.03982385381687521, 0.11064322063028406]
    @test isapprox(w15.weights, wt)

    w16 = optimise!(portfolio, HERC(; rm = Kurt()))
    wt = [0.053466892258046884, 0.10328667521887797, 0.054404919323835864,
          0.05164034525881938, 0.03949521281426285, 0.02166132115612342,
          0.005325106752082301, 0.06987092753743517, 0.019722628041309125,
          0.058905384535264414, 0.04756653865024671, 0.00337013314805977,
          0.005397558580038606, 0.03531833344651656, 0.005579968832386527,
          0.012662292885364586, 0.11359655047116007, 0.16209808354457939,
          0.02645811650511682, 0.11017301104047358]
    @test isapprox(w16.weights, wt)

    w17 = optimise!(portfolio, HERC(; rm = SKurt()))
    wt = [0.09195751022271098, 0.07943849543837476, 0.06646961117264624, 0.0491506750340217,
          0.05429977685689461, 0.026897931456943396, 0.010291696410768226,
          0.05363160327577018, 0.01926078898577857, 0.06072467774497688,
          0.047686505861964906, 0.0030930433701959084, 0.009416070036669964,
          0.037424766409940795, 0.006951516870884198, 0.021088101344911066,
          0.10896121609215866, 0.1536264710999023, 0.028737457378644478,
          0.07089208493584212]
    @test isapprox(w17.weights, wt)

    w18 = optimise!(portfolio, HERC(; rm = Skew()))
    wt = [0.05013622008238557, 0.05843253495835018, 0.0326340699568634, 0.03810785845352964,
          0.04151978371454936, 0.020776349162705425, 0.014419467673840766,
          0.06450029254562262, 0.02337036891691671, 0.0768704564650501, 0.03525278374131375,
          0.007885328106569226, 0.23227643708915915, 0.029736303978302396,
          0.019940256048870408, 0.05591386120058815, 0.04641584792557699,
          0.06294077176658942, 0.036661874867119304, 0.0522091333460975]
    @test isapprox(w18.weights, wt)

    w19 = optimise!(portfolio, HERC(; rm = SSkew()))
    wt = [0.06664919970362215, 0.065688537645969, 0.05827905623198537, 0.04923706156165528,
          0.055949261956495855, 0.030046317963353407, 0.027319842253653435,
          0.08126769722698507, 0.02377683386943261, 0.08094977263933717,
          0.04476608242725907, 0.012406273321936936, 0.028493003361406897,
          0.03436511159723589, 0.014308519726637069, 0.05423891622864567,
          0.07452547794485952, 0.09337434382384531, 0.028961215530580075,
          0.07539747498510434]
    @test isapprox(w19.weights, wt)

    clust_alg = HAC()
    cluster_assets!(portfolio; clust_alg = clust_alg, clust_opt = clust_opt)

    w20 = optimise!(portfolio, HERC(; rm = BDVariance()))
    wt = [0.11814156791628717, 0.1127352219498846, 0.0948040083033778, 0.05910201284574831,
          0.0762034280232549, 0.021900080015085997, 0.005240769029403859,
          0.02492257943111223, 0.013821146124791104, 0.015571892677105771,
          0.036219350442415844, 0.012370220354411125, 0.004197597088141013,
          0.02684034894442968, 0.008021889189561834, 0.01711283206712795,
          0.1478163099569289, 0.03165769900663641, 0.020856810764390717, 0.1524642358699047]
    @test isapprox(w20.weights, wt)

    w21 = optimise!(portfolio, HERC(; rm = Variance()))
    wt = [0.11540105921544155, 0.1145567201312018, 0.09419723372475931,
          0.060483660234210215, 0.07318364748176763, 0.02168713926476103,
          0.005201942574220557, 0.023749598552788367, 0.014061923482203057,
          0.015961970718579934, 0.03686671320264766, 0.012132318435028086,
          0.00441044709602619, 0.02704002898133752, 0.00862810493627861,
          0.016962353908264397, 0.14917275390007956, 0.03182856610967728,
          0.021005280556336624, 0.15346853749439043]
    @test isapprox(w21.weights, wt)

    w22 = optimise!(portfolio, HERC(; rm = SVariance()))
    wt = [0.12119760179878843, 0.10929174557400614, 0.09127529194384341,
          0.06087377541946739, 0.08200705479693754, 0.022938849314970432,
          0.007017216937835133, 0.02434527249121335, 0.014590759260977745,
          0.016875525957101834, 0.03832958126981687, 0.012145368603645653,
          0.005675155901052187, 0.029100679851099043, 0.00994813960444317,
          0.020296360456609625, 0.13924511135169032, 0.03695562715740502,
          0.022394390869475853, 0.13549649143962084]
    @test isapprox(w22.weights, wt)

    w23 = optimise!(portfolio, HERC(; rm = Equal()))
    wt = [0.09923664122137407, 0.09923664122137407, 0.09923664122137407,
          0.09923664122137407, 0.09923664122137407, 0.023487962419260135,
          0.02348796241926013, 0.023487962419260135, 0.023487962419260135,
          0.023487962419260135, 0.023487962419260135, 0.02348796241926013,
          0.02348796241926013, 0.023487962419260135, 0.02348796241926013,
          0.02348796241926013, 0.09923664122137407, 0.023487962419260135,
          0.023487962419260135, 0.09923664122137407]
    @test isapprox(w23.weights, wt)

    w24 = optimise!(portfolio, HERC(; rm = VaR()))
    wt = [0.10414300257102177, 0.09871202573006761, 0.08959251861207183,
          0.07555514734087271, 0.09769669156091847, 0.022943300926461043,
          0.016563538745652133, 0.028652238501006684, 0.019401391461582398,
          0.017501979293453335, 0.030610780860929044, 0.02395924092174919,
          0.013446069288029138, 0.025629238086958404, 0.01730739640456328,
          0.028564197108681265, 0.11491557263229325, 0.027680710219061735,
          0.02230467164850204, 0.12482028808612478]
    @test isapprox(w24.weights, wt)

    w25 = optimise!(portfolio, HERC(; rm = DaR()))
    wt = [0.14733227960307277, 0.06614386458233755, 0.15133705565662284,
          0.035584333675786964, 0.11463609164605132, 0.005518410268992302,
          0.013171538035678308, 0.010620417669043803, 0.014023858241571917,
          0.017001289131758357, 0.026781482271749004, 0.007217550381915265,
          0.0035803904597587304, 0.020388510168558176, 0.0065705021611778035,
          0.029518131826492447, 0.16035186159841677, 0.026148373666210922,
          0.027631100950263884, 0.11644295800454063]
    @test isapprox(w25.weights, wt)

    w26 = optimise!(portfolio, HERC(; rm = DaR_r()))
    wt = [0.14774355938966005, 0.0684328493553202, 0.15485994881153173, 0.0411979500128519,
          0.11305584705890495, 0.007216566584313541, 0.011970976561070937,
          0.011775533919110818, 0.013661517786887336, 0.016951592576994145,
          0.02646193157189205, 0.008427695298573155, 0.006529147510212775,
          0.019339548023735872, 0.007726723460706738, 0.02381526338245954,
          0.15152867379279225, 0.025316906660827858, 0.025417970255231653,
          0.11856979798692265]
    @test isapprox(w26.weights, wt)

    w27 = optimise!(portfolio, HERC(; rm = MDD_r()))
    wt = [0.15079446061615842, 0.08603152475897777, 0.12374041606663375,
          0.050529282163624914, 0.0857760767414279, 0.009923344774006223,
          0.0134887577916876, 0.016005392064562644, 0.0148967949294955,
          0.020046408045779833, 0.029228399800324625, 0.0099793139298191,
          0.008133340149925354, 0.02141948291605225, 0.00938330394572054,
          0.02127772796435491, 0.12849798956868397, 0.030338927541390587,
          0.025120832103469992, 0.14538822412790406]
    @test isapprox(w27.weights, wt)

    w28 = optimise!(portfolio, HERC(; rm = ADD_r()))
    wt = [0.12371584689868499, 0.07328365011343618, 0.16977338386193674,
          0.03354181126220188, 0.14624734118080304, 0.009376911249269603,
          0.008150033847862812, 0.008167198562468324, 0.012903517155053858,
          0.011746529657682171, 0.021870250127331976, 0.005917245694323535,
          0.0032768170628257485, 0.011813734015517456, 0.003743956814916424,
          0.01775306443497017, 0.19872759251615896, 0.020009027345803686,
          0.025007088649056427, 0.09497499954969608]
    @test isapprox(w28.weights, wt)

    w29 = optimise!(portfolio, HERC(; rm = CDaR_r()))
    wt = [0.1482304036785224, 0.07559110451590667, 0.14480725317353932, 0.04480939008007434,
          0.10254791389098332, 0.00791620073213515, 0.011996465335688086,
          0.012913340180870978, 0.014391268171680365, 0.01805225079803855,
          0.0270384713414632, 0.008561039338370812, 0.006787624092846017,
          0.01981584825797337, 0.0079340412211216, 0.021763902803450537, 0.1447239549586514,
          0.026833292915610436, 0.025804676072481453, 0.12948155844059195]
    @test isapprox(w29.weights, wt)

    w30 = optimise!(portfolio, HERC(; rm = UCI_r()))
    wt = [0.13177358314927293, 0.07247391549233687, 0.16362956503582105,
          0.03659195084440874, 0.126920998889531, 0.00897565819404876, 0.010308036952094911,
          0.010070348969290802, 0.013930178878301605, 0.014394572348790095,
          0.025070306799710288, 0.006958517882519902, 0.004449124903473799,
          0.01545135954194139, 0.005243228396110122, 0.021086596858518908,
          0.17918434540380382, 0.02317354007700532, 0.02618369531594223,
          0.10413047606707744]
    @test isapprox(w30.weights, wt)

    w31 = optimise!(portfolio, HERC(; rm = EDaR_r()))
    wt = [0.1488274121958717, 0.07965043465691322, 0.1338236845244531, 0.04685245137285603,
          0.09455930834611771, 0.008714107998123267, 0.012663731601521146,
          0.014222803628373831, 0.015035762837750218, 0.01910358970690992,
          0.028184597388505987, 0.009107245853358044, 0.007272820589254927,
          0.020899558111554476, 0.008458685760279014, 0.021738533486077228,
          0.14067164133836854, 0.028485905052245995, 0.026525362442459087,
          0.13520236310900652]
    @test isapprox(w31.weights, wt, rtol = 1.0e-7)

    w32 = optimise!(portfolio, HERC(; rm = RLDaR_r()))
    wt = [0.1496829904780472, 0.08274800482010956, 0.1280934678460413, 0.0484163894788309,
          0.08949048983985806, 0.009338809042079375, 0.013100310497212761,
          0.015211346816708672, 0.01528834777372118, 0.01979313339434657,
          0.02904499687337739, 0.009477735173530136, 0.007610137617200528,
          0.02127110999508211, 0.008826450722690645, 0.021726663668314845,
          0.13497973743198086, 0.029701766327373105, 0.026307135016014324,
          0.1398909771874805]
    @test isapprox(w32.weights, wt)
end

@testset "HERC and NCO mixed parameters" begin
    portfolio = Portfolio(; prices = prices,
                          solvers = PortOptSolver(; name = :Clarabel,
                                                  check_sol = (; allow_local = true,
                                                               allow_almost = true),
                                                  solver = Clarabel.Optimizer,
                                                  params = ["verbose" => false,
                                                            "max_step_fraction" => 0.75]))

    asset_statistics!(portfolio)
    clust_alg = DBHT(; root_type = EqualDBHT())
    clust_opt = ClustOpt()
    cluster_assets!(portfolio; clust_alg = clust_alg, clust_opt = clust_opt)

    w1 = optimise!(portfolio,
                   NCO(;
                       internal = NCOArgs(;
                                          type = Trad(; rm = Variance(),
                                                      obj = Sharpe(; rf = rf),
                                                      kelly = EKelly())),
                       external = NCOArgs(;
                                          type = Trad(; rm = CDaR(),
                                                      obj = Utility(; l = 10 * l),
                                                      kelly = NoKelly()))))
    wt = [8.746980512426739e-9, 0.02131840205146324, 0.010059177511047296,
          0.0006390401133837214, 0.23847424877396836, 2.7855378749498208e-9,
          0.03433829339292149, 0.00922798787029745, 2.192467451561324e-8,
          1.2946199986589789e-8, 2.891103081143254e-8, 2.500256934208249e-9,
          1.5087789156517654e-9, 5.291617985989037e-9, 1.6616784928558542e-9,
          0.11722712873716697, 0.4120482933288371, 0.016905831547980782,
          0.12415377521862672, 0.015607735177551017]
    @test isapprox(w1.weights, wt, rtol = 5.0e-5)

    w2 = optimise!(portfolio,
                   NCO(;
                       internal = NCOArgs(;
                                          type = Trad(; rm = Variance(),
                                                      obj = Sharpe(; rf = rf),
                                                      kelly = NoKelly())),
                       external = NCOArgs(;
                                          type = Trad(; rm = CDaR(),
                                                      obj = Utility(; l = 10 * l),
                                                      kelly = EKelly()))))
    wt = [3.1657474869830236e-10, 0.011350048898912344, 0.00590192500599762,
          4.145863348570475e-9, 0.24703174384661478, 1.1554967125924781e-10,
          0.035711668009318916, 2.0093201218328118e-9, 1.1359951214950826e-9,
          5.740702868245685e-10, 9.252352831708804e-10, 1.1350185465541757e-10,
          6.975182236561415e-11, 2.340817140646339e-10, 6.928487479990741e-11,
          0.12923169151529168, 0.4471580486746947, 2.5109043555437692e-9,
          0.12361485036122226, 1.1467814516690299e-8]
    @test isapprox(w2.weights, wt)

    w3 = optimise!(portfolio,
                   NCO(;
                       internal = NCOArgs(;
                                          type = Trad(; rm = Variance(),
                                                      obj = Utility(; l = 10 * l),
                                                      kelly = EKelly())),
                       external = NCOArgs(;
                                          type = Trad(; rm = CDaR(),
                                                      obj = Sharpe(; rf = rf),
                                                      kelly = NoKelly()))))
    wt = [4.435123223202006e-9, 0.02981292829609958, 0.010894696928370992,
          0.011896392392422558, 0.044384434210418426, 2.3521987025134175e-9,
          0.005106857526739849, 0.0694692584065797, 3.2889458276343186e-9,
          0.02819781371232407, 0.28810640413584065, 8.305498316970034e-10,
          4.0191361393894005e-10, 0.04082436761974192, 4.4329905506900117e-10,
          0.0491186969507823, 0.2017493720580537, 0.10295500017433815, 0.061402464206477235,
          0.05608130162978066]
    @test isapprox(w3.weights, wt, rtol = 5.0e-8)

    w4 = optimise!(portfolio,
                   NCO(;
                       internal = NCOArgs(;
                                          type = Trad(; rm = Variance(),
                                                      obj = Utility(; l = 10 * l),
                                                      kelly = NoKelly())),
                       external = NCOArgs(;
                                          type = Trad(; rm = CDaR(),
                                                      obj = Sharpe(; rf = rf),
                                                      kelly = EKelly()))))
    wt = [2.055480878765726e-8, 0.0298193787413796, 0.010923751816092748,
          0.011868581992717992, 0.04541627972831881, 9.98392319910959e-9,
          0.00526131981959445, 0.06930598880049342, 2.1332823691396522e-8,
          0.027639808973600202, 0.2870813214020614, 2.7962632861247776e-10,
          3.6419120035415975e-9, 0.038007009998689924, 3.3607657782551183e-9,
          0.049529606042490584, 0.2034402582222892, 0.10272050416693007,
          0.06299462695235125, 0.05599150418913066]
    @test isapprox(w4.weights, wt, rtol = 5.0e-5)
    @test !isapprox(w1.weights, w2.weights)
    @test !isapprox(w1.weights, w3.weights)
    @test !isapprox(w1.weights, w4.weights)
    @test !isapprox(w2.weights, w3.weights)
    @test !isapprox(w2.weights, w4.weights)
    @test !isapprox(w3.weights, w4.weights)

    clust_alg = DBHT()
    cluster_assets!(portfolio; clust_alg = clust_alg, clust_opt = clust_opt)
    w5 = optimise!(portfolio, HERC(; rm_o = SD(), rm = CDaR()))
    wt = [0.10871059727246735, 0.05431039601849186, 0.10533650868181384,
          0.027317993835046576, 0.07431929926304212, 0.00954218610227609,
          0.024606833580412473, 0.04020099981391352, 0.022469670005659467,
          0.05391899731269113, 0.04317380104646033, 0.006286394179643389,
          0.006060907562898212, 0.0320291710414021, 0.005842729905950518,
          0.044283643115509565, 0.10749469436263087, 0.09602771642660826,
          0.04410905860746655, 0.09395840186561562]
    @test isapprox(w5.weights, wt)

    w6 = optimise!(portfolio, HERC(; rm = SD(), rm_o = CDaR()))
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

@testset "NCO vector rm" begin
    portfolio = Portfolio(; prices = prices,
                          solvers = PortOptSolver(; name = :Clarabel,
                                                  check_sol = (; allow_local = true,
                                                               allow_almost = true),
                                                  solver = Clarabel.Optimizer,
                                                  params = ["verbose" => false,
                                                            "max_step_fraction" => 0.75]))

    asset_statistics!(portfolio)
    cluster_assets!(portfolio)

    w1 = optimise!(portfolio,
                   NCO(;
                       internal = NCOArgs(;
                                          type = Trad(;
                                                      rm = Variance(;
                                                                    sigma = portfolio.cov)))))
    w2 = optimise!(portfolio,
                   NCO(;
                       internal = NCOArgs(;
                                          type = Trad(;
                                                      rm = [[Variance(;
                                                                      sigma = portfolio.cov),
                                                             Variance()]]))))
    w3 = optimise!(portfolio,
                   NCO(;
                       internal = NCOArgs(;
                                          type = Trad(;
                                                      rm = [Variance(;
                                                                     settings = RMSettings(;
                                                                                           scale = 2))]))))
    @test isapprox(w1.weights, w2.weights, rtol = 5.0e-5)
    @test isapprox(w1.weights, w3.weights, rtol = 1.0e-5)
    @test isapprox(w2.weights, w3.weights, rtol = 1.0e-5)

    w4 = optimise!(portfolio,
                   NCO(; internal = NCOArgs(; type = Trad(; rm = [Variance(), CDaR()]))))
    w5 = optimise!(portfolio,
                   NCO(;
                       internal = NCOArgs(;
                                          type = Trad(;
                                                      rm = [[Variance(), Variance()],
                                                            [CDaR(), CDaR()]]))))
    w6 = optimise!(portfolio,
                   NCO(;
                       internal = NCOArgs(;
                                          type = Trad(;
                                                      rm = [Variance(;
                                                                     settings = RMSettings(;
                                                                                           scale = 2)),
                                                            CDaR(;
                                                                 settings = RMSettings(;
                                                                                       scale = 2))]))))
    @test isapprox(w4.weights, w5.weights, rtol = 5.0e-5)
    @test isapprox(w4.weights, w6.weights, rtol = 5.0e-9)
    @test isapprox(w5.weights, w6.weights, rtol = 5.0e-5)
end

@testset "Shorting with NCO" begin
    portfolio = Portfolio(; prices = prices,
                          solvers = PortOptSolver(; name = :Clarabel,
                                                  check_sol = (; allow_local = true,
                                                               allow_almost = true),
                                                  solver = Clarabel.Optimizer,
                                                  params = ["verbose" => false,
                                                            "max_step_fraction" => 0.75]))

    asset_statistics!(portfolio)
    clust_alg = HAC()
    clust_opt = ClustOpt()
    cluster_assets!(portfolio; clust_alg = clust_alg, clust_opt = clust_opt)

    portfolio.w_min = -0.2
    portfolio.w_max = 0.8

    w1 = optimise!(portfolio,
                   NCO(;
                       internal = NCOArgs(; type = Trad(; obj = Sharpe()),
                                          port_kwargs = (; short = true, budget = 1 - 0.2))))
    wt = [-0.08730120213693945, 0.011322099554646166, 0.013258701666108447,
          0.004456083736416882, 0.22464712182353377, -0.04347233672893314,
          0.0430431906337467, 0.025386958821748403, 7.149349161172941e-10,
          5.56766815404211e-8, 0.03768814394629829, -7.264476589962164e-11,
          -0.013820028859141065, -1.1279888590095968e-10, -0.01540642352606083,
          0.10308907493700885, 0.18282202434296457, 0.03429684377765735,
          0.11998968659630506, 5.20846720848841e-9]
    @test isapprox(w1.weights, wt, rtol = 1.0e-7)
    @test all(w1.weights[w1.weights .>= 0] .<= 0.8)
    @test all(w1.weights[w1.weights .<= 0] .>= -0.2)
    @test sum(w1.weights[w1.weights .>= 0]) <= 0.8
    @test sum(abs.(w1.weights[w1.weights .<= 0])) <= 0.2

    w2 = optimise!(portfolio,
                   NCO(;
                       internal = NCOArgs(; type = Trad(; obj = Sharpe()),
                                          port_kwargs = (; short = true,
                                                         short_budget = -0.3,
                                                         short_lb = -0.3, long_ub = 0.6,
                                                         budget = 0.6 - 0.3))))
    wt = [-0.016918125661751243, 0.0017126889827391004, 0.0031512295564527604,
          0.00087743545602701, 0.02890363413289535, -0.0319389435912711,
          0.0073800612070850065, 0.006474270699306938, 1.194385717489518e-10,
          0.0035435264221449813, 0.01622401110353967, -0.0014765953212344742,
          -0.0042051490570914635, -0.005246447619489081, -0.006973408726769097,
          0.017930245342585454, 0.023139955294567992, 0.012461570130616495,
          0.035667407091996665, -0.0007073655617895324]
    @test isapprox(w2.weights, wt)
    @test all(w2.weights[w2.weights .>= 0] .<= 0.8)
    @test all(w2.weights[w2.weights .<= 0] .>= -0.2)
    @test sum(w2.weights[w2.weights .>= 0]) <= 0.6
    @test sum(abs.(w2.weights[w2.weights .<= 0])) <= 0.3

    w3 = optimise!(portfolio,
                   NCO(; internal = NCOArgs(; type = Trad(; obj = Sharpe())),
                       external = NCOArgs(; type = Trad(; obj = Sharpe()),
                                          port_kwargs = (; short = true,
                                                         short_budget = -0.6,
                                                         short_lb = -0.6, long_ub = 0.4,
                                                         budget = 0.4 - 0.6))))
    wt = [1.0339757656912842e-25, 4.9630836753181646e-24, 8.271806125530274e-25,
          8.271806125530274e-25, 5.551115123125781e-17, -1.0339757656912842e-25,
          6.938893903907226e-18, -2.7755575615628904e-17, -8.271806125530274e-25,
          -8.271806125530274e-25, -8.673617379884033e-19, 3.877409121342316e-26,
          1.2924697071141053e-26, -2.0679515313825685e-25, 1.2924697071141053e-26,
          1.3877787807814452e-17, 5.551115123125781e-17, -1.3877787807814452e-17,
          -0.19999999999999996, 4.135903062765137e-25]
    @test isapprox(w3.weights, wt)
    @test all(w3.weights[w3.weights .>= 0] .<= 0.8)
    @test all(w3.weights[w3.weights .<= 0] .>= -0.2)
    @test sum(w3.weights[w3.weights .>= 0]) <= 0.6
    @test sum(abs.(w3.weights[w3.weights .<= 0])) <= 0.3

    w4 = optimise!(portfolio,
                   NCO(;
                       internal = NCOArgs(; type = Trad(; obj = Sharpe()),
                                          port_kwargs = (; short = true,
                                                         short_budget = -0.3,
                                                         short_lb = -0.3, long_ub = 0.6,
                                                         budget = 0.6 - 0.3)),
                       external = NCOArgs(; type = Trad(; obj = Sharpe()),
                                          port_kwargs = (; short = true, budget = 1 - 0.2))))
    wt = [-0.045115002821432035, 0.004567170727617816, 0.008403279014043575,
          0.002339827937546212, 0.07707635949313746, -0.08517051763050223,
          0.01968016081423041, 0.017264722145996954, 3.1850286318443816e-10,
          0.009449403946282449, 0.04326403031405012, -0.003937587042169357,
          -0.011213729448736826, -0.013990527213158148, -0.018595754332555198,
          0.04781398173797328, 0.06170654889757941, 0.033230854222851665,
          0.0951130871276981, -0.0018863082089564777]
    @test isapprox(w4.weights, wt)
    @test all(w4.weights[w4.weights .>= 0] .<= 0.8)
    @test all(w4.weights[w4.weights .<= 0] .>= -0.2)
    @test sum(w4.weights[w4.weights .>= 0]) <= 0.6
    @test sum(abs.(w4.weights[w4.weights .<= 0])) <= 0.3

    w5 = optimise!(portfolio,
                   NCO(;
                       external = NCOArgs(; type = Trad(; obj = Sharpe()),
                                          port_kwargs = (; short = true,
                                                         short_budget = -0.6,
                                                         short_lb = -0.6, long_ub = 0.4,
                                                         budget = 0.4 - 0.6)),
                       internal = NCOArgs(; type = Trad(; obj = Sharpe()),
                                          port_kwargs = (; short = true, budget = 1 - 0.2))))
    wt = [-0.004706879982371151, 0.0006104356234245735, 0.0007148483175128095,
          0.000240251575296871, 0.012111941358492199, 0.0280346551249056,
          0.020593269042285294, -0.016371667335844244, -4.6105075821153227e-10,
          -3.590505325760551e-8, -0.024304516327714805, -3.4755630023204015e-11,
          -0.006611953442069123, 7.274230240773602e-11, -0.0073709364937928085,
          0.04932118238078999, 0.009856924139104854, -0.022117517932719263,
          -0.19999999999999996, 2.8081663759928254e-10]
    @test isapprox(w5.weights, wt)
    @test all(w5.weights[w5.weights .>= 0] .<= 0.8)
    @test all(w5.weights[w5.weights .<= 0] .>= -0.2)
    @test sum(w5.weights[w5.weights .>= 0]) <= 0.4
    @test sum(abs.(w5.weights[w5.weights .<= 0])) <= 0.6
end
