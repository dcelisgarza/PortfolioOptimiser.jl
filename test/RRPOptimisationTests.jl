using CSV, TimeSeries, DataFrames, StatsBase, Statistics, LinearAlgebra, Test, Clarabel,
      PortfolioOptimiser

prices = TimeArray(CSV.File("./assets/stock_prices.csv"); timestamp = :date)
rf = 1.0329^(1 / 252) - 1
l = 2.0

@testset "RRP" begin
    portfolio = Portfolio2(; prices = prices,
                           solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                            :params => Dict("verbose" => false,
                                                                            "max_step_fraction" => 0.75))))
    asset_statistics2!(portfolio)

    type = RRP2(; version = NoRRP())
    w1 = optimise2!(portfolio; type = type)
    rc1 = risk_contribution(portfolio; type = :RRP2, rm = SD2())
    lrc1, hrc1 = extrema(rc1)

    portfolio.risk_budget = 1:size(portfolio.returns, 2)
    w2 = optimise2!(portfolio; type = type)
    rc2 = risk_contribution(portfolio; type = :RRP2, rm = SD2())
    lrc2, hrc2 = extrema(rc2)

    w1t = [0.05063226479167363, 0.05124795179217185, 0.046905305668578534,
           0.04369000590266316, 0.04571523414069997, 0.0561504736045873,
           0.027632874707693277, 0.07705758767773013, 0.03949584161061521,
           0.047233362744388926, 0.08435232781169791, 0.03385562285583431,
           0.027546000241178287, 0.06206725606864607, 0.03563645238467477,
           0.04413102217023868, 0.0508526986276921, 0.07142013095945429,
           0.04529644142711541, 0.05908114481266622]
    w2t = [0.005639469044832487, 0.011008565638056241, 0.01558191867721576,
           0.01936965971929752, 0.025437280051017967, 0.032657021338339344,
           0.020371105753645584, 0.059571551767319604, 0.03298226974416533,
           0.044829463418802465, 0.0874116876205738, 0.03822896613818936,
           0.03143795043334007, 0.07961004371546289, 0.04614268829885533,
           0.06097518104030126, 0.08329762064545874, 0.11639291288060472,
           0.07928202193671541, 0.10977262213780618]
    @test isapprox(w1.weights, w1t)
    @test isapprox(w2.weights, w2t)
    @test isapprox(hrc1 / lrc1, 1, rtol = 5.0e-6)
    @test isapprox(hrc2 / lrc2, 20, rtol = 5.0e-5)

    type = RRP2(; version = RegRRP())
    portfolio.risk_budget = []
    w3 = optimise2!(portfolio; type = type)
    rc3 = risk_contribution(portfolio; type = :RRP2, rm = SD2())
    lrc3, hrc3 = extrema(rc3)

    portfolio.risk_budget = 1:size(portfolio.returns, 2)
    w4 = optimise2!(portfolio; type = type)
    rc4 = risk_contribution(portfolio; type = :RRP2, rm = SD2())
    lrc4, hrc4 = extrema(rc4)

    w3t = [0.050632270730510257, 0.051247955272033616, 0.04690531847722307,
           0.04369002239964449, 0.04571524916916414, 0.05615046484821999,
           0.027632904568915057, 0.07705752361934728, 0.03949586646990367,
           0.047233374046681875, 0.08435225311195688, 0.03385565255115441,
           0.027546029437221013, 0.06206723047660951, 0.035636477479759204,
           0.04413103748385253, 0.0508527033037059, 0.07142008062495746,
           0.045296457596162756, 0.05908112833297674]
    w4t = [0.005639461552718216, 0.011008557375561395, 0.015581908682111244,
           0.019369650058062862, 0.025437270632592938, 0.03265701514557647,
           0.020371093928756717, 0.05957158741316405, 0.032982254152707266,
           0.04482945069969184, 0.08741177245126876, 0.03822894980586257,
           0.031437935765919155, 0.0796100512239314, 0.04614267208265509,
           0.06097516596688575, 0.08329761438311381, 0.11639294963092589,
           0.07928200651128077, 0.10977263253721378]
    @test isapprox(w3.weights, w3t)
    @test isapprox(w4.weights, w4t)
    @test isapprox(hrc3 / lrc3, 1, rtol = 5.0e-6)
    @test isapprox(hrc4 / lrc4, 20, rtol = 5.0e-5)

    type = RRP2(; version = RegPenRRP(; penalty = 1))
    portfolio.risk_budget = []
    w5 = optimise2!(portfolio; type = type)
    rc5 = risk_contribution(portfolio; type = :RRP2, rm = SD2())
    lrc5, hrc5 = extrema(rc5)

    portfolio.risk_budget = 1:size(portfolio.returns, 2)
    w6 = optimise2!(portfolio; type = type)
    rc6 = risk_contribution(portfolio; type = :RRP2, rm = SD2())
    lrc6, hrc6 = extrema(rc6)

    w5t = [0.05063226685509387, 0.05124795374842603, 0.04690530823181724,
           0.04369000763867945, 0.045715235863210156, 0.056150474428725373,
           0.027632877632568886, 0.07705757228254563, 0.03949584655734763,
           0.04723336591667807, 0.08435231111770436, 0.03385562639058922,
           0.027546003010931384, 0.06206725533783764, 0.03563645410937717,
           0.044131022956236346, 0.050852702447945715, 0.07142012424723757,
           0.04529644649169936, 0.05908114473534884]
    w6t = [0.0056395011255634345, 0.011008597273272958, 0.015581943908876048,
           0.019369676991244516, 0.025437294593588882, 0.032657044157293064,
           0.02037111119087388, 0.059571547421196615, 0.03298228536692838,
           0.044829472793304674, 0.08741165574203688, 0.0382289721921966,
           0.03143795617738833, 0.07961003000881385, 0.046142690224315,
           0.060975177653712395, 0.08329760503151232, 0.116392848160018,
           0.07928201765576764, 0.10977257233209663]
    @test isapprox(w5.weights, w5t)
    @test isapprox(w6.weights, w6t)
    @test isapprox(hrc5 / lrc5, 1, rtol = 5.0e-6)
    @test isapprox(hrc6 / lrc6, 20, rtol = 5.0e-5)

    type = RRP2(; version = RegPenRRP(; penalty = 5))
    portfolio.risk_budget = []
    w7 = optimise2!(portfolio; type = type)
    rc7 = risk_contribution(portfolio; type = :RRP2, rm = SD2())
    lrc7, hrc7 = extrema(rc7)

    portfolio.risk_budget = 1:size(portfolio.returns, 2)
    w8 = optimise2!(portfolio; type = type)
    rc8 = risk_contribution(portfolio; type = :RRP2, rm = SD2())
    lrc8, hrc8 = extrema(rc8)

    w7t = [0.05063227092483982, 0.051247957970220706, 0.046905313223576876,
           0.04369000832670002, 0.04571523607142845, 0.056150476586436994,
           0.027632885874903337, 0.077057507452318, 0.03949586365421145,
           0.047233374862919744, 0.08435226765306823, 0.03385563474314793,
           0.02754601140611397, 0.062067261204944754, 0.035636456892157733,
           0.044131020188731974, 0.05085272421479202, 0.07142010982372254,
           0.04529647152631459, 0.059081147399450785]
    w8t = [0.016803340442582207, 0.019323023095702243, 0.015068043628123692,
           0.0188459404418481, 0.024570916279022192, 0.032040946880747094,
           0.02000931588824339, 0.0584656081395763, 0.032347338194528834,
           0.04397847882346323, 0.08577138764605112, 0.03753687980177539,
           0.031062491168826374, 0.07812625162681952, 0.04562332519102003,
           0.0600409335633433, 0.08125333541684177, 0.11408515333305874,
           0.07775046163627136, 0.10729682880215521]
    @test isapprox(w7.weights, w7t)
    @test isapprox(w8.weights, w8t, rtol = 5.0e-7)
    @test isapprox(hrc7 / lrc7, 1, rtol = 5.0e-6)
    @test isapprox(hrc8 / lrc8, 7, rtol = 0.05)

    portfolio.risk_budget = []
    portfolio.risk_budget[1] = 5
    w9 = optimise2!(portfolio; type = type)
    rc9 = risk_contribution(portfolio; type = :RRP2, rm = SD2())
    lrc9, hrc9 = extrema(rc9)
    @test isapprox(hrc9 / lrc9, 100, rtol = 0.1)
    @test isapprox(sum(portfolio.risk_budget), 1)
end
