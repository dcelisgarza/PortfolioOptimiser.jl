using CSV, TimeSeries, DataFrames, StatsBase, Statistics, LinearAlgebra, Test, Clarabel,
      PortfolioOptimiser

path = joinpath(@__DIR__, "assets/stock_prices.csv")
prices = TimeArray(CSV.File(path); timestamp = :date)
rf = 1.0329^(1 / 252) - 1
l = 2.0

@testset "RRB" begin
    portfolio = Portfolio(; prices = prices,
                          solvers = PortOptSolver(; name = :Clarabel,
                                                  solver = Clarabel.Optimizer,
                                                  check_sol = (; allow_local = true,
                                                               allow_almost = true),
                                                  params = Dict("verbose" => false,
                                                                "max_step_fraction" => 0.75)))

    asset_statistics!(portfolio)

    type = RRB(; version = BasicRRB())
    w1 = optimise!(portfolio, type)
    rc1 = risk_contribution(portfolio, :RRB; rm = SD())
    lrc1, hrc1 = extrema(rc1)

    portfolio.risk_budget = 1:size(portfolio.returns, 2)
    portfolio.risk_budget /= sum(portfolio.risk_budget)
    w2 = optimise!(portfolio, type)
    rc2 = risk_contribution(portfolio, :RRB; rm = SD())
    lrc2, hrc2 = extrema(rc2)

    w1t = [0.05063226719737215, 0.05124795346377443, 0.046905309681765, 0.04369001060019904,
           0.045715238687852246, 0.05615047218925998, 0.027632882659481054,
           0.07705756982281324, 0.039495848725731246, 0.04723336639532292,
           0.08435230037136743, 0.03385563107907952, 0.02754600799616904,
           0.06206724990679756, 0.035636459225178725, 0.04413102663431261,
           0.050852700653626126, 0.07142011714386644, 0.045296446435196515,
           0.05908114113083477]
    w2t = [0.0056394328053155935, 0.011008525225536076, 0.015581880501501943,
           0.019369627395031896, 0.025437250436485714, 0.03265699451596497,
           0.020371074866607088, 0.05957158932810877, 0.032982230567180776,
           0.04482943739737062, 0.08741181468820308, 0.03822892507181403,
           0.031437909445635515, 0.07961008642009944, 0.04614264707824496,
           0.060975152663028956, 0.08329764074659259, 0.11639306202439181,
           0.07928201138340045, 0.1097727074394857]
    @test isapprox(w1.weights, w1t)
    @test isapprox(w2.weights, w2t, rtol = 5.0e-8)
    @test isapprox(hrc1 / lrc1, 1, rtol = 5.0e-6)
    @test isapprox(hrc2 / lrc2, 20, rtol = 5.0e-5)

    type = RRB(; version = RegRRB())
    portfolio.risk_budget = []
    w3 = optimise!(portfolio, type)
    rc3 = risk_contribution(portfolio, :RRB; rm = SD())
    lrc3, hrc3 = extrema(rc3)

    portfolio.risk_budget = 1:size(portfolio.returns, 2)
    portfolio.risk_budget /= sum(portfolio.risk_budget)
    w4 = optimise!(portfolio, type)
    rc4 = risk_contribution(portfolio, :RRB; rm = SD())
    lrc4, hrc4 = extrema(rc4)

    w3t = [0.05063226265208921, 0.051247949474025334, 0.04690530448046275,
           0.04369000560063853, 0.04571523333348346, 0.05615047014854784,
           0.02763287718852671, 0.07705758820220819, 0.03949584208794699,
           0.04723336112178464, 0.08435234266549022, 0.03385562462544245,
           0.027546002498251905, 0.0620672527117022, 0.035636453477427506,
           0.04413102144458269, 0.050852696280369744, 0.07142013017556112,
           0.04529644020541152, 0.05908114162604694]
    w4t = [0.005639438814041853, 0.011008531361532535, 0.01558188507106465,
           0.019369630646355763, 0.025437253002110277, 0.032656998235127625,
           0.020371075955375982, 0.05957160028067468, 0.03298223178290795,
           0.04482943642019504, 0.0874118338049165, 0.038228925836318324,
           0.031437911352374934, 0.07961007844116762, 0.04614264806095568,
           0.06097515046173879, 0.08329763091250345, 0.11639304355652262,
           0.07928200344615229, 0.10977269255796346]
    @test isapprox(w3.weights, w3t)
    @test isapprox(w4.weights, w4t)
    @test isapprox(hrc3 / lrc3, 1, rtol = 5.0e-6)
    @test isapprox(hrc4 / lrc4, 20, rtol = 5.0e-5)

    type = RRB(; version = RegPenRRB(; penalty = 1))
    portfolio.risk_budget = []
    w5 = optimise!(portfolio, type)
    rc5 = risk_contribution(portfolio, :RRB; rm = SD())
    lrc5, hrc5 = extrema(rc5)

    portfolio.risk_budget = 1:size(portfolio.returns, 2)
    portfolio.risk_budget /= sum(portfolio.risk_budget)
    w6 = optimise!(portfolio, type)
    rc6 = risk_contribution(portfolio, :RRB; rm = SD())
    lrc6, hrc6 = extrema(rc6)

    w5t = [0.05063225921797402, 0.05124794743224515, 0.04690529686433053,
           0.04368999559401664, 0.045715224130448255, 0.05615047560068797,
           0.02763285938908474, 0.07705762217018997, 0.03949582806450663,
           0.04723335493433706, 0.08435238942126634, 0.033855607083500786,
           0.027545985038285082, 0.062067267770753144, 0.03563643806030149,
           0.04413101164925449, 0.050852694321241586, 0.0714201600521726,
           0.04529643152307289, 0.05908115168233062]
    w6t = [0.0056394498604367335, 0.011008542472726762, 0.015581894575246162,
           0.01936963768123024, 0.025437259127920445, 0.03265700666010347,
           0.020371079391988425, 0.059571583863501386, 0.0329822406198552,
           0.04482944306052096, 0.08741179657725219, 0.03822893018790038,
           0.03143791443774447, 0.07961007627949554, 0.046142650282107386,
           0.06097515209885989, 0.08329763128680183, 0.11639302203782459,
           0.07928200811003576, 0.10977268138844819]
    @test isapprox(w5.weights, w5t, rtol = 5.0e-8)
    @test isapprox(w6.weights, w6t)
    @test isapprox(hrc5 / lrc5, 1, rtol = 5.0e-6)
    @test isapprox(hrc6 / lrc6, 20, rtol = 5.0e-5)

    type = RRB(; version = RegPenRRB(; penalty = 5))
    portfolio.risk_budget = []
    w7 = optimise!(portfolio, type)
    rc7 = risk_contribution(portfolio, :RRB; rm = SD())
    lrc7, hrc7 = extrema(rc7)

    portfolio.risk_budget = 1:size(portfolio.returns, 2)
    portfolio.risk_budget /= sum(portfolio.risk_budget)
    w8 = optimise!(portfolio, type)
    rc8 = risk_contribution(portfolio, :RRB; rm = SD())
    lrc8, hrc8 = extrema(rc8)

    w7t = [0.0506322603507147, 0.05124794887677536, 0.046905298286355805,
           0.04368999501422867, 0.0457152234226552, 0.05615047642203035,
           0.027632861151734713, 0.07705760320143674, 0.03949583375402043,
           0.047233357690021016, 0.08435237498526184, 0.033855608948126226,
           0.027545986861098583, 0.062067271226004056, 0.035636437968050175,
           0.04413100982799048, 0.050852702383175234, 0.07142015602173758,
           0.04529644057134947, 0.05908115303723339]
    w8t = [0.0168032717844555, 0.019323019852586762, 0.015067515113501753,
           0.018845911318337724, 0.024570910918581264, 0.03204079970574152,
           0.020009314893338397, 0.05846564720558206, 0.032347325644593306,
           0.04397848890542005, 0.08577149765413748, 0.03753689202285338,
           0.031062487227465334, 0.07812632725091691, 0.045623332402971575,
           0.060040960107111775, 0.08125342791160793, 0.1140853504071086,
           0.07775051878049052, 0.10729700089319821]
    @test isapprox(w7.weights, w7t)
    @test isapprox(w8.weights, w8t, rtol = 5.0e-7)
    @test isapprox(hrc7 / lrc7, 1, rtol = 5.0e-6)
    @test isapprox(hrc8 / lrc8, 7, rtol = 0.05)

    portfolio.risk_budget = fill(inv(20), 20)
    portfolio.risk_budget[1] = 5
    portfolio.risk_budget /= sum(portfolio.risk_budget)
    w9 = optimise!(portfolio, type)
    rc9 = risk_contribution(portfolio, :RRB; rm = SD())
    lrc9, hrc9 = extrema(rc9)
    @test isapprox(hrc9 / lrc9, 100, rtol = 0.1)
    @test isapprox(sum(portfolio.risk_budget), 1)
end
