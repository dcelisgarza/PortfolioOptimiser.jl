using COSMO,
    CSV,
    Clarabel,
    HiGHS,
    LinearAlgebra,
    OrderedCollections,
    PortfolioOptimiser,
    Statistics,
    Test,
    TimeSeries,
    SCS

prices = TimeArray(CSV.File("./test/assets/stock_prices.csv"); timestamp = :date)

rf = 1.0329^(1 / 252) - 1
l = 2.0
PortfolioOptimiser.RiskMeasures[(end - 5):end]

portfolio = Portfolio(
    prices = prices[(end - 200):end],
    solvers = OrderedDict(
        :Clarabel =>
            Dict(:solver => Clarabel.Optimizer, :params => Dict("verbose" => false)),
        :COSMO => Dict(:solver => COSMO.Optimizer, :params => Dict("verbose" => false)),
    ),
)
asset_statistics!(portfolio)

portfolio.risk_budget = []
w1 = opt_port!(portfolio; class = :Classic, type = :RP, rm = :GMD)
rc1 = risk_contribution(portfolio, type = :RP, rm = :GMD)
lrc1, hrc1 = extrema(rc1)

portfolio.risk_budget = 1:size(portfolio.returns, 2)
w2 = opt_port!(portfolio; class = :Classic, type = :RP, rm = :GMD)
rc2 = risk_contribution(portfolio, type = :RP, rm = :GMD)
lrc2, hrc2 = extrema(rc2)

portfolio.risk_budget = []
w3 = opt_port!(portfolio; class = :Classic, type = :RP, rm = :OWA)
rc3 = risk_contribution(portfolio, type = :RP, rm = :OWA)
lrc3, hrc3 = extrema(rc3)

portfolio.risk_budget = 1:size(portfolio.returns, 2)
w4 = opt_port!(portfolio; class = :Classic, type = :RP, rm = :OWA)
rc4 = risk_contribution(portfolio, type = :RP, rm = :OWA)
lrc4, hrc4 = extrema(rc4)

portfolio.risk_budget = []
portfolio.owa_w = owa_gmd(size(portfolio.returns, 1))
w5 = opt_port!(portfolio; class = :Classic, type = :RP, rm = :OWA)
rc5 = risk_contribution(portfolio, type = :RP, rm = :OWA)
lrc5, hrc5 = extrema(rc5)

portfolio.risk_budget = 1:size(portfolio.returns, 2)
w6 = opt_port!(portfolio; class = :Classic, type = :RP, rm = :OWA)
rc6 = risk_contribution(portfolio, type = :RP, rm = :OWA)
lrc6, hrc6 = extrema(rc6)

########################################
println("w1t = ", w1.weights, "\n")
println("w2t = ", w2.weights, "\n")
println("w3t = ", w3.weights, "\n")
println("w4t = ", w4.weights, "\n")
println("w5t = ", w5.weights, "\n")
println("w6t = ", w6.weights, "\n")
println("w7t = ", w7.weights, "\n")
println("w8t = ", w8.weights, "\n")
println("w9t = ", w9.weights, "\n")
println("w10t = ", w10.weights, "\n")
println("w11t = ", w11.weights, "\n")
println("w12t = ", w12.weights, "\n")
println("w13t = ", w13.weights, "\n")
println("w14t = ", w14.weights, "\n")
println("w15t = ", w15.weights, "\n")
println("w16t = ", w16.weights, "\n")
println("w17t = ", w17.weights, "\n")
println("w18t = ", w18.weights, "\n")
println("w19t = ", w19.weights, "\n")
#######################################

for rtol in [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0]
    a1, a2 = [
        7.582611034176794e-9,
        1.3520238401744016e-8,
        7.920576899789422e-9,
        1.7788884601713304e-8,
        0.8926326244897583,
        2.5003950980519887e-9,
        3.3915993611893507e-9,
        1.3106809645162286e-8,
        5.9048667365163836e-8,
        1.2781497501715161e-8,
        6.347762517432187e-9,
        4.606841189195817e-9,
        2.637510299634555e-9,
        6.064623605923661e-9,
        3.256782493558278e-9,
        1.0542559016028295e-7,
        0.10736705894388004,
        8.295445876259894e-9,
        3.570143925724501e-8,
        6.589086127667783e-9,
    ],
    [
        5.281732660436822e-9,
        9.419037695239505e-9,
        5.520434444023928e-9,
        1.239429971089012e-8,
        0.892633132842242,
        1.7376428817709362e-9,
        2.3619131030770845e-9,
        9.135101873512987e-9,
        4.106859242165867e-8,
        8.91342343822351e-9,
        4.42222741610075e-9,
        3.2092641333224004e-9,
        1.836533109671395e-9,
        4.2254267691398205e-9,
        2.268748888169778e-9,
        7.336116178740828e-8,
        0.10736664677420021,
        5.780713471779663e-9,
        2.4855646106661036e-8,
        4.591657831988204e-9,
    ]
    if isapprox(a1, a2, rtol = rtol)
        println(", rtol = $(rtol)")
        break
    end
end

isapprox(1.0359036144810312, 1; atol = 0.04)
