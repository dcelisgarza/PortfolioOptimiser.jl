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

portfolio.owa_w = owa_gmd(size(portfolio.returns, 1))
portfolio.risk_budget = []
w5 = opt_port!(portfolio; class = :Classic, type = :RP, rm = :OWA)
rc5 = risk_contribution(portfolio, type = :RP, rm = :OWA)
lrc5, hrc5 = extrema(rc5)

portfolio.risk_budget = 1:size(portfolio.returns, 2)
w6 = opt_port!(portfolio; class = :Classic, type = :RP, rm = :OWA)
rc6 = risk_contribution(portfolio, type = :RP, rm = :OWA)
lrc6, hrc6 = extrema(rc6)

w1t = [
    0.04802942512713756,
    0.051098633965431545,
    0.045745604303095315,
    0.04016462272315026,
    0.048191113840881636,
    0.04947756880152616,
    0.029822819526080995,
    0.06381728597897235,
    0.04728967147101808,
    0.04658112360117908,
    0.06791610810062289,
    0.02694231133104885,
    0.02315124666092132,
    0.07304653981777988,
    0.028275385479207917,
    0.04916987940293215,
    0.05681801358545005,
    0.07493768749423871,
    0.05401384214831002,
    0.07551111664101536,
]

w2t = [
    0.005218035899700653,
    0.01087674740630968,
    0.015253430555507657,
    0.018486437380714205,
    0.028496039178234907,
    0.027469099246927333,
    0.02325200625061058,
    0.046280910697853825,
    0.037790152343153555,
    0.04316409123859577,
    0.06424564256322021,
    0.028231869870286582,
    0.02515578633314724,
    0.09118437505023558,
    0.03757948263778634,
    0.06310045219606651,
    0.09826555499518812,
    0.11626540122133404,
    0.09025737976430415,
    0.12942710517082315,
]

@test isapprox(w1.weights, w1t, rtol = 1.0e-5)
@test isapprox(w2.weights, w2t, rtol = 1.0e-5)
@test isapprox(w1.weights, w3.weights)
@test isapprox(w2.weights, w4.weights)
@test isapprox(w1.weights, w5.weights)
@test isapprox(w2.weights, w6.weights)
@test isapprox(hrc1 / lrc1, 1, atol = 1e-3)
@test isapprox(hrc2 / lrc2, 20, atol = 1e-2)
@test isapprox(rc1, rc3)
@test isapprox(rc2, rc4)
@test isapprox(rc1, rc5)
@test isapprox(rc2, rc6)

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
        5.267987079518073e-9,
        9.394604382793292e-9,
        5.506040831392521e-9,
        1.2361625005429612e-8,
        0.8926332264310529,
        1.7331116339899812e-9,
        2.355741605645437e-9,
        9.111700979133426e-9,
        4.0970367442672564e-8,
        8.890303968195031e-9,
        4.410836572592712e-9,
        3.200982478793182e-9,
        1.8317608750453154e-9,
        4.214497059934697e-9,
        2.2628471028987997e-9,
        7.320198222749485e-8,
        0.10736655371458953,
        5.765824160821282e-9,
        2.4794360475682598e-8,
        4.579783692469168e-9,
    ],
    [
        6.8783457384567556e-9,
        1.2274774655426258e-8,
        7.210231160923748e-9,
        1.6157723542114e-8,
        0.8926318884419521,
        2.236669895286051e-9,
        3.0721278132094936e-9,
        1.1931574111287059e-8,
        5.313990853277208e-8,
        1.1675553033694841e-8,
        5.76375880463489e-9,
        4.1812752119557695e-9,
        2.3871108169602047e-9,
        5.509568787249582e-9,
        2.9557632944416428e-9,
        9.517716377203498e-8,
        0.10736782516025675,
        7.544504043725093e-9,
        3.230881102698668e-8,
        5.9929266940059665e-9,
    ]
    if isapprox(a1, a2, rtol = rtol)
        println(", rtol = $(rtol)")
        break
    end
end

isapprox(1.0359036144810312, 1; atol = 0.04)