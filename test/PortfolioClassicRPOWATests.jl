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

prices = TimeArray(CSV.File("./assets/stock_prices.csv"); timestamp = :date)

rf = 1.0329^(1 / 252) - 1
l = 2.0

@testset "$(:Classic), $(:RP), $(:GMD), blank $(:OWA) and owa_w = owa_gmd(T)" begin
    portfolio = Portfolio(
        prices = prices[(end - 200):end],
        solvers = OrderedDict(
            :Clarabel => Dict(
                :solver => Clarabel.Optimizer,
                :params => Dict("verbose" => false),
            ),
            :COSMO =>
                Dict(:solver => COSMO.Optimizer, :params => Dict("verbose" => false)),
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

    @test isapprox(w1.weights, w1t, rtol = 1.0e-3)
    @test isapprox(w2.weights, w2t, rtol = 1.0e-4)
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
end
