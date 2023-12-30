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

@testset "$(:Classic), $(:RP), $(:SD)" begin
    portfolio = Portfolio(
        prices = prices,
        solvers = OrderedDict(
            :Clarabel => Dict(
                :solver => Clarabel.Optimizer,
                :params => Dict("verbose" => false, "max_step_fraction" => 0.75),
            ),
            :COSMO =>
                Dict(:solver => COSMO.Optimizer, :params => Dict("verbose" => false)),
        ),
    )
    asset_statistics!(portfolio)

    w1 = opt_port!(portfolio; class = :Classic, type = :RP, rm = :SD)
    rc1 = risk_contribution(portfolio, type = :RP, rm = :SD)
    lrc1, hrc1 = extrema(rc1)

    portfolio.risk_budget = 1:size(portfolio.returns, 2)
    w2 = opt_port!(portfolio; class = :Classic, type = :RP, rm = :SD)
    rc2 = risk_contribution(portfolio, type = :RP, rm = :SD)
    lrc2, hrc2 = extrema(rc2)

    @test isapprox(hrc1 / lrc1, 1, atol = 1e-3)
    @test isapprox(hrc2 / lrc2, 20, atol = 1e-2)
end

@testset "$(:Classic), $(:RP), $(:MAD)" begin
    portfolio = Portfolio(
        prices = prices,
        solvers = OrderedDict(
            :Clarabel => Dict(
                :solver => Clarabel.Optimizer,
                :params => Dict("verbose" => false, "max_step_fraction" => 0.75),
            ),
            :COSMO =>
                Dict(:solver => COSMO.Optimizer, :params => Dict("verbose" => false)),
        ),
    )
    asset_statistics!(portfolio)

    w1 = opt_port!(portfolio; class = :Classic, type = :RP, rm = :MAD)
    rc1 = risk_contribution(portfolio, type = :RP, rm = :MAD)
    lrc1, hrc1 = extrema(rc1)

    portfolio.risk_budget = 1:size(portfolio.returns, 2)
    w2 = opt_port!(portfolio; class = :Classic, type = :RP, rm = :MAD)
    rc2 = risk_contribution(portfolio, type = :RP, rm = :MAD)
    lrc2, hrc2 = extrema(rc2)

    @test isapprox(hrc1 / lrc1, 1, atol = 1e-2)
    @test isapprox(hrc2 / lrc2, 20, atol = 1e-1)
end