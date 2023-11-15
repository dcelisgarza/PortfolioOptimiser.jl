using Test,
    OrderedCollections,
    JuMP,
    PortfolioOptimiser,
    CSV,
    DataFrames,
    Dates,
    TimeSeries,
    ECOS,
    Clarabel,
    COSMO

A = TimeArray(CSV.File("./assets/stock_prices.csv"), timestamp = :date)
Y = percentchange(A)
returns = dropmissing!(DataFrame(Y))

@testset "Log tests" begin
    portfolio = Portfolio(
        returns = returns,
        solvers = OrderedDict(
            :ECOS =>
                Dict(:solver => ECOS.Optimizer, :params => Dict("verbose" => false)),
        ),
    )
    asset_statistics!(portfolio, calc_kurt = false)
    opt_port!(portfolio)

    portfolio.solvers = OrderedDict(
        :ECOS => Dict(
            :solver => ECOS.Optimizer,
            :params => Dict("verbose" => false, "maxit" => 2),
        ),
    )
    asset_statistics!(portfolio, calc_kurt = false)

    r1 = calc_risk(portfolio; rm = :EVaR)
    portfolio.solvers = OrderedDict(
        :ECOS => Dict(
            :solver => ECOS.Optimizer,
            :params => Dict("verbose" => false, "maxit" => 100),
        ),
    )
    r2 = calc_risk(portfolio; rm = :EVaR)
    @test r1 > r2

    portfolio.solvers = OrderedDict(
        :ECOS => Dict(
            :solver => ECOS.Optimizer,
            :params => Dict("verbose" => false, "maxit" => 2),
        ),
    )
    @test_throws JuMP.OptimizeNotCalled() calc_risk(portfolio; rm = :RVaR)

    portfolio.solvers = OrderedDict(
        :ECOS => Dict(
            :solver => ECOS.Optimizer,
            :params => Dict("verbose" => false, "maxit" => 2),
        ),
    )
    w1 = opt_port!(portfolio, rm = :EVaR)
    @test isempty(w1)

    portfolio.solvers = OrderedDict(
        :ECOS => Dict(:solver => ECOS.Optimizer, :params => Dict("verbose" => false)),
    )
    opt_port!(portfolio)

    portfolio.solvers = OrderedDict(
        :ECOS => Dict(
            :solver => ECOS.Optimizer,
            :params => Dict("verbose" => false, "max_iter" => 2),
        ),
    )

    w1 = opt_port!(portfolio; rm = :RVaR)
    @test isempty(w1)
end
