using Test,
    Logging,
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
    test_logger = TestLogger()
    with_logger(test_logger) do
        calc_risk(portfolio; rm = :EVaR)
    end
    @test !isempty(test_logger.logs)

    portfolio.solvers = OrderedDict(
        :ECOS => Dict(
            :solver => ECOS.Optimizer,
            :params => Dict("verbose" => false, "maxit" => 2),
        ),
    )
    @test_throws JuMP.OptimizeNotCalled() calc_risk(portfolio; rm = :RVaR)

    test_logger = TestLogger()
    with_logger(test_logger) do
        opt_port!(portfolio, rm = :RVaR)
    end
    @test !isempty(test_logger.logs)

    portfolio.solvers = OrderedDict(
        :ECOS => Dict(
            :solver => ECOS.Optimizer,
            :params => Dict("verbose" => false, "maxit" => 2),
        ),
    )
    test_logger = TestLogger()
    with_logger(test_logger) do
        opt_port!(portfolio, rm = :EVaR)
    end
    @test !isempty(test_logger.logs)

    portfolio.solvers = OrderedDict(
        :ECOS => Dict(:solver => ECOS.Optimizer, :params => Dict("verbose" => false)),
    )
    opt_port!(portfolio)

    portfolio.solvers = OrderedDict(
        :Clarabel => Dict(
            :solver => Clarabel.Optimizer,
            :params => Dict("verbose" => false, "max_iter" => 2),
        ),
    )
    test_logger = TestLogger()
    with_logger(test_logger) do
        calc_risk(portfolio; rm = :RVaR)
    end
    @test !isempty(test_logger.logs)
end
