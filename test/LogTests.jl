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
    COSMO,
    HiGHS

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

    portfolio = Portfolio(
        returns = returns,
        solvers = OrderedDict(
            :Clarabel => Dict(
                :solver => Clarabel.Optimizer,
                :params => Dict("verbose" => false),
            ),
        ),
    )
    asset_statistics!(portfolio, calc_kurt = false)
    opt_port!(portfolio)

    portfolio.solvers = OrderedDict(
        :Clarabel => Dict(
            :solver => Clarabel.Optimizer,
            :params => Dict("verbose" => false, "max_iter" => 2),
        ),
    )
    @test calc_risk(portfolio, rm = :RVaR) < 0
    @test isapprox(calc_risk(portfolio, rm = :Equal), 1 / size(portfolio.returns, 2))

    opt_port!(portfolio, type = :RP)
    @test !isempty(portfolio.fail[:Clarabel_RP])
    opt_port!(portfolio, type = :RRP)
    @test !isempty(portfolio.fail[:Clarabel_RRP])

    portfolio = Portfolio(
        returns = returns,
        solvers = OrderedDict(
            :Clarabel => Dict(
                :solver => Clarabel.Optimizer,
                :params => Dict("verbose" => false, "max_step_fraction" => 0.75),
            ),
        ),
        alloc_solvers = OrderedDict(
            :Clarabel => Dict(
                :solver => Clarabel.Optimizer,
                :params => Dict("verbose" => false),
            ),
        ),
    )

    asset_statistics!(portfolio, calc_kurt = false)
    opt_port!(portfolio)
    alloc = allocate_port!(
        portfolio,
        alloc_type = :LP,
        latest_prices = Vector(DataFrame(A)[end, 2:end]),
    )
    @test isempty(alloc)
    @test !isempty(portfolio.alloc_fail)

    portfolio = Portfolio(
        returns = returns,
        solvers = OrderedDict(
            :Clarabel => Dict(
                :solver => Clarabel.Optimizer,
                :params => Dict("verbose" => false, "max_step_fraction" => 0.75),
            ),
        ),
        alloc_solvers = OrderedDict(
            :Clarabel => Dict(
                :solver => Clarabel.Optimizer,
                :params => Dict("verbose" => false),
            ),
        ),
    )

    asset_statistics!(portfolio, calc_kurt = false)
    opt_port!(portfolio)

    portfolio.alloc_solvers = OrderedDict(
        :Clarabel =>
            Dict(:solver => Clarabel.Optimizer, :params => Dict("verbose" => false)),
    )
    alloc = allocate_port!(
        portfolio,
        alloc_type = :LP,
        latest_prices = Vector(DataFrame(A)[end, 2:end]),
        investment = 0,
    )
    @test isempty(alloc)
    @test !isempty(portfolio.alloc_fail)

    w1 = owa_l_moment_crm(
        50;
        k = 8,
        method = :E,
        g = 0.5,
        max_phi = 0.5,
        solvers = Dict(
            :Clarabel => Dict(
                :solver => Clarabel.Optimizer,
                :params => Dict("verbose" => false, "max_iter" => 1),
            ),
        ),
    )

    w2 = owa_l_moment_crm(50; k = 8, method = :CRRA, g = 0.5, max_phi = 0.5)

    @test isapprox(w2, w1)

    w3 = owa_l_moment_crm(
        50;
        k = 8,
        method = :E,
        g = 0.5,
        max_phi = 0.5,
        solvers = Dict(
            :HiGHS => Dict(
                :solver => HiGHS.Optimizer,
                :params => Dict("log_to_console" => false),
            ),
        ),
    )

    @test isapprox(w1, w3)
end
