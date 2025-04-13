using CSV, Clarabel, DataFrames, HiGHS, LinearAlgebra, PortfolioOptimiser, Statistics, Test,
      TimeSeries, Logging, JuMP

path = joinpath(@__DIR__, "assets/stock_prices.csv")
prices = TimeArray(CSV.File(path); timestamp = :date)

rf = 1.0329^(1 / 252) - 1
l = 2.0

@testset "ERM and RRM logs" begin
    portfolio = Portfolio(; prices = prices,
                          solvers = PortOptSolver(; name = :Clarabel,
                                                  check_sol = (; allow_local = true,
                                                               allow_almost = true),
                                                  solver = Clarabel.Optimizer,
                                                  params = Dict("verbose" => false,
                                                                "max_step_fraction" => 0.75)))

    asset_statistics!(portfolio)
    optimise!(portfolio, Trad())

    solvers = PortOptSolver(; name = :Clarabel, solver = Clarabel.Optimizer,
                            check_sol = (; allow_local = true, allow_almost = true),
                            params = Dict("verbose" => false, "max_iter" => 1))

    solvers_mip = PortOptSolver(; name = :HiGHS, solver = HiGHS.Optimizer,
                                check_sol = (; allow_local = true, allow_almost = true),
                                params = Dict("log_to_console" => false))
    portfolio.solvers = solvers
    test_logger = TestLogger()
    with_logger(test_logger) do
        expected_risk(portfolio; rm = EDaR())
        expected_risk(portfolio; rm = RLDaR())
        portfolio.solvers = solvers_mip
        expected_risk(portfolio; rm = RLDaR())
        return nothing
    end

    @test test_logger.logs[1].level == Warn
    @test test_logger.logs[2].level == Warn
    @test test_logger.logs[3].level == Warn
    @test contains(test_logger.logs[1].message,
                   "PortfolioOptimiser.ERM: Model could not be optimised satisfactorily.")
    @test contains(test_logger.logs[2].message,
                   "PortfolioOptimiser.RRM: Model could not be optimised satisfactorily.")
    @test contains(test_logger.logs[3].message,
                   "PortfolioOptimiser.RRM: Model could not be optimised satisfactorily.")
end

@testset "EVaR ERM" begin
    portfolio = Portfolio(; prices = prices,
                          solvers = PortOptSolver(; name = :Clarabel,
                                                  check_sol = (; allow_local = true,
                                                               allow_almost = true),
                                                  solver = Clarabel.Optimizer,
                                                  params = Dict("verbose" => false,
                                                                "max_step_fraction" => 0.75)))
    asset_statistics!(portfolio)
    optimise!(portfolio, Trad(; rm = EVaR(), obj = Sharpe()))

    x = portfolio.returns * portfolio.optimal[:Trad].weights

    alpha = 0.05
    r1 = PortfolioOptimiser.ERM(x, get_z(portfolio, EVaR()), alpha)
    r1t = expected_risk(portfolio; rm = EVaR(; alpha = alpha))

    alpha = 0.1
    r2 = PortfolioOptimiser.ERM(x, get_z(portfolio, EVaR()), alpha)
    r2t = expected_risk(portfolio; rm = EVaR(; alpha = alpha))

    alpha = 0.15
    r3 = PortfolioOptimiser.ERM(x, get_z(portfolio, EVaR()), alpha)
    r3t = expected_risk(portfolio; rm = EVaR(; alpha = alpha))

    @test isapprox(r1, r1t, rtol = 5e-6)
    @test isapprox(r2, r2t, rtol = 3e-2)
    @test isapprox(r3, r3t, rtol = 5e-2)
end

@testset "EDaR ERM" begin
    portfolio = Portfolio(; prices = prices,
                          solvers = PortOptSolver(; name = :Clarabel,
                                                  check_sol = (; allow_local = true,
                                                               allow_almost = true),
                                                  solver = Clarabel.Optimizer,
                                                  params = Dict("verbose" => false,
                                                                "max_step_fraction" => 0.75)))
    asset_statistics!(portfolio)
    optimise!(portfolio, Trad(; rm = EDaR(), obj = Sharpe()))

    x = portfolio.returns * portfolio.optimal[:Trad].weights
    pushfirst!(x, 1)
    cs = cumsum(x)
    peak = -Inf
    dd = similar(cs)
    for (idx, i) ∈ pairs(cs)
        if i > peak
            peak = i
        end
        dd[idx] = -(peak - i)
    end
    popfirst!(dd)

    alpha = 0.05
    r1 = PortfolioOptimiser.ERM(dd, get_z(portfolio, EDaR()), alpha)
    r1t = expected_risk(portfolio; rm = EDaR(; alpha = alpha))

    alpha = 0.1
    r2 = PortfolioOptimiser.ERM(dd, get_z(portfolio, EDaR()), alpha)
    r2t = expected_risk(portfolio; rm = EDaR(; alpha = alpha))

    alpha = 0.15
    r3 = PortfolioOptimiser.ERM(dd, get_z(portfolio, EDaR()), alpha)
    r3t = expected_risk(portfolio; rm = EDaR(; alpha = alpha))

    @test isapprox(r1, r1t, rtol = 5e-6)
    @test isapprox(r2, r2t, rtol = 5e-2)
    @test isapprox(r3, r3t, rtol = 5e-2)
end
