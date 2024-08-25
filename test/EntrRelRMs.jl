using CSV, Clarabel, DataFrames, HiGHS, LinearAlgebra, OrderedCollections,
      PortfolioOptimiser, Statistics, Test, TimeSeries

prices = TimeArray(CSV.File("./assets/stock_prices.csv"); timestamp = :date)

rf = 1.0329^(1 / 252) - 1
l = 2.0

@testset "ERM and RRM logs" begin
    portfolio = Portfolio(; prices = prices,
                          solvers = OrderedDict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                                  :params => Dict("verbose" => false,
                                                                                  "max_step_fraction" => 0.75))))
    asset_statistics!(portfolio; calc_kurt = false)
    optimise!(portfolio)

    solvers = OrderedDict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                            :params => Dict("verbose" => false,
                                                            "max_iter" => 1)))
    solvers_mip = OrderedDict(:HiGHS => Dict(:solver => HiGHS.Optimizer,
                                             :params => Dict("log_to_console" => false)))
    portfolio.solvers = solvers
    test_logger = TestLogger()
    with_logger(test_logger) do
        calc_risk(portfolio; rm = :EDaR)
        calc_risk(portfolio; rm = :RDaR)
        portfolio.solvers = solvers_mip
        calc_risk(portfolio; rm = :RDaR)
        return nothing
    end

    @test test_logger.logs[1].level == Warn
    @test test_logger.logs[2].level == Warn
    @test test_logger.logs[3].level == Warn
    @test contains(test_logger.logs[1].message,
                   "PortfolioOptimiser.ERM: model could not be optimised satisfactorily.")
    @test contains(test_logger.logs[2].message,
                   "PortfolioOptimiser.RRM: model could not be optimised satisfactorily.")
    @test contains(test_logger.logs[3].message,
                   "PortfolioOptimiser.RRM: model could not be optimised satisfactorily.")
end

@testset "EVaR ERM" begin
    portfolio = Portfolio(; prices = prices,
                          solvers = OrderedDict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                                  :params => Dict("verbose" => false,
                                                                                  "max_step_fraction" => 0.75))))
    asset_statistics!(portfolio; calc_kurt = false)
    optimise!(portfolio, OptimiseOpt(; rm = :EVaR, obj = :Sharpe))

    x = portfolio.returns * portfolio.optimal[:Trad].weights

    alpha = 0.05
    portfolio.alpha = alpha
    r1 = PortfolioOptimiser.ERM(x, portfolio.z[:Trad_z_evar], alpha)
    r1t = calc_risk(portfolio; rm = :EVaR)

    alpha = 0.1
    portfolio.alpha = alpha
    r2 = PortfolioOptimiser.ERM(x, portfolio.z[:Trad_z_evar], alpha)
    r2t = calc_risk(portfolio; rm = :EVaR)

    alpha = 0.15
    portfolio.alpha = alpha
    r3 = PortfolioOptimiser.ERM(x, portfolio.z[:Trad_z_evar], alpha)
    r3t = calc_risk(portfolio; rm = :EVaR)

    @test isapprox(r1, r1t, rtol = 5e-6)
    @test isapprox(r2, r2t, rtol = 3e-2)
    @test isapprox(r3, r3t, rtol = 5e-2)
end

@testset "EDaR ERM" begin
    portfolio = Portfolio(; prices = prices,
                          solvers = OrderedDict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                                  :params => Dict("verbose" => false,
                                                                                  "max_step_fraction" => 0.75))))
    asset_statistics!(portfolio; calc_kurt = false)
    optimise!(portfolio, OptimiseOpt(; rm = :EDaR, obj = :Sharpe))

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
    portfolio.alpha = alpha
    r1 = PortfolioOptimiser.ERM(dd, portfolio.z[:Trad_z_edar], alpha)
    r1t = calc_risk(portfolio; rm = :EDaR)

    alpha = 0.1
    portfolio.alpha = alpha
    r2 = PortfolioOptimiser.ERM(dd, portfolio.z[:Trad_z_edar], alpha)
    r2t = calc_risk(portfolio; rm = :EDaR)

    alpha = 0.15
    portfolio.alpha = alpha
    r3 = PortfolioOptimiser.ERM(dd, portfolio.z[:Trad_z_edar], alpha)
    r3t = calc_risk(portfolio; rm = :EDaR)

    @test isapprox(r1, r1t, rtol = 1e-6)
    @test isapprox(r2, r2t, rtol = 5e-2)
    @test isapprox(r3, r3t, rtol = 5e-2)
end
