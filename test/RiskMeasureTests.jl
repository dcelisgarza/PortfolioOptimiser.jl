using Test,
    PortfolioOptimiser,
    DataFrames,
    TimeSeries,
    CSV,
    Dates,
    ECOS,
    SCS,
    Clarabel,
    COSMO,
    OrderedCollections,
    LinearAlgebra,
    StatsBase

A = TimeArray(CSV.File("./assets/stock_prices.csv"), timestamp = :date)
Y = percentchange(A)
returns = dropmissing!(DataFrame(Y))

@testset "Risk measures" begin
    portfolio = Portfolio(
        returns = returns,
        solvers = OrderedDict(
            :Clarabel => Dict(
                :solver => (Clarabel.Optimizer),
                :params => Dict("verbose" => false),
            ),
            :COSMO =>
                Dict(:solver => COSMO.Optimizer, :params => Dict("verbose" => false)),
            :ECOS =>
                Dict(:solver => ECOS.Optimizer, :params => Dict("verbose" => false)),
            :SCS => Dict(:solver => SCS.Optimizer, :params => Dict("verbose" => 0)),
        ),
    )
    asset_statistics!(portfolio)

    N = length(portfolio.assets)
    w = fill(1 / N, N)

    @test abs(
        calc_risk(w, portfolio.returns, portfolio.cov; rm = :mv)[1] - 0.000101665490230637,
    ) < eps()
    @test abs(
        calc_risk(w, portfolio.returns, portfolio.cov; rm = :msd)[1] - 0.010082930637004155,
    ) < eps()
    @test abs(
        calc_risk(w, portfolio.returns, portfolio.cov; rm = :mad)[1] - 0.007418863748729646,
    ) < eps()
    @test abs(
        calc_risk(w, portfolio.returns, portfolio.cov; rm = :msv)[1] - 0.007345533015355076,
    ) < eps()
    @test abs(
        calc_risk(w, portfolio.returns, portfolio.cov; rm = :flpm)[1] -
        0.0034827678064358134,
    ) < eps()
    @test abs(
        calc_risk(w, portfolio.returns, portfolio.cov; rm = :slpm)[1] -
        0.007114744825145661,
    ) < eps()
    @test abs(
        calc_risk(w, portfolio.returns, portfolio.cov; rm = :wr)[1] - 0.043602428699089285,
    ) < eps()
    @test abs(
        calc_risk(w, portfolio.returns, portfolio.cov; rm = :var)[1] - 0.016748899891587572,
    ) < eps()
    @test abs(
        calc_risk(w, portfolio.returns, portfolio.cov; rm = :cvar)[1] - 0.02405795664064266,
    ) < eps()
    @test abs(
        calc_risk(
            w,
            portfolio.returns,
            portfolio.cov;
            rm = :evar,
            solvers = portfolio.solvers,
        )[1] - 0.030225422932337445,
    ) < 9e-8
    @test abs(
        calc_risk(
            w,
            portfolio.returns,
            portfolio.cov;
            rm = :rvar,
            solvers = portfolio.solvers,
        )[1] - 0.03586321171352101,
    ) < 1e-5
    @test abs(
        calc_risk(
            w,
            portfolio.returns,
            portfolio.cov;
            rm = :mdd,
            solvers = portfolio.solvers,
        )[1] - 0.1650381304766847,
    ) < 2.1 * eps()
    @test abs(
        calc_risk(
            w,
            portfolio.returns,
            portfolio.cov;
            rm = :add,
            solvers = portfolio.solvers,
        )[1] - 0.02762516797999026,
    ) < eps()
    @test abs(
        calc_risk(
            w,
            portfolio.returns,
            portfolio.cov;
            rm = :dar,
            solvers = portfolio.solvers,
        )[1] - 0.09442013028621254,
    ) < 4.6 * eps()
    @test abs(
        calc_risk(
            w,
            portfolio.returns,
            portfolio.cov;
            rm = :cdar,
            solvers = portfolio.solvers,
        )[1] - 0.11801077171629008,
    ) < 2 * eps()
    @test abs(
        calc_risk(
            w,
            portfolio.returns,
            portfolio.cov;
            rm = :uci,
            solvers = portfolio.solvers,
        )[1] - 0.0402491262027023,
    ) < eps()
    @test abs(
        calc_risk(
            w,
            portfolio.returns,
            portfolio.cov;
            rm = :edar,
            solvers = portfolio.solvers,
        )[1] - 0.13221264782750258,
    ) < 4e-8
    @test abs(
        calc_risk(
            w,
            portfolio.returns,
            portfolio.cov;
            rm = :rdar,
            solvers = portfolio.solvers,
        )[1] - 0.14475878303476786,
    ) < 4.5e-6
    @test abs(
        calc_risk(
            w,
            portfolio.returns,
            portfolio.cov;
            rm = :mdd_r,
            solvers = portfolio.solvers,
        )[1] - 0.15747952419681518,
    ) < eps()
    @test abs(
        calc_risk(
            w,
            portfolio.returns,
            portfolio.cov;
            rm = :add_r,
            solvers = portfolio.solvers,
        )[1] - 0.0283271101845512,
    ) < eps()
    @test abs(
        calc_risk(
            w,
            portfolio.returns,
            portfolio.cov;
            rm = :dar_r,
            solvers = portfolio.solvers,
        )[1] - 0.09518744803693206,
    ) < eps()
    @test abs(
        calc_risk(
            w,
            portfolio.returns,
            portfolio.cov;
            rm = :cdar_r,
            solvers = portfolio.solvers,
        )[1] - 0.11577944159793968,
    ) < eps()
    @test abs(
        calc_risk(
            w,
            portfolio.returns,
            portfolio.cov;
            rm = :uci_r,
            solvers = portfolio.solvers,
        )[1] - 0.040563874281498415,
    ) < eps()
    @test abs(
        calc_risk(
            w,
            portfolio.returns,
            portfolio.cov;
            rm = :edar_r,
            solvers = portfolio.solvers,
        )[1] - 0.12775945574727807,
    ) < 7.7e-8
    @test abs(
        calc_risk(
            w,
            portfolio.returns,
            portfolio.cov;
            rm = :rdar_r,
            solvers = portfolio.solvers,
        )[1] - 0.13863825698673474,
    ) < 8.3e-6
    @test abs(
        calc_risk(
            w,
            portfolio.returns,
            portfolio.cov;
            rm = :krt,
            solvers = portfolio.solvers,
        )[1] - 0.0002220921162540514,
    ) < eps()
    @test abs(
        calc_risk(
            w,
            portfolio.returns,
            portfolio.cov;
            rm = :skrt,
            solvers = portfolio.solvers,
        )[1] - 0.00017326399202890477,
    ) < eps()
    @test abs(
        calc_risk(w, portfolio.returns, portfolio.cov; rm = :gmd)[1] - 0.010916540360808049,
    ) < eps()
    @test abs(
        calc_risk(
            w,
            portfolio.returns,
            portfolio.cov;
            rm = :rg,
            solvers = portfolio.solvers,
        )[1] - 0.08841083118500939,
    ) < eps()
    @test abs(
        calc_risk(
            w,
            portfolio.returns,
            portfolio.cov;
            rm = :rcvar,
            solvers = portfolio.solvers,
        )[1] - 0.046068669089612116,
    ) < eps()
    @test abs(
        calc_risk(
            w,
            portfolio.returns,
            portfolio.cov;
            rm = :tg,
            solvers = portfolio.solvers,
        )[1] - 0.027380708685309275,
    ) < eps()
    @test abs(
        calc_risk(
            w,
            portfolio.returns,
            portfolio.cov;
            rm = :rtg,
            solvers = portfolio.solvers,
        )[1] - 0.051977750343340984,
    ) < eps()
end
