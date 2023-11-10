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
    StatsBase,
    Logging

Logging.disable_logging(Logging.Warn)

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
        calc_risk(w, portfolio.returns; cov = portfolio.cov, rm = :Variance) -
        0.000101665490230637,
    ) < eps()
    @test abs(
        calc_risk(w, portfolio.returns; cov = portfolio.cov, rm = :SD) -
        0.010082930637004155,
    ) < eps()
    @test abs(
        calc_risk(w, portfolio.returns; cov = portfolio.cov, rm = :MAD) -
        0.007418863748729646,
    ) < eps()
    @test abs(
        calc_risk(w, portfolio.returns; cov = portfolio.cov, rm = :SSD) -
        0.007345533015355076,
    ) < eps()
    @test abs(
        calc_risk(w, portfolio.returns; cov = portfolio.cov, rm = :FLPM) -
        0.0034827678064358134,
    ) < eps()
    @test abs(
        calc_risk(w, portfolio.returns; cov = portfolio.cov, rm = :SLPM) -
        0.007114744825145661,
    ) < eps()
    @test abs(
        calc_risk(w, portfolio.returns; cov = portfolio.cov, rm = :WR) -
        0.043602428699089285,
    ) < eps()
    @test abs(
        calc_risk(w, portfolio.returns; cov = portfolio.cov, rm = :VaR) -
        0.016748899891587572,
    ) < eps()
    @test abs(
        calc_risk(w, portfolio.returns; cov = portfolio.cov, rm = :CVaR) -
        0.02405795664064266,
    ) < eps()
    @test abs(
        calc_risk(
            w,
            portfolio.returns;
            cov = portfolio.cov,
            rm = :EVaR,
            solvers = portfolio.solvers,
        ) - 0.030225422932337445,
    ) < 9e-8
    @test abs(
        calc_risk(
            w,
            portfolio.returns;
            cov = portfolio.cov,
            rm = :RVaR,
            solvers = portfolio.solvers,
        ) - 0.03586321171352101,
    ) < 1e-5
    @test abs(
        calc_risk(
            w,
            portfolio.returns;
            cov = portfolio.cov,
            rm = :MDD,
            solvers = portfolio.solvers,
        ) - 0.1650381304766847,
    ) < 2.1 * eps()
    @test abs(
        calc_risk(
            w,
            portfolio.returns;
            cov = portfolio.cov,
            rm = :ADD,
            solvers = portfolio.solvers,
        ) - 0.02762516797999026,
    ) < eps()
    @test abs(
        calc_risk(
            w,
            portfolio.returns;
            cov = portfolio.cov,
            rm = :DaR,
            solvers = portfolio.solvers,
        ) - 0.09442013028621254,
    ) < 4.6 * eps()
    @test abs(
        calc_risk(
            w,
            portfolio.returns;
            cov = portfolio.cov,
            rm = :CDaR,
            solvers = portfolio.solvers,
        ) - 0.11801077171629008,
    ) < 2 * eps()
    @test abs(
        calc_risk(
            w,
            portfolio.returns;
            cov = portfolio.cov,
            rm = :UCI,
            solvers = portfolio.solvers,
        ) - 0.0402491262027023,
    ) < eps()
    @test abs(
        calc_risk(
            w,
            portfolio.returns;
            cov = portfolio.cov,
            rm = :EDaR,
            solvers = portfolio.solvers,
        ) - 0.13221264782750258,
    ) < 4e-8

    @test abs(
        calc_risk(
            w,
            portfolio.returns;
            cov = portfolio.cov,
            rm = :RDaR,
            solvers = portfolio.solvers,
        ) - 0.14476333638845212,
    ) < 4.6e-6

    @test abs(
        calc_risk(
            w,
            portfolio.returns;
            cov = portfolio.cov,
            rm = :MDD_r,
            solvers = portfolio.solvers,
        ) - 0.15747952419681518,
    ) < eps()
    @test abs(
        calc_risk(
            w,
            portfolio.returns;
            cov = portfolio.cov,
            rm = :ADD_r,
            solvers = portfolio.solvers,
        ) - 0.0283271101845512,
    ) < eps()
    @test abs(
        calc_risk(
            w,
            portfolio.returns;
            cov = portfolio.cov,
            rm = :DaR_r,
            solvers = portfolio.solvers,
        ) - 0.09518744803693206,
    ) < eps()
    @test abs(
        calc_risk(
            w,
            portfolio.returns;
            cov = portfolio.cov,
            rm = :CDaR_r,
            solvers = portfolio.solvers,
        ) - 0.11577944159793968,
    ) < eps()
    @test abs(
        calc_risk(
            w,
            portfolio.returns;
            cov = portfolio.cov,
            rm = :uci_r,
            solvers = portfolio.solvers,
        ) - 0.040563874281498415,
    ) < eps()
    @test abs(
        calc_risk(
            w,
            portfolio.returns;
            cov = portfolio.cov,
            rm = :EDaR_r,
            solvers = portfolio.solvers,
        ) - 0.12775945574727807,
    ) < 7.7e-8
    @test abs(
        calc_risk(
            w,
            portfolio.returns;
            cov = portfolio.cov,
            rm = :RDaR_r,
            solvers = portfolio.solvers,
        ) - 0.13863825698673474,
    ) < 8.3e-6
    @test abs(
        calc_risk(
            w,
            portfolio.returns;
            cov = portfolio.cov,
            rm = :Kurt,
            solvers = portfolio.solvers,
        ) - 0.0002220921162540514,
    ) < eps()
    @test abs(
        calc_risk(
            w,
            portfolio.returns;
            cov = portfolio.cov,
            rm = :SKurt,
            solvers = portfolio.solvers,
        ) - 0.00017326399202890477,
    ) < eps()
    @test abs(
        calc_risk(w, portfolio.returns; cov = portfolio.cov, rm = :GMD) -
        0.010916540360808049,
    ) < eps()
    @test abs(
        calc_risk(
            w,
            portfolio.returns;
            cov = portfolio.cov,
            rm = :RG,
            solvers = portfolio.solvers,
        ) - 0.08841083118500939,
    ) < eps()
    @test abs(
        calc_risk(
            w,
            portfolio.returns;
            cov = portfolio.cov,
            rm = :RCVaR,
            solvers = portfolio.solvers,
        ) - 0.046068669089612116,
    ) < eps()
    @test abs(
        calc_risk(
            w,
            portfolio.returns;
            cov = portfolio.cov,
            rm = :TG,
            solvers = portfolio.solvers,
        ) - 0.027380708685309275,
    ) < eps()
    @test abs(
        calc_risk(
            w,
            portfolio.returns;
            cov = portfolio.cov,
            rm = :RTG,
            solvers = portfolio.solvers,
        ) - 0.051977750343340984,
    ) < eps()
end
