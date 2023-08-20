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

A = TimeArray(CSV.File("./test/assets/stock_prices.csv"), timestamp = :date)
Y = percentchange(A)
returns = dropmissing!(DataFrame(Y))

@testset "Default parameter tests" begin
    port1 = Portfolio(
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
    asset_statistics!(port1)

    @test abs(calc_risk(w, port1.returns, port1.cov; rm = :mv) - 0.000101665490230637) <
          eps()
    @test abs(calc_risk(w, port1.returns, port1.cov; rm = :msd) - 0.010082930637004155) <
          eps()
    @test abs(calc_risk(w, port1.returns, port1.cov; rm = :mad) - 0.007418863748729646) <
          eps()
    @test abs(calc_risk(w, port1.returns, port1.cov; rm = :msv) - 0.007345533015355076) <
          eps()
    @test abs(calc_risk(w, port1.returns, port1.cov; rm = :flpm) - 0.0034827678064358134) <
          eps()
    @test abs(calc_risk(w, port1.returns, port1.cov; rm = :slpm) - 0.007114744825145661) <
          eps()
    @test abs(calc_risk(w, port1.returns, port1.cov; rm = :wr) - 0.043602428699089285) <
          eps()
    @test abs(calc_risk(w, port1.returns, port1.cov; rm = :var) - 0.016748899891587572) <
          eps()
    @test abs(calc_risk(w, port1.returns, port1.cov; rm = :cvar) - 0.02405795664064266) <
          eps()
    @test abs(
        calc_risk(w, port1.returns, port1.cov; rm = :evar, solvers = port1.solvers) -
        0.030225422932337445,
    ) < 9e-8
    @test abs(
        calc_risk(w, port1.returns, port1.cov; rm = :rvar, solvers = port1.solvers) -
        0.03586321171352101,
    ) < 4.1e-7
    @test abs(
        calc_risk(w, port1.returns, port1.cov; rm = :mdd, solvers = port1.solvers) -
        0.1650381304766847,
    ) < 2.1 * eps()
    @test abs(
        calc_risk(w, port1.returns, port1.cov; rm = :add, solvers = port1.solvers) -
        0.02762516797999026,
    ) < eps()
    @test abs(
        calc_risk(w, port1.returns, port1.cov; rm = :dar, solvers = port1.solvers) -
        0.09442013028621254,
    ) < 4.6 * eps()
    @test abs(
        calc_risk(w, port1.returns, port1.cov; rm = :cdar, solvers = port1.solvers) -
        0.11801077171629008,
    ) < 2 * eps()
    @test abs(
        calc_risk(w, port1.returns, port1.cov; rm = :uci, solvers = port1.solvers) -
        0.0402491262027023,
    ) < eps()
    @test abs(
        calc_risk(w, port1.returns, port1.cov; rm = :edar, solvers = port1.solvers) -
        0.13221264782750258,
    ) < 4e-8
    @test abs(
        calc_risk(w, port1.returns, port1.cov; rm = :rdar, solvers = port1.solvers) -
        0.14475878303476786,
    ) < 4.5e-6
    @test abs(
        calc_risk(w, port1.returns, port1.cov; rm = :mdd_r, solvers = port1.solvers) -
        0.15747952419681518,
    ) < eps()
    @test abs(
        calc_risk(w, port1.returns, port1.cov; rm = :add_r, solvers = port1.solvers) -
        0.0283271101845512,
    ) < eps()
    @test abs(
        calc_risk(w, port1.returns, port1.cov; rm = :dar_r, solvers = port1.solvers) -
        0.09518744803693206,
    ) < eps()
    @test abs(
        calc_risk(w, port1.returns, port1.cov; rm = :cdar_r, solvers = port1.solvers) -
        0.11577944159793968,
    ) < eps()
    @test abs(
        calc_risk(w, port1.returns, port1.cov; rm = :uci_r, solvers = port1.solvers) -
        0.040563874281498415,
    ) < eps()
    @test abs(
        calc_risk(w, port1.returns, port1.cov; rm = :edar_r, solvers = port1.solvers) -
        0.12775945574727807,
    ) < 1.2e-8
    @test abs(
        calc_risk(w, port1.returns, port1.cov; rm = :rdar_r, solvers = port1.solvers) -
        0.13863825698673474,
    ) < 8.3e-6
    @test abs(
        calc_risk(w, port1.returns, port1.cov; rm = :krt, solvers = port1.solvers) -
        0.0002220921162540514,
    ) < eps()
    @test abs(
        calc_risk(w, port1.returns, port1.cov; rm = :skrt, solvers = port1.solvers) -
        0.00017326399202890477,
    ) < eps()
    @test abs(calc_risk(w, port1.returns, port1.cov; rm = :gmd) - 0.010916540360808049) <
          eps()
    @test abs(
        calc_risk(w, port1.returns, port1.cov; rm = :rg, solvers = port1.solvers) -
        0.08841083118500939,
    ) < eps()
    @test abs(
        calc_risk(w, port1.returns, port1.cov; rm = :rcvar, solvers = port1.solvers) -
        0.046068669089612116,
    ) < eps()
    @test abs(
        calc_risk(w, port1.returns, port1.cov; rm = :tg, solvers = port1.solvers) -
        0.027380708685309275,
    ) < eps()
    @test abs(
        calc_risk(w, port1.returns, port1.cov; rm = :rtg, solvers = port1.solvers) -
        0.051977750343340984,
    ) < eps()
end