using CSV, TimeSeries, DataFrames, StatsBase, Statistics, LinearAlgebra, Test, Clarabel,
      JuMP, PortfolioOptimiser

path = joinpath(@__DIR__, "assets/stock_prices.csv")
prices = TimeArray(CSV.File(path); timestamp = :date)
rf = 1.0329^(1 / 252) - 1
l = 2.0

@testset "Objective scalarisation Trad" begin
    portfolio = Portfolio(; prices = prices,
                          solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                           :check_sol => (allow_local = true,
                                                                          allow_almost = true),
                                                           :params => Dict("verbose" => false,
                                                                           "max_step_fraction" => 0.75))))

    asset_statistics!(portfolio)

    w1 = optimise!(portfolio,
                   Trad(; obj = MinRisk(), rm = SD(), scalarisation = ScalarSum()))
    w2 = optimise!(portfolio,
                   Trad(; obj = MinRisk(), rm = SD(), scalarisation = ScalarMax()))
    w3 = optimise!(portfolio,
                   Trad(; obj = MinRisk(), rm = SD(),
                        scalarisation = ScalarLogSumExp(; gamma = 1e-3)))
    w4 = optimise!(portfolio,
                   Trad(; obj = MinRisk(), rm = SD(),
                        scalarisation = ScalarLogSumExp(; gamma = 1e0)))
    w5 = optimise!(portfolio,
                   Trad(; obj = MinRisk(), rm = SD(),
                        scalarisation = ScalarLogSumExp(; gamma = 1e3)))

    @test isapprox(w1.weights, w2.weights, rtol = 5.0e-7)
    @test isapprox(w1.weights, w3.weights, rtol = 0.25)
    @test isapprox(w1.weights, w4.weights, rtol = 1.0e-5)
    @test isapprox(w1.weights, w5.weights, rtol = 1.0e-5)
    @test isapprox(w2.weights, w3.weights, rtol = 0.25)
    @test isapprox(w2.weights, w4.weights, rtol = 1.0e-5)
    @test isapprox(w2.weights, w5.weights, rtol = 1.0e-5)

    w6 = optimise!(portfolio,
                   Trad(; obj = MinRisk(), rm = SVariance(), scalarisation = ScalarSum()))
    w7 = optimise!(portfolio,
                   Trad(; obj = MinRisk(), rm = SVariance(), scalarisation = ScalarMax()))
    w8 = optimise!(portfolio,
                   Trad(; obj = MinRisk(), rm = SVariance(),
                        scalarisation = ScalarLogSumExp(; gamma = 1e-3)))
    w9 = optimise!(portfolio,
                   Trad(; obj = MinRisk(), rm = SVariance(),
                        scalarisation = ScalarLogSumExp(; gamma = 1e0)))
    w10 = optimise!(portfolio,
                    Trad(; obj = MinRisk(), rm = SVariance(),
                         scalarisation = ScalarLogSumExp(; gamma = 1e3)))

    @test isapprox(w6.weights, w7.weights, rtol = 5.0e-5)
    @test isapprox(w6.weights, w8.weights, rtol = 0.5)
    @test isapprox(w6.weights, w9.weights, rtol = 1.0e-4)
    @test isapprox(w6.weights, w10.weights, rtol = 1.0e-4)
    @test isapprox(w7.weights, w8.weights, rtol = 0.5)
    @test isapprox(w7.weights, w9.weights, rtol = 5.0e-5)
    @test isapprox(w7.weights, w10.weights, rtol = 5.0e-5)

    w11 = optimise!(portfolio,
                    Trad(; obj = MinRisk(),
                         rm = [SD(), SVariance(; settings = RMSettings(; scale = 50))],
                         scalarisation = ScalarSum()))
    w12 = optimise!(portfolio,
                    Trad(; obj = MinRisk(),
                         rm = [SD(), SVariance(; settings = RMSettings(; scale = 50))],
                         scalarisation = ScalarMax()))
    w13 = optimise!(portfolio,
                    Trad(; obj = MinRisk(),
                         rm = [SD(), SVariance(; settings = RMSettings(; scale = 50))],
                         scalarisation = ScalarLogSumExp(; gamma = 5e-1)))
    w14 = optimise!(portfolio,
                    Trad(; obj = MinRisk(),
                         rm = [SD(), SVariance(; settings = RMSettings(; scale = 50))],
                         scalarisation = ScalarLogSumExp(; gamma = 1e0)))
    w15 = optimise!(portfolio,
                    Trad(; obj = MinRisk(),
                         rm = [SD(), SVariance(; settings = RMSettings(; scale = 50))],
                         scalarisation = ScalarLogSumExp(; gamma = 1e3)))

    @test isapprox(w11.weights, w12.weights, rtol = 0.05)
    @test isapprox(w11.weights, w13.weights, rtol = 1.0e-4)
    @test isapprox(w11.weights, w14.weights, rtol = 5.0e-4)
    @test isapprox(w11.weights, w15.weights, rtol = 0.05)
    @test isapprox(w12.weights, w1.weights, rtol = 5.0e-6)
    @test isapprox(w12.weights, w13.weights, rtol = 0.05)
    @test isapprox(w12.weights, w14.weights, rtol = 0.05)
    @test isapprox(w12.weights, w15.weights, rtol = 5.0e-4)
end

@testset "Objective scalarisation NOC" begin
    portfolio = Portfolio(; prices = prices,
                          solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                           :check_sol => (allow_local = true,
                                                                          allow_almost = true),
                                                           :params => Dict("verbose" => false,
                                                                           "max_step_fraction" => 0.75))))

    asset_statistics!(portfolio)

    w1 = optimise!(portfolio,
                   NOC(; obj = MinRisk(), rm = SD(), scalarisation = ScalarSum()))
    w2 = optimise!(portfolio,
                   NOC(; obj = MinRisk(), rm = SD(), scalarisation = ScalarMax()))
    w3 = optimise!(portfolio,
                   NOC(; obj = MinRisk(), rm = SD(),
                       scalarisation = ScalarLogSumExp(; gamma = 1e0)))
    w4 = optimise!(portfolio,
                   NOC(; obj = MinRisk(), rm = SD(),
                       scalarisation = ScalarLogSumExp(; gamma = 1e3)))

    @test isapprox(w1.weights, w2.weights, rtol = 5.0e-4)
    @test isapprox(w1.weights, w3.weights, rtol = 5.0e-4)
    @test isapprox(w1.weights, w4.weights, rtol = 0.5)
    @test isapprox(w2.weights, w3.weights, rtol = 5.0e-4)
    @test isapprox(w2.weights, w4.weights, rtol = 0.5)

    w5 = optimise!(portfolio,
                   NOC(; obj = MinRisk(), rm = FLPM(), scalarisation = ScalarSum()))
    w6 = optimise!(portfolio,
                   NOC(; obj = MinRisk(), rm = FLPM(), scalarisation = ScalarMax()))
    w7 = optimise!(portfolio,
                   NOC(; obj = MinRisk(), rm = FLPM(),
                       scalarisation = ScalarLogSumExp(; gamma = 1e0)))
    w8 = optimise!(portfolio,
                   NOC(; obj = MinRisk(), rm = FLPM(),
                       scalarisation = ScalarLogSumExp(; gamma = 1e3)))

    @test isapprox(w5.weights, w6.weights, rtol = 5.0e-4)
    @test isapprox(w5.weights, w7.weights, rtol = 5.0e-4)
    @test isapprox(w5.weights, w8.weights, rtol = 0.5)
    @test isapprox(w6.weights, w7.weights, rtol = 5.0e-4)
    @test isapprox(w6.weights, w8.weights, rtol = 0.5)

    w11 = optimise!(portfolio,
                    NOC(; obj = MinRisk(), rm = [SD(), FLPM()],
                        scalarisation = ScalarSum()))
    w12 = optimise!(portfolio,
                    NOC(; obj = MinRisk(), rm = [SD(), FLPM()],
                        scalarisation = ScalarMax()))
    w13 = optimise!(portfolio,
                    NOC(; obj = MinRisk(), rm = [SD(), FLPM()],
                        scalarisation = ScalarLogSumExp(; gamma = 1e0)))
    w14 = optimise!(portfolio,
                    NOC(; obj = MinRisk(), rm = [SD(), FLPM()],
                        scalarisation = ScalarLogSumExp(; gamma = 1e3)))

    @test isapprox(w11.weights, w12.weights, rtol = 0.05)
    @test isapprox(w11.weights, w13.weights, rtol = 5.0e-3)
    @test isapprox(w11.weights, w14.weights, rtol = 0.5)
    @test isapprox(w12.weights, w1.weights, rtol = 5.0e-4)
    @test isapprox(w12.weights, w13.weights, rtol = 0.05)
    @test isapprox(w12.weights, w14.weights, rtol = 0.5)
end
