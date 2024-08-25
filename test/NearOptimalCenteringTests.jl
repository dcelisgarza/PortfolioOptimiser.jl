using CSV, TimeSeries, DataFrames, StatsBase, Statistics, LinearAlgebra, Test, Clarabel,
      HiGHS, PortfolioOptimiser

prices = TimeArray(CSV.File("./assets/stock_prices.csv"); timestamp = :date)
rf = 1.0329^(1 / 252) - 1
l = 2.0

@testset "NOC" begin
    portfolio = Portfolio2(; prices = prices,
                           solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                            :params => Dict("verbose" => false,
                                                                            "max_step_fraction" => 0.75))))
    asset_statistics2!(portfolio)
    rm = SD2(; formulation = SimpleSD())

    obj = SR(; rf = rf)

    kelly = NoKelly()
    w1 = optimise2!(portfolio; rm = rm, kelly = kelly, obj = MinRisk()).weights
    w2 = optimise2!(portfolio; rm = rm, kelly = kelly, obj = MaxRet()).weights
    w3 = optimise2!(portfolio; rm = rm, kelly = kelly, obj = obj).weights

    w4 = optimise2!(portfolio; type = NOC(; w_min = w1, w_max = w2, w_opt = w3), rm = rm,
                    kelly = kelly, obj = obj)
    w5 = optimise2!(portfolio; type = NOC(;), rm = rm, kelly = kelly, obj = obj)
    @test isapprox(w5.weights, w4.weights)

    w6 = optimise2!(portfolio; type = NOC(; w_min = w1, w_max = w2, w_opt = w3, bins = 0.5),
                    rm = rm, kelly = kelly, obj = obj)
    w7 = optimise2!(portfolio; type = NOC(; w_min = w1, w_max = w2, w_opt = w3, bins = 1),
                    rm = rm, kelly = kelly, obj = obj)
    w8 = optimise2!(portfolio; type = NOC(; w_min = w1, w_max = w2, w_opt = w3, bins = 4),
                    rm = rm, kelly = kelly, obj = obj)
    w9 = optimise2!(portfolio; type = NOC(; w_min = w1, w_max = w2, w_opt = w3, bins = 32),
                    rm = rm, kelly = kelly, obj = obj)
    w10 = optimise2!(portfolio;
                     type = NOC(; w_min = w1, w_max = w2, w_opt = w3, bins = 512), rm = rm,
                     kelly = kelly, obj = obj)
    w11 = optimise2!(portfolio;
                     type = NOC(; w_min = w1, w_max = w2, w_opt = w3, bins = 16384),
                     rm = rm, kelly = kelly, obj = obj)
    w12 = optimise2!(portfolio;
                     type = NOC(; w_min = w1, w_max = w2, w_opt = w3, bins = 1.048576e6),
                     rm = rm, kelly = kelly, obj = obj)

    @test rmsd(w3, w6.weights) >
          rmsd(w3, w7.weights) >
          rmsd(w3, w8.weights) >
          rmsd(w3, w4.weights) >
          rmsd(w3, w9.weights) >
          rmsd(w3, w10.weights) >
          rmsd(w3, w11.weights) >
          rmsd(w3, w12.weights)

    @test isapprox(w3, w6.weights, rtol = 1)
    @test isapprox(w3, w7.weights, rtol = 1)
    @test isapprox(w3, w8.weights, rtol = 1)
    @test isapprox(w3, w9.weights, rtol = 1)
    @test isapprox(w3, w10.weights, rtol = 0.1)
    @test isapprox(w3, w11.weights, rtol = 0.005)
    @test isapprox(w3, w12.weights, rtol = 5.0e-5)

    kelly = EKelly()
    w13 = optimise2!(portfolio; rm = rm, kelly = kelly, obj = MinRisk()).weights
    w14 = optimise2!(portfolio; rm = rm, kelly = kelly, obj = MaxRet()).weights
    w15 = optimise2!(portfolio; rm = rm, kelly = kelly, obj = obj).weights

    w16 = optimise2!(portfolio; type = NOC(; w_min = w13, w_max = w14, w_opt = w15),
                     rm = rm, kelly = kelly, obj = obj)
    w17 = optimise2!(portfolio; type = NOC(;), rm = rm, kelly = kelly, obj = obj)
    @test isapprox(w17.weights, w16.weights)

    w18 = optimise2!(portfolio;
                     type = NOC(; w_min = w13, w_max = w14, w_opt = w15, bins = 0.5),
                     rm = rm, kelly = kelly, obj = obj)
    w19 = optimise2!(portfolio;
                     type = NOC(; w_min = w13, w_max = w14, w_opt = w15, bins = 1), rm = rm,
                     kelly = kelly, obj = obj)
    w20 = optimise2!(portfolio;
                     type = NOC(; w_min = w13, w_max = w14, w_opt = w15, bins = 4), rm = rm,
                     kelly = kelly, obj = obj)
    w21 = optimise2!(portfolio;
                     type = NOC(; w_min = w13, w_max = w14, w_opt = w15, bins = 32),
                     rm = rm, kelly = kelly, obj = obj)
    w22 = optimise2!(portfolio;
                     type = NOC(; w_min = w13, w_max = w14, w_opt = w15, bins = 512),
                     rm = rm, kelly = kelly, obj = obj)
    w23 = optimise2!(portfolio;
                     type = NOC(; w_min = w13, w_max = w14, w_opt = w15, bins = 16384),
                     rm = rm, kelly = kelly, obj = obj)
    w24 = optimise2!(portfolio;
                     type = NOC(; w_min = w13, w_max = w14, w_opt = w15, bins = 1.048576e6),
                     rm = rm, kelly = kelly, obj = obj)

    @test rmsd(w15, w18.weights) >
          rmsd(w15, w19.weights) >
          rmsd(w15, w20.weights) >
          rmsd(w15, w16.weights) >
          rmsd(w15, w21.weights) >
          rmsd(w15, w22.weights) >
          rmsd(w15, w23.weights) >
          rmsd(w15, w24.weights)

    @test isapprox(w15, w18.weights, rtol = 1)
    @test isapprox(w15, w19.weights, rtol = 1)
    @test isapprox(w15, w20.weights, rtol = 0.5)
    @test isapprox(w15, w21.weights, rtol = 0.5)
    @test isapprox(w15, w22.weights, rtol = 0.05)
    @test isapprox(w15, w23.weights, rtol = 0.005)
    @test isapprox(w15, w24.weights, rtol = 0.0005)
end

@testset "NOC scale and vec" begin
    portfolio = Portfolio2(; prices = prices,
                           solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                            :params => Dict("verbose" => false,
                                                                            "max_step_fraction" => 0.75))))
    asset_statistics2!(portfolio)

    kelly = NoKelly()
    obj = SR(; rf = rf)

    rm = SD2(; formulation = SimpleSD())
    w1 = optimise2!(portfolio; type = NOC(), rm = rm, kelly = kelly, obj = obj)

    rm = SD2(; formulation = SimpleSD(), settings = RiskMeasureSettings(; scale = 5))
    w2 = optimise2!(portfolio; type = NOC(), rm = rm, kelly = kelly, obj = obj)
    @test isapprox(w2.weights, w1.weights, rtol = 5.0e-5)

    rm = [SD2(; formulation = SimpleSD())]
    w3 = optimise2!(portfolio; type = NOC(), rm = rm, kelly = kelly, obj = obj)
    @test isapprox(w3.weights, w1.weights)
    @test isapprox(w3.weights, w2.weights, rtol = 5.0e-5)

    rm = [[SD2(; formulation = SimpleSD())]]
    w4 = optimise2!(portfolio; type = NOC(), rm = rm, kelly = kelly, obj = obj)
    @test isapprox(w4.weights, w1.weights, rtol = 5.0e-5)
    @test isapprox(w4.weights, w2.weights, rtol = 5.0e-5)
    @test isapprox(w4.weights, w3.weights, rtol = 5.0e-5)

    rm = [[SD2(; formulation = SimpleSD()), SD2(; formulation = SimpleSD()),
           SD2(; formulation = SimpleSD(), settings = RiskMeasureSettings(; scale = 0.75)),
           SD2(; formulation = SimpleSD(), settings = RiskMeasureSettings(; scale = 2.25))]]
    w5 = optimise2!(portfolio; type = NOC(), rm = rm, kelly = kelly, obj = obj)
    @test isapprox(w5.weights, w1.weights, rtol = 5.0e-5)
    @test isapprox(w5.weights, w2.weights, rtol = 5.0e-5)
    @test isapprox(w5.weights, w3.weights, rtol = 5.0e-5)
    @test isapprox(w5.weights, w4.weights, rtol = 5.0e-5)
end

@testset "NOC vec convergence" begin
    portfolio = Portfolio2(; prices = prices,
                           solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                            :params => Dict("verbose" => false,
                                                                            "max_step_fraction" => 0.75))))
    asset_statistics2!(portfolio)
    rm = [SD2(; formulation = SimpleSD()), CVaR2()]

    obj = SR(; rf = rf)

    kelly = NoKelly()
    w1 = optimise2!(portfolio; rm = rm, kelly = kelly, obj = MinRisk()).weights
    w2 = optimise2!(portfolio; rm = rm, kelly = kelly, obj = MaxRet()).weights
    w3 = optimise2!(portfolio; rm = rm, kelly = kelly, obj = obj).weights

    w4 = optimise2!(portfolio; type = NOC(; w_min = w1, w_max = w2, w_opt = w3), rm = rm,
                    kelly = kelly, obj = obj)
    w5 = optimise2!(portfolio; type = NOC(;), rm = rm, kelly = kelly, obj = obj)
    @test isapprox(w5.weights, w4.weights)

    w6 = optimise2!(portfolio; type = NOC(; w_min = w1, w_max = w2, w_opt = w3, bins = 0.5),
                    rm = rm, kelly = kelly, obj = obj)
    w7 = optimise2!(portfolio; type = NOC(; w_min = w1, w_max = w2, w_opt = w3, bins = 1),
                    rm = rm, kelly = kelly, obj = obj)
    w8 = optimise2!(portfolio; type = NOC(; w_min = w1, w_max = w2, w_opt = w3, bins = 4),
                    rm = rm, kelly = kelly, obj = obj)
    w9 = optimise2!(portfolio; type = NOC(; w_min = w1, w_max = w2, w_opt = w3, bins = 32),
                    rm = rm, kelly = kelly, obj = obj)
    w10 = optimise2!(portfolio;
                     type = NOC(; w_min = w1, w_max = w2, w_opt = w3, bins = 512), rm = rm,
                     kelly = kelly, obj = obj)
    w11 = optimise2!(portfolio;
                     type = NOC(; w_min = w1, w_max = w2, w_opt = w3, bins = 16384),
                     rm = rm, kelly = kelly, obj = obj)
    w12 = optimise2!(portfolio;
                     type = NOC(; w_min = w1, w_max = w2, w_opt = w3, bins = 1.048576e6),
                     rm = rm, kelly = kelly, obj = obj)

    @test rmsd(w3, w6.weights) >
          rmsd(w3, w7.weights) >
          rmsd(w3, w8.weights) >
          rmsd(w3, w4.weights) >
          rmsd(w3, w9.weights) >
          rmsd(w3, w10.weights) >
          rmsd(w3, w11.weights) >
          rmsd(w3, w12.weights)

    @test isapprox(w3, w6.weights, rtol = 1)
    @test isapprox(w3, w7.weights, rtol = 1)
    @test isapprox(w3, w8.weights, rtol = 1)
    @test isapprox(w3, w9.weights, rtol = 0.5)
    @test isapprox(w3, w10.weights, rtol = 0.05)
    @test isapprox(w3, w11.weights, rtol = 0.005)
    @test isapprox(w3, w12.weights, rtol = 0.001)

    rm = [SD2(; formulation = SimpleSD(), settings = RiskMeasureSettings(; scale = 7.3)),
          [CVaR2(; alpha = 0.1, settings = RiskMeasureSettings(; scale = 5)),
           CVaR2(; settings = RiskMeasureSettings(; scale = 1.6))]]
    w13 = optimise2!(portfolio; rm = rm, kelly = kelly, obj = MinRisk()).weights
    w14 = optimise2!(portfolio; rm = rm, kelly = kelly, obj = MaxRet()).weights
    w15 = optimise2!(portfolio; rm = rm, kelly = kelly, obj = obj).weights

    w16 = optimise2!(portfolio; type = NOC(; w_min = w13, w_max = w14, w_opt = w15),
                     rm = rm, kelly = kelly, obj = obj)
    w17 = optimise2!(portfolio; type = NOC(;), rm = rm, kelly = kelly, obj = obj)
    @test isapprox(w17.weights, w16.weights)

    w18 = optimise2!(portfolio;
                     type = NOC(; w_min = w13, w_max = w14, w_opt = w15, bins = 0.5),
                     rm = rm, kelly = kelly, obj = obj)
    w19 = optimise2!(portfolio;
                     type = NOC(; w_min = w13, w_max = w14, w_opt = w15, bins = 1), rm = rm,
                     kelly = kelly, obj = obj)
    w20 = optimise2!(portfolio;
                     type = NOC(; w_min = w13, w_max = w14, w_opt = w15, bins = 4), rm = rm,
                     kelly = kelly, obj = obj)
    w21 = optimise2!(portfolio;
                     type = NOC(; w_min = w13, w_max = w14, w_opt = w15, bins = 32),
                     rm = rm, kelly = kelly, obj = obj)
    w22 = optimise2!(portfolio;
                     type = NOC(; w_min = w13, w_max = w14, w_opt = w15, bins = 512),
                     rm = rm, kelly = kelly, obj = obj)
    w23 = optimise2!(portfolio;
                     type = NOC(; w_min = w13, w_max = w14, w_opt = w15, bins = 16384),
                     rm = rm, kelly = kelly, obj = obj)
    w24 = optimise2!(portfolio;
                     type = NOC(; w_min = w13, w_max = w14, w_opt = w15, bins = 1.048576e6),
                     rm = rm, kelly = kelly, obj = obj)

    @test rmsd(w15, w18.weights) >
          rmsd(w15, w19.weights) >
          rmsd(w15, w20.weights) >
          rmsd(w15, w16.weights) >
          rmsd(w15, w21.weights) >
          rmsd(w15, w22.weights) >
          rmsd(w15, w23.weights) >
          rmsd(w15, w24.weights)

    @test isapprox(w15, w18.weights, rtol = 1)
    @test isapprox(w15, w19.weights, rtol = 1)
    @test isapprox(w15, w20.weights, rtol = 1)
    @test isapprox(w15, w21.weights, rtol = 0.5)
    @test isapprox(w15, w22.weights, rtol = 0.05)
    @test isapprox(w15, w23.weights, rtol = 0.005)
    @test isapprox(w15, w24.weights, rtol = 5.0e-5)
end
