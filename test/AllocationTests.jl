using CSV, Clarabel, HiGHS, LinearAlgebra, PortfolioOptimiser, Statistics, Test, TimeSeries,
      JuMP

prices = TimeArray(CSV.File("./assets/stock_prices.csv"); timestamp = :date)

rf = 1.0329^(1 / 252) - 1
l = 2.0

@testset "Allocation" begin
    solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                     :check_sol => (allow_local = true, allow_almost = true),
                                     :params => Dict("verbose" => false,
                                                     "max_step_fraction" => 0.75)))
    alloc_solvers = Dict(:HiGHS => Dict(:solver => HiGHS.Optimizer,
                                        :check_sol => (allow_local = true,
                                                       allow_almost = true),
                                        :params => Dict("log_to_console" => false)))
    portfolio = Portfolio(; prices = prices, solvers = solvers,
                          alloc_solvers = alloc_solvers)
    asset_statistics!(portfolio)
    w0 = optimise!(portfolio)

    w1 = allocate!(portfolio; method = LP())
    w2 = allocate!(portfolio; method = Greedy())
    w3 = allocate!(portfolio; method = LP(), investment = 1e4)
    w4 = allocate!(portfolio; method = Greedy(), investment = 1e4)
    w5 = allocate!(portfolio; method = LP(), investment = 1e2)
    w6 = allocate!(portfolio; method = Greedy(), investment = 1e2)
    w7 = allocate!(portfolio; method = LP(), investment = 1e8)
    w8 = allocate!(portfolio; method = Greedy(), investment = 1e8)

    portfolio.short = true
    w9 = optimise!(portfolio)
    w10 = allocate!(portfolio; method = LP())
    w11 = allocate!(portfolio; method = Greedy())
    w12 = allocate!(portfolio; method = LP(), investment = 1e4)
    w13 = allocate!(portfolio; method = Greedy(), investment = 1e4)
    w14 = allocate!(portfolio; method = LP(), investment = 1e2)
    w15 = allocate!(portfolio; method = Greedy(), investment = 1e2)
    w16 = allocate!(portfolio; method = LP(), investment = 1e8)
    w17 = allocate!(portfolio; method = Greedy(), investment = 1e8)
    portfolio.long_u = 1 + portfolio.short_u
    w18 = allocate!(portfolio; method = LP(), investment = 1e6)
    w19 = allocate!(portfolio; method = Greedy(), investment = 1e6)

    @test isapprox(w0.weights, w1.weights, rtol = 0.01)
    @test isapprox(w0.weights, w2.weights, rtol = 0.01)
    @test isapprox(w0.weights, w3.weights, rtol = 0.1)
    @test isapprox(w0.weights, w4.weights, rtol = 0.1)
    @test isapprox(w0.weights, w5.weights, rtol = 1)
    @test isapprox(w0.weights, w6.weights, rtol = 1)
    @test isapprox(w0.weights, w7.weights, rtol = 0.001)
    @test isapprox(w0.weights, w8.weights, rtol = 0.001)
    @test isapprox(w9.weights, w10.weights, rtol = 0.25)
    @test isapprox(w9.weights, w11.weights, rtol = 0.25)
    @test isapprox(w9.weights, w12.weights, rtol = 0.25)
    @test isapprox(w9.weights, w13.weights, rtol = 0.25)
    @test isapprox(w9.weights, w14.weights, rtol = 0.7)
    @test isapprox(w9.weights, w15.weights, rtol = 0.7)
    @test isapprox(w9.weights, w16.weights, rtol = 0.25)
    @test isapprox(w9.weights, w17.weights, rtol = 0.25)
    @test isapprox(portfolio.budget * 1e6, sum(w10.cost[w10.weights .>= 0]), rtol = 1e-5)
    @test isapprox(portfolio.budget * 1e6, sum(w11.cost[w11.weights .>= 0]), rtol = 1e-5)
    @test isapprox(portfolio.budget * 1e4, sum(w12.cost[w12.weights .>= 0]), rtol = 1e-3)
    @test isapprox(portfolio.budget * 1e4, sum(w13.cost[w13.weights .>= 0]), rtol = 1e-2)
    @test isapprox(portfolio.budget * 1e2, sum(w14.cost[w14.weights .>= 0]), rtol = 1e-1)
    @test isapprox(portfolio.budget * 1e2, sum(w15.cost[w15.weights .>= 0]), rtol = 1e-2)
    @test isapprox(portfolio.budget * 1e8, sum(w16.cost[w16.weights .>= 0]), rtol = 5e-7)
    @test isapprox(portfolio.budget * 1e8, sum(w17.cost[w17.weights .>= 0]), rtol = 5e-7)
    @test isapprox(sum(w18.cost[w18.cost .> 0]),
                   sum(w10.cost[w10.cost .> 0]) - sum(w10.cost[w10.cost .< 0]), rtol = 3e-4)
    @test isapprox(sum(w19.cost[w19.cost .> 0]),
                   sum(w11.cost[w11.cost .> 0]) - sum(w11.cost[w11.cost .< 0]), rtol = 0.25)

    portfolio.alloc_solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                     :params => Dict("verbose" => false,
                                                                     "max_step_fraction" => 0.75)))
    w20 = allocate!(portfolio; method = LP(), investment = 1e6)
    @test !is_solved_and_feasible(portfolio.alloc_model)

    w21 = allocate!(portfolio; method = Greedy(), investment = 69)
end
