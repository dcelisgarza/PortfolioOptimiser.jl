using CSV, Clarabel, HiGHS, LinearAlgebra, PortfolioOptimiser, Statistics, Test, TimeSeries,
      JuMP

path = joinpath(@__DIR__, "assets/stock_prices.csv")
prices = TimeArray(CSV.File(path); timestamp = :date)
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
    w0 = optimise!(portfolio, Trad())

    w1 = allocate!(portfolio; method = LP())
    w2 = allocate!(portfolio; method = Greedy())
    w3 = allocate!(portfolio; method = LP(), investment = 1e4)
    w4 = allocate!(portfolio; method = Greedy(), investment = 1e4)
    w5 = allocate!(portfolio; method = LP(), investment = 1e2)
    w6 = allocate!(portfolio; method = Greedy(), investment = 1e2)
    w7 = allocate!(portfolio; method = LP(), investment = 1e8)
    w8 = allocate!(portfolio; method = Greedy(), investment = 1e8)

    portfolio.short = true
    w9 = optimise!(portfolio, Trad())
    w10 = allocate!(portfolio; method = LP())
    w11 = allocate!(portfolio; method = Greedy())
    w12 = allocate!(portfolio; method = LP(), investment = 1e4)
    w13 = allocate!(portfolio; method = Greedy(), investment = 1e4)
    w14 = allocate!(portfolio; method = LP(), investment = 1e2)
    w15 = allocate!(portfolio; method = Greedy(), investment = 1e2)
    w16 = allocate!(portfolio; method = LP(), investment = 1e8)
    w17 = allocate!(portfolio; method = Greedy(), investment = 1e8)
    portfolio.budget = 1
    w9_2 = optimise!(portfolio, Trad())
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
    @test isapprox(sum(w18.cost), 1e6 * sum(w9_2.weights), rtol = 5.0e-5)
    @test isapprox(sum(w18.cost[w18.cost .>= 0]),
                   1e6 * sum(w9_2.weights[w9_2.weights .>= 0]), rtol = 1e-5)
    @test isapprox(sum(w18.cost[w18.cost .< 0]), 1e6 * sum(w9_2.weights[w9_2.weights .< 0]),
                   rtol = 0.0005)
    @test isapprox(sum(w18.cost), 1e6 * portfolio.budget, rtol = 5.0e-5)
    @test sum(w18.cost[w18.cost .< 0]) >= 1e6 * portfolio.short_budget

    @test isapprox(sum(w19.cost), 1e6 * sum(w9_2.weights), rtol = 5.0e-5)
    @test isapprox(sum(w19.cost[w19.cost .>= 0]),
                   1e6 * sum(w9_2.weights[w9_2.weights .>= 0]), rtol = 1e-5)
    @test isapprox(sum(w19.cost[w19.cost .< 0]), 1e6 * sum(w9_2.weights[w9_2.weights .< 0]),
                   rtol = 0.0005)
    @test isapprox(sum(w19.cost), 1e6 * portfolio.budget, rtol = 5.0e-5)
    @test sum(w19.cost[w19.cost .< 0]) >= 1e6 * portfolio.short_budget

    portfolio.alloc_solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                     :params => Dict("verbose" => false,
                                                                     "max_step_fraction" => 0.75)))
    w20 = allocate!(portfolio; method = LP(), investment = 1e6)
    @test !is_solved_and_feasible(portfolio.alloc_model)

    w21 = allocate!(portfolio; method = Greedy(), investment = 69)
end
