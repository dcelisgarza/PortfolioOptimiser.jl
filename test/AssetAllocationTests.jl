using CSV, Clarabel, HiGHS, LinearAlgebra, OrderedCollections, PortfolioOptimiser,
      Statistics, Test, TimeSeries, Logging

Logging.disable_logging(Logging.Warn)

prices = TimeArray(CSV.File("./assets/stock_prices.csv"); timestamp = :date)

rf = 1.0329^(1 / 252) - 1
l = 2.0

@testset "Asset allocation" begin
    solvers = OrderedDict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                            :params => Dict("verbose" => false,
                                                            "max_step_fraction" => 0.75)))
    alloc_solvers = Dict(:HiGHS => Dict(:solver => HiGHS.Optimizer,
                                        :params => Dict("log_to_console" => false)))
    portfolio = Portfolio(; prices = prices, solvers = solvers,
                          alloc_solvers = alloc_solvers)
    asset_statistics!(portfolio; calc_kurt = false)
    w0 = optimise!(portfolio, OptimiseOpt(; obj = :Min_Risk))

    alloc_opt = AllocOpt(; method = :LP)

    allocate!(portfolio, alloc_opt; save_opt_params = false)
    alloc_params1 = copy(portfolio.alloc_params)
    alloc_opt.method = :Greedy
    allocate!(portfolio, alloc_opt; save_opt_params = false)
    alloc_params2 = copy(portfolio.alloc_params)

    alloc_opt.method = :LP
    w1 = allocate!(portfolio, alloc_opt)
    alloc_opt.method = :Greedy
    w2 = allocate!(portfolio, alloc_opt)

    alloc_opt.method = :LP
    alloc_opt.investment = 1e4
    w3 = allocate!(portfolio, alloc_opt)
    alloc_opt.method = :Greedy
    w4 = allocate!(portfolio, alloc_opt)

    alloc_opt.method = :LP
    alloc_opt.investment = 1e2
    w5 = allocate!(portfolio, alloc_opt)
    alloc_opt.method = :Greedy
    w6 = allocate!(portfolio, alloc_opt)

    alloc_opt.method = :LP
    alloc_opt.investment = 1e8
    w7 = allocate!(portfolio, alloc_opt)
    alloc_opt.method = :Greedy
    w8 = allocate!(portfolio, alloc_opt)

    portfolio.short = true
    alloc_opt = AllocOpt(; method = :LP)
    w9 = optimise!(portfolio, OptimiseOpt(; obj = :Min_Risk))
    alloc_opt.method = :LP
    w10 = allocate!(portfolio, alloc_opt)
    alloc_opt.method = :Greedy
    w11 = allocate!(portfolio, alloc_opt)

    alloc_opt.method = :LP
    alloc_opt.investment = 1e4
    w12 = allocate!(portfolio, alloc_opt)
    alloc_opt.method = :Greedy
    w13 = allocate!(portfolio, alloc_opt)

    alloc_opt.method = :LP
    alloc_opt.investment = 1e2
    w14 = allocate!(portfolio, alloc_opt)
    alloc_opt.method = :Greedy
    w15 = allocate!(portfolio, alloc_opt)

    alloc_opt.method = :LP
    alloc_opt.investment = 1e8
    w16 = allocate!(portfolio, alloc_opt)
    alloc_opt.method = :Greedy
    w17 = allocate!(portfolio, alloc_opt)

    alloc_opt.method = :LP
    alloc_opt.investment = 1e6
    alloc_opt.reinvest = true
    w18 = allocate!(portfolio, alloc_opt)
    alloc_opt.method = :Greedy
    w19 = allocate!(portfolio, alloc_opt)

    @test isempty(alloc_params1)
    @test isempty(alloc_params2)
    @test isapprox(w0.weights, w1.weights, rtol = 0.01)
    @test isapprox(w0.weights, w2.weights, rtol = 0.01)
    @test isapprox(w0.weights, w3.weights, rtol = 0.1)
    @test isapprox(w0.weights, w4.weights, rtol = 0.1)
    @test isapprox(w0.weights, w5.weights, rtol = 1)
    @test isapprox(w0.weights, w6.weights, rtol = 1)
    @test isapprox(w0.weights, w7.weights, rtol = 0.001)
    @test isapprox(w0.weights, w8.weights, rtol = 0.001)
    @test isapprox(w9.weights, w10.weights, rtol = 0.001)
    @test isapprox(w9.weights, w11.weights, rtol = 0.001)
    @test isapprox(w9.weights, w12.weights, rtol = 0.05)
    @test isapprox(w9.weights, w13.weights, rtol = 0.05)
    @test isapprox(w9.weights, w14.weights, rtol = 0.7)
    @test isapprox(w9.weights, w15.weights, rtol = 0.7)
    @test isapprox(w9.weights, w16.weights, rtol = 0.00001)
    @test isapprox(w9.weights, w17.weights, rtol = 0.00001)
    @test isapprox(portfolio.sum_short_long * 1e6, sum(w10.cost), rtol = 1e-5)
    @test isapprox(portfolio.sum_short_long * 1e6, sum(w11.cost), rtol = 1e-5)
    @test isapprox(portfolio.sum_short_long * 1e4, sum(w12.cost), rtol = 1e-3)
    @test isapprox(portfolio.sum_short_long * 1e4, sum(w13.cost), rtol = 1e-2)
    @test isapprox(portfolio.sum_short_long * 1e2, sum(w14.cost), rtol = 1e-1)
    @test isapprox(portfolio.sum_short_long * 1e2, sum(w15.cost), rtol = 1e-2)
    @test isapprox(portfolio.sum_short_long * 1e8, sum(w16.cost), rtol = 5e-7)
    @test isapprox(portfolio.sum_short_long * 1e8, sum(w17.cost), rtol = 5e-7)
    @test isapprox(sum(w18.cost[w18.cost .> 0]),
                   sum(w10.cost[w10.cost .> 0]) - sum(w10.cost[w10.cost .< 0]), rtol = 3e-4)
    @test isapprox(sum(w19.cost[w19.cost .> 0]),
                   sum(w11.cost[w11.cost .> 0]) - sum(w11.cost[w11.cost .< 0]), rtol = 3e-5)
end
