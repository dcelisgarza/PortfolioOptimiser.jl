using COSMO, CSV, Clarabel, HiGHS, LinearAlgebra, OrderedCollections, PortfolioOptimiser,
      Statistics, Test, TimeSeries, Logging

Logging.disable_logging(Logging.Warn)

prices = TimeArray(CSV.File("./assets/stock_prices.csv"); timestamp = :date)

rf = 1.0329^(1 / 252) - 1
l = 2.0

@testset "Linear Programming" begin
    solvers = OrderedDict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                            :params => Dict("verbose" => false,
                                                            "max_step_fraction" => 0.75)),
                          :COSMO => Dict(:solver => COSMO.Optimizer,
                                         :params => Dict("verbose" => false)))
    alloc_solvers = Dict(:HiGHS => Dict(:solver => HiGHS.Optimizer,
                                        :params => Dict("log_to_console" => false)))
    portfolio = Portfolio(; prices = prices, solvers = solvers,
                          alloc_solvers = alloc_solvers)
    asset_statistics!(portfolio)
    w0 = optimise!(portfolio; obj = :Min_Risk)
    w1 = allocate!(portfolio; alloc_type = :LP)
    w2 = allocate!(portfolio; alloc_type = :Greedy)
    w3 = allocate!(portfolio; alloc_type = :LP, investment = 1e4)
    w4 = allocate!(portfolio; alloc_type = :Greedy, investment = 1e4)
    w5 = allocate!(portfolio; alloc_type = :LP, investment = 1e2)
    w6 = allocate!(portfolio; alloc_type = :Greedy, investment = 1e2)
    w7 = allocate!(portfolio; alloc_type = :LP, investment = 1e8)
    w8 = allocate!(portfolio; alloc_type = :Greedy, investment = 1e8)

    @test isapprox(w0.weights, w1.weights, rtol = 0.01)
    @test isapprox(w0.weights, w2.weights, rtol = 0.01)
    @test isapprox(w0.weights, w3.weights, rtol = 0.1)
    @test isapprox(w0.weights, w4.weights, rtol = 0.1)
    @test isapprox(w0.weights, w5.weights, rtol = 1)
    @test isapprox(w0.weights, w6.weights, rtol = 1)
    @test isapprox(w0.weights, w7.weights, rtol = 0.001)
    @test isapprox(w0.weights, w8.weights, rtol = 0.001)
end
