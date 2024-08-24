using CSV, TimeSeries, DataFrames, StatsBase, Statistics, LinearAlgebra, Test, Clarabel,
      PortfolioOptimiser

prices = TimeArray(CSV.File("./test/assets/stock_prices.csv"); timestamp = :date)
rf = 1.0329^(1 / 252) - 1
l = 2.0

# @testset "HRP" begin
portfolio = HCPortfolio2(; prices = prices,
                         solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                          :params => Dict("verbose" => false,
                                                                          "max_step_fraction" => 0.75))))

asset_statistics2!(portfolio)
hclust_alg = DBHT()
hclust_opt = HClustOpt()
type = HRP2()
w1 = optimise2!(portfolio; rm = SD2(), cluster = true, type = type, hclust_alg = hclust_alg,
                hclust_opt = hclust_opt)
wt = [0.03692879524929352, 0.06173471900996101, 0.05788485449055379, 0.02673494374797732,
      0.051021461747188426, 0.06928891082994745, 0.024148698656475776, 0.047867702815293824,
      0.03125471206249845, 0.0647453556976089, 0.10088112288028349, 0.038950479132407664,
      0.025466032983548218, 0.046088041928035575, 0.017522279227008376, 0.04994166864441962,
      0.076922254387969, 0.05541444481863126, 0.03819947378487138, 0.07900404790602693]
@test isapprox(w1.weights, wt)
w2 = optimise2!(portfolio; rm = SD2(), cluster = false, type = type,
                hclust_alg = hclust_alg, hclust_opt = hclust_opt)
# end
