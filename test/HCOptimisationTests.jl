using CSV, TimeSeries, DataFrames, StatsBase, Statistics, LinearAlgebra, Test, Clarabel,
      PortfolioOptimiser

prices = TimeArray(CSV.File("./test/assets/stock_prices.csv"); timestamp = :date)
rf = 1.0329^(1 / 252) - 1
l = 2.0

portfolio = HCPortfolio2(; prices = prices,
                         solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                          :params => Dict("verbose" => false,
                                                                          "max_step_fraction" => 0.75))))

asset_statistics2!(portfolio)
@time w1 = optimise2!(portfolio; rm = RDaR2(), cluster = false,
                      type = NCO2(; options = (; obj = SR(; rf = rf))), hclust_alg = DBHT(),
                      hclust_opt = HClustOpt())

@test vec(portfolio.clusters.merges) ==
      [-14, -11, -19, -18, -17, -7, -16, -13, -1, -4, -3, -9, 10, 2, 13, 12, 6, 8, 17, -15,
       -6, -12, -8, -2, 5, -10, 7, -5, 9, -20, 3, 11, 1, 4, 14, 15, 16, 18]
@test vec(portfolio.clusters.heights) ==
      [0.05263157894736842, 0.05555555555555555, 0.058823529411764705, 0.0625,
       0.06666666666666667, 0.07142857142857142, 0.07692307692307693, 0.08333333333333333,
       0.09090909090909091, 0.1, 0.1111111111111111, 0.125, 0.14285714285714285,
       0.16666666666666666, 0.2, 0.25, 0.3333333333333333, 0.5, 1.0]
@test vec(portfolio.clusters.order) ==
      [7, 17, 2, 4, 1, 5, 3, 20, 18, 8, 13, 16, 10, 9, 19, 12, 11, 6, 14, 15]
@test vec(portfolio.clusters.labels) == 1:20
@test portfolio.clusters.method == :DBHT
@test portfolio.k == 3

using Distances
struct POCorDist <: Distances.UnionMetric end
function Distances.pairwise(::POCorDist, mtx, i)
    return sqrt.(clamp!((1 .- mtx) / 2, 0, 1))
end
dbht_d(corr, dist) = 2 .- (dist .^ 2) / 2
portfolio2 = HCPortfolio(; prices = prices,
                         solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                          :params => Dict("verbose" => false,
                                                                          "max_step_fraction" => 0.75))))
asset_statistics!(portfolio2; cor_opt = CorOpt(; dist = DistOpt(; method = POCorDist())))

cluster_opt = ClusterOpt(; linkage = :DBHT, genfunc = GenericFunction(; func = dbht_d))

@time w1_o = optimise!(portfolio2; type = :HERC, rm = :SD, rf = rf,
                       cluster_opt = cluster_opt, cluster = true)

@time w5 = optimise!(portfolio2; type = :NCO,
                     nco_opt = OptimiseOpt(; rm = :RDaR, obj = :Sharpe, rf = rf, l = l),
                     #    nco_opt_o = OptimiseOpt(; rm = :RDaR, obj = :Sharpe, rf = rf, l = l),
                     cluster = false, cluster_opt = cluster_opt)

println(kurt_flag)
println(skurt_flag)
println(skew_flag)
println(sskew_flag)
