using Test, PortfolioOptimiser, DataFrames, CSV, Dates, Clarabel, LinearAlgebra, Makie,
      TimeSeries

prices = TimeArray(CSV.File("./assets/stock_prices.csv"); timestamp = :date)

# using CairoMakie
# @testset "Plot returns" begin
portfolio = Portfolio(; prices = prices,
                      solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                       :check_sol => (allow_local = true,
                                                                      allow_almost = true),
                                                       :params => Dict("verbose" => false,
                                                                       "max_step_fraction" => 0.75))))
asset_statistics!(portfolio)
obj = MinRisk()
rm = SD()
w1 = optimise!(portfolio; type = Trad(), rm = rm, kelly = EKelly(), obj = obj)

idx, clustering, k, S, D = cluster_assets(portfolio; clust_alg = HAC(),
                                          clust_opt = ClustOpt(;
                                                               k_method = StdSilhouette()))

plot_clusters(portfolio.assets, S, idx, clustering, k, (-1, 1), :Spectral, true)

plot_range(portfolio.optimal[:Trad].weights, portfolio.returns)

plot_range(portfolio)

plot_hist(portfolio)

pdd = plot_drawdown(portfolio)

fw = efficient_frontier!(portfolio; rm = rm, points = 5)
pfa = plot_frontier_area(fw; rm = rm, t_factor = 252, palette = :Spectral)
pfa = plot_frontier_area(fw; rm = rm, t_factor = 252, palette = :Spectral)
plot_frontier_area(portfolio)

prp = plot_returns(portfolio; type = :Trad)
pra = plot_returns(portfolio; type = :Trad, per_asset = true)
pb = plot_bar(portfolio; type = :Trad)
prc = plot_risk_contribution(portfolio; type = :Trad, rm = rm, percentage = true)
pf = plot_frontier(portfolio; kelly = EKelly(), rm = rm)

# end

# using StatsPlots
# using GraphRecipes
# portfolio2 = Portfolio(; prices = prices,
#                        solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
#                                                         :params => Dict("verbose" => false,
#                                                                         "max_step_fraction" => 0.75))))
# asset_statistics!(portfolio2)
# rm = :SD
# obj = :Min_Risk
# fw2 = efficient_frontier!(portfolio2; points = 5)
# prc = plot_frontier_area(fw2)

# w = optimise!(portfolio2, OptimiseOpt(; type = :RP, rm = rm, obj = obj);
#               save_opt_params = true)
# plt1 = plot_risk_contribution(portfolio2; type = :RP, rm = rm, percentage = false)
# prp = plot_returns(portfolio)
# pra = plot_returns(portfolio; per_asset = true)
# pb = plot_bar(portfolio)
# prc = plot_risk_contribution(portfolio, :RP; rm = rm, percentage = true)
