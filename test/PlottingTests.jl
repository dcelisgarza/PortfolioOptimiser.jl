using Test, PortfolioOptimiser, DataFrames, CSV, Dates, Clarabel, LinearAlgebra, Makie,
      TimeSeries

prices = TimeArray(CSV.File("./assets/stock_prices.csv"); timestamp = :date)

# using CairoMakie
# @testset "Plot returns" begin
portfolio = Portfolio(; prices = prices,
                      solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                       :params => Dict("verbose" => false,
                                                                       "max_step_fraction" => 0.75))))
asset_statistics!(portfolio)
rm = SD()
obj = MinRisk()
w1 = optimise!(portfolio; type = RP(), rm = rm, kelly = AKelly(), obj = obj)

prp = plot_returns(portfolio, :RP)
pra = plot_returns(portfolio, :RP; per_asset = true)
pb = plot_bar(portfolio, :RP)
prc = plot_risk_contribution(portfolio, :RP; rm = rm, percentage = true)
fw = efficient_frontier!(portfolio; kelly = NoKelly(), rm = rm, points = 5)
pf = plot_frontier(portfolio; kelly = NoKelly(), rm = rm)

fw = efficient_frontier!(portfolio; rm = rm, points = 5)
pf = plot_frontier(portfolio; rm = rm)

pfa = plot_frontier_area2(fw; rm = rm, t_factor = 252)
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
