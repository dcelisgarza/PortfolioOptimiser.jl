using Test, PortfolioOptimiser, DataFrames, CSV, Dates, Clarabel, LinearAlgebra, Makie,
      TimeSeries

prices = TimeArray(CSV.File("./test/assets/stock_prices.csv"); timestamp = :date)

using CairoMakie
# @testset "Plot returns" begin
portfolio = Portfolio2(; prices = prices,
                       solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                        :params => Dict("verbose" => false,
                                                                        "max_step_fraction" => 0.75))))
asset_statistics2!(portfolio)
rm = CDaR2()
obj = MinRisk()
w1 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)

prp = plot_returns2(portfolio)
pra = plot_returns2(portfolio; per_asset = true)
pb = plot_bar2(portfolio)
prc = plot_risk_contribution2(portfolio; rm = RDaR_r2(), percentage = false)
# end

# using GraphRecipes, StatsPlots
# portfolio = Portfolio(; prices = prices,
#                        solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
#                                                         :params => Dict("verbose" => false,
#                                                                         "max_step_fraction" => 0.75))))
# asset_statistics!(portfolio)
# rm = :CDaR
# obj = :Min_Risk
# w = optimise!(portfolio, OptimiseOpt(; rm = rm, obj = obj); save_opt_params = true)
# plt1 = plot_risk_contribution(portfolio; rm = :RDaR_r, percentage = false)

# prp = plot_returns2(portfolio)
# pra = plot_returns2(portfolio; per_asset = true)
# pb = plot_bar2(portfolio)
# prc = plot_risk_contribution2(portfolio; rm = RDaR_r2(), percentage = true)