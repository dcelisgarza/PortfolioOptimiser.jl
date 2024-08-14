using Test, PortfolioOptimiser, DataFrames, CSV, Dates, Clarabel, LinearAlgebra, Makie,
      TimeSeries

prices = TimeArray(CSV.File("./assets/stock_prices.csv"); timestamp = :date)

# using CairoMakie
# @testset "Plot returns" begin
portfolio = Portfolio2(; prices = prices,
                       solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                        :params => Dict("verbose" => false))))
asset_statistics2!(portfolio)
rm = SD2()
obj = MinRisk()
w1 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)

prp = plot_returns2(portfolio)
pra = plot_returns2(portfolio; per_asset = true)
pb = plot_bar2(portfolio)
# end
