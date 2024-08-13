using Test, PortfolioOptimiser, DataFrames, CSV, Dates, Clarabel, LinearAlgebra, Makie,
      TimeSeries

prices = TimeArray(CSV.File("./test/assets/stock_prices.csv"); timestamp = :date)

# using CairoMakie
# @testset "Plot returns" begin
portfolio = Portfolio2(; prices = prices,
                       solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                        :params => Dict("verbose" => false))))
asset_statistics2!(portfolio)
rm = SD2()
obj = MinRisk()
w1 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)

plot_returns2(portfolio; per_asset = true)
# end