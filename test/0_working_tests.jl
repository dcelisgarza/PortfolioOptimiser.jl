using CSV, TimeSeries, DataFrames, StatsBase, Statistics, LinearAlgebra, Test, Pajarito,
      JuMP, Clarabel, PortfolioOptimiser, HiGHS

path = joinpath(@__DIR__, "assets/stock_prices.csv")
prices = TimeArray(CSV.File(path); timestamp = :date)
rf = 1.0329^(1 / 252) - 1
l = 2.0

r = logrange(1e-10, 1; length = 20)
function print_rtol(a, b) end
