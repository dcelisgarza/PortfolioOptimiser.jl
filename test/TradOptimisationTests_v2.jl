using CSV, TimeSeries, DataFrames, StatsBase, Statistics, LinearAlgebra, Test, Clarabel,
      PortfolioOptimiser

prices = TimeArray(CSV.File("./test/assets/stock_prices.csv"); timestamp = :date)
rf = 1.0329^(1 / 252) - 1
l = 2.0

include("TradOptimisationTests_1.jl")
include("TradOptimisationTests_2.jl")
include("TradOptimisationTests_3.jl")
include("TradOptimisationTests_4.jl")
