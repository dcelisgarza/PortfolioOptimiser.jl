using CSV, TimeSeries, StatsBase, Statistics, NearestCorrelationMatrix, LinearAlgebra, Test,
      PortfolioOptimiser

prices = TimeArray(CSV.File("./assets/stock_prices.csv"); timestamp = :date)

include("StatsTest_1.jl")
include("StatsTest_2.jl")
include("StatsTest_3.jl")
include("StatsTest_4.jl")
