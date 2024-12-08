using CSV, TimeSeries, CovarianceEstimation, StatsBase, Statistics,
      NearestCorrelationMatrix, LinearAlgebra, Test, PortfolioOptimiser, Distances

path = joinpath(@__DIR__, "assets/stock_prices.csv")
prices = TimeArray(CSV.File(path); timestamp = :date)

include("StatsTest_1.jl")
include("StatsTest_2.jl")
include("StatsTest_3.jl")
include("StatsTest_4.jl")
