using CSV, TimeSeries, StatsBase, Statistics, CovarianceEstimation,
      NearestCorrelationMatrix, LinearAlgebra, Test, PortfolioOptimiser

prices = TimeArray(CSV.File("./assets/stock_prices.csv"); timestamp = :date)

include("StatsTest_v2_1.jl")
include("StatsTest_v2_2.jl")
include("StatsTest_v2_3.jl")
include("StatsTest_v2_4.jl")
