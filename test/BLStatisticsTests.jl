using COSMO, CovarianceEstimation, CSV, Clarabel, HiGHS, LinearAlgebra, OrderedCollections,
      PortfolioOptimiser, Statistics, StatsBase, Test, TimeSeries, Logging

Logging.disable_logging(Logging.Warn)

prices_assets = TimeArray(CSV.File("./assets/stock_prices.csv"); timestamp = :date)
prices_factors = TimeArray(CSV.File("./assets/factor_prices.csv"); timestamp = :date)

rf = 1.0329^(1 / 252) - 1
l = 2.0

@testset "Factor Statistics" begin
    portfolio = Portfolio(; prices = prices_assets, f_prices = prices_factors)
end
