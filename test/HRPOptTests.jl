using Test

using PortfolioOptimiser.BaseOptimiser
using PortfolioOptimiser.HierarchicalOptimiser

using JuMP, LinearAlgebra, Ipopt, Statistics

@testset "HRP Opt" begin
    tickers = ["AAPL", "GOOG", "AMD", "MSFT"]
    n = length(tickers)
    ret = (randn(1000, n) .+ 0.01) / 100
    ret[:, 1] = ret[:, 2] .+ 1
    cov_mtx = cov(ret) * 252
    hropt = HRPOpt(tickers, cov_mtx = cov_mtx)
    @test hropt.weights[1] ≈ hropt.weights[2]
    portfolio_performance(hropt)

    k = 5
    ret = (randn(1000, n) .+ 0.01) / 100
    ret[:, 2] = k * ret[:, 1]
    cov_mtx = cov(ret) * 252
    hropt = HRPOpt(tickers, returns = ret, cov_mtx = cov_mtx)
    @test hropt.weights[1] ≈ hropt.weights[2] * k^2
    portfolio_performance(hropt)

    k = 2
    c = 6
    ret = (randn(1000, n) .+ 0.01) / 100
    ret[:, 4] = k * ret[:, 2] .- 6
    cov_mtx = cov(ret) * 252
    hropt = HRPOpt(tickers, returns = ret)
    @test hropt.weights[2] ≈ hropt.weights[4] * k^2
    portfolio_performance(hropt)
end