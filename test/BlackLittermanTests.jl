using Test
using PortfolioOptimiser.BlackLittermanOptimiser
using Statistics, LinearAlgebra

@testset "Black Litterman" begin
    tickers = ["AAPL", "GOOG", "AMD", "MSFT"]
    n = length(tickers)
    ret = (randn(1000, n) .+ 0.01) / 100
    # ret[:, 1] = 2*ret[:, 2] .+ randn(1000) / 1000
    cov_mtx = cov(ret) * 252
    mean_ret = vec(mean(ret, dims = 1))

    absolute_views = Dict()
    for ticker in tickers
        push!(absolute_views, (ticker => 0))
    end

    priors = [absolute_views[ticker] for ticker in tickers]
    cov_mtx = Matrix(I, length(tickers), length(tickers))
    bla = BlackLitterman(tickers, cov_mtx; absolute_views = absolute_views, pi = :equal)
    @test all(bla.weights .≈ 1 / length(tickers))

    pi = [0, 0, 0, 1]
    bla = BlackLitterman(tickers, cov_mtx; absolute_views = absolute_views, pi = pi)
    @test bla.weights[4] ≈ 1

    cov_mtx = Matrix(I, length(tickers), length(tickers))
    pi = [0, 1, 0, 1]
    bla = BlackLitterman(tickers, cov_mtx; absolute_views = absolute_views, pi = pi)
    tmp = sum(pi + priors)
    @test bla.weights ≈ pi / tmp

    pi = [1, 2, 3, 4]
    bla = BlackLitterman(tickers, cov_mtx; absolute_views = absolute_views, pi = pi)
    tmp = sum(pi + priors)
    @test bla.weights ≈ pi / tmp

    for ticker in tickers
        push!(absolute_views, (ticker => 1))
    end
    priors = [absolute_views[ticker] for ticker in tickers]

    cov_mtx = Matrix(I, length(tickers), length(tickers))
    bla = BlackLitterman(tickers, cov_mtx; absolute_views = absolute_views, pi = :equal)
    tmp = ones(length(tickers)) + priors
    tmpN = sum(tmp)
    @test bla.weights ≈ tmp / tmpN

    pi = [0, 0, 0, 1]
    bla = BlackLitterman(tickers, cov_mtx; absolute_views = absolute_views, pi = pi)
    tmp = pi + priors
    tmpN = sum(tmp)
    @test bla.weights ≈ tmp / tmpN

    pi = [1, 0, 0, 1]
    bla = BlackLitterman(tickers, cov_mtx; absolute_views = absolute_views, pi = pi)
    tmp = pi + priors
    tmpN = sum(tmp)
    @test bla.weights ≈ tmp / tmpN

    pi = [1, 0, 1, 1]
    bla = BlackLitterman(tickers, cov_mtx; absolute_views = absolute_views, pi = pi)
    tmp = pi + priors
    tmpN = sum(tmp)
    @test bla.weights ≈ tmp / tmpN

    for ticker in tickers
        push!(absolute_views, (ticker => rand(1:69666420)))
    end
    priors = [absolute_views[ticker] for ticker in tickers]

    pi = [420, 103, 69, 666]
    bla = BlackLitterman(tickers, cov_mtx; absolute_views = absolute_views, pi = pi)
    tmp = pi + priors
    tmpN = sum(tmp)
    @test bla.weights ≈ tmp / tmpN
end