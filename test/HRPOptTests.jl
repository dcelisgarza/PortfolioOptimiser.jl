using Test

using PortfolioOptimiser, CSV, DataFrames, Statistics

@testset "HRP Opt" begin
    tickers = ["AAPL", "GOOG", "AMD", "MSFT"]
    n = length(tickers)
    ret = (randn(1000, n) .+ 0.01) / 100
    ret[:, 1] = ret[:, 2] .+ 1
    cov_mtx = cov(ret) * 252
    hropt = HRPOpt(tickers, cov_mtx = cov_mtx)
    optimise!(hropt, min_volatility)
    @test hropt.weights[1] ≈ hropt.weights[2]
    mu, sigma, sr = portfolio_performance(hropt, verbose = true)
    @test isnan(mu)
    @test isnan(sr)

    k = 5
    ret = (randn(1000, n) .+ 0.01) / 100
    ret[:, 2] = k * ret[:, 1]
    cov_mtx = cov(ret) * 252

    hropt = HRPOpt(tickers, returns = ret, cov_mtx = cov_mtx)
    optimise!(hropt, min_volatility)
    @test hropt.weights[1] ≈ hropt.weights[2] * k^2
    portfolio_performance(hropt, verbose = true)

    k = 2
    c = 6
    ret = (randn(1000, n) .+ 0.01) / 100
    ret[:, 4] = k * ret[:, 2] .- 6
    cov_mtx = cov(ret) * 252
    hropt = HRPOpt(tickers, returns = ret)
    optimise!(hropt, min_volatility)
    @test hropt.weights[2] ≈ hropt.weights[4] * k^2
    portfolio_performance(hropt, verbose = true)

    # Reading in the data; preparing expected returns and a risk model
    df = CSV.read("./assets/stock_prices.csv", DataFrame)
    dropmissing!(df)
    tickers = names(df)[2:end]
    returns = returns_from_prices(df[!, 2:end])

    mu = vec(ret_model(MRet(), Matrix(returns)))
    S = risk_matrix(Cov(), Matrix(returns))

    hrp = HRPOpt(tickers, returns = Matrix(returns))
    optimise!(hrp, min_volatility)
    idx = sortperm(tickers)
    testweights = [
        0.05141029982354615,
        0.012973422626800228,
        0.02018216653246635,
        0.04084621303656177,
        0.015567906456662952,
        0.0377521927556203,
        0.04075799338381744,
        0.06072154753518565,
        0.04354241996221517,
        0.03182464218785058,
        0.02325487286365021,
        0.04956897986914887,
        0.10700323656585976,
        0.017325239748703498,
        0.08269670342726206,
        0.010999466705659471,
        0.15533136214701582,
        0.02353673037019126,
        0.10170965737619252,
        0.07299494662558993,
    ]
    @test hrp.weights[idx] ≈ testweights
    mu, sigma, sr = portfolio_performance(hrp, verbose = true)
    mutest, sigmatest, srtest = 0.10779113291906073, 0.1321728564045751, 0.6642145392571078
    @test mu ≈ mutest
    @test sigma ≈ sigmatest
    @test sr ≈ srtest

    @test_throws ArgumentError HRPOpt(tickers)
end