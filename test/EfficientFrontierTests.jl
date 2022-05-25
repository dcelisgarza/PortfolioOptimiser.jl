using Test
using PortfolioOptimiser, DataFrames, CSV, Statistics

@testset "Mean Variance Optimisation and Allocation" begin
    df = CSV.read("./assets/stock_prices.csv", DataFrame)
    dropmissing!(df)
    returns = returns_from_prices(df[!, 2:end])
    tickers = names(df)[2:end]

    mu = vec(ret_model(MRet(), Matrix(returns)))
    S = risk_matrix(Cov(), Matrix(returns))

    spy_prices = CSV.read("./assets/spy_prices.csv", DataFrame)
    delta = market_implied_risk_aversion(spy_prices[!, 2])
    # In the order of the dataframes, the
    mcapsdf = DataFrame(
        ticker = tickers,
        mcap = [
            927e9,
            1.19e12,
            574e9,
            533e9,
            867e9,
            96e9,
            43e9,
            339e9,
            301e9,
            51e9,
            61e9,
            78e9,
            0,
            295e9,
            1e9,
            22e9,
            288e9,
            212e9,
            422e9,
            102e9,
        ],
    )

    prior = market_implied_prior_returns(mcapsdf[!, 2], S, delta)

    # 1. SBUX drop by 20%
    # 2. GOOG outperforms FB by 10%
    # 3. BAC and JPM will outperform T and GE by 15%
    views = [-0.20, 0.10, 0.15]
    picking = hcat(
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, -0.5, 0, 0, 0.5, 0, -0.5, 0, 0, 0, 0, 0, 0, 0, 0.5, 0],
    )

    bl = BlackLitterman(
        mcapsdf[!, 1],
        S;
        rf = 0,
        tau = 0.01,
        pi = prior, # either a vector, `nothing`, `:equal`, or `:market`
        Q = views,
        P = picking,
    )

    ef = MeanVar(names(df)[2:end], bl.post_ret, S)
    min_volatility!(ef)
    testweights = [
        0.007909381852655,
        0.030690045397095,
        0.010506892855973,
        0.027486977795442,
        0.012277615067487,
        0.033411624151502,
        0.0,
        0.139848395652807,
        0.0,
        0.0,
        0.287822361245511,
        0.0,
        0.0,
        0.125283674542888,
        0.0,
        0.015085474081069,
        0.0,
        0.193123876049454,
        0.0,
        0.116553681308117,
    ]
    @test isapprox(ef.weights, testweights, rtol = 1e-4)
    mu, sigma, sr = portfolio_performance(ef)
    mutest, sigmatest, srtest =
        0.011450350358250433, 0.1223065907437908, -0.06990342539805944
    @test isapprox(mu, mutest, rtol = 1e-4)
    @test isapprox(sigma, sigmatest, rtol = 1e-6)
    @test isapprox(sr, srtest, rtol = 1e-3)

    ef.weights .= testweights
    lpAlloc, remaining =
        Allocation(LP(), ef, Vector(df[end, ef.tickers]); investment = 10000)
    testshares = [2, 1, 2, 26, 16, 82, 16, 2, 54, 20]
    @test lpAlloc.shares == testshares

    gAlloc, remaining =
        Allocation(Greedy(), ef, Vector(df[end, ef.tickers]); investment = 10000)
    testshares = [81, 54, 16, 16, 20, 25, 2, 2, 3, 1]
    @test gAlloc.shares == testshares
end