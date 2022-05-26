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

    ef = MeanVar(tickers, bl.post_ret, S)
    max_sharpe!(ef)
    testweights = [
        0.2218961322310675,
        0.0,
        0.0,
        0.0678019811907844,
        0.0,
        0.0,
        0.0185451412830906,
        0.0,
        0.6917567452950577,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    @test ef.weights ≈ testweights
    mu, sigma, sr = portfolio_performance(ef; verbose = true)
    mutest, sigmatest, srtest = 0.08173248653446699, 0.2193583713684134, 0.2814229798906876
    @test mu ≈ mutest
    @test sigma ≈ sigmatest
    @test sr ≈ srtest

    ef.weights .= testweights
    lpAlloc, remaining =
        Allocation(LP(), ef, Vector(df[end, ef.tickers]); investment = 10000)
    testshares = [2, 4, 32, 232]
    @test lpAlloc.shares == testshares

    gAlloc, remaining =
        Allocation(Greedy(), ef, Vector(df[end, ef.tickers]); investment = 10000)
    testshares = [231, 2, 4, 19]
    @test gAlloc.shares == testshares

    ef = MeanVar(tickers, bl.post_ret, S)
    max_sharpe!(ef; rf = 0.03)
    testweights = [
        0.151542234028701,
        0.0,
        0.0,
        0.0477917898904942,
        0.0,
        0.0,
        0.0189565466074949,
        0.0,
        0.7817094294733101,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    @test ef.weights ≈ testweights
    mu, sigma, sr = portfolio_performance(ef, rf = 0.03)
    mutest, sigmatest, srtest =
        0.08467118320076503, 0.23080769881841998, 0.23686897569121254
    @test mu ≈ mutest
    @test sigma ≈ sigmatest
    @test sr ≈ srtest

    ef.weights .= testweights
    lpAlloc, remaining =
        Allocation(LP(), ef, Vector(df[end, ef.tickers]); investment = 10000)
    testshares = [1, 5, 20, 262, 1]
    @test lpAlloc.shares == testshares

    gAlloc, remaining =
        Allocation(Greedy(), ef, Vector(df[end, ef.tickers]); investment = 10000)
    testshares = [261, 1, 3, 19]
    @test gAlloc.shares == testshares

    ef = MeanVar(tickers, bl.post_ret, S, weight_bounds = (-1, 1), market_neutral = true)
    max_sharpe!(ef)
    testweights = [
        0.7629861545939385,
        0.2963858528342824,
        -0.4867169654536772,
        0.1770031902301856,
        0.1219934777789852,
        -0.463311778342318,
        0.0126761766739924,
        0.1905438219555414,
        0.5519529120031599,
        -0.0048640993308814,
        -0.1059039551376619,
        -0.0395702832452855,
        -0.0240980482485146,
        0.304351666002147,
        0.0157830822841903,
        -0.0012535995862868,
        -0.0387261734791586,
        0.31185831103011,
        0.4189102574372514,
        -1.0,
    ]
    @test isapprox(ef.weights, testweights, rtol = 1e-7)
    mu, sigma, sr = portfolio_performance(ef)
    mutest, sigmatest, srtest = 0.23339921135959843, 0.3472734444465163, 0.6144990778080186
    @test isapprox(mu, mutest, atol = 1e-7)
    @test isapprox(sigma, sigmatest, atol = 1e-7)
    @test isapprox(sr, srtest)

    ef.weights .= testweights
    lpAlloc, remaining =
        Allocation(LP(), ef, Vector(df[end, ef.tickers]); investment = 10000)
    testshares = [3, 5, 3, 2, 7, 58, 12, 3, 25, 12, -29, -358, -2, -30, -24, -76, -2, -169]
    @test lpAlloc.shares == testshares

    gAlloc, remaining =
        Allocation(Greedy(), ef, Vector(df[end, ef.tickers]); investment = 10000)
    testshares =
        [3, 58, 11, 27, 12, 5, 7, 3, 4, 4, -168, -29, -357, -30, -24, -2, -73, -2, -1]
    @test gAlloc.shares == testshares

    ef = MeanVar(tickers, bl.post_ret, S, weight_bounds = (-1, 1), market_neutral = true)
    max_sharpe!(ef; rf = 0.03)
    testweights = [
        0.859721100506911,
        0.3247835315495498,
        -0.5696453302820396,
        0.197757767120159,
        0.1169784737986242,
        -0.555730070015875,
        0.0192460007591768,
        0.1776434832537423,
        0.6369712226611369,
        -0.0171479766187555,
        -0.1750899998161749,
        -0.0475780373523676,
        -0.0283526177717667,
        0.3305666121890862,
        0.0208128324804424,
        -0.0139215444556887,
        -0.0721800419910253,
        0.3192304997153685,
        0.4759340942694961,
        -1.0,
    ]
    @test isapprox(ef.weights, testweights, rtol = 1e-7)
    mu, sigma, sr = portfolio_performance(ef, rf = 0.03)
    mutest, sigmatest, srtest = 0.2522500753886163, 0.37870359471655435, 0.5868707836136657
    @test isapprox(mu, mutest, rtol = 1e-7)
    @test isapprox(sigma, sigmatest, rtol = 1e-7)
    @test isapprox(sr, srtest)

    ef.weights .= testweights
    lpAlloc, remaining =
        Allocation(LP(), ef, Vector(df[end, ef.tickers]); investment = 10000)
    testshares = [3, 5, 3, 6, 61, 12, 4, 25, 12, -34, -428, -5, -50, -28, -86, -2, -4, -169]
    @test lpAlloc.shares == testshares

    gAlloc, remaining =
        Allocation(Greedy(), ef, Vector(df[end, ef.tickers]); investment = 10000)
    testshares =
        [3, 61, 12, 12, 5, 26, 3, 5, 4, 5, -168, -34, -428, -50, -4, -29, -86, -5, -2]
    @test gAlloc.shares == testshares

    ef = MeanVar(tickers, bl.post_ret, S)
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

    ef = MeanVar(tickers, bl.post_ret, S, weight_bounds = (-1, 1), market_neutral = true)
    min_volatility!(ef)
    testweights = [
        0.0035893754885593,
        0.0375547056505898,
        0.0176788401342915,
        0.0330861423819678,
        0.01248643676087,
        0.053795551664664,
        -0.0097109321070962,
        0.1412184247312541,
        -0.0108961096879344,
        0.0185932679351068,
        0.2836823571307645,
        -0.0211412215698546,
        -0.0091328939148965,
        0.1458832849899442,
        0.0007735883622543,
        0.0254316960520564,
        0.0145689458116083,
        0.2035857178557598,
        -0.0642098569591094,
        0.1231626792892006,
    ]
    @test isapprox(ef.weights, testweights, rtol = 1e-6)
    mu, sigma, sr = portfolio_performance(ef)
    mutest, sigmatest, srtest =
        0.007547970562401391, 0.12111738873121844, -0.10280959297456398
    @test isapprox(mu, mutest, rtol = 1e-6)
    @test isapprox(sigma, sigmatest)
    @test isapprox(sr, srtest, rtol = 1e-6)

    ef.weights .= testweights
    lpAlloc, remaining =
        Allocation(LP(), ef, Vector(df[end, ef.tickers]); investment = 10000)
    testshares = [2, 1, 2, 37, 15, 4, 72, 17, 1, 3, 1, 51, 19, -10, -3, -12, -28, -6]
    @test lpAlloc.shares == testshares

    gAlloc, remaining =
        Allocation(Greedy(), ef, Vector(df[end, ef.tickers]); investment = 10000)
    testshares = [72, 51, 17, 15, 19, 37, 2, 2, 3, 4, 1, 1, 1, -6, -12, -3, -10, -28]
    @test gAlloc.shares == testshares

    ef = MeanVar(tickers, bl.post_ret, S)
    max_quadratic_utility!(ef)
    testweights = [
        0.142307545055626,
        0.0,
        0.0,
        0.045165241857315,
        0.0,
        0.0,
        0.019010547882617,
        0.0,
        0.793516665204442,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    @test isapprox(ef.weights, testweights, rtol = 1e-5)
    mu, sigma, sr = portfolio_performance(ef)
    mutest, sigmatest, srtest = 0.0850569180412342, 0.2324517001148201, 0.27987284244038296
    @test isapprox(mu, mutest, rtol = 1e-6)
    @test isapprox(sigma, sigmatest, rtol = 1e-6)
    @test isapprox(sr, srtest, rtol = 1e-6)

    ef.weights .= testweights
    lpAlloc, remaining =
        Allocation(LP(), ef, Vector(df[end, ef.tickers]); investment = 10000)
    testshares = [1, 3, 20, 276]
    @test lpAlloc.shares == testshares

    gAlloc, remaining =
        Allocation(Greedy(), ef, Vector(df[end, ef.tickers]); investment = 10000)
    testshares = [265, 1, 3, 19]
    @test gAlloc.shares == testshares

    max_quadratic_utility!(ef; risk_aversion = 2)
    testweights = [
        0.260285197239515,
        0.0259623849226945,
        0.0,
        0.0787427931282114,
        0.0,
        0.0,
        0.0140665423719514,
        0.0,
        0.4714296781051146,
        0.0,
        0.0,
        0.0,
        0.0,
        0.093976806372668,
        0.0,
        0.0,
        0.0,
        0.007694867323402,
        0.0478417305364431,
        0.0,
    ]
    @test isapprox(ef.weights, testweights, rtol = 1e-4)
    mu, sigma, sr = portfolio_performance(ef)
    mutest, sigmatest, srtest = 0.07268001277300389, 0.19307519783090837, 0.2728471257045664
    @test isapprox(mu, mutest, rtol = 1e-5)
    @test isapprox(sigma, sigmatest, rtol = 1e-5)
    @test isapprox(sr, srtest, rtol = 1e-5)

    ef.weights .= testweights
    lpAlloc, remaining =
        Allocation(LP(), ef, Vector(df[end, ef.tickers]); investment = 10000)
    testshares = [3, 1, 4, 4, 157, 12, 2, 3]
    @test lpAlloc.shares == testshares

    gAlloc, remaining =
        Allocation(Greedy(), ef, Vector(df[end, ef.tickers]); investment = 10000)
    testshares = [157, 2, 12, 5, 5, 2, 14, 3]
    @test gAlloc.shares == testshares

    ef = MeanVar(tickers, bl.post_ret, S, market_neutral = true)
    max_quadratic_utility!(ef)
    testweights = [
        1.0,
        0.4269138491084772,
        -0.8991129303952179,
        0.2765688553830926,
        0.1356254077969869,
        -1.0,
        0.0559694772916339,
        -0.0330083195852073,
        1.0,
        -0.0904643271628715,
        -0.7706840366064389,
        -0.0670112383932255,
        -0.0414072455079296,
        0.3106533484886974,
        0.0332371844698535,
        -0.1041749504644787,
        -0.2153156196396117,
        0.1692065670683684,
        0.8130039781478708,
        -1.0,
    ]
    @test isapprox(ef.weights, testweights, rtol = 1e-5)
    mu, sigma, sr = portfolio_performance(ef)
    mutest, sigmatest, srtest = 0.30865288246835887, 0.49285633685434205, 0.585673472944849
    @test isapprox(mu, mutest, rtol = 1e-5)
    @test isapprox(sigma, sigmatest, rtol = 1e-5)
    @test isapprox(sr, srtest, rtol = 1e-6)

    ef.weights .= testweights
    lpAlloc, remaining =
        Allocation(LP(), ef, Vector(df[end, ef.tickers]); investment = 10000)
    testshares = [
        2,
        7,
        4,
        20,
        82,
        11,
        6,
        13,
        18,
        -54,
        -771,
        -4,
        -23,
        -219,
        -40,
        -126,
        -15,
        -12,
        -169,
    ]
    @test lpAlloc.shares == testshares

    gAlloc, remaining =
        Allocation(Greedy(), ef, Vector(df[end, ef.tickers]); investment = 10000)
    testshares =
        [3, 79, 17, 5, 9, 3, 11, 14, 5, -771, -168, -54, -218, -13, -14, -23, -40, -126, -4]
    @test gAlloc.shares == testshares

    max_quadratic_utility!(ef; risk_aversion = 2)
    testweights = [
        0.6723080121060555,
        0.2293283866220051,
        -0.4458142682741828,
        0.1273168971350576,
        0.0976966343703026,
        -0.4567575678466532,
        0.0196730674931566,
        0.0444253566541527,
        0.497775243542892,
        -0.0204005817850093,
        -0.3441364891480292,
        -0.0160923309254676,
        -0.0131563044740263,
        0.1400571977540013,
        0.0131652059983546,
        -0.0232593961232808,
        -0.0461313939845397,
        0.0961089569301834,
        0.4278933739550276,
        -1.0,
    ]
    @test isapprox(ef.weights, testweights, rtol = 1e-7)
    mu, sigma, sr = portfolio_performance(ef)
    mutest, sigmatest, srtest = 0.2003415194346615, 0.28848826844075925, 0.6251260074088401
    @test isapprox(mu, mutest, rtol = 1e-7)
    @test isapprox(sigma, sigmatest, rtol = 1e-7)
    @test isapprox(sr, srtest, rtol = 1e-9)

    ef.weights .= testweights
    lpAlloc, remaining =
        Allocation(LP(), ef, Vector(df[end, ef.tickers]); investment = 10000)
    testshares =
        [3, 6, 3, 9, 2, 71, 8, 4, 12, 17, -27, -352, -5, -97, -9, -37, -3, -3, -168]
    @test lpAlloc.shares == testshares

    gAlloc, remaining =
        Allocation(Greedy(), ef, Vector(df[end, ef.tickers]); investment = 10000)
    testshares =
        [3, 70, 17, 6, 8, 3, 11, 3, 9, 3, -168, -352, -26, -98, -3, -4, -5, -10, -40]
    @test gAlloc.shares == testshares
end