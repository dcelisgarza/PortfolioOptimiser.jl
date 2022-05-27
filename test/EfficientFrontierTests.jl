using Test
using PortfolioOptimiser, DataFrames, CSV, Statistics, StatsBase

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
    @test (0.0, 0.0, 0.0) == portfolio_performance(ef)

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

    lAlloc, remaining =
        Allocation(Lazy(), ef, Vector(df[end, ef.tickers]); investment = 10000)
    testshares = [231, 2, 4, 19]
    @test lAlloc.shares == testshares

    ef = MeanVar(tickers, bl.post_ret, S)
    max_sharpe!(ef, 0.03)
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

    ef = MeanVar(tickers, bl.post_ret, S, weight_bounds = (-1, 1))
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

    ef = MeanVar(tickers, bl.post_ret, S, weight_bounds = (-1, 1))
    max_sharpe!(ef, 0.03)
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

    max_quadratic_utility!(ef, 2)
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

    efficient_return!(ef, 0.07)
    testweights = [
        0.248314255432587,
        0.034008459762881,
        0.0,
        0.07614301178335,
        0.0,
        0.0,
        0.012512720842415,
        0.0,
        0.425848064696512,
        0.0,
        0.0,
        0.0,
        0.0,
        0.11002301819797,
        0.0,
        0.0,
        0.0,
        0.038837837899579,
        0.054312631384705,
        0.0,
    ]
    @test isapprox(ef.weights, testweights, rtol = 1e-4)
    mu, sigma, sr = portfolio_performance(ef)
    mutest, sigmatest, srtest = 0.07, 0.18628889663990178, 0.26840032284184107
    @test isapprox(mu, mutest, rtol = 1e-6)
    @test isapprox(sigma, sigmatest, rtol = 1e-7)
    @test isapprox(sr, srtest, rtol = 1e-6)

    ef.weights .= testweights
    lpAlloc, remaining =
        Allocation(LP(), ef, Vector(df[end, ef.tickers]); investment = 10000)
    testshares = [2, 2, 5, 13, 143, 15, 11, 5]
    @test rmsd(Int.(lpAlloc.shares), testshares) <= 1

    gAlloc, remaining =
        Allocation(Greedy(), ef, Vector(df[end, ef.tickers]); investment = 10000)
    testshares = [142, 2, 14, 5, 5, 11, 2, 13]
    @test gAlloc.shares == testshares

    efficient_risk!(ef, 0.2)
    testweights = [
        0.2682965379536906,
        0.0130172761324803,
        6.941516e-10,
        0.0808298072358772,
        2.7783689e-09,
        7.370853e-10,
        0.0156956683424278,
        3.0964532e-09,
        0.5290628313405414,
        1.5998805e-09,
        1.5906307e-09,
        6.62934e-10,
        5.250114e-10,
        0.0650898725396162,
        4.40140943e-08,
        1.4353985e-09,
        2.1663947e-09,
        1.62755797e-08,
        0.0280079305642093,
        3.151541e-10,
    ]
    @test isapprox(ef.weights, testweights, rtol = 1e-4)
    mu, sigma, sr = portfolio_performance(ef)
    mutest, sigmatest, srtest = 0.07528445695449039, 0.1999999998533396, 0.2764222849751529
    @test isapprox(mu, mutest, rtol = 1e-8)
    @test isapprox(sigma, sigmatest, rtol = 1e-6)
    @test isapprox(sr, srtest, rtol = 1e-6)

    ef.weights .= testweights
    lpAlloc, remaining =
        Allocation(LP(), ef, Vector(df[end, ef.tickers]); investment = 10000)
    testshares = [3, 4, 13, 176, 8, 2]
    @test lpAlloc.shares == testshares

    gAlloc, remaining =
        Allocation(Greedy(), ef, Vector(df[end, ef.tickers]); investment = 10000)
    testshares = [176, 2, 5, 9, 3, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1]
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

    max_quadratic_utility!(ef, 2)
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

    efficient_return!(ef, 0.07)
    testweights = [
        1.955786393513531e-01,
        7.211643474355210e-02,
        -1.076236901818929e-01,
        3.417582789309590e-02,
        5.082732970587820e-02,
        -1.015078387959005e-01,
        1.330070131094100e-03,
        3.558932040881150e-02,
        1.290677217451837e-01,
        5.048447975984000e-03,
        -7.700222068272380e-02,
        2.006233620643200e-03,
        -1.052761181828400e-03,
        3.362061842342390e-02,
        1.461368350657000e-04,
        4.184759818585200e-03,
        1.808508509531880e-02,
        3.554402444179850e-02,
        1.298148468599680e-01,
        -4.599489862074102e-01,
    ]
    @test isapprox(ef.weights, testweights, rtol = 1e-7)
    mu, sigma, sr = portfolio_performance(ef)
    mutest, sigmatest, srtest = 0.06999999999999999, 0.09749810496318828, 0.5128304803347528
    @test isapprox(mu, mutest, rtol = 1e-8)
    @test isapprox(sigma, sigmatest, rtol = 1e-8)
    @test isapprox(sr, srtest, rtol = 1e-8)

    ef.weights .= testweights
    lpAlloc, remaining =
        Allocation(LP(), ef, Vector(df[end, ef.tickers]); investment = 10000)
    testshares = [3, 6, 3, 2, 6, 58, 2, 2, 6, 1, 1, 1, 14, 16, -6, -79, -22, -4, -78]
    @test lpAlloc.shares == testshares

    gAlloc, remaining =
        Allocation(Greedy(), ef, Vector(df[end, ef.tickers]); investment = 10000)
    testshares = [2, 15, 57, 6, 1, 5, 13, 3, 6, 1, 1, 1, 2, -77, -6, -78, -22, -4]
    @test gAlloc.shares == testshares

    efficient_risk!(ef, 0.2)
    testweights = [
        0.4011943951922512,
        0.1479341183049058,
        -0.2207710538718072,
        0.0701058973786375,
        0.1042631570687963,
        -0.2082259922773494,
        0.0027284318265696,
        0.0730047098589424,
        0.2647605713507618,
        0.0103558681743425,
        -0.1579565672577331,
        0.0041153260963731,
        -0.0021595940748951,
        0.0689669718741666,
        0.0002997857006797,
        0.0085841244901788,
        0.0370980606306332,
        0.0729122541734031,
        0.2662917880920323,
        -0.9435022527308914,
    ]
    @test isapprox(ef.weights, testweights, rtol = 1e-5)
    mu, sigma, sr = portfolio_performance(ef)
    mutest, sigmatest, srtest = 0.14359253441694841, 0.19999999989330683, 0.6179626724144041
    @test isapprox(mu, mutest, rtol = 1e-6)
    @test isapprox(sigma, sigmatest, rtol = 1e-6)
    @test isapprox(sr, srtest, rtol = 1e-7)

    ef.weights .= testweights
    lpAlloc, remaining =
        Allocation(LP(), ef, Vector(df[end, ef.tickers]); investment = 10000)
    testshares = [3, 6, 3, 2, 6, 58, 2, 2, 6, 1, 1, 1, 14, 16, -13, -162, -45, -7, -159]
    @test rmsd(lpAlloc.shares, testshares) <= 0.5

    gAlloc, remaining =
        Allocation(Greedy(), ef, Vector(df[end, ef.tickers]); investment = 10000)
    testshares = [2, 15, 57, 6, 1, 5, 13, 3, 6, 1, 1, 1, 2, -158, -13, -160, -45, -7]
    @test gAlloc.shares == testshares
end

@testset "Mean Semivar" begin
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

    ef = MeanSemivar(tickers, bl.post_ret, Matrix(returns))
    @test (0.0, 0.0, 0.0) == portfolio_performance(ef)

    max_sortino!(ef)
    mumax, varmax, smax = portfolio_performance(ef)

    ef = MeanSemivar(tickers, bl.post_ret, Matrix(returns))
    efficient_risk!(ef, varmax)
    mu, sigma, sr = portfolio_performance(ef)
    @test isapprox(mu, mumax, rtol = 1e-5)
    @test isapprox(sigma, varmax, rtol = 1e-5)
    @test isapprox(sr, smax, rtol = 1e-6)

    efficient_return!(ef, mumax)
    mu, sigma, sr = portfolio_performance(ef)
    @test isapprox(mu, mumax, rtol = 1e-5)
    @test isapprox(sigma, varmax, rtol = 1e-4)
    @test isapprox(sr, smax, rtol = 1e-4)

    ef = MeanSemivar(tickers, bl.post_ret, Matrix(returns))
    max_sortino!(ef, 0.03)
    mumax, varmax, smax = portfolio_performance(ef, rf = 0.03)

    ef = MeanSemivar(tickers, bl.post_ret, Matrix(returns))
    efficient_risk!(ef, varmax)
    mu, sigma, sr = portfolio_performance(ef, rf = 0.03)
    @test isapprox(mu, mumax, rtol = 1e-5)
    @test isapprox(sigma, varmax, rtol = 1e-5)
    @test isapprox(sr, smax, rtol = 1e-7)

    efficient_return!(ef, mumax)
    mu, sigma, sr = portfolio_performance(ef, rf = 0.03)
    @test isapprox(mu, mumax, rtol = 1e-5)
    @test isapprox(sigma, varmax, rtol = 1e-3)
    @test isapprox(sr, smax, rtol = 1e-3)

    ef = MeanSemivar(tickers, bl.post_ret, Matrix(returns), weight_bounds = (-1, 1))
    max_sortino!(ef)
    mumax, varmax, smax = portfolio_performance(ef)

    ef = MeanSemivar(tickers, bl.post_ret, Matrix(returns), weight_bounds = (-1, 1))
    efficient_risk!(ef, varmax)
    mu, sigma, sr = portfolio_performance(ef)
    @test isapprox(mu, mumax, rtol = 1e-5)
    @test isapprox(sigma, varmax, rtol = 1e-5)
    @test isapprox(sr, smax, rtol = 1e-6)

    efficient_return!(ef, mumax)
    mu, sigma, sr = portfolio_performance(ef)
    @test isapprox(mu, mumax, rtol = 1e-5)
    @test isapprox(sigma, varmax, rtol = 1e-4)
    @test isapprox(sr, smax, rtol = 1e-4)

    ef = MeanSemivar(tickers, bl.post_ret, Matrix(returns), weight_bounds = (-1, 1))
    max_sortino!(ef, 0.03)
    mumax, varmax, smax = portfolio_performance(ef, rf = 0.03)

    ef = MeanSemivar(tickers, bl.post_ret, Matrix(returns), weight_bounds = (-1, 1))
    efficient_risk!(ef, varmax)
    mu, sigma, sr = portfolio_performance(ef, rf = 0.03)
    @test isapprox(mu, mumax, rtol = 1e-5)
    @test isapprox(sigma, varmax, rtol = 1e-5)
    @test isapprox(sr, smax, rtol = 1e-6)

    efficient_return!(ef, mumax)
    mu, sigma, sr = portfolio_performance(ef, rf = 0.03)
    @test isapprox(mu, mumax, rtol = 1e-5)
    @test isapprox(sigma, varmax, rtol = 1e-4)
    @test isapprox(sr, smax, rtol = 1e-4)

    ef = MeanSemivar(tickers, bl.post_ret, Matrix(returns))
    min_semivar!(ef)
    testweights = [
        -8.8369599e-09,
        0.0521130667306068,
        -5.54696556e-08,
        0.0032731356678013,
        0.0324650564450746,
        6.3744722e-09,
        3.37125568e-07,
        0.1255659243075475,
        1.633273845e-07,
        4.5898985e-08,
        0.3061337299353124,
        -6.68458646e-08,
        -1.4187474e-08,
        0.0967467542201713,
        0.004653018274916,
        0.0198838224019178,
        0.0018092577849403,
        0.2315968397352792,
        1.032235721e-07,
        0.1257588839643445,
    ]
    @test isapprox(ef.weights, testweights, rtol = 1e-1)
    mu, sigma, sr = portfolio_performance(ef, verbose = true)
    mutest, sigmatest, srtest =
        0.011139799284510227, 0.08497381732464267, -0.1042697738485651
    @test isapprox(mu, mutest, atol = 1e-2)
    @test isapprox(sigma, sigmatest, rtol = 1e-3)
    @test isapprox(sr, srtest, atol = 1e-1)

    ef.weights .= testweights
    lpAlloc, remaining =
        Allocation(LP(), ef, Vector(df[end, ef.tickers]); investment = 10000)
    testshares = [3, 1, 1, 2, 15, 87, 13, 4, 3, 65, 22]
    @test lpAlloc.shares == testshares

    gAlloc, remaining =
        Allocation(Greedy(), ef, Vector(df[end, ef.tickers]); investment = 10000)
    testshares = [86, 64, 21, 15, 13, 3, 3, 3, 1, 1, 1]
    @test gAlloc.shares == testshares

    max_quadratic_utility!(ef)
    testweights = [
        1.100000000000000e-15,
        1.200000000000000e-15,
        1.100000000000000e-15,
        1.000000000000000e-15,
        1.200000000000000e-15,
        1.000000000000000e-15,
        5.000000000000000e-16,
        1.500000000000000e-15,
        9.999999999999784e-01,
        9.000000000000000e-16,
        1.500000000000000e-15,
        8.000000000000000e-16,
        7.000000000000000e-16,
        1.200000000000000e-15,
        9.000000000000000e-16,
        1.100000000000000e-15,
        1.000000000000000e-15,
        1.300000000000000e-15,
        5.000000000000000e-16,
        1.300000000000000e-15,
    ]
    @test isapprox(ef.weights, testweights, rtol = 1e-3)
    mu, sigma, sr = portfolio_performance(ef)
    mutest, sigmatest, srtest = 0.091419219703374, 0.1829724238719523, 0.39032777831786586
    @test isapprox(mu, mutest, rtol = 1e-3)
    @test isapprox(sigma, sigmatest, rtol = 1e-3)
    @test isapprox(sr, srtest, rtol = 1e-4)

    ef.weights .= testweights
    lpAlloc, remaining =
        Allocation(LP(), ef, Vector(df[end, ef.tickers]); investment = 10000)
    testshares = [334]
    @test lpAlloc.shares == testshares

    gAlloc, remaining =
        Allocation(Greedy(), ef, Vector(df[end, ef.tickers]); investment = 10000)
    testshares = [334, 1]
    @test gAlloc.shares == testshares

    max_quadratic_utility!(ef, 2)
    testweights = [
        0.1462022850123406,
        1.72696369e-08,
        2.8627388e-09,
        2.761140791e-07,
        1.01336466e-08,
        2.513371e-09,
        0.0066277509841478,
        4.751995e-09,
        0.8471695614090771,
        4.6036865e-09,
        3.7028104e-09,
        2.755922e-09,
        1.9606178e-09,
        1.04848137e-08,
        1.19804731e-08,
        4.6723839e-09,
        6.2853476e-09,
        8.8590514e-09,
        3.23131513e-08,
        1.3307094e-09,
    ]
    @test isapprox(ef.weights, testweights, rtol = 1e-2)
    mu, sigma, sr = portfolio_performance(ef)
    mutest, sigmatest, srtest = 0.08650132737011566, 0.16695152293028043, 0.3983271682875685
    @test isapprox(mu, mutest, rtol = 1e-3)
    @test isapprox(sigma, sigmatest, rtol = 1e-3)
    @test isapprox(sr, srtest, rtol = 1e-3)

    ef.weights .= testweights
    lpAlloc, remaining =
        Allocation(LP(), ef, Vector(df[end, ef.tickers]); investment = 10000)
    testshares = [1, 1, 7, 284, 1, 1, 1, 1, 1]
    @test lpAlloc.shares == testshares

    gAlloc, remaining =
        Allocation(Greedy(), ef, Vector(df[end, ef.tickers]); investment = 10000)
    testshares = [283, 1, 7, 1, 1, 1, 1, 1, 1]
    @test gAlloc.shares == testshares

    efficient_return!(ef, 0.09)
    testweights = [
        3.904746815385050e-02,
        5.923397958900000e-06,
        9.829962419900001e-06,
        4.485521037400000e-06,
        6.398473463000000e-06,
        1.058079920190000e-05,
        8.609310912354300e-03,
        9.421584972200000e-06,
        9.522038516156038e-01,
        7.426588225800000e-06,
        1.019932942990000e-05,
        9.315528047900000e-06,
        1.118486159000000e-05,
        6.553049888500000e-06,
        5.498591175600000e-06,
        8.239081503600000e-06,
        6.940805469200000e-06,
        7.319651581700000e-06,
        3.430088268500000e-06,
        1.661143469400000e-05,
    ]
    @test isapprox(ef.weights, testweights, rtol = 1e-2)
    mu, sigma, sr = portfolio_performance(ef)
    mutest, sigmatest, srtest = 0.09000090563810152, 0.17811334827609804, 0.393013248673487
    @test isapprox(mu, mutest, rtol = 1e-5)
    @test isapprox(sigma, sigmatest, rtol = 1e-4)
    @test isapprox(sr, srtest, rtol = 1e-4)

    ef.weights .= testweights
    lpAlloc, remaining =
        Allocation(LP(), ef, Vector(df[end, ef.tickers]); investment = 10000)
    testshares = [1, 9, 319, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    @test lpAlloc.shares == testshares

    gAlloc, remaining =
        Allocation(Greedy(), ef, Vector(df[end, ef.tickers]); investment = 10000)
    testshares = [318, 9, 1, 1, 1, 1, 1, 1, 1, 1]
    @test gAlloc.shares == testshares

    efficient_risk!(ef, 0.13)
    testweights = [
        2.874750375415112e-01,
        6.195275902493500e-02,
        4.152863104000000e-07,
        3.754883121164440e-02,
        2.335468505800000e-06,
        4.139595347000000e-07,
        5.180351073718700e-03,
        2.282885531500000e-06,
        4.040036784043188e-01,
        8.569500938000000e-07,
        1.073950326300000e-06,
        4.148599173000000e-07,
        3.056287327000000e-07,
        5.362159168381390e-02,
        3.496258672140000e-05,
        1.238539893500000e-06,
        1.317666202400000e-06,
        6.146402530123640e-02,
        8.870790761308990e-02,
        2.002169627000000e-07,
    ]
    @test isapprox(ef.weights, testweights, rtol = 1e-2)
    mu, sigma, sr = portfolio_performance(ef)
    mutest, sigmatest, srtest = 0.06959704494304779, 0.1300556974853854, 0.38135234289617415
    @test isapprox(mu, mutest, rtol = 1e-3)
    @test isapprox(sigma, sigmatest, rtol = 1e-3)
    @test isapprox(sr, srtest, rtol = 1e-3)

    ef.weights .= testweights
    lpAlloc, remaining =
        Allocation(LP(), ef, Vector(df[end, ef.tickers]); investment = 10000)
    testshares = [3, 3, 2, 135, 7, 17, 8]
    @test lpAlloc.shares == testshares

    gAlloc, remaining =
        Allocation(Greedy(), ef, Vector(df[end, ef.tickers]); investment = 10000)
    testshares = [135, 3, 8, 3, 17, 6, 2, 6, 1, 1]
    @test gAlloc.shares == testshares

    ef = MeanSemivar(tickers, bl.post_ret, Matrix(returns), market_neutral = true)
    min_semivar!(ef)
    testweights = [
        -6.87746970453e-05,
        -1.40007211953e-05,
        2.38011464746e-05,
        4.6974224158e-06,
        0.000127662765229,
        -0.0001817372873211,
        1.84999231954e-05,
        -2.16770346062e-05,
        -5.9067519062e-05,
        1.08932533038e-05,
        1.38594330145e-05,
        -4.04318857464e-05,
        -1.23208757829e-05,
        -7.53221096413e-05,
        -3.7905535553e-05,
        3.82318960547e-05,
        7.14342416822e-05,
        -1.0279450203e-05,
        0.0002096363297839,
        2.800705002e-06,
    ]
    @test isapprox(ef.weights, testweights, atol = 1e-1)
    mu, sigma, sr = portfolio_performance(ef)
    mutest, sigmatest, srtest =
        9.68355572016792e-06, 3.830973475511849e-05, -521.8077486586869
    @test isapprox(mu, mutest, atol = 1e-3)
    @test isapprox(sigma, sigmatest, atol = 1e-2)

    ef.weights .= testweights
    lpAlloc, remaining =
        Allocation(LP(), ef, Vector(df[end, ef.tickers]); investment = 10000)
    testshares = [2, 2, 34, 5, 6, 10, 8, 36, -1]
    @test lpAlloc.shares == testshares

    gAlloc, remaining =
        Allocation(Greedy(), ef, Vector(df[end, ef.tickers]); investment = 10000)
    testshares = [36, 2, 7, 10, 2, 37, 8, 5, 1, -1]
    @test gAlloc.shares == testshares

    max_quadratic_utility!(ef)
    testweights = [
        0.9999999987191048,
        0.8025061567310201,
        -0.9999999983660633,
        0.6315188600428658,
        0.4922749743437155,
        -0.9999999990605686,
        0.1762382499326327,
        -0.733252470272586,
        0.9999999993131146,
        -0.2585591997727487,
        -0.9999999983706372,
        -0.2591719865317014,
        -0.1281535712856913,
        0.4788874233441203,
        -0.0263096469774795,
        -0.0874576034104335,
        -0.4552360118252822,
        0.3667148242725146,
        0.9999999989095066,
        -0.9999999997354032,
    ]
    @test isapprox(ef.weights, testweights, rtol = 1e-2)
    mu, sigma, sr = portfolio_performance(ef)
    mutest, sigmatest, srtest = 0.3695444374167076, 0.4300511641888641, 0.8127973286062292
    @test isapprox(mu, mutest, rtol = 1e-3)
    @test isapprox(sigma, sigmatest, rtol = 1e-3)
    @test isapprox(sr, srtest, rtol = 1e-3)

    ef.weights .= testweights
    lpAlloc, remaining =
        Allocation(LP(), ef, Vector(df[end, ef.tickers]); investment = 10000)
    testshares = [
        1,
        8,
        6,
        1,
        30,
        57,
        11,
        17,
        15,
        -60,
        -771,
        -85,
        -66,
        -284,
        -155,
        -387,
        -17,
        -12,
        -27,
        -168,
    ]
    @test lpAlloc.shares == testshares

    gAlloc, remaining =
        Allocation(Greedy(), ef, Vector(df[end, ef.tickers]); investment = 10000)
    testshares = [
        57,
        15,
        1,
        8,
        6,
        1,
        11,
        17,
        30,
        -168,
        -771,
        -284,
        -60,
        -85,
        -27,
        -154,
        -66,
        -389,
        -12,
        -17,
    ]
    @test gAlloc.shares == testshares

    max_quadratic_utility!(ef, 2)
    testweights = [
        0.9999999857360712,
        0.4685479704127186,
        -0.8592431647111499,
        0.2920727402492227,
        0.3445846893910677,
        -0.999999984329561,
        0.0734859850410818,
        -0.172920245654445,
        0.9999999347257726,
        -0.2143822719074016,
        -0.937463177415737,
        -0.1008400899077779,
        -0.0510006884730285,
        0.2396024358477434,
        -0.0269876564283418,
        -0.038065236125291,
        -0.314010116118947,
        0.2966189911453665,
        0.9999998963472027,
        -0.9999999978245666,
    ]
    @test isapprox(ef.weights, testweights, rtol = 1e-2)
    mu, sigma, sr = portfolio_performance(ef)
    mutest, sigmatest, srtest = 0.3248302531838691, 0.3470639260949393, 0.8783115451200273
    @test isapprox(mu, mutest, rtol = 1e-3)
    @test isapprox(sigma, sigmatest, rtol = 1e-3)
    @test isapprox(sr, srtest, rtol = 1e-4)

    ef.weights .= testweights
    lpAlloc, remaining =
        Allocation(LP(), ef, Vector(df[end, ef.tickers]); investment = 10000)
    testshares = [
        2,
        5,
        2,
        1,
        12,
        69,
        6,
        16,
        19,
        -52,
        -772,
        -20,
        -55,
        -266,
        -61,
        -156,
        -18,
        -5,
        -18,
        -168,
    ]
    @test lpAlloc.shares == testshares

    gAlloc, remaining =
        Allocation(Greedy(), ef, Vector(df[end, ef.tickers]); investment = 10000)
    testshares = [
        2,
        70,
        19,
        6,
        17,
        4,
        7,
        15,
        -168,
        -771,
        -265,
        -52,
        -18,
        -55,
        -20,
        -60,
        -154,
        -6,
        -18,
    ]
    @test gAlloc.shares == testshares

    efficient_return!(ef, 0.09)
    testweights = [
        0.26587342118737,
        0.0965371129134666,
        -0.1083440586709533,
        0.0472631405569012,
        0.0925397377541989,
        -0.16577616699613,
        0.0072119213407655,
        0.0227400114825395,
        0.10350347319794,
        -0.0071334231129069,
        -0.0883015612045468,
        0.0134635120313752,
        -0.0009731371433595,
        0.010494163524052,
        -0.0066909561294794,
        0.0173324674445928,
        -0.0122390919128145,
        0.0763576474308947,
        0.232051643617103,
        -0.5959098573110091,
    ]
    @test isapprox(ef.weights, testweights, rtol = 5e-2)
    mu, sigma, sr = portfolio_performance(ef)
    mutest, sigmatest, srtest = 0.0900000005909446, 0.08157952690824329, 0.8580584276944535
    @test isapprox(mu, mutest, rtol = 1e-4)
    @test isapprox(sigma, sigmatest, rtol = 1e-3)
    @test isapprox(sr, srtest, rtol = 1e-3)

    ef.weights .= testweights
    lpAlloc, remaining =
        Allocation(LP(), ef, Vector(df[end, ef.tickers]); investment = 10000)
    testshares =
        [2, 6, 3, 1, 7, 3, 35, 7, 2, 3, 22, 21, -6, -128, -2, -26, -3, -5, -1, -100]
    @test lpAlloc.shares == testshares

    gAlloc, remaining =
        Allocation(Greedy(), ef, Vector(df[end, ef.tickers]); investment = 10000)
    testshares =
        [2, 21, 35, 6, 1, 21, 3, 3, 3, 8, 2, 8, -100, -128, -6, -25, -1, -2, -5, -3]
    @test gAlloc.shares == testshares

    efficient_risk!(ef, 0.13)
    testweights = [
        0.4238401998849985,
        0.1538686236477102,
        -0.1727359058771851,
        0.0753461438850074,
        0.1475612389187346,
        -0.2642402283037698,
        0.0115008067529429,
        0.0362772030456925,
        0.1649564844571324,
        -0.0113530132083892,
        -0.1407414688712099,
        0.0214516748483527,
        -0.0015593312594264,
        0.0166505514359901,
        -0.0106664370188864,
        0.0276341390914952,
        -0.0195065353158665,
        0.1216951812979697,
        0.3699322405966654,
        -0.9499115680157854,
    ]
    @test isapprox(ef.weights, testweights, rtol = 1e-3)
    mu, sigma, sr = portfolio_performance(ef)
    mutest, sigmatest, srtest = 0.14346415547400485, 0.1300415327156715, 0.9494209495665686
    @test isapprox(mu, mutest, rtol = 1e-3)
    @test isapprox(sigma, sigmatest, rtol = 1e-3)
    @test isapprox(sr, srtest, rtol = 1e-4)

    ef.weights .= testweights
    lpAlloc, remaining =
        Allocation(LP(), ef, Vector(df[end, ef.tickers]); investment = 10000)
    testshares =
        [2, 6, 3, 1, 7, 3, 35, 7, 2, 3, 22, 21, -10, -204, -3, -40, -8, -11, -1, -160]
    @test lpAlloc.shares == testshares

    gAlloc, remaining =
        Allocation(Greedy(), ef, Vector(df[end, ef.tickers]); investment = 10000)
    testshares =
        [2, 21, 35, 6, 1, 21, 3, 3, 3, 8, 2, 8, -159, -204, -11, -39, -1, -3, -7, -4]
    @test gAlloc.shares == testshares
end

@testset "Efficient CVaR" begin
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
end