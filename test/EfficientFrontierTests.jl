using Test
using PortfolioOptimiser, DataFrames, CSV, Statistics, StatsBase, JuMP

@testset "Mean Variance" begin
    df = CSV.read("./assets/stock_prices.csv", DataFrame)
    dropmissing!(df)
    returns = returns_from_prices(df[!, 2:end])
    tickers = names(df)[2:end]

    mean_ret = ret_model(MRet(), Matrix(returns))
    S = risk_model(Cov(), Matrix(returns))

    ef = MeanVar(tickers, mean_ret, S)
    sectors = ["Tech", "Medical", "RealEstate", "Oil"]
    sector_map = Dict(tickers[1:4:20] .=> sectors[1])
    merge!(sector_map, Dict(tickers[2:4:20] .=> sectors[2]))
    merge!(sector_map, Dict(tickers[3:4:20] .=> sectors[3]))
    merge!(sector_map, Dict(tickers[4:4:20] .=> sectors[4]))
    sector_lower =
        Dict([(sectors[1], 0.2), (sectors[2], 0.1), (sectors[3], 0.15), (sectors[4], 0.05)])
    sector_upper =
        Dict([(sectors[1], 0.4), (sectors[2], 0.5), (sectors[3], 0.2), (sectors[4], 0.2)])

    # Do it twice for the coverage.
    add_sector_constraint!(ef, sector_map, sector_lower, sector_upper)
    add_sector_constraint!(ef, sector_map, sector_lower, sector_upper)

    max_sharpe!(ef)

    @test 0.2 - 1e-9 <= sum(ef.weights[1:4:20]) <= 0.4 + 1e-9
    @test 0.1 - 1e-9 <= sum(ef.weights[2:4:20]) <= 0.5 + 1e-9
    @test 0.15 - 1e-9 <= sum(ef.weights[3:4:20]) <= 0.2 + 1e-9
    @test 0.05 - 1e-9 <= sum(ef.weights[4:4:20]) <= 0.2 + 1e-9

    ef = MeanVar(tickers, mean_ret, S, market_neutral = true)
    add_sector_constraint!(ef, sector_map, sector_lower, sector_upper)

    ef = MeanVar(tickers, mean_ret, S)
    efficient_risk!(ef, 0.01, optimiser_attributes = "tol" => 1e-3)
    @test termination_status(ef.model) == MOI.NUMERICAL_ERROR

    efficient_risk!(ef, 0.01, optimiser_attributes = ("tol" => 1e-3, "max_iter" => 20))
    @test termination_status(ef.model) == MOI.ITERATION_LIMIT

    ef = MeanVar(tickers, mean_ret, S)
    max_sharpe!(ef)
    sr1 = sharpe_ratio(ef.weights, ef.mean_ret, ef.cov_mtx)
    mu, sigma, sr = portfolio_performance(ef)
    @test sr ≈ sr1
    @test L2_reg(ef.weights, 0.69) ≈ 0.2823863907490376
    @test transaction_cost(ef.weights, fill(1 / 20, 20), 0.005) ≈ 0.007928708353566113
    @test ex_ante_tracking_error(ef.weights, S, fill(1 / 20, 20)) ≈ 0.022384515086395627
    @test ex_post_tracking_error(ef.weights, Matrix(returns), fill(1 / 895, 895)) ≈
          0.16061468446601332

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

    # L1 Regularisation
    mu = vec(ret_model(MRet(), Matrix(dropmissing(returns))))
    S = risk_model(Cov(), Matrix(dropmissing(returns)))

    n = length(tickers)
    prev_weights = fill(1 / n, n)

    k = 0.001
    ef = MeanVar(
        tickers,
        mu,
        S;
        extra_vars = [:(0 <= l1)],
        extra_constraints = [
            :([model[:l1]; (model[:w] - $prev_weights)] in MOI.NormOneCone($(n + 1))),
        ],
        extra_obj_terms = [quote
            $k * model[:l1]
        end],
    )
    min_volatility!(ef)
    testweights = [
        0.0190824077428914,
        0.0416074082548058,
        0.014327445825398,
        0.0283293437359261,
        0.0158256985460487,
        0.05,
        0.0,
        0.1350653428695091,
        0.0,
        0.0095134235345545,
        0.275220844540597,
        0.0,
        0.0,
        0.1068488025990047,
        0.0,
        0.0212151308897821,
        0.0194625229450863,
        0.1745302544719146,
        0.0,
        0.0889713740444816,
    ]
    @test isapprox(ef.weights, testweights, rtol = 1e-4)

    max_quadratic_utility!(ef)
    testweights = [
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        -3e-16,
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
        0.0,
        0.0,
    ]
    @test isapprox(ef.weights, testweights, rtol = 1e-7)

    efficient_return!(ef, 0.05)
    testweights = [
        0.0190824077428914,
        0.0416074082548058,
        0.014327445825398,
        0.0283293437359261,
        0.0158256985460487,
        0.05,
        0.0,
        0.1350653428695091,
        0.0,
        0.0095134235345545,
        0.275220844540597,
        0.0,
        0.0,
        0.1068488025990047,
        0.0,
        0.0212151308897821,
        0.0194625229450863,
        0.1745302544719146,
        0.0,
        0.0889713740444816,
    ]
    @test isapprox(ef.weights, testweights, rtol = 1e-4)

    efficient_risk!(ef, 0.15)
    testweights = [
        3.862392e-10,
        1.462361578e-07,
        3.1757104e-09,
        1.8336637e-09,
        0.2776719131226072,
        1.233935e-10,
        8.6056733e-09,
        0.0807754777430426,
        5.576187e-10,
        6.772662e-10,
        0.1792988250175469,
        3.90913e-11,
        2.22677e-11,
        3.825667e-10,
        3.17097e-11,
        0.0681959031493882,
        0.1846163039508845,
        0.1107650269744363,
        0.0777902349701932,
        0.0208861530004506,
    ]
    @test isapprox(ef.weights, testweights, rtol = 1e-3)

    k = 0.001
    ef = MeanVar(
        tickers,
        mu,
        S;
        extra_vars = [:(0 <= l1)],
        extra_constraints = [
            :([model[:l1]; (model[:w] - $prev_weights)] in MOI.NormOneCone($(n + 1))),
        ],
        extra_obj_terms = [quote
            $k * model[:l1]
        end],
    )
    max_sharpe!(ef)
    testweights = [
        0.0,
        0.0,
        0.0,
        0.0,
        0.5864001037614084,
        0.0,
        0.0081872941075679,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.1043355596305029,
        0.2250264480349554,
        0.0,
        0.0760505944655653,
        0.0,
    ]
    @test isapprox(ef.weights, testweights, rtol = 1e-6)

    k = 0.01
    ef = MeanVar(
        tickers,
        mu,
        S;
        extra_vars = [:(0 <= l1)],
        extra_constraints = [
            :([model[:l1]; (model[:w] - $prev_weights)] in MOI.NormOneCone($(n + 1))),
            :(model[:w][6] == 0.2),
        ],
        extra_obj_terms = [quote
            $k * model[:l1]
        end],
    )
    max_sharpe!(ef)
    testweights = [
        0.0,
        0.0,
        0.0,
        0.0,
        0.6910750346160773,
        0.2,
        0.0125752537981832,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0784821104326499,
        0.0178676011530893,
        0.0,
        0.0,
        0.0,
    ]
    @test isapprox(ef.weights, testweights, rtol = 1e-5)

    k = 0.0001
    ef = MeanVar(
        tickers,
        mu,
        S;
        extra_vars = [:(0 <= l1)],
        extra_constraints = [
            :([model[:l1]; (model[:w] - $prev_weights)] in MOI.NormOneCone($(n + 1))),
            :(model[:w][6] == 0.2),
            :(model[:w][1] >= 0.01),
            :(model[:w][16] <= 0.03),
        ],
        extra_obj_terms = [quote
            $k * model[:l1]
        end],
    )
    constraint = max_sharpe!(ef)
    testweights = [
        0.01,
        0.0,
        0.0,
        0.0,
        0.6804859766794547,
        0.1999999999999998,
        0.0064046100477132,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.03,
        0.0731094132728335,
        0.0,
        0.0,
        0.0,
    ]
    @test isapprox(ef.weights, testweights, rtol = 1e-6)
end

@testset "Mean Semivariance" begin
    df = CSV.read("./assets/stock_prices.csv", DataFrame)
    dropmissing!(df)
    returns = returns_from_prices(df[!, 2:end])
    tickers = names(df)[2:end]

    mean_ret = vec(ret_model(MRet(), Matrix(returns)))
    S = risk_model(Cov(), Matrix(returns))

    ef = MeanSemivar(tickers, mean_ret, S, market_neutral = true)
    sectors = ["Tech", "Medical", "RealEstate", "Oil"]
    sector_map = Dict(tickers[1:4:20] .=> sectors[1])
    merge!(sector_map, Dict(tickers[2:4:20] .=> sectors[2]))
    merge!(sector_map, Dict(tickers[3:4:20] .=> sectors[3]))
    merge!(sector_map, Dict(tickers[4:4:20] .=> sectors[4]))
    sector_lower =
        Dict([(sectors[1], 0.2), (sectors[2], 0.1), (sectors[3], 0.15), (sectors[4], 0.05)])
    sector_upper =
        Dict([(sectors[1], 0.4), (sectors[2], 0.5), (sectors[3], 0.2), (sectors[4], 0.2)])

    # Do it for the warning.
    add_sector_constraint!(ef, sector_map, sector_lower, sector_upper)

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

    # L1 Regularisation
    mean_ret = ret_model(MRet(), Matrix(returns))
    S = risk_model(Cov(), Matrix(returns))
    n = length(tickers)
    prev_weights = fill(1 / n, n)

    k = 0.00001
    ef = MeanSemivar(
        tickers,
        mean_ret,
        Matrix(returns);
        extra_vars = [:(0 <= l1)],
        extra_constraints = [
            :([model[:l1]; (model[:w] - $prev_weights)] in MOI.NormOneCone($(n + 1))),
            # :(model[:w][6] == 0.2),
            # :(model[:w][1] >= 0.01),
            # :(model[:w][16] <= 0.03),
        ],
        extra_obj_terms = [quote
            $k * model[:l1]
        end],
    )
    min_semivar!(ef)
    testweights = [
        0.05000003815106,
        0.0500000323406222,
        0.0305470536068529,
        0.0119165898801848,
        0.0499999897938178,
        0.0500000778586521,
        -9.70902283e-08,
        0.1050006178939432,
        -1.074573985e-07,
        0.0217993269186788,
        0.2267216752364789,
        4.54544546e-08,
        2.475203931e-07,
        0.0500000608483738,
        0.0195085389944512,
        0.0500000224593989,
        0.050000026895504,
        0.1345057919377536,
        0.0499999927807392,
        0.0500000757003792,
    ]
    @test isapprox(ef.weights, testweights, rtol = 5e-2)

    max_quadratic_utility!(ef)
    testweights = [
        -1.926600572e-07,
        -4.32976813e-08,
        -2.100198937e-07,
        -2.227540243e-07,
        0.9999993287957882,
        1.339119054e-07,
        -1.90544883e-07,
        2.103890502e-07,
        2.02309627e-08,
        4.75400441e-08,
        2.628872504e-07,
        -6.86900634e-08,
        1.741756607e-07,
        1.917324164e-07,
        1.195934128e-07,
        1.58547084e-07,
        -1.05257917e-08,
        2.052601162e-07,
        9.02686384e-08,
        -5.2371344e-09,
    ]
    @test isapprox(ef.weights, testweights, rtol = 1e-4)

    efficient_return!(ef, 0.17)
    testweights = [
        0.0500000020066915,
        0.050000024220804,
        0.0499999820760388,
        0.0129296743862539,
        0.125422488440367,
        1.58893521e-07,
        3.848623616e-07,
        0.0926245150180408,
        0.0111676929257631,
        0.0193112602977583,
        0.2152467221514765,
        6.67588294e-08,
        1.043403337e-07,
        0.0500000617361081,
        3.382072052e-07,
        0.049999966749425,
        0.0499999984345341,
        0.1232965674920904,
        0.0499999660885184,
        0.0500000247739652,
    ]
    @test isapprox(ef.weights, testweights, rtol = 5e-2)

    efficient_risk!(ef, 0.13)
    testweights = [
        3.329183953e-07,
        9.835480279e-07,
        5.250592795e-07,
        3.243999157e-07,
        0.5778004406239079,
        1.522240624e-07,
        5.005016494e-07,
        2.4055191111e-06,
        5.712576846e-07,
        3.741311414e-07,
        0.0607364237999792,
        9.86216476e-08,
        7.53996833e-08,
        3.345659355e-07,
        8.62979338e-08,
        0.1264635025652296,
        0.1651783353678204,
        1.01678964876e-05,
        0.0698034082988953,
        9.565700518e-07,
    ]
    @test isapprox(ef.weights, testweights, rtol = 5e-4)

    k = 0.00001
    ef = MeanSemivar(
        tickers,
        mean_ret,
        Matrix(returns);
        extra_vars = [:(0 <= l1)],
        extra_constraints = [
            :([model[:l1]; (model[:w] - $prev_weights)] in MOI.NormOneCone($(n + 1))),
            # :(model[:w][6] == 0.2),
            # :(model[:w][1] >= 0.01),
            # :(model[:w][16] <= 0.03),
        ],
        extra_obj_terms = [quote
            $k * model[:l1]
        end],
    )
    max_sortino!(ef)
    mumax, sigmamax, srmax = portfolio_performance(ef, verbose = true)

    k = 0.00001
    ef = MeanSemivar(
        tickers,
        mean_ret,
        Matrix(returns);
        extra_vars = [:(0 <= l1)],
        extra_constraints = [
            :([model[:l1]; (model[:w] - $prev_weights)] in MOI.NormOneCone($(n + 1))),
            # :(model[:w][6] == 0.2),
            # :(model[:w][1] >= 0.01),
            # :(model[:w][16] <= 0.03),
        ],
        extra_obj_terms = [quote
            $k * model[:l1]
        end],
    )
    efficient_return!(ef, mumax)
    mu, sigma, sr = portfolio_performance(ef)
    @test isapprox(mumax, mu, rtol = 1e-4)
    @test isapprox(sigmamax, sigma, rtol = 1e-2)
    @test isapprox(srmax, sr, rtol = 1e-2)

    efficient_risk!(ef, sigmamax)
    mu, sigma, sr = portfolio_performance(ef)
    @test isapprox(mumax, mu, rtol = 1e-4)
    @test isapprox(sigmamax, sigma, rtol = 1e-4)
    @test isapprox(srmax, sr, rtol = 1e-4)

    k = 0.00001
    ef = MeanSemivar(
        tickers,
        mean_ret,
        Matrix(returns);
        extra_vars = [:(0 <= l1)],
        extra_constraints = [
            :([model[:l1]; (model[:w] - $prev_weights)] in MOI.NormOneCone($(n + 1))),
            :(model[:w][6] == 0.2),
            :(model[:w][1] >= 0.01),
            :(model[:w][16] <= 0.03),
        ],
        extra_obj_terms = [quote
            $k * model[:l1]
        end],
    )
    max_sortino!(ef, -0.01)

    @test ef.weights[6] ≈ 0.2
    @test ef.weights[1] >= 0.01
    @test ef.weights[16] <= 0.03
end

@testset "Efficient CVaR" begin
    df = CSV.read("./assets/stock_prices.csv", DataFrame)
    dropmissing!(df)
    returns = returns_from_prices(df[!, 2:end])
    tickers = names(df)[2:end]

    mu = vec(ret_model(MRet(), Matrix(returns)))
    S = risk_model(Cov(), Matrix(returns))

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

    cv = EfficientCVaR(tickers, bl.post_ret, Matrix(returns))
    min_cvar!(cv)
    testweights = [
        1.196700000000000e-12,
        4.242033345496910e-02,
        1.247300000000000e-12,
        3.092200000000000e-12,
        7.574027808866600e-03,
        1.305400000000000e-12,
        1.735000000000000e-13,
        9.464947870552300e-02,
        5.519000000000000e-13,
        7.337000000000000e-13,
        3.040110655916818e-01,
        6.947000000000000e-13,
        3.879000000000000e-13,
        6.564167130410940e-02,
        1.189600000000000e-12,
        2.937161110264830e-02,
        1.548900000000000e-12,
        3.663101128889494e-01,
        7.108000000000000e-13,
        9.002169913041960e-02,
    ]
    @test isapprox(cv.weights, testweights, rtol = 1e-3)
    mu, sigma = portfolio_performance(cv, verbose = true)
    mutest, sigmatest = 0.014253439792781208, 0.017049502122532846
    @test isapprox(mu, mutest, rtol = 1e-3)
    @test isapprox(sigma, sigmatest, rtol = 1e-3)

    cv.weights .= testweights
    lpAlloc, remaining =
        Allocation(LP(), cv, Vector(df[end, cv.tickers]); investment = 10000)
    testshares = [3, 11, 86, 8, 4, 103, 15]
    @test lpAlloc.shares == testshares

    gAlloc, remaining =
        Allocation(Greedy(), cv, Vector(df[end, cv.tickers]); investment = 10000)
    testshares = [102, 86, 11, 16, 8, 3, 4]
    @test gAlloc.shares == testshares

    efficient_return!(cv, 0.07)
    testweights = [
        0.3839859145444188,
        0.0359698109636541,
        6.78e-14,
        7.7866e-12,
        6.22e-13,
        9.53e-14,
        8.145e-13,
        8.463e-13,
        0.4466960681039454,
        2.409e-13,
        3.971e-13,
        1.291e-13,
        4.19e-14,
        0.1071394799383709,
        6.6342e-12,
        4.764e-13,
        3.925e-13,
        0.0262087264251456,
        5.954e-12,
        -3.32e-14,
    ]
    isapprox(cv.weights, testweights)
    mu, sigma = portfolio_performance(cv)
    mutest, sigmatest = 0.06999999999999391, 0.028255572821056847
    isapprox(mu, mutest, rtol = 1e-7)
    isapprox(sigma, sigmatest, rtol = 1e-4)

    cv.weights .= testweights
    lpAlloc, remaining =
        Allocation(LP(), cv, Vector(df[end, cv.tickers]); investment = 10000)
    testshares = [4, 2, 147, 12, 7]
    @test lpAlloc.shares == testshares

    gAlloc, remaining =
        Allocation(Greedy(), cv, Vector(df[end, cv.tickers]); investment = 10000)
    testshares = [149, 3, 14, 3, 8, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    @test gAlloc.shares == testshares

    efficient_return!(cv, 0.07)
    testweights = [
        0.3839859145444188,
        0.0359698109636541,
        6.78e-14,
        7.7866e-12,
        6.22e-13,
        9.53e-14,
        8.145e-13,
        8.463e-13,
        0.4466960681039454,
        2.409e-13,
        3.971e-13,
        1.291e-13,
        4.19e-14,
        0.1071394799383709,
        6.6342e-12,
        4.764e-13,
        3.925e-13,
        0.0262087264251456,
        5.954e-12,
        -3.32e-14,
    ]
    @test isapprox(cv.weights, testweights, rtol = 1e-3)
    mu, sigma = portfolio_performance(cv)
    mutest, sigmatest = 0.06999999999999391, 0.028255572821056847
    @test isapprox(mu, mutest, rtol = 1e-5)
    @test isapprox(sigma, sigmatest, rtol = 1e-4)

    cv.weights .= testweights
    lpAlloc, remaining =
        Allocation(LP(), cv, Vector(df[end, cv.tickers]); investment = 10000)
    testshares = [4, 2, 147, 12, 7]
    @test lpAlloc.shares == testshares

    gAlloc, remaining =
        Allocation(Greedy(), cv, Vector(df[end, cv.tickers]); investment = 10000)
    testshares = [149, 3, 14, 3, 8, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    @test gAlloc.shares == testshares

    efficient_risk!(cv, 0.14813)
    testweights = [
        2.85e-14,
        1.58e-14,
        3.5e-15,
        -5.3e-15,
        2.06e-14,
        1.3e-15,
        3.591e-13,
        9.2e-15,
        0.9999999999994234,
        1.31e-14,
        6.9e-15,
        0.0,
        -5.4e-15,
        1.53e-14,
        8.6e-15,
        9.6e-15,
        1.8e-14,
        1.95e-14,
        6.73e-14,
        -9.4e-15,
    ]
    @test isapprox(cv.weights, testweights, rtol = 1e-6)
    mu, sigma = portfolio_performance(cv)
    mutest, sigmatest = 0.09141921970336199, 0.148136422458718
    @test isapprox(mu, mutest, rtol = 1e-6)
    @test isapprox(sigma, sigmatest, rtol = 1e-3)

    cv.weights .= testweights
    lpAlloc, remaining =
        Allocation(LP(), cv, Vector(df[end, cv.tickers]); investment = 10000)
    testshares = [334]
    @test lpAlloc.shares == testshares

    gAlloc, remaining =
        Allocation(Greedy(), cv, Vector(df[end, cv.tickers]); investment = 10000)
    testshares = [334, 1]
    @test gAlloc.shares == testshares

    cv = EfficientCVaR(tickers, bl.post_ret, Matrix(returns), market_neutral = true)
    min_cvar!(cv)
    testweights = [
        -2.2800e-12,
        -5.5010e-13,
        5.6220e-13,
        -3.1620e-13,
        5.1780e-12,
        -4.6641e-12,
        8.1920e-13,
        -1.2637e-12,
        -2.9427e-12,
        2.0640e-13,
        3.8680e-13,
        -1.8054e-12,
        -5.7220e-13,
        -3.0342e-12,
        -1.6556e-12,
        1.6296e-12,
        2.5777e-12,
        -4.7750e-13,
        8.4177e-12,
        -2.1610e-13,
    ]
    @test isapprox(cv.weights, testweights, atol = 1e-4)
    mu, sigma = portfolio_performance(cv)
    mutest, sigmatest = 3.279006735579045e-13, 5.19535240389637e-13
    @test isapprox(mu, mutest, atol = 1e-1)
    @test isapprox(sigma, sigmatest, atol = 1e-1)

    cv.weights .= testweights
    lpAlloc, remaining =
        Allocation(LP(), cv, Vector(df[end, cv.tickers]); investment = 10000)
    testshares = [2, 2, 41, 2, 4, 11, 7, 38]
    @test lpAlloc.shares == testshares

    gAlloc, remaining =
        Allocation(Greedy(), cv, Vector(df[end, cv.tickers]); investment = 10000)
    testshares = [38, 2, 7, 12, 43, 1, 5, 3]
    @test gAlloc.shares == testshares

    efficient_return!(cv, 0.07)
    testweights = [
        0.119904777207769,
        0.105260613527609,
        -0.059118345889513,
        0.047994829008171,
        0.080535064283592,
        -0.07038968929214,
        -0.000695786742806,
        0.054006019393697,
        0.088529728836899,
        -0.01637772952951,
        -0.063653950435258,
        0.012986548220159,
        0.019165300733247,
        0.021582076971202,
        -0.00296786688154,
        0.018096675589053,
        -0.024160465458667,
        0.072693909074444,
        0.137412159501253,
        -0.540803868117662,
    ]
    @test isapprox(cv.weights, testweights, rtol = 1e-3)
    mu, sigma = portfolio_performance(cv)
    mutest, sigmatest = 0.07000000000000023, 0.01208285385909055
    @test isapprox(mu, mutest, rtol = 1e-7)
    @test isapprox(sigma, sigmatest, rtol = 1e-3)

    cv.weights .= testweights
    lpAlloc, remaining =
        Allocation(LP(), cv, Vector(df[end, cv.tickers]); investment = 10000)
    testshares =
        [1, 8, 4, 1, 8, 38, 10, 75, 4, 3, 26, 16, -4, -54, -1, -4, -18, -2, -1, -91]
    @test rmsd(lpAlloc.shares, testshares) < 0.5

    gAlloc, remaining =
        Allocation(Greedy(), cv, Vector(df[end, cv.tickers]); investment = 10000)
    testshares =
        [16, 1, 8, 38, 1, 26, 8, 4, 4, 75, 3, 10, -91, -54, -18, -4, -1, -4, -2, -1]
    @test gAlloc.shares == testshares

    efficient_risk!(cv, 0.18)
    testweights = [
        0.999999999998078,
        0.999999999994987,
        -0.999999999998141,
        0.999999999996761,
        0.999999999951026,
        -0.9999999999986,
        0.999999999934693,
        -0.999999999995853,
        0.999999999999143,
        -0.999999998594101,
        -0.999999999998166,
        -0.999999999986599,
        -0.83525041495262,
        0.999999999989784,
        0.709272218478147,
        -0.163990750353505,
        -0.345299781283349,
        -0.364731273180876,
        0.999999999998686,
        -0.999999999999495,
    ]
    @test isapprox(cv.weights, testweights, rtol = 1e-4)
    mu, sigma = portfolio_performance(cv)
    mutest, sigmatest = 0.4902004788284181, 0.1799999999999532
    @test isapprox(mu, mutest, rtol = 1e-5)
    @test isapprox(sigma, sigmatest, rtol = 1e-7)

    cv.weights .= testweights
    lpAlloc, remaining =
        Allocation(LP(), cv, Vector(df[end, cv.tickers]); investment = 10000)
    testshares = [
        1,
        7,
        6,
        1,
        117,
        38,
        14,
        54,
        10,
        -60,
        -771,
        -117,
        -256,
        -284,
        -597,
        -2533,
        -23,
        -20,
        -102,
        -168,
    ]
    @test rmsd(lpAlloc.shares, testshares) < 0.5

    gAlloc, remaining =
        Allocation(Greedy(), cv, Vector(df[end, cv.tickers]); investment = 10000)
    testshares = [
        38,
        10,
        1,
        6,
        7,
        14,
        1,
        117,
        54,
        -168,
        -771,
        -284,
        -60,
        -117,
        -597,
        -256,
        -2531,
        -102,
        -20,
        -23,
    ]
    @test gAlloc.shares == testshares

    cv = EfficientCVaR(
        tickers,
        bl.post_ret,
        Matrix(returns),
        beta = 0.2,
        market_neutral = false,
    )
    min_cvar!(cv)
    testweights = [
        1.2591e-12,
        0.0239999111487575,
        0.0650098845478511,
        0.0531407521736819,
        0.0990996575024338,
        3.624e-13,
        0.0106099759402764,
        0.1031880979043503,
        8.525e-13,
        0.0238096685816571,
        0.2741982381970403,
        8.64e-14,
        6.49e-14,
        0.0021107467292905,
        1.032e-13,
        0.0302164465597255,
        0.053838138362372,
        0.1363179609151427,
        0.0210516374364299,
        0.1034088839982625,
    ]
    @test isapprox(cv.weights, testweights, rtol = 1e-2)
    mu, cvar = portfolio_performance(cv)
    mutest, cvartest = 0.015141313656655424, 0.0020571077411184347
    @test isapprox(mu, mutest, rtol = 1e-2)
    @test isapprox(cvar, cvartest, rtol = 1e-2)

    efficient_return!(cv, 0.09)
    testweights = [
        5.398e-13,
        1.53e-14,
        2.9e-15,
        0.0208931899503421,
        2.98e-14,
        1.4e-15,
        0.0561037656427129,
        3.6e-15,
        0.9230030444062806,
        4.5e-15,
        2.6e-15,
        1.9e-15,
        7e-16,
        7e-15,
        3.9e-15,
        4.1e-15,
        8e-15,
        6.5e-15,
        3.2e-14,
        2e-16,
    ]
    @test isapprox(cv.weights, testweights, rtol = 1e-3)
    mu, cvar = portfolio_performance(cv)
    mutest, cvartest = 0.09000000000000004, 0.004545680961201856
    @test isapprox(mu, mutest, rtol = 1e-6)
    @test isapprox(cvar, cvartest, rtol = 1e-3)

    efficient_risk!(cv, 0.1438)
    testweights = [
        2.1e-14,
        1.09e-14,
        -2.19e-14,
        5.82e-14,
        -1.67e-14,
        -1.52e-14,
        4.488e-12,
        -3.38e-14,
        0.9999999999954172,
        1.4e-15,
        -3.02e-14,
        -4.3e-15,
        -1.04e-14,
        6.9e-15,
        9.49e-14,
        -1.49e-14,
        -5.5e-15,
        -1.65e-14,
        8.64e-14,
        -1.56e-14,
    ]
    @test isapprox(cv.weights, testweights, rtol = 1e-6)
    mu, cvar = portfolio_performance(cv)
    mutest, cvartest = 0.09141921970332006, 0.14377337204585958
    @test isapprox(mu, mutest, rtol = 1e-6)
    @test isapprox(cvar, cvartest, rtol = 1e-3)

    cv = EfficientCVaR(
        tickers,
        bl.post_ret,
        Matrix(returns),
        beta = 1,
        market_neutral = false,
    )
    @test (0, 0) == portfolio_performance(cv)
    @test cv.beta == 0.95
    cv = EfficientCVaR(
        tickers,
        bl.post_ret,
        Matrix(returns),
        beta = -0.1,
        market_neutral = false,
    )
    @test cv.beta == 0.95

    mean_ret = ret_model(MRet(), Matrix(returns))
    S = risk_model(Cov(), Matrix(returns))
    n = length(tickers)
    prev_weights = fill(1 / n, n)

    k = 0.0001
    ef = EfficientCVaR(
        tickers,
        mean_ret,
        Matrix(returns);
        extra_vars = [:(0 <= l1)],
        extra_constraints = [
            :([model[:l1]; (model[:w] - $prev_weights)] in MOI.NormOneCone($(n + 1))),
            # :(model[:w][6] == 0.2),
            # :(model[:w][1] >= 0.01),
            # :(model[:w][16] <= 0.03),
        ],
        extra_obj_terms = [quote
            $k * model[:l1]
        end],
    )
    min_cvar!(ef)
    testweights = [
        1.0892e-12,
        0.0500000000076794,
        1.0671e-12,
        3.4685e-12,
        0.0071462860549324,
        1.4497e-12,
        1.053e-13,
        0.09689425274277,
        5.118e-13,
        6.801e-13,
        0.306535765681518,
        6.189e-13,
        3.304e-13,
        0.0681247368437245,
        1.7277e-12,
        0.0285905010302666,
        1.3816e-12,
        0.3642755366704354,
        6.405e-13,
        0.078432920955603,
    ]
    @test isapprox(ef.weights, testweights, rtol = 1e-3)

    efficient_return!(ef, 0.11)
    testweights = [
        1.338e-13,
        0.0365748381251478,
        1.341e-13,
        6.271e-13,
        0.0612399946270939,
        1.195e-13,
        2.08e-14,
        0.1040143575055467,
        7.79e-14,
        5.98e-14,
        0.2563667017115727,
        6.27e-14,
        1.88e-14,
        0.0529676397826387,
        7.08e-14,
        0.0313099048708647,
        3.72e-13,
        0.3619202934010016,
        1.261e-13,
        0.0956062699743107,
    ]
    @test isapprox(ef.weights, testweights, rtol = 1e-3)

    efficient_risk!(ef, 0.03)
    testweights = [
        5.7e-15,
        2.17e-14,
        1.59e-14,
        -5e-16,
        0.6048157267914637,
        -1.08e-14,
        -2e-16,
        1.147e-13,
        1.22e-14,
        0.0,
        6.89e-14,
        -1.44e-14,
        -1.67e-14,
        1e-15,
        -1.59e-14,
        0.1658143853764952,
        0.2293698878316262,
        1.107e-13,
        8.55e-14,
        3.65e-14,
    ]
    @test isapprox(ef.weights, testweights, rtol = 1e-5)

    cv = EfficientCVaR(tickers, mean_ret, Matrix(returns))

    max_quadratic_utility!(cv, 1e8)
    mu, cvar = portfolio_performance(cv)

    min_cvar!(cv)
    muinf, cvarinf = portfolio_performance(cv)
    @test isapprox(mu, muinf, rtol = 1e-3)
    @test isapprox(cvar, cvarinf, rtol = 1e-3)

    max_quadratic_utility!(cv, 0.25)
    mu1, cvar1 = portfolio_performance(cv)
    max_quadratic_utility!(cv, 0.5)
    mu2, cvar2 = portfolio_performance(cv)
    max_quadratic_utility!(cv, 1)
    mu3, cvar3 = portfolio_performance(cv)
    max_quadratic_utility!(cv, 2)
    mu4, cvar4 = portfolio_performance(cv)
    max_quadratic_utility!(cv, 4)
    mu5, cvar5 = portfolio_performance(cv)
    max_quadratic_utility!(cv, 8)
    mu6, cvar6 = portfolio_performance(cv)
    max_quadratic_utility!(cv, 16)
    mu7, cvar7 = portfolio_performance(cv)

    @test cvar1 > cvar2 > cvar3 > cvar4 > cvar5 > cvar6 > cvar7 > cvarinf

    mean_ret = ret_model(MRet(), Matrix(returns))
    S = risk_model(Cov(), Matrix(returns))
    n = length(tickers)
    prev_weights = fill(1 / n, n)
    k = 0.001
    cv = EfficientCVaR(
        tickers,
        mean_ret,
        Matrix(returns);
        extra_vars = [:(0 <= l1)],
        extra_constraints = [
            :([model[:l1]; (model[:w] - $prev_weights)] in MOI.NormOneCone($(n + 1))),
            # :(model[:w][6] == 0.2),
            # :(model[:w][1] >= 0.01),
            # :(model[:w][16] <= 0.03),
        ],
        extra_obj_terms = [quote
            $k * model[:l1]
        end],
    )
    max_quadratic_utility!(cv)
    testweights = [
        3.113756569142173e-6
        0.03651110428668516
        1.7559327229467127e-6
        2.4988508813368607e-6
        0.2884507824236885
        4.740307839121232e-7
        6.35965677451175e-7
        0.059851753966193834
        1.2328589244485448e-6
        7.729653606291163e-7
        0.23925600103724068
        2.821757987240221e-7
        2.236857664879175e-7
        2.4956855515063193e-6
        2.3603525249476997e-7
        0.0499988249909567
        0.05000484516920409
        0.18054754949754553
        0.045424798952762205
        0.04994061773243416
    ]
    @test isapprox(cv.weights, testweights)
end

@testset "Efficient CDaR" begin
    df = CSV.read("./assets/stock_prices.csv", DataFrame)
    dropmissing!(df)
    returns = returns_from_prices(df[!, 2:end])
    tickers = names(df)[2:end]

    mu = vec(ret_model(MRet(), Matrix(returns)))
    S = risk_model(Cov(), Matrix(returns))

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

    cdar = EfficientCDaR(tickers, bl.post_ret, Matrix(returns))
    @test (0, 0) == portfolio_performance(cdar)
    min_cdar!(cdar)
    testweights = [
        5.500000000000000e-15,
        4.210000000000000e-13,
        1.010000000000000e-14,
        4.000000000000000e-16,
        3.409901158173600e-03,
        -3.300000000000000e-15,
        -1.070000000000000e-14,
        7.904282392519640e-02,
        1.250000000000000e-14,
        -2.900000000000000e-15,
        3.875931701996790e-01,
        4.740000000000000e-14,
        6.370000000000000e-14,
        1.063000000000000e-13,
        5.545152701592000e-04,
        9.598828930536250e-02,
        2.679088825496902e-01,
        1.995000000000000e-13,
        6.560171960161000e-04,
        1.648464003948736e-01,
    ]
    @test isapprox(cdar.weights, testweights, rtol = 1e-3)
    mu, sigma = portfolio_performance(cdar)
    mutest, sigmatest = 0.0046414239877397775, 0.05643312227060557
    @test isapprox(mu, mutest, rtol = 1e-2)
    @test isapprox(sigma, sigmatest, rtol = 1e-4)

    cdar.weights .= testweights
    lpAlloc, remaining =
        Allocation(LP(), cdar, Vector(df[end, cdar.tickers]); investment = 10000)
    testshares = [9, 110, 13, 16, 28]
    @test lpAlloc.shares == testshares

    gAlloc, remaining =
        Allocation(Greedy(), cdar, Vector(df[end, cdar.tickers]); investment = 10000)
    testshares = [109, 16, 27, 14, 9, 1, 1]
    @test gAlloc.shares == testshares

    efficient_return!(cdar, 0.071)
    testweights = [
        1.803135468327820e-01,
        -7.000000000000000e-16,
        -1.100000000000000e-15,
        -8.000000000000000e-16,
        -9.000000000000000e-16,
        -1.200000000000000e-15,
        3.736518284742810e-02,
        -8.000000000000000e-16,
        1.446130925597511e-01,
        -1.100000000000000e-15,
        -1.000000000000000e-15,
        -1.100000000000000e-15,
        -1.200000000000000e-15,
        5.000000000000000e-16,
        3.035022274167050e-02,
        -9.000000000000000e-16,
        -9.000000000000000e-16,
        -9.000000000000000e-16,
        6.073579550183820e-01,
        -1.300000000000000e-15,
    ]
    @test isapprox(cdar.weights, testweights, rtol = 1e-4)
    mu, sigma = portfolio_performance(cdar, verbose = true)
    mutest, sigmatest = 0.07099999999999985, 0.14924652616273293
    @test isapprox(mu, mutest, rtol = 1e-6)
    @test isapprox(sigma, sigmatest, rtol = 1e-5)

    cdar.weights .= testweights
    lpAlloc, remaining =
        Allocation(LP(), cdar, Vector(df[end, cdar.tickers]); investment = 10000)
    testshares = [2, 27, 48, 19, 54]
    @test lpAlloc.shares == testshares

    gAlloc, remaining =
        Allocation(Greedy(), cdar, Vector(df[end, cdar.tickers]); investment = 10000)
    testshares = [54, 1, 48, 38, 20, 1]
    @test gAlloc.shares == testshares

    efficient_risk!(cdar, 0.11)
    testweights = [
        3.411720847078938e-01,
        4.200000000000000e-15,
        6.000000000000000e-16,
        -4.000000000000000e-16,
        3.050000000000000e-14,
        -7.000000000000000e-16,
        1.300000000000000e-14,
        3.767304612523230e-02,
        6.100000000000000e-15,
        -2.000000000000000e-16,
        2.916000000000000e-13,
        6.210000000000000e-14,
        2.300000000000000e-15,
        6.480796061687440e-02,
        1.261914308485920e-02,
        3.719518575826500e-02,
        1.120000000000000e-14,
        6.700000000000000e-15,
        5.065325797064497e-01,
        -1.400000000000000e-15,
    ]
    @test isapprox(cdar.weights, testweights, rtol = 1e-3)
    mu, sigma = portfolio_performance(cdar)
    mutest, sigmatest = 0.060150020563327425, 0.11000000000000373
    @test isapprox(mu, mutest, rtol = 1e-4)
    @test isapprox(sigma, sigmatest, rtol = 1e-7)

    cdar.weights .= testweights
    lpAlloc, remaining =
        Allocation(LP(), cdar, Vector(df[end, cdar.tickers]); investment = 10000)
    testshares = [3, 6, 9, 10, 6, 46]
    @test lpAlloc.shares == testshares

    gAlloc, remaining =
        Allocation(Greedy(), cdar, Vector(df[end, cdar.tickers]); investment = 10000)
    testshares = [45, 3, 9, 5, 6, 9, 1, 1, 1, 1, 1, 1]
    @test gAlloc.shares == testshares

    cdar = EfficientCDaR(tickers, bl.post_ret, Matrix(returns), market_neutral = true)
    min_cdar!(cdar)
    testweights = [
        -7.8660e-13,
        -5.1690e-13,
        5.5350e-13,
        2.5550e-13,
        1.0952e-12,
        -1.6305e-12,
        2.7030e-13,
        -2.7810e-13,
        -1.7863e-12,
        -1.7250e-13,
        3.4360e-13,
        -4.9670e-13,
        -1.1400e-13,
        -6.5110e-13,
        -2.9790e-13,
        5.5730e-13,
        4.9150e-13,
        -6.2820e-13,
        2.9461e-12,
        8.4570e-13,
    ]
    @test isapprox(cdar.weights, testweights, atol = 1e-3)
    mu, sigma = portfolio_performance(cdar)
    mutest, sigmatest = -3.270061253855585e-14, 2.788899603525084e-13
    @test isapprox(mu, mutest, atol = 1e-5)
    @test isapprox(sigma, sigmatest, atol = 1e-5)

    cdar.weights .= testweights
    lpAlloc, remaining =
        Allocation(LP(), cdar, Vector(df[end, cdar.tickers]); investment = 10000)
    testshares = [5, 2, 1, 35, 13, 11, 4, 36, 19]
    @test lpAlloc.shares == testshares

    gAlloc, remaining =
        Allocation(Greedy(), cdar, Vector(df[end, cdar.tickers]); investment = 10000)
    testshares = [36, 1, 19, 10, 5, 4, 14, 38, 2]
    @test gAlloc.shares == testshares

    efficient_return!(cdar, 0.071)
    testweights = [
        0.025107592782914,
        0.057594547532327,
        -0.029106830395298,
        -0.055379879544535,
        0.277094131770879,
        -0.218125545473876,
        0.01760029260404,
        0.033400679475767,
        -0.361271598572642,
        -0.088479023848706,
        -0.184621283291016,
        -0.023896668602229,
        -0.006314246903803,
        0.277651628420218,
        -0.069684404622742,
        0.084480397324026,
        0.11757646255151,
        -0.008078698917481,
        0.690348263568902,
        -0.535895815858256,
    ]
    @test isapprox(cdar.weights, testweights, rtol = 1e-4)
    mu, sigma = portfolio_performance(cdar)
    mutest, sigmatest = 0.07099999999999981, 0.06860632727537126
    @test isapprox(mu, mutest, rtol = 1e-6)
    @test isapprox(sigma, sigmatest, rtol = 1e-4)

    cdar.weights .= testweights
    lpAlloc, remaining =
        Allocation(LP(), cdar, Vector(df[end, cdar.tickers]); investment = 10000)
    testshares =
        [3, 1, 12, 3, 23, 8, 5, 40, -2, -3, -168, -121, -23, -52, -14, -19, -47, -2, -90]
    @test rmsd(lpAlloc.shares, testshares) < 1

    gAlloc, remaining =
        Allocation(Greedy(), cdar, Vector(df[end, cdar.tickers]); investment = 10000)
    testshares =
        [39, 23, 1, 5, 8, 3, 3, 12, -90, -121, -168, -52, -23, -47, -3, -2, -14, -2, -19]
    @test gAlloc.shares == testshares

    efficient_risk!(cdar, 0.11)
    testweights = [
        0.028783758447304,
        0.154930722945264,
        -0.052888584214774,
        -0.090606438318005,
        0.438460257717241,
        -0.34165630482699,
        0.02387460049364,
        0.054720588895777,
        -0.519576218315878,
        -0.132186114282055,
        -0.232431229341235,
        -0.006075698448925,
        -0.022163078215555,
        0.434087117994023,
        -0.130403026894326,
        0.14155104241332,
        0.175044671976498,
        -0.036231017136812,
        0.999999999999363,
        -0.887235050887876,
    ]
    @test isapprox(cdar.weights, testweights, rtol = 1e-4)
    mu, sigma = portfolio_performance(cdar)
    mutest, sigmatest = 0.11347395924861056, 0.11000000000006227
    @test isapprox(mu, mutest, rtol = 1e-5)
    @test isapprox(sigma, sigmatest, rtol = 1e-7)

    cdar.weights .= testweights
    lpAlloc, remaining =
        Allocation(LP(), cdar, Vector(df[end, cdar.tickers]); investment = 10000)
    testshares =
        [4, 1, 10, 3, 23, 9, 5, 37, -3, -5, -263, -174, -34, -66, -4, -68, -87, -10, -150]
    @test rmsd(lpAlloc.shares, testshares) < 0.5

    gAlloc, remaining =
        Allocation(Greedy(), cdar, Vector(df[end, cdar.tickers]); investment = 10000)
    testshares =
        [36, 1, 22, 4, 4, 8, 3, 10, -149, -173, -263, -66, -33, -86, -5, -4, -10, -67, -3]
    @test gAlloc.shares == testshares

    cdar = EfficientCDaR(tickers, bl.post_ret, Matrix(returns), beta = 0.2)
    min_cdar!(cdar)
    testweights = [
        1.229e-13,
        0.0444141271754544,
        0.1587818250703651,
        1.98e-14,
        0.0678958478066876,
        6.64e-14,
        2.56e-14,
        0.061989718949941,
        3.84e-14,
        6.16e-14,
        0.2102532269364574,
        2.378e-13,
        0.0012132758140646,
        0.01721619579299,
        2.7e-15,
        0.0690692111656633,
        0.1014609071930568,
        0.0224777909586333,
        0.1303610329941893,
        0.1148668401419223,
    ]
    @test isapprox(cdar.weights, testweights, rtol = 1e-2)
    mu, cvar = portfolio_performance(cdar)
    mutest, cvartest = 0.01656029948191691, 0.017697217976780904
    @test isapprox(mu, mutest, rtol = 1e-3)
    @test isapprox(cvar, cvartest, rtol = 1e-3)

    efficient_return!(cdar, 0.08)
    testweights = [
        0.098647832481146,
        0.0477105526357058,
        -3.1e-15,
        2.73e-14,
        0.0622008742187827,
        -4e-15,
        0.088263264789127,
        -2.4e-15,
        0.6634843793037455,
        -2.8e-15,
        -3.3e-15,
        -3e-15,
        -2.5e-15,
        -4e-16,
        0.0396930965712687,
        3.4e-15,
        1e-15,
        -9e-16,
        2.195e-13,
        -4.6e-15,
    ]
    @test isapprox(cdar.weights, testweights, rtol = 1e-3)
    mu, cvar = portfolio_performance(cdar)
    mutest, cvartest = 0.07999999999999943, 0.05673303047360887
    @test isapprox(mu, mutest, rtol = 1e-6)
    @test isapprox(cvar, cvartest, rtol = 1e-4)

    efficient_risk!(cdar, 0.06)
    testweights = [
        0.0836985709866768,
        0.0280783943809699,
        -4.4e-15,
        4.527e-13,
        0.0561853909940908,
        -1.24e-14,
        0.0933927094981601,
        2.2e-15,
        0.6995287341291265,
        -1.3e-15,
        -6.6e-15,
        -3.5e-15,
        1.9e-15,
        1.89e-14,
        0.0391162000087298,
        5.46e-14,
        3.16e-14,
        1.55e-14,
        1.7186e-12,
        -1.94e-14,
    ]
    @test isapprox(cdar.weights, testweights, rtol = 1e-2)
    mu, cvar = portfolio_performance(cdar)
    mutest, cvartest = 0.08164234341670347, 0.06000000000004213
    @test isapprox(mu, mutest, rtol = 1e-4)
    @test isapprox(cvar, cvartest, rtol = 1e-7)

    cdar = EfficientCDaR(tickers, bl.post_ret, Matrix(returns), beta = 1)
    @test (0, 0) == portfolio_performance(cdar)
    @test cdar.beta == 0.95
    cdar = EfficientCDaR(
        tickers,
        bl.post_ret,
        Matrix(returns),
        beta = -0.1,
        market_neutral = false,
    )
    @test cdar.beta == 0.95

    mean_ret = ret_model(MRet(), Matrix(returns))
    S = risk_model(Cov(), Matrix(returns))
    n = length(tickers)
    prev_weights = fill(1 / n, n)

    k = 0.0001
    ef = EfficientCDaR(
        tickers,
        mean_ret,
        Matrix(returns);
        extra_vars = [:(0 <= l1)],
        extra_constraints = [
            :([model[:l1]; (model[:w] - $prev_weights)] in MOI.NormOneCone($(n + 1))),
            # :(model[:w][6] == 0.2),
            # :(model[:w][1] >= 0.01),
            # :(model[:w][16] <= 0.03),
        ],
        extra_obj_terms = [quote
            $k * model[:l1]
        end],
    )
    min_cdar!(ef)
    testweights = [
        5.8e-15,
        5.127e-13,
        1.02e-14,
        5e-16,
        0.0035594792687726,
        -3e-15,
        -1.04e-14,
        0.079973068549514,
        1.26e-14,
        -2.8e-15,
        0.3871598884148665,
        5.02e-14,
        6.05e-14,
        1.133e-13,
        0.0002218291232121,
        0.0965257689615006,
        0.2659397480643742,
        2.058e-13,
        0.0019205861934667,
        0.164699631423338,
    ]
    @test isapprox(ef.weights, testweights, rtol = 1e-3)

    efficient_return!(ef, 0.17)
    testweights = [
        1e-16,
        2e-15,
        1.23e-14,
        -7e-16,
        0.0819555819215858,
        -9e-16,
        -1.1e-15,
        0.1018672906381434,
        1e-16,
        -6e-16,
        0.3566693544811642,
        -9e-16,
        -1e-15,
        1.3e-15,
        -8e-16,
        0.12248240804338,
        0.1788992614241954,
        8e-16,
        6.76e-14,
        0.1581261034914527,
    ]
    @test isapprox(ef.weights, testweights, rtol = 1e-4)

    efficient_risk!(ef, 0.09)
    testweights = [
        -3.72e-14,
        -2.56e-14,
        0.0968865336134717,
        -5.39e-14,
        0.382879247996457,
        -5.87e-14,
        -5.97e-14,
        0.1747389572731268,
        -5.47e-14,
        -5.31e-14,
        0.1223279091478751,
        -5.75e-14,
        -6.29e-14,
        -3.87e-14,
        -5.69e-14,
        0.2231673519674347,
        4.01e-14,
        -3.27e-14,
        -2.96e-14,
        2.216e-12,
    ]
    @test isapprox(ef.weights, testweights, rtol = 1e-4)

    cd = EfficientCDaR(tickers, mean_ret, Matrix(returns))

    max_quadratic_utility!(cd, 1e8)
    mu, cdar = portfolio_performance(cd)
    min_cdar!(cd)
    muinf, cdarinf = portfolio_performance(cd)
    @test isapprox(mu, muinf, rtol = 1e-3)
    @test isapprox(cdar, cdarinf, rtol = 1e-3)

    max_quadratic_utility!(cd, 0.25)
    mu1, cdar1 = portfolio_performance(cd)
    max_quadratic_utility!(cd, 0.5)
    mu2, cdar2 = portfolio_performance(cd)
    max_quadratic_utility!(cd, 1)
    mu3, cdar3 = portfolio_performance(cd)
    max_quadratic_utility!(cd, 2)
    mu4, cdar4 = portfolio_performance(cd)
    max_quadratic_utility!(cd, 4)
    mu5, cdar5 = portfolio_performance(cd)
    max_quadratic_utility!(cd, 8)
    mu6, cdar6 = portfolio_performance(cd)
    max_quadratic_utility!(cd, 16)
    mu7, cdar7 = portfolio_performance(cd)

    @test cdar1 > cdar2 > cdar3 > cdar4 > cdar5 > cdar6 > cdar7

    k = 0.001
    cd = EfficientCDaR(
        tickers,
        mean_ret,
        Matrix(returns);
        extra_vars = [:(0 <= l1)],
        extra_constraints = [
            :([model[:l1]; (model[:w] - $prev_weights)] in MOI.NormOneCone($(n + 1))),
            # :(model[:w][6] == 0.2),
            # :(model[:w][1] >= 0.01),
            # :(model[:w][16] <= 0.03),
        ],
        extra_obj_terms = [quote
            $k * model[:l1]
        end],
    )
    max_quadratic_utility!(cd)
    testweights = [
        2.4310205346857125e-7
        1.3076816436003056e-6
        5.128595922385004e-7
        1.3722438999845795e-7
        0.01602055169559408
        1.0376808957724839e-7
        3.0145546986152965e-8
        0.07460084218292454
        2.565000474740614e-7
        1.2369417424546358e-7
        0.38921831557277276
        2.0060734523276934e-7
        1.2113214134486977e-7
        1.1858065328182865e-6
        1.7026638199337542e-7
        0.1169266302620751
        0.2495346328937271
        6.678082003005919e-6
        0.0009745300671170129
        0.15271342645584743
    ]
    @test isapprox(cd.weights, testweights)
end