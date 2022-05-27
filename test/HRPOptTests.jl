using Test

using PortfolioOptimiser, CSV, DataFrames, Statistics, LinearAlgebra

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

    hrp = HRPOpt(
        tickers,
        returns = Matrix(returns),
        D = Symmetric(sqrt.(clamp.((1 .- cov2cor(cov(Matrix(returns)))) / 2, 0, 1))),
    )
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
    @test_throws ArgumentError HRPOpt(tickers, returns = Matrix(returns), D = :custom)

    hrp = HRPOpt(tickers, returns = Matrix(returns), risk_aversion = 1)
    optimise!(hrp, max_quadratic_utility)
    testweights = [
        0.053786684146864876,
        -0.41265611750315045,
        0.09686673404189124,
        0.08257285280695957,
        0.14492069705070706,
        0.4073098824162846,
        0.06828156176492603,
        -0.010348133086424266,
        0.056139267832765134,
        0.06197799306215815,
        0.024927864699722278,
        0.035677589026737885,
        -0.15478152888511781,
        -0.013222509868836205,
        0.14335770434850123,
        0.24250548421907214,
        0.1476799007815831,
        -0.013608071045093663,
        0.06258638595359726,
        -0.023974241763148114,
    ]
    @test hrp.weights ≈ testweights
    mu, sigma, sr = portfolio_performance(hrp, verbose = true)
    mutest, sigmatest, srtest = 0.1501811670967654, 0.23503576141992305, 0.5538781260787767
    @test mu ≈ mutest
    @test sigma ≈ sigmatest
    @test sr ≈ srtest

    hrp = HRPOpt(tickers, returns = Matrix(returns))
    optimise!(hrp, max_return)
    testweights = [
        0.05013407346217203,
        0.048276502560897244,
        0.09408501541630035,
        0.0788117358003851,
        0.1267946493836637,
        -0.035564734092364235,
        0.3600343922715497,
        -0.03072369999576692,
        0.05720258969644472,
        0.06924370086105518,
        0.023107676851654508,
        0.04308557677784081,
        -0.26203476517294183,
        -0.004221459422587845,
        0.16760793372492894,
        0.08377045535965326,
        0.13862447845231057,
        -0.025212700319432475,
        0.05952526315294496,
        -0.04254668476870775,
    ]
    @test hrp.weights ≈ testweights
    mu, sigma, sr = portfolio_performance(hrp, verbose = true)
    mutest, sigmatest, srtest = 0.40298014693544526, 0.35199194401087913, 1.0880366822361376
    @test mu ≈ mutest
    @test sigma ≈ sigmatest
    @test sr ≈ srtest

    hrp = HRPOpt(tickers, returns = Matrix(returns))
    optimise!(hrp, max_sharpe)
    testweights = [
        0.03141171260716669
        0.05607323388592659
        0.049991283854799964
        0.05331560720461475
        0.023606797993473632
        0.06520423495686638
        0.029557095278713925
        0.0703731377181911
        0.021880093246291782
        0.052179566737008694
        0.10742256946559307
        0.03513599666560551
        0.0319955788667069
        0.07048502012137124
        0.04090318596288987
        0.05505181579640627
        0.05443134061322866
        0.06656069708539113
        0.026694897990971828
        0.05772613394878202
    ]
    @test hrp.weights ≈ testweights
    mu, sigma, sr = portfolio_performance(hrp, verbose = true)
    mutest, sigmatest, srtest = 0.10230747789346417, 0.14487879024359007, 0.5681126806420564
    @test mu ≈ mutest
    @test sigma ≈ sigmatest
    @test sr ≈ srtest
end