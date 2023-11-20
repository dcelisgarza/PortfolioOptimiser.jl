using Test

using PortfolioOptimiser, CSV, DataFrames, Statistics, LinearAlgebra

@testset "HRP Opt" begin
    tickers = ["AAPL", "GOOG", "AMD", "MSFT"]
    n = length(tickers)
    ret = (randn(1000, n) .+ 0.01) / 100
    ret[:, 1] = ret[:, 2] .+ 1
    cov_mtx = cov(ret) * 252
    hropt = HRPOpt(tickers, cov_mtx = cov_mtx)
    optimise!(hropt, min_risk!)
    @test hropt.weights[1] ≈ hropt.weights[2]
    mu, sigma, sr = portfolio_performance(hropt, verbose = true)
    @test isnan(mu)
    @test isnan(sr)

    k = 5
    ret = (randn(1000, n) .+ 0.01) / 100
    ret[:, 2] = k * ret[:, 1]
    cov_mtx = cov(ret) * 252

    hropt = HRPOpt(tickers, returns = ret, cov_mtx = cov_mtx)
    optimise!(hropt, min_risk!)
    @test hropt.weights[1] ≈ hropt.weights[2] * k^2
    portfolio_performance(hropt, verbose = true)

    k = 2
    c = 6
    ret = (randn(1000, n) .+ 0.01) / 100
    ret[:, 4] = k * ret[:, 2] .- 6
    cov_mtx = cov(ret) * 252
    hropt = HRPOpt(tickers, returns = ret)
    optimise!(hropt, min_risk!)
    @test hropt.weights[2] ≈ hropt.weights[4] * k^2
    portfolio_performance(hropt, verbose = true)

    # Reading in the data; preparing expected returns and a risk model
    df = CSV.read("./assets/stock_prices.csv", DataFrame)
    dropmissing!(df)
    tickers = names(df)[2:end]
    returns = returns_from_prices(df[!, 2:end])

    mean_ret = vec(ret_model(MRet(), Matrix(returns)))
    S = cov(Cov(), Matrix(returns))

    hrp = HRPOpt(tickers, returns = Matrix(returns))
    optimise!(hrp, min_risk!)
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
    mutest, sigmatest, srtest =
        0.00042774259094865367, 0.008326107336179916, 0.04193528078202753
    @test mu ≈ mutest
    @test sigma ≈ sigmatest
    @test sr ≈ srtest

    hrp = HRPOpt(
        tickers,
        returns = Matrix(returns),
        D = Symmetric(sqrt.(clamp.((1 .- cov_to_cor(cov(Matrix(returns)))) / 2, 0, 1))),
    )
    optimise!(hrp, min_risk!)
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
    mutest, sigmatest, srtest =
        0.00042774259094865367, 0.008326107336179916, 0.04193528078202753
    @test mu ≈ mutest
    @test sigma ≈ sigmatest
    @test sr ≈ srtest

    @test_throws ArgumentError HRPOpt(tickers)
    @test_throws ArgumentError HRPOpt(tickers, returns = Matrix(returns), D = :custom)

    hrp = HRPOpt(tickers, returns = Matrix(returns), mean_ret = mean_ret)
    optimise!(hrp, max_utility!)
    testweights = [
        0.04674274491618855,
        0.020932673998682715,
        0.06005728656496047,
        0.1046648514939463,
        0.1367697430468015,
        0.015727935174239236,
        0.00875227558147826,
        0.0031852304574162037,
        0.06286634905700891,
        0.040554201299668705,
        0.03316908880371446,
        0.00040646389091307864,
        0.0006279335697330204,
        0.015647456111014046,
        0.01888733139253485,
        0.058052730171978,
        0.18666420055128985,
        0.03807125394788301,
        0.07712057408877951,
        0.07109967588176931,
    ]
    @test hrp.weights ≈ testweights
    mu, sigma, sr = portfolio_performance(hrp, verbose = true)
    mutest, sigmatest, srtest =
        0.0008917982567342776, 0.010139837113092838, 0.08019984006445079
    @test mu ≈ mutest
    @test sigma ≈ sigmatest
    @test sr ≈ srtest

    hrp = HRPOpt(tickers, returns = Matrix(returns))
    optimise!(hrp, max_return!)
    testweights = [
        0.05103241439915362,
        0.008261786006720851,
        0.07544550629047544,
        0.09285819682738837,
        0.1374027774715286,
        0.008151756921041897,
        0.02376545498103861,
        0.002266258652057186,
        0.06373404842684177,
        0.06023612564516282,
        0.027990402473918322,
        0.0006494224862853693,
        0.01096404107484476,
        0.010282320209842395,
        0.02244149316656006,
        0.07392800079630253,
        0.16934156099369138,
        0.032676986540569226,
        0.07107928041610259,
        0.05749216622047417,
    ]
    @test hrp.weights ≈ testweights
    mu, sigma, sr = portfolio_performance(hrp)
    mutest, sigmatest, srtest =
        0.0008970044853848074, 0.01032271413321323, 0.07928336800172553
    @test mu ≈ mutest
    @test sigma ≈ sigmatest
    @test sr ≈ srtest

    optimise!(hrp, max_utility!, 1e-12)
    mu_mq, sigma_mq, sr_mq = portfolio_performance(hrp)
    @test mu ≈ mu_mq
    @test sigma ≈ sigma_mq
    @test sr ≈ sr_mq

    optimise!(hrp, min_risk!)
    mu, sigma, sr = portfolio_performance(hrp)
    optimise!(hrp, max_utility!, 1e12)
    mu_mq, sigma_mq, sr_mq = portfolio_performance(hrp)
    @test mu ≈ mu_mq
    @test sigma ≈ sigma_mq
    @test sr ≈ sr_mq

    hrp = HRPOpt(tickers, returns = Matrix(returns))
    optimise!(hrp, max_sharpe!)
    testweights = [
        0.05124355676249418,
        0.024857474970493086,
        0.05660496186546185,
        0.10462278456948915,
        0.1201542970032186,
        0.018281752187296992,
        0.015453145244820284,
        0.0022025032912748924,
        0.055679519333108635,
        0.04108068400702428,
        0.04342445740110776,
        0.0007839959917912433,
        0.00693028584640944,
        0.019119140222166167,
        0.05415170966058442,
        0.031678038267495746,
        0.19275347732008583,
        0.03045365728754988,
        0.07690178675420695,
        0.053622772013920614,
    ]
    @test hrp.weights ≈ testweights
    mu, sigma, sr = portfolio_performance(hrp)
    mutest, sigmatest, srtest =
        0.0007812467410052671, 0.010169692839688221, 0.06909370913135263
    @test mu ≈ mutest
    @test sigma ≈ sigmatest
    @test sr ≈ srtest
end
