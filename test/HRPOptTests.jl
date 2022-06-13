using Test

using PortfolioOptimiser, CSV, DataFrames, Statistics, LinearAlgebra

@testset "HRP Opt" begin
    tickers = ["AAPL", "GOOG", "AMD", "MSFT"]
    n = length(tickers)
    ret = (randn(1000, n) .+ 0.01) / 100
    ret[:, 1] = ret[:, 2] .+ 1
    cov_mtx = cov(ret) * 252
    hropt = HRPOpt(tickers, cov_mtx = cov_mtx)
    optimise!(hropt, min_volatility!)
    @test hropt.weights[1] ≈ hropt.weights[2]
    mu, sigma, sr = portfolio_performance(hropt, verbose = true)
    @test isnan(mu)
    @test isnan(sr)

    k = 5
    ret = (randn(1000, n) .+ 0.01) / 100
    ret[:, 2] = k * ret[:, 1]
    cov_mtx = cov(ret) * 252

    hropt = HRPOpt(tickers, returns = ret, cov_mtx = cov_mtx)
    optimise!(hropt, min_volatility!)
    @test hropt.weights[1] ≈ hropt.weights[2] * k^2
    portfolio_performance(hropt, verbose = true)

    k = 2
    c = 6
    ret = (randn(1000, n) .+ 0.01) / 100
    ret[:, 4] = k * ret[:, 2] .- 6
    cov_mtx = cov(ret) * 252
    hropt = HRPOpt(tickers, returns = ret)
    optimise!(hropt, min_volatility!)
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
    optimise!(hrp, min_volatility!)
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
    optimise!(hrp, min_volatility!)
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

    hrp = HRPOpt(tickers, returns = Matrix(returns), mean_ret = mean_ret)
    optimise!(hrp, max_quadratic_utility!)
    testweights = [
        0.05322978571963428,
        0.0007366743249836424,
        0.07544723203311376,
        0.08166673007984035,
        0.16404173578207706,
        0.0006158870845536016,
        0.029075109239338887,
        0.0033299024042351827,
        0.05932923127907018,
        0.061314572056272124,
        0.023130441823266446,
        0.0010802311544558937,
        0.020114032361395837,
        0.009122881135802348,
        0.026208412089363305,
        0.08159706578005416,
        0.16244508468678345,
        0.028638680799551387,
        0.06687051853947568,
        0.05200579162673235,
    ]
    @test hrp.weights ≈ testweights
    mu, sigma, sr = portfolio_performance(hrp, verbose = true)
    mutest, sigmatest, srtest = 0.23097962331818328, 0.1668730376162429, 1.2643122360088639
    @test mu ≈ mutest
    @test sigma ≈ sigmatest
    @test sr ≈ srtest

    hrp = HRPOpt(tickers, returns = Matrix(returns))
    optimise!(hrp, max_return!)
    testweights = [
        0.053215775973689344,
        0.0007792010994413279,
        0.07549746407230083,
        0.08164522293333767,
        0.16394762797243273,
        0.0006506856473305975,
        0.028772641690166374,
        0.0033608966492525124,
        0.05933305965116203,
        0.061351686897633866,
        0.023122626771250834,
        0.0010939075270547547,
        0.020002772626956895,
        0.009101244131990852,
        0.02623560761994445,
        0.08205344231864094,
        0.1623978785944996,
        0.02862072025216071,
        0.06685508940506842,
        0.05196244816568531,
    ]
    @test hrp.weights ≈ testweights
    mu, sigma, sr = portfolio_performance(hrp)
    mutest, sigmatest, srtest = 0.23094803490038732, 0.16683960195621084, 1.2643762777362253
    @test mu ≈ mutest
    @test sigma ≈ sigmatest
    @test sr ≈ srtest

    optimise!(hrp, max_quadratic_utility!, 1e-12)
    mu_mq, sigma_mq, sr_mq = portfolio_performance(hrp)
    @test mu ≈ mu_mq
    @test sigma ≈ sigma_mq
    @test sr ≈ sr_mq

    optimise!(hrp, min_volatility!)
    mu, sigma, sr = portfolio_performance(hrp)
    optimise!(hrp, max_quadratic_utility!, 1e12)
    mu_mq, sigma_mq, sr_mq = portfolio_performance(hrp)
    @test mu ≈ mu_mq
    @test sigma ≈ sigma_mq
    @test sr ≈ sr_mq

    hrp = HRPOpt(tickers, returns = Matrix(returns))
    optimise!(hrp, max_sharpe!)
    testweights = [
        0.05541498467845542,
        0.01321315607100166,
        0.05879284685588932,
        0.09098114189215895,
        0.14861346669656336,
        0.011475802152224386,
        0.018207753673800363,
        0.001996729401593725,
        0.05012315893824599,
        0.04311861555137755,
        0.0354866498145466,
        0.0008069353069836714,
        0.012287777063740324,
        0.01757206461257125,
        0.06094073258126751,
        0.05988202434293816,
        0.17882024226595147,
        0.02567627593782026,
        0.0699409077092408,
        0.046648734453629236,
    ]
    @test hrp.weights ≈ testweights
    mu, sigma, sr = portfolio_performance(hrp)
    mutest, sigmatest, srtest = 0.20410030703063975, 0.164241878468217, 1.1209096531751226
    @test mu ≈ mutest
    @test sigma ≈ sigmatest
    @test sr ≈ srtest
end