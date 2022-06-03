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
    S = risk_matrix(Cov(), Matrix(returns))

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
        0.0013525614095372441,
        0.0845701131689161,
        0.08166673007984035,
        0.16404173578207706,
        0.0006158870845536016,
        0.049189141600734725,
        0.0069323722289936765,
        0.05932923127907018,
        0.061314572056272124,
        0.023130441823266446,
        0.005234336476474105,
        0.029075109239338887,
        0.009122881135802348,
        0.08064447242628374,
        0.1282639687104863,
        0.16244508468678345,
        0.03794588221718694,
        0.06687051853947568,
        0.06890700229846011,
    ]
    @test hrp.weights ≈ testweights
    mu, sigma, sr = portfolio_performance(hrp)
    mutest, sigmatest, srtest = 0.23977731989029213, 0.1990931650935869, 1.103891837708152
    @test mu ≈ mutest
    @test sigma ≈ sigmatest
    @test sr ≈ srtest

    hrp = HRPOpt(tickers, returns = Matrix(returns))
    optimise!(hrp, max_return!)
    testweights = [
        0.053215775973689344,
        0.0014298867467719253,
        0.08459870820429168,
        0.08164522293333767,
        0.16394762797243273,
        0.0006506856473305975,
        0.04877541431712327,
        0.006966529106643553,
        0.05933305965116203,
        0.061351686897633866,
        0.023122626771250834,
        0.005255850404371056,
        0.028772641690166374,
        0.009101244131990852,
        0.08058316841784602,
        0.1283171317054279,
        0.1623978785944996,
        0.03793881981412327,
        0.06685508940506842,
        0.0688799562236672,
    ]
    @test hrp.weights ≈ testweights
    mu, sigma, sr = portfolio_performance(hrp)
    mutest, sigmatest, srtest = 0.23967354225156112, 0.19889330638531708, 1.1044793122699987
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

    hrp = HRPOpt(tickers, returns = Matrix(returns), mean_ret = mean_ret)
    optimise!(hrp, max_sharpe!)
    testweights = [
        0.060950709858938946,
        0.024688958223226046,
        0.07636491146846057,
        0.09098114189215895,
        0.16345933040156524,
        0.01321315607100166,
        0.030495530737540685,
        0.0041676003148035014,
        0.05012315893824599,
        0.047425984889118235,
        0.0354866498145466,
        0.0029681045872571015,
        0.018207753673800363,
        0.01757206461257125,
        0.07232501039144949,
        0.09318121978905623,
        0.17882024226595147,
        0.04731099202207161,
        0.0699409077092408,
        0.0859547509506454,
    ]
    @test hrp.weights ≈ testweights
    mu, sigma, sr = portfolio_performance(hrp)
    mutest, sigmatest, srtest = 0.23690508565115886, 0.19266051071806212, 1.1258409148960271
    @test mu ≈ mutest
    @test sigma ≈ sigmatest
    @test sr ≈ srtest

    optimise!(hrp, min_volatility!)
    mu1, sigma1, sr1 = portfolio_performance(hrp)
    optimise!(hrp, max_return!)
    mu2, sigma2, sr2 = portfolio_performance(hrp)
    optimise!(hrp, max_quadratic_utility!)
    mu3, sigma3, sr3 = portfolio_performance(hrp)

    @test sr > sr1
    @test sr > sr2
    @test sr > sr3
end