using Test
using PortfolioOptimiser, CSV, DataFrames

@testset "Custom optimiser" begin
    df = CSV.read("./assets/stock_prices.csv", DataFrame)
    tickers = names(df)[2:end]
    returns = dropmissing(returns_from_prices(df[!, 2:end]))

    mu = vec(ret_model(MRet(), Matrix(returns)))
    S = risk_matrix(Cov(), Matrix(returns))

    lower_bounds, upper_bounds, strict_bounds = 0.02, 0.03, 0.1
    ef = MeanVar(
        tickers,
        mu,
        S;
        extra_constraints = [
            :(model[:w][1] >= $lower_bounds),
            :(model[:w][end] <= $upper_bounds),
            :(model[:w][6] == $strict_bounds),
        ],
    )
    obj_params = [ef.mean_ret, ef.cov_mtx, 100]
    custom_optimiser!(ef, kelly_objective, obj_params)
    @test ef.weights[1] >= lower_bounds
    @test ef.weights[end] <= upper_bounds
    @test isapprox(ef.weights[6], strict_bounds)

    ef = MeanVar(tickers, mu, S)
    obj_params = [ef.mean_ret, ef.cov_mtx, 1000]
    custom_optimiser!(ef, kelly_objective, obj_params, initial_guess = fill(1 / 20, 20))
    testweights = [
        0.004758143945488986,
        0.0313612711044923,
        0.011581566953884194,
        0.027540175507339622,
        0.018637959513280566,
        0.02649173712097102,
        4.490597294476849e-16,
        0.1389075022528616,
        1.4513664512737593e-16,
        2.0829028957690552e-15,
        0.28879549783043423,
        5.648363280996347e-16,
        1.0047135830164754e-15,
        0.12182714167506264,
        1.8172509176062567e-15,
        0.016375182319275105,
        0.0023305748424262045,
        0.19471263615932596,
        8.619879289218291e-17,
        0.1166806107751625,
    ]
    @test isapprox(ef.weights, testweights, rtol = 1e-3)
end

@testset "Custom NL optimiser" begin
    df = CSV.read("./assets/stock_prices.csv", DataFrame)
    tickers = names(df)[2:end]
    returns = dropmissing(returns_from_prices(df[!, 2:end]))

    mu = vec(ret_model(MRet(), Matrix(returns)))
    S = risk_matrix(Cov(), Matrix(returns))

    function logarithmic_barrier(w::T...) where {T}
        cov_mtx = obj_params[1]
        k = obj_params[2]
        w = [i for i in w]
        PortfolioOptimiser.logarithmic_barrier(w, cov_mtx, k)
    end
    mean_ret = ret_model(MRet(), Matrix(returns))
    ef = MeanVar(tickers, mean_ret, S, weight_bounds = (0.03, 0.2))
    obj_params = (ef.cov_mtx, 0.001)
    custom_nloptimiser!(ef, logarithmic_barrier, obj_params)
    testweights = [
        0.0459759150798785,
        0.0465060033736648,
        0.0406223479132712,
        0.0367443944616427,
        0.0394265018793106,
        0.0533782687664828,
        0.0300000004313922,
        0.0940987390957382,
        0.0307216371198716,
        0.0402226547518765,
        0.1209594547099574,
        0.0300000012819006,
        0.0300000004033861,
        0.065153111884118,
        0.0300000023324234,
        0.0367907531898106,
        0.0457521843192034,
        0.0858732773342319,
        0.037645669856962,
        0.0601290817911846,
    ]
    @test minimum(abs.(0.2 .- ef.weights)) >= 0
    @test minimum(abs.(ef.weights .- 0.03)) <= 1e-8

    @test isapprox(ef.weights, testweights, rtol = 1e-6)

    @test_throws ArgumentError custom_nloptimiser!(ef, logarithmic_barrier, obj_params)

    ef = MeanVar(tickers, mean_ret, S, weight_bounds = fill((0.03, 0.2), 20))
    obj_params = (ef.cov_mtx, 0.001)
    custom_nloptimiser!(
        ef,
        logarithmic_barrier,
        obj_params;
        initial_guess = fill(1 / 20, 20),
    )
    @test isapprox(ef.weights, testweights, rtol = 1e-6)

    @test_throws ArgumentError MeanVar(
        tickers,
        mean_ret,
        S,
        weight_bounds = fill((0.03, 0.2), 19),
    )

    ef = MeanVar(tickers, mean_ret, S, weight_bounds = fill([0.03, 0.2], 20))
    obj_params = (ef.cov_mtx, 0.001)
    custom_nloptimiser!(
        ef,
        logarithmic_barrier,
        obj_params;
        initial_guess = fill(1 / 20, 20),
    )
    @test isapprox(ef.weights, testweights, rtol = 1e-6)

    weight_bounds = [
        (0.03, 0.2)
        (0.03, 0.2)
        (0.03, 0.2)
        (0.03, 0.2)
        (0.03, 0.2)
        (0.03, 0.2)
        (0.03, 0.2)
        (0.03, 0.2)
        (0.03, 0.2)
        (0.03, 0.2)
        [0.03, 0.2]
        (0.03, 0.2)
        (0.03, 0.2)
        (0.03, 0.2)
        (0.03, 0.2)
        (0.03, 0.2)
        (0.03, 0.2)
        (0.03, 0.2)
        (0.03, 0.2)
        (0.03, 0.2)
    ]

    @test_throws ArgumentError MeanVar(tickers, mean_ret, S, weight_bounds = weight_bounds)

    weight_bounds = [
        (0.03, 0.2)
        (0.03, 0.2)
        (0.03, 0.2)
        (0.03, 0.2)
        (0.03, 0.2)
        (0.03, 0.2)
        (0.03, 0.2)
        (0.03, 0.2)
        (0.03, 0.2)
        (0.03, 0.2)
        (0.03, 0.2, 0.5)
        (0.03, 0.2)
        (0.03, 0.2)
        (0.03, 0.2)
        (0.03, 0.2)
        (0.03, 0.2)
        (0.03, 0.2)
        (0.03, 0.2)
        (0.03, 0.2)
        (0.03, 0.2)
    ]

    @test_throws ArgumentError MeanVar(tickers, mean_ret, S, weight_bounds = weight_bounds)

    function sharpe_ratio_nl(w::T...) where {T}
        mean_ret = obj_params[1]
        cov_mtx = obj_params[2]
        rf = obj_params[3]

        w = [i for i in w]
        sr = PortfolioOptimiser.sharpe_ratio(w, mean_ret, cov_mtx, rf)

        return -sr
    end
    ef = MeanVar(tickers, mean_ret, S)
    obj_params = [ef.mean_ret, ef.cov_mtx, ef.rf]
    custom_nloptimiser!(ef, sharpe_ratio_nl, obj_params)
    mu, sigma, sr = portfolio_performance(ef)

    ef2 = MeanVar(tickers, mean_ret, S)
    max_sharpe!(ef2)
    mu2, sigma2, sr2 = portfolio_performance(ef2)

    @test isapprox(mu, mu2, rtol = 1e-6)
    @test isapprox(sigma, sigma2, rtol = 1e-6)
    @test isapprox(sr, sr2, rtol = 1e-6)
end