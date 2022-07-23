using Test
using PortfolioOptimiser, CSV, DataFrames

@testset "Custom optimiser" begin
    df = CSV.read("./assets/stock_prices.csv", DataFrame)
    tickers = names(df)[2:end]
    returns = dropmissing(returns_from_prices(df[!, 2:end]))

    mu = vec(ret_model(MRet(), Matrix(returns)))
    S = cov(Cov(), Matrix(returns))

    lower_bounds, upper_bounds, strict_bounds = 0.02, 0.03, 0.1
    ef = EffMeanVar(
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

    ef = EffMeanVar(tickers, mu, S)
    obj_params = [ef.mean_ret, ef.cov_mtx, 1000]
    custom_optimiser!(ef, kelly_objective, obj_params, initial_guess = fill(1 / 20, 20))
    testweights = [
        0.005767442440593688,
        0.03154081133764419,
        0.011642424046938063,
        0.027730804163849977,
        0.016914645304759907,
        0.026146623485008156,
        5.212701881376053e-8,
        0.13902777785485385,
        9.810751772787176e-8,
        3.0636345199128637e-6,
        0.2890167913451911,
        4.9338483149047114e-8,
        3.4404354300215894e-8,
        0.12226858861714271,
        1.7542852298587199e-7,
        0.016190054982200913,
        0.0015896888326559673,
        0.19489319656761958,
        1.3053117484579068e-7,
        0.11726754744995008,
    ]
    @test isapprox(ef.weights, testweights, rtol = 1e-3)
end

# @testset "Custom NL optimiser" begin
#     df = CSV.read("./assets/stock_prices.csv", DataFrame)
#     tickers = names(df)[2:end]
#     returns = dropmissing(returns_from_prices(df[!, 2:end]))

#     mean_ret = ret_model(MRet(), Matrix(returns))
#     S = cov(Cov(), Matrix(returns))

#     function logarithmic_barrier(w::T...) where {T}
#         cov_mtx = obj_params[1]
#         k = obj_params[2]
#         w = [i for i in w]
#         PortfolioOptimiser.logarithmic_barrier(w, cov_mtx, k)
#     end
#     ef = EffMeanVar(tickers, mean_ret, S, weight_bounds = (0.03, 0.2))
#     obj_params = (ef.cov_mtx, 0.001)
#     custom_nloptimiser!(ef, logarithmic_barrier, obj_params)
#     testweights = [
#         0.0459759150798785,
#         0.0465060033736648,
#         0.0406223479132712,
#         0.0367443944616427,
#         0.0394265018793106,
#         0.0533782687664828,
#         0.0300000004313922,
#         0.0940987390957382,
#         0.0307216371198716,
#         0.0402226547518765,
#         0.1209594547099574,
#         0.0300000012819006,
#         0.0300000004033861,
#         0.065153111884118,
#         0.0300000023324234,
#         0.0367907531898106,
#         0.0457521843192034,
#         0.0858732773342319,
#         0.037645669856962,
#         0.0601290817911846,
#     ]
#     @test minimum(abs.(0.2 .- ef.weights)) >= 0
#     @test minimum(abs.(ef.weights .- 0.03)) <= 1e-8

#     @test isapprox(ef.weights, testweights, rtol = 1e-6)

#     @test_throws ArgumentError custom_nloptimiser!(ef, logarithmic_barrier, obj_params)

#     ef = EffMeanVar(tickers, mean_ret, S, weight_bounds = fill((0.03, 0.2), 20))
#     obj_params = (ef.cov_mtx, 0.001)
#     custom_nloptimiser!(
#         ef,
#         logarithmic_barrier,
#         obj_params;
#         initial_guess = fill(1 / 20, 20),
#     )
#     @test isapprox(ef.weights, testweights, rtol = 1e-6)

#     @test_throws ArgumentError EffMeanVar(
#         tickers,
#         mean_ret,
#         S,
#         weight_bounds = fill((0.03, 0.2), 19),
#     )

#     ef = EffMeanVar(tickers, mean_ret, S, weight_bounds = fill([0.03, 0.2], 20))
#     obj_params = (ef.cov_mtx, 0.001)
#     custom_nloptimiser!(
#         ef,
#         logarithmic_barrier,
#         obj_params;
#         initial_guess = fill(1 / 20, 20),
#     )
#     @test isapprox(ef.weights, testweights, rtol = 1e-6)

#     weight_bounds = [
#         (0.03, 0.2)
#         (0.03, 0.2)
#         (0.03, 0.2)
#         (0.03, 0.2)
#         (0.03, 0.2)
#         (0.03, 0.2)
#         (0.03, 0.2)
#         (0.03, 0.2)
#         (0.03, 0.2)
#         (0.03, 0.2)
#         [0.03, 0.2]
#         (0.03, 0.2)
#         (0.03, 0.2)
#         (0.03, 0.2)
#         (0.03, 0.2)
#         (0.03, 0.2)
#         (0.03, 0.2)
#         (0.03, 0.2)
#         (0.03, 0.2)
#         (0.03, 0.2)
#     ]

#     @test_throws ArgumentError EffMeanVar(
#         tickers,
#         mean_ret,
#         S,
#         weight_bounds = weight_bounds,
#     )

#     weight_bounds = [
#         (0.03, 0.2)
#         (0.03, 0.2)
#         (0.03, 0.2)
#         (0.03, 0.2)
#         (0.03, 0.2)
#         (0.03, 0.2)
#         (0.03, 0.2)
#         (0.03, 0.2)
#         (0.03, 0.2)
#         (0.03, 0.2)
#         (0.03, 0.2, 0.5)
#         (0.03, 0.2)
#         (0.03, 0.2)
#         (0.03, 0.2)
#         (0.03, 0.2)
#         (0.03, 0.2)
#         (0.03, 0.2)
#         (0.03, 0.2)
#         (0.03, 0.2)
#         (0.03, 0.2)
#     ]

#     @test_throws ArgumentError EffMeanVar(
#         tickers,
#         mean_ret,
#         S,
#         weight_bounds = weight_bounds,
#     )

#     function sharpe_ratio_nl(w::T...) where {T}
#         mean_ret = obj_params[1]
#         cov_mtx = obj_params[2]
#         rf = obj_params[3]

#         w = [i for i in w]
#         sr = PortfolioOptimiser.sharpe_ratio(w, mean_ret, cov_mtx, rf)

#         return -sr
#     end
#     ef = EffMeanVar(tickers, mean_ret, S)
#     obj_params = [ef.mean_ret, ef.cov_mtx, ef.rf]
#     custom_nloptimiser!(ef, sharpe_ratio_nl, obj_params)
#     mu, sigma, sr = portfolio_performance(ef)

#     ef2 = EffMeanVar(tickers, mean_ret, S)
#     max_sharpe!(ef2)
#     mu2, sigma2, sr2 = portfolio_performance(ef2)

#     @test isapprox(mu, mu2, rtol = 1e-6)
#     @test isapprox(sigma, sigma2, rtol = 1e-6)
#     @test isapprox(sr, sr2, rtol = 1e-6)

#     cd = EffCDaR(tickers, mean_ret, Matrix(returns))
#     function cdar_ratio(w...)
#         mean_ret = obj_params[1]
#         beta = obj_params[2]
#         samples = obj_params[3]
#         n = obj_params[4]
#         o = obj_params[5]
#         p = obj_params[6]

#         weights = [i for i in w[1:n]]
#         alpha = w[o]
#         z = [i for i in w[p:end]]

#         mu = PortfolioOptimiser.port_return(weights, mean_ret)
#         CDaR = PortfolioOptimiser.cdar(alpha, z, samples, beta)

#         return -mu / CDaR
#     end

#     obj_params = []
#     extra_vars = []
#     push!(obj_params, mean_ret)
#     push!(obj_params, cd.beta)
#     push!(obj_params, size(cd.returns, 1))
#     push!(obj_params, 20)
#     push!(obj_params, 21)
#     push!(obj_params, 22)
#     push!(extra_vars, (cd.model[:alpha], 0.1))
#     push!(extra_vars, (cd.model[:z], fill(1 / length(cd.model[:z]), length(cd.model[:z]))))
#     custom_nloptimiser!(cd, cdar_ratio, obj_params, extra_vars)
#     mu, cdar1 = portfolio_performance(cd)
#     @test mu / cdar1 ≈ 3.2819464291074305

#     cd2 = EffCDaR(tickers, mean_ret, Matrix(returns))
#     min_risk!(cd2)
#     mu2, cdar2 = portfolio_performance(cd2)
#     mu2 / cdar2

#     @test mu / cdar1 > mu2 / cdar2
# end