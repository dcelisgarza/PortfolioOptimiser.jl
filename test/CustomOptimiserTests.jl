using Test
using PortfolioOptimiser, CSV, DataFrames

@testset "Custom optimiser" begin
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
        # PortfolioOptimiser.logarithmic_barrier(w, cov_mtx, k)
        # logarithmic_barrier2(w, cov_mtx, k)
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
    @test isapprox(ef.weights, testweights, rtol = 1e-6)
end