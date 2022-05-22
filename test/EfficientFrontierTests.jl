using Test

using PortfolioOptimiser.BaseOptimiser
using PortfolioOptimiser.EfficientFrontierOptimiser
using PortfolioOptimiser.ExpectedReturns

using JuMP, LinearAlgebra, Ipopt

@testset "Efficient MeanVar" begin
    tickers = ["AAPL", "GOOG", "AMD", "MSFT"]
    sectors = ["Tech", "Food", "Oil", "Tech"]
    n = length(tickers)

    mean_ret = rand(n)
    cov_mtx = rand(n, n)

    mean_ret = [1, 1, 1, 1]
    cov_mtx = zeros(n, n)
    cov_mtx[I(4)] .= 1
    weight_bounds = (0, 1)
    maxSharpe = MaxSharpe(tickers, mean_ret, cov_mtx)
    @test typeof(maxSharpe) <: MaxSharpe
    max_sharpe!(maxSharpe)
    @test all(maxSharpe.weights .≈ 0.25)
    @test sum(maxSharpe.weights) ≈ 1
    @test abs(port_variance(maxSharpe.weights, maxSharpe.cov_mtx) - 0.25) <
          sqrt(eps()) * length(cov_mtx)
    @test abs(port_return(maxSharpe.weights, maxSharpe.mean_ret) - 1) <
          sqrt(eps()) * length(cov_mtx)

    mean_ret = [1, 2, 3, 4]
    cov_mtx = zeros(n, n)
    cov_mtx[I(4)] .= 1
    weight_bounds = (0, 1)
    maxSharpe = MeanVar(tickers, mean_ret, cov_mtx)
    @test typeof(maxSharpe) <: MeanVar
    max_sharpe!(maxSharpe)
    @test sum(maxSharpe.weights[i] - i / 10 for i in 1:n) < sqrt(eps()) * length(cov_mtx)
    @test sum(maxSharpe.weights) ≈ 1
    @test abs(
        port_variance(maxSharpe.weights, maxSharpe.cov_mtx) -
        dot(maxSharpe.weights, maxSharpe.cov_mtx, maxSharpe.weights),
    ) < sqrt(eps()) * length(cov_mtx)
    @test abs(
        port_return(maxSharpe.weights, maxSharpe.mean_ret) -
        dot(maxSharpe.weights, maxSharpe.mean_ret),
    ) < sqrt(eps()) * length(cov_mtx)

    mean_ret = [1, 2, 3, 4]
    cov_mtx = ones(n, n)
    cov_mtx[I(4)] .= 1
    weight_bounds = (0, 1)
    maxSharpe = MeanVar(tickers, mean_ret, cov_mtx)
    max_sharpe!(maxSharpe)
    @test abs(maxSharpe.weights[4] - 1) < sqrt(eps()) * length(cov_mtx)
    @test sum(maxSharpe.weights) ≈ 1
    @test abs(port_variance(maxSharpe.weights, maxSharpe.cov_mtx) - 1) <
          sqrt(eps()) * length(cov_mtx)
    @test abs(port_return(maxSharpe.weights, maxSharpe.mean_ret) - 4) <
          sqrt(eps()) * length(cov_mtx)

    mean_ret = [1, 1, 1, 1]
    cov_mtx = ones(n, n)
    cov_mtx[I(4)] .= 1
    weight_bounds = (0, 1)
    extra_obj_terms = [quote
        L2_reg(model[:w], 1)
    end]
    maxSharpe = MeanVar(tickers, mean_ret, cov_mtx; extra_obj_terms = extra_obj_terms)
    max_sharpe!(maxSharpe)
    objective_function(maxSharpe.model)
    @test all(maxSharpe.weights .≈ 0.25)
    @test sum(maxSharpe.weights) ≈ 1
    @test abs(port_variance(maxSharpe.weights, maxSharpe.cov_mtx) - 1) <
          sqrt(eps()) * length(cov_mtx)
    @test abs(port_return(maxSharpe.weights, maxSharpe.mean_ret) - 1) <
          sqrt(eps()) * length(cov_mtx)

    mean_ret = [1, 1, 100, 100]
    cov_mtx = [1 0 1 1; 0 1 1 1; 1 1 1 1; 1 1 1 1]
    cov_mtx[I(4)] .= 1
    weight_bounds = (0, 1)
    minVolatility = MinVolatility(tickers, cov_mtx)
    μ, σ, sr = portfolio_performance(minVolatility, mean_ret; verbose = false)

    @test typeof(minVolatility) <: MinVolatility
    min_volatility!(minVolatility)
    @test minVolatility.weights[1] ≈ minVolatility.weights[2]
    @test abs(minVolatility.weights[1] - 0.5) < sqrt(eps()) * length(cov_mtx)
    @test sum(minVolatility.weights) ≈ 1
    @test abs(port_variance(minVolatility.weights, minVolatility.cov_mtx) - 0.5) <
          sqrt(eps()) * length(cov_mtx)

    mean_ret = [1, 1, 100, 100]
    cov_mtx = [1 0 0.5 0.3; 0 1 1 1; 0.5 1 1 1; 0.3 1 1 1]
    cov_mtx[I(4)] .= 1
    weight_bounds = (0, 1)
    minVolatility = MeanVar(tickers, mean_ret, cov_mtx)
    @test typeof(minVolatility) <: MeanVar
    min_volatility!(minVolatility)
    @test minVolatility.weights[1] ≈ minVolatility.weights[2]
    @test abs(minVolatility.weights[1] - 0.5) < sqrt(eps()) * length(cov_mtx)
    @test sum(minVolatility.weights) ≈ 1
    @test abs(port_variance(minVolatility.weights, minVolatility.cov_mtx) - 0.5) <
          sqrt(eps()) * length(cov_mtx)

    mean_ret = [1, 1, 1, 1]
    cov_mtx = ones(n, n)
    cov_mtx[I(4)] .= 1
    weight_bounds = (0, 1)
    maxSharpe = MeanVar(tickers, mean_ret, cov_mtx; rf = 0)
    maxRetMod = max_return(maxSharpe)
    @test all(value.(maxRetMod[:w]) .≈ 0.25)
    @test objective_value(maxRetMod) ≈ 1

    mean_ret = [1, 1, 1, 1]
    cov_mtx = zeros(n, n)
    cov_mtx[I(4)] .= 1
    weight_bounds = (0, 1)
    maxSharpe = MeanVar(tickers, mean_ret, cov_mtx; rf = 0)
    maxRetMod = max_return(maxSharpe)
    @test all(value.(maxRetMod[:w]) .≈ 0.25)
    @test objective_value(maxRetMod) ≈ 1

    mean_ret = [1, 4, 5, 2]
    cov_mtx = ones(n, n)
    cov_mtx[I(4)] .= 1
    weight_bounds = (0, 1)
    maxSharpe = MeanVar(tickers, mean_ret, cov_mtx; rf = 0)
    maxRetMod = max_return(maxSharpe)
    @test value.(maxRetMod[:w][3]) ≈ 1
    @test objective_value(maxRetMod) ≈ mean_ret[3]

    mean_ret = [1, 3.2, 3, 2]
    cov_mtx = zeros(n, n)
    cov_mtx[I(4)] .= 1
    weight_bounds = (0, 1)
    maxSharpe = MeanVar(tickers, mean_ret, cov_mtx; rf = 0)
    maxRetMod = max_return(maxSharpe)
    @test value.(maxRetMod[:w][2]) ≈ 1
    @test objective_value(maxRetMod) ≈ mean_ret[2]

    mean_ret = [1, 1, 1, 1]
    cov_mtx = ones(n, n)
    cov_mtx[I(4)] .= 1
    weight_bounds = (0, 1)
    quadUtil = MaxQuadraticUtility(tickers, mean_ret, cov_mtx; market_neutral = true)
    @test typeof(quadUtil) <: MaxQuadraticUtility
    max_quadratic_utility!(quadUtil)
    @test all(abs.(quadUtil.weights) .< eps())

    mean_ret = [1, 1, 1, 1]
    cov_mtx = zeros(n, n)
    cov_mtx[I(4)] .= 1
    weight_bounds = (0, 1)
    quadUtil = MeanVar(tickers, mean_ret, cov_mtx; rf = 0, market_neutral = true)
    @test typeof(quadUtil) <: MeanVar
    max_quadratic_utility!(quadUtil)
    @test all(abs.(quadUtil.weights) .< eps())

    mean_ret = [1, 2, 3, 1]
    cov_mtx = ones(n, n)
    cov_mtx[I(4)] .= 1
    weight_bounds = (0, 1)
    quadUtil = MeanVar(tickers, mean_ret, cov_mtx; rf = 0, market_neutral = true)
    max_quadratic_utility!(quadUtil)
    #     @test quadUtil.weights
    quadUtil.weights[1] ≈ quadUtil.weights[4] ≈ -quadUtil.weights[2] ≈ -quadUtil.weights[3]

    mean_ret = [1, 2, 4, 3]
    cov_mtx = zeros(n, n)
    cov_mtx[I(4)] .= 1
    weight_bounds = (0, 1)
    quadUtil = MeanVar(tickers, mean_ret, cov_mtx; rf = 0, market_neutral = true)
    max_quadratic_utility!(quadUtil)
    @test quadUtil.weights[1] ≈ -quadUtil.weights[3]
    @test quadUtil.weights[2] ≈ -quadUtil.weights[4]

    mean_ret = [1, 2, 3, 4]
    cov_mtx = ones(n, n)
    cov_mtx[I(4)] .= 1
    weight_bounds = (0, 1)
    quadUtil = MeanVar(tickers, mean_ret, cov_mtx; rf = 0, market_neutral = true)
    max_quadratic_utility!(quadUtil)
    @test abs(quadUtil.weights[4] - 1) < sqrt(eps()) * length(cov_mtx)

    mean_ret = [1, 2, 3, 4]
    cov_mtx = zeros(n, n)
    cov_mtx[I(4)] .= 1
    weight_bounds = (0, 1)
    quadUtil = MeanVar(
        tickers,
        mean_ret,
        cov_mtx,
        rf = 0,
        market_neutral = false,
        risk_aversion = 1_000_000_000_000,
    )
    max_quadratic_utility!(quadUtil)
    @test all(abs.(quadUtil.weights .- 0.25) .< sqrt(eps()) * length(cov_mtx))

    mean_ret = [1, 2, 3, 4]
    cov_mtx = ones(n, n)
    cov_mtx[I(4)] .= 1
    weight_bounds = (0, 1)
    quadUtil = MeanVar(
        tickers,
        mean_ret,
        cov_mtx,
        rf = 0,
        market_neutral = false,
        risk_aversion = 1_000_000_000_000,
    )
    max_quadratic_utility!(quadUtil)
    @test quadUtil.weights[4] ≈ 1

    mean_ret = [1, 4, 6, 8]
    cov_mtx = zeros(n, n)
    cov_mtx[I(4)] .= 1
    efficientRisk = EfficientRisk(tickers, mean_ret, cov_mtx)
    @test typeof(efficientRisk) <: EfficientRisk
    efficient_risk!(efficientRisk)
    @test all(abs.(efficientRisk.weights .- 0.25) .< sqrt(eps() * 4) * length(cov_mtx)^3)
    @test sum(efficientRisk.weights) ≈ 1

    mean_ret = [1, 4, 6, 8]
    cov_mtx = zeros(n, n)
    cov_mtx[I(4)] .= 1
    efficientRisk = MeanVar(tickers, mean_ret, cov_mtx; target_volatility = 1)
    @test typeof(efficientRisk) <: MeanVar
    efficient_risk!(efficientRisk)
    @test abs(efficientRisk.weights[4] - 1) < sqrt(eps()) * length(cov_mtx)
    @test sum(efficientRisk.weights) ≈ 1

    mean_ret = [1, 4, 6, 8]
    cov_mtx = ones(n, n)
    cov_mtx[I(4)] .= 1
    efficientRisk = MeanVar(tickers, mean_ret, cov_mtx; target_volatility = 1)
    @test_throws SingularException efficient_risk!(efficientRisk)

    mean_ret = [1, 4, 6, 8]
    cov_mtx = zeros(n, n)
    cov_mtx[I(4)] .= 1
    efficientReturn = EfficientReturn(tickers, mean_ret, cov_mtx; target_ret = 1)
    @test typeof(efficientReturn) <: EfficientReturn
    efficient_return!(efficientReturn)
    @test all(abs.(efficientReturn.weights .- 0.25) .< sqrt(eps()) * length(cov_mtx))
    @test port_return(efficientReturn.weights, efficientReturn.mean_ret) >= 1

    mean_ret = [1, 4, 6, 8]
    cov_mtx = ones(n, n)
    cov_mtx[I(4)] .= 1
    efficientReturn = MeanVar(tickers, mean_ret, cov_mtx; target_ret = 5)
    @test typeof(efficientReturn) <: MeanVar
    efficient_return!(efficientReturn)
    @test sum(efficientReturn.weights) ≈ 1
    @test port_return(efficientReturn.weights, efficientReturn.mean_ret) >= 5

    mean_ret = [1, 4, 6, 8]
    cov_mtx = zeros(n, n)
    cov_mtx[I(4)] .= 1
    efficientReturn = MeanVar(tickers, mean_ret, cov_mtx; target_ret = 10)
    efficient_return!(efficientReturn)
    efficientReturn.weights

    @test all(abs.(efficientReturn.weights .- 0.25) .< sqrt(eps()) * length(cov_mtx))
    @test abs(port_return(efficientReturn.weights, efficientReturn.mean_ret) - 4) >
          sqrt(eps()) * length(cov_mtx)

    mean_ret = [1, 4, 6, 8]
    cov_mtx = ones(n, n)
    cov_mtx[I(4)] .= 1
    efficientReturn = MeanVar(tickers, mean_ret, cov_mtx; target_ret = 10)
    efficient_return!(efficientReturn)

    @test abs(port_return(efficientReturn.weights, efficientReturn.mean_ret) - 4) >
          sqrt(eps()) * length(cov_mtx)
    μ, σ, sr = portfolio_performance(efficientReturn)
    @test abs(μ - 4) > sqrt(eps()) * length(cov_mtx)
    @test abs(σ - 1) < sqrt(eps()) * length(cov_mtx)
    @test abs(sr - (4 - 0.02) / 1) > sqrt(eps()) * length(cov_mtx)

    mean_ret = [1, 4, 6, 8]
    cov_mtx = ones(n, n)
    cov_mtx[I(4)] .= 1
    efficientReturn = MeanVar(tickers, mean_ret, cov_mtx; target_ret = 10, rf = 0)
    sector_map = Dict()
    for (ticker, sector) in zip(tickers, sectors)
        push!(sector_map, (ticker => sector))
    end
    sector_lower = Dict("Tech" => 0.2, "Oil" => 0.5, "Food" => 0.3)
    sector_upper = Dict("Tech" => 0.4, "Oil" => 0.8, "Food" => 0.6)
    add_sector_constraint!(efficientReturn, sector_map, sector_lower, sector_upper)
    constraint_object(efficientReturn.model[:Food_lower]).set.lower == sector_lower["Food"]
    constraint_object(efficientReturn.model[:Oil_lower]).set.lower == sector_lower["Oil"]
    constraint_object(efficientReturn.model[:Tech_lower]).set.lower == sector_lower["Tech"]
    constraint_object(efficientReturn.model[:Food_upper]).set.upper == sector_upper["Food"]
    constraint_object(efficientReturn.model[:Oil_upper]).set.upper == sector_upper["Oil"]
    constraint_object(efficientReturn.model[:Tech_upper]).set.upper == sector_upper["Tech"]
    efficient_return!(efficientReturn)
    @test 0.2 - sqrt(eps()) * length(cov_mtx) <=
          abs(efficientReturn.weights[1] + efficientReturn.weights[4]) <=
          0.4 + sqrt(eps()) * length(cov_mtx)
    @test 0.5 - sqrt(eps()) * length(cov_mtx) <=
          efficientReturn.weights[3] <=
          0.8 + sqrt(eps()) * length(cov_mtx)
    @test 0.3 - sqrt(eps()) * length(cov_mtx) <=
          efficientReturn.weights[2] <=
          0.6 + sqrt(eps()) * length(cov_mtx)

    mean_ret = [1, 4, 6, 8]
    cov_mtx = zeros(n, n)
    cov_mtx[I(4)] .= 1
    efficientFrontier = MeanVar(tickers, mean_ret, cov_mtx)

    max_sharpe!(efficientFrontier)
    @test haskey(efficientFrontier.model, :k)
    @test haskey(efficientFrontier.model, :k_positive)
    @test haskey(efficientFrontier.model, :max_sharpe_return)

    efficient_return!(efficientFrontier)
    @test !haskey(efficientFrontier.model, :k)
    @test !haskey(efficientFrontier.model, :k_positive)
    @test !haskey(efficientFrontier.model, :max_sharpe_return)
    @test haskey(efficientFrontier.model, :target_ret)

    efficient_risk!(efficientFrontier)
    @test !haskey(efficientFrontier.model, :target_ret)
    @test haskey(efficientFrontier.model, :target_variance)

    max_quadratic_utility!(efficientFrontier)
    @test !haskey(efficientFrontier.model, :target_variance)

    max_sharpe!(efficientFrontier)
    min_volatility!(efficientFrontier)
    @test !haskey(efficientFrontier.model, :k)
    @test !haskey(efficientFrontier.model, :k_positive)
    @test !haskey(efficientFrontier.model, :max_sharpe_return)
end

@testset "Efficient Semivar" begin
    tickers = ["AAPL", "GOOG", "AMD", "MSFT"]
    sectors = ["Tech", "Food", "Oil", "Tech"]
    n = length(tickers)

    ret = randn(1000, n) / 10
    ret[:, 1] .= 0.01
    cumret = cumprod(ret .+ 1, dims = 1)
    mean_ret = ret_model(MRet(), ret)

    effSemiVar = MeanSemivar(tickers, mean_ret, ret; benchmark = 10)
    minSemiVar = MinSemiVar(tickers, ret; benchmark = 10)
    min_semivar!(effSemiVar)
    min_semivar!(minSemiVar)
    @test all(abs.(effSemiVar.weights - minSemiVar.weights) .< sqrt(eps() * size(ret, 1)))
    @test abs(sum(effSemiVar.weights) - 1) < sqrt(eps() * size(ret, 1))
    @test abs(effSemiVar.weights[1] - 1) < sqrt(eps() * size(ret, 1))
    @test typeof(effSemiVar) <: MeanSemivar
    @test typeof(minSemiVar) <: MinSemiVar

    effSemiVar = MeanSemivar(tickers, mean_ret, ret)
    maxSemiVarQuad = MaxSemiVarQuadraticUtility(tickers, mean_ret, ret)
    max_quadratic_utility!(effSemiVar)
    max_quadratic_utility!(maxSemiVarQuad)
    @test typeof(effSemiVar) <: MeanSemivar
    @test typeof(maxSemiVarQuad) <: MaxSemiVarQuadraticUtility

    @test objective_function(minSemiVar.model) != objective_function(maxSemiVarQuad.model)
    @test all(
        abs.(effSemiVar.weights - maxSemiVarQuad.weights) .< sqrt(eps() * size(ret, 1)),
    )
    @test abs(sum(effSemiVar.weights) - 1) < sqrt(eps() * size(ret, 1))
    @test abs(effSemiVar.weights[1] - 1) < sqrt(eps() * size(ret, 1))
    efficient_risk!(effSemiVar)
    @test abs(effSemiVar.weights[1] - 1) < sqrt(eps() * size(ret, 1))

    ret = randn(1000, n) / 10
    ret[:, 1] .= 0.01
    ret[:, 2] .= 0.01
    mean_ret = ret_model(EMRet(), ret)

    effSemiVar = MeanSemivar(tickers, mean_ret, ret)
    effSemiRisk = EfficientSemiVarRisk(tickers, mean_ret, ret)
    efficient_risk!(effSemiVar)
    efficient_risk!(effSemiRisk)
    @test typeof(effSemiRisk) <: EfficientSemiVarRisk
    @test abs(sum(effSemiVar.weights) - 1) < sqrt(eps() * size(ret, 1))
    @test abs(effSemiVar.weights[1] - effSemiVar.weights[2]) < sqrt(eps() * size(ret, 1))
    @test all(abs.(effSemiVar.weights - effSemiRisk.weights) .< sqrt(eps() * size(ret, 1)))

    effSemiVar = MeanSemivar(tickers, mean_ret, ret)
    efficient_return!(effSemiVar)

    effSemiRet = EfficientSemiVarReturn(tickers, mean_ret, ret)
    efficient_return!(effSemiRet)
    @test typeof(effSemiRet) <: EfficientSemiVarReturn
    @test all(abs.(effSemiRet.weights .- effSemiVar.weights) .< sqrt(eps() * size(ret, 1)))

    ret = randn(1000, n) / 100
    mean_ret = ret_model(CAPMRet(), ret)

    effSemiVar = MeanSemivar(tickers, mean_ret, ret)
    minSemiVar = MinSemiVar(tickers, ret)
    maxSemiVarQuad = MaxSemiVarQuadraticUtility(tickers, mean_ret, ret)
    effSemiRisk = EfficientSemiVarRisk(tickers, mean_ret, ret)
    effSemiRet = EfficientSemiVarReturn(tickers, mean_ret, ret)
    min_semivar!(effSemiVar)
    min_semivar!(minSemiVar)
    min_semivar!(maxSemiVarQuad)
    min_semivar!(effSemiRisk)
    min_semivar!(effSemiRet)

    @test effSemiVar.weights ≈
          minSemiVar.weights ≈
          maxSemiVarQuad.weights ≈
          effSemiRisk.weights ≈
          effSemiRet.weights

    μ1, semi_σ1, sortino1 = portfolio_performance(effSemiVar; verbose = false)
    μ2, semi_σ2, sortino2 = portfolio_performance(minSemiVar, mean_ret)
    μ3, semi_σ3, sortino3 = portfolio_performance(maxSemiVarQuad)
    μ4, semi_σ4, sortino4 = portfolio_performance(effSemiRisk)
    μ5, semi_σ5, sortino5 = portfolio_performance(effSemiRet)

    μ1 ≈ μ2 ≈ μ3 ≈ μ4 ≈ μ5
    semi_σ1 ≈ semi_σ2 ≈ semi_σ3 ≈ semi_σ4 ≈ semi_σ5
    sortino1 ≈ sortino2 ≈ sortino3 ≈ sortino4 ≈ sortino5
end

@testset "Efficient CDaR" begin
    tickers = ["AAPL", "GOOG", "AMD", "MSFT"]
    sectors = ["Tech", "Food", "Oil", "Tech"]
    n = length(tickers)

    ret = (randn(1000, n) .+ 0.01) / 100
    mean_ret = ret_model(EMRet(), ret)

    effCDaR1 = EfficientCDaR(tickers, mean_ret, ret)
    minCDaR = MinCDaR(tickers, ret)
    min_cdar!(effCDaR1)
    min_cdar!(minCDaR)
    mu1, cdar1 = portfolio_performance(effCDaR1)
    mu2, cdar2 = portfolio_performance(minCDaR, mean_ret)
    @test mu1 ≈ mu2
    @test cdar1 ≈ cdar2
    @test sum(minCDaR.weights) ≈ 1

    retCDaR = EfficientCDaRReturn(tickers, mean_ret, ret)
    efficient_return!(effCDaR1)
    efficient_return!(retCDaR)
    mu3, cdar3 = portfolio_performance(effCDaR1)
    mu4, cdar4 = portfolio_performance(retCDaR)
    @test mu3 ≈ mu4
    @test cdar3 ≈ cdar4
    min_cdar!(retCDaR)
    mu4, cdar4 = portfolio_performance(retCDaR)
    @test mu4 ≈ mu2
    @test cdar4 ≈ cdar2
    @test sum(retCDaR.weights) ≈ 1

    riskCDaR = EfficientCDaRRisk(tickers, mean_ret, ret)
    efficient_risk!(effCDaR1)
    efficient_risk!(riskCDaR)
    mu5, cdar5 = portfolio_performance(effCDaR1)
    mu6, cdar6 = portfolio_performance(riskCDaR)
    @test mu5 ≈ mu6
    @test cdar5 ≈ cdar6
    @test sum(riskCDaR.weights) ≈ 1

    min_cdar!(riskCDaR)
    mu6, cdar6 = portfolio_performance(riskCDaR)
    @test mu6 ≈ mu2
    @test cdar6 ≈ cdar2
    @test sum(riskCDaR.weights) ≈ 1
end

@testset "Efficient CVaR" begin
    tickers = ["AAPL", "GOOG", "AMD", "MSFT"]
    sectors = ["Tech", "Food", "Oil", "Tech"]
    n = length(tickers)

    ret = (randn(1000, n) .+ 0.01) / 100
    mean_ret = ret_model(EMRet(), ret)

    effCVaR1 = EfficientCVaR(tickers, mean_ret, ret)
    minCVaR = MinCVaR(tickers, ret)
    min_cvar!(effCVaR1)
    min_cvar!(minCVaR)
    mu1, cvar1 = portfolio_performance(effCVaR1)
    mu2, cvar2 = portfolio_performance(minCVaR, mean_ret)
    @test mu1 ≈ mu2
    @test cvar1 ≈ cvar2
    @test sum(minCVaR.weights) ≈ 1

    retCVaR = EfficientCVaRReturn(tickers, mean_ret, ret)
    efficient_return!(effCVaR1)
    efficient_return!(retCVaR)
    mu3, cvar3 = portfolio_performance(effCVaR1)
    mu4, cvar4 = portfolio_performance(retCVaR)
    @test mu3 ≈ mu4
    @test cvar3 ≈ cvar4
    @test sum(retCVaR.weights) ≈ 1

    min_cvar!(retCVaR)
    mu4, cvar4 = portfolio_performance(retCVaR)
    @test mu4 ≈ mu2
    @test cvar4 ≈ cvar2
    @test sum(retCVaR.weights) ≈ 1

    riskCVaR = EfficientCVaRRisk(tickers, mean_ret, ret)
    efficient_risk!(effCVaR1)
    efficient_risk!(riskCVaR)
    mu5, cvar5 = portfolio_performance(effCVaR1)
    mu6, cvar6 = portfolio_performance(riskCVaR)
    @test mu5 ≈ mu6
    @test cvar5 ≈ cvar6
    @test sum(riskCVaR.weights) ≈ 1

    min_cvar!(riskCVaR)
    mu6, cvar6 = portfolio_performance(riskCVaR)
    @test mu6 ≈ mu2
    @test cvar6 ≈ cvar2
    @test sum(riskCVaR.weights) ≈ 1
end
