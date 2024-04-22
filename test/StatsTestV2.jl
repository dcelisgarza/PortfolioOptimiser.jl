using CSV, TimeSeries, StatsBase, Statistics, CovarianceEstimation, LinearAlgebra, Test,
      PortfolioOptimiser

prices = TimeArray(CSV.File("./assets/stock_prices.csv"); timestamp = :date)

@testset "New Expected Returns Vector Unweighted" begin
    portfolio = Portfolio(; prices = prices)
    ret = portfolio.returns

    cv0 = cov(ret)
    mu0 = vec(mean(ret; dims = 1))

    me1 = SimpleMean(; w = nothing)
    me1_1 = MuOpt(; method = :Default)
    target = TargetGM()
    me2 = MeanJS(; target = target, w = nothing, sigma = cv0)
    me2_1 = MuOpt(; method = :JS, sigma = cv0)
    me3 = MeanBS(; target = target, w = nothing, sigma = cv0)
    me3_1 = MuOpt(; method = :BS, sigma = cv0)
    me4 = MeanBOP(; target = target, w = nothing, sigma = cv0)
    me4_1 = MuOpt(; method = :BOP, sigma = cv0)
    target = TargetVW()
    me5 = MeanJS(; target = target, w = nothing, sigma = cv0)
    me5_1 = MuOpt(; method = :JS, target = :VW, sigma = cv0)
    me6 = MeanBS(; target = target, w = nothing, sigma = cv0)
    me6_1 = MuOpt(; method = :BS, target = :VW, sigma = cv0)
    me7 = MeanBOP(; target = target, w = nothing, sigma = cv0)
    me7_1 = MuOpt(; method = :BOP, target = :VW, sigma = cv0)
    target = TargetSE()
    me8 = MeanJS(; target = target, w = nothing, sigma = cv0)
    me8_1 = MuOpt(; method = :JS, target = :SE, sigma = cv0)
    me9 = MeanBS(; target = target, w = nothing, sigma = cv0)
    me9_1 = MuOpt(; method = :BS, target = :SE, sigma = cv0)
    me10 = MeanBOP(; target = target, w = nothing, sigma = cv0)
    me10_1 = MuOpt(; method = :BOP, target = :SE, sigma = cv0)

    mu1 = mean(me1, ret)
    mu1_1 = PortfolioOptimiser.mean_vec(ret, me1_1)
    @test isapprox(mu1, mu0)
    @test isapprox(mu1, mu1_1)

    mu2 = mean(me2, ret)
    mu2_1 = PortfolioOptimiser.mean_vec(ret, me2_1)
    @test isapprox(mu2, mu2_1)

    mu3 = mean(me3, ret)
    mu3_1 = PortfolioOptimiser.mean_vec(ret, me3_1)
    @test isapprox(mu3, mu3_1)

    mu4 = mean(me4, ret)
    mu4_1 = PortfolioOptimiser.mean_vec(ret, me4_1)
    @test isapprox(mu4, mu4_1)

    mu5 = mean(me5, ret)
    mu5_1 = PortfolioOptimiser.mean_vec(ret, me5_1)
    @test isapprox(mu5, mu5_1)

    mu6 = mean(me6, ret)
    mu6_1 = PortfolioOptimiser.mean_vec(ret, me6_1)
    @test isapprox(mu6, mu6_1)

    mu7 = mean(me7, ret)
    mu7_1 = PortfolioOptimiser.mean_vec(ret, me7_1)
    @test isapprox(mu7, mu7_1)

    mu8 = mean(me8, ret)
    mu8_1 = PortfolioOptimiser.mean_vec(ret, me8_1)
    @test isapprox(mu8, mu8_1)

    mu9 = mean(me9, ret)
    mu9_1 = PortfolioOptimiser.mean_vec(ret, me9_1)
    @test isapprox(mu9, mu9_1)

    mu10 = mean(me10, ret)
    mu10_1 = PortfolioOptimiser.mean_vec(ret, me10_1)
    @test isapprox(mu10, mu10_1)
end

@testset "New Expected Returns Vector Weighted" begin
    portfolio = Portfolio(; prices = prices)
    ret = portfolio.returns

    w = eweights(size(ret, 1), 1 / size(ret, 1); scale = true)

    cv0 = cov(ret, w)
    mu0 = vec(mean(ret, w; dims = 1))

    me1 = SimpleMean(; w = w)
    me1_1 = MuOpt(;
                  genfunc = GenericFunction(; func = StatsBase.mean, args = (w,),
                                            kwargs = (; dims = 1)), method = :Default)
    target = TargetGM()
    me2 = MeanJS(; target = target, w = w, sigma = cv0)
    me2_1 = MuOpt(;
                  genfunc = GenericFunction(; func = StatsBase.mean, args = (w,),
                                            kwargs = (; dims = 1)), method = :JS,
                  sigma = cv0)
    me3 = MeanBS(; target = target, w = w, sigma = cv0)
    me3_1 = MuOpt(;
                  genfunc = GenericFunction(; func = StatsBase.mean, args = (w,),
                                            kwargs = (; dims = 1)), method = :BS,
                  sigma = cv0)
    me4 = MeanBOP(; target = target, w = w, sigma = cv0)
    me4_1 = MuOpt(;
                  genfunc = GenericFunction(; func = StatsBase.mean, args = (w,),
                                            kwargs = (; dims = 1)), method = :BOP,
                  sigma = cv0)
    target = TargetVW()
    me5 = MeanJS(; target = target, w = w, sigma = cv0)
    me5_1 = MuOpt(;
                  genfunc = GenericFunction(; func = StatsBase.mean, args = (w,),
                                            kwargs = (; dims = 1)), method = :JS,
                  target = :VW, sigma = cv0)
    me6 = MeanBS(; target = target, w = w, sigma = cv0)
    me6_1 = MuOpt(;
                  genfunc = GenericFunction(; func = StatsBase.mean, args = (w,),
                                            kwargs = (; dims = 1)), method = :BS,
                  target = :VW, sigma = cv0)
    me7 = MeanBOP(; target = target, w = w, sigma = cv0)
    me7_1 = MuOpt(;
                  genfunc = GenericFunction(; func = StatsBase.mean, args = (w,),
                                            kwargs = (; dims = 1)), method = :BOP,
                  target = :VW, sigma = cv0)
    target = TargetSE()
    me8 = MeanJS(; target = target, w = w, sigma = cv0)
    me8_1 = MuOpt(;
                  genfunc = GenericFunction(; func = StatsBase.mean, args = (w,),
                                            kwargs = (; dims = 1)), method = :JS,
                  target = :SE, sigma = cv0)
    me9 = MeanBS(; target = target, w = w, sigma = cv0)
    me9_1 = MuOpt(;
                  genfunc = GenericFunction(; func = StatsBase.mean, args = (w,),
                                            kwargs = (; dims = 1)), method = :BS,
                  target = :SE, sigma = cv0)
    me10 = MeanBOP(; target = target, w = w, sigma = cv0)
    me10_1 = MuOpt(;
                   genfunc = GenericFunction(; func = StatsBase.mean, args = (w,),
                                             kwargs = (; dims = 1)), method = :BOP,
                   target = :SE, sigma = cv0)

    mu1 = mean(me1, ret)
    mu1_1 = PortfolioOptimiser.mean_vec(ret, me1_1)
    @test isapprox(mu1, mu0)
    @test isapprox(mu1, mu1_1)

    mu2 = mean(me2, ret)
    mu2_1 = PortfolioOptimiser.mean_vec(ret, me2_1)
    @test isapprox(mu2, mu2_1)

    mu3 = mean(me3, ret)
    mu3_1 = PortfolioOptimiser.mean_vec(ret, me3_1)
    @test isapprox(mu3, mu3_1)

    mu4 = mean(me4, ret)
    mu4_1 = PortfolioOptimiser.mean_vec(ret, me4_1)
    @test isapprox(mu4, mu4_1)

    mu5 = mean(me5, ret)
    mu5_1 = PortfolioOptimiser.mean_vec(ret, me5_1)
    @test isapprox(mu5, mu5_1)

    mu6 = mean(me6, ret)
    mu6_1 = PortfolioOptimiser.mean_vec(ret, me6_1)
    @test isapprox(mu6, mu6_1)

    mu7 = mean(me7, ret)
    mu7_1 = PortfolioOptimiser.mean_vec(ret, me7_1)
    @test isapprox(mu7, mu7_1)

    mu8 = mean(me8, ret)
    mu8_1 = PortfolioOptimiser.mean_vec(ret, me8_1)
    @test isapprox(mu8, mu8_1)

    mu9 = mean(me9, ret)
    mu9_1 = PortfolioOptimiser.mean_vec(ret, me9_1)
    @test isapprox(mu9, mu9_1)

    mu10 = mean(me10, ret)
    mu10_1 = PortfolioOptimiser.mean_vec(ret, me10_1)
    @test isapprox(mu10, mu10_1)
end

@testset "Covariance Matrix Unweighted" begin
    portfolio = Portfolio(; prices = prices)
    ret = portfolio.returns

    rf = 0.0329 / 252
    c0 = StatsBase.SimpleCovariance(; corrected = true)
    cv0 = cov(c0, ret)

    c1 = CovFull(; ce = c0)
    c1_1 = CovOpt(; method = :Full, estimation = CovEstOpt(; estimator = c0))
    c2 = CovSemi(; ce = c0, target = rf)
    c2_1 = CovOpt(; method = :Semi,
                  estimation = CovEstOpt(; estimator = c0, target_ret = rf))
    normalise = false
    c3 = CorGerber0(; normalise = normalise)
    c3_1 = CovOpt(; method = :Gerber0, gerber = GerberOpt(; normalise = normalise))
    c4 = CorGerber1(; normalise = normalise)
    c4_1 = CovOpt(; method = :Gerber1, gerber = GerberOpt(; normalise = normalise))
    c4 = CorGerber2(; normalise = normalise)
    c4_1 = CovOpt(; method = :Gerber2, gerber = GerberOpt(; normalise = normalise))
    c5 = CorSB0(; normalise = normalise)
    c5_1 = CovOpt(; method = :SB0, gerber = GerberOpt(; normalise = normalise))
    c6 = CorSB1(; normalise = normalise)
    c6_1 = CovOpt(; method = :SB1, gerber = GerberOpt(; normalise = normalise))
    c7 = CorGerberSB0(; normalise = normalise)
    c7_1 = CovOpt(; method = :Gerber_SB0, gerber = GerberOpt(; normalise = normalise))
    c8 = CorGerberSB1(; normalise = normalise)
    c8_1 = CovOpt(; method = :Gerber_SB1, gerber = GerberOpt(; normalise = normalise))
    normalise = true
    c9 = CorGerber0(; normalise = normalise)
    c9_1 = CovOpt(; method = :Gerber0, gerber = GerberOpt(; normalise = normalise))
    c10 = CorGerber1(; normalise = normalise)
    c10_1 = CovOpt(; method = :Gerber1, gerber = GerberOpt(; normalise = normalise))
    c11 = CorGerber2(; normalise = normalise)
    c11_1 = CovOpt(; method = :Gerber2, gerber = GerberOpt(; normalise = normalise))
    c12 = CorSB0(; normalise = normalise)
    c12_1 = CovOpt(; method = :SB0, gerber = GerberOpt(; normalise = normalise))
    c13 = CorSB1(; normalise = normalise)
    c13_1 = CovOpt(; method = :SB1, gerber = GerberOpt(; normalise = normalise))
    c14 = CorGerberSB0(; normalise = normalise)
    c14_1 = CovOpt(; method = :Gerber_SB0, gerber = GerberOpt(; normalise = normalise))
    c15 = CorGerberSB1(; normalise = normalise)
    c15_1 = CovOpt(; method = :Gerber_SB1, gerber = GerberOpt(; normalise = normalise))

    cv1 = cov(c1, ret)
    cv1_1 = PortfolioOptimiser.covar_mtx(ret, c1_1)
    @test isapprox(cv1, cv0)
    @test isapprox(cv1, cv1_1)

    cv2 = cov(c2, ret)
    cv2_1 = PortfolioOptimiser.covar_mtx(ret, c2_1)
    @test isapprox(cv2, cv2_1)

    cv3 = cov(c3, ret)
    cv3_1 = PortfolioOptimiser.covar_mtx(ret, c3_1)
    @test isapprox(cv3, cv3_1)

    cv4 = cov(c4, ret)
    cv4_1 = PortfolioOptimiser.covar_mtx(ret, c4_1)
    @test isapprox(cv4, cv4_1)

    cv5 = cov(c5, ret)
    cv5_1 = PortfolioOptimiser.covar_mtx(ret, c5_1)
    @test isapprox(cv5, cv5_1)

    cv6 = cov(c6, ret)
    cv6_1 = PortfolioOptimiser.covar_mtx(ret, c6_1)
    @test isapprox(cv6, cv6_1)

    cv7 = cov(c7, ret)
    cv7_1 = PortfolioOptimiser.covar_mtx(ret, c7_1)
    @test isapprox(cv7, cv7_1)

    cv8 = cov(c8, ret)
    cv8_1 = PortfolioOptimiser.covar_mtx(ret, c8_1)
    @test isapprox(cv8, cv8_1)

    cv9 = cov(c9, ret)
    cv9_1 = PortfolioOptimiser.covar_mtx(ret, c9_1)
    @test isapprox(cv9, cv9_1)

    cv10 = cov(c10, ret)
    cv10_1 = PortfolioOptimiser.covar_mtx(ret, c10_1)
    @test isapprox(cv10, cv10_1)

    cv11 = cov(c11, ret)
    cv11_1 = PortfolioOptimiser.covar_mtx(ret, c11_1)
    @test isapprox(cv11, cv11_1)

    cv12 = cov(c12, ret)
    cv12_1 = PortfolioOptimiser.covar_mtx(ret, c12_1)
    @test isapprox(cv12, cv12_1)

    cv13 = cov(c13, ret)
    cv13_1 = PortfolioOptimiser.covar_mtx(ret, c13_1)
    @test isapprox(cv13, cv13_1)

    cv14 = cov(c14, ret)
    cv14_1 = PortfolioOptimiser.covar_mtx(ret, c14_1)
    @test isapprox(cv14, cv14_1)
end