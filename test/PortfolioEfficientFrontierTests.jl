using COSMO, CSV, Clarabel, HiGHS, LinearAlgebra, OrderedCollections, PortfolioOptimiser,
      Statistics, Test, TimeSeries, Logging

Logging.disable_logging(Logging.Warn)

prices = TimeArray(CSV.File("./assets/stock_prices.csv"); timestamp = :date)

rf = 1.0329^(1 / 252) - 1
l = 2.0

@testset "$(:Classic), $(:Trad), $(:SD)" begin
    portfolio = Portfolio(; prices = prices,
                          solvers = OrderedDict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                                  :params => Dict("verbose" => false,
                                                                                  "max_step_fraction" => 0.75)),
                                                :COSMO => Dict(:solver => COSMO.Optimizer,
                                                               :params => Dict("verbose" => false))))
    asset_statistics!(portfolio)
    returns = portfolio.returns
    sigma = portfolio.cov
    alpha_i = portfolio.alpha_i
    alpha = portfolio.alpha
    a_sim = portfolio.a_sim
    beta_i = portfolio.beta_i
    beta = portfolio.beta
    b_sim = portfolio.b_sim
    kappa = portfolio.kappa
    owa_w = portfolio.owa_w
    solvers = portfolio.solvers

    rm = :SD
    opt = OptimiseOpt(; rf = rf, l = l, class = :Classic, hist = 1, type = :Trad,
                      rrp_ver = :None, u_mu = :None, u_cov = :None, rm = rm,
                      obj = :Min_Risk, kelly = :None)
    w1 = optimise!(portfolio, opt)
    opt.obj = :Max_Ret
    w2 = optimise!(portfolio, opt)
    opt.obj = :Sharpe
    w3 = optimise!(portfolio, opt)
    fw1 = efficient_frontier!(portfolio, opt; points = 5)

    risks1 = [calc_risk(fw1[:weights][!, i], returns; rm = :SD, rf = rf, sigma = sigma,
                        alpha_i = alpha_i, alpha = alpha, a_sim = a_sim, beta_i = beta_i,
                        beta = beta, b_sim = b_sim, kappa = kappa, owa_w = owa_w,
                        solvers = solvers,) for i ∈ 2:size(Matrix(fw1[:weights]), 2)]
    rets1 = [dot(fw1[:weights][!, i], portfolio.mu)
             for i ∈ 2:size(Matrix(fw1[:weights]), 2)]
    idx = findlast(x -> x < risks1[end], risks1) + 1
    tmp = risks1[end]
    risks1[(idx + 1):end] = risks1[idx:(end - 1)]
    risks1[idx] = tmp
    tmp = rets1[end]
    rets1[(idx + 1):end] = rets1[idx:(end - 1)]
    rets1[idx] = tmp

    @test reduce(<=, risks1)
    @test reduce(<=, rets1)
    @test isapprox(fw1[:weights][!, 2], w1.weights)
    @test isapprox(fw1[:weights][!, end - 1], w2.weights)
    @test isapprox(fw1[:weights][!, end], w3.weights)
end

@testset "$(:Classic), $(:Trad), $(:MAD)" begin
    portfolio = Portfolio(; prices = prices,
                          solvers = OrderedDict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                                  :params => Dict("verbose" => false,
                                                                                  "max_step_fraction" => 0.75)),
                                                :COSMO => Dict(:solver => COSMO.Optimizer,
                                                               :params => Dict("verbose" => false))))
    asset_statistics!(portfolio)
    returns = portfolio.returns
    sigma = portfolio.cov
    alpha_i = portfolio.alpha_i
    alpha = portfolio.alpha
    a_sim = portfolio.a_sim
    beta_i = portfolio.beta_i
    beta = portfolio.beta
    b_sim = portfolio.b_sim
    kappa = portfolio.kappa
    owa_w = portfolio.owa_w
    solvers = portfolio.solvers

    rm = :MAD
    opt = OptimiseOpt(; rf = rf, l = l, class = :Classic, hist = 1, type = :Trad,
                      rrp_ver = :None, u_mu = :None, u_cov = :None, rm = rm,
                      obj = :Min_Risk, kelly = :None)
    w1 = optimise!(portfolio, opt)
    opt.obj = :Max_Ret
    w2 = optimise!(portfolio, opt)
    opt.obj = :Sharpe
    w3 = optimise!(portfolio, opt)
    fw1 = efficient_frontier!(portfolio, opt; points = 5)

    risks1 = [calc_risk(fw1[:weights][!, i], returns; rm = :MAD, rf = rf, sigma = sigma,
                        alpha_i = alpha_i, alpha = alpha, a_sim = a_sim, beta_i = beta_i,
                        beta = beta, b_sim = b_sim, kappa = kappa, owa_w = owa_w,
                        solvers = solvers,) for i ∈ 2:size(Matrix(fw1[:weights]), 2)]
    rets1 = [dot(fw1[:weights][!, i], portfolio.mu)
             for i ∈ 2:size(Matrix(fw1[:weights]), 2)]
    idx = findlast(x -> x < risks1[end], risks1) + 1
    tmp = risks1[end]
    risks1[(idx + 1):end] = risks1[idx:(end - 1)]
    risks1[idx] = tmp
    tmp = rets1[end]
    rets1[(idx + 1):end] = rets1[idx:(end - 1)]
    rets1[idx] = tmp

    @test reduce(<=, risks1)
    @test reduce(<=, rets1)
    @test isapprox(fw1[:weights][!, 2], w1.weights)
    @test isapprox(fw1[:weights][!, end - 1], w2.weights)
    @test isapprox(fw1[:weights][!, end], w3.weights)
end

@testset "$(:Classic), $(:Trad), $(:SSD)" begin
    portfolio = Portfolio(; prices = prices,
                          solvers = OrderedDict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                                  :params => Dict("verbose" => false,
                                                                                  "max_step_fraction" => 0.75)),
                                                :COSMO => Dict(:solver => COSMO.Optimizer,
                                                               :params => Dict("verbose" => false))))
    asset_statistics!(portfolio)
    returns = portfolio.returns
    sigma = portfolio.cov
    alpha_i = portfolio.alpha_i
    alpha = portfolio.alpha
    a_sim = portfolio.a_sim
    beta_i = portfolio.beta_i
    beta = portfolio.beta
    b_sim = portfolio.b_sim
    kappa = portfolio.kappa
    owa_w = portfolio.owa_w
    solvers = portfolio.solvers

    rm = :SSD
    opt = OptimiseOpt(; rf = rf, l = l, class = :Classic, hist = 1, type = :Trad,
                      rrp_ver = :None, u_mu = :None, u_cov = :None, rm = rm,
                      obj = :Min_Risk, kelly = :None)
    w1 = optimise!(portfolio, opt)
    opt.obj = :Max_Ret
    w2 = optimise!(portfolio, opt)
    opt.obj = :Sharpe
    w3 = optimise!(portfolio, opt)
    fw1 = efficient_frontier!(portfolio, opt; points = 5)

    risks1 = [calc_risk(fw1[:weights][!, i], returns; rm = :SSD, rf = rf, sigma = sigma,
                        alpha_i = alpha_i, alpha = alpha, a_sim = a_sim, beta_i = beta_i,
                        beta = beta, b_sim = b_sim, kappa = kappa, owa_w = owa_w,
                        solvers = solvers,) for i ∈ 2:size(Matrix(fw1[:weights]), 2)]
    rets1 = [dot(fw1[:weights][!, i], portfolio.mu)
             for i ∈ 2:size(Matrix(fw1[:weights]), 2)]
    idx = findlast(x -> x < risks1[end], risks1) + 1
    tmp = risks1[end]
    risks1[(idx + 1):end] = risks1[idx:(end - 1)]
    risks1[idx] = tmp
    tmp = rets1[end]
    rets1[(idx + 1):end] = rets1[idx:(end - 1)]
    rets1[idx] = tmp

    @test reduce(<=, risks1)
    @test reduce(<=, rets1)
    @test isapprox(fw1[:weights][!, 2], w1.weights)
    @test isapprox(fw1[:weights][!, end - 1], w2.weights)
    @test isapprox(fw1[:weights][!, end], w3.weights)
end

@testset "$(:Classic), $(:Trad), $(:FLPM)" begin
    portfolio = Portfolio(; prices = prices,
                          solvers = OrderedDict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                                  :params => Dict("verbose" => false,
                                                                                  "max_step_fraction" => 0.75)),
                                                :COSMO => Dict(:solver => COSMO.Optimizer,
                                                               :params => Dict("verbose" => false))))
    asset_statistics!(portfolio)
    returns = portfolio.returns
    sigma = portfolio.cov
    alpha_i = portfolio.alpha_i
    alpha = portfolio.alpha
    a_sim = portfolio.a_sim
    beta_i = portfolio.beta_i
    beta = portfolio.beta
    b_sim = portfolio.b_sim
    kappa = portfolio.kappa
    owa_w = portfolio.owa_w
    solvers = portfolio.solvers

    rm = :FLPM
    opt = OptimiseOpt(; rf = rf, l = l, class = :Classic, hist = 1, type = :Trad,
                      rrp_ver = :None, u_mu = :None, u_cov = :None, rm = rm,
                      obj = :Min_Risk, kelly = :None)
    w1 = optimise!(portfolio, opt)
    opt.obj = :Max_Ret
    w2 = optimise!(portfolio, opt)
    opt.obj = :Sharpe
    w3 = optimise!(portfolio, opt)
    fw1 = efficient_frontier!(portfolio, opt; points = 5)

    risks1 = [calc_risk(fw1[:weights][!, i], returns; rm = :FLPM, rf = rf, sigma = sigma,
                        alpha_i = alpha_i, alpha = alpha, a_sim = a_sim, beta_i = beta_i,
                        beta = beta, b_sim = b_sim, kappa = kappa, owa_w = owa_w,
                        solvers = solvers,) for i ∈ 2:size(Matrix(fw1[:weights]), 2)]
    rets1 = [dot(fw1[:weights][!, i], portfolio.mu)
             for i ∈ 2:size(Matrix(fw1[:weights]), 2)]
    idx = findlast(x -> x < risks1[end], risks1) + 1
    tmp = risks1[end]
    risks1[(idx + 1):end] = risks1[idx:(end - 1)]
    risks1[idx] = tmp
    tmp = rets1[end]
    rets1[(idx + 1):end] = rets1[idx:(end - 1)]
    rets1[idx] = tmp

    @test reduce(<=, risks1)
    @test reduce(<=, rets1)
    @test isapprox(fw1[:weights][!, 2], w1.weights)
    @test isapprox(fw1[:weights][!, end - 1], w2.weights)
    @test isapprox(fw1[:weights][!, end], w3.weights)
end

@testset "$(:Classic), $(:Trad), $(:SLPM)" begin
    portfolio = Portfolio(; prices = prices,
                          solvers = OrderedDict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                                  :params => Dict("verbose" => false,
                                                                                  "max_step_fraction" => 0.75)),
                                                :COSMO => Dict(:solver => COSMO.Optimizer,
                                                               :params => Dict("verbose" => false))))
    asset_statistics!(portfolio)
    returns = portfolio.returns
    sigma = portfolio.cov
    alpha_i = portfolio.alpha_i
    alpha = portfolio.alpha
    a_sim = portfolio.a_sim
    beta_i = portfolio.beta_i
    beta = portfolio.beta
    b_sim = portfolio.b_sim
    kappa = portfolio.kappa
    owa_w = portfolio.owa_w
    solvers = portfolio.solvers

    rm = :SLPM
    opt = OptimiseOpt(; rf = rf, l = l, class = :Classic, hist = 1, type = :Trad,
                      rrp_ver = :None, u_mu = :None, u_cov = :None, rm = rm,
                      obj = :Min_Risk, kelly = :None)
    w1 = optimise!(portfolio, opt)
    opt.obj = :Max_Ret
    w2 = optimise!(portfolio, opt)
    opt.obj = :Sharpe
    w3 = optimise!(portfolio, opt)
    fw1 = efficient_frontier!(portfolio, opt; points = 5)

    risks1 = [calc_risk(fw1[:weights][!, i], returns; rm = :SLPM, rf = rf, sigma = sigma,
                        alpha_i = alpha_i, alpha = alpha, a_sim = a_sim, beta_i = beta_i,
                        beta = beta, b_sim = b_sim, kappa = kappa, owa_w = owa_w,
                        solvers = solvers,) for i ∈ 2:size(Matrix(fw1[:weights]), 2)]
    rets1 = [dot(fw1[:weights][!, i], portfolio.mu)
             for i ∈ 2:size(Matrix(fw1[:weights]), 2)]
    idx = findlast(x -> x < risks1[end], risks1) + 1
    tmp = risks1[end]
    risks1[(idx + 1):end] = risks1[idx:(end - 1)]
    risks1[idx] = tmp
    tmp = rets1[end]
    rets1[(idx + 1):end] = rets1[idx:(end - 1)]
    rets1[idx] = tmp

    @test reduce(<=, risks1)
    @test reduce(<=, rets1)
    @test isapprox(fw1[:weights][!, 2], w1.weights)
    @test isapprox(fw1[:weights][!, end - 1], w2.weights)
    @test isapprox(fw1[:weights][!, end], w3.weights)
end

@testset "$(:Classic), $(:Trad), $(:WR)" begin
    portfolio = Portfolio(; prices = prices,
                          solvers = OrderedDict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                                  :params => Dict("verbose" => false,
                                                                                  "max_step_fraction" => 0.75)),
                                                :COSMO => Dict(:solver => COSMO.Optimizer,
                                                               :params => Dict("verbose" => false))))
    asset_statistics!(portfolio)
    returns = portfolio.returns
    sigma = portfolio.cov
    alpha_i = portfolio.alpha_i
    alpha = portfolio.alpha
    a_sim = portfolio.a_sim
    beta_i = portfolio.beta_i
    beta = portfolio.beta
    b_sim = portfolio.b_sim
    kappa = portfolio.kappa
    owa_w = portfolio.owa_w
    solvers = portfolio.solvers

    rm = :WR
    opt = OptimiseOpt(; rf = rf, l = l, class = :Classic, hist = 1, type = :Trad,
                      rrp_ver = :None, u_mu = :None, u_cov = :None, rm = rm,
                      obj = :Min_Risk, kelly = :None)
    w1 = optimise!(portfolio, opt)
    opt.obj = :Max_Ret
    w2 = optimise!(portfolio, opt)
    opt.obj = :Sharpe
    w3 = optimise!(portfolio, opt)
    fw1 = efficient_frontier!(portfolio, opt; points = 5)

    risks1 = [calc_risk(fw1[:weights][!, i], returns; rm = :WR, rf = rf, sigma = sigma,
                        alpha_i = alpha_i, alpha = alpha, a_sim = a_sim, beta_i = beta_i,
                        beta = beta, b_sim = b_sim, kappa = kappa, owa_w = owa_w,
                        solvers = solvers,) for i ∈ 2:size(Matrix(fw1[:weights]), 2)]
    rets1 = [dot(fw1[:weights][!, i], portfolio.mu)
             for i ∈ 2:size(Matrix(fw1[:weights]), 2)]
    idx = findlast(x -> x < risks1[end], risks1) + 1
    tmp = risks1[end]
    risks1[(idx + 1):end] = risks1[idx:(end - 1)]
    risks1[idx] = tmp
    tmp = rets1[end]
    rets1[(idx + 1):end] = rets1[idx:(end - 1)]
    rets1[idx] = tmp

    @test reduce(<=, risks1)
    @test reduce(<=, rets1)
    @test isapprox(fw1[:weights][!, 2], w1.weights)
    @test isapprox(fw1[:weights][!, end - 1], w2.weights)
    @test isapprox(fw1[:weights][!, end], w3.weights)
end

@testset "$(:Classic), $(:Trad), $(:CVaR)" begin
    portfolio = Portfolio(; prices = prices,
                          solvers = OrderedDict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                                  :params => Dict("verbose" => false,
                                                                                  "max_step_fraction" => 0.75)),
                                                :COSMO => Dict(:solver => COSMO.Optimizer,
                                                               :params => Dict("verbose" => false))))
    asset_statistics!(portfolio)
    returns = portfolio.returns
    sigma = portfolio.cov
    alpha_i = portfolio.alpha_i
    alpha = portfolio.alpha
    a_sim = portfolio.a_sim
    beta_i = portfolio.beta_i
    beta = portfolio.beta
    b_sim = portfolio.b_sim
    kappa = portfolio.kappa
    owa_w = portfolio.owa_w
    solvers = portfolio.solvers

    rm = :CVaR
    opt = OptimiseOpt(; rf = rf, l = l, class = :Classic, hist = 1, type = :Trad,
                      rrp_ver = :None, u_mu = :None, u_cov = :None, rm = rm,
                      obj = :Min_Risk, kelly = :None)
    w1 = optimise!(portfolio, opt)
    opt.obj = :Max_Ret
    w2 = optimise!(portfolio, opt)
    opt.obj = :Sharpe
    w3 = optimise!(portfolio, opt)
    fw1 = efficient_frontier!(portfolio, opt; points = 5)

    risks1 = [calc_risk(fw1[:weights][!, i], returns; rm = :CVaR, rf = rf, sigma = sigma,
                        alpha_i = alpha_i, alpha = alpha, a_sim = a_sim, beta_i = beta_i,
                        beta = beta, b_sim = b_sim, kappa = kappa, owa_w = owa_w,
                        solvers = solvers,) for i ∈ 2:size(Matrix(fw1[:weights]), 2)]
    rets1 = [dot(fw1[:weights][!, i], portfolio.mu)
             for i ∈ 2:size(Matrix(fw1[:weights]), 2)]
    idx = findlast(x -> x < risks1[end], risks1) + 1
    tmp = risks1[end]
    risks1[(idx + 1):end] = risks1[idx:(end - 1)]
    risks1[idx] = tmp
    tmp = rets1[end]
    rets1[(idx + 1):end] = rets1[idx:(end - 1)]
    rets1[idx] = tmp

    @test reduce(<=, risks1)
    @test reduce(<=, rets1)
    @test isapprox(fw1[:weights][!, 2], w1.weights)
    @test isapprox(fw1[:weights][!, end - 1], w2.weights)
    @test isapprox(fw1[:weights][!, end], w3.weights)
end

@testset "$(:Classic), $(:Trad), $(:EVaR)" begin
    portfolio = Portfolio(; prices = prices,
                          solvers = OrderedDict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                                  :params => Dict("verbose" => false,
                                                                                  "max_step_fraction" => 0.75)),
                                                :COSMO => Dict(:solver => COSMO.Optimizer,
                                                               :params => Dict("verbose" => false))))
    asset_statistics!(portfolio)
    returns = portfolio.returns
    sigma = portfolio.cov
    alpha_i = portfolio.alpha_i
    alpha = portfolio.alpha
    a_sim = portfolio.a_sim
    beta_i = portfolio.beta_i
    beta = portfolio.beta
    b_sim = portfolio.b_sim
    kappa = portfolio.kappa
    owa_w = portfolio.owa_w
    solvers = portfolio.solvers

    rm = :EVaR
    opt = OptimiseOpt(; rf = rf, l = l, class = :Classic, hist = 1, type = :Trad,
                      rrp_ver = :None, u_mu = :None, u_cov = :None, rm = rm,
                      obj = :Min_Risk, kelly = :None)
    w1 = optimise!(portfolio, opt)
    opt.obj = :Max_Ret
    w2 = optimise!(portfolio, opt)
    opt.obj = :Sharpe
    w3 = optimise!(portfolio, opt)
    fw1 = efficient_frontier!(portfolio, opt; points = 5)

    risks1 = [calc_risk(fw1[:weights][!, i], returns; rm = :EVaR, rf = rf, sigma = sigma,
                        alpha_i = alpha_i, alpha = alpha, a_sim = a_sim, beta_i = beta_i,
                        beta = beta, b_sim = b_sim, kappa = kappa, owa_w = owa_w,
                        solvers = solvers,) for i ∈ 2:size(Matrix(fw1[:weights]), 2)]
    rets1 = [dot(fw1[:weights][!, i], portfolio.mu)
             for i ∈ 2:size(Matrix(fw1[:weights]), 2)]
    idx = findlast(x -> x < risks1[end], risks1) + 1
    tmp = risks1[end]
    risks1[(idx + 1):end] = risks1[idx:(end - 1)]
    risks1[idx] = tmp
    tmp = rets1[end]
    rets1[(idx + 1):end] = rets1[idx:(end - 1)]
    rets1[idx] = tmp

    @test reduce(<=, risks1)
    @test reduce(<=, rets1)
    @test isapprox(fw1[:weights][!, 2], w1.weights)
    @test isapprox(fw1[:weights][!, end - 1], w2.weights)
    @test isapprox(fw1[:weights][!, end], w3.weights)
end

@testset "$(:Classic), $(:Trad), $(:RVaR)" begin
    portfolio = Portfolio(; prices = prices,
                          solvers = OrderedDict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                                  :params => Dict("verbose" => false,
                                                                                  "max_step_fraction" => 0.75)),
                                                :COSMO => Dict(:solver => COSMO.Optimizer,
                                                               :params => Dict("verbose" => false))))
    asset_statistics!(portfolio)
    returns = portfolio.returns
    sigma = portfolio.cov
    alpha_i = portfolio.alpha_i
    alpha = portfolio.alpha
    a_sim = portfolio.a_sim
    beta_i = portfolio.beta_i
    beta = portfolio.beta
    b_sim = portfolio.b_sim
    kappa = portfolio.kappa
    owa_w = portfolio.owa_w
    solvers = portfolio.solvers

    rm = :RVaR
    opt = OptimiseOpt(; rf = rf, l = l, class = :Classic, hist = 1, type = :Trad,
                      rrp_ver = :None, u_mu = :None, u_cov = :None, rm = rm,
                      obj = :Min_Risk, kelly = :None)
    w1 = optimise!(portfolio, opt)
    opt.obj = :Max_Ret
    w2 = optimise!(portfolio, opt)
    opt.obj = :Sharpe
    w3 = optimise!(portfolio, opt)
    fw1 = efficient_frontier!(portfolio, opt; points = 5)

    risks1 = [calc_risk(fw1[:weights][!, i], returns; rm = :RVaR, rf = rf, sigma = sigma,
                        alpha_i = alpha_i, alpha = alpha, a_sim = a_sim, beta_i = beta_i,
                        beta = beta, b_sim = b_sim, kappa = kappa, owa_w = owa_w,
                        solvers = solvers,) for i ∈ 2:size(Matrix(fw1[:weights]), 2)]
    rets1 = [dot(fw1[:weights][!, i], portfolio.mu)
             for i ∈ 2:size(Matrix(fw1[:weights]), 2)]
    idx = findlast(x -> x < risks1[end], risks1) + 1
    tmp = risks1[end]
    risks1[(idx + 1):end] = risks1[idx:(end - 1)]
    risks1[idx] = tmp
    tmp = rets1[end]
    rets1[(idx + 1):end] = rets1[idx:(end - 1)]
    rets1[idx] = tmp

    @test reduce(<=, risks1)
    @test reduce(<=, rets1)
    @test isapprox(fw1[:weights][!, 2], w1.weights)
    @test isapprox(fw1[:weights][!, end - 1], w2.weights)
    @test isapprox(fw1[:weights][!, end], w3.weights)
end

@testset "$(:Classic), $(:Trad), $(:MDD)" begin
    portfolio = Portfolio(; prices = prices,
                          solvers = OrderedDict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                                  :params => Dict("verbose" => false,
                                                                                  "max_step_fraction" => 0.75)),
                                                :COSMO => Dict(:solver => COSMO.Optimizer,
                                                               :params => Dict("verbose" => false))))
    asset_statistics!(portfolio)
    returns = portfolio.returns
    sigma = portfolio.cov
    alpha_i = portfolio.alpha_i
    alpha = portfolio.alpha
    a_sim = portfolio.a_sim
    beta_i = portfolio.beta_i
    beta = portfolio.beta
    b_sim = portfolio.b_sim
    kappa = portfolio.kappa
    owa_w = portfolio.owa_w
    solvers = portfolio.solvers

    rm = :MDD
    opt = OptimiseOpt(; rf = rf, l = l, class = :Classic, hist = 1, type = :Trad,
                      rrp_ver = :None, u_mu = :None, u_cov = :None, rm = rm,
                      obj = :Min_Risk, kelly = :None)
    w1 = optimise!(portfolio, opt)
    opt.obj = :Max_Ret
    w2 = optimise!(portfolio, opt)
    opt.obj = :Sharpe
    w3 = optimise!(portfolio, opt)
    fw1 = efficient_frontier!(portfolio, opt; points = 5)

    risks1 = [calc_risk(fw1[:weights][!, i], returns; rm = :MDD, rf = rf, sigma = sigma,
                        alpha_i = alpha_i, alpha = alpha, a_sim = a_sim, beta_i = beta_i,
                        beta = beta, b_sim = b_sim, kappa = kappa, owa_w = owa_w,
                        solvers = solvers,) for i ∈ 2:size(Matrix(fw1[:weights]), 2)]
    rets1 = [dot(fw1[:weights][!, i], portfolio.mu)
             for i ∈ 2:size(Matrix(fw1[:weights]), 2)]
    idx = findlast(x -> x < risks1[end], risks1) + 1
    tmp = risks1[end]
    risks1[(idx + 1):end] = risks1[idx:(end - 1)]
    risks1[idx] = tmp
    tmp = rets1[end]
    rets1[(idx + 1):end] = rets1[idx:(end - 1)]
    rets1[idx] = tmp

    @test reduce(<=, risks1)
    @test reduce(<=, rets1)
    @test isapprox(fw1[:weights][!, 2], w1.weights)
    @test isapprox(fw1[:weights][!, end - 1], w2.weights)
    @test isapprox(fw1[:weights][!, end], w3.weights)
end

@testset "$(:Classic), $(:Trad), $(:ADD)" begin
    portfolio = Portfolio(; prices = prices,
                          solvers = OrderedDict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                                  :params => Dict("verbose" => false,
                                                                                  "max_step_fraction" => 0.75)),
                                                :COSMO => Dict(:solver => COSMO.Optimizer,
                                                               :params => Dict("verbose" => false))))
    asset_statistics!(portfolio)
    returns = portfolio.returns
    sigma = portfolio.cov
    alpha_i = portfolio.alpha_i
    alpha = portfolio.alpha
    a_sim = portfolio.a_sim
    beta_i = portfolio.beta_i
    beta = portfolio.beta
    b_sim = portfolio.b_sim
    kappa = portfolio.kappa
    owa_w = portfolio.owa_w
    solvers = portfolio.solvers

    rm = :ADD
    opt = OptimiseOpt(; rf = rf, l = l, class = :Classic, hist = 1, type = :Trad,
                      rrp_ver = :None, u_mu = :None, u_cov = :None, rm = rm,
                      obj = :Min_Risk, kelly = :None)
    w1 = optimise!(portfolio, opt)
    opt.obj = :Max_Ret
    w2 = optimise!(portfolio, opt)
    opt.obj = :Sharpe
    w3 = optimise!(portfolio, opt)
    fw1 = efficient_frontier!(portfolio, opt; points = 5)

    risks1 = [calc_risk(fw1[:weights][!, i], returns; rm = :ADD, rf = rf, sigma = sigma,
                        alpha_i = alpha_i, alpha = alpha, a_sim = a_sim, beta_i = beta_i,
                        beta = beta, b_sim = b_sim, kappa = kappa, owa_w = owa_w,
                        solvers = solvers,) for i ∈ 2:size(Matrix(fw1[:weights]), 2)]
    rets1 = [dot(fw1[:weights][!, i], portfolio.mu)
             for i ∈ 2:size(Matrix(fw1[:weights]), 2)]
    idx = findlast(x -> x < risks1[end], risks1) + 1
    tmp = risks1[end]
    risks1[(idx + 1):end] = risks1[idx:(end - 1)]
    risks1[idx] = tmp
    tmp = rets1[end]
    rets1[(idx + 1):end] = rets1[idx:(end - 1)]
    rets1[idx] = tmp

    @test reduce(<=, risks1)
    @test reduce(<=, rets1)
    @test isapprox(fw1[:weights][!, 2], w1.weights)
    @test isapprox(fw1[:weights][!, end - 1], w2.weights)
    @test isapprox(fw1[:weights][!, end], w3.weights)
end

@testset "$(:Classic), $(:Trad), $(:CDaR)" begin
    portfolio = Portfolio(; prices = prices,
                          solvers = OrderedDict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                                  :params => Dict("verbose" => false,
                                                                                  "max_step_fraction" => 0.75)),
                                                :COSMO => Dict(:solver => COSMO.Optimizer,
                                                               :params => Dict("verbose" => false))))
    asset_statistics!(portfolio)
    returns = portfolio.returns
    sigma = portfolio.cov
    alpha_i = portfolio.alpha_i
    alpha = portfolio.alpha
    a_sim = portfolio.a_sim
    beta_i = portfolio.beta_i
    beta = portfolio.beta
    b_sim = portfolio.b_sim
    kappa = portfolio.kappa
    owa_w = portfolio.owa_w
    solvers = portfolio.solvers

    rm = :CDaR
    opt = OptimiseOpt(; rf = rf, l = l, class = :Classic, hist = 1, type = :Trad,
                      rrp_ver = :None, u_mu = :None, u_cov = :None, rm = rm,
                      obj = :Min_Risk, kelly = :None)
    w1 = optimise!(portfolio, opt)
    opt.obj = :Max_Ret
    w2 = optimise!(portfolio, opt)
    opt.obj = :Sharpe
    w3 = optimise!(portfolio, opt)
    fw1 = efficient_frontier!(portfolio, opt; points = 5)

    risks1 = [calc_risk(fw1[:weights][!, i], returns; rm = :CDaR, rf = rf, sigma = sigma,
                        alpha_i = alpha_i, alpha = alpha, a_sim = a_sim, beta_i = beta_i,
                        beta = beta, b_sim = b_sim, kappa = kappa, owa_w = owa_w,
                        solvers = solvers,) for i ∈ 2:size(Matrix(fw1[:weights]), 2)]
    rets1 = [dot(fw1[:weights][!, i], portfolio.mu)
             for i ∈ 2:size(Matrix(fw1[:weights]), 2)]
    idx = findlast(x -> x < risks1[end], risks1) + 1
    tmp = risks1[end]
    risks1[(idx + 1):end] = risks1[idx:(end - 1)]
    risks1[idx] = tmp
    tmp = rets1[end]
    rets1[(idx + 1):end] = rets1[idx:(end - 1)]
    rets1[idx] = tmp

    @test reduce(<=, risks1)
    @test reduce(<=, rets1)
    @test isapprox(fw1[:weights][!, 2], w1.weights)
    @test isapprox(fw1[:weights][!, end - 1], w2.weights)
    @test isapprox(fw1[:weights][!, end], w3.weights)
end

@testset "$(:Classic), $(:Trad), $(:UCI)" begin
    portfolio = Portfolio(; prices = prices,
                          solvers = OrderedDict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                                  :params => Dict("verbose" => false,
                                                                                  "max_step_fraction" => 0.75)),
                                                :COSMO => Dict(:solver => COSMO.Optimizer,
                                                               :params => Dict("verbose" => false))))
    asset_statistics!(portfolio)
    returns = portfolio.returns
    sigma = portfolio.cov
    alpha_i = portfolio.alpha_i
    alpha = portfolio.alpha
    a_sim = portfolio.a_sim
    beta_i = portfolio.beta_i
    beta = portfolio.beta
    b_sim = portfolio.b_sim
    kappa = portfolio.kappa
    owa_w = portfolio.owa_w
    solvers = portfolio.solvers

    rm = :UCI
    opt = OptimiseOpt(; rf = rf, l = l, class = :Classic, hist = 1, type = :Trad,
                      rrp_ver = :None, u_mu = :None, u_cov = :None, rm = rm,
                      obj = :Min_Risk, kelly = :None)
    w1 = optimise!(portfolio, opt)
    opt.obj = :Max_Ret
    w2 = optimise!(portfolio, opt)
    opt.obj = :Sharpe
    w3 = optimise!(portfolio, opt)
    fw1 = efficient_frontier!(portfolio, opt; points = 5)

    risks1 = [calc_risk(fw1[:weights][!, i], returns; rm = :UCI, rf = rf, sigma = sigma,
                        alpha_i = alpha_i, alpha = alpha, a_sim = a_sim, beta_i = beta_i,
                        beta = beta, b_sim = b_sim, kappa = kappa, owa_w = owa_w,
                        solvers = solvers,) for i ∈ 2:size(Matrix(fw1[:weights]), 2)]
    rets1 = [dot(fw1[:weights][!, i], portfolio.mu)
             for i ∈ 2:size(Matrix(fw1[:weights]), 2)]
    idx = findlast(x -> x < risks1[end], risks1) + 1
    tmp = risks1[end]
    risks1[(idx + 1):end] = risks1[idx:(end - 1)]
    risks1[idx] = tmp
    tmp = rets1[end]
    rets1[(idx + 1):end] = rets1[idx:(end - 1)]
    rets1[idx] = tmp

    @test reduce(<=, risks1)
    @test reduce(<=, rets1)
    @test isapprox(fw1[:weights][!, 2], w1.weights)
    @test isapprox(fw1[:weights][!, end - 1], w2.weights)
    @test isapprox(fw1[:weights][!, end], w3.weights)
end

@testset "$(:Classic), $(:Trad), $(:EDaR)" begin
    portfolio = Portfolio(; prices = prices,
                          solvers = OrderedDict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                                  :params => Dict("verbose" => false,
                                                                                  "max_step_fraction" => 0.75)),
                                                :COSMO => Dict(:solver => COSMO.Optimizer,
                                                               :params => Dict("verbose" => false))))
    asset_statistics!(portfolio)
    returns = portfolio.returns
    sigma = portfolio.cov
    alpha_i = portfolio.alpha_i
    alpha = portfolio.alpha
    a_sim = portfolio.a_sim
    beta_i = portfolio.beta_i
    beta = portfolio.beta
    b_sim = portfolio.b_sim
    kappa = portfolio.kappa
    owa_w = portfolio.owa_w
    solvers = portfolio.solvers

    rm = :EDaR
    opt = OptimiseOpt(; rf = rf, l = l, class = :Classic, hist = 1, type = :Trad,
                      rrp_ver = :None, u_mu = :None, u_cov = :None, rm = rm,
                      obj = :Min_Risk, kelly = :None)
    w1 = optimise!(portfolio, opt)
    opt.obj = :Max_Ret
    w2 = optimise!(portfolio, opt)
    opt.obj = :Sharpe
    w3 = optimise!(portfolio, opt)
    fw1 = efficient_frontier!(portfolio, opt; points = 5)

    risks1 = [calc_risk(fw1[:weights][!, i], returns; rm = :EDaR, rf = rf, sigma = sigma,
                        alpha_i = alpha_i, alpha = alpha, a_sim = a_sim, beta_i = beta_i,
                        beta = beta, b_sim = b_sim, kappa = kappa, owa_w = owa_w,
                        solvers = solvers,) for i ∈ 2:size(Matrix(fw1[:weights]), 2)]
    rets1 = [dot(fw1[:weights][!, i], portfolio.mu)
             for i ∈ 2:size(Matrix(fw1[:weights]), 2)]
    idx = findlast(x -> x < risks1[end], risks1) + 1
    tmp = risks1[end]
    risks1[(idx + 1):end] = risks1[idx:(end - 1)]
    risks1[idx] = tmp
    tmp = rets1[end]
    rets1[(idx + 1):end] = rets1[idx:(end - 1)]
    rets1[idx] = tmp

    @test reduce(<=, risks1)
    @test reduce(<=, rets1)
    @test isapprox(fw1[:weights][!, 2], w1.weights)
    @test isapprox(fw1[:weights][!, end - 1], w2.weights)
    @test isapprox(fw1[:weights][!, end], w3.weights)
end

@testset "$(:Classic), $(:Trad), $(:RDaR)" begin
    portfolio = Portfolio(; prices = prices,
                          solvers = OrderedDict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                                  :params => Dict("verbose" => false,
                                                                                  "max_step_fraction" => 0.75)),
                                                :COSMO => Dict(:solver => COSMO.Optimizer,
                                                               :params => Dict("verbose" => false))))
    asset_statistics!(portfolio)
    returns = portfolio.returns
    sigma = portfolio.cov
    alpha_i = portfolio.alpha_i
    alpha = portfolio.alpha
    a_sim = portfolio.a_sim
    beta_i = portfolio.beta_i
    beta = portfolio.beta
    b_sim = portfolio.b_sim
    kappa = portfolio.kappa
    owa_w = portfolio.owa_w
    solvers = portfolio.solvers

    rm = :RDaR
    opt = OptimiseOpt(; rf = rf, l = l, class = :Classic, hist = 1, type = :Trad,
                      rrp_ver = :None, u_mu = :None, u_cov = :None, rm = rm,
                      obj = :Min_Risk, kelly = :None)
    w1 = optimise!(portfolio, opt)
    opt.obj = :Max_Ret
    w2 = optimise!(portfolio, opt)
    opt.obj = :Sharpe
    w3 = optimise!(portfolio, opt)
    fw1 = efficient_frontier!(portfolio, opt; points = 5)

    risks1 = [calc_risk(fw1[:weights][!, i], returns; rm = :RDaR, rf = rf, sigma = sigma,
                        alpha_i = alpha_i, alpha = alpha, a_sim = a_sim, beta_i = beta_i,
                        beta = beta, b_sim = b_sim, kappa = kappa, owa_w = owa_w,
                        solvers = solvers,) for i ∈ 2:size(Matrix(fw1[:weights]), 2)]
    rets1 = [dot(fw1[:weights][!, i], portfolio.mu)
             for i ∈ 2:size(Matrix(fw1[:weights]), 2)]
    idx = findlast(x -> x < risks1[end], risks1) + 1
    tmp = risks1[end]
    risks1[(idx + 1):end] = risks1[idx:(end - 1)]
    risks1[idx] = tmp
    tmp = rets1[end]
    rets1[(idx + 1):end] = rets1[idx:(end - 1)]
    rets1[idx] = tmp

    @test reduce(<=, risks1)
    @test reduce(<=, rets1)
    @test isapprox(fw1[:weights][!, 2], w1.weights)
    @test isapprox(fw1[:weights][!, end - 1], w2.weights)
    @test isapprox(fw1[:weights][!, end], w3.weights)
end

@testset "$(:Classic), $(:Trad), Full $(:Kurt)" begin
    portfolio = Portfolio(; prices = prices,
                          solvers = OrderedDict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                                  :params => Dict("verbose" => false,
                                                                                  "max_step_fraction" => 0.75)),
                                                :COSMO => Dict(:solver => COSMO.Optimizer,
                                                               :params => Dict("verbose" => false))))
    asset_statistics!(portfolio)
    returns = portfolio.returns
    sigma = portfolio.cov
    alpha_i = portfolio.alpha_i
    alpha = portfolio.alpha
    a_sim = portfolio.a_sim
    beta_i = portfolio.beta_i
    beta = portfolio.beta
    b_sim = portfolio.b_sim
    kappa = portfolio.kappa
    owa_w = portfolio.owa_w
    solvers = portfolio.solvers

    rm = :Kurt
    opt = OptimiseOpt(; rf = rf, l = l, class = :Classic, hist = 1, type = :Trad,
                      rrp_ver = :None, u_mu = :None, u_cov = :None, rm = rm,
                      obj = :Min_Risk, kelly = :None)
    w1 = optimise!(portfolio, opt)
    opt.obj = :Max_Ret
    w2 = optimise!(portfolio, opt)
    opt.obj = :Sharpe
    w3 = optimise!(portfolio, opt)
    fw1 = efficient_frontier!(portfolio, opt; points = 5)

    risks1 = [calc_risk(fw1[:weights][!, i], returns; rm = :Kurt, rf = rf, sigma = sigma,
                        alpha_i = alpha_i, alpha = alpha, a_sim = a_sim, beta_i = beta_i,
                        beta = beta, b_sim = b_sim, kappa = kappa, owa_w = owa_w,
                        solvers = solvers,) for i ∈ 2:size(Matrix(fw1[:weights]), 2)]
    rets1 = [dot(fw1[:weights][!, i], portfolio.mu)
             for i ∈ 2:size(Matrix(fw1[:weights]), 2)]
    idx = findlast(x -> x < risks1[end], risks1) + 1
    tmp = risks1[end]
    risks1[(idx + 1):end] = risks1[idx:(end - 1)]
    risks1[idx] = tmp
    tmp = rets1[end]
    rets1[(idx + 1):end] = rets1[idx:(end - 1)]
    rets1[idx] = tmp

    @test reduce(<=, risks1)
    @test reduce(<=, rets1)
    @test isapprox(fw1[:weights][!, 2], w1.weights)
    @test isapprox(fw1[:weights][!, end - 1], w2.weights)
    @test isapprox(fw1[:weights][!, end], w3.weights)
end

@testset "$(:Classic), $(:Trad), Reduced $(:Kurt)" begin
    portfolio = Portfolio(; prices = prices,
                          solvers = OrderedDict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                                  :params => Dict("verbose" => false,
                                                                                  "max_step_fraction" => 0.75)),
                                                :COSMO => Dict(:solver => COSMO.Optimizer,
                                                               :params => Dict("verbose" => false))),
                          max_num_assets_kurt = 1)
    asset_statistics!(portfolio)
    returns = portfolio.returns
    sigma = portfolio.cov
    alpha_i = portfolio.alpha_i
    alpha = portfolio.alpha
    a_sim = portfolio.a_sim
    beta_i = portfolio.beta_i
    beta = portfolio.beta
    b_sim = portfolio.b_sim
    kappa = portfolio.kappa
    owa_w = portfolio.owa_w
    solvers = portfolio.solvers

    rm = :Kurt
    opt = OptimiseOpt(; rf = rf, l = l, class = :Classic, hist = 1, type = :Trad,
                      rrp_ver = :None, u_mu = :None, u_cov = :None, rm = rm,
                      obj = :Min_Risk, kelly = :None)
    w1 = optimise!(portfolio, opt)
    opt.obj = :Max_Ret
    w2 = optimise!(portfolio, opt)
    opt.obj = :Sharpe
    w3 = optimise!(portfolio, opt)
    fw1 = efficient_frontier!(portfolio, opt; points = 5)

    risks1 = [calc_risk(fw1[:weights][!, i], returns; rm = :Kurt, rf = rf, sigma = sigma,
                        alpha_i = alpha_i, alpha = alpha, a_sim = a_sim, beta_i = beta_i,
                        beta = beta, b_sim = b_sim, kappa = kappa, owa_w = owa_w,
                        solvers = solvers,) for i ∈ 2:size(Matrix(fw1[:weights]), 2)]
    rets1 = [dot(fw1[:weights][!, i], portfolio.mu)
             for i ∈ 2:size(Matrix(fw1[:weights]), 2)]
    idx = findlast(x -> x < risks1[end], risks1) + 1
    tmp = risks1[end]
    risks1[(idx + 1):end] = risks1[idx:(end - 1)]
    risks1[idx] = tmp
    tmp = rets1[end]
    rets1[(idx + 1):end] = rets1[idx:(end - 1)]
    rets1[idx] = tmp

    @test reduce(<=, risks1)
    @test reduce(<=, rets1)
    @test isapprox(fw1[:weights][!, 2], w1.weights)
    @test isapprox(fw1[:weights][!, end - 1], w2.weights)
    @test isapprox(fw1[:weights][!, end], w3.weights)
end

@testset "$(:Classic), $(:Trad), Full $(:SKurt)" begin
    portfolio = Portfolio(; prices = prices,
                          solvers = OrderedDict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                                  :params => Dict("verbose" => false,
                                                                                  "max_step_fraction" => 0.75)),
                                                :COSMO => Dict(:solver => COSMO.Optimizer,
                                                               :params => Dict("verbose" => false))))
    asset_statistics!(portfolio)
    returns = portfolio.returns
    sigma = portfolio.cov
    alpha_i = portfolio.alpha_i
    alpha = portfolio.alpha
    a_sim = portfolio.a_sim
    beta_i = portfolio.beta_i
    beta = portfolio.beta
    b_sim = portfolio.b_sim
    kappa = portfolio.kappa
    owa_w = portfolio.owa_w
    solvers = portfolio.solvers

    rm = :SKurt
    opt = OptimiseOpt(; rf = rf, l = l, class = :Classic, hist = 1, type = :Trad,
                      rrp_ver = :None, u_mu = :None, u_cov = :None, rm = rm,
                      obj = :Min_Risk, kelly = :None)
    w1 = optimise!(portfolio, opt)
    opt.obj = :Max_Ret
    w2 = optimise!(portfolio, opt)
    opt.obj = :Sharpe
    w3 = optimise!(portfolio, opt)
    fw1 = efficient_frontier!(portfolio, opt; points = 5)

    risks1 = [calc_risk(fw1[:weights][!, i], returns; rm = :SKurt, rf = rf, sigma = sigma,
                        alpha_i = alpha_i, alpha = alpha, a_sim = a_sim, beta_i = beta_i,
                        beta = beta, b_sim = b_sim, kappa = kappa, owa_w = owa_w,
                        solvers = solvers,) for i ∈ 2:size(Matrix(fw1[:weights]), 2)]
    rets1 = [dot(fw1[:weights][!, i], portfolio.mu)
             for i ∈ 2:size(Matrix(fw1[:weights]), 2)]
    idx = findlast(x -> x < risks1[end], risks1) + 1
    tmp = risks1[end]
    risks1[(idx + 1):end] = risks1[idx:(end - 1)]
    risks1[idx] = tmp
    tmp = rets1[end]
    rets1[(idx + 1):end] = rets1[idx:(end - 1)]
    rets1[idx] = tmp

    @test reduce(<=, risks1)
    @test reduce(<=, rets1)
    @test isapprox(fw1[:weights][!, 2], w1.weights)
    @test isapprox(fw1[:weights][!, end - 1], w2.weights)
    @test isapprox(fw1[:weights][!, end], w3.weights)
end

@testset "$(:Classic), $(:Trad), Reduced $(:SKurt)" begin
    portfolio = Portfolio(; prices = prices,
                          solvers = OrderedDict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                                  :params => Dict("verbose" => false,
                                                                                  "max_step_fraction" => 0.75)),
                                                :COSMO => Dict(:solver => COSMO.Optimizer,
                                                               :params => Dict("verbose" => false))),
                          max_num_assets_kurt = 1)
    asset_statistics!(portfolio)
    returns = portfolio.returns
    sigma = portfolio.cov
    alpha_i = portfolio.alpha_i
    alpha = portfolio.alpha
    a_sim = portfolio.a_sim
    beta_i = portfolio.beta_i
    beta = portfolio.beta
    b_sim = portfolio.b_sim
    kappa = portfolio.kappa
    owa_w = portfolio.owa_w
    solvers = portfolio.solvers

    rm = :SKurt
    opt = OptimiseOpt(; rf = rf, l = l, class = :Classic, hist = 1, type = :Trad,
                      rrp_ver = :None, u_mu = :None, u_cov = :None, rm = rm,
                      obj = :Min_Risk, kelly = :None)
    w1 = optimise!(portfolio, opt)
    opt.obj = :Max_Ret
    w2 = optimise!(portfolio, opt)
    opt.obj = :Sharpe
    w3 = optimise!(portfolio, opt)
    fw1 = efficient_frontier!(portfolio, opt; points = 5)

    risks1 = [calc_risk(fw1[:weights][!, i], returns; rm = :SKurt, rf = rf, sigma = sigma,
                        alpha_i = alpha_i, alpha = alpha, a_sim = a_sim, beta_i = beta_i,
                        beta = beta, b_sim = b_sim, kappa = kappa, owa_w = owa_w,
                        solvers = solvers,) for i ∈ 2:size(Matrix(fw1[:weights]), 2)]
    rets1 = [dot(fw1[:weights][!, i], portfolio.mu)
             for i ∈ 2:size(Matrix(fw1[:weights]), 2)]
    idx = findlast(x -> x < risks1[end], risks1) + 1
    tmp = risks1[end]
    risks1[(idx + 1):end] = risks1[idx:(end - 1)]
    risks1[idx] = tmp
    tmp = rets1[end]
    rets1[(idx + 1):end] = rets1[idx:(end - 1)]
    rets1[idx] = tmp

    @test reduce(<=, risks1)
    @test reduce(<=, rets1)
    @test isapprox(fw1[:weights][!, 2], w1.weights)
    @test isapprox(fw1[:weights][!, end - 1], w2.weights)
    @test isapprox(fw1[:weights][!, end], w3.weights)
end

@testset "Efficient frontier, all kelly returns" begin
    portfolio = Portfolio(; prices = prices,
                          solvers = OrderedDict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                                  :params => Dict("verbose" => false,
                                                                                  "max_step_fraction" => 0.75)),
                                                :COSMO => Dict(:solver => COSMO.Optimizer,
                                                               :params => Dict("verbose" => false))))
    asset_statistics!(portfolio)

    rm = :SKurt
    opt = OptimiseOpt(; rf = rf, l = l, class = :Classic, hist = 1, type = :Trad,
                      rrp_ver = :None, u_mu = :None, u_cov = :None, rm = rm,
                      obj = :Min_Risk, kelly = :None)
    w1 = optimise!(portfolio, opt)
    sr1 = sharpe_ratio(portfolio; rf = rf, rm = rm)
    opt.obj = :Max_Ret
    w2 = optimise!(portfolio, opt)
    sr2 = sharpe_ratio(portfolio; rf = rf, rm = rm)
    opt.obj = :Sharpe
    w3 = optimise!(portfolio, opt)
    sr3 = sharpe_ratio(portfolio; rf = rf, rm = rm)
    fw1 = efficient_frontier!(portfolio, opt; points = 25)

    risks1 = [calc_risk(fw1[:weights][!, i], portfolio.returns; rm = rm, rf = rf,
                        solvers = portfolio.solvers,)
              for i ∈ 2:size(Matrix(fw1[:weights]), 2)]
    rets1 = [dot(fw1[:weights][!, i], portfolio.mu)
             for i ∈ 2:size(Matrix(fw1[:weights]), 2)]
    sharpes1 = (rets1 .- rf) ./ risks1
    sharpes1t = [sharpe_ratio(fw1[:weights][!, i], portfolio.mu, portfolio.returns; rm = rm,
                              rf = rf, kelly = false)
                 for i ∈ 2:size(Matrix(fw1[:weights]), 2)]

    idx = findlast(x -> x < risks1[end], risks1) + 1
    tmp = risks1[end]
    risks1[(idx + 1):end] = risks1[idx:(end - 1)]
    risks1[idx] = tmp
    tmp = rets1[end]
    rets1[(idx + 1):end] = rets1[idx:(end - 1)]
    rets1[idx] = tmp

    rm = :SKurt
    opt = OptimiseOpt(; rf = rf, l = l, class = :Classic, hist = 1, type = :Trad,
                      rrp_ver = :None, u_mu = :None, u_cov = :None, rm = rm,
                      obj = :Min_Risk, kelly = :Approx)
    w4 = optimise!(portfolio, opt)
    sr4 = sharpe_ratio(portfolio; rf = rf, rm = rm, kelly = true)
    opt.obj = :Max_Ret
    w5 = optimise!(portfolio, opt)
    sr5 = sharpe_ratio(portfolio; rf = rf, rm = rm, kelly = true)
    opt.obj = :Sharpe
    w6 = optimise!(portfolio, opt)
    sr6 = sharpe_ratio(portfolio; rf = rf, rm = rm, kelly = true)
    fw2 = efficient_frontier!(portfolio, opt; points = 25)
    risks2 = [calc_risk(fw2[:weights][!, i], portfolio.returns; rm = rm, rf = rf,
                        solvers = portfolio.solvers,)
              for i ∈ 2:size(Matrix(fw2[:weights]), 2)]
    rets2 = [1 / size(portfolio.returns, 1) *
             sum(log.(1 .+ portfolio.returns * fw2[:weights][!, i]))
             for i ∈ 2:size(Matrix(fw2[:weights]), 2)]
    idx = findlast(x -> x < risks2[end], risks2) + 1
    tmp = risks2[end]
    risks2[(idx + 1):end] = risks2[idx:(end - 1)]
    risks2[idx] = tmp
    tmp = rets2[end]
    rets2[(idx + 1):end] = rets2[idx:(end - 1)]
    rets2[idx] = tmp

    rm = :CDaR
    opt = OptimiseOpt(; rf = rf, l = l, class = :Classic, hist = 1, type = :Trad,
                      rrp_ver = :None, u_mu = :None, u_cov = :None, rm = rm,
                      obj = :Min_Risk, kelly = :Exact)
    w7 = optimise!(portfolio, opt)
    sr7 = sharpe_ratio(portfolio; rf = rf, rm = rm, kelly = true)
    opt.obj = :Max_Ret
    w8 = optimise!(portfolio, opt)
    sr8 = sharpe_ratio(portfolio; rf = rf, rm = rm, kelly = true)
    opt.obj = :Sharpe
    w9 = optimise!(portfolio, opt)
    sr9 = sharpe_ratio(portfolio; rf = rf, rm = rm, kelly = true)
    fw3 = efficient_frontier!(portfolio, opt; points = 25)
    risks3 = [calc_risk(fw3[:weights][!, i], portfolio.returns; rm = rm, rf = rf,
                        solvers = portfolio.solvers,)
              for i ∈ 2:size(Matrix(fw3[:weights]), 2)]
    rets3 = [1 / size(portfolio.returns, 1) *
             sum(log.(1 .+ portfolio.returns * fw3[:weights][!, i]))
             for i ∈ 2:size(Matrix(fw3[:weights]), 2)]
    sharpes3 = (rets3 .- rf) ./ risks3
    sharpes3t = [sharpe_ratio(fw3[:weights][!, i], portfolio.mu, portfolio.returns; rm = rm,
                              rf = rf, kelly = true)
                 for i ∈ 2:size(Matrix(fw1[:weights]), 2)]

    idx = findlast(x -> x < risks3[end], risks3) + 1
    tmp = risks3[end]
    risks3[(idx + 1):end] = risks3[idx:(end - 1)]
    risks3[idx] = tmp
    tmp = rets3[end]
    rets3[(idx + 1):end] = rets3[idx:(end - 1)]
    rets3[idx] = tmp
    @test isapprox(sharpes1, sharpes1t)
    @test isapprox(sharpes1[1], sr1)
    @test isapprox(sharpes1[end - 1], sr2)
    @test isapprox(sharpes1[end], sr3)
    @test reduce(<=, risks1)
    @test reduce(<=, rets1)
    @test isapprox(fw1[:weights][!, 2], w1.weights)
    @test isapprox(fw1[:weights][!, end - 1], w2.weights)
    @test isapprox(fw1[:weights][!, end], w3.weights)
    @test reduce(<=, risks2)
    @test reduce(<=, rets2)
    @test isapprox(fw2[:weights][!, 2], w4.weights)
    @test isapprox(fw2[:weights][!, end - 1], w5.weights)
    @test isapprox(fw2[:weights][!, end], w6.weights)
    @test isapprox(sharpes3, sharpes3t)
    @test isapprox(sharpes3[1], sr7)
    @test isapprox(sharpes3[end - 1], sr8)
    @test isapprox(sharpes3[end], sr9)
    @test reduce(<=, risks3)
    @test reduce(<=, rets3)
    @test isapprox(fw3[:weights][!, 2], w7.weights)
    @test isapprox(fw3[:weights][!, end - 1], w8.weights)
    @test isapprox(fw3[:weights][!, end], w9.weights)
end
