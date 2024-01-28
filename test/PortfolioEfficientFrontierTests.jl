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
    @test reduce(<=, risks2[3:end])
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

@testset "Efficient frontier near optimal ordering" begin
    portfolio = Portfolio(; prices = prices,
                          solvers = OrderedDict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                                  :params => Dict("verbose" => false,
                                                                                  "max_step_fraction" => 0.7)),
                                                :COSMO => Dict(:solver => COSMO.Optimizer,
                                                               :params => Dict("verbose" => false))))
    asset_statistics!(portfolio)
    frontier = efficient_frontier!(portfolio,
                                   OptimiseOpt(; type = :Trad, rm = :CVaR, near_opt = true,
                                               n = 5); points = 10)
    frontiert = reshape([0.051486360012397016, 0.05236983288917021, 0.05132527587399171,
                         0.04986422716287365, 0.0570083550281748, 0.045796008131193965,
                         0.048068240664882085, 0.054942698822322146, 0.048930962313161616,
                         0.0480730247923202, 0.05347248935088241, 0.04327427528324469,
                         0.03961302483795674, 0.049721873111004, 0.04044184014873579,
                         0.05357451247151647, 0.053409588164720044, 0.05403200381208819,
                         0.0513814825682843, 0.05321392456108, 0.026821268798996076,
                         0.030277546224179892, 0.024110298236767504, 0.025200174485100337,
                         0.31781909387081353, 0.009521499550705926, 0.03860941857585781,
                         0.06023468191033268, 0.01779430032575157, 0.01575469608851992,
                         0.0722861132897799, 0.00819289606352098, 0.0056255371943901775,
                         0.01952983067904024, 0.006286692485911797, 0.08181427464482686,
                         0.0708798405394817, 0.09271216923235816, 0.030463759170697696,
                         0.046065908632967174, 0.02257872852038554, 0.024536111583012415,
                         0.029400229801466445, 0.025726947684143738, 0.48305625882020437,
                         0.010183813115227603, 0.14591193416608383, 0.01743842685190719,
                         0.02485440605927779, 0.01807893337221225, 0.016667535309586895,
                         0.010600203390568098, 0.007523273071820424, 0.01348604256468001,
                         0.0078053379956584895, 0.04209240187898889, 0.03264573581864106,
                         0.018289629452301608, 0.02697553593745076, 0.022148514606382497,
                         0.014013249672769559, 0.014958382016165537, 0.01841752710637425,
                         0.016524037217702566, 0.5149873461783283, 0.006405976988549117,
                         0.2581687568487945, 0.010418560222877753, 0.015630531492763264,
                         0.011218062480867912, 0.010053571291418718, 0.006786733521275428,
                         0.004806947357755105, 0.008309759095310669, 0.004963807223459894,
                         0.024223287105782892, 0.01938348921393103, 0.010940957426087802,
                         0.016376042899818352, 0.01341297463996747, 0.013705614739666137,
                         0.01457432851387161, 0.017960327728509074, 0.016294074841513725,
                         0.4275436268346442, 0.006325912155529782, 0.35118351295240374,
                         0.010148253965269931, 0.015266596621300794, 0.011008571260714623,
                         0.009815309133938126, 0.006702068862146002, 0.004764770631800752,
                         0.008160951964809232, 0.004929037960149233, 0.02320956714853085,
                         0.018791844825991565, 0.010656903238207656, 0.01592250360425309,
                         0.013036223016749877, 0.013554075423113217, 0.014396084352292829,
                         0.017711980783940522, 0.0161759895553917, 0.3654973533468737,
                         0.006300460121646455, 0.41588289621708935, 0.010035193992124537,
                         0.01511337361258717, 0.010916281514823229, 0.009709630367625439,
                         0.006669503537459898, 0.004752988819649873, 0.008098702507701094,
                         0.00492184924431572, 0.02265453206113468, 0.018483477580504094,
                         0.010537420565674754, 0.01570804931151351, 0.012880157084538213,
                         0.013393451640655354, 0.014212370934066495, 0.017453791369429785,
                         0.01599064164771719, 0.3342206137150228, 0.006244873165037023,
                         0.44992011891561634, 0.009906436778978085, 0.014923301817736568,
                         0.010800667517320332, 0.009593622103020144, 0.006615331583175004,
                         0.004714295554363717, 0.008014788859958061, 0.004886151718028586,
                         0.02227727835404245, 0.018217692262057873, 0.010404188109198109,
                         0.015485194434977053, 0.0127251895195991, 0.013219786577852288,
                         0.014040310612926713, 0.017268474735408938, 0.015815957803047603,
                         0.3112376210069283, 0.006183192397466366, 0.47551503526062255,
                         0.009785091923076392, 0.014745627777312004, 0.010682448633460056,
                         0.009481947125956531, 0.006549133069360527, 0.00467367551739762,
                         0.00792412116318791, 0.004842492994898218, 0.02194439632282262,
                         0.017974475149029796, 0.010273919741916538, 0.015283057325334635,
                         0.012559234861994378, 0.013048478140206, 0.013854832126576695,
                         0.01703853587914295, 0.015617020614165346, 0.29558456027489594,
                         0.006111767158909742, 0.49393025152843606, 0.009659092048917668,
                         0.014557783735968648, 0.01054868135128449, 0.009359445370684135,
                         0.006473959448772166, 0.004622300082268987, 0.007830217556712617,
                         0.004788840759744425, 0.02163031654357611, 0.01772571058792744,
                         0.010142630347936782, 0.015085137290175736, 0.012390439153697941,
                         0.012879927764277755, 0.013673903235410852, 0.01681093862961253,
                         0.015413410656460427, 0.2843568453030244, 0.006036800615322954,
                         0.5079302504339147, 0.009533387434620046, 0.014368969457392269,
                         0.010416381319155859, 0.009242115746001067, 0.0063964700498197365,
                         0.00456858581674156, 0.0077323479994042595, 0.004733448544173203,
                         0.02129758789417074, 0.017485726294248365, 0.010012155696351537,
                         0.014883553766076174, 0.012227193343821576], 20, :)
    @test isapprox(Matrix(frontier[:weights][!, 2:end]), frontiert)
end
