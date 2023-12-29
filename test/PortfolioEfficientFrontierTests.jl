using COSMO,
    CSV,
    Clarabel,
    HiGHS,
    LinearAlgebra,
    OrderedCollections,
    PortfolioOptimiser,
    Statistics,
    Test,
    TimeSeries,
    Logging

Logging.disable_logging(Logging.Warn)

prices = TimeArray(CSV.File("./assets/stock_prices.csv"); timestamp = :date)

rf = 1.0329^(1 / 252) - 1
l = 2.0

@testset "$(:Classic), $(:Trad), $(:SD)" begin
    portfolio = Portfolio(
        prices = prices,
        solvers = OrderedDict(
            :Clarabel => Dict(
                :solver => Clarabel.Optimizer,
                :params => Dict("verbose" => false, "max_step_fraction" => 0.75),
            ),
            :COSMO =>
                Dict(:solver => COSMO.Optimizer, :params => Dict("verbose" => false)),
        ),
    )
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

    w1 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = :Classic,
        hist = 1,
        type = :Trad,
        rrp_ver = :None,
        u_mu = :None,
        u_cov = :None,
        rm = :SD,
        obj = :Min_Risk,
        kelly = :None,
    )
    w2 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = :Classic,
        hist = 1,
        type = :Trad,
        rrp_ver = :None,
        u_mu = :None,
        u_cov = :None,
        rm = :SD,
        obj = :Max_Ret,
        kelly = :None,
    )
    w3 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = :Classic,
        hist = 1,
        type = :Trad,
        rrp_ver = :None,
        u_mu = :None,
        u_cov = :None,
        rm = :SD,
        obj = :Sharpe,
        kelly = :None,
    )
    fw1 = efficient_frontier!(
        portfolio;
        class = :Classic,
        hist = 1,
        kelly = :None,
        rf = rf,
        rm = :SD,
        points = 5,
    )

    risks1 = [
        calc_risk(
            fw1[:weights][!, i],
            returns;
            rm = :SD,
            rf = rf,
            sigma = sigma,
            alpha_i = alpha_i,
            alpha = alpha,
            a_sim = a_sim,
            beta_i = beta_i,
            beta = beta,
            b_sim = b_sim,
            kappa = kappa,
            owa_w = owa_w,
            solvers = solvers,
        ) for i in 2:size(Matrix(fw1[:weights]), 2)
    ]
    rets1 =
        [dot(fw1[:weights][!, i], portfolio.mu) for i in 2:size(Matrix(fw1[:weights]), 2)]
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
    portfolio = Portfolio(
        prices = prices,
        solvers = OrderedDict(
            :Clarabel => Dict(
                :solver => Clarabel.Optimizer,
                :params => Dict("verbose" => false, "max_step_fraction" => 0.75),
            ),
            :COSMO =>
                Dict(:solver => COSMO.Optimizer, :params => Dict("verbose" => false)),
        ),
    )
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

    w1 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = :Classic,
        hist = 1,
        type = :Trad,
        rrp_ver = :None,
        u_mu = :None,
        u_cov = :None,
        rm = :MAD,
        obj = :Min_Risk,
        kelly = :None,
    )
    w2 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = :Classic,
        hist = 1,
        type = :Trad,
        rrp_ver = :None,
        u_mu = :None,
        u_cov = :None,
        rm = :MAD,
        obj = :Max_Ret,
        kelly = :None,
    )
    w3 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = :Classic,
        hist = 1,
        type = :Trad,
        rrp_ver = :None,
        u_mu = :None,
        u_cov = :None,
        rm = :MAD,
        obj = :Sharpe,
        kelly = :None,
    )
    fw1 = efficient_frontier!(
        portfolio;
        class = :Classic,
        hist = 1,
        kelly = :None,
        rf = rf,
        rm = :MAD,
        points = 5,
    )

    risks1 = [
        calc_risk(
            fw1[:weights][!, i],
            returns;
            rm = :MAD,
            rf = rf,
            sigma = sigma,
            alpha_i = alpha_i,
            alpha = alpha,
            a_sim = a_sim,
            beta_i = beta_i,
            beta = beta,
            b_sim = b_sim,
            kappa = kappa,
            owa_w = owa_w,
            solvers = solvers,
        ) for i in 2:size(Matrix(fw1[:weights]), 2)
    ]
    rets1 =
        [dot(fw1[:weights][!, i], portfolio.mu) for i in 2:size(Matrix(fw1[:weights]), 2)]
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
    portfolio = Portfolio(
        prices = prices,
        solvers = OrderedDict(
            :Clarabel => Dict(
                :solver => Clarabel.Optimizer,
                :params => Dict("verbose" => false, "max_step_fraction" => 0.75),
            ),
            :COSMO =>
                Dict(:solver => COSMO.Optimizer, :params => Dict("verbose" => false)),
        ),
    )
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

    w1 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = :Classic,
        hist = 1,
        type = :Trad,
        rrp_ver = :None,
        u_mu = :None,
        u_cov = :None,
        rm = :SSD,
        obj = :Min_Risk,
        kelly = :None,
    )
    w2 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = :Classic,
        hist = 1,
        type = :Trad,
        rrp_ver = :None,
        u_mu = :None,
        u_cov = :None,
        rm = :SSD,
        obj = :Max_Ret,
        kelly = :None,
    )
    w3 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = :Classic,
        hist = 1,
        type = :Trad,
        rrp_ver = :None,
        u_mu = :None,
        u_cov = :None,
        rm = :SSD,
        obj = :Sharpe,
        kelly = :None,
    )
    fw1 = efficient_frontier!(
        portfolio;
        class = :Classic,
        hist = 1,
        kelly = :None,
        rf = rf,
        rm = :SSD,
        points = 5,
    )

    risks1 = [
        calc_risk(
            fw1[:weights][!, i],
            returns;
            rm = :SSD,
            rf = rf,
            sigma = sigma,
            alpha_i = alpha_i,
            alpha = alpha,
            a_sim = a_sim,
            beta_i = beta_i,
            beta = beta,
            b_sim = b_sim,
            kappa = kappa,
            owa_w = owa_w,
            solvers = solvers,
        ) for i in 2:size(Matrix(fw1[:weights]), 2)
    ]
    rets1 =
        [dot(fw1[:weights][!, i], portfolio.mu) for i in 2:size(Matrix(fw1[:weights]), 2)]
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
    portfolio = Portfolio(
        prices = prices,
        solvers = OrderedDict(
            :Clarabel => Dict(
                :solver => Clarabel.Optimizer,
                :params => Dict("verbose" => false, "max_step_fraction" => 0.75),
            ),
            :COSMO =>
                Dict(:solver => COSMO.Optimizer, :params => Dict("verbose" => false)),
        ),
    )
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

    w1 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = :Classic,
        hist = 1,
        type = :Trad,
        rrp_ver = :None,
        u_mu = :None,
        u_cov = :None,
        rm = :FLPM,
        obj = :Min_Risk,
        kelly = :None,
    )
    w2 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = :Classic,
        hist = 1,
        type = :Trad,
        rrp_ver = :None,
        u_mu = :None,
        u_cov = :None,
        rm = :FLPM,
        obj = :Max_Ret,
        kelly = :None,
    )
    w3 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = :Classic,
        hist = 1,
        type = :Trad,
        rrp_ver = :None,
        u_mu = :None,
        u_cov = :None,
        rm = :FLPM,
        obj = :Sharpe,
        kelly = :None,
    )
    fw1 = efficient_frontier!(
        portfolio;
        class = :Classic,
        hist = 1,
        kelly = :None,
        rf = rf,
        rm = :FLPM,
        points = 5,
    )

    risks1 = [
        calc_risk(
            fw1[:weights][!, i],
            returns;
            rm = :FLPM,
            rf = rf,
            sigma = sigma,
            alpha_i = alpha_i,
            alpha = alpha,
            a_sim = a_sim,
            beta_i = beta_i,
            beta = beta,
            b_sim = b_sim,
            kappa = kappa,
            owa_w = owa_w,
            solvers = solvers,
        ) for i in 2:size(Matrix(fw1[:weights]), 2)
    ]
    rets1 =
        [dot(fw1[:weights][!, i], portfolio.mu) for i in 2:size(Matrix(fw1[:weights]), 2)]
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
    portfolio = Portfolio(
        prices = prices,
        solvers = OrderedDict(
            :Clarabel => Dict(
                :solver => Clarabel.Optimizer,
                :params => Dict("verbose" => false, "max_step_fraction" => 0.75),
            ),
            :COSMO =>
                Dict(:solver => COSMO.Optimizer, :params => Dict("verbose" => false)),
        ),
    )
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

    w1 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = :Classic,
        hist = 1,
        type = :Trad,
        rrp_ver = :None,
        u_mu = :None,
        u_cov = :None,
        rm = :SLPM,
        obj = :Min_Risk,
        kelly = :None,
    )
    w2 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = :Classic,
        hist = 1,
        type = :Trad,
        rrp_ver = :None,
        u_mu = :None,
        u_cov = :None,
        rm = :SLPM,
        obj = :Max_Ret,
        kelly = :None,
    )
    w3 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = :Classic,
        hist = 1,
        type = :Trad,
        rrp_ver = :None,
        u_mu = :None,
        u_cov = :None,
        rm = :SLPM,
        obj = :Sharpe,
        kelly = :None,
    )
    fw1 = efficient_frontier!(
        portfolio;
        class = :Classic,
        hist = 1,
        kelly = :None,
        rf = rf,
        rm = :SLPM,
        points = 5,
    )

    risks1 = [
        calc_risk(
            fw1[:weights][!, i],
            returns;
            rm = :SLPM,
            rf = rf,
            sigma = sigma,
            alpha_i = alpha_i,
            alpha = alpha,
            a_sim = a_sim,
            beta_i = beta_i,
            beta = beta,
            b_sim = b_sim,
            kappa = kappa,
            owa_w = owa_w,
            solvers = solvers,
        ) for i in 2:size(Matrix(fw1[:weights]), 2)
    ]
    rets1 =
        [dot(fw1[:weights][!, i], portfolio.mu) for i in 2:size(Matrix(fw1[:weights]), 2)]
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
    portfolio = Portfolio(
        prices = prices,
        solvers = OrderedDict(
            :Clarabel => Dict(
                :solver => Clarabel.Optimizer,
                :params => Dict("verbose" => false, "max_step_fraction" => 0.75),
            ),
            :COSMO =>
                Dict(:solver => COSMO.Optimizer, :params => Dict("verbose" => false)),
        ),
    )
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

    w1 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = :Classic,
        hist = 1,
        type = :Trad,
        rrp_ver = :None,
        u_mu = :None,
        u_cov = :None,
        rm = :WR,
        obj = :Min_Risk,
        kelly = :None,
    )
    w2 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = :Classic,
        hist = 1,
        type = :Trad,
        rrp_ver = :None,
        u_mu = :None,
        u_cov = :None,
        rm = :WR,
        obj = :Max_Ret,
        kelly = :None,
    )
    w3 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = :Classic,
        hist = 1,
        type = :Trad,
        rrp_ver = :None,
        u_mu = :None,
        u_cov = :None,
        rm = :WR,
        obj = :Sharpe,
        kelly = :None,
    )
    fw1 = efficient_frontier!(
        portfolio;
        class = :Classic,
        hist = 1,
        kelly = :None,
        rf = rf,
        rm = :WR,
        points = 5,
    )

    risks1 = [
        calc_risk(
            fw1[:weights][!, i],
            returns;
            rm = :WR,
            rf = rf,
            sigma = sigma,
            alpha_i = alpha_i,
            alpha = alpha,
            a_sim = a_sim,
            beta_i = beta_i,
            beta = beta,
            b_sim = b_sim,
            kappa = kappa,
            owa_w = owa_w,
            solvers = solvers,
        ) for i in 2:size(Matrix(fw1[:weights]), 2)
    ]
    rets1 =
        [dot(fw1[:weights][!, i], portfolio.mu) for i in 2:size(Matrix(fw1[:weights]), 2)]
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
    portfolio = Portfolio(
        prices = prices,
        solvers = OrderedDict(
            :Clarabel => Dict(
                :solver => Clarabel.Optimizer,
                :params => Dict("verbose" => false, "max_step_fraction" => 0.75),
            ),
            :COSMO =>
                Dict(:solver => COSMO.Optimizer, :params => Dict("verbose" => false)),
        ),
    )
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

    w1 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = :Classic,
        hist = 1,
        type = :Trad,
        rrp_ver = :None,
        u_mu = :None,
        u_cov = :None,
        rm = :CVaR,
        obj = :Min_Risk,
        kelly = :None,
    )
    w2 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = :Classic,
        hist = 1,
        type = :Trad,
        rrp_ver = :None,
        u_mu = :None,
        u_cov = :None,
        rm = :CVaR,
        obj = :Max_Ret,
        kelly = :None,
    )
    w3 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = :Classic,
        hist = 1,
        type = :Trad,
        rrp_ver = :None,
        u_mu = :None,
        u_cov = :None,
        rm = :CVaR,
        obj = :Sharpe,
        kelly = :None,
    )
    fw1 = efficient_frontier!(
        portfolio;
        class = :Classic,
        hist = 1,
        kelly = :None,
        rf = rf,
        rm = :CVaR,
        points = 5,
    )

    risks1 = [
        calc_risk(
            fw1[:weights][!, i],
            returns;
            rm = :CVaR,
            rf = rf,
            sigma = sigma,
            alpha_i = alpha_i,
            alpha = alpha,
            a_sim = a_sim,
            beta_i = beta_i,
            beta = beta,
            b_sim = b_sim,
            kappa = kappa,
            owa_w = owa_w,
            solvers = solvers,
        ) for i in 2:size(Matrix(fw1[:weights]), 2)
    ]
    rets1 =
        [dot(fw1[:weights][!, i], portfolio.mu) for i in 2:size(Matrix(fw1[:weights]), 2)]
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
    portfolio = Portfolio(
        prices = prices,
        solvers = OrderedDict(
            :Clarabel => Dict(
                :solver => Clarabel.Optimizer,
                :params => Dict("verbose" => false, "max_step_fraction" => 0.75),
            ),
            :COSMO =>
                Dict(:solver => COSMO.Optimizer, :params => Dict("verbose" => false)),
        ),
    )
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

    w1 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = :Classic,
        hist = 1,
        type = :Trad,
        rrp_ver = :None,
        u_mu = :None,
        u_cov = :None,
        rm = :EVaR,
        obj = :Min_Risk,
        kelly = :None,
    )
    w2 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = :Classic,
        hist = 1,
        type = :Trad,
        rrp_ver = :None,
        u_mu = :None,
        u_cov = :None,
        rm = :EVaR,
        obj = :Max_Ret,
        kelly = :None,
    )
    w3 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = :Classic,
        hist = 1,
        type = :Trad,
        rrp_ver = :None,
        u_mu = :None,
        u_cov = :None,
        rm = :EVaR,
        obj = :Sharpe,
        kelly = :None,
    )
    fw1 = efficient_frontier!(
        portfolio;
        class = :Classic,
        hist = 1,
        kelly = :None,
        rf = rf,
        rm = :EVaR,
        points = 5,
    )

    risks1 = [
        calc_risk(
            fw1[:weights][!, i],
            returns;
            rm = :EVaR,
            rf = rf,
            sigma = sigma,
            alpha_i = alpha_i,
            alpha = alpha,
            a_sim = a_sim,
            beta_i = beta_i,
            beta = beta,
            b_sim = b_sim,
            kappa = kappa,
            owa_w = owa_w,
            solvers = solvers,
        ) for i in 2:size(Matrix(fw1[:weights]), 2)
    ]
    rets1 =
        [dot(fw1[:weights][!, i], portfolio.mu) for i in 2:size(Matrix(fw1[:weights]), 2)]
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
    portfolio = Portfolio(
        prices = prices,
        solvers = OrderedDict(
            :Clarabel => Dict(
                :solver => Clarabel.Optimizer,
                :params => Dict("verbose" => false, "max_step_fraction" => 0.75),
            ),
            :COSMO =>
                Dict(:solver => COSMO.Optimizer, :params => Dict("verbose" => false)),
        ),
    )
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

    w1 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = :Classic,
        hist = 1,
        type = :Trad,
        rrp_ver = :None,
        u_mu = :None,
        u_cov = :None,
        rm = :RVaR,
        obj = :Min_Risk,
        kelly = :None,
    )
    w2 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = :Classic,
        hist = 1,
        type = :Trad,
        rrp_ver = :None,
        u_mu = :None,
        u_cov = :None,
        rm = :RVaR,
        obj = :Max_Ret,
        kelly = :None,
    )
    w3 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = :Classic,
        hist = 1,
        type = :Trad,
        rrp_ver = :None,
        u_mu = :None,
        u_cov = :None,
        rm = :RVaR,
        obj = :Sharpe,
        kelly = :None,
    )
    fw1 = efficient_frontier!(
        portfolio;
        class = :Classic,
        hist = 1,
        kelly = :None,
        rf = rf,
        rm = :RVaR,
        points = 5,
    )

    risks1 = [
        calc_risk(
            fw1[:weights][!, i],
            returns;
            rm = :RVaR,
            rf = rf,
            sigma = sigma,
            alpha_i = alpha_i,
            alpha = alpha,
            a_sim = a_sim,
            beta_i = beta_i,
            beta = beta,
            b_sim = b_sim,
            kappa = kappa,
            owa_w = owa_w,
            solvers = solvers,
        ) for i in 2:size(Matrix(fw1[:weights]), 2)
    ]
    rets1 =
        [dot(fw1[:weights][!, i], portfolio.mu) for i in 2:size(Matrix(fw1[:weights]), 2)]
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
    portfolio = Portfolio(
        prices = prices,
        solvers = OrderedDict(
            :Clarabel => Dict(
                :solver => Clarabel.Optimizer,
                :params => Dict("verbose" => false, "max_step_fraction" => 0.75),
            ),
            :COSMO =>
                Dict(:solver => COSMO.Optimizer, :params => Dict("verbose" => false)),
        ),
    )
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

    w1 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = :Classic,
        hist = 1,
        type = :Trad,
        rrp_ver = :None,
        u_mu = :None,
        u_cov = :None,
        rm = :MDD,
        obj = :Min_Risk,
        kelly = :None,
    )
    w2 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = :Classic,
        hist = 1,
        type = :Trad,
        rrp_ver = :None,
        u_mu = :None,
        u_cov = :None,
        rm = :MDD,
        obj = :Max_Ret,
        kelly = :None,
    )
    w3 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = :Classic,
        hist = 1,
        type = :Trad,
        rrp_ver = :None,
        u_mu = :None,
        u_cov = :None,
        rm = :MDD,
        obj = :Sharpe,
        kelly = :None,
    )
    fw1 = efficient_frontier!(
        portfolio;
        class = :Classic,
        hist = 1,
        kelly = :None,
        rf = rf,
        rm = :MDD,
        points = 5,
    )

    risks1 = [
        calc_risk(
            fw1[:weights][!, i],
            returns;
            rm = :MDD,
            rf = rf,
            sigma = sigma,
            alpha_i = alpha_i,
            alpha = alpha,
            a_sim = a_sim,
            beta_i = beta_i,
            beta = beta,
            b_sim = b_sim,
            kappa = kappa,
            owa_w = owa_w,
            solvers = solvers,
        ) for i in 2:size(Matrix(fw1[:weights]), 2)
    ]
    rets1 =
        [dot(fw1[:weights][!, i], portfolio.mu) for i in 2:size(Matrix(fw1[:weights]), 2)]
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
    portfolio = Portfolio(
        prices = prices,
        solvers = OrderedDict(
            :Clarabel => Dict(
                :solver => Clarabel.Optimizer,
                :params => Dict("verbose" => false, "max_step_fraction" => 0.75),
            ),
            :COSMO =>
                Dict(:solver => COSMO.Optimizer, :params => Dict("verbose" => false)),
        ),
    )
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

    w1 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = :Classic,
        hist = 1,
        type = :Trad,
        rrp_ver = :None,
        u_mu = :None,
        u_cov = :None,
        rm = :ADD,
        obj = :Min_Risk,
        kelly = :None,
    )
    w2 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = :Classic,
        hist = 1,
        type = :Trad,
        rrp_ver = :None,
        u_mu = :None,
        u_cov = :None,
        rm = :ADD,
        obj = :Max_Ret,
        kelly = :None,
    )
    w3 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = :Classic,
        hist = 1,
        type = :Trad,
        rrp_ver = :None,
        u_mu = :None,
        u_cov = :None,
        rm = :ADD,
        obj = :Sharpe,
        kelly = :None,
    )
    fw1 = efficient_frontier!(
        portfolio;
        class = :Classic,
        hist = 1,
        kelly = :None,
        rf = rf,
        rm = :ADD,
        points = 5,
    )

    risks1 = [
        calc_risk(
            fw1[:weights][!, i],
            returns;
            rm = :ADD,
            rf = rf,
            sigma = sigma,
            alpha_i = alpha_i,
            alpha = alpha,
            a_sim = a_sim,
            beta_i = beta_i,
            beta = beta,
            b_sim = b_sim,
            kappa = kappa,
            owa_w = owa_w,
            solvers = solvers,
        ) for i in 2:size(Matrix(fw1[:weights]), 2)
    ]
    rets1 =
        [dot(fw1[:weights][!, i], portfolio.mu) for i in 2:size(Matrix(fw1[:weights]), 2)]
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
    portfolio = Portfolio(
        prices = prices,
        solvers = OrderedDict(
            :Clarabel => Dict(
                :solver => Clarabel.Optimizer,
                :params => Dict("verbose" => false, "max_step_fraction" => 0.75),
            ),
            :COSMO =>
                Dict(:solver => COSMO.Optimizer, :params => Dict("verbose" => false)),
        ),
    )
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

    w1 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = :Classic,
        hist = 1,
        type = :Trad,
        rrp_ver = :None,
        u_mu = :None,
        u_cov = :None,
        rm = :CDaR,
        obj = :Min_Risk,
        kelly = :None,
    )
    w2 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = :Classic,
        hist = 1,
        type = :Trad,
        rrp_ver = :None,
        u_mu = :None,
        u_cov = :None,
        rm = :CDaR,
        obj = :Max_Ret,
        kelly = :None,
    )
    w3 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = :Classic,
        hist = 1,
        type = :Trad,
        rrp_ver = :None,
        u_mu = :None,
        u_cov = :None,
        rm = :CDaR,
        obj = :Sharpe,
        kelly = :None,
    )
    fw1 = efficient_frontier!(
        portfolio;
        class = :Classic,
        hist = 1,
        kelly = :None,
        rf = rf,
        rm = :CDaR,
        points = 5,
    )

    risks1 = [
        calc_risk(
            fw1[:weights][!, i],
            returns;
            rm = :CDaR,
            rf = rf,
            sigma = sigma,
            alpha_i = alpha_i,
            alpha = alpha,
            a_sim = a_sim,
            beta_i = beta_i,
            beta = beta,
            b_sim = b_sim,
            kappa = kappa,
            owa_w = owa_w,
            solvers = solvers,
        ) for i in 2:size(Matrix(fw1[:weights]), 2)
    ]
    rets1 =
        [dot(fw1[:weights][!, i], portfolio.mu) for i in 2:size(Matrix(fw1[:weights]), 2)]
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
    portfolio = Portfolio(
        prices = prices,
        solvers = OrderedDict(
            :Clarabel => Dict(
                :solver => Clarabel.Optimizer,
                :params => Dict("verbose" => false, "max_step_fraction" => 0.75),
            ),
            :COSMO =>
                Dict(:solver => COSMO.Optimizer, :params => Dict("verbose" => false)),
        ),
    )
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

    w1 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = :Classic,
        hist = 1,
        type = :Trad,
        rrp_ver = :None,
        u_mu = :None,
        u_cov = :None,
        rm = :UCI,
        obj = :Min_Risk,
        kelly = :None,
    )
    w2 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = :Classic,
        hist = 1,
        type = :Trad,
        rrp_ver = :None,
        u_mu = :None,
        u_cov = :None,
        rm = :UCI,
        obj = :Max_Ret,
        kelly = :None,
    )
    w3 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = :Classic,
        hist = 1,
        type = :Trad,
        rrp_ver = :None,
        u_mu = :None,
        u_cov = :None,
        rm = :UCI,
        obj = :Sharpe,
        kelly = :None,
    )
    fw1 = efficient_frontier!(
        portfolio;
        class = :Classic,
        hist = 1,
        kelly = :None,
        rf = rf,
        rm = :UCI,
        points = 5,
    )

    risks1 = [
        calc_risk(
            fw1[:weights][!, i],
            returns;
            rm = :UCI,
            rf = rf,
            sigma = sigma,
            alpha_i = alpha_i,
            alpha = alpha,
            a_sim = a_sim,
            beta_i = beta_i,
            beta = beta,
            b_sim = b_sim,
            kappa = kappa,
            owa_w = owa_w,
            solvers = solvers,
        ) for i in 2:size(Matrix(fw1[:weights]), 2)
    ]
    rets1 =
        [dot(fw1[:weights][!, i], portfolio.mu) for i in 2:size(Matrix(fw1[:weights]), 2)]
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
    portfolio = Portfolio(
        prices = prices,
        solvers = OrderedDict(
            :Clarabel => Dict(
                :solver => Clarabel.Optimizer,
                :params => Dict("verbose" => false, "max_step_fraction" => 0.75),
            ),
            :COSMO =>
                Dict(:solver => COSMO.Optimizer, :params => Dict("verbose" => false)),
        ),
    )
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

    w1 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = :Classic,
        hist = 1,
        type = :Trad,
        rrp_ver = :None,
        u_mu = :None,
        u_cov = :None,
        rm = :EDaR,
        obj = :Min_Risk,
        kelly = :None,
    )
    w2 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = :Classic,
        hist = 1,
        type = :Trad,
        rrp_ver = :None,
        u_mu = :None,
        u_cov = :None,
        rm = :EDaR,
        obj = :Max_Ret,
        kelly = :None,
    )
    w3 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = :Classic,
        hist = 1,
        type = :Trad,
        rrp_ver = :None,
        u_mu = :None,
        u_cov = :None,
        rm = :EDaR,
        obj = :Sharpe,
        kelly = :None,
    )
    fw1 = efficient_frontier!(
        portfolio;
        class = :Classic,
        hist = 1,
        kelly = :None,
        rf = rf,
        rm = :EDaR,
        points = 5,
    )

    risks1 = [
        calc_risk(
            fw1[:weights][!, i],
            returns;
            rm = :EDaR,
            rf = rf,
            sigma = sigma,
            alpha_i = alpha_i,
            alpha = alpha,
            a_sim = a_sim,
            beta_i = beta_i,
            beta = beta,
            b_sim = b_sim,
            kappa = kappa,
            owa_w = owa_w,
            solvers = solvers,
        ) for i in 2:size(Matrix(fw1[:weights]), 2)
    ]
    rets1 =
        [dot(fw1[:weights][!, i], portfolio.mu) for i in 2:size(Matrix(fw1[:weights]), 2)]
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
    portfolio = Portfolio(
        prices = prices,
        solvers = OrderedDict(
            :Clarabel => Dict(
                :solver => Clarabel.Optimizer,
                :params => Dict("verbose" => false, "max_step_fraction" => 0.75),
            ),
            :COSMO =>
                Dict(:solver => COSMO.Optimizer, :params => Dict("verbose" => false)),
        ),
    )
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

    w1 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = :Classic,
        hist = 1,
        type = :Trad,
        rrp_ver = :None,
        u_mu = :None,
        u_cov = :None,
        rm = :RDaR,
        obj = :Min_Risk,
        kelly = :None,
    )
    w2 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = :Classic,
        hist = 1,
        type = :Trad,
        rrp_ver = :None,
        u_mu = :None,
        u_cov = :None,
        rm = :RDaR,
        obj = :Max_Ret,
        kelly = :None,
    )
    w3 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = :Classic,
        hist = 1,
        type = :Trad,
        rrp_ver = :None,
        u_mu = :None,
        u_cov = :None,
        rm = :RDaR,
        obj = :Sharpe,
        kelly = :None,
    )
    fw1 = efficient_frontier!(
        portfolio;
        class = :Classic,
        hist = 1,
        kelly = :None,
        rf = rf,
        rm = :RDaR,
        points = 5,
    )

    risks1 = [
        calc_risk(
            fw1[:weights][!, i],
            returns;
            rm = :RDaR,
            rf = rf,
            sigma = sigma,
            alpha_i = alpha_i,
            alpha = alpha,
            a_sim = a_sim,
            beta_i = beta_i,
            beta = beta,
            b_sim = b_sim,
            kappa = kappa,
            owa_w = owa_w,
            solvers = solvers,
        ) for i in 2:size(Matrix(fw1[:weights]), 2)
    ]
    rets1 =
        [dot(fw1[:weights][!, i], portfolio.mu) for i in 2:size(Matrix(fw1[:weights]), 2)]
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
    portfolio = Portfolio(
        prices = prices,
        solvers = OrderedDict(
            :Clarabel => Dict(
                :solver => Clarabel.Optimizer,
                :params => Dict("verbose" => false, "max_step_fraction" => 0.75),
            ),
            :COSMO =>
                Dict(:solver => COSMO.Optimizer, :params => Dict("verbose" => false)),
        ),
    )
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

    w1 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = :Classic,
        hist = 1,
        type = :Trad,
        rrp_ver = :None,
        u_mu = :None,
        u_cov = :None,
        rm = :Kurt,
        obj = :Min_Risk,
        kelly = :None,
    )
    w2 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = :Classic,
        hist = 1,
        type = :Trad,
        rrp_ver = :None,
        u_mu = :None,
        u_cov = :None,
        rm = :Kurt,
        obj = :Max_Ret,
        kelly = :None,
    )
    w3 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = :Classic,
        hist = 1,
        type = :Trad,
        rrp_ver = :None,
        u_mu = :None,
        u_cov = :None,
        rm = :Kurt,
        obj = :Sharpe,
        kelly = :None,
    )
    fw1 = efficient_frontier!(
        portfolio;
        class = :Classic,
        hist = 1,
        kelly = :None,
        rf = rf,
        rm = :Kurt,
        points = 5,
    )

    risks1 = [
        calc_risk(
            fw1[:weights][!, i],
            returns;
            rm = :Kurt,
            rf = rf,
            sigma = sigma,
            alpha_i = alpha_i,
            alpha = alpha,
            a_sim = a_sim,
            beta_i = beta_i,
            beta = beta,
            b_sim = b_sim,
            kappa = kappa,
            owa_w = owa_w,
            solvers = solvers,
        ) for i in 2:size(Matrix(fw1[:weights]), 2)
    ]
    rets1 =
        [dot(fw1[:weights][!, i], portfolio.mu) for i in 2:size(Matrix(fw1[:weights]), 2)]
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
    portfolio = Portfolio(
        prices = prices,
        solvers = OrderedDict(
            :Clarabel => Dict(
                :solver => Clarabel.Optimizer,
                :params => Dict("verbose" => false, "max_step_fraction" => 0.75),
            ),
            :COSMO =>
                Dict(:solver => COSMO.Optimizer, :params => Dict("verbose" => false)),
        ),
        max_num_assets_kurt = 1,
    )
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

    w1 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = :Classic,
        hist = 1,
        type = :Trad,
        rrp_ver = :None,
        u_mu = :None,
        u_cov = :None,
        rm = :Kurt,
        obj = :Min_Risk,
        kelly = :None,
    )
    w2 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = :Classic,
        hist = 1,
        type = :Trad,
        rrp_ver = :None,
        u_mu = :None,
        u_cov = :None,
        rm = :Kurt,
        obj = :Max_Ret,
        kelly = :None,
    )
    w3 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = :Classic,
        hist = 1,
        type = :Trad,
        rrp_ver = :None,
        u_mu = :None,
        u_cov = :None,
        rm = :Kurt,
        obj = :Sharpe,
        kelly = :None,
    )
    fw1 = efficient_frontier!(
        portfolio;
        class = :Classic,
        hist = 1,
        kelly = :None,
        rf = rf,
        rm = :Kurt,
        points = 5,
    )

    risks1 = [
        calc_risk(
            fw1[:weights][!, i],
            returns;
            rm = :Kurt,
            rf = rf,
            sigma = sigma,
            alpha_i = alpha_i,
            alpha = alpha,
            a_sim = a_sim,
            beta_i = beta_i,
            beta = beta,
            b_sim = b_sim,
            kappa = kappa,
            owa_w = owa_w,
            solvers = solvers,
        ) for i in 2:size(Matrix(fw1[:weights]), 2)
    ]
    rets1 =
        [dot(fw1[:weights][!, i], portfolio.mu) for i in 2:size(Matrix(fw1[:weights]), 2)]
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
    portfolio = Portfolio(
        prices = prices,
        solvers = OrderedDict(
            :Clarabel => Dict(
                :solver => Clarabel.Optimizer,
                :params => Dict("verbose" => false, "max_step_fraction" => 0.75),
            ),
            :COSMO =>
                Dict(:solver => COSMO.Optimizer, :params => Dict("verbose" => false)),
        ),
    )
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

    w1 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = :Classic,
        hist = 1,
        type = :Trad,
        rrp_ver = :None,
        u_mu = :None,
        u_cov = :None,
        rm = :SKurt,
        obj = :Min_Risk,
        kelly = :None,
    )
    w2 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = :Classic,
        hist = 1,
        type = :Trad,
        rrp_ver = :None,
        u_mu = :None,
        u_cov = :None,
        rm = :SKurt,
        obj = :Max_Ret,
        kelly = :None,
    )
    w3 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = :Classic,
        hist = 1,
        type = :Trad,
        rrp_ver = :None,
        u_mu = :None,
        u_cov = :None,
        rm = :SKurt,
        obj = :Sharpe,
        kelly = :None,
    )
    fw1 = efficient_frontier!(
        portfolio;
        class = :Classic,
        hist = 1,
        kelly = :None,
        rf = rf,
        rm = :SKurt,
        points = 5,
    )

    risks1 = [
        calc_risk(
            fw1[:weights][!, i],
            returns;
            rm = :SKurt,
            rf = rf,
            sigma = sigma,
            alpha_i = alpha_i,
            alpha = alpha,
            a_sim = a_sim,
            beta_i = beta_i,
            beta = beta,
            b_sim = b_sim,
            kappa = kappa,
            owa_w = owa_w,
            solvers = solvers,
        ) for i in 2:size(Matrix(fw1[:weights]), 2)
    ]
    rets1 =
        [dot(fw1[:weights][!, i], portfolio.mu) for i in 2:size(Matrix(fw1[:weights]), 2)]
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
    portfolio = Portfolio(
        prices = prices,
        solvers = OrderedDict(
            :Clarabel => Dict(
                :solver => Clarabel.Optimizer,
                :params => Dict("verbose" => false, "max_step_fraction" => 0.75),
            ),
            :COSMO =>
                Dict(:solver => COSMO.Optimizer, :params => Dict("verbose" => false)),
        ),
        max_num_assets_kurt = 1,
    )
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

    w1 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = :Classic,
        hist = 1,
        type = :Trad,
        rrp_ver = :None,
        u_mu = :None,
        u_cov = :None,
        rm = :SKurt,
        obj = :Min_Risk,
        kelly = :None,
    )
    w2 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = :Classic,
        hist = 1,
        type = :Trad,
        rrp_ver = :None,
        u_mu = :None,
        u_cov = :None,
        rm = :SKurt,
        obj = :Max_Ret,
        kelly = :None,
    )
    w3 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = :Classic,
        hist = 1,
        type = :Trad,
        rrp_ver = :None,
        u_mu = :None,
        u_cov = :None,
        rm = :SKurt,
        obj = :Sharpe,
        kelly = :None,
    )
    fw1 = efficient_frontier!(
        portfolio;
        class = :Classic,
        hist = 1,
        kelly = :None,
        rf = rf,
        rm = :SKurt,
        points = 5,
    )

    risks1 = [
        calc_risk(
            fw1[:weights][!, i],
            returns;
            rm = :SKurt,
            rf = rf,
            sigma = sigma,
            alpha_i = alpha_i,
            alpha = alpha,
            a_sim = a_sim,
            beta_i = beta_i,
            beta = beta,
            b_sim = b_sim,
            kappa = kappa,
            owa_w = owa_w,
            solvers = solvers,
        ) for i in 2:size(Matrix(fw1[:weights]), 2)
    ]
    rets1 =
        [dot(fw1[:weights][!, i], portfolio.mu) for i in 2:size(Matrix(fw1[:weights]), 2)]
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
    portfolio = Portfolio(
        prices = prices,
        solvers = OrderedDict(
            :Clarabel => Dict(
                :solver => Clarabel.Optimizer,
                :params => Dict("verbose" => false, "max_step_fraction" => 0.75),
            ),
            :COSMO =>
                Dict(:solver => COSMO.Optimizer, :params => Dict("verbose" => false)),
        ),
    )
    asset_statistics!(portfolio)

    w1 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = :Classic,
        hist = 1,
        type = :Trad,
        rrp_ver = :None,
        u_mu = :None,
        u_cov = :None,
        rm = :SKurt,
        obj = :Min_Risk,
        kelly = :None,
    )
    w2 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = :Classic,
        hist = 1,
        type = :Trad,
        rrp_ver = :None,
        u_mu = :None,
        u_cov = :None,
        rm = :SKurt,
        obj = :Max_Ret,
        kelly = :None,
    )
    w3 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = :Classic,
        hist = 1,
        type = :Trad,
        rrp_ver = :None,
        u_mu = :None,
        u_cov = :None,
        rm = :SKurt,
        obj = :Sharpe,
        kelly = :None,
    )
    fw1 = efficient_frontier!(
        portfolio;
        class = :Classic,
        hist = 1,
        kelly = :None,
        rf = rf,
        rm = :SKurt,
        points = 25,
    )
    risks1 = [
        calc_risk(
            fw1[:weights][!, i],
            portfolio.returns;
            rm = :SKurt,
            rf = rf,
            solvers = portfolio.solvers,
        ) for i in 2:size(Matrix(fw1[:weights]), 2)
    ]
    rets1 =
        [dot(fw1[:weights][!, i], portfolio.mu) for i in 2:size(Matrix(fw1[:weights]), 2)]
    idx = findlast(x -> x < risks1[end], risks1) + 1
    tmp = risks1[end]
    risks1[(idx + 1):end] = risks1[idx:(end - 1)]
    risks1[idx] = tmp
    tmp = rets1[end]
    rets1[(idx + 1):end] = rets1[idx:(end - 1)]
    rets1[idx] = tmp
    w4 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = :Classic,
        hist = 1,
        type = :Trad,
        rrp_ver = :None,
        u_mu = :None,
        u_cov = :None,
        rm = :SKurt,
        obj = :Min_Risk,
        kelly = :Approx,
    )
    w5 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = :Classic,
        hist = 1,
        type = :Trad,
        rrp_ver = :None,
        u_mu = :None,
        u_cov = :None,
        rm = :SKurt,
        obj = :Max_Ret,
        kelly = :Approx,
    )
    w6 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = :Classic,
        hist = 1,
        type = :Trad,
        rrp_ver = :None,
        u_mu = :None,
        u_cov = :None,
        rm = :SKurt,
        obj = :Sharpe,
        kelly = :Approx,
    )
    fw2 = efficient_frontier!(
        portfolio;
        class = :Classic,
        hist = 1,
        kelly = :Approx,
        rf = rf,
        rm = :SKurt,
        points = 25,
    )
    risks2 = [
        calc_risk(
            fw2[:weights][!, i],
            portfolio.returns;
            rm = :SKurt,
            rf = rf,
            solvers = portfolio.solvers,
        ) for i in 2:size(Matrix(fw2[:weights]), 2)
    ]
    rets2 = [
        1 / size(portfolio.returns, 1) *
        sum(log.(1 .+ portfolio.returns * fw2[:weights][!, i])) for
        i in 2:size(Matrix(fw2[:weights]), 2)
    ]
    idx = findlast(x -> x < risks2[end], risks2) + 1
    tmp = risks2[end]
    risks2[(idx + 1):end] = risks2[idx:(end - 1)]
    risks2[idx] = tmp
    tmp = rets2[end]
    rets2[(idx + 1):end] = rets2[idx:(end - 1)]
    rets2[idx] = tmp
    w7 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = :Classic,
        hist = 1,
        type = :Trad,
        rrp_ver = :None,
        u_mu = :None,
        u_cov = :None,
        rm = :CDaR,
        obj = :Min_Risk,
        kelly = :Exact,
    )
    w8 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = :Classic,
        hist = 1,
        type = :Trad,
        rrp_ver = :None,
        u_mu = :None,
        u_cov = :None,
        rm = :CDaR,
        obj = :Max_Ret,
        kelly = :Exact,
    )
    w9 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = :Classic,
        hist = 1,
        type = :Trad,
        rrp_ver = :None,
        u_mu = :None,
        u_cov = :None,
        rm = :CDaR,
        obj = :Sharpe,
        kelly = :Exact,
    )
    fw3 = efficient_frontier!(
        portfolio;
        class = :Classic,
        hist = 1,
        kelly = :Exact,
        rf = rf,
        rm = :CDaR,
        points = 25,
    )
    risks3 = [
        calc_risk(
            fw3[:weights][!, i],
            portfolio.returns;
            rm = :CDaR,
            rf = rf,
            solvers = portfolio.solvers,
        ) for i in 2:size(Matrix(fw3[:weights]), 2)
    ]
    rets3 = [
        1 / size(portfolio.returns, 1) *
        sum(log.(1 .+ portfolio.returns * fw3[:weights][!, i])) for
        i in 2:size(Matrix(fw3[:weights]), 2)
    ]
    idx = findlast(x -> x < risks3[end], risks3) + 1
    tmp = risks3[end]
    risks3[(idx + 1):end] = risks3[idx:(end - 1)]
    risks3[idx] = tmp
    tmp = rets3[end]
    rets3[(idx + 1):end] = rets3[idx:(end - 1)]
    rets3[idx] = tmp
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
    @test reduce(<=, risks3)
    @test reduce(<=, rets3)
    @test isapprox(fw3[:weights][!, 2], w7.weights)
    @test isapprox(fw3[:weights][!, end - 1], w8.weights)
    @test isapprox(fw3[:weights][!, end], w9.weights)

    plot_frontier(
        fw1,
        returns = portfolio.returns,
        mu = portfolio.mu,
        rm = :SKurt,
        kelly = false,
    )
    plot_frontier(
        fw1,
        returns = portfolio.returns,
        mu = portfolio.mu,
        rm = :SKurt,
        kelly = true,
    )
    plot_frontier(
        fw2,
        returns = portfolio.returns,
        mu = portfolio.mu,
        rm = :SKurt,
        kelly = true,
    )
    plot_frontier(
        fw2,
        returns = portfolio.returns,
        mu = portfolio.mu,
        rm = :SKurt,
        kelly = false,
    )
    plot_frontier(
        fw3,
        returns = portfolio.returns,
        mu = portfolio.mu,
        rm = :CDaR,
        kelly = true,
    )
    plot_frontier(
        fw3,
        returns = portfolio.returns,
        mu = portfolio.mu,
        rm = :CDaR,
        kelly = false,
    )
end
