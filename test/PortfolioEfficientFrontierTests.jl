using COSMO,
    CSV,
    Clarabel,
    HiGHS,
    LinearAlgebra,
    OrderedCollections,
    PortfolioOptimiser,
    Statistics,
    Test,
    TimeSeries

prices = TimeArray(CSV.File("./assets/stock_prices.csv"); timestamp = :date)

rf = 1.0329^(1 / 252) - 1
l = 2.0

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
