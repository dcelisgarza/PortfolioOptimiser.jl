using CSV, TimeSeries, DataFrames, StatsBase, Statistics, LinearAlgebra, Test, Clarabel,
      PortfolioOptimiser, JuMP, Clustering

prices = TimeArray(CSV.File("./assets/stock_prices.csv"); timestamp = :date)
factors = TimeArray(CSV.File("./assets/factor_prices.csv"); timestamp = :date)

rf = 1.0329^(1 / 252) - 1
l = 2.0

@testset "Portfolio" begin
    portfolio = Portfolio(; prices = prices,
                          solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                           :check_sol => (allow_local = true,
                                                                          allow_almost = true),
                                                           :params => Dict("verbose" => false,
                                                                           "max_step_fraction" => 0.75))))
    asset_statistics!(portfolio)
    @test_throws AssertionError portfolio.max_num_assets_kurt = -1
    @test portfolio.max_num_assets_kurt == 0
    portfolio.max_num_assets_kurt = 5
    @test portfolio.max_num_assets_kurt == 5
    @test portfolio.max_num_assets_kurt_scale == 2
    portfolio.max_num_assets_kurt_scale = 1
    @test portfolio.max_num_assets_kurt_scale == 1
    portfolio.max_num_assets_kurt_scale = size(portfolio.returns, 2) + 5
    @test portfolio.max_num_assets_kurt_scale == size(portfolio.returns, 2)
    @test isempty(portfolio.f_returns)
    portfolio.f_returns = portfolio.returns
    @test isequal(portfolio.f_returns, portfolio.returns)

    portfolio_copy = deepcopy(portfolio)
    for property ∈ propertynames(portfolio)
        port2_prop = getproperty(portfolio_copy, property)
        port_prop = getproperty(portfolio, property)
        if isa(port_prop, JuMP.Model)
            continue
        end
        @test isequal(port_prop, port2_prop)
    end

    N = size(prices, 2)
    M = size(factors, 2)
    A = rand(N, N)
    a_mtx_ineq = rand(3, N)
    skew = rand(N, N^2)
    sskew = rand(N, N^2)
    f_cov = rand(M, M)
    fm_cov = rand(N, N)
    bl_cov = rand(N, N)
    blfm_cov = rand(N, N)
    cov_l = rand(N, N)
    cov_u = rand(N, N)
    cov_mu = rand(N, N)
    cov_sigma = rand(N^2, N^2)
    portfolio = Portfolio(; prices = prices, f_prices = factors,
                          rebalance = TR(; val = 3, w = fill(inv(N), N)),
                          turnover = TR(; val = 5, w = fill(inv(2 * N), N)),
                          tracking_err = TrackWeight(; err = 7, w = fill(inv(3 * N), N)),
                          bl_bench_weights = fill(inv(4 * N), N),
                          network_method = SDP(; A = A), a_vec_cent = fill(inv(5 * N), N),
                          a_mtx_ineq = a_mtx_ineq, risk_budget = 1:N,
                          f_risk_budget = 1:div(N, 2), skew = skew, sskew = sskew,
                          f_mu = fill(inv(M), M), f_cov = f_cov,
                          fm_mu = fill(inv(6 * N), N), fm_cov = fm_cov,
                          bl_mu = fill(inv(7 * N), N), bl_cov = bl_cov,
                          blfm_mu = fill(inv(8 * N), N), blfm_cov = blfm_cov, cov_l = cov_l,
                          cov_u = cov_u, cov_mu = cov_mu, cov_sigma = cov_sigma,
                          d_mu = fill(inv(9 * N), N))
    portfolio.returns = 2 * portfolio.returns
    @test portfolio.rebalance.val == 3
    @test portfolio.rebalance.w == fill(inv(N), N)
    @test portfolio.turnover.val == 5
    @test portfolio.turnover.w == fill(inv(2 * N), N)
    @test portfolio.tracking_err.err == 7
    @test portfolio.tracking_err.w == fill(inv(3 * N), N)
    @test portfolio.bl_bench_weights == fill(inv(4 * N), N)
    @test portfolio.network_method.A == A
    @test portfolio.a_vec_cent == fill(inv(5 * N), N)
    @test portfolio.a_mtx_ineq == a_mtx_ineq
    @test portfolio.risk_budget == collect(1:N) / sum(1:N)
    @test portfolio.f_risk_budget == collect(1:div(N, 2)) / sum(1:div(N, 2))
    @test portfolio.skew == skew
    @test portfolio.sskew == sskew
    @test portfolio.f_mu == fill(inv(M), M)
    @test portfolio.f_cov == f_cov
    @test portfolio.fm_mu == fill(inv(6 * N), N)
    @test portfolio.fm_cov == fm_cov
    @test portfolio.bl_mu == fill(inv(7 * N), N)
    @test portfolio.bl_cov == bl_cov
    @test portfolio.blfm_mu == fill(inv(8 * N), N)
    @test portfolio.blfm_cov == blfm_cov
    @test portfolio.cov_l == cov_l
    @test portfolio.cov_u == cov_u
    @test portfolio.cov_mu == cov_mu
    @test portfolio.cov_sigma == cov_sigma
    @test portfolio.d_mu == fill(inv(9 * N), N)

    M = size(portfolio_copy.returns, 1)
    portfolio = Portfolio(; prices = prices, f_prices = factors,
                          rebalance = TR(; val = fill(inv(N), N)),
                          turnover = TR(; val = fill(inv(2 * N), N)),
                          risk_budget = collect(1.0:N),
                          f_risk_budget = collect(1.0:div(N, 2)),
                          tracking_err = TrackRet(; err = 11, w = fill(inv(100 * M), M)))
    @test portfolio.rebalance.val == fill(inv(N), N)
    @test portfolio.turnover.val == fill(inv(2 * N), N)
    @test portfolio.risk_budget == collect(1:N) / sum(1:N)
    @test portfolio.f_risk_budget == collect(1:div(N, 2)) / sum(1:div(N, 2))
    @test portfolio.tracking_err.err == 11
    @test portfolio.tracking_err.w == fill(inv(100 * M), M)

    @test_throws AssertionError Portfolio(; prices = prices, rebalance = TR(; val = -eps()))
end

@testset "HC Portfolio" begin
    portfolio = HCPortfolio(; prices = prices,
                            solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                             :check_sol => (allow_local = true,
                                                                            allow_almost = true),
                                                             :params => Dict("verbose" => false,
                                                                             "max_step_fraction" => 0.75))))
    asset_statistics!(portfolio)
    cluster_assets!(portfolio)

    portfolio_copy = deepcopy(portfolio)
    for property ∈ propertynames(portfolio)
        port2_prop = getproperty(portfolio_copy, property)
        port_prop = getproperty(portfolio, property)
        if isa(port_prop, JuMP.Model) ||
           isa(port_prop, PortfolioOptimiser.PortfolioOptimiserCovCor) ||
           isa(port_prop, Clustering.Hclust)
            continue
        end
        @test isequal(port_prop, port2_prop)
    end
    @test portfolio_copy.clusters.merges == portfolio.clusters.merges
    @test portfolio_copy.clusters.heights == portfolio.clusters.heights
    @test portfolio_copy.clusters.order == portfolio.clusters.order

    @test_throws AssertionError portfolio.w_min = 1:(size(portfolio.returns, 2) + 1)
    @test_throws AssertionError portfolio.w_max = 1:(size(portfolio.returns, 2) + 1)

    returns = dropmissing!(DataFrame(percentchange(prices)))
    N = length(names(returns)) - 1

    sigma = rand(N, N)
    kurt = rand(N^2, N^2)
    skurt = rand(N^2, N^2)
    skew = rand(N, N^2)
    sskew = rand(N, N^2)
    V = rand(N, N)
    SV = rand(N, N)
    rho = rand(N, N)
    delta = rand(N, N)

    portfolio = HCPortfolio(; assets = setdiff(names(returns), ("timestamp",)),
                            ret = Matrix(returns[!,
                                                 setdiff(names(returns), ("timestamp",))]),
                            mu = fill(inv(N), N), cov = sigma, kurt = kurt, skurt = skurt,
                            skew = skew, sskew = sskew, V = V, SV = SV, w_min = 0.2,
                            w_max = 0.8, cor = rho, dist = delta)
    @test portfolio.assets == setdiff(names(returns), ("timestamp",))
    @test portfolio.returns == Matrix(returns[!, setdiff(names(returns), ("timestamp",))])
    @test portfolio.mu == fill(inv(N), N)
    @test portfolio.cov == sigma
    @test portfolio.kurt == kurt
    @test portfolio.skurt == skurt
    @test portfolio.skew == skew
    @test portfolio.sskew == sskew
    @test portfolio.V == V
    @test portfolio.SV == SV
    @test portfolio.cor == rho
    @test portfolio.dist == delta

    portfolio = HCPortfolio(; prices = prices, w_min = 0.01:0.01:0.2,
                            w_max = 0.02:0.01:0.21)
    @test portfolio.w_min == 0.01:0.01:0.2
    @test portfolio.w_max == 0.02:0.01:0.21

    portfolio = HCPortfolio(; prices = prices, w_min = 1, w_max = 1:20)
    @test portfolio.w_min == 1
    @test portfolio.w_max == 1:20

    portfolio = HCPortfolio(; prices = prices, w_max = 21, w_min = 1:20)
    @test portfolio.w_min == 1:20
    @test portfolio.w_max == 21

    portfolio = HCPortfolio(; prices = prices, w_min = 1, w_max = 1:20)
    @test portfolio.w_min == 1
    @test portfolio.w_max == 1:20

    portfolio.w_min = 1:20
    portfolio.w_max = 21
    @test portfolio.w_min == 1:20
    @test portfolio.w_max == 21
end
