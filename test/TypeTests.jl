using CSV, TimeSeries, DataFrames, StatsBase, Statistics, LinearAlgebra, Test, Clarabel,
      PortfolioOptimiser, JuMP

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
    portfolio = Portfolio(; prices = prices, f_prices = factors, rebalance = TR(; val = 3),
                          turnover = TR(; val = 5),
                          tracking_err = TrackWeight(; err = 7, w = fill(inv(N), N)),
                          bl_bench_weights = fill(inv(2 * N), N),
                          network_method = SDP(; A = A), a_vec_cent = fill(inv(3 * N), N),
                          a_mtx_ineq = zeros(3, N), risk_budget = 1:N,
                          f_risk_budget = 1:div(N, 2), skew = skew, sskew = sskew,
                          f_mu = fill(inv(M), M), f_cov = f_cov,
                          fm_mu = fill(inv(5 * N), N), fm_cov = fm_cov,
                          bl_mu = fill(inv(6 * N), N), bl_cov = bl_cov,
                          blfm_mu = fill(inv(7 * N), N), blfm_cov = blfm_cov, cov_l = cov_l,
                          cov_u = cov_u, cov_mu = cov_mu, cov_sigma = cov_sigma,
                          d_mu = fill(inv(8 * N), N))
    @test portfolio.rebalance.val == 3
    @test portfolio.turnover.val == 5
    @test portfolio.tracking_err.err == 7
    @test portfolio.tracking_err.w == fill(inv(N), N)

    @test_throws AssertionError Portfolio(; prices = prices, rebalance = TR(; val = -eps()))
end
