using CSV, TimeSeries, DataFrames, StatsBase, Statistics, LinearAlgebra, Test, Clarabel,
      PortfolioOptimiser, JuMP, Clustering, SparseArrays

prices_path = joinpath(@__DIR__, "assets/stock_prices.csv")
prices = TimeArray(CSV.File(prices_path); timestamp = :date)
factors_path = joinpath(@__DIR__, "assets/factor_prices.csv")
factors = TimeArray(CSV.File(factors_path); timestamp = :date)
rf = 1.0329^(1 / 252) - 1
l = 2.0

@testset "Type tests" begin
    portfolio = Portfolio(; prices = prices, short = true, budget = 3.0,
                          short_budget = -0.5, long_ub = 1.0, short_lb = -0.3,
                          solvers = PortOptSolver(; name = :Clarabel,
                                                  solver = Clarabel.Optimizer,
                                                  check_sol = (; allow_local = true,
                                                               allow_almost = true),
                                                  params = Dict("verbose" => false,
                                                                "max_step_fraction" => 0.75)))
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
    portfolio.bl_bench_weights = 1:20
    @test portfolio.bl_bench_weights == collect(Float64, 1:20)
    portfolio.bl_bench_weights = []
    @test isempty(portfolio.bl_bench_weights)

    portfolio_copy = deepcopy(portfolio)
    for property ∈ propertynames(portfolio)
        port2_prop = getproperty(portfolio_copy, property)
        port_prop = getproperty(portfolio, property)
        if isa(port_prop, JuMP.Model) || isa(port_prop, Clustering.Hclust)
            continue
        end
        @test isequal(port_prop, port2_prop)
    end

    N = size(prices, 2)
    M = size(factors, 2)
    A = rand(N, N)
    a_ineq = rand(3, N)
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
    V = rand(N, N)
    SV = rand(N, N)
    L_2 = sprand(Float16, Int(N * (N + 1) / 2), N^2, 0.2)
    S_2 = sprand(Float16, Int(N * (N + 1) / 2), N^2, 0.2)

    portfolio = Portfolio(; prices = prices, network_adj = SDP(; A = A),
                          cluster_adj = SDP(; A = A))
    portfolio = Portfolio(; prices = prices, network_adj = IP(; A = A),
                          cluster_adj = IP(; A = A))
    portfolio = Portfolio(; prices = prices, f_prices = factors,
                          fees = Fees(; long = fill(1, N), short = fill(-3, N)),
                          bl_bench_weights = 2 * (1:N),
                          rebalance = TR(; val = 3, w = fill(inv(N), N)),
                          turnover = TR(; val = 5, w = fill(inv(2 * N), N)),
                          tracking = TrackWeight(; err = 11, w = fill(inv(3 * N), N)),
                          network_adj = SDP(; A = A), cluster_adj = IP(; A = A),
                          a_eq = fill(inv(5 * N), N)', a_ineq = a_ineq, risk_budget = 1:N,
                          f_risk_budget = 1:5, L_2 = L_2, S_2 = S_2, skew = skew,
                          sskew = sskew, f_mu = fill(inv(M), M), f_cov = f_cov,
                          fm_mu = fill(inv(6 * N), N), fm_cov = fm_cov,
                          bl_mu = fill(inv(7 * N), N), bl_cov = bl_cov,
                          blfm_mu = fill(inv(8 * N), N), blfm_cov = blfm_cov, cov_l = cov_l,
                          cov_u = cov_u, cov_mu = cov_mu, cov_sigma = cov_sigma,
                          d_mu = fill(inv(9 * N), N), V = V, SV = SV)
    portfolio.returns = 2 * portfolio.returns
    @test portfolio.fees.long == fill(1, N)
    @test portfolio.fees.short == fill(-3, N)
    @test portfolio.rebalance.val == 3
    @test portfolio.rebalance.w == fill(inv(N), N)
    @test portfolio.turnover.val == 5
    @test portfolio.turnover.w == fill(inv(2 * N), N)
    @test portfolio.tracking.err == 11
    @test portfolio.tracking.w == fill(inv(3 * N), N)
    @test portfolio.network_adj.A == A
    @test portfolio.a_eq == fill(inv(5 * N), N)'
    @test portfolio.a_ineq == a_ineq
    @test portfolio.risk_budget == collect(1:N)
    @test portfolio.bl_bench_weights == 2 * (1:N)
    @test portfolio.f_risk_budget == collect(1:5)
    @test portfolio.L_2 == L_2
    @test portfolio.S_2 == S_2
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
    @test portfolio.V == V
    @test portfolio.SV == SV

    @test_throws AssertionError Portfolio(prices = prices, w_min = 0.2,
                                          w_max = fill(0.1, N))
    portfolio = Portfolio(; prices = prices, w_min = 0.2, w_max = fill(0.2, N))
    @test all(isapprox.(portfolio.w_min, portfolio.w_max))

    @test_throws AssertionError Portfolio(prices = prices, w_max = 0.1,
                                          w_min = fill(0.2, N))
    portfolio = Portfolio(; prices = prices, w_max = 0.2, w_min = fill(0.2, N))
    @test all(isapprox.(portfolio.w_max, portfolio.w_min))

    portfolio.w_min = zeros(N)
    portfolio.w_max = 1

    portfolio.long_ub = 0.7
    portfolio.budget_lb = 0.7
    portfolio.budget = Inf
    portfolio.budget_lb = Inf
    @test isone(portfolio.budget)

    portfolio.short = true
    portfolio.short_budget_ub = -0.1
    portfolio.short_lb = -0.1
    portfolio.short_budget = -Inf
    portfolio.short_budget_ub = -Inf
    @test isapprox(portfolio.short_budget, -0.2)

    portfolio.short_budget_lb = -0.1
    portfolio.short_lb = -0.1
    portfolio.short_budget = -Inf
    portfolio.short_budget_lb = -Inf
    @test isapprox(portfolio.short_budget, -0.2)

    A = rand(3, N)
    B = rand(3)
    portfolio.a_card_eq = A
    @test isapprox(portfolio.a_card_eq, A)
    portfolio.b_card_eq = B
    @test isapprox(portfolio.b_card_eq, B)
    @test_throws AssertionError portfolio.a_card_eq = rand(3, N + 1)
    @test_throws AssertionError portfolio.a_card_eq = rand(4, N)
    @test_throws AssertionError portfolio.b_card_eq = rand(4)

    l, s, fl, fs = rand(N), -rand(N), rand(N), -rand(N)
    portfolio.fees = Fees(; long = l, short = s, fixed_long = fl, fixed_short = fs)
    @test isequal(portfolio.fees,
                  Fees(; long = l, short = s, fixed_long = fl, fixed_short = fs))
    @test_throws AssertionError portfolio.fees = Fees(; long = rand(5), short = s,
                                                      fixed_long = fl, fixed_short = fs)
    @test_throws AssertionError portfolio.fees = Fees(; long = 5, short = -rand(2),
                                                      fixed_long = fl, fixed_short = fs)
    @test_throws AssertionError portfolio.fees = Fees(; long = 5, short = -2,
                                                      fixed_long = rand(7),
                                                      fixed_short = fs)
    @test_throws AssertionError portfolio.fees = Fees(; long = 5, short = -2,
                                                      fixed_long = 7,
                                                      fixed_short = -rand(1))

    @test_throws AssertionError portfolio.scale_constr = -1
    @test_throws AssertionError portfolio.scale_obj = -1

    M = size(portfolio_copy.returns, 1)
    kurt = rand(N^2, N^2)
    skurt = rand(N^2, N^2)
    portfolio = Portfolio(; prices = prices, f_prices = factors, kurt = kurt, skurt = skurt,
                          rebalance = TR(; val = fill(inv(N), N)),
                          turnover = TR(; val = fill(inv(2 * N), N)), w_min = 0.01:0.01:0.2,
                          w_max = 0.02:0.01:0.21, risk_budget = collect(1.0:N),
                          bl_bench_weights = collect(2 * (1:N)),
                          f_risk_budget = collect(1.0:5),
                          tracking = TrackRet(; err = 11, w = fill(inv(100 * M), M)))
    @test portfolio.w_min == 0.01:0.01:0.2
    @test portfolio.w_max == 0.02:0.01:0.21
    @test portfolio.rebalance.val == fill(inv(N), N)
    @test portfolio.turnover.val == fill(inv(2 * N), N)
    @test portfolio.risk_budget == collect(1:N)
    @test portfolio.bl_bench_weights == collect(2 * (1:N))
    @test portfolio.f_risk_budget == collect(1:5)
    @test portfolio.tracking.err == 11
    @test portfolio.tracking.w == fill(inv(100 * M), M)
    @test portfolio.kurt == kurt
    @test portfolio.skurt == skurt
    @test_throws AssertionError Portfolio(; prices = prices, rebalance = TR(; val = -eps()))
    @test_throws AssertionError Portfolio(; prices = prices, long_t = fill(0, 19))
    @test_throws AssertionError Portfolio(; prices = prices, long_t = -1)
    @test_throws AssertionError Portfolio(; prices = prices, long_t = fill(-1, 20))
    portfolio = Portfolio(; prices = prices, budget = Inf)
    @test isone(portfolio.budget)
    portfolio = Portfolio(; prices = prices, budget = Inf, budget_lb = 1.3, budget_ub = 1.5)
    @test isinf(portfolio.budget)
    @test portfolio.budget_lb == 1.3
    @test portfolio.budget_ub == 1.5
    @test_throws AssertionError portfolio.budget_lb = 1.6
    @test_throws AssertionError portfolio.budget_ub = 1.2
    portfolio.budget_ub = 1.5
    @test_throws AssertionError portfolio.budget_lb = 1.6
    portfolio.budget_lb = Inf
    portfolio.budget_ub = Inf
    @test isone(portfolio.budget)
    @test_throws AssertionError Portfolio(; prices = prices, budget = Inf, budget_lb = 1.7,
                                          budget_ub = 1.3)
    portfolio = Portfolio(; prices = prices, short = true, short_budget = Inf,
                          short_budget_lb = -0.5, short_budget_ub = -0.3)
    @test isinf(portfolio.short_budget)
    @test portfolio.short_budget_lb == -0.5
    @test portfolio.short_budget_ub == -0.3
    portfolio = Portfolio(; prices = prices, short = true, budget = Inf, budget_lb = 0.7,
                          budget_ub = 1.3, short_budget = Inf, short_budget_lb = -0.5,
                          short_budget_ub = -0.3)
    @test portfolio.budget == Inf
    @test portfolio.budget_lb == 0.7
    @test portfolio.budget_ub == 1.3
    @test portfolio.short_budget == Inf
    @test portfolio.short_budget_lb == -0.5
    @test portfolio.short_budget_ub == -0.3

    portfolio = Portfolio(; prices = prices)
    portfolio.budget = Inf
    @assert isone(portfolio.budget)

    A = [0 1 0;
         1 0 0;
         0 0 0]
    ip = IP(; A = A, k = [1, 3])
    @test ip.A == [1 1 0;
                   0 0 1]
    @test ip.k == [1, 3]

    ip.A = A
    @test ip.A == [1 1 0;
                   0 0 1]

    ip = IP(; k = [1, 2, 3])
    A = [0 1 0 1;
         1 0 0 1;
         0 0 0 1;
         1 1 1 0]
    ip.A = A
    @test ip.A == [1.0  1.0  0.0  1.0;
                   0.0  0.0  1.0  1.0;
                   1.0  1.0  1.0  1.0]
    @test ip.k == [1, 2, 3]
    ip.k = [3, 4, 5]
    @test ip.k == [3, 4, 5]

    ip = IP(; A = A)

    ip.k = [6, 8, 9]
    @test ip.k == [6, 8, 9]

    portfolio.short = false
    portfolio.budget = 1.5
    portfolio.long_ub = 1.5
    portfolio.short_budget = -0.5
    portfolio.short_lb = -0.5

    @test_throws AssertionError portfolio.w_min = 1:21
    @test_throws AssertionError portfolio.w_max = 1:21

    @test_throws AssertionError Sharpe(rf = -1, ohf = -1)
    rm = Sharpe()
    @test_throws AssertionError rm.ohf = -1

    @test_throws AssertionError ScalarLogSumExp(gamma = -1)
    sc = ScalarLogSumExp()
    @test_throws AssertionError sc.gamma = -1
    sc.gamma = 1
    @test sc.gamma == 1

    @test_throws AssertionError SchurParams(; gamma = -1)
    @test_throws AssertionError SchurParams(; gamma = 2)
    @test_throws AssertionError SchurParams(; prop_coef = -1)
    @test_throws AssertionError SchurParams(; prop_coef = 2)
    @test_throws AssertionError SchurParams(; tol = 0)
    @test_throws AssertionError SchurParams(; max_iter = 0)

    scp = SchurParams()
    @test_throws AssertionError scp.gamma = -1
    @test_throws AssertionError scp.gamma = 2
    @test_throws AssertionError scp.prop_coef = -1
    @test_throws AssertionError scp.prop_coef = 2
    @test_throws AssertionError scp.tol = 0
    @test_throws AssertionError scp.max_iter = 0
    scp.tol = 1e-6
    @test scp.tol == 1e-6

    class = FM()
    @test_throws AssertionError class.type = 3
    class.type = 2
    @test class.type == 2
    class.type = 1
    @test class.type == 1

    class = BL()
    @test_throws AssertionError class.type = 3
    class.type = 2
    @test class.type == 2
    class.type = 1
    @test class.type == 1

    class = BLFM()
    @test_throws AssertionError class.type = 4
    class.type = 3
    @test class.type == 3
    class.type = 2
    @test class.type == 2
    class.type = 1
    @test class.type == 1

    root_type = NCO(;
                    internal = NCOArgs(;
                                       type = NCO(;
                                                  internal = NCOArgs(;
                                                                     type = NCO(;
                                                                                internal = NCOArgs(;
                                                                                                   type = Trad(;
                                                                                                               obj = MinRisk(),
                                                                                                               rm = SD())),
                                                                                external = NCOArgs(;
                                                                                                   type = Trad(;
                                                                                                               obj = Utility(),
                                                                                                               rm = MAD())))),
                                                  external = NCOArgs(;
                                                                     type = NCO(;
                                                                                internal = NCOArgs(;
                                                                                                   type = Trad(;
                                                                                                               obj = Sharpe(),
                                                                                                               rm = Kurt())),
                                                                                external = NCOArgs(;
                                                                                                   type = Trad(;
                                                                                                               obj = MaxRet(),
                                                                                                               rm = RG())))))),
                    external = NCOArgs(; type = Trad(; rm = SSD())))

    @test isa(root_type.rm, SD)
    @test isa(root_type.rm_o, SSD)
    @test isa(root_type.internal.type.internal.type.rm, SD)
    @test isa(root_type.internal.type.internal.type.rm_o, MAD)
    @test isa(root_type.internal.type.internal.type.internal.type.rm, SD)
    @test isa(root_type.internal.type.internal.type.external.type.rm, MAD)
    @test isa(root_type.internal.type.external.type.rm, Kurt)
    @test isa(root_type.internal.type.external.type.rm_o, RG)
    @test isa(root_type.internal.type.external.type.internal.type.rm, Kurt)
    @test isa(root_type.internal.type.external.type.external.type.rm, RG)

    rm = Sharpe()
    @test_throws AssertionError rm.rf = -0.1
    rm.rf = 0.0001
    @test rm.rf == 0.0001
    @test_throws AssertionError rm.ohf = -0.1
    rm.ohf = 0.0001
    @test rm.ohf == 0.0001

    A = PortOptSolver(; name = :Clarabel, solver = Clarabel.Optimizer,
                      check_sol = (; allow_local = true, allow_almost = true),
                      params = Dict("verbose" => false, "max_step_fraction" => 0.75))
    B = PortOptSolver(; name = :Clarabel, solver = Clarabel.Optimizer,
                      check_sol = (; allow_local = true, allow_almost = true),
                      params = Dict("verbose" => false, "max_step_fraction" => 0.76))

    @test A != B
end
