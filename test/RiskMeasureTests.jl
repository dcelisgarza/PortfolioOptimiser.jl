using CSV, TimeSeries, StatsBase, Statistics, LinearAlgebra, Test, Clarabel,
      PortfolioOptimiser

path = joinpath(@__DIR__, "assets/stock_prices.csv")
prices = TimeArray(CSV.File(path); timestamp = :date)

@testset "Risk measures" begin
    portfolio = Portfolio(; prices = prices,
                          solvers = PortOptSolver(; name = :Clarabel,
                                                  solver = Clarabel.Optimizer,
                                                  check_sol = (; allow_local = true,
                                                               allow_almost = true),
                                                  params = "verbose" => false))

    asset_statistics!(portfolio)
    optimise!(portfolio, Trad(; obj = Sharpe()))

    risks = [0.012732818438689692, 0.009054428305486948, 0.008974573704221727,
             0.0038588790457400107, 0.008278637116930432, 0.04767699262536566,
             0.029180900700100272, 0.035143848572313445, 0.039976800810474805,
             0.2582974717056197, 0.027337478660655532, 0.1608684430842145,
             0.04867273393167054, 0.1972857886760955, 0.2244731723112284,
             0.00040887694942707355, 0.00026014656311027184, 0.013414894061397202,
             0.11233644033448399, 0.0604743621201116, 0.03342440909106323,
             0.07034954287279171, 0.013414894061397202, 0.0005034445044981896,
             9.018123494110702e-6, 2.7400249395078545e-5, 0.0001621246653926362, 0.05,
             0.01882393038479881, 0.10822447250466571, 0.11100155140762302,
             0.23276849033778024, 0.02742352667080502, 0.1529564256640314,
             0.04740180533190617, 0.18166001933455683, 0.2039781842753333]

    rms = [SD(), MAD(), SSD(), FLPM(), SLPM(), WR(), CVaR(), EVaR(), RLVaR(), MDD(), ADD(),
           CDaR(), UCI(), EDaR(), RLDaR(), Kurt(), SKurt(), GMD(), RG(), CVaRRG(), TG(),
           TGRG(), OWA(), BDVariance(), NQSkew(), NQSSkew(), Variance(), Equal(), VaR(),
           DaR(), DaR_r(), MDD_r(), ADD_r(), CDaR_r(), UCI_r(), EDaR_r(), RLDaR_r()]

    for (risk, rm) ∈ zip(risks, rms)
        r = calc_risk(portfolio, :Trad; rm = rm)
        @test isapprox(risk, r, rtol = 5e-7)
        @test length(rm) == 1
        @test rm[rand(Int)] == rm
    end
end

@testset "Aux functions" begin
    rm = [SD(), [FLPM()], [SLPM(), WR()]]

    rm1 = PortfolioOptimiser.get_first_rm(rm)
    rm2 = PortfolioOptimiser.get_rm_symbol(rm)
    rm3 = PortfolioOptimiser.get_rm_symbol(WR())
    rm4 = PortfolioOptimiser.get_rm_symbol(SD())

    @test rm1 == rm[1]
    # @test rm2 == :SD2_FLPM2_SLPM2_WR2
    # @test rm3 == :WR
    # @test rm4 == :SD
end

@testset "Constructors and setters" begin
    @test_throws AssertionError SD(sigma = rand(5, 3))
    @test_throws AssertionError SD(; sigma = Matrix(undef, 0, 1))
    rm = SD()
    @test_throws AssertionError rm.sigma = Matrix(undef, 0, 1)

    @test_throws AssertionError Variance(sigma = rand(5, 3))
    @test_throws AssertionError Variance(; sigma = Matrix(undef, 0, 1))
    rm = Variance()
    @test_throws AssertionError rm.sigma = Matrix(undef, 0, 1)

    rm = VaR()
    @test_throws AssertionError rm.alpha = 1
    @test_throws AssertionError rm.alpha = 0
    rm.alpha = 0.5
    @test rm.alpha == 0.5

    rm = CVaR()
    @test_throws AssertionError rm.alpha = 1
    @test_throws AssertionError rm.alpha = 0
    rm.alpha = 0.5
    @test rm.alpha == 0.5

    rm = EVaR()
    @test_throws AssertionError rm.alpha = 1
    @test_throws AssertionError rm.alpha = 0
    rm.alpha = 0.5
    @test rm.alpha == 0.5
    @test Symbol(rm) == :EVaR
    @test String(rm) == "EVaR"

    rm = RLVaR()
    @test_throws AssertionError rm.alpha = 1
    @test_throws AssertionError rm.alpha = 0
    @test_throws AssertionError rm.kappa = 1
    @test_throws AssertionError rm.kappa = 0
    rm.alpha = 0.5
    @test rm.alpha == 0.5
    rm.kappa = 0.5
    @test rm.kappa == 0.5
    @test Symbol(rm) == :RLVaR
    @test String(rm) == "RLVaR"

    rm = DaR()
    @test_throws AssertionError rm.alpha = 1
    @test_throws AssertionError rm.alpha = 0
    rm.alpha = 0.5
    @test rm.alpha == 0.5

    rm = CDaR()
    @test_throws AssertionError rm.alpha = 1
    @test_throws AssertionError rm.alpha = 0
    rm.alpha = 0.5
    @test rm.alpha == 0.5

    rm = EDaR()
    @test_throws AssertionError rm.alpha = 1
    @test_throws AssertionError rm.alpha = 0
    @test Symbol(rm) == :EDaR
    @test String(rm) == "EDaR"
    rm.alpha = 0.5
    @test rm.alpha == 0.5

    rm = RLDaR()
    @test_throws AssertionError rm.alpha = 1
    @test_throws AssertionError rm.alpha = 0
    @test_throws AssertionError rm.kappa = 1
    @test_throws AssertionError rm.kappa = 0
    rm.alpha = 0.5
    @test rm.alpha == 0.5
    rm.kappa = 0.5
    @test rm.kappa == 0.5
    @test Symbol(rm) == :RLDaR
    @test String(rm) == "RLDaR"

    rm = DaR_r()
    @test_throws AssertionError rm.alpha = 1
    @test_throws AssertionError rm.alpha = 0
    rm.alpha = 0.5
    @test rm.alpha == 0.5

    rm = CDaR_r()
    @test_throws AssertionError rm.alpha = 1
    @test_throws AssertionError rm.alpha = 0
    rm.alpha = 0.5
    @test rm.alpha == 0.5

    rm = EDaR_r()
    @test_throws AssertionError rm.alpha = 1
    @test_throws AssertionError rm.alpha = 0
    rm.alpha = 0.5
    @test rm.alpha == 0.5

    rm = RLDaR_r()
    @test_throws AssertionError rm.alpha = 1
    @test_throws AssertionError rm.alpha = 0
    @test_throws AssertionError rm.kappa = 1
    @test_throws AssertionError rm.kappa = 0
    rm.alpha = 0.5
    @test rm.alpha == 0.5
    rm.kappa = 0.5
    @test rm.kappa == 0.5

    rm = CVaRRG()
    @test_throws AssertionError rm.alpha = 1
    @test_throws AssertionError rm.alpha = 0
    @test_throws AssertionError rm.beta = 1
    @test_throws AssertionError rm.beta = 0
    rm.alpha = 0.5
    @test rm.alpha == 0.5
    rm.beta = 0.5
    @test rm.beta == 0.5
    rm.beta = 0.5
    @test rm.beta == 0.5

    rm = TG()
    @test_throws AssertionError rm.alpha = 1
    @test_throws AssertionError rm.alpha = 0
    @test_throws AssertionError rm.alpha_i = 1
    @test_throws AssertionError rm.alpha_i = 0
    @test_throws AssertionError rm.a_sim = 0
    rm.alpha = 0.5
    @test rm.alpha == 0.5
    rm.alpha_i = 0.05
    @test rm.alpha_i == 0.05
    rm.a_sim = 5
    @test rm.a_sim == 5

    rm = TGRG()
    @test_throws AssertionError rm.alpha = 1
    @test_throws AssertionError rm.alpha = 0
    @test_throws AssertionError rm.alpha_i = 1
    @test_throws AssertionError rm.alpha_i = 0
    @test_throws AssertionError rm.a_sim = 0
    @test_throws AssertionError rm.beta = 1
    @test_throws AssertionError rm.beta = 0
    @test_throws AssertionError rm.beta_i = 1
    @test_throws AssertionError rm.beta_i = 0
    @test_throws AssertionError rm.b_sim = 0
    rm.alpha = 0.5
    @test rm.alpha == 0.5
    rm.alpha_i = 0.05
    @test rm.alpha_i == 0.05
    rm.a_sim = 5
    @test rm.a_sim == 5
    rm.beta = 0.5
    @test rm.beta == 0.5
    rm.beta_i = 0.05
    @test rm.beta_i == 0.05
    rm.b_sim = 5
    @test rm.b_sim == 5

    settings = RMSettings(; scale = 0.1, ub = 0.5)
    @test settings.scale == 0.1
    @test settings.ub == 0.5
    settings.scale = 0.6
    settings.ub = 10.0
    @test settings.scale == 0.6
    @test settings.ub == 10.0
    settings.ub = Inf
    @test isinf(settings.ub)

    hcsettings = HCRMSettings(; scale = 0.1)
    @test hcsettings.scale == 0.1
    hcsettings.scale = 3.0
    @test hcsettings.scale == 3.0
end
