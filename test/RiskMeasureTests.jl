using CSV, TimeSeries, StatsBase, Statistics, LinearAlgebra, Test, Clarabel,
      PortfolioOptimiser

prices = TimeArray(CSV.File("./assets/stock_prices.csv"); timestamp = :date)

@testset "Risk measures" begin
    portfolio = Portfolio(; prices = prices,
                          solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                           :check_sol => (allow_local = true,
                                                                          allow_almost = true),
                                                           :params => Dict("verbose" => false))))
    asset_statistics!(portfolio)
    optimise!(portfolio; obj = Sharpe())

    risks = [0.012732818438689692, 0.009054428305486948, 0.008974573704221727,
             0.003858879045730223, 0.008278637116930432, 0.04767699262536566,
             0.029180900700100272, 0.035143848572313445, 0.039976809079791106,
             0.2582974717056197, 0.027337478660655532, 0.1608684430842145,
             0.04867273393167054, 0.1972857998888489, 0.2244731723112284,
             0.00040887694942707355, 0.00026014656311027184, 0.013414894061397202,
             0.11233644033448399, 0.0604743621201116, 0.03342440909106323,
             0.07034954287279171, 0.013414894061397202, 0.0005034445044981896,
             0.003003019063204115, 0.00523452475348268, 0.0001621246653926362, 0.05,
             0.01882393038479881, 0.10822447250466571, 0.11100155140762302,
             0.23276849033778024, 0.02742352667080502, 0.1529564256640314,
             0.04740180533190617, 0.18166001933455683, 0.2039781842753333]

    rms = [SD(), MAD(), SSD(), FLPM(), SLPM(), WR(), CVaR(), EVaR(), RVaR(), MDD(), ADD(),
           CDaR(), UCI(), EDaR(), RDaR(), Kurt(), SKurt(), GMD(), RG(), RCVaR(), TG(),
           RTG(), OWA(), DVar(), Skew(), SSkew(), Variance(), Equal(), VaR(), DaR(),
           DaR_r(), MDD_r(), ADD_r(), CDaR_r(), UCI_r(), EDaR_r(), RDaR_r()]

    for (risk, rm) ∈ zip(risks, rms)
        @test isapprox(risk, calc_risk(portfolio; type = :Trad, rm = rm), rtol = 5e-7)
        @test length(rm) == 1
        @test rm[rand(Int)] == rm
    end
end

@testset "Aux functions" begin
    rm = [SD(), [FLPM()], [SLPM(), WR()]]

    rm1 = PortfolioOptimiser.get_first_rm(rm)
    rm2 = PortfolioOptimiser.get_rm_string(rm)
    rm3 = PortfolioOptimiser.get_rm_string(WR())
    rm4 = PortfolioOptimiser.get_rm_string(SD())

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

    rm = RVaR()
    @test_throws AssertionError rm.alpha = 1
    @test_throws AssertionError rm.alpha = 0
    @test_throws AssertionError rm.kappa = 1
    @test_throws AssertionError rm.kappa = 0
    rm.alpha = 0.5
    @test rm.alpha == 0.5
    rm.kappa = 0.5
    @test rm.kappa == 0.5
    @test Symbol(rm) == :RVaR
    @test String(rm) == "RVaR"

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

    rm = RDaR()
    @test_throws AssertionError rm.alpha = 1
    @test_throws AssertionError rm.alpha = 0
    @test_throws AssertionError rm.kappa = 1
    @test_throws AssertionError rm.kappa = 0
    rm.alpha = 0.5
    @test rm.alpha == 0.5
    rm.kappa = 0.5
    @test rm.kappa == 0.5
    @test Symbol(rm) == :RDaR
    @test String(rm) == "RDaR"

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

    rm = RDaR_r()
    @test_throws AssertionError rm.alpha = 1
    @test_throws AssertionError rm.alpha = 0
    @test_throws AssertionError rm.kappa = 1
    @test_throws AssertionError rm.kappa = 0
    rm.alpha = 0.5
    @test rm.alpha == 0.5
    rm.kappa = 0.5
    @test rm.kappa == 0.5

    rm = RCVaR()
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

    rm = RTG()
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
end
