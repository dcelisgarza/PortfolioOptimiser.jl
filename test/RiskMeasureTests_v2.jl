using CSV, TimeSeries, StatsBase, Statistics, LinearAlgebra, Test, Clarabel,
      PortfolioOptimiser

prices = TimeArray(CSV.File("./test/assets/stock_prices.csv"); timestamp = :date)

@testset "Risk measures v2" begin
    portfolio = Portfolio2(; prices = prices,
                           solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                            :params => Dict("verbose" => false))))
    asset_statistics2!(portfolio)
    optimise!(portfolio)

    risks = [0.012732818438689692, 0.009054428305486948, 0.008974573704221727,
             0.003858879045730223, 0.008278637116930432, 0.04767699262536566,
             0.029180900700100272, 0.035143848572313445, 0.03997681539353343,
             0.2582974717056197, 0.027337478660655532, 0.1608684430842145,
             0.04867273393167054, 0.1972857998888489, 0.2244731723112284,
             0.00040887694942707355, 0.00026014656311027184, 0.013414894061397202,
             0.11233644033448399, 0.0604743621201116, 0.03342440909106323,
             0.07034954287279171, 0.013414894061397202, 0.0005034445044981896,
             0.003003019063204115, 0.00523452475348268, 0.0001621246653926362, 0.05,
             0.01882393038479881, 0.10822447250466571, 0.11100155140762302,
             0.23276849033778024, 0.02742352667080502, 0.1529564256640314,
             0.04740180533190617, 0.18166001933455683, 0.2039781842753333]

    rms = [SD2(), MAD2(), SSD2(), FLPM2(), SLPM2(), WR2(), CVaR2(), EVaR2(), RVaR2(),
           MDD2(), ADD2(), CDaR2(), UCI2(), EDaR2(), RDaR2(), Kurt2(), SKurt2(), GMD2(),
           RG2(), RCVaR2(), TG2(), RTG2(), OWA2(), DVar2(), Skew2(), SSkew2(), Variance2(),
           Equal2(), VaR2(), DaR2(), DaR_r2(), MDD_r2(), ADD_r2(), CDaR_r2(), UCI_r2(),
           EDaR_r2(), RDaR_r2()]

    for (risk, rm) ∈ zip(risks, rms)
        @test isapprox(risk, calc_risk(portfolio; type = :Trad, rm = rm))
    end
end
