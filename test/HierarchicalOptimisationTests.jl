using Test,
    PortfolioOptimiser,
    DataFrames,
    TimeSeries,
    CSV,
    Dates,
    ECOS,
    SCS,
    Clarabel,
    COSMO,
    OrderedCollections,
    LinearAlgebra,
    StatsBase

A = TimeArray(CSV.File("./assets/stock_prices.csv"), timestamp = :date)
Y = percentchange(A)
returns = dropmissing!(DataFrame(Y))
rf = 1.0329^(1 / 252) - 1
l = 2.0

println("Hierarchical optimisation tests...")

@testset "HRP" begin
    println("HRP tests...")

    portfolio = HCPortfolio(
        returns = returns,
        solvers = OrderedDict(
            :SCS => Dict(:solver => SCS.Optimizer, :params => Dict("verbose" => 0)),
            :Clarabel => Dict(
                :solver => (Clarabel.Optimizer),
                :params => Dict("verbose" => false, "max_step_fraction" => 0.75),
            ),
            :COSMO =>
                Dict(:solver => COSMO.Optimizer, :params => Dict("verbose" => false)),
            :ECOS => Dict(
                :solver => ECOS.Optimizer,
                :params => Dict("verbose" => false, "maxit" => 500),
            ),
        ),
    )
    asset_statistics!(portfolio)

    type = :hrp
    rm = :mv
    obj = :min_risk
    kelly = :none
    linkage = :dbht
    branchorder = :default

    w1 = opt_port!(
        portfolio;
        type = type,
        rm = rm,
        obj = obj,
        kelly = kelly,
        rf = rf,
        l = l,
        linkage = linkage,
        branchorder = branchorder,
    )

    w1t = [
        0.05508878886665077,
        0.07921980553056827,
        0.030128391561595795,
        0.028872972320853722,
        0.033334402615699414,
        0.06358468477746226,
        0.010166844125574337,
        0.06322418256310877,
        0.03881462357698402,
        0.06553219418190419,
        0.10808979041234959,
        0.015170184483397027,
        0.008202518714882668,
        0.07891942014031887,
        0.01571166898114577,
        0.027326705729654836,
        0.08348923079358755,
        0.08473133008827005,
        0.06130631600043508,
        0.04908594453555704,
    ]

    @test isapprox(w1t, w1.weights)
end
