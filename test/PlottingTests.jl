using Test, PortfolioOptimiser, DataFrames, TimeSeries, CSV, Dates, Clarabel, LinearAlgebra,
      StatsPlots, GraphRecipes

path = joinpath(@__DIR__, "assets/stock_prices.csv")
prices = TimeArray(CSV.File(path); timestamp = :date)

@testset "Plotting" begin
    portfolio = Portfolio(; prices = prices,
                          solvers = Dict(:Clarabel => Dict(:solver => (Clarabel.Optimizer),
                                                           :params => Dict("verbose" => false,
                                                                           "max_step_fraction" => 0.75))))
    asset_statistics!(portfolio)
    rm = Variance()
    w = optimise!(portfolio, RB())
    plt1 = plot_returns(portfolio; type = :RB, per_asset = true)
    @test plt1.n == 20
    plt2 = plot_returns(portfolio; type = :RB, per_asset = false)
    @test plt2.n == 1
    plt3 = plot_bar(portfolio; type = :RB)
    @test plt3.n == 1
    plt4 = plot_risk_contribution(portfolio; type = :RB, rm = rm, percentage = true)
    @test plt4.n == 2
    plt5 = plot_risk_contribution(portfolio; type = :RB, rm = rm, percentage = false)
    @test plt5.n == 2
    frontier = efficient_frontier!(portfolio, Trad(; rm = rm))
    plt6 = plot_frontier(portfolio; rm = rm)
    @test plt6.n == 2
    plt7 = plot_frontier_area(portfolio; rm = rm)
    @test plt7.n == 21
    plt8 = plot_drawdown(portfolio; type = :RB)
    @test plt8.n == 9
    plt9 = plot_hist(portfolio; type = :RB)
    @test plt9.n == 12
    plt10 = plot_range(portfolio; type = :RB)
    @test plt10.n == 4

    cluster_assets!(portfolio; clust_alg = HAC(),
                    clust_opt = ClustOpt(; k_method = StdSilhouette()))
    w = optimise!(portfolio, HRP())

    plt11 = plot_clusters(portfolio; cluster = false)
    @test plt11.n == 51
    plt12 = plot_dendrogram(portfolio; cluster = false)
    @test plt12.n == 23
    plt13 = plot_clusters(portfolio; cluster = true)
    @test plt13.n == 48
    plt14 = plot_dendrogram(portfolio; cluster = true)
    @test plt14.n == 22
    plt15 = plot_network(portfolio; type = :HRP, cluster = true)
    @test plt15.n == 40
    plt16 = plot_network(portfolio; type = :HRP, cluster = false)
    @test plt16.n == 40
    plt17 = plot_clusters(portfolio; cluster = false)
    @test plt17.n == 51
    plt18 = plot_dendrogram(portfolio; cluster = false)
    @test plt18.n == 23
    plt19 = plot_clusters(portfolio; cluster = true, clust_alg = DBHT())
    @test plt19.n == 48
    plt20 = plot_dendrogram(portfolio; cluster = true, clust_alg = DBHT())
    @test plt20.n == 22
    ptl21 = plot_network(portfolio; type = :RB, cluster = true, network_type = TMFG())
    @test ptl21.n == 75
    ptl22 = plot_network(portfolio; type = :RB, cluster = false, network_type = TMFG())
    @test ptl22.n == 75
end
