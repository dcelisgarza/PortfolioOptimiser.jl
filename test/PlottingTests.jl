using Test, PortfolioOptimiser, DataFrames, TimeSeries, CSV, Dates, Clarabel, LinearAlgebra,
      StatsPlots, GraphRecipes

prices = TimeArray(CSV.File("./test/assets/stock_prices.csv"); timestamp = :date)

@testset "Plotting" begin
    portfolio = Portfolio(; prices = prices,
                          solvers = Dict(:Clarabel => Dict(:solver => (Clarabel.Optimizer),
                                                           :params => Dict("verbose" => false,
                                                                           "max_step_fraction" => 0.75))))
    asset_statistics!(portfolio)
    rm = SD()
    w = optimise!(portfolio; type = RP())
    plt1 = plot_returns(portfolio, :RP; per_asset = true)
    plt1 = plot_returns(portfolio, :RP; per_asset = false)
    plt3 = plot_bar(portfolio, :RP)
    plt4 = plot_risk_contribution(portfolio, :RP; rm = rm, percentage = true)
    plt5 = plot_risk_contribution(portfolio, :RP; rm = rm, percentage = false)
    frontier = efficient_frontier!(portfolio; rm = rm)
    plt6 = plot_frontier(portfolio; rm = rm)
    plt7 = plot_frontier_area(portfolio; rm = rm)
    plt8 = plot_drawdown(portfolio, :RP)
    plt9 = plot_hist(portfolio, :RP)
    plt10 = plot_range(portfolio, :RP)

    hcportfolio = HCPortfolio(; prices = prices,
                              solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                               :params => Dict("verbose" => false,
                                                                               "max_step_fraction" => 0.75))),)
    asset_statistics!(hcportfolio)
    cluster_assets!(hcportfolio; hclust_alg = DBHT(),
                    hclust_opt = HCType(; k_method = StdSilhouette()))
    plt11 = plot_clusters(hcportfolio; cluster = false)
    plt12 = plot_dendrogram(hcportfolio; cluster = false)

    plt13 = plot_clusters(hcportfolio; cluster = true)
    plt14 = plot_dendrogram(hcportfolio; cluster = true)

    # plt11 = plot_clusters(hcportfolio)
    # plt13 = plot_dendrogram(hcportfolio;
    #                         cluster_opt = ClusterOpt(; linkage = :DBHT,
    #                                                  branchorder = :optimal,
    #                                                  dbht_method = :Unique))

    # plt17 = plot_network(portfolio)
    # plt18 = plot_cluster_network(portfolio)
end
