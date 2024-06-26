using Test, PortfolioOptimiser, DataFrames, TimeSeries, CSV, Dates, Clarabel, LinearAlgebra,
      StatsPlots, GraphRecipes

prices = TimeArray(CSV.File("./assets/stock_prices.csv"); timestamp = :date)

@testset "Plotting" begin
    portfolio = Portfolio(; prices = prices,
                          solvers = Dict(:Clarabel => Dict(:solver => (Clarabel.Optimizer),
                                                           :params => Dict("verbose" => false,
                                                                           "max_step_fraction" => 0.75))))
    asset_statistics!(portfolio)
    rm = :SD
    obj = :Min_Risk
    w = optimise!(portfolio, OptimiseOpt(; rm = rm, obj = obj); save_opt_params = true)
    plt1 = plot_risk_contribution(portfolio; rm = rm, percentage = true)
    plt2 = plot_risk_contribution(portfolio; rm = rm, percentage = false)
    frontier = efficient_frontier!(portfolio, OptimiseOpt(; rm = rm))
    plt3 = plot_frontier(portfolio; rm = rm)
    plt4 = plot_frontier_area(portfolio; rm = rm)
    plt5 = plot_frontier_area(frontier; rm = rm)
    plt6 = plot_drawdown(portfolio)
    plt7 = plot_hist(portfolio)
    plt8 = plot_range(portfolio)
    plt9 = plot_returns(portfolio)
    plt10 = plot_returns(portfolio; per_asset = true)
    plt11 = plot_bar(portfolio)

    hcportfolio = HCPortfolio(; prices = prices,
                              solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                               :params => Dict("verbose" => false,
                                                                               "max_step_fraction" => 0.75))),)
    asset_statistics!(hcportfolio; calc_kurt = false)
    plt12 = plot_clusters(hcportfolio;
                          cluster_opt = ClusterOpt(; linkage = :DBHT, branchorder = :r,
                                                   dbht_method = :Unique))
    plt13 = plot_dendrogram(hcportfolio;
                            cluster_opt = ClusterOpt(; linkage = :DBHT,
                                                     branchorder = :optimal,
                                                     dbht_method = :Unique))
    optimise!(hcportfolio; type = :HERC,
              cluster_opt = ClusterOpt(; linkage = :DBHT, branchorder = :optimal,
                                       dbht_method = :Unique))
    plt14 = plot_clusters(hcportfolio; cluster = false)
    plt15 = plot_clusters(hcportfolio.assets, hcportfolio.returns)
    plt16 = plot_clusters(hcportfolio.assets, hcportfolio.returns; linkage = :DBHT)
    plt17 = plot_network(portfolio)
    plt18 = plot_cluster_network(portfolio)
end
