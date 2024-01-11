using Test, PortfolioOptimiser, DataFrames, TimeSeries, CSV, Dates, Clarabel, LinearAlgebra

@testset "Plotting" begin
    A = TimeArray(CSV.File("./assets/stock_prices.csv"); timestamp = :date)
    Y = percentchange(A)
    returns = dropmissing!(DataFrame(Y))

    portfolio = Portfolio(; returns = returns,
                          solvers = Dict(:Clarabel => Dict(:solver => (Clarabel.Optimizer),
                                                           :params => Dict("verbose" => false,
                                                                           "max_step_fraction" => 0.75))))
    asset_statistics!(portfolio)
    rm = :SD
    obj = :Min_Risk
    w = opt_port!(portfolio; rm = rm, obj = obj, save_opt_params = true)
    plt1 = plot_risk_contribution(portfolio; rm = rm, percentage = true)
    plt2 = plot_risk_contribution(portfolio; rm = rm, percentage = false)
    frontier = efficient_frontier!(portfolio; rm = rm)
    plt3 = plot_frontier(portfolio; rm = rm)
    plt4 = plot_frontier_area(portfolio; rm = rm)
    plt5 = plot_frontier_area(frontier; rm = rm)
    plt6 = plot_drawdown(portfolio)
    plt7 = plot_hist(portfolio)
    plt8 = plot_range(portfolio)
    plt9 = plot_returns(portfolio)
    plt10 = plot_returns(portfolio; per_asset = true)
    plt11 = plot_bar(portfolio)

    hcportfolio = HCPortfolio(; returns = returns,
                              solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                               :params => Dict("verbose" => false,
                                                                               "max_step_fraction" => 0.75))),)
    asset_statistics!(hcportfolio; calc_kurt = false)
    plt12 = plot_clusters(hcportfolio; max_k = 10, linkage = :DBHT, branchorder = :r,
                          dbht_method = :Unique,)
    plt13 = plot_dendrogram(hcportfolio; max_k = 10, linkage = :DBHT,
                            branchorder = :optimal, dbht_method = :Unique,)
    opt_port!(hcportfolio; type = :HERC)
    plt14 = plot_clusters(hcportfolio; cluster = false)
end
