using Test, PortfolioOptimiser, DataFrames, TimeSeries, CSV, Dates, Clarabel, LinearAlgebra

@testset "Plotting" begin
    A = TimeArray(CSV.File("./assets/stock_prices.csv"), timestamp = :date)
    Y = percentchange(A)
    returns = dropmissing!(DataFrame(Y))

    portfolio = Portfolio(
        returns = returns,
        solvers = Dict(
            :Clarabel => Dict(
                :solver => (Clarabel.Optimizer),
                :params => Dict("verbose" => false, "max_step_fraction" => 0.75),
            ),
        ),
    )
    asset_statistics!(portfolio)
    rm = :SD
    obj = :Min_Risk
    w = opt_port!(portfolio; rm = rm, obj = obj, save_opt_params = true)
    plt1 = plot_risk_contribution(portfolio; rm = rm, percentage = true)
    plt2 = plot_risk_contribution(portfolio; rm = rm, percentage = false)
    frontier = efficient_frontier(portfolio; rm = rm)
    plt3 = plot_frontier(portfolio; rm = rm)
    plt4 = plot_frontier_area(frontier; rm = rm)
    plt5 = plot_drawdown(portfolio)
    plt6 = plot_hist(portfolio)
    plt7 = plot_range(portfolio)
    plt8 = plot_returns(portfolio)
    plt9 = plot_returns(portfolio; per_asset = true)
    plt10 = plot_bar(portfolio)

    hcportfolio = HCPortfolio(;
        returns = returns,
        solvers = Dict(
            :Clarabel => Dict(
                :solver => Clarabel.Optimizer,
                :params => Dict("verbose" => false, "max_step_fraction" => 0.75),
            ),
        ),
    )
    asset_statistics!(hcportfolio; calc_cov = false, calc_mu = false, calc_kurt = false)
    plt11 = plot_clusters(
        hcportfolio;
        max_k = 10,
        linkage = :DBHT,
        branchorder = :r,
        dbht_method = :Unique,
    )
    plt12 = plot_dendrogram(
        hcportfolio;
        max_k = 10,
        linkage = :DBHT,
        branchorder = :optimal,
        dbht_method = :Unique,
    )
end
