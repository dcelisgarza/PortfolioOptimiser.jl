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
    plt3 = plot_frontier_area(frontier, rm)
    plt4 = plot_drawdown(portfolio)
    plt5 = plot_hist(portfolio)
    plt6 = plot_range(portfolio)
    plt7 = plot_returns(portfolio)
    plt8 = plot_returns(portfolio; per_asset = true)
    plt9 = plot_bar(portfolio)
end
