
# # Simple Mean Variance Optimisation

# This is a minimal working example of `PortfolioOptimiser.jl`. We use only the default keyword arguments for all functions. In later examples we will explore more of `PortfolioOptimiser.jl`'s functionality.

#=
!!! note
    `PortfolioOptimiser.jl` currently uses `Plots.jl`, where some recipes are a bit janky (particularly graph plots). We'll be looking to switch to `Makie.jl` in the future, once the copatibility issues resolve.
=#

using PortfolioOptimiser, PrettyTables, TimeSeries, DataFrames, CSV, Clarabel

## This is a helper function for displaying tables 
## with numbers as percentages with 3 decimal points.
fmt = (v, i, j) -> begin
    if j == 1
        return v
    else
        return isa(v, Number) ? "$(round(v*100, digits=3)) %" : v
    end
end;

# ## Creating a [`Portfolio`]() instance

# We can create an instance of [`Portfolio`]() or [`HCPortfolio`]() by calling its keyword constructor. This is a minimum viable example. There are many other keyword arguments that fine-tune the portfolio. Alternatively, you can directly modify the instance's fields. Many are guarded by assertions to ensure correctness, some are immutable for the same reason.

# We can directly provide a `TimeArray` of price data to the constructor. Which computes the return data as follows.

prices = TimeArray(CSV.File(joinpath(@__DIR__, "stock_prices.csv")); timestamp = :date)
returns = dropmissing!(DataFrame(percentchange(prices)))
pretty_table(returns[1:5, :]; formatters = fmt)

# The advantage of using pricing information over returns is that all `missing` data is dropped, which ensures the statistics are well-behaved. It also allows the function to automatically find the latest pricing information, which is needed for discretely allocating portfolios according to available funds and stock prices.

portfolio = Portfolio(;
                      ## Prices TimeArray, the returns are internally computed.
                      prices = prices,
                      ## We need to provide solvers and solver-specific options.
                      solvers = Dict(
                                     ## We will use the Clarabel.jl optimiser. In this case we use a dictionary
                                     ## for the value, but we can also use named tuples, all we need are key-value
                                     ## pairs.
                                     :Clarabel => Dict(
                                                       ## :solver key must contain the optimiser recipe.
                                                       ## Can also call JuMP.optimizer_with_attributes()
                                                       ## https://jump.dev/JuMP.jl/stable/api/JuMP/#JuMP.optimizer_with_attributes 
                                                       ## to add the attributes directly to the solver.
                                                       ## An equivalent configuration using this approach would be:
                                                       ## :solver => JuMP.optimizer_with_attributes(
                                                       ##               Clarabel.Optimizer, 
                                                       ##               "verbose" => false, "max_step_fraction" => 0.75
                                                       ##            )
                                                       :solver => Clarabel.Optimizer,
                                                       ## :params key is optional, but if it is present, it 
                                                       ## defines solver-specific attributes/configurations. 
                                                       ## This often needs to be a dictionary as the solver 
                                                       ## attributes are usually strings.
                                                       :params => Dict("verbose" => false,
                                                                       "max_step_fraction" => 0.75))));

# We can show how the `prices` `TimeArray` is used to compute the returns series, which is decomposed into a vector of timestamps, and the returns `Matrix`. We can check this is the case by reconstructing the `returns` `DataFrame` form above.

returns == hcat(DataFrame(; timestamp = portfolio.timestamps),
                DataFrame([portfolio.returns[:, i] for i ∈ axes(portfolio.returns, 2)],
                          portfolio.assets))

# Another nice thing about [`Portfolio()`]() and [`HCPortfolio()`]() is that the asset tickers and timestamps can be obtained from either a `TimeArray` with price information, or `DataFrame` with returns information. Since we used pricing data, we can obtain the latest prices too.

pretty_table(DataFrame(; assets = portfolio.assets,
                       latest_prices = portfolio.latest_prices))

# ## Optimal Risk-adjusted Return Ratio

# For some risk measures/constraints, we need to compute some statistical quantities. Since we're going to showcase a mean-variance optimisation, we need to estimate the asset mean returns and covariance. We can do this by calling [`asset_statistics!`](). This function also has myriad keyword options, but we'll stick to the basics.

asset_statistics!(portfolio; calc_kurt = false, calc_skew = false)

# We can then call [`optimise!`]() with default arguments, which optimises for the risk adjusted return ratio of the mean variance portfolio.

w = optimise!(portfolio)
pretty_table(w; formatters = fmt)

# ### Informative Plots

# There are a number of plots we can create from an optimised portfolio. For example we can plot the composition as a bar chart.

fig1 = plot_bar(portfolio)

# We can also plot various expected ranges of returns of this portfolio.

fig2 = plot_range(portfolio)

# We can view downside risk measures as well.

fig3 = plot_hist(portfolio)

# And we can view cumulative uncompounded drawdowns too.

fig4 = plot_drawdown(portfolio)

# ## Efficient Frontier

# We can also efficiently compute the asset weights of the portfolio's efficient frontier. This can be done manually but having a dedicated function is nice. We compute 50 linearly distributed points along the frontier, plus the point that maximises the risk adjusted return ratio in the final column.

frontier = efficient_frontier!(portfolio; points = 50)
pretty_table(frontier[:weights]; formatters = fmt)

# ### Informative Plots

# We also have informative plots for the efficient frontier. Such as the frontier itself.

fig5 = plot_frontier(portfolio)

# And the area plot of assets accross the points in the frontier.
fig6 = plot_frontier_area(portfolio)

# ## Asset Clusters

# Since we also provide various hierarchical optimisation methods we can use some of this machinery to showcase how assets relate to one another via hierarchical clustering. 

# And plot the clusters defined by it, we use Ward's linkage function because it gives the best cluster separation.
cluster_opt = ClusterOpt(; linkage = :ward);

fig7 = plot_clusters(portfolio; cluster_opt = cluster_opt)

# There's also a function to plot the dendrogram, but it's not as interesting.

fig8 = plot_dendrogram(portfolio; cluster_opt = cluster_opt)

# ## Asset and Cluster Networks

# We can also visualise the asset network.

fig9 = plot_network(portfolio; cluster_opt = cluster_opt, kwargs = (; method = :stress))

# And the cluster network.

fig10 = plot_cluster_network(portfolio; cluster_opt = cluster_opt,
                             kwargs = (; method = :stress))

# By default, all nodes are the same size, but we can scale them according to their allocation. We will show this in a later example.
