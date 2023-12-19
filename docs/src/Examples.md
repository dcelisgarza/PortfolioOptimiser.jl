# Examples

## Simple Mean-Variance Optimisation

Our first example will showcase a basic example.

First we load the packages that we'll use and define our format function.

```@example 1; continued=true
using PortfolioOptimiser, PrettyTables, TimeSeries, DataFrames, CSV, Clarabel

# Format numbers as percentages up to three decimal points.
fmt = (v, i, j) -> begin
    if j == 1
        return v
    else
        return isa(v, Number) ? "$(round(v*100, digits=3)) %" : v
    end
end
```

We can create an instance of [`Portfolio`](@ref) or [`HCPortfolio`](@ref) by calling its keyword constructor. This is a minimum viable example. There are many other keyword arguments that fine-tune the portfolio. Alternatively, you can directly modify the instance's fields. Many are guarded by assertions to ensure correctness, some are immutable for the same reason.

We can directly provide a `TimeArray` of price data to the constructor. Which computes the return data as follows.

```@example 1
prices = TimeArray(CSV.File("../../test/assets/stock_prices.csv"); timestamp=:date)
returns = dropmissing!(DataFrame(percentchange(prices)))
pretty_table(returns[1:5, :]; formatters = fmt)
```

The advantage of using pricing information over returns is that all `missing` data is dropped, which ensures the statistics are well-behaved. It also allows the function to automatically find the latest pricing information, which is needed for discretely allocating portfolios according to available funds and stock prices.

```@example 1
portfolio = Portfolio(;
    # Prices TimeArray, the returns are internally computed.
    prices = prices,
    # We need to provide solvers and solver-specific options.
    solvers = Dict(
        # We will use the Clarabel.jl optimiser. In this case we use a dictionary
        # for the value, but we can also use named tuples, all we need are key-value
        # pairs.
        :Clarabel => Dict(
            # :solver key must contain the optimiser recipe.
            # Can also call JuMP.optimizer_with_attributes()
            # https://jump.dev/JuMP.jl/stable/api/JuMP/#JuMP.optimizer_with_attributes 
            # to add the attributes directly to the solver.
            # An equivalent configuration using this approach would be:
            # :solver => JuMP.optimizer_with_attributes(
            #               Clarabel.Optimizer, 
            #               "verbose" => false, "max_step_fraction" => 0.75
            #            )
            :solver => Clarabel.Optimizer,
            # :params key is optional, but if it is present, it defines solver-specific
            # attributes/configurations. This often needs to be a dictionary as the 
            # solver attributes are usually strings.
            :params => Dict("verbose" => false, "max_step_fraction" => 0.75),
        ),
    ),
)
```

We can show how the `prices` `TimeArray` is used to compute the returns series, which is decomposed into timestamps and the returns `Matrix`.

```@example 1
pretty_table([portfolio.timestamps[1:5] portfolio.returns[1:5, :]]; formatters = fmt)
```

Another nice thing about [`Portfolio()`](@ref) and [`HCPortfolio()`](@ref) is that the asset tickers and timestamps can be obtained from either a `TimeArray` with price information, or `DataFrame` with returns information. Since we used pricing data, we can obtain the latest prices too.

```@example 1
[portfolio.assets portfolio.latest_prices]
```

For some risk measures/constraints, we need to compute some statistical quantities. Since we're going to showcase a mean-variance optimisation, we need to estimate the asset mean returns and covariance. We can do this by calling [`asset_statistics!`](@ref). This function also has myriad keyword options, but we'll stick to the basics.

```@example 1
asset_statistics!(portfolio)
```

We can then call [`opt_port!`](@ref) with default arguments, which optimises for the risk adjusted return ratio of the mean variance portfolio.

```@example 1
w = opt_port!(portfolio)
pretty_table(w; formatters = fmt)
```

```@example 1
fig1 = plot_bar(portfolio)
```

```@example 1
fig2 = plot_range(portfolio)
```

```@example 1
fig3 = plot_hist(portfolio)
```

```@example 1
fig4 = plot_drawdown(portfolio)
```

```@example 1
frontier = efficient_frontier!(portfolio; points = 50)
frontier[:SD]
pretty_table(frontier[:SD][:weights]; formatters=fmt)
```

```@example 1
fig5 = plot_frontier(portfolio)
```

```@example 1
fig6 = plot_frontier_area(portfolio)
```

```@example 1
# We instantiate a heirarchical portfolio instance.
hcportfolio = HCPortfolio(;
    # Returns dataframe
    prices = prices,
)
# We can then compute various statistics required for clustering.
asset_statistics!(hcportfolio; calc_kurt = false)
```

```@example 1
fig7 = plot_clusters(hcportfolio; linkage = :ward)
```

```@example 1
fig8 = plot_dendrogram(hcportfolio; linkage = :ward)
```
