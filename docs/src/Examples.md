# Example

```@example 1; continued=true
using PortfolioOptimiser, PrettyTables, TimeSeries, DataFrames, CSV, Clarabel

wdffmt = (v, i, j) -> begin
    if j == 1
        return v
    else
        return "$(round(v*100, digits=3)) %"
    end
end

prices = TimeArray(CSV.File("../../test/assets/stock_prices.csv"); timestamp=:date)
returns = dropmissing!(DataFrame(percentchange(prices)))

```

```@example 1
pretty_table(returns[1:5, :]; formatters = wdffmt)
```

```@example 1
portfolio = Portfolio(;
    # Returns dataframe
    returns = returns,
    # Solvers is a Dictionary.
    solvers = Dict(
        # We will use the Clarabel.jl optimiser. In this case we use a dictionary
        # for the value, but we can also use named tuples, all we need are key-value
        # pairs.
        :Clarabel => Dict(
            # :solver key must contain the optimiser structure.
            :solver => Clarabel.Optimizer,
            # :params key is optional, but if it is present, it defines solver-specific
            # attributes/configurations. This often needs to be a dictionary as the 
            # solver attributes are usually strings.
            :params => Dict("verbose" => false, "max_step_fraction" => 0.75),
        ),
    ),
)
```

```@example 1
asset_statistics!(portfolio; calc_kurt = false)
w = opt_port!(portfolio)
pretty_table(w; formatters=wdffmt)
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
pretty_table(frontier[:SD][:weights]; formatters=wdffmt)
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
    returns=returns,
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
