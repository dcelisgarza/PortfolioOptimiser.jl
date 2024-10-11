The source files for all examples can be found in [/examples](https://github.com/dcelisgarza/PortfolioOptimiser.jl/tree/main/examples/).

```@meta
EditURL = "../../../examples/7_worst_case_statistics.jl"
```

# Example 7: Worst case statistics

This example follows from previous ones. If something in the preamble is confusing, it is explained there.

This example focuses on the [`wc_statistics!`](@ref) used in the [`WC`](@ref) optimisation type of [`Portfolio`](@ref).

## 7.1 Downloading the data

````@example 7_worst_case_statistics
# using Pkg
# Pkg.add.(["StatsPlots", "GraphRecipes", "YFinance", "Clarabel", "HiGHS", "CovarianceEstimation", "SparseArrays"])
using Clarabel, CovarianceEstimation, DataFrames, Dates, GraphRecipes, HiGHS, YFinance,
      PortfolioOptimiser, Statistics, StatsBase, StatsPlots, TimeSeries, LinearAlgebra,
      PrettyTables, Random

fmt1 = (v, i, j) -> begin
    if j == 1
        return v
    else
        return if isa(v, Number)
            "$(round(v*100, digits=3)) %"
        else
            v
        end
    end
end;

function stock_price_to_time_array(x)
    coln = collect(keys(x))[3:end] # only get the keys that are not ticker or datetime
    m = hcat([x[k] for k ∈ coln]...) #Convert the dictionary into a matrix
    return TimeArray(x["timestamp"], m, Symbol.(coln), x["ticker"])
end
assets = ["AAL", "AAPL", "AMC", "BB", "BBY", "DELL", "DG", "DRS", "GME", "INTC", "LULU",
          "MARA", "MCI", "MSFT", "NKLA", "NVAX", "NVDA", "PARA", "PLNT", "SAVE", "SBUX",
          "SIRI", "STX", "TLRY", "TSLA"]
Date_0 = "2019-01-01"
Date_1 = "2023-01-01"
prices = get_prices.(assets; startdt = Date_0, enddt = Date_1)
prices = stock_price_to_time_array.(prices)
prices = hcat(prices...)
cidx = colnames(prices)[occursin.(r"adj", string.(colnames(prices)))]
prices = prices[cidx]
TimeSeries.rename!(prices, Symbol.(assets));
nothing #hide
````

## 7.2 Instantiating an instance of [`Portfolio`](@ref).

We'll compute basic statistics for this.

````@example 7_worst_case_statistics
portfolio = Portfolio(; prices = prices,
                      # Continuous optimiser.
                      solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                       :check_sol => (allow_local = true,
                                                                      allow_almost = true),
                                                       :params => Dict("verbose" => false))),
                      # MIP optimiser for the discrete allocation.
                      alloc_solvers = Dict(:HiGHS => Dict(:solver => HiGHS.Optimizer,
                                                          :check_sol => (allow_local = true,
                                                                         allow_almost = true),
                                                          :params => Dict("log_to_console" => false))));

asset_statistics!(portfolio)
````

## 7.3 Effect of the Worst Case Mean Variance statistics

The previous tutorial showed how to perform worst case mean variance optimisations. This one goes into more detail on computing the uncertainty sets needed for this optimisation type.

The function in charge of doing so is [`wc_statistics!`](@ref) via the [`WCType`](@ref) type. Consult the docs for details.

There are a lot of combinations for this, so we will not be showing an exhaustive list. We will explore a representative subset. Since we used the default values for our previous tutorial we will explore a few of the other options.

We'll first use the default statistics for computing the optimised worst case mean variance portfolio.

````@example 7_worst_case_statistics
# Set random seed for reproducible results.
Random.seed!(123)
````

Lets compute the default worst case statistics.

````@example 7_worst_case_statistics
wc_statistics!(portfolio)
````

We'll use the box set for the expected returns vector and the elliptical set for the covariance matrix. We'll maximise the risk-adjusted return ratio.

````@example 7_worst_case_statistics
type = WC(; mu = Box(), cov = Ellipse())
obj = Sharpe(3.5 / 100 / 252)
w1 = optimise!(portfolio; type = type, obj = obj);
nothing #hide
````

[`WCType`](@ref) can produce a wealth of uncertainty sets depending on the user provided parameters. You can experiment by changing the values of `wc` and computing the statistics again.

We'll now use a completely different set of parameters for computing the worst case statistics, but we will optimise the same problem.

````@example 7_worst_case_statistics
wc = WCType(; cov_type = PortCovCor(; ce = CorGerber1(; normalise = true)),
            mu_type = MuBOP(), box = NormalWC(), ellipse = ArchWC(), k_sigma = KNormalWC(),
            k_mu = KGeneralWC(), diagonal = false)
wc_statistics!(portfolio, wc)
w2 = optimise!(portfolio; type = type, obj = obj)

pretty_table(DataFrame(; tickers = w1.tickers, w1 = w1.weights, w2 = w2.weights);
             formatters = fmt1)
````

When compared to the previous tutorial, the takeaway here is that the type of uncertainty set used has much more of an impact on the results of the optimisation [6.4 Optimising the portfolio](@ref) than the parameters used to compute the worst case sets. However, more robust statistics will produce more robust uncertainty sets.

* * *

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*
