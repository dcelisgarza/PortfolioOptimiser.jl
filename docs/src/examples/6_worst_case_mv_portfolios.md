The source files for all examples can be found in [/examples](https://github.com/dcelisgarza/PortfolioOptimiser.jl/tree/main/examples/).

```@meta
EditURL = "../../../examples/6_worst_case_mv_portfolios.jl"
```

# Example 6: Worst case Mean-Variance

This example follows from previous ones. If something in the preamble is confusing, it is explained there.

This example focuses on the [`WC`](@ref) optimisation type of [`Portfolio`](@ref).

## 6.1 Downloading the data

````@example 6_worst_case_mv_portfolios
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

## 6.2 Instantiating an instance of [`Portfolio`](@ref).

We'll compute basic statistics for this.

````@example 6_worst_case_mv_portfolios
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

## 6.3 Worst case statistics

In order to perform a worst case mean variance optimisation we need to compute uncertainty sets for the expected returns vector and covariance matrix. We can do this via [`wc_statistics!`](@ref).

For the purposes of this tutorial we'll use the defaults. We will explore the other options one can sue for computing the uncertainty sets in a subsequent tutorial.

````@example 6_worst_case_mv_portfolios
# Set random seed for reproducible results.
Random.seed!(123)
wc_statistics!(portfolio)
````

## 6.4 Optimising the portfolio

Having computed our worst case statistics, we can optimise the portfolio. The [`WC`](@ref) struct defines which set types to use in the worst case mean variance optimisation. [`WC`](@ref) defaults to using [`Box`](@ref) constraints for both the expected returns vector and covariance matrix.

User-provided risk measures have no effect on this type of optimisation will only perform a mean variance optimisation with uncertainty sets.

This type of optimisation can take any [`PortfolioOptimiser.ObjectiveFunction`](@ref).

````@example 6_worst_case_mv_portfolios
# User-provided risk measures have no effect.
rm = CVaR()
# Worst case mean-variance optimisation using default set types.
type = WC()
# We'll maximise the risk-adjusted return ratio.
obj = Sharpe(; rf = 3.5 / 100 / 252)

# Box uncertainty set for the expected returns vector and covariance matrix.
w1 = optimise!(portfolio; type = type, rm = rm, obj = obj)

# Ellipse uncertainty set for the expected returns vector and box uncertainty set for the covariance matrix.
type.mu = Ellipse()
w2 = optimise!(portfolio; type = type, rm = rm, obj = obj)

# Box uncertainty set for the expected returns vector and ellipse uncertainty set for the covariance matrix.
type.mu = Box()
type.cov = Ellipse()
w3 = optimise!(portfolio; type = type, rm = rm, obj = obj)

# Ellipse uncertainty set for the expected returns vector and ellipse uncertainty set for the covariance matrix.
type.mu = Ellipse()
w4 = optimise!(portfolio; type = type, rm = rm, obj = obj)

pretty_table(DataFrame(; tickers = w1.tickers, box_box = w1.weights, ellip_box = w2.weights,
                       box_ellip = w3.weights, ellip_ellip = w4.weights); formatters = fmt1)
````

As you can see, the type of constraint used can have a large impact on the results of the optimisation. This is accentuated by the fact that we maximised the risk-adjusted return ratio. We'll now minimise the risk, on which the uncertainty set for the expected returns vector has a smaller impact.

````@example 6_worst_case_mv_portfolios
type = WC()
# We'll maximise the risk-adjusted return ratio.
obj = MinRisk()

# Box uncertainty set for the expected returns vector and covariance matrix.
w5 = optimise!(portfolio; type = type, rm = rm, obj = obj)

# Ellipse uncertainty set for the expected returns vector and box uncertainty set for the covariance matrix.
type.mu = Ellipse()
w6 = optimise!(portfolio; type = type, rm = rm, obj = obj)

# Box uncertainty set for the expected returns vector and ellipse uncertainty set for the covariance matrix.
type.mu = Box()
type.cov = Ellipse()
w7 = optimise!(portfolio; type = type, rm = rm, obj = obj)

# Ellipse uncertainty set for the expected returns vector and ellipse uncertainty set for the covariance matrix.
type.mu = Ellipse()
w8 = optimise!(portfolio; type = type, rm = rm, obj = obj)

pretty_table(DataFrame(; tickers = w5.tickers, box_box = w5.weights, ellip_box = w6.weights,
                       box_ellip = w7.weights, ellip_ellip = w8.weights); formatters = fmt1)
````

It's also posible to disable the worst set constraint for the expected returns vector and covariance matrix independently. We'll disable them both and see that we recover the traditional mean variance optimisation.

````@example 6_worst_case_mv_portfolios
type = WC(; mu = NoWC(), cov = NoWC())

obj = MinRisk()
w9 = optimise!(portfolio; type = type, obj = obj)
w10 = optimise!(portfolio; type = Trad(), obj = obj)

obj = Sharpe(; rf = 3.5 / 100 / 252)
w11 = optimise!(portfolio; type = type, obj = obj)
w12 = optimise!(portfolio; type = Trad(), obj = obj)

pretty_table(DataFrame(; tickers = w9.tickers, nowc_risk = w9.weights,
                       trad_risk = w10.weights, nowc_sharpe = w11.weights,
                       trad_sharpe = w12.weights); formatters = fmt1)
````

We don't recover the weights of the sharpe ratio exactly because one of the risk-adjusted return constraints is slightly relaxed with respect to the traditional optimisation.

* * *

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*
