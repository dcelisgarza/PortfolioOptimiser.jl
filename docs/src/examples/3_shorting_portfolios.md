The source files for all examples can be found in [/examples](https://github.com/dcelisgarza/PortfolioOptimiser.jl/tree/main/examples/).

```@meta
EditURL = "../../../examples/3_shorting_portfolios.jl"
```

# Example 3: Shorting and leveraged portfolios

This example follows from previous ones. If something in the preamble is confusing, it is explained there.

This example focuses on using the shorting constraints available to [`Trad`](@ref) and [`WC`](@ref) optimisations of [`Portfolio`](@ref).

## 3.1 Downloading the data

````@example 3_shorting_portfolios
# using Pkg
# Pkg.add.(["StatsPlots", "GraphRecipes", "YFinance", "Clarabel", "HiGHS", "CovarianceEstimation", "SparseArrays"])
using Clarabel, CovarianceEstimation, DataFrames, Dates, GraphRecipes, HiGHS, YFinance,
      PortfolioOptimiser, Statistics, StatsBase, StatsPlots, TimeSeries, LinearAlgebra,
      PrettyTables

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

## 3.2 Instantiating an instance of [`Portfolio`](@ref).

````@example 3_shorting_portfolios
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
nothing #hide
````

````@example 3_shorting_portfolios
mu_type = MuSimple()
cov_type = PortCovCor()
asset_statistics!(portfolio; mu_type = mu_type, cov_type = cov_type)
````

````@example 3_shorting_portfolios
# Risk free rate.
rf = 3.5 / 100 / 252
# Risk aversion.
l = 2.0
# Objective function.
obj = MinRisk()
# Risk measure.
rm = SD()
# Money available to us.
investment = 6750;
nothing #hide
````

## 3.3.1 Long-only portfolio

First we will optimise the portfolio without shorting and plot the weights and the efficient frontier.

````@example 3_shorting_portfolios
portfolio.short = false
portfolio.optimal[:ns] = optimise!(portfolio; rm = rm, obj = obj)
plot_bar(portfolio; type = :ns)
portfolio.frontier[:ns] = efficient_frontier!(portfolio; rm = rm, points = 30)
plot_frontier(portfolio; type = :ns)
````

We'll now allocate the portfolio according to our means. We'll use both allocation methods:

  - Linear Mixed-integer Programming (LP): (default) can only allocate discrete integer shares and requires an MIP solver.
  - Greedy algorithm, can round down to the nearest `integer + N*rounding`, but is not guaranteed to be globally optimal. The rounding also rounds down, as it ensures the investment will not be exceeded.

````@example 3_shorting_portfolios
portfolio.optimal[:nsal] = allocate!(portfolio; type = :ns, method = LP(),
                                     investment = investment)
portfolio.optimal[:nsag] = allocate!(portfolio; type = :ns,
                                     method = Greedy(; rounding = 0.3),
                                     investment = investment);
nothing #hide
````

Lets verify that the allocations used the money we have available. We'll also compare the would-be optimal portfolio.

Optimal portfolio

````@example 3_shorting_portfolios
long_optimal_idx = portfolio.optimal[:ns].weights .>= 0
short_optimal_idx = .!long_optimal_idx
println("Optimal investment = $(sum(investment * portfolio.optimal[:ns].weights[long_optimal_idx]))")
println("Sum of weights = $(sum(portfolio.optimal[:ns].weights[long_optimal_idx]))")
````

LP allocated portfolio

````@example 3_shorting_portfolios
long_LP_idx = portfolio.optimal[:nsal].weights .>= 0
short_LP_idx = .!long_LP_idx
println("Allocation investment = $(dot(portfolio.latest_prices[long_LP_idx], portfolio.optimal[:nsal].shares[long_LP_idx]))")
println("Sum of weights = $(sum(portfolio.optimal[:nsal].weights[long_LP_idx]))")
````

Greedy allocated portfolio

````@example 3_shorting_portfolios
long_Greedy_idx = portfolio.optimal[:nsag].weights .>= 0
short_Greedy_idx = .!long_Greedy_idx
println("Allocation investment = $(dot(portfolio.latest_prices[long_Greedy_idx], portfolio.optimal[:nsag].shares[long_Greedy_idx]))")
println("Sum of weights = $(sum(portfolio.optimal[:nsag].weights[long_Greedy_idx]))")
````

As you can see, the greedy algorithm doesn't make optimal use of the available investment.

Lets now see what the long-only portfolio looks like, in both optimal and allocated form.

````@example 3_shorting_portfolios
pretty_table(DataFrame(; tickers = portfolio.assets,
                       # Optimal weights without shorting.
                       ns_w = portfolio.optimal[:ns].weights,
                       # Discretely allocated optimal weights without shorting.
                       # Linear programming.
                       nsal_w = portfolio.optimal[:nsal].weights,
                       # Discretely allocated shares without shorting.
                       # Linear programming.
                       nsal_s = portfolio.optimal[:nsal].shares,
                       # Discretely allocated optimal weights without shorting.
                       # Greedy algorithm.
                       nsag_w = portfolio.optimal[:nsag].weights,
                       # Discretely allocated shares without shorting.
                       # Greedy algorithm.
                       nsag_s = portfolio.optimal[:nsag].shares))
````

## 3.3.2 Shorting

Enabling shorting is very simple. This will allow negative weights, which correspond to shorting portfolios. It is generally a good idea to start with little to no leverage.

````@example 3_shorting_portfolios
portfolio.short = true;
nothing #hide
````

How short- or long-heavy we want to be is mediated by the `short_lb`, `long_ub`, `short_budget` and `budget` properties. They set the upper bound for the absolute value of the sum of the short and long weights respectively.

  - `budget`: the sum of all the weights will be equal to this value.
  - `short_budget`: upper bound for the absolute value of the sum of the short weights.
  - `long_ub`: upper bound of each of the long weights.
  - `short_lb`: upper bound of the absolute value of each of the short weights.

These values multiply the cash at our disposal when we allocate the portfolio. So when [`allocate!`](@ref) is called, the long investment will be `investment * long_ub`. And if shorting is enabled, the short investment (the amount shorted) will be `short_lb * investment`.

Lets short the market whithout reinvesting the earnings, meaning we'll have a cash reserve in our balance that is equal to the short sale value. You can change this by increasing `long_ub`, if you set it to `1 + short_lb` it means the profits from short selling will be reinvested into the portfolio.

We will use the default values.

````@example 3_shorting_portfolios
# The absolute value of the sum of the short weights is equal to `0.2`.
portfolio.short_budget = 0.2
# The portfolio weights will add up to 0.8, meaning the portfolio will be underleveraged.
portfolio.budget = 0.8;
# Each short position can have a maximum value of -0.2.
portfolio.short_lb = 0.2
# Each long position can have a maximum value of 1.
portfolio.long_ub = 1.0
````

The portfolio `budget` gives us the leverage characteristics of the portfolio. This is a property that is automatically computed and cannot be cahnged. There are verious scenarios that `budget` describes.

  - `budget < 0`: the short sale value of the portfolio is higher than the long-sale value.
  - `budget == 0`: the short and long values of the portfolio are equal. The market neutral portfolio is found by maximising the return given these conditions.
  - `0 < budget < 1`: the portfolio is under-leveraged, meaning there is a cash reserve that is not being used.
  - `budget == 1`: the portfolio has no leverage. If shorting is enabled, this means the profits from shorting are being invested in long positions.
  - `budget > 1`: the portfolio is leveraged, meaning it's using more money than is available.

Here the portfolio is under-leveraged.

Lets optimise the short-long portfolio.

````@example 3_shorting_portfolios
portfolio.optimal[:s] = optimise!(portfolio; rm = rm, obj = obj)
plot_bar(portfolio; type = :s)
portfolio.frontier[:s] = efficient_frontier!(portfolio; rm = rm, points = 30)
plot_frontier(portfolio; type = :s)
````

Lets allocate the short-long portfolio.

````@example 3_shorting_portfolios
# Allocating the short-long portfolio.
portfolio.optimal[:sal] = allocate!(portfolio; type = :s, investment = investment)
portfolio.optimal[:sag] = allocate!(portfolio; type = :s, method = Greedy(; rounding = 0.3),
                                    investment = investment);
nothing #hide
````

Lets verify that the allocations used the money we have available.

Optimal portfolio

````@example 3_shorting_portfolios
long_optimal_idx = portfolio.optimal[:s].weights .>= 0
short_optimal_idx = .!long_optimal_idx
println("Optimal investment")
println("long = $(sum(investment * portfolio.optimal[:s].weights[long_optimal_idx]))")
println("short = $(sum(investment * portfolio.optimal[:s].weights[short_optimal_idx]))")
println("Sum of weights")
println("long = $(sum(portfolio.optimal[:s].weights[long_optimal_idx]))")
println("short = $(sum(portfolio.optimal[:s].weights[short_optimal_idx]))")
````

LP allocated portfolio

````@example 3_shorting_portfolios
long_LP_idx = portfolio.optimal[:sal].weights .>= 0
short_LP_idx = .!long_LP_idx
println("Allocation investment")
println("long = $(dot(portfolio.latest_prices[long_LP_idx], portfolio.optimal[:sal].shares[long_LP_idx]))")
println("short = $(dot(portfolio.latest_prices[short_LP_idx], portfolio.optimal[:sal].shares[short_LP_idx]))")
println("Sum of weights")
println("long = $(sum(portfolio.optimal[:sal].weights[long_LP_idx]))")
println("short = $(sum(portfolio.optimal[:sal].weights[short_LP_idx]))")
````

Greedy allocated portfolio

````@example 3_shorting_portfolios
long_Greedy_idx = portfolio.optimal[:sag].weights .>= 0
short_Greedy_idx = .!long_Greedy_idx
println("Allocation investment")
println("long = $(dot(portfolio.latest_prices[long_Greedy_idx], portfolio.optimal[:sag].shares[long_Greedy_idx]))")
println("short = $(dot(portfolio.latest_prices[short_Greedy_idx], portfolio.optimal[:sag].shares[short_Greedy_idx]))")
println("Sum of weights")
println("long = $(sum(portfolio.optimal[:sag].weights[long_Greedy_idx]))")
println("short = $(sum(portfolio.optimal[:sag].weights[short_Greedy_idx]))")
````

Here's what the short-long portfolio looks like. See how this differs from the long-only portfolio.

````@example 3_shorting_portfolios
pretty_table(DataFrame(; tickers = portfolio.assets,
                       # Optimal weights with shorting.
                       s_w = portfolio.optimal[:s].weights,
                       # Discretely allocated optimal weights with shorting.
                       # Linear programming.
                       sal_w = portfolio.optimal[:sal].weights,
                       # Discretely allocated shares with shorting.
                       # Linear programming.
                       sal_s = portfolio.optimal[:sal].shares,
                       # Discretely allocated optimal weights with shorting.
                       # Greedy algorithm.
                       sag_w = portfolio.optimal[:sag].weights,
                       # Discretely allocated shares with shorting.
                       # Greedy algorithm.
                       sag_s = portfolio.optimal[:sag].shares))
````

## 3.3.3 Shorting with reinvestment

In this section we'll reinvest the money made from short selling, this can be acomplished by setting the value of `long_ub = 1 + short_lb`.

````@example 3_shorting_portfolios
portfolio.short = true

# The absolute value of the sum of the short weights is equal to `0.2`.
portfolio.short_budget = 0.2
# Reinvest the earnings from short selling.
portfolio.budget = 1

portfolio.optimal[:sr] = optimise!(portfolio; rm = rm, obj = obj)
plot_bar(portfolio; type = :sr)
portfolio.frontier[:sr] = efficient_frontier!(portfolio; rm = rm, points = 30)
plot_frontier(portfolio; type = :sr)
````

Lets allocate the short-long portfolio.

````@example 3_shorting_portfolios
# Allocating the short-long portfolio.
portfolio.optimal[:sral] = allocate!(portfolio; type = :sr, investment = investment)
portfolio.optimal[:srag] = allocate!(portfolio; type = :sr,
                                     method = Greedy(; rounding = 0.3),
                                     investment = investment);
nothing #hide
````

Lets verify that the allocations used the money we have available.

Optimal portfolio

````@example 3_shorting_portfolios
long_optimal_idx = portfolio.optimal[:sr].weights .>= 0
short_optimal_idx = .!long_optimal_idx
println("Optimal investment")
println("long = $(sum(investment * portfolio.optimal[:sr].weights[long_optimal_idx]))")
println("long = $(investment + abs(sum(investment * portfolio.optimal[:sr].weights[short_optimal_idx]))) = $(investment) + $(abs(sum(investment * portfolio.optimal[:sr].weights[short_optimal_idx]))) = investment + short_profit")
println("short = $(sum(investment * portfolio.optimal[:sr].weights[short_optimal_idx]))")
println("Sum of weights")
println("long = $(sum(portfolio.optimal[:sr].weights[long_optimal_idx]))")
println("short = $(sum(portfolio.optimal[:sr].weights[short_optimal_idx]))")
````

LP allocated portfolio

````@example 3_shorting_portfolios
long_LP_idx = portfolio.optimal[:sral].weights .>= 0
short_LP_idx = .!long_LP_idx
println("Allocation investment")
println("long = $(dot(portfolio.latest_prices[long_LP_idx], portfolio.optimal[:sral].shares[long_LP_idx]))")
println("long ≈ $(investment + abs(dot(portfolio.latest_prices[short_LP_idx], portfolio.optimal[:sral].shares[short_LP_idx]))) ≈ $(investment) + $(abs(dot(portfolio.latest_prices[short_LP_idx], portfolio.optimal[:sral].shares[short_LP_idx]))) ≈ investment + short_profit")
println("short = $(dot(portfolio.latest_prices[short_LP_idx], portfolio.optimal[:sral].shares[short_LP_idx]))")
println("Sum of weights")
println("long = $(sum(portfolio.optimal[:sral].weights[long_LP_idx]))")
println("short = $(sum(portfolio.optimal[:sral].weights[short_LP_idx]))")
````

Greedy allocated portfolio

````@example 3_shorting_portfolios
long_Greedy_idx = portfolio.optimal[:srag].weights .>= 0
short_Greedy_idx = .!long_Greedy_idx
println("Allocation investment")
println("long = $(dot(portfolio.latest_prices[long_Greedy_idx], portfolio.optimal[:srag].shares[long_Greedy_idx]))")
println("long ≈ $(investment + abs(dot(portfolio.latest_prices[short_Greedy_idx], portfolio.optimal[:srag].shares[short_Greedy_idx]))) ≈ $(investment) + $(abs(dot(portfolio.latest_prices[short_Greedy_idx], portfolio.optimal[:srag].shares[short_Greedy_idx]))) ≈ investment + short_profit")
println("short = $(dot(portfolio.latest_prices[short_Greedy_idx], portfolio.optimal[:srag].shares[short_Greedy_idx]))")
println("Sum of weights")
println("long = $(sum(portfolio.optimal[:srag].weights[long_Greedy_idx]))")
println("short = $(sum(portfolio.optimal[:srag].weights[short_Greedy_idx]))")
````

Here's what the short-long portfolio looks like when we reinvest profits from shorting.

````@example 3_shorting_portfolios
pretty_table(DataFrame(; tickers = portfolio.assets,
                       # Optimal weights with shorting.
                       s_w = portfolio.optimal[:sr].weights,
                       # Discretely allocated optimal weights with shorting.
                       # Linear programming.
                       sal_w = portfolio.optimal[:sral].weights,
                       # Discretely allocated shares with shorting.
                       # Linear programming.
                       sal_s = portfolio.optimal[:sral].shares,
                       # Discretely allocated optimal weights with shorting.
                       # Greedy algorithm.
                       sag_w = portfolio.optimal[:srag].weights,
                       # Discretely allocated shares with shorting.
                       # Greedy algorithm.
                       sag_s = portfolio.optimal[:srag].shares))
````

* * *

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*
