#src
#src Copywrite (c) 2025
#src Author: Daniel Celis Garza <daniel.celis.garza@gmail.com>
#src SPDX-License-Identifier: MIT

#=
# Example 7: Near Optimal Centering (NOC) Mean Risk Optimisation

## 1. Download data.

=#

using PortfolioOptimiser, TimeSeries, DataFrames, PrettyTables, Clarabel, HiGHS, YFinance,
      GraphRecipes, StatsPlots, JuMP

## Format for pretty tables.
fmt1 = (v, i, j) -> begin
    if j == 1
        return Date(v)
    else
        return v
    end
end;
fmt2 = (v, i, j) -> begin
    if j == 1
        return v
    else
        return isa(v, Number) ? "$(round(v*100, digits=3)) %" : v
    end
end;

## Convert prices to time array.
function stock_price_to_time_array(x)
    ## Only get the keys that are not ticker or datetime.
    coln = collect(keys(x))[3:end]
    ## Convert the dictionary into a matrix.
    m = hcat([x[k] for k ∈ coln]...)
    return TimeArray(x["timestamp"], m, Symbol.(coln), x["ticker"])
end

## Asset tickers.
assets = sort!(["AAPL", "ADI", "ADP", "AMGN", "AMZN", "BKNG", "CMCSA", "COST", "CSCO",
                "GILD", "GOOG", "GOOGL", "HON", "ISRG", "LIN", "MAR", "META", "MRK", "MSFT",
                "NFLX", "NVDA", "ORLY", "PANW", "QCOM", "SBUX", "T", "TMUS", "TSLA", "TXN",
                "VRTX"])

## Prices date range.
Date_0 = "2019-01-01"
Date_1 = "2025-01-31"

## Download the price data using YFinance.
prices = get_prices.(assets; startdt = Date_0, enddt = Date_1)
prices = stock_price_to_time_array.(prices)
prices = hcat(prices...)
cidx = colnames(prices)[occursin.(r"adj", string.(colnames(prices)))]
prices = prices[cidx]
TimeSeries.rename!(prices, Symbol.(assets))
pretty_table(prices[1:5]; formatters = fmt1)

#=
## 2. Estimating NOC Mean Risk Portfolios

### 2.1. Optimising portfolio

This is a simple example so we will only use default parameters for computing the statistics.

NOC optimisations are only compatible with optimisations whose risk measures are [`AffExpr`](https://jump.dev/JuMP.jl/stable/api/JuMP/#AffExpr) because [`QuadExpr`](https://jump.dev/JuMP.jl/stable/api/JuMP/#QuadExpr) are not strictly convex.

For API details and options available see: [`Portfolio`](@ref), [`PortOptSolver`](@ref), [`PortfolioOptimiser.MeanEstimator`](@ref), [`PortfolioOptimiser.PortfolioOptimiserCovCor`](@ref), [`asset_statistics!`](@ref), [`RiskMeasure`](@ref), [`PortfolioOptimiser.ObjectiveFunction`](@ref), [`PortfolioOptimiser.OptimType`](@ref), [`NOC`](@ref).
=#

## Creating the portfolio object. Internally computes the returns if you give a prices TimeArray.
port = Portfolio(; prices = prices,
                 ## Continuous solvers.
                 solvers = PortOptSolver(; name = :Clarabel, solver = Clarabel.Optimizer,
                                         check_sol = (allow_local = true,
                                                      allow_almost = true),
                                         params = Dict("verbose" => false,
                                                       "max_step_fraction" => 0.65)),
                 ## Discrete solvers (for discrete allocation).
                 alloc_solvers = PortOptSolver(; name = :HiGHS,
                                               solver = optimizer_with_attributes(HiGHS.Optimizer,
                                                                                  MOI.Silent() => true)))

## Compute relevant statistics.
## Expected returns and covariance estimation methods.
mu_type = MuSimple()
cov_type = PortCovCor()

## Only compute `mu` and `cov`.
asset_statistics!(port; mu_type = mu_type, cov_type = cov_type, set_kurt = false,
                  set_skurt = false, set_skew = false, set_sskew = false)

#=
The Near Optimal Centering (NOC) formulation has the curious property that maximising the risk-adjusted ratio will not, in fact, maximise the risk-adjusted ratio. It finds the analytical centre of the neighbourhood around a point on the efficient frontier, which in this case is the point which maximises the risk-adjusted return ratio. This makes it possible to create near-optimal portfolios with higher risk-adjusted return ratios. Such portfolios are not the centre of the analytic region of the portfolio with the highest risk-adjusted return ratio, they are the centre of a different analytic region around a different point of the efficient frontier. It's possible that such points can have higher risk-adjusted return ratios.

In fact, the NOC portfolio which maximises the risk-adjusted return ratio, is the point of minimum risk of a region of the NOC efficient frontier with relatively high risk-adjusted returns.

By defining larger analytic regions (using fewer `bins`), the more portfolios will exist with greater risk-adjusted return ratios than the canonical NOC maximum risk-adjusted return ratio portfolio.

- `bins → ∞`: the NOC portfolio converges to the optimal portfolio.
- `bins → 0`: the NOC portfolio converges to the equal weight portfolio.

You can verify for yourself by changing the value of `bins` above and re-running the script. Note how the maximum risk-adjusted return portfolio migrates further down the NOC frontier as `bins` decreases, and how there are more portfolios with higher risk-adjusted return rations.
=#

## Creating the optimisation object.
rm = SD() # Risk measure.
obj = Sharpe() # Objective function. Can be `MinRisk()`, `Utility()`, `Sharpe()`, `MaxRet()`.
bins = 20 ## Number of bins for defining the analytic region.
## `NOC` optimisation corresponds to the near optimal mean risk optimisation.
type = NOC(; rm = rm, obj = obj, bins = bins)

## Optimise portfolio.
w1 = optimise!(port, type)
pretty_table(w1; formatters = fmt2)

#=
### 2.2. Plotting portfolio composition.
=#

plot_bar(port, :NOC)

#=
### 2.3. Efficient frontier
=#

points = 50
frontier = efficient_frontier!(port, type; points = points)
pretty_table(frontier[:weights]; formatters = fmt2)

# Plot frontier.
plot_frontier(port, :NOC; rm = rm)

# Plot frontier area.
plot_frontier_area(port, :NOC; rm = rm, kwargs_a = (; legendfontsize = 7))

#=
We want to see how the canonical maximum risk-adjusted return ratio portfolio stacks up against other portfolios with greater risk-adjusted return ratios. We can do so using the `frontier` variable.

Frist, we check if the frontier contains the portfolio which maximises the risk-adjusted return ratio (the optimisation can fail in [`efficient_frontier!`](@ref) so we should always check if it succeeded), there is a flag `:sharpe` which is `true` if the optimisation succeeded. The maximum risk-adjusted ratio optimisation is performed last, so its statistics and weights will correspond to the last ones in their respective containers.

We will display them at the start to make it easier to see the behaviour described in [2.1](#2.1.-Optimising-portfolio). We annualise the risk, return, and risk-adjusted return ratio to make comparisons easier.
=#

## Check if the maximum risk-adjusted return ratio optimisation succeeded
if frontier[:sharpe]
    ## Optimisations can fail in [`efficient_frontier!`](@ref) so we need to find out how many actually succeeded.
    N = length(frontier[:sharpes])
    ## Find all points in the efficient frontier with an actual sharpe ratio higher than the canonical one.
    idx = findall(frontier[:sharpes] .> frontier[:sharpes][end])
    ## Add the index of the max risk-adjusted return ratio portfolio at the start.
    idx = [N; idx]
    ## Display the anualised returns and risks of the canonical max risk-adjusted return ratio portfolio (the first in the table) as well as all others whose ratios are greater. The first row contains the canonical portfolio.
    pretty_table(DataFrame(:idx => idx, :rets => frontier[:rets][idx] * 252,
                           :risks => frontier[:risks][idx] * sqrt(252),
                           :sharpes => frontier[:sharpes][idx] * sqrt(252));
                 formatters = fmt2)
    ## Display their weights. The first column displays the canonical portfolio.
    pretty_table(frontier[:weights][!, [1; (idx .+ 1)]]; formatters = fmt2)
end
