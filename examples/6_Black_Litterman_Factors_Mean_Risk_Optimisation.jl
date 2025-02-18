#src
#src Copywrite (c) 2025
#src Author: Daniel Celis Garza <daniel.celis.garza@gmail.com>
#src SPDX-License-Identifier: MIT

#=
# Example 6: Black Litterman Factors Mean Risk Optimisation

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
## 2. Estimating Black Litterman Portfolios

### 2.1. Reference portfolio

This is a simple example so we will only use default parameters for computing the statistics.

For API details and options available see: [`Portfolio`](@ref), [`PortOptSolver`](@ref), [`PortfolioOptimiser.MeanEstimator`](@ref), [`PortfolioOptimiser.PortfolioOptimiserCovCor`](@ref), [`BLType`](@ref), [`asset_statistics!`](@ref), [`black_litterman`](@ref), [`RiskMeasure`](@ref), [`PortfolioOptimiser.ObjectiveFunction`](@ref), [`PortfolioOptimiser.OptimType`](@ref).
=#

## Creating the portfolio object. Internally computes the returns if you give a prices TimeArray.
port = Portfolio(; prices = prices,
                 ## Continuous solvers.
                 solvers = PortOptSolver(; name = :Clarabel, solver = Clarabel.Optimizer,
                                         params = Dict("verbose" => false)),
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
## Creating the optimisation object.
rm = SD() # Risk measure.
obj = Sharpe() # Objective function. Can be `MinRisk()`, `Utility()`, `Sharpe()`, `MaxRet()`.
## `Trad` optimisation corresponds to the classic mean risk optimisation.
type = Trad(; rm = rm, obj = obj)

## Optimise portfolio.
w1 = optimise!(port, type)
pretty_table(w1; formatters = fmt2)
plot_bar(port)
