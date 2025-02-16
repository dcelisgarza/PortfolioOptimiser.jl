#src
#src Copywrite (c) 2025
#src Author: Daniel Celis Garza <daniel.celis.garza@gmail.com>
#src SPDX-License-Identifier: MIT

#=
# Example 4: Mean Risk Optimisation with Factor Models with Linear Factor Constraints

## 1. Download data.

=#

using PortfolioOptimiser, TimeSeries, DataFrames, PrettyTables, Clarabel, HiGHS, YFinance,
      GraphRecipes, StatsPlots, JuMP, GLM

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
        return isa(v, Number) ? "$(round(v, digits=5))" : v
    end
end;
fmt3 = (v, i, j) -> begin
    if j == 1
        return v
    else
        return isa(v, Number) ? "$(round(v*100, digits=4)) %" : v
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
## Factor tickers.
factors = sort!(["MTUM", "QUAL", "SIZE", "USMV", "VLUE", "LRGF", "INTF", "GLOF", "EFAV",
                 "EEMV"])

tickers = [assets; factors]

## Prices date range.
Date_0 = "2019-01-01"
Date_1 = "2025-01-31"

## Download the price data using YFinance.
prices = get_prices.(tickers; startdt = Date_0, enddt = Date_1)
prices = stock_price_to_time_array.(prices)
prices = hcat(prices...)
cidx = colnames(prices)[occursin.(r"adj", string.(colnames(prices)))]
prices = prices[cidx]
TimeSeries.rename!(prices, Symbol.(tickers))
pretty_table(prices[1:5]; formatters = fmt1)

#=
## 2. Estimating Mean Risk Portfolios

### 2.1. Factor and asset statistics

In order to use factor models, we need to estimate how the factors and assets are related. The relationships are summarised by the loadings matrix, which is obtained via regression. For this basic tutorial we will use the defaults.

This is a simple example so we will only use default parameters for computing the statistics.

[`PortfolioOptimiser.MeanEstimator`](@ref), [`PortfolioOptimiser.PortfolioOptimiserCovCor`](@ref), [`FactorType`](@ref), [`asset_statistics!`](@ref), [`factor_statistics!`](@ref).
=#

## Creating the portfolio object. Internally computes the returns if you give a prices TimeArray.
port = Portfolio(; prices = prices[Symbol.(assets)], f_prices = prices[Symbol.(factors)],
                 ## Continuous solvers.
                 solvers = PortOptSolver(; name = :Clarabel, solver = Clarabel.Optimizer,
                                         params = Dict("verbose" => false)),
                 ## Discrete solvers (for discrete allocation).
                 alloc_solvers = PortOptSolver(; name = :HiGHS,
                                               solver = optimizer_with_attributes(HiGHS.Optimizer,
                                                                                  MOI.Silent() => true)))
## Compute relevant statistics.
## Expected returns, covariance estimation and factor estimation methods.
mu_type = MuSimple()
cov_type = PortCovCor()
factor_type = FactorType()

asset_statistics!(port; mu_type = mu_type, cov_type = cov_type, set_kurt = false,
                  set_skurt = false, set_skew = false, set_sskew = false)
factor_statistics!(port; factor_type = factor_type, cov_type = cov_type, mu_type = mu_type)

## Show the loadings matrix.
pretty_table(port.loadings; formatters = fmt2)

#=
### 2.2. Optimise Portfolios
=#

## Creating the optimisation object.
rm = SD() # Risk measure.
obj = Sharpe() # Objective function. Can be `MinRisk()`, `Utility()`, `Sharpe()`, `MaxRet()`.
class = FM()
## `Trad` optimisation corresponds to the classic mean risk optimisation.
type = Trad(; rm = rm, obj = obj, class = class)
## Classic portfolio.
w1 = optimise!(port, type)
## Factor model portfolio.
pretty_table(w1; formatters = fmt3)

#=
## 3. Factor constraints

### 3.1. Creating the constraints

The function [`factor_constraints`](@ref) takes in two dataframes, one defining the constraints, and the loadings matrix and turns them into a matrix and vector which sets the constraints as ``\\mathbf{A} \\bm{x} >= \\bm{b}``.

First lets check out the loadings matrix in order to create feasable constraints.
=#

pretty_table(describe(port.loadings); formatters = fmt3)

# Constrain factors.
constraints = DataFrame(; Enabled = [true, true, true, true],
                        Factor = ["SIZE", "QUAL", "USMV", "MTUM"],
                        Sign = [">=", ">=", "<=", ">="], Value = [0.2, 0.6, -0.7, 0.45],
                        Relative_Factor = ["", "LRGF", "", ""])
pretty_table(constraints)

# Create linear constraint matrix and vector and optimise with these constraints.
A1, B1 = factor_constraints(constraints, port.loadings)

## Clear the arrays because the code asserts the dimensions.
port.a_ineq = Matrix(undef, 0, 0)
port.b_ineq = Vector(undef, 0)
port.a_ineq = A1
port.b_ineq = B1

w2 = optimise!(port, type)
pretty_table(w2; formatters = fmt3)

#=
In order to verify that the constraints have been held, we can perform a regression.
=#

## Add constant term.
X = [ones(size(port.fm_returns, 1)) port.f_returns]
## Portfolio returns according to the factor model.
y = port.fm_returns * w2.weights
## Generalised linear model linking the portfolio returns to the factors in X.
res = GLM.lm(X, y)
## Generate dataframe with the factors and their regression coefficients.
df = DataFrame(; :factors => ["const"; factors], :coefs => coef(res))
pretty_table(df; formatters = fmt3)

#=
We can see that the constraints hold.

| Factor |     Constraint     |              Value               |
|-------:|:------------------:|:---------------------------------|
|  SIZE  |    SIZE >= 0.2     |         0.395261 >= 0.2          |
|  QUAL  | LRGF - QUAL >= 0.6 | 1.106847 - 0.506847 = 0.6 >= 0.6 |
|  USMV  |    USMV <= -0.7    |         -0.75905 <= -0.7         |
|  MTUM  |    MTUM >= 0.45    |           0.45 >= 0.45           |
=#

#=
### 3.2. Efficient Frontier

We can plot the efficient frontier for the factor model. Which will look different factor model one because the constraints will have to be satisfied at every point.
=#
points = 50
frontier = efficient_frontier!(port, type; points = points)
pretty_table(frontier[:weights]; formatters = fmt3)

# Plot frontier.
plot_frontier(port; rm = rm)

# Plot frontier area.
plot_frontier_area(port; rm = rm, kwargs_a = (; legendfontsize = 7))
