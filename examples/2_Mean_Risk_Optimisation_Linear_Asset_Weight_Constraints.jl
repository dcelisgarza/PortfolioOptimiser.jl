#src
#src Copywrite (c) 2025
#src Author: Daniel Celis Garza <daniel.celis.garza@gmail.com>
#src SPDX-License-Identifier: MIT

#=
# Example 2: Mean Risk Optimisation with Linear Asset Weight Constraints

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
## 2. Estimating Mean Risk Portfolios

### 2.1. Optimising portfolio

This is a simple example so we will only use default parameters for computing the statistics.

For API details and options available see: [`Portfolio`](@ref), [`PortOptSolver`](@ref), [`PortfolioOptimiser.MeanEstimator`](@ref), [`PortfolioOptimiser.PortfolioOptimiserCovCor`](@ref), [`asset_statistics!`](@ref), [`RiskMeasure`](@ref), [`PortfolioOptimiser.ObjectiveFunction`](@ref), [`PortfolioOptimiser.OptimType`](@ref).
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
obj = MinRisk() # Objective function. Can be `MinRisk()`, `Utility()`, `Sharpe()`, `MaxRet()`.
## `Trad` optimisation corresponds to the classic mean risk optimisation.
type = Trad(; rm = rm, obj = obj)

## Optimise portfolio.
w1 = optimise!(port, type)
pretty_table(w1; formatters = fmt2)

#=
### 2.2. Plotting portfolio composition.
=#

plot_bar(port)

#=
## 3. Asset and asset set constraints

### 3.1. Creating the constraints

The function [`asset_constraints`](@ref) takes in two dataframes, one defining the asset sets, and another defining the constraints and turns them into a matrix and vector which sets the constraints as ``\\mathbf{A} \\bm{x} >= \\bm{b}``.
=#
asset_sets = DataFrame(;
                       Asset = ["AAPL", "ADI", "ADP", "AMGN", "AMZN", "BKNG", "CMCSA",
                                "COST", "CSCO", "GILD", "GOOG", "GOOGL", "HON", "ISRG",
                                "LIN", "MAR", "META", "MRK", "MSFT", "NFLX", "NVDA", "ORLY",
                                "PANW", "QCOM", "SBUX", "T", "TMUS", "TSLA", "TXN", "VRTX"],
                       Sector = ["Technology", "Technology", "Technology", "Health Care",
                                 "Consumer Discretionary", "Consumer Discretionary",
                                 "Telecommunications", "Consumer Discretionary",
                                 "Telecommunications", "Health Care", "Technology",
                                 "Technology", "Industrials", "Health Care", "Industrials",
                                 "Consumer Discretionary", "Technology", "Health Care",
                                 "Technology", "Consumer Discretionary", "Technology",
                                 "Consumer Discretionary", "Technology", "Technology",
                                 "Consumer Discretionary", "Telecommunications",
                                 "Telecommunications", "Consumer Discretionary",
                                 "Technology", "Health Care"],
                       Industry = ["Computer Manufacturing", "Semiconductors",
                                   "EDP Services",
                                   "Biotechnology: Biological Products (No Diagnostic Substances)",
                                   "Catalog/Specialty Distribution",
                                   "Transportation Services",
                                   "Cable & Other Pay Television Services",
                                   "Department/Specialty Retail Stores",
                                   "Computer Communications Equipment",
                                   "Biotechnology: Biological Products (No Diagnostic Substances)",
                                   "Computer Software: Programming Data Processing",
                                   "Computer Software: Programming Data Processing",
                                   "Aerospace", "Industrial Specialties", "Major Chemicals",
                                   "Hotels/Resorts",
                                   "Computer Software: Programming Data Processing",
                                   "Biotechnology: Pharmaceutical Preparations",
                                   "Computer Software: Prepackaged Software",
                                   "Consumer Electronics/Video Chains", "Semiconductors",
                                   "Auto & Home Supply Stores",
                                   "Computer peripheral equipment",
                                   "Radio And Television Broadcasting And Communications Equipment",
                                   "Restaurants", "Telecommunications Equipment",
                                   "Telecommunications Equipment", "Auto Manufacturing",
                                   "Semiconductors",
                                   "Biotechnology: Pharmaceutical Preparations"])
pretty_table(asset_sets)

#=
We will create different sets of constraints because it makes demonstrations easier. 

It's also worth noting that constraints may make problems infeasable, for example constraining all `N` assets to have weights greater than `1/(N-1)`, or it may be impossible to simultaneously satisfy multiple constraints, for example constraining all assets to have weights greater than or equal to 0.1, and requiring one asset to have a weight smaller than or equal to 0.05.
=#

# Constrain individual assets.
constraints_1 = DataFrame(; Enabled = [true, true, true, true, true, true],
                          Type = ["Asset", "Asset", "Asset", "Asset", "Asset", "Asset"],
                          Set = ["", "", "", "", "", ""],
                          Position = ["COST", "AAPL", "ADP", "T", "GILD", "GOOG"],
                          Sign = ["<=", ">=", "<=", ">=", "<=", ">="],
                          Weight = [0.13, 0.04, "", "", "", ""],
                          Relative_Type = ["", "", "Asset", "Asset", "Subset", "Subset"],
                          Relative_Set = ["", "", "", "", "Sector", "Industry"],
                          Relative_Position = ["", "", "MAR", "MRK", "Telecommunications",
                                               "Telecommunications Equipment"],
                          Factor = ["", "", 2, 0.7, 0.3, 1])
pretty_table(constraints_1)

# Create linear constraint matrix and vector and optimise with these constraints.
A1, B1 = asset_constraints(constraints_1, asset_sets)

## Clear the arrays because the code asserts the dimensions.
port.a_ineq = Matrix(undef, 0, 0)
port.b_ineq = Vector(undef, 0)
port.a_ineq = A1
port.b_ineq = B1

w2 = optimise!(port, type)
pretty_table(w2; formatters = fmt2)
#
plot_bar(port)

# Constrain all assets.
constraints_2 = DataFrame(; Enabled = [true, true, true, true, true, true],
                          Type = ["All Assets", "All Assets", "All Assets", "All Assets",
                                  "All Assets", "All Assets"],
                          Set = ["", "", "", "", "", ""],
                          Position = ["", "", "", "", "", ""],
                          Sign = [">=", "<=", ">=", "<=", ">=", "<="],
                          Weight = [0.01, 0.2, "", "", "", ""],
                          Relative_Type = ["", "", "Subset", "Subset", "Asset", "Asset"],
                          Relative_Set = ["", "", "Sector", "Industry", "", ""],
                          Relative_Position = ["", "", "Consumer Discretionary",
                                               "Biotechnology: Pharmaceutical Preparations",
                                               "TMUS", "META"],
                          Factor = ["", "", 0.1, 2.1, 0.5, 1.3])
pretty_table(constraints_2)

# Create linear constraint matrix and vector and optimise with these constraints.
A2, B2 = asset_constraints(constraints_2, asset_sets)
## Clear the arrays because the code asserts the dimensions.
port.a_ineq = Matrix(undef, 0, 0)
port.b_ineq = Vector(undef, 0)
port.a_ineq = A2
port.b_ineq = B2

w3 = optimise!(port, type)
pretty_table(w3; formatters = fmt2)
#
plot_bar(port)

# Constrain asset sets.
constraints_3 = DataFrame(; Enabled = [true, true, true, true, true, true],
                          Type = ["Subset", "Subset", "Subset", "Subset", "Subset",
                                  "Subset"],
                          Set = ["Sector", "Sector", "Sector", "Industry", "Industry",
                                 "Industry"],
                          Position = ["Technology", "Consumer Discretionary", "Health Care",
                                      "Semiconductors",
                                      "Biotechnology: Pharmaceutical Preparations",
                                      "Biotechnology: Biological Products (No Diagnostic Substances)"],
                          Sign = [">=", "<=", "<=", ">=", "<=", ">="],
                          Weight = [0.13, 0.25, "", "", "", ""],
                          Relative_Type = ["", "", "Asset", "Asset", "Subset", "Subset"],
                          Relative_Set = ["", "", "", "", "Sector", "Industry"],
                          Relative_Position = ["", "", "MAR", "MRK", "Telecommunications",
                                               "Telecommunications Equipment"],
                          Factor = ["", "", 2, 0.7, 0.1, 0.3])
pretty_table(constraints_3)

# Create linear constraint matrix and vector and optimise with these constraints.
A3, B3 = asset_constraints(constraints_3, asset_sets)

## Clear the arrays because the code asserts the dimensions.
port.a_ineq = Matrix(undef, 0, 0)
port.b_ineq = Vector(undef, 0)
port.a_ineq = A3
port.b_ineq = B3

w4 = optimise!(port, type)
pretty_table(w4; formatters = fmt2)
#
plot_bar(port)

# Constrain all asset subsets.
constraints_4 = DataFrame(; Enabled = [true, true, true],
                          Type = ["All Subsets", "All Subsets", "All Subsets"],
                          Set = ["Industry", "Industry", "Sector"], Position = ["", "", ""],
                          Sign = [">=", ">=", "<="], Weight = [0.01, "", ""],
                          Relative_Type = ["", "Asset", "Subset"],
                          Relative_Set = ["", "", "Industry"],
                          Relative_Position = ["", "T", "Semiconductors"],
                          Factor = ["", 0.7, 1.3])
pretty_table(constraints_4)

# Create linear constraint matrix and vector and optimise with these constraints.
A4, B4 = asset_constraints(constraints_4, asset_sets)

## Clear the arrays because the code asserts the dimensions.
port.a_ineq = Matrix(undef, 0, 0)
port.b_ineq = Vector(undef, 0)
port.a_ineq = A4
port.b_ineq = B4

w5 = optimise!(port, type)
pretty_table(w5; formatters = fmt2)
#
plot_bar(port)

# Constrain each asset in subset.
constraints_5 = DataFrame(; Enabled = [true, true, true],
                          Type = ["Each Asset in Subset", "Each Asset in Subset",
                                  "Each Asset in Subset"],
                          Set = ["Sector", "Industry", "Sector"],
                          Position = ["Telecommunications",
                                      "Biotechnology: Biological Products (No Diagnostic Substances)",
                                      "Consumer Discretionary"], Sign = [">=", ">=", "<="],
                          Weight = [0.03, "", ""], Relative_Type = ["", "Asset", "Subset"],
                          Relative_Set = ["", "", "Industry"],
                          Relative_Position = ["", "T", "Semiconductors"],
                          Factor = ["", 0.7, 1.3])
pretty_table(constraints_5)

# Create linear constraint matrix and vector and optimise with these constraints.
A5, B5 = asset_constraints(constraints_5, asset_sets)

## Clear the arrays because the code asserts the dimensions.
port.a_ineq = Matrix(undef, 0, 0)
port.b_ineq = Vector(undef, 0)
port.a_ineq = A5
port.b_ineq = B5

w6 = optimise!(port, type)
pretty_table(w6; formatters = fmt2)
#
plot_bar(port)

#=
## 4. Efficient Frontier

It's possible to compute the efficient frontier with constraints. It will be different to the vanilla one in the previous tutorial because the constraints will be applied to every optimisation.
=#
port.a_ineq = Matrix(undef, 0, 0)
port.b_ineq = Vector(undef, 0)
port.a_ineq = A2
port.b_ineq = B2

points = 50
frontier = efficient_frontier!(port, type; points = points)
pretty_table(frontier[:weights]; formatters = fmt2)

# Plot frontier.
plot_frontier(port; rm = rm)

# Plot frontier area.
plot_frontier_area(port; rm = rm, kwargs_a = (; legendfontsize = 7))
