#=
# Example 5: Risk parity

This example follows from previous ones. If something in the preamble is confusing, it is explained there.

This example focuses on the [`RP`](@ref) optimisation type of [`Portfolio`](@ref).

## 5.1 Downloading the data
=#

## using Pkg
## Pkg.add.(["StatsPlots", "GraphRecipes", "YFinance", "Clarabel", "HiGHS", "CovarianceEstimation", "SparseArrays"])
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

#=
## 5.2 Instantiating an instance of [`Portfolio`](@ref).

We'll compute basic statistics for this.
=#

portfolio = Portfolio(; prices = prices,
                      ## Continuous optimiser.
                      solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                       :check_sol => (allow_local = true,
                                                                      allow_almost = true),
                                                       :params => Dict("verbose" => false))),
                      ## MIP optimiser for the discrete allocation.
                      alloc_solvers = Dict(:HiGHS => Dict(:solver => HiGHS.Optimizer,
                                                          :check_sol => (allow_local = true,
                                                                         allow_almost = true),
                                                          :params => Dict("log_to_console" => false))));

asset_statistics!(portfolio)

#=
## 5.3 Optimising the portfolio

The [`RP`](@ref) uses a risk budget vector for defining the risk contribution of each asset. The vector defaults to equal risk contribution for all assets.

Risk parity portfolios don't use a user-provided objective function. They minimise the risk subject to a constraint that minimises the deviation
=#

rm = SD()
type = RP()
w1 = optimise!(portfolio; type = type, rm = rm)
pretty_table(w1; formatters = fmt1)

#=
We can check that the risk budget and risk contribution match.
=#

## Risk budget.
rb = portfolio.risk_budget
## Compute the risk contribution, for the [`SD`](@ref) risk measure.
rc = risk_contribution(portfolio; type = :RP, rm = rm)
## Normalise risk contribution so it adds up to 1 a.k.a. 100%.
rc ./= sum(rc)

pretty_table(hcat(w1, DataFrame(; budget = rb, contribution = rc)); formatters = fmt1)

#=
As you can see, the weights of each asset in the portfolio are such that the asset contributes 4% of the risk of the portfolio. This is because we used the default value for the risk budget, which defaults to equal risk contribution per asset, which is equal to `1/N`, where `N` is the number of assets.

We can also plot the risk contribution in asbolute and relative terms.
=#

#nb display(plot_risk_contribution(portfolio; rm = rm, type = :RP, percentage = false))
#nb display(plot_risk_contribution(portfolio; rm = rm, type = :RP, percentage = true))
#md plot_risk_contribution(portfolio; rm = rm, type = :RP, percentage = false)
#md plot_risk_contribution(portfolio; rm = rm, type = :RP, percentage = true)

#=
Lets change the risk budget to something a little bit more interesting. The risk budget can be provided from to the [`Portfolio`](@ref) constructor, or after instatiation. Either way, the risk budget will be normalised to add up to 1. If using the latter method, the element type of the vector provided must match that of `risk_budget` because the normalisation is done in-place to avoid unecessary allocations.
=#

x = range(; start = 0, stop = 2pi, length = length(w1.weights))
portfolio.risk_budget = sin.(x) .^ 2;

# Lets optimise using this risk budget to see what happens.

w1 = optimise!(portfolio; type = type, rm = rm)
rb = portfolio.risk_budget
## Compute the risk contribution, for the [`SD`](@ref) risk measure.
rc = risk_contribution(portfolio; type = :RP, rm = rm)
## Normalise risk contribution so it adds up to 1 a.k.a. 100%.
rc ./= sum(rc)
pretty_table(hcat(w1, DataFrame(; budget = rb, contribution = rc)); formatters = fmt1)

# Lets plot the results.

#nb display(plot_risk_contribution(portfolio; rm = rm, type = :RP, percentage = false))
#nb display(plot_risk_contribution(portfolio; rm = rm, type = :RP, percentage = true))
#md plot_risk_contribution(portfolio; rm = rm, type = :RP, percentage = false)
#md plot_risk_contribution(portfolio; rm = rm, type = :RP, percentage = true)

#=
We've used the [`SD`](@ref) risk function for computing the risk parity portfolio, as well as for computing and plotting the risk contribution, but there's nothing stopping us from computing the risk contribution and/or plotting the risk contribution for risk measures other than the one that was optimised.
=#

## Compute the risk contribution, for the [`CDaR`](@ref) risk measure.
rc = risk_contribution(portfolio; type = :RP, rm = CDaR())
## Normalise risk contribution so it adds up to 1 a.k.a. 100%.
rc ./= sum(rc)
pretty_table(hcat(w1, DataFrame(; budget_SD = rb, contribution_CDaR = rc));
             formatters = fmt1)

#nb display(plot_risk_contribution(portfolio; rm = CDaR(), type = :RP, percentage = false))
#nb display(plot_risk_contribution(portfolio; rm = CDaR(), type = :RP, percentage = true))
#md plot_risk_contribution(portfolio; rm = CDaR(), type = :RP, percentage = false)
#md plot_risk_contribution(portfolio; rm = CDaR(), type = :RP, percentage = true)
