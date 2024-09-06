#=
# Example 3: Shorting portfolios

This tutorial follows from previous tutorials. If something in the preamble is confusing, it is explained there.

This tutorial focuses on using the optimisation constraints available to [`Trad`](@ref) optimisations of [`Portfolio`](@ref).

## 1. Downloading the data
=#

## using Pkg
## Pkg.add.(["StatsPlots", "GraphRecipes", "MarketData", "Clarabel", "HiGHS", "CovarianceEstimation", "SparseArrays"])
using Clarabel, CovarianceEstimation, DataFrames, Dates, GraphRecipes, HiGHS, MarketData,
      PortfolioOptimiser, Statistics, StatsBase, StatsPlots, TimeSeries, LinearAlgebra,
      PrettyTables

fmt1 = (v, i, j) -> begin
    if j ∈ (1, 5, 7, 9)
        return v
    else
        return if isa(v, Number)
            "$(round(v*100, digits=3)) %"
        else
            v
        end
    end
end;

assets = Symbol.(["AAL", "AAPL", "AMC", "BB", "BBY", "DELL", "DG", "DRS", "GME", "INTC",
                  "LULU", "MARA", "MCI", "MSFT", "NKLA", "NVAX", "NVDA", "PARA", "PLNT",
                  "SAVE", "SBUX", "SIRI", "STX", "TLRY", "TSLA"])

Date_0 = DateTime(2019, 01, 01)
Date_1 = DateTime(2023, 01, 01)

function get_prices(assets)
    prices = TimeSeries.rename!(yahoo(assets[1],
                                      YahooOpt(; period1 = Date_0, period2 = Date_1))[:AdjClose],
                                assets[1])
    for asset ∈ assets[2:end]
        ## Yahoo doesn't like regular calls to their API.
        sleep(rand() / 10)
        prices = merge(prices,
                       TimeSeries.rename!(yahoo(asset,
                                                YahooOpt(; period1 = Date_0,
                                                         period2 = Date_1))[:AdjClose],
                                          asset), :outer)
    end
    return prices
end

prices = get_prices(assets)

# ## 2. Instantiating an instance of [`Portfolio`](@ref).

portfolio = Portfolio(; prices = prices,
                      ## Continuous optimiser.
                      solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                       :check_sol => (allow_local = true,
                                                                      allow_almost = true),
                                                       :params => Dict("verbose" => false,
                                                                       "max_step_fraction" => 0.7))),
                      ## MIP optimiser for the discrete allocation.
                      alloc_solvers = Dict(:HiGHS => Dict(:solver => HiGHS.Optimizer,
                                                          :check_sol => (allow_local = true,
                                                                         allow_almost = true),
                                                          :params => Dict("log_to_console" => false))));
#
mu_type = MuSimple()
cov_type = PortCovCor()
asset_statistics!(portfolio; mu_type = mu_type, cov_type = cov_type)

#
## Risk free rate.
rf = 3.5 / 100 / 252
## Risk aversion.
l = 2.0
## Objective function.
obj = MinRisk()
## Risk measure.
rm = SD()

#=
## 3.1 Optimising the portfolio

Enabling shorting is quite easy. It alows negative weights, which correspond to shorting portfolios. It is a generally good idea to start with small amounds of leverage, and to minimise risk on leveraged portfolios. However, all objective functions work.

The default shorting portfolio is 20 % short, 80 % long.

First we will optimise the portfolio without shorting and plot the weights and the efficient frontier.
=#

portfolio.short = false
portfolio.optimal[:ns] = optimise!(portfolio; rm = rm, obj = obj)
display(plot_bar(portfolio, :ns))
portfolio.frontier[:ns] = efficient_frontier!(portfolio; rm = rm, points = 30)
display(plot_frontier(portfolio, :ns))

#=
We'll now enable shorting. As you can see, some became negative, and the efficient frontier changes slightly. This is because we can now take advantage of negative returns. However, shorting comes with a lot more risk, as growth is uncapped while losses are capped at zero. Thus, it requires more active management and risk mitigation strategies. Case in point, NVDA is assigned -10.6 % in the short minimum risk optimisation, which would have turned out very poorly for someone shorting the stock.
=#

portfolio.short = true
## The absolute value of the sum of short weights is equal to 0.2
portfolio.short_u = 0.2
## Long weights add up to 1.0
portfolio.long_u = 1
## The sum of short and long weights. This cannot be changed, it is automatically computed from `sum_short_long = long_u - short_u`, i.e. the sum of the short and long weights.
portfolio.sum_short_long == 0.8

portfolio.optimal[:s] = optimise!(portfolio; rm = rm, obj = obj)
display(plot_bar(portfolio, :s))
portfolio.frontier[:s] = efficient_frontier!(portfolio; rm = rm, points = 30)
display(plot_frontier(portfolio, :s))

## Check that absolute value of the sum of short weights adds up to `portfolio.short_u``
println(isapprox(abs(sum(portfolio.optimal[:s].weights[portfolio.optimal[:s].weights .<= 0])),
                 portfolio.short_u; rtol = 1e-4))

## Check that the sum of long weights adds up to `portfolio.long_u`
println(isapprox(abs(sum(portfolio.optimal[:s].weights[portfolio.optimal[:s].weights .>= 0])),
                 portfolio.long_u; rtol = 1e-4))

#=
## 3.2 LP Allocation 

We'll now allocate the portfolio according to our means. We'll use LP allocation first.
=#

## Our investment.
investment = 6750

## Allocating the long-only portfolio.
portfolio.optimal[:nsa] = allocate!(portfolio, :ns; method = LP(), investment = investment)

## Allocating the short-long portfolio without reinvesting the money we make from shorting.
## If you dont provide `method`, it will default to LP().
portfolio.optimal[:sanr] = allocate!(portfolio, :s; investment = investment,
                                     reinvest = false)

## Allocating the short-long portfolio reinvesting the money we make from shorting.
portfolio.optimal[:sar] = allocate!(portfolio, :s; investment = investment, reinvest = true)

#=
We can print the long, short-long optimal portfolios, as well as the long, short-long without reinvesting, and short-long with reinvesting portfolios for comparison. Note how the short-long portfolio with reinvesting means we can purchase more long-shares.
=#
pretty_table(DataFrame(; tickers = portfolio.assets,
                       ## Optimal weights without shorting.
                       ns_w = portfolio.optimal[:ns].weights,
                       ## Optimal weights with shorting.
                       s_w = portfolio.optimal[:s].weights,
                       ## Discretely allocated optimal weights without shorting.
                       nsa_w = portfolio.optimal[:nsa].weights,
                       ## Discretely allocated shares without shorting.
                       nsa_s = portfolio.optimal[:nsa].shares,
                       ## Discretely allocated weights with shorting, and reinvesting
                       ## money made from shorting.
                       sar_w = portfolio.optimal[:sar].weights,
                       ## Discretely allocated weights with shorting, and reinvesting
                       ## money made from shorting.
                       sar_s = portfolio.optimal[:sar].shares,
                       ## Discretely allocated weights with shorting, without
                       ## reinvesting money made from shorting.
                       sanr_w = portfolio.optimal[:sanr].weights,
                       ## Discretely allocated weights with shorting, without
                       ## reinvesting money made from shorting.
                       sanr_s = portfolio.optimal[:sanr].shares); formatters = fmt1)

#=
We can also check that our shorting requirements are being met. Not reinvesting the money immediately gained from shorting will ensure the portfolio keeps the leverage we've previously defined. If reinvesting is enabled, the portfolio will be more highly leveraged, and the allocated weights will add up to approximately `portfolio.long_u + portfolio.short_u`.

We show this in the next cell.
=#

## Without reinvesting the allocated short weights stay around the target of portfolio.short_u
isapprox(abs(sum(portfolio.optimal[:sanr].weights[portfolio.optimal[:sanr].weights .<= 0])),
         portfolio.short_u; rtol = 1e-2)

## Without reinvesting the allocated long weights stay around the target of portfolio.long_u
isapprox(abs(sum(portfolio.optimal[:sanr].weights[portfolio.optimal[:sanr].weights .>= 0])),
         portfolio.long_u; rtol = 1e-4)

## With reinvesting the allocated short weights stay around the target of portfolio.short_u
isapprox(abs(sum(portfolio.optimal[:sar].weights[portfolio.optimal[:sar].weights .<= 0])),
         portfolio.short_u; rtol = 1e-2)

## With reinvesting the allocated long weights stay around the target of portfolio.long_u + portfolio.short_u.
## This is because whatever money we've received from short-selling was invested into long positions.
## This obviously means the portfolio is more highly leveraged, so the default is to not reinvest.
isapprox(abs(sum(portfolio.optimal[:sar].weights[portfolio.optimal[:sar].weights .>= 0])),
         portfolio.long_u + portfolio.short_u; rtol = 1e-3)

#=
## 3.3 Greedy Allocation 

We'll repeat the process using greedy allocation. This allocation will never fail, but it may not yield the optimal solution to the allocation. However, it can allocate fractional shares. Lets say we can allocate up to 0.3 shares.
=#

## Our investment.
investment = 6750
## We'll buy fractional shares, we'll round down to the nearest `integer + 0.3 N`, where `N` is an integer.
alloc_method = Greedy(; rounding = 0.3)

## Allocating the long-only portfolio.

portfolio.optimal[:nsa] = allocate!(portfolio, :ns; method = alloc_method,
                                    investment = investment)

## Allocating the short-long portfolio without reinvesting the money we make from shorting.
portfolio.optimal[:sanr] = allocate!(portfolio, :s; method = alloc_method,
                                     investment = investment, reinvest = false)

## Allocating the short-long portfolio reinvesting the money we make from shorting.
portfolio.optimal[:sar] = allocate!(portfolio, :s; method = alloc_method,
                                    investment = investment, reinvest = true)

#=
We can print the long, short-long optimal portfolios, as well as the long, short-long without reinvesting, and short-long with reinvesting portfolios for comparison. Note how the short-long portfolio with reinvesting means we can purchase more long-shares.
=#
pretty_table(DataFrame(; tickers = portfolio.assets,
                       ## Optimal weights without shorting.
                       ns_w = portfolio.optimal[:ns].weights,
                       ## Optimal weights with shorting.
                       s_w = portfolio.optimal[:s].weights,
                       ## Discretely allocated optimal weights without shorting.
                       nsa_w = portfolio.optimal[:nsa].weights,
                       ## Discretely allocated shares without shorting.
                       nsa_s = portfolio.optimal[:nsa].shares,
                       ## Discretely allocated weights with shorting, and reinvesting
                       ## money made from shorting.
                       sar_w = portfolio.optimal[:sar].weights,
                       ## Discretely allocated weights with shorting, and reinvesting
                       ## money made from shorting.
                       sar_s = portfolio.optimal[:sar].shares,
                       ## Discretely allocated weights with shorting, without
                       ## reinvesting money made from shorting.
                       sanr_w = portfolio.optimal[:sanr].weights,
                       ## Discretely allocated weights with shorting, without
                       ## reinvesting money made from shorting.
                       sanr_s = portfolio.optimal[:sanr].shares); formatters = fmt1)

#=
We can also check that our shorting requirements are being met. Not reinvesting the money immediately gained from shorting will ensure the portfolio keeps the leverage we've previously defined. If reinvesting is enabled, the portfolio will be more highly leveraged, and the allocated weights will add up to approximately `portfolio.long_u + portfolio.short_u`. However, as previously stated, this is not a true optimum, thus the weights do not follow the expected value as closely.

We show this in the next cell.
=#

## Without reinvesting the allocated short weights stay around the target of portfolio.short_u, 
## but less accurately than the LP method.
isapprox(abs(sum(portfolio.optimal[:sanr].weights[portfolio.optimal[:sanr].weights .<= 0])),
         portfolio.short_u; rtol = 1e-1)

## Without reinvesting the allocated long weights stay around the target of portfolio.long_u,
## but less accurately than the LP method.
isapprox(abs(sum(portfolio.optimal[:sanr].weights[portfolio.optimal[:sanr].weights .>= 0])),
         portfolio.long_u; rtol = 1e-1)

## With reinvesting the allocated short weights stay around the target of portfolio.short_u,
## but less accurately than the LP method.
isapprox(abs(sum(portfolio.optimal[:sar].weights[portfolio.optimal[:sar].weights .<= 0])),
         portfolio.short_u; rtol = 1e-1)

## With reinvesting the allocated long weights stay around the target of portfolio.long_u + portfolio.short_u.
## This is because whatever money we've received from short-selling was invested into long positions.
## This obviously means the portfolio is more highly leveraged, so the default is to not reinvest.
isapprox(abs(sum(portfolio.optimal[:sar].weights[portfolio.optimal[:sar].weights .>= 0])),
         portfolio.long_u + portfolio.short_u; rtol = 1e-1)
