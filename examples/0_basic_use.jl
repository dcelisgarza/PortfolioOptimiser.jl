# # PortfolioOptimisation.jl Tutorial:

# ## Tutorial 0: Basic use

# This tutorial should serve as a minimum working example for using [`PortfolioOptimiser.jl`](https://github.com/dcelisgarza/PortfolioOptimiser.jl/).

# ### 1. Downloading the data

#=
[`PortfolioOptimiser`](https://github.com/dcelisgarza/PortfolioOptimiser.jl) does not ship with supporting packages that are not integral to its internal functionality. This means users are responsible for installing packages to load and download data, [`JuMP`]()-compatible solvers, pretty printing, and the plotting functionality is an extension which requires [`GraphRecipes`](https://github.com/JuliaPlots/GraphRecipes.jl) and [`StatsPlots`](https://github.com/JuliaPlots/StatsPlots.jl).

Which means we need a few extra packages to be installed. Uncomment the first two lines if these packages are not in your Julia environment.
=#

## using Pkg
## Pkg.add.(["StatsPlots", "GraphRecipes", "MarketData", "Clarabel", "HiGHS", "PrettyTables"])
using Clarabel, DataFrames, Dates, GraphRecipes, HiGHS, MarketData, PortfolioOptimiser,
      PrettyTables, Statistics, StatsBase, StatsPlots, TimeSeries

## These are helper functions for formatting tables.
fmt1 = (v, i, j) -> begin
    if j == 1
        return v
    else
        return isa(v, Number) ? "$(round(v*100, digits=3)) %" : v
    end
end;

fmt2 = (v, i, j) -> begin
    if j == 5
        return isa(v, Number) ? "$(round(v*100, digits=3)) %" : v
    else
        return v
    end
end;

fmt3 = (v, i, j) -> begin
    if j ∈ (2, 6, 7)
        return isa(v, Number) ? "$(round(v*100, digits=3)) %" : v
    else
        return v
    end
end;

# We define our list of meme stonks and a generous date range. We will only be keeping the adjusted close price. In practice it doesn't really matter because we're using daily data.

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

# ### 2. Instantiating an instance of [`Portfolio`](@ref).

# Now that we have our data we can instantiate a portfolio. We also need to give it an optimiser for the continuous optimisation and an MIP optimiser for the discrete allocation of funds, we'll use [`Clarabel.jl`](https://github.com/oxfordcontrol/Clarabel.jl) and [`HiGHS.jl`](https://github.com/jump-dev/HiGHS.jl).

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
                                                          :params => Dict("log_to_console" => false))))

# The constructor automatically computes the returns, sets the assets, and timestamps if you give it the price data. Users can also provide these directly, the timestamps aren't needed anywhere but plotting so they are not required. This structure contains a lot of data. But we will only show the basics for now.

# ### 3. Computing statistics

#=
There are myriad of statistics and methods for computing said statistics. Users can define their own methods with Julia's multiple dispatch and [`StatsAPI.jl`](https://github.com/JuliaStats/StatsAPI.jl) interface.

We will do a simple Mean-Variance optimisation, using the simplest methods for computing the expected returns vector and covariance matrices. Later tutorials will go more in-depth.
=#

## T is the number of observations, N the number of assets.
T, N = size(portfolio.returns)

# Simple mean, it also accepts weights for computing weighted means. This will compute the weighted mean for lambda = 1/T.

mu_type = MuSimple(;)
## mu_type = MuSimple(; w = eweights(1:T, 1/T, scale=true))

#=
[`PortCovCor`](@ref) is quite special because it accepts covariance estimators, methods for fixing non positive definite matrices, denoising methods, and a graph-based approach for computing the covariance by identifying related asset groups. 

The estimators are all different, and PortfolioOptimiser offers a few of them. They all have their own parameters. They all subtype [`StatBase.CovarianceEstimator`](https://juliastats.org/StatsBase.jl/stable/cov/#StatsBase.CovarianceEstimator), meaning [`PortCovCor`](@ref) works with external estimators such as those in [`CovarianceEstimation.jl`](https://github.com/mateuszbaran/CovarianceEstimation.jl), as well as user-defined methods.

This defaults to the full sample covariance estimator. We can provide the estimator with some weights to compute a weighted covariance too. We'll leave other estimators for future tutorials. For the time being, feel free to play around with the weights. There is some nesting involved, but that is due to the fact that we are composing various estimators to get our desired outcome.
=#

cov_type = PortCovCor(;)
## cov_type = PortCovCor(;
##                       ce = CovFull(; ce = StatsBase.SimpleCovariance(; corrected = false),
##                                    w = eweights(1:T, 1 / T; scale = true)))

# We can then call [`asset_statistics!`](@ref) which computes all the asset statistics. It will also compute other statistics by default but we can set some flags to stop this from happening.

asset_statistics!(portfolio; mu_type = mu_type, cov_type = cov_type, set_kurt = false,
                  set_skurt = false, set_skew = false, set_sskew = false);

# ### 4. Optimising the portfolio

#=
We will only look at a vanilla optimisation in this tutorial. 

There are quite a few risk measures, some require statistics we have not computed, others don't require any precomputed statistics at all. You can see the risk measure has a few internal parameters, they all do. We'll only show the classic mean variance risk measure, but you can uncomment each of the next few lines in turn and try some of the others.
=#

rm = SD() # Variance
## rm = MAD() # Mean absolute deviation
## rm = SSD() # Semi variance
## rm = CVaR() # Critical Value at Risk
## rm = CDaR() # Critical Drawdown at Risk

# There are four objective functions we can use they all serve their purpose but for the tutorial, we'll be minimising the risk. Try the other objective functions and change the parameters to see their effects.

obj = MinRisk()
## obj = MaxRet() # Only useful for maximising the return while constraining the risk to be under a given value.
## obj = Utility(; l = 2) # Maximises the utility = return - l * risk
## obj = Sharpe(; rf = 3.5/100/254) # Maximises the sharpe ratio = (mu - rf)/risk, where mu is the expected return.

w1 = optimise!(portfolio; rm = rm, obj = obj)
pretty_table(w1; formatters = fmt1)

# We now have the portfolio for these assets and this period of time that minimises the variance. [`PortfolioOptimiser`](https://github.com/dcelisgarza/PortfolioOptimiser.jl/) also lets you constrain the optimisation such that you have a minimum required return. However this will be explored in a later tutorial. 

# ### 5. Asset allocation

#=
For now we want to know how many assets we need to buy, this only gives us the mathematically optimal weights. We have a function for this. Given that we used the price data directly, [`Portfolio`](@ref) will take the last entry in the prices and use that as the current price for each asset. You can of course change this, or directly provide them as a vector to the [`allocate!`](@ref) function, though the order of the prices must be the same as the original asset order.

For this example, lets say we have 1000 dollars. Change this value to see how the allocation changes. The larger it becomes, the closer they get to the optimal value.
=#

investment = 1000;

# There are two methods we can use for allocating. The one we have chosen uses mixed-integer linear programming to discretly allocate assets. This however can only allocate discrete quantities. The second method is a greedy algorithm that allocates based on which asset has the highest difference between its mathematical optimum and current weight, it only tries to allocate based on whether you can afford it. However, it can allocate fractional shares up to an integer multiple of the value in rounding (defaults to 1). Given it is a greedy algorithm, it is not guaranteed to give an optimum solution, but will always find a solution, unlike the LP() method which can fail. Also the higher the investment, the more accurate it gets.

method = LP()
## method = Greedy(; rounding=0.5)

w2 = allocate!(portfolio; investment = investment)
pretty_table(w2; formatters = fmt2)

# ### 6. Plotting the portfolio

# This section is the one most bound to change as the plotting functions are still somewhat preliminary. There are a variety of plots but we'll only show the most basic ones. By default, the plots will take the mathematically optimal portfolio. Lets see what it looks like.

## Plot the portfolio returns.
display(plot_returns(portfolio))

## Plot the portfolio returns per asset.
display(plot_returns(portfolio; per_asset = true))

## Plot portfolio composition.
display(plot_bar(portfolio))

## Plot the returns histogram.
display(plot_hist(portfolio))

## Plot the portfolio range of returns.
display(plot_range(portfolio))

## Plot portfolio drawdown.
display(plot_drawdown(portfolio))

# This is, however not our actual portfolio, it is the optimal one. To plot the allocated portfolio we need to know the key it is stored under and pass that on to the plotting functions along with a flag. The key is the symbol composed of the allocation method, in this case LP() and the portfolio type, which is something we have not discussed, but defaults to Trad(), as a Symbol.

## Side by side optimal portfolio vs allocated portfolio.
pretty_table(hcat(portfolio.optimal[:Trad],
                  DataFrames.rename!(portfolio.alloc_optimal[:LP_Trad][!, 2:end],
                                     Dict(:weights => :alloc_weights)),
                  DataFrame(;
                            weight_diff = portfolio.optimal[:Trad].weights -
                                          portfolio.alloc_optimal[:LP_Trad].weights));
             formatters = fmt3)

## Plot the portfolio returns.
display(plot_returns(portfolio, :LP_Trad; allocated = true))

## Plot the portfolio returns per asset.
display(plot_returns(portfolio, :LP_Trad; allocated = true, per_asset = true))

## Plot portfolio composition.
display(plot_bar(portfolio, :LP_Trad; allocated = true))

## Plot the returns histogram.
display(plot_hist(portfolio, :LP_Trad; allocated = true))

## Plot the portfolio range of returns.
display(plot_range(portfolio, :LP_Trad; allocated = true))

## Plot portfolio drawdown.
display(plot_drawdown(portfolio, :LP_Trad; allocated = true))
