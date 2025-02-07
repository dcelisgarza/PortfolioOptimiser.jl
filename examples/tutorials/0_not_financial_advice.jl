# Copywrite (c) 2025
# Author: Daniel Celis Garza <daniel.celis.garza@gmail.com>
# SPDX-License-Identifier: MIT

#=
# Example 0: Not financial advice

This example goes over a sample workflow using [`PortfolioOptimiser.jl`](https://github.com/dcelisgarza/PortfolioOptimiser.jl/). I use a similar strategy myself. This is just an example of the things that can be done with the library.

## 0.1 Downloading the data

[`PortfolioOptimiser`](https://github.com/dcelisgarza/PortfolioOptimiser.jl) does not ship with supporting packages that are not integral to its internal functionality. This means users are responsible for installing packages to load and download data, [`JuMP`](https://jump.dev/JuMP.jl/stable/installation/#Supported-solvers)-compatible solvers, pretty printing, and the plotting functionality is an extension which requires [`GraphRecipes`](https://github.com/JuliaPlots/GraphRecipes.jl) and [`StatsPlots`](https://github.com/JuliaPlots/StatsPlots.jl).

Which means we need a few extra packages to be installed. Uncomment the first two lines if these packages are not in your Julia environment.
=#

## using Pkg
## Pkg.add.(["StatsPlots", "GraphRecipes", "YFinance", "Clarabel", "HiGHS", "PrettyTables"])
using Clarabel, DataFrames, Dates, GraphRecipes, HiGHS, YFinance, PortfolioOptimiser,
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
    if j != 5
        return v
    else
        return isa(v, Number) ? "$(round(v*100, digits=3)) %" : v
    end
end;

# We define our list of meme stonks and a generous date range. We will only be keeping the adjusted close price. In practice it doesn't really matter because we're using daily data.

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
TimeSeries.rename!(prices, Symbol.(assets))

#=
## 0.2 Filter worst stocks

If we have hundreds or thousands of stocks, we should probably do some pruning of the worst stocks using a cheap method. For this we'll use the [`HERC`](@ref) optimisation type. We'll filter the stocks using a few different risk measures. The order matters here, as each risk measure will filter out the worst performing stocks for each iteration.

First we need our filter functions.
=#

## This tells us the bottom percentile we need to eliminate at each iteration so we have at most `x %` of the original stocks after `n` steps.
percentile_after_n(x, n) = 1 - exp(log(x) / n)

function filter_best(assets, rms, best, cov_type, cor_type)
    ## Copy the assets to a vector that will be shrunk at every iteration.
    assets_best = copy(assets)
    ## Compute the bottom percentile we need to remove after each iteration.
    q = percentile_after_n(best, length(rms))
    ## Loop over all risk measures.
    for rm ∈ rms
        hp = Portfolio(; prices = prices[Symbol.(assets_best)])
        asset_statistics!(hp; cov_type = covcor_type, cor_type = covcor_type,
                          set_kurt = false, set_skurt = false, set_mu = false,
                          set_skew = isa(rm, NQSkew) ? true : false, set_sskew = false)
        cluster_assets!(hp; clust_opt = ClustOpt(; k_method = StdSilhouette()))
        w = optimise!(hp, HERC(; rm = rm))

        if isempty(w)
            continue
        end

        w = w.weights

        ## Only take the stocks above the q'th quantile at each step.
        qidx = w .>= quantile(w, q)
        assets_best = assets_best[qidx]
    end
    return assets_best
end

# Now we can define the parameters for our filtering procedure.

## Risk measures.
rms = [SD(), SSD(), CVaR(), CDaR(), NQSkew()]

## Lets say we want to have 50% of all stocks at the end.
best = 0.5

## Lets use denoised and detoned covariance and correlation types so we can get rid of market forces. We're using the normal covariance as it's not very expensive to compute and we've made it more robust by denoising and detoning.
covcor_type = PortCovCor(; ce = CovFull(), denoise = DenoiseFixed(; detone = true))

## Filter assets to only have the best ones.
assets_best = filter_best(assets, rms, best, covcor_type, covcor_type)

# We can see that we end up with the best 11 stocks.

assets_best

# We can now use fancier optimisations and statistics with the smaller stock universe.

hp = Portfolio(; prices = prices[Symbol.(assets_best)],
               ## Continuous optimiser.
               solvers = Dict(:Clarabel1 => Dict(:solver => Clarabel.Optimizer,
                                                 :check_sol => (allow_local = true,
                                                                allow_almost = true),
                                                 :params => Dict("verbose" => false))),
               ## MIP optimiser for the discrete allocation.
               alloc_solvers = Dict(:HiGHS => Dict(:solver => HiGHS.Optimizer,
                                                   :check_sol => (allow_local = true,
                                                                  allow_almost = true),
                                                   :params => Dict("log_to_console" => false))))

covcor_type = PortCovCor(; ce = CorGerber1())
mu_type = MuBOP()
asset_statistics!(hp; cov_type = covcor_type, cor_type = covcor_type, mu_type = mu_type,
                  set_kurt = false, set_skurt = false, set_skew = false, set_sskew = false)
cluster_assets!(hp; clust_opt = ClustOpt(; k_method = TwoDiff()))

# We'll use the nested clustering optimisation. We will also use the maximum risk adjusted return ratio objective function. We will also allocate the portfolio according to our availabe cash and the latest prices.

w = optimise!(hp,
              NCO(;
                  internal = NCOArgs(;
                                     type = Trad(; rm = RLDaR(),
                                                 obj = Sharpe(; rf = 3.5 / 100 / 252)))))

## Say we have 3000 dollars at our disposal to allocate the portfolio
wa = allocate!(hp, :NCO; investment = 3000)

pretty_table(w; formatters = fmt1)
pretty_table(wa; formatters = fmt2)

# However, we can do one better, we can take the worst performing stocks as well and short them. Since we're starting from so few stocks we'll adjust the best percentage to only take the best 30% after all filters.

function filter_worst(assets, rms, best, cov_type, cor_type)
    assets_worst = copy(assets)
    ## Compute the bottom percentile we need to remove after each iteration.
    q = percentile_after_n(best, length(rms))
    ## Loop over all risk measures.
    for rm ∈ rms
        hp = Portfolio(; prices = prices[Symbol.(assets_worst)])
        asset_statistics!(hp; cov_type = covcor_type, cor_type = covcor_type,
                          set_kurt = false, set_skurt = false, set_mu = false,
                          set_skew = isa(rm, NQSkew) ? true : false, set_sskew = false)
        cluster_assets!(hp; clust_opt = ClustOpt(; k_method = StdSilhouette()))
        w = optimise!(hp, HERC(; rm = rm))

        if isempty(w)
            continue
        end

        w = w.weights

        ## Only take the stocks below the (1-q)'th quantile at each step.
        qidx = w .<= quantile(w, 1 - q)
        assets_worst = assets_worst[qidx]
    end
    return assets_worst
end

# Now we can define the parameters for our filtering procedures.

## Risk measures.
rms = [SD(), SSD(), CVaR(), CDaR(), NQSkew()]

## Lets say we want to have 50% of all stocks at the end, 30% of the best, and 20% of the worst.
best = 0.3
worst = 0.2

## Lets use denoised and detoned covariance and correlation types so we can get rid of market forces. We're using the normal covariance as it's not very expensive to compute and we've made it more robust by denoising and detoning.
covcor_type = PortCovCor(; ce = CovFull(), denoise = DenoiseFixed(; detone = true))

## Filter assets to only have the best ones.
assets_best = filter_best(assets, rms, best, covcor_type, covcor_type)

## Filter assets to only have the worst ones.
assets_worst = filter_worst(assets, rms, worst, covcor_type, covcor_type)

## Lets join the best and worst tickers into a single vector.
assets_best_worst = union(assets_best, assets_worst)

# This time we'll make a market neutral portfolio using the NCO optimisation type.

hp = Portfolio(; prices = prices[Symbol.(assets_best_worst)],
               ## Continuous optimiser.
               solvers = Dict(:Clarabel1 => Dict(:solver => Clarabel.Optimizer,
                                                 :check_sol => (allow_local = true,
                                                                allow_almost = true),
                                                 :params => Dict("verbose" => false))),
               ## MIP optimiser for the discrete allocation.
               alloc_solvers = Dict(:HiGHS => Dict(:solver => HiGHS.Optimizer,
                                                   :check_sol => (allow_local = true,
                                                                  allow_almost = true),
                                                   :params => Dict("log_to_console" => false))))

covcor_type = PortCovCor(; ce = CorGerber1())
mu_type = MuBOP()
asset_statistics!(hp; cov_type = covcor_type, cor_type = covcor_type, mu_type = mu_type,
                  set_kurt = false, set_skurt = false, set_skew = false, set_sskew = false)
cluster_assets!(hp; clust_opt = clust_opt = ClustOpt(; k_method = TwoDiff()))

# For this we need to use the max ret objective and set the appropriate bounds on the asset weights.

## We need to set w_min and w_max weight constraints of the hierarchical clustering portfolio so the weights can be negative.
hp.w_min = -1
hp.w_max = 1

## The short parameters for the portfolios optimised via NCO.
short = true

## Absolute value of the sum of the short weights.
short_budget = -1

## Sum of all the portfolio weights.
budget = 0

## Upper bound for the value of each short weight.
short_lb = -1

## Upper bound for the value of each long weight.
long_ub = 1
w = optimise!(hp,
              NCO(;
                  internal = NCOArgs(; type = Trad(; rm = RLDaR(), obj = MaxRet()),
                                     port_kwargs = (; short = short, budget = budget,
                                                    short_budget = short_budget,
                                                    long_ub = long_ub, short_lb = short_lb))))

wa = allocate!(hp, :NCO; investment = 3000, short = short, budget = budget,
               short_budget = short_budget)

pretty_table(w; formatters = fmt1)
pretty_table(wa; formatters = fmt2)
