#=
# Example 0: Not financial advice

This example goes over a sample workflow using [`PortfolioOptimiser.jl`](https://github.com/dcelisgarza/PortfolioOptimiser.jl/).

## 1. Downloading the data

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
## 2. Filter worst stocks

If we have hundreds or thousands of stocks, we should probably do some pruning of the worst stocks using a cheap method. For this we'll use the [`HERC`](@ref) optimisation type. We'll filter the stocks using a few different risk measures. The order matters here, as each risk measure will filter out the worst performing stocks for each iteration.
=#

rms = [SD(), SSD(), CVaR(), CDaR(), Skew()]

# This tells us the bottom percentile we need to eliminate at each iteration so we have at most `x %` of the original stocks after `n` steps.
percentile_after_n(x, n) = 1 - exp(log(x) / n)

## Lets say we want to have 50% of all stocks at the end.
best = 0.5
## Copy the assets to a vector that will be shrunk at every iteration.
assets_best = copy(assets)
## Compute the bottom percentile we need to remove after each iteration.
q = percentile_after_n(best, length(rms))
## Lets use denoised and detoned covariance and correlation types so we can get rid of market forces. We're using the normal covariance as it's not very expensive to compute and we've made it more robust by denoising and detoning.
covcor_type = PortCovCor(; ce = CovFull(), denoise = DenoiseFixed(; detone = true))
## Loop over all risk measures.
for rm ∈ rms
    hp = HCPortfolio(; prices = prices[Symbol.(assets_best)])
    asset_statistics!(hp; cov_type = covcor_type, cor_type = covcor_type, set_kurt = false,
                      set_skurt = false, set_mu = false,
                      set_skew = isa(rm, Skew) ? true : false, set_sskew = false)
    w = optimise!(hp; type = HERC(), rm = rm,
                  hclust_opt = HCOpt(; k_method = StdSilhouette()))

    if isempty(w)
        continue
    end

    w = w.weights

    qidx = w .>= quantile(w, q)
    assets_best = assets_best[qidx]
end

# We can see that we end up with the best 11 stocks.

assets_best

# We can then use fancier optimisations and statistics with the smaller stock universe.

hp = HCPortfolio(;
                 prices = prices[Symbol.(["AAPL", "BBY", "DELL", "DG", "DRS", "LULU", "MCI",
                                          "MSFT", "NVDA", "PLNT", "SBUX", "SIRI", "STX"])],
                 ## Continuous optimiser.
                 solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                  :check_sol => (allow_local = true,
                                                                 allow_almost = true),
                                                  :params => Dict("verbose" => false,
                                                                  "max_step_fraction" => 0.85))),
                 ## MIP optimiser for the discrete allocation.
                 alloc_solvers = Dict(:HiGHS => Dict(:solver => HiGHS.Optimizer,
                                                     :check_sol => (allow_local = true,
                                                                    allow_almost = true),
                                                     :params => Dict("log_to_console" => false))))

covcor_type = PortCovCor(; ce = CorGerber1())
mu_type = MuBOP()
asset_statistics!(hp; cov_type = covcor_type, cor_type = covcor_type, mu_type = mu_type,
                  set_kurt = false, set_skurt = false, set_skew = false, set_sskew = false)

# We'll use the nested clustering optimisation. We will also use the near optimal centering type of portfolio for the intra- and inter-cluster optimisations with the risk adjusted return 
w = optimise!(hp;
              type = NCO(;
                         opt_kwargs = (; type = NOC(),
                                       obj = Sharpe(; rf = 3.5 / 100 / 254))), rm = RLDaR(),
              hclust_opt = HCOpt(; k_method = TwoDiff()))
