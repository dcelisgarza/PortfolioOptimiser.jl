The source files for all examples can be found in [/examples](https://github.com/dcelisgarza/PortfolioOptimiser.jl/tree/main/examples/).

```@meta
EditURL = "../../../examples/0_not_financial_advice.jl"
```

# Not Financial Advice

This is not financial advice, this is merely how I use the library for my own purposes. I typically use thousands of assets and whittle them down using something akin to this. I'm also using severely outdated pricing information and a meme stock for these examples. Use your own data and do your own research.

````@example 0_not_financial_advice
using PortfolioOptimiser, Clarabel, COSMO, CSV, CovarianceEstimation, DataFrames, GLPK,
      JuMP, OrderedCollections, Pajarito, Statistics, StatsBase, TimeSeries
````

The test data only contains a few stocks for convenience.

````@example 0_not_financial_advice
prices = TimeArray(CSV.File(joinpath(@__DIR__, "stock_prices.csv")); timestamp = :date);
nothing #hide
````

I like to use `Clarabel.jl` as a conic solver since it's quite fast and supports a wide variety of cones. For the times when `Clarabel.jl` fails, `COSMO.jl` usually succeeds. For MIP constraints, `Pajarito.jl` with `GLPK.jl` and `Clarabel.jl` has been my go-to.

````@example 0_not_financial_advice
solvers = OrderedDict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                        :params => Dict("verbose" => false,
                                                        "max_step_fraction" => 0.7)),
                      :COSMO => Dict(:solver => COSMO.Optimizer,
                                     :params => Dict("verbose" => false)),
                      :PClGL => Dict(:solver => optimizer_with_attributes(Pajarito.Optimizer,
                                                                          MOI.Silent() => true,
                                                                          "oa_solver" => optimizer_with_attributes(GLPK.Optimizer,
                                                                                                                   MOI.Silent() => true),
                                                                          "conic_solver" => optimizer_with_attributes(Clarabel.Optimizer,
                                                                                                                      "verbose" => false,
                                                                                                                      "max_step_fraction" => 0.7))));
nothing #hide
````

The allocation solver takes care of the discrete asset allocation at the end. There's also a greedy algorithm, that won't necessarily give the global optimum but can be a good fallback if any of the MIP optimisers fail.

````@example 0_not_financial_advice
alloc_solvers = Dict(:GLPK => Dict(:solver => GLPK.Optimizer,
                                   :params => Dict(MOI.Silent() => true)))
````

## Maximum Number of Assets Approach

This approach constrains the maximum number of assets in the portfolio to reduce the diversification.

First we create the portfolio instance.

````@example 0_not_financial_advice
portfolio = Portfolio(; prices = prices, solvers = solvers, alloc_solvers = alloc_solvers);
nothing #hide
````

I often have too many assets to start with (a few hundreds to thoudands), so I filter them first just to keep the best ones. There are multiple ways of doing this, the first is to use the maximum number of assets constraint. This may not be suitable when the number of assets is too large, and I haven't tried it with thousands of assets yet.

I like to minimise my drawdown risk so I like to minimise the relativistic drawdown at risk (:RDaR) with default parameters. Say we want to use only the top `x %` least risky assets. Since we only have 20 assets, we'll make this the top `25 %`. We can do this by setting the maximum number of assets to equal the number of assets divided by two.

````@example 0_not_financial_advice
portfolio.max_number_assets = div(length(portfolio.assets), 4)
````

Because this is a drawdown risk measure, and we're minimising the risk, we don't need to compute any statistics. We can optimise directly.

````@example 0_not_financial_advice
w = optimise!(portfolio, OptimiseOpt(; rm = :RDaR, obj = :Min_Risk))
````

We'll save the significant tickers for later.

````@example 0_not_financial_advice
idx = w.weights .>= 1e-6;
tickers_1 = Symbol.(w.tickers[idx]);
nothing #hide
````

We will use these weights later in this example.

## Hierarchical Clustering to Filter Assets

The approach I've used is to filter assets by putting them through hierarchical clustering optimisations with different downside risk measures. First we generate our hierarchical portfolio instance and an instance of our correlation/distnace matrix options.

````@example 0_not_financial_advice
hcportfolio = HCPortfolio(; prices = prices, solvers = solvers,
                          alloc_solvers = alloc_solvers);
nothing #hide
````

The Gerber 2 statistic is quite a robust covariance matrix, which is turned into a correlation and distance matrices within the `asset_statistics!` function.

````@example 0_not_financial_advice
cor_opt = CorOpt(; method = :Gerber2)
cluster_opt = ClusterOpt(; linkage = :ward)
````

Now lets define a function that leaves us with `x %` after removing the bottom `q'th` quantile `n` times. In other words, ``x = (1-q)^n``, solving for ``q``, ``q = 1 - \exp\left(\dfrac{1}{n}\log x\right)``.

````@example 0_not_financial_advice
gen_q(x, n) = 1 - exp(log(x) / n)
````

Why do we need this? Well, because we'll filter the stocks using a few risk measures, I like to use downside and drawdown risk measures. For our example we'll only use drawdown risk measures.

````@example 0_not_financial_advice
function get_best_tickers(tickers, x, rms)
    new_tickers = tickers
    q = gen_q(x, length(rms))
    for rm ∈ rms
        hp = HCPortfolio(; prices = prices[Symbol.(new_tickers)], solvers = solvers)
        asset_statistics!(hp; calc_mu = false, calc_cov = false, calc_kurt = false,
                          cor_opt = cor_opt)

        w = optimise!(hp; type = :HERC, rm = rm, cluster_opt = cluster_opt).weights

        qidx = w .>= quantile(w, q)
        new_tickers = new_tickers[qidx]
    end

    return Symbol.(new_tickers)
end

tickers_2 = get_best_tickers(hcportfolio.assets, 0.25, [:DaR, :CDaR, :EDaR, :RDaR])
````

`tickers_2` contains the filtered tickers.

Observant readers may have figured out that we can take a similar approach with the traditional portfolio without the need for constraining the maximum number of assets, it's just much more computationally intensive. One could also use a single step and simply take the top `q'th` percentile. We're simply showcasing a couple of simple approaches.

## Optimising and Allocating our Reduced Portfolio

Now we can do a couple of things, we can use `tickers_1` and/or `tickers_2` in different ways. We can use the union or the intersect. We'll use the union for this example.

````@example 0_not_financial_advice
tickers = union(tickers_1, tickers_2)
````

We'll use both a traditional portfolio and hierarchical one.

````@example 0_not_financial_advice
portfolio = Portfolio(; prices = prices[tickers], solvers = solvers,
                      alloc_solvers = alloc_solvers);
hcportfolio = HCPortfolio(; prices = prices[tickers], solvers = solvers,
                          alloc_solvers = alloc_solvers);
nothing #hide
````

In order for the clustering to work, we need to compute the correlation and distance matrices, so we need to call `asset_statistics!`. We'll use the robust correlation method that we defined earlier. We'll be using kelly returns and drawdown risk measures so we won't need anything else for either portfolio.

````@example 0_not_financial_advice
asset_statistics!(hcportfolio; calc_mu = false, calc_cov = false, calc_kurt = false,
                  cor_opt = cor_opt);
nothing #hide
````

Since we filtered the assets by minimising the risk measure, we can be more comfortable in using a different objective function. As such we'll maximise the risk-return (Sharpe) ratio using exact kelly returns, which have to be computed in accordance to the asset weights whilst being optimised, which is why we didn't need to compute the mean returns.

````@example 0_not_financial_advice
opt = OptimiseOpt(; rm = :RDaR, obj = :Sharpe, kelly = :Exact)
````

We'll be using both a traditional optimisation,

````@example 0_not_financial_advice
w1 = optimise!(portfolio, opt)
````

and a nested cluster optimisation (NCO), which is a series of traditional optimisations that make use of hierarchical clustering. This is a special optimisation for which we need to provide optimisation options to pass on to each sub-optimisation.

````@example 0_not_financial_advice
w2 = optimise!(hcportfolio; type = :NCO, nco_opt = opt, cluster_opt = cluster_opt)
````

## Discrete Allocation of Assets

Again we have a few things we can do here, we can either allocate each portfolio individually, or we can combine them in some way. We'll combine them into a single portfolio using the risk return ratio for RDaR. For this need the mean returns, the type doesn't matter because we'll be using the same for both and the ratio should cancel it out anyway.

````@example 0_not_financial_advice
asset_statistics!(portfolio; calc_mu = true, calc_cov = false, calc_kurt = false)
asset_statistics!(hcportfolio; calc_mu = true, calc_cov = false, calc_kurt = false,
                  calc_cor = false);
nothing #hide
````

We don't need to provide the portfolio type since it defaults to `:Trad` for `Portfolio` variables.

````@example 0_not_financial_advice
sr1 = sharpe_ratio(portfolio; rm = :RDaR)
````

We need to tell the sharpe ratio function to take the :NCO type because it defaults to `:HRP` for `HCPortfolio` variables.

````@example 0_not_financial_advice
sr2 = sharpe_ratio(hcportfolio; type = :NCO, rm = :RDaR)
````

The hierarchical optimisation will not have a sharpe ratio that is as large, but it usually leads to more robust portfolios, particularly if the correlation used is robust itself. So we'll take them as they are.

````@example 0_not_financial_advice
alpha = sr1 / (sr1 + sr2)
beta = 1 - alpha
````

Now we can take these values and use them to make a linear combination of the weights and renormalise.

````@example 0_not_financial_advice
weights3 = alpha * w1.weights + beta * w2.weights
weights3 ./= sum(weights3)
````

We can create a dataframe and then assign it to one of the portfolios, it doesn't matter which.

````@example 0_not_financial_advice
portfolio.optimal[:Combo] = DataFrame(; tickers = w1.tickers, weights = weights3)
````

Finally, we can discretely allocate the combined portfolio. We have to provide the `port_type` argument because for `Portfolio` it defaults to `:Trad` and for `HCPortfolio` to `:HRP`. We have to tell the function to take the assets and weights from `optimal[:Combo]`. We also need to provide how much money we can invest, the default is 10_000. Say we only have 2674 dollars.

!!! note
    
    The value of `investment` is not a currency, so you have to ensure the currency of the prices and currency of the investment match.

````@example 0_not_financial_advice
w4 = allocate!(portfolio; port_type = :Combo, investment = 2674)
````

This gives the portfolio that we can afford that minimises the difference between its weights and the ideal weights. By default, it assumes the asset price to be the last row of `prices`. Alternatively you can provide the keyword argument `asset_prices`, which takes a vector of prices where the `i'th` entry is the price of the `i'th` asset.

* * *

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*
