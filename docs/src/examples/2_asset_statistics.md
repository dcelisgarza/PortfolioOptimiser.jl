The source files for all examples can be found in [/examples](https://github.com/dcelisgarza/PortfolioOptimiser.jl/tree/main/examples/).

```@meta
EditURL = "../../../examples/2_asset_statistics!.jl"
```

# Example 2: Asset statistics - expected returns and covariance matrix

This tutorial follows from [Tutorial 1](https://github.com/dcelisgarza/PortfolioOptimiser.jl/blob/main/examples/0_basic_use.ipynb). If something in the preamble is confusing, it is explained there.

This tutorial focuses on the computation of asset statistics. This is one of the most important functions in [`PortfolioOptimiser`](https://github.com/dcelisgarza/PortfolioOptimiser.jl). It can be used for computing any and all statistics needed to optimise [`Portfolio`](@ref) and [`HCPortfolio`](@ref). It is also possible to define any and all statistics at variable instantiation, or by changing the relevant property in the variable, both cases will perform the necessary checks so the variables have the required characteristics.

## 1. Downloading the data

````@example 2_asset_statistics!
# using Pkg
# Pkg.add.(["StatsPlots", "GraphRecipes", "MarketData", "Clarabel", "HiGHS", "CovarianceEstimation", "SparseArrays"])
using Clarabel, CovarianceEstimation, DataFrames, Dates, GraphRecipes, HiGHS, MarketData,
      PortfolioOptimiser, Statistics, StatsBase, StatsPlots, TimeSeries, LinearAlgebra,
      SparseArrays

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
        # Yahoo doesn't like regular calls to their API.
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
````

## 2. Instantiating an instance of [`Portfolio`](@ref).

````@example 2_asset_statistics!
portfolio = Portfolio(; prices = prices,
                      # Continuous optimiser.
                      solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                       :check_sol => (allow_local = true,
                                                                      allow_almost = true),
                                                       :params => Dict("verbose" => false,
                                                                       "max_step_fraction" => 0.7))),
                      # MIP optimiser for the discrete allocation.
                      alloc_solvers = Dict(:HiGHS => Dict(:solver => HiGHS.Optimizer,
                                                          :check_sol => (allow_local = true,
                                                                         allow_almost = true),
                                                          :params => Dict("log_to_console" => false))));
nothing #hide
````

## 3 Asset statistics

When you first create a [`Portfolio`](@ref) in this way, it does not contain any statistics other than the returns. So we must compute them.

[`PortfolioOptimiser`](https://github.com/dcelisgarza/PortfolioOptimiser.jl) uses the [`StatsAPI.jl`](https://github.com/JuliaStats/StatsAPI.jl) interfaces through [`StatsBase.jl`](https://juliastats.org/StatsBase.jl/stable/). Meaning it is composable with other packages which use the common framework, and it also makes it easy for users to define their custom methods by using Julia's typesystem.

We'll only focus on the expected returns and covariance matrix. The default parameters are the arithmetic mean and sample covariance.

````@example 2_asset_statistics!
asset_statistics!(portfolio; set_kurt = false, set_skurt = false, set_skew = false,
                  set_sskew = false);

# Save these for later use.
mu1 = copy(portfolio.mu);
cov1 = copy(portfolio.cov);
nothing #hide
````

We can prove this by computing the arithmetic mean and sample covariance of the returns.

````@example 2_asset_statistics!
println(isapprox(mu1, vec(mean(portfolio.returns; dims = 1)))) # true
println(isapprox(cov1, cov(portfolio.returns; dims = 1))) # true
````

These statistics are not very robust, so they're not very reliable. We can make them a bit better by using weights. First we need to explain the estimators.

### 3.1 Mean estimators

Lets start with the easier one, [`MeanEstimator`](@ref). There are four of these, [`MuSimple`](@ref), [`MuJS`](@ref), [`MuBS`](@ref), [`MuBOP`](@ref). As you can see, they are all subtypes of [`MeanEstimator`](@ref), we will use this later on to define our own method. Lets first focus on the first estimator, which is also the default.

We've already seen its default behaviour, we know from above it's the same as the arithmetic mean. But it can take a vector of [`AbstractWeights`](https://juliastats.org/StatsBase.jl/stable/weights/).

First lets get the number of timestamps `T`, and number of assets `N`. We'll use `T` for defining our weights.

````@example 2_asset_statistics!
T, N = size(portfolio.returns)
````

There are a variety of weights, but the only ones that make sense with no prior knowledge are exponential weights. Now lets use this to compute the asset expected returns vector, we do this by passing the argument `mu_type = mu_type_1` to the function, we've also set the `set_cov = false` so it doesn't recompute the covariance.

````@example 2_asset_statistics!
# Play around with the value of lambda (1/T, in the example) to see the effect
# it has on the weights and computed expected returns vector.
w = eweights(1:T, 1 / T; scale = true)
mu_type_1 = MuSimple(; w = w)
asset_statistics!(portfolio; mu_type = mu_type_1, set_cov = false, set_kurt = false,
                  set_skurt = false, set_skew = false, set_sskew = false)
mu2 = copy(portfolio.mu)

println(isapprox(mu1, mu2)) # false
````

The other three estimators included in [`PortfolioOptimiser`](https://github.com/dcelisgarza/PortfolioOptimiser.jl) require a target and a covariance matrix, since they use these to correct the estimate of the arithmetic mean. The available targets are [`GM`](@ref), [`VW`](@ref), [`SE`](@ref), they all default to [`GM`](@ref). They can also take an [`AbstractWeights`](https://juliastats.org/StatsBase.jl/stable/weights/), which they will use to compute the arithmetic mean that is then corrected with the target and covariane matrix. We'll try a few combinations.

The covariance matrix is not needed, if it is empty, it will be computed by [`asset_statistics!`](@ref) from the parameters given to it via `cov_type` even if `set_cov = false`, it just won't replace the old covariance matrix with the one that's been computed for the mean estimator, once the calculation is done, the `sigma` field of the estimator will be set to empty once more. If a covariance matrix is provided, then [`asset_statistics!`](@ref) will use this rather than computing one for it.

Feel free to mix and match, and to play around with various combinations.

````@example 2_asset_statistics!
mu_type_2 = MuJS(; target = GM())
asset_statistics!(portfolio; mu_type = mu_type_2, set_cov = false, set_kurt = false,
                  set_skurt = false, set_skew = false, set_sskew = false)
mu3 = copy(portfolio.mu)

mu_type_3 = MuBS(; target = VW(), w = w)
asset_statistics!(portfolio; mu_type = mu_type_3, set_cov = false, set_kurt = false,
                  set_skurt = false, set_skew = false, set_sskew = false)
mu4 = copy(portfolio.mu)

# Using a custom covariance with random noise.
# It's not guaranteed to be positive definite.
noise = randn(N, N) / N^2
noise = transpose(noise) * noise
mu_type_4 = MuBOP(; target = SE(), sigma = cov1 + noise)
asset_statistics!(portfolio; mu_type = mu_type_4, set_cov = false, set_kurt = false,
                  set_skurt = false, set_skew = false, set_sskew = false)
mu5 = copy(portfolio.mu);
nothing #hide
````

All targets subtype [`MeanTarget`](@ref). It is possible for users to define a one by creating a concrete subtype of [`MeanTarget`](@ref) and defining a new [`target_mean`](@ref) for the custom target.

```
struct CustomMeanTarget <: MeanTarget
    ...
end
function target_mean(ct::CustomMeanTarget, mu::AbstractVector,
                     sigma::AbstractMatrix, inv_sigma,
                     T::Integer, N::Integer)
    ...
end
```

However, this limits the target to using the same data as the current ones. It's easier to define a new concrete subtype of [`MeanEstimator`](@ref). We will do this in the following section.

### 3.2 Defining a custom mean method

In order to define a new method all you need to do is create a new subtype of [`PortfolioOptimiser.MeanEstimator`](@ref) (it's not exported so it must be qualified) and define a new [`StatsBase.mean`](https://juliastats.org/StatsBase.jl/stable/scalarstats/#Weighted-sum-and-mean) function.

This is all we need, we can now define a custom mean that is the same as the [`MuSimple`](@ref), but scales the vector. You can scale the vector uniformly, by providing a scalar, or scale each item individually by providing an `AbstractVector`.

````@example 2_asset_statistics!
mutable struct MyScaledMean{T1, T2} <: PortfolioOptimiser.MeanEstimator
    scale::T1
    w::T2
end
function MyScaledMean(; scale::Union{<:AbstractVector{<:Real}, Real} = 1, w = nothing)
    return MyScaledMean{typeof(scale), typeof(w)}(scale, w)
end

# We have to turn this into a vec so we can scale by a vector.
function StatsBase.mean(me::MyScaledMean, X::AbstractArray; dims::Int = 1)
    return me.scale .*
           vec((isnothing(me.w) ? mean(X; dims = dims) : mean(X, me.w; dims = dims)))
end

scale = 5
mu_type_5 = MyScaledMean(; scale = scale)
asset_statistics!(portfolio; mu_type = mu_type_5, set_cov = false, set_kurt = false,
                  set_skurt = false, set_skew = false, set_sskew = false)
mu6 = copy(portfolio.mu)
# Should be a vector of 5's.
println(mu6 ./ mu1)

scale = 1:N
mu_type_6 = MyScaledMean(; scale = scale)
asset_statistics!(portfolio; mu_type = mu_type_6, set_cov = false, set_kurt = false,
                  set_skurt = false, set_skew = false, set_sskew = false)
mu7 = copy(portfolio.mu)
# Should be a vector going from 1 to N.
println(mu7 ./ mu1)
````

### 3.3 Covariance estimators

[`PortfolioOptimiser`](https://github.com/dcelisgarza/PortfolioOptimiser.jl) comes with quite a few covariance estimators. However, it is best to wrap them all with [`PortCovCor`](@ref). This is because it contains methods for denoising, fixing non-positive definite matrices, and using a graph-based algorithm for computing the covariance based on its relational structure.

[Portfoliooptimiser](https://github.com/dcelisgarza/PortfolioOptimiser.jl)'s mean and covariance estimators are based on the idea of subtyping [`StatsBase.CovarianceEstimator`](https://juliastats.org/StatsBase.jl/stable/cov/#StatsBase.CovarianceEstimator) to specialise their respective functions.

We will also not illustrate how to define custom methods as it follows the same principles as for the mean estimators, but instead by creating a concrete subtype (a struct) of [`StatsBase.CovarianceEstimator`](https://juliastats.org/StatsBase.jl/stable/cov/#StatsBase.CovarianceEstimator) and implementing [`StatsBase.cov`](https://juliastats.org/StatsBase.jl/stable/cov/#Statistics.cov-Tuple%7BCovarianceEstimator,%20AbstractMatrix%7D) for the custom type.

All estimators are different, some can nest other estimators, and those estimators may or may not support weights or even certain types of weights. There are so many different permutations that we will not go over an exhaustive list. For details on external methods please refer back to the documentation in their respective packages. For details on [`PortfolioOptimiser`](https://github.com/dcelisgarza/PortfolioOptimiser.jl/)'s estimators please refer to the docs.

As far as my recommendation/preference, I like the [`CorGerberSB1`](@ref) as it's fairly tuneable and produces less noisy matrices, but it is among the more expensive ones to compute.

````@example 2_asset_statistics!
# `corrected = true` does not support weights
ce0_a = StatsBase.SimpleCovariance(; corrected = false)

# Method from [CovarianceEstimation.jl](https://github.com/mateuszbaran/CovarianceEstimation.jl),
# does not support weights.
ce0_b = CovarianceEstimation.AnalyticalNonlinearShrinkage()

# Exponential weights, try something else if you wish.
w = eweights(1:T, 1 / sqrt(T); scale = true)

# Risk free rate, we will use as the minimum acceptable return for the semi covariance.
rf = 3.5 / 100 / 252
````

Lets put our estimators in a vector to make programming easier.

````@example 2_asset_statistics!
ces = CovarianceEstimator[]
# Full covariance.
push!(ces, CovFull(; ce = ce0_a, w = w))

# Semi covariance, only focuses on variations that take
# the returns below the target value.
push!(ces, CovSemi(; ce = ce0_b, target = rf))

# This bins the data and uses the information overlap between variables.
push!(ces, CorMutualInfo())

# This computes the distance covariance.
push!(ces, CorDistance())

# This uses how closely related the lower tail of the returns are between assets.
# It combines the idea behind the Value at Risk and rank correlation.
push!(ces, CorLTD())

# The Gerber family of robust covariance estimators that count
# co-movements and filters large and small movements by classifying
# them into zones. They are a rank-based (counting) approach.
push!(ces, CorGerber0())
push!(ces, CorGerber1())
push!(ces, CorGerber2())

# Smyth-Broby family of covariance estimators that modifies the
# Gerber statistics by scoring the quality of the classification
# of co-movements (instead of adding 1 they add a score).
push!(ces, CorSB0())
push!(ces, CorSB1())

# Smyth-Broby-Gerber family of covariance estimators that modifies
# the Gerber statistics by counting co-movements, scoring them,
# and then it multiplies the scores by the counts.
push!(ces, CorGerberSB0())
push!(ces, CorGerberSB1());
nothing #hide
````

We then instantiate some [`PortCovCor`](@ref) estimators and push them to a vector for convenience.

````@example 2_asset_statistics!
pces = PortCovCor[]
for ce ∈ ces
    push!(pces, PortCovCor(; ce = ce))
end
````

We can now call [`asset_statistics!`](@ref) with all these different estimators.

````@example 2_asset_statistics!
covs = Matrix[]
for pce ∈ pces
    asset_statistics!(portfolio; cov_type = pce, set_mu = false, set_kurt = false,
                      set_skurt = false, set_skew = false, set_sskew = false)
    push!(covs, copy(portfolio.cov))
end
````

Try changing `idx` 1 to 12, see how different the covariances look. What happens if you change some parameter in the covariance estimators? Are there methods that give you the most obvious ones? Try to see if you can spot patterns in the relationships between certain assets (ape together strong).

We're showing the correlation matrix since it makes it easier to see the differences. We've set the colour limits to go from 0 to 1 `clim = (0, 1)`, since all these assets appear to be postively correlated. Strictly speaking, only [`CorMutualInfo`](@ref), [`CorDistance`](@ref) and [`CorLTD`](@ref) are guaranteed to be only positive. More information on all the methods is found in the docs.

````@example 2_asset_statistics!
for idx ∈ 1:12
    display(plot(cov2cor(covs[idx]); st = :heatmap, clim = (0, 1),
                 yticks = (1:N, portfolio.assets), xticks = (1:N, portfolio.assets),
                 xrotation = 90, colorbar = true, xlim = (0.5, N + 0.5),
                 ylim = (0.5, N + 0.5), color = cgrad(:Spectral), yflip = true,
                 aspect_ratio = :equal))
end
````

### 3.4 Covariance denoising

Though the Gerber and its modified methods work well out of the box, other methods can benefit from extra processing. We'll use the default [`CovFull`](@ref) for this. First we will denoise it, for which we have three methods, [`DenoiseFixed`](@ref), [`DenoiseSpectral`](@ref), [`DenoiseShrink`](@ref), which use fixed, spectral and shrink denoising methods described in [MLAM; Chapter 2](@cite). Each denoise method contains various tuning parameters. We will only explore the effects of `detone` and `mkt_comp` and leave the rest as defaults.

````@example 2_asset_statistics!
ces_denoise = PortCovCor[]
push!(ces_denoise, PortCovCor(;))
push!(ces_denoise, PortCovCor(; denoise = DenoiseFixed()))
push!(ces_denoise, PortCovCor(; denoise = DenoiseSpectral()))
push!(ces_denoise, PortCovCor(; denoise = DenoiseShrink()))
covs_denoise = Matrix[]
for ce ∈ ces_denoise
    asset_statistics!(portfolio; cov_type = ce, set_mu = false, set_kurt = false,
                      set_skurt = false, set_skew = false, set_sskew = false)
    push!(covs_denoise, copy(portfolio.cov))
end
````

Try changing `idx` from 1 to 4 to see how the matrix changes.

````@example 2_asset_statistics!
for idx ∈ 1:4
    display(plot(cov2cor(covs_denoise[idx]); st = :heatmap, clim = (0, 1),
                 yticks = (1:N, portfolio.assets), xticks = (1:N, portfolio.assets),
                 xrotation = 90, colorbar = true, xlim = (0.5, N + 0.5),
                 ylim = (0.5, N + 0.5), color = cgrad(:Spectral), yflip = true,
                 aspect_ratio = :equal))
end
````

The market prices are not independent of each other, they are subject to market forces that affect all products. Hence why most assets have positive covariances with other assets. These market forces can wash out the true relationships between assets. To do this, we can `detone` the denoised matrix by removing the largest M eigenvalues. Typically, only the one largest is removed but we give the option for removing more. This operation however, can make the matrix singular. Which means it can't be used for mean variance optimisation, but can be more useful for clustering than a standard covariance. However, [`PortCovCor`](@ref) also contains an option for fixing non-positive definite correlation matrices, which may be able to make the detoned matrix non-singular. We will pass the argument `posdef = NoPosdef()` to ensure the matrices aren't fixed.

````@example 2_asset_statistics!
ces_detone = PortCovCor[]
push!(ces_detone, PortCovCor(;))

# `mkt_comp` defaults to 1 when `detone == true`, the value of `mkt_comp`
# defines how many of the largest eigenvalues are removed from the denoised matrix.
push!(ces_detone,
      PortCovCor(; posdef = NoPosdef(),
                 denoise = DenoiseFixed(; detone = true, mkt_comp = 1)))
push!(ces_detone,
      PortCovCor(; posdef = NoPosdef(),
                 denoise = DenoiseSpectral(; detone = true, mkt_comp = 1)))
push!(ces_detone,
      PortCovCor(; posdef = NoPosdef(),
                 denoise = DenoiseShrink(; detone = true, mkt_comp = 1)))
covs_detone = Matrix[]
for ce ∈ ces_detone
    asset_statistics!(portfolio; cov_type = ce, set_mu = false, set_kurt = false,
                      set_skurt = false, set_skew = false, set_sskew = false)
    push!(covs_detone, copy(portfolio.cov))
end
````

Try changing `idx` from 1 to 4 to see how the matrix changes. Try commenting out `posdef = NoPosdef()` from the cell above and see how it changes the output of this cell.

````@example 2_asset_statistics!
for idx ∈ 1:4
    display(plot(cov2cor(covs_detone[idx]); st = :heatmap, clim = (0, 1),
                 yticks = (1:N, portfolio.assets), xticks = (1:N, portfolio.assets),
                 xrotation = 90, colorbar = true, xlim = (0.5, N + 0.5),
                 ylim = (0.5, N + 0.5), color = cgrad(:Spectral), yflip = true,
                 aspect_ratio = :equal))
    println("idx = $idx, isposdef = $(isposdef(covs_detone[idx]))")
end
````

Denoising and detoning can be applied to any [`PortfolioOptimiserCovCor`](@ref) method, but others already have contingencies for capturing true signals. Denoising has to be applied with care, otherwise you risk washing out true effects. Denoising and detoning can be quite powerful for clustering assets, since it can remove market noise as well as systemic market effects.

### 3.5 LoGo covariance

The LoGo covariance uses graph theory to sparsify the inverse covariance matrix according to graph cliques and separators. This ensures only the strongest components of the realtionships are reflected in the covariance matrix. Although the inverse covariance is sparsified, the actual covariance is not, therefore it still encodes relationships between all assets.

The structure [`LoGo`](@ref) contains a distance and a similarity parameter, details can be found in its documentation. We will use the defaults, feel free to change them or create your by subtyping the relevant abstract types and defining the relevant methods.

````@example 2_asset_statistics!
ces_logo = PortCovCor[]
push!(ces_logo, PortCovCor(;))
push!(ces_logo, PortCovCor(; logo = LoGo(;)))

covs_logo = Matrix[]
for ce ∈ ces_logo
    asset_statistics!(portfolio; cov_type = ce, set_mu = false, set_kurt = false,
                      set_skurt = false, set_skew = false, set_sskew = false)
    push!(covs_logo, copy(portfolio.cov))
end
````

Try changing `idx` from 1 to 2 to see how the matrix changes.

````@example 2_asset_statistics!
for idx ∈ 1:2
    display(plot(cov2cor(covs_logo[idx]); st = :heatmap, clim = (0, 1),
                 yticks = (1:N, portfolio.assets), xticks = (1:N, portfolio.assets),
                 xrotation = 90, colorbar = true, xlim = (0.5, N + 0.5),
                 ylim = (0.5, N + 0.5), color = cgrad(:Spectral), yflip = true,
                 aspect_ratio = :equal))
end
````

If you invert `covs_logo[2]` and remove very small values (the covariance is recovered by inverting the inverse and so the would-be zeros are very small values), you can see the sparsity of the ivnerse covariance.

````@example 2_asset_statistics!
J = covs_logo[2] \ I
J[abs.(J) .<= 1e-10] .= zero(eltype(J))
sparse(J)
````

## Other statistics

There are other asset statistics that can be computed with [`asset_statistics!`](@ref). However, they do not use anything that has not been covered in this tutorial. For details on the statistics refer to the docs of [`asset_statistics!`](@ref).

* * *

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*
