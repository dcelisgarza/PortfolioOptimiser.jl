The source files for all examples can be found in [/examples](https://github.com/dcelisgarza/PortfolioOptimiser.jl/tree/main/examples/).

```@meta
EditURL = "../../../examples/4_hrp_portfolios.jl"
```

# Example 4: Hierarchical risk parity

This example follows from previous ones. If something in the preamble is confusing, it is explained there.

This example focuses on the [`HRP`](@ref) optimisation type of [`HCPortfolio`](@ref).

## 1. Downloading the data

````@example 4_hrp_portfolios
# using Pkg
# Pkg.add.(["StatsPlots", "GraphRecipes", "YFinance", "Clarabel", "HiGHS", "CovarianceEstimation", "SparseArrays"])
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
nothing #hide
````

## 2. Instantiating an instance of [`HCPortfolio`](@ref).

Since we're going to be performing [`HRP`](@ref) optimisations, we only need `solvers` for entropic and relativistic risk measures. Others don't make use of a solver, they can be computed from the asset statistics.

````@example 4_hrp_portfolios
portfolio = HCPortfolio(; prices = prices,
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

We will first use the most basic statistics. We'll later see how we can change the characteristics by changing them.

````@example 4_hrp_portfolios
cov_type = PortCovCor()
cor_type = PortCovCor()
dist_type = DistanceCanonical()
asset_statistics!(portfolio; cov_type = cov_type, cor_type = cor_type,
                  dist_type = dist_type)
````

# 3 Basic HRP portfolio

## 3.1 Hierarchical clustering

All [`HCPortfolio`]s use the assets' correlation structure to optimise the portfolios based on their correlation structure. [`PortfolioOptimiser`](https://github.com/dcelisgarza/PortfolioOptimiser.jl) comes with two clustering methods.

  - Hierarchical clustering using [`Clustering.jl`](https://github.com/JuliaStats/Clustering.jl).
  - [`Direct Bubble Hierarchy Trees`](https://uk.mathworks.com/matlabcentral/fileexchange/46750-dbht).

We'll use the default values for everything, see [`optimise!`](@ref) for details.

````@example 4_hrp_portfolios
# Standard deviation
rm = SD()

# Hierachical clustering with Ward's linkage.
hclust_alg = HAC(; linkage = :ward)
# Method for determining the number of clusters is the two-difference gap statistic [`TwoDiff`](@ref).
hclust_opt = HCOpt(; k_method = TwoDiff())
# Optimise.
w1 = optimise!(portfolio; rm = rm, hclust_alg = hclust_alg, hclust_opt = hclust_opt)
pretty_table(w1; formatters = fmt1)
````

[`HRP`](@ref) uses the clustering structure, but it splits the dendrogram naïvely down the middle. This means it can't take full advantage of the clustering structure, and may split closely related assets into separate clusters, which is not ideal. However, the next example will go over the [`HERC`](@ref) optimisation type, which does consider the clustering structure.

Regardless, we'll plot the clusters to see the structure of the relationships between assets. We don't want to clusterise again so we'll set `cluster = false`, which is also a flag in [`optimise!`](@ref), which saves on processing when the assets have been previously clusterised, the default is `cluster = true`.

````@example 4_hrp_portfolios
plot_clusters(portfolio; cluster = false)
````

Before moving on to DBHT clustering, we'll use a different linkage function. Generally speaking, Ward's clustering is the most robust when dealing with noisy data, which is why it's the default method. Lets see what complete clustering looks like. And now we'll use a different method for determining the clusters, [`StdSilhouette`](@ref).

````@example 4_hrp_portfolios
# Hierarchical clustering with complete linkage.
hclust_alg = HAC(; linkage = :complete)
# Method for determining the number of clusters is the Standard silhouette score [`StdSilhouette`](@ref).
hclust_opt = HCOpt(; k_method = StdSilhouette())
# Optimise.
w2 = optimise!(portfolio; rm = rm, hclust_alg = hclust_alg, hclust_opt = hclust_opt)
pretty_table(w2; formatters = fmt1)
plot_clusters(portfolio; cluster = false)
````

## 3.2 DBHT clustering

Direct Bubble Hierarchy Tree (DBHT) clustering, is a type of clustering based on graph-theoretic filtering. The same idea is used to compute the [`LoGo`](@ref) covariance, which we explored in [Example 2](https://github.com/dcelisgarza/PortfolioOptimiser.jl/blob/main/examples/2_asset_statistics.jl).

DBHT clustering also uses a similarity matrix, the original `MATLAB` code proposes two methods. [`PortfolioOptimiser`](https://github.com/dcelisgarza/PortfolioOptimiser.jl) implements them both using [`DBHTMaxDist`](@ref) and [`DBHTExp`](@ref). Though users can define their own creating a concrete subtype of [`PortfolioOptimiser.DBHTSimilarity`](@ref) and implementing [`dbht_similarity`](@ref) for it. There are also two methods for defining roots of graphs, either of which can be used [`UniqueDBHT`](@ref) and [`EqualDBHT`](@ref).

For now we will use teh defaults.

Again we will use default parameters first. We're not setting `cluster = flase` in this optimisation since we want the assets to be clustered using this new algorithm.

````@example 4_hrp_portfolios
# DBHT clustering, using the distance from the maximum value of the dissimilarity matrix [`DBHTMaxDist`](@ref).
hclust_alg = DBHT(; similarity = DBHTMaxDist())
hclust_opt = HCOpt(; k_method = TwoDiff())
w3 = optimise!(portfolio; rm = rm, hclust_alg = hclust_alg, hclust_opt = hclust_opt)
pretty_table(w3; formatters = fmt1)
````

We will again plot the clusters. Note how different the clusters are.

````@example 4_hrp_portfolios
plot_clusters(portfolio; cluster = false)
````

Now we'll see the effect changing the similarity matrix calculation to exponential decay of the disimilarity score.

````@example 4_hrp_portfolios
hclust_alg = DBHT(; similarity = DBHTExp())
hclust_opt = HCOpt(; k_method = TwoDiff())
w4 = optimise!(portfolio; rm = rm, hclust_alg = hclust_alg, hclust_opt = hclust_opt)
pretty_table(w4; formatters = fmt1)
plot_clusters(portfolio; cluster = false)
````

Now we'll define our own method for the similarity matrix. We'll use one of the potential definitions given in [`DBHTs`](@ref).

As a general rule, [`PortfolioOptimiser`](https://github.com/dcelisgarza/PortfolioOptimiser.jl) doesn't export abstract types, so they have to be explicitly imported.

We'll define our similarity matrix using only the correlation, denoted here by `S`, the distance matrix is denoted by `D` [`dbht_similarity`](@ref).

````@example 4_hrp_portfolios
struct DBHTClamp <: PortfolioOptimiser.DBHTSimilarity end
function PortfolioOptimiser.dbht_similarity(::DBHTClamp, S, D)
    return S .+ abs(minimum(S))
end
hclust_alg = DBHT(; similarity = DBHTClamp())
hclust_opt = HCOpt(; k_method = TwoDiff())
w5 = optimise!(portfolio; rm = rm, hclust_alg = hclust_alg, hclust_opt = hclust_opt)
pretty_table(w5; formatters = fmt1)
plot_clusters(portfolio; cluster = false)
````

We'll make another method that uses both the correlation and distance matrices to create the DBHT similarity matrix. We'll also make it tuneable.

````@example 4_hrp_portfolios
@kwdef mutable struct DBHTTuneableLinComboMaxDistExp{T1 <: Real, T2 <: Real, T3 <: Real} <:
                      PortfolioOptimiser.DBHTSimilarity
    maxdist_c::T1 = 1.0
    expdeca_c::T2 = 1.0
    argcoef::T3 = 0.5
end
function PortfolioOptimiser.dbht_similarity(DBHT::DBHTTuneableLinComboMaxDistExp, S, D)
    max_dist = DBHT.maxdist_c * PortfolioOptimiser.dbht_similarity(DBHTMaxDist(), S, D)
    exp_dec = exp.(-DBHT.argcoef * D)
    return max_dist + exp_dec
end
hclust_alg = DBHT(;
                  similarity = DBHTTuneableLinComboMaxDistExp(; maxdist_c = 0.3,
                                                              expdeca_c = 1, argcoef = 0.4))
hclust_opt = HCOpt(; k_method = TwoDiff())
w6 = optimise!(portfolio; rm = rm, hclust_alg = hclust_alg, hclust_opt = hclust_opt)
pretty_table(w6; formatters = fmt1)
plot_clusters(portfolio; cluster = false)
````

## 3.3 Using detoned matrices.

As mentioned in [Example 2](https://github.com/dcelisgarza/PortfolioOptimiser.jl/blob/main/examples/2_asset_statistics.jl), detoned matrices can be of great value in hierarchical optimisations. We'll see their effect here.

We will repeat the exact same steps as above, but without redefining the structures.

````@example 4_hrp_portfolios
cov_type = PortCovCor(; denoise = DenoiseFixed(; detone = true))
cor_type = PortCovCor(; denoise = DenoiseFixed(; detone = true))
dist_type = DistanceCanonical()
asset_statistics!(portfolio; cov_type = cov_type, cor_type = cor_type,
                  dist_type = dist_type)
````

First we try Ward's linkage.

````@example 4_hrp_portfolios
hclust_alg = HAC(; linkage = :ward)
hclust_opt = HCOpt(; k_method = TwoDiff())
w7 = optimise!(portfolio; rm = rm, hclust_alg = hclust_alg, hclust_opt = hclust_opt)
pretty_table(w7; formatters = fmt1)
plot_clusters(portfolio; cluster = false)
````

Then complete, and we'll see categorise the number of clusters according to the standard silhouette score.

````@example 4_hrp_portfolios
hclust_alg = HAC(; linkage = :complete)
hclust_opt = HCOpt(; k_method = StdSilhouette())
w8 = optimise!(portfolio; rm = rm, hclust_alg = hclust_alg, hclust_opt = hclust_opt)
pretty_table(w8; formatters = fmt1)
plot_clusters(portfolio; cluster = false)
````

Now we'll use DBHT clustering with max distance similarity.

````@example 4_hrp_portfolios
hclust_alg = DBHT(; similarity = DBHTMaxDist())
hclust_opt = HCOpt(; k_method = TwoDiff())
w9 = optimise!(portfolio; rm = rm, hclust_alg = hclust_alg, hclust_opt = hclust_opt)
pretty_table(w9; formatters = fmt1)
plot_clusters(portfolio; cluster = false)
````

Now we'll use the exponential decay similarity.

````@example 4_hrp_portfolios
hclust_alg = DBHT(; similarity = DBHTExp())
hclust_opt = HCOpt(; k_method = TwoDiff())
w10 = optimise!(portfolio; rm = rm, hclust_alg = hclust_alg, hclust_opt = hclust_opt)
pretty_table(w10; formatters = fmt1)
plot_clusters(portfolio; cluster = false)
````

Now we'll try the clamp method.

````@example 4_hrp_portfolios
hclust_alg = DBHT(; similarity = DBHTClamp())
hclust_opt = HCOpt(; k_method = TwoDiff())
w11 = optimise!(portfolio; rm = rm, hclust_alg = hclust_alg, hclust_opt = hclust_opt)
pretty_table(w11; formatters = fmt1)
plot_clusters(portfolio; cluster = false)
````

And finally the tuneable linear combination of max distance and exponential decay.

````@example 4_hrp_portfolios
hclust_alg = DBHT(;
                  similarity = DBHTTuneableLinComboMaxDistExp(; maxdist_c = 0.3,
                                                              expdeca_c = 1, argcoef = 0.4))
hclust_opt = HCOpt(; k_method = TwoDiff())
w12 = optimise!(portfolio; rm = rm, hclust_alg = hclust_alg, hclust_opt = hclust_opt)
pretty_table(w12; formatters = fmt1)
plot_clusters(portfolio; cluster = false)
````

As you can see, there are drastic differences in the correlation matrices. We'll display the weights of the sample  correlations and covariances (:weights) and their detoned counterparts (:weights_d) side by side.

````@example 4_hrp_portfolios
# Ward's linkage
pretty_table(DataFrames.rename!(hcat(w1, w7.weights), :x1 => :weights_d); formatters = fmt1)
# complete linkage
pretty_table(DataFrames.rename!(hcat(w2, w8.weights), :x1 => :weights_d); formatters = fmt1)
# DBHT max dist
pretty_table(DataFrames.rename!(hcat(w3, w9.weights), :x1 => :weights_d); formatters = fmt1)
# DBHT exp decay
pretty_table(DataFrames.rename!(hcat(w4, w10.weights), :x1 => :weights_d);
             formatters = fmt1)
# DBHT clamp (custom method)
pretty_table(DataFrames.rename!(hcat(w5, w11.weights), :x1 => :weights_d);
             formatters = fmt1)
# DBHT tuneable linear combination max dist exp decay
pretty_table(DataFrames.rename!(hcat(w6, w12.weights), :x1 => :weights_d);
             formatters = fmt1)
````

* * *

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*