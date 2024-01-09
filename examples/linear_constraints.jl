# # Linear Constraint Matrices

# This tutorial explains step by step how the constraint functions can be used to generate linear constraints. We also show how we can generate our own sets using clustering techniques.

using Clustering,
    CovarianceEstimation, CSV, DataFrames, PortfolioOptimiser, PrettyTables, TimeSeries

## This is a helper function for displaying tables 
## with numbers as percentages with 3 decimal points.
fmt = (v, i, j) -> begin
    if j == 1
        return v
    else
        return isa(v, Number) ? "$(round(v*100, digits=3)) %" : v
    end
end;

# Load the stock prices, get the asset names and compute the returns.

prices = TimeArray(CSV.File("./stock_prices.csv"); timestamp = :date)
assets = colnames(prices)
returns = dropmissing!(DataFrame(percentchange(prices)))
pretty_table(returns[1:5, :]; formatters = fmt)
returns = Matrix(returns[!, 2:end])

## Asset Constraints

# We can use predefined sets of assets, such as the sectors that are traditionally used to classify companies. Instead, we'll define our own classes using various clustering methods. These use a distance matrix to compute the clusters, for this we can use a correlation matrix. We provide various types of correlation matrices from which we'll pick a few examples. For a more thorough explanation of these methods see [`cor_mtx`](@ref).

# Note that returns data can be noisy which means some covariance/correlation measures can be thrown off from the true value by outliers and statistically insignificant noise. This can be problem for algorithms that depend on covariance/correlation matrices, such as mean-variance optimisation and clustering.

# In order to avoid this overfitting, we can compute robust covariances. In this case, we use the package [`CovarianceEstimation.jl`](https://github.com/mateuszbaran/CovarianceEstimation.jl), as well as some of the functions provided by `PortfolioOptimiser.jl`.

# We use the methods `:Pearson` and `:Semi_Pearson` compute the Pearson correlation, i.e. use `StatsBase.cor`, which takes an optional `StatsBase.CovarianceEstimator` type, for which we use `CovarianceEstimation.AnalyticalNonlinearShrinkage()`, as it's a good estimator for tall matrices (i.e. matrices with many more rows than columns). `:Semi_Pearson` uses the semi-covariance at a given target return value to compute the correlation matrix. We'll elaborate more on a different tutorial, for this we'll use the default, which is 0.

## Building the asset classes dataframe.
asset_classes = DataFrame(:assets => assets)
cor_settings = CorSettings(;
    estimation = CorEstSettings(;
        estimator = CovarianceEstimation.AnalyticalNonlinearShrinkage(),
        ## `target_ret` is used as the return threshold for classifying returns as
        ## sufficiently bad as to be considered unappealing, and thus taken into
        ## account when computing the semi covariance. The default is 0, but we
        ## can make it any number. A good choice is to use the daily risk-free 
        ## rate, because if an asset returns on average less than a bond, then 
        ## you'd be better off buying bonds because they're both less risky and
        ## more profitable than the asset.
        ## target_ret = 1.0329^(1 / 252) - 1,
    ),
)
for method in (:Pearson, :Semi_Pearson, :Gerber2)
    for linkage in (:ward, :DBHT)
        ## We define our asset sets based on the covariance and linkage methods.
        colname = Symbol(string(method)[1] * string(linkage)[1])

        ## Set the method for computing the covariance matrix.
        cor_settings.method = method

        ## Clusterise assets.
        clustering, k =
            cluster_assets(returns; linkage = linkage, cor_settings = cor_settings)

        ## Cut the tree at k clusters and return the label each asset belongs to
        ## in each set.
        asset_classes[!, colname] = cutree(clustering, k = k)
    end
end

asset_classes
