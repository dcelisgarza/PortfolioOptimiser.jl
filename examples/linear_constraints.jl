# # Linear asset constraints

# This tutorial explains step by step how the constraint functions can be used to generate linear constraints. We also show how we can generate our own sets using clustering techniques.

using Clustering, CovarianceEstimation, CSV, DataFrames, LinearAlgebra, PortfolioOptimiser,
      PrettyTables, TimeSeries

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
## We'll be sorting the assets by alphabetical order.
assets = sort(colnames(prices))
## Make sure we index the prices by the sorted assets to maintain consistency.
returns = dropmissing!(DataFrame(percentchange(prices[assets])))
pretty_table(returns[1:5, :]; formatters = fmt)
returns = Matrix(returns[!, 2:end]);

#=
!!! warning
    It is important that the asset order remains consistent accross all steps. If one wishes to sort the assets in a particular way, one should sort them immediately after importing them. This ensures everything is computed using the correct asset order.
=#

# ## Defining asset sets with hierachical clustering

# We can use predefined sets of assets, such as the economic sectors that are traditionally used to classify companies. Instead, we'll define our own sets using various clustering methods. These use a distance matrix to compute the clusters, for this we can use a correlation matrix. We provide various types of correlation matrices from which we'll pick a few examples. For a more thorough explanation of these methods see [`cor_dist_mtx`](@ref).

# Note that returns data can be noisy which means some covariance/correlation measures can be thrown off from the true value by outliers and statistically insignificant noise. This can be problem for algorithms that depend on covariance/correlation matrices, such as mean-variance optimisation and clustering.

# In order to avoid this overfitting, we can compute robust covariances. In this case, we use the package [`CovarianceEstimation.jl`](https://github.com/mateuszbaran/CovarianceEstimation.jl), as well as some of the functions provided by `PortfolioOptimiser.jl`.

# When we use `:Pearson` and `:Semi_Pearson` to compute the correlation, we're internally making use of `StatsBase.cor`, which takes various range of optional arguments, one of which is a covariance estimator, `StatsBase.CovarianceEstimator`, which makes the computation of the covariance matrix more robust. In this example we use `CovarianceEstimation.AnalyticalNonlinearShrinkage()`, as it's a good estimator for tall matrices (i.e. matrices with many more rows than columns). `:Semi_Pearson` also uses a semi-covariance at a given target return value to compute the correlation matrix. We'll elaborate more on a different tutorial, for this we'll use the default value of the target return, which is `0`.

# ### Building the asset sets dataframe.
asset_sets = DataFrame(:Asset => assets)
cor_settings = CorOpt(;
                      estimation = CorEstOpt(; estimator = AnalyticalNonlinearShrinkage(),
                                             ## `target_ret` is used as the return threshold for 
                                             ## classifying returns as sufficiently bad as to be
                                             ## considered unappealing, and thus taken into 
                                             ## account when computing the semi covariance. The 
                                             ## default is 0, but we can make it any number. A 
                                             ## good choice is to use the daily risk-free rate,
                                             ## because if an asset returns on average less than
                                             ## a bond, then you'd be better off buying bonds
                                             ## because they're both less risky and more profitable
                                             ## than the asset.
                                             ## target_ret = 1.0329^(1 / 252) - 1,
                                             ),)
for method ∈ (:Pearson, :Semi_Pearson, :Gerber2)
    for linkage ∈ (:ward, :DBHT)
        ## We define our asset sets based on the covariance and linkage methods.
        colname = Symbol(string(method)[1] * string(linkage)[1])

        ## Set the method for computing the covariance matrix.
        cor_settings.method = method

        ## Clusterise assets.
        clustering, k = cluster_assets(returns; linkage = linkage,
                                       cor_settings = cor_settings)

        ## Cut the tree at k clusters and return the label each asset belongs to
        ## in each set.
        asset_sets[!, colname] = cutree(clustering; k = k)
    end
end

asset_sets

# ## Single asset constraints

# ### Constraining the weight contribution of a single asset

# As previously mentioned, assets are categorised on their industry sectors, but this way we can create our own in a data-driven way. We'll use these to build our linear constraints. We'll be building them one type of constraint by one, but you can also create them all in one go by concatenating the vectors.

# Say we want to ensure we have at least 3% of our portfolio allocated to `GOOG`, ``w_{\mathrm{goog}} \geq 0.03``.

constraints = DataFrame(
                        ## Enable the constraint.
                        :Enabled => [true],
                        ## We are constraining a single asset.
                        :Type => ["Asset"],
                        ## The asset we're constraining.
                        :Position => [:GOOG],
                        ## Greater than.
                        :Sign => [">="],
                        ## Lower bound of the weight of the asset.
                        :Weight => [0.03],
                        ## The following categories are not used for this
                        ## example.
                        :Set => [""], :Relative_Type => [""], :Relative_Set => [""],
                        :Relative_Position => [""], :Factor => [""])

A, B = asset_constraints(constraints, asset_sets);

# Recall that the linear constraints are defined as ``\mathbf{A} \bm{w} \geq \bm{b}``, we can show that the constraint will be applied to `GOOG`, as its weight coefficient is 1, while all the others are `0`.

hcat(asset_sets[!, :Asset], DataFrame(:A_t => vec(A)))

# Using the definition of the linear constraints, `B` should be `0.03`.

B

# Substituting these values into the linear constraint definition and removing all zero values, we arrive at the constraint ``w_{\mathrm{goog}} \geq 0.03``.

# Say we also don't want our portfolio to be more than 10 % `GOOG`. Here's how we can do so. Note how the only things that change are `:Sign` and `:Weight`.

constraints = DataFrame(:Enabled => [true],
                        ## Constraining a single asset.
                        :Type => ["Asset"],
                        ## Asset we will be constraining.
                        :Position => [:GOOG],
                        ## Less than or equal to.
                        :Sign => ["<="],
                        ## Upper bound of the weight of the asset.
                        :Weight => [0.1],
                        ## The following categories are not used for this
                        ## example.
                        :Set => [""], :Relative_Type => [""], :Relative_Set => [""],
                        :Relative_Position => [""], :Factor => [""])

A, B = asset_constraints(constraints, asset_sets);

# In practice, we would create all constraints at once, but this is a showcase so we'll be making them one by one. One can also do that, and simply vertically concatenate `A` and `B` with new constraints, which is literally what the function does.

# The sign in the `A` matrix will be inverted.

hcat(asset_sets[!, :Asset], DataFrame(:A_t => vec(A)))

# And so will the sign of the `B` vector.

B

# By substituting these values into the definition of the linear constraints above, removing all zero values, and rearranging, ``-w_{\mathrm{goog}} \geq -0.1 \to 0.1 \geq w_{\mathrm{goog}}``. This sign inversion works the same for every constraint type, if the constraint denotes an upper bound (the corresponding value in `:Sign` is `<=`), both sides of the constraint are multiplied by `-1`.

# ### Constraining the weight contribution of a single asset relative to another one

# We can also constrain an asset with respect to another one. For example, say we wish to ensure `:AAPL` contributes at least a factor of what `:T` contributes. Mathematically, the constraint can be written as ``w_{\mathrm{aapl}} \geq f w_{\mathrm{t}}``, where ``f`` is a scalar factor. Here we use `f = 0.5`

constraints = DataFrame(:Enabled => [true],
                        ## We want to constrain an asset.
                        :Type => ["Asset"],
                        ## The asset we want to constrain.
                        :Position => [:AAPL],
                        ## The the weight of the asset should 
                        ## be greater than or equal to something else.
                        :Sign => [">="],
                        ## Should be greater than another asset.
                        :Relative_Type => ["Asset"],
                        ## The name of the other asset.
                        :Relative_Position => [:T],
                        ## The factor by which to multiply the
                        ## weight of the relative asset.
                        :Factor => [0.5],
                        ## The following categories are not used 
                        ## in this example.
                        :Weight => [""], :Set => [""], :Relative_Set => [""])

A, B = asset_constraints(constraints, asset_sets);

# Recalling the definition of the linear constraints, as well as the constraint we want to create, we can conclude that `A` will contain all zero coefficients except for those of `:AAPL` and `:T`. These two will be `1` and `-f = -0.5`, respectively.
hcat(asset_sets[!, :Asset], DataFrame(:A_t => vec(A)))

# Naturally `B` should be 0.

B

# This pattern of `A` and `B` is the case for all relative constraints. The `anchor` asset/subset will have its coeficient set to 1, the relative asset/set will be set to `-f`, and `B` will be 0. If the sign is `<=`, the values will be multiplied by -1. To illustrate this, we will constrain `:AMZN` to contribute at most twice of what `:GM` does.

constraints = DataFrame(:Enabled => [true],
                        ## We want to constrain an asset.
                        :Type => ["Asset"],
                        ## The asset we want to constrain.
                        :Position => [:AMZN],
                        ## The the weight of the asset should 
                        ## be less than or equal to something else.
                        :Sign => ["<="],
                        ## Should be greater than another asset.
                        :Relative_Type => ["Asset"],
                        ## The name of the other asset.
                        :Relative_Position => [:GM],
                        ## The factor by which to multiply the
                        ## weight of the other asset.
                        :Factor => [2],
                        ## The following categories are not used
                        ## in this example.
                        :Weight => [""], :Set => [""], :Relative_Set => [""])

A, B = asset_constraints(constraints, asset_sets);

# Again B is equal to `0`.

B

# See how the coefficient of `:AMZN` is `-1` while the one for `:GM` is `2`. When substituting into the linear constraint definition, noting that `B` is equal to `0`, we get ``-w_{\mathrm{amzn}} + 2 w_{\mathrm{gm}} \geq 0 \to 2 w_{\mathrm{gm}} \geq w_{\mathrm{amzn}}.``

hcat(asset_sets[!, :Asset], DataFrame(:A_t => vec(A)))

# ### Constraining the weight contribution of a single asset relative to a subset of assets

# We can also constrain a single asset relative to a whole asset set. This is essentially the same as the previous case. But instead of constraining the weight of the `anchor` asset to some factor of the weight of another one, we constrain it to some factor multiplied by the sum of the weights of a subset of assets. Mathematically, this is ``w_{x} - f \sum \bm{w}_{\mathcal{\Omega}} \geq 0``, when the constraint is a lower bound, and ``-\left(w_{x} - f \sum\left(\bm{w}_{\mathcal{\Omega}}\right)\right) \geq 0`` when it is an upper bound. Where ``x`` is an asset, ``\mathcal{\Omega}`` is a subset of assets, and ``f`` is the relative factor.

# We will constrain the weight of asset `:T` to have a lower bound greater than or equal to `0.03` times the sum of the weights of subset `3` of set `:PD`.

constraints = DataFrame(:Enabled => [true],
                        ## Constraining a single asset.
                        :Type => ["Asset"],
                        ## Asset we want to constrain.
                        :Position => [:T],
                        ## Constraining the weight of the asset to be
                        ## greater than or equal to something else.
                        :Sign => [">="],
                        ## Constraining relative to a subset of assets.
                        :Relative_Type => ["Subset"],
                        ## The set we want the subset to be taken from.
                        :Relative_Set => [:PD],
                        ## The subset we want to use as a relative constrain.
                        :Relative_Position => [3],
                        ## The factor to multiply the weights of the asset
                        ## subset.
                        :Factor => [0.03],
                        ## The following categories are not used
                        ## in this example.
                        :Set => [""], :Weight => [""])

A, B = asset_constraints(constraints, asset_sets);

# As expected, `B` is equal to `0`.

B

# And if we place `A` next to the asset set dataframe and the set `:PD`, we see that the only assets with non-zero coefficients in `A`, are the `anchor` and the `subset` from `:Pw` with label `3`. Of note is the fact that `:T` has a coefficient of `0.97`, this is because it belongs to subset `3` of set `:Pw`, so its coefficient will be `1 - f`, and as per above `f := 0.03`. Substituting and rearranging into the definition of the linear constraints we get ``0.97 w_{x} \geq \sum \bm{w}_{\mathcal{\Omega} \setminus x}``, where `x := :T` and `Ω := 3 ∈ {:PD}`.

hcat(asset_sets[!, [:Asset, :PD]], DataFrame(:A_t => vec(A)))

# Now we will show how the sign is inverted when we're defining an upper bound. We will constrain the weight of asset `:T` to have an upper bound less than or equal to `0.4` times the sum of the weights of subset `1` of set `:SD`.

constraints = DataFrame(:Enabled => [true],
                        ## Constraining a single asset.
                        :Type => ["Asset"],
                        ## Asset we want to constrain.
                        :Position => [:T],
                        ## Constraining the weight of the asset to be
                        ## less than or equal to something else.
                        :Sign => ["<="],
                        ## Constraining relative to a subset of assets.
                        :Relative_Type => ["Subset"],
                        ## The set we want the subset to be taken from.
                        :Relative_Set => [:SD],
                        ## The subset we want to use as a relative constrain.
                        :Relative_Position => [1],
                        ## The factor to multiply the weights of the asset
                        ## subset.
                        :Factor => [0.4],
                        ## The following categories are not used
                        ## in this example.
                        :Set => [""], :Weight => [""])

A, B = asset_constraints(constraints, asset_sets);

# Again `B` is equal to `0`.

B

# And as expected, the coefficient of `:T` is `-1`, and the coefficients of all assets belonging to subset `1` in set `:SD` are `0.4`. Substituting into the definition of the linear constraints and rearranging, we get ``\sum \bm{w}_{\mathcal{\Omega}} \geq w_{x}``, where `x := :T` and `Ω := 1 ∈ {:SD}`.

hcat(asset_sets[!, [:Asset, :SD]], DataFrame(:A_t => vec(A)))

# ## All assets constraints

# We can constrain all assets simultaneously. This is equivalent to constraining single assets one at a time. Here we constrain all assets to contribute at least `0.01` to the portfolio.

constraints = DataFrame(:Enabled => [true],
                        ## Constrain all assets.
                        :Type => ["All Assets"],
                        ## Greater than or equal to.
                        :Sign => [">="],
                        ## Value of the lower bound.
                        :Weight => [0.01],
                        ## These are not used in this example.
                        :Set => [""], :Position => [""], :Relative_Type => [""],
                        :Relative_Set => [""], :Relative_Position => [""], :Factor => [""])

A, B = asset_constraints(constraints, asset_sets);

# Since we have 20 assets, `A` should be a `20×20` Identity matrix and `B` a `20×1` row vector whose entries are all `0.01`.

A == I, B == fill(0.01, length(assets))

# For completeness we will show how all other cases generalise to constraining all assets. First we constrain the assets to be less than or equal to `0.35`.

constraints = DataFrame(:Enabled => [true],
                        ## Constrain all assets.
                        :Type => ["All Assets"],
                        ## Less than or equal to.
                        :Sign => ["<="],
                        ## Value of the upper bound.
                        :Weight => [0.35],
                        ## These are not used in this example.
                        :Set => [""], :Position => [""], :Relative_Type => [""],
                        :Relative_Set => [""], :Relative_Position => [""], :Factor => [""])

A, B = asset_constraints(constraints, asset_sets);

A == -I, B == fill(-0.35, length(assets))

# Constraining all assets relative to another asset.

constraints = DataFrame(:Enabled => [true],
                        ## Constrain all assets.
                        :Type => ["All Assets"],
                        ## Greater than or equal to.
                        :Sign => [">="],
                        ## Constraining all assets relative
                        ## to another asset.
                        :Relative_Type => ["Asset"],
                        ## Name of the other asset.
                        :Relative_Position => [:GM],
                        ## Factor by which to multiply the relative
                        ## asset's weight.
                        :Factor => [0.3],
                        ## These are not used in this example.
                        :Weight => [""], :Set => [""], :Position => [""],
                        :Relative_Set => [""])

A, B = asset_constraints(constraints, asset_sets);

# Since it's a relative constraint, `B` is a `20×1` vector of zeros.

B == zeros(20)

# We can show that the column vector for each asset has the coefficient for the asset as `1` and the coefficient of the relative asset as `-0.3`. The exception is `:GM`, where the asset and relative asset are the same, so the coefficient is the sum of `1 - 0.3 = 0.7`.

hcat(asset_sets[!, :Asset], DataFrame(A', asset_sets[!, :Asset]))

# By making the constraint an upper bound, the coeffcients in `A` have their signs reversed.

constraints = DataFrame(:Enabled => [true],
                        ## Constraining all assets.
                        :Type => ["All Assets"],
                        ## Relative to a single asset.
                        :Relative_Type => ["Asset"],
                        ## Less than or equal to.
                        :Sign => ["<="],
                        ## Relative asset.
                        :Relative_Position => [:PFE],
                        ## Factor to multiply relative asset weight by.
                        :Factor => [2],
                        ## These are not used in this example.
                        :Weight => [""], :Set => [""], :Position => [""],
                        :Relative_Set => [""])

A, B = asset_constraints(constraints, asset_sets);

# `B` is again a `20×1` vector of zeros.

B == zeros(20)

# The column vector for each asset has the coefficient for the asset as `-1` and the coefficient of the relative asset as `2`. The exception is `:PFE`, where the asset and relative asset are the same, so the coefficient is the sum of `-1 + 2 = 1`.

hcat(asset_sets[!, :Asset], DataFrame(A', asset_sets[!, :Asset]))

# ## Asset subset constraints

# We can constrain the sum of the weights of a subset, ``\sum\left(\bm{w}_{\mathcal{\Omega}}\right)\right) \geq B`` where ``\mathcal{\Omega}`` is a subset of assets. We'll use subset `3` from set `:PD`, and constrain the sum of their weights to be greater than or equal to `0.04`.

constraints = DataFrame(:Enabled => [true],
                        ## Subset constraint.
                        :Type => ["Subset"],
                        ## Set from which the subset is to be taken.
                        :Set => [:PD],
                        ## Subset of :PD
                        :Position => [3],
                        ## Greater than or equal to.
                        :Sign => [">="],
                        ## Value of the lower bound.
                        :Weight => [0.04],
                        ## These are not used in this example.
                        :Relative_Type => [""], :Relative_Set => [""],
                        :Relative_Position => [""], :Factor => [""])

A, B = asset_constraints(constraints, asset_sets);

# `B` is equal to `0.04`

B

# If we set `A` next to the assets and the set from which it was derived, we can see that the subset `3` of set `:PD` has coefficients equal to `1` while the rest have coefficients equal to `0`.

hcat(asset_sets[!, [:Asset, :PD]], DataFrame(:A_t => vec(A)))

# If we want a lower bound, the signs of `A` and `B` will be inverted so ``B \geq \sum\left(\bm{w}_{\mathcal{\Omega}}\right)\right)`` where ``\mathcal{\Omega}`` is a subset of assets. We will constrain the sum of the weights of subset `2` from set `:PD` to be less than or equal to `0.2`.

constraints = DataFrame(:Enabled => [true],
                        ## Subset constraint.
                        :Type => ["Subset"],
                        ## Set from which to take the subset.
                        :Set => [:PD],
                        ## Subset from the set.
                        :Position => [2],
                        ## Less than or equal to.
                        :Sign => ["<="],
                        ## Value of the upper bound.
                        :Weight => [0.2],
                        ## These are not used in this example.
                        :Relative_Type => [""], :Relative_Set => [""],
                        :Relative_Position => [""], :Factor => [""])

A, B = asset_constraints(constraints, asset_sets);

# `B` is `-0.2`.

B

# If we set `A` next to the assets and the set from which it was derived, we can see that the subset `2` of set `:PD` has coefficients equal to `-1` while the rest have coefficients equal to `0`. As we've done before, after subsituting into the definition of the linear constraints and rearranging we recover the aforementioned equation defining the constraint.

hcat(asset_sets[!, [:Asset, :PD]], DataFrame(:A_t => vec(A)))
