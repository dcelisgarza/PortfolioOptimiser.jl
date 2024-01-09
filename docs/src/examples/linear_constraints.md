The source files for all examples can be found in [/examples](https://github.com/dcelisgarza/PortfolioOptimiser.jl/tree/main/examples/).
```@meta
EditURL = "../../../examples/linear_constraints.jl"
```

# Linear constraint matrices

This tutorial explains step by step how the constraint functions can be used to generate linear constraints. We also show how we can generate our own sets using clustering techniques.

````@example linear_constraints
using Clustering,
    CovarianceEstimation, CSV, DataFrames, PortfolioOptimiser, PrettyTables, TimeSeries

# This is a helper function for displaying tables
# with numbers as percentages with 3 decimal points.
fmt = (v, i, j) -> begin
    if j == 1
        return v
    else
        return isa(v, Number) ? "$(round(v*100, digits=3)) %" : v
    end
end;
nothing #hide
````

Load the stock prices, get the asset names and compute the returns.

````@example linear_constraints
prices = TimeArray(CSV.File("./stock_prices.csv"); timestamp = :date)
# We'll be sorting the assets by alphabetical order.
assets = sort(colnames(prices))
# Make sure we index the prices by the sorted assets to maintain consistency.
returns = dropmissing!(DataFrame(percentchange(prices[assets])))
pretty_table(returns[1:5, :]; formatters = fmt)
returns = Matrix(returns[!, 2:end]);
nothing #hide
````

!!! warning
    It is important that the asset order remains consistent accross all steps. If one wishes to sort the assets in a particular way, one should sort them immediately after importing them. This ensures everything is computed using the correct asset order.

## Asset constraints

### Generating our own asset classes with hierarchical clustering

We can use predefined sets of assets, such as the sectors that are traditionally used to classify companies. Instead, we'll define our own classes using various clustering methods. These use a distance matrix to compute the clusters, for this we can use a correlation matrix. We provide various types of correlation matrices from which we'll pick a few examples. For a more thorough explanation of these methods see [`cor_dist_mtx`](@ref).

Note that returns data can be noisy which means some covariance/correlation measures can be thrown off from the true value by outliers and statistically insignificant noise. This can be problem for algorithms that depend on covariance/correlation matrices, such as mean-variance optimisation and clustering.

In order to avoid this overfitting, we can compute robust covariances. In this case, we use the package [`CovarianceEstimation.jl`](https://github.com/mateuszbaran/CovarianceEstimation.jl), as well as some of the functions provided by `PortfolioOptimiser.jl`.

When we use `:Pearson` and `:Semi_Pearson` to compute the correlation, we're internally making use of `StatsBase.cor`, which takes various range of optional arguments, one of which is a covariance estimator (`StatsBase.CovarianceEstimator`), which makes the computation of the covariance matrix more robust. In this example we use `CovarianceEstimation.AnalyticalNonlinearShrinkage()`, as it's a good estimator for tall matrices (i.e. matrices with many more rows than columns). `:Semi_Pearson` also uses a semi-covariance at a given target return value to compute the correlation matrix. We'll elaborate more on a different tutorial, for this we'll use the default value of the target return, which is 0.

### Building the asset classes dataframe.

````@example linear_constraints
asset_classes = DataFrame(:Assets => assets)
cor_settings = CorSettings(;
    estimation = CorEstSettings(;
        estimator = CovarianceEstimation.AnalyticalNonlinearShrinkage(),
        # `target_ret` is used as the return threshold for classifying returns as
        # sufficiently bad as to be considered unappealing, and thus taken into
        # account when computing the semi covariance. The default is 0, but we
        # can make it any number. A good choice is to use the daily risk-free
        # rate, because if an asset returns on average less than a bond, then
        # you'd be better off buying bonds because they're both less risky and
        # more profitable than the asset.
        # target_ret = 1.0329^(1 / 252) - 1,
    ),
)
for method in (:Pearson, :Semi_Pearson, :Gerber2)
    for linkage in (:ward, :DBHT)
        # We define our asset sets based on the covariance and linkage methods.
        colname = Symbol(string(method)[1] * string(linkage)[1])

        # Set the method for computing the covariance matrix.
        cor_settings.method = method

        # Clusterise assets.
        clustering, k =
            cluster_assets(returns; linkage = linkage, cor_settings = cor_settings)

        # Cut the tree at k clusters and return the label each asset belongs to
        # in each set.
        asset_classes[!, colname] = cutree(clustering, k = k)
    end
end

asset_classes
````

### Generating the constraint matrices

As previously mentioned, assets are categorised on their industry sectors, but this way we can create our own in a data-driven way. We'll use these to build our linear constraints. We'll be building them one type of constraint by one, but you can also create them all in one go by concatenating the vectors.

Say we want to ensure we have at least 3% of our portfolio allocated to `GOOG`, ``w_{\mathrm{goog}} \geq 0.03``.

````@example linear_constraints
constraints = DataFrame(
    # Enable the constraint.
    :Enabled => [true],
    # We are constraining a single asset.
    :Type => ["Assets"],
    # The asset we're constraining.
    :Position => [:GOOG],
    # Greater than.
    :Sign => [">="],
    # Lower bound of the weight of the asset.
    :Weight => [0.03],
    # The following categories are not used for this
    # example.
    :Class_Set => [""],
    :Relative_Type => [""],
    :Relative_Class_Set => [""],
    :Relative_Position => [""],
    :Factor => [""],
)

A, B = asset_constraints(constraints, asset_classes);
nothing #hide
````

Recall that the linear constraints are defined as ``\mathbf{A} \bm{w} \geq \bm{b}``, we can show that the constraint will be applied to `GOOG`, as its weight coefficient is 1, while all the others are 0.

````@example linear_constraints
hcat(asset_classes[!, :Assets], DataFrame(:A_t => vec(A)))
````

Using the definition of the linear constraints, `B` should be 0.03.

````@example linear_constraints
B
````

Substituting these values into the linear constraint definition, removing zero values we arrive at the constraint ``w_{\mathrm{goog}} \geq 0.03``.

Say we also want to constrain the upper bound on `GOOG` so we're not overexposed. Lets make it 10%. In practice we'd make it all in a single step but this is a pedagogical example. Note how the only things that change are `:Sign` and `:Weight`.

````@example linear_constraints
constraints = DataFrame(
    :Enabled => [true],
    :Type => ["Assets"],
    :Class_Set => [""],
    :Position => [:GOOG],
    # Less than or equal to.
    :Sign => ["<="],
    # Upper bound of the weight of the asset.
    :Weight => [0.1],
    :Relative_Type => [""],
    :Relative_Class_Set => [""],
    :Relative_Position => [""],
    :Factor => [""],
)

A, B = asset_constraints(constraints, asset_classes);
nothing #hide
````

The sign in the `A` matrix will be inverted.

````@example linear_constraints
hcat(asset_classes[!, :Assets], DataFrame(:A_t => vec(A)))
````

And so will the sign of the `B` vector.

````@example linear_constraints
B
````

By substituting these values into the definition of the linear constraints above, removing all zero values, and rearranging, ``-w_{\mathrm{goog}} \geq -0.1 \to 0.1 \geq w_{\mathrm{goog}}``. This sign inversion works the same for every constraint type, if the constraint denotes an upper bound (the corresponding value in `:Sign` is `<=`), both sides of the constraint are multiplied by `-1`.

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

