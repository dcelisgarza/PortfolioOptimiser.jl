```@meta
CurrentModule = PortfolioOptimiser
```

# PortfolioOptimiser

```@docs
PortfolioOptimiser
```

## Description

[`PortfolioOptimiser`](https://github.com/dcelisgarza/PortfolioOptimiser.jl) is a library for portfolio optimisation. It offers a broad range of functionality, and is designed with ease of use, composability, extensibility, maintainability in mind. It does so by leveraging Julia's type system, multiple dispatch, and the principle of separation of concerns.

[`PortfolioOptimiser`](https://github.com/dcelisgarza/PortfolioOptimiser.jl) takes a hands-off approach when it comes to solvers and solver settings. The built-in forumations are mostly conic in nature as these tend to yield more accurate solutions that are easier to solve. This means the choice of solver and its parameters are left to the user.

The library currently focues purely on optimisation and parameter estimation. There is currently no plan for backtesting, model selection, validation, or returns series generation. These may be implemented in the future as separate packages.

## Quick Start

The following example shows how one can download the data and perform a simple optimisation.

```@example quick-start
using PortfolioOptimiser, TimeSeries, DataFrames, Clarabel, HiGHS, YFinance, Dates, JuMP

function stock_price_to_time_array(x)
    # Only get the keys that are not ticker or datetime.
    coln = collect(keys(x))[3:end]
    # Convert the dictionary into a matrix.
    m = hcat([x[k] for k ∈ coln]...)
    return TimeArray(x["timestamp"], m, Symbol.(coln), x["ticker"])
end

# Tickers of the assets we want to download.
assets = sort!(["SOUN", "RIVN", "GME", "AMC", "SOFI", "ENVX", "ANVS", "LUNR", "EOSE", "SMR",
                "NVAX", "NKLA", "ACHR", "RKLB", "MARA"])

# Prices date range.
Date_0 = "2024-01-01"
Date_1 = "2025-01-01"

# Download the price data using YFinance.
prices = get_prices.(assets; startdt = Date_0, enddt = Date_1)

# Convert vector of ordered dicts into a TimeArray.
prices = stock_price_to_time_array.(prices)
prices = hcat(prices...)

# Select only the adjusted close prices.
cidx = colnames(prices)[occursin.(r"adj", string.(colnames(prices)))]
prices = prices[cidx]

# Rename the columns to the asset tickers.
TimeSeries.rename!(prices, Symbol.(assets))

# Generate the portfolio.
portfolio = Portfolio(; prices = prices,
                      # Continuous solvers.
                      solvers = PortOptSolver(
                                              # Key-value pair for the solver, solution acceptance 
                                              # criteria, model bridge argument, and solver attributes.
                                              ; name = :Clarabel,
                                              # Solver we wish to use.
                                              solver = Clarabel.Optimizer,
                                              # (Optional) Solution acceptance criteria.
                                              check_sol = (allow_local = true,
                                                           allow_almost = true),
                                              # (Optional) Solver-specific attributes.
                                              params = ["verbose" => false,
                                                        "max_step_fraction" => 0.75,
                                                        "tol_gap_abs" => 1e-9,
                                                        "tol_gap_rel" => 1e-9]),
                      # Discrete solvers (for discrete allocation).
                      alloc_solvers = PortOptSolver(
                                                    # Key-value pair for the solver, solution acceptance 
                                                    # criteria, model bridge argument, and solver attributes.
                                                    ; name = :HiGHS,
                                                    # Solver we wish to use.
                                                    solver = optimizer_with_attributes(HiGHS.Optimizer,
                                                                                       MOI.Silent() => true),
                                                    # (Optional) Solution acceptance criteria.
                                                    check_sol = (allow_local = true,
                                                                 allow_almost = true)))

# Compute the asset statistics.
asset_statistics!(portfolio)

# Optimise the portfolio using the traditional Markowitz model with a conic formulation.
w = optimise!(portfolio, Trad())

# Discretely allocate 69420 dollars in shares of the assets.
wd = allocate!(portfolio; investment = 69420)

# Show discrete allocation with optimal allocation.
wd.optimal_weights = w.weights
DataFrames.rename!(wd, :weights => :allocated_weights)
show(wd)
```

## Functionality

  - Non-hierarchical optimisation models
    
      + Traditional, [`Trad`](@ref).
      + Risk Budgeting, [`RB`](@ref).
      + Relaxed Risk Budgetting (Variance only), [`RRB`](@ref).
      + Near Optimal Centering, [`NOC`](@ref).

  - Hierarchical optimisation models
    
      + Hierarchical Risk Parity, [`HRP`](@ref).
      + Hierarchical Risk Parity Schur Complement (Variance only), [`SchurHRP`](@ref).
      + Hierarchical Equal Risk Parity, [`HERC`](@ref).
      + Nested Clustered Optimisation, [`NCO`](@ref).

## Expected returns estimators

  - Arithmetic (weighted and unweighted), [`MuSimple`](@ref).

  - Equilibrium, [`MuEquil`](@ref).
  - Shringage with Grand Mean [`GM`](@ref), Volatility Weighted [`VW`](@ref), and Mean Square Error [`MSE`](@ref) targets:
    
      + James-Stein, [`MuJS`](@ref).
      + Bayes-Stein, [`MuBS`](@ref).
      + Bodnar-Okhrin-Parolya, [`MuBOP`](@ref).

## Covariance estimators

These utilise [`StatsBase`](https://juliastats.org/StatsBase.jl/stable/cov/#Scatter-Matrix-and-Covariance)'s API to define covariance estimators. Which means [`PortfolioOptimiser`](https://github.com/dcelisgarza/PortfolioOptimiser.jl) is compatible with [`CovarianceEstimation`](https://github.com/mateuszbaran/CovarianceEstimation.jl).

  - PortfolioOptimiser, [`PortCovCor`](@ref).
  - Full, [`CovFull`](@ref).
  - Semi, [`CovSemi`](@ref).
  - Mutual Information, [`CovMutualInfo`](@ref).
  - Distance, [`CovDistance`](@ref).
  - Lower Tail Dependence, [`CovLTD`](@ref).
  - Gerber, [`CovGerber0`](@ref), [`CovGerber1`](@ref), [`CovGerber2`](@ref).
  - Smyth-Broby, [`CovSB0`](@ref), [`CovSB1`](@ref), [`CovSB2`](@ref).
  - Smyth-Broby-Gerber, [`CovGerberSB0`](@ref), [`CovGerberSB1`](@ref), [`CovGerberSB2`](@ref).

## Correlation estimators

These utilise [`StatsBase`](https://juliastats.org/StatsBase.jl/stable/cov/#Scatter-Matrix-and-Covariance)'s API to define covariance estimators. Which means [`PortfolioOptimiser`](https://github.com/dcelisgarza/PortfolioOptimiser.jl) is compatible with [`CovarianceEstimation`](https://github.com/mateuszbaran/CovarianceEstimation.jl).

  - All covariance estimators.
  - Spearman, [`CorSpearman`](@ref).
  - Kendall, [`CorKendall`](@ref).

## Distance estimators

We provide distance and distance of distances estimators. Distance estimators have the prefix `Dist`, distance of distances have the prefix `DistDist`.

  - Canonical, [`DistCanonical`](@ref), [`DistDistCanonical`](@ref).
  - Marcos López de Prado, [`DistMLP`](@ref), [`DistDistMLP`](@ref).
  - Generalised Marcos López de Prado, [`GenDistMLP`](@ref), [`GenDistDistMLP`](@ref).
  - Log, [`DistLog`](@ref), [`DistDistLog`](@ref).
  - Distance correlation, [`DistCor`](@ref), [`DistDistCor`](@ref).
  - Variation of information, [`DistVarInfo`](@ref), [`DistDistVarInfo`](@ref).

## Cokurtosis estimators

  - Full, [`KurtFull`](@ref).
  - Semi, [`KurtSemi`](@ref).

## Coskewness estimators

  - Full, [`SkewFull`](@ref).
  - Semi, [`SkewSemi`](@ref).

## Square matrix post-processing

  - Fixing non-positive definite matrices, [`NoPosdef`](@ref), [`PosdefNearest`](@ref).
  - Matrix denoising, [`NoDenoise`](@ref), [`DenoiseFixed`](@ref), [`DenoiseSpectral`](@ref), [`DenoiseShrink`](@ref).
  - Matrix detoning, [`NoDetone`](@ref), [`Detone`](@ref).
  - Local-global sparsification of the matrix inverse, [`NoLoGo`](@ref), [`LoGo`](@ref).

## Prior estimators

  - Empirical, [`asset_statistics!`](@ref).
  - Worst-case uncertainty sets, [`wc_statistics!`](@ref).
  - Factor models, [`factor_statistics!`](@ref).
  - Black-Litterman, [`black_litterman_statistics!`](@ref).
  - Black-Litterman factor models, [`black_litterman_factor_statistics!`](@ref).

## Regression models

  - Forward and Backward Regression, [`FReg`](@ref), [`BReg`](@ref), with criteria.
    
      + p-value threshold, [`PVal`](@ref).
      + Akaike's Information Criterion, [`AIC`](@ref).
      + Corrected Akaike's Information Criterion for small sample sizes, [`AICC`](@ref).
      + Bayesian Information Criterion, [`BIC`](@ref).
      + R² of a linear model, [`RSq`](@ref).
      + adjusted R² for a linear model, [`AdjRSq`](@ref).

  - Principal Component-Based Regression, [`PCAReg`](@ref), with targets.
    
      + PCA target, [`PCATarget`](@ref).
      + Probabilistic PCA target, [`PPCATarget`](@ref).

## Worst-case uncertainty set estimators

These are only for the covariance and expected returns.

  - Box and Elliptical sets, [`Box`](@ref), [`Ellipse`](@ref).
    
      + Autoregressive Conditional Heteroskedasticity models, [`ArchWC`](@ref), with bootstraps.
        
          * Stationary, [`StationaryBS`](@ref).
          * Circular, [`CircularBS`](@ref).
          * Moving, [`MovingBS`](@ref).
    
      + Normal, [`NormalWC`](@ref).

  - Box sets only.
    
      + Delta, [`DeltaWC`](@ref).
  - Elliptical set constraint error size estimation.
    
      + Normal, [`KNormalWC`](@ref).
      + General, [`KGeneralWC`](@ref)

## Clustering

  - Direct Bubble Hierarchy Trees, [`DBHT`](@ref).

  - Hierarchical clustering, [`HAC`](@ref).
  - Optimal number of clusters.
    
      + Two-difference gap statistic, [`TwoDiff`](@ref).
      + Standardised silhouette scores, [`StdSilhouette`](@ref).

## Networks

  - Triangular maximally filtered graphs (TMFG), [`TMFG`](@ref).

  - Minimum spanning trees (MST), [`MST`](@ref).
    
      + Kruskal, [`KruskalTree`](@ref).
      + Boruvka, [`BoruvkaTree`](@ref).
      + Prim, [`PrimTree`](@ref).
  - Centrality measures.
    
      + Betweenness, [`BetweennessCentrality`](@ref).
      + Closeness, [`ClosenessCentrality`](@ref).
      + Degree, [`DegreeCentrality`](@ref).
      + Eigenvector, [`EigenvectorCentrality`](@ref).
      + Katz, [`KatzCentrality`](@ref).
      + Pagerank, [`Pagerank`](@ref).
      + Radiality, [`RadialityCentrality`](@ref).
      + Stress, [`StressCentrality`](@ref).

UP TO HERE

### Black Litterman models

  - Black Litterman, [`BLType`](@ref).

#### Black Litterman factor models

  - Augmented Black Litterman, [`ABLType`](@ref).
  - Bayesian Black Litterman, [`BBLType`](@ref).

### Linear moments (L-moments)

  - Normalised constant relative risk aversion coefficients, [`CRRA`](@ref).
  - Maximum entropy, [`MaxEntropy`](@ref).
  - Minimum Sum of Squares, [`MinSumSq`](@ref).
  - Minimum Square Distance, [`MinSqDist`](@ref).

## Portfolio optimisation

These types of optimisations act on instances of [`Portfolio`](@ref).

### Traditional, [`Trad`](@ref)

This type of optimisation is the traditional efficient frontier optimisation.

#### Classes, [`PortClass`](@ref)

  - Classic, [`Classic`](@ref).
  - Factor model, [`FM`](@ref).
  - Black Litterman, [`BL`](@ref).
  - Black Litterman Factor model, [`BLFM`](@ref).

#### Expected returns

  - Arithmetic returns, [`NoKelly`](@ref).
  - Approximate logarithmic mean returns, [`AKelly`](@ref).
  - Exact logarithmic mean returns, [`EKelly`](@ref).

#### Objective functions

  - Minimum risk, [`MinRisk`](@ref).
  - Maximum utility, [`Utility`](@ref).
  - Maximum risk adjusted return ratio, [`Sharpe`](@ref).
  - Maximum return, [`MaxRet`](@ref).

#### Constraints

  - Maximum expected risk constraints.
  - Minimum expected return constraint.
  - Linear weight constraints.
  - Connected asset centrality constraints.
  - Asset network constraints.
  - Leverage constraints.
  - Maximum number of assets constraint.
  - Minimum number of effective assets constraint.
  - Tracking error (weights or returns) constraint.
  - Turnover constraint.
  - Rebalancing penalty.

#### Risk measures

#### Dispersion

  - Full dispersion.
    
      + Standard deviation, [`SD`](@ref).
      + Mean absolute deviation (MAD), [`MAD`](@ref).
      + Square root kurtosis, [`Kurt`](@ref).
      + Range, [`RG`](@ref).
      + Conditional value at risk range (CVaR range), [`CVaRRG`](@ref).
      + Tail Gini range, [`TGRG`](@ref).
      + Gini mean difference (GMD), [`GMD`](@ref).
      + Quadratic negative skewness, [`Skew`](@ref).
      + Brownian distance variance (BDVariance), [`BDVariance`](@ref).

  - Downside dispersion.
    
      + Semi standard deviation, [`SSD`](@ref).
      + First lower partial moment (Omega ratio), [`FLPM`](@ref).
      + Second lower partial moment (Sortino ratio), [`SLPM`](@ref).
      + Square root semi kurtosis, [`SKurt`](@ref).
      + Quadratic negative semi skewness, [`SSkew`](@ref).

#### Downside

  - Worst case realisation (Minimax), [`WR`](@ref).
  - Conditional value at risk (CVaR), [`CVaR`](@ref).
  - Entropic value at risk (EVaR), [`EVaR`](@ref).
  - Relativistic value at risk (RLVaR), [`RLVaR`](@ref).
  - Tail Gini, [`TG`](@ref).

#### Drawdown

  - Maximum drawdown (Calmar ratio) for uncompounded cumulative returns, [`MDD`](@ref).
  - Average drawdown for uncompounded cumulative returns, [`ADD`](@ref).
  - Ulcer index for uncompounded cumulative returns, [`UCI`](@ref).
  - Conditional drawdown at risk for uncompounded cumulative returns (CDaR), [`CDaR`](@ref).
  - Entropic drawdown at risk for uncompounded cumulative returns (EDaR), [`EDaR`](@ref).
  - Relativistic drawdown at risk for uncompounded cumulative returns (RLDaR), [`RLDaR`](@ref).

#### Linear moments (L-moments)

  - L-moment ordered weight array, [`OWA`](@ref).

### Worst case mean variance

This type of optimisation requires worst case sets for the covariance and expected returns. The optimisation uses these sets to perform a mean variance optimisation.

#### Constraints

  - Maximum expected worst case standard deviation constraint.
  - Minimum expected worst case return constraint.
  - Linear weight constraints.
  - Connected asset centrality constraints.
  - Asset network constraints.
  - Leverage constraints.
  - Maximum number of assets constraint.
  - Minimum number of effective assets constraint.
  - Tracking error (weights or returns) constraint.
  - Turnover constraint.
  - Rebalancing penalty.

### Risk parity, [`RB`](@ref)

This type of optimisation requires a risk budget per asset or factor. The optimisation attempts to minimise the difference between the risk budget and risk contribution of the asset or factor in the optimised portfolio.

#### Classes, [`PortClass`](@ref)

  - Classic, [`Classic`](@ref).
  - Factor model, [`FM`](@ref).
  - Factor risk contribution, [`FC`](@ref).

#### Constraints

  - Minimum expected return constraint.
  - Linear weight constraints.

#### Risk measures

#### Dispersion

  - Full dispersion.
    
      + Standard deviation, [`SD`](@ref).
      + Mean absolute deviation (MAD), [`MAD`](@ref).
      + Square root kurtosis, [`Kurt`](@ref).
      + Range, [`RG`](@ref).
      + Conditional value at risk range (CVaR range), [`CVaRRG`](@ref).
      + Tail Gini range, [`TGRG`](@ref).
      + Gini mean difference (GMD), [`GMD`](@ref).
      + Quadratic negative skewness, [`Skew`](@ref).
      + Brownian distance variance (BDVariance), [`BDVariance`](@ref).

  - Downside dispersion.
    
      + Semi standard deviation, [`SSD`](@ref).
      + First lower partial moment (Omega ratio), [`FLPM`](@ref).
      + Second lower partial moment (Sortino ratio), [`SLPM`](@ref).
      + Square root semi kurtosis, [`SKurt`](@ref).
      + Quadratic negative semi skewness, [`SSkew`](@ref).

#### Downside

  - Worst case realisation (Minimax), [`WR`](@ref).
  - Conditional value at risk (CVaR), [`CVaR`](@ref).
  - Entropic value at risk (EVaR), [`EVaR`](@ref).
  - Relativistic value at risk (RLVaR), [`RLVaR`](@ref).
  - Tail Gini, [`TG`](@ref).

#### Drawdown

  - Maximum drawdown (Calmar ratio) for uncompounded cumulative returns, [`MDD`](@ref).
  - Average drawdown for uncompounded cumulative returns, [`ADD`](@ref).
  - Ulcer index for uncompounded cumulative returns, [`UCI`](@ref).
  - Conditional drawdown at risk for uncompounded cumulative returns (CDaR), [`CDaR`](@ref).
  - Entropic drawdown at risk for uncompounded cumulative returns (EDaR), [`EDaR`](@ref).
  - Relativistic drawdown at risk for uncompounded cumulative returns (RLDaR), [`RLDaR`](@ref).

#### Linear moments (L-moments)

  - L-moment ordered weight array, [`OWA`](@ref).

### Relaxed risk parity mean variance, [`RRB`](@ref)

This type of optimisation requires a risk budget per asset. The optimisation attempts to minimise the difference between the risk budget and relaxed formulation of the standard deviation risk measure.

#### Classes, [`PortClass`](@ref)

  - Classic, [`Classic`](@ref).
  - Factor model, [`FM`](@ref).

#### Constraints

  - Minimum expected return constraint.
  - Linear weight constraints.

### Near Optimal Centering, [`NOC`](@ref)

Near optimal centering utilise the weights of an optimised portfolio. It computes a region of near optimality using the bounds of the efficient frontier, the expected risk and return of the optimal portfolio, and a user-provided parameter. It then optimises for a portfolio that best describes the region. It provides more diversification and robustness than [`Trad`](@ref) and smooths out the weight transitions as the efficient frontier is traversed.

#### Classes, [`PortClass`](@ref)

  - Classic, [`Classic`](@ref).
  - Factor model, [`FM`](@ref).
  - Black Litterman, [`BL`](@ref).
  - Black Litterman Factor model, [`BLFM`](@ref).

#### Constraints

  - Minimum expected return constraint.
  - Linear weight constraints.

#### Classes, [`PortClass`](@ref)

  - Classic, [`Classic`](@ref).
  - Factor model, [`FM`](@ref).
  - Black Litterman, [`BL`](@ref).
  - Black Litterman Factor model, [`BLFM`](@ref).

#### Expected returns

  - Arithmetic returns, [`NoKelly`](@ref).
  - Approximate logarithmic mean returns, [`AKelly`](@ref).
  - Exact logarithmic mean returns, [`EKelly`](@ref).

#### Constraints

  - Maximum expected risk constraints.
  - Minimum expected return constraint.

#### Risk measures

#### Dispersion

  - Full dispersion.
    
      + Standard deviation, [`SD`](@ref).
      + Mean absolute deviation (MAD), [`MAD`](@ref).
      + Square root kurtosis, [`Kurt`](@ref).
      + Range, [`RG`](@ref).
      + Conditional value at risk range (CVaR range), [`CVaRRG`](@ref).
      + Tail Gini range, [`TGRG`](@ref).
      + Gini mean difference (GMD), [`GMD`](@ref).
      + Quadratic negative skewness, [`Skew`](@ref).
      + Brownian distance variance (BDVariance), [`BDVariance`](@ref).

  - Downside dispersion.
    
      + Semi standard deviation, [`SSD`](@ref).
      + First lower partial moment (Omega ratio), [`FLPM`](@ref).
      + Second lower partial moment (Sortino ratio), [`SLPM`](@ref).
      + Square root semi kurtosis, [`SKurt`](@ref).
      + Quadratic negative semi skewness, [`SSkew`](@ref).

#### Downside

  - Worst case realisation (Minimax), [`WR`](@ref).
  - Conditional value at risk (CVaR), [`CVaR`](@ref).
  - Entropic value at risk (EVaR), [`EVaR`](@ref).
  - Relativistic value at risk (RLVaR), [`RLVaR`](@ref).
  - Tail Gini, [`TG`](@ref).

#### Drawdown

  - Maximum drawdown (Calmar ratio) for uncompounded cumulative returns, [`MDD`](@ref).
  - Average drawdown for uncompounded cumulative returns, [`ADD`](@ref).
  - Ulcer index for uncompounded cumulative returns, [`UCI`](@ref).
  - Conditional drawdown at risk for uncompounded cumulative returns (CDaR), [`CDaR`](@ref).
  - Entropic drawdown at risk for uncompounded cumulative returns (EDaR), [`EDaR`](@ref).
  - Relativistic drawdown at risk for uncompounded cumulative returns (RLDaR), [`RLDaR`](@ref).

#### Linear moments (L-moments)

  - L-moment ordered weight array, [`OWA`](@ref).

## Hierarchical portfolio optimisation

These types of optimisations act on instances of .

### Hierarchical risk parity, [`HRP`](@ref), and hierarchical equal risk parity, [`HERC`](@ref)

#### Hierarchical risk parity, [`HRP`](@ref)

Hierarchical risk parity optimisations use the hierarchical clustering of assets to assign risk contributions by iteratively splitting the dendrogram in half and assigning weights to each half according to the relative risk each half represents with respect to the other. It does this until it splits the dendrogram all the way down to single leaves.

#### Hierarchical equal risk parity, [`HERC`](@ref)

Hierarchical equal risk parity optimisations use the hierarchical clustering relationships between assets to assign risk contributions by splitting the dendrogram into `k` clusters. It starts with the full dendrogram and progressively cuts it into `k-1` levels (since the comparison for each side belongs to the `k`-th level). At each step, it loops through the clusters and checks to which side of the sub-dendrogram the cluster belongs. It accumulates the risk of that cluster to the risk of the side it belongs to. The weights for the assets on each side of the dendrogram are assigned based on the relative (with respect to the other side) aggregate risk from all clusters belonging to it, these are the inter-cluster weights. It then computes the risk for each cluster, assigning weights to each asset according to the relative risk it represents with respect to other assets, these are the intra-cluster weights. It then elementwise multiplies both weights to get the final asset weights.

[`HERC`](@ref) can make use of two risk measure arguments, one for the intra-cluster and one for the inter-cluster risk calculation. They can take linear combinations of risk measures.

#### Constraints

  - Minimum and maximum weights per asset.

#### Risk measures

#### Dispersion

  - Full dispersion.
    
      + Variance, [`Variance`](@ref).
      + Standard deviation, [`SD`](@ref).
      + Mean absolute deviation (MAD), [`MAD`](@ref).
      + Square root kurtosis, [`Kurt`](@ref).
      + Range, [`RG`](@ref).
      + Conditional value at risk range (CVaR range), [`CVaRRG`](@ref).
      + Tail Gini range, [`TGRG`](@ref).
      + Gini mean difference (GMD), [`GMD`](@ref).
      + Quadratic negative skewness, [`Skew`](@ref).
      + Brownian distance variance (BDVariance), [`BDVariance`](@ref).

  - Downside dispersion.
    
      + Semi Variance, [`Variance`](@ref).
      + Semi standard deviation, [`SSD`](@ref).
      + First lower partial moment (Omega ratio), [`FLPM`](@ref).
      + Second lower partial moment (Sortino ratio), [`SLPM`](@ref).
      + Square root semi kurtosis, [`SKurt`](@ref).
      + Quadratic negative semi skewness, [`SSkew`](@ref).

#### Downside

  - Worst case realisation (Minimax), [`WR`](@ref).
  - Value at risk (VaR), [`VaR`](@ref).
  - Conditional value at risk (CVaR), [`CVaR`](@ref).
  - Entropic value at risk (EVaR), [`EVaR`](@ref).
  - Relativistic value at risk (RLVaR), [`RLVaR`](@ref).
  - Tail Gini, [`TG`](@ref).

#### Drawdown

  - Maximum drawdown (Calmar ratio) for uncompounded cumulative returns, [`MDD`](@ref).
  - Average drawdown for uncompounded cumulative returns, [`ADD`](@ref).
  - Ulcer index for uncompounded cumulative returns, [`UCI`](@ref).
  - Drawdown at for uncompounded cumulative returns risk (DaR), [`DaR`](@ref).
  - Conditional drawdown at risk for uncompounded cumulative returns (CDaR), [`CDaR`](@ref).
  - Entropic drawdown at risk for uncompounded cumulative returns (EDaR), [`EDaR`](@ref).
  - Relativistic drawdown at risk for uncompounded cumulative returns (RLDaR), [`RLDaR`](@ref).
  - Maximum drawdown (Calmar ratio) for compounded cumulative returns, [`MDD_r`](@ref).
  - Average drawdown for compounded cumulative returns, [`ADD_r`](@ref).
  - Ulcer index for compounded cumulative returns, [`UCI_r`](@ref).
  - Drawdown at for compounded cumulative returns risk (DaR), [`DaR_r`](@ref).
  - Conditional drawdown at risk for compounded cumulative returns (CDaR), [`CDaR_r`](@ref).
  - Entropic drawdown at risk for compounded cumulative returns (EDaR), [`EDaR_r`](@ref).
  - Relativistic drawdown at risk for compounded cumulative returns (RLDaR), [`RLDaR_r`](@ref).

#### Linear moments (L-moments)

  - L-moment ordered weight array, [`OWA`](@ref).

#### Equal Risk Contribution

  - Equal risk contribution, [`Equal`](@ref).

### Nested clustered optimisation, [`NCO`](@ref)

Nested clustered optimisation combines the ideas of hierarchical equal risk parity optimisations and portfolio optimisations. They use the hierarchical clustering relationships between assets and splitting the dendrogram into `k` clusters. It then treats each cluster as its own isntance of [`Portfolio`](@ref) which is optimised in the usual way. The weights of each cluster are saved in a matrix, these are the intra-cluster weights. Then each cluster as a whole is treated as a synthetic asset, it statistics are internally computed from the fields in the [`NCO`](@ref) type. An instance of [`Portfolio`](@ref) is created from these synthetic assets and then optimised, these are the inter-cluster weights. The inter-cluster and intra-cluster weights are multiplied to give the asset weights.

[`NCO`](@ref) can make use of two risk measure arguments, one for the intra-cluster and one for the inter-cluster risk calculation. They can take linear combinations of risk measures.

#### Sub-types

[`NCO`](@ref) can take keyword arguments that define the supported by optimisations of [`Portfolio`](@ref). Since there are intra- and inter-cluster optimisations, it can take individual arguments for both. This means it can perform any combination of [`Portfolio`](@ref) optimisations.

#### NCO-Trad, NCO-NOC-Trad

##### Objective functions

  - Minimum risk, [`MinRisk`](@ref).
  - Maximum utility, [`Utility`](@ref).
  - Maximum risk adjusted return ratio, [`Sharpe`](@ref).
  - Maximum return, [`MaxRet`](@ref).

##### Constraints

When applied to the intra-cluster optimisation the same constraint will be applied to all every cluster.

  - Maximum expected risk constraints.
  - Minimum expected return constraint.
  - Leverage constraints.

#### NCO-RB, NCO-RRB, NCO-NOC-RB, NCO-NOC-RRB

##### Constraints

When applied to the intra-cluster optimisation the same constraint will be applied to all every cluster.

  - Minimum expected return constraint.
