# PortfolioOptimiser

| Category | Badge                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
|:--------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| Docs     | [![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://dcelisgarza.github.io/PortfolioOptimiser.jl/stable) [![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://dcelisgarza.github.io/PortfolioOptimiser.jl/dev)                                                                                                                                                                                                                                                                                                                                                    |
| Examples | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/dcelisgarza/PortfolioOptimiser.jl/HEAD?labpath=%2Fexamples)                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| CI       | [![Tests](https://github.com/dcelisgarza/PortfolioOptimiser.jl/actions/workflows/Tests.yml/badge.svg)](https://github.com/dcelisgarza/PortfolioOptimiser.jl/actions/workflows/Tests.yml) [![Documentation](https://github.com/dcelisgarza/PortfolioOptimiser.jl/actions/workflows/Documentation.yml/badge.svg)](https://github.com/dcelisgarza/PortfolioOptimiser.jl/actions/workflows/Documentation.yml) [![Aqua](https://github.com/dcelisgarza/PortfolioOptimiser.jl/actions/workflows/Aqua.yml/badge.svg)](https://github.com/dcelisgarza/PortfolioOptimiser.jl/actions/workflows/Aqua.yml) |
| Coverage | [![Codecov](https://codecov.io/gh/dcelisgarza/PortfolioOptimiser.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/dcelisgarza/PortfolioOptimiser.jl) [![Coveralls](https://coveralls.io/repos/github/dcelisgarza/PortfolioOptimiser.jl/badge.svg?branch=main)](https://coveralls.io/github/dcelisgarza/PortfolioOptimiser.jl?branch=main)                                                                                                                                                                                                                                                 |

## Description

PortfolioOptimiser is a library for portfolio optimisation. It was written with composability and extensibility in mind. It offers a broad range of functionality out of the box. It wraps a lot of external functionality in types for dispatch purposes and to allow easy development and extensibility.

## Parameter estimation

### Matrix processing

These only apply to covariance, correlation, and cokurtosis estimators. Dissimilarity and similarity matrices use the results of correlation estimators, so they are indirectly used there.

#### Sparsification

  - Local/Global parsimonious estimator (LoGo), [`LoGo`](@ref).

#### Fixing non-positive definite matrices

  - Nearest correlation matrix, [`PosdefNearest`](@ref).

#### Denoising methods

  - Fixed, [`DenoiseFixed`](@ref).
  - Spectral, [`DenoiseSpectral`](@ref).
  - Shrink, [`DenoiseShrink`](@ref).

### Expected mean returns estimators

  - Simple mean, [`MuSimple`](@ref).
  - James-Stein (JS), [`MuJS`](@ref).
  - Bayes-Stein (BS), [`MuBS`](@ref).
  - Bodnar-Okhrin-Parolya (BOP), [`MuBOP`](@ref).

The JS, BS and BOP estimators also use a target for correcting their estimates.

  - Grand mean (GM), [`GM`](@ref).
  - Volatility-weighted grand mean (VW), [`VW`](@ref).
  - Mean square error of sample mean (SE), [`SE`](@ref).

### Covariance estimators

  - Full, [`CovFull`](@ref).
  - Semi, [`CovSemi`](@ref).
  - Mutual information, [`CorMutualInfo`](@ref).
  - Brownian distance, [`CovDistance`](@ref).
  - Lower tail dependence, [`CorLTD`](@ref).
  - Gerber type 0, [`CorGerber0`](@ref).
  - Gerber type 1, [`CorGerber1`](@ref).
  - Gerber type 2, [`CorGerber2`](@ref).
  - Smyth-Broby modification of Gerber type 0, [`CorSB0`](@ref).
  - Smyth-Broby modification of Gerber type 1, [`CorSB1`](@ref).
  - Smyth-Broby modification with vote counting of Gerber type 0, [`CorGerberSB0`](@ref).
  - Smyth-Broby modification with vote counting of Gerber type 1, [`CorGerberSB1`](@ref).

### Correlation estimators

All covariance estimators can be used for correlation estimation.

  - Spearman rank, [`CorSpearman`](@ref).
  - Kendall rank, [`CorKendall`](@ref).

### Disimilarity/distance matrix functions

  - Marcos López de Prado, [`DistMLP`](@ref).
  - Marcos López de Prado distance of distance, [`DistDistMLP`](@ref).
  - Negative log, [`DistLog`](@ref).
  - Variation of information, [`DistVarInfo`](@ref).

### Triangulated maximally filtered graph similarity matrix functions

  - Exponential decay, [`DBHTExp`](@ref).
  - Square distance from maximum, [`DBHTMaxDist`](@ref).

### Bin width estimation functions

  - Knuth, require `PyCall` and `astropy` to be installed, [`Knuth`](@ref).
  - Freedman, require `PyCall` and `astropy` to be installed, [`Freedman`](@ref).
  - Scott, require `PyCall` and `astropy` to be installed, [`Scott`](@ref).
  - Hacine-Gharbi and Ravier, [`HGR`](@ref).

### Cokurtosis estimators

  - Full cokurtosis, [`KurtFull`](@ref).
  - Semi cokurtosis, [`KurtSemi`](@ref).

### Coskewness estimators

  - Full coskewness, [`SkewFull`](@ref).
  - Semi coskewness, [`SkewSemi`](@ref).

### Clustering

  - Hierarchical clustering, [`HAC`](@ref).
  - Direct Bubble Hierarchy Trees clustering, [`DBHT`](@ref).

#### Determining number of clusters

  - Two different gap statistic, [`TwoDiff`](@ref).
  - Standardised silhouette scores, [`StdSilhouette`](@ref).

### Networks

  - Triangular maximally filtered graphs (TMFG), [`TMFG`](@ref).
  - Minimum spanning tree (MST), [`MST`](@ref).

#### Centrality measures

  - Betweenness, [`BetweennessCentrality`](@ref).
  - Closeness, [`ClosenessCentrality`](@ref).
  - Degree, [`DegreeCentrality`](@ref).
  - Eigenvector, [`EigenvectorCentrality`](@ref).
  - Katz, [`KatzCentrality`](@ref).
  - Pagerank, [`Pagerank`](@ref).
  - Radiality, [`RadialityCentrality`](@ref).
  - Stress, [`StressCentrality`](@ref).

#### Minimum spanning tree algorithms

  - Kruskal, [`KruskalTree`](@ref).
  - Boruvka, [`BoruvkaTree`](@ref).
  - Prim, [`PrimTree`](@ref).

### Worst case expected mean returns sets and covariance

  - Box sets, [`Box`](@ref).
  - Ellipse, [`Ellipse`](@ref).

#### Bootstrapping methods

  - ARCH methods, require `PyCall` and `ARCH` to be installed, [`ArchWC`](@ref).
    
      + Stationary bootstrap, [`StationaryBS`](@ref).
      + Circular bootstrap, [`CircularBS`](@ref).
      + Moving bootstrap, [`MovingBS`](@ref).

  - Normal, [`NormalWC`](@ref).
  - Delta, [`DeltaWC`](@ref).

#### Elliptical constraint error size estimation

  - Normal, [`KNormalWC`](@ref).
  - General, [`KGeneralWC`](@ref)

### Regression methods

#### Stepwise methods

  - Forward regression, [`FReg`](@ref).
  - Backward regression, [`BReg`](@ref).

#### Regression criteria

  - p-value threshold, [`PVal`](@ref).

  - Model quality indicators.
    
      + Akaike's Information Criterion, [`AIC`](@ref).
      + Corrected Akaike's Information Criterion for small sample sizes, [`AICC`](@ref).
      + Bayesian Information Criterion, [`BIC`](@ref).
      + R² of a linear model, [`RSq`](@ref).
      + adjusted R² for a linear model, [`AdjRSq`](@ref).

#### Dimensionality reduction

  - Principal component analysis (PCA) based regression, [`PCAReg`](@ref).
    
      + PCA target, [`PCATarget`](@ref).
      + Probabilistic PCA target, [`PPCATarget`](@ref).

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

### Worst case mean variance, [`WC`](@ref)

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

### Risk parity, [`RP`](@ref)

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

### Relaxed risk parity mean variance, [`RRP`](@ref)

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

These types of optimisations act on instances of [`HCPortfolio`](@ref).

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

[`NCO`](@ref) can take keyword arguments that define the supported by optimisations of [`Portfolio`](@ref). Since there are intra- and inter-cluster optimisations, it can take individual arguments for both. This means it can perform any combination of [`Portfolio`](@ref) optimisations, except for [`WC`](@ref).

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

#### NCO-RP, NCO-RRP, NCO-NOC-RP, NCO-NOC-RRP

##### Constraints

When applied to the intra-cluster optimisation the same constraint will be applied to all every cluster.

  - Minimum expected return constraint.
