# Risk Measures

Risk measures are the backbone of portfolio optimisation. They allow for the quantification of risk in different ways. This section describes the various risk measures included in [`PortfolioOptimiser`](https://github.com/dcelisgarza/PortfolioOptimiser.jl), explains their properties, formulations and compatibility with the various optimisation types in the library.

## Solvers

Some optimisations and risk measures use [`JuMP`](https://github.com/jump-dev/JuMP.jl) models, and therefore require solvers to solve/compute. These can be specified by any container which implements the [`AbstractDict`](https://docs.julialang.org/en/v1/base/collections/#Base.AbstractDict) interfaces. This dictionary must contain key-value pairs for all solvers one wants to use. The key can be of any type, but the value must be an abstract dictionary with the following key-value pairs.

  - `:solver`: defines the solver to be used. One can also use [`JuMP.optimizer_with_attributes`](https://jump.dev/JuMP.jl/stable/api/JuMP/#optimizer_with_attributes) to direcly provide a solver with attributes already attached.
  - `:check_sol`: (optional) defines the keyword arguments passed on to [`JuMP.is_solved_and_feasible`](https://jump.dev/JuMP.jl/stable/api/JuMP/#is_solved_and_feasible) for accepting/rejecting solutions.
  - `:params`: (optional) defines solver-specific parameters/attributes.
  - `:add_bridges`: (optional) value of the `add_bridges` kwarg of [`JuMP.set_optimizer`](https://jump.dev/JuMP.jl/stable/api/JuMP/#JuMP.set_optimizer), if not provided defaults to `true`.

```@example solvers_dict
using JuMP, Clarabel
solvers = Dict(
               # Key-value pair for the solver, solution acceptance 
               # criteria, model bridge argument, and solver attributes.
               :Clarabel => Dict(
                                 # Solver we wish to use.
                                 :solver => Clarabel.Optimizer,
                                 # (Optional) Solution acceptance criteria.
                                 :check_sol => (allow_local = true, allow_almost = true),
                                 # (Optional) Solver-specific attributes.
                                 :params => Dict("verbose" => false),
                                 # (Optional) Flag for adding JuMP bridges to JuMP.Model()
                                 # defaults to true (https://jump.dev/JuMP.jl/stable/api/JuMP/#Model).
                                 :add_bridges => false))
```

Users are able to provide multiple solvers by adding additional key-value pairs to the top-level dictionary as in the following snippet.

```@example solvers_dict
solvers = Dict(:Clarabel_1 => Dict(:solver => Clarabel.Optimizer,
                                   :check_sol => (allow_local = true, allow_almost = true),
                                   :params => Dict("verbose" => false)),
               # Provide solver with pre-attached attributes and no arguments 
               # for the `JuMP.is_solved_and_feasible` function.
               :Clarabel_2 => Dict(:solver => JuMP.optimizer_with_attributes(Clarabel.Optimizer,
                                                                             "max_step_fraction" => 0.75),
                                   # Do not add JuMP bridges to JuMP.Model()
                                   :add_bridges => false))
```

[`PortfolioOptimiser`](https://github.com/dcelisgarza/PortfolioOptimiser.jl) will iterate over the solvers until it finds the first one to successfully solve the problem. If a deterministic ordering is desired (for example when going from relaxed to strict tolerances), use an [`OrderedDict`](https://juliacollections.github.io/OrderedCollections.jl/dev/ordered_containers/#OrderedDicts).

## Settings

### Multiple risk measures simultaneously

Portfolio Optimisation has typically limited itself to using a single risk measure. However, some risk measures can/must use internal parameters in order to compute the risk. There is nothing preventing us from using multiple measures simultaneously, or even multiple instances of the same risk measure with different hyperparameters. This can be achieved via multiple objective optimisation (not currently supported, but may in the future), or a scalarisation procedure [`PortfolioOptimiser.AbstractScalarisation`](@ref). Each instance's contribution to the overall risk expression is tunable via a weight parameter (called `scale` for disambiguation purposes).

For risk measures that can be used in optimisations using [`JuMP`](https://github.com/jump-dev/JuMP.jl) models, we must also provide a way to define their upper bound, and whether or not they should be included in the risk expression or used only for setting a risk upper bound.

For these purposes, we have two special structures that are to be used in optimisations, and not simply as performance metrics. Which one to use depends on what optimisation types the risk measure is compatible with.

```@docs
PortfolioOptimiser.AbstractRMSettings
RMSettings
HCRMSettings
```

### Ordered Weight Array settings

Certain risk measures make use of Ordered Weight Array formulations in optimisations which use [`JuMP`](https://github.com/jump-dev/JuMP.jl) models. [`PortfolioOptimiser`](https://github.com/dcelisgarza/PortfolioOptimiser.jl/) implements two formulations.

  - An exact but expensive formulation.
  - An approximate, tunable one based on [3D Power Cones](https://jump.dev/JuMP.jl/stable/tutorials/conic/tips_and_tricks/#PowerCone).

We therefore provide a structure for dispatching the exact method, and a structure for tuning and dispatching the approximate one.

```@docs
PortfolioOptimiser.OWAFormulation
OWAApprox
OWAExact
```

## Risk measures

### Abstract types

[`PortfolioOptimiser`](https://github.com/dcelisgarza/PortfolioOptimiser.jl/)'s risk measures are implemented using `Julia`'s type hierarchy. This makes it easy to add new ones by implementing the relevant types and methods.

```@docs
PortfolioOptimiser.AbstractRiskMeasure
RiskMeasure
HCRiskMeasure
NoOptRiskMeasure
RiskMeasureSigma
RiskMeasureMu
HCRiskMeasureMu
NoOptRiskMeasureMu
RiskMeasureTarget       
HCRiskMeasureTarget
RiskMeasureSolvers
HCRiskMeasureSolvers
RiskMeasureOWA
RiskMeasureSkew
```

These risk measures are compatible with all optimisation types that accept a risk measure.

### Support functions

Some risk measures require the computation of certain statistics, these are performed by the following functions.

```@docs
PortfolioOptimiser.calc_ret_mu
PortfolioOptimiser.calc_target_ret_mu
```

### Constants

Some risk measures can be classified by the properties they contain. They are not types because there is overlap in some of them, if needing .

```@docs
PortfolioOptimiser.RMSolvers
PortfolioOptimiser.RMSigma
PortfolioOptimiser.RMSkew
PortfolioOptimiser.RMOWA
PortfolioOptimiser.RMMu
PortfolioOptimiser.RMTarget
```

### Dispersion risk measures

These measure the spread of the returns distribution.

#### Full dispersion risk measures

These measure how far the returns deviate from the mean in both the positive and negative directions.

```@docs
PortfolioOptimiser.VarianceFormulation
Quad
SOC
Variance
SD
MAD
Kurt
RG
CVaRRG
TGRG
GMD
Skew
BDVariance
```

#### Tracking and turnover risk measures

```@docs
PortfolioOptimiser.AbstractTR
PortfolioOptimiser.TrackingErr
TrackRet
TrackWeight
NoTracking
TR
NoTR
```

#### Downside dispersion risk measures

These measure how far the returns deviate from the mean in the negative direction.

```@docs
SSD
FLPM
SLPM
SKurt
SSkew
Kurtosis
SKurtosis
```

#### Worst case variance

```@docs
PortfolioOptimiser.WorstCaseSet
Box
Ellipse
NoWC
```

### Downside risk measures

These measure different aspects of the tail (negative side) of the returns distribution.

```@docs
WR
CVaR
EVaR
RLVaR
TG
```

### Drawdown risk measures

These measure the drops in portfolio value from local maxima to subsequent local minima.

```@docs
MDD
ADD
UCI
CDaR
EDaR
RLDaR
```

### Linear moments (L-moments)

This is used to measure linear moments of the returns distribution.

```@docs
OWA
```

## Hierarchical risk measures

These risk measures are compatible with . Different risk measures account for different aspects of the returns.

### Dispersion hierarchical risk measures

These measure the characteristics of the returns distribution.

#### Downside dispersion hierarchical risk measures

These measure how far the returns deviate from the mean in the negative direction.

```@docs
SVariance
```

### Downside hierarchical risk measures

These measure different aspects of the tail (negative side) of the returns distribution.

```@docs
VaR
```

### Drawdown hierarchical risk measures

These measure the drops in portfolio value from local maxima to subsequent local minima.

```@docs
DaR
DaR_r
MDD_r
ADD_r
UCI_r
CDaR_r
EDaR_r
RLDaR_r
```

### Equal risk contribution

This assumes the risk is equally distributed among the variables.

```@docs
Equal
```
