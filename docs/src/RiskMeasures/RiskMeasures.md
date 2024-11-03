# Types

## Specifying solvers

Some risk measures require solvers to compute the risk measure via [`calc_risk`](@ref). When using high level functions that take in an instance of [`Portfolio`](@ref)/[`HCPortfolio`](@ref) as an argument, the solvers will be taken from it. However, it is possible to override them by directly providing the solvers to the risk measure instance.

The solvers can be specified by any container which implements the [`AbstractDict`](https://docs.julialang.org/en/v1/base/collections/#Base.AbstractDict) interfaces. This dictionary must contain the key-value pairs of any solvers one wishes to use. Where the key can be of any type, but the value must be a dictionary with the following key-value pairs.

  - `:solver`: defines the solver to use. One can also use [`JuMP.optimizer_with_attributes`](https://jump.dev/JuMP.jl/stable/api/JuMP/#optimizer_with_attributes) to direcly provide a solver with attributes already attached.
  - `:check_sol`: (optional) defines the keyword arguments passed on to [`JuMP.is_solved_and_feasible`](https://jump.dev/JuMP.jl/stable/api/JuMP/#is_solved_and_feasible) for accepting/rejecting solutions.
  - `:params`: (optional) defines solver-specific parameters.
  - `:add_bridges`: (optional) value of the `add_bridges` kwarg of [`JuMP.Model`](https://jump.dev/JuMP.jl/stable/api/JuMP/#Model), if not provided defaults to `true`.

```@setup solvers_dict
using JuMP, Clarabel
```

```@example solvers_dict
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

## Abstract types

[`PortfolioOptimiser`](https://github.com/dcelisgarza/PortfolioOptimiser.jl/)'s risk measures are implemented using Julia's type hierarchy. This makes it easy to add new ones by implementing relevant types and methods.

```@docs
PortfolioOptimiser.AbstractRiskMeasure
RiskMeasure
HCRiskMeasure
```

## Settings

[`PortfolioOptimiser`](https://github.com/dcelisgarza/PortfolioOptimiser.jl/) lets users provide multiple risk measures at the same time. They can be added to the risk expression in the optimisation objective, or used to set upper risk bounds. If multiple risk measures are to be used as part of the risk expression in the optimisation objective, it makes sense provide a way to scale each risk contribution in the objective, especially because different risk expressions may have different units, and therefore require appropriate scaling.

Furthermore, certain risk measures make use of Ordered Weight Array formulations, for which there is an expensive exact formulation, and an approximate, tunable one. We let users decide which one to use, and if they want to use the approximate one they can tune it as they see fit.

The following structures provide said functionality.

```@docs
RMSettings
HCRMSettings
OWASettings
```

## Risk measures

These risk measures are compatible with both [`Portfolio`](@ref) and [`HCPortfolio`](@ref) optimisations. Different risk measures account for different aspects of the returns.

### Dispersion risk measures

These measure the spread of the returns distribution.

#### Full dispersion risk measures

These measure how far the returns deviate from the mean in both the positive and negative directions.

```@docs
PortfolioOptimiser.SDFormulation
PortfolioOptimiser.SDSquaredFormulation
QuadSD
SOCSD
SimpleSD
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

#### Downside dispersion risk measures

These measure how far the returns deviate from the mean in the negative direction.

```@docs
SSD
FLPM
SLPM
SKurt
SSkew
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

These risk measures are compatible with [`HCPortfolio`](@ref). Different risk measures account for different aspects of the returns.

### Dispersion hierarchical risk measures

These measure the characteristics of the returns distribution.

#### Full dispersion hierarchical risk measures

These measure how far the returns deviate from the mean in both the positive and negative directions.

```@docs
Variance
```

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
