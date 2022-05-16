# API

## Objective functions

```@docs
port_variance
port_return
sharpe_ratio
L2_reg
quadratic_utility
semi_ret
transaction_cost
ex_ante_tracking_error
ex_post_tracking_error
logarithmic_barrier
kelly_objective
```

## Risk models

```@docs
AbstractRiskModel
risk_matrix
```

## Expected returns

```@docs
AbstractReturnModel
ret_model
```

## Efficient Frontier

### Mean-var

### Mean-semivar

```@docs
EfficientSemiVar
```

### Critical Value at Risk (CVaR)

### Critical Drawdown at Risk (CDaR)

## Hierarchical Risk Parity Optimisation

## Black-Litterman Optimisation

## Critical line algorithm

## Asset allocation

```@docs
AbstractAllocation
Allocation{T1,T2,T3}
Allocation(
    type::AbstractAllocation,
    portfolio::AbstractPortfolioOptimiser,
    latest_prices::AbstractVector;
    investment = 1e4,
    rounding = 1,
    reinvest = false,
    short_ratio = nothing,
    optimiser = HiGHS.Optimizer,
    silent = true,
)
roundmult
```
