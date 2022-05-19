# API

## Objective Functions

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

## Expected Returns

```@docs
AbstractReturnModel
ret_model
returns_from_prices
prices_from_returns
```

## Risk Models

```@docs
AbstractFixPosDef
AbstractRiskModel
risk_matrix
make_pos_def
cov2cor
```

## Portfolio Optimisation

```@docs
AbstractPortfolioOptimiser
custom_optimiser!
custom_nloptimiser!
add_sector_constraint!
_refresh_model!
_create_weight_bounds
_make_weight_sum_constraint!
_add_var_to_model!
_add_constraint_to_model!
_add_to_objective!
```

### Efficient Frontier

```@docs
AbstractEfficient
```

#### Mean-variance

```@docs
```

#### Mean-semivariance

```@docs
EfficientSemiVar
```

#### Critical Value at Risk (CVaR)

```@docs
```

#### Critical Drawdown at Risk (CDaR)

```@docs
```

### Hierarchical Risk Parity

```@docs
```

### Black-Litterman

```@docs
```

## Critical line algorithm

```@docs
```

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
