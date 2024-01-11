# Asset allocation

```@docs
AbstractAllocation
Allocation
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
_short_allocation
_sub_allocation
_clean_zero_shares
```
