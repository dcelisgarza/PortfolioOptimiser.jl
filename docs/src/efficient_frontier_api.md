# Efficient Frontier

```@docs
AbstractEfficient
_refresh_add_var_and_constraints
_function_vs_portfolio_val_warn
_val_compare_benchmark
```

## Mean-variance

```@docs
AbstractEfficientMeanVar
EfficientMeanVar
EfficientMeanVar(
    tickers,
    mean_ret,
    cov_mtx;
    weight_bounds = (0, 1),
    rf = 0.02,
    market_neutral = false,
    risk_aversion = 1.0,
    target_volatility = rank(cov_mtx) < size(cov_mtx, 1) ? 1 / sum(diag(cov_mtx)) :
                        sqrt(1 / sum(inv(cov_mtx))),
    target_ret = mean(mean_ret),
    extra_vars = [],
    extra_constraints = [],
    extra_obj_terms = [],
)
```

## Mean-semivariance

```@docs
EfficientSemiVar
```

## Critical Value at Risk (CVaR)

```@docs
```

## Critical Drawdown at Risk (CDaR)

```@docs
```