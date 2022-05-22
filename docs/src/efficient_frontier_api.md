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
min_volatility!(
    portfolio::EfficientMeanVar,
    optimiser = Ipopt.Optimizer,
    silent = true,
)
max_return(portfolio::EfficientMeanVar, optimiser = Ipopt.Optimizer, silent = true)
max_sharpe!(
    portfolio::EfficientMeanVar,
    rf = portfolio.rf,
    optimiser = Ipopt.Optimizer,
    silent = true,
)
max_quadratic_utility!(
    portfolio::EfficientMeanVar,
    risk_aversion = portfolio.risk_aversion,
    optimiser = Ipopt.Optimizer,
    silent = true,
)
efficient_return!(
    portfolio::EfficientMeanVar,
    target_ret = portfolio.target_ret,
    optimiser = Ipopt.Optimizer,
    silent = true,
)
efficient_risk!(
    portfolio::EfficientMeanVar,
    target_volatility = portfolio.target_volatility,
    optimiser = Ipopt.Optimizer,
    silent = true,
)
refresh_model!(portfolio::AbstractEfficientMeanVar)
portfolio_performance(portfolio::EfficientMeanVar; verbose = false)
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