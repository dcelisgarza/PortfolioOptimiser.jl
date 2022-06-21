# Efficient Frontier

```@docs
AbstractEfficient
_refresh_add_var_and_constraints
_function_vs_portfolio_val_warn
_val_compare_benchmark
```

## Mean-variance

```@docs
AbstractEffMeanVar
EffMeanVar
EffMeanVar(
    tickers,
    mean_ret,
    cov_mtx;
    weight_bounds = (0, 1),
    rf = 0.02,
    market_neutral = false,
    risk_aversion = 1.0,
    target_risk = rank(cov_mtx) < size(cov_mtx, 1) ? 1 / sum(diag(cov_mtx)) :
                        sqrt(1 / sum(inv(cov_mtx))),
    target_ret = mean(mean_ret),
    extra_vars = [],
    extra_constraints = [],
    extra_obj_terms = [],
)
min_risk!(
    portfolio::EffMeanVar;
    optimiser = Ipopt.Optimizer,
    silent = true,
    optimiser_attributes = (),
)
max_return(portfolio::EffMeanVar; optimiser = Ipopt.Optimizer, silent = true, optimiser_attributes = (),)
max_sharpe!(
    portfolio::EffMeanVar,
    rf = portfolio.rf;
    optimiser = Ipopt.Optimizer,
    silent = true,
    optimiser_attributes = (),
)
max_utility!(
    portfolio::EffMeanVar,
    risk_aversion = portfolio.risk_aversion;
    optimiser = Ipopt.Optimizer,
    silent = true,
    optimiser_attributes = (),
)
efficient_return!(
    portfolio::EffMeanVar,
    target_ret = portfolio.target_ret;
    optimiser = Ipopt.Optimizer,
    silent = true,
    optimiser_attributes = (),
)
efficient_risk!(
    portfolio::EffMeanVar,
    target_risk = portfolio.target_risk;
    optimiser = Ipopt.Optimizer,
    silent = true,
    optimiser_attributes = (),
)
refresh_model!(portfolio::AbstractEffMeanVar)
portfolio_performance(portfolio::EffMeanVar; rf = portfolio.rf, verbose = false)
```

## Mean-semivariance

```@docs
AbstractEffMeanSemivar
EffMeanSemivar
EffMeanSemivar(
    tickers,
    mean_ret,
    returns;
    weight_bounds = (0.0, 1.0),
    freq = 252,
    target = 0,
    rf = 0.02,
    market_neutral = false,
    risk_aversion = 1.0,
    target_risk = std(returns),
    target_ret = mean(mean_ret),
    extra_vars = [],
    extra_constraints = [],
    extra_obj_terms = [],
)
refresh_model!(portfolio::AbstractEffMeanSemivar)
portfolio_performance(portfolio::EffMeanSemivar; rf = portfolio.rf, verbose = false)
```

## Critical Value at Risk (CVaR)

```@docs
```

## Critical Drawdown at Risk (CDaR)

```@docs
```