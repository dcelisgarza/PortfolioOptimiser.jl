# Efficient Frontier

```@docs
AbstractEfficient
_refresh_add_var_and_constraints
_function_vs_portfolio_val_warn
_val_compare_benchmark
```

## Mean-variance

```@docs
AbstractMeanVar
MeanVar
MeanVar(
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
    portfolio::MeanVar;
    optimiser = Ipopt.Optimizer,
    silent = true,
    optimiser_attributes = (),
)
max_return(portfolio::MeanVar; optimiser = Ipopt.Optimizer, silent = true, optimiser_attributes = (),)
max_sharpe!(
    portfolio::MeanVar;
    rf = portfolio.rf,
    optimiser = Ipopt.Optimizer,
    silent = true,
    optimiser_attributes = (),
)
max_quadratic_utility!(
    portfolio::MeanVar;
    risk_aversion = portfolio.risk_aversion,
    optimiser = Ipopt.Optimizer,
    silent = true,
    optimiser_attributes = (),
)
efficient_return!(
    portfolio::MeanVar;
    target_ret = portfolio.target_ret,
    optimiser = Ipopt.Optimizer,
    silent = true,
    optimiser_attributes = (),
)
efficient_risk!(
    portfolio::MeanVar;
    target_volatility = portfolio.target_volatility,
    optimiser = Ipopt.Optimizer,
    silent = true,
    optimiser_attributes = (),
)
refresh_model!(portfolio::AbstractMeanVar)
portfolio_performance(portfolio::MeanVar; rf = portfolio.rf, verbose = false)
```

## Mean-semivariance

```@docs
AbstractMeanSemivar
MeanSemivar
MeanSemivar(
    tickers,
    mean_ret,
    returns;
    weight_bounds = (0.0, 1.0),
    freq = 252,
    benchmark = 0,
    rf = 0.02,
    market_neutral = false,
    risk_aversion = 1.0,
    target_semidev = std(returns),
    target_ret = mean(mean_ret),
    extra_vars = [],
    extra_constraints = [],
    extra_obj_terms = [],
)
refresh_model!(portfolio::AbstractMeanSemivar)
portfolio_performance(portfolio::MeanSemivar; rf = portfolio.rf, verbose = false)
```

## Critical Value at Risk (CVaR)

```@docs
```

## Critical Drawdown at Risk (CDaR)

```@docs
```