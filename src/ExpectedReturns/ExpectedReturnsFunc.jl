"""
Compute the mean returns given a concrete type of [`AbstractReturnModel`](@ref), and returns array where each column is an asset and each row is an entry. 

For all methods:

- `compound`: if `true` the compute compound mean returns, if false compute uncompounded mean returns.
- `freq`: frequency at which the returns are logged, defaults to the average number of days in a trading year.

## Mean

### Arithmetic mean

```
ret_model(::MRet, returns; compound = true, freq = 252)
```

Compute the mean returns.

### Exponentially weighted arithmetic mean

```
ret_model(::EMRet, returns; compound = true, freq = 252, span = Int(ceil(freq / 1.4)))
```

Compute the exponentially weighted areithmetic mean returns. More recent returns are weighted more heavily.

- `span`: defines the weighing parameter, `α = 2 / (span + 1)`.

## Capital Asset Price Model (CAPM) returns

Compute the Capital Asset Price Model (CAPM) returns.

If the market returns are not provided, it estimates them by averaging across all assets and uses those as the market returns. These methods require computing the covariance between asset and market returns. These methods give a fairer value to a stock returns according to how they relate to the market.

- `cov_type`: a concrete type of [`AbstractRiskModel`](@ref) for computing the covariance of the assets to the market.

### CAPM with arithmetic mean

```
ret_model(
    ::CAPMRet,
    returns,
    market_returns = nothing;
    rf = 0.02,
    compound = true,
    freq = 252,
    cov_type::AbstractRiskModel = Cov(),
    target = 1.02^(1 / 252) - 1,
    fix_method::Union{SFix, DFix} = SFix(),
    span = Int(ceil(freq / 1.4)),
    )
```

Capital Asset Pricing Model (CAPM) returns. `CAPMRet()` uses the normal mean to estimate the mean market returns.

- `target`, `fix_method`, and `span` correspond to the same keyword arguments of [`risk_matrix`](@ref) for computing the covariance specified by `cov_type`.

### CAPM with exponentially weighted arithmetic mean

```
ret_model(
    ::ECAPMRet,
    returns,
    market_returns = nothing;
    rf = 0.02,
    compound = true,
    freq = 252,
    rspan = Int(ceil(freq / 1.4)),
    cov_type::AbstractRiskModel = Cov(),
    target = 1.02^(1 / 252) - 1,
    fix_method::Union{SFix, DFix} = SFix(),
    cspan = Int(ceil(freq / 1.4)),
)
```

Exponentially weighted Capital Asset Pricing Model (ECAPM) returns. `ECAPMRet()` uses the exponentially weighted mean to estimate the mean market returns.

- `rspan`: span for the exponentially weighted mean returns, same as `span` for `EMRet()` above.
- `cspan`: span for the exponentially weighted covariance, same as `span` for [`risk_matrix`](@ref).
"""
function ret_model(::MRet, returns; compound = true, freq = 252)
    if compound
        return prod(returns .+ 1, dims = 1) .^ (freq / size(returns, 1)) .- 1
    else
        return mean(returns, dims = 1) * freq
    end
end

function ret_model(
    ::EMRet,
    returns;
    compound = true,
    freq = 252,
    span = Int(ceil(freq / 1.4)),
)
    N = size(returns, 1)
    if compound
        return (1 .+ mean(returns, eweights(N, 2 / (span + 1)), dims = 1)) .^ freq .- 1
    else
        return mean(returns, eweights(N, 2 / (span + 1)), dims = 1) * freq
    end
end

function ret_model(
    ::CAPMRet,
    returns,
    market_returns = nothing;
    rf = 0.02,
    compound = true,
    freq = 252,
    cov_type::AbstractRiskModel = Cov(),
    target = 1.02^(1 / 252) - 1,
    fix_method::Union{SFix, DFix} = SFix(),
    span = Int(ceil(freq / 1.4)),
)
    β = _compute_betas(market_returns, returns, cov_type, target, fix_method, span)

    # Mean market return.
    mkt_mean_ret = ret_model(MRet(), returns[:, end]; compound = compound, freq = freq)

    # Capital asset pricing.
    return rf .+ β * (mkt_mean_ret - rf)
end

function ret_model(
    ::ECAPMRet,
    returns,
    market_returns = nothing;
    rf = 0.02,
    compound = true,
    freq = 252,
    rspan = Int(ceil(freq / 1.4)),
    cov_type::AbstractRiskModel = Cov(),
    target = 1.02^(1 / 252) - 1,
    fix_method::Union{SFix, DFix} = SFix(),
    cspan = Int(ceil(freq / 1.4)),
)
    β = _compute_betas(market_returns, returns, cov_type, target, fix_method, cspan)

    # Exponentially weighted mean market return.
    mkt_mean_ret =
        ret_model(EMRet(), returns[:, end]; compound = compound, freq = freq, span = rspan)

    # Capital asset pricing.
    return rf .+ β * (mkt_mean_ret - rf)
end

function _compute_betas(market_returns, returns, cov_type, target, fix_method, cspan)

    # Add the market returns to the right of the returns Array.
    if isnothing(market_returns)
        # Compute the market returns if it is not provided.
        returns = hcat(returns, mean(returns, dims = 2))
    else
        returns = hcat(returns, market_returns)
    end
    # Covariance with the market returns.
    cov_mtx = risk_matrix(
        cov_type,
        returns;
        target = target,
        fix_method = fix_method,
        freq = 1,
        span = cspan,
    )

    # The rightmost column is the covariance to the market.
    β = cov_mtx[:, end] / cov_mtx[end, end]
    β = β[1:(end - 1)]

    return β
end

"""
```
returns_from_prices(prices, log_ret = false)
```

Compute the returns from prices. 

- `prices`: array of prices, each column is an asset, each row an entry.
- `log_ret`: if `false` compute the normal returns, if `true` compute the logarithmic returns.
"""
function returns_from_prices(prices, log_ret = false)
    if log_ret
        return log.(prices[2:end, :] ./ prices[1:(end - 1), :])
    else
        return prices[2:end, :] ./ prices[1:(end - 1), :] .- 1
    end
end

"""
```
prices_from_returns(returns, log_ret = false)
```

Compute the prices from returns. 

- `returns`: array of returns, each column is an asset, each row an entry.
- `log_ret`: if `false` asume normal returns, if `true` asume logarithmic returns.
"""
function prices_from_returns(returns, log_ret = false)
    if log_ret
        ret = exp.(returns)
    else
        ret = 1 .+ returns
    end

    return cumprod([1; returns])
end