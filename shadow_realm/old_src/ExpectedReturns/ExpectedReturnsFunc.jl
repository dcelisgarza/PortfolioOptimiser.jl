"""
Compute the mean returns given a concrete type of [`AbstractReturnModel`](@ref), and returns array where each column is an asset and each row is an entry. 

For all methods:

- `compound`: if `true` the compute compound mean returns, if false compute uncompounded mean returns.

## Mean

### Arithmetic mean

```
ret_model(::MRet, returns; compound = true)
```

Compute the mean returns.

### Exponentially weighted arithmetic mean

```
ret_model(::EMRet, returns; compound = true, span = ceil(Int,4*size(returns, 1) / log(size(returns, 1) + 2)))
```

Compute the exponentially weighted areithmetic mean returns. More recent returns are weighted more heavily.

- `span`: defines the weighing parameter, `α = 2 / (span + 1)`.

## Capital Asset Price Model (CAPM) returns

Compute the Capital Asset Price Model (CAPM) returns.

If the market returns are not provided, it estimates them by averaging across all assets and uses those as the market returns. These methods require computing the covariance between asset and market returns. These methods give a fairer value to a stock returns according to how they relate to the market.

- `cov_method`: a concrete type of [`AbstractRiskModel`](@ref) for computing the covariance of the assets to the market.

### CAPM with arithmetic mean

```
ret_model(
    ::CAPMRet,
    returns,
    market_returns = nothing;
    rf = 1.02^(1 / 252) - 1,
    compound = true,
    cov_method::AbstractRiskModel = Cov(),
    target = 1.02^(1 / 252) - 1,
    fix_method::AbstractFixPosDef = SFix(),
    span = ceil(Int,4*size(returns, 1) / log(size(returns, 1) + 2)),
    custom_cov_estimator = nothing,
    custom_cov_args = (),
    custom_cov_kwargs = (),
    )
```

Capital Asset Pricing Model (CAPM) returns. `CAPMRet()` uses the normal mean to estimate the mean market returns.

- `target`, `fix_method`, and `span` correspond to the same keyword arguments of [`cov`](@ref) for computing the covariance specified by `cov_method`.

### CAPM with exponentially weighted arithmetic mean

```
ret_model(
    ::ECAPMRet,
    returns,
    market_returns = nothing;
    rf = 1.02^(1 / 252) - 1,
    compound = true,
    rspan = ceil(Int,4*size(returns, 1) / log(size(returns, 1) + 2)),
    cov_method::AbstractRiskModel = ECov(),
    target = 1.02^(1 / 252) - 1,
    fix_method::AbstractFixPosDef = SFix(),
    cspan = ceil(Int,4*size(returns, 1) / log(size(returns, 1) + 2)),
    custom_cov_estimator = nothing,
    custom_cov_args = (),
    custom_cov_kwargs = (),
)
```

Exponentially weighted Capital Asset Pricing Model (ECAPM) returns. `ECAPMRet()` uses the exponentially weighted mean to estimate the mean market returns.

- `rspan`: span for the exponentially weighted mean returns, same as `span` for `EMRet()` above.
- `cspan`: span for the exponentially weighted covariance, same as `span` for [`cov`](@ref).
"""
function ret_model(::MRet, returns; compound = true, frequency = 1)
    if compound
        return vec(prod(returns .+ 1; dims = 1) .^ (frequency / size(returns, 1)) .- 1)
    else
        return vec(mean(returns; dims = 1)) * frequency
    end
end

function ret_model(::EMRet,
                   returns;
                   compound = true,
                   frequency = 1,
                   span = ceil(Int, 4 * size(returns, 1) / log(size(returns, 1) + 2)),)
    N = size(returns, 1)
    if compound
        return vec((1 .+ mean(returns, eweights(N, 2 / (span + 1)); dims = 1)) .^
                   frequency .- 1)
    else
        return vec(mean(returns, eweights(N, 2 / (span + 1)); dims = 1)) * frequency
    end
end

function ret_model(::CAPMRet,
                   returns,
                   market_returns = nothing;
                   rf = 1.02^(1 / 252) - 1,
                   compound = true,
                   frequency = 1,
                   cov_method::AbstractRiskModel = Cov(),
                   target = 1.02^(1 / 252) - 1,
                   fix_method::AbstractFixPosDef = SFix(),
                   span = ceil(Int, 4 * size(returns, 1) / log(size(returns, 1) + 2)),
                   scale = nothing,
                   custom_cov_estimator = nothing,
                   custom_cov_args = (),
                   custom_cov_kwargs = (),)
    β, returns = _compute_betas(market_returns,
                                returns,
                                cov_method,
                                target,
                                fix_method,
                                span,
                                scale,
                                custom_cov_estimator,
                                custom_cov_args,
                                custom_cov_kwargs)

    # Mean market return.
    mkt_mean_ret = ret_model(MRet(), returns[:, end]; compound = compound,
                             frequency = frequency)[1]

    # Capital asset pricing.
    return rf .+ β * (mkt_mean_ret - rf)
end

function ret_model(::ECAPMRet,
                   returns,
                   market_returns = nothing;
                   rf = 1.02^(1 / 252) - 1,
                   compound = true,
                   frequency = 1,
                   rspan = ceil(Int, 4 * size(returns, 1) / log(size(returns, 1) + 2)),
                   cov_method::AbstractRiskModel = ECov(),
                   target = 1.02^(1 / 252) - 1,
                   fix_method::AbstractFixPosDef = SFix(),
                   cspan = ceil(Int, 4 * size(returns, 1) / log(size(returns, 1) + 2)),
                   scale = nothing,
                   custom_cov_estimator = nothing,
                   custom_cov_args = (),
                   custom_cov_kwargs = (),)
    β, returns = _compute_betas(market_returns,
                                returns,
                                cov_method,
                                target,
                                fix_method,
                                cspan,
                                scale,
                                custom_cov_estimator,
                                custom_cov_args,
                                custom_cov_kwargs)

    # Exponentially weighted mean market return.
    mkt_mean_ret = ret_model(EMRet(),
                             returns[:, end];
                             compound = compound,
                             frequency = frequency,
                             span = rspan,)[1]

    # Capital asset pricing.
    return rf .+ β * (mkt_mean_ret - rf)
end

function _compute_betas(market_returns,
                        returns,
                        cov_method,
                        target,
                        fix_method,
                        cspan,
                        scale,
                        custom_cov_estimator,
                        custom_cov_args,
                        custom_cov_kwargs)

    # Add the market returns to the right of the returns Array.
    if isnothing(market_returns)
        # Compute the market returns if it is not provided.
        returns = hcat(returns, mean(returns; dims = 2))
    else
        returns = hcat(returns, market_returns)
    end
    # Covariance with the market returns.
    cov_mtx = cov(cov_method,
                  returns,
                  target,
                  fix_method,
                  cspan,
                  scale,
                  custom_cov_estimator,
                  custom_cov_args...;
                  custom_cov_kwargs...,)

    # The rightmost column is the covariance to the market.
    β = cov_mtx[:, end] / cov_mtx[end, end]
    β = β[1:(end - 1)]

    return β, returns
end

"""
```
returns_from_prices(prices, log_ret = false)
```

Compute the returns from prices. 

- `prices`: array of prices, each column is an asset, each row an entry.
- `log_ret`: if `false` compute the normal returns, if `true` compute the logarithmic returns.
"""
function returns_from_prices(prices,
                             log_ret = false;
                             capm = false,
                             market_returns = nothing,
                             rf = 1.02^(1 / 252) - 1,
                             cov_method::AbstractRiskModel = ECov(),
                             target = 1.02^(1 / 252) - 1,
                             fix_method::AbstractFixPosDef = SFix(),
                             span = size(prices, 1) != 0 ?
                                    ceil(Int,
                                         4 * (size(prices, 1) - 1) /
                                         log(size(prices, 1) - 1 + 2)) : 1,
                             scale = nothing,
                             custom_cov_estimator = nothing,
                             custom_cov_args = (),
                             custom_cov_kwargs = (),)
    returns = if log_ret
        log.(prices[2:end, :] ./ prices[1:(end - 1), :])
    else
        prices[2:end, :] ./ prices[1:(end - 1), :] .- 1
    end

    replace_nan(v) = map!(x -> isnan(x) ? zero(x) : x, v, v)
    replace_inf(v) = map!(x -> isinf(x) ? zero(x) : x, v, v)

    map(replace_nan, eachcol(returns))
    map(replace_inf, eachcol(returns))

    if capm
        β, returns = _compute_betas(market_returns,
                                    returns,
                                    cov_method,
                                    target,
                                    fix_method,
                                    span,
                                    scale,
                                    custom_cov_estimator,
                                    custom_cov_args,
                                    custom_cov_kwargs)

        returns = rf .+ (returns[:, end] .- rf) * β'
    end

    return returns
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

    # ret[1, :] .= 1
    # println(size(ret, 2))
    ret = [ones(1, size(ret, 2)); ret]
    return cumprod(ret; dims = 1)
end
