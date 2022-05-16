"""
Compute the mean returns given a concrete type of [`AbstractReturnModel`](@ref), and returns array where each column is an asset and each row is an entry. If `compound` is true, calcualte the compounded returns, the `freq` parameter refers to the frequency at which the returns are logged, default corresponds to the average number of trading days in a year.

```
ret_model(::MRet, returns; compound = true, freq = 252)
```
Compute the mean returns.

```
ret_model(::EMRet, returns; compound = true, freq = 252, span = Int(ceil(freq / 1.4)))
```
Compute the exponentially weighted mean returns using the span as a parameter, `α = 2 / (span + 1)`.

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
Compute the Capital Asset Price Model returns. If the market returns are not provided, it estimates them by averaging across all assets and uses those as the market returns. This gives each asset a fair market value. 

This method requires computing the covariance between the asset and market returns. The keyword argument `cov_type` corresponds to a concrete type of [`AbstractRiskModel`](@ref). The keyword arguments, `target`, `fix_method` and `span` correspond to the same keyword arguments of [`risk_matrix`](@ref).

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
    span = Int(ceil(freq / 1.4)),
)
```
Same as the Capital Asset Pricing Model above, but instead of using the normal mean to estimate the mean market return, it uses the exponentially weighted mean. The values of `rspan` and `cspan` are the `span`` values corresponding to the exponentially weighted mean, and exponentially weighted covariance/semicovariance matrix respectively.
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
    β = _calculate_betas(market_returns, returns, cov_type, target, fix_method, span)

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
    β = _calculate_betas(market_returns, returns, cov_type, target, fix_method, cspan)

    # Exponentially weighted mean market return.
    mkt_mean_ret =
        ret_model(EMRet(), returns[:, end]; compound = compound, freq = freq, span = rspan)

    # Capital asset pricing.
    return rf .+ β * (mkt_mean_ret - rf)
end

function _calculate_betas(market_returns, returns, cov_type, target, fix_method, cspan)

    # Add the market returns to the right of the returns Array.
    if isnothing(market_returns)
        # Calculate the market returns if it is not provided.
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

function returns_from_prices(prices, log_ret = false)
    if log_ret
        return log.(prices[2:end, :] ./ prices[1:(end - 1), :])
    else
        return prices[2:end, :] ./ prices[1:(end - 1), :] .- 1
    end
end

function prices_from_returns(returns, log_ret = false)
    if log_ret
        ret = exp.(returns)
    else
        ret = 1 .+ returns
    end

    return cumprod([1; returns])
end