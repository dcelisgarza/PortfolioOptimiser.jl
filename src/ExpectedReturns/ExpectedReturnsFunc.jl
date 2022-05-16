function ret_model(::MeanRet, returns; compound = true, freq = 252)
    if compound
        return vec(prod(returns .+ 1, dims = 1) .^ (freq / size(returns, 1)) .- 1)
    else
        return vec(mean(returns, dims = 1) * freq)
    end
end

function ret_model(
    ::ExpMeanRet,
    returns;
    compound = true,
    freq = 252,
    span = Int(ceil(freq / 1.4)),
)
    N = size(returns, 1)
    if compound
        return vec(
            (1 .+ mean(returns, eweights(N, 2 / (span + 1)), dims = 1)) .^ freq .- 1,
        )
    else
        return vec(mean(returns, eweights(N, 2 / (span + 1)), dims = 1) * freq)
    end
end

function ret_model(
    ::CAPMRet,
    returns,
    market_returns = nothing;
    rf = 0.02,
    compound = true,
    freq = 252,
)
    # Add the market returns to the right of the returns Array.
    if isnothing(market_returns)
        # Calculate the market returns if it is not provided.
        returns = hcat(returns, mean(returns, dims = 2))
    else
        returns = hcat(returns, market_returns)
    end
    # Covariance with the market returns.
    cov_mtx = cov(returns)

    # The rightmost column is the covariance to the market.
    β = cov_mtx[:, end] / cov_mtx[end, end]
    β = β[1:(end - 1)]

    # Mean market return.
    if compound
        mkt_mean_ret = prod(1 .+ returns[:, end]) .^ (freq / size(returns, 1)) .- 1
    else
        mkt_mean_ret = mean(returns[:, end]) * freq
    end

    # Capital asset pricing.
    return rf .+ β * (mkt_mean_ret - rf)
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