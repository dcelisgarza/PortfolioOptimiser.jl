"""
```
roundmult(val, prec [, args...] [; kwargs...])
```

Round a number to a multiple of `prec`. Uses the same defaults and has the same `args` and `kwargs` of the built-in `Base.round`.

Equivalent to:

```
round(div(val, prec) * prec, args...; kwargs...)
```
"""
function roundmult(val, prec, args...; kwargs...)
    return round(div(val, prec) * prec, args...; kwargs...)
end
function _greedy_sub_allocation!(tickers, weights, latest_prices, investment, rounding,
                                 ratio)
    if isempty(tickers)
        return String[], Vector{eltype(latest_prices)}(undef, 0),
               Vector{eltype(latest_prices)}(undef, 0),
               Vector{eltype(latest_prices)}(undef, 0),
               Vector{eltype(latest_prices)}(undef, 0), zero(eltype(latest_prices))
    end

    idx = sortperm(weights; rev = true)
    weights = weights[idx]
    tickers = tickers[idx]
    latest_prices = latest_prices[idx]

    N = length(tickers)
    available_funds = investment
    shares = zeros(typeof(rounding), N)
    weights ./= sum(weights)

    # First loop
    for i ∈ 1:N
        price = latest_prices[i]
        n_shares = roundmult(weights[i] * investment / price, rounding, RoundDown)
        cost = n_shares * price
        if cost > available_funds
            break
        end
        available_funds -= cost
        shares[i] = n_shares
    end

    # Second loop
    while available_funds > 0
        # Calculate equivalent continuous weights of what has already been bought.
        current_weights = latest_prices .* shares
        current_weights /= sum(current_weights)

        deficit = weights - current_weights

        # Try to buy tickers whose deficit is the greatest.
        idx = argmax(deficit)
        price = latest_prices[idx]

        # If we can't afford it, go through the rest of the tickers from highest deviation to lowest
        while price > available_funds
            deficit[idx] = 0
            idx = argmax(deficit)
            if deficit[idx] <= 0
                break
            end
            price = latest_prices[idx]
        end
        if deficit[idx] <= 0
            break
        end
        # Buy one share*rounding at a time.
        shares[idx] += rounding
        available_funds -= price
    end

    cost = latest_prices .* shares
    allocated_weights = cost / sum(cost)
    allocated_weights *= ratio

    return tickers, shares, latest_prices, cost, allocated_weights, available_funds
end
function _greedy_allocation!(port, port_type, latest_prices, investment, rounding, reinvest)
    key = Symbol("Greedy_" * string(port_type))

    weights = port.optimal[port_type].weights
    tickers = port.assets

    long_idx, short_idx, long_investment, short_investment, long_ratio, short_ratio = _setup_alloc_optim(weights,
                                                                                                         investment,
                                                                                                         reinvest)

    long_tickers, long_shares, long_latest_prices, long_cost, long_allocated_weights, long_leftover = _greedy_sub_allocation!(tickers[long_idx],
                                                                                                                              weights[long_idx],
                                                                                                                              latest_prices[long_idx],
                                                                                                                              long_investment,
                                                                                                                              rounding,
                                                                                                                              long_ratio)

    short_tickers, short_shares, short_latest_prices, short_cost, short_allocated_weights, short_leftover = _greedy_sub_allocation!(tickers[short_idx],
                                                                                                                                    -weights[short_idx],
                                                                                                                                    latest_prices[short_idx],
                                                                                                                                    short_investment,
                                                                                                                                    rounding,
                                                                                                                                    short_ratio)

    _combine_allocations!(port, key, long_tickers, short_tickers, long_shares, short_shares,
                          long_latest_prices, short_latest_prices, long_cost, short_cost,
                          long_allocated_weights, short_allocated_weights)

    idx = [findfirst(x -> x == t, port.alloc_optimal[key].tickers) for t ∈ tickers]
    port.alloc_optimal[key] = port.alloc_optimal[key][idx, :]
    port.alloc_leftover[key] = long_leftover + short_leftover

    return port.alloc_optimal[key]
end
function _allocate!(::LP, port, type, latest_prices, investment, reinvest, string_names)
    return _lp_allocation!(port, type, latest_prices, investment, reinvest, string_names)
end
function _allocate!(method::Greedy, port, type, latest_prices, investment, reinvest, ::Any)
    return _greedy_allocation!(port, type, latest_prices, investment, method.rounding,
                               reinvest)
end
