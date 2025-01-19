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
function greedy_sub_allocation!(tickers, weights, latest_prices, investment, rounding,
                                total_investment)
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
    for i ∈ eachindex(weights)
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
    alpha = sum(cost)
    allocated_weights = cost / alpha
    alpha /= total_investment
    allocated_weights .*= alpha

    return tickers, shares, latest_prices, cost, allocated_weights, available_funds
end
function greedy_allocation!(port, port_key, latest_prices, investment, short, budget,
                            short_budget, rounding)
    key = Symbol("Greedy_" * string(port_key))

    weights = port.optimal[port_key].weights
    tickers = port.assets

    long_idx, short_idx, long_investment, short_investment = _setup_alloc_optim(weights,
                                                                                investment,
                                                                                short,
                                                                                budget,
                                                                                short_budget)

    long_tickers, long_shares, long_latest_prices, long_cost, long_allocated_weights, long_leftover = greedy_sub_allocation!(tickers[long_idx],
                                                                                                                             weights[long_idx],
                                                                                                                             latest_prices[long_idx],
                                                                                                                             long_investment,
                                                                                                                             rounding,
                                                                                                                             investment)

    short_tickers, short_shares, short_latest_prices, short_cost, short_allocated_weights, short_leftover = greedy_sub_allocation!(tickers[short_idx],
                                                                                                                                   -weights[short_idx],
                                                                                                                                   latest_prices[short_idx],
                                                                                                                                   short_investment,
                                                                                                                                   rounding,
                                                                                                                                   investment)

    combine_allocations!(port, key, long_tickers, short_tickers, long_shares, short_shares,
                         long_latest_prices, short_latest_prices, long_cost, short_cost,
                         long_allocated_weights, short_allocated_weights)

    idx = [findfirst(x -> x == t, port.alloc_optimal[key].tickers) for t ∈ tickers]
    port.alloc_optimal[key] = port.alloc_optimal[key][idx, :]
    port.alloc_leftover[key] = long_leftover + short_leftover

    return port.alloc_optimal[key]
end
function allocate!(port::AbstractPortfolio, type::Greedy; key::Symbol = :Trad,
                   latest_prices = port.latest_prices, investment::Real = 1e6,
                   short = port.short, budget = port.budget,
                   short_budget = port.short_budget, keargs...)
    return greedy_allocation!(port, key, latest_prices, investment, short, budget,
                              short_budget, type.rounding)
end
