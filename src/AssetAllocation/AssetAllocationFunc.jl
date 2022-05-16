"""
```
Allocation(
    type::AbstractAllocation,
    portfolio::AbstractPortfolioOptimiser,
    latest_prices::AbstractVector;
    investment = 1e4,
    rounding = 1,
    reinvest = false,
    short_ratio = nothing,
    optimiser = HiGHS.Optimizer,
    silent = true,
)
```
Returns a tuple of an [`Allocation`](@ref) structure and remaining money after allocating assets.

!!! note
    Short positions have negative weights and shares.

- `type`: Allocation algorithm to use, can be `Lazy()`, `Greedy()` or `LP()` (see [`AbstractAllocation`](@ref)).
- `portfolio`: Portfolio [`AbstractPortfolioOptimiser`](@ref).
- `latest_prices`: Vector of latest prices, entries should be in the same order as `portfolio.tickers`.
- `investment`: value to be invested in the portfolio.
- `rounding`: `Greedy()` and `Lazy()` support fractional shares, rounds shares down to the nearest multiple of `rounding`. This has no effect on an `LP()` allocation.
- `reinvest`: if `true`, reinvests the money earned from short positions of the portfolio into buying more long positions.
- `short_ratio`: long to short portfolio ratio, 0.2 corresponds to 120/20 long to short shares. If `nothing` defaults to portfolio weights.
- `optimiser`: only used for the `LP()` allocation. Needs to support mixed-interger linear programming.
- `silent`: if `true` the optimiser is silent.

## Example
```
ef = EfficientFrontier(
    tickers,
    mean_ret,
    cov_mtx;
    market_neutral = true,
    weight_bounds = (-1, 1),
)
max_quadratic_utility!(ef)

# Mixed integer, programming, no rounding available.
allocLP, leftoverLP = Allocation(LP(), ef, latest_prices; investment = 3590)

# Greedy allocation up to quarter shares.
allocGreedy, leftoverGreedy = Allocation(Greedy(), ef, latest_prices; rounding = 0.25)

# Lazy allocation with a 110/10 long-short ratio and reinvesting 
# earnings from selling the short positions into buying long positions.
allocLazy, leftoverLazy = Allocation(
                            Lazy(), ef, latest_prices; 
                            short_ratio = 0.1, reinvest = true
                        )
```
"""
function Allocation(
    type::AbstractAllocation,
    portfolio::AbstractPortfolioOptimiser,
    latest_prices::AbstractVector;
    investment = 1e4,
    rounding = 1,
    reinvest = false,
    short_ratio = nothing,
    optimiser = HiGHS.Optimizer,
    silent = true,
)
    @assert isnothing(short_ratio) || short_ratio > 0

    return _Allocation(
        type,
        portfolio.tickers,
        portfolio.weights,
        latest_prices,
        investment,
        rounding,
        reinvest,
        short_ratio,
        optimiser,
        silent,
    )
end

"""
```
_Allocation(
    type::Lazy,
    tickers::AbstractArray,
    weights::AbstractArray,
    latest_prices::AbstractVector,
    investment::Real = 1e4,
    rounding::Real = 1,
    reinvest::Bool = false,
    short_ratio::Union{Real, Nothing} = nothing,
    optimiser = nothing,
    silent = nothing,
)
```
Lazy asset allocation. Simply finds how many shares up to the smallest multiple of `rounding` you can buy for each ticker.

!!! note
    `optimiser` and `silent` are not used but are in the function signature for dispatch purposes.
"""
function _Allocation(
    type::Lazy,
    tickers::AbstractArray,
    weights::AbstractArray,
    latest_prices::AbstractVector,
    investment::Real = 1e4,
    rounding::Real = 1,
    reinvest::Bool = false,
    short_ratio::Union{Real, Nothing} = nothing,
    optimiser = nothing,
    silent = nothing,
)
    idx = sortperm(weights, rev = true)

    weights = weights[idx]
    tickers = tickers[idx]
    latest_prices = latest_prices[idx]

    if weights[end] < 0
        return _short_allocation(
            type,
            tickers,
            weights,
            latest_prices,
            investment,
            rounding,
            reinvest,
            short_ratio,
            optimiser,
            silent,
        )
    end

    # If there is no shorting, continue with lazy allocation.
    n_tickers = length(tickers)
    shares_bought = zeros(n_tickers)
    share_costs = zeros(n_tickers)

    norm_price = latest_prices[end]
    norm_weights = weights / weights[end]

    for i in 1:n_tickers
        share_costs[i] = norm_price * norm_weights[i]
        shares_bought[i] = share_costs[i] / latest_prices[i]
    end

    ratio = investment / sum(share_costs)
    shares_bought = roundmult.(shares_bought * ratio, rounding, RoundDown)

    # Remove zero weights.
    tickers, allocated_weights, shares_bought, sum_weights =
        _clean_zero_shares(shares_bought, tickers, latest_prices)

    return Allocation(tickers, allocated_weights, shares_bought), investment - sum_weights
end

"""
```
_Allocation(
    type::Greedy,
    tickers::AbstractArray,
    weights::AbstractArray,
    latest_prices::AbstractVector,
    investment::Real = 1e4,
    rounding::Real = 1,
    reinvest::Bool = false,
    short_ratio::Union{Real, Nothing} = nothing,
    optimiser = nothing,
    silent = nothing,
)
```
Greedy algorithm that allocates assets according to their optimised weights. It priorites assets according to the ones with the greatest differnce between their weight in the portfolio and ideal weight. It allocates assets by progressively adding `rounding` to each share until the cost exceeds the available funds.

!!! note
    `optimiser` and `silent` are not used but are in the function signature for dispatch purposes.
"""
function _Allocation(
    type::Greedy,
    tickers::AbstractArray,
    weights::AbstractArray,
    latest_prices::AbstractVector,
    investment::Real = 1e4,
    rounding::Real = 1,
    reinvest::Bool = false,
    short_ratio::Union{Real, Nothing} = nothing,
    optimiser = nothing,
    silent = nothing,
)
    idx = sortperm(weights, rev = true)

    weights = weights[idx]
    tickers = tickers[idx]
    latest_prices = latest_prices[idx]

    if weights[end] < 0
        return _short_allocation(
            type,
            tickers,
            weights,
            latest_prices,
            investment,
            rounding,
            reinvest,
            short_ratio,
            optimiser,
            silent,
        )
    end

    # If there is no shorting, continue with greedy allocation.
    available_funds = investment
    n_tickers = length(tickers)
    shares_bought = zeros(n_tickers)

    # First loop
    for i in 1:n_tickers
        price = latest_prices[i]
        n_shares = roundmult(weights[i] * investment / price, rounding, RoundDown)
        cost = n_shares * price
        cost > available_funds && break
        available_funds -= cost
        shares_bought[i] = n_shares
    end

    # Second loop
    while available_funds > 0
        # Calculate equivalent continuous weights of what has already been bought.
        current_weights = latest_prices .* shares_bought
        current_weights /= sum(current_weights)

        deficit = weights - current_weights

        # Try to buy ticker whose deficit is the greatest.
        idx = argmax(deficit)
        price = latest_prices[idx]

        # If we can't afford it, go through the rest of the tickers from highest deviation to lowest
        while price > available_funds
            deficit[idx] = 0
            idx = argmax(deficit)
            deficit[idx] <= 0 && break
            price = latest_prices[idx]
        end
        deficit[idx] <= 0 && break
        # Buy one share*rounding at a time.
        shares_bought[idx] += rounding
        available_funds -= price
    end

    # Remove zero weights.
    tickers, allocated_weights, shares_bought, missing =
        _clean_zero_shares(shares_bought, tickers, latest_prices)

    return Allocation(tickers, allocated_weights, shares_bought), available_funds
end

"""
```
_Allocation(
    type::LP,
    tickers::AbstractArray,
    weights::AbstractArray,
    latest_prices::AbstractVector,
    investment::Real = 1e4,
    rounding = nothing,
    reinvest::Bool = false,
    short_ratio::Union{Real, Nothing} = nothing,
    optimiser = HiGHS.Optimizer,
    silent = true,
)
```
Linear mixed-integer optimisation of assets.

!!! note
    `rounding` has no effect, as it can only allocate integer numbers of shares.
"""
function _Allocation(
    type::LP,
    tickers::AbstractArray,
    weights::AbstractArray,
    latest_prices::AbstractVector,
    investment::Real = 1e4,
    rounding = nothing,
    reinvest::Bool = false,
    short_ratio::Union{Real, Nothing} = nothing,
    optimiser = HiGHS.Optimizer,
    silent = true,
)
    if any(x -> x < 0, weights)
        return _short_allocation(
            type,
            tickers,
            weights,
            latest_prices,
            investment,
            rounding,
            reinvest,
            short_ratio,
            optimiser,
            silent,
        )
    end

    num_tickers = length(tickers)

    model = Model()

    # Integer allocation
    # x := number of shares
    @variable(model, x[1:num_tickers], Int)
    # u := bounding variable
    @variable(model, u[1:num_tickers])

    x = model[:x]
    u = model[:u]
    # Remaining money
    r = investment - dot(latest_prices, x)
    # weights * investment - allocation * latest_prices
    eta = weights * investment - x .* latest_prices

    @constraint(model, eta_leq_u, eta .<= u)
    @constraint(model, eta_geq_mu, eta .>= -u)
    @constraint(model, x_geq_0, x .>= 0)
    @constraint(model, r_geq_0, r >= 0)

    @objective(model, Min, sum(u) + r)

    MOI.set(model, MOI.Silent(), silent)
    set_optimizer(model, optimiser)
    optimize!(model)

    shares_bought = Int.(round.(value.(x)))
    available_funds = value(r)

    # Remove zero weights.
    tickers, allocated_weights, shares_bought, missing =
        _clean_zero_shares(shares_bought, tickers, latest_prices)

    return Allocation(tickers, allocated_weights, shares_bought), available_funds
end

"""
```
function _short_allocation(
    type,
    tickers,
    weights,
    latest_prices,
    investment,
    rounding,
    reinvest,
    short_ratio,
    optimiser,
    silent
)
```
Helper function for allocating short-long portfolios. Ensures the shares and weights of the shorts are negative.
"""
function _short_allocation(
    type,
    tickers,
    weights,
    latest_prices,
    investment,
    rounding,
    reinvest,
    short_ratio,
    optimiser,
    silent,
)
    lidx = weights .>= 0
    sidx = .!lidx

    if isnothing(short_ratio)
        short_ratio = -sum(weights[sidx])
    end
    @assert short_ratio > 0 "short ratio: $short_ratio, must be larger than 0"

    # Long only allocation.
    short_val = investment * short_ratio
    long_val = investment

    reinvest && (long_val += short_val)

    # Allocate only long positions.
    longAlloc, long_leftover = _sub_allocation(
        type,
        tickers,
        weights,
        latest_prices,
        long_val,
        rounding,
        lidx,
        false,
        nothing,
        optimiser,
        silent,
    )

    # Only take the indices of short stocks and multiply the weights by -1 to make the short weights positive, ensuring we don't infinitely recurse.
    shortAlloc, short_leftover = _sub_allocation(
        type,
        tickers,
        -weights,
        latest_prices,
        short_val,
        rounding,
        sidx,
        false,
        nothing,
        optimiser,
        silent,
    )

    # Combine long and short positions. Short shares and weights are negative.
    tickers = [longAlloc.tickers; shortAlloc.tickers]
    weights = [longAlloc.weights; -shortAlloc.weights]
    shares = [longAlloc.shares; -shortAlloc.shares]

    return Allocation(tickers, weights, shares), long_leftover + short_leftover
end

"""
```
_sub_allocation(type, tickers, weights, latest_prices, investment, rounding, idx)
```
Helper function for creating sub allocations. It calls `Allocation`` again but only for the `idx` provided.
"""
function _sub_allocation(
    type,
    tickers,
    weights,
    latest_prices,
    investment,
    rounding,
    idx,
    reinvest,
    short_ratio,
    optimiser,
    silent,
)
    tickers = tickers[idx]
    weights = weights[idx]
    weights /= sum(weights)
    latest_prices = latest_prices[idx]
    subAlloc, sub_leftover = _Allocation(
        type,
        tickers,
        weights,
        latest_prices,
        investment,
        rounding,
        reinvest,
        short_ratio,
        optimiser,
        silent,
    )

    return subAlloc, sub_leftover
end

"""
```
_clean_zero_shares(shares_bought, tickers, latest_prices)
```
Helper function for removing tickers with zero shares.
"""
function _clean_zero_shares(shares_bought, tickers, latest_prices)
    idx = shares_bought .!= 0
    tickers = tickers[idx]
    shares_bought = shares_bought[idx]
    allocated_weights = latest_prices[idx] .* shares_bought
    sum_weights = sum(allocated_weights)
    allocated_weights /= sum_weights

    return tickers, allocated_weights, shares_bought, sum_weights
end