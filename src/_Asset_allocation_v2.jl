#=
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

function _setup_alloc_optim(weights, investment, reinvest)
    long_idx = weights .>= 0
    short_idx = .!long_idx

    long_ratio = if !isempty(long_idx)
        sum(weights[long_idx])
    else
        zero(eltype(weights))
    end

    short_ratio = if !isempty(short_idx)
        -sum(weights[short_idx])
    else
        zero(eltype(weights))
    end

    short_investment = investment * short_ratio
    long_investment = investment * long_ratio

    if reinvest
        long_investment += short_investment
    end

    return long_idx, short_idx, long_investment, short_investment, long_ratio, short_ratio
end

function _optimise_allocation(port, tickers, latest_prices)
    model = port.alloc_model
    solvers = port.alloc_solvers
    term_status = termination_status(model)
    solvers_tried = Dict()

    for (key, val) ∈ solvers
        key = Symbol(String(key))

        if haskey(val, :solver)
            set_optimizer(model, val[:solver])
        end

        if haskey(val, :params)
            for (attribute, value) ∈ val[:params]
                set_attribute(model, attribute, value)
            end
        end

        try
            JuMP.optimize!(model)
        catch jump_error
            push!(solvers_tried, key => Dict(:jump_error => jump_error))
            continue
        end

        term_status = termination_status(model)

        if term_status ∈ ValidTermination
            break
        end

        shares = round.(Int, value.(model[:x]))
        cost = latest_prices .* shares
        weights = cost / sum(cost)
        available_funds = value(model[:r])

        push!(solvers_tried,
              key => Dict(:objective_val => objective_value(model),
                          :term_status => term_status,
                          :params => haskey(val, :params) ? val[:params] : missing,
                          :available_funds => available_funds,
                          :allocation => DataFrame(; tickers = tickers, shares = shares,
                                                   cost = cost, weights = weights)))
    end

    return term_status, solvers_tried
end

function _combine_allocations!(port, key, long_tickers, short_tickers, long_shares,
                               short_shares, long_latest_prices, short_latest_prices,
                               long_cost, short_cost, long_allocated_weights,
                               short_allocated_weights)
    tickers = [long_tickers; short_tickers]
    shares = [long_shares; -short_shares]
    latest_prices = [long_latest_prices; -short_latest_prices]
    cost = [long_cost; -short_cost]
    weights = [long_allocated_weights; -short_allocated_weights]

    port.alloc_optimal[key] = if !isempty(tickers) &&
                                      !isempty(shares) &&
                                      !isempty(cost) &&
                                      !isempty(weights)
        DataFrame(; tickers = tickers, shares = shares, price = latest_prices, cost = cost,
                  weights = weights)
    else
        DataFrame()
    end

    return nothing
end

function _handle_alloc_errors_and_finalise(port, term_status, solvers_tried, key,
                                           label, latest_prices)
    model = port.alloc_model
    key = Symbol(string(key) * "_" * string(label))

    retval = if term_status ∉ ValidTermination
        funcname = "$(fullname(PortfolioOptimiser)[1]).$(nameof(PortfolioOptimiser._lp_sub_allocation!))"
        @warn("$funcname: model could not be optimised satisfactorily.\nSolvers: $solvers_tried.")
        port.alloc_fail[key] = solvers_tried

        (String[], Vector{eltype(latest_prices)}(undef, 0),
         Vector{eltype(latest_prices)}(undef, 0), zero(eltype(latest_prices)))

    else
        shares = round.(Int, value.(model[:x]))
        cost = latest_prices .* shares
        weights = cost / sum(cost)
        available_funds = value(model[:r])

        (shares, cost, weights, available_funds)
    end

    return retval
end

function _lp_sub_allocation!(port, key, label, tickers, weights, latest_prices,
                             investment, string_names, ratio)
    if isempty(tickers)
        return String[], Vector{eltype(latest_prices)}(undef, 0),
               Vector{eltype(latest_prices)}(undef, 0),
               Vector{eltype(latest_prices)}(undef, 0),
               Vector{eltype(latest_prices)}(undef, 0), zero(eltype(latest_prices))
    end

    port.alloc_model = JuMP.Model()
    model = port.alloc_model

    set_string_names_on_creation(model, string_names)

    weights /= sum(weights)

    N = length(tickers)
    # Integer allocation
    # x := number of shares
    @variable(model, x[1:N] .>= 0, Int)
    # u := bounding variable
    @variable(model, u)

    # Remaining money
    @expression(model, r, investment - dot(latest_prices, x))
    # weights * investment - allocation * latest_prices
    eta = weights * investment - x .* latest_prices

    @constraint(model, [u; eta] ∈ MOI.NormOneCone(N + 1))
    @constraint(model, r >= 0)

    @objective(model, Min, u + r)

    term_status, solvers_tried = _optimise_allocation(port, tickers, latest_prices)

    shares, cost, allocated_weights, available_funds = _handle_alloc_errors_and_finalise(port,
                                                                                         term_status,
                                                                                         solvers_tried,
                                                                                         key,
                                                                                         label,
                                                                                         latest_prices)

    allocated_weights *= ratio
    return tickers, shares, latest_prices, cost, allocated_weights, available_funds
end

function _lp_allocation!(port, type, latest_prices, investment, reinvest,
                         string_names)
    key = Symbol("LP_" * string(type))

    weights = port.optimal[type].weights
    tickers = port.assets

    long_idx, short_idx, long_investment, short_investment, long_ratio, short_ratio = _setup_alloc_optim(weights,
                                                                                                         investment,
                                                                                                         reinvest)

    long_tickers, long_shares, long_latest_prices, long_cost, long_allocated_weights, long_leftover = _lp_sub_allocation!(port,
                                                                                                                          key,
                                                                                                                          :long,
                                                                                                                          tickers[long_idx],
                                                                                                                          weights[long_idx],
                                                                                                                          latest_prices[long_idx],
                                                                                                                          long_investment,
                                                                                                                          string_names,
                                                                                                                          long_ratio)

    short_tickers, short_shares, short_latest_prices, short_cost, short_allocated_weights, short_leftover = _lp_sub_allocation!(port,
                                                                                                                                key,
                                                                                                                                :short,
                                                                                                                                tickers[short_idx],
                                                                                                                                -weights[short_idx],
                                                                                                                                latest_prices[short_idx],
                                                                                                                                short_investment,
                                                                                                                                string_names,
                                                                                                                                short_ratio)

    _combine_allocations!(port, key, long_tickers, short_tickers, long_shares,
                          short_shares, long_latest_prices, short_latest_prices, long_cost,
                          short_cost, long_allocated_weights, short_allocated_weights)

    if !isempty(short_tickers)
        idx = [findfirst(x -> x == t, port.alloc_optimal[key].tickers) for t ∈ tickers]
        port.alloc_optimal[key] = port.alloc_optimal[key][idx, :]
    end

    return port.alloc_optimal[key], long_leftover + short_leftover
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

function _greedy_allocation!(port, type, latest_prices, investment, rounding,
                             reinvest)
    key = Symbol("Greedy_" * string(type))

    weights = port.optimal[type].weights
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

    _combine_allocations!(port, key, long_tickers, short_tickers, long_shares,
                          short_shares, long_latest_prices, short_latest_prices, long_cost,
                          short_cost, long_allocated_weights, short_allocated_weights)

    idx = [findfirst(x -> x == t, port.alloc_optimal[key].tickers) for t ∈ tickers]
    port.alloc_optimal[key] = port.alloc_optimal[key][idx, :]

    return port.alloc_optimal[key], long_leftover + short_leftover
end

function _save_alloc_opt_params(port, type, alloc_type, investment, rounding,
                                reinvest, leftover, save_opt_params)
    if !save_opt_params
        return nothing
    end

    key = Symbol(string(alloc_type) * "_" * string(type))

    port.alloc_params[key] = Dict(:investment => investment, :rounding => rounding,
                                       :reinvest => reinvest, :leftover => leftover)

    return nothing
end
=#
"""
```julia
allocate!(port;
          opt::AllocOpt = AllocOpt(; type = if isa(port, Portfolio)
                                       :Trad
                                   else
                                       :HRP
                                   end, latest_prices = port.latest_prices),
          string_names = false, save_opt_params = true)
```
"""
function _allocate!(::LP2, port, type, latest_prices, investment, reinvest, string_names)
    return _lp_allocation!(port, type, latest_prices, investment, reinvest, string_names)
end
function _allocate!(method::Greedy2, port, type, latest_prices, investment, reinvest, ::Any)
    return _greedy_allocation!(port, type, latest_prices, investment, method.rounding,
                               reinvest)
end
function allocate2!(port::AbstractPortfolio2;
                    type::Symbol = isa(port, Portfolio2) ? :Trad2 : :HRP2,
                    method::AllocationMethod = LP2(), latest_prices = port.latest_prices,
                    investment::Real = 1e6, reinvest::Bool = false,
                    string_names::Bool = false)
    return _allocate!(method, port, type, latest_prices, investment, reinvest, string_names)
end
export allocate2!
