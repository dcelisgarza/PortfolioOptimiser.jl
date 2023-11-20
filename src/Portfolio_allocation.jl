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

function _setup_alloc_optim(weights, short_ratio, investment, reinvest)
    long_idx = weights .>= 0
    short_idx = .!long_idx

    isnothing(short_ratio) && (short_ratio = -sum(weights[short_idx]))

    @assert(
        short_ratio >= 0,
        "short ratio = $short_ratio, must be bigger than or equal to zero"
    )

    short_investment = investment * short_ratio
    long_investment = investment

    reinvest && (long_investment += short_investment)

    return long_idx, short_idx, long_investment, short_investment
end

function _optimise_allocation(portfolio, model, tickers, latest_prices)
    solvers = portfolio.alloc_solvers
    term_status = termination_status(model)
    solvers_tried = Dict()

    for (key, val) in solvers
        key = Symbol(String(key))

        haskey(val, :solver) && set_optimizer(model, val[:solver])

        if haskey(val, :params)
            for (attribute, value) in val[:params]
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

        term_status in ValidTermination && break

        shares = round.(Int, value.(model[:x]))
        cost = latest_prices .* shares
        weights = cost / sum(cost)
        available_funds = value(model[:r])

        push!(
            solvers_tried,
            key => Dict(
                :objective_val => objective_value(model),
                :term_status => term_status,
                :params => haskey(val, :params) ? val[:params] : missing,
                :available_funds => available_funds,
                :allocation => DataFrame(
                    tickers = tickers,
                    shares = shares,
                    cost = cost,
                    weights = weights,
                ),
            ),
        )
    end

    return term_status, solvers_tried
end

function _combine_allocations(
    portfolio,
    key,
    long_tickers,
    short_tickers,
    long_shares,
    short_shares,
    long_cost,
    short_cost,
    long_allocated_weights,
    short_allocated_weights,
)
    tickers = [long_tickers; short_tickers]
    shares = [long_shares; -short_shares]
    cost = [long_cost; -short_cost]
    weights = [long_allocated_weights; -short_allocated_weights]

    portfolio.alloc_optimal[key] =
        !isempty(tickers) && !isempty(shares) && !isempty(cost) && !isempty(weights) ?
        DataFrame(tickers = tickers, shares = shares, cost = cost, weights = weights) :
        DataFrame()

    return portfolio.alloc_optimal[key]
end

function _handle_alloc_errors_and_finalise(
    portfolio,
    model,
    term_status,
    solvers_tried,
    key,
    label,
    latest_prices,
)
    key = Symbol(string(key) * "_" * string(label))

    retval = if term_status ∉ ValidTermination
        funcname = "$(fullname(PortfolioOptimiser)[1]).$(nameof(PortfolioOptimiser._lp_sub_allocation!))"
        @warn(
            "$funcname: model could not be optimised satisfactorily.\nSolvers: $solvers_tried."
        )
        portfolio.alloc_fail[key] = solvers_tried

        (
            return String[],
            Vector{eltype(latest_prices)}(undef, 0),
            Vector{eltype(latest_prices)}(undef, 0),
            Vector{eltype(latest_prices)}(undef, 0),
            0
        )

    else
        shares = round.(Int, value.(model[:x]))
        cost = latest_prices .* shares
        weights = cost / sum(cost)
        available_funds = value(model[:r])

        portfolio.alloc_model[key] = model

        (shares, cost, weights, available_funds)
    end

    return retval
end

function _lp_sub_allocation!(
    portfolio,
    key,
    label,
    tickers,
    weights,
    latest_prices,
    investment,
    string_names,
)
    isempty(tickers) && (
        return String[],
        Vector{eltype(latest_prices)}(undef, 0),
        Vector{eltype(latest_prices)}(undef, 0),
        Vector{eltype(latest_prices)}(undef, 0),
        0
    )

    model = JuMP.Model()
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

    @constraint(model, [u; eta] in MOI.NormOneCone(N + 1))
    @constraint(model, r >= 0)

    @objective(model, Min, u + r)

    term_status, solvers_tried =
        _optimise_allocation(portfolio, model, tickers, latest_prices)

    shares, cost, allocated_weights, available_funds = _handle_alloc_errors_and_finalise(
        portfolio,
        model,
        term_status,
        solvers_tried,
        key,
        label,
        latest_prices,
    )

    return tickers, shares, cost, allocated_weights, available_funds
end

function _lp_allocation!(
    portfolio,
    port_type,
    latest_prices,
    investment,
    reinvest,
    short_ratio,
    string_names,
)
    key = Symbol("LP_" * string(port_type))

    weights = portfolio.optimal[port_type].weights
    tickers = portfolio.assets

    long_idx, short_idx, long_investment, short_investment =
        _setup_alloc_optim(weights, short_ratio, investment, reinvest)

    long_tickers, long_shares, long_cost, long_allocated_weights, long_leftover =
        _lp_sub_allocation!(
            portfolio,
            key,
            :long,
            tickers[long_idx],
            weights[long_idx],
            latest_prices[long_idx],
            long_investment,
            string_names,
        )

    short_tickers, short_shares, short_cost, short_allocated_weights, short_leftover =
        _lp_sub_allocation!(
            portfolio,
            key,
            :short,
            tickers[short_idx],
            -weights[short_idx],
            latest_prices[short_idx],
            short_investment,
            string_names,
        )

    retval = _combine_allocations(
        portfolio,
        key,
        long_tickers,
        short_tickers,
        long_shares,
        short_shares,
        long_cost,
        short_cost,
        long_allocated_weights,
        short_allocated_weights,
    )

    !isa(long_leftover, Number) && (long_leftover = long_investment)
    !isa(short_leftover, Number) && (short_leftover = short_investment)

    return retval, long_leftover + short_leftover
end

function _greedy_sub_allocation!(tickers, weights, latest_prices, investment, rounding)
    isempty(tickers) && (
        return String[],
        Vector{eltype(latest_prices)}(undef, 0),
        Vector{eltype(latest_prices)}(undef, 0),
        Vector{eltype(latest_prices)}(undef, 0),
        zero(eltype(latest_prices))
    )

    idx = sortperm(weights, rev = true)
    weights = weights[idx]
    tickers = tickers[idx]
    latest_prices = latest_prices[idx]

    N = length(tickers)
    available_funds = investment
    shares = zeros(eltype(rounding), N)
    weights ./= sum(weights)

    # First loop
    for i in 1:N
        price = latest_prices[i]
        n_shares = roundmult(weights[i] * investment / price, rounding, RoundDown)
        cost = n_shares * price
        cost > available_funds && break
        available_funds -= cost
        shares[i] = n_shares
    end

    # Second loop
    while available_funds > 0
        # Calculate equivalent continuous weights of what has already been bought.
        current_weights = latest_prices .* shares
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
        shares[idx] += rounding
        available_funds -= price
    end

    cost = latest_prices .* shares
    allocated_weights = cost / sum(cost)

    return tickers, shares, cost, allocated_weights, available_funds
end

function _greedy_allocation!(
    portfolio,
    port_type,
    latest_prices,
    investment,
    rounding,
    reinvest,
    short_ratio,
)
    key = Symbol("Greedy_" * string(port_type))

    weights = portfolio.optimal[port_type].weights
    tickers = portfolio.assets

    long_idx, short_idx, long_investment, short_investment =
        _setup_alloc_optim(weights, short_ratio, investment, reinvest)

    long_tickers, long_shares, long_cost, long_allocated_weights, long_leftover =
        _greedy_sub_allocation!(
            tickers[long_idx],
            weights[long_idx],
            latest_prices[long_idx],
            long_investment,
            rounding,
        )

    short_tickers, short_shares, short_cost, short_allocated_weights, short_leftover =
        _greedy_sub_allocation!(
            tickers[short_idx],
            -weights[short_idx],
            latest_prices[short_idx],
            short_investment,
            rounding,
        )

    retval = _combine_allocations(
        portfolio,
        key,
        long_tickers,
        short_tickers,
        long_shares,
        short_shares,
        long_cost,
        short_cost,
        long_allocated_weights,
        short_allocated_weights,
    )

    return retval, long_leftover + short_leftover
end

function _save_alloc_opt_params(
    portfolio,
    port_type,
    alloc_type,
    latest_prices,
    investment,
    rounding,
    reinvest,
    short_ratio,
    leftover,
    save_opt_params,
)
    !save_opt_params && return nothing

    key = Symbol(string(alloc_type) * "_" * string(port_type))

    portfolio.alloc_params[key] = Dict(
        :latest_prices => latest_prices,
        :investment => investment,
        :rounding => rounding,
        :reinvest => reinvest,
        :short_ratio => short_ratio,
        :leftover => leftover,
    )

    return nothing
end

function allocate_port!(
    portfolio;
    port_type = isa(portfolio, Portfolio) ? :Trad : :HRP,
    alloc_type = :LP,
    latest_prices = portfolio.latest_prices,
    investment = 1e4,
    rounding = 1,
    reinvest = false,
    short_ratio = nothing,
    string_names = false,
    save_opt_params = true,
)
    isa(portfolio, Portfolio) ?
    @assert(port_type ∈ PortTypes, "port_type = $port_type, must be one of $PortTypes") :
    @assert(port_type ∈ HCPortTypes, "port_type = $port_type, must be one of $HCPortTypes")

    @assert(alloc_type ∈ AllocTypes, "alloc_type = $alloc_type, must be one of $AllocTypes")

    retval, leftover =
        alloc_type == :LP ?
        _lp_allocation!(
            portfolio,
            port_type,
            latest_prices,
            investment,
            reinvest,
            short_ratio,
            string_names,
        ) :
        _greedy_allocation!(
            portfolio,
            port_type,
            latest_prices,
            investment,
            rounding,
            reinvest,
            short_ratio,
        )

    _save_alloc_opt_params(
        portfolio,
        port_type,
        alloc_type,
        latest_prices,
        investment,
        rounding,
        reinvest,
        short_ratio,
        leftover,
        save_opt_params,
    )

    return retval
end

export allocate_port!
