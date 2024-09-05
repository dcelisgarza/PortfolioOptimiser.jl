function _resolve_model(model, latest_prices)
    shares = round.(Int, value.(model[:x]))
    cost = latest_prices .* shares
    weights = cost / sum(cost)
    available_funds = value(model[:r])

    return shares, cost, weights, available_funds
end
function _optimise_allocation(port, label, tickers, latest_prices)
    model = port.alloc_model
    solvers = port.alloc_solvers
    term_status = termination_status(model)
    solvers_tried = Dict()

    success = false
    key = nothing
    for (key, val) ∈ solvers
        if haskey(val, :solver)
            set_optimizer(model, val[:solver])
        end

        if haskey(val, :params)
            for (attribute, value) ∈ val[:params]
                set_attribute(model, attribute, value)
            end
        end

        if haskey(val, :check_sol)
            check_sol = val[:check_sol]
        else
            check_sol = (;)
        end

        try
            JuMP.optimize!(model)
        catch jump_error
            push!(solvers_tried, key => Dict(:jump_error => jump_error))
            continue
        end

        if is_solved_and_feasible(model; check_sol...)
            success = true
            break
        else
            term_status = termination_status(model)
        end

        shares, cost, weights, available_funds = _resolve_model(model, latest_prices)

        push!(solvers_tried,
              key => Dict(:objective_val => objective_value(model),
                          :term_status => term_status,
                          :params => haskey(val, :params) ? val[:params] : missing,
                          :available_funds => available_funds,
                          :allocation => DataFrame(; tickers = tickers, shares = shares,
                                                   cost = cost, weights = weights)))
    end

    key = Symbol(string(key) * "_" * string(label))

    return if success
        shares, cost, weights, available_funds = _resolve_model(model, latest_prices)
    else
        funcname = "$(fullname(PortfolioOptimiser)[1]).$(nameof(PortfolioOptimiser._lp_sub_allocation!))"
        @warn("$funcname: model could not be optimised satisfactorily.\nSolvers: $solvers_tried.")
        port.alloc_fail[key] = solvers_tried

        (String[], Vector{eltype(latest_prices)}(undef, 0),
         Vector{eltype(latest_prices)}(undef, 0), zero(eltype(latest_prices)))
    end
end
function _lp_sub_allocation!(port, key, label, tickers, weights, latest_prices, investment,
                             string_names, ratio)
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

    shares, cost, allocated_weights, available_funds = _optimise_allocation(port, label,
                                                                            tickers,
                                                                            latest_prices)

    allocated_weights *= ratio
    return tickers, shares, latest_prices, cost, allocated_weights, available_funds
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
function _lp_allocation!(port, port_type, latest_prices, investment, reinvest, string_names)
    key = Symbol("LP_" * string(port_type))

    weights = port.optimal[port_type].weights
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

    _combine_allocations!(port, key, long_tickers, short_tickers, long_shares, short_shares,
                          long_latest_prices, short_latest_prices, long_cost, short_cost,
                          long_allocated_weights, short_allocated_weights)

    if !isempty(short_tickers) && !isempty(port.alloc_optimal[key])
        idx = [findfirst(x -> x == t, port.alloc_optimal[key].tickers) for t ∈ tickers]
        port.alloc_optimal[key] = port.alloc_optimal[key][idx, :]
    end
    port.alloc_leftover[key] = long_leftover + short_leftover

    return port.alloc_optimal[key]
end