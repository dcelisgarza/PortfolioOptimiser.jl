# Copywrite (c) 2025
# Author: Daniel Celis Garza <daniel.celis.garza@gmail.com>
# SPDX-License-Identifier: MIT

function resolve_model(model, latest_prices)
    shares = round.(Int, value.(model[:x]))
    cost = latest_prices .* shares
    weights = cost / sum(cost)
    available_funds = value(model[:r])

    return shares, cost, weights, available_funds
end
function optimise_allocation(port, label, tickers, latest_prices)
    model = port.alloc_model
    solvers = port.alloc_solvers
    solvers_tried = Dict()

    success = false
    name = nothing
    for solver ∈ solvers
        name = solver.name
        solver_i = solver.solver
        params = solver.params
        add_bridges = solver.add_bridges
        check_sol = solver.check_sol

        set_optimizer(model, solver_i; add_bridges = add_bridges)
        if !isnothing(params) && !isempty(params)
            for (k, v) ∈ params
                set_attribute(model, k, v)
            end
        end

        try
            JuMP.optimize!(model)
        catch jump_error
            push!(solvers_tried, name => Dict(:jump_error => jump_error))
            continue
        end

        try
            assert_is_solved_and_feasible(model; check_sol...)
            success = true
            break
        catch err
            shares, cost, weights, available_funds = resolve_model(model, latest_prices)
            push!(solvers_tried,
                  name => Dict(:objective_val => objective_value(model), :err => err,
                               :params => params, :available_funds => available_funds,
                               :allocation => DataFrame(; tickers = tickers,
                                                        shares = shares, cost = cost,
                                                        weights = weights)))
        end
    end

    name = Symbol(string(name) * "_" * string(label))

    return if success
        shares, cost, weights, available_funds = resolve_model(model, latest_prices)
    else
        funcname = "$(fullname(PortfolioOptimiser)[1]).$(nameof(PortfolioOptimiser.lp_sub_allocation!))"
        @warn("$funcname: model could not be optimised satisfactorily.\nSolvers: $solvers_tried.")
        port.alloc_fail[name] = solvers_tried

        (String[], Vector{eltype(latest_prices)}(undef, 0),
         Vector{eltype(latest_prices)}(undef, 0), zero(eltype(latest_prices)))
    end
end
function lp_sub_allocation!(port, label, tickers, weights, latest_prices, investment,
                            string_names, total_investment)
    if isempty(tickers)
        return String[], Vector{Int}(undef, 0), Vector{eltype(latest_prices)}(undef, 0),
               Vector{eltype(latest_prices)}(undef, 0),
               Vector{eltype(latest_prices)}(undef, 0), zero(eltype(latest_prices))
    end
    scale_constr = port.scale_constr
    scale_obj = port.scale_obj

    model = port.alloc_model = JuMP.Model()
    set_string_names_on_creation(model, string_names)

    weights ./= sum(weights)

    N = length(tickers)
    # Integer allocation
    # x := number of shares
    # u := bounding variable
    @variables(model, begin
                   x[1:N] .>= 0, Int
                   u
               end)

    # r := remaining money
    # eta := ideal_investment - discrete_investment
    @expressions(model, begin
                     r, investment - dot(latest_prices, x)
                     eta, weights * investment - x .* latest_prices
                 end)

    @constraints(model, begin
                     scale_constr * r >= 0
                     [scale_constr * u; scale_constr * eta] ∈ MOI.NormOneCone(N + 1)
                 end)

    @objective(model, Min, scale_obj * (u + r))

    shares, cost, allocated_weights, available_funds = optimise_allocation(port, label,
                                                                           tickers,
                                                                           latest_prices)
    if !isempty(shares)
        alpha = dot(latest_prices, shares)
        alpha /= total_investment
        allocated_weights .*= alpha
    end
    return tickers, shares, latest_prices, cost, allocated_weights, available_funds
end
function combine_allocations!(port, key, long_tickers, short_tickers, long_shares,
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
function lp_allocation!(port, port_key, investment, string_names)
    key = Symbol("LP_" * string(port_key))

    weights = port.optimal[port_key].weights
    tickers = port.assets

    latest_prices = port.latest_prices

    long_idx, short_idx, long_investment, short_investment = setup_alloc_optim(port,
                                                                               weights,
                                                                               investment)

    long_tickers, long_shares, long_latest_prices, long_cost, long_allocated_weights, long_leftover = lp_sub_allocation!(port,
                                                                                                                         :long,
                                                                                                                         tickers[long_idx],
                                                                                                                         weights[long_idx],
                                                                                                                         latest_prices[long_idx],
                                                                                                                         long_investment,
                                                                                                                         string_names,
                                                                                                                         investment)

    short_tickers, short_shares, short_latest_prices, short_cost, short_allocated_weights, short_leftover = lp_sub_allocation!(port,
                                                                                                                               :short,
                                                                                                                               tickers[short_idx],
                                                                                                                               -weights[short_idx],
                                                                                                                               latest_prices[short_idx],
                                                                                                                               short_investment,
                                                                                                                               string_names,
                                                                                                                               investment)

    combine_allocations!(port, key, long_tickers, short_tickers, long_shares, short_shares,
                         long_latest_prices, short_latest_prices, long_cost, short_cost,
                         long_allocated_weights, short_allocated_weights)

    if !isempty(short_tickers) && !isempty(port.alloc_optimal[key])
        idx = [findfirst(x -> x == t, port.alloc_optimal[key].tickers) for t ∈ tickers]
        port.alloc_optimal[key] = port.alloc_optimal[key][idx, :]
    end
    port.alloc_leftover[key] = long_leftover + short_leftover

    return port.alloc_optimal[key]
end
function allocate!(port::AbstractPortfolio, ::LP; key::Symbol = :Trad,
                   investment::Real = 1e6, string_names::Bool = false)
    return lp_allocation!(port, key, investment, string_names)
end
