# Copywrite (c) 2025
# Author: Daniel Celis Garza <daniel.celis.garza@gmail.com>
# SPDX-License-Identifier: MIT

function calc_fees(w::AbstractVector, fees::Union{AbstractVector{<:Real}, Real},
                   latest_prices::AbstractVector, op::Function)
    return if isa(fees, Real) && !iszero(fees)
        idx = op(w, zero(eltype(w)))
        sum(fees * w[idx] .* latest_prices[idx])
    elseif isa(fees, AbstractVector) && !(isempty(fees) || all(iszero.(fees)))
        idx = op(w, zero(eltype(w)))
        dot(fees[idx], w[idx] .* latest_prices[idx])
    else
        zero(eltype(w))
    end
end
function calc_fees(w::AbstractVector, rebalance::AbstractTR, latest_prices::AbstractVector)
    return if isa(rebalance, TR)
        rebal_fees = rebalance.val
        benchmark = rebalance.w
        if isa(rebal_fees, Real)
            sum(rebal_fees * abs.(benchmark .- w) .* latest_prices)
        else
            dot(rebal_fees, abs.(benchmark .- w) .* latest_prices)
        end
    else
        zero(eltype(w))
    end
end
function calc_fees(w::AbstractVector, latest_prices::AbstractVector,
                   long_fees::Union{AbstractVector{<:Real}, Real} = 0,
                   short_fees::Union{AbstractVector{<:Real}, Real} = 0,
                   rebalance::AbstractTR = NoTR())
    long_fees = calc_fees(w, long_fees, latest_prices, .>=)
    short_fees = calc_fees(w, short_fees, latest_prices, .<)
    rebal_fees = calc_fees(w, rebalance, latest_prices)
    return long_fees + short_fees + rebal_fees
end
function setup_alloc_optim(port, weights, investment)
    short = port.short
    budget = port.budget
    short_budget = port.short_budget
    latest_prices = port.latest_prices
    long_fees = port.long_fees
    short_fees = port.short_fees
    rebalance = port.rebalance

    fees = calc_fees(weights, latest_prices, long_fees, short_fees, rebalance)
    investment -= fees

    long_idx = weights .>= zero(eltype(weights))

    if short
        long_budget = min(budget - short_budget, sum(weights[long_idx]))
        short_idx = .!long_idx
        short_budget = max(short_budget, sum(weights[short_idx]))
        short_investment = -investment * short_budget
    else
        long_budget = budget
        short_idx = Vector{eltype(weights)}(undef, 0)
        short_investment = zero(eltype(weights))
    end

    long_investment = investment * long_budget

    return long_idx, short_idx, long_investment, short_investment
end
