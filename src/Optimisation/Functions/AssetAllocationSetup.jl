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
function calc_fixed_fees(w::AbstractVector, fees::Union{AbstractVector{<:Real}, Real},
                         tol_kwargs::NamedTuple, op::Function)
    return if isa(fees, Real) && !iszero(fees)
        idx1 = op(w, zero(eltype(w)))
        idx2 = .!isapprox.(w[idx1], zero(eltype(w)); tol_kwargs...)
        fees * sum(idx2)
    elseif isa(fees, AbstractVector) && !(isempty(fees) || all(iszero.(fees)))
        idx1 = op(w, zero(eltype(w)))
        idx2 = .!isapprox.(w[idx1], zero(eltype(w)); tol_kwargs...)
        sum(fees[idx1][idx2])
    else
        zero(eltype(w))
    end
end
function calc_fees(w::AbstractVector, rebalance::AbstractTR, latest_prices::AbstractVector)
    return if isa(rebalance, TR)
        fees_rebal = rebalance.val
        benchmark = rebalance.w
        if isa(fees_rebal, Real)
            sum(fees_rebal * abs.(benchmark .- w) .* latest_prices)
        else
            dot(fees_rebal, abs.(benchmark .- w) .* latest_prices)
        end
    else
        zero(eltype(w))
    end
end
function calc_fees(w::AbstractVector, latest_prices::AbstractVector, fees::Fees = Fees(),
                   rebalance::AbstractTR = NoTR())
    fees_long = calc_fees(w, fees.long, latest_prices, .>=)
    fees_fixed_long = calc_fixed_fees(w, fees.fixed_long, fees.tol_kwargs, .>=)
    fees_short = calc_fees(w, fees.short, latest_prices, .<)
    fees_fixed_short = -calc_fixed_fees(w, fees.fixed_short, fees.tol_kwargs, .<)
    fees_rebal = calc_fees(w, rebalance, latest_prices)
    return fees_long + fees_fixed_long + fees_short + fees_fixed_short + fees_rebal
end
function setup_alloc_optim(port, weights, investment)
    short = port.short
    budget = port.budget
    short_budget = port.short_budget
    latest_prices = port.latest_prices
    fees = port.fees
    rebalance = port.rebalance

    fees = calc_fees(weights, latest_prices, fees, rebalance)
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
