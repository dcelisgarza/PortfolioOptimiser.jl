# Copywrite (c) 2025
# Author: Daniel Celis Garza <daniel.celis.garza@gmail.com>
# SPDX-License-Identifier: MIT

function setup_alloc_optim(weights, investment, short, budget, short_budget)
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
