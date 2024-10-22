function _setup_alloc_optim(weights, investment, short, budget, short_budget)
    long_idx = weights .>= zero(eltype(weights))

    if short
        long_budget = budget + short_budget
        short_idx = .!long_idx
        short_investment = investment * short_budget
    else
        long_budget = budget
        short_idx = Vector{eltype(weights)}(undef, 0)
        short_investment = zero(eltype(weights))
    end

    long_investment = investment * long_budget

    return long_idx, short_idx, long_investment, short_investment
end
