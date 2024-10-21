function _setup_alloc_optim(weights, investment, short)
    long_idx = weights .>= zero(eltype(weights))
    long_budget = sum(weights[long_idx])
    long_investment = investment * long_budget

    if short
        short_idx = .!long_idx
        short_budget = -sum(weights[short_idx])
        short_investment = investment * short_budget
    else
        short_idx = Vector{eltype(weights)}(undef, 0)
        short_investment = zero(eltype(weights))
    end

    return long_idx, short_idx, long_investment, short_investment
end
