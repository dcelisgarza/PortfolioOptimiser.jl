function _setup_alloc_optim(weights, investment, short, long_u, short_u)
    long_idx = weights .>= zero(eltype(weights))
    long_investment = investment * long_u

    if short
        short_idx = .!long_idx
        short_investment = investment * short_u
    else
        short_idx = Vector{eltype(weights)}(undef, 0)
        short_investment = zero(eltype(weights))
    end

    return long_idx, short_idx, long_investment, short_investment
end
