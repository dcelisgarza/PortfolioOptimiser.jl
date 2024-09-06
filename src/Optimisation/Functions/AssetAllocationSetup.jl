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

    return long_idx, short_idx, long_investment, short_investment
end
