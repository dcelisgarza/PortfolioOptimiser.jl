"""
```
abstract type AbstractNOC <: AbstractPortfolioOptimiser end
```

Abstract type for subtyping efficient frontier optimisers.
"""
abstract type AbstractNOC <: AbstractPortfolioOptimiser end

struct NearOptCentering{T1, T2, T3, T4, T5} <: AbstractNOC
    opt_port::T1
    c12::T2
    e12::T3
    weights::T4
    model::T5
end
function NearOptCentering(opt_port::AbstractEfficient)
    c12 = zeros(2)
    e12 = zeros(2)
    weights = zeros(length(opt_port.weights))
    model = copy(opt_port.model)

    return NearOptCentering(opt_port, c12, e12, weights, model)
end
