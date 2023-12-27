"""
```julia
AbstractPortfolio
```
Abstract type for portfolios. Concrete portfolios subtype this see [`Portfolio`](@ref) and [`HCPortfolio`](@ref).
"""
abstract type AbstractPortfolio end

include("_Portfolio_type.jl")
include("_Portfolio_optim_funcs.jl")
include("_HCPortfolio_type.jl")
include("_HCPortfolio_optim_funcs.jl")
include("_NearOptPortfolio_type.jl")
include("_NearOptPortfolio_funcs.jl")
include("_Asset_allocation.jl")

export Portfolio,
    HCPortfolio, opt_port!, frontier_limits!, efficient_frontier!, allocate_port!
