include("./AssetAllocationSetup.jl")
include("./AssetAllocationLP.jl")
include("./AssetAllocationGreedy.jl")
"""
```
allocate!(port::AbstractPortfolio;
                   type::Symbol = isa(port, Portfolio) ? :Trad : :HRP,
                   method::AllocationMethod = LP(), latest_prices = port.latest_prices,
                   investment::Real = 1e6, 
                   string_names::Bool = false)
```
"""
function allocate!(port::AbstractPortfolio; type::Symbol = :Trad,
                   method::AllocationMethod = LP(), latest_prices = port.latest_prices,
                   investment::Real = 1e6, short = port.short, budget = port.budget,
                   short_budget = port.short_budget, string_names::Bool = false)
    return _allocate!(method, port, type, latest_prices, investment, short, budget,
                      short_budget, string_names)
end
