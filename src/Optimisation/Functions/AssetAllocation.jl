include("./AssetAllocationSetup.jl")
include("./AssetAllocationLP.jl")
include("./AssetAllocationGreedy.jl")
"""
```
allocate!(port::AbstractPortfolio;
                   type::Symbol = isa(port, Portfolio) ? :Trad : :HRP,
                   type::AllocationType = LP(), latest_prices = port.latest_prices,
                   investment::Real = 1e6, 
                   string_names::Bool = false)
```
"""
function allocate!(port::AbstractPortfolio; type::AllocationType = LP(),
                   key::Symbol = :Trad, latest_prices = port.latest_prices,
                   investment::Real = 1e6, short = port.short, budget = port.budget,
                   short_budget = port.short_budget, string_names::Bool = false)
    return allocate!(port, type; key = key, latest_prices = latest_prices,
                     investment = investment, short = short, budget = budget,
                     short_budget = short_budget, string_names = string_names)
end
