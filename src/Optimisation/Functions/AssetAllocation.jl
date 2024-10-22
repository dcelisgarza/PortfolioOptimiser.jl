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
function allocate!(port::AbstractPortfolio;
                   type::Symbol = isa(port, Portfolio) ? :Trad : :HRP,
                   method::AllocationMethod = LP(), latest_prices = port.latest_prices,
                   investment::Real = 1e6, string_names::Bool = false,
                   short = isa(port, Portfolio) ? port.short : false,
                   budget = isa(port, Portfolio) ? port.budget : 1,
                   short_budget = isa(port, Portfolio) ? port.short_budget : 0)
    return _allocate!(method, port, type, latest_prices, investment, short, budget,
                      short_budget, string_names)
end
