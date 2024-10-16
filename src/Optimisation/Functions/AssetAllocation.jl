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
                   short_u = if isa(port, Portfolio)
                       min(port.short_u, port.short_budget)
                   else
                       0
                   end, long_u = isa(port, Portfolio) ? port.long_u : 1)
    return _allocate!(method, port, type, latest_prices, investment, short, long_u, short_u,
                      string_names)
end
