include("./AssetAllocationSetup.jl")
include("./AssetAllocationLP.jl")
include("./AssetAllocationGreedy.jl")
"""
```
allocate!(port::AbstractPortfolio,
                   type::Symbol = isa(port, Portfolio) ? :Trad : :HRP;
                   method::AllocationMethod = LP(), latest_prices = port.latest_prices,
                   investment::Real = 1e6, 
                   string_names::Bool = false)
```
"""
function allocate!(port::AbstractPortfolio,
                   type::Symbol = isa(port, Portfolio) ? :Trad : :HRP;
                   method::AllocationMethod = LP(), latest_prices = port.latest_prices,
                   investment::Real = 1e6, string_names::Bool = false)
    return _allocate!(method, port, type, latest_prices, investment, port.short,
                      port.long_u, port.short_u, string_names)
end
