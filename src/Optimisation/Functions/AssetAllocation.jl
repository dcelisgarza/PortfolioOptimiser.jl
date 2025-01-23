# Copywrite (c) 2025
# Author: Daniel Celis Garza <daniel.celis.garza@gmail.com>
# SPDX-License-Identifier: MIT

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
                   key::Symbol = :Trad, investment::Real = 1e6, string_names::Bool = false)
    return allocate!(port, type; key = key, investment = investment,
                     string_names = string_names)
end
