# Copywrite (c) 2025
# Author: Daniel Celis Garza <daniel.celis.garza@gmail.com>
# SPDX-License-Identifier: MIT

include("./PortfolioOptimisation.jl")
include("./PortfolioHCOptimisation.jl")
include("./AssetAllocation.jl")
function optimise!(port::Portfolio; type::AbstractOptimType = Trad())
    return optimise!(port, type)
end

export optimise!, frontier_limits!, efficient_frontier!, allocate!
