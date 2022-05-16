using JuMP, HiGHS, LinearAlgebra

include("AssetAllocationType.jl")
include("AssetAllocationFunc.jl")
include("AssetAllocationUtil.jl")

export AbstractAllocation, Lazy, Greedy, LP
export Allocation
export roundmult