abstract type CentralityType end

@kwdef mutable struct BetweennessCentrality <: CentralityType
    args::Tuple = ()
    kwargs::NamedTuple = (;)
end
@kwdef mutable struct ClosenessCentrality <: CentralityType
    args::Tuple = ()
    kwargs::NamedTuple = (;)
end
@kwdef mutable struct DegreeCentrality{T1 <: Integer} <: CentralityType
    type::T1 = 0
    kwargs::NamedTuple = (;)
end
struct EigenvectorCentrality <: CentralityType end
@kwdef mutable struct KatzCentrality{T1 <: Real} <: CentralityType
    alpha::T1 = 0.3
end
@kwdef mutable struct Pagerank{T1 <: Real, T2 <: Integer, T3 <: Real} <: CentralityType
    alpha::T1 = 0.85
    n::T2 = 100
    epsilon::T3 = 1e-6
end
struct RadialityCentrality <: CentralityType end
@kwdef mutable struct StressCentrality <: CentralityType
    args::Tuple = ()
    kwargs::NamedTuple = (;)
end

abstract type TreeType end
@kwdef mutable struct KruskalTree <: TreeType
    args::Tuple = ()
    kwargs::NamedTuple = (;)
end
@kwdef mutable struct BoruvkaTree <: TreeType
    args::Tuple = ()
    kwargs::NamedTuple = (;)
end
@kwdef mutable struct PrimTree <: TreeType
    args::Tuple = ()
    kwargs::NamedTuple = (;)
end

"""
```
abstract type NetworkType end
```
"""
abstract type NetworkType end

@kwdef mutable struct TMFG{T1 <: Integer} <: NetworkType
    similarity::DBHTSimilarity = DBHTMaxDist()
    steps::T1 = 1
    centrality::CentralityType = DegreeCentrality()
end

@kwdef mutable struct MST{T1 <: Integer} <: NetworkType
    tree::TreeType = KruskalTree()
    steps::T1 = 1
    centrality::CentralityType = DegreeCentrality()
end

export BetweennessCentrality, ClosenessCentrality, DegreeCentrality, EigenvectorCentrality,
       KatzCentrality, Pagerank, RadialityCentrality, StressCentrality, KruskalTree,
       BoruvkaTree, PrimTree, TMFG, MST
