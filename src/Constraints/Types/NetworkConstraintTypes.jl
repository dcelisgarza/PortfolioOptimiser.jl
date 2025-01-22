# Copywrite (c) 2025
# Author: Daniel Celis Garza <daniel.celis.garza@gmail.com>
# SPDX-License-Identifier: MIT

"""
```
abstract type CentralityType end
```
"""
abstract type CentralityType end

"""
```
@kwdef mutable struct BetweennessCentrality <: CentralityType
    args::Tuple = ()
    kwargs::NamedTuple = (;)
end
```
"""
mutable struct BetweennessCentrality <: CentralityType
    args::Tuple
    kwargs::NamedTuple
end
function BetweennessCentrality(; args::Tuple = (), kwargs::NamedTuple = (;))
    return BetweennessCentrality(args, kwargs)
end

"""
```
@kwdef mutable struct ClosenessCentrality <: CentralityType
    args::Tuple = ()
    kwargs::NamedTuple = (;)
end
```
"""
mutable struct ClosenessCentrality <: CentralityType
    args::Tuple
    kwargs::NamedTuple
end
function ClosenessCentrality(; args::Tuple = (), kwargs::NamedTuple = (;))
    return ClosenessCentrality(args, kwargs)
end

"""
```
@kwdef mutable struct DegreeCentrality{T1 <: Integer} <: CentralityType
    type::T1 = 0
    kwargs::NamedTuple = (;)
end
```
"""
mutable struct DegreeCentrality{T1 <: Integer} <: CentralityType
    type::T1
    kwargs::NamedTuple
end
function DegreeCentrality(; type::Integer = 0, kwargs::NamedTuple = (;))
    return DegreeCentrality(type, kwargs)
end

"""
```
struct EigenvectorCentrality <: CentralityType end
```
"""
struct EigenvectorCentrality <: CentralityType end

"""
```
@kwdef mutable struct KatzCentrality{T1 <: Real} <: CentralityType
    alpha::T1 = 0.3
end
```
"""
mutable struct KatzCentrality{T1 <: Real} <: CentralityType
    alpha::T1
end
function KatzCentrality(; alpha::Real = 0.3)
    return KatzCentrality(alpha)
end

"""
```
@kwdef mutable struct Pagerank{T1 <: Real, T2 <: Integer, T3 <: Real} <: CentralityType
    alpha::T1 = 0.85
    n::T2 = 100
    epsilon::T3 = 1e-6
end
```
"""
mutable struct Pagerank{T1 <: Real, T2 <: Integer, T3 <: Real} <: CentralityType
    alpha::T1
    n::T2
    epsilon::T3
end
function Pagerank(; alpha::Real = 0.85, n::Integer = 100, epsilon::Real = 1e-6)
    return Pagerank(alpha, n, epsilon)
end

"""
```
struct RadialityCentrality <: CentralityType end
```
"""
struct RadialityCentrality <: CentralityType end

"""
```
@kwdef mutable struct StressCentrality <: CentralityType
    args::Tuple = ()
    kwargs::NamedTuple = (;)
end
```
"""
mutable struct StressCentrality <: CentralityType
    args::Tuple
    kwargs::NamedTuple
end
function StressCentrality(; args::Tuple = (), kwargs::NamedTuple = (;))
    return StressCentrality(args, kwargs)
end

"""
```
abstract type TreeType end
```
"""
abstract type TreeType end

"""
```
@kwdef mutable struct KruskalTree <: TreeType
    args::Tuple = ()
    kwargs::NamedTuple = (;)
end
```
"""
mutable struct KruskalTree <: TreeType
    args::Tuple
    kwargs::NamedTuple
end
function KruskalTree(; args::Tuple = (), kwargs::NamedTuple = (;))
    return KruskalTree(args, kwargs)
end

"""
```
@kwdef mutable struct BoruvkaTree <: TreeType
    args::Tuple = ()
    kwargs::NamedTuple = (;)
end
```
"""
mutable struct BoruvkaTree <: TreeType
    args::Tuple
    kwargs::NamedTuple
end
function BoruvkaTree(; args::Tuple = (), kwargs::NamedTuple = (;))
    return BoruvkaTree(args, kwargs)
end

"""
```
@kwdef mutable struct PrimTree <: TreeType
    args::Tuple = ()
    kwargs::NamedTuple = (;)
end
```
"""
mutable struct PrimTree <: TreeType
    args::Tuple
    kwargs::NamedTuple
end
function PrimTree(; args::Tuple = (), kwargs::NamedTuple = (;))
    return PrimTree(args, kwargs)
end

"""
```
abstract type NetworkType end
```
"""
abstract type NetworkType end

"""
```
@kwdef mutable struct TMFG{T1 <: Integer} <: NetworkType
    similarity::DBHTSimilarity = DBHTMaxDist()
    steps::T1 = 1
    centrality::CentralityType = DegreeCentrality()
end
```
"""
mutable struct TMFG{T1 <: Integer} <: NetworkType
    similarity::DBHTSimilarity
    steps::T1
    centrality::CentralityType
end
function TMFG(; similarity::DBHTSimilarity = DBHTMaxDist(), steps::Integer = 1,
              centrality::CentralityType = DegreeCentrality())
    return TMFG{typeof(steps)}(similarity, steps, centrality)
end

"""
```
@kwdef mutable struct MST{T1 <: Integer} <: NetworkType
    tree::TreeType = KruskalTree()
    steps::T1 = 1
    centrality::CentralityType = DegreeCentrality()
end
```
"""
mutable struct MST{T1 <: Integer} <: NetworkType
    tree::TreeType
    steps::T1
    centrality::CentralityType
end
function MST(; tree::TreeType = KruskalTree(), steps::Integer = 1,
             centrality::CentralityType = DegreeCentrality())
    return MST(tree, steps, centrality)
end

export BetweennessCentrality, ClosenessCentrality, DegreeCentrality, EigenvectorCentrality,
       KatzCentrality, Pagerank, RadialityCentrality, StressCentrality, KruskalTree,
       BoruvkaTree, PrimTree, TMFG, MST
