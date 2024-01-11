"""
```
abstract type AbstractAllocation end
struct Greedy <: AbstractAllocation end
struct LP <: AbstractAllocation end
```

Types for allocation dispatch.

  - `Greedy`: Greedy allocation algorithm. Can do fractional shares.
  - `LP`: Mixed linear integer programming algorithm. Can only handle full shares.
"""
abstract type AbstractAllocation end
struct Greedy <: AbstractAllocation end
struct LP <: AbstractAllocation end

"""
```
struct Allocation{T1,T2,T3}
    tickers::T1
    shares::T2
    weights::T3
end
```

Concrete allocation created by [`Allocation`](@ref).

  - `tickers`: list of tickers.
  - `shares`: list of corresponding shares according to their optimal allocation.
  - `weights`: list of corresponding weights according to share allocation and latest prices.
"""
struct Allocation{T1, T2, T3}
    tickers::T1
    shares::T2
    weights::T3
end
