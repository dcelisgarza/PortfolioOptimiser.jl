"""
```
abstract type AbstractAllocation end
struct Lazy <: AbstractAllocation end
struct Greedy <: AbstractAllocation end
struct LP <: AbstractAllocation end
```

Types for allocation dispatch.

- `Lazy`: Lazy allocation algorithm. Can do fractional shares.
- `Greedy`: Greedy allocation algorithm. Can do fractional shares.
- `LP`: Mixed linear integer programming algorithm. Can only handle full shares.
"""
abstract type AbstractAllocation end
struct Lazy <: AbstractAllocation end
struct Greedy <: AbstractAllocation end
struct LP <: AbstractAllocation end

"""
```
struct Allocation{T1,T2,T3}
    tickers::T1
    weights::T2
    shares::T3
end
```
Concrete allocation created by [`Allocation`](@ref).

- `tickers`: list of tickers.
- `weights`: list of corresponding weights according to share allocation and latest prices.
- `shares`: list of corresponding shares according to their optimal allocation.
"""
struct Allocation{T1, T2, T3}
    tickers::T1
    weights::T2
    shares::T3
end
