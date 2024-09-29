"""
```
abstract type DistanceMethod end
```

Abstract type for subtyping methods for computing distance matrices from correlation ones.
"""
abstract type DistanceMethod end

"""
```
@kwdef mutable struct DistMLP <: DistanceMethod
    absolute::Bool = false
end
```

Defines the distance matrix from a correlation matrix [HRP1](@cite) in [`dist`](@ref).

```math
\\begin{align}
D_{i,\\,j} &= 
    \\begin{cases}
        \\sqrt{\\dfrac{1}{2} \\left(1 - C_{i,\\,j}\\right)} &\\quad \\mathrm{if~ absolute = false}\\\\
        \\sqrt{1 - \\lvert C_{i,\\,j} \\rvert} &\\quad \\mathrm{if~ absolute = true}\\,.
    \\end{cases}
\\end{align}
```

Where:

  - ``D_{i,\\,j}``: is the ``(i,\\,j)``-th entry of the `N×N` distance matrix ``\\mathbf{C}``.

  - ``C_{i,\\,j}``: is the ``(i,\\,j)``-th entry of the `N×N` correlation matrix ``\\mathbf{D}``.
  - absolute:

      + if `true`: the correlation being used is absolute.

# Parameters

  - `absolute`:

      + if `true`: the correlation being used is absolute.
"""
mutable struct DistMLP <: DistanceMethod
    absolute::Bool
end
function DistMLP(; absolute::Bool = false)
    return DistMLP(absolute)
end

"""
```
@kwdef mutable struct DistDistMLP <: DistanceMethod
    absolute::Bool = false
    distance::Distances.UnionMetric = Distances.Euclidean()
    args::Tuple = ()
    kwargs::NamedTuple = (;)
end
```

Defines the distance of distances matrix from a correlation matrix [HRP1](@cite) in [`dist`](@ref).

```math
\\begin{align}
\\tilde{D}_{i,\\,j} &= f_{m}\\left(\\bm{D}_{i},\\, \\bm{D}_j\\right)\\,.
\\end{align}
```

Where:

  - ``\\bm{D}_{i}``: is the ``i``-th column/row of the `N×N` distance matrix defined in [`DistMLP`](@ref).

  - ``f_{m}``: is the pairwise distance function for metric ``m``. We use the [`Distances.pairwise`](https://github.com/JuliaStats/Distances.jl?tab=readme-ov-file#computing-pairwise-distances) function which computes the entire matrix at once, the output is a vector so it gets reshaped into an `N×N` matrix.
  - ``\\tilde{D}_{i,\\,j}``: is the ``(i,\\,j)``-th entry of the `N×N` distances of distances matrix.
  - absolute:

      + if `true`: the correlation being used is absolute.

# Parameters

  - `absolute`:

      + if `true`: the correlation being used is absolute.

  - `distance`: distance metric from [`Distances.jl`](https://github.com/JuliaStats/Distances.jl).
  - `args`: args for the [`Distances.pairwise`](https://github.com/JuliaStats/Distances.jl?tab=readme-ov-file#computing-pairwise-distances) function.
  - `kwargs`: key word args for the [`Distances.pairwise`](https://github.com/JuliaStats/Distances.jl?tab=readme-ov-file#computing-pairwise-distances) function.
"""
mutable struct DistDistMLP <: DistanceMethod
    absolute::Bool
    distance::Distances.UnionMetric
    args::Tuple
    kwargs::NamedTuple
end
function DistDistMLP(; absolute::Bool = false,
                     distance::Distances.UnionMetric = Distances.Euclidean(),
                     args::Tuple = (), kwargs::NamedTuple = (;))
    return DistDistMLP(absolute, distance, args, kwargs)
end
const AbsoluteDist = Union{DistMLP, DistDistMLP}

"""
```
struct DistLog <: DistanceMethod end
```

Defines the log-distance matrix from the correlation matrix.

```math
\\begin{align}
D_{i,\\,j} &= -\\log\\left(C_{i,\\,j}\\right)\\,.
\\end{align}
```

Where:

  - ``D_{i,\\,j}``: is the ``(i,\\,j)``-th entry of the `N×N` log-distance matrix.
  - ``C_{i,\\,j}``: is the  ``(i,\\,j)``-th entry of an absolute correlation matrix.
"""
struct DistLog <: DistanceMethod end

"""
```
struct DistCor <: DistanceMethod end
```

Defines the distance matrix from the correlation matrix.

```math
\\begin{align}
D_{i,\\,j} &= \\left(1 - C_{i,\\,j}\\right)}\\,.
\\end{algin}
```

Where:

  - ``D_{i,\\,j}``: is the ``(i,\\,j)``-th entry of the `N×N` distance matrix.
  - ``C_{i,\\,j}``: is the  ``(i,\\,j)``-th entry of a distance correlation matrix.
"""
struct DistCor <: DistanceMethod end

"""
```
struct DistCanonical <: DistanceMethod end
```

Struct for computing the canonical distance for a given correlation estimator in [`_get_default_dist`](@ref).

  - `isa(cor_type, CorMutualInfo)`: `DistVarInfo(; bins = cor_type.bins, normalise = cor_type.normalise)`.
  - `isa(cor_type.ce, CorMutualInfo)`: `DistVarInfo(; bins = cor_type.ce.bins, normalise = cor_type.ce.normalise)`.
  - `isa(cor_type.ce, CorLTD) || isa(cor_type, CorLTD)`: `DistLog()`.
  - else: `DistMLP()`.
"""
struct DistCanonical <: DistanceMethod end

"""
```
abstract type AbstractBins end
```

Abstract type for defining bin width estimation functions when computing [`DistVarInfo`](@ref) and [`CorMutualInfo`](@ref) distance and correlation matrices respectively.
"""
abstract type AbstractBins end

"""
```
abstract type AstroBins <: AbstractBins end
```

Abstract type for defining which bin width function to use from [`astropy`](https://docs.astropy.org/en/stable/visualization/histogram.html).
"""
abstract type AstroBins <: AbstractBins end

"""
```
struct Knuth <: AstroBins end
```

Knuth's bin width algorithm from [`astropy`](https://docs.astropy.org/en/stable/api/astropy.stats.knuth_bin_width.html#astropy.stats.knuth_bin_width).
"""
struct Knuth <: AstroBins end

"""
```
struct Freedman <: AstroBins end
```

Freedman's bin width algorithm from [`astropy`](https://docs.astropy.org/en/stable/api/astropy.stats.freedman_bin_width.html#astropy.stats.freedman_bin_width).
"""
struct Freedman <: AstroBins end

"""
```
struct Scott <: AstroBins end
```

Scott's bin width algorithm from [`astropy`](https://docs.astropy.org/en/stable/api/astropy.stats.scott_bin_width.html#astropy.stats.scott_bin_width).
"""
struct Scott <: AstroBins end

"""
```
struct HGR <: AbstractBins end
```

Hacine-Gharbi and Ravier's bin width algorithm [HGR](@cite).
"""
struct HGR <: AbstractBins end

"""
```
@kwdef mutable struct DistVarInfo <: DistanceMethod
    bins::Union{<:Integer, <:AbstractBins} = HGR()
    normalise::Bool = true
end
```

Defines the variation of information distance matrix.

# Parameters

  - `bins`: defines the bin function, or bin width directly and if so `bins > 0`.
  - `normalise`: whether or not to normalise the variation of information.
"""
mutable struct DistVarInfo <: DistanceMethod
    bins::Union{<:Integer, <:AbstractBins}
    normalise::Bool
end
function DistVarInfo(; bins::Union{<:Integer, <:AbstractBins} = HGR(),
                     normalise::Bool = true)
    if isa(bins, Integer)
        @smart_assert(bins > zero(bins))
    end
    return DistVarInfo(bins, normalise)
end
function Base.setproperty!(obj::DistVarInfo, sym::Symbol, val)
    if sym == :bins
        if isa(val, Integer)
            @smart_assert(val > zero(val))
        end
    end
    return setfield!(obj, sym, val)
end

export DistMLP, DistDistMLP, DistLog, DistCanonical, Knuth, Freedman, Scott, HGR,
       DistVarInfo
