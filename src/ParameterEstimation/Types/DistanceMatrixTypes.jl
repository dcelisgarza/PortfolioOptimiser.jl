"""
```
abstract type DistanceMethod end
```

Abstract type for subtyping methods for computing distance matrices from correlation ones.
"""
abstract type DistanceMethod end

"""
```
@kwdef mutable struct DistanceMLP <: DistanceMethod
    absolute::Bool = false
end
```

Defines the distance matrix from a correlation matrix [HRP1](@cite) in [`dist`](@ref).

```math
D_{i,\\,j} = 
    \\begin{cases}
        \\sqrt{\\dfrac{1}{2} \\left(1 - C_{i,\\,j}\\right)} &\\quad \\mathrm{if~ absolute = false}\\\\
        \\sqrt{1 - \\lvert C_{i,\\,j} \\rvert} &\\quad \\mathrm{if~ absolute = true}\\nonumber\\,.
    \\end{cases}
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
mutable struct DistanceMLP <: DistanceMethod
    absolute::Bool
end
function DistanceMLP(; absolute::Bool = false)
    return DistanceMLP(absolute)
end

"""
```
@kwdef mutable struct DistanceSqMLP <: DistanceMethod
    absolute::Bool = false
    distance::Distances.UnionMetric = Distances.Euclidean()
    args::Tuple = ()
    kwargs::NamedTuple = (;)
end
```

Defines the distance of distances matrix from a correlation matrix [HRP1](@cite) in [`dist`](@ref).

```math
\\tilde{D}_{i,\\,j} = f_{m}\\left(\\bm{D}_{i},\\, \\bm{D}_j\\right)\\,.
```

Where:

  - ``\\bm{D}_{i}``: is the ``i``-th column/row of the `N×N` distance matrix defined in [`DistanceMLP`](@ref).

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
mutable struct DistanceSqMLP <: DistanceMethod
    absolute::Bool
    distance::Distances.UnionMetric
    args::Tuple
    kwargs::NamedTuple
end
function DistanceSqMLP(; absolute::Bool = false,
                       distance::Distances.UnionMetric = Distances.Euclidean(),
                       args::Tuple = (), kwargs::NamedTuple = (;))
    return DistanceSqMLP(absolute, distance, args, kwargs)
end
const AbsoluteDist = Union{DistanceMLP, DistanceSqMLP}

"""
```
struct DistanceLog <: DistanceMethod end
```

Defines the log-distance matrix from the correlation matrix.

```math
D_{i,\\,j} = -\\log\\left(C_{i,\\,j}\\right)\\,.
```

Where:

  - ``D_{i,\\,j}``: is the ``(i,\\,j)``-th entry of the `N×N` log-distance matrix.
  - ``C_{i,\\,j}``: is the  ``(i,\\,j)``-th entry of an absolute correlation matrix.
"""
struct DistanceLog <: DistanceMethod end

"""
```
struct DistanceCanonical <: DistanceMethod end
```

Struct for computing the canonical distance for a given correlation subtype of [`PortfolioOptimiserCovCor`](@ref).
"""
struct DistanceCanonical <: DistanceMethod end

"""
```
abstract type AbstractBins end
```

Abstract type for defining bin width estimation functions when computing [`DistanceVarInfo`](@ref) and [`CorMutualInfo`](@ref) distance and correlation matrices respectively.
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
@kwdef mutable struct DistanceVarInfo <: DistanceMethod
    bins::Union{<:Integer, <:AbstractBins} = HGR()
    normalise::Bool = true
end
```

Defines the variation of information distance matrix.

# Parameters

  - `bins`: defines the bin function, or bin width directly and if so `bins > 0`.
  - `normalise`: whether or not to normalise the variation of information.
"""
mutable struct DistanceVarInfo <: DistanceMethod
    bins::Union{<:Integer, <:AbstractBins}
    normalise::Bool
end
function DistanceVarInfo(; bins::Union{<:Integer, <:AbstractBins} = HGR(),
                         normalise::Bool = true)
    if isa(bins, Integer)
        @smart_assert(bins > zero(bins))
    end
    return DistanceVarInfo(bins, normalise)
end
function Base.setproperty!(obj::DistanceVarInfo, sym::Symbol, val)
    if sym == :bins
        if isa(val, Integer)
            @smart_assert(val > zero(val))
        end
    end
    return setfield!(obj, sym, val)
end

export DistanceMLP, DistanceSqMLP, DistanceLog, DistanceCanonical, Knuth, Freedman, Scott,
       HGR, DistanceVarInfo
