# Copywrite (c) 2025
# Author: Daniel Celis Garza <daniel.celis.garza@gmail.com>
# SPDX-License-Identifier: MIT

"""
```
abstract type DistType end
```

Abstract type for subtyping types for computing distance matrices from correlation ones.
"""
abstract type DistType end

abstract type AbsoluteDistType <: DistType end

"""
```
@kwdef mutable struct DistMLP <: AbsoluteDistType
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
mutable struct DistMLP <: AbsoluteDistType
    overwrite::Bool
    absolute::Bool
end
function DistMLP(; overwrite::Bool = true, absolute::Bool = false)
    return DistMLP(overwrite, absolute)
end

"""
    mutable struct GenDistMLP{T1} <: AbsoluteDistType
"""
mutable struct GenDistMLP{T1} <: AbsoluteDistType
    overwrite::Bool
    absolute::Bool
    power::T1
end
function GenDistMLP(; overwrite::Bool = true, absolute::Bool = false, power::Integer = 1)
    @smart_assert(power > zero(power))
    return GenDistMLP(overwrite, absolute, power)
end
function Base.setproperty!(obj::GenDistMLP, sym::Symbol, val)
    if sym == :power
        @smart_assert(val > zero(val))
    end
    return setfield!(obj, sym, val)
end

"""
```
@kwdef mutable struct DistDistMLP <: AbsoluteDistType
    absolute::Bool = false
    distance::Distances.Metric = Distances.Euclidean()
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

  - ``f_{m}``: is the pairwise distance function for metric ``m``. We use the [`Distances.pairwise`](https://github.com/JuliaStats/Distances.jl?tab=readme-ov-file#computing-pairwise-distances) function which computes the entire matrix at once.
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
mutable struct DistDistMLP <: AbsoluteDistType
    overwrite::Bool
    absolute::Bool
    distance::Distances.Metric
    args::Tuple
    kwargs::NamedTuple
end
function DistDistMLP(; overwrite::Bool = true, absolute::Bool = false,
                     distance::Distances.Metric = Distances.Euclidean(), args::Tuple = (),
                     kwargs::NamedTuple = (;))
    return DistDistMLP(overwrite, absolute, distance, args, kwargs)
end

"""
    mutable struct GenDistDistMLP{T1} <: AbsoluteDistType
"""
mutable struct GenDistDistMLP{T1} <: AbsoluteDistType
    overwrite::Bool
    absolute::Bool
    power::T1
    distance::Distances.Metric
    args::Tuple
    kwargs::NamedTuple
end
function GenDistDistMLP(; overwrite::Bool = true, absolute::Bool = false,
                        power::Integer = 1,
                        distance::Distances.Metric = Distances.Euclidean(),
                        args::Tuple = (), kwargs::NamedTuple = (;))
    @smart_assert(power > zero(power))
    return GenDistDistMLP(overwrite, absolute, power, distance, args, kwargs)
end
function Base.setproperty!(obj::GenDistDistMLP, sym::Symbol, val)
    if sym == :power
        @smart_assert(val > zero(val))
    end
    return setfield!(obj, sym, val)
end

"""
```
struct DistLog <: DistType end
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
struct DistLog <: DistType end

"""
```
@kwdef mutable struct DistDistLog <: DistType
    distance::Distances.Metric
    args::Tuple
    kwargs::NamedTuple
end
```

Defines the distance of distances matrix from the correlation matrix.

```math
\\begin{align}
D_{i,\\,j} &= -\\log\\left(C_{i,\\,j}\\right)\\,.
\\end{align}
```

Where:

  - ``D_{i,\\,j}``: is the ``(i,\\,j)``-th entry of the `N×N` log-distance matrix.
  - ``C_{i,\\,j}``: is the  ``(i,\\,j)``-th entry of an absolute correlation matrix.

# Parameters

  - `absolute`:

      + if `true`: the correlation being used is absolute.

  - `distance`: distance metric from [`Distances.jl`](https://github.com/JuliaStats/Distances.jl).
  - `args`: args for the [`Distances.pairwise`](https://github.com/JuliaStats/Distances.jl?tab=readme-ov-file#computing-pairwise-distances) function.
  - `kwargs`: key word args for the [`Distances.pairwise`](https://github.com/JuliaStats/Distances.jl?tab=readme-ov-file#computing-pairwise-distances) function.
"""
mutable struct DistDistLog <: DistType
    distance::Distances.Metric
    args::Tuple
    kwargs::NamedTuple
end
function DistDistLog(; distance::Distances.Metric = Distances.Euclidean(), args::Tuple = (),
                     kwargs::NamedTuple = (;))
    return DistDistLog(distance, args, kwargs)
end

"""
```
struct DistCor <: DistType end
```

Defines the distance matrix from the correlation matrix.

```math
\\begin{align}
D_{i,\\,j} &= \\sqrt{1 - C_{i,\\,j}}\\,.
\\end{align}
```

Where:

  - ``D_{i,\\,j}``: is the ``(i,\\,j)``-th entry of the `N×N` distance matrix.
  - ``C_{i,\\,j}``: is the  ``(i,\\,j)``-th entry of a distance correlation matrix.
"""
struct DistCor <: DistType end

"""
```
struct DistdistCor <: DistType end
```

Defines the distance of distances matrix from the correlation matrix.

```math
\\begin{align}
D_{i,\\,j} &= \\sqrt{1 - C_{i,\\,j}}\\,.
\\end{align}
```

Where:

  - ``D_{i,\\,j}``: is the ``(i,\\,j)``-th entry of the `N×N` distance matrix.
  - ``C_{i,\\,j}``: is the  ``(i,\\,j)``-th entry of a distance correlation matrix.
"""
struct DistDistCor <: DistType
    distance::Distances.Metric
    args::Tuple
    kwargs::NamedTuple
end
function DistDistCor(; distance::Distances.Metric = Distances.Euclidean(), args::Tuple = (),
                     kwargs::NamedTuple = (;))
    return DistDistCor(distance, args, kwargs)
end

"""
```
abstract type AbstractBins end
```

Abstract type for defining the bin width estimation functions when computing [`DistVarInfo`](@ref) and [`CovMutualInfo`](@ref) distance and correlation matrices respectively.
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
@kwdef mutable struct DistVarInfo <: DistType
    bins::Union{<:Integer, <:AbstractBins} = HGR()
    normalise::Bool = true
end
```

Defines the variation of information distance matrix.

# Parameters

  - `bins`:

      + if `isa(bins, AbstractBins)`: defines the function for computing bin widths.
      + if `isa(bins, Integer)` and `bins > 0`: directly provide the number of bins.

  - `normalise`:

      + if `true`: normalise the mutual information.
"""
mutable struct DistVarInfo <: DistType
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

"""
```
@kwdef mutable struct DistDistVarInfo <: DistType
    bins::Union{<:Integer, <:AbstractBins} = HGR()
    normalise::Bool = true
end
```

Defines the variation of information distance of distances matrix.

# Parameters

  - `bins`:

      + if `isa(bins, AbstractBins)`: defines the function for computing bin widths.
      + if `isa(bins, Integer)` and `bins > 0`: directly provide the number of bins.

  - `normalise`:

      + if `true`: normalise the mutual information.
"""
mutable struct DistDistVarInfo <: DistType
    bins::Union{<:Integer, <:AbstractBins}
    normalise::Bool
    distance::Distances.Metric
    args::Tuple
    kwargs::NamedTuple
end
function DistDistVarInfo(; bins::Union{<:Integer, <:AbstractBins} = HGR(),
                         normalise::Bool = true,
                         distance::Distances.Metric = Distances.Euclidean(),
                         args::Tuple = (), kwargs::NamedTuple = (;))
    if isa(bins, Integer)
        @smart_assert(bins > zero(bins))
    end
    return DistDistVarInfo(bins, normalise, distance, args, kwargs)
end
function Base.setproperty!(obj::DistDistVarInfo, sym::Symbol, val)
    if sym == :bins
        if isa(val, Integer)
            @smart_assert(val > zero(val))
        end
    end
    return setfield!(obj, sym, val)
end

"""
```
struct DistCanonical <: DistType end
```

Struct for computing the canonical distance for a given correlation estimator in [`default_dist`](@ref).

| Correlation estimator   | Canonical distance    |
|:----------------------- | ---------------------:|
| [`CovMutualInfo`](@ref) | [`DistVarInfo`](@ref) |
| [`CovLTD`](@ref)        | [`DistLog`](@ref)     |
| [`CovDistance`](@ref)   | [`DistCor`](@ref)     |
| Any other estimator     | [`DistMLP`](@ref)     |
"""
struct DistCanonical <: DistType end

"""
```
struct DistDistCanonical <: DistType end
```

Struct for computing the canonical distance for a given correlation estimator in [`default_dist`](@ref).

| Correlation estimator   | Canonical distance        |
|:----------------------- | -------------------------:|
| [`CovMutualInfo`](@ref) | [`DistDistVarInfo`](@ref) |
| [`CovLTD`](@ref)        | [`DistDistLog`](@ref)     |
| [`CovDistance`](@ref)   | [`DistDistCor`](@ref)     |
| Any other estimator     | [`DistDistMLP`](@ref)     |
"""
mutable struct DistDistCanonical <: DistType
    distance::Distances.Metric
    args::Tuple
    kwargs::NamedTuple
end
function DistDistCanonical(; distance::Distances.Metric = Distances.Euclidean(),
                           args::Tuple = (), kwargs::NamedTuple = (;))
    return DistDistCanonical(distance, args, kwargs)
end

export DistMLP, DistDistMLP, GenDistMLP, GenDistDistMLP, DistLog, DistDistLog, DistCor,
       DistDistCor, Knuth, Freedman, Scott, HGR, DistVarInfo, DistDistVarInfo,
       DistCanonical, DistDistCanonical
