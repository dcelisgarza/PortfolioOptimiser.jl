"""
```
abstract type Denoise end
```

Abstract type for subtyping denoising methods.
"""
abstract type Denoise end

"""
```
struct NoDenoise <: Denoise end
```

No denoising is performed in [`denoise!`](@ref).
"""
struct NoDenoise <: Denoise end

"""
```
@kwdef mutable struct DenoiseFixed{T1, T2, T3, T4} <: Denoise
    detone::Bool = false
    mkt_comp::Integer = 1
    kernel = AverageShiftedHistograms.Kernels.gaussian
    m::Integer = 10
    n::Integer = 1000
    args::Tuple = ()
    kwargs::NamedTuple = (;)
end
```

Defines the parameters for using the fixed method in [`denoise!`](@ref) [MLAM; Chapter 2](@cite).

# Parameters

  - `detone`:

      + `true`: remove the largest `mkt_comp` eigenvalues from the correlation matrix.

!!! warning

    Removing eigenvalues from the matrix may make it singular.

  - `mkt_comp`: the number of largest eigenvalues to remove from the correlation matrix.
  - `kernel`: kernel for fitting the average shifted histograms from [`AverageShiftedHistograms.jl` Kernel Functions](https://joshday.github.io/AverageShiftedHistograms.jl/latest/kernels/).
  - `m`: number of adjacent histograms to smooth over [`AverageShiftedHistograms.ash`](https://joshday.github.io/AverageShiftedHistograms.jl/latest/#Usage).
  - `n`: number of points used when creating the range of values to which the average shifted histogram is to be fitted [`AverageShiftedHistograms.ash`](https://joshday.github.io/AverageShiftedHistograms.jl/latest/#Usage).
  - `args`: arguments for [`Optim.optimize`](https://julianlsolvers.github.io/Optim.jl/stable/user/config/)
  - `kwargs`: keyword arguments for [`Optim.optimize`](https://julianlsolvers.github.io/Optim.jl/stable/user/config/)
"""
mutable struct DenoiseFixed{T1, T2, T3, T4} <: Denoise
    detone::Bool
    mkt_comp::T1
    kernel::T2
    m::T3
    n::T4
    args::Tuple
    kwargs::NamedTuple
end
function DenoiseFixed(; detone::Bool = false, mkt_comp::Integer = 1,
                      kernel = AverageShiftedHistograms.Kernels.gaussian, m::Integer = 10,
                      n::Integer = 1000, args::Tuple = (), kwargs::NamedTuple = (;))
    return DenoiseFixed{typeof(mkt_comp), typeof(kernel), typeof(m), typeof(n)}(detone,
                                                                                mkt_comp,
                                                                                kernel, m,
                                                                                n, args,
                                                                                kwargs)
end

"""
```
@kwdef mutable struct DenoiseSpectral{T1, T2, T3, T4} <: Denoise
    detone::Bool = false
    mkt_comp::Integer = 1
    kernel = AverageShiftedHistograms.Kernels.gaussian
    m::Integer = 10
    n::Integer = 1000
    args::Tuple = ()
    kwargs::NamedTuple = (;)
end
```

Defines the parameters for using the spectral method in [`denoise!`](@ref) [MLAM; Chapter 2](@cite).

# Parameters

  - `detone`:

      + `true`: take only the largest `mkt_comp` eigenvalues from the correlation matrix.

!!! warning

    Removing eigenvalues from the matrix may make it singular.

  - `mkt_comp`: the number of largest eigenvalues to keep from the correlation matrix.
  - `kernel`: kernel for fitting the average shifted histograms from [`AverageShiftedHistograms.jl` Kernel Functions](https://joshday.github.io/AverageShiftedHistograms.jl/latest/kernels/).
  - `m`: number of adjacent histograms to smooth over [`AverageShiftedHistograms.ash`](https://joshday.github.io/AverageShiftedHistograms.jl/latest/#Usage).
  - `n`: number of points used when creating the range of values to which the average shifted histogram is to be fitted [`AverageShiftedHistograms.ash`](https://joshday.github.io/AverageShiftedHistograms.jl/latest/#Usage).
  - `args`: arguments for [`Optim.optimize`](https://julianlsolvers.github.io/Optim.jl/stable/user/config/)
  - `kwargs`: keyword arguments for [`Optim.optimize`](https://julianlsolvers.github.io/Optim.jl/stable/user/config/)
"""
mutable struct DenoiseSpectral{T1, T2, T3, T4} <: Denoise
    detone::Bool
    mkt_comp::T1
    kernel::T2
    m::T3
    n::T4
    args::Tuple
    kwargs::NamedTuple
end
function DenoiseSpectral(; detone::Bool = false, mkt_comp::Integer = 1,
                         kernel = AverageShiftedHistograms.Kernels.gaussian,
                         m::Integer = 10, n::Integer = 1000, args::Tuple = (),
                         kwargs::NamedTuple = (;))
    return DenoiseSpectral{typeof(mkt_comp), typeof(kernel), typeof(m), typeof(n)}(detone,
                                                                                   mkt_comp,
                                                                                   kernel,
                                                                                   m, n,
                                                                                   args,
                                                                                   kwargs)
end

"""
```
@kwdef mutable struct DenoiseShrink{T1, T2, T3, T4, T5} <: Denoise
    alpha::Real = 0.0
    detone::Bool = false
    mkt_comp::Integer = 1
    kernel = AverageShiftedHistograms.Kernels.gaussian
    m::Integer = 10
    n::Integer = 1000
    args::Tuple = ()
    kwargs::NamedTuple = (;)
end
```

Defines the parameters for using the shrink method in [`denoise!`](@ref) [MLAM; Chapter 2](@cite).

# Parameters

  - `alpha`: tuning parameter for how much the matrix should be shrunk, `alpha ∈ [0, 1]`.

  - `detone`:

      + `true`: take only the largest `mkt_comp` eigenvalues from the correlation matrix.

!!! warning

    Removing eigenvalues from the matrix may make it singular.

  - `mkt_comp`: the number of largest eigenvalues to keep from the correlation matrix.
  - `kernel`: kernel for fitting the average shifted histograms from [`AverageShiftedHistograms.jl` Kernel Functions](https://joshday.github.io/AverageShiftedHistograms.jl/latest/kernels/).
  - `m`: number of adjacent histograms to smooth over [`AverageShiftedHistograms.ash`](https://joshday.github.io/AverageShiftedHistograms.jl/latest/#Usage).
  - `n`: number of points used when creating the range of values to which the average shifted histogram is to be fitted [`AverageShiftedHistograms.ash`](https://joshday.github.io/AverageShiftedHistograms.jl/latest/#Usage).
  - `args`: arguments for [`Optim.optimize`](https://julianlsolvers.github.io/Optim.jl/stable/user/config/)
  - `kwargs`: keyword arguments for [`Optim.optimize`](https://julianlsolvers.github.io/Optim.jl/stable/user/config/)
"""
mutable struct DenoiseShrink{T1, T2, T3, T4, T5} <: Denoise
    alpha::T1
    detone::Bool
    mkt_comp::T2
    kernel::T3
    m::T4
    n::T5
    args::Tuple
    kwargs::NamedTuple
end
function DenoiseShrink(; alpha::Real = 0.0, detone::Bool = false, mkt_comp::Integer = 1,
                       kernel = AverageShiftedHistograms.Kernels.gaussian, m::Integer = 10,
                       n::Integer = 1000, args::Tuple = (), kwargs::NamedTuple = (;))
    @smart_assert(zero(alpha) <= alpha <= one(alpha))
    return DenoiseShrink{typeof(alpha), typeof(mkt_comp), typeof(kernel), typeof(m),
                         typeof(n)}(alpha, detone, mkt_comp, kernel, m, n, args, kwargs)
end

export NoDenoise, DenoiseFixed, DenoiseSpectral, DenoiseShrink
