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

Apply no denoising in [`denoise!`](@ref).
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

Defines the parameters for using the fixed method in [`denoise!`](@ref) [MLAM; Chapter 2](@cite). This method performs an eigendecomposition of the original correlation matrix, sets the eigenvalues that are below the noise significance threshold to their average, and reconstructs the correlation matrix using the modified values.

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

Defines the parameters for using the spectral method in [`denoise!`](@ref) [MLAM; Chapter 2](@cite). This method performs an eigendecomposition of the original correlation matrix, sets the eigenvalues that are below the noise significance threshold to zero, and reconstructs the correlation matrix using the modified values.

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

Defines the parameters for using the shrink method in [`denoise!`](@ref) [MLAM; Chapter 2](@cite). This method shrinks the covariance to a target of unequal variance of the matrix reconstructed with the eigenvalues that are below the noise significance value. The degree of shrinkage is parametrised by `alpha`.

```math
\\begin{align}
\\mathbf{C} &= \\mathbf{C}_{u} + \\alpha \\mathbf{C}_{l} + (1 - \\alpha) \\mathrm{Diag}(\\mathbf{C})\\\\
\\mathbf{C}_{u} &= \\mathbf{E}_{u} \\mathbf{\\lambda}_{u} \\mathbf{E}_{u}^{\\intercal}\\\\
\\mathbf{C}_{l} &= \\mathbf{E}_{l} \\mathbf{\\lambda}_{l} \\mathbf{E}_{l}^{\\intercal}\\,.
\\end{align}
```

Where:

  - ``\\mathbf{C}`` is the denoised correlation matrix.
  - ``\\mathbf{C}_{u}`` is the reconstructed correlation matrix out of the eigenvalues that are above the noise significance threshold, and their corresponding eigenvectors.
  - ``\\mathbf{C}_{l}`` is the reconstructed correlation matrix out of the eigenvalues that are below the noise significance threshold, and their corresponding eigenvectors.
  - ``\\mathbf{E}_{u}`` are the eigenvectors corresponding to the eigenvalues that are above the noise significance threshold.
  - ``\\mathbf{\\lambda}_{u}`` is the diagonal matrix of eigenvalues that are above the noise significance threshold.
  - ``\\mathbf{E}_{l}`` are the eigenvectors corresponding to the eigenvalues that are below the noise significance threshold.
  - ``\\mathbf{\\lambda}_{l}`` is the diagonal matrix of eigenvalues that are below the noise significance threshold.

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
