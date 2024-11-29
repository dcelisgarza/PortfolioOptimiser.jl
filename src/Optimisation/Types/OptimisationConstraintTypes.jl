# ## Constraints

# ### Network

"""
```
abstract type AdjacencyConstraint end
```
"""
abstract type AdjacencyConstraint end

"""
```
struct NoAdj <: AdjacencyConstraint end
```
"""
struct NoAdj <: AdjacencyConstraint end

"""
```
@kwdef mutable struct SDP{T1 <: AbstractMatrix{<:Real}, T2 <: Real} <: AdjacencyConstraint
    A::T1 = Matrix{Float64}(undef, 0, 0)
    penalty::T2 = 0.05
end
```

Defines an approximate network constraint using semi-definite programming.

```math
\\begin{align}
\\begin{bmatrix}
\\mathbf{W} & \\bm{w}\\\\
\\bm{w}^{\\intercal} & 1
\\end{bmatrix} &\\succeq 0\\\\
\\mathbf{W} &= \\mathbf{W}^{\\intercal}\\\\
\\mathbf{A} \\odot \\mathbf{W} &= \\bm{0}\\,.
\\end{align}
```

Where:

  - ``\\mathbf{W}`` is an auxiliary variable that approximates the outer product of asset weights ``\\bm{w} \\otimes \\bm{w}``.
  - ``\\mathbf{A}`` is the ``N\\times N`` adjacency matrix. It tells us which assets are connected. The matrix can only take values of `1` or `0`. If entry ``(i,\\,j)`` is equal to `1`, assets ``i`` and ``j`` are connected.
  - ``\\odot`` is the Hadamard (element-wise) product.

When the variance risk measure [`SD`](@ref) is being used, whether in the objective function or as one of the risk constraints. Its definition will change when this constraint is active. The new definition is this.

```math
\\begin{align}
\\phi_{\\mathrm{var}}(\\bm{w}) &= \\mathrm{Tr}\\left(\\mathbf{\\Sigma}\\mathbf{W}\\right)\\,.
\\end{align}
```

Where:

  - ``\\mathrm{Tr}(\\cdot)`` is the trace operator.
  - ``\\mathbf{\\Sigma}`` is the covariance matrix.

However, this will not work if the variance is _not_ being constrained, or if it is _not_ in the objective function. For that we add/subtract the following penalty factor to the objective function.

```math
\\begin{align}
\\underset{\\bm{w}}{\\mathrm{opt}} &\\quad \\phi(\\bm{w}) \\pm \\lambda \\mathrm{Tr}\\left(\\mathbf{X}\\right)\\,.
\\end{align}
```

Where:

  - ``\\mathrm{opt}`` is ``\\mathrm{min}`` when the objective is convex and ``\\mathrm{max}`` when it is concave.
  - ``\\pm`` is ``+`` when the objective is convex and ``-`` when it is concave.
  - ``\\lambda`` is a penalty factor.

This approach works better than [`IP`](@ref) when ``\\mathbf{A}`` is close to the all ones matrix, even though it's an approximation.

# Parameters

  - `A`: `N×N` adjacency matrix.
  - `penalty`: penalty factor when the variance [`SD`](@ref) risk measure isn't being used, either in a constraint or in the objective function.
"""
mutable struct SDP{T1 <: AbstractMatrix{<:Real}, T2 <: Real} <: AdjacencyConstraint
    A::T1
    penalty::T2
end
function SDP(; A::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
             penalty::Real = 0.05)
    return SDP{typeof(A), typeof(penalty)}(A, penalty)
end

"""
```
@kwdef struct IP{T1 <: AbstractMatrix{<:Real},
                 T2 <: Union{<:Integer, <:AbstractVector{<:Integer}}, T3 <: Real} <: AdjacencyConstraint
    A::T1 = Matrix{Float64}(undef, 0, 0)
    k::T2 = 1
    scale::T3 = 100_000.0
end
```

Defines the exact network constraint using mixed-integer programming.

```math
\\begin{align}
\\left(\\mathbf{A} + \\mathbf{I}\\right)\\bm{y} &\\leq \\bm{k}\\\\
w_{i} &\\leq b_{u} y_{i} \\quad \\forall i = 1,\\,\\ldots,\\,N \\\\
w_{i} &\\geq b_{l} y_{i} \\quad \\forall i = 1,\\,\\ldots,\\,N\\,.
\\end{align}
```

Where:

  - ``\\mathbf{A}`` is the ``N \\times N`` adjacency matrix.

  - ``\\mathbf{I}`` is the identity matrix.

Since each row of ``\\left(\\mathbf{A} + \\mathbf{I}\\right)`` corresponds to a path, duplicate rows add no new information whilst increasing the problem's size, therefore we only store unique rows.

  - ``\\bm{y}`` is an ``N \\times 1`` vector of binary ``\\{0,\\,1\\}`` decision variables, which decide whether or not the asset should be included in the portfolio.

  - ``\\bm{k}``:

      + if is a vector: ``M\\times 1`` vector defining the maximum number of assets allowed per unique path, where ``M`` is the number of unique paths.
      + if is a scalar: defines the maximum number of assets allowed for all unique paths.
  - ``w_{i}`` is the ``i``-th asset weight.
  - ``b_{u},\\,b_{l}`` are the upper and lower bounds of the sum of the long and sum of the short asset weights, respectively.

    Thus the constraint means we will invest in _at most_ ``\\bm{k}`` assets per corresponding unique path.

This approach can be appied to any risk measure without work arounds like [`SDP`](@ref). However it is more computationally costly to optimise, and may fail when ``\\mathbf{A}`` is close to the all ones matrix.

# Parameters

  - `A`: adjacency matrix, only stores `unique(A + I, dims = 1)`.

  - `k`:

      + if is a vector: maximum number of assets per unique path.

          * if `A` is not empty, checks that the length of `k` is equal to the size of `unique(A + I, dims = 1)`.

      + if is a scalar: maximum number of assets for all unique paths.
  - `scale`: scaling variable for an auxiliary binary decision variable when optimising the [`Sharpe`](@ref) objective function.
"""
mutable struct IP{T1 <: AbstractMatrix{<:Real},
                  T2 <: Union{<:Integer, <:AbstractVector{<:Integer}}, T3 <: Real} <:
               AdjacencyConstraint
    A::T1
    k::T2
    scale::T3
end
function IP(; A::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
            k::Union{<:Integer, <:AbstractVector{<:Integer}} = 1, scale::Real = 100_000.0)
    if !isempty(A)
        A = unique(A + I; dims = 1)
    end
    if isa(k, AbstractVector) && !isempty(A) && !isempty(k)
        @smart_assert(size(A, 1) == length(k))
    end
    return IP{typeof(A), Union{<:Integer, <:AbstractVector{<:Integer}}, typeof(scale)}(A, k,
                                                                                       scale)
end
function Base.setproperty!(obj::IP, sym::Symbol, val)
    if sym == :k
        if isa(val, AbstractVector) && !isempty(obj.A) && !isempty(val)
            @smart_assert(size(obj.A, 1) == length(val))
        end
    elseif sym == :A
        if !isempty(val)
            val = unique(val + I; dims = 1)
        end
        if isa(obj.k, AbstractVector) && !isempty(val) && !isempty(obj.k)
            @smart_assert(size(val, 1) == length(obj.k))
        end
        val = convert(typeof(getfield(obj, sym)), val)
    end
    return setfield!(obj, sym, val)
end

# ### Return

"""
```
abstract type RetType end
```
"""
abstract type RetType end

"""
```
struct NoKelly <: RetType end
```
"""
mutable struct NoKelly <: RetType
    wc_set::WorstCaseSet
end
function NoKelly(; wc_set::WorstCaseSet = NoWC())
    return NoKelly(wc_set)
end

"""
```
@kwdef mutable struct AKelly <: RetType
    formulation::SDSquaredFormulation = SOC()
end
```
"""
mutable struct AKelly <: RetType
    formulation::SDSquaredFormulation
end
function AKelly(; formulation::SDSquaredFormulation = SOC())
    return AKelly(formulation)
end

"""
```
struct EKelly <: RetType end
```
"""
struct EKelly <: RetType end

# ### Tracking

"""
```
abstract type TrackingErr end
```
"""
abstract type TrackingErr end

"""
```
struct NoTracking <: TrackingErr end
```
"""
struct NoTracking <: TrackingErr end

"""
```
@kwdef mutable struct TrackWeight{T1 <: Real, T2 <: AbstractVector{<:Real}} <: TrackingErr
    err::T1 = 0.0
    w::T2 = Vector{Float64}(undef, 0)
end
```
"""
mutable struct TrackWeight{T1 <: Real, T2 <: AbstractVector{<:Real}} <: TrackingErr
    err::T1
    w::T2
end
function TrackWeight(; err::Real = 0.0,
                     w::AbstractVector{<:Real} = Vector{Float64}(undef, 0))
    return TrackWeight{typeof(err), typeof(w)}(err, w)
end

"""
```
@kwdef mutable struct TrackRet{T1 <: Real, T2 <: AbstractVector{<:Real}} <: TrackingErr
    err::T1 = 0.0
    w::T2 = Vector{Float64}(undef, 0)
end
```
"""
mutable struct TrackRet{T1 <: Real, T2 <: AbstractVector{<:Real}} <: TrackingErr
    err::T1
    w::T2
end
function TrackRet(; err::Real = 0.0, w::AbstractVector{<:Real} = Vector{Float64}(undef, 0))
    return TrackRet{typeof(err), typeof(w)}(err, w)
end

# ### Turnover and rebalance

"""
```
abstract type AbstractTR end
```
"""
abstract type AbstractTR end

"""
```
struct NoTR <: AbstractTR end
```
"""
struct NoTR <: AbstractTR end

"""
```
@kwdef mutable struct TR{T1 <: Union{<:Real, <:AbstractVector{<:Real}},
                         T2 <: AbstractVector{<:Real}} <: AbstractTR
    val::T1 = 0.0
    w::T2 = Vector{Float64}(undef, 0)
end
```
"""
mutable struct TR{T1 <: Union{<:Real, <:AbstractVector{<:Real}},
                  T2 <: AbstractVector{<:Real}} <: AbstractTR
    val::T1
    w::T2
end
function TR(; val::Union{<:Real, <:AbstractVector{<:Real}} = 0.0,
            w::AbstractVector{<:Real} = Vector{Float64}(undef, 0))
    return TR{typeof(val), typeof(w)}(val, w)
end

abstract type CustomConstraint end
struct NoCustomConstraint <: CustomConstraint end

export NoAdj, SDP, IP, NoKelly, AKelly, EKelly, NoTracking, TrackWeight, TrackRet, NoTR, TR
