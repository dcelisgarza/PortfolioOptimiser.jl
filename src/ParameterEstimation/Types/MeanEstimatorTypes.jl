"""
```
abstract type MeanEstimator end
```
"""
abstract type MeanEstimator end

"""
```
abstract type MeanTarget end
```
"""
abstract type MeanTarget end

"""
```
struct GM <: MeanTarget end
```
"""
struct GM <: MeanTarget end

"""
```
struct VW <: MeanTarget end
```
"""
struct VW <: MeanTarget end

"""
```
struct SE <: MeanTarget end
```
"""
struct SE <: MeanTarget end

"""
```
@kwdef mutable struct MuSimple <: MeanEstimator
    w::Union{<:AbstractWeights, Nothing} = nothing
end
```
"""
mutable struct MuSimple <: MeanEstimator
    w::Union{<:AbstractWeights, Nothing}
end
function MuSimple(; w::Union{<:AbstractWeights, Nothing} = nothing)
    return MuSimple(w)
end

"""
```
@kwdef mutable struct MuJS{T1} <: MeanEstimator
    target::MeanTarget = GM()
    w::Union{<:AbstractWeights, Nothing} = nothing
    sigma::T1 = Matrix{Float64}(undef, 0, 0)
end
```
"""
mutable struct MuJS{T1} <: MeanEstimator
    target::MeanTarget
    w::Union{<:AbstractWeights, Nothing}
    sigma::T1
end
function MuJS(; target::MeanTarget = GM(), w::Union{<:AbstractWeights, Nothing} = nothing,
              sigma::AbstractMatrix = Matrix{Float64}(undef, 0, 0))
    return MuJS{typeof(sigma)}(target, w, sigma)
end

"""
```
@kwdef mutable struct MuBS{T1} <: MeanEstimator
    target::MeanTarget = GM()
    w::Union{<:AbstractWeights, Nothing} = nothing
    sigma::T1 = Matrix{Float64}(undef, 0, 0)
end
```
"""
mutable struct MuBS{T1} <: MeanEstimator
    target::MeanTarget
    w::Union{<:AbstractWeights, Nothing}
    sigma::T1
end
function MuBS(; target::MeanTarget = GM(), w::Union{<:AbstractWeights, Nothing} = nothing,
              sigma::AbstractMatrix = Matrix{Float64}(undef, 0, 0))
    return MuBS{typeof(sigma)}(target, w, sigma)
end

"""
```
@kwdef mutable struct MuBOP{T1} <: MeanEstimator
    target::MeanTarget = GM()
    w::Union{<:AbstractWeights, Nothing} = nothing
    sigma::T1 = Matrix{Float64}(undef, 0, 0)
end
```
"""
mutable struct MuBOP{T1} <: MeanEstimator
    target::MeanTarget
    w::Union{<:AbstractWeights, Nothing}
    sigma::T1
end
function MuBOP(; target::MeanTarget = GM(), w::Union{<:AbstractWeights, Nothing} = nothing,
               sigma::AbstractMatrix = Matrix{Float64}(undef, 0, 0))
    return MuBOP{typeof(sigma)}(target, w, sigma)
end

export GM, VW, SE, MuSimple, MuJS, MuBS, MuBOP
