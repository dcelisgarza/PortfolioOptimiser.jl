"""
```
abstract type MeanEstimator end
```
"""
abstract type MeanEstimator end
abstract type MeanSigmaEstimator <: MeanEstimator end

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
@kwdef mutable struct MuJS <: MeanSigmaEstimator
    target::MeanTarget = GM()
    w::Union{<:AbstractWeights, Nothing} = nothing
    sigma::Union{<:AbstractMatrix, Nothing} = nothing
end
```
"""
mutable struct MuJS <: MeanSigmaEstimator
    target::MeanTarget
    w::Union{<:AbstractWeights, Nothing}
    sigma::Union{<:AbstractMatrix, Nothing}
end
function MuJS(; target::MeanTarget = GM(), w::Union{<:AbstractWeights, Nothing} = nothing,
              sigma::Union{<:AbstractMatrix, Nothing} = nothing)
    return MuJS(target, w, sigma)
end
"""
```
@kwdef mutable struct MuBS <: MeanSigmaEstimator
    target::MeanTarget = GM()
    w::Union{<:AbstractWeights, Nothing} = nothing
    sigma::Union{<:AbstractMatrix, Nothing} = nothing
end
```
"""
mutable struct MuBS <: MeanSigmaEstimator
    target::MeanTarget
    w::Union{<:AbstractWeights, Nothing}
    sigma::Union{<:AbstractMatrix, Nothing}
end
function MuBS(; target::MeanTarget = GM(), w::Union{<:AbstractWeights, Nothing} = nothing,
              sigma::Union{<:AbstractMatrix, Nothing} = nothing)
    return MuBS{typeof(sigma)}(target, w, sigma)
end

"""
```
@kwdef mutable struct MuBOP <: MeanSigmaEstimator
    target::MeanTarget = GM()
    w::Union{<:AbstractWeights, Nothing} = nothing
    sigma::Union{<:AbstractMatrix, Nothing} = nothing
end
```
"""
mutable struct MuBOP <: MeanSigmaEstimator
    target::MeanTarget
    w::Union{<:AbstractWeights, Nothing}
    sigma::Union{<:AbstractMatrix, Nothing}
end
function MuBOP(; target::MeanTarget = GM(), w::Union{<:AbstractWeights, Nothing} = nothing,
               sigma::Union{<:AbstractMatrix, Nothing} = nothing)
    return MuBOP{typeof(sigma)}(target, w, sigma)
end

function set_mean_sigma(mu_type::MeanSigmaEstimator, sigma)
    old_sigma = mu_type.sigma
    if isnothing(mu_type.sigma) || isempty(mu_type.sigma)
        mu_type.sigma = sigma
    end
    return old_sigma
end
function set_mean_sigma(args...)
    return nothing
end

function unset_mean_sigma(mu_type::MeanSigmaEstimator, sigma)
    mu_type.sigma = sigma
    return nothing
end
function unset_mean_sigma(args...)
    return nothing
end

export GM, VW, SE, MuSimple, MuJS, MuBS, MuBOP
