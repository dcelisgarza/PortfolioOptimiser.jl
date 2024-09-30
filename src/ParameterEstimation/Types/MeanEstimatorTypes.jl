"""
```
abstract type MeanEstimator end
```

Abstract type for subtyping expected returns vectors estimators.
"""
abstract type MeanEstimator end

"""
```
abstract type MeanSigmaEstimator <: MeanEstimator end
```

Abstract type for subtyping expected returns vectors estimators that use covariance matrices for their corrections.
"""
abstract type MeanSigmaEstimator <: MeanEstimator end

"""
```
abstract type MeanTarget end
```

Abstract type for subtyping correction targets of expected returns estimators that use covariance matrices for their corrections.
"""
abstract type MeanTarget end

"""
```
struct GM <: MeanTarget end
```

Grand mean target.
"""
struct GM <: MeanTarget end

"""
```
struct VW <: MeanTarget end
```

Volatility-weighted grand mean.
"""
struct VW <: MeanTarget end

"""
```
struct SE <: MeanTarget end
```

Mean square error of sample mean.
"""
struct SE <: MeanTarget end

"""
```
@kwdef mutable struct MuSimple <: MeanEstimator
    w::Union{<:AbstractWeights, Nothing} = nothing
end
```

Simple expected returns vector estimator.

# Parameters

  - `w`: optional `T×1` vector of weights for computing the expected returns vector.
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

James-Stein [JS1, JS2](@cite) expected returns vector estimator.

# Parameters

  - `target`: correction target for the estimator.

  - `w`: optional `T×1` vector of weights for computing the simple expected returns vector.
  - `sigma`: value of the covariance matrix used for correcting the simple expected returns vector.

      + if computing with [`asset_statistics!`](@ref) and `isnothing(sigma)`: use the covariance matrix computed by .
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

Bayes-Stein [BS](@cite) expected returns vector estimator.

# Parameters

  - `target`: correction target for the estimator.

  - `w`: optional `T×1` vector of weights for computing the simple expected returns vector.
  - `sigma`: value of the covariance matrix used for correcting the simple expected returns vector.

      + if computing with [`asset_statistics!`](@ref) and `isnothing(sigma)`: use the covariance matrix computed by .
"""
mutable struct MuBS <: MeanSigmaEstimator
    target::MeanTarget
    w::Union{<:AbstractWeights, Nothing}
    sigma::Union{<:AbstractMatrix, Nothing}
end
function MuBS(; target::MeanTarget = GM(), w::Union{<:AbstractWeights, Nothing} = nothing,
              sigma::Union{<:AbstractMatrix, Nothing} = nothing)
    return MuBS(target, w, sigma)
end

"""
```
@kwdef mutable struct MuBOP <: MeanSigmaEstimator
    target::MeanTarget = GM()
    w::Union{<:AbstractWeights, Nothing} = nothing
    sigma::Union{<:AbstractMatrix, Nothing} = nothing
end
```

Bodnar-Okhrin-Parolya [BOP](@cite) expected returns vector estimator.

# Parameters

  - `target`: correction target for the estimator.

  - `w`: optional `T×1` vector of weights for computing the simple expected returns vector.
  - `sigma`: value of the covariance matrix used for correcting the simple expected returns vector.

      + if computing with [`asset_statistics!`](@ref) and `isnothing(sigma)`: use the covariance matrix computed by .
"""
mutable struct MuBOP <: MeanSigmaEstimator
    target::MeanTarget
    w::Union{<:AbstractWeights, Nothing}
    sigma::Union{<:AbstractMatrix, Nothing}
end
function MuBOP(; target::MeanTarget = GM(), w::Union{<:AbstractWeights, Nothing} = nothing,
               sigma::Union{<:AbstractMatrix, Nothing} = nothing)
    return MuBOP(target, w, sigma)
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
