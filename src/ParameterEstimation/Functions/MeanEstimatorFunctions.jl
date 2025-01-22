# Copywrite (c) 2025
# Author: Daniel Celis Garza <daniel.celis.garza@gmail.com>
# SPDX-License-Identifier: MIT

function StatsBase.mean(me::MuSimple, X::AbstractMatrix; dims::Int = 1)
    return isnothing(me.w) ? mean(X; dims = dims) : mean(X, me.w; dims = dims)
end
"""
```
target_mean(::GM, mu::AbstractVector, sigma::AbstractMatrix, inv_sigma, T::Integer,
                     N::Integer)
```
"""
function target_mean(::GM, mu::AbstractVector, sigma::AbstractMatrix, inv_sigma, T::Integer,
                     N::Integer)
    return fill(mean(mu), N)
end
function target_mean(::VW, mu::AbstractVector, sigma::AbstractMatrix, inv_sigma, T::Integer,
                     N::Integer)
    ones = range(one(eltype(sigma)); stop = one(eltype(sigma)), length = N)
    if isnothing(inv_sigma)
        inv_sigma = sigma \ I
    end
    return fill(dot(ones, inv_sigma, mu) / dot(ones, inv_sigma, ones), N)
end
function target_mean(::MSE, mu::AbstractVector, sigma::AbstractMatrix, inv_sigma,
                     T::Integer, N::Integer)
    return fill(tr(sigma) / T, N)
end
function StatsBase.mean(me::MuJS, X::AbstractMatrix; dims::Int = 1)
    @smart_assert(dims ∈ (1, 2))
    T, N = dims == 1 ? size(X) : size(transpose(X))
    mu = vec(isnothing(me.w) ? mean(X; dims = dims) : mean(X, me.w; dims = dims))
    sigma = me.sigma
    b = target_mean(me.target, mu, sigma, nothing, T, N)
    evals = eigvals(sigma)
    mb = mu - b
    alpha = (N * mean(evals) - 2 * maximum(evals)) / dot(mb, mb) / T
    return (1 - alpha) * mu + alpha * b
end
function StatsBase.mean(me::MuBS, X::AbstractMatrix; dims::Int = 1)
    @smart_assert(dims ∈ (1, 2))
    T, N = dims == 1 ? size(X) : size(transpose(X))
    mu = vec(isnothing(me.w) ? mean(X; dims = dims) : mean(X, me.w; dims = dims))
    sigma = me.sigma
    inv_sigma = sigma \ I
    b = target_mean(me.target, mu, sigma, inv_sigma, T, N)
    mb = mu - b
    alpha = (N + 2) / ((N + 2) + T * dot(mb, inv_sigma, mb))
    return (1 - alpha) * mu + alpha * b
end
function StatsBase.mean(me::MuBOP, X::AbstractMatrix; dims::Int = 1)
    @smart_assert(dims ∈ (1, 2))
    T, N = dims == 1 ? size(X) : size(transpose(X))
    mu = vec(isnothing(me.w) ? mean(X; dims = dims) : mean(X, me.w; dims = dims))
    sigma = me.sigma
    inv_sigma = sigma \ I
    b = target_mean(me.target, mu, sigma, inv_sigma, T, N)
    u = dot(mu, inv_sigma, mu)
    v = dot(b, inv_sigma, b)
    w = dot(mu, inv_sigma, b)
    alpha = (u - N / (T - N)) * v - w^2
    alpha /= u * v - w^2
    beta = (1 - alpha) * w / u
    return alpha * mu + beta * b
end
function StatsBase.mean(me::MuEquil, X::AbstractMatrix; kwargs...)
    l = me.l
    w = me.w
    sigma = me.sigma
    if isnothing(sigma)
        sigma = X
    end
    N, M = size(sigma)
    @smart_assert(N == M)
    if isnothing(me.w)
        w = fill(inv(N), N)
    end
    @smart_assert(length(w) == N)

    return l * sigma * w
end

export target_mean
