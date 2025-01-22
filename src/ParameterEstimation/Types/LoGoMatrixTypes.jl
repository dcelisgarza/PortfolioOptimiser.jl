# Copywrite (c) 2025
# Author: Daniel Celis Garza <daniel.celis.garza@gmail.com>
# SPDX-License-Identifier: MIT

"""
```
abstract type AbstractLoGo end
```

Abstract type for subtyping LoGo covariance and correlation matrix estimators.
"""
abstract type AbstractLoGo end

"""
```
struct NoLoGo <: AbstractLoGo end
```

Leave the matrix as is.
"""
struct NoLoGo <: AbstractLoGo end

"""
```
@kwdef mutable struct LoGo <: AbstractLoGo
    distance::DistType = DistMLP()
    similarity::DBHTSimilarity = DBHTMaxDist()
end
```

Compute the LoGo covariance and correlation matrix estimator.

# Parameters

  - `distance`: type for computing the distance (disimilarity) matrix from the correlation matrix if the distance matrix is not provided to [`logo!`](@ref).
  - `similarity`: type for computing the similarity matrix from the correlation and distance matrices. The distance matrix is used to compute sparsity pattern of the inverse of the LoGo covariance and correlation matrices.
"""
mutable struct LoGo <: AbstractLoGo
    distance::DistType
    similarity::DBHTSimilarity
end
function LoGo(; distance::DistType = DistMLP(), similarity::DBHTSimilarity = DBHTMaxDist())
    return LoGo(distance, similarity)
end

export NoLoGo, LoGo
