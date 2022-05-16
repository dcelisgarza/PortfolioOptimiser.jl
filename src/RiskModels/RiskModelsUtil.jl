@inline function make_pos_def(::SFix, matrix)
    isposdef(matrix) && return matrix

    @warn("Covariance matrix is not positive definite. Fixing eigenvalues.")

    vals, vecs = eigen(matrix)
    vals[vals .< 0] .= 0
    fixed_matrix = vecs * Diagonal(vals) * vecs'

    _isposdef = isposdef(fixed_matrix)
    !_isposdef && @warn("Covariance matrix could not be fixed. Try a different risk model.")
    return fixed_matrix
end

@inline function make_pos_def(::DFix, matrix)
    isposdef(matrix) && return matrix

    @warn("Covariance matrix is not positive definite. Fixing eigenvalues.")

    vals = eigvals(matrix)

    min_val = minimum(vals)
    fixed_matrix = matrix - 1.1 * min_val * I(size(matrix, 1))

    _isposdef = isposdef(fixed_matrix)
    !_isposdef && @warn("Covariance matrix could not be fixed. Try a different risk model.")

    return fixed_matrix
end

import StatsBase.cov2cor
"""
```
cov2or(cov_mtx)
```
Uses the fact that the diagonal elements of the covariance matrix are the variances. Puts the covariances in a diagonal matrix, takes their square root and inverts it. It then left and right multiplies the covariance matrix. Basically performs all the necessary divisions by the individual standard deviations.
"""
cov2cor(cov_mtx) = cov2cor(cov_mtx, sqrt.(diag(cov_mtx)))
