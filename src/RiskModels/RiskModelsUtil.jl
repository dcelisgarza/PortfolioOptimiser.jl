"""
Utility function for fixing non-positive definite matrices.

## Spectral method

```
make_pos_def(::SFix, matrix)
```

Uses eigenvalue decomposition to make the negative eigenvalues equal to zero and reconstruct the matrix with new eigenvalues.

## Diagonal method

```
make_pos_def(::DFix, matrix, scale = 1.1)
```

Uses a Levenberg-Marquardt factor, `c = scale` * `min(eigenvalues)`, which is used to subtract a scaled identity matrix from `matrix`.
"""
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

@inline function make_pos_def(::DFix, matrix, scale = 1.1)
    isposdef(matrix) && return matrix

    @warn("Covariance matrix is not positive definite. Fixing eigenvalues.")

    vals = eigvals(matrix)

    min_val = minimum(vals)
    fixed_matrix = matrix - scale * min_val * I(size(matrix, 1))

    _isposdef = isposdef(fixed_matrix)
    !_isposdef && @warn("Covariance matrix could not be fixed. Try a different risk model.")

    return fixed_matrix
end

import StatsBase.cov2cor
"""
```
cov2or(cov_mtx)
```
Wraps `StatsBase.cov2cor` to provide the standard deviations vector from the diagonal of `cov_mtx`.
"""
cov2cor(cov_mtx) = cov2cor(cov_mtx, sqrt.(diag(cov_mtx)))
