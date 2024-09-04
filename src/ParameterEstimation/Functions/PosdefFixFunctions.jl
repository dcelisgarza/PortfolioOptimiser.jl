"""
```
_posdef_fix!(method::PosdefNearest, X::AbstractMatrix)
```

Overload this for other posdef fix methods.
"""
function _posdef_fix!(method::PosdefNearest, X::AbstractMatrix)
    NearestCorrelationMatrix.nearest_cor!(X, method)
    return nothing
end
function posdef_fix!(::NoPosdef, ::AbstractMatrix)
    return nothing
end
"""
```
posdef_fix!(method::PosdefFix, X::AbstractMatrix)
```
"""
function posdef_fix!(method::PosdefFix, X::AbstractMatrix)
    if isposdef(X)
        return nothing
    end

    s = diag(X)
    iscov = any(.!isone.(s))
    _X = if iscov
        s .= sqrt.(s)
        cov2cor(X, s)
    else
        X
    end

    _posdef_fix!(method, _X)

    if !isposdef(_X)
        @warn("Matrix could not be made positive definite.")
        return nothing
    end

    if iscov
        StatsBase.cor2cov!(_X, s)
    end

    X .= _X

    return nothing
end

export posdef_fix!