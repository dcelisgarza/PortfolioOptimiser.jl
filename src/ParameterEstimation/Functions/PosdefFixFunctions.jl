"""
```
posdef_fix!(type::PosdefNearest, X::AbstractMatrix)
```

Overload this for other posdef fix types.
"""
function posdef_fix!(type::PosdefNearest, X::AbstractMatrix)
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

    NearestCorrelationMatrix.nearest_cor!(X, type)

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
function posdef_fix!(::NoPosdef, ::AbstractMatrix)
    return nothing
end

export posdef_fix!
