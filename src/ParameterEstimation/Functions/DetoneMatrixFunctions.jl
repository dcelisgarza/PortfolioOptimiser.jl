function detone!(ce::NoDetone, ::Any, X::AbstractMatrix)
    return nothing
end
function detone!(ce::Detone, posdef::AbstractPosdefFix, X::AbstractMatrix)
    mkt_comp = ce.mkt_comp
    @smart_assert(one(size(X, 1)) <= mkt_comp <= size(X, 1))
    mkt_comp -= 1

    s = diag(X)
    iscov = any(.!isone.(s))
    if iscov
        s .= sqrt.(s)
        StatsBase.cov2cor!(X, s)
    end

    vals, vecs = eigen(X)
    _vals = Diagonal(vals)[(end - mkt_comp):end, (end - mkt_comp):end]
    _vecs = vecs[:, (end - mkt_comp):end]
    X .-= _vecs * _vals * transpose(_vecs)
    X .= cov2cor(X)

    posdef_fix!(posdef, X)

    if iscov
        StatsBase.cor2cov!(X, s)
    end

    return nothing
end

export detone!
