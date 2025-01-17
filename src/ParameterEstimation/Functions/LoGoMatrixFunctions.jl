"""
```
logo!(::NoLoGo, ::AbstractPosdefFix, ::AbstractMatrix, D = nothing)
```
"""
function logo!(::NoLoGo, ::AbstractPosdefFix, ::AbstractMatrix, D = nothing)
    return nothing
end
"""
```
logo!(je::LoGo, posdef::AbstractPosdefFix, X::AbstractMatrix, D = nothing)
```
"""
function logo!(je::LoGo, posdef::AbstractPosdefFix, X::AbstractMatrix, D = nothing)
    if isnothing(D)
        s = diag(X)
        iscov = any(.!isone.(s))
        S = if iscov
            s .= sqrt.(s)
            StatsBase.cov2cor(X, s)
        else
            X
        end
        D = dist(je.distance, S, nothing)
    end

    S = dbht_similarity(je.similarity, S, D)
    separators, cliques = PMFG_T2s(S, 4)[3:4]
    X .= J_LoGo(X, separators, cliques) \ I

    posdef_fix!(posdef, X)

    return nothing
end

export logo!
