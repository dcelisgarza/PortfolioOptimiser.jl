"""
```
dbht_similarity(::DBHTExp, S, D)
```
"""
function dbht_similarity(::DBHTExp, S, D)
    return exp.(-D)
end
function dbht_similarity(::DBHTMaxDist, S, D)
    return ceil(maximum(D)^2) .- D .^ 2
end

export dbht_similarity