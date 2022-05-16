"""
```
roundmultiple(val, prec [, args...] [; kwargs...])
```
Round a number to a multiple of `prec`. Uses `args` and `kwargs` of the built-in `round`.
"""
function roundmultiple(val, prec, args...; kwargs...)
    return round(div(val, prec) * prec, args...; kwargs...)
end