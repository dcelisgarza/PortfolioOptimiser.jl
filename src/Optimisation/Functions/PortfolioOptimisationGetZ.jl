
"""
```
get_z
```
"""
function get_z(port::Portfolio, rm::Union{AbstractVector, <:RiskMeasure}, obj::Any)
    return get_z_from_model(port.model, rm, obj)
end
