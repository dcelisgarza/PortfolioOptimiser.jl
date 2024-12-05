"""
```
get_z_from_model
```
"""
function get_z_from_model(model::JuMP.Model, ::EVaR)
    return value(model[:z_evar]) / value(model[:k])
end
function get_z_from_model(model::JuMP.Model, ::AbstractVector{<:EVaR})
    return value.(model[:z_evar]) / value(model[:k])
end
function get_z_from_model(model::JuMP.Model, ::RLVaR)
    return value(model[:z_rvar]) / value(model[:k])
end
function get_z_from_model(model::JuMP.Model, ::AbstractVector{<:RLVaR})
    return value.(model[:z_rvar]) / value(model[:k])
end
function get_z_from_model(model::JuMP.Model, ::EDaR)
    return value(model[:z_edar]) / value(model[:k])
end
function get_z_from_model(model::JuMP.Model, ::AbstractVector{<:EDaR})
    return value.(model[:z_edar]) / value(model[:k])
end
function get_z_from_model(model::JuMP.Model, ::RLDaR)
    return value(model[:z_rdar]) / value(model[:k])
end
function get_z_from_model(model::JuMP.Model, ::AbstractVector{<:RLDaR})
    return value.(model[:z_rdar]) / value(model[:k])
end
function get_z(port::OmniPortfolio, rm::Union{AbstractVector, <:RiskMeasure})
    return get_z_from_model(port.model, rm)
end

export get_z_from_model, get_z
