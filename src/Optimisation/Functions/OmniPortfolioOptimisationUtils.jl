function adjust_model_value_for_obj(model::JuMP.Model, val, ::Sharpe)
    return val /= value(model[:k])
end
function adjust_model_value_for_obj(::Any, val, ::Any)
    return val
end
"""
```
get_z_from_model
```
"""
function get_z_from_model(model::JuMP.Model, ::EVaR, obj::Any)
    return adjust_model_value_for_obj(model, value(model[:z_evar]), obj)
end
function get_z_from_model(model::JuMP.Model, ::AbstractVector{<:EVaR}, obj::Any)
    return adjust_model_value_for_obj(model, value.(model[:z_evar]), obj)
end
function get_z_from_model(model::JuMP.Model, ::RLVaR, obj::Any)
    return adjust_model_value_for_obj(model, value(model[:z_rvar]), obj)
end
function get_z_from_model(model::JuMP.Model, ::AbstractVector{<:RLVaR}, obj::Any)
    return adjust_model_value_for_obj(model, value.(model[:z_rvar]), obj)
end
function get_z_from_model(model::JuMP.Model, ::EDaR, obj::Any)
    return adjust_model_value_for_obj(model, value(model[:z_edar]), obj)
end
function get_z_from_model(model::JuMP.Model, ::AbstractVector{<:EDaR}, obj::Any)
    return adjust_model_value_for_obj(model, value.(model[:z_edar]), obj)
end
function get_z_from_model(model::JuMP.Model, ::RLDaR, obj::Any)
    return adjust_model_value_for_obj(model, value(model[:z_rdar]), obj)
end
function get_z_from_model(model::JuMP.Model, ::AbstractVector{<:RLDaR}, obj::Any)
    return adjust_model_value_for_obj(model, value.(model[:z_rdar]), obj)
end
function get_z(port::OmniPortfolio, rm::Union{AbstractVector, <:RiskMeasure}, obj::Any)
    return get_z_from_model(port.model, rm, obj)
end

export get_z_from_model, get_z