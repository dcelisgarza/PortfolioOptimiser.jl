function set_objective_function(port, obj, ::WC, ::Any)
    model = port.model
    _objective(WC(), obj, nothing, model)
    return nothing
end
function _objective(::WC, ::MinRisk, ::Any, model)
    risk = model[:risk]
    @expression(model, obj_func, risk)
    add_objective_penalty(model, obj_func, 1)
    if haskey(model, :obj_penalty)
        add_to_expression!(obj_func, model[:obj_penalty])
    end
    @objective(model, Min, obj_func)
    return nothing
end
function _objective(::WC, obj::Sharpe, ::Any, model)
    ret = model[:ret]
    @expression(model, obj_func, ret)
    add_objective_penalty(model, obj_func, -1)
    if haskey(model, :obj_penalty)
        add_to_expression!(obj_func, -model[:obj_penalty])
    end
    @objective(model, Max, obj_func)
    return nothing
end
