function add_objective_penalty(model, obj_func, d)
    if haskey(model, :l1_reg)
        l1_reg = model[:l1_reg]
        add_to_expression!(obj_func, d, l1_reg)
    end
    if haskey(model, :l2_reg)
        l2_reg = model[:l2_reg]
        add_to_expression!(obj_func, d, l2_reg)
    end
    if haskey(model, :network_penalty)
        network_penalty = model[:network_penalty]
        add_to_expression!(obj_func, d, network_penalty)
    end
    if haskey(model, :cluster_penalty)
        cluster_penalty = model[:cluster_penalty]
        add_to_expression!(obj_func, d, cluster_penalty)
    end
    return nothing
end
function custom_objective(model, obj_func, sense::Integer,
                          custom_obj::Union{NoCustomObjective, Nothing})
    return nothing
end
function _objective(::Trad, ::Sharpe, ::Union{AKelly, EKelly}, model, custom_obj)
    obj_scale = model[:obj_scale]
    ret = model[:ret]
    @expression(model, obj_func, ret)
    add_objective_penalty(model, obj_func, -1)
    custom_objective(model, obj_func, -1, custom_obj)
    @objective(model, Max, obj_scale * obj_func)
    return nothing
end
function _objective(::Trad, ::Sharpe, ::Any, model, custom_obj)
    obj_scale = model[:obj_scale]
    if !haskey(model, :alt_sr)
        risk = model[:risk]
        @expression(model, obj_func, risk)
        add_objective_penalty(model, obj_func, 1)
        custom_objective(model, obj_func, 1, custom_obj)
        @objective(model, Min, obj_scale * obj_func)
    else
        ret = model[:ret]
        @expression(model, obj_func, ret)
        add_objective_penalty(model, obj_func, -1)
        custom_objective(model, obj_func, -1, custom_obj)
        @objective(model, Max, obj_scale * obj_func)
    end
end
function _objective(::Union{Trad, DRCVaR}, ::MinRisk, ::Any, model, custom_obj)
    obj_scale = model[:obj_scale]
    risk = model[:risk]
    @expression(model, obj_func, risk)
    add_objective_penalty(model, obj_func, 1)
    custom_objective(model, obj_func, 1, custom_obj)
    @objective(model, Min, obj_scale * obj_func)
    return nothing
end
function _objective(::Any, obj::Utility, ::Any, model, custom_obj)
    obj_scale = model[:obj_scale]
    ret = model[:ret]
    risk = model[:risk]
    l = obj.l
    @expression(model, obj_func, ret - l * risk)
    add_objective_penalty(model, obj_func, -1)
    custom_objective(model, obj_func, -1, custom_obj)
    @objective(model, Max, obj_scale * obj_func)
    return nothing
end
function _objective(::Any, obj::MaxRet, ::Any, model, custom_obj)
    obj_scale = model[:obj_scale]
    ret = model[:ret]
    @expression(model, obj_func, ret)
    add_objective_penalty(model, obj_func, -1)
    custom_objective(model, obj_func, -1, custom_obj)
    @objective(model, Max, obj_scale * obj_func)
    return nothing
end
function set_objective_function(port, obj, type::Union{Trad, DRCVaR}, kelly, custom_obj)
    model = port.model
    _objective(type, obj, kelly, model, custom_obj)
    return nothing
end
function set_objective_function(port, ::NOC, custom_obj)
    model = port.model
    obj_scale = model[:obj_scale]
    log_ret = model[:log_ret]
    log_risk = model[:log_risk]
    log_w = model[:log_w]
    log_1mw = model[:log_1mw]
    @expression(model, obj_func, -log_ret - log_risk - sum(log_w + log_1mw))
    add_objective_penalty(model, obj_func, 1)
    custom_objective(model, obj_func, 1, custom_obj)
    @objective(model, Min, obj_scale * obj_func)
    return nothing
end
function set_objective_function(port, ::Union{RP, RRP}, custom_obj)
    model = port.model
    obj_scale = model[:obj_scale]
    risk = model[:risk]
    @expression(model, obj_func, risk)
    add_objective_penalty(model, obj_func, 1)
    custom_objective(model, obj_func, 1, custom_obj)
    @objective(model, Min, obj_scale * obj_func)
    return nothing
end
