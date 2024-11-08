function _objective(::Trad, ::Sharpe, ::Union{AKelly, EKelly}, model)
    ret = model[:ret]
    @expression(model, obj_func, ret)
    if haskey(model, :obj_penalty)
        add_to_expression!(obj_func, -model[:obj_penalty])
    end
    @objective(model, Max, obj_func)
    return nothing
end
function _objective(::Trad, ::Sharpe, ::Any, model)
    if !haskey(model, :alt_sr)
        risk = model[:risk]
        @expression(model, obj_func, risk)
        if haskey(model, :obj_penalty)
            add_to_expression!(obj_func, model[:obj_penalty])
        end
        @objective(model, Min, obj_func)
    else
        ret = model[:ret]
        @expression(model, obj_func, ret)
        if haskey(model, :obj_penalty)
            add_to_expression!(obj_func, -model[:obj_penalty])
        end
        @objective(model, Max, obj_func)
    end
end
function _objective(::Trad, ::MinRisk, ::Any, model)
    risk = model[:risk]
    @expression(model, obj_func, risk)
    if haskey(model, :obj_penalty)
        add_to_expression!(obj_func, model[:obj_penalty])
    end
    @objective(model, Min, obj_func)
    return nothing
end
function _objective(::WC, obj::Sharpe, ::Any, model)
    ret = model[:ret]
    @expression(model, obj_func, ret)
    if haskey(model, :obj_penalty)
        add_to_expression!(obj_func, -model[:obj_penalty])
    end
    @objective(model, Max, obj_func)
    return nothing
end
function _objective(::WC, ::MinRisk, ::Any, model)
    risk = model[:risk]
    @expression(model, obj_func, risk)
    if haskey(model, :obj_penalty)
        add_to_expression!(obj_func, model[:obj_penalty])
    end
    @objective(model, Min, obj_func)
    return nothing
end
function _objective(::Any, obj::Utility, ::Any, model)
    ret = model[:ret]
    risk = model[:risk]
    l = obj.l
    @expression(model, obj_func, ret - l * risk)
    if haskey(model, :obj_penalty)
        add_to_expression!(obj_func, -model[:obj_penalty])
    end
    @objective(model, Max, obj_func)
    return nothing
end
function _objective(::Any, obj::MaxRet, ::Any, model)
    ret = model[:ret]
    @expression(model, obj_func, ret)
    if haskey(model, :obj_penalty)
        add_to_expression!(obj_func, -model[:obj_penalty])
    end
    @objective(model, Max, obj_func)
    return nothing
end
function objective_function(port, obj, ::Trad, kelly)
    _objective(Trad(), obj, kelly, port.model)
    return nothing
end
function objective_function(port, obj, ::WC, ::Any)
    _objective(WC(), obj, nothing, port.model)
    return nothing
end
function objective_function(port, ::Any, ::NOC, ::Any)
    model = port.model
    log_ret = model[:log_ret]
    log_risk = model[:log_risk]
    log_w = model[:log_w]
    log_1mw = model[:log_1mw]
    @expression(model, obj_func, -log_ret - log_risk - sum(log_w + log_1mw))
    @objective(model, Min, obj_func)
    return nothing
end
