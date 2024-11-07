function _objective(::Trad, ::Sharpe, ::Union{AKelly, EKelly}, model)
    ret = model[:ret]
    p = model[:obj_penalty]
    @objective(model, Max, ret - p)
    return nothing
end
function _objective(::Trad, ::Sharpe, ::Any, model)
    p = model[:obj_penalty]
    if !haskey(model, :alt_sr)
        risk = model[:risk]
        @objective(model, Min, risk + p)
    else
        ret = model[:ret]
        @objective(model, Max, ret - p)
    end
end
function _objective(::Trad, ::MinRisk, ::Any, model)
    risk = model[:risk]
    p = model[:obj_penalty]
    @objective(model, Min, risk + p)
    return nothing
end
function _objective(::WC, obj::Sharpe, ::Any, model)
    ret = model[:ret]
    p = model[:obj_penalty]
    @objective(model, Max, ret - p)
    return nothing
end
function _objective(::WC, ::MinRisk, ::Any, model)
    risk = model[:risk]
    p = model[:obj_penalty]
    @objective(model, Min, risk + p)
    return nothing
end
function _objective(::Any, obj::Utility, ::Any, model)
    ret = model[:ret]
    risk = model[:risk]
    p = model[:obj_penalty]
    l = obj.l
    @objective(model, Max, ret - l * risk - p)
    return nothing
end
function _objective(::Any, obj::MaxRet, ::Any, model)
    ret = model[:ret]
    p = model[:obj_penalty]
    @objective(model, Max, ret - p)
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
    @objective(model, Min, -log_ret - log_risk - sum(log_w + log_1mw))
    return nothing
end
