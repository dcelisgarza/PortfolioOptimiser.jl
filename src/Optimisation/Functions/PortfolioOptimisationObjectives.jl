function _objective(::Trad, ::Sharpe, ::Union{AKelly, EKelly}, model, p)
    ret = model[:ret]
    @objective(model, Max, ret - p)
    return nothing
end
function _objective(::Trad, ::Sharpe, ::Any, model, p)
    if !haskey(model, :alt_sr)
        risk = model[:risk]
        @objective(model, Min, risk + p)
    else
        ret = model[:ret]
        @objective(model, Max, ret - p)
    end
end
function _objective(::Trad, ::MinRisk, ::Any, model, p)
    risk = model[:risk]
    @objective(model, Min, risk + p)
    return nothing
end
function _objective(::WC, obj::Sharpe, ::Any, model, p)
    ret = model[:ret]
    @objective(model, Max, ret - p)
    return nothing
end
function _objective(::WC, ::MinRisk, ::Any, model, p)
    risk = model[:risk]
    @objective(model, Min, risk + p)
    return nothing
end
function _objective(::Any, obj::Utility, ::Any, model, p)
    ret = model[:ret]
    risk = model[:risk]
    l = obj.l
    @objective(model, Max, ret - l * risk - p)
    return nothing
end
function _objective(::Any, obj::MaxRet, ::Any, model, p)
    ret = model[:ret]
    @objective(model, Max, ret - p)
    return nothing
end
function objective_function(port, obj, ::Trad, kelly)
    p = zero(eltype(port.returns))
    if haskey(port.model, :network_penalty)
        p += port.model[:network_penalty]
    end
    if haskey(port.model, :cluster_penalty)
        p += port.model[:cluster_penalty]
    end
    if haskey(port.model, :sum_t_rebal)
        p += port.model[:sum_t_rebal]
    end
    _objective(Trad(), obj, kelly, port.model, p)
    return nothing
end
function objective_function(port, obj, ::WC, ::Any)
    p = zero(eltype(port.returns))
    if haskey(port.model, :sum_t_rebal)
        p += port.model[:sum_t_rebal]
    end
    _objective(WC(), obj, nothing, port.model, p)
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
