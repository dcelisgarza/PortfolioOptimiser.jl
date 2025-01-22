# Copywrite (c) 2025
# Author: Daniel Celis Garza <daniel.celis.garza@gmail.com>
# SPDX-License-Identifier: MIT

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
function custom_objective(port, obj_func, sense::Integer,
                          custom_obj::Union{NoCustomObjective, Nothing})
    return nothing
end
function set_objective_function(port, ::Sharpe, ::Union{AKelly, EKelly}, custom_obj)
    model = port.model
    scale_obj = model[:scale_obj]
    ret = model[:ret]
    @expression(model, obj_func, ret)
    add_objective_penalty(model, obj_func, -1)
    custom_objective(port, obj_func, -1, custom_obj)
    @objective(model, Max, scale_obj * obj_func)
    return nothing
end
function set_objective_function(port, ::Sharpe, ::Any, custom_obj)
    model = port.model
    scale_obj = model[:scale_obj]
    if !haskey(model, :constr_sr_risk)
        risk = model[:risk]
        @expression(model, obj_func, risk)
        add_objective_penalty(model, obj_func, 1)
        custom_objective(port, obj_func, 1, custom_obj)
        @objective(model, Min, scale_obj * obj_func)
    else
        ret = model[:ret]
        @expression(model, obj_func, ret)
        add_objective_penalty(model, obj_func, -1)
        custom_objective(port, obj_func, -1, custom_obj)
        @objective(model, Max, scale_obj * obj_func)
    end
end
function set_objective_function(port, ::MinRisk, ::Any, custom_obj)
    model = port.model
    scale_obj = model[:scale_obj]
    risk = model[:risk]
    @expression(model, obj_func, risk)
    add_objective_penalty(model, obj_func, 1)
    custom_objective(port, obj_func, 1, custom_obj)
    @objective(model, Min, scale_obj * obj_func)
    return nothing
end
function set_objective_function(port, obj::Utility, ::Any, custom_obj)
    model = port.model
    scale_obj = model[:scale_obj]
    ret = model[:ret]
    risk = model[:risk]
    l = obj.l
    @expression(model, obj_func, ret - l * risk)
    add_objective_penalty(model, obj_func, -1)
    custom_objective(port, obj_func, -1, custom_obj)
    @objective(model, Max, scale_obj * obj_func)
    return nothing
end
function set_objective_function(port, ::MaxRet, ::Any, custom_obj)
    model = port.model
    scale_obj = model[:scale_obj]
    ret = model[:ret]
    @expression(model, obj_func, ret)
    add_objective_penalty(model, obj_func, -1)
    custom_objective(port, obj_func, -1, custom_obj)
    @objective(model, Max, scale_obj * obj_func)
    return nothing
end
function set_objective_function(port, ::NOC, custom_obj)
    model = port.model
    scale_obj = model[:scale_obj]
    log_ret = model[:log_ret]
    log_risk = model[:log_risk]
    log_w = model[:log_w]
    log_1mw = model[:log_1mw]
    @expression(model, obj_func, -log_ret - log_risk - sum(log_w + log_1mw))
    add_objective_penalty(model, obj_func, 1)
    custom_objective(port, obj_func, 1, custom_obj)
    @objective(model, Min, scale_obj * obj_func)
    return nothing
end
function set_objective_function(port, ::Union{RB, RRB}, custom_obj)
    model = port.model
    scale_obj = model[:scale_obj]
    risk = model[:risk]
    @expression(model, obj_func, risk)
    add_objective_penalty(model, obj_func, 1)
    custom_objective(port, obj_func, 1, custom_obj)
    @objective(model, Min, scale_obj * obj_func)
    return nothing
end
