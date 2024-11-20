function _wc_return_constraints(::Box, port)
    model = port.model
    N = length(port.mu)
    @variable(model, abs_w[1:N])
    w = model[:w]
    @constraint(model, [i = 1:N], [abs_w[i]; w[i]] ∈ MOI.NormOneCone(2))
    @expression(model, ret, dot(port.mu, w) - dot(port.d_mu, abs_w))
    return nothing
end
function _wc_return_constraints(::Ellipse, port)
    model = port.model
    G = sqrt(port.cov_mu)
    w = model[:w]
    @expression(model, x_gw, G * w)
    @variable(model, t_gw)
    @constraint(model, [t_gw; x_gw] ∈ SecondOrderCone())
    @expression(model, ret, dot(port.mu, w) - port.k_mu * t_gw)
    return nothing
end
function _wc_return_constraints(::NoWC, port)
    w = port.model[:w]
    @expression(port.model, ret, dot(port.mu, w))
    return nothing
end
function _wc_risk_constraints(::Box, port, obj)
    _sdp(port, obj)
    model = port.model
    N = size(port.returns, 2)
    @variable(model, Au[1:N, 1:N] .>= 0, Symmetric)
    @variable(model, Al[1:N, 1:N] .>= 0, Symmetric)
    W = model[:W]
    @constraint(model, Au .- Al .== W)
    @expression(model, risk, tr(Au * port.cov_u) - tr(Al * port.cov_l))
    return nothing
end
function _wc_risk_constraints(::Ellipse, port, obj)
    _sdp(port, obj)
    sigma = port.cov
    G_sigma = sqrt(port.cov_sigma)
    model = port.model
    N = size(port.returns, 2)
    @variable(model, E[1:N, 1:N], Symmetric)
    @constraint(model, E ∈ PSDCone())
    W = model[:W]
    @expression(model, W_p_E, W .+ E)
    @expression(model, x_ge, G_sigma * vec(W_p_E))
    @variable(model, t_ge)
    @constraint(model, [t_ge; x_ge] ∈ SecondOrderCone())
    @expression(model, risk, tr(sigma * W_p_E) + port.k_sigma * t_ge)
    return nothing
end
function _wc_risk_constraints(type::NoWC, port, ::Any)
    _sd_risk(_get_ntwk_clust_method(Trad(), port), type.formulation, port.model, port.cov)
    sd_risk = port.model[:sd_risk]
    @expression(port.model, risk, sd_risk)
    return nothing
end
function _wc_sharpe_constraints(obj::Sharpe, port)
    ret = port.model[:ret]
    k = port.model[:k]
    add_to_expression!(ret, -obj.rf, k)
    risk = port.model[:risk]
    @constraint(port.model, risk <= 1)
    return nothing
end
function _wc_sharpe_constraints(::Any, ::Any)
    return nothing
end
function wc_constraints(port, obj, type)
    _wc_return_constraints(type.mu, port)
    _wc_risk_constraints(type.cov, port, obj)
    _wc_sharpe_constraints(obj, port)
    return nothing
end
function _optimise!(type::WC, port::Portfolio, ::Any, obj::ObjectiveFunction, ::Any, ::Any,
                    w_ini::AbstractVector,
                    c_const_obj_pen::Union{<:CustomConstraintObjectivePenalty, Nothing},
                    str_names::Bool)
    port.model = JuMP.Model()
    model = port.model
    set_string_names_on_creation(model, str_names)
    initial_w(port, w_ini)
    set_sr_k(obj, model)
    network_constraints(port.network_adj, port, obj, type)
    cluster_constraints(port.cluster_adj, port, obj, type)
    wc_constraints(port, obj, type)
    linear_constraints(port, obj)
    centrality_constraints(port, obj)
    weight_constraints(port, obj)
    num_assets_constraints(port, obj)
    tracking_err_constraints(port.tracking_err, port, port.returns, obj)
    turnover_constraints(port.turnover, port, obj)
    rebalance_penalty(port.rebalance, port, obj)
    L1_reg(port)
    L2_reg(port)
    custom_constraint_objective_penatly(c_const_obj_pen, port)
    set_objective_function(port, obj, type, nothing)
    return convex_optimisation(port, obj, type, nothing)
end
