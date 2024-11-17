function _optimise!(type::Trad, port::Portfolio, rm::Union{AbstractVector, <:RiskMeasure},
                    obj::ObjectiveFunction, kelly::RetType, class::PortClass,
                    w_ini::AbstractVector,
                    c_const_obj_pen::Union{<:CustomConstraintObjectivePenalty, Nothing},
                    str_names::Bool)
    mu, sigma, returns = mu_sigma_returns_class(port, class)
    port.model = JuMP.Model()
    model = port.model
    set_string_names_on_creation(model, str_names)
    initial_w(port, w_ini)
    set_sr_k(obj, model)
    kelly_approx_idx = Int[]
    risk_constraints(port, obj, type, rm, mu, sigma, returns, kelly_approx_idx)
    return_constraints(port, obj, kelly, mu, sigma, returns, kelly_approx_idx)
    linear_constraints(port, obj)
    centrality_constraints(port, obj)
    weight_constraints(port, obj)
    num_assets_constraints(port, obj)
    network_constraints(port.network_adj, port, obj, type)
    cluster_constraints(port.cluster_adj, port, obj, type)
    tracking_err_constraints(port.tracking_err, port, returns, obj)
    turnover_constraints(port.turnover, port, obj)
    rebalance_penalty(port.rebalance, port, obj)
    L1_reg(port)
    L2_reg(port)
    custom_constraint_objective_penatly(c_const_obj_pen, port)
    objective_function(port, obj, type, kelly)
    return convex_optimisation(port, obj, type, class)
end
function optimal_homogenisation_factor(port, mu, ::Sharpe)
    val = if !isempty(mu)
        min(1e3, max(1e-3, mean(abs.(mu))))
    else
        1.0
    end
    model = port.model
    @expression(model, ohf, val)
    return nothing
end
function optimal_homogenisation_factor(args...)
    return nothing
end
function set_k(port, ::Sharpe)
    model = port.model
    @variable(model, k >= 0)
    return nothing
end
function set_k(port, ::Any)
    model = port.model
    @expression(model, k, zero(eltype(port.returns)))
    return nothing
end
function _optimise!(type::Trad, port::OmniPortfolio,
                    rm::Union{AbstractVector, <:RiskMeasure}, obj::ObjectiveFunction,
                    kelly::RetType, class::PortClass, w_ini::AbstractVector,
                    c_const_obj_pen::Union{<:CustomConstraintObjectivePenalty, Nothing},
                    str_names::Bool)
    port.model = JuMP.Model()
    set_string_names_on_creation(port.model, str_names)

    mu, sigma, returns = mu_sigma_returns_class(port, class)
    optimal_homogenisation_factor(port, mu, obj)
    initial_w(port, w_ini)
    set_k(port, obj)
    #! TODO
    # Risk constraints
    # Return constraints
    weight_constraints(port)
    MIP_constraints(port)
    tracking_error_constraints(port, returns)
    turnover_constraints(port)
    L1_regularisation(port)
    L2_regularisation(port)
    management_fee(port)
    rebalance_cost(port)
    SDP_network_cluster_constraints(port, true)
    SDP_network_cluster_constraints(port, false)
    return nothing
end