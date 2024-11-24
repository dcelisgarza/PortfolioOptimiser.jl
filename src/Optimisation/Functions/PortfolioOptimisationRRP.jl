function _rrp_constraints(port, type::RRP, sigma)
    model = port.model
    N = size(port.returns, 2)
    @variable(model, psi)
    @variable(model, gamma >= 0)
    @variable(model, zeta[1:N] .>= 0)
    @expression(model, risk, psi - gamma)
    # RRP constraints.
    w = model[:w]
    @constraint(model, zeta .== sigma * w)
    @constraint(model, sum(w) == 1)
    @constraint(model, w >= 0)
    @constraint(model, [i = 1:N],
                [w[i] + zeta[i]
                 2 * gamma * sqrt(port.risk_budget[i])
                 w[i] - zeta[i]] ∈ SecondOrderCone())
    _rrp_ver_constraints(type.version, model, sigma)
    return nothing
end
function rrp_constraints(type::RRP, port, sigma)
    model = port.model
    @variable(model, k)
    if isempty(port.risk_budget)
        port.risk_budget = ()
    elseif !isapprox(sum(port.risk_budget), one(eltype(port.returns)))
        port.risk_budget ./= sum(port.risk_budget)
    end
    _rrp_constraints(port, type, sigma)
    return nothing
end
function _optimise!(type::RRP, port::Portfolio, ::Any, ::Any, ::Any, class::PortClass,
                    w_ini::AbstractVector, ::Any, str_names::Bool)
    mu, sigma = mu_sigma_returns_class(port, class)[1:2]
    port.model = JuMP.Model()
    model = port.model
    set_string_names_on_creation(model, str_names)
    initial_w(port, w_ini)
    rrp_constraints(type, port, sigma)
    set_returns(nothing, NoKelly(), port; mu = mu)
    linear_constraints(port, nothing)
    risk = model[:risk]
    @objective(model, Min, risk)
    return convex_optimisation(port, nothing, type, class)
end
function _rrp_ver_constraints(::BasicRRP, model, sigma)
    w = model[:w]
    psi = model[:psi]
    G = sqrt(sigma)
    @constraint(model, [psi; G * w] ∈ SecondOrderCone())
    return nothing
end
function _rrp_ver_constraints(::RegRRP, model, sigma)
    w = model[:w]
    psi = model[:psi]
    G = sqrt(sigma)
    @variable(model, rho >= 0)
    @constraints(model, begin
                     [2 * psi; 2 * G * w; -2 * rho] ∈ SecondOrderCone()
                     [rho; G * w] ∈ SecondOrderCone()
                 end)
    return nothing
end
function _rrp_ver_constraints(version::RegPenRRP, model, sigma)
    w = model[:w]
    psi = model[:psi]
    G = sqrt(sigma)
    theta = Diagonal(sqrt.(diag(sigma)))
    penalty = version.penalty
    @variable(model, rho >= 0)
    @constraints(model, begin
                     [2 * psi; 2 * G * w; -2 * rho] ∈ SecondOrderCone()
                     [rho; sqrt(penalty) * theta * w] ∈ SecondOrderCone()
                 end)
    return nothing
end
function rrp_constraints(port::OmniPortfolio, type::RRP, sigma)
    model = port.model
    w = model[:w]
    N = length(w)

    risk_budget = port.risk_budget
    if isempty(risk_budget)
        risk_budget = port.risk_budget = fill(inv(N), N)
    end

    @variables(model, begin
                   psi >= 0
                   gamma >= 0
                   zeta[1:N] .>= 0
               end)
    @expression(model, risk, psi - gamma)
    # RRP constraints.
    @constraints(model,
                 begin
                     zeta .== sigma * w
                     [i = 1:N],
                     [w[i] + zeta[i]
                      2 * gamma * sqrt(risk_budget[i])
                      w[i] - zeta[i]] ∈ SecondOrderCone()
                 end)
    _rrp_ver_constraints(type.version, model, sigma)
    return nothing
end
function _optimise!(type::RRP, port::OmniPortfolio, ::Any, ::Any, kelly::RetType,
                    class::PortClass, w_ini::AbstractVector, custom_constr, custom_obj,
                    ::Any, str_names::Bool = false)
    old_short = nothing
    if port.short
        old_short = port.short
        port.short = false
    end
    port.model = JuMP.Model()
    set_string_names_on_creation(port.model, str_names)
    mu, sigma, returns = mu_sigma_returns_class(port, class)
    # Weight constraints
    initial_w(port, w_ini)
    set_k(port, nothing)
    weight_constraints(port)
    MIP_constraints(port)
    SDP_network_cluster_constraints(port, nothing)
    # Tracking
    tracking_error_constraints(port, returns)
    turnover_constraints(port)
    # Fees
    management_fee(port)
    rebalance_fee(port)
    # Risk
    rrp_constraints(port, type, sigma)
    # Returns
    expected_return_constraints(port, nothing, kelly, mu, sigma, returns, nothing)
    # Objective function penalties
    L1_regularisation(port)
    L2_regularisation(port)
    SDP_network_cluster_penalty(port)
    # Custom constraints
    custom_constraint(port, custom_constr)
    # Objective function and custom penalties
    set_objective_function(port, type, custom_obj)
    retval = convex_optimisation(port, nothing, type, class)
    if !isnothing(old_short)
        port.short = old_short
    end
    return retval
end
