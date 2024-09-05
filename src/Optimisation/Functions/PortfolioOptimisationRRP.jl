function _rrp_ver_constraints(::BasicRRP, model, sigma)
    G = sqrt(sigma)
    psi = model[:psi]
    w = model[:w]
    @constraint(model, [psi; G * w] ∈ SecondOrderCone())
    return nothing
end
function _rrp_ver_constraints(::RegRRP, model, sigma)
    G = sqrt(sigma)
    @variable(model, rho)
    psi = model[:psi]
    w = model[:w]
    @constraint(model, [2 * psi; 2 * G * w; -2 * rho] ∈ SecondOrderCone())
    @constraint(model, [rho; G * w] ∈ SecondOrderCone())
    return nothing
end
function _rrp_ver_constraints(version::RegPenRRP, model, sigma)
    G = sqrt(sigma)
    @variable(model, rho)
    psi = model[:psi]
    w = model[:w]
    @constraint(model, [2 * psi; 2 * G * w; -2 * rho] ∈ SecondOrderCone())
    theta = Diagonal(sqrt.(diag(sigma)))
    @constraint(model, [rho; sqrt(version.penalty) * theta * w] ∈ SecondOrderCone())
    return nothing
end
function _rrp_constraints(type::RRP, port, sigma)
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
    _sd_risk(NoNtwk(), SOCSD(), model, sigma)
    _set_sd_risk_upper_bound(nothing, nothing, type, model, Inf)
    _rrp_constraints(type, port, sigma)
    return nothing
end
function _optimise!(type::RRP, port::Portfolio, ::Any, ::Any, ::Any,
                    class::Union{Classic, FM}, w_ini::AbstractVector, str_names::Bool)
    mu, sigma, returns = mu_sigma_returns_class(port, class)

    port.model = JuMP.Model()
    model = port.model
    set_string_names_on_creation(model, str_names)
    initial_w(port, w_ini)
    rrp_constraints(type, port, sigma)
    set_returns(nothing, NoKelly(), port.model, port.mu_l; mu = mu)
    linear_constraints(port, nothing)
    risk = model[:risk]
    @objective(model, Min, risk)
    return convex_optimisation(port, nothing, type, class)
end
