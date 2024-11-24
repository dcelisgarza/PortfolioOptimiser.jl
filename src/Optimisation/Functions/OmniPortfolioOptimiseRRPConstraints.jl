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
