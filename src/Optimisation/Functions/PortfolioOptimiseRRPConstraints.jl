function _rrp_ver_constraints(::BasicRRP, model, sigma)
    constr_scale = model[:constr_scale]
    w = model[:w]
    psi = model[:psi]
    G = sqrt(sigma)
    @constraint(model, [constr_scale * psi; constr_scale * G * w] ∈ SecondOrderCone())
    return nothing
end
function _rrp_ver_constraints(::RegRRP, model, sigma)
    constr_scale = model[:constr_scale]
    w = model[:w]
    psi = model[:psi]
    G = sqrt(sigma)
    @variable(model, rho >= 0)
    @constraints(model,
                 begin
                     [constr_scale * 2 * psi; constr_scale * 2 * G * w;
                      constr_scale * -2 * rho] ∈ SecondOrderCone()
                     [constr_scale * rho; constr_scale * G * w] ∈ SecondOrderCone()
                 end)
    return nothing
end
function _rrp_ver_constraints(version::RegPenRRP, model, sigma)
    constr_scale = model[:constr_scale]
    w = model[:w]
    psi = model[:psi]
    G = sqrt(sigma)
    theta = Diagonal(sqrt.(diag(sigma)))
    penalty = version.penalty
    @variable(model, rho >= 0)
    @constraints(model,
                 begin
                     [constr_scale * 2 * psi; constr_scale * 2 * G * w;
                      constr_scale * -2 * rho] ∈ SecondOrderCone()
                     [constr_scale * rho; constr_scale * sqrt(penalty) * theta * w] ∈
                     SecondOrderCone()
                 end)
    return nothing
end
function rrp_constraints(port::Portfolio, version, sigma)
    model = port.model
    constr_scale = model[:constr_scale]
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
                     constr_scale * zeta .== constr_scale * sigma * w
                     [i = 1:N],
                     [constr_scale * (w[i] + zeta[i])
                      constr_scale * (2 * gamma * sqrt(risk_budget[i]))
                      constr_scale * (w[i] - zeta[i])] ∈ SecondOrderCone()
                 end)
    _rrp_ver_constraints(version, model, sigma)
    return nothing
end
