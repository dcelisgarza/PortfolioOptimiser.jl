function rrb_ver_constraints(::BasicRRB, model, sigma)
    scale_constr = model[:scale_constr]
    w = model[:w]
    psi = model[:psi]
    G = sqrt(sigma)
    @constraint(model, constr_basic_rrp,
                [scale_constr * psi; scale_constr * G * w] ∈ SecondOrderCone())
    return nothing
end
function rrb_ver_constraints(::RegRRB, model, sigma)
    scale_constr = model[:scale_constr]
    w = model[:w]
    psi = model[:psi]
    G = sqrt(sigma)
    @variable(model, rho >= 0)
    @constraints(model,
                 begin
                     constr_reg_rrp_soc_1,
                     [scale_constr * 2 * psi; scale_constr * 2 * G * w;
                      scale_constr * -2 * rho] ∈ SecondOrderCone()
                     constr_reg_rrp_soc_2,
                     [scale_constr * rho; scale_constr * G * w] ∈ SecondOrderCone()
                 end)
    return nothing
end
function rrb_ver_constraints(version::RegPenRRB, model, sigma)
    scale_constr = model[:scale_constr]
    w = model[:w]
    psi = model[:psi]
    G = sqrt(sigma)
    theta = Diagonal(sqrt.(diag(sigma)))
    penalty = version.penalty
    @variable(model, rho >= 0)
    @constraints(model,
                 begin
                     constr_reg_pen_rrp_soc_1,
                     [scale_constr * 2 * psi; scale_constr * 2 * G * w;
                      scale_constr * -2 * rho] ∈ SecondOrderCone()
                     constr_reg_pen_rrp_soc_2,
                     [scale_constr * rho;
                      scale_constr * sqrt(penalty) * theta * w] ∈ SecondOrderCone()
                 end)
    return nothing
end
function rrb_constraints(port::Portfolio, version, sigma)
    model = port.model
    scale_constr = model[:scale_constr]
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
    # RRB constraints.
    @constraints(model,
                 begin
                     constr_rrp, scale_constr * zeta .== scale_constr * sigma * w
                     constr_rrp_soc[i = 1:N],
                     [scale_constr * (w[i] + zeta[i])
                      scale_constr * (2 * gamma * sqrt(risk_budget[i]))
                      scale_constr * (w[i] - zeta[i])] ∈ SecondOrderCone()
                 end)
    rrb_ver_constraints(version, model, sigma)
    return nothing
end
