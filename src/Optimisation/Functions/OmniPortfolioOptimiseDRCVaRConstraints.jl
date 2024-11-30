function drcvar_risk(port, returns, l1, alpha, r)
    T, N = size(returns)
    model = port.model
    w = model[:w]
    get_portfolio_returns(model, returns)
    X = model[:X]

    a1 = -one(alpha)
    a2 = a1 - l1 * inv(alpha)
    l2 = l1 * (1 - inv(alpha))
    ovec = range(; start = one(alpha), stop = one(alpha), length = N)
    @variables(model, begin
                   lb
                   tau
                   s[1:T]
                   u[1:T, 1:N] >= 0
                   v[1:T, 1:N] >= 0
               end)
    @constraints(model, begin
                     l1 * tau .+ a1 * X .+ (u .* (1 .+ returns)) * ovec .<= s
                     l2 * tau .+ a2 * X .+ (v .* (1 .+ returns)) * ovec .<= s
                 end)

    # constraints += [
    #     u * self._scale_constraints >= 0,
    #     v * self._scale_constraints >= 0,
    #     l * tau * self._scale_constraints
    #     + a1 * (prior_model.returns @ w) * self._scale_constraints
    #     + cp.multiply(u, (1 + prior_model.returns)) @ ovec * self._scale_constraints
    #     <= s * self._scale_constraints,
    #     l2 * tau * self._scale_constraints
    #     + a2 * (prior_model.returns @ w) * self._scale_constraints
    #     + cp.multiply(v, (1 + prior_model.returns)) @ ovec * self._scale_constraints
    #     <= s * self._scale_constraints,
    # ]

    # for i in range(T):
    #     # noinspection PyTypeChecker
    #     constraints.append(
    #         cp.norm(-u[i] - a1 * w, np.inf) * self._scale_constraints
    #         <= lb * self._scale_constraints
    #     )
    #     # noinspection PyTypeChecker
    #     constraints.append(
    #         cp.norm(-v[i] - a2 * w, np.inf) * self._scale_constraints
    #         <= lb * self._scale_constraints
    #     )

    # # custom objectives and constraints
    # custom_objective = self._get_custom_objective(w=w)
    # constraints += self._get_custom_constraints(w=w)

    # objective = cp.Minimize(
    #     cp.Constant(self.wasserstein_ball_radius) * lb * self._scale_objective
    #     + (1 / T) * cp.sum(s) * self._scale_objective
    #     + custom_objective * self._scale_objective
    # )

    return nothing
end