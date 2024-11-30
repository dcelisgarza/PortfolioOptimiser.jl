function drcvar_risk(port, returns, l1, alpha, r)
    T, N = size(returns)
    model = port.model
    w = model[:w]
    get_portfolio_returns(model, returns)
    X = model[:X]

    a1 = -one(alpha)
    a2 = a1 - l1 * inv(alpha)
    l2 = l1 * (one(alpha) - inv(alpha))
    ovec = range(; start = one(alpha), stop = one(alpha), length = N)
    @variables(model, begin
                   lb
                   tau
                   s[1:T]
                   u[1:T, 1:N] >= 0
                   v[1:T, 1:N] >= 0
                   tu_drcvar[1:T]
                   tv_drcvar[1:T]
               end)
    @constraints(model,
                 begin
                     l1 * tau .+ a1 * X .+ (u .* (1 .+ returns)) * ovec .<= s
                     l2 * tau .+ a2 * X .+ (v .* (1 .+ returns)) * ovec .<= s
                     [i = 1:T],
                     [tu_drcvar[i]; -u[i, :] .- a1 * w] in MOI.NormInfinityCone(1 + N)
                     [i = 1:T],
                     [tv_drcvar[i]; -v[i, :] .- a2 * w] in MOI.NormInfinityCone(1 + N)
                     tu_drcvar .<= lb
                     tv_drcvar .<= lb
                 end)

    @expression(model, risk, r * lb + sum(s) * inv(T))

    return nothing
end