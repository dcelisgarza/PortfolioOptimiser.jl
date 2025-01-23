# Copywrite (c) 2025
# Author: Daniel Celis Garza <daniel.celis.garza@gmail.com>
# SPDX-License-Identifier: MIT

function noc_constraints(port::Portfolio, risk0, ret0)
    model = port.model
    w = model[:w]
    scale_constr = model[:scale_constr]
    risk = model[:risk]
    ret = model[:ret]
    N = length(w)
    @variables(model, begin
                   log_ret
                   log_risk
                   log_w[1:N]
                   log_1mw[1:N]
               end)
    @constraints(model,
                 begin
                     constr_log_risk,
                     [scale_constr * log_risk, scale_constr * 1,
                      scale_constr * (risk0 - risk)] in MOI.ExponentialCone()
                     constr_log_ret,
                     [scale_constr * log_ret, scale_constr * 1,
                      scale_constr * (ret - ret0)] in MOI.ExponentialCone()
                     constr_log_w[i = 1:N],
                     [scale_constr * log_w[i], scale_constr * 1, scale_constr * w[i]] ∈
                     MOI.ExponentialCone()
                     constr_log_1mw[i = 1:N],
                     [scale_constr * log_1mw[i], scale_constr * 1,
                      scale_constr * (1 - w[i])] ∈ MOI.ExponentialCone()
                 end)
    return nothing
end
function noc_risks(::ScalarSum, rm, port, returns, sigma, w1, w2, w3, fees1, fees2, fees3)
    rm = reduce(vcat, rm)
    risk1 = zero(eltype(returns))
    risk2 = zero(eltype(returns))
    risk3 = zero(eltype(returns))
    for r ∈ rm
        scale = r.settings.scale
        solver_flag, sigma_flag, skew_flag, sskew_flag = set_rm_properties!(r, port.solvers,
                                                                            sigma, port.V,
                                                                            port.SV)
        risk1 += calc_risk(r, w1; X = returns, fees = fees1) * scale
        risk2 += calc_risk(r, w2; X = returns, fees = fees2) * scale
        risk3 += calc_risk(r, w3; X = returns, fees = fees3) * scale
        unset_set_rm_properties!(r, solver_flag, sigma_flag, skew_flag, sskew_flag)
    end
    return risk1, risk2, risk3
end
function noc_risks(scalarisation::ScalarLogSumExp, rm, port, returns, sigma, w1, w2, w3,
                   fees1, fees2, fees3)
    rm = reduce(vcat, rm)
    risk1 = zero(eltype(returns))
    risk2 = zero(eltype(returns))
    risk3 = zero(eltype(returns))
    gamma = scalarisation.gamma
    for r ∈ rm
        scale = r.settings.scale * gamma
        solver_flag, sigma_flag, skew_flag, sskew_flag = set_rm_properties!(r, port.solvers,
                                                                            sigma, port.V,
                                                                            port.SV)
        risk1 += exp(calc_risk(r, w1; X = returns, fees = fees1) * scale)
        risk2 += exp(calc_risk(r, w2; X = returns, fees = fees2) * scale)
        risk3 += exp(calc_risk(r, w3; X = returns, fees = fees3) * scale)
        unset_set_rm_properties!(r, solver_flag, sigma_flag, skew_flag, sskew_flag)
    end
    risk1 = log(risk1) / gamma
    risk2 = log(risk2) / gamma
    risk3 = log(risk3) / gamma
    return risk1, risk2, risk3
end
function noc_risks(::ScalarMax, rm, port, returns, sigma, w1, w2, w3, fees1, fees2, fees3)
    rm = reduce(vcat, rm)
    risk1 = -Inf
    risk2 = -Inf
    risk3 = -Inf
    for r ∈ rm
        scale = r.settings.scale
        solver_flag, sigma_flag, skew_flag, sskew_flag = set_rm_properties!(r, port.solvers,
                                                                            sigma, port.V,
                                                                            port.SV)
        risk1_i = calc_risk(r, w1; X = returns, fees = fees1) * scale
        risk2_i = calc_risk(r, w2; X = returns, fees = fees2) * scale
        risk3_i = calc_risk(r, w3; X = returns, fees = fees3) * scale
        if risk1_i >= risk1
            risk1 = risk1_i
        end
        if risk2_i >= risk2
            risk2 = risk2_i
        end
        if risk3_i >= risk3
            risk3 = risk3_i
        end
        unset_set_rm_properties!(r, solver_flag, sigma_flag, skew_flag, sskew_flag)
    end
    return risk1, risk2, risk3
end
function noc_risk_ret(port::Portfolio, type)
    (; bins, w_opt, w_min, w_max, w_opt_ini, w_min_ini, w_max_ini, rm, obj, kelly, class, custom_constr, custom_obj, scalarisation) = type

    mu, sigma, returns = mu_sigma_returns_class(port, class)
    w1 = if isempty(w_min)
        _w_min = optimise!(port,
                           Trad(; rm = rm, obj = MinRisk(), kelly = kelly, class = class,
                                w_ini = w_min_ini, custom_constr = custom_constr,
                                custom_obj = custom_obj, scalarisation = scalarisation))
        !isempty(_w_min) ? _w_min.weights : Vector{eltype(returns)}(undef, 0)
    else
        w_min
    end

    w2 = if isempty(w_max)
        _w_max = optimise!(port,
                           Trad(; rm = rm, obj = MaxRet(), kelly = kelly, class = class,
                                w_ini = w_max_ini, custom_constr = custom_constr,
                                custom_obj = custom_obj, scalarisation = scalarisation))
        !isempty(_w_max) ? _w_max.weights : Vector{eltype(returns)}(undef, 0)
    else
        w_max
    end

    w3 = if isempty(w_opt)
        _w_opt = optimise!(port,
                           Trad(; rm = rm, obj = obj, kelly = kelly, class = class,
                                w_ini = w_opt_ini, custom_constr = custom_constr,
                                custom_obj = custom_obj, scalarisation = scalarisation))
        !isempty(_w_opt) ? _w_opt.weights : Vector{eltype(returns)}(undef, 0)
    else
        w_opt
    end

    fees1 = calc_fees(w1, port.long_fees, port.short_fees, port.rebalance)
    fees2 = calc_fees(w2, port.long_fees, port.short_fees, port.rebalance)
    fees3 = calc_fees(w3, port.long_fees, port.short_fees, port.rebalance)

    if isa(kelly, NoKelly)
        ret1 = dot(mu, w1) - fees1
        ret2 = dot(mu, w2) - fees2
        ret3 = dot(mu, w3) - fees3
    else
        ret1 = sum(log.(one(eltype(mu)) .+ returns * w1)) / size(returns, 1) - fees1
        ret2 = sum(log.(one(eltype(mu)) .+ returns * w2)) / size(returns, 1) - fees2
        ret3 = sum(log.(one(eltype(mu)) .+ returns * w3)) / size(returns, 1) - fees3
    end

    risk1, risk2, risk3 = noc_risks(scalarisation, rm, port, returns, sigma, w1, w2, w3,
                                    fees1, fees2, fees3)

    d_ret = (ret2 - ret1) / bins
    d_risk = (risk2 - risk1) / bins

    ret3 -= d_ret
    risk3 += d_risk

    return w3, risk3, ret3
end
