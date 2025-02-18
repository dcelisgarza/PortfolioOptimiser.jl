# Copywrite (c) 2025
# Author: Daniel Celis Garza <daniel.celis.garza@gmail.com>
# SPDX-License-Identifier: MIT

function noc_constraints(port::Portfolio, risk0, ret0)
    model = port.model
    w = model[:w]
    scale_constr = model[:scale_constr]
    risk = model[:risk]
    ret = model[:ret]
    long_ub = port.long_ub
    if isa(long_ub, Real)
        long_ub = range(long_ub; stop = long_ub, length = length(w))
    end
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
                      scale_constr * (long_ub[i] - w[i])] ∈ MOI.ExponentialCone()
                 end)
    return nothing
end
function noc_risks(::ScalarSum, rm, port, returns, sigma, w1, w2, w3, fees, rebalance)
    rm = reduce(vcat, rm)
    risk1 = zero(eltype(returns))
    risk2 = zero(eltype(returns))
    risk3 = zero(eltype(returns))
    for r ∈ rm
        scale = r.settings.scale
        solver_flag, sigma_flag, skew_flag, sskew_flag = set_rm_properties!(r, port.solvers,
                                                                            sigma, port.V,
                                                                            port.SV)
        risk1 += expected_risk(r, w1; X = returns, fees = fees, rebalance = rebalance) *
                 scale
        risk2 += expected_risk(r, w2; X = returns, fees = fees, rebalance = rebalance) *
                 scale
        risk3 += expected_risk(r, w3; X = returns, fees = fees, rebalance = rebalance) *
                 scale
        unset_set_rm_properties!(r, solver_flag, sigma_flag, skew_flag, sskew_flag)
    end
    return risk1, risk2, risk3
end
function noc_risks(scalarisation::ScalarLogSumExp, rm, port, returns, sigma, w1, w2, w3,
                   fees, rebalance)
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
        risk1 += exp(expected_risk(r, w1; X = returns, fees = fees, rebalance = rebalance) *
                     scale)
        risk2 += exp(expected_risk(r, w2; X = returns, fees = fees, rebalance = rebalance) *
                     scale)
        risk3 += exp(expected_risk(r, w3; X = returns, fees = fees, rebalance = rebalance) *
                     scale)
        unset_set_rm_properties!(r, solver_flag, sigma_flag, skew_flag, sskew_flag)
    end
    risk1 = log(risk1) / gamma
    risk2 = log(risk2) / gamma
    risk3 = log(risk3) / gamma
    return risk1, risk2, risk3
end
function noc_risks(::ScalarMax, rm, port, returns, sigma, w1, w2, w3, fees, rebalance)
    rm = reduce(vcat, rm)
    risk1 = -Inf
    risk2 = -Inf
    risk3 = -Inf
    for r ∈ rm
        scale = r.settings.scale
        solver_flag, sigma_flag, skew_flag, sskew_flag = set_rm_properties!(r, port.solvers,
                                                                            sigma, port.V,
                                                                            port.SV)
        risk1_i = expected_risk(r, w1; X = returns, fees = fees, rebalance = rebalance) *
                  scale
        risk2_i = expected_risk(r, w2; X = returns, fees = fees, rebalance = rebalance) *
                  scale
        risk3_i = expected_risk(r, w3; X = returns, fees = fees, rebalance = rebalance) *
                  scale
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
        old_mu_min = port.mu_l
        port.mu_l = Inf
        _w_min = optimise!(port,
                           Trad(; rm = rm, obj = MinRisk(), kelly = kelly, class = class,
                                w_ini = w_min_ini, custom_constr = custom_constr,
                                custom_obj = custom_obj, scalarisation = scalarisation))
        port.mu_l = old_mu_min
        !isempty(_w_min) ? _w_min.weights : Vector{eltype(returns)}(undef, 0)
    else
        w_min
    end

    w2 = if isempty(w_max)
        old_ubs = Vector{Pair{Int, eltype(port.returns)}}(undef, 0)
        rm_flat = reduce(vcat, rm)
        for (i, r) ∈ enumerate(rm_flat)
            ub = r.settings.ub
            if isfinite(ub)
                push!(old_ubs, Pair(i, ub))
                r.settings.ub = Inf
            end
        end
        _w_max = optimise!(port,
                           Trad(; rm = rm, obj = MaxRet(), kelly = kelly, class = class,
                                w_ini = w_max_ini, custom_constr = custom_constr,
                                custom_obj = custom_obj, scalarisation = scalarisation))
        for old_ub ∈ old_ubs
            i = old_ub.first
            ub = old_ub.second
            rm_flat[i].settings.ub = ub
        end
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

    fees = port.fees
    rebalance = port.rebalance

    ret1 = expected_ret(w1; mu = mu, X = returns, kelly = kelly, fees = fees,
                        rebalance = rebalance)
    ret2 = expected_ret(w2; mu = mu, X = returns, kelly = kelly, fees = fees,
                        rebalance = rebalance)
    ret3 = expected_ret(w3; mu = mu, X = returns, kelly = kelly, fees = fees,
                        rebalance = rebalance)

    risk1, risk2, risk3 = noc_risks(scalarisation, rm, port, returns, sigma, w1, w2, w3,
                                    fees, rebalance)

    d_ret = (ret2 - ret1) / bins
    d_risk = (risk2 - risk1) / bins

    ret3 -= d_ret
    risk3 += d_risk

    return w3, risk3, ret3
end
