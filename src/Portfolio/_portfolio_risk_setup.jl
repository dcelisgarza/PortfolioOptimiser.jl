const RiskMeasures = (
    :mv,
    :mad,
    :msd,
    :cvar,
    :wr,
    :flpm,
    :slpm,
    :mdd,
    :add,
    :cdar,
    :uci,
    :evar,
    :edar,
    # :rdar,
    # :rvar,
)

function _mv_setup(portfolio, sigma, rm, kelly, obj)
    dev_u = portfolio.dev_u

    !(rm == :mv || kelly == :approx || isfinite(dev_u)) && (return nothing)

    model = portfolio.model

    w = model[:w]
    k = model[:k]
    @variable(model, tdev >= 0)
    @expression(model, dev_risk, tdev * tdev)
    G = sqrt(sigma)
    @constraint(model, tdev_sqrt_sigma_soc, [tdev; transpose(G) * w] in SecondOrderCone())

    if isfinite(dev_u)
        if obj == :sharpe
            @constraint(model, tdev_leq_udev, tdev <= dev_u * k)
        else
            @constraint(model, tdev_leq_udev, tdev <= dev_u)
        end
    end

    if rm == :mv
        @expression(model, risk, dev_risk)
    end

    return nothing
end

function _mad_setup(portfolio, rm, T, returns, mu, obj)
    mad_u = portfolio.mad_u
    sdev_u = portfolio.sdev_u

    !(rm == :mad || rm == :msd || isfinite(mad_u) || isfinite(sdev_u)) && (return nothing)

    model = portfolio.model
    w = model[:w]
    k = model[:k]

    @variable(model, tmad[1:T] >= 0)
    abs_dev = returns .- transpose(mu)
    @constraint(model, abs_dev_w_geq_neg_tmad, abs_dev * w >= -tmad)

    if rm == :mad || isfinite(mad_u)
        @expression(model, mad_risk, sum(tmad) / T)

        if isfinite(mad_u)
            if obj == :sharpe
                @constraint(model, mad_risk_leq_umad_div_2, mad_risk * 2 <= mad_u * k)
            else
                @constraint(model, mad_risk_leq_umad_div_2, mad_risk * 2 <= mad_u)
            end
        end

        if rm == :mad
            @expression(model, risk, mad_risk)
        end
    end

    !(rm == :msd || isfinite(sdev_u)) && (return nothing)

    @variable(model, tmsd >= 0)
    @constraint(model, tmsd_tmad_soc, [tmsd; tmad] in SecondOrderCone())
    @expression(model, msd_risk, tmsd / sqrt(T - 1))

    if isfinite(sdev_u)
        if obj == :sharpe
            @constraint(model, msd_risk_leq_umsd, msd_risk <= sdev_u * k)
        else
            @constraint(model, msd_risk_leq_umsd, msd_risk <= sdev_u)
        end
    end

    if rm == :msd
        @expression(model, risk, msd_risk)
    end

    return nothing
end

function _var_setup(portfolio, rm, T, returns, obj, ln_k)
    cvar_u = portfolio.cvar_u
    evar_u = portfolio.evar_u
    rvar_u = portfolio.rvar_u

    !(
        rm == :cvar ||
        rm == :evar ||
        rm == :rvar ||
        isfinite(evar_u) ||
        isfinite(cvar_u) ||
        isfinite(rvar_u)
    ) && (return nothing)

    alpha = portfolio.alpha
    model = portfolio.model
    w = model[:w]
    k = model[:k]

    if !haskey(model, :hist_ret)
        @expression(model, hist_ret, returns * w)
    end
    hist_ret = model[:hist_ret]

    if rm == :cvar || isfinite(cvar_u)
        @variable(model, var)
        @variable(model, z_var[1:T] >= 0)
        @constraint(model, z_var_p_hist_ret_p_var_geq_0, z_var .+ hist_ret .+ var .>= 0)
        @expression(model, cvar_risk, var + sum(z_var) / (alpha * T))

        if isfinite(cvar_u)
            if obj == :sharpe
                @constraint(model, cvar_risk_leq_ucvar, cvar_risk <= cvar_u * k)
            else
                @constraint(model, cvar_risk_leq_ucvar, cvar_risk <= cvar_u)
            end
        end

        if rm == :cvar
            @expression(model, risk, cvar_risk)
        end
    end

    if rm == :evar || isfinite(evar_u)
        @variable(model, t_evar)
        @variable(model, s_evar >= 0)
        @variable(model, u_evar[1:T])
        @constraint(model, evar1, sum(u_evar) - s_evar <= 0)
        @constraint(
            model,
            evar_expc[i = 1:T],
            [-hist_ret[i] - t_evar, s_evar, u_evar[i]] in MOI.ExponentialCone()
        )
        @expression(model, evar_risk, t_evar - s_evar * log(alpha * T))

        if isfinite(evar_u)
            if obj == :sharpe
                @constraint(model, evar_leq_uevar, evar_risk <= evar_u * k)
            else
                @constraint(model, evar_leq_uevar, evar_risk <= evar_u)
            end
        end

        if rm == :evar
            @expression(model, risk, evar_risk)
        end
    end

    !(rm == :rvar || isfinite(rvar_u)) && (return nothing)

    k = portfolio.kappa
    opk = 1 + k
    omk = 1 - k
    k2 = 2 * k
    invk = 1 / k
    @variable(model, t_rvar)
    @variable(model, s_rvar >= 0)
    @variable(model, omega_rvar[1:T])
    @variable(model, psi_rvar[1:T])
    @variable(model, theta_rvar[1:T])
    @variable(model, epsilon_rvar[1:T])
    @constraint(
        model,
        rvar1[i = 1:T],
        [s_rvar * opk / k2, psi_rvar[i] * opk * invk, epsilon_rvar[i]] in
        MOI.PowerCone(1 / opk)
    )
    @constraint(
        model,
        rvar2[i = 1:T],
        [omega_rvar[i] / omk, theta_rvar[i] * invk, -s_rvar / k2] in MOI.PowerCone(omk)
    )
    @constraint(model, rvar3, -hist_ret .- t_rvar .+ epsilon_rvar .+ omega_rvar .<= 0)
    @expression(model, rvar_risk, t_rvar + ln_k * s_rvar + sum(psi_rvar .+ theta_rvar))

    if isfinite(rvar_u)
        if obj == :sharpe
            @constraint(model, rvar_leq_urvar, rvar_risk <= rvar_u * k)
        else
            @constraint(model, rvar_leq_urvar, rvar_risk <= rvar_u * k)
        end
    end

    if rm == :rvar
        @expression(model, risk, rvar_risk)
    end

    return nothing
end

function _wr_setup(portfolio, rm, returns, obj)
    wr_u = portfolio.wr_u

    !(rm == :wr || isfinite(wr_u)) && (return nothing)

    model = portfolio.model
    w = model[:w]
    k = model[:k]

    @variable(model, twr)
    if !haskey(model, :hist_ret)
        @expression(model, hist_ret, returns * w)
    end
    hist_ret = model[:hist_ret]
    @constraint(model, twr_p_hist_ret_geq_0, hist_ret .+ twr .>= 0)
    @expression(model, wr_risk, twr)

    if isfinite(wr_u)
        if obj == :sharpe
            @constraint(model, hist_ret_p_uwr_geq_0, hist_ret .+ wr_u * k .>= 0)
        else
            @constraint(model, hist_ret_p_uwr_geq_0, hist_ret .+ wr_u .>= 0)
        end
    end

    if rm == :wr
        @expression(model, risk, wr_risk)
    end

    return nothing
end

function _lpm_setup(portfolio, rm, T, returns, obj, rf)
    flpm_u = portfolio.flpm_u
    slpm_u = portfolio.slpm_u

    !(rm == :flpm || rm == :slpm || isfinite(flpm_u) || isfinite(slpm_u)) &&
        (return nothing)

    model = portfolio.model
    w = model[:w]
    k = model[:k]

    @variable(model, tlpm[1:T] .>= 0)
    if !haskey(model, :hist_ret)
        @expression(model, hist_ret, returns * w)
    end
    hist_ret = model[:hist_ret]
    if obj == :sharpe
        @constraint(model, tlpm_p_hist_ret_geq_rf, tlpm .+ hist_ret .>= rf * k)
    else
        @constraint(model, tlpm_p_hist_ret_geq_rf, tlpm .+ hist_ret .>= rf)
    end

    if rm == :flpm || isfinite(flpm_u)
        @expression(model, flpm_risk, sum(tlpm) / T)

        if isfinite(flpm_u)
            if obj == :sharpe
                @constraint(model, flpm_risk_leq_uflpm <= flpm_u * k)
            else
                @constraint(model, flpm_risk_leq_uflpm <= flpm_u)
            end
        end

        if rm == :flpm
            @expression(model, risk, flpm_risk)
        end
    end

    !(rm == :slpm || isfinite(slpm_u)) && (return nothing)

    @constraint(model, tslpm >= 0)
    @constraint(model, tslpm_tlpm_soc, [tslpm; tlpm] in SecondOrderCone())
    @expression(model, slpm_risk, tslpm / sqrt(T - 1))

    if isfinite(slpm_u)
        if obj == :sharpe
            @constraint(model, slpm_risk_leq_uslpm <= slpm_u * k)
        else
            @constraint(model, slpm_risk_leq_uslpm <= slpm_u)
        end
    end

    if rm == :slpm
        @expression(model, risk, slpm_risk)
    end

    return nothing
end

function _drawdown_setup(portfolio, rm, T, returns, obj, ln_k)
    mdd_u = portfolio.mdd_u
    add_u = portfolio.add_u
    cdar_u = portfolio.cdar_u
    uci_u = portfolio.uci_u
    edar_u = portfolio.edar_u
    rdar_u = portfolio.rdar_u

    !(
        rm == :mdd ||
        rm == :add ||
        rm == :cdar ||
        rm == :uci ||
        rm == :edar ||
        rm == :rdar ||
        isfinite(mdd_u) ||
        isfinite(add_u) ||
        isfinite(cdar_u) ||
        isfinite(uci_u) ||
        isfinite(edar_u) ||
        isfinite(rdar_u)
    ) && (return nothing)

    model = portfolio.model
    w = model[:w]
    k = model[:k]

    @variable(model, tdd[1:(T + 1)])
    if !haskey(model, :hist_ret)
        @expression(model, hist_ret, returns * w)
    end
    hist_ret = model[:hist_ret]
    @constraint(model, tdd1_cntr, tdd[2:end] .- tdd[1:(end - 1)] .+ hist_ret .>= 0)
    @constraint(model, tdd2_cntr, tdd[2:end] .>= 0)
    @constraint(model, tdd3_cntr, tdd[1] == 0)

    if rm == :mdd || isfinite(mdd_u)
        @variable(model, tmdd)
        @constraint(model, tmdd_neg_dd_geq_0, tmdd .- tdd[2:end] .>= 0)
        @expression(model, mdd_risk, tmdd)

        if isfinite(mdd_u)
            if obj == :sharpe
                @constraint(model, tdd_leq_umdd, tdd[2:end] .<= mdd_u * k)
            else
                @constraint(model, tdd_leq_umdd, tdd[2:end] .<= mdd_u)
            end
        end

        if rm == :mdd
            @expression(model, risk, mdd_risk)
        end
    end

    if rm == :add || isfinite(add_u)
        @expression(model, add_risk, sum(tdd[2:end]) / T)

        if isfinite(mdd_u)
            if obj == :sharpe
                @constraint(model, add_risk_leq_uadd, add_risk .<= add_u * k)
            else
                @constraint(model, add_risk_leq_uadd, add_risk .<= add_u)
            end
        end

        if rm == :add
            @expression(model, risk, add_risk)
        end
    end

    if rm == :cdar || isfinite(cdar_u)
        alpha = portfolio.alpha

        @variable(model, tdar)
        @variable(model, zdar[1:T] .>= 0)
        @constraint(model, cdar_cntr, zdar .- tdd[2:end] .+ tdar .>= 0)
        @expression(model, cdar_risk, tdar + sum(zdar) / (alpha * T))

        if isfinite(mdd_u)
            if obj == :sharpe
                @constraint(model, cdar_leq_ucdar_risk, cdar_risk .<= cdar_u * k)
            else
                @constraint(model, cdar_leq_ucdar_risk, cdar_risk .<= cdar_u)
            end
        end

        if rm == :add
            @expression(model, risk, cdar_risk)
        end
    end

    if rm == :uci || isfinite(uci_u)
        @variable(model, tuci >= 0)
        @constraint(model, tuci_tdd_soc, [tuci; tdd[2:end]] in SecondOrderCone())
        @expression(model, uci_risk, tuci / sqrt(T))

        if isfinite(uci_u)
            if obj == :sharpe
                @constraint(model, uci_risk_leq_uuci, uci_risk <= uci_u * k)
            else
                @constraint(model, uci_risk_leq_uuci, uci_risk <= uci_u)
            end
        end

        if rm == :uci
            @expression(model, risk, uci_risk)
        end
    end

    if rm == :edar || isfinite(edar_u)
        @variable(model, t_edar)
        @variable(model, s_edar >= 0)
        @variable(model, u_edar[1:T])
        @constraint(model, edar1, sum(u_edar) - s_edar <= 0)
        @constraint(
            model,
            edar_expc[i = 1:T],
            [tdd[i + 1] - t_edar, s_edar, u_edar[i]] in MOI.ExponentialCone()
        )
        @expression(model, edar_risk, t_edar - s_edar * log(alpha * T))

        if isfinite(edar_u)
            if obj == :sharpe
                @constraint(model, edar_leq_uedar, edar_risk <= edar_u * k)
            else
                @constraint(model, edar_leq_uedar, edar_risk <= edar_u)
            end
        end

        if rm == :edar
            @expression(model, risk, edar_risk)
        end
    end

    !(rm == :rdar || isfinite(rdar_u)) && (return nothing)

    k = portfolio.kappa
    opk = 1 + k
    omk = 1 - k
    k2 = 2 * k
    invk = 1 / k
    @variable(model, t_rdar)
    @variable(model, s_rdar >= 0)
    @variable(model, omega_rdar[1:T])
    @variable(model, psi_rdar[1:T])
    @variable(model, theta_rdar[1:T])
    @variable(model, epsilon_rdar[1:T])
    @constraint(
        model,
        rdar1[i = 1:T],
        [s_rdar * opk / k2, psi_rdar[i] * opk * invk, epsilon_rdar[i]] in
        MOI.PowerCone(1 / opk)
    )
    @constraint(
        model,
        rdar2[i = 1:T],
        [omega_rdar[i] / omk, theta_rdar[i] * invk, -s_rdar / k2] in MOI.PowerCone(omk)
    )
    @constraint(model, rdar3, tdd[2:end] .- t_rdar .+ epsilon_rdar .+ omega_rdar .<= 0)
    @expression(model, rdar_risk, t_rdar + ln_k * s_rdar + sum(psi_rdar .+ theta_rdar))

    if isfinite(rdar_u)
        if obj == :sharpe
            @constraint(model, rdar_leq_urdar, rdar_risk <= rdar_u * k)
        else
            @constraint(model, rdar_leq_urdar, rdar_risk <= rdar_u * k)
        end
    end

    if rm == :rdar
        @expression(model, risk, rdar_risk)
    end

    return nothing
end