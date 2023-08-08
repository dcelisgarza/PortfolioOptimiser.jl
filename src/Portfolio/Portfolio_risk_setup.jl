const RiskMeasures = (
    :mv,    # _mv
    :mad,   # _mad
    :msv,   # _mad
    :flpm,  # _lpm
    :slpm,  # _lpm
    :wr,    # _wr
    :cvar,  # _var
    :evar,  # _var
    :rvar,  # _var
    :mdd,   # _dar
    :add,   # _dar
    :cdar,  # _dar
    :uci,   # _dar
    :edar,  # _dar
    :rdar,  # _dar
    :krt,   # _krt
    :skrt,  # _krt
    :gmd,   # _owa
    :rg,    # _owa
    :rcvar, # _owa
    :tg,    # _owa
    :rtg,   # _owa
    :owa,   # _owa
)
function _calc_var_dar_constants(portfolio, rm, T)
    !(
        rm == :cvar ||
        rm == :evar ||
        rm == :rvar ||
        rm == :cdar ||
        rm == :edar ||
        rm == :rdar
    ) && (return nothing)

    alpha = portfolio.alpha
    portfolio.at = alpha * T
    portfolio.invat = 1 / portfolio.at

    !(rm == :rvar || rm == :rdar) && (return nothing)

    kappa = portfolio.kappa
    invat = portfolio.invat
    portfolio.ln_k = (invat^kappa - invat^(-kappa)) / (2 * kappa)
    portfolio.opk = 1 + kappa
    portfolio.omk = 1 - kappa
    portfolio.invkappa2 = 1 / (2 * kappa)
    portfolio.invk = 1 / kappa
    portfolio.invopk = 1 / portfolio.opk
    portfolio.invomk = 1 / portfolio.omk

    return nothing
end

function _mv_setup(portfolio, sigma, rm, kelly, obj, type)
    dev_u = portfolio.dev_u

    !(rm == :mv || kelly == :approx || isfinite(dev_u)) && (return nothing)

    model = portfolio.model

    @variable(model, dev)
    G = sqrt(sigma)
    @constraint(model, [dev; G * model[:w]] in SecondOrderCone())
    @expression(model, dev_risk, dev * dev)

    if isfinite(dev_u) && type == :trad
        if obj == :sharpe
            @constraint(model, dev <= dev_u * model[:k])
        else
            @constraint(model, dev <= dev_u)
        end
    end

    if rm == :mv && type != :rrp
        @expression(model, risk, dev_risk)
    end

    return nothing
end

function _mad_setup(portfolio, rm, T, returns, mu, obj, type)
    mad_u = portfolio.mad_u
    sdev_u = portfolio.sdev_u

    !(rm == :mad || rm == :msv || isfinite(mad_u) || isfinite(sdev_u)) && (return nothing)

    model = portfolio.model
    msv_target = portfolio.msv_target

    abs_dev = if isempty(msv_target) || (isa(msv_target, Real) && isinf(msv_target))
        returns .- transpose(mu)
    elseif isa(msv_target, Real) && isfinite(msv_target)
        returns .- msv_target
    else
        returns .- transpose(msv_target)
    end

    @variable(model, mad[1:T] >= 0)
    @constraint(model, abs_dev * model[:w] .>= -mad)

    if rm == :mad || isfinite(mad_u)
        @expression(model, mad_risk, sum(mad) / T)

        if isfinite(mad_u) && type == :trad
            if obj == :sharpe
                @constraint(model, mad_risk <= 0.5 * mad_u * model[:k])
            else
                @constraint(model, mad_risk <= 0.5 * mad_u)
            end
        end

        if rm == :mad
            @expression(model, risk, mad_risk)
        end
    end

    !(rm == :msv || isfinite(sdev_u)) && (return nothing)

    @variable(model, sdev)
    @constraint(model, [sdev; mad] in SecondOrderCone())

    @expression(model, sdev_risk, sdev / sqrt(T - 1))

    if isfinite(sdev_u) && type == :trad
        if obj == :sharpe
            @constraint(model, sdev_risk <= sdev_u * model[:k])
        else
            @constraint(model, sdev_risk <= sdev_u)
        end
    end

    if rm == :msv
        @expression(model, risk, sdev_risk)
    end

    return nothing
end

function _lpm_setup(portfolio, rm, T, returns, obj, rf, type)
    flpm_u = portfolio.flpm_u
    slpm_u = portfolio.slpm_u

    !(rm == :flpm || rm == :slpm || isfinite(flpm_u) || isfinite(slpm_u)) &&
        (return nothing)

    model = portfolio.model

    lpm_target = portfolio.lpm_target

    lpm_t = if isempty(lpm_target) || (isa(lpm_target, Real) && isinf(lpm_target))
        rf
    elseif isa(lpm_target, Real) && isfinite(lpm_target)
        lpm_target
    else
        transpose(lpm_target)
    end

    @variable(model, lpm[1:T] .>= 0)
    !haskey(model, :hist_ret) && (@expression(model, hist_ret, returns * model[:w]))

    if obj == :sharpe || type == :rp
        @constraint(model, lpm .>= lpm_t * model[:k] .- model[:hist_ret])
    else
        @constraint(model, lpm .>= lpm_t .- model[:hist_ret])
    end

    if rm == :flpm || isfinite(flpm_u)
        @expression(model, flpm_risk, sum(lpm) / T)

        if isfinite(flpm_u) && type == :trad
            if obj == :sharpe
                @constraint(model, flpm_risk <= flpm_u * model[:k])
            else
                @constraint(model, flpm_risk <= flpm_u)
            end
        end

        if rm == :flpm
            @expression(model, risk, flpm_risk)
        end
    end

    !(rm == :slpm || isfinite(slpm_u)) && (return nothing)

    @variable(model, slpm)
    @constraint(model, [slpm; lpm] in SecondOrderCone())
    @expression(model, slpm_risk, slpm / sqrt(T - 1))

    if isfinite(slpm_u) && type == :trad
        if obj == :sharpe
            @constraint(model, slpm_risk <= slpm_u * model[:k])
        else
            @constraint(model, slpm_risk <= slpm_u)
        end
    end

    if rm == :slpm
        @expression(model, risk, slpm_risk)
    end

    return nothing
end

function _wr_setup(portfolio, rm, returns, obj, type)
    wr_u = portfolio.wr_u

    !(rm == :wr || isfinite(wr_u)) && (return nothing)

    model = portfolio.model

    @variable(model, wr)
    !haskey(model, :hist_ret) && (@expression(model, hist_ret, returns * model[:w]))
    @constraint(model, -model[:hist_ret] .<= wr)
    @expression(model, wr_risk, wr)

    if isfinite(wr_u) && type == :trad
        if obj == :sharpe
            @constraint(model, -model[:hist_ret] .<= wr_u * model[:k])
        else
            @constraint(model, -model[:hist_ret] .<= wr_u)
        end
    end

    if rm == :wr
        @expression(model, risk, wr_risk)
    end

    return nothing
end

function _var_setup(portfolio, rm, T, returns, obj, type)
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

    model = portfolio.model

    !haskey(model, :hist_ret) && (@expression(model, hist_ret, returns * model[:w]))

    if rm == :cvar || isfinite(cvar_u)
        invat = portfolio.invat
        @variable(model, var)
        @variable(model, z_var[1:T] >= 0)
        @constraint(model, z_var .>= -model[:hist_ret] .- var)
        @expression(model, cvar_risk, var + sum(z_var) * invat)

        if isfinite(cvar_u) && type == :trad
            if obj == :sharpe
                @constraint(model, cvar_risk <= cvar_u * model[:k])
            else
                @constraint(model, cvar_risk <= cvar_u)
            end
        end

        if rm == :cvar
            @expression(model, risk, cvar_risk)
        end
    end

    if rm == :evar || isfinite(evar_u)
        at = portfolio.at
        @variable(model, t_evar)
        @variable(model, z_evar >= 0)
        @variable(model, u_evar[1:T])
        @constraint(model, sum(u_evar) <= z_evar)
        @constraint(
            model,
            [i = 1:T],
            [-model[:hist_ret][i] - t_evar, z_evar, u_evar[i]] in MOI.ExponentialCone()
        )
        @expression(model, evar_risk, t_evar - z_evar * log(at))

        if isfinite(evar_u) && type == :trad
            if obj == :sharpe
                @constraint(model, evar_risk <= evar_u * model[:k])
            else
                @constraint(model, evar_risk <= evar_u)
            end
        end

        if rm == :evar
            @expression(model, risk, evar_risk)
        end
    end

    !(rm == :rvar || isfinite(rvar_u)) && (return nothing)

    ln_k = portfolio.ln_k
    opk = portfolio.opk
    omk = portfolio.omk
    invkappa2 = portfolio.invkappa2
    invk = portfolio.invk
    invopk = portfolio.invopk
    invomk = portfolio.invomk

    @variable(model, t_rvar)
    @variable(model, z_rvar >= 0)
    @variable(model, omega_rvar[1:T])
    @variable(model, psi_rvar[1:T])
    @variable(model, theta_rvar[1:T])
    @variable(model, epsilon_rvar[1:T])
    @constraint(
        model,
        [i = 1:T],
        [z_rvar * opk * invkappa2, psi_rvar[i] * opk * invk, epsilon_rvar[i]] in
        MOI.PowerCone(invopk)
    )
    @constraint(
        model,
        [i = 1:T],
        [omega_rvar[i] * invomk, theta_rvar[i] * invk, -z_rvar * invkappa2] in
        MOI.PowerCone(omk)
    )
    @constraint(model, -model[:hist_ret] .- t_rvar .+ epsilon_rvar .+ omega_rvar .<= 0)
    @expression(model, rvar_risk, t_rvar + ln_k * z_rvar + sum(psi_rvar .+ theta_rvar))

    if isfinite(rvar_u) && type == :trad
        if obj == :sharpe
            @constraint(model, rvar_risk <= rvar_u * model[:k])
        else
            @constraint(model, rvar_risk <= rvar_u)
        end
    end

    if rm == :rvar
        @expression(model, risk, rvar_risk)
    end

    return nothing
end

function _drawdown_setup(portfolio, rm, T, returns, obj, type)
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

    @variable(model, dd[1:(T + 1)])
    !haskey(model, :hist_ret) && (@expression(model, hist_ret, returns * model[:w]))
    @constraint(model, dd[2:end] .>= dd[1:(end - 1)] .- model[:hist_ret])
    @constraint(model, dd[2:end] .>= 0)
    @constraint(model, dd[1] == 0)

    if rm == :mdd || isfinite(mdd_u)
        @variable(model, mdd)
        @constraint(model, mdd .>= dd[2:end])
        @expression(model, mdd_risk, mdd)

        if isfinite(mdd_u) && type == :trad
            if obj == :sharpe
                @constraint(model, dd[2:end] .<= mdd_u * model[:k])
            else
                @constraint(model, dd[2:end] .<= mdd_u)
            end
        end

        if rm == :mdd
            @expression(model, risk, mdd_risk)
        end
    end

    if rm == :add || isfinite(add_u)
        @expression(model, add_risk, sum(dd[2:end]) / T)

        if isfinite(add_u) && type == :trad
            if obj == :sharpe
                @constraint(model, add_risk .<= add_u * model[:k])
            else
                @constraint(model, add_risk .<= add_u)
            end
        end

        if rm == :add
            @expression(model, risk, add_risk)
        end
    end

    if rm == :cdar || isfinite(cdar_u)
        invat = portfolio.invat
        @variable(model, dar)
        @variable(model, z_dar[1:T] .>= 0)
        @constraint(model, z_dar .>= dd[2:end] .- dar)
        @expression(model, cdar_risk, dar + sum(z_dar) * invat)

        if isfinite(cdar_u) && type == :trad
            if obj == :sharpe
                @constraint(model, cdar_risk .<= cdar_u * model[:k])
            else
                @constraint(model, cdar_risk .<= cdar_u)
            end
        end

        if rm == :cdar
            @expression(model, risk, cdar_risk)
        end
    end

    if rm == :uci || isfinite(uci_u)
        @variable(model, uci)
        @constraint(model, [uci; dd[2:end]] in SecondOrderCone())
        @expression(model, uci_risk, uci / sqrt(T))

        if isfinite(uci_u) && type == :trad
            if obj == :sharpe
                @constraint(model, uci_risk <= uci_u * model[:k])
            else
                @constraint(model, uci_risk <= uci_u)
            end
        end

        if rm == :uci
            @expression(model, risk, uci_risk)
        end
    end

    if rm == :edar || isfinite(edar_u)
        at = portfolio.at
        @variable(model, t_edar)
        @variable(model, z_edar >= 0)
        @variable(model, u_edar[1:T])
        @constraint(model, sum(u_edar) <= z_edar)
        @constraint(
            model,
            [i = 1:T],
            [dd[i + 1] - t_edar, z_edar, u_edar[i]] in MOI.ExponentialCone()
        )
        @expression(model, edar_risk, t_edar - z_edar * log(at))

        if isfinite(edar_u)
            if obj == :sharpe
                @constraint(model, edar_risk <= edar_u * model[:k])
            else
                @constraint(model, edar_risk <= edar_u)
            end
        end

        if rm == :edar
            @expression(model, risk, edar_risk)
        end
    end

    !(rm == :rdar || isfinite(rdar_u)) && (return nothing)

    ln_k = portfolio.ln_k
    opk = portfolio.opk
    omk = portfolio.omk
    invkappa2 = portfolio.invkappa2
    invk = portfolio.invk
    invopk = portfolio.invopk
    invomk = portfolio.invomk

    @variable(model, t_rdar)
    @variable(model, z_rdar >= 0)
    @variable(model, omega_rdar[1:T])
    @variable(model, psi_rdar[1:T])
    @variable(model, theta_rdar[1:T])
    @variable(model, epsilon_rdar[1:T])
    @constraint(
        model,
        [i = 1:T],
        [z_rdar * opk * invkappa2, psi_rdar[i] * opk * invk, epsilon_rdar[i]] in
        MOI.PowerCone(invopk)
    )
    @constraint(
        model,
        [i = 1:T],
        [omega_rdar[i] * invomk, theta_rdar[i] * invk, -z_rdar * invkappa2] in
        MOI.PowerCone(omk)
    )
    @constraint(model, dd[2:end] .- t_rdar .+ epsilon_rdar .+ omega_rdar .<= 0)
    @expression(model, rdar_risk, t_rdar + ln_k * z_rdar + sum(psi_rdar .+ theta_rdar))

    if isfinite(rdar_u) && type == :trad
        if obj == :sharpe
            @constraint(model, rdar_risk <= rdar_u * model[:k])
        else
            @constraint(model, rdar_risk <= rdar_u)
        end
    end

    if rm == :rdar
        @expression(model, risk, rdar_risk)
    end

    return nothing
end

function _kurtosis_setup(portfolio, kurtosis, skurtosis, rm, N, obj, type)
    krt_u = portfolio.krt_u
    skrt_u = portfolio.skrt_u

    !(rm == :krt || rm == :skrt || isfinite(krt_u) || isfinite(skrt_u)) && (return nothing)

    model = portfolio.model

    if rm == :krt || isfinite(krt_u)
        max_num_assets_kurt = portfolio.max_num_assets_kurt
        @variable(model, W[1:N, 1:N], Symmetric)
        @expression(model, M1, vcat(W, transpose(model[:w])))
        if obj == :sharpe
            @expression(model, M2, vcat(model[:w], model[:k]))
        else
            @expression(model, M2, vcat(model[:w], 1))
        end
        @expression(model, M3, hcat(M1, M2))
        @constraint(model, M3 in PSDCone())

        @variable(model, t_kurt)
        if !iszero(max_num_assets_kurt) && N > max_num_assets_kurt
            N2 = 2 * N
            @variable(model, x_kurt[1:N2])
            @variable(model, r_kurt[1:N2])
            @constraint(
                model,
                [i = 1:N2],
                [r_kurt[i], t_kurt, x_kurt[i]] in MOI.PowerCone(0.5)
            )
            @constraint(model, sum(r_kurt) == t_kurt)

            A = block_vec_pq(kurtosis, N, N)
            vals_A, vecs_A = eigen(A)
            vals_A = clamp.(real.(vals_A), 0, Inf) .+ clamp.(imag.(vals_A), 0, Inf)im
            Bi = Vector{Matrix{eltype(kurtosis)}}(undef, N2)
            for i in 1:N2
                B = reshape(real.(complex(sqrt(vals_A[i])) * vecs_A[:, i]), N, N)
                Bi[i] = B
            end
            @constraint(model, [i = 1:N2], x_kurt[i] == tr(Bi[i] * W))
        else
            L_2 = portfolio.L_2
            S_2 = portfolio.S_2
            sqrt_sigma_4 = sqrt(S_2 * kurtosis * transpose(S_2))
            @expression(model, zkurt, L_2 * vec(W))
            @constraint(model, [t_kurt; sqrt_sigma_4 * zkurt] in SecondOrderCone())
        end
        @expression(model, kurt_risk, t_kurt)

        if isfinite(krt_u) && type == :trad
            if obj == :sharpe
                @constraint(model, kurt_risk <= krt_u * model[:k])
            else
                @constraint(model, kurt_risk <= krt_u)
            end
        end

        if rm == :krt
            @expression(model, risk, kurt_risk)
        end
    end

    if rm == :skrt || isfinite(skrt_u)
        max_num_assets_kurt = portfolio.max_num_assets_kurt
        @variable(model, SW[1:N, 1:N], Symmetric)
        @expression(model, SM1, vcat(SW, transpose(model[:w])))
        if obj == :sharpe
            @expression(model, SM2, vcat(model[:w], model[:k]))
        else
            @expression(model, SM2, vcat(model[:w], 1))
        end
        @expression(model, SM3, hcat(SM1, SM2))
        @constraint(model, SM3 in PSDCone())

        @variable(model, t_skurt)
        if !iszero(max_num_assets_kurt) && N > max_num_assets_kurt
            N2 = 2 * N
            @variable(model, x_skurt[1:N2])
            @variable(model, r_skurt[1:N2])
            @constraint(
                model,
                [i = 1:N2],
                [r_skurt[i], t_skurt, x_skurt[i]] in MOI.PowerCone(0.5)
            )
            @constraint(model, sum(r_skurt) == t_skurt)

            A = block_vec_pq(skurtosis, N, N)
            vals_A, vecs_A = eigen(A)
            vals_A = clamp.(real.(vals_A), 0, Inf) .+ clamp.(imag.(vals_A), 0, Inf)im
            SBi = Vector{Matrix{eltype(skurtosis)}}(undef, N2)
            for i in 1:N2
                B = reshape(real.(sqrt(complex(vals_A[i])) * vecs_A[:, i]), N, N)
                SBi[i] = B
            end
            @constraint(model, [i = 1:N2], x_skurt[i] == tr(SBi[i] * SW))
        else
            L_2 = portfolio.L_2
            S_2 = portfolio.S_2
            sqrt_sigma_4 = sqrt(S_2 * skurtosis * transpose(S_2))
            @expression(model, zskurt, L_2 * vec(SW))
            @constraint(model, [t_skurt; sqrt_sigma_4 * zskurt] in SecondOrderCone())
        end
        @expression(model, skurt_risk, t_skurt)

        if isfinite(skrt_u) && type == :trad
            if obj == :sharpe
                @constraint(model, skurt_risk <= skrt_u * model[:k])
            else
                @constraint(model, skurt_risk <= skrt_u)
            end
        end

        if rm == :skrt
            @expression(model, risk, skurt_risk)
        end
    end
end

function _owa_setup(portfolio, rm, T, returns, obj, type)
    gmd_u = portfolio.gmd_u
    rg_u = portfolio.rg_u
    tg_u = portfolio.tg_u
    rcvar_u = portfolio.rcvar_u
    rtg_u = portfolio.rtg_u
    owa_u = portfolio.owa_u

    !(
        rm == :gmd ||
        rm == :rg ||
        rm == :tg ||
        rm == :rcvar ||
        rm == :rtg ||
        rm == :owa ||
        isfinite(gmd_u) ||
        isfinite(tg_u) ||
        isfinite(rg_u) ||
        isfinite(rcvar_u) ||
        isfinite(rtg_u) ||
        isfinite(owa_u)
    ) && (return nothing)

    onesvec = range(1, stop = 1, length = T)
    model = portfolio.model

    !haskey(model, :hist_ret) && (@expression(model, hist_ret, returns * model[:w]))
    @variable(model, owa[1:T])
    @constraint(model, model[:hist_ret] == owa)

    if rm == :gmd || isfinite(gmd_u)
        @variable(model, gmda[1:T])
        @variable(model, gmdb[1:T])
        @expression(model, gmd_risk, sum(gmda .+ gmdb))
        gmd_w = owa_gmd(T) / 2
        @constraint(
            model,
            owa * transpose(gmd_w) .<=
            onesvec * transpose(gmda) + gmdb * transpose(onesvec)
        )

        if isfinite(gmd_u) && type == :trad
            if obj == :sharpe
                @constraint(model, gmd_risk <= gmd_u * model[:k] * 0.5)
            else
                @constraint(model, gmd_risk <= gmd_u * 0.5)
            end
        end

        if rm == :gmd
            @expression(model, risk, gmd_risk)
        end
    end

    if rm == :rg || isfinite(rg_u)
        @variable(model, rga[1:T])
        @variable(model, rgb[1:T])
        @expression(model, rg_risk, sum(rga .+ rgb))
        rg_w = owa_rg(T)
        @constraint(
            model,
            owa * transpose(rg_w) .<= onesvec * transpose(rga) + rgb * transpose(onesvec)
        )

        if isfinite(rg_u) && type == :trad
            if obj == :sharpe
                @constraint(model, rg_risk <= rg_u * model[:k])
            else
                @constraint(model, rg_risk <= rg_u)
            end
        end

        if rm == :rg
            @expression(model, risk, rg_risk)
        end
    end

    if rm == :rcvar || isfinite(rcvar_u)
        alpha = portfolio.alpha
        beta = portfolio.beta
        isinf(beta) && (beta = alpha)

        @variable(model, rcvara[1:T])
        @variable(model, rcvarb[1:T])
        @expression(model, rcvar_risk, sum(rcvara .+ rcvarb))
        rcvar_w = owa_rcvar(T; alpha = alpha, beta = beta)
        @constraint(
            model,
            owa * transpose(rcvar_w) .<=
            onesvec * transpose(rcvara) + rcvarb * transpose(onesvec)
        )

        if isfinite(rcvar_u) && type == :trad
            if obj == :sharpe
                @constraint(model, rcvar_risk <= rcvar_u * model[:k])
            else
                @constraint(model, rcvar_risk <= rcvar_u)
            end
        end

        if rm == :rcvar
            @expression(model, risk, rcvar_risk)
        end
    end

    if rm == :tg || isfinite(tg_u)
        alpha = portfolio.alpha
        a_sim = portfolio.a_sim
        alpha_i = portfolio.alpha_i
        @variable(model, tga[1:T])
        @variable(model, tgb[1:T])
        @expression(model, tg_risk, sum(tga .+ tgb))
        tg_w = owa_tg(T; alpha_i = alpha_i, alpha = alpha, a_sim = a_sim)
        @constraint(
            model,
            owa * transpose(tg_w) .<= onesvec * transpose(tga) + tgb * transpose(onesvec)
        )

        if isfinite(tg_u) && type == :trad
            if obj == :sharpe
                @constraint(model, tg_risk <= tg_u * model[:k])
            else
                @constraint(model, tg_risk <= tg_u)
            end
        end

        if rm == :tg
            @expression(model, risk, tg_risk)
        end
    end

    if rm == :rtg || isfinite(rtg_u)
        alpha = portfolio.alpha
        a_sim = portfolio.a_sim
        alpha_i = portfolio.alpha_i
        beta = portfolio.beta
        b_sim = portfolio.b_sim
        beta_i = portfolio.beta_i

        isinf(beta) && (beta = alpha)
        b_sim < 0 && (b_sim = a_sim)
        isinf(beta_i) && (beta_i = alpha_i)

        @variable(model, rtga[1:T])
        @variable(model, rtgb[1:T])
        @expression(model, rtg_risk, sum(rtga .+ rtgb))
        rtg_w = owa_rtg(
            T;
            alpha_i = alpha_i,
            alpha = alpha,
            a_sim = a_sim,
            beta_i = beta_i,
            beta = beta,
            b_sim = b_sim,
        )
        @constraint(
            model,
            owa * transpose(rtg_w) .<=
            onesvec * transpose(rtga) + rtgb * transpose(onesvec)
        )

        if isfinite(rtg_u) && type == :trad
            if obj == :sharpe
                @constraint(model, rtg_risk <= rtg_u * model[:k])
            else
                @constraint(model, rtg_risk <= rtg_u)
            end
        end

        if rm == :rtg
            @expression(model, risk, rtg_risk)
        end
    end

    !(rm == :owa || isfinite(owa_u)) && (return nothing)

    @variable(model, owa_a[1:T])
    @variable(model, owa_b[1:T])
    @expression(model, owa_risk, sum(owa_a .+ owa_b))

    owa_w = portfolio.owa_w
    if isempty(owa_w)
        owa_w = owa_gmd(T) / 2
    elseif isa(owa_w, Vector)
        owa_w = portfolio.owa_w
    else
        owa_w = fill(1 / T, T)
    end

    @constraint(
        model,
        owa * transpose(owa_w) .<= onesvec * transpose(owa_a) + owa_b * transpose(onesvec)
    )

    if isfinite(owa_u) && type == :trad
        if obj == :sharpe
            @constraint(model, owa_risk <= owa_u * model[:k])
        else
            @constraint(model, owa_risk <= owa_u)
        end
    end

    if rm == :owa
        @expression(model, risk, owa_risk)
    end

    return nothing
end

function _rp_setup(portfolio, N)
    model = portfolio.model
    rb = portfolio.risk_budget
    @variable(model, log_w[1:N])
    @constraint(model, dot(rb, log_w) >= 1)
    @constraint(model, [i = 1:N], [log_w[i], 1, model[:w][i]] in MOI.ExponentialCone())
    @constraint(model, model[:w] .>= 0)
    @constraint(model, sum(model[:w]) == model[:k])
end

function _rrp_setup(portfolio, sigma, N, rrp_ver, rrp_penalty)
    G = sqrt(sigma)
    model = portfolio.model
    rb = portfolio.risk_budget

    @variable(model, psi)
    @variable(model, gamma >= 0)
    @variable(model, zeta[1:N] .>= 0)
    @expression(model, risk, psi - gamma)
    # RRP constraints.
    @constraint(model, zeta .== sigma * model[:w])
    @constraint(model, sum(model[:w]) == 1)
    @constraint(model, model[:w] >= 0)
    @constraint(
        model,
        [i = 1:N],
        [
            model[:w][i] + zeta[i]
            2 * gamma * sqrt(rb[i])
            model[:w][i] - zeta[i]
        ] in SecondOrderCone()
    )
    # RRP version constraints
    if rrp_ver == :reg || rrp_ver == :reg_pen
        @variable(model, rho)
        @constraint(model, [2 * psi; 2 * G * model[:w]; -2 * rho] in SecondOrderCone())
    end

    if rrp_ver == :none
        @constraint(model, [psi; G * model[:w]] in SecondOrderCone())
    elseif rrp_ver == :reg
        @constraint(model, [rho; G * model[:w]] in SecondOrderCone())
    elseif rrp_ver == :reg_pen
        theta = Diagonal(sqrt.(diag(sigma)))
        @constraint(
            model,
            [rho; sqrt(rrp_penalty) * theta * model[:w]] in SecondOrderCone()
        )
    end

    return nothing
end

function _wc_setup(portfolio, obj, N, rf, mu, sigma, u_mu, u_cov)
    obj == :min_risk && isnothing(mu) && return nothing

    model = portfolio.model

    # Return uncertainy sets.
    @expression(model, _ret, dot(mu, model[:w]))
    if u_mu == :box
        d_mu = portfolio.d_mu[!, :val]
        @variable(model, abs_w[1:N])
        @constraint(model, [i = 1:N], [abs_w[i]; model[:w][i]] in MOI.NormOneCone(2))
        @expression(model, ret, _ret - dot(d_mu, abs_w))
        if obj == :sharpe
            @constraint(model, ret - rf * model[:k] >= 1)
        end
    elseif u_mu == :ellipse
        k_mu = portfolio.k_mu
        cov_mu = portfolio.cov_mu
        G = sqrt(cov_mu)
        @expression(model, x_gw, G * model[:w])
        @variable(model, t_gw)
        @constraint(model, [t_gw; x_gw] in SecondOrderCone())
        @expression(model, ret, _ret - k_mu * t_gw)
        if obj == :sharpe
            @constraint(model, ret - rf * model[:k] >= 1)
        end
    else
        @expression(model, ret, _ret)
        if obj == :sharpe
            @constraint(model, ret - rf * model[:k] >= 1)
        end
    end

    # Cov uncertainty sets.
    if u_cov == :box
        cov_u = portfolio.cov_u
        cov_l = portfolio.cov_l
        @variable(model, Au[1:N, 1:N] .>= 0, Symmetric)
        @variable(model, Al[1:N, 1:N] .>= 0, Symmetric)
        @expression(model, M1, vcat(Au - Al, transpose(model[:w])))
        if obj == :sharpe
            @expression(model, M2, vcat(model[:w], model[:k]))
        else
            @expression(model, M2, vcat(model[:w], 1))
        end
        @expression(model, M3, hcat(M1, M2))
        @constraint(model, M3 in PSDCone())
        @expression(model, risk, tr(Au * cov_u) - tr(Al * cov_l))
    elseif u_cov == :ellipse
        k_sigma = portfolio.k_sigma
        G_sigma = sqrt(portfolio.cov_sigma)
        @variable(model, E1[1:N, 1:N], Symmetric)
        @variable(model, E2[1:N, 1:N], Symmetric)
        @expression(M1, vcat(E1, transpose(model[:w])))
        if obj == :sharpe
            @expression(M2, vcat(model[:w], model[:k]))
        else
            @expression(M2, vcat(model[:w], 1))
        end
        @expression(M3, hcat(M1, M2))
        @constraint(model, M3 in PSDCone())
        @constraint(model, E2 in PSDCone())
        @expression(model, x_ge, G_sigma * vec(E1 .+ E2))
        @variable(model, t_ge)
        @constraint(model, [t_ge; x_gw] in SecondOrderCone())
        @expression(model, risk, tr(sigma * (E1 .+ E2)) + k_sigma * t_ge)
    else
        @variable(model, dev)
        G = sqrt(sigma)
        @constraint(model, [dev; G * model[:w]] in SecondOrderCone())
        @expression(model, risk, dev * dev)
    end
end

const HRRiskMeasures = (
    :msd,
    RiskMeasures...,
    :equal,
    :var,
    :dar,
    :mdd_r,
    :add_r,
    :dar_r,
    :cdar_r,
    :edar_r,
    :rdar_r,
)
function _naive_risk(portfolio, returns, covariance; rm = :mv, rf = 0.0)
    N = size(returns, 2)
    if rm == :equal
        weights = fill(1 / N, N)
    else
        inv_risk = zeros(N)
        w = zeros(N)
        for i in 1:N
            fill!(w, 0)
            w[i] = 1
            risk = calc_risk(
                w,
                returns,
                covariance;
                rm = rm,
                rf = rf,
                alpha = portfolio.alpha,
                a_sim = portfolio.a_sim,
                beta = portfolio.beta,
                b_sim = portfolio.b_sim,
                kappa = portfolio.kappa,
                solvers = portfolio.solvers,
            )
            inv_risk[i] = 1 / risk
        end
        weights = inv_risk / sum(inv_risk)
    end

    return weights
end

function _opt_w(
    portfolio,
    assets,
    returns,
    mu,
    cov,
    obj = :min_risk,
    rm = :mv,
    rf = 0.0,
    l = 2.0,
)
    port = Portfolio(assets = assets, ret = returns, solvers = portfolio.solvers)
    asset_statistics!(port; calc_kurt = false)
    port.cov = cov

    weights = if obj ∈ (:min_risk, :utility, :sharpe)
        !isnothing(mu) && (port.mu = mu)
        opt_port!(port; type = :trad, class = :classic, rm = rm, obj = obj, rf = rf, l = l)
    elseif obj == :erc
        opt_port!(port; type = :rp, class = :classic, rm = rm, rf = rf, l = l)
    end

    return weights[!, :weights]
end

function _hierarchical_clustering(
    portfolio::HCPortfolio,
    model = :hrp,
    linkage = :ward,
    codependence = :pearson,
    max_k = 10,
    branchorder = :optimal,
)
    @assert(codependence ∈ CodepTypes, "codependence must be one of $CodepTypes")

    codep = portfolio.codep
    returns = portfolio.returns
    bins_info = portfolio.bins_info

    codeps1 = (:pearson, :spearman, :kendall, :gerber1, :gerber2, :custom)
    codeps2 = (:abs_pearson, :abs_spearman, :abs_kendall, :distance)

    dist = if codependence ∈ codeps1
        sqrt.(clamp!((1 .- codep) / 2, 0, 1))
    elseif codependence ∈ codeps2
        sqrt.(clamp!(1 .- codep, 0, 1))
    elseif codependence == :mutual_info
        info_mtx(returns, bins_info, :variation)
    elseif codependence == :tail
        -log.(codep)
    end

    dist = issymmetric(dist) ? dist : Symmetric(dist)
    codep = issymmetric(codep) ? codep : Symmetric(codep)

    if linkage == :dbht
        codep = codependence ∈ codeps1 ? 1 - dist .^ 2 : codep
        missing, missing, missing, missing, missing, missing, clustering =
            DBHTs(dist, codep, branchorder)
    else
        clustering = hclust(
            dist;
            linkage = linkage,
            branchorder = branchorder == :default ? :r : branchorder,
        )
    end

    if model ∈ (:herc, :herc2, :nco)
        k = two_diff_gap_stat(dist, clustering, max_k)
    else
        k = nothing
    end

    return clustering, k
end

function leaves_list(clustering)
    merges = transpose(clustering.merges)
    idx = findall(x -> x < 0, merges)
    leaves = merges[idx]
    return leaves
end

function _cluster_risk(portfolio, returns, covariance, cluster; rm = :mv, rf = 0.0)
    cret = returns[:, cluster]
    ccov = covariance[cluster, cluster]
    cw = _naive_risk(portfolio, cret, ccov; rm = rm, rf = rf)
    crisk = calc_risk(
        cw,
        cret,
        ccov;
        rm = rm,
        rf = rf,
        alpha = portfolio.alpha,
        a_sim = portfolio.a_sim,
        beta = portfolio.beta,
        b_sim = portfolio.b_sim,
        kappa = portfolio.kappa,
        solvers = portfolio.solvers,
    )

    return crisk
end

function _hr_weight_bounds(upper_bound, lower_bound, weights, sort_order, lc, rc, alpha_1)
    if (any(upper_bound .< weights[sort_order]) || any(lower_bound .> weights[sort_order]))
        lmaxw = weights[lc[1]]
        a1 = sum(upper_bound[lc]) / lmaxw
        a2 = max(sum(lower_bound[lc]) / lmaxw, alpha_1)
        alpha_1 = min(a1, a2)

        rmaxw = weights[rc[1]]
        a1 = sum(upper_bound[rc]) / rmaxw
        a2 = max(sum(lower_bound[rc]) / rmaxw, 1 - alpha_1)
        alpha_1 = 1 - min(a1, a2)
    end

    return alpha_1
end

function _recursive_bisection(
    portfolio,
    sort_order;
    rm = :mv,
    rf = 0.0,
    upper_bound = nothing,
    lower_bound = nothing,
)
    N = length(portfolio.assets)
    weights = fill(1.0, N)
    items = [sort_order]
    returns = portfolio.returns
    covariance = portfolio.covariance

    while length(items) > 0
        items = [
            i[j:k] for i in items for
            (j, k) in ((1, div(length(i), 2)), (1 + div(length(i), 2), length(i))) if
            length(i) > 1
        ]

        for i in 1:2:length(items)
            lc = items[i]
            rc = items[i + 1]

            # Left cluster.
            lrisk = _cluster_risk(portfolio, returns, covariance, lc; rm = rm, rf = rf)

            # Right cluster.
            rrisk = _cluster_risk(portfolio, returns, covariance, rc; rm = rm, rf = rf)

            # Allocate weight to clusters.
            alpha_1 = 1 - lrisk / (lrisk + rrisk)

            # Weight constraints.
            alpha_1 = _hr_weight_bounds(
                upper_bound,
                lower_bound,
                weights,
                sort_order,
                lc,
                rc,
                alpha_1,
            )

            weights[lc] *= alpha_1
            weights[rc] *= 1 - alpha_1
        end
    end

    return weights
end

function _hierarchical_recursive_bisection() end

export leaves_list
