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

    !(rm == :rvar || rm == :rdar) && (return nothing)

    kappa = portfolio.kappa
    invat = portfolio.invat = 1 / portfolio.at
    portfolio.ln_k = (invat^kappa - invat^(-kappa)) / (2 * kappa)
    portfolio.opk = 1 + kappa
    portfolio.omk = 1 - kappa
    portfolio.invkappa2 = 1 / (2 * kappa)
    portfolio.invk = 1 / kappa
    portfolio.invopk = 1 / portfolio.opk
    portfolio.invomk = 1 / portfolio.omk

    return nothing
end

function _mv_setup(portfolio, sigma, rm, kelly, obj)
    dev_u = portfolio.dev_u

    !(rm == :mv || kelly == :approx || isfinite(dev_u)) && (return nothing)

    model = portfolio.model

    @variable(model, dev >= 0)
    @expression(model, dev_risk, dev * dev)
    G = sqrt(sigma)
    @constraint(model, [dev; G * model[:w]] in SecondOrderCone())

    if isfinite(dev_u)
        if obj == :sharpe
            @constraint(model, dev <= dev_u * model[:k])
        else
            @constraint(model, dev <= dev_u)
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

    !(rm == :mad || rm == :msv || isfinite(mad_u) || isfinite(sdev_u)) && (return nothing)

    model = portfolio.model

    @variable(model, mad[1:T] >= 0)
    abs_dev = returns .- transpose(mu)
    @constraint(model, abs_dev * model[:w] >= -mad)

    if rm == :mad || isfinite(mad_u)
        @expression(model, mad_risk, sum(mad) / T)

        if isfinite(mad_u)
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

    @variable(model, sdev >= 0)
    @constraint(model, [sdev; mad] in SecondOrderCone())
    @expression(model, sdev_risk, sdev / sqrt(T - 1))

    if isfinite(sdev_u)
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

function _var_setup(portfolio, rm, T, returns, obj)
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

        if isfinite(cvar_u)
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
        @variable(model, s_evar >= 0)
        @variable(model, u_evar[1:T])
        @constraint(model, sum(u_evar) <= s_evar)
        @constraint(
            model,
            [i = 1:T],
            [-model[:hist_ret][i] - t_evar, s_evar, u_evar[i]] in MOI.ExponentialCone()
        )
        @expression(model, evar_risk, t_evar - s_evar * log(at))

        if isfinite(evar_u)
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
    @variable(model, s_rvar >= 0)
    @variable(model, omega_rvar[1:T])
    @variable(model, psi_rvar[1:T])
    @variable(model, theta_rvar[1:T])
    @variable(model, epsilon_rvar[1:T])
    @constraint(
        model,
        [i = 1:T],
        [s_rvar * opk * invkappa2, psi_rvar[i] * opk * invk, epsilon_rvar[i]] in
        MOI.PowerCone(invopk)
    )
    @constraint(
        model,
        [i = 1:T],
        [omega_rvar[i] * invomk, theta_rvar[i] * invk, -s_rvar * invkappa2] in
        MOI.PowerCone(omk)
    )
    @constraint(model, -model[:hist_ret] .- t_rvar .+ epsilon_rvar .+ omega_rvar .<= 0)
    @expression(model, rvar_risk, t_rvar + ln_k * s_rvar + sum(psi_rvar .+ theta_rvar))

    if isfinite(rvar_u)
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

function _wr_setup(portfolio, rm, returns, obj)
    wr_u = portfolio.wr_u

    !(rm == :wr || isfinite(wr_u)) && (return nothing)

    model = portfolio.model

    @variable(model, twr)
    !haskey(model, :hist_ret) && (@expression(model, hist_ret, returns * model[:w]))
    @constraint(model, -model[:hist_ret] .<= twr)
    @expression(model, wr_risk, twr)

    if isfinite(wr_u)
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

function _lpm_setup(portfolio, rm, T, returns, obj, rf)
    flpm_u = portfolio.flpm_u
    slpm_u = portfolio.slpm_u

    !(rm == :flpm || rm == :slpm || isfinite(flpm_u) || isfinite(slpm_u)) &&
        (return nothing)

    model = portfolio.model

    @variable(model, tlpm[1:T] .>= 0)
    !haskey(model, :hist_ret) && (@expression(model, hist_ret, returns * model[:w]))

    if obj == :sharpe
        @constraint(model, tlpm .>= rf * model[:k] .- model[:hist_ret])
    else
        @constraint(model, tlpm .>= rf .- model[:hist_ret])
    end

    if rm == :flpm || isfinite(flpm_u)
        @expression(model, flpm_risk, sum(tlpm) / T)

        if isfinite(flpm_u)
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

    @variable(model, tslpm >= 0)
    @constraint(model, [tslpm; tlpm] in SecondOrderCone())
    @expression(model, slpm_risk, tslpm / sqrt(T - 1))

    if isfinite(slpm_u)
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

function _drawdown_setup(portfolio, rm, T, returns, obj)
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

    @variable(model, tdd[1:(T + 1)])
    !haskey(model, :hist_ret) && (@expression(model, hist_ret, returns * model[:w]))
    @constraint(model, tdd[2:end] .>= tdd[1:(end - 1)] .- model[:hist_ret])
    @constraint(model, tdd[2:end] .>= 0)
    @constraint(model, tdd[1] == 0)

    if rm == :mdd || isfinite(mdd_u)
        @variable(model, tmdd)
        @constraint(model, tmdd .>= tdd[2:end])
        @expression(model, mdd_risk, tmdd)

        if isfinite(mdd_u)
            if obj == :sharpe
                @constraint(model, tdd[2:end] .<= mdd_u * model[:k])
            else
                @constraint(model, tdd[2:end] .<= mdd_u)
            end
        end

        if rm == :mdd
            @expression(model, risk, mdd_risk)
        end
    end

    if rm == :add || isfinite(add_u)
        @expression(model, add_risk, sum(tdd[2:end]) / T)

        if isfinite(add_u)
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
        @variable(model, tdar)
        @variable(model, z_dar[1:T] .>= 0)
        @constraint(model, z_dar .>= tdd[2:end] .- tdar)
        @expression(model, cdar_risk, tdar + sum(z_dar) * invat)

        if isfinite(cdar_u)
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
        @variable(model, tuci >= 0)
        @constraint(model, [tuci; tdd[2:end]] in SecondOrderCone())
        @expression(model, uci_risk, tuci / sqrt(T))

        if isfinite(uci_u)
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
        @variable(model, s_edar >= 0)
        @variable(model, u_edar[1:T])
        @constraint(model, sum(u_edar) <= s_edar)
        @constraint(
            model,
            [i = 1:T],
            [tdd[i + 1] - t_edar, s_edar, u_edar[i]] in MOI.ExponentialCone()
        )
        @expression(model, edar_risk, t_edar - s_edar * log(at))

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
    @variable(model, s_rdar >= 0)
    @variable(model, omega_rdar[1:T])
    @variable(model, psi_rdar[1:T])
    @variable(model, theta_rdar[1:T])
    @variable(model, epsilon_rdar[1:T])
    @constraint(
        model,
        [i = 1:T],
        [s_rdar * opk * invkappa2, psi_rdar[i] * opk * invk, epsilon_rdar[i]] in
        MOI.PowerCone(invopk)
    )
    @constraint(
        model,
        [i = 1:T],
        [omega_rdar[i] * invomk, theta_rdar[i] * invk, -s_rdar * invkappa2] in
        MOI.PowerCone(omk)
    )
    @constraint(model, tdd[2:end] .- t_rdar .+ epsilon_rdar .+ omega_rdar .<= 0)
    @expression(model, rdar_risk, t_rdar + ln_k * s_rdar + sum(psi_rdar .+ theta_rdar))

    if isfinite(rdar_u)
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

function _owa_setup(portfolio, rm, T, returns, obj)
    gmd_u = portfolio.gmd_u
    tg_u = portfolio.tg_u
    rg_u = portfolio.rg_u
    rcvar_u = portfolio.rcvar_u
    rtg_u = portfolio.rtg_u

    !(
        rm == :gmd ||
        rm == :tg ||
        rm == :rg ||
        rm == :rcvar ||
        rm == :rtg ||
        isfinite(gmd_u) ||
        isfinite(tg_u) ||
        isfinite(rg_u) ||
        isfinite(rcvar_u) ||
        isfinite(rtg_u)
    ) && (return nothing)

    onesvec = ones(T)
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

        if isfinite(gmd_u)
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

        if isfinite(tg_u)
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

    if rm == :rg || isfinite(rg_u)
        @variable(model, rga[1:T])
        @variable(model, rgb[1:T])
        @expression(model, rg_risk, sum(rga .+ rgb))
        rg_w = owa_rg(T)
        @constraint(
            model,
            owa * transpose(rg_w) .<= onesvec * transpose(rga) + rgb * transpose(onesvec)
        )

        if isfinite(rg_u)
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

        if isfinite(rcvar_u)
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

    !(rm == :rtg || isfinite(rtg_u)) && (return nothing)

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
        owa * transpose(rtg_w) .<= onesvec * transpose(rtga) + rtgb * transpose(onesvec)
    )

    if isfinite(rtg_u)
        if obj == :sharpe
            @constraint(model, rtg_risk <= rtg_u * model[:k])
        else
            @constraint(model, rtg_risk <= rtg_u)
        end
    end

    if rm == :rtg
        @expression(model, risk, rtg_risk)
    end

    return nothing
end

function _kurtosis_setup(portfolio, rm, N, obj)
    kurt = portfolio.kurt
    skurt = portfolio.skurt
    krt_u = portfolio.krt_u
    skrt_u = portfolio.skrt_u

    !(
        !isnothing(kurt) ||
        !isnothing(skurt) ||
        rm == :krt ||
        rm == :skrt ||
        isfinite(krt_u) ||
        isfinite(skrt_u)
    ) && (return nothing)

    model = portfolio.model

    if !isnothing(kurt) && (rm == :krt || isfinite(krt_u))
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

        @variable(model, tkurt)
        if !iszero(max_num_assets_kurt) && N > max_num_assets_kurt
            N2 = 2 * N
            @variable(model, xkurt[1:N2])
            @variable(model, rkurt[1:N2])
            @constraint(
                model,
                [i = 1:N2],
                [rkurt[i], tkurt, xkurt[i]] in MOI.PowerCone(1 / 2)
            )
            @constraint(model, sum(rkurt) == tkurt)
            A = block_vec_pq(kurt, N, N)
            vals_A, vecs_A = eigen(A)
            vals_A = real.(vals_A)
            vecs_A = real.(vecs_A)
            clamp!.(vals_A, 0, Inf)
            Bi = Vector{Matrix{eltype(kurt), eltype(kurt)}}(undef, N2)
            for i in 1:N2
                B = sqrt(vals_A[i]) * vecs_A[:, i]
                B = reshape(B, N, N)
                Bi[i] = B
            end
            @constraint(model, [i = 1:N2], xkurt[i] == sum(diag(Bi[i] * W)))
        else
            L_2 = portfolio.L_2
            S_2 = portfolio.S_2
            sqrt_sigma_4 = sqrt(S_2 * kurt * transpose(S_2))
            @constraint(model, tkurt >= 0)
            @expression(model, zkurt, L_2 * vec(W))
            @constraint(model, [tkurt; sqrt_sigma_4 * zkurt] in SecondOrderCone())
        end
        @expression(model, kurt_risk, tkurt)

        if isfinite(krt_u)
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

    if !isnothing(skurt) && (rm == :skrt || isfinite(skrt_u))
        max_num_assets_skurt = portfolio.max_num_assets_skurt
        @variable(model, SW[1:N, 1:N], Symmetric)
        @expression(model, SM1, vcat(SW, transpose(model[:w])))
        if obj == :sharpe
            @expression(model, SM2, vcat(model[:w], model[:k]))
        else
            @expression(model, SM2, vcat(model[:w], 1))
        end
        @expression(model, SM3, hcat(SM1, SM2))
        @constraint(model, SM3 in PSDCone())

        @variable(model, tskurt)
        if !iszero(max_num_assets_kurt) && N > max_num_assets_skurt
            N2 = 2 * N
            @variable(model, xskurt[1:N2])
            @variable(model, rskurt[1:N2])
            @constraint(
                model,
                [i = 1:N2],
                [rskurt[i], tskurt, xskurt[i]] in MOI.PowerCone(1 / 2)
            )
            @constraint(model, sum(rskurt) == tskurt)
            A = block_vec_pq(skurt, N, N)
            vals_A, vecs_A = eigen(A)
            vals_A = real.(vals_A)
            vecs_A = real.(vecs_A)
            clamp!.(vals_A, 0, Inf)
            SBi = Vector{Matrix{eltype(skurt), eltype(skurt)}}(undef, N2)
            for i in 1:N2
                B = sqrt(vals_A[i]) * vecs_A[:, i]
                B = reshape(B, N, N)
                SBi[i] = B
            end

            @constraint(model, [i = 1:N2], xskurt[i] == sum(diag(SBi[i] * SW)))
        else
            L_2 = portfolio.L_2
            S_2 = portfolio.S_2
            sqrt_sigma_4 = sqrt(S_2 * skurt * transpose(S_2))
            @constraint(model, tskurt >= 0)
            @expression(model, zskurt, L_2 * vec(SW))
            @constraint(model, [tskurt; sqrt_sigma_4 * zskurt] in SecondOrderCone())
        end
        @expression(model, skurt_risk, tskurt)

        if isfinite(skrt_u)
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