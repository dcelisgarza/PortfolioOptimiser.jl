function _mv_setup(portfolio, sigma, rm, kelly, obj)
    dev_u = portfolio.dev_u

    !(rm == :mv || kelly == :approx || isfinite(dev_u)) && (return nothing)

    model = portfolio.model

    w = model[:w]
    k = model[:k]
    @variable(model, tdev >= 0)
    @expression(model, dev_risk, tdev * tdev)
    G = sqrt(sigma)
    @constraint(model, [tdev; transpose(G) * w] in SecondOrderCone())

    if isfinite(dev_u)
        if obj == :sharpe
            @constraint(model, tdev <= dev_u * k)
        else
            @constraint(model, tdev <= dev_u)
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
    w = model[:w]
    k = model[:k]

    @variable(model, tmad[1:T] >= 0)
    abs_dev = returns .- transpose(mu)
    @constraint(model, abs_dev * w >= -tmad)

    if rm == :mad || isfinite(mad_u)
        @expression(model, mad_risk, sum(tmad) / T)

        if isfinite(mad_u)
            if obj == :sharpe
                @constraint(model, mad_risk * 2 <= mad_u * k)
            else
                @constraint(model, mad_risk * 2 <= mad_u)
            end
        end

        if rm == :mad
            @expression(model, risk, mad_risk)
        end
    end

    !(rm == :msv || isfinite(sdev_u)) && (return nothing)

    @variable(model, tmsd >= 0)
    @constraint(model, [tmsd; tmad] in SecondOrderCone())
    @expression(model, msd_risk, tmsd / sqrt(T - 1))

    if isfinite(sdev_u)
        if obj == :sharpe
            @constraint(model, msd_risk <= sdev_u * k)
        else
            @constraint(model, msd_risk <= sdev_u)
        end
    end

    if rm == :msv
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
        @constraint(model, z_var .+ hist_ret .+ var .>= 0)
        @expression(model, cvar_risk, var + sum(z_var) / (alpha * T))

        if isfinite(cvar_u)
            if obj == :sharpe
                @constraint(model, cvar_risk <= cvar_u * k)
            else
                @constraint(model, cvar_risk <= cvar_u)
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
        @constraint(model, sum(u_evar) - s_evar <= 0)
        @constraint(
            model,
            [i = 1:T],
            [-hist_ret[i] - t_evar, s_evar, u_evar[i]] in MOI.ExponentialCone()
        )
        @expression(model, evar_risk, t_evar - s_evar * log(alpha * T))

        if isfinite(evar_u)
            if obj == :sharpe
                @constraint(model, evar_risk <= evar_u * k)
            else
                @constraint(model, evar_risk <= evar_u)
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
        [i = 1:T],
        [s_rvar * opk / k2, psi_rvar[i] * opk * invk, epsilon_rvar[i]] in
        MOI.PowerCone(1 / opk)
    )
    @constraint(
        model,
        [i = 1:T],
        [omega_rvar[i] / omk, theta_rvar[i] * invk, -s_rvar / k2] in MOI.PowerCone(omk)
    )
    @constraint(model, -hist_ret .- t_rvar .+ epsilon_rvar .+ omega_rvar .<= 0)
    @expression(model, rvar_risk, t_rvar + ln_k * s_rvar + sum(psi_rvar .+ theta_rvar))

    if isfinite(rvar_u)
        if obj == :sharpe
            @constraint(model, rvar_risk <= rvar_u * k)
        else
            @constraint(model, rvar_risk <= rvar_u * k)
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
    @constraint(model, hist_ret .+ twr .>= 0)
    @expression(model, wr_risk, twr)

    if isfinite(wr_u)
        if obj == :sharpe
            @constraint(model, hist_ret .+ wr_u * k .>= 0)
        else
            @constraint(model, hist_ret .+ wr_u .>= 0)
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
        @constraint(model, tlpm .+ hist_ret .>= rf * k)
    else
        @constraint(model, tlpm .+ hist_ret .>= rf)
    end

    if rm == :flpm || isfinite(flpm_u)
        @expression(model, flpm_risk, sum(tlpm) / T)

        if isfinite(flpm_u)
            if obj == :sharpe
                @constraint(model, flpm_risk <= flpm_u * k)
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
            @constraint(model, slpm_risk <= slpm_u * k)
        else
            @constraint(model, slpm_risk <= slpm_u)
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
    @constraint(model, tdd[2:end] .- tdd[1:(end - 1)] .+ hist_ret .>= 0)
    @constraint(model, tdd[2:end] .>= 0)
    @constraint(model, tdd[1] == 0)

    if rm == :mdd || isfinite(mdd_u)
        @variable(model, tmdd)
        @constraint(model, tmdd .- tdd[2:end] .>= 0)
        @expression(model, mdd_risk, tmdd)

        if isfinite(mdd_u)
            if obj == :sharpe
                @constraint(model, tdd[2:end] .<= mdd_u * k)
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
                @constraint(model, add_risk .<= add_u * k)
            else
                @constraint(model, add_risk .<= add_u)
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
        @constraint(model, zdar .- tdd[2:end] .+ tdar .>= 0)
        @expression(model, cdar_risk, tdar + sum(zdar) / (alpha * T))

        if isfinite(cdar_u)
            if obj == :sharpe
                @constraint(model, cdar_risk .<= cdar_u * k)
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
                @constraint(model, uci_risk <= uci_u * k)
            else
                @constraint(model, uci_risk <= uci_u)
            end
        end

        if rm == :uci
            @expression(model, risk, uci_risk)
        end
    end

    if rm == :edar || isfinite(edar_u)
        alpha = portfolio.alpha
        @variable(model, t_edar)
        @variable(model, s_edar >= 0)
        @variable(model, u_edar[1:T])
        @constraint(model, sum(u_edar) - s_edar <= 0)
        @constraint(
            model,
            [i = 1:T],
            [tdd[i + 1] - t_edar, s_edar, u_edar[i]] in MOI.ExponentialCone()
        )
        @expression(model, edar_risk, t_edar - s_edar * log(alpha * T))

        if isfinite(edar_u)
            if obj == :sharpe
                @constraint(model, edar_risk <= edar_u * k)
            else
                @constraint(model, edar_risk <= edar_u)
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
        [i = 1:T],
        [s_rdar * opk / k2, psi_rdar[i] * opk * invk, epsilon_rdar[i]] in
        MOI.PowerCone(1 / opk)
    )
    @constraint(
        model,
        [i = 1:T],
        [omega_rdar[i] / omk, theta_rdar[i] * invk, -s_rdar / k2] in MOI.PowerCone(omk)
    )
    @constraint(model, tdd[2:end] .- t_rdar .+ epsilon_rdar .+ omega_rdar .<= 0)
    @expression(model, rdar_risk, t_rdar + ln_k * s_rdar + sum(psi_rdar .+ theta_rdar))

    if isfinite(rdar_u)
        if obj == :sharpe
            @constraint(model, rdar_risk <= rdar_u * k)
        else
            @constraint(model, rdar_risk <= rdar_u * k)
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
    w = model[:w]
    k = model[:k]
    if !haskey(model, :hist_ret)
        @expression(model, hist_ret, returns * w)
    end
    hist_ret = model[:hist_ret]
    @variable(model, owa[1:T])
    @constraint(model, hist_ret == owa)

    if rm == :gmd || isfinite(gmd_u)
        @variable(model, gmda[1:T])
        @variable(model, gmdb[1:T])
        @expression(model, gmd_risk, sum(gmda + gmdb))
        gmd_w = owa_gmd(T) / 2
        @constraint(
            model,
            owa * transpose(gmd_w) .<=
            onesvec * transpose(gmda) + gmdb * transpose(onesvec)
        )

        if isfinite(gmd_u)
            if obj == :sharpe
                @constraint(model, gmd_risk <= gmd_u * k / 2)
            else
                @constraint(model, gmd_risk <= gmd_u / 2)
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
        @expression(model, tg_risk, sum(tga + tgb))
        tg_w = owa_tg(T; alpha_i = alpha_i, alpha = alpha, a_sim = a_sim)
        @constraint(
            model,
            owa * transpose(tg_w) .<= onesvec * transpose(tga) + tgb * transpose(onesvec)
        )

        if isfinite(tg_u)
            if obj == :sharpe
                @constraint(model, tg_risk <= tg_u * k)
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
        @expression(model, rg_risk, sum(rga + rgb))
        rg_w = owa_rg(T)
        @constraint(
            model,
            owa * transpose(rg_w) .<= onesvec * transpose(rga) + rgb * transpose(onesvec)
        )

        if isfinite(rg_u)
            if obj == :sharpe
                @constraint(model, rg_risk <= rg_u * k)
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
        @expression(model, rcvar_risk, sum(rcvara + rcvarb))
        rcvar_w = owa_rcvar(T; alpha = alpha, beta = beta)
        @constraint(
            model,
            owa * transpose(rcvar_w) .<=
            onesvec * transpose(rcvara) + rcvarb * transpose(onesvec)
        )

        if isfinite(rcvar_u)
            if obj == :sharpe
                @constraint(model, rcvar_risk <= rcvar_u * k)
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
    @expression(model, rtg_risk, sum(rtga + rtgb))
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
            @constraint(model, rtg_risk <= rtg_u * k)
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
    w = model[:w]
    k = model[:k]

    if !isnothing(kurt) && (rm == :krt || isfinite(krt_u))
        max_num_assets_kurt = portfolio.max_num_assets_kurt
        @variable(model, W[1:N, 1:N], Symmetric)
        @expression(model, M1, vcat(W, transpose(w)))
        if obj == :sharpe
            @expression(model, M2, vcat(w, k))
        else
            @expression(model, M2, vcat(w, 1))
        end
        @expression(model, M3, hcat(M1, M2))
        @constraint(model, M3 in PSDCone())

        if !iszero(max_num_assets_kurt) && N > max_num_assets_kurt
            K = 2 * N
            @variable(model, xkurt[1:K])
            @variable(model, rkurt[1:K])
            @variable(model, tkurt)
            @constraint(
                model,
                [i = 1:K],
                [rkurt[i], tkurt, xkurt[i]] in MOI.PowerCone(1 / 2)
            )
            @constraint(model, sum(rkurt) == tkurt)
            @expression(model, kurt_risk, tkurt)
            A = block_vec_pw(kurt, N, N)
            vals_A, vecs_A = eigen(A)
            vals_A = real(vals_A)
            vecs_A = real(vecs_A)
            clamp!(vals_A, 0, Inf)
            Bi = Vector{Matrix{Float64, Float64}}(undef, K)
            for i in 1:K
                B = sqrt(vals_A[i]) * vecs_A[:, i]
                B = reshape(B, N, N)
                Bi[i] = B
            end

            @constraint(model, [i = 1:K], xkurt[i] == sum(diag(Bi[i] * W)))
        else
            L_2 = portfolio.L_2
            S_2 = portfolio.S_2
            sqrt_sigma_4 = sqrt(S_2 * kurt * transpose(L_2))
            @variable(model, tkurt >= 0)
            @expression(model, zkurt, L_2 * vec(W))
            @constraint(model, [tkurt; sqrt_sigma_4 * zkurt] in SecondOrderCone())
            @expression(model, kurt_risk, tkurt)
        end

        if isfinite(krt_u)
            if obj == :sharpe
                @constraint(model, kurt_risk <= krt_u * k)
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
        @expression(model, SM1, vcat(SW, transpose(w)))
        if obj == :sharpe
            @expression(model, SM2, vcat(w, k))
        else
            @expression(model, SM2, vcat(w, 1))
        end
        @expression(model, SM3, hcat(SM1, SM2))
        @constraint(model, SM3 in PSDCone())

        if !iszero(max_num_assets_kurt) && N > max_num_assets_skurt
            K = 2 * N
            @variable(model, xskurt[1:K])
            @variable(model, rskurt[1:K])
            @variable(model, tskurt)
            @constraint(
                model,
                [i = 1:K],
                [rskurt[i], tskurt, xskurt[i]] in MOI.PowerCone(1 / 2)
            )
            @constraint(model, sum(rskurt) == tskurt)
            @expression(model, skurt_risk, tskurt)
            A = block_vec_pw(skurt, N, N)
            vals_A, vecs_A = eigen(A)
            vals_A = real(vals_A)
            vecs_A = real(vecs_A)
            clamp!(vals_A, 0, Inf)
            SBi = Vector{Matrix{Float64, Float64}}(undef, K)
            for i in 1:K
                B = sqrt(vals_A[i]) * vecs_A[:, i]
                B = reshape(B, N, N)
                SBi[i] = B
            end

            @constraint(model, [i = 1:K], xskurt[i] == sum(diag(SBi[i] * SW)))
        else
            L_2 = portfolio.L_2
            S_2 = portfolio.S_2
            sqrt_sigma_4 = sqrt(S_2 * skurt * transpose(L_2))
            @variable(model, tskurt >= 0)
            @expression(model, zskurt, L_2 * vec(SW))
            @constraint(model, [tskurt; sqrt_sigma_4 * zskurt] in SecondOrderCone())
            @expression(model, skurt_risk, tskurt)
        end

        if isfinite(skrt_u)
            if obj == :sharpe
                @constraint(model, skurt_risk <= skrt_u * k)
            else
                @constraint(model, skurt_risk <= skrt_u)
            end
        end

        if rm == :skrt
            @expression(model, risk, skurt_risk)
        end
    end
end