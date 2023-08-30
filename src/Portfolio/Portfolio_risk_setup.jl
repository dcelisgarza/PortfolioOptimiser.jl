
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

    isfinite(dev_u) &&
        type == :trad &&
        (
            obj == :sharpe ? @constraint(model, dev <= dev_u * model[:k]) :
            @constraint(model, dev <= dev_u)
        )

    rm == :mv && type != :rrp && @expression(model, risk, dev_risk)

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

        isfinite(mad_u) &&
            type == :trad &&
            (
                obj == :sharpe ? @constraint(model, mad_risk <= 0.5 * mad_u * model[:k]) :
                @constraint(model, mad_risk <= 0.5 * mad_u)
            )

        rm == :mad && @expression(model, risk, mad_risk)
    end

    !(rm == :msv || isfinite(sdev_u)) && (return nothing)

    @variable(model, sdev)
    @constraint(model, [sdev; mad] in SecondOrderCone())

    @expression(model, sdev_risk, sdev / sqrt(T - 1))

    isfinite(sdev_u) &&
        type == :trad &&
        (
            obj == :sharpe ? @constraint(model, sdev_risk <= sdev_u * model[:k]) :
            @constraint(model, sdev_risk <= sdev_u)
        )

    rm == :msv && @expression(model, risk, sdev_risk)

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
    !haskey(model, :hist_ret) && @expression(model, hist_ret, returns * model[:w])

    obj == :sharpe || type == :rp ?
    @constraint(model, lpm .>= lpm_t * model[:k] .- model[:hist_ret]) :
    @constraint(model, lpm .>= lpm_t .- model[:hist_ret])

    if rm == :flpm || isfinite(flpm_u)
        @expression(model, flpm_risk, sum(lpm) / T)

        isfinite(flpm_u) &&
            type == :trad &&
            (
                obj == :sharpe ? @constraint(model, flpm_risk <= flpm_u * model[:k]) :
                @constraint(model, flpm_risk <= flpm_u)
            )

        rm == :flpm && @expression(model, risk, flpm_risk)
    end

    !(rm == :slpm || isfinite(slpm_u)) && (return nothing)

    @variable(model, slpm)
    @constraint(model, [slpm; lpm] in SecondOrderCone())
    @expression(model, slpm_risk, slpm / sqrt(T - 1))

    isfinite(slpm_u) &&
        type == :trad &&
        (
            obj == :sharpe ? @constraint(model, slpm_risk <= slpm_u * model[:k]) :
            @constraint(model, slpm_risk <= slpm_u)
        )

    rm == :slpm && @expression(model, risk, slpm_risk)

    return nothing
end

function _wr_setup(portfolio, rm, returns, obj, type)
    wr_u = portfolio.wr_u

    !(rm == :wr || isfinite(wr_u)) && (return nothing)

    model = portfolio.model

    @variable(model, wr)
    !haskey(model, :hist_ret) && @expression(model, hist_ret, returns * model[:w])
    @constraint(model, -model[:hist_ret] .<= wr)
    @expression(model, wr_risk, wr)

    isfinite(wr_u) &&
        type == :trad &&
        (
            obj == :sharpe ? @constraint(model, -model[:hist_ret] .<= wr_u * model[:k]) :
            @constraint(model, -model[:hist_ret] .<= wr_u)
        )

    rm == :wr && @expression(model, risk, wr_risk)

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

    !haskey(model, :hist_ret) && @expression(model, hist_ret, returns * model[:w])

    if rm == :cvar || isfinite(cvar_u)
        invat = portfolio.invat
        @variable(model, var)
        @variable(model, z_var[1:T] >= 0)
        @constraint(model, z_var .>= -model[:hist_ret] .- var)
        @expression(model, cvar_risk, var + sum(z_var) * invat)

        isfinite(cvar_u) &&
            type == :trad &&
            (
                obj == :sharpe ? @constraint(model, cvar_risk <= cvar_u * model[:k]) :
                @constraint(model, cvar_risk <= cvar_u)
            )

        rm == :cvar && @expression(model, risk, cvar_risk)
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

        isfinite(evar_u) &&
            type == :trad &&
            (
                obj == :sharpe ? @constraint(model, evar_risk <= evar_u * model[:k]) :
                @constraint(model, evar_risk <= evar_u)
            )

        rm == :evar && @expression(model, risk, evar_risk)
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

    isfinite(rvar_u) &&
        type == :trad &&
        (
            obj == :sharpe ? @constraint(model, rvar_risk <= rvar_u * model[:k]) :
            @constraint(model, rvar_risk <= rvar_u)
        )

    rm == :rvar && @expression(model, risk, rvar_risk)

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
    !haskey(model, :hist_ret) && @expression(model, hist_ret, returns * model[:w])
    @constraint(model, dd[2:end] .>= dd[1:(end - 1)] .- model[:hist_ret])
    @constraint(model, dd[2:end] .>= 0)
    @constraint(model, dd[1] == 0)

    if rm == :mdd || isfinite(mdd_u)
        @variable(model, mdd)
        @constraint(model, mdd .>= dd[2:end])
        @expression(model, mdd_risk, mdd)

        isfinite(mdd_u) &&
            type == :trad &&
            (
                obj == :sharpe ? @constraint(model, dd[2:end] .<= mdd_u * model[:k]) :
                @constraint(model, dd[2:end] .<= mdd_u)
            )

        rm == :mdd && @expression(model, risk, mdd_risk)
    end

    if rm == :add || isfinite(add_u)
        @expression(model, add_risk, sum(dd[2:end]) / T)

        isfinite(add_u) &&
            type == :trad &&
            (
                obj == :sharpe ? @constraint(model, add_risk .<= add_u * model[:k]) :
                @constraint(model, add_risk .<= add_u)
            )

        rm == :add && @expression(model, risk, add_risk)
    end

    if rm == :cdar || isfinite(cdar_u)
        invat = portfolio.invat
        @variable(model, dar)
        @variable(model, z_dar[1:T] .>= 0)
        @constraint(model, z_dar .>= dd[2:end] .- dar)
        @expression(model, cdar_risk, dar + sum(z_dar) * invat)

        isfinite(cdar_u) &&
            type == :trad &&
            (
                obj == :sharpe ? @constraint(model, cdar_risk .<= cdar_u * model[:k]) :
                @constraint(model, cdar_risk .<= cdar_u)
            )

        rm == :cdar && @expression(model, risk, cdar_risk)
    end

    if rm == :uci || isfinite(uci_u)
        @variable(model, uci)
        @constraint(model, [uci; dd[2:end]] in SecondOrderCone())
        @expression(model, uci_risk, uci / sqrt(T))

        isfinite(uci_u) &&
            type == :trad &&
            (
                obj == :sharpe ? @constraint(model, uci_risk <= uci_u * model[:k]) :
                @constraint(model, uci_risk <= uci_u)
            )

        rm == :uci && @expression(model, risk, uci_risk)
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

        isfinite(edar_u) &&
            type == :trad &&
            (
                obj == :sharpe ? @constraint(model, edar_risk <= edar_u * model[:k]) :
                @constraint(model, edar_risk <= edar_u)
            )

        rm == :edar && @expression(model, risk, edar_risk)
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

    isfinite(rdar_u) &&
        type == :trad &&
        (
            obj == :sharpe ? @constraint(model, rdar_risk <= rdar_u * model[:k]) :
            @constraint(model, rdar_risk <= rdar_u)
        )

    rm == :rdar && @expression(model, risk, rdar_risk)

    return nothing
end

function block_vec_pq(A, p, q)
    mp, nq = size(A)

    !(mod(mp, p) == 0 && mod(nq, p) == 0) && (throw(
        DimensionMismatch(
            "dimensions A, $(size(A)), must be integer multiples of (p, q) = ($p, $q)",
        ),
    ))

    m = Int(mp / p)
    n = Int(nq / q)

    A_vec = Matrix{eltype(A)}(undef, m * n, p * q)
    for j in 0:(n - 1)
        Aj = Matrix{eltype(A)}(undef, m, p * q)
        for i in 0:(m - 1)
            Aij = vec(A[(1 + (i * p)):((i + 1) * p), (1 + (j * q)):((j + 1) * q)])
            Aj[i + 1, :] .= Aij
        end
        A_vec[(1 + (j * m)):((j + 1) * m), :] .= Aj
    end

    return A_vec
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

        obj == :sharpe ? @expression(model, M2, vcat(model[:w], model[:k])) :
        @expression(model, M2, vcat(model[:w], 1))

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

        isfinite(krt_u) &&
            type == :trad &&
            (
                obj == :sharpe ? @constraint(model, kurt_risk <= krt_u * model[:k]) :
                @constraint(model, kurt_risk <= krt_u)
            )

        rm == :krt && @expression(model, risk, kurt_risk)
    end

    if rm == :skrt || isfinite(skrt_u)
        max_num_assets_kurt = portfolio.max_num_assets_kurt
        @variable(model, SW[1:N, 1:N], Symmetric)
        @expression(model, SM1, vcat(SW, transpose(model[:w])))

        obj == :sharpe ? @expression(model, SM2, vcat(model[:w], model[:k])) :
        @expression(model, SM2, vcat(model[:w], 1))

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

        isfinite(skrt_u) &&
            type == :trad &&
            (
                obj == :sharpe ? @constraint(model, skurt_risk <= skrt_u * model[:k]) :
                @constraint(model, skurt_risk <= skrt_u)
            )

        rm == :skrt && @expression(model, risk, skurt_risk)
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

    !haskey(model, :hist_ret) && @expression(model, hist_ret, returns * model[:w])
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

        isfinite(gmd_u) &&
            type == :trad &&
            (
                obj == :sharpe ? @constraint(model, gmd_risk <= gmd_u * model[:k] * 0.5) :
                @constraint(model, gmd_risk <= gmd_u * 0.5)
            )

        rm == :gmd && @expression(model, risk, gmd_risk)
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

        isfinite(rg_u) &&
            type == :trad &&
            (
                obj == :sharpe ? @constraint(model, rg_risk <= rg_u * model[:k]) :
                @constraint(model, rg_risk <= rg_u)
            )

        rm == :rg && @expression(model, risk, rg_risk)
    end

    if rm == :rcvar || isfinite(rcvar_u)
        alpha = portfolio.alpha
        beta = portfolio.beta

        @variable(model, rcvara[1:T])
        @variable(model, rcvarb[1:T])
        @expression(model, rcvar_risk, sum(rcvara .+ rcvarb))
        rcvar_w = owa_rcvar(T; alpha = alpha, beta = beta)
        @constraint(
            model,
            owa * transpose(rcvar_w) .<=
            onesvec * transpose(rcvara) + rcvarb * transpose(onesvec)
        )

        isfinite(rcvar_u) &&
            type == :trad &&
            (
                obj == :sharpe ? @constraint(model, rcvar_risk <= rcvar_u * model[:k]) :
                @constraint(model, rcvar_risk <= rcvar_u)
            )

        rm == :rcvar && @expression(model, risk, rcvar_risk)
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

        isfinite(tg_u) &&
            type == :trad &&
            (
                obj == :sharpe ? @constraint(model, tg_risk <= tg_u * model[:k]) :
                @constraint(model, tg_risk <= tg_u)
            )

        rm == :tg && @expression(model, risk, tg_risk)
    end

    if rm == :rtg || isfinite(rtg_u)
        alpha = portfolio.alpha
        a_sim = portfolio.a_sim
        alpha_i = portfolio.alpha_i
        beta = portfolio.beta
        b_sim = portfolio.b_sim
        beta_i = portfolio.beta_i

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

        isfinite(rtg_u) &&
            type == :trad &&
            (
                obj == :sharpe ? @constraint(model, rtg_risk <= rtg_u * model[:k]) :
                @constraint(model, rtg_risk <= rtg_u)
            )

        rm == :rtg && @expression(model, risk, rtg_risk)
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

    isfinite(owa_u) &&
        type == :trad &&
        (
            obj == :sharpe ? @constraint(model, owa_risk <= owa_u * model[:k]) :
            @constraint(model, owa_risk <= owa_u)
        )

    rm == :owa && @expression(model, risk, owa_risk)

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
        obj == :sharpe && @constraint(model, ret - rf * model[:k] >= 1)
    elseif u_mu == :ellipse
        k_mu = portfolio.k_mu
        cov_mu = portfolio.cov_mu
        G = sqrt(cov_mu)
        @expression(model, x_gw, G * model[:w])
        @variable(model, t_gw)
        @constraint(model, [t_gw; x_gw] in SecondOrderCone())
        @expression(model, ret, _ret - k_mu * t_gw)
        obj == :sharpe && @constraint(model, ret - rf * model[:k] >= 1)
    else
        @expression(model, ret, _ret)
        obj == :sharpe && @constraint(model, ret - rf * model[:k] >= 1)
    end

    # Cov uncertainty sets.
    if u_cov == :box
        cov_u = portfolio.cov_u
        cov_l = portfolio.cov_l
        @variable(model, Au[1:N, 1:N] .>= 0, Symmetric)
        @variable(model, Al[1:N, 1:N] .>= 0, Symmetric)
        @expression(model, M1, vcat(Au - Al, transpose(model[:w])))

        obj == :sharpe ? @expression(model, M2, vcat(model[:w], model[:k])) :
        @expression(model, M2, vcat(model[:w], 1))

        @expression(model, M3, hcat(M1, M2))
        @constraint(model, M3 in PSDCone())
        @expression(model, risk, tr(Au * cov_u) - tr(Al * cov_l))
    elseif u_cov == :ellipse
        k_sigma = portfolio.k_sigma
        G_sigma = sqrt(portfolio.cov_sigma)
        @variable(model, E1[1:N, 1:N], Symmetric)
        @variable(model, E2[1:N, 1:N], Symmetric)
        @expression(M1, vcat(E1, transpose(model[:w])))

        obj == :sharpe ? @expression(M2, vcat(model[:w], model[:k])) :
        @expression(M2, vcat(model[:w], 1))

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

function _naive_risk(portfolio, returns, covariance; rm = :mv, rf = 0.0)
    N = size(returns, 2)
    tcov = eltype(covariance)

    if rm == :equal
        weights = fill(tcov(1 / N), N)
    else
        inv_risk = Vector{tcov}(undef, N)
        w = Vector{tcov}(undef, N)
        for i in 1:N
            w .= zero(tcov)
            w[i] = one(tcov)
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

function _opt_w(portfolio, returns, mu, icov; obj = :min_risk, rm = :mv, rf = 0.0, l = 2.0)
    port = Portfolio(assets = 1:length(mu), ret = returns, solvers = portfolio.solvers)
    asset_statistics!(port; calc_kurt = false)
    port.cov = icov

    weights = if obj ∈ (:min_risk, :utility, :sharpe)
        !isnothing(mu) && (port.mu = mu)
        opt_port!(port; type = :trad, class = :classic, rm = rm, obj = obj, rf = rf, l = l)
    elseif obj == :erc
        opt_port!(port; type = :rp, class = :classic, rm = rm, rf = rf)
    end

    return weights[!, :weights]
end

function two_diff_gap_stat(dist, clustering, max_k = 10)
    N = size(dist, 1)
    cluster_lvls = [cutree(clustering; k = i) for i in 1:N]

    c1 = min(N, max_k)
    W_list = Vector{eltype(dist)}(undef, c1)

    for i in 1:c1
        lvl = cluster_lvls[i]
        c2 = maximum(unique(lvl))
        mean_dist = 0.0
        for j in 1:c2
            cluster = findall(lvl .== j)
            cluster_dist = dist[cluster, cluster]
            isempty(cluster_dist) && continue

            val = 0.0
            counter = 0
            M = size(cluster_dist, 1)
            for col in 1:M
                for row in (col + 1):M
                    val += cluster_dist[row, col]
                    counter += 1
                end
            end
            counter == 0 && continue
            mean_dist += val / counter
        end
        W_list[i] = mean_dist
    end

    limit_k = floor(Int, min(max_k, sqrt(N)))
    gaps = fill(-Inf, length(W_list))

    length(W_list) > 2 &&
        (gaps[3:end] .= W_list[3:end] .+ W_list[1:(end - 2)] .- 2 * W_list[2:(end - 1)])

    gaps = gaps[1:limit_k]

    k = all(isinf.(gaps)) ? length(gaps) : k = argmax(gaps) + 1

    return k
end

function _hierarchical_clustering(
    portfolio::HCPortfolio,
    type = :hrp,
    linkage = :ward,
    max_k = 10,
    branchorder = :optimal,
)
    codep_type = portfolio.codep_type
    codep = portfolio.codep
    dist = portfolio.dist

    codeps1 = (:pearson, :spearman, :kendall, :gerber1, :gerber2, :custom)

    if linkage == :dbht
        codep = codep_type ∈ codeps1 ? 1 .- dist .^ 2 : codep
        missing, missing, missing, missing, missing, missing, clustering =
            DBHTs(dist, codep, branchorder = branchorder)
    else
        clustering = hclust(
            dist;
            linkage = linkage,
            branchorder = branchorder == :default ? :r : branchorder,
        )
    end

    k = type ∈ (:herc, :herc2, :nco) ? two_diff_gap_stat(dist, clustering, max_k) : nothing

    return clustering, k
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

function _hr_weight_bounds(upper_bound, lower_bound, weights, lc, rc, alpha_1)
    !(any(upper_bound .< weights) || any(lower_bound .> weights)) && return alpha_1

    lmaxw = weights[lc[1]]
    a1 = sum(upper_bound[lc]) / lmaxw
    a2 = max(sum(lower_bound[lc]) / lmaxw, alpha_1)
    alpha_1 = min(a1, a2)

    rmaxw = weights[rc[1]]
    a1 = sum(upper_bound[rc]) / rmaxw
    a2 = max(sum(lower_bound[rc]) / rmaxw, 1 - alpha_1)
    alpha_1 = 1 - min(a1, a2)

    return alpha_1
end

function _opt_weight_bounds(upper_bound, lower_bound, weights, max_iter = 100)
    !(any(upper_bound .< weights) || any(lower_bound .> weights)) && return weights

    for _ in 1:max_iter
        !(any(upper_bound .< weights) || any(lower_bound .> weights)) && break

        old_w = copy(weights)
        weights = max.(min.(weights, upper_bound), lower_bound)
        idx = weights .< upper_bound .&& weights .> lower_bound
        w_add = sum(max.(old_w - upper_bound, 0.0))
        w_sub = sum(min.(old_w - lower_bound, 0.0))
        delta = w_add + w_sub

        delta != 0 && (weights[idx] += delta * weights[idx] / sum(weights[idx]))
    end

    return weights
end

function _recursive_bisection(
    portfolio;
    rm = :mv,
    rf = 0.0,
    upper_bound = nothing,
    lower_bound = nothing,
)
    N = length(portfolio.assets)
    weights = fill(1.0, N)
    sort_order = portfolio.clusters.order
    items = [sort_order]
    returns = portfolio.returns
    covariance = portfolio.cov

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
            alpha_1 = _hr_weight_bounds(upper_bound, lower_bound, weights, lc, rc, alpha_1)

            weights[lc] *= alpha_1
            weights[rc] *= 1 - alpha_1
        end
    end

    return weights
end

struct ClusterNode{tid, tl, tr, td, tcnt}
    id::tid
    left::tl
    right::tr
    dist::td
    count::tcnt

    function ClusterNode(
        id,
        left::Union{ClusterNode, Nothing} = nothing,
        right::Union{ClusterNode, Nothing} = nothing,
        dist::AbstractFloat = 0.0,
        count::Int = 1,
    )
        icount = isnothing(left) ? count : (left.count + right.count)

        new{typeof(id), typeof(left), typeof(right), typeof(dist), typeof(count)}(
            id,
            left,
            right,
            dist,
            icount,
        )
    end
end
import Base.>, Base.<, Base.==
<(a::ClusterNode, b::ClusterNode) = a.dist < b.dist
>(a::ClusterNode, b::ClusterNode) = a.dist > b.dist
==(a::ClusterNode, b::ClusterNode) = a.dist == b.dist
function is_leaf(a::ClusterNode)
    isnothing(a.left)
end
function pre_order(a::ClusterNode, func::Function = x -> x.id)
    n = a.count
    curNode = Vector{ClusterNode}(undef, 2 * n)
    lvisited = Set()
    rvisited = Set()
    curNode[1] = a
    k = 1
    preorder = Int[]

    while k >= 1
        nd = curNode[k]
        ndid = nd.id
        if is_leaf(nd)
            push!(preorder, func(nd))
            k = k - 1
        else
            if ndid ∉ lvisited
                curNode[k + 1] = nd.left
                push!(lvisited, ndid)
                k = k + 1
            elseif ndid ∉ rvisited
                curNode[k + 1] = nd.right
                push!(rvisited, ndid)
                k = k + 1
                # If we've visited the left and right of this non-leaf
                # node already, go up in the tree.
            else
                k = k - 1
            end
        end
    end

    return preorder
end

function to_tree(a::Hclust)
    n = length(a.order)
    d = Vector{ClusterNode}(undef, 2 * n - 1)
    for i in 1:n
        d[i] = ClusterNode(i)
    end
    merges = a.merges
    heights = a.heights
    nd = nothing

    for (i, height) in enumerate(heights)
        fi = merges[i, 1]
        fj = merges[i, 2]

        fi = fi < 0 ? -fi : fi + n
        fj = fj < 0 ? -fj : fj + n

        nd = ClusterNode(i + n, d[fi], d[fj], height)
        d[n + i] = nd
    end
    return nd, d
end

function _hierarchical_recursive_bisection(
    portfolio;
    rm = :mv,
    rf = 0.0,
    type = :herc,
    upper_bound = nothing,
    lower_bound = nothing,
)
    returns = portfolio.returns
    covariance = portfolio.cov
    clustering = portfolio.clusters
    k = portfolio.k
    root, nodes = to_tree(clustering)
    dists = [i.dist for i in nodes]
    idx = sortperm(dists, rev = true)
    nodes = nodes[idx]

    weights = ones(length(portfolio.assets))

    clustering_idx = cutree(clustering; k = k)

    uidx = minimum(clustering_idx):maximum(clustering_idx)

    clusters = Vector{Vector{Int}}(undef, length(uidx))
    for i in eachindex(clusters)
        clusters[i] = findall(clustering_idx .== i)
    end

    # Calculate intra cluster weights. Drill down into clusters closer in similarity.
    for i in nodes[1:(k - 1)]
        is_leaf(i) && continue

        ln = pre_order(i.left)
        rn = pre_order(i.right)

        lrisk = 0.0
        rrisk = 0.0

        lc = Int[]
        rc = Int[]

        if rm == :equal
            alpha_1 = 0.5
        else
            for j in eachindex(clusters)
                if issubset(clusters[j], ln)
                    _lrisk = _cluster_risk(
                        portfolio,
                        returns,
                        covariance,
                        clusters[j];
                        rm = rm,
                        rf = rf,
                    )
                    lrisk += _lrisk
                    append!(lc, clusters[j])
                elseif issubset(clusters[j], rn)
                    _rrisk = _cluster_risk(
                        portfolio,
                        returns,
                        covariance,
                        clusters[j];
                        rm = rm,
                        rf = rf,
                    )
                    rrisk += _rrisk
                    append!(rc, clusters[j])
                end
            end

            alpha_1 = 1 - lrisk / (lrisk + rrisk)

            alpha_1 = _hr_weight_bounds(upper_bound, lower_bound, weights, lc, rc, alpha_1)
        end

        weights[lc] *= alpha_1
        weights[rc] *= 1 - alpha_1
    end

    # If herc2, then each cluster contributes an equal amount of risk.
    type == :herc2 && (rm = :equal)
    # We multiply the intra cluster weights by the weights by the weights of the cluster.
    for i in 1:k
        cidx = clustering_idx .== i
        cret = returns[:, cidx]
        ccov = covariance[cidx, cidx]
        cweights = _naive_risk(portfolio, cret, ccov; rm = rm, rf = rf)
        weights[cidx] .*= cweights
    end

    return weights
end

function _intra_weights(portfolio; obj = :min_risk, rm = :mv, rf = 0.0, l = 2.0)
    returns = portfolio.returns
    mu = portfolio.mu
    covariance = portfolio.cov
    clustering = portfolio.clusters
    k = portfolio.k
    clustering_idx = cutree(clustering; k = k)

    intra_weights = zeros(eltype(covariance), length(portfolio.assets), k)
    for i in 1:k
        idx = clustering_idx .== i
        cmu = !isnothing(mu) ? mu[idx] : nothing
        ccov = covariance[idx, idx]
        cret = returns[:, idx]
        weights = _opt_w(portfolio, cret, cmu, ccov; obj = obj, rm = rm, rf = rf, l = l)
        intra_weights[idx, i] .= weights
    end

    return intra_weights
end

function _inter_weights(
    portfolio,
    intra_weights;
    obj = :min_risk,
    rm = :mv,
    rf = 0.0,
    l = 2.0,
)
    mu = portfolio.mu
    returns = portfolio.returns
    covariance = portfolio.cov
    tmu = !isnothing(mu) ? transpose(intra_weights) * mu : nothing
    tcov = transpose(intra_weights) * covariance * intra_weights
    tret = returns * intra_weights
    inter_weights = _opt_w(portfolio, tret, tmu, tcov; obj = obj, rm = rm, rf = rf, l = l)

    weights = intra_weights * inter_weights

    return weights
end

export pre_order, ClusterNode, to_tree, is_leaf
