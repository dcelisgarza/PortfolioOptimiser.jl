function _mv_risk(model, sigma)
    G = sqrt(sigma)
    @variable(model, dev)
    @constraint(model, [dev; G * model[:w]] in SecondOrderCone())
    @expression(model, dev_risk, dev * dev)
    return nothing
end

function _mv_setup(portfolio, sigma, rm, kelly, obj, type)
    dev_u = portfolio.dev_u

    !(rm == :SD || kelly == :Approx || isfinite(dev_u)) && (return nothing)

    model = portfolio.model

    _mv_risk(model, sigma)

    isfinite(dev_u) &&
        type == :Trad &&
        (
            obj == :Sharpe ? @constraint(model, model[:dev] <= dev_u * model[:k]) :
            @constraint(model, model[:dev] <= dev_u)
        )

    rm == :SD && type != :RRP && @expression(model, risk, model[:dev_risk])

    return nothing
end

function _mad_setup(portfolio, rm, T, returns, mu, obj, type)
    mad_u = portfolio.mad_u
    sdev_u = portfolio.sdev_u

    !(rm == :MAD || rm == :SSD || isfinite(mad_u) || isfinite(sdev_u)) && (return nothing)

    model = portfolio.model
    msv_target = portfolio.msv_target

    abs_dev =
        if (isa(msv_target, Real) && isinf(msv_target)) ||
           (isa(msv_target, AbstractVector) && isempty(msv_target))
            returns .- transpose(mu)
        elseif isa(msv_target, Real) && isfinite(msv_target)
            returns .- msv_target
        else
            returns .- transpose(msv_target)
        end

    @variable(model, mad[1:T] >= 0)
    @constraint(model, abs_dev * model[:w] .>= -mad)

    if rm == :MAD || isfinite(mad_u)
        @expression(model, mad_risk, sum(mad) / T)

        isfinite(mad_u) &&
            type == :Trad &&
            (
                obj == :Sharpe ? @constraint(model, mad_risk <= 0.5 * mad_u * model[:k]) :
                @constraint(model, mad_risk <= 0.5 * mad_u)
            )

        rm == :MAD && @expression(model, risk, mad_risk)
    end

    !(rm == :SSD || isfinite(sdev_u)) && (return nothing)

    @variable(model, sdev)
    @constraint(model, [sdev; mad] in SecondOrderCone())

    @expression(model, sdev_risk, sdev / sqrt(T - 1))

    isfinite(sdev_u) &&
        type == :Trad &&
        (
            obj == :Sharpe ? @constraint(model, sdev_risk <= sdev_u * model[:k]) :
            @constraint(model, sdev_risk <= sdev_u)
        )

    rm == :SSD && @expression(model, risk, sdev_risk)

    return nothing
end

function _lpm_setup(portfolio, rm, T, returns, obj, rf, type)
    flpm_u = portfolio.flpm_u
    slpm_u = portfolio.slpm_u

    !(rm == :FLPM || rm == :SLPM || isfinite(flpm_u) || isfinite(slpm_u)) &&
        (return nothing)

    model = portfolio.model

    lpm_target = portfolio.lpm_target

    lpm_t =
        if (isa(lpm_target, Real) && isinf(lpm_target)) ||
           (isa(lpm_target, AbstractVector) && isempty(lpm_target))
            rf
        elseif isa(lpm_target, Real) && isfinite(lpm_target)
            lpm_target
        else
            transpose(lpm_target)
        end

    @variable(model, lpm[1:T] .>= 0)
    !haskey(model, :hist_ret) && @expression(model, hist_ret, returns * model[:w])

    obj == :Sharpe || type == :RP ?
    @constraint(model, lpm .>= lpm_t * model[:k] .- model[:hist_ret]) :
    @constraint(model, lpm .>= lpm_t .- model[:hist_ret])

    if rm == :FLPM || isfinite(flpm_u)
        @expression(model, flpm_risk, sum(lpm) / T)

        isfinite(flpm_u) &&
            type == :Trad &&
            (
                obj == :Sharpe ? @constraint(model, flpm_risk <= flpm_u * model[:k]) :
                @constraint(model, flpm_risk <= flpm_u)
            )

        rm == :FLPM && @expression(model, risk, flpm_risk)
    end

    !(rm == :SLPM || isfinite(slpm_u)) && (return nothing)

    @variable(model, slpm)
    @constraint(model, [slpm; lpm] in SecondOrderCone())
    @expression(model, slpm_risk, slpm / sqrt(T - 1))

    isfinite(slpm_u) &&
        type == :Trad &&
        (
            obj == :Sharpe ? @constraint(model, slpm_risk <= slpm_u * model[:k]) :
            @constraint(model, slpm_risk <= slpm_u)
        )

    rm == :SLPM && @expression(model, risk, slpm_risk)

    return nothing
end

function _wr_setup(portfolio, rm, returns, obj, type)
    wr_u = portfolio.wr_u

    !(rm == :WR || isfinite(wr_u)) && (return nothing)

    model = portfolio.model

    @variable(model, wr)
    !haskey(model, :hist_ret) && @expression(model, hist_ret, returns * model[:w])
    @constraint(model, -model[:hist_ret] .<= wr)
    @expression(model, wr_risk, wr)

    isfinite(wr_u) &&
        type == :Trad &&
        (
            obj == :Sharpe ? @constraint(model, -model[:hist_ret] .<= wr_u * model[:k]) :
            @constraint(model, -model[:hist_ret] .<= wr_u)
        )

    rm == :WR && @expression(model, risk, wr_risk)

    return nothing
end

function _var_setup(portfolio, rm, T, returns, obj, type)
    cvar_u = portfolio.cvar_u
    evar_u = portfolio.evar_u
    rvar_u = portfolio.rvar_u

    !(
        rm == :CVaR ||
        rm == :EVaR ||
        rm == :RVaR ||
        isfinite(evar_u) ||
        isfinite(cvar_u) ||
        isfinite(rvar_u)
    ) && (return nothing)

    model = portfolio.model

    !haskey(model, :hist_ret) && @expression(model, hist_ret, returns * model[:w])

    if rm == :CVaR || isfinite(cvar_u)
        invat = portfolio.invat
        @variable(model, var)
        @variable(model, z_var[1:T] >= 0)
        @constraint(model, z_var .>= -model[:hist_ret] .- var)
        @expression(model, cvar_risk, var + sum(z_var) * invat)

        isfinite(cvar_u) &&
            type == :Trad &&
            (
                obj == :Sharpe ? @constraint(model, cvar_risk <= cvar_u * model[:k]) :
                @constraint(model, cvar_risk <= cvar_u)
            )

        rm == :CVaR && @expression(model, risk, cvar_risk)
    end

    if rm == :EVaR || isfinite(evar_u)
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
            type == :Trad &&
            (
                obj == :Sharpe ? @constraint(model, evar_risk <= evar_u * model[:k]) :
                @constraint(model, evar_risk <= evar_u)
            )

        rm == :EVaR && @expression(model, risk, evar_risk)
    end

    !(rm == :RVaR || isfinite(rvar_u)) && (return nothing)

    ln_k = portfolio.ln_k
    opk = portfolio.opk
    omk = portfolio.omk
    invk2 = portfolio.invk2
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
        [z_rvar * opk * invk2, psi_rvar[i] * opk * invk, epsilon_rvar[i]] in
        MOI.PowerCone(invopk)
    )
    @constraint(
        model,
        [i = 1:T],
        [omega_rvar[i] * invomk, theta_rvar[i] * invk, -z_rvar * invk2] in
        MOI.PowerCone(omk)
    )
    @constraint(model, -model[:hist_ret] .- t_rvar .+ epsilon_rvar .+ omega_rvar .<= 0)
    @expression(model, rvar_risk, t_rvar + ln_k * z_rvar + sum(psi_rvar .+ theta_rvar))

    isfinite(rvar_u) &&
        type == :Trad &&
        (
            obj == :Sharpe ? @constraint(model, rvar_risk <= rvar_u * model[:k]) :
            @constraint(model, rvar_risk <= rvar_u)
        )

    rm == :RVaR && @expression(model, risk, rvar_risk)

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
        rm == :MDD ||
        rm == :ADD ||
        rm == :CDaR ||
        rm == :UCI ||
        rm == :EDaR ||
        rm == :RDaR ||
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

    if rm == :MDD || isfinite(mdd_u)
        @variable(model, mdd)
        @constraint(model, mdd .>= dd[2:end])
        @expression(model, mdd_risk, mdd)

        isfinite(mdd_u) &&
            type == :Trad &&
            (
                obj == :Sharpe ? @constraint(model, dd[2:end] .<= mdd_u * model[:k]) :
                @constraint(model, dd[2:end] .<= mdd_u)
            )

        rm == :MDD && @expression(model, risk, mdd_risk)
    end

    if rm == :ADD || isfinite(add_u)
        @expression(model, add_risk, sum(dd[2:end]) / T)

        isfinite(add_u) &&
            type == :Trad &&
            (
                obj == :Sharpe ? @constraint(model, add_risk .<= add_u * model[:k]) :
                @constraint(model, add_risk .<= add_u)
            )

        rm == :ADD && @expression(model, risk, add_risk)
    end

    if rm == :CDaR || isfinite(cdar_u)
        invat = portfolio.invat
        @variable(model, dar)
        @variable(model, z_dar[1:T] .>= 0)
        @constraint(model, z_dar .>= dd[2:end] .- dar)
        @expression(model, cdar_risk, dar + sum(z_dar) * invat)

        isfinite(cdar_u) &&
            type == :Trad &&
            (
                obj == :Sharpe ? @constraint(model, cdar_risk .<= cdar_u * model[:k]) :
                @constraint(model, cdar_risk .<= cdar_u)
            )

        rm == :CDaR && @expression(model, risk, cdar_risk)
    end

    if rm == :UCI || isfinite(uci_u)
        @variable(model, uci)
        @constraint(model, [uci; dd[2:end]] in SecondOrderCone())
        @expression(model, uci_risk, uci / sqrt(T))

        isfinite(uci_u) &&
            type == :Trad &&
            (
                obj == :Sharpe ? @constraint(model, uci_risk <= uci_u * model[:k]) :
                @constraint(model, uci_risk <= uci_u)
            )

        rm == :UCI && @expression(model, risk, uci_risk)
    end

    if rm == :EDaR || isfinite(edar_u)
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
            type == :Trad &&
            (
                obj == :Sharpe ? @constraint(model, edar_risk <= edar_u * model[:k]) :
                @constraint(model, edar_risk <= edar_u)
            )

        rm == :EDaR && @expression(model, risk, edar_risk)
    end

    !(rm == :RDaR || isfinite(rdar_u)) && (return nothing)

    ln_k = portfolio.ln_k
    opk = portfolio.opk
    omk = portfolio.omk
    invk2 = portfolio.invk2
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
        [z_rdar * opk * invk2, psi_rdar[i] * opk * invk, epsilon_rdar[i]] in
        MOI.PowerCone(invopk)
    )
    @constraint(
        model,
        [i = 1:T],
        [omega_rdar[i] * invomk, theta_rdar[i] * invk, -z_rdar * invk2] in
        MOI.PowerCone(omk)
    )
    @constraint(model, dd[2:end] .- t_rdar .+ epsilon_rdar .+ omega_rdar .<= 0)
    @expression(model, rdar_risk, t_rdar + ln_k * z_rdar + sum(psi_rdar .+ theta_rdar))

    isfinite(rdar_u) &&
        type == :Trad &&
        (
            obj == :Sharpe ? @constraint(model, rdar_risk <= rdar_u * model[:k]) :
            @constraint(model, rdar_risk <= rdar_u)
        )

    rm == :RDaR && @expression(model, risk, rdar_risk)

    return nothing
end

function _risk_setup(
    portfolio,
    type,
    rm,
    kelly,
    obj,
    rf,
    T,
    N,
    mu,
    returns,
    sigma,
    kurtosis,
    skurtosis,
)
    _mv_setup(portfolio, sigma, rm, kelly, obj, type)
    _mad_setup(portfolio, rm, T, returns, mu, obj, type)
    _lpm_setup(portfolio, rm, T, returns, obj, rf, type)
    _wr_setup(portfolio, rm, returns, obj, type)
    _var_setup(portfolio, rm, T, returns, obj, type)
    _drawdown_setup(portfolio, rm, T, returns, obj, type)
    _kurtosis_setup(portfolio, kurtosis, skurtosis, rm, N, obj, type)
    _owa_setup(portfolio, rm, T, returns, obj, type)

    return nothing
end

function block_vec_pq(A, p, q)
    mp, nq = size(A)

    !(mod(mp, p) == 0 && mod(nq, p) == 0) && (throw(
        DimensionMismatch(
            "size(A) = $(size(A)), must be integer multiples of (p, q) = ($p, $q)",
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
    kurt_u = portfolio.kurt_u
    skurt_u = portfolio.skurt_u

    !(rm == :Kurt || rm == :SKurt || isfinite(kurt_u) || isfinite(skurt_u)) &&
        (return nothing)

    model = portfolio.model

    if rm == :Kurt || isfinite(kurt_u)
        max_num_assets_kurt = portfolio.max_num_assets_kurt
        @variable(model, W[1:N, 1:N], Symmetric)
        @expression(model, M1, vcat(W, transpose(model[:w])))

        obj == :Sharpe ? @expression(model, M2, vcat(model[:w], model[:k])) :
        @expression(model, M2, vcat(model[:w], 1))

        @expression(model, M3, hcat(M1, M2))
        @constraint(model, M3 in PSDCone())

        @variable(model, t_kurt)
        if !iszero(max_num_assets_kurt) && N > max_num_assets_kurt
            N2 = 2 * N
            @variable(model, x_kurt[1:N2])
            @constraint(model, [t_kurt; x_kurt] in SecondOrderCone())

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

        isfinite(kurt_u) &&
            type == :Trad &&
            (
                obj == :Sharpe ? @constraint(model, kurt_risk <= kurt_u * model[:k]) :
                @constraint(model, kurt_risk <= kurt_u)
            )

        rm == :Kurt && @expression(model, risk, kurt_risk)
    end

    if rm == :SKurt || isfinite(skurt_u)
        max_num_assets_kurt = portfolio.max_num_assets_kurt
        @variable(model, SW[1:N, 1:N], Symmetric)
        @expression(model, SM1, vcat(SW, transpose(model[:w])))

        obj == :Sharpe ? @expression(model, SM2, vcat(model[:w], model[:k])) :
        @expression(model, SM2, vcat(model[:w], 1))

        @expression(model, SM3, hcat(SM1, SM2))
        @constraint(model, SM3 in PSDCone())

        @variable(model, t_skurt)
        if !iszero(max_num_assets_kurt) && N > max_num_assets_kurt
            N2 = 2 * N
            @variable(model, x_skurt[1:N2])

            @constraint(model, [t_skurt; x_skurt] in SecondOrderCone())

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

        isfinite(skurt_u) &&
            type == :Trad &&
            (
                obj == :Sharpe ? @constraint(model, skurt_risk <= skurt_u * model[:k]) :
                @constraint(model, skurt_risk <= skurt_u)
            )

        rm == :SKurt && @expression(model, risk, skurt_risk)
    end
end

function _owa_w_choice(owa_w, T)
    return isempty(owa_w) ? owa_gmd(T) / 2 : owa_w
end

function _owa_setup(portfolio, rm, T, returns, obj, type)
    gmd_u = portfolio.gmd_u
    rg_u = portfolio.rg_u
    tg_u = portfolio.tg_u
    rcvar_u = portfolio.rcvar_u
    rtg_u = portfolio.rtg_u
    owa_u = portfolio.owa_u

    !(
        rm == :GMD ||
        rm == :RG ||
        rm == :TG ||
        rm == :RCVaR ||
        rm == :RTG ||
        rm == :OWA ||
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

    if rm == :GMD || isfinite(gmd_u)
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
            type == :Trad &&
            (
                obj == :Sharpe ? @constraint(model, gmd_risk <= gmd_u * model[:k] * 0.5) :
                @constraint(model, gmd_risk <= gmd_u * 0.5)
            )

        rm == :GMD && @expression(model, risk, gmd_risk)
    end

    if rm == :RG || isfinite(rg_u)
        @variable(model, rga[1:T])
        @variable(model, rgb[1:T])
        @expression(model, rg_risk, sum(rga .+ rgb))
        rg_w = owa_rg(T)
        @constraint(
            model,
            owa * transpose(rg_w) .<= onesvec * transpose(rga) + rgb * transpose(onesvec)
        )

        isfinite(rg_u) &&
            type == :Trad &&
            (
                obj == :Sharpe ? @constraint(model, rg_risk <= rg_u * model[:k]) :
                @constraint(model, rg_risk <= rg_u)
            )

        rm == :RG && @expression(model, risk, rg_risk)
    end

    if rm == :RCVaR || isfinite(rcvar_u)
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
            type == :Trad &&
            (
                obj == :Sharpe ? @constraint(model, rcvar_risk <= rcvar_u * model[:k]) :
                @constraint(model, rcvar_risk <= rcvar_u)
            )

        rm == :RCVaR && @expression(model, risk, rcvar_risk)
    end

    if rm == :TG || isfinite(tg_u)
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
            type == :Trad &&
            (
                obj == :Sharpe ? @constraint(model, tg_risk <= tg_u * model[:k]) :
                @constraint(model, tg_risk <= tg_u)
            )

        rm == :TG && @expression(model, risk, tg_risk)
    end

    if rm == :RTG || isfinite(rtg_u)
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
            type == :Trad &&
            (
                obj == :Sharpe ? @constraint(model, rtg_risk <= rtg_u * model[:k]) :
                @constraint(model, rtg_risk <= rtg_u)
            )

        rm == :RTG && @expression(model, risk, rtg_risk)
    end

    !(rm == :OWA || isfinite(owa_u)) && (return nothing)

    @variable(model, owa_a[1:T])
    @variable(model, owa_b[1:T])
    @expression(model, owa_risk, sum(owa_a .+ owa_b))

    owa_w = _owa_w_choice(portfolio.owa_w, T)

    @constraint(
        model,
        owa * transpose(owa_w) .<= onesvec * transpose(owa_a) + owa_b * transpose(onesvec)
    )

    isfinite(owa_u) &&
        type == :Trad &&
        (
            obj == :Sharpe ? @constraint(model, owa_risk <= owa_u * model[:k]) :
            @constraint(model, owa_risk <= owa_u)
        )

    rm == :OWA && @expression(model, risk, owa_risk)

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
    if rrp_ver == :Reg || rrp_ver == :Reg_Pen
        @variable(model, rho)
        @constraint(model, [2 * psi; 2 * G * model[:w]; -2 * rho] in SecondOrderCone())
    end

    if rrp_ver == :None
        @constraint(model, [psi; G * model[:w]] in SecondOrderCone())
    elseif rrp_ver == :Reg
        @constraint(model, [rho; G * model[:w]] in SecondOrderCone())
    elseif rrp_ver == :Reg_Pen
        theta = Diagonal(sqrt.(diag(sigma)))
        @constraint(
            model,
            [rho; sqrt(rrp_penalty) * theta * model[:w]] in SecondOrderCone()
        )
    end

    return nothing
end

function _wc_setup(portfolio, kelly, obj, T, N, rf, mu, sigma, u_mu, u_cov)
    model = portfolio.model

    # Return uncertainy sets.
    (kelly == :Approx || (u_cov != :Box && u_cov != :Ellipse)) && _mv_risk(model, sigma)

    returns = portfolio.returns
    obj == :Sharpe ? _setup_sharpe_ret(kelly, model, T, rf, returns, mu, Inf, false) :
    _setup_ret(kelly, model, T, returns, mu, Inf)

    if haskey(model, :_ret)
        if u_mu == :Box
            d_mu = portfolio.d_mu
            @variable(model, abs_w[1:N])
            @constraint(model, [i = 1:N], [abs_w[i]; model[:w][i]] in MOI.NormOneCone(2))
            @expression(model, ret, model[:_ret] - dot(d_mu, abs_w))
        elseif u_mu == :Ellipse
            k_mu = portfolio.k_mu
            cov_mu = portfolio.cov_mu
            G = sqrt(cov_mu)
            @expression(model, x_gw, G * model[:w])
            @variable(model, t_gw)
            @constraint(model, [t_gw; x_gw] in SecondOrderCone())
            @expression(model, ret, model[:_ret] - k_mu * t_gw)
        else
            @expression(model, ret, model[:_ret])
        end
    end

    # Cov uncertainty sets.
    if u_cov == :Box
        cov_u = portfolio.cov_u
        cov_l = portfolio.cov_l
        @variable(model, Au[1:N, 1:N] .>= 0, Symmetric)
        @variable(model, Al[1:N, 1:N] .>= 0, Symmetric)
        @expression(model, M1, vcat(Au .- Al, transpose(model[:w])))

        obj == :Sharpe ? @expression(model, M2, vcat(model[:w], model[:k])) :
        @expression(model, M2, vcat(model[:w], 1))

        @expression(model, M3, hcat(M1, M2))
        @constraint(model, M3 in PSDCone())
        @expression(model, risk, tr(Au * cov_u) - tr(Al * cov_l))
    elseif u_cov == :Ellipse
        k_sigma = portfolio.k_sigma
        G_sigma = sqrt(portfolio.cov_sigma)

        @variable(model, E1[1:N, 1:N], Symmetric)
        @variable(model, E2[1:N, 1:N], Symmetric)
        @expression(model, M1, vcat(E1, transpose(model[:w])))

        obj == :Sharpe ? @expression(model, M2, vcat(model[:w], model[:k])) :
        @expression(model, M2, vcat(model[:w], 1))

        @expression(model, M3, hcat(M1, M2))

        @constraint(model, M3 in PSDCone())
        @constraint(model, E2 in PSDCone())

        @expression(model, E1_p_E2, E1 .+ E2)

        @expression(model, x_ge, G_sigma * vec(E1_p_E2))
        @variable(model, t_ge)
        @constraint(model, [t_ge; x_gw] in SecondOrderCone())

        @expression(model, risk, tr(sigma * E1_p_E2) + k_sigma * t_ge)
    else
        @expression(model, risk, model[:dev_risk])
    end

    obj == :Sharpe && (
        kelly != :None ? @constraint(model, model[:risk] <= 1) :
        @constraint(model, ret - rf * model[:k] >= 1)
    )
end

function _setup_sharpe_k(model, obj)
    obj == :Sharpe && @variable(model, k >= 0)
    return nothing
end

function _setup_risk_budget(portfolio)
    model = portfolio.model
    if isempty(portfolio.risk_budget)
        portfolio.risk_budget = ()
    else
        !isapprox(sum(portfolio.risk_budget), one(eltype(portfolio.risk_budget))) &&
            (portfolio.risk_budget .= portfolio.risk_budget)
    end
    @variable(model, k >= 0)
    return nothing
end

function _setup_ret(kelly, model, T, returns, mu, mu_l)
    if kelly == :Exact
        @variable(model, texact_kelly[1:T])
        @expression(model, _ret, sum(texact_kelly) / T)
        @expression(model, kret, 1 .+ returns * model[:w])
        @constraint(
            model,
            [i = 1:T],
            [texact_kelly[i], 1, kret[i]] in MOI.ExponentialCone()
        )
    elseif kelly == :Approx && !isempty(mu)
        @expression(model, _ret, dot(mu, model[:w]) - 0.5 * model[:dev_risk])
    elseif !isempty(mu)
        @expression(model, _ret, dot(mu, model[:w]))
    end

    !isinf(mu_l) && @constraint(model, _ret >= mu_l)

    return nothing
end

function _setup_sharpe_ret(kelly, model, T, rf, returns, mu, mu_l, trad = true)
    if kelly == :Exact
        @variable(model, texact_kelly[1:T])
        @expression(model, _ret, sum(texact_kelly) / T - rf * model[:k])
        @expression(model, kret, model[:k] .+ returns * model[:w])
        @constraint(
            model,
            [i = 1:T],
            [texact_kelly[i], model[:k], kret[i]] in MOI.ExponentialCone()
        )
        trad && @constraint(model, model[:risk] <= 1)
    elseif kelly == :Approx && !isempty(mu)
        @variable(model, tapprox_kelly)
        @constraint(
            model,
            [
                model[:k] + tapprox_kelly
                2 * model[:dev]
                model[:k] - tapprox_kelly
            ] in SecondOrderCone()
        )
        @expression(model, _ret, dot(mu, model[:w]) - 0.5 * tapprox_kelly)
        trad && @constraint(model, model[:risk] <= 1)
    elseif !isempty(mu)
        @expression(model, _ret, dot(mu, model[:w]))
        trad && @constraint(model, _ret - rf * model[:k] == 1)
    end

    !isinf(mu_l) && @constraint(model, _ret >= mu_l * model[:k])

    return nothing
end

function _setup_trad_return(portfolio, class, kelly, obj, T, rf, returns, mu)
    model = portfolio.model
    mu_l = portfolio.mu_l

    kelly = class == :Classic ? kelly : :None

    obj == :Sharpe ? _setup_sharpe_ret(kelly, model, T, rf, returns, mu, mu_l) :
    _setup_ret(kelly, model, T, returns, mu, mu_l)

    haskey(model, :_ret) && @expression(model, ret, model[:_ret])

    return nothing
end

function _setup_rp_rrp_return_and_obj(portfolio, kelly, T, returns, mu)
    model = portfolio.model
    mu_l = portfolio.mu_l
    _setup_ret(kelly, model, T, returns, mu, mu_l)
    haskey(model, :_ret) && @expression(model, ret, model[:_ret])

    @objective(model, Min, model[:risk])
    return nothing
end

function _setup_weights(portfolio, obj, N)
    max_number_assets = portfolio.max_number_assets
    factor = portfolio.max_number_assets_factor
    short = portfolio.short
    short_u = portfolio.short_u
    long_u = portfolio.long_u
    sum_short_long = portfolio.sum_short_long

    model = portfolio.model

    # Boolean variables max number of assets.
    if max_number_assets > 0
        if obj == :Sharpe
            @variable(model, tass_bin[1:N], binary = true)
            @variable(model, tass_bin_sharpe[1:N] >= 0)
        else
            @variable(model, tass_bin[1:N], binary = true)
        end
    end

    # Weight constraints.
    if obj == :Sharpe
        @constraint(model, sum(model[:w]) == sum_short_long * model[:k])

        # Maximum number of assets constraints.
        if max_number_assets > 0
            @constraint(model, sum(tass_bin) <= max_number_assets)
            @constraint(model, tass_bin_sharpe .<= model[:k])
            @constraint(model, tass_bin_sharpe .<= factor * tass_bin)
            @constraint(model, tass_bin_sharpe .>= model[:k] .- factor * (1 .- tass_bin))
            @constraint(model, model[:w] .<= long_u * tass_bin_sharpe)
        end

        if short == false
            @constraint(model, model[:w] .<= long_u * model[:k])
            @constraint(model, model[:w] .>= 0)
        else
            @variable(model, tw_ulong[1:N] .>= 0)
            @variable(model, tw_ushort[1:N] .>= 0)

            @constraint(model, sum(tw_ulong) <= long_u * model[:k])
            @constraint(model, sum(tw_ushort) <= short_u * model[:k])

            @constraint(model, model[:w] .<= tw_ulong)
            @constraint(model, model[:w] .>= -tw_ushort)

            # Maximum number of assets constraints.
            max_number_assets > 0 &&
                @constraint(model, model[:w] .>= -short_u * tass_bin_sharpe)
        end
    else
        @constraint(model, sum(model[:w]) == sum_short_long)

        # Maximum number of assets constraints.
        if max_number_assets > 0
            @constraint(model, sum(tass_bin) <= max_number_assets)
            @constraint(model, model[:w] .<= long_u * tass_bin)
        end

        if short == false
            @constraint(model, model[:w] .<= long_u)
            @constraint(model, model[:w] .>= 0)
        else
            @variable(model, tw_ulong[1:N] .>= 0)
            @variable(model, tw_ushort[1:N] .>= 0)

            @constraint(model, sum(tw_ulong) <= long_u)
            @constraint(model, sum(tw_ushort) <= short_u)

            @constraint(model, model[:w] .<= tw_ulong)
            @constraint(model, model[:w] .>= -tw_ushort)

            # Maximum number of assets constraints.
            max_number_assets > 0 && @constraint(model, model[:w] .>= -short_u * tass_bin)
        end
    end

    return nothing
end

function _setup_linear_constraints(portfolio, obj, type)
    A = portfolio.a_mtx_ineq
    B = portfolio.b_vec_ineq

    (isempty(A) || isempty(B)) && (return nothing)

    model = portfolio.model

    # Linear weight constraints.
    obj == :Sharpe || type == :RP ?
    @constraint(model, A * model[:w] .- B * model[:k] .>= 0) :
    @constraint(model, A * model[:w] .- B .>= 0)

    return nothing
end

function _setup_min_number_effective_assets(portfolio, obj)
    mnea = portfolio.min_number_effective_assets

    (mnea < 1) && (return nothing)

    model = portfolio.model

    @variable(model, tmnea >= 0)
    @constraint(model, [tmnea; model[:w]] in SecondOrderCone())

    obj == :Sharpe ? @constraint(model, tmnea * sqrt(mnea) <= model[:k]) :
    @constraint(model, tmnea * sqrt(mnea) <= 1)

    return nothing
end

function _setup_tracking_err(portfolio, returns, obj, T)
    tracking_err = portfolio.tracking_err

    isinf(tracking_err) && (return nothing)

    kind_tracking_err = portfolio.kind_tracking_err
    tracking_err_weights = portfolio.tracking_err_weights
    tracking_err_returns = portfolio.tracking_err_returns

    tracking_err_flag = false

    if kind_tracking_err == :Weights && !isempty(tracking_err_weights)
        benchmark = returns * tracking_err_weights
        tracking_err_flag = true
    elseif kind_tracking_err == :Returns && !isempty(tracking_err_returns)
        benchmark = tracking_err_returns
        tracking_err_flag = true
    end

    !(tracking_err_flag) && (return nothing)

    model = portfolio.model

    @variable(model, t_track_err >= 0)
    if obj == :Sharpe
        @expression(model, track_err, returns * model[:w] .- benchmark * model[:k])
        @constraint(model, [t_track_err; track_err] in SecondOrderCone())
        @constraint(model, t_track_err <= tracking_err * model[:k] * sqrt(T - 1))
    else
        @expression(model, track_err, returns * model[:w] .- benchmark)
        @constraint(model, [t_track_err; track_err] in SecondOrderCone())
        @constraint(model, t_track_err <= tracking_err * sqrt(T - 1))
    end

    return nothing
end

function _setup_turnover(portfolio, N, obj)
    turnover = portfolio.turnover
    turnover_weights = portfolio.turnover_weights

    (isinf(turnover) || isempty(turnover_weights)) && (return nothing)

    model = portfolio.model

    @variable(model, t_turnov[1:N] >= 0)
    if obj == :Sharpe
        @expression(model, turnov, model[:w] .- turnover_weights * model[:k])
        @constraint(model, [i = 1:N], [t_turnov[i]; turnov[i]] in MOI.NormOneCone(2))
        @constraint(model, t_turnov .<= turnover * model[:k])
    else
        @expression(model, turnov, model[:w] .- turnover_weights)
        @constraint(model, [i = 1:N], [t_turnov[i]; turnov[i]] in MOI.NormOneCone(2))
        @constraint(model, t_turnov .<= turnover)
    end

    return nothing
end

function _setup_trad_wc_constraints(portfolio, obj, T, N, type, class, kelly, l, returns)
    _setup_weights(portfolio, obj, N)
    _setup_min_number_effective_assets(portfolio, obj)
    _setup_tracking_err(portfolio, returns, obj, T)
    _setup_turnover(portfolio, N, obj)
    _setup_trad_wc_objective_function(portfolio, type, obj, class, kelly, l)

    return nothing
end

function _setup_trad_wc_objective_function(portfolio, type, obj, class, kelly, l)
    model = portfolio.model
    if obj == :Sharpe
        (type == :Trad && class == :Classic || type == :WC) && kelly != :None ?
        @objective(model, Max, model[:ret]) : @objective(model, Min, model[:risk])
    elseif obj == :Min_Risk
        @objective(model, Min, model[:risk])
    elseif obj == :Utility
        @objective(model, Max, model[:ret] - l * model[:risk])
    elseif obj == :Max_Ret
        @objective(model, Max, model[:ret])
    end
    return nothing
end

function _optimize_portfolio(portfolio, type, obj)
    solvers = portfolio.solvers
    model = portfolio.model

    N = size(portfolio.returns, 2)
    rtype = eltype(portfolio.returns)
    term_status = termination_status(model)
    solvers_tried = Dict()

    for (key, val) in solvers
        key = Symbol(String(key) * "_" * String(type))

        haskey(val, :solver) && set_optimizer(model, val[:solver])

        if haskey(val, :params)
            for (attribute, value) in val[:params]
                set_attribute(model, attribute, value)
            end
        end

        try
            JuMP.optimize!(model)
        catch jump_error
            push!(solvers_tried, key => Dict(:jump_error => jump_error))
            continue
        end

        term_status = termination_status(model)

        all_finite_weights = all(isfinite.(value.(model[:w])))
        all_non_zero_weights = all(abs.(0.0 .- value.(model[:w])) .> eps())

        term_status ∈ ValidTermination &&
            all_finite_weights &&
            all_non_zero_weights &&
            break

        weights = Vector{rtype}(undef, N)
        if type == :Trad || type == :WC
            if obj == :Sharpe
                val_k = value(model[:k])
                val_k = val_k > 0 ? val_k : 1
                weights .= value.(model[:w]) / val_k
            else
                weights .= value.(model[:w])
            end

            short = portfolio.short
            sum_short_long = portfolio.sum_short_long
            if short == false
                sum_w = sum(abs.(weights))
                sum_w = sum_w > eps() ? sum_w : 1
                weights .= abs.(weights) / sum_w * sum_short_long
            end
        elseif type == :RP || type == :RRP
            weights .= value.(model[:w])
            sum_w = sum(abs.(weights))
            sum_w = sum_w > eps() ? sum_w : 1
            weights .= abs.(weights) / sum_w
        end

        push!(
            solvers_tried,
            key => Dict(
                :objective_val => objective_value(model),
                :term_status => term_status,
                :params => haskey(val, :params) ? val[:params] : missing,
                :finite_weights => all_finite_weights,
                :nonzero_weights => all_non_zero_weights,
                :portfolio => DataFrame(tickers = portfolio.assets, weights = weights),
            ),
        )
    end

    isempty(solvers_tried) &&
        term_status ∉ ValidTermination &&
        push!(solvers_tried, :error => term_status)

    return term_status, solvers_tried
end

function _finalise_portfolio(portfolio, returns, N, solvers_tried, type, rm, obj)
    model = portfolio.model

    if (type == :Trad || type == :RP) && rm ∈ (:EVaR, :EDaR, :RVaR, :RDaR)
        z_key = "z_" * lowercase(string(rm))
        z_key2 = Symbol(string(type) * "_" * z_key)
        portfolio.z[z_key2] = value(portfolio.model[Symbol(z_key)])
        type == :Trad &&
            obj == :Sharpe &&
            (portfolio.z[z_key2] /= value(portfolio.model[:k]))
    end

    weights = Vector{eltype(returns)}(undef, N)
    if type == :Trad || type == :WC
        if obj == :Sharpe
            val_k = value(model[:k])
            val_k = val_k > 0 ? val_k : 1
            weights .= value.(model[:w]) / val_k
        else
            weights .= value.(model[:w])
        end

        short = portfolio.short
        sum_short_long = portfolio.sum_short_long
        if short == false
            sum_w = sum(abs.(weights))
            sum_w = sum_w > eps() ? sum_w : 1
            weights .= abs.(weights) / sum_w * sum_short_long
        end
    elseif type == :RP || type == :RRP
        weights .= value.(model[:w])
        sum_w = sum(abs.(weights))
        sum_w = sum_w > eps() ? sum_w : 1
        weights .= abs.(weights) / sum_w
    end

    portfolio.optimal[type] = DataFrame(tickers = portfolio.assets, weights = weights)

    isempty(solvers_tried) ? portfolio.fail = Dict() : portfolio.fail = solvers_tried

    return portfolio.optimal[type]
end

function _handle_errors_and_finalise(
    portfolio,
    term_status,
    returns,
    N,
    solvers_tried,
    type,
    rm,
    obj,
)
    retval =
        if term_status ∉ ValidTermination || any(.!isfinite.(value.(portfolio.model[:w])))
            funcname = "$(fullname(PortfolioOptimiser)[1]).$(nameof(PortfolioOptimiser.opt_port!))"
            @warn(
                "$funcname: model could not be optimised satisfactorily.\nSolvers: $solvers_tried."
            )
            portfolio.fail = solvers_tried
            DataFrame()
        else
            _finalise_portfolio(portfolio, returns, N, solvers_tried, type, rm, obj)
        end

    return retval
end

function _p_save_opt_params(
    portfolio,
    type,
    class,
    hist,
    rm,
    obj,
    kelly,
    rrp_ver,
    rf,
    l,
    rrp_penalty,
    u_mu,
    u_cov,
    string_names,
    save_opt_params,
)
    !save_opt_params && return nothing

    opt_params_dict = if type == :Trad
        Dict(
            :class => class,
            :rm => rm,
            :obj => obj,
            :kelly => kelly,
            :rf => rf,
            :l => l,
            :string_names => string_names,
            :hist => hist,
        )
    elseif type == :RP
        Dict(
            :class => class,
            :rm => rm,
            :obj => :Min_Risk,
            :kelly => (kelly == :Exact) ? :None : kelly,
            :rf => rf,
            :string_names => string_names,
            :hist => hist,
        )
    elseif type == :RRP
        Dict(
            :class => class,
            :rm => :SD,
            :obj => :Min_Risk,
            :kelly => kelly,
            :rrp_penalty => rrp_penalty,
            :rrp_ver => rrp_ver,
            :string_names => string_names,
            :hist => hist,
        )
    elseif type == :WC
        Dict(
            :rm => :SD,
            :obj => obj,
            :kelly => kelly,
            :rf => rf,
            :l => l,
            :u_mu => u_mu,
            :u_cov => u_cov,
            :string_names => string_names,
            :hist => hist,
        )
    end

    portfolio.opt_params[type] = opt_params_dict

    return nothing
end

function _setup_model_class(portfolio, class, hist)
    mu, sigma, returns = (
        Vector{eltype(portfolio.mu)}(undef, 0),
        Matrix{eltype(portfolio.cov)}(undef, 0, 0),
        Matrix{eltype(portfolio.returns)}(undef, 0, 0),
    )

    class != :Classic && @assert(hist ∈ BLHist, "hist = $hist, must be one of $BLHist")

    if class == :Classic
        mu = portfolio.mu
        sigma = portfolio.cov
        returns = portfolio.returns
    elseif class == :FM
        mu = portfolio.mu_fm
        if hist == 1
            sigma = portfolio.cov_fm
            returns = portfolio.returns_fm
        elseif hist == 2
            sigma = portfolio.cov
            returns = portfolio.returns
        else
            throw(AssertionError("for class = $class, hist = $hist can only be 1 or 2"))
        end
    elseif class == :BL
        mu = portfolio.mu_bl
        if hist == 1
            sigma = portfolio.cov_bl
        elseif hist == 2
            sigma = portfolio.cov
        else
            throw(AssertionError("for class = $class, hist = $hist can only be 1 or 2"))
        end
        returns = portfolio.returns
    elseif class == :BLFM
        mu = portfolio.mu_bl_fm
        if hist == 1
            sigma = portfolio.cov_bl_fm
            returns = portfolio.returns_fm
        elseif hist == 2
            sigma = portfolio.cov
            returns = portfolio.returns
        else
            sigma = portfolio.cov_fm
            returns = portfolio.returns_fm
        end
    end

    return mu, sigma, returns
end

"""
```julia
opt_port!(
    portfolio::Portfolio;
    class::Symbol = :Classic,
    hist::Integer = 1,
    kelly::Symbol = :None,
    l::Real = 2.0,
    obj::Symbol = :Sharpe,
    rf::Real = 0.0,
    rm::Symbol = :SD,
    rrp_penalty::Real = 1.0,
    rrp_ver::Symbol = :None,
    save_opt_params::Bool = true,
    string_names::Bool = false,
    type::Symbol = :Trad,
    u_cov::Symbol = :Box,
    u_mu::Symbol = :Box,
)
```
"""
function opt_port!(
    portfolio::Portfolio;
    class::Symbol = :Classic,
    hist::Integer = 1,
    kelly::Symbol = :None,
    l::Real = 2.0,
    obj::Symbol = :Sharpe,
    rf::Real = 0.0,
    rm::Symbol = :SD,
    rrp_penalty::Real = 1.0,
    rrp_ver::Symbol = :None,
    save_opt_params::Bool = true,
    string_names::Bool = false,
    type::Symbol = :Trad,
    u_cov::Symbol = :Box,
    u_mu::Symbol = :Box,
)
    @assert(type ∈ PortTypes, "type = $type, must be one of $PortTypes")
    @assert(class ∈ PortClasses, "class = $class, must be one of $PortClasses")
    @assert(rm ∈ RiskMeasures, "rm = $rm, must be one of $RiskMeasures")
    @assert(obj ∈ ObjFuncs, "obj = $obj, must be one of $ObjFuncs")
    @assert(kelly ∈ KellyRet, "kelly = $kelly, must be one of $KellyRet")
    @assert(rrp_ver ∈ RRPVersions)
    @assert(u_mu ∈ UncertaintyTypes, "u_mu = $u_mu, must be one of $UncertaintyTypes")
    @assert(u_cov ∈ UncertaintyTypes, "u_cov = $u_cov, must be one of $UncertaintyTypes")
    @assert(
        0 < portfolio.alpha < 1,
        "portfolio.alpha = $(portfolio.alpha), must be greater than 0 and less than 1"
    )
    @assert(
        0 < portfolio.kappa < 1,
        "portfolio.kappa = $(portfolio.kappa), must be greater than 0 and less than 1"
    )
    @assert(
        portfolio.kind_tracking_err ∈ TrackingErrKinds,
        "portfolio.kind_tracking_err = $(portfolio.kind_tracking_err), must be one of $TrackingErrKinds"
    )

    _p_save_opt_params(
        portfolio,
        type,
        class,
        hist,
        rm,
        obj,
        kelly,
        rrp_ver,
        rf,
        l,
        rrp_penalty,
        u_mu,
        u_cov,
        string_names,
        save_opt_params,
    )

    portfolio.model = JuMP.Model()

    # mu, sigma, returns
    mu, sigma, returns = _setup_model_class(portfolio, class, hist)
    T, N = size(returns)
    kurtosis = portfolio.kurt
    skurtosis = portfolio.skurt

    # Model variables.
    model = portfolio.model
    set_string_names_on_creation(model, string_names)
    @variable(model, w[1:N])

    if type == :Trad
        _setup_sharpe_k(model, obj)
        _risk_setup(
            portfolio,
            :Trad,
            rm,
            kelly,
            obj,
            rf,
            T,
            N,
            mu,
            returns,
            sigma,
            kurtosis,
            skurtosis,
        )
        _setup_trad_return(portfolio, class, kelly, obj, T, rf, returns, mu)
        _setup_trad_wc_constraints(portfolio, obj, T, N, :Trad, class, kelly, l, returns)
    elseif type == :RP
        _setup_risk_budget(portfolio)
        _rp_setup(portfolio, N)
        _risk_setup(
            portfolio,
            :RP,
            rm,
            kelly,
            obj,
            rf,
            T,
            N,
            mu,
            returns,
            sigma,
            kurtosis,
            skurtosis,
        )
        _setup_rp_rrp_return_and_obj(portfolio, kelly, T, returns, mu)
    elseif type == :RRP
        _setup_risk_budget(portfolio)
        _mv_setup(portfolio, sigma, rm, kelly, obj, :RRP)
        _rrp_setup(portfolio, sigma, N, rrp_ver, rrp_penalty)
        _setup_rp_rrp_return_and_obj(portfolio, kelly, T, returns, mu)
    else
        _setup_sharpe_k(model, obj)
        _wc_setup(portfolio, kelly, obj, T, N, rf, mu, sigma, u_mu, u_cov)
        _setup_trad_wc_constraints(portfolio, obj, T, N, :WC, class, kelly, l, returns)
    end

    _setup_linear_constraints(portfolio, obj, type)

    term_status, solvers_tried = _optimize_portfolio(portfolio, type, obj)

    retval = _handle_errors_and_finalise(
        portfolio,
        term_status,
        returns,
        N,
        solvers_tried,
        type,
        rm,
        obj,
    )

    return retval
end

"""
```julia
frontier_limits!(
    portfolio::Portfolio;
    class::Symbol = :Classic,
    hist::Integer = 1,
    kelly::Symbol = :None,
    rf::Real = 0.0,
    rm::Symbol = :SD,
)
```
"""
function frontier_limits!(
    portfolio::Portfolio;
    class::Symbol = :Classic,
    hist::Integer = 1,
    kelly::Symbol = :None,
    rf::Real = 0.0,
    rm::Symbol = :SD,
)
    optimal1 = deepcopy(portfolio.optimal)
    fail1 = deepcopy(portfolio.fail)

    w_min = opt_port!(
        portfolio;
        class = class,
        hist = hist,
        kelly = kelly,
        obj = :Min_Risk,
        rf = rf,
        rm = rm,
        save_opt_params = false,
    )

    w_max = opt_port!(
        portfolio;
        class = class,
        hist = hist,
        kelly = kelly,
        obj = :Max_Ret,
        rf = rf,
        rm = rm,
        save_opt_params = false,
    )

    limits = hcat(w_min, DataFrame(x1 = w_max[!, 2]))
    DataFrames.rename!(limits, :weights => :w_min, :x1 => :w_max)
    portfolio.limits[rm] = limits

    portfolio.optimal = optimal1
    portfolio.fail = fail1

    return portfolio.limits[rm]
end

"""
```julia
efficient_frontier!(
    portfolio::Portfolio;
    class::Symbol = :Classic,
    hist::Integer = 1,
    kelly::Symbol = :None,
    rf::Real = 0.0,
    rm::Symbol = :SD,
    points::Integer = 20,
)
```
"""
function efficient_frontier!(
    portfolio::Portfolio;
    class::Symbol = :Classic,
    hist::Integer = 1,
    kelly::Symbol = :None,
    rf::Real = 0.0,
    rm::Symbol = :SD,
    points::Integer = 20,
)
    optimal1 = deepcopy(portfolio.optimal)
    fail1 = deepcopy(portfolio.fail)

    mu, sigma, returns = _setup_model_class(portfolio, class, hist)

    fl = frontier_limits!(
        portfolio;
        class = class,
        hist = hist,
        kelly = kelly,
        rf = rf,
        rm = rm,
    )

    w1 = fl.w_min
    w2 = fl.w_max

    ret1 = dot(mu, w1)
    ret2 = dot(mu, w2)

    alpha_i = portfolio.alpha_i
    alpha = portfolio.alpha
    a_sim = portfolio.a_sim
    beta_i = portfolio.beta_i
    beta = portfolio.beta
    b_sim = portfolio.b_sim
    kappa = portfolio.kappa
    owa_w = portfolio.owa_w
    solvers = portfolio.solvers

    risk1, risk2 = _ul_risk(
        rm,
        returns,
        w1,
        w2,
        sigma,
        rf,
        solvers,
        alpha,
        kappa,
        alpha_i,
        beta,
        a_sim,
        beta_i,
        b_sim,
        owa_w,
        0,
    )

    mus = range(ret1, stop = ret2, length = points)
    risks = range(risk1, stop = risk2, length = points)

    ur = (
        SD = :dev_u,
        MAD = :mad_u,
        SSD = :sdev_u,
        FLPM = :flpm_u,
        SLPM = :slpm_u,
        WR = :wr_u,
        CVaR = :cvar_u,
        EVaR = :evar_u,
        RVaR = :rvar_u,
        MDD = :mdd_u,
        ADD = :add_u,
        CDaR = :cdar_u,
        UCI = :uci_u,
        EDaR = :edar_u,
        RDaR = :rdar_u,
        Kurt = :kurt_u,
        SKurt = :skurt_u,
        GMD = :gmd_u,
        RG = :rg_u,
        RCVaR = :rcvar_u,
        TG = :tg_u,
        RTG = :rtg_u,
        OWA = :owa_u,
    )

    rmf = ur[rm]

    frontier = Vector{typeof(risk1)}(undef, 0)
    srisk = Vector{typeof(risk1)}(undef, 0)

    i = 0
    for (j, (r, m)) in enumerate(zip(risks, mus))
        if i == 0
            w = opt_port!(
                portfolio;
                class = class,
                hist = hist,
                kelly = kelly,
                obj = :Min_Risk,
                rf = rf,
                rm = rm,
                save_opt_params = false,
            )
        else
            j != length(risks) ? setproperty!(portfolio, rmf, r) :
            setproperty!(portfolio, rmf, Inf)
            w = opt_port!(
                portfolio;
                class = class,
                hist = hist,
                kelly = kelly,
                obj = :Max_Ret,
                rf = rf,
                rm = rm,
                save_opt_params = false,
            )
            # Fallback in case :Max_Ret with maximum risk bounds fails.
            if isempty(w)
                setproperty!(portfolio, rmf, Inf)
                j != length(risks) ? portfolio.mu_l = m : portfolio.mu_l = Inf
                w = opt_port!(
                    portfolio;
                    class = class,
                    hist = hist,
                    kelly = kelly,
                    obj = :Min_Risk,
                    rf = rf,
                    rm = rm,
                    save_opt_params = false,
                )
            end
        end
        isempty(w) && continue
        rk = calc_risk(
            w.weights,
            returns;
            rm = rm,
            rf = rf,
            sigma = sigma,
            alpha_i = alpha_i,
            alpha = alpha,
            a_sim = a_sim,
            beta_i = beta_i,
            beta = beta,
            b_sim = b_sim,
            kappa = kappa,
            owa_w = owa_w,
            solvers = solvers,
        )
        append!(frontier, w.weights)
        push!(srisk, rk)
        i += 1
    end
    setproperty!(portfolio, rmf, Inf)

    w = opt_port!(
        portfolio;
        class = class,
        hist = hist,
        kelly = kelly,
        obj = :Sharpe,
        rf = rf,
        rm = rm,
        save_opt_params = false,
    )
    sharpe = false
    if !isempty(w)
        rk = calc_risk(
            w.weights,
            returns;
            rm = rm,
            rf = rf,
            sigma = sigma,
            alpha_i = alpha_i,
            alpha = alpha,
            a_sim = a_sim,
            beta_i = beta_i,
            beta = beta,
            b_sim = b_sim,
            kappa = kappa,
            owa_w = owa_w,
            solvers = solvers,
        )
        append!(frontier, w.weights)
        push!(srisk, rk)
        i += 1
        sharpe = true
    end

    portfolio.frontier[rm] = Dict(
        :weights => hcat(
            DataFrame(tickers = portfolio.assets),
            DataFrame(reshape(frontier, length(w1), :), string.(range(1, i))),
        ),
        :class => class,
        :hist => hist,
        :kelly => kelly,
        :rf => rf,
        :points => points,
        :risk => srisk,
        :sharpe => sharpe,
    )

    portfolio.optimal = optimal1
    portfolio.fail = fail1

    return portfolio.frontier
end
