function _sdp_setup(portfolio, obj, rm, type, N)
    if type == :RP || type == :RRP || portfolio.network_method != :SDP
        return nothing
    end

    model = portfolio.model
    network_penalty = portfolio.network_penalty
    sd_u = portfolio.sd_u
    network_sdp = portfolio.network_sdp

    @variable(model, Wn[1:N, 1:N], Symmetric)
    @expression(model, M1n, vcat(Wn, transpose(model[:w])))

    if obj == :Sharpe
        @expression(model, M2n, vcat(model[:w], model[:k]))
    else
        @expression(model, M2n, vcat(model[:w], 1))
    end

    @expression(model, M3n, hcat(M1n, M2n))
    @constraint(model, M3n in PSDCone())

    if type == :Trad && rm != :SD && isinf(sd_u)
        @expression(model, penalty_factor, network_penalty * tr(model[:Wn]))
    end

    @constraint(model, network_sdp .* model[:Wn] .== 0)

    return nothing
end

function _mv_risk(model, sigma, network_method)
    if network_method != :SDP
        G = sqrt(sigma)
        @variable(model, dev)
        @constraint(model, [dev; G * model[:w]] in SecondOrderCone())
        @expression(model, dev_risk, dev * dev)
    else
        @expression(model, dev_risk, tr(sigma * model[:Wn]))
    end
    return nothing
end

function _mv_setup(portfolio, sigma, rm, kelly, obj, type, network_method)
    sd_u = portfolio.sd_u

    if !(rm == :SD || kelly == :Approx || isfinite(sd_u))
        return nothing
    end

    model = portfolio.model

    _mv_risk(model, sigma, network_method)

    if isfinite(sd_u) && type == :Trad
        if obj == :Sharpe
            @constraint(model, model[:dev] <= sd_u * model[:k])
        else
            @constraint(model, model[:dev] <= sd_u)
        end
    end

    if rm == :SD && type != :RRP
        @expression(model, risk, model[:dev_risk])
    end

    return nothing
end

function _mad_setup(portfolio, rm, T, returns, mu, obj, type)
    mad_u = portfolio.mad_u
    ssd_u = portfolio.ssd_u

    if !(rm == :MAD || rm == :SSD || isfinite(mad_u) || isfinite(ssd_u))
        return nothing
    end

    model = portfolio.model
    msv_target = portfolio.msv_target

    abs_dev = if (isa(msv_target, Real) && isinf(msv_target)) ||
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

        if isfinite(mad_u) && type == :Trad
            if obj == :Sharpe
                @constraint(model, mad_risk <= 0.5 * mad_u * model[:k])
            else
                @constraint(model, mad_risk <= 0.5 * mad_u)
            end
        end

        if rm == :MAD
            @expression(model, risk, mad_risk)
        end
    end

    if !(rm == :SSD || isfinite(ssd_u))
        return nothing
    end

    @variable(model, sdev)
    @constraint(model, [sdev; mad] in SecondOrderCone())

    @expression(model, sdev_risk, sdev / sqrt(T - 1))

    if isfinite(ssd_u) && type == :Trad
        if obj == :Sharpe
            @constraint(model, sdev_risk <= ssd_u * model[:k])
        else
            @constraint(model, sdev_risk <= ssd_u)
        end
    end

    if rm == :SSD
        @expression(model, risk, sdev_risk)
    end

    return nothing
end

function _lpm_setup(portfolio, rm, T, returns, obj, rf, type)
    flpm_u = portfolio.flpm_u
    slpm_u = portfolio.slpm_u

    if !(rm == :FLPM || rm == :SLPM || isfinite(flpm_u) || isfinite(slpm_u))
        return nothing
    end

    model = portfolio.model

    lpm_target = portfolio.lpm_target

    lpm_t = if (isa(lpm_target, Real) && isinf(lpm_target)) ||
               (isa(lpm_target, AbstractVector) && isempty(lpm_target))
        rf
    elseif isa(lpm_target, Real) && isfinite(lpm_target)
        lpm_target
    else
        transpose(lpm_target)
    end

    @variable(model, lpm[1:T] .>= 0)
    if !haskey(model, :hist_ret)
        @expression(model, hist_ret, returns * model[:w])
    end

    if obj == :Sharpe || type == :RP
        @constraint(model, lpm .>= lpm_t * model[:k] .- model[:hist_ret])
    else
        @constraint(model, lpm .>= lpm_t .- model[:hist_ret])
    end

    if rm == :FLPM || isfinite(flpm_u)
        @expression(model, flpm_risk, sum(lpm) / T)

        if isfinite(flpm_u) && type == :Trad
            if obj == :Sharpe
                @constraint(model, flpm_risk <= flpm_u * model[:k])
            else
                @constraint(model, flpm_risk <= flpm_u)
            end
        end

        if rm == :FLPM
            @expression(model, risk, flpm_risk)
        end
    end

    if !(rm == :SLPM || isfinite(slpm_u))
        return nothing
    end

    @variable(model, slpm)
    @constraint(model, [slpm; lpm] in SecondOrderCone())
    @expression(model, slpm_risk, slpm / sqrt(T - 1))

    if isfinite(slpm_u) && type == :Trad
        if obj == :Sharpe
            @constraint(model, slpm_risk <= slpm_u * model[:k])
        else
            @constraint(model, slpm_risk <= slpm_u)
        end
    end

    if rm == :SLPM
        @expression(model, risk, slpm_risk)
    end

    return nothing
end

function _wr_setup(portfolio, rm, returns, obj, type)
    wr_u = portfolio.wr_u

    if !(rm == :WR || isfinite(wr_u))
        return nothing
    end

    model = portfolio.model

    @variable(model, wr)
    if !haskey(model, :hist_ret)
        @expression(model, hist_ret, returns * model[:w])
    end
    @constraint(model, -model[:hist_ret] .<= wr)
    @expression(model, wr_risk, wr)

    if isfinite(wr_u) && type == :Trad
        if obj == :Sharpe
            @constraint(model, -model[:hist_ret] .<= wr_u * model[:k])
        else
            @constraint(model, -model[:hist_ret] .<= wr_u)
        end
    end

    if rm == :WR
        @expression(model, risk, wr_risk)
    end

    return nothing
end

function _var_setup(portfolio, rm, T, returns, obj, type)
    cvar_u = portfolio.cvar_u
    evar_u = portfolio.evar_u
    rvar_u = portfolio.rvar_u

    if !(rm == :CVaR ||
         rm == :EVaR ||
         rm == :RVaR ||
         isfinite(evar_u) ||
         isfinite(cvar_u) ||
         isfinite(rvar_u))
        return nothing
    end

    model = portfolio.model

    if !haskey(model, :hist_ret)
        @expression(model, hist_ret, returns * model[:w])
    end

    if rm == :CVaR || isfinite(cvar_u)
        invat = portfolio.invat
        @variable(model, var)
        @variable(model, z_var[1:T] >= 0)
        @constraint(model, z_var .>= -model[:hist_ret] .- var)
        @expression(model, cvar_risk, var + sum(z_var) * invat)

        if isfinite(cvar_u) && type == :Trad
            if obj == :Sharpe
                @constraint(model, cvar_risk <= cvar_u * model[:k])
            else
                @constraint(model, cvar_risk <= cvar_u)
            end
        end

        if rm == :CVaR
            @expression(model, risk, cvar_risk)
        end
    end

    if rm == :EVaR || isfinite(evar_u)
        at = portfolio.at
        @variable(model, t_evar)
        @variable(model, z_evar >= 0)
        @variable(model, u_evar[1:T])
        @constraint(model, sum(u_evar) <= z_evar)
        @constraint(model, [i = 1:T],
                    [-model[:hist_ret][i] - t_evar, z_evar, u_evar[i]] in
                    MOI.ExponentialCone())
        @expression(model, evar_risk, t_evar - z_evar * log(at))

        if isfinite(evar_u) && type == :Trad
            if obj == :Sharpe
                @constraint(model, evar_risk <= evar_u * model[:k])
            else
                @constraint(model, evar_risk <= evar_u)
            end
        end

        if rm == :EVaR
            @expression(model, risk, evar_risk)
        end
    end

    if !(rm == :RVaR || isfinite(rvar_u))
        return nothing
    end

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
    @constraint(model, [i = 1:T],
                [z_rvar * opk * invk2, psi_rvar[i] * opk * invk, epsilon_rvar[i]] in
                MOI.PowerCone(invopk))
    @constraint(model, [i = 1:T],
                [omega_rvar[i] * invomk, theta_rvar[i] * invk, -z_rvar * invk2] in
                MOI.PowerCone(omk))
    @constraint(model, -model[:hist_ret] .- t_rvar .+ epsilon_rvar .+ omega_rvar .<= 0)
    @expression(model, rvar_risk, t_rvar + ln_k * z_rvar + sum(psi_rvar .+ theta_rvar))

    if isfinite(rvar_u) && type == :Trad
        if obj == :Sharpe
            @constraint(model, rvar_risk <= rvar_u * model[:k])
        else
            @constraint(model, rvar_risk <= rvar_u)
        end
    end

    if rm == :RVaR
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

    if !(rm == :MDD ||
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
         isfinite(rdar_u))
        return nothing
    end

    model = portfolio.model

    @variable(model, dd[1:(T + 1)])
    if !haskey(model, :hist_ret)
        @expression(model, hist_ret, returns * model[:w])
    end
    @constraint(model, dd[2:end] .>= dd[1:(end - 1)] .- model[:hist_ret])
    @constraint(model, dd[2:end] .>= 0)
    @constraint(model, dd[1] == 0)

    if rm == :MDD || isfinite(mdd_u)
        @variable(model, mdd)
        @constraint(model, mdd .>= dd[2:end])
        @expression(model, mdd_risk, mdd)

        if isfinite(mdd_u) && type == :Trad
            if obj == :Sharpe
                @constraint(model, dd[2:end] .<= mdd_u * model[:k])
            else
                @constraint(model, dd[2:end] .<= mdd_u)
            end
        end

        if rm == :MDD
            @expression(model, risk, mdd_risk)
        end
    end

    if rm == :ADD || isfinite(add_u)
        @expression(model, add_risk, sum(dd[2:end]) / T)

        if isfinite(add_u) && type == :Trad
            if obj == :Sharpe
                @constraint(model, add_risk .<= add_u * model[:k])
            else
                @constraint(model, add_risk .<= add_u)
            end
        end

        if rm == :ADD
            @expression(model, risk, add_risk)
        end
    end

    if rm == :CDaR || isfinite(cdar_u)
        invat = portfolio.invat
        @variable(model, dar)
        @variable(model, z_dar[1:T] .>= 0)
        @constraint(model, z_dar .>= dd[2:end] .- dar)
        @expression(model, cdar_risk, dar + sum(z_dar) * invat)

        if isfinite(cdar_u) && type == :Trad
            if obj == :Sharpe
                @constraint(model, cdar_risk .<= cdar_u * model[:k])
            else
                @constraint(model, cdar_risk .<= cdar_u)
            end
        end

        if rm == :CDaR
            @expression(model, risk, cdar_risk)
        end
    end

    if rm == :UCI || isfinite(uci_u)
        @variable(model, uci)
        @constraint(model, [uci; dd[2:end]] in SecondOrderCone())
        @expression(model, uci_risk, uci / sqrt(T))

        if isfinite(uci_u) && type == :Trad
            if obj == :Sharpe
                @constraint(model, uci_risk <= uci_u * model[:k])
            else
                @constraint(model, uci_risk <= uci_u)
            end
        end

        if rm == :UCI
            @expression(model, risk, uci_risk)
        end
    end

    if rm == :EDaR || isfinite(edar_u)
        at = portfolio.at
        @variable(model, t_edar)
        @variable(model, z_edar >= 0)
        @variable(model, u_edar[1:T])
        @constraint(model, sum(u_edar) <= z_edar)
        @constraint(model, [i = 1:T],
                    [dd[i + 1] - t_edar, z_edar, u_edar[i]] in MOI.ExponentialCone())
        @expression(model, edar_risk, t_edar - z_edar * log(at))

        if isfinite(edar_u) && type == :Trad
            if obj == :Sharpe
                @constraint(model, edar_risk <= edar_u * model[:k])
            else
                @constraint(model, edar_risk <= edar_u)
            end
        end

        if rm == :EDaR
            @expression(model, risk, edar_risk)
        end
    end

    if !(rm == :RDaR || isfinite(rdar_u))
        return nothing
    end

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
    @constraint(model, [i = 1:T],
                [z_rdar * opk * invk2, psi_rdar[i] * opk * invk, epsilon_rdar[i]] in
                MOI.PowerCone(invopk))
    @constraint(model, [i = 1:T],
                [omega_rdar[i] * invomk, theta_rdar[i] * invk, -z_rdar * invk2] in
                MOI.PowerCone(omk))
    @constraint(model, dd[2:end] .- t_rdar .+ epsilon_rdar .+ omega_rdar .<= 0)
    @expression(model, rdar_risk, t_rdar + ln_k * z_rdar + sum(psi_rdar .+ theta_rdar))

    if isfinite(rdar_u) && type == :Trad
        if obj == :Sharpe
            @constraint(model, rdar_risk <= rdar_u * model[:k])
        else
            @constraint(model, rdar_risk <= rdar_u)
        end
    end

    if rm == :RDaR
        @expression(model, risk, rdar_risk)
    end

    return nothing
end

function _risk_setup(portfolio, type, rm, kelly, obj, rf, T, N, mu, returns, sigma,
                     kurtosis, skurtosis, network_method)
    _sdp_setup(portfolio, obj, rm, type, N)
    _mv_setup(portfolio, sigma, rm, kelly, obj, type, network_method)
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

    if !(mod(mp, p) == 0 && mod(nq, p) == 0)
        throw(DimensionMismatch("size(A) = $(size(A)), must be integer multiples of (p, q) = ($p, $q)"))
    end

    m = Int(mp / p)
    n = Int(nq / q)

    A_vec = Matrix{eltype(A)}(undef, m * n, p * q)
    for j ∈ 0:(n - 1)
        Aj = Matrix{eltype(A)}(undef, m, p * q)
        for i ∈ 0:(m - 1)
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

    if !(rm == :Kurt || rm == :SKurt || isfinite(kurt_u) || isfinite(skurt_u))
        return nothing
    end

    model = portfolio.model

    if rm == :Kurt || isfinite(kurt_u)
        max_num_assets_kurt = portfolio.max_num_assets_kurt
        @variable(model, W[1:N, 1:N], Symmetric)
        @expression(model, M1, vcat(W, transpose(model[:w])))

        if obj == :Sharpe
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
            @constraint(model, [t_kurt; x_kurt] in SecondOrderCone())

            A = block_vec_pq(kurtosis, N, N)
            vals_A, vecs_A = eigen(A)
            vals_A = clamp.(real.(vals_A), 0, Inf) .+ clamp.(imag.(vals_A), 0, Inf)im
            Bi = Vector{Matrix{eltype(kurtosis)}}(undef, N2)
            for i ∈ 1:N2
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

        if isfinite(kurt_u) && type == :Trad
            if obj == :Sharpe
                @constraint(model, kurt_risk <= kurt_u * model[:k])
            else
                @constraint(model, kurt_risk <= kurt_u)
            end
        end

        if rm == :Kurt
            @expression(model, risk, kurt_risk)
        end
    end

    if rm == :SKurt || isfinite(skurt_u)
        max_num_assets_kurt = portfolio.max_num_assets_kurt
        @variable(model, SW[1:N, 1:N], Symmetric)
        @expression(model, SM1, vcat(SW, transpose(model[:w])))

        if obj == :Sharpe
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

            @constraint(model, [t_skurt; x_skurt] in SecondOrderCone())

            A = block_vec_pq(skurtosis, N, N)
            vals_A, vecs_A = eigen(A)
            vals_A = clamp.(real.(vals_A), 0, Inf) .+ clamp.(imag.(vals_A), 0, Inf)im
            SBi = Vector{Matrix{eltype(skurtosis)}}(undef, N2)
            for i ∈ 1:N2
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

        if isfinite(skurt_u) && type == :Trad
            if obj == :Sharpe
                @constraint(model, skurt_risk <= skurt_u * model[:k])
            else
                @constraint(model, skurt_risk <= skurt_u)
            end
        end

        if rm == :SKurt
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

    if !(rm == :GMD ||
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
         isfinite(owa_u))
        return nothing
    end

    onesvec = range(1; stop = 1, length = T)
    model = portfolio.model

    if !haskey(model, :hist_ret)
        @expression(model, hist_ret, returns * model[:w])
    end
    @variable(model, owa[1:T])
    @constraint(model, model[:hist_ret] == owa)

    if rm == :GMD || isfinite(gmd_u)
        @variable(model, gmda[1:T])
        @variable(model, gmdb[1:T])
        @expression(model, gmd_risk, sum(gmda .+ gmdb))
        gmd_w = owa_gmd(T)
        @constraint(model,
                    owa * transpose(gmd_w) .<=
                    onesvec * transpose(gmda) + gmdb * transpose(onesvec))

        if isfinite(gmd_u) && type == :Trad
            if obj == :Sharpe
                @constraint(model, gmd_risk <= gmd_u * model[:k])
            else
                @constraint(model, gmd_risk <= gmd_u)
            end
        end

        if rm == :GMD
            @expression(model, risk, gmd_risk)
        end
    end

    if rm == :RG || isfinite(rg_u)
        @variable(model, rga[1:T])
        @variable(model, rgb[1:T])
        @expression(model, rg_risk, sum(rga .+ rgb))
        rg_w = owa_rg(T)
        @constraint(model,
                    owa * transpose(rg_w) .<=
                    onesvec * transpose(rga) + rgb * transpose(onesvec))

        if isfinite(rg_u) && type == :Trad
            if obj == :Sharpe
                @constraint(model, rg_risk <= rg_u * model[:k])
            else
                @constraint(model, rg_risk <= rg_u)
            end
        end

        if rm == :RG
            @expression(model, risk, rg_risk)
        end
    end

    if rm == :RCVaR || isfinite(rcvar_u)
        alpha = portfolio.alpha
        beta = portfolio.beta

        @variable(model, rcvara[1:T])
        @variable(model, rcvarb[1:T])
        @expression(model, rcvar_risk, sum(rcvara .+ rcvarb))
        rcvar_w = owa_rcvar(T; alpha = alpha, beta = beta)
        @constraint(model,
                    owa * transpose(rcvar_w) .<=
                    onesvec * transpose(rcvara) + rcvarb * transpose(onesvec))

        if isfinite(rcvar_u) && type == :Trad
            if obj == :Sharpe
                @constraint(model, rcvar_risk <= rcvar_u * model[:k])
            else
                @constraint(model, rcvar_risk <= rcvar_u)
            end
        end

        if rm == :RCVaR
            @expression(model, risk, rcvar_risk)
        end
    end

    if rm == :TG || isfinite(tg_u)
        alpha = portfolio.alpha
        a_sim = portfolio.a_sim
        alpha_i = portfolio.alpha_i

        @variable(model, tga[1:T])
        @variable(model, tgb[1:T])
        @expression(model, tg_risk, sum(tga .+ tgb))
        tg_w = owa_tg(T; alpha_i = alpha_i, alpha = alpha, a_sim = a_sim)
        @constraint(model,
                    owa * transpose(tg_w) .<=
                    onesvec * transpose(tga) + tgb * transpose(onesvec))

        if isfinite(tg_u) && type == :Trad
            if obj == :Sharpe
                @constraint(model, tg_risk <= tg_u * model[:k])
            else
                @constraint(model, tg_risk <= tg_u)
            end
        end

        if rm == :TG
            @expression(model, risk, tg_risk)
        end
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
        rtg_w = owa_rtg(T; alpha_i = alpha_i, alpha = alpha, a_sim = a_sim, beta_i = beta_i,
                        beta = beta, b_sim = b_sim)
        @constraint(model,
                    owa * transpose(rtg_w) .<=
                    onesvec * transpose(rtga) + rtgb * transpose(onesvec))

        if isfinite(rtg_u) && type == :Trad
            if obj == :Sharpe
                @constraint(model, rtg_risk <= rtg_u * model[:k])
            else
                @constraint(model, rtg_risk <= rtg_u)
            end
        end

        if rm == :RTG
            @expression(model, risk, rtg_risk)
        end
    end

    if !(rm == :OWA || isfinite(owa_u))
        return nothing
    end

    @variable(model, owa_a[1:T])
    @variable(model, owa_b[1:T])
    @expression(model, owa_risk, sum(owa_a .+ owa_b))

    owa_w = isempty(portfolio.owa_w) ? owa_gmd(T) : portfolio.owa_w

    @constraint(model,
                owa * transpose(owa_w) .<=
                onesvec * transpose(owa_a) + owa_b * transpose(onesvec))

    if isfinite(owa_u) && type == :Trad
        if obj == :Sharpe
            @constraint(model, owa_risk <= owa_u * model[:k])
        else
            @constraint(model, owa_risk <= owa_u)
        end
    end

    if rm == :OWA
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

    return nothing
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
    @constraint(model, [i = 1:N],
                [model[:w][i] + zeta[i]
                 2 * gamma * sqrt(rb[i])
                 model[:w][i] - zeta[i]] in SecondOrderCone())
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
        @constraint(model,
                    [rho; sqrt(rrp_penalty) * theta * model[:w]] in SecondOrderCone())
    end

    return nothing
end

function _wc_setup(portfolio, kelly, obj, T, N, rf, mu, sigma, u_mu, u_cov, network_method)
    model = portfolio.model

    # Return uncertainy sets.
    if kelly == :Approx || (u_cov != :Box && u_cov != :Ellipse)
        _mv_risk(model, sigma, network_method)
    end

    returns = portfolio.returns
    if obj == :Sharpe
        _setup_sharpe_ret(kelly, model, T, rf, returns, mu, Inf, false)
    else
        _setup_ret(kelly, model, T, returns, mu, Inf)
    end

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

        if obj == :Sharpe
            @expression(model, M2, vcat(model[:w], model[:k]))
        else
            @expression(model, M2, vcat(model[:w], 1))
        end

        @expression(model, M3, hcat(M1, M2))
        @constraint(model, M3 in PSDCone())
        @expression(model, risk, tr(Au * cov_u) - tr(Al * cov_l))
    elseif u_cov == :Ellipse
        k_sigma = portfolio.k_sigma
        G_sigma = sqrt(portfolio.cov_sigma)

        @variable(model, E1[1:N, 1:N], Symmetric)
        @variable(model, E2[1:N, 1:N], Symmetric)
        @expression(model, M1, vcat(E1, transpose(model[:w])))

        if obj == :Sharpe
            @expression(model, M2, vcat(model[:w], model[:k]))
        else
            @expression(model, M2, vcat(model[:w], 1))
        end

        @expression(model, M3, hcat(M1, M2))

        @constraint(model, M3 in PSDCone())
        @constraint(model, E2 in PSDCone())

        @expression(model, E1_p_E2, E1 .+ E2)

        @expression(model, x_ge, G_sigma * vec(E1_p_E2))
        @variable(model, t_ge)
        @constraint(model, [t_ge; x_ge] in SecondOrderCone())

        @expression(model, risk, tr(sigma * E1_p_E2) + k_sigma * t_ge)
    else
        @expression(model, risk, model[:dev_risk])
    end

    if obj == :Sharpe
        if kelly != :None
            @constraint(model, model[:risk] <= 1)
        else
            @constraint(model, ret - rf * model[:k] >= 1)
        end
    end

    return nothing
end

function _setup_sharpe_k(model, obj)
    if obj == :Sharpe
        @variable(model, k >= 0)
    end
    return nothing
end

function _setup_risk_budget(portfolio)
    model = portfolio.model
    if isempty(portfolio.risk_budget)
        portfolio.risk_budget = ()
    else
        if !isapprox(sum(portfolio.risk_budget), one(eltype(portfolio.risk_budget)))
            portfolio.risk_budget .= portfolio.risk_budget
        end
    end
    @variable(model, k >= 0)
    return nothing
end

function _setup_ret(kelly, model, T, returns, mu, mu_l)
    if kelly == :Exact
        @variable(model, texact_kelly[1:T])
        @expression(model, _ret, sum(texact_kelly) / T)
        @expression(model, kret, 1 .+ returns * model[:w])
        @constraint(model, [i = 1:T],
                    [texact_kelly[i], 1, kret[i]] in MOI.ExponentialCone())
    elseif kelly == :Approx && !isempty(mu)
        @expression(model, _ret, dot(mu, model[:w]) - 0.5 * model[:dev_risk])
    elseif !isempty(mu)
        @expression(model, _ret, dot(mu, model[:w]))
    end

    if !isinf(mu_l)
        @constraint(model, _ret >= mu_l)
    end

    return nothing
end

function _setup_sharpe_ret(kelly, model, T, rf, returns, mu, mu_l, trad = true)
    if kelly == :Exact
        @variable(model, texact_kelly[1:T])
        @expression(model, _ret, sum(texact_kelly) / T - rf * model[:k])
        @expression(model, kret, model[:k] .+ returns * model[:w])
        @constraint(model, [i = 1:T],
                    [texact_kelly[i], model[:k], kret[i]] in MOI.ExponentialCone())
        if trad
            @constraint(model, model[:risk] <= 1)
        end
    elseif kelly == :Approx && !isempty(mu)
        @variable(model, tapprox_kelly)
        @constraint(model,
                    [model[:k] + tapprox_kelly
                     2 * model[:dev]
                     model[:k] - tapprox_kelly] in SecondOrderCone())
        @expression(model, _ret, dot(mu, model[:w]) - 0.5 * tapprox_kelly)
        if trad
            @constraint(model, model[:risk] <= 1)
        end
    elseif !isempty(mu)
        @expression(model, _ret, dot(mu, model[:w]))
        if trad
            @constraint(model, _ret - rf * model[:k] == 1)
        end
    end

    if !isinf(mu_l)
        @constraint(model, _ret >= mu_l * model[:k])
    end

    return nothing
end

function _setup_trad_return(portfolio, class, kelly, obj, T, rf, returns, mu)
    model = portfolio.model
    mu_l = portfolio.mu_l

    kelly = class == :Classic ? kelly : :None

    if obj == :Sharpe
        _setup_sharpe_ret(kelly, model, T, rf, returns, mu, mu_l)
    else
        _setup_ret(kelly, model, T, returns, mu, mu_l)
    end

    if haskey(model, :_ret)
        @expression(model, ret, model[:_ret])
    end

    return nothing
end

function _setup_rp_rrp_return_and_obj(portfolio, kelly, T, returns, mu)
    model = portfolio.model
    mu_l = portfolio.mu_l
    _setup_ret(kelly, model, T, returns, mu, mu_l)
    if haskey(model, :_ret)
        @expression(model, ret, model[:_ret])
    end

    @objective(model, Min, model[:risk])
    return nothing
end

function _setup_weights(portfolio, obj, N)
    max_number_assets = portfolio.max_number_assets
    factor_mna = portfolio.max_number_assets_factor
    factor_nip = portfolio.network_ip_factor
    short = portfolio.short
    short_u = portfolio.short_u
    long_u = portfolio.long_u
    sum_short_long = portfolio.sum_short_long
    network_method = portfolio.network_method
    network_ip = portfolio.network_ip

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
    if network_method == :IP
        if obj == :Sharpe
            @variable(model, tass_bin2[1:N], binary = true)
            @variable(model, tass_bin_sharpe2[1:N] >= 0)
        else
            @variable(model, tass_bin2[1:N], binary = true)
        end
    end

    # Weight constraints.
    if obj == :Sharpe
        @constraint(model, sum(model[:w]) == sum_short_long * model[:k])

        if haskey(model, :tass_bin)
            @constraint(model, tass_bin_sharpe .<= model[:k])
            @constraint(model, tass_bin_sharpe .<= factor_mna * tass_bin)
            @constraint(model,
                        tass_bin_sharpe .>= model[:k] .- factor_mna * (1 .- tass_bin))
            @constraint(model, model[:w] .<= long_u * tass_bin_sharpe)
        end

        if haskey(model, :tass_bin2)
            @constraint(model, tass_bin_sharpe2 .<= model[:k])
            @constraint(model, tass_bin_sharpe2 .<= factor_nip * tass_bin2)
            @constraint(model,
                        tass_bin_sharpe2 .>= model[:k] .- factor_nip * (1 .- tass_bin2))
            @constraint(model, model[:w] .<= long_u * tass_bin_sharpe2)
        end

        # Maximum number of assets constraints.
        if max_number_assets > 0
            @constraint(model, sum(tass_bin) <= max_number_assets)
        end

        if network_method == :IP
            @constraint(model, unique(network_ip + I; dims = 1) * tass_bin2 .<= 1)
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
            if max_number_assets > 0
                @constraint(model, model[:w] .>= -short_u * tass_bin_sharpe)
            end

            if network_method == :IP
                @constraint(model, model[:w] .>= -short_u * tass_bin_sharpe2)
            end
        end
    else
        @constraint(model, sum(model[:w]) == sum_short_long)

        if haskey(model, :tass_bin)
            @constraint(model, model[:w] .<= long_u * tass_bin)
        end

        if haskey(model, :tass_bin2)
            @constraint(model, model[:w] .<= long_u * tass_bin2)
        end

        # Maximum number of assets constraints.
        if max_number_assets > 0
            @constraint(model, sum(tass_bin) <= max_number_assets)
        end

        if network_method == :IP
            @constraint(model, unique(network_ip + I; dims = 1) * tass_bin2 .<= 1)
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
            if max_number_assets > 0
                @constraint(model, model[:w] .>= -short_u * tass_bin)
            end

            if network_method == :IP
                @constraint(model, model[:w] .>= -short_u * tass_bin2)
            end
        end
    end

    return nothing
end

function _setup_linear_constraints(portfolio, obj, type)
    A = portfolio.a_mtx_ineq
    B = portfolio.b_vec_ineq

    if isempty(A) || isempty(B)
        return nothing
    end

    model = portfolio.model

    # Linear weight constraints.
    if obj == :Sharpe || type == :RP
        @constraint(model, A * model[:w] .- B * model[:k] .>= 0)
    else
        @constraint(model, A * model[:w] .- B .>= 0)
    end

    return nothing
end

function _setup_centrality_constraints(portfolio, obj)
    A = portfolio.a_vec_cent
    B = portfolio.b_cent

    if isempty(A) || isinf(B)
        return nothing
    end

    model = portfolio.model

    if obj == :Sharpe
        @constraint(model, transpose(A) * model[:w] - B * model[:k] == 0)
    else
        @constraint(model, transpose(A) * model[:w] - B == 0)
    end

    return nothing
end

function _setup_min_number_effective_assets(portfolio, obj)
    mnea = portfolio.min_number_effective_assets

    if mnea < 1
        return nothing
    end

    model = portfolio.model

    @variable(model, tmnea >= 0)
    @constraint(model, [tmnea; model[:w]] in SecondOrderCone())

    if obj == :Sharpe
        @constraint(model, tmnea * sqrt(mnea) <= model[:k])
    else
        @constraint(model, tmnea * sqrt(mnea) <= 1)
    end

    return nothing
end

function _setup_tracking_err(portfolio, returns, obj, T)
    tracking_err = portfolio.tracking_err

    if isinf(tracking_err)
        return nothing
    end

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

    if !tracking_err_flag
        return nothing
    end

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

    if isinf(turnover) || isempty(turnover_weights)
        return nothing
    end

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
    _setup_centrality_constraints(portfolio, obj)
    _setup_weights(portfolio, obj, N)
    _setup_min_number_effective_assets(portfolio, obj)
    _setup_tracking_err(portfolio, returns, obj, T)
    _setup_turnover(portfolio, N, obj)
    _setup_trad_wc_objective_function(portfolio, type, obj, class, kelly, l)

    return nothing
end

function _setup_trad_wc_objective_function(portfolio, type, obj, class, kelly, l)
    model = portfolio.model

    pf = zero(eltype(portfolio.returns))

    if haskey(model, :penalty_factor) && type == :Trad
        pf = model[:penalty_factor]
    end

    if obj == :Sharpe
        if (type == :Trad && class == :Classic || type == :WC) && kelly != :None
            @objective(model, Max, model[:ret] - pf)
        else
            @objective(model, Min, model[:risk] + pf)
        end
    elseif obj == :Min_Risk
        @objective(model, Min, model[:risk] + pf)
    elseif obj == :Utility
        @objective(model, Max, model[:ret] - l * model[:risk] - pf)
    elseif obj == :Max_Ret
        @objective(model, Max, model[:ret] - pf)
    end
    return nothing
end

function _optimise_portfolio(portfolio, type, obj, near_opt = false, coneopt = true)
    solvers = portfolio.solvers
    model = portfolio.model

    N = size(portfolio.returns, 2)
    rtype = eltype(portfolio.returns)
    term_status = termination_status(model)
    solvers_tried = Dict()

    strtype = if !near_opt
        "_" * String(type)
    else
        tmp = "_Near_"
        tmp = coneopt ? tmp * "Cone_" : tmp * "NL_"
        tmp * String(type)
    end

    for (key, val) ∈ solvers
        key = Symbol(String(key) * strtype)

        if haskey(val, :solver)
            set_optimizer(model, val[:solver])
        end

        if haskey(val, :params)
            for (attribute, value) ∈ val[:params]
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
        all_non_zero_weights = !all(isapprox.(abs.(value.(model[:w])),
                                              zero(eltype(portfolio.returns))))

        if term_status in ValidTermination && all_finite_weights && all_non_zero_weights
            break
        end

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

        push!(solvers_tried,
              key => Dict(:objective_val => objective_value(model),
                          :term_status => term_status,
                          :params => haskey(val, :params) ? val[:params] : missing,
                          :finite_weights => all_finite_weights,
                          :nonzero_weights => all_non_zero_weights,
                          :portfolio => DataFrame(; tickers = portfolio.assets,
                                                  weights = weights)))
    end

    if isempty(solvers_tried) && term_status ∉ ValidTermination
        push!(solvers_tried, :error => term_status)
    end

    return term_status, solvers_tried
end

function _finalise_portfolio(portfolio, returns, N, solvers_tried, type, rm, obj,
                             near_opt = false, coneopt = true)
    model = portfolio.model

    strtype = if !near_opt
        String(type)
    else
        tmp = "Near_"
        tmp = coneopt ? tmp * "Cone_" : tmp * "NL_"
        tmp * String(type)
    end

    if type in (:Trad, :RP) && rm in (:EVaR, :EDaR, :RVaR, :RDaR)
        z_key = "z_" * lowercase(string(rm))
        z_key2 = Symbol(strtype * "_" * z_key)
        portfolio.z[z_key2] = value(portfolio.model[Symbol(z_key)])
        if type == :Trad && obj == :Sharpe
            (portfolio.z[z_key2] /= value(portfolio.model[:k]))
        end
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

    if near_opt
        (type = Symbol(strtype))
    end

    portfolio.optimal[type] = DataFrame(; tickers = portfolio.assets, weights = weights)

    isempty(solvers_tried) ? portfolio.fail = Dict() : portfolio.fail = solvers_tried

    return portfolio.optimal[type]
end

function _handle_errors_and_finalise(portfolio, term_status, returns, N, solvers_tried,
                                     type, rm, obj, near_opt = false, coneopt = true)
    retval = if term_status ∉ ValidTermination ||
                any(.!isfinite.(value.(portfolio.model[:w])))
        funcname = "$(fullname(PortfolioOptimiser)[1]).$(nameof(PortfolioOptimiser.optimise!))"
        @warn("$funcname: model could not be optimised satisfactorily.\nSolvers: $solvers_tried.")
        portfolio.fail = solvers_tried
        if near_opt
            tmp = "Near_"
            tmp = coneopt ? tmp * "Cone_" : tmp * "NL_"
            type = Symbol(tmp * String(type))
        end
        portfolio.optimal[type] = DataFrame()
    else
        _finalise_portfolio(portfolio, returns, N, solvers_tried, type, rm, obj, near_opt,
                            coneopt)
    end

    return retval
end

function _p_save_opt_params(portfolio, type, class, hist, rm, obj, kelly, rrp_ver, rf, l,
                            rrp_penalty, u_mu, u_cov, string_names, w_ini, M, w_min, w_max,
                            save_opt_params)
    if !save_opt_params
        return nothing
    end

    opt_params_dict = if type == :Trad
        Dict(:class => class, :rm => rm, :obj => obj, :kelly => kelly, :rf => rf, :l => l,
             :string_names => string_names, :hist => hist, :M => M, :w_ini => w_ini,
             :w_min => w_min, :w_max => w_max)
    elseif type == :RP
        Dict(:class => class, :rm => rm, :obj => :Min_Risk,
             :kelly => (kelly == :Exact) ? :None : kelly, :rf => rf,
             :string_names => string_names, :hist => hist, :M => M, :w_ini => w_ini,
             :w_min => w_min, :w_max => w_max)
    elseif type == :RRP
        Dict(:class => class, :rm => :SD, :obj => :Min_Risk, :kelly => kelly,
             :rrp_penalty => rrp_penalty, :rrp_ver => rrp_ver,
             :string_names => string_names, :hist => hist, :M => M, :w_ini => w_ini,
             :w_min => w_min, :w_max => w_max)
    elseif type == :WC
        Dict(:rm => :SD, :obj => obj, :kelly => kelly, :rf => rf, :l => l, :u_mu => u_mu,
             :u_cov => u_cov, :string_names => string_names, :hist => hist, :M => M,
             :w_ini => w_ini, :w_min => w_min, :w_max => w_max)
    end

    portfolio.opt_params[type] = opt_params_dict

    return nothing
end

function _setup_model_class(portfolio, class, hist)
    mu, sigma, returns = (Vector{eltype(portfolio.mu)}(undef, 0),
                          Matrix{eltype(portfolio.cov)}(undef, 0, 0),
                          Matrix{eltype(portfolio.returns)}(undef, 0, 0))

    if class != :Classic
        @smart_assert(hist in BLHist)
    end

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

function _near_optimal_centering(portfolio, class, hist, kelly, rf, rm, mu, returns, sigma,
                                 w_opt, M, type, N, obj,
                                 w1 = Vector{eltype(returns)}(undef, 0),
                                 w2 = Vector{eltype(returns)}(undef, 0))
    if isempty(w1) || isempty(w2)
        fl = frontier_limits!(portfolio; class = class, hist = hist, kelly = kelly, rf = rf,
                              rm = rm, save_model = true)

        w1 = fl.w_min
        w2 = fl.w_max
    end

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

    risk1, risk2 = _ul_risk(rm, returns, w1, w2, sigma, rf, solvers, alpha, kappa, alpha_i,
                            beta, a_sim, beta_i, b_sim, owa_w, 0)

    w3 = w_opt.weights
    ret3 = dot(mu, w3)
    risk3 = calc_risk(w3, returns; rm = rm, rf = rf, sigma = sigma, alpha_i = alpha_i,
                      alpha = alpha, a_sim = a_sim, beta_i = beta_i, beta = beta,
                      b_sim = b_sim, kappa = kappa, owa_w = owa_w, solvers = solvers)

    c1 = (ret2 - ret1) / M
    c2 = (risk2 - risk1) / M
    e1 = ret3 - c1
    e2 = risk3 + c2

    model = portfolio.model

    @constraint(model, model[:ret] >= e1)
    @constraint(model, model[:risk] <= e2)

    model1 = copy(model)
    @variable(model, log_ret)
    @constraint(model, [-log_ret, 1, model[:ret] - e1] in MOI.ExponentialCone())
    @variable(model, log_risk)
    @constraint(model, [-log_risk, 1, e2 - model[:risk]] in MOI.ExponentialCone())
    if !haskey(model, :log_w)
        @variable(model, log_w[1:N])
        @constraint(model, [i = 1:N], [log_w[i], 1, model[:w][i]] in MOI.ExponentialCone())
    else
        log_w = model[:log_w]
    end
    @variable(model, log_omw[1:N])
    @constraint(model, [i = 1:N],
                [log_omw[i], 1, 1 - model[:w][i]] in MOI.ExponentialCone())
    @expression(model, neg_sum_log_ws, -sum(log_w .+ log_omw))
    @expression(model, near_opt_risk, log_ret + log_risk + neg_sum_log_ws)
    @objective(model, Min, near_opt_risk)

    term_status, solvers_tried = _optimise_portfolio(portfolio, type, obj, true, true)
    retval = _handle_errors_and_finalise(portfolio, term_status, returns, N, solvers_tried,
                                         type, rm, obj, true, true)

    if term_status ∉ ValidTermination || any(.!isfinite.(value.(portfolio.model[:w])))
        model = portfolio.model = copy(model1)
        @expression(model, log_ret, -log(model[:ret] - e1))
        @expression(model, log_risk, -log(e2 - model[:risk]))
        @expression(model, neg_sum_log_ws, -sum(log.(1 .+ model[:w]) .+ log.(model[:w])))
        @expression(model, near_opt_risk, log_ret + log_risk + neg_sum_log_ws)
        @objective(model, Min, near_opt_risk)
        term_status2, solvers_tried2 = _optimise_portfolio(portfolio, type, obj, true,
                                                           false)
        retval = _handle_errors_and_finalise(portfolio, term_status2, returns, N,
                                             merge(solvers_tried, solvers_tried2), type, rm,
                                             obj, true, false)
    end

    return retval
end

"""
```julia
optimise!(portfolio::Portfolio; class::Symbol = :Classic, hist::Integer = 1,
          kelly::Symbol = :None, l::Real = 2.0, obj::Symbol = :Sharpe, rf::Real = 0.0,
          rm::Symbol = :SD, rrp_penalty::Real = 1.0, rrp_ver::Symbol = :None,
          save_opt_params::Bool = true, string_names::Bool = false, type::Symbol = :Trad,
          u_cov::Symbol = :Box, u_mu::Symbol = :Box)
```
"""
function optimise!(portfolio::Portfolio; class::Symbol = :Classic, hist::Integer = 1,
                   kelly::Symbol = :None, type::Symbol = :Trad, rm::Symbol = :SD,
                   obj::Symbol = :Sharpe, rf::Real = 0.0, l::Real = 2.0,
                   rrp_ver::Symbol = :None, rrp_penalty::Real = 1.0, u_cov::Symbol = :Box,
                   u_mu::Symbol = :Box, near_opt::Bool = false,
                   M::Real = near_opt ? ceil(sqrt(size(portfolio.returns, 2))) : 0,
                   w_ini::AbstractVector = Vector{eltype(portfolio.returns)}(undef, 0),
                   w_min::AbstractVector = Vector{eltype(portfolio.returns)}(undef, 0),
                   w_max::AbstractVector = Vector{eltype(portfolio.returns)}(undef, 0),
                   save_opt_params::Bool = true, string_names::Bool = false)
    @smart_assert(type in PortTypes)
    @smart_assert(class in PortClasses)
    @smart_assert(rm in RiskMeasures)
    @smart_assert(obj in ObjFuncs)
    @smart_assert(kelly in KellyRet)
    @smart_assert(rrp_ver in RRPVersions)
    @smart_assert(u_mu in UncertaintyTypes)
    @smart_assert(u_cov in UncertaintyTypes)
    @smart_assert(portfolio.kind_tracking_err ∈ TrackingErrKinds)
    if !isempty(w_ini)
        @smart_assert(length(w_ini) == size(portfolio.returns, 2))
    end
    if near_opt
        @smart_assert(M > 0)
        @smart_assert(length(w_min) == size(portfolio.returns, 2))
        @smart_assert(length(w_max) == size(portfolio.returns, 2))
    end

    portfolio.model = JuMP.Model()

    mu, sigma, returns = _setup_model_class(portfolio, class, hist)
    T, N = size(returns)
    kurtosis = portfolio.kurt
    skurtosis = portfolio.skurt
    network_method = portfolio.network_method

    # Model variables.
    model = portfolio.model
    set_string_names_on_creation(model, string_names)
    @variable(model, w[1:N])

    if !isempty(w_ini)
        set_start_value.(w, w_ini)
    end

    if type == :Trad
        _setup_sharpe_k(model, obj)
        _risk_setup(portfolio, :Trad, rm, kelly, obj, rf, T, N, mu, returns, sigma,
                    kurtosis, skurtosis, network_method)
        _setup_trad_return(portfolio, class, kelly, obj, T, rf, returns, mu)
        _setup_trad_wc_constraints(portfolio, obj, T, N, :Trad, class, kelly, l, returns)
    elseif type == :RP
        _setup_risk_budget(portfolio)
        _rp_setup(portfolio, N)
        _risk_setup(portfolio, :RP, rm, kelly, obj, rf, T, N, mu, returns, sigma, kurtosis,
                    skurtosis, network_method)
        _setup_rp_rrp_return_and_obj(portfolio, kelly, T, returns, mu)
    elseif type == :RRP
        _setup_risk_budget(portfolio)
        _mv_setup(portfolio, sigma, rm, kelly, obj, :RRP, network_method)
        _rrp_setup(portfolio, sigma, N, rrp_ver, rrp_penalty)
        _setup_rp_rrp_return_and_obj(portfolio, kelly, T, returns, mu)
    else
        _setup_sharpe_k(model, obj)
        _wc_setup(portfolio, kelly, obj, T, N, rf, mu, sigma, u_mu, u_cov, network_method)
        _setup_trad_wc_constraints(portfolio, obj, T, N, :WC, class, kelly, l, returns)
    end

    _setup_linear_constraints(portfolio, obj, type)

    term_status, solvers_tried = _optimise_portfolio(portfolio, type, obj)
    retval = _handle_errors_and_finalise(portfolio, term_status, returns, N, solvers_tried,
                                         type, rm, obj)

    if near_opt
        retval = _near_optimal_centering(portfolio, class, hist, kelly, rf, rm, mu, returns,
                                         sigma, retval, M, type, N, obj, w_min, w_max)
    end

    _p_save_opt_params(portfolio, type, class, hist, rm, obj, kelly, rrp_ver, rf, l,
                       rrp_penalty, u_mu, u_cov, string_names, w_ini, M, w_min, w_max,
                       save_opt_params)

    return retval
end

"""
```julia
frontier_limits!(portfolio::Portfolio; class::Symbol = :Classic, hist::Integer = 1,
                 kelly::Symbol = :None, rf::Real = 0.0, rm::Symbol = :SD,
                 save_model::Bool = false)
```
"""
function frontier_limits!(portfolio::Portfolio; class::Symbol = :Classic, hist::Integer = 1,
                          kelly::Symbol = :None, rf::Real = 0.0, rm::Symbol = :SD,
                          save_model::Bool = false)
    optimal1 = deepcopy(portfolio.optimal)
    fail1 = deepcopy(portfolio.fail)
    if save_model
        (model1 = copy(portfolio.model))
    end

    w_min = optimise!(portfolio; class = class, hist = hist, kelly = kelly, obj = :Min_Risk,
                      rf = rf, rm = rm, save_opt_params = false)

    w_max = optimise!(portfolio; class = class, hist = hist, kelly = kelly, obj = :Max_Ret,
                      rf = rf, rm = rm, save_opt_params = false)

    limits = hcat(w_min, DataFrame(; x1 = w_max[!, 2]))
    DataFrames.rename!(limits, :weights => :w_min, :x1 => :w_max)
    portfolio.limits[rm] = limits

    portfolio.optimal = optimal1
    portfolio.fail = fail1
    if save_model
        (portfolio.model = model1)
    end

    return portfolio.limits[rm]
end

"""
```julia
efficient_frontier!(portfolio::Portfolio; class::Symbol = :Classic, hist::Integer = 1,
                    kelly::Symbol = :None, rf::Real = 0.0, rm::Symbol = :SD,
                    points::Integer = 20)
```
"""
function efficient_frontier!(portfolio::Portfolio; class::Symbol = :Classic,
                             hist::Integer = 1, kelly::Symbol = :None, rf::Real = 0.0,
                             rm::Symbol = :SD, points::Integer = 20)
    optimal1 = deepcopy(portfolio.optimal)
    fail1 = deepcopy(portfolio.fail)

    mu, sigma, returns = _setup_model_class(portfolio, class, hist)

    fl = frontier_limits!(portfolio; class = class, hist = hist, kelly = kelly, rf = rf,
                          rm = rm)

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

    risk1, risk2 = _ul_risk(rm, returns, w1, w2, sigma, rf, solvers, alpha, kappa, alpha_i,
                            beta, a_sim, beta_i, b_sim, owa_w, 0)

    mus = range(ret1; stop = ret2, length = points)
    risks = range(risk1; stop = risk2, length = points)

    rmf = Symbol(lowercase(string(rm)) * "_u")

    frontier = Vector{typeof(risk1)}(undef, 0)
    srisk = Vector{typeof(risk1)}(undef, 0)
    w_ini = Vector{typeof(risk1)}(undef, 0)

    i = 0
    for (j, (r, m)) ∈ enumerate(zip(risks, mus))
        if i == 0
            w = optimise!(portfolio; class = class, hist = hist, kelly = kelly,
                          obj = :Min_Risk, rf = rf, rm = rm, save_opt_params = false)
        else
            if !isempty(w)
                (w_ini = w.weights)
            end
            if j != length(risks)
                setproperty!(portfolio, rmf, r)
            else
                setproperty!(portfolio, rmf, Inf)
            end
            w = optimise!(portfolio; class = class, hist = hist, kelly = kelly,
                          obj = :Max_Ret, rf = rf, rm = rm, save_opt_params = false,
                          w_ini = w_ini)
            # Fallback in case :Max_Ret with maximum risk bounds fails.
            if isempty(w)
                setproperty!(portfolio, rmf, Inf)
                j != length(risks) ? portfolio.mu_l = m : portfolio.mu_l = Inf
                w = optimise!(portfolio; class = class, hist = hist, kelly = kelly,
                              obj = :Min_Risk, rf = rf, rm = rm, save_opt_params = false,
                              w_ini = w_ini)
                portfolio.mu_l = Inf
            end
        end
        if isempty(w)
            continue
        end
        rk = calc_risk(w.weights, returns; rm = rm, rf = rf, sigma = sigma,
                       alpha_i = alpha_i, alpha = alpha, a_sim = a_sim, beta_i = beta_i,
                       beta = beta, b_sim = b_sim, kappa = kappa, owa_w = owa_w,
                       solvers = solvers)
        append!(frontier, w.weights)
        push!(srisk, rk)
        i += 1
    end
    setproperty!(portfolio, rmf, Inf)

    w = optimise!(portfolio; class = class, hist = hist, kelly = kelly, obj = :Sharpe,
                  rf = rf, rm = rm, save_opt_params = false)
    sharpe = false
    if !isempty(w)
        rk = calc_risk(w.weights, returns; rm = rm, rf = rf, sigma = sigma,
                       alpha_i = alpha_i, alpha = alpha, a_sim = a_sim, beta_i = beta_i,
                       beta = beta, b_sim = b_sim, kappa = kappa, owa_w = owa_w,
                       solvers = solvers)
        append!(frontier, w.weights)
        push!(srisk, rk)
        i += 1
        sharpe = true
    end

    portfolio.frontier[rm] = Dict(:weights => hcat(DataFrame(; tickers = portfolio.assets),
                                                   DataFrame(reshape(frontier, length(w1),
                                                                     :),
                                                             string.(range(1, i)))),
                                  :class => class, :hist => hist, :kelly => kelly,
                                  :rf => rf, :points => points, :risk => srisk,
                                  :sharpe => sharpe)

    portfolio.optimal = optimal1
    portfolio.fail = fail1

    return portfolio.frontier[rm]
end
