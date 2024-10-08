function _sdp_setup(portfolio, obj, rm, type, N, u_cov = :None)
    network_method = portfolio.network_method
    kurt_u = portfolio.kurt_u
    skurt_u = portfolio.skurt_u

    if !(type ∈ (:Trad, :RP) && rm ∈ (:Kurt, :SKurt) ||
         type ∈ (:Trad, :WC) && network_method == :SDP ||
         type == :Trad && (isfinite(kurt_u) || isfinite(skurt_u)) ||
         type == :WC && u_cov ∈ (:Box, :Ellipse))
        return nothing
    end

    model = portfolio.model
    @variable(model, W[1:N, 1:N], Symmetric)
    @expression(model, M1, vcat(W, transpose(model[:w])))
    if type ∈ (:Trad, :WC) && obj == :Sharpe
        @expression(model, M2, vcat(model[:w], model[:k]))
    else
        @expression(model, M2, vcat(model[:w], 1))
    end
    @expression(model, M3, hcat(M1, M2))
    @constraint(model, M3 ∈ PSDCone())

    if type ∈ (:Trad, :WC) && network_method == :SDP
        network_sdp = portfolio.network_sdp
        @constraint(model, network_sdp .* model[:W] .== 0)
        sd_u = portfolio.sd_u
        if type == :Trad && rm != :SD && isinf(sd_u)
            network_penalty = portfolio.network_penalty
            @expression(model, network_penalty_factor, network_penalty * tr(model[:W]))
        end
    end

    return nothing
end

function _mv_risk(model, sigma, type, network_method, sd_cone::Bool = true)
    if type ∈ (:Trad, :WC) && network_method == :SDP
        @expression(model, dev_risk, tr(sigma * model[:W]))
    else
        if sd_cone
            G = sqrt(sigma)
            @variable(model, dev)
            @constraint(model, [dev; G * model[:w]] ∈ SecondOrderCone())
            @expression(model, dev_risk, dev^2)
        else
            @expression(model, dev_risk, dot(model[:w], sigma, model[:w]))
        end
    end
    return nothing
end

function _mv_setup(portfolio, sigma, rm, kelly, obj, type, network_method,
                   sd_cone::Bool = true)
    sd_u = portfolio.sd_u

    if !(rm == :SD || kelly == :Approx || isfinite(sd_u))
        return nothing
    end

    model = portfolio.model

    _mv_risk(model, sigma, type, network_method, sd_cone)

    if isfinite(sd_u) && type == :Trad
        if obj == :Sharpe
            if network_method != :SDP && sd_cone
                @constraint(model, model[:dev] <= sd_u * model[:k])
            else
                @constraint(model, model[:dev_risk] <= sd_u^2 * model[:k])
            end
        else
            if network_method != :SDP && sd_cone
                @constraint(model, model[:dev] <= sd_u)
            else
                @constraint(model, model[:dev_risk] <= sd_u^2)
            end
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

    if !(rm ∈ (:MAD, :SSD) || isfinite(mad_u) || isfinite(ssd_u))
        return nothing
    end

    model = portfolio.model
    msv_target = portfolio.msv_target

    abs_dev = if isa(msv_target, Real) && isinf(msv_target) ||
                 isa(msv_target, AbstractVector) && isempty(msv_target)
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
    @constraint(model, [sdev; mad] ∈ SecondOrderCone())

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

    if !(rm ∈ (:FLPM, :SLPM) || isfinite(flpm_u) || isfinite(slpm_u))
        return nothing
    end

    model = portfolio.model

    lpm_target = portfolio.lpm_target

    lpm_t = if isa(lpm_target, Real) && isinf(lpm_target) ||
               isa(lpm_target, AbstractVector) && isempty(lpm_target)
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
    @constraint(model, [slpm; lpm] ∈ SecondOrderCone())
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
    rg_u = portfolio.rg_u

    if !(rm ∈ (:WR, :RG) || isfinite(wr_u) || isfinite(rg_u))
        return nothing
    end

    model = portfolio.model

    if !haskey(model, :hist_ret)
        @expression(model, hist_ret, returns * model[:w])
    end

    if rm == :WR || isfinite(wr_u)
        @variable(model, wr)
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
    end

    if !(rm == :RG || isfinite(rg_u))
        return nothing
    end

    if !haskey(model, :wr_risk)
        @variable(model, wr)
        @constraint(model, -model[:hist_ret] .<= wr)
        @expression(model, wr_risk, wr)
    end

    @variable(model, br)
    @constraint(model, -model[:hist_ret] .>= br)
    @expression(model, rg_risk, wr_risk - br)

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

    return nothing
end

function _var_setup(portfolio, rm, T, returns, obj, type)
    cvar_u = portfolio.cvar_u
    rcvar_u = portfolio.rcvar_u
    evar_u = portfolio.evar_u
    rvar_u = portfolio.rvar_u

    if !(rm ∈ (:CVaR, :CVaRRG, :EVaR, :RLVaR) ||
         isfinite(cvar_u) ||
         isfinite(rcvar_u) ||
         isfinite(evar_u) ||
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
        @variable(model, z_var[1:T] .>= 0)
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

    if rm == :CVaRRG || isfinite(rcvar_u)
        if !haskey(model, :cvar_risk)
            invat = portfolio.invat
            @variable(model, var)
            @variable(model, z_var[1:T] .>= 0)
            @constraint(model, z_var .>= -model[:hist_ret] .- var)
            @expression(model, cvar_risk, var + sum(z_var) * invat)
        end

        invbt = portfolio.invbt
        @variable(model, var2)
        @variable(model, z_var2[1:T] .<= 0)
        @constraint(model, z_var2 .<= -model[:hist_ret] .- var2)
        @expression(model, cvar2_risk, var2 + sum(z_var2) * invbt)

        @expression(model, rcvar_risk, cvar_risk - cvar2_risk)

        if isfinite(rcvar_u) && type == :Trad
            if obj == :Sharpe
                @constraint(model, rcvar_risk <= rcvar_u * model[:k])
            else
                @constraint(model, rcvar_risk <= rcvar_u)
            end
        end

        if rm == :CVaRRG
            @expression(model, risk, rcvar_risk)
        end
    end

    if rm == :EVaR || isfinite(evar_u)
        at = portfolio.at
        @variable(model, t_evar)
        @variable(model, z_evar >= 0)
        @variable(model, u_evar[1:T])
        @constraint(model, sum(u_evar) <= z_evar)
        @constraint(model, [i = 1:T],
                    [-model[:hist_ret][i] - t_evar, z_evar, u_evar[i]] ∈
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

    if !(rm == :RLVaR || isfinite(rvar_u))
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
                [z_rvar * opk * invk2, psi_rvar[i] * opk * invk, epsilon_rvar[i]] ∈
                MOI.PowerCone(invopk))
    @constraint(model, [i = 1:T],
                [omega_rvar[i] * invomk, theta_rvar[i] * invk, -z_rvar * invk2] ∈
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

    if rm == :RLVaR
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

    if !(rm ∈ (:MDD, :ADD, :CDaR, :UCI, :EDaR, :RLDaR) ||
         isfinite(mdd_u) ||
         isfinite(add_u) ||
         isfinite(cdar_u) ||
         isfinite(uci_u) ||
         isfinite(edar_u) ||
         isfinite(rdar_u))
        return nothing
    end

    model = portfolio.model

    if !haskey(model, :hist_ret)
        @expression(model, hist_ret, returns * model[:w])
    end
    @variable(model, dd[1:(T + 1)])
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
                @constraint(model, add_risk <= add_u * model[:k])
            else
                @constraint(model, add_risk <= add_u)
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
        @constraint(model, [uci; dd[2:end]] ∈ SecondOrderCone())
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
                    [dd[i + 1] - t_edar, z_edar, u_edar[i]] ∈ MOI.ExponentialCone())
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

    if !(rm == :RLDaR || isfinite(rdar_u))
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
    @expression(model, rdar_risk, t_rdar + ln_k * z_rdar + sum(psi_rdar .+ theta_rdar))
    @constraint(model, [i = 1:T],
                [z_rdar * opk * invk2, psi_rdar[i] * opk * invk, epsilon_rdar[i]] ∈
                MOI.PowerCone(invopk))
    @constraint(model, [i = 1:T],
                [omega_rdar[i] * invomk, theta_rdar[i] * invk, -z_rdar * invk2] ∈
                MOI.PowerCone(omk))
    @constraint(model, dd[2:end] .- t_rdar .+ epsilon_rdar .+ omega_rdar .<= 0)
    if isfinite(rdar_u) && type == :Trad
        if obj == :Sharpe
            @constraint(model, rdar_risk <= rdar_u * model[:k])
        else
            @constraint(model, rdar_risk <= rdar_u)
        end
    end

    if rm == :RLDaR
        @expression(model, risk, rdar_risk)
    end

    return nothing
end

function _dvar_setup(portfolio, rm, T, returns, obj, type)
    dvar_u = portfolio.dvar_u

    if !(rm == :DVar || isfinite(dvar_u))
        return nothing
    end

    model = portfolio.model

    ovec = range(1; stop = 1, length = T)
    @variable(model, Dt[1:T, 1:T], Symmetric)
    if !haskey(model, :hist_ret)
        @expression(model, hist_ret, returns * model[:w])
    end
    @expression(model, Dx,
                Symmetric(model[:hist_ret] * transpose(ovec) -
                          ovec * transpose(model[:hist_ret])))
    @constraint(model, [i = 1:T, j = i:T], Dt[i, j] >= -Dx[i, j])
    @constraint(model, [i = 1:T, j = i:T], Dt[i, j] >= Dx[i, j])

    dt = vec(Dt)
    invT2 = 1 / T^2
    @expression(model, dvar_risk, invT2 * (dot(dt, dt) + invT2 * dot(ovec, Dt, ovec)^2))

    if isfinite(dvar_u) && type == :Trad
        if obj == :Sharpe
            @constraint(model, dvar_risk <= dvar_u * model[:k])
        else
            @constraint(model, dvar_risk <= dvar_u)
        end
    end

    if rm == :DVar
        @expression(model, risk, dvar_risk)
    end

    return nothing
end

function _risk_setup(portfolio, type, rm, kelly, obj, rf, T, N, mu, returns, sigma,
                     kurtosis, skurtosis, network_method, sd_cone, owa_approx)
    _sdp_setup(portfolio, obj, rm, type, N)
    _mv_setup(portfolio, sigma, rm, kelly, obj, type, network_method, sd_cone)
    _mad_setup(portfolio, rm, T, returns, mu, obj, type)
    _lpm_setup(portfolio, rm, T, returns, obj, rf, type)
    _wr_setup(portfolio, rm, returns, obj, type)
    _var_setup(portfolio, rm, T, returns, obj, type)
    _drawdown_setup(portfolio, rm, T, returns, obj, type)
    _kurtosis_setup(portfolio, kurtosis, skurtosis, rm, N, obj, type)
    _owa_setup(portfolio, rm, T, returns, obj, type, owa_approx)
    _dvar_setup(portfolio, rm, T, returns, obj, type)
    _skew_setup(portfolio, rm, N, obj, type, sd_cone)
    _add_skew_to_risk(portfolio, rm)

    return nothing
end

function _add_skew_to_risk(portfolio, rm)
    model = portfolio.model
    skew_factor = portfolio.skew_factor
    sskew_factor = portfolio.sskew_factor

    if !(isfinite(skew_factor) || isfinite(sskew_factor))
        return nothing
    end

    if isfinite(skew_factor) && !iszero(skew_factor) && rm != :Skew
        @expression(model, tmp, model[:risk] + skew_factor * model[:skew_risk])
        unregister(model, :risk)
        @expression(model, risk, tmp)
        unregister(model, :tmp)
    end

    if isfinite(sskew_factor) && !iszero(sskew_factor) && rm != :SSkew
        @expression(model, tmp, model[:risk] + sskew_factor * model[:sskew_risk])
        unregister(model, :risk)
        @expression(model, risk, tmp)
        unregister(model, :tmp)
    end

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

function _skew_setup(portfolio, rm, N, obj, type, sd_cone)
    skew_u = portfolio.skew_u
    skew_factor = portfolio.skew_factor
    sskew_u = portfolio.sskew_u
    sskew_factor = portfolio.sskew_factor

    if !(rm ∈ (:Skew, :SSkew) ||
         isfinite(skew_u) ||
         isfinite(sskew_u) ||
         isfinite(skew_factor) ||
         isfinite(sskew_factor))
        return nothing
    end

    model = portfolio.model

    if rm == :Skew || isfinite(skew_u) || isfinite(skew_factor) && !iszero(skew_factor)
        V = portfolio.V

        if sd_cone
            G = real(sqrt(V))
            @variable(model, t_skew)
            @constraint(model, [t_skew; G * model[:w]] ∈ SecondOrderCone())
            @expression(model, skew_risk, t_skew^2)
        else
            @expression(model, skew_risk, dot(model[:w], V, model[:w]))
        end

        if isfinite(skew_u) && type == :Trad
            if obj == :Sharpe
                if sd_cone
                    @constraint(model, t_skew <= skew_u * model[:k])
                else
                    @constraint(model, skew_risk <= skew_u^2 * model[:k])
                end
            else
                if sd_cone
                    @constraint(model, t_skew <= skew_u)
                else
                    @constraint(model, skew_risk <= skew_u^2)
                end
            end
        end

        if rm == :Skew
            @expression(model, risk, skew_risk)
        end
    end

    if !(rm == :SSkew ||
         isfinite(sskew_u) ||
         isfinite(sskew_factor) && !iszero(sskew_factor))
        return nothing
    end

    SV = portfolio.SV

    if sd_cone
        G = real(sqrt(SV))
        @variable(model, t_sskew)
        @constraint(model, [t_sskew; G * model[:w]] ∈ SecondOrderCone())
        @expression(model, sskew_risk, t_sskew^2)
    else
        @expression(model, sskew_risk, dot(model[:w], SV, model[:w]))
    end

    if isfinite(sskew_u) && type == :Trad
        if obj == :Sharpe
            if sd_cone
                @constraint(model, t_sskew <= sskew_u * model[:k])
            else
                @constraint(model, sskew_risk <= sskew_u^2 * model[:k])
            end
        else
            if sd_cone
                @constraint(model, t_sskew <= sskew_u)
            else
                @constraint(model, sskew_risk <= sskew_u^2)
            end
        end
    end

    if rm == :SSkew
        @expression(model, risk, sskew_risk)
    end

    return nothing
end

function _kurtosis_setup(portfolio, kurtosis, skurtosis, rm, N, obj, type)
    kurt_u = portfolio.kurt_u
    skurt_u = portfolio.skurt_u

    if !(rm ∈ (:Kurt, :SKurt) || isfinite(kurt_u) || isfinite(skurt_u))
        return nothing
    end

    model = portfolio.model

    if rm == :Kurt || isfinite(kurt_u)
        max_num_assets_kurt = portfolio.max_num_assets_kurt
        @variable(model, t_kurt)
        if !iszero(max_num_assets_kurt) && N > max_num_assets_kurt
            factor = portfolio.max_num_assets_kurt_scale

            N2 = factor * N
            @variable(model, x_kurt[1:N2])
            @constraint(model, [t_kurt; x_kurt] ∈ SecondOrderCone())
            A = block_vec_pq(kurtosis, N, N)
            vals_A, vecs_A = eigen(A)
            vals_A = clamp.(real(vals_A), 0, Inf) .+ clamp.(imag(vals_A), 0, Inf)im
            Bi = Vector{Matrix{eltype(kurtosis)}}(undef, N2)
            for i ∈ 1:N2
                j = i - 1
                B = reshape(real(complex(sqrt(vals_A[end - j])) * vecs_A[:, end - j]), N, N)
                Bi[i] = B
            end
            @constraint(model, [i = 1:N2], x_kurt[i] == tr(Bi[i] * model[:W]))
        else
            L_2 = portfolio.L_2
            S_2 = portfolio.S_2
            sqrt_sigma_4 = sqrt(S_2 * kurtosis * transpose(S_2))
            @expression(model, zkurt, L_2 * vec(model[:W]))
            @constraint(model, [t_kurt; sqrt_sigma_4 * zkurt] ∈ SecondOrderCone())
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

    if !(rm == :SKurt || isfinite(skurt_u))
        return nothing
    end

    max_num_assets_kurt = portfolio.max_num_assets_kurt
    @variable(model, t_skurt)
    if !iszero(max_num_assets_kurt) && N > max_num_assets_kurt
        factor = portfolio.max_num_assets_kurt_scale

        N2 = factor * N
        @variable(model, x_skurt[1:N2])

        @constraint(model, [t_skurt; x_skurt] ∈ SecondOrderCone())

        A = block_vec_pq(skurtosis, N, N)
        vals_A, vecs_A = eigen(A)
        vals_A = clamp.(real(vals_A), 0, Inf) .+ clamp.(imag(vals_A), 0, Inf)im
        SBi = Vector{Matrix{eltype(skurtosis)}}(undef, N2)
        for i ∈ 1:N2
            j = i - 1
            B = reshape(real(sqrt(complex(vals_A[end - j])) * vecs_A[:, end - j]), N, N)
            SBi[i] = B
        end
        @constraint(model, [i = 1:N2], x_skurt[i] == tr(SBi[i] * model[:W]))
    else
        L_2 = portfolio.L_2
        S_2 = portfolio.S_2
        sqrt_sigma_4 = sqrt(S_2 * skurtosis * transpose(S_2))
        @expression(model, zskurt, L_2 * vec(model[:W]))
        @constraint(model, [t_skurt; sqrt_sigma_4 * zskurt] ∈ SecondOrderCone())
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

function _owa_setup(portfolio, rm, T, returns, obj, type, owa_approx)
    gmd_u = portfolio.gmd_u
    tg_u = portfolio.tg_u
    rtg_u = portfolio.rtg_u
    owa_u = portfolio.owa_u

    if !(rm ∈ (:GMD, :TG, :TGRG, :OWA) ||
         isfinite(gmd_u) ||
         isfinite(tg_u) ||
         isfinite(rtg_u) ||
         isfinite(owa_u))
        return nothing
    end

    onesvec = range(1; stop = 1, length = T)
    model = portfolio.model

    if !haskey(model, :hist_ret)
        @expression(model, hist_ret, returns * model[:w])
    end

    if !owa_approx
        @variable(model, owa[1:T])
        @constraint(model, model[:hist_ret] == owa)
    else
        owa_p = portfolio.owa_p
        M = length(owa_p)
    end

    if rm == :GMD || isfinite(gmd_u)
        if !owa_approx
            @variable(model, gmda[1:T])
            @variable(model, gmdb[1:T])
            @expression(model, gmd_risk, sum(gmda .+ gmdb))

            gmd_w = owa_gmd(T)

            @constraint(model,
                        owa * transpose(gmd_w) .<=
                        onesvec * transpose(gmda) + gmdb * transpose(onesvec))
        else
            @variable(model, gmd_t)
            @variable(model, gmd_nu[1:T] .>= 0)
            @variable(model, gmd_eta[1:T] .>= 0)

            @variable(model, gmd_epsilon[1:T, 1:M])
            @variable(model, gmd_psi[1:T, 1:M])
            @variable(model, gmd_z[1:M])
            @variable(model, gmd_y[1:M] .>= 0)

            gmd_w = -owa_gmd(T)

            gmd_s = sum(gmd_w)
            gmd_l = minimum(gmd_w)
            gmd_h = maximum(gmd_w)

            gmd_d = [norm(gmd_w, p) for p ∈ owa_p]

            @expression(model, gmd_risk,
                        gmd_s * gmd_t - gmd_l * sum(gmd_nu) +
                        gmd_h * sum(gmd_eta) +
                        dot(gmd_d, gmd_y))

            @constraint(model,
                        model[:hist_ret] .+ gmd_t .- gmd_nu .+ gmd_eta .-
                        vec(sum(gmd_epsilon; dims = 2)) .== 0)

            @constraint(model, gmd_z .+ gmd_y .== vec(sum(gmd_psi; dims = 1)))

            @constraint(model, [i = 1:M, j = 1:T],
                        [-gmd_z[i] * owa_p[i], gmd_psi[j, i] * owa_p[i] / (owa_p[i] - 1),
                         gmd_epsilon[j, i]] ∈ MOI.PowerCone(1 / owa_p[i]))
        end

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

    if rm == :TG || isfinite(tg_u)
        alpha = portfolio.alpha
        a_sim = portfolio.a_sim
        alpha_i = portfolio.alpha_i
        if !owa_approx
            @variable(model, tga[1:T])
            @variable(model, tgb[1:T])
            @expression(model, tg_risk, sum(tga .+ tgb))
            tg_w = owa_tg(T; alpha_i = alpha_i, alpha = alpha, a_sim = a_sim)
            @constraint(model,
                        owa * transpose(tg_w) .<=
                        onesvec * transpose(tga) + tgb * transpose(onesvec))

        else
            @variable(model, tg_t)
            @variable(model, tg_nu[1:T] .>= 0)
            @variable(model, tg_eta[1:T] .>= 0)

            @variable(model, tg_epsilon[1:T, 1:M])
            @variable(model, tg_psi[1:T, 1:M])
            @variable(model, tg_z[1:M])
            @variable(model, tg_y[1:M] .>= 0)

            tg_w = -owa_tg(T; alpha_i = alpha_i, alpha = alpha, a_sim = a_sim)

            tg_s = sum(tg_w)
            tg_l = minimum(tg_w)
            tg_h = maximum(tg_w)

            tg_d = [norm(tg_w, p) for p ∈ owa_p]

            @expression(model, tg_risk,
                        tg_s * tg_t - tg_l * sum(tg_nu) +
                        tg_h * sum(tg_eta) +
                        dot(tg_d, tg_y))

            @constraint(model,
                        model[:hist_ret] .+ tg_t .- tg_nu .+ tg_eta .-
                        vec(sum(tg_epsilon; dims = 2)) .== 0)

            @constraint(model, tg_z .+ tg_y .== vec(sum(tg_psi; dims = 1)))

            @constraint(model, [i = 1:M, j = 1:T],
                        [-tg_z[i] * owa_p[i], tg_psi[j, i] * owa_p[i] / (owa_p[i] - 1),
                         tg_epsilon[j, i]] ∈ MOI.PowerCone(1 / owa_p[i]))
        end

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

    if rm == :TGRG || isfinite(rtg_u)
        alpha = portfolio.alpha
        a_sim = portfolio.a_sim
        alpha_i = portfolio.alpha_i
        beta = portfolio.beta
        b_sim = portfolio.b_sim
        beta_i = portfolio.beta_i

        if !owa_approx
            @variable(model, rtga[1:T])
            @variable(model, rtgb[1:T])
            @expression(model, rtg_risk, sum(rtga .+ rtgb))
            rtg_w = owa_rtg(T; alpha_i = alpha_i, alpha = alpha, a_sim = a_sim,
                            beta_i = beta_i, beta = beta, b_sim = b_sim)
            @constraint(model,
                        owa * transpose(rtg_w) .<=
                        onesvec * transpose(rtga) + rtgb * transpose(onesvec))
        else
            if !haskey(model, :tg_risk)
                @variable(model, tg_t)
                @variable(model, tg_nu[1:T] .>= 0)
                @variable(model, tg_eta[1:T] .>= 0)

                @variable(model, tg_epsilon[1:T, 1:M])
                @variable(model, tg_psi[1:T, 1:M])
                @variable(model, tg_z[1:M])
                @variable(model, tg_y[1:M] .>= 0)

                tg_w = -owa_tg(T; alpha_i = alpha_i, alpha = alpha, a_sim = a_sim)

                tg_s = sum(tg_w)
                tg_l = minimum(tg_w)
                tg_h = maximum(tg_w)

                tg_d = [norm(tg_w, p) for p ∈ owa_p]

                @expression(model, tg_risk,
                            tg_s * tg_t - tg_l * sum(tg_nu) +
                            tg_h * sum(tg_eta) +
                            dot(tg_d, tg_y))

                @constraint(model,
                            model[:hist_ret] .+ tg_t .- tg_nu .+ tg_eta .-
                            vec(sum(tg_epsilon; dims = 2)) .== 0)

                @constraint(model, tg_z .+ tg_y .== vec(sum(tg_psi; dims = 1)))

                @constraint(model, [i = 1:M, j = 1:T],
                            [-tg_z[i] * owa_p[i], tg_psi[j, i] * owa_p[i] / (owa_p[i] - 1),
                             tg_epsilon[j, i]] ∈ MOI.PowerCone(1 / owa_p[i]))
            end

            @variable(model, tg2_t)
            @variable(model, tg2_nu[1:T] .>= 0)
            @variable(model, tg2_eta[1:T] .>= 0)

            @variable(model, tg2_epsilon[1:T, 1:M])
            @variable(model, tg2_psi[1:T, 1:M])
            @variable(model, tg2_z[1:M])
            @variable(model, tg2_y[1:M] .>= 0)

            tg2_w = -owa_tg(T; alpha_i = beta_i, alpha = beta, a_sim = b_sim)

            tg2_s = sum(tg2_w)
            tg2_l = minimum(tg2_w)
            tg2_h = maximum(tg2_w)

            tg2_d = [norm(tg2_w, p) for p ∈ owa_p]

            @expression(model, rtg_risk,
                        tg_risk + tg2_s * tg2_t - tg2_l * sum(tg2_nu) +
                        tg2_h * sum(tg2_eta) +
                        dot(tg2_d, tg2_y))

            @constraint(model,
                        -model[:hist_ret] .+ tg2_t .- tg2_nu .+ tg2_eta .-
                        vec(sum(tg2_epsilon; dims = 2)) .== 0)

            @constraint(model, tg2_z .+ tg2_y .== vec(sum(tg2_psi; dims = 1)))

            @constraint(model, [i = 1:M, j = 1:T],
                        [-tg2_z[i] * owa_p[i], tg2_psi[j, i] * owa_p[i] / (owa_p[i] - 1),
                         tg2_epsilon[j, i]] ∈ MOI.PowerCone(1 / owa_p[i]))
        end

        if isfinite(rtg_u) && type == :Trad
            if obj == :Sharpe
                @constraint(model, rtg_risk <= rtg_u * model[:k])
            else
                @constraint(model, rtg_risk <= rtg_u)
            end
        end

        if rm == :TGRG
            @expression(model, risk, rtg_risk)
        end
    end

    if !(rm == :OWA || isfinite(owa_u))
        return nothing
    end

    if !owa_approx
        @variable(model, owa_a[1:T])
        @variable(model, owa_b[1:T])
        @expression(model, owa_risk, sum(owa_a .+ owa_b))

        owa_w = isempty(portfolio.owa_w) ? owa_gmd(T) : portfolio.owa_w

        @constraint(model,
                    owa * transpose(owa_w) .<=
                    onesvec * transpose(owa_a) + owa_b * transpose(onesvec))
    else
        @variable(model, owa_t)
        @variable(model, owa_nu[1:T] .>= 0)
        @variable(model, owa_eta[1:T] .>= 0)

        @variable(model, owa_epsilon[1:T, 1:M])
        @variable(model, owa_psi[1:T, 1:M])
        @variable(model, owa_z[1:M])
        @variable(model, owa_y[1:M] .>= 0)

        owa_w = isempty(portfolio.owa_w) ? -owa_gmd(T) : -portfolio.owa_w

        owa_s = sum(owa_w)
        owa_l = minimum(owa_w)
        owa_h = maximum(owa_w)

        owa_d = [norm(owa_w, p) for p ∈ owa_p]

        @expression(model, owa_risk,
                    owa_s * owa_t - owa_l * sum(owa_nu) +
                    owa_h * sum(owa_eta) +
                    dot(owa_d, owa_y))

        @constraint(model,
                    model[:hist_ret] .+ owa_t .- owa_nu .+ owa_eta .-
                    vec(sum(owa_epsilon; dims = 2)) .== 0)

        @constraint(model, owa_z .+ owa_y .== vec(sum(owa_psi; dims = 1)))

        @constraint(model, [i = 1:M, j = 1:T],
                    [-owa_z[i] * owa_p[i], owa_psi[j, i] * owa_p[i] / (owa_p[i] - 1),
                     owa_epsilon[j, i]] ∈ MOI.PowerCone(1 / owa_p[i]))
    end

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

function _calc_factors_b1_b2_b3(B::DataFrame, factors::AbstractMatrix,
                                loadings_opt::LoadingsOpt = LoadingsOpt(;))
    namesB = names(B)

    B = Matrix(B[!, setdiff(namesB, ("tickers", "const"))])

    if loadings_opt.method == :MVR
        mvr_opt = loadings_opt.mvr_opt

        std_genfunc = mvr_opt.std_genfunc
        pca_s_genfunc = mvr_opt.pca_s_genfunc
        pca_genfunc = mvr_opt.pca_genfunc

        X = transpose(factors)

        pca_s_func = pca_s_genfunc.func
        pca_s_args = pca_s_genfunc.args
        pca_s_kwargs = pca_s_genfunc.kwargs
        X_std = pca_s_func(pca_s_args..., X; pca_s_kwargs...)

        pca_func = pca_genfunc.func
        pca_args = pca_genfunc.args
        pca_kwargs = pca_genfunc.kwargs
        model = pca_func(pca_args..., X_std; pca_kwargs...)
        Vp = projection(model)

        std_func = std_genfunc.func
        std_args = std_genfunc.args
        std_kwargs = std_genfunc.kwargs
        sdev = vec(std_func(X, std_args...; std_kwargs...))

        B = transpose(pinv(Vp) * transpose(B .* transpose(sdev)))
    end

    b1 = pinv(transpose(B))
    b2 = pinv(transpose(nullspace(transpose(B))))
    b3 = pinv(transpose(b2))

    return b1, b2, b3, B
end

function _rp_setup(portfolio, N, class, nullflag)
    model = portfolio.model

    model = portfolio.model
    @variable(model, k)

    if class != :FC
        if isempty(portfolio.risk_budget)
            portfolio.risk_budget = ()
        elseif !isapprox(sum(portfolio.risk_budget), one(eltype(portfolio.returns)))
            portfolio.risk_budget ./= sum(portfolio.risk_budget)
        end
        rb = portfolio.risk_budget
        @variable(model, log_w[1:N])
        @constraint(model, dot(rb, log_w) >= 1)
        @constraint(model, [i = 1:N], [log_w[i], 1, model[:w][i]] ∈ MOI.ExponentialCone())
        @constraint(model, model[:w] .>= 0)
    else
        b1, b2, missing, missing = _calc_factors_b1_b2_b3(portfolio.loadings,
                                                          portfolio.f_returns,
                                                          portfolio.loadings_opt)

        N_f = size(b1, 2)

        rb = portfolio.f_risk_budget
        if isempty(rb) || length(rb) != N_f
            rb = portfolio.f_risk_budget = fill(1 / N_f, N_f)
        elseif !isapprox(sum(portfolio.f_risk_budget), one(eltype(portfolio.returns)))
            portfolio.f_risk_budget ./= sum(portfolio.f_risk_budget)
        end

        @variable(model, w1[1:N_f])
        delete(model, model[:w])
        unregister(model, :w)
        if nullflag
            @variable(model, w2[1:(N - N_f)])
            @expression(model, w, b1 * w1 + b2 * w2)
        else
            @expression(model, w, b1 * w1)
        end
        @variable(model, log_w[1:N_f])
        @constraint(model, dot(rb, log_w) >= 1)
        @constraint(model, [i = 1:N_f],
                    [log_w[i], 1, model[:w1][i]] ∈ MOI.ExponentialCone())
    end

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
                 model[:w][i] - zeta[i]] ∈ SecondOrderCone())
    # RRP version constraints
    if rrp_ver ∈ (:Reg, :Reg_Pen)
        @variable(model, rho)
        @constraint(model, [2 * psi; 2 * G * model[:w]; -2 * rho] ∈ SecondOrderCone())
    end

    if rrp_ver == :None
        @constraint(model, [psi; G * model[:w]] ∈ SecondOrderCone())
    elseif rrp_ver == :Reg
        @constraint(model, [rho; G * model[:w]] ∈ SecondOrderCone())
    elseif rrp_ver == :Reg_Pen
        theta = Diagonal(sqrt.(diag(sigma)))
        @constraint(model, [rho; sqrt(rrp_penalty) * theta * model[:w]] ∈ SecondOrderCone())
    end

    return nothing
end

function _wc_setup(portfolio, kelly, obj, T, N, rf, mu, sigma, u_mu, u_cov, network_method,
                   sd_cone::Bool = true)
    model = portfolio.model

    # Return uncertainy sets.
    _sdp_setup(portfolio, obj, rm, :WC, N, u_cov)
    if kelly == :Approx || u_cov ∉ (:Box, :Ellipse)
        _mv_risk(model, sigma, :WC, network_method, sd_cone)
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
            @constraint(model, [i = 1:N], [abs_w[i]; model[:w][i]] ∈ MOI.NormOneCone(2))
            @expression(model, ret, model[:_ret] - dot(d_mu, abs_w))
        elseif u_mu == :Ellipse
            k_mu = portfolio.k_mu
            cov_mu = portfolio.cov_mu
            G = sqrt(cov_mu)
            @expression(model, x_gw, G * model[:w])
            @variable(model, t_gw)
            @constraint(model, [t_gw; x_gw] ∈ SecondOrderCone())
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
        @constraint(model, Au .- Al .== model[:W])
        @expression(model, risk, tr(Au * cov_u) - tr(Al * cov_l))
    elseif u_cov == :Ellipse
        k_sigma = portfolio.k_sigma
        G_sigma = sqrt(portfolio.cov_sigma)

        @variable(model, E[1:N, 1:N], Symmetric)
        @constraint(model, E ∈ PSDCone())

        @expression(model, W_p_E, model[:W] .+ E)
        @expression(model, x_ge, G_sigma * vec(W_p_E))
        @variable(model, t_ge)
        @constraint(model, [t_ge; x_ge] ∈ SecondOrderCone())
        @expression(model, risk, tr(sigma * W_p_E) + k_sigma * t_ge)
    else
        @expression(model, risk, model[:dev_risk])
    end

    if obj == :Sharpe
        if kelly != :None
            @constraint(model, model[:risk] <= 1)
        else
            @constraint(model, model[:ret] - rf * model[:k] >= 1)
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
    elseif !isapprox(sum(portfolio.risk_budget), one(eltype(portfolio.returns)))
        portfolio.risk_budget ./= sum(portfolio.risk_budget)
    end
    @variable(model, k)
    return nothing
end

function _setup_ret(kelly, model, T, returns, mu, mu_l)
    if kelly == :Exact
        @variable(model, texact_kelly[1:T])
        @expression(model, _ret, sum(texact_kelly) / T)
        @expression(model, kret, 1 .+ returns * model[:w])
        @constraint(model, [i = 1:T], [texact_kelly[i], 1, kret[i]] ∈ MOI.ExponentialCone())
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
                    [texact_kelly[i], model[:k], kret[i]] ∈ MOI.ExponentialCone())
        if trad
            @constraint(model, model[:risk] <= 1)
        end
    elseif kelly == :Approx && !isempty(mu)
        @variable(model, tapprox_kelly)
        @constraint(model,
                    [model[:k] + tapprox_kelly
                     2 * model[:dev]
                     model[:k] - tapprox_kelly] ∈ SecondOrderCone())
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
    num_assets_u = portfolio.num_assets_u
    scale_nau = portfolio.num_assets_u_scale
    scale_nip = portfolio.network_ip_scale
    short = portfolio.short
    short_u = portfolio.short_u
    long_u = portfolio.long_u
    budget = portfolio.budget
    network_method = portfolio.network_method
    network_ip = portfolio.network_ip

    model = portfolio.model

    # Boolean variables max number of assets.
    if num_assets_u > 0
        if obj == :Sharpe
            @variable(model, tnau_bin[1:N], binary = true)
            @variable(model, tnau_bin_sharpe[1:N] >= 0)
        else
            @variable(model, tnau_bin[1:N], binary = true)
        end
        @constraint(model, sum(tnau_bin) <= num_assets_u)
    end
    if network_method == :IP
        if obj == :Sharpe
            @variable(model, tnau_bin2[1:N], binary = true)
            @variable(model, tnau_bin_sharpe2[1:N] >= 0)
        else
            @variable(model, tnau_bin2[1:N], binary = true)
        end
        @constraint(model, unique(network_ip + I; dims = 1) * tnau_bin2 .<= 1)
    end

    # Weight constraints.
    if obj == :Sharpe
        @constraint(model, sum(model[:w]) == budget * model[:k])
        if haskey(model, :tnau_bin)
            @constraint(model, tnau_bin_sharpe .<= model[:k])
            @constraint(model, tnau_bin_sharpe .<= scale_nau * tnau_bin)
            @constraint(model, tnau_bin_sharpe .>= model[:k] .- scale_nau * (1 .- tnau_bin))
            @constraint(model, model[:w] .<= long_u * tnau_bin_sharpe)
        end

        if haskey(model, :tnau_bin2)
            @constraint(model, tnau_bin_sharpe2 .<= model[:k])
            @constraint(model, tnau_bin_sharpe2 .<= scale_nip * tnau_bin2)
            @constraint(model,
                        tnau_bin_sharpe2 .>= model[:k] .- scale_nip * (1 .- tnau_bin2))
            @constraint(model, model[:w] .<= long_u * tnau_bin_sharpe2)
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
            if num_assets_u > 0
                @constraint(model, model[:w] .>= -short_u * tnau_bin_sharpe)
            end

            if network_method == :IP
                @constraint(model, model[:w] .>= -short_u * tnau_bin_sharpe2)
            end
        end
    else
        @constraint(model, sum(model[:w]) == budget)

        if haskey(model, :tnau_bin)
            @constraint(model, model[:w] .<= long_u * tnau_bin)
        end

        if haskey(model, :tnau_bin2)
            @constraint(model, model[:w] .<= long_u * tnau_bin2)
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
            if num_assets_u > 0
                @constraint(model, model[:w] .>= -short_u * tnau_bin)
            end

            if network_method == :IP
                @constraint(model, model[:w] .>= -short_u * tnau_bin2)
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
        @constraint(model, dot(A, model[:w]) - B * model[:k] == 0)
    else
        @constraint(model, dot(A, model[:w]) - B == 0)
    end

    return nothing
end

function _setup_min_number_effective_assets(portfolio, obj)
    nal = portfolio.num_assets_l

    if iszero(nal)
        return nothing
    end

    model = portfolio.model

    @variable(model, tnal >= 0)
    @constraint(model, [tnal; model[:w]] ∈ SecondOrderCone())

    if obj == :Sharpe
        @constraint(model, tnal * sqrt(nal) <= model[:k])
    else
        @constraint(model, tnal * sqrt(nal) <= 1)
    end

    return nothing
end

function _setup_tracking_err(portfolio, returns, obj, T)
    tracking_err = portfolio.tracking_err
    kind_tracking_err = portfolio.kind_tracking_err
    tracking_err_weights = portfolio.tracking_err_weights
    tracking_err_returns = portfolio.tracking_err_returns

    if kind_tracking_err == :None ||
       isinf(tracking_err) ||
       isfinite(tracking_err) &&
       (kind_tracking_err == :Weights && isempty(tracking_err_weights) ||
        kind_tracking_err == :Returns && isempty(tracking_err_returns))
        return nothing
    end

    benchmark = if kind_tracking_err == :Weights
        returns * tracking_err_weights
    elseif kind_tracking_err == :Returns
        tracking_err_returns
    end

    model = portfolio.model

    @variable(model, t_track_err >= 0)
    if obj == :Sharpe
        @expression(model, track_err, returns * model[:w] .- benchmark * model[:k])
        @constraint(model, [t_track_err; track_err] ∈ SecondOrderCone())
        @constraint(model, t_track_err <= tracking_err * model[:k] * sqrt(T - 1))
    else
        @expression(model, track_err, returns * model[:w] .- benchmark)
        @constraint(model, [t_track_err; track_err] ∈ SecondOrderCone())
        @constraint(model, t_track_err <= tracking_err * sqrt(T - 1))
    end

    return nothing
end

function _setup_turnover(portfolio, N, obj)
    turnover = portfolio.turnover
    turnover_weights = portfolio.turnover_weights

    if isa(turnover, Real) && isinf(turnover) ||
       isa(turnover, AbstractVector) && isempty(turnover) ||
       isempty(turnover_weights)
        return nothing
    end

    model = portfolio.model

    @variable(model, t_turnov[1:N] >= 0)
    if obj == :Sharpe
        @expression(model, turnov, model[:w] .- turnover_weights * model[:k])
        @constraint(model, [i = 1:N], [t_turnov[i]; turnov[i]] ∈ MOI.NormOneCone(2))
        @constraint(model, t_turnov .<= turnover * model[:k])
    else
        @expression(model, turnov, model[:w] .- turnover_weights)
        @constraint(model, [i = 1:N], [t_turnov[i]; turnov[i]] ∈ MOI.NormOneCone(2))
        @constraint(model, t_turnov .<= turnover)
    end

    return nothing
end

function _setup_rebalance(portfolio, N, obj)
    rebalance = portfolio.rebalance
    rebalance_weights = portfolio.rebalance_weights
    if isa(rebalance, Real) && (isinf(rebalance) || iszero(rebalance)) ||
       isa(rebalance, AbstractVector) && isempty(rebalance) ||
       isempty(rebalance_weights)
        return
    end

    model = portfolio.model

    @variable(model, t_rebal[1:N] >= 0)
    if obj == :Sharpe
        @expression(model, rebal, model[:w] .- rebalance_weights * model[:k])
        @constraint(model, [i = 1:N], [t_rebal[i]; rebal[i]] ∈ MOI.NormOneCone(2))
        @expression(model, sum_t_rebal, sum(rebalance .* t_rebal))
    else
        @expression(model, rebal, model[:w] .- rebalance_weights)
        @constraint(model, [i = 1:N], [t_rebal[i]; rebal[i]] ∈ MOI.NormOneCone(2))
        @expression(model, sum_t_rebal, sum(rebalance .* t_rebal))
    end
end

function _setup_trad_wc_constraints(portfolio, obj, T, N, type, class, kelly, l, returns)
    _setup_centrality_constraints(portfolio, obj)
    _setup_weights(portfolio, obj, N)
    _setup_min_number_effective_assets(portfolio, obj)
    _setup_tracking_err(portfolio, returns, obj, T)
    _setup_turnover(portfolio, N, obj)
    _setup_rebalance(portfolio, N, obj)
    _setup_trad_wc_objective_function(portfolio, type, obj, class, kelly, l)
    return nothing
end

function _setup_trad_wc_objective_function(portfolio, type, obj, class, kelly, l)
    model = portfolio.model

    npf = if type == :Trad && haskey(model, :network_penalty_factor)
        model[:network_penalty_factor]
    else
        zero(eltype(portfolio.returns))
    end

    rbf = if haskey(model, :sum_t_rebal)
        model[:sum_t_rebal]
    else
        zero(eltype(portfolio.returns))
    end

    if obj == :Sharpe
        if (type == :Trad && class == :Classic || type == :WC) && kelly != :None
            @objective(model, Max, model[:ret] - npf - rbf)
        else
            @objective(model, Min, model[:risk] + npf + rbf)
        end
    elseif obj == :Min_Risk
        @objective(model, Min, model[:risk] + npf + rbf)
    elseif obj == :Utility
        @objective(model, Max, model[:ret] - l * model[:risk] - npf - rbf)
    elseif obj == :Max_Ret
        @objective(model, Max, model[:ret] - npf - rbf)
    end
    return nothing
end

function _optimise_portfolio(portfolio, class, type, obj, near_opt = false)
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
        all_non_zero_weights = !all(isapprox.(abs.(value.(model[:w])), zero(rtype)))

        if term_status ∈ ValidTermination && all_finite_weights && all_non_zero_weights
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
            budget = portfolio.budget
            if short == false
                sum_w = sum(abs.(weights))
                sum_w = sum_w > eps() ? sum_w : 1
                weights .= abs.(weights) / sum_w * budget
            end
        elseif type == :RP
            weights .= value.(model[:w])
            if class != :FC
                sum_w = sum(abs.(weights))
                sum_w = sum_w > eps() ? sum_w : 1
                weights .= abs.(weights) / sum_w
            else
                sum_w = value(model[:k])
                sum_w = abs(sum_w) > eps() ? sum_w : 1
                weights .= weights / sum_w
            end
        elseif type == :RRP
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

    return term_status, solvers_tried
end

function _finalise_portfolio(portfolio, class, returns, N, solvers_tried, type, rm, obj,
                             near_opt = false)
    model = portfolio.model

    strtype = if !near_opt
        String(type)
    else
        tmp = "Near_"
        tmp * String(type)
    end

    if type ∈ (:Trad, :RP) && rm ∈ (:EVaR, :EDaR, :RLVaR, :RLDaR)
        z_key = "z_" * lowercase(string(rm))
        z_key2 = Symbol(strtype * "_" * z_key)
        portfolio.z[z_key2] = value(portfolio.model[Symbol(z_key)])
        if type == :Trad && obj == :Sharpe
            portfolio.z[z_key2] /= value(portfolio.model[:k])
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
        budget = portfolio.budget
        if short == false
            sum_w = sum(abs.(weights))
            sum_w = sum_w > eps() ? sum_w : 1
            weights .= abs.(weights) / sum_w * budget
        end
    elseif type == :RP
        weights .= value.(model[:w])
        if class != :FC
            sum_w = sum(abs.(weights))
            sum_w = sum_w > eps() ? sum_w : 1
            weights .= abs.(weights) / sum_w
        else
            sum_w = value(model[:k])
            sum_w = abs(sum_w) > eps() ? sum_w : 1
            weights .= weights / sum_w
        end
    elseif type == :RRP
        weights .= value.(model[:w])
        sum_w = sum(abs.(weights))
        sum_w = sum_w > eps() ? sum_w : 1
        weights .= abs.(weights) / sum_w
    end

    if near_opt
        type = Symbol(strtype)
    end

    portfolio.optimal[type] = DataFrame(; tickers = portfolio.assets, weights = weights)

    isempty(solvers_tried) ? portfolio.fail = Dict() : portfolio.fail = solvers_tried

    return portfolio.optimal[type]
end

function _handle_errors_and_finalise(portfolio, class, term_status, returns, N,
                                     solvers_tried, type, rm, obj, near_opt = false)
    model = portfolio.model

    retval = if term_status ∉ ValidTermination ||
                any(.!isfinite.(value.(model[:w]))) ||
                all(isapprox.(abs.(value.(model[:w])), zero(eltype(portfolio.returns))))
        funcname = "$(fullname(PortfolioOptimiser)[1]).$(nameof(PortfolioOptimiser.optimise!))"
        @warn("$funcname: model could not be optimised satisfactorily.\nSolvers: $solvers_tried.")
        portfolio.fail = solvers_tried
        if near_opt
            tmp = "Near_"
            type = Symbol(tmp * String(type))
        end
        portfolio.optimal[type] = DataFrame()
    else
        _finalise_portfolio(portfolio, class, returns, N, solvers_tried, type, rm, obj,
                            near_opt)
    end

    return retval
end

function _p_save_opt_params(portfolio, opt, string_names, save_opt_params)
    if !save_opt_params
        return nothing
    end

    portfolio.opt_params[opt.type] = Dict(:opt => opt, :string_names => string_names,
                                          :save_opt_params => save_opt_params)

    return nothing
end

function _setup_model_class(portfolio, class, hist)
    if class ∉ (:Classic, :FC)
        @smart_assert(hist ∈ ClassHist)
    end

    if class ∈ (:Classic, :FC)
        mu = portfolio.mu
        sigma = portfolio.cov
        returns = portfolio.returns
    elseif class == :FM
        mu = portfolio.fm_mu
        if hist == 1
            sigma = portfolio.fm_cov
            returns = portfolio.fm_returns
        elseif hist == 2
            sigma = portfolio.cov
            returns = portfolio.returns
        else
            throw(AssertionError("for class = $class, hist = $hist can only be 1 or 2"))
        end
    elseif class == :BL
        mu = portfolio.bl_mu
        returns = portfolio.returns
        if hist == 1
            sigma = portfolio.bl_cov
        elseif hist == 2
            sigma = portfolio.cov
        else
            throw(AssertionError("for class = $class, hist = $hist can only be 1 or 2"))
        end
    elseif class == :BLFM
        mu = portfolio.blfm_mu
        if hist == 1
            sigma = portfolio.blfm_cov
            returns = portfolio.fm_returns
        elseif hist == 2
            sigma = portfolio.cov
            returns = portfolio.returns
        else
            sigma = portfolio.fm_cov
            returns = portfolio.fm_returns
        end
    end

    return mu, sigma, returns
end

function _near_optimal_centering(portfolio, class, mu, returns, sigma, w_opt, T, N, opt)
    type = opt.type
    rm = opt.rm
    obj = opt.obj
    rf = opt.rf
    n = opt.n
    w1 = opt.w_min
    w2 = opt.w_max

    if isempty(w1) || isempty(w2)
        fl = frontier_limits!(portfolio, opt; save_model = true)
        w1 = fl.w_min
        w2 = fl.w_max
    end
    w3 = w_opt.weights

    alpha_i = portfolio.alpha_i
    alpha = portfolio.alpha
    a_sim = portfolio.a_sim
    beta_i = portfolio.beta_i
    beta = portfolio.beta
    b_sim = portfolio.b_sim
    kappa = portfolio.kappa
    owa_w = portfolio.owa_w
    V = portfolio.V
    SV = portfolio.SV
    solvers = portfolio.solvers

    risk1, risk2 = _ul_risk(rm, returns, w1, w2, sigma, rf, solvers, alpha, kappa, alpha_i,
                            beta, a_sim, beta_i, b_sim, owa_w, V, SV, 0)

    risk3 = calc_risk(w3, returns; rm = rm, rf = rf, sigma = sigma, alpha_i = alpha_i,
                      alpha = alpha, a_sim = a_sim, beta_i = beta_i, beta = beta,
                      b_sim = b_sim, kappa = kappa, owa_w = owa_w, V = V, SV = SV,
                      solvers = solvers)

    if opt.kelly == :None
        ret1 = dot(mu, w1)
        ret2 = dot(mu, w2)
        ret3 = dot(mu, w3)
    else
        ret1 = sum(log.(one(risk1) .+ returns * w1)) / T
        ret2 = sum(log.(one(risk2) .+ returns * w2)) / T
        ret3 = sum(log.(one(risk3) .+ returns * w3)) / T
    end

    c1 = (ret2 - ret1) / n
    c2 = (risk2 - risk1) / n
    e1 = ret3 - c1
    e2 = risk3 + c2

    # Make the model from scratch. Make dev_risk = dev, skew_risk = t_skew, and sskew_risk = t_sskew. Rather than their original squared form.
    # Remake the return and risk expressions. 
    # The only constraints for this model should be the ones here since we are going from the original portfolio weights.
    # The starting weights are already the result of all the constraints. Take them out. Maybe we should split this into its own optimise function.
    model = portfolio.model

    set_start_value.(model[:w], w3)

    # @constraint(model, model[:ret] >= e1)
    # @constraint(model, model[:risk] <= e2)
    @variable(model, log_ret)
    @constraint(model, [-log_ret, 1, model[:ret] - e1] ∈ MOI.ExponentialCone())
    @variable(model, log_risk)
    @constraint(model, [-log_risk, 1, e2 - model[:risk]] ∈ MOI.ExponentialCone())
    @variable(model, log_w[1:N])
    @constraint(model, [i = 1:N], [log_w[i], 1, model[:w][i]] ∈ MOI.ExponentialCone())
    @variable(model, log_omw[1:N])
    @constraint(model, [i = 1:N], [log_omw[i], 1, 1 - model[:w][i]] ∈ MOI.ExponentialCone())
    @expression(model, neg_sum_log_ws, -sum(log_w .+ log_omw))
    @expression(model, near_opt_risk, log_ret + log_risk + neg_sum_log_ws)
    @objective(model, Min, near_opt_risk)

    term_status, solvers_tried = _optimise_portfolio(portfolio, class, type, obj, true)
    retval = _handle_errors_and_finalise(portfolio, class, term_status, returns, N,
                                         solvers_tried, type, rm, obj, true)

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
function optimise!(portfolio::Portfolio, opt::OptimiseOpt = OptimiseOpt(;);
                   string_names::Bool = false, save_opt_params::Bool = false)
    type = opt.type
    rm = opt.rm
    obj = opt.obj
    kelly = opt.kelly
    class = opt.class
    nullflag = opt.nullflag
    rrp_ver = opt.rrp_ver
    u_cov = opt.u_cov
    u_mu = opt.u_mu
    sd_cone = opt.sd_cone
    owa_approx = opt.owa_approx
    near_opt = opt.near_opt
    hist = opt.hist
    rf = opt.rf
    l = opt.l
    rrp_penalty = opt.rrp_penalty
    w_ini = opt.w_ini
    w_min = opt.w_min
    w_max = opt.w_max

    if near_opt
        w_min = opt.w_min
        if !isempty(w_min)
            @smart_assert(length(w_min) == size(portfolio.returns, 2))
        end
        w_max = opt.w_max
        if !isempty(w_max)
            @smart_assert(length(w_max) == size(portfolio.returns, 2))
        end
    end
    _p_save_opt_params(portfolio, opt, string_names, save_opt_params)

    mu, sigma, returns = _setup_model_class(portfolio, class, hist)
    T, N = size(returns)
    kurtosis = portfolio.kurt
    skurtosis = portfolio.skurt
    network_method = portfolio.network_method

    portfolio.model = JuMP.Model()
    model = portfolio.model
    set_string_names_on_creation(model, string_names)
    @variable(model, w[1:N])

    if !isempty(w_ini)
        @smart_assert(length(w_ini) == size(portfolio.returns, 2))
        set_start_value.(w, w_ini)
    end

    if type == :Trad
        _setup_sharpe_k(model, obj)
        _risk_setup(portfolio, :Trad, rm, kelly, obj, rf, T, N, mu, returns, sigma,
                    kurtosis, skurtosis, network_method, sd_cone, owa_approx)
        _setup_trad_return(portfolio, class, kelly, obj, T, rf, returns, mu)
        _setup_trad_wc_constraints(portfolio, obj, T, N, :Trad, class, kelly, l, returns)
    elseif type == :RP
        _rp_setup(portfolio, N, class, nullflag)
        _risk_setup(portfolio, :RP, rm, kelly, obj, rf, T, N, mu, returns, sigma, kurtosis,
                    skurtosis, network_method, sd_cone, owa_approx)
        _setup_rp_rrp_return_and_obj(portfolio, kelly, T, returns, mu)
    elseif type == :RRP
        _setup_risk_budget(portfolio)
        _mv_setup(portfolio, sigma, rm, kelly, obj, :RRP, network_method, sd_cone)
        _rrp_setup(portfolio, sigma, N, rrp_ver, rrp_penalty)
        _setup_rp_rrp_return_and_obj(portfolio, kelly, T, returns, mu)
    else
        _setup_sharpe_k(model, obj)
        _wc_setup(portfolio, kelly, obj, T, N, rf, mu, sigma, u_mu, u_cov, network_method,
                  sd_cone)
        _setup_trad_wc_constraints(portfolio, obj, T, N, :WC, class, kelly, l, returns)
    end

    _setup_linear_constraints(portfolio, obj, type)

    term_status, solvers_tried = _optimise_portfolio(portfolio, class, type, obj)
    retval = _handle_errors_and_finalise(portfolio, class, term_status, returns, N,
                                         solvers_tried, type, rm, obj)

    if near_opt && type ∈ (:Trad, :WC)
        retval = _near_optimal_centering(portfolio, class, mu, returns, sigma, retval, T, N,
                                         opt)
    end

    return retval
end

"""
```julia
frontier_limits!(portfolio::Portfolio; class::Symbol = :Classic, hist::Integer = 1,
                 kelly::Symbol = :None, rf::Real = 0.0, rm::Symbol = :SD,
                 save_model::Bool = false)
```
"""
function frontier_limits!(portfolio::Portfolio, opt::OptimiseOpt = OptimiseOpt(;);
                          save_model::Bool = false)
    obj1 = opt.obj
    near_opt1 = opt.near_opt
    opt.near_opt = false
    optimal1 = deepcopy(portfolio.optimal)
    fail1 = deepcopy(portfolio.fail)
    if save_model
        model1 = copy(portfolio.model)
    end

    opt.obj = :Min_Risk
    w_min = optimise!(portfolio, opt)

    opt.obj = :Max_Ret
    w_max = optimise!(portfolio, opt)

    limits = hcat(w_min, DataFrame(; x1 = w_max[!, 2]))
    DataFrames.rename!(limits, :weights => :w_min, :x1 => :w_max)
    portfolio.limits[opt.rm] = limits

    opt.obj = obj1
    opt.near_opt = near_opt1
    portfolio.optimal = optimal1
    portfolio.fail = fail1
    if save_model
        portfolio.model = model1
    end

    return portfolio.limits[opt.rm]
end

"""
```julia
efficient_frontier!(portfolio::Portfolio; class::Symbol = :Classic, hist::Integer = 1,
                    kelly::Symbol = :None, rf::Real = 0.0, rm::Symbol = :SD,
                    points::Integer = 20)
```
"""
function efficient_frontier!(portfolio::Portfolio, opt::OptimiseOpt = OptimiseOpt(;);
                             points::Integer = 20)
    @smart_assert(opt.type == :Trad)
    obj1 = opt.obj
    w_ini1 = opt.w_ini
    optimal1 = deepcopy(portfolio.optimal)
    fail1 = deepcopy(portfolio.fail)

    class = opt.class
    hist = opt.hist
    mu, sigma, returns = _setup_model_class(portfolio, class, hist)

    fl = frontier_limits!(portfolio, opt)

    w1 = fl.w_min
    w2 = fl.w_max

    if opt.kelly == :None
        ret1 = dot(mu, w1)
        ret2 = dot(mu, w2)
    else
        ret1 = sum(log.(one(eltype(mu)) .+ returns * w1)) / size(returns, 1)
        ret2 = sum(log.(one(eltype(mu)) .+ returns * w2)) / size(returns, 1)
    end

    alpha_i = portfolio.alpha_i
    alpha = portfolio.alpha
    a_sim = portfolio.a_sim
    beta_i = portfolio.beta_i
    beta = portfolio.beta
    b_sim = portfolio.b_sim
    kappa = portfolio.kappa
    owa_w = portfolio.owa_w
    V = portfolio.V
    SV = portfolio.SV
    solvers = portfolio.solvers

    rm = opt.rm
    rf = opt.rf

    risk1, risk2 = _ul_risk(rm, returns, w1, w2, sigma, rf, solvers, alpha, kappa, alpha_i,
                            beta, a_sim, beta_i, b_sim, owa_w, V, SV, 0)

    mus = range(ret1; stop = ret2, length = points)
    risks = range(risk1; stop = risk2, length = points)

    rmf = Symbol(lowercase(string(rm)) * "_u")

    frontier = Vector{typeof(risk1)}(undef, 0)
    srisk = Vector{typeof(risk1)}(undef, 0)
    w_ini = Vector{typeof(risk1)}(undef, 0)

    i = 0
    for (j, (r, m)) ∈ enumerate(zip(risks, mus))
        if i == 0
            opt.obj = :Min_Risk
            w = optimise!(portfolio, opt)
        else
            if !isempty(w)
                opt.w_ini = w.weights
            end
            if j != length(risks)
                setproperty!(portfolio, rmf, r)
            else
                setproperty!(portfolio, rmf, Inf)
            end
            opt.obj = :Max_Ret
            w = optimise!(portfolio, opt)
            # Fallback in case :Max_Ret with maximum risk bounds fails.
            if isempty(w)
                opt.obj = :Min_Risk
                setproperty!(portfolio, rmf, Inf)
                portfolio.mu_l = m
                w = optimise!(portfolio, opt)
                portfolio.mu_l = Inf
            end
        end
        if isempty(w)
            continue
        end
        rk = calc_risk(w.weights, returns; rm = rm, rf = rf, sigma = sigma,
                       alpha_i = alpha_i, alpha = alpha, a_sim = a_sim, beta_i = beta_i,
                       beta = beta, b_sim = b_sim, kappa = kappa, owa_w = owa_w, V = V,
                       SV = SV, solvers = solvers)

        append!(frontier, w.weights)
        push!(srisk, rk)
        i += 1
    end
    setproperty!(portfolio, rmf, Inf)

    opt.obj = :Sharpe
    w = optimise!(portfolio, opt)
    sharpe = false
    if !isempty(w)
        rk = calc_risk(w.weights, returns; rm = rm, rf = rf, sigma = sigma,
                       alpha_i = alpha_i, alpha = alpha, a_sim = a_sim, beta_i = beta_i,
                       beta = beta, b_sim = b_sim, kappa = kappa, owa_w = owa_w, V = V,
                       SV = SV, solvers = solvers)
        append!(frontier, w.weights)
        push!(srisk, rk)
        i += 1
        sharpe = true
    end

    key = if opt.near_opt
        Symbol("Near_" * string(rm))
    else
        rm
    end

    portfolio.frontier[key] = Dict(:weights => hcat(DataFrame(; tickers = portfolio.assets),
                                                    DataFrame(reshape(frontier, length(w1),
                                                                      :),
                                                              string.(range(1, i)))),
                                   :opt => opt, :points => points, :risk => srisk,
                                   :sharpe => sharpe)

    opt.obj = obj1
    opt.w_ini = w_ini1
    portfolio.optimal = optimal1
    portfolio.fail = fail1

    return portfolio.frontier[key]
end
