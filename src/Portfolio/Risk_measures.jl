function mv(w, cov)
    return dot(w, cov, w)
end

function msd(w, cov)
    return sqrt(mv(w, cov))
end

function mad(x)
    mu = mean(x)
    return mean(abs.(x .- mu))
end

function msv(x)
    T = length(x)
    mu = mean(x)
    val = mu .- x
    return sqrt(sum(val[val .>= 0] .^ 2) / (T - 1))
end

function flpm(x, min_ret = 0.0)
    T = length(x)
    val = min_ret .- x
    return sum(val[val .>= 0]) / T
end

function slpm(x, min_ret = 0.0)
    T = length(x)
    val = min_ret .- x
    val = sum(val[val .>= 0] .^ 2) / (T - 1)
    return sqrt(val)
end

function wr(x)
    return -minimum(x)
end

function var(x, alpha = 0.05)
    sort!(x)
    idx = ceil(Int, alpha * length(x))
    return -x[idx]
end

function cvar(x, alpha = 0.05)
    sort!(x)
    idx = ceil(Int, alpha * length(x))
    var = -x[idx]
    sum_var = 0.0
    for i in 1:(idx - 1)
        sum_var += x[i] + var
    end
    return var - sum_var / (alpha * length(x))
end

function _optimize_rm(model, solvers)
    term_status = termination_status(model)
    solvers_tried = Dict()

    for (key, val) in solvers
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

        term_status in ValidTermination && break

        push!(
            solvers_tried,
            key => Dict(
                :objective_val => objective_value(model),
                :term_status => term_status,
                :params => haskey(val, :params) ? val[:params] : missing,
            ),
        )
    end

    return solvers_tried
end

function _entropic_rm(x, solvers, alpha = 0.05)
    model = JuMP.Model()
    set_string_names_on_creation(model, false)

    T = length(x)
    at = alpha * T
    @variable(model, t)
    @variable(model, z >= 0)
    @variable(model, u[1:T])
    @constraint(model, sum(u) <= z)
    @constraint(model, [i = 1:T], [-x[i] - t, z, u[i]] in MOI.ExponentialCone())
    @expression(model, risk, t - z * log(at))
    @objective(model, Min, risk)

    solvers_tried = _optimize_rm(model, solvers)

    term_status = termination_status(model)
    obj_val = objective_value(model)

    if term_status ∉ ValidTermination || !isfinite(obj_val)
        funcname = "$(fullname(PortfolioOptimiser)[1]).$(nameof(PortfolioOptimiser._entropic_rm))"
        @warn(
            "$funcname: model could not be optimised satisfactorily. Solvers: $solvers_tried"
        )
    end

    return obj_val
end

function evar(x, solvers, alpha = 0.05)
    return _entropic_rm(x, solvers, alpha)
end

function _relativistic_rm(x, solvers, alpha = 0.05, kappa = 0.3)
    model = JuMP.Model()
    set_string_names_on_creation(model, false)

    T = length(x)
    at = alpha * T
    invat = 1 / at
    ln_k = (invat^kappa - invat^(-kappa)) / (2 * kappa)
    opk = 1 + kappa
    omk = 1 - kappa
    invkappa2 = 1 / (2 * kappa)
    invk = 1 / kappa
    invopk = 1 / opk
    invomk = 1 / omk

    @variable(model, t)
    @variable(model, z >= 0)
    @variable(model, omega[1:T])
    @variable(model, psi[1:T])
    @variable(model, theta[1:T])
    @variable(model, epsilon[1:T])
    @constraint(
        model,
        [i = 1:T],
        [z * opk * invkappa2, psi[i] * opk * invk, epsilon[i]] in MOI.PowerCone(invopk)
    )
    @constraint(
        model,
        [i = 1:T],
        [omega[i] * invomk, theta[i] * invk, -z * invkappa2] in MOI.PowerCone(omk)
    )
    @constraint(model, -x .- t .+ epsilon .+ omega .<= 0)
    @expression(model, risk, t + ln_k * z + sum(psi .+ theta))
    @objective(model, Min, risk)

    solvers_tried = _optimize_rm(model, solvers)

    term_status = termination_status(model)
    obj_val = objective_value(model)

    if term_status ∉ ValidTermination || !isfinite(obj_val)
        funcname = "$(fullname(PortfolioOptimiser)[1]).$(nameof(PortfolioOptimiser._optimize_entropic_rm))"
        @warn(
            "$funcname: model could not be optimised satisfactorily. Solvers: $solvers_tried"
        )
    end

    return obj_val
end

function rvar(x, solvers, alpha = 0.05, kappa = 0.3)
    return _relativistic_rm(x, solvers, alpha, kappa)
end

function mdd_abs(x)
    insert!(x, 1, 1)
    cs = cumsum(x)
    val = 0.0
    peak = -Inf
    for i in cs
        i > peak && (peak = i)
        dd = peak - i
        dd > val && (val = dd)
    end

    return val
end

function add_abs(x)
    T = length(x)
    insert!(x, 1, 1)
    cs = cumsum(x)
    val = 0.0
    peak = -Inf
    for i in cs
        i > peak && (peak = i)
        dd = peak - i
        dd > 0 && (val += dd)
    end

    return val / T
end

function dar_abs(x, alpha)
    T = length(x)
    insert!(x, 1, 1)
    cs = cumsum(x)
    peak = -Inf
    dd = similar(cs)
    for (idx, i) in enumerate(cs)
        i > peak && (peak = i)
        dd[idx] = i - peak
    end
    deleteat!(dd, 1)
    sort!(dd)
    idx = ceil(Int, alpha * T)
    return -dd[idx]
end

function cdar_abs(x, alpha)
    T = length(x)
    insert!(x, 1, 1)
    cs = cumsum(x)
    peak = -Inf
    dd = similar(cs)
    for (idx, i) in enumerate(cs)
        i > peak && (peak = i)
        dd[idx] = i - peak
    end
    deleteat!(dd, 1)
    sort!(dd)
    idx = ceil(Int, alpha * T)
    var = -dd[idx]
    sum_var = 0.0
    for i in 1:(idx - 1)
        sum_var += dd[i] + var
    end
    return var - sum_var / (alpha * T)
end

function uci_abs(x)
    T = length(x)
    insert!(x, 1, 1)
    cs = cumsum(x)
    val = 0.0
    peak = -Inf
    for i in cs
        i > peak && (peak = i)
        dd = peak - i
        dd > 0 && (val += dd^2)
    end

    return sqrt(val / T)
end

function edar_abs(x, solvers, alpha = 0.05)
    insert!(x, 1, 1)
    cs = cumsum(x)
    peak = -Inf
    dd = similar(cs)
    for (idx, i) in enumerate(cs)
        i > peak && (peak = i)
        dd[idx] = -(peak - i)
    end
    deleteat!(dd, 1)
    return _entropic_rm(dd, solvers, alpha)
end

function rdar_abs(x, solvers, alpha = 0.05, kappa = 0.3)
    insert!(x, 1, 1)
    cs = cumsum(x)
    peak = -Inf
    dd = similar(cs)
    for (idx, i) in enumerate(cs)
        i > peak && (peak = i)
        dd[idx] = i - peak
    end
    deleteat!(dd, 1)
    return _relativistic_rm(dd, solvers, alpha, kappa)
end

function mdd_rel(x)
    x .= insert!(x, 1, 0) .+ 1
    cs = cumprod(x)
    val = 0.0
    peak = -Inf
    for i in cs
        i > peak && (peak = i)
        dd = 1 - i / peak
        dd > val && (val = dd)
    end

    return val
end

function add_rel(x)
    T = length(x)
    x .= insert!(x, 1, 0) .+ 1
    cs = cumprod(x)
    val = 0.0
    peak = -Inf
    for i in cs
        i > peak && (peak = i)
        dd = 1 - i / peak
        dd > 0 && (val += dd)
    end

    return val / T
end

function dar_rel(x, alpha)
    T = length(x)
    x .= insert!(x, 1, 0) .+ 1
    cs = cumprod(x)
    peak = -Inf
    dd = similar(cs)
    for (idx, i) in enumerate(cs)
        i > peak && (peak = i)
        dd[idx] = i / peak - 1
    end
    deleteat!(dd, 1)
    sort!(dd)
    idx = ceil(Int, alpha * T)
    return -dd[idx]
end

function cdar_rel(x, alpha)
    T = length(x)
    x .= insert!(x, 1, 0) .+ 1
    cs = cumprod(x)
    peak = -Inf
    dd = similar(cs)
    for (idx, i) in enumerate(cs)
        i > peak && (peak = i)
        dd[idx] = i / peak - 1
    end
    deleteat!(dd, 1)
    sort!(dd)
    idx = ceil(Int, alpha * T)
    var = -dd[idx]
    sum_var = 0.0
    for i in 1:(idx - 1)
        sum_var += dd[i] + var
    end
    return var - sum_var / (alpha * T)
end

function uci_rel(x)
    T = length(x)
    x .= insert!(x, 1, 0) .+ 1
    cs = cumprod(x)
    val = 0.0
    peak = -Inf
    for i in cs
        i > peak && (peak = i)
        dd = 1 - i / peak
        dd > 0 && (val += dd^2)
    end

    return sqrt(val / T)
end

function edar_rel(x, solvers, alpha)
    x .= insert!(x, 1, 0) .+ 1
    cs = cumprod(x)
    peak = -Inf
    dd = similar(cs)
    for (idx, i) in enumerate(cs)
        i > peak && (peak = i)
        dd[idx] = i / peak - 1
    end
    deleteat!(dd, 1)
    return _entropic_rm(dd, solvers, alpha)
end

function rdar_rel(x, solvers, alpha = 0.05, kappa = 0.3)
    x .= insert!(x, 1, 0) .+ 1
    cs = cumprod(x)
    peak = -Inf
    dd = similar(cs)
    for (idx, i) in enumerate(cs)
        i > peak && (peak = i)
        dd[idx] = i / peak - 1
    end
    deleteat!(dd, 1)
    return _relativistic_rm(dd, solvers, alpha, kappa)
end

function kurt(x)
    T = length(x)
    mu = mean(x)
    val = x .- mu
    return sqrt(sum(val .^ 4) / T)
end

function skurt(x)
    T = length(x)
    mu = mean(x)
    val = x .- mu
    return sqrt(sum(val[val .< 0] .^ 4) / T)
end

function gmd(x)
    T = length(x)
    w = owa_gmd(T)
    return dot(w, sort!(x))
end

function rg(x)
    T = length(x)
    w = owa_rg(T)
    return dot(w, sort!(x))
end

function rcvar(x; alpha = 0.05, beta = nothing)
    T = length(x)
    w = owa_rcvar(T; alpha = alpha, beta = beta)
    return dot(w, sort!(x))
end

function tg(x; alpha_i = 0.0001, alpha = 0.05, a_sim = 100)
    T = length(x)
    w = owa_tg(T; alpha_i = alpha_i, alpha = alpha, a_sim = a_sim)
    return dot(w, sort!(x))
end

function rtg(
    x;
    alpha_i = 0.0001,
    alpha = 0.05,
    a_sim = 100,
    beta_i = nothing,
    beta = nothing,
    b_sim = nothing,
)
    T = length(x)
    w = owa_rtg(
        T;
        alpha_i = alpha_i,
        alpha = alpha,
        a_sim = a_sim,
        beta_i = beta_i,
        beta = beta,
        b_sim = b_sim,
    )
    return dot(w, sort!(x))
end

function owa(x, w)
    return dot(w, sort!(x))
end

function calc_risk(
    w,
    returns,
    cov;
    rm = :mv,
    rf = 0.0,
    alpha_i = 0.0001,
    alpha = 0.05,
    a_sim = 100,
    beta_i = nothing,
    beta = nothing,
    b_sim = nothing,
    kappa = 0.3,
    solvers = nothing,
)
    x = (rm != :mv || rm != :msd) && returns * w

    risk = if rm == :msd
        msd(w, cov)
    elseif rm == :mv
        mv(w, cov)
    elseif rm == :mad
        mad(x)
    elseif rm == :msv
        msv(x)
    elseif rm == :flpm
        flpm(x, rf)
    elseif rm == :slpm
        slpm(x, rf)
    elseif rm == :wr
        wr(x)
    elseif rm == :var
        var(x, alpha)
    elseif rm == :cvar
        cvar(x, alpha)
    elseif rm == :evar
        evar(x, solvers, alpha)
    elseif rm == :rvar
        rvar(x, solvers, alpha, kappa)
    elseif rm == :mdd
        mdd_abs(x)
    elseif rm == :add
        add_abs(x)
    elseif rm == :dar
        dar_abs(x, alpha)
    elseif rm == :cdar
        cdar_abs(x, alpha)
    elseif rm == :uci
        uci_abs(x)
    elseif rm == :edar
        edar_abs(x, solvers, alpha)
    elseif rm == :rdar
        rdar_abs(x, solvers, alpha, kappa)
    elseif rm == :mdd_r
        mdd_rel(x)
    elseif rm == :add_r
        add_rel(x)
    elseif rm == :dar_r
        dar_rel(x, alpha)
    elseif rm == :cdar_r
        cdar_rel(x, alpha)
    elseif rm == :uci_r
        uci_rel(x)
    elseif rm == :edar_r
        edar_rel(x, solvers, alpha)
    elseif rm == :rdar_r
        rdar_rel(x, solvers, alpha, kappa)
    elseif rm == :krt
        kurt(x)
    elseif rm == :skrt
        skurt(x)
    elseif rm == :gmd
        gmd(x)
    elseif rm == :rg
        rg(x)
    elseif rm == :rcvar
        rcvar(x; alpha = alpha, beta = beta)
    elseif rm == :tg
        tg(x; alpha_i = alpha_i, alpha = alpha, a_sim = a_sim)
    elseif rm == :rtg
        rtg(
            x;
            alpha_i = alpha_i,
            alpha = alpha,
            a_sim = a_sim,
            beta_i = beta_i,
            beta = beta,
            b_sim = b_sim,
        )
    elseif rm == :owa
        owa(x, w)
    end

    return risk
end

function calc_risk(portfolio::AbstractPortfolio; type = :trad, rm = :mv, rf = 0.0)
    weights = if isa(portfolio, Portfolio)
        @assert(type ∈ PortTypes, "type must be one of $PortTypes")
        # @assert(rm ∈ RiskMeasures, "rm must be one of $RiskMeasures")
        if type == :trad
            portfolio.p_optimal[!, :weights]
        elseif type == :rp
            portfolio.rp_optimal[!, :weights]
        elseif type == :rrp
            portfolio.rrp_optimal[!, :weights]
        elseif type == :wc
            portfolio.wc_optimal[!, :weights]
        end
    else
        # @assert(rm ∈ HRRiskMeasures, "rm must be one of $HRRiskMeasures")
        portfolio.p_optimal[!, :weights]
    end

    return calc_risk(
        weights,
        portfolio.returns,
        portfolio.cov;
        rm = rm,
        rf = rf,
        alpha_i = portfolio.alpha_i,
        alpha = portfolio.alpha,
        a_sim = portfolio.a_sim,
        beta_i = portfolio.beta_i,
        beta = portfolio.beta,
        b_sim = portfolio.b_sim,
        kappa = portfolio.kappa,
        solvers = portfolio.solvers,
    )
end

export calc_risk