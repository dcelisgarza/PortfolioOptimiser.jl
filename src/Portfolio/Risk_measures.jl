"""
```julia
Variance(w::AbstractVector, Σ::AbstractMatrix)
```
Compute the Variance of a portfolio with weights `w` and covariance `Σ`. Square of [`SD`](@ref).

```math
\\mathrm{Variance}(\\bm{w},\\, \\mathbf{\\Sigma}) = \\bm{w}^\\intercal \\, \\mathbf{\\Sigma}\\, \\bm{w}\\,.
```
"""
function Variance(w::AbstractVector, cov::AbstractMatrix)
    return dot(w, cov, w)
end

"""
```julia
SD(w::AbstractVector, Σ::AbstractMatrix)
```
Compute the Standard Deviation of a portfolio with weights `w` and covariance `Σ`. Square root of [`Variance`](@ref).

```math
\\mathrm{SD}(\\bm{w},\\, \\mathbf{\\Sigma}) = \\left[\\bm{w}^\\intercal \\, \\mathbf{\\Sigma} \\, \\bm{w}\\right]^{1/2}\\,.
```
"""
function SD(w::AbstractVector, cov::AbstractMatrix)
    return sqrt(Variance(w, cov))
end

"""
```julia
MAD(x::AbstractVector)
```
Compute the Mean Absolute Deviation of a vector `x` of portfolio returns.

```math
\\mathrm{MAD}(\\bm{x}) = \\dfrac{1}{T} \\sum\\limits_{t=1}^T \\left\\lvert x_t - \\mathbb{E}(\\bm{x}) \\right\\rvert\\,.
```
"""
function MAD(x::AbstractVector)
    mu = mean(x)
    return mean(abs.(x .- mu))
end
"""
```julia
Semi_SD(x::AbstractVector)
```
Compute the mean Semi-Standard Deviation of a vector `x` of portfolio returns.

```math
\\mathrm{SemiSD}(\\bm{x}) = \\left[\\dfrac{1}{T-1} \\sum\\limits_{t=1}^{T}\\min\\left( \\bm{x}_{t} - \\mathbb{E}(\\bm{x}),\\, 0\\right)^{2}\\right]^{1/2}\\,.
```
"""
function Semi_SD(x::AbstractVector)
    T = length(x)
    mu = mean(x)
    val = mu .- x
    return sqrt(sum(val[val .>= 0] .^ 2) / (T - 1))
end

"""
```julia
FLPM(x::AbstractVector, r::Real = 0.0)
```
Compute the First Lower Partial Moment (omega ratio) of a vector `x` of portfolio returns with a minimum return target of `r`.

```math
\\mathrm{FLPM}(\\bm{x},\\, r) = \\dfrac{1}{T}  \\sum\\limits_{t=1}^{T}\\max\\left(r - \\bm{x}_{t},\\, 0\\right)\\,.
```
"""
function FLPM(x::AbstractVector, min_ret::Real = 0.0)
    T = length(x)
    val = min_ret .- x
    return sum(val[val .>= 0]) / T
end

"""
```julia
SLPM(x::AbstractVector, r::Real = 0.0)
```
Compute the Second Lower Partial Moment (Omega Ratio) of a vector `x` of portfolio returns with a minimum return target of `r`.`

```math
\\mathrm{SLPM}(\\bm{x},\\, r) = \\left[\\dfrac{1}{T-1} \\sum\\limits_{t=1}^{T}\\max\\left(r - \\bm{x}_{t},\\, 0\\right)^{2}\\right]^{1/2}\\,
```
"""
function SLPM(x, min_ret = 0.0)
    T = length(x)
    val = min_ret .- x
    val = sum(val[val .>= 0] .^ 2) / (T - 1)
    return sqrt(val)
end

"""
```julia
WR(x::AbstractVector)
```
Compute the Worst Realisation or Worst Case Scenario of a returns vector `x`.

```math
\\begin{align*}
\\mathrm{WR}(\\bm{x}) &= -\\min(\\bm{x})\\\\
                      &= \\max(-\\bm{x})\\,.
\\end{align*}
```
"""
function WR(x::AbstractVector)
    return -minimum(x)
end

"""
```julia
VaR(x::AbstractVector, α::Real = 0.05)
```
Compute the Value at Risk of a returns vector `x` at a significance level of `α`.

```math
\\mathrm{VaR}(\\bm{x},\\, \\alpha) = -\\inf_{t \\in (0,\\, T)} \\left\\{ x_{t} \\in \\mathbb{R} : F_{\\bm{x}}(x_{t}) > \\alpha \\right\\}\\,,
```
"""
function VaR(x::AbstractVector, alpha::Real = 0.05)
    sort!(x)
    idx = ceil(Int, alpha * length(x))
    return -x[idx]
end

"""
```
CVaR(x::AbstractVector, α::Real = 0.05)
```
Compute the Conditional Value at Risk of a returns vector `x` at a significance level of `α`.

```math
\\begin{align*}
\\mathrm{CVaR}(\\bm{x},\\, \\alpha) &= \\mathrm{VaR}(\\bm{x},\\, \\alpha) - \\dfrac{1}{\\alpha T} \\sum\\limits_{t=1}^{T} \\min\\left( x_t + \\mathrm{VaR}(\\bm{x},\\, \\alpha),\\, 0\\right)\\\\
                                    &= \\mathrm{VaR}(\\bm{x},\\, \\alpha) + \\dfrac{1}{\\alpha T} \\sum\\limits_{t=1}^{T} \\max\\left( -x_t - \\mathrm{VaR}(\\bm{x},\\, \\alpha),\\, 0\\right)\\,.
\\end{align*}
```
"""
function CVaR(x::AbstractVector, alpha::Real = 0.05)
    sort!(x)
    idx = ceil(Int, alpha * length(x))
    var = -x[idx]
    sum_var = 0.0
    for i in 1:(idx - 1)
        sum_var += x[i] + var
    end
    return var - sum_var / (alpha * length(x))
end

function _optimize_rm(model, solvers::AbstractDict)
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

"""
```julia
ERM(x::AbstractVector, z::Real = 1.0, α::Real = 0.05)
```
Compute the Entropic Risk Measure of a vector `x` at a significance level of `α` for a given value of `z`.

```julia
ERM(x::AbstractVector, solvers::AbstractDict, α::Real = 0.05)
```
Compute the Entropic Risk Measure at a significance level of `α` by minimising with respect to `z`, using a dictionary of `JuMP`-supported `solvers`. This is because in general we don't know the value of `z` that minimises the function.

```math
\\mathrm{ERM}(\\bm{x},\\, z, \\,\\alpha) = z \\ln \\left( \\dfrac{M_{\\bm{x}}\\left(z^{-1}\\right)}{\\alpha} \\right)\\,,
```
where ``M_{\\bm{x}}\\left(z^{-1}\\right)`` is the moment generating function of ``\\bm{x}``.
"""
function ERM(x::AbstractVector, z::Real = 1.0, alpha::Real = 0.05)
    val = mean(exp.(-x / z))
    val = z * log(val / alpha)
    return val
end
function ERM(x::AbstractVector, solvers::AbstractDict, alpha::Real = 0.05)
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
        funcname = "$(fullname(PortfolioOptimiser)[1]).$(nameof(PortfolioOptimiser._ERM))"
        @warn(
            "$funcname: model could not be optimised satisfactorily. Solvers: $solvers_tried"
        )
    end

    return obj_val
end

"""
```julia
EVaR(x::AbstractVector, solvers::AbstractDict, α::Real = 0.05)
```
Compute the Entropic Value at Risk of a returns vector `x` at a significance level of `α`.

```math
\\mathrm{EVaR}(\\bm{x},\\alpha) = \\inf_{z > 0} \\left\\{\\mathrm{ERM}(\\bm{x},\\, z, \\,\\alpha)\\right\\}\\,.
```
"""
function EVaR(x::AbstractVector, solvers::AbstractDict, alpha::Real = 0.05)
    return ERM(x, solvers, alpha)
end

function RRM(x, solvers, alpha = 0.05, kappa = 0.3)
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
        funcname = "$(fullname(PortfolioOptimiser)[1]).$(nameof(PortfolioOptimiser._optimizeRRM))"
        @warn(
            "$funcname: model could not be optimised satisfactorily. Solvers: $solvers_tried"
        )
    end

    return obj_val
end

function RVaR(x, solvers, alpha = 0.05, kappa = 0.3)
    return RRM(x, solvers, alpha, kappa)
end

function MDD_abs(x)
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

function ADD_abs(x)
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

function DaR_abs(x, alpha)
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

function CDaR_abs(x, alpha)
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

function UCI_abs(x)
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

function EDaR_abs(x, solvers, alpha = 0.05)
    insert!(x, 1, 1)
    cs = cumsum(x)
    peak = -Inf
    dd = similar(cs)
    for (idx, i) in enumerate(cs)
        i > peak && (peak = i)
        dd[idx] = -(peak - i)
    end
    deleteat!(dd, 1)
    return ERM(dd, solvers, alpha)
end

function RDaR_abs(x, solvers, alpha = 0.05, kappa = 0.3)
    insert!(x, 1, 1)
    cs = cumsum(x)
    peak = -Inf
    dd = similar(cs)
    for (idx, i) in enumerate(cs)
        i > peak && (peak = i)
        dd[idx] = i - peak
    end
    deleteat!(dd, 1)
    return RRM(dd, solvers, alpha, kappa)
end

function MDD_rel(x)
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

function ADD_rel(x)
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

function DaR_rel(x, alpha)
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

function CDaR_rel(x, alpha)
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

function UCI_rel(x)
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

function EDaR_rel(x, solvers, alpha)
    x .= insert!(x, 1, 0) .+ 1
    cs = cumprod(x)
    peak = -Inf
    dd = similar(cs)
    for (idx, i) in enumerate(cs)
        i > peak && (peak = i)
        dd[idx] = i / peak - 1
    end
    deleteat!(dd, 1)
    return ERM(dd, solvers, alpha)
end

function RDaR_rel(x, solvers, alpha = 0.05, kappa = 0.3)
    x .= insert!(x, 1, 0) .+ 1
    cs = cumprod(x)
    peak = -Inf
    dd = similar(cs)
    for (idx, i) in enumerate(cs)
        i > peak && (peak = i)
        dd[idx] = i / peak - 1
    end
    deleteat!(dd, 1)
    return RRM(dd, solvers, alpha, kappa)
end

function Kurt(x)
    T = length(x)
    mu = mean(x)
    val = x .- mu
    return sqrt(sum(val .^ 4) / T)
end

function Semi_Kurt(x)
    T = length(x)
    mu = mean(x)
    val = x .- mu
    return sqrt(sum(val[val .< 0] .^ 4) / T)
end

function GMD(x)
    T = length(x)
    w = owa_gmd(T)
    return dot(w, sort!(x))
end

function RG(x)
    T = length(x)
    w = owa_rg(T)
    return dot(w, sort!(x))
end

function RCVaR(x; alpha = 0.05, beta = nothing)
    T = length(x)
    w = owa_rcvar(T; alpha = alpha, beta = beta)
    return dot(w, sort!(x))
end

function TG(x; alpha_i = 0.0001, alpha = 0.05, a_sim = 100)
    T = length(x)
    w = owa_tg(T; alpha_i = alpha_i, alpha = alpha, a_sim = a_sim)
    return dot(w, sort!(x))
end

function RTG(
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

function OWA(x, w)
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
        SD(w, cov)
    elseif rm == :mv
        Variance(w, cov)
    elseif rm == :mad
        MAD(x)
    elseif rm == :msv
        Semi_SD(x)
    elseif rm == :flpm
        FLPM(x, rf)
    elseif rm == :slpm
        SLPM(x, rf)
    elseif rm == :wr
        WR(x)
    elseif rm == :var
        VaR(x, alpha)
    elseif rm == :cvar
        CVaR(x, alpha)
    elseif rm == :evar
        EVaR(x, solvers, alpha)
    elseif rm == :rvar
        RVaR(x, solvers, alpha, kappa)
    elseif rm == :mdd
        MDD_abs(x)
    elseif rm == :add
        ADD_abs(x)
    elseif rm == :dar
        DaR_abs(x, alpha)
    elseif rm == :cdar
        CDaR_abs(x, alpha)
    elseif rm == :uci
        UCI_abs(x)
    elseif rm == :edar
        EDaR_abs(x, solvers, alpha)
    elseif rm == :rdar
        RDaR_abs(x, solvers, alpha, kappa)
    elseif rm == :mdd_r
        MDD_rel(x)
    elseif rm == :add_r
        ADD_rel(x)
    elseif rm == :dar_r
        DaR_rel(x, alpha)
    elseif rm == :cdar_r
        CDaR_rel(x, alpha)
    elseif rm == :uci_r
        UCI_rel(x)
    elseif rm == :edar_r
        EDaR_rel(x, solvers, alpha)
    elseif rm == :rdar_r
        RDaR_rel(x, solvers, alpha, kappa)
    elseif rm == :krt
        Kurt(x)
    elseif rm == :skrt
        Semi_Kurt(x)
    elseif rm == :gmd
        GMD(x)
    elseif rm == :rg
        RG(x)
    elseif rm == :rcvar
        RCVaR(x; alpha = alpha, beta = beta)
    elseif rm == :tg
        TG(x; alpha_i = alpha_i, alpha = alpha, a_sim = a_sim)
    elseif rm == :rtg
        RTG(
            x;
            alpha_i = alpha_i,
            alpha = alpha,
            a_sim = a_sim,
            beta_i = beta_i,
            beta = beta,
            b_sim = b_sim,
        )
    elseif rm == :owa
        OWA(x, w)
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

export calc_risk, Variance, SD, MAD, Semi_SD, FLPM, SLPM, WR, VaR, CVaR, ERM, EVaR