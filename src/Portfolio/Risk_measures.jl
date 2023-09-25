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
SSD(x::AbstractVector)
```
Compute the mean Semi-Standard Deviation of a vector `x` of portfolio returns.
```math
\\mathrm{SSD}(\\bm{x}) = \\left[\\dfrac{1}{T-1} \\sum\\limits_{t=1}^{T}\\min\\left(\\bm{x}_{t} - \\mathbb{E}(\\bm{x}),\\, 0\\right)^{2}\\right]^{1/2}\\,.
```
"""
function SSD(x::AbstractVector)
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
Compute the Value at Risk of a returns vector `x` at a significance level of `α ∈ (0, 1)`.
```math
\\mathrm{VaR}(\\bm{x},\\, \\alpha) = -\\underset{t \\in (0,\\, T)}{\\inf} \\left\\{ x_{t} \\in \\mathbb{R} : F_{\\bm{x}}(x_{t}) > \\alpha \\right\\}\\,,
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
Compute the Conditional Value at Risk of a returns vector `x` at a significance level of `α ∈ (0, 1)`.
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
Compute the Entropic Risk Measure of a vector `x` at a significance level of `α ∈ (0, 1)` for a given value of `z`,
```math
\\mathrm{ERM}(\\bm{x},\\, z, \\,\\alpha) = z \\ln \\left( \\dfrac{M_{\\bm{x}}\\left(z^{-1}\\right)}{\\alpha} \\right)\\,,
```
where ``M_{\\bm{x}}\\left(z^{-1}\\right)`` is the moment generating function of ``\\bm{x}``.
```julia
ERM(x::AbstractVector, solvers::AbstractDict, α::Real = 0.05)
```
Compute the Entropic Risk Measure at a significance level of `α ∈ (0, 1)` by minimising with respect to `z`, using a dictionary of `JuMP`-supported `solvers` with Exponential Cone support. This is because in general we don't know the value of `z` that minimises the function,
```math
\\mathrm{ERM} = \\begin{cases}
\\underset{z,\\, t,\\, u}{\\min} & t + z \\ln(\\dfrac{1}{\\alpha T})\\\\
\\mathrm{s.t.} & z \\geq \\sum\\limits_{i=1}^{T} u_{i}\\\\
& (x_{i}-t,\\, z,\\, u_{i}) \\in \\mathcal{K}_{\\exp} \\, \\forall i=1,\\,\\dots{},\\, T
\\end{cases}\\,,
```
where ``\\mathcal{K}_{\exp}`` is the exponential cone.
"""
function ERM(x::AbstractVector, z::Real = 1.0, alpha::Real = 0.05)
    @assert(0 < alpha < 1, "alpha must be greater than 0 and smaller than 1")
    val = mean(exp.(-x / z))
    val = z * log(val / alpha)
    return val
end
function ERM(x::AbstractVector, solvers::AbstractDict, alpha::Real = 0.05)
    @assert(0 < alpha < 1, "alpha must be greater than 0 and smaller than 1")

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
Compute the Entropic Value at Risk of a returns vector `x` at a significance level of `α ∈ (0, 1)`, using a dictionary of `JuMP`-supported `solvers` with Exponential Cone support.
```math
\\mathrm{EVaR}(\\bm{x},\\alpha) = \\underset{z > 0}{\\inf} \\left\\{\\mathrm{ERM}(\\bm{x},\\, z, \\,\\alpha)\\right\\}\\,,
```
where ``\\mathrm{ERM}(\\bm{x},\\, z, \\,\\alpha)`` is the entropic risk measure at a significance level of `α ∈ (0, 1)` as defined by [`ERM`](@ref).
"""
function EVaR(x::AbstractVector, solvers::AbstractDict, alpha::Real = 0.05)
    return ERM(x, solvers, alpha)
end

"""
```julia
RRM(
    x::AbstractVector,
    solvers::AbstractDict,
    α::Real = 0.05,
    κ::Real = 0.3,
)
```
Compute the Relativistic Risk Measure of a vector `x` at a significance level of `α ∈ (0, 1)` and deformation parameter of `κ ∈ (0, 1)`, using a dictionary of `JuMP`-supported `solvers` with 3D Power Cone support.
```math
\\mathrm{RRM}(\\bm{x},\\, \\alpha,\\, \\kappa) = \\begin{cases}
\\underset{z,\\, t,\\, \\psi,\\, \\theta,\\, \\varepsilon,\\, \\omega}{\\min} & t + z \\ln_{\\kappa}\\left(\\dfrac{1}{\\alpha T}\\right) + \\sum\\limits_{i=1}^{T} \\left(\\psi_{i} + \\theta_{i}\\right) \\\\
\\mathrm{s.t.} & z \\geq 0 \\\\
& -x_{i} - t + \\varepsilon_{i} + \\omega_{i} \\leq 0 \\\\
& \\left(z\\left(\\dfrac{1+\\kappa}{2\\kappa}\\right),\\, \\psi_{i}\\left(\\dfrac{1+\\kappa}{\\kappa}\\right),\\, \\varepsilon_{i} \\right) \\in \\mathcal{P}_{3}^{1/(1+\\kappa),\\, \\kappa/(1+\\kappa)} \\\\
& \\left(\\omega_{i}\\left(\\dfrac{1}{1-\\kappa}\\right),\\, \\theta_{i}\\left(\\dfrac{1}{\\kappa}\\right),\\, -z\\left(\\dfrac{1}{2\\kappa}\\right) \\right) \\in \\mathcal{P}_{3}^{1-\\kappa,\\, \\kappa} \\\\
& \\forall i = 1,\\,\\ldots{},\\, T
\\end{cases}\\,,
```
where ``\\ln_{\\kappa}(x) = \\dfrac{x^{\\kappa} - x^{-\\kappa}}{2 \\kappa}`` and ``\\mathcal{P}_3^{\\alpha,\\, 1-\\alpha}`` is the 3D Power Cone.
"""
function RRM(
    x::AbstractVector,
    solvers::AbstractDict,
    alpha::Real = 0.05,
    kappa::Real = 0.3,
)
    @assert(0 < alpha < 1, "alpha must be greater than 0 and smaller than 1")
    @assert(0 < kappa < 1, "kappa must be greater than 0 and smaller than 1")

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
"""
```julia
RVaR(
    x::AbstractVector,
    solvers::AbstractDict,
    α::Real = 0.05,
    κ::Real = 0.3,
)
```
Compute the Relativistic Value at Risk of a vector of returns `x` at a significance level of `α ∈ (0, 1)` and deformation parameter of `κ ∈ (0, 1)`, using a dictionary of `JuMP`-supported `solvers` with 3D Power Cone support.
```math
\\mathrm{RVaR}(\\bm{x},\\, \\alpha,\\, \\kappa) = \\mathrm{RRM}(\\bm{x},\\, \\alpha,\\, \\kappa)\\,,
```
where ``\\mathrm{RRM}(\\bm{x},\\, \\alpha,\\, \\kappa)`` is the Relativistic Risk Measure as defined in [`RRM`](@ref).
"""
function RVaR(
    x::AbstractVector,
    solvers::AbstractDict,
    alpha::Real = 0.05,
    kappa::Real = 0.3,
)
    return RRM(x, solvers, alpha, kappa)
end

"""
```julia
DaR_abs(x::AbstractArray, alpha::Real = 0.05)
```
Compute the Drawdown at Risk forof a returns vector `x` for a significance level `α ∈ (0, 1)` using uncompounded cumulative returns.
```math
\\begin{align*}
\\mathrm{DaR_{a}}(\\bm{x},\\, \\alpha) &= \\underset{j \\in (0,\\, T)}{\\max} \\left\\{ \\mathrm{DD_{a}}(\\bm{x},\\, j) \\in \\mathbb{R} : F_{\\mathrm{DD}}\\left(\\mathrm{DD_{a}}(\\bm{x},\\, j)\\right) < 1 - \\alpha \\right\\}\\\\
\\mathrm{DD_{a}}(\\bm{x},\\, j) &= \\underset{t \\in (0,\\, j)}{\\max}\\left( \\sum\\limits_{i=0}^{t} x_{i} \\right) - \\sum\\limits_{i=0}^{j} x_{i}
\\end{align*}\\,.
```
"""
function DaR_abs(x::AbstractArray, alpha::Real = 0.05)
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

"""
```julia
MDD_abs(x::AbstractVector)
```
Compute the Maximum Drawdown of a returns vector `x` using uncompounded cumulative returns.
```math
\\mathrm{MDD_{a}}(\\bm{x}) = \\underset{j \\in (0,\\, T)}{\\max} \\mathrm{DD_{a}}(\\bm{x},\\, j)\\,,
```
where ``\\mathrm{DD_{a}}(\\bm{x},\\, j)`` is the uncompounded cumulative Drawdown as defined in [`DaR_abs`](@ref).
"""
function MDD_abs(x::AbstractVector)
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

"""
```julia
ADD_abs(x::AbstractVector)
```
Compute the Average Drawdown of a returns vector `x` using uncompounded cumulative returns.
```math
\\mathrm{ADD_{a}}(\\bm{x}) = \\dfrac{1}{T} \\sum\\limits_{j=0}^{T} \\mathrm{DD_{a}}(\\bm{x},\\, j)\\,,
```
where ``\\mathrm{DD_{a}}(\\bm{x},\\, j)`` is the uncompounded cumulative Drawdown as defined in [`DaR_abs`](@ref).
"""
function ADD_abs(x::AbstractVector)
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

"""
```julia
CDaR_abs(x::AbstractVector, alpha::Real = 0.05)
```
Compute the Conditional Drawdown at Risk of a returns vector `x` using uncompounded cumulative returns at a significance level of `α ∈ (0, 1)`.
```math
\\mathrm{CDaR_{a}}(\\bm{x},\\, \\alpha) = \\mathrm{DaR_{a}}(\\bm{x},\\, \\alpha) + \\dfrac{1}{\\alpha T} \\sum\\limits_{j=0}^{T} \\max\\left[\\mathrm{DD_{a}}(\\bm{x},\\, j) - \\mathrm{DaR_{a}}(\\bm{x},\\, \\alpha),\\, 0 \\right] \\,,
```
where ``\\mathrm{DD_{a}}(\\bm{x},\\, j)`` is the uncompounded cumulative Drawdown as defined in [`DaR_abs`](@ref), and ``\\mathrm{DaR_{a}}(\\bm{x},\\, \\alpha)`` the Drawdown at Risk of uncompounded cumulative returns as defined in [`DaR_abs`](@ref).
"""
function CDaR_abs(x::AbstractVector, alpha::Real = 0.05)
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

"""
```julia
UCI_abs(x::AbstractVector)
```
Compute the Ulcer Index of a returns vector `x` using uncompounded cumulative returns.
```math
\\mathrm{UCI_{a}}(\\bm{x}) = \\left[\\dfrac{1}{T} \\sum\\limits_{j=0}^{T} \\mathrm{DD_{a}}(\\bm{x},\\, j)^{2}\\right]^{1/2}\\,,
```
where ``\\mathrm{DD_{a}}(\\bm{x},\\, j)`` is the uncompounded cumulative Drawdown as defined in [`DaR_abs`](@ref).
"""
function UCI_abs(x::AbstractVector)
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

"""
```julia
EDaR_abs(x::AbstractVector, solvers::AbstractDict, α::Real = 0.05)
```
Compute the Entropic Drawdown at Risk of uncompounded cummulative returns of a returns vector `x` at a significance level of `α ∈ (0, 1)`, using a dictionary of `JuMP`-supported `solvers` with Exponential Cone support.
```math
\\begin{align*}
\\mathrm{EDaR_{a}}(\\bm{x},\\alpha) &= \\underset{z > 0}{\\inf} \\left\\{\\mathrm{ERM}(\\mathrm{DD_{a}}(\\bm{x}),\\, z, \\,\\alpha)\\right\\}\\\\
\\mathrm{DD_{a}}(\\bm{x}) &= \\left\\{j \\in (0,\\, T) : \\mathrm{DD_{a}}(\\bm{x},\\, j) \\right\\}\\,,
\\end{align*}
```
where ``\\mathrm{ERM}(\\bm{x},\\, z, \\,\\alpha)`` is the entropic risk measure at a significance level of `α ∈ (0, 1)` as defined by [`ERM`](@ref) and ``\\mathrm{DD_{a}}(\\bm{x},\\, j)`` the drawdown of uncompounded returns as defined in [`DaR_abs`](@ref).
"""
function EDaR_abs(x::AbstractVector, solvers::AbstractDict, alpha::Real = 0.05)
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

"""
```julia
RDaR_abs(
    x::AbstractVector,
    solvers::AbstractDict,
    alpha::Real = 0.05,
    kappa::Real = 0.3,
)
```
Compute the Relativistic Drawdown at Risk of uncompounded cumulative returns of a vector of returns `x` at a significance level of `α ∈ (0, 1)` and deformation parameter of `κ ∈ (0, 1)`, using a dictionary of `JuMP`-supported `solvers` with 3D Power Cone support.
```math
\\mathrm{RDaR_{a}}(\\bm{x},\\, \\alpha,\\, \\kappa) = \\mathrm{RRM}(\\mathrm{DD_{a}}(\\bm{x}),\\, \\alpha,\\, \\kappa)\\,,
```
where ``\\mathrm{RRM}(\\mathrm{DD_{a}}(\\bm{x}),\\, \\alpha,\\, \\kappa)`` is the Relativistic Risk Measure as defined in [`RRM`](@ref) where the returns vector, ``\\mathrm{DD_{a}}(\\bm{x})``, is the set of drawdowns of uncompounded returns for the timespan ``(0,\\, T)`` as defined in [`EDaR_abs`](@ref).
"""
function RDaR_abs(
    x::AbstractVector,
    solvers::AbstractDict,
    alpha::Real = 0.05,
    kappa::Real = 0.3,
)
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

"""
```julia
DaR_ret(x::AbstractArray, alpha::Real = 0.05)
```
Compute the Drawdown at Risk of a returns vector `x` for a significance level `α ∈ (0, 1)` using compounded cumulative returns.
```math
\\begin{align*}
\\mathrm{DaR_{r}}(\\bm{x},\\, \\alpha) &= \\underset{j \\in (0,\\, T)}{\\max} \\left\\{ \\mathrm{DD_{r}}(\\bm{x},\\, j) \\in \\mathbb{R} : F_{\\mathrm{DD}}\\left(\\mathrm{DD_{r}}(\\bm{x},\\, j)\\right) < 1 - \\alpha \\right\\}\\\\
\\mathrm{DD_{r}}(\\bm{x},\\, j) &= \\underset{t \\in (0,\\, j)}{\\max}\\left( \\prod\\limits_{i=0}^{t} \\left(1+x_{i}\\right) \\right) - \\prod\\limits_{i=0}^{j} \\left(1+x_{i}\\right) 
\\end{align*}\\,.
```
"""
function DaR_rel(x::AbstractArray, alpha::Real = 0.05)
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

"""
```julia
MDD_rel(x::AbstractVector)
```
Compute the Maximum Drawdown of a returns vector `x` using compounded cumulative returns.
```math
\\mathrm{MDD_{r}}(\\bm{x}) = \\underset{j \\in (0,\\, T)}{\\max} \\mathrm{DD_{r}}(\\bm{x},\\, j)\\,,
```
where ``\\mathrm{DD_{r}}(\\bm{x},\\, j)`` is the compounded cumulative Drawdown as defined in [`DaR_rel`](@ref).
"""
function MDD_rel(x::AbstractVector)
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

"""
```julia
ADD_rel(x::AbstractVector)
```
Compute the Average Drawdown of a returns vector `x` using compounded cumulative returns.
```math
\\mathrm{ADD_{r}}(\\bm{r}) = \\dfrac{1}{T} \\sum\\limits_{j=0}^{T} \\mathrm{DD_{r}}(\\bm{x},\\, j)\\,,
```
where ``\\mathrm{DD_{r}}(\\bm{x},\\, j)`` is the compounded cumulative Drawdown as defined in [`DaR_rel`](@ref).
"""
function ADD_rel(x::AbstractVector)
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

"""
```julia
CDaR_rel(x::AbstractVector, alpha::Real = 0.05)
```
Compute the Conditional Drawdown at Risk of a returns vector `x` using compounded cumulative returns at a significance level of `α ∈ (0, 1)`.
```math
\\mathrm{CDaR_{r}}(\\bm{x},\\, \\alpha) = \\mathrm{DaR_{r}}(\\bm{x},\\, \\alpha) + \\dfrac{1}{\\alpha T} \\sum\\limits_{j=0}^{T} \\max\\left[\\mathrm{DD_{r}}(\\bm{x},\\, j) - \\mathrm{DaR_{r}}(\\bm{x},\\, \\alpha),\\, 0 \\right] \\,,
```
where ``\\mathrm{DD_{r}}(\\bm{x},\\, j)`` is the compounded cumulative Drawdown as defined in [`DaR_rel`](@ref), and ``\\mathrm{DaR_{r}}(\\bm{x},\\, \\alpha)`` the Drawdown at Risk of compounded cumulative returns as defined in [`DaR_rel`](@ref).
"""
function CDaR_rel(x::AbstractVector, alpha::Real = 0.05)
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

"""
```julia
UCI_rel(x::AbstractVector)
```
Compute the Ulcer Index of a returns vector `x` using compounded cumulative returns.
```math
\\mathrm{UCI_{r}}(\\bm{x}) = \\left[\\dfrac{1}{T} \\sum\\limits_{j=0}^{T} \\mathrm{DD_{r}}(\\bm{x},\\, j)^{2}\\right]^{1/2}\\,,
```
where ``\\mathrm{DD_{r}}(\\bm{x},\\, j)`` is the compounded cumulative Drawdown as defined in [`DaR_rel`](@ref).
"""
function UCI_rel(x::AbstractVector)
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

"""
```julia
EDaR_rel(x::AbstractVector, solvers::AbstractDict, α::Real = 0.05)
```
Compute the Entropic Drawdown at Risk of compounded cummulative returns of a returns vector `x` at a significance level of `α ∈ (0, 1)`, using a dictionary of `JuMP`-supported `solvers` with Exponential Cone support.
```math
\\begin{align*}
\\mathrm{EDaR_{r}}(\\bm{x},\\alpha) &= \\underset{z > 0}{\\inf} \\left\\{\\mathrm{ERM}(\\mathrm{DD_{r}}(\\bm{x}),\\, z, \\,\\alpha)\\right\\}\\\\
\\mathrm{DD_{r}}(\\bm{x}) &= \\left\\{j \\in (0,\\, T) : \\mathrm{DD_{r}}(\\bm{x},\\, j) \\right\\}\\,,
\\end{align*}
```
where ``\\mathrm{ERM}(\\bm{x},\\, z, \\,\\alpha)`` is the entropic risk measure at a significance level of `α ∈ (0, 1)` as defined by [`ERM`](@ref) and ``\\mathrm{DD_{r}}(\\bm{x},\\, j)`` the drawdown of compounded returns as defined in [`DaR_rel`](@ref).
"""
function EDaR_rel(x::AbstractVector, solvers::AbstractDict, alpha::Real = 0.05)
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

"""
```julia
RDaR_rel(
    x::AbstractVector,
    solvers::AbstractDict,
    alpha::Real = 0.05,
    kappa::Real = 0.3,
)
```
Compute the Relativistic Drawdown at Risk of compounded cumulative returns of a vector of returns `x` at a significance level of `α ∈ (0, 1)` and deformation parameter of `κ ∈ (0, 1)`, using a dictionary of `JuMP`-supported `solvers` with 3D Power Cone support.
```math
\\mathrm{RDaR_{r}}(\\bm{x},\\, \\alpha,\\, \\kappa) = \\mathrm{RRM}(\\mathrm{DD_{r}}(\\bm{x}),\\, \\alpha,\\, \\kappa)\\,,
```
where ``\\mathrm{RRM}(\\mathrm{DD_{r}}(\\bm{x}),\\, \\alpha,\\, \\kappa)`` is the Relativistic Risk Measure as defined in [`RRM`](@ref) where the returns vector, ``\\mathrm{DD_{r}}(\\bm{x})``, is the set of drawdowns of compounded returns for the timespan ``(0,\\, T)`` as defined in [`EDaR_rel`](@ref).
"""
function RDaR_rel(
    x::AbstractVector,
    solvers::AbstractDict,
    alpha::Real = 0.05,
    kappa::Real = 0.3,
)
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

"""
```julia
Kurt(x::AbstractVector)
```
Compute the square root kurtosis of a returns vector `x`.
```math
\\mathrm{Kurt} = \\left[\\dfrac{1}{T} \\sum\\limits_{t=1}^{T} \\left( x_{t} - \\mathbb{E}(\\bm{x}) \\right)^{4} \\right]^{1/2}\\,.
```
"""
function Kurt(x::AbstractVector)
    T = length(x)
    mu = mean(x)
    val = x .- mu
    return sqrt(sum(val .^ 4) / T)
end

"""
```julia
SKurt(x::AbstractVector)
```
Compute the square root semi-kurtosis of a returns vector `x`.
```math
\\mathrm{SKurt} = \\left[\\dfrac{1}{T} \\sum\\limits_{t=1}^{T} \\min\\left( x_{t} - \\mathbb{E}(\\bm{x}),\\, 0 \\right)^{4} \\right]^{1/2}\\,.
```
"""
function SKurt(x::AbstractVector)
    T = length(x)
    mu = mean(x)
    val = x .- mu
    return sqrt(sum(val[val .< 0] .^ 4) / T)
end

"""
```julia
GMD(x::AbstractVector)
```
Compute the Gini Mean Difference of a returns vector `x`.
"""
function GMD(x::AbstractVector)
    T = length(x)
    w = owa_gmd(T)
    return dot(w, sort!(x))
end

"""
```julia
RG(x::AbstractVector)
```
Compute the Range of a returns vector `x`.
"""
function RG(x::AbstractVector)
    T = length(x)
    w = owa_rg(T)
    return dot(w, sort!(x))
end

"""
```julia
RCVaR(
    x::AbstractVector;
    α::Real = 0.05,
    β::Union{<:Real, Nothing} = nothing,
)
```
Compute the CVaR Range of a returns vector `x` for a significance level of CVaR losses `α ∈ (0, 1)` and CVaR gains `β ∈ (0, 1)`. If `β` is `nothing`, it takes on the value of `α`.
"""
function RCVaR(
    x::AbstractVector;
    alpha::Real = 0.05,
    beta::Union{<:Real, Nothing} = nothing,
)
    T = length(x)
    w = owa_rcvar(T; alpha = alpha, beta = beta)
    return dot(w, sort!(x))
end

"""
```julia
TG(x::AbstractVector; α_i::Real = 0.0001, α::Real = 0.05, α_sim::Int = 100)
```
Compute the Tail Gini of a returns vector `x` for CVaR losses starting at `α_i` through to `α`, using `α_sim` steps.
"""
function TG(x::AbstractVector; alpha_i::Real = 0.0001, alpha::Real = 0.05, a_sim::Int = 100)
    @assert(alpha_i < alpha, "alpha_i must be less than alpha")
    T = length(x)
    w = owa_tg(T; alpha_i = alpha_i, alpha = alpha, a_sim = a_sim)
    return dot(w, sort!(x))
end

"""
```julia
RTG(
    x::AbstractVector;
    α_i::Real = 0.0001,
    α::Real = 0.05,
    α_sim::Real = 100,
    β_i::Union{<:Real, Nothing} = nothing,
    β::Union{<:Real, Nothing} = nothing,
    β_sim::Union{Int, Nothing} = nothing,
)
```
Compute the Tail Gini Range of a returns vector `x` for CVaR losses starting at `α_i` through to `α`, using `α_sim` steps, and for CVaR gains starting at `β_i` through to `β`, using `β_sim` steps. If any argument relating to CVaR gains is `nothing`, it will take the value of its corresponding CVaR loss argument.
"""
function RTG(
    x::AbstractVector;
    alpha_i::Real = 0.0001,
    alpha::Real = 0.05,
    a_sim::Real = 100,
    beta_i::Union{<:Real, Nothing} = nothing,
    beta::Union{<:Real, Nothing} = nothing,
    b_sim::Union{Int, Nothing} = nothing,
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

"""
```julia
OWA(x::AbstractVector, w::AbstractVector)
```
Compute the Ordered Weight Array risk measure from a returns vector `x` and am Ordered Weights Array vector `w`.
"""
function OWA(x::AbstractVector, w::AbstractVector)
    return dot(w, sort!(x))
end

"""
```julia
calc_risk(
    w::AbstractVector,
    returns::AbstractMatrix,
    cov::AbstractMatrix;
    rm::Symbol = :SD,
    rf::Real = 0.0,
    α_i::Real = 0.0001,
    α::Real = 0.05,
    α_sim::Int = 100,
    β_i::Union{<:Real, Nothing} = nothing,
    β::Union{<:Real, Nothing} = nothing,
    β_sim::Union{<:Real, Nothing} = nothing,
    κ::Real = 0.3,
    solvers::Union{<:AbstractDict, Nothing} = nothing,
)
```
Compute the risk for a given weights vector `w`, returns matrix `returns`, covariance matrix `cov`, risk measure `rm`, risk free rate `rf`, CVaR losses starting at `α_i` through to `α`, using `α_sim` steps, and for CVaR gains starting at `β_i` through to `β`, using `β_sim` steps. If any argument relating to CVaR gains is `nothing`, it will take the value of its corresponding CVaR loss argument. Relativistic risk measure deformation parameter `κ` and dictionary of `solvers` with Exponential Cone and/or 3D Power Cone support in case of using an entropic or relativistic risk measure. Some arguments are only used for specific risk measures.
```julia
calc_risk(
    portfolio::AbstractPortfolio;
    type::Symbol = :trad,
    rm::Symbol = :SD,
    rf::Real = 0.0,
)
```
Compute the risk for a given `portfolio` of `type`, for risk measure `rm`, and risk free rate `rf`.
"""
function calc_risk(
    w::AbstractVector,
    returns::AbstractMatrix,
    cov::AbstractMatrix;
    rm::Symbol = :SD,
    rf::Real = 0.0,
    alpha_i::Real = 0.0001,
    alpha::Real = 0.05,
    a_sim::Int = 100,
    beta_i::Union{<:Real, Nothing} = nothing,
    beta::Union{<:Real, Nothing} = nothing,
    b_sim::Union{<:Real, Nothing} = nothing,
    kappa::Real = 0.3,
    solvers::Union{<:AbstractDict, Nothing} = nothing,
)
    x = (rm != :Variance || rm != :SD) && returns * w

    risk = if rm == :SD
        SD(w, cov)
    elseif rm == :Variance
        Variance(w, cov)
    elseif rm == :MAD
        MAD(x)
    elseif rm == :SSD
        SSD(x)
    elseif rm == :FLPM
        FLPM(x, rf)
    elseif rm == :SLPM
        SLPM(x, rf)
    elseif rm == :WR
        WR(x)
    elseif rm == :VaR
        VaR(x, alpha)
    elseif rm == :CVaR
        CVaR(x, alpha)
    elseif rm == :EVaR
        EVaR(x, solvers, alpha)
    elseif rm == :RVaR
        RVaR(x, solvers, alpha, kappa)
    elseif rm == :DaR
        DaR_abs(x, alpha)
    elseif rm == :MDD
        MDD_abs(x)
    elseif rm == :ADD
        ADD_abs(x)
    elseif rm == :CDaR
        CDaR_abs(x, alpha)
    elseif rm == :UCI
        UCI_abs(x)
    elseif rm == :EDaR
        EDaR_abs(x, solvers, alpha)
    elseif rm == :RDaR
        RDaR_abs(x, solvers, alpha, kappa)
    elseif rm == :DaR_r
        DaR_rel(x, alpha)
    elseif rm == :MDD_r
        MDD_rel(x)
    elseif rm == :ADD_r
        ADD_rel(x)
    elseif rm == :CDaR_r
        CDaR_rel(x, alpha)
    elseif rm == :uci_r
        UCI_rel(x)
    elseif rm == :EDaR_r
        EDaR_rel(x, solvers, alpha)
    elseif rm == :RDaR_r
        RDaR_rel(x, solvers, alpha, kappa)
    elseif rm == :Kurt
        Kurt(x)
    elseif rm == :SKurt
        SKurt(x)
    elseif rm == :GMD
        GMD(x)
    elseif rm == :RG
        RG(x)
    elseif rm == :RCVaR
        RCVaR(x; alpha = alpha, beta = beta)
    elseif rm == :TG
        TG(x; alpha_i = alpha_i, alpha = alpha, a_sim = a_sim)
    elseif rm == :RTG
        RTG(
            x;
            alpha_i = alpha_i,
            alpha = alpha,
            a_sim = a_sim,
            beta_i = beta_i,
            beta = beta,
            b_sim = b_sim,
        )
    elseif rm == :OWA
        OWA(x, w)
    else
        throw(ArgumentError("rm must be one of $(union(RiskMeasures, HRRiskMeasures))"))
    end

    return risk
end

function calc_risk(
    portfolio::AbstractPortfolio;
    type::Symbol = :trad,
    rm::Symbol = :SD,
    rf::Real = 0.0,
)
    weights = if isa(portfolio, Portfolio)
        @assert(type ∈ PortTypes, "type must be one of $PortTypes")
        if type == :trad
            portfolio.p_optimal.weights
        elseif type == :rp
            portfolio.rp_optimal.weights
        elseif type == :rrp
            portfolio.rrp_optimal.weights
        elseif type == :wc
            portfolio.wc_optimal.weights
        end
    else
        portfolio.p_optimal.weights
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

export Variance,
    SD,
    MAD,
    SSD,
    FLPM,
    SLPM,
    WR,
    VaR,
    CVaR,
    ERM,
    EVaR,
    RRM,
    RVaR,
    DaR_abs,
    MDD_abs,
    ADD_abs,
    CDaR_abs,
    UCI_abs,
    EDaR_abs,
    RDaR_abs,
    DaR_rel,
    MDD_rel,
    ADD_rel,
    CDaR_rel,
    UCI_rel,
    EDaR_rel,
    RDaR_rel,
    Kurt,
    Semi_Kurt,
    calc_risk
