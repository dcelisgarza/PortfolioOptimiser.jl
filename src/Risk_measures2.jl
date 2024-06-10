#=
"""
```julia
Variance(w::AbstractVector, Σ::AbstractMatrix)
```

Compute the Variance. Square of [`SD`](@ref).

```math
\\mathrm{Variance}(\\bm{w},\\, \\mathbf{\\Sigma}) = \\bm{w}^\\intercal \\, \\mathbf{\\Sigma}\\, \\bm{w}\\,.
```

# Inputs

  - `w`: vector of asset weights.
  - `Σ`: covariance matrix of asset returns.
"""
function Variance(w::AbstractVector, cov::AbstractMatrix)
    return dot(w, cov, w)
end

"""
```julia
SD(w::AbstractVector, Σ::AbstractMatrix)
```

Compute the Standard Deviation. Square root of [`Variance`](@ref).

```math
\\mathrm{SD}(\\bm{w},\\, \\mathbf{\\Sigma}) = \\left[\\bm{w}^\\intercal \\, \\mathbf{\\Sigma} \\, \\bm{w}\\right]^{1/2}\\,.
```

# Inputs

  - `w`: vector of asset weights.
  - `Σ`: covariance matrix of asset returns.
"""
function SD(w::AbstractVector, cov::AbstractMatrix)
    return sqrt(Variance(w, cov))
end

"""
```julia
MAD(x::AbstractVector)
```

Compute the Mean Absolute Deviation.

```math
\\mathrm{MAD}(\\bm{x}) = \\dfrac{1}{T} \\sum\\limits_{t=1}^T \\left\\lvert x_t - \\mathbb{E}(\\bm{x}) \\right\\rvert\\,.
```

# Inputs

  - `x`: vector of portfolio returns.
"""
function MAD(x::AbstractVector)
    mu = mean(x)
    return mean(abs.(x .- mu))
end
"""
```julia
SSD(x::AbstractVector)
```

Compute the mean Semi-Standard Deviation.

```math
\\mathrm{SSD}(\\bm{x}) = \\left[\\dfrac{1}{T-1} \\sum\\limits_{t=1}^{T}\\min\\left(\\bm{x}_{t} - \\mathbb{E}(\\bm{x}),\\, 0\\right)^{2}\\right]^{1/2}\\,.
```

# Inputs

  - `x`: vector of portfolio returns.
"""
function SSD(x::AbstractVector)
    T = length(x)
    mu = mean(x)
    val = mu .- x
    return sqrt(sum(val[val .>= 0] .^ 2) / (T - 1))
end

"""
```julia
FLPM(x::AbstractVector; r::Real = 0.0)
```

Compute the First Lower Partial Moment (Omega ratio).

```math
\\mathrm{FLPM}(\\bm{x},\\, r) = \\dfrac{1}{T}  \\sum\\limits_{t=1}^{T}\\max\\left(r - \\bm{x}_{t},\\, 0\\right)\\,.
```

# Inputs

  - `x`: vector of portfolio returns.
  - `r`: minimum return target.
"""
function FLPM(x::AbstractVector, min_ret::Real = 0.0)
    T = length(x)
    val = min_ret .- x
    return sum(val[val .>= 0]) / T
end

"""
```julia
SLPM(x::AbstractVector; r::Real = 0.0)
```

Compute the Second Lower Partial Moment (Sortino Ratio).

```math
\\mathrm{SLPM}(\\bm{x},\\, r) = \\left[\\dfrac{1}{T-1} \\sum\\limits_{t=1}^{T}\\max\\left(r - \\bm{x}_{t},\\, 0\\right)^{2}\\right]^{1/2}\\,```
# Inputs
- `x`: vector of portfolio returns.
- `r`: minimum return target.
```
"""
function SLPM(x::AbstractVector, min_ret::Real = 0.0)
    T = length(x)
    val = min_ret .- x
    return sqrt(sum(val[val .>= 0] .^ 2) / (T - 1))
end

"""
```julia
WR(x::AbstractVector)
```

Compute the Worst Realisation or Worst Case Scenario.

```math
\\mathrm{WR}(\\bm{x}) = -\\min(\\bm{x})\\,.
```

# Inputs

  - `x`: vector of portfolio returns.
"""
function WR(x::AbstractVector)
    return -minimum(x)
end

"""
```julia
VaR(x::AbstractVector; alpha::Real = 0.05)
```

Compute the Value at Risk, used in [`CVaR`](@ref).

```math
\\mathrm{VaR}(\\bm{x},\\, \\alpha) = -\\underset{t \\in (0,\\, T)}{\\inf} \\left\\{ x_{t} \\in \\mathbb{R} : F_{\\bm{x}}(x_{t}) > \\alpha \\right\\}\\,,```
# Inputs
- `x`: vector of portfolio returns.
- `alpha`: significance level, alpha in (0, 1).
```
"""
function VaR(x::AbstractVector, alpha::Real = 0.05)
    sort!(x)
    idx = ceil(Int, alpha * length(x))
    return -x[idx]
end

"""
```
CVaR(x::AbstractVector, alpha::Real = 0.05)
```

Compute the Conditional Value at Risk.

```math
\\mathrm{CVaR}(\\bm{x},\\, \\alpha) = \\mathrm{VaR}(\\bm{x},\\, \\alpha) - \\dfrac{1}{\\alpha T} \\sum\\limits_{t=1}^{T} \\min\\left( x_t + \\mathrm{VaR}(\\bm{x},\\, \\alpha),\\, 0\\right)\\,,```
where ``\\mathrm{VaR}(\\bm{x},\\, \\alpha)`` is the value at risk as defined in [`VaR`](@ref).
# Inputs
- `x`: vector of portfolio returns.
- `alpha`: significance level, alpha in (0, 1).
```
"""
function CVaR(x::AbstractVector, alpha::Real = 0.05)
    sort!(x)
    idx = ceil(Int, alpha * length(x))
    var = -x[idx]
    sum_var = 0.0
    for i ∈ 1:(idx - 1)
        sum_var += x[i] + var
    end
    return var - sum_var / (alpha * length(x))
end

function _optimize_rm(model, solvers::AbstractDict)
    term_status = termination_status(model)
    solvers_tried = Dict()

    for (key, val) ∈ solvers
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

        if term_status ∈ ValidTermination
            break
        end

        push!(solvers_tried,
              key => Dict(:objective_val => objective_value(model),
                          :term_status => term_status,
                          :params => haskey(val, :params) ? val[:params] : missing))
    end

    return solvers_tried
end

"""
```julia
ERM(x::AbstractVector, z::Real = 1.0, alpha::Real = 0.05)
```
Compute the Entropic Risk Measure.
```math
\\mathrm{ERM}(\\bm{x},\\, z, \\,\\alpha) = z \\ln \\left( \\dfrac{M_{\\bm{x}}\\left(z^{-1}\\right)}{\\alpha} \\right)\\,,```
where ``M_{\\bm{x}}\\left(z^{-1}\\right)`` is the moment generating function of ``\\bm{x}``.
# Inputs
- `x`: vector.
- `alpha`: significance level, alpha in (0, 1).
- `z`: free parameter.
```julia
ERM(x::AbstractVector, solvers::AbstractDict, alpha::Real = 0.05)
```
Compute the Entropic Risk Measure by minimising the function with respect to `z`. Used in [`EVaR`](@ref), [`EDaR_abs`](@ref) and [`EDaR_rel`](@ref).
```math
\\mathrm{ERM} = \\begin{cases}
\\underset{z,\\, t,\\, u}{\\min} & t + z \\ln(\\dfrac{1}{\\alpha T})\\\\
\\mathrm{s.t.} & z \\geq \\sum\\limits_{i=1}^{T} u_{i}\\\\
& (-x_{i}-t,\\, z,\\, u_{i}) \\in \\mathcal{K}_{\\exp} \\, \\forall \\, i=1,\\,\\dots{},\\, T
\\end{cases}\\,,```
where ``\\mathcal{K}_{\\exp}`` is the exponential cone.
# Inputs
- `x`: vector.
$(_solver_desc("the `JuMP` model.", "", "`MOI.ExponentialCone`"))
- `alpha`: significance level, alpha in (0, 1).

"""
function ERM(x::AbstractVector, z::Real = 1.0, alpha::Real = 0.05)
    @smart_assert(zero(alpha) < alpha < one(alpha))
    val = mean(exp.(-x / z))
    val = z * log(val / alpha)
    return val
end
function ERM(x::AbstractVector, solvers::AbstractDict, alpha::Real = 0.05)
    @smart_assert(zero(alpha) < alpha < one(alpha))

    model = JuMP.Model()
    set_string_names_on_creation(model, false)

    T = length(x)
    at = alpha * T
    @variable(model, t)
    @variable(model, z >= 0)
    @variable(model, u[1:T])
    @constraint(model, sum(u) <= z)
    @constraint(model, [i = 1:T], [-x[i] - t, z, u[i]] ∈ MOI.ExponentialCone())
    @expression(model, risk, t - z * log(at))
    @objective(model, Min, risk)

    solvers_tried = _optimize_rm(model, solvers)
    term_status = termination_status(model)

    if term_status ∉ ValidTermination
        funcname = "$(fullname(PortfolioOptimiser)[1]).$(nameof(PortfolioOptimiser.ERM))"
        @warn("$funcname: model could not be optimised satisfactorily.\nSolvers: $solvers_tried.")
        return NaN
    end

    obj_val = objective_value(model)

    return obj_val
end

"""
```julia
EVaR(x::AbstractVector, solvers::AbstractDict, alpha::Real = 0.05)
```
Compute the Entropic Value at Risk.
```math
\\mathrm{EVaR}(\\bm{x},\\alpha) = \\underset{z > 0}{\\inf} \\left\\{\\mathrm{ERM}(\\bm{x},\\, z, \\,\\alpha)\\right\\}\\,,```
where ``\\mathrm{ERM}(\\bm{x},\\, z, \\,\\alpha)`` is the entropic risk measure as defined in [`ERM`](@ref).
# Inputs
- `x`: vector of portfolio returns.
$(_solver_desc("the `JuMP` model.", "", "`MOI.ExponentialCone`"))
- `alpha`: significance level, alpha in (0, 1).
"""
function EVaR(x::AbstractVector, solvers::AbstractDict, alpha::Real = 0.05)
    return ERM(x, solvers, alpha)
end

"""
```julia
RRM(
    x::AbstractVector,    solvers::AbstractDict,    alpha::Real = 0.05,    κ::Real = 0.3)
```
Compute the Relativistic Risk Measure.
```math
\\mathrm{RRM}(\\bm{x},\\, \\alpha,\\, \\kappa) = \\begin{cases}
\\underset{z,\\, t,\\, \\psi,\\, \\theta,\\, \\varepsilon,\\, \\omega}{\\min} & t + z \\ln_{\\kappa}\\left(\\dfrac{1}{\\alpha T}\\right) + \\sum\\limits_{i=1}^{T} \\left(\\psi_{i} + \\theta_{i}\\right) \\\\
\\mathrm{s.t.} & z \\geq 0 \\\\
& -x_{i} - t + \\varepsilon_{i} + \\omega_{i} \\leq 0 \\\\
& \\left(z\\left(\\dfrac{1+\\kappa}{2\\kappa}\\right),\\, \\psi_{i}\\left(\\dfrac{1+\\kappa}{\\kappa}\\right),\\, \\varepsilon_{i} \\right) \\in \\mathcal{P}_{3}^{1/(1+\\kappa),\\, \\kappa/(1+\\kappa)} \\\\
& \\left(\\omega_{i}\\left(\\dfrac{1}{1-\\kappa}\\right),\\, \\theta_{i}\\left(\\dfrac{1}{\\kappa}\\right),\\, -z\\left(\\dfrac{1}{2\\kappa}\\right) \\right) \\in \\mathcal{P}_{3}^{1-\\kappa,\\, \\kappa} \\\\
& \\forall \\, i = 1,\\,\\ldots{},\\, T
\\end{cases}\\,,```
where ``\\ln_{\\kappa}(x) = \\dfrac{x^{\\kappa} - x^{-\\kappa}}{2 \\kappa}`` and ``\\mathcal{P}_3^{\\alpha,\\, 1-\\alpha}`` is the 3D Power Cone.
# Inputs
- `x`: vector.
$(_solver_desc("the `JuMP` model.", "", "`MOI.PowerCone`"))
- `alpha`: significance level, alpha in (0, 1).
- `κ`: relativistic deformation parameter.
"""
function RRM(x::AbstractVector, solvers::AbstractDict, alpha::Real = 0.05,
             kappa::Real = 0.3)
    @smart_assert(zero(alpha) < alpha < one(alpha))
    @smart_assert(zero(kappa) < kappa < one(kappa))

    T = length(x)

    at = alpha * T
    invat = 1 / at
    ln_k = (invat^kappa - invat^(-kappa)) / (2 * kappa)
    invk2 = 1 / (2 * kappa)
    opk = 1 + kappa
    omk = 1 - kappa
    invk = 1 / kappa
    invopk = 1 / opk
    invomk = 1 / omk

    model = JuMP.Model()
    set_string_names_on_creation(model, false)

    @variable(model, t)
    @variable(model, z >= 0)
    @variable(model, omega[1:T])
    @variable(model, psi[1:T])
    @variable(model, theta[1:T])
    @variable(model, epsilon[1:T])
    @constraint(model, [i = 1:T],
                [z * opk * invk2, psi[i] * opk * invk, epsilon[i]] ∈ MOI.PowerCone(invopk))
    @constraint(model, [i = 1:T],
                [omega[i] * invomk, theta[i] * invk, -z * invk2] ∈ MOI.PowerCone(omk))
    @constraint(model, -x .- t .+ epsilon .+ omega .<= 0)
    @expression(model, risk, t + ln_k * z + sum(psi .+ theta))
    @objective(model, Min, risk)

    solvers_tried = _optimize_rm(model, solvers)
    term_status = termination_status(model)

    if term_status ∉ ValidTermination
        model = JuMP.Model()
        set_string_names_on_creation(model, false)

        @variable(model, z[1:T])
        @variable(model, nu[1:T])
        @variable(model, tau[1:T])
        @constraint(model, sum(z) == 1)
        @constraint(model, sum(nu .- tau) * invk2 <= ln_k)
        @constraint(model, [i = 1:T], [nu[i], 1, z[i]] ∈ MOI.PowerCone(invopk))
        @constraint(model, [i = 1:T], [z[i], 1, tau[i]] ∈ MOI.PowerCone(omk))
        @expression(model, risk, -transpose(z) * x)
        @objective(model, Max, risk)

        solvers_tried = _optimize_rm(model, solvers)
        term_status = termination_status(model)
    end

    if term_status ∉ ValidTermination
        funcname = "$(fullname(PortfolioOptimiser)[1]).$(nameof(PortfolioOptimiser.RRM))"
        @warn("$funcname: model could not be optimised satisfactorily.\nSolvers: $solvers_tried.")
        return NaN
    end

    obj_val = objective_value(model)

    return obj_val
end
"""
```julia
RVaR(
    x::AbstractVector,    solvers::AbstractDict,    alpha::Real = 0.05,    κ::Real = 0.3)
```
Compute the Relativistic Value at Risk.
```math
\\mathrm{RVaR}(\\bm{x},\\, \\alpha,\\, \\kappa) = \\mathrm{RRM}(\\bm{x},\\, \\alpha,\\, \\kappa)\\,,```
where ``\\mathrm{RRM}(\\bm{x},\\, \\alpha,\\, \\kappa)`` is the Relativistic Risk Measure as defined in [`RRM`](@ref).
# Inputs
- `x`: vector of portfolio returns.
$(_solver_desc("the `JuMP` model.", "", "`MOI.PowerCone`"))
- `alpha`: significance level, alpha in (0, 1).
- `κ`: relativistic deformation parameter.
"""
function RVaR(x::AbstractVector, solvers::AbstractDict, alpha::Real = 0.05,
              kappa::Real = 0.3)
    return RRM(x, solvers, alpha, kappa)
end

"""
```julia
DaR_abs(x::AbstractArray; alpha::Real = 0.05)
```

Compute the Drawdown at Risk of uncompounded cumulative returns.

```math
\\begin{align*}
\\mathrm{DaR_{a}}(\\bm{x},\\, \\alpha) &= \\underset{j \\in (0,\\, T)}{\\max} \\left\\{ \\mathrm{DD_{a}}(\\bm{x},\\, j) \\in \\mathbb{R} : F_{\\mathrm{DD}}\\left(\\mathrm{DD_{a}}(\\bm{x},\\, j)\\right) < 1 - \\alpha \\right\\}\\\\
\\mathrm{DD_{a}}(\\bm{x},\\, j) &= \\underset{t \\in (0,\\, j)}{\\max}\\left( \\sum\\limits_{i=0}^{t} x_{i} \\right) - \\sum\\limits_{i=0}^{j} x_{i}
\\end{align*}\\,.
```

# Inputs

  - `x`: vector of portfolio returns.
  - `alpha`: significance level, alpha in (0, 1).
"""
function DaR_abs(x::AbstractArray, alpha::Real = 0.05)
    T = length(x)
    pushfirst!(x, 1)
    cs = cumsum(x)
    peak = -Inf
    dd = similar(cs)
    for (idx, i) ∈ pairs(cs)
        if i > peak
            peak = i
        end
        dd[idx] = i - peak
    end
    popfirst!(dd)
    sort!(dd)
    idx = ceil(Int, alpha * T)
    return -dd[idx]
end

"""
```julia
MDD_abs(x::AbstractVector)
```

Compute the Maximum Drawdown of uncompounded cumulative returns.

```math
\\mathrm{MDD_{a}}(\\bm{x}) = \\underset{j \\in (0,\\, T)}{\\max} \\mathrm{DD_{a}}(\\bm{x},\\, j)\\,,```
where ``\\mathrm{DD_{a}}(\\bm{x},\\, j)`` is the Drawdown of uncompounded cumulative returns as defined in [`DaR_abs`](@ref).
# Inputs
- `x`: vector of portfolio returns.
```
"""
function MDD_abs(x::AbstractVector)
    pushfirst!(x, 1)
    cs = cumsum(x)
    val = 0.0
    peak = -Inf
    for i ∈ cs
        if i > peak
            peak = i
        end
        dd = peak - i
        if dd > val
            val = dd
        end
    end

    return val
end

"""
```julia
ADD_abs(x::AbstractVector)
```

Compute the Average Drawdown of uncompounded cumulative returns.

```math
\\mathrm{ADD_{a}}(\\bm{x}) = \\dfrac{1}{T} \\sum\\limits_{j=0}^{T} \\mathrm{DD_{a}}(\\bm{x},\\, j)\\,,```
where ``\\mathrm{DD_{a}}(\\bm{x},\\, j)`` is the Drawdown of uncompounded cumulative returns as defined in [`DaR_abs`](@ref).
# Inputs
- `x`: vector of portfolio returns.
```
"""
function ADD_abs(x::AbstractVector)
    T = length(x)
    pushfirst!(x, 1)
    cs = cumsum(x)
    val = 0.0
    peak = -Inf
    for i ∈ cs
        if i > peak
            peak = i
        end
        dd = peak - i
        if dd > 0
            val += dd
        end
    end

    return val / T
end

"""
```julia
CDaR_abs(x::AbstractVector; alpha::Real = 0.05)
```

Compute the Conditional Drawdown at Risk of uncompounded cumulative returns.

```math
\\mathrm{CDaR_{a}}(\\bm{x},\\, \\alpha) = \\mathrm{DaR_{a}}(\\bm{x},\\, \\alpha) + \\dfrac{1}{\\alpha T} \\sum\\limits_{j=0}^{T} \\max\\left[\\mathrm{DD_{a}}(\\bm{x},\\, j) - \\mathrm{DaR_{a}}(\\bm{x},\\, \\alpha),\\, 0 \\right] \\,,```
where ``\\mathrm{DD_{a}}(\\bm{x},\\, j)`` is the Drawdown of uncompounded cumulative returns as defined in [`DaR_abs`](@ref), and ``\\mathrm{DaR_{a}}(\\bm{x},\\, \\alpha)`` the Drawdown at Risk of uncompounded cumulative returns as defined in [`DaR_abs`](@ref).
# Inputs
- `x`: vector of portfolio returns.
- `alpha`: significance level, alpha in (0, 1).
```
"""
function CDaR_abs(x::AbstractVector, alpha::Real = 0.05)
    T = length(x)
    pushfirst!(x, 1)
    cs = cumsum(x)
    peak = -Inf
    dd = similar(cs)
    for (idx, i) ∈ pairs(cs)
        if i > peak
            peak = i
        end
        dd[idx] = i - peak
    end
    popfirst!(dd)
    sort!(dd)
    idx = ceil(Int, alpha * T)
    var = -dd[idx]
    sum_var = 0.0
    for i ∈ 1:(idx - 1)
        sum_var += dd[i] + var
    end
    return var - sum_var / (alpha * T)
end

"""
```julia
UCI_abs(x::AbstractVector)
```

Compute the Ulcer Index of uncompounded cumulative returns.

```math
\\mathrm{UCI_{a}}(\\bm{x}) = \\left[\\dfrac{1}{T} \\sum\\limits_{j=0}^{T} \\mathrm{DD_{a}}(\\bm{x},\\, j)^{2}\\right]^{1/2}\\,,```
where ``\\mathrm{DD_{a}}(\\bm{x},\\, j)`` is the Drawdown of uncompounded cumulative returns as defined in [`DaR_abs`](@ref).
# Inputs
- `x`: vector of portfolio returns.
```
"""
function UCI_abs(x::AbstractVector)
    T = length(x)
    pushfirst!(x, 1)
    cs = cumsum(x)
    val = 0.0
    peak = -Inf
    for i ∈ cs
        if i > peak
            peak = i
        end
        dd = peak - i
        if dd > 0
            val += dd^2
        end
    end

    return sqrt(val / T)
end

"""
```julia
EDaR_abs(x::AbstractVector, solvers::AbstractDict, alpha::Real = 0.05)
```
Compute the Entropic Drawdown at Risk of uncompounded cumulative returns.
```math
\\begin{align*}
\\mathrm{EDaR_{a}}(\\bm{x},\\alpha) &= \\underset{z > 0}{\\inf} \\left\\{\\mathrm{ERM}(\\mathrm{DD_{a}}(\\bm{x}),\\, z, \\,\\alpha)\\right\\}\\\\
\\mathrm{DD_{a}}(\\bm{x}) &= \\left\\{j \\in (0,\\, T) : \\mathrm{DD_{a}}(\\bm{x},\\, j) \\right\\}\\,,\\end{align*}
```
where ``\\mathrm{ERM}(\\bm{x},\\, z, \\,\\alpha)`` is the entropic risk measure as defined in [`ERM`](@ref) and ``\\mathrm{DD_{a}}(\\bm{x},\\, j)`` the drawdown of uncompounded cumulative returns as defined in [`DaR_abs`](@ref).
# Inputs
- `x`: vector of portfolio returns.
$(_solver_desc("the `JuMP` model.", "", "`MOI.ExponentialCone`"))
- `alpha`: significance level, alpha in (0, 1).
"""
function EDaR_abs(x::AbstractVector, solvers::AbstractDict, alpha::Real = 0.05)
    pushfirst!(x, 1)
    cs = cumsum(x)
    peak = -Inf
    dd = similar(cs)
    for (idx, i) ∈ pairs(cs)
        if i > peak
            peak = i
        end
        dd[idx] = -(peak - i)
    end
    popfirst!(dd)
    return ERM(dd, solvers, alpha)
end

"""
```julia
RDaR_abs(
    x::AbstractVector,    solvers::AbstractDict,    alpha::Real = 0.05,    kappa::Real = 0.3)
```
Compute the Relativistic Drawdown at Risk of uncompounded cumulative returns.
```math
\\mathrm{RDaR_{a}}(\\bm{x},\\, \\alpha,\\, \\kappa) = \\mathrm{RRM}(\\mathrm{DD_{a}}(\\bm{x}),\\, \\alpha,\\, \\kappa)\\,,```
where ``\\mathrm{RRM}(\\mathrm{DD_{a}}(\\bm{x}),\\, \\alpha,\\, \\kappa)`` is the relativistic risk measure as defined in [`RRM`](@ref), and ``\\mathrm{DD_{a}}(\\bm{x})`` the drawdown of uncompounded cumulative returns as defined in [`DaR_abs`](@ref).
# Inputs
- `x`: vector of portfolio returns.
$(_solver_desc("the `JuMP` model.", "", "`MOI.PowerCone`"))
- `alpha`: significance level, alpha in (0, 1).
- `κ`: relativistic deformation parameter.
"""
function RDaR_abs(x::AbstractVector, solvers::AbstractDict, alpha::Real = 0.05,
                  kappa::Real = 0.3)
    pushfirst!(x, 1)
    cs = cumsum(x)
    peak = -Inf
    dd = similar(cs)
    for (idx, i) ∈ pairs(cs)
        if i > peak
            peak = i
        end
        dd[idx] = i - peak
    end
    popfirst!(dd)
    return RRM(dd, solvers, alpha, kappa)
end

"""
```julia
DaR_rel(x::AbstractArray; alpha::Real = 0.05)
```

Compute the Drawdown at Risk of compounded cumulative returns.

```math
\\begin{align*}
\\mathrm{DaR_{r}}(\\bm{x},\\, \\alpha) &= \\underset{j \\in (0,\\, T)}{\\max} \\left\\{ \\mathrm{DD_{r}}(\\bm{x},\\, j) \\in \\mathbb{R} : F_{\\mathrm{DD}}\\left(\\mathrm{DD_{r}}(\\bm{x},\\, j)\\right) < 1 - \\alpha \\right\\}\\\\
\\mathrm{DD_{r}}(\\bm{x},\\, j) &= \\underset{t \\in (0,\\, j)}{\\max}\\left( \\prod\\limits_{i=0}^{t} \\left(1+x_{i}\\right) \\right) - \\prod\\limits_{i=0}^{j} \\left(1+x_{i}\\right) 
\\end{align*}\\,.
```

# Inputs

  - `x`: vector of portfolio returns.
  - `alpha`: significance level, alpha in (0, 1).
"""
function DaR_rel(x::AbstractArray, alpha::Real = 0.05)
    T = length(x)
    x .= pushfirst!(x, 0) .+ 1
    cs = cumprod(x)
    peak = -Inf
    dd = similar(cs)
    for (idx, i) ∈ pairs(cs)
        if i > peak
            peak = i
        end
        dd[idx] = i / peak - 1
    end
    popfirst!(dd)
    sort!(dd)
    idx = ceil(Int, alpha * T)
    return -dd[idx]
end

"""
```julia
MDD_rel(x::AbstractVector)
```

Compute the Maximum Drawdown of compounded cumulative returns.

```math
\\mathrm{MDD_{r}}(\\bm{x}) = \\underset{j \\in (0,\\, T)}{\\max} \\mathrm{DD_{r}}(\\bm{x},\\, j)\\,,```
where ``\\mathrm{DD_{a}}(\\bm{x},\\, j)`` is the Drawdown of compounded cumulative returns as defined in [`DaR_rel`](@ref).
# Inputs
- `x`: vector of portfolio returns.
```
"""
function MDD_rel(x::AbstractVector)
    x .= pushfirst!(x, 0) .+ 1
    cs = cumprod(x)
    val = 0.0
    peak = -Inf
    for i ∈ cs
        if i > peak
            peak = i
        end
        dd = 1 - i / peak
        if dd > val
            val = dd
        end
    end

    return val
end

"""
```julia
ADD_rel(x::AbstractVector)
```

Compute the Average Drawdown of compounded cumulative returns.

```math
\\mathrm{ADD_{r}}(\\bm{r}) = \\dfrac{1}{T} \\sum\\limits_{j=0}^{T} \\mathrm{DD_{r}}(\\bm{x},\\, j)\\,,```
where ``\\mathrm{DD_{a}}(\\bm{x},\\, j)`` is the Drawdown of compounded cumulative returns as defined in [`DaR_rel`](@ref).
# Inputs
- `x`: vector of portfolio returns.
```
"""
function ADD_rel(x::AbstractVector)
    T = length(x)
    x .= pushfirst!(x, 0) .+ 1
    cs = cumprod(x)
    val = 0.0
    peak = -Inf
    for i ∈ cs
        if i > peak
            peak = i
        end
        dd = 1 - i / peak
        if dd > 0
            val += dd
        end
    end

    return val / T
end

"""
```julia
CDaR_rel(x::AbstractVector; alpha::Real = 0.05)
```

Compute the Conditional Drawdown at Risk of compounded cumulative returns.

```math
\\mathrm{CDaR_{r}}(\\bm{x},\\, \\alpha) = \\mathrm{DaR_{r}}(\\bm{x},\\, \\alpha) + \\dfrac{1}{\\alpha T} \\sum\\limits_{j=0}^{T} \\max\\left[\\mathrm{DD_{r}}(\\bm{x},\\, j) - \\mathrm{DaR_{r}}(\\bm{x},\\, \\alpha),\\, 0 \\right] \\,,```
where ``\\mathrm{DD_{r}}(\\bm{x},\\, j)`` is the Drawdown of compounded cumulative returns as defined in [`DaR_rel`](@ref), and ``\\mathrm{DaR_{r}}(\\bm{x},\\, \\alpha)`` the Drawdown at Risk of compounded cumulative returns as defined in [`DaR_rel`](@ref).
# Inputs
- `x`: vector of portfolio returns.
- `alpha`: significance level, alpha in (0, 1).
```
"""
function CDaR_rel(x::AbstractVector, alpha::Real = 0.05)
    T = length(x)
    x .= pushfirst!(x, 0) .+ 1
    cs = cumprod(x)
    peak = -Inf
    dd = similar(cs)
    for (idx, i) ∈ pairs(cs)
        if i > peak
            peak = i
        end
        dd[idx] = i / peak - 1
    end
    popfirst!(dd)
    sort!(dd)
    idx = ceil(Int, alpha * T)
    var = -dd[idx]
    sum_var = 0.0
    for i ∈ 1:(idx - 1)
        sum_var += dd[i] + var
    end
    return var - sum_var / (alpha * T)
end

"""
```julia
UCI_rel(x::AbstractVector)
```

Compute the Ulcer Index of compounded cumulative returns.

```math
\\mathrm{UCI_{r}}(\\bm{x}) = \\left[\\dfrac{1}{T} \\sum\\limits_{j=0}^{T} \\mathrm{DD_{r}}(\\bm{x},\\, j)^{2}\\right]^{1/2}\\,,```
where ``\\mathrm{DD_{r}}(\\bm{x},\\, j)`` is the Drawdown of compounded cumulative returns as defined in [`DaR_rel`](@ref).
# Inputs
- `x`: vector of portfolio returns.
```
"""
function UCI_rel(x::AbstractVector)
    T = length(x)
    x .= pushfirst!(x, 0) .+ 1
    cs = cumprod(x)
    val = 0.0
    peak = -Inf
    for i ∈ cs
        if i > peak
            peak = i
        end
        dd = 1 - i / peak
        if dd > 0
            val += dd^2
        end
    end

    return sqrt(val / T)
end

"""
```julia
EDaR_rel(x::AbstractVector, solvers::AbstractDict, alpha::Real = 0.05)
```
Compute the Entropic Drawdown at Risk of compounded cumulative returns.
```math
\\begin{align*}
\\mathrm{EDaR_{r}}(\\bm{x},\\alpha) &= \\underset{z > 0}{\\inf} \\left\\{\\mathrm{ERM}(\\mathrm{DD_{r}}(\\bm{x}),\\, z, \\,\\alpha)\\right\\}\\\\
\\mathrm{DD_{r}}(\\bm{x}) &= \\left\\{j \\in (0,\\, T) : \\mathrm{DD_{r}}(\\bm{x},\\, j) \\right\\}\\,,\\end{align*}
```
where ``\\mathrm{ERM}(\\bm{x},\\, z, \\,\\alpha)`` is the entropic risk measure as defined in [`ERM`](@ref) and ``\\mathrm{DD_{r}}(\\bm{x},\\, j)`` the drawdown of compounded cumulative returns as defined in [`DaR_rel`](@ref).
# Inputs
- `x`: vector of portfolio returns.
$(_solver_desc("the `JuMP` model.", "", "`MOI.ExponentialCone`"))
- `alpha`: significance level, alpha in (0, 1).
- `κ`: relativistic deformation parameter.
"""
function EDaR_rel(x::AbstractVector, solvers::AbstractDict, alpha::Real = 0.05)
    x .= pushfirst!(x, 0) .+ 1
    cs = cumprod(x)
    peak = -Inf
    dd = similar(cs)
    for (idx, i) ∈ pairs(cs)
        if i > peak
            peak = i
        end
        dd[idx] = i / peak - 1
    end
    popfirst!(dd)
    return ERM(dd, solvers, alpha)
end

"""
```julia
RDaR_rel(
    x::AbstractVector,    solvers::AbstractDict,    alpha::Real = 0.05,    kappa::Real = 0.3)
```
Compute the Relativistic Drawdown at Risk of compounded cumulative returns.
```math
\\mathrm{RDaR_{r}}(\\bm{x},\\, \\alpha,\\, \\kappa) = \\mathrm{RRM}(\\mathrm{DD_{r}}(\\bm{x}),\\, \\alpha,\\, \\kappa)\\,,```
where ``\\mathrm{RRM}(\\mathrm{DD_{r}}(\\bm{x}),\\, \\alpha,\\, \\kappa)`` is the Relativistic Risk Measure as defined in [`RRM`](@ref) where the returns vector, and ``\\mathrm{DD_{r}}(\\bm{x})`` the drawdown of compounded cumulative returns as defined in [`DaR_rel`](@ref).
# Inputs
- `x`: vector of portfolio returns.
$(_solver_desc("the `JuMP` model.", "", "`MOI.PowerCone`"))
- `alpha`: significance level, alpha in (0, 1).
- `κ`: relativistic deformation parameter.
"""
function RDaR_rel(x::AbstractVector, solvers::AbstractDict, alpha::Real = 0.05,
                  kappa::Real = 0.3)
    x .= pushfirst!(x, 0) .+ 1
    cs = cumprod(x)
    peak = -Inf
    dd = similar(cs)
    for (idx, i) ∈ pairs(cs)
        if i > peak
            peak = i
        end
        dd[idx] = i / peak - 1
    end
    popfirst!(dd)
    return RRM(dd, solvers, alpha, kappa)
end

"""
```julia
Kurt(x::AbstractVector)
```

Compute the square root kurtosis.

```math
\\mathrm{Kurt}(\\bm{x}) = \\left[\\dfrac{1}{T} \\sum\\limits_{t=1}^{T} \\left( x_{t} - \\mathbb{E}(\\bm{x}) \\right)^{4} \\right]^{1/2}\\,.
```

# Inputs

  - `x`: vector of portfolio returns.
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

Compute the square root semi-kurtosis.

```math
\\mathrm{SKurt}(\\bm{x}) = \\left[\\dfrac{1}{T} \\sum\\limits_{t=1}^{T} \\min\\left( x_{t} - \\mathbb{E}(\\bm{x}),\\, 0 \\right)^{4} \\right]^{1/2}\\,.
```

# Inputs

  - `x`: vector of portfolio returns.
"""
function SKurt(x::AbstractVector)
    T = length(x)
    mu = mean(x)
    val = x .- mu
    return sqrt(sum(val[val .< 0] .^ 4) / T)
end

"""
```
Skew(w::AbstractVector, V::AbstractArray)
```
"""
function Skew(w::AbstractVector, V::AbstractArray)
    return sqrt(dot(w, V, w))
end

"""
```julia
GMD(x::AbstractVector)
```

Compute the Gini Mean Difference.

# Inputs

  - `x`: vector of portfolio returns.
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

Compute the Range.

# Inputs

  - `x`: vector of portfolio returns.
"""
function RG(x::AbstractVector)
    T = length(x)
    w = owa_rg(T)
    return dot(w, sort!(x))
end

"""
```julia
RCVaR(x::AbstractVector; alpha::Real = 0.05, beta::Real = alpha)
```

Compute the CVaR Range.

# Inputs

  - `x`: vector of portfolio returns.
  - `alpha`: significance level of CVaR losses, `alpha in (0, 1)`.
  - `beta`: significance level of CVaR gains, `beta in (0, 1)`.
"""
function RCVaR(x::AbstractVector; alpha::Real = 0.05, beta::Real = alpha)
    T = length(x)
    w = owa_rcvar(T; alpha = alpha, beta = beta)
    return dot(w, sort!(x))
end

"""
```julia
TG(x::AbstractVector; alpha_i::Real = 0.0001, alpha::Real = 0.05, a_sim::Int = 100)
```

Compute the Tail Gini.

# Inputs

  - `x`: vector of portfolio returns.
  - `alpha_i`: start value of the significance level of CVaR losses, `0 <alpha_i < alpha < 1`.
  - `alpha`: end value of the significance level of CVaR losses, `alpha in (0, 1)`.
  - `a_sim`: number of CVaRs to approximate the Tail Gini losses, `a_sim > 0`.
"""
function TG(x::AbstractVector; alpha_i::Real = 0.0001, alpha::Real = 0.05, a_sim::Int = 100)
    T = length(x)
    w = owa_tg(T; alpha_i = alpha_i, alpha = alpha, a_sim = a_sim)
    return dot(w, sort!(x))
end

"""
```julia
RTG(x::AbstractVector; alpha_i::Real = 0.0001, alpha::Real = 0.05, a_sim::Real = 100,
    beta_i::Real = alpha_i, beta::Real = alpha, b_sim::Integer = a_sim)
```

Compute the Tail Gini Range.

# Inputs

  - `x`: vector of portfolio returns.
  - `alpha_i`: start value of the significance level of CVaR losses, `0 <alpha_i < alpha < 1`.
  - `alpha`: end value of the significance level of CVaR losses, `alpha in (0, 1)`.
  - `a_sim`: number of CVaRs to approximate the Tail Gini losses, `a_sim > 0`.
  - `beta_i`: start value of the significance level of CVaR gains, `0 < beta_i < beta < 1`.
  - `beta`: end value of the significance level of CVaR gains, `beta in (0, 1)`.
  - `b_sim`: number of CVaRs to approximate the Tail Gini gains, `b_sim > 0`.
"""
function RTG(x::AbstractVector; alpha_i::Real = 0.0001, alpha::Real = 0.05,
             a_sim::Real = 100, beta_i::Real = alpha_i, beta::Real = alpha,
             b_sim::Integer = a_sim)
    T = length(x)
    w = owa_rtg(T; alpha_i = alpha_i, alpha = alpha, a_sim = a_sim, beta_i = beta_i,
                beta = beta, b_sim = b_sim)
    return dot(w, sort!(x))
end

"""
```julia
OWA(x::AbstractVector, w::AbstractVector)
```

Compute the Ordered Weight Array risk measure.

# Inputs

  - `w`: vector of asset weights.
  - `x`: vector of portfolio returns.
"""
function OWA(x::AbstractVector, w::AbstractVector)
    return dot(w, sort!(x))
end

# function L_moment(x::AbstractVector, k = 2)
#     T = length(x)
#     w = owa_l_moment(T, k)

#     return dot(w, sort!(x))
# end

# function L_Moment_CRM(
#     x::AbstractVector;#     k = 2,#     method = :SD,#     g = 0.5,#     max_phi = 0.5,#     solvers = Dict(),# )
#     T = length(x)
#     w = owa_l_moment_crm(
#         T;#         k = k,#         method = method,#         g = g,#         max_phi = max_phi,#         solvers = solvers,#     )

#     return dot(w, sort!(x))
# end

"""
DVar(x::AbstractVector)
"""
function DVar(x::AbstractVector)
    T = length(x)
    invT = one(T) / T
    invT2 = invT^2
    ovec = range(1; stop = 1, length = T)
    D = abs.(x * transpose(ovec) - ovec * transpose(x))
    d = vec(D)
    return invT2 * (dot(vec(d), vec(d)) + invT2 * dot(ovec, D, ovec)^2)
end
=#
abstract type RiskMeasure end
abstract type TradRiskMeasure <: RiskMeasure end
abstract type HCRiskMeasure <: RiskMeasure end
struct SD2 <: TradRiskMeasure end
struct MAD2 <: TradRiskMeasure end
struct SSD2 <: TradRiskMeasure end
struct FLPM2 <: TradRiskMeasure end
struct SLPM2 <: TradRiskMeasure end
struct WR2 <: TradRiskMeasure end
struct CVaR2 <: TradRiskMeasure end
struct EVaR2 <: TradRiskMeasure end
struct RVaR2 <: TradRiskMeasure end
struct MDD2 <: TradRiskMeasure end
struct ADD2 <: TradRiskMeasure end
struct CDaR2 <: TradRiskMeasure end
struct UCI2 <: TradRiskMeasure end
struct EDaR2 <: TradRiskMeasure end
struct RDaR2 <: TradRiskMeasure end
struct Kurt2 <: TradRiskMeasure end
struct SKurt2 <: TradRiskMeasure end
struct GMD2 <: TradRiskMeasure end
struct RG2 <: TradRiskMeasure end
struct RCVaR2 <: TradRiskMeasure end
struct TG2 <: TradRiskMeasure end
struct RTG2 <: TradRiskMeasure end
struct OWA2 <: TradRiskMeasure end
struct DVar2 <: TradRiskMeasure end
struct Skew2 <: TradRiskMeasure end
struct SSkew2 <: TradRiskMeasure end

struct Variance2 <: HCRiskMeasure end
struct Equal2 <: HCRiskMeasure end
struct VaR2 <: HCRiskMeasure end
struct DaR2 <: HCRiskMeasure end
struct DaR_r2 <: HCRiskMeasure end
struct MDD_r2 <: HCRiskMeasure end
struct ADD_r2 <: HCRiskMeasure end
struct CDaR_r2 <: HCRiskMeasure end
struct UCI_r2 <: HCRiskMeasure end
struct EDaR_r2 <: HCRiskMeasure end
struct RDaR_r2 <: HCRiskMeasure end

function calc_risk(::SD2, w::AbstractVector; sigma::AbstractMatrix, kwargs...)
    return SD(w, sigma)
end
function calc_risk(::Variance2, w::AbstractVector; sigma::AbstractMatrix, kwargs...)
    return Variance(w, sigma)
end
function calc_risk(::MAD2, w::AbstractVector; X::AbstractMatrix, kwargs...)
    return MAD(X * w)
end
function calc_risk(::SSD2, w::AbstractVector; X::AbstractMatrix, kwargs...)
    return SSD(X * w)
end
function calc_risk(::FLPM2, w::AbstractVector; X::AbstractMatrix, rf::Real = 0.0, kwargs...)
    return FLPM(X * w, rf)
end
function calc_risk(::SLPM2, w::AbstractVector; X::AbstractMatrix, rf::Real = 0.0, kwargs...)
    return SLPM(X * w, rf)
end
function calc_risk(::WR2, w::AbstractVector; X::AbstractMatrix, kwargs...)
    return WR(X * w)
end
function calc_risk(::VaR2, w::AbstractVector; X::AbstractMatrix, alpha::Real = 0.05,
                   kwargs...)
    return VaR(X * w, alpha)
end
function calc_risk(::CVaR2, w::AbstractVector; X::AbstractMatrix, alpha::Real = 0.05,
                   kwargs...)
    return CVaR(X * w, alpha)
end
function calc_risk(::EVaR2, w::AbstractVector; X::AbstractMatrix, solvers::Dict,
                   alpha::Real = 0.05, kwargs...)
    return EVaR(X * w, solvers, alpha)
end
function calc_risk(::RVaR2, w::AbstractVector; X::AbstractMatrix, solvers::Dict,
                   alpha::Real = 0.05, kappa::Real = 0.3, kwargs...)
    return RVaR(X * w, solvers, alpha, kappa)
end
function calc_risk(::DaR2, w::AbstractVector; X::AbstractMatrix, alpha::Real = 0.05,
                   kwargs...)
    return DaR_abs(X * w, alpha)
end
function calc_risk(::MDD2, w::AbstractVector; X::AbstractMatrix, kwargs...)
    return MDD_abs(X * w)
end
function calc_risk(::ADD2, w::AbstractVector; X::AbstractMatrix, kwargs...)
    return ADD_abs(X * w)
end
function calc_risk(::CDaR2, w::AbstractVector; X::AbstractMatrix, alpha::Real = 0.05,
                   kwargs...)
    return CDaR_abs(X * w, alpha)
end
function calc_risk(::UCI2, w::AbstractVector; X::AbstractMatrix, kwargs...)
    return UCI_abs(X * w)
end
function calc_risk(::EDaR2, w::AbstractVector; X::AbstractMatrix, solvers::Dict,
                   alpha::Real = 0.05, kwargs...)
    return EDaR_abs(X * w, solvers, alpha)
end
function calc_risk(::RDaR2, w::AbstractVector; X::AbstractMatrix, solvers::Dict,
                   alpha::Real = 0.05, kappa::Real = 0.3, kwargs...)
    return RDaR_abs(X * w, solvers, alpha, kappa)
end

function calc_risk(::DaR_r2, w::AbstractVector; X::AbstractMatrix, alpha::Real = 0.05,
                   kwargs...)
    return DaR_rel(X * w, alpha)
end
function calc_risk(::MDD_r2, w::AbstractVector; X::AbstractMatrix, kwargs...)
    return MDD_rel(X * w)
end
function calc_risk(::ADD_r2, w::AbstractVector; X::AbstractMatrix, kwargs...)
    return ADD_rel(X * w)
end
function calc_risk(::CDaR_r2, w::AbstractVector; X::AbstractMatrix, alpha::Real = 0.05,
                   kwargs...)
    return CDaR_rel(X * w, alpha)
end
function calc_risk(::UCI_r2, w::AbstractVector; X::AbstractMatrix, kwargs...)
    return UCI_rel(X * w)
end
function calc_risk(::EDaR_r2, w::AbstractVector; X::AbstractMatrix, solvers::Dict,
                   alpha::Real = 0.05, kwargs...)
    return EDaR_rel(X * w, solvers, alpha)
end
function calc_risk(::RDaR_r2, w::AbstractVector; X::AbstractMatrix, solvers::Dict,
                   alpha::Real = 0.05, kappa::Real = 0.3, kwargs...)
    return RDaR_rel(X * w, solvers, alpha, kappa)
end
function calc_risk(::Kurt2, w::AbstractVector; X::AbstractMatrix, kwargs...)
    return Kurt(X * w)
end
function calc_risk(::SKurt2, w::AbstractVector; X::AbstractMatrix, kwargs...)
    return SKurt(X * w)
end
function calc_risk(::GMD2, w::AbstractVector; X::AbstractMatrix, kwargs...)
    return GMD(X * w)
end
function calc_risk(::RG2, w::AbstractVector; X::AbstractMatrix, kwargs...)
    return RG(X * w)
end
function calc_risk(::RCVaR2, w::AbstractVector; X::AbstractMatrix, alpha::Real = 0.05,
                   beta::Real = 0.05, kwargs...)
    return RCVaR(X * w; alpha = alpha, beta = beta)
end
function calc_risk(::TG2, w::AbstractVector; X::AbstractMatrix, alpha_i::Real = 0.0001,
                   alpha::Real = 0.05, a_sim::Int = 100, kwargs...)
    return TG(X * w; alpha_i = alpha_i, alpha = alpha, a_sim = a_sim)
end
function calc_risk(::RTG2, w::AbstractVector; X::AbstractMatrix, alpha_i::Real = 0.0001,
                   alpha::Real = 0.05, a_sim::Int = 100, beta_i::Real = alpha_i,
                   beta::Real = alpha, b_sim::Integer = a_sim, kwargs...)
    return RTG(X * w; alpha_i = alpha_i, alpha = alpha, a_sim = a_sim, beta_i = beta_i,
               beta = beta, b_sim = b_sim)
end
function calc_risk(::OWA2, w::AbstractVector; X::AbstractMatrix, owa_w::AbstractVector,
                   kwargs...)
    return OWA(X * w, isempty(owa_w) ? owa_gmd(size(X, 1)) : owa_w)
end
function calc_risk(::DVar2, w::AbstractVector; X::AbstractMatrix, kwargs...)
    return DVar(X * w)
end
function calc_risk(::Skew2, w::AbstractVector; V::AbstractMatrix, kwargs...)
    return Skew(w, V)
end
function calc_risk(::SSkew2, w::AbstractVector; SV::AbstractMatrix, kwargs...)
    return Skew(w, SV)
end
function calc_risk(::Equal2, w::AbstractVector; kwargs...)
    return 1 / length(w)
end
