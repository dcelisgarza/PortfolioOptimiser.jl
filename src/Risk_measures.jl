const _rmstr = """
                - `:SD`: standard deviation ([`SD`](@ref)).$(_solver_reqs("`MOI.SecondOrderCone`"))
                - `:MAD`: max absolute deviation ([`MAD`](@ref)).
                - `:SSD`: semi standard deviation ([`SSD`](@ref)).$(_solver_reqs("`MOI.SecondOrderCone`"))
                - `:FLPM`: first lower partial moment (omega ratio) ([`FLPM`](@ref)).
                - `:SLPM`: second lower partial moment (sortino ratio) ([`SLPM`](@ref)).$(_solver_reqs("`MOI.SecondOrderCone`"))
                - `:WR`: worst realisation ([`WR`](@ref)).
                - `:CVaR`: conditional value at risk ([`CVaR`](@ref)).
                - `:EVaR`: entropic value at risk ([`EVaR`](@ref)).$(_solver_reqs("`MOI.ExponentialCone`"))
                - `:RVaR`: relativistic value at risk ([`RVaR`](@ref)).$(_solver_reqs("`MOI.PowerCone`"))
                - `:MDD`: maximum drawdown of uncompounded cumulative returns ([`MDD_abs`](@ref)).
                - `:ADD`: average drawdown of uncompounded cumulative returns ([`ADD_abs`](@ref)).
                - `:CDaR`: conditional drawdown at risk of uncompounded cumulative returns ([`CDaR_abs`](@ref)).
                - `:UCI`: ulcer index of uncompounded cumulative returns ([`UCI_abs`](@ref)).$(_solver_reqs("`MOI.SecondOrderCone`"))
                - `:EDaR`: entropic drawdown at risk of uncompounded cumulative returns ([`EDaR_abs`](@ref)).$(_solver_reqs("`MOI.ExponentialCone`"))
                - `:RDaR`: relativistic drawdown at risk of uncompounded cumulative returns ([`RDaR_abs`](@ref)).$(_solver_reqs("`MOI.PowerCone`"))
                - `:Kurt`: square root kurtosis ([`Kurt`](@ref)).$(_solver_reqs("`MOI.PSDCone` and `MOI.SecondOrderCone`"))
                - `:SKurt`: square root semi-kurtosis ([`SKurt`](@ref)).$(_solver_reqs("`MOI.PSDCone` and `MOI.SecondOrderCone`"))
                - `:GMD`: gini mean difference ([`GMD`](@ref)).
                - `:RG`: range of returns ([`RG`](@ref)).
                - `:RCVaR`: range of conditional value at risk ([`RCVaR`](@ref)).
                - `:TG`: tail gini ([`TG`](@ref)).
                - `:RTG`: range of tail gini ([`RTG`](@ref)).
                - `:OWA`: ordered weight array (generic OWA weights) ([`OWA`](@ref)).
               """

"""
```julia
RiskMeasures = (
    :SD,    :MAD,    :SSD,    :FLPM,    :SLPM,    :WR,    :CVaR,    :EVaR,    :RVaR,    :MDD,    :ADD,    :CDaR,    :UCI,    :EDaR,    :RDaR,    :Kurt,    :SKurt,    :GMD,    :RG,    :RCVaR,    :TG,    :RTG,    :OWA)
```
Available risk measures for `:Trad` and `:RP` type (see [`PortTypes`](@ref)) of [`Portfolio`](@ref).
$_rmstr
"""
const RiskMeasures = (:SD,    # _mv
                      :MAD,   # _mad
                      :SSD,   # _mad
                      :FLPM,  # _lpm
                      :SLPM,  # _lpm
                      :WR,    # _wr
                      :CVaR,  # _var
                      :EVaR,  # _var
                      :RVaR,  # _var
                      :MDD,   # _dar
                      :ADD,   # _dar
                      :CDaR,  # _dar
                      :UCI,   # _dar
                      :EDaR,  # _dar
                      :RDaR,  # _dar
                      :Kurt,  # _krt
                      :SKurt, # _krt
                      :GMD,   # _owa
                      :RG,    # _owa
                      :RCVaR, # _owa
                      :TG,    # _owa
                      :RTG,   # _owa
                      :OWA)

"""
```julia
HCRiskMeasures = (
    :SD,    :MAD,    :SSD,    :FLPM,    :SLPM,    :WR,    :CVaR,    :EVaR,    :RVaR,    :MDD,    :ADD,    :CDaR,    :UCI,    :EDaR,    :RDaR,    :Kurt,    :SKurt,    :GMD,    :RG,    :RCVaR,    :TG,    :RTG,    :OWA,    :Variance,    :Equal,    :VaR,    :DaR,    :DaR_r,    :MDD_r,    :ADD_r,    :CDaR_r,    :EDaR_r,    :RDaR_r)
```
Available risk measures for optimisations of [`HCPortfolio`](@ref). When performing an `:NCO` optimisation, or using an entropic or relativistic risk measure, the solver must meet the stated requirements.
$_rmstr
- `:Variance`: variance ([`Variance`](@ref)).
- `:Equal`: equal risk contribution, `1/N` where N is the number of assets.
- `:VaR`: value at risk ([`VaR`](@ref)).
- `:DaR`: drawdown at risk of uncompounded cumulative returns ([`DaR_abs`](@ref)).
- `:DaR_r`: drawdown at risk of compounded cumulative returns ([`DaR_rel`](@ref)).
- `:MDD_r`: maximum drawdown of compounded cumulative returns ([`MDD_rel`](@ref)).
- `:ADD_r`: average drawdown of compounded cumulative returns ([`ADD_rel`](@ref)).
- `:CDaR_r`: conditional drawdown at risk of compounded cumulative returns ([`CDaR_rel`](@ref)).
- `:UCI_r`: ulcer index of compounded cumulative returns ([`UCI_rel`](@ref)).$(_solver_reqs("`MOI.SecondOrderCone`"))
- `:EDaR_r`: entropic drawdown at risk of compounded cumulative returns ([`EDaR_rel`](@ref)).$(_solver_reqs("`MOI.ExponentialCone`"))
- `:RDaR_r`: relativistic drawdown at risk of compounded cumulative returns ([`RDaR_rel`](@ref)).$(_solver_reqs("`MOI.PowerCone`"))
"""
const HCRiskMeasures = (RiskMeasures..., :Variance, :Equal, :VaR, :DaR, :DaR_r, :MDD_r,
                        :ADD_r, :CDaR_r, :UCI_r, :EDaR_r, :RDaR_r)

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
    @smart_assert(0 < alpha < 1)
    val = mean(exp.(-x / z))
    val = z * log(val / alpha)
    return val
end
function ERM(x::AbstractVector, solvers::AbstractDict, alpha::Real = 0.05)
    @smart_assert(0 < alpha < 1)

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
    @smart_assert(0 < alpha < 1)
    @smart_assert(0 < kappa < 1)

    model = JuMP.Model()
    set_string_names_on_creation(model, false)

    T = length(x)
    at = alpha * T
    invat = 1 / at
    ln_k = (invat^kappa - invat^(-kappa)) / (2 * kappa)
    opk = 1 + kappa
    omk = 1 - kappa
    invk2 = 1 / (2 * kappa)
    invk = 1 / kappa
    invopk = 1 / opk
    invomk = 1 / omk

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
  - `alpha_i`: start value of the significance level of CVaR losses, `0 < alpha_i < alpha < 1`.
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
  - `alpha_i`: start value of the significance level of CVaR losses, `0 < alpha_i < alpha < 1`.
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
```julia
calc_risk(
    w::AbstractVector,    returns::AbstractMatrix;    rm::Symbol = :SD,    rf::Real = 0.0,    sigma::AbstractMatrix = Matrix{Float64}(undef, 0, 0),    alpha_i::Real = 0.0001,    alpha::Real = 0.05,    a_sim::Int = 100,    beta_i::Real = alpha_i,    beta::Real = alpha,    b_sim::Integer = a_sim,    kappa::Real = 0.3,    owa_w::AbstractVector{<:Real} = Vector{Float64}(undef, 0),    solvers::Union{<:AbstractDict, Nothing} = nothing)
```
Compute the value of a risk measure given a vector of asset weights and returns.
# Inputs
- `w`: vector of asset weights.
- `returns`: matrix of asset returns where columns are assets and rows are timesteps.
- `rm`: risk measure from [`RiskMeasures`](@ref) and [`HCRiskMeasures`](@ref).
- `rf`: risk-free rate at the frequency of `returns`, used as the minimum return target, `r`, in [`FLPM`](@ref) and [`SLPM`](@ref).
- `sigma`: covariance matrix of asset returns.
- `alpha_i`: start value of the significance level of CVaR losses, `0 < alpha_i < alpha < 1`.
- `alpha`: `alpha in (0, 1)`
    - if `rm` *is not* an OWA range measure: significance level; 
    - if `rm` *is* an OWA range measure: end value of the significance level of CVaR losses.
- `a_sim`: number of CVaRs to approximate the Tail Gini losses, `a_sim > 0`.
- `beta_i`: start value of the significance level of CVaR gains, `0 < beta_i < beta < 1`.
- `beta`: end value of the significance level of CVaR gains, `beta in (0, 1)`.
- `b_sim`: number of CVaRs to approximate the Tail Gini gains, `b_sim > 0`.
$(_solver_desc("the `JuMP` model.", "", "`MOI.ExponentialCone` and `MOI.PowerCone` respectively"))
```julia
calc_risk(
    portfolio::AbstractPortfolio;    type::Symbol = isa(portfolio, Portfolio) ? :Trad : :HRP,    rm::Symbol = :SD,    rf::Real = 0.0)
```
Compute the value of a risk measure given a portfolio.
# Inputs
- `portfolio`: optimised portfolio.
- `type`: type of portfolio from [`PortTypes`](@ref) or [`HCPortTypes`](@ref).
- `rm`: risk measure from [`RiskMeasures`](@ref) and [`HCRiskMeasures`](@ref).
- `rf`: risk-free rate at the frequency of `portfolio.returns`, used as the minimum return target, `r`, in [`FLPM`](@ref) and [`SLPM`](@ref).
"""
function calc_risk(w::AbstractVector, returns::AbstractMatrix; rm::Symbol = :SD,
                   rf::Real = 0.0, sigma::AbstractMatrix = Matrix{Float64}(undef, 0, 0),
                   alpha_i::Real = 0.0001, alpha::Real = 0.05, a_sim::Int = 100,
                   beta_i::Real = alpha_i, beta::Real = alpha, b_sim::Integer = a_sim,
                   kappa::Real = 0.3,
                   owa_w::AbstractVector{<:Real} = Vector{Float64}(undef, 0),
                   solvers::Union{<:AbstractDict,Nothing} = nothing)
    @smart_assert(rm ∈ HCRiskMeasures)

    x = (rm != :Variance || rm != :SD) && returns * w

    risk = if rm == :SD
        SD(w, sigma)
    elseif rm == :Variance
        Variance(w, sigma)
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
    elseif rm == :UCI_r
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
        RTG(x; alpha_i = alpha_i, alpha = alpha, a_sim = a_sim, beta_i = beta_i,
            beta = beta, b_sim = b_sim)
    elseif rm == :OWA
        T = size(returns, 1)
        w = isempty(owa_w) ? owa_gmd(T) : owa_w
        OWA(x, w)
    elseif rm == :Equal
        1 / length(w)
    end
    return risk
end

function calc_risk(portfolio::AbstractPortfolio;
                   type::Symbol = isa(portfolio, Portfolio) ? :Trad : :HRP,
                   rm::Symbol = :SD, rf::Real = 0.0,
                   owa_w::AbstractVector{<:Real} = portfolio.owa_w)
    return calc_risk(portfolio.optimal[type].weights, portfolio.returns; rm = rm, rf = rf,
                     sigma = portfolio.cov, alpha_i = portfolio.alpha_i,
                     alpha = portfolio.alpha, a_sim = portfolio.a_sim,
                     beta_i = portfolio.beta_i, beta = portfolio.beta,
                     b_sim = portfolio.b_sim, kappa = portfolio.kappa, owa_w = owa_w,
                     solvers = portfolio.solvers)
end

function _ul_risk(rm, returns, w1, w2, sigma, rf, solvers, alpha, kappa, alpha_i, beta,
                  a_sim, beta_i, b_sim, owa_w, di)
    @smart_assert(rm ∈ HCRiskMeasures)

    a1 = returns * w1
    a2 = returns * w2

    if rm == :SD
        r1 = SD(w1, sigma)
        r2 = SD(w2, sigma)
    elseif rm == :Variance
        r1 = Variance(w1, sigma)
        r2 = Variance(w2, sigma)
    elseif rm == :MAD
        r1 = MAD(a1)
        r2 = MAD(a2)
    elseif rm == :SSD
        r1 = SSD(a1)
        r2 = SSD(a2)
    elseif rm == :FLPM
        r1 = FLPM(a1, rf)
        r2 = FLPM(a2, rf)
    elseif rm == :SLPM
        r1 = SLPM(a1, rf)
        r2 = SLPM(a2, rf)
    elseif rm == :WR
        r1 = WR(a1)
        r2 = WR(a2)
    elseif rm == :VaR
        r1 = VaR(a1, alpha)
        r2 = VaR(a2, alpha)
    elseif rm == :CVaR
        r1 = CVaR(a1, alpha)
        r2 = CVaR(a2, alpha)
    elseif rm == :EVaR
        r1 = EVaR(a1, solvers, alpha)
        r2 = EVaR(a2, solvers, alpha)
    elseif rm == :RVaR
        r1 = RVaR(a1, solvers, alpha, kappa)
        r2 = RVaR(a2, solvers, alpha, kappa)
    elseif rm == :DaR
        r1 = DaR_abs(a1, alpha)
        r2 = DaR_abs(a2, alpha)
    elseif rm == :MDD
        r1 = MDD_abs(a1)
        r2 = MDD_abs(a2)
    elseif rm == :ADD
        r1 = ADD_abs(a1)
        r2 = ADD_abs(a2)
    elseif rm == :CDaR
        r1 = CDaR_abs(a1, alpha)
        r2 = CDaR_abs(a2, alpha)
    elseif rm == :UCI
        r1 = UCI_abs(a1)
        r2 = UCI_abs(a2)
    elseif rm == :EDaR
        r1 = EDaR_abs(a1, solvers, alpha)
        r2 = EDaR_abs(a2, solvers, alpha)
    elseif rm == :RDaR
        r1 = RDaR_abs(a1, solvers, alpha, kappa)
        r2 = RDaR_abs(a2, solvers, alpha, kappa)
    elseif rm == :DaR_r
        r1 = DaR_rel(a1, alpha)
        r2 = DaR_rel(a2, alpha)
    elseif rm == :MDD_r
        r1 = MDD_rel(a1)
        r2 = MDD_rel(a2)
    elseif rm == :ADD_r
        r1 = ADD_rel(a1)
        r2 = ADD_rel(a2)
    elseif rm == :CDaR_r
        r1 = CDaR_rel(a1, alpha)
        r2 = CDaR_rel(a2, alpha)
    elseif rm == :UCI_r
        r1 = UCI_rel(a1)
        r2 = UCI_rel(a2)
    elseif rm == :EDaR_r
        r1 = EDaR_rel(a1, solvers, alpha)
        r2 = EDaR_rel(a2, solvers, alpha)
    elseif rm == :RDaR_r
        r1 = RDaR_rel(a1, solvers, alpha, kappa)
        r2 = RDaR_rel(a2, solvers, alpha, kappa)
    elseif rm == :Kurt
        r1 = Kurt(a1) * 0.5
        r2 = Kurt(a2) * 0.5
    elseif rm == :SKurt
        r1 = SKurt(a1) * 0.5
        r2 = SKurt(a2) * 0.5
    elseif rm == :GMD
        r1 = GMD(a1)
        r2 = GMD(a2)
    elseif rm == :RG
        r1 = RG(a1)
        r2 = RG(a2)
    elseif rm == :RCVaR
        r1 = RCVaR(a1; alpha = alpha, beta = beta)
        r2 = RCVaR(a2; alpha = alpha, beta = beta)
    elseif rm == :TG
        r1 = TG(a1; alpha_i = alpha_i, alpha = alpha, a_sim = a_sim)
        r2 = TG(a2; alpha_i = alpha_i, alpha = alpha, a_sim = a_sim)
    elseif rm == :RTG
        r1 = RTG(a1; alpha_i = alpha_i, alpha = alpha, a_sim = a_sim, beta_i = beta_i,
                 beta = beta, b_sim = b_sim)
        r2 = RTG(a2; alpha_i = alpha_i, alpha = alpha, a_sim = a_sim, beta_i = beta_i,
                 beta = beta, b_sim = b_sim)
    elseif rm == :OWA
        T = size(returns, 1)
        w = isempty(owa_w) ? owa_gmd(T) : owa_w
        r1 = OWA(a1, w)
        r2 = OWA(a2, w)
    elseif rm == :Equal
        r1 = 1 / length(w1) + di
        r2 = 1 / length(w1) - di
    end
    return r1, r2
end

function risk_contribution(w::AbstractVector, returns::AbstractMatrix; rm::Symbol = :SD,
                           rf::Real = 0.0,
                           sigma::AbstractMatrix = Matrix{Float64}(undef, 0, 0),
                           alpha_i::Real = 0.0001, alpha::Real = 0.05, a_sim::Int = 100,
                           beta_i::Real = alpha_i, beta::Real = alpha,
                           b_sim::Integer = a_sim, di::Real = 1e-6, kappa::Real = 0.3,
                           owa_w::AbstractVector{<:Real} = Vector{Float64}(undef, 0),
                           solvers::Union{<:AbstractDict,Nothing} = nothing)
    ew = eltype(w)
    rc = zeros(ew, length(w))
    w1 = zeros(ew, length(w))
    w2 = zeros(ew, length(w))

    for i ∈ eachindex(w)
        w1 .= zero(ew)
        w1 .= w
        w1[i] += di

        w2 .= zero(ew)
        w2 .= w
        w2[i] -= di

        r1, r2 = _ul_risk(rm, returns, w1, w2, sigma, rf, solvers, alpha, kappa, alpha_i,
                          beta, a_sim, beta_i, b_sim, owa_w, di)

        rci = (r1 - r2) / (2 * di) * w[i]
        rc[i] = rci
    end

    return rc
end

function risk_contribution(portfolio::AbstractPortfolio; di::Real = 1e-6,
                           type::Symbol = isa(portfolio, Portfolio) ? :Trad : :HRP,
                           rm::Symbol = :SD, rf::Real = 0.0,
                           owa_w::AbstractVector{<:Real} = portfolio.owa_w)
    return risk_contribution(portfolio.optimal[type].weights, portfolio.returns; rm = rm,
                             rf = rf, sigma = portfolio.cov, alpha_i = portfolio.alpha_i,
                             alpha = portfolio.alpha, a_sim = portfolio.a_sim,
                             beta_i = portfolio.beta_i, beta = portfolio.beta,
                             b_sim = portfolio.b_sim, di = di, kappa = portfolio.kappa,
                             owa_w = owa_w, solvers = portfolio.solvers)
end

function sharpe_ratio(w::AbstractVector, mu::AbstractVector, returns::AbstractMatrix;
                      rm::Symbol                             = :SD,
                      rf::Real                               = 0.0,
                      kelly                                  = false,
                      sigma::AbstractMatrix                  = Matrix{Float64}(undef, 0, 0),
                      alpha_i::Real                          = 0.0001,
                      alpha::Real                            = 0.05,
                      a_sim::Int                             = 100,
                      beta_i::Real                           = alpha_i,
                      beta::Real                             = alpha,
                      b_sim::Integer                         = a_sim,
                      kappa::Real                            = 0.3,
                      owa_w                                  = Vector{Float64}(undef, 0),
                      solvers::Union{<:AbstractDict,Nothing} = nothing)
    ret = if kelly
        1 / size(returns, 1) * sum(log.(1 .+ returns * w))
    else
        dot(mu, w)
    end

    risk = calc_risk(w, returns; rm = rm, rf = rf, sigma = sigma, alpha_i = alpha_i,
                     alpha = alpha, a_sim = a_sim, beta_i = beta_i, beta = beta,
                     b_sim = b_sim, kappa = kappa, owa_w = owa_w, solvers = solvers)

    return (ret - rf) / risk
end

function sharpe_ratio(portfolio::AbstractPortfolio;
                      type::Symbol = isa(portfolio, Portfolio) ? :Trad : :HRP,
                      rm::Symbol = :SD, rf::Real = 0.0,
                      owa_w::AbstractVector{<:Real} = portfolio.owa_w, kelly = false)
    return sharpe_ratio(portfolio.optimal[type].weights, portfolio.mu, portfolio.returns;
                        rm = rm, rf = rf, kelly = kelly, sigma = portfolio.cov,
                        alpha_i = portfolio.alpha_i, alpha = portfolio.alpha,
                        a_sim = portfolio.a_sim, beta_i = portfolio.beta_i,
                        beta = portfolio.beta, b_sim = portfolio.b_sim,
                        kappa = portfolio.kappa, owa_w = owa_w, solvers = portfolio.solvers)
end

export Variance, SD, MAD, SSD, FLPM, SLPM, WR, VaR, CVaR, ERM, EVaR, RRM, RVaR, DaR_abs,
       MDD_abs, ADD_abs, CDaR_abs, UCI_abs, EDaR_abs, RDaR_abs, DaR_rel, MDD_rel, ADD_rel,
       CDaR_rel, UCI_rel, EDaR_rel, RDaR_rel, Kurt, SKurt, GMD, RG, RCVaR, TG, RTG, OWA,
       calc_risk, risk_contribution, sharpe_ratio
