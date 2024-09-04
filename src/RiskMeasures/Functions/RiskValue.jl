"""
```
_Variance(w::AbstractVector, Σ::AbstractMatrix)
```

Compute the _Variance. Square of [`_SD`](@ref).

```math
\\mathrm{Variance}(\\bm{w},\\, \\mathbf{\\Sigma}) = \\bm{w}^\\intercal \\, \\mathbf{\\Sigma}\\, \\bm{w}\\,.
```

# Inputs

  - `w`: vector of asset weights.
  - `Σ`: covariance matrix of asset returns.
"""
function _Variance(w::AbstractVector, cov::AbstractMatrix)
    return dot(w, cov, w)
end

"""
```
_SD(w::AbstractVector, Σ::AbstractMatrix)
```

Compute the Standard Deviation. Square root of [`_Variance`](@ref).

```math
\\mathrm{SD}(\\bm{w},\\, \\mathbf{\\Sigma}) = \\left[\\bm{w}^\\intercal \\, \\mathbf{\\Sigma} \\, \\bm{w}\\right]^{1/2}\\,.
```

# Inputs

  - `w`: vector of asset weights.
  - `Σ`: covariance matrix of asset returns.
"""
function _SD(w::AbstractVector, cov::AbstractMatrix)
    return sqrt(_Variance(w, cov))
end

"""
```
_MAD(x::AbstractVector, w::Union{AbstractWeights, Nothing} = nothing)
```

Compute the Mean Absolute Deviation.

```math
\\mathrm{MAD}(\\bm{X}) = \\dfrac{1}{T} \\sum\\limits_{t=1}^T \\left\\lvert X_{t} - \\mathbb{E}(\\bm{X}) \\right\\rvert\\,.
```

# Inputs

  - `x`: vector of portfolio returns.
  - `w`: optional vector of weights for computing the mean.
"""
function _MAD(x::AbstractVector, w::Union{AbstractWeights, Nothing} = nothing)
    mu = isnothing(w) ? mean(x) : mean(x, w)
    return mean(abs.(x .- mu))
end

"""
```
_SSD(x::AbstractVector, r::Real = 0.0, w::Union{AbstractWeights, Nothing} = nothing)
```

Compute the mean Semi-Standard Deviation.

```math
\\mathrm{SSD}(\\bm{X}) = \\left[\\dfrac{1}{T-1} \\sum\\limits_{t=1}^{T}\\min\\left(\\bm{X}_{t} - \\mathbb{E}(\\bm{X}),\\, r\\right)^{2}\\right]^{1/2}\\,.
```

# Inputs

  - `x`: vector of portfolio returns.
  - `r`: minimum return target.
  - `w`: optional vector of weights for computing the mean.
"""
function _SSD(x::AbstractVector, target::Real = 0.0,
              w::Union{AbstractWeights, Nothing} = nothing)
    T = length(x)
    mu = isnothing(w) ? mean(x) : mean(x, w)
    val = mu .- x
    return sqrt(sum(val[val .>= target] .^ 2) / (T - 1))
end

"""
```
_FLPM(x::AbstractVector, r::Real = 0.0)
```

Compute the First Lower Partial Moment (Omega ratio).

```math
\\mathrm{FLPM}(\\bm{X},\\, r) = \\dfrac{1}{T}  \\sum\\limits_{t=1}^{T}\\max\\left(r - \\bm{X}_{t},\\, 0\\right)\\,.
```

# Inputs

  - `x`: vector of portfolio returns.
  - `r`: minimum return target.
"""
function _FLPM(x::AbstractVector, target::Real = 0.0)
    T = length(x)
    val = target .- x
    return sum(val[val .>= zero(target)]) / T
end

"""
```
_SLPM(x::AbstractVector, r::Real = 0.0)
```

Compute the Second Lower Partial Moment (Sortino Ratio).

```math
\\mathrm{SLPM}(\\bm{X},\\, r) = \\left[\\dfrac{1}{T-1} \\sum\\limits_{t=1}^{T}\\max\\left(r - \\bm{X}_{t},\\, 0\\right)^{2}\\right]^{1/2}\\,.
```

# Inputs

  - `x`: vector of portfolio returns.
  - `r`: minimum return target.
"""
function _SLPM(x::AbstractVector, target::Real = 0.0)
    T = length(x)
    val = target .- x
    return sqrt(sum(val[val .>= 0] .^ 2) / (T - 1))
end

"""
```
_WR(x::AbstractVector)
```

Compute the Worst Realisation or Worst Case Scenario.

```math
\\mathrm{WR}(\\bm{X}) = -\\min(\\bm{X})\\,.
```

# Inputs

  - `x`: vector of portfolio returns.
"""
function _WR(x::AbstractVector)
    return -minimum(x)
end

"""
```
_VaR(x::AbstractVector, α::Real = 0.05)
```

Compute the Value at Risk, used in [`_CVaR`](@ref).

```math
\\mathrm{VaR}(\\bm{X},\\, \\alpha) = -\\underset{t \\in (0,\\, T)}{\\inf} \\left\\{ X_{t} \\in \\mathbb{R} : F_{\\bm{X}}(X_{t}) > \\alpha \\right\\}\\,.
```

# Inputs

  - `x`: vector of portfolio returns.
  - `α`: significance level, `α ∈ (0, 1)`.

!!! warning

    In-place sorts the input vector.
"""
function _VaR(x::AbstractVector, alpha::Real = 0.05)
    sort!(x)
    idx = ceil(Int, alpha * length(x))
    return -x[idx]
end

"""
```
_CVaR(x::AbstractVector, α::Real = 0.05)
```

Compute the Conditional Value at Risk.

```math
\\mathrm{CVaR}(\\bm{X},\\, \\alpha) = \\mathrm{VaR}(\\bm{X},\\, \\alpha) - \\dfrac{1}{\\alpha T} \\sum\\limits_{t=1}^{T} \\min\\left( X_{t} + \\mathrm{VaR}(\\bm{X},\\, \\alpha),\\, 0\\right)\\,.
```

Where ``\\mathrm{VaR}(\\bm{X},\\, \\alpha)`` is the value at risk as defined in [`_VaR`](@ref).

# Inputs

  - `x`: vector of portfolio returns.
  - `α`: significance level, `α ∈ (0, 1)`.

!!! warning

    In-place sorts the input vector.
"""
function _CVaR(x::AbstractVector, alpha::Real = 0.05)
    sort!(x)
    idx = ceil(Int, alpha * length(x))
    var = -x[idx]
    sum_var = 0.0
    for i ∈ 1:(idx - 1)
        sum_var += x[i] + var
    end
    return var - sum_var / (alpha * length(x))
end

"""
```
ERM(x::AbstractVector, z::Real = 1.0, α::Real = 0.05)
```

Compute the Entropic Risk Measure.

```math
\\mathrm{ERM}(\\bm{X},\\, z, \\,\\alpha) = z \\ln \\left(\\dfrac{M_{\\bm{X}}\\left(z^{-1}\\right)}{\\alpha} \\right)\\,
```

Where ``M_{\\bm{X}}\\left(z^{-1}\\right)`` is the moment generating function of ``\\bm{X}``.

# Inputs

  - `x`: vector.
  - `α`: significance level, `α ∈ (0, 1)`.
  - `z`: entropic moment, can be obtained from [`get_z_from_model`](@ref) and [`get_z`](@ref) after optimising a [`Portfolio`](@ref).

```
ERM(x::AbstractVector, solvers:AbstractDict, α::Real = 0.05)
```

Compute the Entropic Risk Measure by minimising the function with respect to `z`. Used in [`_EVaR`](@ref), [`_EDaR`](@ref) and [`_EDaR_r`](@ref).

```math
\\mathrm{ERM} = \\begin{cases}
\\underset{z,\\, t,\\, u}{\\min} & t + z \\ln\\left(\\dfrac{1}{\\alpha T}\\right)\\\\
\\mathrm{s.t.} & z \\geq \\sum\\limits_{i=1}^{T} u_{i}\\\\
& (-x_{i}-t,\\, z,\\, u_{i}) \\in \\mathcal{K}_{\\exp} \\, \\forall \\, i=1,\\,\\dots{},\\, T
\\end{cases}\\,.
```

Where ``\\mathcal{K}_{\\exp}`` is the exponential cone.

# Inputs

  - `x`: vector of portfolio returns.
  - `solvers`: abstract dict containing the a JuMP-compatible solver capable of solving exponential conic problems, this argument can be formulated in various ways depending on the user's needs.

```
solvers = Dict(
    # Key-value pair for the solver, solution acceptance criteria, and solver attributes.
    :Clarabel => Dict(
        # Solver we wish to use.
        :solver => Clarabel.Optimizer, 
        # (Optional) Solution acceptance criteria.
        :check_sol => (allow_local = true, allow_almost = true), 
        # (Optional) Solver-specific attributes.
        :params => Dict("verbose" => false)
    )
)
```

The dictionary contains a key value pair for each solver (plus optional solution acceptance criteria and optional attributes) we want to use.

  - `:solver`: defines the solver to use. One can also use [`JuMP.optimizer_with_attributes`](https://jump.dev/JuMP.jl/stable/api/JuMP/#optimizer_with_attributes) to direcly provide a solver with attributes already attached.
  - `:check_sol`: (optional) defines the keyword arguments passed on to [`JuMP.is_solved_and_feasible`](https://jump.dev/JuMP.jl/stable/api/JuMP/#is_solved_and_feasible) for accepting/rejecting solutions.
  - `:params`: (optional) defines solver-specific parameters.

Users are also able to provide multiple solvers by adding additional key-value pairs to the top-level dictionary/tuple as in the following snippet.

```
solvers = Dict(
    :Clarabel => Dict(
        :solver => Clarabel.Optimizer, 
        :check_sol => (allow_local = true, allow_almost = true), 
        :params => Dict("verbose" => false)
    ),
    :COSMO => Dict(
        :solver => COSMO.Optimizer,
        ...
    ), ...
)
```

`PortfolioOptimiser` will iterate over the solvers until it finds the first one to successfully solve the problem.

  - `α`: significance level, `α ∈ (0, 1)`.

If no valid solution is found then `NaN` will be returned.
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

    success, solvers_tried = _optimise_JuMP_model(model, solvers)
    return if success
        objective_value(model)
    else
        funcname = "$(fullname(PortfolioOptimiser)[1]).$(nameof(PortfolioOptimiser.ERM))"
        @warn("$funcname: model could not be optimised satisfactorily.\nSolvers: $solvers_tried.")
        NaN
    end
end

"""
```
_EVaR(x::AbstractVector, solvers::AbstractDict, alpha::Real = 0.05)
```

Compute the Entropic Value at Risk.

```math
\\mathrm{EVaR}(\\bm{X},\\alpha) = \\underset{z > 0}{\\inf} \\left\\{\\mathrm{ERM}(\\bm{X},\\, z, \\,\\alpha)\\right\\}\\,.
```

Where ``\\mathrm{ERM}(\\bm{X},\\, z, \\,\\alpha)`` is the entropic risk measure as defined in [`ERM`](@ref).

# Inputs

  - `x`: vector of portfolio returns.
  - `α`: significance level, `α ∈ (0, 1)`.
"""
function _EVaR(x::AbstractVector, solvers::AbstractDict, alpha::Real = 0.05)
    return ERM(x, solvers, alpha)
end
"""
```
RRM(x::AbstractVector, solvers::AbstractDict, alpha::Real = 0.05,
             kappa::Real = 0.3)
```
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

    success, solvers_tried = _optimise_JuMP_model(model, solvers)
    return if success
        objective_value(model)
    else
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
        success, solvers_tried = _optimise_JuMP_model(model, solvers)
        if success
            objective_value(model)
        else
            funcname = "$(fullname(PortfolioOptimiser)[1]).$(nameof(PortfolioOptimiser.RRM))"
            @warn("$funcname: model could not be optimised satisfactorily.\nSolvers: $solvers_tried.")
            NaN
        end
    end
end

"""
```
_RLVaR(x::AbstractVector, solvers::AbstractDict, alpha::Real = 0.05, κ::Real = 0.3)
```

Compute the Relativistic Value at Risk.

```math
\\mathrm{RLVaR}(\\bm{X},\\, \\alpha,\\, \\kappa) = \\mathrm{RRM}(\\bm{X},\\, \\alpha,\\, \\kappa)\\,.
```

Where ``\\mathrm{RRM}(\\bm{X},\\, \\alpha,\\, \\kappa)`` is the Relativistic Risk Measure as defined in [`RRM`](@ref).

# Inputs

  - `x`: vector of portfolio returns.
  - `α`: significance level, `α ∈ (0, 1)`.
  - `κ`: relativistic deformation parameter.
"""
function _RLVaR(x::AbstractVector, solvers::AbstractDict, alpha::Real = 0.05,
                kappa::Real = 0.3)
    return RRM(x, solvers, alpha, kappa)
end

"""
```
_DaR(x::AbstractArray, alpha::Real = 0.05)
```

Compute the Drawdown at Risk of uncompounded cumulative returns.

```math
\\begin{align*}
\\mathrm{DaR_{a}}(\\bm{X},\\, \\alpha) &= \\underset{j \\in (0,\\, T)}{\\max} \\left\\{ \\mathrm{DD_{a}}(\\bm{X},\\, j) \\in \\mathbb{R} : F_{\\mathrm{DD}}\\left(\\mathrm{DD_{a}}(\\bm{X},\\, j)\\right) < 1 - \\alpha \\right\\}\\\\
\\mathrm{DD_{a}}(\\bm{X},\\, j) &= \\underset{t \\in (0,\\, j)}{\\max}\\left( \\sum\\limits_{i=0}^{t} x_{i} \\right) - \\sum\\limits_{i=0}^{j} x_{i}
\\end{align*}\\,.
```

Where ``\\mathrm{DD_{a}}(\\bm{X},\\, j)`` is the Drawdown of uncompounded cumulative returns, and ``\\mathrm{DaR_{a}}(\\bm{X},\\, \\alpha)`` the Drawdown at Risk of uncompounded cumulative returns.

# Inputs

  - `x`: vector of portfolio returns.
  - `α`: significance level, `α ∈ (0, 1)`.
"""
function _DaR(x::AbstractArray, alpha::Real = 0.05)
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
    popfirst!(x)
    popfirst!(dd)
    sort!(dd)
    idx = ceil(Int, alpha * T)
    return -dd[idx]
end

"""
```
_MDD(x::AbstractVector)
```

Compute the Maximum Drawdown of uncompounded cumulative returns.

```math
\\mathrm{MDD_{a}}(\\bm{X}) = \\underset{j \\in (0,\\, T)}{\\max} \\mathrm{DD_{a}}(\\bm{X},\\, j)\\,.
```

Where ``\\mathrm{DD_{a}}(\\bm{X},\\, j)`` is the Drawdown of uncompounded cumulative returns as defined in [`_DaR`](@ref).

# Inputs

  - `x`: vector of portfolio returns.
"""
function _MDD(x::AbstractVector)
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
    popfirst!(x)
    return val
end

"""
```
_ADD(x::AbstractVector)
```

Compute the Average Drawdown of uncompounded cumulative returns.

```math
\\mathrm{ADD_{a}}(\\bm{X}) = \\dfrac{1}{T} \\sum\\limits_{j=0}^{T} \\mathrm{DD_{a}}(\\bm{X},\\, j)\\,.
```

Where ``\\mathrm{DD_{a}}(\\bm{X},\\, j)`` is the Drawdown of uncompounded cumulative returns as defined in [`_DaR`](@ref).

# Inputs

  - `x`: vector of portfolio returns.
"""
function _ADD(x::AbstractVector)
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
    popfirst!(x)
    return val / T
end

"""
```
_CDaR(x::AbstractVector, alpha::Real = 0.05)
```

Compute the Conditional Drawdown at Risk of uncompounded cumulative returns.

```math
\\mathrm{CDaR_{a}}(\\bm{X},\\, \\alpha) = \\mathrm{DaR_{a}}(\\bm{X},\\, \\alpha) + \\dfrac{1}{\\alpha T} \\sum\\limits_{j=0}^{T} \\max\\left[\\mathrm{DD_{a}}(\\bm{X},\\, j) - \\mathrm{DaR_{a}}(\\bm{X},\\, \\alpha),\\, 0 \\right]\\,.
```

Where ``\\mathrm{DD_{a}}(\\bm{X},\\, j)`` is the Drawdown of uncompounded cumulative returns as defined in [`_DaR`](@ref), and ``\\mathrm{DaR_{a}}(\\bm{X},\\, \\alpha)`` the Drawdown at Risk of uncompounded cumulative returns as defined in [`_DaR`](@ref).

# Inputs

  - `x`: vector of portfolio returns.
  - `α`: significance level, `α ∈ (0, 1)`.
"""
function _CDaR(x::AbstractVector, alpha::Real = 0.05)
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
    popfirst!(x)
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
```
_UCI(x::AbstractVector)
```

Compute the Ulcer Index of uncompounded cumulative returns.

```math
\\mathrm{UCI_{a}}(\\bm{X}) = \\left[\\dfrac{1}{T} \\sum\\limits_{j=0}^{T} \\mathrm{DD_{a}}(\\bm{X},\\, j)^{2}\\right]^{1/2}\\,.
```

Where ``\\mathrm{DD_{a}}(\\bm{X},\\, j)`` is the Drawdown of uncompounded cumulative returns as defined in [`_DaR`](@ref).

# Inputs

  - `x`: vector of portfolio returns.
"""
function _UCI(x::AbstractVector)
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
    popfirst!(x)
    return sqrt(val / T)
end

"""
```
_EDaR(x::AbstractVector, solvers::AbstractDict; alpha::Real = 0.05)
```

Compute the Entropic Drawdown at Risk of uncompounded cumulative returns.

```math
\\begin{align*}
\\mathrm{EDaR_{a}}(\\bm{X},\\alpha) &= \\underset{z > 0}{\\inf} \\left\\{\\mathrm{ERM}(\\mathrm{DD_{a}}(\\bm{X}),\\, z, \\,\\alpha)\\right\\}\\\\
\\mathrm{DD_{a}}(\\bm{X}) &= \\left\\{j \\in (0,\\, T) : \\mathrm{DD_{a}}(\\bm{X},\\, j) \\right\\}
\\end{align*}\\,.
```

Where ``\\mathrm{ERM}(\\bm{X},\\, z, \\,\\alpha)`` is the entropic risk measure as defined in [`ERM`](@ref) and ``\\mathrm{DD_{a}}(\\bm{X},\\, j)`` the drawdown of uncompounded cumulative returns as defined in [`_DaR`](@ref).

# Inputs

  - `x`: vector of portfolio returns.
  - `α`: significance level, `α ∈ (0, 1)`.
"""
function _EDaR(x::AbstractVector, solvers::AbstractDict, alpha::Real = 0.05)
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
    popfirst!(x)
    popfirst!(dd)
    return ERM(dd, solvers, alpha)
end

"""
```
_RLDaR(x::AbstractVector, solvers::AbstractDict; alpha::Real = 0.05, kappa::Real = 0.3)
```

Compute the Relativistic Drawdown at Risk of uncompounded cumulative returns.

```math
\\mathrm{RLDaR_{a}}(\\bm{X},\\, \\alpha,\\, \\kappa) = \\mathrm{RRM}(\\mathrm{DD_{a}}(\\bm{X}),\\, \\alpha,\\, \\kappa)\\,.
```

Where ``\\mathrm{RRM}(\\mathrm{DD_{a}}(\\bm{X}),\\, \\alpha,\\, \\kappa)`` is the relativistic risk measure as defined in [`RRM`](@ref), and ``\\mathrm{DD_{a}}(\\bm{X})`` the drawdown of uncompounded cumulative returns as defined in [`_DaR`](@ref).

# Inputs

  - `x`: vector of portfolio returns.
  - `α`: significance level, `α ∈ (0, 1)`.
  - `κ`: relativistic deformation parameter.
"""
function _RLDaR(x::AbstractVector, solvers::AbstractDict, alpha::Real = 0.05,
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
    popfirst!(x)
    popfirst!(dd)
    return RRM(dd, solvers, alpha, kappa)
end

"""
```
_DaR_r(x::AbstractArray, α::Real = 0.05)
```

Compute the Drawdown at Risk of compounded cumulative returns.

```math
\\begin{align*}
\\mathrm{DaR_{r}}(\\bm{X},\\, \\alpha) &= \\underset{j \\in (0,\\, T)}{\\max} \\left\\{ \\mathrm{DD_{r}}(\\bm{X},\\, j) \\in \\mathbb{R} : F_{\\mathrm{DD}}\\left(\\mathrm{DD_{r}}(\\bm{X},\\, j)\\right) < 1 - \\alpha \\right\\}\\\\
\\mathrm{DD_{r}}(\\bm{X},\\, j) &= \\underset{t \\in (0,\\, j)}{\\max}\\left( \\prod\\limits_{i=0}^{t} \\left(1+x_{i}\\right) \\right) - \\prod\\limits_{i=0}^{j} \\left(1+x_{i}\\right) 
\\end{align*}\\,.
```

Where ``\\mathrm{DD_{r}}(\\bm{X},\\, j)`` is the Drawdown of compounded cumulative returns, and ``\\mathrm{DaR_{r}}(\\bm{X},\\, \\alpha)`` the Drawdown at Risk of compounded cumulative returns.

# Inputs

  - `x`: vector of portfolio returns.
  - `α`: significance level, `α ∈ (0, 1)`.
"""
function _DaR_r(x::AbstractArray, alpha::Real = 0.05)
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
```
_MDD_r(x::AbstractVector)
```

Compute the Maximum Drawdown of compounded cumulative returns.

```math
\\mathrm{MDD_{r}}(\\bm{X}) = \\underset{j \\in (0,\\, T)}{\\max} \\mathrm{DD_{r}}(\\bm{X},\\, j)\\,.
```

Where ``\\mathrm{DD_{a}}(\\bm{X},\\, j)`` is the Drawdown of compounded cumulative returns as defined in [`_DaR_r`](@ref).

# Inputs

  - `x`: vector of portfolio returns.
"""
function _MDD_r(x::AbstractVector)
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
```
_ADD_r(x::AbstractVector)
```

Compute the Average Drawdown of compounded cumulative returns.

```math
\\mathrm{ADD_{r}}(\\bm{r}) = \\dfrac{1}{T} \\sum\\limits_{j=0}^{T} \\mathrm{DD_{r}}(\\bm{X},\\, j)\\,.
```

Where ``\\mathrm{DD_{a}}(\\bm{X},\\, j)`` is the Drawdown of compounded cumulative returns as defined in [`_DaR_r`](@ref).

# Inputs

  - `x`: vector of portfolio returns.
"""
function _ADD_r(x::AbstractVector)
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
```
_CDaR_r(x::AbstractVector, alpha::Real = 0.05)
```

Compute the Conditional Drawdown at Risk of compounded cumulative returns.

```math
\\mathrm{CDaR_{r}}(\\bm{X},\\, \\alpha) = \\mathrm{DaR_{r}}(\\bm{X},\\, \\alpha) + \\dfrac{1}{\\alpha T} \\sum\\limits_{j=0}^{T} \\max\\left[\\mathrm{DD_{r}}(\\bm{X},\\, j) - \\mathrm{DaR_{r}}(\\bm{X},\\, \\alpha),\\, 0 \\right]\\,.
```

Where ``\\mathrm{DD_{r}}(\\bm{X},\\, j)`` is the Drawdown of compounded cumulative returns as defined in [`_DaR_r`](@ref), and ``\\mathrm{DaR_{r}}(\\bm{X},\\, \\alpha)`` the Drawdown at Risk of compounded cumulative returns as defined in [`_DaR_r`](@ref).

# Inputs

  - `x`: vector of portfolio returns.
  - `α`: significance level, `α ∈ (0, 1)`.
"""
function _CDaR_r(x::AbstractVector, alpha::Real = 0.05)
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
```
_UCI_r(x::AbstractVector)
```

Compute the Ulcer Index of compounded cumulative returns.

```math
\\mathrm{UCI_{r}}(\\bm{X}) = \\left[\\dfrac{1}{T} \\sum\\limits_{j=0}^{T} \\mathrm{DD_{r}}(\\bm{X},\\, j)^{2}\\right]^{1/2}\\,.
```

Where ``\\mathrm{DD_{r}}(\\bm{X},\\, j)`` is the Drawdown of compounded cumulative returns as defined in [`_DaR_r`](@ref).

# Inputs

  - `x`: vector of portfolio returns.
"""
function _UCI_r(x::AbstractVector)
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
```
_EDaR_r(x::AbstractVector, solvers::AbstractDict; alpha::Real = 0.05)
```

Compute the Entropic Drawdown at Risk of compounded cumulative returns.

```math
\\begin{align*}
\\mathrm{EDaR_{r}}(\\bm{X},\\alpha) &= \\underset{z > 0}{\\inf} \\left\\{\\mathrm{ERM}(\\mathrm{DD_{r}}(\\bm{X}),\\, z, \\,\\alpha)\\right\\}\\\\
\\mathrm{DD_{r}}(\\bm{X}) &= \\left\\{j \\in (0,\\, T) : \\mathrm{DD_{r}}(\\bm{X},\\, j) \\right\\}
\\end{align*}\\,.
```

Where ``\\mathrm{ERM}(\\bm{X},\\, z, \\,\\alpha)`` is the entropic risk measure as defined in [`ERM`](@ref) and ``\\mathrm{DD_{r}}(\\bm{X},\\, j)`` the drawdown of compounded cumulative returns as defined in [`_DaR_r`](@ref).

# Inputs

  - `x`: vector of portfolio returns.
  - `α`: significance level, `α ∈ (0, 1)`.
  - `κ`: relativistic deformation parameter.
"""
function _EDaR_r(x::AbstractVector, solvers::AbstractDict, alpha::Real = 0.05)
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
```
_RLDaR_r(x::AbstractVector, solvers::AbstractDict; alpha::Real = 0.05, kappa::Real = 0.3)
```

Compute the Relativistic Drawdown at Risk of compounded cumulative returns.

```math
\\mathrm{RLDaR_{r}}(\\bm{X},\\, \\alpha,\\, \\kappa) = \\mathrm{RRM}(\\mathrm{DD_{r}}(\\bm{X}),\\, \\alpha,\\, \\kappa)\\,.
```

Where ``\\mathrm{RRM}(\\mathrm{DD_{r}}(\\bm{X}),\\, \\alpha,\\, \\kappa)`` is the Relativistic Risk Measure as defined in [`RRM`](@ref) where the returns vector, and ``\\mathrm{DD_{r}}(\\bm{X})`` the drawdown of compounded cumulative returns as defined in [`_DaR_r`](@ref).

# Inputs

  - `x`: vector of portfolio returns.
  - `α`: significance level, `α ∈ (0, 1)`.
  - `κ`: relativistic deformation parameter.
"""
function _RLDaR_r(x::AbstractVector, solvers::AbstractDict, alpha::Real = 0.05,
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
```
_Kurt(x::AbstractVector)
```

Compute the square root kurtosis.

```math
\\mathrm{Kurt}(\\bm{X}) = \\left[\\dfrac{1}{T} \\sum\\limits_{t=1}^{T} \\left( X_{t} - \\mathbb{E}(\\bm{X}) \\right)^{4} \\right]^{1/2}\\,.
```

# Inputs

  - `x`: vector of portfolio returns.
"""
function _Kurt(x::AbstractVector, w::Union{AbstractWeights, Nothing} = nothing;
               scale::Bool = false)
    T = length(x)
    mu = isnothing(w) ? mean(x) : mean(x, w)
    val = x .- mu
    kurt = sqrt(sum(val .^ 4) / T)
    return !scale ? kurt : kurt / 2
end

"""
```
_SKurt(x::AbstractVector)
```

Compute the square root semi-kurtosis.

```math
\\mathrm{SKurt}(\\bm{X}) = \\left[\\dfrac{1}{T} \\sum\\limits_{t=1}^{T} \\min\\left( X_{t} - \\mathbb{E}(\\bm{X}),\\, 0 \\right)^{4} \\right]^{1/2}\\,.
```

# Inputs

  - `x`: vector of portfolio returns.
"""
function _SKurt(x::AbstractVector, w::Union{AbstractWeights, Nothing} = nothing;
                scale::Bool = false)
    T = length(x)
    mu = isnothing(w) ? mean(x) : mean(x, w)
    val = x .- mu
    skurt = sqrt(sum(val[val .< 0] .^ 4) / T)
    return !scale ? skurt : skurt / 2
end

"""
```
_Skew(w::AbstractVector, V::AbstractArray)
```
"""
function _Skew(w::AbstractVector, V::AbstractArray)
    return sqrt(dot(w, V, w))
end

"""
```
_GMD(x::AbstractVector)
```

Compute the Gini Mean Difference.

# Inputs

  - `x`: vector of portfolio returns.
"""
function _GMD(x::AbstractVector)
    T = length(x)
    w = owa_gmd(T)
    return dot(w, sort!(x))
end

"""
```
_RG(x::AbstractVector)
```

Compute the Range.

# Inputs

  - `x`: vector of portfolio returns.
"""
function _RG(x::AbstractVector)
    T = length(x)
    w = owa_rg(T)
    return dot(w, sort!(x))
end

"""
```
_CVaRRG(x::AbstractVector; alpha::Real = 0.05, beta::Real = alpha)
```

Compute the _CVaR Range.

# Inputs

  - `x`: vector of portfolio returns.
  - `α`: significance level of _CVaR losses, `α ∈ (0, 1)`.
  - `β`: significance level of _CVaR gains, `β ∈ (0, 1)`.

!!! warning

    In-place sorts the input vector.
"""
function _CVaRRG(x::AbstractVector; alpha::Real = 0.05, beta::Real = alpha)
    T = length(x)
    w = owa_rcvar(T; alpha = alpha, beta = beta)
    return dot(w, sort!(x))
end

"""
```
_TG(x::AbstractVector; alpha_i::Real = 0.0001, alpha::Real = 0.05, a_sim::Int = 100)
```

Compute the Tail Gini.

# Inputs

  - `x`: vector of portfolio returns.
  - `alpha_i`: start value of the significance level of _CVaR losses, `0 < alpha_i < alpha < 1`.
  - `α`: end value of the significance level of _CVaR losses, `α ∈ (0, 1)`.
  - `a_sim`: number of CVaRs to approximate the Tail Gini losses, `a_sim > 0`.

!!! warning

    In-place sorts the input vector.
"""
function _TG(x::AbstractVector; alpha_i::Real = 0.0001, alpha::Real = 0.05,
             a_sim::Int = 100)
    T = length(x)
    w = owa_tg(T; alpha_i = alpha_i, alpha = alpha, a_sim = a_sim)
    return dot(w, sort!(x))
end

"""
```
_TGRG(x::AbstractVector; alpha_i::Real = 0.0001, alpha::Real = 0.05, a_sim::Real = 100,
     beta_i::Real = alpha_i, beta::Real = alpha, b_sim::Integer = a_sim)
```

Compute the Tail Gini Range.

# Inputs

  - `x`: vector of portfolio returns.
  - `alpha_i`: start value of the significance level of _CVaR losses, `0 <alpha_i < alpha < 1`.
  - `α`: end value of the significance level of _CVaR losses, `α ∈ (0, 1)`.
  - `a_sim`: number of CVaRs to approximate the Tail Gini losses, `a_sim > 0`.
  - `beta_i`: start value of the significance level of _CVaR gains, `0 < beta_i < beta < 1`.
  - `β`: end value of the significance level of _CVaR gains, `β ∈ (0, 1)`.
  - `b_sim`: number of CVaRs to approximate the Tail Gini gains, `b_sim > 0`.

!!! warning

    In-place sorts the input vector.
"""
function _TGRG(x::AbstractVector; alpha_i::Real = 0.0001, alpha::Real = 0.05,
               a_sim::Real = 100, beta_i::Real = alpha_i, beta::Real = alpha,
               b_sim::Integer = a_sim)
    T = length(x)
    w = owa_rtg(T; alpha_i = alpha_i, alpha = alpha, a_sim = a_sim, beta_i = beta_i,
                beta = beta, b_sim = b_sim)
    return dot(w, sort!(x))
end

"""
```
_OWA(x::AbstractVector, w::AbstractVector)
```

Compute the Ordered Weight Array risk measure.

# Inputs

  - `w`: vector of asset weights.
  - `x`: vector of portfolio returns.
"""
function _OWA(x::AbstractVector, w::AbstractVector)
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
```
_dVar(x::AbstractVector)
```

Compute the Brownian distance variance.

```math
\\begin{align*}
\\mathrm{dVar}(\\bm{X}) &= \\mathrm{dCov}(\\bm{X},\\, \\bm{X}) =  \\dfrac{1}{T^{2}} \\sum\\limits_{i=1}^{T}\\sum\\limits_{j=1}^{T} A_{i,\\,j}^2\\\\
\\mathrm{dCov}(\\bm{X},\\, \\bm{Y}) &= \\dfrac{1}{T^{2}} \\sum\\limits_{i=1}^{T} \\sum\\limits_{j=1}^{T} A_{i,\\,j} B_{i,\\,j}\\\\
A_{i,\\,j} &= a_{i,\\,j} - \\bar{a}_{i\\,.} - \\bar{a}_{.\\,j} + \\bar{a}_{.\\,.}\\\\
B_{i,\\,j} &= b_{i,\\,j} - \\bar{b}_{i\\,.} - \\bar{b}_{.\\,j} + \\bar{b}_{.\\,.}\\\\
a_{i,\\,j} &= \\lVert X_{i} - X_{j} \\rVert_{2}, \\quad \\forall i,\\, j = 1,\\, \\ldots ,\\, T\\\\
b_{i,\\,j} &= \\lVert Y_{i} - Y_{j} \\rVert_{2}, \\quad \\forall i,\\, j = 1,\\, \\ldots ,\\, T
\\end{align*}\\,.
```

where:

  - ``\\bm{X}`` and ``\\bm{Y}`` are random variables, they are equal in this case as they are the portfolio returns.
  - ``a_{i,\\,j}`` and ``b_{i,\\,j}`` are entries of a distance matrix where ``i`` and ``j`` are points in time. Each entry is defined as the Euclidean distance ``\\lVert \\cdot \\rVert_{2}`` between the value of the random variable at time ``i`` and its value at time ``j``.
  - ``\\bar{a}_{i,\\,\\cdot}`` and ``\\bar{b}_{i,\\,\\cdot}`` are the ``i``-th row means of their respective matrices.
  - ``\\bar{a}_{\\cdot,\\,j}`` and ``\\bar{b}_{\\cdot,\\,j}`` are the ``j``-th column means of their respective matrices.
  - ``\\bar{a}_{\\cdot,\\,\\cdot}`` and ``\\bar{b}_{\\cdot,\\,\\cdot}`` are the grand means of their respective matrices.
  - ``A_{i,\\,j}`` and ``B_{i,\\,j}`` are doubly centered distances.

# Inputs

  - `x`: vector of portfolio returns.
"""
function _dVar(x::AbstractVector)
    T = length(x)
    invT = one(T) / T
    invT2 = invT^2
    ovec = range(1; stop = 1, length = T)
    D = abs.(x * transpose(ovec) - ovec * transpose(x))
    d = vec(D)
    sd = sum(D)
    return invT2 * (dot(d, d) + invT2 * sd^2)
end
function calc_risk(sd::SD, w::AbstractVector; kwargs...)
    return _SD(w, sd.sigma)
end
function calc_risk(variance::Variance, w::AbstractVector; kwargs...)
    return _Variance(w, variance.sigma)
end
function calc_risk(mad::MAD, w::AbstractVector; X::AbstractMatrix, kwargs...)
    return _MAD(X * w, mad.w)
end
function calc_risk(ssd::SSD, w::AbstractVector; X::AbstractMatrix, kwargs...)
    return _SSD(X * w, ssd.target, ssd.w)
end
function calc_risk(flpm::FLPM, w::AbstractVector; X::AbstractMatrix, kwargs...)
    return _FLPM(X * w, flpm.target)
end
function calc_risk(slpm::SLPM, w::AbstractVector; X::AbstractMatrix, kwargs...)
    return _SLPM(X * w, slpm.target)
end
function calc_risk(::WR, w::AbstractVector; X::AbstractMatrix, kwargs...)
    return _WR(X * w)
end
function calc_risk(var::VaR, w::AbstractVector; X::AbstractMatrix, kwargs...)
    return _VaR(X * w, var.alpha)
end
function calc_risk(cvar::CVaR, w::AbstractVector; X::AbstractMatrix, kwargs...)
    return _CVaR(X * w, cvar.alpha)
end
function calc_risk(evar::EVaR, w::AbstractVector; X::AbstractMatrix, kwargs...)
    return _EVaR(X * w, evar.solvers, evar.alpha)
end
function calc_risk(rvar::RLVaR, w::AbstractVector; X::AbstractMatrix, kwargs...)
    return _RLVaR(X * w, rvar.solvers, rvar.alpha, rvar.kappa)
end
function calc_risk(dar::DaR, w::AbstractVector; X::AbstractMatrix, kwargs...)
    return _DaR(X * w, dar.alpha)
end
function calc_risk(::MDD, w::AbstractVector; X::AbstractMatrix, kwargs...)
    return _MDD(X * w)
end
function calc_risk(::ADD, w::AbstractVector; X::AbstractMatrix, kwargs...)
    return _ADD(X * w)
end
function calc_risk(cdar::CDaR, w::AbstractVector; X::AbstractMatrix, kwargs...)
    return _CDaR(X * w, cdar.alpha)
end
function calc_risk(::UCI, w::AbstractVector; X::AbstractMatrix, kwargs...)
    return _UCI(X * w)
end
function calc_risk(edar::EDaR, w::AbstractVector; X::AbstractMatrix, kwargs...)
    return _EDaR(X * w, edar.solvers, edar.alpha)
end
function calc_risk(rdar::RLDaR, w::AbstractVector; X::AbstractMatrix, kwargs...)
    return _RLDaR(X * w, rdar.solvers, rdar.alpha, rdar.kappa)
end
function calc_risk(dar::DaR_r, w::AbstractVector; X::AbstractMatrix, kwargs...)
    return _DaR_r(X * w, dar.alpha)
end
function calc_risk(::MDD_r, w::AbstractVector; X::AbstractMatrix, kwargs...)
    return _MDD_r(X * w)
end
function calc_risk(::ADD_r, w::AbstractVector; X::AbstractMatrix, kwargs...)
    return _ADD_r(X * w)
end
function calc_risk(cdar::CDaR_r, w::AbstractVector; X::AbstractMatrix, kwargs...)
    return _CDaR_r(X * w, cdar.alpha)
end
function calc_risk(::UCI_r, w::AbstractVector; X::AbstractMatrix, kwargs...)
    return _UCI_r(X * w)
end
function calc_risk(edar::EDaR_r, w::AbstractVector; X::AbstractMatrix, kwargs...)
    return _EDaR_r(X * w, edar.solvers, edar.alpha)
end
function calc_risk(rdar::RLDaR_r, w::AbstractVector; X::AbstractMatrix, kwargs...)
    return _RLDaR_r(X * w, rdar.solvers, rdar.alpha, rdar.kappa)
end
function calc_risk(kt::Kurt, w::AbstractVector; X::AbstractMatrix, kwargs...)
    return _Kurt(X * w, kt.w)
end
function calc_risk(skt::SKurt, w::AbstractVector; X::AbstractMatrix, kwargs...)
    return _SKurt(X * w, skt.w)
end
function calc_risk(::GMD, w::AbstractVector; X::AbstractMatrix, kwargs...)
    return _GMD(X * w)
end
function calc_risk(::RG, w::AbstractVector; X::AbstractMatrix, kwargs...)
    return _RG(X * w)
end
function calc_risk(rcvar::CVaRRG, w::AbstractVector; X::AbstractMatrix, kwargs...)
    return _CVaRRG(X * w; alpha = rcvar.alpha, beta = rcvar.beta)
end
function calc_risk(tg::TG, w::AbstractVector; X::AbstractMatrix, kwargs...)
    return _TG(X * w; alpha_i = tg.alpha_i, alpha = tg.alpha, a_sim = tg.a_sim)
end
function calc_risk(rtg::TGRG, w::AbstractVector; X::AbstractMatrix, kwargs...)
    return _TGRG(X * w; alpha_i = rtg.alpha_i, alpha = rtg.alpha, a_sim = rtg.a_sim,
                 beta_i = rtg.beta_i, beta = rtg.beta, b_sim = rtg.b_sim)
end
function calc_risk(owa::OWA, w::AbstractVector; X::AbstractMatrix, kwargs...)
    return _OWA(X * w, isnothing(owa.w) ? owa_gmd(size(X, 1)) : owa.w)
end
function calc_risk(::dVar, w::AbstractVector; X::AbstractMatrix, kwargs...)
    return _dVar(X * w)
end
function calc_risk(::Skew, w::AbstractVector; V::AbstractMatrix, kwargs...)
    return _Skew(w, V)
end
function calc_risk(::SSkew, w::AbstractVector; SV::AbstractMatrix, kwargs...)
    return _Skew(w, SV)
end
function calc_risk(::Equal, w::AbstractVector; delta::Real = 0, kwargs...)
    return 1 / length(w) + delta
end

export ERM, RRM, calc_risk