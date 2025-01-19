"""
    _Variance(w::AbstractVector, Σ::AbstractMatrix)

# Description

Compute the Variance. This is the square of [`_SD`](@ref).

```math
\\begin{align}
\\mathrm{Variance}(\\bm{w},\\, \\mathbf{\\Sigma}) &= \\bm{w}^\\intercal \\, \\mathbf{\\Sigma}\\, \\bm{w}\\,.
\\end{align}
```

See also: [`Variance`](@ref), [`calc_risk(::Variance, ::AbstractVector)`](@ref), [`_SD`](@ref), [`SD`](@ref).

# Inputs

  - `w::AbstractVector`: `N×1` vector of asset weights.
  - `Σ::AbstractMatrix`: `N×N` covariance matrix of asset returns.

# Outputs

  - `variance::Real`: variance.

# Examples

```@example
# Number of assets
N = 3

# Create sample covariance matrix
Σ = [0.04 0.02 0.01;
     0.02 0.09 0.03;
     0.01 0.03 0.06]

# Create weight vector
w = [0.3, 0.4, 0.3]

# Calculate portfolio variance
variance = _Variance(w, Σ)
```
"""
function _Variance(w::AbstractVector, cov::AbstractMatrix)
    return dot(w, cov, w)
end

"""
    _SVariance(x::AbstractVector, r::Real = 0.0, w::Union{AbstractWeights, Nothing} = nothing)

# Description

Compute the Semi Variance. This is the square of [`_SSD`](@ref).

```math
\\begin{align}
\\mathrm{SVariance}(\\bm{X}) &= \\dfrac{1}{T-1} \\sum\\limits_{t=1}^{T}\\min\\left(X_{t} - \\mathbb{E}(\\bm{X}),\\, r\\right)^{2}\\,.
\\end{align}
```

See also: [`SVariance`](@ref), [`calc_risk(::SVariance, ::AbstractVector)`](@ref), [`_SSD`](@ref), [`SSD`](@ref).

# Inputs

  - `x::AbstractVector`: `T×1` returns vector.
  - `r::Real = 0.0`: minimum return target.
  - `w::Union{AbstractWeights, Nothing} = nothing`: `T×1` optional vector of weights for computing the expected return.

# Outputs

  - `svariance::Real`: semi variance.

# Behaviour

  - If `w` is `nothing`: uses simple arithmetic mean.
  - If `w` is provided: uses weighted mean for calculating deviations.

# Examples

```@example
# Sample returns vector
returns = [0.05, -0.03, 0.02, -0.01, 0.04]

# Calculate the semi variance with default parameters
sv1 = _SVariance(returns)

# Calculate with custom target return
sv2 = _SVariance(returns, 0.01)

# Calculate with weights
weights = eweights(1:length(return), 0.3)
sv3 = _SVariance(returns, 0.01, weights)
```
"""
function _SVariance(x::AbstractVector, target::Real = 0.0,
                    w::Union{AbstractWeights, Nothing} = nothing)
    T = length(x)
    mu = isnothing(w) ? mean(x) : mean(x, w)
    val = x .- mu
    return sum(val[val .<= target] .^ 2) / (T - 1)
end

"""
    _SD(w::AbstractVector, Σ::AbstractMatrix)

# Description

Compute the Standard Deviation. This is the square root of [`_Variance`](@ref).

```math
\\begin{align}
\\mathrm{SD}(\\bm{w},\\, \\mathbf{\\Sigma}) &= \\left(\\bm{w}^\\intercal \\, \\mathbf{\\Sigma} \\, \\bm{w}\\right)^{1/2}\\,.
\\end{align}
```

See also: [`SD`](@ref), [`calc_risk(::SD, ::AbstractVector)`](@ref), [`_Variance`](@ref), [`Variance`](@ref).

# Inputs

  - `w::AbstractVector`: `N×1` vector of asset weights.
  - `Σ::AbstractMatrix`: `N×N` covariance matrix of asset returns.

# Outputs

  - `sd::Real`: standard deviation.

# Examples

```@example
# Number of assets
N = 3

# Create sample covariance matrix
Σ = [0.04 0.02 0.01;
     0.02 0.09 0.03;
     0.01 0.03 0.06]

# Create weight vector
w = [0.3, 0.4, 0.3]

# Calculate portfolio standard deviation
sd = _SD(w, Σ)
```
"""
function _SD(w::AbstractVector, cov::AbstractMatrix)
    return sqrt(_Variance(w, cov))
end

"""
    _MAD(x::AbstractVector, w::Union{AbstractWeights, Nothing} = nothing)

# Description

Compute the Mean Absolute Deviation.

```math
\\begin{align}
\\mathrm{MAD}(\\bm{X}) &= \\dfrac{1}{T} \\sum\\limits_{t=1}^T \\left\\lvert X_{t} - \\mathbb{E}(\\bm{X}) \\right\\rvert\\,.
\\end{align}
```

See also: [`MAD`](@ref), [`calc_risk(::MAD, ::AbstractVector)`](@ref).

# Inputs

  - `x::AbstractVector`: `T×1` returns vector.
  - `w::Union{AbstractWeights, Nothing} = nothing`: `T×1` optional vector of weights for computing the expected return.

# Outputs

  - `mad::Real`: mean absolute deviation.

# Behaviour

  - If `w` is `nothing`: uses simple arithmetic mean.
  - If `w` is provided: uses weighted mean for calculating deviations.

# Examples

```@example
# Sample returns vector
returns = [0.05, -0.03, 0.02, -0.01, 0.04]

# Calculate the mean absolute deviation with default parameters
mad1 = _MAD(returns)

# Calculate with weights
weights = eweights(1:length(return), 0.3)
mad2 = _MAD(returns, weights)
```
"""
function _MAD(x::AbstractVector, w::Union{AbstractWeights, Nothing} = nothing)
    mu = isnothing(w) ? mean(x) : mean(x, w)
    return mean(abs.(x .- mu))
end

"""
    _SSD(x::AbstractVector, r::Real = 0.0, w::Union{AbstractWeights, Nothing} = nothing)

# Description

Compute the Semi Standard Deviation. This is the square root of [`_SVariance`](@ref).

```math
\\begin{align}
\\mathrm{SSD}(\\bm{X}) &= \\left(\\dfrac{1}{T-1} \\sum\\limits_{t=1}^{T}\\min\\left(X_{t} - \\mathbb{E}(\\bm{X}),\\, r\\right)^{2}\\right)^{1/2}\\,.
\\end{align}
```

See also: [`SSD`](@ref), [`calc_risk(::SSD, ::AbstractVector)`](@ref), [`_SVariance`](@ref), [`SVariance`](@ref).

# Inputs

  - `x::AbstractVector`: `T×1` returns vector.
  - `r::Real = 0.0`: minimum return target.
  - `w::Union{AbstractWeights, Nothing} = nothing`: `T×1` optional vector of weights for computing the expected return.

# Outputs

  - `ssd::Real`: semi standard deviation.

# Behaviour

  - If `w` is `nothing`: uses simple arithmetic mean.
  - If `w` is provided: uses weighted mean for calculating deviations.

# Examples

```@example
# Sample returns vector
returns = [0.05, -0.03, 0.02, -0.01, 0.04]

# Calculate the semi standard deviation with default parameters
ssd1 = _SSD(returns)

# Calculate with custom target return
ssd2 = _SSD(returns, 0.01)

# Calculate with weights
weights = eweights(1:length(return), 0.3)
ssd3 = _SSD(returns, 0.01, weights)
```
"""
function _SSD(x::AbstractVector, target::Real = 0.0,
              w::Union{AbstractWeights, Nothing} = nothing)
    T = length(x)
    mu = isnothing(w) ? mean(x) : mean(x, w)
    val = x .- mu
    return sqrt(sum(val[val .<= target] .^ 2) / (T - 1))
end

"""
    _FLPM(x::AbstractVector, r::Real = 0.0)

# Description

Compute the First Lower Partial Moment (Omega ratio).

```math
\\begin{align}
\\mathrm{FLPM}(\\bm{X},\\, r) &= \\dfrac{1}{T} \\sum\\limits_{t=1}^{T}\\max\\left(r - X_{t},\\, 0\\right)\\,.
\\end{align}
```

See also: [`FLPM`](@ref), [`calc_risk(::FLPM, ::AbstractVector)`](@ref).

# Inputs

  - `x::AbstractVector`: `T×1` returns vector.
  - `r::Real = 0.0`: minimum return target.

# Outputs

  - `flpm::Real`: first lower partial moment.

# Examples

```@example
# Sample returns vector
returns = [0.05, -0.03, 0.02, -0.01, 0.04]

# Calculate the first lower partial moment with default parameters
flpm1 = _FLPM(returns)

# Calculate with custom target return
flpm2 = _FLPM(returns, 0.01)
```
"""
function _FLPM(x::AbstractVector, target::Real = 0.0,
               w::Union{AbstractWeights, Nothing} = nothing)
    T = length(x)
    target = if !isinf(target)
        target
    else
        isnothing(w) ? mean(x) : mean(x, w)
    end
    val = x .- target
    return -sum(val[val .<= zero(target)]) / T
end

"""
    _SLPM(x::AbstractVector, r::Real = 0.0)

# Description

Compute the Second Lower Partial Moment (Sortino Ratio).

```math
\\begin{align}
\\mathrm{SLPM}(\\bm{X},\\, r) &= \\left(\\dfrac{1}{T-1} \\sum\\limits_{t=1}^{T}\\max\\left(r - X_{t},\\, 0\\right)^{2}\\right)^{1/2}\\,.
\\end{align}
```

See also: [`SLPM`](@ref), [`calc_risk(::SLPM, ::AbstractVector)`](@ref).

# Inputs

  - `x::AbstractVector`: `T×1` returns vector.
  - `r::Real = 0.0`: minimum return target.

# Outputs

  - `slpm::Real`: second lower partial moment.

# Examples

```@example
# Sample returns vector
returns = [0.05, -0.03, 0.02, -0.01, 0.04]

# Calculate the second lower partial moment with default parameters
slpm1 = _SLPM(returns)

# Calculate with custom target return
slpm2 = _SLPM(returns, 0.01)
```
"""
function _SLPM(x::AbstractVector, target::Real = 0.0,
               w::Union{AbstractWeights, Nothing} = nothing)
    T = length(x)
    target = if !isinf(target)
        target
    else
        isnothing(w) ? mean(x) : mean(x, w)
    end
    val = x .- target
    val = val[val .<= zero(target)]
    return sqrt(dot(val, val) / (T - 1))
end

"""
    _WR(x::AbstractVector)

# Description

Compute the Worst Realisation or Worst Case Scenario.

```math
\\begin{align}
\\mathrm{WR}(\\bm{X}) &= -\\min(\\bm{X})\\,.
\\end{align}
```

See also: [`WR`](@ref), [`calc_risk(::WR, ::AbstractVector)`](@ref).

# Inputs

  - `x::AbstractVector`: `T×1` returns vector.

# Outputs

  - `wr::Real`: worst realisation.

# Examples

```@example
# Sample returns vector
returns = [0.05, -0.03, 0.02, -0.01, 0.04]

# Calculate the worst realisation with default parameters
wr1 = _WR(returns)

# Calculate with custom target return
wr2 = _WR(returns, 0.01)
```
"""
function _WR(x::AbstractVector)
    return -minimum(x)
end

"""
    _VaR(x::AbstractVector, α::Real = 0.05)

# Description

Compute the Value at Risk.

```math
\\begin{align}
\\mathrm{VaR}(\\bm{X},\\, \\alpha) &= -\\underset{t \\in (0,\\, T)}{\\inf} \\left\\{ X_{t} \\in \\mathbb{R} \\, | \\, F_{\\bm{X}}(X_{t}) > \\alpha \\right\\}\\,.
\\end{align}
```

See also: [`VaR`](@ref), [`calc_risk(::VaR, ::AbstractVector)`](@ref), [`_CVaR`](@ref), [`calc_risk(::CVaR, ::AbstractVector)`](@ref).

# Inputs

  - `x::AbstractVector`: `T×1` returns vector.
  - `α::Real = 0.05`: significance level, `α ∈ (0, 1)`.

# Behaviour

!!! warning

      - In-place sorts the input vector.
      - `α` is not validated because this is an internal function. It should have been validated by [`VaR`](@ref).

# Outputs

  - `var::Real`: Value at Risk.

# Examples

```@example
# Sample returns vector
returns = [0.05, -0.03, 0.02, -0.01, 0.04]

# Calculate the value at risk with default parameters
var1 = _VaR(returns)

# Calculate with 7 % significance parameter
var2 = _VaR(returns, 0.07)
```
"""
function _VaR(x::AbstractVector, alpha::Real = 0.05)
    sort!(x)
    idx = ceil(Int, alpha * length(x))
    return -x[idx]
end

"""
    _CVaR(x::AbstractVector, α::Real = 0.05)

# Description

Compute the Conditional Value at Risk.

```math
\\begin{align}
\\mathrm{CVaR}(\\bm{X},\\, \\alpha) &= \\mathrm{VaR}(\\bm{X},\\, \\alpha) + \\dfrac{1}{\\alpha T} \\sum\\limits_{t=1}^{T} \\max\\left(-X_{t} - \\mathrm{VaR}(\\bm{X},\\, \\alpha),\\, 0\\right)\\,.
\\end{align}
```

Where:

  - ``\\mathrm{VaR}(\\bm{X},\\, \\alpha)`` is the Value at Risk as defined in [`_VaR`](@ref).

See also: [`CVaR`](@ref), [`calc_risk(::CVaR, ::AbstractVector)`](@ref), [`_VaR`](@ref), [`calc_risk(::VaR, ::AbstractVector)`](@ref).

# Inputs

  - `x::AbstractVector`: `T×1` returns vector.
  - `α::Real = 0.05`: significance level, `α ∈ (0, 1)`.

# Behaviour

!!! warning

      - In-place sorts the input vector.
      - `α` is not validated because this is an internal function. It should have been validated by [`CVaR`](@ref).

# Outputs

  - `cvar::Real`: Conditional Value at Risk.

# Examples

```@example
# Sample returns vector
returns = [0.05, -0.03, 0.02, -0.01, 0.04]

# Calculate the conditional value at risk with default parameters
cvar1 = _CVaR(returns)

# Calculate with 7 % significance parameter
cvar2 = _CVaR(returns, 0.07)
```
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
    ERM(x::AbstractVector, z::Real = 1.0, α::Real = 0.05)

# Description

Compute the Entropic Risk Measure.

```math
\\begin{align}
\\mathrm{ERM}(\\bm{X},\\, z, \\,\\alpha) &= z \\ln \\left(\\dfrac{M_{\\bm{X}}\\left(z^{-1}\\right)}{\\alpha} \\right)\\,.
\\end{align}
```

Where:

  - ``M_{\\bm{X}}\\left(t\\right)`` is the moment generating function of ``\\bm{X}``.
  - ``\\alpha \\in (0,\\,1)`` is the significance parameter.

# Inputs

  - `x::AbstractVector`: `T×1` returns vector.
  - `z::Real = 1.0`: entropic moment, can be obtained from [`get_z_from_model`](@ref) and [`get_z`](@ref) after optimising a [`Portfolio`](@ref).
  - `α::Real = 0.05`: significance level, `α ∈ (0, 1)`.

# Outputs

  - `er::Real`: entropic risk.

# Examples

```@example
# Sample returns vector
returns = [0.05, -0.03, 0.02, -0.01, 0.04]

# Calculate the entropic risk measure with default parameters
er1 = ERM(returns, 2.3, 0.03)

# Calculate with a 2.3 entropic moment and 3 % significance parameter
er2 = ERM(returns, 2.3, 0.03)
```
"""
function ERM(x::AbstractVector, z::Real = 1.0, alpha::Real = 0.05)
    val = mean(exp.(-x / z))
    val = z * log(val / alpha)
    return val
end

"""
    ERM(x::AbstractVector, solvers::AbstractDict, α::Real = 0.05)

# Description

Compute the Entropic Risk Measure.

```math
\\begin{align}
\\mathrm{ERM}(\\bm{X},\\, z, \\,\\alpha) &= 
    \\begin{cases}
        \\underset{z,\\, t,\\, u}{\\inf} & t + z \\ln\\left(\\dfrac{1}{\\alpha T}\\right)\\\\
        \\mathrm{s.t.} & z \\geq \\sum\\limits_{i=1}^{T} u_{i}\\nonumber\\\\
        & (-X_{i}-t,\\, z,\\, u_{i}) \\in \\mathcal{K}_{\\exp} \\, \\forall \\, i=1,\\,\\dots{},\\, T
    \\end{cases}\\\\
&= \\underset{z>0}{\\inf}\\left\\{ z \\ln \\left(\\dfrac{M_{\\bm{X}}\\left(z^{-1}\\right)}{\\alpha} \\right)\\right\\}\\,.
\\end{align}
```

Where:

  - ``M_{\\bm{X}}\\left(t\\right)`` is the moment generating function of ``\\bm{X}``.
  - ``\\mathcal{K}_{\\mathrm{exp}}`` is the exponential cone.
  - ``\\alpha \\in (0,\\,1)`` is the significance parameter.

See also: [`EVaR`](@ref), [`calc_risk(::EVaR, ::AbstractVector)`](@ref), [`_EVaR`](@ref), [`EDaR`](@ref), [`calc_risk(::EDaR, ::AbstractVector)`](@ref), [`_EDaR`](@ref), [`EDaR_r`](@ref), [`calc_risk(::EDaR_r, ::AbstractVector)`](@ref), [`_EDaR_r`](@ref).

# Inputs

  - `x::AbstractVector`: `T×1` returns vector.
  - `solvers::AbstractDict`: JuMP-compatible solvers for exponential cone problems.
  - `α::Real = 0.05`: significance level, `α ∈ (0, 1)`.

# Behaviour

  - If no valid solution is found returns `NaN`.

!!! warning

      - `α` is not validated because this is an internal function. It should have been validated by [`EVaR`](@ref), [`EDaR`](@ref), or [`EDaR_r`](@ref).

# Outputs

  - `er::Real`: entropic risk.

# Examples

```@example
# Sample returns vector
returns = [0.05, -0.03, 0.02, -0.01, 0.04]

# Calculate the entropic risk measure with default parameters
er1 = ERM(returns, Dict("solver" => my_solver))

# Calculate with a 3 % significance parameter
er2 = ERM(returns, Dict("solver" => my_solver), 0.03)
```
"""
function ERM(x::AbstractVector, solvers::AbstractDict, alpha::Real = 0.05)
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

    success, solvers_tried = optimise_JuMP_model(model, solvers)
    return if success
        objective_value(model)
    else
        funcname = "$(fullname(PortfolioOptimiser)[1]).$(nameof(PortfolioOptimiser.ERM))"
        @warn("$funcname: model could not be optimised satisfactorily.\nSolvers: $solvers_tried.")
        NaN
    end
end

"""
    _EVaR(x::AbstractVector, solvers::AbstractDict, α::Real = 0.05)

# Description

Compute the Entropic Value at Risk.

```math
\\begin{align}
\\mathrm{EVaR}(\\bm{X},\\alpha) &= \\underset{z > 0}{\\inf} \\left\\{\\mathrm{ERM}(\\bm{X},\\, z, \\,\\alpha)\\right\\}\\,.
\\end{align}
```

Where:

  - ``\\mathrm{ERM}(\\bm{X},\\, z, \\,\\alpha)`` is the entropic risk measure as defined in [`ERM`](@ref).

See also: [`ERM`](@ref), [`calc_risk(::EVaR, ::AbstractVector)`](@ref), [`_EVaR`](@ref).

# Inputs

  - `x::AbstractVector`: `T×1` returns vector.
  - `solvers::AbstractDict`: JuMP-compatible solvers for exponential cone problems.
  - `α::Real = 0.05`: significance level, `α ∈ (0, 1)`.

# Behaviour

  - If no valid solution is found returns `NaN`.

!!! warning

      - `α` is not validated because this is an internal function. It should have been validated by [`EVaR`](@ref).

# Outputs

  - `evar::Real`: Entropic Value at Risk.

# Examples

```@example
# Sample returns vector
returns = [0.05, -0.03, 0.02, -0.01, 0.04]

# Calculate the entropic value at risk with default parameters
evar1 = _EVaR(returns, Dict("solver" => my_solver))

# Calculate with a 3 % significance parameter
evar2 = _EVaR(returns, Dict("solver" => my_solver), 0.03)
```
"""
function _EVaR(x::AbstractVector, solvers::AbstractDict, alpha::Real = 0.05)
    return ERM(x, solvers, alpha)
end

"""
    RRM(x::AbstractVector, solvers::AbstractDict, α::Real = 0.05, κ::Real = 0.3)

# Description

Compute the Relativistic Risk Measure. Used in [`_RLVaR`](@ref), [`_RLDaR`](@ref) and [`_RLDaR_r`](@ref).

```math
\\begin{align}
\\mathrm{RRM}(X,\\,\\alpha,\\,\\kappa) &= \\left\\{
    \\begin{align}
        &\\underset{z,\\, t,\\, \\psi,\\, \\theta,\\,  \\varepsilon,\\, \\omega}{\\text{inf}} && t + z \\ln_{\\kappa} \\left(\\dfrac{1}{\\alpha T}\\right) + \\sum\\limits_{i=1}^T \\left(\\psi_{i} + \\theta_{i}  \\right) \\nonumber\\\\
        &\\mathrm{s.t.} && -X  - t + \\varepsilon + \\omega \\leq 0 \\nonumber\\\\
        &&&z \\geq 0 \\\\
        &&&\\left( z\\left(\\dfrac{1+\\kappa}{2\\kappa}\\right),\\, \\psi_{i}\\left(\\dfrac{1+\\kappa}{\\kappa}\\right),\\, \\varepsilon_{i} \\right) \\in \\mathcal{P}_{3}^{1/(1+\\kappa),\\, \\kappa/(1+\\kappa)}\\nonumber\\\\
        &&&\\left( \\omega_{i}\\left(\\dfrac{1}{1-\\kappa}\\right),\\, \\theta_{i}\\left(\\dfrac{1}{\\kappa}\\right),\\, -z \\left(\\dfrac{1}{2\\kappa}\\right) \\right) \\in \\mathcal{P}_{3}^{1-\\kappa,\\, \\kappa}\\nonumber\\\\
        &&&\\forall \\, i=1,\\,\\dots{},\\, T \\nonumber
    \\end{align}
\\right.\\,.
\\end{align}
```

Where:

  - ``\\mathcal{P}_3^{\\alpha,\\, 1-\\alpha}`` is the 3D power cone.
  - ``\\alpha \\in (0,\\,1)`` is the significance parameter.
  - ``\\kappa \\in (0,\\,1)`` is the relativistic deformation parameter.

See also: [`RLVaR`](@ref), [`calc_risk(::RLVaR, ::AbstractVector)`](@ref), [`_RLVaR`](@ref), [`RLDaR`](@ref), [`calc_risk(::RLDaR, ::AbstractVector)`](@ref), [`_RLDaR`](@ref), [`RLDaR_r`](@ref), [`calc_risk(::RLDaR_r, ::AbstractVector)`](@ref), [`_RLDaR_r`](@ref).

# Inputs

  - `x::AbstractVector`: `T×1` returns vector.
  - `solvers::AbstractDict`: JuMP-compatible solvers for 3D power cone problems.
  - `α::Real = 0.05`: significance level, `α ∈ (0, 1)`.
  - `κ::Real = 0.3`: relativistic deformation parameter, `κ ∈ (0, 1)`.

# Behaviour

  - If no valid solution is found returns `NaN`.

!!! warning

      - `α` and `κ` are not validated because this is an internal function. They should have been validated by [`RLVaR`](@ref), [`RLDaR`](@ref), or [`RLDaR_r`](@ref).

# Outputs

  - `rlr::Real`: relativistic risk.

# Examples

```@example
# Sample returns vector
returns = [0.05, -0.03, 0.02, -0.01, 0.04]

# Calculate the relativistic risk with default parameters
rlr1 = RRM(returns, Dict("solver" => my_solver))

# Calculate with a 3 % significance parameter and 80 % deformation parameter
rlr2 = RRM(returns, Dict("solver" => my_solver), 0.03, 0.8)
```
"""
function RRM(x::AbstractVector, solvers::AbstractDict, alpha::Real = 0.05,
             kappa::Real = 0.3)
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

    success, solvers_tried = optimise_JuMP_model(model, solvers)
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
        success, solvers_tried = optimise_JuMP_model(model, solvers)
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
    _RLVaR(x::AbstractVector, solvers::AbstractDict, α::Real = 0.05, κ::Real = 0.3)

# Description

Compute the Relativistic Value at Risk.

```math
\\begin{align}
\\mathrm{RLVaR}(\\bm{X},\\, \\alpha,\\, \\kappa) &= \\mathrm{RRM}(\\bm{X},\\, \\alpha,\\, \\kappa)\\,.
\\end{align}
```

Where:

  - ``\\mathrm{RRM}(\\bm{X},\\, \\alpha,\\, \\kappa)`` is the Relativistic Risk Measure as defined in [`RRM`](@ref).

See also: [`RRM`](@ref), [`RLVaR`](@ref), [`calc_risk(::RLVaR, ::AbstractVector)`](@ref).

# Inputs

  - `x::AbstractVector`: `T×1` returns vector.
  - `solvers::AbstractDict`: JuMP-compatible solvers for 3D power cone problems.
  - `α::Real = 0.05`: significance level, `α ∈ (0, 1)`.
  - `κ::Real = 0.3`: relativistic deformation parameter, `κ ∈ (0, 1)`.

# Behaviour

  - If no valid solution is found returns `NaN`.

!!! warning

      - `α` and `κ` are not validated because this is an internal function. They should have been validated by [`RLVaR`](@ref).

# Outputs

  - `rlvar::Real`: Relativistic Value at Risk.

# Examples

```@example
# Sample returns vector
returns = [0.05, -0.03, 0.02, -0.01, 0.04]

# Calculate the relativistic value at risk with default parameters
rlvar1 = _RLVaR(returns, Dict("solver" => my_solver))

# Calculate with a 3 % significance parameter and 80 % deformation parameter
rlvar2 = _RLVaR(returns, Dict("solver" => my_solver), 0.03, 0.8)
```
"""
function _RLVaR(x::AbstractVector, solvers::AbstractDict, alpha::Real = 0.05,
                kappa::Real = 0.3)
    return RRM(x, solvers, alpha, kappa)
end

"""
    _DaR(x::AbstractVector, α::Real = 0.05)

# Description

Compute the Drawdown at Risk of uncompounded cumulative returns.

```math
\\begin{align}
\\mathrm{DaR_{a}}(\\bm{X},\\, \\alpha) &= \\underset{j \\in (0,\\, T)}{\\max} \\left\\{ \\mathrm{DD_{a}}(\\bm{X},\\, j) \\in \\mathbb{R} \\, | \\, F_{\\mathrm{DD}}\\left(\\mathrm{DD_{a}}(\\bm{X},\\, j)\\right) < 1 - \\alpha \\right\\}\\\\
\\mathrm{DD_{a}}(\\bm{X},\\, j) &= \\underset{t \\in (0,\\, j)}{\\max}\\left( \\sum\\limits_{i=0}^{t} X_{i} \\right) - \\sum\\limits_{i=0}^{j} X_{i}\\\\
\\mathrm{DD_{a}}(\\bm{X}) &= \\left\\{j \\in (0,\\,T) \\, | \\, \\mathrm{DD_{a}}(\\bm{X},\\, j)\\right\\}\\,.
\\end{align}
```

Where:

  - ``\\mathrm{DD_{a}}(\\bm{X})`` is the Drawdown of uncompounded cumulative returns.
  - ``\\mathrm{DD_{a}}(\\bm{X},\\, j)`` is the Drawdown of uncompounded cumulative returns at time ``j``.
  - ``\\mathrm{DaR_{a}}(\\bm{X},\\, \\alpha)`` the Drawdown at Risk of uncompounded cumulative returns.

See also: [`DaR`](@ref), [`calc_risk(::DaR, ::AbstractVector)`](@ref), [`DaR_r`](@ref), [`calc_risk(::DaR_r, ::AbstractVector)`](@ref), [`_CDaR`](@ref), [`calc_risk(::CDaR, ::AbstractVector)`](@ref), [`_CDaR_r`](@ref), [`calc_risk(::CDaR_r, ::AbstractVector)`](@ref).

# Inputs

  - `x::AbstractVector`: `T×1` returns vector.
  - `α::Real = 0.05`: significance level, `α ∈ (0, 1)`.

# Behaviour

!!! warning

      - `α` is not validated because this is an internal function. It should have been validated by [`DaR`](@ref).

# Outputs

  - `dar::Real`: Drawdown at Risk of uncompounded cumulative returns.

# Examples

```@example
# Sample returns vector
returns = [0.05, -0.03, 0.02, -0.01, 0.04]

# Calculate the drawdown of uncompounded cumulative returns with default parameters
dar1 = _DaR(returns)

# Calculate with 7 % significance parameter
dar2 = _DaR(returns, 0.07)
```
"""
function _DaR(x::AbstractVector, alpha::Real = 0.05)
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
    _MDD(x::AbstractVector)

# Description

Compute the Maximum Drawdown (Calmar ratio) of uncompounded cumulative returns.

```math
\\begin{align}
\\mathrm{MDD_{a}}(\\bm{X}) &= \\max\\mathrm{DD_{a}}(\\bm{X})\\,.
\\end{align}
```

Where:

  - ``\\mathrm{DD_{a}}(\\bm{X})`` is the Drawdown of uncompounded cumulative returns as defined in [`_DaR`](@ref).

See also: [`MDD`](@ref), [`calc_risk(::MDD, ::AbstractVector)`](@ref), [`MDD_r`](@ref), [`calc_risk(::MDD_r, ::AbstractVector)`](@ref).

# Inputs

  - `x::AbstractVector`: `T×1` returns vector.

# Outputs

  - `mdd::Real`: Maximum Drawdown at Risk of uncompounded cumulative returns.

# Examples

```@example
# Sample returns vector
returns = [0.05, -0.03, 0.02, -0.01, 0.04]

# Calculate maximum drawdown of uncompounded cumulative returns
mdd = _MDD(returns)
```
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
    _ADD(x::AbstractVector)

# Description

Compute the Average Drawdown of uncompounded cumulative returns.

```math
\\begin{align}
\\mathrm{ADD_{a}}(\\bm{X}) &= \\dfrac{1}{T} \\sum\\limits_{j=1}^{T} \\mathrm{DD_{a}}(\\bm{X}, j)\\,.
\\end{align}
```

Where:

  - ``\\mathrm{DD_{a}}(\\bm{X}, j)`` is the Drawdown of uncompounded cumulative returns at time ``j`` as defined in [`_DaR`](@ref).

See also: [`ADD`](@ref), [`calc_risk(::ADD, ::AbstractVector)`](@ref), [`ADD_r`](@ref), [`calc_risk(::ADD_r, ::AbstractVector)`](@ref).

# Inputs

  - `x::AbstractVector`: `T×1` returns vector.

# Outputs

  - `add::Real`: Average Drawdown at Risk of uncompounded cumulative returns.

# Examples

```@example
# Sample returns vector
returns = [0.05, -0.03, 0.02, -0.01, 0.04]

# Calculate average drawdown of uncompounded cumulative returns
add = _ADD(returns)
```
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
    _CDaR(x::AbstractVector, α::Real = 0.05)

# Description

Compute the Conditional Drawdown at Risk of uncompounded cumulative returns.

```math
\\begin{align}
\\mathrm{CDaR_{a}}(\\bm{X},\\, \\alpha) &= \\mathrm{DaR_{a}}(\\bm{X},\\, \\alpha) + \\dfrac{1}{\\alpha T} \\sum\\limits_{j=0}^{T} \\max\\left(\\mathrm{DD_{a}}(\\bm{X},\\, j) - \\mathrm{DaR_{a}}(\\bm{X},\\, \\alpha),\\, 0 \\right)\\,.
\\end{align}
```

Where:

  - ``\\mathrm{DD_{a}}(\\bm{X},\\, j)`` is the Drawdown of uncompounded cumulative returns at time ``j`` as defined in [`_DaR`](@ref).
  - ``\\mathrm{DaR_{a}}(\\bm{X},\\, \\alpha)`` is the Drawdown at Risk of uncompounded cumulative returns as defined in [`_DaR`](@ref).

See also: [`CDaR`](@ref), [`calc_risk(::CDaR, ::AbstractVector)`](@ref), [`_DaR`](@ref), [`calc_risk(::DaR, ::AbstractVector)`](@ref), [`CDaR_r`](@ref), [`calc_risk(::CDaR_r, ::AbstractVector)`](@ref), [`_DaR_r`](@ref), [`calc_risk(::DaR_r, ::AbstractVector)`](@ref).

# Inputs

  - `x::AbstractVector`: `T×1` returns vector.
  - `α::Real = 0.05`: significance level, `α ∈ (0, 1)`.

# Behaviour

!!! warning

      - `α` is not validated because this is an internal function. It should have been validated by [`CDaR`](@ref).

# Outputs

  - `cdar::Real`: Conditional Drawdown at Risk of uncompounded cumulative returns..

# Examples

```@example
# Sample returns vector
returns = [0.05, -0.03, 0.02, -0.01, 0.04]

# Calculate the conditional drawdown at risk of 
# uncompounded cumulative returns with default parameters
cdar1 = _CDaR(returns)

# Calculate with 7 % significance parameter
cdar2 = _CDaR(returns, 0.07)
```
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
    _UCI(x::AbstractVector)

# Description

Compute the Ulcer Index of uncompounded cumulative returns.

```math
\\begin{align}
\\mathrm{UCI_{a}}(\\bm{X}) &= \\left(\\dfrac{1}{T} \\sum\\limits_{j=0}^{T} \\mathrm{DD_{a}}(\\bm{X},\\, j)^{2}\\right)^{1/2}\\,.
\\end{align}
```

Where:

  - ``\\mathrm{DD_{a}}(\\bm{X},\\, j)`` is the Drawdown of uncompounded cumulative returns at time ``j`` as defined in [`_DaR`](@ref).

See also: [`UCI`](@ref), [`calc_risk(::UCI, ::AbstractVector)`](@ref), [`UCI_r`](@ref), [`calc_risk(::UCI_r, ::AbstractVector)`](@ref).

# Inputs

  - `x::AbstractVector`: `T×1` returns vector.

# Outputs

  - `uci::Real`: Ulcer Index of uncompounded cumulative returns.

# Examples

```@example
# Sample returns vector
returns = [0.05, -0.03, 0.02, -0.01, 0.04]

# Calculate ulcer index of uncompounded cumulative returns
uci = _UCI(returns)
```
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
    _EDaR(x::AbstractVector, solvers::AbstractDict, α::Real = 0.05)

# Description

Compute the Entropic Drawdown at Risk of uncompounded cumulative returns.

```math
\\begin{align}
\\mathrm{EDaR_{a}}(\\bm{X},\\alpha) &= \\underset{z > 0}{\\inf} \\left\\{\\mathrm{ERM}(\\mathrm{DD_{a}}(\\bm{X}),\\, z, \\,\\alpha)\\right\\}\\,.
\\end{align}
```

Where:

  - ``\\mathrm{ERM}(\\mathrm{DD_{a}}(\\bm{X}),\\, z, \\,\\alpha)`` is the Entropic Risk Measure as defined in [`ERM`](@ref), using the Drawdown of uncompounded cumulative returns as defined in [`_DaR`](@ref).

See also: [`ERM`](@ref), [`calc_risk(::EDaR, ::AbstractVector)`](@ref), [`_EDaR`](@ref), [`EDaR_r`](@ref), [`calc_risk(::EDaR_r, ::AbstractVector)`](@ref), [`_EDaR_r`](@ref).

# Inputs

  - `x::AbstractVector`: `T×1` returns vector.
  - `solvers::AbstractDict`: JuMP-compatible solvers for exponential cone problems.
  - `α::Real = 0.05`: significance level, `α ∈ (0, 1)`.

# Behaviour

  - If no valid solution is found returns `NaN`.

!!! warning

      - `α` is not validated because this is an internal function. It should have been validated by [`EDaR`](@ref).

# Outputs

  - `edar::Real`: Entropic Drawdown at Risk of uncompounded cumulative returns.

# Examples

```@example
# Sample returns vector
returns = [0.05, -0.03, 0.02, -0.01, 0.04]

# Calculate the entropic drawdown at risk of 
# uncompounded cumulative returns with default parameters
edar1 = _EDaR(returns, Dict("solver" => my_solver))

# Calculate with a 3 % significance parameter
edar2 = _EDaR(returns, Dict("solver" => my_solver), 0.03)
```
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
    _RLDaR(x::AbstractVector, solvers::AbstractDict, α::Real = 0.05, κ::Real = 0.3)

# Description

Compute the Relativistic Drawdown at Risk of uncompounded cumulative returns.

```math
\\begin{align}
\\mathrm{RLDaR_{a}}(\\bm{X},\\, \\alpha,\\, \\kappa) &= \\mathrm{RRM}(\\mathrm{DD_{a}}(\\bm{X}),\\, \\alpha,\\, \\kappa)\\,.
\\end{align}
```

Where:

  - ``\\mathrm{RRM}(\\mathrm{DD_{a}}(\\bm{X}),\\, \\alpha,\\, \\kappa)`` is the Relativistic Risk Measure as defined in [`RRM`](@ref), using the Drawdown of uncompounded cumulative returns as defined in [`_DaR`](@ref).

See also: [`RRM`](@ref), [`calc_risk(::RLDaR, ::AbstractVector)`](@ref), [`_RLDaR`](@ref), [`RLDaR_r`](@ref), [`calc_risk(::RLDaR_r, ::AbstractVector)`](@ref), [`_RLDaR_r`](@ref).

# Inputs

  - `x::AbstractVector`: `T×1` returns vector.
  - `solvers::AbstractDict`: JuMP-compatible solvers for exponential cone problems.
  - `α::Real = 0.05`: significance level, `α ∈ (0, 1)`.
  - `κ::Real = 0.3`: significance level, `κ ∈ (0, 1)`.

# Behaviour

  - If no valid solution is found returns `NaN`.

!!! warning

      - `α` and `κ` are not validated because this is an internal function. They should have been validated by [`RLDaR`](@ref).

# Outputs

  - `rldar::Real`: Relativistic Drawdown at Risk of uncompounded cumulative returns.

# Examples

```@example
# Sample returns vector
returns = [0.05, -0.03, 0.02, -0.01, 0.04]

# Calculate the entropic drawdown at risk of 
# uncompounded cumulative returns with default parameters
rldar1 = _RLDaR(returns, Dict("solver" => my_solver))

# Calculate with a 3 % significance parameter and 75 % deformation parameter
rldar2 = _RLDaR(returns, Dict("solver" => my_solver), 0.03, 0.75)
```
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
_DaR_r(x::AbstractVector, α::Real = 0.05)

# Description

Compute the Drawdown at Risk of compounded cumulative returns.

```math
\\begin{align}
\\mathrm{DaR_{r}}(\\bm{X},\\, \\alpha) &= \\underset{j \\in (0,\\, T)}{\\max} \\left\\{ \\mathrm{DD_{r}}(\\bm{X},\\, j) \\in \\mathbb{R} \\, | \\, F_{\\mathrm{DD}}\\left(\\mathrm{DD_{r}}(\\bm{X},\\, j)\\right) < 1 - \\alpha \\right\\}\\\\
\\mathrm{DD_{r}}(\\bm{X},\\, j) &= \\underset{t \\in (0,\\, j)}{\\max}\\left( \\prod\\limits_{i=0}^{t} \\left(1+X_{i}\\right) \\right) - \\prod\\limits_{i=0}^{j} \\left(1+X_{i}\\right)\\\\
\\mathrm{DD_{r}}(\\bm{X}) &= \\left\\{j \\in (0,\\,T) \\, | \\, \\mathrm{DD_{r}}(\\bm{X},\\, j)\\right\\}\\,.
\\end{align}
```

Where:

  - ``\\mathrm{DD_{r}}(\\bm{X})`` is the Drawdown of compounded cumulative returns.
  - ``\\mathrm{DD_{r}}(\\bm{X},\\, j)`` is the Drawdown of compounded cumulative returns at time ``j``.
  - ``\\mathrm{DaR_{r}}(\\bm{X},\\, \\alpha)`` the Drawdown at Risk of compounded cumulative returns.

See also: [`DaR`](@ref), [`calc_risk(::DaR, ::AbstractVector)`](@ref), [`DaR_r`](@ref), [`calc_risk(::DaR_r, ::AbstractVector)`](@ref), [`_CDaR`](@ref), [`calc_risk(::CDaR, ::AbstractVector)`](@ref), [`_CDaR_r`](@ref), [`calc_risk(::CDaR_r, ::AbstractVector)`](@ref).

# Inputs

  - `x::AbstractVector`: `T×1` returns vector.
  - `α::Real = 0.05`: significance level, `α ∈ (0, 1)`.

# Behaviour

!!! warning

      - `α` is not validated because this is an internal function. It should have been validated by [`DaR`](@ref).

# Outputs

  - `dar_r::Real`: Drawdown at Risk of compounded cumulative returns.

# Examples

```@example
# Sample returns vector
returns = [0.05, -0.03, 0.02, -0.01, 0.04]

# Calculate the drawdown of compounded cumulative returns with default parameters
dar_r1 = _DaR_r(returns)

# Calculate with 7 % significance parameter
dar_r2 = _DaR_r(returns, 0.07)
```
"""
function _DaR_r(x::AbstractVector, alpha::Real = 0.05)
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
    _MDD_r(x::AbstractVector)

# Description

Compute the Maximum Drawdown of compounded cumulative returns.

```math
\\begin{align}
\\mathrm{MDD_{r}}(\\bm{X}) &= \\max \\mathrm{DD_{r}}(\\bm{X})\\,.
\\end{align}
```

Where:

  - ``\\mathrm{DD_{r}}(\\bm{X})`` is the Drawdown of compounded cumulative returns as defined in [`_DaR_r`](@ref).

See also: [`MDD`](@ref), [`calc_risk(::MDD, ::AbstractVector)`](@ref), [`MDD_r`](@ref), [`calc_risk(::MDD_r, ::AbstractVector)`](@ref).

# Inputs

  - `x::AbstractVector`: `T×1` returns vector.

# Outputs

  - `mdd_r::Real`: Maximum Drawdown at Risk of compounded cumulative returns.

# Examples

```@example
# Sample returns vector
returns = [0.05, -0.03, 0.02, -0.01, 0.04]

# Calculate maximum drawdown of compounded cumulative returns
mdd_r = _MDD_r(returns)
```
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
    _ADD_r(x::AbstractVector)

# Description

Compute the Average Drawdown of compounded cumulative returns.

```math
\\begin{align}
\\mathrm{ADD_{r}}(\\bm{X}) &= \\dfrac{1}{T} \\sum\\limits_{j=0}^{T} \\mathrm{DD_{r}}(\\bm{X},\\, j)\\,.
\\end{align}
```

Where:

  - ``\\mathrm{DD_{r}}(\\bm{X},\\, j)`` is the Drawdown of compounded cumulative returns at time ``j`` as defined in [`_DaR_r`](@ref).

See also: [`ADD`](@ref), [`calc_risk(::ADD, ::AbstractVector)`](@ref), [`ADD_r`](@ref), [`calc_risk(::ADD_r, ::AbstractVector)`](@ref).

# Inputs

  - `x::AbstractVector`: `T×1` returns vector.

# Outputs

  - `add_r::Real`: Average Drawdown at Risk of compounded cumulative returns.

# Examples

```@example
# Sample returns vector
returns = [0.05, -0.03, 0.02, -0.01, 0.04]

# Calculate average drawdown of compounded cumulative returns
add = _ADD_r(returns)
```
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
    _CDaR_r(x::AbstractVector, α::Real = 0.05)

# Description

Compute the Conditional Drawdown at Risk of compounded cumulative returns.

```math
\\begin{align}
\\mathrm{CDaR_{r}}(\\bm{X},\\, \\alpha) &= \\mathrm{DaR_{r}}(\\bm{X},\\, \\alpha) + \\dfrac{1}{\\alpha T} \\sum\\limits_{j=0}^{T} \\max\\left(\\mathrm{DD_{r}}(\\bm{X},\\, j) - \\mathrm{DaR_{r}}(\\bm{X},\\, \\alpha),\\, 0 \\right)\\,.
\\end{align}
```

Where:

  - ``\\mathrm{DD_{r}}(\\bm{X},\\, j)`` is the Drawdown of compounded cumulative returns at time ``j`` as defined in [`_DaR_r`](@ref).
  - ``\\mathrm{DaR_{r}}(\\bm{X},\\, \\alpha)`` the Drawdown at Risk of compounded cumulative returns as defined in [`_DaR_r`](@ref).

See also: [`CDaR`](@ref), [`calc_risk(::CDaR, ::AbstractVector)`](@ref), [`_DaR`](@ref), [`calc_risk(::DaR, ::AbstractVector)`](@ref), [`CDaR_r`](@ref), [`calc_risk(::CDaR_r, ::AbstractVector)`](@ref), [`_DaR_r`](@ref), [`calc_risk(::DaR_r, ::AbstractVector)`](@ref).

# Inputs

  - `x::AbstractVector`: `T×1` returns vector.
  - `α::Real = 0.05`: significance level, `α ∈ (0, 1)`.

# Behaviour

!!! warning

      - `α` is not validated because this is an internal function. It should have been validated by [`CDaR_r`](@ref).

# Outputs

  - `cdar_r::Real`: Conditional Drawdown at Risk of compounded cumulative returns..

# Examples

```@example
# Sample returns vector
returns = [0.05, -0.03, 0.02, -0.01, 0.04]

# Calculate the conditional drawdown at risk of 
# compounded cumulative returns with default parameters
cdar_r1 = _CDaR_r(returns)

# Calculate with 7 % significance parameter
cdar_r2 = _CDaR_r(returns, 0.07)
```
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
    _UCI_r(x::AbstractVector)

# Description

Compute the Ulcer Index of compounded cumulative returns.

```math
\\begin{align}
\\mathrm{UCI_{r}}(\\bm{X}) &= \\left(\\dfrac{1}{T} \\sum\\limits_{j=0}^{T} \\mathrm{DD_{r}}(\\bm{X},\\, j)^{2}\\right)^{1/2}\\,.
\\end{align}
```

Where:

  - ``\\mathrm{DD_{r}}(\\bm{X},\\, j)`` is the Drawdown of compounded cumulative returns at time ``j`` as defined in [`_DaR_r`](@ref).

See also: [`UCI`](@ref), [`calc_risk(::UCI, ::AbstractVector)`](@ref), [`UCI_r`](@ref), [`calc_risk(::UCI_r, ::AbstractVector)`](@ref).

# Inputs

  - `x::AbstractVector`: `T×1` returns vector.

# Outputs

  - `uci_r::Real`: Ulcer Index of compounded cumulative returns.

# Examples

```@example
# Sample returns vector
returns = [0.05, -0.03, 0.02, -0.01, 0.04]

# Calculate ulcer index of compounded cumulative returns
uci_r = _UCI_r(returns)
```
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
    _EDaR_r(x::AbstractVector, solvers::AbstractDict, α::Real = 0.05)

# Description

Compute the Entropic Drawdown at Risk of compounded cumulative returns.

```math
\\begin{align}
\\mathrm{EDaR_{r}}(\\bm{X},\\alpha) &= \\underset{z > 0}{\\inf} \\left\\{\\mathrm{ERM}(\\mathrm{DD_{r}}(\\bm{X}),\\, z, \\,\\alpha)\\right\\}\\,.
\\end{align}
```

Where:

  - ``\\mathrm{ERM}(\\mathrm{DD_{r}}(\\bm{X}),\\, z, \\,\\alpha)`` is the Entropic Risk Measure as defined in [`ERM`](@ref), using the Drawdown of compounded cumulative returns as defined in [`_DaR_r`](@ref).

See also: [`ERM`](@ref), [`calc_risk(::EDaR, ::AbstractVector)`](@ref), [`_EDaR`](@ref), [`EDaR_r`](@ref), [`calc_risk(::EDaR_r, ::AbstractVector)`](@ref), [`_EDaR_r`](@ref).

# Inputs

  - `x::AbstractVector`: `T×1` returns vector.
  - `solvers::AbstractDict`: JuMP-compatible solvers for exponential cone problems.
  - `α::Real = 0.05`: significance level, `α ∈ (0, 1)`.

# Behaviour

  - If no valid solution is found returns `NaN`.

!!! warning

      - `α` is not validated because this is an internal function. It should have been validated by [`EDaR_r`](@ref).

# Outputs

  - `edar_r::Real`: Entropic Drawdown at Risk of compounded cumulative returns.

# Examples

```@example
# Sample returns vector
returns = [0.05, -0.03, 0.02, -0.01, 0.04]

# Calculate the entropic drawdown at risk of 
# compounded cumulative returns with default parameters
edar_r1 = _EDaR_r(returns, Dict("solver" => my_solver))

# Calculate with a 3 % significance parameter
edar_r2 = _EDaR_r(returns, Dict("solver" => my_solver), 0.03)
```
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
    _RLDaR_r(x::AbstractVector, solvers::AbstractDict, α::Real = 0.05, κ::Real = 0.3)

# Description

Compute the Relativistic Drawdown at Risk of compounded cumulative returns.

```math
\\begin{align}
\\mathrm{RLDaR_{r}}(\\bm{X},\\, \\alpha,\\, \\kappa) &= \\mathrm{RRM}(\\mathrm{DD_{r}}(\\bm{X}),\\, \\alpha,\\, \\kappa)\\,.
\\end{align}
```

Where:

  - ``\\mathrm{RRM}(\\mathrm{DD_{r}}(\\bm{X}),\\, \\alpha,\\, \\kappa)`` is the Relativistic Risk Measure as defined in [`RRM`](@ref), using the Drawdown of compounded cumulative returns as defined in [`_DaR_r`](@ref).

See also: [`RRM`](@ref), [`calc_risk(::RLDaR, ::AbstractVector)`](@ref), [`_RLDaR`](@ref), [`RLDaR_r`](@ref), [`calc_risk(::RLDaR_r, ::AbstractVector)`](@ref), [`_RLDaR_r`](@ref).

# Inputs

  - `x::AbstractVector`: `T×1` returns vector.
  - `solvers::AbstractDict`: JuMP-compatible solvers for exponential cone problems.
  - `α::Real = 0.05`: significance level, `α ∈ (0, 1)`.
  - `κ::Real = 0.3`: significance level, `κ ∈ (0, 1)`.

# Behaviour

  - If no valid solution is found returns `NaN`.

!!! warning

      - `α` and `κ` are not validated because this is an internal function. They should have been validated by [`RLDaR`](@ref).

# Outputs

  - `rldar_r::Real`: Relativistic Drawdown at Risk of compounded cumulative returns.

# Examples

```@example
# Sample returns vector
returns = [0.05, -0.03, 0.02, -0.01, 0.04]

# Calculate the entropic drawdown at risk of 
# compounded cumulative returns with default parameters
rldar_r1 = _RLDaR_r(returns, Dict("solver" => my_solver))

# Calculate with a 3 % significance parameter and 75 % deformation parameter
rldar_r2 = _RLDaR_r(returns, Dict("solver" => my_solver), 0.03, 0.75)
```
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
    _Kurt(x::AbstractVector, w::Union{AbstractWeights, Nothing} = nothing; scale::Bool = false)

# Description

Compute the Square Root Kurtosis.

```math
\\begin{align}
\\mathrm{Kurt}(\\bm{X}) &= \\left(\\dfrac{1}{T} \\sum\\limits_{t=1}^{T} \\left( X_{t} - \\mathbb{E}(\\bm{X}) \\right)^{4} \\right)^{1/2}\\,.
\\end{align}
```

See also: [`Kurt`](@ref), [`calc_risk(::Kurt, ::AbstractWeights)`](@ref), [`risk_contribution`](@ref).

# Inputs

## Positional

  - `x::AbstractVector`: `T×1` returns vector.
  - `w::Union{AbstractWeights, Nothing} = nothing`: `T×1` optional vector of weights for computing the expected return.

## Named

  - `scale::Bool = false`: flag for scaling risk in [`risk_contribution`](@ref).

# Behaviour

  - If `w` is `nothing`: uses simple arithmetic mean.
  - If `w` is provided: uses weighted mean for calculating deviations.
  - If `scale` is `false`: no effect.
  - If `scale` is `true`: divide by 2, used in [`risk_contribution`](@ref).

# Outputs

  - `kurt::Real`: square root kurtosis.

# Examples

```@example
# Sample returns vector
returns = [0.05, -0.03, 0.02, -0.01, 0.04]

# Calculate the square root kurtosis
kurt1 = _Kurt(returns)

# Calculate the square root kurtosis using weights
kurt2 = _Kurt(returns, eweights(1:length(returns), 0.3))

# Calculate the square root kurtosis using weights,
# for use in risk_contribution
kurt3 = _Kurt(returns, eweights(1:length(returns), 0.1); scale = true)
```
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
    _SKurt(x::AbstractVector, r::Real = 0.0, w::Union{AbstractWeights, Nothing} = nothing; scale::Bool = false)

# Description

Compute the Square Root Semi Kurtosis.

```math
\\begin{align}
\\mathrm{SKurt}(\\bm{X}) &= \\left(\\dfrac{1}{T} \\sum\\limits_{t=1}^{T} \\min\\left( X_{t} - \\mathbb{E}(\\bm{X}),\\, r \\right)^{4} \\right)^{1/2}\\,.
\\end{align}
```

See also: [`SKurt`](@ref), [`calc_risk(::SKurt, ::AbstractWeights)`](@ref), [`risk_contribution`](@ref).

# Inputs

## Positional

  - `x::AbstractVector`: `T×1` returns vector.
  - `r::Real = 0.0`: minimum return threshold for downside classification.
  - `w::Union{AbstractWeights, Nothing} = nothing`: `T×1` optional vector of weights for computing the expected return.

## Named

  - `scale::Bool = false`: flag for scaling risk in [`risk_contribution`](@ref).

# Behaviour

  - If `w` is `nothing`: uses simple arithmetic mean.
  - If `w` is provided: uses weighted mean for calculating deviations.
  - If `scale` is `false`: no effect.
  - If `scale` is `true`: divide by 2, used in [`risk_contribution`](@ref).

# Outputs

  - `skurt::Real`: square root semi kurtosis.

# Examples

```@example
# Sample returns vector
returns = [0.05, -0.03, 0.02, -0.01, 0.04]

# Calculate the square root kurtosis
skurt1 = _SKurt(returns)

# Calculate the square root semi kurtosis using weights
skurt2 = _SKurt(returns, eweights(1:length(returns), 0.3))

# Calculate the square root semi kurtosis using weights,
# for use in risk_contribution
skurt3 = _SKurt(returns, eweights(1:length(returns), 0.1); scale = true)
```
"""
function _SKurt(x::AbstractVector, target::Real = 0.0,
                w::Union{AbstractWeights, Nothing} = nothing; scale::Bool = false)
    T = length(x)
    mu = isnothing(w) ? mean(x) : mean(x, w)
    val = x .- mu
    skurt = sqrt(sum(val[val .<= target] .^ 4) / T)
    return !scale ? skurt : skurt / 2
end

"""
    _Skew(w::AbstractVector, V::AbstractMatrix)

# Description

Compute the Quadratic Skewness/Semi Skewness.

```math
\\begin{align}
\\nu &= \\bm{w}^{\\intercal} \\mathbf{V} \\bm{w}\\\\
\\end{align}
```

Where:

  - ``\\bm{w}`` is the vector of asset weights.
  - ``\\mathbf{V}`` is the sum of the symmetric negative spectral slices of the coskewness or semicoskewness.

See also: [`Skew`](@ref), [`calc_risk(::Skew, ::AbstractWeights)`](@ref), [`SSkew`](@ref), [`calc_risk(::SSkew, ::AbstractWeights)`](@ref).

# Inputs

  - `w::AbstractVector`: `N×1` vector of weights.
  - `V::AbstractMatrix`: `N×N` matrix of the sum of negative spectral slices of the coskewness or semi coskewness.

# Outputs

  - `skew::Real`: quadradic skew or quadratic semi skew.

# Examples

```@example
# Number of assets
N = 3

# Create sample sum of negative spectral 
# slices of the coskewness
V = [0.04 0.02 0.01;
     0.02 0.09 0.03;
     0.01 0.03 0.06]

# Create weight vector
w = [0.3, 0.4, 0.3]

# Calculate portfolio skew
skew = _Skew(w, V)
```
"""
function _Skew(w::AbstractVector, V::AbstractMatrix)
    return sqrt(dot(w, V, w))
end

"""
    _GMD(x::AbstractVector)

# Description

Compute the Gini Mean Difference.

See also: [`GMD`](@ref), [`calc_risk(::GMD, ::AbstractVector)`](@ref).

# Inputs

  - `x::AbstractVector`: `T×1` returns vector.

# Behaviour

!!! warning

      - In-place sorts the input vector.

# Outputs

  - `gmd::Real`: Gini Mean Difference.

# Examples

```@example
# Sample returns vector
returns = [0.05, -0.03, 0.02, -0.01, 0.04]

# Calculate the gini mean difference
gmd = _GMD(returns)
```
"""
function _GMD(x::AbstractVector)
    T = length(x)
    w = owa_gmd(T)
    return dot(w, sort!(x))
end

"""
    _RG(x::AbstractVector)

# Description

Compute the Range.

See also: [`RG`](@ref), [`calc_risk(::RG, ::AbstractVector)`](@ref).

# Inputs

  - `x::AbstractVector`: `T×1` returns vector.

# Behaviour

!!! warning

      - In-place sorts the input vector.

# Outputs

  - `rg::Real`: Range.

# Examples

```@example
# Sample returns vector
returns = [0.05, -0.03, 0.02, -0.01, 0.04]

# Calculate the gini mean difference
rg = _RG(returns)
```
"""
function _RG(x::AbstractVector)
    T = length(x)
    w = owa_rg(T)
    return dot(w, sort!(x))
end

"""
    _CVaRRG(x::AbstractVector; alpha::Real = 0.05, beta::Real = 0.05)

# Description

Compute the Conditional Value at Risk Range.

See also: [`CVaRRG`](@ref), [`calc_risk(::CVaRRG, ::AbstractVector)`](@ref).

# Inputs

## Positional

  - `x::AbstractVector`: `T×1` returns vector.

## Named

  - `alpha::Real = 0.05`: significance level of losses, `alpha ∈ (0, 1)`.
  - `beta::Real = 0.05`: significance level of gains, `beta ∈ (0, 1)`.

# Behaviour

!!! warning

      - In-place sorts the input vector.
      - `alpha` and `beta` are not validated because this is an internal function. They should have been validated by [`CVaRRG`](@ref).

# Outputs

  - `cvarrg::Real`: Conditional Value at Risk Range.

# Examples

```@example
# Sample returns vector
returns = [0.05, -0.03, 0.02, -0.01, 0.04]

# Calculate the conditional value at risk range with default parameters
cvarrg1 = _CVaRRG(returns)

# Calculate with 7 % significance parameter of losses and
# 12 % significance parameter of gains.
cvarrg2 = _CVaRRG(returns; alpha = 0.07, beta = 0.12)
```
"""
function _CVaRRG(x::AbstractVector; alpha::Real = 0.05, beta::Real = 0.05)
    T = length(x)
    w = owa_rcvar(T; alpha = alpha, beta = beta)
    return dot(w, sort!(x))
end

"""
    _TG(x::AbstractVector; alpha_i::Real = 0.0001, alpha::Real = 0.05, a_sim::Integer = 100)

# Description

Compute the Tail Gini.

See also: [`TG`](@ref), [`calc_risk(::TG, ::AbstractVector)`](@ref).

# Inputs

## Positional

  - `x::AbstractVector`: `T×1` returns vector.

## Named

  - `alpha_i::Real = 0.0001`: start value of the significance level of CVaR losses, `0 < alpha_i < alpha < 1`.
  - `alpha::Real = 0.05`: significance level of losses, `alpha ∈ (0, 1)`.
  - `a_sim::Integer = 100`: number of CVaRs to approximate the Tail Gini losses, `a_sim > 0`.

# Behaviour

!!! warning

      - In-place sorts the input vector.
      - `alpha_i`, `alpha`, and `a_sim` are not validated because this is an internal function. They should have been validated by [`TG`](@ref).

# Outputs

  - `tg::Real`: Tail Gini.

# Examples

```@example
# Sample returns vector
returns = [0.05, -0.03, 0.02, -0.01, 0.04]

# Calculate the tail gini with default parameters
tg1 = _TG(returns)

# Calculate with custom parameters
tg2 = _TG(returns; alpha_i = 0.003, alpha = 0.07, a_sim = 131)
```
"""
function _TG(x::AbstractVector; alpha_i::Real = 0.0001, alpha::Real = 0.05,
             a_sim::Integer = 100)
    T = length(x)
    w = owa_tg(T; alpha_i = alpha_i, alpha = alpha, a_sim = a_sim)
    return dot(w, sort!(x))
end

"""
    _TGRG(x::AbstractVector; 
          alpha_i::Real = 0.0001, alpha::Real = 0.05, a_sim::Integer = 100,
          beta_i::Real = 0.0001, beta::Real = 0.05, b_sim::Integer = 100)

# Description

Compute the Tail Gini Range.

See also: [`TG`](@ref), [`calc_risk(::TG, ::AbstractVector)`](@ref).

# Inputs

## Positional

  - `x::AbstractVector`: `T×1` returns vector.

## Named

  - `alpha_i::Real = 0.0001`: start value of the significance level of CVaR losses, `0 < alpha_i < alpha < 1`.
  - `alpha::Real = 0.05`: significance level of losses, `alpha ∈ (0, 1)`.
  - `a_sim::Integer = 100`: number of CVaRs to approximate the Tail Gini losses, `a_sim > 0`.
  - `beta_i::Real = 0.0001`: start value of the significance level of CVaR gains, `0 < beta_i < beta < 1`.
  - `beta::Real = 0.05`: end value of the significance level of CVaR gains, `beta ∈ (0, 1)`.
  - `b_sim = 100`: number of CVaRs to approximate the Tail Gini gains, `b_sim > 0`.

# Behaviour

!!! warning

      - In-place sorts the input vector.
      - `alpha_i`, `alpha`, `a_sim`, `beta_i`, `beta`, and `b_sim` are not validated because this is an internal function. They should have been validated by [`TGRG`](@ref).

# Outputs

  - `tgrg::Real`: Tail Gini Range.

# Examples

```@example
# Sample returns vector
returns = [0.05, -0.03, 0.02, -0.01, 0.04]

# Calculate the tail gini range with default parameters
tgrg1 = _TGRG(returns)

# Calculate with custom parameters
tgrg2 = _TGRG(returns; alpha_i = 0.003, alpha = 0.07, a_sim = 131, beta_i = 0.001,
              beta = 0.03, b_sim = 97)
```
"""
function _TGRG(x::AbstractVector; alpha_i::Real = 0.0001, alpha::Real = 0.05,
               a_sim::Integer = 100, beta_i::Real = 0.0001, beta::Real = 0.05,
               b_sim::Integer = 100)
    T = length(x)
    w = owa_rtg(T; alpha_i = alpha_i, alpha = alpha, a_sim = a_sim, beta_i = beta_i,
                beta = beta, b_sim = b_sim)
    return dot(w, sort!(x))
end

"""
    _OWA(x::AbstractVector, w::AbstractVector)

# Description

Compute the Ordered Weight Array risk measure.

See also: [`OWA`](@ref), [`calc_risk(::OWA, ::AbstractVector)`](@ref).

# Inputs

  - `x::AbstractVector`: `T×1` returns vector.
  - `w::AbstractVector`: `T×1` precomputed vector of OWA weights. Can be computed with [`owa_gmd`](@ref), [`owa_rg`](@ref), [`owa_rcvar`](@ref), [`owa_tg`](@ref), [`owa_rtg`](@ref), [`owa_l_moment`](@ref) or [`owa_l_moment_crm`](@ref).

# Behaviour

!!! warning

    In-place sorts the input vector.

# Outputs

  - `owa::Real`: Ordered Weight Array risk.

# Examples

```@example
# Sample returns vector
returns = [0.05, -0.03, 0.02, -0.01, 0.04]

w = owa_l_moment_crm(length(returns))

# Calculate the tail gini range with default parameters
owa = _OWA(returns, w)
```
"""
function _OWA(x::AbstractVector, w::AbstractVector)
    return dot(w, sort!(x))
end

"""
    _BDVariance(x::AbstractVector)

# Description

Compute the Brownian Distance variance.

```math
\\begin{align}
\\mathrm{BDVariance}(\\bm{X}) &= \\mathrm{BDCov}(\\bm{X},\\, \\bm{X}) =  \\dfrac{1}{T^{2}} \\sum\\limits_{i=1}^{T}\\sum\\limits_{j=1}^{T} A_{i,\\,j}^2\\\\
\\mathrm{BDCov}(\\bm{X},\\, \\bm{Y}) &= \\dfrac{1}{T^{2}} \\sum\\limits_{i=1}^{T} \\sum\\limits_{j=1}^{T} A_{i,\\,j} B_{i,\\,j}\\\\
A_{i,\\,j} &= a_{i,\\,j} - \\bar{a}_{i\\,.} - \\bar{a}_{.\\,j} + \\bar{a}_{.\\,.}\\\\
B_{i,\\,j} &= b_{i,\\,j} - \\bar{b}_{i\\,.} - \\bar{b}_{.\\,j} + \\bar{b}_{.\\,.}\\\\
a_{i,\\,j} &= \\lVert X_{i} - X_{j} \\rVert_{2}, \\quad \\forall i,\\, j = 1,\\, \\ldots ,\\, T\\\\
b_{i,\\,j} &= \\lVert Y_{i} - Y_{j} \\rVert_{2}, \\quad \\forall i,\\, j = 1,\\, \\ldots ,\\, T\\,.
\\end{align}
```

where:

  - ``\\bm{X}`` and ``\\bm{Y}`` are random variables, they are equal in this case as they are the portfolio returns.
  - ``a_{i,\\,j}`` and ``b_{i,\\,j}`` are entries of a distance matrix where ``i`` and ``j`` are points in time. Each entry is defined as the Euclidean distance ``\\lVert \\cdot \\rVert_{2}`` between the value of the random variable at time ``i`` and its value at time ``j``.
  - ``\\bar{a}_{i,\\,\\cdot}`` and ``\\bar{b}_{i,\\,\\cdot}`` are the ``i``-th row means of their respective matrices.
  - ``\\bar{a}_{\\cdot,\\,j}`` and ``\\bar{b}_{\\cdot,\\,j}`` are the ``j``-th column means of their respective matrices.
  - ``\\bar{a}_{\\cdot,\\,\\cdot}`` and ``\\bar{b}_{\\cdot,\\,\\cdot}`` are the grand means of their respective matrices.
  - ``A_{i,\\,j}`` and ``B_{i,\\,j}`` are doubly centered distances.

See also: [`BDVariance`](@ref), [`calc_risk(::BDVariance, ::AbstractVector)`](@ref).

# Inputs

  - `x::AbstractVector`: `T×1` returns vector.

# Outputs

  - `dvar::Real`: Brownian Distance Variance.

# Examples

```@example
# Sample returns vector
returns = [0.05, -0.03, 0.02, -0.01, 0.04]

# Calculate the gini mean difference
bdvar = _BDVariance(returns)
```
"""
function _BDVariance(x::AbstractVector)
    T = length(x)
    iT2 = inv(T^2)
    D = Matrix{eltype(x)}(undef, T, T)
    D .= x
    D .-= transpose(x)
    D .= abs.(D)
    return iT2 * (dot(D, D) + iT2 * sum(D)^2)
end

function _TCM(x::AbstractVector, w::Union{AbstractWeights, Nothing} = nothing)
    T = length(x)
    mu = isnothing(w) ? mean(x) : mean(x, w)
    val = x .- mu
    return sum(val .^ 3) / T
end

function _TLPM(x::AbstractVector, target::Real = 0.0,
               w::Union{AbstractWeights, Nothing} = nothing)
    T = length(x)
    target = if !isinf(target)
        target
    else
        isnothing(w) ? mean(x) : mean(x, w)
    end
    val = x .- target
    return -sum(val[val .<= zero(target)] .^ 3) / T
end

function _Skewness(x::AbstractVector, mean_w::Union{AbstractWeights, Nothing} = nothing,
                   ve::CovarianceEstimator = SimpleVariance(),
                   std_w::Union{AbstractWeights, Nothing} = nothing)
    T = length(x)
    mu = isnothing(mean_w) ? mean(x) : mean(x, mean_w)
    dev = isnothing(std_w) ? std(ve, x; mean = mu) : std(ve, x, std_w; mean = mu)
    val = x .- mu
    return sum(val .^ 3) / T / dev^3
end

function _SSkewness(x::AbstractVector, target::Real = 0.0,
                    mean_w::Union{AbstractWeights, Nothing} = nothing,
                    ce::CovarianceEstimator = SimpleVariance(),
                    std_w::Union{AbstractWeights, Nothing} = nothing)
    T = length(x)
    mu = isnothing(mean_w) ? mean(x) : mean(x, mean_w)
    dev = isnothing(std_w) ? std(ce, x; mean = mu) : std(ce, x, std_w; mean = mu)
    val = x .- mu
    return -sum(val[val .<= target] .^ 3) / T / dev^3
end

function _FTCM(x::AbstractVector, w::Union{AbstractWeights, Nothing} = nothing)
    T = length(x)
    mu = isnothing(w) ? mean(x) : mean(x, w)
    val = x .- mu
    return sum(val .^ 4) / T
end

function _FTLPM(x::AbstractVector, target::Real = 0.0,
                w::Union{AbstractWeights, Nothing} = nothing)
    T = length(x)
    target = if !isinf(target)
        target
    else
        isnothing(w) ? mean(x) : mean(x, w)
    end
    val = x .- target
    return sum(val[val .<= zero(target)] .^ 4) / T
end

function _Kurtosis(x::AbstractVector, mean_w::Union{AbstractWeights, Nothing} = nothing,
                   ve::CovarianceEstimator = SimpleVariance(),
                   std_w::Union{AbstractWeights, Nothing} = nothing)
    T = length(x)
    mu = isnothing(mean_w) ? mean(x) : mean(x, mean_w)
    dev = isnothing(std_w) ? std(ve, x; mean = mu) : std(ve, x, std_w; mean = mu)
    val = x .- mu
    return sum(val .^ 4) / T / dev^4
end

function _SKurtosis(x::AbstractVector, target::Real = 0.0,
                    mean_w::Union{AbstractWeights, Nothing} = nothing,
                    ce::CovarianceEstimator = SimpleVariance(),
                    std_w::Union{AbstractWeights, Nothing} = nothing)
    T = length(x)
    mu = isnothing(mean_w) ? mean(x) : mean(x, mean_w)
    dev = isnothing(std_w) ? std(ce, x; mean = mu) : std(ce, x, std_w; mean = mu)
    val = x .- mu
    return sum(val[val .<= target] .^ 4) / T / dev^4
end

function cumulative_returns(x::AbstractVector, compound::Bool = false)
    return !compound ? cumsum(x) : cumprod(one(eltype(x)) .+ x)
end

function drawdown(x::AbstractVector, compound::Bool = false)
    x = cumulative_returns(x, compound)
    return if !compound
        x ./ accumulate(max, x) .- one(eltype(x))
    else
        x .- accumulate(max, x)
    end
end

"""
    calc_risk(sd::SD, w::AbstractVector; kwargs...)

# Description

Compute the [`SD`](@ref) via [`_SD`](@ref).

See also: [`SD`](@ref), [`_SD`](@ref).

# Inputs

## Positional

  - `sd::SD`: risk measure.
  - `w::AbstractVector`: `N×1` vector of asset weights.

## Named

  - `kwargs`: used for other types.

# Outputs

  - `risk::Real`: standard deviation.

# Examples

```@example
# Number of assets
N = 3

# Create sample covariance matrix
Σ = [0.04 0.02 0.01;
     0.02 0.09 0.03;
     0.01 0.03 0.06]

# Create weight vector
w = [0.3, 0.4, 0.3]

# Create instance of risk measure.
sd_rm = SD(; sigma = Σ)

# Calculate portfolio standard deviation
risk = calc_risk(sd_rm, w)
```
"""
function calc_risk(sd::SD, w::AbstractVector; kwargs...)
    return _SD(w, sd.sigma)
end

"""
    calc_risk(variance::Variance, w::AbstractVector; kwargs...)

# Description

Compute the [`Variance`](@ref) via [`_Variance`](@ref).

See also: [`Variance`](@ref), [`_Variance`](@ref).

# Inputs

## Positional

  - `variance::Variance`: risk measure.
  - `w::AbstractVector`: `N×1` vector of asset weights.

## Named

  - `kwargs`: used for other types.

# Outputs

  - `risk::Real`: variance.

# Examples

```@example
# Number of assets
N = 3

# Create sample covariance matrix
Σ = [0.04 0.02 0.01;
     0.02 0.09 0.03;
     0.01 0.03 0.06]

# Create weight vector
w = [0.3, 0.4, 0.3]

# Create instance of risk measure.
variance_rm = Variance(; sigma = Σ)

# Calculate portfolio standard deviation
risk = calc_risk(variance_rm, w)
```
"""
function calc_risk(variance::Variance, w::AbstractVector; kwargs...)
    return _Variance(w, variance.sigma)
end

"""
    calc_risk(mad::MAD, w::AbstractVector; X::AbstractMatrix, kwargs...)

# Description

Compute the [`MAD`](@ref) via [`_MAD`](@ref).

See also: [`MAD`](@ref), [`_MAD`](@ref).

# Inputs

## Positional

  - `mad::MAD`: risk measure.
  - `w::AbstractVector`: `N×1` vector of asset weights.

## Named

  - `X::AbstractMatrix`: `T×N` matrix of asset returns.

# Outputs

  - `mad::Real`: mean absolute deviation.

# Examples

```@example
# Sample returns matrix
returns = [ 0.19 -0.41 -0.70;
            1.15 -1.20 -1.27;
           -0.27 -1.98 -0.77;
           -0.65  0.22  0.59;
           -0.04  0.35 -0.99]

# Sample weights vector
w = [0.3, 0.5, 0.2]

# Calculate the mean absolute deviation with default parameters
mad_rm1 = MAD()
mad_risk1 = calc_risk(mad_rm1, w; X = returns)

# Calculate using mean returns using weighted mean
mad_rm2 = MAD(; w = eweights(1:size(returns, 1), 0.3))
mad_risk2 = calc_risk(mad_rm2, w; X = returns)
```
"""
function calc_risk(mad::MAD, w::AbstractVector; X::AbstractMatrix, kwargs...)
    return _MAD(X * w, mad.w)
end

"""
    calc_risk(ssd::SSD, w::AbstractVector; X::AbstractMatrix, kwargs...)

# Description

Compute the [`SSD`](@ref) via [`_SSD`](@ref).

See also: [`SSD`](@ref), [`_SSD`](@ref).

# Inputs

## Positional

  - `ssd::SSD`: risk measure.
  - `w::AbstractVector`: `N×1` vector of asset weights.

## Named

  - `X::AbstractMatrix`: `T×N` matrix of asset returns.

# Outputs

  - `ssd::Real`: semi standard deviation.

# Examples

```@example
# Sample returns matrix
returns = [ 0.19 -0.41 -0.70;
            1.15 -1.20 -1.27;
           -0.27 -1.98 -0.77;
           -0.65  0.22  0.59;
           -0.04  0.35 -0.99]

# Sample weights vector
w = [0.3, 0.5, 0.2]

# Calculate the mean absolute deviation with default parameters
ssd_rm1 = SSD()
ssd_risk1 = calc_risk(ssd_rm1, w; X = returns)

# Calculate using mean returns using weighted mean and 1.5 % return target
ssd_rm2 = SSD(; w = eweights(1:size(returns, 1), 0.3), target = 0.015)
ssd_risk2 = calc_risk(ssd_rm2, w; X = returns)
```
"""
function calc_risk(ssd::SSD, w::AbstractVector; X::AbstractMatrix, kwargs...)
    return _SSD(X * w, ssd.target, ssd.w)
end

"""
    calc_risk(svariance::SVariance, w::AbstractVector; X::AbstractMatrix, kwargs...)

# Description

Compute the [`SVariance`](@ref) via [`_SVariance`](@ref).

See also: [`SVariance`](@ref), [`_SVariance`](@ref).

# Inputs

## Positional

  - `svariance::SVariance`: risk measure.
  - `w::AbstractVector`: `N×1` vector of asset weights.

## Named

  - `X::AbstractMatrix`: `T×N` matrix of asset returns.

# Outputs

  - `svariance::Real`: semi variance.

# Examples

```@example
# Sample returns matrix
returns = [ 0.19 -0.41 -0.70;
            1.15 -1.20 -1.27;
           -0.27 -1.98 -0.77;
           -0.65  0.22  0.59;
           -0.04  0.35 -0.99]

# Sample weights vector
w = [0.3, 0.5, 0.2]

# Calculate the mean absolute deviation with default parameters
svariance_rm1 = SSD()
svariance_risk1 = calc_risk(svariance_rm1, w; X = returns)

# Calculate using mean returns using weighted mean and 1.5 % return target
svariance_rm2 = SSD(; w = eweights(1:size(returns, 1), 0.3), target = 0.015)
svariance_risk2 = calc_risk(svariance_rm2, w; X = returns)
```
"""
function calc_risk(svariance::SVariance, w::AbstractVector; X::AbstractMatrix, kwargs...)
    return _SVariance(X * w, svariance.target, svariance.w)
end

"""
    calc_risk(flpm::FLPM, w::AbstractVector; X::AbstractMatrix, kwargs...)

# Description

Compute the [`FLPM`](@ref) via [`_FLPM`](@ref).

See also: [`FLPM`](@ref), [`_FLPM`](@ref).

# Inputs

## Positional

  - `flpm::FLPM`: risk measure.
  - `w::AbstractVector`: `N×1` vector of asset weights.

## Named

  - `X::AbstractMatrix`: `T×N` matrix of asset returns.

# Outputs

  - `flpm::Real`: first lower partial moment.

# Examples

```@example
# Sample returns matrix
returns = [ 0.19 -0.41 -0.70;
            1.15 -1.20 -1.27;
           -0.27 -1.98 -0.77;
           -0.65  0.22  0.59;
           -0.04  0.35 -0.99]

# Sample weights vector
w = [0.3, 0.5, 0.2]

# Calculate the mean absolute deviation with default parameters
flpm_rm1 = FLPM()
flpm_risk1 = calc_risk(flpm_rm1, w; X = returns)

# Calculate using mean returns using a 1.5 % return target
flpm_rm2 = FLPM(; target = 0.015)
flpm_risk2 = calc_risk(flpm_rm2, w; X = returns)
```
"""
function calc_risk(flpm::FLPM, w::AbstractVector; X::AbstractMatrix, kwargs...)
    return _FLPM(X * w, flpm.target, flpm.w)
end

"""
    calc_risk(slpm::SLPM, w::AbstractVector; X::AbstractMatrix, kwargs...)

# Description

Compute the [`SLPM`](@ref) via [`_SLPM`](@ref).

See also: [`SLPM`](@ref), [`_SLPM`](@ref).

# Inputs

## Positional

  - `slpm::SLPM`: risk measure.
  - `w::AbstractVector`: `N×1` vector of asset weights.

## Named

  - `X::AbstractMatrix`: `T×N` matrix of asset returns.

# Outputs

  - `slpm::Real`: second lower partial moment.

# Examples

```@example
# Sample returns matrix
returns = [ 0.19 -0.41 -0.70;
            1.15 -1.20 -1.27;
           -0.27 -1.98 -0.77;
           -0.65  0.22  0.59;
           -0.04  0.35 -0.99]

# Sample weights vector
w = [0.3, 0.5, 0.2]

# Calculate the mean absolute deviation with default parameters
slpm_rm1 = SLPM()
slpm_risk1 = calc_risk(slpm_rm1, w; X = returns)

# Calculate using mean returns using a 1.5 % return target
slpm_rm2 = SLPM(; target = 0.015)
slpm_risk2 = calc_risk(slpm_rm2, w; X = returns)
```
"""
function calc_risk(slpm::SLPM, w::AbstractVector; X::AbstractMatrix, kwargs...)
    return _SLPM(X * w, slpm.target, slpm.w)
end

"""
    calc_risk(::WR, w::AbstractVector; X::AbstractMatrix, kwargs...)

# Description

Compute the [`WR`](@ref) via [`_WR`](@ref).

See also: [`WR`](@ref), [`_WR`](@ref).

# Inputs

## Positional

  - `wr::WR`: risk measure.
  - `w::AbstractVector`: `N×1` vector of asset weights.

## Named

  - `X::AbstractMatrix`: `T×N` matrix of asset returns.

# Outputs

  - `wr::Real`: worst realisation.

# Examples

```@example
# Sample returns matrix
returns = [ 0.19 -0.41 -0.70;
            1.15 -1.20 -1.27;
           -0.27 -1.98 -0.77;
           -0.65  0.22  0.59;
           -0.04  0.35 -0.99]

# Sample weights vector
w = [0.3, 0.5, 0.2]

# Calculate the mean absolute deviation with default parameters
wr_rm = WR()
wr_risk = calc_risk(wr_rm, w; X = returns)
```
"""
function calc_risk(::WR, w::AbstractVector; X::AbstractMatrix, kwargs...)
    return _WR(X * w)
end

"""
    calc_risk(var::VaR, w::AbstractVector; X::AbstractMatrix, kwargs...)

# Description

Compute the [`VaR`](@ref) via [`_VaR`](@ref).

See also: [`VaR`](@ref), [`_VaR`](@ref).

# Inputs

## Positional

  - `var::VaR`: risk measure.
  - `w::AbstractVector`: `N×1` vector of asset weights.

## Named

  - `X::AbstractMatrix`: `T×N` matrix of asset returns.

# Outputs

  - `var::Real`: Value at Risk.

# Examples

```@example
# Sample returns matrix
returns = [ 0.19 -0.41 -0.70;
            1.15 -1.20 -1.27;
           -0.27 -1.98 -0.77;
           -0.65  0.22  0.59;
           -0.04  0.35 -0.99]

# Sample weights vector
w = [0.3, 0.5, 0.2]

# Calculate the value at risk with default parameters
var_rm1 = VaR()
var_risk1 = calc_risk(var_rm1, w; X = returns)

# Calculate the value at risk with 50 % significance parameter
var_rm2 = VaR(; alpha = 0.5)
var_risk2 = calc_risk(var_rm2, w; X = returns)
```
"""
function calc_risk(var::VaR, w::AbstractVector; X::AbstractMatrix, kwargs...)
    return _VaR(X * w, var.alpha)
end

"""
    calc_risk(cvar::Union{CVaR, DRCVaR}, w::AbstractVector; X::AbstractMatrix,
                   kwargs...)

# Description

Compute the [`CVaR`](@ref) via [`_CVaR`](@ref).

See also: [`CVaR`](@ref), [`_CVaR`](@ref).

# Inputs

## Positional

  - `cvar::CVaR`: risk measure.
  - `w::AbstractVector`: `N×1` vector of asset weights.

## Named

  - `X::AbstractMatrix`: `T×N` matrix of asset returns.

# Outputs

  - `cvar::Real`: Conditional Value at Risk.

# Examples

```@example
# Sample returns matrix
returns = [ 0.19 -0.41 -0.70;
            1.15 -1.20 -1.27;
           -0.27 -1.98 -0.77;
           -0.65  0.22  0.59;
           -0.04  0.35 -0.99]

# Sample weights vector
w = [0.3, 0.5, 0.2]

# Calculate the conditional value at risk with default parameters
cvar_rm1 = CVaR()
cvar_risk1 = calc_risk(cvar_rm1, w; X = returns)

# Calculate the conditional value at risk with 50 % significance parameter
cvar_rm2 = CVaR(; alpha = 0.5)
cvar_risk2 = calc_risk(cvar_rm2, w; X = returns)
```
"""
function calc_risk(cvar::Union{CVaR, DRCVaR}, w::AbstractVector; X::AbstractMatrix,
                   kwargs...)
    return _CVaR(X * w, cvar.alpha)
end

"""
calc_risk(evar::EVaR, w::AbstractVector; X::AbstractMatrix, kwargs...)

# Description

Compute the [`EVaR`](@ref) via [`_EVaR`](@ref).

See also: [`EVaR`](@ref), [`_EVaR`](@ref).

# Inputs

## Positional

  - `evar::EVaR`: risk measure.
  - `w::AbstractVector`: `N×1` vector of asset weights.

## Named

  - `X::AbstractMatrix`: `T×N` matrix of asset returns.

# Outputs

  - `evar::Real`: Entropic Value at Risk.

# Examples

```@example
# Sample returns matrix
returns = [ 0.19 -0.41 -0.70;
            1.15 -1.20 -1.27;
           -0.27 -1.98 -0.77;
           -0.65  0.22  0.59;
           -0.04  0.35 -0.99]

# Sample weights vector
w = [0.3, 0.5, 0.2]

# Calculate the entropic value at risk with default parameters
evar_rm1 = EVaR(; solvers = my_solvers)
evar_risk1 = calc_risk(evar_rm1, w; X = returns)

# Calculate the entropic value at risk with 50 % significance parameter
evar_rm2 = EVaR(; solvers = my_solvers, alpha = 0.5)
evar_risk2 = calc_risk(evar_rm2, w; X = returns)
```
"""
function calc_risk(evar::EVaR, w::AbstractVector; X::AbstractMatrix, kwargs...)
    return _EVaR(X * w, evar.solvers, evar.alpha)
end

"""
    calc_risk(rlvar::RLVaR, w::AbstractVector; X::AbstractMatrix, kwargs...)

# Description

Compute the [`RLVaR`](@ref) via [`_RLVaR`](@ref).

See also: [`RLVaR`](@ref), [`_RLVaR`](@ref).

# Inputs

## Positional

  - `rlvar::RLVaR`: risk measure.
  - `w::AbstractVector`: `N×1` vector of asset weights.

## Named

  - `X::AbstractMatrix`: `T×N` matrix of asset returns.

# Outputs

  - `rlvar::Real`: Relativistic Value at Risk.

# Examples

```@example
# Sample returns matrix
returns = [ 0.19 -0.41 -0.70;
            1.15 -1.20 -1.27;
           -0.27 -1.98 -0.77;
           -0.65  0.22  0.59;
           -0.04  0.35 -0.99]

# Sample weights vector
w = [0.3, 0.5, 0.2]

# Calculate the relativistic value at risk with default parameters
rlvar_rm1 = RLVaR(; solvers = my_solvers)
rlvar_risk1 = calc_risk(rlvar_rm1, w; X = returns)

# Calculate the relativistic value at risk with 50 % significance parameter
# and 15 % deformation parameter.
rlvar_rm2 = RLVaR(; solvers = my_solvers, alpha = 0.5, kappa = 0.15)
rlvar_risk2 = calc_risk(rlvar_rm2, w; X = returns)
```
"""
function calc_risk(rlvar::RLVaR, w::AbstractVector; X::AbstractMatrix, kwargs...)
    return _RLVaR(X * w, rlvar.solvers, rlvar.alpha, rlvar.kappa)
end

"""
    calc_risk(dar::DaR, w::AbstractVector; X::AbstractMatrix, kwargs...)

# Description

Compute the [`DaR`](@ref) via [`_DaR`](@ref).

See also: [`DaR`](@ref), [`_DaR`](@ref).

# Inputs

## Positional

  - `dar::DaR`: risk measure.
  - `w::AbstractVector`: `N×1` vector of asset weights.

## Named

  - `X::AbstractMatrix`: `T×N` matrix of asset returns.

# Outputs

  - `dar::Real`: Drawdown at Risk of uncompounded cumulative returns.

# Examples

```@example
# Sample returns matrix
returns = [ 0.19 -0.41 -0.70;
            1.15 -1.20 -1.27;
           -0.27 -1.98 -0.77;
           -0.65  0.22  0.59;
           -0.04  0.35 -0.99]

# Sample weights vector
w = [0.3, 0.5, 0.2]

# Calculate the drawdown at risk of uncompounded cumulative 
# returns with default parameters
dar_rm1 = DaR()
dar_risk1 = calc_risk(dar_rm1, w; X = returns)

# Calculate the drawdown at risk of uncompounded cumulative 
# returns with 50 % significance parameter
dar_rm2 = DaR(; alpha = 0.5)
dar_risk2 = calc_risk(dar_rm2, w; X = returns)
```
"""
function calc_risk(dar::DaR, w::AbstractVector; X::AbstractMatrix, kwargs...)
    return _DaR(X * w, dar.alpha)
end

"""
    calc_risk(::MDD, w::AbstractVector; X::AbstractMatrix, kwargs...)

# Description

Compute the [`MDD`](@ref) via [`_MDD`](@ref).

See also: [`MDD`](@ref), [`_MDD`](@ref).

# Inputs

## Positional

  - `mdd::MDD`: risk measure.
  - `w::AbstractVector`: `N×1` vector of asset weights.

## Named

  - `X::AbstractMatrix`: `T×N` matrix of asset returns.

# Outputs

  - `mdd::Real`: Maximum Drawdown at Risk of uncompounded cumulative returns.

# Examples

```@example
# Sample returns matrix
returns = [ 0.19 -0.41 -0.70;
            1.15 -1.20 -1.27;
           -0.27 -1.98 -0.77;
           -0.65  0.22  0.59;
           -0.04  0.35 -0.99]

# Sample weights vector
w = [0.3, 0.5, 0.2]

# Calculate the drawdown at risk of uncompounded cumulative 
# returns
mdd_rm = MDD()
mdd_risk = calc_risk(mdd_rm, w; X = returns)
```
"""
function calc_risk(::MDD, w::AbstractVector; X::AbstractMatrix, kwargs...)
    return _MDD(X * w)
end

"""
    calc_risk(::ADD, w::AbstractVector; X::AbstractMatrix, kwargs...)

# Description

Compute the [`ADD`](@ref) via [`_ADD`](@ref).

See also: [`ADD`](@ref), [`_ADD`](@ref).

# Inputs

## Positional

  - `add::ADD`: risk measure.
  - `w::AbstractVector`: `N×1` vector of asset weights.

## Named

  - `X::AbstractMatrix`: `T×N` matrix of asset returns.

# Outputs

  - `add::Real`: Average Drawdown at Risk of uncompounded cumulative returns.

# Examples

```@example
# Sample returns matrix
returns = [ 0.19 -0.41 -0.70;
            1.15 -1.20 -1.27;
           -0.27 -1.98 -0.77;
           -0.65  0.22  0.59;
           -0.04  0.35 -0.99]

# Sample weights vector
w = [0.3, 0.5, 0.2]

# Calculate the drawdown at risk of uncompounded cumulative 
# returns with default parameters
add_rm = ADD()
add_risk = calc_risk(add_rm, w; X = returns)
```
"""
function calc_risk(::ADD, w::AbstractVector; X::AbstractMatrix, kwargs...)
    return _ADD(X * w)
end

"""
    calc_risk(cdar::CDaR, w::AbstractVector; X::AbstractMatrix, kwargs...)

# Description

Compute the [`CDaR`](@ref) via [`_CDaR`](@ref).

See also: [`CDaR`](@ref), [`_CDaR`](@ref).

# Inputs

## Positional

  - `cdar::CDaR`: risk measure.
  - `w::AbstractVector`: `N×1` vector of asset weights.

## Named

  - `X::AbstractMatrix`: `T×N` matrix of asset returns.

# Outputs

  - `cdar::Real`: Conditional Drawdown at Risk of uncompounded cumulative returns.

# Examples

```@example
# Sample returns matrix
returns = [ 0.19 -0.41 -0.70;
            1.15 -1.20 -1.27;
           -0.27 -1.98 -0.77;
           -0.65  0.22  0.59;
           -0.04  0.35 -0.99]

# Sample weights vector
w = [0.3, 0.5, 0.2]

# Calculate the conditional drawdown at risk of uncompounded cumulative 
# returns with default parameters
cdar_rm1 = CDaR()
cdar_risk1 = calc_risk(cdar_rm1, w; X = returns)

# Calculate the conditional drawdown at risk of uncompounded cumulative 
# returns with 50 % significance parameter
cdar_rm2 = CDaR(; alpha = 0.5)
cdar_risk2 = calc_risk(cdar_rm2, w; X = returns)
```
"""
function calc_risk(cdar::CDaR, w::AbstractVector; X::AbstractMatrix, kwargs...)
    return _CDaR(X * w, cdar.alpha)
end

"""
    calc_risk(::UCI, w::AbstractVector; X::AbstractMatrix, kwargs...)

# Description

Compute the [`UCI`](@ref) via [`_UCI`](@ref).

See also: [`UCI`](@ref), [`_UCI`](@ref).

# Inputs

## Positional

  - `uci::UCI`: risk measure.
  - `w::AbstractVector`: `N×1` vector of asset weights.

## Named

  - `X::AbstractMatrix`: `T×N` matrix of asset returns.

# Outputs

  - `uci::Real`: Ulcer Index of uncompounded cumulative returns.

# Examples

```@example
# Sample returns matrix
returns = [ 0.19 -0.41 -0.70;
            1.15 -1.20 -1.27;
           -0.27 -1.98 -0.77;
           -0.65  0.22  0.59;
           -0.04  0.35 -0.99]

# Sample weights vector
w = [0.3, 0.5, 0.2]

# Calculate the ulcer index at risk of uncompounded cumulative 
# returns with default parameters
uci_rm = UCI()
uci_risk = calc_risk(uci_rm, w; X = returns)
```
"""
function calc_risk(::UCI, w::AbstractVector; X::AbstractMatrix, kwargs...)
    return _UCI(X * w)
end

"""
    calc_risk(edar::EDaR, w::AbstractVector; X::AbstractMatrix, kwargs...)

# Description

Compute the [`EDaR`](@ref) via [`_EDaR`](@ref).

See also: [`EDaR`](@ref), [`_EDaR`](@ref).

# Inputs

## Positional

  - `edar::EDaR`: risk measure.
  - `w::AbstractVector`: `N×1` vector of asset weights.

## Named

  - `X::AbstractMatrix`: `T×N` matrix of asset returns.

# Outputs

  - `edar::Real`: Entropic Drawdown at Risk of uncompounded cumulative returns.

# Examples

```@example
# Sample returns matrix
returns = [ 0.19 -0.41 -0.70;
            1.15 -1.20 -1.27;
           -0.27 -1.98 -0.77;
           -0.65  0.22  0.59;
           -0.04  0.35 -0.99]

# Sample weights vector
w = [0.3, 0.5, 0.2]

# Calculate the entropic drawdown at risk of uncompounded cumulative 
# returns with default parameters
edar_rm1 = EDaR(; solvers = my_solvers)
edar_risk1 = calc_risk(edar_rm1, w; X = returns)

# Calculate the entropic drawdown at risk of uncompounded cumulative
# returns with 50 % significance parameter
edar_rm2 = EDaR(; solvers = my_solvers, alpha = 0.5)
edar_risk2 = calc_risk(edar_rm2, w; X = returns)
```
"""
function calc_risk(edar::EDaR, w::AbstractVector; X::AbstractMatrix, kwargs...)
    return _EDaR(X * w, edar.solvers, edar.alpha)
end

"""
    calc_risk(rldar::RLDaR, w::AbstractVector; X::AbstractMatrix, kwargs...)

# Description

Compute the [`RLDaR`](@ref) via [`_RLDaR`](@ref).

See also: [`RLDaR`](@ref), [`_RLDaR`](@ref).

# Inputs

## Positional

  - `rldar::RLDaR`: risk measure.
  - `w::AbstractVector`: `N×1` vector of asset weights.

## Named

  - `X::AbstractMatrix`: `T×N` matrix of asset returns.

# Outputs

  - `rldar::Real`: Relativistic Drawdown at Risk of uncompounded cumulative returns.

# Examples

```@example
# Sample returns matrix
returns = [ 0.19 -0.41 -0.70;
            1.15 -1.20 -1.27;
           -0.27 -1.98 -0.77;
           -0.65  0.22  0.59;
           -0.04  0.35 -0.99]

# Sample weights vector
w = [0.3, 0.5, 0.2]

# Calculate the entropic drawdown at risk of uncompounded cumulative 
# returns with default parameters
rldar_rm1 = RLDaR(; solvers = my_solvers)
rldar_risk1 = calc_risk(rldar_rm1, w; X = returns)

# Calculate the entropic drawdown at risk of uncompounded cumulative
# returns with 50 % significance parameter
rldar_rm2 = RLDaR(; solvers = my_solvers, alpha = 0.5)
rldar_risk2 = calc_risk(rldar_rm2, w; X = returns)
```
"""
function calc_risk(rldar::RLDaR, w::AbstractVector; X::AbstractMatrix, kwargs...)
    return _RLDaR(X * w, rldar.solvers, rldar.alpha, rldar.kappa)
end

"""
    calc_risk(dar_r::DaR_r, w::AbstractVector; X::AbstractMatrix, kwargs...)

# Description

Compute the [`DaR_r`](@ref) via [`_DaR_r`](@ref).

See also: [`DaR_r`](@ref), [`_DaR_r`](@ref).

# Inputs

## Positional

  - `dar_r::DaR_r`: risk measure.
  - `w::AbstractVector`: `N×1` vector of asset weights.

## Named

  - `X::AbstractMatrix`: `T×N` matrix of asset returns.

# Outputs

  - `dar_r::Real`: Drawdown at Risk of compounded cumulative returns.

# Examples

```@example
# Sample returns matrix
returns = [ 0.19 -0.41 -0.70;
            1.15 -1.20 -1.27;
           -0.27 -1.98 -0.77;
           -0.65  0.22  0.59;
           -0.04  0.35 -0.99]

# Sample weights vector
w = [0.3, 0.5, 0.2]

# Calculate the drawdown at risk of compounded cumulative 
# returns with default parameters
dar_r_rm1 = DaR_r()
dar_r_risk1 = calc_risk(dar_r_rm1, w; X = returns)

# Calculate the drawdown at risk of compounded cumulative 
# returns with 50 % significance parameter
dar_r_rm2 = DaR_r(; alpha = 0.5)
dar_r_risk2 = calc_risk(dar_r_rm2, w; X = returns)
```
"""
function calc_risk(dar_r::DaR_r, w::AbstractVector; X::AbstractMatrix, kwargs...)
    return _DaR_r(X * w, dar_r.alpha)
end

"""
    calc_risk(::MDD_r, w::AbstractVector; X::AbstractMatrix, kwargs...)

# Description

Compute the [`MDD_r`](@ref) via [`_MDD_r`](@ref).

See also: [`MDD_r`](@ref), [`_MDD_r`](@ref).

# Inputs

## Positional

  - `mdd_r::MDD_r`: risk measure.
  - `w::AbstractVector`: `N×1` vector of asset weights.

## Named

  - `X::AbstractMatrix`: `T×N` matrix of asset returns.

# Outputs

  - `mdd_r::Real`: Maximum Drawdown at Risk of compounded cumulative returns.

# Examples

```@example
# Sample returns matrix
returns = [ 0.19 -0.41 -0.70;
            1.15 -1.20 -1.27;
           -0.27 -1.98 -0.77;
           -0.65  0.22  0.59;
           -0.04  0.35 -0.99]

# Sample weights vector
w = [0.3, 0.5, 0.2]

# Calculate the drawdown at risk of compounded cumulative 
# returns
mdd_r_rm = MDD_r()
mdd_r_risk = calc_risk(mdd_rm, w; X = returns)
```
"""
function calc_risk(::MDD_r, w::AbstractVector; X::AbstractMatrix, kwargs...)
    return _MDD_r(X * w)
end

"""
    calc_risk(::ADD_r, w::AbstractVector; X::AbstractMatrix, kwargs...)

# Description

Compute the [`ADD_r`](@ref) via [`_ADD_r`](@ref).

See also: [`ADD_r`](@ref), [`_ADD_r`](@ref).

# Inputs

## Positional

  - `add_r::ADD_r`: risk measure.
  - `w::AbstractVector`: `N×1` vector of asset weights.

## Named

  - `X::AbstractMatrix`: `T×N` matrix of asset returns.

# Outputs

  - `add_r::Real`: Average Drawdown at Risk of compounded cumulative returns.

# Examples

```@example
# Sample returns matrix
returns = [ 0.19 -0.41 -0.70;
            1.15 -1.20 -1.27;
           -0.27 -1.98 -0.77;
           -0.65  0.22  0.59;
           -0.04  0.35 -0.99]

# Sample weights vector
w = [0.3, 0.5, 0.2]

# Calculate the drawdown at risk of compounded cumulative 
# returns with default parameters
add_r_rm = ADD_r()
add_r_risk = calc_risk(add_r_rm, w; X = returns)
```
"""
function calc_risk(::ADD_r, w::AbstractVector; X::AbstractMatrix, kwargs...)
    return _ADD_r(X * w)
end

"""
calc_risk(cdar_r::CDaR_r, w::AbstractVector; X::AbstractMatrix, kwargs...)

# Description

Compute the [`CDaR_r`](@ref) via [`_CDaR_r`](@ref).

See also: [`CDaR_r`](@ref), [`_CDaR_r`](@ref).

# Inputs

## Positional

  - `cdar::CDaR_r`: risk measure.
  - `w::AbstractVector`: `N×1` vector of asset weights.

## Named

  - `X::AbstractMatrix`: `T×N` matrix of asset returns.

# Outputs

  - `cdar_r::Real`: Conditional Drawdown at Risk of compounded cumulative returns.

# Examples

```@example
# Sample returns matrix
returns = [ 0.19 -0.41 -0.70;
            1.15 -1.20 -1.27;
           -0.27 -1.98 -0.77;
           -0.65  0.22  0.59;
           -0.04  0.35 -0.99]

# Sample weights vector
w = [0.3, 0.5, 0.2]

# Calculate the conditional drawdown at risk of compounded cumulative 
# returns with default parameters
cdar_r_rm1 = CDaR_r()
cdar_r_risk1 = calc_risk(cdar_r_rm1, w; X = returns)

# Calculate the conditional drawdown at risk of compounded cumulative 
# returns with 50 % significance parameter
cdar_r_rm2 = CDaR_r(; alpha = 0.5)
cdar_r_risk2 = calc_risk(cdar_r_rm2, w; X = returns)
```
"""
function calc_risk(cdar_r::CDaR_r, w::AbstractVector; X::AbstractMatrix, kwargs...)
    return _CDaR_r(X * w, cdar_r.alpha)
end

"""
    calc_risk(::UCI_r, w::AbstractVector; X::AbstractMatrix, kwargs...)

Compute the [`UCI_r`](@ref) via [`_UCI_r`](@ref).

See also: [`UCI_r`](@ref), [`_UCI_r`](@ref).

# Inputs

## Positional

  - `uci_r::UCI_r`: risk measure.
  - `w::AbstractVector`: `N×1` vector of asset weights.

## Named

  - `X::AbstractMatrix`: `T×N` matrix of asset returns.

# Outputs

  - `uci_r::Real`: Ulcer Index of compounded cumulative returns.

# Examples

```@example
# Sample returns matrix
returns = [ 0.19 -0.41 -0.70;
            1.15 -1.20 -1.27;
           -0.27 -1.98 -0.77;
           -0.65  0.22  0.59;
           -0.04  0.35 -0.99]

# Sample weights vector
w = [0.3, 0.5, 0.2]

# Calculate the ulcer index at risk of compounded cumulative 
# returns with default parameters
uci_r_rm = UCI_r()
uci_r_risk = calc_risk(uci_r_rm, w; X = returns)
```
"""
function calc_risk(::UCI_r, w::AbstractVector; X::AbstractMatrix, kwargs...)
    return _UCI_r(X * w)
end

"""
    calc_risk(edar_r::EDaR_r, w::AbstractVector; X::AbstractMatrix, kwargs...)

# Description

Compute the [`EDaR_r`](@ref) via [`_EDaR_r`](@ref).

See also: [`EDaR_r`](@ref), [`_EDaR_r`](@ref).

# Inputs

## Positional

  - `edar_r::EDaR_r`: risk measure.
  - `w::AbstractVector`: `N×1` vector of asset weights.

## Named

  - `X::AbstractMatrix`: `T×N` matrix of asset returns.

# Outputs

  - `edar_r::Real`: Entropic Drawdown at Risk of compounded cumulative returns.

# Examples

```@example
# Sample returns matrix
returns = [ 0.19 -0.41 -0.70;
            1.15 -1.20 -1.27;
           -0.27 -1.98 -0.77;
           -0.65  0.22  0.59;
           -0.04  0.35 -0.99]

# Sample weights vector
w = [0.3, 0.5, 0.2]

# Calculate the entropic drawdown at risk of compounded cumulative 
# returns with default parameters
edar_r_rm1 = EDaR_r(; solvers = my_solvers)
edar_r_risk1 = calc_risk(edar_r_rm1, w; X = returns)

# Calculate the entropic drawdown at risk of compounded cumulative
# returns with 50 % significance parameter
edar_r_rm2 = EDaR_r(; solvers = my_solvers, alpha = 0.5)
edar_r_risk2 = calc_risk(edar_r_rm2, w; X = returns)
```
"""
function calc_risk(edar_r::EDaR_r, w::AbstractVector; X::AbstractMatrix, kwargs...)
    return _EDaR_r(X * w, edar_r.solvers, edar_r.alpha)
end

"""
    calc_risk(rldar_r::RLDaR_r, w::AbstractVector; X::AbstractMatrix, kwargs...)

# Description

Compute the [`RLDaR_r`](@ref) via [`_RLDaR_r`](@ref).

See also: [`RLDaR_r`](@ref), [`_RLDaR_r`](@ref).

# Inputs

## Positional

  - `rldar_r::RLDaR_r`: risk measure.
  - `w::AbstractVector`: `N×1` vector of asset weights.

## Named

  - `X::AbstractMatrix`: `T×N` matrix of asset returns.

# Outputs

  - `rldar_r::Real`: Relativistic Drawdown at Risk of compounded cumulative returns.

# Examples

```@example
# Sample returns matrix
returns = [ 0.19 -0.41 -0.70;
            1.15 -1.20 -1.27;
           -0.27 -1.98 -0.77;
           -0.65  0.22  0.59;
           -0.04  0.35 -0.99]

# Sample weights vector
w = [0.3, 0.5, 0.2]

# Calculate the relativistic drawdown at risk of compounded cumulative 
# returns with default parameters
rldar_r_rm1 = RLDaR_r(; solvers = my_solvers)
rldar_r_risk1 = calc_risk(rldar_r_rm1, w; X = returns)

# Calculate the relativistic drawdown at risk of compounded cumulative
# returns with 50 % significance parameter
rldar_r_rm2 = RLDaR_r(; solvers = my_solvers, alpha = 0.5)
rldar_r_risk2 = calc_risk(rldar_r_rm2, w; X = returns)
```
"""
function calc_risk(rldar_r::RLDaR_r, w::AbstractVector; X::AbstractMatrix, kwargs...)
    return _RLDaR_r(X * w, rldar_r.solvers, rldar_r.alpha, rldar_r.kappa)
end

"""
    calc_risk(kt::Kurt, w::AbstractVector; X::AbstractMatrix, kwargs...)

# Description

Compute the [`Kurt`](@ref) via [`_Kurt`](@ref).

See also: [`Kurt`](@ref), [`_Kurt`](@ref).

# Inputs

## Positional

  - `kurt::Kurt`: risk measure.
  - `w::AbstractVector`: `N×1` vector of asset weights.

## Named

  - `X::AbstractMatrix`: `T×N` matrix of asset returns.

# Outputs

  - `kurt::Real`: square root kurtosis.

# Examples

```@example
# Sample returns matrix
returns = [ 0.19 -0.41 -0.70;
            1.15 -1.20 -1.27;
           -0.27 -1.98 -0.77;
           -0.65  0.22  0.59;
           -0.04  0.35 -0.99]

# Sample weights vector
w = [0.3, 0.5, 0.2]

# Calculate the square root kurtosis with default parameters
kurt_rm1 = Kurt()
kurt_risk1 = calc_risk(kurt_rm1, w; X = returns)

# Calculate the square root kurtosis using a weighted 
# mean returns using weighted mean
kurt_rm2 = Kurt(; w = eweights(1:size(returns, 1), 0.3))
kurt_risk2 = calc_risk(kurt_rm2, w; X = returns)
```
"""
function calc_risk(kt::Kurt, w::AbstractVector; X::AbstractMatrix, kwargs...)
    return _Kurt(X * w, kt.w)
end

"""
    calc_risk(skt::SKurt, w::AbstractVector; X::AbstractMatrix, kwargs...)

# Description

Compute the [`SKurt`](@ref) via [`_SKurt`](@ref).

See also: [`SKurt`](@ref), [`_SKurt`](@ref).

# Inputs

## Positional

  - `skurt::SKurt`: risk measure.
  - `w::AbstractVector`: `N×1` vector of asset weights.

## Named

  - `X::AbstractMatrix`: `T×N` matrix of asset returns.

# Outputs

  - `skurt::Real`: square root semi.

# Examples

```@example
# Sample returns matrix
returns = [ 0.19 -0.41 -0.70;
            1.15 -1.20 -1.27;
           -0.27 -1.98 -0.77;
           -0.65  0.22  0.59;
           -0.04  0.35 -0.99]

# Sample weights vector
w = [0.3, 0.5, 0.2]

# Calculate the square root semi kurtosis with default parameters
skurt_rm1 = SKurt()
skurt_risk1 = calc_risk(skurt_rm1, w; X = returns)

# Calculate the square root semi kurtosis using mean returns using a 
# weighted mean and 1.5 % return target
skurt_rm2 = SKurt(; w = eweights(1:size(returns, 1), 0.3), target = 0.015)
skurt_risk2 = calc_risk(skurt_rm2, w; X = returns)
```
"""
function calc_risk(skt::SKurt, w::AbstractVector; X::AbstractMatrix, kwargs...)
    return _SKurt(X * w, skt.target, skt.w)
end

"""
    calc_risk(::GMD, w::AbstractVector; X::AbstractMatrix, kwargs...)

# Description

Compute the [`GMD`](@ref) via [`_GMD`](@ref).

See also: [`GMD`](@ref), [`_GMD`](@ref).

# Inputs

## Positional

  - `gmd::GMD`: risk measure.
  - `w::AbstractVector`: `N×1` vector of asset weights.

## Named

  - `X::AbstractMatrix`: `T×N` matrix of asset returns.

# Outputs

  - `gmd::Real`: Gini Mean Difference.

# Examples

```@example
# Sample returns matrix
returns = [ 0.19 -0.41 -0.70;
            1.15 -1.20 -1.27;
           -0.27 -1.98 -0.77;
           -0.65  0.22  0.59;
           -0.04  0.35 -0.99]

# Sample weights vector
w = [0.3, 0.5, 0.2]

# Calculate the gini mean difference with default parameters
gmd_rm = GMD()
gmd_risk = calc_risk(gmd_rm, w; X = returns)
```
"""
function calc_risk(::GMD, w::AbstractVector; X::AbstractMatrix, kwargs...)
    return _GMD(X * w)
end

"""
    calc_risk(::RG, w::AbstractVector; X::AbstractMatrix, kwargs...)

# Description

Compute the [`RG`](@ref) via [`_RG`](@ref).

See also: [`RG`](@ref), [`RG`](@ref).

# Inputs

## Positional

  - `rg::RG`: risk measure.
  - `w::AbstractVector`: `N×1` vector of asset weights.

## Named

  - `X::AbstractMatrix`: `T×N` matrix of asset returns.

# Outputs

  - `rg::Real`: Gini Mean Difference.

# Examples

```@example
# Sample returns matrix
returns = [ 0.19 -0.41 -0.70;
            1.15 -1.20 -1.27;
           -0.27 -1.98 -0.77;
           -0.65  0.22  0.59;
           -0.04  0.35 -0.99]

# Sample weights vector
w = [0.3, 0.5, 0.2]

# Calculate the range with default parameters
rg_rm = RG()
rg_risk = calc_risk(rg_rm, w; X = returns)
```
"""
function calc_risk(::RG, w::AbstractVector; X::AbstractMatrix, kwargs...)
    return _RG(X * w)
end

"""
    calc_risk(rcvar::CVaRRG, w::AbstractVector; X::AbstractMatrix, kwargs...)

# Description

Compute the [`CVaRRG`](@ref) via [`_CVaRRG`](@ref).

See also: [`CVaRRG`](@ref), [`_CVaRRG`](@ref).

# Inputs

## Positional

  - `cvarrg::CVaRRG`: risk measure.
  - `w::AbstractVector`: `N×1` vector of asset weights.

## Named

  - `X::AbstractMatrix`: `T×N` matrix of asset returns.

# Outputs

  - `cvarrg::Real`: Conditional Value at Risk Range.

# Examples

```@example
# Sample returns matrix
returns = [ 0.19 -0.41 -0.70;
            1.15 -1.20 -1.27;
           -0.27 -1.98 -0.77;
           -0.65  0.22  0.59;
           -0.04  0.35 -0.99]

# Sample weights vector
w = [0.3, 0.5, 0.2]

# Calculate the conditional value at risk range with default parameters
cvarrg_rm1 = CVaRRG()
cvarrg_risk1 = calc_risk(cvarrg_rm1, w; X = returns)

# Calculate conditional value at risk range  with 7 % significance 
# parameter of losses and 32 % significance parameter of gains.
cvarrg_rm2 = CVaRRG(; alpha = 0.07, beta = 0.32)
cvarrg_risk2 = calc_risk(cvarrg_rm2, w; X = returns)
```
"""
function calc_risk(rcvar::CVaRRG, w::AbstractVector; X::AbstractMatrix, kwargs...)
    return _CVaRRG(X * w; alpha = rcvar.alpha, beta = rcvar.beta)
end

"""
    calc_risk(tg::TG, w::AbstractVector; X::AbstractMatrix, kwargs...)

# Description

Compute the [`TG`](@ref) via [`_TG`](@ref).

See also: [`TG`](@ref), [`_TG`](@ref).

# Inputs

## Positional

  - `tg::TG`: risk measure.
  - `w::AbstractVector`: `N×1` vector of asset weights.

## Named

  - `X::AbstractMatrix`: `T×N` matrix of asset returns.

# Outputs

  - `tg::Real`: Tail Gini.

# Examples

```@example
# Sample returns matrix
returns = [ 0.19 -0.41 -0.70;
            1.15 -1.20 -1.27;
           -0.27 -1.98 -0.77;
           -0.65  0.22  0.59;
           -0.04  0.35 -0.99]

# Sample weights vector
w = [0.3, 0.5, 0.2]

# Calculate the tail gini with default parameters
tg_rm1 = TG()
tg_risk1 = calc_risk(tg_rm1, w; X = returns)

# Calculate the tail gini with custom parameters
tg_rm2 = TG(; alpha_i = 0.003, alpha = 0.07, a_sim = 131)
tg_risk2 = calc_risk(tg_rm2, w; X = returns)
```
"""
function calc_risk(tg::TG, w::AbstractVector; X::AbstractMatrix, kwargs...)
    return _TG(X * w; alpha_i = tg.alpha_i, alpha = tg.alpha, a_sim = tg.a_sim)
end

"""
    calc_risk(rtg::TGRG, w::AbstractVector; X::AbstractMatrix, kwargs...)

# Description

Compute the [`TGRG`](@ref) via [`_TGRG`](@ref).

See also: [`TGRG`](@ref), [`_TGRG`](@ref).

# Inputs

## Positional

  - `tgrg::TGRG`: risk measure.
  - `w::AbstractVector`: `N×1` vector of asset weights.

## Named

  - `X::AbstractMatrix`: `T×N` matrix of asset returns.

# Outputs

  - `tgrg::Real`: Tail Gini Range.

# Examples

```@example
# Sample returns matrix
returns = [ 0.19 -0.41 -0.70;
            1.15 -1.20 -1.27;
           -0.27 -1.98 -0.77;
           -0.65  0.22  0.59;
           -0.04  0.35 -0.99]

# Sample weights vector
w = [0.3, 0.5, 0.2]

# Calculate the tail gini range with default parameters
tgrg_rm1 = TGRG()
tgrg_risk1 = calc_risk(tgrg_rm1, w; X = returns)

# Calculate the tail gini range with custom parameters
tgrg_rm2 = TGRG(; alpha_i = 0.003, alpha = 0.07, a_sim = 131, beta_i = 0.001, beta = 0.03,
                b_sim = 97)
tgrg_risk2 = calc_risk(tgrg_rm2, w; X = returns)
```
"""
function calc_risk(rtg::TGRG, w::AbstractVector; X::AbstractMatrix, kwargs...)
    return _TGRG(X * w; alpha_i = rtg.alpha_i, alpha = rtg.alpha, a_sim = rtg.a_sim,
                 beta_i = rtg.beta_i, beta = rtg.beta, b_sim = rtg.b_sim)
end

"""
    calc_risk(owa::OWA, w::AbstractVector; X::AbstractMatrix, kwargs...)

# Description

Compute the [`OWA`](@ref) via [`_OWA`](@ref).

See also: [`OWA`](@ref), [`_OWA`](@ref).

# Inputs

## Positional

  - `owa::OWA`: risk measure.
  - `w::AbstractVector`: `N×1` vector of asset weights.

## Named

  - `X::AbstractMatrix`: `T×N` matrix of asset returns.

# Outputs

  - `owa::Real`: Ordered Weight Array risk.

# Examples

```@example
# Sample returns matrix
returns = [ 0.19 -0.41 -0.70;
            1.15 -1.20 -1.27;
           -0.27 -1.98 -0.77;
           -0.65  0.22  0.59;
           -0.04  0.35 -0.99]

# Sample weights vector
w = [0.3, 0.5, 0.2]

# Calculate the ordered weight array risk for the default L-moment
owa_rm = OWA(; w = owa_l_moment_crm(size(returns, 1)))
owa_risk = calc_risk(owa_rm, w; X = returns)
```
"""
function calc_risk(owa::OWA, w::AbstractVector; X::AbstractMatrix, kwargs...)
    return _OWA(X * w, isnothing(owa.w) ? owa_gmd(size(X, 1)) : owa.w)
end

"""
    calc_risk(::BDVariance, w::AbstractVector; X::AbstractMatrix, kwargs...)

# Description

Compute the [`BDVariance`](@ref) via [`_BDVariance`](@ref).

See also: [`BDVariance`](@ref), [`_BDVariance`](@ref).

# Inputs

## Positional

  - `bdvariance::BDVariance`: risk measure.
  - `w::AbstractVector`: `N×1` vector of asset weights.

## Named

  - `X::AbstractMatrix`: `T×N` matrix of asset returns.

# Outputs

  - `bdvariance::Real`: Brownian Distance Variance.

# Examples

```@example
# Sample returns matrix
returns = [ 0.19 -0.41 -0.70;
            1.15 -1.20 -1.27;
           -0.27 -1.98 -0.77;
           -0.65  0.22  0.59;
           -0.04  0.35 -0.99]

# Sample weights vector
w = [0.3, 0.5, 0.2]

# Calculate the Brownian distance variance
bdvariance_rm = BDVariance()
bdvariance_risk = calc_risk(bdvariance_rm, w; X = returns)
```
"""
function calc_risk(::BDVariance, w::AbstractVector; X::AbstractMatrix, kwargs...)
    return _BDVariance(X * w)
end

"""
    calc_risk(::RMSkew, w::AbstractVector; kwargs...)

# Description

Compute the [`Skew`](@ref) via [`_Skew`](@ref).

See also: [`Skew`](@ref), [`_Skew`](@ref).

# Inputs

## Positional

  - `skew::Skew`: risk measure.
  - `w::AbstractVector`: `N×1` vector of asset weights.

# Outputs

  - `skew::Real`: Quadratic Skewness.

# Examples

```@example
# Create sample sum of the symmetric negative spectral slices 
# of the coskewness
V = [0.04 0.02 0.01;
     0.02 0.09 0.03;
     0.01 0.03 0.06]

# Sample weights vector
w = [0.3, 0.5, 0.2]

# Calculate the skew
skew_rm = Skew()
skew_risk = calc_risk(skew_rm, w; V = V)
```
"""
function calc_risk(skew::RMSkew, w::AbstractVector; kwargs...)
    return _Skew(w, skew.V)
end

"""
    calc_risk(::Equal, w::AbstractVector; delta::Real = 0, kwargs...)

# Description

Compute the risk as the inverse of the length of `w`.

# Inputs

## Positional

  - `equal::Equal`: risk measure.
  - `w::AbstractVector`: `N×1` vector of asset weights.

## Named

  - `delta::Real`: is a displacement, used in [`risk_contribution`](@ref) and [`factor_risk_contribution`](@ref).

# Outputs

  - `equal::Real`: Equal Risk.

# Examples

```@example
# Sample weights vector
w = [0.3, 0.5, 0.2]

# Calculate the equal risk measure
equal_rm = Equal()
equal_risk = calc_risk(equal_rm, w)
```
"""
function calc_risk(::Equal, w::AbstractVector; delta::Real = 0, kwargs...)
    return inv(length(w)) + delta
end

export calc_risk, cumulative_returns, drawdown
