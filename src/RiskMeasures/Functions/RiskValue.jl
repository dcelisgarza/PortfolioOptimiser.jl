# Copywrite (c) 2025
# Author: Daniel Celis Garza <daniel.celis.garza@gmail.com>
# SPDX-License-Identifier: MIT

"""
    ERM(x::AbstractVector{<:Real}, z::Real = 1.0, α::Real = 0.05)

Computes the Entropic Risk Measure.

```math
\\begin{align}
\\mathrm{ERM}(\\bm{X},\\, z, \\,\\alpha) &= z \\ln \\left(\\dfrac{M_{\\bm{X}}\\left(z^{-1}\\right)}{\\alpha} \\right)\\,.
\\end{align}
```

Where:

  - ``\\bm{X}``: is the `T×1` vector of portfolio returns.
  - ``z``: is the entropic moment.
  - ``M_{\\bm{X}}\\left(t\\right)``: is the moment generating function of ``\\bm{X}``.
  - ``\\alpha``: is the significance parameter, ``\\alpha \\in (0,\\,1)``.

# Positional Arguments

  - `x::AbstractVector`: `T×1` vector of portfolio returns.
  - `z::Real = 1.0`: entropic moment.
  - `α::Real = 0.05`: significance level, `α ∈ (0, 1)`.

# Outputs

  - `er::Real`: entropic risk.

# Examples
"""
function ERM(x::AbstractVector{<:Real}, z::Real = 1.0, alpha::Real = 0.05)
    val = mean(exp.(-x / z))
    val = z * log(val / alpha)
    return val
end

"""
    ERM(x::AbstractVector{<:Real},
             solvers::Union{PortOptSolver, <:AbstractVector{PortOptSolver}},
             alpha::Real = 0.05)

Computes the Entropic Risk Measure.

```math
\\begin{align}
\\mathrm{ERM}(\\bm{X}, \\,\\alpha) &= 
    \\begin{cases}
        \\underset{z,\\, t,\\, u}{\\inf} & t + z \\ln\\left(\\dfrac{1}{\\alpha T}\\right)\\\\
        \\mathrm{s.t.} & z \\geq \\sum\\limits_{i=1}^{T} u_{i}\\nonumber\\\\
        & (-X_{i}-t,\\, z,\\, u_{i}) \\in \\mathcal{K}_{\\exp} \\, \\forall \\, i=1,\\,\\dots{},\\, T
    \\end{cases}\\\\
&= \\underset{z>0}{\\inf}\\left\\{ z \\ln \\left(\\dfrac{M_{\\bm{X}}\\left(z^{-1}\\right)}{\\alpha} \\right)\\right\\}\\,.
\\end{align}
```

Where:

  - ``\\bm{X}``: is the `T×1` vector of portfolio returns.
  - ``\\alpha``: is the significance parameter, ``\\alpha \\in (0,\\,1)``.
  - ``z``: is the entropic moment.
  - ``t`` and ``\\bm{u}``: are auxiliary variables for modelling the moment generating function.
  - ``T``: is the number of observations.
  - ``X_{i}``: is the `i`-th entry of the portfolio returns vector.
  - ``M_{\\bm{X}}\\left(t\\right)``: is the moment generating function of ``\\bm{X}``.
  - ``\\mathcal{K}_{\\mathrm{exp}}``: is the [ExponentialCone](https://jump.dev/JuMP.jl/stable/tutorials/conic/tips_and_tricks/#Exponential-Cone).

See also: [`EVaR`](@ref), [`EDaR`](@ref), [`EDaR_r`](@ref), [`expected_risk`](@ref).

# Inputs

  - `x::AbstractVector`: `T×1` vector of portfolio returns.
  - `solvers::Union{PortOptSolver, <:AbstractVector{PortOptSolver}}`: instance or `AbstractVector` of [`PortOptSolver`](@ref) which supports [ExponentialCone](https://jump.dev/JuMP.jl/stable/tutorials/conic/tips_and_tricks/#Exponential-Cone) constraints.
  - `α::Real = 0.05`: significance level, `α ∈ (0, 1)`.

# Behaviour

  - If no valid solution is found returns `NaN`.

!!! warning

      - `α` is not validated because this is an internal function. It should have been validated by [`EVaR`](@ref), [`EDaR`](@ref), or [`EDaR_r`](@ref).

# Outputs

  - `er::Real`: entropic risk.

# Examples
"""
function ERM(x::AbstractVector{<:Real},
             solvers::Union{PortOptSolver, <:AbstractVector{PortOptSolver}},
             alpha::Real = 0.05)
    model = JuMP.Model()
    set_string_names_on_creation(model, false)
    T = length(x)
    at = alpha * T

    @variables(model, begin
                   t
                   z >= 0
                   u[1:T]
               end)
    @constraints(model, begin
                     sum(u) <= z
                     [i = 1:T], [-x[i] - t, z, u[i]] ∈ MOI.ExponentialCone()
                 end)
    @expression(model, risk, t - z * log(at))
    @objective(model, Min, risk)
    success, solvers_tried = optimise_JuMP_model(model, solvers)
    return if success
        objective_value(model)
    else
        funcname = "$(fullname(PortfolioOptimiser)[1]).$(nameof(PortfolioOptimiser.ERM))"
        @warn("$funcname: Model could not be optimised satisfactorily.\nSolvers: $solvers_tried.")
        NaN
    end
end

"""
    RRM(x::AbstractVector, solvers::AbstractDict, α::Real = 0.05, κ::Real = 0.3)

Computes the Relativistic Risk Measure.

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

  - ``\\bm{X}``: is the `T×1` vector of portfolio returns.
  - ``\\alpha``: is the significance parameter, ``\\alpha \\in (0,\\,1)``.
  - ``\\kappa``: is the relativistic deformation parameter, ``\\kappa \\in (0,\\,1)``.
  - ``z``: is the relativistic moment.
  - ``t``, ``\\psi``, ``\\theta``, ``\\varepsilon``, and ``\\omega``: are auxiliary variables for modelling the relativistic deformation of the moment generating function.
  - ``T``: is the number of observations.
  - ``\\mathcal{P}_3^{c,\\, 1-c}``: is the [PowerCone](https://jump.dev/JuMP.jl/stable/tutorials/conic/tips_and_tricks/#PowerCone) parametrised by ``c \\in (0,\\,1)``.

See also: [`RLVaR`](@ref), [`RLDaR`](@ref), [`RLDaR_r`](@ref), [`expected_risk`](@ref).

# Inputs

  - `x::AbstractVector`: `T×1` vector of portfolio returns.
  - `solvers::Union{Nothing, PortOptSolver, <:AbstractVector{PortOptSolver}}`: instance or `AbstractVector` of [`PortOptSolver`](@ref) which supports [ExponentialCone](https://jump.dev/JuMP.jl/stable/tutorials/conic/tips_and_tricks/#PowerCone) constraints.
  - `α::Real = 0.05`: significance level, `α ∈ (0, 1)`.
  - `κ::Real = 0.3`: relativistic deformation parameter, `κ ∈ (0, 1)`.

# Behaviour

  - If no valid solution is found returns `NaN`.

!!! warning

      - `α` and `κ` are not validated because this is an internal function. They should have been validated by [`RLVaR`](@ref), [`RLDaR`](@ref), or [`RLDaR_r`](@ref).

# Outputs

  - `rlr::Real`: relativistic risk.

# Examples
"""
function RRM(x::AbstractVector,
             solvers::Union{PortOptSolver, <:AbstractVector{PortOptSolver}},
             alpha::Real = 0.05, kappa::Real = 0.3)
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

    @variables(model, begin
                   t
                   z >= 0
                   omega[1:T]
                   psi[1:T]
                   theta[1:T]
                   epsilon[1:T]
               end)
    @constraints(model,
                 begin
                     [i = 1:T],
                     [z * opk * invk2, psi[i] * opk * invk, epsilon[i]] ∈
                     MOI.PowerCone(invopk)
                     [i = 1:T],
                     [omega[i] * invomk, theta[i] * invk, -z * invk2] ∈ MOI.PowerCone(omk)
                     -x .- t .+ epsilon .+ omega .<= 0
                 end)
    @expression(model, risk, t + ln_k * z + sum(psi .+ theta))
    @objective(model, Min, risk)
    success, solvers_tried = optimise_JuMP_model(model, solvers)
    return if success
        objective_value(model)
    else
        model = JuMP.Model()
        set_string_names_on_creation(model, false)
        @variables(model, begin
                       z[1:T]
                       nu[1:T]
                       tau[1:T]
                   end)
        @constraints(model, begin
                         sum(z) == 1
                         sum(nu .- tau) * invk2 <= ln_k
                         [i = 1:T], [nu[i], 1, z[i]] ∈ MOI.PowerCone(invopk)
                         [i = 1:T], [z[i], 1, tau[i]] ∈ MOI.PowerCone(omk)
                     end)
        @expression(model, risk, -dot(z, x))
        @objective(model, Max, risk)
        success, solvers_tried = optimise_JuMP_model(model, solvers)
        if success
            objective_value(model)
        else
            funcname = "$(fullname(PortfolioOptimiser)[1]).$(nameof(PortfolioOptimiser.RRM))"
            @warn("$funcname: Model could not be optimised satisfactorily.\nSolvers: $solvers_tried.")
            NaN
        end
    end
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
    expected_risk(::Equal, w::AbstractVector; delta::Real = 0, kwargs...)

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
equal_risk = expected_risk(equal_rm, w)
```
"""
function expected_risk(equal::Equal, w::AbstractVector; delta::Real = 0, kwargs...)
    return equal(w, delta)
end
function calc_fees(w::AbstractVector, fees::Union{AbstractVector{<:Real}, Real},
                   op::Function)
    return if isa(fees, Real) && !iszero(fees)
        idx = op(w, zero(eltype(w)))
        sum(fees * w[idx])
    elseif isa(fees, AbstractVector) && !(isempty(fees) || all(iszero.(fees)))
        idx = op(w, zero(eltype(w)))
        dot(fees[idx], w[idx])
    else
        zero(eltype(w))
    end
end
function calc_fees(w::AbstractVector, rebalance::AbstractTR)
    return if isa(rebalance, TR)
        fees_rebal = rebalance.val
        benchmark = rebalance.w
        if isa(fees_rebal, Real)
            sum(fees_rebal * abs.(benchmark .- w))
        elseif isa(fees_rebal, AbstractVector) &&
               !(isempty(fees_rebal) || all(iszero.(fees_rebal)))
            dot(fees_rebal, abs.(benchmark .- w))
        else
            zero(eltype(w))
        end
    else
        zero(eltype(w))
    end
end
function calc_fixed_fees(w::AbstractVector, fees::Union{AbstractVector{<:Real}, Real},
                         tol_kwargs::NamedTuple, op::Function)
    return if isa(fees, Real) && !iszero(fees)
        idx1 = op(w, zero(eltype(w)))
        idx2 = .!isapprox.(w[idx1], zero(eltype(w)); tol_kwargs...)
        fees * sum(idx2)
    elseif isa(fees, AbstractVector) && !(isempty(fees) || all(iszero.(fees)))
        idx1 = op(w, zero(eltype(w)))
        idx2 = .!isapprox.(w[idx1], zero(eltype(w)); tol_kwargs...)
        sum(fees[idx1][idx2])
    else
        zero(eltype(w))
    end
end
function calc_fees(w::AbstractVector, fees::Fees = Fees())
    fees_long = calc_fees(w, fees.long, .>=)
    fees_short = calc_fees(w, fees.short, .<)
    fees_fixed_long = calc_fixed_fees(w, fees.fixed_long, fees.tol_kwargs, .>=)
    fees_fixed_short = -calc_fixed_fees(w, fees.fixed_short, fees.tol_kwargs, .<)
    fees_rebal = calc_fees(w, fees.rebalance)
    return fees_long + fees_short + fees_fixed_long + fees_fixed_short + fees_rebal
end
function calc_asset_fees(w::AbstractVector, fees::Union{AbstractVector{<:Real}, Real},
                         op::Function)
    fees_w = zeros(eltype(w), length(w))
    if isa(fees, Real) && !iszero(fees)
        idx = op(w, zero(eltype(w)))
        fees_w[idx] .= fees * w[idx]
    elseif isa(fees, AbstractVector) && !(isempty(fees) || all(iszero.(fees)))
        idx = op(w, zero(eltype(w)))
        fees_w[idx] .= fees[idx] .* w[idx]
    end
    return fees_w
end
function calc_asset_fees(w::AbstractVector, rebalance::AbstractTR)
    fees_w = zeros(eltype(w), length(w))
    if isa(rebalance, TR)
        fees_rebal = rebalance.val
        benchmark = rebalance.w
        if isa(fees_rebal, Real)
            fees_w .= fees_rebal * abs.(benchmark .- w)
        elseif isa(fees_rebal, AbstractVector) &&
               !(isempty(fees_rebal) || all(iszero.(fees_rebal)))
            fees_w .= fees_rebal .* abs.(benchmark .- w)
        end
    end
    return fees_w
end
function calc_asset_fixed_fees(w::AbstractVector, fees::Union{AbstractVector{<:Real}, Real},
                               tol_kwargs::NamedTuple, op::Function)
    fees_w = zeros(eltype(w), length(w))
    if isa(fees, Real) && !iszero(fees)
        idx1 = op(w, zero(eltype(w)))
        idx2 = .!isapprox.(w[idx1], zero(eltype(w)); tol_kwargs...)
        fees_w[idx1] .= fees * idx2
    elseif isa(fees, AbstractVector) && !(isempty(fees) || all(iszero.(fees)))
        idx1 = op(w, zero(eltype(w)))
        idx2 = .!isapprox.(w[idx1], zero(eltype(w)); tol_kwargs...)
        fees_w[idx1] .= fees[idx1][idx2]
    end
    return fees_w
end
function calc_asset_fees(w::AbstractVector, fees::Fees = Fees())
    fees_long = calc_asset_fees(w, fees.long, .>=)
    fees_short = calc_asset_fees(w, fees.short, .<)
    fees_fixed_long = calc_asset_fixed_fees(w, fees.fixed_long, fees.tol_kwargs, .>=)
    fees_fixed_short = -calc_asset_fixed_fees(w, fees.fixed_short, fees.tol_kwargs, .<)
    fees_rebal = calc_asset_fees(w, fees.rebalance)
    return fees_long + fees_short + fees_fixed_long + fees_fixed_short + fees_rebal
end
function calc_net_returns(X::AbstractMatrix, w::AbstractVector, fees::Fees = Fees())
    return X * w .- calc_fees(w, fees)
end
function calc_net_asset_returns(X::AbstractMatrix, w::AbstractVector, fees::Fees = Fees())
    return X .* transpose(w) .- transpose(calc_asset_fees(w, fees))
end
function expected_risk(rm::Union{WR, VaR, VaRRG, CVaR, DRCVaR, EVaR, EVaRRG, RLVaR, RLVaRRG,
                                 DaR, MDD, ADD, CDaR, UCI, EDaR, RLDaR, DaR_r, MDD_r, ADD_r,
                                 CDaR_r, UCI_r, EDaR_r, RLDaR_r, GMD, RG, CVaRRG, TG, TGRG,
                                 OWA, TCM, FTCM, Skewness, SSkewness, Kurtosis, SKurtosis},
                       w::AbstractVector; X::AbstractMatrix, fees::Fees = Fees(), kwargs...)
    return rm(calc_net_returns(X, w, fees))
end
function expected_risk(rm::Union{Kurt, SKurt, SVariance, BDVariance}, w::AbstractVector;
                       X::AbstractMatrix, scale::Bool = false, fees::Fees = Fees(),
                       kwargs...)
    return rm(X, w, fees; scale = scale)
end
function expected_risk(rm::Union{MAD, SSD, FLPM, TLPM, FTLPM, TrackingRM},
                       w::AbstractVector; X::AbstractMatrix, fees::Fees = Fees(), kwargs...)
    return rm(X, w, fees)
end
function expected_risk(rm::Union{SD, NSkew, NSSkew, TurnoverRM}, w::AbstractVector;
                       kwargs...)
    return rm(w)
end
function expected_risk(rm::Union{Variance, WCVariance, NQSkew, NQSSkew}, w::AbstractVector;
                       scale::Bool = false, kwargs...)
    return rm(w; scale = scale)
end

export expected_risk, cumulative_returns, drawdown, calc_fees, calc_fixed_fees,
       calc_asset_fees, calc_asset_fixed_fees, calc_net_returns, calc_net_asset_returns
