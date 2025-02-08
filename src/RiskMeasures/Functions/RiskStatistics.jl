# Copywrite (c) 2025
# Author: Daniel Celis Garza <daniel.celis.garza@gmail.com>
# SPDX-License-Identifier: MIT

"""
    risk_bounds(rm::AbstractRiskMeasure, w1::AbstractVector, w2::AbstractVector;
                X::AbstractMatrix = Matrix{Float64}(undef, 0, 0),
                V::AbstractMatrix = Matrix{Float64}(undef, 0, 0),
                SV::AbstractMatrix = Matrix{Float64}(undef, 0, 0), delta::Real = 1e-6,
                scale::Bool = false, kwargs...)

# Description

Compute the risk bounds for an [`AbstractRiskMeasure`](@ref) and pair of asset weight vectors.

See also: [`AbstractRiskMeasure`](@ref), [`calc_risk`](@ref).

# Inputs

## Positional

  - `rm::AbstractRiskMeasure`: risk measure.
  - `w1::AbstractVector`: `N×1` vector of asset weights for lower bound.
  - `w2::AbstractVector`: `N×1` vector of asset weights for upper bound.

## Named

  - `X::AbstractMatrix = Matrix{Float64}(undef, 0, 0)`: `T×N` matrix of asset returns.
  - `V::AbstractMatrix = Matrix{Float64}(undef, 0, 0)`: `N×N` matrix of the sum of negative spectral slices of the coskewness.
  - `SV::AbstractMatrix = Matrix{Float64}(undef, 0, 0)`: `N×N` matrix of the sum of negative spectral slices of the semi coskewness.
  - `delta::Real = 1e-6`: small displacement used for computing the [`Equal`](@ref) risk measure.
  - `scale::Bool = false`: flag for scaling the kurtosis and semi kurtosis in [`risk_contribution`](@ref).

# Outputs

  - `r1::Real`: lower risk bound/risk corresponding to `w1`.
  - `r2::Real`: upper risk bound/risk corresponding to `w2`.

# Examples

```@example
# Sample returns matrix
returns = [ 0.19 -0.41 -0.70;
            1.15 -1.20 -1.27;
           -0.27 -1.98 -0.77;
           -0.65  0.22  0.59;
           -0.04  0.35 -0.99]

# Sample weights vector
w1 = [0.7, 0.2, 0.1]
w2 = [0.3, 0.5, 0.2]

# Calculate the risk bounds for the default conditional value at risk
r1, r2 = risk_bounds(CVaR(), w1, w2; X = returns)
```
"""
function risk_bounds(rm::AbstractRiskMeasure, w1::AbstractVector, w2::AbstractVector;
                     X::AbstractMatrix = Matrix{Float64}(undef, 0, 0), delta::Real = 1e-6,
                     scale::Bool = false, fees::Fees = Fees(),
                     rebalance::AbstractTR = NoTR(), kwargs...)
    r1 = calc_risk(rm, w1; X = X, delta = delta, scale = scale, fees = fees,
                   rebalance = rebalance, kwargs...)
    r2 = calc_risk(rm, w2; X = X, delta = -delta, scale = scale, fees = fees,
                   rebalance = rebalance, kwargs...)
    return r1, r2
end

"""
    risk_contribution(rm::AbstractRiskMeasure, w::AbstractVector;
                      X::AbstractMatrix = Matrix{Float64}(undef, 0, 0),
                      delta::Real = 1e-6, marginal::Bool = false, 
                      kwargs...)

# Description

Compute the asset risk contribution for an [`AbstractRiskMeasure`](@ref) and asset weight vector.

See also: [`AbstractRiskMeasure`](@ref), [`risk_bounds`](@ref), [`calc_risk`](@ref).

# Inputs

## Positional

  - `rm::AbstractRiskMeasure`: risk measure.
  - `w1::AbstractVector`: `N×1` vector of asset weights for lower bound.
  - `w2::AbstractVector`: `N×1` vector of asset weights for upper bound.

## Named

  - `X::AbstractMatrix = Matrix{Float64}(undef, 0, 0)`: `T×N` matrix of asset returns.

  - `delta::Real = 1e-6`: small displacement used for computing the [`Equal`](@ref) risk measure.
  - `marginal::Bool = false`:

      + if `true`: compute the marginal risk contribution.
      + else: compute the risk contribution by multiplying the marginal risk by the asset weight.

# Outputs

  - `rc::AbstractVector`: `N×1` vector of risk contribution per asset.

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

# Calculate the risk bounds for the default conditional value at risk
rc = risk_contribution(CVaR(), w; X = returns)
```
"""
function risk_contribution(rm::AbstractRiskMeasure, w::AbstractVector;
                           X::AbstractMatrix = Matrix{Float64}(undef, 0, 0),
                           delta::Real = 1e-6, marginal::Bool = false, fees::Fees = Fees(),
                           rebalance::AbstractTR = NoTR(), kwargs...)
    N = length(w)
    ew = eltype(w)
    rc = Vector{ew}(undef, N)
    w1 = Vector{ew}(undef, N)
    w2 = Vector{ew}(undef, N)

    for i ∈ eachindex(w)
        w1 .= zero(ew)
        w1 .= w
        w1[i] += delta

        w2 .= zero(ew)
        w2 .= w
        w2[i] -= delta

        r1, r2 = risk_bounds(rm, w1, w2; X = X, delta = delta, scale = true, fees = fees,
                             rebalance = rebalance, kwargs...)

        rci = (r1 - r2) / (2 * delta)
        if !marginal
            rci *= w[i]
        end
        rc[i] = rci
    end
    return rc
end

"""
    factor_risk_contribution(rm::AbstractRiskMeasure, w::AbstractVector;
                            X::AbstractMatrix = Matrix{Float64}(undef, 0, 0),
                            assets::AbstractVector = Vector{String}(undef, 0),
                            F::AbstractMatrix = Matrix{Float64}(undef, 0, 0),
                            f_assets::AbstractVector = Vector{String}(undef, 0),
                            B::DataFrame = DataFrame(),
                            regression_type::RegressionType = FReg(),
                            delta::Real = 1e-6, kwargs...)

# Description

Compute the factor risk contribution for an [`AbstractRiskMeasure`](@ref), assets, and factors.

See also: [`AbstractRiskMeasure`](@ref), [`risk_bounds`](@ref), [`calc_risk`](@ref), [`regression`](@ref), [`RegressionType`](@ref).

# Inputs

## Positional

  - `rm::AbstractRiskMeasure`: risk measure [`AbstractRiskMeasure`](@ref).
  - `w::AbstractVector`: `Na×1` vector of asset weights.

## Named

  - `X::AbstractMatrix = Matrix{Float64}(undef, 0, 0)`: `T×Na` matrix of asset returns.
  - `assets::AbstractVector = Vector{String}(undef, 0)`: `Na×1` vector of asset names.
  - `F::AbstractMatrix = Matrix{Float64}(undef, 0, 0)`: `T×Nf` matrix of factor returns.
  - `f_assets::AbstractVector = Vector{String}(undef, 0)`: `Nf×` vector of factor names.
  - `B::DataFrame = DataFrame()`: optional `Na×Nf` loadings matrix.
  - `regression_type::RegressionType = FReg()`: regression type used for computing the loadings matrix.
  - `delta::Real = 1e-6`: small displacement used for computing the marginal risk and equal risk measure [`Equal`](@ref).

# Behaviour

  - If `B` is empty: computes the loadings matrix, `B`, via [`regression`](@ref) using `regression_type`, `F`, `f_assets`, `X`, `assets`.

# Outputs

  - `rc_f::AbstractVector`: `Nf×1` vector of risk contribution per factor.

# Examples

```@example
# Sample returns matrix
returns = [ 0.57 -0.54 -0.15  0.42 -1.78;
           -0.12 -0.02 -0.04  1.61  0.74;
            0.09  2.58 -0.40 -0.36 -1.07;
           -1.33 -1.13  0.35 -0.89 -1.01;
            0.59 -1.48  0.01 -1.79 -0.18;
            0.22  0.39 -0.04  0.22  0.64;
            2.39  0.05 -0.24  0.76 -0.08;
           -0.22 -0.66  0.51  1.21 -0.36;
            1.56 -0.39  2.13  0.01  1.77;
            0.88  0.49 -1.24  1.24  0.71]

# Asset names
assets = ["A1", "A2", "A3", "A4", "A5"]

# Sample factor returns matrix
f_returns = [ 3.00  1.55;
              0.52 -0.57;
             -0.27  0.71;
              0.30  0.00;
              2.27  1.10;
              0.23  1.33;
              0.64  0.00;
              0.35  1.02;
             -1.33 -1.60;
              0.49 -1.66]

# Factor names
f_assets = ["F1", "F2"]

# Sample asset weights vector
w = [0.15, 0.1, 0.3, 0.2, 0.25]

# Risk measure
cvar_rm = CVaR()

# Compute the risk factor contribution by computing the loadings matrix using
# the default parameters.
fc1 = factor_risk_contribution(cvar_rm, w; X = returns, assets = assets, F = f_returns,
                               f_assets = f_assets)

# Compute the risk factor contribution by computing the loadings matrix using
# a different regression type.
fc2 = factor_risk_contribution(cvar_rm, w; X = returns, assets = assets, F = f_returns,
                               f_assets = f_assets, regression_type = BReg())

# Provide the loadings matrix directly.
B = DataFrame(:tickers => assets,
              :const => [0.019628056331070173, -0.4630372691196401, -0.051116594784858346,
                         0.6244845397620361, -0.46039779836908995],
              :F1 => [-0.5382419657683666, 0.0, -0.5447539204822568, 0.0,
                      -0.4713171689983393],
              :F2 => [0.0, -0.48659032199680796, 0.44882375725309853, -0.23791307331955935,
                      0.0])

fc3 = factor_risk_contribution(cvar_rm, w; X = returns, F = f_returns, B = B)
```
"""
function factor_risk_contribution(rm::AbstractRiskMeasure, w::AbstractVector;
                                  X::AbstractMatrix = Matrix{Float64}(undef, 0, 0),
                                  assets::AbstractVector = Vector{String}(undef, 0),
                                  F::AbstractMatrix = Matrix{Float64}(undef, 0, 0),
                                  f_assets::AbstractVector = Vector{String}(undef, 0),
                                  B::DataFrame = DataFrame(),
                                  regression_type::RegressionType = FReg(),
                                  delta::Real = 1e-6, fees::Fees = Fees(),
                                  rebalance::AbstractTR = NoTR(), kwargs...)
    marginal_risk = risk_contribution(rm, w; X = X, delta = delta, marginal = true,
                                      scale = true, fees = fees, rebalance = rebalance,
                                      kwargs...)

    if isempty(B)
        B = regression(regression_type, DataFrame(F, f_assets), DataFrame(X, assets))
    end
    b1, b2, b3, B = factors_b1_b2_b3(B, F, regression_type)

    rc_f = (transpose(B) * w) .* (transpose(b1) * marginal_risk)
    rc_of = sum((transpose(b2) * w) .* (transpose(b3) * marginal_risk))
    rc_f = [rc_f; rc_of]

    return rc_f
end

"""
    sharpe_ratio(rm::AbstractRiskMeasure, w::AbstractVector;
                mu::AbstractVector = Vector{Float64}(undef, 0),
                X::AbstractMatrix = Matrix{Float64}(undef, 0, 0),
                delta::Real = 1e-6, rf::Real = 0.0, 
                kelly::Bool = false)

# Description

Compute the risk-adjusted return ratio for an [`AbstractRiskMeasure`](@ref) and asset weights vector.

# Inputs

  - `rm`: risk measure [`AbstractRiskMeasure`](@ref).

  - `w`: `N×1` vector of asset weights.
  - `mu`: `N×1` vector of expected returns.
  - `X`: `T×N` matrix of asset returns.
  - `V`: `N×N` matrix of the sum of negative spectral slices of the coskewness.
  - `SV`: `N×N` matrix of the sum of negative spectral slices of the semi coskewness.
  - `delta`: small displacement used for computing the [`Equal`](@ref) risk measure.
  - `rf`: risk free rate.
  - `kelly`:

      + if `true`: use the kelly return.
      + else: use the arithmetic return.

# Outputs

  - `sr`: risk adjusted return ratio.
"""
function sharpe_ratio(rm::AbstractRiskMeasure, w::AbstractVector;
                      mu::AbstractVector = Vector{Float64}(undef, 0),
                      X::AbstractMatrix = Matrix{Float64}(undef, 0, 0), delta::Real = 1e-6,
                      rf::Real = 0.0, kelly::Bool = false, fees::Fees = Fees(),
                      rebalance::AbstractTR = NoTR())
    ret = if !kelly
        dot(mu, w) - calc_fees(w, fees, rebalance)
    else
        sum(log.(one(eltype(X)) .+ X * w)) / size(X, 1) - calc_fees(w, fees, rebalance)
    end
    risk = calc_risk(rm, w; X = X, delta = delta, fees = fees, rebalance = rebalance)
    return (ret - rf) / risk
end
function sharpe_ratio_info_criteria(rm::AbstractRiskMeasure, w::AbstractVector;
                                    mu::AbstractVector = Vector{Float64}(undef, 0),
                                    X::AbstractMatrix = Matrix{Float64}(undef, 0, 0),
                                    delta::Real = 1e-6, rf::Real = 0.0, kelly::Bool = false,
                                    fees::Fees = Fees(), rebalance::AbstractTR = NoTR())
    T, N = size(X)
    sr = sharpe_ratio(rm, w; mu = mu, X = X, delta = delta, rf = rf, kelly = kelly,
                      fees = fees, rebalance = rebalance)
    return sr - N / (T * sr)
end

export risk_bounds, risk_contribution, factor_risk_contribution, sharpe_ratio,
       sharpe_ratio_info_criteria
