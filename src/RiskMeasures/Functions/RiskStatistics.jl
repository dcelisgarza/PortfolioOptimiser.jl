"""
```
risk_bounds(rm::AbstractRiskMeasure, w1::AbstractVector, w2::AbstractVector;
            X::AbstractMatrix = Matrix{Float64}(undef, 0, 0),
            V::AbstractMatrix = Matrix{Float64}(undef, 0, 0),
            SV::AbstractMatrix = Matrix{Float64}(undef, 0, 0), delta::Real = 1e-6,
            scale::Bool = false, kwargs...)
```

Compute the risk bounds for an [`AbstractRiskMeasure`](@ref) and pair of asset weight vectors.

# Inputs

  - `rm`: risk measure.
  - `w1`: `N×1` vector of asset weights for lower bound.
  - `w2`: `N×1` vector of asset weights for upper bound.
  - `X`: `T×N` matrix of asset returns.
  - `V`: `N×N` matrix of sum of negative spectral slices of the coskewness.
  - `SV`: `N×N` matrix of sum of negative spectral slices of the semi coskewness.
  - `scale`: if true divides the kurtosis and semi kurtosis by 2, used in [`risk_contribution`](@ref).
  - `kwargs`: catch-all for any missing keyword arguments for [`calc_risk`](@ref).

# Outputs

  - `r1`: lower risk bound.
  - `r2`: upper risk bound.
"""
function risk_bounds(rm::AbstractRiskMeasure, w1::AbstractVector, w2::AbstractVector;
                     X::AbstractMatrix = Matrix{Float64}(undef, 0, 0),
                     V::AbstractMatrix = Matrix{Float64}(undef, 0, 0),
                     SV::AbstractMatrix = Matrix{Float64}(undef, 0, 0), delta::Real = 1e-6,
                     scale::Bool = false, kwargs...)
    r1 = calc_risk(rm, w1; X = X, V = V, SV = SV, delta = delta, scale = scale, kwargs...)
    r2 = calc_risk(rm, w2; X = X, V = V, SV = SV, delta = -delta, scale = scale, kwargs...)
    return r1, r2
end

"""
```
risk_contribution(rm::AbstractRiskMeasure, w::AbstractVector;
                  X::AbstractMatrix = Matrix{Float64}(undef, 0, 0),
                  V::AbstractMatrix = Matrix{Float64}(undef, 0, 0),
                  SV::AbstractMatrix = Matrix{Float64}(undef, 0, 0),
                  delta::Real = 1e-6, marginal::Bool = false, kwargs...)
```

Compute the asset risk contribution for an [`AbstractRiskMeasure`](@ref) and asset weight vector.

# Inputs

  - `rm`: risk measure.

  - `w`: `N×1` vector of asset weights.
  - `X`: `T×N` matrix of asset returns.
  - `V`: `N×N` matrix of sum of negative spectral slices of the coskewness.
  - `SV`: `N×N` matrix of sum of negative spectral slices of the semi coskewness.
  - `delta`: small displacement used for computing the marginal risk.
  - `marginal`:

      + if `true`: compute the marginal risk contribution.
      + if `false`: compute the risk by contribution by multiplying the marginal risk by the asset weight.
  - `kwargs`: catch-all for any missing keyword arguments for [`calc_risk`](@ref).

# Outputs

  - `rc`: `N×1` vector of risk contribution per asset.
"""
function risk_contribution(rm::AbstractRiskMeasure, w::AbstractVector;
                           X::AbstractMatrix = Matrix{Float64}(undef, 0, 0),
                           V::AbstractMatrix = Matrix{Float64}(undef, 0, 0),
                           SV::AbstractMatrix = Matrix{Float64}(undef, 0, 0),
                           delta::Real = 1e-6, marginal::Bool = false, kwargs...)
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

        r1, r2 = risk_bounds(rm, w1, w2; X = X, V = V, SV = SV, delta = delta, scale = true,
                             kwargs...)

        rci = (r1 - r2) / (2 * delta)
        if !marginal
            rci *= w[i]
        end
        rc[i] = rci
    end
    return rc
end

"""
```
factor_risk_contribution(rm::AbstractRiskMeasure, w::AbstractVector;
                         X::AbstractMatrix = Matrix{Float64}(undef, 0, 0),
                         assets::AbstractVector = Vector{String}(undef, 0),
                         F::AbstractMatrix = Matrix{Float64}(undef, 0, 0),
                         f_assets::AbstractVector = Vector{String}(undef, 0),
                         B::DataFrame = DataFrame(),
                         regression_type::RegressionType = FReg(),
                         V::AbstractMatrix = Matrix{Float64}(undef, 0, 0),
                         SV::AbstractMatrix = Matrix{Float64}(undef, 0, 0),
                         delta::Real = 1e-6, kwargs...)
```

Compute the factor risk contribution for an [`AbstractRiskMeasure`](@ref) and asset weight vector.

# Inputs

  - `rm`: risk measure.
  - `w`: `Na×1` vector of asset weights.
  - `X`: `T×Na` matrix of asset returns.
  - `assets`: vector of asset names.
  - `F`: `T×Nf` matrix of factor returns.
  - `f_assets`: vector of factor names.
  - `B`: loadings matrix.
  - `regression_type`: regression type used for computing the loadings matrix.
  - `V`: `Na×Na` matrix of sum of negative spectral slices of the coskewness.
  - `SV`: `Na×Na` matrix of sum of negative spectral slices of the semi coskewness.
  - `delta`: small displacement used for computing the marginal risk.
  - `kwargs`: catch-all for any missing keyword arguments for [`calc_risk`](@ref).

# Outputs

  - `rc_f`: `Nf×1` vector of risk contribution per factor.
"""
function factor_risk_contribution(rm::AbstractRiskMeasure, w::AbstractVector;
                                  X::AbstractMatrix = Matrix{Float64}(undef, 0, 0),
                                  assets::AbstractVector = Vector{String}(undef, 0),
                                  F::AbstractMatrix = Matrix{Float64}(undef, 0, 0),
                                  f_assets::AbstractVector = Vector{String}(undef, 0),
                                  B::DataFrame = DataFrame(),
                                  regression_type::RegressionType = FReg(),
                                  V::AbstractMatrix = Matrix{Float64}(undef, 0, 0),
                                  SV::AbstractMatrix = Matrix{Float64}(undef, 0, 0),
                                  delta::Real = 1e-6, kwargs...)
    marginal_risk = risk_contribution(rm, w; X = X, V = V, SV = SV, delta = delta,
                                      marginal = true, kwargs...)

    if isempty(B)
        B = regression(regression_type, DataFrame(F, f_assets), DataFrame(X, assets))
    end
    b1, b2, b3, B = _factors_b1_b2_b3(B, F, regression_type)

    rc_f = (transpose(B) * w) .* (transpose(b1) * marginal_risk)
    rc_of = sum((transpose(b2) * w) .* (transpose(b3) * marginal_risk))
    rc_f = [rc_f; rc_of]

    return rc_f
end

"""
```
sharpe_ratio(rm::AbstractRiskMeasure, w::AbstractVector;
             mu::AbstractVector = Vector{Float64}(undef, 0),
             X::AbstractMatrix = Matrix{Float64}(undef, 0, 0),
             V::AbstractMatrix = Matrix{Float64}(undef, 0, 0),
             SV::AbstractMatrix = Matrix{Float64}(undef, 0, 0), delta::Real = 1e-6,
             rf::Real = 0.0, kelly::Bool = false)
```

Compute the risk-adjusted return ratio for an [`AbstractRiskMeasure`](@ref) and asset weights vector.

# Inputs

  - `rm`: risk measure.

  - `w`: `N×1` vector of asset weights.
  - `X`: `T×N` matrix of asset returns.
  - `V`: `N×N` matrix of sum of negative spectral slices of the coskewness.
  - `SV`: `N×N` matrix of sum of negative spectral slices of the semi coskewness.
  - `delta`: small displacement used for computing the [`Equal`](@ref) risk measure.
  - `rf`: risk free rate.
  - `kelly`:

      + if `true`: use the kelly return.
      + if `false`: use the arithmetic return.
"""
function sharpe_ratio(rm::AbstractRiskMeasure, w::AbstractVector;
                      mu::AbstractVector = Vector{Float64}(undef, 0),
                      X::AbstractMatrix = Matrix{Float64}(undef, 0, 0),
                      V::AbstractMatrix = Matrix{Float64}(undef, 0, 0),
                      SV::AbstractMatrix = Matrix{Float64}(undef, 0, 0), delta::Real = 1e-6,
                      rf::Real = 0.0, kelly::Bool = false)
    ret = if kelly
        1 / size(X, 1) * sum(log.(one(eltype(X)) .+ X * w))
    else
        dot(mu, w)
    end
    risk = calc_risk(rm, w; X = X, V = V, SV = SV, delta = delta)
    return (ret - rf) / risk
end

export risk_bounds, risk_contribution, factor_risk_contribution, sharpe_ratio
