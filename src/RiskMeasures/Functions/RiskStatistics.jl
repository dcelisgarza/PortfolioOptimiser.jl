function risk_bounds(rm::RiskMeasure, w1::AbstractVector, w2::AbstractVector;
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
risk_contribution(rm::RiskMeasure, w::AbstractVector;
                           X::AbstractMatrix = Matrix{Float64}(undef, 0, 0),
                           V::AbstractMatrix = Matrix{Float64}(undef, 0, 0),
                           SV::AbstractMatrix = Matrix{Float64}(undef, 0, 0),
                           delta::Real = 1e-6, marginal::Bool = false, kwargs...)
```
"""
function risk_contribution(rm::RiskMeasure, w::AbstractVector;
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

        rci = if !marginal
            (r1 - r2) / (2 * delta) * w[i]
        else
            (r1 - r2) / (2 * delta)
        end
        rc[i] = rci
    end
    return rc
end
function factor_risk_contribution(rm::RiskMeasure, w::AbstractVector;
                                  X::AbstractMatrix = Matrix{Float64}(undef, 0, 0),
                                  assets::AbstractVector = Vector{String}(undef, 0),
                                  F::AbstractMatrix = Matrix{Float64}(undef, 0, 0),
                                  f_assets::AbstractVector = Vector{String}(undef, 0),
                                  B::DataFrame = DataFrame(),
                                  loadings_opt::RegressionType = FReg(),
                                  V::AbstractMatrix = Matrix{Float64}(undef, 0, 0),
                                  SV::AbstractMatrix = Matrix{Float64}(undef, 0, 0),
                                  delta::Real = 1e-6, kwargs...)
    marginal_risk = risk_contribution(rm, w; X = X, V = V, SV = SV, delta = delta,
                                      marginal = true, kwargs...)

    if isempty(B)
        B = regression(loadings_opt, DataFrame(F, f_assets), DataFrame(X, assets))
    end
    b1, b2, b3, B = _factors_b1_b2_b3(B, F, loadings_opt)

    rc_f = (transpose(B) * w) .* (transpose(b1) * marginal_risk)
    rc_of = sum((transpose(b2) * w) .* (transpose(b3) * marginal_risk))
    rc_f = [rc_f; rc_of]

    return rc_f
end
function sharpe_ratio(rm::RiskMeasure, w::AbstractVector;
                      mu::AbstractVector = Vector{Float64}(undef, 0),
                      X::AbstractMatrix = Matrix{Float64}(undef, 0, 0),
                      V::AbstractMatrix = Matrix{Float64}(undef, 0, 0),
                      SV::AbstractMatrix = Matrix{Float64}(undef, 0, 0), delta::Real = 1e-6,
                      rf::Real = 0.0, kelly::Bool = false)
    ret = if kelly
        sum(log.(one(eltype(X)) .+ X * w)) / size(X, 1)
    else
        dot(mu, w)
    end
    risk = calc_risk(rm, w; X = X, V = V, SV = SV, delta = delta)
    return (ret - rf) / risk
end

export risk_bounds, risk_contribution, factor_risk_contribution, sharpe_ratio
