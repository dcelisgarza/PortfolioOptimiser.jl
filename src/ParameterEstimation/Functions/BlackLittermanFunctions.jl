"""
```
_mu_cov_w(tau, omega, P, Pi, Q, rf, sigma, delta, T, N, opt, cov_type, cov_flag = true)
```

Internal function for computing the Black Litterman statistics as defined in [`black_litterman`](@ref). See .

# Inputs

  - `tau`: variable of the same name in the Black-Litterman model.
  - `omega`: variable of the same name in the Black-Litterman model.
  - `P`: variable of the same name in the Black-Litterman model.
  - `Pi`: variable of the same name in the Black-Litterman model.
  - `Q`: variable of the same name in the Black-Litterman model.
  - `rf`: variable of the same name in the Black-Litterman model.
  - `sigma`: variable of the same name in the Black-Litterman model.
  - `delta`: variable of the same name in the Black-Litterman model.
  - `T`: variable of the same name in the Black-Litterman model.
  - `N`: variable of the same name in the Black-Litterman model.
  - `opt`: any valid instance of `opt` for .
  - `cov_type`: any valid value from .
  - `cov_flag`: whether the matrix is a covariance matrix or not.

# Outputs

  - `mu`: asset expected returns vector obtained via the Black-Litterman model.
  - `cov_mtx`: asset covariance matrix obtained via the Black-Litterman model.
  - `w`: asset weights obtained via the Black-Litterman model.
  - `Pi_`: equilibrium excess returns after being adjusted by the views.
"""
function _bl_mu_cov_w(tau, omega, P, Pi, Q, rf, sigma, delta, T, N, posdef, denoise, logo)
    inv_tau_sigma = (tau * sigma) \ I
    inv_omega = omega \ I
    Pi_ = ((inv_tau_sigma + transpose(P) * inv_omega * P) \ I) *
          (inv_tau_sigma * Pi + transpose(P) * inv_omega * Q)
    M = (inv_tau_sigma + transpose(P) * inv_omega * P) \ I

    mu = Pi_ .+ rf
    sigma = sigma + M

    posdef_fix!(posdef, sigma)
    denoise!(denoise, posdef, sigma, T / N)
    logo!(logo, posdef, sigma)

    w = ((delta * sigma) \ I) * Pi_

    return mu, sigma, w, Pi_
end
function _omega(P, tau_sigma)
    return Diagonal(P * tau_sigma * transpose(P))
end
function _Pi(eq, delta, sigma, w, mu, rf)
    return eq ? delta * sigma * w : mu .- rf
end
"""
```
black_litterman(bl::BLType, X::AbstractMatrix, P::AbstractMatrix,
                         Q::AbstractVector, w::AbstractVector;
                         cov_type::PortfolioOptimiserCovCor = PortCovCor(),
                         mu_type::MeanEstimator = MuSimple())
```
"""
function black_litterman(bl::BLType, X::AbstractMatrix, P::AbstractMatrix,
                         Q::AbstractVector, w::AbstractVector;
                         cov_type::PortfolioOptimiserCovCor = PortCovCor(),
                         mu_type::MeanEstimator = MuSimple())
    sigma, mu = _sigma_mu(X, cov_type, mu_type)

    T, N = size(X)

    tau = 1 / T
    omega = _omega(P, tau * sigma)
    Pi = _Pi(bl.eq, bl.delta, sigma, w, mu, bl.rf)

    mu, sigma, w = _bl_mu_cov_w(tau, omega, P, Pi, Q, bl.rf, sigma, bl.delta, T, N,
                                bl.posdef, bl.denoise, bl.logo)[1:3]

    return mu, sigma, w
end
function black_litterman_statistics(X, mu, sigma; P::AbstractMatrix, Q::AbstractVector,
                                    w::AbstractVector = Vector{Float64}(undef, 0),
                                    cov_type::PortfolioOptimiserCovCor = PortCovCor(),
                                    mu_type::MeanEstimator = MuSimple(),
                                    bl_type::BLType = BLType())
    if isempty(w)
        w = fill(1 / size(X, 2), size(X, 2))
    end

    if isnothing(bl_type.delta)
        bl_type.delta = (dot(mu, w) - bl_type.rf) / dot(w, sigma, w)
    end

    bl_mu, bl_cov, bl_w = black_litterman(bl_type, X, P, Q, w; cov_type = cov_type,
                                          mu_type = mu_type)

    return w, bl_mu, bl_cov, bl_w
end
function black_litterman(bl::BBLType, X::AbstractMatrix; F::AbstractMatrix,
                         B::AbstractMatrix, P_f::AbstractMatrix, Q_f::AbstractVector,
                         cov_type::PortfolioOptimiserCovCor = PortCovCor(),
                         mu_type::MeanEstimator = MuSimple(), kwargs...)
    f_sigma, f_mu = _sigma_mu(F, cov_type, mu_type)

    f_mu .-= bl.rf

    if bl.constant
        alpha = B[:, 1]
        B = B[:, 2:end]
    end

    T, N = size(X)
    tau = inv(T)

    sigma = B * f_sigma * transpose(B)

    if bl.error
        D = X - F * transpose(B)
        D = Diagonal(vec(if isnothing(bl.var_w)
                             var(bl.ve, D; dims = 1)
                         else
                             var(bl.ve, D, bl.var_w; dims = 1)
                         end))
        sigma .+= D
    end

    omega_f = _omega(P_f, tau * f_sigma)

    inv_sigma = sigma \ I
    inv_sigma_f = f_sigma \ I
    inv_omega_f = omega_f \ I
    tpf_invof = transpose(P_f) * inv_omega_f
    inv_sigma_hat = (inv_sigma_f + tpf_invof * P_f)
    Pi_hat = (inv_sigma_hat \ I) * (inv_sigma_f * f_mu + tpf_invof * Q_f)
    inv_sigma_b = inv_sigma * B
    tb = transpose(B)
    iish_b_is_b = (inv_sigma_hat + tb * inv_sigma_b) \ I
    is_b_iish_b_is_b = inv_sigma_b * iish_b_is_b
    sigma_bbl = (inv_sigma - is_b_iish_b_is_b * tb * inv_sigma) \ I
    Pi_bbl = sigma_bbl * is_b_iish_b_is_b * inv_sigma_hat * Pi_hat
    mu = Pi_bbl .+ bl.rf

    if bl.constant
        mu .+= alpha
    end

    w = ((bl.delta * sigma_bbl) \ I) * mu

    posdef_fix!(bl.posdef, sigma)
    denoise!(bl.denoise, bl.posdef, sigma, T / N)
    logo!(bl.logo, bl.posdef, sigma)

    return mu, sigma_bbl, w
end
function black_litterman(bl::ABLType, X::AbstractMatrix; w::AbstractVector,
                         F::Union{AbstractMatrix, Nothing}    = nothing,
                         B::Union{AbstractMatrix, Nothing}    = nothing,
                         P::Union{AbstractMatrix, Nothing}    = nothing,
                         P_f::Union{AbstractMatrix, Nothing}  = nothing,
                         Q::Union{AbstractVector, Nothing}    = nothing,
                         Q_f::Union{AbstractVector, Nothing}  = nothing,
                         cov_type::PortfolioOptimiserCovCor   = PortCovCor(;),
                         mu_type::MeanEstimator               = MuSimple(;),
                         f_cov_type::PortfolioOptimiserCovCor = PortCovCor(;),
                         f_mu_type::MeanEstimator             = MuSimple(;))
    asset_tuple = (!isnothing(P), !isnothing(Q))
    any_asset_provided = any(asset_tuple)
    all_asset_provided = all(asset_tuple)
    @smart_assert(any_asset_provided == all_asset_provided,
                  "If any of P or Q is provided, then both must be provided.")

    factor_tuple = (!isnothing(P_f), !isnothing(Q_f))
    any_factor_provided = any(factor_tuple)
    all_factor_provided = all(factor_tuple)
    @smart_assert(any_factor_provided == all_factor_provided,
                  "If any of P_f or Q_f is provided, then both must be provided.")

    if all_factor_provided
        @smart_assert(!isnothing(B) && !isnothing(F),
                      "If P_f and Q_f are provided, then B and F must be provided.")
    end

    if !all_asset_provided && !all_factor_provided
        throw(AssertionError("Please provide either:\n- P and Q,\n- B, F, P_f and Q_f, or\n- P, Q, B, F, P_f and Q_f."))
    end

    if all_asset_provided
        sigma, mu = _sigma_mu(X, cov_type, mu_type)
    end

    if all_factor_provided
        f_sigma, f_mu = _sigma_mu(F, f_cov_type, f_mu_type)
    end

    if all_factor_provided && bl.constant
        alpha = B[:, 1]
        B = B[:, 2:end]
    end

    T, N = size(X)

    tau = 1 / T

    if all_asset_provided && !all_factor_provided
        sigma_a = sigma
        P_a = P
        Q_a = Q
        omega_a = _omega(P_a, tau * sigma_a)
        Pi_a = _Pi(bl.eq, bl.delta, sigma_a, w, mu, bl.rf)
    elseif !all_asset_provided && all_factor_provided
        sigma_a = f_sigma
        P_a = P_f
        Q_a = Q_f
        omega_a = _omega(P_a, tau * sigma_a)
        Pi_a = _Pi(bl.eq, bl.delta, sigma_a * transpose(B), w, f_mu, bl.rf)
    elseif all_asset_provided && all_factor_provided
        sigma_a = hcat(vcat(sigma, f_sigma * transpose(B)), vcat(B * f_sigma, f_sigma))

        zeros_1 = zeros(size(P_f, 1), size(P, 2))
        zeros_2 = zeros(size(P, 1), size(P_f, 2))

        P_a = hcat(vcat(P, zeros_1), vcat(zeros_2, P_f))
        Q_a = vcat(Q, Q_f)

        omega = _omega(P, tau * sigma)
        omega_f = _omega(P_f, tau * f_sigma)

        zeros_3 = zeros(size(omega, 1), size(omega_f, 1))

        omega_a = hcat(vcat(omega, transpose(zeros_3)), vcat(zeros_3, omega_f))

        Pi_a = _Pi(bl.eq, bl.delta, vcat(sigma, f_sigma * transpose(B)), w, vcat(mu, f_mu),
                   bl.rf)
    end

    mu_a, sigma_a, w_a, Pi_a_ = _bl_mu_cov_w(tau, omega_a, P_a, Pi_a, Q_a, bl.rf, sigma_a,
                                             bl.delta, T, N, bl.posdef, bl.denoise, bl.logo)

    if !all_asset_provided && all_factor_provided
        mu_a = B * mu_a
        sigma_a = B * sigma_a * transpose(B)
        posdef_fix!(bl.posdef, sigma_a)
        denoise!(bl.denoise, bl.posdef, sigma_a, T / N)
        logo!(bl.logo, bl.posdef, sigma_a)
        w_a = ((bl.delta * sigma_a) \ I) * B * Pi_a_
    end

    if all_factor_provided && bl.constant
        mu_a = mu_a[1:N] .+ alpha
    end

    return mu_a[1:N], sigma_a[1:N, 1:N], w_a[1:N]
end

function black_litterman_factor_statistics(assets, X, mu, sigma, f_assets, F;
                                           w::AbstractVector = Vector{Float64}(undef, 0),
                                           B::Union{DataFrame, Nothing} = nothing,
                                           P::Union{AbstractMatrix, Nothing} = nothing,
                                           P_f::Union{AbstractMatrix, Nothing} = nothing,
                                           Q::Union{AbstractVector, Nothing} = nothing,
                                           Q_f::Union{AbstractVector, Nothing} = nothing,
                                           factor_type::FactorType = FactorType(),
                                           cov_type::PortfolioOptimiserCovCor = PortCovCor(),
                                           mu_type::MeanEstimator = MuSimple(),
                                           f_cov_type::PortfolioOptimiserCovCor = PortCovCor(),
                                           f_mu_type::MeanEstimator = MuSimple(),
                                           bl_type::BlackLittermanFactor = BBLType())
    if isempty(w)
        w = fill(1 / size(X, 2), size(X, 2))
    end

    if isnothing(bl_type.delta)
        bl_type.delta = (dot(mu, w) - bl_type.rf) / dot(w, sigma, w)
    end

    if isnothing(B) || isempty(B)
        B = regression(factor_type.type, DataFrame(F, f_assets), DataFrame(X, assets))
    end

    namesB = names(B)
    bl_type.constant = "const" ∈ namesB

    blfm_mu, blfm_cov, blfm_w = black_litterman(bl_type, X; w = w, F = F,
                                                B = Matrix(B[!,
                                                             setdiff(namesB, ("tickers",))]),
                                                P = P, P_f = P_f, Q = Q, Q_f = Q_f,
                                                cov_type = cov_type, mu_type = mu_type,
                                                f_cov_type = f_cov_type,
                                                f_mu_type = f_mu_type)
    return w, B, blfm_mu, blfm_cov, blfm_w
end

export black_litterman, black_litterman_statistics, black_litterman_factor_statistics
