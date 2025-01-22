# Copywrite (c) 2025
# Author: Daniel Celis Garza <daniel.celis.garza@gmail.com>
# SPDX-License-Identifier: MIT

"""
```
asset_statistics!(port::AbstractPortfolio;
                  cov_type::PortfolioOptimiserCovCor = PortCovCor(),
                  set_cov::Bool = true, mu_type::MeanEstimator = MuSimple(),
                  set_mu::Bool = true, kurt_type::KurtFull = KurtFull(),
                  set_kurt::Bool = true, skurt_type::KurtSemi = KurtSemi(),
                  set_skurt::Bool = true, skew_type::SkewFull = SkewFull(),
                  set_skew::Bool = true, sskew_type::SkewSemi = SkewSemi(),
                  set_sskew::Bool = true,
                  cor_type::PortfolioOptimiserCovCor = PortCovCor(),
                  set_cor::Bool = true,
                  dist_type::DistType = DistCanonical(),
                  set_dist::Bool = true)
```

Compute the asset statistics for a portfolio. See the argument types' docs for details. If a statistic requires another to be computed, the funciton will do so from the relevant estimator.

The `set_*` variables are flags for deciding whether or not to set the statistic. If a statistic's flag is `false` the statistic will not be set. Furthermore, if the flag is `false` _and_ the statistic is not required by another one, it will not be computed.

# Inputs

  - `port`: portfolio [`AbstractPortfolio`](@ref).
  - `cov_type`: covariance estimator [`PortfolioOptimiserCovCor`](@ref).
  - `set_cov`: flag for setting `port.cov`
  - `mu_type`: expected returns estimator [`MeanEstimator`](@ref).
  - `set_mu`: flag for setting `port.mu`
  - `kurt_type`: cokurtosis matrix estimator [`KurtFull`](@ref).
  - `set_kurt`: flag for setting `port.kurt`.
  - `skurt_type`: cokurtosis matrix estimator [`KurtSemi`](@ref).
  - `set_skurt`: flag for setting `port.skurt`.
  - `skew_type`: coskew estimator [`SkewFull`](@ref).
  - `set_skew`: set `port.skew` and `port.V`.
  - `sskew_type`: semi coskew estimator [`SkewSemi`](@ref).
  - `set_sskew`: set `port.skew` and `port.SV`.

## Only relevant for .

  - `cor_type`: correlation matrix estimator [`PortfolioOptimiserCovCor`](@ref).
  - `set_cor`: flag for setting `port.cor`.
  - `dist_type`: type for computing the distance matrix [`DistType`](@ref). [`asset_statistics!`](@ref) uses [`default_dist`](@ref) to ensure the computed distance is consistent with `dist_type` and either `cor_type.ce` or `cor_type` whichever is applicable.
  - `set_dist`: flag for setting `port.dist`.
"""
function asset_statistics!(port::AbstractPortfolio;
                           cov_type::PortfolioOptimiserCovCor = PortCovCor(),
                           set_cov::Bool = true, mu_type::MeanEstimator = MuSimple(),
                           set_mu::Bool = true, kurt_type::KurtFull = KurtFull(),
                           set_kurt::Bool = true, skurt_type::KurtSemi = KurtSemi(),
                           set_skurt::Bool = true, skew_type::SkewFull = SkewFull(),
                           set_skew::Bool = true, sskew_type::SkewSemi = SkewSemi(),
                           set_sskew::Bool = true,
                           cor_type::PortfolioOptimiserCovCor = PortCovCor(),
                           set_cor::Bool = true, dist_type::DistType = DistCanonical(),
                           set_dist::Bool = true)
    returns = port.returns

    if set_cov ||
       set_mu &&
       isa(mu_type, MeanSigmaEstimator) &&
       (isnothing(mu_type.sigma) || isempty(mu_type.sigma))
        sigma = Matrix(cov(cov_type, returns))
        if set_cov
            port.cov = sigma
        end
    end
    if set_mu || set_kurt || set_skurt || set_skew || set_sskew
        old_sigma = set_mean_sigma!(mu_type, @isdefined(sigma) ? sigma : nothing)
        mu = vec(mean(mu_type, returns))
        if set_mu
            port.mu = mu
        end
        unset_mean_sigma!(mu_type, old_sigma)
    end
    if set_kurt || set_skurt
        if set_kurt
            port.kurt = cokurt(kurt_type, returns, mu)
        end
        if set_skurt
            port.skurt = cokurt(skurt_type, returns, mu)
        end
        port.L_2, port.S_2 = dup_elim_sum_matrices(size(returns, 2))[2:3]
    end
    if set_skew
        port.skew, port.V = coskew(skew_type, returns, mu)
    end
    if set_sskew
        port.sskew, port.SV = coskew(sskew_type, returns, mu)
    end
    if set_cor || set_dist
        rho = Matrix(cor(cor_type, returns))
        if set_cor
            port.cor = rho
        end
    end
    if set_dist
        dist_type = default_dist(dist_type, cor_type)
        port.dist = dist(dist_type, rho, returns)
    end

    return nothing
end

"""
```
wc_statistics!(port::Portfolio; wc_type::WCType = WCType(), set_box::Bool = true,
                        set_ellipse::Bool = true)
```

Compute the worst case mean-variance statistics. Only used in  optimisations. The `set_*` variables are used to compute and set the relevant statistics. See the argument types' docs for details.

# Inputs

  - `port`: portfolio [`Portfolio`](@ref).

  - `wc`: worst-case mean-variance statistics estimator [`WCType`](@ref).
  - `set_box`:

      + if `true`: compute and set the box uncertainty sets, `port.cov_l`, `port.cov_u`, `port.d_mu`.
  - `set_ellipse`:

      + if `true`: compute and set the elliptical uncertainty sets and parameters, `port.cov_mu`, `port.cov_sigma`, `port.k_mu`, `port.k_sigma`.
"""
function wc_statistics!(port::Portfolio; wc_type::WCType = WCType(), set_box::Bool = true,
                        set_ellipse::Bool = true)
    returns = port.returns
    cov_type = wc_type.cov_type
    mu_type = wc_type.mu_type
    posdef = wc_type.posdef

    sigma, mu = sigma_mu(returns, cov_type, mu_type)

    covs = nothing
    cov_mu = nothing
    if set_box
        cov_l, cov_u, d_mu, covs, cov_mu = calc_sets(Box(), wc_type.box, wc_type.cov_type,
                                                     wc_type.mu_type, returns, sigma, mu)
        posdef_fix!(posdef, cov_l)
        posdef_fix!(posdef, cov_u)

        port.cov_l = cov_l
        port.cov_u = cov_u
        port.d_mu = d_mu
    end

    if set_ellipse
        cov_sigma, cov_mu, A_sigma, A_mu = calc_sets(Ellipse(), wc_type.ellipse,
                                                     wc_type.cov_type, wc_type.mu_type,
                                                     returns, sigma, mu, covs, cov_mu)
        posdef_fix!(posdef, cov_sigma)
        posdef_fix!(posdef, cov_mu)

        if wc_type.diagonal
            cov_mu .= Diagonal(cov_mu)
            cov_sigma .= Diagonal(cov_sigma)
        end

        k_sigma = calc_k_wc(wc_type.k_sigma, wc_type.ellipse.q, A_sigma, cov_sigma)
        k_mu = calc_k_wc(wc_type.k_mu, wc_type.ellipse.q, A_mu, cov_mu)

        port.cov_mu = cov_mu
        port.cov_sigma = cov_sigma
        port.k_mu = k_mu
        port.k_sigma = k_sigma
    end

    return nothing
end

"""
```
factor_statistics!(port::AbstractPortfolio; factor_type::FactorType = FactorType(),
                   cov_type::PortfolioOptimiserCovCor = PortCovCor(),
                   mu_type::MeanEstimator = MuSimple())
```

Compute the factor statistics. See the argument types' docs for details.

# Inputs

  - `port`: portfolio [`AbstractPortfolio`](@ref).
  - `factor_type`: factor statistics estimator [`FactorType`](@ref).
  - `cov_type`: covariance estimator [`PortfolioOptimiserCovCor`](@ref).
  - `mu_type`: expected returns estimator [`MeanEstimator`](@ref).
"""
function factor_statistics!(port::AbstractPortfolio; factor_type::FactorType = FactorType(),
                            cov_type::PortfolioOptimiserCovCor = PortCovCor(),
                            mu_type::MeanEstimator = MuSimple())
    port.f_cov, port.f_mu, port.fm_mu, port.fm_cov, port.fm_returns, port.loadings = factor_statistics(port.assets,
                                                                                                       port.returns,
                                                                                                       port.f_assets,
                                                                                                       port.f_returns;
                                                                                                       factor_type = factor_type,
                                                                                                       cov_type = cov_type,
                                                                                                       mu_type = mu_type)
    port.regression_type = factor_type.type

    return nothing
end

"""
```
black_litterman_statistics!(port::AbstractPortfolio; P::AbstractMatrix, Q::AbstractVector,
                            w::AbstractVector = port.bl_bench_weights,
                            cov_type::PortfolioOptimiserCovCor = PortCovCor(),
                            mu_type::MeanEstimator = MuSimple(),
                            bl_type::BLType = BLType())
```

Compute the factor statistics. `N` is the number of assets, `Nv` is the number of asset views. See the argument types' docs for details.

# Inputs

  - `port`: portfolio [`AbstractPortfolio`](@ref).
  - `P`: `Nv×N` matrix of asset views.
  - `Q`: `Nv×1` vector of asset views.
  - `w`: `N×1` vector of benchmark weights for the Black-Litterman model.
  - `cov_type`: covariance estimator [`PortfolioOptimiserCovCor`](@ref).
  - `mu_type`: expected returns estimator [`MeanEstimator`](@ref).
  - `bl_type`: Black Litterman model estimator [`BLType`](@ref).
"""
function black_litterman_statistics!(port::AbstractPortfolio; P::AbstractMatrix,
                                     Q::AbstractVector,
                                     w::AbstractVector = port.bl_bench_weights,
                                     cov_type::PortfolioOptimiserCovCor = PortCovCor(),
                                     mu_type::MeanEstimator = MuSimple(),
                                     bl_type::BLType = BLType())
    port.bl_bench_weights, port.bl_mu, port.bl_cov = black_litterman_statistics(port.returns,
                                                                                port.mu,
                                                                                port.cov;
                                                                                P = P,
                                                                                Q = Q,
                                                                                w = w,
                                                                                cov_type = cov_type,
                                                                                mu_type = mu_type,
                                                                                bl_type = bl_type)[1:3]

    return nothing
end

"""
```
black_litterman_factor_statistics!(port::AbstractPortfolio;
                                   w::AbstractVector = port.bl_bench_weights,
                                   B::Union{DataFrame, Nothing} = port.loadings,
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
```

Compute the Black Litterman factor model statistics. `Na` is the number of assets, `Nva` is the number of asset views, `Nf` is the number of factors, `Nvf` is the number of factors views. See the argument types' docs for details.

# Inputs

  - `port`: portfolio [`AbstractPortfolio`](@ref).

  - `w`: `N×1` vector of benchmark weights for the Black-Litterman model.
  - `B`: loadings matrix.

      + if `isempty(B)`: computes the loadings matrix using `factor_type`.
  - `P`: `Nva×Na` matrix of asset views.
  - `P_f`: `Nvf×Nf` matrix of factor views.
  - `Q`: `Nva×1` vector of asset views.
  - `Q_f`: `Nvf×1` vector of factor views.
  - `factor_type`: factor statistics estimator [`FactorType`](@ref).
  - `cov_type`: asset covariance estimator [`PortfolioOptimiserCovCor`](@ref).
  - `mu_type`: asset expected returns estimator [`MeanEstimator`](@ref).
  - `f_cov_type`: factor covariance estimator [`PortfolioOptimiserCovCor`](@ref).
  - `f_mu_type`: factor expected returns estimator [`MeanEstimator`](@ref).
  - `bl_type`: Black Litterman factor model estimator [`BlackLittermanFactor`](@ref).
"""
function black_litterman_factor_statistics!(port::AbstractPortfolio;
                                            w::AbstractVector = port.bl_bench_weights,
                                            B::Union{DataFrame, Nothing} = port.loadings,
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
    port.bl_bench_weights, port.loadings, port.blfm_mu, port.blfm_cov = black_litterman_factor_statistics(port.assets,
                                                                                                          port.returns,
                                                                                                          port.mu,
                                                                                                          port.cov,
                                                                                                          port.f_assets,
                                                                                                          port.f_returns;
                                                                                                          w = w,
                                                                                                          B = B,
                                                                                                          P = P,
                                                                                                          P_f = P_f,
                                                                                                          Q = Q,
                                                                                                          Q_f = Q_f,
                                                                                                          factor_type = factor_type,
                                                                                                          cov_type = cov_type,
                                                                                                          mu_type = mu_type,
                                                                                                          f_cov_type = f_cov_type,
                                                                                                          f_mu_type = f_mu_type,
                                                                                                          bl_type = bl_type)[1:4]
    return nothing
end

export asset_statistics!, wc_statistics!, factor_statistics!, black_litterman_statistics!,
       black_litterman_factor_statistics!
