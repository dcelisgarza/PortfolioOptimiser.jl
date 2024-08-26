
function asset_statistics2!(portfolio::AbstractPortfolio2;
                            cov_type::PortfolioOptimiserCovCor = PortCovCor(),
                            set_cov::Bool = true, mu_type::MeanEstimator = MeanSimple(),
                            set_mu::Bool = true, kurt_type::KurtFull = KurtFull(),
                            set_kurt::Bool = true, skurt_type::KurtSemi = KurtSemi(),
                            set_skurt::Bool = true, skew_type::SkewFull = SkewFull(),
                            set_skew::Bool = true, sskew_type::SkewSemi = SkewSemi(),
                            set_sskew::Bool = true,
                            cor_type::PortfolioOptimiserCovCor = PortCovCor(),
                            set_cor::Bool = true,
                            dist_type::DistanceMethod = DistanceDefault(),
                            set_dist::Bool = true)
    returns = portfolio.returns

    if set_cov || set_mu && hasproperty(mu_type, :sigma)
        sigma = cov(cov_type, returns)
        if set_cov
            portfolio.cov = sigma
        end
    end
    if set_mu || set_kurt || set_skurt || set_skew || set_sskew
        if hasproperty(mu_type, :sigma)
            mu_type.sigma = sigma
        end
        mu = mean(mu_type, returns)
        if set_mu
            portfolio.mu = mu
        end
    end
    if set_kurt || set_skurt
        if set_kurt
            portfolio.kurt = cokurt(kurt_type, returns, mu)
        end
        if set_skurt
            portfolio.skurt = cokurt(skurt_type, returns, mu)
        end
        portfolio.L_2, portfolio.S_2 = dup_elim_sum_matrices(size(returns, 2))[2:3]
    end
    if set_skew
        portfolio.skew, portfolio.V = coskew(skew_type, returns, mu)
    end
    if set_sskew
        portfolio.sskew, portfolio.SV = coskew(sskew_type, returns, mu)
    end

    if isa(portfolio, HCPortfolio2)
        if set_cor || set_dist
            rho = cor(cor_type, returns)
            if set_cor
                portfolio.cor = rho
                portfolio.cor_type = cor_type
            end
        end

        if set_dist
            dist_type = _get_default_dist(dist_type, cor_type)
            portfolio.dist = dist(dist_type, rho, returns)
        end
    end

    return nothing
end

function wc_statistics2!(portfolio::Portfolio2, wc::WCType = WCType(); set_box::Bool = true,
                         set_ellipse::Bool = true)
    returns = portfolio.returns
    cov_type = wc.cov_type
    mu_type = wc.mu_type
    posdef = wc.posdef

    sigma, mu = _sigma_mu(returns, cov_type, mu_type)

    covs = nothing
    cov_mu = nothing
    if set_box
        cov_l, cov_u, d_mu, covs, cov_mu = calc_sets(WCBox(), wc.box, wc.cov_type,
                                                     wc.mu_type, returns, sigma, mu)
        posdef_fix!(posdef, cov_l)
        posdef_fix!(posdef, cov_u)

        portfolio.cov_l = cov_l
        portfolio.cov_u = cov_u
        portfolio.d_mu = d_mu
    end

    if set_ellipse
        cov_sigma, cov_mu, A_sigma, A_mu = calc_sets(WCEllipse(), wc.ellipse, wc.cov_type,
                                                     wc.mu_type, returns, sigma, mu, covs,
                                                     cov_mu)
        posdef_fix!(posdef, cov_sigma)
        posdef_fix!(posdef, cov_mu)

        if wc.diagonal
            cov_mu .= Diagonal(cov_mu)
            cov_sigma .= Diagonal(cov_sigma)
        end

        k_sigma = calc_k(wc.k_sigma, wc.ellipse.q, A_sigma, cov_sigma)
        k_mu = calc_k(wc.k_mu, wc.ellipse.q, A_mu, cov_mu)

        portfolio.cov_mu = cov_mu
        portfolio.cov_sigma = cov_sigma
        portfolio.k_mu = k_mu
        portfolio.k_sigma = k_sigma
    end

    return nothing
end

function factor_statistics2!(portfolio::Portfolio2; factor_type::FactorType = FactorType(),
                             cov_type::PortfolioOptimiserCovCor = PortCovCor(),
                             mu_type::MeanEstimator = MeanSimple())
    returns = portfolio.returns
    f_returns = portfolio.f_returns

    portfolio.f_cov, portfolio.f_mu = _sigma_mu(f_returns, cov_type, mu_type)

    portfolio.fm_mu, portfolio.fm_cov, portfolio.fm_returns, portfolio.loadings = risk_factors2(DataFrame(f_returns,
                                                                                                          portfolio.f_assets),
                                                                                                DataFrame(returns,
                                                                                                          portfolio.assets);
                                                                                                factor_type = factor_type,
                                                                                                cov_type = cov_type,
                                                                                                mu_type = mu_type)

    portfolio.loadings_opt = factor_type.method

    return nothing
end

function black_litterman_statistics2!(portfolio::Portfolio2; P::AbstractMatrix,
                                      Q::AbstractVector,
                                      w::AbstractVector = portfolio.bl_bench_weights,
                                      cov_type::PortfolioOptimiserCovCor = PortCovCor(),
                                      mu_type::MeanEstimator = MeanSimple(),
                                      bl_type::BLType = BLType())
    if isempty(w)
        w = fill(1 / size(portfolio.returns, 2), size(portfolio.returns, 2))
    end
    portfolio.bl_bench_weights = w

    if isnothing(bl_type.delta)
        bl_type.delta = (dot(portfolio.mu, w) - bl_type.rf) / dot(w, portfolio.cov, w)
    end

    portfolio.bl_mu, portfolio.bl_cov, missing = black_litterman(bl_type, portfolio.returns,
                                                                 P, Q, w;
                                                                 cov_type = cov_type,
                                                                 mu_type = mu_type)

    return nothing
end

function black_litterman_factor_statistics2!(portfolio::Portfolio2;
                                             w::AbstractVector = portfolio.bl_bench_weights,
                                             B::Union{DataFrame, Nothing} = portfolio.loadings,
                                             P::Union{AbstractMatrix, Nothing} = nothing,
                                             P_f::Union{AbstractMatrix, Nothing} = nothing,
                                             Q::Union{AbstractVector, Nothing} = nothing,
                                             Q_f::Union{AbstractVector, Nothing} = nothing,
                                             factor_type::FactorType = FactorType(),
                                             cov_type::PortfolioOptimiserCovCor = PortCovCor(),
                                             mu_type::MeanEstimator = MeanSimple(),
                                             f_cov_type::PortfolioOptimiserCovCor = PortCovCor(),
                                             f_mu_type::MeanEstimator = MeanSimple(),
                                             bl_type::BlackLittermanFactor = BBLType())
    if isempty(w)
        w = fill(1 / size(portfolio.returns, 2), size(portfolio.returns, 2))
    end
    portfolio.bl_bench_weights = w

    if isnothing(bl_type.delta)
        bl_type.delta = (dot(portfolio.mu, w) - bl_type.rf) / dot(w, portfolio.cov, w)
    end

    if isnothing(B) || isempty(B)
        if isempty(portfolio.loadings)
            portfolio.loadings = regression(factor_type.method,
                                            DataFrame(portfolio.f_returns,
                                                      portfolio.f_assets),
                                            DataFrame(portfolio.returns, portfolio.assets))
            portfolio.loadings_opt = factor_type.method
        end
        B = portfolio.loadings
    else
        portfolio.loadings = B
    end

    namesB = names(B)
    bl_type.constant = "const" ∈ namesB
    B = Matrix(B[!, setdiff(namesB, ("tickers",))])

    portfolio.blfm_mu, portfolio.blfm_cov, missing = black_litterman(bl_type,
                                                                     portfolio.returns;
                                                                     w = w,
                                                                     F = portfolio.f_returns,
                                                                     B = B, P = P,
                                                                     P_f = P_f, Q = Q,
                                                                     Q_f = Q_f,
                                                                     cov_type = cov_type,
                                                                     mu_type = mu_type,
                                                                     f_cov_type = f_cov_type,
                                                                     f_mu_type = f_mu_type)
    return nothing
end

export asset_statistics2!, wc_statistics2!, factor_statistics2!,
       black_litterman_statistics2!, black_litterman_factor_statistics2!
