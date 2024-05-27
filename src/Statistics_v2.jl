
function asset_statistics2!(portfolio::AbstractPortfolio;
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

    if isa(portfolio, HCPortfolio)
        if set_cor || set_dist
            rho = cor(cor_type, returns)
            if set_cor
                portfolio.cor = rho
            end
        end

        if set_dist
            if isa(dist_type, DistanceDefault)
                dist_type = if isa(cor_type.ce, CorMutualInfo)
                    DistanceVarInfo(; bins = cor_type.ce.bins,
                                    normalise = cor_type.ce.normalise)
                elseif isa(cor_type.ce, CorLTD)
                    DistanceLog()
                else
                    DistanceMLP()
                end
            end

            if hasproperty(cor_type.ce, :absolute) && hasproperty(dist_type, :absolute)
                dist_type.absolute = cor_type.ce.absolute
            end
            portfolio.dist = dist(dist_type, rho, returns)
        end
    end

    return nothing
end

function wc_statistics2!(portfolio::Portfolio, wc::WCType = WCType(); set_box::Bool = true,
                         set_ellipse::Bool = true)
    returns = portfolio.returns
    cov_type = wc.cov_type
    mu_type = wc.mu_type
    posdef = wc.posdef

    sigma, mu = _sigma_mu(returns, cov_type, mu_type)

    covs = nothing
    cov_mu = nothing
    if set_box
        cov_l, cov_u, d_mu, covs, cov_mu = calc_sets(WorstCaseBox(), wc.box, wc.cov_type,
                                                     wc.mu_type, returns, sigma, mu)
        posdef_fix!(posdef, cov_l)
        posdef_fix!(posdef, cov_u)

        portfolio.cov_l = cov_l
        portfolio.cov_u = cov_u
        portfolio.d_mu = d_mu
    end

    if set_ellipse
        cov_sigma, cov_mu, A_sigma, A_mu = calc_sets(WorstCaseEllipse(), wc.ellipse,
                                                     wc.cov_type, wc.mu_type, returns,
                                                     sigma, mu, covs, cov_mu)
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

function factor_statistics2!(portfolio::Portfolio; factor_type::FactorType = FactorType(),
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

export CovFull, CovSemi, CorSpearman, CorKendall, CorMutualInfo, CorDistance, CorLTD,
       CorGerber0, CorGerber1, CorGerber2, CorSB0, CorSB1, CorGerberSB0, CorGerberSB1,
       DistanceMLP, dist, PortCovCor, DistanceVarInfo, BinKnuth, BinFreedman, BinScott,
       BinHGR, DistanceLog, DistanceMLP2, MeanEstimator, MeanTarget, TargetGM, TargetVW,
       TargetSE, MeanSimple, MeanJS, MeanBS, MeanBOP, SimpleVariance, asset_statistics2!,
       JLoGo, SkewFull, SkewSemi, KurtFull, KurtSemi, DenoiseFixed, DenoiseSpectral,
       DenoiseShrink, NoPosdef, NoJLoGo, DBHTExp, DBHT, wc_statistics2!, WCType,
       WorstCaseArch, WorstCaseNormal, WorstCaseDelta, WorstCaseKNormal, WorstCaseKGeneral,
       StationaryBootstrap, CircularBootstrap, MovingBootstrap, loadings_matrix2, AIC, AICC,
       BIC, R2, AdjR2, ForwardReg, BackwardReg, DimensionReductionReg, PCATarget, PVal,
       FactorType, risk_factors2, factor_statistics2!
