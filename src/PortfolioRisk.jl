function set_rm_properties(rm, solvers, sigma)
    if hasproperty(rm, :solvers) && (isnothing(rm.solvers) || isempty(rm.solvers))
        rm.solvers = solvers
    end
    if hasproperty(rm, :sigma) && (isnothing(rm.sigma) || isempty(rm.sigma))
        rm.sigma = sigma
    end
    return nothing
end

function calc_risk(portfolio::AbstractPortfolio2; X::AbstractMatrix = portfolio.returns,
                   type::Symbol = isa(portfolio, Portfolio2) ? :Trad2 : :HRP,
                   rm::RiskMeasure = SD2())
    set_rm_properties(rm, portfolio.solvers, portfolio.cov)
    return calc_risk(rm, portfolio.optimal[type].weights; X = X, V = portfolio.V,
                     SV = portfolio.SV)
end
function risk_contribution(portfolio::AbstractPortfolio2;
                           X::AbstractMatrix = portfolio.returns,
                           type::Symbol = isa(portfolio, Portfolio2) ? :Trad2 : :HRP,
                           rm::RiskMeasure = SD2(), delta::Real = 1e-6,
                           marginal::Bool = false)
    set_rm_properties(rm, portfolio.solvers, portfolio.cov)
    return risk_contribution(rm, portfolio.optimal[type].weights; X = X, V = portfolio.V,
                             SV = portfolio.SV, delta = delta, marginal = marginal)
end
function factor_risk_contribution(portfolio::AbstractPortfolio2;
                                  type::Symbol = if isa(portfolio, Portfolio2)
                                      :Trad2
                                  else
                                      :HRP
                                  end, rm::RiskMeasure = SD2(), delta::Real = 1e-6)
    set_rm_properties(rm, portfolio.solvers, portfolio.cov)
    return factor_risk_contribution(rm, portfolio.optimal[type].weights;
                                    X = portfolio.returns, assets = portfolio.assets,
                                    F = portfolio.f_returns, f_assets = portfolio.f_assets,
                                    B = portfolio.loadings,
                                    loadings_opt = portfolio.loadings_opt, V = portfolio.V,
                                    SV = portfolio.SV, delta = delta)
end
function sharpe_ratio(portfolio::AbstractPortfolio2; X::AbstractMatrix = portfolio.returns,
                      mu::AbstractVector = portfolio.mu,
                      type::Symbol = isa(portfolio, Portfolio2) ? :Trad2 : :HRP,
                      rm::RiskMeasure = SD2(), delta::Real = 1e-6, rf::Real = 0.0,
                      kelly::Bool = false)
    set_rm_properties(rm, portfolio.solvers, portfolio.cov)
    return sharpe_ratio(rm, portfolio.optimal[type].weights; mu = mu, X = X,
                        V = portfolio.V, SV = portfolio.SV, delta = delta, rf = rf,
                        kelly = kelly)
end

export set_rm_properties
