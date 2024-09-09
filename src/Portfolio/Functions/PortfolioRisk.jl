function calc_risk(port::AbstractPortfolio; X::AbstractMatrix = port.returns,
                   type::Symbol = isa(port, Portfolio) ? :Trad : :HRP,
                   rm::AbstractRiskMeasure = SD())
    solver_flag, sigma_flag = set_rm_properties(rm, port.solvers, port.cov)
    risk = calc_risk(rm, port.optimal[type].weights; X = X, V = port.V, SV = port.SV)
    unset_set_rm_properties(rm, solver_flag, sigma_flag)
    return risk
end

function risk_contribution(port::AbstractPortfolio; X::AbstractMatrix = port.returns,
                           type::Symbol = isa(port, Portfolio) ? :Trad : :HRP,
                           rm::AbstractRiskMeasure = SD(), delta::Real = 1e-6,
                           marginal::Bool = false)
    solver_flag, sigma_flag = set_rm_properties(rm, port.solvers, port.cov)
    risk = risk_contribution(rm, port.optimal[type].weights; X = X, V = port.V,
                             SV = port.SV, delta = delta, marginal = marginal)
    unset_set_rm_properties(rm, solver_flag, sigma_flag)
    return risk
end

function factor_risk_contribution(port::AbstractPortfolio; X::AbstractMatrix = port.returns,
                                  F::AbstractMatrix = port.f_returns,
                                  type::Symbol = isa(port, Portfolio) ? :Trad : :HRP,
                                  rm::AbstractRiskMeasure = SD(), delta::Real = 1e-6)
    solver_flag, sigma_flag = set_rm_properties(rm, port.solvers, port.cov)
    risk = factor_risk_contribution(rm, port.optimal[type].weights; X = X,
                                    assets = port.assets, F = F, f_assets = port.f_assets,
                                    B = port.loadings,
                                    regression_type = port.regression_type, V = port.V,
                                    SV = port.SV, delta = delta)
    unset_set_rm_properties(rm, solver_flag, sigma_flag)
    return risk
end

function sharpe_ratio(port::AbstractPortfolio; X::AbstractMatrix = port.returns,
                      mu::AbstractVector = port.mu,
                      type::Symbol = isa(port, Portfolio) ? :Trad : :HRP,
                      rm::AbstractRiskMeasure = SD(), delta::Real = 1e-6, rf::Real = 0.0,
                      kelly::Bool = false)
    solver_flag, sigma_flag = set_rm_properties(rm, port.solvers, port.cov)
    risk = sharpe_ratio(rm, port.optimal[type].weights; mu = mu, X = X, V = port.V,
                        SV = port.SV, delta = delta, rf = rf, kelly = kelly)
    unset_set_rm_properties(rm, solver_flag, sigma_flag)
    return risk
end
