function set_rm_properties(rm, solvers, sigma)
    solver_flag = false
    sigma_flag = false
    if hasproperty(rm, :solvers) && (isnothing(rm.solvers) || isempty(rm.solvers))
        rm.solvers = solvers
        solver_flag = true
    end
    if hasproperty(rm, :sigma) && (isnothing(rm.sigma) || isempty(rm.sigma))
        rm.sigma = sigma
        sigma_flag = true
    end
    return solver_flag, sigma_flag
end
function unset_set_rm_properties(rm, solver_flag, sigma_flag)
    if solver_flag
        rm.solvers = nothing
    end
    if sigma_flag
        rm.sigma = nothing
    end
    return nothing
end

function calc_risk(port::AbstractPortfolio2; X::AbstractMatrix = port.returns,
                   type::Symbol = isa(port, Portfolio2) ? :Trad2 : :HRP,
                   rm::RiskMeasure = SD2())
    solver_flag, sigma_flag = set_rm_properties(rm, port.solvers, port.cov)
    risk = calc_risk(rm, port.optimal[type].weights; X = X, V = port.V, SV = port.SV)
    unset_set_rm_properties(rm, solver_flag, sigma_flag)
    return risk
end
function risk_contribution(port::AbstractPortfolio2; X::AbstractMatrix = port.returns,
                           type::Symbol = isa(port, Portfolio2) ? :Trad2 : :HRP,
                           rm::RiskMeasure = SD2(), delta::Real = 1e-6,
                           marginal::Bool = false)
    solver_flag, sigma_flag = set_rm_properties(rm, port.solvers, port.cov)
    risk = risk_contribution(rm, port.optimal[type].weights; X = X, V = port.V,
                             SV = port.SV, delta = delta, marginal = marginal)
    unset_set_rm_properties(rm, solver_flag, sigma_flag)
    return risk
end
function factor_risk_contribution(port::AbstractPortfolio2;
                                  type::Symbol = isa(port, Portfolio2) ? :Trad2 : :HRP,
                                  rm::RiskMeasure = SD2(), delta::Real = 1e-6)
    solver_flag, sigma_flag = set_rm_properties(rm, port.solvers, port.cov)
    risk = factor_risk_contribution(rm, port.optimal[type].weights; X = port.returns,
                                    assets = port.assets, F = port.f_returns,
                                    f_assets = port.f_assets, B = port.loadings,
                                    loadings_opt = port.loadings_opt, V = port.V,
                                    SV = port.SV, delta = delta)
    unset_set_rm_properties(rm, solver_flag, sigma_flag)
    return risk
end
function sharpe_ratio(port::AbstractPortfolio2; X::AbstractMatrix = port.returns,
                      mu::AbstractVector = port.mu,
                      type::Symbol = isa(port, Portfolio2) ? :Trad2 : :HRP,
                      rm::RiskMeasure = SD2(), delta::Real = 1e-6, rf::Real = 0.0,
                      kelly::Bool = false)
    solver_flag, sigma_flag = set_rm_properties(rm, port.solvers, port.cov)
    risk = sharpe_ratio(rm, port.optimal[type].weights; mu = mu, X = X, V = port.V,
                        SV = port.SV, delta = delta, rf = rf, kelly = kelly)
    unset_set_rm_properties(rm, solver_flag, sigma_flag)
    return risk
end

export set_rm_properties, unset_set_rm_properties
