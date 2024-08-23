function set_rm_properties(rm, solvers, sigma)
    if hasproperty(rm, :solvers) && (isnothing(rm.solvers) || isempty(rm.solvers))
        rm.solvers = solvers
    end
    if hasproperty(rm, :sigma) && (isnothing(rm.sigma) || isempty(rm.sigma))
        rm.sigma = sigma
    end
    return nothing
end

function calc_risk(port::AbstractPortfolio2; X::AbstractMatrix = port.returns,
                   type::Symbol = isa(port, Portfolio2) ? :Trad2 : :HRP,
                   rm::RiskMeasure = SD2())
    set_rm_properties(rm, port.solvers, port.cov)
    return calc_risk(rm, port.optimal[type].weights; X = X, V = port.V, SV = port.SV)
end
function risk_contribution(port::AbstractPortfolio2; X::AbstractMatrix = port.returns,
                           type::Symbol = isa(port, Portfolio2) ? :Trad2 : :HRP,
                           rm::RiskMeasure = SD2(), delta::Real = 1e-6,
                           marginal::Bool = false)
    set_rm_properties(rm, port.solvers, port.cov)
    return risk_contribution(rm, port.optimal[type].weights; X = X, V = port.V,
                             SV = port.SV, delta = delta, marginal = marginal)
end
function factor_risk_contribution(port::AbstractPortfolio2;
                                  type::Symbol = isa(port, Portfolio2) ? :Trad2 : :HRP,
                                  rm::RiskMeasure = SD2(), delta::Real = 1e-6)
    set_rm_properties(rm, port.solvers, port.cov)
    return factor_risk_contribution(rm, port.optimal[type].weights; X = port.returns,
                                    assets = port.assets, F = port.f_returns,
                                    f_assets = port.f_assets, B = port.loadings,
                                    loadings_opt = port.loadings_opt, V = port.V,
                                    SV = port.SV, delta = delta)
end
function sharpe_ratio(port::AbstractPortfolio2; X::AbstractMatrix = port.returns,
                      mu::AbstractVector = port.mu,
                      type::Symbol = isa(port, Portfolio2) ? :Trad2 : :HRP,
                      rm::RiskMeasure = SD2(), delta::Real = 1e-6, rf::Real = 0.0,
                      kelly::Bool = false)
    set_rm_properties(rm, port.solvers, port.cov)
    return sharpe_ratio(rm, port.optimal[type].weights; mu = mu, X = X, V = port.V,
                        SV = port.SV, delta = delta, rf = rf, kelly = kelly)
end

export set_rm_properties
