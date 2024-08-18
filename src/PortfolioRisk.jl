function calc_risk(portfolio::AbstractPortfolio2;
                   type::Symbol = isa(portfolio, Portfolio2) ? :Trad2 : :HRP,
                   rm::RiskMeasure = SD2())
    if hasproperty(rm, :solvers) && (isnothing(rm.solvers) || isempty(rm.solvers))
        rm.solvers = portfolio.solvers
    end
    if hasproperty(rm, :sigma) && (isnothing(rm.sigma) || isempty(rm.sigma))
        rm.sigma = portfolio.cov
    end
    return calc_risk(rm, portfolio.optimal[type].weights; X = portfolio.returns,
                     V = portfolio.V, SV = portfolio.SV)
end
function calc_risk_contribution(portfolio::AbstractPortfolio2;
                                type::Symbol = isa(portfolio, Portfolio2) ? :Trad2 : :HRP,
                                rm::RiskMeasure = SD2(), delta::Real = 1e-6,
                                marginal::Bool = false)
    if hasproperty(rm, :solvers) && (isnothing(rm.solvers) || isempty(rm.solvers))
        rm.solvers = portfolio.solvers
    end
    if hasproperty(rm, :sigma) && (isnothing(rm.sigma) || isempty(rm.sigma))
        rm.sigma = portfolio.cov
    end
    return calc_risk_contribution(rm, portfolio.optimal[type].weights;
                                  X = portfolio.returns, V = portfolio.V, SV = portfolio.SV,
                                  delta = delta, marginal = marginal)
end
