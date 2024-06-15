function calc_risk(portfolio::AbstractPortfolio2;
                   type::Symbol = isa(portfolio, Portfolio) ? :Trad : :HRP,
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