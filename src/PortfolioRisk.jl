function calc_risk(portfolio::AbstractPortfolio2;
                   type::Symbol = isa(portfolio, Portfolio) ? :Trad : :HRP,
                   rm::RiskMeasure = SD2())
    if hasproperty(rm, :solvers) && isempty(rm.solvers)
        rm.solvers = portfolio.solvers
    end
    return calc_risk(rm, portfolio.optimal[type].weights; X = portfolio.returns,
                     sigma = portfolio.cov, V = portfolio.V, SV = portfolio.SV,
                     solvers = portfolio.solvers)
end