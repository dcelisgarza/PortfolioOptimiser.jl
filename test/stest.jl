using Test,
    PortfolioOptimiser,
    DataFrames,
    TimeSeries,
    CSV,
    Dates,
    ECOS,
    SCS,
    Clarabel,
    COSMO,
    OrderedCollections,
    LinearAlgebra,
    StatsBase

A = TimeArray(CSV.File("./test/assets/stock_prices.csv"), timestamp = :date)
Y = percentchange(A)
RET = dropmissing!(DataFrame(Y))

PTypes = PortfolioOptimiser.PortTypes
RMs = PortfolioOptimiser.RiskMeasures
Kret = PortfolioOptimiser.KellyRet
ObjF = PortfolioOptimiser.ObjFuncs

rf = 1.0329^(1 / 252) - 1
l = 2.0
type = :hrp
