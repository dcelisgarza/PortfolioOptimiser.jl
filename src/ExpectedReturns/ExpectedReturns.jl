using Statistics, StatsBase

include("ExpectedReturnsType.jl")
include("ExpectedReturnsFunc.jl")

export AbstractReturnModel, MeanRet, ExpMeanRet, CAPMRet
export ret_model, returns_from_prices, prices_from_returns