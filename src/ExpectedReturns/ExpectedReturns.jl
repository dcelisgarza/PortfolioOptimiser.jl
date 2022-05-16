using Statistics, StatsBase

include("ExpectedReturnsType.jl")
include("ExpectedReturnsFunc.jl")

export AbstractReturnModel, MRet, EMRet, CAPMRet, ECAPMRet
export ret_model, returns_from_prices, prices_from_returns