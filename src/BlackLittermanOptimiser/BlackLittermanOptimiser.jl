using Statistics, LinearAlgebra

include("BlackLittermanType.jl")
include("BlackLittermanFunc.jl")
include("BlackLittermanUtil.jl")

export BlackLitterman
export calc_weights!,
    calc_weights, market_implied_prior_returns, market_implied_risk_aversion
export portfolio_performance