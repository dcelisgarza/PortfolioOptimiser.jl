# Implements the Hierarchical Risk Parity portfolio optimisation of Marcos Lopez de Prado (2016).

using Clustering, LinearAlgebra, Statistics

include("HRPOptType.jl")
include("HRPOptFunc.jl")
include("HRPOptUtil.jl")

export HRPOpt,
    optimise!,
    min_volatility,
    max_quadratic_utility,
    max_return,
    max_sharpe,
    portfolio_performance