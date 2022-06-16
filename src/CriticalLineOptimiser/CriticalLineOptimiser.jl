# Implements critical line optimiser by Marcos Lopez de Prado and David Bailey
using LinearAlgebra

include("CriticalLineType.jl")
include("CriticalLineFunc.jl")
include("CriticalLineUtil.jl")

export CriticalLine
export max_sharpe!, min_risk!, efficient_frontier!
export portfolio_performance