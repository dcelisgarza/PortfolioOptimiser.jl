using JuMP, LinearAlgebra, Ipopt, Statistics

# Common utility for all efficient frontier protfolios.
include("EfficientFrontierType.jl")
include("EfficientFrontierUtil.jl")
export AbstractEfficient

# Classic mean variance optimisations.
include("./MeanVar/MeanVarType.jl")
include("./MeanSemivar/MeanSemivarType.jl")
include("./MeanAbsDev/MeanAbsDevType.jl")
include("./Minimax/MinimaxType.jl")
include("./CVaR/CVaRType.jl")
include("./CDaR/CDaRType.jl")
include("./MaxDaR/MaxDaRType.jl")
include("./MeanDaR/MeanDaRType.jl")
include("./UlcerIndex/UlcerIndexType.jl")
export AbstractEffMeanVar, EffMeanVar
export AbstractEffMeanSemivar, EffMeanSemivar
export AbstractEffMeanAbsDev, EffMeanAbsDev
export AbstractEffMinimax, EffMinimax
export AbstractEffCVaR, EffCVaR
export AbstractEffCDaR, EffCDaR
export AbstractEffMaxDaR, EffMaxDaR
export AbstractEffMeanDaR, EffMeanDaR
export AbstractEffUlcer, EffUlcer

# Utility functions.
include("./MeanVar/MeanVarUtil.jl")
include("./MeanSemivar/MeanSemivarUtil.jl")
include("./MeanAbsDev/MeanAbsDevUtil.jl")
include("./Minimax/MinimaxUtil.jl")
include("./CVaR/CVaRUtil.jl")
include("./CDaR/CDaRUtil.jl")
include("./MaxDaR/MaxDaRUtil.jl")
include("./MeanDaR/MeanDaRUtil.jl")
include("./UlcerIndex/UlcerIndexUtil.jl")

# Optimisation functions.
include("./EfficientFrontierFunc.jl")

# Common functions among all efficient frontier optimisations.
export min_risk!, max_sharpe!, max_utility!, efficient_risk!, efficient_return!
export refresh_model!, portfolio_performance
