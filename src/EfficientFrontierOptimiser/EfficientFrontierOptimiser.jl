using JuMP, LinearAlgebra, Ipopt, Statistics

# Common utility for all efficient frontier protfolios.
include("EfficientFrontierType.jl")
include("EfficientFrontierUtil.jl")
export AbstractEfficient

# Classic mean variance optimisations.
include("./MeanVar/MeanVarType.jl")
include("./MeanSemivar/MeanSemivarType.jl")
include("./MeanAbsDev/MeanAbsDevType.jl")
include("./CVaR/CVaRType.jl")
include("./CDaR/CDaRType.jl")
export AbstractEffMeanVar, EffMeanVar
export AbstractEffMeanSemivar, EffMeanSemivar
export AbstractEffMeanAbsDev, EffMeanAbsDev
export AbstractEffCVaR, EffCVaR
export AbstractEffCDaR, EffCDaR

# Utility functions.
include("./MeanVar/MeanVarUtil.jl")
include("./MeanSemivar/MeanSemivarUtil.jl")
include("./MeanAbsDev/MeanAbsDevUtil.jl")
include("./CVaR/CVaRUtil.jl")
include("./CDaR/CDaRUtil.jl")

# Optimisation functions.
include("./EfficientFrontierFunc.jl")

# Common functions among all efficient frontier optimisations.
export min_risk!, max_sharpe!, max_utility!, efficient_risk!, efficient_return!
export refresh_model!, portfolio_performance
