using JuMP, LinearAlgebra, Ipopt, Statistics

# Common utility for all efficient frontier protfolios.
include("EfficientFrontierType.jl")
include("EfficientFrontierUtil.jl")
export AbstractEfficient
export refresh_model!

# Classic mean variance optimisations.
include("./MeanVar/MeanVarType.jl")
include("./MeanSemivar/MeanSemivarType.jl")
include("./MeanVar/MeanVarFunc.jl")
include("./MeanVar/MeanVarUtil.jl")
export AbstractEffMeanVar, EffMeanVar

# Mean semi variance optimisations.
# include("./MeanSemivar/MeanSemivarFunc.jl")
include("./MeanSemivar/MeanSemivarUtil.jl")
export AbstractEffMeanSemivar, EffMeanSemivar

# Critical Value at Risk (path dependent)
include("./CVaR/CVaRType.jl")
include("./CVaR/CVaRFunc.jl")
include("./CVaR/CVaRUtil.jl")
export EffCVaR

# Critical Drawdown at Risk (path dependent)
include("./CDaR/CDaRType.jl")
include("./CDaR/CDaRFunc.jl")
include("./CDaR/CDaRUtil.jl")
export AbstractEffCDaR, EffCDaR

# Common functions among all efficient frontier optimisations.
export min_risk!, max_sharpe!, max_utility!, efficient_risk!, efficient_return!
export portfolio_performance
