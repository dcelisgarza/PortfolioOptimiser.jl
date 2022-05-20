using JuMP, LinearAlgebra, Ipopt, Statistics

# Common utility for all efficient frontier protfolios.
include("EfficientFrontierType.jl")
include("EfficientFrontierUtil.jl")
export AbstractEfficient
export refresh_model!

# Classic mean variance optimisations.
include("./MeanVar/MeanVarType.jl")
include("./MeanVar/MeanVarFunc.jl")
include("./MeanVar/MeanVarUtil.jl")
export EfficientMeanVar
export min_volatility!, max_return, max_sharpe!

# Mean semi variance optimisations.
include("./MeanSemiVar/MeanSemiVarType.jl")
include("./MeanSemiVar/MeanSemiVarFunc.jl")
include("./MeanSemiVar/MeanSemiVarUtil.jl")
export EfficientSemiVar
export min_semivar!, max_sortino!

# Critical Drawdown at Risk (path dependent)
include("./CDaR/CDaRType.jl")
include("./CDaR/CDaRFunc.jl")
include("./CDaR/CDaRUtil.jl")
export EfficientCDaR
export min_cdar!

# Critical Value at Risk (path dependent)
include("./CVaR/CVaRType.jl")
include("./CVaR/CVaRFunc.jl")
include("./CVaR/CVaRUtil.jl")
export EfficientCVaR
export min_cvar!

# Common functions among all efficient frontier optimisations.
export max_quadratic_utility!, efficient_risk!, efficient_return!
export portfolio_performance

# end