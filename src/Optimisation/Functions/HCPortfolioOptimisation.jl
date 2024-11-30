include("./HCPortfolioOptimisationSetup.jl")
include("./HCPortfolioOptimisationRiskMeasure.jl")
include("./HCPortfolioOptimisationConstraints.jl")
include("./HCPortfolioOptimisationFinalisation.jl")
include("./HCPortfolioOptimisationHRP.jl")
include("./HCPortfolioOptimisationHERC.jl")
include("./HCPortfolioOptimisationNCO.jl")

"""
```
optimise!(port::HCPortfolio; rm::Union{AbstractVector, <:AbstractRiskMeasure} = SD(),
                   rm_o::Union{AbstractVector, <:AbstractRiskMeasure} = rm,
                   type::HCOptimType = HRP(), cluster::Bool = true,
                   clust_alg::ClustAlg = HAC(), clust_opt::ClustOpt = ClustOpt(),
                   max_iter::Int = 100)
```
"""
function optimise!(port::HCPortfolio;
                   rm::Union{AbstractVector, <:AbstractRiskMeasure} = SD(),
                   rm_o::Union{AbstractVector, <:AbstractRiskMeasure} = rm,
                   type::HCOptimType = HRP(), class::PortClass = Classic(),
                   max_iter::Int = 100)
    empty!(port.fail)
    lo, hi = w_limits(type, eltype(port.returns))
    w_min, w_max = set_hc_weights(port.w_min, port.w_max, size(port.returns, 2), lo, hi)
    w = _optimise!(type, port, class, rm, rm_o, w_min, w_max)
    return finalise_weights(type, port, w, w_min, w_max, HWF())
end
