module PortfolioOptimiserMakieExt
using PortfolioOptimiser, Makie, SmartAsserts, Statistics, MultivariateStats, Distributions,
      Clustering, Graphs, SimpleWeightedGraphs, LinearAlgebra

const PO = PortfolioOptimiser

"""
```
plot_returns(timestamps, assets, returns, weights; per_asset = false, kwargs...)
```
"""
function PO.plot_returns2(timestamps, assets, returns, weights; per_asset = false,
                          kwargs...)
    f = Figure()
    if per_asset
        ax = Axis(f[1, 1]; xlabel = "Date", ylabel = "Asset Cummulative Returns")
        ret = returns .* transpose(weights)
        ret = vcat(zeros(1, length(weights)), ret)
        ret .+= 1
        ret = cumprod(ret; dims = 1)
        ret = ret[2:end, :]
        for (i, asset) ∈ enumerate(assets)
            lines!(ax, timestamps, view(ret, :, i); label = asset)
        end
    else
        ax = Axis(f[1, 1]; xlabel = "Date", ylabel = "Portfolio Cummulative Returns")
        ret = returns * weights
        pushfirst!(ret, 0)
        ret .+= 1
        ret = cumprod(ret)
        popfirst!(ret)
        lines!(ax, timestamps, ret; label = "Portfolio")
    end

    axislegend(; position = :lt, merge = true)

    return f
end
function PO.plot_returns2(portfolio, type = isa(portfolio, HCPortfolio2) ? :HRP2 : :Trad2;
                          per_asset = false, kwargs...)
    return PO.plot_returns2(portfolio.timestamps, portfolio.assets, portfolio.returns,
                            portfolio.optimal[type].weights; per_asset = per_asset,
                            kwargs...)
end

end