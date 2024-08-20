module PortfolioOptimiserMakieExt
using PortfolioOptimiser, Makie, SmartAsserts, Statistics, MultivariateStats, Distributions,
      Clustering, Graphs, SimpleWeightedGraphs, LinearAlgebra

import PortfolioOptimiser: AbstractPortfolio2
const PO = PortfolioOptimiser

"""
```
plot_returns(timestamps, assets, returns, weights; per_asset = false, kwargs...)
```
"""
function PO.plot_returns2(timestamps, assets, returns, weights; per_asset = false)
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
function PO.plot_returns2(portfolio::AbstractPortfolio2,
                          type = isa(portfolio, HCPortfolio2) ? :HRP2 : :Trad2;
                          per_asset = false)
    return PO.plot_returns2(portfolio.timestamps, portfolio.assets, portfolio.returns,
                            portfolio.optimal[type].weights; per_asset = per_asset)
end

function PO.plot_bar2(assets, weights)
    f = Figure()
    ax = Axis(f[1, 1]; xticks = (1:length(assets), assets),
              ylabel = "Portfolio Composition, %", xticklabelrotation = pi / 2)
    barplot!(ax, weights * 100)
    return f
end
function PO.plot_bar2(portfolio::AbstractPortfolio2,
                      type = isa(portfolio, HCPortfolio) ? :HRP2 : :Trad2; kwargs...)
    return PO.plot_bar2(portfolio.assets, portfolio.optimal[type].weights, kwargs...)
end

function PO.plot_risk_contribution2(assets::AbstractVector, w::AbstractVector,
                                    X::AbstractMatrix; rm::RiskMeasure = SD2(),
                                    V::AbstractMatrix = Matrix{Float64}(undef, 0, 0),
                                    SV::AbstractMatrix = Matrix{Float64}(undef, 0, 0),
                                    percentage::Bool = false, erc_line::Bool = true,
                                    t_factor = 252, delta::Real = 1e-6,
                                    marginal::Bool = false, kwargs...)
    rc = calc_risk_contribution(rm, w; X = X, V = V, SV = SV, delta = delta,
                                marginal = marginal, kwargs...)

    DDs = (DaR2, MDD2, ADD2, CDaR2, EDaR2, RDaR2, UCI2, DaR_r2, MDD_r2, ADD_r2, CDaR_r2,
           EDaR_r2, RDaR_r2, UCI_r2)

    if !any(typeof(rm) .<: DDs)
        rc *= sqrt(t_factor)
    end
    ylabel = "Risk Contribution"
    if percentage
        rc .= 100 * rc / sum(rc)
        ylabel *= ", %"
    end

    rmstr = string(rm)
    rmstr = rmstr[1:(findfirst('{', rmstr) - 1)]
    title = "Risk Contribution - $rmstr"
    if any(typeof(rm) .<:
           (CVaR2, TG2, EVaR2, RVaR2, RCVaR2, RTG2, CDaR2, EDaR2, RDaR2, CDaR_r2, EDaR_r2,
            RDaR_r2))
        title *= " α = $(round(rm.alpha*100, digits=2))%"
    end
    if any(typeof(rm) .<: (RCVaR2, RTG2))
        title *= ", β = $(round(rm.beta*100, digits=2))%"
    end
    if any(typeof(rm) .<: (RVaR2, RDaR2, RDaR_r2))
        title *= ", κ = $(round(rm.kappa, digits=2))"
    end

    # if !haskey(kwargs_bar, :title)
    #     kwargs_bar = (kwargs_bar..., title = title)
    # end
    # if !haskey(kwargs_bar, :ylabel)
    #     kwargs_bar = (kwargs_bar..., ylabel = ylabel)
    # end
    # if !haskey(kwargs_bar, :xlabel)
    #     kwargs_bar = (kwargs_bar..., xlabel = "Assets")
    # end
    # if !haskey(kwargs_bar, :xticks)
    #     kwargs_bar = (kwargs_bar...,
    #                   xticks = (range(0.5; step = 1, length = length(assets)), assets))
    # end
    # if !haskey(kwargs_bar, :xrotation)
    #     kwargs_bar = (kwargs_bar..., xrotation = 90)
    # end
    # if !haskey(kwargs_bar, :legend)
    #     kwargs_bar = (kwargs_bar..., legend = false)
    # end

    # plt = bar(assets, rc; kwargs_bar...)

    f = Figure()
    ax = Axis(f[1, 1]; xticks = (1:length(assets), assets), title = title, ylabel = ylabel,
              xticklabelrotation = pi / 2)
    barplot!(ax, rc)

    if erc_line
        if percentage
            erc = 100 / length(rc)
        else
            erc = calc_risk(rm, w; X = X, V = V, SV = SV)

            erc /= length(rc)

            if !any(typeof(rm) .<: DDs)
                erc *= sqrt(t_factor)
            end
        end

        hlines!(ax, erc)
    end

    return f
end
function PO.plot_risk_contribution2(portfolio::AbstractPortfolio2,
                                    type = isa(portfolio, HCPortfolio) ? :HRP2 : :Trad2;
                                    rm::RiskMeasure = SD2(), percentage::Bool = false,
                                    erc_line::Bool = true, t_factor = 252,
                                    delta::Real = 1e-6, marginal::Bool = false, kwargs...)
    if hasproperty(rm, :solvers) && (isnothing(rm.solvers) || isempty(rm.solvers))
        rm.solvers = portfolio.solvers
    end
    if hasproperty(rm, :sigma) && (isnothing(rm.sigma) || isempty(rm.sigma))
        rm.sigma = portfolio.cov
    end
    return PO.plot_risk_contribution2(portfolio.assets, portfolio.optimal[type].weights,
                                      portfolio.returns; rm = rm, V = portfolio.V,
                                      SV = portfolio.SV, percentage = percentage,
                                      erc_line = erc_line, t_factor = t_factor,
                                      delta = delta, marginal = marginal, kwargs...)
end

end
