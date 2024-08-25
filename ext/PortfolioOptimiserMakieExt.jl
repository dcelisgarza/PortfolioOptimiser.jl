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
function PO.plot_returns2(port::AbstractPortfolio2,
                          type = isa(port, HCPortfolio2) ? :HRP2 : :Trad2;
                          per_asset = false)
    return PO.plot_returns2(port.timestamps, port.assets, port.returns,
                            port.optimal[type].weights; per_asset = per_asset)
end

function PO.plot_bar2(assets, weights)
    f = Figure()
    ax = Axis(f[1, 1]; xticks = (1:length(assets), assets),
              ylabel = "Portfolio Composition, %", xticklabelrotation = pi / 2)
    barplot!(ax, weights * 100)
    return f
end
function PO.plot_bar2(port::AbstractPortfolio2,
                      type = isa(port, HCPortfolio) ? :HRP2 : :Trad2; kwargs...)
    return PO.plot_bar2(port.assets, port.optimal[type].weights, kwargs...)
end

function PO.plot_risk_contribution2(assets::AbstractVector, w::AbstractVector,
                                    X::AbstractMatrix; rm::RiskMeasure = SD2(),
                                    V::AbstractMatrix = Matrix{Float64}(undef, 0, 0),
                                    SV::AbstractMatrix = Matrix{Float64}(undef, 0, 0),
                                    percentage::Bool = false, erc_line::Bool = true,
                                    t_factor = 252, delta::Real = 1e-6,
                                    marginal::Bool = false, kwargs...)
    rc = risk_contribution(rm, w; X = X, V = V, SV = SV, delta = delta, marginal = marginal,
                           kwargs...)

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
function PO.plot_risk_contribution2(port::AbstractPortfolio2,
                                    type = isa(port, HCPortfolio) ? :HRP2 : :Trad2;
                                    X = port.returns, rm::RiskMeasure = SD2(),
                                    percentage::Bool = false, erc_line::Bool = true,
                                    t_factor = 252, delta::Real = 1e-6,
                                    marginal::Bool = false, kwargs...)
    solver_flag, sigma_flag = set_rm_properties(rm, port.solvers, port.cov)
    fig = PO.plot_risk_contribution2(port.assets, port.optimal[type].weights, X; rm = rm,
                                     V = port.V, SV = port.SV, percentage = percentage,
                                     erc_line = erc_line, t_factor = t_factor,
                                     delta = delta, marginal = marginal, kwargs...)
    unset_set_rm_properties(rm, solver_flag, sigma_flag)

    return fig
end

function PO.plot_frontier2(frontier; X::AbstractMatrix = Matrix{Float64}(undef, 0, 0),
                           mu::AbstractVector = Vector{Float64}(undef, 0), rf::Real = 0.0,
                           rm::RiskMeasure = SD2(), kelly::RetType = NoKelly(),
                           t_factor = 252, palette = :Spectral)
    risks = copy(frontier[:risk])
    weights = Matrix(frontier[:weights][!, 2:end])

    if isa(kelly, NoKelly)
        ylabel = "Expected Arithmetic Return"
        rets = transpose(weights) * mu
    else
        ylabel = "Expected Kelly Return"
        rets = 1 / size(X, 1) * vec(sum(log.(1 .+ X * weights); dims = 1))
    end

    rets .*= t_factor

    if !any(typeof(rm) .<: (MDD2, ADD2, CDaR2, EDaR2, RDaR2, UCI2))
        risks .*= sqrt(t_factor)
    end

    ratios = (rets .- rf) ./ risks
    N = length(ratios)

    title = "$(get_rm_string(rm))"
    if any(typeof(rm) .<: (CVaR2, TG2, EVaR2, RVaR2, RCVaR2, RTG2, CDaR2, EDaR2, RDaR2))
        title *= " α = $(round(rm.alpha*100, digits=2))%"
    end
    if any(typeof(rm) .<: (RCVaR2, RTG2))
        title *= ", β = $(round(rm.beta*100, digits=2))%"
    end
    if any(typeof(rm) .<: (RVaR2, RDaR2))
        title *= ", κ = $(round(rm.kappa, digits=2))"
    end

    f = Figure()
    ax = Axis(f[1, 1]; title = title, ylabel = ylabel, xlabel = "Expected Risk")
    Colorbar(f[1, 2]; label = "Risk Adjusted Return Ratio", limits = extrema(ratios),
             colormap = palette)

    if frontier[:sharpe]
        scatter!(ax, risks[1:(end - 1)], rets[1:(end - 1)]; color = ratios[1:(N - 1)],
                 colormap = palette, marker = :circle, markersize = 15)
        scatter!(ax, risks[end], rets[end]; color = cgrad(palette)[ratios[N]],
                 marker = :star5, markersize = 15, label = "Max Risk Adjusted Return Ratio")
    else
        scatter(ax, risks[1:end], rets[1:end]; color = ratios, colormap = palette)
    end
    axislegend(ax; position = :rb)

    return f
end
function PO.plot_frontier2(port::AbstractPortfolio2, key = nothing;
                           X::AbstractMatrix = port.returns, mu::AbstractVector = port.mu,
                           rm::RiskMeasure = SD2(), rf::Real = 0.0,
                           kelly::RetType = NoKelly(), t_factor = 252, palette = :Spectral)
    if isnothing(key)
        key = get_rm_string(rm)
    end
    return PO.plot_frontier2(port.frontier[key]; X = X, mu = mu, rf = rf, rm = rm,
                             kelly = kelly, t_factor = t_factor, palette = palette)
end

function PO.plot_frontier_area2(frontier; rm::RiskMeasure = SD2(), t_factor = 252,
                                palette = :Spectral)
    risks = copy(frontier[:risk])
    assets = reshape(frontier[:weights][!, "tickers"], 1, :)
    weights = Matrix(frontier[:weights][!, 2:end])

    if !any(typeof(rm) .<: (MDD2, ADD2, CDaR2, EDaR2, RDaR2, UCI2))
        risks .*= sqrt(t_factor)
    end

    # sharpe = nothing
    # if frontier[:sharpe]
    #     sharpe = risks[end]
    #     risks = risks[1:(end - 1)]
    #     weights = weights[:, 1:(end - 1)]
    # end

    idx = sortperm(risks)
    risks = risks[idx]
    weights = weights[:, idx]

    title = "$(get_rm_string(rm))"
    if any(typeof(rm) .<: (CVaR2, TG2, EVaR2, RVaR2, RCVaR2, RTG2, CDaR2, EDaR2, RDaR2))
        title *= " α = $(round(rm.alpha*100, digits=2))%"
    end
    if any(typeof(rm) .<: (RCVaR2, RTG2))
        title *= ", β = $(round(rm.beta*100, digits=2))%"
    end
    if any(typeof(rm) .<: (RVaR2, RDaR2))
        title *= ", κ = $(round(rm.kappa, digits=2))"
    end

    f = Figure()
    ax = Axis(f[1, 1]; title = title, ylabel = "Composition", xlabel = "Expected Risk",
              limits = (minimum(risks), maximum(risks), 0, 1))

    N = length(risks)
    weights = cumsum(weights; dims = 1)
    # colours = cgrad(palette)
    for i ∈ axes(weights, 1)
        if i == 1
            band!(ax, risks, range(0, 0, N), weights[i, :]; label = assets[i])
        else
            band!(ax, risks, weights[i - 1, :], weights[i, :]; label = assets[i])
        end
    end
    axislegend(ax; position = :rc)

    # f = Figure()
    # Axis(f[1, 1])

    # xs = range(risks)
    # ys_low = -0.2 .* sin.(xs) .- 0.25
    # ys_high = 0.2 .* sin.(xs) .+ 0.25

    # band!(xs, ys_low, ys_high)
    # band!(xs, ys_low .- 1, ys_high .- 1; color = :red)
    # Colorbar(f[1, 2]; label = "Risk Adjusted Return Ratio", limits = extrema(ratios),
    #          colormap = palette)

    # plt = areaplot(risks, weights; kwargs_a...)

    # if !isnothing(sharpe) && show_sharpe
    #     if !haskey(kwargs_l, :color)
    #         kwargs_l = (kwargs_l..., color = :black)
    #     end
    #     if !haskey(kwargs_l, :linewidth)
    #         kwargs_l = (kwargs_l..., linewidth = 3)
    #     end
    #     plot!(plt, [sharpe, sharpe], [0, 1]; label = nothing, kwargs_l...)
    #     annotate!([sharpe * 1.1], [0.5], text("Max Risk\nAdjusted\nReturn Ratio", :left, 8))
    # end

    return f
end

end
