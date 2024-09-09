module PortfolioOptimiserPlotsExt

using PortfolioOptimiser, GraphRecipes, StatsPlots, SmartAsserts, Statistics,
      MultivariateStats, Distributions, Clustering, Graphs, SimpleWeightedGraphs,
      LinearAlgebra

"""
```
plot_returns(timestamps, assets, returns, weights; per_asset = false, kwargs...)
```
"""
function PortfolioOptimiser.plot_returns(timestamps, assets, returns, weights;
                                         per_asset = false, kwargs...)
    if per_asset
        ret = returns .* transpose(weights)
        ret = vcat(zeros(1, length(weights)), ret)
        ret .+= 1
        ret = cumprod(ret; dims = 1)
        ret = ret[2:end, :]
        if !haskey(kwargs, :label)
            kwargs = (kwargs..., label = reshape(assets, 1, :))
        end
    else
        ret = returns * weights
        pushfirst!(ret, 0)
        ret .+= 1
        ret = cumprod(ret)
        popfirst!(ret)
        if !haskey(kwargs, :legend)
            kwargs = (kwargs..., legend = false)
        end
    end
    if !haskey(kwargs, :ylabel)
        kwargs = (kwargs..., ylabel = "Cummulative Return")
    end
    if !haskey(kwargs, :xlabel)
        kwargs = (kwargs..., xlabel = "Date")
    end

    return plot(timestamps, ret; kwargs...)
end
function PortfolioOptimiser.plot_returns(port, type = isa(port, Portfolio) ? :Trad : :HRP;
                                         allocated::Bool = false, per_asset::Bool = false,
                                         kwargs...)
    return PortfolioOptimiser.plot_returns(port.timestamps, port.assets, port.returns,
                                           if !allocated
                                               port.optimal[type].weights
                                           else
                                               port.alloc_optimal[type].weights
                                           end; per_asset = per_asset, kwargs...)
end

function PortfolioOptimiser.plot_bar(assets, data; kwargs...)
    if !haskey(kwargs, :ylabel)
        kwargs = (kwargs..., ylabel = "Percentage Composition")
    end
    if !haskey(kwargs, :xlabel)
        kwargs = (kwargs..., xlabel = "Assets")
    end
    if !haskey(kwargs, :xticks)
        kwargs = (kwargs...,
                  xticks = (range(0.5; step = 1, length = length(assets)), assets))
    end
    if !haskey(kwargs, :xrotation)
        kwargs = (kwargs..., xrotation = 90)
    end
    if !haskey(kwargs, :legend)
        kwargs = (kwargs..., legend = false)
    end

    return bar(assets, data * 100; kwargs...)
end
function PortfolioOptimiser.plot_bar(port::PortfolioOptimiser.AbstractPortfolio,
                                     type = isa(port, Portfolio) ? :Trad : :HRP;
                                     allocated::Bool = false, kwargs...)
    return PortfolioOptimiser.plot_bar(port.assets, if !allocated
                                           port.optimal[type].weights
                                       else
                                           port.alloc_optimal[type].weights
                                       end, kwargs...)
end

function PortfolioOptimiser.plot_risk_contribution(assets::AbstractVector,
                                                   w::AbstractVector, X::AbstractMatrix;
                                                   rm::PortfolioOptimiser.AbstractRiskMeasure = SD(),
                                                   V::AbstractMatrix = Matrix{Float64}(undef,
                                                                                       0,
                                                                                       0),
                                                   SV::AbstractMatrix = Matrix{Float64}(undef,
                                                                                        0,
                                                                                        0),
                                                   delta::Real = 1e-6,
                                                   percentage::Bool = false,
                                                   erc_line::Bool = true, t_factor = 252,
                                                   marginal::Bool = false, kwargs_bar = (;),
                                                   kwargs_line = (;))
    rc = risk_contribution(rm, w; X = X, V = V, SV = SV, delta = delta, marginal = marginal)

    DDs = (DaR, MDD, ADD, CDaR, EDaR, RLDaR, UCI, DaR_r, MDD_r, ADD_r, CDaR_r, EDaR_r,
           RLDaR_r, UCI_r)

    if !any(typeof(rm) .<: DDs)
        rc *= sqrt(t_factor)
    end
    ylabel = "Risk Contribution"
    if percentage
        rc /= sum(rc)
        ylabel *= " Percentage"
    end

    rmstr = string(rm)
    rmstr = rmstr[1:(findfirst('{', rmstr) - 1)]
    title = "Risk Contribution - $rmstr"
    if any(typeof(rm) .<:
           (CVaR, TG, EVaR, RLVaR, CVaRRG, TGRG, CDaR, EDaR, RLDaR, CDaR_r, EDaR_r,
            RLDaR_r))
        title *= " α = $(round(rm.alpha*100, digits=2))%"
    end
    if any(typeof(rm) .<: (CVaRRG, TGRG))
        title *= ", β = $(round(rm.beta*100, digits=2))%"
    end
    if any(typeof(rm) .<: (RLVaR, RLDaR, RLDaR_r))
        title *= ", κ = $(round(rm.kappa, digits=2))"
    end

    if !haskey(kwargs_bar, :title)
        kwargs_bar = (kwargs_bar..., title = title)
    end
    if !haskey(kwargs_bar, :ylabel)
        kwargs_bar = (kwargs_bar..., ylabel = ylabel)
    end
    if !haskey(kwargs_bar, :xlabel)
        kwargs_bar = (kwargs_bar..., xlabel = "Assets")
    end
    if !haskey(kwargs_bar, :xticks)
        kwargs_bar = (kwargs_bar...,
                      xticks = (range(0.5; step = 1, length = length(assets)), assets))
    end
    if !haskey(kwargs_bar, :xrotation)
        kwargs_bar = (kwargs_bar..., xrotation = 90)
    end
    if !haskey(kwargs_bar, :legend)
        kwargs_bar = (kwargs_bar..., legend = false)
    end

    plt = bar(assets, rc; kwargs_bar...)

    if erc_line
        if percentage
            erc = 1 / length(rc)
        else
            erc = calc_risk(rm, w; X = X, V = V, SV = SV)

            erc /= length(rc)

            if rm ∉ DDs
                erc *= sqrt(t_factor)
            end
        end

        hline!([erc]; kwargs_line...)
    end

    return plt
end
function PortfolioOptimiser.plot_risk_contribution(port::PortfolioOptimiser.AbstractPortfolio,
                                                   type = if isa(port, Portfolio)
                                                       :Trad
                                                   else
                                                       :HRP
                                                   end; X = port.returns,
                                                   rm::PortfolioOptimiser.AbstractRiskMeasure = SD(),
                                                   percentage::Bool = false,
                                                   erc_line::Bool = true, t_factor = 252,
                                                   delta::Real = 1e-6,
                                                   marginal::Bool = false,
                                                   allocated::Bool = false,
                                                   kwargs_bar = (;), kwargs_line = (;))
    solver_flag, sigma_flag = set_rm_properties(rm, port.solvers, port.cov)
    fig = PortfolioOptimiser.plot_risk_contribution(port.assets,
                                                    if !allocated
                                                        port.optimal[type].weights
                                                    else
                                                        port.alloc_optimal[type].weights
                                                    end, X; rm = rm, V = port.V,
                                                    SV = port.SV, percentage = percentage,
                                                    erc_line = erc_line,
                                                    t_factor = t_factor, delta = delta,
                                                    marginal = marginal,
                                                    kwargs_bar = kwargs_bar,
                                                    kwargs_line = kwargs_line)
    unset_set_rm_properties(rm, solver_flag, sigma_flag)
    return fig
end

function PortfolioOptimiser.plot_frontier(frontier;
                                          X::AbstractMatrix = Matrix{Float64}(undef, 0, 0),
                                          mu::AbstractVector = Vector{Float64}(undef, 0),
                                          rf::Real = 0.0,
                                          rm::PortfolioOptimiser.AbstractRiskMeasure = SD(),
                                          kelly::PortfolioOptimiser.RetType = NoKelly(),
                                          t_factor = 252, theme = :Spectral, kwargs_f = (;),
                                          kwargs_s = (;))
    risks = copy(frontier[:risks])
    weights = Matrix(frontier[:weights][!, 2:end])

    rets = if isa(kelly, NoKelly)
        transpose(weights) * mu
    else
        1 / size(X, 1) * vec(sum(log.(1 .+ X * weights); dims = 1))
    end
    rets .*= t_factor

    if !any(typeof(rm) .<: (MDD, ADD, CDaR, EDaR, RLDaR, UCI))
        risks .*= sqrt(t_factor)
    end

    ratios = (rets .- rf) ./ risks

    msg = "$(PortfolioOptimiser.get_rm_symb(rm))"
    if any(typeof(rm) .<: (CVaR, TG, EVaR, RLVaR, CVaRRG, TGRG, CDaR, EDaR, RLDaR))
        msg *= " α = $(round(rm.alpha*100, digits=2))%"
    end
    if any(typeof(rm) .<: (CVaRRG, TGRG))
        msg *= ", β = $(round(rm.beta*100, digits=2))%"
    end
    if any(typeof(rm) .<: (RLVaR, RLDaR))
        msg *= ", κ = $(round(rm.kappa, digits=2))"
    end

    if !haskey(kwargs_f, :ylabel)
        kwargs_f = if isa(kelly, NoKelly)
            (kwargs_f..., ylabel = "Expected Arithmetic Return")
        else
            (kwargs_f..., ylabel = "Expected Kelly Return")
        end
    end
    if !haskey(kwargs_f, :xlabel)
        kwargs_f = (kwargs_f..., xlabel = "Expected Risk - " * msg)
    end
    if !haskey(kwargs_f, :seriescolor)
        kwargs_f = (kwargs_f..., seriescolor = theme)
    end
    if !haskey(kwargs_f, :colorbar)
        kwargs_f = (kwargs_f..., colorbar = true)
    end
    if !haskey(kwargs_f, :colorbar_title)
        kwargs_f = (kwargs_f..., colorbar_title = "\nRisk Adjusted Return Ratio")
    end
    if !haskey(kwargs_f, :right_margin)
        kwargs_f = (kwargs_f..., right_margin = 4.5 * Plots.mm)
    end
    if !haskey(kwargs_f, :marker_z)
        kwargs_f = (kwargs_f..., marker_z = ratios)
    end
    if !haskey(kwargs_f, :label)
        kwargs_f = (kwargs_f..., label = "")
    end

    if frontier[:sharpe]
        if !haskey(kwargs_s, :label)
            kwargs_s = (kwargs_s..., label = "Max Risk Adjusted Return Ratio")
        end
        if !haskey(kwargs_s, :markershape)
            kwargs_s = (kwargs_s..., markershape = :star)
        end
        colour = palette(theme, length(ratios))[findlast(x -> x < ratios[end], ratios) + 1]
        if !haskey(kwargs_s, :color)
            kwargs_s = (kwargs_s..., color = colour)
        end
        if !haskey(kwargs_s, :markersize)
            kwargs_s = (kwargs_s..., markersize = 8)
        end
    end

    plt = if frontier[:sharpe]
        scatter(risks[1:(end - 1)], rets[1:(end - 1)]; kwargs_f...)
        scatter!([risks[end]], [rets[end]]; kwargs_s...)
    else
        scatter(risks[1:end], rets[1:end]; kwargs_f...)
    end

    return plt
end
function PortfolioOptimiser.plot_frontier(port::PortfolioOptimiser.AbstractPortfolio,
                                          key = nothing; X::AbstractMatrix = port.returns,
                                          mu::AbstractVector = port.mu,
                                          rm::PortfolioOptimiser.AbstractRiskMeasure = SD(),
                                          rf::Real = 0.0,
                                          kelly::PortfolioOptimiser.RetType = NoKelly(),
                                          t_factor = 252, theme = :Spectral, kwargs_f = (;),
                                          kwargs_s = (;))
    if isnothing(key)
        key = PortfolioOptimiser.get_rm_symb(rm)
    end
    return PortfolioOptimiser.plot_frontier(port.frontier[key]; X = X, mu = mu, rf = rf,
                                            rm = rm, kelly = kelly, t_factor = t_factor,
                                            theme = theme, kwargs_f = kwargs_f,
                                            kwargs_s = kwargs_s)
end

function PortfolioOptimiser.plot_frontier_area(frontier;
                                               rm::PortfolioOptimiser.AbstractRiskMeasure = SD(),
                                               t_factor = 252, kwargs_a = (;),
                                               theme = :Spectral, kwargs_l = (;),
                                               show_sharpe = true)
    risks = copy(frontier[:risks])
    assets = reshape(frontier[:weights][!, "tickers"], 1, :)
    weights = transpose(Matrix(frontier[:weights][!, 2:end]))

    if !any(typeof(rm) .<: (MDD, ADD, CDaR, EDaR, RLDaR, UCI))
        risks .*= sqrt(t_factor)
    end

    sharpe = nothing
    if frontier[:sharpe]
        sharpe = risks[end]
        risks = risks[1:(end - 1)]
        weights = weights[1:(end - 1), :]
    end

    msg = "$(PortfolioOptimiser.get_rm_symb(rm))"
    if any(typeof(rm) .<: (CVaR, TG, EVaR, RLVaR, CVaRRG, TGRG, CDaR, EDaR, RLDaR))
        msg *= " α = $(round(rm.alpha*100, digits=2))%"
    end
    if any(typeof(rm) .<: (CVaRRG, TGRG))
        msg *= ", β = $(round(rm.beta*100, digits=2))%"
    end
    if any(typeof(rm) .<: (RLVaR, RLDaR))
        msg *= ", κ = $(round(rm.kappa, digits=2))"
    end

    if !haskey(kwargs_a, :xlabel)
        kwargs_a = (kwargs_a..., xlabel = "Expected Risk - " * msg)
    end
    if !haskey(kwargs_a, :ylabel)
        kwargs_a = (kwargs_a..., ylabel = "Composition")
    end
    if !haskey(kwargs_a, :label)
        kwargs_a = (kwargs_a..., label = assets)
    end
    if !haskey(kwargs_a, :legend)
        kwargs_a = (kwargs_a..., legend = :outerright)
    end
    if !haskey(kwargs_a, :xtick)
        kwargs_a = (kwargs_a..., xtick = :auto)
    end
    if !haskey(kwargs_a, :xlim)
        kwargs_a = (kwargs_a..., xlim = extrema(risks))
    end
    if !haskey(kwargs_a, :ylim)
        kwargs_a = (kwargs_a..., ylim = (0, 1))
    end
    if !haskey(kwargs_a, :ylim)
        kwargs_a = (kwargs_a..., ylim = (0, 1))
    end
    if !haskey(kwargs_a, :seriescolor)
        kwargs_a = (kwargs_a...,
                    seriescolor = reshape(collect(palette(theme, length(assets))), 1, :))
    end

    plt = areaplot(risks, weights; kwargs_a...)

    if !isnothing(sharpe) && show_sharpe
        if !haskey(kwargs_l, :color)
            kwargs_l = (kwargs_l..., color = :black)
        end
        if !haskey(kwargs_l, :linewidth)
            kwargs_l = (kwargs_l..., linewidth = 3)
        end
        plot!(plt, [sharpe, sharpe], [0, 1]; label = nothing, kwargs_l...)
        annotate!([sharpe * 1.1], [0.5], text("Max Risk\nAdjusted\nReturn Ratio", :left, 8))
    end

    return plt
end
function PortfolioOptimiser.plot_frontier_area(port::PortfolioOptimiser.AbstractPortfolio,
                                               key = nothing; rm = SD(), t_factor = 252,
                                               theme = :Spectral, kwargs_a = (;),
                                               kwargs_l = (;), show_sharpe = true)
    if isnothing(key)
        key = PortfolioOptimiser.get_rm_symb(rm)
    end
    return PortfolioOptimiser.plot_frontier_area(port.frontier[key]; rm = rm,
                                                 t_factor = t_factor, theme = theme,
                                                 kwargs_a = kwargs_a, kwargs_l = kwargs_l,
                                                 show_sharpe = show_sharpe)
end

function PortfolioOptimiser.plot_drawdown(timestamps::AbstractVector, w::AbstractVector,
                                          returns::AbstractMatrix; alpha::Real = 0.05,
                                          kappa::Real = 0.3,
                                          solvers::Union{<:AbstractDict, Nothing} = nothing,
                                          theme = :Dark2_5, kwargs_ret = (;),
                                          kwargs_dd = (;), kwargs_risks = (;), kwargs = (;))
    ret = returns * w

    cret = copy(ret)
    pushfirst!(cret, 0)
    cret .+= 1
    cret = cumprod(cret)
    popfirst!(cret)
    ucret = cumsum(copy(ret)) .+ 1

    dd = similar(ucret)
    peak = -Inf
    for i ∈ eachindex(ucret)
        if ucret[i] > peak
            peak = ucret[i]
        end
        dd[i] = ucret[i] - peak
    end

    dd .*= 100

    risks = [-PortfolioOptimiser._ADD(ret), -PortfolioOptimiser._UCI(ret),
             -PortfolioOptimiser._DaR(ret, alpha), -PortfolioOptimiser._CDaR(ret, alpha),
             -PortfolioOptimiser._EDaR(ret, solvers, alpha),
             -PortfolioOptimiser._RLDaR(ret, solvers, alpha, kappa),
             -PortfolioOptimiser._MDD(ret)] * 100

    conf = round((1 - alpha) * 100; digits = 2)

    risk_labels = ("Average Drawdown: $(round(risks[1], digits = 2))%",
                   "Ulcer Index: $(round(risks[2], digits = 2))%",
                   "$(conf)% Confidence DaR: $(round(risks[3], digits = 2))%",
                   "$(conf)% Confidence CDaR: $(round(risks[4], digits = 2))%",
                   "$(conf)% Confidence EDaR: $(round(risks[5], digits = 2))%",
                   "$(conf)% Confidence RLDaR ($(round(kappa, digits=2))): $(round(risks[6], digits = 2))%",
                   "Maximum Drawdown: $(round(risks[7], digits = 2))%")

    colours = palette(theme, length(risk_labels) + 1)

    if !haskey(kwargs_dd, :ylabel)
        kwargs_dd = (kwargs_dd..., ylabel = "Percentage Drawdown")
    end
    if !haskey(kwargs_ret, :yguidefontsize)
        kwargs_ret = (kwargs_ret..., yguidefontsize = 10)
    end
    if !haskey(kwargs_dd, :xlabel)
        kwargs_dd = (kwargs_dd..., xlabel = "Date")
    end
    if !haskey(kwargs_dd, :ylim)
        kwargs_dd = (kwargs_dd..., ylim = [minimum(dd) * 1.5, 0.01])
    end
    if !haskey(kwargs_dd, :label)
        kwargs_dd = (kwargs_dd..., label = "Uncompounded Cummulative Drawdown")
    end
    if !haskey(kwargs_dd, :linewidth)
        kwargs_dd = (kwargs_dd..., linewidth = 2)
    end
    dd_plt = plot(timestamps, dd; color = colours[1], kwargs_dd...)

    if !haskey(kwargs_risks, :linewidth)
        kwargs_risks = (kwargs_risks..., linewidth = 2)
    end
    for (i, (risk, label)) ∈ enumerate(zip(risks, risk_labels))
        hline!([risk]; label = label, color = colours[i + 1], kwargs_risks...)
    end

    if !haskey(kwargs_ret, :ylabel)
        kwargs_ret = (kwargs_ret..., ylabel = "Cummulative Returns")
    end
    if !haskey(kwargs_ret, :yguidefontsize)
        kwargs_ret = (kwargs_ret..., yguidefontsize = 10)
    end
    if !haskey(kwargs_ret, :legend)
        kwargs_ret = (kwargs_ret..., legend = false)
    end
    if !haskey(kwargs_ret, :linewidth)
        kwargs_ret = (kwargs_ret..., linewidth = 2)
    end
    ret_plt = plot(timestamps, cret; color = colours[1], kwargs_ret...)

    if !haskey(kwargs, :legend_font_pointsize)
        kwargs = (kwargs..., legend_font_pointsize = 8)
    end
    if !haskey(kwargs, :size)
        kwargs = (kwargs..., size = (750, ceil(Integer, 750 / 1.618)))
    end
    full_plt = plot(ret_plt, dd_plt; layout = (2, 1), kwargs...)

    return full_plt
end
function PortfolioOptimiser.plot_drawdown(port::PortfolioOptimiser.AbstractPortfolio,
                                          type::Symbol = if isa(port, Portfolio)
                                              :Trad
                                          else
                                              :HRP
                                          end; alpha::Real = 0.05, kappa::Real = 0.3,
                                          allocated::Bool = false, theme = :Dark2_5,
                                          kwargs_ret = (;), kwargs_dd = (;),
                                          kwargs_risks = (;), kwargs = (;))
    return PortfolioOptimiser.plot_drawdown(port.timestamps,
                                            if !allocated
                                                port.optimal[type].weights
                                            else
                                                port.alloc_optimal[type].weights
                                            end, port.returns; alpha = alpha, kappa = kappa,
                                            solvers = port.solvers, theme = theme,
                                            kwargs_ret = kwargs_ret, kwargs_dd = kwargs_dd,
                                            kwargs_risks = kwargs_risks, kwargs = kwargs)
end

function PortfolioOptimiser.plot_hist(w::AbstractVector, returns::AbstractMatrix;
                                      alpha_i::Real = 0.0001, alpha::Real = 0.05,
                                      a_sim::Int = 100, kappa::Real = 0.3,
                                      solvers::Union{<:AbstractDict, Nothing} = nothing,
                                      points::Integer = ceil(Int,
                                                             4 * sqrt(size(returns, 1))),
                                      theme = :Paired_10, kwargs_h = (;),
                                      kwargs_risks = (;))
    ret = returns * w * 100

    mu = mean(ret)
    sigma = std(ret)

    x = range(minimum(ret); stop = maximum(ret), length = points)

    mad = PortfolioOptimiser._MAD(ret)
    gmd = PortfolioOptimiser._GMD(ret)
    risks = (mu, mu - sigma, mu - mad, mu - gmd, -PortfolioOptimiser._VaR(ret, alpha),
             -PortfolioOptimiser._CVaR(ret, alpha),
             -PortfolioOptimiser._TG(ret; alpha_i = alpha_i, alpha = alpha, a_sim = a_sim),
             -PortfolioOptimiser._EVaR(ret, solvers, alpha),
             -PortfolioOptimiser._RLVaR(x, solvers, alpha, kappa),
             -PortfolioOptimiser._WR(ret))

    conf = round((1 - alpha) * 100; digits = 2)

    risk_labels = ("Mean: $(round(risks[1], digits=2))%",
                   "Mean - Std. Dev. ($(round(sigma, digits=2))%): $(round(risks[2], digits=2))%",
                   "Mean - MAD ($(round(mad,digits=2))%): $(round(risks[3], digits=2))%",
                   "Mean - GMD ($(round(gmd,digits=2))%): $(round(risks[4], digits=2))%",
                   "$(conf)% Confidence VaR: $(round(risks[5], digits=2))%",
                   "$(conf)% Confidence CVaR: $(round(risks[6], digits=2))%",
                   "$(conf)% Confidence Tail Gini: $(round(risks[7], digits=2))%",
                   "$(conf)% Confidence EVaR: $(round(risks[8], digits=2))%",
                   "$(conf)% Confidence RLVaR ($(round(kappa, digits=2))): $(round(risks[9], digits=2))%",
                   "Worst Realisation: $(round(risks[10], digits=2))%")

    D = fit(Normal, ret)

    if !haskey(kwargs_h, :ylabel)
        kwargs_h = (kwargs_h..., ylabel = "Probability Density")
    end
    if !haskey(kwargs_h, :xlabel)
        kwargs_h = (kwargs_h..., xlabel = "Percentage Returns")
    end

    colours = palette(theme, length(risk_labels) + 2)

    plt = histogram(ret; normalize = :pdf, label = "", color = colours[1], kwargs_h...)

    if !haskey(kwargs_risks, :linewidth)
        kwargs_risks = (kwargs_risks..., linewidth = 2)
    end
    for (i, (risk, label)) ∈ enumerate(zip(risks, risk_labels))
        vline!([risk]; label = label, color = colours[i + 1], kwargs_risks...)
    end

    if !haskey(kwargs_h, :size)
        kwargs_h = (kwargs_h..., size = (750, ceil(Integer, 750 / 1.618)))
    end

    if !haskey(kwargs_h, :linewidth)
        kwargs_h = (kwargs_h..., linewidth = 2)
    end
    plot!(x, pdf.(D, x);
          label = "Normal: μ = $(round(mean(D), digits=2))%, σ = $(round(std(D), digits=2))%",
          color = colours[end], kwargs_h...)

    return plt
end
function PortfolioOptimiser.plot_hist(port::PortfolioOptimiser.AbstractPortfolio,
                                      type::Symbol = if isa(port, Portfolio)
                                          :Trad
                                      else
                                          :HRP
                                      end;
                                      points::Integer = ceil(Int,
                                                             4 *
                                                             sqrt(size(port.returns, 1))),
                                      alpha_i::Real = 0.0001, alpha::Real = 0.05,
                                      a_sim::Int = 100, kappa::Real = 0.3,
                                      allocated::Bool = false, theme = :Paired_10,
                                      kwargs_h = (;), kwargs_risks = (;))
    return PortfolioOptimiser.plot_hist(if !allocated
                                            port.optimal[type].weights
                                        else
                                            port.alloc_optimal[type].weights
                                        end, port.returns; alpha_i = alpha_i, alpha = alpha,
                                        a_sim = a_sim, kappa = kappa,
                                        solvers = port.solvers, theme = theme,
                                        points = points, kwargs_h = kwargs_h,
                                        kwargs_risks = kwargs_risks)
end

function PortfolioOptimiser.plot_range(w::AbstractVector, returns::AbstractMatrix;
                                       alpha_i::Real = 0.0001, alpha::Real = 0.05,
                                       a_sim::Int = 100, beta_i::Real = alpha_i,
                                       beta::Real = alpha, b_sim::Integer = a_sim,
                                       theme = :Set1_5, kwargs_h = (;), kwargs_risks = (;))
    if isinf(beta)
        beta = alpha
    end

    ret = returns * w * 100

    risks = (PortfolioOptimiser._RG(ret),
             PortfolioOptimiser._CVaRRG(ret; alpha = alpha, beta = beta),
             PortfolioOptimiser._TGRG(ret; alpha_i = alpha_i, alpha = alpha, a_sim = a_sim,
                                      beta_i = beta_i, beta = beta, b_sim = b_sim))

    lo_conf = 1 - alpha
    hi_conf = 1 - beta
    risk_labels = ("Range: $(round(risks[1], digits=2))%",
                   "Tail Gini Range ($(round(lo_conf,digits=2)), $(round(hi_conf,digits=2))): $(round(risks[2], digits=2))%",
                   "CVaR Range ($(round(lo_conf,digits=2)), $(round(hi_conf,digits=2))): $(round(risks[3], digits=2))%")

    colours = palette(theme, length(risk_labels) + 1)

    if !haskey(kwargs_h, :ylabel)
        kwargs_h = (kwargs_h..., ylabel = "Probability Density")
    end
    if !haskey(kwargs_h, :xlabel)
        kwargs_h = (kwargs_h..., xlabel = "Percentage Returns")
    end

    plt = histogram(ret; normalize = :pdf, label = "", color = colours[1], kwargs_h...)

    bounds = [minimum(ret) -PortfolioOptimiser._TG(ret; alpha_i = alpha_i, alpha = alpha, a_sim = a_sim) -PortfolioOptimiser._CVaR(ret, alpha);
              maximum(ret) PortfolioOptimiser._TG(-ret; alpha_i = alpha_i, alpha = alpha, a_sim = a_sim) PortfolioOptimiser._CVaR(-ret, alpha)]

    D = fit(Normal, ret)
    y = pdf(D, mean(D))
    ys = (y / 4, y / 2, y * 3 / 4)

    if !haskey(kwargs_risks, :linewidth)
        kwargs_risks = (kwargs_risks..., linewidth = 2)
    end
    for i ∈ eachindex(risks)
        plot!([bounds[1, i], bounds[1, i], bounds[2, i], bounds[2, i]],
              [0, ys[i], ys[i], 0]; label = risk_labels[i], color = colours[i + 1],
              kwargs_risks...)
    end

    return plt
end

function PortfolioOptimiser.plot_range(port::PortfolioOptimiser.AbstractPortfolio,
                                       type::Symbol = if isa(port, Portfolio)
                                           :Trad
                                       else
                                           :HRP
                                       end; alpha_i::Real = 0.0001, alpha::Real = 0.05,
                                       a_sim::Int = 100, beta_i::Real = alpha_i,
                                       beta::Real = alpha, b_sim::Integer = a_sim,
                                       allocated::Bool = false, theme = :Set1_5,
                                       kwargs_h = (;), kwargs_risks = (;))
    return PortfolioOptimiser.plot_range(if !allocated
                                             port.optimal[type].weights
                                         else
                                             port.alloc_optimal[type].weights
                                         end, port.returns; alpha_i = alpha_i,
                                         alpha = alpha, a_sim = a_sim, beta_i = beta_i,
                                         beta = beta, b_sim = b_sim, theme = theme,
                                         kwargs_h = kwargs_h, kwargs_risks = kwargs_risks)
end

function PortfolioOptimiser.plot_clusters(assets::AbstractVector, rho::AbstractMatrix,
                                          clustering_idx::AbstractVector{<:Integer},
                                          clustering::Hclust, k::Integer,
                                          clim::Tuple{<:Real, <:Real} = (-1, 1);
                                          show_clusters = true, theme_d = :Spectral,
                                          theme_h = :Spectral, theme_h_kwargs = (;),
                                          kwargs_d1 = (;), kwargs_d2 = (;), kwargs_h = (;),
                                          kwargs_l = (;), kwargs = (;))
    N = length(assets)

    heights = clustering.heights
    sort_order = clustering.order
    ordered_corr = rho[sort_order, sort_order]
    ordered_assets = assets[sort_order]

    uidx = minimum(clustering_idx):maximum(clustering_idx)

    clusters = Vector{Vector{Int}}(undef, length(uidx))
    for i ∈ eachindex(clusters)
        clusters[i] = findall(clustering_idx .== i)
    end

    colours = palette(theme_d, k)
    colgrad = cgrad(theme_h; theme_h_kwargs...)

    #yticks=(1:nrows,rowlabels)
    hmap = plot(ordered_corr; st = :heatmap, yticks = (1:length(assets), ordered_assets),
                xticks = (1:length(assets), ordered_assets), xrotation = 90,
                colorbar = false, clim = clim, xlim = (0.5, N + 0.5), ylim = (0.5, N + 0.5),
                color = colgrad, yflip = true, kwargs_h...)
    dend1 = plot(clustering; xticks = false, ylim = (0, 1), kwargs_d1...)
    dend2 = plot(clustering; yticks = false, xrotation = 90, orientation = :horizontal,
                 yflip = true, xlim = (0, 1), kwargs_d2...)

    if !haskey(kwargs_l, :color)
        kwargs_l = (kwargs_l..., color = :black)
    end
    if !haskey(kwargs_l, :linewidth)
        kwargs_l = (kwargs_l..., linewidth = 3)
    end

    nodes = -clustering.merges
    if show_clusters
        for (i, cluster) ∈ enumerate(clusters)
            a = [findfirst(x -> x == c, sort_order) for c ∈ cluster]
            a = a[.!isnothing.(a)]
            xmin = minimum(a)
            xmax = xmin + length(cluster)

            i1 = [findfirst(x -> x == c, nodes[:, 1]) for c ∈ cluster]
            i1 = i1[.!isnothing.(i1)]
            i2 = [findfirst(x -> x == c, nodes[:, 2]) for c ∈ cluster]
            i2 = i2[.!isnothing.(i2)]
            i3 = unique([i1; i2])
            h = min(maximum(heights[i3]) * 1.1, 1)

            plot!(hmap,
                  [xmin - 0.5, xmax - 0.5, xmax - 0.5, xmax - 0.5, xmax - 0.5, xmin - 0.5,
                   xmin - 0.5, xmin - 0.5],
                  [xmin - 0.5, xmin - 0.5, xmin - 0.5, xmax - 0.5, xmax - 0.5, xmax - 0.5,
                   xmax - 0.5, xmin - 0.5]; legend = false, kwargs_l...)

            plot!(dend1,
                  [xmin - 0.25, xmax - 0.75, xmax - 0.75, xmax - 0.75, xmax - 0.75,
                   xmin - 0.25, xmin - 0.25, xmin - 0.25], [0, 0, 0, h, h, h, h, 0];
                  color = nothing, legend = false,
                  fill = (0, 0.5, colours[(i - 1) % k + 1]))

            plot!(dend2, [0, 0, 0, h, h, h, h, 0],
                  [xmin - 0.25, xmax - 0.75, xmax - 0.75, xmax - 0.75, xmax - 0.75,
                   xmin - 0.25, xmin - 0.25, xmin - 0.25]; color = nothing, legend = false,
                  fill = (0, 0.5, colours[(i - 1) % k + 1]))
        end
    end

    if !haskey(kwargs, :size)
        kwargs = (kwargs..., size = (600, 600))
    end

    # https://docs.juliaplots.org/latest/generated/statsplots/#Dendrogram-on-the-right-side
    l = StatsPlots.grid(2, 2; heights = [0.2, 0.8], widths = [0.8, 0.2])
    plt = plot(dend1, plot(; ticks = nothing, border = :none, background_color = nothing),
               hmap, dend2; layout = l, kwargs...)

    return plt
end

function PortfolioOptimiser.plot_clusters(port::PortfolioOptimiser.AbstractPortfolio;
                                          cor_type::PortfolioOptimiser.PortfolioOptimiserCovCor = PortCovCor(),
                                          dist_type::PortfolioOptimiser.DistanceMethod = DistanceCanonical(),
                                          hclust_alg::PortfolioOptimiser.HClustAlg = HAC(),
                                          hclust_opt::PortfolioOptimiser.HCOpt = HCOpt(),
                                          cluster::Bool = true, show_clusters::Bool = true,
                                          theme_d = :Spectral, theme_h = :Spectral,
                                          theme_h_kwargs = (;), kwargs_d1 = (;),
                                          kwargs_d2 = (;), kwargs_h = (;), kwargs_l = (;),
                                          kwargs = (;))
    idx, clustering, k, S = if cluster || isa(port, PortfolioOptimiser.Portfolio)
        cluster_assets(port.returns; cor_type = cor_type, dist_type = dist_type,
                       hclust_alg = hclust_alg, hclust_opt = hclust_opt)[1:4]
    else
        clustering, k, S = port.clusters, port.k, port.cor
        idx = cutree(clustering; k = k)
        idx, clustering, k, S
    end

    clim = if isnothing(findfirst(x -> x < zero(eltype(S)), S))
        (0, 1)
    else
        (-1, 1)
    end

    return plot_clusters(port.assets, S, idx, clustering, k, clim;
                         show_clusters = show_clusters, theme_d = theme_d,
                         theme_h = theme_h, theme_h_kwargs = theme_h_kwargs,
                         kwargs_d1 = kwargs_d1, kwargs_d2 = kwargs_d2, kwargs_h = kwargs_h,
                         kwargs_l = kwargs_l, kwargs = kwargs)
end

function PortfolioOptimiser.plot_dendrogram(assets::AbstractVector,
                                            clustering_idx::AbstractVector{<:Integer},
                                            clustering::Hclust, k::Integer;
                                            show_clusters = true, theme = :Spectral,
                                            kwargs_d = (;), kwargs = (;))
    heights = clustering.heights
    sort_order = clustering.order
    ordered_assets = assets[sort_order]

    uidx = minimum(clustering_idx):maximum(clustering_idx)

    clusters = Vector{Vector{Int}}(undef, length(uidx))
    for i ∈ eachindex(clusters)
        clusters[i] = findall(clustering_idx .== i)
    end

    colours = palette(theme, k)

    dend1 = plot(clustering; normalize = false, xticks = (1:length(assets), ordered_assets),
                 ylim = (0, 1), xrotation = 90, kwargs_d...)

    nodes = -clustering.merges
    if show_clusters
        for (i, cluster) ∈ enumerate(clusters)
            a = [findfirst(x -> x == c, sort_order) for c ∈ cluster]
            a = a[.!isnothing.(a)]
            xmin = minimum(a)
            xmax = xmin + length(cluster)

            i1 = [findfirst(x -> x == c, nodes[:, 1]) for c ∈ cluster]
            i1 = i1[.!isnothing.(i1)]
            i2 = [findfirst(x -> x == c, nodes[:, 2]) for c ∈ cluster]
            i2 = i2[.!isnothing.(i2)]
            i3 = unique([i1; i2])
            h = min(maximum(heights[i3]) * 1.1, 1)

            plot!(dend1,
                  [xmin - 0.25, xmax - 0.75, xmax - 0.75, xmax - 0.75, xmax - 0.75,
                   xmin - 0.25, xmin - 0.25, xmin - 0.25], [0, 0, 0, h, h, h, h, 0];
                  color = nothing, legend = false,
                  fill = (0, 0.5, colours[(i - 1) % k + 1]))
        end
    end

    plt = plot(dend1; kwargs...)

    return plt
end

function PortfolioOptimiser.plot_dendrogram(port::PortfolioOptimiser.AbstractPortfolio;
                                            cor_type::PortfolioOptimiser.PortfolioOptimiserCovCor = PortCovCor(),
                                            dist_type::PortfolioOptimiser.DistanceMethod = DistanceCanonical(),
                                            hclust_alg::PortfolioOptimiser.HClustAlg = HAC(),
                                            hclust_opt::PortfolioOptimiser.HCOpt = HCOpt(),
                                            cluster::Bool = true, theme = :Spectral,
                                            kwargs_d = (;), kwargs = (;))
    idx, clustering, k = if cluster || isa(port, PortfolioOptimiser.Portfolio)
        cluster_assets(port.returns; cor_type = cor_type, dist_type = dist_type,
                       hclust_alg = hclust_alg, hclust_opt = hclust_opt)[1:3]
    else
        clustering, k = port.clusters, port.k
        idx = cutree(clustering; k = k)
        idx, clustering, k
    end

    return plot_dendrogram(port.assets, idx, clustering, k; show_clusters = true,
                           theme = theme, kwargs_d = kwargs_d, kwargs = kwargs)
end

function PortfolioOptimiser.plot_network(assets::AbstractVector, rho::AbstractMatrix,
                                         delta::AbstractMatrix,
                                         clustering_idx::AbstractVector{<:Integer};
                                         k::Integer = length(unique(clustering_idx)),
                                         network_type::PortfolioOptimiser.NetworkType = MST(),
                                         allocation = true, w = nothing, theme = :Spectral,
                                         kwargs = (;))
    G = PortfolioOptimiser._calc_adjacency(network_type, rho, delta)

    colours = palette(theme, k)

    if !haskey(kwargs, :names)
        assets = assets
        ml = maximum(length.(assets))
        names = similar(assets)
        for (i, asset) ∈ pairs(assets)
            diff = ml - length(asset)
            d, r = divrem(diff, 2)
            name = repeat(" ", d + r + 1) * asset * repeat(" ", d + 1)
            names[i] = name
        end
        kwargs = (kwargs..., names = names)
    end

    if !haskey(kwargs, :nodecolor)
        nodecols = Vector{eltype(colours)}(undef, size(rho, 1))
        for (i, j) ∈ pairs(clustering_idx)
            nodecols[i] = colours[j]
        end
        kwargs = (kwargs..., nodecolor = nodecols)
    end

    if !haskey(kwargs, :nodeshape)
        kwargs = (kwargs..., nodeshape = :circle)
    end

    if allocation && !isnothing(w)
        kwargs = (kwargs..., nodeweights = w)
    end

    return graphplot(G; kwargs...)
end

function PortfolioOptimiser.plot_network(port::PortfolioOptimiser.AbstractPortfolio,
                                         type::Symbol = if isa(port, Portfolio)
                                             :Trad
                                         else
                                             :HRP
                                         end;
                                         cor_type::PortfolioOptimiser.PortfolioOptimiserCovCor = PortCovCor(),
                                         dist_type::PortfolioOptimiser.DistanceMethod = DistanceCanonical(),
                                         hclust_alg::PortfolioOptimiser.HClustAlg = HAC(),
                                         hclust_opt::PortfolioOptimiser.HCOpt = HCOpt(),
                                         network_type::PortfolioOptimiser.NetworkType = MST(),
                                         cluster::Bool = true, allocation::Bool = true,
                                         allocated::Bool = false, theme = :Spectral,
                                         kwargs = (;))
    idx, clustering, k, S, D = if cluster || isa(port, PortfolioOptimiser.Portfolio)
        cluster_assets(port.returns; cor_type = cor_type, dist_type = dist_type,
                       hclust_alg = hclust_alg, hclust_opt = hclust_opt)
    else
        clustering, k, S, D = port.clusters, port.k, port.cor, port.dist
        idx = cutree(clustering; k = k)
        idx, clustering, k, S, D
    end

    return plot_network(port.assets, S, D, idx; k = k, network_type = network_type,
                        allocation = allocation, w = if !allocated
                            port.optimal[type].weights
                        else
                            port.alloc_optimal[type].weights
                        end, theme = theme, kwargs = kwargs)
end

end
