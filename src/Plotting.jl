function plot_returns(timestamps, assets, returns, weights; per_asset = false, kwargs...)
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
function plot_returns(portfolio, type = isa(portfolio, HCPortfolio) ? :HRP : :Trad;
                      per_asset = false, kwargs...)
    return plot_returns(portfolio.timestamps, portfolio.assets, portfolio.returns,
                        portfolio.optimal[type].weights; per_asset = per_asset, kwargs...)
end

function plot_bar(assets, data; kwargs...)
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
function plot_bar(portfolio::AbstractPortfolio,
                  type = isa(portfolio, HCPortfolio) ? :HRP : :Trad, kwargs...)
    return plot_bar(portfolio.assets, portfolio.optimal[type].weights, kwargs...)
end

function plot_risk_contribution(
                                # RC args
                                assets::AbstractVector, w::AbstractVector,
                                returns::AbstractMatrix; rm::Symbol = :SD, rf::Real = 0.0,
                                sigma::AbstractMatrix, alpha_i::Real = 0.0001,
                                alpha::Real = 0.05, a_sim::Int = 100,
                                beta_i::Real = alpha_i, beta::Real = alpha,
                                b_sim::Integer = a_sim, di::Real = 1e-6, kappa::Real = 0.3,
                                owa_w::AbstractVector{<:Real} = Vector{Float64}(undef, 0),
                                solvers::Union{<:AbstractDict, Nothing} = nothing,                                # Plot args
                                percentage::Bool = false, erc_line::Bool = true,
                                t_factor = 252, kwargs_bar = (;), kwargs_line = (;))
    rc = risk_contribution(w, returns; rm = rm, rf = rf, sigma = sigma, alpha_i = alpha_i,
                           alpha = alpha, a_sim = a_sim, beta_i = beta_i, beta = beta,
                           b_sim = b_sim, di = di, kappa = kappa, owa_w = owa_w,
                           solvers = solvers)

    DDs = (:DaR, :MDD, :ADD, :CDaR, :EDaR, :RDaR, :UCI, :DaR_r, :MDD_r, :ADD_r, :CDaR_r,
           :EDaR_r, :RDaR_r, :UCI_r)

    if rm ∉ DDs
        rc *= sqrt(t_factor)
    end
    ylabel = "Risk Contribution"
    if percentage
        rc /= sum(rc)
        ylabel *= " Percentage"
    end

    title = "Risk Contribution - $(RiskMeasureNames[rm]) ($(rm))"
    if rm ∈ (:CVaR, :TG, :EVaR, :RVaR, :RCVaR, :RTG, :CDaR, :EDaR, :RDaR)
        title *= " α = $(round(alpha*100, digits=2))%"
    end
    if rm ∈ (:RCVaR, :RTG)
        title *= ", β = $(round(beta*100, digits=2))%"
    end
    if rm ∈ (:RVaR, :RDaR)
        title *= ", κ = $(round(kappa, digits=2))"
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
            erc = calc_risk(w, returns; rm = rm, rf = rf, sigma = sigma, alpha_i = alpha_i,
                            alpha = alpha, a_sim = a_sim, beta_i = beta_i, beta = beta,
                            b_sim = b_sim, kappa = kappa, owa_w = owa_w, solvers = solvers)

            erc /= length(rc)
            if rm ∉ DDs
                erc *= sqrt(t_factor)
            end
        end

        hline!([erc]; kwargs_line...)
    end

    return plt
end
function plot_risk_contribution(portfolio; di::Real = 1e-6,
                                type::Symbol = isa(portfolio, Portfolio) ? :Trad : :HRP,
                                rm::Symbol = :SD, rf::Real = 0.0,
                                owa_w::AbstractVector{<:Real} = if isa(portfolio,
                                                                       Portfolio)
                                    portfolio.owa_w
                                else
                                    Vector{Float64}(undef, 0)
                                end, percentage::Bool = false, erc_line::Bool = true,
                                t_factor = 252, kwargs_bar = (;), kwargs_line = (;))
    return plot_risk_contribution(
                                  # RC args
                                  portfolio.assets, portfolio.optimal[type].weights,
                                  portfolio.returns; rm = rm, rf = rf,
                                  sigma = portfolio.cov, alpha_i = portfolio.alpha_i,
                                  alpha = portfolio.alpha, a_sim = portfolio.a_sim,
                                  beta_i = portfolio.beta_i, beta = portfolio.beta,
                                  b_sim = portfolio.b_sim, di = di, kappa = portfolio.kappa,
                                  owa_w = owa_w, solvers = portfolio.solvers,                                  # Plot args
                                  percentage = percentage, erc_line = erc_line,
                                  t_factor = t_factor, kwargs_bar = kwargs_bar,
                                  kwargs_line = kwargs_line)
end

function plot_frontier(frontier; alpha::Real = 0.05, beta::Real = alpha, kappa::Real = 0.3,
                       returns::AbstractMatrix = Matrix{Float64}(undef, 0, 0),
                       t_factor = 252, kelly::Bool = false,
                       mu::AbstractVector = Vector{Float64}(undef, 0), rf::Real = 0.0,
                       rm::Symbol = :SD, theme = :Spectral, kwargs_f = (;), kwargs_s = (;))
    @smart_assert(rm ∈ RiskMeasures)

    if isinf(beta)
        beta = alpha
    end

    risks = copy(frontier[:risk])
    weights = Matrix(frontier[:weights][!, 2:end])

    rets = if kelly
        1 / size(returns, 1) * vec(sum(log.(1 .+ returns * weights); dims = 1))
    else
        transpose(weights) * mu
    end
    rets .*= t_factor

    if rm ∉ (:MDD, :ADD, :CDaR, :EDaR, :RDaR, :UCI)
        risks .*= sqrt(t_factor)
    end

    ratios = (rets .- rf) ./ risks

    msg = "$(RiskMeasureNames[rm]) ($(rm))"
    if rm ∈ (:CVaR, :TG, :EVaR, :RVaR, :RCVaR, :RTG, :CDaR, :EDaR, :RDaR)
        msg *= " α = $(round(alpha*100, digits=2))%"
    end
    if rm ∈ (:RCVaR, :RTG)
        msg *= ", β = $(round(beta*100, digits=2))%"
    end
    if rm ∈ (:RVaR, :RDaR)
        msg *= ", κ = $(round(kappa, digits=2))"
    end

    if !haskey(kwargs_f, :ylabel)
        kwargs_f = if !kelly
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
function plot_frontier(portfolio::AbstractPortfolio; rm::Symbol = :SD, rf::Real = 0.0,
                       kelly::Bool = false, t_factor = 252, theme = :Spectral,
                       kwargs_f = (;), kwargs_s = (;))
    return plot_frontier(portfolio.frontier[rm]; alpha = portfolio.alpha,
                         beta = portfolio.beta, kappa = portfolio.kappa, mu = portfolio.mu,
                         returns = portfolio.returns, t_factor = t_factor, kelly = kelly,
                         rf = rf, rm = rm, theme = theme, kwargs_f = kwargs_f,
                         kwargs_s = kwargs_s)
end

function plot_frontier_area(frontier; alpha::Real = 0.05, beta::Real = alpha,
                            kappa::Real = 0.3, rm = :SD, t_factor = 252, kwargs_a = (;),
                            kwargs_l = (;), show_sharpe = true)
    risks = copy(frontier[:risk])
    assets = reshape(frontier[:weights][!, "tickers"], 1, :)
    weights = transpose(Matrix(frontier[:weights][!, 2:end]))
    if isinf(beta)
        beta = alpha
    end

    if rm ∉ (:MDD, :ADD, :CDaR, :EDaR, :RDaR, :UCI)
        risks .*= sqrt(t_factor)
    end

    sharpe = nothing
    if frontier[:sharpe]
        sharpe = risks[end]
        risks = risks[1:(end - 1)]
        weights = weights[1:(end - 1), :]
    end

    msg = "$(RiskMeasureNames[rm]) ($(rm))"
    if rm ∈ (:CVaR, :TG, :EVaR, :RVaR, :RCVaR, :RTG, :CDaR, :EDaR, :RDaR)
        msg *= " α = $(round(alpha*100, digits=2))%"
    end
    if rm ∈ (:RCVaR, :RTG)
        msg *= ", β = $(round(beta*100, digits=2))%"
    end
    if rm ∈ (:RVaR, :RDaR)
        msg *= ", κ = $(round(kappa, digits=2))"
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
                    seriescolor = reshape(collect(palette(:Spectral, length(assets))), 1,
                                          :))
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
function plot_frontier_area(portfolio::AbstractPortfolio; rm = :SD, t_factor = 252,
                            kwargs_a = (;), kwargs_l = (;), show_sharpe = true)
    return plot_frontier_area(portfolio.frontier[rm]; alpha = portfolio.alpha,
                              beta = portfolio.beta, kappa = portfolio.kappa, rm = rm,
                              t_factor = t_factor, kwargs_a = kwargs_a, kwargs_l = kwargs_l,
                              show_sharpe = show_sharpe)
end

function plot_drawdown(timestamps::AbstractVector, w::AbstractVector,
                       returns::AbstractMatrix; alpha::Real = 0.05, kappa::Real = 0.3,
                       solvers::Union{<:AbstractDict, Nothing} = nothing, theme = :Dark2_5,
                       kwargs_ret = (;), kwargs_dd = (;), kwargs_risks = (;), kwargs = (;))
    ret = returns * w

    prices = copy(ret)
    pushfirst!(prices, 0)
    prices .+= 1
    prices = cumprod(prices)
    popfirst!(prices)
    prices2 = cumsum(copy(ret)) .+ 1

    dd = similar(prices2)
    peak = -Inf
    for i ∈ eachindex(prices2)
        if prices2[i] > peak
            peak = prices2[i]
        end
        dd[i] = prices2[i] - peak
    end

    dd .*= 100

    risks = [-ADD_abs(ret), -UCI_abs(ret), -DaR_abs(ret, alpha), -CDaR_abs(ret, alpha),
             -EDaR_abs(ret, solvers, alpha), -RDaR_abs(ret, solvers, alpha, kappa),
             -MDD_abs(ret)] * 100

    conf = round((1 - alpha) * 100; digits = 2)

    risk_labels = ("Average Drawdown: $(round(risks[1], digits = 2))%",
                   "Ulcer Index: $(round(risks[2], digits = 2))%",
                   "$(conf)% Confidence DaR: $(round(risks[3], digits = 2))%",
                   "$(conf)% Confidence CDaR: $(round(risks[4], digits = 2))%",
                   "$(conf)% Confidence EDaR: $(round(risks[5], digits = 2))%",
                   "$(conf)% Confidence RDaR ($(round(kappa, digits=2))): $(round(risks[6], digits = 2))%",
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
    ret_plt = plot(timestamps, prices; color = colours[1], kwargs_ret...)

    if !haskey(kwargs, :legend_font_pointsize)
        kwargs = (kwargs..., legend_font_pointsize = 8)
    end
    if !haskey(kwargs, :size)
        kwargs = (kwargs..., size = (750, ceil(Integer, 750 / 1.618)))
    end
    full_plt = plot(ret_plt, dd_plt; layout = (2, 1), kwargs...)

    return full_plt
end
function plot_drawdown(portfolio::AbstractPortfolio;
                       type::Symbol = isa(portfolio, Portfolio) ? :Trad : :HRP,
                       theme = :Dark2_5, kwargs_ret = (;), kwargs_dd = (;),
                       kwargs_risks = (;), kwargs = (;))
    return plot_drawdown(portfolio.timestamps, portfolio.optimal[type].weights,
                         portfolio.returns; alpha = portfolio.alpha,
                         kappa = portfolio.kappa, solvers = portfolio.solvers,
                         theme = theme, kwargs_ret = kwargs_ret, kwargs_dd = kwargs_dd,
                         kwargs_risks = kwargs_risks, kwargs = kwargs)
end

function plot_hist(w::AbstractVector, returns::AbstractMatrix; alpha_i::Real = 0.0001,
                   alpha::Real = 0.05, a_sim::Int = 100, kappa::Real = 0.3,
                   solvers::Union{<:AbstractDict, Nothing} = nothing,
                   points::Integer = ceil(Int, 4 * sqrt(size(returns, 1))),
                   theme = :Paired_10, kwargs_h = (;), kwargs_risks = (;))
    ret = returns * w * 100

    mu = mean(ret)
    sigma = std(ret)

    x = range(minimum(ret); stop = maximum(ret), length = points)

    mad = MAD(ret)
    gmd = GMD(ret)
    risks = (mu, mu - sigma, mu - mad, mu - gmd, -VaR(ret, alpha), -CVaR(ret, alpha),
             -TG(ret; alpha_i = alpha_i, alpha = alpha, a_sim = a_sim),
             -EVaR(ret, solvers, alpha), -RVaR(x, solvers, alpha, kappa), -WR(ret))

    conf = round((1 - alpha) * 100; digits = 2)

    risk_labels = ("Mean: $(round(risks[1], digits=2))%",
                   "Mean - Std. Dev. ($(round(sigma, digits=2))%): $(round(risks[2], digits=2))%",
                   "Mean - MAD ($(round(mad,digits=2))%): $(round(risks[3], digits=2))%",
                   "Mean - GMD ($(round(gmd,digits=2))%): $(round(risks[4], digits=2))%",
                   "$(conf)% Confidence VaR: $(round(risks[5], digits=2))%",
                   "$(conf)% Confidence CVaR: $(round(risks[6], digits=2))%",
                   "$(conf)% Confidence Tail Gini: $(round(risks[7], digits=2))%",
                   "$(conf)% Confidence EVaR: $(round(risks[8], digits=2))%",
                   "$(conf)% Confidence RVaR ($(round(kappa, digits=2))): $(round(risks[9], digits=2))%",
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
function plot_hist(portfolio::AbstractPortfolio;
                   type::Symbol = isa(portfolio, Portfolio) ? :Trad : :HRP,
                   points::Integer = ceil(Int, 4 * sqrt(size(portfolio.returns, 1))),
                   theme = :Paired_10, kwargs_h = (;), kwargs_risks = (;))
    return plot_hist(portfolio.optimal[type].weights, portfolio.returns;
                     alpha_i = portfolio.alpha_i, alpha = portfolio.alpha,
                     a_sim = portfolio.a_sim, kappa = portfolio.kappa,
                     solvers = portfolio.solvers, theme = theme, points = points,
                     kwargs_h = kwargs_h, kwargs_risks = kwargs_risks)
end

function plot_range(w::AbstractVector, returns::AbstractMatrix; alpha_i::Real = 0.0001,
                    alpha::Real = 0.05, a_sim::Int = 100, beta_i::Real = alpha_i,
                    beta::Real = alpha, b_sim::Integer = a_sim, theme = :Set1_5,
                    kwargs_h = (;), kwargs_risks = (;))
    if isinf(beta)
        beta = alpha
    end

    ret = returns * w * 100

    risks = (RG(ret), RCVaR(ret; alpha = alpha, beta = beta),
             RTG(ret; alpha_i = alpha_i, alpha = alpha, a_sim = a_sim, beta_i = beta_i,
                 beta = beta, b_sim = b_sim))

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

    bounds = [minimum(ret) -TG(ret; alpha_i = alpha_i, alpha = alpha, a_sim = a_sim) -CVaR(ret, alpha);
              maximum(ret) TG(-ret; alpha_i = alpha_i, alpha = alpha, a_sim = a_sim) CVaR(-ret, alpha)]

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

function plot_range(portfolio::AbstractPortfolio;
                    type::Symbol = isa(portfolio, Portfolio) ? :Trad : :HRP,
                    theme = :Set1_5, kwargs_h = (;), kwargs_risks = (;))
    return plot_range(portfolio.optimal[type].weights, portfolio.returns;
                      alpha_i = portfolio.alpha_i, alpha = portfolio.alpha,
                      a_sim = portfolio.a_sim, beta_i = portfolio.beta_i,
                      beta = portfolio.beta, b_sim = portfolio.b_sim, theme = theme,
                      kwargs_h = kwargs_h, kwargs_risks = kwargs_risks)
end

function plot_clusters(portfolio; max_k = ceil(Int, sqrt(size(portfolio.dist, 1))),
                       linkage = :single, branchorder = :optimal, dbht_method = :Unique,
                       cluster = true, show_clusters = true, theme_d = :Spectral,
                       theme_h = :Spectral, theme_h_kwargs = (;), kwargs_d1 = (;),
                       kwargs_d2 = (;), kwargs_h = (;), kwargs_l = (;), kwargs = (;))
    corr = portfolio.cor
    assets = portfolio.assets
    cor_method = portfolio.cor_method
    N = length(assets)

    if cluster
        clustering_idx, clustering, k = cluster_assets(portfolio; linkage = linkage,
                                                       max_k = max_k,
                                                       branchorder = branchorder,
                                                       k = portfolio.k,
                                                       dbht_method = dbht_method)
        sort_order = clustering.order
        heights = clustering.heights
    else
        clustering = portfolio.clusters
        sort_order = clustering.order
        heights = clustering.heights
        k = portfolio.k
        clustering_idx = cutree(clustering; k = k)
    end

    ordered_corr = corr[sort_order, sort_order]
    ordered_assets = assets[sort_order]

    uidx = minimum(clustering_idx):maximum(clustering_idx)

    clusters = Vector{Vector{Int}}(undef, length(uidx))
    for i ∈ eachindex(clusters)
        clusters[i] = findall(clustering_idx .== i)
    end

    cors = (:Pearson, :Semi_Pearson, :Spearman, :Kendall, :Gerber1, :Gerber2, :custom)
    if cor_method ∈ cors
        clim = (-1, 1)
    else
        clim = (0, 1)
    end

    colours = palette(theme_d, k)
    colgrad = cgrad(theme_h; theme_h_kwargs...)

    #yticks=(1:nrows,rowlabels)
    hmap = plot(ordered_corr; st = :heatmap, yticks = (1:length(assets), ordered_assets),
                xticks = (1:length(assets), ordered_assets), xrotation = 90,
                colorbar = false, clim = clim, xlim = (0.5, N + 0.5), ylim = (0.5, N + 0.5),
                color = colgrad, kwargs_h...)
    dend1 = plot(clustering; xticks = false, ylim = (0, 1), kwargs_d1...)
    dend2 = plot(clustering; yticks = false, xrotation = 90, orientation = :horizontal,
                 xlim = (0, 1), kwargs_d2...)

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
    l = StatsPlots.grid(2, 2; heights = [0.2, 0.8, 0.2, 0.8], widths = [0.8, 0.2, 0.8, 0.2])
    plt = plot(dend1, plot(; ticks = nothing, border = :none, background_color = nothing),
               hmap, dend2; layout = l, kwargs...)

    return plt
end

function plot_clusters(assets::AbstractVector, returns::AbstractMatrix;
                       cor_opt::CorOpt = CorOpt(;), linkage = :single,
                       max_k = ceil(Int, sqrt(size(returns, 2))), branchorder = :optimal,
                       k = 0, dbht_method = :Unique, show_clusters = true,
                       theme_d = :Spectral, theme_h = :Spectral, theme_h_kwargs = (;),
                       kwargs_d1 = (;), kwargs_d2 = (;), kwargs_h = (;), kwargs_l = (;),
                       kwargs = (;))
    @smart_assert(linkage ∈ LinkageTypes)

    N = length(assets)
    cor_method = cor_opt.method
    corr, dist = cor_dist_mtx(returns, cor_opt)

    cors = (:Pearson, :Semi_Pearson, :Spearman, :Kendall, :Gerber1, :Gerber2, :custom)

    if linkage == :DBHT
        corr = cor_method ∈ cors ? 1 .- dist .^ 2 : corr
        missing, missing, missing, missing, missing, missing, clustering = DBHTs(dist, corr;
                                                                                 branchorder = branchorder,
                                                                                 method = dbht_method)
    else
        clustering = hclust(dist; linkage = linkage,
                            branchorder = branchorder == :default ? :r : branchorder)
    end

    tk = _two_diff_gap_stat(dist, clustering, max_k)

    k = iszero(k) ? tk : k

    clustering_idx = cutree(clustering; k = k)

    heights = clustering.heights
    sort_order = clustering.order
    ordered_corr = corr[sort_order, sort_order]
    ordered_assets = assets[sort_order]

    uidx = minimum(clustering_idx):maximum(clustering_idx)

    clusters = Vector{Vector{Int}}(undef, length(uidx))
    for i ∈ eachindex(clusters)
        clusters[i] = findall(clustering_idx .== i)
    end

    cors = (:Pearson, :Semi_Pearson, :Spearman, :Kendall, :Gerber1, :Gerber2, :custom)
    if cor_method ∈ cors
        clim = (-1, 1)
    else
        clim = (0, 1)
    end

    colours = palette(theme_d, k)
    colgrad = cgrad(theme_h; theme_h_kwargs...)

    #yticks=(1:nrows,rowlabels)
    hmap = plot(ordered_corr; st = :heatmap, yticks = (1:length(assets), ordered_assets),
                xticks = (1:length(assets), ordered_assets), xrotation = 90,
                colorbar = false, clim = clim, xlim = (0.5, N + 0.5), ylim = (0.5, N + 0.5),
                color = colgrad, kwargs_h...)
    dend1 = plot(clustering; xticks = false, ylim = (0, 1), kwargs_d1...)
    dend2 = plot(clustering; yticks = false, xrotation = 90, orientation = :horizontal,
                 xlim = (0, 1), kwargs_d2...)

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
    l = StatsPlots.grid(2, 2; heights = [0.2, 0.8, 0.2, 0.8], widths = [0.8, 0.2, 0.8, 0.2])
    plt = plot(dend1, plot(; ticks = nothing, border = :none, background_color = nothing),
               hmap, dend2; layout = l, kwargs...)

    return plt
end

function plot_dendrogram(portfolio; max_k = ceil(Int, sqrt(size(portfolio.dist, 1))),
                         linkage = :single, branchorder = :optimal, dbht_method = :Unique,
                         show_clusters = true, cluster = true, theme = :Spectral,
                         kwargs = (;))
    corr = portfolio.cor
    assets = portfolio.assets
    cor_method = portfolio.cor_method
    N = length(assets)

    if cluster
        clustering_idx, clustering, k = cluster_assets(portfolio; linkage = linkage,
                                                       max_k = max_k,
                                                       branchorder = branchorder,
                                                       k = portfolio.k,
                                                       dbht_method = dbht_method)
        sort_order = clustering.order
        heights = clustering.heights
    else
        clustering = portfolio.clusters
        sort_order = clustering.order
        heights = clustering.heights
        k = portfolio.k
        clustering_idx = cutree(clustering; k = k)
    end

    ordered_corr = corr[sort_order, sort_order]
    ordered_assets = assets[sort_order]

    uidx = minimum(clustering_idx):maximum(clustering_idx)

    clusters = Vector{Vector{Int}}(undef, length(uidx))
    for i ∈ eachindex(clusters)
        clusters[i] = findall(clustering_idx .== i)
    end

    colours = palette(theme, k)
    dend1 = plot(clustering; xticks = (sort_order, ordered_assets), xrotation = 90,
                 ylim = (0, 1))

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

    return plot(dend1; kwargs...)
end

function plot_network()
    # https://docs.juliaplots.org/latest/GraphRecipes/introduction/
    # https://juliagraphs.org/GraphPlot.jl/index.html#usage
end

export plot_returns, plot_bar, plot_risk_contribution, plot_frontier_area, plot_drawdown,
       plot_hist, plot_range, plot_frontier, plot_clusters, plot_dendrogram, plot_network
