# module PortfolioOptimiserMakieExt
# using PortfolioOptimiser, Makie, SmartAsserts, Statistics, MultivariateStats, Distributions,
#       Clustering, Graphs, SimpleWeightedGraphs, LinearAlgebra

# import PortfolioOptimiser: AbstractPortfolio, RiskMeasure, RetType

# """
# ```
# plot_returns(timestamps, assets, returns, weights; per_asset = false, kwargs...)
# ```
# """
# function PortfolioOptimiser.plot_returns(timestamps, assets, returns, weights;
#                                          per_asset = false)
#     f = Figure()
#     if per_asset
#         ax = Axis(f[1, 1]; xlabel = "Date", ylabel = "Asset Cummulative Returns")
#         ret = returns .* transpose(weights)
#         ret = vcat(zeros(1, length(weights)), ret)
#         ret .+= 1
#         ret = cumprod(ret; dims = 1)
#         ret = ret[2:end, :]
#         for (i, asset) ∈ enumerate(assets)
#             lines!(ax, timestamps, view(ret, :, i); label = asset)
#         end
#     else
#         ax = Axis(f[1, 1]; xlabel = "Date", ylabel = "Portfolio Cummulative Returns")
#         ret = returns * weights
#         pushfirst!(ret, 0)
#         ret .+= 1
#         ret = cumprod(ret)
#         popfirst!(ret)
#         lines!(ax, timestamps, ret; label = "Portfolio")
#     end
#     axislegend(; position = :lt, merge = true)
#     return f
# end
# function PortfolioOptimiser.plot_returns(port::AbstractPortfolio,
#                                          type = isa(port, HCPortfolio) ? :HRP : :Trad;
#                                          per_asset = false)
#     return PortfolioOptimiser.plot_returns(port.timestamps, port.assets, port.returns,
#                                            port.optimal[type].weights;
#                                            per_asset = per_asset)
# end

# function PortfolioOptimiser.plot_bar(assets, weights)
#     f = Figure()
#     ax = Axis(f[1, 1]; xticks = (1:length(assets), assets),
#               ylabel = "Portfolio Composition, %", xticklabelrotation = pi / 2)
#     barplot!(ax, weights * 100)
#     return f
# end
# function PortfolioOptimiser.plot_bar(port::AbstractPortfolio,
#                                      type = isa(port, HCPortfolio) ? :HRP : :Trad;
#                                      kwargs...)
#     return PortfolioOptimiser.plot_bar(port.assets, port.optimal[type].weights, kwargs...)
# end

# function PortfolioOptimiser.plot_risk_contribution(assets::AbstractVector,
#                                                    w::AbstractVector, X::AbstractMatrix;
#                                                    rm::RiskMeasure = SD(),
#                                                    V::AbstractMatrix = Matrix{Float64}(undef,
#                                                                                        0,
#                                                                                        0),
#                                                    SV::AbstractMatrix = Matrix{Float64}(undef,
#                                                                                         0,
#                                                                                         0),
#                                                    percentage::Bool = false,
#                                                    erc_line::Bool = true, t_factor = 252,
#                                                    delta::Real = 1e-6,
#                                                    marginal::Bool = false, kwargs...)
#     rc = risk_contribution(rm, w; X = X, V = V, SV = SV, delta = delta, marginal = marginal,
#                            kwargs...)

#     DDs = (DaR, MDD, ADD, CDaR, EDaR, RLDaR, UCI, DaR_r, MDD_r, ADD_r, CDaR_r, EDaR_r,
#            RLDaR_r, UCI_r)

#     if !any(typeof(rm) .<: DDs)
#         rc *= sqrt(t_factor)
#     end
#     ylabel = "Risk Contribution"
#     if percentage
#         rc .= 100 * rc / sum(rc)
#         ylabel *= ", %"
#     end

#     rmstr = string(rm)
#     rmstr = rmstr[1:(findfirst('{', rmstr) - 1)]
#     title = "Risk Contribution - $rmstr"
#     if any(typeof(rm) .<:
#            (CVaR, TG, EVaR, RLVaR, CVaRRG, TGRG, CDaR, EDaR, RLDaR, CDaR_r, EDaR_r, RLDaR_r))
#         title *= " α = $(round(rm.alpha*100, digits=2))%"
#     end
#     if any(typeof(rm) .<: (CVaRRG, TGRG))
#         title *= ", β = $(round(rm.beta*100, digits=2))%"
#     end
#     if any(typeof(rm) .<: (RLVaR, RLDaR, RLDaR_r))
#         title *= ", κ = $(round(rm.kappa, digits=2))"
#     end

#     f = Figure()
#     ax = Axis(f[1, 1]; xticks = (1:length(assets), assets), title = title, ylabel = ylabel,
#               xticklabelrotation = pi / 2)
#     barplot!(ax, rc)

#     if erc_line
#         if percentage
#             erc = 100 / length(rc)
#         else
#             erc = calc_risk(rm, w; X = X, V = V, SV = SV)

#             erc /= length(rc)

#             if !any(typeof(rm) .<: DDs)
#                 erc *= sqrt(t_factor)
#             end
#         end

#         hlines!(ax, erc)
#     end

#     return f
# end
# function PortfolioOptimiser.plot_risk_contribution(port::AbstractPortfolio,
#                                                    type = if isa(port, HCPortfolio)
#                                                        :HRP
#                                                    else
#                                                        :Trad
#                                                    end; X = port.returns,
#                                                    rm::RiskMeasure = SD(),
#                                                    percentage::Bool = false,
#                                                    erc_line::Bool = true, t_factor = 252,
#                                                    delta::Real = 1e-6,
#                                                    marginal::Bool = false, kwargs...)
#     solver_flag, sigma_flag = set_rm_properties(rm, port.solvers, port.cov)
#     fig = PortfolioOptimiser.plot_risk_contribution(port.assets, port.optimal[type].weights,
#                                                     X; rm = rm, V = port.V, SV = port.SV,
#                                                     percentage = percentage,
#                                                     erc_line = erc_line,
#                                                     t_factor = t_factor, delta = delta,
#                                                     marginal = marginal, kwargs...)
#     unset_set_rm_properties(rm, solver_flag, sigma_flag)

#     return fig
# end

# function PortfolioOptimiser.plot_frontier(frontier;
#                                           X::AbstractMatrix = Matrix{Float64}(undef, 0, 0),
#                                           mu::AbstractVector = Vector{Float64}(undef, 0),
#                                           rf::Real = 0.0, rm::RiskMeasure = SD(),
#                                           kelly::RetType = NoKelly(), t_factor = 252,
#                                           palette = :Spectral)
#     risks = copy(frontier[:risk])
#     weights = Matrix(frontier[:weights][!, 2:end])

#     if isa(kelly, NoKelly)
#         ylabel = "Expected Arithmetic Return"
#         rets = transpose(weights) * mu
#     else
#         ylabel = "Expected Kelly Return"
#         rets = 1 / size(X, 1) * vec(sum(log.(1 .+ X * weights); dims = 1))
#     end

#     rets .*= t_factor

#     if !any(typeof(rm) .<: (MDD, ADD, CDaR, EDaR, RLDaR, UCI))
#         risks .*= sqrt(t_factor)
#     end

#     ratios = (rets .- rf) ./ risks
#     N = length(ratios)

#     title = "$(get_rm_string(rm))"
#     if any(typeof(rm) .<: (CVaR, TG, EVaR, RLVaR, CVaRRG, TGRG, CDaR, EDaR, RLDaR))
#         title *= " α = $(round(rm.alpha*100, digits=2))%"
#     end
#     if any(typeof(rm) .<: (CVaRRG, TGRG))
#         title *= ", β = $(round(rm.beta*100, digits=2))%"
#     end
#     if any(typeof(rm) .<: (RLVaR, RLDaR))
#         title *= ", κ = $(round(rm.kappa, digits=2))"
#     end

#     f = Figure()
#     ax = Axis(f[1, 1]; title = title, ylabel = ylabel, xlabel = "Expected Risk")
#     Colorbar(f[1, 2]; label = "Risk Adjusted Return Ratio", limits = extrema(ratios),
#              colormap = palette)

#     if frontier[:sharpe]
#         scatter!(ax, risks[1:(end - 1)], rets[1:(end - 1)]; color = ratios[1:(N - 1)],
#                  colormap = palette, marker = :circle, markersize = 15)
#         scatter!(ax, risks[end], rets[end]; color = cgrad(palette)[ratios[N]],
#                  marker = :star5, markersize = 15, label = "Max Risk Adjusted Return Ratio")
#     else
#         scatter(ax, risks[1:end], rets[1:end]; color = ratios, colormap = palette)
#     end
#     axislegend(ax; position = :rb)

#     return f
# end
# function PortfolioOptimiser.plot_frontier(port::AbstractPortfolio, key = nothing;
#                                           X::AbstractMatrix = port.returns,
#                                           mu::AbstractVector = port.mu,
#                                           rm::RiskMeasure = SD(), rf::Real = 0.0,
#                                           kelly::RetType = NoKelly(), t_factor = 252,
#                                           palette = :Spectral)
#     if isnothing(key)
#         key = get_rm_string(rm)
#     end
#     return PortfolioOptimiser.plot_frontier(port.frontier[key]; X = X, mu = mu, rf = rf,
#                                             rm = rm, kelly = kelly, t_factor = t_factor,
#                                             palette = palette)
# end

# function PortfolioOptimiser.plot_frontier_area(frontier; rm::RiskMeasure = SD(),
#                                                t_factor = 252, palette = :Spectral)
#     risks = copy(frontier[:risk])
#     assets = reshape(frontier[:weights][!, "tickers"], 1, :)
#     weights = Matrix(frontier[:weights][!, 2:end])

#     if !any(typeof(rm) .<: (MDD, ADD, CDaR, EDaR, RLDaR, UCI))
#         risks .*= sqrt(t_factor)
#     end

#     sharpe = nothing
#     if frontier[:sharpe]
#         sharpe = risks[end]
#     end

#     idx = sortperm(risks)
#     risks = risks[idx]
#     weights = weights[:, idx]

#     title = "$(get_rm_string(rm))"
#     if any(typeof(rm) .<: (CVaR, TG, EVaR, RLVaR, CVaRRG, TGRG, CDaR, EDaR, RLDaR))
#         title *= " α = $(round(rm.alpha*100, digits=2))%"
#     end
#     if any(typeof(rm) .<: (CVaRRG, TGRG))
#         title *= ", β = $(round(rm.beta*100, digits=2))%"
#     end
#     if any(typeof(rm) .<: (RLVaR, RLDaR))
#         title *= ", κ = $(round(rm.kappa, digits=2))"
#     end

#     f = Figure()
#     ax = Axis(f[1, 1]; title = title, ylabel = "Composition", xlabel = "Expected Risk",
#               limits = (minimum(risks), maximum(risks), 0, 1))

#     N = length(risks)
#     weights = cumsum(weights; dims = 1)
#     colors = cgrad(palette, size(weights, 1); categorical = true, scale = true)
#     for i ∈ axes(weights, 1)
#         if i == 1
#             band!(ax, risks, range(0, 0, N), weights[i, :]; label = assets[i],
#                   color = colors[i])
#         else
#             band!(ax, risks, weights[i - 1, :], weights[i, :]; label = assets[i],
#                   color = colors[i])
#         end
#     end
#     if !isnothing(sharpe)
#         vlines!(ax, sharpe; ymin = 0.0, ymax = 1.0, color = :black, linewidth = 3)
#         text!(ax, (sharpe * 1.1, 0.5); text = "Max Risk\nAdjusted\nReturn Ratio",
#               align = (:left, :baseline))
#     end
#     axislegend(ax; position = :rc)
#     return f
# end
# function PortfolioOptimiser.plot_frontier_area(port::AbstractPortfolio, key = nothing;
#                                                rm = SD(), t_factor = 252,
#                                                palette = :Spectral)
#     if isnothing(key)
#         key = get_rm_string(rm)
#     end
#     return PortfolioOptimiser.plot_frontier_area(port.frontier[key]; t_factor = t_factor,
#                                                  palette = palette)
# end

# function PortfolioOptimiser.plot_drawdown(timestamps::AbstractVector, w::AbstractVector,
#                                           X::AbstractMatrix; alpha::Real = 0.05,
#                                           kappa::Real = 0.3,
#                                           solvers::Union{<:AbstractDict, Nothing} = nothing,
#                                           palette = :Spectral)
#     ret = X * w

#     prices = copy(ret)
#     pushfirst!(prices, 0)
#     prices .+= 1
#     prices = cumprod(prices)
#     popfirst!(prices)
#     prices2 = cumsum(copy(ret)) .+ 1

#     dd = similar(prices2)
#     peak = -Inf
#     for i ∈ eachindex(prices2)
#         if prices2[i] > peak
#             peak = prices2[i]
#         end
#         dd[i] = prices2[i] - peak
#     end

#     dd .*= 100

#     risks = [-PortfolioOptimiser._ADD(ret), -PortfolioOptimiser._UCI(ret),
#              -PortfolioOptimiser._DaR(ret, alpha), -PortfolioOptimiser._CDaR(ret, alpha),
#              -PortfolioOptimiser._EDaR(ret, solvers, alpha),
#              -PortfolioOptimiser._RLDaR(ret, solvers, alpha, kappa),
#              -PortfolioOptimiser._MDD(ret)] * 100

#     conf = round((1 - alpha) * 100; digits = 2)

#     risk_labels = ("Average Drawdown: $(round(risks[1], digits = 2))%",
#                    "Ulcer Index: $(round(risks[2], digits = 2))%",
#                    "$(conf)% Confidence DaR: $(round(risks[3], digits = 2))%",
#                    "$(conf)% Confidence CDaR: $(round(risks[4], digits = 2))%",
#                    "$(conf)% Confidence EDaR: $(round(risks[5], digits = 2))%",
#                    "$(conf)% Confidence RLDaR ($(round(kappa, digits=2))): $(round(risks[6], digits = 2))%",
#                    "Maximum Drawdown: $(round(risks[7], digits = 2))%")

#     f = Figure()
#     ax1 = Axis(f[1, 1]; ylabel = "Cummulative Returns")
#     ax2 = Axis(f[2, 1]; ylabel = "Percentage Drawdowns", xlabel = "Date")

#     colors = cgrad(palette, length(risk_labels) + 1; categorical = true, scale = true)

#     lines!(ax1, timestamps, prices; label = "Cummulative Returns", color = colors[1])
#     lines!(ax2, timestamps, dd; label = "Uncompounded Cummulative Drawdown",
#            color = colors[1])
#     for (i, (risk, label)) ∈ enumerate(zip(risks, risk_labels))
#         hlines!(ax2, risk; xmin = 0.0, xmax = 1.0, label = label, color = colors[i + 1])
#     end
#     Legend(f[2, 2], ax2; labelsize = 10)

#     return f
# end
# function PortfolioOptimiser.plot_drawdown(port::AbstractPortfolio,
#                                           type = isa(port, HCPortfolio) ? :HRP : :Trad;
#                                           X = port.returns, alpha::Real = 0.05,
#                                           kappa::Real = 0.3, palette = :Spectral)
#     return PortfolioOptimiser.plot_drawdown(port.timestamps, port.optimal[type].weights, X;
#                                             alpha = alpha, kappa = kappa, port.solvers,
#                                             palette = palette)
# end

# function PortfolioOptimiser.plot_hist(w::AbstractVector, X::AbstractMatrix;
#                                       alpha_i::Real = 0.0001, alpha::Real = 0.05,
#                                       a_sim::Int = 100, kappa::Real = 0.3,
#                                       solvers::Union{<:AbstractDict, Nothing} = nothing,
#                                       points::Integer = ceil(Int, 4 * sqrt(size(X, 1))),
#                                       palette = :Spectral)
#     ret = X * w * 100
#     mu = mean(ret)
#     sigma = std(ret)

#     x = range(minimum(ret); stop = maximum(ret), length = points)

#     mad = PortfolioOptimiser._MAD(ret)
#     gmd = PortfolioOptimiser._GMD(ret)
#     risks = (mu, mu - sigma, mu - mad, mu - gmd, -PortfolioOptimiser._VaR(ret, alpha),
#              -PortfolioOptimiser._CVaR(ret, alpha),
#              -PortfolioOptimiser._TG(ret; alpha_i = alpha_i, alpha = alpha, a_sim = a_sim),
#              -PortfolioOptimiser._EVaR(ret, solvers, alpha),
#              -PortfolioOptimiser._RLVaR(x, solvers, alpha, kappa),
#              -PortfolioOptimiser._WR(ret))

#     conf = round((1 - alpha) * 100; digits = 2)

#     risk_labels = ("Mean: $(round(risks[1], digits=2))%",
#                    "Mean - Std. Dev. ($(round(sigma, digits=2))%): $(round(risks[2], digits=2))%",
#                    "Mean - MAD ($(round(mad,digits=2))%): $(round(risks[3], digits=2))%",
#                    "Mean - GMD ($(round(gmd,digits=2))%): $(round(risks[4], digits=2))%",
#                    "$(conf)% Confidence VaR: $(round(risks[5], digits=2))%",
#                    "$(conf)% Confidence CVaR: $(round(risks[6], digits=2))%",
#                    "$(conf)% Confidence Tail Gini: $(round(risks[7], digits=2))%",
#                    "$(conf)% Confidence EVaR: $(round(risks[8], digits=2))%",
#                    "$(conf)% Confidence RLVaR ($(round(kappa, digits=2))): $(round(risks[9], digits=2))%",
#                    "Worst Realisation: $(round(risks[10], digits=2))%")

#     D = fit(Normal, ret)

#     colors = cgrad(palette, length(risk_labels) + 2; categorical = true, scale = true)

#     f = Figure()
#     ax = Axis(f[1, 1]; ylabel = "Probability Density")
#     hist!(ax, ret; normalization = :pdf, color = colors[1])
#     for (i, (label, risk)) ∈ enumerate(zip(risk_labels, risks))
#         vlines!(ax, risk; ymin = 0.0, ymax = 1.0, color = colors[i + 1], linewidth = 3,
#                 label = label)
#     end
#     lines!(ax, x, pdf.(D, x);
#            label = "Normal: μ = $(round(mean(D), digits=2))%, σ = $(round(std(D), digits=2))%",
#            color = colors[end], linewidth = 3)
#     Legend(f[1, 2], ax; labelsize = 10)

#     return f
# end

# function PortfolioOptimiser.plot_hist(port::AbstractPortfolio,
#                                       type = isa(port, HCPortfolio) ? :HRP : :Trad;
#                                       X = port.returns, alpha_i::Real = 0.0001,
#                                       alpha::Real = 0.05, a_sim::Int = 100,
#                                       kappa::Real = 0.3,
#                                       points::Integer = ceil(Int,
#                                                              4 *
#                                                              sqrt(size(port.returns, 1))),
#                                       palette = :Spectral)
#     return plot_hist(port.optimal[type].weights, X; alpha_i = alpha_i, alpha = alpha,
#                      a_sim = a_sim, kappa = kappa, solvers = port.solvers, points = points,
#                      palette = palette)
# end

# function PortfolioOptimiser.plot_range(w::AbstractVector, X::AbstractMatrix;
#                                        alpha_i::Real = 0.0001, alpha::Real = 0.05,
#                                        a_sim::Int = 100, beta_i::Real = alpha_i,
#                                        beta::Real = alpha, b_sim::Integer = a_sim,
#                                        palette = :Spectral)
#     ret = X * w * 100

#     risks = (PortfolioOptimiser._RG(ret),
#              PortfolioOptimiser._CVaRRG(ret; alpha = alpha, beta = beta),
#              PortfolioOptimiser._TGRG(ret; alpha_i = alpha_i, alpha = alpha, a_sim = a_sim,
#                                      beta_i = beta_i, beta = beta, b_sim = b_sim))

#     lo_conf = 1 - alpha
#     hi_conf = 1 - beta
#     risk_labels = ("Range: $(round(risks[1], digits=2))%",
#                    "Tail Gini Range ($(round(lo_conf,digits=2)), $(round(hi_conf,digits=2))): $(round(risks[2], digits=2))%",
#                    "CVaR Range ($(round(lo_conf,digits=2)), $(round(hi_conf,digits=2))): $(round(risks[3], digits=2))%")

#     D = fit(Normal, ret)
#     y = pdf(D, mean(D))
#     ys = (y / 4, y / 2, y * 3 / 4)

#     colors = cgrad(palette, length(risk_labels) + 1; categorical = true, scale = true)

#     f = Figure()
#     ax = Axis(f[1, 1]; ylabel = "Probability Density")
#     hist!(ax, ret; normalization = :pdf, color = colors[1])

#     bounds = [minimum(ret) -PortfolioOptimiser._TG(ret; alpha_i = alpha_i, alpha = alpha, a_sim = a_sim) -PortfolioOptimiser._CVaR(ret, alpha);
#               maximum(ret) PortfolioOptimiser._TG(-ret; alpha_i = alpha_i, alpha = alpha, a_sim = a_sim) PortfolioOptimiser._CVaR(-ret, alpha)]

#     for i ∈ eachindex(risks)
#         lines!(ax, [bounds[1, i], bounds[1, i], bounds[2, i], bounds[2, i]],
#                [0, ys[i], ys[i], 0]; label = risk_labels[i], color = colors[i + 1],
#                linewidth = 3)
#     end
#     Legend(f[1, 2], ax; labelsize = 10)

#     return f
# end

# function PortfolioOptimiser.plot_range(port::AbstractPortfolio,
#                                        type = isa(port, HCPortfolio) ? :HRP : :Trad;
#                                        X = port.returns, alpha_i::Real = 0.0001,
#                                        alpha::Real = 0.05, a_sim::Int = 100,
#                                        beta_i::Real = alpha_i, beta::Real = alpha,
#                                        b_sim::Integer = a_sim, palette = :Spectral)
#     return plot_range(port.optimal[type].weights, X; alpha_i = alpha_i, alpha = alpha,
#                       a_sim = a_sim, beta_i = beta_i, beta = beta, b_sim = b_sim,
#                       palette = palette)
# end

# function PortfolioOptimiser.plot_clusters(assets::AbstractVector, rho::AbstractMatrix,
#                                           idx::AbstractVector{<:Integer},
#                                           clustering::Hclust, k::Integer,
#                                           limits::Tuple{<:Real, <:Real} = (-1, 1),
#                                           palette = :Spectral, show_clusters::Bool = true)
#     N = length(assets)
#     order = clustering.order
#     heights = clustering.heights

#     rho = rho[order, order]
#     assets = assets[order]

#     uidx = minimum(idx):maximum(idx)
#     clusters = Vector{Vector{Int}}(undef, length(uidx))
#     for i ∈ eachindex(clusters)
#         clusters[i] = findall(idx .== i)
#     end

#     colors = cgrad(palette, k; categorical = true)

#     f = Figure()
#     ax = Axis(f[1, 1]; xticks = (1:N, assets), yticks = (1:N, assets),
#               xticklabelrotation = pi / 2, yreversed = true)
#     Colorbar(f[1, 2]; label = "Correlation", limits = limits, colormap = palette)
#     heatmap!(ax, 1:N, 1:N, rho)

#     nodes = -clustering.merges
#     if show_clusters
#         for (i, cluster) ∈ enumerate(clusters)
#             a = [findfirst(x -> x == c, order) for c ∈ cluster]
#             a = a[.!isnothing.(a)]
#             xmin = minimum(a)
#             xmax = xmin + length(cluster)

#             i1 = [findfirst(x -> x == c, nodes[:, 1]) for c ∈ cluster]
#             i1 = i1[.!isnothing.(i1)]
#             i2 = [findfirst(x -> x == c, nodes[:, 2]) for c ∈ cluster]
#             i2 = i2[.!isnothing.(i2)]
#             i3 = unique([i1; i2])
#             h = min(maximum(heights[i3]) * 1.1, 1)

#             lines!(ax,
#                    [xmin - 0.5, xmax - 0.5, xmax - 0.5, xmax - 0.5, xmax - 0.5, xmin - 0.5,
#                     xmin - 0.5, xmin - 0.5],
#                    [xmin - 0.5, xmin - 0.5, xmin - 0.5, xmax - 0.5, xmax - 0.5, xmax - 0.5,
#                     xmax - 0.5, xmin - 0.5]; color = :black, linewidth = 3)

#             # plot!(dend1,
#             #       [xmin - 0.25, xmax - 0.75, xmax - 0.75, xmax - 0.75, xmax - 0.75,
#             #        xmin - 0.25, xmin - 0.25, xmin - 0.25], [0, 0, 0, h, h, h, h, 0];
#             #       color = nothing, legend = false,
#             #       fill = (0, 0.5, colours[(i - 1) % k + 1]))

#             # plot!(dend2, [0, 0, 0, h, h, h, h, 0],
#             #       [xmin - 0.25, xmax - 0.75, xmax - 0.75, xmax - 0.75, xmax - 0.75,
#             #        xmin - 0.25, xmin - 0.25, xmin - 0.25]; color = nothing, legend = false,
#             #       fill = (0, 0.5, colours[(i - 1) % k + 1]))
#         end
#     end

#     return f
# end

# end
