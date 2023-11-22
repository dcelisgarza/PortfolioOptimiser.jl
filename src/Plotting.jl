function plot_returns(timestamps, assets, returns, weights; per_asset = false, kwargs...)
    if per_asset
        ret = returns .* transpose(weights)
        ret = vcat(zeros(1, length(weights)), ret)
        ret .+= 1
        ret = cumprod(ret, dims = 1)
        ret = ret[2:end, :]
        !haskey(kwargs, :label) && (kwargs = (kwargs..., label = reshape(assets, 1, :)))
    else
        ret = returns * weights
        pushfirst!(ret, 0)
        ret .+= 1
        ret = cumprod(ret)
        popfirst!(ret)
        !haskey(kwargs, :legend) && (kwargs = (kwargs..., legend = false))
    end
    !haskey(kwargs, :ylabel) && (kwargs = (kwargs..., ylabel = "Cummulative Return"))
    !haskey(kwargs, :xlabel) && (kwargs = (kwargs..., xlabel = "Date"))

    plot(timestamps, ret; kwargs...)
end
function plot_returns(
    portfolio,
    type = isa(portfolio, HCPortfolio) ? :HRP : :Trad;
    per_asset = false,
    kwargs...,
)
    return plot_returns(
        portfolio.timestamps,
        portfolio.assets,
        portfolio.returns,
        portfolio.optimal[type].weights;
        per_asset = per_asset,
        kwargs...,
    )
end

function plot_bar(assets, data; kwargs...)
    !haskey(kwargs, :ylabel) && (kwargs = (kwargs..., ylabel = "Percentage Composition"))
    !haskey(kwargs, :xlabel) && (kwargs = (kwargs..., xlabel = "Assets"))
    !haskey(kwargs, :xticks) && (
        kwargs = (
            kwargs...,
            xticks = (range(0.5, step = 1, length = length(assets)), assets),
        )
    )
    !haskey(kwargs, :xrotation) && (kwargs = (kwargs..., xrotation = 60))
    !haskey(kwargs, :legend) && (kwargs = (kwargs..., legend = false))

    return bar(assets, data * 100; kwargs...)
end
function plot_bar(
    portfolio::AbstractPortfolio,
    type = isa(portfolio, HCPortfolio) ? :HRP : :Trad,
    kwargs...,
)
    return plot_bar(portfolio.assets, portfolio.optimal[type].weights, kwargs...)
end

function plot_risk_contribution(
    # RC args
    assets::AbstractVector,
    w::AbstractVector,
    returns::AbstractMatrix;
    rm::Symbol = :SD,
    rf::Real = 0.0,
    sigma::AbstractMatrix,
    alpha_i::Real = 0.0001,
    alpha::Real = 0.05,
    a_sim::Int = 100,
    beta_i::Union{<:Real, Nothing} = nothing,
    beta::Union{<:Real, Nothing} = nothing,
    b_sim::Union{<:Real, Nothing} = nothing,
    di::Real = 1e-6,
    kappa::Real = 0.3,
    owa_w::AbstractVector = Vector{Float64}(undef, 0),
    solvers::Union{<:AbstractDict, Nothing} = nothing,
    # Plot args
    percentage::Bool = false,
    erc_line::Bool = true,
    t_factor = 252,
    kwargs_bar = (;),
    kwargs_line = (;),
)
    rc = risk_contribution(
        w,
        returns;
        rm = rm,
        rf = rf,
        sigma = sigma,
        alpha_i = alpha_i,
        alpha = alpha,
        a_sim = a_sim,
        beta_i = beta_i,
        beta = beta,
        b_sim = b_sim,
        di = di,
        kappa = kappa,
        owa_w = owa_w,
        solvers = solvers,
    )

    DDs = (
        :DaR,
        :MDD,
        :ADD,
        :CDaR,
        :EDaR,
        :RDaR,
        :UCI,
        :DaR_r,
        :MDD_r,
        :ADD_r,
        :CDaR_r,
        :EDaR_r,
        :RDaR_r,
        :UCI_r,
    )

    rm ∉ DDs && (rc *= sqrt(t_factor))
    msg = ""
    if percentage
        rc /= sum(rc)
        msg = "Percentage "
    end

    !haskey(kwargs_bar, :ylabel) &&
        (kwargs_bar = (kwargs_bar..., ylabel = msg * "Risk Contribution"))
    !haskey(kwargs_bar, :xlabel) && (kwargs_bar = (kwargs_bar..., xlabel = "Assets"))
    !haskey(kwargs_bar, :xticks) && (
        kwargs_bar = (
            kwargs_bar...,
            xticks = (range(0.5, step = 1, length = length(assets)), assets),
        )
    )
    !haskey(kwargs_bar, :xrotation) && (kwargs_bar = (kwargs_bar..., xrotation = 60))
    !haskey(kwargs_bar, :legend) && (kwargs_bar = (kwargs_bar..., legend = false))

    plt = bar(assets, rc; kwargs_bar...)

    if erc_line
        if percentage
            erc = 1 / length(rc)
        else
            erc = calc_risk(
                w,
                returns;
                rm = rm,
                rf = rf,
                sigma = sigma,
                alpha_i = alpha_i,
                alpha = alpha,
                a_sim = a_sim,
                beta_i = beta_i,
                beta = beta,
                b_sim = b_sim,
                kappa = kappa,
                owa_w = owa_w,
                solvers = solvers,
            )

            erc /= length(rc)
            rm ∉ DDs && (erc *= sqrt(t_factor))
        end

        hline!([erc]; kwargs_line...)
    end

    return plt
end
function plot_risk_contribution(
    portfolio;
    di::Real = 1e-6,
    type::Symbol = isa(portfolio, Portfolio) ? :Trad : :HRP,
    rm::Symbol = :SD,
    rf::Real = 0.0,
    owa_w = isa(portfolio, Portfolio) ? portfolio.owa_w : Vector{Float64}(undef, 0),
    percentage::Bool = false,
    erc_line::Bool = true,
    t_factor = 252,
    kwargs_bar = (;),
    kwargs_line = (;),
)
    plot_risk_contribution(
        # RC args
        portfolio.assets,
        portfolio.optimal[type].weights,
        portfolio.returns;
        rm = rm,
        rf = rf,
        sigma = portfolio.cov,
        alpha_i = portfolio.alpha_i,
        alpha = portfolio.alpha,
        a_sim = portfolio.a_sim,
        beta_i = portfolio.beta_i,
        beta = portfolio.beta,
        b_sim = portfolio.b_sim,
        di = di,
        kappa = portfolio.kappa,
        owa_w = owa_w,
        solvers = portfolio.solvers,
        # Plot args
        percentage = percentage,
        erc_line = erc_line,
        t_factor = t_factor,
        kwargs_bar = kwargs_bar,
        kwargs_line = kwargs_line,
    )
end

function plot_frontier(
    frontier;
    alpha_i::Real = 0.0001,
    alpha::Real = 0.05,
    a_sim::Int = 100,
    beta_i::Union{<:Real, Nothing} = nothing,
    beta::Union{<:Real, Nothing} = nothing,
    b_sim::Union{<:Real, Nothing} = nothing,
    kappa::Real = 0.3,
    owa_w::AbstractVector = Vector{Float64}(undef, 0),
    sigma::AbstractMatrix = Matrix{Float64}(undef, 0, 0),
    returns::AbstractMatrix = Matrix{Float64}(undef, 0, 0),
    t_factor = 252,
    kelly::Bool = false,
    mu::AbstractVector = Vector{Float64}(undef, 0),
    rf::Real = 0.0,
    rm::Symbol = :SD,
    f_kwargs = (;),
    e_kwargs = (;),
    w::AbstractVector = Vector{Float64}(undef, 0),
)
    @assert(rm ∈ RiskMeasures, "rm = $rm, must be one of $RiskMeasures")

    if haskey(f_kwargs, :ylabel)
        f_kwargs = if kelly
            (f_kwargs..., ylabel = "Expected Arithmetic Return")
        else
            (f_kwargs..., ylabel = "Expected Kelly Return")
        end
    end
    !haskey(f_kwargs, :xlabel) && (f_kwargs = (f_kwargs..., xlabel = "Expected Risk"))

    risks = copy(frontier[rm][:risk])
    weights = Matrix(frontier[rm][:weights][!, 2:end])

    rets = if kelly
        1 / size(returns, 1) * vec(sum(log.(1 .+ returns * weights), dims = 1))
    else
        transpose(weights) * mu
    end
    rets .*= t_factor

    if rm ∉ (:MDD, :ADD, :CDaR, :EDaR, :RLDaR, :UCI)
        risks .*= sqrt(t_factor)
    end

    ratios = (rets .- rf) ./ risks

    plt = if frontier[rm][:sharpe]
        scatter(
            risks[1:(end - 1)],
            rets[1:(end - 1)],
            c = :viridis,
            colorbar = true,
            zcolor = ratios[1:(end - 1)],
            label = "",
        )
        scatter!(
            [risks[end]],
            [rets[end]],
            label = "Max Risk Adjusted Return Ratio",
            markershape = :star,
            color = :red,
        )
    else
        scatter(risks[1:end], rets[1:end], c = :viridis, colorbar = true, zcolor = ratios)
    end
    return plt
end

function plot_frontier(
    portfolio::AbstractPortfolio;
    rm::Symbol = :SD,
    rf::Real = 0.0,
    kelly::Bool = false,
    t_factor = 252,
    f_kwargs = (;),
    e_kwargs = (;),
    w::AbstractVector = Vector{Float64}(undef, 0),
)
    plot_frontier(
        portfolio.frontier;
        alpha_i = portfolio.alpha_i,
        alpha = portfolio.alpha,
        a_sim = portfolio.a_sim,
        beta_i = portfolio.beta_i,
        beta = portfolio.beta,
        b_sim = portfolio.b_sim,
        kappa = portfolio.kappa,
        owa_w = portfolio.owa_w,
        sigma = portfolio.cov,
        mu = portfolio.mu,
        returns = portfolio.returns,
        t_factor = t_factor,
        kelly = kelly,
        rf = rf,
        rm = rm,
        f_kwargs = f_kwargs,
        e_kwargs = e_kwargs,
        w = w,
    )
end

function plot_frontier_area(frontier, rm = :SD; t_factor = 252, kwargs...)
    risks = copy(frontier[rm][:risk])
    assets = reshape(frontier[rm][:weights][!, "tickers"], 1, :)
    weights = transpose(Matrix(frontier[rm][:weights][!, 2:end]))

    !haskey(kwargs, :ylabel) && (kwargs = (kwargs..., ylabel = "Percentage Composition"))
    !haskey(kwargs, :xlabel) && (kwargs = (kwargs..., xlabel = "Risk"))
    !haskey(kwargs, :label) && (kwargs = (kwargs..., label = assets))
    if frontier[rm][:sharpe]
        risks = risks[1:(end - 1)]
        weights = weights[:, 1:(end - 1)]
    end
    if rm ∉ (:MDD, :ADD, :CDaR, :EDaR, :RLDaR, :UCI)
        risks .*= sqrt(t_factor)
    end
    !haskey(kwargs, :xtick) && (kwargs = (kwargs..., xtick = round.(risks, digits = 3)))
    !haskey(kwargs, :xrotation) && (kwargs = (kwargs..., xrotation = 60))

    return areaplot(risks, weights; kwargs...)
end

function plot_drawdown(
    timestamps::AbstractVector,
    w::AbstractVector,
    returns::AbstractMatrix;
    alpha::Real = 0.05,
    kappa::Real = 0.3,
    solvers::Union{<:AbstractDict, Nothing} = nothing,
    kwargs_ret = (;),
    kwargs_dd = (;),
    kwargs_risks = (;),
    kwargs_all = (;),
)
    ret = returns * w

    prices = copy(ret)
    pushfirst!(prices, 0)
    prices .+= 1
    prices = cumprod(prices)
    popfirst!(prices)
    prices2 = cumsum(copy(ret)) .+ 1

    dd = similar(prices2)
    peak = -Inf
    for i in eachindex(prices2)
        prices2[i] > peak && (peak = prices2[i])
        dd[i] = prices2[i] - peak
    end

    dd .*= 100

    risks =
        [
            -ADD_abs(ret),
            -UCI_abs(ret),
            -DaR_abs(ret, alpha),
            -CDaR_abs(ret, alpha),
            -EDaR_abs(ret, solvers, alpha),
            -RDaR_abs(ret, solvers, alpha, kappa),
            -MDD_abs(ret),
        ] * 100

    conf = round((1 - alpha) * 100, digits = 2)

    risk_labels = (
        "Average Drawdown: $(round(risks[1], digits = 2))%",
        "Ulcer Index: $(round(risks[2], digits = 2))%",
        "$(conf)% Confidence DaR: $(round(risks[3], digits = 2))%",
        "$(conf)% Confidence CDaR: $(round(risks[4], digits = 2))%",
        "$(conf)% Confidence EDaR: $(round(risks[5], digits = 2))%",
        "$(conf)% Confidence RDaR ($kappa): $(round(risks[6], digits = 2))%",
        "Maximum Drawdown: $(round(risks[7], digits = 2))%",
    )

    !haskey(kwargs_dd, :ylabel) &&
        (kwargs_dd = (kwargs_dd..., ylabel = "Percentage Drawdown"))
    !haskey(kwargs_ret, :yguidefontsize) &&
        (kwargs_ret = (kwargs_ret..., yguidefontsize = 10))
    !haskey(kwargs_dd, :xlabel) && (kwargs_dd = (kwargs_dd..., xlabel = "Date"))
    !haskey(kwargs_dd, :ylim) &&
        (kwargs_dd = (kwargs_dd..., ylim = [minimum(dd) * 1.5, 0.01]))
    !haskey(kwargs_dd, :label) &&
        (kwargs_dd = (kwargs_dd..., label = "Uncompounded Cummulative Drawdown"))
    dd_plt = plot(timestamps, dd; kwargs_dd...)

    for (risk, label) in zip(risks, risk_labels)
        hline!([risk]; label = label, kwargs_risks...)
    end

    !haskey(kwargs_ret, :ylabel) &&
        (kwargs_ret = (kwargs_ret..., ylabel = "Cummulative Returns"))
    !haskey(kwargs_ret, :yguidefontsize) &&
        (kwargs_ret = (kwargs_ret..., yguidefontsize = 10))
    !haskey(kwargs_ret, :legend) && (kwargs_ret = (kwargs_ret..., legend = false))
    ret_plt = plot(timestamps, prices; kwargs_ret...)

    !haskey(kwargs_all, :legend_font_pointsize) &&
        (kwargs_all = (kwargs_all..., legend_font_pointsize = 8))
    !haskey(kwargs_all, :size) &&
        (kwargs_all = (kwargs_all..., size = (750, ceil(Integer, 750 / 1.618))))
    full_plt = plot(ret_plt, dd_plt; layout = (2, 1), kwargs_all...)

    return full_plt
end
function plot_drawdown(
    portfolio::AbstractPortfolio;
    type::Symbol = isa(portfolio, Portfolio) ? :Trad : :HRP,
    kwargs_ret = (;),
    kwargs_dd = (;),
    kwargs_risks = (;),
    kwargs_all = (;),
)
    return plot_drawdown(
        portfolio.timestamps,
        portfolio.optimal[type].weights,
        portfolio.returns;
        alpha = portfolio.alpha,
        kappa = portfolio.kappa,
        solvers = portfolio.solvers,
        kwargs_ret = kwargs_ret,
        kwargs_dd = kwargs_dd,
        kwargs_risks = kwargs_risks,
        kwargs_all = kwargs_all,
    )
end

function plot_hist(
    w::AbstractVector,
    returns::AbstractMatrix;
    alpha_i::Real = 0.0001,
    alpha::Real = 0.05,
    a_sim::Int = 100,
    kappa::Real = 0.3,
    solvers::Union{<:AbstractDict, Nothing} = nothing,
    points::Integer = ceil(Int, 4 * sqrt(size(returns, 1))),
    kwargs_hist = (;),
    kwargs_risks = (;),
)
    ret = returns * w * 100

    mu = mean(ret)
    sigma = std(ret)

    x = range(minimum(ret), stop = maximum(ret), length = points)
    D = fit(Normal, ret)

    !haskey(kwargs_hist, :ylabel) &&
        (kwargs_hist = (kwargs_hist..., ylabel = "Probability Density"))
    !haskey(kwargs_hist, :xlabel) &&
        (kwargs_hist = (kwargs_hist..., xlabel = "Percentage Returns"))

    plt = histogram(ret; normalize = :pdf, label = "", kwargs_hist...)

    mad = MAD(ret)
    gmd = GMD(ret)
    risks = [
        mu,
        mu - sigma,
        mu - mad,
        mu - gmd,
        -VaR(ret, alpha),
        -CVaR(ret, alpha),
        -TG(ret; alpha_i = alpha_i, alpha = alpha, a_sim = a_sim),
        -EVaR(ret, solvers, alpha),
        -RVaR(x, solvers, alpha, kappa),
        -WR(ret),
    ]

    conf = round((1 - alpha) * 100, digits = 2)

    risk_labels = [
        "Mean: $(round(risks[1], digits=2))%",
        "Mean - Std. Dev. ($(round(sigma, digits=2))%): $(round(risks[2], digits=2))%",
        "Mean - MAD ($(round(mad,digits=2))%): $(round(risks[3], digits=2))%",
        "Mean - GMD ($(round(gmd,digits=2))%): $(round(risks[4], digits=2))%",
        "$(conf)% Confidence VaR: $(round(risks[5], digits=2))%",
        "$(conf)% Confidence CVaR: $(round(risks[6], digits=2))%",
        "$(conf)% Confidence Tail Gini: $(round(risks[7], digits=2))%",
        "$(conf)% Confidence EVaR: $(round(risks[8], digits=2))%",
        "$(conf)% Confidence RVaR ($kappa): $(round(risks[9], digits=2))%",
        "Worst Realisation: $(round(risks[10], digits=2))%",
    ]

    for (risk, label) in zip(risks, risk_labels)
        vline!([risk]; label = label, kwargs_risks...)
    end

    !haskey(kwargs_hist, :size) &&
        (kwargs_hist = (kwargs_hist..., size = (750, ceil(Integer, 750 / 1.618))))

    plot!(
        x,
        pdf.(D, x),
        label = "Normal: μ = $(round(mean(D), digits=2))%, σ = $(round(std(D), digits=2))%";
        kwargs_hist...,
    )

    return plt
end

function plot_hist(
    portfolio::AbstractPortfolio;
    type::Symbol = isa(portfolio, Portfolio) ? :Trad : :HRP,
    points::Integer = ceil(Int, 4 * sqrt(size(portfolio.returns, 1))),
    kwargs_hist = (;),
    kwargs_risks = (;),
)
    return plot_hist(
        portfolio.optimal[type].weights,
        portfolio.returns;
        alpha_i = portfolio.alpha_i,
        alpha = portfolio.alpha,
        a_sim = portfolio.a_sim,
        kappa = portfolio.kappa,
        solvers = portfolio.solvers,
        points = points,
        kwargs_hist = kwargs_hist,
        kwargs_risks = kwargs_risks,
    )
end

function plot_range(
    w::AbstractVector,
    returns::AbstractMatrix;
    alpha_i::Real = 0.0001,
    alpha::Real = 0.05,
    a_sim::Int = 100,
    beta_i = nothing,
    beta = nothing,
    b_sim = nothing,
    points::Integer = ceil(Int, 4 * sqrt(size(returns, 1))),
    kwargs_hist = (;),
    kwargs_risks = (;),
)
    isnothing(beta) && (beta = alpha)

    ret = returns * w * 100

    mu = mean(ret)
    sigma = std(ret)

    !haskey(kwargs_hist, :ylabel) &&
        (kwargs_hist = (kwargs_hist..., ylabel = "Probability Density"))
    !haskey(kwargs_hist, :xlabel) &&
        (kwargs_hist = (kwargs_hist..., xlabel = "Percentage Returns"))

    plt = histogram(ret; normalize = :pdf, label = "", kwargs_hist...)

    risks = (
        RG(ret),
        RCVaR(ret; alpha = alpha, beta = beta),
        RTG(
            ret;
            alpha_i = alpha_i,
            alpha = alpha,
            a_sim = a_sim,
            beta_i = beta_i,
            beta = beta,
            b_sim = b_sim,
        ),
    )

    lo_conf = 1 - alpha
    hi_conf = 1 - beta
    risk_labels = (
        "Range: $(round(risks[1], digits=2))%",
        "Tail Gini Range ($(round(lo_conf,digits=2)), $(round(hi_conf,digits=2))): $(round(risks[2], digits=2))%",
        "CVaR Range ($(round(lo_conf,digits=2)), $(round(hi_conf,digits=2))): $(round(risks[3], digits=2))%",
    )

    bounds = [
        minimum(ret) -TG(ret; alpha_i = alpha_i, alpha = alpha, a_sim = a_sim) -CVaR(ret, alpha)
        maximum(ret) TG(-ret; alpha_i = alpha_i, alpha = alpha, a_sim = a_sim) CVaR(-ret, alpha)
    ]

    D = fit(Normal, ret)
    y = pdf(D, mean(D))
    ys = (y / 4, y / 2, y * 3 / 4)

    for i in eachindex(risks)
        plot!(
            [bounds[1, i], bounds[1, i], bounds[2, i], bounds[2, i]],
            [0, ys[i], ys[i], 0],
            label = risk_labels[i],
            kwargs_risks...,
        )
    end

    return plt
end

function plot_range(
    portfolio::AbstractPortfolio;
    type::Symbol = isa(portfolio, Portfolio) ? :Trad : :HRP,
    points::Integer = ceil(Int, 4 * sqrt(size(portfolio.returns, 1))),
    kwargs_hist = (;),
    kwargs_risks = (;),
)
    return plot_range(
        portfolio.optimal[type].weights,
        portfolio.returns;
        alpha_i = portfolio.alpha_i,
        alpha = portfolio.alpha,
        a_sim = portfolio.a_sim,
        beta_i = portfolio.beta_i,
        beta = portfolio.beta,
        b_sim = portfolio.b_sim,
        points = points,
        kwargs_hist = kwargs_hist,
        kwargs_risks = kwargs_risks,
    )
end

export plot_returns,
    plot_bar,
    plot_risk_contribution,
    plot_frontier_area,
    plot_drawdown,
    plot_hist,
    plot_range,
    plot_frontier
