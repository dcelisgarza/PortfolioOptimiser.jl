# distinguishable_colors{T <: Color}(
#     n::Integer,
#     seed::AbstractVector{T};
#     dropseed = false,
#     transform::Function = identity,
#     lchoices::AbstractVector = range(0, stop = 100, length = 15),
#     cchoices::AbstractVector = range(0, stop = 100, length = 15),
#     hchoices::AbstractVector = range(0, stop = 342, length = 20),
# )

function plot_returns(timestamps, assets, returns, weights; per_asset = false, kwargs...)
    if per_asset
        pret = returns .* transpose(weights)
        pret = vcat(zeros(1, length(weights)), pret)
        pret .+= 1
        pret = cumprod(pret, dims = 1)
        pret = pret[2:end, :]
        !haskey(kwargs, :label) && (kwargs = (kwargs..., label = reshape(assets, 1, :)))
    else
        pret = returns * weights
        pushfirst!(pret, 0)
        pret .+= 1
        pret = cumprod(pret)
        popfirst!(pret)
        !haskey(kwargs, :label) && (kwargs = (kwargs..., label = "Portfolio"))
    end
    !haskey(kwargs, :ylabel) && (kwargs = (kwargs..., ylabel = "Cummulative Return"))
    !haskey(kwargs, :xlabel) && (kwargs = (kwargs..., xlabel = "Date"))

    plot(timestamps, pret; kwargs...)
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
    factor = 252,
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

    rm ∉ DDs && (rc *= sqrt(factor))
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
            rm ∉ DDs && (erc *= sqrt(factor))
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
    factor = 252,
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
        factor = factor,
        kwargs_bar = kwargs_bar,
        kwargs_line = kwargs_line,
    )
end

function plot_frontier(w_frontier, rm = :SD; kwargs...)
    risks = w_frontier[rm][:risk]
    assets = reshape(w_frontier[rm][:weights][!, "tickers"], 1, :)
    weights = transpose(Matrix(w_frontier[rm][:weights][!, 2:end]))

    !haskey(kwargs, :ylabel) && (kwargs = (kwargs..., ylabel = "Percentage Composition"))
    !haskey(kwargs, :xlabel) && (kwargs = (kwargs..., xlabel = "Risk"))
    !haskey(kwargs, :label) && (kwargs = (kwargs..., label = assets))
    !haskey(kwargs, :xtick) && (kwargs = (kwargs..., xtick = round.(risks, digits = 3)))
    !haskey(kwargs, :xrotation) && (kwargs = (kwargs..., xrotation = 60))

    return areaplot(risks, weights; kwargs...)
end

function plot_drawdown(
    timestamps,
    w::AbstractVector,
    returns::AbstractMatrix;
    alpha::Real = 0.05,
    kappa::Real = 0.3,
    solvers::Union{<:AbstractDict, Nothing} = nothing,
    kwargs_lines = (;),
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

    data = [prices, dd]
    risks = (
        -DaR_abs(ret, alpha),
        -MDD_abs(ret),
        -ADD_abs(ret),
        -CDaR_abs(ret, alpha),
        -UCI_abs(ret),
        -EDaR_abs(ret, solvers, alpha),
        -RDaR_abs(ret, solvers, alpha, kappa),
    )

    plt = plot(timestamps, dd)
    for risk in risks
        hline!([risk]; kwargs_lines...)
    end

    return plt
end
export plot_returns, plot_bar, plot_risk_contribution, plot_frontier, plot_drawdown
