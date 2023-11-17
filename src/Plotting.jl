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

function plot_bar(assets, weights, others = 0.05; kwargs...)
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

    bar(assets, weights * 100; kwargs...)
end

function plot_bar(
    portfolio::AbstractPortfolio,
    type = isa(portfolio, HCPortfolio) ? :HRP : :Trad,
    others = 0.05;
    kwargs...,
)
    return plot_bar(
        portfolio.assets,
        portfolio.optimal[type].weights,
        others = others;
        kwargs...,
    )
end
export plot_returns, plot_bar
