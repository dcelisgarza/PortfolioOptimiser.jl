abstract type AbstractHRPOpt <: AbstractPortfolioOptimiser end

struct HRPOpt{T1, T2, T3, T4, T5, T6} <: AbstractHRPOpt
    tickers::T1
    weights::T2
    returns::T3
    cov_mtx::T4
    linkage::T5
    clusters::T6
end
function HRPOpt(
    tickers::AbstractVector{<:AbstractString};
    linkage::Symbol = :single,
    returns = nothing,
    cov_mtx = nothing,
)
    if isnothing(returns) && isnothing(cov_mtx)
        throw(ArgumentError("Either returns or cov_mtx must be defined."))
    elseif isnothing(returns)
        @assert size(cov_mtx, 1) == size(cov_mtx, 2) == length(tickers)
        cor_mtx = cov2cor(cov_mtx)
    elseif isnothing(cov_mtx)
        @assert size(returns, 2) == length(tickers)
        cov_mtx = cov(returns)
        cor_mtx = cor(returns)
    else
        @assert size(cov_mtx, 1) == size(cov_mtx, 2) == size(returns, 2) == length(tickers)
        cor_mtx = cov2cor(cov_mtx)
    end

    D = Symmetric(sqrt.(clamp.((1 .- cor_mtx) / 2, 0, 1)))
    clusters = hclust(D, linkage = linkage)
    order = clusters.order
    weights = hrp_allocation(order, cov_mtx)

    return HRPOpt(tickers, weights, returns, cov_mtx, linkage, clusters)
end

function cluster_var(cluster_ticker_idx, cov_mtx)
    cov_slice = cov_mtx[cluster_ticker_idx, cluster_ticker_idx]
    weights = diag(inv(Diagonal(cov_slice)))
    weights /= sum(weights)
    return dot(weights, cov_slice, weights)
end

function hrp_allocation(ordered_ticker_idx, cov_mtx)
    w = ones(length(ordered_ticker_idx))
    cluster_tickers = [ordered_ticker_idx] # All items in one cluster.

    while length(cluster_tickers) > 0
        cluster_tickers = [
            i[j:k] for i in cluster_tickers for
            (j, k) in ((1, div(length(i), 2)), (div(length(i), 2) + 1, length(i))) if
            length(i) > 1
        ] # Bisecting
        # For each pair optimise locally.
        for i in 1:2:length(cluster_tickers)
            first_cluster = cluster_tickers[i]
            second_cluster = cluster_tickers[i + 1]
            # Form the inverse variance portfolio for this pair
            first_variance = cluster_var(first_cluster, cov_mtx)
            second_variance = cluster_var(second_cluster, cov_mtx)
            alpha = 1 - first_variance / (first_variance + second_variance)
            w[first_cluster] *= alpha  # weight 1
            w[second_cluster] *= 1 - alpha  # weight 2
        end
    end
    return w
end

function portfolio_performance(portfolio::HRPOpt; rf = 0.02, freq = 252, verbose = false)
    w = portfolio.weights

    if isnothing(portfolio.returns)
        cov_mtx = portfolio.cov_mtx
        μ = NaN
        sr = NaN

        σ = sqrt(port_variance(w, cov_mtx))

        if verbose
            println("Annual volatility: $(round(100*σ, digits=2)) %")
        end
    else
        cov_mtx = cov(portfolio.returns) * freq
        mean_ret = vec(mean(portfolio.returns, dims = 1) * freq)

        μ = dot(w, mean_ret)
        σ = sqrt(port_variance(w, cov_mtx))
        sr = sharpe_ratio(μ, σ, rf)

        if verbose
            println("Expected annual return: $(round(100*μ, digits=2)) %")
            println("Annual volatility: $(round(100*σ, digits=2)) %")
            println("Sharpe Ratio: $(round(sr, digits=3))")
        end
    end

    return μ, σ, sr
end