abstract type AbstractHRPOpt <: AbstractPortfolioOptimiser end

struct HRPOpt{T1, T2, T3, T4, T5, T6, T7, T8} <: AbstractHRPOpt
    tickers::T1
    mean_ret::T2
    weights::T3
    returns::T4
    cov_mtx::T5
    risk_aversion::T6
    linkage::T7
    clusters::T8
end
function HRPOpt(
    tickers::AbstractVector{<:AbstractString};
    linkage::Symbol = :single,
    returns = nothing,
    cov_mtx = nothing,
    mean_ret = nothing,
    risk_aversion = 1,
    D = :default,
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

    if D == :default
        D = Symmetric(sqrt.(clamp.((1 .- cor_mtx) / 2, 0, 1)))
    elseif D <: AbstractArray
        @assert size(D) == size(cov_mtx)
    else
        throw(
            ArgumentError(
                "Distance matrix D must be :default, or a square matrix if size equal to the covariance matrix.",
            ),
        )
    end
    clusters = hclust(D, linkage = linkage)

    weights = zeros(length(tickers))

    return HRPOpt(
        tickers,
        mean_ret,
        weights,
        returns,
        cov_mtx,
        risk_aversion,
        linkage,
        clusters,
    )
end

function min_volatility(portfolio::HRPOpt, cluster_ticker_idx)
    cov_slice = portfolio.cov_mtx[cluster_ticker_idx, cluster_ticker_idx]

    weights = _hrp_sub_weights(cov_slice)

    return port_variance(weights, cov_slice)
end

function max_return(portfolio::HRPOpt, cluster_ticker_idx)
    cov_slice = portfolio.cov_mtx[cluster_ticker_idx, cluster_ticker_idx]

    weights = _hrp_sub_weights(cov_slice)

    return port_return(weights, cov_slice)
end

function max_quadratic_utility(portfolio::HRPOpt, cluster_ticker_idx)
    isnothing(portfolio.mean_ret) ?
    mean_ret = mean(portfolio.returns[:, cluster_ticker_idx], dims = 1) :
    mean_ret = portfolio.mean_ret[cluster_ticker_idx]

    cov_slice = portfolio.cov_mtx[cluster_ticker_idx, cluster_ticker_idx]

    risk_aversion = portfolio.risk_aversion

    weights = _hrp_sub_weights(cov_slice)

    return 1 / quadratic_utility(weights, mean_ret, cov_slice, risk_aversion)
end

function _hrp_sub_weights(cov_slice)
    weights = diag(inv(Diagonal(cov_slice)))
    weights /= sum(weights)

    return weights
end

function optimise!(portfolio::HRPOpt, obj, obj_params...)
    ordered_ticker_idx = portfolio.clusters.order

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
            # Maximise the inverse of the convex objective.
            first_measure = obj(portfolio, first_cluster, obj_params...)
            second_measure = obj(portfolio, second_cluster, obj_params...)
            alpha = 1 - first_measure / (first_measure + second_measure)
            w[first_cluster] *= alpha  # weight 1
            w[second_cluster] *= 1 - alpha  # weight 2
        end
    end

    portfolio.weights .= w

    return nothing
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