function max_sharpe(portfolio::HRPOpt, cluster_ticker_idx, rf = portfolio.rf)
    isnothing(portfolio.mean_ret) ?
    mean_ret = mean(portfolio.returns[:, cluster_ticker_idx], dims = 1) :
    mean_ret = portfolio.mean_ret[cluster_ticker_idx]

    cov_slice = portfolio.cov_mtx[cluster_ticker_idx, cluster_ticker_idx]

    weights = _hrp_sub_weights(cov_slice)

    mu = port_return(weights, mean_ret)
    sigma = sqrt(port_variance(weights, cov_slice))
    return sigma / (mu - rf)
end

function min_volatility(portfolio::HRPOpt, cluster_ticker_idx)
    cov_slice = portfolio.cov_mtx[cluster_ticker_idx, cluster_ticker_idx]

    weights = _hrp_sub_weights(cov_slice)

    return port_variance(weights, cov_slice)
end

function max_return(portfolio::HRPOpt, cluster_ticker_idx)
    isnothing(portfolio.mean_ret) ?
    mean_ret = mean(portfolio.returns[:, cluster_ticker_idx], dims = 1) :
    mean_ret = portfolio.mean_ret[cluster_ticker_idx]

    cov_slice = portfolio.cov_mtx[cluster_ticker_idx, cluster_ticker_idx]

    weights = _hrp_sub_weights(cov_slice)

    return 1 / port_return(weights, mean_ret)
end

function max_quadratic_utility(
    portfolio::HRPOpt,
    cluster_ticker_idx,
    risk_aversion = portfolio.risk_aversion,
)
    isnothing(portfolio.mean_ret) ?
    mean_ret = mean(portfolio.returns[:, cluster_ticker_idx], dims = 1) :
    mean_ret = portfolio.mean_ret[cluster_ticker_idx]

    cov_slice = portfolio.cov_mtx[cluster_ticker_idx, cluster_ticker_idx]

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