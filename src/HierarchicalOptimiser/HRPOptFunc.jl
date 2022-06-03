function max_sharpe!(portfolio::HRPOpt, w, cluster1_idx, cluster2_idx, rf = portfolio.rf)
    mean_ret1, mean_ret2 = _get_mean_ret(portfolio, cluster1_idx, cluster2_idx)

    cov_slice1 = portfolio.cov_mtx[cluster1_idx, cluster1_idx]
    cov_slice2 = portfolio.cov_mtx[cluster2_idx, cluster2_idx]

    weights1 = _hrp_sub_weights(cov_slice1)
    weights2 = _hrp_sub_weights(cov_slice2)

    sr1 = sharpe_ratio(weights1, mean_ret1, cov_slice1, rf)
    sr2 = sharpe_ratio(weights2, mean_ret2, cov_slice2, rf)

    _hrp_maximise(w, cluster1_idx, cluster2_idx, sr1, sr2)
end

function min_volatility!(portfolio::HRPOpt, w, cluster1_idx, cluster2_idx)
    cov_slice1 = portfolio.cov_mtx[cluster1_idx, cluster1_idx]
    cov_slice2 = portfolio.cov_mtx[cluster2_idx, cluster2_idx]

    weights1 = _hrp_sub_weights(cov_slice1)
    weights2 = _hrp_sub_weights(cov_slice2)

    σ2_1 = port_variance(weights1, cov_slice1)
    σ2_2 = port_variance(weights2, cov_slice2)

    _hrp_minimise(w, cluster1_idx, cluster2_idx, σ2_1, σ2_2)

    return nothing
end

function max_return!(portfolio::HRPOpt, w, cluster1_idx, cluster2_idx)
    mean_ret1, mean_ret2 = _get_mean_ret(portfolio, cluster1_idx, cluster2_idx)

    cov_slice1 = portfolio.cov_mtx[cluster1_idx, cluster1_idx]
    cov_slice2 = portfolio.cov_mtx[cluster2_idx, cluster2_idx]

    weights1 = _hrp_sub_weights(cov_slice1)
    weights2 = _hrp_sub_weights(cov_slice2)

    ret1 = port_return(weights1, mean_ret1)
    ret2 = port_return(weights2, mean_ret2)

    _hrp_maximise(w, cluster1_idx, cluster2_idx, ret1, ret2)
end

function max_quadratic_utility!(
    portfolio::HRPOpt,
    w,
    cluster1_idx,
    cluster2_idx,
    risk_aversion = portfolio.risk_aversion,
)
    mean_ret1, mean_ret2 = _get_mean_ret(portfolio, cluster1_idx, cluster2_idx)

    cov_slice1 = portfolio.cov_mtx[cluster1_idx, cluster1_idx]
    cov_slice2 = portfolio.cov_mtx[cluster2_idx, cluster2_idx]

    weights1 = _hrp_sub_weights(cov_slice1)
    weights2 = _hrp_sub_weights(cov_slice2)

    qu1 = quadratic_utility(weights1, mean_ret1, cov_slice1, risk_aversion)
    qu2 = quadratic_utility(weights2, mean_ret2, cov_slice2, risk_aversion)

    _hrp_maximise(w, cluster1_idx, cluster2_idx, qu1, qu2)
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

            obj(portfolio, w, first_cluster, second_cluster, obj_params...)
        end
    end

    portfolio.weights .= w

    return nothing
end

function _hrp_sub_weights(cov_slice)
    weights = 1 ./ diag(cov_slice)
    weights /= sum(weights)

    return weights
end

function _hrp_maximise(w, cluster1_idx, cluster2_idx, val1, val2)
    if val1 < 0 && val2 < 0
        alpha = val1 / (val1 + val2)

        w[cluster1_idx] *= 1 - alpha  # weight 1
        w[cluster2_idx] *= alpha  # weight 2
        return nothing
    end

    if val1 < 0
        alpha = val1 / (val1 - val2)
        w[cluster1_idx] *= alpha  # weight 1

        return nothing
    end

    if val2 < 0
        alpha = val2 / (val2 - val1)
        w[cluster2_idx] *= alpha  # weight 1

        return nothing
    end

    alpha = val1 / (val1 + val2)

    w[cluster1_idx] *= alpha  # weight 1
    w[cluster2_idx] *= 1 - alpha  # weight 2

    return nothing
end

function _hrp_minimise(w, cluster1_idx, cluster2_idx, val1, val2)
    if val1 < 0 && val2 < 0
        alpha = 1 - val1 / (val1 + val2)

        w[cluster1_idx] *= 1 - alpha  # weight 1
        w[cluster2_idx] *= alpha  # weight 2
        return nothing
    end

    if val1 < 0
        alpha = 1 - val1 / (val1 - val2)
        w[cluster1_idx] *= alpha  # weight 1

        return nothing
    end

    if val2 < 0
        alpha = 1 - val2 / (val2 - val1)
        w[cluster2_idx] *= alpha  # weight 1

        return nothing
    end

    alpha = 1 - val1 / (val1 + val2)

    w[cluster1_idx] *= alpha  # weight 1
    w[cluster2_idx] *= 1 - alpha  # weight 2

    return nothing
end

function _get_mean_ret(portfolio, cluster1_idx, cluster2_idx)
    if isnothing(portfolio.mean_ret)
        mean_ret1 =
            ret_model(MRet(), portfolio.returns[:, cluster1_idx], freq = portfolio.freq)
        mean_ret2 =
            ret_model(MRet(), portfolio.returns[:, cluster2_idx], freq = portfolio.freq)
    else
        mean_ret1 = portfolio.mean_ret[cluster1_idx]
        mean_ret2 = portfolio.mean_ret[cluster2_idx]
    end

    return mean_ret1, mean_ret2
end