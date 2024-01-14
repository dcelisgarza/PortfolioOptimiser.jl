function max_sharpe!(portfolio::HRPOpt, w, cluster1_idx, cluster2_idx, rf = portfolio.rf)
    mean_ret1, mean_ret2 = _get_mean_ret(portfolio, cluster1_idx, cluster2_idx)

    cov_slice1 = portfolio.cov_mtx[cluster1_idx, cluster1_idx]
    cov_slice2 = portfolio.cov_mtx[cluster2_idx, cluster2_idx]

    weights1 = _hrp_sub_weights(cov_slice1)
    weights2 = _hrp_sub_weights(cov_slice2)

    sr1 = sharpe_ratio(weights1, mean_ret1, cov_slice1, rf)
    sr2 = sharpe_ratio(weights2, mean_ret2, cov_slice2, rf)

    return _hrp_minimise(w, cluster1_idx, cluster2_idx, -sr1, -sr2)
end

function min_risk!(portfolio::HRPOpt, w, cluster1_idx, cluster2_idx)
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

    return _hrp_minimise(w, cluster1_idx, cluster2_idx, -ret1, -ret2)
end

function max_utility!(portfolio::HRPOpt, w, cluster1_idx, cluster2_idx,
                      risk_aversion = portfolio.risk_aversion)
    mean_ret1, mean_ret2 = _get_mean_ret(portfolio, cluster1_idx, cluster2_idx)

    cov_slice1 = portfolio.cov_mtx[cluster1_idx, cluster1_idx]
    cov_slice2 = portfolio.cov_mtx[cluster2_idx, cluster2_idx]

    weights1 = _hrp_sub_weights(cov_slice1)
    weights2 = _hrp_sub_weights(cov_slice2)

    qu1 = quadratic_utility(weights1, mean_ret1, cov_slice1, risk_aversion)
    qu2 = quadratic_utility(weights2, mean_ret2, cov_slice2, risk_aversion)

    return _hrp_minimise(w, cluster1_idx, cluster2_idx, -qu1, -qu2)
end

function optimise!(portfolio::HRPOpt, obj, obj_params...)
    ordered_ticker_idx = portfolio.clusters.order

    w = ones(length(ordered_ticker_idx))
    cluster_tickers = [ordered_ticker_idx] # All items in one cluster.

    while length(cluster_tickers) > 0
        cluster_tickers = [i[j:k] for i ∈ cluster_tickers
                           for (j, k) ∈
                               ((1, div(length(i), 2)), (div(length(i), 2) + 1, length(i)))
                           if length(i) > 1] # Bisecting
        # For each pair optimise locally.
        for i ∈ 1:2:length(cluster_tickers)
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

# function _hrp_maximise(w, cluster1_idx, cluster2_idx, val1, val2)
#     if val1 < 0 && val2 < 0
#         # If both val1 and val2 are negative, we want the least negative of the two to have the larger weight because we're maximising the function.
#         # If abs(val1) >>> abs(val2), alpha -> 1, the most negative would be val1 so we want its weight to be smaller.
#         alpha = val1 / (val1 + val2)

#         w[cluster1_idx] *= 1 - alpha  # weight 1
#         w[cluster2_idx] *= alpha  # weight 2
#         return nothing
#     end

#     if val1 < 0
#         alpha = val1 / (val1 - val2)
#         if alpha > 0.5
#             w[cluster1_idx] *= 1 - alpha  # weight 1
#             w[cluster2_idx] *= alpha  # weight 1
#         else
#             w[cluster1_idx] *= alpha  # weight 1
#             w[cluster2_idx] *= 1 - alpha  # weight 1
#         end

#         return nothing
#     end

#     if val2 < 0
#         alpha = val2 / (val2 - val1)
#         if alpha > 0.5
#             w[cluster1_idx] *= alpha  # weight 2
#             w[cluster2_idx] *= 1 - alpha  # weight 2
#         else
#             w[cluster1_idx] *= 1 - alpha  # weight 2
#             w[cluster2_idx] *= alpha  # weight 2
#         end

#         return nothing
#     end

#     # If both val1 and val2 are positive, we want the larger of the two to have the larger weight because we're maximising the function.
#     # If val1 >>> val2, alpha -> 1.
#     alpha = val1 / (val1 + val2)

#     w[cluster1_idx] *= alpha        # weight 1
#     w[cluster2_idx] *= 1 - alpha    # weight 2

#     return nothing
# end

function _hrp_minimise(w, cluster1_idx, cluster2_idx, val1, val2)
    if val1 < 0 && val2 < 0
        # If both val1 and val2 are negative, we want the cluster with the most negative of the two to have the larger weight because we're minimising the function.
        # If abs(val1) >>> abs(val2), alpha -> 0.
        # If abs(val2) >>> abs(val1), alpha -> 1.
        alpha = 1 - val1 / (val1 + val2)

        w[cluster1_idx] *= 1 - alpha    # weight 1
        w[cluster2_idx] *= alpha        # weight 2
        return nothing
    end

    if val1 < 0
        # If only val1 is negative, we want the weight of cluster1_idx to be multiplied by the number closer to 1.
        # As val1 -> -inf, alpha -> 1.
        # As val1 -> 0, alpha -> 0.
        # As val2 -> inf, alpha -> 0
        # As val2 -> 0, alpha -> 1.
        alpha = val1 / (val1 - val2)
        # The if statement ensures cluster1_idx is always multiplied by the number closer to 1.
        if alpha > 0.5
            w[cluster1_idx] *= alpha  # weight 1
            w[cluster2_idx] *= 1 - alpha  # weight 1
        else
            w[cluster1_idx] *= 1 - alpha  # weight 1
            w[cluster2_idx] *= alpha  # weight 1
        end

        return nothing
    end

    if val2 < 0
        # If only val2 is negative, we want the weight of cluster2_idx to be multiplied by the number closer to 1.
        # As val2 -> -inf, alpha -> 1.
        # As val2 -> 0, alpha -> 0.
        # As val1 -> inf, alpha -> 0
        # As val1 -> 0, alpha -> 1.
        alpha = val2 / (val2 - val1)
        # The if statement ensures cluster2_idx is always multiplied by the number closer to 1.
        if alpha > 0.5
            w[cluster1_idx] *= 1 - alpha  # weight 1
            w[cluster2_idx] *= alpha  # weight 1
        else
            w[cluster1_idx] *= alpha  # weight 1
            w[cluster2_idx] *= 1 - alpha  # weight 1
        end

        return nothing
    end

    # If both val1 and val2 are positive, we want the weight cluster corresponding to the smaller value to be multiplied by the number closest to 1.
    # As val1 -> 0, alpha -> 1.
    # As val1 -> inf, alpha -> 0.
    # As val2 -> 0, alpha -> 0.
    # As val2 -> inf, alpha -> 1.
    alpha = 1 - val1 / (val1 + val2)

    w[cluster1_idx] *= alpha        # weight 1
    w[cluster2_idx] *= 1 - alpha    # weight 2

    return nothing
end

function _get_mean_ret(portfolio, cluster1_idx, cluster2_idx)
    if isnothing(portfolio.mean_ret)
        mean_ret1 = ret_model(MRet(), portfolio.returns[:, cluster1_idx])
        mean_ret2 = ret_model(MRet(), portfolio.returns[:, cluster2_idx])
    else
        mean_ret1 = portfolio.mean_ret[cluster1_idx]
        mean_ret2 = portfolio.mean_ret[cluster2_idx]
    end

    return mean_ret1, mean_ret2
end
