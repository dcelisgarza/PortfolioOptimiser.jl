function cluster_weight_bounds(w_min, w_max, weights, lc, rc, alpha_1)
    if !(any(w_max .< weights) || any(w_min .> weights))
        return alpha_1
    end
    lmaxw = weights[lc[1]]
    if !iszero(lmaxw)
        a1 = sum(w_max[lc]) / lmaxw
        a2 = max(sum(w_min[lc]) / lmaxw, alpha_1)
        alpha_1 = min(a1, a2)
    end

    rmaxw = weights[rc[1]]
    if !iszero(rmaxw)
        a1 = sum(w_max[rc]) / rmaxw
        a2 = max(sum(w_min[rc]) / rmaxw, 1 - alpha_1)
        alpha_1 = one(a1) - min(a1, a2)
    end
    return alpha_1
end

function opt_weight_bounds(w_min, w_max, weights, max_iter = 100)
    if !(any(w_max .< weights) || any(w_min .> weights))
        return weights
    end

    for _ ∈ 1:max_iter
        if !(any(w_max .< weights) || any(w_min .> weights))
            break
        end

        old_w = copy(weights)
        weights = max.(min.(weights, w_max), w_min)
        idx = weights .< w_max .&& weights .> w_min
        w_add = sum(max.(old_w - w_max, 0.0))
        w_sub = sum(min.(old_w - w_min, 0.0))
        delta = w_add + w_sub

        if delta != 0
            weights[idx] += delta * weights[idx] / sum(weights[idx])
        end
    end
    return weights
end
