
function _parse_views(tickers, views::Dict)
    K = length(views)
    Q = zeros(K)
    P = zeros(length(tickers), K)

    for (i, ticker) ∈ enumerate(keys(views))
        Q[i] = views[ticker]
        P[findfirst(x -> x == ticker, tickers), i] = 1
    end

    return Q, P
end

function market_implied_prior_returns(market_caps, cov_mtx, risk_aversion = 1,
                                      rf = 1.02^(1 / 252) - 1)
    mkt_weights = market_caps / sum(market_caps)

    return risk_aversion * cov_mtx * mkt_weights .+ rf
end

function market_implied_risk_aversion(market_prices, rf = 1.02^(1 / 252) - 1)
    rets = returns_from_prices(market_prices)
    μ = mean(rets)
    σ = var(rets)
    return (μ - rf) / σ
end

function idzorek(view_confidence, cov_mtx, Q, P, tau)
    lq = length(Q)
    view_omegas = Vector{eltype(cov_mtx)}(undef, lq)

    for i ∈ 1:lq
        conf = view_confidence[i]

        if !(0 <= conf <= 1)
            throw(DomainError("view confidences must be between 0 and 1, errored at view $i with value $conf"))
        end

        if conf < eps()
            view_omegas[i] = Inf
            continue
        end

        P_view = P[:, i]
        alpha = (1 - conf) / conf # eq (44)
        omega = tau * alpha * dot(P_view, cov_mtx, P_view) # eq (45)
        view_omegas[i] = omega
    end

    return Diagonal(view_omegas)
end
