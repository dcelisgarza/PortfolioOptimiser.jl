
function _parse_views(tickers, views::Dict)
    K = length(views)
    Q = zeros(K)
    P = zeros(K, length(tickers))

    for (i, ticker) in enumerate(keys(views))
        Q[i] = views[ticker]
        P[i, findfirst(x -> x == ticker, tickers)] = 1
    end

    return Q, P
end

function market_implied_prior_returns(market_caps, cov_mtx, risk_aversion = 1, rf = 0.02)
    mkt_weights = market_caps / sum(market_caps)

    return risk_aversion * cov_mtx * mkt_weights .+ rf
end

function market_implied_risk_aversion(market_prices, freq = 252, rf = 0.02)
    rets = returns_from_prices(market_prices)
    μ = mean(rets) * freq
    σ = var(rets) * freq
    return (μ - rf) / σ
end

function idzorek(view_confidence, cov_mtx, Q, P, tau)
    lq = length(Q)
    view_omegas = Vector{Float64}(undef, lq)

    for i in 1:lq
        conf = view_confidence[i]

        if !(0 <= conf <= 1)
            throw(
                ArgumentError(
                    "view confidences must be between 0 and 1, errored at view $i with value $conf",
                ),
            )
        end

        if conf < eps()
            view_omegas[i] = Inf
            continue
        end

        P_view = P[i, :]
        alpha = (1 - conf) / conf # eq (44)
        omega = tau * alpha * dot(P_view, cov_mtx, P_view) # eq (45)
        view_omegas[i] = omega
    end

    return Diagonal(view_omegas)
end

function calc_weights!(
    portfolio::AbstractBlackLitterman,
    risk_aversion = portfolio.risk_aversion,
)
    if risk_aversion <= 0
        @warn(
            "risk_aversion: $risk_aversion, must be greater than zero, defaulting to the value registered in portfolio.risk_aversion: $(portfolio.risk_aversion)"
        )
        risk_aversion = portfolio.risk_aversion
    end

    if risk_aversion != portfolio.risk_aversion
        @warn(
            "the weights of the portfolio will match the risk aversion provided to this function: $risk_aversion, and not the one registered in portfolio.risk_aversion: $(portfolio.risk_aversion)"
        )
    end

    cov_mtx = portfolio.cov_mtx
    post_ret = portfolio.post_ret

    weights = (risk_aversion * cov_mtx) \ post_ret
    weights /= sum(weights)

    portfolio.weights .= weights
end

function calc_weights(
    portfolio::AbstractBlackLitterman,
    risk_aversion = portfolio.risk_aversion,
)
    if risk_aversion <= 0
        @warn(
            "risk_aversion: $risk_aversion, must be greater than zero, defaulting to the value registered in portfolio.risk_aversion: $(portfolio.risk_aversion)"
        )
        risk_aversion = portfolio.risk_aversion
    end

    cov_mtx = portfolio.cov_mtx
    post_ret = portfolio.post_ret

    weights = (risk_aversion * cov_mtx) \ post_ret
    weights /= sum(weights)

    return weights
end