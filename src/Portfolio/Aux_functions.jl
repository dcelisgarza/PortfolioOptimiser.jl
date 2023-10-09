
#=
function gen_dataframes(portfolio)
    nms = portfolio.assets
    nms2 = vec(["$(i)-$(j)" for i in nms, j in nms])

    df_returns = hcat(
        DataFrame(timestamp = portfolio.timestamps),
        DataFrame(portfolio.returns, portfolio.assets),
    )
    df_mu = DataFrame(ticker = nms, val = portfolio.mu)
    df_cov = DataFrame(portfolio.cov, nms)
    df_kurt = DataFrame(portfolio.kurt, nms2)
    df_skurt = DataFrame(portfolio.skurt, nms2)

    df_cov_l = DataFrame(portfolio.cov_l, nms)
    df_cov_u = DataFrame(portfolio.cov_u, nms)
    df_cov_mu = DataFrame(portfolio.cov_mu, nms)
    df_cov_sigma = DataFrame(portfolio.cov_sigma, nms2)
    df_dmu = DataFrame(ticker = nms, val = portfolio.d_mu)

    return df_returns,
    df_mu,
    df_cov,
    df_kurt,
    df_skurt,
    df_cov_l,
    df_cov_u,
    df_cov_mu,
    df_cov_sigma,
    df_dmu
end

function cov_returns(x; seed = nothing, rng = Random.default_rng(), len = 10, iters = 5)
    !isnothing(seed) && Random.seed!(rng, seed)

    n = size(x)[1]
    a = randn(rng, n + len, n)

    for _ in 1:iters
        _cov = cov(a)
        _C = cholesky(_cov)
        a .= transpose(_C.L \ transpose(a))
        _cov = cov(a)
        _desv = transpose(sqrt.(diag(_cov)))
        a .= (a .- mean(a, dims = 1)) ./ _desv
    end

    C = cholesky(x)
    return a * C.U
end

export cov_returns

=#
