function asset_statistics!(
    portfolio::Portfolio,
    target_ret::AbstractFloat = 0.0,
    mean_func::Function = mean,
    cov_func::Function = cov,
    mean_args = (),
    cov_args = ();
    mean_kwargs = (;),
    cov_kwargs = (;),
)
    N = ncol(portfolio.returns) - 1

    mu = vec(
        !haskey(mean_kwargs, :dims) ?
        mean_func(
            Matrix(portfolio.returns[!, 2:end]),
            mean_args...;
            dims = 1,
            mean_kwargs...,
        ) :
        mean_func(Matrix(portfolio.returns[!, 2:end]), mean_args...; mean_kwargs...),
    )

    portfolio.mu = DataFrame(tickers = names(portfolio.returns)[2:end], val = mu)
    cov_mtx = cov_func(Matrix(portfolio.returns[!, 2:end]), cov_args...; cov_kwargs...)
    nms = names(portfolio.returns[!, 2:end])
    portfolio.cov = DataFrame(cov_mtx, nms)
    portfolio.kurt = cokurt(portfolio.returns[!, 2:end], transpose(mu))
    portfolio.skurt = scokurt(portfolio.returns[!, 2:end], transpose(mu), target_ret)
    missing, portfolio.L_2, portfolio.S_2 = dup_elim_sum_matrices(N)

    return nothing
end

const EllipseTypes = (:stationary, :circular, :moving, :normal)
const BoxTypes = (EllipseTypes..., :delta)
function wc_statistics!(
    portfolio;
    box = :stationary,
    ellipse = :stationary,
    q = 0.05,
    n_sim = 3_000,
    window = 3,
    dmu = 0.1,
    dcov = 0.1,
    seed = nothing,
    rng = Random.default_rng(),
)
    @assert(box ∈ BoxTypes, "box must be one of $BoxTypes")
    @assert(ellipse ∈ EllipseTypes, "ellipse must be one of $EllipseTypes")

    returns = Matrix(portfolio.returns[!, 2:end])
    T, N = size(returns)

    if box == :normal || box == :delta || ellipse == :normal
        box == :delta && (mu = vec(mean(returns, dims = 1)))
        covariance = cov(returns)
    end

    if box == :stationary || box == :circular || box == :moving
        mus, covs = gen_bootstrap(returns, box, window, seed, rng)

        mu_l, mu_u, cov_l, cov_u = calc_lo_hi_mu_cov(mus, covs, q, n_sim)

        d_mu = (mu_u - mu_l) / 2
    elseif box == :normal
    end
end

export asset_statistics!