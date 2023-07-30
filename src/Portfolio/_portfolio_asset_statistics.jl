using Distributions, Random

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
    calc_box = true,
    calc_ellipse = true,
    q = 0.05,
    n_sim = 3_000,
    window = 3,
    dmu = 0.1,
    dcov = 0.1,
    n_samples = 10_000,
    seed = nothing,
    rng = Random.default_rng(),
)
    @assert(box ∈ BoxTypes, "box must be one of $BoxTypes")
    @assert(ellipse ∈ EllipseTypes, "ellipse must be one of $EllipseTypes")
    @assert(
        calc_box || calc_ellipse,
        "at least one of calc_box = $calc_box, or calc_ellipse = $calc_ellipse must be true"
    )

    returns = Matrix(portfolio.returns[!, 2:end])
    nms = names(portfolio.returns)[2:end]
    nms2 = vec(["$(i)-$(j)" for i in nms, j in nms])
    T, N = size(returns)

    if box == :normal || box == :delta || ellipse == :normal
        box == :delta && (mu = vec(mean(returns, dims = 1)))
        sigma = cov(returns)
    end

    if calc_box && (box == :stationary || box == :circular || box == :moving) ||
       calc_ellipse &&
       (ellipse == :stationary || ellipse == :circular || ellipse == :moving)
        mus, covs = gen_bootstrap(returns, box, window, seed, rng)
    end

    if calc_box
        if box == :stationary || box == :circular || box == :moving
            mu_l, mu_u, cov_l, cov_u = calc_lo_hi_mu_cov(mus, covs, q, n_sim)
            d_mu = (mu_u - mu_l) / 2
        elseif box == :normal
            !isnothing(seed) && Random.seed!(rng, seed)
            d_mu = cquantile(Normal(), q / 2) * sqrt.(diag(sigma) / T)
            A = vcat(# Vertically concatenate the vectors into an (n_samples x N^2) matrix.
                transpose.(# Transpose all vectors into row vectors.
                    vec.(# Turn all (N x N) matrices into vectors of length N^2.
                        rand(# Generate a vector of length n_samples where each entry is a Wishart matrix sampled from the sigma (N x N).
                            Wishart(T, sigma / T),
                            n_samples,
                        )#
                    ),#
                )...,# Splat the transposed vectors into vcat so they get concatenated into a matrix, else they'd be concatenated into a vector of length n_samples where each entry is a vector of length N^2.
            )#

            # Each column in A corresponds to an entry in the original Wishart matrix, each row corresponds to a sample. This effectively calculates quantiles accross samples for equivalent entries in the sampled Wishart matrices. We then reshape back into an N x N matrix.
            cov_l = reshape([quantile(A[:, i], q / 2) for i in 1:(N * N)], N, N)
            cov_u = reshape([quantile(A[:, i], 1 - q / 2) for i in 1:(N * N)], N, N)

            args = ()
            kwargs = (;)
            !isposdef(cov_l) && fix_cov!(cov_l, args..., kwargs...)
            !isposdef(cov_u) && fix_cov!(cov_u, args..., kwargs...)
        elseif box == :delta
            d_mu = dmu * abs.(mu)
            cov_l = sigma - dcov * abs.(sigma)
            cov_u = sigma + dcov * abs.(sigma)
        end
    end

    if calc_ellipse
        if ellipse == :stationary || ellipse == :circular || ellipse == :moving
            cov_mu, cov_sigma = calc_cov_mu_cov_sigma(mus, covs, q, n_sim)
        elseif ellipse == :normal
            cov_mu = Diagonal(sigma) / T
            K = commutation_matrix(sigma)
            cov_sigma = Diagonal((I + K) * kron(cov_mu, cov_mu)) * T
        end
    end

    cov_l, cov_u, d_mu = if calc_box
        (DataFrame(cov_l, nms), DataFrame(cov_u, nms), DataFrame(tickers = nms, val = d_mu))
    else
        (DataFrame(), DataFrame(), DataFrame())
    end

    cov_mu, cov_sigma = if calc_ellipse
        (DataFrame(Matrix(cov_mu), nms), DataFrame(Matrix(cov_sigma), nms2))
    else
        (DataFrame(), DataFrame())
    end

    k_mu = sqrt(cquantile(Chisq(N), q))
    k_sigma = sqrt(cquantile(Chisq(N * N), q))

    portfolio.cov_l = cov_l
    portfolio.cov_u = cov_u
    portfolio.cov_mu = cov_mu
    portfolio.cov_sigma = cov_sigma
    portfolio.d_mu = d_mu
    portfolio.k_mu = k_mu
    portfolio.k_sigma = k_sigma

    return nothing
end

export asset_statistics!, wc_statistics!