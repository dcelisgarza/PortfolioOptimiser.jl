function cokurt(x::AbstractMatrix, mean_func::Function = mean, args...; kwargs...)
    T, N = size(x)
    mu =
        !haskey(kwargs, :dims) ? mean_func(x, args...; dims = 1, kwargs...) :
        mean_func(x, args...; kwargs...)
    y = x .- mu
    ex = eltype(y)
    o = transpose(range(start = one(ex), stop = one(ex), length = N))
    z = kron(o, y) .* kron(y, o)
    cokurt = transpose(z) * z / T
    return cokurt
end

function cokurt(x::AbstractMatrix, mu::AbstractArray)
    T, N = size(x)
    y = x .- mu
    ex = eltype(y)
    o = transpose(range(start = one(ex), stop = one(ex), length = N))
    z = kron(o, y) .* kron(y, o)
    cokurt = transpose(z) * z / T
    return cokurt
end

function scokurt(
    x::AbstractMatrix,
    mean_func::Function = mean,
    target_ret::AbstractFloat = 0.0,
    args...;
    kwargs...,
)
    T, N = size(x)
    mu =
        !haskey(kwargs, :dims) ? mean_func(x, args...; dims = 1, kwargs...) :
        mean_func(x, args...; kwargs...)
    y = x .- mu
    y .= min.(y, target_ret)
    ex = eltype(y)
    o = transpose(range(start = one(ex), stop = one(ex), length = N))
    z = kron(o, y) .* kron(y, o)
    scokurt = transpose(z) * z / T
    return scokurt
end

function scokurt(x::AbstractMatrix, mu::AbstractArray, target_ret::AbstractFloat = 0.0)
    T, N = size(x)
    y = x .- mu
    y .= min.(y, target_ret)
    ex = eltype(y)
    o = transpose(range(start = one(ex), stop = one(ex), length = N))
    z = kron(o, y) .* kron(y, o)
    scokurt = transpose(z) * z / T
    return scokurt
end

function fix_cov!(covariance, args...; kwargs...)
    println("IMPLEMENT ME")
end

function asset_statistics!(
    portfolio::AbstractPortfolio;
    target_ret::AbstractFloat = 0.0,
    mean_func::Function = mean,
    cov_func::Function = cov,
    cor_dist_func::Function = cor,
    custom_mu = nothing,
    custom_cov = nothing,
    custom_kurt = nothing,
    custom_skurt = nothing,
    mean_args = (),
    cov_args = (),
    cor_dist_args = (),
    calc_kurt = true,
    mean_kwargs = (;),
    cov_kwargs = (;),
    cor_dist_kwargs = (;),
    uplo = :L,
)
    returns = portfolio.returns
    N = size(returns, 2)

    portfolio.mu =
        mu =
            isnothing(custom_mu) ?
            vec(
                !haskey(mean_kwargs, :dims) ?
                mean_func(returns, mean_args...; dims = 1, mean_kwargs...) :
                mean_func(returns, mean_args...; mean_kwargs...),
            ) : custom_mu

    portfolio.cov =
        isnothing(custom_cov) ? cov_func(returns, cov_args...; cov_kwargs...) : custom_cov

    if isa(portfolio, Portfolio)
        if calc_kurt
            portfolio.kurt =
                isnothing(custom_kurt) ? cokurt(returns, transpose(mu)) : custom_kurt
            portfolio.skurt =
                isnothing(custom_skurt) ? cokurt(returns, transpose(mu), target_ret) :
                custom_skurt
            missing, portfolio.L_2, portfolio.S_2 = dup_elim_sum_matrices(N)
        end
    else
        codep_type = portfolio.codep_type

        if codep_type == :pearson
            codep = cor(returns)
            dist = sqrt.(clamp!((1 .- codep) / 2, 0, 1))
        elseif codep_type == :spearman
            codep = corspearman(returns)
            dist = sqrt.(clamp!((1 .- codep) / 2, 0, 1))
        elseif codep_type == :kendall
            codep = corkendall(returns)
            dist = sqrt.(clamp!((1 .- codep) / 2, 0, 1))
        elseif codep_type == :abs_pearson
            codep = abs.(cor(returns))
            dist = sqrt.(clamp!(1 .- codep, 0, 1))
        elseif codep_type == :abs_spearman
            codep = abs.(corspearman(returns))
            dist = sqrt.(clamp!(1 .- codep, 0, 1))
        elseif codep_type == :abs_kendall
            codep = abs.(corkendall(returns))
            dist = sqrt.(clamp!(1 .- codep, 0, 1))
        elseif codep_type == :gerber1
            codep = covgerber1(returns, portfolio.gs_threshold)
            dist = sqrt.(clamp!((1 .- codep) / 2, 0, 1))
        elseif codep_type == :gerber2
            codep = covgerber2(returns, portfolio.gs_threshold)
            dist = sqrt.(clamp!((1 .- codep) / 2, 0, 1))
        elseif codep_type == :distance
            codep = cordistance(returns)
            dist = sqrt.(clamp!(1 .- codep, 0, 1))
        elseif codep_type == :mutual_info
            bins_info = portfolio.bins_info
            codep = info_mtx(returns, bins_info, :mutual)
            dist = info_mtx(returns, bins_info, :variation)
        elseif codep_type == :tail
            codep = ltdi_mtx(returns, portfolio.alpha_tail)
            dist = -log.(codep)
        elseif codep_type == :custom_cov
            codep = cov2cor(portfolio.cov)
            dist = sqrt.(clamp!((1 .- codep) / 2, 0, 1))
        elseif codep_type == :custom_cor
            codep, dist = cor_dist_func(portfolio, cor_dist_args...; cor_dist_kwargs...)
        end

        portfolio.dist = issymmetric(dist) ? dist : Symmetric(dist, uplo)
        portfolio.codep = issymmetric(codep) ? codep : Symmetric(codep, uplo)
    end

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
    fix_cov_args = (),
    fix_cov_kwargs = (;),
)
    @assert(box ∈ BoxTypes, "box must be one of $BoxTypes")
    @assert(ellipse ∈ EllipseTypes, "ellipse must be one of $EllipseTypes")
    @assert(
        calc_box || calc_ellipse,
        "at least one of calc_box = $calc_box, or calc_ellipse = $calc_ellipse must be true"
    )

    returns = portfolio.returns
    T, N = size(returns)

    if box == :normal || box == :delta || ellipse == :normal
        box == :delta && (mu = vec(mean(returns, dims = 1)))
        sigma = cov(returns)
    end

    if calc_box
        if box == :stationary || box == :circular || box == :moving
            mus, covs = gen_bootstrap(returns, box, n_sim, window, seed, rng)

            mu_s = vec_of_vecs_to_mtx(mus)
            mu_l = [quantile(mu_s[:, i], q / 2) for i in 1:N]
            mu_u = [quantile(mu_s[:, i], 1 - q / 2) for i in 1:N]

            cov_s = vec_of_vecs_to_mtx(vec.(covs))
            cov_l = reshape([quantile(cov_s[:, i], q / 2) for i in 1:(N * N)], N, N)
            cov_u = reshape([quantile(cov_s[:, i], 1 - q / 2) for i in 1:(N * N)], N, N)

            args = ()
            kwargs = (;)
            !isposdef(cov_l) && fix_cov!(cov_l, fix_cov_args...; fix_cov_kwargs...)
            !isposdef(cov_u) && fix_cov!(cov_u, fix_cov_args...; fix_cov_kwargs...)

            d_mu = (mu_u - mu_l) / 2
        elseif box == :normal
            !isnothing(seed) && Random.seed!(rng, seed)
            d_mu = cquantile(Normal(), q / 2) * sqrt.(diag(sigma) / T)
            cov_s = vec_of_vecs_to_mtx(vec.(rand(Wishart(T, sigma / T), n_samples)))

            cov_l = reshape([quantile(cov_s[:, i], q / 2) for i in 1:(N * N)], N, N)
            cov_u = reshape([quantile(cov_s[:, i], 1 - q / 2) for i in 1:(N * N)], N, N)
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
            mus, covs = gen_bootstrap(returns, ellipse, n_sim, window, seed, rng)

            cov_mu = Diagonal(cov(vec_of_vecs_to_mtx([mu_s .- mu for mu_s in mus])))
            cov_sigma = Diagonal(
                cov(vec_of_vecs_to_mtx([vec(cov_s) .- vec(cov) for cov_s in covs])),
            )
        elseif ellipse == :normal
            cov_mu = Diagonal(sigma) / T
            K = commutation_matrix(sigma)
            cov_sigma = Diagonal((I + K) * kron(cov_mu, cov_mu)) * T
        end
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

export cokurt, scokurt, asset_statistics!, wc_statistics!, fix_cov!