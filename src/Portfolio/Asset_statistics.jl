function cokurt(x::AbstractMatrix, mu::AbstractArray)
    T, N = size(x)
    y = x .- mu
    ex = eltype(y)
    o = transpose(range(start = one(ex), stop = one(ex), length = N))
    z = kron(o, y) .* kron(y, o)
    cokurt = transpose(z) * z / T
    return cokurt
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

function duplication_matrix(n::Int)
    cols = Int(n * (n + 1) / 2)
    rows = n * n
    mtx = spzeros(rows, cols)
    for j in 1:n
        for i in j:n
            u = spzeros(1, cols)
            col = Int((j - 1) * n + i - (j * (j - 1)) / 2)
            u[col] = 1
            T = spzeros(n, n)
            T[i, j] = 1
            T[j, i] = 1
            mtx .+= vec(T) * u
        end
    end
    return mtx
end

function elimination_matrix(n::Int)
    rows = Int(n * (n + 1) / 2)
    cols = n * n
    mtx = spzeros(rows, cols)
    for j in 1:n
        ej = spzeros(1, n)
        ej[j] = 1
        for i in j:n
            u = spzeros(rows)
            row = Int((j - 1) * n + i - (j * (j - 1)) / 2)
            u[row] = 1
            ei = spzeros(1, n)
            ei[i] = 1
            mtx .+= kron(u, kron(ej, ei))
        end
    end
    return mtx
end

function summation_matrix(n::Int)
    d = duplication_matrix(n)
    l = elimination_matrix(n)

    s = transpose(d) * d * l

    return s
end

function dup_elim_sum_matrices(n::Int)
    d = duplication_matrix(n)
    l = elimination_matrix(n)
    s = transpose(d) * d * l

    return d, l, s
end

"""
```julia
asset_statistics!(
    portfolio::AbstractPortfolio;
    target_ret::AbstractFloat = 0.0,
    mean_func::Function = mean,
    cov_func::Function = cov,
    cor_func::Function = cor,
    std_func = std,
    dist_func::Function = x -> sqrt.(clamp!((1 .- x) / 2, 0, 1)),
    codep_type::Symbol = isa(portfolio, HCPortfolio) ? portfolio.codep_type : :Pearson,
    custom_mu = nothing,
    custom_cov = nothing,
    custom_kurt = nothing,
    custom_skurt = nothing,
    mean_args = (),
    cov_args = (),
    cor_args = (),
    dist_args = (),
    std_args = (),
    calc_kurt = true,
    mean_kwargs = (; dims = 1),
    cov_kwargs = (;),
    cor_kwargs = (;),
    dist_kwargs = (;),
    std_kwargs = (;),
    uplo = :L,
)
```
"""
function asset_statistics!(
    portfolio::AbstractPortfolio;
    target_ret::AbstractFloat = 0.0,
    mean_func::Function = mean,
    cov_func::Function = cov,
    cor_func::Function = cor,
    std_func = std,
    dist_func::Function = x -> sqrt.(clamp!((1 .- x) / 2, 0, 1)),
    codep_type::Symbol = isa(portfolio, HCPortfolio) ? portfolio.codep_type : :Pearson,
    custom_mu = nothing,
    custom_cov = nothing,
    custom_kurt = nothing,
    custom_skurt = nothing,
    mean_args = (),
    cov_args = (),
    cor_args = (),
    dist_args = (),
    std_args = (),
    calc_kurt = true,
    mean_kwargs = (; dims = 1),
    cov_kwargs = (;),
    cor_kwargs = (;),
    dist_kwargs = (;),
    std_kwargs = (;),
    uplo = :L,
)
    returns = portfolio.returns
    N = size(returns, 2)

    portfolio.mu =
        isnothing(custom_mu) ? vec(mean_func(returns, mean_args...; mean_kwargs...)) :
        custom_mu

    portfolio.cov =
        isnothing(custom_cov) ? cov_func(returns, cov_args...; cov_kwargs...) : custom_cov

    if isa(portfolio, Portfolio)
        if calc_kurt
            portfolio.kurt =
                isnothing(custom_kurt) ? cokurt(returns, transpose(portfolio.mu)) :
                custom_kurt
            portfolio.skurt =
                isnothing(custom_skurt) ?
                scokurt(returns, transpose(portfolio.mu), target_ret) : custom_skurt
            missing, portfolio.L_2, portfolio.S_2 = dup_elim_sum_matrices(N)
        end
    else
        if codep_type == :Pearson
            codep = cor(returns)
            dist = sqrt.(clamp!((1 .- codep) / 2, 0, 1))
        elseif codep_type == :Spearman
            codep = corspearman(returns)
            dist = sqrt.(clamp!((1 .- codep) / 2, 0, 1))
        elseif codep_type == :Kendall
            codep = corkendall(returns)
            dist = sqrt.(clamp!((1 .- codep) / 2, 0, 1))
        elseif codep_type == :Abs_Pearson
            codep = abs.(cor(returns))
            dist = sqrt.(clamp!(1 .- codep, 0, 1))
        elseif codep_type == :Abs_Spearman
            codep = abs.(corspearman(returns))
            dist = sqrt.(clamp!(1 .- codep, 0, 1))
        elseif codep_type == :Abs_Kendall
            codep = abs.(corkendall(returns))
            dist = sqrt.(clamp!(1 .- codep, 0, 1))
        elseif codep_type == :Gerber1
            codep = cov_to_cor(
                covgerber1(
                    returns,
                    portfolio.gs_threshold;
                    std_func = std_func,
                    std_args = std_args,
                    std_kwargs = std_kwargs,
                ),
            )
            dist = sqrt.(clamp!((1 .- codep) / 2, 0, 1))
        elseif codep_type == :Gerber2
            codep = cov_to_cor(
                covgerber2(
                    returns,
                    portfolio.gs_threshold;
                    std_func = std_func,
                    std_args = std_args,
                    std_kwargs = std_kwargs,
                ),
            )
            dist = sqrt.(clamp!((1 .- codep) / 2, 0, 1))
        elseif codep_type == :Distance
            codep = cordistance(returns)
            dist = sqrt.(clamp!(1 .- codep, 0, 1))
        elseif codep_type == :Mutual_Info
            bins_info = portfolio.bins_info
            codep, dist = mut_var_info_mtx(returns, bins_info)
        elseif codep_type == :Tail
            codep = ltdi_mtx(returns, portfolio.alpha_tail)
            dist = -log.(codep)
        elseif codep_type == :Custom_Cov
            codep = cov_to_cor(portfolio.cov)
            dist = dist_func(codep, dist_args...; dist_kwargs...)
        elseif codep_type == :Custom_Cor
            codep = cor_func(returns, cor_args...; cor_kwargs...)
            dist = dist_func(codep, dist_args...; dist_kwargs...)
        end

        portfolio.dist = issymmetric(dist) ? dist : Symmetric(dist, uplo)
        portfolio.codep = issymmetric(codep) ? codep : Symmetric(codep, uplo)
    end

    return nothing
end

function commutation_matrix(x::AbstractMatrix)
    m, n = size(x)
    mn = m * n
    row = 1:mn
    col = vec(transpose(reshape(row, m, n)))
    data = range(start = 1, stop = 1, length = mn)
    com = sparse(row, col, data, mn, mn)
    return com
end

function gen_bootstrap(
    returns,
    kind = :Stationary,
    n_sim = 3_000,
    window = 3,
    seed = nothing,
    rng = Random.default_rng(),
)
    @assert(kind ∈ KindBootstrap, "kind must be one of $KindBootstrap")
    !isnothing(seed) && Random.seed!(rng, seed)

    mus = Vector{Vector{eltype(returns)}}(undef, n_sim)
    covs = Vector{Matrix{eltype(returns)}}(undef, n_sim)

    bootstrap_func = if kind == :Stationary
        pyimport("arch.bootstrap").StationaryBootstrap
    elseif kind == :Circular
        pyimport("arch.bootstrap").CircularBlockBootstrap
    elseif kind == :Moving
        pyimport("arch.bootstrap").MovingBlockBootstrap
    end

    gen = bootstrap_func(window, returns, seed = seed)
    for (i, data) in enumerate(gen.bootstrap(n_sim))
        A = data[1][1]
        mus[i] = vec(mean(A, dims = 1))
        covs[i] = cov(A)
    end

    return mus, covs
end

function vec_of_vecs_to_mtx(x::AbstractVector{<:AbstractVector})
    return vcat(transpose.(x)...)
end

"""
```julia
wc_statistics!(
    portfolio;
    box = :Stationary,
    ellipse = :Stationary,
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
```
Worst case optimisation statistics.
"""
function wc_statistics!(
    portfolio;
    box = :Stationary,
    ellipse = :Stationary,
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

    if box == :Delta || ellipse == :Stationary || ellipse == :Circular || ellipse == :Moving
        mu = vec(mean(returns, dims = 1))
    end

    if calc_ellipse || box == :Normal || box == :Delta
        sigma = cov(returns)
    end

    if calc_box
        if box == :Stationary || box == :Circular || box == :Moving
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
        elseif box == :Normal
            !isnothing(seed) && Random.seed!(rng, seed)
            d_mu = cquantile(Normal(), q / 2) * sqrt.(diag(sigma) / T)
            cov_s = vec_of_vecs_to_mtx(vec.(rand(Wishart(T, sigma / T), n_samples)))

            cov_l = reshape([quantile(cov_s[:, i], q / 2) for i in 1:(N * N)], N, N)
            cov_u = reshape([quantile(cov_s[:, i], 1 - q / 2) for i in 1:(N * N)], N, N)
            args = ()
            kwargs = (;)
            !isposdef(cov_l) && fix_cov!(cov_l, args..., kwargs...)
            !isposdef(cov_u) && fix_cov!(cov_u, args..., kwargs...)
        elseif box == :Delta
            d_mu = dmu * abs.(mu)
            cov_l = sigma - dcov * abs.(sigma)
            cov_u = sigma + dcov * abs.(sigma)
        end
    end

    if calc_ellipse
        if ellipse == :Stationary || ellipse == :Circular || ellipse == :Moving
            mus, covs = gen_bootstrap(returns, ellipse, n_sim, window, seed, rng)

            cov_mu = Diagonal(cov(vec_of_vecs_to_mtx([mu_s .- mu for mu_s in mus])))
            cov_sigma = Diagonal(
                cov(vec_of_vecs_to_mtx([vec(cov_s) .- vec(sigma) for cov_s in covs])),
            )
        elseif ellipse == :Normal
            cov_mu = Diagonal(sigma) / T
            K = commutation_matrix(sigma)
            cov_sigma = T * Diagonal((I + K) * kron(cov_mu, cov_mu))
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
