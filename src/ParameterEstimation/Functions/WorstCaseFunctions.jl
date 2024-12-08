function _bootstrap_func(::StationaryBS, block_size, X, seed)
    return pyimport("arch.bootstrap").StationaryBootstrap(block_size, X; seed = seed)
end
function _bootstrap_func(::CircularBS, block_size, X, seed)
    return pyimport("arch.bootstrap").CircularBlockBootstrap(block_size, X; seed = seed)
end
function _bootstrap_func(::MovingBS, block_size, X, seed)
    return pyimport("arch.bootstrap").MovingBlockBootstrap(block_size, X; seed = seed)
end
function bootstrap_generator(method::ArchWC, X)
    return _bootstrap_func(method.bootstrap, method.block_size, Py(X).to_numpy(),
                           method.seed)
end
function _sigma_mu(X::AbstractMatrix, cov_type::PortfolioOptimiserCovCor,
                   mu_type::MeanEstimator)
    sigma = Matrix(cov(cov_type, X))
    old_sigma = set_mean_sigma(mu_type, sigma)
    mu = vec(mean(mu_type, X))
    unset_mean_sigma(mu_type, old_sigma)

    return sigma, mu
end
function gen_bootstrap(method::ArchWC, cov_type::PortfolioOptimiserCovCor,
                       mu_type::MeanEstimator, X::AbstractMatrix)
    covs = Vector{Matrix{eltype(X)}}(undef, 0)
    sizehint!(covs, method.n_sim)
    mus = Vector{Vector{eltype(X)}}(undef, 0)
    sizehint!(mus, method.n_sim)

    gen = bootstrap_generator(method, X)
    for data ∈ gen.bootstrap(method.n_sim)
        A = pyconvert(Array, data)[1][1]
        sigma, mu = _sigma_mu(A, cov_type, mu_type)
        push!(covs, sigma)
        push!(mus, mu)
    end

    return covs, mus
end
function vec_of_vecs_to_mtx(x::AbstractVector{<:AbstractArray})
    return vcat(transpose.(x)...)
end
function calc_sets(::Box, method::ArchWC, cov_type::PortfolioOptimiserCovCor,
                   mu_type::MeanEstimator, X::AbstractMatrix, ::Any, ::Any)
    q = method.q
    N = size(X, 2)

    covs, mus = gen_bootstrap(method, cov_type, mu_type, X)

    cov_s = vec_of_vecs_to_mtx(vec.(covs))
    cov_l = reshape([quantile(cov_s[:, i], q / 2) for i ∈ 1:(N * N)], N, N)
    cov_u = reshape([quantile(cov_s[:, i], 1 - q / 2) for i ∈ 1:(N * N)], N, N)

    mu_s = vec_of_vecs_to_mtx(mus)
    mu_l = [quantile(mu_s[:, i], q / 2) for i ∈ 1:N]
    mu_u = [quantile(mu_s[:, i], 1 - q / 2) for i ∈ 1:N]
    d_mu = (mu_u - mu_l) / 2

    return cov_l, cov_u, d_mu, nothing, nothing
end
function calc_sets(::Ellipse, method::ArchWC, cov_type::PortfolioOptimiserCovCor,
                   mu_type::MeanEstimator, X::AbstractMatrix, sigma::AbstractMatrix,
                   mu::AbstractVector, ::Any, ::Any)
    covs, mus = gen_bootstrap(method, cov_type, mu_type, X)

    A_sigma = vec_of_vecs_to_mtx([vec(cov_s) .- vec(sigma) for cov_s ∈ covs])
    cov_sigma = Matrix(cov(cov_type, A_sigma))

    A_mu = vec_of_vecs_to_mtx([mu_s .- mu for mu_s ∈ mus])
    cov_mu = Matrix(cov(cov_type, A_mu))

    return cov_sigma, cov_mu, A_sigma, A_mu
end
function calc_sets(::Box, method::NormalWC, ::Any, ::Any, X::AbstractMatrix,
                   sigma::AbstractMatrix, ::Any)
    Random.seed!(method.rng, method.seed)
    q = method.q
    T, N = size(X)

    cov_mu = sigma / T

    covs = vec_of_vecs_to_mtx(vec.(rand(Wishart(T, cov_mu), method.n_sim)))
    cov_l = reshape([quantile(covs[:, i], q / 2) for i ∈ 1:(N * N)], N, N)
    cov_u = reshape([quantile(covs[:, i], 1 - q / 2) for i ∈ 1:(N * N)], N, N)

    d_mu = cquantile(Normal(), q / 2) * sqrt.(diag(cov_mu))

    return cov_l, cov_u, d_mu, covs, cov_mu
end
function commutation_matrix(x::AbstractMatrix)
    m, n = size(x)
    mn = m * n
    row = 1:mn
    col = vec(transpose(reshape(row, m, n)))
    data = range(; start = 1, stop = 1, length = mn)
    com = sparse(row, col, data, mn, mn)
    return com
end
function calc_sets(::Ellipse, method::NormalWC, cov_type::PortfolioOptimiserCovCor,
                   mu_type::MeanEstimator, X::AbstractMatrix, sigma::AbstractMatrix,
                   mu::AbstractVector, covs::Union{AbstractMatrix, Nothing},
                   cov_mu::Union{AbstractMatrix, Nothing})
    Random.seed!(method.rng, method.seed)
    T = size(X, 1)

    A_mu = transpose(rand(MvNormal(mu, sigma), method.n_sim))
    if isnothing(covs) || isnothing(cov_mu)
        cov_mu = sigma / T
        covs = vec_of_vecs_to_mtx(vec.(rand(Wishart(T, cov_mu), method.n_sim)))
    end
    A_sigma = covs .- transpose(vec(sigma))

    K = commutation_matrix(sigma)
    cov_sigma = T * (I + K) * kron(cov_mu, cov_mu)
    return cov_sigma, cov_mu, A_sigma, A_mu
end
function calc_sets(::Box, method::DeltaWC, ::Any, ::Any, X::AbstractMatrix,
                   sigma::AbstractMatrix, mu::AbstractVector)
    d_mu = method.dmu * abs.(mu)
    cov_l = sigma - method.dcov * abs.(sigma)
    cov_u = sigma + method.dcov * abs.(sigma)

    return cov_l, cov_u, d_mu, nothing, nothing
end
function calc_k_wc(::KNormalWC, q::Real, X::AbstractMatrix, cov_X::AbstractMatrix)
    k_mus = diag(X * (cov_X \ I) * transpose(X))
    return sqrt(quantile(k_mus, 1 - q))
end
function calc_k_wc(::KGeneralWC, q::Real, args...)
    return sqrt((1 - q) / q)
end
function calc_k_wc(method::Real, args...)
    return method
end

export gen_bootstrap, vec_of_vecs_to_mtx, calc_sets, commutation_matrix, calc_k_wc
