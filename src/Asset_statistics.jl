function cokurt(x::AbstractMatrix, mu::AbstractArray)
    T, N = size(x)
    y = x .- mu
    ex = eltype(y)
    o = transpose(range(start = one(ex), stop = one(ex), length = N))
    z = kron(o, y) .* kron(y, o)
    cokurt = transpose(z) * z / T
    return cokurt
end

function scokurt(x::AbstractMatrix, mu::AbstractArray, target_ret::Real = 0.0)
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

function nearest_cov(mtx::AbstractMatrix, method = NearestCorrelationMatrix.Newton())
    s = sqrt.(diag(mtx))
    corr = cov2cor(mtx)
    NearestCorrelationMatrix.nearest_cor!(corr, method)
    nmtx = cor2cov(corr, s)

    return any(!isfinite.(nmtx)) ? nmtx : mtx
end

function posdef_fix!(
    mtx::AbstractMatrix,
    posdef_fix::Symbol,
    posdef_func::Function,
    posdef_args::Tuple,
    posdef_kwargs::NamedTuple,
)
    posdef_fix == :None || isposdef(mtx) && return nothing

    @assert(
        posdef_fix ∈ PosdefFixes,
        "posdef_fix = $posdef_fix, must be one of $PosdefFixes"
    )

    if posdef_fix == :Nearest
        mtx .= nearest_cov(mtx, posdef_args...; posdef_kwargs...)
    elseif posdef_fix == :Custom_Func
        mtx .= posdef_func(mtx, posdef_args...; posdef_kwargs...)
    end

    !isposdef(mtx) && @warn(
        "matrix could not be made postive definite, please try a different method or a tighter tolerance"
    )

    return nothing
end
const ASH = AverageShiftedHistograms
function errPDF(x, vals; kernel = ASH.Kernels.gaussian, m = 10, n = 1000, q = 1000)
    e_min, e_max = x * (1 - sqrt(1.0 / q))^2, x * (1 + sqrt(1.0 / q))^2
    rg = range(e_min, e_max, length = n)
    pdf1 = q ./ (2 * pi * x * rg) .* sqrt.(clamp.((e_max .- rg) .* (rg .- e_min), 0, Inf))

    e_min, e_max = x * (1 - sqrt(1.0 / q))^2, x * (1 + sqrt(1.0 / q))^2
    res = ash(vals; rng = range(e_min, e_max, length = n), kernel = kernel, m = m)
    pdf2 = [ASH.pdf(res, i) for i in pdf1]
    pdf2[.!isfinite.(pdf2)] .= 0.0
    sse = sum((pdf2 - pdf1) .^ 2)

    return sse
end

function find_max_eval(
    vals,
    q;
    kernel = ASH.Kernels.gaussian,
    m::Integer = 10,
    n::Integer = 1000,
    opt_args = (),
    opt_kwargs = (;),
)
    res = Optim.optimize(
        x -> errPDF(x, vals; kernel = kernel, m = m, n = n, q = q),
        0.0,
        1.0,
        opt_args...;
        opt_kwargs...,
    )

    x = Optim.converged(res) ? Optim.minimizer(res) : 1.0

    e_max = x * (1.0 + sqrt(1.0 / q))^2

    return e_max, x
end
export find_max_eval

function denoise_cov(
    mtx::AbstractMatrix,
    q::Real,
    method::Symbol = :Fixed;
    alpha::Real = 0.0,
    detone::Bool = false,
    kernel = ASH.Kernels.gaussian,
    m::Integer = 10,
    n::Integer = 1000,
    mkt_comp::Integer = 1,
    opt_args = (),
    opt_kwargs = (;),
)
    corr = cov2cor(mtx)
    s = sqrt.(diag(mtx))

    vals, vecs = eigen(corr)
    idx = sortperm(vals)
    vals .= vals[idx]
    vecs .= vecs[:, idx]

    find_max_eval(vals, q; kernel = kernel, m = m, n = n, opt_args = (), opt_kwargs = (;))
end

function mu_esimator(
    returns::AbstractMatrix,
    mu_type::Symbol = :JS,
    target::Symbol = :GM;
    dims::Integer = 1,
    mu_weights::Union{AbstractWeights, Nothing} = nothing,
    sigma::AbstractMatrix = cov(returns),
)
    @assert(
        mu_type ∈ (:JS, :BS, :BOP),
        "mu_type = $mu_type, must be one of (:JS, :BS, :BOP)"
    )
    @assert(target ∈ MuTargets, "target = $target, must be one of $MuTargets")

    T, N = size(returns)
    mu =
        isnothing(mu_weights) ? vec(mean(returns; dims = dims)) :
        vec(mean(returns, mu_weights; dims = dims))

    inv_sigma = inv(sigma)
    evals = eigvals(sigma)
    ones = range(1, stop = 1, length = N)

    b = if target == :GM
        fill(mean(mu), N)
    elseif target == :VW
        fill(dot(ones, inv_sigma, mu) / dot(ones, inv_sigma, ones), N)
    else
        fill(tr(sigma) / T, N)
    end

    if mu_type == :JS
        alpha = (N * mean(evals) - 2 * maximum(evals)) / dot(mu - b, mu - b) / T
        mu = (1 - alpha) * mu + alpha * b
    elseif mu_type == :BS
        alpha = (N + 2) / ((N + 2) + T * dot(mu - b, inv_sigma, mu - b))
        mu = (1 - alpha) * mu + alpha * b
    else
        alpha =
            (dot(mu, inv_sigma, mu) - N / (T - N)) * dot(b, inv_sigma, b) -
            dot(mu, inv_sigma, b)^2
        alpha /= dot(mu, inv_sigma, mu) * dot(b, inv_sigma, b) - dot(mu, inv_sigma, b)^2
        beta = (1 - alpha) * dot(mu, inv_sigma, b) / dot(mu, inv_sigma, mu)
        mu = alpha * mu + beta * b
    end

    return mu
end

function covar_mtx(
    returns::AbstractMatrix;
    cov_args::Tuple = (),
    cov_est::CovarianceEstimator = StatsBase.SimpleCovariance(; corrected = true),
    cov_func::Function = cov,
    cov_kwargs::NamedTuple = (;),
    cov_type::Symbol = :Full,
    cov_weights::Union{AbstractWeights, Nothing} = nothing,
    custom_cov::Union{AbstractMatrix, Nothing} = nothing,
    gs_threshold::Real = 0.5,
    jlogo::Bool = false,
    posdef_args::Tuple = (),
    posdef_fix::Symbol = :None,
    posdef_func::Function = x -> x,
    posdef_kwargs::NamedTuple = (;),
    std_args::Tuple = (),
    std_func::Function = std,
    std_kwargs::NamedTuple = (;),
    target_ret::Union{Real, Vector{<:Real}} = 0.0,
)
    @assert(cov_type ∈ CovTypes, "cov_type = $cov_type, must be one of $CovTypes")
    cov_mtx = if cov_type == :Full
        isnothing(cov_weights) ? StatsBase.cov(cov_est, returns; cov_kwargs...) :
        StatsBase.cov(cov_est, returns, cov_weights; cov_kwargs...)
    elseif cov_type == :Semi
        semi_returns =
            isa(target_ret, Real) ? min.(returns .- target_ret, 0) :
            min.(returns .- transpose(target_ret), 0)
        isnothing(cov_weights) ?
        StatsBase.cov(cov_est, semi_returns; mean = 0.0, cov_kwargs...) :
        StatsBase.cov(cov_est, semi_returns, cov_weights; mean = 0.0, cov_kwargs...)
    elseif cov_type == :Gerber1
        covgerber1(
            returns,
            gs_threshold;
            std_func = std_func,
            std_args = std_args,
            std_kwargs = std_kwargs,
        )
    elseif cov_type == :Gerber2
        covgerber2(
            returns,
            gs_threshold;
            std_func = std_func,
            std_args = std_args,
            std_kwargs = std_kwargs,
        )
    elseif cov_type == :Custom_Func
        cov_func(returns, cov_args...; cov_kwargs...)
    elseif cov_type == :Custom_Val
        custom_cov
    end

    if jlogo
        codep = cov2cor(cov_mtx)
        dist = sqrt.(clamp!((1 .- codep) / 2, 0, 1))
        separators, cliques = PMFG_T2s(1 .- dist .^ 2, 4)[3:4]
        cov_mtx .= inv(JLogo(cov_mtx, separators, cliques))
    end

    posdef_fix!(cov_mtx, posdef_fix, posdef_func, posdef_args, posdef_kwargs)

    return cov_mtx
end

function mean_vec(
    returns::AbstractMatrix;
    custom_mu::Union{AbstractVector, Nothing} = nothing,
    mean_args::Tuple = (),
    mean_func::Function = mean,
    mean_kwargs::NamedTuple = (;),
    mu_target::Symbol = :GM,
    mu_type::Symbol = :Default,
    mu_weights::Union{AbstractWeights, Nothing} = nothing,
    sigma::Union{AbstractMatrix, Nothing} = nothing,
)
    @assert(mu_type ∈ MuTypes, "mu_type = $mu_type, must be one of $MuTypes")
    mu = if mu_type == :Default
        isnothing(mu_weights) ? vec(mean(returns; dims = 1)) :
        vec(mean(returns, mu_weights; dims = 1))
    elseif mu_type ∈ (:JS, :BS, :BOP)
        mu_esimator(
            returns,
            mu_type,
            mu_target;
            dims = 1,
            mu_weights = mu_weights,
            sigma = sigma,
        )
    elseif mu_type == :Custom_Func
        vec(mean_func(returns, mean_args...; mean_kwargs...))
    elseif mu_type == :Custom_Val
        custom_mu
    end

    return mu
end

function cokurt_mtx(
    returns::AbstractMatrix,
    mu::AbstractVector;
    custom_kurt::Union{AbstractMatrix, Nothing} = nothing,
    custom_skurt::Union{AbstractMatrix, Nothing} = nothing,
    posdef_args::Tuple = (),
    posdef_fix::Symbol = :None,
    posdef_func::Function = x -> x,
    posdef_kwargs::NamedTuple = (;),
    target_ret::Union{Real, Vector{<:Real}} = 0.0,
)
    kurt = isnothing(custom_kurt) ? cokurt(returns, transpose(mu)) : custom_kurt
    posdef_fix!(kurt, posdef_fix, posdef_func, posdef_args, posdef_kwargs)

    skurt =
        isnothing(custom_skurt) ? scokurt(returns, transpose(mu), target_ret) : custom_skurt
    posdef_fix!(skurt, posdef_fix, posdef_func, posdef_args, posdef_kwargs)

    N = length(mu)
    missing, L_2, S_2 = dup_elim_sum_matrices(N)

    return kurt, skurt, L_2, S_2
end

function codep_dist_mtx(
    returns::AbstractMatrix;
    alpha_tail::Real = 0.05,
    bins_info::Union{Symbol, Integer} = :KN,
    codep_type::Symbol = :Pearson,
    cor_args::Tuple = (),
    cor_func::Function = cor,
    cor_kwargs::NamedTuple = (;),
    custom_cor::Union{AbstractMatrix, Nothing} = nothing,
    dist_args::Tuple = (),
    dist_func::Function = x -> sqrt.(clamp!((1 .- x) / 2, 0, 1)),
    dist_kwargs::NamedTuple = (;),
    gs_threshold::Real = 0.5,
    sigma::Union{AbstractMatrix, Nothing} = nothing,
    std_args::Tuple = (),
    std_func::Function = std,
    std_kwargs::NamedTuple = (;),
    uplo::Symbol = :L,
)
    @assert(codep_type ∈ CodepTypes, "codep_type = $codep_type, must be one of $CodepTypes")
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
        codep = cov2cor(
            covgerber1(
                returns,
                gs_threshold;
                std_func = std_func,
                std_args = std_args,
                std_kwargs = std_kwargs,
            ),
        )
        dist = sqrt.(clamp!((1 .- codep) / 2, 0, 1))
    elseif codep_type == :Gerber2
        codep = cov2cor(
            covgerber2(
                returns,
                gs_threshold;
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
        codep, dist = mut_var_info_mtx(returns, bins_info)
    elseif codep_type == :Tail
        codep = ltdi_mtx(returns, alpha_tail)
        dist = -log.(codep)

    elseif codep_type == :Cov_to_Cor
        !isa(sigma, Matrix) &&
            size(sigma, 1) == size(sigma, 2) &&
            throw(
                AssertionError(
                    "sigma = $sigma, size(sigma) = $(size(sigma)), must be a square matrix",
                ),
            )
        codep = cov2cor(sigma)
        dist = dist_func(codep, dist_args...; dist_kwargs...)
    elseif codep_type == :Custom_Func
        codep = cor_func(returns, cor_args...; cor_kwargs...)
        dist = dist_func(codep, dist_args...; dist_kwargs...)
    elseif codep_type == :Custom_Val
        !isa(custom_cor, Matrix) &&
            size(custom_cor, 1) == size(custom_cor, 2) &&
            throw(
                AssertionError(
                    "custom_cor = $custom_cor, size(custom_cor) = $(size(custom_cor)), must be a square matrix",
                ),
            )
        codep = custom_cor
        dist = dist_func(codep, dist_args...; dist_kwargs...)
    end

    codep = issymmetric(codep) ? codep : Symmetric(codep, uplo)
    dist = issymmetric(dist) ? dist : Symmetric(dist, uplo)

    return codep, dist
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
    mean_args::Tuple = (),
    cov_args::Tuple = (),
    cor_args::Tuple = (),
    dist_args::Tuple = (),
    std_args::Tuple = (),
    calc_kurt = true,
    mean_kwargs = (; dims = 1),
    cov_kwargs::NamedTuple = (;),
    cor_kwargs::NamedTuple = (;),
    dist_kwargs::NamedTuple = (;),
    std_kwargs::NamedTuple = (;),
    uplo = :L,
)
```
"""
function asset_statistics!(
    portfolio::AbstractPortfolio;
    # flags
    calc_codep::Bool = true,
    calc_cov::Bool = true,
    calc_mu::Bool = true,
    calc_kurt::Bool = true,
    # cov_mtx
    cov_args::Tuple = (),
    cov_est::CovarianceEstimator = StatsBase.SimpleCovariance(; corrected = true),
    cov_func::Function = cov,
    cov_kwargs::NamedTuple = (;),
    cov_type::Symbol = portfolio.cov_type,
    cov_weights::Union{AbstractWeights, Nothing} = nothing,
    custom_cov::Union{AbstractVector, Nothing} = nothing,
    gs_threshold::Real = portfolio.gs_threshold,
    jlogo::Bool = portfolio.jlogo,
    posdef_args::Tuple = (),
    posdef_fix::Symbol = portfolio.posdef_fix,
    posdef_func::Function = x -> x,
    posdef_kwargs::NamedTuple = (;),
    std_args::Tuple = (),
    std_func::Function = std,
    std_kwargs::NamedTuple = (;),
    target_ret::Union{Real, Vector{<:Real}} = 0.0,
    # mean_vec
    custom_mu::Union{AbstractVector, Nothing} = nothing,
    mean_args::Tuple = (),
    mean_func::Function = mean,
    mean_kwargs::NamedTuple = (;),
    mu_target::Symbol = :GM,
    mu_type::Symbol = portfolio.mu_type,
    mu_weights::Union{AbstractWeights, Nothing} = nothing,
    # codep_dist_mtx
    alpha_tail::Union{Real, Nothing} = isa(portfolio, HCPortfolio) ? portfolio.alpha_tail :
                                       nothing,
    bins_info::Union{Symbol, Integer, Nothing} = isa(portfolio, HCPortfolio) ?
                                                 portfolio.bins_info : nothing,
    codep_type::Union{Symbol, Nothing} = isa(portfolio, HCPortfolio) ?
                                         portfolio.codep_type : nothing,
    cor_args::Tuple = (),
    cor_func::Function = cor,
    cor_kwargs::NamedTuple = (;),
    custom_cor::Union{AbstractMatrix, Nothing} = nothing,
    dist_args::Tuple = (),
    dist_func::Function = x -> sqrt.(clamp!((1 .- x) / 2, 0, 1)),
    dist_kwargs::NamedTuple = (;),
    custom_kurt::Union{AbstractMatrix, Nothing} = nothing,
    custom_skurt::Union{AbstractMatrix, Nothing} = nothing,
    uplo::Symbol = :L,
)
    returns = portfolio.returns

    # Covariance
    if calc_cov
        portfolio.cov = covar_mtx(
            returns;
            cov_args = cov_args,
            cov_est = cov_est,
            cov_func = cov_func,
            cov_kwargs = cov_kwargs,
            cov_type = cov_type,
            cov_weights = cov_weights,
            custom_cov = custom_cov,
            gs_threshold = gs_threshold,
            jlogo = jlogo,
            posdef_args = posdef_args,
            posdef_fix = posdef_fix,
            posdef_func = posdef_func,
            posdef_kwargs = posdef_kwargs,
            std_args = std_args,
            std_func = std_func,
            std_kwargs = std_kwargs,
            target_ret = target_ret,
        )

        portfolio.cov_type = cov_type

        (cov_type == :Gerber1 || cov_type == :Gerber2) &&
            (portfolio.gs_threshold = gs_threshold)

        jlogo && (portfolio.jlogo = jlogo)
    end

    # Mu
    if calc_mu
        portfolio.mu = mean_vec(
            returns;
            custom_mu = custom_mu,
            mean_args = mean_args,
            mean_func = mean_func,
            mean_kwargs = mean_kwargs,
            mu_target = mu_target,
            mu_type = mu_type,
            mu_weights = mu_weights,
            sigma = isnothing(custom_cov) ? portfolio.cov : custom_cov,
        )

        portfolio.mu_type = mu_type
    end

    # Type specific
    if isa(portfolio, Portfolio)
        if calc_kurt
            portfolio.kurt, portfolio.skurt, portfolio.L_2, portfolio.S_2 = cokurt_mtx(
                returns,
                portfolio.mu;
                custom_kurt = custom_kurt,
                custom_skurt = custom_skurt,
                posdef_args = posdef_args,
                posdef_fix = posdef_fix,
                posdef_func = posdef_func,
                posdef_kwargs = posdef_kwargs,
                target_ret = target_ret,
            )
        end
    else
        if calc_codep
            portfolio.codep, portfolio.dist = codep_dist_mtx(
                returns;
                alpha_tail = alpha_tail,
                bins_info = bins_info,
                codep_type = codep_type,
                cor_args = cor_args,
                cor_func = cor_func,
                cor_kwargs = cor_kwargs,
                custom_cor = custom_cor,
                dist_args = dist_args,
                dist_func = dist_func,
                dist_kwargs = dist_kwargs,
                gs_threshold = gs_threshold,
                sigma = isnothing(custom_cov) ? portfolio.cov : custom_cov,
                std_args = std_args,
                std_func = std_func,
                std_kwargs = std_kwargs,
                uplo = uplo,
            )

            (cov_type == :Gerber1 || cov_type == :Gerber2) &&
                (portfolio.gs_threshold = gs_threshold)

            codep_type == :Mutual_Info && (portfolio.bins_info = bins_info)

            codep_type == :Tail && (portfolio.alpha_tail = alpha_tail)

            portfolio.codep_type = codep_type
        end
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
    returns::AbstractVector,
    kind::Symbol = :Stationary,
    n_sim::Integer = 3_000,
    window::Integer = 3,
    seed::Union{Integer, Nothing} = nothing,
    rng = Random.default_rng(),
)
    @assert(kind ∈ KindBootstrap, "kind = $kind, must be one of $KindBootstrap")
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
    for (i, data) in pairs(gen.bootstrap(n_sim))
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
    fix_cov_args::Tuple = (),
    fix_cov_kwargs::NamedTuple = (;),
)
```
Worst case optimisation statistics.
"""
function wc_statistics!(
    portfolio::AbstractPortfolio;
    box::Symbol = :Stationary,
    calc_box::Bool = true,
    calc_ellipse::Bool = true,
    dcov::Real = 0.1,
    dmu::Real = 0.1,
    ellipse::Symbol = :Stationary,
    fix_cov_args::Tuple = (),
    fix_cov_kwargs::NamedTuple = (;),
    n_samples::Integer = 10_000,
    n_sim::Integer = 3_000,
    q::Real = 0.05,
    rng = Random.default_rng(),
    seed::Union{Integer, Nothing} = nothing,
    window::Integer = 3,
)
    @assert(box ∈ BoxTypes, "box = $box, must be one of $BoxTypes")
    @assert(ellipse ∈ EllipseTypes, "ellipse = $ellipse, must be one of $EllipseTypes")
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

function forward_regression(
    x::DataFrame,
    y::Union{Vector, DataFrame},
    criterion::Union{Symbol, Function} = :pval,
    threshold::Real = 0.05,
)
    @assert(criterion ∈ RegCriteria, "criterion = $criterion, must be one of $RegCriteria")
    isa(y, DataFrame) && (y = Vector(y))

    included = String[]
    N = length(y)
    ovec = ones(N)
    namesx = names(x)

    if criterion == :pval
        pvals = Float64[]
        val = 0.0
        while val <= threshold
            excluded = setdiff(namesx, included)
            best_pval = Inf
            new_feature = ""

            for i in excluded
                factors = [included; i]
                x1 = [ovec Matrix(x[!, factors])]
                fit_result = lm(x1, y)
                new_pvals = coeftable(fit_result).cols[4][2:end]

                idx = findfirst(x -> x == i, factors)
                test_pval = new_pvals[idx]
                if best_pval > test_pval && maximum(new_pvals) <= threshold
                    best_pval = test_pval
                    new_feature = i
                    pvals = copy(new_pvals)
                end
            end

            isempty(new_feature) ? break : push!(included, new_feature)
            !isempty(pvals) && (val = maximum(pvals))
        end

        if isempty(included)
            excluded = setdiff(namesx, included)
            best_pval = Inf
            new_feature = ""

            for i in excluded
                factors = [included; i]
                x1 = [ovec Matrix(x[!, factors])]
                fit_result = lm(x1, y)
                new_pvals = coeftable(fit_result).cols[4][2:end]

                idx = findfirst(x -> x == i, factors)
                test_pval = new_pvals[idx]
                if best_pval > test_pval
                    best_pval = test_pval
                    new_feature = i
                    pvals = copy(new_pvals)
                end
            end

            @warn(
                "No asset with p-value lower than threshold. Best we can do is $new_feature, with p-value $best_pval."
            )

            push!(included, new_feature)
        end
    else
        if criterion ∈ (GLM.aic, GLM.aicc, GLM.bic)
            threshold = Inf
        else
            threshold = -Inf
        end

        excluded = namesx
        for _ in 1:N
            ni = length(excluded)
            value = Dict()

            for i in excluded
                factors = copy(included)
                push!(factors, i)

                x1 = [ovec Matrix(x[!, factors])]
                fit_result = lm(x1, y)

                value[i] = criterion(fit_result)
            end

            isempty(value) && break

            if criterion ∈ (GLM.aic, GLM.aicc, GLM.bic)
                val, key = findmin(value)
                idx = findfirst(x -> x == key, excluded)
                if val < threshold
                    push!(included, popat!(excluded, idx))
                    threshold = val
                end
            else
                val, key = findmax(value)
                idx = findfirst(x -> x == key, excluded)
                if val > threshold
                    push!(included, popat!(excluded, idx))
                    threshold = val
                end
            end

            ni == length(excluded) && break
        end
    end

    return included
end

function backward_regression(
    x::DataFrame,
    y::Union{Vector, DataFrame},
    criterion::Union{Symbol, Function} = :pval,
    threshold::Real = 0.05,
)
    @assert(criterion ∈ RegCriteria, "criterion = $criterion, must be one of $RegCriteria")
    isa(y, DataFrame) && (y = Vector(y))

    N = length(y)
    ovec = ones(N)

    fit_result = lm([ovec Matrix(x)], y)

    included = names(x)
    namesx = names(x)

    if criterion == :pval
        excluded = String[]
        pvals = coeftable(fit_result).cols[4][2:end]
        val = maximum(pvals)
        while val > threshold
            factors = setdiff(namesx, excluded)
            included = factors

            isempty(factors) && break

            x1 = [ovec Matrix(x[!, factors])]
            fit_result = lm(x1, y)
            pvals = coeftable(fit_result).cols[4][2:end]

            val, idx2 = findmax(pvals)
            push!(excluded, factors[idx2])
        end

        if isempty(included)
            excluded = setdiff(namesx, included)
            best_pval = Inf
            new_feature = ""
            pvals = Float64[]

            for i in excluded
                factors = [included; i]
                x1 = [ovec Matrix(x[!, factors])]
                fit_result = lm(x1, y)
                new_pvals = coeftable(fit_result).cols[4][2:end]

                idx = findfirst(x -> x == i, factors)
                test_pval = new_pvals[idx]

                if best_pval > test_pval
                    best_pval = test_pval
                    new_feature = i
                    pvals = copy(new_pvals)
                end
            end

            value = maximum(pvals)
            push!(included, new_feature)
        end
    else
        threshold = criterion(fit_result)

        for _ in 1:N
            ni = length(included)
            value = Dict()
            for (i, factor) in pairs(included)
                factors = copy(included)
                popat!(factors, i)
                !isempty(factors) ? (x1 = [ovec Matrix(x[!, factors])]) :
                x1 = reshape(ovec, :, 1)
                fit_result = lm(x1, y)
                value[factor] = criterion(fit_result)
            end

            isempty(value) && break

            if criterion ∈ (GLM.aic, GLM.aicc, GLM.bic)
                val, idx = findmin(value)
                if val < threshold
                    i = findfirst(x -> x == idx, included)
                    popat!(included, i)
                    threshold = val
                end
            else
                val, idx = findmax(value)
                if val > threshold
                    i = findfirst(x -> x == idx, included)
                    popat!(included, i)
                    threshold = val
                end
            end

            ni == length(included) && break
        end
    end

    return included
end

function pcr(
    x::DataFrame,
    y::Union{Vector, DataFrame};
    mean_args::Tuple = (),
    mean_func::Function = mean,
    mean_kwargs::NamedTuple = (;),
    pca_kwargs::NamedTuple = (;),
    pca_std_kwargs::NamedTuple = (;),
    pca_std_type = ZScoreTransform,
    std_args::Tuple = (),
    std_func::Function = std,
    std_kwargs::NamedTuple = (;),
)
    N = nrow(x)
    X = transpose(Matrix(x))

    X_std = standardize(pca_std_type, X; dims = 2, pca_std_kwargs...)

    model = fit(PCA, X_std; pca_kwargs...)
    Xp = transpose(predict(model, X_std))
    Vp = projection(model)

    x1 = [ones(N) Xp]
    fit_result = lm(x1, y)
    beta_pc = coef(fit_result)[2:end]
    avg = vec(mean_func(X, mean_args...; dims = 2, mean_kwargs...))
    sdev = vec(std_func(X, std_args...; dims = 2, std_kwargs...))

    beta = Vp * beta_pc ./ sdev
    beta0 = mean(y) - dot(beta, avg)
    pushfirst!(beta, beta0)

    return beta
end

function loadings_matrix(
    x::DataFrame,
    y::DataFrame,
    type::Symbol = :FReg;
    criterion::Union{Symbol, Function} = :pval,
    mean_args::Tuple = (),
    mean_kwargs::NamedTuple = (;),
    pca_kwargs::NamedTuple = (;),
    pca_std_kwargs::NamedTuple = (;),
    pca_std_type = ZScoreTransform,
    std_args::Tuple = (),
    std_func::Function = std,
    mean_func::Function = mean,
    std_kwargs::NamedTuple = (;),
    threshold::Real = 0.05,
)
    @assert(type ∈ LoadingMtxType, "type = $type, must be one of $LoadingMtxType")
    features = names(x)
    rows = ncol(y)
    cols = ncol(x) + 1

    N = nrow(y)
    ovec = ones(N)

    loadings = zeros(rows, cols)

    for i in 1:rows
        if type == :FReg || type == :BReg
            included = if type == :FReg
                forward_regression(x, y[!, i], criterion, threshold)
            else
                backward_regression(x, y[!, i], criterion, threshold)
            end

            !isempty(included) ? (x1 = [ovec Matrix(x[!, included])]) :
            x1 = reshape(ovec, :, 1)

            fit_result = lm(x1, y[!, i])

            params = coef(fit_result)

            loadings[i, 1] = params[1]
            isempty(included) && continue
            idx = [findfirst(x -> x == i, features) + 1 for i in included]
            loadings[i, idx] .= params[2:end]
        else
            beta = pcr(
                x,
                y[!, i];
                mean_args = mean_args,
                mean_func = mean_func,
                mean_kwargs = mean_kwargs,
                pca_kwargs = pca_kwargs,
                pca_std_kwargs = pca_std_kwargs,
                pca_std_type = pca_std_type,
                std_args = std_args,
                std_func = std_func,
                std_kwargs = std_kwargs,
            )
            loadings[i, :] .= beta
        end
    end

    return hcat(DataFrame(ticker = names(y)), DataFrame(loadings, ["const"; features]))
end

function risk_factors(
    x::DataFrame,
    y::DataFrame;
    # cov_mtx
    cov_args::Tuple = (),
    cov_est::CovarianceEstimator = StatsBase.SimpleCovariance(; corrected = true),
    cov_func::Function = cov,
    cov_kwargs::NamedTuple = (;),
    cov_type::Symbol = :Full,
    cov_weights::Union{AbstractWeights, Nothing} = nothing,
    custom_cov::Union{AbstractMatrix, Nothing} = nothing,
    gs_threshold::Real = 0.5,
    jlogo::Bool = false,
    posdef_args::Tuple = (),
    posdef_fix::Symbol = :None,
    posdef_func::Function = x -> x,
    posdef_kwargs::NamedTuple = (;),
    std_args::Tuple = (),
    std_func::Function = std,
    std_kwargs::NamedTuple = (;),
    target_ret::Union{Real, Vector{<:Real}} = 0.0,
    # mean_vec
    custom_mu::Union{AbstractVector, Nothing} = nothing,
    mean_args::Tuple = (),
    mean_func::Function = mean,
    mean_kwargs::NamedTuple = (;),
    mu_target::Symbol = :GM,
    mu_type::Symbol = :Default,
    mu_weights::Union{AbstractWeights, Nothing} = nothing,
    # Loadings matrix
    B::Union{DataFrame, Nothing} = nothing,
    criterion::Union{Symbol, Function} = :pval,
    error::Bool = true,
    pca_kwargs::NamedTuple = (;),
    pca_std_kwargs::NamedTuple = (;),
    pca_std_type = ZScoreTransform,
    reg_type::Symbol = :FReg,
    threshold::Real = 0.05,
    var_func::Function = var,
    var_args::Tuple = (),
    var_kwargs::NamedTuple = (;),
)
    isnothing(B) && (
        B = loadings_matrix(
            x,
            y,
            reg_type;
            criterion = criterion,
            mean_args = mean_args,
            mean_func = mean_func,
            mean_kwargs = mean_kwargs,
            pca_kwargs = pca_kwargs,
            pca_std_kwargs = pca_std_kwargs,
            pca_std_type = pca_std_type,
            std_args = std_args,
            std_func = std_func,
            std_kwargs = std_kwargs,
            threshold = threshold,
        )
    )
    namesB = names(B)
    x1 = "const" ∈ namesB ? [ones(nrow(y)) Matrix(x)] : Matrix(x)
    B = Matrix(B[!, setdiff(namesB, ["ticker"])])

    cov_f = covar_mtx(
        x1;
        cov_args = cov_args,
        cov_est = cov_est,
        cov_func = cov_func,
        cov_kwargs = cov_kwargs,
        cov_type = cov_type,
        cov_weights = cov_weights,
        custom_cov = custom_cov,
        gs_threshold = gs_threshold,
        jlogo = jlogo,
        posdef_args = posdef_args,
        posdef_fix = posdef_fix,
        posdef_func = posdef_func,
        posdef_kwargs = posdef_kwargs,
        std_args = std_args,
        std_func = std_func,
        std_kwargs = std_kwargs,
        target_ret = target_ret,
    )

    mu_f = mean_vec(
        x1;
        custom_mu = custom_mu,
        mean_args = mean_args,
        mean_func = mean_func,
        mean_kwargs = mean_kwargs,
        mu_target = mu_target,
        mu_type = mu_type,
        mu_weights = mu_weights,
        sigma = isnothing(custom_cov) ? cov_f : custom_cov,
    )

    returns = x1 * transpose(B)
    mu = B * mu_f

    sigma = if error
        e = Matrix(y) - returns
        S_e = diagm(vec(var_func(e, var_args...; dims = 1, var_kwargs...)))
        B * cov_f * transpose(B) + S_e
    else
        B * cov_f * transpose(B)
    end

    return mu, sigma, returns
end

function _omega(P, tau_sigma)
    Diagonal(P * tau_sigma * transpose(P))
end
function _Pi(eq, delta, sigma, w, mu, rf)
    eq ? delta * sigma * w : mu .- rf
end
function _mu_cov_w(tau, omega, P, Pi, Q, rf, sigma, delta)
    inv_tau_sigma = (tau * sigma) \ I
    inv_omega = omega \ I
    Pi_ =
        ((inv_tau_sigma + transpose(P) * inv_omega * P) \ I) *
        (inv_tau_sigma * Pi + transpose(P) * inv_omega * Q)
    M = (inv_tau_sigma + transpose(P) * inv_omega * P) \ I

    mu = Pi_ .+ rf
    cov_mtx = sigma + M
    w = ((delta * cov_mtx) \ I) * Pi_

    return mu, cov_mtx, w, Pi_
end

function black_litterman(
    returns::AbstractMatrix,
    w::AbstractVector,
    P::AbstractMatrix,
    Q::AbstractVector;
    # cov_mtx
    cov_args::Tuple = (),
    cov_est::CovarianceEstimator = StatsBase.SimpleCovariance(; corrected = true),
    cov_func::Function = cov,
    cov_kwargs::NamedTuple = (;),
    cov_type::Symbol = :Full,
    cov_weights::Union{AbstractWeights, Nothing} = nothing,
    custom_cov::Union{AbstractMatrix, Nothing} = nothing,
    gs_threshold::Real = 0.5,
    jlogo::Bool = false,
    posdef_args::Tuple = (),
    posdef_fix::Symbol = :None,
    posdef_func::Function = x -> x,
    posdef_kwargs::NamedTuple = (;),
    std_args::Tuple = (),
    std_func::Function = std,
    std_kwargs::NamedTuple = (;),
    target_ret::Union{Real, Vector{<:Real}} = 0.0,
    # mean_vec
    custom_mu::Union{AbstractVector, Nothing} = nothing,
    mean_args::Tuple = (),
    mean_func::Function = mean,
    mean_kwargs::NamedTuple = (;),
    mu_target::Symbol = :GM,
    mu_type::Symbol = :Default,
    mu_weights::Union{AbstractWeights, Nothing} = nothing,
    # Black Litterman
    delta::Real = 1.0,
    eq::Bool = true,
    rf::Real = 0.0,
)
    sigma = covar_mtx(
        returns;
        cov_args = cov_args,
        cov_est = cov_est,
        cov_func = cov_func,
        cov_kwargs = cov_kwargs,
        cov_type = cov_type,
        cov_weights = cov_weights,
        custom_cov = custom_cov,
        gs_threshold = gs_threshold,
        jlogo = jlogo,
        posdef_args = posdef_args,
        posdef_fix = posdef_fix,
        posdef_func = posdef_func,
        posdef_kwargs = posdef_kwargs,
        std_args = std_args,
        std_func = std_func,
        std_kwargs = std_kwargs,
        target_ret = target_ret,
    )

    mu = mean_vec(
        returns;
        custom_mu = custom_mu,
        mean_args = mean_args,
        mean_func = mean_func,
        mean_kwargs = mean_kwargs,
        mu_target = mu_target,
        mu_type = mu_type,
        mu_weights = mu_weights,
        sigma = isnothing(custom_cov) ? sigma : custom_cov,
    )

    tau = 1 / size(returns, 1)
    omega = _omega(P, tau * sigma)
    Pi = _Pi(eq, delta, sigma, w, mu, rf)

    mu, cov_mtx, w, missing = _mu_cov_w(tau, omega, P, Pi, Q, rf, sigma, delta)

    return mu, cov_mtx, w
end

function augmented_black_litterman(
    returns::AbstractMatrix,
    w::AbstractVector;
    # cov_mtx
    cov_args::Tuple = (),
    cov_est::CovarianceEstimator = StatsBase.SimpleCovariance(; corrected = true),
    cov_func::Function = cov,
    cov_kwargs::NamedTuple = (;),
    cov_type::Symbol = :Full,
    cov_weights::Union{AbstractWeights, Nothing} = nothing,
    custom_cov::Union{AbstractMatrix, Nothing} = nothing,
    gs_threshold::Real = 0.5,
    jlogo::Bool = false,
    posdef_args::Tuple = (),
    posdef_fix::Symbol = :None,
    posdef_func::Function = x -> x,
    posdef_kwargs::NamedTuple = (;),
    std_args::Tuple = (),
    std_func::Function = std,
    std_kwargs::NamedTuple = (;),
    target_ret::Union{Real, Vector{<:Real}} = 0.0,
    # mean_vec
    custom_mu::Union{AbstractVector, Nothing} = nothing,
    mean_args::Tuple = (),
    mean_func::Function = mean,
    mean_kwargs::NamedTuple = (;),
    mu_target::Symbol = :GM,
    mu_type::Symbol = :Default,
    mu_weights::Union{AbstractWeights, Nothing} = nothing,
    # Black Litterman
    B::Union{AbstractMatrix, Nothing} = nothing,
    F::Union{AbstractMatrix, Nothing} = nothing,
    P::Union{AbstractMatrix, Nothing} = nothing,
    P_f::Union{AbstractMatrix, Nothing} = nothing,
    Q::Union{AbstractVector, Nothing} = nothing,
    Q_f::Union{AbstractVector, Nothing} = nothing,
    constant::Bool = true,
    delta::Real = 1.0,
    eq::Bool = true,
    rf::Real = 0.0,
)
    asset_tuple = (!isnothing(P), !isnothing(Q))
    any_asset_provided = any(asset_tuple)
    all_asset_provided = all(asset_tuple)
    @assert(
        any_asset_provided == all_asset_provided,
        "If any of P or Q is provided, then both must be provided."
    )

    factor_tuple = (!isnothing(B), !isnothing(F), !isnothing(P_f), !isnothing(Q_f))
    any_factor_provided = any(factor_tuple)
    all_factor_provided = all(factor_tuple)
    @assert(
        any_factor_provided == all_factor_provided,
        "If any of B, F, P_f or Q_f is provided (any(.!isnothing.(B, F, P_f, Q_f)) = $any_factor_provided), then all must be provided (all(.!isnothing.(B, F, P_f, Q_f)) = $all_factor_provided))."
    )

    !all_asset_provided &&
        !all_factor_provided &&
        throw(
            AssertionError(
                "Please provide either:\n- P and Q,\n- B, F, P_f and Q_f, or\n- P, Q, B, F, P_f and Q_f.",
            ),
        )

    if all_asset_provided
        sigma = covar_mtx(
            returns;
            cov_args = cov_args,
            cov_est = cov_est,
            cov_func = cov_func,
            cov_kwargs = cov_kwargs,
            cov_type = cov_type,
            cov_weights = cov_weights,
            custom_cov = custom_cov,
            gs_threshold = gs_threshold,
            jlogo = jlogo,
            posdef_args = posdef_args,
            posdef_fix = posdef_fix,
            posdef_func = posdef_func,
            posdef_kwargs = posdef_kwargs,
            std_args = std_args,
            std_func = std_func,
            std_kwargs = std_kwargs,
            target_ret = target_ret,
        )

        mu = mean_vec(
            returns;
            custom_mu = custom_mu,
            mean_args = mean_args,
            mean_func = mean_func,
            mean_kwargs = mean_kwargs,
            mu_target = mu_target,
            mu_type = mu_type,
            mu_weights = mu_weights,
            sigma = isnothing(custom_cov) ? sigma : custom_cov,
        )
    end

    if all_factor_provided
        sigma_f = covar_mtx(
            F;
            cov_args = cov_args,
            cov_est = cov_est,
            cov_func = cov_func,
            cov_kwargs = cov_kwargs,
            cov_type = cov_type,
            cov_weights = cov_weights,
            custom_cov = custom_cov,
            gs_threshold = gs_threshold,
            jlogo = jlogo,
            posdef_args = posdef_args,
            posdef_fix = posdef_fix,
            posdef_func = posdef_func,
            posdef_kwargs = posdef_kwargs,
            std_args = std_args,
            std_func = std_func,
            std_kwargs = std_kwargs,
            target_ret = target_ret,
        )

        mu_f = mean_vec(
            F;
            custom_mu = custom_mu,
            mean_args = mean_args,
            mean_func = mean_func,
            mean_kwargs = mean_kwargs,
            mu_target = mu_target,
            mu_type = mu_type,
            mu_weights = mu_weights,
            sigma = isnothing(custom_cov) ? sigma_f : custom_cov,
        )
    end

    if all_factor_provided && constant
        alpha = B[:, 1]
        B = B[:, 2:end]
    end

    tau = 1 / size(returns, 1)

    if all_asset_provided && !all_factor_provided
        sigma_a = sigma
        P_a = P
        Q_a = Q
        omega_a = _omega(P_a, tau * sigma_a)
        Pi_a = _Pi(eq, delta, sigma_a, w, mu, rf)
    elseif !all_asset_provided && all_factor_provided
        sigma_a = sigma_f
        P_a = P_f
        Q_a = Q_f
        omega_a = _omega(P_a, tau * sigma_a)
        Pi_a = _Pi(eq, delta, sigma_a * transpose(B), w, mu_f, rf)
    elseif all_asset_provided && all_factor_provided
        sigma_a = hcat(vcat(sigma, sigma_f * transpose(B)), vcat(B * sigma_f, sigma_f))

        zeros_1 = zeros(size(P_f, 1), size(P, 2))
        zeros_2 = zeros(size(P, 1), size(P_f, 2))

        P_a = hcat(vcat(P, zeros_1), vcat(zeros_2, P_f))
        Q_a = vcat(Q, Q_f)

        omega = _omega(P, tau * sigma)
        omega_f = _omega(P_f, tau * sigma_f)

        zeros_3 = zeros(size(omega, 1), size(omega_f, 1))

        omega_a = hcat(vcat(omega, transpose(zeros_3)), vcat(zeros_3, omega_f))

        Pi_a = _Pi(eq, delta, vcat(sigma, sigma_f * transpose(B)), w, vcat(mu, mu_f), rf)
    end

    mu_a, cov_mtx_a, w_a, Pi_a_ =
        _mu_cov_w(tau, omega_a, P_a, Pi_a, Q_a, rf, sigma_a, delta)

    if !all_asset_provided && all_factor_provided
        mu_a = B * mu_a
        cov_mtx_a = B * cov_mtx_a * transpose(B)
        w_a = ((delta * cov_mtx_a) \ I) * B * Pi_a_
    end

    N = size(returns, 2)
    all_factor_provided && constant && (mu_a = mu_a[1:N] .+ alpha)

    return mu_a[1:N], cov_mtx_a[1:N, 1:N], w_a[1:N]
end

function bayesian_black_litterman(
    returns::AbstractMatrix,
    F::AbstractMatrix,
    B::AbstractMatrix,
    P_f::AbstractMatrix,
    Q_f::AbstractVector;
    # cov_mtx
    cov_args::Tuple = (),
    cov_est::CovarianceEstimator = StatsBase.SimpleCovariance(; corrected = true),
    cov_func::Function = cov,
    cov_kwargs::NamedTuple = (;),
    cov_type::Symbol = :Full,
    cov_weights::Union{AbstractWeights, Nothing} = nothing,
    custom_cov::Union{AbstractMatrix, Nothing} = nothing,
    gs_threshold::Real = 0.5,
    jlogo::Bool = false,
    posdef_args::Tuple = (),
    posdef_fix::Symbol = :None,
    posdef_func::Function = x -> x,
    posdef_kwargs::NamedTuple = (;),
    std_args::Tuple = (),
    std_func::Function = std,
    std_kwargs::NamedTuple = (;),
    target_ret::Union{Real, Vector{<:Real}} = 0.0,
    # mean_vec
    custom_mu::Union{AbstractVector, Nothing} = nothing,
    mean_args::Tuple = (),
    mean_func::Function = mean,
    mean_kwargs::NamedTuple = (;),
    mu_target::Symbol = :GM,
    mu_type::Symbol = :Default,
    mu_weights::Union{AbstractWeights, Nothing} = nothing,
    # Black Litterman
    constant::Bool = true,
    delta::Real = 1.0,
    diagonal::Bool = true,
    rf::Real = 0.0,
    var_args::Tuple = (),
    var_func::Function = var,
    var_kwargs::NamedTuple = (;),
)
    sigma_f = covar_mtx(
        F;
        cov_args = cov_args,
        cov_est = cov_est,
        cov_func = cov_func,
        cov_kwargs = cov_kwargs,
        cov_type = cov_type,
        cov_weights = cov_weights,
        custom_cov = custom_cov,
        gs_threshold = gs_threshold,
        jlogo = jlogo,
        posdef_args = posdef_args,
        posdef_fix = posdef_fix,
        posdef_func = posdef_func,
        posdef_kwargs = posdef_kwargs,
        std_args = std_args,
        std_func = std_func,
        std_kwargs = std_kwargs,
        target_ret = target_ret,
    )

    mu_f = mean_vec(
        F;
        custom_mu = custom_mu,
        mean_args = mean_args,
        mean_func = mean_func,
        mean_kwargs = mean_kwargs,
        mu_target = mu_target,
        mu_type = mu_type,
        mu_weights = mu_weights,
        sigma = isnothing(custom_cov) ? sigma_f : custom_cov,
    )

    mu_f .-= rf

    if constant
        alpha = B[:, 1]
        B = B[:, 2:end]
    end

    tau = 1 / size(returns, 1)

    sigma = B * sigma_f * transpose(B)

    if diagonal
        D = returns - F * transpose(B)
        D = Diagonal(vec(var_func(D, var_args...; dims = 1, var_kwargs...)))
        sigma .+= D
    end

    omega_f = _omega(P_f, tau * sigma_f)

    inv_sigma = sigma \ I
    inv_sigma_f = sigma_f \ I
    inv_omega_f = omega_f \ I
    sigma_hat = (inv_sigma_f + transpose(P_f) * inv_omega_f * P_f) \ I
    Pi_hat = sigma_hat * (inv_sigma_f * mu_f + transpose(P_f) * inv_omega_f * Q_f)
    inv_sigma_hat = sigma_hat \ I
    iish_b_is_b = (inv_sigma_hat + transpose(B) * inv_sigma * B) \ I
    is_b_iish_b_is_b = inv_sigma * B * iish_b_is_b

    sigma_bbl = (inv_sigma - is_b_iish_b_is_b * transpose(B) * inv_sigma) \ I
    Pi_bbl = (sigma_bbl * is_b_iish_b_is_b * inv_sigma_hat * Pi_hat)

    mu = Pi_bbl .+ rf

    constant && (mu .+= alpha)

    w = ((delta * sigma_bbl) \ I) * mu

    return mu, sigma_bbl, w
end

function black_litterman_statistics!(
    portfolio::AbstractPortfolio,
    P::AbstractMatrix,
    Q::AbstractVector,
    w::AbstractVector = Vector{Float64}(undef, 0);
    # cov_mtx
    cov_args::Tuple = (),
    cov_est::CovarianceEstimator = StatsBase.SimpleCovariance(; corrected = true),
    cov_func::Function = cov,
    cov_kwargs::NamedTuple = (;),
    cov_type::Symbol = :Full,
    cov_weights::Union{AbstractWeights, Nothing} = nothing,
    custom_cov::Union{AbstractMatrix, Nothing} = nothing,
    gs_threshold::Real = portfolio.gs_threshold,
    jlogo::Bool = false,
    posdef_args::Tuple = (),
    posdef_fix::Symbol = :None,
    posdef_func::Function = x -> x,
    posdef_kwargs::NamedTuple = (;),
    std_args::Tuple = (),
    std_func::Function = std,
    std_kwargs::NamedTuple = (;),
    target_ret::Union{Real, Vector{<:Real}} = 0.0,
    # mean_vec
    custom_mu::Union{AbstractVector, Nothing} = nothing,
    mean_args::Tuple = (),
    mean_func::Function = mean,
    mean_kwargs::NamedTuple = (;),
    mu_target::Symbol = :GM,
    mu_type::Symbol = :Default,
    mu_weights::Union{AbstractWeights, Nothing} = nothing,
    # Black Litterman
    delta::Union{Real, Nothing} = nothing,
    eq::Bool = true,
    rf::Real = 0.0,
)
    returns = portfolio.returns
    if isempty(w)
        isempty(portfolio.bl_bench_weights) && (
            portfolio.bl_bench_weights =
                fill(1 / length(portfolio.assets), length(portfolio.assets))
        )
        w = portfolio.bl_bench_weights
    else
        portfolio.bl_bench_weights = w
    end

    isnothing(delta) && (delta = (dot(portfolio.mu, w) - rf) / dot(w, portfolio.cov, w))

    portfolio.mu_bl, portfolio.cov_bl, missing = black_litterman(
        returns,
        w,
        P,
        Q;
        # cov_mtx
        cov_args = cov_args,
        cov_est = cov_est,
        cov_func = cov_func,
        cov_kwargs = cov_kwargs,
        cov_type = cov_type,
        cov_weights = cov_weights,
        custom_cov = custom_cov,
        gs_threshold = gs_threshold,
        jlogo = jlogo,
        posdef_args = posdef_args,
        posdef_fix = posdef_fix,
        posdef_func = posdef_func,
        posdef_kwargs = posdef_kwargs,
        std_args = std_args,
        std_func = std_func,
        std_kwargs = std_kwargs,
        target_ret = target_ret,
        # mean_vec
        custom_mu = custom_mu,
        mean_args = mean_args,
        mean_func = mean_func,
        mean_kwargs = mean_kwargs,
        mu_target = mu_target,
        mu_type = mu_type,
        mu_weights = mu_weights,
        # Black Litterman
        delta = delta,
        eq = eq,
        rf = rf,
    )

    return nothing
end

function factor_statistics!(
    portfolio::AbstractPortfolio;
    # cov_mtx
    cov_args::Tuple = (),
    cov_est::CovarianceEstimator = StatsBase.SimpleCovariance(; corrected = true),
    cov_func::Function = cov,
    cov_kwargs::NamedTuple = (;),
    cov_type::Symbol = :Full,
    cov_weights::Union{AbstractWeights, Nothing} = nothing,
    custom_cov::Union{AbstractMatrix, Nothing} = nothing,
    gs_threshold::Real = portfolio.gs_threshold,
    jlogo::Bool = false,
    posdef_args::Tuple = (),
    posdef_fix::Symbol = :None,
    posdef_func::Function = x -> x,
    posdef_kwargs::NamedTuple = (;),
    std_args::Tuple = (),
    std_func::Function = std,
    std_kwargs::NamedTuple = (;),
    target_ret::Union{Real, Vector{<:Real}} = 0.0,
    # mean_vec
    custom_mu::Union{AbstractVector, Nothing} = nothing,
    mean_args::Tuple = (),
    mean_func::Function = mean,
    mean_kwargs::NamedTuple = (;),
    mu_target::Symbol = :GM,
    mu_type::Symbol = :Default,
    mu_weights::Union{AbstractWeights, Nothing} = nothing,
    # Loadings matrix
    B::Union{DataFrame, Nothing} = nothing,
    criterion::Union{Symbol, Function} = :pval,
    error::Bool = true,
    pca_kwargs::NamedTuple = (;),
    pca_std_kwargs::NamedTuple = (;),
    pca_std_type = ZScoreTransform,
    reg_type::Symbol = :FReg,
    threshold::Real = 0.05,
    var_func::Function = var,
    var_args::Tuple = (),
    var_kwargs::NamedTuple = (;),
)
    returns = portfolio.returns
    f_returns = portfolio.f_returns

    portfolio.cov_f = covar_mtx(
        returns;
        cov_args = cov_args,
        cov_est = cov_est,
        cov_func = cov_func,
        cov_kwargs = cov_kwargs,
        cov_type = cov_type,
        cov_weights = cov_weights,
        custom_cov = custom_cov,
        gs_threshold = gs_threshold,
        jlogo = jlogo,
        posdef_args = posdef_args,
        posdef_fix = posdef_fix,
        posdef_func = posdef_func,
        posdef_kwargs = posdef_kwargs,
        std_args = std_args,
        std_func = std_func,
        std_kwargs = std_kwargs,
        target_ret = target_ret,
    )

    portfolio.mu_f = mean_vec(
        returns;
        custom_mu = custom_mu,
        mean_args = mean_args,
        mean_func = mean_func,
        mean_kwargs = mean_kwargs,
        mu_target = mu_target,
        mu_type = mu_type,
        mu_weights = mu_weights,
        sigma = isnothing(custom_cov) ? portfolio.cov_f : custom_cov,
    )

    portfolio.mu_fm, portfolio.cov_fm, portfolio.returns_fm = risk_factors(
        DataFrame(f_returns, portfolio.f_assets),
        DataFrame(returns, portfolio.assets);
        # cov_mtx
        cov_args = cov_args,
        cov_est = cov_est,
        cov_func = cov_func,
        cov_kwargs = cov_kwargs,
        cov_type = cov_type,
        cov_weights = cov_weights,
        custom_cov = custom_cov,
        gs_threshold = gs_threshold,
        jlogo = jlogo,
        posdef_args = posdef_args,
        posdef_fix = posdef_fix,
        posdef_func = posdef_func,
        posdef_kwargs = posdef_kwargs,
        std_args = std_args,
        std_func = std_func,
        std_kwargs = std_kwargs,
        target_ret = target_ret,
        # mean_vec
        custom_mu = custom_mu,
        mean_args = mean_args,
        mean_func = mean_func,
        mean_kwargs = mean_kwargs,
        mu_target = mu_target,
        mu_type = mu_type,
        mu_weights = mu_weights,
        # Loadings matrix
        B = B,
        error = error,
        reg_type = reg_type,
        criterion = criterion,
        threshold = threshold,
        pca_kwargs = pca_kwargs,
        pca_std_kwargs = pca_std_kwargs,
        pca_std_type = pca_std_type,
        var_func = var_func,
        var_args = var_args,
        var_kwargs = var_kwargs,
    )

    return nothing
end

function black_litterman_factor_satistics!(
    portfolio::AbstractPortfolio,
    w::AbstractVector = Vector{Float64}(undef, 0);
    # cov_mtx
    cov_args::Tuple = (),
    cov_est::CovarianceEstimator = StatsBase.SimpleCovariance(; corrected = true),
    cov_func::Function = cov,
    cov_kwargs::NamedTuple = (;),
    cov_type::Symbol = :Full,
    cov_weights::Union{AbstractWeights, Nothing} = nothing,
    custom_cov::Union{AbstractMatrix, Nothing} = nothing,
    gs_threshold::Real = portfolio.gs_threshold,
    jlogo::Bool = false,
    posdef_args::Tuple = (),
    posdef_fix::Symbol = :None,
    posdef_func::Function = x -> x,
    posdef_kwargs::NamedTuple = (;),
    std_args::Tuple = (),
    std_func::Function = std,
    std_kwargs::NamedTuple = (;),
    target_ret::Union{Real, Vector{<:Real}} = 0.0,
    # mean_vec
    custom_mu::Union{AbstractVector, Nothing} = nothing,
    mean_args::Tuple = (),
    mean_func::Function = mean,
    mean_kwargs::NamedTuple = (;),
    mu_target::Symbol = :GM,
    mu_type::Symbol = :Default,
    mu_weights::Union{AbstractWeights, Nothing} = nothing,
    # Black Litterman
    B::Union{DataFrame, Nothing} = nothing,
    P::Union{AbstractMatrix, Nothing} = nothing,
    P_f::Union{AbstractMatrix, Nothing} = nothing,
    Q::Union{AbstractVector, Nothing} = nothing,
    Q_f::Union{AbstractVector, Nothing} = nothing,
    bl_type::Symbol = :B,
    delta::Real = 1.0,
    diagonal::Bool = true,
    eq::Bool = true,
    rf::Real = 0.0,
    var_args::Tuple = (),
    var_func::Function = var,
    var_kwargs::NamedTuple = (;),
    # Loadings matrix
    criterion::Union{Symbol, Function} = :pval,
    pca_kwargs::NamedTuple = (;),
    pca_std_kwargs::NamedTuple = (;),
    pca_std_type = ZScoreTransform,
    reg_type::Symbol = :FReg,
    threshold::Real = 0.05,
)
    @assert(
        bl_type ∈ BlackLittermanType,
        "bl_type = $bl_type, must be one of $BlackLittermanType"
    )

    returns = portfolio.returns
    f_returns = portfolio.f_returns

    if isempty(w)
        isempty(portfolio.bl_bench_weights) && (
            portfolio.bl_bench_weights =
                fill(1 / length(portfolio.assets), length(portfolio.assets))
        )
        w = portfolio.bl_bench_weights
    else
        portfolio.bl_bench_weights = w
    end

    isnothing(delta) && (delta = (dot(portfolio.mu, w) - rf) / dot(w, portfolio.cov, w))

    if isnothing(B)
        B = loadings_matrix(
            DataFrame(f_returns, portfolio.f_assets),
            DataFrame(returns, portfolio.assets),
            reg_type;
            criterion = criterion,
            mean_args = mean_args,
            mean_func = mean_func,
            mean_kwargs = mean_kwargs,
            pca_kwargs = pca_kwargs,
            pca_std_kwargs = pca_std_kwargs,
            pca_std_type = pca_std_type,
            std_args = std_args,
            std_func = std_func,
            std_kwargs = std_kwargs,
            threshold = threshold,
        )
    end
    namesB = names(B)
    constant = "const" ∈ namesB
    B = Matrix(B[!, setdiff(namesB, ["ticker"])])

    portfolio.mu_bl_fm, portfolio.cov_bl_fm, missing = if bl_type == :B
        bayesian_black_litterman(
            returns,
            f_returns,
            B,
            P_f,
            Q_f;
            # cov_mtx
            cov_args = cov_args,
            cov_est = cov_est,
            cov_func = cov_func,
            cov_kwargs = cov_kwargs,
            cov_type = cov_type,
            cov_weights = cov_weights,
            custom_cov = custom_cov,
            gs_threshold = gs_threshold,
            jlogo = jlogo,
            posdef_args = posdef_args,
            posdef_fix = posdef_fix,
            posdef_func = posdef_func,
            posdef_kwargs = posdef_kwargs,
            std_args = std_args,
            std_func = std_func,
            std_kwargs = std_kwargs,
            target_ret = target_ret,
            # mean_vec
            custom_mu = custom_mu,
            mean_args = mean_args,
            mean_func = mean_func,
            mean_kwargs = mean_kwargs,
            mu_target = mu_target,
            mu_type = mu_type,
            mu_weights = mu_weights,
            # Black Litterman
            constant = constant,
            delta = delta,
            diagonal = diagonal,
            rf = rf,
            var_args = var_args,
            var_func = var_func,
            var_kwargs = var_kwargs,
        )
    else
        augmented_black_litterman(
            returns,
            w;
            # cov_mtx
            cov_args = cov_args,
            cov_est = cov_est,
            cov_func = cov_func,
            cov_kwargs = cov_kwargs,
            cov_type = cov_type,
            cov_weights = cov_weights,
            custom_cov = custom_cov,
            gs_threshold = gs_threshold,
            jlogo = jlogo,
            posdef_args = posdef_args,
            posdef_fix = posdef_fix,
            posdef_func = posdef_func,
            posdef_kwargs = posdef_kwargs,
            std_args = std_args,
            std_func = std_func,
            std_kwargs = std_kwargs,
            target_ret = target_ret,
            # mean_vec
            custom_mu = custom_mu,
            mean_args = mean_args,
            mean_func = mean_func,
            mean_kwargs = mean_kwargs,
            mu_target = mu_target,
            mu_type = mu_type,
            mu_weights = mu_weights,
            # Black Litterman
            B = B,
            F = f_returns,
            P = P,
            P_f = P_f,
            Q = Q,
            Q_f = Q_f,
            constant = constant,
            delta = delta,
            eq = eq,
            rf = rf,
        )
    end

    return nothing
end

export block_vec_pq,
    duplication_matrix,
    elimination_matrix,
    summation_matrix,
    dup_elim_sum_matrices,
    cokurt,
    scokurt,
    asset_statistics!,
    wc_statistics!,
    fix_cov!,
    forward_regression,
    backward_regression,
    pcr,
    loadings_matrix,
    risk_factors,
    black_litterman,
    augmented_black_litterman,
    bayesian_black_litterman,
    black_litterman_statistics!,
    factor_statistics!,
    black_litterman_factor_satistics!,
    nearest_cov
