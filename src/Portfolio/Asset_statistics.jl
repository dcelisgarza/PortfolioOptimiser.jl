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

function _posdef_fix!(
    mtx::AbstractMatrix,
    posdef_fix,
    posdef_func,
    posdef_args,
    posdef_kwargs,
)
    isposdef(mtx) && return nothing

    if posdef_fix == :Custom_Func
        mtx .= posdef_func(mtx, posdef_args...; posdef_kwargs...)
    end

    !isposdef(mtx) &&
        @warn("matrix could not be made postive definite, please try a different method")

    return nothing
end

function mu_esimator(
    returns,
    mu_type,
    target = :GM;
    dims = 1,
    mu_weights = nothing,
    sigma = nothing,
)
    @assert(target ∈ MuTargets, "target must be one of $MuTargets")

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
    target_ret::Union{Real, Vector{<:Real}} = 0.0,
    mu_type::Symbol = portfolio.mu_type,
    cov_type::Symbol = portfolio.cov_type,
    jlogo::Bool = portfolio.jlogo,
    posdef_fix::Symbol = portfolio.posdef_fix,
    gs_threshold = isa(portfolio, HCPortfolio) ? portfolio.gs_threshold : 0.5,
    alpha_tail = isa(portfolio, HCPortfolio) ? portfolio.alpha_tail : nothing,
    bins_info = isa(portfolio, HCPortfolio) ? portfolio.bins_info : nothing,
    mean_func::Function = mean,
    cov_func::Function = cov,
    mu_weights::Union{AbstractWeights, Nothing} = nothing,
    cov_weights::Union{AbstractWeights, Nothing} = nothing,
    posdef_func::Function = x -> x,
    cor_func::Function = cor,
    std_func::Function = std,
    dist_func::Function = x -> sqrt.(clamp!((1 .- x) / 2, 0, 1)),
    codep_type = isa(portfolio, HCPortfolio) ? portfolio.codep_type : nothing,
    custom_mu = nothing,
    custom_cov = nothing,
    custom_kurt = nothing,
    custom_skurt = nothing,
    custom_cor = nothing,
    mu_target = :GM,
    calc_mu = true,
    calc_cov = true,
    calc_kurt = true,
    calc_codep = true,
    cov_est::CovarianceEstimator = StatsBase.SimpleCovariance(; corrected = true),
    mean_args = (),
    cov_args = (),
    posdef_args = (),
    cor_args = (),
    std_args = (),
    dist_args = (),
    mean_kwargs = (;),
    cov_kwargs = (;),
    posdef_kwargs = (;),
    cor_kwargs = (;),
    std_kwargs = (;),
    dist_kwargs = (;),
    uplo = :L,
)
    returns = portfolio.returns

    # Covariance
    if calc_cov
        @assert(cov_type ∈ CovTypes, "cov_type must be one of $CovTypes")
        portfolio.cov = if cov_type == :Full
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
            isa(portfolio, HCPortfolio) && (portfolio.gs_threshold = gs_threshold)
            covgerber1(
                returns,
                gs_threshold;
                std_func = std_func,
                std_args = std_args,
                std_kwargs = std_kwargs,
            )
        elseif cov_type == :Gerber2
            isa(portfolio, HCPortfolio) && (portfolio.gs_threshold = gs_threshold)
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
        portfolio.cov_type = cov_type

        if jlogo
            sigma = portfolio.cov
            codep = cov2cor(sigma)
            dist = sqrt.(clamp!((1 .- codep) / 2, 0, 1))
            separators, cliques = PMFG_T2s(1 .- dist .^ 2, 4)[3:4]
            @time portfolio.cov = inv(JLogo(sigma, separators, cliques))
            portfolio.jlogo = jlogo
        end

        if posdef_fix != :None
            @assert(posdef_fix ∈ PosdefFixes, "posdef_fix must be one of $PosdefFixes")
            _posdef_fix!(portfolio.cov, posdef_fix, posdef_func, posdef_args, posdef_kwargs)
        end
    end

    # Mu
    if calc_mu
        @assert(mu_type ∈ MuTypes, "mu_type must be one of $MuTypes")
        portfolio.mu = if mu_type == :Default
            isnothing(mu_weights) ? vec(mean(returns; dims = 1)) :
            vec(mean(returns, mu_weights; dims = 1))
        elseif mu_type ∈ (:JS, :BS, :BOP)
            mu_esimator(
                returns,
                mu_type,
                mu_target;
                mu_weights = mu_weights,
                sigma = isnothing(custom_cov) ? portfolio.cov : custom_cov,
                dims = 1,
            )
        elseif mu_type == :Custom_Func
            vec(mean_func(returns, mean_args...; mean_kwargs...))
        elseif mu_type == :Custom_Val
            custom_mu
        end
        portfolio.mu_type = mu_type
    end

    # Type specific
    if isa(portfolio, Portfolio)
        if calc_kurt
            portfolio.kurt =
                isnothing(custom_kurt) ? cokurt(returns, transpose(portfolio.mu)) :
                custom_kurt
            portfolio.skurt =
                isnothing(custom_skurt) ?
                scokurt(returns, transpose(portfolio.mu), target_ret) : custom_skurt

            N = length(portfolio.mu)
            missing, portfolio.L_2, portfolio.S_2 = dup_elim_sum_matrices(N)

            _posdef_fix!(
                portfolio.kurt,
                posdef_fix,
                posdef_func,
                posdef_args,
                posdef_kwargs,
            )

            _posdef_fix!(
                portfolio.skurt,
                posdef_fix,
                posdef_func,
                posdef_args,
                posdef_kwargs,
            )
        end
    else
        if calc_codep
            @assert(codep_type ∈ CodepTypes, "codep_type must be one of $CodepTypes")
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
                portfolio.gs_threshold = gs_threshold
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
                portfolio.gs_threshold = gs_threshold
            elseif codep_type == :Distance
                codep = cordistance(returns)
                dist = sqrt.(clamp!(1 .- codep, 0, 1))
            elseif codep_type == :Mutual_Info
                codep, dist = mut_var_info_mtx(returns, bins_info)
                portfolio.bins_info = bins_info
            elseif codep_type == :Tail
                codep = ltdi_mtx(returns, alpha_tail)
                dist = -log.(codep)
                portfolio.alpha_tail = alpha_tail
            elseif codep_type == :Cov_to_Cor
                codep = cov2cor(portfolio.cov)
                dist = dist_func(codep, dist_args...; dist_kwargs...)
            elseif codep_type == :Custom_Func
                codep = cor_func(returns, cor_args...; cor_kwargs...)
                dist = dist_func(codep, dist_args...; dist_kwargs...)
            elseif codep_type == :Custom_Val
                codep = custom_cor
                dist = dist_func(codep, dist_args...; dist_kwargs...)
            end

            portfolio.codep_type = codep_type
            portfolio.dist = issymmetric(dist) ? dist : Symmetric(dist, uplo)
            portfolio.codep = issymmetric(codep) ? codep : Symmetric(codep, uplo)
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

function forward_regression(
    x::DataFrame,
    y::Union{Vector, DataFrame},
    criterion::Union{Symbol, Function} = :pval,
    threshold = 0.05,
)
    @assert(criterion ∈ RegCriteria, "criterion, $criterion, must be one of $RegCriteria")
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

        excluded = names(x)
        for k in 1:N
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
    criterion = :pval,
    threshold = 0.05,
)
    @assert(criterion ∈ RegCriteria, "criterion, $criterion, must be one of $RegCriteria")
    isa(y, DataFrame) && (y = Vector(y))

    N = length(y)
    ovec = ones(N)

    fit_result = lm([ovec Matrix(x)], y)
    included = names(x)

    if criterion == :pval
        namesx = copy(included)
        excluded = String[]
        pvals = coeftable(fit_result).cols[4][2:end]
        val = maximum(pvals)
        while val > threshold
            included = setdiff(namesx, excluded)
            idx1 = isinf.(pvals)
            included = included[.!idx1]
            x1 = [ovec Matrix(x[!, included])]
            fit_result = lm(x1, y)
            pvals = coeftable(fit_result).cols[4][2:end]

            isempty(pvals) && break
            val, idx2 = findmax(pvals)

            append!(excluded, included[idx1], included[idx2])
        end

        if isempty(included)
            excluded = setdiff(namesx, included)
            best_pval = Inf
            new_feature = ""
            pvals = Float64[]

            for (i, factor) in enumerate(excluded)
                fit_result = _fit_model(included, factor, ovec, x, y)
                new_pvals = coeftable(fit_result).cols[4][2:end]
                test_pval = new_pvals[i]
                if best_pval > test_pval
                    best_pval = test_pval
                    new_feature = factor
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
            for i in axes(included, 1)
                factors = copy(included)
                popat!(factors, i)
                x1 = [ovec Matrix(x[!, included])]
                fit_result = lm(x1, y)
                value[i] = criterion(fit_result)
            end

            if criterion ∈ (:aic, :aicc, :bic)
                val, idx = findmin(value)
                if val < threshold
                    popat!(included, idx)
                    threshold = val
                end
            else
                val, idx = findmax(value)
                if val > threshold
                    popat!(included, idx)
                    threshold = val
                end
            end

            ni == length(included) && break
        end
    end

    return included
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
    backward_regression
