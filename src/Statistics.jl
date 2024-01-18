
function _calc_num_bins(xj, xi, j, i, bin_width_func)
    k1 = (maximum(xj) - minimum(xj)) / bin_width_func(xj)
    bins = if j != i
        k2 = (maximum(xi) - minimum(xi)) / bin_width_func(xi)
        Int(round(max(k1, k2)))
    else
        Int(round(k1))
    end
    return bins
end

function _calc_num_bins(N, corr = nothing)
    bins = if isnothing(corr)
        z = cbrt(8 + 324 * N + 12 * sqrt(36 * N + 729 * N^2))
        Int(round(z / 6 + 2 / (3 * z) + 1 / 3))
    else
        Int(round(sqrt(1 + sqrt(1 + 24 * N / (1 - corr^2))) / sqrt(2)))
    end

    return bins
end

function mutualinfo(A::AbstractMatrix{<:Real})
    p_i = vec(sum(A; dims = 2))
    p_j = vec(sum(A; dims = 1))

    if length(p_i) == 1 || length(p_j) == 1
        return zero(eltype(p_j))
    end

    mask = findall(A .!= 0)

    nz = vec(A[mask])
    nz_sum = sum(nz)
    log_nz = log.(nz)
    nz_nm = nz / nz_sum

    outer = p_i[getindex.(mask, 1)] .* p_j[getindex.(mask, 2)]
    log_outer = -log.(outer) .+ log(sum(p_i)) .+ log(sum(p_j))

    mi = (nz_nm .* (log_nz .- log(nz_sum)) .+ nz_nm .* log_outer)
    mi[abs.(mi) .< eps(eltype(mi))] .= 0.0

    return sum(mi)
end

function _calc_hist_data(xj, xi, bins)
    xjl = minimum(xj) - eps(eltype(xj))
    xjh = maximum(xj) + eps(eltype(xj))

    xil = minimum(xi) - eps(eltype(xi))
    xih = maximum(xi) + eps(eltype(xi))

    hx = fit(Histogram, xj, range(xjl; stop = xjh, length = bins + 1)).weights
    hx /= sum(hx)

    hy = fit(Histogram, xi, range(xil; stop = xih, length = bins + 1)).weights
    hy /= sum(hy)

    ex = entropy(hx)
    ey = entropy(hy)

    hxy = fit(Histogram, (xj, xi),
              (range(xjl; stop = xjh, length = bins + 1),
               range(xil; stop = xih, length = bins + 1))).weights

    return ex, ey, hxy
end

"""
```julia
mut_var_info_mtx(x::AbstractMatrix{<:Real}; bins_info::Union{Symbol,<:Integer} = :KN,
                 normed::Bool = true)
```

Compute the mutual information and variation of information matrices.

# Inputs

  - `x`: `T×N` array containing the returns series of the assets. `T` is the number of observations and `N` the number of assets.

  - `bins_info`: selection criterion for computing the number of bins used to calculate the mutual and variation of information statistics. Can take on an integer value or the following values:

      + An integer value explicitly defines the number of bins.

      + A choice of optimal bin width selection algorithms from [`BinMethods`](@ref).

          * `:KN`: Knuth's choice.
          * `:FD`: Freedman-Diaconis' choice.
          * `:SC`: Schotts' choice.
          * `:HGR`: Hacine-Gharbi and Ravier's choice.
"""
function mut_var_info_mtx(x::AbstractMatrix{<:Real},
                          bins_info::Union{Symbol,<:Integer} = :KN, normed::Bool = true)
    @smart_assert(bins_info ∈ BinMethods ||
                  isa(bins_info, Int) && bins_info > zero(bins_info))

    bin_width_func = if bins_info == :KN
        pyimport("astropy.stats").knuth_bin_width
    elseif bins_info == :FD
        pyimport("astropy.stats").freedman_bin_width
    elseif bins_info == :SC
        pyimport("astropy.stats").scott_bin_width
    end

    T, N = size(x)

    mut_mtx = Matrix{eltype(x)}(undef, N, N)
    var_mtx = Matrix{eltype(x)}(undef, N, N)

    for j ∈ 1:N
        xj = x[:, j]
        for i ∈ j:N
            xi = x[:, i]
            bins = if bins_info == :HGR
                corr = cor(xj, xi)
                corr == 1 ? _calc_num_bins(T) : _calc_num_bins(T, corr)
            elseif isa(bins_info, Int)
                bins_info
            else
                _calc_num_bins(xj, xi, j, i, bin_width_func)
            end

            ex, ey, hxy = _calc_hist_data(xj, xi, bins)

            mut_ixy = mutualinfo(hxy)
            var_ixy = ex + ey - 2 * mut_ixy
            if normed
                vxy = ex + ey - mut_ixy
                var_ixy = var_ixy / vxy
                mut_ixy /= min(ex, ey)
            end

            if abs(mut_ixy) < eps(typeof(mut_ixy)) || mut_ixy < 0.0
                mut_ixy = 0.0
            end
            if abs(var_ixy) < eps(typeof(var_ixy)) || var_ixy < 0.0
                var_ixy = 0.0
            end

            mut_mtx[i, j] = mut_ixy
            var_mtx[i, j] = var_ixy
        end
    end

    return Symmetric(mut_mtx, :L), Symmetric(var_mtx, :L)
end

function cordistance(v1::AbstractVector, v2::AbstractVector)
    N = length(v1)
    @smart_assert(N == length(v2) && N > 1)

    N2 = N^2

    a = pairwise(Euclidean(), v1)
    b = pairwise(Euclidean(), v2)
    A = a .- mean(a; dims = 1) .- mean(a; dims = 2) .+ mean(a)
    B = b .- mean(b; dims = 1) .- mean(b; dims = 2) .+ mean(b)

    dcov2_xx = sum(A .* A) / N2
    dcov2_xy = sum(A .* B) / N2
    dcov2_yy = sum(B .* B) / N2

    val = sqrt(dcov2_xy) / sqrt(sqrt(dcov2_xx) * sqrt(dcov2_yy))

    return val
end

function cordistance(x::AbstractMatrix)
    N = size(x, 2)

    mtx = Matrix{eltype(x)}(undef, N, N)
    for j ∈ 1:N
        xj = x[:, j]
        for i ∈ j:N
            mtx[i, j] = cordistance(x[:, i], xj)
        end
    end

    return Symmetric(mtx, :L)
end

function ltdi_mtx(x, alpha = 0.05)
    @smart_assert(0 <= alpha <= 1)
    T, N = size(x)
    k = ceil(Int, T * alpha)
    mtx = Matrix{eltype(x)}(undef, N, N)

    if k > 0
        for j ∈ 1:N
            xj = x[:, j]
            v = sort(xj)[k]
            maskj = xj .<= v
            for i ∈ j:N
                xi = x[:, i]
                u = sort(xi)[k]
                ltd = sum(xi .<= u .&& maskj) / k
                mtx[i, j] = clamp(ltd, 0, 1)
            end
        end
    end

    return Symmetric(mtx, :L)
end

function covgerber0(x, opt::GerberOpt = GerberOpt(;))
    threshold = opt.threshold
    func = opt.genfunc.func
    args = opt.genfunc.args
    kwargs = opt.genfunc.kwargs
    std_vec = vec(func(x, args...; kwargs...))

    T, N = size(x)
    mtx = Matrix{eltype(x)}(undef, N, N)

    for j ∈ 1:N
        for i ∈ 1:j
            neg = 0
            pos = 0
            for k ∈ 1:T
                if (x[k, i] >= threshold * std_vec[i] && x[k, j] >= threshold * std_vec[j]) ||
                   (x[k, i] <= -threshold * std_vec[i] &&
                    x[k, j] <= -threshold * std_vec[j])
                    pos += 1
                elseif (x[k, i] >= threshold * std_vec[i] &&
                        x[k, j] <= -threshold * std_vec[j]) ||
                       (x[k, i] <= -threshold * std_vec[i] &&
                        x[k, j] >= threshold * std_vec[j])
                    neg += 1
                end
            end
            mtx[i, j] = (pos - neg) / (pos + neg)
        end
    end

    mtx .= Matrix(Symmetric(mtx, :U))

    posdef_fix!(mtx, opt.posdef; msg = "Gerber0 Covariance ")

    return mtx .* (std_vec * transpose(std_vec))
end

function covgerber1(x, opt::GerberOpt = GerberOpt(;))
    threshold = opt.threshold
    func = opt.genfunc.func
    args = opt.genfunc.args
    std_kwargs = opt.genfunc.kwargs
    std_vec = vec(func(x, args...; std_kwargs...))

    T, N = size(x)
    mtx = Matrix{eltype(x)}(undef, N, N)
    for j ∈ 1:N
        for i ∈ 1:j
            neg = 0
            pos = 0
            nn = 0
            for k ∈ 1:T
                if (x[k, i] >= threshold * std_vec[i] && x[k, j] >= threshold * std_vec[j]) ||
                   (x[k, i] <= -threshold * std_vec[i] &&
                    x[k, j] <= -threshold * std_vec[j])
                    pos += 1
                elseif (x[k, i] >= threshold * std_vec[i] &&
                        x[k, j] <= -threshold * std_vec[j]) ||
                       (x[k, i] <= -threshold * std_vec[i] &&
                        x[k, j] >= threshold * std_vec[j])
                    neg += 1
                elseif abs(x[k, i]) < threshold * std_vec[i] &&
                       abs(x[k, j]) < threshold * std_vec[j]
                    nn += 1
                end
            end
            mtx[i, j] = (pos - neg) / (T - nn)
        end
    end

    mtx .= Symmetric(mtx, :U)

    return mtx .* (std_vec * transpose(std_vec))
end

function covgerber2(x, opt::GerberOpt = GerberOpt(;))
    threshold = opt.threshold
    func = opt.genfunc.func
    args = opt.genfunc.args
    std_kwargs = opt.genfunc.kwargs
    std_vec = vec(func(x, args...; std_kwargs...))

    T, N = size(x)
    U = Matrix{Bool}(undef, T, N)
    D = Matrix{Bool}(undef, T, N)

    for i ∈ 1:N
        U[:, i] .= x[:, i] .>= std_vec[i] * threshold
        D[:, i] .= x[:, i] .<= -std_vec[i] * threshold
    end

    # nconc = transpose(U) * U + transpose(D) * D
    # ndisc = transpose(U) * D + transpose(D) * U
    # H = nconc - ndisc

    UmD = U - D
    H = transpose(U) * UmD - transpose(D) * UmD

    h = sqrt.(diag(H))

    mtx = H ./ (h * transpose(h))

    return mtx .* (std_vec * transpose(std_vec))
end

function cov_returns(x::AbstractMatrix; iters::Integer = 5, len::Integer = 10,
                     rng = Random.default_rng(), seed::Union{Nothing,<:Integer} = nothing)
    if !isnothing(seed)
        Random.seed!(rng, seed)
    end

    n = size(x, 1)
    a = randn(rng, n + len, n)

    for _ ∈ 1:iters
        _cov = cov(a)
        _C = cholesky(_cov)
        a .= a * (_C.U \ I)
        _cov = cov(a)
        _s = transpose(sqrt.(diag(_cov)))
        a .= (a .- mean(a; dims = 1)) ./ _s
    end

    C = cholesky(x)
    return a * C.U
end

function cokurt(x::AbstractMatrix, mu::AbstractArray)
    T, N = size(x)
    y = x .- mu
    ex = eltype(y)
    o = transpose(range(; start = one(ex), stop = one(ex), length = N))
    z = kron(o, y) .* kron(y, o)
    cokurt = transpose(z) * z / T
    return cokurt
end

function scokurt(x::AbstractMatrix, mu::AbstractArray,
                 target_ret::Union{Real,<:AbstractVector{<:Real}} = 0.0)
    T, N = size(x)
    y = x .- mu
    y .= min.(y, target_ret)
    ex = eltype(y)
    o = transpose(range(; start = one(ex), stop = one(ex), length = N))
    z = kron(o, y) .* kron(y, o)
    scokurt = transpose(z) * z / T
    return scokurt
end

function coskew(x::AbstractMatrix, mu::AbstractArray)
    T, N = size(x)
    y = x .- mu
    ex = eltype(y)
    o = transpose(range(; start = one(ex), stop = one(ex), length = N))
    z = kron(o, y) .* kron(y, o)
    coskew = transpose(x) * z / T
    return coskew
end

function scoskew(x::AbstractMatrix, mu::AbstractArray,
                 target_ret::Union{Real,<:AbstractVector{<:Real}} = 0.0)
    T, N = size(x)
    y = x .- mu
    y .= min.(y, target_ret)
    ex = eltype(y)
    o = transpose(range(; start = one(ex), stop = one(ex), length = N))
    z = kron(o, y) .* kron(y, o)
    scoskew = transpose(x) * z / T
    return scoskew
end

function duplication_matrix(n::Int)
    cols = Int(n * (n + 1) / 2)
    rows = n * n
    mtx = spzeros(rows, cols)
    for j ∈ 1:n
        for i ∈ j:n
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
    for j ∈ 1:n
        ej = spzeros(1, n)
        ej[j] = 1
        for i ∈ j:n
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

function nearest_cov(mtx::AbstractMatrix, method = NCM.Newton())
    clamp!(mtx, zero(eltype(mtx)), Inf)
    s = sqrt.(diag(mtx))
    corr = cov2cor(mtx)
    corr[.!isfinite.(corr)] .= zero(eltype(corr))
    NCM.nearest_cor!(corr, method)
    _mtx = cor2cov(corr, s)

    return any(.!isfinite.(_mtx)) ? mtx : _mtx
end

function posdef_fix!(mtx::AbstractMatrix, opt::PosdefFixOpt = PosdefFixOpt(;);
                     msg::String = "")
    method = opt.method

    if method == :None || isposdef(mtx)
        return nothing
    end

    @smart_assert(method ∈ PosdefFixMethods)

    func = opt.genfunc.func
    args = opt.genfunc.args
    kwargs = opt.genfunc.kwargs

    _mtx = if method == :Nearest
        nearest_cov(mtx, args...; kwargs...)
    elseif method == :Custom_Func
        func(mtx, args...; kwargs...)
    end

    if !isposdef(_mtx)
        @warn(msg *
              "matrix could not be made postive definite, please try a different method or a tighter tolerance")
    else
        mtx .= _mtx
    end

    return nothing
end
export posdef_fix!

function errPDF(x, vals; kernel = ASH.Kernels.gaussian, m = 10, n = 1000, q = 1000)
    e_min, e_max = x * (1 - sqrt(1.0 / q))^2, x * (1 + sqrt(1.0 / q))^2
    rg = range(e_min, e_max; length = n)
    pdf1 = q ./ (2 * pi * x * rg) .* sqrt.(clamp.((e_max .- rg) .* (rg .- e_min), 0, Inf))

    e_min, e_max = x * (1 - sqrt(1.0 / q))^2, x * (1 + sqrt(1.0 / q))^2
    res = ash(vals; rng = range(e_min, e_max; length = n), kernel = kernel, m = m)
    pdf2 = [ASH.pdf(res, i) for i ∈ pdf1]
    pdf2[.!isfinite.(pdf2)] .= 0.0
    sse = sum((pdf2 - pdf1) .^ 2)

    return sse
end

function find_max_eval(vals, q; kernel = ASH.Kernels.gaussian, m::Integer = 10,
                       n::Integer = 1000, args = (), kwargs = (;))
    res = Optim.optimize(x -> errPDF(x, vals; kernel = kernel, m = m, n = n, q = q), 0.0,
                         1.0, args...; kwargs...)

    x = Optim.converged(res) ? Optim.minimizer(res) : 1.0

    e_max = x * (1.0 + sqrt(1.0 / q))^2

    return e_max, x
end
export find_max_eval

function denoise_cor(vals, vecs, num_factors, method = :Fixed)
    @smart_assert(method ∈ (:Fixed, :Spectral))

    _vals = copy(vals)

    if method == :Fixed
        _vals[1:num_factors] .= sum(_vals[1:num_factors]) / num_factors
    else
        _vals[1:num_factors] .= 0
    end

    corr = cov2cor(vecs * Diagonal(_vals) * transpose(vecs))

    return corr
end
export denoise_cor

function shrink_cor(vals, vecs, num_factors, alpha = zero(eltype(vals)))
    @smart_assert(zero(alpha) <= alpha <= one(alpha))
    # Small
    vals_l = vals[1:num_factors]
    vecs_l = vecs[:, 1:num_factors]

    # Large
    vals_r = vals[(num_factors + 1):end]
    vecs_r = vecs[:, (num_factors + 1):end]

    corr0 = vecs_r * Diagonal(vals_r) * transpose(vecs_r)
    corr1 = vecs_l * Diagonal(vals_l) * transpose(vecs_l)

    corr = corr0 + alpha * corr1 + (1 - alpha) * Diagonal(corr1)

    return corr
end
export shrink_cor

function denoise_cov(mtx::AbstractMatrix, q::Real, opt::DenoiseOpt = DenoiseOpt(;))
    method = opt.method

    if method == :None
        return mtx
    end

    alpha = opt.alpha
    detone = opt.detone
    mkt_comp = opt.mkt_comp
    kernel = opt.kernel
    m = opt.m
    n = opt.n
    args = opt.genfunc.args
    kwargs = opt.genfunc.kwargs

    corr = cov2cor(mtx)
    s = sqrt.(diag(mtx))

    vals, vecs = eigen(corr)

    max_val, missing = find_max_eval(vals, q; kernel = kernel, m = m, n = n, args = args,
                                     kwargs = kwargs)

    num_factors = findlast(vals .< max_val)
    corr = if method ∈ (:Fixed, :Spectral)
        denoise_cor(vals, vecs, num_factors, method)
    else
        shrink_cor(vals, vecs, num_factors, alpha)
    end

    if detone
        @smart_assert(one(size(mtx, 1)) <= mkt_comp <= size(mtx, 1))
        mkt_comp -= 1
        _vals = Diagonal(vals)[(end - mkt_comp):end, (end - mkt_comp):end]
        _vecs = vecs[:, (end - mkt_comp):end]
        _corr = _vecs * _vals * transpose(_vecs)
        corr .-= _corr
    end

    cov_mtx = cor2cov(corr, s)

    return cov_mtx
end
export denoise_cov

"""
```julia
mu_estimator
```
"""
function mu_estimator(returns::AbstractMatrix, opt::MuOpt = MuOpt(;))
    method = opt.method
    @smart_assert(method ∈ (:JS, :BS, :BOP, :CAPM))

    target = opt.target
    func = opt.genfunc.func
    args = opt.genfunc.args
    kwargs = opt.genfunc.kwargs
    sigma = opt.sigma

    if method != :CAPM
        T, N = size(returns)
        mu = vec(func(returns, args...; kwargs...))

        inv_sigma = sigma \ I
        evals = eigvals(sigma)
        ones = range(1; stop = 1, length = N)

        b = if target == :GM
            fill(mean(mu), N)
        elseif target == :VW
            fill(dot(ones, inv_sigma, mu) / dot(ones, inv_sigma, ones), N)
        else
            fill(tr(sigma) / T, N)
        end

        if method == :JS
            alpha = (N * mean(evals) - 2 * maximum(evals)) / dot(mu - b, mu - b) / T
            mu = (1 - alpha) * mu + alpha * b
        elseif method == :BS
            alpha = (N + 2) / ((N + 2) + T * dot(mu - b, inv_sigma, mu - b))
            mu = (1 - alpha) * mu + alpha * b
        else
            alpha = (dot(mu, inv_sigma, mu) - N / (T - N)) * dot(b, inv_sigma, b) -
                    dot(mu, inv_sigma, b)^2
            alpha /= dot(mu, inv_sigma, mu) * dot(b, inv_sigma, b) - dot(mu, inv_sigma, b)^2
            beta = (1 - alpha) * dot(mu, inv_sigma, b) / dot(mu, inv_sigma, mu)
            mu = alpha * mu + beta * b
        end
    else
        rf = opt.rf
        betas = sigma[:, end] / sigma[end, end]
        betas = betas[1:(end - 1)]
        mkt_mean_ret = func(returns[:, end], args...; kwargs...)[1]
        mu = rf .+ betas * (mkt_mean_ret - rf)
    end

    return mu
end

const DenoiseLoGoNames = (:cov, :cor, :kurt, :skurt, :bl_cov, :a_bl_cov, :af_bl_cov)
function _denoise_logo_mtx(T, N, mtx, opt, mtx_name::Symbol = :cov)
    @smart_assert(mtx_name ∈ DenoiseLoGoNames)

    if mtx_name == :cov
        msg = "Covariance "
        msg2 = "\n\t Try a different opt.method = $(opt.method), from $CovMethods."
    elseif mtx_name == :cor
        msg = "Correlation "
        msg2 = "\n\t Try a different opt.method = $(opt.method), from $CorMethods."
    elseif mtx_name == :kurt
        msg = "Kurtosis "
        msg2 = ""
    elseif mtx_name == :skurt
        msg = "Semi Kurtosis "
        msg2 = ""
    elseif mtx_name == :bl_cov
        msg = "Black Litterman Covariance "
        msg2 = "\n\t Try some different asset views."
    elseif mtx_name == :a_bl_cov
        msg = "Augmented Black Litterman Covariance "
        msg2 = "\n\t Try a different combination of asset views, factor views, and/or loadings matrix."
    elseif mtx_name == :af_bl_cov
        msg = "Augmented Black Litterman Covariance with no asset views "
        msg2 = "\n\t Try a different combination of factor views, and/or loadings matrix."
    end

    mtx = denoise_cov(mtx, T / N, opt.denoise)

    posdef_fix!(mtx, opt.posdef; msg = msg)

    if opt.jlogo
        try
            corr = cov2cor(mtx)
            dist = sqrt.(clamp!((1 .- corr) / 2, 0, 1))
            separators, cliques = PMFG_T2s(1 .- dist .^ 2, 4)[3:4]
            mtx .= J_LoGo(mtx, separators, cliques) \ I
        catch SingularException
            throw(ErrorException("$msg matrix is singular = $(SingularException). Please try one or a combination of the following:\n\t* Set opt.posdef.method = $(opt.posdef.method), to a different method from $PosdefFixMethods.\n\t* Set denoise = true.\n\t* Try both approaches at the same time.$(msg2)"))
        end

        posdef_fix!(mtx, opt.posdef; msg = "J-LoGo $msg ")
    end

    return mtx
end

"""
```julia
covar_mtx
```
"""
function covar_mtx(returns::AbstractMatrix, opt::CovOpt = CovOpt(;))
    method = opt.method

    @smart_assert(method ∈ CovMethods)

    mtx = if method ∈ (:Full, :Semi)
        estimation = opt.estimation
        estimator = estimation.estimator
        args = estimation.genfunc.args
        kwargs = estimation.genfunc.kwargs
        if method == :Semi
            target_ret = opt.estimation.target_ret
            zro = zero(eltype(returns))
            returns = if isa(target_ret, Real)
                min.(returns .- target_ret, zro)
            else
                min.(returns .- transpose(target_ret), zro)
            end
            if !haskey(kwargs, :mean)
                kwargs = (kwargs..., mean = zro)
            end
        end
        StatsBase.cov(estimator, returns, args...; kwargs...)
    elseif method == :Gerber0
        covgerber0(returns, opt.gerber)
    elseif method == :Gerber1
        covgerber1(returns, opt.gerber)
    elseif method == :Gerber2
        covgerber2(returns, opt.gerber)
    elseif method == :Custom_Func
        estimation = opt.estimation
        func = estimation.genfunc.func
        args = estimation.genfunc.args
        kwargs = estimation.genfunc.kwargs
        func(returns, args...; kwargs...)
    elseif method == :Custom_Val
        opt.estimation.custom
    end
    T, N = size(returns)
    mtx = _denoise_logo_mtx(T, N, mtx, opt, :cov)

    return mtx
end

"""
```julia
mean_vec(returns::AbstractMatrix; custom_mu::Union{AbstractVector,Nothing} = nothing,
         mean_args::Tuple = (), mean_func::Function = mean, mean_kwargs::NamedTuple = (;),
         mu_target::Symbol = :GM, mu_method::Symbol = :Default,
         mu_weights::Union{AbstractWeights,Nothing} = nothing, rf::Real = 0.0,
         sigma::Union{AbstractMatrix,Nothing} = nothing)
```
"""
function mean_vec(returns::AbstractMatrix, opt::MuOpt = MuOpt(;))
    method = opt.method
    mu = if method ∈ (:Default, :Custom_Func)
        func = opt.genfunc.func
        args = opt.genfunc.args
        kwargs = opt.genfunc.kwargs
        vec(func(returns, args...; kwargs...))
    elseif method ∈ (:JS, :BS, :BOP, :CAPM)
        mu_estimator(returns, opt)
    elseif method == :Custom_Val
        opt.custom
    end

    return mu
end

"""
```julia
cokurt_mtx
```
"""
function cokurt_mtx(returns::AbstractMatrix, mu::AbstractVector, opt::KurtOpt = KurtOpt(;))
    custom_kurt = opt.estimation.custom_kurt
    T, N = size(returns)
    if isnothing(custom_kurt)
        kurt = cokurt(returns, transpose(mu))
        kurt = _denoise_logo_mtx(T, N, kurt, opt, :kurt)
    else
        kurt = custom_kurt
    end

    target_ret = opt.estimation.target_ret
    custom_skurt = opt.estimation.custom_skurt
    if isnothing(custom_skurt)
        skurt = scokurt(returns, transpose(mu), target_ret)
        skurt = _denoise_logo_mtx(T, N, skurt, opt, :skurt)
    else
        skurt = custom_skurt
    end

    missing, L_2, S_2 = dup_elim_sum_matrices(N)

    return kurt, skurt, L_2, S_2
end

"""
```
cor_dist_mtx(
    returns::AbstractMatrix;    alpha_tail::Real = 0.05,    bins_info::Union{Symbol, Integer} = :KN,    cor_method::Symbol = :Pearson,    cor_args::Tuple = (),    cor_func::Function = cor,    cor_kwargs::NamedTuple = (;),    custom_cor::Union{AbstractMatrix, Nothing} = nothing,    dist_args::Tuple = (),    dist_func::Function = x -> sqrt.(clamp!((1 .- x) / 2, 0, 1)),    dist_kwargs::NamedTuple = (;),    gs_threshold::Real = 0.5,    posdef_args::Tuple = (),    posdef_fix::Symbol = :Nearest,    posdef_func::Function = x -> x,    posdef_kwargs::NamedTuple = (;),    sigma::Union{AbstractMatrix, Nothing} = nothing,    std_args::Tuple = (),    std_func::Function = std,    std_kwargs::NamedTuple = (;),    uplo::Symbol = :L)
```
"""
function cor_dist_mtx(returns::AbstractMatrix, opt::CorOpt = CorOpt(;))
    method = opt.method
    T, N = size(returns)
    if method ∈ (:Pearson, :Semi_Pearson)
        estimation = opt.estimation
        estimator = estimation.estimator
        args = estimation.cor_genfunc.args
        kwargs = estimation.cor_genfunc.kwargs
        if method == :Semi_Pearson
            target_ret = opt.estimation.target_ret
            zro = zero(eltype(returns))
            returns = if isa(target_ret, Real)
                min.(returns .- target_ret, zro)
            else
                min.(returns .- transpose(target_ret), zro)
            end
            if !haskey(kwargs, :mean)
                kwargs = (kwargs..., mean = zro)
            end
        end
        corr = try
            StatsBase.cor(estimator, returns, args...; kwargs...)
        catch
            StatsBase.cov2cor(Matrix(StatsBase.cov(estimator, returns, args...; kwargs...)))
        end
        corr = _denoise_logo_mtx(T, N, corr, opt, :cor)
        dist = sqrt.(clamp!((1 .- corr) / 2, 0, 1))
    elseif method == :Spearman
        corr = corspearman(returns)
        corr = _denoise_logo_mtx(T, N, corr, opt, :cor)
        dist = sqrt.(clamp!((1 .- corr) / 2, 0, 1))
    elseif method == :Kendall
        corr = corkendall(returns)
        corr = _denoise_logo_mtx(T, N, corr, opt, :cor)
        dist = sqrt.(clamp!((1 .- corr) / 2, 0, 1))
    elseif method ∈ (:Abs_Pearson, :Abs_Semi_Pearson)
        estimation = opt.estimation
        estimator = estimation.estimator
        args = estimation.cor_genfunc.args
        kwargs = estimation.cor_genfunc.kwargs
        if method == :Abs_Semi_Pearson
            target_ret = opt.estimation.target_ret
            zro = zero(eltype(returns))
            returns = if isa(target_ret, Real)
                min.(returns .- target_ret, zro)
            else
                min.(returns .- transpose(target_ret), zro)
            end
            if !haskey(kwargs, :mean)
                kwargs = (kwargs..., mean = zro)
            end
        end
        corr = try
            abs.(StatsBase.cor(estimator, returns, args...; kwargs...))
        catch
            abs.(StatsBase.cov2cor(Matrix(StatsBase.cov(estimator, returns, args...;
                                                        kwargs...))))
        end
        corr = _denoise_logo_mtx(T, N, corr, opt, :cor)
        dist = sqrt.(clamp!(1 .- corr, 0, 1))
    elseif method == :Abs_Spearman
        corr = abs.(corspearman(returns))
        corr = _denoise_logo_mtx(T, N, corr, opt, :cor)
        dist = sqrt.(clamp!(1 .- corr, 0, 1))
    elseif method == :Abs_Kendall
        corr = abs.(corkendall(returns))
        corr = _denoise_logo_mtx(T, N, corr, opt, :cor)
        dist = sqrt.(clamp!(1 .- corr, 0, 1))
    elseif method == :Gerber0
        corr = cov2cor(covgerber0(returns, opt.gerber))
        corr = _denoise_logo_mtx(T, N, corr, opt, :cor)
        dist = sqrt.(clamp!((1 .- corr) / 2, 0, 1))
    elseif method == :Gerber1
        corr = cov2cor(covgerber1(returns, opt.gerber))
        corr = _denoise_logo_mtx(T, N, corr, opt, :cor)
        dist = sqrt.(clamp!((1 .- corr) / 2, 0, 1))
    elseif method == :Gerber2
        corr = cov2cor(covgerber2(returns, opt.gerber))
        corr = _denoise_logo_mtx(T, N, corr, opt, :cor)
        dist = sqrt.(clamp!((1 .- corr) / 2, 0, 1))
    elseif method == :Distance
        corr = cordistance(returns)
        corr = _denoise_logo_mtx(T, N, corr, opt, :cor)
        dist = sqrt.(clamp!(1 .- corr, 0, 1))
    elseif method == :Mutual_Info
        corr, dist = mut_var_info_mtx(returns, opt.estimation.bins_info)
    elseif method == :Tail
        corr = ltdi_mtx(returns, opt.estimation.alpha)
        corr = _denoise_logo_mtx(T, N, corr, opt, :cor)
        dist = -log.(corr)
    elseif method == :Cov_to_Cor
        estimation = opt.estimation
        sigma = estimation.sigma
        dist_func = estimation.dist_genfunc.func
        dist_args = estimation.dist_genfunc.args
        dist_kwargs = estimation.dist_genfunc.kwargs
        corr = cov2cor(sigma)
        dist = dist_func(corr, dist_args...; dist_kwargs...)
    elseif method == :Custom_Func
        estimation = opt.estimation
        cor_func = estimation.cor_genfunc.func
        cor_args = estimation.cor_genfunc.args
        cor_kwargs = estimation.cor_genfunc.kwargs
        dist_func = estimation.dist_genfunc.func
        dist_args = estimation.dist_genfunc.args
        dist_kwargs = estimation.dist_genfunc.kwargs
        corr = cor_func(returns, cor_args...; cor_kwargs...)
        dist = dist_func(corr, dist_args...; dist_kwargs...)
    elseif method == :Custom_Val
        estimation = opt.estimation
        corr = estimation.custom_cor
        dist = estimation.custom_dist
    end

    corr = issymmetric(corr) ? corr : Symmetric(corr, opt.uplo)
    dist = issymmetric(dist) ? dist : Symmetric(dist, opt.uplo)

    return corr, dist
end

function covar_mtx_mean_vec(returns; cov_opt::CovOpt = CovOpt(;), mu_opt::MuOpt = MuOpt(;))
    mu_method = mu_opt.method
    if mu_method == :CAPM
        mkt_ret = mu_opt.mkt_ret
        if isnothing(mkt_ret)
            returns = hcat(returns, mean(returns; dims = 2))
        else
            returns = hcat(returns, mkt_ret)
        end
    end

    sigma = covar_mtx(returns, cov_opt)

    mu_opt.sigma = sigma
    mu = mean_vec(returns, mu_opt)

    if mu_method == :CAPM
        sigma = sigma[1:(end - 1), 1:(end - 1)]
    end

    return sigma, mu
end

"""
```julia
asset_statistics!(portfolio::AbstractPortfolio; target_ret::AbstractFloat = 0.0,
                  mean_func::Function = mean, cov_func::Function = cov,
                  cor_func::Function = cor, std_func = std,
                  dist_func::Function = x -> sqrt.(clamp!((1 .- x) / 2, 0, 1)),
                  cor_method::Symbol = if isa(portfolio, HCPortfolio)
                      portfolio.cor_method                           # flags
                  else                           # flags
                      :Pearson
                  end, custom_mu = nothing, custom_cov = nothing, custom_kurt = nothing,
                  custom_skurt = nothing, mean_args::Tuple = (),                           # cov_mtx
                  cov_args::Tuple = (), cor_args::Tuple = (), dist_args::Tuple = (),
                  std_args::Tuple = (), calc_kurt = true, mean_kwargs = (; dims = 1),
                  cov_kwargs::NamedTuple = (;), cor_kwargs::NamedTuple = (;),
                  dist_kwargs::NamedTuple = (;), std_kwargs::NamedTuple = (;), uplo = :L)                           # flags
```
"""
function asset_statistics!(portfolio::AbstractPortfolio;                           # flags
                           calc_cor::Bool = true, calc_cov::Bool = true,
                           calc_mu::Bool = true, calc_kurt::Bool = true,                           # cov_mtx
                           cov_opt::CovOpt = CovOpt(;), mu_opt::MuOpt = MuOpt(;),
                           kurt_opt::KurtOpt = KurtOpt(;), cor_opt::CorOpt = CorOpt(;))
    returns = portfolio.returns

    if calc_cov || calc_mu
        sigma, mu = covar_mtx_mean_vec(returns; cov_opt = cov_opt, mu_opt = mu_opt)
    end
    if calc_cov
        portfolio.cov = sigma
    end
    if calc_mu
        portfolio.mu = mu
    end

    if calc_kurt
        portfolio.kurt, portfolio.skurt, portfolio.L_2, portfolio.S_2 = cokurt_mtx(returns,
                                                                                   portfolio.mu,
                                                                                   kurt_opt)
    end

    # Type specific
    if isa(portfolio, HCPortfolio) && calc_cor
        cor_opt.estimation.sigma = portfolio.cov
        portfolio.cor, portfolio.dist = cor_dist_mtx(returns, cor_opt)
        portfolio.cor_method = cor_opt.method
    end

    return nothing
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

function gen_bootstrap(returns::AbstractMatrix, kind::Symbol = :Stationary,
                       n_sim::Integer = 3_000, window::Integer = 3,
                       seed::Union{<:Integer,Nothing} = nothing, rng = Random.default_rng())
    @smart_assert(kind ∈ BootstrapMethods)
    if !isnothing(seed)
        Random.seed!(rng, seed)
    end

    mus = Vector{Vector{eltype(returns)}}(undef, 0)
    sizehint!(mus, n_sim)
    covs = Vector{Matrix{eltype(returns)}}(undef, 0)
    sizehint!(covs, n_sim)

    bootstrap_func = if kind == :Stationary
        pyimport("arch.bootstrap").StationaryBootstrap
    elseif kind == :Circular
        pyimport("arch.bootstrap").CircularBlockBootstrap
    elseif kind == :Moving
        pyimport("arch.bootstrap").MovingBlockBootstrap
    end

    gen = bootstrap_func(window, returns; seed = seed)
    for data ∈ gen.bootstrap(n_sim)
        A = data[1][1]
        push!(mus, vec(mean(A; dims = 1)))
        push!(covs, cov(A))
    end

    return mus, covs
end

function vec_of_vecs_to_mtx(x::AbstractVector{<:AbstractArray})
    return vcat(transpose.(x)...)
end

"""
```julia
wc_statistics!(portfolio; box = :Stationary, ellipse = :Stationary, calc_box = true,
               calc_ellipse = true, q = 0.05, n_sim = 3_000, window = 3, dmu = 0.1,
               dcov = 0.1, seed = nothing, rng = Random.default_rng(),
               fix_cov_args::Tuple = (), fix_cov_kwargs::NamedTuple = (;))
```

Worst case optimisation statistics.
"""
function wc_statistics!(portfolio::AbstractPortfolio, opt::WCOpt = WCOpt(;))
    calc_box = opt.calc_box
    calc_ellipse = opt.calc_ellipse
    box = opt.box
    ellipse = opt.ellipse
    dcov = opt.dcov
    dmu = opt.dmu
    q = opt.q
    rng = opt.rng
    seed = opt.seed
    n_sim = opt.n_sim
    window = opt.window
    posdef = opt.posdef

    @smart_assert(calc_box || calc_ellipse)

    returns = portfolio.returns
    T, N = size(returns)

    if box == :Delta || ellipse == :Stationary || ellipse == :Circular || ellipse == :Moving
        mu = vec(mean(returns; dims = 1))
    end

    if calc_ellipse || box == :Normal || box == :Delta
        sigma = cov(returns)
    end

    cov_l = Matrix{eltype(returns)}(undef, 0, 0)
    cov_u = Matrix{eltype(returns)}(undef, 0, 0)

    if calc_box
        if box == :Stationary || box == :Circular || box == :Moving
            mus, covs = gen_bootstrap(returns, box, n_sim, window, seed, rng)

            mu_s = vec_of_vecs_to_mtx(mus)
            mu_l = [quantile(mu_s[:, i], q / 2) for i ∈ 1:N]
            mu_u = [quantile(mu_s[:, i], 1 - q / 2) for i ∈ 1:N]

            cov_s = vec_of_vecs_to_mtx(vec.(covs))
            cov_l = reshape([quantile(cov_s[:, i], q / 2) for i ∈ 1:(N * N)], N, N)
            cov_u = reshape([quantile(cov_s[:, i], 1 - q / 2) for i ∈ 1:(N * N)], N, N)

            d_mu = (mu_u - mu_l) / 2
        elseif box == :Normal
            if !isnothing(seed)
                Random.seed!(rng, seed)
            end
            d_mu = cquantile(Normal(), q / 2) * sqrt.(diag(sigma) / T)
            cov_s = vec_of_vecs_to_mtx(vec.(rand(Wishart(T, sigma / T), n_sim)))

            cov_l = reshape([quantile(cov_s[:, i], q / 2) for i ∈ 1:(N * N)], N, N)
            cov_u = reshape([quantile(cov_s[:, i], 1 - q / 2) for i ∈ 1:(N * N)], N, N)
        elseif box == :Delta
            d_mu = dmu * abs.(mu)
            cov_l = sigma - dcov * abs.(sigma)
            cov_u = sigma + dcov * abs.(sigma)
        end
    end

    if calc_ellipse
        if ellipse == :Stationary || ellipse == :Circular || ellipse == :Moving
            mus, covs = gen_bootstrap(returns, ellipse, n_sim, window, seed, rng)

            cov_mu = Diagonal(cov(vec_of_vecs_to_mtx([mu_s .- mu for mu_s ∈ mus])))
            cov_sigma = Diagonal(cov(vec_of_vecs_to_mtx([vec(cov_s) .- vec(sigma)
                                                         for cov_s ∈ covs])))
        elseif ellipse == :Normal
            cov_mu = Diagonal(sigma) / T
            K = commutation_matrix(sigma)
            cov_sigma = T * Diagonal((I + K) * kron(cov_mu, cov_mu))
        end
    end

    if !isposdef(cov_l)
        posdef_fix!(cov_l, posdef; msg = "WC cov_l ")
    end
    if !isposdef(cov_u)
        posdef_fix!(cov_u, posdef; msg = "WC cov_u ")
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

function forward_regression(x::DataFrame, y::Union{Vector,DataFrame},
                            criterion::Symbol = :pval, threshold::Real = 0.05)
    @smart_assert(criterion ∈ RegCriteria)
    if isa(y, DataFrame)
        y = Vector(y)
    end

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

            for i ∈ excluded
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
            if !isempty(pvals)
                val = maximum(pvals)
            end
        end

        if isempty(included)
            excluded = setdiff(namesx, included)
            best_pval = Inf
            new_feature = ""

            for i ∈ excluded
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

            @warn("No asset with p-value lower than threshold. Best we can do is $new_feature, with p-value $best_pval.")

            push!(included, new_feature)
        end
    else
        threshold = criterion ∈ (:aic, :aicc, :bic) ? Inf : -Inf

        criterion_func = if criterion == :aic
            GLM.aic
        elseif criterion == :aicc
            GLM.aicc
        elseif criterion == :bic
            GLM.bic
        elseif criterion == :r2
            GLM.r2
        elseif criterion == :adjr2
            GLM.adjr2
        end

        excluded = namesx
        for _ ∈ 1:N
            ni = length(excluded)
            value = Dict()

            for i ∈ excluded
                factors = copy(included)
                push!(factors, i)

                x1 = [ovec Matrix(x[!, factors])]
                fit_result = lm(x1, y)

                value[i] = criterion_func(fit_result)
            end

            if isempty(value)
                break
            end

            if criterion ∈ (:aic, :aicc, :bic)
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

            if ni == length(excluded)
                break
            end
        end
    end

    return included
end

function backward_regression(x::DataFrame, y::Union{Vector,DataFrame},
                             criterion::Symbol = :pval, threshold::Real = 0.05)
    @smart_assert(criterion ∈ RegCriteria)
    if isa(y, DataFrame)
        y = Vector(y)
    end

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

            if isempty(factors)
                break
            end

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

            for i ∈ excluded
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
        criterion_func = if criterion == :aic
            GLM.aic
        elseif criterion == :aicc
            GLM.aicc
        elseif criterion == :bic
            GLM.bic
        elseif criterion == :r2
            GLM.r2
        elseif criterion == :adjr2
            GLM.adjr2
        end

        threshold = criterion_func(fit_result)

        for _ ∈ 1:N
            ni = length(included)
            value = Dict()
            for (i, factor) ∈ pairs(included)
                factors = copy(included)
                popat!(factors, i)
                if !isempty(factors)
                    x1 = [ovec Matrix(x[!, factors])]
                else
                    x1 = reshape(ovec, :, 1)
                end
                fit_result = lm(x1, y)
                value[factor] = criterion_func(fit_result)
            end

            if isempty(value)
                break
            end

            if criterion ∈ (:aic, :aicc, :bic)
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

            if ni == length(included)
                break
            end
        end
    end

    return included
end

function pcr(x::DataFrame, y::Union{Vector,DataFrame}, opt::PCROpt = PCROpt(;))
    mean_genfunc = opt.mean_genfunc
    std_genfunc = opt.std_genfunc
    pca_s_genfunc = opt.pca_s_genfunc
    pca_genfunc = opt.pca_genfunc

    N = nrow(x)
    X = transpose(Matrix(x))

    pca_s_func = pca_s_genfunc.func
    pca_s_args = pca_s_genfunc.args
    pca_s_kwargs = pca_s_genfunc.kwargs
    X_std = pca_s_func(pca_s_args..., X; pca_s_kwargs...)

    pca_func = pca_genfunc.func
    pca_args = pca_genfunc.args
    pca_kwargs = pca_genfunc.kwargs
    model = pca_func(pca_args..., X_std; pca_kwargs...)
    Xp = transpose(predict(model, X_std))
    Vp = projection(model)

    x1 = [ones(N) Xp]
    fit_result = lm(x1, y)
    beta_pc = coef(fit_result)[2:end]

    mean_func = mean_genfunc.func
    mean_args = mean_genfunc.args
    mean_kwargs = mean_genfunc.kwargs
    avg = vec(mean_func(X, mean_args...; mean_kwargs...))

    std_func = std_genfunc.func
    std_args = std_genfunc.args
    std_kwargs = std_genfunc.kwargs
    sdev = vec(std_func(X, std_args...; std_kwargs...))

    beta = Vp * beta_pc ./ sdev
    beta0 = mean(y) - dot(beta, avg)
    pushfirst!(beta, beta0)

    return beta
end

function loadings_matrix(x::DataFrame, y::DataFrame, opt::LoadingsOpt = LoadingsOpt(;))
    features = names(x)
    rows = ncol(y)
    cols = ncol(x) + 1

    N = nrow(y)
    ovec = ones(N)

    loadings = zeros(rows, cols)

    method = opt.method
    flag = method ∈ (:FReg, :BReg)
    criterion = opt.criterion
    threshold = opt.threshold
    pcr_opt = opt.pcr_opt
    for i ∈ 1:rows
        if flag
            included = if method == :FReg
                forward_regression(x, y[!, i], criterion, threshold)
            else
                backward_regression(x, y[!, i], criterion, threshold)
            end

            x1 = !isempty(included) ? [ovec Matrix(x[!, included])] : reshape(ovec, :, 1)

            fit_result = lm(x1, y[!, i])

            params = coef(fit_result)

            loadings[i, 1] = params[1]
            if isempty(included)
                continue
            end
            idx = [findfirst(x -> x == i, features) + 1 for i ∈ included]
            loadings[i, idx] .= params[2:end]
        else
            beta = pcr(x, y[!, i], pcr_opt)
            loadings[i, :] .= beta
        end
    end

    return hcat(DataFrame(; ticker = names(y)), DataFrame(loadings, ["const"; features]))
end

function risk_factors(x::DataFrame, y::DataFrame; factor_opt::FactorOpt = FactorOpt(;),
                      cov_opt::CovOpt = CovOpt(;), mu_opt::MuOpt = MuOpt(;))
    B = factor_opt.B

    if isnothing(B)
        B = loadings_matrix(x, y, factor_opt.loadings_opt)
    end
    namesB = names(B)
    x1 = "const" ∈ namesB ? [ones(nrow(y)) Matrix(x)] : Matrix(x)
    B_mtx = Matrix(B[!, setdiff(namesB, ("ticker",))])

    cov_f, mu_f = covar_mtx_mean_vec(x1; cov_opt = cov_opt, mu_opt = mu_opt)

    returns = x1 * transpose(B_mtx)
    mu = B_mtx * mu_f

    sigma = if factor_opt.error
        var_func = factor_opt.var_genfunc.func
        var_args = factor_opt.var_genfunc.args
        var_kwargs = factor_opt.var_genfunc.kwargs
        e = Matrix(y) - returns
        S_e = diagm(vec(var_func(e, var_args...; var_kwargs...)))
        B_mtx * cov_f * transpose(B_mtx) + S_e
    else
        B_mtx * cov_f * transpose(B_mtx)
    end

    posdef_fix!(sigma, cov_opt.posdef; msg = "Factor Covariance ")

    return mu, sigma, returns, B
end

function _omega(P, tau_sigma)
    return Diagonal(P * tau_sigma * transpose(P))
end
function _Pi(eq, delta, sigma, w, mu, rf)
    return eq ? delta * sigma * w : mu .- rf
end
function _mu_cov_w(tau, omega, P, Pi, Q, rf, sigma, delta, T, N, opt, cov_type)
    inv_tau_sigma = (tau * sigma) \ I
    inv_omega = omega \ I
    Pi_ = ((inv_tau_sigma + transpose(P) * inv_omega * P) \ I) *
          (inv_tau_sigma * Pi + transpose(P) * inv_omega * Q)
    M = (inv_tau_sigma + transpose(P) * inv_omega * P) \ I

    mu = Pi_ .+ rf
    cov_mtx = sigma + M

    cov_mtx = _denoise_logo_mtx(T, N, cov_mtx, opt, cov_type)

    w = ((delta * cov_mtx) \ I) * Pi_

    return mu, cov_mtx, w, Pi_
end

function black_litterman(returns::AbstractMatrix, P::AbstractMatrix, Q::AbstractVector,
                         w::AbstractVector; cov_opt::CovOpt = CovOpt(;),
                         mu_opt::MuOpt = MuOpt(;), bl_opt::BLOpt = BLOpt(;))
    eq = bl_opt.eq
    delta = bl_opt.delta
    rf = bl_opt.rf

    sigma, mu = covar_mtx_mean_vec(returns; cov_opt = cov_opt, mu_opt = mu_opt)

    tau = 1 / size(returns, 1)
    omega = _omega(P, tau * sigma)
    Pi = _Pi(eq, delta, sigma, w, mu, rf)

    T, N = size(returns)
    mu, cov_mtx, w, missing = _mu_cov_w(tau, omega, P, Pi, Q, rf, sigma, delta, T, N,
                                        bl_opt, :bl_cov)

    return mu, cov_mtx, w
end

function bayesian_black_litterman(returns::AbstractMatrix, F::AbstractMatrix,
                                  B::AbstractMatrix, P_f::AbstractMatrix,
                                  Q_f::AbstractVector; cov_opt::CovOpt = CovOpt(;),
                                  mu_opt::MuOpt = MuOpt(;), bl_opt::BLOpt = BLOpt(;))
    sigma_f, mu_f = covar_mtx_mean_vec(F; cov_opt = cov_opt, mu_opt = mu_opt)

    constant = bl_opt.constant
    diagonal = bl_opt.diagonal
    delta = bl_opt.delta
    rf = bl_opt.rf
    var_args = bl_opt.var_genfunc.args
    var_func = bl_opt.var_genfunc.func
    var_kwargs = bl_opt.var_genfunc.kwargs

    mu_f .-= rf

    if constant
        alpha = B[:, 1]
        B = B[:, 2:end]
    end

    tau = 1 / size(returns, 1)

    sigma = B * sigma_f * transpose(B)

    if diagonal
        D = returns - F * transpose(B)
        D = Diagonal(vec(var_func(D, var_args...; var_kwargs...)))
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

    if constant
        mu .+= alpha
    end

    w = ((delta * sigma_bbl) \ I) * mu

    return mu, sigma_bbl, w
end

function augmented_black_litterman(returns::AbstractMatrix, w::AbstractVector;
                                   F::Union{AbstractMatrix,Nothing}   = nothing,
                                   B::Union{AbstractMatrix,Nothing}   = nothing,
                                   P::Union{AbstractMatrix,Nothing}   = nothing,
                                   P_f::Union{AbstractMatrix,Nothing} = nothing,
                                   Q::Union{AbstractVector,Nothing}   = nothing,
                                   Q_f::Union{AbstractVector,Nothing} = nothing,
                                   cov_opt::CovOpt                    = CovOpt(;),
                                   mu_opt::MuOpt                      = MuOpt(;),
                                   f_cov_opt::CovOpt                  = CovOpt(;),
                                   f_mu_opt::MuOpt                    = MuOpt(;),
                                   bl_opt::BLOpt                      = BLOpt(;))
    asset_tuple = (!isnothing(P), !isnothing(Q))
    any_asset_provided = any(asset_tuple)
    all_asset_provided = all(asset_tuple)
    @smart_assert(any_asset_provided == all_asset_provided,
                  "If any of P or Q is provided, then both must be provided.")

    factor_tuple = (!isnothing(P_f), !isnothing(Q_f))
    any_factor_provided = any(factor_tuple)
    all_factor_provided = all(factor_tuple)
    @smart_assert(any_factor_provided == all_factor_provided,
                  "If any of P_f or Q_f is provided, then both must be provided.")

    if all_factor_provided
        @smart_assert(!isnothing(B) && !isnothing(F),
                      "If P_f and Q_f are provided, then B and F must be provided.")
    end

    if !all_asset_provided && !all_factor_provided
        throw(AssertionError("Please provide either:\n- P and Q,\n- B, F, P_f and Q_f, or\n- P, Q, B, F, P_f and Q_f."))
    end

    if all_asset_provided
        sigma, mu = covar_mtx_mean_vec(returns; cov_opt = cov_opt, mu_opt = mu_opt)
    end

    if all_factor_provided
        sigma_f, mu_f = covar_mtx_mean_vec(F; cov_opt = f_cov_opt, mu_opt = f_mu_opt)
    end

    constant = bl_opt.constant
    eq = bl_opt.eq
    delta = bl_opt.delta
    rf = bl_opt.rf

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

    T, N = size(returns)
    mu_a, cov_mtx_a, w_a, Pi_a_ = _mu_cov_w(tau, omega_a, P_a, Pi_a, Q_a, rf, sigma_a,
                                            delta, T, N, bl_opt, :a_bl_cov)

    if !all_asset_provided && all_factor_provided
        mu_a = B * mu_a
        cov_mtx_a = B * cov_mtx_a * transpose(B)
        cov_mtx_a = _denoise_logo_mtx(T, N, cov_mtx_a, bl_opt, :af_bl_cov)
        w_a = ((delta * cov_mtx_a) \ I) * B * Pi_a_
    end

    if all_factor_provided && constant
        mu_a = mu_a[1:N] .+ alpha
    end

    return mu_a[1:N], cov_mtx_a[1:N, 1:N], w_a[1:N]
end

"""
```julia
black_litterman_statistics!(portfolio::AbstractPortfolio, P::AbstractMatrix,
                            Q::AbstractVector;
                            w::AbstractVector                              = Vector{Float64}(undef, 0),    # cov_mtx
                            cov_args::Tuple                                = (),
                            cov_est::CovarianceEstimator                   = StatsBase.SimpleCovariance(; corrected = true),
                            cov_func::Function                             = cov,
                            cov_kwargs::NamedTuple                         = (;),
                            cov_method::Symbol                             = :Full,
                            cov_weights::Union{AbstractWeights,Nothing}    = nothing,
                            custom_cov::Union{AbstractMatrix,Nothing}      = nothing,
                            gs_threshold::Real                             = portfolio.gs_threshold,
                            jlogo::Bool                                    = false,
                            posdef_args::Tuple                             = (),
                            posdef_fix::Symbol                             = :Nearest,
                            posdef_func::Function                          = x -> x,
                            posdef_kwargs::NamedTuple                      = (;),
                            std_args::Tuple                                = (),
                            std_func::Function                             = std,
                            std_kwargs::NamedTuple                         = (;),
                            target_ret::Union{Real,AbstractVector{<:Real}} = 0.0,    # mean_vec
                            custom_mu::Union{AbstractVector,Nothing}       = nothing,
                            mean_args::Tuple                               = (),
                            mean_func::Function                            = mean,
                            mean_kwargs::NamedTuple                        = (;),
                            mkt_ret::Union{AbstractVector,Nothing}         = nothing,
                            mu_target::Symbol                              = :GM,
                            mu_method::Symbol                              = :Default,
                            mu_weights::Union{AbstractWeights,Nothing}     = nothing,    # Black Litterman
                            delta::Union{Real,Nothing}                     = nothing,
                            eq::Bool                                       = true,
                            rf::Real                                       = 0.0)
```
"""
function black_litterman_statistics!(portfolio::AbstractPortfolio, P::AbstractMatrix,
                                     Q::AbstractVector,
                                     w::AbstractVector = Vector{Float64}(undef, 0);
                                     cov_opt::CovOpt = CovOpt(;), mu_opt::MuOpt = MuOpt(;),
                                     bl_opt::BLOpt = BLOpt(;))
    returns = portfolio.returns
    if isempty(w)
        if isempty(portfolio.bl_bench_weights)
            portfolio.bl_bench_weights = fill(1 / size(portfolio.returns, 2),
                                              size(portfolio.returns, 2))
        end
        w = portfolio.bl_bench_weights
    else
        portfolio.bl_bench_weights = w
    end

    if isnothing(bl_opt.delta)
        bl_opt.delta = (dot(portfolio.mu, w) - bl_opt.rf) / dot(w, portfolio.cov, w)
    end

    portfolio.mu_bl, portfolio.cov_bl, missing = black_litterman(returns, P, Q, w;
                                                                 cov_opt = cov_opt,
                                                                 mu_opt = mu_opt,
                                                                 bl_opt = bl_opt)

    return nothing
end

"""
```julia
factor_statistics!(portfolio::AbstractPortfolio;    # cov_mtx
                   alpha::Real = 0.0, cov_args::Tuple = (),
                   cov_est::CovarianceEstimator = StatsBase.SimpleCovariance(;
                                                                             corrected = true),
                   cov_func::Function = cov, cov_kwargs::NamedTuple = (;),
                   cov_method::Symbol = :Full,
                   cov_weights::Union{AbstractWeights,Nothing} = nothing,
                   custom_cov::Union{AbstractMatrix,Nothing} = nothing,
                   denoise::Bool = false, detone::Bool = false,
                   gs_threshold::Real = portfolio.gs_threshold, jlogo::Bool = false,
                   kernel = ASH.Kernels.gaussian, m::Integer = 10, method::Symbol = :Fixed,
                   mkt_comp::Integer = 1, n::Integer = 1000, opt_args = (),
                   opt_kwargs = (;), posdef_args::Tuple = (), posdef_fix::Symbol = :Nearest,
                   posdef_func::Function = x -> x, posdef_kwargs::NamedTuple = (;),
                   std_args::Tuple = (), std_func::Function = std,
                   std_kwargs::NamedTuple = (;),
                   target_ret::Union{Real,AbstractVector{<:Real}} = 0.0,    # mean_vec
                   custom_mu::Union{AbstractVector,Nothing} = nothing,
                   mean_args::Tuple = (), mean_func::Function = mean,
                   mean_kwargs::NamedTuple = (;),
                   mkt_ret::Union{AbstractVector,Nothing} = nothing,
                   mu_target::Symbol = :GM, mu_method::Symbol = :Default,
                   mu_weights::Union{AbstractWeights,Nothing} = nothing, rf = 0.0,    # Loadings matrix
                   B::Union{DataFrame,Nothing} = nothing, criterion::Symbol = :pval,
                   error::Bool = true, pca_kwargs::NamedTuple = (;),
                   pca_std_kwargs::NamedTuple = (;), pca_std_type = ZScoreTransform,
                   reg_method::Symbol = :FReg, threshold::Real = 0.05,
                   var_func::Function = var, var_args::Tuple = (),
                   var_kwargs::NamedTuple = (;))
```
"""
function factor_statistics!(portfolio::AbstractPortfolio; cov_f_opt::CovOpt = CovOpt(;),
                            mu_f_opt::MuOpt = MuOpt(;), cov_fm_opt::CovOpt = CovOpt(;),
                            mu_fm_opt::MuOpt = MuOpt(;),
                            factor_opt::FactorOpt = FactorOpt(;))
    returns = portfolio.returns
    f_returns = portfolio.f_returns

    portfolio.cov_f, portfolio.mu_f = covar_mtx_mean_vec(f_returns; cov_opt = cov_f_opt,
                                                         mu_opt = mu_f_opt)

    portfolio.mu_fm, portfolio.cov_fm, portfolio.returns_fm, portfolio.loadings = risk_factors(DataFrame(f_returns,
                                                                                                         portfolio.f_assets),
                                                                                               DataFrame(returns,
                                                                                                         portfolio.assets);
                                                                                               factor_opt = factor_opt,
                                                                                               cov_opt = cov_fm_opt,
                                                                                               mu_opt = mu_fm_opt)

    return nothing
end

"""
```julia
black_litterman_factor_satistics!(portfolio::AbstractPortfolio;
                                  w::AbstractVector = Vector{Float64}(undef, 0),    # cov_mtx
                                  cov_args::Tuple = (),
                                  cov_est::CovarianceEstimator = StatsBase.SimpleCovariance(;
                                                                                            corrected = true),
                                  cov_func::Function = cov, cov_kwargs::NamedTuple = (;),
                                  cov_method::Symbol = :Full,
                                  cov_weights::Union{AbstractWeights,Nothing} = nothing,
                                  custom_cov::Union{AbstractMatrix,Nothing} = nothing,
                                  gs_threshold::Real = portfolio.gs_threshold,
                                  jlogo::Bool = false, posdef_args::Tuple = (),                                           # Black Litterman
                                  posdef_fix::Symbol = :Nearest,
                                  posdef_func::Function = x -> x,
                                  posdef_kwargs::NamedTuple = (;), std_args::Tuple = (),
                                  std_func::Function = std, std_kwargs::NamedTuple = (;),
                                  target_ret::Union{Real,AbstractVector{<:Real}} = 0.0,    # mean_vec                                           # Settings
                                  custom_mu::Union{AbstractVector,Nothing} = nothing,
                                  mean_args::Tuple = (), mean_func::Function = mean,
                                  mean_kwargs::NamedTuple = (;),
                                  mkt_ret::Union{AbstractVector,Nothing} = nothing,
                                  mu_target::Symbol = :GM, mu_method::Symbol = :Default,
                                  mu_weights::Union{AbstractWeights,Nothing} = nothing,    # Black Litterman
                                  B::Union{DataFrame,Nothing} = nothing,
                                  P::Union{AbstractMatrix,Nothing} = nothing,
                                  P_f::Union{AbstractMatrix,Nothing} = nothing,
                                  Q::Union{AbstractVector,Nothing} = nothing,
                                  Q_f::Union{AbstractVector,Nothing} = nothing,
                                  bl_method::Symbol = :B, delta::Real = 1.0,
                                  diagonal::Bool = true, eq::Bool = true, rf::Real = 0.0,
                                  var_args::Tuple = (), var_func::Function = var,
                                  var_kwargs::NamedTuple = (;),    # Loadings matrix
                                  criterion::Symbol = :pval, pca_kwargs::NamedTuple = (;),
                                  pca_std_kwargs::NamedTuple = (;),
                                  pca_std_type = ZScoreTransform,
                                  reg_method::Symbol = :FReg, threshold::Real = 0.05)
```
"""
function black_litterman_factor_satistics!(portfolio::AbstractPortfolio,
                                           w::AbstractVector                  = Vector{Float64}(undef, 0);
                                           B::Union{DataFrame,Nothing}        = nothing,
                                           P::Union{AbstractMatrix,Nothing}   = nothing,
                                           P_f::Union{AbstractMatrix,Nothing} = nothing,
                                           Q::Union{AbstractVector,Nothing}   = nothing,
                                           Q_f::Union{AbstractVector,Nothing} = nothing,
                                           loadings_opt::LoadingsOpt          = LoadingsOpt(;),
                                           cov_opt::CovOpt                    = CovOpt(;),
                                           mu_opt::MuOpt                      = MuOpt(;),
                                           f_cov_opt::CovOpt                  = CovOpt(;),
                                           f_mu_opt::MuOpt                    = MuOpt(;),
                                           bl_opt::BLOpt                      = BLOpt(;))
    returns = portfolio.returns
    F = portfolio.f_returns

    if isempty(w)
        if isempty(portfolio.bl_bench_weights)
            portfolio.bl_bench_weights = fill(1 / size(portfolio.returns, 2),
                                              size(portfolio.returns, 2))
        end
        w = portfolio.bl_bench_weights
    else
        portfolio.bl_bench_weights = w
    end

    if isnothing(bl_opt.delta)
        bl_opt.delta = (dot(portfolio.mu, w) - bl_opt.rf) / dot(w, portfolio.cov, w)
    end

    if isnothing(B)
        if isempty(portfolio.loadings)
            portfolio.loadings = loadings_matrix(DataFrame(F, portfolio.f_assets),
                                                 DataFrame(returns, portfolio.assets),
                                                 loadings_opt)
        end
        B = portfolio.loadings
    else
        portfolio.loadings = B
    end
    namesB = names(B)
    bl_opt.constant = "const" ∈ namesB
    B = Matrix(B[!, setdiff(namesB, ("ticker",))])

    portfolio.mu_bl_fm, portfolio.cov_bl_fm, missing = if bl_opt.method == :B
        bayesian_black_litterman(returns, F, B, P_f, Q_f; cov_opt = cov_opt,
                                 mu_opt = mu_opt, bl_opt = bl_opt)
    else
        augmented_black_litterman(returns, w; F = F, B = B, P = P, P_f = P_f, Q = Q,
                                  Q_f = Q_f, cov_opt = cov_opt, mu_opt = mu_opt,
                                  f_cov_opt = f_cov_opt, f_mu_opt = f_mu_opt,
                                  bl_opt = bl_opt)
    end

    return nothing
end

function cluster_assets(portfolio::HCPortfolio; linkage = :single,
                        max_k = ceil(Int, sqrt(size(portfolio.dist, 1))),
                        branchorder = :optimal, k = portfolio.k, dbht_method = :Unique)
    @smart_assert(linkage ∈ LinkageTypes)

    clustering, tk = _hierarchical_clustering(portfolio, linkage, max_k, branchorder,
                                              dbht_method)

    k = iszero(k) ? tk : k

    clustering_idx = cutree(clustering; k = k)

    return clustering_idx, clustering, k
end

function cluster_assets!(portfolio::HCPortfolio; linkage = :single,
                         max_k = ceil(Int, sqrt(size(portfolio.dist, 1))),
                         branchorder = :optimal, k = portfolio.k, dbht_method = :Unique)
    @smart_assert(linkage ∈ LinkageTypes)

    clustering, tk = _hierarchical_clustering(portfolio, linkage, max_k, branchorder,
                                              dbht_method)

    k = iszero(k) ? tk : k

    portfolio.clusters = clustering
    portfolio.k = k

    return nothing
end

function _hierarchical_clustering(returns::AbstractMatrix, opt::CorOpt = CorOpt(;),
                                  linkage = :single,
                                  max_k = ceil(Int, sqrt(size(returns, 2))),
                                  branchorder = :optimal, dbht_method = :Unique)
    cor_method = opt.method
    corr, dist = cor_dist_mtx(returns, opt)
    clustering, k = _hcluster_choice(dist, corr, cor_method, linkage, branchorder,
                                     dbht_method, max_k)

    return clustering, k
end

function cluster_assets(returns::AbstractMatrix, opt::CorOpt = CorOpt(;); linkage = :single,
                        max_k = ceil(Int, sqrt(size(returns, 2))), branchorder = :optimal,
                        k = 0, dbht_method = :Unique)
    @smart_assert(linkage ∈ LinkageTypes)

    clustering, tk = _hierarchical_clustering(returns, opt, linkage, max_k, branchorder,
                                              dbht_method)

    k = iszero(k) ? tk : k

    clustering_idx = cutree(clustering; k = k)

    return clustering_idx, clustering, k
end

function cluster_assets(portfolio::Portfolio, opt::CorOpt = CorOpt(;); linkage = :single,
                        max_k = ceil(Int, sqrt(size(portfolio.returns, 2))),
                        branchorder = :optimal, k = 0, dbht_method = :Unique)
    return cluster_assets(portfolio.returns, opt; linkage = linkage, max_k = max_k,
                          branchorder = branchorder, k = k, dbht_method = dbht_method)
end

export covgerber0, covgerber1, covgerber2, mut_var_info_mtx, cov_returns, block_vec_pq,
       duplication_matrix, elimination_matrix, summation_matrix, dup_elim_sum_matrices,
       cokurt, scokurt, asset_statistics!, wc_statistics!, forward_regression,
       backward_regression, pcr, loadings_matrix, risk_factors, black_litterman,
       augmented_black_litterman, bayesian_black_litterman, black_litterman_statistics!,
       factor_statistics!, black_litterman_factor_satistics!, nearest_cov, covar_mtx,
       mean_vec, cokurt_mtx, mu_estimator, cor_dist_mtx, cluster_assets
